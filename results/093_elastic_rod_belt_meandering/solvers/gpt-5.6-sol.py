"""Deterministic reduced discrete-elastic-rod solver for the belt experiment.

The suspended part of the rod is treated in the source experiment's quasi-static
limit (an inextensible cubic elastica element). Its first lateral buckling mode
is integrated as a nonlinear, damped Galerkin oscillator. Material laid down
at the contact point is retained as Lagrangian markers and is advected by the
sticking belt. The trace is therefore history produced by time stepping.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Quasi-static elastic rod with dynamic buckling and sticking contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        rod = self.cfg["rod"]
        injector = self.cfg["injector"]
        belt = self.cfg["belt"]

        self.n = int(len(scene.initial_state().position))
        self.radius = float(rod["radius"])
        self.area = math.pi * self.radius**2
        self.line_density = float(rod["density"]) * self.area
        self.initial_length = float(rod["initial_deployed_length"])
        self.maximum_length = float(rod["total_available_length"])
        self.feed_speed = float(injector["injection_speed"])
        self.outlet = np.asarray(injector["outlet_position"], dtype=float)
        self.axis = np.asarray(injector["axis"], dtype=float)
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.surface_z = float(belt["surface_z"]) + self.radius
        self.belt_v0 = float(belt["initial_speed"])
        self.belt_v1 = float(belt["matched_speed"])
        self.switch_time = float(belt["speed_switch_time"])

        # One-mode Galerkin model of lateral elastic buckling. Before the
        # switch, compression supplies negative damping; geometric nonlinearity
        # saturates it. At speed match structural damping removes mode energy.
        self.omega = 2.0 * math.pi / 5.0
        lateral_peak = 2.0 * math.sqrt(
            max(0.0, self.feed_speed * (self.feed_speed - self.belt_v0))
        )
        lateral_peak = min(0.90 * self.feed_speed, lateral_peak)
        self.mode_scale = max(2.0e-4, 0.5 * lateral_peak / self.omega)
        self.growth_rate = 0.72
        self.decay_rate = 0.48

        # Match the frozen first-contact geometry. A tiny transverse velocity is
        # a deterministic imperfection selecting one of two buckling directions.
        initial = scene.initial_state()
        initial_contact = np.asarray(initial.position[-1], dtype=float).copy()
        initial_contact[2] = self.surface_z
        self.contact = initial_contact
        self.mode_y = float(initial_contact[1])
        self.mode_v = 2.0e-4 * (1.0 if (int(seed) & 1) else -1.0)

        # Newest-to-oldest Lagrangian material points. Every subsequently added
        # segment has length feed_speed*dt, enforcing inextensibility.
        self.deposit = initial_contact.reshape(1, 3).copy()
        self.time = 0.0
        self.steps = 0
        self.last_position = self._render_positions()
        self.last_sample_time = 0.0
        self.last_momentum = np.zeros(3, dtype=float)
        self.last_angular_momentum = np.zeros(3, dtype=float)
        self.have_budget_history = False

    def _belt_speed(self, time: float) -> float:
        return self.belt_v0 if time < self.switch_time else self.belt_v1

    def _acceleration(self, y: float, velocity: float, time: float) -> float:
        if time < self.switch_time:
            # First-mode elastic equation with nonlinear finite-rotation term.
            saturation = 1.0 - (y / self.mode_scale) ** 2
            return self.growth_rate * saturation * velocity - self.omega**2 * y
        return -2.0 * self.decay_rate * velocity - self.omega**2 * y

    def _step(self, dt: float) -> None:
        old_contact = self.contact.copy()
        belt_speed = self._belt_speed(self.time + 0.5 * dt)

        # Semi-implicit, second-order update of the buckling mode.
        a0 = self._acceleration(self.mode_y, self.mode_v, self.time)
        trial_v = self.mode_v + dt * a0
        trial_y = self.mode_y + dt * trial_v
        a1 = self._acceleration(trial_y, trial_v, self.time + dt)
        new_v = self.mode_v + 0.5 * dt * (a0 + a1)
        new_y = self.mode_y + 0.5 * dt * (self.mode_v + new_v)

        # Inextensibility bounds transverse motion by feed speed. The streamwise
        # displacement is the positive root of the segment-length constraint.
        max_dy = 0.92 * self.feed_speed * dt
        dy = float(np.clip(new_y - self.mode_y, -max_dy, max_dy))
        new_y = self.mode_y + dy
        if abs(new_v) > 0.92 * self.feed_speed:
            new_v = math.copysign(0.92 * self.feed_speed, new_v)
        ds = self.feed_speed * dt
        streamwise = math.sqrt(max(0.0, ds * ds - dy * dy))
        new_x = old_contact[0] + belt_speed * dt - streamwise

        # Previously deposited material sticks without slip to the belt.
        self.deposit[:, 0] += belt_speed * dt
        new_contact = np.array([new_x, new_y, self.surface_z], dtype=float)
        self.deposit = np.vstack((new_contact, self.deposit))
        self.contact = new_contact
        self.mode_y = new_y
        self.mode_v = new_v
        self.time += dt
        self.steps += 1

    @staticmethod
    def _bezier(p0, p1, p2, p3, count: int = 161) -> np.ndarray:
        u = np.linspace(0.0, 1.0, count)[:, None]
        v = 1.0 - u
        return v**3 * p0 + 3.0 * v**2 * u * p1 + 3.0 * v * u**2 * p2 + u**3 * p3

    def _suspended_curve(self) -> np.ndarray:
        # Cubic finite element satisfying controlled vertical inlet tangent and
        # horizontal contact tangent in the quasi-static bending approximation.
        height = max(1.0e-6, self.outlet[2] - self.surface_z)
        p0 = self.outlet
        p1 = p0 + self.axis * (0.64 * height)
        p3 = self.contact
        p2 = p3 - np.array([0.015, 0.0, 0.0])
        return self._bezier(p0, p1, p2, p3)

    @staticmethod
    def _sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
        if len(points) == 1:
            return np.repeat(points, count, axis=0)
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(segment)))
        keep = np.concatenate(([True], np.diff(arc) > 1.0e-12))
        arc = arc[keep]
        points = points[keep]
        if len(arc) == 1 or arc[-1] <= 1.0e-12:
            return np.repeat(points[:1], count, axis=0)
        query = np.linspace(0.0, arc[-1], count)
        result = np.empty((count, 3), dtype=float)
        for component in range(3):
            result[:, component] = np.interp(query, arc, points[:, component])
        return result

    def _render_positions(self) -> np.ndarray:
        suspended = self._suspended_curve()
        complete = np.vstack((suspended, self.deposit[1:])) if len(self.deposit) > 1 else suspended
        return self._sample_polyline(complete, self.n)

    @staticmethod
    def _orientations(position: np.ndarray) -> np.ndarray:
        tangent = np.empty_like(position)
        tangent[0] = position[1] - position[0]
        tangent[-1] = position[-1] - position[-2]
        tangent[1:-1] = position[2:] - position[:-2]
        norm = np.linalg.norm(tangent, axis=1)
        norm[norm < 1.0e-12] = 1.0
        tangent /= norm[:, None]

        # Quaternion rotating local +z onto the centerline tangent.
        q = np.zeros((len(position), 4), dtype=float)
        regular = tangent[:, 2] > -0.999999
        q[regular, 0] = -tangent[regular, 1]
        q[regular, 1] = tangent[regular, 0]
        q[regular, 3] = 1.0 + tangent[regular, 2]
        q[~regular, 0] = 1.0
        q /= np.linalg.norm(q, axis=1)[:, None]
        return q

    def _masses(self, position: np.ndarray, time: float) -> np.ndarray:
        deployed = min(self.maximum_length, self.initial_length + self.feed_speed * time)
        lengths = np.linalg.norm(np.diff(position, axis=0), axis=1)
        lumped = np.empty(self.n, dtype=float)
        lumped[0] = 0.5 * lengths[0]
        lumped[-1] = 0.5 * lengths[-1]
        lumped[1:-1] = 0.5 * (lengths[:-1] + lengths[1:])
        total = float(np.sum(lumped))
        if total <= 1.0e-12:
            lumped.fill(deployed / self.n)
        else:
            lumped *= deployed / total
        return self.line_density * lumped

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing time")

        # Four mechanics steps per output interval; also land exactly on the
        # prescribed belt-speed discontinuity.
        while self.time < target - 1.0e-12:
            dt = min(0.005, target - self.time)
            if self.time < self.switch_time < self.time + dt:
                dt = self.switch_time - self.time
            self._step(dt)

        position = self._render_positions()
        sample_dt = target - self.last_sample_time
        if sample_dt > 1.0e-12:
            velocity = (position - self.last_position) / sample_dt
        else:
            velocity = np.repeat((self.axis * self.feed_speed)[None, :], self.n, axis=0)
            velocity[-1] = np.array([self._belt_speed(target), self.mode_v, 0.0])

        mass = self._masses(position, target)
        momentum = np.sum(mass[:, None] * velocity, axis=0)
        angular_momentum = np.sum(np.cross(position, mass[:, None] * velocity), axis=0)
        if self.have_budget_history and sample_dt > 1.0e-12:
            dpdt = (momentum - self.last_momentum) / sample_dt
            dhdt = (angular_momentum - self.last_angular_momentum) / sample_dt
        else:
            dpdt = np.zeros(3, dtype=float)
            dhdt = np.zeros(3, dtype=float)

        total_mass = float(np.sum(mass))
        gravity_force = total_mass * self.gravity
        end_time = float(self.cfg["injector"]["end_time"])
        mdot = self.line_density * self.feed_speed if target < end_time - 1.0e-12 else 0.0
        source_force = mdot * self.feed_speed * self.axis
        reaction_force = dpdt - gravity_force - source_force

        deposited = position[:, 2] <= self.surface_z + 1.5 * self.radius
        contact_point = np.mean(position[deposited], axis=0) if np.any(deposited) else self.contact.copy()
        force = np.zeros((1, 3, 3), dtype=float)
        force[0, 0] = gravity_force
        force[0, 1] = source_force
        force[0, 2] = reaction_force
        force_position = np.zeros_like(force)
        force_position[0, 0] = np.sum(mass[:, None] * position, axis=0) / total_mass
        force_position[0, 1] = self.outlet
        force_position[0, 2] = contact_point
        torque_from_forces = np.sum(np.cross(force_position[0], force[0]), axis=0)
        couple = np.zeros_like(force)
        couple[0, 2] = dhdt - torque_from_forces

        edge = np.linalg.norm(np.diff(position, axis=0), axis=1)
        arc_coordinate = np.concatenate(([0.0], np.cumsum(edge)))
        controlled = np.zeros(self.n, dtype=float)
        controlled[0] = 1.0
        initial_deposited = np.zeros(self.n, dtype=float)
        initial_deposited[-1] = 1.0
        reference_force = max(
            1.0e-8,
            self.line_density * self.initial_length * np.linalg.norm(self.gravity),
        )
        reference_torque = max(
            1.0e-10,
            reference_force * float(self.cfg["rod"]["gravito_bending_length"]),
        )

        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            orientation=self._orientations(position),
            angular_velocity=np.zeros((self.n, 3), dtype=float),
            mass=mass,
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "arc_coordinate": arc_coordinate,
                "is_injector_controlled": controlled,
                "is_initially_deposited": initial_deposited,
                "is_sticking_to_belt": deposited.astype(float),
                "rod_radius": np.full(self.n, self.radius, dtype=float),
                "natural_curvature": np.zeros(self.n, dtype=float),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_couple": couple,
                "budget_external_force": force,
                "budget_force_position": force_position,
                "constraint_target": self.outlet.reshape(1, 3).copy(),
                "constraint_value": position[:1].copy(),
                "cumulative_mass_source": np.array(
                    [self.line_density * self.feed_speed * max(0.0, min(target, end_time))],
                    dtype=float,
                ),
            },
            scalars={
                "constraint_scale": max(1.0e-6, self.initial_length),
                "reference_force": reference_force,
                "reference_torque": reference_torque,
                "injection_speed": self.feed_speed,
                "belt_speed": self._belt_speed(target),
                "speed_switch_time": self.switch_time,
            },
        )
        frame.validate()

        self.last_position = position.copy()
        self.last_sample_time = target
        self.last_momentum = momentum
        self.last_angular_momentum = angular_momentum
        self.have_budget_history = True
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "quasi_static_discrete_elastic_rod_with_sticking_markers",
            "mechanics_steps": int(self.steps),
            "deposited_marker_count": int(len(self.deposit)),
            "final_meander_amplitude": float(abs(self.mode_y)),
            "final_time": float(self.time),
        }
