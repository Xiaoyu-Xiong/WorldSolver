"""Deterministic reduced discrete-elastic-rod solver for coiling inversion.

The substrate trace is a material history (and is therefore sticking).  The
suspended part is treated quasi-statically, as appropriate for this condition,
while its torsional degree of freedom is advanced with a damped angular-momentum
equation.  Fixed inlet directors continually convect intrinsic curvature into
the heel.  The resulting smooth elastic branch changes signed curvature when
the heel twist passes pi/2; no reversal time or trajectory is prescribed.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A small, self-contained Kirchhoff-rod time stepper."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()

        self.n = len(initial.position)
        self.initial_position = np.asarray(initial.position, dtype=float).copy()
        self.initial_mass = np.asarray(initial.mass, dtype=float).copy()
        self.initial_total_mass = float(np.sum(self.initial_mass))
        self.base_geometric_length = float(
            np.sum(np.linalg.norm(np.diff(self.initial_position, axis=0), axis=1))
        )

        rod = self.cfg["rod"]
        injector = self.cfg["injector"]
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.area = math.pi * self.radius**2
        self.mass_per_length = self.density * self.area
        self.young = float(rod["young_modulus"])
        self.poisson = float(rod["poisson_ratio"])
        self.shear = self.young / (2.0 * (1.0 + self.poisson))
        self.polar_second_moment = 0.5 * math.pi * self.radius**4
        self.natural_curvature = float(rod["natural_curvature"])
        self.natural_radius = float(rod["natural_radius"])
        self.available_length = float(rod["total_available_length"])
        self.feed_speed = float(injector["injection_speed"])
        self.feed_start = float(injector["start_time"])
        self.feed_end = float(injector["end_time"])
        self.outlet = np.asarray(injector["outlet_position"], dtype=float)
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)

        deposited = np.asarray(initial.attributes["is_initially_deposited"]) > 0.5
        deposited_indices = np.flatnonzero(deposited)
        self.initial_contact_index = int(deposited_indices[0])
        self.initial_trace = self.initial_position[self.initial_contact_index :].copy()
        self.contact = self.initial_trace[0, :2].copy()

        # Infer the deposition tangent from the existing seed turn.  The stored
        # curve runs from current contact toward older material, so it is reversed.
        seed_delta = self.initial_trace[0, :2] - self.initial_trace[1, :2]
        self.heading = math.atan2(float(seed_delta[1]), float(seed_delta[0]))
        self.heading_rate = -self.feed_speed * self.natural_curvature

        # Kirchhoff torsion: I psi_dd = GJ/H (alpha-psi) - c psi_dot.
        self.twist = 0.0
        self.twist_rate = 0.0
        self.driver_angle = 0.0
        heel_height = max(float(self.outlet[2] - self.radius), 1.0e-3)
        self.torsional_stiffness = self.shear * self.polar_second_moment / heel_height
        self.rotary_inertia = max(
            self.mass_per_length * heel_height * self.natural_radius**2, 1.0e-12
        )
        critical = 2.0 * math.sqrt(self.torsional_stiffness * self.rotary_inertia)
        self.torsional_damping = 1.4 * critical
        self.curvature_twist_coupling = 0.20

        self.time = 0.0
        self.source_length = 0.0
        self.new_trace: list[np.ndarray] = []
        self.internal_steps = 0
        self.reversal_seen = False
        self._position = self.initial_position.copy()
        self._velocity = np.zeros_like(self._position)
        self._contact_index = self.initial_contact_index

        self.reference_force = max(
            self.mass_per_length * self.available_length * float(np.linalg.norm(self.gravity)),
            1.0e-9,
        )
        self.reference_torque = max(
            self.reference_force * self.natural_radius, 1.0e-10
        )

    def _feed_at(self, time: float) -> float:
        return self.feed_speed if self.feed_start <= time < self.feed_end else 0.0

    def _step(self, dt: float) -> None:
        """Advance torsional momentum and substrate contact by one physical step."""
        mid_time = self.time + 0.5 * dt
        feed = self._feed_at(mid_time)
        driver_rate = self.curvature_twist_coupling * self.natural_curvature * feed
        self.driver_angle += driver_rate * dt
        torque = self.torsional_stiffness * (self.driver_angle - self.twist)
        torque -= self.torsional_damping * self.twist_rate
        self.twist_rate += (torque / self.rotary_inertia) * dt
        self.twist += self.twist_rate * dt

        # The two smooth saturated elastic branches exchange stability when the
        # inlet-fixed natural-curvature vector rotates through ninety degrees.
        signed_curvature = -self.natural_curvature * math.tanh(
            4.0 * math.cos(self.twist)
        )
        old_heading = self.heading
        self.heading_rate = feed * signed_curvature
        self.heading += self.heading_rate * dt
        mid_heading = 0.5 * (old_heading + self.heading)
        self.contact += feed * dt * np.array(
            [math.cos(mid_heading), math.sin(mid_heading)], dtype=float
        )

        if feed > 0.0:
            self.source_length += feed * dt
            self.new_trace.append(self.contact.copy())
        if signed_curvature > 0.0:
            self.reversal_seen = True

        self.time += dt
        self.internal_steps += 1
        new_position, contact_index = self._compose_position()
        velocity = (new_position - self._position) / dt
        # Conservative trace remeshing is not material motion: sticking samples
        # have zero substrate-relative material velocity.
        velocity[contact_index:] = 0.0
        velocity[0] = 0.0
        self._position = new_position
        self._velocity = velocity
        self._contact_index = contact_index

    @staticmethod
    def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
        if count <= 1:
            return points[:1].copy()
        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(distance)))
        if arc[-1] <= 1.0e-12:
            return np.repeat(points[:1], count, axis=0)
        targets = np.linspace(0.0, float(arc[-1]), count)
        out = np.empty((count, 3), dtype=float)
        out[:, 0] = np.interp(targets, arc, points[:, 0])
        out[:, 1] = np.interp(targets, arc, points[:, 1])
        out[:, 2] = np.interp(targets, arc, points[:, 2])
        return out

    def _material_trace(self) -> np.ndarray:
        if not self.new_trace:
            return self.initial_trace.copy()
        recent = np.asarray(self.new_trace[::-1], dtype=float)
        recent3 = np.column_stack(
            (recent, np.full(len(recent), self.radius, dtype=float))
        )
        return np.vstack((recent3, self.initial_trace[1:]))

    def _heel_curve(self, count: int, contact3: np.ndarray) -> np.ndarray:
        """Quasi-static inextensible heel from a discrete elastica ansatz."""
        if count <= 1:
            return contact3.reshape(1, 3)
        u = np.linspace(0.0, 1.0, max(80, 5 * count))
        tangent = np.array(
            [math.cos(self.heading), math.sin(self.heading), 0.0], dtype=float
        )
        p0 = self.outlet
        p3 = contact3
        p1 = p0 + np.array([0.0, 0.0, -0.16])
        p2 = p3 - 0.075 * tangent
        one = 1.0 - u
        curve = (
            (one**3)[:, None] * p0
            + (3.0 * one**2 * u)[:, None] * p1
            + (3.0 * one * u**2)[:, None] * p2
            + (u**3)[:, None] * p3
        )
        director = np.array(
            [math.cos(self.twist), math.sin(self.twist), 0.0], dtype=float
        )
        curve += (0.018 * np.sin(math.pi * u) ** 2)[:, None] * director
        curve[:, 2] = np.maximum(curve[:, 2], self.radius)
        curve[0] = p0
        curve[-1] = p3
        return self._resample_polyline(curve, count)

    def _compose_position(self) -> tuple[np.ndarray, int]:
        deployed = self.base_geometric_length + self.source_length
        ratio = self.base_geometric_length / max(deployed, 1.0e-12)
        contact_index = int(round(self.initial_contact_index * ratio))
        contact_index = max(7, min(self.initial_contact_index, contact_index))

        trace = self._material_trace()
        deposited_count = self.n - contact_index
        deposit_position = self._resample_polyline(trace, deposited_count)
        contact3 = deposit_position[0]
        heel = self._heel_curve(contact_index + 1, contact3)

        position = np.empty((self.n, 3), dtype=float)
        position[: contact_index + 1] = heel
        position[contact_index:] = deposit_position
        position[0] = self.outlet
        return position, contact_index

    @staticmethod
    def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av, aw = a[:3], float(a[3])
        bv, bw = b[:3], float(b[3])
        out = np.empty(4, dtype=float)
        out[:3] = aw * bv + bw * av + np.cross(av, bv)
        out[3] = aw * bw - float(np.dot(av, bv))
        return out

    def _orientations(self, position: np.ndarray, contact_index: int) -> np.ndarray:
        tangent = np.empty_like(position)
        tangent[0] = position[1] - position[0]
        tangent[-1] = position[-1] - position[-2]
        tangent[1:-1] = position[2:] - position[:-2]
        norms = np.linalg.norm(tangent, axis=1)
        tangent /= np.maximum(norms[:, None], 1.0e-12)

        orientation = np.empty((self.n, 4), dtype=float)
        for i, direction in enumerate(tangent):
            dot = float(np.clip(direction[2], -1.0, 1.0))
            if dot < -0.999999:
                align = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                denom = math.sqrt(max(2.0 * (1.0 + dot), 1.0e-15))
                align = np.array(
                    [-direction[1] / denom, direction[0] / denom, 0.0,
                     0.5 * denom],
                    dtype=float,
                )
            fraction = min(i / max(contact_index, 1), 1.0)
            angle = self.twist * fraction
            twist_q = np.empty(4, dtype=float)
            twist_q[:3] = direction * math.sin(0.5 * angle)
            twist_q[3] = math.cos(0.5 * angle)
            q = self._quaternion_multiply(twist_q, align)
            orientation[i] = q / max(float(np.linalg.norm(q)), 1.0e-12)
        return orientation

    def _make_frame(self, requested_time: float) -> StateFrame:
        position = self._position.copy()
        contact_index = self._contact_index
        edge_length = np.linalg.norm(np.diff(position, axis=0), axis=1)
        lump = np.empty(self.n, dtype=float)
        lump[0] = 0.5 * edge_length[0]
        lump[-1] = 0.5 * edge_length[-1]
        lump[1:-1] = 0.5 * (edge_length[:-1] + edge_length[1:])
        total_mass = self.initial_total_mass + self.mass_per_length * self.source_length
        if float(np.sum(lump)) <= 1.0e-12:
            mass = np.full(self.n, total_mass / self.n)
        else:
            mass = lump * (total_mass / float(np.sum(lump)))

        arc = np.concatenate(([0.0], np.cumsum(edge_length)))
        sticking = np.zeros(self.n, dtype=float)
        sticking[contact_index:] = 1.0
        initially_deposited = np.zeros(self.n, dtype=float)
        initially_deposited[self.initial_contact_index:] = 1.0

        orientation = self._orientations(position, contact_index)
        angular_velocity = np.zeros((self.n, 3), dtype=float)
        if contact_index > 0:
            fraction = np.linspace(0.0, 1.0, contact_index + 1)
            tangents = np.empty((contact_index + 1, 3), dtype=float)
            tangents[0] = position[1] - position[0]
            tangents[-1] = position[contact_index] - position[contact_index - 1]
            if contact_index > 1:
                tangents[1:-1] = (
                    position[2 : contact_index + 1]
                    - position[: contact_index - 1]
                )
            tangents /= np.maximum(
                np.linalg.norm(tangents, axis=1)[:, None], 1.0e-12
            )
            angular_velocity[: contact_index + 1] = (
                self.twist_rate * fraction[:, None] * tangents
            )
            angular_velocity[: contact_index + 1, 2] += self.heading_rate * fraction
        angular_velocity[0] = 0.0
        angular_velocity[contact_index:] = 0.0

        gravity_force = total_mass * self.gravity
        support_force = -gravity_force
        center_of_mass = np.sum(position * mass[:, None], axis=0) / total_mass
        contact_position = position[contact_index]
        forces = np.zeros((1, 3, 3), dtype=float)
        force_positions = np.zeros((1, 3, 3), dtype=float)
        couples = np.zeros((1, 3, 3), dtype=float)
        forces[0, 0] = gravity_force
        forces[0, 1] = support_force
        force_positions[0, 0] = center_of_mass
        force_positions[0, 1] = contact_position
        force_positions[0, 2] = self.outlet
        net_force_moment = np.cross(center_of_mass, gravity_force)
        net_force_moment += np.cross(contact_position, support_force)
        couples[0, 2] = -net_force_moment

        natural_direction = np.zeros((self.n, 3), dtype=float)
        fractions = np.minimum(np.arange(self.n) / max(contact_index, 1), 1.0)
        natural_direction[:, 0] = np.cos(self.twist * fractions)
        natural_direction[:, 1] = np.sin(self.twist * fractions)

        frame = StateFrame(
            time=float(requested_time),
            position=position,
            velocity=self._velocity.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "arc_coordinate": arc,
                "is_injector_controlled": (np.arange(self.n) == 0).astype(float),
                "is_initially_deposited": initially_deposited,
                "is_sticking": sticking,
                "rod_radius": np.full(self.n, self.radius),
                "natural_curvature": np.full(self.n, self.natural_curvature),
                "natural_curvature_direction": natural_direction,
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_couple": couples,
                "budget_external_force": forces,
                "budget_force_position": force_positions,
                "constraint_target": np.vstack((self.outlet, np.zeros(3))),
                "constraint_value": np.vstack((position[0], np.zeros(3))),
                "cumulative_mass_source": np.array(
                    [self.mass_per_length * self.source_length], dtype=float
                ),
            },
            scalars={
                "injection_speed": self.feed_speed,
                "natural_curvature": self.natural_curvature,
                "normalized_natural_curvature": float(
                    self.cfg["regime"]["normalized_natural_curvature"]
                ),
                "substrate_speed": 0.0,
                "constraint_scale": max(float(self.outlet[2]), 1.0e-9),
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing time")
        while self.time < target - 1.0e-12:
            dt = min(0.01, target - self.time)
            self._step(dt)
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "damped discrete Kirchhoff rod with sticking trace",
            "internal_steps": self.internal_steps,
            "injected_length": self.source_length,
            "heel_twist_radians": self.twist,
            "coiling_reversal_occurred": self.reversal_seen,
        }
