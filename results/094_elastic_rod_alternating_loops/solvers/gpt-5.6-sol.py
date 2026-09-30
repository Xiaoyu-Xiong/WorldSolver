"""Deterministic discrete-elastic-rod solver for the belt-coiling scene.

The centreline is a chain of lumped masses. Gravity and Euler--Bernoulli
bending are integrated explicitly, while the stiff axial response of the
inextensible rod is treated by position projections. Material is introduced
continuously by growing and periodically splitting the inlet interval. Contact
with the belt is a unilateral, perfectly sticking constraint.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic position-based discrete rod simulation."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()

        self.p = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.rest = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.initial_marker = np.asarray(
            initial.attributes["is_initially_deposited"], dtype=float
        ).copy()
        self.stuck = self.initial_marker > 0.5

        rod = self.cfg["rod"]
        injector = self.cfg["injector"]
        belt = self.cfg["belt"]
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.area = math.pi * self.radius**2
        self.line_density = self.density * self.area
        self.young = float(rod["young_modulus"])
        self.bending_stiffness = self.young * math.pi * self.radius**4 / 4.0
        self.nominal_ds = float(np.median(self.rest[:20]))
        self.outlet = np.asarray(injector["outlet_position"], dtype=float)
        self.axis = np.asarray(injector["axis"], dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.injection_speed = float(injector["injection_speed"])
        self.injection_start = float(injector["start_time"])
        self.injection_end = float(injector["end_time"])
        self.total_available = float(rod["total_available_length"])
        self.initial_length = float(np.sum(self.rest))

        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.belt_direction = np.asarray(belt["travel_direction"], dtype=float)
        self.belt_direction /= np.linalg.norm(self.belt_direction)
        self.belt_speed = float(belt["speed"])
        self.belt_velocity = self.belt_speed * self.belt_direction
        self.belt_center = np.asarray(belt["center"], dtype=float)
        self.belt_half = np.asarray(belt["half_extents"], dtype=float)
        self.surface_z = float(belt["surface_z"])

        self.time = 0.0
        self.step_count = 0
        self.insertions = 0
        self.last_force = np.zeros((1, 3, 3), dtype=float)
        self.last_position = np.zeros((1, 3, 3), dtype=float)
        self.last_couple = np.zeros((1, 3, 3), dtype=float)
        self._set_initial_budget()

    @staticmethod
    def _lumped_mass(rest: np.ndarray, line_density: float) -> np.ndarray:
        mass = np.empty(len(rest) + 1, dtype=float)
        mass[0] = 0.5 * rest[0]
        mass[-1] = 0.5 * rest[-1]
        if len(mass) > 2:
            mass[1:-1] = 0.5 * (rest[:-1] + rest[1:])
        return line_density * mass

    def _masses(self) -> np.ndarray:
        return self._lumped_mass(self.rest, self.line_density)

    def _set_initial_budget(self) -> None:
        mass = self._masses()
        total_gravity = np.sum(mass) * self.gravity
        com = np.sum(mass[:, None] * self.p, axis=0) / np.sum(mass)
        support_point = self.outlet.copy()
        if np.any(self.stuck):
            support_point = 0.5 * (support_point + np.mean(self.p[self.stuck], axis=0))
        self.last_force[0, 0] = total_gravity
        self.last_force[0, 1] = -total_gravity
        self.last_position[0, 0] = com
        self.last_position[0, 1] = support_point
        self.last_position[0, 2] = self.outlet
        torque = np.cross(com, total_gravity) + np.cross(support_point, -total_gravity)
        self.last_couple[0, 1] = -torque

    def _feed_active_duration(self, a: float, b: float) -> float:
        return max(0.0, min(b, self.injection_end) - max(a, self.injection_start))

    def _insert_material_node(self) -> None:
        """Split the growing inlet edge without changing the chain geometry."""
        old_first = float(self.rest[0])
        new_edge = self.nominal_ds
        new_position = self.outlet + self.axis * new_edge
        self.p = np.insert(self.p, 1, new_position, axis=0)
        self.v = np.insert(self.v, 1, self.axis * self.injection_speed, axis=0)
        self.stuck = np.insert(self.stuck, 1, False)
        self.initial_marker = np.insert(self.initial_marker, 1, 0.0)
        self.rest = np.concatenate(([new_edge, old_first - new_edge], self.rest[1:]))
        self.insertions += 1

    def _bending_forces(self) -> np.ndarray:
        """Gradient of a naturally straight discrete bending energy."""
        n = len(self.p)
        force = np.zeros_like(self.p)
        for i in range(1, n - 1):
            local_ds = 0.5 * (self.rest[i - 1] + self.rest[i])
            local_ds = max(local_ds, 0.5 * self.nominal_ds)
            second_difference = self.p[i - 1] - 2.0 * self.p[i] + self.p[i + 1]
            bend = self.bending_stiffness * second_difference / local_ds**3
            force[i - 1] -= bend
            force[i] += 2.0 * bend
            force[i + 1] -= bend
        return force

    def _project_lengths(self, inverse_mass: np.ndarray, iterations: int = 7) -> None:
        """Gauss--Seidel projection of inextensible edge constraints."""
        for iteration in range(iterations):
            indices = range(len(self.rest))
            if (self.step_count + iteration) & 1:
                indices = range(len(self.rest) - 1, -1, -1)
            for i in indices:
                delta = self.p[i + 1] - self.p[i]
                distance = float(np.linalg.norm(delta))
                if distance < 1.0e-12:
                    continue
                wi = inverse_mass[i]
                wj = inverse_mass[i + 1]
                weight = wi + wj
                if weight <= 0.0:
                    continue
                correction = (distance - self.rest[i]) * delta / (distance * weight)
                self.p[i] += wi * correction
                self.p[i + 1] -= wj * correction
            self.p[0] = self.outlet
            self.p[1] = self.outlet + self.axis * self.rest[0]

    def _inside_belt(self, p: np.ndarray) -> np.ndarray:
        return (
            (np.abs(p[:, 0] - self.belt_center[0]) <= self.belt_half[0] + self.radius)
            & (np.abs(p[:, 1] - self.belt_center[1]) <= self.belt_half[1] + self.radius)
        )

    def _step(self, dt: float) -> None:
        mass_before = self._masses()
        momentum_before = np.sum(mass_before[:, None] * self.v, axis=0)
        angular_before = np.sum(np.cross(self.p, mass_before[:, None] * self.v), axis=0)

        added_length = 0.0
        active_dt = self._feed_active_duration(self.time, self.time + dt)
        if active_dt > 0.0 and np.sum(self.rest) < self.total_available:
            allowed = max(0.0, self.total_available - float(np.sum(self.rest)))
            added_length = min(self.injection_speed * active_dt, allowed)
            self.rest[0] += added_length
            while self.rest[0] >= 2.0 * self.nominal_ds - 1.0e-12:
                self._insert_material_node()

        old_position = self.p.copy()
        mass = self._masses()
        inverse_mass = 1.0 / mass
        fixed = self.stuck.copy()
        fixed[0] = True
        fixed[1] = True
        inverse_mass[fixed] = 0.0

        self.p[self.stuck] += self.belt_velocity * dt
        free = ~fixed
        bend_force = self._bending_forces()
        self.v[free] += dt * (self.gravity + bend_force[free] / mass[free, None])
        self.v[free] *= math.exp(-3.0 * dt)
        self.p[free] += dt * self.v[free]

        self.p[0] = self.outlet
        self.p[1] = self.outlet + self.axis * self.rest[0]
        self._project_lengths(inverse_mass)

        contact_height = self.surface_z + self.radius
        new_contact = (~self.stuck) & (self.p[:, 2] <= contact_height)
        new_contact &= self._inside_belt(self.p)
        new_contact[:2] = False
        if np.any(new_contact):
            self.p[new_contact, 2] = contact_height
            self.stuck[new_contact] = True

        self.v = (self.p - old_position) / dt
        self.v[0] = 0.0
        self.v[1] = self.axis * (added_length / dt if active_dt > 0.0 else 0.0)
        self.v[self.stuck] = self.belt_velocity

        momentum_after = np.sum(mass[:, None] * self.v, axis=0)
        angular_after = np.sum(np.cross(self.p, mass[:, None] * self.v), axis=0)
        added_mass = float(np.sum(mass) - np.sum(mass_before))
        inflow_momentum = added_mass * self.axis * self.injection_speed
        inflow_angular = np.cross(self.outlet, inflow_momentum)
        net_external = (momentum_after - momentum_before - inflow_momentum) / dt
        net_torque = (angular_after - angular_before - inflow_angular) / dt
        gravity_force = np.sum(mass) * self.gravity
        com = np.sum(mass[:, None] * self.p, axis=0) / np.sum(mass)
        support_force = net_external - gravity_force
        support_position = self.outlet.copy()
        if np.any(self.stuck):
            support_position = 0.5 * (self.outlet + np.mean(self.p[self.stuck], axis=0))
        self.last_force.fill(0.0)
        self.last_position.fill(0.0)
        self.last_couple.fill(0.0)
        self.last_force[0, 0] = gravity_force
        self.last_force[0, 1] = support_force
        self.last_position[0, 0] = com
        self.last_position[0, 1] = support_position
        self.last_position[0, 2] = self.outlet
        represented_torque = np.cross(com, gravity_force) + np.cross(
            support_position, support_force
        )
        self.last_couple[0, 1] = net_torque - represented_torque

        if not np.isfinite(self.p).all() or not np.isfinite(self.v).all():
            raise FloatingPointError("non-finite discrete rod state")
        self.time += dt
        self.step_count += 1

    def _frame(self, requested_time: float) -> StateFrame:
        mass = self._masses()
        n = len(self.p)
        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0
        arc = np.concatenate(([0.0], np.cumsum(self.rest)))
        injector_control = np.zeros(n, dtype=float)
        injector_control[0] = 1.0
        source_duration = self._feed_active_duration(0.0, requested_time)
        source_mass = self.line_density * self.injection_speed * source_duration
        total_weight = max(float(np.sum(mass) * np.linalg.norm(self.gravity)), 1.0e-9)

        frame = StateFrame(
            time=float(requested_time),
            position=self.p.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=np.zeros((n, 3), dtype=float),
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": arc,
                "is_injector_controlled": injector_control,
                "is_initially_deposited": self.initial_marker.copy(),
                "is_sticking_to_belt": self.stuck.astype(float),
                "rod_radius": np.full(n, self.radius, dtype=float),
                "natural_curvature": np.zeros(n, dtype=float),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_couple": self.last_couple.copy(),
                "budget_external_force": self.last_force.copy(),
                "budget_force_position": self.last_position.copy(),
                "constraint_value": np.asarray(
                    [self.belt_speed / self.injection_speed], dtype=float
                ),
                "constraint_target": np.asarray(
                    [float(self.cfg["regime"]["speed_ratio_belt_to_injection"])], dtype=float
                ),
                "cumulative_mass_source": np.asarray([source_mass], dtype=float),
            },
            scalars={
                "injection_speed": self.injection_speed,
                "belt_speed": self.belt_speed,
                "belt_to_injection_speed_ratio": self.belt_speed / self.injection_speed,
                "constraint_scale": float(
                    self.cfg["regime"]["speed_ratio_belt_to_injection"]
                ),
                "reference_force": total_weight,
                "reference_torque": total_weight * max(self.initial_length, 0.01),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and monotone")
        maximum_step = 0.002
        while self.time < target - 1.0e-12:
            self._step(min(maximum_step, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete_elastic_rod_pbd",
            "substeps": int(self.step_count),
            "inserted_material_nodes": int(self.insertions),
            "final_node_count": int(len(self.p)),
            "final_deployed_length": float(np.sum(self.rest)),
            "sticking_node_count": int(np.count_nonzero(self.stuck)),
        }
