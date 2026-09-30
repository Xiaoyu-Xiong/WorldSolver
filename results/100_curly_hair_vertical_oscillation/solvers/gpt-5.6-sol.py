"""Deterministic discrete elastic rod solver for the curly guide strand."""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Inextensible point-mass strand with bending, damping, gravity and drag."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = self.position.shape[0]
        self.initial_position = self.position.copy()
        self.rest_edges = np.diff(self.position, axis=0)
        self.rest_lengths = np.linalg.norm(self.rest_edges, axis=1)
        self.ds = float(np.mean(self.rest_lengths))

        cfg = self.config
        guide = cfg["physical_guide"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.drag_nu = float(guide["ambient_drag_nu"])
        self.internal_gamma = float(guide["internal_damping_gamma"])
        self.material_length = float(guide["material_length"])

        rows = np.arange(self.n - 2)
        self.B = np.zeros((self.n - 2, self.n), dtype=float)
        self.B[rows, rows] = 1.0
        self.B[rows, rows + 1] = -2.0
        self.B[rows, rows + 2] = 1.0
        self.curvature_matrix = self.B.T @ self.B
        self.rest_curvature = self.B @ self.position
        radius = float(guide["radius"])
        second_moment = math.pi * radius**4 / 4.0
        bending_rigidity = 2.5e6 * second_moment
        self.curvature_stiffness = bending_rigidity / self.ds**3
        self.curvature_damping = self.internal_gamma / self.ds**3

        control_length = np.full(self.n, self.ds, dtype=float)
        control_length[[0, -1]] *= 0.5
        self.air_damping = self.drag_nu * control_length
        self.air_damping[0] = 0.0
        self.fixed = np.array([0, 1], dtype=int)
        self.free = np.arange(2, self.n, dtype=int)
        self.inverse_mass = np.zeros(self.n, dtype=float)
        positive = self.mass > 0.0
        self.inverse_mass[positive] = 1.0 / self.mass[positive]
        self.inverse_mass[self.fixed] = 0.0
        self._factor_cache = {}
        self.last_external_force = np.zeros((1, 3, 3), dtype=float)
        self.initial_momentum = np.sum(self.mass[:, None] * self.velocity, axis=0)
        total_mass = float(np.sum(self.mass))
        self.last_external_force[0, 0] = total_mass * self.gravity
        self.reference_momentum = max(total_mass * math.sqrt(9.81 * self.material_length), 1.0e-12)
        self.step_count = 0

    def _root_state(self, t: float):
        drive = self.config["root_motion"]
        rest = np.asarray(drive["rest_position"], dtype=float)
        axis = np.asarray(drive["axis"], dtype=float)
        start = float(drive["start_time"])
        if t < start:
            return rest.copy(), np.zeros(3, dtype=float)
        omega = 2.0 * math.pi * float(drive["frequency"])
        phase = omega * (t - start) + float(drive.get("phase", 0.0))
        amp = float(drive["amplitude"])
        return rest + axis * (amp * math.sin(phase)), axis * (amp * omega * math.cos(phase))

    def _linear_factor(self, h: float):
        key = round(float(h), 14)
        if key in self._factor_cache:
            return self._factor_cache[key]
        damping = np.diag(self.air_damping) + self.curvature_damping * self.curvature_matrix
        system = np.diag(self.mass) + h * damping + h * h * self.curvature_stiffness * self.curvature_matrix
        ff = system[np.ix_(self.free, self.free)]
        fc = system[np.ix_(self.free, self.fixed)]
        result = (cho_factor(ff, lower=True, check_finite=False), fc)
        self._factor_cache[key] = result
        return result

    def _project_lengths(self, x: np.ndarray) -> None:
        for iteration in range(18):
            indices = range(self.n - 1) if iteration % 2 == 0 else range(self.n - 2, -1, -1)
            for i in indices:
                j = i + 1
                delta = x[j] - x[i]
                length = float(np.linalg.norm(delta))
                if length <= 1.0e-14:
                    continue
                wi, wj = self.inverse_mass[i], self.inverse_mass[j]
                denom = wi + wj
                if denom == 0.0:
                    continue
                correction = (length - self.rest_lengths[i]) * delta / (length * denom)
                x[i] += wi * correction
                x[j] -= wj * correction

    def _project_velocities(self, velocity: np.ndarray) -> None:
        for _ in range(3):
            for i in range(self.n - 1):
                j = i + 1
                direction = self.position[j] - self.position[i]
                norm = float(np.linalg.norm(direction))
                if norm <= 1.0e-14:
                    continue
                direction /= norm
                wi, wj = self.inverse_mass[i], self.inverse_mass[j]
                denom = wi + wj
                if denom == 0.0:
                    continue
                impulse = float(np.dot(velocity[j] - velocity[i], direction)) / denom
                velocity[i] += wi * impulse * direction
                velocity[j] -= wj * impulse * direction

    def _step(self, h: float) -> None:
        old_position = self.position.copy()
        old_velocity = self.velocity.copy()
        old_momentum = np.sum(self.mass[:, None] * old_velocity, axis=0)
        end_time = self.time + h
        root_position, root_velocity = self._root_state(end_time)
        fixed_position = np.vstack((root_position, root_position + self.rest_edges[0]))

        damping = np.diag(self.air_damping) + self.curvature_damping * self.curvature_matrix
        rhs = (self.mass[:, None] * (old_position + h * old_velocity + h * h * self.gravity)
               + h * (damping @ old_position)
               + h * h * self.curvature_stiffness * (self.B.T @ self.rest_curvature))
        factor, fixed_coupling = self._linear_factor(h)
        new_position = np.empty_like(old_position)
        new_position[self.fixed] = fixed_position
        new_position[self.free] = cho_solve(
            factor, rhs[self.free] - fixed_coupling @ fixed_position, check_finite=False)
        self._project_lengths(new_position)
        new_position[self.fixed] = fixed_position
        new_velocity = (new_position - old_position) / h
        new_velocity[:2] = root_velocity
        self.position = new_position
        self._project_velocities(new_velocity)
        new_velocity[:2] = root_velocity
        self.velocity = new_velocity
        self.time = end_time
        self.step_count += 1

        gravity_force = np.sum(self.mass) * self.gravity
        drag_force = -np.sum(self.air_damping[:, None] * new_velocity, axis=0)
        new_momentum = np.sum(self.mass[:, None] * new_velocity, axis=0)
        clamp_force = (new_momentum - old_momentum) / h - gravity_force - drag_force
        self.last_external_force[0, 0] = gravity_force
        self.last_external_force[0, 1] = drag_force
        self.last_external_force[0, 2] = clamp_force

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to time must be finite and non-negative")
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(0.0025, target - self.time))
        self.time = target

        momentum = np.sum(self.mass[:, None] * self.velocity, axis=0)
        segment_lengths = np.linalg.norm(np.diff(self.position, axis=0), axis=1)
        node_index = np.arange(self.n, dtype=int)
        frame = StateFrame(
            time=target, position=self.position.copy(), velocity=self.velocity.copy(), mass=self.mass.copy(),
            attributes={
                "node_index": node_index, "is_root": (node_index == 0).astype(int),
                "is_free_tip": (node_index == self.n - 1).astype(int),
                "kinematic_root": (node_index == 0).astype(int), "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": (momentum - self.initial_momentum)[None, :],
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_domain_weights": np.ones((1, self.n), dtype=float),
                "budget_external_force": self.last_external_force.copy(),
                "constraint_target": self.rest_lengths.copy(), "constraint_value": segment_lengths,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "material_length": self.material_length, "internal_damping_gamma": self.internal_gamma,
                "ambient_drag_nu": self.drag_nu, "constraint_scale": self.ds,
                "reference_momentum": self.reference_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "implicit discrete elastic rod with inextensibility projection",
                "substeps": self.step_count, "node_count": self.n, "material_length": self.material_length}
