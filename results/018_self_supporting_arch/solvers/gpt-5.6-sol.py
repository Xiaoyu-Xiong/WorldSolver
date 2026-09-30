"""Damped rigid-block contact solver for the self-supporting arch."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Advance the arch using a deterministic semi-implicit contact model."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._scene = scene
        self._time = 0.0
        self._position = np.asarray(initial.position, dtype=float).copy()
        self._velocity = np.asarray(initial.velocity, dtype=float).copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=int).copy()
        self._initial_position = self._position.copy()
        self._gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._mu = float(scene.config.get("friction", {}).get("coefficient", 0.6))
        self._n = self._position.shape[0]
        if self._n < 2:
            raise ValueError("The arch requires at least two rigid blocks")
        self._rest_distance = np.linalg.norm(
            self._initial_position[1:] - self._initial_position[:-1], axis=1
        )
        self._support_z = self._initial_position[[0, -1], 2].copy()
        self._contact_stiffness = 2.5e4
        self._contact_damping = 2.0e3
        self._ground_stiffness = 4.0e4
        self._ground_damping = 2.5e3
        self._viscous_damping = 0.20
        self._max_step = min(float(scene.config.get("sample_dt", 0.02)) / 4.0, 0.005)
        self._last_constraint_error = 0.0
        self._initialized = True

    def _forces(self) -> np.ndarray:
        p, v = self._position, self._velocity
        force = self._mass[:, None] * self._gravity[None, :]
        for i in range(self._n - 1):
            delta = p[i + 1] - p[i]
            dist = float(np.linalg.norm(delta))
            if dist < 1.0e-12:
                continue
            normal = delta / dist
            rel = v[i + 1] - v[i]
            rel_n = float(np.dot(rel, normal))
            extension = dist - self._rest_distance[i]
            fn = self._contact_stiffness * extension + self._contact_damping * rel_n
            # Equal and opposite restoring force along the centroid line.
            # Positive extension pulls the two bodies together; negative
            # extension pushes them apart.
            pair_force = fn * normal
            force[i] += pair_force
            force[i + 1] -= pair_force
            tangent = rel - rel_n * normal
            tang_norm = float(np.linalg.norm(tangent))
            if tang_norm > 1.0e-12:
                normal_load = max(0.0, -fn)
                ft = min(self._mu * normal_load, 150.0) + 2.0 * self._viscous_damping
                friction = -ft * tangent / tang_norm
                force[i] += friction
                force[i + 1] -= friction
        for body, z_ref in ((0, self._support_z[0]), (self._n - 1, self._support_z[1])):
            gap = p[body, 2] - z_ref
            if gap < 0.0:
                normal_speed = v[body, 2]
                normal_force = -self._ground_stiffness * gap - self._ground_damping * normal_speed
                force[body, 2] += max(0.0, normal_force)
                tangential = v[body].copy()
                tangential[2] = 0.0
                tnorm = float(np.linalg.norm(tangential))
                if tnorm > 1.0e-12:
                    force[body] -= min(self._mu * max(0.0, normal_force), 200.0) * tangential / tnorm
        force -= self._viscous_damping * v
        return force

    def _step(self, dt: float) -> None:
        acceleration = self._forces() / self._mass[:, None]
        self._velocity += dt * acceleration
        self._position += dt * self._velocity
        for _ in range(3):
            for i in range(self._n - 1):
                delta = self._position[i + 1] - self._position[i]
                dist = float(np.linalg.norm(delta))
                if dist < 1.0e-12:
                    continue
                error = dist - self._rest_distance[i]
                if abs(error) < 1.0e-7:
                    continue
                normal = delta / dist
                wi, wj = 1.0 / self._mass[i], 1.0 / self._mass[i + 1]
                correction = error / (wi + wj)
                self._position[i] += wi * correction * normal
                self._position[i + 1] -= wj * correction * normal
            self._position[0, 2] = max(self._position[0, 2], self._support_z[0])
            self._position[-1, 2] = max(self._position[-1, 2], self._support_z[1])
        for i in range(self._n - 1):
            delta = self._position[i + 1] - self._position[i]
            dist = float(np.linalg.norm(delta))
            if dist < 1.0e-12:
                continue
            normal = delta / dist
            rel_n = float(np.dot(self._velocity[i + 1] - self._velocity[i], normal))
            if rel_n > 0.0:
                wi, wj = 1.0 / self._mass[i], 1.0 / self._mass[i + 1]
                impulse = rel_n / (wi + wj)
                self._velocity[i] += wi * impulse * normal
                self._velocity[i + 1] -= wj * impulse * normal
        if self._position[0, 2] <= self._support_z[0] + 1.0e-9:
            self._velocity[0, 2] = max(0.0, self._velocity[0, 2])
        if self._position[-1, 2] <= self._support_z[1] + 1.0e-9:
            self._velocity[-1, 2] = max(0.0, self._velocity[-1, 2])

    def _frame(self, time: float) -> StateFrame:
        pair_errors = np.linalg.norm(self._position[1:] - self._position[:-1], axis=1) - self._rest_distance
        ground_errors = self._position[[0, -1], 2] - self._support_z
        constraint_value = np.concatenate((pair_errors, ground_errors)).astype(float)
        self._last_constraint_error = float(np.max(np.abs(constraint_value)))
        frame = StateFrame(
            time=float(time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self._n, 1)),
            angular_velocity=np.zeros((self._n, 3), dtype=float),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields={
                "constraint_target": np.zeros_like(constraint_value),
                "constraint_value": constraint_value,
                "contact_signed_distance": constraint_value.copy(),
                "external_force": self._mass[:, None] * self._gravity[None, :],
            },
            scalars={
                "constraint_scale": float(max(np.mean(self._rest_distance), 1.0e-6)),
                "contact_length_scale": 0.5,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to time must be finite and non-negative")
        if target < self._time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self._time < target - 1.0e-12:
            dt = min(self._max_step, target - self._time)
            self._step(dt)
            self._time += dt
        self._time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_contact",
            "body_count": int(self._n) if self._initialized else 0,
            "constraint_error": float(self._last_constraint_error),
            "friction_coefficient": float(self._mu) if self._initialized else 0.6,
        }
