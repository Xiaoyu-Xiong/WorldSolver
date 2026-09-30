"""Deterministic rigid-sphere solver for the billiard impulse-chain scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Time-step six equal billiard balls on a horizontal felt surface."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.time = float(initial.time)
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.angular_velocity = np.zeros_like(self.velocity)
        config = scene.config
        self.gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.rolling_mu = float(config.get("rolling_friction", 0.015))
        self.radius = 0.2
        self.diameter = 2.0 * self.radius
        self.restitution = 0.92
        self.max_step = 1.0 / 1200.0
        self._pairs = [(i, j) for i in range(len(self.position))
                       for j in range(i + 1, len(self.position))]
        self._pair_impulse = np.zeros((len(self._pairs), 3), dtype=float)
        self._external_impulse = np.zeros_like(self.velocity)
        self._interval_start = self.time
        self._step_count = 0
        self._collision_count = 0

    def _apply_ground(self, h: float) -> None:
        gravity_impulse = self.mass[:, None] * self.gravity[None, :] * h
        self.velocity += gravity_impulse / self.mass[:, None]
        self._external_impulse += gravity_impulse
        self.position += self.velocity * h
        for i in range(len(self.position)):
            if self.position[i, 2] <= self.radius:
                self.position[i, 2] = self.radius
                if self.velocity[i, 2] < 0.0:
                    impulse_z = -self.mass[i] * self.velocity[i, 2]
                    self.velocity[i, 2] = 0.0
                    self._external_impulse[i, 2] += impulse_z
                horizontal = self.velocity[i, :2].copy()
                speed = float(np.linalg.norm(horizontal))
                if speed > 0.0:
                    deceleration = self.rolling_mu * abs(float(self.gravity[2]))
                    new_speed = max(0.0, speed - deceleration * h)
                    new_horizontal = horizontal * (new_speed / speed)
                    self.velocity[i, :2] = new_horizontal
                    self._external_impulse[i, :2] += self.mass[i] * (new_horizontal - horizontal)
                self.angular_velocity[i, 0] = -self.velocity[i, 1] / self.radius
                self.angular_velocity[i, 1] = self.velocity[i, 0] / self.radius

    def _solve_sphere_contacts(self) -> None:
        eps = 1.0e-12
        for _ in range(10):
            changed = False
            for pair_index, (i, j) in enumerate(self._pairs):
                delta = self.position[j] - self.position[i]
                distance = float(np.linalg.norm(delta))
                if distance > self.diameter + 1.0e-9:
                    continue
                normal = delta / distance if distance > eps else np.array((1.0, 0.0, 0.0))
                relative_normal_speed = float(np.dot(self.velocity[j] - self.velocity[i], normal))
                if relative_normal_speed < -1.0e-10:
                    inv_mass_sum = 1.0 / self.mass[i] + 1.0 / self.mass[j]
                    magnitude = -(1.0 + self.restitution) * relative_normal_speed / inv_mass_sum
                    impulse = magnitude * normal
                    self.velocity[i] -= impulse / self.mass[i]
                    self.velocity[j] += impulse / self.mass[j]
                    self._pair_impulse[pair_index] += impulse
                    self._collision_count += 1
                    changed = True
                penetration = self.diameter - distance
                if penetration > 0.0:
                    inv_i, inv_j = 1.0 / self.mass[i], 1.0 / self.mass[j]
                    correction = normal * (penetration + 1.0e-10) / (inv_i + inv_j)
                    self.position[i] -= inv_i * correction
                    self.position[j] += inv_j * correction
                    changed = True
            if not changed:
                break

    def _step(self, h: float) -> None:
        self._apply_ground(h)
        self._solve_sphere_contacts()
        self.time += h
        self._step_count += 1

    def _contact_distances(self) -> np.ndarray:
        sphere_gaps = [np.linalg.norm(self.position[j] - self.position[i]) - self.diameter
                       for i, j in self._pairs]
        return np.concatenate((np.asarray(sphere_gaps, dtype=float), self.position[:, 2] - self.radius))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to requires a finite, nondecreasing time")
        while self.time < target - 1.0e-14:
            self._step(min(self.max_step, target - self.time))
        self.time = target
        interval = target - self._interval_start
        external_force = self._external_impulse / interval if interval > 0.0 else np.zeros_like(self.velocity)
        pair_force = self._pair_impulse / interval if interval > 0.0 else np.zeros_like(self._pair_impulse)
        interaction_force = np.empty((2 * len(self._pairs), 3), dtype=float)
        interaction_force[0::2] = -pair_force
        interaction_force[1::2] = pair_force
        frame = StateFrame(
            time=target, position=self.position.copy(), velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(),
            fields={"contact_signed_distance": self._contact_distances(),
                    "external_force": external_force.copy(), "interaction_force": interaction_force},
            scalars={"contact_length_scale": self.diameter})
        self._external_impulse.fill(0.0)
        self._pair_impulse.fill(0.0)
        self._interval_start = target
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "semi_implicit_euler_sequential_impulses",
                "step_count": self._step_count, "collision_impulse_count": self._collision_count,
                "time": self.time}
