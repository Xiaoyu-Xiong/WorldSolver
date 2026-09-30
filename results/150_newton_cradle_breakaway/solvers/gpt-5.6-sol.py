"""Deterministic, self-contained Newton-cradle rigid-body solver."""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Spherical pendula coupled by unilateral rigid contact impulses."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        config = scene.config
        initial = scene.initial_state()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.orientation = np.asarray(initial.orientation, dtype=float).copy()
        nominal = np.asarray(initial.position, dtype=float)
        self.n = len(self.mass)

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.radius = 0.23
        self.length = 1.0
        self.anchor_x = nominal[:, 0].copy()
        self.anchor_y = nominal[:, 1].copy()
        self.anchor_z = nominal[:, 2] + self.length

        self.theta = np.zeros(self.n, dtype=float)
        self.omega = np.zeros(self.n, dtype=float)
        # The task calls for a released displaced end ball, while the listed
        # transforms are the hanging/rest transforms. Apply that initial
        # condition on the same rigid support.
        self.theta[0] = -0.72

        contact = config["contact"]
        self.contact_length_scale = float(contact["length_scale"])
        # Treat the small supplied coefficient as fractional impact loss. This
        # retains the nearly elastic propagation characteristic of a cradle.
        loss = float(contact["restitution"])
        self.impact_restitution = float(np.clip(1.0 - loss, 0.0, 1.0))
        self.angular_drag = 0.003
        self.internal_dt = 5.0e-4
        self.contact_skin = 2.0e-4
        self.time = 0.0
        self.step_count = 0

    def _kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sine = np.sin(self.theta)
        cosine = np.cos(self.theta)
        tangent = np.column_stack((cosine, np.zeros(self.n), sine))
        position = np.column_stack(
            (
                self.anchor_x + self.length * sine,
                self.anchor_y,
                self.anchor_z - self.length * cosine,
            )
        )
        velocity = self.length * self.omega[:, None] * tangent
        return position, velocity, tangent

    def _contact_impulses(self) -> np.ndarray:
        """Resolve impacts with a causal left-to-right sequential sweep."""
        impulse_vectors = np.zeros((self.n - 1, 3), dtype=float)
        position, _, tangent = self._kinematics()
        inertia = self.mass * self.length * self.length

        for i in range(self.n - 1):
            j = i + 1
            delta = position[j] - position[i]
            distance = float(np.linalg.norm(delta))
            if distance > 2.0 * self.radius + self.contact_skin:
                continue
            normal = delta / distance if distance > 1.0e-12 else np.array([1.0, 0.0, 0.0])
            # Generalized gap-rate Jacobians for the two pendulum angles.
            gi = -self.length * float(np.dot(normal, tangent[i]))
            gj = self.length * float(np.dot(normal, tangent[j]))
            normal_speed = gi * self.omega[i] + gj * self.omega[j]
            if normal_speed >= -1.0e-10:
                continue
            inverse_effective_mass = gi * gi / inertia[i] + gj * gj / inertia[j]
            if inverse_effective_mass <= 1.0e-14:
                continue
            impulse = -(1.0 + self.impact_restitution) * normal_speed
            impulse /= inverse_effective_mass
            self.omega[i] += gi * impulse / inertia[i]
            self.omega[j] += gj * impulse / inertia[j]
            impulse_vectors[i] += impulse * normal

        # Project tiny time-discretization overlaps out along the constrained
        # degrees of freedom without adding velocity or kinetic energy.
        for _ in range(2):
            position, _, tangent = self._kinematics()
            changed = False
            for i in range(self.n - 1):
                j = i + 1
                delta = position[j] - position[i]
                distance = float(np.linalg.norm(delta))
                penetration = 2.0 * self.radius - distance
                if penetration <= 1.0e-9:
                    continue
                normal = delta / distance if distance > 1.0e-12 else np.array([1.0, 0.0, 0.0])
                gi = -self.length * float(np.dot(normal, tangent[i]))
                gj = self.length * float(np.dot(normal, tangent[j]))
                denominator = gi * gi / inertia[i] + gj * gj / inertia[j]
                if denominator <= 1.0e-14:
                    continue
                multiplier = penetration / denominator
                self.theta[i] += gi * multiplier / inertia[i]
                self.theta[j] += gj * multiplier / inertia[j]
                changed = True
            if not changed:
                break
        return impulse_vectors

    def _step(self, h: float) -> np.ndarray:
        # Symplectic Euler is stable for the pendulum Hamiltonian.
        g = max(0.0, -float(self.gravity[2]))
        acceleration = -(g / self.length) * np.sin(self.theta)
        acceleration -= self.angular_drag * self.omega
        self.omega += h * acceleration
        self.theta += h * self.omega
        impulses = self._contact_impulses()
        self.step_count += 1
        return impulses

    def _external_force(self) -> np.ndarray:
        # Gravity and support constraint; internal contact is exported below.
        sine = np.sin(self.theta)
        cosine = np.cos(self.theta)
        outward = np.column_stack((sine, np.zeros(self.n), -cosine))
        radial_gravity = outward @ self.gravity
        tension = self.mass * (self.length * self.omega * self.omega + radial_gravity)
        tension = np.maximum(tension, 0.0)
        return self.mass[:, None] * self.gravity[None, :] - tension[:, None] * outward

    def _make_frame(self, time: float, pair_force: np.ndarray) -> StateFrame:
        position, velocity, _ = self._kinematics()
        gaps = np.linalg.norm(position[1:] - position[:-1], axis=1) - 2.0 * self.radius
        interaction = np.empty((2 * (self.n - 1), 3), dtype=float)
        interaction[0::2] = -pair_force
        interaction[1::2] = pair_force

        frame = StateFrame(
            float(time), position, velocity,
            orientation=self.orientation.copy(),
            angular_velocity=np.zeros_like(position),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": gaps.astype(float),
                "external_force": self._external_force(),
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to requires nondecreasing sample times")

        interval = target - self.time
        accumulated_impulse = np.zeros((self.n - 1, 3), dtype=float)
        while self.time < target - 1.0e-12:
            h = min(self.internal_dt, target - self.time)
            accumulated_impulse += self._step(h)
            self.time += h
        self.time = target
        pair_force = (accumulated_impulse / interval if interval > 1.0e-14
                      else np.zeros_like(accumulated_impulse))
        return self._make_frame(target, pair_force)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "constrained_pendulum_sequential_impulse",
            "internal_dt": self.internal_dt,
            "steps": self.step_count,
            "time": self.time,
        }
