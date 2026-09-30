"""Physical solver for the inclined plastic chair scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        initial = scene.initial_state()
        self.mass_value = float(np.asarray(initial.mass)[0]) if initial.mass is not None else 10.8
        self.position0 = np.asarray(initial.position[0], dtype=float).copy()
        self.orientation = np.asarray(initial.orientation[0], dtype=float).copy()
        self.theta = 2.0 * np.arctan2(self.orientation[1], self.orientation[3])
        self.normal = np.array([np.sin(self.theta), 0.0, np.cos(self.theta)], dtype=float)
        self.tangent = np.array([np.cos(self.theta), 0.0, -np.sin(self.theta)], dtype=float)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.gn = -float(np.dot(self.gravity, self.normal))
        self.gt = float(np.dot(self.gravity, self.tangent))

        # Generalized normal and tangential contact coordinates.
        self.s = 0.0
        self.v = 0.0
        self.gap = 0.0
        self.gap_velocity = 0.0
        self.last_time = 0.0
        self.sticking = False
        self.slip_events = 0

        # Compliant plastic feet plus Coulomb friction.  The normal contact
        # mode modulates the friction cone and causes repeated stick/slip.
        self.normal_k = 12000.0
        self.normal_damping = 120.0
        self.tangent_damping = 2.0
        self.mu_static = 0.40
        self.mu_dynamic = 0.34
        self.velocity_epsilon = 2.0e-3

    def _step(self, dt: float) -> None:
        m = self.mass_value
        compression = max(0.0, -self.gap)
        normal_force = max(
            0.0,
            self.normal_k * compression - self.normal_damping * self.gap_velocity,
        )
        normal_acceleration = normal_force / m - self.gn

        required = m * self.gt
        static_capacity = self.mu_static * normal_force
        was_sticking = self.sticking
        if abs(self.v) <= self.velocity_epsilon and required <= static_capacity:
            self.sticking = True
            self.v = 0.0
            tangential_acceleration = 0.0
        else:
            if was_sticking:
                self.slip_events += 1
            self.sticking = False
            direction = 1.0 if self.v >= 0.0 else -1.0
            friction = self.mu_dynamic * normal_force * direction
            tangential_acceleration = (
                self.gt - friction / m - self.tangent_damping * self.v / m
            )
        if (not was_sticking) and self.sticking:
            self.slip_events += 1

        # Semi-implicit Euler: stable and mildly dissipative for contact.
        self.v += dt * tangential_acceleration
        if self.sticking:
            self.v = 0.0
        self.s += dt * self.v
        self.gap_velocity += dt * normal_acceleration
        self.gap += dt * self.gap_velocity

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.last_time:
            raise ValueError("advance_to requires finite nondecreasing time")
        remaining = target - self.last_time
        while remaining > 1.0e-12:
            dt = min(5.0e-4, remaining)
            self._step(dt)
            remaining -= dt
        self.last_time = target

        position = self.position0 + self.s * self.tangent + self.gap * self.normal
        velocity = self.v * self.tangent + self.gap_velocity * self.normal
        external_force = self.mass_value * self.gravity
        frame = StateFrame(
            target,
            position[None, :],
            velocity[None, :],
            orientation=self.orientation[None, :],
            angular_velocity=np.zeros((1, 3), dtype=float),
            mass=np.array([self.mass_value], dtype=float),
            body_id=np.array([0], dtype=np.int64),
            fields={
                "constraint_target": np.zeros((1, 1), dtype=float),
                "constraint_value": np.array([[self.v]], dtype=float),
                "contact_signed_distance": np.array([self.gap], dtype=float),
                "external_force": external_force[None, :],
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": 0.025,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_penalty_contact",
            "stick_slip_events": int(self.slip_events),
            "incline_angle_radians": float(self.theta),
        }
