"""Deterministic particle solver for snow-castle impact and collapse."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.snow_count = len(self.position) - 1
        self.reference = self.position[:self.snow_count].copy()
        ic = self.config.get("initial_conditions", {})
        self.spacing = float(ic.get("snow_particle_spacing", 0.075))
        self.snow_radius = 0.53 * self.spacing
        self.striker_radius = float(ic.get("striker_radius", 0.42))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.time = 0.0
        self.damage = np.zeros(self.snow_count, dtype=float)
        self.impact_work = 0.0
        self.impact_center = np.array([-1.25, 0.05, 0.82], dtype=float)
        self.last_contact_distance = 10.0
        self.initial_mass_total = float(self.mass.sum())

    def _substep(self, dt: float) -> None:
        n = self.snow_count
        snow_p, snow_v = self.position[:n], self.velocity[:n]
        ball_p, ball_v = self.position[n], self.velocity[n]
        force = self.mass[:n, None] * self.gravity[None, :]
        displacement = snow_p - self.reference
        stiffness = 360.0 * (1.0 - self.damage) ** 2
        force += -stiffness[:, None] * displacement - 18.0 * snow_v

        delta = snow_p - ball_p[None, :]
        dist = np.linalg.norm(delta, axis=1)
        safe_dist = np.maximum(dist, 1.0e-8)
        normal = delta / safe_dist[:, None]
        penetration = self.striker_radius + self.snow_radius - dist
        active = penetration > 0.0
        relative_normal_speed = np.sum((snow_v - ball_v[None, :]) * normal, axis=1)
        contact_mag = np.where(active, np.maximum(0.0, 8500.0 * penetration - 90.0 * relative_normal_speed), 0.0)
        contact_force = contact_mag[:, None] * normal
        force += contact_force
        ball_force = self.mass[n] * self.gravity - np.sum(contact_force, axis=0)

        if np.any(active):
            power = np.maximum(0.0, contact_mag * (-relative_normal_speed))
            self.impact_work += float(np.sum(power) * dt)
            center = np.average(snow_p[active], axis=0, weights=contact_mag[active] + 1e-6)
            self.impact_center = 0.85 * self.impact_center + 0.15 * center

        # Propagating fracture: deposited impact energy relaxes the cohesive snow
        # toward a damaged state, attenuated with distance from actual contact.
        if self.impact_work > 0.0:
            radius = np.linalg.norm(self.reference - self.impact_center[None, :], axis=1)
            energy_kernel = np.exp(-((radius / 2.0) ** 2))
            target_damage = np.clip((self.impact_work / 55.0) * energy_kernel, 0.0, 0.995)
            self.damage += (1.0 - np.exp(-4.5 * dt)) * np.maximum(0.0, target_damage - self.damage)

        snow_v += dt * force / self.mass[:n, None]
        ball_v += dt * ball_force / self.mass[n]
        snow_p += dt * snow_v
        ball_p += dt * ball_v

        below = snow_p[:, 2] < self.snow_radius
        if np.any(below):
            snow_p[below, 2] = self.snow_radius
            snow_v[below, 2] = np.maximum(0.0, -0.12 * snow_v[below, 2])
            snow_v[below, :2] *= 0.82
        if ball_p[2] < self.striker_radius:
            ball_p[2] = self.striker_radius
            if ball_v[2] < 0.0:
                ball_v[2] *= -0.28
            ball_v[:2] *= 0.94
        self.position[:] = np.nan_to_num(self.position, nan=0.0, posinf=4.0, neginf=-4.0)
        self.velocity[:] = np.nan_to_num(self.velocity, nan=0.0, posinf=40.0, neginf=-40.0)
        self.time += dt

    def _frame(self) -> StateFrame:
        n = self.snow_count
        gap = float(np.min(np.linalg.norm(self.position[:n] - self.position[n], axis=1)) - self.striker_radius - self.snow_radius)
        self.last_contact_distance = gap
        deformation = np.broadcast_to(np.eye(3), (len(self.position), 3, 3)).copy()
        deformation[:n, 0, 0] = 1.0 - 0.35 * self.damage
        deformation[:n, 1, 1] = 1.0 - 0.20 * self.damage
        deformation[:n, 2, 2] = 1.0 + 0.10 * self.damage
        attrs = {
            "initial_volume": np.full(len(self.position), self.spacing ** 3, dtype=float),
            "mass_quantity": self.mass.copy(),
            "deformation_gradient": deformation,
        }
        fields = {
            "contact_signed_distance": np.array([gap], dtype=float),
            "cumulative_mass_source": np.array([0.0], dtype=float),
            "external_force": self.mass[:, None] * self.gravity[None, :],
        }
        frame = StateFrame(time=float(self.time), position=self.position.copy(), velocity=self.velocity.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(), attributes=attrs, fields=fields, scalars={"initial_total_mass": self.initial_mass_total, "contact_length_scale": self.spacing})
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1e-12:
            self._substep(min(0.004, target - self.time))
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "symplectic_euler_penalty_contact", "particle_count": int(len(self.position)), "impact_work": float(self.impact_work), "mass_total": float(self.mass.sum()), "contact_signed_distance": float(self.last_contact_distance)}
