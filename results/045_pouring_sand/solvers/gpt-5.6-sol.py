"""Deterministic lightweight granular solver for the pouring-sand scene."""
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
        self.radius = np.asarray(initial.attributes["nominal_radius"], dtype=float).copy()
        self.attributes = {k: np.asarray(v, dtype=float).copy() for k, v in initial.attributes.items()}
        self.spacing = float(self.config["initial_conditions"]["particle_spacing"])
        self.rho = float(self.config["initial_conditions"]["bulk_density"])
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.current_time = 0.0
        self.internal_dt = min(0.004, float(self.config.get("sample_dt", 0.02)) / 4.0)
        self.restitution = 0.16
        self.friction = 0.82
        self._rng = np.random.default_rng(int(seed))
        noise = self._rng.normal(0.0, 1.0, size=(len(self.position), 2))
        self.roughness = 0.035 * noise / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        self._impacted = np.zeros(len(self.position), dtype=bool)
        self._last_contact = np.zeros(len(self.position), dtype=float)
        self._initial_mass = float(np.sum(self.mass))
        self.position[:, 2] = np.maximum(self.position[:, 2], self.radius)

    @staticmethod
    def _hopper_radius(z):
        z = np.asarray(z)
        out = np.full(z.shape, np.inf, dtype=float)
        neck = (z >= 1.42) & (z < 1.52)
        out[neck] = 0.075
        slope = (z >= 1.52) & (z < 1.72)
        out[slope] = 0.075 + (z[slope] - 1.52) * (0.34 - 0.075) / 0.20
        upper = (z >= 1.72) & (z <= 2.30)
        out[upper] = 0.34
        return out

    def _pile_surface(self, x, y):
        count = int(np.count_nonzero(self._impacted))
        if count == 0:
            return np.zeros_like(x, dtype=float)
        volume = count * float(np.mean(self.mass)) / self.rho
        tan_a = np.tan(np.deg2rad(32.0))
        height = (3.0 * volume * tan_a * tan_a / np.pi) ** (1.0 / 3.0)
        rr = np.sqrt(x * x + y * y)
        return np.maximum(0.0, height - tan_a * np.maximum(rr - 0.045, 0.0))

    def _separate_reservoir_columns(self):
        z = self.position[:, 2]
        inside = (z >= 1.40) & (z <= 2.31)
        if not np.any(inside):
            return
        cell = max(2.0 * float(np.mean(self.radius)), self.spacing * 0.9)
        xy = self.position[inside, :2]
        ix = np.floor((xy[:, 0] + 0.5) / cell).astype(np.int64)
        iy = np.floor((xy[:, 1] + 0.5) / cell).astype(np.int64)
        keys = ix * 10000 + iy
        order = np.argsort(keys)
        ids = np.flatnonzero(inside)[order]
        keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
        ends = np.r_[starts[1:], len(ids)]
        min_sep = 1.75 * float(np.mean(self.radius))
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            group = ids[a:b]
            so = group[np.argsort(self.position[group, 2])]
            zz = self.position[so, 2]
            self.position[so, 2] = np.maximum.accumulate(zz - np.arange(len(zz)) * min_sep) + np.arange(len(zz)) * min_sep
            self.position[so, 2] = np.minimum(self.position[so, 2], 2.31)

    def _step(self, dt):
        p, v, r = self.position, self.velocity, self.radius
        v += self.gravity[None, :] * dt
        v[:, :2] += self.roughness * dt
        v *= np.exp(-0.045 * dt)
        p += v * dt

        hr = self._hopper_radius(p[:, 2])
        radial = np.linalg.norm(p[:, :2], axis=1)
        limit = hr - r
        hit = np.isfinite(hr) & (radial > limit)
        if np.any(hit):
            safe = np.maximum(radial[hit], 1e-12)
            nxy = p[hit, :2] / safe[:, None]
            p[hit, :2] = nxy * np.maximum(limit[hit], 0.005)[:, None]
            vn = np.sum(v[hit, :2] * nxy, axis=1)
            inward = vn > 0.0
            ids = np.flatnonzero(hit)[inward]
            if len(ids):
                v[ids, :2] -= (1.0 + self.restitution) * vn[inward, None] * nxy[inward]
                v[ids, :2] *= self.friction

        rr = np.linalg.norm(p[:, :2], axis=1)
        self._impacted |= (p[:, 2] < 0.42) & (rr < 1.35)

        pile_h = self._pile_surface(p[:, 0], p[:, 1])
        gap = p[:, 2] - pile_h - r
        contact = gap < 0.0
        if np.any(contact):
            ids = np.flatnonzero(contact)
            p[ids, 2] -= gap[ids]
            x, y = p[ids, 0], p[ids, 1]
            rad = np.sqrt(x * x + y * y)
            slope = np.tan(np.deg2rad(32.0))
            n = np.stack((np.where(rad > 1e-8, slope * x / rad, 0.0), np.where(rad > 1e-8, slope * y / rad, 0.0), np.ones_like(x)), axis=1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            vn = np.sum(v[ids] * n, axis=1)
            approaching = vn < 0.0
            if np.any(approaching):
                aid, na = ids[approaching], n[approaching]
                v[aid] -= (1.0 + self.restitution) * np.sum(v[aid] * na, axis=1)[:, None] * na
                v[aid] *= self.friction
            self._last_contact = np.minimum(self._last_contact, gap)

        resting = (p[:, 2] <= pile_h + r + 1e-6) & (np.abs(v[:, 2]) < 0.35)
        v[resting, :2] *= np.exp(-5.0 * dt)
        v[resting, 2] = np.maximum(v[resting, 2], 0.0)
        p[:, 2] = np.maximum(p[:, 2], r)
        p[:, 0] = np.clip(p[:, 0], -1.58, 1.58)
        p[:, 1] = np.clip(p[:, 1], -1.28, 1.28)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1e-9:
            raise ValueError("advance_to requires finite nondecreasing time")
        steps = 0
        while self.current_time < target - 1e-12:
            dt = min(self.internal_dt, target - self.current_time)
            self._step(dt)
            self.current_time += dt
            steps += 1
            if steps % 4 == 0:
                self._separate_reservoir_columns()

        pile_h = self._pile_surface(self.position[:, 0], self.position[:, 1])
        hopper_gap = self._hopper_radius(self.position[:, 2]) - np.linalg.norm(self.position[:, :2], axis=1) - self.radius
        floor_gap = self.position[:, 2] - self.radius
        contact_gap = np.minimum(floor_gap, np.where(np.isfinite(hopper_gap), hopper_gap, floor_gap))
        contact_gap = np.minimum(contact_gap, self.position[:, 2] - pile_h - self.radius)
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={**self.attributes, "mass_quantity": self.mass.copy(), "volume_quantity": self.mass / self.rho},
            fields={"cumulative_mass_source": np.zeros(1), "external_force": self.mass[:, None] * self.gravity[None, :], "contact_signed_distance": contact_gap.copy()},
            scalars={"contact_length_scale": float(2.0 * np.mean(self.radius)), "initial_total_mass": self._initial_mass},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "semi_implicit_granular_particle", "particle_count": int(len(self.position)), "total_mass": float(np.sum(self.mass)), "impacted_particles": int(np.count_nonzero(self._impacted))}
