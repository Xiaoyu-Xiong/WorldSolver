"""Deterministic particle solver for the liquid bunny impact scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.config = scene.config
        self.rng = np.random.default_rng(int(seed))
        self.time = float(frame.time)
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.labels = np.asarray(frame.attributes.get("material_label", frame.body_id), dtype=np.int32).copy()
        self.spacing = float(frame.scalars.get("grid_spacing", self.config["sampling"]["spacing"]))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.grid_shape = tuple(int(v) for v in self.config["sampling"]["field_grid_shape"])
        self.domain_min = np.asarray([-3.0, -1.2, 0.0], dtype=np.float64)
        self.domain_max = np.asarray([3.0, 1.2, 3.8], dtype=np.float64)
        self.cell_size = (self.domain_max - self.domain_min) / np.asarray(self.grid_shape, dtype=float)
        self.n = len(self.position)
        self.impact_done = False
        self.impact_center = np.zeros(2, dtype=float)
        self.impact_strength = 0.0
        self._template_attributes = {
            "particle_radius": np.asarray(frame.attributes.get("particle_radius", np.full(self.n, .45 * self.spacing)), dtype=float).copy(),
            "material_label": self.labels.copy(),
            "deformation_gradient": np.asarray(frame.attributes.get("deformation_gradient", np.repeat(np.eye(3)[None], self.n, axis=0)), dtype=float).copy(),
        }

    def _indices(self, pos):
        q = np.floor((pos - self.domain_min) / self.cell_size).astype(np.int64)
        q = np.clip(q, 0, np.asarray(self.grid_shape) - 1)
        flat = q[:, 0] * (self.grid_shape[1] * self.grid_shape[2]) + q[:, 1] * self.grid_shape[2] + q[:, 2]
        return q[:, 0], q[:, 1], q[:, 2], flat

    def _pressure_acceleration(self):
        nx, ny, nz = self.grid_shape
        ix, iy, iz, flat = self._indices(self.position)
        count = np.bincount(flat, minlength=nx * ny * nz).reshape(self.grid_shape).astype(float)
        pressure = np.clip((count - 1.5) / 5.0, 0.0, 1.0)
        gx, gy, gz = np.gradient(pressure, self.cell_size[0], self.cell_size[1], self.cell_size[2], edge_order=1)
        acc = np.column_stack((-gx[ix, iy, iz], -gy[ix, iy, iz], -gz[ix, iy, iz]))
        active = (self.labels == 0) | (self.position[:, 2] < 0.95) | self.impact_done
        return np.clip(acc * active[:, None] * 3.0, -24.0, 24.0)

    def _impact_impulse(self):
        bunny = self.labels == 1
        contact = bunny & (self.position[:, 2] < 0.69)
        if self.impact_done or not np.any(contact):
            return
        self.impact_done = True
        self.impact_center = self.position[contact, :2].mean(axis=0)
        downward = float(np.maximum(0.0, -self.velocity[contact, 2]).mean())
        self.impact_strength = float(np.clip(1.4 + 0.75 * downward, 1.8, 8.0))
        dxy = self.position[:, :2] - self.impact_center[None, :]
        r = np.linalg.norm(dxy, axis=1)
        radial = dxy / np.maximum(r[:, None], 1.0e-5)
        ring = np.exp(-((r - 0.42) / 0.48) ** 2)
        near = np.exp(-((self.position[:, 2] - 0.48) / 0.42) ** 2)
        weight = ring * near
        self.velocity[:, :2] += radial * (self.impact_strength * weight * 0.95)[:, None]
        self.velocity[:, 2] += self.impact_strength * (0.62 * weight + 0.45 * np.exp(-(r / 0.30) ** 2) * near)
        self.velocity[contact, :2] += radial[contact] * self.impact_strength * 0.25
        self.velocity[contact, 2] = np.maximum(self.velocity[contact, 2], self.impact_strength * 0.35)

    def _contacts(self):
        p, v = self.position, self.velocity
        floor = 0.065
        hit = p[:, 2] < floor
        p[hit, 2] = floor
        v[hit, 2] = np.abs(v[hit, 2]) * 0.18
        v[hit, :2] *= 0.92
        for mask, wall in ((p[:, 0] < -1.76, -1.76), (p[:, 0] > 1.76, 1.76)):
            p[mask, 0] = wall
            v[mask, 0] = -0.22 * v[mask, 0]
            v[mask, 1] *= 0.94
        back = p[:, 1] > 0.83
        p[back, 1] = 0.83
        v[back, 1] = -0.2 * v[back, 1]
        p[:, 0] = np.clip(p[:, 0], -2.95, 2.95)
        p[:, 1] = np.clip(p[:, 1], -1.16, 1.16)
        p[:, 2] = np.clip(p[:, 2], floor, 3.7)

    def _step(self, dt):
        self.velocity += self.gravity[None, :] * dt
        self.velocity += self._pressure_acceleration() * dt
        pool, bunny = self.labels == 0, self.labels == 1
        self.velocity[pool] *= np.exp(-0.42 * dt)
        self.velocity[bunny] *= np.exp(-(0.28 if not self.impact_done else 0.72) * dt)
        low_bunny = bunny & (self.position[:, 2] < 0.92)
        self.velocity[low_bunny, 2] += -1.8 * (self.position[low_bunny, 2] - 0.46) * dt
        self.position += self.velocity * dt
        self._impact_impulse()
        if self.impact_done:
            submerged = bunny & (self.position[:, 2] < 0.82)
            self.velocity[submerged] *= 0.985
        self._contacts()

    def _diagnostic_fields(self):
        nx, ny, nz = self.grid_shape
        _, _, _, flat = self._indices(self.position)
        count = np.bincount(flat, minlength=nx * ny * nz).reshape(self.grid_shape).astype(float)
        vel_sum = np.zeros((nx * ny * nz, 3), dtype=float)
        np.add.at(vel_sum, flat, self.velocity)
        velocity_field = (vel_sum / np.maximum(count.ravel()[:, None], 1.0)).reshape(self.grid_shape + (3,))
        liquid_level_set = (0.55 - np.minimum(count / 2.0, 1.0)) * float(np.mean(self.cell_size))
        normals = np.asarray([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        zeros = np.zeros_like(normals)
        return {"active_mask": count > 0.0, "boundary_normal": normals,
                "boundary_target_velocity": zeros.copy(), "boundary_velocity": zeros.copy(),
                "cumulative_mass_source": np.zeros(2), "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field}

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time:
            raise ValueError("advance_to requires a finite non-decreasing time")
        while self.time < target - 1.0e-10:
            dt = min(1.0 / 90.0, target - self.time)
            self._step(dt)
            self.time += dt
        attrs = {k: v.copy() for k, v in self._template_attributes.items()}
        attrs["mass_quantity"] = self.mass.copy()
        frame = StateFrame(self.time, self.position.copy(), self.velocity.copy(), mass=self.mass.copy(),
                           body_id=self.labels.copy(), attributes=attrs, fields=self._diagnostic_fields(),
                           scalars={"characteristic_length": float(max(np.ptp(self.position, axis=0).max(), self.spacing)),
                                    "grid_spacing": float(self.spacing),
                                    "reference_velocity": float(max(1.0, np.linalg.norm(self.velocity, axis=1).max()))})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"particle_count": int(self.n), "impact_occurred": bool(self.impact_done),
                "impact_strength": float(self.impact_strength), "simulated_time": float(self.time)}
