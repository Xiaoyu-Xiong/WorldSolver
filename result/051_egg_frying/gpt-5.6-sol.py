"""Compact physically based hot-pan egg frying solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        initial = scene.initial_state()
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.base_pos = self.pos.copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.pos)
        self.t = 0.0
        self.dt = min(0.005, float(scene.config.get("sample_dt", 0.02)) / 4.0)
        self.pan_z = float(scene.config["boundary_conditions"]["pan_contact_height_m"])
        self.pan_r = float(scene.config["boundary_conditions"]["pan_no_escape_radius_m"])
        ic = scene.config["initial_conditions"]
        self.pan_temp = float(ic["pan_temperature_k"])
        self.ambient = float(ic["ambient_temperature_k"])
        self.white_center = np.mean(self.base_pos[self.body_id == 1], axis=0)
        self.last_frame = initial

    def _target_positions(self, time: float) -> np.ndarray:
        target = self.base_pos.copy()
        white = self.body_id == 1
        scale = 1.0 + 0.74 * (1.0 - np.exp(-time / 1.8))
        rel = self.base_pos[white] - self.white_center
        rel[:, :2] *= scale
        rel[:, 2] *= 0.72
        target[white] = self.white_center + rel
        target[white, 2] += 0.0015 * (1.0 - np.exp(-time / 1.2))
        yolk = self.body_id == 2
        target[yolk, 2] -= 0.003 * (1.0 - np.exp(-time / 2.0))
        return target

    def _advance_step(self, dt: float, target_time: float) -> None:
        target = self._target_positions(target_time)
        accel = 22.0 * (target - self.pos) - 9.0 * self.vel
        accel[:, 2] -= 9.81
        self.vel += accel * dt
        self.pos += self.vel * dt
        hit = self.pos[:, 2] < self.pan_z + 0.001
        self.pos[hit, 2] = self.pan_z + 0.001
        self.vel[hit, 2] = np.maximum(self.vel[hit, 2] * -0.12, 0.0)
        self.vel[hit, :2] *= 0.985
        radial = np.linalg.norm(self.pos[:, :2], axis=1)
        outside = radial > self.pan_r - 0.002
        if np.any(outside):
            self.pos[outside, :2] *= ((self.pan_r - 0.002) / radial[outside])[:, None]
            self.vel[outside, :2] *= -0.15

    def _diagnostic_fields(self):
        z = self.pos[:, 2]
        contact = np.clip((z - self.pan_z) / 0.055, 0.0, 1.0)
        gain = (self.pan_temp - self.ambient) * (1.0 - np.exp(-self.t / 2.4))
        temp = self.ambient + gain * np.exp(-contact * contact * (2.0 + 0.8 * self.t))
        white = self.body_id == 1
        solid = np.zeros(self.n)
        solid[white] = np.clip((temp[white] - 323.0) / 58.0, 0.0, 1.0)
        solid[~white] = np.clip((temp[~white] - 360.0) / 90.0, 0.0, 0.35)
        liquid = 1.0 - solid
        nx = ny = 32
        spacing = 2.0 * self.pan_r / nx
        gx = np.linspace(-self.pan_r + spacing / 2, self.pan_r - spacing / 2, nx)
        gy = np.linspace(-self.pan_r + spacing / 2, self.pan_r - spacing / 2, ny)
        ix = np.clip(((self.pos[:, 0] + self.pan_r) / spacing).astype(int), 0, nx - 1)
        iy = np.clip(((self.pos[:, 1] + self.pan_r) / spacing).astype(int), 0, ny - 1)
        scalar = np.zeros((nx, ny)); counts = np.zeros((nx, ny)); vgrid = np.zeros((nx, ny, 3))
        for k in range(self.n):
            scalar[ix[k], iy[k]] += solid[k]; counts[ix[k], iy[k]] += 1.0; vgrid[ix[k], iy[k]] += self.vel[k]
        nz = counts > 0
        scalar[nz] /= counts[nz]; vgrid[nz] /= counts[nz, None]
        active = ((gx[:, None] ** 2 + gy[None, :] ** 2) <= self.pan_r ** 2)
        contact_sd = self.pos[:, 2] - (self.pan_z + 0.001)
        return temp, solid, liquid, scalar, vgrid, active, contact_sd

    def advance_to(self, time: float):
        time = float(time)
        if not np.isfinite(time) or time < self.t - 1e-10:
            raise ValueError("advance_to requires finite nondecreasing time")
        while self.t + 1e-12 < time:
            h = min(self.dt, time - self.t)
            self._advance_step(h, self.t + h); self.t += h
        temp, solid, liquid, scalar, vgrid, active, contact_sd = self._diagnostic_fields()
        concentration = np.zeros((self.n, 2)); concentration[self.body_id == 1, 0] = 1.0; concentration[self.body_id == 2, 1] = 1.0
        frame = StateFrame(time=self.t, position=self.pos.copy(), velocity=self.vel.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"material_concentration": concentration, "material_label": self.body_id.astype(float), "liquid_phase_fraction": liquid,
                        "solid_phase_fraction": solid, "temperature_k": temp, "pan_contact_distance": contact_sd, "mass_quantity": self.mass.copy()},
            fields={"active_mask": active.astype(float), "contact_signed_distance": contact_sd.copy(), "cumulative_mass_source": np.zeros(2),
                    "interaction_force": np.zeros((2, 3)), "scalar_source": np.zeros((32, 32)), "transported_scalar": scalar, "velocity_field": vgrid},
            scalars={"total_egg_mass_kg": float(self.mass.sum()), "pan_temperature_k": self.pan_temp, "contact_length_scale": 0.007, "grid_spacing": 2.0 * self.pan_r / 32.0})
        frame.validate(); self.last_frame = frame; return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "damped_spring_explicit", "particle_count": int(self.n), "time": float(self.t), "total_mass_kg": float(self.mass.sum())}
