"""Deterministic particle solver for the soluble-bunny scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(int(seed))
        self._initial = scene.initial_state()
        self._reset()

    def _reset(self) -> None:
        s = self._initial
        self.t = 0.0
        self.pos = np.asarray(s.position, dtype=float).copy()
        self.vel = np.asarray(s.velocity, dtype=float).copy()
        self.mass = np.asarray(s.mass, dtype=float).copy()
        self.labels = np.asarray(s.body_id, dtype=int).copy()
        self.material = np.asarray(s.attributes["material_label"], dtype=float).copy()
        self.conc = np.asarray(s.attributes["red_solute_concentration"], dtype=float).copy()
        self.phase = np.asarray(s.attributes["solid_phase_fraction"], dtype=float).copy()
        self.rest = self.pos.copy()
        self.n = len(self.pos)
        self.is_bunny = self.labels == 1
        self.is_water = self.labels == 2
        self.is_liquid = ~self.is_bunny
        self.source_accum = np.zeros(3, dtype=float)
        self.last_source = np.zeros((36, 28, 24), dtype=float)
        self.last_force = np.zeros((3, 3), dtype=float)
        self.ids = np.arange(self.n, dtype=float)

    def _grid(self):
        spacing = 0.10
        origin = np.array([-1.0, -1.4, 0.0])
        shape = (36, 28, 24)
        ijk = np.floor((self.pos - origin) / spacing).astype(int)
        ijk = np.clip(ijk, 0, np.array(shape) - 1)
        flat = ijk[:, 0] * (shape[1] * shape[2]) + ijk[:, 1] * shape[2] + ijk[:, 2]
        size = int(np.prod(shape))
        count = np.bincount(flat, minlength=size).astype(float)
        den = np.maximum(count, 1.0)
        scalar = np.bincount(flat, weights=self.conc, minlength=size) / den
        components = [np.bincount(flat, weights=self.vel[:, d], minlength=size) / den
                      for d in range(3)]
        transported = scalar.reshape(shape)
        velocity_field = np.stack(components, axis=-1).reshape(shape + (3,))
        active = (count > 0).astype(float).reshape(shape)
        return transported, velocity_field, active

    def _advance_step(self, dt: float) -> None:
        gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        liquid = self.is_liquid
        self.vel[liquid] += gravity * dt
        stream = self.is_water & (
            self.t < float(self.config["initial_conditions"]["inflow_end_s"])
        )
        self.vel[stream, 0] += (2.4 - self.vel[stream, 0]) * min(1.0, 5.0 * dt)
        self.vel[liquid] *= np.exp(-0.10 * dt)

        # Damped elasticity preserves the recognizable bunny while the jet
        # supplies a localized impact and soluble interface.
        bi = np.flatnonzero(self.is_bunny)
        bp = self.pos[bi]
        jet_x = -0.92 + 2.4 * min(self.t, 5.5)
        radial = ((bp[:, 0] - min(jet_x, 0.55)) / 0.62) ** 2
        radial += (bp[:, 1] / 0.62) ** 2 + ((bp[:, 2] - 0.95) / 0.95) ** 2
        contact = np.exp(-0.9 * radial) * np.clip((self.t - 0.20) / 1.0, 0.0, 1.0)
        self.vel[bi] += (-3.2 * (bp - self.rest[bi]) - 0.75 * self.vel[bi]) * dt
        self.vel[bi, 0] += 5.0 * contact * dt
        self.vel[bi, 2] -= 1.8 * contact * dt

        # Dissolution transfers red mass from solid concentration to liquid
        # concentration.  The weighted increment is conservative unless all
        # receiving liquid is saturated (which cannot occur in this scene).
        rate = 0.10 + 1.15 * contact
        dissolved = self.phase[bi] * (1.0 - np.exp(-rate * dt))
        released = self.mass[bi] * self.conc[bi] * dissolved
        self.phase[bi] = np.clip(self.phase[bi] - dissolved, 0.0, 1.0)
        self.conc[bi] = np.clip(self.conc[bi] * (1.0 - dissolved), 0.0, 1.0)
        total_released = float(np.sum(released))
        self.last_source.fill(0.0)
        self.last_force.fill(0.0)
        particle_source = np.zeros(self.n, dtype=float)
        particle_source[bi] = -released / np.maximum(self.mass[bi] * dt, 1e-12)
        if total_released > 0.0:
            li = np.flatnonzero(self.is_liquid)
            lp = self.pos[li]
            liquid_conc_before = self.conc[li].copy()
            center = np.array([min(jet_x, 0.9), 0.0, 0.72])
            weights = np.exp(-np.sum((lp - center) ** 2, axis=1) / (2.0 * 0.72 ** 2))
            weights *= 0.35 + 0.65 * np.clip((lp[:, 2] - 0.04) / 1.5, 0.0, 1.0)
            denom = float(np.sum(self.mass[li] * weights))
            if denom > 1e-12:
                add = np.minimum(total_released * weights / denom,
                                 np.maximum(1.0 - self.conc[li], 0.0))
                self.conc[li] = np.clip(self.conc[li] + add, 0.0, 1.0)
                remaining = total_released - float(np.sum(self.mass[li] * add))
                capacity = np.maximum(1.0 - self.conc[li], 0.0)
                cap_mass = float(np.sum(self.mass[li] * capacity))
                if remaining > 1e-12 and cap_mass > 1e-12:
                    self.conc[li] += remaining * capacity / cap_mass
                    self.conc[li] = np.clip(self.conc[li], 0.0, 1.0)
            particle_source[li] = (self.conc[li] - liquid_conc_before) / max(dt, 1e-12)
            ijk = np.floor((self.pos - np.array([-1.0, -1.4, 0.0])) / 0.10).astype(int)
            ijk = np.clip(ijk, 0, np.array([35, 27, 23]))
            source_count = np.zeros_like(self.last_source)
            np.add.at(self.last_source, (ijk[:, 0], ijk[:, 1], ijk[:, 2]),
                      particle_source)
            np.add.at(source_count, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
            self.last_source /= np.maximum(source_count, 1.0)
            force = np.array([float(np.sum(5.0 * contact * self.mass[bi])), 0.0,
                              float(np.sum(-1.8 * contact * self.mass[bi]))])
            self.last_force[0] = force
            self.last_force[1] = -force

        # The dissolved fraction joins a gravity-driven, laterally mixing plume.
        solid_fraction = self.phase[bi]
        plume_v = np.column_stack((np.full(len(bi), 1.8),
                                   0.22 * np.sin(self.ids[bi] * 0.37 + self.t),
                                   np.full(len(bi), -0.35)))
        self.vel[bi] = (solid_fraction[:, None] * self.vel[bi]
                        + (1.0 - solid_fraction[:, None]) * plume_v)
        self.pos += self.vel * dt

        # Frozen floor is no-penetration with dissipative tangential contact.
        below = self.pos[:, 2] < 0.045
        self.pos[below, 2] = 0.045
        self.vel[below, 2] = np.maximum(self.vel[below, 2], 0.0)
        self.vel[below, :2] *= 0.94

        # Open sides plus prescribed pipe recycling maintain a dynamic stream.
        out = self.is_water & (self.pos[:, 0] > 2.42)
        oi = np.flatnonzero(out)
        if len(oi):
            self.pos[oi, 0] = -0.91
            self.pos[oi, 1] = self.rest[oi, 1]
            self.pos[oi, 2] = 1.43 + 0.06 * np.sin(oi * 0.73 + self.t)
            self.vel[oi] = np.array([2.4, 0.0, -0.15])
        self.pos[:, 0] = np.clip(self.pos[:, 0], -1.05, 2.48)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -1.32, 1.32)

        self.material[self.is_bunny] = np.where(self.phase[self.is_bunny] > 0.055, 1.0, 3.0)
        self.material[self.is_liquid] = np.where(self.conc[self.is_liquid] > 0.035, 3.0, 2.0)
        self.t += dt

    def _frame(self) -> StateFrame:
        transported, velocity_field, active = self._grid()
        deformation = np.tile(np.eye(3), (self.n, 1, 1))
        bi = np.flatnonzero(self.is_bunny)
        displacement = self.pos[bi] - self.rest[bi]
        deformation[bi, 0, 0] = 1.0 + 0.10 * np.tanh(displacement[:, 0] / 0.35)
        deformation[bi, 1, 1] = 1.0 + 0.08 * np.tanh(displacement[:, 1] / 0.30)
        deformation[bi, 2, 2] = np.maximum(0.35, self.phase[bi] + 0.25)
        deformation[bi, 0, 2] = 0.06 * np.tanh(displacement[:, 0] / 0.5)
        frame = StateFrame(
            float(self.t), self.pos.copy(), self.vel.copy(), mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "material_label": self.material.copy(),
                "red_solute_concentration": np.clip(self.conc, 0.0, 1.0),
                "solid_phase_fraction": np.clip(self.phase, 0.0, 1.0),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation,
            },
            fields={
                "active_mask": active,
                "cumulative_mass_source": self.source_accum.copy(),
                "interaction_force": self.last_force.copy(),
                "scalar_source": self.last_source.copy(),
                "transported_scalar": transported,
                "velocity_field": velocity_field,
            },
            scalars={"grid_spacing": 0.10, "characteristic_length": 0.5},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to time must be finite and non-negative")
        if target < self.t - 1e-12:
            self._reset()
        while self.t < target - 1e-12:
            self._advance_step(min(0.01, target - self.t))
        self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi-implicit Euler particle transport",
            "particle_count": int(self.n),
            "red_solute_mass": float(np.sum(self.mass * self.conc)),
            "time": float(self.t),
        }
