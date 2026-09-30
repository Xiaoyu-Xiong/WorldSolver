"""Reduced-order physical solver for the cornstarch impact scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        cfg = scene.config
        self.cfg = cfg
        self.rest = np.asarray(scene.vertices, dtype=float).copy()
        self.n = self.rest.shape[0]
        self.mass_total = float(cfg["subject_mass"])
        self.mass = np.full(self.n, self.mass_total / self.n, dtype=float)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_length = float(cfg.get("contact_length_scale", 0.04))
        self.c0 = self.rest.mean(axis=0)
        self.local = self.rest - self.c0
        self.center = self.c0.copy()
        self.center_v = np.asarray(
            cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float
        ).copy()
        self.sr = self.sz = 1.0
        self.sr_v = self.sz_v = 0.0
        self.rest_r = self.rest_z = 1.0
        self.last_time = 0.0
        self.yielded = 0.0
        self.material_k = max(4.0, float(cfg.get("stiffness", 16.0)))
        self.material_damp = max(0.05, float(cfg.get("damping", 1.1)))
        self.restitution = float(np.clip(cfg.get("restitution", 0.54), 0.0, 1.0))
        self.friction = max(0.0, float(cfg.get("friction", 0.68)))
        self.yield_time = max(0.2, float(cfg.get("yield_time", 1.65)))
        self.dt_internal = 0.0025

    def _positions(self) -> np.ndarray:
        p = self.local.copy()
        p[:, :2] *= self.sr
        p[:, 2] *= self.sz
        return p + self.center

    def _velocities(self) -> np.ndarray:
        v = np.empty((self.n, 3), dtype=float)
        v[:, 0] = self.center_v[0] + self.sr_v * self.local[:, 0]
        v[:, 1] = self.center_v[1] + self.sr_v * self.local[:, 1]
        v[:, 2] = self.center_v[2] + self.sz_v * self.local[:, 2]
        return v

    def _substep(self, h: float) -> None:
        pos = self._positions()
        vel = self._velocities()
        gap = pos[:, 2] - self.ground_z
        penetration = np.maximum(-gap, 0.0)
        approaching = np.minimum(vel[:, 2], 0.0)

        # Penalty contact with velocity damping set by the configured restitution.
        contact_rate = 4200.0
        damping_rate = (
            2.0 * np.sqrt(contact_rate) * (1.0 - self.restitution + 0.08)
        )
        active = penetration > 0.0
        fn = self.mass * np.maximum(
            contact_rate * penetration - damping_rate * approaching, 0.0
        )
        fn *= active
        contact_fz = float(np.sum(fn))
        acc = self.gravity.copy()
        acc[2] += contact_fz / self.mass_total
        contact_fraction = float(np.count_nonzero(penetration)) / max(1, self.n)
        if contact_fraction > 0.0:
            speed = float(np.linalg.norm(self.center_v[:2]))
            if speed > 1e-10:
                acc[:2] -= (
                    self.friction
                    * 9.81
                    * contact_fraction
                    * self.center_v[:2]
                    / speed
                )
        self.center_v += h * acc
        self.center += h * self.center_v

        # Elastic affine stretch equations, including the generalized force of
        # floor contact on vertical scale.  This produces impact compression and
        # recovery without prescribing a trajectory.
        iz = self.local[:, 2]
        ir2 = self.local[:, 0] ** 2 + self.local[:, 1] ** 2
        effective_m = self.mass_total * 0.42
        kz = self.material_k * self.mass_total * 30.0
        kr = self.material_k * self.mass_total * 11.0
        fz_int = -kz * (self.sz - self.rest_z)
        fz_int -= self.material_damp * self.mass_total * 5.0 * self.sz_v
        fr_int = -kr * (self.sr - self.rest_r)
        fr_int -= self.material_damp * self.mass_total * 2.0 * self.sr_v
        fz_gen = float(np.sum(fn * iz))
        sz_acc = (fz_int + fz_gen) / max(
            effective_m * (np.mean(iz * iz) + 1e-4), 1e-5
        )
        sr_acc = fr_int / max(effective_m * (np.mean(ir2) + 1e-4), 1e-5)
        self.sz_v += h * sz_acc
        self.sr_v += h * sr_acc
        self.sz += h * self.sz_v
        self.sr += h * self.sr_v
        self.sz = float(np.clip(self.sz, 0.16, 1.35))
        self.sr = float(np.clip(self.sr, 0.72, 2.8))

        # Stress-dependent plastic rest-shape evolution: strong impacts shorten
        # the thickness and grow the radial rest scale at near-constant volume.
        stress = contact_fz / max(self.mass_total, 1e-9)
        excess = max(0.0, stress - 13.0) / 35.0
        if excess > 0.0:
            rate = min(5.0, excess / self.yield_time)
            self.rest_z += h * rate * (self.sz - self.rest_z)
            target_r = max(1.0, 1.0 / np.sqrt(max(self.rest_z, 0.2)))
            self.rest_r += h * rate * (target_r - self.rest_r)
            self.yielded += h * excess
        self.rest_z = float(np.clip(self.rest_z, 0.22, 1.0))
        self.rest_r = float(np.clip(self.rest_r, 1.0, 2.45))

        self.center_v = np.clip(self.center_v, -20.0, 20.0)
        self.sr_v = float(np.clip(self.sr_v, -12.0, 12.0))
        self.sz_v = float(np.clip(self.sz_v, -12.0, 12.0))
        pos = self._positions()
        min_gap = float(np.min(pos[:, 2] - self.ground_z))
        if min_gap < -0.01:
            self.center[2] -= min_gap + 0.002
            if self.center_v[2] < 0.0:
                self.center_v[2] *= -self.restitution

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.last_time - 1e-9:
            raise ValueError("advance_to requires finite nondecreasing time")
        remaining = target - self.last_time
        while remaining > 1e-10:
            h = min(self.dt_internal, remaining)
            self._substep(h)
            self.last_time += h
            remaining -= h

        position = self._positions()
        velocity = self._velocities()
        deformation = np.zeros((self.n, 3, 3), dtype=float)
        deformation[:, 0, 0] = self.sr
        deformation[:, 1, 1] = self.sr
        deformation[:, 2, 2] = self.sz
        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation,
            },
            fields={
                "contact_signed_distance": position[:, 2] - self.ground_z,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": self.contact_length},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_affine_viscoplastic",
            "internal_dt": self.dt_internal,
            "yield_accumulator": float(self.yielded),
            "final_radial_rest_stretch": float(self.rest_r),
            "final_vertical_rest_stretch": float(self.rest_z),
        }
