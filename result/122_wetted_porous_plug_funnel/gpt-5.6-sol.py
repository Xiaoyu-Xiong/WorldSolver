"""Self-contained wet porous-plug and particle-liquid physics solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Lagrangian liquid coupled to an affine Kelvin--Voigt porous solid."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene, self.cfg = scene, scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.rest = np.asarray(scene.plug_vertices, float).copy()
        self.np = len(self.rest)
        self.wx = np.asarray(initial.position[self.np:], float).copy()
        self.wv = np.asarray(initial.velocity[self.np:], float).copy()
        self.wm = np.asarray(initial.mass[self.np:], float).copy()
        p = self.cfg["plug"]
        self.c0 = np.asarray(p["center"], float)
        self.rel = self.rest - self.c0
        self.c, self.cv = self.c0.copy(), np.zeros(3)
        self.qr = self.qz = 1.0
        self.qrv = self.qzv = 0.0
        self.sat = float(p["initial_saturation"])
        self.pm_total = float(p["mass"])
        self.pm = np.full(self.np, self.pm_total / self.np)
        self.mass = np.r_[self.pm, self.wm]
        self.n = len(self.mass)
        self.body = np.r_[np.ones(self.np, dtype=np.int64),
                          np.full(len(self.wx), 2, dtype=np.int64)]
        self.g = np.asarray(self.cfg["gravity"], float)
        self.r0, self.h0 = float(p["radius"]), float(p["height"])
        self.vol0 = np.pi * self.r0**2 * self.h0
        self.mqr = max(.5 * self.pm_total * self.r0**2, 1e-5)
        self.mqz = max(self.pm_total * self.h0**2 / 12, 1e-5)
        self.E0 = float(p["dry_young_modulus"])
        self.wetfactor = float(p["wet_softening_factor"])
        self.phi = float(p["porosity"])
        self.perm = float(p["permeability"])
        # Budgets: global, solid, liquid. Terms: gravity, boundary, coupling.
        self.weights = np.zeros((3, self.n))
        self.weights[0] = 1
        self.weights[1, :self.np] = 1
        self.weights[2, self.np:] = 1
        self.impulse = np.zeros((3, 3))
        self.budget_force = np.zeros((3, 3, 3))
        self.interaction = np.zeros((2, 3))
        self.zero_momentum = np.zeros((3, 3))
        self.steps = 0
        scale = max(float(self.cfg["funnel"]["rim_z"]), self.h0)
        self.p_ref = float(self.mass.sum() * np.sqrt(np.linalg.norm(self.g) * scale))

    @staticmethod
    def _profile(z):
        """Piecewise-linear frozen inner wall: radius, slope, active mask."""
        z = np.asarray(z)
        radius, slope = np.full_like(z, .31), np.zeros_like(z)
        active = (z >= .86) & (z <= 2.15)
        zs, rs = (.86, 1.02, 1.48, 1.88, 2.15), (.31, .31, .58, .88, 1.10)
        for j in range(4):
            hit = (z >= zs[j]) & (z <= zs[j + 1])
            s = (rs[j + 1] - rs[j]) / (zs[j + 1] - zs[j])
            radius[hit], slope[hit] = rs[j] + s * (z[hit] - zs[j]), s
        return radius, slope, active

    def _plug_state(self):
        d = self.rel.copy()
        d[:, :2] *= self.qr
        d[:, 2] *= self.qz
        x = self.c + d
        v = np.broadcast_to(self.cv, x.shape).copy()
        v[:, :2] += self.rel[:, :2] * self.qrv
        v[:, 2] += self.rel[:, 2] * self.qzv
        return x, v

    def _solid_velocity(self, x):
        v = np.broadcast_to(self.cv, x.shape).copy()
        v[:, :2] += (x[:, :2] - self.c[:2]) * self.qrv / max(self.qr, 1e-6)
        v[:, 2] += (x[:, 2] - self.c[2]) * self.qzv / max(self.qz, 1e-6)
        return v

    def _plug_contact(self):
        x, v = self._plug_state()
        r = np.linalg.norm(x[:, :2], axis=1)
        rw, slope, active = self._profile(x[:, 2])
        pen = r - rw
        hit = active & (pen > 0) & (r > 1e-9)
        f = np.zeros_like(x)
        if hit.any():
            er = x[hit, :2] / r[hit, None]
            norm = np.sqrt(1 + slope[hit]**2)
            nxy, nz = er / norm[:, None], -slope[hit] / norm
            vn = (v[hit, :2] * nxy).sum(1) + v[hit, 2] * nz
            # Vertices are surface quadrature samples: scale the body-level
            # penalty by the 16 samples on each circular contact rim.
            mag = (5000 * pen[hit] + 20 * np.maximum(vn, 0)) / 16.0
            f[hit, :2], f[hit, 2] = -mag[:, None] * nxy, -mag * nz
        floor_pen = .06 - x[:, 2]
        hit = floor_pen > 0
        f[hit, 2] += (8000 * floor_pen[hit]
                      + 25 * np.maximum(-v[hit, 2], 0)) / 16.0
        return f.sum(0), float((f[:, :2] * self.rel[:, :2]).sum()), \
            float((f[:, 2] * self.rel[:, 2]).sum())

    def _porous_drag(self, dt):
        d = self.wx - self.c
        inside = ((d[:, :2]**2).sum(1) <= (self.r0 * self.qr)**2) & \
                 (np.abs(d[:, 2]) <= .5 * self.h0 * self.qz)
        fw = np.zeros(3)
        if inside.any():
            tau = (.10 + .30 / (1 + self.perm)) * (1 + .6 * self.sat)
            a = (self._solid_velocity(self.wx[inside]) - self.wv[inside]) / tau
            amag = np.linalg.norm(a, axis=1)
            a *= np.minimum(1, 45 / np.maximum(amag, 1e-12))[:, None]
            self.wv[inside] += dt * a
            fw = (self.wm[inside, None] * a).sum(0)
            particle_vol = float(self.wm.mean() / self.cfg["liquid"]["density"])
            capacity = max(self.phi * self.vol0 / particle_vol, 1)
            occupancy = min(inside.sum() / capacity, 1)
            self.sat += dt * 5 * occupancy * (1 - self.sat)
            self.sat = float(np.clip(self.sat, 0, 1))
        return -fw

    def _water_contact(self, oldv):
        r = np.linalg.norm(self.wx[:, :2], axis=1)
        rw, slope, active = self._profile(self.wx[:, 2])
        hit = active & (r > rw) & (r > 1e-10)
        if hit.any():
            ids = np.flatnonzero(hit)
            er = self.wx[ids, :2] / r[ids, None]
            norm = np.sqrt(1 + slope[ids]**2)
            nxy, nz = er / norm[:, None], -slope[ids] / norm
            pen = r[ids] - rw[ids]
            self.wx[ids, :2] -= (pen / norm)[:, None] * nxy
            self.wx[ids, 2] -= (pen / norm) * nz
            vn = (self.wv[ids, :2] * nxy).sum(1) + self.wv[ids, 2] * nz
            inbound = vn > 0
            if inbound.any():
                ii, delta = ids[inbound], -1.05 * vn[inbound]
                self.wv[ii, :2] += delta[:, None] * nxy[inbound]
                self.wv[ii, 2] += delta * nz[inbound]
        r = np.linalg.norm(self.wx[:, :2], axis=1)
        floor = (r <= 1.35) & (self.wx[:, 2] < .06)
        if floor.any():
            self.wx[floor, 2] = .06
            down = np.flatnonzero(floor & (self.wv[:, 2] < 0))
            self.wv[down, 2] *= -.08
            self.wv[floor, :2] *= .82
        return (self.wm[:, None] * (self.wv - oldv)).sum(0)

    def _step(self, dt):
        self.wv += dt * self.g
        fcouple = self._porous_drag(dt)
        fcontact, fqr, fqz = self._plug_contact()
        fpg = self.pm_total * self.g
        self.cv += dt * (fpg + fcontact + fcouple) / self.pm_total
        self.cv /= 1 + .10 * dt
        speed = np.linalg.norm(self.cv)
        if speed > 35:
            self.cv *= 35 / speed
        self.c += dt * self.cv

        # Saturation-dependent Kelvin--Voigt elasticity, with volume coupling.
        ratio = self.wetfactor + (1 - self.wetfactor) * (1 - self.sat)**3
        E = self.E0 * ratio
        kr, kz = .25 * E * self.vol0, .16 * E * self.vol0
        cr = 1.1 * np.sqrt(max(kr * self.mqr, 1e-12))
        cz = 1.1 * np.sqrt(max(kz * self.mqz, 1e-12))
        target_z = np.clip(1 / max(self.qr**2, .20), .72, 2.25)
        target_r = np.clip(1 / np.sqrt(max(self.qz, .25)), .52, 1.12)
        self.qrv = (self.qrv + dt * (fqr - kr * (self.qr - target_r)) / self.mqr) \
                   / (1 + dt * cr / self.mqr)
        self.qzv = (self.qzv + dt * (fqz - kz * (self.qz - target_z)) / self.mqz) \
                   / (1 + dt * cz / self.mqz)
        self.qr += dt * self.qrv
        self.qz += dt * self.qzv
        if self.qr < .50: self.qr, self.qrv = .50, max(self.qrv, 0)
        if self.qr > 1.16: self.qr, self.qrv = 1.16, min(self.qrv, 0)
        if self.qz < .68: self.qz, self.qzv = .68, max(self.qzv, 0)
        if self.qz > 2.30: self.qz, self.qzv = 2.30, min(self.qzv, 0)

        oldv = self.wv.copy()
        self.wx += dt * self.wv
        fw_contact = self._water_contact(oldv) / dt
        fwg = self.wm.sum() * self.g
        b = np.zeros((3, 3, 3))
        b[0, 0], b[0, 1] = fpg + fwg, fcontact + fw_contact
        b[1, 0], b[1, 1], b[1, 2] = fpg, fcontact, fcouple
        b[2, 0], b[2, 1], b[2, 2] = fwg, fw_contact, -fcouple
        self.budget_force = b
        self.impulse += dt * b.sum(1)
        self.interaction[0], self.interaction[1] = fcouple, -fcouple
        self.time += dt
        self.steps += 1

    def _frame(self, t):
        px, pv = self._plug_state()
        pos, vel = np.vstack((px, self.wx)), np.vstack((pv, self.wv))
        saturation = np.r_[np.full(self.np, self.sat), np.ones(len(self.wx))]
        porosity = np.r_[np.full(self.np, self.phi), np.zeros(len(self.wx))]
        phase = np.r_[np.zeros(self.np), np.ones(len(self.wx))]
        F = np.tile(np.eye(3), (self.n, 1, 1))
        F[:self.np, 0, 0] = F[:self.np, 1, 1] = self.qr
        F[:self.np, 2, 2] = self.qz
        mq = np.zeros((self.n, 2))
        mq[:self.np, 0], mq[self.np:, 1] = self.pm, self.wm
        E = self.E0 * (self.wetfactor + (1 - self.wetfactor) * (1 - self.sat)**3)
        frame = StateFrame(
            time=float(t), position=pos, velocity=vel, mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={"phase_id": phase, "saturation": saturation,
                        "porosity": porosity, "deformation_gradient": F,
                        "mass_quantity": mq},
            fields={
                "external_force": self.mass[:, None] * self.g,
                "budget_cumulative_external_impulse": self.impulse.copy(),
                "budget_cumulative_inflow_momentum": self.zero_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.zero_momentum.copy(),
                "budget_domain_weights": self.weights.copy(),
                "budget_external_force": self.budget_force.copy(),
                "cumulative_mass_source": np.zeros(2),
                "interaction_force": self.interaction.copy()},
            scalars={"reference_momentum": self.p_ref,
                     "dry_young_modulus": self.E0,
                     "wet_softening_factor": self.wetfactor,
                     "effective_young_modulus": float(E),
                     "permeability": self.perm,
                     "throat_radius": float(self.cfg["funnel"]["throat_radius"])})
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-10:
            raise ValueError("advance_to requires finite nondecreasing times")
        while self.time < target - 1e-12:
            self._step(min(.001, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"method": "particle liquid / affine Kelvin-Voigt porous solid",
                "steps": self.steps, "saturation": self.sat,
                "radial_stretch": self.qr, "axial_stretch": self.qz,
                "plug_center_z": float(self.c[2])}
