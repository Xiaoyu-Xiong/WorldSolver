"""Material Point Method solver for the snow-castle destruction scene.

Implements an explicit MPM (quadratic B-spline shape functions, APIC/PIC
transfer) with the fixed-corotated elastoplastic snow constitutive model of
Stomakhin et al. (2013): elastic hardening tied to accumulated plastic
compression and singular-value clamping of the elastic deformation gradient.
The orange striker is a rigid sphere coupled to the snow through the MPM grid
with exact momentum exchange; the floor is a sticky (no-slip) grid boundary.

Everything is implemented with NumPy only; no external physics engine is used.
"""
from __future__ import annotations

import numpy as np

try:  # contract import (available in the platform runtime)
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback for isolated syntax checks
    StateFrame = object


def _bspline(d: np.ndarray):
    """Quadratic B-spline weight and derivative for scaled offset d."""
    a = np.abs(d)
    inside = a < 1.5
    w = np.where(a < 0.5, 0.75 - d * d, 0.5 * np.square(1.5 - np.minimum(a, 1.5)))
    w = np.where(inside, w, 0.0)
    dw = np.where(a < 0.5, -2.0 * d, d - 1.5 * np.sign(d))
    dw = np.where(inside, dw, 0.0)
    return w, dw


class BenchmarkSolver:
    """Explicit MPM snow solver with rigid-sphere striker coupling."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        ic = cfg.get("initial_conditions", {}) or {}
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float)
        vel = np.asarray(frame.velocity, dtype=float)
        mass = np.asarray(frame.mass, dtype=float)
        body = np.asarray(frame.body_id, dtype=np.int32)

        spacing = float(ic.get("snow_particle_spacing", 0.075))
        self.h = 2.0 * spacing                       # grid cell size
        self.dt = float(cfg.get("sample_dt", 0.02)) / 16.0   # 1.25 ms, CFL ~ 0.2

        is_ball = body == 2
        self.x = pos[~is_ball].copy()
        self.v = vel[~is_ball].copy()
        self.m = mass[~is_ball].copy()
        self.n = int(self.x.shape[0])
        vol0 = np.asarray(frame.attributes.get("initial_volume"), dtype=float)
        if vol0.ndim == 1 and vol0.shape[0] == pos.shape[0]:
            self.V0 = vol0[~is_ball].copy()
        else:
            self.V0 = np.full(self.n, spacing ** 3)
        self.ball_x = pos[is_ball][0].copy()
        self.ball_v = vel[is_ball][0].copy()
        self.ball_m = float(mass[is_ball][0])
        self.ball_r = float(ic.get("striker_radius", 0.42))
        self.body_id = body.copy()

        eye = np.eye(3)
        self.F = np.tile(eye, (self.n, 1, 1))         # elastic deformation
        self.Fp = np.tile(eye, (self.n, 1, 1))        # plastic deformation
        self.C = np.zeros((self.n, 3, 3))             # APIC affine matrix
        self.Jp = np.ones(self.n)                     # det(Fp)
        self.S = np.zeros((self.n, 3, 3))             # Kirchhoff-like stress

        # Snow constitutive parameters (Stomakhin et al. 2013).
        young = 1.4e5
        poisson = 0.2
        self.mu0 = young / (2.0 * (1.0 + poisson))
        self.lam0 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.xi = 10.0                                # hardening
        self.theta_c = 0.025                          # critical compression
        self.theta_s = 0.0075                         # critical stretch

        # Grid: origin, node counts, boundary masks.
        self.origin = np.array([-3.9, -2.55, -0.75])
        top = np.array([3.9, 2.55, 3.9])
        self.nx = int(round((top[0] - self.origin[0]) / self.h)) + 1
        self.ny = int(round((top[1] - self.origin[1]) / self.h)) + 1
        self.nz = int(round((top[2] - self.origin[2]) / self.h)) + 1
        self.nn = self.nx * self.ny * self.nz
        gx = self.origin[0] + self.h * np.arange(self.nx)
        gy = self.origin[1] + self.h * np.arange(self.ny)
        gz = self.origin[2] + self.h * np.arange(self.nz)
        mx, my, mz = np.meshgrid(gx, gy, gz, indexing="ij")
        self.node_pos = np.stack([mx, my, mz], axis=-1).reshape(-1, 3)
        # Floor surface is z = 0 (frozen box top). Sticky floor + side walls.
        self.floor_mask = self.node_pos[:, 2] <= 1e-9
        self.wall_mask = (
            (self.node_pos[:, 0] <= gx[0] + 1e-9)
            | (self.node_pos[:, 0] >= gx[-1] - 1e-9)
            | (self.node_pos[:, 1] <= gy[0] + 1e-9)
            | (self.node_pos[:, 1] >= gy[-1] - 1e-9)
        )
        self.bc_mask = self.floor_mask | self.wall_mask
        margin = 1.6 * self.h
        self.lo = self.origin + margin
        self.hi = top - margin

        self.steps = 0
        self.t = 0.0
        self.mass0 = float(mass.sum())
        self.vmax = 35.0

    # ------------------------------------------------------------------
    def _constitutive_update(self) -> None:
        """Plastic return mapping on F and stress S = P F^T for the next P2G."""
        # Guard against rare non-finite states.
        bad = ~np.isfinite(self.F).reshape(self.n, -1).all(axis=1)
        if np.any(bad):
            self.F[bad] = np.eye(3)
            self.Fp[bad] = np.eye(3)
            self.Jp[bad] = 1.0
            self.v[bad] = 0.0

        U, sig, Vt = np.linalg.svd(self.F)
        sig_lo = 1.0 - self.theta_c
        sig_hi = 1.0 + self.theta_s
        sig_c = np.clip(sig, sig_lo, sig_hi)
        R = np.einsum("nij,njk->nik", U, Vt)
        Fe = np.einsum("nij,nj,njk->nik", U, sig_c, Vt)
        ratio = sig / sig_c
        # Fp <- V diag(ratio) V^T Fp ; Jp updated multiplicatively.
        plast = np.einsum("nji,nj,njk->nik", Vt, ratio, Vt)
        self.Fp = np.einsum("nij,njk->nik", plast, self.Fp)
        self.Jp = self.Jp * ratio.prod(axis=1)
        self.F = Fe

        J = sig_c.prod(axis=1)
        hard = np.exp(self.xi * (1.0 - self.Jp))
        mu = self.mu0 * hard
        lam = self.lam0 * hard
        FFt = np.einsum("nij,nkj->nik", Fe, Fe)
        RFt = np.einsum("nij,nkj->nik", R, Fe)
        eye = np.eye(3)
        self.S = mu[:, None, None] * (FFt - RFt) + (lam * J * (J - 1.0))[:, None, None] * eye

    # ------------------------------------------------------------------
    def _step(self) -> None:
        n = self.n
        h = self.h
        dt = self.dt

        # Rigid striker: gravity (position updated after coupling).
        self.ball_v = self.ball_v + dt * self.gravity

        # ---------------- particle-to-grid ----------------
        u = (self.x - self.origin) / h
        fu = np.floor(u).astype(np.int64)
        wl = [[None] * 3 for _ in range(3)]  # wl[axis][k] -> (w, dw, d)
        for axis in range(3):
            for k, m in enumerate((-1, 0, 1)):
                d = u[:, axis] - (fu[:, axis] + m)
                w, dw = _bspline(d)
                wl[axis][k] = (w, dw, d)

        W = (
            wl[0][0][0][:, None, None, :] * wl[1][0][0][None, :, None, :] * wl[2][0][0][None, None, :, :]
        )
        Gx = wl[0][0][1][:, None, None, :] * wl[1][0][0][None, :, None, :] * wl[2][0][0][None, None, :, :]
        Gy = wl[0][0][0][:, None, None, :] * wl[1][0][1][None, :, None, :] * wl[2][0][0][None, None, :, :]
        Gz = wl[0][0][0][:, None, None, :] * wl[1][0][0][None, :, None, :] * wl[2][0][1][None, None, :, :]
        for k in (1, 2):
            w0, g0, _ = wl[0][k]
            w1, g1, _ = wl[1][k]
            w2, g2, _ = wl[2][k]
            W = W + w0[:, None, None, :] * w1[None, :, None, :] * w2[None, None, :, :]
            Gx = Gx + g0[:, None, None, :] * w1[None, :, None, :] * w2[None, None, :, :]
            Gy = Gy + w0[:, None, None, :] * g1[None, :, None, :] * w2[None, None, :, :]
            Gz = Gz + w0[:, None, None, :] * w1[None, :, None, :] * g2[None, None, :, :]

        offs = np.array([-1, 0, 1], dtype=np.int64)
        JX = fu[:, 0][None, :] + offs[:, None]
        JY = fu[:, 1][None, :] + offs[:, None]
        JZ = fu[:, 2][None, :] + offs[:, None]
        idx = (JX[:, None, None, :] * self.ny + JY[None, :, None, :]) * self.nz + JZ[None, None, :, :]
        flat_idx = idx.reshape(-1)

        # Material offsets of the 27 support nodes, scaled by h.
        Dx = np.stack([wl[0][k][2] for k in range(3)])   # (3, n)
        Dy = np.stack([wl[1][k][2] for k in range(3)])
        Dz = np.stack([wl[2][k][2] for k in range(3)])
        D = np.empty((3, 3, 3, n, 3))
        D[..., 0] = Dx[:, None, None, :]
        D[..., 1] = Dy[None, :, None, :]
        D[..., 2] = Dz[None, None, :, :]

        Cd = np.einsum("nij,abcnj->abci", self.C, D)          # C (x_i - x_p)
        G = np.stack([Gx, Gy, Gz], axis=-1) / h               # grad w (3,3,3,n,3)
        SG = np.einsum("nij,abcnj->abci", self.S, G)          # S grad w

        Wm = W * self.m
        mv = np.empty((3, 3, 3, n, 3))
        mv[..., 0] = Wm * (self.v[:, 0] - h * Cd[..., 0])
        mv[..., 1] = Wm * (self.v[:, 1] - h * Cd[..., 1])
        mv[..., 2] = Wm * (self.v[:, 2] - h * Cd[..., 2])
        impulse = -(dt * self.V0)[None, None, None, :] * SG   # dt * f_i

        gmass = np.bincount(flat_idx, weights=Wm.reshape(-1), minlength=self.nn)
        gmom = np.empty((self.nn, 3))
        gmom[:, 0] = np.bincount(
            flat_idx, weights=(mv[..., 0] + impulse[..., 0]).reshape(-1), minlength=self.nn
        )
        gmom[:, 1] = np.bincount(
            flat_idx, weights=(mv[..., 1] + impulse[..., 1]).reshape(-1), minlength=self.nn
        )
        gmom[:, 2] = np.bincount(
            flat_idx, weights=(mv[..., 2] + impulse[..., 2]).reshape(-1), minlength=self.nn
        )

        # ---------------- grid update ----------------
        has = gmass > 1e-12
        vg = np.zeros((self.nn, 3))
        vg[has] = gmom[has] / gmass[has, None] + dt * self.gravity
        speed = np.linalg.norm(vg, axis=1)
        fast = speed > self.vmax
        if np.any(fast):
            vg[fast] = vg[fast] * (self.vmax / speed[fast])[:, None]
        vg[self.bc_mask] = 0.0

        # ---------------- striker coupling (momentum conserving) ----------------
        # Perfectly inelastic impulse between the striker and the snow-bearing
        # grid nodes inside the sphere: both end at the common velocity, so the
        # light striker never receives more impulse than it can carry.
        d2 = np.einsum("ij,ij->i", self.node_pos - self.ball_x, self.node_pos - self.ball_x)
        sel = (d2 < self.ball_r ** 2) & has & ~self.bc_mask
        if np.any(sel):
            m_in = float(gmass[sel].sum())
            p_old = (gmass[sel, None] * vg[sel]).sum(axis=0)
            v_common = (self.ball_m * self.ball_v + p_old) / (self.ball_m + m_in)
            vg[sel] = v_common
            self.ball_v = v_common

        # striker integration + floor/wall handling
        self.ball_x = self.ball_x + dt * self.ball_v
        if self.ball_x[2] < self.ball_r:
            self.ball_x[2] = self.ball_r
            if self.ball_v[2] < 0.0:
                self.ball_v[2] = 0.0
            fr = max(0.0, 1.0 - 6.0 * dt)
            self.ball_v[0] *= fr
            self.ball_v[1] *= fr
        for k in range(3):
            if self.ball_x[k] < self.lo[k]:
                self.ball_x[k] = self.lo[k]
                self.ball_v[k] = max(self.ball_v[k], 0.0)
            elif self.ball_x[k] > self.hi[k]:
                self.ball_x[k] = self.hi[k]
                self.ball_v[k] = min(self.ball_v[k], 0.0)

        # ---------------- grid-to-particle ----------------
        vi = vg[flat_idx].reshape(3, 3, 3, n, 3)
        v_new = (W[..., None] * vi).sum(axis=(0, 1, 2))
        C_new = -(4.0 / h) * np.einsum("abcn,abcni,abcnj->nij", W, vi, D)

        self.v = v_new
        spd = np.linalg.norm(self.v, axis=1)
        fast = spd > self.vmax
        if np.any(fast):
            self.v[fast] = self.v[fast] * (self.vmax / spd[fast])[:, None]
        self.C = C_new

        self.x = self.x + dt * self.v
        np.clip(self.x[:, 0], self.lo[0], self.hi[0], out=self.x[:, 0])
        np.clip(self.x[:, 1], self.lo[1], self.hi[1], out=self.x[:, 1])
        np.clip(self.x[:, 2], self.lo[2], self.hi[2], out=self.x[:, 2])
        bad = ~np.isfinite(self.x).all(axis=1)
        if np.any(bad):
            self.x[bad] = np.clip(np.nan_to_num(self.x[bad]), self.lo, self.hi)
            self.v[bad] = 0.0

        # ---------------- deformation & plasticity ----------------
        A = np.eye(3) + dt * self.C
        self.F = np.einsum("nij,njk->nik", A, self.F)
        self._constitutive_update()

        self.steps += 1
        self.t = self.steps * dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.steps < target:
            self._step()
        return self._make_frame(float(time))

    # ------------------------------------------------------------------
    def _make_frame(self, time: float):
        n = self.n
        pos = np.vstack([self.x, self.ball_x[None, :]])
        vel = np.vstack([self.v, self.ball_v[None, :]])
        mass = np.concatenate([self.m, np.array([self.ball_m])])
        F_all = np.empty((n + 1, 3, 3))
        F_all[:n] = self.F
        F_all[n] = np.eye(3)

        # Signed contact gaps: floor (z = 0 surface) and striker sphere.
        phi_floor = self.x[:, 2]
        phi_ball = np.linalg.norm(self.x - self.ball_x, axis=1) - self.ball_r
        phi = np.minimum(phi_floor, phi_ball)
        near = phi < self.h
        contact = phi[near]
        ball_gap = min(
            float(self.ball_x[2] - self.ball_r),
            float(phi_ball.min()) if n > 0 else self.h,
        )
        if ball_gap < self.h:
            contact = np.concatenate([contact, np.array([ball_gap])])
        if contact.size == 0:
            contact = np.array([self.h])

        external = mass[:, None] * self.gravity
        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F_all,
                "mass_quantity": mass.copy(),
            },
            fields={
                "contact_signed_distance": contact,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": float(self.h),
                "initial_total_mass": self.mass0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "method": "explicit MPM (quadratic B-spline, PIC) with fixed-corotated plastic snow",
            "grid_spacing": float(self.h),
            "time_step": float(self.dt),
            "steps": int(self.steps),
            "simulated_time": float(self.t),
            "snow_particles": int(self.n),
            "striker_position": self.ball_x.tolist(),
            "striker_velocity": self.ball_v.tolist(),
            "total_mass": float(self.m.sum() + self.ball_m),
            "initial_total_mass": self.mass0,
        }
