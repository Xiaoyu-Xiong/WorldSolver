"""MPM Drucker-Prager sand with rigid ball impact (KGPSTJ16 Fig.2 style).

Particles: sand bed (body_id 0) simulated with an explicit MPM solver using a
Drucker-Prager elastoplastic return map in log-strain space; rigid ball
(body_id 1) integrated analytically and coupled to the MPM grid through
node-level rigid-body collision impulses.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = scene.config
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)

        pos = np.asarray(frame0.position, float).copy()
        mass = np.asarray(frame0.mass, float).copy()
        body = np.asarray(frame0.body_id).astype(np.int64)
        self.mass = mass
        self.body = body
        self.n_total = len(pos)
        self.sand_idx = np.flatnonzero(body == 0)
        ball_idx = int(np.flatnonzero(body != 0)[0])
        self.ball_idx = ball_idx

        self.radius = np.asarray(frame0.attributes["particle_radius"], float).copy()
        self.R_ball = 0.54
        self.m_ball = float(mass[ball_idx])
        self.ball_x = pos[ball_idx].copy()
        self.ball_v = np.zeros(3)

        self.scalars0 = {
            "contact_length_scale": float(frame0.scalars["contact_length_scale"]),
            "grid_spacing": float(frame0.scalars["grid_spacing"]),
            "characteristic_length": float(frame0.scalars["characteristic_length"]),
            "reference_velocity": 1.0,
        }

        density = float(cfg["sampling"]["density"])
        P = len(self.sand_idx)
        self.xp = pos[self.sand_idx].copy()
        self.vp = np.zeros((P, 3))
        self.F = np.repeat(np.eye(3)[None], P, axis=0)
        self.sigma = np.zeros((P, 3, 3))
        self.C = np.zeros((P, 3, 3))
        self.mp = mass[self.sand_idx].copy()
        self.V0 = self.mp / density

        # --- constitutive (Drucker-Prager sand) ---
        E, nu = 1.0e6, 0.2
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        phi = np.deg2rad(35.0)
        M = np.sqrt(2.0 / 3.0) * 2.0 * np.sin(phi) / (3.0 - np.sin(phi))
        K = self.lam + 2.0 * self.mu / 3.0
        self.alpha_q = M * K / (2.0 * self.mu)  # eps_q <= alpha_q * (-tr)

        # --- grid ---
        self.dx = 0.06
        self.origin = np.array([-1.6, -1.2, 0.0])
        self.NX, self.NY, self.NZ = 55, 41, 61
        n_nodes = self.NX * self.NY * self.NZ
        self.grid_m = np.zeros(n_nodes)
        self.grid_mom = np.zeros((n_nodes, 3))
        self.grid_v = np.zeros((n_nodes, 3))
        ix, iy, iz = np.meshgrid(
            np.arange(self.NX), np.arange(self.NY), np.arange(self.NZ), indexing="ij")
        self.node_xyz = (np.stack((ix, iy, iz), -1).reshape(-1, 3) * self.dx
                         + self.origin[None, :])
        self.node_iz = iz.ravel()

        jj = np.array([[a, b, c] for c in range(3) for b in range(3) for a in range(3)])
        self.jj = jj
        self.j0, self.j1, self.j2 = jj[:, 0], jj[:, 1], jj[:, 2]

        self.t = 0.0
        self.dt_max = 5.0e-4
        self.p_lo = self.origin + 0.02
        self.p_hi = self.origin + np.array([self.NX - 1, self.NY - 1, self.NZ - 1]) * self.dx - 0.02

    # ---------------- MPM substep ----------------
    def _substep(self, dt: float) -> None:
        dx = self.dx
        P = len(self.xp)
        NX, NY = self.NX, self.NY

        # interpolation weights
        g = (self.xp - self.origin[None, :]) / dx
        base = np.floor(g - 0.5).astype(np.int64)
        np.clip(base[:, 0], 0, NX - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, self.NY - 3, out=base[:, 1])
        np.clip(base[:, 2], 0, self.NZ - 3, out=base[:, 2])
        f = g - base
        w1 = 0.5 * (1.5 - f) ** 2
        w2 = 0.75 - (f - 1.0) ** 2
        w3 = 0.5 * (f - 0.5) ** 2
        wx = np.stack((w1[:, 0], w2[:, 0], w3[:, 0]), 1)
        wy = np.stack((w1[:, 1], w2[:, 1], w3[:, 1]), 1)
        wz = np.stack((w1[:, 2], w2[:, 2], w3[:, 2]), 1)
        dwx = np.stack((-(1.5 - f[:, 0]), -2.0 * (f[:, 0] - 1.0), f[:, 0] - 0.5), 1) / dx
        dwy = np.stack((-(1.5 - f[:, 1]), -2.0 * (f[:, 1] - 1.0), f[:, 1] - 0.5), 1) / dx
        dwz = np.stack((-(1.5 - f[:, 2]), -2.0 * (f[:, 2] - 1.0), f[:, 2] - 0.5), 1) / dx

        idx = ((base[:, 2][:, None] + self.j2[None, :]) * NY
               + (base[:, 1][:, None] + self.j1[None, :])) * NX \
            + (base[:, 0][:, None] + self.j0[None, :])
        j0, j1, j2 = self.j0, self.j1, self.j2
        W = wx[:, j0] * wy[:, j1] * wz[:, j2]
        gradW = np.empty((P, 27, 3))
        gradW[:, :, 0] = dwx[:, j0] * wy[:, j1] * wz[:, j2]
        gradW[:, :, 1] = wx[:, j0] * dwy[:, j1] * wz[:, j2]
        gradW[:, :, 2] = wx[:, j0] * wy[:, j1] * dwz[:, j2]
        dxg = self.node_xyz[idx] - self.xp[:, None, :]

        # P2G
        self.grid_m[:] = 0.0
        self.grid_mom[:] = 0.0
        np.add.at(self.grid_m, idx, W * self.mp[:, None])
        Cd = np.einsum("pij,pkj->pik", self.C, dxg)
        forceterm = dt * self.V0[:, None, None] * np.einsum("pij,pkj->pik", self.sigma, gradW)
        mom_c = W[:, :, None] * (self.mp[:, None, None]
                                 * (self.vp[:, None, :] + Cd)) - forceterm
        np.add.at(self.grid_mom, idx, mom_c)

        # grid ops
        v = self.grid_v
        v[:] = 0.0
        act = self.grid_m > 1e-12
        v[act] = self.grid_mom[act] / self.grid_m[act, None]
        v[act] += self.gravity[None, :] * dt

        # ball-grid rigid collision
        dvec = self.node_xyz - self.ball_x[None, :]
        dist2 = np.einsum("ij,ij->i", dvec, dvec)
        sel = act & (dist2 < self.R_ball * self.R_ball) & (self.node_iz > 0)
        if sel.any():
            si = np.flatnonzero(sel)
            d = np.sqrt(dist2[si])
            n = dvec[si] / d[:, None]
            mn = self.grid_m[si]
            vrel = v[si] - self.ball_v[None, :]
            vrn = np.einsum("ij,ij->i", vrel, n)
            app = vrn < 0.0
            if app.any():
                sia = si[app]
                na = n[app]
                vrna = vrn[app]
                mna = mn[app]
                dv = -(vrna[:, None]) * na  # e = 0
                vt = vrel[app] - vrna[:, None] * na
                vtn = np.linalg.norm(vt, axis=1)
                fric = 0.4
                scale = np.minimum(1.0, fric * (-vrna) / np.maximum(vtn, 1e-9))
                dv = dv - scale[:, None] * vt
                v[sia] += dv
                imp = mna[:, None] * dv
                self.ball_v += -imp.sum(0) / self.m_ball

        # floor boundary (iz == 0)
        fl = act & (self.node_iz == 0)
        if fl.any():
            fi = np.flatnonzero(fl)
            vz = v[fi, 2]
            neg = vz < 0.0
            fin = fi[neg]
            v[fin, 2] = 0.0
            v[fin, 0] *= 0.9
            v[fin, 1] *= 0.9

        self.ball_v += self.gravity * dt

        # G2P
        vsel = v[idx]
        self.vp = np.einsum("pk,pka->pa", W, vsel)
        self.C = (4.0 / dx / dx) * np.einsum("pk,pka,pkb->pab", W, vsel, dxg)

        # advect particles
        self.xp += self.vp * dt
        np.clip(self.xp, self.p_lo[None, :], self.p_hi[None, :], out=self.xp)

        # advect ball
        self.ball_x += self.ball_v * dt
        if self.ball_x[2] < self.R_ball:
            self.ball_x[2] = self.R_ball
            if self.ball_v[2] < 0.0:
                self.ball_v[2] = 0.0

        # deformation gradient + Drucker-Prager return map + stress
        self.F = self.F + dt * np.einsum("pij,pjk->pik", self.C, self.F)
        U, s, Vh = np.linalg.svd(self.F)
        s = np.clip(s, 1e-9, 1e9)
        sig = np.log(s)
        tr = sig.sum(1, keepdims=True)
        dev = sig - tr / 3.0
        epsq = np.sqrt(1.5 * np.einsum("pi,pi->p", dev, dev))[:, None]
        yield_q = self.alpha_q * np.maximum(0.0, -tr)
        scale = np.where(epsq > 1e-12,
                         np.minimum(1.0, yield_q / np.maximum(epsq, 1e-12)), 1.0)
        sig_new = np.where(tr > 0.0, 0.0, tr / 3.0 + dev * scale)
        e_new = np.exp(sig_new)
        self.F = np.einsum("pij,pj,pkj->pik", U, e_new, Vh)
        tr_new = sig_new.sum(1, keepdims=True)
        tau_diag = 2.0 * self.mu * sig_new + self.lam * tr_new
        J = np.exp(tr_new)
        self.sigma = np.einsum("pij,pj,pkj->pik", U, tau_diag, U) / J[:, :, None]

        self.t += dt

    # ---------------- driver ----------------
    def advance_to(self, time: float):
        time = float(time)
        span = time - self.t
        if span > 1e-12:
            n = int(np.ceil(span / self.dt_max))
            dt = span / n
            for _ in range(n):
                self._substep(dt)
        return self._make_frame(time)

    def _make_frame(self, t: float) -> StateFrame:
        pos = np.zeros((self.n_total, 3))
        vel = np.zeros((self.n_total, 3))
        pos[self.sand_idx] = self.xp
        vel[self.sand_idx] = self.vp
        pos[self.ball_idx] = self.ball_x
        vel[self.ball_idx] = self.ball_v

        F_full = np.repeat(np.eye(3)[None], self.n_total, axis=0)
        F_full[self.sand_idx] = self.F

        r = self.radius
        gap_floor = pos[:, 2] - r
        d_ball = (np.linalg.norm(pos - self.ball_x[None, :], axis=1)
                  - self.R_ball - r)
        sd = np.minimum(gap_floor, d_ball)
        sd[self.ball_idx] = self.ball_x[2] - self.R_ball

        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            float(t), pos, vel, mass=self.mass.copy(), body_id=self.body.copy(),
            attributes={"mass_quantity": self.mass.copy(),
                        "particle_radius": r.copy(),
                        "material_label": self.body.copy(),
                        "deformation_gradient": F_full},
            fields={"cumulative_mass_source": np.zeros(self.n_total),
                    "contact_signed_distance": sd,
                    "external_force": ext_force},
            scalars=dict(self.scalars0),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"ball_z": float(self.ball_x[2]),
                "ball_vz": float(self.ball_v[2]),
                "sand_min_z": float(self.xp[:, 2].min()),
                "time": float(self.t)}
