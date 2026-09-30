"""Viscoelastic pool splash & rebound solver.

Particle-based viscoelastic fluid following the Clavet et al. (2005)
position-based formulation of Goktepin et al. (2004) viscoelastic fluids:
double-density relaxation (pressure + near-pressure), plastic springs for
elastic memory / rebound, and pairwise radial viscosity impulses.  Rigid
basin/ground collisions are handled with box signed-distance projection.
Per-particle deformation gradients are integrated from a least-squares
velocity gradient.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # Frozen static collision boxes (center, half_extents) from scene_geometry.
    BOXES = (
        ((0.0, 0.0, -0.05), (3.2, 1.4, 0.05)),   # ground
        ((0.0, 0.72, 0.30), (1.45, 0.08, 0.34)),  # basin back
        ((-1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),  # basin left
        ((1.45, 0.0, 0.30), (0.08, 0.72, 0.34)),   # basin right
    )

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        f0 = scene.initial_state()
        self.spacing = float(cfg["sampling"]["spacing"])
        self.sample_dt = float(cfg["sample_dt"])
        self.dt = self.sample_dt / 8.0
        self.g = np.asarray(cfg["gravity"], dtype=np.float64)
        self.seed = int(seed)

        self.x = np.array(f0.position, dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(f0.mass, dtype=np.float64)
        self.body_id = np.array(f0.body_id)
        self.N = len(self.x)
        self.t = 0.0
        self.nsteps = 0

        h = self.h = 2.15 * self.spacing
        # material / numerical parameters
        self.k = 400.0            # pressure stiffness
        self.kn = 200.0           # near-pressure stiffness
        self.Pcoh = 5.0           # tension clamp (cohesion) on pressure
        self.k_spring = 800.0     # elastic spring stiffness
        self.gamma_plastic = 2.0  # plastic flow rate (1/s)
        self.sigma = 3.0          # linear viscosity (approach)
        self.beta = 8.0           # quadratic viscosity (approach)
        self.sigma_sep = 3.0      # elongational viscosity (separation)
        self.pair_cap = 0.04 * h  # max per-pair displacement / substep
        self.part_cap = 0.08 * h  # max per-particle stage displacement
        self.v_max = 30.0
        self.margin = 0.01

        self.pi, self.pj = np.triu_indices(self.N, 1)
        self.F = np.repeat(np.eye(3)[None], self.N, axis=0)
        self.J_bound = np.zeros((self.N, 3))
        self.last_out_t = 0.0

        # spring storage (sorted by key = i*N + j)
        self.sp_i = np.zeros(0, np.int64)
        self.sp_j = np.zeros(0, np.int64)
        self.sp_key = np.zeros(0, np.int64)
        self.sp_r0 = np.zeros(0, np.float64)

        self._lo = np.array([-3.15, -1.35, 0.005])
        self._hi = np.array([3.15, 1.35, 3.75])
        self._boxes = [(np.array(c), np.array(e)) for c, e in self.BOXES]
        self._radius = np.full(self.N, 0.45 * self.spacing)
        self._char_len = float(f0.scalars.get("characteristic_length", 3.0))

        # calibrate rest densities from the initial configuration
        rho, rhon, _ = self._density(self.x)
        self.rho0 = float(np.percentile(rho, 85))
        self.rho_n0 = float(np.percentile(rhon, 85))

    # ------------------------------------------------------------- helpers
    def _pairs(self, x):
        dx = x[self.pi] - x[self.pj]
        r2 = np.einsum("ij,ij->i", dx, dx)
        m = r2 < self.h * self.h
        si = self.pi[m]
        sj = self.pj[m]
        d = dx[m]
        r = np.sqrt(np.maximum(r2[m], 1e-12))
        w = 1.0 - r / self.h
        return si, sj, d, r, w

    def _density(self, x):
        si, sj, d, r, w = self._pairs(x)
        w2 = w * w
        w3 = w2 * w
        rho = np.bincount(si, w2, self.N) + np.bincount(sj, w2, self.N)
        rhon = np.bincount(sj, w3, self.N) + np.bincount(si, w3, self.N)
        return rho, rhon, (si, sj, d, r, w)

    def _scatter3(self, i, j, C):
        """Accumulate +C at i and -C at j (antisymmetric pair quantity)."""
        out = np.zeros((self.N, 3))
        idx = np.concatenate((i, j))
        vals = np.concatenate((C, -C))
        for a in range(3):
            out[:, a] = np.bincount(idx, weights=vals[:, a], minlength=self.N)
        return out

    @staticmethod
    def _box_sdf(p, c, e):
        d = p - c
        q = np.abs(d) - e
        qc = np.maximum(q, 0.0)
        out = np.linalg.norm(qc, axis=1)
        ins = np.max(q, axis=1)
        sdf = out + np.minimum(ins, 0.0)
        sgn = np.sign(d)
        g_out = qc * sgn
        g_out /= np.maximum(np.linalg.norm(g_out, axis=1, keepdims=True), 1e-9)
        k = np.argmax(q, axis=1)
        rows = np.arange(len(p))
        g_in = np.zeros_like(p)
        g_in[rows, k] = sgn[rows, k]
        grad = np.where(out[:, None] > 1e-9, g_out, g_in)
        return sdf, grad

    def _contact_sdf(self, x):
        sdf = np.full(len(x), np.inf)
        for c, e in self._boxes:
            s, _ = self._box_sdf(x, c, e)
            sdf = np.minimum(sdf, s)
        return sdf

    # ------------------------------------------------------------- substep
    def _substep(self):
        dt = self.dt
        N = self.N

        # 1) gravity
        self.v += self.g * dt

        # neighbor list (reused for all interaction stages this substep)
        si, sj, d, r, w = self._pairs(self.x)
        rhat = d / r[:, None]

        # 2) viscosity impulses (radial)
        if len(si):
            u = np.einsum("ij,ij->i", self.v[si] - self.v[sj], rhat)
            I = np.where(
                u < 0.0,
                dt * w * (self.sigma * (-u) + self.beta * u * u),
                -dt * w * self.sigma_sep * u,
            )
            I = np.minimum(I, np.abs(u))  # never reverse radial motion
            C = (0.5 * I)[:, None] * rhat
            self.v += self._scatter3(si, sj, C)

        # 3) advect
        xp = self.x.copy()
        self.x = xp + self.v * dt

        disp = np.zeros((N, 3))

        # 4) viscoelastic springs (create / flow / relax / break)
        if len(si):
            keys = si.astype(np.int64) * N + sj
            if len(self.sp_key):
                pos = np.clip(np.searchsorted(self.sp_key, keys), 0, len(self.sp_key) - 1)
                exists = self.sp_key[pos] == keys
            else:
                exists = np.zeros(len(keys), dtype=bool)
            new = ~exists
            if new.any():
                self.sp_i = np.concatenate((self.sp_i, si[new]))
                self.sp_j = np.concatenate((self.sp_j, sj[new]))
                self.sp_r0 = np.concatenate((self.sp_r0, r[new]))
                self.sp_key = np.concatenate((self.sp_key, keys[new]))
                o = np.argsort(self.sp_key)
                self.sp_i, self.sp_j = self.sp_i[o], self.sp_j[o]
                self.sp_key, self.sp_r0 = self.sp_key[o], self.sp_r0[o]

        if len(self.sp_key):
            dxs = self.x[self.sp_i] - self.x[self.sp_j]
            rs = np.sqrt(np.maximum(np.einsum("ij,ij->i", dxs, dxs), 1e-12))
            # plastic flow of rest length toward current length
            self.sp_r0 += dt * self.gamma_plastic * (rs - self.sp_r0)
            keep = rs <= self.h
            if not keep.all():
                self.sp_i, self.sp_j = self.sp_i[keep], self.sp_j[keep]
                self.sp_key, self.sp_r0 = self.sp_key[keep], self.sp_r0[keep]
                dxs, rs = dxs[keep], rs[keep]
            if len(rs):
                rhs = dxs / rs[:, None]
                ws = 1.0 - rs / self.h
                Ds = dt * dt * self.k_spring * ws * (rs - self.sp_r0)
                Ds2 = np.clip(0.5 * Ds, -self.pair_cap, self.pair_cap)
                disp += self._scatter3(self.sp_i, self.sp_j, -(Ds2[:, None] * rhs))

        # 5) double density relaxation
        if len(si):
            w2 = w * w
            w3 = w2 * w
            rho = np.bincount(si, w2, N) + np.bincount(sj, w2, N)
            rhon = np.bincount(si, w3, N) + np.bincount(sj, w3, N)
            P = np.maximum(self.k * (rho - self.rho0), -self.Pcoh)
            Pn = np.maximum(self.kn * (rhon - self.rho_n0), 0.0)
            Pbar = 0.5 * (P[si] + P[sj])
            Pnbar = 0.5 * (Pn[si] + Pn[sj])
            D = dt * dt * (Pbar * w + Pnbar * w2)
            Dh = np.clip(0.5 * D, -self.pair_cap, self.pair_cap)
            disp += self._scatter3(si, sj, Dh[:, None] * rhat)

        # per-particle displacement cap
        dn = np.linalg.norm(disp, axis=1)
        over = dn > self.part_cap
        if over.any():
            disp[over] *= (self.part_cap / dn[over])[:, None]
        self.x += disp

        # 6) rigid boundary projection
        delta = np.zeros((N, 3))
        for c, e in self._boxes:
            sdf, grad = self._box_sdf(self.x, c, e)
            pen = self.margin - sdf
            act = pen > 0.0
            if act.any():
                delta[act] += pen[act, None] * grad[act]
        xc = np.clip(self.x, self._lo, self._hi)
        delta += xc - self.x
        if np.any(delta):
            self.x += delta
            self.J_bound += self.mass[:, None] * (delta / dt)

        # 7) velocity from position change
        v = (self.x - xp) / dt
        sp = np.linalg.norm(v, axis=1)
        over = sp > self.v_max
        if over.any():
            v[over] *= (self.v_max / sp[over])[:, None]
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            self.x = xp
        self.v = v

        # 8) deformation gradient from least-squares velocity gradient
        if len(si):
            dv = self.v[sj] - self.v[si]
            # e = x_j - x_i = -d ;  B = sum w dv e^T ; A = sum w e e^T
            outer_de = d[:, :, None] * d[:, None, :] * w[:, None, None]
            outer_dv = -(dv[:, :, None] * d[:, None, :]) * w[:, None, None]
            A = np.zeros((N, 9))
            B = np.zeros((N, 9))
            idx = np.concatenate((si, sj))
            for a in range(9):
                ca = outer_de[:, a]
                cb = outer_dv[:, a]
                A[:, a] = np.bincount(idx, weights=np.concatenate((ca, ca)), minlength=N)
                B[:, a] = np.bincount(idx, weights=np.concatenate((cb, cb)), minlength=N)
            A = A.reshape(N, 3, 3)
            B = B.reshape(N, 3, 3)
            A[:, 0, 0] += 1e-3
            A[:, 1, 1] += 1e-3
            A[:, 2, 2] += 1e-3
            L = np.linalg.solve(A, B.transpose(0, 2, 1)).transpose(0, 2, 1)
            self.F = self.F + dt * np.einsum("nij,njk->nik", L, self.F)
            np.clip(self.F, -10.0, 10.0, out=self.F)
            det = np.linalg.det(self.F)
            bad = ~np.isfinite(det) | (det <= 1e-4)
            if bad.any():
                self.F[bad] = np.eye(3)

        self.t += dt
        self.nsteps += 1

    # --------------------------------------------------------------- frame
    def _frame(self, time: float) -> StateFrame:
        sdf = self._contact_sdf(self.x)
        fext = self.mass[:, None] * self.g[None, :]
        elapsed = time - self.last_out_t
        if elapsed > 1e-12:
            fext = fext + self.J_bound / elapsed
        self.J_bound[:] = 0.0
        self.last_out_t = time
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self._radius.copy(),
                "material_label": self.body_id.copy(),
                "deformation_gradient": self.F.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "contact_signed_distance": sdf,
                "external_force": fext,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": self._char_len,
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ interface
    def advance_to(self, time: float):
        time = float(time)
        n = int(round((time - self.t) / self.dt))
        for _ in range(max(n, 0)):
            self._substep()
        if time >= self.t:
            self.t = time
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "substeps": self.nsteps,
            "n_springs": int(len(self.sp_key)),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self.N else 0.0,
            "kinetic_energy": float(0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.v, self.v))),
        }
