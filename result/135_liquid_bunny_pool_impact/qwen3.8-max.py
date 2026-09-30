"""Position-Based-Fluids solver for the liquid bunny pool impact scene.

Physics: incompressible particle fluid (Macklin & Li 2013, "Position Based
Fluids").  The liquid bunny (decimated surface samples of the mesh asset with
physically consistent per-particle mass) falls under gravity onto a still pool
inside a rigid basin, impacts, splashes, merges, and relaxes into damped waves.

Each particle keeps a frozen rest density evaluated from the initial
configuration, so the free-falling bunny preserves its silhouette (its density
constraint stays satisfied) while the pool supports itself hydrostatically.
"""
from __future__ import annotations

import numpy as np

try:  # optional, for a true signed-distance level set
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:  # pragma: no cover
    _edt = None

from worldsolver.contracts import StateFrame

_H = 0.28            # kernel radius (2x pool particle spacing)
_SUB_DT = 1.0 / 60.0
_ITERS = 3
_XSPH = 0.08
_V_MAX = 12.0
_DP_MAX_FRAC = 0.05  # per-iteration position correction cap (fraction of h)


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], float)
        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        density = float(sampling["density"])
        shape = tuple(int(v) for v in sampling["field_grid_shape"])

        frame0 = scene.initial_state()
        pos0 = np.array(frame0.position, float)
        mass0 = np.array(frame0.mass, float)
        lab0 = np.asarray(frame0.attributes["material_label"]).astype(np.int64)

        pool = lab0 == 0
        bunny = lab0 == 1
        # Set the pool at rest directly on the basin floor (scene block starts
        # one half-layer above z=0; remove the gap so it does not free-fall).
        pos0[pool, 2] -= (pos0[pool, 2].min() - 0.5 * self.spacing)
        # Decimate the mesh-vertex bunny shell (geometric simplification is
        # allowed) to a resolvable particle coating and give it a physically
        # consistent liquid mass (~0.7 m^3 of water in the bunny volume).
        b_idx = np.flatnonzero(bunny)
        b_keep = b_idx[::8]
        keep = np.concatenate([np.flatnonzero(pool), b_keep])
        keep.sort()
        self.labels = lab0[keep].copy()
        self.pos = pos0[keep].copy()
        self.mass = mass0[keep].copy()
        is_b = self.labels == 1
        n_b = int(is_b.sum())
        bunny_liquid_volume = 0.7
        self.mass[is_b] = density * bunny_liquid_volume / max(n_b, 1)
        self.vel = np.zeros_like(self.pos)
        self.n = len(self.pos)

        # Frozen diagnostics / normalisation from the *initial* configuration.
        self.char_len = float(max(np.ptp(pos0, axis=0).max(), self.spacing))
        self.ref_vel = float(np.sqrt(2.0 * abs(self.gravity[2]) *
                                     max(self.pos[is_b, 2].min() - self.pos[pool, 2].max(), 0.2)))
        self.t = 0.0

        # Rest densities frozen at t=0 (per-particle; handles free surfaces).
        pi, pj, r = self._pairs(self.pos)
        self.rho0 = self._density_of(pi, pj, r)
        # Frozen relaxation epsilon per particle from the initial gradient norm.
        denom0, _ = self._denom_and_lambda(pi, pj, r, self.rho0, self.rho0)
        self.eps = np.maximum(0.01 * denom0, 1e-18)

        # Diagnostic grid (uniform spacing, covers basin and splash envelope).
        self.grid_shape = shape
        self.grid_dx = self.spacing
        self.grid_origin = np.array([
            -0.5 * shape[0] * self.grid_dx,
            -0.5 * shape[1] * self.grid_dx,
            0.0])

        # Boundary wall samples (inner faces of ground + basin walls).
        self._build_boundary_samples()

        # Static per-frame arrays.
        self.deform = np.repeat(np.eye(3)[None], self.n, axis=0)
        self.mass_quantity = np.zeros((self.n, 2))
        self.mass_quantity[np.arange(self.n), self.labels] = self.mass
        self.particle_radius = np.full(self.n, 0.45 * self.spacing)
        self.cum_source = np.zeros(2)
        self.dp_max = _DP_MAX_FRAC * _H

        # Poly6 / spiky kernel constants.
        h = _H
        self.poly6 = 315.0 / (64.0 * np.pi * h ** 9)
        self.spiky_grad = 45.0 / (np.pi * h ** 6)
        self.h = h
        self.h2 = h * h

    # ------------------------------------------------------------- neighbors
    def _pairs(self, pos):
        """Grid-hash neighbour search; returns (i_idx, j_idx, r_vec)."""
        h = self.h
        inv = 1.0 / h
        ci = np.floor(pos * inv).astype(np.int64)
        ci -= ci.min(axis=0)
        dims = ci.max(axis=0) + 4
        keys = (ci[:, 0] * dims[1] + ci[:, 1]) * dims[2] + ci[:, 2]
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        shifts = np.array([(a, b, c)
                           for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
                          dtype=np.int64)
        q = (((ci[:, None, 0] + shifts[None, :, 0]) * dims[1]
              + (ci[:, None, 1] + shifts[None, :, 1])) * dims[2]
             + (ci[:, None, 2] + shifts[None, :, 2]))
        q = np.maximum(q.ravel(), 0)
        lo = np.searchsorted(skeys, q, "left")
        hi = np.searchsorted(skeys, q, "right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            e = np.zeros(0, dtype=np.int64)
            return e, e, np.zeros((0, 3))
        src = np.repeat(np.arange(len(q)), cnt)
        base = np.repeat(lo, cnt)
        within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pi = src // 27
        pj = order[base + within]
        d = pos[pi] - pos[pj]
        r2 = np.einsum("ij,ij->i", d, d)
        m = (r2 > 1e-12) & (r2 < self.h2)
        return pi[m], pj[m], d[m]

    # ------------------------------------------------------- PBF core pieces
    def _density_of(self, pi, pj, rvec):
        r2 = np.einsum("ij,ij->i", rvec, rvec)
        w = self.poly6 * np.maximum(self.h2 - r2, 0.0) ** 3
        rho = self.mass + np.bincount(pi, weights=w * self.mass[pj], minlength=self.n)
        return rho

    def _denom_and_lambda(self, pi, pj, rvec, rho, rho0):
        r2 = np.einsum("ij,ij->i", rvec, rvec)
        r = np.sqrt(r2)
        coef = self.spiky_grad * np.maximum(self.h - r, 0.0) ** 2 / np.maximum(r, 1e-9)
        # A_ij = m_j * coef * (p_i - p_j)  ( = -rho0 * grad_{p_i} W contribution )
        A = (self.mass[pj] * coef)[:, None] * rvec
        grad_i = np.zeros((self.n, 3))
        np.add.at(grad_i, pi, -A)  # sum_j grad_{p_i} W_ij * m_j (up to 1/rho0)
        denom = (np.bincount(pi, weights=np.einsum("ij,ij->i", A, A), minlength=self.n)
                 + np.einsum("ij,ij->i", grad_i, grad_i)) / (rho0 ** 2)
        C = rho / rho0 - 1.0
        lam = np.maximum(-C, 0.0) / (denom + self.eps)
        return denom, lam

    def _boundary_clamp(self, p):
        wall_top = 0.64
        r = 0.02
        p[:, 2] = np.clip(p[:, 2], r, 3.6)
        inside = p[:, 2] < wall_top
        p[inside, 0] = np.clip(p[inside, 0], -1.77 + r, 1.77 - r)
        p[inside, 1] = np.clip(p[inside, 1], -1.35 + r, 0.84 - r)
        p[:, 0] = np.clip(p[:, 0], -3.15, 3.15)
        p[:, 1] = np.clip(p[:, 1], -1.38, 1.38)
        return p

    # ------------------------------------------------------------------ step
    def _step(self, dt):
        self.vel += self.gravity * dt
        pred = self._boundary_clamp(self.pos + self.vel * dt)  # fresh array, safe
        pi, pj, d = self._pairs(pred)
        if len(pi):
            for _ in range(_ITERS):
                rho = self._density_of(pi, pj, d)
                _, lam = self._denom_and_lambda(pi, pj, d, rho, self.rho0)
                r = np.sqrt(np.einsum("ij,ij->i", d, d))
                dp = np.zeros((self.n, 3))
                wvec = ((lam[pi] + lam[pj]) / self.rho0[pi])[:, None] * \
                    (self.spiky_grad * np.maximum(self.h - r, 0.0) ** 2)[:, None] * d
                np.add.at(dp, pi, wvec)
                norm = np.linalg.norm(dp, axis=1)
                over = norm > self.dp_max
                dp[over] *= (self.dp_max / norm[over])[:, None]
                pred = self._boundary_clamp(pred + dp)
                d = pred[pi] - pred[pj]
            # velocity update + XSPH viscous damping (damped waves afterwards)
            self.vel = (pred - self.pos) / dt
            r2 = np.einsum("ij,ij->i", d, d)
            w = self.poly6 * np.maximum(self.h2 - r2, 0.0) ** 3
            rho = self._density_of(pi, pj, d)
            dv = np.zeros((self.n, 3))
            wj = w * self.mass[pj] / np.maximum(rho[pj], 1e-9)
            np.add.at(dv, pi, wj[:, None] * (self.vel[pj] - self.vel[pi]))
            self.vel += _XSPH * dv
        else:
            self.vel = (pred - self.pos) / dt
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > _V_MAX
        self.vel[over] *= (_V_MAX / speed[over])[:, None]
        self.pos = pred

    # -------------------------------------------------------------- outputs
    def _build_boundary_samples(self):
        pts, nrm = [], []
        xs = np.arange(-1.8, 1.81, 0.3)
        ys = np.arange(-1.2, 0.85, 0.3)
        g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        pts.append(np.column_stack([g[:, 0], g[:, 1], np.zeros(len(g))]))
        nrm.append(np.tile([0.0, 0.0, 1.0], (len(g), 1)))
        wx = np.arange(-1.7, 1.71, 0.34)
        wz = np.array([0.1, 0.35, 0.6])
        g = np.stack(np.meshgrid(wx, wz, indexing="ij"), -1).reshape(-1, 2)
        pts.append(np.column_stack([g[:, 0], np.full(len(g), 0.84), g[:, 1]]))
        nrm.append(np.tile([0.0, -1.0, 0.0], (len(g), 1)))
        wy = np.arange(-0.8, 0.81, 0.4)
        g = np.stack(np.meshgrid(wy, wz, indexing="ij"), -1).reshape(-1, 2)
        pts.append(np.column_stack([np.full(len(g), -1.77), g[:, 0], g[:, 1]]))
        nrm.append(np.tile([1.0, 0.0, 0.0], (len(g), 1)))
        pts.append(np.column_stack([np.full(len(g), 1.77), g[:, 0], g[:, 1]]))
        nrm.append(np.tile([-1.0, 0.0, 0.0], (len(g), 1)))
        self.bnd_pos = np.concatenate(pts)
        self.bnd_nrm = np.concatenate(nrm)
        self.bnd_target = np.zeros_like(self.bnd_pos)

    def _grid_fields(self):
        shape, dx = self.grid_shape, self.grid_dx
        idx = np.floor((self.pos - self.grid_origin) / dx).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        ii = idx[inside]
        flat = (ii[:, 0] * shape[1] + ii[:, 1]) * shape[2] + ii[:, 2]
        ncell = int(np.prod(shape))
        cnt = np.bincount(flat, minlength=ncell).reshape(shape)
        mom = np.zeros((ncell, 3))
        mw = self.mass[inside]
        np.add.at(mom, flat, mw[:, None] * self.vel[inside])
        msum = np.bincount(flat, weights=mw, minlength=ncell).reshape(shape)
        vfield = np.zeros(shape + (3,))
        occ = cnt > 0
        vfield[occ] = mom.reshape(shape + (3,))[occ] / msum.reshape(shape)[occ, None]
        vol = self.mass / 1000.0
        vols = np.bincount(flat, weights=vol[inside], minlength=ncell).reshape(shape)
        vf = vols / dx ** 3
        active = (vf > 0.08).astype(float)
        if _edt is not None:
            d_out = _edt(~occ, sampling=dx)
            d_in = _edt(occ, sampling=dx)
            ls = np.where(occ, -d_in, d_out)
        else:
            ls = dx * (0.5 - np.clip(vf, 0.0, 1.0))
        return vfield, ls, active

    def _boundary_velocity(self):
        d = self.bnd_pos[:, None, :] - self.pos[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        w = np.maximum(0.09 - r2, 0.0) ** 2 * self.mass[None, :]
        s = w.sum(axis=1)
        bv = np.zeros_like(self.bnd_pos)
        ok = s > 1e-12
        bv[ok] = (w[ok] @ self.vel) / s[ok, None]
        return bv

    def _make_frame(self, t):
        vfield, ls, active = self._grid_fields()
        return StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.particle_radius,
                "material_label": self.labels,
                "deformation_gradient": self.deform,
            },
            fields={
                "velocity_field": vfield,
                "liquid_level_set": ls,
                "active_mask": active,
                "boundary_normal": self.bnd_nrm,
                "boundary_target_velocity": self.bnd_target,
                "boundary_velocity": self._boundary_velocity(),
                "cumulative_mass_source": self.cum_source,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.grid_dx,
                "characteristic_length": self.char_len,
                "reference_velocity": self.ref_vel,
            },
        )

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        while self.t < t - 1e-9:
            dt = min(_SUB_DT, t - self.t)
            self._step(dt)
            self.t += dt
        frame = self._make_frame(t)
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "particles": int(self.n),
            "method": "position_based_fluids",
            "kernel_radius": _H,
            "sub_dt": _SUB_DT,
            "time": float(self.t),
        }
