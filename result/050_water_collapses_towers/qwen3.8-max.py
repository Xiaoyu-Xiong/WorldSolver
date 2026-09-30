"""Physically-based solver for 050_water_collapses_towers.

Water: weakly-compressible SPH (cubic spline kernel, Tait EOS calibrated to the
initial lattice rest density, pairwise viscosity, XSPH transport smoothing)
advanced with symplectic Euler on a uniform neighbour grid.

Towers: 18 rigid capsule-proxied columns (cylinders/cubes) with full rigid-body
dynamics (translation + quaternion rotation, body-frame inertia), penalty
contacts against the basin floor/walls (with Coulomb friction), segment-segment
penalty contacts between columns, and a momentum-conserving penalty coupling to
the SPH water applied along each capsule surface (reaction pushed back onto the
water particles).

All required raw trajectory fields are exported every sample frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
G = np.array([0.0, 0.0, -9.81])
H = 0.30                     # SPH smoothing length / grid cell size
H2 = H * H
SIGMA64 = 64.0 / (np.pi * H ** 3)
GAMMA_EOS = 7.0
C0 = 28.0                    # artificial sound speed
DT = 0.0025                  # substep
XSPH = 0.2
NU_VISC = 2.0                # pairwise viscosity coefficient
WALL_MARGIN = 0.10
WALL_K = 3.0e6               # per unit mass (acceleration per metre penetration)
WALL_C = 3.0e3               # per unit mass
WALL_AMAX = 6.0e3
WATER_VMAX = 12.0
COL_VMAX = 8.0
COL_WMAX = 15.0
# water <-> column coupling
COUP_BAND = 0.25
COUP_K = 1.5e5
COUP_C = 400.0
COUP_CT = 300.0
COUP_AMAX_W = 6.0e3          # accel clamp on a water particle per column
COUP_AMAX_C = 4.0e3          # accel clamp on a column (total from water)
# column <-> column / floor contacts
RR_K = 2.0e5
RR_C = 1500.0
FL_K = 4.0e5
FL_C = 2500.0
FRICTION = 0.45
CDAMP = 1.5                  # gentle linear damping on columns
CDAMP_ROT = 120.0
CONTACT_BAND_RR = 0.06
CONTACT_BAND_FLOOR = 0.05
CONTACT_BAND_WC = 0.18

# basin interior (from config boundaries)
XMIN, XMAX = -3.95, 3.95
YMIN, YMAX = -1.95, 1.95

# fixed neighbour-grid domain (covers basin plus splash headroom)
G_ORIGIN = np.array([-4.4, -2.4, -0.4])
G_DIMS = np.array([30, 17, 14])
NC = int(np.prod(G_DIMS))

OFF27 = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])


def _qmul(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), axis=1)


class BenchmarkSolver:
    # ------------------------------------------------------------ lifecycle
    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(seed)
        f0 = scene.initial_state()
        self.scene_group = np.asarray(f0.attributes["scene_group"]).astype(float)
        p = np.array(f0.position, dtype=float)
        n = len(p)
        self.n_total = n
        self.wmask = self.scene_group == 10
        self.nw = int(self.wmask.sum())
        self.nc = n - self.nw
        # scene builds water first, then 9 left + 9 right column samples
        self.wp = np.array(p[: self.nw], dtype=float)
        self.wv = np.array(f0.velocity[: self.nw], dtype=float)
        self.wm = np.array(f0.mass[: self.nw], dtype=float)
        self.ccom = np.array(p[self.nw:], dtype=float)
        self.cvel = np.zeros((self.nc, 3))
        self.cquat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nc, 1))
        self.comega = np.zeros((self.nc, 3))
        self.cm = np.array(f0.mass[self.nw:], dtype=float)
        self.cgroup = self.scene_group[self.nw:].astype(int)
        # material index: 0 water, 1 light cylinders, 2 heavy cubes
        self.cmat = np.where((self.cgroup % 10) == 1, 2, 1)
        self.col_body_ids = np.array(f0.body_id[self.nw:], dtype=int)
        # capsule proxy geometry (identical for all columns)
        self.R = 0.115
        self.A = 0.5 * 1.85 - self.R
        # tiny deterministic jitter for symmetry breaking
        self.ccom += rng.normal(0.0, 0.002, self.ccom.shape)
        # body-frame inertia tensors (local z along the column axis)
        R, L = self.R, 1.85
        self.Ibody = np.zeros((self.nc, 3))
        for k in range(self.nc):
            m = self.cm[k]
            if self.cmat[k] == 1:  # solid cylinder
                iax = 0.5 * m * R ** 2
                iperp = m * (3 * R ** 2 + L ** 2) / 12.0
            else:  # 0.23 x 0.23 x 1.85 box
                s = 0.23
                iax = m * (2 * s ** 2) / 12.0
                iperp = m * (s ** 2 + L ** 2) / 12.0
            self.Ibody[k] = [iperp, iperp, iax]
        self.Iinv_body = 1.0 / self.Ibody
        # column volume proxies (for output density)
        self.cvol = np.where(self.cmat == 1, np.pi * R ** 2 * L, 0.23 * 0.23 * L)
        # calibrate SPH rest density on the initial lattice
        self._build_grid()
        rho, _ = self._density()
        self.rho0 = float(np.median(rho))
        self.B_eos = self.rho0 * C0 ** 2 / GAMMA_EOS
        self.t = 0.0
        self.nsteps = 0
        self.rep = None
        self._diag = {"substeps": 0, "max_water_speed": 0.0, "max_column_speed": 0.0}

    # ------------------------------------------------------- neighbour grid
    def _build_grid(self):
        ci = np.floor((self.wp - G_ORIGIN) / H).astype(np.int64)
        ci = np.clip(ci, 0, G_DIMS - 1)
        key = (ci[:, 0] * G_DIMS[1] + ci[:, 1]) * G_DIMS[2] + ci[:, 2]
        order = np.argsort(key, kind="stable")
        self.g_key_sorted = key[order]
        self.g_idx = order
        ar = np.arange(NC)
        self.g_lo = np.searchsorted(self.g_key_sorted, ar, side="left")
        self.g_hi = np.searchsorted(self.g_key_sorted, ar, side="right")

    def _ball_indices(self, center, radius):
        c = np.floor((np.asarray(center) - G_ORIGIN) / H).astype(np.int64)
        k = int(np.ceil(radius / H))
        lo = c - k
        hi = c + k
        lo_c = np.clip(lo, 0, G_DIMS - 1)
        hi_c = np.clip(hi, 0, G_DIMS - 1)
        xs = np.arange(lo_c[0], hi_c[0] + 1)
        ys = np.arange(lo_c[1], hi_c[1] + 1)
        zs = np.arange(lo_c[2], hi_c[2] + 1)
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return np.zeros(0, dtype=np.int64)
        cx, cy, cz = np.meshgrid(xs, ys, zs, indexing="ij")
        cells = (cx.ravel() * G_DIMS[1] + cy.ravel()) * G_DIMS[2] + cz.ravel()
        cells = np.unique(cells)
        lo = self.g_lo[cells]
        hi = self.g_hi[cells]
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            return np.zeros(0, dtype=np.int64)
        rep = np.repeat(np.arange(len(cells)), cnt)
        within = np.arange(tot) - np.concatenate(([0], np.cumsum(cnt)[:-1]))[rep]
        idx = self.g_idx[lo[rep] + within]
        d = self.wp[idx] - center
        keep = np.einsum("ij,ij->i", d, d) <= radius * radius
        return idx[keep]

    # ----------------------------------------------------------------- SPH
    def _pairs(self):
        """Ordered neighbour pairs (both (i,j) and (j,i) appear across offsets)."""
        ci = np.floor((self.wp - G_ORIGIN) / H).astype(np.int64)
        ci = np.clip(ci, 0, G_DIMS - 1)
        ib_all, jb_all, dr_all = [], [], []
        src = np.arange(self.nw)
        for off in OFF27:
            cj = ci + off
            ok = (cj[:, 0] >= 0) & (cj[:, 0] < G_DIMS[0]) & \
                 (cj[:, 1] >= 0) & (cj[:, 1] < G_DIMS[1]) & \
                 (cj[:, 2] >= 0) & (cj[:, 2] < G_DIMS[2])
            if not np.any(ok):
                continue
            cjc = np.clip(cj[ok], 0, G_DIMS - 1)
            fj = (cjc[:, 0] * G_DIMS[1] + cjc[:, 1]) * G_DIMS[2] + cjc[:, 2]
            lo = self.g_lo[fj]
            hi = self.g_hi[fj]
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            rep = np.repeat(src[ok], cnt)
            within = np.arange(tot) - np.concatenate(([0], np.cumsum(cnt)[:-1]))[rep]
            jb = self.g_idx[lo[rep] + within]
            ib = rep
            dr = self.wp[ib] - self.wp[jb]
            r2 = np.einsum("ij,ij->i", dr, dr)
            keep = r2 < H2  # kernel support = H (grid cell): +/-1-cell search is exact
            if off[0] == 0 and off[1] == 0 and off[2] == 0:
                keep &= ib != jb
            ib_all.append(ib[keep])
            jb_all.append(jb[keep])
            dr_all.append(dr[keep])
        if not ib_all:
            e = np.zeros(0, dtype=np.int64)
            return e, e, np.zeros((0, 3))
        return np.concatenate(ib_all), np.concatenate(jb_all), np.concatenate(dr_all)

    def _density(self):
        ib, jb, dr = self._pairs()
        r2 = np.einsum("ij,ij->i", dr, dr)
        r = np.sqrt(r2)
        q = r / H
        W = np.where(q < 1.0, SIGMA64 * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3), 0.0)
        rho_side = np.bincount(ib, weights=self.wm[jb] * W, minlength=self.nw)
        rho = 0.5 * rho_side + self.wm * SIGMA64
        return rho, (ib, jb, dr, r, q, W)

    def _sph_accel(self, ib, jb, dr, r, q, W, rho):
        rho = np.maximum(rho, 0.5 * self.rho0)
        P = np.maximum(self.B_eos * ((rho / self.rho0) ** GAMMA_EOS - 1.0), 0.0)
        pterm = P / rho ** 2
        dWdr = np.where(q < 1.0, SIGMA64 / H * (-3.0 * q + 2.25 * q ** 2), 0.0)
        r_safe = np.maximum(r, 1e-6)
        r2_safe = r * r + 0.01 * H2
        # symmetric pressure force on i from j: -mi mj (Pi/rho_i^2 + Pj/rho_j^2) gradW
        coef = -self.wm[ib] * self.wm[jb] * (pterm[ib] + pterm[jb]) * dWdr / r_safe
        fp = coef[:, None] * dr
        # pairwise viscosity: mi mj nu (vj-vi).dr / (r^2+eps) * dWdr / r * dr
        dv = self.wv[jb] - self.wv[ib]
        dot = np.einsum("ij,ij->i", dv, dr)
        cv = NU_VISC * self.wm[ib] * self.wm[jb] * dot / r2_safe * dWdr / r_safe
        fv = cv[:, None] * dr
        f_pair = fp + fv
        a = np.stack([np.bincount(ib, weights=f_pair[:, d], minlength=self.nw)
                      for d in range(3)], axis=1) / self.wm[:, None]
        # XSPH correction velocity
        wgt = self.wm[jb] * W / (0.5 * (rho[ib] + rho[jb]))
        wsum = np.bincount(ib, weights=wgt, minlength=self.nw)
        xsum = np.stack([np.bincount(ib, weights=wgt * dv[:, d], minlength=self.nw)
                         for d in range(3)], axis=1)
        vx = np.zeros_like(xsum)
        m = wsum > 1e-12
        vx[m] = xsum[m] / wsum[m][:, None]
        return a, vx

    def _wall_accel_water(self):
        a = np.zeros((self.nw, 3))
        lo = np.array([XMIN, YMIN, 0.0])
        hi = np.array([XMAX, YMAX, np.inf])
        for d in range(3):
            if np.isfinite(hi[d]):
                pen = WALL_MARGIN - (hi[d] - self.wp[:, d])
                m = pen > 0.0
                if np.any(m):
                    vn = self.wv[m, d]  # outward (+) when approaching this wall
                    acc = WALL_K * pen[m] + WALL_C * np.maximum(vn, 0.0)
                    a[m, d] -= np.clip(acc, 0.0, WALL_AMAX)
            pen = WALL_MARGIN - (self.wp[:, d] - lo[d])
            m = pen > 0.0
            if np.any(m):
                vn = -self.wv[m, d]
                acc = WALL_K * pen[m] + WALL_C * np.maximum(vn, 0.0)
                a[m, d] += np.clip(acc, 0.0, WALL_AMAX)
        return a

    # ------------------------------------------------------------- columns
    def _column_forces(self, collect_report):
        Rmat = Rotation.from_quat(self.cquat).as_matrix()
        F = np.zeros((self.nc, 3))
        T = np.zeros((self.nc, 3))
        F_water_on_col = np.zeros((self.nc, 3))
        a_water = np.zeros((self.nw, 3))
        wc_gaps = []
        # ------------- water <-> columns (penalty + normal damping + tangential drag)
        probe_ts = np.linspace(-self.A, self.A, 5)
        Rg = self.R + COUP_BAND
        for k in range(self.nc):
            axis = Rmat[k][:, 2]
            idxs = [self._ball_indices(self.ccom[k] + t * axis, Rg) for t in probe_ts]
            nonempty = [x for x in idxs if len(x)]
            if not nonempty:
                continue
            wi = np.unique(np.concatenate(nonempty))
            pts = self.wp[wi]
            rel = pts - self.ccom[k]
            t = np.clip(rel @ axis, -self.A, self.A)
            cp = self.ccom[k] + t[:, None] * axis[None, :]
            vec = pts - cp
            d = np.linalg.norm(vec, axis=1)
            pen = (self.R + COUP_BAND) - d
            m = (pen > 0.0) & (d > 1e-9)
            if not np.any(m):
                continue
            wm = wi[m]
            nhat = vec[m] / d[m][:, None]  # axis point -> water particle
            vcol_pt = self.cvel[k] + np.cross(self.comega[k], cp[m] - self.ccom[k])
            vrel = self.wv[wm] - vcol_pt
            vn = np.einsum("ij,ij->i", vrel, nhat)
            fn = COUP_K * pen[m] + COUP_C * np.maximum(-vn, 0.0)
            vt = vrel - vn[:, None] * nhat
            f_on_w = fn[:, None] * nhat - COUP_CT * vt
            # clamp per-particle acceleration
            mag = np.linalg.norm(f_on_w, axis=1) / self.wm[wm]
            over = mag > COUP_AMAX_W
            if np.any(over):
                f_on_w[over] *= (COUP_AMAX_W / mag[over])[:, None]
            np.add.at(a_water, wm, -f_on_w / self.wm[wm][:, None])
            # reaction on the column: force = -sum(f_on_w); torque about com
            f_pairs = -f_on_w
            f_col = f_pairs.sum(axis=0)
            acc = np.linalg.norm(f_col) / self.cm[k]
            if acc > COUP_AMAX_C:
                f_col *= COUP_AMAX_C / acc
                f_pairs = f_pairs * (COUP_AMAX_C / acc)
            F[k] += f_col
            F_water_on_col[k] += f_col
            T[k] += np.cross(cp[m] - self.ccom[k], f_pairs).sum(axis=0)
            if collect_report:
                g = d[m] - self.R
                sel = g < CONTACT_BAND_WC
                if np.any(sel):
                    wc_gaps.append(g[sel])
        # ------------- column <-> column (capsule segment-segment penalty)
        rr_gaps = []
        seg_lo = self.ccom - self.A * Rmat[:, :, 2]
        seg_hi = self.ccom + self.A * Rmat[:, :, 2]
        for i in range(self.nc):
            for j in range(i + 1, self.nc):
                gap, cp_i, cp_j, nrm = self._seg_seg(seg_lo[i], seg_hi[i], seg_lo[j], seg_hi[j])
                g = gap - 2.0 * self.R
                if collect_report and g < CONTACT_BAND_RR:
                    rr_gaps.append(g)
                if g >= 0.0:
                    continue
                vci = self.cvel[i] + np.cross(self.comega[i], cp_i - self.ccom[i])
                vcj = self.cvel[j] + np.cross(self.comega[j], cp_j - self.ccom[j])
                vn = float((vci - vcj) @ nrm)
                fn = -RR_K * g + RR_C * max(-vn, 0.0)
                fi = fn * nrm
                ai = np.linalg.norm(fi) / self.cm[i]
                if ai > 5.0e3:
                    fi = fi * (5.0e3 / ai)
                fj = -fi
                F[i] += fi
                F[j] += fj
                T[i] += np.cross(cp_i - self.ccom[i], fi)
                T[j] += np.cross(cp_j - self.ccom[j], fj)
        # ------------- columns vs floor (penalty + Coulomb friction) and walls
        floor_gaps = []
        F_bound = np.zeros((self.nc, 3))
        for k in range(self.nc):
            axis = Rmat[k][:, 2]
            p_bot = self.ccom[k] - self.A * axis
            p_top = self.ccom[k] + self.A * axis
            p_low = p_bot if p_bot[2] <= p_top[2] else p_top
            # lowest surface point of the capsule
            surf = p_low.copy()
            surf[2] -= self.R * (1.0 if abs(axis[2]) > 0.99 else abs(axis[2]) + (1.0 - abs(axis[2])))
            surf[2] = p_low[2] - self.R
            g = surf[2]
            if collect_report:
                floor_gaps.append(g)
            if g < 0.05:
                cpk = p_low.copy()
                cpk[2] = max(cpk[2] - self.R, 0.0)
                vc = self.cvel[k] + np.cross(self.comega[k], cpk - self.ccom[k])
                pen = max(-g, 0.0)
                fn = FL_K * pen - FL_C * min(vc[2], 0.0)
                fn = max(fn, 0.0)
                if fn > 0.0 or pen > 0.0:
                    fvec = np.array([0.0, 0.0, fn])
                    F[k] += fvec
                    F_bound[k] += fvec
                    T[k] += np.cross(cpk - self.ccom[k], fvec)
                    vt = vc.copy()
                    vt[2] = 0.0
                    sp = np.linalg.norm(vt)
                    if sp > 1e-6 and fn > 0.0:
                        ff = min(FRICTION * fn, 800.0 * sp * self.cm[k])
                        ffr = np.array([-ff * vt[0] / sp, -ff * vt[1] / sp, 0.0])
                        F[k] += ffr
                        F_bound[k] += ffr
                        T[k] += np.cross(cpk - self.ccom[k], ffr)
            # lateral walls: treat column as a 0.4 m-radius sphere at its com
            Rw = 0.40
            for d, lo_w, hi_w in ((0, XMIN, XMAX), (1, YMIN, YMAX)):
                pen = (Rw + 0.05) - (self.ccom[k][d] - lo_w)
                if pen > 0.0:
                    f = min(RR_K * pen, 2.0e4)
                    F[k][d] += f
                    F_bound[k][d] += f
                pen = (Rw + 0.05) - (hi_w - self.ccom[k][d])
                if pen > 0.0:
                    f = min(RR_K * pen, 2.0e4)
                    F[k][d] -= f
                    F_bound[k][d] -= f
        # damping
        F -= CDAMP * self.cm[:, None] * self.cvel
        T -= CDAMP_ROT * self.comega
        if collect_report:
            parts = [np.asarray(a, dtype=float) for a in wc_gaps if len(a)]
            parts.append(np.array(rr_gaps, dtype=float))
            parts.append(np.array(floor_gaps, dtype=float))
            parts = [a for a in parts if a.size]
            allg = np.concatenate(parts) if parts else np.array([0.05])
            if allg.size > 4000:
                allg = allg[:4000]
            self.rep = {
                "contact_gaps": allg,
                "a_water_coupling": a_water.copy(),
                "F_water_on_col": F_water_on_col.copy(),
                "F_col_bound": F_bound.copy(),
                "rho_w": None,
            }
        return F, T, a_water, F_bound, Rmat

    @staticmethod
    def _seg_seg(p1, q1, p2, q2):
        """Closest distance between two segments; returns (dist, cp1, cp2, normal)."""
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = float(d1 @ d1)
        e = float(d2 @ d2)
        f = float(d2 @ r)
        if a <= 1e-12 and e <= 1e-12:
            return float(np.linalg.norm(p1 - p2)), p1.copy(), p2.copy(), np.array([0.0, 0.0, 1.0])
        if a <= 1e-12:
            s = 0.0
            t = float(np.clip(f / e, 0.0, 1.0))
        else:
            c = float(d1 @ r)
            if e <= 1e-12:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            else:
                b = float(d1 @ d2)
                denom = a * e - b * b
                if denom > 1e-12:
                    s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
                else:
                    s = 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = float(np.clip(-c / a, 0.0, 1.0))
                elif t > 1.0:
                    t = 1.0
                    s = float(np.clip((b - c) / a, 0.0, 1.0))
        cp1 = p1 + d1 * s
        cp2 = p2 + d2 * t
        diff = cp1 - cp2
        dist = float(np.linalg.norm(diff))
        nrm = diff / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0])
        return dist, cp1, cp2, nrm

    # ------------------------------------------------------------- stepping
    def _substep(self, collect_report):
        self._build_grid()
        rho, (ib, jb, dr, r, q, W) = self._density()
        a_sph, v_xsph = self._sph_accel(ib, jb, dr, r, q, W, rho)
        a_wall = self._wall_accel_water()
        Fc, Tc, a_coup, F_bound, Rmat = self._column_forces(collect_report)

        # integrate water
        a_w = a_sph + a_wall + a_coup + G[None, :]
        self.wv += DT * a_w
        sp = np.linalg.norm(self.wv, axis=1)
        over = sp > WATER_VMAX
        if np.any(over):
            self.wv[over] *= (WATER_VMAX / sp[over])[:, None]
        self.wp += DT * (self.wv + XSPH * v_xsph)
        # hard containment (soft penalty above already applied)
        np.maximum(self.wp[:, 0], XMIN + 0.01, out=self.wp[:, 0])
        np.minimum(self.wp[:, 0], XMAX - 0.01, out=self.wp[:, 0])
        np.maximum(self.wp[:, 1], YMIN + 0.01, out=self.wp[:, 1])
        np.minimum(self.wp[:, 1], YMAX - 0.01, out=self.wp[:, 1])
        below = self.wp[:, 2] < 0.015
        if np.any(below):
            self.wp[below, 2] = 0.015
            np.maximum(self.wv[below, 2], 0.0, out=self.wv[below, 2])

        # integrate columns (linear + rotational)
        self.cvel += DT * (Fc / self.cm[:, None])
        sp = np.linalg.norm(self.cvel, axis=1)
        over = sp > COL_VMAX
        if np.any(over):
            self.cvel[over] *= (COL_VMAX / sp[over])[:, None]
        self.ccom += DT * self.cvel
        # alpha = R Iinv_body R^T torque
        Tb = np.transpose(Rmat, (0, 2, 1)) @ Tc[:, :, None]
        alpha = (Rmat @ (self.Iinv_body[:, :, None] * Tb))[:, :, 0]
        self.comega += DT * alpha
        wn = np.linalg.norm(self.comega, axis=1)
        over = wn > COL_WMAX
        if np.any(over):
            self.comega[over] *= (COL_WMAX / wn[over])[:, None]
        ang = wn * DT
        m = wn > 1e-12
        dq = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nc, 1))
        if np.any(m):
            ax = self.comega[m] / wn[m][:, None]
            ha = 0.5 * ang[m]
            dq[m, :3] = ax * np.sin(ha)[:, None]
            dq[m, 3] = np.cos(ha)
        self.cquat = _qmul(dq, self.cquat)
        self.cquat /= np.linalg.norm(self.cquat, axis=1)[:, None]
        # keep capsules above the floor
        Rm = Rotation.from_quat(self.cquat).as_matrix()
        ends = np.stack([self.ccom - self.A * Rm[:, :, 2],
                         self.ccom + self.A * Rm[:, :, 2]], axis=1)
        zlow = ends[:, :, 2].min(axis=1) - self.R
        bad = zlow < 0.005
        if np.any(bad):
            self.ccom[bad, 2] += 0.005 - zlow[bad]
            np.maximum(self.cvel[bad, 2], 0.0, out=self.cvel[bad, 2])

        self.t += DT
        self.nsteps += 1
        if collect_report and self.rep is not None:
            self.rep["rho_w"] = rho.copy()
        self._diag["substeps"] = self.nsteps
        self._diag["max_water_speed"] = float(np.linalg.norm(self.wv, axis=1).max()) if self.nw else 0.0
        self._diag["max_column_speed"] = float(np.linalg.norm(self.cvel, axis=1).max())

    # -------------------------------------------------------------- output
    def advance_to(self, time: float):
        target_steps = int(round(time / DT))
        while self.nsteps < target_steps:
            collect = (self.nsteps == target_steps - 1)
            self._substep(collect)
        if self.rep is None:  # t == 0 frame
            self._build_grid()
            rho, _ = self._density()
            self._column_forces(True)
            self.rep["rho_w"] = rho.copy()
        return self._frame(time)

    def _frame(self, time):
        n = self.n_total
        rep = self.rep
        pos = np.vstack((self.wp, self.ccom))
        vel = np.vstack((self.wv, self.cvel))
        mass = np.concatenate((self.wm, self.cm))
        orient = np.vstack((np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nw, 1)), self.cquat))
        angvel = np.vstack((np.zeros((self.nw, 3)), self.comega))
        body_id = np.concatenate((np.zeros(self.nw, dtype=int), self.col_body_ids))
        rho_now = np.zeros(n)
        rho_now[: self.nw] = rep["rho_w"] if rep.get("rho_w") is not None else self.rho0
        rho_now[self.nw:] = self.cm / self.cvol
        # external force: gravity + basin boundary penalty (excludes coupling pairs)
        ext_w = self.wm[:, None] * (G[None, :] + self._wall_accel_water())
        ext_c = self.cm[:, None] * G[None, :] + rep["F_col_bound"]
        # interaction force [K,3]: net paired multiphysics/contact force per material
        a_coup = rep["a_water_coupling"]
        F_wc = rep["F_water_on_col"]
        inter = np.zeros((3, 3))
        inter[0] = -np.sum(self.wm[:, None] * a_coup, axis=0)  # water <- columns
        light = self.cmat == 1
        heavy = self.cmat == 2
        inter[1] = F_wc[light].sum(axis=0)   # light cylinders <- water/contacts
        inter[2] = F_wc[heavy].sum(axis=0)   # heavy cubes <- water/contacts
        attrs = {
            "scene_group": self.scene_group.copy(),
            "density": rho_now.copy(),
            "mass_quantity": mass.copy(),
        }
        fields = {
            "density": rho_now.copy(),
            "contact_signed_distance": np.asarray(rep["contact_gaps"], dtype=float),
            "cumulative_mass_source": np.zeros(3),
            "external_force": np.vstack((ext_w, ext_c)),
            "interaction_force": inter,
        }
        scalars = {
            "contact_length_scale": 0.16,
            "water_rest_density": self.rho0,
        }
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=orient,
            angular_velocity=angvel,
            mass=mass,
            body_id=body_id,
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return dict(self._diag)
