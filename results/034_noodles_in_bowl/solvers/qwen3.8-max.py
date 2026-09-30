"""Physically-based solver for 034_noodles_in_bowl.

625 rod-modeled noodles (2 nodes each, length 0.4 m, radius 0.006 m) fall
under gravity into an axisymmetric bowl, intertwine and settle.  The
simulation uses a position-based-dynamics (XPBD-style) time integrator:

* semi-implicit Euler prediction with gravity,
* exact per-rod length projection (each node belongs to one rod constraint),
* bowl contact via a signed-distance field of the inner profile polyline
  (revolution surface) with projection to the noodle radius,
* noodle--noodle contact via exact segment-segment distance on candidate
  pairs from a uniform-grid broad phase over sampled rod points,
* Coulomb-like tangential friction damping at contacts and a floor safety
  plane so escaped nodes cannot fall forever.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

_S = 1024          # key-encoding stride
_OFF = 128         # cell-index offset (keeps indices non-negative)


def _scatter_add(M: int, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """out[idx] += vals (duplicate-safe, fast bincount version)."""
    out = np.zeros((M, 3))
    if idx.size:
        for c in range(3):
            out[:, c] = np.bincount(idx, weights=vals[:, c], minlength=M)
    return out


def _scatter_min(M: int, idx: np.ndarray, vals: np.ndarray, init: float) -> np.ndarray:
    """out[idx] = minimum(out[idx], vals) (duplicate-safe, sort based)."""
    out = np.full(M, float(init))
    if idx.size:
        order = np.lexsort((vals, idx))
        si = idx[order]
        sv = vals[order]
        keep = np.concatenate(([True], si[1:] != si[:-1]))
        out[si[keep]] = sv[keep]
    return out


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()
        self.X = np.array(frame0.position, dtype=float)
        self.V = np.array(frame0.velocity, dtype=float)
        self.N = self.X.shape[0]
        mass = frame0.mass
        if mass is None:
            mass = self.X[:, 0] * 0.0 + 0.00125
        self.M = np.asarray(mass, dtype=float).reshape(self.N)
        self.W = 1.0 / self.M

        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.duration = float(cfg.get("duration", 3.0))
        self.radius = float(cfg.get("radius", 0.006))
        self.rod_len = float(cfg.get("noodle_length", 0.4))
        self.nrod = self.N // 2
        self.n0 = 2 * np.arange(self.nrod)
        self.n1 = self.n0 + 1

        self.NSUB = 16
        self.dt_sub = self.sample_dt / self.NSUB
        self.t = 0.0

        # bowl inner profile polyline (r, z), extended to the axis
        prof = np.array([[0.0, 0.0], [0.12, 0.0], [0.38, 0.08],
                         [0.66, 0.34], [0.72, 0.48]])
        self.pA = prof[:-1]
        self.pE = prof[1:] - prof[:-1]
        self.pL2 = (self.pE ** 2).sum(1)
        elen = np.sqrt(self.pL2)
        self.pN = np.stack([-self.pE[:, 1], self.pE[:, 0]], 1) / elen[:, None]

        # collision sampling along each rod (spacing 0.1 m)
        self.ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.ns = len(self.ts)
        self.sidx = np.repeat(np.arange(self.nrod), self.ns)
        self.R_search = 0.125
        self.cell = self.R_search
        offs = np.array([(dx, dy, dz)
                         for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1)
                         for dz in (-1, 0, 1)], dtype=np.int64)
        self.off_keys = (offs[:, 0] * _S + offs[:, 1]) * _S + offs[:, 2]

        # frozen frame outputs
        self.Fext = self.M[:, None] * self.g[None, :]
        self.mass_qty = np.array(self.M)
        self.cum_mass = np.zeros(1)
        self.contact_len = float(self.radius)
        self.body_id = np.repeat(np.arange(self.nrod), 2).astype(np.int64)

        self.mu_bowl = 0.25
        self.mu_nood = 0.20
        self._active_contacts = 0

    # ------------------------------------------------------------- bowl SDF
    def _bowl(self, P):
        """Signed distance (positive inside) and inward unit normal."""
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.sqrt(x * x + y * y)
        q = np.stack([r, z], 1)                                   # [N,2]
        dq = q[:, None, :] - self.pA[None, :, :]                  # [N,K,2]
        u = np.clip((dq * self.pE[None]).sum(2) / self.pL2[None], 0.0, 1.0)
        cp = self.pA[None] + u[:, :, None] * self.pE[None]        # [N,K,2]
        d = q[:, None, :] - cp                                    # [N,K,2]
        absd = np.sqrt((d ** 2).sum(2))
        sgn = (d * self.pN[None]).sum(2)
        k = np.argmin(absd, 1)
        rows = np.arange(self.N)
        sd = sgn[rows, k]
        n2 = self.pN[k]                                           # [N,2]
        rsafe = np.maximum(r, 1e-9)
        n3 = np.stack([n2[:, 0] * x / rsafe, n2[:, 0] * y / rsafe, n2[:, 1]], 1)
        n3 /= np.maximum(np.linalg.norm(n3, axis=1, keepdims=True), 1e-12)
        return sd, n3

    # ---------------------------------------------------------- rod length
    def _rods(self, P):
        d = P[self.n1] - P[self.n0]
        dist = np.sqrt((d * d).sum(1))
        dist = np.maximum(dist, 1e-12)
        corr = ((dist - self.rod_len) / dist)[:, None] * d
        P[self.n0] += 0.5 * corr
        P[self.n1] -= 0.5 * corr

    # ---------------------------------------------------------- broad phase
    def _candidate_rod_pairs(self, P):
        ts = self.ts[None, :, None]
        SP = (P[self.n0][:, None, :] * (1.0 - ts)
              + P[self.n1][:, None, :] * ts).reshape(-1, 3)       # [M,3]
        ci = np.floor(SP / self.cell).astype(np.int64) + _OFF
        key = (ci[:, 0] * _S + ci[:, 1]) * _S + ci[:, 2]
        order = np.argsort(key, kind="stable")
        skey = key[order]
        ukeys, ustart, ucount = np.unique(skey, return_index=True,
                                          return_counts=True)
        si_l, sj_l = [], []
        ar = np.arange(len(ukeys))
        for dk in self.off_keys:
            tgt = ukeys + dk
            pos = np.searchsorted(ukeys, tgt)
            ok = pos < len(ukeys)
            posc = pos[ok]
            ok2 = ukeys[posc] == tgt[ok]
            A = ar[ok][ok2]
            B = posc[ok2]
            if A.size == 0:
                continue
            cntA = ucount[A]
            cntB = ucount[B]
            prod = cntA * cntB
            tot = int(prod.sum())
            if tot == 0:
                continue
            rep = np.repeat(np.arange(prod.size), prod)
            cumb = np.zeros(prod.size, dtype=np.int64)
            cumb[1:] = np.cumsum(prod)[:-1]
            kk = np.arange(tot) - np.repeat(cumb, prod)
            cntBr = np.repeat(cntB, prod)
            ia = kk // cntBr
            ib = kk - ia * cntBr
            si_l.append(order[np.repeat(ustart[A], prod) + ia])
            sj_l.append(order[np.repeat(ustart[B], prod) + ib])
        if not si_l:
            return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                    np.zeros(0), np.zeros(0))
        si = np.concatenate(si_l)
        sj = np.concatenate(sj_l)
        m = si < sj
        si, sj = si[m], sj[m]
        if si.size == 0:
            return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                    np.zeros(0), np.zeros(0))
        dvec = SP[si] - SP[sj]
        d2 = (dvec * dvec).sum(1)
        m = d2 < self.R_search ** 2
        si, sj = si[m], sj[m]
        ra = self.sidx[si]
        rb = self.sidx[sj]
        m = ra != rb
        ra, rb = ra[m], rb[m]
        lo = np.minimum(ra, rb)
        hi = np.maximum(ra, rb)
        code = np.unique(lo * self.nrod + hi)
        A_rod = code // self.nrod
        B_rod = code % self.nrod
        return A_rod, B_rod, si, sj

    # --------------------------------------------------- segment-segment
    def _seg_seg(self, P, A_rod, B_rod):
        p1 = P[2 * A_rod]
        q1 = P[2 * A_rod + 1]
        p2 = P[2 * B_rod]
        q2 = P[2 * B_rod + 1]
        d1 = q1 - p1
        d2 = q2 - p2
        r0 = p1 - p2
        a = (d1 * d1).sum(1)
        e = (d2 * d2).sum(1)
        f = (d2 * r0).sum(1)
        c = (d1 * r0).sum(1)
        b = (d1 * d2).sum(1)
        denom = a * e - b * b
        s = np.where(denom > 1e-12,
                     (b * f - c * e) / np.maximum(denom, 1e-12), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + f) / np.maximum(e, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        s = (b * t - c) / np.maximum(a, 1e-12)
        s = np.clip(s, 0.0, 1.0)
        g = (p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])
        dist = np.sqrt(np.maximum((g * g).sum(1), 1e-18))
        return s, t, g, dist

    def _solve_contacts(self, P, A_rod, B_rod, iters=2):
        two_r = 2.0 * self.radius
        wm = self.W[0]  # uniform node mass
        nacc = np.zeros((self.N, 3))
        active_total = 0
        for _ in range(iters):
            # --- bowl ---
            sd, n3 = self._bowl(P)
            act = sd < self.radius
            if act.any():
                ia = np.nonzero(act)[0]
                P[ia] += (self.radius - sd[ia])[:, None] * n3[ia]
                nacc[ia] += n3[ia] * self.mu_bowl / max(self.mu_bowl, 1e-9)
            # --- floor safety plane ---
            act = P[:, 2] < self.radius
            if act.any():
                ia = np.nonzero(act)[0]
                P[ia, 2] = self.radius
                nacc[ia, 2] += self.mu_bowl / max(self.mu_bowl, 1e-9)
            # --- noodle vs noodle ---
            if A_rod.size:
                s, t, g, dist = self._seg_seg(P, A_rod, B_rod)
                hit = (dist < two_r) & (dist > 1e-9)
                if hit.any():
                    ih = np.nonzero(hit)[0]
                    sh, th, gh, dh = s[ih], t[ih], g[ih], dist[ih]
                    nhat = gh / dh[:, None]
                    effA = wm * ((1.0 - sh) ** 2 + sh ** 2)
                    effB = wm * ((1.0 - th) ** 2 + th ** 2)
                    lam = (two_r - dh) / (effA + effB)
                    a0 = 2 * A_rod[ih]
                    a1 = a0 + 1
                    b0 = 2 * B_rod[ih]
                    b1 = b0 + 1
                    wgt = self.mu_nood / max(self.mu_bowl, 1e-9)
                    nacc = nacc + _scatter_add(self.N, a0, lam[:, None] * 0 + (wgt * (1 - sh))[:, None] * nhat)
                    nacc = nacc + _scatter_add(self.N, a1, (wgt * sh)[:, None] * nhat)
                    nacc = nacc + _scatter_add(self.N, b0, (-wgt * (1 - th))[:, None] * nhat)
                    nacc = nacc + _scatter_add(self.N, b1, (-wgt * th)[:, None] * nhat)
                    idx = np.concatenate([a0, a1, b0, b1])
                    c0 = np.concatenate([lam * wm * (1 - sh), lam * wm * sh,
                                         -lam * wm * (1 - th), -lam * wm * th])
                    disp = c0[:, None] * np.concatenate([nhat, nhat, nhat, nhat])
                    P += _scatter_add(self.N, idx, disp)
                    active_total += ih.size
        self._active_contacts = active_total // max(iters, 1)
        return nacc

    # --------------------------------------------------------------- step
    def _step(self, dt):
        self.V += self.g * dt
        Xold = self.X
        P = self.X + self.V * dt
        self._rods(P)
        A_rod, B_rod, _, _ = self._candidate_rod_pairs(P)
        nacc = self._solve_contacts(P, A_rod, B_rod, iters=2)
        self._rods(P)
        self.X = P
        self.V = (self.X - Xold) / dt
        # tangential friction at contacting nodes
        nn = np.linalg.norm(nacc, axis=1)
        con = nn > 1e-9
        if con.any():
            nh = nacc[con] / nn[con][:, None]
            v = self.V[con]
            vn = (v * nh).sum(1)[:, None]
            vt = v - vn * nh
            self.V[con] = v - self.mu_bowl * vt
        # mild drag + speed cap for robustness
        self.V *= (1.0 - min(0.5 * dt, 0.01))
        sp = np.linalg.norm(self.V, axis=1)
        bad = sp > 15.0
        if bad.any():
            self.V[bad] *= (15.0 / sp[bad])[:, None]
        self.t += dt

    # -------------------------------------------------------------- frame
    def _make_frame(self, time):
        sd, _ = self._bowl(self.X)
        gap_bowl = sd - self.radius
        gap_nood = np.full(self.N, 0.5)
        A_rod, B_rod, _, _ = self._candidate_rod_pairs(self.X)
        if A_rod.size:
            _, _, _, dist = self._seg_seg(self.X, A_rod, B_rod)
            near = dist < 0.06
            if near.any():
                Ar, Br, dv = A_rod[near], B_rod[near], dist[near] - 2.0 * self.radius
                idx = np.concatenate([2 * Ar, 2 * Ar + 1, 2 * Br, 2 * Br + 1])
                vals = np.concatenate([dv, dv, dv, dv])
                gap_nood = _scatter_min(self.N, idx, vals, 0.5)
        gap = np.minimum(gap_bowl, gap_nood)
        gap = np.minimum(gap, 0.5)
        return StateFrame(
            time=float(time),
            position=self.X.copy(),
            velocity=self.V.copy(),
            mass=self.M.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass_qty.copy()},
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": self.cum_mass.copy(),
                "external_force": self.Fext.copy(),
            },
            scalars={"contact_length_scale": self.contact_len},
        )

    # ------------------------------------------------------------ public
    def advance_to(self, time: float):
        eps = 1e-9
        n = int(round((time - self.t) / self.dt_sub))
        n = max(n, 0)
        for _ in range(n):
            if self.t >= time - eps:
                break
            self._step(min(self.dt_sub, time - self.t))
        self.t = max(self.t, time) if abs(self.t - time) < 1e-6 else self.t
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        return {
            "noodles": int(self.nrod),
            "nodes": int(self.N),
            "active_contacts_last_substep": int(self._active_contacts),
            "sim_time": float(self.t),
        }
