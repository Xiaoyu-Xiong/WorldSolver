"""Physically-based solver for scene 036_cloth_stack_5mm.

Ten 5 mm-thick square cloth panels fall under gravity onto a static board,
collide, fold and settle into a frictional multilayer stack.

Method
------
* Each panel is a 13x13 nodal cloth mesh (from the frozen topology asset).
* Elasticity is modeled with structural / shear / bending spring networks
  whose rest lengths are taken from the frozen initial state.
* Time integration: semi-implicit (symplectic) Euler with small fixed
  substeps (5e-4 s) for stability and determinism.
* Contact: penalty contact between cloth nodes (as spheres of radius
  thickness/2) and (a) the board top plane and (b) triangles of the other
  panels (point-triangle closest point, mid-surface separation enforced to
  the panel thickness).  Broad-phase collision detection uses a uniform
  spatial hash grid rebuilt every substep.
* Friction: velocity-level Coulomb friction with coefficient mu = 0.1,
  applied wherever a normal contact force acts.
* A hard projection backstop removes any deep penetration so states stay
  finite and interpenetration stays bounded.

No keyframes, no precomputed trajectories: every sample is produced by
stepping the physics forward from the frozen initial state.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # provided by the platform runtime
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback for isolated syntax checks
    StateFrame = object


def _point_tri_closest(p, a, b, c):
    """Vectorized closest point on triangles.

    Returns (closest_point, distance, u, v) with barycentric coordinates
    (1-u-v, u, v) of the closest point.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    nrm = np.cross(ab, ac)
    nn = np.maximum(np.einsum("ij,ij->i", nrm, nrm), 1e-30)
    s = np.einsum("ij,ij->i", nrm, ap) / nn
    q = p - nrm * s[:, None]          # projection of p onto triangle plane
    aq = q - a
    d11 = np.einsum("ij,ij->i", ab, ab)
    d22 = np.einsum("ij,ij->i", ac, ac)
    d12 = np.einsum("ij,ij->i", ab, ac)
    b1 = np.einsum("ij,ij->i", ab, aq)
    b2 = np.einsum("ij,ij->i", ac, aq)
    det = d11 * d22 - d12 * d12
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)
    u0 = (d22 * b1 - d12 * b2) / det
    v0 = (d11 * b2 - d12 * b1) / det
    case_a = u0 < 0.0
    case_b = (v0 < 0.0) & ~case_a
    case_c = (u0 + v0 > 1.0) & ~case_a & ~case_b
    u = np.where(case_a, 0.0, u0)
    v = np.where(case_a, np.clip(b2 / np.maximum(d22, 1e-30), 0.0, 1.0), v0)
    u = np.where(case_b, np.clip(b1 / np.maximum(d11, 1e-30), 0.0, 1.0), u)
    v = np.where(case_b, 0.0, v)
    bc = c - b
    d33 = np.maximum(np.einsum("ij,ij->i", bc, bc), 1e-30)
    w = np.clip(np.einsum("ij,ij->i", bc, q - b) / d33, 0.0, 1.0)
    u = np.where(case_c, 1.0 - w, u)
    v = np.where(case_c, w, v)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    cp = a + u[:, None] * ab + v[:, None] * ac
    d = p - cp
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    return cp, dist, u, v


class BenchmarkSolver:
    """Mass-spring cloth stack solver with penalty contact and friction."""

    def __init__(self) -> None:
        self._nsub = 0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        f0 = scene.initial_state()

        self.X = np.array(f0.position, dtype=float)
        self.V = np.array(f0.velocity, dtype=float)
        n = len(self.X)
        self.body = np.array(f0.body_id, dtype=np.int64)
        if getattr(f0, "mass", None) is not None:
            self.m = np.array(f0.mass, dtype=float)
        else:
            nper = int(cfg["panel"]["grid_shape"][0]) * int(cfg["panel"]["grid_shape"][1])
            self.m = np.full(n, float(cfg["panel"]["total_mass"]) / nper)
        attrs = dict(f0.attributes)
        self.uv = np.array(attrs.get("uv", np.zeros((n, 2))), dtype=float)
        if "mass_quantity" in attrs:
            self.mq = np.array(attrs["mass_quantity"], dtype=float)
        else:
            self.mq = self.m.copy()
        scal0 = dict(f0.scalars)
        self.constraint_scale = float(scal0.get("constraint_scale", 1.35))

        self.g = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gz = float(self.g[2])
        self.thickness = float(cfg["panel"]["thickness"])
        self.mu = float(cfg.get("friction", 0.1))
        self.board_top = float(cfg.get("board_top", 0.05))
        self.n_panels = int(cfg.get("panel_count", 10))
        gs = cfg["panel"]["grid_shape"]
        self.nx, self.ny = int(gs[0]), int(gs[1])
        self.nper = self.nx * self.ny

        # ---- topology (frozen asset, regular-grid fallback) ---------------
        self.F = self._load_faces()

        # ---- spring network ----------------------------------------------
        k_struct, k_shear, k_bend = 1200.0, 600.0, 5.0
        zeta = 0.05
        mbar = float(np.mean(self.m))
        sa, sb, ks = [], [], []
        nx, ny, nper = self.nx, self.ny, self.nper

        def gid(p, i, j):
            return p * nper + i * nx + j

        for p in range(self.n_panels):
            for i in range(ny):
                for j in range(nx):
                    n0 = gid(p, i, j)
                    if j + 1 < nx:
                        sa.append(n0); sb.append(n0 + 1); ks.append(k_struct)
                    if i + 1 < ny:
                        sa.append(n0); sb.append(n0 + nx); ks.append(k_struct)
                    if j + 2 < nx:
                        sa.append(n0); sb.append(n0 + 2); ks.append(k_bend)
                    if i + 2 < ny:
                        sa.append(n0); sb.append(n0 + 2 * nx); ks.append(k_bend)
                    if i + 1 < ny and j + 1 < nx:
                        sa.append(n0); sb.append(n0 + nx + 1); ks.append(k_shear)
                        sa.append(n0 + 1); sb.append(n0 + nx); ks.append(k_shear)
        self.sa = np.array(sa, dtype=np.int64)
        self.sb = np.array(sb, dtype=np.int64)
        self.ks = np.array(ks, dtype=float)
        self.L0 = np.linalg.norm(self.X[self.sa] - self.X[self.sb], axis=1)
        self.L0 = np.maximum(self.L0, 1e-9)
        self.cs = 2.0 * zeta * np.sqrt(self.ks * mbar)   # spring damping

        # ---- contact / integration parameters -----------------------------
        self.dt_sub = 5.0e-4
        self.kc = 2500.0                       # contact penalty stiffness
        self.cc = 2.0 * 0.3 * np.sqrt(self.kc * mbar)   # contact damping
        self.kv = 0.05                         # mild aerodynamic drag (1/s)
        self.R = 0.012                         # contact detection radius
        self.R_csd = 0.03                      # near-contact reporting radius
        self.cell = 0.18                       # spatial hash cell size
        self.KKEY = 100000
        self.zrest = self.board_top + 0.5 * self.thickness
        self.tri_body = self.body[self.F[:, 0]]
        self.t_sim = 0.0

    def _load_faces(self) -> np.ndarray:
        faces = None
        try:
            root = Path(__file__).resolve().parent.parent / "assets"
            topo = json.loads((root / "panel_topology.json").read_text(encoding="utf-8"))
            faces = np.array(topo["faces"], dtype=np.int64)
        except Exception:
            faces = None
        if faces is None or len(faces) == 0:
            f = []
            for i in range(self.ny - 1):
                for j in range(self.nx - 1):
                    a = i * self.nx + j
                    b = a + 1
                    c = a + self.nx
                    d = c + 1
                    f.append((a, b, d)); f.append((a, d, c))
            faces = np.array(f, dtype=np.int64)
        return np.vstack([faces + p * self.nper for p in range(self.n_panels)])

    # ------------------------------------------------------- broad phase
    def _candidates(self, X, R):
        """Node/triangle candidate pairs from a uniform xy hash grid."""
        T = X[self.F]
        mn = T.min(axis=1)
        mx = T.max(axis=1)
        cell, K = self.cell, self.KKEY
        i0 = np.floor((mn[:, 0] - R) / cell).astype(np.int64)
        i1 = np.floor((mx[:, 0] + R) / cell).astype(np.int64)
        j0 = np.floor((mn[:, 1] - R) / cell).astype(np.int64)
        j1 = np.floor((mx[:, 1] + R) / cell).astype(np.int64)
        si = i1 - i0 + 1
        sj = j1 - j0 + 1
        cnt = si * sj
        tri_rep = np.repeat(np.arange(len(self.F)), cnt)
        tot = int(cnt.sum())
        if tot == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        off = np.arange(tot) - np.repeat(starts, cnt)
        sjr = np.repeat(sj, cnt)
        di = off // sjr
        dj = off % sjr
        key = (i0[tri_rep] + di) * K + (j0[tri_rep] + dj)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        ni = np.floor(X[:, 0] / cell).astype(np.int64)
        nj = np.floor(X[:, 1] / cell).astype(np.int64)
        nkey = ni * K + nj
        lo = np.searchsorted(sk, nkey, "left")
        hi = np.searchsorted(sk, nkey, "right")
        cn = hi - lo
        total = int(cn.sum())
        if total == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e
        gstart = np.cumsum(cn) - cn
        idx = np.arange(total) - np.repeat(gstart, cn) + np.repeat(lo, cn)
        cand = order[idx]
        node_rep = np.repeat(np.arange(len(X)), cn)
        keep = self.body[node_rep] != self.tri_body[cand]
        z = X[node_rep, 2]
        keep &= (z >= mn[cand, 2] - R) & (z <= mx[cand, 2] + R)
        return node_rep[keep], cand[keep]

    # ------------------------------------------------------------- stepping
    def _substep(self, h):
        X, V, m = self.X, self.V, self.m
        n = len(X)
        f = np.zeros_like(X)
        f[:, 2] += m * self.gz
        # aerodynamic drag
        f -= (self.kv * m)[:, None] * V

        # ---- internal elasticity (springs) ----
        xa = X[self.sa]
        xb = X[self.sb]
        d = xa - xb
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        u_ = d / L[:, None]
        rv = V[self.sa] - V[self.sb]
        fs = (self.ks * (L - self.L0)
              + self.cs * np.einsum("ij,ij->i", rv, u_))[:, None] * u_
        for c in range(3):
            f[:, c] += (np.bincount(self.sa, fs[:, c], minlength=n)
                        - np.bincount(self.sb, fs[:, c], minlength=n))

        # ---- panel-panel penalty contact + Coulomb friction ----
        node_rep, cand = self._candidates(X, self.R)
        if len(node_rep):
            p = X[node_rep]
            a = X[self.F[cand, 0]]
            b = X[self.F[cand, 1]]
            cc_ = X[self.F[cand, 2]]
            cp, dist, u, v = _point_tri_closest(p, a, b, cc_)
            pen = self.thickness - dist
            act = pen > 0.0
            if np.any(act):
                nr = node_rep[act]
                cn = cand[act]
                dn = np.maximum(dist[act], 1e-9)
                nhat = (p[act] - cp[act]) / dn[:, None]
                w0 = 1.0 - u[act] - v[act]
                w1 = u[act]
                w2 = v[act]
                vtri = (V[self.F[cn, 0]] * w0[:, None]
                        + V[self.F[cn, 1]] * w1[:, None]
                        + V[self.F[cn, 2]] * w2[:, None])
                vrel = V[nr] - vtri
                vn = np.einsum("ij,ij->i", vrel, nhat)
                Nc = self.kc * pen[act] + self.cc * np.maximum(0.0, -vn)
                vt = vrel - vn[:, None] * nhat
                vtm = np.linalg.norm(vt, axis=1)
                Fmag = np.minimum(self.mu * Nc, m[nr] * vtm / h)
                Fu = np.where(vtm > 1e-9, Fmag / np.maximum(vtm, 1e-12), 0.0)
                Ft = Nc[:, None] * nhat - Fu[:, None] * vt
                for c in range(3):
                    fv = Ft[:, c]
                    f[:, c] += np.bincount(nr, fv, minlength=n)
                    f[:, c] -= np.bincount(self.F[cn, 0], fv * w0, minlength=n)
                    f[:, c] -= np.bincount(self.F[cn, 1], fv * w1, minlength=n)
                    f[:, c] -= np.bincount(self.F[cn, 2], fv * w2, minlength=n)
                # deep penetration backstop: project node out of the triangle
                deep = pen[act] > 0.8 * self.thickness
                if np.any(deep):
                    dr = nr[deep]
                    X[dr] = cp[deep] + nhat[deep] * (0.8 * self.thickness)
                    vn_d = np.einsum("ij,ij->i", V[dr], nhat[deep])
                    V[dr] -= np.minimum(vn_d, 0.0)[:, None] * nhat[deep]

        # ---- board contact (nodes are spheres of radius thickness/2) ----
        gap = X[:, 2] - self.zrest
        actb = gap < 0.0
        if np.any(actb):
            ib = np.nonzero(actb)[0]
            Nb = self.kc * (-gap[ib]) + self.cc * np.maximum(0.0, -V[ib, 2])
            f[ib, 2] += Nb
            vt = V[ib, :2]
            vtm = np.linalg.norm(vt, axis=1)
            Fmag = np.minimum(self.mu * Nb, m[ib] * vtm / h)
            Fu = np.where(vtm > 1e-9, Fmag / np.maximum(vtm, 1e-12), 0.0)
            f[ib, :2] -= Fu[:, None] * vt

        # ---- semi-implicit Euler ----
        V += (f / m[:, None]) * h
        X += V * h
        # hard board backstop (mid-surface cannot dive into the board)
        np.maximum(X[:, 2], self.board_top + 0.3 * self.thickness, out=X[:, 2])
        self._nsub += 1

    # ------------------------------------------------------------- sampling
    def _signed_gaps(self):
        """Per-node signed contact gap (negative = penetration)."""
        X = self.X
        csd = X[:, 2] - self.zrest
        node_rep, cand = self._candidates(X, self.R_csd)
        if len(node_rep):
            p = X[node_rep]
            a = X[self.F[cand, 0]]
            b = X[self.F[cand, 1]]
            c = X[self.F[cand, 2]]
            _, dist, _, _ = _point_tri_closest(p, a, b, c)
            g = dist - self.thickness
            mn = np.full(len(X), np.inf)
            np.minimum.at(mn, node_rep, g)
            csd = np.minimum(csd, mn)
        return csd

    def _frame(self, time):
        csd = self._signed_gaps()
        pen = max(0.0, -float(np.min(csd)))
        fields = {
            "contact_signed_distance": np.asarray(csd, dtype=float),
            "constraint_value": np.array([pen], dtype=float),
            "constraint_target": np.array([0.0], dtype=float),
            "cumulative_mass_source": np.array([0.0], dtype=float),
        }
        attributes = {
            "uv": self.uv.copy(),
            "mass_quantity": self.mq.copy(),
        }
        scalars = {
            "contact_length_scale": float(self.thickness),
            "constraint_scale": float(self.constraint_scale),
        }
        return StateFrame(
            float(time),
            self.X.copy(),
            self.V.copy(),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t_sim < target - 1e-9 and guard < 10_000_000:
            h = min(self.dt_sub, target - self.t_sim)
            self._substep(h)
            self.t_sim += h
            guard += 1
        if not (np.isfinite(self.X).all() and np.isfinite(self.V).all()):
            raise RuntimeError("non-finite state encountered")
        return self._frame(target)

    def diagnostics(self) -> dict:
        csd = self._signed_gaps()
        return {
            "time": float(self.t_sim),
            "substeps": int(self._nsub),
            "min_signed_gap": float(np.min(csd)),
            "max_z": float(np.max(self.X[:, 2])),
        }

