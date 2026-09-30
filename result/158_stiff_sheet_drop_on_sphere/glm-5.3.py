"""Stiff thin-sheet drop on a fixed sphere.

Physics: XPBD (extended position based dynamics) mass-sheet solver.

- Stretch constraints (grid edges) with ~zero compliance -> nearly
  inextensible stiff membrane.
- Shear constraints (cell diagonals).
- Bending constraints (skip-2 and skip-3 grid neighbors) with small
  compliance -> high bending stiffness so the sheet forms sharp folds
  instead of draping softly.
- Rigid obstacle contacts: fixed sphere and ground box, projected each
  constraint iteration (no penetration).
- Vertex-vertex self contact with a minimum separation distance.
- Explicit velocity level friction on contacts and mild air drag so the
  folded configuration settles.

Everything is stepped with a fixed substep (2000 Hz) explicit
symplectic-Euler predictor followed by constraint projection; velocities
are recovered from the position change (standard PBD).  No keyframes, no
precomputed motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- numerical parameters ------------------------------------------------
    SUBSTEP = 5.0e-4       # physics substep (s)
    ITERS = 8              # constraint iterations per substep
    VMAX = 10.0            # velocity clamp (m/s)
    DRAG = 0.15            # global air drag rate (1/s)
    FRICTION_RATE = 1.5    # tangential contact damping rate (1/s)
    CONTACT_EPS = 0.01     # near-contact band for friction (m)
    SELF_DIST = 0.035      # vertex-vertex self separation (m)
    NEAR_MARGIN = 0.2      # near-contact reporting margin (m)

    # XPBD compliances (m / N)
    A_STRETCH = 0.0
    A_SHEAR = 1.0e-5
    A_BEND2 = 5.0e-5
    A_BEND3 = 2.0e-4

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame0.mass, dtype=float)
        self.w = 1.0 / self.mass
        self.n = len(self.x)
        self.t = 0.0

        self.g = np.array(cfg["gravity"], dtype=float)
        self.thickness = float(cfg["cloth"]["thickness"])

        nx, ny = (int(v) for v in cfg["cloth"]["grid_shape"])
        self.nx, self.ny = nx, ny

        # fixed obstacles
        self.c_s = np.array([0.0, 0.0, 0.68])
        self.R = 0.62
        self.c_g = np.array([0.0, 0.0, -0.05])
        self.h_g = np.array([2.4, 2.0, 0.05])

        # deterministic symmetry-breaking perturbation (stiff sheets choose
        # fold directions from imperfections)
        rng = np.random.default_rng(self.seed)
        self.x[:, 2] += rng.uniform(-2.0e-4, 2.0e-4, self.n)

        # --- constraint graph -------------------------------------------------
        def idx(i, j):
            return j * nx + i

        groups: dict[float, list] = {}

        def add(a, b, alpha):
            groups.setdefault(alpha, []).append((a, b))

        for j in range(ny):
            for i in range(nx):
                a = idx(i, j)
                if i + 1 < nx:
                    add(a, idx(i + 1, j), self.A_STRETCH)
                if j + 1 < ny:
                    add(a, idx(i, j + 1), self.A_STRETCH)
                if i + 1 < nx and j + 1 < ny:
                    add(a, idx(i + 1, j + 1), self.A_SHEAR)
                    add(idx(i + 1, j), idx(i, j + 1), self.A_SHEAR)
                if i + 2 < nx:
                    add(a, idx(i + 2, j), self.A_BEND2)
                if j + 2 < ny:
                    add(a, idx(i, j + 2), self.A_BEND2)
                if i + 3 < nx:
                    add(a, idx(i + 3, j), self.A_BEND3)
                if j + 3 < ny:
                    add(a, idx(i, j + 3), self.A_BEND3)

        self.groups = []
        for alpha, pairs in groups.items():
            ia = np.array([p[0] for p in pairs], dtype=np.int64)
            ib = np.array([p[1] for p in pairs], dtype=np.int64)
            d = self.x[ib] - self.x[ia]
            rest = np.sqrt((d * d).sum(1))
            self.groups.append((ia, ib, rest, float(alpha)))

        # --- triangles (for deformation gradient) ------------------------------
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i, j + 1)
                d = idx(i + 1, j + 1)
                tris.append((a, b, c))
                tris.append((b, d, c))
        self.tris = np.array(tris, dtype=np.int64)

        # material (reference) coordinates
        self.mat2 = self.x[:, :2].copy()

        # --- self-collision neighbor mask --------------------------------------
        gi = np.arange(self.n) % nx
        gj = np.arange(self.n) // nx
        di = np.abs(gi[:, None] - gi[None, :])
        dj = np.abs(gj[:, None] - gj[None, :])
        self.allowed = (di <= 2) & (dj <= 2)
        np.fill_diagonal(self.allowed, True)

        # carry-through attributes from the scaffold initial frame
        self.boundary = np.asarray(
            frame0.attributes.get("boundary_vertex",
                                  np.zeros(self.n, dtype=np.int32))
        ).astype(np.int32)

        self._steps = 0

    # ------------------------------------------------------------------ stepping
    def _project_groups(self, dt2: float) -> None:
        x, w = self.x, self.w
        for ia, ib, rest, alpha in self.groups:
            d = x[ib] - x[ia]
            L = np.sqrt((d * d).sum(1))
            L = np.maximum(L, 1e-12)
            C = L - rest
            denom = w[ia] + w[ib] + alpha / dt2
            s = -C / denom
            dhat = d / L[:, None]
            ca = (w[ia] * s)[:, None] * dhat
            # accumulate: index arrays contain duplicates (a vertex is the
            # first endpoint of several constraints)
            np.add.at(x, ia, ca)
            np.add.at(x, ib, -ca)

    def _collide_obstacles(self) -> None:
        # sphere
        rel = self.x - self.c_s
        d = np.sqrt((rel * rel).sum(1))
        pen = d < self.R
        if pen.any():
            dd = np.maximum(d[pen], 1e-12)
            self.x[pen] = self.c_s + rel[pen] / dd[:, None] * self.R
        # ground box
        q = np.abs(self.x - self.c_g) - self.h_g
        inside = (q < 0.0).all(axis=1)
        if inside.any():
            rows = np.nonzero(inside)[0]
            ax = np.argmax(q[rows], axis=1)
            sgn = np.sign(self.x[rows] - self.c_g[None, :])
            k = np.arange(len(rows))
            target = self.c_g[ax] + sgn[k, ax] * self.h_g[ax]
            self.x[rows, ax] = target

    def _self_collide(self) -> None:
        diff = self.x[None, :, :] - self.x[:, None, :]
        dist = np.sqrt((diff * diff).sum(-1))
        mask = (~self.allowed) & (dist < self.SELF_DIST) & (dist > 1e-9)
        pi, pj = np.nonzero(mask)
        if pi.size:
            dhat = diff[pi, pj] / dist[pi, pj][:, None]
            corr = 0.5 * (self.SELF_DIST - dist[pi, pj])[:, None] * dhat
            np.add.at(self.x, pi, corr)
            np.add.at(self.x, pj, -corr)

    def _sphere_gap(self):
        rel = self.x - self.c_s
        d = np.sqrt((rel * rel).sum(1))
        return d - self.R, d

    def _ground_gap(self):
        q = np.abs(self.x - self.c_g) - self.h_g
        outside = np.sqrt((np.maximum(q, 0.0) ** 2).sum(1))
        return outside + np.minimum(q.max(axis=1), 0.0), q

    def _apply_friction(self, dt: float) -> None:
        f = np.exp(-self.FRICTION_RATE * dt)
        # sphere contact
        gs, d = self._sphere_gap()
        ms = gs < self.CONTACT_EPS
        if ms.any():
            n = np.zeros((int(ms.sum()), 3))
            rel = self.x[ms] - self.c_s
            dd = np.maximum(d[ms], 1e-12)
            n = rel / dd[:, None]
            v = self.v[ms]
            vn = (v * n).sum(1, keepdims=True) * n
            self.v[ms] = vn + (v - vn) * f
        # ground contact
        gg, q = self._ground_gap()
        mg = gg < self.CONTACT_EPS
        if mg.any():
            ax = np.argmax(q[mg], axis=1)
            sgn = np.sign(self.x[mg] - self.c_g[None, :])
            n = np.zeros((int(mg.sum()), 3))
            k = np.arange(len(ax))
            n[k, ax] = sgn[k, ax]
            v = self.v[mg]
            vn = (v * n).sum(1, keepdims=True) * n
            self.v[mg] = vn + (v - vn) * f

    def _step(self, dt: float) -> None:
        self.v = self.v + self.g[None, :] * dt
        self.v *= np.exp(-self.DRAG * dt)
        xp = self.x.copy()
        self.x = self.x + self.v * dt
        dt2 = dt * dt
        for it in range(self.ITERS):
            self._project_groups(dt2)
            self._collide_obstacles()
            if it % 2 == 1:
                self._self_collide()
        self._self_collide()
        self._collide_obstacles()
        self.v = (self.x - xp) / dt
        self._apply_friction(dt)
        sp = np.sqrt((self.v * self.v).sum(1))
        over = sp > self.VMAX
        if over.any():
            self.v[over] *= (self.VMAX / sp[over])[:, None]
        self.t += dt
        self._steps += 1

    # ------------------------------------------------------------- output frames
    def _deformation_gradient(self) -> np.ndarray:
        t3 = self.tris
        pa, pb, pc = self.x[t3[:, 0]], self.x[t3[:, 1]], self.x[t3[:, 2]]
        ma, mb, mc = self.mat2[t3[:, 0]], self.mat2[t3[:, 1]], self.mat2[t3[:, 2]]

        e1w = pb - pa
        e2w = pc - pa
        e1m = mb - ma
        e2m = mc - ma

        l1 = np.maximum(np.sqrt((e1w * e1w).sum(1)), 1e-12)
        t1 = e1w / l1[:, None]
        nrm = np.cross(e1w, e2w)
        nl = np.sqrt((nrm * nrm).sum(1))
        ok = nl > 1e-12
        n = np.zeros_like(nrm)
        n[ok] = nrm[ok] / nl[ok, None]
        t2 = np.cross(n, t1)

        # deformed in-plane coordinates of the triangle corners
        ub = np.stack([l1, np.zeros_like(l1)], axis=1)          # (T,2)
        uc = np.stack([(e2w * t1).sum(1), (e2w * t2).sum(1)], axis=1)

        # material coordinate matrices and their inverses
        M = np.stack([e1m, e2m], axis=-1)                        # (T,2,2)
        a, b = M[:, 0, 0], M[:, 0, 1]
        c, d = M[:, 1, 0], M[:, 1, 1]
        det = a * d - b * c
        good = np.abs(det) > 1e-14
        Minv = np.zeros_like(M)
        Minv[:, 0, 0], Minv[:, 0, 1] = d, -b
        Minv[:, 1, 0], Minv[:, 1, 1] = -c, a
        Minv[good] /= det[good][:, None, None]

        U = np.stack([ub, uc], axis=-1)                          # (T,2,2)
        F = np.matmul(U, Minv)
        F[~good] = np.eye(2)

        # scatter-average to vertices
        Fv = np.zeros((self.n, 2, 2))
        cnt = np.zeros(self.n)
        np.add.at(Fv, t3.ravel(), np.repeat(F, 3, axis=0))
        np.add.at(cnt, t3.ravel(), 1.0)
        cnt = np.maximum(cnt, 1.0)
        Fv /= cnt[:, None, None]
        return Fv

    def _contact_signed_distance(self) -> np.ndarray:
        gs, _ = self._sphere_gap()
        gg, _ = self._ground_gap()
        # self-contact gaps (vertex-vertex against non-neighbors)
        diff = self.x[None, :, :] - self.x[:, None, :]
        dist = np.sqrt((diff * diff).sum(-1))
        masked = np.where(self.allowed, np.inf, dist)
        np.fill_diagonal(masked, np.inf)
        with np.errstate(invalid="ignore"):
            gself = masked.min(axis=1) - self.SELF_DIST
        gself[~np.isfinite(gself)] = np.inf
        gap = np.minimum(np.minimum(gs, gg), gself)
        sel = gap < self.NEAR_MARGIN
        csd = gap[sel]
        if csd.size == 0:
            csd = np.zeros(0)
        return csd

    def _frame(self, time: float) -> StateFrame:
        n = self.n
        ext = self.mass[:, None] * self.g[None, :]
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        t_target = float(time)
        span = t_target - self.t
        if span > 1e-9:
            m = int(np.ceil(span / self.SUBSTEP - 1e-9))
            m = max(m, 1)
            dt = span / m
            for _ in range(m):
                self._step(dt)
            self.t = t_target
        return self._frame(t_target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "num_vertices": int(self.n),
            "max_speed": float(np.sqrt((self.v ** 2).sum(1)).max())
            if self.n else 0.0,
        }
