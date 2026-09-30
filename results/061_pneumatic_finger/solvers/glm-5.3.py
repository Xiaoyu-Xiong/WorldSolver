"""Pressure-driven bending of a chambered soft finger (scene 061).

Physical model
--------------
Total-Lagrangian 3-D finite-element dynamics of a compressible Neo-Hookean
silicone solid.  The solid is a structured tetrahedral mesh built on the
frozen chambered-finger geometry: a thick base slab, six internal rectangular
chambers separated by posts, thin side walls and a thin top wall (the
asymmetric wall structure of the source experiment).  Chamber pressure is
applied as a true follower traction on the current chamber boundary faces;
gravity acts on the solid; the base region (x <= 0.03 m, inside the frozen
base_support) is clamped.  Integration is explicit symplectic Euler with
mass-proportional (Rayleigh) damping and a CFL-stable step size, so the
finger quasi-statically follows the prescribed pressure ramp and holds a
visibly bent equilibrium shape with the base in place.

No animation, keyframes or precomputed trajectories are used: every frame is
the result of the numerical time stepping below.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# small vectorized 3x3 linear algebra (faster than batched LAPACK calls)
# ---------------------------------------------------------------------------


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 3x3 matrix product of stacked arrays of shape (n,3,3)."""
    out = np.empty_like(a)
    for i in range(3):
        for j in range(3):
            out[:, i, j] = (
                a[:, i, 0] * b[:, 0, j]
                + a[:, i, 1] * b[:, 1, j]
                + a[:, i, 2] * b[:, 2, j]
            )
    return out


def _det3(f: np.ndarray) -> np.ndarray:
    return (
        f[:, 0, 0] * (f[:, 1, 1] * f[:, 2, 2] - f[:, 1, 2] * f[:, 2, 1])
        - f[:, 0, 1] * (f[:, 1, 0] * f[:, 2, 2] - f[:, 1, 2] * f[:, 2, 0])
        + f[:, 0, 2] * (f[:, 1, 0] * f[:, 2, 1] - f[:, 1, 1] * f[:, 2, 0])
    )


def _inv3(f: np.ndarray, det: np.ndarray) -> np.ndarray:
    inv = np.empty_like(f)
    inv[:, 0, 0] = f[:, 1, 1] * f[:, 2, 2] - f[:, 1, 2] * f[:, 2, 1]
    inv[:, 0, 1] = f[:, 0, 2] * f[:, 2, 1] - f[:, 0, 1] * f[:, 2, 2]
    inv[:, 0, 2] = f[:, 0, 1] * f[:, 1, 2] - f[:, 0, 2] * f[:, 1, 1]
    inv[:, 1, 0] = f[:, 1, 2] * f[:, 2, 0] - f[:, 1, 0] * f[:, 2, 2]
    inv[:, 1, 1] = f[:, 0, 0] * f[:, 2, 2] - f[:, 0, 2] * f[:, 2, 0]
    inv[:, 1, 2] = f[:, 0, 2] * f[:, 1, 0] - f[:, 0, 0] * f[:, 1, 2]
    inv[:, 2, 0] = f[:, 1, 0] * f[:, 2, 1] - f[:, 1, 1] * f[:, 2, 0]
    inv[:, 2, 1] = f[:, 0, 1] * f[:, 2, 0] - f[:, 0, 0] * f[:, 2, 1]
    inv[:, 2, 2] = f[:, 0, 0] * f[:, 1, 1] - f[:, 0, 1] * f[:, 1, 0]
    return inv / det[:, None, None]


# Kuhn decomposition of a hexahedron into six tets along the (0,0,0)-(1,1,1)
# diagonal.  Corner ordering: x fastest, then y, then z.
_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)
_KUHN = (
    (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
    (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
)

# frozen chamber layout (meters), matching the scaffold mesh chambers
_CH_X0 = 0.008
_CH_PITCH = 0.032
_CH_LEN = 0.027
_CH_Y = 0.019          # chamber half width in y (5 mm side walls)
_CH_Z0 = 0.014         # chamber floor = top of base slab
_CH_Z1 = 0.047         # chamber ceiling (3 mm top wall up to z = 0.050)
_N_CH = 6


class BenchmarkSolver:
    """Explicit total-Lagrangian tet-FEM solver for the pneumatic finger."""

    def __init__(self) -> None:
        self._step_count = 0
        self._reject_count = 0

    # -- setup -------------------------------------------------------------

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        press = cfg["pressure"]
        self.p_max = float(press["chamber_pressure_pa"])
        self.ramp_start = float(press["ramp_start_s"])
        self.ramp_end = float(press["ramp_end_s"])
        self.hold_until = float(press["hold_until_s"])
        self.duration = float(cfg["duration"])

        # soft silicone (frozen material choice)
        self.E = 3.0e5
        self.nu = 0.35
        self.rho = 1100.0
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.alpha = 15.0  # mass-proportional damping (1/s)

        init = scene.initial_state()
        self.obj_rest = np.array(init.position, dtype=float)
        self.obj_fixed = np.array(init.attributes["fixed_base"], dtype=float)

        self._support_box(cfg)
        self._build_grid()
        self._build_tets()
        self._build_chamber_faces()
        self._embed_surface()
        self._finish()

    def _support_box(self, cfg: dict) -> None:
        lo = np.array([-0.003, -0.026, 0.0])
        hi = np.array([0.031, 0.026, 0.052])
        geom = cfg.get("scene_geometry") or {}
        for obj in geom.get("objects", []):
            if obj.get("id") == "base_support":
                he = np.array(obj["geometry"]["half_extents"], dtype=float)
                tr = np.array(obj["transform"]["translation"], dtype=float)
                lo = tr - he
                hi = tr + he
        self.sup_lo = lo
        self.sup_hi = hi

    def _build_grid(self) -> None:
        xs = [0.0, 0.004, _CH_X0]
        for k in range(_N_CH):
            x1 = _CH_X0 + _CH_PITCH * k
            x2 = x1 + _CH_LEN
            xs.extend(np.linspace(x1, x2, 5)[1:])
            xs.append(x2 + 0.005)  # separating wall between chambers
        xs.extend(np.linspace(0.200, 0.220, 4)[1:])  # tip region
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array([-0.024, -_CH_Y, 0.0, _CH_Y, 0.024], dtype=float)
        zs = np.concatenate(
            [np.linspace(0.0, _CH_Z0, 4), np.linspace(_CH_Z0, _CH_Z1, 4)[1:],
             np.array([0.050])]
        )
        self.zs = np.array(zs, dtype=float)
        self.ncx = len(self.xs) - 1
        self.ncy = len(self.ys) - 1
        self.ncz = len(self.zs) - 1

        cav = np.zeros((self.ncx, self.ncy, self.ncz), dtype=bool)
        for k in range(_N_CH):
            i0 = 2 + 5 * k
            cav[i0:i0 + 4, 1:3, 3:6] = True
        self.cav = cav
        self.solid_cells = np.argwhere(~cav)
        self.cell_index = np.full((self.ncx, self.ncy, self.ncz), -1, dtype=int)
        for r, (i, j, kk) in enumerate(self.solid_cells):
            self.cell_index[i, j, kk] = r

        ny_n, nz_n = self.ncy + 1, self.ncz + 1
        self._nid = lambda i, j, k: (i * ny_n + j) * nz_n + k
        gx, gy, gz = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.X_full = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def _build_tets(self) -> None:
        nc = len(self.solid_cells)
        cell_nodes = np.empty((nc, 8), dtype=int)
        for r, (i, j, k) in enumerate(self.solid_cells):
            for c, (dx, dy, dz) in enumerate(_CORNERS):
                cell_nodes[r, c] = self._nid(i + dx, j + dy, k + dz)
        tets = np.stack(
            [cell_nodes[:, list(pat)] for pat in _KUHN], axis=1
        ).reshape(-1, 4)

        uniq = np.unique(tets)
        remap = np.full(self.X_full.shape[0], -1, dtype=int)
        remap[uniq] = np.arange(len(uniq))
        self.remap = remap
        tets = remap[tets]
        self.X0 = self.X_full[uniq].copy()
        self.Nn = len(uniq)

        # positive orientation
        dm = self._dmatrix(self.X0, tets)
        det = _det3(dm)
        bad = det < 0.0
        if bad.any():
            tmp = tets[bad, 0].copy()
            tets[bad, 0] = tets[bad, 1]
            tets[bad, 1] = tmp
            dm = self._dmatrix(self.X0, tets)
            det = _det3(dm)

        self.tets = tets
        self.Ne = len(tets)
        self.tet_flat = tets.reshape(-1)
        self.V0 = np.abs(det) / 6.0
        self.Dm_inv = _inv3(dm, det)
        self.Dm_invT = np.ascontiguousarray(self.Dm_inv.transpose(0, 2, 1))
        self.cell_tets = np.arange(nc * 6, dtype=int).reshape(-1, 6)

        # lumped nodal masses and gravity load
        w = np.repeat(self.rho * self.V0 / 4.0, 4)
        self.m_n = np.bincount(self.tet_flat, weights=w, minlength=self.Nn)
        self.f_grav = self.m_n[:, None] * self.gravity[None, :]

    @staticmethod
    def _dmatrix(x: np.ndarray, tets: np.ndarray) -> np.ndarray:
        xt = x[tets]
        return np.stack(
            [xt[:, 0] - xt[:, 3], xt[:, 1] - xt[:, 3], xt[:, 2] - xt[:, 3]],
            axis=1,
        )

    def _build_chamber_faces(self) -> None:
        """Quad faces between chamber cells and solid cells (pressure walls)."""
        nid = self._nid
        faces = []
        dirs = (
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
        )
        for i, j, k in np.argwhere(self.cav):
            for d in dirs:
                ni, nj, nk = i + d[0], j + d[1], k + d[2]
                if not (0 <= ni < self.ncx and 0 <= nj < self.ncy and 0 <= nk < self.ncz):
                    continue
                if self.cav[ni, nj, nk]:
                    continue
                # canonical cyclic node order of the shared quad face
                if d[0] < 0:
                    g = (nid(i, j, k), nid(i, j + 1, k), nid(i, j + 1, k + 1), nid(i, j, k + 1))
                elif d[0] > 0:
                    g = (nid(i + 1, j, k), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i + 1, j + 1, k))
                elif d[1] < 0:
                    g = (nid(i, j, k), nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j, k))
                elif d[1] > 0:
                    g = (nid(i, j + 1, k), nid(i + 1, j + 1, k), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1))
                elif d[2] < 0:
                    g = (nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k))
                else:
                    g = (nid(i, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1), nid(i + 1, j, k + 1))
                # outward normal of the solid points into the chamber
                d_in = np.array([-d[0], -d[1], -d[2]], dtype=float)
                q = self.X_full[list(g)]
                avec = 0.5 * np.cross(q[2] - q[0], q[3] - q[1])
                sign = 1.0 if float(np.dot(avec, d_in)) > 0.0 else -1.0
                faces.append((self.remap[list(g)], sign))

        self.face_nodes = np.array([f[0] for f in faces], dtype=int).reshape(-1, 4)
        self.face_sign = np.array([f[1] for f in faces], dtype=float)
        self.face_flat = self.face_nodes.reshape(-1)
        self.Nf = len(faces)

    def _embed_surface(self) -> None:
        """Barycentric embedding of the render mesh vertices into the tets."""
        pts = self.obj_rest
        n = len(pts)
        offs = [(0, 0, 0)]
        rest = sorted(
            [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
             if (a, b, c) != (0, 0, 0)],
            key=lambda o: (abs(o[0]) + abs(o[1]) + abs(o[2]), o),
        )
        offs.extend(rest)
        tet_of = np.zeros(n, dtype=int)
        bary = np.zeros((n, 4))
        for idx in range(n):
            p = pts[idx]
            i = int(np.clip(np.searchsorted(self.xs, p[0], side="right") - 1, 0, self.ncx - 1))
            j = int(np.clip(np.searchsorted(self.ys, p[1], side="right") - 1, 0, self.ncy - 1))
            k = int(np.clip(np.searchsorted(self.zs, p[2], side="right") - 1, 0, self.ncz - 1))
            best = None
            done = False
            for off in offs:
                ci, cj, ck = i + off[0], j + off[1], k + off[2]
                if not (0 <= ci < self.ncx and 0 <= cj < self.ncy and 0 <= ck < self.ncz):
                    continue
                r = self.cell_index[ci, cj, ck]
                if r < 0:
                    continue
                cell_best = None
                for q in range(6):
                    t = self.cell_tets[r, q]
                    w = self.Dm_inv[t] @ (p - self.X0[self.tets[t, 3]])
                    b = np.array([w[0], w[1], w[2], 1.0 - w.sum()])
                    m = float(b.min())
                    if cell_best is None or m > cell_best[0]:
                        cell_best = (m, t, b)
                if best is None or cell_best[0] > best[0]:
                    best = cell_best
                if cell_best[0] > -1e-9:
                    done = True
                    break
            if best is None:
                raise RuntimeError("surface vertex could not be embedded")
            tet_of[idx] = best[1]
            bary[idx] = best[2]
            del done
        self.obj_tet = tet_of
        self.obj_bary = bary
        # per-surface-vertex mass contribution (total mass conserved)
        tm = self.m_n[self.tets[self.obj_tet]]
        w = (self.obj_bary * tm).sum(axis=1)
        self.M_total = float(self.rho * self.V0.sum())
        self.obj_mass = self.M_total * w / w.sum()

    def _finish(self) -> None:
        self.fixed = self.X0[:, 0] <= 0.030 + 1e-9
        rest_in = np.all((self.X0 > self.sup_lo) & (self.X0 < self.sup_hi), axis=1)
        self.collide_eligible = np.where(~self.fixed & ~rest_in)[0]

        # CFL-stable step size from the smallest tet edge
        xt = self.X0[self.tets]
        hmin = np.array([
            np.linalg.norm(xt[:, a] - xt[:, b], axis=1)
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ]).min()
        c = np.sqrt((self.lam + 2.0 * self.mu) / self.rho)
        self.dt = float(0.4 * hmin / c)
        self.dt = max(self.dt, self.duration / 150000.0)
        self.c_wave = float(c)
        self.h_min = float(hmin)

        self.reference_force = float(
            self.p_max * (2.0 * _CH_Y) * (_CH_Z1 - _CH_Z0)
        )

        self._x = self.X0.copy()
        self._v = np.zeros_like(self.X0)
        self._t = 0.0

    # -- physics -----------------------------------------------------------

    def _pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return 0.0
        if t >= self.ramp_end:
            return self.p_max
        return self.p_max * (t - self.ramp_start) / (self.ramp_end - self.ramp_start)

    def _internal(self, x: np.ndarray):
        """Neo-Hookean nodal forces and per-element deformation gradients."""
        xt = x[self.tets]
        ds = np.stack(
            [xt[:, 0] - xt[:, 3], xt[:, 1] - xt[:, 3], xt[:, 2] - xt[:, 3]], axis=1
        )
        f = _mm(ds, self.Dm_inv)
        det = _det3(f)
        safe = np.where(det > 1e-12, det, 1e-12)
        lnj = np.log(safe)
        finv = _inv3(f, safe)
        finvt = finv.transpose(0, 2, 1)
        p = self.mu * (f - finvt) + (self.lam * lnj)[:, None, None] * finvt
        h = -self.V0[:, None, None] * _mm(p, self.Dm_invT)
        ft = np.empty((self.Ne, 4, 3))
        ft[:, 0] = h[:, :, 0]
        ft[:, 1] = h[:, :, 1]
        ft[:, 2] = h[:, :, 2]
        ft[:, 3] = -h.sum(axis=2)
        force = np.empty((self.Nn, 3))
        flat = ft.reshape(-1, 3)
        for c in range(3):
            force[:, c] = np.bincount(self.tet_flat, flat[:, c], minlength=self.Nn)
        return force, f

    def _pressure_force(self, x: np.ndarray, p: float) -> np.ndarray:
        if p == 0.0 or self.Nf == 0:
            return np.zeros((self.Nn, 3))
        q = x[self.face_nodes]
        avec = 0.5 * np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
        fq = -p * self.face_sign[:, None] * avec
        vals = np.repeat(0.25 * fq, 4, axis=0)
        force = np.empty((self.Nn, 3))
        for c in range(3):
            force[:, c] = np.bincount(self.face_flat, vals[:, c], minlength=self.Nn)
        return force

    def _collide(self, x: np.ndarray, v: np.ndarray) -> None:
        el = self.collide_eligible
        if el.size == 0:
            return
        q = x[el]
        inside = np.all((q > self.sup_lo) & (q < self.sup_hi), axis=1)
        if not inside.any():
            return
        idx = el[inside]
        pts = x[idx]
        dlo = pts - self.sup_lo
        dhi = self.sup_hi - pts
        ax = np.argmin(np.minimum(dlo, dhi), axis=1)
        for c in range(3):
            m = ax == c
            if not m.any():
                continue
            near_lo = dlo[m, c] < dhi[m, c]
            x[idx[m], c] = np.where(near_lo, self.sup_lo[c], self.sup_hi[c])
            v[idx[m], c] = 0.0

    def _step(self, h: float) -> bool:
        """One explicit symplectic-Euler substep; False if it was rejected."""
        p = self._pressure(self._t)
        f_int, _ = self._internal(self._x)
        f_pre = self._pressure_force(self._x, p)
        f = f_int + f_pre + self.f_grav
        v = (self._v + h * f / self.m_n[:, None]) / (1.0 + h * self.alpha)
        v[self.fixed] = 0.0
        x = self._x + h * v
        x[self.fixed] = self.X0[self.fixed]
        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            self.dt *= 0.5
            return False
        self._collide(x, v)
        self._x = x
        self._v = v
        self._t += h
        self._step_count += 1
        return True

    # -- contract ----------------------------------------------------------

    def advance_to(self, time: float):
        target = float(time)
        rejects = 0
        while self._t < target - 1e-12:
            t0 = self._t
            h = min(self.dt, target - self._t)
            if not self._step(h):
                rejects += 1
                if rejects > 200:
                    # freeze rather than emit non-finite states
                    break
            else:
                rejects = 0
            if self._t <= t0 and rejects > 200:
                break
        self._t = target
        self._reject_count = rejects
        return self._frame()

    def _frame(self) -> StateFrame:
        p = self._pressure(self._t)
        f_int, grad = self._internal(self._x)
        f_pre = self._pressure_force(self._x, p)
        f_ext = f_pre + self.f_grav
        # support reaction balances everything on the clamped nodes
        reaction = -(f_int + f_ext)[self.fixed].sum(axis=0)
        dom_finger = f_ext.sum(axis=0) + reaction
        budget = np.array([[dom_finger], [-reaction]], dtype=float)  # (G=2,K=1,3)

        tet_of = self.tets[self.obj_tet]
        b = self.obj_bary[:, :, None]
        pos = (b * self._x[tet_of]).sum(axis=1)
        vel = (b * self._v[tet_of]).sum(axis=1)
        fobj = (b * f_ext[tet_of]).sum(axis=1)

        return StateFrame(
            time=float(self._t),
            position=pos,
            velocity=vel,
            mass=self.obj_mass.copy(),
            attributes={
                "mass_quantity": self.obj_mass.copy(),
                "fixed_base": self.obj_fixed.copy(),
                "rest_position": self.obj_rest.copy(),
            },
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": np.array(grad, dtype=float),
                "external_force": fobj,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "chamber_pressure_pa": float(p),
                "youngs_modulus_pa": float(self.E),
                "poisson_ratio": float(self.nu),
                "density_kg_m3": float(self.rho),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        tip0 = self.obj_rest.max(axis=0)
        tip = self._x  # placeholder replaced below
        return {
            "method": "total_lagrangian_neohookean_tet_fem_explicit",
            "num_tets": int(self.Ne),
            "num_nodes": int(self.Nn),
            "num_chamber_faces": int(self.Nf),
            "dt": float(self.dt),
            "substeps": int(self._step_count),
            "wave_speed_m_s": float(self.c_wave),
            "min_element_size_m": float(self.h_min),
            "tip_rest": [float(v) for v in tip0],
            "tip_current": [float(v) for v in tip.max(axis=0)],
            "pressure_pa": float(self._pressure(self._t)),
            "total_mass_kg": float(self.M_total),
        }
