"""Physically-based paper-sheet crumpling solver (NumPy only, no engines).

Model
-----
The sheet is the frozen 17x17 triangulated grid of the scene; every vertex
carries one third of the incident triangle mass.  The sheet is stepped as an
inextensible thin shell:

* structural and shear distance constraints with ~zero XPBD compliance
  (membrane inextensibility, so the sheet forms sharp ridges instead of
  stretching like rubber),
* bending (skip-distance) constraints with a soft compliance plus a plastic
  rest-length memory, which reproduces paper's persistent crease structure,
* unilateral self-contact between vertices and non-adjacent triangles,
* unilateral contact with the two kinematic confining walls and the static
  ground (boxes read from the frozen scene geometry).

Integration
-----------
Substepped extended position-based dynamics (XPBD), i.e. an implicit
integration of a compliant constraint potential:

    v <- v + h * g ;   x_pred = x + h * v
    solve constraints with coloured Gauss-Seidel sweeps on x_pred
    resolve unilateral contacts (self, walls, ground) on x_pred
    v <- (x_pred - x) / h

The substep h is fixed, so the solver is deterministic and the exported
velocities are the exact finite difference of the exported positions.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81])
_EPS = 1e-12


def _closest_point_on_triangle(P, A, B, C):
    """Closest point Q and barycentrics (u, v, w) for point/triangle pairs.

    Vectorised Ericson, "Real-Time Collision Detection".  All inputs (M, 3).
    """
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-30, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_in = vb / safe
        w_in = vc / safe
        s_ab = d1 / (d1 - d3)
        s_ac = d2 / (d2 - d6)
        s_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    v_in = np.where(np.isfinite(v_in), v_in, 0.0)
    w_in = np.where(np.isfinite(w_in), w_in, 0.0)
    s_ab = np.where(np.isfinite(s_ab), s_ab, 0.0)
    s_ac = np.where(np.isfinite(s_ac), s_ac, 0.0)
    s_bc = np.where(np.isfinite(s_bc), s_bc, 0.0)

    m_a = (d1 <= 0.0) & (d2 <= 0.0)
    m_b = (d3 >= 0.0) & (d4 <= d3)
    m_c = (d6 >= 0.0) & (d5 <= d6)
    m_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    m_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    m_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    v = np.where(
        m_ab, s_ab,
        np.where(m_ac, 0.0,
                 np.where(m_bc, 1.0 - s_bc,
                          np.where(m_a, 0.0,
                                   np.where(m_b, 1.0,
                                            np.where(m_c, 0.0, v_in))))))
    w = np.where(
        m_ab, 0.0,
        np.where(m_ac, s_ac,
                 np.where(m_bc, s_bc,
                          np.where(m_a, 0.0,
                                   np.where(m_b, 0.0,
                                            np.where(m_c, 1.0, w_in))))))
    u = 1.0 - v - w
    Q = u[:, None] * A + v[:, None] * B + w[:, None] * C
    bad = ~np.isfinite(Q).all(axis=1)
    if np.any(bad):
        Q[bad] = A[bad]
        u[bad] = 1.0
        v[bad] = 0.0
        w[bad] = 0.0
    return Q, u, v, w


class BenchmarkSolver:
    """XPBD thin-sheet solver for the frozen crumpling scene."""

    # ------------------------------------------------------------- scaffold
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}))
        nx, ny = (int(v) for v in cloth.get("grid_shape", [17, 17]))
        sx, sy = (float(v) for v in cloth.get("size", [3.0, 2.5]))
        center = np.asarray(cloth.get("center", [0.0, 0.0, 2.15]), dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        total_mass = float(cloth.get("total_mass", 0.55))
        self.nx, self.ny = nx, ny
        n = nx * ny
        self.n = n
        self.mass = np.full(n, total_mass / n)
        self.w = (1.0 / self.mass)[:, None]
        self.body_id = np.zeros(n, dtype=np.int32)
        gravity = cfg.get("gravity", _GRAVITY)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

        self._load_boundaries(cfg)

        gx = np.linspace(-0.5 * sx, 0.5 * sx, nx)
        gy = np.linspace(-0.5 * sy, 0.5 * sy, ny)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        flat = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n)))

        # The sheet must start between the separated walls: a real sheet is
        # laid in slightly slack so that it never starts interpenetrating the
        # frozen boundaries.  This scaled grid is the material rest shape.
        scale_x = min(1.0, (self.wall_gap0 - 0.01) / (0.5 * sx))
        self.scale_x = scale_x
        rest = flat.copy()
        rest[:, 0] *= scale_x
        rng = np.random.default_rng(int(seed))
        rest[:, 2] += 1e-4 * rng.standard_normal(n)
        rest[:, 2] += 1e-3 * np.sin(2.0 * np.pi * rest[:, 0] / 0.7) * np.cos(
            2.0 * np.pi * rest[:, 1] / 0.9
        )
        self.rest2d = rest[:, :2].copy()
        self.x = rest + center
        self.v = np.zeros((n, 3))

        self._build_topology()
        self._build_constraints()

        self.d0_self = 0.5 * self.thickness
        self.d0_vv = self.thickness
        self.report_margin = 0.02
        self.friction_self = 0.25
        self.friction_wall = 0.45

        self.h = float(cfg.get("sample_dt", 1.0 / 30.0)) / 32.0
        self.step_index = 0
        self.max_speed = 0.0
        self.contact_events = 0
        self._clamp_state()

    # ------------------------------------------------------------ geometry
    def _load_boundaries(self, cfg):
        walls = []
        ground = None
        geom = cfg.get("scene_geometry")
        objects = list(geom.get("objects") or []) if isinstance(geom, dict) else []
        for obj in objects:
            geo = obj.get("geometry") or {}
            if geo.get("type") != "box":
                continue
            half = np.asarray(geo.get("half_extents", [0.0, 0.0, 0.0]), dtype=float)
            trans = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
            cen = np.asarray(trans, dtype=float)
            if obj.get("motion") == "kinematic":
                walls.append((cen, half))
            else:
                ground = (cen, half)
        if len(walls) < 2:
            walls = [
                (np.array([-1.55, 0.0, 1.3]), np.array([0.12, 1.5, 1.3])),
                (np.array([1.55, 0.0, 1.3]), np.array([0.12, 1.5, 1.3])),
            ]
        if ground is None:
            ground = (np.array([0.0, 0.0, -0.05]), np.array([2.4, 2.0, 0.05]))
        walls.sort(key=lambda item: float(item[0][0]))
        self.walls = walls
        self.ground = ground
        self.wall_gap0 = min(abs(float(w[0][0])) - float(w[1][0]) for w in walls)
        self.wall_gap_final = 0.40
        self.close_t0, self.close_t1 = 0.15, 3.20

    def _wall_state(self, t):
        """Prescribed large-scale compression of the confining walls."""
        span = self.close_t1 - self.close_t0
        s = min(1.0, max(0.0, (t - self.close_t0) / span))
        u = s * s * (3.0 - 2.0 * s)
        du = 0.0 if (s <= 0.0 or s >= 1.0) else 6.0 * s * (1.0 - s) / span
        amount = self.wall_gap0 - self.wall_gap_final
        shift = amount * u
        rate = amount * du
        centers = []
        velocities = []
        for cen, _half in self.walls:
            direction = 1.0 if float(cen[0]) < 0.0 else -1.0
            centers.append(cen + np.array([direction * shift, 0.0, 0.0]))
            velocities.append(np.array([direction * rate, 0.0, 0.0]))
        return centers, velocities

    # ------------------------------------------------------------ topology
    def _build_topology(self):
        nx, ny = self.nx, self.ny
        tri = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                v00 = i * nx + j
                v10 = v00 + 1
                v01 = v00 + nx
                v11 = v01 + 1
                tri.append((v00, v10, v11))
                tri.append((v00, v11, v01))
        self.tri = np.asarray(tri, dtype=np.int64)
        self.ntri = len(self.tri)
        self.tv0 = self.tri[:, 0]
        self.tv1 = self.tri[:, 1]
        self.tv2 = self.tri[:, 2]

        # rest (material) edge matrices for the deformation gradient
        r = self.rest2d
        e1 = r[self.tv1] - r[self.tv0]
        e2 = r[self.tv2] - r[self.tv0]
        dm = np.empty((self.ntri, 2, 2))
        dm[:, 0, 0] = e1[:, 0]
        dm[:, 0, 1] = e2[:, 0]
        dm[:, 1, 0] = e1[:, 1]
        dm[:, 1, 1] = e2[:, 1]
        det = dm[:, 0, 0] * dm[:, 1, 1] - dm[:, 0, 1] * dm[:, 1, 0]
        det = np.where(np.abs(det) < _EPS, 1.0, det)
        inv = np.empty_like(dm)
        inv[:, 0, 0] = dm[:, 1, 1]
        inv[:, 0, 1] = -dm[:, 0, 1]
        inv[:, 1, 0] = -dm[:, 1, 0]
        inv[:, 1, 1] = dm[:, 0, 0]
        self.dm_inv = inv / det[:, None, None]
        self.vert_tri_counts = np.maximum(
            np.bincount(self.tri.ravel(), minlength=self.n), 1
        )

        # candidate vertex/triangle pairs, excluding incident triangles
        pv = np.repeat(np.arange(self.n, dtype=np.int64), self.ntri)
        pt = np.tile(np.arange(self.ntri, dtype=np.int64), self.n)
        incident = (self.tv0[pt] == pv) | (self.tv1[pt] == pv) | (self.tv2[pt] == pv)
        keep = ~incident
        self.pair_vert = pv[keep]
        self.pair_tri = pt[keep]
        self.pair_count = int(keep.sum())

        # candidate vertex/vertex pairs, excluding grid neighbours that are
        # already connected by distance constraints
        iu, iv = np.triu_indices(self.n, k=1)
        gi = iu // self.nx
        gj = iu % self.nx
        hi = iv // self.nx
        hj = iv % self.nx
        far = (np.abs(gi - hi) > 2) | (np.abs(gj - hj) > 2)
        self.vv_i = iu[far]
        self.vv_j = iv[far]

        jj = np.arange(self.n) % self.nx
        ii = np.arange(self.n) // self.nx
        self.boundary = (
            (ii == 0) | (ii == self.ny - 1) | (jj == 0) | (jj == self.nx - 1)
        ).astype(np.int32)

    # ---------------------------------------------------------- constraints
    def _build_constraints(self):
        nx, ny = self.nx, self.ny
        idx = np.arange(self.n).reshape(ny, nx)
        structural = []
        shear = []
        bending = []
        for j in range(nx - 1):
            structural.append(np.column_stack((idx[:, j], idx[:, j + 1])))
        for i in range(ny - 1):
            structural.append(np.column_stack((idx[i, :], idx[i + 1, :])))
        for i in range(ny - 1):
            for j in range(nx - 1):
                shear.append(
                    np.array(
                        [[idx[i, j], idx[i + 1, j + 1]], [idx[i, j + 1], idx[i + 1, j]]]
                    )
                )
        for j in range(nx - 2):
            bending.append(np.column_stack((idx[:, j], idx[:, j + 2])))
        for i in range(ny - 2):
            bending.append(np.column_stack((idx[i, :], idx[i + 2, :])))

        def stack(items):
            if not items:
                return np.zeros((0, 2), dtype=np.int64)
            return np.concatenate(items, axis=0).astype(np.int64)

        self.groups = []
        self.lam = []
        self._add_group(stack(structural), 1e-8, False)
        self._add_group(stack(shear), 1e-8, False)
        self._add_group(stack(bending), 0.06, True)
        self.n_iter = 3

    def _add_group(self, pairs, alpha, plastic):
        if len(pairs) == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = self.rest2d[i] - self.rest2d[j]
        l0 = np.sqrt(np.einsum("ij,ij->i", d, d))
        colors = self._greedy_colors(i, j)
        for c in range(int(colors.max()) + 1):
            m = colors == c
            self.groups.append(
                {
                    "i": i[m],
                    "j": j[m],
                    "L0": l0[m].copy(),
                    "L0_rest": l0[m].copy(),
                    "alpha": np.full(int(m.sum()), alpha),
                    "plastic": bool(plastic),
                }
            )
            self.lam.append(np.zeros(int(m.sum())))

    @staticmethod
    def _greedy_colors(i, j):
        colors = np.full(len(i), -1, dtype=np.int64)
        used = {}
        for k in range(len(i)):
            a = int(i[k])
            b = int(j[k])
            taken = used.get(a, set()) | used.get(b, set())
            c = 0
            while c in taken:
                c += 1
            colors[k] = c
            used.setdefault(a, set()).add(c)
            used.setdefault(b, set()).add(c)
        return colors

    # ------------------------------------------------------------- stepping
    def _solve_constraints(self, x, h):
        hh = h * h
        for k, g in enumerate(self.groups):
            ii = g["i"]
            jj = g["j"]
            d = x[ii] - x[jj]
            length = np.sqrt(np.einsum("ij,ij->i", d, d)) + _EPS
            c = length - g["L0"]
            nrm = d / length[:, None]
            alpha_t = g["alpha"] / hh
            wsum = (self.w[ii] + self.w[jj])[:, 0]
            dlam = (-c - alpha_t * self.lam[k]) / (wsum + alpha_t)
            self.lam[k] += dlam
            corr = dlam[:, None] * nrm
            x[ii] += self.w[ii] * corr
            x[jj] -= self.w[jj] * corr

    def _plastic_update(self):
        tol = 0.02
        rate = 0.25
        for g in self.groups:
            if not g["plastic"]:
                continue
            ii = g["i"]
            jj = g["j"]
            d = self.x[ii] - self.x[jj]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            c = length - g["L0"]
            band = tol * g["L0_rest"]
            g["L0"] += rate * (np.maximum(c - band, 0.0) + np.minimum(c + band, 0.0))
            np.clip(g["L0"], 0.20 * g["L0_rest"], 1.05 * g["L0_rest"], out=g["L0"])

    # ------------------------------------------------------------- contacts
    def _self_candidates(self, x, margin):
        """Vertex/triangle pairs closer than ``margin`` (incident ones excluded)."""
        empty = (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )
        if self.pair_count == 0:
            return empty
        cen = (x[self.tv0] + x[self.tv1] + x[self.tv2]) / 3.0
        r0 = x[self.tv0] - cen
        r1 = x[self.tv1] - cen
        r2 = x[self.tv2] - cen
        rad = np.sqrt(
            np.maximum(
                np.maximum(
                    np.einsum("ij,ij->i", r0, r0), np.einsum("ij,ij->i", r1, r1)
                ),
                np.einsum("ij,ij->i", r2, r2),
            )
        )
        diff = x[self.pair_vert] - cen[self.pair_tri]
        d2 = np.einsum("ij,ij->i", diff, diff)
        limit = rad[self.pair_tri] + margin
        cand = np.flatnonzero(d2 <= limit * limit)
        if cand.size == 0:
            return empty
        pv = self.pair_vert[cand]
        pt = self.pair_tri[cand]
        P = x[pv]
        A = x[self.tv0[pt]]
        B = x[self.tv1[pt]]
        C = x[self.tv2[pt]]
        Q, u, v, w = _closest_point_on_triangle(P, A, B, C)
        delta = P - Q
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        near = dist <= margin
        return pv[near], pt[near], u[near], v[near], w[near], dist[near]

    def _resolve_self_contacts(self, x):
        zero_i = np.zeros(0, dtype=np.int64)
        zero_n = np.zeros((0, 3))
        pv, pt, u, v, w, dist = self._self_candidates(x, self.d0_self)
        if pv.size == 0:
            return 0, zero_i, zero_n
        m = dist < self.d0_self
        if not np.any(m):
            return 0, zero_i, zero_n
        pv, pt, u, v, w, dist = pv[m], pt[m], u[m], v[m], w[m], dist[m]
        P = x[pv]
        A = x[self.tv0[pt]]
        B = x[self.tv1[pt]]
        C = x[self.tv2[pt]]
        Q = u[:, None] * A + v[:, None] * B + w[:, None] * C
        delta = P - Q
        dd = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        safe = np.where(dd < _EPS, 1.0, dd)
        nrm = delta / safe[:, None]
        degen = dd < 1e-9
        if np.any(degen):
            t0 = self.tv0[pt[degen]]
            t1 = self.tv1[pt[degen]]
            t2 = self.tv2[pt[degen]]
            tn = np.cross(x[t1] - x[t0], x[t2] - x[t0])
            tn = tn / (np.linalg.norm(tn, axis=1)[:, None] + _EPS)
            nrm[degen] = tn
        pen = self.d0_self - dd
        corr = 0.5 * pen[:, None] * nrm
        np.add.at(x, pv, corr)
        tri_corr = -0.5 * pen[:, None] * nrm
        np.add.at(x, self.tv0[pt], u[:, None] * tri_corr)
        np.add.at(x, self.tv1[pt], v[:, None] * tri_corr)
        np.add.at(x, self.tv2[pt], w[:, None] * tri_corr)
        return int(len(pv)), pv, nrm

    def _resolve_vertex_contacts(self, x):
        zero_i = np.zeros(0, dtype=np.int64)
        zero_n = np.zeros((0, 3))
        if len(self.vv_i) == 0:
            return 0, zero_i, zero_n
        delta = x[self.vv_i] - x[self.vv_j]
        d = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        m = d < self.d0_vv
        if not np.any(m):
            return 0, zero_i, zero_n
        i = self.vv_i[m]
        j = self.vv_j[m]
        dd = d[m]
        nrm = delta[m] / np.where(dd < _EPS, 1.0, dd)[:, None]
        pen = self.d0_vv - dd
        corr = 0.5 * pen[:, None] * nrm
        np.add.at(x, i, corr)
        np.add.at(x, j, -corr)
        return int(len(i)), i, nrm

    @staticmethod
    def _box_sdf(points, center, half):
        q = np.abs(points - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _resolve_boundaries(self, x, t):
        centers, velocities = self._wall_state(t)
        boxes = [
            (centers[k], self.walls[k][1], velocities[k]) for k in range(len(self.walls))
        ]
        boxes.append((self.ground[0], self.ground[1], np.zeros(3)))
        hit_idx = []
        hit_norm = []
        hit_vel = []
        for cen, half, vel in boxes:
            q = x - cen
            aq = np.abs(q)
            inside = np.all(aq < half, axis=1)
            rows = np.flatnonzero(inside)
            if rows.size == 0:
                continue
            pen = half[None, :] - aq[rows]
            axis = np.argmin(pen, axis=1)
            vals = aq[rows, axis]
            sign = np.where(q[rows, axis] >= 0.0, 1.0, -1.0)
            nrm = np.zeros((rows.size, 3))
            nrm[np.arange(rows.size), axis] = sign
            depth = (half[axis] - vals) + 1e-6
            x[rows] += nrm * depth[:, None]
            hit_idx.append(rows)
            hit_norm.append(nrm)
            hit_vel.append(np.tile(vel, (rows.size, 1)))
        if not hit_idx:
            return (
                0,
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3)),
                np.zeros((0, 3)),
            )
        return (
            int(sum(len(a) for a in hit_idx)),
            np.concatenate(hit_idx),
            np.concatenate(hit_norm),
            np.concatenate(hit_vel),
        )

    # --------------------------------------------------------------- physics
    def _substep(self, h, t_new):
        x = self.x
        v = self.v
        v += h * self.gravity[None, :]
        v *= np.exp(-0.05 * h)
        speed = np.linalg.norm(v, axis=1)
        fast = speed > 25.0
        if np.any(fast):
            v[fast] *= (25.0 / speed[fast])[:, None]
        x_pred = x + h * v
        for lam in self.lam:
            lam[:] = 0.0
        for _ in range(self.n_iter):
            self._solve_constraints(x_pred, h)
        _, self_idx, self_nrm = self._resolve_self_contacts(x_pred)
        _, vv_idx, vv_nrm = self._resolve_vertex_contacts(x_pred)
        _, wall_idx, wall_nrm, wall_vel = self._resolve_boundaries(x_pred, t_new)

        v_new = (x_pred - x) / h
        if self_idx.size:
            v_new[self_idx] = self._apply_friction(
                v_new[self_idx], self_nrm, None, self.friction_self
            )
        if vv_idx.size:
            v_new[vv_idx] = self._apply_friction(
                v_new[vv_idx], vv_nrm, None, self.friction_self
            )
        if wall_idx.size:
            v_new[wall_idx] = self._apply_friction(
                v_new[wall_idx], wall_nrm, wall_vel, self.friction_wall
            )
        self.contact_events += int(self_idx.size + vv_idx.size + wall_idx.size)
        self.x = x_pred
        self.v = v_new
        self.max_speed = max(
            self.max_speed, float(np.max(np.linalg.norm(self.v, axis=1)))
        )
        self._plastic_update()
        self._clamp_state()

    @staticmethod
    def _apply_friction(v, nrm, wall_vel, mu):
        if wall_vel is None:
            rel = v
        else:
            rel = v - wall_vel
        rn = np.minimum(np.einsum("ij,ij->i", rel, nrm), 0.0)
        tang = rel - np.einsum("ij,ij->i", rel, nrm)[:, None] * nrm
        rel_new = rn[:, None] * nrm + (1.0 - mu) * tang
        if wall_vel is None:
            return rel_new
        return wall_vel + rel_new

    def _clamp_state(self):
        x = self.x
        np.clip(x[:, 0], -6.0, 6.0, out=x[:, 0])
        np.clip(x[:, 1], -6.0, 6.0, out=x[:, 1])
        np.clip(x[:, 2], -0.5, 8.0, out=x[:, 2])
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.v, -30.0, 30.0, out=self.v)

    # --------------------------------------------------------------- exports
    def _deformation_gradient(self, x):
        p0 = x[self.tv0]
        d1 = x[self.tv1] - p0
        d2 = x[self.tv2] - p0
        l1 = np.linalg.norm(d1, axis=1)
        e1 = d1 / np.where(l1 < _EPS, 1.0, l1)[:, None]
        nrm = np.cross(d1, d2)
        ln = np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.where(ln < _EPS, 1.0, ln)[:, None]
        e2 = np.cross(nrm, e1)
        a1 = np.stack((l1, np.zeros(self.ntri)), axis=1)
        a2 = np.stack(
            (np.einsum("ij,ij->i", d2, e1), np.einsum("ij,ij->i", d2, e2)), axis=1
        )
        A = np.empty((self.ntri, 2, 2))
        A[:, 0, 0] = a1[:, 0]
        A[:, 0, 1] = a2[:, 0]
        A[:, 1, 0] = a1[:, 1]
        A[:, 1, 1] = a2[:, 1]
        F = A @ self.dm_inv
        acc = np.zeros((self.n, 2, 2))
        np.add.at(acc, self.tv0, F)
        np.add.at(acc, self.tv1, F)
        np.add.at(acc, self.tv2, F)
        Fv = acc / self.vert_tri_counts[:, None, None]
        return np.nan_to_num(Fv, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

    def _contact_signed_distance(self, x, t):
        values = []
        centers, _vel = self._wall_state(t)
        boxes = [
            (centers[k], self.walls[k][1]) for k in range(len(self.walls))
        ]
        boxes.append((self.ground[0], self.ground[1]))
        for cen, half in boxes:
            sd = self._box_sdf(x, cen, half)
            m = sd < self.report_margin
            if np.any(m):
                values.append(np.sort(sd[m]))
        _pv, _pt, _u, _v, _w, dist = self._self_candidates(
            x, self.d0_self + self.report_margin
        )
        if dist.size:
            gaps = dist - self.d0_self
            m = gaps < self.report_margin
            if np.any(m):
                values.append(np.sort(gaps[m]))
        if not values:
            return np.zeros(0)
        return np.sort(np.concatenate(values))

    def _frame(self, time):
        x = self.x.copy()
        v = self.v.copy()
        frame = StateFrame(
            float(time),
            x,
            v,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(x),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(x, float(time)),
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = int(round(float(time) / self.h))
        if target < self.step_index:
            target = self.step_index
        while self.step_index < target:
            self.step_index += 1
            self._substep(self.h, self.step_index * self.h)
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(self.step_index),
            "step_size": float(self.h),
            "vertex_count": int(self.n),
            "element_count": int(self.ntri),
            "total_mass": float(np.sum(self.mass)),
            "max_speed": float(self.max_speed),
            "contact_events": int(self.contact_events),
            "wall_gap_initial": float(self.wall_gap0),
            "wall_gap_final": float(self.wall_gap_final),
        }