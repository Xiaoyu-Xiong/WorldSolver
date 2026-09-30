"""Physically-based cloth solver: large sheet falling through a narrow funnel.

Method: XPBD/PBD-style position based dynamics on the cloth grid with
- semi-implicit velocity prediction under gravity,
- Jacobi-iterated distance constraints (structural / shear / bend),
- hard mesh projection against the frozen funnel triangle mesh and ground box,
- particle-based dense self-contact (vertices + edge midpoints + cell centres),
- Coulomb-like friction and normal-velocity removal at contacts.

The state is advanced with a fixed substep (1 ms) deterministic explicit
time stepping; no keyframes, animation tables or precomputed trajectories.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# ----------------------------------------------------------------------------- IO


def _load_obj(path: Path):
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("v "):
            verts.append([float(tok) for tok in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            if len(idx) >= 3:
                faces.append(idx[:3])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


# ------------------------------------------------------------------ geometry math


def _closest_point_segment(P, X, Y):
    d = Y - X
    l2 = (d * d).sum(axis=1)
    l2s = np.where(l2 < 1e-18, 1.0, l2)
    s = np.clip(((P - X) * d).sum(axis=1) / l2s, 0.0, 1.0)
    return X + s[:, None] * d


def _closest_point_triangle(P, A, B, C):
    """Exact closest point on triangle(s) to point(s), all (M,3)."""
    ab = B - A
    ac = C - A
    n = np.cross(ab, ac)
    nn = (n * n).sum(axis=1)
    ap = P - A
    s = (ap * n).sum(axis=1) / np.where(nn > 1e-18, nn, 1.0)
    proj = P - s[:, None] * n

    d11 = (ab * ab).sum(axis=1)
    d12 = (ab * ac).sum(axis=1)
    d22 = (ac * ac).sum(axis=1)
    bp = proj - A
    b1 = (bp * ab).sum(axis=1)
    b2 = (bp * ac).sum(axis=1)
    det = d11 * d22 - d12 * d12
    dets = np.where(np.abs(det) > 1e-18, det, 1.0)
    u = (d22 * b1 - d12 * b2) / dets
    v = (d11 * b2 - d12 * b1) / dets
    inside = (nn > 1e-18) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)

    q1 = _closest_point_segment(P, A, B)
    q2 = _closest_point_segment(P, A, C)
    q3 = _closest_point_segment(P, B, C)
    d1 = ((P - q1) ** 2).sum(axis=1)
    d2 = ((P - q2) ** 2).sum(axis=1)
    d3 = ((P - q3) ** 2).sum(axis=1)
    q = np.where((d1 <= d2)[:, None], q1, q2)
    dq = np.minimum(d1, d2)
    q = np.where((d3 <= dq)[:, None], q3, q)
    return np.where(inside[:, None], proj, q)


def _mesh_closest(P, V, F):
    """Distance and closest point from points P to triangle mesh (V,F)."""
    A = V[F[:, 0]][None, :, :]
    B = V[F[:, 1]][None, :, :]
    C = V[F[:, 2]][None, :, :]
    Pb = P[:, None, :]
    Q = _closest_point_triangle(Pb, A, B, C)  # (N,T,3)
    D = np.sqrt(((Pb - Q) ** 2).sum(axis=2))  # (N,T)
    t = np.argmin(D, axis=1)
    n = len(P)
    rows = np.arange(n)
    return D[rows, t], Q[rows, t]


def _mesh_inside(P, V, F):
    """Ray parity test along +z (points inside a closed solid shell)."""
    A = V[F[:, 0]][None, :, :]
    B = V[F[:, 1]][None, :, :]
    C = V[F[:, 2]][None, :, :]
    e1 = B - A
    e2 = C - A
    pvec = np.cross(np.array([0.0, 0.0, 1.0]), e2)  # (1,T,3)
    det = (e1 * pvec).sum(axis=2)  # (1,T)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    tvec = P[:, None, :] - A  # (N,T,3)
    u = (tvec * pvec).sum(axis=2) * inv
    qvec = np.cross(tvec, e1)
    v = qvec[:, :, 2] * inv
    t = (e2 * qvec).sum(axis=2) * inv
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
    return (hit.sum(axis=1) % 2) == 1


def _box_sdf(P, c, h):
    q = np.abs(P - c[None, :]) - h[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _box_closest(P, c, h):
    q = P - c[None, :]
    cl = np.clip(q, -h[None, :], h[None, :])
    outside = np.linalg.norm(np.maximum(np.abs(q) - h[None, :], 0.0), axis=1) > 1e-12
    qbox = c[None, :] + cl
    # interior: push to the face with least slack
    slack = h[None, :] - np.abs(q)
    ax = np.argmin(slack, axis=1)
    rows = np.arange(len(P))
    qin = q.copy()
    qin[rows, ax] = np.where(q[rows, ax] >= 0.0, h[ax], -h[ax])
    qin = c[None, :] + qin
    return np.where(outside[:, None], qbox, qin), outside


# ------------------------------------------------------------------------- solver


class BenchmarkSolver:
    SUBSTEP = 1.0e-3
    N_ITERS = 8
    RELAX = 0.35
    K_STRUCT = 1.0
    K_SHEAR = 0.45
    K_BEND = 0.015
    SELF_RADIUS = 0.045
    SELF_RELAX = 0.55
    CONTACT_MARGIN = 0.01
    FRICTION = 0.999  # tangential retention per contact substep
    DRAG = 0.05  # 1/s air drag
    VMAX = 12.0

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        init = scene.initial_state()
        self.p0 = np.array(init.position, dtype=float)
        self.p = self.p0.copy()
        self.v = np.zeros_like(self.p)
        self.mass = np.array(init.mass, dtype=float)
        self.n = len(self.p)
        self.inv_mass = 1.0 / self.mass
        self.thickness = float(cfg["cloth"]["thickness"])
        self.nx, self.ny = (int(v) for v in cfg["cloth"]["grid_shape"])

        root = Path(__file__).resolve().parents[1]
        geom = cfg.get("scene_geometry") or {}
        funnel_asset = root / "assets" / "funnel.obj"
        ground_c = np.array([0.0, 0.0, -0.05])
        ground_h = np.array([2.4, 2.0, 0.05])
        for obj in geom.get("objects", []):
            if obj.get("id") == "funnel":
                asset = (obj.get("geometry") or {}).get("asset")
                if asset:
                    cand = Path(asset)
                    if not cand.is_absolute():
                        cand = root / cand
                    if cand.is_file():
                        funnel_asset = cand
            if obj.get("id") == "ground":
                g = obj.get("geometry") or {}
                if g.get("type") == "box":
                    ground_h = np.asarray(g["half_extents"], dtype=float)
                    tr = (obj.get("transform") or {}).get("translation")
                    if tr is not None:
                        ground_c = np.asarray(tr, dtype=float)
        self.fv, self.ff = _load_obj(funnel_asset)
        self.ground_c = ground_c
        self.ground_h = ground_h

        # ---- topology (regular grid, xy plane) --------------------------------
        idx = np.arange(self.n).reshape(self.ny, self.nx)  # row = y, col = x
        pairs: list[tuple[np.ndarray, np.ndarray, float]] = []

        def add(ei, ej, k):
            ei = np.asarray(ei, dtype=np.int64)
            ej = np.asarray(ej, dtype=np.int64)
            d = np.linalg.norm(self.p0[ei] - self.p0[ej], axis=1)
            pairs.append((ei, ej, d, k))

        add(idx[:, :-1].ravel(), idx[:, 1:].ravel(), self.K_STRUCT)
        add(idx[:-1, :].ravel(), idx[1:, :].ravel(), self.K_STRUCT)
        add(idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), self.K_SHEAR)
        add(idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), self.K_SHEAR)
        add(idx[:, :-2].ravel(), idx[:, 2:].ravel(), self.K_BEND)
        add(idx[:-2, :].ravel(), idx[2:, :].ravel(), self.K_BEND)
        self.cons = pairs

        # ---- self-contact particle set ----------------------------------------
        vert_idx = np.arange(self.n)
        mid_a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
        mid_b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
        quad = np.stack(
            [idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
             idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
        cen_idx = quad
        self.mid_a, self.mid_b = mid_a, mid_b
        self.cen_idx = cen_idx

        rest = [self.p0[vert_idx],
                0.5 * (self.p0[mid_a] + self.p0[mid_b]),
                self.p0[quad].mean(axis=1)]
        self.R = np.concatenate(rest, axis=0)
        self.m_nvert = self.n
        self.m_nmid = len(mid_a)
        self.m_ncen = cen_idx.shape[0]
        self.M = self.R.shape[0]

        d2 = ((self.R[:, None, :] - self.R[None, :, :]) ** 2).sum(axis=2)
        excl = d2 < (0.34 ** 2)
        iu = np.triu_indices(self.M, k=1)
        self.ia, self.ib = iu[0], iu[1]
        self.excl_pairs = excl[iu]
        # slave -> vertex scatter index/weight tables
        w_mid = np.full(self.m_nmid, 0.5)
        w_cen = np.full(self.m_ncen, 0.25)
        self.slave_src = np.concatenate([mid_a, mid_b,
                                         cen_idx[:, 0], cen_idx[:, 1],
                                         cen_idx[:, 2], cen_idx[:, 3]])
        self.slave_w = np.concatenate([w_mid, w_mid, w_cen, w_cen, w_cen, w_cen])
        self.slave_base = np.concatenate([np.zeros(self.m_nmid * 2, dtype=np.int64),
                                          (np.arange(4 * self.m_ncen) // 4 + self.n + self.m_nmid)])

        # ---- neighbour structure for deformation gradient ----------------------
        nbrs: list[list[int]] = [[] for _ in range(self.n)]
        for (ei, ej, _rest, _k) in pairs[:4]:  # structural + shear edges
            for a, b in zip(ei.tolist(), ej.tolist()):
                nbrs[a].append(b)
                nbrs[b].append(a)
        self.nbrs = [np.unique(np.asarray(x, dtype=np.int64)) for x in nbrs]

        self.nsteps = 0
        self.t = 0.0
        self.boundary = np.array(init.attributes.get("boundary_vertex",
                                                     np.zeros(self.n)), dtype=np.int32)

    # ------------------------------------------------------------------ stepping

    def _solve_constraints(self):
        p, w = self.p, self.inv_mass
        dp = np.zeros_like(p)
        for (ei, ej, rest, k) in self.cons:
            d = p[ei] - p[ej]
            L = np.sqrt((d * d).sum(axis=1))
            Ls = np.where(L < 1e-9, 1.0, L)
            C = L - rest
            wsum = w[ei] + w[ej]
            grad = d / Ls[:, None]
            mag = -(k * C / np.where(wsum > 0, wsum, 1.0))[:, None] * grad
            np.add.at(dp, ei, mag * (w[ei] / np.where(wsum > 0, wsum, 1.0))[:, None])
            np.add.at(dp, ej, -mag * (w[ej] / np.where(wsum > 0, wsum, 1.0))[:, None])
        self.p = p + self.RELAX * dp

    def _project_funnel(self, collect=False):
        p = self.p
        d, q = _mesh_closest(p, self.fv, self.ff)
        near = d < self.CONTACT_MARGIN
        if np.any(near):
            sub = np.nonzero(near)[0]
            diff = p[sub] - q[sub]
            dist = np.sqrt((diff * diff).sum(axis=1))
            dist = np.where(dist < 1e-12, 1.0, dist)
            nhat = diff / dist[:, None]
            inside = _mesh_inside(p[sub], self.fv, self.ff)
            sgn = np.where(inside, -1.0, 1.0)[:, None]
            p[sub] = q[sub] + sgn * nhat * self.CONTACT_MARGIN
            if collect:
                self._cn[sub] = sgn * nhat
                self._cmask[sub] = True
        return d

    def _project_ground(self, collect=False):
        p = self.p
        sdf = _box_sdf(p, self.ground_c, self.ground_h)
        near = sdf < self.CONTACT_MARGIN
        if np.any(near):
            sub = np.nonzero(near)[0]
            q, outside = _box_closest(p[sub], self.ground_c, self.ground_h)
            diff = p[sub] - q
            dist = np.sqrt((diff * diff).sum(axis=1))
            dist = np.where(dist < 1e-12, 1.0, dist)
            nhat = diff / dist[:, None]
            sgn = np.where(outside, 1.0, -1.0)[:, None]
            p[sub] = q[sub] + sgn * nhat * self.CONTACT_MARGIN
            if collect:
                self._cn[sub] = sgn * nhat
                self._cmask[sub] = True
        return sdf

    def _self_contact(self):
        p = self.p
        Pm = np.concatenate(
            [p, 0.5 * (p[self.mid_a] + p[self.mid_b]),
             p[self.cen_idx].mean(axis=1)], axis=0)
        diff = Pm[self.ia] - Pm[self.ib]
        d = np.sqrt((diff * diff).sum(axis=1))
        viol = (d < self.SELF_RADIUS) & (~self.excl_pairs) & (d > 1e-9)
        if not np.any(viol):
            return
        vi = np.nonzero(viol)[0]
        a, b = self.ia[vi], self.ib[vi]
        dv = d[vi]
        dirv = diff[vi] / dv[:, None]
        push = (self.SELF_RADIUS - dv)[:, None] * dirv * 0.5 * self.SELF_RELAX
        corr = np.zeros_like(Pm)
        np.add.at(corr, a, push)
        np.add.at(corr, b, -push)
        dvert = corr[: self.n].copy()
        np.add.at(dvert, self.slave_src, corr[self.n:] * self.slave_w[:, None])
        self.p = p + dvert

    def _substep(self):
        dt = self.SUBSTEP
        self._cmask = np.zeros(self.n, dtype=bool)
        self._cn = np.zeros((self.n, 3))
        self.v += self.gravity[None, :] * dt
        self.v *= 1.0 - self.DRAG * dt
        prev = self.p.copy()
        self.p = self.p + self.v * dt

        for _ in range(self.N_ITERS):
            self._solve_constraints()
        self._project_funnel(collect=True)
        self._self_contact()
        for _ in range(3):
            self._solve_constraints()
        self._project_funnel(collect=True)
        self._project_ground(collect=True)
        self._self_contact()
        self._solve_constraints()
        self._project_funnel(collect=True)
        self._project_ground(collect=True)

        self.p = np.nan_to_num(self.p, nan=0.0, posinf=0.0, neginf=0.0)
        self.v = (self.p - prev) / dt
        # contact response: remove inward normal velocity, damp tangential
        m = self._cmask
        if np.any(m):
            vn = (self.v[m] * self._cn[m]).sum(axis=1)
            v_n = vn[:, None] * self._cn[m]
            vt = self.v[m] - v_n
            vn_clamped = np.where(vn < 0.0, 0.0, vn)[:, None] * self._cn[m]
            self.v[m] = vn_clamped + vt * self.FRICTION
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.sqrt((self.v * self.v).sum(axis=1))
        scale = np.where(speed > self.VMAX, self.VMAX / np.where(speed > 0, speed, 1.0), 1.0)
        self.v = self.v * scale[:, None]
        self.p = np.clip(self.p, -50.0, 50.0)

        self.nsteps += 1
        self.t = self.nsteps * dt

    # ------------------------------------------------------------------- outputs

    def _deformation_gradient(self):
        F = np.zeros((self.n, 3, 3))
        p = self.p
        for i in range(self.n):
            js = self.nbrs[i]
            if len(js) < 2:
                F[i] = np.eye(3)
                continue
            u = self.p0[js, :2] - self.p0[i, :2]  # (k,2) material
            dw = p[js] - p[i]  # (k,3) world
            A = u.T @ u  # (2,2)
            B = dw.T @ u  # (3,2)
            try:
                Fi = np.zeros((3, 3))
                Fi[:, :2] = np.linalg.solve(A + 1e-12 * np.eye(2), B.T).T
                c1 = Fi[:, 0]
                c2 = Fi[:, 1]
                nrm = np.cross(c1, c2)
                ln = np.linalg.norm(nrm)
                if ln > 1e-9:
                    Fi[:, 2] = nrm / ln
                else:
                    Fi[:, 2] = np.array([0.0, 0.0, 1.0])
                F[i] = Fi
            except np.linalg.LinAlgError:
                F[i] = np.eye(3)
        return F

    def _signed_distances(self):
        """Per-vertex min signed distance to funnel mesh and ground box."""
        d_fun, q_fun = _mesh_closest(self.p, self.fv, self.ff)
        inside = _mesh_inside(self.p, self.fv, self.ff)
        sdf_fun = np.where(inside, -d_fun, d_fun)
        sdf_gnd = _box_sdf(self.p, self.ground_c, self.ground_h)
        return np.minimum(sdf_fun, sdf_gnd)

    def _frame(self, time: float) -> StateFrame:
        sdf = self._signed_distances()
        near = np.abs(sdf) < 0.08
        contact_dist = sdf[near]
        if contact_dist.size == 0:
            contact_dist = np.zeros(0)
        g = self.gravity
        ext = self.mass[:, None] * g[None, :]
        frame = StateFrame(
            time=time,
            position=self.p.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": contact_dist,
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": ext,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / self.SUBSTEP))
        if target < self.nsteps:
            target = self.nsteps
        while self.nsteps < target:
            self._substep()
        self.t = self.nsteps * self.SUBSTEP
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position-based-dynamics cloth with mesh projection and "
                      "particle self-contact",
            "substep": self.SUBSTEP,
            "steps_taken": int(self.nsteps),
            "self_contact_radius": self.SELF_RADIUS,
            "particles": int(self.M),
        }
