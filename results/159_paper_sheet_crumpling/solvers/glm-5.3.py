"""Physics solver for scene 159_paper_sheet_crumpling.

Thin paper-like sheet crumpled between two kinematic closing walls.

Method: position-based dynamics (PBD/XPBD style) on the nodal grid of the
sheet supplied by the frozen scene scaffold.

* Inextensible structural + shear distance constraints (paper essentially
  does not stretch under crumpling loads).
* Two-hop bending constraints with reduced stiffness so sharp ridges and
  creases can form while spurious curvature is resisted.
* Box obstacles (ground + two walls).  The walls are kinematic and move
  inward on a smooth C1 schedule finishing before the end of the run,
  imposing the prescribed large-scale compression.
* Point-vs-triangle self-contact with finite sheet thickness so many
  layers pack together without fold-through, plus vertex-vertex repulsion
  for edge-crossing robustness.
* Semi-implicit velocity integration, per-substep position projection,
  normal contact response and Coulomb-like friction.

Everything is deterministic.  The only stochastic ingredient is a seeded
imperfection field that breaks the perfect symmetry of the flat sheet
(the buckling direction of a mathematically perfect sheet is undefined).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _closest_point_on_triangles(p, a, b, c):
    """Vectorized closest point of p[] on triangles (a, b, c).

    Returns (q, u, v, w) with q = u*a + v*b + w*c and u + v + w = 1.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-14, 1.0, denom)
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w

    def _ediv(num, den):
        den = np.where(np.abs(den) < 1e-14, 1.0, den)
        return num / den

    mA = (d1 <= 0) & (d2 <= 0)
    mB = (d3 >= 0) & (d4 <= d3)
    mC = (d6 >= 0) & (d5 <= d6)
    mAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    mAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    mBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    tAB = np.clip(_ediv(d1, d1 - d3), 0.0, 1.0)
    tAC = np.clip(_ediv(d2, d2 - d6), 0.0, 1.0)
    tBC = np.clip(_ediv(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)

    u = np.where(mAB, 1.0 - tAB, u)
    v = np.where(mAB, tAB, v)
    w = np.where(mAB, 0.0, w)

    u = np.where(mAC, 1.0 - tAC, u)
    v = np.where(mAC, 0.0, v)
    w = np.where(mAC, tAC, w)

    u = np.where(mBC, 0.0, u)
    v = np.where(mBC, 1.0 - tBC, v)
    w = np.where(mBC, tBC, w)

    u = np.where(mA, 1.0, u)
    v = np.where(mA, 0.0, v)
    w = np.where(mA, 0.0, w)

    u = np.where(mB, 0.0, u)
    v = np.where(mB, 1.0, v)
    w = np.where(mB, 0.0, w)

    u = np.where(mC, 0.0, u)
    v = np.where(mC, 0.0, v)
    w = np.where(mC, 1.0, w)

    q = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return q, u, v, w


class BenchmarkSolver:
    # --- numerical parameters -------------------------------------------------
    SUB_DT = 1.0 / 600.0          # physics substep (s)
    ITERS = 8                      # constraint iterations per substep
    K_STRUCT = 1.0                 # structural stiffness factor
    K_SHEAR = 0.85                 # shear stiffness factor
    K_BEND = 0.30                  # bending stiffness factor (creases form)
    LIN_DAMP = 1.0                 # velocity damping (1/s)
    FRICTION = 0.35
    VMAX = 20.0                    # velocity clamp (m/s)
    SELF_PASSES = 2                # self-contact passes per substep
    NEAR_CONTACT = 0.10            # near-contact reporting threshold (m)

    # --- prescribed wall motion ------------------------------------------------
    WALL_START = 1.55              # frozen initial wall center offset (x)
    WALL_TRAVEL = 0.95             # inward displacement of each wall
    WALL_PERIOD = 6.0              # compression finishes at t = 6 s

    def __init__(self) -> None:
        self.t = 0.0
        self._init_done = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.thickness = float(cloth["thickness"])
        self.half_thick = 0.5 * self.thickness
        self.dmin = self.thickness          # self-contact separation
        self.contact_length_scale = self.thickness

        frame0 = scene.initial_state()
        x0 = np.array(frame0.position, dtype=float)
        self.mass = np.full(self.n, float(cloth["total_mass"]) / self.n)
        self.body_id = np.zeros(self.n, dtype=np.int32)
        bv = frame0.attributes.get("boundary_vertex")
        self.boundary_vertex = (
            np.array(bv, dtype=np.int32) if bv is not None else np.zeros(self.n, np.int32)
        )

        # grid coordinates (scene.py ravel order: k = iy*nx + ix)
        sx, sy = (float(v) for v in cloth["size"])
        self.dx = sx / (self.nx - 1)
        self.dy = sy / (self.ny - 1)
        self.ix = np.tile(np.arange(self.nx), self.ny)
        self.iy = np.repeat(np.arange(self.ny), self.nx)
        self.material = x0[:, :2].copy()    # rest (material) coordinates

        # Deterministic imperfection (buckling direction of a perfect sheet is
        # undefined); the sheet stays "mostly flat".
        rng = np.random.default_rng(int(seed))
        span_x = max(x0[:, 0].max() - x0[:, 0].min(), 1e-9)
        span_y = max(x0[:, 1].max() - x0[:, 1].min(), 1e-9)
        mode = np.sin(np.pi * (x0[:, 0] - x0[:, 0].min()) / span_x)
        mode *= np.sin(np.pi * (x0[:, 1] - x0[:, 1].min()) / span_y)
        x0[:, 2] += 2.0e-3 * mode + rng.normal(0.0, 3.0e-4, self.n)

        self.x = x0
        self.v = np.zeros_like(self.x)
        self.t = 0.0

        # ---- constraints -----------------------------------------------------
        kgrid = np.arange(self.n).reshape(self.ny, self.nx)

        def build(pi, pj, rest, kfac):
            pi = np.asarray(pi).ravel()
            pj = np.asarray(pj).ravel()
            ok = (pi >= 0) & (pj >= 0) & (pi < self.n) & (pj < self.n)
            return pi[ok].astype(np.int64), pj[ok].astype(np.int64), np.full(int(ok.sum()), float(rest), float)

        diag = float(np.hypot(self.dx, self.dy))
        self.constraints = [
            build(kgrid[:, :-1], kgrid[:, 1:], self.dx, self.K_STRUCT),
            build(kgrid[:-1, :], kgrid[1:, :], self.dy, self.K_STRUCT),
            build(kgrid[:-1, :-1], kgrid[1:, 1:], diag, self.K_SHEAR),
            build(kgrid[:-1, 1:], kgrid[1:, :-1], diag, self.K_SHEAR),
            build(kgrid[:, :-2], kgrid[:, 2:], 2.0 * self.dx, self.K_BEND),
            build(kgrid[:-2, :], kgrid[2:, :], 2.0 * self.dy, self.K_BEND),
        ]

        # ---- triangles -------------------------------------------------------
        tri = []
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                v00 = kgrid[j, i]
                v10 = kgrid[j, i + 1]
                v01 = kgrid[j + 1, i]
                v11 = kgrid[j + 1, i + 1]
                tri.append((v00, v10, v11))
                tri.append((v00, v11, v01))
        self.tri = np.asarray(tri, dtype=np.int64)          # [T, 3]
        self.T = len(tri)

        # per-triangle broadphase radius (max vertex distance from centroid)
        self.tri_radius = None  # computed lazily from rest geometry (fixed)

        # ---- forbidden point-triangle pairs (incident or 1-ring adjacent) ----
        forbid = np.zeros((self.n, self.T), dtype=bool)
        for t in range(self.T):
            for vtx in self.tri[t]:
                jv, iv = divmod(int(vtx), self.nx)
                j0, j1 = max(0, jv - 1), min(self.ny, jv + 2)
                i0, i1 = max(0, iv - 1), min(self.nx, iv + 2)
                forbid[kgrid[j0:j1, i0:i1].ravel(), t] = True
        self.forbid = forbid

        # ---- vertex-vertex contact candidate pairs (skip 1-ring) -------------
        iu, ju = np.triu_indices(self.n, k=1)
        dj = np.abs(self.iy[iu] - self.iy[ju])
        di = np.abs(self.ix[iu] - self.ix[ju])
        far = (dj > 1) | (di > 1)
        self.vv_i = iu[far].astype(np.int64)
        self.vv_j = ju[far].astype(np.int64)

        # ---- obstacles -------------------------------------------------------
        self.ground = {
            "center": np.array([0.0, 0.0, -0.05]),
            "he": np.array([2.4, 2.0, 0.05]),
            "vel": np.zeros(3),
        }
        self.wall_he = np.array([0.12, 1.5, 1.3])
        self.wall_z = 1.3

        # Resolve the (small) initial penetration of the sheet edges into the
        # frozen wall positions statically, so the first dynamics substep does
        # not impart a spurious velocity kick.
        obs0 = self._obstacles(0.0)
        for _ in range(30):
            x_before = self.x.copy()
            for _ in range(4):
                self._project_constraints()
            for ob in obs0:
                self._project_box(ob["center"], ob["he"], self.half_thick)
            if np.max(np.linalg.norm(self.x - x_before, axis=1)) < 1e-6:
                break

        self._init_done = True

    # ------------------------------------------------------- wall kinematics
    def _wall_offset(self, t: float) -> float:
        u = min(max(t / self.WALL_PERIOD, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)          # smoothstep (C1)
        return self.WALL_START - self.WALL_TRAVEL * s

    def _wall_speed(self, t: float) -> float:
        u = min(max(t / self.WALL_PERIOD, 0.0), 1.0)
        if u <= 0.0 or u >= 1.0:
            return 0.0
        return self.WALL_TRAVEL * 6.0 * u * (1.0 - u) / self.WALL_PERIOD

    def _obstacles(self, t: float):
        off = self._wall_offset(t)
        spd = self._wall_speed(t)
        return [
            self.ground,
            {
                "center": np.array([-off, 0.0, self.wall_z]),
                "he": self.wall_he,
                "vel": np.array([spd, 0.0, 0.0]),
            },
            {
                "center": np.array([off, 0.0, self.wall_z]),
                "he": self.wall_he,
                "vel": np.array([-spd, 0.0, 0.0]),
            },
        ]

    # ------------------------------------------------------------- projection
    def _bincount_add(self, idx, delta):
        for ax in range(3):
            self.x[:, ax] += np.bincount(idx, weights=delta[:, ax], minlength=self.n)

    def _project_constraints(self):
        for pi, pj, rest, kfac in self.constraints:
            d = self.x[pj] - self.x[pi]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            corr = (kfac * (dist - rest) / dist) * 0.5
            cd = d * corr[:, None]
            self._bincount_add(pi, cd)
            self._bincount_add(pj, -cd)

    def _project_box(self, center, he, margin):
        loc = self.x - center
        cl = np.clip(loc, -he, he)
        diff = loc - cl
        nd = np.linalg.norm(diff, axis=1)
        inside = ~np.any(np.abs(loc) > he, axis=1)
        near_out = (~inside) & (nd < margin) & (nd > 1e-12)

        newx = self.x.copy()
        if np.any(near_out):
            n_out = diff[near_out] / nd[near_out][:, None]
            newx[near_out] = cl[near_out] + margin * n_out
        if np.any(inside):
            dist_face = he[None, :] - np.abs(loc[inside])
            axis = np.argmin(dist_face, axis=1)
            rows = np.arange(int(np.count_nonzero(inside)))
            sub = loc[inside].copy()
            sgn = np.sign(sub[rows, axis])
            sgn[sgn == 0] = 1.0
            sub[rows, axis] = sgn * (he[axis] + margin)
            newx[inside] = center[None, :] + sub
        self.x = newx

    def _contact_frame(self, center, he, margin):
        """Contact normal / in-contact mask / outside distance per particle."""
        loc = self.x - center
        cl = np.clip(loc, -he, he)
        diff = loc - cl
        nd = np.linalg.norm(diff, axis=1)
        inside = ~np.any(np.abs(loc) > he, axis=1)
        nrm = np.zeros_like(self.x)
        out = (~inside) & (nd > 1e-12)
        nrm[out] = diff[out] / nd[out][:, None]
        if np.any(inside):
            dist_face = he[None, :] - np.abs(loc[inside])
            axis = np.argmin(dist_face, axis=1)
            rows = np.arange(int(np.count_nonzero(inside)))
            sgn = np.sign(loc[inside][rows, axis])
            sgn[sgn == 0] = 1.0
            sub = np.zeros_like(loc[inside])
            sub[rows, axis] = sgn
            nrm[inside] = sub
        contact = inside | (nd < margin + 1.0e-3)
        return nrm, contact

    def _velocity_contact_response(self, obs, margin):
        for ob in obs:
            nrm, contact = self._contact_frame(ob["center"], ob["he"], margin)
            if not np.any(contact):
                continue
            vw = ob["vel"]
            vrel = self.v[contact] - vw[None, :]
            nn = nrm[contact]
            vn = np.einsum("ij,ij->i", vrel, nn)
            # remove approaching normal velocity
            vn_keep = np.maximum(vn, 0.0)
            vrel = vrel - (vn - vn_keep)[:, None] * nn
            # Coulomb-like tangential friction
            vt = vrel - np.einsum("ij,ij->i", vrel, nn)[:, None] * nn
            vrel = vrel - self.FRICTION * vt
            self.v[contact] = vw[None, :] + vrel

    # ------------------------------------------------------------ self-contact
    def _self_contact(self):
        x = self.x
        # broadphase radii from current triangle geometry
        a = x[self.tri[:, 0]]
        b = x[self.tri[:, 1]]
        c = x[self.tri[:, 2]]
        cen = (a + b + c) / 3.0
        rad = np.maximum(
            np.linalg.norm(a - cen, axis=1),
            np.maximum(np.linalg.norm(b - cen, axis=1), np.linalg.norm(c - cen, axis=1)),
        )
        cut = rad + self.dmin + 0.02
        # point-triangle candidates
        d = x[:, None, :] - cen[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        cand = (dist2 < cut[None, :] ** 2) & (~self.forbid)
        pi, ti = np.nonzero(cand)
        if pi.size:
            p = x[pi]
            ta = x[self.tri[ti, 0]]
            tb = x[self.tri[ti, 1]]
            tc = x[self.tri[ti, 2]]
            q, bu, bv, bw = _closest_point_on_triangles(p, ta, tb, tc)
            nvec = p - q
            dist = np.linalg.norm(nvec, axis=1)
            hit = dist < self.dmin
            if np.any(hit):
                pih = pi[hit]
                tih = ti[hit]
                nrm = np.zeros((int(hit.sum()), 3))
                dd = dist[hit]
                good = dd > 1e-9
                nrm[good] = nvec[hit][good] / dd[good][:, None]
                # degenerate fallback: triangle normal
                if (~good).any():
                    e1 = ta[hit][~good] - tb[hit][~good]
                    e2 = tc[hit][~good] - tb[hit][~good]
                    fn = np.cross(e1, e2)
                    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
                    nrm[~good] = fn
                push = (self.dmin - dd) * 0.5
                delta_p = push[:, None] * nrm
                np.add.at(self.x, pih, delta_p)
                np.add.at(self.x, self.tri[tih, 0], -bu[hit][:, None] * delta_p)
                np.add.at(self.x, self.tri[tih, 1], -bv[hit][:, None] * delta_p)
                np.add.at(self.x, self.tri[tih, 2], -bw[hit][:, None] * delta_p)
        # vertex-vertex repulsion (edge-crossing robustness)
        if self.vv_i.size:
            diff = x[self.vv_j] - x[self.vv_i]
            dist = np.linalg.norm(diff, axis=1)
            hit = dist < self.dmin
            if np.any(hit):
                ii = self.vv_i[hit]
                jj = self.vv_j[hit]
                dd = np.maximum(dist[hit], 1e-9)
                nrm = diff[hit] / dd[:, None]
                push = (self.dmin - dd) * 0.5
                np.add.at(self.x, ii, push[:, None] * nrm)
                np.add.at(self.x, jj, -push[:, None] * nrm)

    # -------------------------------------------------------------- time step
    def _step(self, h: float):
        obs = self._obstacles(self.t + 0.5 * h)
        damp = float(np.exp(-self.LIN_DAMP * h))
        self.v += h * self.gravity[None, :]
        self.v *= damp
        # velocity clamp (stability guard)
        sp = np.linalg.norm(self.v, axis=1)
        fast = sp > self.VMAX
        if np.any(fast):
            self.v[fast] *= (self.VMAX / sp[fast])[:, None]
        xprev = self.x.copy()
        self.x += h * self.v
        for _ in range(self.ITERS):
            self._project_constraints()
            for ob in obs:
                self._project_box(ob["center"], ob["he"], self.half_thick)
        for _ in range(self.SELF_PASSES):
            self._self_contact()
            for ob in obs:
                self._project_box(ob["center"], ob["he"], self.half_thick)
        self.v = (self.x - xprev) / h
        self._velocity_contact_response(obs, self.half_thick)
        self.t += h

    # ------------------------------------------------------------- frame data
    def _deformation_gradient(self):
        """Least-squares per-vertex 2x2 map from material coords to world xy."""
        # 1-ring directed edges in grid adjacency
        verts = np.arange(self.n).reshape(self.ny, self.nx)
        si, sj = [], []
        for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)):
            a = verts[max(0, -dj):self.ny - max(0, dj), max(0, -di):self.nx - max(0, di)]
            bb = verts[max(0, dj):self.ny - max(0, -dj), max(0, di):self.nx - max(0, -di)]
            si.append(a.ravel())
            sj.append(bb.ravel())
        si = np.concatenate(si)
        sj = np.concatenate(sj)
        m = self.material[sj] - self.material[si]              # [E,2]
        c = self.x[sj, :2] - self.x[si, :2]                    # [E,2]
        A = np.zeros((self.n, 2, 2))
        B = np.zeros((self.n, 2, 2))
        np.add.at(A, si, m[:, None, :] * m[:, :, None])
        np.add.at(B, si, c[:, None, :] * m[:, :, None])
        Areg = A + 1e-9 * np.eye(2)[None, :, :]
        Ft = np.linalg.solve(Areg, np.swapaxes(B, 1, 2))       # = F^T
        F = np.swapaxes(Ft, 1, 2)
        return F

    def _box_sdf(self, center, he):
        loc = self.x - center
        cl = np.clip(loc, -he, he)
        diff = loc - cl
        nd = np.linalg.norm(diff, axis=1)
        inside = ~np.any(np.abs(loc) > he, axis=1)
        sdf = np.where(inside, -np.min(he - np.abs(loc), axis=1), nd)
        return sdf

    def _contact_gaps(self):
        """Per-vertex signed contact gap (obstacles and self-contact)."""
        obs = self._obstacles(self.t)
        g_obs = np.full(self.n, np.inf)
        for ob in obs:
            g_obs = np.minimum(g_obs, self._box_sdf(ob["center"], ob["he"]) - self.half_thick)
        # self-contact distance via point-triangle broadphase
        a = self.x[self.tri[:, 0]]
        b = self.x[self.tri[:, 1]]
        c = self.x[self.tri[:, 2]]
        cen = (a + b + c) / 3.0
        rad = np.maximum(
            np.linalg.norm(a - cen, axis=1),
            np.maximum(np.linalg.norm(b - cen, axis=1), np.linalg.norm(c - cen, axis=1)),
        )
        cut = rad + self.dmin + self.NEAR_CONTACT
        d = self.x[:, None, :] - cen[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        cand = (dist2 < cut[None, :] ** 2) & (~self.forbid)
        g_self = np.full(self.n, np.inf)
        pi, ti = np.nonzero(cand)
        if pi.size:
            q, _, _, _ = _closest_point_on_triangles(
                self.x[pi],
                self.x[self.tri[ti, 0]],
                self.x[self.tri[ti, 1]],
                self.x[self.tri[ti, 2]],
            )
            dist = np.linalg.norm(self.x[pi] - q, axis=1)
            np.minimum.at(g_self, pi, dist - self.dmin)
        gap = np.minimum(g_obs, g_self)
        near = gap < self.NEAR_CONTACT
        return gap[near]

    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        gaps = self._contact_gaps()
        if gaps.size == 0:
            gaps = np.zeros(0)
        external_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": external_force,
                "contact_signed_distance": gaps,
                "boundary_velocity": np.array(
                    [[ob["vel"][0], 0.0, 0.0] for ob in self._obstacles(float(time))]
                ),
                "boundary_target_velocity": np.array(
                    [[ob["vel"][0], 0.0, 0.0] for ob in self._obstacles(float(time) + 1e-6)]
                ),
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not self._init_done:
            raise RuntimeError("initialize() must be called before advance_to()")
        if target > self.t + 1e-12:
            span = target - self.t
            nsub = int(np.ceil(span / self.SUB_DT - 1e-9))
            nsub = max(1, min(nsub, 200))
            h = span / nsub
            for _ in range(nsub):
                self._step(h)
            self.t = target
        return self._frame(max(target, self.t))

    def diagnostics(self) -> dict:
        off = self._wall_offset(self.t)
        return {
            "time": float(self.t),
            "wall_offset_x": float(off),
            "wall_inner_gap": float(2.0 * (off - self.wall_he[0])),
            "n_vertices": int(self.n),
            "n_triangles": int(self.T),
            "substep": float(self.SUB_DT),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self._init_done else 0.0,
        }
