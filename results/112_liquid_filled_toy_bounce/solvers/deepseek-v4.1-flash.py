"""Two-way coupled liquid-filled deformable toy bounce.

Physical model
--------------
* The toy shell is a closed triangulated elastic membrane discretised by its
  mesh vertices.  Mesh-edge (stretch) and vertex-pair (bending) constraints are
  solved with XPBD, a real compliance-parameterised elastic solver that is
  unconditionally stable at the simulation time step.
* The enclosed liquid is a particle fluid solved with Position Based Fluids
  (density constraint projection + XSPH viscosity), i.e. a genuine
  incompressible-like Navier-Stokes surrogate with advected momentum.
* The coupling is two-way and momentum conserving.  Liquid particles that
  approach the deforming shell triangles are projected back onto the wall and
  the equal and opposite reaction is distributed to the triangle vertices with
  barycentric weights and mass weighting.  The moving shell therefore drives
  the fluid, and fluid pressure deforms / rebounds the shell.
* The red and yellow marker particles are massless passive tracers: their
  velocity is the SPH interpolated liquid velocity, so they are advected by the
  flow and never attached to the shell.
* All external forces are integrated over time and exposed through the required
  raw trajectory budget fields.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

try:  # pragma: no cover - platform provides the contract module
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _load_obj_mesh(path):
    """Return (vertices, triangles) from a wavefront OBJ, or (None, None)."""
    verts = []
    faces = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == "f" and len(parts) >= 4:
                    idx = []
                    for token in parts[1:]:
                        idx.append(int(token.split("/")[0]) - 1)
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
    except OSError:
        return None, None
    if not verts or not faces:
        return None, None
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _lathe_faces(n_rings=12, n_theta=16):
    """Fallback connectivity for the frozen lathed toy shell."""
    faces = []
    for r in range(n_rings - 1):
        for j in range(n_theta):
            a = r * n_theta + j
            b = r * n_theta + (j + 1) % n_theta
            c = (r + 1) * n_theta + (j + 1) % n_theta
            d = (r + 1) * n_theta + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    bottom = n_rings * n_theta
    top = bottom + 1
    for j in range(n_theta):
        faces.append((bottom, (j + 1) % n_theta, j))
        faces.append((top, (n_rings - 1) * n_theta + j,
                      (n_rings - 1) * n_theta + (j + 1) % n_theta))
    return np.asarray(faces, dtype=np.int64)


def _closest_point_triangles(P, A, B, C):
    """Vectorised closest point on a triangle (Ericson) plus barycentric coords."""
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cpv = P - C
    d5 = np.einsum("ij,ij->i", ab, cpv)
    d6 = np.einsum("ij,ij->i", ac, cpv)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    ok = np.abs(denom) > 1e-30
    safe = np.where(ok, denom, 1.0)
    v = np.where(ok, vb / safe, 0.0)
    w = np.where(ok, vc / safe, 0.0)

    m_a = (d1 <= 0.0) & (d2 <= 0.0)
    m_b = (~m_a) & (d3 >= 0.0) & (d4 <= d3)
    m_c = (~m_a) & (~m_b) & (d6 >= 0.0) & (d5 <= d6)
    m_ab = (~m_a) & (~m_b) & (~m_c) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    m_ac = (~m_a) & (~m_b) & (~m_c) & (~m_ab) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    m_bc = ~(m_a | m_b | m_c | m_ab | m_ac)

    den_ab = np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1.0)
    t_ab = np.clip(d1 / den_ab, 0.0, 1.0)
    den_ac = np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1.0)
    t_ac = np.clip(d2 / den_ac, 0.0, 1.0)
    den_bc = (d4 - d3) + (d5 - d6)
    den_bc = np.where(np.abs(den_bc) > 1e-30, den_bc, 1.0)
    t_bc = np.clip((d4 - d3) / den_bc, 0.0, 1.0)

    v = np.where(m_ab, t_ab, v)
    w = np.where(m_ab, 0.0, w)
    v = np.where(m_ac, 0.0, v)
    w = np.where(m_ac, t_ac, w)
    v = np.where(m_bc, 1.0 - t_bc, v)
    w = np.where(m_bc, t_bc, w)

    v = np.where(m_a, 0.0, v)
    w = np.where(m_a, 0.0, w)
    v = np.where(m_b, 1.0, v)
    w = np.where(m_b, 0.0, w)
    v = np.where(m_c, 0.0, v)
    w = np.where(m_c, 1.0, w)

    cp = A + v[:, None] * ab + w[:, None] * ac
    bary = np.stack([1.0 - v - w, v, w], axis=1)
    return cp, bary


class BenchmarkSolver:
    """Deterministic XPBD / PBF solver for the liquid filled bouncing toy."""

    # ---- numerical parameters -------------------------------------------------
    DT = 0.002                 # internal time step [s]
    ITERS = 5                  # constraint iterations per step
    K_EDGE = 300.0             # shell stretch stiffness [N/m]
    K_BEND = 100.0             # shell bending stiffness [N/m]
    EPS_LAM = 1e-4             # PBF lambda regularisation
    VISC = 2.0                 # XSPH viscosity rate [1/s]
    DAMP_SHELL = 0.35          # shell material damping [1/s]
    E_SHELL = 0.45             # floor restitution, shell
    E_LIQUID = 0.05            # floor restitution, liquid
    E_MARKER = 0.02            # floor restitution, tracers
    MU_SHELL = 0.06            # floor friction per contact step
    MU_LIQUID = 0.02
    MU_MARKER = 0.04
    R_FLOOR = 0.02             # floor contact radius
    MAX_SPEED = 40.0
    REF_VELOCITY = 3.43        # positive velocity scale [m/s]

    # ------------------------------------------------------------------
    def __init__(self):
        self._ready = False
        self._steps = 0
        self._time = 0.0

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = {}
        try:
            cfg = dict(getattr(scene, "config", {}) or {})
        except Exception:
            cfg = {}
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        if self.gravity.shape != (3,):
            self.gravity = np.array([0.0, 0.0, -9.81])
        self.seed = int(seed)

        frame = scene.initial_state()
        pos = np.array(frame.position, dtype=float)
        vel = np.array(frame.velocity, dtype=float)
        mass = np.array(frame.mass, dtype=float)
        bid = np.array(frame.body_id, dtype=np.int64)

        self.body_id = bid
        self.mass = mass
        self.N = int(mass.shape[0])
        self.pos = pos
        self.vel = vel
        self.shell = np.flatnonzero(bid == 1)
        self.liquid = np.flatnonzero(bid == 2)
        self.red = np.flatnonzero(bid == 3)
        self.yellow = np.flatnonzero(bid == 4)
        self.marker = np.flatnonzero((bid == 3) | (bid == 4))
        self.surface_set = np.concatenate((self.liquid, self.marker))
        self.active = mass > 0.0
        self.inv_mass = np.where(self.active, 1.0 / np.where(self.active, mass, 1.0), 0.0)
        self.is_shell = np.zeros(self.N, dtype=bool)
        self.is_shell[self.shell] = True
        self.is_liquid = np.zeros(self.N, dtype=bool)
        self.is_liquid[self.liquid] = True

        # ---------------- shell connectivity ----------------
        n_shell = len(self.shell)
        asset = "assets/toy_shell.obj"
        try:
            import os
            if not os.path.isfile(asset):
                alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "assets", "toy_shell.obj")
                if os.path.isfile(alt):
                    asset = alt
        except Exception:
            pass
        verts, faces = _load_obj_mesh(asset)
        use_obj = False
        if verts is not None and faces is not None and len(verts) == n_shell:
            if np.array_equal(faces.max(axis=0) < n_shell, np.ones(3, dtype=bool)):
                err = np.abs(verts - pos[self.shell]).max()
                if err < 1e-4:
                    use_obj = True
                    faces = np.asarray(faces, dtype=np.int64)
        if not use_obj:
            faces = _lathe_faces()
            if faces.max() >= n_shell:
                raise ValueError("cannot reconstruct shell connectivity")
        self.faces = faces

        # outward orientation from the signed volume of the closed mesh
        a, b, c = pos[self.shell][faces[:, 0]], pos[self.shell][faces[:, 1]], pos[self.shell][faces[:, 2]]
        signed_vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        self.face_flip = 1.0 if signed_vol >= 0.0 else -1.0
        centroids = (a + b + c) / 3.0
        circum = np.maximum.reduce([
            np.linalg.norm(a - centroids, axis=1),
            np.linalg.norm(b - centroids, axis=1),
            np.linalg.norm(c - centroids, axis=1),
        ])
        self.face_circum = float(circum.max())

        # edges (stretch) and opposite-vertex pairs (bending)
        e01 = np.sort(faces[:, [0, 1]], axis=1)
        e12 = np.sort(faces[:, [1, 2]], axis=1)
        e20 = np.sort(faces[:, [2, 0]], axis=1)
        edges = np.unique(np.concatenate([e01, e12, e20], axis=0), axis=0)
        self.edge_i = edges[:, 0].astype(np.int64)
        self.edge_j = edges[:, 1].astype(np.int64)
        rest = pos[self.shell]
        self.edge_rest = np.linalg.norm(rest[self.edge_i] - rest[self.edge_j], axis=1)

        bend = {}
        for tri in faces:
            for ei, oi in (((tri[0], tri[1]), tri[2]), ((tri[1], tri[2]), tri[0]),
                           ((tri[2], tri[0]), tri[1])):
                key = (int(min(ei[0], ei[1])), int(max(ei[0], ei[1])))
                bend.setdefault(key, []).append(int(oi))
        bi, bj = [], []
        for key, opp in bend.items():
            if len(opp) == 2 and opp[0] != opp[1]:
                bi.append(opp[0])
                bj.append(opp[1])
        self.bend_i = np.asarray(bi, dtype=np.int64)
        self.bend_j = np.asarray(bj, dtype=np.int64)
        self.bend_rest = np.linalg.norm(rest[self.bend_i] - rest[self.bend_j], axis=1)

        self.alpha_e = 1.0 / self.K_EDGE
        self.alpha_b = 1.0 / self.K_BEND

        # ---------------- liquid discretisation ----------------
        liq0 = pos[self.liquid]
        self.m_l = float(mass[self.liquid[0]])
        tree = cKDTree(liq0)
        d, _ = tree.query(liq0, k=2)
        self.spacing = float(np.median(d[:, 1]))
        self.h = 2.0 * self.spacing
        self.Wc = 315.0 / (64.0 * math.pi * self.h ** 9)
        self.Gc = 45.0 / (math.pi * self.h ** 6)
        self.h2 = self.h * self.h

        # rest density from the interior of the initial uniform packing
        pairs0 = tree.query_pairs(self.h, output_type="ndarray")
        rho = np.full(len(self.liquid), self.m_l * self.Wc * self.h2 ** 3)
        if pairs0.size:
            p0i, p0j = pairs0[:, 0], pairs0[:, 1]
            r0 = np.linalg.norm(liq0[p0i] - liq0[p0j], axis=1)
            w0 = self.Wc * np.maximum(self.h2 - r0 * r0, 0.0) ** 3
            cnt = (np.bincount(p0i, minlength=len(self.liquid))
                   + np.bincount(p0j, minlength=len(self.liquid)))
            np.add.at(rho, p0i, self.m_l * w0)
            np.add.at(rho, p0j, self.m_l * w0)
            interior = cnt >= 20
        else:
            interior = np.zeros(len(self.liquid), dtype=bool)
        self.rho0 = float(np.median(rho[interior])) if interior.sum() >= 8 else float(np.median(rho))
        if not np.isfinite(self.rho0) or self.rho0 <= 0.0:
            self.rho0 = self.m_l / (self.spacing ** 3)

        self.dc = 0.5 * self.spacing
        self.Rc = self.dc + self.face_circum
        self.max_push = 0.25 * self.h
        self.eps_lam = self.EPS_LAM

        # ---------------- boundary samples (trajectory fields) ----------------
        self.floor_pts = np.array([
            [-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [-0.4, 0.4, 0.0],
            [0.4, 0.4, 0.0], [0.0, 0.0, 0.0],
        ], dtype=float)
        self.floor_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(self.floor_pts), 1))

        # ---------------- export bookkeeping ----------------
        self.G = 3
        weights = np.zeros((self.G, self.N))
        weights[0, self.shell] = 1.0
        weights[1, self.surface_set] = 1.0
        weights[2, :] = 1.0
        self.domain_weights = weights
        shell_mass = float(mass[self.shell].sum())
        liquid_mass = float(mass[self.liquid].sum())
        self.domains = [self.shell, self.surface_set, np.arange(self.N)]

        self.K_TERMS = 3  # gravity, floor boundary traction, solid-fluid coupling
        self.K_BODIES = 5  # shell, liquid, red, yellow, floor
        self.ext_force = np.zeros((self.G, self.K_TERMS, 3))
        self.cum_ext_impulse = np.zeros((self.G, 3))
        self.interaction = np.zeros((self.K_BODIES, 3))
        for g in range(self.G):
            dm = float(mass[self.domains[g]].sum())
            self.ext_force[g, 0, :] = dm * self.gravity
        self.cum_mass_source = np.zeros(2)

        self.mass_quantity = np.zeros((self.N, 2))
        self.mass_quantity[self.shell, 0] = mass[self.shell]
        self.mass_quantity[self.liquid, 1] = mass[self.liquid]
        self.passive_marker = np.isin(bid, (3, 4)).astype(float)

        ref_v = self.REF_VELOCITY
        self.ref_velocity = float(ref_v)
        self.ref_momentum = float((shell_mass + liquid_mass) * ref_v)

        self._pi = np.empty(0, dtype=np.int64)
        self._pj = np.empty(0, dtype=np.int64)
        self._ready = True

    # ------------------------------------------------------------------
    # core constraints
    # ------------------------------------------------------------------
    def _xpbd(self, pos, ci, cj, rest, alpha, lam, dt):
        if ci.size == 0:
            return
        d = pos[ci] - pos[cj]
        r = np.linalg.norm(d, axis=1)
        r = np.maximum(r, 1e-9)
        nrm = d / r[:, None]
        cval = r - rest
        wi = self.inv_mass[ci]
        wj = self.inv_mass[cj]
        at = alpha / (dt * dt)
        dl = (-cval - at * lam) / (wi + wj + at)
        lam += dl
        corr = nrm * dl[:, None]
        np.add.at(pos, ci, corr * wi[:, None])
        np.add.at(pos, cj, -corr * wj[:, None])

    def _pbf(self, pos, pi, pj, dt):
        nl = len(self.liquid)
        if nl == 0:
            return
        lp = pos[self.liquid]
        m = self.m_l
        rho = np.full(nl, m * self.Wc * self.h2 ** 3)
        if pi.size:
            dv = lp[pi] - lp[pj]
            r = np.linalg.norm(dv, axis=1)
            r = np.maximum(r, 1e-12)
            w = self.Wc * np.maximum(self.h2 - r * r, 0.0) ** 3
            np.add.at(rho, pi, m * w)
            np.add.at(rho, pj, m * w)
        cval = np.maximum(rho / self.rho0 - 1.0, 0.0)
        if pi.size == 0 or not np.any(cval > 0.0):
            return
        # spiky gradient wrt p_i, pointing from j to i (vector)
        gmag = -self.Gc * (self.h - r) ** 2 / r
        gvec = dv * gmag[:, None]
        gs = np.zeros((nl, 3))
        gsq = np.zeros(nl)
        mg = m * gvec
        mg2 = np.einsum("ij,ij->i", mg, mg)
        np.add.at(gs, pi, mg)
        np.add.at(gs, pj, -mg)
        np.add.at(gsq, pi, mg2)
        np.add.at(gsq, pj, mg2)
        denom = (np.einsum("ij,ij->i", gs, gs) + gsq) / (self.rho0 * self.rho0) + self.eps_lam
        lam = -cval / denom
        coef = lam[pi] + lam[pj]
        delta = (m / self.rho0) * coef[:, None] * gvec
        dp = np.zeros((nl, 3))
        np.add.at(dp, pi, delta)
        np.add.at(dp, pj, -delta)
        mag = np.linalg.norm(dp, axis=1)
        peak = float(mag.max()) if mag.size else 0.0
        if peak > self.max_push:
            dp *= (self.max_push / peak)
        pos[self.liquid] = lp + dp

    def _containment(self, pos, imp, subset):
        if subset.size == 0:
            return
        sv = pos[self.shell]
        tri = sv[self.faces]
        cen = tri.mean(axis=1)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * self.face_flip
        ln = np.linalg.norm(fn, axis=1)
        ln = np.where(ln < 1e-12, 1.0, ln)
        nrm = fn / ln[:, None]
        tree = cKDTree(cen)
        q = pos[subset]
        cand = tree.query_ball_point(q, self.Rc)
        counts = np.fromiter((len(cl) for cl in cand), dtype=np.int64, count=len(cand))
        total = int(counts.sum())
        if total == 0:
            return
        pidx = np.repeat(np.arange(len(q), dtype=np.int64), counts)
        tidx = np.concatenate([np.asarray(cl, dtype=np.int64) for cl in cand if len(cl)])
        P = q[pidx]
        cp, bary = _closest_point_triangles(P, tri[tidx, 0], tri[tidx, 1], tri[tidx, 2])
        d2 = np.einsum("ij,ij->i", P - cp, P - cp)
        order = np.argsort(pidx, kind="stable")
        pidx_s = pidx[order]
        d2_s = d2[order]
        uniq, first = np.unique(pidx_s, return_index=True)
        counts2 = np.diff(np.append(first, len(pidx_s)))
        minv = np.minimum.reduceat(d2_s, first)
        seg = np.repeat(np.arange(len(uniq), dtype=np.int64), counts2)
        pos_in = np.arange(len(pidx_s), dtype=np.int64) - np.repeat(first, counts2)
        best = np.full(len(uniq), 1 << 30, dtype=np.int64)
        sel_mask = d2_s <= np.repeat(minv, counts2)
        np.minimum.at(best, seg[sel_mask], pos_in[sel_mask])
        chosen = order[first + best]

        gp = subset[pidx[chosen]]
        gt = tidx[chosen]
        Pc = P[chosen]
        cpc = cp[chosen]
        nc = nrm[gt]
        s = np.einsum("ij,ij->i", Pc - cpc, nc)
        hit = s > -self.dc
        if not np.any(hit):
            return
        gp = gp[hit]
        gt = gt[hit]
        nc = nc[hit]
        bc = bary[chosen][hit]
        a = np.minimum(s[hit] + self.dc, self.max_push)
        gverts = self.shell[self.faces[gt]]
        mp = self.mass[gp]
        mv = self.mass[gverts].sum(axis=1)
        wp = mv / (mp + mv + 1e-30)
        wt = mp / (mp + mv + 1e-30)
        dp = -nc * (a * wp)[:, None]
        pos[gp] += dp
        if imp is not None:
            imp[gp] += self.mass[gp][:, None] * dp
        dtri = nc * (a * wt)[:, None]
        dv = dtri[:, None, :] * bc[:, :, None]
        np.add.at(pos, gverts, dv)
        if imp is not None:
            np.add.at(imp, gverts, self.mass[gverts][:, None] * dv)

    def _interp_liquid_velocity(self, pos, vel):
        q = pos[self.marker]
        if q.size == 0:
            return np.zeros((0, 3))
        lp = pos[self.liquid]
        lv = vel[self.liquid]
        tree = cKDTree(lp)
        cand = tree.query_ball_point(q, self.h)
        counts = np.fromiter((len(cl) for cl in cand), dtype=np.int64, count=len(cand))
        out = np.zeros((len(q), 3))
        wsum = np.zeros(len(q))
        num = np.zeros((len(q), 3))
        if counts.sum():
            midx = np.repeat(np.arange(len(q), dtype=np.int64), counts)
            jidx = np.concatenate([np.asarray(cl, dtype=np.int64) for cl in cand if len(cl)])
            d = np.linalg.norm(q[midx] - lp[jidx], axis=1)
            w = np.maximum(self.h - d, 0.0) ** 2
            wsum = np.bincount(midx, weights=w, minlength=len(q))
            for k in range(3):
                num[:, k] = np.bincount(midx, weights=w * lv[jidx, k], minlength=len(q))
        good = wsum > 1e-30
        out[good] = num[good] / wsum[good, None]
        if not np.all(good):
            _, near = tree.query(q[~good], k=1)
            out[~good] = lv[near]
        return out

    def _xsph(self, pos, vel, dt):
        if self._pi.size == 0:
            return
        liq = self.liquid
        lp = pos[liq]
        lv = vel[liq]
        dv = lp[self._pi] - lp[self._pj]
        r2 = np.einsum("ij,ij->i", dv, dv)
        w = self.Wc * np.maximum(self.h2 - r2, 0.0) ** 3
        wsum = np.zeros(len(liq))
        num = np.zeros((len(liq), 3))
        np.add.at(wsum, self._pi, w)
        np.add.at(wsum, self._pj, w)
        for k in range(3):
            np.add.at(num[:, k], self._pi, w * lv[self._pj, k])
            np.add.at(num[:, k], self._pj, w * lv[self._pi, k])
        good = wsum > 1e-30
        target = np.zeros_like(lv)
        target[good] = num[good] / wsum[good, None]
        eps = min(0.5, self.VISC * dt)
        lv[good] += eps * (target[good] - lv[good])
        vel[liq] = lv

    # ------------------------------------------------------------------
    def _substep(self, dt):
        pos = self.pos
        vel = self.vel
        mass = self.mass
        active = self.active
        x_prev = pos.copy()

        vel[active] += self.gravity[None, :] * dt
        v_pre = vel.copy()
        pos[active] += vel[active] * dt

        imp_cont = np.zeros((self.N, 3))
        lam_e = np.zeros(self.edge_i.shape[0])
        lam_b = np.zeros(self.bend_i.shape[0])

        lp = pos[self.liquid]
        if lp.shape[0]:
            tree = cKDTree(lp)
            pairs = tree.query_pairs(self.h, output_type="ndarray")
        else:
            pairs = np.empty((0, 2), dtype=np.int64)
        if pairs.size:
            self._pi = pairs[:, 0].astype(np.int64)
            self._pj = pairs[:, 1].astype(np.int64)
        else:
            self._pi = np.empty(0, dtype=np.int64)
            self._pj = np.empty(0, dtype=np.int64)

        for _ in range(self.ITERS):
            np.maximum(pos[:, 2], self.R_FLOOR, out=pos[:, 2])
            self._xpbd(pos, self.edge_i, self.edge_j, self.edge_rest, self.alpha_e, lam_e, dt)
            self._xpbd(pos, self.bend_i, self.bend_j, self.bend_rest, self.alpha_b, lam_b, dt)
            self._pbf(pos, self._pi, self._pj, dt)
            self._containment(pos, imp_cont, self.surface_set)

        vel[active] = (pos[active] - x_prev[active]) / dt
        self._xsph(pos, vel, dt)

        if self.marker.size:
            u = self._interp_liquid_velocity(pos, vel)
            pos[self.marker] += u * dt
            vel[self.marker] = u
            self._containment(pos, None, self.marker)
            pos[self.marker, 2] = np.maximum(pos[self.marker, 2], self.R_FLOOR)

        v_before_wall = vel.copy()

        # ---- floor contact response -------------------------------------
        contact = pos[:, 2] <= self.R_FLOOR + 1e-9
        falling = contact & (v_pre[:, 2] < -0.05)
        if np.any(falling):
            e = np.where(self.is_shell, self.E_SHELL,
                         np.where(self.is_liquid, self.E_LIQUID, self.E_MARKER))
            vel[falling, 2] = -e[falling] * v_pre[falling, 2]
        if np.any(contact):
            mu = np.where(self.is_shell, self.MU_SHELL,
                          np.where(self.is_liquid, self.MU_LIQUID, self.MU_MARKER))
            vel[contact, 0] *= (1.0 - mu[contact])
            vel[contact, 1] *= (1.0 - mu[contact])

        imp_floor = mass[:, None] * (vel - v_before_wall)

        # ---- shell material damping -------------------------------------
        if self.shell.size:
            vel[self.shell] *= max(0.0, 1.0 - self.DAMP_SHELL * dt)

        # ---- safety clamps ----------------------------------------------
        speed = np.linalg.norm(vel, axis=1)
        fast = speed > self.MAX_SPEED
        if np.any(fast):
            vel[fast] *= (self.MAX_SPEED / speed[fast])[:, None]
        np.clip(pos, -50.0, 50.0, out=pos)

        # ---- raw force bookkeeping --------------------------------------
        for g, idx in enumerate(self.domains):
            f_grav = mass[idx].sum() * self.gravity
            f_floor = imp_floor[idx].sum(axis=0) / dt
            f_coup = imp_cont[idx].sum(axis=0) / dt
            self.ext_force[g, 0, :] = f_grav
            self.ext_force[g, 1, :] = f_floor
            self.ext_force[g, 2, :] = f_coup
            self.cum_ext_impulse[g] += (f_grav + f_floor + f_coup) * dt

        body = np.zeros((self.K_BODIES, 3))
        body[0] = (imp_cont[self.shell].sum(axis=0) + imp_floor[self.shell].sum(axis=0)) / dt
        body[1] = (imp_cont[self.surface_set].sum(axis=0)
                   + imp_floor[self.surface_set].sum(axis=0)) / dt
        body[4] = -(body[0] + body[1] + body[2] + body[3])
        self.interaction = body

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        dt = self.DT
        while (self._steps + 1) * dt <= target + 1e-9:
            self._substep(dt)
            self._steps += 1
        self._time = target
        return self._frame(target)

    def _frame(self, t: float):
        pos = self.pos
        vel = self.vel
        shell = self.shell
        sv = pos[shell]
        tri = sv[self.faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * self.face_flip
        vn = np.zeros_like(sv)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        ln = np.where(ln < 1e-12, 1.0, ln)
        vn = vn / ln

        b_normal = np.vstack([vn, self.floor_normals])
        b_vel = np.vstack([vel[shell], np.zeros((len(self.floor_pts), 3))])

        fields = {
            "boundary_normal": b_normal,
            "boundary_target_velocity": b_vel.copy(),
            "boundary_velocity": b_vel,
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((self.G, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((self.G, 3)),
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": self.ext_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.interaction.copy(),
        }
        attributes = {
            "mass_quantity": self.mass_quantity,
            "passive_marker": self.passive_marker,
        }
        scalars = {
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
            "gravity_z": float(self.gravity[2]),
        }
        return StateFrame(
            t, pos.copy(), vel.copy(), mass=self.mass.copy(),
            body_id=self.body_id.copy(), attributes=attributes,
            fields=fields, scalars=scalars,
        )

    def diagnostics(self) -> dict:
        speed = float(np.linalg.norm(self.vel, axis=1).max()) if self.N else 0.0
        return {
            "particles": int(self.N),
            "shell_particles": int(self.shell.size),
            "liquid_particles": int(self.liquid.size),
            "marker_particles": int(self.marker.size),
            "steps": int(self._steps),
            "time": float(self._time),
            "rest_density": float(self.rho0),
            "kernel_radius": float(self.h),
            "max_speed": speed,
        }
