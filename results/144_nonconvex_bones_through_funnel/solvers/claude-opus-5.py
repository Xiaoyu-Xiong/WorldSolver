"""Self-contained rigid-body solver: nonconvex bones falling through a funnel.

Physical model
-------------
* Each dynamic body is a rigid body whose collision proxy is a sphere
  decomposition derived directly from the frozen triangle-mesh asset
  (slice-wise inscribed-sphere fit on an inside/outside voxel sampling of the
  mesh).  The decomposition is nonconvex (knobs + shaft), so bridging and
  multi-point jamming emerge from the geometry rather than from scripting.
* Contacts (body/body, body/funnel-mesh, body/floor-box) use a damped penalty
  (Kelvin-Voigt) normal law with the damping ratio derived from the frozen
  restitution, plus regularized Coulomb friction with the frozen coefficient.
* Time integration is semi-implicit (symplectic) Euler on linear momentum and
  on angular momentum in the world frame (including the gyroscopic term), with
  quaternion integration and renormalization.

No external physics engine is used; only NumPy.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sdiv(a, b):
    ok = np.abs(b) > 1e-20
    return np.where(ok, a / np.where(ok, b, 1.0), 0.0)


def _load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0] - 1, idx[k] - 1, idx[k + 1] - 1))
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _components(F):
    """Label connected components of a triangle soup (union-find over vertices)."""
    if len(F) == 0:
        return np.zeros(0, np.int64)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in F:
        union(int(f[0]), int(f[1]))
        union(int(f[1]), int(f[2]))
    roots = {}
    lab = np.empty(len(F), np.int64)
    for i, f in enumerate(F):
        r = find(int(f[0]))
        if r not in roots:
            roots[r] = len(roots)
        lab[i] = roots[r]
    return lab


def _mesh_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _quat_mat(q):
    q = np.asarray(q, float).reshape(-1, 4)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(nrm > 1e-12, nrm, 1.0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R
def _closest_on_tri(P, A, B, C):
    """Vectorized closest point on triangles (Ericson, Real-Time Collision Detection)."""
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

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    inv = _sdiv(1.0, denom)
    v = vb * inv
    w = vc * inv
    Q = A + v[:, None] * ab + w[:, None] * ac

    # edge AB
    m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if m.any():
        t = np.clip(_sdiv(d1, d1 - d3), 0.0, 1.0)[:, None]
        Q = np.where(m[:, None], A + t * ab, Q)
    # edge AC
    m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if m.any():
        t = np.clip(_sdiv(d2, d2 - d6), 0.0, 1.0)[:, None]
        Q = np.where(m[:, None], A + t * ac, Q)
    # edge BC
    m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if m.any():
        t = np.clip(_sdiv(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)[:, None]
        Q = np.where(m[:, None], B + t * (C - B), Q)
    # vertex regions
    m = (d1 <= 0.0) & (d2 <= 0.0)
    Q = np.where(m[:, None], A, Q)
    m = (d3 >= 0.0) & (d4 <= d3)
    Q = np.where(m[:, None], B, Q)
    m = (d6 >= 0.0) & (d5 <= d6)
    Q = np.where(m[:, None], C, Q)
    # degenerate triangles
    Q = np.where((np.abs(denom) <= 1e-20)[:, None], A, Q)
    return Q


def _inside_points(V, F, spacing):
    """Grid samples strictly inside a closed mesh via +z ray-parity counting."""
    lo = V.min(0) - 1e-6
    hi = V.max(0) + 1e-6
    axes = []
    for k in range(3):
        n = max(3, int(math.ceil((hi[k] - lo[k]) / spacing)))
        axes.append(lo[k] + (np.arange(n) + 0.5) * (hi[k] - lo[k]) / n)
    X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
    cols = np.column_stack([X.ravel(), Y.ravel()])
    Z = axes[2]

    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    inside = np.zeros((len(cols), len(Z)), bool)
    # 2D barycentric test in xy, then z of intersection
    ax, ay = A[:, 0], A[:, 1]
    v0 = B[:, :2] - A[:, :2]
    v1 = C[:, :2] - A[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good = np.abs(den) > 1e-14
    for i in range(len(cols)):
        px, py = cols[i]
        wx = px - ax
        wy = py - ay
        s = (wx * v1[:, 1] - wy * v1[:, 0])
        t = (wy * v0[:, 0] - wx * v0[:, 1])
        s = _sdiv(s, den)
        t = _sdiv(t, den)
        hit = good & (s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)
        if not hit.any():
            continue
        zi = (A[hit, 2] + s[hit] * (B[hit, 2] - A[hit, 2])
              + t[hit] * (C[hit, 2] - A[hit, 2]))
        # parity of crossings above each sample z
        cnt = (zi[None, :] > Z[:, None]).sum(axis=1)
        inside[i] = (cnt % 2) == 1
    ii, jj = np.nonzero(inside)
    if len(ii) == 0:
        return np.zeros((0, 3))
    return np.column_stack([cols[ii, 0], cols[ii, 1], Z[jj]])


def _fit_spheres(V, F, spacing=0.022, slab=0.075):
    """Sphere decomposition of a mesh: slabs along the longest axis, each slab
    covered by inscribed-ish spheres placed along the slab's major direction."""
    ext = V.max(0) - V.min(0)
    axis = int(np.argmax(ext))
    o1, o2 = [k for k in range(3) if k != axis]
    P = _inside_points(V, F, spacing)
    if len(P) < 8:
        c = 0.5 * (V.max(0) + V.min(0))
        return np.array([c]), np.array([0.5 * float(np.max(ext))])

    lo = P[:, axis].min()
    hi = P[:, axis].max()
    nslab = max(1, int(round((hi - lo) / slab)))
    edges = np.linspace(lo - 1e-9, hi + 1e-9, nslab + 1)
    half_t = 0.5 * (edges[1] - edges[0])

    centers = []
    radii = []
    for s in range(nslab):
        m = (P[:, axis] >= edges[s]) & (P[:, axis] <= edges[s + 1])
        if m.sum() < 2:
            continue
        Q = P[m]
        a_mid = 0.5 * (edges[s] + edges[s + 1])
        c1 = Q[:, o1]
        c2 = Q[:, o2]
        w1 = c1.max() - c1.min()
        w2 = c2.max() - c2.min()
        if w1 >= w2:
            maj, mnr = o1, o2
            wmaj, wmin = w1, w2
        else:
            maj, mnr = o2, o1
            wmaj, wmin = w2, w1
        r = 0.5 * max(wmin, spacing)
        r = float(math.sqrt(r * r + half_t * half_t)) if half_t < r else float(
            max(r, half_t))
        nsub = max(1, int(round(wmaj / max(2.0 * r, 1e-6))))
        mid = 0.5 * (Q[:, maj].max() + Q[:, maj].min())
        if nsub == 1:
            offs = [mid]
        else:
            span = 0.5 * max(wmaj - 2.0 * r, 0.0)
            offs = list(np.linspace(mid - span, mid + span, nsub))
        for off in offs:
            c = np.zeros(3)
            c[axis] = a_mid
            c[maj] = off
            c[mnr] = 0.5 * (Q[:, mnr].max() + Q[:, mnr].min())
            centers.append(c)
            radii.append(r)

    centers = np.asarray(centers, float).reshape(-1, 3)
    radii = np.asarray(radii, float)
    if len(centers) == 0:
        c = 0.5 * (V.max(0) + V.min(0))
        return np.array([c]), np.array([0.5 * float(np.max(ext))])

    # prune spheres contained in a larger one
    order = np.argsort(-radii)
    centers, radii = centers[order], radii[order]
    keep = []
    for i in range(len(radii)):
        red = False
        for j in keep:
            if np.linalg.norm(centers[i] - centers[j]) + radii[i] <= radii[j] + 1e-9:
                red = True
                break
        if not red:
            keep.append(i)
    return centers[keep], radii[keep]


def _mass_props(centers, radii, mass):
    vol = radii ** 3
    tot = float(vol.sum())
    if tot <= 0.0:
        vol = np.ones_like(radii)
        tot = float(vol.sum())
    m = mass * vol / tot
    com = (m[:, None] * centers).sum(0) / mass
    d = centers - com
    I = np.zeros((3, 3))
    for k in range(len(radii)):
        I += 0.4 * m[k] * radii[k] ** 2 * np.eye(3)
        I += m[k] * (float(d[k] @ d[k]) * np.eye(3) - np.outer(d[k], d[k]))
    I = 0.5 * (I + I.T)
    ev = np.linalg.eigvalsh(I)
    if ev.min() <= 1e-9:
        I += np.eye(3) * (1e-9 - min(ev.min(), 0.0) + 1e-9)
    return com, m, I
class BenchmarkSolver:
    """Penalty-contact rigid-body solver with mesh-derived sphere proxies."""

    # numerical / material parameters
    DT = 2.0e-4
    K_N = 2.0e4
    DELTA_CAP = 6.0e-3
    C_T = 80.0
    F_MAX = 260.0
    F_BODY_MAX = 900.0
    T_BODY_MAX = 90.0
    LIN_DAMP = 0.02
    ANG_DAMP = 0.15
    V_MAX = 20.0
    W_MAX = 40.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        contact = dict(cfg.get("contact", {}) or {})
        self.mu = float(contact.get("friction", 0.5))
        self.restitution = float(contact.get("restitution", 0.1))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 1.0))

        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, float).copy()
        q0 = np.asarray(
            frame0.orientation
            if frame0.orientation is not None
            else np.tile([0.0, 0.0, 0.0, 1.0], (len(x0), 1)),
            float,
        ).copy()
        self.mass = np.asarray(frame0.mass, float).copy()
        self.body_id = (
            np.asarray(frame0.body_id, np.int64).copy()
            if frame0.body_id is not None
            else np.arange(len(x0), dtype=np.int64)
        )
        self.n = len(x0)

        assets = self._load_geometry()
        # ---- collision proxies for the dynamic bodies -------------------
        templates = {}
        local, radii, sph_body = [], [], []
        self.com_local = np.zeros((self.n, 3))
        Ib = np.zeros((self.n, 3, 3))
        for b in range(self.n):
            path = assets["dynamic"][b] if b < len(assets["dynamic"]) else None
            if path not in templates:
                if path is not None and os.path.isfile(path):
                    V, F = _load_obj(path)
                    c, r = _fit_spheres(V, F)
                else:
                    c = np.zeros((1, 3))
                    r = np.array([0.1])
                templates[path] = (c, r)
            c, r = templates[path]
            com, _m, I = _mass_props(c, r, float(self.mass[b]))
            self.com_local[b] = com
            Ib[b] = I
            local.append(c - com)
            radii.append(r)
            sph_body.append(np.full(len(r), b, np.int64))

        self.sph_local = np.concatenate(local, axis=0)
        self.sph_r = np.concatenate(radii, axis=0)
        self.sph_body = np.concatenate(sph_body, axis=0)
        self.Ib = Ib
        self.Ibi = np.linalg.inv(Ib)
        self.inv_mass = 1.0 / self.mass

        s = len(self.sph_r)
        ii, jj = np.triu_indices(s, k=1)
        keep = self.sph_body[ii] != self.sph_body[jj]
        self.pi = ii[keep]
        self.pj = jj[keep]

        # ---- static colliders -------------------------------------------
        self.tri_a = np.zeros((0, 3))
        self.tri_b = np.zeros((0, 3))
        self.tri_c = np.zeros((0, 3))
        self.tri_n = np.zeros((0, 3))
        self.tri_comp = np.zeros(0, np.int64)
        ncomp = 0
        for V, F in assets["static_meshes"]:
            A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            nrm = np.cross(B - A, C - A)
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.where(ln > 1e-14, ln, 1.0)
            lab = _components(F)
            # orient each closed component outward (signed volume > 0)
            for cid in range(int(lab.max()) + 1 if len(lab) else 0):
                m = lab == cid
                if _mesh_volume(V, F[m]) < 0.0:
                    Bm = B[m].copy()
                    B[m] = C[m]
                    C[m] = Bm
                    nrm[m] = -nrm[m]
            lab = lab + ncomp
            ncomp = int(lab.max()) + 1 if len(lab) else ncomp
            self.tri_a = np.vstack([self.tri_a, A])
            self.tri_b = np.vstack([self.tri_b, B])
            self.tri_c = np.vstack([self.tri_c, C])
            self.tri_n = np.vstack([self.tri_n, nrm])
            self.tri_comp = np.concatenate([self.tri_comp, lab])
        self.n_comp = max(1, ncomp)
        allv = np.stack([self.tri_a, self.tri_b, self.tri_c], axis=1)
        self.tri_lo = allv.min(axis=1) if len(allv) else np.zeros((0, 3))
        self.tri_hi = allv.max(axis=1) if len(allv) else np.zeros((0, 3))
        self.boxes = assets["boxes"]

        # ---- state -------------------------------------------------------
        R0 = _quat_mat(q0)
        self.x = x0 + np.einsum("nij,nj->ni", R0, self.com_local)
        self.q = q0
        self.v = np.zeros((self.n, 3))
        self.w = np.zeros((self.n, 3))
        self.nsteps = 0

        m_eff = float(np.mean(self.mass)) * 0.5
        e = min(max(self.restitution, 1e-3), 0.999)
        le = math.log(e)
        zeta = -le / math.sqrt(math.pi * math.pi + le * le)
        self.C_N = 2.0 * zeta * math.sqrt(self.K_N * max(m_eff, 1e-6))
        self.margin = 2.0 * self.length_scale
        self._max_pen = 0.0
        self._last_contacts = 0

    # ------------------------------------------------------------------
    def _load_geometry(self):
        out = {"dynamic": [], "static_meshes": [], "boxes": []}
        geom_path = os.path.join(_ROOT, "scene_geometry.yaml")
        spec = None
        try:
            import yaml

            with open(geom_path, "r") as fh:
                spec = yaml.safe_load(fh)
        except Exception:
            spec = None
        if not spec:
            bone = os.path.join(_ROOT, "assets", "bone.obj")
            out["dynamic"] = [bone] * self.n
            fun = os.path.join(_ROOT, "assets", "funnel.obj")
            if os.path.isfile(fun):
                V, F = _load_obj(fun)
                out["static_meshes"].append((V, F))
            out["boxes"].append(
                (np.array([-3.2, -2.2, -0.1]), np.array([3.2, 2.2, 0.0]))
            )
            return out

        for obj in spec.get("objects", []) or []:
            geo = obj.get("geometry", {}) or {}
            tf = obj.get("transform", {}) or {}
            tr = np.asarray(tf.get("translation", [0.0, 0.0, 0.0]), float)
            dynamic = str(obj.get("motion", "static")) == "dynamic"
            if geo.get("type") == "triangle_mesh":
                path = os.path.join(_ROOT, str(geo.get("asset", "")))
                if dynamic:
                    out["dynamic"].append(path)
                elif os.path.isfile(path):
                    V, F = _load_obj(path)
                    out["static_meshes"].append((V + tr, F))
            elif geo.get("type") == "box":
                he = np.asarray(geo.get("half_extents", [1.0, 1.0, 0.1]), float)
                if not dynamic:
                    out["boxes"].append((tr - he, tr + he))
        if not out["dynamic"]:
            out["dynamic"] = [os.path.join(_ROOT, "assets", "bone.obj")] * self.n
        return out
    # ------------------------------------------------------------------
    def _world_spheres(self, R):
        return self.x[self.sph_body] + np.einsum(
            "sij,sj->si", R[self.sph_body], self.sph_local
        )

    def _pair_force(self, bi, cp, nA, depth, bj=None):
        """Damped-penalty normal law + regularized Coulomb friction."""
        r_i = cp - self.x[bi]
        vrel = self.v[bi] + np.cross(self.w[bi], r_i)
        r_j = None
        if bj is not None:
            r_j = cp - self.x[bj]
            vrel = vrel - (self.v[bj] + np.cross(self.w[bj], r_j))
        vn = np.einsum("ij,ij->i", vrel, nA)
        dcap = np.clip(depth, 0.0, self.DELTA_CAP)
        fn = np.clip(self.K_N * dcap - self.C_N * vn, 0.0, self.F_MAX)
        fn = np.where(depth > 0.0, fn, 0.0)
        vt = vrel - vn[:, None] * nA
        vtn = np.linalg.norm(vt, axis=1)
        ftm = np.minimum(self.mu * fn, self.C_T * vtn)
        ft = -(_sdiv(ftm, vtn))[:, None] * vt
        return fn[:, None] * nA + ft, r_i, r_j

    def _contacts(self, R, collect=False):
        c = self._world_spheres(R)
        Fb = np.zeros((self.n, 3))
        Tb = np.zeros((self.n, 3))
        gaps = []
        forces = []
        ncontact = 0
        maxpen = 0.0

        # ---- body / body -------------------------------------------------
        d = c[self.pj] - c[self.pi]
        dist = np.linalg.norm(d, axis=1)
        rsum = self.sph_r[self.pi] + self.sph_r[self.pj]
        gap = dist - rsum
        sel = gap < self.margin
        min_gap = float(gap.min()) if len(gap) else 0.0
        if sel.any():
            k = np.nonzero(sel)[0]
            si, sj = self.pi[k], self.pj[k]
            bi, bj = self.sph_body[si], self.sph_body[sj]
            ds = dist[k]
            nij = _sdiv(d[k], ds[:, None])
            bad = ds <= 1e-12
            if bad.any():
                nij[bad] = np.array([0.0, 0.0, 1.0])
            nA = -nij
            g = gap[k]
            cp = c[si] + nij * (self.sph_r[si] + 0.5 * g)[:, None]
            f, r_i, r_j = self._pair_force(bi, cp, nA, -g, bj)
            np.add.at(Fb, bi, f)
            np.add.at(Tb, bi, np.cross(r_i, f))
            np.add.at(Fb, bj, -f)
            np.add.at(Tb, bj, np.cross(r_j, -f))
            ncontact += int((g < 0.0).sum())
            if len(g):
                maxpen = max(maxpen, float(max(-g.min(), 0.0)))
            if collect:
                gaps.append(g)
                forces.append(f)

        # ---- body / static triangle mesh (funnel) ------------------------
        if len(self.tri_a):
            rmax = float(self.sph_r.max())
            pad = rmax + self.margin
            lo = c.min(0) - pad
            hi = c.max(0) + pad
            tri_ok = np.nonzero(
                np.all(self.tri_hi >= lo, axis=1) & np.all(self.tri_lo <= hi, axis=1)
            )[0]
            if len(tri_ok):
                ns = len(c)
                si = np.repeat(np.arange(ns), len(tri_ok))
                ti = np.tile(tri_ok, ns)
                rr = self.sph_r[si] + self.margin
                near = np.all(
                    (c[si] + rr[:, None] >= self.tri_lo[ti])
                    & (c[si] - rr[:, None] <= self.tri_hi[ti]),
                    axis=1,
                )
                si, ti = si[near], ti[near]
                if len(si):
                    Q = _closest_on_tri(
                        c[si], self.tri_a[ti], self.tri_b[ti], self.tri_c[ti]
                    )
                    dv = c[si] - Q
                    ds = np.linalg.norm(dv, axis=1)
                    nrm = self.tri_n[ti]
                    sd = np.einsum("ij,ij->i", c[si] - self.tri_a[ti], nrm)
                    outside = sd >= 0.0
                    # outside the face plane: push along (center - closest point);
                    # behind it (center buried in the wall slab): push along the
                    # outward face normal by the full crossing depth.
                    nA = np.where(outside[:, None], _sdiv(dv, ds[:, None]), nrm)
                    degen = outside & (ds <= 1e-9)
                    if degen.any():
                        nA[degen] = nrm[degen]
                    depth = np.where(outside, self.sph_r[si] - ds, self.sph_r[si] + ds)
                    ok = depth > -self.margin
                    if ok.any():
                        si2, ti2 = si[ok], ti[ok]
                        nA, depth, Qk = nA[ok], depth[ok], Q[ok]
                        # keep the single deepest face per (sphere, wall component)
                        keys = si2 * self.n_comp + self.tri_comp[ti2]
                        order = np.lexsort((-depth, keys))
                        keys_s = keys[order]
                        first = np.empty(len(keys_s), bool)
                        first[0] = True
                        if len(keys_s) > 1:
                            first[1:] = keys_s[1:] != keys_s[:-1]
                        sel_t = order[first]
                        si2 = si2[sel_t]
                        nA = nA[sel_t]
                        depth = depth[sel_t]
                        Qk = Qk[sel_t]
                        g2 = -depth
                        bi = self.sph_body[si2]
                        f, r_i, _ = self._pair_force(bi, Qk, nA, depth)
                        np.add.at(Fb, bi, f)
                        np.add.at(Tb, bi, np.cross(r_i, f))
                        ncontact += int((g2 < 0.0).sum())
                        if len(g2):
                            maxpen = max(maxpen, float(max(-g2.min(), 0.0)))
                        if collect:
                            gaps.append(g2)
                            forces.append(f)

        # ---- body / static box (floor) -----------------------------------
        for blo, bhi in self.boxes:
            Q = np.clip(c, blo, bhi)
            dv = c - Q
            ds = np.linalg.norm(dv, axis=1)
            inside = ds <= 1e-9
            nA = _sdiv(dv, ds[:, None])
            if inside.any():
                nA[inside] = np.array([0.0, 0.0, 1.0])
                Q[inside, 2] = bhi[2]
            g = np.where(inside, -self.sph_r, ds - self.sph_r)
            ok = g < self.margin
            if ok.any():
                idx = np.nonzero(ok)[0]
                bi = self.sph_body[idx]
                f, r_i, _ = self._pair_force(bi, Q[idx], nA[idx], -g[idx])
                np.add.at(Fb, bi, f)
                np.add.at(Tb, bi, np.cross(r_i, f))
                ncontact += int((g[idx] < 0.0).sum())
                maxpen = max(maxpen, float(max(-g[idx].min(), 0.0)))
                if collect:
                    gaps.append(g[idx])
                    forces.append(f)

        self._last_contacts = ncontact
        self._max_pen = max(self._max_pen, maxpen)
        if not collect:
            return Fb, Tb, None, None
        if gaps:
            gap_arr = np.concatenate(gaps)
            f_arr = np.concatenate(forces, axis=0)
        else:
            gap_arr = np.array([min_gap])
            f_arr = np.zeros((1, 3))
        return Fb, Tb, gap_arr, f_arr
    # ------------------------------------------------------------------
    def _step(self):
        R = _quat_mat(self.q)
        Fb, Tb, _, _ = self._contacts(R)

        fn = np.linalg.norm(Fb, axis=1)
        scale = np.where(fn > self.F_BODY_MAX, self.F_BODY_MAX / np.maximum(fn, 1e-12), 1.0)
        Fb = Fb * scale[:, None]
        tn = np.linalg.norm(Tb, axis=1)
        scale = np.where(tn > self.T_BODY_MAX, self.T_BODY_MAX / np.maximum(tn, 1e-12), 1.0)
        Tb = Tb * scale[:, None]

        dt = self.DT
        # linear momentum (semi-implicit Euler)
        acc = Fb * self.inv_mass[:, None] + self.g[None, :]
        self.v = self.v + dt * acc
        # angular momentum in the world frame, incl. gyroscopic term
        Iw = np.einsum("nij,njk,nlk->nil", R, self.Ib, R)
        Iwi = np.einsum("nij,njk,nlk->nil", R, self.Ibi, R)
        Lw = np.einsum("nij,nj->ni", Iw, self.w)
        alpha = np.einsum("nij,nj->ni", Iwi, Tb - np.cross(self.w, Lw))
        self.w = self.w + dt * alpha

        # artificial damping (aerodynamic-like, keeps piles settling)
        self.v *= max(0.0, 1.0 - self.LIN_DAMP * dt)
        self.w *= max(0.0, 1.0 - self.ANG_DAMP * dt)
        vn = np.linalg.norm(self.v, axis=1)
        s = np.where(vn > self.V_MAX, self.V_MAX / np.maximum(vn, 1e-12), 1.0)
        self.v *= s[:, None]
        wn = np.linalg.norm(self.w, axis=1)
        s = np.where(wn > self.W_MAX, self.W_MAX / np.maximum(wn, 1e-12), 1.0)
        self.w *= s[:, None]

        self.x = self.x + dt * self.v
        # quaternion integration (xyzw) and renormalization
        wq = self.w
        qx, qy, qz, qw = self.q[:, 0], self.q[:, 1], self.q[:, 2], self.q[:, 3]
        dq = np.empty_like(self.q)
        dq[:, 0] = qw * wq[:, 0] + qy * wq[:, 2] - qz * wq[:, 1]
        dq[:, 1] = qw * wq[:, 1] + qz * wq[:, 0] - qx * wq[:, 2]
        dq[:, 2] = qw * wq[:, 2] + qx * wq[:, 1] - qy * wq[:, 0]
        dq[:, 3] = -(qx * wq[:, 0] + qy * wq[:, 1] + qz * wq[:, 2])
        self.q = self.q + 0.5 * dt * dq
        nrm = np.linalg.norm(self.q, axis=1, keepdims=True)
        self.q = self.q / np.where(nrm > 1e-12, nrm, 1.0)

        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        self.q = np.nan_to_num(self.q, nan=0.0, posinf=0.0, neginf=0.0)
        bad = np.linalg.norm(self.q, axis=1) < 1e-6
        if bad.any():
            self.q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        self.nsteps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.DT))
        while self.nsteps < target:
            self._step()

        R = _quat_mat(self.q)
        _Fb, _Tb, gaps, fpair = self._contacts(R, collect=True)
        gaps = np.asarray(gaps, float).reshape(-1)
        fpair = np.asarray(fpair, float).reshape(-1, 3)
        if gaps.size == 0:
            gaps = np.zeros(1)
            fpair = np.zeros((1, 3))
        if gaps.size > 4096:
            order = np.argsort(gaps, kind="stable")[:4096]
            order = np.sort(order)
            gaps = gaps[order]
            fpair = fpair[order]
        gaps = np.nan_to_num(gaps, nan=0.0, posinf=0.0, neginf=0.0)
        fpair = np.nan_to_num(fpair, nan=0.0, posinf=0.0, neginf=0.0)

        pos = self.x - np.einsum("nij,nj->ni", R, self.com_local)
        ext = self.mass[:, None] * self.g[None, :]
        frame = StateFrame(
            time=max(0.0, float(self.nsteps * self.DT)),
            position=np.ascontiguousarray(pos, float),
            velocity=np.ascontiguousarray(self.v, float),
            orientation=np.ascontiguousarray(self.q, float),
            angular_velocity=np.ascontiguousarray(self.w, float),
            mass=np.ascontiguousarray(self.mass, float),
            body_id=np.ascontiguousarray(self.body_id, np.int32),
            fields={
                "contact_signed_distance": gaps,
                "external_force": np.ascontiguousarray(ext, float),
                "interaction_force": fpair,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "penalty-contact rigid bodies, sphere-decomposed meshes",
            "integrator": "semi-implicit Euler + quaternion, world-frame angular momentum",
            "dt": float(self.DT),
            "steps": int(self.nsteps),
            "spheres": int(len(self.sph_r)),
            "bodies": int(self.n),
            "static_triangles": int(len(self.tri_a)),
            "normal_stiffness": float(self.K_N),
            "normal_damping": float(self.C_N),
            "friction": float(self.mu),
            "restitution": float(self.restitution),
            "max_penetration": float(self._max_pen),
            "last_active_contacts": int(self._last_contacts),
        }
