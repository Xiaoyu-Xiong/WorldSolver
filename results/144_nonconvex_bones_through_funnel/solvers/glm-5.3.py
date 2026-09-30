"""Impulse-based rigid-body solver: nonconvex bones through a funnel.

Self-contained physics engine (no genesis): semi-implicit Euler time stepping
with sequential-impulse contact resolution (Coulomb friction, restitution,
speculative contacts) and vertex/triangle-mesh collision detection against
low-poly proxies built procedurally from the frozen bone asset.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# tunables
# ----------------------------------------------------------------------------
DT = 0.001                 # substep (s)
DETECT_MARGIN = 0.015      # geometric query margin (m)
EXPORT_MARGIN = 0.010      # exported near-contact margin (m)
SLOP = 0.0005              # allowed penetration slop (m)
BETA = 0.25                # Baumgarte factor
MAX_BIAS = 2.0             # pushout bias velocity clamp (m/s)
SI_ITERS = 10              # sequential impulse iterations
REST_THRESH = 0.3          # restitution velocity threshold (m/s)
MAX_V = 15.0               # safety velocity clamp (m/s)
MAX_W = 60.0               # safety angular velocity clamp (rad/s)
ANG_DAMP = 0.05            # mild viscous angular damping (1/s)


# ----------------------------------------------------------------------------
# small quaternion / matrix helpers (xyzw order)
# ----------------------------------------------------------------------------
def _quat_to_mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rand_quats(rng, count):
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


# ----------------------------------------------------------------------------
# obj loading
# ----------------------------------------------------------------------------
def _load_obj(path):
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, float), np.asarray(faces, int)


def _signed_volume(verts, faces):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


class TriMesh(object):
    """Triangle soup with precomputed normals and per-triangle AABBs."""

    __slots__ = ("a", "b", "c", "n", "ctr", "h", "ntri", "verts", "faces")

    def __init__(self, verts, faces):
        if _signed_volume(verts, faces) < 0.0:
            faces = faces[:, [0, 2, 1]]
        a = verts[faces[:, 0]]
        b = verts[faces[:, 1]]
        c = verts[faces[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1)
        ln[ln < 1e-14] = 1.0
        self.verts, self.faces = verts, faces
        self.a, self.b, self.c = a, b, c
        self.n = nrm / ln[:, None]
        self.ctr = ((a + b + c) / 3.0).astype(np.float32)
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        self.h = ((hi - lo) / 2.0).astype(np.float32)
        self.ntri = len(faces)


def _closest_pt_tri(p, a, b, c):
    """Vectorized closest point on triangles (Ericson). p,a,b,c: (K,3)."""
    ab, ac, ap = b - a, c - a, p - a
    dot = np.einsum
    d1 = dot("ij,ij->i", ab, ap)
    d2 = dot("ij,ij->i", ac, ap)
    bp = p - b
    d3 = dot("ij,ij->i", ab, bp)
    d4 = dot("ij,ij->i", ac, bp)
    cp = p - c
    d5 = dot("ij,ij->i", ab, cp)
    d6 = dot("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def _safe_div(num, den):
        out = np.zeros_like(num)
        np.divide(num, den, out=out, where=np.abs(den) > 1e-14)
        return out

    denom = va + vb + vc
    v_in = _safe_div(vb, denom)
    w_in = _safe_div(vc, denom)
    p_in = a + ab * v_in[:, None] + ac * w_in[:, None]

    dd = d1 - d3
    v_ab = _safe_div(d1, dd)
    p_ab = a + ab * v_ab[:, None]
    de = d2 - d6
    w_ac = _safe_div(d2, de)
    p_ac = a + ac * w_ac[:, None]
    df = (d4 - d3) + (d5 - d6)
    w_bc = _safe_div(d4 - d3, df)
    p_bc = b + (c - b) * w_bc[:, None]

    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (d6 >= 0) & (d5 <= d6),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    pts = [a, b, c, p_ab, p_ac, p_bc]
    return np.select(conds, pts, default=p_in)


_EMPTY = (np.zeros(0, int), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))


def _query(points, mesh, margin):
    """Closest surface sample per vertex. Returns (vtx_idx, sd, q, normal)."""
    P, T = points.shape[0], mesh.ntri
    if P == 0 or T == 0:
        return _EMPTY
    lb2 = np.zeros((P, T), np.float32)
    for k in range(3):
        ad = np.abs(points[:, k, None].astype(np.float32) - mesh.ctr[None, :, k])
        ad -= mesh.h[None, :, k]
        np.maximum(ad, 0.0, out=ad)
        lb2 += ad * ad
    vi, ti = np.nonzero(lb2 < np.float32(margin * margin))
    if vi.size == 0:
        return _EMPTY
    p = points[vi]
    q = _closest_pt_tri(p, mesh.a[ti], mesh.b[ti], mesh.c[ti])
    diff = p - q
    d2 = np.einsum("ij,ij->i", diff, diff)
    sd = np.einsum("ij,ij->i", diff, mesh.n[ti])
    order = np.lexsort((d2, vi))
    vi, ti, sd, q = vi[order], ti[order], sd[order], q[order]
    d2 = d2[order]
    keep = np.ones(vi.shape, bool)
    keep[1:] = vi[1:] != vi[:-1]
    return vi[keep], sd[keep], q[keep], mesh.n[ti][keep]


# ----------------------------------------------------------------------------
# procedural low-poly proxy: generalized cylinder sampled from the asset
# ----------------------------------------------------------------------------
def _build_proxy(verts, ns=13, nseg=10):
    vmin, vmax = verts.min(0), verts.max(0)
    ax = int(np.argmax(vmax - vmin))
    others = [i for i in range(3) if i != ax]
    mp = [ax, others[0], others[1]]  # proxy col i -> world axis mp[i]
    sub = verts[:, mp]               # col 0 = long axis
    s = sub[:, 0]
    smin, smax = float(s.min()), float(s.max())
    sk = np.linspace(smin, smax, ns)
    bidx = np.argmin(np.abs(s[:, None] - sk[None, :]), axis=1)
    r1 = np.zeros(ns)
    r2 = np.zeros(ns)
    for k in range(ns):
        m = bidx == k
        if m.any():
            r1[k] = float(np.abs(sub[m, 1]).max())
            r2[k] = float(np.abs(sub[m, 2]).max())
    filled = [k for k in range(ns) if r1[k] > 1e-9]
    for k in range(ns):
        if r1[k] <= 1e-9 and filled:
            r1[k] = r1[min(filled, key=lambda j: abs(j - k))]
            r2[k] = r2[min(filled, key=lambda j: abs(j - k))]
    th = np.arange(nseg) * (2.0 * np.pi / nseg)
    ring = np.empty((ns * nseg, 3))
    for k in range(ns):
        ring[k * nseg:(k + 1) * nseg, 0] = r1[k] * np.cos(th)
        ring[k * nseg:(k + 1) * nseg, 1] = r2[k] * np.sin(th)
        ring[k * nseg:(k + 1) * nseg, 2] = sk[k]
    cb, ct = ns * nseg, ns * nseg + 1
    V = np.vstack([ring, [[0.0, 0.0, smin]], [[0.0, 0.0, smax]]])
    F = []
    for k in range(ns - 1):
        for j in range(nseg):
            j2 = (j + 1) % nseg
            v00, v01 = k * nseg + j, k * nseg + j2
            v10, v11 = (k + 1) * nseg + j, (k + 1) * nseg + j2
            F.append((v00, v01, v11))
            F.append((v00, v11, v10))
    for j in range(nseg):
        j2 = (j + 1) % nseg
        F.append((cb, j2, j))
        F.append((ct, (ns - 1) * nseg + j, (ns - 1) * nseg + j2))
    F = np.asarray(F, int)
    Vw = np.empty_like(V)
    Vw[:, mp] = V
    if _signed_volume(Vw, F) < 0.0:
        F = F[:, [0, 2, 1]]
    return Vw, F


def _mass_properties(verts, faces, mass):
    """Volume, center of mass, inertia about COM (uniform density)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vt = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = float(vt.sum())
    if abs(vol) < 1e-9:
        vol = 1e-9
    s = a + b + c
    ss = np.einsum("k,ki,kj->ij", vt / 20.0, s, s)
    pp = (np.einsum("ki,kj->kij", p, p) for p in (a, b, c))
    for term in pp:
        ss += np.einsum("k,kij->ij", vt / 20.0, term)
    com = np.einsum("k,ki->i", vt, s / 4.0) / vol
    I0 = np.trace(ss) * np.eye(3) - ss
    Ic = I0 - (float(com @ com) * np.eye(3) - np.outer(com, com))
    Ib = Ic * (mass / vol)
    return vol, com, 0.5 * (Ib + Ib.T)


class _Box(object):
    __slots__ = ("c", "h")

    def __init__(self, center, half):
        self.c = np.asarray(center, float)
        self.h = np.asarray(half, float)

    def sd(self, points):
        """Signed distance and outward normal of points vs box."""
        d = points - self.c
        cl = np.clip(d, -self.h, self.h)
        diff = d - cl
        dist = np.linalg.norm(diff, axis=1)
        outside = dist > 1e-12
        inner = self.h - np.abs(d)
        k = np.argmin(inner, axis=1)
        sd_in = -inner[np.arange(len(d)), k]
        n_in = np.sign(d) * np.eye(3)[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            n_out = diff / np.where(dist[:, None] > 1e-12, dist[:, None], 1.0)
        sd = np.where(outside, dist, sd_in)
        nrm = np.where(outside[:, None], n_out, n_in)
        return sd, nrm


# ----------------------------------------------------------------------------
# contact
# ----------------------------------------------------------------------------
class _Contact(object):
    __slots__ = (
        "ia", "ib", "px", "py", "pz", "nx", "ny", "nz", "t1x", "t1y", "t1z",
        "t2x", "t2y", "t2z", "sd", "vn0", "target", "kn", "kt1", "kt2",
        "jnA", "jnB", "jt1A", "jt1B", "jt2A", "jt2B",
        "pnA", "pnB", "pt1A", "pt1B", "pt2A", "pt2B",
        "ln", "lt1", "lt2", "active", "fx", "fy", "fz",
    )


class BenchmarkSolver(object):
    """Sequential-impulse rigid body engine for the bone-funnel scene."""

    def __init__(self):
        self.step = 0
        self.last_contacts = []

    # -- geometry loading --------------------------------------------------
    @staticmethod
    def _root():
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        if os.path.isdir(os.path.join(root, "assets")):
            return root
        return os.getcwd()

    def _asset_path(self, asset):
        root = self._root()
        for cand in (os.path.join(root, asset), os.path.join(os.getcwd(), asset), asset):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(asset)

    def _geom_objects(self, cfg):
        objs = {}
        try:
            geom = cfg.get("scene_geometry") or {}
            for obj in geom.get("objects", []):
                objs[obj.get("id")] = obj
        except Exception:
            objs = {}
        return objs

    def _floor_box(self, objs):
        try:
            obj = objs["floor"]
            half = [float(h) for h in obj["geometry"]["half_extents"]]
            tr = [float(t) for t in obj.get("transform", {}).get("translation", [0, 0, -0.05])]
            return _Box(tr, half)
        except Exception:
            return _Box([0.0, 0.0, -0.05], [3.2, 2.2, 0.05])

    def _funnel_mesh(self, objs):
        asset, tr = "assets/funnel.obj", np.zeros(3)
        try:
            obj = objs["funnel"]
            asset = obj["geometry"]["asset"]
            tr = np.asarray([float(t) for t in obj.get("transform", {}).get("translation", [0, 0, 0])])
        except Exception:
            pass
        verts, faces = _load_obj(self._asset_path(asset))
        return TriMesh(verts + tr, faces)

    def _bone_asset(self, objs):
        asset = "assets/bone.obj"
        try:
            asset = objs["bone_0"]["geometry"]["asset"]
        except Exception:
            pass
        return self._asset_path(asset)

    # -- initialization ----------------------------------------------------
    def initialize(self, scene, seed):
        cfg = {}
        try:
            cfg = scene.config or {}
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        cont = cfg.get("contact") or {}
        self.mu = float(cont.get("friction", 0.55))
        self.e = float(cont.get("restitution", 0.12))
        self.length_scale = float(cont.get("length_scale", 0.02))

        init = None
        if hasattr(scene, "initial_state"):
            try:
                init = scene.initial_state()
            except Exception:
                init = None
        if init is not None:
            pos = np.array(init.position, float)
            mass = np.array(init.mass, float)
            ids = np.array(init.body_id, int)
        else:
            bodies = cfg.get("bodies", [])
            pos = np.array([b["position"] for b in bodies], float)
            mass = np.array([b["mass"] for b in bodies], float)
            ids = np.arange(len(bodies), dtype=int)
        self.N = int(len(pos))
        self.mass = mass.astype(float)
        self.imass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-12), 0.0)
        self.ids = ids.astype(np.int32)

        objs = self._geom_objects(cfg)
        self.floor = self._floor_box(objs)
        self.funnel = self._funnel_mesh(objs)

        bone_v, bone_f = _load_obj(self._bone_asset(objs))
        pv, pf = _build_proxy(bone_v)
        _, com, Ib = _mass_properties(pv, pf, 1.0)
        self.com = com
        vcom = pv - com
        self.proxy = TriMesh(vcom, pf)
        self.pvert = np.ascontiguousarray(vcom)
        self.phalf = (vcom.max(0) - vcom.min(0)) / 2.0
        self.brad = float(np.linalg.norm(vcom, axis=1).max())
        self.pcount = vcom.shape[0]

        # per-body inertia (identical geometry, per-body mass)
        self.Ibinv, self.Iwt, self.R = [], [], []
        for i in range(self.N):
            Ib_i = Ib * self.mass[i]
            Ib_i = 0.5 * (Ib_i + Ib_i.T) + np.eye(3) * 1e-9
            self.Ibinv.append(np.linalg.inv(Ib_i))

        # initial state: frozen positions, seeded varied orientations chosen
        # to maximize starting clearance between neighbours
        rng = np.random.default_rng(int(seed))
        self.q = np.zeros((self.N, 4))
        self.xc = np.zeros((self.N, 3))
        chosen = []
        ident = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(self.N):
            cands = np.vstack([_rand_quats(rng, 24), ident[None, :]])
            if not chosen:
                q = cands[0]
            else:
                best_q, best_score = cands[0], -1e18
                for cq in cands:
                    score = 0.1
                    for j in range(i):
                        score = min(score, self._trial_gap(cq, pos[i], self.q[j], pos[j]))
                    if score > best_score:
                        best_score, best_q = score, cq
                q = cands[0] if best_score >= 0.099 else best_q
            self.q[i] = q
            chosen.append(q)
        self.v = np.zeros((self.N, 3))
        self.w = np.zeros((self.N, 3))
        self.vl = [[0.0, 0.0, 0.0] for _ in range(self.N)]
        self.wl = [[0.0, 0.0, 0.0] for _ in range(self.N)]
        self.imassl = self.imass.tolist()
        self.xcl = self.xc.tolist()
        self.xo = np.zeros((self.N, 3))
        for i in range(self.N):
            R = _quat_to_mat(self.q[i])
            self.R.append(R)
            self.xc[i] = pos[i] + R @ self.com
            self.xo[i] = self.xc[i] - R @ self.com
        self.step = 0
        self.xcl = self.xc.tolist()
        contacts = self._detect()
        self.last_contacts = [c for c in contacts if c.sd < EXPORT_MARGIN]

    def _trial_gap(self, qi, pi, qj, pj):
        """Worst signed gap between body i (trial pose) and body j (fixed)."""
        Ri = _quat_to_mat(qi)
        Rj = _quat_to_mat(qj)
        xci = pi + Ri @ self.com
        xcj = pj + Rj @ self.com
        if float(np.linalg.norm(xci - xcj)) - 2.0 * self.brad > 0.10:
            return 0.10
        Wi = self.pvert @ Ri.T + xci
        Wj = self.pvert @ Rj.T + xcj
        best = 0.10
        _, sd, _, _ = _query((Wi - xcj) @ Rj, self.proxy, 0.12)
        if sd.size:
            best = min(best, float(sd.min()))
        _, sd, _, _ = _query((Wj - xci) @ Ri, self.proxy, 0.12)
        if sd.size:
            best = min(best, float(sd.min()))
        return best

    # -- contact assembly --------------------------------------------------
    def _mk(self, ia, ib, p, n, sd):
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-9:
            return None
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        if abs(nz) < 0.9:
            tl = math.sqrt(nx * nx + ny * ny)
            t1x, t1y, t1z = ny / tl, -nx / tl, 0.0
        else:
            tl = math.sqrt(ny * ny + nz * nz)
            t1x, t1y, t1z = 0.0, nz / tl, -ny / tl
        t2x, t2y, t2z = (ny * t1z - nz * t1y, nz * t1x - nx * t1z, nx * t1y - ny * t1x)

        xax, xay, xaz = self.xcl[ia]
        vax, vay, vaz = self.vl[ia]
        wax, way, waz = self.wl[ia]
        imA = self.imassl[ia]
        a00, a01, a02, a10, a11, a12, a20, a21, a22 = self.Iwt[ia]
        if ib >= 0:
            xbx, xby, xbz = self.xcl[ib]
            vbx, vby, vbz = self.vl[ib]
            wbx, wby, wbz = self.wl[ib]
            imB = self.imassl[ib]
            b00, b01, b02, b10, b11, b12, b20, b21, b22 = self.Iwt[ib]
        else:
            xbx = xby = xbz = vbx = vby = vbz = 0.0
            wbx = wby = wbz = imB = 0.0
            b00 = b01 = b02 = b10 = b11 = b12 = b20 = b21 = b22 = 0.0

        rax, ray, raz = px - xax, py - xay, pz - xaz
        rbx, rby, rbz = px - xbx, py - xby, pz - xbz

        c = _Contact()
        c.ia, c.ib = ia, ib
        c.px, c.py, c.pz = px, py, pz
        c.nx, c.ny, c.nz = nx, ny, nz
        c.t1x, c.t1y, c.t1z = t1x, t1y, t1z
        c.t2x, c.t2y, c.t2z = t2x, t2y, t2z
        c.sd = float(sd)

        for tag, dx, dy, dz in (("n", nx, ny, nz), ("t1", t1x, t1y, t1z), ("t2", t2x, t2y, t2z)):
            cax = ray * dz - raz * dy
            cay = raz * dx - rax * dz
            caz = rax * dy - ray * dx
            aax = a00 * cax + a01 * cay + a02 * caz
            aay = a10 * cax + a11 * cay + a12 * caz
            aaz = a20 * cax + a21 * cay + a22 * caz
            cbx = rby * dz - rbz * dy
            cby = rbz * dx - rbx * dz
            cbz = rbx * dy - rby * dx
            abx = b00 * cbx + b01 * cby + b02 * cbz
            aby = b10 * cbx + b11 * cby + b12 * cbz
            abz = b20 * cbx + b21 * cby + b22 * cbz
            k = dx * dx * (imA + imB) + dy * dy * (imA + imB) + dz * dz * (imA + imB)
            k += cax * aax + cay * aay + caz * aaz
            k += cbx * abx + cby * aby + cbz * abz
            if k < 1e-9:
                k = 1e-9
            if tag == "n":
                c.kn = k
                c.jnA = (dx, dy, dz, cax, cay, caz)
                c.jnB = (dx, dy, dz, cbx, cby, cbz)
                c.pnA = (dx * imA, dy * imA, dz * imA, aax, aay, aaz)
                c.pnB = (dx * imB, dy * imB, dz * imB, abx, aby, abz)
                c.vn0 = (dx * (vax - vbx) + dy * (vay - vby) + dz * (vaz - vbz)
                         + cax * wax + cay * way + caz * waz
                         - cbx * wbx - cby * wby - cbz * wbz)
            elif tag == "t1":
                c.kt1 = k
                c.jt1A = (dx, dy, dz, cax, cay, caz)
                c.jt1B = (dx, dy, dz, cbx, cby, cbz)
                c.pt1A = (dx * imA, dy * imA, dz * imA, aax, aay, aaz)
                c.pt1B = (dx * imB, dy * imB, dz * imB, abx, aby, abz)
            else:
                c.kt2 = k
                c.jt2A = (dx, dy, dz, cax, cay, caz)
                c.jt2B = (dx, dy, dz, cbx, cby, cbz)
                c.pt2A = (dx * imA, dy * imA, dz * imA, aax, aay, aaz)
                c.pt2B = (dx * imB, dy * imB, dz * imB, abx, aby, abz)

        c.ln = c.lt1 = c.lt2 = 0.0
        c.fx = c.fy = c.fz = 0.0
        c.active = (c.sd <= 1e-9) or (c.sd + c.vn0 * DT <= 1e-9)

        bp = (BETA / DT) * max(0.0, -c.sd - SLOP) if c.sd < -SLOP else -1e18
        br = -self.e * c.vn0 if (c.vn0 < -REST_THRESH and c.sd < 0.001) else -1e18
        spec = -c.sd / DT
        target = max(bp, br, spec)
        c.target = MAX_BIAS if target > MAX_BIAS else target
        return c

    # -- collision detection -----------------------------------------------
    def _pair_contacts(self, i, j, Wi, Wj, contacts):
        m = DETECT_MARGIN
        Rj, xcj = self.R[j], self.xc[j]
        vi, sd, q, nrm = _query((Wi - xcj) @ Rj, self.proxy, m)
        for k in range(vi.size):
            if sd[k] < m:
                qw = q[k] @ Rj.T + xcj
                nw = nrm[k] @ Rj.T
                p = 0.5 * (Wi[vi[k]] + qw)
                cc = self._mk(i, j, p, nw, float(sd[k]))
                if cc is not None:
                    contacts.append(cc)
        Ri, xci = self.R[i], self.xc[i]
        vi, sd, q, nrm = _query((Wj - xci) @ Ri, self.proxy, m)
        for k in range(vi.size):
            if sd[k] < m:
                qw = q[k] @ Ri.T + xci
                nw = -(nrm[k] @ Ri.T)
                p = 0.5 * (Wj[vi[k]] + qw)
                cc = self._mk(i, j, p, nw, float(sd[k]))
                if cc is not None:
                    contacts.append(cc)

    def _detect(self):
        N, P = self.N, self.pcount
        W = np.empty((N, P, 3))
        lo = np.empty((N, 3))
        hi = np.empty((N, 3))
        self.Iwt = []
        for i in range(N):
            R = self.R[i]
            W[i] = self.pvert @ R.T + self.xc[i]
            ext = np.abs(R) @ self.phalf
            lo[i] = self.xc[i] - ext
            hi[i] = self.xc[i] + ext
            Iw = R @ self.Ibinv[i] @ R.T
            self.Iwt.append((Iw[0, 0], Iw[0, 1], Iw[0, 2],
                             Iw[1, 0], Iw[1, 1], Iw[1, 2],
                             Iw[2, 0], Iw[2, 1], Iw[2, 2]))
        m = DETECT_MARGIN
        contacts = []
        for i in range(N):
            for j in range(i + 1, N):
                if bool(np.any(lo[i] > hi[j])) or bool(np.any(lo[j] > hi[i])):
                    continue
                self._pair_contacts(i, j, W[i], W[j], contacts)
        fverts = self.funnel.verts
        for i in range(N):
            vi, sd, q, nrm = _query(W[i], self.funnel, m)
            for k in range(vi.size):
                if sd[k] < m:
                    p = 0.5 * (W[i][vi[k]] + q[k])
                    cc = self._mk(i, -1, p, nrm[k], float(sd[k]))
                    if cc is not None:
                        contacts.append(cc)
            Ri, xci = self.R[i], self.xc[i]
            vi, sd, q, nrm = _query((fverts - xci) @ Ri, self.proxy, m)
            for k in range(vi.size):
                if sd[k] < m:
                    qw = q[k] @ Ri.T + xci
                    nw = -(nrm[k] @ Ri.T)
                    p = 0.5 * (fverts[vi[k]] + qw)
                    cc = self._mk(i, -1, p, nw, float(sd[k]))
                    if cc is not None:
                        contacts.append(cc)
            fsd, fnrm = self.floor.sd(W[i])
            for k in np.nonzero(fsd < m)[0]:
                cc = self._mk(i, -1, W[i][k], fnrm[k], float(fsd[k]))
                if cc is not None:
                    contacts.append(cc)
        return contacts

    # -- sequential impulses -------------------------------------------------
    def _solve(self, contacts):
        v, w, mu = self.vl, self.wl, self.mu
        for _ in range(SI_ITERS):
            for c in contacts:
                if not c.active:
                    continue
                ia, ib = c.ia, c.ib
                va, wa = v[ia], w[ia]
                vb = wb = None
                if ib >= 0:
                    vb, wb = v[ib], w[ib]
                # normal
                jx, jy, jz, ax, ay, az = c.jnA
                vn = (jx * va[0] + jy * va[1] + jz * va[2]
                      + ax * wa[0] + ay * wa[1] + az * wa[2])
                if vb is not None:
                    jx, jy, jz, ax, ay, az = c.jnB
                    vn -= (jx * vb[0] + jy * vb[1] + jz * vb[2]
                           + ax * wb[0] + ay * wb[1] + az * wb[2])
                ln = c.ln + (c.target - vn) * c.kn
                if ln < 0.0:
                    ln = 0.0
                dl = ln - c.ln
                c.ln = ln
                if dl != 0.0:
                    ix, iy, iz, rx, ry, rz = c.pnA
                    va[0] += dl * ix
                    va[1] += dl * iy
                    va[2] += dl * iz
                    wa[0] += dl * rx
                    wa[1] += dl * ry
                    wa[2] += dl * rz
                    if vb is not None:
                        ix, iy, iz, rx, ry, rz = c.pnB
                        vb[0] -= dl * ix
                        vb[1] -= dl * iy
                        vb[2] -= dl * iz
                        wb[0] -= dl * rx
                        wb[1] -= dl * ry
                        wb[2] -= dl * rz
                # friction
                if c.ln <= 0.0:
                    continue
                jx, jy, jz, ax, ay, az = c.jt1A
                vt1 = (jx * va[0] + jy * va[1] + jz * va[2]
                       + ax * wa[0] + ay * wa[1] + az * wa[2])
                jx, jy, jz, ax, ay, az = c.jt2A
                vt2 = (jx * va[0] + jy * va[1] + jz * va[2]
                       + ax * wa[0] + ay * wa[1] + az * wa[2])
                if vb is not None:
                    jx, jy, jz, ax, ay, az = c.jt1B
                    vt1 -= (jx * vb[0] + jy * vb[1] + jz * vb[2]
                            + ax * wb[0] + ay * wb[1] + az * wb[2])
                    jx, jy, jz, ax, ay, az = c.jt2B
                    vt2 -= (jx * vb[0] + jy * vb[1] + jz * vb[2]
                            + ax * wb[0] + ay * wb[1] + az * wb[2])
                lt1 = c.lt1 - vt1 * c.kt1
                lt2 = c.lt2 - vt2 * c.kt2
                fmax = mu * c.ln
                fm2 = lt1 * lt1 + lt2 * lt2
                if fm2 > fmax * fmax and fm2 > 0.0:
                    sc = fmax / math.sqrt(fm2)
                    lt1 *= sc
                    lt2 *= sc
                d1 = lt1 - c.lt1
                d2 = lt2 - c.lt2
                c.lt1, c.lt2 = lt1, lt2
                if d1 != 0.0:
                    ix, iy, iz, rx, ry, rz = c.pt1A
                    va[0] += d1 * ix
                    va[1] += d1 * iy
                    va[2] += d1 * iz
                    wa[0] += d1 * rx
                    wa[1] += d1 * ry
                    wa[2] += d1 * rz
                    if vb is not None:
                        ix, iy, iz, rx, ry, rz = c.pt1B
                        vb[0] -= d1 * ix
                        vb[1] -= d1 * iy
                        vb[2] -= d1 * iz
                        wb[0] -= d1 * rx
                        wb[1] -= d1 * ry
                        wb[2] -= d1 * rz
                if d2 != 0.0:
                    ix, iy, iz, rx, ry, rz = c.pt2A
                    va[0] += d2 * ix
                    va[1] += d2 * iy
                    va[2] += d2 * iz
                    wa[0] += d2 * rx
                    wa[1] += d2 * ry
                    wa[2] += d2 * rz
                    if vb is not None:
                        ix, iy, iz, rx, ry, rz = c.pt2B
                        vb[0] -= d2 * ix
                        vb[1] -= d2 * iy
                        vb[2] -= d2 * iz
                        wb[0] -= d2 * rx
                        wb[1] -= d2 * ry
                        wb[2] -= d2 * rz

    # -- integration ---------------------------------------------------------
    def _integrate(self):
        half_dt = 0.5 * DT
        for i in range(self.N):
            self.xc[i] = self.xc[i] + self.v[i] * DT
            q = self.q[i]
            wx, wy, wz = self.w[i]
            dq = _quat_mul(np.array([wx, wy, wz, 0.0]), q) * half_dt
            q = q + dq
            nq = math.sqrt(float(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2))
            if nq < 1e-12:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                q = q / nq
            self.q[i] = q
            R = _quat_to_mat(q)
            self.R[i] = R
            self.xo[i] = self.xc[i] - R @ self.com

    # -- substep ---------------------------------------------------------------
    def _substep(self):
        self.v = self.v + self.gravity[None, :] * DT
        self.w = self.w * (1.0 - ANG_DAMP * DT)
        self.vl = self.v.tolist()
        self.wl = self.w.tolist()
        self.xcl = self.xc.tolist()
        contacts = self._detect()
        if contacts:
            self._solve(contacts)
            self.v = np.asarray(self.vl, float)
            self.w = np.asarray(self.wl, float)
        np.clip(self.v, -MAX_V, MAX_V, out=self.v)
        np.clip(self.w, -MAX_W, MAX_W, out=self.w)
        if not (np.isfinite(self.v).all() and np.isfinite(self.w).all()):
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        self._integrate()
        if not np.isfinite(self.xc).all():
            self.xc = np.nan_to_num(self.xc, nan=0.0, posinf=0.0, neginf=0.0)
        inv_dt = 1.0 / DT
        for c in contacts:
            if c.active:
                c.fx = (c.ln * c.nx + c.lt1 * c.t1x + c.lt2 * c.t2x) * inv_dt
                c.fy = (c.ln * c.ny + c.lt1 * c.t1y + c.lt2 * c.t2y) * inv_dt
                c.fz = (c.ln * c.nz + c.lt1 * c.t1z + c.lt2 * c.t2z) * inv_dt
        self.last_contacts = [c for c in contacts if c.sd < EXPORT_MARGIN]
        self.step += 1

    # -- public API ------------------------------------------------------------
    def advance_to(self, time):
        t = float(time)
        target = int(round(t / DT))
        if target < 0:
            target = 0
        while self.step < target:
            self._substep()
        return self._frame(t)

    def _frame(self, t):
        last = self.last_contacts
        csd = np.asarray([c.sd for c in last], dtype=float)
        inf = np.asarray(
            [[c.fx, c.fy, c.fz] for c in last], dtype=float
        ).reshape(-1, 3)
        pair = np.asarray(
            [[float(c.ia), float(c.ib)] for c in last], dtype=float
        ).reshape(-1, 2)
        ext = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            float(t),
            np.array(self.xo, dtype=float),
            np.array(self.v, dtype=float),
            orientation=np.array(self.q, dtype=float),
            angular_velocity=np.array(self.w, dtype=float),
            mass=np.array(self.mass, dtype=float),
            body_id=np.array(self.ids, dtype=np.int32),
            fields={
                "contact_signed_distance": csd,
                "external_force": ext,
                "interaction_force": inf,
                "interaction_body_ids": pair,
            },
            scalars={"contact_length_scale": self.length_scale},
        )

    def diagnostics(self):
        return {
            "substeps": int(self.step),
            "substep_dt": DT,
            "bodies": int(self.N),
            "exported_contacts": int(len(self.last_contacts)),
            "friction": self.mu,
            "restitution": self.e,
        }
