"""Physically-based rigid-body solver for the chess-piece pile settling scene.

Implementation (no external physics engine):

* Each nonconvex chess piece is a rigid body whose mesh vertices double as
  contact sample points.  The *other* body in a pairwise query is represented
  by a cylindrical radial-height collision proxy (a discretised surface of
  revolution R(z, theta) built from the actual mesh vertices), which gives an
  analytic-ish signed distance and outward normal for arbitrary query points,
  including exact flat top/bottom caps.
* Contacts are detected symmetrically (vertices of A vs proxy of B and vice
  versa) plus exact vertex-vs-plane contacts against the static floor.
* Contact response uses a penalty (spring-damper) normal force with Coulomb
  tangential springs (classic DEM stick-slip friction with accumulated
  tangential displacement and force projection at the friction cone).
* Time integration is semi-implicit (symplectic) Euler on both translational
  and rotational degrees of freedom with quaternion orientation updates.

Everything is driven only by the frozen configuration, geometry assets and
gravity; no keyframes or scripted motion are used.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np

try:  # platform contract
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - minimal fallback
    from dataclasses import dataclass, field as _dc_field

    @dataclass
    class StateFrame:  # type: ignore
        time: float
        position: object
        velocity: object
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: dict = _dc_field(default_factory=dict)
        fields: dict = _dc_field(default_factory=dict)
        scalars: dict = _dc_field(default_factory=dict)

        def validate(self) -> None:
            pass


TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# small quaternion / rotation helpers (xyzw convention)
# --------------------------------------------------------------------------
def _quat_to_mat(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
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


def _quat_normalize(q):
    n = math.sqrt(float(np.dot(q, q)))
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


# --------------------------------------------------------------------------
# OBJ loading and mesh mass properties
# --------------------------------------------------------------------------
def _load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, dtype=float)
    F = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    if len(F) == 0:
        F = np.zeros((0, 3), dtype=np.int64)
    return V, F


def _mesh_mass_properties(V, F):
    """Volume, centre of mass and unit-density inertia tensor of a closed mesh.

    Uses the divergence-theorem tetrahedron integration (reference tetra
    moments, affine mapped per face).  Returns None if the mesh is unusable.
    """
    if len(F) < 4 or len(V) < 4:
        return None
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))  # 6 * signed tet volume
    vol = float(det.sum()) / 6.0
    if not np.isfinite(vol) or vol <= 1e-9:
        return None
    M1 = (det[:, None] * (v0 + v1 + v2)).sum(axis=0) / 24.0
    com = M1 / vol
    Mref = (np.ones((3, 3)) + np.eye(3)) / 120.0
    A = np.stack([v0, v1, v2], axis=1)          # columns are tet edge vectors
    AM = A @ Mref
    AMA = np.einsum("mij,mjk->mik", AM, A.transpose(0, 2, 1))
    S = np.einsum("m,mij->ij", det, AMA)        # \int x x^T dV
    cov = S / vol - np.outer(com, com)
    inertia = (float(np.trace(cov)) * np.eye(3) - cov)
    if not (np.isfinite(inertia).all() and np.isfinite(com).all()):
        return None
    # a few inconsistently wound triangles can give tiny negative moments;
    # clamp to a positive-definite tensor instead of rejecting the mesh
    evals, evecs = np.linalg.eigh(inertia)
    if evals[-1] <= 0.0:
        return None
    floor = max(float(evals[-1]) * 1e-3, 1e-9)
    evals = np.maximum(evals, floor)
    inertia = evecs @ np.diag(evals) @ evecs.T
    inertia = 0.5 * (inertia + inertia.T)
    return vol, com, inertia


def _fallback_mass_properties(V):
    """Point-mass fallback (mesh not closed): vertices weighted uniformly."""
    com = V.mean(axis=0)
    d = V - com
    m_pt = 1.0 / len(V)
    cov = m_pt * (d.T @ d)
    inertia = float(np.trace(cov)) * np.eye(3) - cov
    vol = max(float((V.max(axis=0) - V.min(axis=0)).prod()) * 0.25, 1e-4)
    return vol, com, inertia


# --------------------------------------------------------------------------
# cylindrical radial-profile collision proxy (approximate SDF)
# --------------------------------------------------------------------------
class _RadialProfile:
    """Discretised R(z, theta) solid-of-revolution collision proxy.

    Built from the mesh vertices: per theta bin the radius is linearly
    interpolated in z between vertex rings (matching the ruled lathe
    surface), empty bins are filled from angular neighbours, and a small
    radial pad keeps the proxy conservative between vertex samples.
    """

    def __init__(self, verts, target_dz=0.01, ntheta=32, pad=0.003):
        z = verts[:, 2]
        self.z0 = float(z.min())
        self.z1 = float(z.max())
        if self.z1 - self.z0 < 1e-6:
            self.z1 = self.z0 + 1e-6
        span = self.z1 - self.z0
        self.nz = int(np.clip(int(math.ceil(span / target_dz)), 24, 260))
        self.nth = int(ntheta)
        self.dz = span / self.nz
        self.dth = TWO_PI / self.nth
        zc = self.z0 + (np.arange(self.nz) + 0.5) * self.dz
        zlo = self.z0 + np.arange(self.nz) * self.dz
        zhi = zlo + self.dz

        x = verts[:, 0]
        y = verts[:, 1]
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        ith = (np.floor((th + TWO_PI) / self.dth).astype(np.int64)) % self.nth

        grid = np.full((self.nz, self.nth), np.nan)
        for j in range(self.nth):
            m = ith == j
            if not np.any(m):
                continue
            zs = z[m]
            rs = r[m]
            order = np.argsort(zs)
            zs = zs[order]
            rs = rs[order]
            # radius envelope of the ruled profile across each row's full
            # z interval: never under-covers sharp steps or protrusions
            col = np.maximum.reduce([
                np.interp(zc, zs, rs),
                np.interp(zlo, zs, rs),
                np.interp(zhi, zs, rs),
            ])
            iz_v = np.clip(np.floor((zs - self.z0) / self.dz).astype(np.int64),
                           0, self.nz - 1)
            np.maximum.at(col, iz_v, rs)
            grid[:, j] = col

        # fill remaining NaN bins from angular neighbours (same z row)
        if np.isnan(grid).any():
            row_max = np.nanmax(np.where(np.isnan(grid), -np.inf, grid), axis=1)
            global_max = float(np.nanmax(grid))
            for i in range(self.nz):
                row = grid[i]
                if np.all(np.isnan(row)):
                    grid[i] = global_max
                    continue
                if np.any(np.isnan(row)):
                    rm = row_max[i]
                    if not np.isfinite(rm):
                        rm = global_max
                    row[np.isnan(row)] = rm

        grid = np.maximum(grid, 0.0) + pad
        self.grid = grid
        self.slope = np.gradient(grid, self.dz, axis=0)  # dR/dz
        self.rmax = float(grid.max())
        self.zmin = self.z0
        self.zmax = self.z1

    def evaluate(self, pts):
        """Signed distance (max-of-constraints form) and outward normals.

        pts are expressed in the proxy's local frame.  sd < 0 inside,
        sd > 0 outside; the value is a lower bound of the true distance
        outside, which is conservative for near-contact detection.
        """
        pts = np.asarray(pts, dtype=float)
        m = len(pts)
        if m == 0:
            return np.zeros(0), np.zeros((0, 3))
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        ith = (np.floor((th + TWO_PI) / self.dth).astype(np.int64)) % self.nth
        iz = np.clip(np.floor((z - self.z0) / self.dz).astype(np.int64), 0, self.nz - 1)

        R = self.grid[iz, ith]
        sd_lat = r - R
        sd_top = z - self.z1
        sd_bot = self.z0 - z
        sd = np.maximum(sd_lat, np.maximum(sd_top, sd_bot))

        is_top = (sd_top >= sd_lat) & (sd_top >= sd_bot)
        is_bot = (~is_top) & (sd_bot >= sd_lat)
        lat = ~(is_top | is_bot)

        nrm = np.zeros((m, 3))
        if np.any(is_top):
            nrm[is_top, 2] = 1.0
        if np.any(is_bot):
            nrm[is_bot, 2] = -1.0
        if np.any(lat):
            xl = x[lat]
            yl = y[lat]
            rl = r[lat]
            safe = rl > 1e-9
            vx = np.zeros(len(rl))
            vy = np.zeros(len(rl))
            vx[safe] = xl[safe] / rl[safe]
            vy[safe] = yl[safe] / rl[safe]
            vz = -self.slope[iz[lat], ith[lat]]
            vec = np.stack([vx, vy, vz], axis=1)
            nlen = np.linalg.norm(vec, axis=1)
            bad = nlen < 1e-9
            if np.any(bad):
                vec[bad] = np.array([0.0, 0.0, 1.0])
                nlen[bad] = 1.0
            nrm[lat] = vec / nlen[:, None]
        return sd, nrm


# --------------------------------------------------------------------------
# frozen workspace inputs
# --------------------------------------------------------------------------
def _workspace_root():
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parents[1])
    except Exception:
        pass
    candidates.append(Path.cwd())
    for c in candidates:
        if (c / "scene_geometry.yaml").exists() and (c / "assets").is_dir():
            return c
    for c in candidates:
        if (c / "assets").is_dir():
            return c
    return candidates[0]


def _parse_scene_geometry(root):
    """Return (asset_by_object_id, floor_top_z) from the frozen geometry file."""
    asset_map = {}
    floor_top = 0.0
    path = root / "scene_geometry.yaml"
    if not path.exists():
        return asset_map, floor_top
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        objects = data.get("objects", []) if isinstance(data, dict) else []
        for obj in objects:
            oid = obj.get("id")
            geom = obj.get("geometry") or {}
            if geom.get("type") == "triangle_mesh" and geom.get("asset"):
                asset_map[oid] = str(geom["asset"])
            elif geom.get("type") == "box" and obj.get("semantic") == "boundary":
                tr = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
                he = geom.get("half_extents") or [1.0, 1.0, 0.05]
                floor_top = float(tr[2]) + float(he[2])
    except Exception:
        # regex fallback: object blocks in order
        blocks = re.split(r"\n(?=- id:)", text)
        for block in blocks:
            mid = re.search(r"- id:\s*(\S+)", block)
            if not mid:
                continue
            oid = mid.group(1)
            ma = re.search(r"asset:\s*(\S+\.obj)", block)
            if ma:
                asset_map[oid] = ma.group(1)
            mt = re.search(r"translation:\s*\[\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\s*\]", block)
            mh = re.search(r"half_extents:\s*\[\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\s*\]", block)
            if "type: box" in block and mt and mh and "semantic: boundary" in block:
                floor_top = float(mt.group(3)) + float(mh.group(3))
    return asset_map, floor_top


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """Rigid-body penalty-contact solver with DEM Coulomb friction."""

    DT = 2.5e-4            # physics substep (s)
    K_PAIR = 6.0e4         # normal stiffness budget per body pair (N/m)
    ZETA = 0.55            # normal contact damping ratio (~restitution 0.1)
    PEN_MAX = 0.012        # clamp on treated penetration depth (m)
    F_MAX = 150.0          # cap on a single contact normal force (N)
    MARGIN = 0.02          # near-contact detection margin (m)
    LIN_DAMP = 0.05        # viscous linear damping (1/s)
    ANG_DAMP = 0.25        # viscous angular damping (1/s)

    def __init__(self):
        self._initialized = False
        self._t = 0.0
        self._diag = {"steps": 0, "max_contacts": 0, "peak_speed": 0.0}

    # ---------------- initialization ----------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self._cfg = cfg
        self._seed = int(seed)
        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact = cfg.get("contact", {})
        self._mu = float(contact.get("friction", 0.5))
        self._restitution = float(contact.get("restitution", 0.1))
        self._length_scale = float(contact.get("length_scale", 0.02))

        root = _workspace_root()
        asset_map, floor_top = _parse_scene_geometry(root)

        init = scene.initial_state()
        bodies = cfg.get("bodies", [])
        n = len(bodies)
        self._n = n
        self._mass = np.asarray([float(b.get("mass", 1.0)) for b in bodies], dtype=float)
        self._body_ids = np.arange(n, dtype=np.int32)
        self._release = np.asarray(init.attributes.get(
            "release_time", np.zeros(n)), dtype=float)

        # per-body geometry
        self._profiles = [None] * n
        self._plocal = [None] * n       # contact sample points, COM frame
        self._com_local = [None] * n    # mesh COM in mesh-local coords
        self._inertia_local = [None] * n
        self._inertia_inv = [None] * n
        cache = {}
        default_cycle = ["pawn", "rook", "knight", "bishop", "queen", "king"]
        for i, b in enumerate(bodies):
            oid = b.get("id", "piece_%d" % i)
            asset = asset_map.get(oid)
            if asset is None:
                asset = "assets/%s.obj" % default_cycle[i % len(default_cycle)]
            mpath = root / asset
            if not mpath.exists():
                mpath = root / "assets" / (Path(asset).name)
            if str(mpath) not in cache:
                V, F = _load_obj(mpath)
                props = _mesh_mass_properties(V, F)
                if props is None:
                    props = _fallback_mass_properties(V)
                cache[str(mpath)] = (V, props)
            V, (vol, com, inertia_unit) = cache[str(mpath)]
            density = self._mass[i] / vol
            self._com_local[i] = com
            self._inertia_local[i] = density * inertia_unit
            I = self._inertia_local[i]
            # regularise slightly so the inverse always exists
            I = I + np.eye(3) * (1e-6 * max(float(np.trace(I)), 1e-9))
            self._inertia_inv[i] = np.linalg.inv(I)
            # proxy and sample points share the COM-centred local frame
            Vc = V - com
            self._profiles[i] = _RadialProfile(Vc)
            self._plocal[i] = Vc  # contact points in the COM frame

        # floor
        self._floor_z = float(floor_top)

        # dynamic state (reference point = centre of mass)
        pos0 = np.asarray(init.position, dtype=float)      # mesh-origin frame
        q0 = np.asarray(init.orientation if init.orientation is not None
                        else np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), dtype=float)
        self._q = np.array([_quat_normalize(qq) for qq in q0])
        self._x = np.zeros((n, 3))
        for i in range(n):
            R = _quat_to_mat(self._q[i])
            self._x[i] = pos0[i] + R @ self._com_local[i]
        self._v = np.zeros((n, 3))
        self._w = np.zeros((n, 3))

        # friction state: per ordered pair (a, b) an [n_pts_a, 3] array of
        # accumulated tangential displacements (world frame).
        self._fric = {}
        self._active_pairs = set()

        # static pseudo-body (floor) appended at index n for vectorisation
        self._t_current = 0.0
        self._initialized = True

    # ---------------- contact handling ----------------
    def _world_points(self):
        """World-space contact sample points and per-body AABBs."""
        n = self._n
        PW = [None] * n
        lo = np.zeros((n, 3))
        hi = np.zeros((n, 3))
        margin = self.MARGIN + 1e-3
        for i in range(n):
            R = _quat_to_mat(self._q[i])
            pts = self._x[i] + self._plocal[i] @ R.T
            PW[i] = pts
            lo[i] = pts.min(axis=0) - margin
            hi[i] = pts.max(axis=0) + margin
        return PW, lo, hi

    def _gather_contacts(self, PW, lo, hi):
        """Detect all contacts (active + near).

        Returns arrays (A, B, IA, P, N, SD) where A owns the sample point,
        B is the proxy body (n = floor pseudo-index).
        """
        n = self._n
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if (lo[i, 0] <= hi[j, 0] and lo[j, 0] <= hi[i, 0] and
                        lo[i, 1] <= hi[j, 1] and lo[j, 1] <= hi[i, 1] and
                        lo[i, 2] <= hi[j, 2] and lo[j, 2] <= hi[i, 2]):
                    pairs.append((i, j))

        A_list, B_list, IA_list, P_list, N_list, SD_list = [], [], [], [], [], []

        for (i, j) in pairs:
            # vertices of i against proxy of j
            Rj = _quat_to_mat(self._q[j])
            pl = (PW[i] - self._x[j]) @ Rj
            sd, nl = self._profiles[j].evaluate(pl)
            mask = sd < self.MARGIN
            if np.any(mask):
                idx = np.nonzero(mask)[0]
                nw = nl[idx] @ Rj.T
                A_list.append(np.full(len(idx), i, dtype=np.int64))
                B_list.append(np.full(len(idx), j, dtype=np.int64))
                IA_list.append(idx)
                P_list.append(PW[i][idx])
                N_list.append(nw)
                SD_list.append(sd[idx])
            # vertices of j against proxy of i
            Ri = _quat_to_mat(self._q[i])
            pl = (PW[j] - self._x[i]) @ Ri
            sd, nl = self._profiles[i].evaluate(pl)
            mask = sd < self.MARGIN
            if np.any(mask):
                idx = np.nonzero(mask)[0]
                nw = nl[idx] @ Ri.T
                A_list.append(np.full(len(idx), j, dtype=np.int64))
                B_list.append(np.full(len(idx), i, dtype=np.int64))
                IA_list.append(idx)
                P_list.append(PW[j][idx])
                N_list.append(nw)
                SD_list.append(sd[idx])

        # floor contacts (exact vertex vs plane)
        up = np.array([0.0, 0.0, 1.0])
        for i in range(n):
            z = PW[i][:, 2]
            mask = z - self._floor_z < self.MARGIN
            if np.any(mask):
                idx = np.nonzero(mask)[0]
                k = len(idx)
                A_list.append(np.full(k, i, dtype=np.int64))
                B_list.append(np.full(k, n, dtype=np.int64))
                IA_list.append(idx)
                P_list.append(PW[i][idx])
                N_list.append(np.tile(up, (k, 1)))
                SD_list.append(z[idx] - self._floor_z)

        if not A_list:
            z0 = np.zeros(0, dtype=np.int64)
            return (z0, z0, z0, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        return (np.concatenate(A_list), np.concatenate(B_list),
                np.concatenate(IA_list), np.concatenate(P_list, axis=0),
                np.concatenate(N_list, axis=0), np.concatenate(SD_list))

    def _contact_forces(self, contacts, h, mutate):
        """Penalty normal + DEM Coulomb friction forces for all contacts.

        Returns (F, T, sd_export, inter_export) with F/T per body and the
        raw trajectory exports (signed gaps for every contact and the
        interaction force vector for every body-body contact).
        """
        A, B, IA, P, N, SD = contacts
        n = self._n
        m = len(A)
        F = np.zeros((n, 3))
        T = np.zeros((n, 3))
        if m == 0:
            if mutate:
                for pk in self._active_pairs:
                    arr = self._fric.get(pk)
                    if arr is not None:
                        arr[:] = 0.0
                self._active_pairs = set()
            return F, T, np.zeros(0), np.zeros((0, 3))

        mass = self._mass
        mext = np.append(mass, 1.0)
        vext = np.vstack([self._v, np.zeros((1, 3))])
        wext = np.vstack([self._w, np.zeros((1, 3))])
        xext = np.vstack([self._x, np.zeros((1, 3))])

        # per ordered-pair contact counts -> shared stiffness/damping budget
        key = A * (n + 1) + B
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        k_ct = self.K_PAIR / cnt[inv]
        m_eff = np.where(B < n,
                         mext[A] * mext[B] / np.maximum(mext[A] + mext[B], 1e-9),
                         mext[A])
        cn_tot = 2.0 * self.ZETA * np.sqrt(self.K_PAIR * m_eff)
        cn_ct = cn_tot / cnt[inv]
        ct_ct = 1.6 * cn_ct

        # kinematics at contact points
        ra = P - xext[A]
        rb = P - xext[B]
        va = vext[A] + np.cross(wext[A], ra)
        vb = vext[B] + np.cross(wext[B], rb)
        vrel = va - vb
        vn = np.einsum("ij,ij->i", vrel, N)
        pen = np.clip(-SD, 0.0, self.PEN_MAX)
        # normal damping only while actually touching (sd < 0), so pure
        # near-contact samples produce zero force instead of a cushion layer
        Fn = np.clip(k_ct * pen - cn_ct * vn * (SD < 0.0), 0.0, self.F_MAX)

        vt = vrel - vn[:, None] * N

        # tangential displacement state per ordered body pair
        u_arr = np.zeros((m, 3))
        pair_arrays = []
        for kk in range(len(uk)):
            a = int(uk[kk] // (n + 1))
            b = int(uk[kk] % (n + 1))
            pk = (a, b)
            arr = self._fric.get(pk)
            if arr is None:
                arr = np.zeros((len(self._plocal[a]), 3))
                self._fric[pk] = arr
            g = inv == kk
            u_arr[g] = arr[IA[g]]
            pair_arrays.append((pk, arr, g))

        kt = k_ct
        Ft = -(kt[:, None] * u_arr + ct_ct[:, None] * vt)
        fmag = np.linalg.norm(Ft, axis=1)
        cap = self._mu * Fn
        slip = fmag > cap
        scale = np.where(slip, cap / np.maximum(fmag, 1e-12), 1.0)
        Ft = Ft * scale[:, None]

        Fvec = Fn[:, None] * N + Ft

        # accumulate forces / torques
        np.add.at(F, A, Fvec)
        np.add.at(T, A, np.cross(ra, Fvec))
        dyn = B < n
        if np.any(dyn):
            np.add.at(F, B[dyn], -Fvec[dyn])
            np.add.at(T, B[dyn], -np.cross(rb[dyn], Fvec[dyn]))

        # update friction state (stick: accumulate; slip: project)
        if mutate and h > 0.0:
            current = set()
            for pk, arr, g in pair_arrays:
                idx = IA[g]
                newu = np.where(slip[g][:, None],
                                -Ft[g] / kt[g][:, None],
                                u_arr[g] + vt[g] * h)
                arr[idx] = newu
                touched = np.zeros(len(arr), dtype=bool)
                touched[idx] = True
                arr[~touched] = 0.0
                current.add(pk)
            for pk in self._active_pairs:
                if pk not in current:
                    arr = self._fric.get(pk)
                    if arr is not None:
                        arr[:] = 0.0
            self._active_pairs = current

        inter = Fvec[dyn] if np.any(dyn) else np.zeros((0, 3))
        return F, T, SD, inter

    # ---------------- integration ----------------
    def _step(self, h):
        n = self._n
        PW, lo, hi = self._world_points()
        contacts = self._gather_contacts(PW, lo, hi)
        F, T, _, _ = self._contact_forces(contacts, h, mutate=True)
        lin = max(0.0, 1.0 - self.LIN_DAMP * h)
        ang = max(0.0, 1.0 - self.ANG_DAMP * h)
        for i in range(n):
            R = _quat_to_mat(self._q[i])
            Iinv_w = R @ self._inertia_inv[i] @ R.T
            self._v[i] = (self._v[i] + h * (F[i] / self._mass[i] + self._gravity)) * lin
            self._w[i] = (self._w[i] + h * (Iinv_w @ T[i])) * ang
            self._x[i] = self._x[i] + h * self._v[i]
            wq = np.array([self._w[i][0], self._w[i][1], self._w[i][2], 0.0])
            dq = _quat_mul(wq, self._q[i])
            self._q[i] = _quat_normalize(self._q[i] + 0.5 * h * dq)
        self._t_current += h
        self._diag["steps"] += 1
        self._diag["max_contacts"] = max(self._diag["max_contacts"], len(contacts[0]))
        speed = float(np.linalg.norm(self._v, axis=1).max())
        self._diag["peak_speed"] = max(self._diag["peak_speed"], speed)

    # ---------------- frame export ----------------
    def _make_frame(self, time):
        n = self._n
        PW, lo, hi = self._world_points()
        contacts = self._gather_contacts(PW, lo, hi)
        _, _, sd, inter = self._contact_forces(contacts, 0.0, mutate=False)

        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        for i in range(n):
            R = _quat_to_mat(self._q[i])
            off = R @ self._com_local[i]        # mesh origin seen from COM
            pos[i] = self._x[i] - off
            vel[i] = self._v[i] + np.cross(self._w[i], -off)

        ext = self._mass[:, None] * self._gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=np.array([_quat_normalize(q) for q in self._q]),
            angular_velocity=self._w.copy(),
            mass=self._mass.copy(),
            body_id=self._body_ids.copy(),
            attributes={"release_time": self._release.copy()},
            fields={
                "contact_signed_distance": np.asarray(sd, dtype=float),
                "external_force": ext,
                "interaction_force": np.asarray(inter, dtype=float).reshape(-1, 3),
            },
            scalars={"contact_length_scale": float(self._length_scale)},
        )
        frame.validate()
        return frame

    # ---------------- public API ----------------
    def advance_to(self, time):
        if not self._initialized:
            raise RuntimeError("solver not initialized")
        target = max(0.0, float(time))
        guard = 0
        max_steps = int(round(float(self._cfg.get("duration", 8.0)) / self.DT)) + 64
        while self._t_current < target - 1e-9 and guard < 4 * max_steps:
            h = min(self.DT, target - self._t_current)
            if h < 1e-12:
                break
            self._step(h)
            guard += 1
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        n = self._n
        if not self._initialized:
            return dict(self._diag)
        ke = 0.0
        for i in range(n):
            ke += 0.5 * self._mass[i] * float(np.dot(self._v[i], self._v[i]))
            R = _quat_to_mat(self._q[i])
            Iw = R @ self._inertia_local[i] @ R.T
            ke += 0.5 * float(self._w[i] @ (Iw @ self._w[i]))
        out = dict(self._diag)
        out["time"] = float(self._t_current)
        out["kinetic_energy"] = float(ke)
        out["max_speed"] = float(np.linalg.norm(self._v, axis=1).max()) if n else 0.0
        out["bodies"] = int(n)
        return out
