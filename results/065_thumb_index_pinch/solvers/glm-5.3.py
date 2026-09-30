"""Volumetric FEM solver for the thumb-index web pinch scene.

The deformable flesh is simulated as a 3-D corotated (Neo-Hookean-type)
elastic solid on a tetrahedral grid voxelized from the frozen rest surface
mesh.  Surface vertices are embedded in the tetrahedra with barycentric
weights, so the rendered surface follows the volumetric deformation.  The
thumb bone is a kinematic capsule driven by the frozen rotation/translation
schedule; flesh nodes immediately surrounding a bone are rigidly attached to
it, the forearm cuff region is fixed, and the remaining flesh obeys
symplectic-Euler elastodynamics with penalty-free projection contacts
against the bone capsules and short-range node-node non-penetration
constraints.  All external impulses (bone drive reactions, clamp reactions,
contacts, drag) are accumulated so the exported budget force is the true
net external force on the deformable body.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _smoothstep(t: float, t0: float, t1: float) -> float:
    u = (t - t0) / (t1 - t0)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _load_obj(path: Path):
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, float), np.asarray(faces, np.int64)


# Half-neighborhood cell offsets (lexicographically positive) for pair search.
_HALF_OFFSETS = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if (di, dj, dk) != (0, 0, 0)
    and (di, dj, dk) > (0, 0, 0)
]


class BenchmarkSolver:
    """Corotated-Elastic FEM flesh solver with driven bone capsules."""

    VOXEL_H = 0.0085          # tet grid spacing [m]
    DT = 2.5e-4               # integration substep [s]
    E_YOUNG = 4.0e4           # flesh Young modulus [Pa]
    NU = 0.42                 # Poisson ratio (near incompressible)
    DAMPING = 4.0             # mass-proportional drag [1/s]
    VMAX = 12.0               # velocity clamp [m/s]
    NODE_MIN_FACTOR = 0.7     # node-node non-penetration distance (x h)
    ATTACH_MARGIN = 0.002     # flesh-bone rigid attachment shell [m]
    REFERENCE_FORCE = 25.0    # frozen positive force scale [N]
    SEARCH_RADIUS = 4         # embedding search rings (x h)

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        init = scene.initial_state()
        self.V0 = np.array(init.position, dtype=float)
        self.n_surf = len(self.V0)
        self.fixed_surf = np.asarray(init.attributes["fixed_mask"], dtype=np.int32)
        self.web_surf = np.asarray(init.attributes["web_region"], dtype=np.int32)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)
        self.duration = float(cfg.get("duration", 2.4))

        root = Path(__file__).resolve().parents[1]
        verts, faces = _load_obj(root / "assets" / "hand_flesh.obj")

        self._build_bones(cfg)
        self._build_volume(verts, faces)
        self._build_embedding()
        self._finish_setup()

        # dynamic state
        self.x = self.X0_nodes.copy()
        self.v = np.zeros_like(self.X0_nodes)
        self.t = 0.0
        self.J_interval = np.zeros(3)
        self.t_frame_prev = 0.0
        self.frame_count = 0
        self.thumb_prev = self._thumb_endpoints(0.0)
        self._peak_speed = 0.0
        self._min_det = 1.0

    # ----------------------------------------------------------- scene setup
    def _build_bones(self, cfg: dict) -> None:
        objects = {}
        geo = cfg.get("scene_geometry") or {}
        for item in geo.get("objects", []):
            objects[item.get("id")] = item

        def capsule(obj):
            g = obj["geometry"]
            tr = obj.get("transform", {})
            c = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), float)
            R = _quat_to_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            axis = R @ np.array([0.0, 0.0, 1.0])
            half = float(g["height"]) / 2.0
            return c - axis * half, c + axis * half, float(g["radius"])

        self.index_seg = capsule(objects["index_bone"])
        self.thumb_rest = capsule(objects["thumb_bone"])
        self.thumb_center = 0.5 * (self.thumb_rest[0] + self.thumb_rest[1])

        drive = cfg.get("thumb_drive", {})
        self.drive_t0 = float(drive.get("start_time", 0.3))
        self.drive_t1 = float(drive.get("end_time", 1.65))
        axis = np.asarray(drive.get("rotation_axis", [0.0, 0.0, 1.0]), float)
        self.drive_axis = axis / np.linalg.norm(axis)
        self.drive_deg = float(drive.get("rotation_degrees", 34.0))
        self.drive_trans = np.asarray(drive.get("translation", [0.0, 0.0, 0.0]), float)

    def _thumb_pose(self, t: float):
        s = _smoothstep(t, self.drive_t0, self.drive_t1)
        theta = math.radians(self.drive_deg) * s
        c, sn = math.cos(theta), math.sin(theta)
        trans = self.drive_trans * s
        return c, sn, trans

    def _thumb_apply(self, pts: np.ndarray, t: float) -> np.ndarray:
        c, sn, trans = self._thumb_pose(t)
        p = pts - self.thumb_center
        out = np.empty_like(p)
        out[:, 0] = c * p[:, 0] - sn * p[:, 1]
        out[:, 1] = sn * p[:, 0] + c * p[:, 1]
        out[:, 2] = p[:, 2]
        return out + self.thumb_center + trans

    def _thumb_endpoints(self, t: float) -> np.ndarray:
        return self._thumb_apply(np.stack([self.thumb_rest[0], self.thumb_rest[1]]), t)

    # ------------------------------------------------------ tet mesh build
    def _build_volume(self, verts: np.ndarray, faces: np.ndarray) -> None:
        h = self.VOXEL_H
        lo, hi = verts.min(0), verts.max(0)
        origin = lo - h
        dims = np.ceil((hi - lo + 2 * h) / h).astype(int) + 1  # lattice nodes/dim
        nx, ny, nz = (int(v) for v in dims)
        tri = verts[faces]  # (T,3,3)

        # ---- bin triangles by 2-D cell (dilated bbox) for ray parity test
        ci0 = np.clip(((tri[:, :, 0].min(1) - origin[0]) / h).astype(int) - 1, 0, nx - 2)
        ci1 = np.clip(((tri[:, :, 0].max(1) - origin[0]) / h).astype(int) + 1, 0, nx - 2)
        cj0 = np.clip(((tri[:, :, 1].min(1) - origin[1]) / h).astype(int) - 1, 0, ny - 2)
        cj1 = np.clip(((tri[:, :, 1].max(1) - origin[1]) / h).astype(int) + 1, 0, ny - 2)
        cnt = (ci1 - ci0 + 1) * (cj1 - cj0 + 1)
        tri_id = np.repeat(np.arange(len(tri)), cnt)
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        k = np.arange(int(cnt.sum())) - starts[tri_id]
        wid = cj1 - cj0 + 1
        cell_i = ci0[tri_id] + k // wid[tri_id]
        cell_j = cj0[tri_id] + k % wid[tri_id]
        cell_id = cell_j * (nx - 1) + cell_i
        order = np.argsort(cell_id, kind="stable")
        sorted_cell = cell_id[order]
        sorted_tri = tri_id[order]

        # ---- per-column vertical ray parity for cube centers
        inside = np.zeros((nx - 1, ny - 1, nz - 1), dtype=bool)
        eps = 1e-7
        z_levels = origin[2] + (np.arange(nz - 1) + 0.5) * h
        for i in range(nx - 1):
            for j in range(ny - 1):
                cid = j * (nx - 1) + i
                a = np.searchsorted(sorted_cell, cid, "left")
                b = np.searchsorted(sorted_cell, cid, "right")
                if b <= a:
                    continue
                tris = sorted_tri[a:b]
                A, B, C = tri[tris, 0], tri[tris, 1], tri[tris, 2]
                px = origin[0] + (i + 0.5) * h + eps
                py = origin[1] + (j + 0.5) * h + eps
                den = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - \
                      (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
                good = np.abs(den) > 1e-14
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = ((px - A[:, 0]) * (C[:, 1] - A[:, 1]) -
                         (py - A[:, 1]) * (C[:, 0] - A[:, 0])) / den
                    v = ((B[:, 0] - A[:, 0]) * (py - A[:, 1]) -
                         (B[:, 1] - A[:, 1]) * (px - A[:, 0])) / den
                m = good & (u > 0) & (v > 0) & (u + v < 1)
                if not m.any():
                    continue
                zh = A[m, 2] + u[m] * (B[m, 2] - A[m, 2]) + v[m] * (C[m, 2] - A[m, 2])
                zh = np.sort(zh)
                above = len(zh) - np.searchsorted(zh, z_levels, "right")
                inside[i, j, :] = (above % 2) == 1

        # drop cubes fully inside the fixed forearm clamp region
        corner_max_x = origin[0] + (np.arange(nx - 1) + 1) * h
        inside &= corner_max_x[:, None, None] > -0.150 + 1e-9
        self.cube_inside = inside

        kept = np.nonzero(inside.ravel())[0]
        ki, kj, kk = np.unravel_index(kept, inside.shape)
        nc = len(kept)

        # lattice corner ids of each kept cube
        lin = lambda i, j, k: (i * ny + j) * nz + k
        corners = np.empty((nc, 8), dtype=np.int64)
        for b in range(8):
            di, dj, dk = (b & 1), ((b >> 1) & 1), ((b >> 2) & 1)
            corners[:, b] = lin(ki + di, kj + dj, kk + dk)
        uniq, inverse = np.unique(corners, return_inverse=True)
        node_ids = inverse.reshape(nc, 8)

        li, lj, lk = np.unravel_index(uniq, (nx, ny, nz))
        self.X0_nodes = np.stack([origin[0] + li * h,
                                  origin[1] + lj * h,
                                  origin[2] + lk * h], axis=1)

        # conforming 6-tet split around the main cube diagonal
        tet_local = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                              [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]], dtype=np.int64)
        self.tets = node_ids[:, tet_local].reshape(nc * 6, 4)

        self.grid = (origin, (nx, ny, nz))

    # ------------------------------------------------- surface embedding
    def _build_embedding(self) -> None:
        origin, (nx, ny, nz) = self.grid
        h = self.VOXEL_H
        cube_id = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int64)
        cube_id[self.cube_inside] = np.arange(self.cube_inside.sum())
        nc = int(self.cube_inside.sum())
        tet_local = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                              [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]], dtype=np.int64)
        kept_flat = np.nonzero(self.cube_inside.ravel())[0]
        ki, kj, kk = np.unravel_index(kept_flat, self.cube_inside.shape)
        # map lattice corner -> node id
        lin = lambda i, j, k: (i * ny + j) * nz + k
        # rebuild node ids per cube (same as in _build_volume)
        corners = np.empty((nc, 8), dtype=np.int64)
        for b in range(8):
            di, dj, dk = (b & 1), ((b >> 1) & 1), ((b >> 2) & 1)
            corners[:, b] = lin(ki + di, kj + dj, kk + dk)
        uniq, inverse = np.unique(corners, return_inverse=True)
        node_ids = inverse.reshape(nc, 8)
        cube_tets = node_ids[:, tet_local].reshape(nc * 6, 4).reshape(nc, 6, 4)

        base = np.floor((self.V0 - origin) / h).astype(int)
        base[:, 0] = np.clip(base[:, 0], 0, nx - 2)
        base[:, 1] = np.clip(base[:, 1], 0, ny - 2)
        base[:, 2] = np.clip(base[:, 2], 0, nz - 2)

        n = self.n_surf
        embed_tet = np.full(n, -1, dtype=np.int64)
        bary = np.zeros((n, 4))
        resolved = np.zeros(n, dtype=bool)

        for radius in range(self.SEARCH_RADIUS + 1):
            if resolved.all():
                break
            if radius == 0:
                shells = [(0, 0, 0)]
            else:
                shells = [(di, dj, dk)
                          for di in range(-radius, radius + 1)
                          for dj in range(-radius, radius + 1)
                          for dk in range(-radius, radius + 1)
                          if max(abs(di), abs(dj), abs(dk)) == radius]
            for off in shells:
                cand = np.nonzero(~resolved)[0]
                if len(cand) == 0:
                    break
                ijk = base[cand] + np.asarray(off)
                ok = ((ijk[:, 0] >= 0) & (ijk[:, 0] <= nx - 2) &
                      (ijk[:, 1] >= 0) & (ijk[:, 1] <= ny - 2) &
                      (ijk[:, 2] >= 0) & (ijk[:, 2] <= nz - 2))
                cand = cand[ok]
                if len(cand) == 0:
                    continue
                ijk = ijk[ok]
                cid = cube_id[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                sel = cid >= 0
                cand = cand[sel]
                cid = cid[sel]
                if len(cand) == 0:
                    continue
                t6 = cube_tets[cid]                       # (m,6,4)
                Xn = self.X0_nodes[t6]                    # (m,6,4,3)
                P = self.V0[cand]                         # (m,3)
                D = Xn[:, :, 1:, :] - Xn[:, :, 0:1, :]
                rhs = (P[:, None, :] - Xn[:, :, 0, :])[:, :, :, None]
                with np.errstate(all="ignore"):
                    sol = np.linalg.solve(D, rhs)[:, :, :, 0]   # (m,6,3)
                okb = (sol >= -1e-9).all(-1) & (sol.sum(-1) <= 1.0 + 1e-9)
                anyok = okb.any(1)
                if not anyok.any():
                    continue
                idx = np.nonzero(anyok)[0]
                first = np.argmax(okb[idx], axis=1)
                vt = cand[idx]
                embed_tet[vt] = t6[idx, first]
                b3 = sol[idx, first]                      # (m,3)
                b0 = 1.0 - b3.sum(-1)
                bary[vt, 1:] = b3
                bary[vt, 0] = b0
                resolved[vt] = True

        rem = np.nonzero(~resolved)[0]
        if len(rem):
            centroids = self.X0_nodes[self.tets].mean(axis=1)
            tree = cKDTree(centroids)
            _, nn = tree.query(self.V0[rem])
            tn = self.tets[nn]
            Xn = self.X0_nodes[tn]
            D = Xn[:, 1:, :] - Xn[:, 0:1, :]
            sol = np.linalg.solve(D, (self.V0[rem] - Xn[:, 0, :])[:, :, None])[:, :, 0]
            embed_tet[rem] = tn
            bary[rem, 1:] = sol
            bary[rem, 0] = 1.0 - sol.sum(-1)
            resolved[rem] = True

        self.embed_tet = embed_tet
        self.embed_bary = bary

    # ------------------------------------------------------------ setup
    def _finish_setup(self) -> None:
        X = self.X0_nodes
        tets = self.tets
        P = X[tets]
        Dm = np.transpose(P[:, 1:, :] - P[:, 0:1, :], (0, 2, 1))
        self.inv_Dm = np.linalg.inv(Dm)
        vol = np.linalg.det(Dm) / 6.0
        vol = np.maximum(vol, 1e-12)
        self.vol = vol

        density = 1040.0
        m = np.zeros(len(X))
        for k in range(4):
            m += np.bincount(tets[:, k], weights=0.25 * vol, minlength=len(X))
        self.node_mass = m * density

        # fixed / attached partition
        def seg_dist(Pp, seg):
            a, b, _ = seg
            ab = b - a
            tt = np.clip(np.einsum("ij,j->i", Pp - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            return np.linalg.norm(Pp - (a + tt[:, None] * ab), axis=1)

        self.fixed = (X[:, 0] <= -0.150)
        self.fixed |= seg_dist(X, self.index_seg) < self.index_seg[2] + self.ATTACH_MARGIN
        self.thumb_att = seg_dist(X, self.thumb_rest) < self.thumb_rest[2] + self.ATTACH_MARGIN
        self.thumb_att &= ~self.fixed
        self.constr = self.fixed | self.thumb_att
        self.free = ~self.constr

        # active tets (at least one non-immovable node)
        allfix = self.fixed[tets].all(axis=1)
        self.active = np.nonzero(~allfix)[0]
        self.tets_a = tets[self.active]
        self.inv_Dm_a = self.inv_Dm[self.active]
        self.vol_a = self.vol[self.active]

        mu = self.E_YOUNG / (2.0 * (1.0 + self.NU))
        lam = self.E_YOUNG * self.NU / ((1.0 + self.NU) * (1.0 - 2.0 * self.NU))
        self.mu, self.lam = mu, lam

        # per-surface-vertex mass consistent with the embedding (W^T a = m)
        tetn = tets[self.embed_tet]                       # (Ns,4)
        s = np.zeros(len(X))
        for k in range(4):
            s += np.bincount(tetn[:, k], weights=self.embed_bary[:, k], minlength=len(X))
        share = np.zeros(len(X))
        good = s > 1e-12
        share[good] = self.node_mass[good] / s[good]
        surf_mass = (self.embed_bary * share[tetn]).sum(axis=1)
        # distribute mass of interior (unembedded) nodes proportionally
        interior = self.node_mass.sum() - surf_mass.sum()
        if interior > 0.0 and surf_mass.sum() > 0.0:
            surf_mass = surf_mass * (1.0 + interior / surf_mass.sum())
        self.surf_mass = surf_mass

    # ------------------------------------------------------------- physics
    def _elastic_forces(self) -> np.ndarray:
        tets = self.tets_a
        X = self.x[tets]                                  # (Na,4,3)
        Ds = np.transpose(X[:, 1:, :] - X[:, 0:1, :], (0, 2, 1))
        F = np.matmul(Ds, self.inv_Dm_a)
        U, S, Vt = np.linalg.svd(F)
        RU = np.matmul(U, Vt)
        sgn = np.sign(np.linalg.det(RU))
        D = np.ones_like(S)
        D[:, 2] = sgn
        R = np.einsum("nij,nj,njk->nik", U, D, Vt)
        J = S.prod(axis=1) * sgn
        self._min_det = min(self._min_det, float(J.min()))
        with np.errstate(all="ignore"):
            FinvT = np.transpose(np.linalg.inv(F), (0, 2, 1))
        P = 2.0 * self.mu * (F - R) + self.lam * ((J - 1.0) * J)[:, None, None] * FinvT
        H = -self.vol_a[:, None, None] * np.matmul(P, np.transpose(self.inv_Dm_a, (0, 2, 1)))
        contrib = np.empty((len(tets), 4, 3))
        contrib[:, 0, :] = -H.sum(axis=2)
        contrib[:, 1, :] = H[:, :, 0]
        contrib[:, 2, :] = H[:, :, 1]
        contrib[:, 3, :] = H[:, :, 2]
        idx = tets.ravel()
        flat = contrib.reshape(-1, 3)
        f = np.empty((len(self.x), 3))
        for d in range(3):
            f[:, d] = np.bincount(idx, weights=flat[:, d], minlength=len(self.x))
        return f

    def _element_F(self) -> np.ndarray:
        tets = self.tets
        X = self.x[tets]
        Ds = np.transpose(X[:, 1:, :] - X[:, 0:1, :], (0, 2, 1))
        F = np.matmul(Ds, self.inv_Dm)
        return F

    def _bone_contact(self, t_new: float, dt: float) -> None:
        caps = []
        a_t, b_t = self._thumb_endpoints(t_new)
        va = (a_t - self.thumb_prev[0]) / dt
        vb = (b_t - self.thumb_prev[1]) / dt
        caps.append((a_t, b_t, self.thumb_rest[2], va, vb))
        ia, ib, ir = self.index_seg
        caps.append((ia, ib, ir, np.zeros(3), np.zeros(3)))

        free = self.free
        fidx = np.nonzero(free)[0]
        for a, b, r, va, vb in caps:
            P = self.x[fidx]
            ab = b - a
            tt = np.clip(np.einsum("ij,j->i", P - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            cp = a + tt[:, None] * ab
            diff = P - cp
            d = np.linalg.norm(diff, axis=1)
            hit = (d < r) & (d > 1e-9)
            if not hit.any():
                continue
            n = np.where(hit[:, None], diff / np.maximum(d, 1e-12)[:, None], 0.0)
            v = self.v[fidx]
            vcp = va + tt[:, None] * (vb - va)
            vn = np.einsum("ij,ij->i", v - vcp, n)
            dv = np.where(((vn < 0) & hit)[:, None], -vn[:, None] * n, 0.0)
            newx = np.where(hit[:, None], cp + r * n, P)
            self.x[fidx] = newx
            self.v[fidx] = v + dv
            self.J_interval += (self.node_mass[fidx][:, None] * dv).sum(axis=0)

    def _node_node(self, dt: float) -> None:
        h = self.VOXEL_H
        dmin = self.NODE_MIN_FACTOR * h
        x = self.x
        n = len(x)
        ijk = np.floor(x / h).astype(np.int64)
        O = 1 << 11
        key = ((ijk[:, 0] + O) * 4096 + (ijk[:, 1] + O)) * 4096 + (ijk[:, 2] + O)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        pi: list[np.ndarray] = []
        pj: list[np.ndarray] = []
        for di, dj, dk in _HALF_OFFSETS:
            koff = (np.int64(di) * 4096 + dj) * 4096 + dk
            target = key + koff
            pos = np.clip(np.searchsorted(sk, target), 0, n - 1)
            hit = sk[pos] == target
            ii = np.nonzero(hit)[0]
            if len(ii):
                pi.append(ii)
                pj.append(order[pos[ii]])
        same = np.nonzero(sk[1:] == sk[:-1])[0]
        if len(same):
            pi.append(order[same])
            pj.append(order[same + 1])
        if not pi:
            return
        i = np.concatenate(pi)
        j = np.concatenate(pj)
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1)
        rest = np.linalg.norm(self.X0_nodes[j] - self.X0_nodes[i], axis=1)
        keep = (dist < dmin) & (rest > 1.5 * h) & (dist > 1e-9)
        i, j, d, dist = i[keep], j[keep], d[keep], dist[keep]
        if len(i) == 0:
            return
        nhat = d / dist[:, None]
        corr = dmin - dist
        fi = self.free[i]
        fj = self.free[j]
        mi = self.node_mass[i]
        mj = self.node_mass[j]
        both = fi & fj
        onlyi = fi & ~fj
        onlyj = fj & ~fi
        wsum = mi + mj
        wi = np.where(both, mj / np.maximum(wsum, 1e-30), np.where(onlyi, 1.0, 0.0))
        wj = np.where(both, mi / np.maximum(wsum, 1e-30), np.where(onlyj, 1.0, 0.0))
        np.add.at(x, i, (-wi * corr)[:, None] * nhat)
        np.add.at(x, j, (wj * corr)[:, None] * nhat)
        vrel = np.einsum("ij,ij->i", self.v[j] - self.v[i], nhat)
        mu = np.where(both, mi * mj / np.maximum(wsum, 1e-30),
                      np.where(onlyi | onlyj, np.where(onlyi, mi, mj), 0.0))
        imp = np.where(vrel < 0.0, -vrel * mu, 0.0)
        dvi = np.where(fi[:, None], (-(imp / np.maximum(mi, 1e-30))[:, None] * nhat), 0.0)
        dvj = np.where(fj[:, None], ((imp / np.maximum(mj, 1e-30))[:, None] * nhat), 0.0)
        np.add.at(self.v, i, dvi)
        np.add.at(self.v, j, dvj)
        self.J_interval += (mi[:, None] * dvi + mj[:, None] * dvj).sum(axis=0)

    def _substep(self, dt: float) -> None:
        t_new = self.t + dt
        att = self.thumb_att
        v_old = self.v[att].copy()
        xk = self._thumb_apply(self.X0_nodes[att], t_new)
        vk = (xk - self.x[att]) / dt
        self.x[att] = xk
        self.v[att] = vk

        f = self._elastic_forces()
        f += self.node_mass[:, None] * self.gravity
        self.J_interval += dt * (self.node_mass[:, None] * self.gravity).sum(axis=0)
        self.J_interval -= dt * f[self.constr].sum(axis=0)
        self.J_interval += (self.node_mass[att, None] * (vk - v_old)).sum(axis=0)

        free = self.free
        fidx = np.nonzero(free)[0]
        self.v[fidx] += dt * f[fidx] / self.node_mass[fidx, None]

        damp = max(0.0, 1.0 - self.DAMPING * dt)
        before = self.v[fidx].copy()
        self.v[fidx] *= damp
        self.J_interval += (self.node_mass[fidx, None] * (self.v[fidx] - before)).sum(axis=0)

        sp = np.linalg.norm(self.v[fidx], axis=1)
        self._peak_speed = max(self._peak_speed, float(sp.max()) if len(sp) else 0.0)
        over = sp > self.VMAX
        if over.any():
            scale = np.where(over, self.VMAX / np.maximum(sp, 1e-12), 1.0)
            pre = self.v[fidx].copy()
            self.v[fidx] *= scale[:, None]
            self.J_interval += (self.node_mass[fidx, None] * (self.v[fidx] - pre)).sum(axis=0)

        self.x[fidx] += dt * self.v[fidx]

        self._bone_contact(t_new, dt)
        self._node_node(dt)

        # finite-state guard (deterministic recovery)
        bad = ~np.isfinite(self.x).all(1) | ~np.isfinite(self.v).all(1)
        if bad.any():
            bi = np.nonzero(bad)[0]
            self.x[bi] = self.X0_nodes[bi]
            self.v[bi] = 0.0

        self.thumb_prev = self._thumb_endpoints(t_new)
        self.t = t_new

    # ------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        while t - self.t > 1e-12:
            self._substep(min(self.DT, t - self.t))
        return self._frame(t)

    def _frame(self, t: float) -> StateFrame:
        tetn = self.tets[self.embed_tet]
        w = self.embed_bary
        pos = np.einsum("nk,nkd->nd", w, self.x[tetn])
        vel = np.einsum("nk,nkd->nd", w, self.v[tetn])
        F = self._element_F()

        dti = t - self.t_frame_prev
        if self.frame_count == 0 or dti <= 0.0:
            fext = np.zeros(3)
        else:
            fext = self.J_interval / dti

        frame = StateFrame(
            time=t,
            position=pos,
            velocity=vel,
            mass=self.surf_mass,
            attributes={
                "mass_quantity": self.surf_mass,
                "fixed_mask": self.fixed_surf,
                "web_region": self.web_surf,
            },
            fields={
                "budget_external_force": fext[None, None, :],
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F,
            },
            scalars={
                "reference_force": self.REFERENCE_FORCE,
                "solid_dimension": 3.0,
                "nominal_density": 1040.0,
            },
        )
        frame.validate()
        self.J_interval = np.zeros(3)
        self.t_frame_prev = t
        self.frame_count += 1
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "corotated-fem-embedded-surface",
            "nodes": int(len(self.X0_nodes)),
            "tets": int(len(self.tets)),
            "active_tets": int(len(self.tets_a)),
            "mu": float(self.mu),
            "lambda": float(self.lam),
            "substep_dt": self.DT,
            "peak_node_speed": self._peak_speed,
            "min_element_jacobian": self._min_det,
            "final_time": self.t,
        }
