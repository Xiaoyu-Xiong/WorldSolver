"""Volumetric flesh solver for the thumb-index pinch scene.

Physically based model:
  * Tetrahedral FEM discretisation built at initialisation from the watertight
    hand surface (scipy Delaunay over surface vertices + interior lattice
    points, filtered by an inside/outside occupancy grid and quality metrics).
  * Compressible neo-Hookean hyperelasticity (mu/2 (tr(F^T F) - 3) - mu ln J
    + lambda/2 (ln J)^2), stress free at F = I, with a log-J barrier plus the
    F^{-T} term that keeps inverted elements strongly repulsive (the stable
    neo-Hookean flesh behaviour of the source paper).
  * Explicit sub-stepped dynamics below the CFL limit with lumped masses and
    light velocity damping (flesh viscoelasticity / numerical stabilisation).
  * Kinematic boundary conditions: forearm cuff nodes (x <= -0.150) fixed,
    nodes near the index bone held fixed, nodes near the thumb bone rigidly
    carried by the configured thumb drive (rotation about z + translation with
    smoothstep easing). Constraint tractions are measured and exported as the
    external-force budget terms.

Exported trajectory fields (raw physical state only):
  attributes.mass_quantity            [N]  kg
  fields.element_deformation_gradient [Ne,3,3]
  fields.budget_external_force        [1,3,3] N (cuff, index-bone, thumb-bone)
  fields.cumulative_mass_source       [1] kg (no mass added/removed)
  scalars.reference_force             N
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay, cKDTree
from worldsolver.contracts import StateFrame

ASSET = Path(__file__).resolve().parent.parent / "assets" / "hand_flesh.obj"

RHO = 1040.0          # kg/m^3 (nominal flesh density from config)
YOUNG = 3000.0        # Pa, soft flesh
NU = 0.40             # nearly incompressible
MU = YOUNG / (2.0 * (1.0 + NU))
LAM = YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
GAMMA = 6.0           # velocity damping rate (1/s)
VMAX = 5.0            # safety velocity clamp (m/s)
JMIN = 1.0e-6

DEFAULT_DRIVE = {
    "start_time": 0.30,
    "end_time": 1.65,
    "hold_until": 2.40,
    "rotation_degrees": 34.0,
    "translation": [0.018, 0.052, 0.004],
}
THUMB_PIVOT = np.array([0.025, -0.074, -0.002])   # thumb bone centre
INDEX_SEG = np.array([0.105, 0.030, 0.012])       # index bone centre
THUMB_HALF = 0.0525
INDEX_HALF = 0.1025


def _load_obj(path: Path):
    verts = []
    faces = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, float), np.asarray(faces, dtype=np.int64)


def _segment_distance(points: np.ndarray, center: np.ndarray, half_len: float) -> np.ndarray:
    # capsule axis along +x through `center`
    dx = np.clip(points[:, 0] - center[0], -half_len, half_len)
    radial = points - center
    radial[:, 0] -= dx
    return np.linalg.norm(radial, axis=1)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        base = scene.initial_state()
        cfg = getattr(scene, "config", {}) or {}
        drive = dict(DEFAULT_DRIVE)
        drive.update(cfg.get("thumb_drive", {}) or {})
        self.drive = drive
        self.rho = float(cfg.get("initial_conditions", {}).get("nominal_density", RHO))

        self.X = np.array(base.position, float)
        self.N = len(self.X)
        self.node_mass = np.array(base.mass, float)
        self.attrs0 = {k: np.array(v) for k, v in base.attributes.items()}
        verts, faces = _load_obj(ASSET)

        lo = verts.min(0) - 0.02
        hi = verts.max(0) + 0.02
        self._build_occupancy(verts, faces, lo, hi)
        self._build_tet_mesh(lo, hi)
        self._build_boundary_sets()

        self.x = self.X.copy()
        self.v = np.zeros_like(self.X)
        self.t = 0.0
        self.reactions = np.zeros((3, 3))
        self.v_kin_old = np.zeros((len(self.thumb_idx), 3))
        self.ref_force = float(MU * 0.02 * 0.02)  # mu * L^2 with L = 20 mm
        self._diag = {
            "elements": int(self.Ne),
            "dt": float(self.dt),
            "dynamic_nodes": int(self.dyn_idx.size),
            "kinematic_nodes": int(self.thumb_idx.size + self.fixedkin_idx.size),
            "passive_nodes": int(self.passive_idx.size),
        }

    def _build_occupancy(self, verts, faces, lo, hi, cell=0.005):
        dims = np.ceil((hi - lo) / cell).astype(int)
        v0 = verts[faces[:, 0]]
        e1 = verts[faces[:, 1]] - v0
        e2 = verts[faces[:, 2]] - v0
        det = e1[:, 2] * e2[:, 1] - e1[:, 1] * e2[:, 2]
        valid = np.abs(det) > 1e-14
        invdet = np.zeros_like(det)
        invdet[valid] = 1.0 / det[valid]
        A1 = -e2[:, 2]
        A2 = e2[:, 1]
        B1 = e1[:, 2]
        B2 = -e1[:, 1]
        grid = np.zeros(dims, bool)
        ox = lo[0] - 0.01
        xc = lo[0] + (np.arange(dims[0]) + 0.5) * cell
        for iy in range(dims[1]):
            y = lo[1] + (iy + 0.5) * cell
            dy = y - v0[:, 1]
            for iz in range(dims[2]):
                z = lo[2] + (iz + 0.5) * cell
                dz = z - v0[:, 2]
                u = (dy * A1 + dz * A2) * invdet
                w = (dy * B1 + dz * B2) * invdet
                hit = valid & (u >= 0.0) & (u <= 1.0) & (w >= 0.0) & (u + w <= 1.0)
                xh = v0[:, 0] + e1[:, 0] * u + e2[:, 0] * w
                xs = np.sort(xh[hit & (xh > ox)])
                cnt = np.searchsorted(xs, xc, side="left")
                grid[:, iy, iz] = (cnt % 2) == 1
        self.occ_lo = lo
        self.occ_cell = cell
        self.occ_dims = dims
        self.occ = grid

    def _inside(self, points: np.ndarray) -> np.ndarray:
        idx = np.floor((points - self.occ_lo) / self.occ_cell).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < self.occ_dims), axis=1)
        out = np.zeros(len(points), bool)
        if ok.any():
            ii = idx[ok]
            out[ok] = self.occ[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def _build_tet_mesh(self, lo, hi):
        # interior lattice points (~10 mm spacing) strictly inside the surface
        step = 0.010
        axes = [np.arange(a + 0.5 * step, b - 0.4 * step, step) for a, b in zip(lo, hi)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        inside = self._inside(cand)
        interior = cand[inside]
        points = np.vstack([self.X, interior]) if len(interior) else self.X.copy()

        tri = Delaunay(points)
        tet = tri.simplices
        a = points[tet[:, 0]]
        ab = points[tet[:, 1]] - a
        ac = points[tet[:, 2]] - a
        ad = points[tet[:, 3]] - a
        vol = np.abs(np.einsum("ni,ni->n", np.cross(ab, ac), ad)) / 6.0
        e01 = np.linalg.norm(ab, axis=1)
        e02 = np.linalg.norm(ac, axis=1)
        e03 = np.linalg.norm(ad, axis=1)
        e12 = np.linalg.norm(points[tet[:, 2]] - points[tet[:, 1]], axis=1)
        e13 = np.linalg.norm(points[tet[:, 3]] - points[tet[:, 1]], axis=1)
        e23 = np.linalg.norm(points[tet[:, 3]] - points[tet[:, 2]], axis=1)
        min_edge = np.minimum(np.minimum(np.minimum(e01, e02), np.minimum(e03, e12)),
                              np.minimum(e13, e23))
        centroid = (a + points[tet[:, 1]] + points[tet[:, 2]] + points[tet[:, 3]]) / 4.0
        # circumradius (regularised solve: singular tets get R = inf and drop)
        M = 2.0 * np.stack([ab, ac, ad], axis=1)  # rows: [ab;ac;ad] per tet
        M[:, np.arange(3), np.arange(3)] += 1e-12
        rhs = np.stack([
            np.einsum("ni,ni->n", ab, ab),
            np.einsum("ni,ni->n", ac, ac),
            np.einsum("ni,ni->n", ad, ad),
        ], axis=1)
        with np.errstate(all="ignore"):
            cc = np.linalg.solve(M, rhs[..., None])[..., 0]
            R = np.linalg.norm(cc, axis=1)
        R = np.where(np.isfinite(R), R, np.inf)
        keep = (vol > 1e-12) & (min_edge > 0.0015) & (R < 0.045) & self._inside(centroid)
        if keep.sum() < 2000:  # relax quality filters rather than fail the run
            keep = (vol > 1e-13) & (min_edge > 0.0004) & (R < 0.09) & self._inside(centroid)
        tets = tet[keep]
        self.Ne = len(tets)
        self.tets = tets
        self.h_min = float(min_edge[keep].min())

        x0 = self.X[tets[:, 0]]
        Dm = np.stack([self.X[tets[:, 1]] - x0,
                       self.X[tets[:, 2]] - x0,
                       self.X[tets[:, 3]] - x0], axis=2)
        self.DmDet = np.abs(_det3(Dm))
        self.W = np.linalg.inv(Dm)             # Dm^-1
        self.Wt = np.swapaxes(self.W, 1, 2)
        self.V0 = self.DmDet / 6.0

        nodes4 = tets.reshape(-1)
        data = np.ones(4 * self.Ne)
        self.Asm = coo_matrix((data, (nodes4, np.arange(4 * self.Ne))),
                              shape=(self.N, 4 * self.Ne)).tocsr()
        contrib = np.repeat(self.rho * self.V0 / 4.0, 4)
        self.m_lump = np.asarray(self.Asm @ contrib)
        self.has_tet = self.m_lump > 1e-12
        self.inv_m = np.zeros(self.N)
        np.divide(1.0, self.m_lump, out=self.inv_m, where=self.has_tet)

        c_wave = np.sqrt((LAM + 2.0 * MU) / self.rho)
        self.dt = float(np.clip(0.3 * self.h_min / c_wave, 4e-5, 5e-4))

    def _build_boundary_sets(self):
        fixed_mask = self.attrs0.get("fixed_mask")
        if fixed_mask is None:
            fixed = self.X[:, 0] <= -0.150
        else:
            fixed = np.asarray(fixed_mask) > 0
        d_thumb = _segment_distance(self.X, THUMB_PIVOT, THUMB_HALF)
        d_index = _segment_distance(self.X, INDEX_SEG, INDEX_HALF)
        thumb_att = d_thumb <= 0.016
        index_att = (d_index <= 0.0135) & ~thumb_att
        fixed = fixed & ~thumb_att & ~index_att

        self.thumb_idx = np.flatnonzero(thumb_att)
        self.thumb_local = self.X[self.thumb_idx] - THUMB_PIVOT
        self.fixedkin_idx = np.flatnonzero(fixed | index_att)
        self.cuff_idx = np.flatnonzero(fixed)
        self.index_idx = np.flatnonzero(index_att)

        kinematic = fixed | thumb_att | index_att
        passive = (~kinematic) & (~self.has_tet)
        dynamic = self.has_tet & (~kinematic) & (~passive)
        self.passive_idx = np.flatnonzero(passive)
        self.dyn_idx = np.flatnonzero(dynamic)
        if self.passive_idx.size:
            tree = cKDTree(self.X[self.has_tet])
            _, near = tree.query(self.X[self.passive_idx])
            self.passive_anchor = np.flatnonzero(self.has_tet)[near]
        else:
            self.passive_anchor = np.zeros(0, dtype=np.int64)

    # ---------------------------------------------------------------- driving
    def _drive_scale(self, t: float) -> float:
        d = self.drive
        t0 = float(d["start_time"])
        t1 = float(d["end_time"])
        if t <= t0:
            return 0.0
        if t >= t1:
            return 1.0
        u = (t - t0) / (t1 - t0)
        return u * u * (3.0 - 2.0 * u)

    def _thumb_kinematics(self, t: float):
        s = self._drive_scale(t)
        ang = np.deg2rad(float(self.drive["rotation_degrees"])) * s
        c, sn = np.cos(ang), np.sin(ang)
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        trans = np.asarray(self.drive["translation"], float) * s
        pos = self.thumb_local @ R.T + THUMB_PIVOT + trans
        return pos

    # ------------------------------------------------------------------ FEM
    def _elastic(self, x: np.ndarray, want_F: bool = False):
        xs = x[self.tets]
        De = np.empty((self.Ne, 3, 3))
        De[:, :, 0] = xs[:, 1] - xs[:, 0]
        De[:, :, 1] = xs[:, 2] - xs[:, 0]
        De[:, :, 2] = xs[:, 3] - xs[:, 0]
        F = De @ self.W
        c0 = F[:, :, 0]
        c1 = F[:, :, 1]
        c2 = F[:, :, 2]
        cr1 = np.cross(c1, c2)
        cr2 = np.cross(c2, c0)
        cr3 = np.cross(c0, c1)
        J = np.einsum("ni,ni->n", c0, cr1)
        Jc = np.maximum(J, JMIN)
        invJ = 1.0 / np.where(np.abs(J) < JMIN, JMIN, J)
        FinvT = np.empty_like(F)
        FinvT[:, :, 0] = cr1 * invJ[:, None]
        FinvT[:, :, 1] = cr2 * invJ[:, None]
        FinvT[:, :, 2] = cr3 * invJ[:, None]
        P = MU * (F - FinvT) + (LAM * np.log(Jc))[:, None, None] * FinvT
        G = self.V0[:, None, None] * (P @ self.Wt)
        fe = np.empty((self.Ne, 4, 3))
        fe[:, 1] = -G[:, :, 0]
        fe[:, 2] = -G[:, :, 1]
        fe[:, 3] = -G[:, :, 2]
        fe[:, 0] = -(fe[:, 1] + fe[:, 2] + fe[:, 3])
        f = np.asarray(self.Asm @ fe.reshape(-1, 3))
        if want_F:
            return f, F
        return f

    def _substep(self, t_new: float, dt: float, record: bool):
        # elastic forces from current configuration
        f = self._elastic(self.x)

        # prescribed thumb motion
        thumb_new = self._thumb_kinematics(t_new)
        v_thumb = (thumb_new - self.x[self.thumb_idx]) / dt
        if record:
            if self.thumb_idx.size:
                a_thumb = (v_thumb - self.v_kin_old) / dt
                m = self.m_lump[self.thumb_idx][:, None]
                self.reactions[2] = np.sum(m * a_thumb - f[self.thumb_idx], axis=0)
            if self.cuff_idx.size:
                self.reactions[0] = -np.sum(f[self.cuff_idx], axis=0)
            if self.index_idx.size:
                self.reactions[1] = -np.sum(f[self.index_idx], axis=0)

        # integrate dynamic nodes
        di = self.dyn_idx
        self.v[di] += dt * f[di] * self.inv_m[di][:, None]
        self.v[di] *= 1.0 / (1.0 + GAMMA * dt)
        self.v[di] = np.clip(self.v[di], -VMAX, VMAX)
        self.x[di] += dt * self.v[di]

        # kinematic nodes
        self.x[self.thumb_idx] = thumb_new
        self.v[self.thumb_idx] = v_thumb
        self.v_kin_old = v_thumb.copy()
        self.v[self.fixedkin_idx] = 0.0

        # passive skin nodes follow their anchor displacement
        if self.passive_idx.size:
            anc = self.passive_anchor
            self.x[self.passive_idx] = self.X[self.passive_idx] + (self.x[anc] - self.X[anc])
            self.v[self.passive_idx] = self.v[anc]
        self.t = t_new

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time > self.t + 1e-12:
            n = max(1, int(np.ceil((time - self.t) / self.dt)))
            dt = (time - self.t) / n
            for i in range(n):
                self._substep(self.t + dt, dt, record=(i == n - 1))
        elif not hasattr(self, "reactions"):
            self.reactions = np.zeros((3, 3))
        _, F = self._elastic(self.x, want_F=True)
        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.node_mass.copy(),
            attributes={**self.attrs0, "mass_quantity": self.node_mass.copy()},
            fields={
                "element_deformation_gradient": F.astype(np.float32),
                "budget_external_force": self.reactions.reshape(1, 3, 3).astype(float),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "reference_force": self.ref_force,
                "solid_dimension": 3.0,
                "nominal_density": float(self.rho),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)


def _det3(M: np.ndarray) -> np.ndarray:
    return (M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1])
            - M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0])
            + M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0]))
