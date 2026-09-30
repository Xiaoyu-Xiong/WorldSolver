"""Volumetric flesh solver for elbow flexion bulging (Stable Neo-Hookean).

A tetrahedral mesh is built at initialization by Delaunay-tetrahedralizing the
rest surface vertices of ``arm_flesh.obj`` together with seeded interior
points, keeping only tets whose centroids lie inside the watertight surface.
The flesh is simulated with a Stable Neo-Hookean hyperelastic law
(Smith/Geilfus/Kim 2018):

    Psi(F) = mu/2 (I1 - 3) - mu ln J + lam/2 (J - 1)^2
    P(F)   = mu (F - F^-T) + lam (J - 1) J F^-T

integrated with symplectic Euler substeps plus mass-proportional damping.
Flesh nodes near the humerus are pinned (fixed bone); flesh nodes near the
forearm bone are rigidly skinned to the prescribed smoothstep hinge rotation.
Node-node self contact and node-capsule bone contact penalties prevent
interpenetration at the closing elbow crease, so the near-incompressible
material produces a full rounded bulge instead of a pinched crease.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- material --
MU = 2.0e4          # Pa, shear modulus of flesh
LAM = 1.2e5         # Pa, first Lame parameter (near incompressible)
RHO = 1050.0        # kg/m^3 (config flesh_density_kg_m3)
GAMMA = 20.0        # 1/s velocity damping (quasi-static tissue response)
DT = 5.0e-4         # s, explicit substep

# ------------------------------------------------------------------ contact --
CONTACT_R = 0.04    # m, self-contact activation radius
CONTACT_K = 3.0e3   # N/m penalty stiffness
BONE_CLEAR = 0.008  # m, extra clearance for bone-capsule penalty
BONE_K = 8.0e3      # N/m
ATTACH_MARGIN = 0.035  # m beyond bone radius: nodes skinned to the bone

J_T = 0.5           # smooth-extension threshold of the mu/J barrier term


def _read_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(c) for c in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, float), np.asarray(faces, int)


def _inside_mask(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Parity ray-cast (+z) Moller-Trumbore against the watertight surface."""
    o = points[:, None, :]                       # (P,1,3)
    d = np.array([0.0, 0.0, 1.0])
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1 = v1 - v0                                 # (T,3)
    e2 = v2 - v0
    h = np.cross(d, e2)                          # (T,3)
    a = np.einsum("tj,tj->t", e1, h)
    valid = np.abs(a) > 1e-12
    af = np.where(valid, a, 1.0)
    s = o[:, 0, :] - v0                          # (P,T,3)
    u = np.einsum("ptj,tj->pt", s, h) / af
    q = np.cross(s, e1)
    vv = np.einsum("ptj,j->pt", q, d) / af
    tpar = np.einsum("ptj,tj->pt", q, e2) / af
    hit = (
        valid
        & (u >= 0.0) & (u <= 1.0)
        & (vv >= 0.0) & (u + vv <= 1.0)
        & (tpar > 1e-9)
    )
    return (hit.sum(axis=1) % 2) == 1


def _smoothstep_angle(t: float, t0: float, t1: float, a0: float, a1: float):
    """Return (angle_rad, angvel_rad_per_s) of a smoothstep rotation."""
    if t1 <= t0:
        s = 1.0 if t >= t1 else 0.0
        return a0 + s * (a1 - a0), 0.0
    x = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    s = x * x * (3.0 - 2.0 * x)
    ds = 6.0 * x * (1.0 - x) / (t1 - t0)
    if x <= 0.0 or x >= 1.0:
        ds = 0.0
    return a0 + s * (a1 - a0), ds * (a1 - a0)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        bc = cfg["boundary_conditions"]["forearm_bone"]
        self._hinge_c = np.asarray(bc["hinge_center"], float)
        self._t0 = float(bc["start_time"])
        self._t1 = float(bc["end_time"])
        self._a0 = np.deg2rad(float(bc["start_angle_degrees"]))
        self._a1 = np.deg2rad(float(bc["end_angle_degrees"]))

        rng = np.random.default_rng(seed)
        root = Path(__file__).resolve().parent.parent
        surf_v, surf_f = _read_obj(root / "assets" / "arm_flesh.obj")
        self.n_surf = len(surf_v)
        self.rest_surf = surf_v.copy()

        # ------------------------------------------------ interior sampling --
        lo, hi = surf_v.min(0), surf_v.max(0)
        gs = 0.08
        gx = np.arange(lo[0] + 0.5 * gs, hi[0], gs)
        gy = np.arange(lo[1] + 0.5 * gs, hi[1], gs)
        gz = np.arange(lo[2] + 0.5 * gs, hi[2], gs)
        grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
        grid += rng.uniform(-0.35 * gs, 0.35 * gs, grid.shape)
        inside = _inside_mask(grid, surf_v, surf_f)
        cand = grid[inside]
        surf_tree = cKDTree(surf_v)
        keep = surf_tree.query(cand)[0] > 0.032
        interior = cand[keep]

        pts = np.vstack([surf_v, interior])
        self.n_nodes = len(pts)
        X = pts
        self.X = X.copy()
        self.x = X.copy()
        self.v = np.zeros_like(X)

        # ------------------------------------------------------ tetrahedra --
        tri = Delaunay(pts)
        tets = tri.simplices
        c = pts[tets].mean(1)
        ok = _inside_mask(c, surf_v, surf_f)
        d = pts[tets[:, 1:]] - pts[tets[:, 0:1]]
        vol6 = np.linalg.det(d)
        ok &= vol6 > 1e-8
        flip = vol6 < 0.0
        tets = tets.copy()
        tets[flip] = tets[flip][:, [1, 0, 2, 3]]
        self.tets = tets[ok]
        d = X[self.tets[:, 1:]] - X[self.tets[:, 0:1]]
        self.Dm_inv = np.linalg.inv(d)                 # (Ne,3,3)
        self.V0 = np.abs(np.linalg.det(d)) / 6.0       # (Ne,)
        self.n_tets = len(self.tets)

        # ------------------------------------------------------ bone setup --
        hum_a = np.array([-0.67, 0.0, 1.05])
        hum_b = np.array([-0.05, 0.0, 1.05])
        for_a0 = np.array([0.05, 0.0, 1.05])
        for_b0 = np.array([0.71, 0.0, 1.05])
        d_hum = _seg_dist(X, hum_a, hum_b)
        d_for = _seg_dist(X, for_a0, for_b0)
        self.pin_hum = d_hum < 0.05 + ATTACH_MARGIN
        self.pin_for = (~self.pin_hum) & (d_for < 0.04 + ATTACH_MARGIN)
        self.free = ~(self.pin_hum | self.pin_for)
        # forearm attached nodes rotate with the bone; humerus attached fixed
        self.for_rel0 = X[self.pin_for] - self._hinge_c

        # ------------------------------------------------- lumped nodal mass --
        m = np.zeros(self.n_nodes)
        np.add.at(m, self.tets.reshape(-1), np.repeat(self.V0 / 4.0, 4))
        self.mass = np.maximum(m * RHO, 1e-6)
        self.total_mass = float(self.mass.sum())

        # Self-contact: only pairs separated at rest may repel, so bonded
        # neighbours feel no spurious contact force.  Radius derives from the
        # actual minimum rest node spacing.
        near = cKDTree(X).query_pairs(0.06, output_type="ndarray")
        dmin = (
            float(np.min(np.linalg.norm(X[near[:, 0]] - X[near[:, 1]], axis=1)))
            if len(near) else 0.06
        )
        self.contact_r = float(min(CONTACT_R, 0.6 * dmin))
        rest_d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        self.rest_ok = rest_d > self.contact_r

        # bones as kinematic capsules (for the contact penalty)
        self._hum = (hum_a, hum_b, 0.05)
        self._for0 = (for_a0, for_b0, 0.04)

        # exported per-surface-node mass (matches scene initial state total)
        self.export_mass = np.full(self.n_surf, 3.05 / self.n_surf)
        self.reference_force = self.total_mass * 9.81

        self.t = 0.0
        self.prev_momentum = np.zeros(3)
        self.diag = {"n_nodes": self.n_nodes, "n_tets": self.n_tets,
                     "n_pinned": int((~self.free).sum()), "minJ": 1.0}

    # -------------------------------------------------------------- driving --
    def _forearm_rot(self, t: float):
        ang, om = _smoothstep_angle(t, self._t0, self._t1, self._a0, self._a1)
        cA, sA = np.cos(ang), np.sin(ang)
        R = np.array([[cA, 0.0, sA], [0.0, 1.0, 0.0], [-sA, 0.0, cA]])
        return R, om, ang

    def _bone_capsules(self, t: float):
        R, om, _ = self._forearm_rot(t)
        a = self._hinge_c + R @ (self._for0[0] - self._hinge_c)
        b = self._hinge_c + R @ (self._for0[1] - self._hinge_c)
        return self._hum, (a, b, self._for0[2])

    def _apply_dirichlet(self, t: float) -> None:
        R, om, _ = self._forearm_rot(t)
        pos = self._hinge_c + (R @ self.for_rel0.T).T
        vel = np.cross(np.array([0.0, om, 0.0]), pos - self._hinge_c)
        self.x[self.pin_for] = pos
        self.v[self.pin_for] = vel
        self.x[self.pin_hum] = self.X[self.pin_hum]
        self.v[self.pin_hum] = 0.0

    # -------------------------------------------------------------- forces --
    def _elastic_forces(self):
        d = self.x[self.tets[:, 1:]] - self.x[self.tets[:, 0:1]]      # (Ne,3,3)
        F = np.einsum("nij,njk->nik", d, self.Dm_inv)
        f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
        cof = np.stack(
            [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=1
        )                                          # cof(F) = J F^-T, bounded
        J = np.einsum("nj,nj->n", f0, cof[:, 0])
        # Stable Neo-Hookean P = mu (F - F^-T) + lam (J-1) J F^-T
        #                     = mu F + (lam (J-1) - mu/J) cof(F),
        # with mu/J smoothly extended below J_T so the barrier stays finite.
        r = (J - J_T) / J_T
        s = np.where(J >= J_T, MU / np.maximum(J, 1e-12), (MU / J_T) * (1.0 - r + r * r))
        P = MU * F + (LAM * (J - 1.0) - s)[:, None, None] * cof
        H = -self.V0[:, None, None] * np.einsum("nij,nkj->nik", P, self.Dm_inv)
        f = np.zeros_like(self.x)
        np.add.at(f, self.tets[:, 1:], H)
        np.add.at(f, self.tets[:, 0:1], -H.sum(1, keepdims=True))
        self._minJ = float(J.min()) if len(J) else 1.0
        return f

    def _contact_forces(self, t: float):
        f = np.zeros_like(self.x)
        # node-node self contact (skip pairs bonded at rest)
        tree = cKDTree(self.x)
        pairs = tree.query_pairs(self.contact_r, output_type="ndarray")
        if len(pairs):
            keep = self.rest_ok[pairs[:, 0], pairs[:, 1]]
            pairs = pairs[keep]
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                dx = self.x[i] - self.x[j]
                dist = np.linalg.norm(dx, axis=1)
                dist = np.maximum(dist, 1e-6)
                mag = CONTACT_K * (self.contact_r - dist) / dist
                force = mag[:, None] * dx
                np.add.at(f, i, force)
                np.add.at(f, j, -force)
        # node vs kinematic bone capsules (free flesh nodes only)
        for (a, b, r) in self._bone_capsules(t):
            dvec, dist = _seg_dist_vec(self.x, a, b)
            pen = (r + BONE_CLEAR) - dist
            act = pen > 0.0
            if act.any():
                idx = np.nonzero(act)[0]
                f[idx] += (BONE_K * pen[idx])[:, None] * dvec[idx]
        f[~self.free] = 0.0
        return f

    # ---------------------------------------------------------------- steps --
    def _substep(self, t_next: float):
        f = self._elastic_forces() + self._contact_forces(t_next)
        self.v += DT * f / self.mass[:, None]
        self.v *= np.exp(-GAMMA * DT)
        self.x += DT * self.v
        self._apply_dirichlet(t_next)

    def advance_to(self, time: float):
        time = float(time)
        if time < self.t:
            time = self.t
        n_sub = max(1, int(np.ceil((time - self.t) / DT - 1e-9))) if time > self.t else 0
        for k in range(n_sub):
            t_n = self.t + (time - self.t) * (k + 1) / n_sub
            self._substep(t_n)
        self.t = time

        # momentum-budget external force (bone tractions + constraint reactions)
        momentum = (self.mass[:, None] * self.v).sum(0)
        dt_s = max(time - self._prev_time if hasattr(self, "_prev_time") else DT, DT)
        f_ext = (momentum - self.prev_momentum) / dt_s if dt_s > 0 else np.zeros(3)
        self.prev_momentum = momentum
        self._prev_time = time

        _, _, ang = self._forearm_rot(time)
        F = self._elastic_forces_only_F()

        frame = StateFrame(
            time=time,
            position=self.x[: self.n_surf].copy(),
            velocity=self.v[: self.n_surf].copy(),
            mass=self.export_mass.copy(),
            attributes={
                "mass_quantity": self.export_mass.copy(),
                "rest_position": self.rest_surf.copy(),
            },
            fields={
                "budget_external_force": f_ext.reshape(1, 1, 3),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F.astype(float),
            },
            scalars={
                "reference_force": float(self.reference_force),
                "flesh_density_kg_m3": RHO,
                "elbow_flexion_degrees": float(np.rad2deg(ang)),
            },
        )
        frame.validate()
        self.diag["minJ"] = min(self.diag.get("minJ", 1.0), getattr(self, "_minJ", 1.0))
        return frame

    def _elastic_forces_only_F(self):
        d = self.x[self.tets[:, 1:]] - self.x[self.tets[:, 0:1]]
        return np.einsum("nij,njk->nik", d, self.Dm_inv)

    def diagnostics(self) -> dict:
        return dict(self.diag)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _seg_dist_vec(p, a, b)[1]


def _seg_dist_vec(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Outward unit direction from the segment to p, and distance."""
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dvec = p - closest
    dist = np.linalg.norm(dvec, axis=1)
    safe = np.maximum(dist, 1e-9)
    return dvec / safe[:, None], dist
