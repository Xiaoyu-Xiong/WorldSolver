"""Cloth sleeve on articulated cylinders.

Physical solver: XPBD (extended position based dynamics) cloth on a 17x17
grid wrapped into a tube around two kinematic capsules (articulated limbs).
The joint is driven with a smooth bend profile; cloth-body contact uses
signed-distance projection with static/kinetic friction, plus particle-particle
self repulsion so folds do not inter-penetrate.

All quantities are SI (meters, kilograms, seconds). Deterministic: no
stochastic terms are used.
"""
from __future__ import annotations

import json
import os

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

# ---- frozen scene constants (scene_geometry.yaml / config.yaml) ----
PIVOT = np.array([0.0, 0.0, 1.0])          # joint location
UPPER_CENTER = np.array([-0.55, 0.0, 1.0])
LOWER_CENTER0 = np.array([0.55, 0.0, 1.0])
CAP_HALF = 0.65                             # capsule half length along axis
R_UPPER = 0.34
R_LOWER = 0.31
GROUND_HE = np.array([2.4, 2.0, 0.05])
GROUND_C = np.array([0.0, 0.0, -0.05])

# joint drive: smooth bend up to THETA and back (zero velocity at ends)
THETA_MAX = 1.15        # rad (~66 deg)
BEND_PERIOD = 7.0       # full cycle over the simulation duration

# cloth material parameters
THICKNESS = 0.015
WRAP_RADIUS = 2.5 / (2.0 * np.pi)   # strain-free wrap of the 2.5 m width
GAP0 = 0.5 * THICKNESS              # cloth mid-surface stand-off distance
MU = 0.45                            # friction coefficient cloth <-> bodies
RC_SELF = 0.032                      # self-contact repulsion radius

# numerics
SUBSTEP_H = 1.0 / 720.0
CONST_ITER = 5
RELAX = 1.0
DAMPING = 0.35        # 1/s velocity damping (air)
MAX_SPEED = 30.0

ALPHA_STRUCT = 5.0e-4   # compliance 1/k, k ~ 2000 N/m
ALPHA_SHEAR = 5.0e-3
ALPHA_BEND = 0.2


def _theta(t: float) -> float:
    """Joint angle (rad); smooth 0 -> THETA_MAX -> 0."""
    s = 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, BEND_PERIOD) / BEND_PERIOD))
    return THETA_MAX * s


def _dtheta(t: float) -> float:
    return (THETA_MAX * 0.5 * np.pi / BEND_PERIOD) * np.sin(
        np.pi * np.clip(t, 0.0, BEND_PERIOD) / BEND_PERIOD)


def _box_sdf(p: np.ndarray) -> np.ndarray:
    q = np.abs(p - GROUND_C) - GROUND_HE
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


class BenchmarkSolver:
    def __init__(self) -> None:
        self.t = 0.0
        self.x = None
        self.v = None

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.ni, self.nj = (int(v) for v in cloth["grid_shape"])
        ni, nj = self.ni, self.nj
        self.n = ni * nj
        self.gravity = np.asarray(cfg.get("gravity", GRAVITY), dtype=float)

        # rest / material coordinates from frozen topology asset
        path = os.path.join("assets", "cloth_topology.json")
        if os.path.exists(path):
            with open(path) as fh:
                topo = json.load(fh)
            self.rest = np.asarray(topo["vertices"], dtype=float)
        else:
            u = np.linspace(-1.5, 1.5, ni)
            v = np.linspace(-1.25, 1.25, nj)
            uu, vv = np.meshgrid(u, v, indexing="xy")
            self.rest = np.column_stack((uu.ravel(), vv.ravel(),
                                         np.zeros(ni * nj)))
        self.rest = self.rest.reshape(nj, ni, 3)

        u_ax = self.rest[0, :, 0]
        v_ax = self.rest[:, 0, 1]
        self.u_ax, self.v_ax = u_ax, v_ax

        # per particle mass (uniform, total frozen)
        self.total_mass = float(cloth["total_mass"])
        self.mass = np.full(self.n, self.total_mass / self.n)
        self.inv_mass = 1.0 / self.mass

        # boundary vertices (grid rim) for reference attributes
        rim = np.zeros((nj, ni), dtype=bool)
        rim[0, :] = rim[-1, :] = True
        rim[:, 0] = rim[:, -1] = True
        self.boundary_vertex = rim.ravel().astype(np.int32)

        # ---- initial state: fitted tube around the aligned cylinders ----
        # wrap the 2.5 m width into a full circle (seam at the bottom)
        ang = np.linspace(-np.pi, np.pi, nj)
        pos = np.zeros((nj, ni, 3))
        pos[..., 0] = u_ax[None, :]
        pos[..., 1] = WRAP_RADIUS * np.sin(ang)[:, None]
        pos[..., 2] = PIVOT[2] + WRAP_RADIUS * np.cos(ang)[:, None]
        self.x = pos.reshape(self.n, 3).copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0

        # ---- constraints ----
        def vid(i, j):
            return j * ni + i

        ia, ib, rest_l, comp = [], [], [], []
        du = float(u_ax[1] - u_ax[0])
        dv = float(v_ax[1] - v_ax[0])
        diag = float(np.hypot(du, dv))

        # structural (warp and weft)
        for j in range(nj):
            for i in range(ni - 1):
                ia.append(vid(i, j)); ib.append(vid(i + 1, j))
                rest_l.append(du); comp.append(ALPHA_STRUCT)
        for j in range(nj - 1):
            for i in range(ni):
                ia.append(vid(i, j)); ib.append(vid(i, j + 1))
                rest_l.append(dv); comp.append(ALPHA_STRUCT)
        # shear
        for j in range(nj - 1):
            for i in range(ni - 1):
                ia.append(vid(i, j)); ib.append(vid(i + 1, j + 1))
                rest_l.append(diag); comp.append(ALPHA_SHEAR)
                ia.append(vid(i + 1, j)); ib.append(vid(i, j + 1))
                rest_l.append(diag); comp.append(ALPHA_SHEAR)
        # bending (skip-one distance constraints)
        for j in range(nj):
            for i in range(ni - 2):
                ia.append(vid(i, j)); ib.append(vid(i + 2, j))
                rest_l.append(2.0 * du); comp.append(ALPHA_BEND)
        for j in range(nj - 2):
            for i in range(ni):
                ia.append(vid(i, j)); ib.append(vid(i, j + 2))
                rest_l.append(2.0 * dv); comp.append(ALPHA_BEND)
        # seam stitches (close the tube across the wrap direction)
        for i in range(ni):
            ia.append(vid(i, 0)); ib.append(vid(i, nj - 1))
            rest_l.append(0.0); comp.append(ALPHA_STRUCT)
            if nj >= 2:
                ia.append(vid(i, 0)); ib.append(vid(i, nj - 2))
                rest_l.append(dv); comp.append(ALPHA_STRUCT)
                ia.append(vid(i, 1)); ib.append(vid(i, nj - 1))
                rest_l.append(dv); comp.append(ALPHA_STRUCT)
        for i in range(ni - 1):
            ia.append(vid(i, 0)); ib.append(vid(i + 1, nj - 1))
            rest_l.append(du); comp.append(ALPHA_STRUCT)
            ia.append(vid(i + 1, 0)); ib.append(vid(i, nj - 1))
            rest_l.append(du); comp.append(ALPHA_STRUCT)
            ia.append(vid(i, 1)); ib.append(vid(i + 1, nj - 1))
            rest_l.append(diag); comp.append(ALPHA_SHEAR)
            ia.append(vid(i, 0)); ib.append(vid(i + 1, nj - 2))
            rest_l.append(diag); comp.append(ALPHA_SHEAR)

        self.cia = np.asarray(ia, dtype=np.int64)
        self.cib = np.asarray(ib, dtype=np.int64)
        self.crest = np.asarray(rest_l, dtype=float)
        self.ccomp = np.asarray(comp, dtype=float)
        self.nc = len(self.crest)

        # ---- self contact candidate pairs (grid Chebyshev distance >= 2) ----
        ja = np.repeat(np.arange(nj), ni)
        ia_arr = np.tile(np.arange(ni), nj)
        di = ia_arr[:, None] - ia_arr[None, :]
        dj = ja[:, None] - ja[None, :]
        far = (np.abs(di) > 1) | (np.abs(dj) > 1)
        # exclude seam-coincident duplicate pairs (same i, columns 0 and nj-1):
        # they are stitched to distance 0 and must not repel each other
        seam_dup = (np.abs(dj) == nj - 1) & (di == 0)
        upper = np.triu(np.ones((self.n, self.n), dtype=bool), k=1)
        sel = np.where(far & ~seam_dup & upper)
        self.pa, self.pb = sel[0].astype(np.int64), sel[1].astype(np.int64)

    # ------------------------------------------------------------- kinematics
    def _capsules(self, t: float):
        th = _theta(t)
        dth = _dtheta(t)
        axis_u = np.array([1.0, 0.0, 0.0])
        upper = (UPPER_CENTER - CAP_HALF * axis_u,
                 UPPER_CENTER + CAP_HALF * axis_u, R_UPPER, None, None)
        d = np.array([np.cos(th), 0.0, np.sin(th)])
        c = PIVOT + 0.55 * d
        omega = np.array([0.0, -dth, 0.0])
        lower = (c - CAP_HALF * d, c + CAP_HALF * d, R_LOWER, PIVOT, omega)
        return [upper, lower]

    # ------------------------------------------------------------- substepping
    def _solve_constraints(self, x: np.ndarray, h: float) -> None:
        w = self.inv_mass
        wa = w[self.cia]; wb = w[self.cib]
        wsum = wa + wb + self.ccomp / (h * h)
        cnt = np.zeros(self.n)
        np.add.at(cnt, self.cia, 1.0)
        np.add.at(cnt, self.cib, 1.0)
        scale = RELAX / np.maximum(cnt, 1.0)
        for _ in range(CONST_ITER):
            xa = x[self.cia]; xb = x[self.cib]
            d = xb - xa
            L = np.linalg.norm(d, axis=1)
            n = d / np.maximum(L, 1e-12)[:, None]
            C = L - self.crest
            s = -C / wsum
            corr = s[:, None] * n
            dx = np.zeros_like(x)
            np.add.at(dx, self.cia, wa[:, None] * corr)
            np.add.at(dx, self.cib, -wb[:, None] * corr)
            x += dx * scale[:, None]

    def _collide_capsule(self, x, x_prev, h, p1, p2, R, pivot, omega):
        ab = p2 - p1
        ab2 = float(ab @ ab)
        ap = x - p1
        tt = np.clip((ap @ ab) / ab2, 0.0, 1.0)
        q = p1[None, :] + tt[:, None] * ab[None, :]
        rvec = x - q
        dist = np.linalg.norm(rvec, axis=1)
        n = rvec / np.maximum(dist, 1e-12)[:, None]
        pen = GAP0 - (dist - R)
        act = pen > 0.0
        if not np.any(act):
            return
        # surface velocity of the kinematic body at the contact point
        if pivot is not None:
            vs = np.cross(omega[None, :], q - pivot[None, :])
        else:
            vs = np.zeros_like(q)
        # positional projection onto the stand-off surface
        x[act] = q[act] + n[act] * (R + GAP0)
        # friction: oppose tangential relative displacement this substep
        rel = (x[act] - x_prev[act]) - vs[act] * h
        nt = np.sum(rel * n[act], axis=1, keepdims=True)
        rel_t = rel - nt * n[act]
        lt = np.linalg.norm(rel_t, axis=1)
        f = np.minimum(1.0, MU * np.maximum(pen[act], 0.0) /
                       np.maximum(lt, 1e-12))
        x[act] -= rel_t * f[:, None]

    def _collide_ground(self, x, x_prev, h):
        gap = x[:, 2] - 0.0
        pen = GAP0 - gap
        act = (pen > 0.0) & (np.abs(x[:, 0]) < GROUND_HE[0]) & \
              (np.abs(x[:, 1]) < GROUND_HE[1])
        if not np.any(act):
            return
        x[act, 2] = GAP0
        rel = x[act] - x_prev[act]
        rel_t = rel.copy()
        rel_t[:, 2] = 0.0
        lt = np.linalg.norm(rel_t, axis=1)
        f = np.minimum(1.0, MU * np.maximum(pen[act], 0.0) /
                       np.maximum(lt, 1e-12))
        x[act, :2] -= rel_t[:, :2] * f[:, None]

    def _self_collide(self, x: np.ndarray) -> None:
        d = x[self.pa] - x[self.pb]
        dist2 = np.einsum("ij,ij->i", d, d)
        act = dist2 < RC_SELF * RC_SELF
        if not np.any(act):
            return
        idx = np.where(act)[0]
        dd = d[idx]
        dist = np.linalg.norm(dd, axis=1)
        n = dd / np.maximum(dist, 1e-12)[:, None]
        push = 0.5 * (RC_SELF - dist)[:, None] * n
        np.add.at(x, self.pa[idx], push)
        np.add.at(x, self.pb[idx], -push)

    def _substep(self, h: float, t: float) -> None:
        x_prev = self.x.copy()
        v = self.v + self.gravity * h
        v *= max(0.0, 1.0 - DAMPING * h)
        x = x_prev + v * h

        self._solve_constraints(x, h)

        for (p1, p2, R, pivot, omega) in self._capsules(t):
            self._collide_capsule(x, x_prev, h, p1, p2, R, pivot, omega)
        self._collide_ground(x, x_prev, h)
        self._self_collide(x)

        v_new = (x - x_prev) / h
        speed = np.linalg.norm(v_new, axis=1)
        fast = speed > MAX_SPEED
        if np.any(fast):
            v_new[fast] *= (MAX_SPEED / speed[fast])[:, None]
        self.x = x
        self.v = v_new

    # --------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            target = self.t
        while self.t < target - 1e-12:
            h = min(SUBSTEP_H, target - self.t)
            self._substep(h, self.t)
            self.t += h
        return self._frame()

    # ------------------------------------------------------------- exports
    def _signed_contact_distance(self) -> np.ndarray:
        x = self.x
        best = None
        for (p1, p2, R, _pv, _om) in self._capsules(self.t):
            ab = p2 - p1
            ab2 = float(ab @ ab)
            ap = x - p1
            tt = np.clip((ap @ ab) / ab2, 0.0, 1.0)
            q = p1[None, :] + tt[:, None] * ab[None, :]
            d = np.linalg.norm(x - q, axis=1) - R
            best = d if best is None else np.minimum(best, d)
        best = np.minimum(best, _box_sdf(x))
        return best - 0.5 * THICKNESS

    def _deformation_gradient(self) -> np.ndarray:
        ni, nj = self.ni, self.nj
        X = self.x.reshape(nj, ni, 3)
        Rm = self.rest
        # deformed tangents (3x2 Jacobian columns) via finite differences
        Ju = np.gradient(X, self.u_ax, axis=1, edge_order=1)
        Jv = np.gradient(X, self.v_ax, axis=0, edge_order=1)
        # rest orthonormal tangent frame
        Tu = np.gradient(Rm, self.u_ax, axis=1, edge_order=1)
        Tv = np.gradient(Rm, self.v_ax, axis=0, edge_order=1)
        t1 = Tu / np.maximum(np.linalg.norm(Tu, axis=2, keepdims=True), 1e-12)
        t2 = Tv - np.sum(Tv * t1, axis=2, keepdims=True) * t1
        t2 = t2 / np.maximum(np.linalg.norm(t2, axis=2, keepdims=True), 1e-12)
        n = self.n
        Ju = Ju.reshape(n, 3); Jv = Jv.reshape(n, 3)
        t1 = t1.reshape(n, 3); t2 = t2.reshape(n, 3)
        F = np.empty((n, 2, 2))
        F[:, 0, 0] = np.sum(t1 * Ju, axis=1)
        F[:, 0, 1] = np.sum(t1 * Jv, axis=1)
        F[:, 1, 0] = np.sum(t2 * Ju, axis=1)
        F[:, 1, 1] = np.sum(t2 * Jv, axis=1)
        return F

    def _frame(self) -> StateFrame:
        n = self.n
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(self.t),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": self.boundary_vertex.copy(),
            },
            fields={
                "contact_signed_distance": self._signed_contact_distance(),
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
            },
            scalars={"contact_length_scale": float(THICKNESS)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "num_particles": int(self.n),
            "num_constraints": int(self.nc),
            "substep": float(SUBSTEP_H),
            "joint_angle": float(_theta(self.t)),
        }
