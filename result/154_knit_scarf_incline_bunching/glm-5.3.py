"""Solver: knitted scarf bunching on an incline (154_knit_scarf_incline_bunching).

Physical model
--------------
The scarf is an effective-knit thin sheet discretized on the frozen 17x17
grid (289 particles).  It is advanced with a real time-stepping scheme:
position-based dynamics (PBD) substepping of semi-implicit Euler, with

* asymmetric in-plane distance constraints -- knitted fabric resists
  extension much more strongly than compression, so compressed regions
  buckle instead of compressing (the source of the bunched folds),
* weak skip-neighbor bending constraints (knit compliance),
* signed-distance box contact (frozen ground + incline geometry) with
  Coulomb friction (static + kinetic) in the position solver,
* particle-particle self-contact so folds stack into a thick pile
  instead of self-penetrating.

No keyframes, animation tables or precomputed trajectories are used: the
state at every sample time is the outcome of the explicit time stepping
from the frozen initial condition.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------
# static collision geometry (frozen scene geometry, scene_geometry.yaml)
# --------------------------------------------------------------------------
def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


class _Box:
    """Static oriented box with analytic signed distance and normal."""

    def __init__(self, center, half_extents, quat=None):
        self.c = np.asarray(center, dtype=float)
        self.he = np.asarray(half_extents, dtype=float)
        self.R = np.eye(3) if quat is None else _quat_to_matrix(quat)

    def signed(self, p: np.ndarray, inflate: float):
        """Signed distance of points p to the box inflated by ``inflate``.

        Returns (sd, n_world): sd < 0 means penetration of the inflated
        box; n_world is the outward contact normal.
        """
        pl = (p - self.c) @ self.R                      # world -> local
        q = np.abs(pl) - (self.he + inflate)
        out = np.maximum(q, 0.0)
        d_out = np.sqrt((out * out).sum(axis=1))
        sd = d_out + np.minimum(q.max(axis=1), 0.0)
        sgn = np.where(pl >= 0.0, 1.0, -1.0)
        n_local = out * sgn
        ln = np.sqrt((n_local * n_local).sum(axis=1))
        bad = ln < 1e-12
        if np.any(bad):
            qb = q[bad]
            ax = np.argmax(qb, axis=1)
            rows = np.zeros((len(ax), 3))
            rows[np.arange(len(ax)), ax] = sgn[bad][np.arange(len(ax)), ax]
            n_local[bad] = rows
            ln = np.sqrt((n_local * n_local).sum(axis=1))
        n_local /= np.maximum(ln, 1e-12)[:, None]
        n_world = n_local @ self.R.T                    # local -> world
        return sd, n_world


class BenchmarkSolver:
    # --- numerical parameters -------------------------------------------
    SUBSTEPS_PER_SAMPLE = 20   # h = sample_dt / 20 = 1/600 s
    ITERS = 6                  # constraint-projection iterations / substep

    # effective knit stiffnesses (per-iteration position-correction gains)
    K_STRETCH = 0.8            # extension resistance (knit: moderate)
    K_COMPRESS = 0.02          # in-plane compression: very weak -> buckling
    K_SHEAR_STR = 0.15
    K_SHEAR_COM = 0.01
    K_BEND = 0.03              # bending (compressed skip springs only)

    MU_S = 0.55                # static friction (cloth vs ground / incline)
    MU_K = 0.45                # kinetic friction

    NEAR_CONTACT = 0.06        # [m] gap threshold for exported contacts

    # --------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        cloth = cfg["cloth"] if cfg and "cloth" in cfg else {}
        nx = int(cloth.get("grid_shape", [17, 17])[0])
        ny = int(cloth.get("grid_shape", [17, 17])[1])
        sx, sy = (float(v) for v in cloth.get("size", [3.0, 2.0]))
        self.thickness = float(cloth.get("thickness", 0.015))
        self.gravity = np.asarray(
            cfg.get("gravity", [0.0, 0.0, -9.81]) if cfg else [0.0, 0.0, -9.81],
            dtype=float,
        )

        # --- initial state (frozen scaffold pose) ------------------------
        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame0.mass, dtype=float)
        n = len(self.x)
        assert n == nx * ny, "grid shape does not match scaffold particles"

        rng = np.random.default_rng(int(seed))
        # tiny deterministic perturbation breaks mirror symmetry so folds
        # nucleate (physically: knit irregularity), no scripted motion.
        self.x += rng.normal(0.0, 3e-4, self.x.shape)

        # --- material (rest) coordinates ---------------------------------
        self.nx, self.ny = nx, ny
        self.du = sx / (nx - 1)
        self.dv = sy / (ny - 1)
        self.r_self = 0.04                     # self-contact particle radius
        idx = np.arange(n).reshape(ny, nx)     # ravel order of the scaffold
        self.idx = idx

        # rest in-plane axes (from the untouched scaffold pose)
        pos = self.x.reshape(ny, nx, 3)
        t1 = (pos[:, -1, :] - pos[:, 0, :]).mean(axis=0)
        t2 = (pos[-1, :, :] - pos[0, :, :]).mean(axis=0)
        self.t1 = t1 / np.linalg.norm(t1)
        self.t2 = t2 / np.linalg.norm(t2)

        # --- constraint topology -----------------------------------------
        # structural along material u
        self.c_stx = (idx[:, :-1].ravel(), idx[:, 1:].ravel(), self.du)
        # structural along material v
        self.c_sty = (idx[:-1, :].ravel(), idx[1:, :].ravel(), self.dv)
        # shear (both diagonals)
        d_len = float(np.hypot(self.du, self.dv))
        self.c_sh1 = (idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), d_len)
        self.c_sh2 = (idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), d_len)
        # bending (skip-one neighbors)
        self.c_bx = (idx[:, :-2].ravel(), idx[:, 2:].ravel(), 2.0 * self.du)
        self.c_by = (idx[:-2, :].ravel(), idx[2:, :].ravel(), 2.0 * self.dv)

        # --- static collision boxes (frozen geometry) --------------------
        self.boxes = [
            _Box([0.0, 0.0, -0.05], [2.4, 2.0, 0.05]),
            _Box([0.0, 0.0, 0.8], [2.2, 1.1, 0.08],
                 quat=[0.0, 0.300706, 0.0, 0.953717]),
        ]

        # --- time stepping ------------------------------------------------
        sample_dt = 1.0 / 30.0
        try:
            times = scene.sample_times()
            sample_dt = float(times[1] - times[0]) if len(times) > 1 else 1.0 / 30.0
        except Exception:
            pass
        self.h = sample_dt / self.SUBSTEPS_PER_SAMPLE
        self.damp = float(np.exp(-0.05 * self.h))
        self.t = 0.0

    # ------------------------------------------------------------------
    def _distance_group(self, x, a, b, rest, k_str, k_com, bend_only=False):
        d = x[b] - x[a]
        L = np.sqrt((d * d).sum(axis=1))
        np.maximum(L, 1e-9, out=L)
        dirn = d / L[:, None]
        C = L - rest
        if bend_only:
            k = np.where(C < 0.0, k_str, 0.0)
        else:
            k = np.where(C > 0.0, k_str, k_com)
        corr = (0.5 * k * C)[:, None] * dirn
        x[a] += corr          # a-index unique inside a group
        x[b] -= corr          # b-index unique inside a group

    def _contacts(self, x, x0):
        for box in self.boxes:
            sd, n = box.signed(x, self.thickness)
            pen = sd < 0.0
            if not np.any(pen):
                continue
            xn = x[pen]
            nn = n[pen]
            depth = -sd[pen]                          # > 0
            x[pen] = xn + depth[:, None] * nn         # normal projection
            # Coulomb friction on the tangential displacement this substep
            d = x[pen] - x0[pen]
            dn = (d * nn).sum(axis=1)
            dt_ = d - dn[:, None] * nn
            lt = np.sqrt((dt_ * dt_).sum(axis=1))
            limit = self.MU_S * depth
            scale = np.where(
                lt <= limit,
                1.0,                                          # static: stick
                np.minimum(1.0, self.MU_K * depth / np.maximum(lt, 1e-12)),
            )
            x[pen] -= scale[:, None] * dt_

    def _self_contact(self, x):
        n = len(x)
        diff = x[:, None, :] - x[None, :, :]
        d2 = (diff * diff).sum(axis=-1)
        mind = 2.0 * self.r_self
        mask = (d2 < mind * mind) & (d2 > 1e-12)
        ii, jj = np.nonzero(np.triu(mask, 1))
        if ii.size == 0:
            return
        dd = np.sqrt(d2[ii, jj])
        dirn = (x[ii] - x[jj]) / dd[:, None]
        push = (0.5 * (mind - dd))[:, None] * dirn
        np.add.at(x, ii, push)
        np.add.at(x, jj, -push)

    def _substep(self, h):
        x0 = self.x.copy()
        v = self.v + self.gravity * h
        v *= self.damp
        x = x0 + v * h
        for _ in range(self.ITERS):
            self._distance_group(x, *self.c_stx, self.K_STRETCH, self.K_COMPRESS)
            self._distance_group(x, *self.c_sty, self.K_STRETCH, self.K_COMPRESS)
            self._distance_group(x, *self.c_sh1, self.K_SHEAR_STR, self.K_SHEAR_COM)
            self._distance_group(x, *self.c_sh2, self.K_SHEAR_STR, self.K_SHEAR_COM)
            self._distance_group(x, *self.c_bx, self.K_BEND, 0.0, bend_only=True)
            self._distance_group(x, *self.c_by, self.K_BEND, 0.0, bend_only=True)
            self._contacts(x, x0)
            self._self_contact(x)
        self.v = (x - self.x) / h
        np.clip(self.v, -40.0, 40.0, out=self.v)
        if not np.isfinite(self.v).all():
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.x = x
        self.t += h

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target > self.t:
            guard = int(np.ceil((target - self.t) / self.h)) + 4
            while target - self.t > 1e-9 and guard > 0:
                self._substep(min(self.h, target - self.t))
                guard -= 1
            self.t = target
        return self._frame(max(self.t, 0.0))

    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        """In-plane surface deformation gradient F (2x2) per sample.

        F maps material coordinates (u, v) onto the sheet's rest tangent
        frame; F = I in the undeformed configuration.
        """
        pos = self.x.reshape(self.ny, self.nx, 3)
        e1 = np.empty_like(pos)
        e1[:, :-1] = pos[:, 1:] - pos[:, :-1]
        e1[:, -1] = pos[:, -1] - pos[:, -2]
        e2 = np.empty_like(pos)
        e2[:-1] = pos[1:] - pos[:-1]
        e2[-1] = pos[-1] - pos[-2]
        e1 = e1.reshape(-1, 3)
        e2 = e2.reshape(-1, 3)
        n = len(e1)
        F = np.zeros((n, 2, 2))
        F[:, 0, 0] = (e1 @ self.t1) / self.du
        F[:, 1, 0] = (e1 @ self.t2) / self.du
        F[:, 0, 1] = (e2 @ self.t1) / self.dv
        F[:, 1, 1] = (e2 @ self.t2) / self.dv
        return F

    def _frame(self, t: float) -> StateFrame:
        n = len(self.x)
        gap = None
        for box in self.boxes:
            sd, _ = box.signed(self.x, self.thickness)
            gap = sd if gap is None else np.minimum(gap, sd)
        near = gap < self.NEAR_CONTACT
        csd = np.ascontiguousarray(gap[near])
        ext_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(t),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext_force,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "method": "PBD effective-knit cloth (asymmetric stretch/compression)",
            "substep": float(self.h),
            "particles": int(len(self.x)),
        }
