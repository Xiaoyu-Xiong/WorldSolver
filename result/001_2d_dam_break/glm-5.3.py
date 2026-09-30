"""Position-Based Fluids solver for the 2D dam-break scene.

Implements a real fluid time-stepping scheme (Macklin & Mueller 2013,
"Position Based Fluids"): gravity + density-projection iterations with
static boundary particles for the tank walls/floor, XSPH viscosity for
dissipation, and clamped domain constraints for robustness.  All state is
advanced by the numerical scheme itself; nothing is keyframed.

The 2D liquid lives in the x-z plane (y == 0).  Frozen tank geometry:
inner faces at x = -3.0 / +3.0, floor top at z = 0.05, open air above.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree

try:  # pragma: no cover - exercised by the platform runner
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None

GRID_NX = 101
GRID_NZ = 51
LIQUID_PHASE = 0.0


class BenchmarkSolver:
    """2D dam-break fluid solver using Position-Based Fluids (PBF)."""

    def __init__(self) -> None:
        self.ready = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.n = int(self.pos.shape[0])
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.m0 = float(np.mean(self.mass))
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.seed = int(seed)
        self.t = 0.0
        self.substeps = 0

        self.xmin, self.xmax, self.zfloor, self.ztop = self._tank_bounds(cfg)

        xs = np.unique(np.round(self.pos[:, 0], 9))
        zs = np.unique(np.round(self.pos[:, 2], 9))
        dxs = np.diff(xs) if xs.size > 1 else np.array([0.06])
        dzs = np.diff(zs) if zs.size > 1 else np.array([0.06])
        self.dx = float(min(dxs.min(), dzs.min()))
        self.h = 2.5 * self.dx

        # SPH kernels (2D)
        self.k_poly = 4.0 / (math.pi * self.h ** 8)
        self.k_spiky = 30.0 / (math.pi * self.h ** 5)
        self.w0 = self.k_poly * (self.h ** 2) ** 3

        # characteristic scales (frozen by the initial condition)
        self.characteristic_length = float(
            self.pos[:, 0].max() - self.pos[:, 0].min() + self.dx
        )
        height = float(self.pos[:, 2].max() - self.pos[:, 2].min() + self.dx)
        self.reference_velocity = float(
            math.sqrt(max(np.linalg.norm(self.gravity), 1e-9) * height)
        )

        # static boundary particles for floor and side walls
        self.bnd = self._make_boundary()

        # rest density measured on the initial lattice (surface particles
        # are lighter; use a bulk percentile)
        rho_init = self._density(self.pos)
        self.rho0 = float(np.percentile(rho_init, 70.0))

        # numerics
        self.dt_sub = 0.004
        self.iterations = 4
        self.eps = 1e-3
        self.visc = 1.2
        self.max_speed = 15.0
        self.margin = 0.02 * self.dx / 0.06

        # diagnostic grid
        self.grid_x = np.linspace(self.xmin, self.xmax, GRID_NX)
        self.grid_z = np.linspace(self.zfloor, self.zfloor + 3.0, GRID_NZ)
        self.grid_dx = float(self.grid_x[1] - self.grid_x[0])
        self.grid_dz = float(self.grid_z[1] - self.grid_z[0])
        self.grid_spacing = self.grid_dx

        # wall samples (floor + both side walls), 2D normals in (x, z)
        fx = np.linspace(self.xmin + 0.05, self.xmax - 0.05, 30)
        wz = np.linspace(self.zfloor + 0.10, self.ztop - 0.10, 24)
        floor_pts = np.column_stack((fx, np.full(fx.size, self.zfloor)))
        left_pts = np.column_stack((np.full(wz.size, self.xmin), wz))
        right_pts = np.column_stack((np.full(wz.size, self.xmax), wz))
        self.wall_pts = np.vstack((floor_pts, left_pts, right_pts))
        normals = np.vstack(
            (
                np.tile(np.array([0.0, 1.0]), (fx.size, 1)),
                np.tile(np.array([1.0, 0.0]), (wz.size, 1)),
                np.tile(np.array([-1.0, 0.0]), (wz.size, 1)),
            )
        )
        self.wall_normals = normals

        self.ready = True

    def _tank_bounds(self, cfg):
        xmin, xmax, zfloor, ztop = -3.0, 3.0, 0.05, 3.0
        geom = cfg.get("scene_geometry") or {}
        for obj in geom.get("objects", []):
            oid = obj.get("id")
            geometry = obj.get("geometry") or {}
            if geometry.get("type") != "box":
                continue
            he = geometry.get("half_extents") or []
            tr = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
            if len(he) < 3 or len(tr) < 3:
                continue
            if oid == "tank_floor":
                zfloor = float(tr[2] + he[2])
            elif oid == "tank_left_wall":
                xmin = float(tr[0] + he[0])
                ztop = max(ztop, float(tr[2] + he[2]))
            elif oid == "tank_right_wall":
                xmax = float(tr[0] - he[0])
                ztop = max(ztop, float(tr[2] + he[2]))
        return xmin, xmax, zfloor, ztop

    def _make_boundary(self) -> np.ndarray:
        sp = self.dx
        layers = 3
        pts = []
        # floor slabs below the floor face
        for layer in range(layers):
            z = self.zfloor - (0.5 + layer) * sp
            x0 = self.xmin - (layers - 1) * sp
            x1 = self.xmax + (layers - 1) * sp
            xs = np.arange(x0, x1 + 0.5 * sp, sp)
            pts.append(np.column_stack((xs, np.full(xs.size, z))))
        # side wall columns inside the walls
        z0 = self.zfloor - (layers - 1) * sp
        z1 = self.ztop + sp
        zs = np.arange(z0, z1 + 0.5 * sp, sp)
        for side in (-1.0, 1.0):
            for layer in range(layers):
                x = self.xmin + side * (0.5 + layer) * sp if side < 0 else self.xmax + (0.5 + layer) * sp
                pts.append(np.column_stack((np.full(zs.size, x), zs)))
        bnd2 = np.vstack(pts)
        # deduplicate for determinism and speed
        bnd2 = np.unique(np.round(bnd2, 9), axis=0)
        bnd = np.zeros((bnd2.shape[0], 3), dtype=np.float64)
        bnd[:, 0] = bnd2[:, 0]
        bnd[:, 2] = bnd2[:, 1]
        return bnd

    # ------------------------------------------------------------------
    # SPH helpers
    # ------------------------------------------------------------------
    def _pairs(self, pos: np.ndarray):
        allp = np.vstack((pos, self.bnd))
        tree = cKDTree(allp)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        n = self.n
        if pairs.size == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, e, e
        i = pairs[:, 0]
        j = pairs[:, 1]
        lo = np.minimum(i, j)
        hi = np.maximum(i, j)
        ff = hi < n
        fb = (lo < n) & (hi >= n)
        return i[ff], j[ff], lo[fb], hi[fb]

    def _density(self, pos: np.ndarray) -> np.ndarray:
        fi, fj, fbi, fbj = self._pairs(pos)
        rho = np.full(self.n, self.m0 * self.w0, dtype=np.float64)
        if fi.size:
            r = np.linalg.norm(pos[fi] - pos[fj], axis=1)
            w = self.k_poly * np.maximum(self.h ** 2 - r ** 2, 0.0) ** 3
            rho += self.m0 * (
                np.bincount(fi, w, minlength=self.n)
                + np.bincount(fj, w, minlength=self.n)
            )
        if fbi.size:
            rb = np.linalg.norm(pos[fbi] - self.bnd[fbj], axis=1)
            wb = self.k_poly * np.maximum(self.h ** 2 - rb ** 2, 0.0) ** 3
            rho += self.m0 * np.bincount(fbi, wb, minlength=self.n)
        return rho

    def _clamp_domain(self, pos: np.ndarray) -> np.ndarray:
        pos[:, 0] = np.clip(pos[:, 0], self.xmin + self.margin, self.xmax - self.margin)
        pos[:, 2] = np.clip(pos[:, 2], self.zfloor + self.margin, self.ztop + 1.5)
        pos[:, 1] = 0.0
        return pos

    # ------------------------------------------------------------------
    # PBF projection
    # ------------------------------------------------------------------
    def _project(self, pred: np.ndarray) -> np.ndarray:
        n = self.n
        h = self.h
        m = self.m0
        inv = 1.0 / self.rho0

        fi, fj, fbi, fbj = self._pairs(pred)
        delta = np.zeros_like(pred)

        g = w = gsq = None
        if fi.size:
            d = pred[fi] - pred[fj]
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            t = np.maximum(h - r, 0.0)
            g = -self.k_spiky * t * t * (d / r[:, None])  # grad wrt x_i
            w = self.k_poly * np.maximum(h * h - r * r, 0.0) ** 3
            gsq = np.einsum("ij,ij->i", g, g)
        gb = wb = gbsq = None
        if fbi.size:
            db = pred[fbi] - self.bnd[fbj]
            rb = np.maximum(np.linalg.norm(db, axis=1), 1e-9)
            tb = np.maximum(h - rb, 0.0)
            gb = -self.k_spiky * tb * tb * (db / rb[:, None])
            wb = self.k_poly * np.maximum(h * h - rb * rb, 0.0) ** 3
            gbsq = np.einsum("ij,ij->i", gb, gb)

        # density
        rho = np.full(n, self.m0 * self.w0, dtype=np.float64)
        if fi.size:
            rho += m * (np.bincount(fi, w, minlength=n) + np.bincount(fj, w, minlength=n))
        if fbi.size:
            rho += m * np.bincount(fbi, wb, minlength=n)
        constraint = np.maximum(rho * inv - 1.0, 0.0)

        # constraint gradients
        sum_g = np.zeros((n, 3), dtype=np.float64)
        sum_g2 = np.zeros(n, dtype=np.float64)
        if fi.size:
            for k in range(3):
                sum_g[:, k] += np.bincount(fi, g[:, k], minlength=n)
                sum_g[:, k] -= np.bincount(fj, g[:, k], minlength=n)
            sum_g2 += np.bincount(fi, gsq, minlength=n)
            sum_g2 += np.bincount(fj, gsq, minlength=n)
        if fbi.size:
            for k in range(3):
                sum_g[:, k] += np.bincount(fbi, gb[:, k], minlength=n)
            sum_g2 += np.bincount(fbi, gbsq, minlength=n)

        denom = (m * inv) ** 2 * (
            np.einsum("ij,ij->i", sum_g, sum_g) + sum_g2
        ) + self.eps
        lam = -constraint / denom  # <= 0

        # position correction
        fac = m * inv
        if fi.size:
            lam_sum = lam[fi] + lam[fj]
            for k in range(3):
                contrib = lam_sum * g[:, k]
                delta[:, k] += fac * (
                    np.bincount(fi, contrib, minlength=n)
                    - np.bincount(fj, contrib, minlength=n)
                )
        if fbi.size:
            lam_b = lam[fbi]
            for k in range(3):
                delta[:, k] += fac * np.bincount(
                    fbi, lam_b * gb[:, k], minlength=n
                )

        # safety clamp per particle
        dl = np.linalg.norm(delta, axis=1)
        scale = np.ones(n)
        act = dl > 1e-12
        scale[act] = np.minimum(1.0, 0.25 * h / dl[act])
        pred = pred + delta * scale[:, None]
        return self._clamp_domain(pred)

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        x_old = self.pos.copy()
        v = self.vel + self.gravity * dt
        v[:, 1] = 0.0
        pred = self._clamp_domain(x_old + v * dt)

        for _ in range(self.iterations):
            pred = self._project(pred)

        new_vel = (pred - x_old) / dt
        speed = np.linalg.norm(new_vel, axis=1)
        fast = speed > self.max_speed
        if np.any(fast):
            new_vel[fast] *= (self.max_speed / speed[fast])[:, None]
        self.pos = pred
        self.vel = new_vel

        # XSPH-style shear damping (momentum preserving on fluid pairs)
        fi, fj, fbi, fbj = self._pairs(self.pos)
        coef = self.visc * dt * self.m0 / self.rho0
        dv = np.zeros_like(self.vel)
        if fi.size:
            d = self.pos[fi] - self.pos[fj]
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            w = self.k_poly * np.maximum(self.h ** 2 - r ** 2, 0.0) ** 3
            for k in range(3):
                rel = (self.vel[fj, k] - self.vel[fi, k]) * w
                dv[:, k] += coef * (
                    np.bincount(fi, rel, minlength=self.n)
                    + np.bincount(fj, -rel, minlength=self.n)
                )
        if fbi.size:
            db = self.pos[fbi] - self.bnd[fbj]
            rb = np.maximum(np.linalg.norm(db, axis=1), 1e-9)
            wb = self.k_poly * np.maximum(self.h ** 2 - rb ** 2, 0.0) ** 3
            for k in range(3):
                rel = -self.vel[fbi, k] * wb  # walls at rest
                dv[:, k] += 0.5 * coef * np.bincount(fbi, rel, minlength=self.n)
        self.vel = self.vel + dv
        self.vel[:, 1] = 0.0
        speed = np.linalg.norm(self.vel, axis=1)
        fast = speed > self.max_speed
        if np.any(fast):
            self.vel[fast] *= (self.max_speed / speed[fast])[:, None]
        self.substeps += 1

    def advance_to(self, time: float):
        t = float(time)
        if not self.ready:
            raise RuntimeError("solver is not initialized")
        if t < self.t - 1e-9:
            raise ValueError(f"non-monotonic time request: {t} < {self.t}")
        while self.t < t - 1e-12:
            dt = min(self.dt_sub, t - self.t)
            self._substep(dt)
            self.t += dt
            if abs(self.t - t) <= 1e-12:
                self.t = t
        self.t = t
        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            raise RuntimeError("solver state became non-finite")
        return self._frame(t)

    # ------------------------------------------------------------------
    # StateFrame export
    # ------------------------------------------------------------------
    def _interp_velocity_2d(self, tree: cKDTree, pts: np.ndarray, radius: float):
        k = 12
        dist, idx = tree.query(pts, k=k, distance_upper_bound=radius)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        valid = np.isfinite(dist) & (idx < self.n)
        idxc = np.minimum(idx, self.n - 1)
        w = self.k_poly * np.maximum(self.h ** 2 - dist ** 2, 0.0) * valid
        wsum = w.sum(axis=1)
        vel2 = np.zeros((pts.shape[0], 2), dtype=np.float64)
        for comp, axis in ((0, 0), (1, 2)):
            num = (np.where(valid, self.vel[idxc, axis], 0.0) * w).sum(axis=1)
            good = wsum > 1e-12
            vel2[good, comp] = num[good] / wsum[good]
        return vel2, wsum

    def _frame(self, t: float):
        pos = self.pos
        vel = self.vel
        tree = cKDTree(pos[:, [0, 2]])

        gx, gz = self.grid_x, self.grid_z
        xx, zz = np.meshgrid(gx, gz, indexing="ij")
        nodes = np.column_stack((xx.ravel(), zz.ravel()))
        shape = (gx.size, gz.size)

        radius = 3.0 * self.dx
        vel2, wsum = self._interp_velocity_2d(tree, nodes, radius)
        coverage = (wsum * self.m0 / self.rho0).reshape(shape)
        mask = coverage > 0.5
        mask = np.asarray(binary_fill_holes(mask), dtype=bool)

        inside = distance_transform_edt(mask, sampling=(self.grid_dx, self.grid_dz))
        outside = distance_transform_edt(~mask, sampling=(self.grid_dx, self.grid_dz))
        level_set = np.where(mask, inside, -outside).astype(np.float64)

        velocity_field = np.zeros((shape[0], shape[1], 2), dtype=np.float64)
        velocity_field[..., 0] = vel2[:, 0].reshape(shape)
        velocity_field[..., 1] = vel2[:, 1].reshape(shape)

        wall_vel, _ = self._interp_velocity_2d(tree, self.wall_pts, radius)
        b = self.wall_pts.shape[0]
        boundary_normal = np.array(self.wall_normals, dtype=np.float64)
        boundary_target = np.zeros((b, 2), dtype=np.float64)

        attributes = {
            "phase_id": np.full(self.n, LIQUID_PHASE, dtype=np.float64),
            "sample_volume": np.full(self.n, self.m0 / 1000.0, dtype=np.float64),
        }
        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": wall_vel,
            "liquid_interior_mask": mask.astype(np.float64),
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
        }
        scalars = {
            "characteristic_length": float(self.characteristic_length),
            "grid_spacing": float(self.grid_spacing),
            "liquid_phase_id": float(LIQUID_PHASE),
            "reference_velocity": float(self.reference_velocity),
            "density": 1000.0,
        }

        if StateFrame is not None:
            frame = StateFrame(
                time=float(t),
                position=pos.copy(),
                velocity=vel.copy(),
                mass=self.mass.copy(),
                attributes=attributes,
                fields=fields,
                scalars=scalars,
            )
            frame.validate()
            return frame
        return {
            "time": float(t),
            "position": pos.copy(),
            "velocity": vel.copy(),
            "mass": self.mass.copy(),
            "attributes": attributes,
            "fields": fields,
            "scalars": scalars,
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "boundary_particles": int(self.bnd.shape[0]),
            "rest_density": float(getattr(self, "rho0", 0.0)),
            "smoothing_length": float(getattr(self, "h", 0.0)),
            "substep": float(getattr(self, "dt_sub", 0.0)),
            "substeps_taken": int(getattr(self, "substeps", 0)),
            "final_time": float(getattr(self, "t", 0.0)),
        }
