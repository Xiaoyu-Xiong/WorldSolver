"""Position Based Fluids solver for the liquid bunny pool impact scene.

The solver discretizes the frozen scene into SPH particles:

* the receiving pool keeps the frozen block sampling (spacing 0.14 m),
* the bunny-shaped liquid volume is re-discretized as a solid voxel fill of
  the frozen bunny surface cloud at a finer spacing so that the silhouette
  is preserved while the particle count stays tractable,

and advances the whole liquid with a Position Based Fluids (Macklin &
Mueller, 2013) scheme: predicted positions under gravity, iterative density
constraint projection with a poly6 density kernel / spiky gradients,
XSPH viscosity, and rigid box boundary projection for the basin and ground.
All diagnostics required by the trajectory contract (level set, velocity
grid, active mask, boundary samples, mass bookkeeping) are derived from the
physical particle state at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based-fluid solver for 135_liquid_bunny_pool_impact."""

    SUB_DT = 1.0 / 180.0     # physical substep
    ITERS = 3                # density constraint iterations per substep
    XSPH = 0.06              # XSPH viscosity coefficient
    VMAX = 12.0              # velocity clamp (m/s)
    BUNNY_SPACING = 0.045    # re-discretization spacing for the bunny volume

    # Rigid boundary boxes (center, half extents), frozen scene geometry.
    BOXES = (
        (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05])),      # ground
        (np.array([0.0, 0.92, 0.3]), np.array([1.85, 0.08, 0.34])),     # basin back
        (np.array([-1.85, 0.0, 0.3]), np.array([0.08, 0.92, 0.34])),    # basin left
        (np.array([1.85, 0.0, 0.3]), np.array([0.08, 0.92, 0.34])),     # basin right
    )
    WALL_MARGIN = 0.02       # collision shell around the rigid boxes

    # Diagnostic grid (frozen pool particle domain).
    GRID_MIN = np.array([-3.0, -1.2, 0.0])
    GRID_MAX = np.array([3.0, 1.2, 3.8])
    GRID_SHAPE = (24, 16, 28)

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        sampling = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
        gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.gravity = gravity

        self.spacing = float(sampling.get("spacing", 0.14))
        self.rho0 = float(sampling.get("density", 1000.0))
        self.h = 1.5 * self.spacing
        self.seed = int(seed)

        # SPH kernel constants (poly6 for density, spiky for gradients).
        h = self.h
        self.k6 = 315.0 / (64.0 * np.pi * h ** 9)
        self.cs = 45.0 / (np.pi * h ** 6)
        self.w0 = self.k6 * h ** 6

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, float)
        labels0 = np.asarray(frame0.attributes["material_label"])
        pool_pts = pos0[labels0 == 0]
        bunny_surf = pos0[labels0 == 1]

        # --- particle setup -------------------------------------------------
        s_b = self.BUNNY_SPACING
        bunny_pts = self._voxelize(bunny_surf, s_b)
        if len(bunny_pts) < 40:
            bunny_pts = self._dedupe(bunny_surf, s_b)

        # Rest-consistent masses: m(s) = rho0 / S(s) with S(s) the kernel sum
        # over an infinite lattice of spacing s, so every lattice is at rest.
        s_pool = self.spacing
        s_list = np.array([s_pool, s_b])
        m_unit = np.array([self.rho0 / self._lattice_sum(s) for s in s_list])
        self.mass = np.concatenate([np.full(len(pool_pts), m_unit[0]),
                                    np.full(len(bunny_pts), m_unit[1])])
        self.pos = np.concatenate([pool_pts, bunny_pts])
        self.vel = np.zeros_like(self.pos)
        self.labels = np.concatenate([np.zeros(len(pool_pts), np.int32),
                                      np.ones(len(bunny_pts), np.int32)])
        self.radius = 0.45 * np.where(self.labels == 0, s_pool, s_b)
        self.n = len(self.pos)
        self.t = 0.0

        # --- diagnostics grid ----------------------------------------------
        gx = self.GRID_MIN[0] + (np.arange(self.GRID_SHAPE[0]) + 0.5) * \
            (self.GRID_MAX[0] - self.GRID_MIN[0]) / self.GRID_SHAPE[0]
        gy = self.GRID_MIN[1] + (np.arange(self.GRID_SHAPE[1]) + 0.5) * \
            (self.GRID_MAX[1] - self.GRID_MIN[1]) / self.GRID_SHAPE[1]
        gz = self.GRID_MIN[2] + (np.arange(self.GRID_SHAPE[2]) + 0.5) * \
            (self.GRID_MAX[2] - self.GRID_MIN[2]) / self.GRID_SHAPE[2]
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        self.grid_pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        # --- boundary samples ----------------------------------------------
        samples, normals = self._boundary_samples()
        self.b_samples = samples
        self.b_normals = normals

        # Frozen normalization scalars.
        self.characteristic_length = max(float(np.ptp(self.pos, axis=0).max()),
                                         self.spacing)
        self.grid_spacing = self.spacing
        drop = max(0.1, float(bunny_pts[:, 2].min() - 0.45))
        self.reference_velocity = float(np.sqrt(2.0 * 9.81 * drop))

    # ------------------------------------------------------------------ init

    def _lattice_sum(self, s: float) -> float:
        """Kernel sum over an infinite lattice with spacing ``s``."""
        k = int(np.ceil(self.h / s))
        rng = np.arange(-k, k + 1) * s
        DX, DY, DZ = np.meshgrid(rng, rng, rng, indexing="ij")
        r2 = DX ** 2 + DY ** 2 + DZ ** 2
        m = r2 <= self.h ** 2
        w = self.k6 * np.maximum(self.h ** 2 - r2, 0.0) ** 3
        return float(w[m].sum())

    @staticmethod
    def _voxelize(surface: np.ndarray, s: float, thr_factor: float = 0.9):
        """Solid voxel fill of the surface point cloud (flood fill outside)."""
        tree = cKDTree(surface)
        lo = surface.min(0) - s
        hi = surface.max(0) + s
        n = np.ceil((hi - lo) / s).astype(int) + 1
        axes = [lo[i] + 0.5 * s + np.arange(n[i]) * s for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        centers = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        d, _ = tree.query(centers)
        d = d.reshape(n)
        near = d < thr_factor * s
        lab, _ = ndimage.label(~near)
        border = np.unique(np.concatenate([
            lab[0].ravel(), lab[-1].ravel(),
            lab[:, 0].ravel(), lab[:, -1].ravel(),
            lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        border = border[border != 0]
        inside = (~near) & (~np.isin(lab, border))
        return centers[inside.ravel()]

    @staticmethod
    def _dedupe(points: np.ndarray, s: float) -> np.ndarray:
        """Fallback: keep one point per s-sized voxel (surface shell)."""
        keys = np.floor(points / s).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(idx)]

    def _boundary_samples(self):
        pts, nrm = [], []

        def add(P, N):
            pts.append(np.asarray(P, float))
            nrm.append(np.tile(np.asarray(N, float), (len(P), 1)))

        # basin back wall inner face (y = 0.84)
        xs = np.linspace(-1.68, 1.68, 9)
        zs = np.linspace(0.08, 0.56, 4)
        P = np.stack(np.meshgrid(xs, zs, indexing="ij"), -1).reshape(-1, 2)
        add(np.column_stack((P[:, 0], np.full(len(P), 0.84), P[:, 1])), (0, -1, 0))
        # basin left wall inner face (x = -1.77)
        ys = np.linspace(-0.78, 0.78, 7)
        P = np.stack(np.meshgrid(ys, zs, indexing="ij"), -1).reshape(-1, 2)
        add(np.column_stack((np.full(len(P), -1.77), P[:, 0], P[:, 1])), (1, 0, 0))
        # basin right wall inner face (x = 1.77)
        add(np.column_stack((np.full(len(P), 1.77), P[:, 0], P[:, 1])), (-1, 0, 0))
        # ground (z = 0)
        xs = np.linspace(-1.8, 1.8, 13)
        ys = np.linspace(-1.2, 1.1, 9)
        P = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        add(np.column_stack((P[:, 0], P[:, 1], np.zeros(len(P)))), (0, 0, 1))
        return np.concatenate(pts), np.concatenate(nrm)

    # ---------------------------------------------------------------- physics

    def _pairs(self, p: np.ndarray):
        """Half-pair (i < j) neighbor indices within the smoothing radius."""
        h = self.h
        n = len(p)
        mn = p.min(0)
        ijk = np.floor((p - mn) / h).astype(np.int64)
        dims = ijk.max(0) + 1
        key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        order = np.argsort(key, kind="stable")
        skey = key[order]
        uniq, first, counts = np.unique(skey, return_index=True, return_counts=True)
        own = np.searchsorted(uniq, key)
        own = np.clip(own, 0, len(uniq) - 1)

        cnt_parts, start_parts = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        cnt_parts.append(counts[own].astype(np.int64))
                        start_parts.append(first[own].astype(np.int64))
                        continue
                    nijk = ijk + np.array([dx, dy, dz])
                    ok = np.all((nijk >= 0) & (nijk < dims), axis=1)
                    nkey = (nijk[:, 0] * dims[1] + nijk[:, 1]) * dims[2] + nijk[:, 2]
                    pos = np.searchsorted(uniq, nkey)
                    pos_c = np.clip(pos, 0, len(uniq) - 1)
                    ok &= (pos < len(uniq)) & (uniq[pos_c] == nkey)
                    cnt_parts.append(np.where(ok, counts[pos_c], 0).astype(np.int64))
                    start_parts.append(np.where(ok, first[pos_c], 0).astype(np.int64))

        cnt = np.concatenate(cnt_parts)
        start = np.concatenate(start_parts)
        total = int(cnt.sum())
        if total == 0:
            e = np.empty(0, np.int64)
            return e, e
        entry = np.repeat(np.arange(n, dtype=np.int64), cnt)
        s_arr = np.repeat(start, cnt)
        cum = np.concatenate(([0], np.cumsum(cnt)))[:-1]
        local = np.arange(total, dtype=np.int64) - np.repeat(cum, cnt)
        j = order[s_arr + local]
        i = entry
        keep = i != j
        i, j = i[keep], j[keep]
        d = p[i] - p[j]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = (r2 < h * h) & (i < j)
        return i[keep], j[keep]

    def _project_boxes(self, p: np.ndarray) -> np.ndarray:
        """Project positions out of the inflated rigid boxes."""
        r = self.WALL_MARGIN
        for c, he in self.BOXES:
            d = p - c
            q = np.abs(d) - (he + r)
            inside = q < 0
            hit = inside.all(1)
            if not hit.any():
                continue
            ax = np.argmax(q[hit], axis=1)
            idx = np.where(hit)[0]
            dd = d[idx]
            sign = np.sign(dd[np.arange(len(idx)), ax])
            sign[sign == 0] = 1.0
            p[idx, ax] = c[ax] + sign * (he[ax] + r)
        return p

    def _wall_velocity(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Remove inward normal velocity and apply wall friction on contact."""
        r = self.WALL_MARGIN
        for c, he in self.BOXES:
            d = p - c
            q = np.abs(d) - (he + r)
            hit = (q < 0).all(1)
            if not hit.any():
                continue
            idx = np.where(hit)[0]
            ax = np.argmax(q[idx], axis=1)
            rows = np.arange(len(idx))
            s = np.sign(d[idx, ax])
            s[s == 0] = 1.0
            sub = v[idx]
            vn = sub[rows, ax] * s
            sub[rows, ax] = np.where(vn < 0.0, 0.0, sub[rows, ax])
            tang = np.ones((len(idx), 3), dtype=bool)
            tang[rows, ax] = False
            sub[tang] *= 0.99
            v[idx] = sub
        return v

    def _step(self, dt: float) -> None:
        h, rho0 = self.h, self.rho0
        v = self.vel + self.gravity * dt
        speed = np.linalg.norm(v, axis=1)
        v *= np.minimum(1.0, self.VMAX / np.maximum(speed, 1e-12))[:, None]
        pred = self.pos + v * dt
        pred = self._project_boxes(pred)

        mass = self.mass
        pairs_i = pairs_j = None
        for _ in range(self.ITERS):
            i, j = self._pairs(pred)
            pairs_i, pairs_j = i, j
            if len(i) == 0:
                break
            d = pred[i] - pred[j]
            r2 = np.einsum("ij,ij->i", d, d)
            r = np.sqrt(np.maximum(r2, 1e-12))
            w = self.k6 * (h * h - r2) ** 3
            gmag = -self.cs * (h - r) ** 2
            gx = gmag * d[:, 0] / r
            gy = gmag * d[:, 1] / r
            gz = gmag * d[:, 2] / r
            mi, mj = mass[i], mass[j]

            # densities
            rho = mass * self.w0 \
                + np.bincount(i, mj * w, minlength=self.n) \
                + np.bincount(j, mi * w, minlength=self.n)
            c = rho / rho0 - 1.0

            # gradient sums
            ax_ = np.bincount(i, mj * gx, minlength=self.n) \
                - np.bincount(j, mi * gx, minlength=self.n)
            ay_ = np.bincount(i, mj * gy, minlength=self.n) \
                - np.bincount(j, mi * gy, minlength=self.n)
            az_ = np.bincount(i, mj * gz, minlength=self.n) \
                - np.bincount(j, mi * gz, minlength=self.n)
            gsq = gmag ** 2
            ssq = (ax_ ** 2 + ay_ ** 2 + az_ ** 2) \
                + np.bincount(i, (mi * mi) * gsq, minlength=self.n) \
                + np.bincount(j, (mj * mj) * gsq, minlength=self.n)
            ssq /= rho0 * rho0
            eps = 1e-6 * max(float(ssq.mean()), 1e-30)
            lam = -c / (ssq + eps)

            # position corrections
            coef = (lam[i] * mj + lam[j] * mi) / rho0
            dpx = np.bincount(i, coef * gx, minlength=self.n) \
                - np.bincount(j, coef * gx, minlength=self.n)
            dpy = np.bincount(i, coef * gy, minlength=self.n) \
                - np.bincount(j, coef * gy, minlength=self.n)
            dpz = np.bincount(i, coef * gz, minlength=self.n) \
                - np.bincount(j, coef * gz, minlength=self.n)
            dp = np.stack((dpx, dpy, dpz), axis=1)
            dn = np.linalg.norm(dp, axis=1)
            dp *= np.minimum(1.0, 0.4 * h / np.maximum(dn, 1e-12))[:, None]
            pred = pred + dp
            pred = self._project_boxes(pred)

        # velocity update
        v = (pred - self.pos) / dt
        self.pos = pred
        self.vel = v

        # XSPH viscosity (reuses the last neighbor list)
        i, j = pairs_i, pairs_j
        if i is not None and len(i) > 0:
            d = self.pos[i] - self.pos[j]
            r2 = np.einsum("ij,ij->i", d, d)
            m = r2 < h * h
            i, j, r2 = i[m], j[m], r2[m]
            if len(i) > 0:
                r = np.sqrt(np.maximum(r2, 1e-12))
                w = self.k6 * (h * h - r2) ** 3
                rho = self.mass * self.w0 \
                    + np.bincount(i, self.mass[j] * w, minlength=self.n) \
                    + np.bincount(j, self.mass[i] * w, minlength=self.n)
                for k in range(3):
                    dv = self.vel[i, k] - self.vel[j, k]
                    self.vel[:, k] += self.XSPH * (
                        np.bincount(i, (self.mass[j] / rho[i]) * w * (-dv),
                                    minlength=self.n)
                        + np.bincount(j, (self.mass[i] / rho[j]) * w * dv,
                                      minlength=self.n))

        self.vel = self._wall_velocity(self.pos, self.vel)
        # weak volumetric drag so splash waves damp out
        self.vel *= max(0.0, 1.0 - 0.10 * dt)

    # --------------------------------------------------------------- exports

    def advance_to(self, time: float):
        t = float(time)
        while self.t < t - 1e-9:
            dt = min(self.SUB_DT, t - self.t)
            self._step(dt)
            self.t += dt
        return self._export(t)

    def _export(self, t: float):
        pos, vel = self.pos, self.vel
        tree = cKDTree(pos)
        k = min(6, self.n)

        # liquid level set + active mask from nearest-particle distance
        d1, _ = tree.query(self.grid_pts, k=1)
        level = (d1 - 0.45 * self.spacing).reshape(self.GRID_SHAPE)
        active = (d1 < 2.5 * self.spacing).reshape(self.GRID_SHAPE)

        # collocated diagnostic velocity (kernel-weighted particle average)
        dk, ik = tree.query(self.grid_pts, k=k)
        wts = np.clip(2.0 * self.h - dk, 0.0, None)
        sw = wts.sum(1)
        vgrid = (wts[:, :, None] * vel[ik]).sum(1) / np.maximum(sw, 1e-12)[:, None]
        vgrid[sw <= 0.0] = 0.0
        vfield = vgrid.reshape(self.GRID_SHAPE + (3,))

        # boundary sample velocities
        kb = min(4, self.n)
        db, ib = tree.query(self.b_samples, k=kb)
        wb = np.clip(2.0 * self.h - db, 0.0, None)
        sb = wb.sum(1)
        bvel = (wb[:, :, None] * vel[ib]).sum(1) / np.maximum(sb, 1e-12)[:, None]
        bvel[sb <= 0.0] = 0.0

        fields = {
            "active_mask": active,
            "liquid_level_set": level,
            "velocity_field": vfield,
            "boundary_normal": self.b_normals.copy(),
            "boundary_target_velocity": np.zeros_like(self.b_normals),
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(2),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.radius.copy(),
            "material_label": self.labels.copy(),
            "deformation_gradient": np.tile(np.eye(3), (self.n, 1, 1)),
        }
        scalars = {
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.reference_velocity,
            "contact_length_scale": self.spacing,
        }
        frame = StateFrame(
            time=max(t, 0.0), position=pos.copy(), velocity=vel.copy(),
            mass=self.mass.copy(), body_id=self.labels.copy(),
            attributes=attributes, fields=fields, scalars=scalars)
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "smoothing_radius": float(self.h),
            "substep": self.SUB_DT,
            "iterations": self.ITERS,
            "time": float(self.t),
        }
