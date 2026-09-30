"""Three-dimensional free-surface liquid solver for scene 008 (rotating blade stirring).

Numerical method: Predictive-Corrective Position Based Fluids (Macklin & Mueller,
2013) -- an SPH discretization of the incompressible Navier-Stokes equations with

  * poly6 kernel for density evaluation and spiky kernel gradients for the
    pressure (density) constraint,
  * Jacobi relaxation of per-particle constraint multipliers lambda_i,
  * ghost/mirror boundary density so the tank floor, side walls and the moving
    blade do not create a spurious density deficit that sucks liquid onto them,
  * free-slip impermeable projection against the stationary tank solids and the
    kinematically driven blade (exact oriented-box SDF, analytic closest point
    and outward normal),
  * XSPH velocity smoothing as the numerical viscosity at this resolution.

The blade is a prescribed kinematic rigid body (frozen smoothstep angular-speed
ramp).  It transfers momentum to the liquid only through the moving-solid
boundary: predicted particle positions that penetrate the blade are projected
out along the local outward normal and the relative normal velocity is
cancelled, so the leading face physically pushes liquid and the trailing face
leaves a wake.  No animation, keyframes or precomputed trajectories are used.

Everything is deterministic: fixed time step, fixed particle set, no random
numbers (the seed only seeds nothing).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based incompressible fluid solver with a stirring blade."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {})
        self.scene = scene
        self.seed = int(seed)

        liquid = cfg["liquid"]
        tank = cfg["tank"]
        blade = cfg["blade"]
        domain = cfg["domain"]

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.rho0 = float(liquid["density"])
        self.half = float(tank["interior_half_width"])
        self.wall_top = float(tank["height"])
        self.depth = float(liquid["depth"])
        self.blade_half = np.asarray(blade["half_extents"], dtype=float)
        self.blade_center = np.asarray(blade["center"], dtype=float)
        self.blade_body_id = int(blade["body_id"])

        # Frozen floor footprint (from the authoritative scene geometry).
        geom_objects = {
            obj.get("id"): obj
            for obj in (cfg.get("scene_geometry") or {}).get("objects", [])
        }
        floor = geom_objects.get("tank_floor", {}).get("geometry", {})
        self.floor_extent = float((floor.get("half_extents") or [0.84, 0.84, 0.02])[0])

        self.sample_dt = float(cfg["sample_dt"])
        self.substeps = 5
        self.dt = self.sample_dt / self.substeps

        # ---------------- internal particle lattice (genuinely 3D) ----------
        self.s = 0.04
        nx = int(round(2.0 * self.half / self.s))
        self.nz_layers = max(int(round(self.depth / self.s)), 4)
        self.dz = self.depth / self.nz_layers
        xs = -self.half + (np.arange(nx) + 0.5) * self.s
        zs = (np.arange(self.nz_layers) + 0.5) * self.dz
        gx, gy, gz = np.meshgrid(xs, xs, zs, indexing="ij")
        pts = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)
        margin = 0.5 * min(self.s, self.dz)
        sdf0 = scene.blade_signed_distance(pts, 0.0)
        pts = pts[sdf0 >= margin - 1e-12]

        self.n = int(len(pts))
        self.cell_vol = self.s * self.s * self.dz
        self.particle_mass = self.rho0 * self.cell_vol
        self.margin = margin

        # ---------------- SPH kernels and PBF parameters ---------------------
        self.H = 0.10                       # kernel support radius
        self.K = 88                         # neighbor cap
        self.iterations = 3                 # constraint relaxation sweeps
        self.relax = 0.8                    # Jacobi relaxation factor
        self.eps_cfm = 1.0                  # CFM regularization
        self.dp_clip = 0.02                 # max position correction per sweep
        self.xsph_alpha = 0.03              # numerical viscosity coefficient
        self.v_clip = 20.0                  # finiteness guard

        self.poly6 = 315.0 / (64.0 * np.pi * self.H**9)
        self.spiky = 45.0 / (np.pi * self.H**6)
        self.vp = self.cell_vol             # particle mass / rest density

        # diagnostic interpolation kernel
        self.r_interp = 0.09
        self.k_interp = 48
        self.poly6i = 315.0 / (64.0 * np.pi * self.r_interp**9)

        self.x = pts.copy()
        self.v = np.zeros_like(pts)
        self._steps = 0

        # ---------------- frozen diagnostic grid -----------------------------
        self.hgrid = float(domain["grid_spacing"])
        self.grid_shape = tuple(int(v) for v in domain["reference_grid"])
        self.grid_origin = np.asarray(domain["bounds_min"], dtype=float)
        self.axes = [
            self.grid_origin[a] + (np.arange(self.grid_shape[a]) + 0.5) * self.hgrid
            for a in range(3)
        ]

        self._ready = True

    # --------------------------------------------------------- blade geometry
    def _pose(self, t: float):
        pose = self.scene.blade_kinematics(float(t))
        return (
            np.asarray(pose["rotation_matrix"], dtype=float),
            np.asarray(pose["position"], dtype=float),
            np.asarray(pose["angular_velocity"], dtype=float),
        )

    def _blade_sdf_local(self, local: np.ndarray) -> np.ndarray:
        q = np.abs(local) - self.blade_half
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            q.max(axis=1), 0.0
        )

    def _blade_closest(self, p: np.ndarray, R: np.ndarray, c: np.ndarray):
        """Closest blade surface point, outward normal and signed distance."""
        local = (p - c) @ R
        q = np.abs(local) - self.blade_half
        inside = q.max(axis=1) <= 0.0
        rows = np.arange(len(p))
        axis = np.argmax(q, axis=1)
        sign = np.where(local[rows, axis] >= 0.0, 1.0, -1.0)
        cp_local = np.clip(local, -self.blade_half, self.blade_half)
        cp_local[rows, axis] = sign * self.blade_half[axis]
        n_local = np.zeros_like(local)
        n_local[rows, axis] = sign
        cp_world = cp_local @ R.T + c
        sdf = self._blade_sdf_local(local)
        dvec = p - cp_world
        dist = np.linalg.norm(dvec, axis=1)
        n_face = n_local @ R.T
        safe = dist > 1e-12
        normal = np.where(safe[:, None], dvec / np.maximum(dist, 1e-12)[:, None], n_face)
        normal = np.where(inside[:, None], n_face, normal)
        return cp_world, normal, sdf

    # ------------------------------------------------------- solid projection
    def _project_blade(self, p: np.ndarray, R: np.ndarray, c: np.ndarray) -> np.ndarray:
        cp, normal, sdf = self._blade_closest(p, R, c)
        mask = sdf < self.margin
        if np.any(mask):
            p = p.copy()
            p[mask] = cp[mask] + self.margin * normal[mask]
        return p

    def _project_solids(self, p: np.ndarray, R: np.ndarray, c: np.ndarray) -> np.ndarray:
        p = np.array(p, dtype=float, copy=True)
        lim = self.half - self.margin
        lateral = p[:, 2] < self.wall_top
        p[lateral, 0] = np.clip(p[lateral, 0], -lim, lim)
        p[lateral, 1] = np.clip(p[lateral, 1], -lim, lim)
        on_floor = (
            (np.abs(p[:, 0]) <= self.floor_extent)
            & (np.abs(p[:, 1]) <= self.floor_extent)
        )
        p[on_floor, 2] = np.maximum(p[on_floor, 2], self.margin)
        return self._project_blade(p, R, c)

    # ------------------------------------------------- ghost boundary density
    def _kernel(self, d: np.ndarray) -> np.ndarray:
        out = np.zeros_like(d)
        m = d < self.H
        out[m] = self.poly6 * (self.H**2 - d[m] ** 2) ** 3
        return out

    def _ghost_density(self, p: np.ndarray, R: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Mirror-particle density contribution of tank solids and the blade."""
        add = np.zeros(len(p))
        h2 = 0.5 * self.H
        # floor (top face at z = 0)
        d = p[:, 2]
        m = (d > 0.0) & (d < h2)
        add[m] += self._kernel(2.0 * d[m])
        # side walls
        for ax in (0, 1):
            d = self.half - np.abs(p[:, ax])
            m = (d > 0.0) & (d < h2)
            add[m] += self._kernel(2.0 * d[m])
        # blade (mirror across the closest face)
        local = (p - c) @ R
        sdf = self._blade_sdf_local(local)
        m = (sdf > 0.0) & (sdf < h2)
        add[m] += self._kernel(2.0 * sdf[m])
        return self.particle_mass * add

    # ----------------------------------------------- density constraint sweep
    def _density_iteration(
        self, p: np.ndarray, ii: np.ndarray, jj: np.ndarray, R: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        n = len(p)
        d = p[ii] - p[jj]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = r2 < self.H**2
        if not np.all(keep):
            ii, jj, d, r2 = ii[keep], jj[keep], d[keep], r2[keep]
        r = np.sqrt(r2)
        W = self.poly6 * (self.H**2 - r2) ** 3

        rho = self.particle_mass * np.bincount(ii, weights=W, minlength=n)
        rho = rho + self._ghost_density(p, R, c)
        C = rho / self.rho0 - 1.0

        gmag = self.spiky * (self.H - r) ** 2
        inv_r = 1.0 / np.maximum(r, 1e-9)
        # grad of W(|p_i - p_j|) w.r.t. p_i
        grad = -(d * (gmag * inv_r)[:, None])

        gradC = np.empty((n, 3))
        for k in range(3):
            gradC[:, k] = self.vp * np.bincount(ii, weights=grad[:, k], minlength=n)
        sumsq = np.einsum("ij,ij->i", gradC, gradC) + (self.vp**2) * np.bincount(
            ii, weights=gmag**2, minlength=n
        )
        lam = -C / (sumsq + self.eps_cfm)

        lamsum = lam[ii] + lam[jj]
        dp = np.empty((n, 3))
        for k in range(3):
            dp[:, k] = self.vp * np.bincount(
                ii, weights=lamsum * grad[:, k], minlength=n
            )
        nrm = np.linalg.norm(dp, axis=1)
        scale = np.minimum(1.0, self.dp_clip / np.maximum(nrm, 1e-12))
        dp *= scale[:, None]

        p = p + self.relax * dp
        return self._project_solids(p, R, c)

    # ------------------------------------------------------------ XSPHoothing
    def _xsph(self, p: np.ndarray, v: np.ndarray, ii: np.ndarray, jj: np.ndarray):
        n = len(p)
        d = p[ii] - p[jj]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = r2 < self.H**2
        ii, jj, r2 = ii[keep], jj[keep], r2[keep]
        W = self.poly6 * (self.H**2 - r2) ** 3
        den = np.bincount(ii, weights=W, minlength=n)
        out = np.array(v, dtype=float, copy=True)
        safe = den > 1e-12
        for k in range(3):
            num = np.bincount(ii, weights=W * (v[jj, k] - v[ii, k]), minlength=n)
            corr = np.zeros(n)
            corr[safe] = num[safe] / den[safe]
            out[:, k] += self.xsph_alpha * corr
        return out

    # --------------------------------------------------- boundary velocities
    def _blade_velocity_imprint(
        self,
        p: np.ndarray,
        v: np.ndarray,
        R: np.ndarray,
        c: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        """Cancel relative normal velocity of liquid near the moving blade."""
        cp, normal, sdf = self._blade_closest(p, R, c)
        mask = sdf < 2.0 * self.margin
        if not np.any(mask):
            return v
        v_blade = np.cross(np.broadcast_to(omega, cp.shape), cp - c)
        rel = v - v_blade
        vn = np.einsum("ij,ij->i", rel, normal)
        bad = mask & (vn < 0.0)
        v = np.array(v, dtype=float, copy=True)
        v[bad] -= vn[bad, None] * normal[bad]
        return v

    def _wall_slip(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        lim = self.half - self.margin
        lateral = p[:, 2] < self.wall_top
        v = np.array(v, dtype=float, copy=True)
        for ax in (0, 1):
            hi = lateral & (p[:, ax] > lim - 1e-7) & (v[:, ax] > 0.0)
            lo = lateral & (p[:, ax] < -lim + 1e-7) & (v[:, ax] < 0.0)
            v[hi, ax] = 0.0
            v[lo, ax] = 0.0
        fl = (p[:, 2] < self.margin + 1e-7) & (v[:, 2] < 0.0)
        v[fl, 2] = 0.0
        return v

    # ------------------------------------------------------------------- step
    def _step(self) -> None:
        n = self.n
        t = (self._steps + 1) * self.dt
        R, c, omega = self._pose(t)

        v = self.v + self.dt * self.gravity
        p = self.x + v * self.dt

        tree = cKDTree(p)
        dist, idx = tree.query(p, k=self.K, distance_upper_bound=self.H, workers=-1)
        valid = (np.isfinite(dist) & (idx < n)).ravel()
        ii = np.repeat(np.arange(n), self.K)[valid]
        jj = idx.ravel()[valid]
        keep = dist.ravel()[valid] > 1e-9
        ii, jj = ii[keep], jj[keep]

        for _ in range(self.iterations):
            p = self._density_iteration(p, ii, jj, R, c)

        v = (p - self.x) / self.dt
        v = self._xsph(p, v, ii, jj)
        v = self._blade_velocity_imprint(p, v, R, c, omega)
        v = self._wall_slip(p, v)

        v = np.clip(v, -self.v_clip, self.v_clip)
        p = np.clip(p, -3.0, 3.0)
        np.nan_to_num(v, copy=False)
        np.nan_to_num(p, copy=False)

        self.x = p
        self.v = v
        self._steps += 1

    # ---------------------------------------------------------- interpolation
    def _interp_velocity(self, pts: np.ndarray, tree: cKDTree) -> np.ndarray:
        n = self.n
        pts = np.asarray(pts, dtype=float)
        k = int(min(self.k_interp, n))
        out = np.zeros((len(pts), 3))
        if len(pts) == 0:
            return out
        d, idx = tree.query(pts, k=k, distance_upper_bound=self.r_interp, workers=-1)
        valid = (np.isfinite(d) & (idx < n)).ravel()
        if not np.any(valid):
            return out
        row = np.repeat(np.arange(len(pts)), k)[valid]
        col = idx.ravel()[valid]
        dv = np.minimum(d.ravel()[valid], self.r_interp)
        w = np.maximum(self.poly6i * (self.r_interp**2 - dv**2) ** 3, 0.0)
        den = np.bincount(row, weights=w, minlength=len(pts))
        for kk in range(3):
            out[:, kk] = np.bincount(
                row, weights=w * self.v[col, kk], minlength=len(pts)
            )
        nz = den > 1e-12
        out[nz] /= den[nz, None]
        return out

    # --------------------------------------------------------- frame assembly
    def _surface_height(self) -> np.ndarray:
        """Liquid surface height on the diagnostic xy columns."""
        shape = self.grid_shape
        h = self.hgrid
        ix = np.clip(
            ((self.x[:, 0] - self.grid_origin[0]) / h).astype(np.int64), 0, shape[0] - 1
        )
        iy = np.clip(
            ((self.x[:, 1] - self.grid_origin[1]) / h).astype(np.int64), 0, shape[1] - 1
        )
        hmax = np.zeros(shape[:2])
        np.maximum.at(hmax, (ix, iy), self.x[:, 2])
        h_surf = np.where(hmax > 0.0, hmax + 0.5 * self.dz, 0.0)
        return h_surf

    def _build_frame(self, time: float) -> StateFrame:
        pose = self.scene.blade_kinematics(float(time))
        R = np.asarray(pose["rotation_matrix"], dtype=float)
        c = np.asarray(pose["position"], dtype=float)
        omega = np.asarray(pose["angular_velocity"], dtype=float)

        tree = cKDTree(self.x)
        h = self.hgrid
        shape = self.grid_shape
        h_surf = self._surface_height()

        xc, yc, zc = self.axes
        phi = zc[None, None, :] - h_surf[:, :, None]           # (nx,ny,nz)
        XG, YG, ZG = np.meshgrid(xc, yc, zc, indexing="ij")
        grid_pts = np.stack((XG, YG, ZG), axis=-1).reshape(-1, 3)
        sdf_grid = self._blade_sdf_local((grid_pts - c) @ R).reshape(shape)

        vfield = self._interp_velocity(grid_pts, tree).reshape(shape + (3,))

        active = (
            (phi < -h)
            & (ZG > h)
            & (np.abs(XG) < self.half - h)
            & (np.abs(YG) < self.half - h)
            & (sdf_grid > h)
        )
        wet = (phi < 0.0) & (sdf_grid > 0.0)
        density_field = wet.astype(float) * self.rho0

        # ---------------- stationary tank boundary samples --------------------
        pos_list, nrm_list, wet_list = [], [], []
        # floor
        FX, FY = np.meshgrid(xc, yc, indexing="ij")
        floor_pts = np.stack(
            (FX.ravel(), FY.ravel(), np.zeros(FX.size)), axis=-1
        )
        floor_nrm = np.tile(np.array([0.0, 0.0, 1.0]), (FX.size, 1))
        floor_wet = (hmax_wet := h_surf > 1e-9).ravel()
        pos_list.append(floor_pts)
        nrm_list.append(floor_nrm)
        wet_list.append(floor_wet)
        # side walls
        zlev = np.arange(h / 2.0, self.wall_top, h)
        for ax in (0, 1):
            for sign in (-1, 1):
                A, Z = np.meshgrid(xc, zlev, indexing="ij")  # A: along wall, Z: height
                pts_w = np.zeros((A.size, 3))
                pts_w[:, ax] = sign * self.half
                pts_w[:, 1 - ax] = A.ravel()
                pts_w[:, 2] = Z.ravel()
                nrm_w = np.zeros_like(pts_w)
                nrm_w[:, ax] = -float(sign)
                # nearest interior diagnostic column along the wall
                if ax == 0:
                    col = shape[0] - 1 if sign > 0 else 0
                    hs = np.broadcast_to(h_surf[col, :], A.shape)
                else:
                    col = shape[1] - 1 if sign > 0 else 0
                    hs = np.broadcast_to(h_surf[:, col], A.shape)
                wet_w = Z.ravel() < hs.ravel()
                pos_list.append(pts_w)
                nrm_list.append(nrm_w)
                wet_list.append(wet_w)

        wall_pos = np.concatenate(pos_list)
        wall_nrm = np.concatenate(nrm_list)
        wall_wet = np.concatenate(wet_list)
        wall_pos_w = wall_pos[wall_wet]
        wall_nrm_w = wall_nrm[wall_wet]
        wall_vel_w = self._interp_velocity(wall_pos_w, tree)
        wall_tgt_w = np.zeros_like(wall_vel_w)

        # ---------------- moving blade boundary samples -----------------------
        bs = self.scene.blade_surface_samples(float(time))
        bpos = np.asarray(bs["position"], dtype=float)
        bnrm = np.asarray(bs["normal"], dtype=float)
        btgt = np.asarray(bs["target_velocity"], dtype=float)
        bix = np.clip(
            ((bpos[:, 0] - self.grid_origin[0]) / h).astype(np.int64), 0, shape[0] - 1
        )
        biy = np.clip(
            ((bpos[:, 1] - self.grid_origin[1]) / h).astype(np.int64), 0, shape[1] - 1
        )
        bwet = bpos[:, 2] < h_surf[bix, biy]
        bpos_w, bnrm_w, btgt_w = bpos[bwet], bnrm[bwet], btgt[bwet]
        bvel_w = self._interp_velocity(bpos_w, tree)

        # ---------------- particle state plus kinematic blade body ------------
        count = self.n + 1
        position = np.vstack((self.x, c[None, :]))
        velocity = np.vstack((self.v, np.zeros((1, 3))))
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))
        orientation[-1] = np.asarray(pose["orientation"], dtype=float)
        angular_velocity = np.zeros((count, 3))
        angular_velocity[-1] = omega
        body_id = np.zeros(count, dtype=np.int64)
        body_id[-1] = self.blade_body_id
        mass_quantity = np.r_[
            np.full(self.n, self.particle_mass), 0.0
        ]
        mass = mass_quantity.copy()

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass_quantity,
                "particle_volume": np.r_[np.full(self.n, self.cell_vol), 0.0],
                "particle_radius": np.r_[
                    np.full(self.n, 0.5 * min(self.s, self.dz)), 0.0
                ],
            },
            fields={
                "velocity_field": vfield,
                "active_mask": active.astype(float),
                "liquid_level_set": phi,
                "density_field": density_field,
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": wall_pos_w,
                "boundary_normal": wall_nrm_w,
                "boundary_velocity": wall_vel_w,
                "boundary_target_velocity": wall_tgt_w,
                "blade_boundary_position": bpos_w,
                "blade_boundary_normal": bnrm_w,
                "blade_boundary_velocity": bvel_w,
                "blade_boundary_target_velocity": btgt_w,
                "grid_origin": self.grid_origin.copy(),
            },
            scalars={
                "grid_spacing": self.hgrid,
                "characteristic_length": float(2.0 * self.blade_half[0]),
                "reference_velocity": float(
                    abs(float(self.scene.config["blade"]["angular_speed"]))
                    * self.blade_half[0]
                ),
                "initial_liquid_volume": float(self.n * self.cell_vol),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver not initialized")
        time = float(time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and nonnegative")
        target = int(round(time / self.dt))
        while self._steps < target:
            self._step()
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "internal_spacing": float(self.s),
            "kernel_support": float(self.H),
            "time_step": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "constraint_iterations": int(self.iterations),
            "steps_executed": int(self._steps),
            "liquid_mass_kg": float(self.n * self.particle_mass),
        }
