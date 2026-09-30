"""3-D buoyant smoke solver with static voxel obstacles.

Implements a classic stable-fluid discretization on a fixed MAC-like
staggered grid, using finite-difference pressure projection and
semi-Lagrangian advection.  The solver runs entirely on NumPy/SciPy
and exports the StateFrame fields required by the gas benchmark.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from worldsolver.contracts import Scene, Solver, StateFrame


class BenchmarkSolver(Solver):
    def initialize(self, scene: Scene, seed: int) -> None:
        self.seed = int(seed)
        self.scene = scene
        self.config = scene.config
        self.dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Grid setup from the subject gas domain.
        geom = next(
            obj
            for obj in self.config["scene_geometry"]["objects"]
            if obj["id"] == "subject"
        )
        self.shape = tuple(int(s) for s in geom["geometry"]["grid_shape"])
        bounds_min = np.asarray(geom["geometry"]["bounds_min"], dtype=float)
        bounds_max = np.asarray(geom["geometry"]["bounds_max"], dtype=float)
        self.domain_min = bounds_min
        self.domain_max = bounds_max
        # Cell-centered grid spacing.
        self.h = (bounds_max - bounds_min) / np.asarray(self.shape, dtype=float)
        if not np.allclose(self.h, self.h[0]):
            raise ValueError("Only uniform cubic grids are supported")
        self.h = float(self.h[0])

        # Coordinate arrays.
        self.x = np.linspace(bounds_min[0], bounds_max[0], self.shape[0])
        self.y = np.linspace(bounds_min[1], bounds_max[1], self.shape[1])
        self.z = np.linspace(bounds_min[2], bounds_max[2], self.shape[2])
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # Static obstacle mask.
        self.obstacle = self._build_obstacle_mask()

        # Source location.
        self.source = self._build_source_mask()

        # State arrays.
        self.density = np.zeros(self.shape, dtype=float)
        self.velocity = np.zeros(self.shape + (3,), dtype=float)
        self.pressure = np.zeros(self.shape, dtype=float)

        # Initialize with a small puff from the source region.
        self.density[self.source] = 0.5

        # Substeps for stability.
        self.substeps = 4
        # Viscosity / diffusion coefficients.
        self.nu = 0.02          # kinematic viscosity for velocity
        self.kappa = 0.01       # smoke diffusion
        self.buoyancy = 1.2      # buoyant acceleration per unit density
        self.ambient_temp = 0.0

        self.current_time = 0.0
        self._frame_cache: dict[float, StateFrame] = {}

        # Cache boundary samples/normals (static scene).
        self._cached_boundary_samples = self._build_boundary_samples()
        self._cached_boundary_normals = self._build_boundary_normals()

    def _inside_aabb(self, points: np.ndarray, min_c: np.ndarray, max_c: np.ndarray) -> np.ndarray:
        return np.all((points >= min_c) & (points <= max_c), axis=-1)

    def _build_obstacle_mask(self) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        for obj in self.config["scene_geometry"]["objects"]:
            if obj["semantic"] != "boundary":
                continue
            geo = obj["geometry"]
            if geo["type"] != "box":
                continue
            t = np.asarray(obj["transform"]["translation"], dtype=float)
            he = np.asarray(geo["half_extents"], dtype=float)
            min_c = t - he
            max_c = t + he
            inside = self._inside_aabb(
                np.stack([self.X, self.Y, self.Z], axis=-1).reshape(-1, 3),
                min_c,
                max_c,
            ).reshape(self.shape)
            mask |= inside
        return mask

    def _build_source_mask(self) -> np.ndarray:
        source_obj = next(
            obj
            for obj in self.config["scene_geometry"]["objects"]
            if obj["id"] == "smoke_source"
        )
        t = np.asarray(source_obj["transform"]["translation"], dtype=float)
        r = float(source_obj["geometry"]["radius"])
        h = float(source_obj["geometry"]["height"])
        cyl_axis = np.array([0.0, 0.0, 1.0])
        points = np.stack([self.X, self.Y, self.Z], axis=-1).reshape(-1, 3)
        dz = points @ cyl_axis
        rad = np.linalg.norm(points - (points @ cyl_axis[:, None]) * cyl_axis[None, :], axis=1)
        mask = (np.abs(dz - t[2]) <= h / 2.0) & (rad <= r)
        return mask.reshape(self.shape)

    def _build_boundary_samples(self) -> np.ndarray:
        """Positions of obstacle boundary samples (face-centers adjacent to fluid)."""
        samples = []
        mask = self.obstacle
        if not np.any(mask):
            return np.zeros((0, 3), dtype=float)
        for axis in range(3):
            for direction in (-1, 1):
                shifted = np.roll(mask, direction, axis=axis)
                face = mask & (~shifted)
                indices = np.argwhere(face)
                for idx in indices:
                    pos = self._cell_center(*idx)
                    offset = np.zeros(3)
                    offset[axis] = direction * self.h * 0.5
                    samples.append(pos + offset)
        if not samples:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(samples, dtype=float)

    def _build_boundary_normals(self) -> np.ndarray:
        """Return outward unit normals for boundary samples."""
        samples = self._cached_boundary_samples
        normals = np.zeros_like(samples)
        if samples.size == 0:
            return normals
        for i, p in enumerate(samples):
            best_dist = -1e9
            best_grad = np.array([0.0, 0.0, 1.0])
            for obj in self.config["scene_geometry"]["objects"]:
                if obj["semantic"] != "boundary":
                    continue
                if obj["geometry"]["type"] != "box":
                    continue
                t = np.asarray(obj["transform"]["translation"], dtype=float)
                he = np.asarray(obj["geometry"]["half_extents"], dtype=float)
                delta = np.abs(p - t) - he
                dist = np.linalg.norm(np.maximum(delta, 0.0)) + min(np.max(delta), 0.0)
                outside = delta > 0
                if np.any(outside):
                    vec = np.maximum(delta, 0.0)
                    norm = np.linalg.norm(vec)
                    grad = vec / norm if norm > 0 else np.zeros(3)
                else:
                    argmax = np.argmax(delta)
                    grad = np.zeros(3)
                    grad[argmax] = np.sign(p[argmax] - t[argmax])
                    norm = np.linalg.norm(grad)
                    grad = grad / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
                if dist > best_dist:
                    best_dist = dist
                    best_grad = grad
            normals[i] = best_grad
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 0, normals / norms, normals)
        return normals

    def _boundary_count(self) -> int:
        return self._cached_boundary_samples.shape[0]

    def _boundary_samples(self) -> np.ndarray:
        return self._cached_boundary_samples

    def _boundary_normals(self) -> np.ndarray:
        return self._cached_boundary_normals

    def _cell_center(self, i: int, j: int, k: int) -> np.ndarray:
        return np.array([self.x[i], self.y[j], self.z[k]], dtype=float)

    # -----------------------------------------------------------------
    # Spatial derivative helpers (staggered).
    # -----------------------------------------------------------------
    def _divergence(self, field: np.ndarray) -> np.ndarray:
        h_inv = 1.0 / self.h
        div = np.zeros(self.shape, dtype=float)
        div[1:-1, 1:-1, 1:-1] = (
            field[2:-1, 1:-1, 1:-1, 0] - field[1:-2, 1:-1, 1:-1, 0]
            + field[1:-1, 2:-1, 1:-1, 1] - field[1:-1, 1:-2, 1:-1, 1]
            + field[1:-1, 1:-1, 2:-1, 2] - field[1:-1, 1:-1, 1:-2, 2]
        ) * h_inv
        return div

    def _gradient(self, p: np.ndarray) -> np.ndarray:
        h_inv = 1.0 / self.h
        grad = np.zeros(self.shape + (3,), dtype=float)
        grad[1:-1, 1:-1, 1:-1, 0] = (p[2:-1, 1:-1, 1:-1] - p[1:-2, 1:-1, 1:-1]) * h_inv
        grad[1:-1, 1:-1, 1:-1, 1] = (p[1:-1, 2:-1, 1:-1] - p[1:-1, 1:-2, 1:-1]) * h_inv
        grad[1:-1, 1:-1, 1:-1, 2] = (p[1:-1, 1:-1, 2:-1] - p[1:-1, 1:-1, 1:-2]) * h_inv
        return grad

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        h2_inv = 1.0 / (self.h * self.h)
        lap = np.zeros_like(field)
        for c in range(field.shape[-1]):
            f = field[..., c]
            lap[..., c] = (
                np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                + np.roll(f, 1, 2) + np.roll(f, -1, 2)
                - 6.0 * f
            ) * h2_inv
        return lap

    def _apply_obstacle_velocity(self, vel: np.ndarray) -> None:
        vel[self.obstacle] = 0.0

    def _apply_obstacle_scalar(self, scalar: np.ndarray) -> None:
        # Zero-out scalar inside obstacles; simple no-penetration visual.
        scalar[self.obstacle] = 0.0

    def _enforce_walls(self, vel: np.ndarray) -> None:
        # Zero velocity normal to domain boundaries.
        vel[0, :, :, 0] = 0.0
        vel[-1, :, :, 0] = 0.0
        vel[:, 0, :, 1] = 0.0
        vel[:, -1, :, 1] = 0.0
        vel[:, :, 0, 2] = 0.0
        vel[:, :, -1, 2] = 0.0
        self._apply_obstacle_velocity(vel)

    # -----------------------------------------------------------------
    # Semi-Lagrangian advection helpers.
    # -----------------------------------------------------------------
    def _interpolate(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of field at arbitrary positions.

        field: (nx,ny,nz) or (nx,ny,nz,3) array.
        pos:   (N,3) positions in world coordinates.
        Returns interpolated values of same trailing shape.
        """
        field = np.asarray(field)
        pos = np.asarray(pos)
        # Map position to continuous index space.
        min_c = np.array([self.x[0], self.y[0], self.z[0]])
        idx = (pos - min_c) / self.h

        # Floor coordinates and compute weights.
        i0 = np.floor(idx).astype(int)
        i0 = np.clip(i0, 0, np.array(self.shape) - 2)
        i1 = i0 + 1
        wx, wy, wz = idx - i0
        # Trilinear interpolation using component-wise weights.
        # Reshape weights to (N,1,1,1) for broadcasting.
        wshape = (-1, 1, 1, 1)
        wx_ = wx.reshape(-1, 1, 1, 1)
        wy_ = wy.reshape(-1, 1, 1, 1)
        wz_ = wz.reshape(-1, 1, 1, 1)
        iwx = wx_.shape
        def gather(i, j, k):
            return field[i, j, k]

        if field.ndim == 3:
            values = (
                gather(i0[:, 0], i0[:, 1], i0[:, 2]) * (1 - wx) * (1 - wy) * (1 - wz)
                + gather(i1[:, 0], i0[:, 1], i0[:, 2]) * wx * (1 - wy) * (1 - wz)
                + gather(i0[:, 0], i1[:, 1], i0[:, 2]) * (1 - wx) * wy * (1 - wz)
                + gather(i0[:, 0], i0[:, 1], i1[:, 2]) * (1 - wx) * (1 - wy) * wz
                + gather(i1[:, 0], i1[:, 1], i0[:, 2]) * wx * wy * (1 - wz)
                + gather(i1[:, 0], i0[:, 1], i1[:, 2]) * wx * (1 - wy) * wz
                + gather(i0[:, 0], i1[:, 1], i1[:, 2]) * (1 - wx) * wy * wz
                + gather(i1[:, 0], i1[:, 1], i1[:, 2]) * wx * wy * wz
            )
        else:
            values = (
                gather(i0[:, 0], i0[:, 1], i0[:, 2]) * (1 - wx_)
                + gather(i1[:, 0], i0[:, 1], i0[:, 2]) * wx_
                + gather(i0[:, 0], i1[:, 1], i0[:, 2]) * (1 - wx_)
                + gather(i0[:, 0], i0[:, 1], i1[:, 2]) * (1 - wx_)
                + gather(i1[:, 0], i1[:, 1], i0[:, 2]) * wx_
                + gather(i1[:, 0], i0[:, 1], i1[:, 2]) * wx_
                + gather(i0[:, 0], i1[:, 1], i1[:, 2]) * (1 - wx_)
                + gather(i1[:, 0], i1[:, 1], i1[:, 2]) * wx_
            )
        return values

    def _advect_scalar(self, scalar: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        # Build cell-centered positions.
        pos = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=-1)
        # Back-trace.
        back = pos - dt * vel.reshape(-1, 3)
        new_scalar = self._interpolate(scalar, back).reshape(self.shape)
        return new_scalar

    def _advect_velocity(self, vel: np.ndarray, dt: float) -> np.ndarray:
        new_vel = np.zeros_like(vel)
        # Advect each component using component-staggered positions.
        for c in range(3):
            # Staggered positions for this component.
            pos = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=-1)
            if c == 0:
                pos[:, 0] += 0.5 * self.h
            elif c == 1:
                pos[:, 1] += 0.5 * self.h
            elif c == 2:
                pos[:, 2] += 0.5 * self.h
            back = pos - dt * vel.reshape(-1, 3)
            comp = self._interpolate(vel[..., c], back).reshape(self.shape)
            new_vel[..., c] = comp
        return new_vel

    # -----------------------------------------------------------------
    # Pressure projection.
    # -----------------------------------------------------------------
    def _pressure_project(self, vel: np.ndarray) -> np.ndarray:
        # Divergence on cell centers.
        div = self._divergence(vel)
        # Solve -h^2 L p = h^2 * div / dt  =>  we solve L p = div.
        # Build 7-point Laplacian matrix with obstacle/Neumann boundary conditions.
        nx, ny, nz = self.shape
        n = nx * ny * nz

        # Map active cells: fluid cells not inside obstacle.
        active = ~self.obstacle
        idx = np.full(n, -1, dtype=int)
        idx[active.ravel()] = np.arange(np.count_nonzero(active))

        if np.count_nonzero(active) == 0:
            return vel

        # Coefficients.
        coeffs = []
        rows = []
        cols = []
        b = div[active].ravel()
        count = np.count_nonzero(active)

        def add(i, j, v):
            rows.append(i)
            cols.append(j)
            coeffs.append(v)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if not active[i, j, k]:
                        continue
                    flat = idx[i, j, k]
                    diag = 0.0
                    # Neighbor in x.
                    for offset, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1), (-1, 2), (1, 2)]:
                        # neighbor index
                        nb = [i, j, k]
                        ax = axis
                        nb[ax] += offset
                        if nb[ax] < 0 or nb[ax] >= self.shape[ax]:
                            continue
                        ni, nj, nk = nb
                        if active[ni, nj, nk]:
                            diag -= 1.0
                            nflat = idx[ni, nj, nk]
                            add(flat, nflat, 1.0)
                    diag -= 0.0
                    # We use the standard 6-neighbor Laplacian scaled by 1/h^2 below.
                    add(flat, flat, diag)

        A = sp.csr_matrix((coeffs, (rows, cols)), shape=(count, count))
        # Scale by 1/h^2.
        A = A * (1.0 / (self.h * self.h))

        # Solve.
        try:
            p_flat, info = spl.cg(A, b, atol=1e-6, tol=1e-6, maxiter=500)
            if info < 0:
                raise RuntimeError("Pressure solver failed")
        except Exception:
            p_flat = spl.spsolve(A, b)

        p = np.zeros(n, dtype=float)
        p[active.ravel()] = p_flat
        p = p.reshape(self.shape)
        grad_p = self._gradient(p)
        vel = vel - grad_p * (self.h * self.h)
        self._enforce_walls(vel)
        return vel

    # -----------------------------------------------------------------
    # Main step.
    # -----------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # 1. Add buoyancy.
        buoy = self.buoyancy * (self.density[..., None] - self.ambient_temp)
        self.velocity[..., 2] += buoy * dt
        self.velocity += self.gravity * dt
        # Clamp vertical gravity influence for smoke.
        self.velocity[..., 2] += (self.gravity[2] * 0.05) * dt

        # 2. Advect velocity and scalar.
        self.velocity = self._advect_velocity(self.velocity, dt)
        self.density = self._advect_scalar(self.density, self.velocity, dt)

        # 3. Viscosity / diffusion.
        self.velocity += self.nu * self._laplacian(self.velocity) * dt
        self.density += self.kappa * self._laplacian(self.density[..., None])[..., 0] * dt

        # 4. Pressure projection to enforce incompressibility + wall conditions.
        self.velocity = self._pressure_project(self.velocity)

        # 5. Inject source.
        self.density[self.source] = np.minimum(
            self.density[self.source] + 1.5 * dt,
            1.2,
        )

        # 6. Enforce obstacles / walls.
        self._apply_obstacle_scalar(self.density)
        self._apply_obstacle_velocity(self.velocity)
        self._enforce_walls(self.velocity)

    def _compute_frame(self, time: float) -> StateFrame:
        # Cache positions from the scene's particle cloud to keep N consistent.
        initial = self.scene.initial_state()
        position = initial.position
        velocity = initial.velocity
        mass = initial.mass
        body_id = initial.body_id
        attributes = dict(initial.attributes)

        # Build boundary sample arrays.
        boundary_pos = self._boundary_samples()
        boundary_count = boundary_pos.shape[0]
        boundary_velocity = np.zeros((boundary_count, 3), dtype=float)
        boundary_target = np.zeros((boundary_count, 3), dtype=float)
        boundary_normal = self._boundary_normals()

        # Active mask: fluid cells not inside obstacle.
        active_mask = (~self.obstacle).astype(float)

        # Source term array.
        scalar_source = np.zeros(self.shape, dtype=float)
        scalar_source[self.source] = 1.0

        # Collocated velocity field stored as [nx,ny,nz,3].
        velocity_field = self.velocity.copy()

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields={
                "active_mask": active_mask,
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_target,
                "boundary_velocity": boundary_velocity,
                "scalar_source": scalar_source,
                "transported_scalar": self.density.copy(),
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(np.max(np.ptp(position, axis=0))),
                "grid_spacing": self.h,
                "reference_velocity": 1.0,
            },
        )
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time in self._frame_cache:
            return self._frame_cache[time]

        if time < self.current_time:
            raise ValueError("Cannot advance backwards in time")

        dt_total = time - self.current_time
        if dt_total > 0:
            n = int(np.ceil(dt_total / (self.dt / self.substeps)))
            sub_dt = dt_total / max(n, 1)
            for _ in range(n):
                self._step(sub_dt)
            self.current_time = time

        frame = self._compute_frame(time)
        self._frame_cache[time] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(np.round(self.current_time / self.dt)) if self.dt else 0,
            "max_density": float(self.density.max()) if self.density.size else 0.0,
        }
