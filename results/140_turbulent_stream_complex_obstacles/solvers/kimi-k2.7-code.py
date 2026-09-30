"""Solver Agent-owned implementation.

Implements a simple finite-difference Navier-Stokes smoke solver on the
voxel grid defined by the scene.  The velocity field is advanced with a
stable semi-Lagrangian advection + forward Euler pressure-free momentum
treatment; the smoke scalar is advected semi-Lagrangian.  Boundary conditions
enforce zero normal velocity on the ground and obstacles and a prescribed
inlet velocity at the nozzle.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.scene = scene
        self.config = scene.config
        self.duration = float(self.config["duration"])
        self.sample_dt = float(self.config["sample_dt"])

        # Grid
        shape = tuple(self.config["sampling"]["field_grid_shape"])
        self.Nx, self.Ny, self.Nz = shape
        bounds_min = np.array([-3.0, -1.2, 0.0])
        bounds_max = np.array([3.0, 1.2, 3.8])
        self.L = bounds_max - bounds_min
        self.dx = self.L / np.array([self.Nx, self.Ny, self.Nz])
        self.dx_min = float(self.dx.min())

        # Node coordinates
        self.x = np.linspace(bounds_min[0], bounds_max[0], self.Nx)
        self.y = np.linspace(bounds_min[1], bounds_max[1], self.Ny)
        self.z = np.linspace(bounds_min[2], bounds_max[2], self.Nz)

        # Physical parameters
        self.nu = 0.015  # kinematic viscosity
        self.inlet_speed = 2.2
        self.inlet_center = np.array([-2.72, 0.0, 1.0])
        self.inlet_axis = np.array([1.0, 0.0, 0.0])
        self.inlet_radius = 0.4
        self.dt = self.sample_dt / 4.0  # internal sub-step

        # Scene geometry (boxes only)
        self.obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        for obj in self.config.get("scene_geometry", {}).get("objects", []):
            if obj.get("collision_enabled") and obj.get("geometry", {}).get("type") == "box":
                half = np.array(obj["geometry"]["half_extents"], float)
                trans = np.array(obj["transform"]["translation"], float)
                self.obstacles.append((trans, half))

        # Build static solid mask
        self.solid_mask = np.zeros(shape, dtype=bool)
        for trans, half in self.obstacles:
            mask = (
                (self.x[:, None, None] >= trans[0] - half[0])
                & (self.x[:, None, None] <= trans[0] + half[0])
                & (self.y[None, :, None] >= trans[1] - half[1])
                & (self.y[None, :, None] <= trans[1] + half[1])
                & (self.z[None, None, :] >= trans[2] - half[2])
                & (self.z[None, None, :] <= trans[2] + half[2])
            )
            self.solid_mask |= mask

        # Ground boundary (floor at z=0; mark first layer of cell centers as solid)
        self.solid_mask |= self.z[None, None, :] <= self.dx[2] * 0.5

        # Active mask: fluid can evolve in non-solid cells
        self.active_mask = ~self.solid_mask

        # Initialize scalar from scene and store particle data
        init_frame = scene.initial_state()
        self.scalar: np.ndarray = np.asarray(init_frame.fields["transported_scalar"]).copy()
        self.velocity = np.zeros(shape + (3,), dtype=float)

        self._positions = np.asarray(init_frame.position)
        self._velocities = np.asarray(init_frame.velocity)
        self._masses = np.asarray(init_frame.mass)
        self._labels = np.asarray(init_frame.body_id)
        self._radii = np.asarray(init_frame.attributes["particle_radius"])

        # Boundary samples (uniform sphere-ish set around each obstacle)
        self.boundary_points = np.zeros((0, 3), dtype=float)
        self.boundary_normals_all = np.zeros((0, 3), dtype=float)
        samples: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        for trans, half in self.obstacles:
            pts: list[np.ndarray] = []
            nms: list[np.ndarray] = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    n_pts = 8
                    ax0, ax1 = [i for i in range(3) if i != axis]
                    c0 = np.linspace(-0.5, 0.5, n_pts) * half[ax0] * 2.0
                    c1 = np.linspace(-0.5, 0.5, n_pts) * half[ax1] * 2.0
                    C0, C1 = np.meshgrid(c0, c1, indexing="ij")
                    pts_face = np.zeros((n_pts * n_pts, 3))
                    pts_face[:, axis] = trans[axis] + sign * half[axis]
                    pts_face[:, ax0] = C0.ravel()
                    pts_face[:, ax1] = C1.ravel()
                    normal = np.zeros(3)
                    normal[axis] = sign
                    nms_face = np.tile(normal, (n_pts * n_pts, 1))
                    pts.append(pts_face)
                    nms.append(nms_face)
            if pts:
                samples.append(np.concatenate(pts, axis=0))
                normals.append(np.concatenate(nms, axis=0))
        if samples:
            self.boundary_points = np.concatenate(samples, axis=0)
            self.boundary_normals_all = np.concatenate(normals, axis=0)

        self.current_time = 0.0
        self._frame0 = self._build_frame(0.0)

    def _build_frame(self, time: float) -> StateFrame:
        # Velocity diagnostics at active cells
        vfield = np.zeros((self.Nx, self.Ny, self.Nz, 3), dtype=float)
        vfield[self.active_mask] = self.velocity[self.active_mask]

        # Boundary velocities (sample velocity at boundary points from grid)
        if self.boundary_points.size:
            bvel = self._sample_velocity(self.boundary_points, vfield)
        else:
            bvel = np.zeros((0, 3), dtype=float)
        # Static walls: prescribed (target) velocity is zero; actual is the sampled field.
        bvel_target = np.zeros_like(bvel)

        active_scalar = self.scalar.copy()
        active_scalar[self.solid_mask] = 0.0

        # Scalar source: emit smoke near inlet
        src = self._source_rate(time)

        frame = StateFrame(
            time=time,
            position=self._positions.copy(),
            velocity=self._velocities.copy(),
            mass=self._masses.copy(),
            body_id=self._labels.copy(),
            attributes={
                "mass_quantity": self._masses.copy(),
                "particle_radius": self._radii.copy(),
                "material_label": self._labels.copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], len(self._positions), axis=0),
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "boundary_normal": self.boundary_normals_all.copy(),
                "boundary_target_velocity": bvel_target.copy(),
                "boundary_velocity": bvel.copy(),
                "scalar_source": src.copy(),
                "transported_scalar": active_scalar,
                "velocity_field": vfield,
            },
            scalars={
                "contact_length_scale": self.dx_min,
                "grid_spacing": self.dx_min,
                "characteristic_length": max(float(self.L.max()), self.dx_min),
                "reference_velocity": float(self.inlet_speed),
            },
        )
        return frame

    def _source_rate(self, time: float) -> np.ndarray:
        shape = (self.Nx, self.Ny, self.Nz)
        src = np.zeros(shape, dtype=float)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        dist = np.sqrt((Y - self.inlet_center[1]) ** 2 + (Z - self.inlet_center[2]) ** 2)
        mask = (X < -2.5) & (dist < self.inlet_radius)
        rate = 2.5 * np.exp(-((time - 0.5) / 1.5) ** 2) if time < 3.0 else 1.5
        src[mask] = rate
        return src

    def _inlet_velocity(self) -> np.ndarray:
        v = np.zeros((self.Nx, self.Ny, self.Nz, 3), dtype=float)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        dist = np.sqrt((Y - self.inlet_center[1]) ** 2 + (Z - self.inlet_center[2]) ** 2)
        mask = (X < -2.4) & (dist < self.inlet_radius)
        profile = self.inlet_speed * (1.0 - (dist / self.inlet_radius) ** 2)
        profile = np.clip(profile, 0.0, self.inlet_speed)
        v[mask, 0] = profile[mask]
        return v

    def _apply_inlet(self, v: np.ndarray) -> None:
        v_in = self._inlet_velocity()
        inlet_region = v_in[:, :, :, 0] > 0.0
        v[inlet_region] = v_in[inlet_region]

    def _apply_solid_bc(self, v: np.ndarray) -> None:
        # Zero velocity inside solids
        v[self.solid_mask] = 0.0

    def _interpolate(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        # pos: [N,3] in world coordinates
        bounds_min = np.array([-3.0, -1.2, 0.0])
        idx = (pos - bounds_min[None, :]) / self.dx[None, :]
        i0 = np.floor(idx).astype(int)
        i0 = np.clip(i0, 0, np.array([self.Nx - 2, self.Ny - 2, self.Nz - 2]))
        di = idx - i0
        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        c000 = field[ix, iy, iz]
        c100 = field[ix + 1, iy, iz]
        c010 = field[ix, iy + 1, iz]
        c001 = field[ix, iy, iz + 1]
        c110 = field[ix + 1, iy + 1, iz]
        c101 = field[ix + 1, iy, iz + 1]
        c011 = field[ix, iy + 1, iz + 1]
        c111 = field[ix + 1, iy + 1, iz + 1]
        wx, wy, wz = di[:, 0:1], di[:, 1:2], di[:, 2:3]
        return (
            (1 - wx) * (1 - wy) * (1 - wz) * c000
            + wx * (1 - wy) * (1 - wz) * c100
            + (1 - wx) * wy * (1 - wz) * c010
            + (1 - wx) * (1 - wy) * wz * c001
            + wx * wy * (1 - wz) * c110
            + wx * (1 - wy) * wz * c101
            + (1 - wx) * wy * wz * c011
            + wx * wy * wz * c111
        )

    def _sample_velocity(self, pos: np.ndarray, vfield: np.ndarray) -> np.ndarray:
        return self._interpolate(vfield, pos)

    def _advect_field(self, f: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        # Semi-Lagrangian: trace back and interpolate
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        pos = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        back = pos - dt * v.reshape(-1, 3)
        # Clamp back positions to bounds
        bounds_min = np.array([-3.0, -1.2, 0.0])
        bounds_max = np.array([3.0, 1.2, 3.8])
        back = np.clip(back, bounds_min + 1e-6, bounds_max - 1e-6)
        f_back = self._interpolate(f, back).reshape(f.shape)
        return f_back

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        # interior
        out[1:-1, 1:-1, 1:-1] = (
            f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
            - 6.0 * f[1:-1, 1:-1, 1:-1]
        ) / (self.dx_min ** 2)
        return out

    def _divergence(self, v: np.ndarray) -> np.ndarray:
        div = np.zeros((self.Nx, self.Ny, self.Nz), dtype=float)
        div[1:-1, 1:-1, 1:-1] = (
            (v[2:, 1:-1, 1:-1, 0] - v[:-2, 1:-1, 1:-1, 0]) / (2.0 * self.dx[0])
            + (v[1:-1, 2:, 1:-1, 1] - v[1:-1, :-2, 1:-1, 1]) / (2.0 * self.dx[1])
            + (v[1:-1, 1:-1, 2:, 2] - v[1:-1, 1:-1, :-2, 2]) / (2.0 * self.dx[2])
        )
        return div

    def _pressure_projection(self, v: np.ndarray) -> np.ndarray:
        div = self._divergence(v)
        p = np.zeros_like(div)
        # Simple Jacobi iterations for Poisson equation lap(p) = div
        for _ in range(40):
            p_new = (
                p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
                + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
                + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
                - div[1:-1, 1:-1, 1:-1] * self.dx_min ** 2
            ) / 6.0
            p[1:-1, 1:-1, 1:-1] = p_new
            # Neumann at boundaries (copy)
            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = p[:, :, -2]
        # v -= grad(p)
        grad = np.zeros_like(v)
        grad[1:-1, 1:-1, 1:-1, 0] = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * self.dx[0])
        grad[1:-1, 1:-1, 1:-1, 1] = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * self.dx[1])
        grad[1:-1, 1:-1, 1:-1, 2] = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * self.dx[2])
        v = v - grad
        return v

    def _step(self) -> None:
        # Add inlet velocity condition
        self._apply_inlet(self.velocity)
        self._apply_solid_bc(self.velocity)

        # Advect velocity (semi-Lagrangian)
        v_adv = self._advect_field(self.velocity, self.velocity, self.dt)

        # Viscosity
        lap = self._laplacian(self.velocity)
        v_new = v_adv + self.nu * self.dt * lap

        # Pressure projection
        v_new = self._pressure_projection(v_new)

        # Apply boundary conditions again
        self._apply_inlet(v_new)
        self._apply_solid_bc(v_new)

        self.velocity = v_new

        # Advect scalar
        s_adv = self._advect_field(self.scalar, self.velocity, self.dt)
        # Source term
        src = self._source_rate(self.current_time)
        self.scalar = s_adv + src * self.dt
        self.scalar = np.clip(self.scalar, 0.0, 10.0)
        self.scalar[self.solid_mask] = 0.0

        self.current_time += self.dt

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time == 0.0:
            return self._frame0
        # Advance using internal substeps
        target = time
        while self.current_time < target - 1e-12:
            dt = min(self.dt, target - self.current_time)
            old_dt = self.dt
            self.dt = dt
            self._step()
            self.dt = old_dt
        # After reaching target, particles are diagnostic-only; keep them static
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "finite_difference_smoke",
            "grid_shape": [self.Nx, self.Ny, self.Nz],
            "dx_min": float(self.dx_min),
            "inlet_speed": float(self.inlet_speed),
        }
