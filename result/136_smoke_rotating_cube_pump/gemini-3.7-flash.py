"""Eulerian Grid-based Smoke Simulation Solver with Rotating Cube Obstacle.

Implements fluid flow driven by buoyancy and a rotating kinematic cube pump,
advecting smoke density according to the visual simulation of smoke formulation.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Domain settings
        self.nx, self.ny, self.nz = tuple(self.config["sampling"]["field_grid_shape"])
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)

        # Grid spacing
        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz
        self.spacing = float(self.config["sampling"]["spacing"])

        # Coordinate grids (cell centers)
        xc = np.linspace(self.bounds_min[0] + self.dx * 0.5, self.bounds_max[0] - self.dx * 0.5, self.nx)
        yc = np.linspace(self.bounds_min[1] + self.dy * 0.5, self.bounds_max[1] - self.dy * 0.5, self.ny)
        zc = np.linspace(self.bounds_min[2] + self.dz * 0.5, self.bounds_max[2] - self.dz * 0.5, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")

        # Physical states on collocated grid
        # Velocity [nx, ny, nz, 3]
        self.u = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        # Smoke density [nx, ny, nz]
        self.density = 1.20 * np.exp(-((self.X / 0.40) ** 2 + (self.Y / 0.40) ** 2 + ((self.Z - 0.68) / 0.62) ** 2))

        # Cube parameters
        self.cube_center = np.array([0.0, 0.0, 1.55], dtype=np.float64)
        self.cube_half_extents = np.array([0.45, 0.45, 0.45], dtype=np.float64)
        # Cube rotation axis: rotates around Y axis (or Z axis) to act as a paddle pump deflecting plume
        # In Fedkiw et al. Figure 6, a cube rotates around an axis perpendicular to the camera/gravity plane, e.g. Y-axis
        self.omega = np.array([0.0, 2.5, 0.0], dtype=np.float64)  # angular velocity rad/s

        # Smoke source mask / rate
        r_cyl = 0.34
        h_cyl = 0.12
        r_dist = np.sqrt(self.X**2 + self.Y**2)
        self.source_mask = (r_dist <= r_cyl) & (self.Z >= 0.0) & (self.Z <= h_cyl)
        self.scalar_source = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.scalar_source[self.source_mask] = 2.0

        # Boundary samples (for boundary fields export)
        self._init_boundary_samples()

        # Initial baseline frame for metadata
        self.init_frame = scene.initial_state()
        self.current_time = 0.0

    def _init_boundary_samples(self) -> None:
        # Generate boundary sample points on ground and cube faces
        samples = []
        normals = []

        # Ground plane samples (z = 0, facing +z)
        gx = np.linspace(-3.0, 3.0, 12)
        gy = np.linspace(-1.2, 1.2, 6)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        g_pts = np.column_stack([gxx.ravel(), gyy.ravel(), np.zeros(gxx.size)])
        g_norms = np.tile([0.0, 0.0, 1.0], (len(g_pts), 1))
        samples.append(g_pts)
        normals.append(g_norms)

        # Cube surface samples (approx 6 faces)
        face_pts = []
        face_norms = []
        n_per_side = 4
        s = np.linspace(-0.45, 0.45, n_per_side)
        s1, s2 = np.meshgrid(s, s, indexing="ij")
        s1 = s1.ravel()
        s2 = s2.ravel()

        # +x, -x
        face_pts.append(np.column_stack([np.full_like(s1, 0.45), s1, s2]))
        face_norms.append(np.tile([1.0, 0.0, 0.0], (len(s1), 1)))
        face_pts.append(np.column_stack([np.full_like(s1, -0.45), s1, s2]))
        face_norms.append(np.tile([-1.0, 0.0, 0.0], (len(s1), 1)))
        # +y, -y
        face_pts.append(np.column_stack([s1, np.full_like(s1, 0.45), s2]))
        face_norms.append(np.tile([0.0, 1.0, 0.0], (len(s1), 1)))
        face_pts.append(np.column_stack([s1, np.full_like(s1, -0.45), s2]))
        face_norms.append(np.tile([0.0, -1.0, 0.0], (len(s1), 1)))
        # +z, -z
        face_pts.append(np.column_stack([s1, s2, np.full_like(s1, 0.45)]))
        face_norms.append(np.tile([0.0, 0.0, 1.0], (len(s1), 1)))
        face_pts.append(np.column_stack([s1, s2, np.full_like(s1, -0.45)]))
        face_norms.append(np.tile([0.0, 0.0, -1.0], (len(s1), 1)))

        self.cube_local_pts = np.vstack(face_pts)
        self.cube_local_norms = np.vstack(face_norms)

        self.num_ground = len(g_pts)
        self.num_cube = len(self.cube_local_pts)
        self.ground_pts = g_pts
        self.ground_norms = g_norms

    def _get_cube_transform(self, t: float):
        # Rotation around Y axis by angle theta = omega * t
        theta = self.omega[1] * t
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotation matrix for Y-axis rotation
        R = np.array([
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float64)
        return R

    def _get_obstacle_mask_and_vel(self, t: float):
        R = self._get_cube_transform(t)
        R_inv = R.T  # Orthogonal transpose

        # Transform grid coordinates to cube local coordinates
        coords = np.stack([self.X - self.cube_center[0],
                           self.Y - self.cube_center[1],
                           self.Z - self.cube_center[2]], axis=-1)  # [nx, ny, nz, 3]

        # Local pos = coords @ R
        local_pos = np.einsum('ijkl,ml->ijkm', coords, R_inv)

        # Inside box check with slight margin
        mask = (np.abs(local_pos[..., 0]) <= self.cube_half_extents[0]) & \
               (np.abs(local_pos[..., 1]) <= self.cube_half_extents[1]) & \
               (np.abs(local_pos[..., 2]) <= self.cube_half_extents[2])

        # Obstacle velocity: v = omega x (r - r_center)
        # omega = [0, omega_y, 0]
        # v_x = omega_y * (Z - c_z)
        # v_y = 0
        # v_z = -omega_y * (X - c_x)
        obs_vel = np.zeros_like(self.u)
        obs_vel[..., 0] = self.omega[1] * (self.Z - self.cube_center[2])
        obs_vel[..., 1] = 0.0
        obs_vel[..., 2] = -self.omega[1] * (self.X - self.cube_center[0])

        return mask, obs_vel

    def _advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        # Semi-Lagrangian advection (3D)
        # field can be scalar [nx, ny, nz] or vector [nx, ny, nz, 3]
        # Backtrace
        prev_x = self.X - dt * self.u[..., 0]
        prev_y = self.Y - dt * self.u[..., 1]
        prev_z = self.Z - dt * self.u[..., 2]

        # Normalized coordinates in [0, nx-1], [0, ny-1], [0, nz-1]
        gx = (prev_x - (self.bounds_min[0] + self.dx * 0.5)) / self.dx
        gy = (prev_y - (self.bounds_min[1] + self.dy * 0.5)) / self.dy
        gz = (prev_z - (self.bounds_min[2] + self.dz * 0.5)) / self.dz

        gx = np.clip(gx, 0.0, self.nx - 1.001)
        gy = np.clip(gy, 0.0, self.ny - 1.001)
        gz = np.clip(gz, 0.0, self.nz - 1.001)

        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        k0 = np.floor(gz).astype(np.int32)

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        fx = gx - i0
        fy = gy - j0
        fz = gz - k0

        if field.ndim == 3:
            # Scalar interpolation
            c000 = field[i0, j0, k0]
            c100 = field[i1, j0, k0]
            c010 = field[i0, j1, k0]
            c110 = field[i1, j1, k0]
            c001 = field[i0, j0, k1]
            c101 = field[i1, j0, k1]
            c011 = field[i0, j1, k1]
            c111 = field[i1, j1, k1]

            c00 = c000 * (1 - fx) + c100 * fx
            c10 = c010 * (1 - fx) + c110 * fx
            c01 = c001 * (1 - fx) + c101 * fx
            c11 = c011 * (1 - fx) + c111 * fx

            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy

            return c0 * (1 - fz) + c1 * fz
        else:
            # Vector interpolation
            out = np.zeros_like(field)
            for d in range(field.shape[-1]):
                f_d = field[..., d]
                c000 = f_d[i0, j0, k0]
                c100 = f_d[i1, j0, k0]
                c010 = f_d[i0, j1, k0]
                c110 = f_d[i1, j1, k0]
                c001 = f_d[i0, j0, k1]
                c101 = f_d[i1, j0, k1]
                c011 = f_d[i0, j1, k1]
                c111 = f_d[i1, j1, k1]

                c00 = c000 * (1 - fx) + c100 * fx
                c10 = c010 * (1 - fx) + c110 * fx
                c01 = c001 * (1 - fx) + c101 * fx
                c11 = c011 * (1 - fx) + c111 * fx

                c0 = c00 * (1 - fy) + c10 * fy
                c1 = c01 * (1 - fy) + c11 * fy

                out[..., d] = c0 * (1 - fz) + c1 * fz
            return out

    def _project(self, u: np.ndarray, obs_mask: np.ndarray, obs_vel: np.ndarray) -> np.ndarray:
        # Pressure projection using Jacobi iteration
        # Enforce boundary velocity inside obstacle
        u = u.copy()
        u[obs_mask] = obs_vel[obs_mask]

        # Domain boundary conditions (solid walls)
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, 2] = 0.0

        # Compute divergence
        div = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1, 0] - u[:-2, 1:-1, 1:-1, 0]) / (2.0 * self.dx) +
            (u[1:-1, 2:, 1:-1, 1] - u[1:-1, :-2, 1:-1, 1]) / (2.0 * self.dy) +
            (u[1:-1, 1:-1, 2:, 2] - u[1:-1, 1:-1, :-2, 2]) / (2.0 * self.dz)
        )

        p = np.zeros_like(div)
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        inv_dz2 = 1.0 / (self.dz * self.dz)
        denom = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2)

        # Jacobi iterations
        fluid_mask = (~obs_mask)
        for _ in range(25):
            p_new = p.copy()
            p_new[1:-1, 1:-1, 1:-1] = (
                (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * inv_dx2 +
                (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * inv_dy2 +
                (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * inv_dz2 -
                div[1:-1, 1:-1, 1:-1]
            ) / denom

            # Boundary conditions for pressure (Neumann dp/dn = 0)
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]

            p = np.where(fluid_mask, p_new, 0.0)

        # Subtract pressure gradient from velocity
        grad_p = np.zeros_like(u)
        grad_p[1:-1, 1:-1, 1:-1, 0] = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        grad_p[1:-1, 1:-1, 1:-1, 1] = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        grad_p[1:-1, 1:-1, 1:-1, 2] = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * self.dz)

        u = u - grad_p

        # Re-enforce obstacle velocity
        u[obs_mask] = obs_vel[obs_mask]
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, 2] = 0.0

        return u

    def _step(self, dt: float) -> None:
        t = self.current_time
        obs_mask, obs_vel = self._get_obstacle_mask_and_vel(t)

        # 1. Source injection: continuous smoke emission
        self.density[self.source_mask] = np.maximum(self.density[self.source_mask], 1.2)
        # Smoke source imparts upward velocity
        self.u[self.source_mask, 2] = np.maximum(self.u[self.source_mask, 2], 1.2)

        # 2. Buoyancy force (proportional to smoke density)
        # f_buoyancy = alpha * density * [0, 0, 1]
        alpha = 1.5
        self.u[..., 2] += alpha * self.density * dt

        # 3. Advect velocity
        self.u = self._advect(self.u, dt)

        # 4. Project velocity to make it divergence-free and respect obstacle
        self.u = self._project(self.u, obs_mask, obs_vel)

        # 5. Advect smoke density
        self.density = self._advect(self.density, dt)
        self.density[obs_mask] = 0.0  # Smoke cannot penetrate obstacle
        self.density = np.clip(self.density, 0.0, 5.0)

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Substep to reach target time
        sub_dt = 0.01
        while self.current_time < time - 1e-6:
            step_dt = min(sub_dt, time - self.current_time)
            self._step(step_dt)

        # Compute diagnostic boundary outputs at current time
        R = self._get_cube_transform(self.current_time)
        cube_pts = self.cube_local_pts @ R.T + self.cube_center
        cube_norms = self.cube_local_norms @ R.T

        # Cube velocity at sample points: omega x (r - c)
        cube_v = np.zeros_like(cube_pts)
        cube_v[:, 0] = self.omega[1] * (cube_pts[:, 2] - self.cube_center[2])
        cube_v[:, 1] = 0.0
        cube_v[:, 2] = -self.omega[1] * (cube_pts[:, 0] - self.cube_center[0])

        ground_v = np.zeros_like(self.ground_pts)

        b_pts = np.vstack([self.ground_pts, cube_pts])
        b_norms = np.vstack([self.ground_norms, cube_norms])
        b_v = np.vstack([ground_v, cube_v])

        # Active mask: fluid cells where simulation applies
        obs_mask, _ = self._get_obstacle_mask_and_vel(self.current_time)
        active_mask = ~obs_mask

        fields = {
            "active_mask": active_mask,
            "boundary_normal": b_norms,
            "boundary_target_velocity": b_v,
            "boundary_velocity": b_v,
            "scalar_source": self.scalar_source.copy(),
            "transported_scalar": self.density.copy(),
            "velocity_field": self.u.copy(),
            "cumulative_mass_source": self.init_frame.fields["cumulative_mass_source"].copy()
        }

        scalars = {
            "contact_length_scale": self.spacing,
            "grid_spacing": self.spacing,
            "characteristic_length": float(self.init_frame.scalars["characteristic_length"]),
            "reference_velocity": 1.0
        }

        frame = StateFrame(
            time=time,
            position=self.init_frame.position.copy(),
            velocity=self.init_frame.velocity.copy(),
            mass=self.init_frame.mass.copy(),
            body_id=self.init_frame.body_id.copy(),
            attributes={k: v.copy() if hasattr(v, "copy") else v for k, v in self.init_frame.attributes.items()},
            fields=fields,
            scalars=scalars
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {}

