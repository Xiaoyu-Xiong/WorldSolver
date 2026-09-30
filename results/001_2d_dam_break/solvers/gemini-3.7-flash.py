"""BenchmarkSolver for 2D Dam Break simulation using APIC (Affine Particle-in-Cell)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Get initial state
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.initial_frame = init_frame

        # Gravity
        grav = self.config.get("gravity", [0.0, 0.0, -9.81])
        self.gx = float(grav[0])
        self.gz = float(grav[2])

        # Particles
        # Position in 3D: [N, 3], strictly 2D motion in x-z plane (y = 0)
        self.pos = np.copy(init_frame.position) # [N, 3]
        self.vel = np.copy(init_frame.velocity) # [N, 3]
        self.mass = np.copy(init_frame.mass)     # [N]
        self.n_particles = self.pos.shape[0]

        # APIC affine momentum matrix C for each particle: 2x2 matrix (in x-z)
        self.C = np.zeros((self.n_particles, 2, 2), dtype=np.float64)

        # Simulation domain & boundaries (x in [-3.0, 3.0], z in [0.05, 3.0])
        # Tank floor top is z = 0.05 (half extent 0.05, center 0.0)
        # Left wall right face is x = -3.0 (half extent 0.05, center -3.05)
        # Right wall left face is x = 3.0 (half extent 0.05, center 3.05)
        self.x_min, self.x_max = -3.0, 3.0
        self.z_min, self.z_max = 0.05, 3.0

        # Characteristic scales
        self.char_length = 2.4  # height of column = 2.4m
        self.ref_velocity = np.sqrt(2.0 * 9.81 * self.char_length) # ~6.86 m/s

        # Grid setup for APIC / diagnostic grid
        self.dx = 0.05  # grid resolution
        self.inv_dx = 1.0 / self.dx

        # Grid bounds covering the tank
        self.grid_x_min = -3.0
        self.grid_z_min = 0.0
        self.nx = int(np.round((self.x_max - self.x_min) / self.dx)) + 1  # 121 nodes
        self.nz = int(np.round((3.0 - 0.0) / self.dx)) + 1                 # 61 nodes

        # Diagnostic grid coordinates
        x_coords = np.linspace(self.grid_x_min, self.x_max, self.nx)
        z_coords = np.linspace(self.grid_z_min, 3.0, self.nz)
        self.grid_xx, self.grid_zz = np.meshgrid(x_coords, z_coords, indexing="ij") # [nx, nz]

        # Boundary samples setup
        # Floor: z = 0.05, x in [-3.0, 3.0]
        # Left wall: x = -3.0, z in [0.05, 3.0]
        # Right wall: x = 3.0, z in [0.05, 3.0]
        b_x = np.linspace(-3.0, 3.0, 121)
        b_z = np.linspace(0.05, 3.0, 60)

        # Floor samples (normal points +z)
        floor_pts = np.column_stack([b_x, np.zeros_like(b_x), np.full_like(b_x, 0.05)])
        floor_norm = np.zeros_like(floor_pts)
        floor_norm[:, 2] = 1.0

        # Left wall samples (normal points +x)
        left_pts = np.column_stack([np.full_like(b_z, -3.0), np.zeros_like(b_z), b_z])
        left_norm = np.zeros_like(left_pts)
        left_norm[:, 0] = 1.0

        # Right wall samples (normal points -x)
        right_pts = np.column_stack([np.full_like(b_z, 3.0), np.zeros_like(b_z), b_z])
        right_norm = np.zeros_like(right_pts)
        right_norm[:, 0] = -1.0

        self.boundary_pts = np.vstack([floor_pts, left_pts, right_pts])
        self.boundary_norm = np.vstack([floor_norm, left_norm, right_norm])
        self.n_boundary = self.boundary_pts.shape[0]

        # Substep dt
        self.sub_dt = 0.002

    def _step(self, dt: float) -> None:
        """Advance one APIC step with Tait equation of state (weakly compressible SPH / MPM)."""
        px = self.pos[:, 0]
        pz = self.pos[:, 2]
        vx = self.vel[:, 0]
        vz = self.vel[:, 2]
        C = self.C # [N, 2, 2]

        # Fluid parameters for weakly compressible APIC
        rho0 = 1000.0
        # Speed of sound ~ 40 m/s for incompressibility at dam break speeds (~7 m/s)
        c0 = 40.0
        gamma = 7.0
        B = rho0 * (c0 ** 2) / gamma

        # Grid arrays
        grid_v = np.zeros((self.nx, self.nz, 2), dtype=np.float64)
        grid_m = np.zeros((self.nx, self.nz), dtype=np.float64)

        # Base grid indices for particles (B-spline quadratic or linear)
        # Using quadratic B-spline kernel (standard APIC/MLS-MPM)
        gx = (px - self.grid_x_min) * self.inv_dx
        gz = (pz - self.grid_z_min) * self.inv_dx

        base_i = np.floor(gx - 0.5).astype(np.int32)
        base_k = np.floor(gz - 0.5).astype(np.int32)

        fx = gx - base_i.astype(np.float64)
        fz = gz - base_k.astype(np.float64)

        # Quadratic weights
        # w[0] = 0.5 * (1.5 - f)^2
        # w[1] = 0.75 - (f - 1)^2
        # w[2] = 0.5 * (f - 0.5)^2
        w_x = np.stack([
            0.5 * (1.5 - fx)**2,
            0.75 - (fx - 1.0)**2,
            0.5 * (fx - 0.5)**2
        ], axis=-1) # [N, 3]

        w_z = np.stack([
            0.5 * (1.5 - fz)**2,
            0.75 - (fz - 1.0)**2,
            0.5 * (fz - 0.5)**2
        ], axis=-1) # [N, 3]

        # P2G: Transfer particle mass and momentum to grid
        # APIC affine transfer: affine = C_p * dpos
        # D_inv = 4 / (dx^2) for quadratic B-spline
        D_inv = 4.0 * (self.inv_dx ** 2)

        p_mass = self.mass # [N]

        for i in range(3):
            for k in range(3):
                node_i = base_i + i
                node_k = base_k + k

                # weight
                weight = w_x[:, i] * w_z[:, k] # [N]

                # dpos = (node_pos - particle_pos)
                dpos_x = (node_i.astype(np.float64) - gx) * self.dx
                dpos_z = (node_k.astype(np.float64) - gz) * self.dx

                # Affine velocity contribution: C_p @ dpos
                affine_vx = C[:, 0, 0] * dpos_x + C[:, 0, 1] * dpos_z
                affine_vz = C[:, 1, 0] * dpos_x + C[:, 1, 1] * dpos_z

                mv_x = weight * p_mass * (vx + affine_vx)
                mv_z = weight * p_mass * (vz + affine_vz)
                m_contrib = weight * p_mass

                # Mask valid grid bounds
                valid = (node_i >= 0) & (node_i < self.nx) & (node_k >= 0) & (node_k < self.nz)
                np.add.at(grid_m, (node_i[valid], node_k[valid]), m_contrib[valid])
                np.add.at(grid_v[:, :, 0], (node_i[valid], node_k[valid]), mv_x[valid])
                np.add.at(grid_v[:, :, 1], (node_i[valid], node_k[valid]), mv_z[valid])

        # Grid normalization
        nonzero_mask = grid_m > 1e-12
        grid_v[nonzero_mask] /= grid_m[nonzero_mask, None]

        # Compute fluid pressure / density on particles to apply pressure force on grid
        # Particle density estimation from grid
        p_rho = np.zeros(self.n_particles, dtype=np.float64)
        for i in range(3):
            for k in range(3):
                node_i = base_i + i
                node_k = base_k + k
                weight = w_x[:, i] * w_z[:, k]
                valid = (node_i >= 0) & (node_i < self.nx) & (node_k >= 0) & (node_k < self.nz)
                node_i_clamped = np.clip(node_i, 0, self.nx - 1)
                node_k_clamped = np.clip(node_k, 0, self.nz - 1)
                p_rho += weight * grid_m[node_i_clamped, node_k_clamped] * valid

        cell_vol = self.dx * self.dx
        p_rho = p_rho / cell_vol
        p_rho = np.maximum(p_rho, rho0 * 0.1)

        # Tait equation of state: pressure = B * ((rho / rho0)^gamma - 1)
        # Clamped to non-negative (cavitation / free surface)
        p_pressure = B * np.maximum(0.0, (p_rho / rho0) ** gamma - 1.0)

        # Stress contribution to grid forces:
        # Cauchy stress sigma = -p * I
        # Force on grid node = - sum_p V_p * sigma_p * D_inv * dpos * w_p
        # = sum_p (p_mass / p_rho) * p_pressure * D_inv * dpos * w_p
        stress_coeff = (p_mass / p_rho) * p_pressure * (D_inv * dt)

        for i in range(3):
            for k in range(3):
                node_i = base_i + i
                node_k = base_k + k
                weight = w_x[:, i] * w_z[:, k]
                dpos_x = (node_i.astype(np.float64) - gx) * self.dx
                dpos_z = (node_k.astype(np.float64) - gz) * self.dx

                force_x = weight * stress_coeff * dpos_x
                force_y = weight * stress_coeff * dpos_z

                valid = (node_i >= 0) & (node_i < self.nx) & (node_k >= 0) & (node_k < self.nz)
                np.add.at(grid_v[:, :, 0], (node_i[valid], node_k[valid]), force_x[valid] / np.maximum(grid_m[node_i[valid], node_k[valid]], 1e-12))
                np.add.at(grid_v[:, :, 1], (node_i[valid], node_k[valid]), force_y[valid] / np.maximum(grid_m[node_i[valid], node_k[valid]], 1e-12))

        # Apply gravity on grid
        grid_v[:, :, 0] += dt * self.gx
        grid_v[:, :, 1] += dt * self.gz

        # Grid boundary conditions (solid walls)
        # Floor: z <= 0.05
        # Left wall: x <= -3.0
        # Right wall: x >= 3.0
        # Free-slip / sticky wall boundary enforcement
        for ix in range(self.nx):
            x_val = self.grid_x_min + ix * self.dx
            for iz in range(self.nz):
                z_val = self.grid_z_min + iz * self.dx
                if z_val <= 0.05:
                    if grid_v[ix, iz, 1] < 0:
                        grid_v[ix, iz, 1] = 0.0
                if x_val <= -3.0:
                    if grid_v[ix, iz, 0] < 0:
                        grid_v[ix, iz, 0] = 0.0
                if x_val >= 3.0:
                    if grid_v[ix, iz, 0] > 0:
                        grid_v[ix, iz, 0] = 0.0

        # G2P: Transfer grid velocity and affine momentum back to particles
        new_vx = np.zeros(self.n_particles, dtype=np.float64)
        new_vz = np.zeros(self.n_particles, dtype=np.float64)
        new_C = np.zeros((self.n_particles, 2, 2), dtype=np.float64)

        for i in range(3):
            for k in range(3):
                node_i = base_i + i
                node_k = base_k + k
                weight = w_x[:, i] * w_z[:, k]
                dpos_x = (node_i.astype(np.float64) - gx) * self.dx
                dpos_z = (node_k.astype(np.float64) - gz) * self.dx

                valid = (node_i >= 0) & (node_i < self.nx) & (node_k >= 0) & (node_k < self.nz)
                node_i_clamped = np.clip(node_i, 0, self.nx - 1)
                node_k_clamped = np.clip(node_k, 0, self.nz - 1)

                g_vx = grid_v[node_i_clamped, node_k_clamped, 0] * valid
                g_vz = grid_v[node_i_clamped, node_k_clamped, 1] * valid

                new_vx += weight * g_vx
                new_vz += weight * g_vz

                # APIC C update: C_p = sum_i w_ip * v_i * (x_i - x_p)^T * D_inv
                term = weight * D_inv
                new_C[:, 0, 0] += term * g_vx * dpos_x
                new_C[:, 0, 1] += term * g_vx * dpos_z
                new_C[:, 1, 0] += term * g_vz * dpos_x
                new_C[:, 1, 1] += term * g_vz * dpos_z

        self.vel[:, 0] = new_vx
        self.vel[:, 1] = 0.0
        self.vel[:, 2] = new_vz
        self.C = new_C

        # Advect particles
        self.pos[:, 0] += dt * self.vel[:, 0]
        self.pos[:, 2] += dt * self.vel[:, 2]

        # Particle-level boundary collision / penalty
        r_padding = 0.02
        # Left boundary
        mask_l = self.pos[:, 0] < (self.x_min + r_padding)
        self.pos[mask_l, 0] = self.x_min + r_padding
        self.vel[mask_l, 0] = np.maximum(0.0, self.vel[mask_l, 0])

        # Right boundary
        mask_r = self.pos[:, 0] > (self.x_max - r_padding)
        self.pos[mask_r, 0] = self.x_max - r_padding
        self.vel[mask_r, 0] = np.minimum(0.0, self.vel[mask_r, 0])

        # Floor boundary
        mask_b = self.pos[:, 2] < (self.z_min + r_padding)
        self.pos[mask_b, 2] = self.z_min + r_padding
        self.vel[mask_b, 2] = np.maximum(0.0, self.vel[mask_b, 2])

        # Top boundary limit (if fluid runs up too high)
        mask_t = self.pos[:, 2] > 3.0
        self.pos[mask_t, 2] = 3.0
        self.vel[mask_t, 2] = np.minimum(0.0, self.vel[mask_t, 2])

    def advance_to(self, target_time: float) -> StateFrame:
        # Step simulation forward to target_time
        while self.time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.time)
            self._step(dt)
            self.time += dt

        self.time = target_time
        return self._build_frame(self.time)

    def _build_frame(self, time: float) -> StateFrame:
        """Construct the complete StateFrame with all required trajectory fields."""
        # Diagnostic grid: [nx, nz]
        # Interpolate particle velocity and signed level set to diagnostic grid
        # Velocity field: [nx, nz, 2]
        # Level set: signed distance to liquid surface, negative inside, positive outside
        # Liquid interior mask: 1 inside, 0 outside

        # Fast SPH-like kernel / binning for diagnostic field computation
        px = self.pos[:, 0]
        pz = self.pos[:, 2]
        vx = self.vel[:, 0]
        vz = self.vel[:, 2]

        # Grid coordinates
        gx = (px - self.grid_x_min) * self.inv_dx
        gz = (pz - self.grid_z_min) * self.inv_dx

        base_i = np.clip(np.round(gx).astype(np.int32), 0, self.nx - 1)
        base_k = np.clip(np.round(gz).astype(np.int32), 0, self.nz - 1)

        grid_vx = np.zeros((self.nx, self.nz), dtype=np.float64)
        grid_vz = np.zeros((self.nx, self.nz), dtype=np.float64)
        grid_density = np.zeros((self.nx, self.nz), dtype=np.float64)

        # Splat particle velocities to diagnostic grid
        kernel_r = 1.5 # in cell units
        for di in range(-2, 3):
            for dk in range(-2, 3):
                ni = base_i + di
                nk = base_k + dk
                valid = (ni >= 0) & (ni < self.nx) & (nk >= 0) & (nk < self.nz)
                if not np.any(valid):
                    continue
                dx_p = (ni[valid] - gx[valid])
                dz_p = (nk[valid] - gz[valid])
                r2 = dx_p**2 + dz_p**2
                w = np.maximum(0.0, (kernel_r**2 - r2)**2)
                np.add.at(grid_vx, (ni[valid], nk[valid]), w * vx[valid])
                np.add.at(grid_vz, (ni[valid], nk[valid]), w * vz[valid])
                np.add.at(grid_density, (ni[valid], nk[valid]), w)

        nonzero = grid_density > 1e-6
        grid_vx[nonzero] /= grid_density[nonzero]
        grid_vz[nonzero] /= grid_density[nonzero]

        velocity_field = np.stack([grid_vx, grid_vz], axis=-1) # [nx, nz, 2]

        # Liquid interior mask: based on smoothed particle density threshold
        liquid_interior_mask = (grid_density > 0.5).astype(np.float64)

        # Signed liquid level set: approximate signed distance
        # Positive outside, negative inside
        # Using density / distance approximation
        # For each grid point, min distance to particles minus effective particle radius
        # Approximate with density: level_set = (threshold - density) * scale
        # Or direct accurate distance field:
        from scipy.ndimage import distance_transform_edt
        inside_bool = liquid_interior_mask > 0.5
        if np.any(inside_bool) and not np.all(inside_bool):
            d_out = distance_transform_edt(~inside_bool) * self.dx
            d_in = distance_transform_edt(inside_bool) * self.dx
            liquid_level_set = d_out - d_in
        elif np.all(inside_bool):
            liquid_level_set = -distance_transform_edt(inside_bool) * self.dx
        else:
            liquid_level_set = distance_transform_edt(~inside_bool) * self.dx

        # Boundary velocities
        boundary_vel = np.zeros((self.n_boundary, 3), dtype=np.float64)
        boundary_target_vel = np.zeros((self.n_boundary, 3), dtype=np.float64)

        phase_id = np.ones(self.n_particles, dtype=np.int32)

        frame = StateFrame(
            time=time,
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            attributes={
                "phase_id": phase_id,
            },
            fields={
                "boundary_normal": self.boundary_norm,
                "boundary_target_velocity": boundary_target_vel,
                "boundary_velocity": boundary_vel,
                "liquid_interior_mask": liquid_interior_mask,
                "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(self.char_length),
                "grid_spacing": float(self.dx),
                "liquid_phase_id": 1,
                "reference_velocity": float(self.ref_velocity),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": self.n_particles,
            "grid_resolution": [self.nx, self.nz],
        }

