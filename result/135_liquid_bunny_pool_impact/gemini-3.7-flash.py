"""Position-Based Fluids (PBF) solver for Liquid Bunny Pool Impact."""
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

        init_frame = scene.initial_state()
        self.pos = np.ascontiguousarray(init_frame.position, dtype=np.float64)
        self.vel = np.ascontiguousarray(init_frame.velocity, dtype=np.float64)
        self.mass = np.ascontiguousarray(init_frame.mass, dtype=np.float64)
        self.body_id = np.ascontiguousarray(init_frame.body_id, dtype=np.int32)
        self.material_label = np.ascontiguousarray(
            init_frame.attributes.get("material_label", self.body_id), dtype=np.int32
        )
        self.mass_quantity = np.ascontiguousarray(
            init_frame.attributes.get("mass_quantity", self.mass), dtype=np.float64
        )
        self.num_particles = len(self.pos)
        self.current_time = 0.0

        # Physical parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        sampling = self.config.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.14))
        self.rest_density = float(sampling.get("density", 1000.0))

        # Domain & Grid bounds from scene_geometry / config
        # pool domain bounds: min [-3.0, -1.2, 0.0], max [3.0, 1.2, 3.8]
        # grid shape: [24, 16, 28]
        self.grid_shape = tuple(sampling.get("field_grid_shape", [24, 16, 28]))
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        self.grid_dx = (self.bounds_max - self.bounds_min) / np.array(self.grid_shape, dtype=np.float64)
        self.grid_spacing = float(self.grid_dx.mean())

        # Rigid boundaries: basin walls and ground
        # ground z >= 0.0
        # basin x: [-1.85 + 0.08, 1.85 - 0.08] -> [-1.77, 1.77]
        # basin y: [-0.84, 0.84] (back wall at y=0.92, half_extents 0.08 -> 0.84 inner, front wall open or at -0.84)
        self.x_min, self.x_max = -1.77, 1.77
        self.y_min, self.y_max = -0.84, 0.84
        self.z_min = 0.0

        # PBF parameters
        self.h = self.spacing * 1.5  # smoothing radius
        self.h2 = self.h * self.h
        self.poly6_factor = 315.0 / (64.0 * np.pi * (self.h ** 9))
        self.spiky_factor = -45.0 / (np.pi * (self.h ** 6))
        self.w_zero = self.poly6_factor * (self.h ** 6)

        # Estimate rest density factor or compute from particle volume
        particle_vol = self.spacing ** 3
        self.p_mass = self.rest_density * particle_vol
        self.eps_relaxation = 600.0
        self.sub_dt = 0.0033333333333333335  # 10 sub-steps per sample_dt (0.03333s)
        self.pbf_iters = 3

        # Boundary samples for trajectory contract
        # Sample boundary walls
        bx = np.linspace(self.x_min, self.x_max, 15)
        by = np.linspace(self.y_min, self.y_max, 10)
        bz = np.linspace(self.z_min, 0.64, 6)

        b_pts = []
        b_norms = []
        # Ground samples
        gx, gy = np.meshgrid(bx, by, indexing="ij")
        for x, y in zip(gx.ravel(), gy.ravel()):
            b_pts.append([x, y, self.z_min])
            b_norms.append([0.0, 0.0, 1.0])
        # Left wall (x = x_min)
        wy, wz = np.meshgrid(by, bz, indexing="ij")
        for y, z in zip(wy.ravel(), wz.ravel()):
            b_pts.append([self.x_min, y, z])
            b_norms.append([1.0, 0.0, 0.0])
        # Right wall (x = x_max)
        for y, z in zip(wy.ravel(), wz.ravel()):
            b_pts.append([self.x_max, y, z])
            b_norms.append([-1.0, 0.0, 0.0])
        # Back wall (y = y_max)
        wx, wz = np.meshgrid(bx, bz, indexing="ij")
        for x, z in zip(wx.ravel(), wz.ravel()):
            b_pts.append([x, self.y_max, z])
            b_norms.append([0.0, -1.0, 0.0])
        # Front wall (y = y_min)
        for x, z in zip(wx.ravel(), wz.ravel()):
            b_pts.append([x, self.y_min, z])
            b_norms.append([0.0, 1.0, 0.0])

        self.boundary_pts = np.array(b_pts, dtype=np.float64)
        self.boundary_normals = np.array(b_norms, dtype=np.float64)
        self.num_boundary = len(self.boundary_pts)

        # Spatial hash grid parameters for neighbor search
        self.cell_size = self.h
        self.grid_dim = np.ceil((self.bounds_max - self.bounds_min) / self.cell_size).astype(int)

    def _build_spatial_grid(self, pos: np.ndarray):
        # Discretize particle positions into cell indices
        cell_coords = np.floor((pos - self.bounds_min) / self.cell_size).astype(np.int32)
        np.clip(cell_coords, 0, self.grid_dim - 1, out=cell_coords)
        cell_keys = (
            cell_coords[:, 0]
            + cell_coords[:, 1] * self.grid_dim[0]
            + cell_coords[:, 2] * (self.grid_dim[0] * self.grid_dim[1])
        )
        sort_order = np.argsort(cell_keys)
        sorted_keys = cell_keys[sort_order]
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        cell_dict = {}
        for i, key in enumerate(unique_keys):
            start = split_indices[i]
            end = split_indices[i + 1] if i + 1 < len(split_indices) else len(sort_order)
            cell_dict[key] = sort_order[start:end]
        return cell_dict

    def _get_neighbors(self, pos: np.ndarray, cell_dict: dict):
        cell_coords = np.floor((pos - self.bounds_min) / self.cell_size).astype(np.int32)
        np.clip(cell_coords, 0, self.grid_dim - 1, out=cell_coords)

        # Find candidate particle pairs efficiently
        gx_stride = 1
        gy_stride = self.grid_dim[0]
        gz_stride = self.grid_dim[0] * self.grid_dim[1]

        # For vectorized neighbor search, gather pairs
        src_list = []
        dst_list = []

        # Loop through active cells in cell_dict
        for key, p_indices in cell_dict.items():
            cx = key % self.grid_dim[0]
            cy = (key // self.grid_dim[0]) % self.grid_dim[1]
            cz = key // (self.grid_dim[0] * self.grid_dim[1])

            # Check 3x3x3 neighboring cells
            for dx in (-1, 0, 1):
                nx = cx + dx
                if nx < 0 or nx >= self.grid_dim[0]:
                    continue
                for dy in (-1, 0, 1):
                    ny = cy + dy
                    if ny < 0 or ny >= self.grid_dim[1]:
                        continue
                    for dz in (-1, 0, 1):
                        nz = cz + dz
                        if nz < 0 or nz >= self.grid_dim[2]:
                            continue
                        n_key = nx * gx_stride + ny * gy_stride + nz * gz_stride
                        if n_key in cell_dict:
                            n_indices = cell_dict[n_key]
                            if n_key == key:
                                # Within same cell: all unordered pairs
                                if len(p_indices) > 1:
                                    i_grid, j_grid = np.meshgrid(p_indices, p_indices, indexing="ij")
                                    mask = i_grid < j_grid
                                    src_list.append(i_grid[mask])
                                    dst_list.append(j_grid[mask])
                            elif key < n_key:
                                # Distinct cells: cartesian product once
                                i_grid, j_grid = np.meshgrid(p_indices, n_indices, indexing="ij")
                                src_list.append(i_grid.ravel())
                                dst_list.append(j_grid.ravel())

        if src_list:
            all_src = np.concatenate(src_list)
            all_dst = np.concatenate(dst_list)
            rij = pos[all_src] - pos[all_dst]
            r2 = np.sum(rij * rij, axis=-1)
            valid = r2 < self.h2
            return all_src[valid], all_dst[valid], rij[valid], np.sqrt(r2[valid] + 1e-12)
        else:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, np.empty((0, 3)), np.empty(0)

    def _step(self, dt: float):
        # 1. Apply external forces & predict position
        self.vel += self.gravity * dt
        # Viscous drag / damping
        self.vel *= 0.9995
        pos_pred = self.pos + self.vel * dt

        # Boundary collision on predicted position
        r = self.spacing * 0.4
        np.clip(pos_pred[:, 0], self.x_min + r, self.x_max - r, out=pos_pred[:, 0])
        np.clip(pos_pred[:, 1], self.y_min + r, self.y_max - r, out=pos_pred[:, 1])
        pos_pred[:, 2] = np.maximum(pos_pred[:, 2], self.z_min + r)

        # 2. PBF density solver loop
        for _ in range(self.pbf_iters):
            cell_dict = self._build_spatial_grid(pos_pred)
            src, dst, rij, r_dist = self._get_neighbors(pos_pred, cell_dict)
            if len(src) == 0:
                break

            # Kernel W(r)
            h_sq_minus_r_sq = np.maximum(0.0, self.h2 - r_dist * r_dist)
            w_ij = self.poly6_factor * (h_sq_minus_r_sq ** 3)

            # Compute density rho_i
            density = np.full(self.num_particles, self.w_zero * self.p_mass)
            np.add.at(density, src, self.p_mass * w_ij)
            np.add.at(density, dst, self.p_mass * w_ij)

            # Constraint C_i = rho_i / rho_0 - 1
            C_i = np.maximum(0.0, density / self.rest_density - 1.0)

            # Gradient of Spiky kernel: grad_W = spiky_factor * (h - r)^2 * (rij / r)
            h_minus_r = np.maximum(0.0, self.h - r_dist)
            grad_factor = self.spiky_factor * (h_minus_r ** 2) / (r_dist + 1e-8)
            grad_w = rij * grad_factor[:, None] / self.rest_density  # [E, 3]

            # Sum of squared gradient magnitudes per particle
            grad_sq = np.zeros(self.num_particles)
            grad_sq_contrib = np.sum(grad_w * grad_w, axis=-1)
            np.add.at(grad_sq, src, grad_sq_contrib)
            np.add.at(grad_sq, dst, grad_sq_contrib)

            # Self-gradient vector: sum_j grad_W_ij
            grad_self = np.zeros((self.num_particles, 3))
            np.add.at(grad_self, src, grad_w)
            np.add.at(grad_self, dst, -grad_w)
            grad_sq += np.sum(grad_self * grad_self, axis=-1)

            # Lagrange multiplier lambda_i
            lambda_i = -C_i / (grad_sq + self.eps_relaxation)

            # Position delta: dp_i = sum_j (lambda_i + lambda_j + s_corr) * grad_W
            # Tensile instability artificial pressure s_corr = -k * (W(r) / W(0.1*h))^4
            w_dq = self.poly6_factor * (np.maximum(0.0, self.h2 - (0.2 * self.h) ** 2) ** 3)
            s_corr = -0.001 * ((w_ij / (w_dq + 1e-8)) ** 4)
            lambda_sum = (lambda_i[src] + lambda_i[dst] + s_corr)[:, None]
            dp = lambda_sum * grad_w

            delta_pos = np.zeros((self.num_particles, 3))
            np.add.at(delta_pos, src, dp)
            np.add.at(delta_pos, dst, -dp)

            pos_pred += delta_pos

            # Enforce boundary constraints
            np.clip(pos_pred[:, 0], self.x_min + r, self.x_max - r, out=pos_pred[:, 0])
            np.clip(pos_pred[:, 1], self.y_min + r, self.y_max - r, out=pos_pred[:, 1])
            pos_pred[:, 2] = np.maximum(pos_pred[:, 2], self.z_min + r)

        # 3. Update velocity from position change
        self.vel = (pos_pred - self.pos) / dt
        self.pos = pos_pred

        # Boundary restitution & friction
        near_x = (self.pos[:, 0] <= self.x_min + r + 1e-4) | (self.pos[:, 0] >= self.x_max - r - 1e-4)
        near_y = (self.pos[:, 1] <= self.y_min + r + 1e-4) | (self.pos[:, 1] >= self.y_max - r - 1e-4)
        near_z = self.pos[:, 2] <= self.z_min + r + 1e-4

        self.vel[near_x, 0] *= -0.1
        self.vel[near_y, 1] *= -0.1
        self.vel[near_z, 2] = np.maximum(0.0, self.vel[near_z, 2] * -0.1)

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until reaching the requested time
        while self.current_time + 1e-8 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Build diagnostic Eulerian velocity grid, active mask, and liquid level set
        nx, ny, nz = self.grid_shape
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        count_field = np.zeros((nx, ny, nz), dtype=np.float64)

        # Interpolate particle velocities to Eulerian grid
        cell_coords = np.floor((self.pos - self.bounds_min) / self.grid_dx).astype(np.int32)
        valid = (
            (cell_coords[:, 0] >= 0) & (cell_coords[:, 0] < nx) &
            (cell_coords[:, 1] >= 0) & (cell_coords[:, 1] < ny) &
            (cell_coords[:, 2] >= 0) & (cell_coords[:, 2] < nz)
        )
        c_valid = cell_coords[valid]
        v_valid = self.vel[valid]

        np.add.at(velocity_field, (c_valid[:, 0], c_valid[:, 1], c_valid[:, 2]), v_valid)
        np.add.at(count_field, (c_valid[:, 0], c_valid[:, 1], c_valid[:, 2]), 1.0)

        mask = count_field > 0
        velocity_field[mask] /= count_field[mask, None]

        active_mask = mask

        # Approximate signed liquid level set: negative inside fluid, positive outside
        # Using a fast grid distance proxy based on active cells
        from scipy.ndimage import distance_transform_edt
        if np.any(active_mask):
            dist_outside = distance_transform_edt(~active_mask)
            dist_inside = distance_transform_edt(active_mask)
            liquid_level_set = (dist_outside - dist_inside) * self.grid_spacing
        else:
            liquid_level_set = np.ones((nx, ny, nz), dtype=np.float64) * 10.0

        # Boundary fields
        boundary_norm = self.boundary_normals.copy()
        boundary_vel = np.zeros_like(boundary_norm)
        boundary_target_vel = np.zeros_like(boundary_norm)

        # Cumulative mass source (K materials: K=1 or K=2)
        # 1 conserved material / initial pool & bunny mass
        cumulative_mass_source = np.zeros(2, dtype=np.float64)

        char_len = max(float(np.ptp(self.pos, axis=0).max()), self.spacing)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": np.full(self.num_particles, 0.45 * self.spacing),
                "material_label": self.material_label.copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], self.num_particles, axis=0),
            },
            fields={
                "active_mask": active_mask,
                "boundary_normal": boundary_norm,
                "boundary_target_velocity": boundary_target_vel,
                "boundary_velocity": boundary_vel,
                "cumulative_mass_source": cumulative_mass_source,
                "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(char_len),
                "grid_spacing": float(self.grid_spacing),
                "reference_velocity": 1.0,
                "contact_length_scale": float(self.spacing),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": self.num_particles,
            "current_time": self.current_time,
            "mean_speed": float(np.linalg.norm(self.vel, axis=-1).mean()) if len(self.vel) > 0 else 0.0,
        }

