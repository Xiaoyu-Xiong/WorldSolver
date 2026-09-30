"""Physics simulation solver for 3D rotating blade stirring liquid."""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Get initial state from scene
        self.initial_frame = scene.initial_state()
        self.current_time = 0.0

        # Fluid particles state (all points except the last one which is the rigid blade)
        N_particles = len(self.initial_frame.position) - 1
        self.fluid_pos = self.initial_frame.position[:N_particles].copy()
        self.fluid_vel = self.initial_frame.velocity[:N_particles].copy()
        self.fluid_mass = self.initial_frame.attributes["mass_quantity"][:N_particles].copy()
        self.particle_volume = self.initial_frame.attributes["particle_volume"][:N_particles].copy()
        self.particle_radius = self.initial_frame.attributes["particle_radius"][:N_particles].copy()

        # Domain and geometry parameters
        self.tank_half = float(self.config["tank"]["interior_half_width"])
        self.tank_height = float(self.config["tank"]["height"])
        self.blade_half = np.asarray(self.config["blade"]["half_extents"], dtype=float)
        self.blade_center = np.asarray(self.config["blade"]["center"], dtype=float)
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.density = float(self.config["liquid"]["density"])
        self.h_spacing = float(self.config["liquid"]["particle_spacing"])
        self.grid_spacing = float(self.config["domain"]["grid_spacing"])
        self.bounds_min = np.asarray(self.config["domain"]["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(self.config["domain"]["bounds_max"], dtype=float)
        self.ref_grid = tuple(self.config["domain"]["reference_grid"])

        # SPH / Particle simulation parameters
        self.smoothing_h = self.h_spacing * 1.5
        self.rest_density = self.density
        # Sound speed for weakly compressible SPH / particle pressure
        # Blade tip max speed is ~ 3.0 * 0.5 = 1.5 m/s. Max flow speed ~ 2-3 m/s.
        self.c_s = 15.0
        self.gamma = 7.0
        self.B = (self.rest_density * self.c_s**2) / self.gamma
        self.viscosity_nu = float(self.config["liquid"]["kinematic_viscosity"])
        # Numerical viscosity parameter for stability
        self.alpha_visc = 0.1

        # Fixed time step for physical substepping
        self.sub_dt = 0.002

    def _blade_sdf_and_normal(self, points: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute blade signed distance, outward normal, and surface target velocity."""
        pose = self.scene.blade_kinematics(time)
        rel = points - pose["position"]
        # Transform to local coordinates: R^T * rel = rel @ R
        local = rel @ pose["rotation_matrix"]
        q = np.abs(local) - self.blade_half

        # Outside vector
        outside_vec = np.maximum(q, 0.0)
        dist_out = np.linalg.norm(outside_vec, axis=-1)
        dist_in = np.minimum(np.max(q, axis=-1), 0.0)
        sdf = dist_out + dist_in

        # Local outward gradient / normal
        signs = np.sign(local)
        signs[signs == 0.0] = 1.0

        # For points outside, grad is outside_vec normalized; for inside, grad is axis of max component
        grad_local = np.zeros_like(local)
        is_outside = sdf > 0
        if np.any(is_outside):
            out_len = np.linalg.norm(outside_vec[is_outside], axis=-1, keepdims=True)
            out_len = np.maximum(out_len, 1e-12)
            grad_local[is_outside] = (outside_vec[is_outside] / out_len) * signs[is_outside]

        is_inside = ~is_outside
        if np.any(is_inside):
            # Axis of maximum q
            max_axis = np.argmax(q[is_inside], axis=-1)
            for ax in range(3):
                mask = is_inside & (max_axis == ax)
                grad_local[mask, ax] = signs[mask, ax]

        # Normalize grad_local
        n_len = np.linalg.norm(grad_local, axis=-1, keepdims=True)
        n_len = np.maximum(n_len, 1e-12)
        grad_local = grad_local / n_len

        # Transform back to world: normal_world = grad_local @ R^T
        normal_world = grad_local @ pose["rotation_matrix"].T

        # Blade velocity at points: v_blade = omega x rel
        omega = pose["angular_velocity"]
        vel_blade = np.cross(omega, rel)

        return sdf, normal_world, vel_blade

    def _step_physics(self, dt: float) -> None:
        """Advance fluid particles by dt."""
        t = self.current_time
        N = len(self.fluid_pos)
        if N == 0:
            return

        # 1. External forces (gravity)
        self.fluid_vel += self.gravity * dt

        # 2. Blade interaction & boundary forces
        # Check proximity to moving blade
        sdf, normal, v_blade = self._blade_sdf_and_normal(self.fluid_pos, t)

        # Particle radius
        r = self.h_spacing * 0.5

        # Solid repulsion and velocity coupling for blade
        penetration = r - sdf
        submerged_mask = penetration > 0
        if np.any(submerged_mask):
            p_mask = submerged_mask
            pen = penetration[p_mask, None]
            norm = normal[p_mask]
            v_b = v_blade[p_mask]

            # Position correction
            self.fluid_pos[p_mask] += norm * pen

            # Relative velocity
            v_rel = self.fluid_vel[p_mask] - v_b
            v_normal = np.sum(v_rel * norm, axis=-1, keepdims=True)

            # Inelastic collision / momentum transfer along normal
            approaching = v_normal < 0
            if np.any(approaching):
                app_mask = np.where(submerged_mask)[0][approaching[:, 0]]
                norm_app = norm[approaching[:, 0]]
                v_n_app = v_normal[approaching[:, 0]]
                self.fluid_vel[app_mask] -= (1.0 + 0.2) * v_n_app * norm_app

            # Tangential friction / viscous traction transfer from blade
            v_tangent = v_rel - v_normal * norm
            self.fluid_vel[p_mask] -= 0.1 * v_tangent

        # 3. Particle-particle interaction: pressure & artificial viscosity via grid binning
        # Grid-based neighborhood search
        cell_size = self.smoothing_h
        grid_min = self.bounds_min
        grid_coords = np.floor((self.fluid_pos - grid_min) / cell_size).astype(int)

        # Spatial hashing / neighbor forces
        # Compute local particle density using kernel
        # W(r) = (1 - r/h)^2 for r < h
        h = self.smoothing_h
        h2 = h * h
        inv_h = 1.0 / h

        # Build spatial grid dictionary
        grid_dict: dict[tuple[int, int, int], list[int]] = {}
        for idx in range(N):
            cell = (grid_coords[idx, 0], grid_coords[idx, 1], grid_coords[idx, 2])
            if cell not in grid_dict:
                grid_dict[cell] = []
            grid_dict[cell].append(idx)

        # Compute pairwise pressure & viscosity
        # Tait equation: P = B * ((rho/rho0)^gamma - 1)
        # Using a fast neighbor-based force computation
        acc = np.zeros_like(self.fluid_pos)

        # 3D 27-neighborhood offsets
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        # Compute neighbor pairs and forces
        for cell, p_indices in grid_dict.items():
            # Gather all particles in neighbor cells
            neighbor_indices = []
            for off in offsets:
                n_cell = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
                if n_cell in grid_dict:
                    neighbor_indices.extend(grid_dict[n_cell])

            if not neighbor_indices:
                continue

            p_indices_arr = np.array(p_indices, dtype=int)
            n_indices_arr = np.array(neighbor_indices, dtype=int)

            pos_i = self.fluid_pos[p_indices_arr]
            pos_j = self.fluid_pos[n_indices_arr]
            vel_i = self.fluid_vel[p_indices_arr]
            vel_j = self.fluid_vel[n_indices_arr]

            # Pairwise distances
            diff = pos_i[:, None, :] - pos_j[None, :, :]  # [Ni, Nj, 3]
            dist_sq = np.sum(diff**2, axis=-1)  # [Ni, Nj]

            # Mask pairs within smoothing radius (excluding self)
            mask = (dist_sq < h2) & (dist_sq > 1e-12)
            if not np.any(mask):
                continue

            dist = np.sqrt(dist_sq)
            # Pressure repulsion when particles are closer than particle_spacing
            overlap = np.maximum(0.0, self.h_spacing - dist)
            # Pressure force ~ k * overlap / dist * diff
            k_stiff = 400.0
            p_force_mag = k_stiff * overlap / (dist + 1e-6)

            # Artificial viscosity
            v_diff = vel_i[:, None, :] - vel_j[None, :, :]
            v_dot_r = np.sum(v_diff * diff, axis=-1)
            visc_mag = np.where(v_dot_r < 0, -self.alpha_visc * self.c_s * v_dot_r / (dist_sq + 0.01 * h2), 0.0)

            force_mag = p_force_mag + visc_mag
            force_vec = np.sum((force_mag[:, :, None] * mask[:, :, None]) * (diff / (dist[:, :, None] + 1e-6)), axis=1)

            acc[p_indices_arr] += force_vec

        # Apply internal fluid accelerations
        self.fluid_vel += acc * dt
        # Velocity damping / air drag
        self.fluid_vel *= np.exp(-0.01 * dt)

        # Advect position
        self.fluid_pos += self.fluid_vel * dt

        # 4. Stationary Tank boundary conditions (floor & 4 walls)
        half = self.tank_half
        # Floor z = 0
        floor_penetration = r - self.fluid_pos[:, 2]
        floor_mask = floor_penetration > 0
        if np.any(floor_mask):
            self.fluid_pos[floor_mask, 2] = r
            # Reflect normal velocity with restitution and slip
            self.fluid_vel[floor_mask, 2] = np.maximum(0.0, -0.2 * self.fluid_vel[floor_mask, 2])
            self.fluid_vel[floor_mask, :2] *= 0.98

        # Side walls x = -half, x = half, y = -half, y = half (up to tank_height)
        for axis in (0, 1):
            # Min wall
            min_pen = (self.fluid_pos[:, axis] - (-half + r))
            min_mask = (min_pen < 0) & (self.fluid_pos[:, 2] < self.tank_height + 0.1)
            if np.any(min_mask):
                self.fluid_pos[min_mask, axis] = -half + r
                self.fluid_vel[min_mask, axis] = np.maximum(0.0, -0.2 * self.fluid_vel[min_mask, axis])
                self.fluid_vel[min_mask, 1 - axis] *= 0.98

            # Max wall
            max_pen = (half - r) - self.fluid_pos[:, axis]
            max_mask = (max_pen < 0) & (self.fluid_pos[:, 2] < self.tank_height + 0.1)
            if np.any(max_mask):
                self.fluid_pos[max_mask, axis] = half - r
                self.fluid_vel[max_mask, axis] = np.minimum(0.0, -0.2 * self.fluid_vel[max_mask, axis])
                self.fluid_vel[max_mask, 1 - axis] *= 0.98

        # Second blade contact enforcement after advection
        t_next = t + dt
        sdf2, normal2, v_blade2 = self._blade_sdf_and_normal(self.fluid_pos, t_next)
        pen2 = r - sdf2
        sub_mask2 = pen2 > 0
        if np.any(sub_mask2):
            self.fluid_pos[sub_mask2] += normal2[sub_mask2] * pen2[sub_mask2, None]
            v_rel2 = self.fluid_vel[sub_mask2] - v_blade2[sub_mask2]
            v_n2 = np.sum(v_rel2 * normal2[sub_mask2], axis=-1, keepdims=True)
            push = v_n2 < 0
            if np.any(push):
                push_indices = np.where(sub_mask2)[0][push[:, 0]]
                self.fluid_vel[push_indices] -= 1.2 * v_n2[push[:, 0]] * normal2[push_indices]

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            raise ValueError("Requested time is before current simulation time")

        # Step forward with physical sub-time-steps
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step_physics(dt)
            self.current_time += dt

        self.current_time = time
        return self._build_state_frame(time)

    def _build_state_frame(self, time: float) -> StateFrame:
        # Kinematics of blade at requested time
        pose = self.scene.blade_kinematics(time)
        blade_pos = pose["position"]
        blade_rot = pose["orientation"]
        blade_omega = pose["angular_velocity"]

        # Combined particles: liquid particles + rigid blade actuator
        count = len(self.fluid_pos) + 1
        position = np.vstack((self.fluid_pos, blade_pos))
        velocity = np.vstack((self.fluid_vel, np.zeros(3)))
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
        orientation[-1] = blade_rot
        angular_velocity = np.zeros((count, 3))
        angular_velocity[-1] = blade_omega

        mass_quantity = np.r_[self.fluid_mass, 0.0]
        particle_volume = np.r_[self.particle_volume, 0.0]
        particle_radius = np.r_[self.particle_radius, 0.0]
        body_id = np.zeros(count, dtype=np.int64)
        body_id[-1] = int(self.config["blade"]["body_id"])

        # Construct Eulerian diagnostic grid fields
        h = self.grid_spacing
        domain = self.config["domain"]
        axes = [
            float(lo) + (np.arange(n) + 0.5) * h
            for lo, n in zip(domain["bounds_min"], domain["reference_grid"], strict=True)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # [nx, ny, nz, 3]
        nx, ny, nz, _ = grid.shape

        # Liquid level set: signed distance to liquid surface
        # Estimate surface elevation / liquid level from fluid particles
        # Liquid signed distance approximated on diagnostic grid
        grid_flat = grid.reshape(-1, 3)
        # Blade distance
        solid_dist = self.scene.blade_signed_distance(grid, time)

        # Compute distance to liquid particles on grid
        # For efficiency, compute a narrow-band distance to particles
        # Level set phi: < 0 inside liquid, > 0 outside liquid
        # Default flat reference depth as baseline if no particles
        # Nearest particle distance
        if len(self.fluid_pos) > 0:
            # Grid binning to estimate local liquid presence
            cell_size = self.grid_spacing
            g_min = self.bounds_min
            p_indices = np.floor((self.fluid_pos - g_min) / cell_size).astype(int)
            p_valid = (
                (p_indices[:, 0] >= 0) & (p_indices[:, 0] < nx) &
                (p_indices[:, 1] >= 0) & (p_indices[:, 1] < ny) &
                (p_indices[:, 2] >= 0) & (p_indices[:, 2] < nz)
            )

            # Particle density / count grid
            occ_grid = np.zeros((nx, ny, nz), dtype=float)
            vel_grid_accum = np.zeros((nx, ny, nz, 3), dtype=float)

            p_valid_idx = np.where(p_valid)[0]
            for idx in p_valid_idx:
                gx, gy, gz = p_indices[idx]
                occ_grid[gx, gy, gz] += 1.0
                vel_grid_accum[gx, gy, gz] += self.fluid_vel[idx]

            # Compute velocity field
            occ_mask = occ_grid > 0
            velocity_field = np.zeros((nx, ny, nz, 3), dtype=float)
            velocity_field[occ_mask] = vel_grid_accum[occ_mask] / occ_grid[occ_mask, None]

            # Smooth level set from occupancy: negative where occupied / near liquid
            # Maximum liquid z height in each (x, y) column
            max_z_col = np.full((nx, ny), -1.0)
            for idx in p_valid_idx:
                gx, gy, _ = p_indices[idx]
                pz = self.fluid_pos[idx, 2]
                if pz > max_z_col[gx, gy]:
                    max_z_col[gx, gy] = pz

            # Default empty columns to nominal depth or zero
            max_z_col[max_z_col < 0] = float(self.config["liquid"]["depth"])
            phi = grid[..., 2] - max_z_col[:, :, None]
        else:
            depth = float(self.config["liquid"]["depth"])
            phi = grid[..., 2] - depth
            velocity_field = np.zeros((nx, ny, nz, 3), dtype=float)

        # Active mask (incompressible fluid interior away from boundaries and free surface)
        half = self.tank_half
        active = (
            (phi < -h)
            & (grid[..., 2] > h)
            & (np.abs(grid[..., 0]) < half - h)
            & (np.abs(grid[..., 1]) < half - h)
            & (solid_dist > h)
        )

        # Tank boundary samples
        wall_position, wall_normal = self.scene.initial_tank_samples()
        # Measure fluid velocity near wall
        wall_velocity = np.zeros_like(wall_position)
        wall_target_velocity = np.zeros_like(wall_position)

        # Blade boundary samples
        blade_samples = self.scene.blade_surface_samples(time)
        b_pos = blade_samples["position"]
        b_norm = blade_samples["normal"]
        b_target_vel = blade_samples["target_velocity"]

        # Wetted blade samples: where distance to liquid particles is small (wetted by liquid)
        # Find liquid proximity to blade surface samples
        if len(self.fluid_pos) > 0:
            # Sample wetted if near liquid particles or below local liquid surface
            # Check proximity to nearest particles
            wetted = np.zeros(len(b_pos), dtype=bool)
            # A fast check: below max height of fluid and within fluid domain
            # Or particle distance < 2 * h_spacing
            wetted_cand = b_pos[:, 2] < (np.max(self.fluid_pos[:, 2]) + 0.05)
            cand_indices = np.where(wetted_cand)[0]
            if len(cand_indices) > 0:
                cand_pos = b_pos[cand_indices]
                # Check distance to liquid particles in blocks
                block_size = 500
                cand_wet = np.zeros(len(cand_pos), dtype=bool)
                for b_start in range(0, len(self.fluid_pos), 2000):
                    b_end = min(b_start + 2000, len(self.fluid_pos))
                    f_chunk = self.fluid_pos[b_start:b_end]
                    diff = cand_pos[:, None, :] - f_chunk[None, :, :]
                    dists = np.min(np.linalg.norm(diff, axis=-1), axis=1)
                    cand_wet |= (dists < 2.5 * self.h_spacing)
                wetted[cand_indices[cand_wet]] = True

            # Fallback if no wetted samples detected: use initial liquid depth
            if not np.any(wetted):
                wetted = b_pos[:, 2] < float(self.config["liquid"]["depth"])
        else:
            wetted = b_pos[:, 2] < float(self.config["liquid"]["depth"])

        blade_wetted_pos = b_pos[wetted]
        blade_wetted_norm = b_norm[wetted]
        blade_wetted_target_vel = b_target_vel[wetted]

        # Fluid-side measured velocity at currently wetted moving-blade samples
        # Interpolate from nearby fluid particles
        blade_wetted_fluid_vel = np.zeros_like(blade_wetted_pos)
        if len(self.fluid_pos) > 0 and len(blade_wetted_pos) > 0:
            for i, bp in enumerate(blade_wetted_pos):
                diff = self.fluid_pos - bp
                dists = np.linalg.norm(diff, axis=-1)
                close_mask = dists < 3.0 * self.h_spacing
                if np.any(close_mask):
                    weights = np.maximum(0.0, 1.0 - dists[close_mask] / (3.0 * self.h_spacing))
                    w_sum = np.sum(weights)
                    if w_sum > 1e-6:
                        blade_wetted_fluid_vel[i] = np.sum(self.fluid_vel[close_mask] * weights[:, None], axis=0) / w_sum
                    else:
                        blade_wetted_fluid_vel[i] = blade_wetted_target_vel[i]
                else:
                    blade_wetted_fluid_vel[i] = blade_wetted_target_vel[i]

        density_field = (phi < 0).astype(float) * self.density

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass_quantity.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": mass_quantity,
                "particle_volume": particle_volume,
                "particle_radius": particle_radius,
            },
            fields={
                "velocity_field": velocity_field,
                "active_mask": active.astype(float),
                "liquid_level_set": phi,
                "density_field": density_field,
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": wall_position,
                "boundary_normal": wall_normal,
                "boundary_velocity": wall_velocity,
                "boundary_target_velocity": wall_target_velocity,
                "blade_boundary_position": blade_wetted_pos,
                "blade_boundary_normal": blade_wetted_norm,
                "blade_boundary_velocity": blade_wetted_fluid_vel,
                "blade_boundary_target_velocity": blade_wetted_target_vel,
                "grid_origin": np.asarray(domain["bounds_min"], dtype=float),
            },
            scalars={
                "grid_spacing": h,
                "characteristic_length": float(2 * self.blade_half[0]),
                "reference_velocity": float(
                    abs(self.config["blade"]["angular_speed"]) * self.blade_half[0]
                ),
                "initial_liquid_volume": float(self.initial_frame.scalars["initial_liquid_volume"]),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.fluid_pos),
        }

