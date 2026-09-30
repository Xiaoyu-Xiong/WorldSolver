"""Breaking wave over a submerged shelf solver.

Implements physical SPH / PIC / particle fluid dynamics with boundary handling
for tank walls, gate lifting, and submerged shelf obstacle.
Produces valid StateFrames with all required diagnostic fields.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.initial_frame = scene.initial_state()
        self.pos = self.initial_frame.position.copy().astype(np.float64)
        self.vel = np.zeros_like(self.pos, dtype=np.float64)
        self.n_particles = len(self.pos)
        self.mass = self.initial_frame.mass.copy().astype(np.float64)

        # Sampling and grid config
        sampling = self.config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Field grid definition
        self.field_shape = tuple(sampling.get("field_grid_shape", [24, 16, 28]))
        self.nx, self.ny, self.nz = self.field_shape

        # Tank physical bounds
        # Tank from x in [-3.2, 3.2], y in [-0.92, 0.92], z in [0.0, 1.1]
        self.x_min, self.x_max = -3.1, 3.06
        self.y_min, self.y_max = -0.92, 0.92
        self.z_min, self.z_max = 0.0, 1.1

        # Diagnostic grid bounds covering domain
        self.grid_min = np.array([-3.2, -1.0, 0.0], dtype=np.float64)
        self.grid_max = np.array([3.2, 1.0, 1.2], dtype=np.float64)
        self.grid_spacing = (self.grid_max - self.grid_min) / np.array([self.nx, self.ny, self.nz])

        # Shelf geometry: box centered at [1.45, 0, 0.3], half_extents [1.2, 0.84, 0.3]
        # x: [0.25, 2.65], y: [-0.84, 0.84], z: [0.0, 0.6]
        self.shelf_min = np.array([0.25, -0.84, 0.0])
        self.shelf_max = np.array([2.65, 0.84, 0.60])

        # Gate at x = -0.45, lifts upward starting at t = 0
        self.gate_x = -0.45
        self.gate_lift_speed = 2.0  # m/s upward gate release

        # Wall boundary samples for required fields
        self._init_boundary_samples()

        # Internal simulation time
        self.current_time = 0.0

        # SPH / Particle parameters
        self.h = self.spacing * 1.3
        self.h2 = self.h * self.h
        self.rest_density = self.density
        self.c0 = 20.0  # speed of sound for Tait's EOS
        self.gamma = 7.0
        self.B = self.rest_density * (self.c0 ** 2) / self.gamma
        self.viscosity = 0.05
        self.particle_radius = float(self.initial_frame.attributes["particle_radius"][0])

        # Sub-stepping dt
        self.sub_dt = min(0.002, self.spacing / (2.0 * self.c0))

    def _init_boundary_samples(self):
        # Generate wall boundary sample points and normals
        # Ground (z=0), Left wall (y=-0.92), Right wall (y=0.92), End wall (x=3.06), Back wall (x=-3.1)
        samples = []
        normals = []

        # Ground
        gx = np.linspace(-3.0, 3.0, 16)
        gy = np.linspace(-0.9, 0.9, 8)
        gxx, gyy = np.meshgrid(gx, gy)
        g_pts = np.column_stack((gxx.ravel(), gyy.ravel(), np.zeros(gxx.size)))
        samples.append(g_pts)
        normals.append(np.tile([0.0, 0.0, 1.0], (len(g_pts), 1)))

        # Shelf top face (z = 0.6, x in [0.25, 2.65], y in [-0.84, 0.84])
        sx = np.linspace(0.3, 2.6, 8)
        sy = np.linspace(-0.8, 0.8, 6)
        sxx, syy = np.meshgrid(sx, sy)
        s_pts = np.column_stack((sxx.ravel(), syy.ravel(), np.full(sxx.size, 0.60)))
        samples.append(s_pts)
        normals.append(np.tile([0.0, 0.0, 1.0], (len(s_pts), 1)))

        # Shelf front face (x = 0.25, y in [-0.84, 0.84], z in [0.0, 0.6])
        sy2 = np.linspace(-0.8, 0.8, 6)
        sz2 = np.linspace(0.05, 0.55, 4)
        syy2, szz2 = np.meshgrid(sy2, sz2)
        sf_pts = np.column_stack((np.full(syy2.size, 0.25), syy2.ravel(), szz2.ravel()))
        samples.append(sf_pts)
        normals.append(np.tile([-1.0, 0.0, 0.0], (len(sf_pts), 1)))

        # End wall (x = 3.06)
        ew_y = np.linspace(-0.9, 0.9, 8)
        ew_z = np.linspace(0.1, 1.0, 6)
        ew_yy, ew_zz = np.meshgrid(ew_y, ew_z)
        ew_pts = np.column_stack((np.full(ew_yy.size, 3.06), ew_yy.ravel(), ew_zz.ravel()))
        samples.append(ew_pts)
        normals.append(np.tile([-1.0, 0.0, 0.0], (len(ew_pts), 1)))

        # Back wall (x = -3.1)
        bw_y = np.linspace(-0.9, 0.9, 8)
        bw_z = np.linspace(0.1, 1.0, 6)
        bw_yy, bw_zz = np.meshgrid(bw_y, bw_z)
        bw_pts = np.column_stack((np.full(bw_yy.size, -3.1), bw_yy.ravel(), bw_zz.ravel()))
        samples.append(bw_pts)
        normals.append(np.tile([1.0, 0.0, 0.0], (len(bw_pts), 1)))

        self.boundary_samples = np.vstack(samples)
        self.boundary_normals = np.vstack(normals)
        self.n_boundary = len(self.boundary_samples)

    def _step_simulation(self, dt: float):
        if self.n_particles == 0:
            return

        gate_z_bottom = -0.55 + 0.55 + self.gate_lift_speed * self.current_time  # starts at 0, lifts up

        # Spatial hash grid for neighbor search
        cell_size = self.h
        inv_cs = 1.0 / cell_size

        grid_coords = np.floor(self.pos * inv_cs).astype(np.int64)
        # Hash keys
        p1, p2, p3 = 73856093, 19349663, 83492791
        hash_keys = (grid_coords[:, 0] * p1 ^ grid_coords[:, 1] * p2 ^ grid_coords[:, 2] * p3) % 65536

        # Sort particles by hash key
        sort_idx = np.argsort(hash_keys)
        sorted_pos = self.pos[sort_idx]
        sorted_vel = self.vel[sort_idx]
        sorted_mass = self.mass[sort_idx]

        # Compute density using cubic spline / Lucy kernel approximation
        # Efficient neighbor pairwise interaction using spatial grid bins
        unique_keys, start_indices = np.unique(hash_keys[sort_idx], return_index=True)
        key_map = dict(zip(unique_keys, start_indices))

        densities = np.full(self.n_particles, self.rest_density, dtype=np.float64)
        poly6_const = 315.0 / (64.0 * np.pi * (self.h ** 9))
        spiky_const = -45.0 / (np.pi * (self.h ** 6))
        visc_const = 45.0 / (np.pi * (self.h ** 6))

        # Density evaluation
        for i in range(self.n_particles):
            gx, gy, gz = grid_coords[sort_idx[i]]
            d_sum = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neigh_key = ((gx + dx) * p1 ^ (gy + dy) * p2 ^ (gz + dz) * p3) % 65536
                        if neigh_key in key_map:
                            k_start = key_map[neigh_key]
                            # scan particles with this key
                            k = k_start
                            while k < self.n_particles and hash_keys[sort_idx[k]] == neigh_key:
                                r_vec = sorted_pos[i] - sorted_pos[k]
                                r2 = np.dot(r_vec, r_vec)
                                if r2 < self.h2:
                                    d_sum += sorted_mass[k] * poly6_const * ((self.h2 - r2) ** 3)
                                k += 1
            densities[i] = max(d_sum, self.rest_density * 0.1)

        # Tait's EOS pressure
        pressures = self.B * ((np.maximum(densities / self.rest_density, 0.01) ** self.gamma) - 1.0)
        pressures = np.maximum(pressures, 0.0)

        # Force computation
        forces = np.zeros((self.n_particles, 3), dtype=np.float64)
        for i in range(self.n_particles):
            gx, gy, gz = grid_coords[sort_idx[i]]
            f_press = np.zeros(3)
            f_visc = np.zeros(3)
            p_i = sorted_pos[i]
            v_i = sorted_vel[i]
            rho_i = densities[i]
            pres_i = pressures[i]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neigh_key = ((gx + dx) * p1 ^ (gy + dy) * p2 ^ (gz + dz) * p3) % 65536
                        if neigh_key in key_map:
                            k = key_map[neigh_key]
                            while k < self.n_particles and hash_keys[sort_idx[k]] == neigh_key:
                                if k != i:
                                    r_vec = p_i - sorted_pos[k]
                                    r = np.linalg.norm(r_vec)
                                    if 0 < r < self.h:
                                        # Pressure force (Spiky kernel)
                                        grad_w = spiky_const * ((self.h - r) ** 2) * (r_vec / r)
                                        f_press -= sorted_mass[k] * (pres_i / (rho_i ** 2) + pressures[k] / (densities[k] ** 2)) * grad_w
                                        # Viscosity force (Laplacian)
                                        lap_w = visc_const * (self.h - r)
                                        f_visc += self.viscosity * sorted_mass[k] * (sorted_vel[k] - v_i) / densities[k] * lap_w
                                k += 1
            forces[i] = f_press * rho_i + f_visc

        # Re-index forces back to original particle ordering
        orig_forces = np.zeros_like(self.pos)
        orig_forces[sort_idx] = forces

        # Acceleration: gravity + pressure/viscosity force / density
        orig_densities = np.zeros(self.n_particles)
        orig_densities[sort_idx] = densities
        acc = orig_forces / orig_densities[:, None] + self.gravity[None, :]

        # Symplectic Euler update
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Boundary collisions & penalties
        r_p = self.particle_radius
        restitution = 0.05
        damping = 0.98

        # Ground collision (z = 0)
        below_ground = self.pos[:, 2] < r_p
        self.pos[below_ground, 2] = r_p
        self.vel[below_ground, 2] = np.maximum(0.0, -self.vel[below_ground, 2] * restitution)
        self.vel[below_ground, 0] *= damping
        self.vel[below_ground, 1] *= damping

        # Tank side walls (y = -0.92, y = 0.92)
        left_wall = self.pos[:, 1] < (self.y_min + r_p)
        self.pos[left_wall, 1] = self.y_min + r_p
        self.vel[left_wall, 1] = np.maximum(0.0, -self.vel[left_wall, 1] * restitution)

        right_wall = self.pos[:, 1] > (self.y_max - r_p)
        self.pos[right_wall, 1] = self.y_max - r_p
        self.vel[right_wall, 1] = np.minimum(0.0, -self.vel[right_wall, 1] * restitution)

        # Tank back wall (x = -3.1) and front wall (x = 3.06)
        back_wall = self.pos[:, 0] < (self.x_min + r_p)
        self.pos[back_wall, 0] = self.x_min + r_p
        self.vel[back_wall, 0] = np.maximum(0.0, -self.vel[back_wall, 0] * restitution)

        end_wall = self.pos[:, 0] > (self.x_max - r_p)
        self.pos[end_wall, 0] = self.x_max - r_p
        self.vel[end_wall, 0] = np.minimum(0.0, -self.vel[end_wall, 0] * restitution)

        # Gate collision: gate is at x = self.gate_x, bottom is at gate_z_bottom
        # Particles with x < gate_x cannot cross if z < gate_z_bottom
        in_gate_x = (self.pos[:, 0] > (self.gate_x - r_p)) & (self.pos[:, 0] < (self.gate_x + r_p))
        in_gate_z = self.pos[:, 2] < gate_z_bottom
        gate_blocked = in_gate_x & in_gate_z
        self.pos[gate_blocked, 0] = self.gate_x - r_p
        self.vel[gate_blocked, 0] = np.minimum(0.0, -self.vel[gate_blocked, 0] * restitution)

        # Submerged shelf obstacle collision:
        # Box from x in [0.25, 2.65], y in [-0.84, 0.84], z in [0.0, 0.60]
        in_shelf_x = (self.pos[:, 0] >= self.shelf_min[0] - r_p) & (self.pos[:, 0] <= self.shelf_max[0] + r_p)
        in_shelf_y = (self.pos[:, 1] >= self.shelf_min[1] - r_p) & (self.pos[:, 1] <= self.shelf_max[1] + r_p)
        in_shelf_z = (self.pos[:, 2] <= self.shelf_max[2] + r_p)
        in_shelf = in_shelf_x & in_shelf_y & in_shelf_z

        if np.any(in_shelf):
            shelf_idx = np.where(in_shelf)[0]
            for idx in shelf_idx:
                p = self.pos[idx]
                # Distances to shelf boundary faces
                d_top = (self.shelf_max[2] + r_p) - p[2]
                d_front = p[0] - (self.shelf_min[0] - r_p)
                d_back = (self.shelf_max[0] + r_p) - p[0]

                # Resolve against the closest face
                min_d = min(d_top, d_front, d_back)
                if min_d == d_top:
                    self.pos[idx, 2] = self.shelf_max[2] + r_p
                    self.vel[idx, 2] = max(0.0, -self.vel[idx, 2] * restitution)
                    self.vel[idx, 0] *= damping
                    self.vel[idx, 1] *= damping
                elif min_d == d_front:
                    self.pos[idx, 0] = self.shelf_min[0] - r_p
                    self.vel[idx, 0] = min(0.0, -self.vel[idx, 0] * restitution)
                elif min_d == d_back:
                    self.pos[idx, 0] = self.shelf_max[0] + r_p
                    self.vel[idx, 0] = max(0.0, -self.vel[idx, 0] * restitution)

        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Step simulation to target_time
        while self.current_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step_simulation(dt)

        return self._build_state_frame(target_time)

    def _build_state_frame(self, t: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()

        # Compute diagnostic grid fields
        # Diagnostic grid coordinates
        gx = np.linspace(self.grid_min[0], self.grid_max[0], self.nx)
        gy = np.linspace(self.grid_min[1], self.grid_max[1], self.ny)
        gz = np.linspace(self.grid_min[2], self.grid_max[2], self.nz)
        g_x, g_y, g_z = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_pts = np.stack((g_x, g_y, g_z), axis=-1)  # shape (nx, ny, nz, 3)

        # Velocity grid and liquid level set (signed distance to fluid surface)
        velocity_field = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        active_mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        liquid_level_set = np.full((self.nx, self.ny, self.nz), 1.0, dtype=np.float64)

        if self.n_particles > 0:
            # Grid cell indices for particles
            dx, dy, dz = self.grid_spacing
            ix = np.clip(np.floor((pos[:, 0] - self.grid_min[0]) / dx).astype(int), 0, self.nx - 1)
            iy = np.clip(np.floor((pos[:, 1] - self.grid_min[1]) / dy).astype(int), 0, self.ny - 1)
            iz = np.clip(np.floor((pos[:, 2] - self.grid_min[2]) / dz).astype(int), 0, self.nz - 1)

            # Accumulate velocity into diagnostic grid
            counts = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
            for p_idx in range(self.n_particles):
                cx, cy, cz = ix[p_idx], iy[p_idx], iz[p_idx]
                velocity_field[cx, cy, cz] += vel[p_idx]
                counts[cx, cy, cz] += 1.0

            occupied = counts > 0
            velocity_field[occupied] /= counts[occupied, None]
            active_mask = occupied

            # Liquid level set: approximate signed distance field using min distance to particles minus radius
            # For efficiency on 24x16x28 grid, compute distance using sub-sampled points or bounding box
            flat_grid = grid_pts.reshape(-1, 3)
            # Sample particle positions if large, or compute directly
            sample_p = pos if len(pos) <= 500 else pos[np.random.choice(len(pos), 500, replace=False)]
            # Distance from each grid point to nearest particle
            # Chunked distance calculation
            min_dist = np.full(len(flat_grid), np.inf)
            chunk_size = 500
            for c_start in range(0, len(flat_grid), chunk_size):
                c_end = min(c_start + chunk_size, len(flat_grid))
                diffs = flat_grid[c_start:c_end, None, :] - sample_p[None, :, :]
                dists = np.sqrt(np.min(np.sum(diffs ** 2, axis=-1), axis=1))
                min_dist[c_start:c_end] = dists

            liquid_level_set = (min_dist - self.particle_radius).reshape(self.nx, self.ny, self.nz)

        # Boundary fields
        boundary_target_vel = np.zeros_like(self.boundary_samples)
        boundary_vel = np.zeros_like(self.boundary_samples)
        cumulative_mass_source = np.zeros(self.n_particles, dtype=np.float64)

        char_len = max(float(np.ptp(pos, axis=0).max()) if len(pos) > 0 else 1.0, self.spacing)
        ref_vel = max(float(np.linalg.norm(vel, axis=1).max()) if len(vel) > 0 else 1.0, 1.0)

        frame = StateFrame(
            t,
            pos,
            vel,
            mass=mass,
            body_id=self.initial_frame.body_id.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "particle_radius": self.initial_frame.attributes["particle_radius"].copy(),
                "material_label": self.initial_frame.attributes["material_label"].copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], len(pos), axis=0),
            },
            fields={
                "active_mask": active_mask,
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_target_velocity": boundary_target_vel,
                "boundary_velocity": boundary_vel,
                "cumulative_mass_source": cumulative_mass_source,
                "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": char_len,
                "grid_spacing": float(self.spacing),
                "reference_velocity": ref_vel,
                "contact_length_scale": float(self.spacing),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": self.n_particles,
            "current_time": self.current_time,
        }

