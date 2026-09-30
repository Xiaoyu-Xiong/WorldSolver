"""SPH Fluid Dynamics Solver for Water Pouring into Glass."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        ic = self.config["initial_conditions"]
        self.rest_density = float(ic["liquid_density"])
        self.particle_spacing = float(ic["particle_spacing"])
        self.particle_mass = self.rest_density * (self.particle_spacing ** 3)

        self.h = 1.8 * self.particle_spacing  # SPH smoothing radius
        self.c0 = 25.0  # Speed of sound for Tait EOS (Tait exponent gamma=7)
        self.gamma = 7.0
        self.B = (self.rest_density * (self.c0 ** 2)) / self.gamma
        self.viscosity = 0.001  # Kinematic/dynamic viscosity term
        self.boundary_damping = 0.1

        # Container geometry
        self.glass_inner_radius = float(ic["glass_inner_radius"])
        self.glass_inner_height = float(ic["glass_inner_height"])
        self.glass_bottom = 0.0

        # Jet inflow parameters
        self.jet_start = np.asarray(ic["jet_start"], dtype=np.float64)
        self.jet_end = np.asarray(ic["jet_end"], dtype=np.float64)
        jet_vec = self.jet_end - self.jet_start
        self.jet_dir = jet_vec / np.linalg.norm(jet_vec)
        self.jet_speed = float(ic["jet_speed"])
        self.jet_radius = float(ic["jet_radius"])
        self.source_start_time = float(ic["source_start_time"])
        self.source_end_time = float(ic["source_end_time"])
        self.target_inflow_rate = float(ic["target_inflow_rate_m3_s"])

        # Inflow disk generator
        h_s = self.particle_spacing
        side = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cross = np.cross(self.jet_dir, side)
        cross /= np.linalg.norm(cross)
        self.jet_side = side
        self.jet_cross = cross
        offsets = np.arange(-self.jet_radius + 0.5 * h_s, self.jet_radius, h_s)
        self.inflow_disk = np.array([
            [a, b] for a in offsets for b in offsets
            if a * a + b * b <= (self.jet_radius - 0.2 * h_s) ** 2
        ], dtype=np.float64)
        self.num_particles_per_inflow_slice = len(self.inflow_disk)
        self.inflow_slice_mass = self.num_particles_per_inflow_slice * self.particle_mass
        self.inflow_advance_accum = 0.0

        # Domain and grid
        domain = self.config["domain"]
        self.ref_grid = domain["reference_grid"]  # [55, 55, 120]
        self.bounds_min = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.bounds_max = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.grid_shape = tuple(self.ref_grid)
        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.ref_grid[0]
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ref_grid[1]
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.ref_grid[2]
        self.grid_spacing_scalar = float(self.dx)
        self.char_length = float(2.0 * self.glass_inner_radius)
        self.ref_velocity = float(self.jet_speed)

        # Boundary sample points for required fields
        self.boundary_normals, self.boundary_target_velocities, self.boundary_velocities = self._init_boundary_samples()

        # Diagnostic grid cell centers
        gx = np.linspace(self.bounds_min[0] + 0.5 * self.dx, self.bounds_max[0] - 0.5 * self.dx, self.ref_grid[0])
        gy = np.linspace(self.bounds_min[1] + 0.5 * self.dy, self.bounds_max[1] - 0.5 * self.dy, self.ref_grid[1])
        gz = np.linspace(self.bounds_min[2] + 0.5 * self.dz, self.bounds_max[2] - 0.5 * self.dz, self.ref_grid[2])
        self.grid_coords = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1)  # [nx, ny, nz, 3]

        # Get initial state
        init_frame = self.scene.initial_state()
        self.pos = np.copy(init_frame.position)
        self.vel = np.copy(init_frame.velocity)
        self.mass = np.copy(init_frame.mass)
        self.current_time = 0.0

        # Cumulative mass source tracking
        # Initial jet in scene.initial_state is at t=0; any new slices emitted after t=0 contribute to cumulative mass
        self.cumulative_mass_inflow = 0.0

        # Internal sub-stepping dt
        self.sim_dt = 0.0005

    def _init_boundary_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate static boundary wall sample points and normals for the cylindrical glass."""
        n_theta = 36
        n_z = 25
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.linspace(0.0, self.glass_inner_height, n_z)
        T, Z = np.meshgrid(theta, z)
        # Cylinder walls normals point inward
        cx = -np.cos(T.ravel())
        cy = -np.sin(T.ravel())
        cz = np.zeros_like(cx)
        wall_normals = np.column_stack([cx, cy, cz])

        # Bottom floor normals point upward (+z)
        r = np.linspace(0.01, self.glass_inner_radius * 0.95, 6)
        Tb, Rb = np.meshgrid(theta, r)
        bx = np.zeros(len(Tb.ravel()))
        by = np.zeros(len(Tb.ravel()))
        bz = np.ones(len(Tb.ravel()))
        floor_normals = np.column_stack([bx, by, bz])

        boundary_normals = np.vstack([wall_normals, floor_normals])
        boundary_target_velocity = np.zeros_like(boundary_normals)
        boundary_velocity = np.zeros_like(boundary_normals)
        return boundary_normals, boundary_target_velocity, boundary_velocity

    def _sph_step(self, dt: float) -> None:
        """Advance fluid particles using standard SPH / WCSPH with boundary handling."""
        n_p = len(self.pos)
        if n_p == 0:
            return

        h = self.h
        h2 = h * h
        poly6_const = 315.0 / (64.0 * np.pi * (h ** 9))
        spiky_grad_const = -45.0 / (np.pi * (h ** 6))

        # Build KD-Tree for neighbor searching
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(h, output_type="ndarray")

        # Compute Density
        # Self density contribution
        self_rho = self.particle_mass * poly6_const * (h2 ** 3)
        density = np.full(n_p, self_rho, dtype=np.float64)

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            diff = self.pos[i_idx] - self.pos[j_idx]
            r2 = np.sum(diff ** 2, axis=1)
            dr2 = h2 - r2
            w = self.particle_mass * poly6_const * (dr2 ** 3)
            np.add.at(density, i_idx, w)
            np.add.at(density, j_idx, w)

        # Compute Pressure via Tait EOS
        rho_ratio = np.maximum(density / self.rest_density, 0.5)
        pressure = self.B * (np.power(rho_ratio, self.gamma) - 1.0)
        pressure = np.maximum(pressure, 0.0)

        # Compute Accelerations: Pressure Gradient + Viscosity + Gravity
        acc = np.tile(self.gravity, (n_p, 1))

        if len(pairs) > 0:
            r = np.sqrt(np.maximum(r2, 1e-12))
            inv_r = 1.0 / r
            grad_w = spiky_grad_const * ((h - r) ** 2)[:, None] * (diff * inv_r[:, None])

            # SPH symmetric pressure force: - m * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W
            p_rho2 = pressure / (density ** 2)
            p_term = self.particle_mass * (p_rho2[i_idx] + p_rho2[j_idx])[:, None] * grad_w

            # Viscosity force (Monaghan artificial / SPH standard)
            v_diff = self.vel[i_idx] - self.vel[j_idx]
            v_dot_r = np.sum(v_diff * diff, axis=1)
            visc_term = (
                (2.0 * self.viscosity * self.particle_mass)
                / (density[i_idx] + density[j_idx])
            )[:, None] * (v_dot_r / (r2 + 0.01 * h2))[:, None] * grad_w

            np.add.at(acc, i_idx, -(p_term - visc_term))
            np.add.at(acc, j_idx, (p_term - visc_term))

        # Update velocities and positions (Verlet / Symplectic Euler)
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Enforce Boundary Conditions on cylindrical glass
        self._apply_boundary_conditions()

    def _apply_boundary_conditions(self) -> None:
        """Enforce cylinder and bottom wall collisions."""
        r_xy = np.hypot(self.pos[:, 0], self.pos[:, 1])
        margin = 0.5 * self.particle_spacing
        r_max = self.glass_inner_radius - margin
        z_min = margin
        z_max = self.glass_inner_height

        # Glass bottom reflection
        bottom_mask = (self.pos[:, 2] < z_min) & (r_xy < self.glass_inner_radius)
        if np.any(bottom_mask):
            self.pos[bottom_mask, 2] = z_min
            self.vel[bottom_mask, 2] = -self.vel[bottom_mask, 2] * self.boundary_damping
            self.vel[bottom_mask, :2] *= (1.0 - self.boundary_damping)

        # Glass cylinder wall reflection (only inside the glass height)
        wall_mask = (r_xy > r_max) & (self.pos[:, 2] < z_max) & (self.pos[:, 2] >= self.glass_bottom)
        if np.any(wall_mask):
            inv_r = 1.0 / np.maximum(r_xy[wall_mask], 1e-8)
            nx = self.pos[wall_mask, 0] * inv_r
            ny = self.pos[wall_mask, 1] * inv_r
            normal = np.column_stack([nx, ny, np.zeros_like(nx)])

            # Project position to boundary
            self.pos[wall_mask, 0] = nx * r_max
            self.pos[wall_mask, 1] = ny * r_max

            # Reflect normal velocity
            vn = np.sum(self.vel[wall_mask] * normal, axis=1)
            reflect = vn > 0
            if np.any(reflect):
                indices = np.where(wall_mask)[0][reflect]
                norm_sub = normal[reflect]
                vn_sub = vn[reflect]
                self.vel[indices] -= (1.0 + self.boundary_damping) * vn_sub[:, None] * norm_sub
                self.vel[indices] *= (1.0 - self.boundary_damping)

        # Floor outside glass boundary
        floor_outside = (self.pos[:, 2] < 0.0) & (r_xy >= self.glass_inner_radius)
        if np.any(floor_outside):
            self.pos[floor_outside, 2] = 0.0
            self.vel[floor_outside, 2] = 0.0
            self.vel[floor_outside] *= 0.5

    def _emit_inflow(self, dt: float) -> None:
        """Emit new particles from the pouring source nozzle if within active time window."""
        if self.current_time < self.source_start_time or self.current_time >= self.source_end_time:
            return

        self.inflow_advance_accum += self.jet_speed * dt
        h_s = self.particle_spacing
        while self.inflow_advance_accum >= h_s:
            self.inflow_advance_accum -= h_s
            # Spawn a new slice at jet_start
            n_new = len(self.inflow_disk)
            new_pos = np.zeros((n_new, 3), dtype=np.float64)
            for i, (a, b) in enumerate(self.inflow_disk):
                new_pos[i] = self.jet_start + a * self.jet_side + b * self.jet_cross
            new_vel = np.tile(self.jet_speed * self.jet_dir, (n_new, 1))
            new_mass = np.full(n_new, self.particle_mass, dtype=np.float64)

            self.pos = np.vstack([self.pos, new_pos])
            self.vel = np.vstack([self.vel, new_vel])
            self.mass = np.concatenate([self.mass, new_mass])
            self.cumulative_mass_inflow += n_new * self.particle_mass

    def advance_to(self, target_time: float) -> StateFrame:
        # Step physics forward to target_time
        while self.current_time + 1e-9 < target_time:
            dt = min(self.sim_dt, target_time - self.current_time)
            self._emit_inflow(dt)
            self._sph_step(dt)
            self.current_time += dt

        self.current_time = target_time

        # Compute field outputs
        active_mask, liquid_ls, vel_grid = self._compute_fields()

        frame = StateFrame(
            time=float(target_time),
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            attributes={
                "mass_quantity": np.copy(self.mass),
            },
            fields={
                "active_mask": active_mask,
                "liquid_level_set": liquid_ls,
                "velocity_field": vel_grid,
                "boundary_normal": self.boundary_normals,
                "boundary_target_velocity": self.boundary_target_velocities,
                "boundary_velocity": self.boundary_velocities,
                "cumulative_mass_source": np.array([self.cumulative_mass_inflow], dtype=np.float64),
            },
            scalars={
                "characteristic_length": self.char_length,
                "grid_spacing": self.grid_spacing_scalar,
                "reference_velocity": self.ref_velocity,
            },
        )
        return frame

    def _compute_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute diagnostic Eulerian fields: active_mask, liquid_level_set, velocity_field."""
        nx, ny, nz = self.ref_grid
        vel_grid = np.zeros((nx, ny, nz, 3), dtype=np.float64)

        if len(self.pos) > 0:
            tree = cKDTree(self.pos)
            # Query nearest particle for each cell center
            grid_flat = self.grid_coords.reshape(-1, 3)
            dist, idx = tree.query(grid_flat, k=1)
            dist = dist.reshape(nx, ny, nz)
            idx = idx.reshape(nx, ny, nz)

            # SPH particle radius r_p = 0.5 * particle_spacing
            r_p = 0.55 * self.particle_spacing
            # Signed distance: negative inside liquid, positive outside
            liquid_ls = dist - r_p

            # Interpolate velocity onto grid near fluid
            near_mask = dist <= (2.0 * self.particle_spacing)
            vel_flat = self.vel[idx.ravel()]
            vel_grid_flat = np.where(near_mask.reshape(-1, 1), vel_flat, 0.0)
            vel_grid = vel_grid_flat.reshape(nx, ny, nz, 3)
        else:
            liquid_ls = np.full((nx, ny, nz), 1.0, dtype=np.float64)

        # Active mask: cells where liquid is present or near boundary
        active_mask = (liquid_ls <= (self.particle_spacing * 1.5)).astype(np.float64)

        return active_mask, liquid_ls, vel_grid

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": len(self.pos),
            "sim_time": self.current_time,
            "cumulative_mass_inflow": self.cumulative_mass_inflow,
        }

