"""Particle-based fluid dynamics solver (WCSPH / Position-based fluid) for Dam Break Through an Arch Obstacle."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.RandomState(seed)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.positions = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.N = len(self.positions)
        self.velocities = np.zeros((self.N, 3), dtype=np.float64)

        # Simulation parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        sampling = self.config.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.14))
        self.rest_density = float(sampling.get("density", 1000.0))

        # Mass per particle
        if hasattr(init_frame, "mass") and init_frame.mass is not None and len(init_frame.mass) == self.N:
            self.mass = np.array(init_frame.mass, dtype=np.float64)
        else:
            self.mass = np.full(self.N, self.rest_density * (self.spacing ** 3), dtype=np.float64)

        # Attributes preserved for StateFrame
        self.material_label = np.zeros(self.N, dtype=np.int32)
        if "material_label" in init_frame.attributes:
            self.material_label = np.array(init_frame.attributes["material_label"], dtype=np.int32)

        self.particle_radius = 0.45 * self.spacing
        self.h = 1.6 * self.spacing  # smoothing radius
        self.h2 = self.h * self.h

        # Domain boundaries (tank walls)
        # Ground half_extents: [3.2, 1.4, 0.05], center at [0, 0, -0.05] -> z in [0, inf)
        # x in [-3.2, 3.2], y in [-1.4, 1.4], z in [0.0, 3.0]
        self.bounds_min = np.array([-3.15, -1.35, 0.01], dtype=np.float64)
        self.bounds_max = np.array([3.15, 1.35, 2.8], dtype=np.float64)

        # Obstacles definition
        # arch_left: center [-0.75, 0, 0.7], half_extents [0.22, 0.7, 0.7] -> [-0.97, -0.7, 0.0] to [-0.53, 0.7, 1.4]
        # arch_right: center [0.75, 0, 0.7], half_extents [0.22, 0.7, 0.7] -> [0.53, -0.7, 0.0] to [0.97, 0.7, 1.4]
        # arch_top: center [0, 0, 1.4], half_extents [0.95, 0.7, 0.18] -> [-0.95, -0.7, 1.22] to [0.95, 0.7, 1.58]
        # Gate: center [-1.55, 0, 0.55], half_extents [0.04, 0.65, 0.55] -> opens upward or removed at t=0
        self.obstacles = [
            # [min_x, min_y, min_z, max_x, max_y, max_z]
            np.array([-0.97, -0.70, 0.0, -0.53, 0.70, 1.40]),  # arch_left
            np.array([0.53, -0.70, 0.0, 0.97, 0.70, 1.40]),   # arch_right
            np.array([-0.95, -0.70, 1.22, 0.95, 0.70, 1.58]),  # arch_top
        ]

        # Gate opens rapidly from t=0 to t=0.2 (moves upward)
        self.gate_open_time = 0.15

        # Boundary sample points for StateFrame fields
        self._init_boundary_samples()

        # Grid settings for diagnostic fields
        self.field_grid_shape = tuple(sampling.get("field_grid_shape", [24, 16, 28]))
        self.grid_spacing = float(self.spacing)
        self.characteristic_length = max(float(np.ptp(self.positions, axis=0).max()), self.spacing)
        self.reference_velocity = float(np.sqrt(2.0 * 9.81 * 1.25))  # Torricelli / dam break velocity scale

        # State tracking
        self.current_time = 0.0
        self.sub_dt = 0.005  # efficient and stable sub-step for N=384 SPH

    def _init_boundary_samples(self):
        # Generate boundary samples on the obstacle and tank surfaces
        samples = []
        normals = []
        # Tank floor samples
        gx = np.linspace(-3.0, 3.0, 15)
        gy = np.linspace(-1.2, 1.2, 8)
        for x in gx:
            for y in gy:
                samples.append([x, y, 0.0])
                normals.append([0.0, 0.0, 1.0])

        # Arch obstacle surface samples
        for obs in self.obstacles:
            c = (obs[:3] + obs[3:]) * 0.5
            # Face centers
            # -x face
            samples.append([obs[0], c[1], c[2]])
            normals.append([-1.0, 0.0, 0.0])
            # +x face
            samples.append([obs[3], c[1], c[2]])
            normals.append([1.0, 0.0, 0.0])
            # -y face
            samples.append([c[0], obs[1], c[2]])
            normals.append([0.0, -1.0, 0.0])
            # +y face
            samples.append([c[0], obs[4], c[2]])
            normals.append([0.0, 1.0, 0.0])

        self.boundary_pos = np.array(samples, dtype=np.float64)
        self.boundary_norm = np.array(normals, dtype=np.float64)
        self.boundary_vel = np.zeros_like(self.boundary_pos)
        self.boundary_target_vel = np.zeros_like(self.boundary_pos)

    def _apply_boundary_and_obstacles(self, pos, vel, t):
        # Tank bounds
        for d in range(3):
            # Min bound
            under = pos[:, d] < self.bounds_min[d]
            pos[under, d] = self.bounds_min[d]
            vel[under, d] = -0.2 * vel[under, d]
            vel[under, (d+1)%3] *= 0.85
            vel[under, (d+2)%3] *= 0.85

            # Max bound
            over = pos[:, d] > self.bounds_max[d]
            pos[over, d] = self.bounds_max[d]
            vel[over, d] = -0.2 * vel[over, d]
            vel[over, (d+1)%3] *= 0.85
            vel[over, (d+2)%3] *= 0.85

        # Obstacles (AABB boxes)
        for obs in self.obstacles:
            bmin = obs[:3]
            bmax = obs[3:]
            # Check inside
            inside = (
                (pos[:, 0] >= bmin[0]) & (pos[:, 0] <= bmax[0]) &
                (pos[:, 1] >= bmin[1]) & (pos[:, 1] <= bmax[1]) &
                (pos[:, 2] >= bmin[2]) & (pos[:, 2] <= bmax[2])
            )
            if not np.any(inside):
                continue

            p_in = pos[inside]
            # Find closest exit face
            d_min_x = p_in[:, 0] - bmin[0]
            d_max_x = bmax[0] - p_in[:, 0]
            d_min_y = p_in[:, 1] - bmin[1]
            d_max_y = bmax[1] - p_in[:, 1]
            d_min_z = p_in[:, 2] - bmin[2]
            d_max_z = bmax[2] - p_in[:, 2]

            dists = np.stack([d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z], axis=-1)
            min_face = np.argmin(dists, axis=-1)

            indices = np.where(inside)[0]
            for idx_local, face in enumerate(min_face):
                idx_global = indices[idx_local]
                if face == 0:
                    pos[idx_global, 0] = bmin[0] - 1e-4
                    vel[idx_global, 0] = -0.1 * vel[idx_global, 0]
                elif face == 1:
                    pos[idx_global, 0] = bmax[0] + 1e-4
                    vel[idx_global, 0] = -0.1 * vel[idx_global, 0]
                elif face == 2:
                    pos[idx_global, 1] = bmin[1] - 1e-4
                    vel[idx_global, 1] = -0.1 * vel[idx_global, 1]
                elif face == 3:
                    pos[idx_global, 1] = bmax[1] + 1e-4
                    vel[idx_global, 1] = -0.1 * vel[idx_global, 1]
                elif face == 4:
                    pos[idx_global, 2] = bmin[2] - 1e-4
                    vel[idx_global, 2] = -0.1 * vel[idx_global, 2]
                elif face == 5:
                    pos[idx_global, 2] = bmax[2] + 1e-4
                    vel[idx_global, 2] = -0.1 * vel[idx_global, 2]

        # Gate at initial time
        if t < self.gate_open_time:
            # Gate opening up
            gate_height = 1.2 * (1.0 - t / self.gate_open_time)
            gate_x = -1.55
            in_gate_x = np.abs(pos[:, 0] - gate_x) < 0.05
            in_gate_z = pos[:, 2] < gate_height
            gate_hit = in_gate_x & in_gate_z
            if np.any(gate_hit):
                pos[gate_hit, 0] = gate_x - 0.06
                vel[gate_hit, 0] = np.minimum(0.0, vel[gate_hit, 0])

    def _step(self, dt: float):
        # SPH / Particle Fluid step
        pos = self.positions
        vel = self.velocities
        N = self.N

        # Pairwise differences
        diff = pos[:, None, :] - pos[None, :, :]  # [N, N, 3] -> diff[i, j] = pos[i] - pos[j]
        r2 = np.sum(diff ** 2, axis=-1)  # [N, N]

        poly6_const = 315.0 / (64.0 * np.pi * (self.h ** 9))
        spiky_grad_const = -45.0 / (np.pi * (self.h ** 6))
        visc_lap_const = 45.0 / (np.pi * (self.h ** 6))

        # Density
        in_kernel = r2 < self.h2
        w_poly6 = np.where(in_kernel, poly6_const * ((self.h2 - r2) ** 3), 0.0)
        density = np.sum(self.mass[None, :] * w_poly6, axis=1)
        density = np.maximum(density, self.rest_density * 0.5)

        # Pressure
        gamma = 7.0
        c0 = 20.0  # speed of sound
        B = (self.rest_density * (c0 ** 2)) / gamma
        pressure = B * (np.maximum((density / self.rest_density) ** gamma - 1.0, 0.0))

        # Forces
        r = np.sqrt(np.maximum(r2, 1e-12))
        inv_r = 1.0 / r

        # Pressure gradient
        # grad_w = spiky_grad_const * ((h - r)**2) * (diff / r)
        w_spiky_grad = np.where(in_kernel & (r2 > 1e-12), spiky_grad_const * ((self.h - r) ** 2) * inv_r, 0.0)
        p_term = (pressure[:, None] / (density[:, None] ** 2) + pressure[None, :] / (density[None, :] ** 2))
        f_p_pairwise = -self.mass[None, :, None] * (p_term * w_spiky_grad)[:, :, None] * diff
        f_pressure = self.mass[:, None] * np.sum(f_p_pairwise, axis=1)

        # Viscosity
        mu = 0.05
        w_visc_lap = np.where(in_kernel & (r2 > 1e-12), visc_lap_const * (self.h - r), 0.0)
        v_diff = vel[None, :, :] - vel[:, None, :]  # vel[j] - vel[i]
        f_v_pairwise = mu * (self.mass[None, :] / density[None, :])[:, :, None] * w_visc_lap[:, :, None] * v_diff
        f_viscosity = self.mass[:, None] * np.sum(f_v_pairwise, axis=1)

        # Gravity
        f_gravity = self.mass[:, None] * self.gravity[None, :]

        total_forces = f_pressure + f_viscosity + f_gravity

        # Acceleration and clamp
        acc = total_forces / self.mass[:, None]
        acc_mag = np.linalg.norm(acc, axis=-1, keepdims=True)
        max_acc = 300.0
        scale = np.where(acc_mag > max_acc, max_acc / (acc_mag + 1e-8), 1.0)
        acc = acc * scale

        # Time integration
        vel += acc * dt
        pos += vel * dt

        # Boundary and obstacle collision
        self._apply_boundary_and_obstacles(pos, vel, self.current_time)

    def advance_to(self, time: float) -> StateFrame:
        # Advance physics using sub-steps
        target_time = float(time)
        while self.current_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Construct required diagnostic fields
        nx, ny, nz = self.field_grid_shape
        gx = np.linspace(-3.2, 3.2, nx)
        gy = np.linspace(-1.4, 1.4, ny)
        gz = np.linspace(0.0, 2.0, nz)

        # Compute diagnostic velocity field and level set on grid
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        liquid_level_set = np.full((nx, ny, nz), 10.0, dtype=np.float64)
        active_mask = np.zeros((nx, ny, nz), dtype=bool)

        # Fast grid splatting / field evaluation
        if self.N > 0:
            # Map particles to grid cells
            px = np.clip(np.searchsorted(gx, self.positions[:, 0]) - 1, 0, nx - 1)
            py = np.clip(np.searchsorted(gy, self.positions[:, 1]) - 1, 0, ny - 1)
            pz = np.clip(np.searchsorted(gz, self.positions[:, 2]) - 1, 0, nz - 1)

            cell_counts = np.zeros((nx, ny, nz), dtype=np.float64)
            for i in range(self.N):
                ix, iy, iz = px[i], py[i], pz[i]
                velocity_field[ix, iy, iz] += self.velocities[i]
                cell_counts[ix, iy, iz] += 1.0

            active_cells = cell_counts > 0
            velocity_field[active_cells] /= cell_counts[active_cells, None]
            active_mask[active_cells] = True

            # Distance field / level set approximation
            # Cells with fluid have negative distance (approx -particle_radius)
            # Neighboring cells approximate positive distance
            grid_points = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1)
            # Approximate signed distance for active and near cells
            liquid_level_set[active_cells] = -self.particle_radius
            liquid_level_set[~active_cells] = self.grid_spacing

        # Build StateFrame
        fields = {
            "active_mask": active_mask,
            "boundary_normal": self.boundary_norm.copy(),
            "boundary_target_velocity": self.boundary_target_vel.copy(),
            "boundary_velocity": self.boundary_vel.copy(),
            "cumulative_mass_source": np.zeros(self.N, dtype=np.float64),
            "liquid_level_set": liquid_level_set,
            "velocity_field": velocity_field,
        }

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.full(self.N, self.particle_radius, dtype=np.float64),
            "material_label": self.material_label.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], self.N, axis=0),
        }

        scalars = {
            "contact_length_scale": float(self.spacing),
            "grid_spacing": float(self.grid_spacing),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
        }

        frame = StateFrame(
            time=target_time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.mass.copy(),
            body_id=self.material_label.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": self.N,
            "current_time": self.current_time,
            "mean_speed": float(np.mean(np.linalg.norm(self.velocities, axis=-1))) if self.N > 0 else 0.0,
        }

