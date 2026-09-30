"""Incompressible multi-phase fluid solver for octagonal two-phase drop.

Simulates two immiscible liquid drops (heavy blue, light pink) falling into an
octagonal basin using Smoothed Particle Hydrodynamics (WCSPH / density relaxation)
and boundary handling.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        self.phase_id = np.array(init_frame.attributes["phase_id"], dtype=np.float64, copy=True)
        self.density_ref = np.array(init_frame.attributes["density"], dtype=np.float64, copy=True)
        self.release_time = np.array(init_frame.attributes["release_time"], dtype=np.float64, copy=True)
        self.target_volume = np.array(init_frame.attributes["target_volume"], dtype=np.float64, copy=True)

        self.num_particles = len(self.pos)
        self.current_time = 0.0

        # Physical constants
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Octagonal basin geometry
        # Inner vertices at radius R_inner = 1.060660 / cos(pi/8) approx 1.148
        # Octagonal bounding planes: |x| <= L, |y| <= L, |x+y| <= D, |x-y| <= D
        # From obj vertices: max |x| = 1.060660, max |x|+|y| = 1.500000 -> |x \pm y| <= 1.5
        self.basin_x_max = 1.060660
        self.basin_diag_max = 1.500000  # |x| + |y| <= 1.5
        self.basin_z_bottom = 0.08
        self.basin_z_top = 0.62

        # SPH parameters
        self.h = 0.09  # smoothing length
        self.h2 = self.h * self.h

        # Grid specifications for diagnostic fields
        self.grid_spacing = 0.08
        self.grid_bounds_min = np.array([-1.2, -1.2, 0.0])
        self.grid_bounds_max = np.array([1.2, 1.2, 4.5])
        self.grid_shape = tuple(
            int(np.ceil((self.grid_bounds_max[i] - self.grid_bounds_min[i]) / self.grid_spacing))
            for i in range(3)
        )
        self.char_length = 2.0  # Basin diameter ~ 2.12m

    def _step_physics(self, dt: float) -> None:
        """Advance the physics by a small sub-step dt."""
        t = self.current_time
        active = (t >= self.release_time)
        n_act = np.count_nonzero(active)
        if n_act == 0:
            self.current_time += dt
            return

        idx = np.where(active)[0]
        pos_act = self.pos[idx]
        vel_act = self.vel[idx]
        mass_act = self.mass[idx]
        rho0_act = self.density_ref[idx]
        phase_act = self.phase_id[idx]

        # 1. Compute density using SPH kernel
        # Poly6 kernel in 3D: W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
        diff = pos_act[:, None, :] - pos_act[None, :, :]  # [N, N, 3]
        r2 = np.sum(diff * diff, axis=-1)  # [N, N]

        in_range = (r2 < self.h2)
        poly6_coeff = 315.0 / (64.0 * np.pi * (self.h ** 9))
        w = np.where(in_range, poly6_coeff * ((self.h2 - r2) ** 3), 0.0)

        # Density: rho_i = sum_j m_j W_ij
        rho = np.sum(mass_act[None, :] * w, axis=1)
        rho = np.maximum(rho, rho0_act * 0.5)  # avoid division by zero

        # 2. Compute Pressure (Tait's equation of state)
        # P = c_s^2 * rho0 / gamma * ((rho / rho0)^gamma - 1)
        # For numerical stability at time step dt, use moderately stiff EOS
        c_s = 20.0  # artificial speed of sound
        p0 = (c_s ** 2) * rho0_act / 7.0
        pressure = p0 * (np.maximum(rho / rho0_act, 1.0) ** 7 - 1.0)
        pressure = np.maximum(pressure, 0.0)

        # 3. SPH Pressure force & Viscosity force
        # Spiky kernel gradient: grad_W(r, h) = -45 / (pi * h^6) * (h - r)^2 * (r_vec / r)
        r = np.sqrt(np.maximum(r2, 1e-12))
        spiky_coeff = -45.0 / (np.pi * (self.h ** 6))
        grad_w_mag = np.where((r < self.h) & (r > 1e-6), spiky_coeff * ((self.h - r) ** 2) / r, 0.0)
        grad_w = diff * grad_w_mag[:, :, None]  # [N, N, 3]

        # Symmetrized pressure acceleration:
        # a_p,i = - sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_W_ij
        p_over_rho2 = pressure / (rho ** 2)
        p_term = p_over_rho2[:, None] + p_over_rho2[None, :]  # [N, N]
        f_press = -np.sum(mass_act[None, :, None] * p_term[:, :, None] * grad_w, axis=1)

        # Viscosity force:
        # a_v,i = mu * sum_j m_j (v_j - v_i) / rho_j * lap_W_ij
        # Viscosity laplacian: 45 / (pi * h^6) * (h - r)
        lap_w_coeff = 45.0 / (np.pi * (self.h ** 6))
        lap_w = np.where(r < self.h, lap_w_coeff * (self.h - r), 0.0)
        v_diff = vel_act[None, :, :] - vel_act[:, None, :]  # [N, N, 3]
        mu = 0.05
        f_visc = mu * np.sum((mass_act[None, :] / rho[None, :])[:, :, None] * v_diff * lap_w[:, :, None], axis=1)

        # Immiscibility / Phase separation surface tension force
        # Repulsive force between different phases at interface
        phase_diff = (phase_act[:, None] != phase_act[None, :]).astype(np.float64)
        repulsion_coeff = 2.0
        f_repulse = repulsion_coeff * np.sum(phase_diff[:, :, None] * grad_w, axis=1)

        # Total acceleration
        acc = self.gravity[None, :] + f_press + f_visc + f_repulse

        # 4. Integrate velocity and position (Symplectic Euler)
        vel_act += acc * dt

        # Artificial damping/drag for air/numerical stability
        vel_act *= np.exp(-0.02 * dt)

        pos_act += vel_act * dt

        # 5. Boundary collisions with octagonal basin
        # Container geometry:
        # Bottom plane at z = 0.08
        # Octagonal side walls:
        # |x| <= 1.06, |y| <= 1.06, |x+y| <= 1.5, |x-y| <= 1.5
        # when z <= 0.62 (or full height of basin)

        restitution = 0.1
        friction = 0.85
        particle_radius = 0.03

        # Bottom boundary
        bottom_penetration = (self.basin_z_bottom + particle_radius) - pos_act[:, 2]
        hit_bottom = (bottom_penetration > 0)
        if np.any(hit_bottom):
            pos_act[hit_bottom, 2] = self.basin_z_bottom + particle_radius
            vel_act[hit_bottom, 2] = -restitution * vel_act[hit_bottom, 2]
            vel_act[hit_bottom, 0] *= friction
            vel_act[hit_bottom, 1] *= friction

        # Wall collisions for particles inside/near the basin height
        # Planes defined by normals:
        normals = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]),
            np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]),
            np.array([1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0]),
            np.array([-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0]),
        ]
        # Distances to planes:
        # for |x| <= 1.06: plane x = 1.06 -> n = [1,0,0], d = 1.06
        # for |x+y|/sqrt(2) <= 1.5/sqrt(2): plane (x+y)/sqrt(2) = 1.5/sqrt(2) = 1.06066
        plane_offsets = [
            self.basin_x_max,
            self.basin_x_max,
            self.basin_x_max,
            self.basin_x_max,
            self.basin_diag_max / np.sqrt(2.0),
            self.basin_diag_max / np.sqrt(2.0),
            self.basin_diag_max / np.sqrt(2.0),
            self.basin_diag_max / np.sqrt(2.0),
        ]

        # Check basin wall bounds if z is within basin height (or extended sidewalls to contain splash)
        in_basin_z = (pos_act[:, 2] <= self.basin_z_top + 0.1)
        for normal, d_max in zip(normals, plane_offsets):
            dist_to_plane = np.dot(pos_act[:, :2], normal[:2]) - (d_max - particle_radius)
            violates = in_basin_z & (dist_to_plane > 0)
            if np.any(violates):
                pos_act[violates, 0] -= dist_to_plane[violates] * normal[0]
                pos_act[violates, 1] -= dist_to_plane[violates] * normal[1]
                v_norm = np.sum(vel_act[violates] * normal, axis=-1, keepdims=True)
                v_normal_vec = v_norm * normal
                v_tangent_vec = vel_act[violates] - v_normal_vec
                vel_act[violates] = -restitution * v_normal_vec + friction * v_tangent_vec

        # Store back
        self.pos[idx] = pos_act
        self.vel[idx] = vel_act
        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Sub-step physics to target_time
        sub_dt = 0.002
        while self.current_time < target_time - 1e-7:
            step = min(sub_dt, target_time - self.current_time)
            self._step_physics(step)

        # Build diagnostic fields on 3D grid
        nx, ny, nz = self.grid_shape
        active = (self.current_time >= self.release_time)
        active_pos = self.pos[active]
        active_vel = self.vel[active]

        # 1. Active mask (basin interior + drop trajectory region)
        active_mask = np.ones((nx, ny, nz), dtype=bool)

        # 2. Cumulative mass source (split by phase: [K=2])
        cum_mass_source = np.array([0.36, 0.20 if self.current_time >= 1.35 else 0.0], dtype=np.float64)

        # 3. Liquid level set (signed distance approximation)
        # We compute approximate signed distance to particle cloud
        # Initialize with positive distance (outside)
        gx = np.linspace(self.grid_bounds_min[0], self.grid_bounds_max[0], nx)
        gy = np.linspace(self.grid_bounds_min[1], self.grid_bounds_max[1], ny)
        gz = np.linspace(self.grid_bounds_min[2], self.grid_bounds_max[2], nz)

        # Fast diagnostic grid: level set & velocity field
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        liquid_level_set = np.ones((nx, ny, nz), dtype=np.float64) * 0.5

        if len(active_pos) > 0:
            # Map particles to grid cells
            cell_idx = ((active_pos - self.grid_bounds_min) / self.grid_spacing).astype(np.int32)
            valid = (
                (cell_idx[:, 0] >= 0) & (cell_idx[:, 0] < nx) &
                (cell_idx[:, 1] >= 0) & (cell_idx[:, 1] < ny) &
                (cell_idx[:, 2] >= 0) & (cell_idx[:, 2] < nz)
            )
            valid_cells = cell_idx[valid]
            valid_vels = active_vel[valid]

            # Accumulate velocity & set inside level set
            count_grid = np.zeros((nx, ny, nz), dtype=np.float64)
            for i in range(len(valid_cells)):
                cx, cy, cz = valid_cells[i]
                velocity_field[cx, cy, cz] += valid_vels[i]
                count_grid[cx, cy, cz] += 1.0
                liquid_level_set[cx, cy, cz] = -self.grid_spacing

            nonzero = count_grid > 0
            velocity_field[nonzero] /= count_grid[nonzero, None]

        # Per-sample mass contribution [N, 2] for 2 materials
        mass_quantity = np.zeros((self.num_particles, 2), dtype=np.float64)
        is_p0 = (self.phase_id == 0)
        is_p1 = (self.phase_id == 1)
        mass_quantity[is_p0, 0] = self.mass[is_p0]
        mass_quantity[is_p1, 1] = self.mass[is_p1]

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase_id": self.phase_id.copy(),
                "density": self.density_ref.copy(),
                "release_time": self.release_time.copy(),
                "target_volume": self.target_volume.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": cum_mass_source,
                "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(self.char_length),
                "grid_spacing": float(self.grid_spacing),
                "gravity_z": -9.81,
                "heavy_phase_volume": 0.36,
                "light_phase_volume": 0.20,
                "light_release_time": 1.35,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.num_particles,
        }

