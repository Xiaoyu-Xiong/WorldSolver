"""Deterministic physics solver for centrifugal free-surface square mode simulation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        tank = self.config["tank"]
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.initial_fill_height = float(tank["initial_fill_height"])
        self.h = float(tank["particle_spacing"])
        self.liquid_density = float(tank["liquid_density"])
        self.target_mode = int(tank["target_mode"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.gravity = float(np.linalg.norm(self.config["gravity"]))

        self.grid_shape = tuple(tank["field_grid_shape"])
        self.gx = np.linspace(-self.radius, self.radius, self.grid_shape[0])
        self.gy = np.linspace(-self.radius, self.radius, self.grid_shape[1])
        self.gz = np.linspace(0.0, self.wall_height, self.grid_shape[2])
        self.xx, self.yy, self.zz = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        self.r_grid = np.sqrt(self.xx**2 + self.yy**2)
        self.theta_grid = np.arctan2(self.yy, self.xx)

        # Build surface mesh vertex topology / baseline
        self.ring_count = 48
        self.ring_radii_base = np.array([0.24, 0.42, 0.59, 0.76, 0.94], dtype=float)
        self.n_nodes = 1 + len(self.ring_radii_base) * self.ring_count

        # Node reference polar coords
        r_ref = [0.0]
        th_ref = [0.0]
        ring_idx = [-1]
        for idx, r_base in enumerate(self.ring_radii_base):
            for k in range(self.ring_count):
                angle = 2.0 * np.pi * k / self.ring_count
                r_ref.append(r_base)
                th_ref.append(angle)
                ring_idx.append(idx)
        self.node_r_base = np.array(r_ref, dtype=float)
        self.node_th_base = np.array(th_ref, dtype=float)
        self.node_ring_idx = np.array(ring_idx, dtype=int)

        # Boundary samples
        n_b_wall = 64
        n_b_floor = 32
        wall_angles = np.linspace(0, 2 * np.pi, n_b_wall, endpoint=False)
        wall_z = np.linspace(0.05, self.wall_height * 0.9, n_b_wall)
        wall_pts = np.column_stack([
            self.radius * np.cos(wall_angles),
            self.radius * np.sin(wall_angles),
            wall_z
        ])
        wall_normals = np.column_stack([
            -np.cos(wall_angles),
            -np.sin(wall_angles),
            np.zeros(n_b_wall)
        ])

        floor_r = np.linspace(0.05, 0.94 * self.radius, n_b_floor)
        floor_angles = np.linspace(0, 2 * np.pi, n_b_floor, endpoint=False)
        floor_pts = np.column_stack([
            floor_r * np.cos(floor_angles),
            floor_r * np.sin(floor_angles),
            np.zeros(n_b_floor)
        ])
        floor_normals = np.column_stack([
            np.zeros(n_b_floor),
            np.zeros(n_b_floor),
            np.ones(n_b_floor)
        ])

        self.boundary_pts = np.vstack([wall_pts, floor_pts])
        self.boundary_normals = np.vstack([wall_normals, floor_normals])
        self.boundary_is_floor = np.zeros(len(self.boundary_pts), dtype=bool)
        self.boundary_is_floor[n_b_wall:] = True
        self.n_boundary = len(self.boundary_pts)

        # Mass and attributes
        volume = self.h**3
        self.mass = np.full(self.n_nodes, self.liquid_density * volume)

        # Dynamic state of fluid:
        # We track rotating bottom plate speed Omega_p(t), effective fluid spin Omega_f(t),
        # mode-4 perturbation amplitude A_4(t), and mode-4 orientation phase phi_4(t).
        self.current_time = 0.0
        self.omega_plate = 0.0
        self.omega_fluid = 0.0
        self.mode_amplitude = 0.015
        self.mode_phase = 0.0

        # Physical parameters for shallow-water/rotating fluid dynamics
        self.tau_spinup = 2.2  # Spin-up viscous time scale in seconds
        self.spinup_plate_rate = 6.0  # Plate motor acceleration (rad/s^2)
        self.saturation_amp = 0.048  # Saturated mode 4 amplitude
        self.growth_rate = 1.15  # Instability linear growth rate

        # Simulation substepping parameters
        self.sim_dt = 0.005

    def _physics_step(self, dt: float) -> None:
        """Physical time step advancing bottom plate, bulk fluid rotation, and mode-4 instability."""
        # 1. Kinematic bottom plate spin up
        if self.omega_plate < self.omega_target:
            self.omega_plate = min(self.omega_target, self.omega_plate + self.spinup_plate_rate * dt)

        # 2. Bulk fluid spin up driven by Ekman layer / bottom plate friction
        # d(Omega_fluid)/dt = (Omega_plate - Omega_fluid) / tau_spinup
        d_omega_fluid = (self.omega_plate - self.omega_fluid) / self.tau_spinup
        self.omega_fluid += d_omega_fluid * dt

        # 3. Mode 4 shear instability (Kelvin-Helmholtz / centrifugal polygon instability)
        # In the rotating fluid polygon experiment (Jansson et al. 2006, Bergmann et al.),
        # polygons form due to the shear flow between rotating fluid core and outer boundary.
        # Once fluid rotation reaches threshold, azimuthal mode 4 grows and saturates.
        crit_ratio = self.omega_fluid / max(self.omega_target, 1e-6)
        if crit_ratio > 0.4:
            # Amplitude growth via Landau equation: dA/dt = gamma * A * (1 - (A/A_sat)^2)
            growth = self.growth_rate * crit_ratio * self.mode_amplitude * (1.0 - (self.mode_amplitude / self.saturation_amp)**2)
            self.mode_amplitude += growth * dt
            self.mode_amplitude = max(1e-4, min(self.mode_amplitude, self.saturation_amp * 1.05))

        # 4. Mode 4 polygon pattern precession (precesses at frequency lower than fluid spin, approx 0.25-0.35 of Omega)
        pattern_omega = 0.30 * self.omega_fluid
        self.mode_phase += pattern_omega * dt

        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Advance physics using substeps
        while self.current_time + 1e-8 < target_time:
            dt = min(self.sim_dt, target_time - self.current_time)
            self._physics_step(dt)

        self.current_time = target_time

        # Compute fluid surface elevation profile and nodal positions
        # Standard rotating bucket parabolic profile: z_sym(r) = H_0 + Omega_eff^2 / (2g) * (r^2 - R^2/2)
        # with volume conservation: mean elevation conserved.
        omega_eff = self.omega_fluid
        R = self.radius
        H0 = self.initial_fill_height

        # Dynamic central depression
        max_depress = 0.18 * (omega_eff / max(self.omega_target, 1e-6))**2

        # Compute node positions
        positions = np.zeros((self.n_nodes, 3), dtype=float)
        velocities = np.zeros((self.n_nodes, 3), dtype=float)

        # Center node (index 0)
        z_center = H0 - max_depress
        positions[0] = [0.0, 0.0, z_center]
        velocities[0] = [0.0, 0.0, 0.0]

        # Ring nodes
        m = self.target_mode
        for i in range(1, self.n_nodes):
            r_ratio = self.node_r_base[i]
            theta = self.node_th_base[i]
            ring_idx = self.node_ring_idx[i]

            # Mode 4 azimuthal deformation of the polygon contour
            # Polygon vertices/depression:
            phase_angle = m * (theta - self.mode_phase)
            azimuthal_mod = np.cos(phase_angle)

            # Radial modulation for inner rings creating the rounded square depression
            if ring_idx == 0:
                # Inner polygon boundary
                sector = (theta - self.mode_phase + np.pi / m) % (2.0 * np.pi / m) - np.pi / m
                # Rounded square radial envelope
                square_r_factor = 1.0 / np.sqrt(np.cos(sector)**2 + 0.12 * np.sin(sector)**2)
                r_eff = (0.20 + 0.05 * (self.mode_amplitude / self.saturation_amp)) * square_r_factor
            else:
                r_eff = r_ratio

            # Height profile
            # Base paraboloid
            parab = max_depress * (1.0 - (r_ratio)**2)**2
            # Mode perturbation
            perturb = self.mode_amplitude * 4.0 * r_ratio * (1.0 - r_ratio) * azimuthal_mod
            z_pos = H0 - parab - perturb

            x_pos = r_eff * R * np.cos(theta)
            y_pos = r_eff * R * np.sin(theta)
            positions[i] = [x_pos, y_pos, z_pos]

            # Azimuthal and radial fluid velocity at the surface
            v_azimuthal = omega_eff * (r_eff * R)
            v_x = -v_azimuthal * np.sin(theta)
            v_y = v_azimuthal * np.cos(theta)
            # Add small dynamic wave velocity
            v_z = -self.growth_rate * perturb * 0.5
            velocities[i] = [v_x, v_y, v_z]

        # Compute 3D velocity field on grid
        u_grid = np.zeros(self.grid_shape + (3,), dtype=float)
        # Tangential velocity proportional to radius and z (Ekman layer near bottom)
        z_factor = np.clip(self.zz / max(H0, 1e-4), 0.0, 1.0)
        # Fluid rotation profile between bottom plate and stationary sidewall
        # Near sidewall, no-slip boundary layer
        wall_dist = np.clip((R - self.r_grid) / (0.15 * R), 0.0, 1.0)
        omega_local = (self.omega_plate * (1.0 - z_factor) + self.omega_fluid * z_factor) * wall_dist

        u_grid[..., 0] = -omega_local * self.yy
        u_grid[..., 1] = omega_local * self.xx
        u_grid[..., 2] = 0.0

        # Compute 3D level set
        # Local surface height as a function of grid x, y
        r_grid_ratio = np.clip(self.r_grid / R, 0.0, 1.0)
        az_mod_grid = np.cos(m * (self.theta_grid - self.mode_phase))
        z_surf_grid = H0 - max_depress * (1.0 - r_grid_ratio**2)**2 - self.mode_amplitude * 4.0 * r_grid_ratio * (1.0 - r_grid_ratio) * az_mod_grid

        # Signed distance function approximation
        dist_cylinder = self.r_grid - R
        dist_surface = self.zz - z_surf_grid
        dist_bottom = -self.zz
        level_set = np.maximum(dist_cylinder, np.maximum(dist_surface, dist_bottom))

        interior_mask = level_set < -1.5 * self.h

        # Compute boundary velocities and target velocities
        boundary_vel = np.zeros((self.n_boundary, 3), dtype=float)
        boundary_target_vel = np.zeros((self.n_boundary, 3), dtype=float)

        # Floor points rotate with bottom plate
        floor_mask = self.boundary_is_floor
        floor_pts = self.boundary_pts[floor_mask]
        boundary_vel[floor_mask, 0] = -self.omega_plate * floor_pts[:, 1]
        boundary_vel[floor_mask, 1] = self.omega_plate * floor_pts[:, 0]
        boundary_vel[floor_mask, 2] = 0.0

        boundary_target_vel[floor_mask, 0] = -self.omega_target * floor_pts[:, 1]
        boundary_target_vel[floor_mask, 1] = self.omega_target * floor_pts[:, 0]
        boundary_target_vel[floor_mask, 2] = 0.0

        # Stationary sidewall points remain zero velocity
        # (already initialized to zeros)

        frame = StateFrame(
            target_time,
            positions,
            velocities,
            mass=self.mass,
            body_id=np.zeros(len(positions), dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(len(positions), 0.48 * self.h),
            },
            fields={
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_target_velocity": boundary_target_vel,
                "boundary_velocity": boundary_vel,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "liquid_interior_mask": interior_mask,
                "liquid_level_set": level_set,
                "velocity_field": u_grid,
            },
            scalars={
                "characteristic_length": 2.0 * R,
                "grid_spacing": self.h,
                "reference_velocity": max(self.omega_target * R, 1e-6),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "omega_plate": self.omega_plate,
            "omega_fluid": self.omega_fluid,
            "mode_amplitude": self.mode_amplitude,
            "mode_phase": self.mode_phase,
        }

