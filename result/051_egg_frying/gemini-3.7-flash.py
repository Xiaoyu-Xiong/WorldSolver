"""Solver implementation for egg frying simulation."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Multiphysics particle solver for egg frying on a hot pan."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.config = scene.config

        # Get initial state from scene
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.dt_sub = 0.002  # internal sub-step dt (10 steps per sample dt = 0.02)

        # Particle states
        self.position = np.array(init_frame.position, dtype=np.float64)
        self.velocity = np.array(init_frame.velocity, dtype=np.float64)
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32)
        self.num_particles = len(self.position)

        # Initial attributes
        if "material_concentration" in init_frame.attributes:
            self.concentration = np.array(init_frame.attributes["material_concentration"], dtype=np.float64)
        else:
            self.concentration = np.zeros((self.num_particles, 2), dtype=np.float64)
            self.concentration[self.body_id == 1, 0] = 1.0
            self.concentration[self.body_id == 2, 1] = 1.0

        if "material_label" in init_frame.attributes:
            self.material_label = np.array(init_frame.attributes["material_label"], dtype=np.float64)
        else:
            self.material_label = self.body_id.astype(np.float64)

        if "temperature_k" in init_frame.attributes:
            self.temperature = np.array(init_frame.attributes["temperature_k"], dtype=np.float64)
        else:
            self.temperature = np.full(self.num_particles, float(self.config.get("initial_conditions", {}).get("egg_temperature_k", 291.0)), dtype=np.float64)

        if "solid_phase_fraction" in init_frame.attributes:
            self.solid_phase = np.array(init_frame.attributes["solid_phase_fraction"], dtype=np.float64)
        else:
            self.solid_phase = np.zeros(self.num_particles, dtype=np.float64)

        if "liquid_phase_fraction" in init_frame.attributes:
            self.liquid_phase = np.array(init_frame.attributes["liquid_phase_fraction"], dtype=np.float64)
        else:
            self.liquid_phase = 1.0 - self.solid_phase

        # Boundary conditions & parameters
        self.pan_z = float(self.config.get("boundary_conditions", {}).get("pan_contact_height_m", 0.018))
        self.pan_radius = float(self.config.get("boundary_conditions", {}).get("pan_no_escape_radius_m", 0.185))
        self.pan_temp = float(self.config.get("boundary_conditions", {}).get("pan_surface_temperature_k", 443.0))
        self.ambient_temp = float(self.config.get("initial_conditions", {}).get("ambient_temperature_k", 296.0))

        gravity_cfg = self.config.get("initial_conditions", {}).get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.array(gravity_cfg, dtype=np.float64)

        # Physics parameters
        # SPH / Particle parameters
        self.h = 0.012  # SPH smoothing radius
        self.rest_density = 1030.0  # kg/m^3 (egg white / yolk density)
        self.gas_stiffness_white = 80.0  # raw fluid stiffness
        self.gas_stiffness_yolk = 120.0
        self.viscosity_white_liquid = 0.05
        self.viscosity_white_solid = 8.0  # high viscosity when cooked
        self.viscosity_yolk = 0.4

        # Thermal parameters
        # Egg white cooks (denatures) between 62 C (335 K) and 80 C (353 K)
        # Pan heat transfer coefficient
        self.thermal_diffusivity = 1.4e-7  # m^2/s thermal conduction in egg
        self.pan_heat_transfer_rate = 8.0  # 1/s near pan surface
        self.T_cook_start = 335.0  # K (~62 C)
        self.T_cook_end = 355.0  # K (~82 C)

        # Multi-material interaction force tracking
        self.interaction_force = np.zeros((2, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(2, dtype=np.float64)

        # Diagnostic Eulerian grid setup
        self.grid_spacing = 0.01  # 1 cm grid
        self.grid_bounds_min = np.array([-0.18, -0.18, 0.018])
        self.grid_bounds_max = np.array([0.18, 0.18, 0.088])
        self.nx = int(np.ceil((self.grid_bounds_max[0] - self.grid_bounds_min[0]) / self.grid_spacing))
        self.ny = int(np.ceil((self.grid_bounds_max[1] - self.grid_bounds_min[1]) / self.grid_spacing))
        self.nz = int(np.ceil((self.grid_bounds_max[2] - self.grid_bounds_min[2]) / self.grid_spacing))

        self.contact_length_scale = 0.005  # 5 mm particle radius / contact thickness

    def _step_physics(self, dt: float) -> None:
        """Advance the physics by dt."""
        N = self.num_particles
        pos = self.position
        vel = self.velocity
        mass = self.mass
        body_id = self.body_id
        temp = self.temperature
        solid = self.solid_phase

        # 1. Neighbor search
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=self.h, output_type="ndarray")

        # 2. Density computation
        density = np.zeros(N, dtype=np.float64)
        # Self density: W(0, h) = 315 / (64 * pi * h^3)
        w0 = 315.0 / (64.0 * np.pi * (self.h ** 3))
        density += mass * w0

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            rij = pos[i_idx] - pos[j_idx]
            r2 = np.sum(rij * rij, axis=1)
            h2 = self.h * self.h
            valid = r2 < h2

            # Poly6 kernel
            kernel_val = (315.0 / (64.0 * np.pi * (self.h ** 9))) * ((h2 - r2) ** 3) * valid
            np.add.at(density, i_idx, mass[j_idx] * kernel_val)
            np.add.at(density, j_idx, mass[i_idx] * kernel_val)

        density = np.maximum(density, 100.0)

        # 3. State equation (Pressure)
        # Raw egg white spreads more (lower stiffness), yolk holds dome shape
        stiffness = np.where(body_id == 1, self.gas_stiffness_white, self.gas_stiffness_yolk)
        # As egg cooks, solid fraction increases stiffness
        stiffness = stiffness * (1.0 + 3.0 * solid)
        pressure = stiffness * (density - self.rest_density * 0.8)
        pressure = np.maximum(pressure, 0.0)

        # 4. Forces computation
        forces = np.zeros((N, 3), dtype=np.float64)

        # Gravity
        forces += mass[:, None] * self.gravity[None, :]

        # SPH internal forces (pressure gradient + viscosity)
        inter_mat_force = np.zeros(3, dtype=np.float64)

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            rij = pos[i_idx] - pos[j_idx]
            r2 = np.sum(rij * rij, axis=1)
            r = np.sqrt(np.maximum(r2, 1e-12))
            valid = r < self.h

            # Spiky kernel gradient: -45 / (pi * h^6) * (h - r)^2 * (rij / r)
            grad_w = -45.0 / (np.pi * (self.h ** 6)) * ((self.h - r) ** 2) * (rij / r[:, None]) * valid[:, None]

            # Pressure force: - m_i * m_j * (p_i / rho_i^2 + p_j / rho_j^2) * grad_W
            p_term = (pressure[i_idx] / (density[i_idx] ** 2) + pressure[j_idx] / (density[j_idx] ** 2))
            f_press_ij = - (mass[i_idx] * mass[j_idx] * p_term)[:, None] * grad_w

            # Viscosity force
            # Laplacian of viscosity kernel: 45 / (pi * h^6) * (h - r)
            lap_w = 45.0 / (np.pi * (self.h ** 6)) * (self.h - r) * valid
            # Effective viscosity based on phase
            eff_visc_i = np.where(body_id[i_idx] == 1,
                                  self.viscosity_white_liquid * (1.0 - solid[i_idx]) + self.viscosity_white_solid * solid[i_idx],
                                  self.viscosity_yolk)
            eff_visc_j = np.where(body_id[j_idx] == 1,
                                  self.viscosity_white_liquid * (1.0 - solid[j_idx]) + self.viscosity_white_solid * solid[j_idx],
                                  self.viscosity_yolk)
            avg_visc = 0.5 * (eff_visc_i + eff_visc_j)

            v_diff = vel[j_idx] - vel[i_idx]
            f_visc_ij = (mass[i_idx] * mass[j_idx] * avg_visc / density[j_idx])[:, None] * v_diff * lap_w[:, None]

            # Interfacial surface tension / immiscibility between yolk (id 2) and white (id 1)
            diff_body = body_id[i_idx] != body_id[j_idx]
            # Soft repulsion between white and yolk to prevent concentration mixing
            f_repulsion = np.zeros_like(f_press_ij)
            if np.any(diff_body):
                rep_mag = 0.5 * (1.0 - r / self.h) * diff_body
                f_repulsion = rep_mag[:, None] * (rij / r[:, None])

            total_pair_force = f_press_ij + f_visc_ij + f_repulsion

            np.add.at(forces, i_idx, total_pair_force)
            np.add.at(forces, j_idx, -total_pair_force)

            # Track inter-material interaction force
            if np.any(diff_body):
                inter_mat_force = np.sum(total_pair_force[diff_body], axis=0)

        self.interaction_force[0] = inter_mat_force
        self.interaction_force[1] = -inter_mat_force

        # 5. Integrate velocity and position
        acc = forces / mass[:, None]
        vel += acc * dt
        # Damping for numerical stability
        vel *= (1.0 - 0.05 * dt)
        pos += vel * dt

        # 6. Thermal conduction and phase transformation
        # Distance to hot pan base (z = pan_z)
        z_dist = pos[:, 2] - self.pan_z
        near_pan = z_dist < 0.02
        # Heat transfer from hot pan
        heat_flux = np.zeros(N, dtype=np.float64)
        # Particles close to pan heat up rapidly
        contact_weight = np.exp(-np.maximum(z_dist, 0.0) / 0.005)
        heat_flux += self.pan_heat_transfer_rate * contact_weight * (self.pan_temp - temp)

        # Particle-particle thermal conduction
        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            rij = pos[i_idx] - pos[j_idx]
            r2 = np.sum(rij * rij, axis=1)
            valid = r2 < (self.h * self.h)
            r = np.sqrt(np.maximum(r2, 1e-12))
            lap_w = 45.0 / (np.pi * (self.h ** 6)) * (self.h - r) * valid

            cond_rate = 5.0e-5
            t_diff = temp[j_idx] - temp[i_idx]
            heat_ij = cond_rate * t_diff * lap_w
            np.add.at(heat_flux, i_idx, heat_ij)
            np.add.at(heat_flux, j_idx, -heat_ij)

        # Air cooling at top surface
        heat_flux += 0.5 * (self.ambient_temp - temp)

        temp += heat_flux * dt
        # Clip temperature
        temp = np.clip(temp, self.ambient_temp, self.pan_temp)
        self.temperature = temp

        # Phase diffusion / cooking front evolution (irreversible denaturing)
        # Solid phase fraction evolves as temp increases above T_cook_start
        target_solid = np.clip((temp - self.T_cook_start) / (self.T_cook_end - self.T_cook_start), 0.0, 1.0)
        # Irreversible: solid fraction can only increase
        solid = np.maximum(solid, target_solid)

        # Phase diffusion among neighbors (thermal phase front smoothing)
        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            r2 = np.sum((pos[i_idx] - pos[j_idx]) ** 2, axis=1)
            valid = r2 < (self.h * self.h)
            r = np.sqrt(np.maximum(r2, 1e-12))
            lap_w = 45.0 / (np.pi * (self.h ** 6)) * (self.h - r) * valid
            # Phase diffusion only within same material (white diffusion)
            same_white = (body_id[i_idx] == 1) & (body_id[j_idx] == 1) & valid
            if np.any(same_white):
                phase_diff = 0.02 * (solid[j_idx] - solid[i_idx]) * lap_w * same_white
                np.add.at(solid, i_idx, phase_diff * dt)
                np.add.at(solid, j_idx, -phase_diff * dt)

        solid = np.clip(solid, 0.0, 1.0)
        self.solid_phase = solid
        self.liquid_phase = 1.0 - solid

        # 7. Pan boundary collision and contact constraints
        # Flat bottom boundary at pan_z
        contact_mask = pos[:, 2] < self.pan_z
        pos[contact_mask, 2] = self.pan_z
        # Inelastic collision with pan floor + friction
        vel[contact_mask, 2] = -0.05 * vel[contact_mask, 2]
        vel[contact_mask, 0] *= 0.8
        vel[contact_mask, 1] *= 0.8

        # As solid phase fraction increases, solidify in place (adhere to pan / stop spreading)
        solid_brake = np.clip(1.0 - solid * 0.7, 0.0, 1.0)
        vel[:, 0] *= (solid_brake ** 0.1)
        vel[:, 1] *= (solid_brake ** 0.1)

        # Radial pan boundary (curved pan rim)
        r_xy = np.hypot(pos[:, 0], pos[:, 1])
        rim_mask = r_xy > self.pan_radius
        if np.any(rim_mask):
            pos[rim_mask, 0] *= self.pan_radius / r_xy[rim_mask]
            pos[rim_mask, 1] *= self.pan_radius / r_xy[rim_mask]
            vel[rim_mask, 0] *= -0.2
            vel[rim_mask, 1] *= -0.2

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the target time and return a StateFrame."""
        target_time = float(time)
        while self.time + 1e-7 < target_time:
            dt = min(self.dt_sub, target_time - self.time)
            self._step_physics(dt)
            self.time += dt

        self.time = target_time

        # Compute StateFrame fields
        pan_distance = self.position[:, 2] - self.pan_z

        # Contact signed distance: negative means penetrating the pan
        contact_signed_dist = pan_distance

        # Build diagnostic Eulerian grid
        # Coordinates for the 3D grid
        gx = np.linspace(self.grid_bounds_min[0], self.grid_bounds_max[0], self.nx)
        gy = np.linspace(self.grid_bounds_min[1], self.grid_bounds_max[1], self.ny)
        gz = np.linspace(self.grid_bounds_min[2], self.grid_bounds_max[2], self.nz)

        # Grid fields
        active_mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        transported_scalar = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)  # Temperature field
        scalar_source = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)  # Heat source at bottom
        velocity_field = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        # Map particles to Eulerian grid cells
        ix = np.clip(np.floor((self.position[:, 0] - self.grid_bounds_min[0]) / self.grid_spacing).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((self.position[:, 1] - self.grid_bounds_min[1]) / self.grid_spacing).astype(int), 0, self.ny - 1)
        iz = np.clip(np.floor((self.position[:, 2] - self.grid_bounds_min[2]) / self.grid_spacing).astype(int), 0, self.nz - 1)

        active_mask[ix, iy, iz] = True

        # Deposit temperature and velocity to grid with simple cell-accumulation
        weight_grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        np.add.at(weight_grid, (ix, iy, iz), self.mass)
        np.add.at(transported_scalar, (ix, iy, iz), self.mass * self.temperature)
        for d in range(3):
            np.add.at(velocity_field[:, :, :, d], (ix, iy, iz), self.mass * self.velocity[:, d])

        valid_cells = weight_grid > 0
        transported_scalar[valid_cells] /= weight_grid[valid_cells]
        for d in range(3):
            velocity_field[valid_cells, d] /= weight_grid[valid_cells]

        # Heat source at pan bottom boundary layer (z index 0)
        scalar_source[:, :, 0] = np.where(active_mask[:, :, 0], self.pan_heat_transfer_rate * (self.pan_temp - transported_scalar[:, :, 0]), 0.0)

        frame = StateFrame(
            time=self.time,
            position=np.copy(self.position),
            velocity=np.copy(self.velocity),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "material_concentration": np.copy(self.concentration),
                "material_label": np.copy(self.material_label),
                "liquid_phase_fraction": np.copy(self.liquid_phase),
                "solid_phase_fraction": np.copy(self.solid_phase),
                "temperature_k": np.copy(self.temperature),
                "pan_contact_distance": np.copy(pan_distance),
                "mass_quantity": np.copy(self.mass),
            },
            fields={
                "active_mask": active_mask,
                "contact_signed_distance": np.copy(contact_signed_dist),
                "cumulative_mass_source": np.copy(self.cumulative_mass_source),
                "interaction_force": np.copy(self.interaction_force),
                "scalar_source": scalar_source,
                "transported_scalar": transported_scalar,
                "velocity_field": velocity_field,
            },
            scalars={
                "total_egg_mass_kg": float(np.sum(self.mass)),
                "pan_temperature_k": float(self.pan_temp),
                "contact_length_scale": float(self.contact_length_scale),
                "grid_spacing": float(self.grid_spacing),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_particles": self.num_particles,
            "avg_temp_k": float(np.mean(self.temperature)) if hasattr(self, "temperature") else 0.0,
            "cooked_fraction": float(np.mean(self.solid_phase)) if hasattr(self, "solid_phase") else 0.0,
        }

