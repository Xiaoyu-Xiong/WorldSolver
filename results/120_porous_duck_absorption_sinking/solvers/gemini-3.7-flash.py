"""Physical solver for two-way coupled porous duck absorption and sinking."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Get initial state frame from scene
        self.init_frame = scene.initial_state()
        self.current_time = 0.0

        # Physical constants and parameters
        self.g = float(self.config.get("gravity", [0.0, 0.0, -9.81])[2])
        self.rho_fluid = float(self.config["fluid"]["density_kg_m3"])  # 1000.0
        self.dry_density = float(self.config["duck"]["dry_bulk_density_kg_m3"])  # 640.0
        self.porosity = float(self.config["duck"]["porosity"])  # 0.62
        self.switch_time = float(self.config["permeability_control"]["switch_time"])  # 2.0
        self.k_perm = float(self.config["permeability_control"]["intrinsic_permeability_after_switch_m2"])  # 3.0e-8
        self.mu_water = 1.0e-3  # dynamic viscosity Pa.s

        # Grid / particle setup
        self.spacing = 0.065
        self.cell_vol = self.spacing ** 3

        # Particle indices and properties
        self.body_id = np.array(self.init_frame.body_id, dtype=np.int32, copy=True)
        self.is_duck_surf = (self.body_id == 1)
        self.is_water = (self.body_id == 2)
        self.is_duck_pore = (self.body_id == 3)
        self.is_duck = self.is_duck_surf | self.is_duck_pore

        self.N = len(self.body_id)
        self.pos = np.array(self.init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(self.init_frame.velocity, dtype=np.float64, copy=True)
        self.dry_mass = np.array(self.init_frame.mass, dtype=np.float64, copy=True)
        self.mass = np.array(self.init_frame.mass, dtype=np.float64, copy=True)

        # Duck reference relative positions from duck centroid
        duck_idx = np.where(self.is_duck)[0]
        self.duck_idx = duck_idx
        self.duck_pore_idx = np.where(self.is_duck_pore)[0]
        self.duck_surf_idx = np.where(self.is_duck_surf)[0]
        self.water_idx = np.where(self.is_water)[0]

        # Initial duck center of mass
        duck_masses = self.dry_mass[duck_idx]
        self.duck_com = np.sum(self.pos[duck_idx] * duck_masses[:, None], axis=0) / np.sum(duck_masses)
        self.duck_rel_pos = self.pos[duck_idx] - self.duck_com
        self.duck_vel = np.zeros(3, dtype=np.float64)

        # Porous properties per particle
        self.saturation = np.zeros(self.N, dtype=np.float64)
        self.saturation[self.is_water] = 1.0
        self.pore_water_mass = np.zeros(self.N, dtype=np.float64)
        self.pore_pressure = np.zeros(self.N, dtype=np.float64)
        self.permeability_m2 = np.zeros(self.N, dtype=np.float64)
        self.porosity_arr = np.zeros(self.N, dtype=np.float64)
        self.porosity_arr[self.is_duck] = self.porosity
        self.phase = np.zeros(self.N, dtype=np.float64)
        self.phase[self.is_water] = 1.0

        # Maximum pore water mass per pore particle
        self.max_pore_water_mass = self.porosity * self.rho_fluid * self.cell_vol

        # Tank geometry limits
        # Tank interior: x in [-1.65, 1.65], y in [-0.75, 0.75], z in [0.0, 1.5]
        self.tank_x_min, self.tank_x_max = -1.65, 1.65
        self.tank_y_min, self.tank_y_max = -0.75, 0.75
        self.tank_z_min = 0.0

        # Sub-stepping setup
        self.sim_dt = 0.002  # 2 ms sub-step for numerical stability

        # Cumulative momentum budgets (G=3 domains: 0: Global, 1: Duck, 2: Water; K=3 force terms)
        self.G = 3
        self.K = 3
        self.domain_weights = np.zeros((self.G, self.N), dtype=np.float64)
        self.domain_weights[0, :] = 1.0  # Global
        self.domain_weights[1, self.duck_idx] = 1.0  # Duck
        self.domain_weights[2, self.water_idx] = 1.0  # Water

        self.cum_external_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_inflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_outflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.last_external_force = np.zeros((self.G, self.K, 3), dtype=np.float64)
        self.last_interaction_force = np.zeros((self.K, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(self.K, dtype=np.float64)

        # Reference momentum scale
        total_sys_mass = np.sum(self.mass)
        # Characteristic velocity sqrt(2 * g * h) ~ sqrt(2 * 9.81 * 0.8) ~ 4.0 m/s
        self.ref_momentum = float(total_sys_mass * np.sqrt(2.0 * abs(self.g) * 0.82))

        # Initial water resting profile
        self.water_rest_pos = np.array(self.pos[self.water_idx], copy=True)
        self.water_level = float(self.config["fluid"]["initial_surface_z"])

    def _physics_step(self, dt: float) -> None:
        permeability_active = (self.current_time >= self.switch_time)
        if permeability_active:
            self.permeability_m2[self.duck_pore_idx] = self.k_perm
        else:
            self.permeability_m2[self.duck_pore_idx] = 0.0

        # 1. Porous absorption: Darcy-driven water absorption into submerged pore particles
        absorbed_mass_this_step = 0.0

        if permeability_active:
            # For each pore particle submerged below water level
            submerged_pore_mask = (self.pos[self.duck_pore_idx, 2] < self.water_level) & (self.saturation[self.duck_pore_idx] < 0.999)
            active_pores = self.duck_pore_idx[submerged_pore_mask]

            if len(active_pores) > 0:
                depth = np.maximum(0.0, self.water_level - self.pos[active_pores, 2])
                # Hydrostatic pore pressure: P = rho * g * depth + capillary_pressure
                p_hydro = self.rho_fluid * abs(self.g) * depth
                p_capillary = 400.0  # Pa, capillary suction
                p_pore = p_hydro + p_capillary
                self.pore_pressure[active_pores] = p_pore

                # Darcy flux: q = (k / mu) * grad(P)
                # Absorption rate dm/dt = rho * Area * q
                area = self.spacing ** 2
                grad_p = (p_pore + 100.0) / self.spacing
                flux = (self.k_perm / self.mu_water) * grad_p
                dm_dt = self.rho_fluid * area * flux * 0.05  # absorption rate factor

                dm = dm_dt * dt
                capacity = np.maximum(0.0, self.max_pore_water_mass - self.pore_water_mass[active_pores])
                actual_dm = np.minimum(dm, capacity)

                self.pore_water_mass[active_pores] += actual_dm
                self.mass[active_pores] = self.dry_mass[active_pores] + self.pore_water_mass[active_pores]
                self.saturation[active_pores] = np.clip(self.pore_water_mass[active_pores] / self.max_pore_water_mass, 0.0, 1.0)

                total_absorbed = np.sum(actual_dm)
                absorbed_mass_this_step += total_absorbed

                # Update duck surface wetness from nearby pore saturation
                pore_sub = self.duck_pore_idx
                pore_sat = self.saturation[pore_sub]
                pore_z = self.pos[pore_sub, 2]
                for si in self.duck_surf_idx:
                    sz = self.pos[si, 2]
                    close = np.abs(pore_z - sz) < 0.1
                    if np.any(close):
                        self.saturation[si] = np.clip(np.mean(pore_sat[close]), 0.0, 1.0)

                # Water level drops slightly by conservation of water mass
                tank_surface_area = (self.tank_x_max - self.tank_x_min) * (self.tank_y_max - self.tank_y_min)
                delta_water_level = total_absorbed / (self.rho_fluid * tank_surface_area)
                self.water_level -= delta_water_level

        # 2. Total duck mass and buoyancy calculation
        duck_total_mass = np.sum(self.mass[self.duck_idx])
        duck_pore_masses = self.mass[self.duck_idx]
        self.duck_com = np.sum(self.pos[self.duck_idx] * duck_pore_masses[:, None], axis=0) / duck_total_mass

        # Submerged volume calculation for buoyancy
        submerged_pores = self.duck_pore_idx[self.pos[self.duck_pore_idx, 2] < self.water_level]
        submerged_vol = len(submerged_pores) * self.cell_vol

        # Buoyancy force: F_b = rho_water * V_submerged * g
        f_buoyancy_z = self.rho_fluid * submerged_vol * abs(self.g)
        # Hydrodynamic drag force
        f_drag = -6.0 * np.pi * self.mu_water * 0.3 * self.duck_vel - 0.5 * self.rho_fluid * 0.15 * 0.8 * self.duck_vel * np.abs(self.duck_vel)

        # Net external force on duck
        f_gravity_duck = np.array([0.0, 0.0, duck_total_mass * self.g], dtype=np.float64)
        f_buoy_duck = np.array([0.0, 0.0, f_buoyancy_z], dtype=np.float64)
        f_fluid_duck = f_buoy_duck + f_drag

        # Duck acceleration
        duck_acc = (f_gravity_duck + f_fluid_duck) / duck_total_mass

        # Time step duck velocity and position
        self.duck_vel += duck_acc * dt
        self.duck_vel[0] *= 0.99
        self.duck_vel[1] *= 0.99

        new_duck_com = self.duck_com + self.duck_vel * dt

        # Duck floor contact
        duck_lowest_z = np.min(new_duck_com[2] + self.duck_rel_pos[:, 2])
        contact_force_z = 0.0
        if duck_lowest_z < self.tank_z_min:
            pen = self.tank_z_min - duck_lowest_z
            new_duck_com[2] += pen
            contact_force_z = (duck_total_mass * abs(self.g) - f_buoyancy_z)
            if contact_force_z < 0:
                contact_force_z = 0.0
            if self.duck_vel[2] < 0:
                self.duck_vel[2] = 0.0

        self.duck_com = new_duck_com
        self.pos[self.duck_idx] = self.duck_com + self.duck_rel_pos
        self.vel[self.duck_idx] = self.duck_vel

        # 3. Fluid dynamics and coupling
        water_pos = self.pos[self.water_idx]
        water_vel = self.vel[self.water_idx]

        dz_eq = (self.water_rest_pos[:, 2] - water_pos[:, 2])
        f_restoring = 50.0 * dz_eq - 2.0 * water_vel[:, 2]

        duck_rel = water_pos - self.duck_com
        dist_sq = (duck_rel[:, 0] / 0.70) ** 2 + (duck_rel[:, 1] / 0.30) ** 2 + (duck_rel[:, 2] / 0.45) ** 2
        inside_duck = dist_sq < 1.0

        push_dir = np.zeros_like(water_pos)
        push_dir[:, 0] = duck_rel[:, 0] * 2.0
        push_dir[:, 1] = duck_rel[:, 1] * 2.0
        push_dir[:, 2] = np.maximum(duck_rel[:, 2], 0.1)
        push_norm = np.linalg.norm(push_dir, axis=1, keepdims=True) + 1e-6
        push_dir = push_dir / push_norm

        water_acc = np.zeros_like(water_pos)
        water_acc[:, 2] += self.g + f_restoring

        if np.any(inside_duck):
            overlap = np.maximum(0.0, 1.0 - np.sqrt(dist_sq[inside_duck]))
            water_acc[inside_duck] += push_dir[inside_duck] * (overlap[:, None] * 80.0)

        # Update water velocities and positions
        water_vel += water_acc * dt
        water_vel *= 0.98
        water_pos += water_vel * dt

        # Boundary constraints for water
        water_pos[:, 0] = np.clip(water_pos[:, 0], self.tank_x_min + 0.03, self.tank_x_max - 0.03)
        water_pos[:, 1] = np.clip(water_pos[:, 1], self.tank_y_min + 0.03, self.tank_y_max - 0.03)
        water_pos[:, 2] = np.clip(water_pos[:, 2], self.tank_z_min + 0.02, self.water_level + 0.1)

        self.pos[self.water_idx] = water_pos
        self.vel[self.water_idx] = water_vel

        # 4. Budget updates
        f_ext_duck = f_gravity_duck + np.array([0.0, 0.0, contact_force_z])
        f_ext_water = np.array([0.0, 0.0, np.sum(self.mass[self.water_idx]) * self.g])
        f_ext_global = f_ext_duck + f_ext_water

        self.cum_external_impulse[0] += f_ext_global * dt
        self.cum_external_impulse[1] += f_ext_duck * dt
        self.cum_external_impulse[2] += f_ext_water * dt

        self.last_external_force[0, 0] = f_ext_global
        self.last_external_force[1, 0] = f_ext_duck
        self.last_external_force[2, 0] = f_ext_water

        self.last_interaction_force[0] = f_fluid_duck
        self.last_interaction_force[1] = -f_fluid_duck

        if absorbed_mass_this_step > 0:
            inflow_mom = absorbed_mass_this_step * self.duck_vel
            self.cum_inflow_momentum[1] += inflow_mom
            self.cum_outflow_momentum[2] += inflow_mom

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        while self.current_time < time - 1e-7:
            step_dt = min(self.sim_dt, time - self.current_time)
            self._physics_step(step_dt)

        self.current_time = time

        solid_vel = np.zeros_like(self.vel)
        solid_vel[self.duck_idx] = self.duck_vel

        mass_quantity = np.zeros((self.N, 2), dtype=np.float64)
        mass_quantity[self.duck_idx, 0] = self.dry_mass[self.duck_idx]
        mass_quantity[self.duck_pore_idx, 1] = self.pore_water_mass[self.duck_pore_idx]
        mass_quantity[self.water_idx, 1] = self.mass[self.water_idx]

        frame = StateFrame(
            time=float(time),
            position=np.array(self.pos, dtype=np.float64, copy=True),
            velocity=np.array(self.vel, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.array(self.body_id, dtype=np.int32, copy=True),
            attributes={
                "phase": np.array(self.phase, dtype=np.float64, copy=True),
                "saturation": np.array(self.saturation, dtype=np.float64, copy=True),
                "porosity": np.array(self.porosity_arr, dtype=np.float64, copy=True),
                "permeability_m2": np.array(self.permeability_m2, dtype=np.float64, copy=True),
                "pore_water_mass": np.array(self.pore_water_mass, dtype=np.float64, copy=True),
                "pore_pressure": np.array(self.pore_pressure, dtype=np.float64, copy=True),
                "solid_velocity": solid_vel,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": np.array(self.cum_external_impulse, dtype=np.float64, copy=True),
                "budget_cumulative_inflow_momentum": np.array(self.cum_inflow_momentum, dtype=np.float64, copy=True),
                "budget_cumulative_outflow_momentum": np.array(self.cum_outflow_momentum, dtype=np.float64, copy=True),
                "budget_domain_weights": np.array(self.domain_weights, dtype=np.float64, copy=True),
                "budget_external_force": np.array(self.last_external_force, dtype=np.float64, copy=True),
                "cumulative_mass_source": np.array(self.cumulative_mass_source, dtype=np.float64, copy=True),
                "interaction_force": np.array(self.last_interaction_force, dtype=np.float64, copy=True),
            },
            scalars={
                "reference_momentum": float(self.ref_momentum),
                "gravity_z": float(self.g),
                "permeability_switch_time": float(self.switch_time),
                "initial_water_surface_z": float(self.config["fluid"]["initial_surface_z"]),
                "current_water_surface_z": float(self.water_level),
                "total_absorbed_pore_water_mass": float(np.sum(self.pore_water_mass)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "duck_com": self.duck_com.tolist(),
            "duck_vel": self.duck_vel.tolist(),
            "total_absorbed_water": float(np.sum(self.pore_water_mass)),
        }

