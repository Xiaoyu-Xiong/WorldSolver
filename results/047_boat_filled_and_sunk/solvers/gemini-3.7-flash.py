"""Solver Agent-owned implementation for 047_boat_filled_and_sunk."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Config parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.water_density = float(self.config["fluid_domain"].get("water_density", 1000.0))
        self.boat_dry_mass = float(self.config["boat"]["mass"])
        self.thickness = float(self.config["boat"].get("thin_shell_thickness", 0.018))
        self.inlet_cfg = self.config["inlet"]
        self.inlet_center = np.array(self.inlet_cfg["center"], dtype=float)
        self.inlet_vel = np.array(self.inlet_cfg["velocity"], dtype=float)
        self.inlet_interval = self.inlet_cfg["active_interval"]
        self.flow_rate = float(self.inlet_cfg["volumetric_flow_rate"])  # m^3/s

        # Get initial state from scene
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)

        # Particle arrays
        self.position = np.array(init_frame.position, dtype=float)
        self.velocity = np.array(init_frame.velocity, dtype=float)
        self.mass = np.array(init_frame.mass, dtype=float)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32)
        self.water_region = np.array(init_frame.attributes["water_region"], dtype=np.int32)
        self.phase_id = np.array(init_frame.attributes.get("phase_id", np.ones(len(self.position), dtype=np.int32)), dtype=np.int32)

        # Orientation / angular velocity for rigid body (and dummies for particles)
        self.orientation = np.zeros((len(self.position), 4), dtype=float)
        self.orientation[:, 3] = 1.0
        self.angular_velocity = np.zeros_like(self.position)

        # Boat state
        # Index 0 is the boat rigid body
        self.boat_pos = self.position[0].copy()
        self.boat_vel = self.velocity[0].copy()
        self.boat_ang_vel = np.zeros(3, dtype=float)
        self.boat_rot = np.eye(3, dtype=float)

        # Hull geometry properties
        # Hull dimensions based on assets/open_boat.obj
        # Length ~ 2.84 (-1.42 to 1.42), Width ~ 1.76 (-0.88 to 0.88), Height ~ 0.86 (0.02 to 0.88)
        self.hull_bottom_z = 0.02  # local bottom
        self.hull_gunwale_z = 0.86  # local top
        self.hull_area = 2.4 * 1.4 * 0.7  # effective waterplane area ~ 2.35 m^2
        self.hull_max_int_vol = 1.15  # max interior volume in m^3 before spilling
        self.hull_internal_water_mass = 0.0
        self.cumulative_inlet_mass = 0.0

        # Tank bounds
        self.pool_free_surface_z = float(self.config["fluid_domain"].get("initial_free_surface_z", 1.65))
        self.tank_floor_z = 0.0  # top of tank_bottom box
        self.tank_min = np.array([-2.28, -1.68, 0.0])
        self.tank_max = np.array([2.28, 1.68, 3.5])

        # Track external and interaction forces for StateFrame
        self.last_ext_forces = np.zeros_like(self.position)
        self.last_interaction_force = np.zeros((2, 3), dtype=float)  # 2 material classes
        self.last_contact_gap = np.array([self.boat_pos[2] - (self.tank_floor_z + self.hull_bottom_z)], dtype=float)

        # Simulation substepping parameters
        self.sim_dt = 0.002

    def _step(self, dt: float) -> None:
        """Advance physics by sub-step dt."""
        t_curr = self.time

        # 1. Update Inlet and Stream Particles
        inlet_active = (self.inlet_interval[0] <= t_curr <= self.inlet_interval[1])
        if inlet_active:
            mass_added = self.flow_rate * self.water_density * dt
            self.cumulative_inlet_mass += mass_added
        else:
            mass_added = 0.0

        # 2. Hull water accumulation and leakage
        # Boat current geometry in world coords
        # Boat center is boat_pos
        # The stream falls directly at (0, 0)
        boat_rel_inlet_dist = np.linalg.norm(self.boat_pos[:2] - self.inlet_center[:2])
        boat_gunwale_world_z = self.boat_pos[2] + self.hull_gunwale_z
        boat_bottom_world_z = self.boat_pos[2] + self.hull_bottom_z

        if inlet_active and boat_rel_inlet_dist < 1.0:
            # Water enters hull
            self.hull_internal_water_mass += mass_added

        # Spilling / leakage if submerged or over-filled
        # Internal water level in world z
        int_water_h = self.hull_internal_water_mass / (self.water_density * self.hull_area)
        int_water_world_z = boat_bottom_world_z + int_water_h

        # If gunwales are below pool surface or internal level exceeds gunwale, water spills
        if boat_gunwale_world_z < self.pool_free_surface_z:
            # Submerged: hull completely floods
            flood_rate = 120.0 * dt  # rapid flooding
            max_mass = self.hull_max_int_vol * self.water_density
            self.hull_internal_water_mass = min(max_mass, self.hull_internal_water_mass + flood_rate)
        elif int_water_world_z > boat_gunwale_world_z:
            # Overfilling spill
            spill = (int_water_world_z - boat_gunwale_world_z) * self.hull_area * self.water_density
            self.hull_internal_water_mass = max(0.0, self.hull_internal_water_mass - spill * 0.5)

        # Total boat mass
        total_boat_mass = self.boat_dry_mass + self.hull_internal_water_mass

        # 3. Hydrostatic Buoyancy and Drag Forces on Boat
        # Submergence depth of boat bottom below pool surface
        submergence = max(0.0, self.pool_free_surface_z - boat_bottom_world_z)
        # Displaced volume
        # Hull is flared: displaced volume as function of draft
        effective_draft = min(submergence, self.hull_gunwale_z - self.hull_bottom_z)
        if boat_gunwale_world_z < self.pool_free_surface_z:
            # Fully submerged hull displaces shell volume + trapped volume
            displaced_vol = self.hull_area * (self.hull_gunwale_z - self.hull_bottom_z) * 0.55
        else:
            # Partially submerged hull
            displaced_vol = self.hull_area * effective_draft * 0.75

        f_buoyancy_z = displaced_vol * self.water_density * (-self.gravity[2])

        # Sinking check: if flooded, buoyancy is smaller than total weight
        f_gravity_z = total_boat_mass * self.gravity[2]

        # Hydrodynamic damping / drag on boat
        v_rel_z = self.boat_vel[2]
        c_drag_z = 0.5 * 1000.0 * self.hull_area * 1.2
        f_drag_z = -c_drag_z * abs(v_rel_z) * v_rel_z - 150.0 * v_rel_z
        f_drag_xy = -80.0 * self.boat_vel[:2]

        # Net external force on boat (excluding contact)
        f_ext_boat = np.array([f_drag_xy[0], f_drag_xy[1], f_gravity_z + f_buoyancy_z + f_drag_z], dtype=float)

        # Contact with tank bottom
        penetration = (self.tank_floor_z + self.hull_bottom_z) - boat_bottom_world_z
        f_contact_z = 0.0
        contact_gap = boat_bottom_world_z - (self.tank_floor_z + self.hull_bottom_z)
        if penetration > 0:
            # Contact spring-damper penalty
            k_contact = 5.0e5
            d_contact = 2.0e3
            f_contact_z = k_contact * penetration - d_contact * min(0.0, self.boat_vel[2])
            f_contact_z = max(0.0, f_contact_z)

        # 4. Integrate Boat Rigid Body
        a_boat_z = (f_ext_boat[2] + f_contact_z) / total_boat_mass
        a_boat_xy = f_ext_boat[:2] / total_boat_mass

        self.boat_vel[2] += a_boat_z * dt
        self.boat_vel[:2] += a_boat_xy * dt
        self.boat_pos += self.boat_vel * dt

        # Enforce hard floor collision constraint
        min_boat_z = self.tank_floor_z
        if self.boat_pos[2] < min_boat_z:
            self.boat_pos[2] = min_boat_z
            if self.boat_vel[2] < 0:
                self.boat_vel[2] = 0.0

        # Update boat particle at index 0
        self.position[0] = self.boat_pos
        self.velocity[0] = self.boat_vel
        self.mass[0] = total_boat_mass

        # 5. Particle dynamics (pool and stream particles)
        n_pts = len(self.position)
        ext_forces = np.zeros_like(self.position)
        ext_forces[0] = f_ext_boat

        # Particle velocities & positions
        # Particles are categorized by water_region:
        # region 1: main pool
        # region 2: internal water in boat
        # region 3: inlet stream
        for i in range(1, n_pts):
            region = self.water_region[i]
            p_mass = self.mass[i]
            pos = self.position[i]
            vel = self.velocity[i]

            if region == 3:
                # Inlet stream particle
                if inlet_active:
                    # Accelerate under gravity
                    vel += self.gravity * dt
                    pos += vel * dt
                    ext_forces[i] = p_mass * self.gravity

                    # Check if reached hull or pool
                    if pos[2] <= max(boat_bottom_world_z + 0.1, 0.1):
                        # Reset to inlet nozzle to maintain continuous visible stream
                        offset = (np.random.rand(3) - 0.5) * 0.15
                        pos[0] = self.inlet_center[0] + offset[0]
                        pos[1] = self.inlet_center[1] + offset[1]
                        pos[2] = self.inlet_center[2] + (np.random.rand() - 0.5) * 0.3
                        vel[:] = self.inlet_vel
                        # If inside boat, convert some to internal water
                        if np.linalg.norm(pos[:2] - self.boat_pos[:2]) < 0.6 and pos[2] < boat_gunwale_world_z:
                            self.water_region[i] = 2
                else:
                    # Stream stopped, particles fall into pool
                    vel += self.gravity * dt
                    pos += vel * dt
                    ext_forces[i] = p_mass * self.gravity
                    if pos[2] < self.pool_free_surface_z:
                        self.water_region[i] = 1

            elif region == 2:
                # Internal water particle: moves with the boat
                # Distribute within current internal water height
                local_h = min(self.hull_gunwale_z - self.hull_bottom_z, max(0.05, int_water_h))
                rel_z = (i % 7) / 7.0 * local_h
                rel_x = ((i % 5) - 2) * 0.25
                rel_y = (((i // 5) % 3) - 1) * 0.2

                target_pos = self.boat_pos + np.array([rel_x, rel_y, self.hull_bottom_z + rel_z])
                vel = (target_pos - pos) / max(dt, 1e-3) * 0.2
                pos += vel * dt
                ext_forces[i] = p_mass * self.gravity

            else:
                # Pool particle (region 1)
                # gentle sloshing / hydrostatic equilibrium
                # Damped oscillation around equilibrium
                vel += self.gravity * dt * 0.02  # tiny perturbation damping
                vel *= np.exp(-1.5 * dt)
                pos += vel * dt
                ext_forces[i] = p_mass * self.gravity

            # Boundary clamps
            pos[0] = np.clip(pos[0], self.tank_min[0], self.tank_max[0])
            pos[1] = np.clip(pos[1], self.tank_min[1], self.tank_max[1])
            pos[2] = np.clip(pos[2], self.tank_min[2], self.tank_max[2])

            self.position[i] = pos
            self.velocity[i] = vel

        # Store forces and contact info
        self.last_ext_forces = ext_forces
        self.last_interaction_force[0] = np.array([0.0, 0.0, f_contact_z])  # Boat contact force
        self.last_interaction_force[1] = np.array([0.0, 0.0, f_buoyancy_z])  # Fluid-boat coupling force
        self.last_contact_gap = np.array([contact_gap], dtype=float)

        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation to requested time
        while self.time + 1e-7 < time:
            dt = min(self.sim_dt, time - self.time)
            self._step(dt)

        # Build StateFrame
        n = len(self.position)
        mass_quantity = self.mass.copy()

        # Interaction forces: [K, 3] where K=2 (boat, water)
        interaction_force = self.last_interaction_force.copy()

        # Cumulative mass source: [K]
        cumulative_mass = np.array([0.0, self.cumulative_inlet_mass], dtype=float)

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase_id": self.phase_id.copy(),
                "water_region": self.water_region.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": self.last_contact_gap.copy(),
                "cumulative_mass_source": cumulative_mass,
                "external_force": self.last_ext_forces.copy(),
                "interaction_force": interaction_force,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "boat_floating_height": float(self.boat_pos[2]),
                "hull_internal_water_mass": float(self.hull_internal_water_mass),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "boat_pos": self.boat_pos.tolist(),
            "boat_mass": float(self.mass[0]),
            "hull_internal_water_mass": float(self.hull_internal_water_mass),
            "time": float(self.time),
        }

