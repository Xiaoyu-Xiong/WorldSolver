"""Solver implementation for Rigid Frigate Responding to Water Waves."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        # Particle / body arrays
        self.position = np.array(init_frame.position, dtype=float)
        self.velocity = np.array(init_frame.velocity, dtype=float)
        self.orientation = np.array(init_frame.orientation, dtype=float)
        self.angular_velocity = np.array(init_frame.angular_velocity, dtype=float)
        self.mass = np.array(init_frame.mass, dtype=float)
        self.body_id = np.array(init_frame.body_id, dtype=np.int64)

        # Preserved attributes
        self.phase = np.array(init_frame.attributes["phase"], dtype=float)
        self.ship_visual = np.array(init_frame.attributes["ship_visual"], dtype=float)

        # Domain and fluid parameters
        fluid_cfg = self.config["fluid"]
        wave_cfg = self.config["incoming_waves"]
        ship_cfg = self.config["frigate"]

        self.density = float(fluid_cfg["density"])
        self.domain_min = np.array(fluid_cfg["domain_min"], dtype=float)
        self.domain_max = np.array(fluid_cfg["domain_max"], dtype=float)
        self.still_water = float(fluid_cfg["still_water_level"])
        self.h = float(fluid_cfg["marker_spacing"])

        self.wave_amp = float(wave_cfg["amplitude"])
        self.wave_k = 2.0 * np.pi / float(wave_cfg["wavelength"])
        self.wave_omega = 2.0 * np.pi / float(wave_cfg["period"])
        self.wave_phase = float(wave_cfg["phase_radians"])

        self.ship_mass = float(ship_cfg["mass"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g = abs(self.gravity[2])

        # Ship rigid body DOF constraints: heave (z) and pitch (around y) only
        # Estimate pitch moment of inertia Iyy for hull of mass M, length ~8.7, height ~1.5
        self.Iyy = self.ship_mass * ((8.7 ** 2 + 1.5 ** 2) / 12.0)

        # State tracking for ship (body index 0)
        self.ship_z = float(self.position[0, 2])
        self.ship_vz = float(self.velocity[0, 2])
        self.ship_pitch = 0.0  # pitch angle theta around y (radians)
        self.ship_omega_y = 0.0  # angular velocity dtheta/dt
        self.ship_x0 = float(self.position[0, 0])
        self.ship_y0 = float(self.position[0, 1])

        # Tracking cumulative quantities for required fields
        self.cumulative_impulse = np.zeros((1, 3), dtype=float)
        self.cumulative_inflow_mom = np.zeros((1, 3), dtype=float)
        self.cumulative_outflow_mom = np.zeros((1, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(1, dtype=float)

        # Boundary samples
        self.setup_boundary_samples()

    def setup_boundary_samples(self) -> None:
        # Create boundary sample points along seabed and walls
        xs = np.linspace(self.domain_min[0], self.domain_max[0], 21)
        ys = np.linspace(self.domain_min[1], self.domain_max[1], 11)
        # Seabed points
        xx, yy = np.meshgrid(xs, ys)
        pts_bed = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, self.domain_min[2])))
        normals_bed = np.tile([0.0, 0.0, 1.0], (len(pts_bed), 1))

        # Port wall
        pts_port = np.column_stack((xs, np.full_like(xs, self.domain_max[1]), np.full_like(xs, 1.0)))
        normals_port = np.tile([0.0, -1.0, 0.0], (len(pts_port), 1))

        # Starboard wall
        pts_star = np.column_stack((xs, np.full_like(xs, self.domain_min[1]), np.full_like(xs, 1.0)))
        normals_star = np.tile([0.0, 1.0, 0.0], (len(pts_star), 1))

        self.boundary_pts = np.vstack((pts_bed, pts_port, pts_star))
        self.boundary_normal = np.vstack((normals_bed, normals_port, normals_star))
        self.boundary_velocity = np.zeros_like(self.boundary_pts)
        self.boundary_target_velocity = np.zeros_like(self.boundary_pts)

    def _eval_wave_field(self, x: np.ndarray, z: np.ndarray, t: float):
        """Analytical linear deep/intermediate water wave field for boundary/background forcing."""
        theta = self.wave_k * x - self.wave_omega * t + self.wave_phase
        eta = self.still_water + self.wave_amp * np.cos(theta)

        # Depth decay
        d = self.still_water - self.domain_min[2]
        kd = np.clip(self.wave_k * d, 0.1, 20.0)
        kz = np.clip(self.wave_k * (z - self.domain_min[2]), 0.0, 20.0)

        cosh_kd = np.cosh(kd)
        sinh_kd = np.sinh(kd)
        factor_u = np.cosh(kz) / sinh_kd
        factor_w = np.sinh(kz) / sinh_kd

        u_wave = self.wave_amp * self.wave_omega * factor_u * np.cos(theta)
        w_wave = self.wave_amp * self.wave_omega * factor_w * np.sin(theta)
        return eta, u_wave, w_wave

    def _is_inside_hull(self, rel_pos: np.ndarray) -> np.ndarray:
        """Vectorized hull test in local hull coordinates."""
        qx = rel_pos[:, 0]
        qy = rel_pos[:, 1]
        qz = rel_pos[:, 2]

        half_width = 0.92 * np.clip(1.0 - np.abs(qx) / 4.35, 0.0, 1.0) ** 0.35
        bottom_z = -1.12 + 0.16 * np.abs(qx) / 4.35
        top_z = 0.42

        inside = (
            (np.abs(qx) < 4.35)
            & (np.abs(qy) < half_width)
            & (qz > bottom_z)
            & (qz < top_z)
        )
        return inside

    def _step_simulation(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Perform a single physical sub-step coupling fluid particles and rigid hull."""
        t = self.current_time

        # 1. Update fluid velocities and positions
        fluid_pos = self.position[1:]
        fluid_vel = self.velocity[1:]

        # Compute wave elevation and ideal velocity at fluid particle positions
        eta, u_target, w_target = self._eval_wave_field(fluid_pos[:, 0], fluid_pos[:, 2], t)

        # Buoyancy / pressure gradient / wave-driving forces on fluid particles
        tau_wave = 0.35  # relaxation time to wave kinematics
        accel = np.zeros_like(fluid_vel)
        accel[:, 0] = (u_target - fluid_vel[:, 0]) / tau_wave
        accel[:, 2] = (w_target - fluid_vel[:, 2]) / tau_wave + self.gravity[2] * (fluid_pos[:, 2] - eta) / max(self.still_water, 0.1)

        # Damp lateral (y) motion
        accel[:, 1] = -0.5 * fluid_vel[:, 1]

        # Integrate fluid particles
        fluid_vel += accel * dt
        fluid_pos += fluid_vel * dt

        # Boundary enforcement on fluid particles
        # Seabed
        below_bed = fluid_pos[:, 2] < self.domain_min[2] + 0.5 * self.h
        fluid_pos[below_bed, 2] = self.domain_min[2] + 0.5 * self.h
        fluid_vel[below_bed, 2] = np.maximum(fluid_vel[below_bed, 2], 0.0)

        # Side walls
        y_max = self.domain_max[1] - 0.5 * self.h
        y_min = self.domain_min[1] + 0.5 * self.h
        out_port = fluid_pos[:, 1] > y_max
        fluid_pos[out_port, 1] = y_max
        fluid_vel[out_port, 1] = np.minimum(fluid_vel[out_port, 1], 0.0)
        out_star = fluid_pos[:, 1] < y_min
        fluid_pos[out_star, 1] = y_min
        fluid_vel[out_star, 1] = np.maximum(fluid_vel[out_star, 1], 0.0)

        # 2. Hull - Fluid Two-Way Interaction
        cos_p = np.cos(self.ship_pitch)
        sin_p = np.sin(self.ship_pitch)
        hull_center = np.array([self.ship_x0, self.ship_y0, self.ship_z])

        # Transform fluid particles into hull local frame
        delta_pos = fluid_pos - hull_center
        local_x = delta_pos[:, 0] * cos_p - delta_pos[:, 2] * sin_p
        local_y = delta_pos[:, 1]
        local_z = delta_pos[:, 0] * sin_p + delta_pos[:, 2] * cos_p
        local_pos = np.column_stack((local_x, local_y, local_z))

        # Check collision / penetration with hull
        inside_hull = self._is_inside_hull(local_pos)

        # Interaction forces
        f_fluid_on_hull = np.zeros(3, dtype=float)
        tau_fluid_on_hull = 0.0

        if np.any(inside_hull):
            inside_idx = np.where(inside_hull)[0]
            qx = local_pos[inside_idx, 0]
            qy = local_pos[inside_idx, 1]
            qz = local_pos[inside_idx, 2]

            half_w = 0.92 * np.clip(1.0 - np.abs(qx) / 4.35, 0.01, 1.0) ** 0.35
            bot_z = -1.12 + 0.16 * np.abs(qx) / 4.35

            dist_to_side = half_w - np.abs(qy)
            dist_to_bot = qz - bot_z

            push_side = dist_to_side < dist_to_bot

            repulse_local = np.zeros((len(inside_idx), 3))
            repulse_local[push_side, 1] = np.where(qy[push_side] >= 0, 1.0, -1.0) * (dist_to_side[push_side] + 0.05)
            repulse_local[~push_side, 2] = -(dist_to_bot[~push_side] + 0.05)

            repulse_world_x = repulse_local[:, 0] * cos_p + repulse_local[:, 2] * sin_p
            repulse_world_y = repulse_local[:, 1]
            repulse_world_z = -repulse_local[:, 0] * sin_p + repulse_local[:, 2] * cos_p

            fluid_pos[inside_idx, 0] += repulse_world_x
            fluid_pos[inside_idx, 1] += repulse_world_y
            fluid_pos[inside_idx, 2] += repulse_world_z

            m_p = self.mass[1]
            rx = delta_pos[inside_idx, 0]
            v_contact_z = self.ship_vz - self.ship_omega_y * rx

            fluid_vel[inside_idx, 1] += 2.0 * repulse_local[:, 1] / dt * 0.1
            fluid_vel[inside_idx, 2] = np.maximum(fluid_vel[inside_idx, 2], v_contact_z)

            f_exchange = np.sum(repulse_local, axis=0) * (m_p / (dt ** 2)) * 0.2
            f_fluid_on_hull[2] += f_exchange[2]
            tau_fluid_on_hull += np.sum(rx * (repulse_world_z * m_p / (dt ** 2))) * 0.2

        # 3. Dynamic Hydrostatic and Hydrodynamic pressure integration on hull
        n_hull_sections = 40
        sec_x = np.linspace(-4.2, 4.2, n_hull_sections)
        dx_sec = sec_x[1] - sec_x[0]

        total_buoyancy_force = 0.0
        total_pitch_moment = 0.0

        for sx in sec_x:
            wx = self.ship_x0 + sx * cos_p
            wz_keel = self.ship_z - sx * sin_p + (-1.12 + 0.16 * abs(sx) / 4.35)
            wz_deck = self.ship_z - sx * sin_p + 0.42

            wave_eta_sec, _, _ = self._eval_wave_field(np.array([wx]), np.array([wz_keel]), t)
            wave_h = float(wave_eta_sec[0])

            submerged_top = min(wz_deck, wave_h)
            submerged_depth = max(0.0, submerged_top - wz_keel)

            if submerged_depth > 0:
                hw = 0.92 * max(0.0, 1.0 - abs(sx) / 4.35) ** 0.35
                sec_submerged_area = 2.0 * hw * submerged_depth * 0.85
                sec_volume = sec_submerged_area * dx_sec

                dF_buoy = self.density * self.g * sec_volume
                total_buoyancy_force += dF_buoy

                arm_x = sx * cos_p
                total_pitch_moment -= dF_buoy * arm_x

        # Hydrodynamic wave-action and damping forces
        f_heave_damping = -1800.0 * self.ship_vz
        tau_pitch_damping = -8000.0 * self.ship_omega_y

        GM = 0.65  # meters
        righting_moment = -self.ship_mass * self.g * GM * sin_p

        # Net forces and torques on hull
        f_gravity = -self.ship_mass * self.g
        net_fz = f_gravity + total_buoyancy_force + f_heave_damping + f_fluid_on_hull[2]
        net_torque_y = total_pitch_moment + righting_moment + tau_pitch_damping + tau_fluid_on_hull

        # Integrate ship heave and pitch
        accel_z = net_fz / self.ship_mass
        accel_pitch = net_torque_y / self.Iyy

        self.ship_vz += accel_z * dt
        self.ship_z += self.ship_vz * dt

        self.ship_omega_y += accel_pitch * dt
        self.ship_pitch += self.ship_omega_y * dt

        # Update ship position, velocity, orientation, angular velocity
        self.position[0] = [self.ship_x0, self.ship_y0, self.ship_z]
        self.velocity[0] = [0.0, 0.0, self.ship_vz]

        half_pitch = 0.5 * self.ship_pitch
        self.orientation[0] = [0.0, np.sin(half_pitch), 0.0, np.cos(half_pitch)]
        self.angular_velocity[0] = [0.0, self.ship_omega_y, 0.0]

        # Write back fluid positions and velocities
        self.position[1:] = fluid_pos
        self.velocity[1:] = fluid_vel

        # Track interaction and budget forces
        ext_force = np.array([0.0, 0.0, f_gravity])
        inter_force = np.array([0.0, 0.0, total_buoyancy_force + f_fluid_on_hull[2]])

        return ext_force, inter_force

    def advance_to(self, time: float) -> StateFrame:
        # Time-step solver from current_time to requested sample time
        dt_sub = 0.005  # 5 ms physical sub-steps
        n_steps = max(1, int(np.round((time - self.current_time) / dt_sub)))
        actual_dt = (time - self.current_time) / n_steps if n_steps > 0 and time > self.current_time else 0.0

        last_ext_force = np.array([0.0, 0.0, -self.ship_mass * self.g])
        last_inter_force = np.array([0.0, 0.0, self.ship_mass * self.g])

        if actual_dt > 0.0:
            for _ in range(n_steps):
                ext_f, int_f = self._step_simulation(actual_dt)
                self.current_time += actual_dt
                self.cumulative_impulse[0] += ext_f * actual_dt
                last_ext_force = ext_f
                last_inter_force = int_f

        self.current_time = float(time)

        # Construct required fields according to exact shapes and contracts
        n_total = len(self.position)

        budget_domain_weights = np.ones((1, n_total), dtype=float)
        budget_external_force = last_ext_force.reshape(1, 1, 3)
        interaction_force = last_inter_force.reshape(1, 3)

        b_normal = self.boundary_normal.copy()
        b_target_vel = self.boundary_target_velocity.copy()
        b_vel = self.boundary_velocity.copy()

        mass_quantity = self.mass.copy()
        ref_vel = 1.0
        ref_mom = float(self.ship_mass * ref_vel)

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase": self.phase.copy(),
                "ship_visual": self.ship_visual.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": b_normal,
                "boundary_target_velocity": b_target_vel,
                "boundary_velocity": b_vel,
                "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow_mom.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "interaction_force": interaction_force,
            },
            scalars={
                "still_water_level": self.still_water,
                "reference_momentum": ref_mom,
                "reference_velocity": ref_vel,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "ship_z": self.ship_z,
            "ship_pitch": self.ship_pitch,
            "ship_vz": self.ship_vz,
            "ship_omega_y": self.ship_omega_y,
            "cumulative_impulse": self.cumulative_impulse.tolist(),
        }

