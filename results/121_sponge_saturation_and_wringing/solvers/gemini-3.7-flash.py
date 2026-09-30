"""Benchmark physics simulation solver for sponge wetting and wringing."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Time management
        self.sim_time = 0.0
        self.sub_dt = 0.005  # sub-stepping dt for physics stability

        # Initial frame from scene
        frame0 = scene.initial_state()
        self.n_total_init = len(frame0.position)

        # Parse body masks
        body_ids = np.asarray(frame0.body_id)
        sponge_mask = (body_ids == 1)
        water_mask = (body_ids == 2)
        grip_mask = (body_ids >= 10)

        # Sponge vertices (36 vertices: 9 cross-sections x 4 vertices)
        self.sponge_rest_pos = np.array(frame0.position[sponge_mask], dtype=float)
        self.sponge_pos = np.array(self.sponge_rest_pos, copy=True)
        self.sponge_vel = np.zeros_like(self.sponge_pos)
        self.sponge_mass_dry = np.array(frame0.mass[sponge_mask], dtype=float)
        self.sponge_mass = np.array(self.sponge_mass_dry, copy=True)
        self.n_sponge = len(self.sponge_pos)

        # Sponge properties
        self.sponge_saturation = np.zeros(self.n_sponge, dtype=float)
        self.sponge_porosity = np.full(self.n_sponge, 0.72, dtype=float)
        self.sponge_water_capacity = 0.004  # max water mass per vertex (~0.144 kg total water capacity)
        self.sponge_water_mass = np.zeros(self.n_sponge, dtype=float)

        # Spring-damper network for sponge elasticity
        self.rest_lengths = []
        self.spring_indices = []
        for i in range(self.n_sponge):
            for j in range(i + 1, self.n_sponge):
                d = np.linalg.norm(self.sponge_rest_pos[i] - self.sponge_rest_pos[j])
                if d < 0.10:  # neighbor distance threshold
                    self.spring_indices.append((i, j))
                    self.rest_lengths.append(d)
        self.spring_indices = np.array(self.spring_indices, dtype=int)
        self.rest_lengths = np.array(self.rest_lengths, dtype=float)
        self.k_spring = 250.0  # spring stiffness
        self.d_spring = 8.0    # damping

        # Water particles
        # Initial water particles from scene
        init_water_pos = np.array(frame0.position[water_mask], dtype=float)
        init_water_vel = np.array(frame0.velocity[water_mask], dtype=float)
        init_water_mass = np.array(frame0.mass[water_mask], dtype=float)

        self.water_pos = list(init_water_pos)
        self.water_vel = list(init_water_vel)
        self.water_mass = list(init_water_mass)
        self.water_active = [True] * len(init_water_pos)

        # Emitter parameters
        self.emitter_center = np.array([-0.07, 0.04, 0.58], dtype=float)
        self.emitter_dir = np.array([0.0, 0.0, -1.0], dtype=float)
        self.emitter_speed = 1.0
        self.emitter_radius = 0.016
        self.emitter_start = 0.0
        self.emitter_stop = 6.8
        self.emit_interval = 0.035  # emit batch interval
        self.last_emit_time = 0.0
        self.particle_mass_nominal = 0.00095

        # Grips kinematic state
        self.grip_left_init = np.array([-0.205, 0.0, 0.065], dtype=float)
        self.grip_right_init = np.array([0.205, 0.0, 0.065], dtype=float)
        self.grip_pos = np.array([self.grip_left_init, self.grip_right_init], dtype=float)
        self.grip_vel = np.zeros((2, 3), dtype=float)

        # Accounting for budgets
        self.cumulative_mass_added = 0.0
        self.cumulative_mass_removed = 0.0
        self.cumulative_ext_impulse = np.zeros((3, 3), dtype=float)  # G=3 domains: sponge (0), water (1), combined (2)
        self.cumulative_inflow_mom = np.zeros((3, 3), dtype=float)
        self.cumulative_outflow_mom = np.zeros((3, 3), dtype=float)

        # Constant reference momentum
        self.reference_momentum = 1.0  # positive scalar scale

        # Bounds & floor
        self.floor_z = 0.015  # sink floor upper surface

        self.initialized = True

    def _get_grip_pose(self, t: float):
        """Compute kinematic pose of left and right grips at time t."""
        p_left = np.array(self.grip_left_init, copy=True)
        p_right = np.array(self.grip_right_init, copy=True)
        v_left = np.zeros(3)
        v_right = np.zeros(3)

        lift_start, lift_end = 6.5, 8.0
        wring_start, wring_end = 8.0, 11.0

        # Lift displacement: [0, 0, 0.22]
        lift_disp = np.array([0.0, 0.0, 0.22])

        if t < lift_start:
            pass
        elif t <= lift_end:
            tau = (t - lift_start) / (lift_end - lift_start)
            # Smooth step (quintic or cubic)
            s = 3 * tau**2 - 2 * tau**3
            ds_dt = (6 * tau - 6 * tau**2) / (lift_end - lift_start)
            p_left += lift_disp * s
            p_right += lift_disp * s
            v_left = lift_disp * ds_dt
            v_right = lift_disp * ds_dt
        else:
            p_left += lift_disp
            p_right += lift_disp

        # Twist angle in radians around x-axis
        # Left end twists +150 deg (+5pi/6), Right end twists -150 deg (-5pi/6)
        angle_left = 0.0
        angle_right = 0.0
        omega_left = 0.0
        omega_right = 0.0

        if t > wring_start:
            if t <= wring_end:
                tau = (t - wring_start) / (wring_end - wring_start)
                s = 3 * tau**2 - 2 * tau**3
                ds_dt = (6 * tau - 6 * tau**2) / (wring_end - wring_start)
                max_deg = np.radians(150.0)
                angle_left = max_deg * s
                angle_right = -max_deg * s
                omega_left = max_deg * ds_dt
                omega_right = -max_deg * ds_dt
            else:
                max_deg = np.radians(150.0)
                angle_left = max_deg
                angle_right = -max_deg

        return (p_left, p_right), (v_left, v_right), (angle_left, angle_right), (omega_left, omega_right)

    def _step_physics(self, dt: float):
        """Advance one physical time step."""
        t = self.sim_time
        gravity = np.array([0.0, 0.0, -9.81])

        # 1. Update grips
        (p_left, p_right), (v_left, v_right), (angle_left, angle_right), (omega_left, omega_right) = self._get_grip_pose(t)
        self.grip_pos = np.array([p_left, p_right])
        self.grip_vel = np.array([v_left, v_right])

        # 2. Emit new water particles from faucet
        if self.emitter_start <= t < self.emitter_stop:
            if t - self.last_emit_time >= self.emit_interval:
                cross = np.array([[0.0, 0.0], [0.007, 0.0], [-0.007, 0.0], [0.0, 0.007], [0.0, -0.007]])
                for dx, dy in cross:
                    pos = self.emitter_center + np.array([dx, dy, -0.02])
                    vel = self.emitter_dir * self.emitter_speed
                    m = self.particle_mass_nominal
                    self.water_pos.append(pos)
                    self.water_vel.append(vel)
                    self.water_mass.append(m)
                    self.water_active.append(True)
                    # Momentum inflow
                    p_in = m * vel
                    self.cumulative_mass_added += m
                    self.cumulative_inflow_mom[1] += p_in
                    self.cumulative_inflow_mom[2] += p_in
                self.last_emit_time = t

        # 3. Sponge boundary/grip constraints and internal elasticity
        # Identify left-grip and right-grip region vertices
        # region_x left: [-0.18, -0.125], right: [0.125, 0.18]
        sponge_forces = np.zeros_like(self.sponge_pos)

        # Internal spring forces
        p1 = self.sponge_pos[self.spring_indices[:, 0]]
        p2 = self.sponge_pos[self.spring_indices[:, 1]]
        diff = p2 - p1
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-6)
        dir_vec = diff / dist_safe
        spring_f = self.k_spring * (dist - self.rest_lengths[:, None]) * dir_vec

        # Relative velocity damping
        v1 = self.sponge_vel[self.spring_indices[:, 0]]
        v2 = self.sponge_vel[self.spring_indices[:, 1]]
        damp_f = self.d_spring * np.sum((v2 - v1) * dir_vec, axis=1, keepdims=True) * dir_vec
        total_spring_damp = spring_f + damp_f

        for k, (i, j) in enumerate(self.spring_indices):
            sponge_forces[i] += total_spring_damp[k]
            sponge_forces[j] -= total_spring_damp[k]

        # Gravity and water mass contribution on sponge
        self.sponge_mass = self.sponge_mass_dry + self.sponge_water_mass
        for i in range(self.n_sponge):
            sponge_forces[i] += self.sponge_mass[i] * gravity

        # Damp sponge overall motion
        sponge_forces -= 1.5 * self.sponge_vel * self.sponge_mass[:, None]

        # Target kinematic control for ends during lift/wring
        lift_disp = self.grip_pos[0] - self.grip_left_init  # current lift displacement
        center_y = 0.0
        center_z = 0.065 + lift_disp[2]

        is_pinned = np.zeros(self.n_sponge, dtype=bool)
        for i in range(self.n_sponge):
            rx = self.sponge_rest_pos[i, 0]
            if t >= 6.5:
                if rx <= -0.125:  # Left grip control
                    is_pinned[i] = True
                    # Rotate around center_y, center_z by angle_left
                    dy = self.sponge_rest_pos[i, 1] - 0.0
                    dz = self.sponge_rest_pos[i, 2] - 0.065
                    rot_y = dy * np.cos(angle_left) - dz * np.sin(angle_left)
                    rot_z = dy * np.sin(angle_left) + dz * np.cos(angle_left)
                    target_pos = np.array([rx, center_y + rot_y, center_z + rot_z])
                    target_vel = np.array([0.0, -rot_z * omega_left, rot_y * omega_left]) + np.array([0.0, 0.0, v_left[2]])
                    self.sponge_pos[i] = target_pos
                    self.sponge_vel[i] = target_vel
                elif rx >= 0.125:  # Right grip control
                    is_pinned[i] = True
                    dy = self.sponge_rest_pos[i, 1] - 0.0
                    dz = self.sponge_rest_pos[i, 2] - 0.065
                    rot_y = dy * np.cos(angle_right) - dz * np.sin(angle_right)
                    rot_z = dy * np.sin(angle_right) + dz * np.cos(angle_right)
                    target_pos = np.array([rx, center_y + rot_y, center_z + rot_z])
                    target_vel = np.array([0.0, -rot_z * omega_right, rot_y * omega_right]) + np.array([0.0, 0.0, v_right[2]])
                    self.sponge_pos[i] = target_pos
                    self.sponge_vel[i] = target_vel

        # Integrate unpinned sponge vertices
        acc = sponge_forces / np.maximum(self.sponge_mass[:, None], 1e-6)
        for i in range(self.n_sponge):
            if not is_pinned[i]:
                self.sponge_vel[i] += acc[i] * dt
                self.sponge_pos[i] += self.sponge_vel[i] * dt

                # Floor collision constraint
                if self.sponge_pos[i, 2] < self.floor_z:
                    self.sponge_pos[i, 2] = self.floor_z
                    if self.sponge_vel[i, 2] < 0:
                        # Floor normal impulse
                        impulse = -self.sponge_mass[i] * self.sponge_vel[i, 2]
                        self.cumulative_ext_impulse[0, 2] += impulse
                        self.cumulative_ext_impulse[2, 2] += impulse
                        self.sponge_vel[i, 2] = 0.0
                    self.sponge_vel[i, 0] *= 0.8  # friction
                    self.sponge_vel[i, 1] *= 0.8

        # 4. Integrate free water particles & fluid-solid two-way interaction
        sponge_min = np.min(self.sponge_pos, axis=0) - 0.02
        sponge_max = np.max(self.sponge_pos, axis=0) + 0.02

        w_pos = np.array(self.water_pos)
        w_vel = np.array(self.water_vel)
        n_w = len(w_pos)

        # Absorption into sponge during wetting/dwell
        for j in range(n_w):
            if not self.water_active[j]:
                continue

            # Simple ballistic integration
            w_vel[j] += gravity * dt
            w_pos[j] += w_vel[j] * dt

            # Sink floor collision
            if w_pos[j, 2] <= self.floor_z:
                w_pos[j, 2] = self.floor_z
                if w_vel[j, 2] < 0:
                    impulse = -self.water_mass[j] * w_vel[j, 2]
                    self.cumulative_ext_impulse[1, 2] += impulse
                    self.cumulative_ext_impulse[2, 2] += impulse
                    w_vel[j, 2] = -0.15 * w_vel[j, 2]
                w_vel[j, 0] *= 0.85
                w_vel[j, 1] *= 0.85

            # Sink side walls collision
            for axis, bound in [(0, 0.38), (1, 0.27)]:
                if abs(w_pos[j, axis]) > bound:
                    w_pos[j, axis] = np.sign(w_pos[j, axis]) * bound
                    w_vel[j, axis] = -0.2 * w_vel[j, axis]

            # Water-sponge interaction
            if (t < 7.5 and
                sponge_min[0] <= w_pos[j, 0] <= sponge_max[0] and
                sponge_min[1] <= w_pos[j, 1] <= sponge_max[1] and
                sponge_min[2] <= w_pos[j, 2] <= sponge_max[2]):

                # Find nearest sponge vertex
                dists = np.linalg.norm(self.sponge_pos - w_pos[j], axis=1)
                idx = np.argmin(dists)
                if dists[idx] < 0.045:
                    if self.sponge_water_mass[idx] < self.sponge_water_capacity:
                        # Absorb water particle into sponge
                        absorbed = self.water_mass[j]
                        self.sponge_water_mass[idx] += absorbed
                        self.water_active[j] = False
                        # Transfer momentum to sponge
                        self.sponge_vel[idx] += (w_vel[j] * absorbed) / self.sponge_mass[idx]

        # 5. Pore-liquid diffusion in sponge
        # Liquid spreads across neighboring sponge vertices
        diff_rate = 0.8 * dt
        water_mass_changes = np.zeros(self.n_sponge)
        for (i, j) in self.spring_indices:
            diff_w = (self.sponge_water_mass[i] - self.sponge_water_mass[j]) * diff_rate
            water_mass_changes[i] -= diff_w
            water_mass_changes[j] += diff_w
        self.sponge_water_mass += water_mass_changes
        self.sponge_water_mass = np.clip(self.sponge_water_mass, 0.0, self.sponge_water_capacity)

        # 6. Wringing expulsion: twisting narrows the sponge cross-section and squeezes out stored water
        if t >= 8.0:
            # Measure local twist / narrowing or progress in wring interval
            wring_progress = np.clip((t - 8.0) / 3.0, 0.0, 1.0)
            if wring_progress > 0.05:
                # Capacity decreases due to compression
                reduced_capacity = self.sponge_water_capacity * (1.0 - 0.85 * np.sin(np.pi * wring_progress))
                for i in range(self.n_sponge):
                    excess = self.sponge_water_mass[i] - reduced_capacity
                    if excess > 0.0004:
                        # Expel excess water as droplets falling into sink
                        expel_m = excess * 0.4
                        self.sponge_water_mass[i] -= expel_m
                        # Create expelled droplets
                        droplet_pos = self.sponge_pos[i] + np.random.uniform(-0.005, 0.005, 3)
                        droplet_vel = self.sponge_vel[i] + np.array([
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                            -0.2
                        ])
                        self.water_pos.append(droplet_pos)
                        self.water_vel.append(droplet_vel)
                        self.water_mass.append(expel_m)
                        self.water_active.append(True)

        # Update saturation field: [0, 1]
        self.sponge_saturation = np.clip(self.sponge_water_mass / self.sponge_water_capacity, 0.0, 1.0)

        # Update list arrays back
        self.water_pos = [w_pos[k] for k in range(len(w_pos))]
        self.water_vel = [w_vel[k] for k in range(len(w_vel))]

        self.sim_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Step physics up to target_time
        while self.sim_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.sim_time)
            self._step_physics(dt)

        self.sim_time = target_time

        # Build StateFrame
        # Active water particles
        active_indices = [k for k, act in enumerate(self.water_active) if act]
        if len(active_indices) == 0:
            active_water_pos = np.empty((0, 3), dtype=float)
            active_water_vel = np.empty((0, 3), dtype=float)
            active_water_mass = np.empty((0,), dtype=float)
        else:
            active_water_pos = np.array([self.water_pos[k] for k in active_indices], dtype=float)
            active_water_vel = np.array([self.water_vel[k] for k in active_indices], dtype=float)
            active_water_mass = np.array([self.water_mass[k] for k in active_indices], dtype=float)

        n_active_water = len(active_water_pos)

        # Assemble positions, velocities, masses, body_ids
        pos = np.vstack((self.sponge_pos, active_water_pos, self.grip_pos))
        vel = np.vstack((self.sponge_vel, active_water_vel, self.grip_vel))
        mass = np.concatenate((
            self.sponge_mass,
            active_water_mass,
            np.array([0.02, 0.02], dtype=float)
        ))
        body_id = np.concatenate((
            np.full(self.n_sponge, 1, dtype=np.int32),
            np.full(n_active_water, 2, dtype=np.int32),
            np.array([10, 11], dtype=np.int32)
        ))

        n_total = len(pos)

        # Attributes
        sponge_sat = self.sponge_saturation
        water_sat = np.ones(n_active_water, dtype=float)
        grip_sat = np.zeros(2, dtype=float)
        saturation = np.concatenate((sponge_sat, water_sat, grip_sat))

        porosity = np.concatenate((
            self.sponge_porosity,
            np.zeros(n_active_water + 2, dtype=float)
        ))
        phase = np.concatenate((
            np.zeros(self.n_sponge, dtype=float),
            np.ones(n_active_water, dtype=float),
            np.zeros(2, dtype=float)
        ))
        mass_quantity = np.array(mass, copy=True)

        # Fields
        # G=3 domains: 0: sponge, 1: water, 2: global/combined
        # K=3 materials / force terms: gravity(0), contact/floor(1), coupling/control(2)
        budget_domain_weights = np.zeros((3, n_total), dtype=float)
        budget_domain_weights[0, :self.n_sponge] = 1.0
        budget_domain_weights[1, self.n_sponge:self.n_sponge + n_active_water] = 1.0
        budget_domain_weights[2, :] = 1.0

        gravity_f_sponge = np.sum(self.sponge_mass) * np.array([0.0, 0.0, -9.81])
        gravity_f_water = np.sum(active_water_mass) * np.array([0.0, 0.0, -9.81])
        gravity_f_global = np.sum(mass) * np.array([0.0, 0.0, -9.81])

        budget_external_force = np.zeros((3, 3, 3), dtype=float)
        budget_external_force[0, 0] = gravity_f_sponge
        budget_external_force[1, 0] = gravity_f_water
        budget_external_force[2, 0] = gravity_f_global

        cumulative_mass_source = np.array([
            0.0,                                    # sponge (no source)
            self.cumulative_mass_added - self.cumulative_mass_removed, # water source
            0.0                                     # grips
        ], dtype=float)

        interaction_force = np.zeros((3, 3), dtype=float)
        # Fluid-solid interaction and contact forces
        interaction_force[0] = np.array([0.0, 0.0, 0.0])

        frame = StateFrame(
            time=float(target_time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "saturation": saturation,
                "porosity": porosity,
                "phase": phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow_mom.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": -9.81,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "sponge_vertices": self.n_sponge,
            "active_water_particles": sum(self.water_active),
            "sim_time": self.sim_time,
        }

