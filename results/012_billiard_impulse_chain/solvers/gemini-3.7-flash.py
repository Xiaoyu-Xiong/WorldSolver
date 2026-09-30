"""Physically-based solver for billiard impulse chain simulation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load initial state from scene
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.positions = np.array(init_frame.position, dtype=np.float64, copy=True)  # (6, 3)
        self.velocities = np.array(init_frame.velocity, dtype=np.float64, copy=True)  # (6, 3)
        self.masses = np.array(init_frame.mass, dtype=np.float64, copy=True)  # (6,)
        self.body_ids = np.array(init_frame.body_id, dtype=np.int64, copy=True)  # (6,)

        if init_frame.angular_velocity is not None:
            self.angular_velocities = np.array(init_frame.angular_velocity, dtype=np.float64, copy=True)
        else:
            self.angular_velocities = np.zeros((len(self.masses), 3), dtype=np.float64)

        self.num_balls = len(self.positions)
        self.radius = 0.2
        self.contact_length_scale = self.radius * 2.0  # 0.4 m diameter

        # Physics parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.g_acc = abs(self.gravity[2])
        self.rolling_friction = float(self.config.get("rolling_friction", 0.015))
        self.restitution_ball = 0.98  # Billiard ball coefficient of restitution (elastic collision)

        # Moment of inertia for solid sphere: I = 2/5 * m * R^2
        # r_contact_arm = R
        self.I_sphere = 0.4 * self.masses * (self.radius ** 2)  # (6,)

        # Friction with ground felt
        self.mu_sliding = 0.2  # Sliding friction coefficient
        self.mu_rolling = self.rolling_friction  # Rolling resistance coefficient

        # Sub-stepping time step for high-precision impulse & collision propagation
        self.sub_dt = 1e-4  # 0.1 ms sub-step for accurate chain propagation

        # Cache for interaction forces and contact distances
        self._last_contact_distances = np.zeros(self.num_balls * (self.num_balls - 1) // 2 + self.num_balls, dtype=np.float64)
        self._last_interaction_forces = np.zeros((self.num_balls * (self.num_balls - 1) // 2, 3), dtype=np.float64)
        self._last_external_forces = np.zeros((self.num_balls, 3), dtype=np.float64)

    def _compute_contacts_and_forces(self):
        """Computes current contact signed distances, interaction forces, and external forces."""
        # Pairwise ball-ball contacts
        pair_distances = []
        interaction_forces = []

        for i in range(self.num_balls):
            for j in range(i + 1, self.num_balls):
                delta = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(delta)
                # Signed gap: dist - 2*R. Negative when penetrating.
                signed_dist = dist - 2.0 * self.radius
                pair_distances.append(signed_dist)

                # If touching / compressed, there is normal contact force
                if signed_dist < 0:
                    normal = delta / (dist + 1e-12)
                    # Hertzian / penalty repulsive force estimate or impulse-derived force
                    # Here we record the contact force
                    k_penalty = 1e6
                    f_mag = -k_penalty * signed_dist
                    interaction_forces.append(-f_mag * normal)
                else:
                    interaction_forces.append(np.zeros(3, dtype=np.float64))

        # Ground contacts
        ground_distances = []
        for i in range(self.num_balls):
            # Ball center z - radius - ground_z (ground_z = 0)
            ground_dist = self.positions[i, 2] - self.radius
            ground_distances.append(ground_dist)

        all_contact_distances = np.array(pair_distances + ground_distances, dtype=np.float64)
        interaction_forces_arr = np.array(interaction_forces, dtype=np.float64)

        # External forces: gravity + ground normal support + rolling/sliding friction
        external_forces = np.zeros((self.num_balls, 3), dtype=np.float64)
        for i in range(self.num_balls):
            # Normal force balances gravity exactly when resting on table
            N = self.masses[i] * self.g_acc
            external_forces[i, 2] = 0.0  # gravity (m*g_z) + normal support (N) cancels out in z

            # Friction force in xy plane
            v_xy = self.velocities[i, :2]
            v_norm = np.linalg.norm(v_xy)
            if v_norm > 1e-5:
                # Rolling resistance force opposite to velocity
                f_roll = self.mu_rolling * N * (v_xy / v_norm)
                external_forces[i, :2] -= f_roll

        return all_contact_distances, interaction_forces_arr, external_forces

    def _step(self, dt: float):
        """Perform one physical simulation sub-step."""
        # 1. Handle Ball-Ball Collisions (Sequential / Impulse resolution)
        # Check all pairs
        for i in range(self.num_balls):
            for j in range(i + 1, self.num_balls):
                delta = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(delta)
                min_dist = 2.0 * self.radius

                if dist < min_dist:
                    normal = delta / (dist + 1e-12)
                    # Relative velocity along normal
                    rel_vel = self.velocities[j] - self.velocities[i]
                    v_rel_n = np.dot(rel_vel, normal)

                    if v_rel_n < 0:  # Moving towards each other
                        # 1D elastic / semi-elastic collision impulse
                        m1 = self.masses[i]
                        m2 = self.masses[j]
                        e = self.restitution_ball

                        j_impulse = -(1.0 + e) * v_rel_n / (1.0 / m1 + 1.0 / m2)
                        impulse_vec = j_impulse * normal

                        self.velocities[i] -= impulse_vec / m1
                        self.velocities[j] += impulse_vec / m2

                    # Position correction to prevent penetration overlap
                    overlap = min_dist - dist
                    if overlap > 0:
                        correction = 0.5 * overlap * normal
                        self.positions[i] -= correction
                        self.positions[j] += correction

        # 2. Update Ground Friction & Rolling Dynamics
        for i in range(self.num_balls):
            # Keep on ground table z = radius
            self.positions[i, 2] = self.radius
            self.velocities[i, 2] = 0.0

            # Velocity at contact point with ground: v_contact = v + omega x (-R * z_hat)
            # contact point is at (0, 0, -R) relative to center
            # omega x (0, 0, -R) = (omega_y * (-R) - omega_z * 0, omega_z * 0 - omega_x * (-R), 0)
            # = (-R * omega_y, R * omega_x, 0)
            v_contact_x = self.velocities[i, 0] - self.radius * self.angular_velocities[i, 1]
            v_contact_y = self.velocities[i, 1] + self.radius * self.angular_velocities[i, 0]
            v_contact = np.array([v_contact_x, v_contact_y, 0.0])
            v_slip = np.linalg.norm(v_contact[:2])

            m = self.masses[i]
            N = m * self.g_acc
            I = self.I_sphere[i]

            if v_slip > 1e-4:
                # Sliding friction opposes slip velocity
                slip_dir = v_contact[:2] / v_slip
                f_friction = self.mu_sliding * N

                # Friction force on center of mass
                f_linear = -f_friction * slip_dir
                a_linear = f_linear / m
                self.velocities[i, 0] += a_linear[0] * dt
                self.velocities[i, 1] += a_linear[1] * dt

                # Torque on sphere: tau = r_contact x F_friction
                # r_contact = (0, 0, -R), F_friction = (f_x, f_y, 0)
                # tau_x = -R * f_y * (-1) = R * f_y ? Let's check:
                # r x F = (0*0 - (-R)*f_y, (-R)*f_x - 0*0, 0) = (R*f_y, -R*f_x, 0)
                tau_x = self.radius * f_linear[1]
                tau_y = -self.radius * f_linear[0]

                self.angular_velocities[i, 0] += (tau_x / I) * dt
                self.angular_velocities[i, 1] += (tau_y / I) * dt
            else:
                # Pure rolling condition: v = R * omega (perpendicularly)
                # Rolling friction retards motion
                speed = np.linalg.norm(self.velocities[i, :2])
                if speed > 1e-5:
                    v_dir = self.velocities[i, :2] / speed
                    # Rolling deceleration: a_roll = mu_r * g
                    a_roll = self.mu_rolling * self.g_acc
                    speed_new = max(0.0, speed - a_roll * dt)
                    self.velocities[i, :2] = v_dir * speed_new

                    # Couple angular velocity to rolling speed
                    # v_x = R * omega_y => omega_y = v_x / R
                    # v_y = -R * omega_x => omega_x = -v_y / R
                    self.angular_velocities[i, 0] = -self.velocities[i, 1] / self.radius
                    self.angular_velocities[i, 1] = self.velocities[i, 0] / self.radius
                else:
                    self.velocities[i, :2] = 0.0
                    self.angular_velocities[i, :2] = 0.0

        # 3. Integrate positions
        self.positions += self.velocities * dt
        self.positions[:, 2] = self.radius  # strictly on table

    def advance_to(self, target_time: float) -> StateFrame:
        # Step until target_time
        while self.time + 1e-9 < target_time:
            dt = min(self.sub_dt, target_time - self.time)
            self._step(dt)
            self.time += dt

        self.time = target_time

        # Calculate diagnostics/fields for the StateFrame
        contact_distances, interaction_forces, external_forces = self._compute_contacts_and_forces()

        return StateFrame(
            time=float(self.time),
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            angular_velocity=self.angular_velocities.copy(),
            mass=self.masses.copy(),
            body_id=self.body_ids.copy(),
            fields={
                "contact_signed_distance": contact_distances,
                "interaction_force": interaction_forces,
                "external_force": external_forces,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_balls": self.num_balls,
            "sub_dt": self.sub_dt,
            "current_time": self.time,
        }

