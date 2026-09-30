"""Inclined Block Rigid Body Physics Solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.time = 0.0
        self.mass = 2.0
        self.body_id = 1

        # Dimensions and geometry
        # Block half-extents: [0.45, 0.35, 0.3]
        # Plane half-extents: [3.2, 2.0, 0.08]
        # Incline angle: theta = 30 deg (0.5235987756 rad)
        self.theta = np.deg2rad(30.0)
        self.sin_t = np.sin(self.theta)
        self.cos_t = np.cos(self.theta)
        self.tan_t = np.tan(self.theta)

        # Rotation matrix of plane and block (rotation around X by -30 degrees):
        # [ 1,  0,        0      ]
        # [ 0,  cos(30),  sin(30)]
        # [ 0, -sin(30),  cos(30)]
        self.R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, self.cos_t, self.sin_t],
            [0.0, -self.sin_t, self.cos_t]
        ], dtype=float)

        # In plane local coordinates:
        # e_x = [1, 0, 0] (lateral)
        # e_y = [0, cos(30), -sin(30)] (downslope)
        # e_z = [0, sin(30), cos(30)] (plane normal)
        self.e_x = np.array([1.0, 0.0, 0.0], dtype=float)
        self.e_y = np.array([0.0, self.cos_t, -self.sin_t], dtype=float)
        self.e_z = np.array([0.0, self.sin_t, self.cos_t], dtype=float)

        # Orientation quaternion: xyzw
        self.quat = np.array([-0.258819, 0.0, 0.0, 0.965926], dtype=float)

        # Gravity
        self.g = 9.81
        self.gravity_vec = np.array([0.0, 0.0, -self.g], dtype=float)

        # Friction coefficient mu = 0.6 (from Guendelman et al. 2003 / Stanford paper)
        self.mu = 0.6
        self.contact_length_scale = 0.08  # Plane thickness / scale
        self.constraint_scale = 1.0

        # Contact plane surface height in local normal coord:
        # plane half extent z is 0.08, block half extent z is 0.3
        # Contact distance threshold along normal from plane center to block center:
        self.z_contact = 0.08 + 0.30  # 0.38 m

        # Sub-stepping
        self.sub_dt = 1e-4

        # Dynamic state in local coordinates:
        # pos_local = [x_l, y_l, z_l]
        # vel_local = [vx_l, vy_l, vz_l]
        self.pos_local = np.zeros(3, dtype=float)
        self.vel_local = np.zeros(3, dtype=float)
        self.in_contact = False

    def initialize(self, scene, seed: int) -> None:
        init_frame = scene.initial_state()
        pos_world = np.asarray(init_frame.position[0], dtype=float)
        vel_world = np.asarray(init_frame.velocity[0], dtype=float)

        if init_frame.mass is not None:
            self.mass = float(init_frame.mass[0])
        if init_frame.body_id is not None:
            self.body_id = int(init_frame.body_id[0])

        # Convert initial world position/velocity to local coordinates
        # pos_local = R^T @ pos_world
        self.pos_local = self.R.T @ pos_world
        self.vel_local = self.R.T @ vel_world
        self.time = 0.0
        self.in_contact = (self.pos_local[2] <= self.z_contact + 1e-6)

    def _step(self, dt: float) -> None:
        # Gravity in local frame:
        # g_local = R^T @ [0, 0, -g] = [0, g * sin(30), -g * cos(30)]
        g_x = 0.0
        g_y = self.g * self.sin_t
        g_z = -self.g * self.cos_t

        # Check if in contact with the inclined plane:
        if self.pos_local[2] <= self.z_contact:
            # Enforce non-penetration
            self.pos_local[2] = self.z_contact
            if self.vel_local[2] < 0.0:
                self.vel_local[2] = 0.0
            self.in_contact = True

            # Contact forces:
            # Normal force balances gravity normal component: F_N = m * g * cos(theta)
            # Maximum static friction force: F_f_max = mu * F_N = mu * m * g * cos(theta)
            # Downslope gravity force: F_g = m * g * sin(theta)
            v_slide = self.vel_local[1]

            if abs(v_slide) > 1e-5:
                # Kinetic friction opposing sliding direction
                # friction accel = -sign(v_slide) * mu * g * cos(theta)
                a_f = -np.sign(v_slide) * self.mu * self.g * self.cos_t
                a_y = g_y + a_f
                new_v_slide = v_slide + a_y * dt

                # Check if velocity crossed zero during this step (transition to rest)
                if (v_slide > 0.0 and new_v_slide <= 0.0) or (v_slide < 0.0 and new_v_slide >= 0.0):
                    # Check if static friction can hold it at rest
                    # Static friction can balance gravity if mu >= tan(theta)
                    if self.mu >= self.tan_t:
                        new_v_slide = 0.0
                self.vel_local[1] = new_v_slide
            else:
                # Block is at rest or near rest
                if self.mu >= self.tan_t:
                    self.vel_local[1] = 0.0
                else:
                    # Friction insufficient to hold at rest
                    a_y = g_y - self.mu * self.g * self.cos_t
                    self.vel_local[1] += a_y * dt

            # Lateral velocity damping/friction
            self.vel_local[0] = 0.0

            # Integrate downslope position
            self.pos_local[1] += self.vel_local[1] * dt
            self.pos_local[0] += self.vel_local[0] * dt
        else:
            # Free fall under gravity
            self.in_contact = False
            self.vel_local[0] += g_x * dt
            self.vel_local[1] += g_y * dt
            self.vel_local[2] += g_z * dt

            self.pos_local += self.vel_local * dt

            # Check impact within this step
            if self.pos_local[2] <= self.z_contact:
                self.pos_local[2] = self.z_contact
                # Inelastic impact
                self.vel_local[2] = 0.0
                self.in_contact = True

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation using sub-stepping up to target time
        while self.time + self.sub_dt < time + 1e-12:
            dt = min(self.sub_dt, time - self.time)
            if dt <= 0:
                break
            self._step(dt)
            self.time += dt

        if self.time < time:
            dt = time - self.time
            if dt > 0:
                self._step(dt)
                self.time = time

        # Current world position and velocity
        pos_world = self.R @ self.pos_local
        vel_world = self.R @ self.vel_local

        # Compute contact signed distance:
        # Signed geometric gap along plane normal: gap = z_local - z_contact
        signed_dist = float(self.pos_local[2] - self.z_contact)

        # Constraint target and value:
        # For non-penetration constraint, target is 0.0 (or non-negative gap)
        # constraint_value is the normal penetration/gap distance
        constraint_target = np.array([0.0], dtype=float)
        constraint_val = np.array([max(0.0, signed_dist)], dtype=float)

        # External force: gravity = m * g
        ext_force = (self.mass * self.gravity_vec).reshape(1, 3)

        return StateFrame(
            time=float(time),
            position=pos_world.reshape(1, 3),
            velocity=vel_world.reshape(1, 3),
            orientation=self.quat.reshape(1, 4),
            angular_velocity=np.zeros((1, 3), dtype=float),
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([self.body_id], dtype=int),
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
            fields={
                "constraint_target": constraint_target,
                "constraint_value": constraint_val,
                "contact_signed_distance": np.array([signed_dist], dtype=float),
                "external_force": ext_force,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "final_position": self.pos_local.tolist(),
            "in_contact": self.in_contact,
        }

