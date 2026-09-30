"""BenchmarkSolver implementation for friction ball loop."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu = float(self.config.get("friction_coefficient", 0.18))

        # Ball parameters
        self.r_b = 0.24
        self.m = 1.0
        self.I_ball = (2.0 / 5.0) * self.m * (self.r_b ** 2)

        # Loop track geometry
        # Center of the loop is at (0.0, 0.0, 1.2) in x-z plane (y=0)
        # Inner surface radius is R - t = 1.0 - 0.2 = 0.8
        # Ball-center path radius is rho = 0.8 - 0.24 = 0.56
        self.loop_center = np.array([0.0, 0.0, 1.2], dtype=float)
        self.loop_inner_radius = 0.80
        self.rho = self.loop_inner_radius - self.r_b  # 0.56

        # Ground plane at z = 0 (top surface of box half_extents z=0.08, center z=-0.08)
        self.ground_z = 0.0

        # Initial state from scene
        init_frame = scene.initial_state()
        self.t = float(init_frame.time)
        self.pos = np.array(init_frame.position[0], dtype=float)
        self.vel = np.array(init_frame.velocity[0], dtype=float)
        self.omega = np.array(init_frame.angular_velocity[0], dtype=float)
        self.quat = np.array(init_frame.orientation[0], dtype=float)  # [x, y, z, w]

        # Contact length scale and constraint scale
        self.contact_length_scale = self.r_b
        self.constraint_scale = 1.0

        # Sub-stepping setup
        self.sub_dt = 1e-4

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        # q = [x, y, z, w]
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dtype=float)

    def _quaternion_integrate(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            angle = omega_norm * dt
            axis = omega / omega_norm
            s = math.sin(0.5 * angle)
            c = math.cos(0.5 * angle)
            dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=float)
            q_new = self._quaternion_multiply(dq, q)
        else:
            q_new = q.copy()
        q_new /= np.linalg.norm(q_new)
        return q_new

    def _physics_step(self, dt: float):
        # 1. External forces (gravity)
        f_ext = self.m * self.gravity
        total_force = f_ext.copy()
        total_torque = np.zeros(3, dtype=float)

        # Check contact with loop track:
        # Loop is in the x-z plane around self.loop_center
        # Relative vector in x-z plane from loop center to ball center
        rel_pos = self.pos - self.loop_center
        rel_pos_xz = np.array([rel_pos[0], 0.0, rel_pos[2]], dtype=float)
        r_xz = np.linalg.norm(rel_pos_xz)

        in_contact = False
        signed_dist = 1.0  # default far
        c_val = 0.0
        c_tar = 0.0

        # Loop inner surface signed distance:
        # Distance from loop center to ball surface is r_xz + r_b
        # Inner rail surface is at loop_inner_radius = 0.80 from center
        # gap = loop_inner_radius - (r_xz + r_b)
        gap_loop = self.loop_inner_radius - (r_xz + self.r_b)
        gap_ground = (self.pos[2] - self.r_b) - self.ground_z

        # Penalty contact parameters
        k_n = 5e5
        d_n = 200.0

        if gap_loop < 0.02 and abs(rel_pos[1]) < 0.3:  # Near loop rail
            # Contact normal pointing INTO the ball from the track surface:
            # The track is outside the loop center, so track pushes inwards toward loop center
            if r_xz > 1e-6:
                n_contact = -rel_pos_xz / r_xz
            else:
                n_contact = np.array([0.0, 0.0, 1.0])

            # Signed distance (negative for penetration)
            signed_dist = gap_loop
            c_val = r_xz + self.r_b
            c_tar = self.loop_inner_radius

            # Contact point on ball surface:
            # pos + (-n_contact) * r_b = pos - n_contact * r_b
            r_cp = -n_contact * self.r_b  # vector from ball center to contact point

            # Penetration depth
            delta = -gap_loop
            if delta > 0:
                # Normal relative velocity of contact point
                v_cp = self.vel + np.cross(self.omega, r_cp)
                v_n = np.dot(v_cp, n_contact)

                # Spring-damper normal force
                f_n_mag = max(0.0, k_n * delta - d_n * v_n)
                f_normal = f_n_mag * n_contact

                # Tangential relative velocity at contact point
                v_t = v_cp - v_n * n_contact
                v_t_norm = np.linalg.norm(v_t)

                if v_t_norm > 1e-6:
                    t_dir = v_t / v_t_norm
                    f_t_mag = self.mu * f_n_mag
                    f_friction = -f_t_mag * t_dir
                else:
                    # Rolling / static friction limit
                    f_friction = np.zeros(3, dtype=float)

                f_contact = f_normal + f_friction
                total_force += f_contact
                total_torque += np.cross(r_cp, f_contact)
                in_contact = True

        elif gap_ground < 0.02:  # Ground contact
            signed_dist = gap_ground
            c_val = self.pos[2] - self.r_b
            c_tar = self.ground_z

            delta = -gap_ground
            n_contact = np.array([0.0, 0.0, 1.0], dtype=float)
            r_cp = np.array([0.0, 0.0, -self.r_b], dtype=float)

            if delta > 0:
                v_cp = self.vel + np.cross(self.omega, r_cp)
                v_n = np.dot(v_cp, n_contact)
                f_n_mag = max(0.0, k_n * delta - d_n * v_n)
                f_normal = f_n_mag * n_contact

                v_t = v_cp - v_n * n_contact
                v_t_norm = np.linalg.norm(v_t)
                if v_t_norm > 1e-6:
                    t_dir = v_t / v_t_norm
                    f_t_mag = self.mu * f_n_mag
                    f_friction = -f_t_mag * t_dir
                else:
                    f_friction = np.zeros(3, dtype=float)

                f_contact = f_normal + f_friction
                total_force += f_contact
                total_torque += np.cross(r_cp, f_contact)
                in_contact = True
        else:
            signed_dist = min(gap_loop, gap_ground)
            c_val = self.pos[2]
            c_tar = self.loop_center[2]

        # Integrate linear motion (semi-implicit Euler)
        acc = total_force / self.m
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Integrate angular motion
        alpha = total_torque / self.I_ball
        self.omega += alpha * dt
        self.quat = self._quaternion_integrate(self.quat, self.omega, dt)

        self.t += dt
        return signed_dist, c_val, c_tar

    def advance_to(self, time: float) -> StateFrame:
        # Step until target time
        last_sd = 0.0
        last_c_val = 0.0
        last_c_tar = 0.0

        while self.t < time - 1e-9:
            dt_step = min(self.sub_dt, time - self.t)
            last_sd, last_c_val, last_c_tar = self._physics_step(dt_step)

        # Construct StateFrame
        f_ext = (self.m * self.gravity).reshape(1, 3)

        frame = StateFrame(
            time=float(time),
            position=self.pos.reshape(1, 3).copy(),
            velocity=self.vel.reshape(1, 3).copy(),
            orientation=self.quat.reshape(1, 4).copy(),
            angular_velocity=self.omega.reshape(1, 3).copy(),
            mass=np.array([self.m], dtype=float),
            body_id=np.array([0], dtype=int),
            fields={
                "constraint_target": np.array([last_c_tar], dtype=float),
                "constraint_value": np.array([last_c_val], dtype=float),
                "contact_signed_distance": np.array([last_sd], dtype=float),
                "external_force": f_ext,
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "position": self.pos.tolist(),
            "velocity": self.vel.tolist(),
        }

