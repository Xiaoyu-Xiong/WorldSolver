"""BenchmarkSolver implementation for 014_inclined_chair_stick_slip.

Recreates stick-slip chattering of a plastic chair sliding down an inclined plane.
Uses rigid body dynamics with compliance and friction modeling (Stribeck / Coulomb / spring-damper contact).
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    else:
        return np.eye(3)

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate orientation quaternion given angular velocity omega over dt."""
    w_len = np.linalg.norm(omega)
    if w_len < 1e-12:
        return q.copy()

    theta = w_len * dt
    axis = omega / w_len
    # dq = [axis * sin(theta/2), cos(theta/2)]
    s = math.sin(0.5 * theta)
    c = math.cos(0.5 * theta)
    dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

    # Quaternion multiplication: q_new = dq * q
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q

    q_new = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

    norm = np.linalg.norm(q_new)
    if norm > 0:
        q_new /= norm
    return q_new


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = init_frame.position[0].astype(float).copy()
        self.vel = init_frame.velocity[0].astype(float).copy()
        self.quat = init_frame.orientation[0].astype(float).copy()
        self.omega = init_frame.angular_velocity[0].astype(float).copy()
        self.mass = float(init_frame.mass[0])
        self.body_id = int(init_frame.body_id[0])

        # Incline geometry & surface definition
        # Incline rotation around Y by angle theta
        theta = 2.0 * math.asin(0.1564)
        c, s = math.cos(theta), math.sin(theta)
        self.R_incline = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        self.n_plane = self.R_incline @ np.array([0.0, 0.0, 1.0])  # Plane normal pointing outwards
        self.t_plane = self.R_incline @ np.array([-1.0, 0.0, 0.0])  # Downhill direction
        self.plane_top_world = np.array([0.0, 0.0, 0.35]) + self.R_incline @ np.array([0.0, 0.0, 0.08])

        # Gravity
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Approximate chair principal moments of inertia
        # Chair dimensions ~ 1.3m x 1.1m x 2.7m, mass 10.8 kg
        # Treat as rough bounding box / hollow shell
        Ixx = (1.0 / 12.0) * self.mass * (1.1**2 + 2.7**2) * 0.4
        Iyy = (1.0 / 12.0) * self.mass * (1.3**2 + 2.7**2) * 0.4
        Izz = (1.0 / 12.0) * self.mass * (1.3**2 + 1.1**2) * 0.4
        self.I_body = np.diag([Ixx, Iyy, Izz])
        self.I_body_inv = np.linalg.inv(self.I_body)

        # Foot contact points in local chair coordinates (4 legs)
        self.foot_points_local = np.array([
            [-0.34648147, -0.42, -1.1268518],
            [-0.34648147,  0.42, -1.1268518],
            [ 0.58351851, -0.42, -1.1268518],
            [ 0.58351851,  0.42, -1.1268518],
        ])

        # Physical parameters for contact and stick-slip chatter
        # Contact normal stiffness and damping
        self.k_n = 2.5e5
        self.d_n = 4.0e2

        # Friction parameters (plastic on incline, stick-slip Stribeck model)
        # Incline angle = 18 degrees, tan(18 deg) = 0.3249
        # To get stick-slip: static friction > tan(18 deg) > dynamic friction
        # mu_s = 0.48, mu_k = 0.22, v_s = 0.05
        self.mu_s = 0.48
        self.mu_k = 0.22
        self.v_s = 0.04  # Stribeck velocity scale

        # Internal substep dt for numerical stability
        self.sub_dt = 1e-4

        # Track contact signed distances and friction residuals for diagnostics / fields
        self.last_contact_gaps = np.zeros(4)
        self.last_constraint_values = np.zeros(4)
        self.last_constraint_targets = np.zeros(4)

    def _compute_contact_forces(self, pos: np.ndarray, vel: np.ndarray,
                                quat: np.ndarray, omega: np.ndarray):
        """Compute contact forces and torques at the 4 feet."""
        R = quat_to_rot_matrix(quat)
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        contact_gaps = np.zeros(4)
        constraint_values = np.zeros(4)
        constraint_targets = np.zeros(4)

        for i in range(4):
            r_local = self.foot_points_local[i]
            r_world = R @ r_local
            p_world = pos + r_world

            # Signed distance to incline plane
            gap = np.dot(p_world - self.plane_top_world, self.n_plane)
            contact_gaps[i] = gap

            # Velocity of contact point
            v_point = vel + np.cross(omega, r_world)
            v_n = np.dot(v_point, self.n_plane)
            v_t_vec = v_point - v_n * self.n_plane
            v_t_mag = np.linalg.norm(v_t_vec)

            # Normal force (spring-damper, one-sided)
            if gap < 0.0:
                penetration = -gap
                # Normal force magnitude
                fn_val = self.k_n * penetration - self.d_n * v_n
                fn = max(0.0, fn_val)

                # Friction force with Stribeck effect for stick-slip chatter
                # mu(v) = mu_k + (mu_s - mu_k) * exp(-(v / v_s)^2)
                mu = self.mu_k + (self.mu_s - self.mu_k) * math.exp(-(v_t_mag / self.v_s) ** 2)

                # Tangential regularized direction
                if v_t_mag > 1e-5:
                    t_dir = v_t_vec / v_t_mag
                    # Regularization factor for zero-velocity stability
                    reg = math.tanh(v_t_mag / 0.01)
                    f_friction = -mu * fn * reg * t_dir
                else:
                    reg = v_t_mag / 0.01
                    f_friction = -mu * fn * reg * (v_t_vec / (v_t_mag + 1e-8))

                f_contact = fn * self.n_plane + f_friction
                total_force += f_contact
                total_torque += np.cross(r_world, f_contact)

                # Constraint diagnostics: friction cone residual |f_t| / (mu * fn)
                constraint_values[i] = np.linalg.norm(f_friction)
                constraint_targets[i] = mu * fn
            else:
                constraint_values[i] = 0.0
                constraint_targets[i] = 0.0

        return total_force, total_torque, contact_gaps, constraint_values, constraint_targets

    def _step(self, dt: float) -> None:
        """Advance the physical state by dt using symplectic / semi-implicit Euler."""
        R = quat_to_rot_matrix(self.quat)
        I_world_inv = R @ self.I_body_inv @ R.T

        f_ext = self.mass * self.gravity
        f_contact, tau_contact, gaps, c_vals, c_targets = self._compute_contact_forces(
            self.pos, self.vel, self.quat, self.omega
        )

        f_total = f_ext + f_contact
        tau_total = tau_contact - np.cross(self.omega, (R @ self.I_body @ R.T) @ self.omega)

        # Accelerations
        acc = f_total / self.mass
        alpha = I_world_inv @ tau_total

        # Velocity update
        self.vel += acc * dt
        self.omega += alpha * dt

        # Position update
        self.pos += self.vel * dt
        self.quat = quat_integrate(self.quat, self.omega, dt)

        self.last_contact_gaps = gaps
        self.last_constraint_values = c_vals
        self.last_constraint_targets = c_targets

    def advance_to(self, time: float) -> StateFrame:
        """Step simulation forward to the requested time and return StateFrame."""
        dt_target = time - self.current_time
        if dt_target > 1e-9:
            num_substeps = max(1, int(math.ceil(dt_target / self.sub_dt)))
            sub_dt = dt_target / num_substeps
            for _ in range(num_substeps):
                self._step(sub_dt)
            self.current_time = time

        # Update contact diagnostics at current state
        _, _, gaps, c_vals, c_targets = self._compute_contact_forces(
            self.pos, self.vel, self.quat, self.omega
        )
        self.last_contact_gaps = gaps
        self.last_constraint_values = c_vals
        self.last_constraint_targets = c_targets

        ext_force = (self.mass * self.gravity).reshape(1, 3)

        return StateFrame(
            time=self.current_time,
            position=self.pos.reshape(1, 3),
            velocity=self.vel.reshape(1, 3),
            orientation=self.quat.reshape(1, 4),
            angular_velocity=self.omega.reshape(1, 3),
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([self.body_id], dtype=np.int32),
            fields={
                "constraint_target": self.last_constraint_targets.copy(),
                "constraint_value": self.last_constraint_values.copy(),
                "contact_signed_distance": self.last_contact_gaps.copy(),
                "external_force": ext_force,
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": 0.08,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
        }

