"""Solver Agent-owned implementation for spolling coin decay."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion q given angular velocity omega over dt."""
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return quat_normalize(q)
    axis = omega / omega_norm
    angle = omega_norm * dt
    dq = np.array([axis[0] * np.sin(angle * 0.5),
                   axis[1] * np.sin(angle * 0.5),
                   axis[2] * np.sin(angle * 0.5),
                   np.cos(angle * 0.5)], dtype=float)
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = dq
    q_new = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=float)
    return quat_normalize(q_new)


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.dt_sub = 0.0002  # 0.2 ms for high precision contact dynamics

        # Physical constants / parameters
        self.mass = 0.5
        self.radius = 0.48
        self.height = 0.08
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.friction_coeff = 0.55
        self.restitution = 0.12
        self.length_scale = 0.02

        # Coin state
        self.pos = np.zeros(3, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)  # World angular velocity

        # Inertia tensor in body frame (cylinder along local z axis)
        # I_zz = 0.5 * m * r^2
        # I_xx = I_yy = (1/12) * m * (3*r^2 + h^2)
        I_zz = 0.5 * self.mass * (self.radius ** 2)
        I_trans = (1.0 / 12.0) * self.mass * (3.0 * (self.radius ** 2) + (self.height ** 2))
        self.I_body = np.diag([I_trans, I_trans, I_zz])
        self.I_body_inv = np.linalg.inv(self.I_body)

        # Sampling perimeter points on top and bottom circular rims of cylinder
        # Local cylinder axis is Z (-h/2 to +h/2), radius R in XY plane
        num_rim_points = 64
        angles = np.linspace(0, 2 * np.pi, num_rim_points, endpoint=False)
        top_rim = np.column_stack([self.radius * np.cos(angles), self.radius * np.sin(angles), np.full(num_rim_points, self.height * 0.5)])
        bot_rim = np.column_stack([self.radius * np.cos(angles), self.radius * np.sin(angles), np.full(num_rim_points, -self.height * 0.5)])
        self.rim_points_local = np.vstack([bot_rim, top_rim])  # [128, 3]

        # Trajectory tracking fields
        self.last_contact_signed_distance = 0.0
        self.last_external_force = np.zeros((1, 3), dtype=float)
        self.last_interaction_force = np.zeros((1, 3), dtype=float)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.current_time = 0.0

        config = scene.config
        self.gravity = np.array(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact_cfg = config.get("contact", {})
        self.friction_coeff = float(contact_cfg.get("friction", 0.55))
        self.restitution = float(contact_cfg.get("restitution", 0.12))
        self.length_scale = float(contact_cfg.get("length_scale", 0.02))

        body_cfg = config["bodies"][0]
        self.pos = np.array(body_cfg.get("position", [0.0, 0.0, 0.55]), dtype=float)
        init_quat = np.array(body_cfg.get("orientation", [0.0, 0.45, 0.0, 0.893]), dtype=float)
        self.quat = quat_normalize(init_quat)
        self.mass = float(body_cfg.get("mass", 0.5))

        # Recompute inertia
        I_zz = 0.5 * self.mass * (self.radius ** 2)
        I_trans = (1.0 / 12.0) * self.mass * (3.0 * (self.radius ** 2) + (self.height ** 2))
        self.I_body = np.diag([I_trans, I_trans, I_zz])
        self.I_body_inv = np.linalg.inv(self.I_body)

        # Initial velocity: spolling / Euler's disk initial state
        # A coin tilted and given initial spin + precession
        # Coin local normal is R @ [0, 0, 1]
        R = quat_to_rot_matrix(self.quat)
        normal_world = R @ np.array([0.0, 0.0, 1.0])

        # Initial spin around normal axis and vertical precession
        spin_rate = 22.0  # rad/s along coin axis
        precess_rate = 6.0  # rad/s around vertical Z axis
        self.omega = normal_world * spin_rate + np.array([0.0, 0.0, precess_rate])

        # Initial linear velocity to support rolling contact
        # Find lowest point
        world_rim = (R @ self.rim_points_local.T).T + self.pos
        min_idx = np.argmin(world_rim[:, 2])
        r_contact = world_rim[min_idx] - self.pos
        # v = - omega x r_contact for rolling
        self.vel = -np.cross(self.omega, r_contact)
        self.vel[2] = 0.0  # level translation

        self.last_contact_signed_distance = world_rim[min_idx, 2]
        self.last_external_force = (self.mass * self.gravity).reshape(1, 3)
        self.last_interaction_force = np.zeros((1, 3), dtype=float)

    def _physics_substep(self, dt: float) -> None:
        R = quat_to_rot_matrix(self.quat)
        I_world = R @ self.I_body @ R.T
        I_world_inv = R @ self.I_body_inv @ R.T

        # Find contact points with the floor (z = 0 plane)
        # Floor top surface is at z = 0.0 (box half_extent z=0.05 centered at z=-0.05)
        world_pts = (R @ self.rim_points_local.T).T + self.pos  # [128, 3]
        z_coords = world_pts[:, 2]

        min_z = np.min(z_coords)
        self.last_contact_signed_distance = float(min_z)

        # Active contact threshold
        penetration_threshold = 0.005  # Contact detection zone
        contact_indices = np.where(z_coords < penetration_threshold)[0]

        total_contact_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)

        # External force = gravity
        ext_force = self.mass * self.gravity
        self.last_external_force = ext_force.reshape(1, 3)

        # Euler equation gyroscopic torque: - omega x (I * omega)
        gyroscopic_torque = -np.cross(self.omega, I_world @ self.omega)
        total_torque += gyroscopic_torque

        # Rolling resistance & air drag torque to model energy decay into rest
        # Damping proportional to angular velocity
        damping_torque = -0.0008 * self.omega
        # Additional quadratic dissipation at high precession/spin
        damping_torque -= 0.00005 * np.linalg.norm(self.omega) * self.omega
        total_torque += damping_torque

        if len(contact_indices) > 0:
            # Distribute penalty/barrier contact forces among penetrating/near-penetrating rim points
            # Using soft-contact / compliant contact model with Coulomb friction
            k_contact = 1.5e5   # N/m normal stiffness
            c_contact = 3.0e2   # N*s/m normal damping
            k_tangent = 4.0e4   # Tangential stiffness
            c_tangent = 1.5e2   # Tangential damping

            num_c = len(contact_indices)
            for idx in contact_indices:
                r_c = world_pts[idx] - self.pos  # vector from COM to contact point
                v_pt = self.vel + np.cross(self.omega, r_c)

                pen = -world_pts[idx, 2]  # positive if penetrating below z=0
                if pen > -0.002:
                    # Normal contact force (spring-damper)
                    # Smooth activation for stability
                    overlap = max(0.0, pen)
                    f_normal_mag = k_contact * overlap - c_contact * min(0.0, v_pt[2])
                    f_normal_mag = max(0.0, f_normal_mag) / num_c

                    f_normal = np.array([0.0, 0.0, f_normal_mag])

                    # Friction force in tangent plane (XY)
                    v_tangent = np.array([v_pt[0], v_pt[1], 0.0])
                    v_tangent_norm = np.linalg.norm(v_tangent)
                    max_f_friction = self.friction_coeff * f_normal_mag

                    if v_tangent_norm > 1e-5:
                        f_friction_dir = -v_tangent / v_tangent_norm
                        # Dynamic / viscous friction transition
                        f_friction_mag = min(max_f_friction, c_tangent * v_tangent_norm / num_c)
                        f_friction = f_friction_dir * f_friction_mag
                    else:
                        f_friction = np.zeros(3)

                    f_pt = f_normal + f_friction
                    total_contact_force += f_pt
                    total_torque += np.cross(r_c, f_pt)

        self.last_interaction_force = total_contact_force.reshape(1, 3)

        # Integrate linear motion
        acc = (ext_force + total_contact_force) / self.mass
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Hard non-penetration enforcement for floor boundary
        # If COM drops too low or any point penetrates deeply
        min_pt_z = np.min((quat_to_rot_matrix(self.quat) @ self.rim_points_local.T).T[:, 2] + self.pos[2])
        if min_pt_z < 0.0:
            self.pos[2] -= min_pt_z
            if self.vel[2] < 0.0:
                self.vel[2] = 0.0

        # Integrate angular motion
        ang_acc = I_world_inv @ total_torque
        self.omega += ang_acc * dt
        self.quat = quat_integrate(self.quat, self.omega, dt)

    def advance_to(self, time: float) -> StateFrame:
        # Step forward in small stable physics increments until reaching sample time
        while self.current_time < time - 1e-8:
            dt = min(self.dt_sub, time - self.current_time)
            self._physics_substep(dt)
            self.current_time += dt

        self.current_time = time

        # Format StateFrame
        pos_arr = self.pos.reshape(1, 3).astype(float)
        vel_arr = self.vel.reshape(1, 3).astype(float)
        quat_arr = self.quat.reshape(1, 4).astype(float)
        omega_arr = self.omega.reshape(1, 3).astype(float)
        mass_arr = np.array([self.mass], dtype=float)
        body_id_arr = np.array([0], dtype=np.int32)

        frame = StateFrame(
            time=float(time),
            position=pos_arr,
            velocity=vel_arr,
            orientation=quat_arr,
            angular_velocity=omega_arr,
            mass=mass_arr,
            body_id=body_id_arr,
            scalars={
                "contact_length_scale": float(self.length_scale)
            },
            fields={
                "contact_signed_distance": np.array([self.last_contact_signed_distance], dtype=float),
                "external_force": self.last_external_force.copy(),
                "interaction_force": self.last_interaction_force.copy(),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "pos": self.pos.tolist(),
            "omega": self.omega.tolist(),
            "min_contact_gap": float(self.last_contact_signed_distance)
        }

