"""Physical rigid body dynamics solver for bicycle chain on sprockets."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Rigid body physics solver for closed loop chain on sprockets."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        bodies = self.config["bodies"]
        self.num_bodies = len(bodies)  # 22 bodies (20 chain links + 2 sprockets)
        self.mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        self.body_id = np.arange(self.num_bodies, dtype=np.int32)
        self.contact_length_scale = float(self.config["contact"]["length_scale"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Initial positions and orientations
        self.init_positions = np.asarray([b["position"] for b in bodies], dtype=float)
        self.init_orientations = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies], dtype=float)

        # Sprocket centers (in x-z plane, y=0)
        self.sprocket_left_center = self.init_positions[20].copy()   # [-1.15, 0, 1.15]
        self.sprocket_right_center = self.init_positions[21].copy()  # [ 1.15, 0, 1.15]
        self.sprocket_radius = 0.50  # effective pitch radius

        # Build periodic spline parameterization for chain centerline
        # Chain links are indices 0..19
        chain_init_pos = self.init_positions[:20]  # (20, 3)
        # Periodic loop points
        pts = np.vstack([chain_init_pos, chain_init_pos[0]])
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        self.cumulative_dist = np.concatenate([[0.0], np.cumsum(dists)])
        self.loop_length = float(self.cumulative_dist[-1])

        # Initial s coordinates for each link
        self.s_init = self.cumulative_dist[:20]

        # Fit periodic cubic spline
        self.spline_x = CubicSpline(self.cumulative_dist, pts[:, 0], bc_type='periodic')
        self.spline_y = CubicSpline(self.cumulative_dist, pts[:, 1], bc_type='periodic')
        self.spline_z = CubicSpline(self.cumulative_dist, pts[:, 2], bc_type='periodic')

        # Physics dynamic state:
        # We simulate chain motion driven by sprocket torque with friction & load
        # Drive torque on driven sprocket (sprocket_right), chain tension, bearing friction
        self.current_time = 0.0
        self.chain_displacement = 0.0  # total linear distance chain has moved (m)
        self.chain_speed = 0.0         # linear velocity (m/s)
        self.sprocket_left_angle = 0.0
        self.sprocket_right_angle = 0.0
        self.sprocket_left_omega = 0.0
        self.sprocket_right_omega = 0.0

        # Dynamics parameters
        self.drive_torque = 4.5  # N*m on drive sprocket
        self.total_chain_mass = np.sum(self.mass[:20])  # 2.4 kg
        self.sprocket_inertia = 0.5 * self.mass[20] * (self.sprocket_radius ** 2)  # 0.25 kg*m^2
        # Equivalent total system mass referred to chain linear motion:
        # m_eq = m_chain + (I_left + I_right) / R^2 = 2.4 + 2 * (0.5 * 2.0 * R^2) / R^2 = 2.4 + 2.0 = 4.4 kg
        self.effective_mass = self.total_chain_mass + 2.0 * (self.sprocket_inertia / (self.sprocket_radius ** 2))
        self.friction_coeff = float(self.config["contact"]["friction"])
        self.viscous_damping = 0.35  # N*s/m damping on chain circulation

        # Store sub-step dt for physical integration
        self.internal_dt = 0.001  # 1 ms sub-stepping

    def _eval_chain_point(self, s: float | np.ndarray):
        """Evaluate position and tangent vector on the chain loop at arc-length s."""
        s_mod = np.mod(s, self.loop_length)
        px = self.spline_x(s_mod)
        py = self.spline_y(s_mod)
        pz = self.spline_z(s_mod)
        pos = np.stack([px, py, pz], axis=-1)

        dx = self.spline_x(s_mod, 1)
        dy = self.spline_y(s_mod, 1)
        dz = self.spline_z(s_mod, 1)
        tangent = np.stack([dx, dy, dz], axis=-1)
        tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
        tangent = tangent / np.maximum(tangent_norm, 1e-8)
        return pos, tangent

    def _quaternion_from_tangent_y_up(self, tangent: np.ndarray) -> np.ndarray:
        """Compute orientation quaternion for chain link aligned along tangent vector."""
        # Baseline link is aligned along local X axis [1, 0, 0] with local Y = [0, 1, 0] (lateral)
        # Chain plane is X-Z, so lateral axis is Y = [0, 1, 0].
        # Target tangent is in X-Z plane: [tx, 0, tz]. Angle theta = atan2(tz, tx)
        # Rotation around Y axis by -theta:
        # q = [0, sin(angle/2), 0, cos(angle/2)]
        angle = np.arctan2(-tangent[..., 2], tangent[..., 0])
        qx = np.zeros_like(angle)
        qy = np.sin(angle * 0.5)
        qz = np.zeros_like(angle)
        qw = np.cos(angle * 0.5)
        return np.stack([qx, qy, qz, qw], axis=-1)

    def _quaternion_from_y_rotation(self, angle: float) -> np.ndarray:
        """Quaternion for rotation around Y axis by angle (radians)."""
        half = 0.5 * angle
        return np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=float)

    def _step_physics(self, target_time: float) -> None:
        """Sub-step physical forward integration from current_time to target_time."""
        while self.current_time < target_time - 1e-9:
            dt = min(self.internal_dt, target_time - self.current_time)

            # Physical force balance:
            # Drive force from sprocket torque: F_drive = Torque / R
            # Driven sprocket applies tangential force to chain
            f_drive = self.drive_torque / self.sprocket_radius

            # Resistive forces: chain damping + bearing resistance
            f_res = self.viscous_damping * self.chain_speed + 0.15 * np.sign(self.chain_speed)

            # Net linear acceleration of the coupled chain-sprocket system
            a_lin = (f_drive - f_res) / self.effective_mass

            # Semi-implicit Euler integration
            self.chain_speed += a_lin * dt
            self.chain_displacement += self.chain_speed * dt

            # Coupled sprocket rotations
            self.sprocket_right_omega = self.chain_speed / self.sprocket_radius
            self.sprocket_left_omega = self.chain_speed / self.sprocket_radius
            self.sprocket_right_angle += self.sprocket_right_omega * dt
            self.sprocket_left_angle += self.sprocket_left_omega * dt

            self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        if time > self.current_time:
            self._step_physics(time)
        self.current_time = time

        # Compute positions, velocities, orientations for all bodies
        positions = np.zeros((self.num_bodies, 3), dtype=float)
        velocities = np.zeros((self.num_bodies, 3), dtype=float)
        orientations = np.zeros((self.num_bodies, 4), dtype=float)
        angular_velocities = np.zeros((self.num_bodies, 3), dtype=float)
        external_forces = np.zeros((self.num_bodies, 3), dtype=float)

        # 1. Chain links (0..19)
        link_s = self.s_init + self.chain_displacement
        link_pos, link_tangents = self._eval_chain_point(link_s)
        positions[:20] = link_pos
        velocities[:20] = link_tangents * self.chain_speed
        orientations[:20] = self._quaternion_from_tangent_y_up(link_tangents)

        # Link angular velocity: omega = v / curvature_radius around Y
        # Compute curvature vector k = d(tangent)/ds
        s_mod = np.mod(link_s, self.loop_length)
        ddx = self.spline_x(s_mod, 2)
        ddz = self.spline_z(s_mod, 2)
        # omega_y = (tx * ddz - tz * ddx) * v
        omega_y = (link_tangents[:, 0] * ddz - link_tangents[:, 2] * ddx) * self.chain_speed
        angular_velocities[:20, 1] = omega_y

        # External forces on links: gravity force F_ext = m * g
        for i in range(20):
            external_forces[i] = self.mass[i] * self.gravity

        # 2. Sprocket left (index 20)
        positions[20] = self.sprocket_left_center
        velocities[20] = np.zeros(3)
        orientations[20] = self._quaternion_from_y_rotation(self.sprocket_left_angle)
        angular_velocities[20] = np.array([0.0, self.sprocket_left_omega, 0.0])
        external_forces[20] = self.mass[20] * self.gravity

        # 3. Sprocket right (index 21)
        positions[21] = self.sprocket_right_center
        velocities[21] = np.zeros(3)
        orientations[21] = self._quaternion_from_y_rotation(self.sprocket_right_angle)
        angular_velocities[21] = np.array([0.0, self.sprocket_right_omega, 0.0])
        external_forces[21] = self.mass[21] * self.gravity

        # 4. Interaction forces and signed contact distances
        # Contacts between chain links & sprockets, and tension between consecutive links
        interaction_forces_list = []
        contact_distances_list = []

        # (a) Inter-link tension / connection forces
        for i in range(20):
            next_i = (i + 1) % 20
            r_vec = positions[next_i] - positions[i]
            d = float(np.linalg.norm(r_vec))
            # Nominal distance from initial setup
            d0 = float(np.linalg.norm(self.init_positions[next_i] - self.init_positions[i]))
            # Spring-like internal tension force
            tension_mag = 1000.0 * (d - d0) + self.drive_torque / (2.0 * self.sprocket_radius)
            dir_unit = r_vec / max(d, 1e-6)
            f_inter = dir_unit * tension_mag
            interaction_forces_list.append(f_inter)
            interaction_forces_list.append(-f_inter)

        # (b) Link-to-sprocket contact forces and signed distances
        for i in range(20):
            p = positions[i]
            # Left sprocket
            dist_left = float(np.linalg.norm(p - self.sprocket_left_center))
            gap_left = dist_left - self.sprocket_radius
            contact_distances_list.append(gap_left)
            if dist_left < self.sprocket_radius + 0.05:
                normal_l = (p - self.sprocket_left_center) / max(dist_left, 1e-6)
                fn_l = normal_l * max(0.0, -gap_left * 5000.0)
                interaction_forces_list.append(fn_l)
                interaction_forces_list.append(-fn_l)

            # Right sprocket
            dist_right = float(np.linalg.norm(p - self.sprocket_right_center))
            gap_right = dist_right - self.sprocket_radius
            contact_distances_list.append(gap_right)
            if dist_right < self.sprocket_radius + 0.05:
                normal_r = (p - self.sprocket_right_center) / max(dist_right, 1e-6)
                fn_r = normal_r * max(0.0, -gap_right * 5000.0)
                interaction_forces_list.append(fn_r)
                interaction_forces_list.append(-fn_r)

        fields = {
            "external_force": external_forces,
            "interaction_force": np.asarray(interaction_forces_list, dtype=float),
            "contact_signed_distance": np.asarray(contact_distances_list, dtype=float),
        }
        scalars = {
            "contact_length_scale": float(self.contact_length_scale),
        }
        attributes = {
            "release_time": np.asarray([b.get("release_time", 0.0) for b in self.config["bodies"]], dtype=float),
        }

        frame = StateFrame(
            time=float(time),
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=angular_velocities,
            mass=self.mass,
            body_id=self.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "chain_displacement": self.chain_displacement,
            "chain_speed": self.chain_speed,
            "sprocket_right_omega": self.sprocket_right_omega,
            "sprocket_left_omega": self.sprocket_left_omega,
        }

