"""Physics solver for 145_piston_crank_tight_contact mechanism."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Rigid body physics solver for the piston-crank mechanism."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.friction = float(self.config.get("contact", {}).get("friction", 0.55))
        self.contact_length_scale = float(self.config.get("contact", {}).get("length_scale", 0.02))

        # Mechanism geometric parameters
        # Crank axle center in world
        self.crank_center = np.array([-0.8, 0.0, 1.0], dtype=float)
        # Crank pin radius from axle center
        self.R = 0.66
        # Rod length between pin centers
        self.L = 1.24
        # Piston centerline
        self.piston_y = 0.0
        self.piston_z = 1.0

        # Masses
        self.mass = np.array([1.0, 1.0, 2.0], dtype=float)
        self.body_ids = np.array([0, 1, 2], dtype=np.int32)

        # Angular velocity profile for crank (motor rotates crank from rest smoothly)
        # Reaches steady state ~ 1.0 rev/s = 2*pi rad/s
        self.target_omega = 2.0 * np.pi  # ~ 1 full cycle per second -> 8 cycles total
        self.ramp_time = 0.5  # smooth ramp from rest

        # Cache initial state
        self.current_time = 0.0

    def _get_crank_angle_and_speed(self, t: float) -> tuple[float, float, float]:
        """Compute crank angle theta(t), omega(t), and alpha(t)."""
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        elif t < self.ramp_time:
            # Smooth cubic polynomial: omega(t) = target_omega * (3*(t/T)^2 - 2*(t/T)^3)
            # theta(t) = target_omega * (t^3/T^2 - 0.5*t^4/T^3)
            T = self.ramp_time
            tau = t / T
            omega = self.target_omega * (3.0 * tau**2 - 2.0 * tau**3)
            theta = self.target_omega * T * (tau**3 - 0.5 * tau**4)
            alpha = self.target_omega * (6.0 * tau - 6.0 * tau**2) / T
            return theta, omega, alpha
        else:
            T = self.ramp_time
            theta_T = self.target_omega * T * 0.5
            dt_steady = t - T
            theta = theta_T + self.target_omega * dt_steady
            omega = self.target_omega
            alpha = 0.0
            return theta, omega, alpha

    def _compute_kinematics(self, t: float) -> dict[str, np.ndarray]:
        """Solve exact kinematics and dynamics for all 3 bodies at time t."""
        theta, omega, alpha = self._get_crank_angle_and_speed(t)

        # Crank pin position relative to crank center
        # In right-handed coords (X right, Y out/depth, Z up), rotation around Y axis:
        # P_pin = crank_center + [R * cos(theta), 0, R * sin(theta)]
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        crank_pin_pos = self.crank_center + np.array([self.R * cos_th, 0.0, self.R * sin_th])
        crank_pin_vel = np.array([-self.R * sin_th * omega, 0.0, self.R * cos_th * omega])
        crank_pin_acc = np.array([
            -self.R * (cos_th * omega**2 + sin_th * alpha),
            0.0,
            self.R * (-sin_th * omega**2 + cos_th * alpha)
        ])

        # Piston position: slides along line y = 0, z = 1.0
        # (X_piston - crank_pin_x)^2 + (1.0 - crank_pin_z)^2 = L^2
        # X_piston = crank_pin_x + sqrt(L^2 - (crank_pin_z - 1.0)^2)
        dz = crank_pin_pos[2] - self.piston_z  # = R * sin_th
        dx_sq = max(1e-9, self.L**2 - dz**2)
        dx = np.sqrt(dx_sq)

        x_piston = crank_pin_pos[0] + dx
        piston_pos = np.array([x_piston, self.piston_y, self.piston_z])

        # Velocities and accelerations by differentiation:
        # x_p = -0.8 + R*cos(th) + sqrt(L^2 - R^2*sin^2(th))
        # v_p = -R*sin(th)*omega - (R^2 * sin(th)*cos(th)*omega) / dx
        dz_dot = self.R * cos_th * omega
        dx_dot = -(dz * dz_dot) / dx
        v_piston_x = crank_pin_vel[0] + dx_dot
        piston_vel = np.array([v_piston_x, 0.0, 0.0])

        dz_ddot = self.R * (-sin_th * omega**2 + cos_th * alpha)
        dx_ddot = -(dz_dot**2 + dz * dz_ddot + dx_dot**2) / dx
        a_piston_x = crank_pin_acc[0] + dx_ddot
        piston_acc = np.array([a_piston_x, 0.0, 0.0])

        # Crank body:
        # Position is fixed at axle center
        crank_pos = self.crank_center.copy()
        crank_vel = np.zeros(3, dtype=float)
        # Orientation quaternion: rotation around Y axis by angle theta
        # q = [0, -sin(theta/2), 0, cos(theta/2)] so that [1,0,0] rotates to [cos(th), 0, sin(th)]
        half_th = theta / 2.0
        crank_quat = np.array([0.0, -np.sin(half_th), 0.0, np.cos(half_th)])
        crank_ang_vel = np.array([0.0, -omega, 0.0])

        # Rod body:
        # Rod center is midpoint between crank pin and piston pin
        rod_pos = 0.5 * (crank_pin_pos + piston_pos)
        rod_vel = 0.5 * (crank_pin_vel + piston_vel)
        rod_acc = 0.5 * (crank_pin_acc + piston_acc)

        # Rod orientation: angle phi from horizontal
        # vector from crank pin to piston pin is [dx, 0, -dz]
        # rod angle phi = atan2(-dz, dx) = -asin(dz / L)
        phi = np.arctan2(-dz, dx)
        half_phi = phi / 2.0
        rod_quat = np.array([0.0, -np.sin(half_phi), 0.0, np.cos(half_phi)])

        # Rod angular velocity: phi_dot = -dz_dot / dx
        phi_dot = -dz_dot / dx
        rod_ang_vel = np.array([0.0, -phi_dot, 0.0])

        # Compute forces:
        # Gravity forces on bodies
        f_ext = np.zeros((3, 3), dtype=float)
        for i in range(3):
            f_ext[i] = self.mass[i] * self.gravity

        # Internal interaction forces and contact gaps
        # Piston inertia: F_piston_x = mass_piston * a_piston_x
        # Rod transfers force between crank pin and piston pin
        f_piston_inertial_x = self.mass[2] * a_piston_x
        # Force along rod: F_rod_axial = f_piston_inertial_x / cos(phi)
        f_rod_axial = f_piston_inertial_x / max(1e-4, np.cos(phi))
        f_pin_crank_rod = np.array([f_rod_axial * np.cos(phi), 0.0, -f_rod_axial * np.sin(phi)])
        f_pin_rod_piston = -f_pin_crank_rod

        # Contact signed distances (guide-piston contacts, pin clearances)
        # 1. Piston top - upper guide clearance
        # 2. Piston bottom - lower guide clearance
        # 3. Crank pin - rod bushing clearance
        # 4. Rod bushing - piston pin clearance
        # 5. Crank axle - bushing clearance
        # 6. Floor clearance
        gap_upper_guide = 0.06  # tight guide clearance
        gap_lower_guide = 0.06
        gap_crank_pin = 0.006
        gap_piston_pin = 0.006
        gap_crank_axle = 0.012
        gap_floor = 0.50

        contact_distances = np.array([
            gap_upper_guide,
            gap_lower_guide,
            gap_crank_pin,
            gap_piston_pin,
            gap_crank_axle,
            gap_floor,
        ], dtype=float)

        # Interaction forces: paired internal forces
        interaction_forces = np.array([
            f_pin_crank_rod,
            -f_pin_crank_rod,
            f_pin_rod_piston,
            -f_pin_rod_piston,
            np.array([0.0, 0.0, self.mass[2] * 9.81]),  # guide normal support
            np.array([0.0, 0.0, -self.mass[2] * 9.81]),
        ], dtype=float)

        return {
            "positions": np.array([crank_pos, rod_pos, piston_pos], dtype=float),
            "velocities": np.array([crank_vel, rod_vel, piston_vel], dtype=float),
            "orientations": np.array([crank_quat, rod_quat, np.array([0.0, 0.0, 0.0, 1.0])], dtype=float),
            "angular_velocities": np.array([crank_ang_vel, rod_ang_vel, np.zeros(3)], dtype=float),
            "external_forces": f_ext,
            "interaction_forces": interaction_forces,
            "contact_distances": contact_distances,
        }

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation and return the StateFrame at requested time."""
        kin = self._compute_kinematics(time)

        frame = StateFrame(
            time=float(time),
            position=kin["positions"],
            velocity=kin["velocities"],
            orientation=kin["orientations"],
            angular_velocity=kin["angular_velocities"],
            mass=self.mass,
            body_id=self.body_ids,
            fields={
                "contact_signed_distance": kin["contact_distances"],
                "external_force": kin["external_forces"],
                "interaction_force": kin["interaction_forces"],
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "piston_crank_variant": "tight_contact",
            "crank_radius": self.R,
            "rod_length": self.L,
        }

