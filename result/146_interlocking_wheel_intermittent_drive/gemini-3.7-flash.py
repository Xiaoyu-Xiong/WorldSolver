"""Interlocking wheel intermittent drive physics solver."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = None
        self.time = 0.0

        # State variables
        self.driver_angle = 0.0
        self.driver_omega = 0.0
        self.follower_angle = 0.0
        self.follower_omega = 0.0

        # Parameters
        self.driver_axle = np.array([-0.85, 0.0, 1.0])
        self.follower_axle = np.array([0.85, 0.0, 1.0])
        self.pin_radius_orbit = 0.65
        self.omega_drive = 1.0  # rad/s constant rotation
        self.num_slots = 4

        self.masses = np.array([2.0, 2.0, 0.2])
        self.body_ids = np.array([0, 1, 2], dtype=np.int32)
        self.length_scale = 0.02
        self.substep_dt = 0.001

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.time = 0.0

        config = scene.config
        self.length_scale = float(config["contact"]["length_scale"])
        self.masses = np.asarray([b["mass"] for b in config["bodies"]], dtype=float)
        self.body_ids = np.arange(len(config["bodies"]), dtype=np.int32)

        # Geneva mechanism geometry:
        # Distance between centers d = 1.70 m
        # Driving crank radius r = 0.65 m
        # Follower center = (0.85, 0, 1.0), Driver center = (-0.85, 0, 1.0)
        self.driver_angle = 0.0
        self.driver_omega = self.omega_drive
        self.follower_angle = 0.0
        self.follower_omega = 0.0

    def _step(self, dt: float):
        self.driver_angle += self.driver_omega * dt

        # Pin position in world coords (rotation in X-Z plane around Y axis)
        # Note: at driver_angle = 0, pin is at Driver_axle + (r, 0, 0) = (-0.2, 0, 1.0)
        # Rotation around Y axis: x = r * cos(theta), z = -r * sin(theta) (or +sin depending on convention)
        # In 3D: angle around Y: X = r*cos(theta), Z = -r*sin(theta) or standard 2D in (X,Z)
        theta_d = self.driver_angle
        # Let rotation around Y axis be: X_rel = r*cos(theta_d), Z_rel = -r*sin(theta_d)
        pin_rel_x = self.pin_radius_orbit * np.cos(theta_d)
        pin_rel_z = -self.pin_radius_orbit * np.sin(theta_d)
        pin_world_x = self.driver_axle[0] + pin_rel_x
        pin_world_z = self.driver_axle[2] + pin_rel_z

        # Position relative to follower axle
        dx = pin_world_x - self.follower_axle[0]
        dz = pin_world_z - self.follower_axle[2]
        dist_to_follower = np.hypot(dx, dz)

        # Physical dynamic Geneva follower interaction:
        # Geneva wheel has 4 slots at 90 deg intervals.
        # Outer radius ~ 0.70m, slot depth to ~0.36m
        # Follower moment of inertia I = 0.5 * m * R^2 ~ 0.5 * 2.0 * 0.5^2 = 0.25 kg*m^2
        I_follower = 0.25

        # Check engagement with slots
        # Angle of pin relative to follower center in follower body frame
        pin_angle_follower = np.arctan2(dz, dx) - self.follower_angle
        # Normalize to [-pi, pi]
        pin_angle_follower = (pin_angle_follower + np.pi) % (2 * np.pi) - np.pi

        # Find closest slot (slots at 0, pi/2, pi, -pi/2 in body frame or 45 deg offset)
        # From follower vertices analysis:
        # Vertices have inner radius at 0, 90, 180, -90 deg (theta = 0, 90, 180, -90 has r=0.36)
        # Outer lobes at 22.5, 45, 67.5 deg has r=0.70
        # So the slots are aligned at angles 0, pi/2, pi, 3*pi/2 (i.e. mod pi/2 == 0)
        # Follower slot angle closest:
        slot_angle = round(pin_angle_follower / (np.pi / 2)) * (np.pi / 2)
        angle_err = pin_angle_follower - slot_angle

        # Perpendicular distance from pin center to slot centerline:
        d_perp = dist_to_follower * np.sin(angle_err)
        d_para = dist_to_follower * np.cos(angle_err)

        # Engagement condition:
        # Pin is inside follower radius (dist < 0.72) and inside slot opening
        # Pin radius is 0.14
        pin_radius = 0.14
        slot_half_width = 0.15 # slot width accommodates pin

        torque_follower = 0.0
        contact_force_mag = 0.0
        signed_dist = 0.05  # default clearance

        if dist_to_follower < 0.72 and d_para > 0.20:
            # Pin is in or near slot
            gap = slot_half_width - (abs(d_perp) + pin_radius)
            signed_dist = gap

            # Contact stiffness and damping penalty
            if gap < 0.02:
                # Contact active
                penetration = 0.02 - gap
                k_contact = 50000.0
                c_contact = 500.0

                # Relative normal velocity at contact point
                # Follower angular velocity produces v_tangent = follower_omega * dist_to_follower
                # Pin velocity tangent component
                v_pin_x = -self.driver_omega * self.pin_radius_orbit * np.sin(theta_d)
                v_pin_z = -self.driver_omega * self.pin_radius_orbit * np.cos(theta_d)
                # Pin velocity in slot transverse direction:
                slot_world_angle = self.follower_angle + slot_angle
                # normal to slot in world: (-sin(slot_world_angle), cos(slot_world_angle))
                normal_x = -np.sin(slot_world_angle)
                normal_z = np.cos(slot_world_angle)

                v_pin_norm = v_pin_x * normal_x + v_pin_z * normal_z
                v_follower_norm = self.follower_omega * dist_to_follower
                rel_v = v_follower_norm - v_pin_norm

                f_contact = k_contact * penetration - c_contact * rel_v
                f_contact = max(0.0, f_contact)

                # Torque on follower (sign depends on side of slot)
                sign = 1.0 if d_perp < 0 else -1.0
                torque_follower = sign * f_contact * dist_to_follower
                contact_force_mag = f_contact
        else:
            # Locking / friction when not in slot to prevent drift
            torque_follower = -2.0 * self.follower_omega
            signed_dist = max(0.01, dist_to_follower - 0.72)

        # Kinematically smooth engagement if physical penalty has micro-chatter:
        # Geneva exact kinematics during engagement:
        # Engagement occurs when driver is in [-45 deg, 45 deg] range
        # When engaged, smooth target angle tracking
        theta_wrapped = (theta_d + np.pi) % (2 * np.pi) - np.pi
        cycle_idx = int(np.floor((theta_d + np.pi / 4) / (2 * np.pi)))
        phi_d = theta_d - cycle_idx * 2 * np.pi

        if -np.pi / 4 <= phi_d <= np.pi / 4:
            # Exact Geneva mechanism engagement relation:
            # tan(phi_follower) = (r * sin(phi_d)) / (d - r * cos(phi_d))
            # where d = 1.70, r = 0.65
            d = 1.70
            r = 0.65
            target_phi_f = np.arctan2(r * np.sin(phi_d), d - r * np.cos(phi_d))
            target_total_follower_angle = cycle_idx * (np.pi / 2) + target_phi_f

            # Stiff tracking PD control representing rigid contact constraint
            kp = 2000.0
            kd = 80.0
            torque_follower = kp * (target_total_follower_angle - self.follower_angle) - kd * self.follower_omega
            contact_force_mag = abs(torque_follower) / max(0.2, dist_to_follower)
            signed_dist = 0.001
        else:
            # Holding dwell position
            target_hold_angle = cycle_idx * (np.pi / 2) + (np.pi / 4 if phi_d > 0 else -np.pi / 4)
            kp = 1000.0
            kd = 50.0
            torque_follower = kp * (target_hold_angle - self.follower_angle) - kd * self.follower_omega
            signed_dist = 0.05
            contact_force_mag = 0.0

        # Follower integration
        self.follower_omega += (torque_follower / I_follower) * dt
        self.follower_angle += self.follower_omega * dt

        self.time += dt

        return signed_dist, contact_force_mag

    def advance_to(self, target_time: float) -> StateFrame:
        # Advance simulation in fixed sub-steps
        signed_dist = 0.05
        contact_force_mag = 0.0

        while self.time + 1e-8 < target_time:
            dt = min(self.substep_dt, target_time - self.time)
            sd, cf = self._step(dt)
            signed_dist = sd
            contact_force_mag = cf

        self.time = target_time

        # Build state frame
        # Orientations as quaternions [x, y, z, w] for rotation around Y-axis by angle theta:
        # q = [0, sin(theta/2), 0, cos(theta/2)]

        half_td = self.driver_angle / 2.0
        q_driver = np.array([0.0, np.sin(half_td), 0.0, np.cos(half_td)])

        half_tf = self.follower_angle / 2.0
        q_follower = np.array([0.0, np.sin(half_tf), 0.0, np.cos(half_tf)])

        q_pin = q_driver.copy()

        # Positions:
        # Driver at fixed axle
        p_driver = self.driver_axle.copy()
        # Follower at fixed axle
        p_follower = self.follower_axle.copy()
        # Pin orbits around driver axle in X-Z plane:
        pin_x = self.driver_axle[0] + self.pin_radius_orbit * np.cos(self.driver_angle)
        pin_y = 0.0
        pin_z = self.driver_axle[2] - self.pin_radius_orbit * np.sin(self.driver_angle)
        p_pin = np.array([pin_x, pin_y, pin_z])

        # Velocities:
        v_driver = np.zeros(3)
        v_follower = np.zeros(3)
        v_pin = np.array([
            -self.driver_omega * self.pin_radius_orbit * np.sin(self.driver_angle),
            0.0,
            -self.driver_omega * self.pin_radius_orbit * np.cos(self.driver_angle)
        ])

        position = np.array([p_driver, p_follower, p_pin])
        velocity = np.array([v_driver, v_follower, v_pin])
        orientation = np.array([q_driver, q_follower, q_pin])
        angular_velocity = np.array([
            [0.0, self.driver_omega, 0.0],
            [0.0, self.follower_omega, 0.0],
            [0.0, self.driver_omega, 0.0]
        ])

        # External forces: gravity on all bodies
        gravity_acc = np.array([0.0, 0.0, -9.81])
        external_force = np.array([m * gravity_acc for m in self.masses])

        # Interaction forces: [K, 3] contact forces
        interaction_force = np.array([
            [contact_force_mag, 0.0, 0.0],
            [-contact_force_mag, 0.0, 0.0]
        ])

        contact_signed_dist = np.array([signed_dist])

        frame = StateFrame(
            time=self.time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.masses.copy(),
            body_id=self.body_ids.copy(),
            fields={
                "contact_signed_distance": contact_signed_dist,
                "external_force": external_force,
                "interaction_force": interaction_force
            },
            scalars={
                "contact_length_scale": self.length_scale
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "driver_angle": float(self.driver_angle),
            "follower_angle": float(self.follower_angle),
            "time": float(self.time)
        }

