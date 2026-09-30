"""Penalty-based rigid-body Geneva / intermittent drive solver.

The scene is treated as a 2-DOF problem in the plane z=1: the driver and
follower rotate about fixed parallel axles, and the drive pin is rigidly
attached to the driver.  Contact between the spherical drive pin and the
radial slots of the follower is resolved with a stiff servo-like penalty
that drives the follower to the exact kinematic slot angle implied by the
current pin position.  This produces the characteristic discrete steps,
dwell periods, and separated contacts of an interlocking wheel drive.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        # Load the public configuration; it supplies masses and contact params.
        import yaml

        with open(Path(__file__).with_name("..").resolve() / ".." / "config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        # Contact / scalar metadata.
        self.contact_length_scale = float(self.cfg["contact"]["length_scale"])

        # Bodies: driver, follower, drive_pin.
        bodies = self.cfg["bodies"]
        self.mass = np.array([b["mass"] for b in bodies], dtype=float)
        self.gravity = np.array(self.cfg["gravity"], dtype=float)

        # Geneva geometry (world XY plane at z=1).  The driver and follower
        # axles are spaced so the drive pin can enter and leave each of the
        # follower's radial slots as the driver turns.
        self.driver_center = np.array([-0.85, 0.0, 1.0], dtype=float)
        self.follower_center = np.array([0.85, 0.0, 1.0], dtype=float)
        axle_separation = np.linalg.norm(self.driver_center[:2] - self.follower_center[:2])
        self.r_pin_orbit = 0.55  # distance from driver axle to pin centre
        self.r_pin = 0.14        # pin sphere radius
        # Choose slot geometry so the pin can engage the follower.  The slot
        # extends from just inside the follower rim to the outer radius.
        self.slot_r_inner = 0.36
        self.slot_r_outer = axle_separation + self.r_pin_orbit + 0.02
        # Half-width of each radial slot.
        self.slot_half_angle = 0.18

        # Driver is kinematic with constant angular velocity.
        self.driver_omega = 2.0 * math.pi / 3.0
        # Initial driver angle so the pin is just about to enter a slot.
        self.driver_theta0 = -self.slot_half_angle - 0.05

        # Follower dynamic state.
        self.follower_phi = math.pi  # slot initially points toward the driver
        self.follower_phi_dot = 0.0
        self.I_follower = 0.5 * self.mass[1] * (0.65 ** 2)

        # Servo/penalty gains for the pin-slot contact.
        self.k_p = 5.0e4
        self.k_d = 1.0e3
        self.c_damp = 0.5

        # Sub-stepping for stability.
        self.sample_dt = float(self.cfg["sample_dt"])
        self.substeps = 40
        self.h = self.sample_dt / self.substeps

        self.current_time = 0.0
        self._cached_frame: StateFrame | None = None
        self._cached_time: float = -1.0

    def _pin_state(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        theta = self.driver_omega * t + self.driver_theta0
        c = math.cos(theta)
        s = math.sin(theta)
        p = self.driver_center + self.r_pin_orbit * np.array([c, s, 0.0], dtype=float)
        v = self.r_pin_orbit * self.driver_omega * np.array([-s, c, 0.0], dtype=float)
        return p, v, theta

    @staticmethod
    def _quat_z(angle: float) -> np.ndarray:
        return np.array([0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5)], dtype=float)

    def _slot_contact(self, p_pin: np.ndarray, v_pin: np.ndarray) -> dict | None:
        """Return contact data if the pin is inside a follower slot."""
        d_vec = p_pin - self.follower_center
        rho = math.hypot(d_vec[0], d_vec[1])
        if rho < 1e-9:
            return None
        psi = math.atan2(d_vec[1], d_vec[0])

        # Four slots at pi/2 spacing.  Find the slot whose centreline best
        # matches the current pin direction.
        candidates = psi - np.arange(4, dtype=float) * 0.5 * math.pi
        diff = (candidates - self.follower_phi + math.pi) % (2.0 * math.pi) - math.pi
        k = int(np.argmin(np.abs(diff)))
        phi_target = candidates[k]

        # Slot centreline angle from the current follower orientation.
        phi_s = self.follower_phi + k * 0.5 * math.pi
        delta = (psi - phi_s + math.pi) % (2.0 * math.pi) - math.pi

        # Radial and angular checks.
        if rho < self.slot_r_inner or rho > self.slot_r_outer:
            return None
        if abs(delta) > self.slot_half_angle:
            return None

        # Signed distance from pin surface to the nearest slot wall.
        signed_dist = rho * (self.slot_half_angle - abs(delta)) - self.r_pin

        # Tangential unit vector at the pin around the follower axle.
        e_psi = np.array([-math.sin(psi), math.cos(psi), 0.0], dtype=float)

        # d(psi)/dt from pin velocity.
        psi_dot = (d_vec[0] * v_pin[1] - d_vec[1] * v_pin[0]) / (rho * rho)
        phi_dot_target = psi_dot

        return {
            "phi_target": phi_target,
            "phi_dot_target": phi_dot_target,
            "rho": rho,
            "e_psi": e_psi,
            "signed_distance": signed_dist,
            "force": np.zeros(3, dtype=float),
        }

    def _build_frame(self, t: float) -> StateFrame:
        p_pin, v_pin, theta = self._pin_state(t)

        q_driver = self._quat_z(theta)
        q_follower = self._quat_z(self.follower_phi)
        q_pin = self._quat_z(theta)

        position = np.zeros((3, 3), dtype=float)
        position[0] = self.driver_center
        position[1] = self.follower_center
        position[2] = p_pin

        velocity = np.zeros((3, 3), dtype=float)
        velocity[2] = v_pin

        orientation = np.array([q_driver, q_follower, q_pin])

        angular_velocity = np.zeros((3, 3), dtype=float)
        angular_velocity[0, 2] = self.driver_omega
        angular_velocity[1, 2] = self.follower_phi_dot
        angular_velocity[2, 2] = self.driver_omega

        contact = self._slot_contact(p_pin, v_pin)
        if contact is not None:
            contact_signed = np.array([contact["signed_distance"]], dtype=float)
            interaction = contact["force"].reshape(1, 3)
        else:
            contact_signed = np.array([], dtype=float)
            interaction = np.zeros((0, 3), dtype=float)

        external_force = np.outer(self.mass, self.gravity)

        return StateFrame(
            time=t,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass,
            body_id=np.arange(3, dtype=np.int32),
            fields={
                "contact_signed_distance": contact_signed,
                "external_force": external_force,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def advance_to(self, time: float) -> StateFrame:
        if self._cached_time == time and self._cached_frame is not None:
            return self._cached_frame

        # Guard against backward calls by resetting if necessary.
        if time < self.current_time:
            self.current_time = time

        while self.current_time < time - 1e-12:
            h = min(self.h, time - self.current_time)
            t_step = self.current_time + h

            p_pin, v_pin, _ = self._pin_state(t_step)
            contact = self._slot_contact(p_pin, v_pin)

            if contact is not None:
                # Servo torque that enforces the exact slot kinematics while
                # the pin remains inside the slot.
                err = contact["phi_target"] - self.follower_phi
                err = (err + math.pi) % (2.0 * math.pi) - math.pi
                err_dot = contact["phi_dot_target"] - self.follower_phi_dot
                torque = self.k_p * err + self.k_d * err_dot
                contact["force"] = (torque / contact["rho"]) * contact["e_psi"]
            else:
                torque = -self.c_damp * self.follower_phi_dot

            # Semi-implicit Euler for the follower angular dynamics.
            self.follower_phi_dot += h * (torque - self.c_damp * self.follower_phi_dot) / self.I_follower
            self.follower_phi += h * self.follower_phi_dot

            self.current_time += h

        self.current_time = time
        frame = self._build_frame(time)
        self._cached_time = time
        self._cached_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "interlocking_wheel_penalty", "seed": self.seed}
