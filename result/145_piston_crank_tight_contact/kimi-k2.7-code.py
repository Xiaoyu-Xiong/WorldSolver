"""Planar piston-crank tight-contact mechanism solver.

Implements a constrained rigid-body slider-crank driven by a simple motor.  The
crank rotates about a fixed axle, the connecting rod is pin-jointed to the
crank and piston, and the piston is constrained to the horizontal guide axis.
The time integration is a stable semi-implicit Euler step on the single
crank angle, with all other body states obtained analytically from the
closed-form slider-crank kinematics.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_y(angle: float) -> np.ndarray:
    return np.array([0.0, math.sin(angle * 0.5), 0.0, math.cos(angle * 0.5)], dtype=float)


class BenchmarkSolver:
    def __init__(self):
        # Scene / configuration
        self.scene: object | None = None
        self.seed: int | None = None
        self.gravity: np.ndarray | None = None
        self.dt_sample: float = 0.02
        self.substeps: int = 20
        self.dt: float = 0.0
        self.contact_length_scale: float = 0.02

        # Bodies
        self.bodies: list[str] = []
        self.mass: np.ndarray | None = None

        # Mechanism geometry (units: meters, up-axis z)
        self.axle = np.array([-0.8, 0.0, 1.0])
        self.r_crank = 0.31
        self.l_rod = 1.24
        self.dp = 0.12
        self.piston_z = 1.0

        # Crank dynamics
        self.theta: float = 0.0
        self.omega: float = 0.0
        self.I_crank = 0.05
        self.torque_const = 0.05
        self.damping = 0.01

        # Cached state arrays
        self.positions: np.ndarray | None = None
        self.orientations: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.angular_velocities: np.ndarray | None = None

        self.time: float = 0.0

    # -------------------------------------------------------------------------
    # Public contract
    # -------------------------------------------------------------------------

    def initialize(self, scene: object, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt_sample = float(cfg["sample_dt"])
        self.dt = self.dt_sample / self.substeps
        self.contact_length_scale = float(cfg["contact"]["length_scale"])

        self.bodies = [b["id"] for b in cfg["bodies"]]
        self.mass = np.array([b["mass"] for b in cfg["bodies"]], dtype=float)

        # Start from rest at the assembled slider-crank configuration (theta=0).
        self.theta = 0.0
        self.omega = 0.0
        self.time = 0.0

        # Allocate and fill state arrays.
        self.positions = np.zeros((len(self.bodies), 3), dtype=float)
        self.orientations = np.zeros((len(self.bodies), 4), dtype=float)
        self.velocities = np.zeros((len(self.bodies), 3), dtype=float)
        self.angular_velocities = np.zeros((len(self.bodies), 3), dtype=float)

        self._update_state_arrays()

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("Cannot advance backwards in time")

        # Advance by a deterministic number of substeps.
        if target > self.time:
            n = int(round((target - self.time) / self.dt))
            for _ in range(n):
                self._substep()
                self.time += self.dt
                self._update_state_arrays()

        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "bodies": len(self.bodies),
            "time": self.time,
            "theta": self.theta,
            "omega": self.omega,
        }

    # -------------------------------------------------------------------------
    # Internal physics
    # -------------------------------------------------------------------------

    def _substep(self) -> None:
        """Semi-implicit Euler step on the crank angle."""
        # Motor torque ramps linearly to a constant value.
        torque = min(self.torque_const, self.torque_const * self.time)
        alpha = (torque - self.damping * self.omega) / self.I_crank
        self.omega += alpha * self.dt
        self.theta += self.omega * self.dt

    def _kinematics(self):
        """Return the analytic slider-crank quantities for the current theta."""
        theta = self.theta
        omega = self.omega
        r = self.r_crank
        l = self.l_rod
        dp = self.dp

        c = math.cos(theta)
        s = math.sin(theta)

        crank_pin = self.axle + np.array([r * c, 0.0, r * s])
        # Inline slider-crank: piston pin lies in the x-z plane at z=1.
        a2 = (r * s) ** 2
        a2 = min(a2, l * l - 1e-12)
        A = math.sqrt(l * l - a2)

        piston_pin_x = self.axle[0] + r * c + A
        piston_pin = np.array([piston_pin_x, 0.0, self.piston_z])

        # Rod center and orientation.
        rod_center = 0.5 * (crank_pin + piston_pin)
        rod_angle = math.atan2(-r * s, A)

        # Velocities (derivatives of the slider-crack relations).
        # Piston pin horizontal velocity.
        x_dot_pin = (-r * s - (r * r * s * c) / A) * omega
        piston_vel = np.array([x_dot_pin, 0.0, 0.0])

        crank_pin_vel = np.array([-r * s * omega, 0.0, r * c * omega])
        rod_center_vel = 0.5 * (crank_pin_vel + piston_vel)

        # Angular velocities about the y-axis.
        crank_ang = np.array([0.0, omega, 0.0])
        rod_ang = np.array([0.0, -(r * c / A) * omega, 0.0])
        piston_ang = np.zeros(3, dtype=float)

        return {
            "crank_pin": crank_pin,
            "piston_pin": piston_pin,
            "rod_center": rod_center,
            "rod_angle": rod_angle,
            "piston_vel": piston_vel,
            "rod_center_vel": rod_center_vel,
            "crank_ang": crank_ang,
            "rod_ang": rod_ang,
            "piston_ang": piston_ang,
            "A": A,
        }

    def _update_state_arrays(self) -> None:
        k = self._kinematics()

        self.positions[0] = self.axle
        self.positions[1] = k["rod_center"]
        self.positions[2] = k["piston_pin"] + np.array([self.dp, 0.0, 0.0])

        self.orientations[0] = _quat_y(self.theta)
        self.orientations[1] = _quat_y(k["rod_angle"])
        self.orientations[2] = np.array([0.0, 0.0, 0.0, 1.0])

        self.velocities[0] = np.zeros(3, dtype=float)
        self.velocities[1] = k["rod_center_vel"]
        self.velocities[2] = k["piston_vel"]

        self.angular_velocities[0] = k["crank_ang"]
        self.angular_velocities[1] = k["rod_ang"]
        self.angular_velocities[2] = k["piston_ang"]

    def _build_frame(self, time: float) -> StateFrame:
        # External force: gravity on each body (motor torque is not a force at
        # the center of mass and is omitted from this per-sample vector).
        external_force = self.mass[:, None] * self.gravity[None, :]

        # Interaction force pairs: [crank-rod, rod-piston, piston-lower, piston-upper].
        # The lower guide carries the piston weight to keep it on the z=1 axis.
        interaction = np.zeros((4, 3), dtype=float)
        interaction[2, 2] = -self.mass[2] * self.gravity[2]

        # Signed geometric gap to the guide walls.  Positive = separated,
        # negative = penetration.
        piston_half_z = 0.31
        lower_wall = 0.63
        upper_wall = 1.37
        center_z = self.positions[2, 2]
        signed = np.array(
            [center_z - piston_half_z - lower_wall,
             upper_wall - center_z - piston_half_z],
            dtype=float,
        )

        return StateFrame(
            time=time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientations.copy(),
            angular_velocity=self.angular_velocities.copy(),
            mass=self.mass.copy(),
            body_id=np.arange(len(self.bodies), dtype=np.int32),
            attributes={"release_time": np.zeros(len(self.bodies), dtype=float)},
            fields={
                "external_force": external_force,
                "interaction_force": interaction,
                "contact_signed_distance": signed,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
