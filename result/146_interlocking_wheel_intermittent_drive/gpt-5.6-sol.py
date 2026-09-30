"""Deterministic rigid-body model of a four-slot intermittent drive.

The driver is a prescribed rotational boundary condition. The follower is a
dynamic rotational degree of freedom and is advanced with semi-implicit Euler
integration. During engagement, a compliant two-sided slot constraint applies
equal and opposite contact forces to the pin and follower. A four-position
spring detent represents the locking action of the Geneva wheel between pin
engagements.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physical reduced-coordinate solver for the fixed-axle mechanism."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # This deterministic mechanism has no stochastic parameters.
        initial = scene.initial_state()
        self._centres = np.asarray(initial.position, dtype=float).copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._gravity = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._length_scale = float(scene.config["contact"]["length_scale"])

        # Geometry-derived reduced parameters. The pin centre is initially
        # 0.65 m to the right of the driver's axle; wheel radius is about 0.65 m.
        self._pin_radius = float(np.linalg.norm(self._centres[2] - self._centres[0]))
        self._follower_radius = 0.65
        self._contact_lever = 0.55
        self._follower_inertia = 0.5 * self._mass[1] * self._follower_radius**2

        self._driver_angle = 0.0
        self._driver_rate = math.pi  # one prescribed revolution every two seconds
        self._follower_angle = 0.0
        self._follower_rate = 0.0
        self._time = 0.0

        # A stiff, damped compliant constraint is stable at the chosen substep.
        self._slot_clearance_angle = 0.003
        self._contact_stiffness = 420.0
        self._contact_damping = 15.0
        self._detent_stiffness = 2.4
        self._bearing_damping = 0.22
        self._max_step = 5.0e-4

        self._last_contact_torque = 0.0
        self._last_target_angle = 0.0
        self._last_target_rate = 0.0
        self._last_engaged = True
        self._substeps = 0

    @staticmethod
    def _quat_y(angle: float) -> np.ndarray:
        """Quaternion (xyzw) for rotation around the wheel/axle y axis."""
        half = 0.5 * angle
        return np.asarray([0.0, math.sin(half), 0.0, math.cos(half)], dtype=float)

    def _cam_constraint(self, driver_angle: float) -> tuple[bool, float, float]:
        """Return engagement state and the instantaneous four-slot constraint.

        Each revolution has a half-turn engagement interval. The cosine cam
        profile has zero velocity at entry and exit, as a rounded rigid pin and
        radial slot do, and advances the follower by one quarter turn.
        """
        revolution = 2.0 * math.pi
        cycle = int(math.floor(driver_angle / revolution))
        phase = driver_angle - cycle * revolution
        quarter_turn = 0.5 * math.pi

        if phase <= math.pi:
            progress = 0.5 * (1.0 - math.cos(phase))
            progress_rate = 0.5 * math.sin(phase) * self._driver_rate
            target = -quarter_turn * (cycle + progress)
            target_rate = -quarter_turn * progress_rate
            return True, target, target_rate

        return False, -quarter_turn * (cycle + 1), 0.0

    def _contact_torque(self, target: float, target_rate: float) -> float:
        """Penalty contact against either wall of the follower's active slot."""
        error = self._follower_angle - target
        abs_error = abs(error)
        if abs_error <= self._slot_clearance_angle:
            return 0.0

        side = 1.0 if error >= 0.0 else -1.0
        penetration = abs_error - self._slot_clearance_angle
        normal_speed = side * (self._follower_rate - target_rate)
        magnitude = max(
            0.0,
            self._contact_stiffness * penetration
            + self._contact_damping * normal_speed,
        )
        return -side * magnitude

    def _step(self, dt: float) -> None:
        # The motor angle is sampled at the midpoint for second-order forcing
        # accuracy, while the dynamic follower uses stable semi-implicit Euler.
        mid_driver = self._driver_angle + 0.5 * self._driver_rate * dt
        engaged, target, target_rate = self._cam_constraint(mid_driver)
        contact_torque = self._contact_torque(target, target_rate) if engaged else 0.0

        # Four-fold periodic detent wells model the locking disk and axle drag.
        # It acts throughout engagement too, but contact torque dominates it.
        detent_torque = -self._detent_stiffness * math.sin(4.0 * self._follower_angle)
        damping_torque = -self._bearing_damping * self._follower_rate
        angular_acceleration = (
            contact_torque + detent_torque + damping_torque
        ) / self._follower_inertia

        self._follower_rate += angular_acceleration * dt
        self._follower_angle += self._follower_rate * dt
        self._driver_angle += self._driver_rate * dt
        self._time += dt

        self._last_contact_torque = contact_torque
        self._last_target_angle = target
        self._last_target_rate = target_rate
        self._last_engaged = engaged
        self._substeps += 1

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        if not math.isfinite(target_time) or target_time < self._time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing sample times")

        remaining = target_time - self._time
        if remaining > 1.0e-12:
            count = max(1, int(math.ceil(remaining / self._max_step)))
            dt = remaining / count
            for _ in range(count):
                self._step(dt)
            # Avoid accumulation causing a later sample to appear out of order.
            self._time = target_time

        position = self._centres.copy()
        velocity = np.zeros((3, 3), dtype=float)
        orientation = np.empty((3, 4), dtype=float)
        angular_velocity = np.zeros((3, 3), dtype=float)

        orientation[0] = self._quat_y(self._driver_angle)
        orientation[1] = self._quat_y(self._follower_angle)
        orientation[2] = orientation[0]
        angular_velocity[0, 1] = self._driver_rate
        angular_velocity[1, 1] = self._follower_rate
        angular_velocity[2, 1] = self._driver_rate

        # The spherical pin is rigidly carried by the driver. Positive y-axis
        # rotation maps its initial +x radius toward -z.
        cosine = math.cos(self._driver_angle)
        sine = math.sin(self._driver_angle)
        radius_vector = np.asarray(
            [self._pin_radius * cosine, 0.0, -self._pin_radius * sine], dtype=float
        )
        position[2] = self._centres[0] + radius_vector
        velocity[2] = np.asarray(
            [
                -self._pin_radius * self._driver_rate * sine,
                0.0,
                -self._pin_radius * self._driver_rate * cosine,
            ],
            dtype=float,
        )

        # Convert the generalized contact torque to a tangential force at the
        # follower slot. Rows are explicit equal-and-opposite interaction pairs.
        follower_contact = np.zeros(3, dtype=float)
        if self._last_engaged and self._contact_lever > 0.0:
            force_magnitude = self._last_contact_torque / self._contact_lever
            slot_angle = self._follower_angle + math.pi
            tangent = np.asarray(
                [-math.sin(slot_angle), 0.0, -math.cos(slot_angle)], dtype=float
            )
            follower_contact = force_magnitude * tangent
        pin_contact = -follower_contact

        # The rigid coupling supplies the pin's centripetal acceleration and
        # balances its gravity and contact load. Its reaction acts on the driver.
        pin_acceleration = -(self._driver_rate**2) * radius_vector
        pin_gravity = self._mass[2] * self._gravity
        coupling_on_pin = self._mass[2] * pin_acceleration - pin_gravity - pin_contact
        interaction_force = np.vstack(
            (follower_contact, pin_contact, coupling_on_pin, -coupling_on_pin)
        )

        if self._last_engaged:
            angular_error = abs(self._follower_angle - self._last_target_angle)
            signed_gap = self._contact_lever * (
                self._slot_clearance_angle - angular_error
            )
        else:
            # Positive separation from the pin to the mouth of the inactive slot.
            phase = self._driver_angle % (2.0 * math.pi)
            angular_separation = min(phase - math.pi, 2.0 * math.pi - phase)
            signed_gap = self._contact_lever * (
                self._slot_clearance_angle + max(0.0, angular_separation)
            )

        frame = StateFrame(
            time=target_time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields={
                "contact_signed_distance": np.asarray([signed_gap], dtype=float),
                "external_force": self._mass[:, None] * self._gravity[None, :],
                "interaction_force": interaction_force,
            },
            scalars={"contact_length_scale": self._length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_euler_reduced_rigid_contact",
            "substeps": int(self._substeps),
            "driver_angle": float(self._driver_angle),
            "follower_angle": float(self._follower_angle),
            "follower_angular_velocity": float(self._follower_rate),
        }
