"""Deterministic constrained dynamics for the piston--crank scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Constrained planar slider-crank with a finite-speed prescribed motor."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        cfg = scene.config
        initial = scene.initial_state()
        self._time = 0.0
        self._max_step = 0.001
        self._gravity = np.asarray(cfg["gravity"], dtype=float)
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._length_scale = float(cfg["contact"]["length_scale"])
        self._friction = float(cfg["contact"]["friction"])
        self._restitution = float(cfg["contact"]["restitution"])

        self._pivot_x = float(initial.position[0, 0])
        self._axis_z = float(initial.position[0, 2])
        self._crank_radius = 0.32
        self._rod_left = -0.48
        self._rod_right = 0.76
        self._rod_length = self._rod_right - self._rod_left
        self._piston_joint_x = -0.39
        self._z_lower = 0.55 + 0.08 + 0.31
        self._z_upper = 1.45 - 0.08 - 0.31

        self._theta = 0.0
        self._omega = 0.0
        self._motor_speed = 2.0 * math.pi
        self._motor_gain = 0.80
        self._crank_inertia = 0.080
        self._piston_z = self._axis_z
        self._piston_vz = 0.0
        self._lower_contact = False
        self._upper_contact = False
        self._position, self._velocity, self._orientation, self._angular_velocity = self._kinematics()
        self._linear_acceleration = np.zeros((3, 3), dtype=float)
        self._angular_acceleration = np.zeros(3, dtype=float)
        self._interaction = np.zeros((8, 3), dtype=float)
        self._steps = 0

    @staticmethod
    def _y_quaternion(angle: float) -> np.ndarray:
        half = 0.5 * angle
        return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=float)

    def _kinematics(self):
        r = self._crank_radius
        st, ct = math.sin(self._theta), math.cos(self._theta)
        pin = np.array([self._pivot_x + r * ct, 0.0, self._axis_z + r * st])
        pin_velocity = np.array([-r * st * self._omega, 0.0, r * ct * self._omega])
        dz = self._piston_z - pin[2]
        horizontal = math.sqrt(max(self._rod_length**2 - dz * dz, 1.0e-14))
        piston_pin_x = pin[0] + horizontal
        piston_x = piston_pin_x - self._piston_joint_x
        relative_vz = self._piston_vz - pin_velocity[2]
        horizontal_velocity = -(dz * relative_vz) / horizontal
        piston_vx = pin_velocity[0] + horizontal_velocity
        ux, uz = horizontal / self._rod_length, dz / self._rod_length
        ux_dot, uz_dot = horizontal_velocity / self._rod_length, relative_vz / self._rod_length
        rod_position = pin - self._rod_left * np.array([ux, 0.0, uz])
        rod_velocity = pin_velocity - self._rod_left * np.array([ux_dot, 0.0, uz_dot])
        phi = math.atan2(dz, horizontal)
        phi_rate = (horizontal * relative_vz - dz * horizontal_velocity) / self._rod_length**2
        position = np.array([[self._pivot_x, 0.0, self._axis_z], rod_position,
                             [piston_x, 0.0, self._piston_z]], dtype=float)
        velocity = np.array([[0.0, 0.0, 0.0], rod_velocity,
                             [piston_vx, 0.0, self._piston_vz]], dtype=float)
        orientation = np.array([self._y_quaternion(-self._theta), self._y_quaternion(-phi),
                                 [0.0, 0.0, 0.0, 1.0]], dtype=float)
        angular_velocity = np.array([[0.0, -self._omega, 0.0], [0.0, -phi_rate, 0.0],
                                     [0.0, 0.0, 0.0]], dtype=float)
        return position, velocity, orientation, angular_velocity

    def _step(self, dt: float) -> None:
        old_velocity = self._velocity.copy()
        old_angular_velocity = self._angular_velocity.copy()
        motor_torque = np.clip(self._motor_gain * (self._motor_speed - self._omega), -5.0, 5.0)
        self._omega += (motor_torque / self._crank_inertia) * dt
        self._theta += self._omega * dt
        if abs(self._theta) > 8.0 * math.pi:
            self._theta = math.fmod(self._theta, 2.0 * math.pi)

        self._piston_vz += self._gravity[2] * dt
        self._piston_z += self._piston_vz * dt
        self._lower_contact = self._upper_contact = False
        if self._piston_z < self._z_lower:
            self._piston_z = self._z_lower
            if self._piston_vz < 0.0:
                self._piston_vz = -self._restitution * self._piston_vz
            if abs(self._piston_vz) < 0.03:
                self._piston_vz = 0.0
            self._lower_contact = True
        elif self._piston_z > self._z_upper:
            self._piston_z = self._z_upper
            if self._piston_vz > 0.0:
                self._piston_vz = -self._restitution * self._piston_vz
            if abs(self._piston_vz) < 0.03:
                self._piston_vz = 0.0
            self._upper_contact = True

        self._position, self._velocity, self._orientation, self._angular_velocity = self._kinematics()
        self._linear_acceleration = (self._velocity - old_velocity) / dt
        self._angular_acceleration = (self._angular_velocity[:, 1] - old_angular_velocity[:, 1]) / dt
        self._compute_reactions()
        self._time += dt
        self._steps += 1

    def _compute_reactions(self) -> None:
        mr, mp = self._mass[1], self._mass[2]
        ar, ap = self._linear_acceleration[1], self._linear_acceleration[2]
        pin = np.array([self._pivot_x + self._crank_radius * math.cos(self._theta), 0.0,
                        self._axis_z + self._crank_radius * math.sin(self._theta)])
        dx = self._position[2, 0] + self._piston_joint_x - pin[0]
        dz = self._piston_z - pin[2]
        norm = math.hypot(dx, dz)
        u = np.array([dx / norm, 0.0, dz / norm])
        r_left, r_right = self._rod_left * u, self._rod_right * u
        rod_inertia = mr * self._rod_length**2 / 12.0
        base_left = mr * ar - mr * self._gravity
        normal_guess = (mp * max(0.0, -self._gravity[2]) if self._lower_contact else 0.0)
        if self._upper_contact:
            normal_guess = -mp * max(0.0, -self._gravity[2])
        piston_force = np.zeros(3)
        guide_force = np.zeros(3)
        for _ in range(2):
            guide_force[0] = -self._friction * abs(normal_guess) * math.tanh(self._velocity[2, 0] / 0.03)
            piston_force[0] = mp * ap[0] - guide_force[0]
            constant_torque = (r_left[2] * base_left[0] - r_left[0] * base_left[2] +
                               (r_left[2] - r_right[2]) * piston_force[0])
            coefficient = r_right[0] - r_left[0]
            if abs(coefficient) < 1.0e-8:
                coefficient = math.copysign(1.0e-8, coefficient or 1.0)
            piston_force[2] = (rod_inertia * self._angular_acceleration[1] - constant_torque) / coefficient
            requested_normal = mp * ap[2] - mp * self._gravity[2] - piston_force[2]
            if self._lower_contact:
                normal_guess = max(0.0, requested_normal)
            elif self._upper_contact:
                normal_guess = min(0.0, requested_normal)
            else:
                normal_guess = 0.0
        guide_force[2] = normal_guess
        guide_force[0] = -self._friction * abs(normal_guess) * math.tanh(self._velocity[2, 0] / 0.03)
        piston_force[0] = mp * ap[0] - guide_force[0]
        left_force = base_left + piston_force
        crank_gravity = self._mass[0] * self._gravity
        force_on_crank = -left_force
        axle_force = -(crank_gravity + force_on_crank)
        self._interaction = np.array([left_force, -left_force, -piston_force, piston_force,
                                       guide_force, -guide_force, axle_force, -axle_force], dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self._time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing time")
        while self._time < target - 1.0e-12:
            self._step(min(self._max_step, target - self._time))
        self._time = target
        lower_gap, upper_gap = self._piston_z - self._z_lower, self._z_upper - self._piston_z
        external_force = self._mass[:, None] * self._gravity[None, :]
        frame = StateFrame(target, self._position.copy(), self._velocity.copy(),
                           orientation=self._orientation.copy(), angular_velocity=self._angular_velocity.copy(),
                           mass=self._mass.copy(), body_id=self._body_id.copy(),
                           fields={"contact_signed_distance": np.array([lower_gap, upper_gap]),
                                   "external_force": external_force,
                                   "interaction_force": self._interaction.copy()},
                           scalars={"contact_length_scale": self._length_scale})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "semi_implicit_constrained_slider_crank", "internal_steps": int(self._steps),
                "motor_target_rad_s": float(self._motor_speed),
                "contact_model": "unilateral_Newton_impact_with_Coulomb_friction"}
