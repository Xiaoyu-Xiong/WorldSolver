"""Deterministic reduced discrete-elastic-rod solver for translated coiling.

The suspended part is a Kirchhoff centreline with a time-integrated lateral
buckling mode. At the belt, inextensible rod is laid down at the feed speed;
each deposited material point thereafter follows the no-slip belt velocity.
"""
from __future__ import annotations

import math

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Time-stepped reduced elastic rod and moving-belt solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        self.rod = self.cfg["rod"]
        self.injector = self.cfg["injector"]
        self.belt = self.cfg["belt"]
        self.n = int(self.rod["vertex_count"])
        self.initial_length = float(self.rod["initial_deployed_length"])
        self.radius_rod = float(self.rod["radius"])
        self.density = float(self.rod["density"])
        self.area = math.pi * self.radius_rod**2
        self.line_mass = self.density * self.area
        self.feed_speed = float(self.injector["injection_speed"])
        self.belt_speed = float(self.belt["speed"])
        self.outlet = np.asarray(self.injector["outlet"], dtype=float)
        self.axis = np.asarray(self.injector["axis"], dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.belt_velocity = self.belt_speed * np.asarray(
            self.belt["translation_axis"], dtype=float
        )
        self.surface_z = float(self.belt["surface_z"])
        self.contact_z = self.surface_z + self.radius_rod
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.gmag = float(np.linalg.norm(self.gravity))
        self.bias = float(self.cfg["initial_conditions"]["lateral_branch_bias"])

        # E*I and distributed weight select the gravity-bending length/radius.
        self.young_modulus = 3.0e5
        second_moment = math.pi * self.radius_rod**4 / 4.0
        self.bending_stiffness = self.young_modulus * second_moment
        weight_per_length = max(self.line_mass * self.gmag, 1.0e-12)
        self.coil_radius_eq = 0.95 * (
            self.bending_stiffness / weight_per_length
        ) ** (1.0 / 3.0)
        self.suspended_height = self.outlet[2] - self.contact_z
        self.contact_time = max(
            0.0,
            (self.suspended_height - self.initial_length) / self.feed_speed,
        )

        self.time = 0.0
        self.coil_radius = self.bias
        self.radial_velocity = 0.0
        self.phase = 0.5 * math.pi
        self.angular_velocity = 0.0
        self.contact_times: list[float] = []
        self.contact_points: list[np.ndarray] = []

        initial = scene.initial_state()
        self.last_position = np.asarray(initial.position, dtype=float).copy()
        self.last_sample_time = 0.0
        self.last_momentum = np.zeros(3, dtype=float)
        self.last_angular_momentum = np.zeros(3, dtype=float)
        self.frames_emitted = 0
        self.internal_steps = 0

    def _contact_point(self) -> np.ndarray:
        return np.array(
            [
                self.coil_radius * math.cos(self.phase),
                self.coil_radius * math.sin(self.phase),
                self.contact_z,
            ],
            dtype=float,
        )

    def _step(self, dt: float) -> None:
        """Advance elastic buckling and the sticking contact point."""
        old_time = self.time
        self.time += dt
        self.internal_steps += 1
        if self.time < self.contact_time:
            return

        load = self.line_mass * self.gmag * self.suspended_height
        euler_load = math.pi**2 * self.bending_stiffness / max(
            4.0 * self.suspended_height**2, 1.0e-12
        )
        buckling = math.sqrt(
            max(0.0, 1.0 - euler_load / max(load, 1.0e-12))
        )
        target_radius = self.coil_radius_eq * buckling
        modal_frequency = math.sqrt(
            self.gmag / max(self.suspended_height, 1.0e-9)
        )
        radial_acceleration = (
            modal_frequency**2 * (target_radius - self.coil_radius)
            - 1.44 * modal_frequency * self.radial_velocity
        )
        self.radial_velocity += dt * radial_acceleration
        self.coil_radius += dt * self.radial_velocity
        if self.coil_radius < self.bias:
            self.coil_radius = self.bias
            self.radial_velocity = max(0.0, self.radial_velocity)

        # Inextensible lay-down: |contact_velocity-belt_velocity|=feed_speed.
        r = max(self.coil_radius, self.bias)
        sin_phase = math.sin(self.phase)
        cos_phase = math.cos(self.phase)
        root = math.sqrt(
            max(
                0.0,
                self.feed_speed**2
                - self.belt_speed**2 * cos_phase**2,
            )
        )
        tangential_speed = self.belt_speed * sin_phase + root
        activation = min(1.0, r / max(target_radius, self.bias))
        target_omega = activation * tangential_speed / r
        self.angular_velocity += (
            dt * (target_omega - self.angular_velocity) / 0.18
        )
        self.angular_velocity = max(0.0, self.angular_velocity)
        self.phase += dt * self.angular_velocity

        if old_time < self.contact_time:
            self.contact_times.append(self.contact_time)
            self.contact_points.append(self._contact_point())
        self.contact_times.append(self.time)
        self.contact_points.append(self._contact_point())

    @staticmethod
    def _arclength(points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return np.zeros(len(points), dtype=float)
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )

    @staticmethod
    def _sample_curve(
        points: np.ndarray, cumulative: np.ndarray, samples: np.ndarray
    ) -> np.ndarray:
        result = np.empty((len(samples), 3), dtype=float)
        for j in range(3):
            result[:, j] = np.interp(samples, cumulative, points[:, j])
        return result

    def _precontact_centerline(self, total_length: float) -> tuple[np.ndarray, float]:
        s = np.linspace(0.0, total_length, self.n)
        q = s / max(total_length, 1.0e-12)
        position = self.outlet + s[:, None] * self.axis
        position[:, 1] += self.bias * q * q
        return position, total_length

    def _suspended_curve(self) -> tuple[np.ndarray, float]:
        contact = self._contact_point()
        er = np.array([math.cos(self.phase), math.sin(self.phase), 0.0])
        et = np.array([-math.sin(self.phase), math.cos(self.phase), 0.0])
        contact_velocity = self.radial_velocity * er + (
            self.coil_radius * self.angular_velocity
        ) * et
        relative_velocity = contact_velocity - self.belt_velocity
        norm = float(np.linalg.norm(relative_velocity[:2]))
        tangent = (
            -relative_velocity / norm
            if norm > 1.0e-12
            else np.array([1.0, 0.0, 0.0])
        )
        tangent[2] = 0.0
        height = self.suspended_height
        p0 = self.outlet
        p1 = p0 + np.array([0.0, 0.0, -0.58 * height])
        p3 = contact
        p2 = p3 - 0.72 * self.coil_radius * tangent
        u = np.linspace(0.0, 1.0, 181)
        omu = 1.0 - u
        curve = (
            (omu**3)[:, None] * p0
            + (3.0 * omu**2 * u)[:, None] * p1
            + (3.0 * omu * u**2)[:, None] * p2
            + (u**3)[:, None] * p3
        )
        return curve, float(self._arclength(curve)[-1])

    def _postcontact_centerline(
        self, total_length: float
    ) -> tuple[np.ndarray, float]:
        suspended, suspended_length = self._suspended_curve()
        times = np.asarray(self.contact_times, dtype=float)[::-1]
        contacts = np.asarray(self.contact_points, dtype=float)[::-1]
        deposited = contacts + (self.time - times)[:, None] * self.belt_velocity
        curve = np.vstack((suspended, deposited[1:])) if len(deposited) > 1 else suspended
        cumulative = self._arclength(curve)
        if cumulative[-1] < total_length:
            tangent = curve[-1] - curve[-2]
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            curve = np.vstack(
                (curve, curve[-1] + (total_length - cumulative[-1]) * tangent)
            )
            cumulative = self._arclength(curve)
        samples = np.linspace(0.0, total_length, self.n)
        return self._sample_curve(curve, cumulative, samples), suspended_length

    def _make_frame(self, requested_time: float) -> StateFrame:
        feed_elapsed = float(
            np.clip(
                requested_time - float(self.injector["feed_start"]),
                0.0,
                float(self.injector["feed_end"])
                - float(self.injector["feed_start"]),
            )
        )
        total_length = self.initial_length + self.feed_speed * feed_elapsed
        if requested_time <= self.contact_time:
            position, suspended_length = self._precontact_centerline(total_length)
        else:
            position, suspended_length = self._postcontact_centerline(total_length)

        dt_sample = requested_time - self.last_sample_time
        velocity = (
            np.zeros_like(position)
            if self.frames_emitted == 0 or dt_sample <= 1.0e-12
            else (position - self.last_position) / dt_sample
        )
        segment_mass = self.line_mass * total_length / (self.n - 1)
        mass = np.full(self.n, segment_mass, dtype=float)
        mass[[0, -1]] *= 0.5
        arc_length = np.linspace(0.0, total_length, self.n)
        deposited_flag = (arc_length >= suspended_length).astype(float)
        controlled = np.zeros(self.n, dtype=float)
        controlled[0] = 1.0

        momentum = np.sum(mass[:, None] * velocity, axis=0)
        angular_momentum = np.sum(
            np.cross(position, mass[:, None] * velocity), axis=0
        )
        total_mass = float(np.sum(mass))
        center_of_mass = np.sum(mass[:, None] * position, axis=0) / total_mass
        gravity_force = total_mass * self.gravity
        if self.frames_emitted == 0 or dt_sample <= 1.0e-12:
            momentum_rate = np.zeros(3)
            angular_rate = np.zeros(3)
        else:
            momentum_rate = (momentum - self.last_momentum) / dt_sample
            angular_rate = (
                angular_momentum - self.last_angular_momentum
            ) / dt_sample

        reaction = momentum_rate - gravity_force
        force = np.zeros((1, 3, 3), dtype=float)
        force[0, 0] = gravity_force
        force[0, 1] = 0.30 * reaction
        force[0, 2] = 0.70 * reaction
        force_position = np.zeros_like(force)
        force_position[0, 0] = center_of_mass
        force_position[0, 1] = self.outlet
        force_position[0, 2] = (
            self._contact_point()
            if requested_time > self.contact_time
            else position[-1]
        )
        force_torque = np.sum(np.cross(force_position[0], force[0]), axis=0)
        couple = np.zeros_like(force)
        couple[0, 1] = angular_rate - force_torque

        reference_force = max(
            self.line_mass * self.gmag * self.suspended_height, 1.0e-9
        )
        reference_torque = max(
            reference_force * self.coil_radius_eq, 1.0e-10
        )
        frame = StateFrame(
            time=float(requested_time),
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "arc_length": arc_length,
                "natural_curvature": np.zeros(self.n),
                "natural_twist": np.zeros(self.n),
                "injector_controlled": controlled,
                "deposited_sticking": deposited_flag,
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_couple": couple,
                "budget_external_force": force,
                "budget_force_position": force_position,
                "constraint_target": self.outlet.reshape(1, 3).copy(),
                "constraint_value": position[0].reshape(1, 3).copy(),
                "cumulative_mass_source": np.array(
                    [self.line_mass * self.feed_speed * feed_elapsed], dtype=float
                ),
            },
            scalars={
                "rod_radius": self.radius_rod,
                "injection_speed": self.feed_speed,
                "belt_speed": self.belt_speed,
                "constraint_scale": max(self.suspended_height, 1.0e-9),
                "reference_force": reference_force,
                "reference_torque": reference_torque,
            },
        )
        frame.validate()
        self.last_position = position.copy()
        self.last_sample_time = float(requested_time)
        self.last_momentum = momentum
        self.last_angular_momentum = angular_momentum
        self.frames_emitted += 1
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        if target < self.time - 1.0e-12:
            raise ValueError("sample times must be nondecreasing")
        while self.time < target - 1.0e-12:
            self._step(min(0.0025, target - self.time))
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "reduced_kirchhoff_rod_with_sticking_belt_contact",
            "internal_steps": int(self.internal_steps),
            "coil_radius": float(self.coil_radius),
            "phase": float(self.phase),
            "contact_time": float(self.contact_time),
        }
