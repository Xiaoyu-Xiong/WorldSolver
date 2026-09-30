"""Reduced physical solver for the Wilberforce pendulum."""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Time-step a coupled axial/torsional elastic-spring model."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        spring, load = cfg["spring"], cfg["load"]
        initial, regime = cfg["initial_conditions"], cfg["coupled_mode_regime"]
        self.n_spring = int(spring["vertex_count"])
        self.n = self.n_spring + 2
        self.n_clamped = int(spring["upper_clamped_vertex_count"])
        self.radius = float(spring["coil_radius"])
        self.top_z = float(spring["equilibrium_top_z"])
        self.bottom_z = float(spring["equilibrium_bottom_z"])
        self.load_offset_z = float(load["attachment_offset_body"][2])
        self.marker_offset_z = 0.08
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        self.arc = np.linspace(0.0, 1.0, self.n_spring)
        self.base_angle = np.linspace(
            0.0, 2.0 * math.pi * float(spring["coil_count"]), self.n_spring
        )
        self.base_z = np.linspace(self.top_z, self.bottom_z, self.n_spring)
        self.weight = np.zeros(self.n_spring)
        free_count = self.n_spring - self.n_clamped
        if free_count > 0:
            self.weight[self.n_clamped:] = np.linspace(1.0 / free_count, 1.0, free_count)

        spring_mass, load_mass = float(spring["total_mass"]), float(load["mass"])
        point_mass = spring_mass / self.n_spring
        self.mass = np.r_[np.full(self.n_spring, point_mass), load_mass, 0.0]
        self.body_id = np.r_[np.zeros(self.n_spring, dtype=np.int64), 1, 2]
        weight_square_sum = float(np.dot(self.weight, self.weight))
        self.axial_mass = load_mass + point_mass * weight_square_sum
        self.torsional_inertia = (
            float(load["principal_inertia"][2])
            + point_mass * self.radius**2 * weight_square_sum
        )

        vertical_period = float(regime["target_vertical_period"])
        torsional_period = float(regime["target_torsional_period"])
        wz, wt = 2.0 * math.pi / vertical_period, 2.0 * math.pi / torsional_period
        self.kz = self.axial_mass * wz**2
        self.kt = self.torsional_inertia * wt**2
        # Choose the symmetric stiffness coupling from the requested beat time.
        desired_split = math.pi / float(regime["expected_first_rotation_peak_time"])
        squared_split = (wz + wt) * desired_split
        bare_split = wz**2 - wt**2
        normalized_coupling = 0.5 * math.sqrt(
            max(0.0, squared_split**2 - bare_split**2)
        )
        self.coupling = normalized_coupling * math.sqrt(
            self.axial_mass * self.torsional_inertia
        )
        self.coupling = min(self.coupling, 0.95 * math.sqrt(self.kz * self.kt))
        damping_ratio = 0.0025
        self.cz = 2.0 * damping_ratio * wz * self.axial_mass
        self.ct = 2.0 * damping_ratio * wt * self.torsional_inertia

        self.initial_y = np.array([
            float(initial["upward_displacement"]),
            float(initial["initial_linear_velocity"][2]),
            0.0,
            float(initial["initial_angular_velocity"][2]),
        ])
        self.y = self.initial_y.copy()
        self.time = 0.0
        self.max_step = min(0.002, float(cfg["sample_dt"]) / 4.0)

        self.inertia_tensor = np.zeros((self.n, 3, 3), dtype=float)
        self.inertia_tensor[self.n_spring] = np.diag(
            np.asarray(load["principal_inertia"], dtype=float)
        )
        total_mass = float(np.sum(self.mass))
        self.reference_momentum = max(total_mass * wz * abs(self.initial_y[0]), 1.0e-8)
        self.reference_angular_momentum = max(
            math.sqrt(self.axial_mass * self.torsional_inertia)
            * 0.5 * (wz + wt) * abs(self.initial_y[0]),
            1.0e-8,
        )
        state0 = self._physical_state(self.initial_y)
        self.initial_momentum, self.initial_angular_momentum = self._momenta(state0)

    def _derivative(self, y: np.ndarray) -> np.ndarray:
        x, xdot, angle, angular_speed = y
        xddot = -(
            self.cz * xdot + self.kz * x + self.coupling * angle
        ) / self.axial_mass
        angular_acceleration = -(
            self.ct * angular_speed + self.kt * angle + self.coupling * x
        ) / self.torsional_inertia
        return np.array([xdot, xddot, angular_speed, angular_acceleration])

    def _rk4_step(self, dt: float) -> None:
        y = self.y
        k1 = self._derivative(y)
        k2 = self._derivative(y + 0.5 * dt * k1)
        k3 = self._derivative(y + 0.5 * dt * k2)
        k4 = self._derivative(y + dt * k3)
        self.y = y + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        self.time += dt

    def _physical_state(self, y: np.ndarray) -> dict[str, np.ndarray]:
        x, xdot, angle, angular_speed = y
        derivative = self._derivative(y)
        xddot, angular_acceleration = derivative[1], derivative[3]
        local_angle = self.base_angle + self.weight * angle
        cos_a, sin_a = np.cos(local_angle), np.sin(local_angle)
        local_omega = self.weight * angular_speed
        local_alpha = self.weight * angular_acceleration

        position = np.zeros((self.n, 3), dtype=float)
        velocity = np.zeros_like(position)
        acceleration = np.zeros_like(position)
        position[:self.n_spring, 0] = self.radius * cos_a
        position[:self.n_spring, 1] = self.radius * sin_a
        position[:self.n_spring, 2] = self.base_z + self.weight * x
        velocity[:self.n_spring, 0] = -self.radius * sin_a * local_omega
        velocity[:self.n_spring, 1] = self.radius * cos_a * local_omega
        velocity[:self.n_spring, 2] = self.weight * xdot
        acceleration[:self.n_spring, 0] = self.radius * (
            -cos_a * local_omega**2 - sin_a * local_alpha
        )
        acceleration[:self.n_spring, 1] = self.radius * (
            -sin_a * local_omega**2 + cos_a * local_alpha
        )
        acceleration[:self.n_spring, 2] = self.weight * xddot

        load_i = self.n_spring
        load_z = self.bottom_z - self.load_offset_z + x
        position[load_i] = (0.0, 0.0, load_z)
        position[load_i + 1] = (0.0, 0.0, load_z + self.marker_offset_z)
        velocity[load_i:, 2] = xdot
        acceleration[load_i:, 2] = xddot
        angular_velocity = np.zeros_like(position)
        angular_acceleration_array = np.zeros_like(position)
        angular_velocity[:self.n_spring, 2] = local_omega
        angular_velocity[load_i:, 2] = angular_speed
        angular_acceleration_array[:self.n_spring, 2] = local_alpha
        angular_acceleration_array[load_i:, 2] = angular_acceleration

        orientation = np.zeros((self.n, 4), dtype=float)
        half_angle = 0.5 * self.weight * angle
        orientation[:self.n_spring, 2] = np.sin(half_angle)
        orientation[:self.n_spring, 3] = np.cos(half_angle)
        orientation[load_i:, 2] = math.sin(0.5 * angle)
        orientation[load_i:, 3] = math.cos(0.5 * angle)
        return {
            "position": position, "velocity": velocity,
            "acceleration": acceleration, "orientation": orientation,
            "angular_velocity": angular_velocity,
            "angular_acceleration": angular_acceleration_array,
        }

    def _momenta(self, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        linear = self.mass[:, None] * state["velocity"]
        momentum = np.sum(linear, axis=0)
        angular = np.sum(np.cross(state["position"], linear), axis=0)
        angular += np.sum(np.einsum(
            "nij,nj->ni", self.inertia_tensor, state["angular_velocity"]
        ), axis=0)
        return momentum, angular

    def _budget_fields(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        momentum, angular_momentum = self._momenta(state)
        ma = self.mass[:, None] * state["acceleration"]
        momentum_rate = np.sum(ma, axis=0)
        angular_rate = np.sum(np.cross(state["position"], ma), axis=0)
        angular_rate += np.sum(np.einsum(
            "nij,nj->ni", self.inertia_tensor, state["angular_acceleration"]
        ), axis=0)
        total_mass = float(np.sum(self.mass))
        center = np.sum(self.mass[:, None] * state["position"], axis=0) / total_mass
        gravity_force = total_mass * self.gravity
        force = np.zeros((1, 2, 3), dtype=float)
        force[0, 0] = gravity_force
        force[0, 1] = momentum_rate - gravity_force
        force_position = np.zeros_like(force)
        force_position[0, 0] = center
        force_position[0, 1] = (0.0, 0.0, self.top_z)
        force_torque = np.sum(np.cross(force_position[0], force[0]), axis=0)
        couple = np.zeros_like(force)
        couple[0, 1] = angular_rate - force_torque
        zeros = np.zeros((1, 3), dtype=float)
        return {
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": force,
            "budget_force_position": force_position,
            "budget_external_couple": couple,
            "budget_cumulative_external_impulse": (momentum-self.initial_momentum)[None, :],
            "budget_cumulative_external_angular_impulse": (
                angular_momentum-self.initial_angular_momentum
            )[None, :],
            "budget_cumulative_inflow_momentum": zeros.copy(),
            "budget_cumulative_outflow_momentum": zeros.copy(),
            "budget_cumulative_inflow_angular_momentum": zeros.copy(),
            "budget_cumulative_outflow_angular_momentum": zeros.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("sample time must be finite and nonnegative")
        if target < self.time - 1.0e-12:
            self.y, self.time = self.initial_y.copy(), 0.0
        while self.time + self.max_step < target - 1.0e-14:
            self._rk4_step(self.max_step)
        remainder = target - self.time
        if remainder > 1.0e-14:
            self._rk4_step(remainder)
        self.time = target
        physical = self._physical_state(self.y)
        frame = StateFrame(
            target, physical["position"], physical["velocity"],
            orientation=physical["orientation"],
            angular_velocity=physical["angular_velocity"],
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "arc_parameter": np.r_[self.arc, 1.0, 1.0],
                "is_upper_clamped": np.r_[
                    np.arange(self.n_spring) < self.n_clamped, 0, 0
                ].astype(float),
                "is_orientation_marker": np.r_[np.zeros(self.n_spring + 1), 1.0],
                "inertia_tensor": self.inertia_tensor.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields=self._budget_fields(physical),
            scalars={
                "equilibrium_upward_release_offset": float(self.initial_y[0]),
                "reference_momentum": self.reference_momentum,
                "reference_angular_momentum": self.reference_angular_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "fixed-step RK4 coupled axial-torsional elastic model",
            "internal_step": self.max_step,
            "axial_stiffness": self.kz,
            "torsional_stiffness": self.kt,
            "translation_rotation_coupling": self.coupling,
        }
