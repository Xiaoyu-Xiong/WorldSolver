"""Deterministic torsional-rod solver for the torque-transducer scene.

The rod is a lumped discretization of the one-dimensional torsion equation.
Each rod vertex carries polar inertia and neighboring material frames exchange
equal and opposite elastic/dissipative couples. The two end frames are joined
to independently integrated rigid-body rotations by the same physical joints.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-implicit time integrator for rod and rigid-body axial rotation."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model has no stochastic terms.
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()

        self.time = 0.0
        self.base_position = np.asarray(initial.position, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.component_id = np.asarray(
            initial.attributes["component_id"], dtype=float
        ).copy()
        self.arc_length = np.asarray(
            initial.attributes["arc_length"], dtype=float
        ).copy()
        self.is_driven = np.asarray(
            initial.attributes["is_driven"], dtype=float
        ).copy()

        self.rod_count = int(self.config["rod"]["vertex_count"])
        self.driven_index = self.rod_count
        self.receiver_index = self.rod_count + 1
        self.driven_mark_index = self.rod_count + 2
        self.receiver_mark_index = self.rod_count + 3

        # Generalized coordinates are ordered driven body, rod frames, receiver.
        dof_count = self.rod_count + 2
        self.angle = np.zeros(dof_count, dtype=float)
        self.omega = np.zeros(dof_count, dtype=float)

        rod_radius = float(self.config["rod"]["radius"])
        body_d = self.config["rigid_bodies"]["driven"]
        body_r = self.config["rigid_bodies"]["receiver"]
        driven_inertia = 0.5 * float(body_d["mass"]) * float(body_d["radius"]) ** 2
        receiver_inertia = 0.5 * float(body_r["mass"]) * float(body_r["radius"]) ** 2
        rod_polar_inertia = 0.5 * self.mass[: self.rod_count] * rod_radius**2
        self.dof_inertia = np.r_[driven_inertia, rod_polar_inertia, receiver_inertia]

        # C = GJ is a plausible soft-polymer torsional rigidity. Distributing it
        # over the edges gives a finite-volume torsion discretization.
        rod_length = float(self.config["rod"]["length"])
        edge_length = rod_length / (self.rod_count - 1)
        torsional_rigidity = 1.2  # N m^2
        self.edge_stiffness = torsional_rigidity / edge_length
        self.edge_damping = 1.0e-3  # N m s / rad, internal Kelvin damping
        self.max_step = 1.0e-4

        self.inertia_tensor = self._make_inertia_tensors(edge_length)
        self.mass_quantity = self._make_mass_quantities()
        self.domain_weights = self._make_domain_weights()
        self.cumulative_angular_impulse = np.zeros((4, 3), dtype=float)
        self.step_count = 0

        excitation = self.config["excitation"]
        self.torque_start = float(excitation["start_time"])
        self.torque_rise = float(excitation["rise_time"])
        self.torque_hold = float(excitation["hold_until"])
        self.torque_end = float(excitation["end_time"])
        self.peak_torque = float(excitation["peak_torque"])

        # Fixed, strictly positive scales based on the prescribed impulse.
        pulse_duration = (
            0.5 * self.torque_rise
            + (self.torque_hold - self.torque_start - self.torque_rise)
            + 0.5 * (self.torque_end - self.torque_hold)
        )
        self.reference_angular_momentum = max(
            abs(self.peak_torque * pulse_duration), 1.0e-6
        )
        largest_radius = max(float(body_d["radius"]), float(body_r["radius"]))
        self.reference_momentum = self.reference_angular_momentum / largest_radius

    def _make_inertia_tensors(self, edge_length: float) -> np.ndarray:
        tensors = np.zeros((len(self.mass), 3, 3), dtype=float)
        rod_radius = float(self.config["rod"]["radius"])
        for i, mass in enumerate(self.mass[: self.rod_count]):
            tensors[i, 0, 0] = 0.5 * mass * rod_radius**2
            transverse = mass * (3.0 * rod_radius**2 + edge_length**2) / 12.0
            tensors[i, 1, 1] = transverse
            tensors[i, 2, 2] = transverse

        for index, name in (
            (self.driven_index, "driven"),
            (self.receiver_index, "receiver"),
        ):
            body = self.config["rigid_bodies"][name]
            mass = float(body["mass"])
            radius = float(body["radius"])
            length = float(body["axial_length"])
            tensors[index, 0, 0] = 0.5 * mass * radius**2
            transverse = mass * (3.0 * radius**2 + length**2) / 12.0
            tensors[index, 1, 1] = transverse
            tensors[index, 2, 2] = transverse
        return tensors

    def _make_mass_quantities(self) -> np.ndarray:
        quantities = np.zeros((len(self.mass), 3), dtype=float)
        quantities[: self.rod_count, 0] = self.mass[: self.rod_count]
        quantities[self.driven_index, 1] = self.mass[self.driven_index]
        quantities[self.receiver_index, 2] = self.mass[self.receiver_index]
        return quantities

    def _make_domain_weights(self) -> np.ndarray:
        # Rows are driven assembly, receiver assembly, rod, and complete system.
        weights = np.zeros((4, len(self.mass)), dtype=float)
        weights[0, [self.driven_index, self.driven_mark_index]] = 1.0
        weights[1, [self.receiver_index, self.receiver_mark_index]] = 1.0
        weights[2, : self.rod_count] = 1.0
        weights[3, :] = 1.0
        return weights

    def _applied_torque(self, time: float) -> float:
        if time < self.torque_start or time >= self.torque_end:
            return 0.0
        rise_end = self.torque_start + self.torque_rise
        if time < rise_end:
            return self.peak_torque * (time - self.torque_start) / self.torque_rise
        if time <= self.torque_hold:
            return self.peak_torque
        return self.peak_torque * (self.torque_end - time) / (
            self.torque_end - self.torque_hold
        )

    def _edge_torques(self) -> np.ndarray:
        relative_angle = self.angle[1:] - self.angle[:-1]
        relative_speed = self.omega[1:] - self.omega[:-1]
        return (
            self.edge_stiffness * relative_angle
            + self.edge_damping * relative_speed
        )

    def _step(self, step: float) -> None:
        edge_torque = self._edge_torques()
        applied = self._applied_torque(self.time)

        torque = np.zeros_like(self.angle)
        torque[:-1] += edge_torque
        torque[1:] -= edge_torque
        torque[0] += applied

        # Symplectic Euler is stable for the selected step below the torsional
        # wave CFL limit and does not inject artificial angular momentum.
        self.omega += step * torque / self.dof_inertia
        self.angle += step * self.omega

        start_joint = edge_torque[0]
        end_joint = edge_torque[-1]
        domain_couples = np.array(
            [
                applied + start_joint,
                -end_joint,
                -start_joint + end_joint,
                applied,
            ],
            dtype=float,
        )
        self.cumulative_angular_impulse[:, 0] += step * domain_couples
        self.time += step
        self.step_count += 1

    @staticmethod
    def _x_quaternion(angle: np.ndarray) -> np.ndarray:
        quaternion = np.zeros((len(angle), 4), dtype=float)
        quaternion[:, 0] = np.sin(0.5 * angle)
        quaternion[:, 3] = np.cos(0.5 * angle)
        return quaternion

    def _current_budget_couples(self) -> np.ndarray:
        edge_torque = self._edge_torques()
        applied = self._applied_torque(self.time)
        couples = np.zeros((4, 3, 3), dtype=float)
        # Three columns represent applied, start-joint, and end-joint terms.
        couples[0, 0, 0] = applied
        couples[0, 1, 0] = edge_torque[0]
        couples[1, 2, 0] = -edge_torque[-1]
        couples[2, 1, 0] = -edge_torque[0]
        couples[2, 2, 0] = edge_torque[-1]
        couples[3, 0, 0] = applied
        return couples

    def _make_frame(self) -> StateFrame:
        count = len(self.mass)
        position = self.base_position.copy()
        velocity = np.zeros((count, 3), dtype=float)
        angular_velocity = np.zeros((count, 3), dtype=float)

        rod_angle = self.angle[1:-1]
        rod_omega = self.omega[1:-1]
        driven_angle = self.angle[0]
        receiver_angle = self.angle[-1]
        driven_omega = self.omega[0]
        receiver_omega = self.omega[-1]

        sample_angle = np.r_[
            rod_angle, driven_angle, receiver_angle, driven_angle, receiver_angle
        ]
        orientation = self._x_quaternion(sample_angle)
        angular_velocity[: self.rod_count, 0] = rod_omega
        angular_velocity[self.driven_index, 0] = driven_omega
        angular_velocity[self.receiver_index, 0] = receiver_omega
        angular_velocity[self.driven_mark_index, 0] = driven_omega
        angular_velocity[self.receiver_mark_index, 0] = receiver_omega

        # Welded radial marks orbit their body centers under the solved rotations.
        driven_center = self.base_position[self.driven_index]
        receiver_center = self.base_position[self.receiver_index]
        driven_radius = 0.075
        receiver_radius = 0.0475
        position[self.driven_mark_index] = driven_center + np.array(
            [0.0, -driven_radius * math.sin(driven_angle),
             driven_radius * math.cos(driven_angle)]
        )
        position[self.receiver_mark_index] = receiver_center + np.array(
            [0.0, -receiver_radius * math.sin(receiver_angle),
             receiver_radius * math.cos(receiver_angle)]
        )
        velocity[self.driven_mark_index] = np.array(
            [0.0, -driven_radius * driven_omega * math.cos(driven_angle),
             -driven_radius * driven_omega * math.sin(driven_angle)]
        )
        velocity[self.receiver_mark_index] = np.array(
            [0.0, -receiver_radius * receiver_omega * math.cos(receiver_angle),
             -receiver_radius * receiver_omega * math.sin(receiver_angle)]
        )

        zeros_g3 = np.zeros((4, 3), dtype=float)
        zeros_gk3 = np.zeros((4, 3, 3), dtype=float)
        fields = {
            "budget_cumulative_external_angular_impulse":
                self.cumulative_angular_impulse.copy(),
            "budget_cumulative_external_impulse": zeros_g3.copy(),
            "budget_cumulative_inflow_angular_momentum": zeros_g3.copy(),
            "budget_cumulative_inflow_momentum": zeros_g3.copy(),
            "budget_cumulative_outflow_angular_momentum": zeros_g3.copy(),
            "budget_cumulative_outflow_momentum": zeros_g3.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_couple": self._current_budget_couples(),
            "budget_external_force": zeros_gk3.copy(),
            "budget_force_position": zeros_gk3.copy(),
            "cumulative_mass_source": np.zeros(3, dtype=float),
        }
        attributes = {
            "component_id": self.component_id.copy(),
            "arc_length": self.arc_length.copy(),
            "material_twist": sample_angle.copy(),
            "is_driven": self.is_driven.copy(),
            "inertia_tensor": self.inertia_tensor.copy(),
            "mass_quantity": self.mass_quantity.copy(),
        }
        scalars = {
            "applied_torque": self._applied_torque(self.time),
            "reference_angular_momentum": self.reference_angular_momentum,
            "reference_momentum": self.reference_momentum,
        }
        frame = StateFrame(
            time=float(self.time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            scalars=scalars,
            fields=fields,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-14:
            step = min(self.max_step, target - self.time)
            self._step(step)
        # Eliminate harmless accumulated clock roundoff at requested samples.
        self.time = target
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "lumped torsional rod with symplectic Euler stepping",
            "internal_steps": self.step_count,
            "torsional_rigidity": 1.2,
            "finite": bool(
                np.all(np.isfinite(self.angle)) and np.all(np.isfinite(self.omega))
            ),
        }
