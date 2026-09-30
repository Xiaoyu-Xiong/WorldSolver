"""Physically-based torsion solver for the rod-coupled torque transducer.

The scene consists of two cylindrical rigid bodies connected by a naturally
straight, inextensible elastic rod.  The simulation tracks torsional waves on the
rod and the resulting rotations of the bodies about the rod axis.  The driven
body receives an prescribed torque pulse about the rod axis; the receiver
body responds through the twisting rod.  Positions are frozen along the rod axis
because the scaffold constrains translation; only axial rotations evolve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.config = scene.config
        self.frame0: StateFrame = scene.initial_state()

        # Rod discretisation
        rod_cfg = self.config["rod"]
        self.n_rod: int = int(rod_cfg["vertex_count"])
        self.L: float = float(rod_cfg["length"])
        self.radius: float = float(rod_cfg["radius"])
        self.rho: float = float(rod_cfg["density"])
        self.ds: float = self.L / (self.n_rod - 1)

        # Rod mass per vertex (full segment lumping)
        self.rod_mass: float = self.rho * np.pi * self.radius**2 * self.ds

        # Rigid body parameters
        driven = self.config["rigid_bodies"]["driven"]
        receiver = self.config["rigid_bodies"]["receiver"]
        self.driven_center = np.asarray(driven["center"], dtype=float)
        self.receiver_center = np.asarray(receiver["center"], dtype=float)
        self.driven_radius = float(driven["radius"])
        self.driven_length = float(driven["axial_length"])
        self.driven_mass = float(driven["mass"])
        self.receiver_radius = float(receiver["radius"])
        self.receiver_length = float(receiver["axial_length"])
        self.receiver_mass = float(receiver["mass"])

        # Inertias about the rod axis (x-axis)
        self.I_rod_vertex = 0.5 * self.rod_mass * self.radius**2
        self.I_driven = 0.5 * self.driven_mass * self.driven_radius**2
        self.I_receiver = 0.5 * self.receiver_mass * self.receiver_radius**2

        # Total rotational inertia per DOF (rod vertex + attached body at ends)
        self.I_total = np.full(self.n_rod, self.I_rod_vertex, dtype=float)
        self.I_total[0] += self.I_driven
        self.I_total[-1] += self.I_receiver

        # Torsional stiffness: use a physically plausible shear modulus for a
        # polymer rod.  The torsional wave speed c = sqrt(G/rho) controls the
        # stable time step.
        self.G = 1.0e9  # Pa
        self.J = 0.5 * np.pi * self.radius**4  # polar moment of cross-section
        self.k_t = self.G * self.J / self.ds

        # Light Rayleigh-like damping (c_i = zeta * I_i) so every DOF damps at
        # the same rate zeta without overdamping high modes.
        self.zeta = 0.5  # 1/s

        # Time-stepping state
        self.theta = np.zeros(self.n_rod, dtype=float)
        self.omega = np.zeros(self.n_rod, dtype=float)
        self.t = 0.0

        # Cumulative angular impulse from the applied torque pulse
        self.cum_ang_impulse = 0.0

        # Torque-pulse parameters
        exc = self.config["excitation"]
        self.torque_start = float(exc["start_time"])
        self.torque_rise = float(exc["rise_time"])
        self.torque_hold = float(exc["hold_until"])
        self.torque_end = float(exc["end_time"])
        self.peak_torque = float(exc["peak_torque"])

        # Stable explicit sub-step for the torsional wave
        wave_speed = np.sqrt(self.G / self.rho)
        self.dt_sub = 0.4 * self.ds / wave_speed
        # Cap at a sensible maximum so very soft/stiff parameter choices behave.
        self.dt_sub = min(self.dt_sub, 1.0e-4)
        if self.dt_sub <= 0.0:
            raise ValueError("computed sub-step is non-positive")

        # Reference scales (arbitrary positive units)
        self.ref_momentum = 1.0
        self.ref_ang_momentum = 1.0

        # Total number of samples in the StateFrame
        self.N = len(self.frame0.position)

        # Cached inertia tensors for output
        self._inertia = np.zeros((self.N, 3, 3), dtype=float)
        # Rod vertices (cylindrical segments about x)
        I_rod_yz = (1.0 / 12.0) * self.rod_mass * (3.0 * self.radius**2 + self.ds**2)
        for i in range(self.n_rod):
            self._inertia[i, 0, 0] = self.I_rod_vertex
            self._inertia[i, 1, 1] = I_rod_yz
            self._inertia[i, 2, 2] = I_rod_yz
        # Driven body (cylinder about x)
        driven_Iyz = (1.0 / 12.0) * self.driven_mass * (
            3.0 * self.driven_radius**2 + self.driven_length**2
        )
        self._inertia[self.n_rod, 0, 0] = self.I_driven
        self._inertia[self.n_rod, 1, 1] = driven_Iyz
        self._inertia[self.n_rod, 2, 2] = driven_Iyz
        # Receiver body (cylinder about x)
        receiver_Iyz = (1.0 / 12.0) * self.receiver_mass * (
            3.0 * self.receiver_radius**2 + self.receiver_length**2
        )
        self._inertia[self.n_rod + 1, 0, 0] = self.I_receiver
        self._inertia[self.n_rod + 1, 1, 1] = receiver_Iyz
        self._inertia[self.n_rod + 1, 2, 2] = receiver_Iyz

        # Mark offsets from each body centre (constant in material frame)
        self._driven_mark_offset0 = self.frame0.position[self.n_rod + 2] - self.driven_center
        self._receiver_mark_offset0 = self.frame0.position[self.n_rod + 3] - self.receiver_center

    def _applied_torque(self, t: float) -> float:
        """Piecewise-linear torque pulse about the rod axis."""
        if t < self.torque_start:
            return 0.0
        if t < self.torque_start + self.torque_rise:
            return self.peak_torque * (t - self.torque_start) / self.torque_rise
        if t < self.torque_hold:
            return self.peak_torque
        if t < self.torque_end:
            return self.peak_torque * (self.torque_end - t) / (self.torque_end - self.torque_hold)
        return 0.0

    def _step(self, h: float) -> None:
        """Explicit velocity-Verlet style sub-step for the torsional chain."""
        # Elastic torque from twist differences between adjacent vertices.
        # Segment torque is k_t * (theta_{i+1} - theta_i); sign convention
        # gives equal and opposite torques on the two vertices it connects.
        dtheta = np.diff(self.theta)
        seg_torque = self.k_t * dtheta

        tau = np.zeros(self.n_rod, dtype=float)
        tau[1:-1] = seg_torque[:-1] - seg_torque[1:]
        tau[0] += seg_torque[0]
        tau[-1] -= seg_torque[-1]

        # External torque applied to the driven endpoint.
        tau[0] += self._applied_torque(self.t)

        # Light viscous damping.
        tau -= self.zeta * self.I_total * self.omega

        # Angular accelerations and semi-implicit Euler update.
        alpha = tau / self.I_total
        self.omega += alpha * h
        self.theta += self.omega * h
        self.t += h

    def _quaternion_x(self, angle: float) -> NDArray:
        """Quaternion (xyzw) representing a rotation of `angle` radians about x."""
        return np.array([np.sin(angle * 0.5), 0.0, 0.0, np.cos(angle * 0.5)], dtype=float)

    @staticmethod
    def _rotate_about_x(vector: NDArray, angle: float) -> NDArray:
        x, y, z = vector
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([x, y * c - z * s, y * s + z * c], dtype=float)

    def _build_frame(self, time: float) -> StateFrame:
        n = self.n_rod
        orientation = np.zeros((self.N, 4), dtype=float)
        angular_velocity = np.zeros((self.N, 3), dtype=float)

        # Rod vertices: twist angle maps directly to a rotation about x.
        for i in range(n):
            orientation[i] = self._quaternion_x(self.theta[i])
            angular_velocity[i, 0] = self.omega[i]

        # Driven body, receiver body and their orientation marks share the
        # endpoint rotations.
        q_driven = self._quaternion_x(self.theta[0])
        q_receiver = self._quaternion_x(self.theta[-1])
        orientation[n] = q_driven
        orientation[n + 1] = q_receiver
        orientation[n + 2] = q_driven
        orientation[n + 3] = q_receiver

        angular_velocity[n, 0] = self.omega[0]
        angular_velocity[n + 1, 0] = self.omega[-1]
        angular_velocity[n + 2, 0] = self.omega[0]
        angular_velocity[n + 3, 0] = self.omega[-1]

        # Positions: rod vertices and body centres are frozen along the axis;
        # orientation marks move rigidly with their parent bodies.
        position = self.frame0.position.copy()
        position[n + 2] = self.driven_center + self._rotate_about_x(
            self._driven_mark_offset0, self.theta[0]
        )
        position[n + 3] = self.receiver_center + self._rotate_about_x(
            self._receiver_mark_offset0, self.theta[-1]
        )

        # Velocities of the moving mark centres (v = omega x r).
        velocity = np.zeros((self.N, 3), dtype=float)
        omega_driven = np.array([self.omega[0], 0.0, 0.0], dtype=float)
        omega_receiver = np.array([self.omega[-1], 0.0, 0.0], dtype=float)
        velocity[n + 2] = np.cross(omega_driven, position[n + 2] - self.driven_center)
        velocity[n + 3] = np.cross(omega_receiver, position[n + 3] - self.receiver_center)

        # Mass quantity (single material -> K=1 column).
        mass_quantity = self.frame0.mass.copy().reshape(-1, 1)

        # Budget fields: one global domain, one external couple term.
        torque_vec = np.array([self._applied_torque(time), 0.0, 0.0], dtype=float)
        cum_ang_impulse = np.array([self.cum_ang_impulse, 0.0, 0.0], dtype=float)

        budget_external_couple = np.zeros((1, 1, 3), dtype=float)
        budget_external_couple[0, 0] = torque_vec
        budget_external_force = np.zeros((1, 1, 3), dtype=float)
        budget_force_position = np.zeros((1, 1, 3), dtype=float)
        budget_force_position[0, 0] = self.driven_center.copy()
        budget_domain_weights = np.ones((1, self.N), dtype=float)

        # Material twist attribute follows the simulated rod twist.
        material_twist = np.zeros(self.N, dtype=float)
        material_twist[:n] = self.theta

        attributes = {
            "component_id": self.frame0.attributes["component_id"],
            "arc_length": self.frame0.attributes["arc_length"],
            "material_twist": material_twist,
            "is_driven": self.frame0.attributes["is_driven"],
            "inertia_tensor": self._inertia,
            "mass_quantity": mass_quantity,
        }

        fields = {
            "budget_cumulative_external_angular_impulse": cum_ang_impulse.reshape(1, 3),
            "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=float),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": budget_domain_weights,
            "budget_external_couple": budget_external_couple,
            "budget_external_force": budget_external_force,
            "budget_force_position": budget_force_position,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }

        scalars = {
            "applied_torque": self._applied_torque(time),
            "reference_momentum": self.ref_momentum,
            "reference_angular_momentum": self.ref_ang_momentum,
        }

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.frame0.mass.copy(),
            body_id=self.frame0.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            raise ValueError(f"advance_to called with non-increasing time {time} <= {self.t}")

        while self.t < time:
            h = min(self.dt_sub, time - self.t)
            # Update cumulative angular impulse using the torque at the
            # beginning of the sub-step (rectangle rule).
            self.cum_ang_impulse += self._applied_torque(self.t) * h
            self._step(h)

        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt_sub": self.dt_sub,
            "final_time": self.t,
            "n_dof": self.n_rod,
        }
