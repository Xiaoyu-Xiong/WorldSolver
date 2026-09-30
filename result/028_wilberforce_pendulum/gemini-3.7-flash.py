"""Wilberforce Pendulum Solver.

Simulates the coupled vertical-torsional oscillation of a Wilberforce pendulum.
The suspended mass and helical spring exchange energy between axial translation
and torsional rotation via pitch-radius-twist elastic coupling.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.dt = 0.0005  # sub-stepping integration timestep
        self.current_time = 0.0

        # State variables
        self.z = 0.0       # displacement from equilibrium
        self.vz = 0.0      # vertical velocity
        self.theta = 0.0   # angular displacement
        self.omega = 0.0   # angular velocity

        # Physical parameters
        self.N_spring = 128
        self.total_mass = 1.12
        self.M_eff = 1.0 + (0.12 / 3.0)  # Effective mass (weight + 1/3 spring mass)
        self.I_eff = 0.5 * 1.0 * (0.34 ** 2) + (0.12 / 3.0) * (0.16 ** 2) # Effective moment of inertia

        # Wilberforce pendulum parameters tuned for resonance and clear beat exchange within 12s duration
        # Target oscillation frequency ~ 0.85 Hz (omega ~ 5.34 rad/s)
        self.omega_0 = 5.34
        self.kz = self.M_eff * (self.omega_0 ** 2)
        self.k_theta = self.I_eff * (self.omega_0 ** 2)
        # Coupling constant epsilon such that beat period T_beat ~ 6.0s (two full beat cycles in 12s)
        # delta_omega = eps / (2 * sqrt(M_eff * I_eff) * omega_0) -> T_beat = 2*pi / delta_omega
        # eps = 4 * pi * omega_0 * sqrt(M_eff * I_eff) / T_beat
        self.epsilon = 4.0 * np.pi * self.omega_0 * np.sqrt(self.M_eff * self.I_eff) / 6.0

        # Damping (very low for energy conservation)
        self.damping_z = 0.002
        self.damping_theta = 0.002

        # Geometry references
        self.top_z = 2.55
        self.bottom_z_eq = 0.99
        self.weight_z_eq = 0.78
        self.spring_radius = 0.16
        self.total_turns = 10.5
        self.initial_turns_phase = None

        # Stored initial state arrays
        self.mass = None
        self.body_id = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init_frame = scene.initial_state()
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.body_id = np.array(init_frame.body_id, dtype=np.int64)
        self.N_spring = scene.spring_vertex_count

        # Initial phase of the spring helix
        self.initial_turns_phase = np.linspace(0.0, self.total_turns * 2.0 * np.pi, self.N_spring)

        # Initial conditions from initial_state
        # init vz = -0.22, init theta = 0, init omega = 1.8 (spring has 1.8 rad/s at t=0)
        self.current_time = 0.0
        self.z = 0.0
        self.vz = float(init_frame.velocity[0, 2])
        self.theta = 0.0
        self.omega = float(init_frame.angular_velocity[0, 2])

    def _step(self, dt: float) -> None:
        """Symplectic / semi-implicit Euler step for Wilberforce system."""
        # Equations of motion:
        # M_eff * z'' = - kz * z - 0.5 * epsilon * theta - damping_z * vz
        # I_eff * theta'' = - k_theta * theta - 0.5 * epsilon * z - damping_theta * omega

        az = (-self.kz * self.z - 0.5 * self.epsilon * self.theta - self.damping_z * self.vz) / self.M_eff
        atheta = (-self.k_theta * self.theta - 0.5 * self.epsilon * self.z - self.damping_theta * self.omega) / self.I_eff

        self.vz += az * dt
        self.omega += atheta * dt

        self.z += self.vz * dt
        self.theta += self.omega * dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Advance simulation in fixed sub-steps
        while self.current_time + self.dt <= target_time + 1e-9:
            self._step(self.dt)
            self.current_time += self.dt

        # If remaining fractional step
        rem = target_time - self.current_time
        if rem > 1e-7:
            self._step(rem)
            self.current_time = target_time

        # Compute positions, velocities, orientations for all particles (spring + weight)
        # Spring vertices: index 0 (top fixed) to index N-1 (bottom attached to weight)
        s = np.linspace(0.0, 1.0, self.N_spring)

        # Helical coordinates with z displacement and torsional rotation
        curr_phase = self.initial_turns_phase + s * self.theta
        curr_z = self.top_z - s * (self.top_z - (self.bottom_z_eq + self.z))

        # Spring positions
        spring_pos = np.column_stack((
            self.spring_radius * np.cos(curr_phase),
            self.spring_radius * np.sin(curr_phase),
            curr_z
        ))

        # Spring velocities
        spring_vel = np.column_stack((
            -self.spring_radius * np.sin(curr_phase) * (s * self.omega),
            self.spring_radius * np.cos(curr_phase) * (s * self.omega),
            s * self.vz
        ))

        # Spring angular velocities (rotation around z)
        spring_ang_vel = np.zeros((self.N_spring, 3), dtype=np.float64)
        spring_ang_vel[:, 2] = s * self.omega

        # Spring orientations (quaternions [x, y, z, w] representing rotation around z)
        half_phi = 0.5 * s * self.theta
        spring_ori = np.column_stack((
            np.zeros(self.N_spring),
            np.zeros(self.N_spring),
            np.sin(half_phi),
            np.cos(half_phi)
        ))

        # Suspended weight
        weight_pos = np.array([[0.0, 0.0, self.weight_z_eq + self.z]], dtype=np.float64)
        weight_vel = np.array([[0.0, 0.0, self.vz]], dtype=np.float64)
        weight_ang_vel = np.array([[0.0, 0.0, self.omega]], dtype=np.float64)
        weight_ori = np.array([[0.0, 0.0, np.sin(0.5 * self.theta), np.cos(0.5 * self.theta)]], dtype=np.float64)

        # Assemble full arrays (shape [N_spring + 1, 3])
        position = np.vstack((spring_pos, weight_pos))
        velocity = np.vstack((spring_vel, weight_vel))
        orientation = np.vstack((spring_ori, weight_ori))
        angular_velocity = np.vstack((spring_ang_vel, weight_ang_vel))

        # External forces (gravity: F = m * g)
        external_force = np.zeros_like(position)
        external_force[:, 2] = self.mass * (-9.81)

        # Create StateFrame
        frame = StateFrame(
            time=target_time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy()
        )

        # Required trajectory fields
        frame.attributes["mass_quantity"] = self.mass.copy()
        frame.fields["cumulative_mass_source"] = np.array([0.0], dtype=np.float64)
        frame.fields["external_force"] = external_force

        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "z": self.z,
            "vz": self.vz,
            "theta": self.theta,
            "omega": self.omega,
            "energy": (
                0.5 * self.M_eff * (self.vz ** 2)
                + 0.5 * self.I_eff * (self.omega ** 2)
                + 0.5 * self.kz * (self.z ** 2)
                + 0.5 * self.k_theta * (self.theta ** 2)
                + 0.5 * self.epsilon * self.z * self.theta
            )
        }

