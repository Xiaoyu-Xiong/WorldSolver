"""BenchmarkSolver implementation for 088_rod_torque_transducer scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config
        self.rod_cfg = self.config["rod"]
        self.rigid_cfg = self.config["rigid_bodies"]
        self.excite_cfg = self.config["excitation"]

        # Number of rod vertices
        self.n_rod = int(self.rod_cfg["vertex_count"])  # 65
        self.rod_length = float(self.rod_cfg["length"])  # 1.6
        self.rod_radius = float(self.rod_cfg["radius"])  # 0.018
        self.rod_density = float(self.rod_cfg["density"])  # 1150.0

        self.start_pos = np.asarray(self.config["initial_conditions"]["rod_start"], dtype=float)
        self.end_pos = np.asarray(self.config["initial_conditions"]["rod_end"], dtype=float)

        self.driven_center = np.asarray(self.rigid_cfg["driven"]["center"], dtype=float)
        self.receiver_center = np.asarray(self.rigid_cfg["receiver"]["center"], dtype=float)

        self.m_driven = float(self.rigid_cfg["driven"]["mass"])
        self.r_driven = float(self.rigid_cfg["driven"]["radius"])
        self.h_driven = float(self.rigid_cfg["driven"]["axial_length"])

        self.m_receiver = float(self.rigid_cfg["receiver"]["mass"])
        self.r_receiver = float(self.rigid_cfg["receiver"]["radius"])
        self.h_receiver = float(self.rigid_cfg["receiver"]["axial_length"])

        # Geometry arrays
        self.rod_positions = np.linspace(self.start_pos, self.end_pos, self.n_rod)
        self.driven_mark_offset = np.array([0.0, 0.0, 0.075])
        self.receiver_mark_offset = np.array([0.0, 0.0, 0.0475])

        self.total_samples = self.n_rod + 4  # 69

        # Masses
        self.segment_len = self.rod_length / (self.n_rod - 1)
        self.rod_mass_per_vert = self.rod_density * np.pi * (self.rod_radius**2) * self.segment_len
        self.masses = np.r_[
            np.full(self.n_rod, self.rod_mass_per_vert),
            self.m_driven,
            self.m_receiver,
            0.0,
            0.0,
        ]

        self.body_id = np.r_[np.zeros(self.n_rod, dtype=int), [1, 2, 3, 4]]
        self.component_id = np.r_[np.zeros(self.n_rod), [1.0, 2.0, 3.0, 4.0]]
        self.arc_length = np.r_[np.linspace(0.0, self.rod_length, self.n_rod), np.zeros(4)]
        self.is_driven = np.r_[np.zeros(self.n_rod), [1.0, 0.0, 1.0, 0.0]]

        # Rotational moments of inertia about x-axis (rod axis)
        # Polar second moment of area for circular rod cross-section: J = pi/2 * r^4
        J_polar = 0.5 * np.pi * (self.rod_radius**4)
        I_rod_node = self.rod_density * J_polar * self.segment_len

        # Cylinder rotating about cylinder symmetry axis (aligned with x-axis): I = 1/2 * m * r^2
        I_driven_cyl = 0.5 * self.m_driven * (self.r_driven**2)
        I_receiver_cyl = 0.5 * self.m_receiver * (self.r_receiver**2)

        self.I_eff = np.full(self.n_rod, I_rod_node)
        # Clamped attachment at start (vertex 0) and end (vertex -1)
        self.I_eff[0] += I_driven_cyl
        self.I_eff[-1] += I_receiver_cyl

        # Inertia tensor for all samples [N, 3, 3]
        # Cylinder aligned with x-axis: Ix = 1/2 m r^2, Iy = Iz = 1/12 m (3 r^2 + h^2)
        self.inertia_tensors = np.zeros((self.total_samples, 3, 3), dtype=float)
        # Rod vertices intrinsic inertia
        for i in range(self.n_rod):
            self.inertia_tensors[i, 0, 0] = I_rod_node
            self.inertia_tensors[i, 1, 1] = 0.5 * I_rod_node + (1.0 / 12.0) * self.rod_mass_per_vert * (self.segment_len**2)
            self.inertia_tensors[i, 2, 2] = self.inertia_tensors[i, 1, 1]

        # Driven body (index n_rod)
        Ix_d = 0.5 * self.m_driven * (self.r_driven**2)
        Iyz_d = (1.0 / 12.0) * self.m_driven * (3.0 * (self.r_driven**2) + self.h_driven**2)
        self.inertia_tensors[self.n_rod, 0, 0] = Ix_d
        self.inertia_tensors[self.n_rod, 1, 1] = Iyz_d
        self.inertia_tensors[self.n_rod, 2, 2] = Iyz_d

        # Receiver body (index n_rod + 1)
        Ix_r = 0.5 * self.m_receiver * (self.r_receiver**2)
        Iyz_r = (1.0 / 12.0) * self.m_receiver * (3.0 * (self.r_receiver**2) + self.h_receiver**2)
        self.inertia_tensors[self.n_rod + 1, 0, 0] = Ix_r
        self.inertia_tensors[self.n_rod + 1, 1, 1] = Iyz_r
        self.inertia_tensors[self.n_rod + 1, 2, 2] = Iyz_r

        # Material properties for rod torsion
        # Shear modulus G (typical rubber / elastomer ~ 2.5 MPa)
        self.G_modulus = 2.5e6
        self.torsional_stiffness = self.G_modulus * J_polar  # GJ

        # Time-stepping simulation state
        self.current_time = 0.0
        self.theta = np.zeros(self.n_rod, dtype=float)
        self.omega = np.zeros(self.n_rod, dtype=float)

        # Budget cumulative fields
        self.cum_ext_ang_impulse = np.zeros((1, 3), dtype=float)

        # Excitation parameters
        self.t_start = float(self.excite_cfg["start_time"])  # 0.20
        self.t_rise = float(self.excite_cfg["rise_time"])    # 0.25
        self.t_hold = float(self.excite_cfg["hold_until"])   # 0.85
        self.t_end = float(self.excite_cfg["end_time"])      # 1.10
        self.tau_peak = float(self.excite_cfg["peak_torque"])  # 0.75

        # Internal simulation sub-step size
        self.sim_dt = 5.0e-5

    def _applied_torque(self, t: float) -> float:
        if t < self.t_start:
            return 0.0
        elif t < self.t_rise:
            return self.tau_peak * (t - self.t_start) / (self.t_rise - self.t_start)
        elif t < self.t_hold:
            return self.tau_peak
        elif t < self.t_end:
            return self.tau_peak * (self.t_end - t) / (self.t_end - self.t_hold)
        else:
            return 0.0

    def _physics_step(self, dt: float) -> None:
        d_th = np.diff(self.theta)
        d_om = np.diff(self.omega)

        # Internal torsional torque transmitted through edges
        # Edge i is between vertex i and i+1
        # Viscous internal damping added for numerical stability and physical wave decay
        k_edge = self.torsional_stiffness / self.segment_len
        c_edge = 0.001 * k_edge
        T_edge = k_edge * d_th + c_edge * d_om

        # Net torque on vertices
        T_net = np.zeros(self.n_rod, dtype=float)
        T_net[0] = T_edge[0]
        T_net[1:-1] = T_edge[1:] - T_edge[:-1]
        T_net[-1] = -T_edge[-1]

        # External torque applied to driven body (clamped to vertex 0)
        tau_ext = self._applied_torque(self.current_time)
        T_net[0] += tau_ext

        # Small background damping
        T_net -= 0.0005 * self.omega

        # Acceleration and time integration
        alpha = T_net / self.I_eff
        self.omega += alpha * dt
        self.theta += self.omega * dt

        # Accumulate external angular impulse
        self.cum_ext_ang_impulse[0, 0] += tau_ext * dt
        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until the target time
        while self.current_time < time - 1.0e-9:
            dt_step = min(self.sim_dt, time - self.current_time)
            self._physics_step(dt_step)

        # Ensure current_time is exactly target time
        self.current_time = time

        # Compute orientations (quaternion [qx, qy, qz, qw] for rotation about x-axis by angle theta)
        # q = [sin(theta/2), 0, 0, cos(theta/2)]
        orientations = np.zeros((self.total_samples, 4), dtype=float)
        angular_velocities = np.zeros((self.total_samples, 3), dtype=float)
        positions = np.zeros((self.total_samples, 3), dtype=float)
        velocities = np.zeros((self.total_samples, 3), dtype=float)

        # Rod vertices
        positions[:self.n_rod] = self.rod_positions
        for i in range(self.n_rod):
            th = self.theta[i]
            orientations[i] = [np.sin(th / 2.0), 0.0, 0.0, np.cos(th / 2.0)]
            angular_velocities[i] = [self.omega[i], 0.0, 0.0]

        # Driven body (index n_rod)
        th_driven = self.theta[0]
        om_driven = self.omega[0]
        orientations[self.n_rod] = [np.sin(th_driven / 2.0), 0.0, 0.0, np.cos(th_driven / 2.0)]
        angular_velocities[self.n_rod] = [om_driven, 0.0, 0.0]
        positions[self.n_rod] = self.driven_center

        # Receiver body (index n_rod + 1)
        th_receiver = self.theta[-1]
        om_receiver = self.omega[-1]
        orientations[self.n_rod + 1] = [np.sin(th_receiver / 2.0), 0.0, 0.0, np.cos(th_receiver / 2.0)]
        angular_velocities[self.n_rod + 1] = [om_receiver, 0.0, 0.0]
        positions[self.n_rod + 1] = self.receiver_center

        # Driven orientation mark (index n_rod + 2)
        # Rotates with driven body
        orientations[self.n_rod + 2] = orientations[self.n_rod]
        angular_velocities[self.n_rod + 2] = angular_velocities[self.n_rod]
        # Offset rotated about x-axis by th_driven: [0, -r0*sin(th), r0*cos(th)]
        r_d = self.driven_mark_offset[2]
        positions[self.n_rod + 2] = self.driven_center + np.array([
            0.0,
            -r_d * np.sin(th_driven),
            r_d * np.cos(th_driven),
        ])
        velocities[self.n_rod + 2] = np.array([
            0.0,
            -r_d * om_driven * np.cos(th_driven),
            -r_d * om_driven * np.sin(th_driven),
        ])

        # Receiver orientation mark (index n_rod + 3)
        # Rotates with receiver body
        orientations[self.n_rod + 3] = orientations[self.n_rod + 1]
        angular_velocities[self.n_rod + 3] = angular_velocities[self.n_rod + 1]
        r_r = self.receiver_mark_offset[2]
        positions[self.n_rod + 3] = self.receiver_center + np.array([
            0.0,
            -r_r * np.sin(th_receiver),
            r_r * np.cos(th_receiver),
        ])
        velocities[self.n_rod + 3] = np.array([
            0.0,
            -r_r * om_receiver * np.cos(th_receiver),
            -r_r * om_receiver * np.sin(th_receiver),
        ])

        # Material twist per vertex (spatial derivative of theta along rod)
        mat_twist = np.zeros(self.total_samples, dtype=float)
        d_th = np.diff(self.theta) / self.segment_len
        mat_twist[0] = d_th[0]
        mat_twist[1:self.n_rod - 1] = 0.5 * (d_th[:-1] + d_th[1:])
        mat_twist[self.n_rod - 1] = d_th[-1]

        # External torque scalar
        tau_current = self._applied_torque(self.current_time)

        # Budget fields
        domain_weights = np.ones((1, self.total_samples), dtype=float)
        ext_couple = np.zeros((1, 1, 3), dtype=float)
        ext_couple[0, 0, 0] = tau_current
        ext_force = np.zeros((1, 1, 3), dtype=float)
        force_pos = np.zeros((1, 1, 3), dtype=float)
        force_pos[0, 0] = self.driven_center

        frame = StateFrame(
            time=self.current_time,
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=angular_velocities,
            mass=self.masses,
            body_id=self.body_id,
            attributes={
                "component_id": self.component_id,
                "arc_length": self.arc_length,
                "material_twist": mat_twist,
                "is_driven": self.is_driven,
                "inertia_tensor": self.inertia_tensors,
                "mass_quantity": self.masses,
            },
            fields={
                "budget_domain_weights": domain_weights,
                "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=float),
                "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_external_force": ext_force,
                "budget_force_position": force_pos,
                "budget_external_couple": ext_couple,
                "cumulative_mass_source": np.zeros((1,), dtype=float),
            },
            scalars={
                "applied_torque": tau_current,
                "reference_momentum": 1.0,
                "reference_angular_momentum": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "driven_theta": float(self.theta[0]),
            "receiver_theta": float(self.theta[-1]),
            "driven_omega": float(self.omega[0]),
            "receiver_omega": float(self.omega[-1]),
        }

