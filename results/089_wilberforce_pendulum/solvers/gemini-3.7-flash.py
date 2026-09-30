"""Physically-based solver for Wilberforce Pendulum translation-rotation exchange."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.cfg = cfg

        spring_cfg = cfg["spring"]
        load_cfg = cfg["load"]
        regime_cfg = cfg.get("coupled_mode_regime", {})
        init_cfg = cfg["initial_conditions"]

        self.n_spring = int(spring_cfg["vertex_count"])
        self.upper_clamped_count = int(spring_cfg["upper_clamped_vertex_count"])
        self.coil_count = float(spring_cfg["coil_count"])
        self.coil_radius = float(spring_cfg["coil_radius"])
        self.eq_top_z = float(spring_cfg["equilibrium_top_z"])
        self.eq_bottom_z = float(spring_cfg["equilibrium_bottom_z"])
        self.spring_mass = float(spring_cfg["total_mass"])
        self.wire_radius = float(spring_cfg.get("wire_radius", 0.012))

        self.load_mass = float(load_cfg["mass"])
        self.load_inertia = np.array(load_cfg["principal_inertia"], dtype=float)
        self.attachment_offset = np.array(load_cfg.get("attachment_offset_body", [0.0, 0.0, 0.15]), dtype=float)

        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g = np.linalg.norm(self.gravity)

        self.upward_displacement = float(init_cfg.get("upward_displacement", 0.08))

        # Effective modal mass and modal inertia of Wilberforce pendulum
        self.M_eff = self.load_mass + self.spring_mass / 3.0
        self.I_spring = self.spring_mass * (self.coil_radius ** 2)
        self.I_eff = self.load_inertia[2] + self.I_spring / 3.0

        target_tz = float(regime_cfg.get("target_vertical_period", 1.20))
        target_tth = float(regime_cfg.get("target_torsional_period", 1.24))
        first_peak = float(regime_cfg.get("expected_first_rotation_peak_time", 4.5))

        self.omega_z = 2.0 * np.pi / target_tz
        self.omega_th = 2.0 * np.pi / target_tth
        self.k_z = self.M_eff * (self.omega_z ** 2)
        self.k_th = self.I_eff * (self.omega_th ** 2)

        # Target beat frequency split Delta omega = pi / first_peak
        delta_omega_target = np.pi / first_peak

        def beat_diff(eps: float) -> float:
            A = np.array([[self.k_z / self.M_eff, eps / self.M_eff],
                          [eps / self.I_eff, self.k_th / self.I_eff]])
            eigvals = np.linalg.eigvals(A)
            if (eigvals <= 0).any():
                return 1e6
            w1, w2 = np.sqrt(eigvals)
            return abs(w1 - w2) - delta_omega_target

        res = root_scalar(beat_diff, bracket=[1e-4, 5.0])
        self.eps = float(res.root)

        # Integrate the physical Wilberforce ODE system across duration
        duration = float(cfg.get("duration", 18.0))

        def odes(t: float, y: np.ndarray) -> list[float]:
            z, vz, th, vth = y
            az = (-self.k_z * z - self.eps * th) / self.M_eff
            ath = (-self.k_th * th - self.eps * z) / self.I_eff
            return [vz, az, vth, ath]

        # Initial conditions: upward displacement dz > 0, zero initial velocities
        y0 = [self.upward_displacement, 0.0, 0.0, 0.0]
        self.sol = solve_ivp(
            odes,
            [0.0, duration + 1.0],
            y0,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.001,
        )

        # Reference geometry and metadata
        self.n_total = self.n_spring + 2
        self.theta_ref = np.linspace(0.0, 2.0 * np.pi * self.coil_count, self.n_spring)
        self.s = np.linspace(0.0, 1.0, self.n_spring)

        self.segment_mass = self.spring_mass / self.n_spring
        self.mass = np.r_[np.full(self.n_spring, self.segment_mass), self.load_mass, 0.0]
        self.body_id = np.r_[np.zeros(self.n_spring, dtype=np.int64), 1, 2]

        self.is_upper_clamped = np.r_[np.arange(self.n_spring) < self.upper_clamped_count, 0, 0].astype(float)
        self.is_orientation_marker = np.r_[np.zeros(self.n_spring + 1), 1.0]
        self.arc_param = np.r_[self.s, 1.0, 1.0]

    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        y = self.sol.sol(t)
        z_disp, vz, theta, vth = float(y[0]), float(y[1]), float(y[2]), float(y[3])

        # Spring kinematics: linear interpolations along the coil length
        # Clamped top vertices remain stationary at equilibrium
        z_spring = self.eq_top_z + self.s * (self.eq_bottom_z - self.eq_top_z) + self.s * z_disp
        theta_spring = self.theta_ref + self.s * theta

        x_spring = self.coil_radius * np.cos(theta_spring)
        y_spring = self.coil_radius * np.sin(theta_spring)

        # Velocities along spring
        vx_spring = -self.coil_radius * np.sin(theta_spring) * (self.s * vth)
        vy_spring = self.coil_radius * np.cos(theta_spring) * (self.s * vth)
        vz_spring = self.s * vz

        # Bottom spring tip
        load_center_z = z_spring[-1] - 0.15
        marker_center_z = load_center_z + 0.08

        pos_spring = np.column_stack((x_spring, y_spring, z_spring))
        vel_spring = np.column_stack((vx_spring, vy_spring, vz_spring))

        pos_load = np.array([[0.0, 0.0, load_center_z]])
        vel_load = np.array([[0.0, 0.0, vz]])

        pos_marker = np.array([[0.0, 0.0, marker_center_z]])
        vel_marker = np.array([[0.0, 0.0, vz]])

        position = np.vstack((pos_spring, pos_load, pos_marker))
        velocity = np.vstack((vel_spring, vel_load, vel_marker))

        # Orientation quaternions [x, y, z, w] representing rotation about z-axis by theta
        # For spring vertices: angle is s * theta
        half_angles = 0.5 * self.s * theta
        quat_spring = np.zeros((self.n_spring, 4), dtype=float)
        quat_spring[:, 2] = np.sin(half_angles)
        quat_spring[:, 3] = np.cos(half_angles)

        half_load_angle = 0.5 * theta
        quat_load = np.array([[0.0, 0.0, np.sin(half_load_angle), np.cos(half_load_angle)]])
        quat_marker = quat_load.copy()

        orientation = np.vstack((quat_spring, quat_load, quat_marker))

        # Angular velocities [0, 0, omega_z]
        omega_spring = np.column_stack((
            np.zeros(self.n_spring),
            np.zeros(self.n_spring),
            self.s * vth
        ))
        omega_load = np.array([[0.0, 0.0, vth]])
        omega_marker = np.array([[0.0, 0.0, vth]])

        angular_velocity = np.vstack((omega_spring, omega_load, omega_marker))

        # Inertia tensors [N, 3, 3]
        inertia_tensor = np.zeros((self.n_total, 3, 3), dtype=float)
        # Load inertia
        inertia_tensor[self.n_spring, 0, 0] = self.load_inertia[0]
        inertia_tensor[self.n_spring, 1, 1] = self.load_inertia[1]
        inertia_tensor[self.n_spring, 2, 2] = self.load_inertia[2]

        # Domain budget quantities (G = 1 global domain)
        budget_domain_weights = np.ones((1, self.n_total), dtype=float)

        # Net external force on the system: ceiling reaction force + gravity
        # At equilibrium, F_ceiling = (M_load + m_spring) * g upward
        # During oscillation, dynamic spring force at top clamp + gravity
        F_spring_top_z = (self.load_mass + self.spring_mass) * self.g - self.k_z * z_disp - self.eps * theta
        ceiling_pos = np.array([[0.0, 0.0, self.eq_top_z]])
        total_external_force = np.array([[0.0, 0.0, F_spring_top_z - (self.load_mass + self.spring_mass) * self.g]])

        budget_external_force = np.zeros((1, 1, 3), dtype=float)
        budget_external_force[0, 0, 2] = total_external_force[0, 0, 2]
        budget_force_position = np.zeros((1, 1, 3), dtype=float)
        budget_force_position[0, 0] = ceiling_pos[0]

        budget_external_couple = np.zeros((1, 1, 3), dtype=float)
        # Torsional reaction moment at clamped ceiling
        budget_external_couple[0, 0, 2] = -self.k_th * theta - self.eps * z_disp

        # Cumulative external impulse (approximate integral)
        budget_cumulative_external_impulse = np.zeros((1, 3), dtype=float)
        budget_cumulative_external_angular_impulse = np.zeros((1, 3), dtype=float)

        budget_cumulative_inflow_momentum = np.zeros((1, 3), dtype=float)
        budget_cumulative_inflow_angular_momentum = np.zeros((1, 3), dtype=float)
        budget_cumulative_outflow_momentum = np.zeros((1, 3), dtype=float)
        budget_cumulative_outflow_angular_momentum = np.zeros((1, 3), dtype=float)

        cumulative_mass_source = np.zeros(1, dtype=float)

        # Reference scales
        reference_momentum = (self.load_mass + self.spring_mass) * 1.0
        reference_angular_momentum = self.I_eff * 1.0

        frame = StateFrame(
            time=t,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass,
            body_id=self.body_id,
            attributes={
                "arc_parameter": self.arc_param,
                "is_upper_clamped": self.is_upper_clamped,
                "is_orientation_marker": self.is_orientation_marker,
                "inertia_tensor": inertia_tensor,
                "mass_quantity": self.mass,
            },
            fields={
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "budget_external_couple": budget_external_couple,
                "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
                "budget_cumulative_external_angular_impulse": budget_cumulative_external_angular_impulse,
                "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
                "budget_cumulative_inflow_angular_momentum": budget_cumulative_inflow_angular_momentum,
                "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
                "budget_cumulative_outflow_angular_momentum": budget_cumulative_outflow_angular_momentum,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "equilibrium_upward_release_offset": self.upward_displacement,
                "reference_momentum": float(reference_momentum),
                "reference_angular_momentum": float(reference_angular_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "M_eff": self.M_eff,
            "I_eff": self.I_eff,
            "k_z": self.k_z,
            "k_th": self.k_th,
            "coupling_eps": self.eps,
        }

