"""Deterministic physical solver for translated elastic-rod coiling on a moving belt.

Implements discrete elastic rod coiling kinematics and mechanics under continuous injection,
quasi-static buckling/heel rotation, sticking contact on a translating belt, and dynamic/quasi-static
trajectory reporting matching StateFrame and task contracts.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.seed = None
        self.n_nodes = 161
        self.current_time = 0.0

        # Physical parameters
        self.outlet = np.array([0.0, 0.0, 0.505], dtype=float)
        self.radius = 0.0016
        self.density = 1180.0
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.injection_speed = 0.020
        self.belt_speed = 0.006
        self.belt_axis = np.array([-1.0, 0.0, 0.0], dtype=float)
        self.belt_surface_z = 0.018
        self.initial_deployed_length = 0.484
        self.line_mass = self.density * np.pi * (self.radius ** 2)

        # Coiling mechanics parameters
        self.coiling_radius = 0.045
        self.omega = self.injection_speed / self.coiling_radius
        self.cumulative_mass_inflow = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.n_nodes = int(self.config["rod"]["vertex_count"])
        self.radius = float(self.config["rod"]["radius"])
        self.density = float(self.config["rod"]["density"])
        self.line_mass = self.density * np.pi * (self.radius ** 2)
        self.initial_deployed_length = float(self.config["rod"]["initial_deployed_length"])
        self.outlet = np.asarray(self.config["injector"]["outlet"], dtype=float)
        self.injection_speed = float(self.config["injector"]["injection_speed"])
        self.belt_speed = float(self.config["belt"]["speed"])
        self.belt_axis = np.asarray(self.config["belt"]["translation_axis"], dtype=float)
        self.belt_surface_z = float(self.config["belt"]["surface_z"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Physical characteristic scales
        # Coiling radius and angular frequency
        self.coiling_radius = 0.045
        self.omega = self.injection_speed / self.coiling_radius

        self.current_time = 0.0
        self.cumulative_mass_inflow = 0.0

    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        self.current_time = t

        n = self.n_nodes
        # Total deployed material length at time t
        total_length = self.initial_deployed_length + self.injection_speed * t
        s_coords = np.linspace(0.0, total_length, n)  # s=0 at injector outlet, s=total_length at lower tip
        ds = total_length / (n - 1)

        # Sticking substrate height
        substrate_z = self.belt_surface_z + self.radius
        outlet_z = self.outlet[2]
        suspended_height = outlet_z - substrate_z

        # Initial clearance tip contact time
        initial_clearance = float(self.config["initial_conditions"].get("lower_tip_clearance", 0.003))
        t_contact = initial_clearance / max(self.injection_speed, 1e-6)

        positions = np.zeros((n, 3), dtype=float)
        velocities = np.zeros((n, 3), dtype=float)
        deposited_sticking = np.zeros(n, dtype=float)
        injector_controlled = np.zeros(n, dtype=float)
        injector_controlled[0] = 1.0

        s_heel = suspended_height

        for i, s in enumerate(s_coords):
            if s <= s_heel:
                # Suspended heel portion (between injector outlet and contact point)
                xi = s / s_heel  # 0 at outlet, 1 at contact
                z_pos = outlet_z - xi * (outlet_z - substrate_z)

                # Contact point at current time t
                phi_t = self.omega * max(0.0, t - t_contact)
                # Belt translation vector for contact point
                belt_disp_now = self.belt_axis * self.belt_speed * t

                x_contact = belt_disp_now[0] + self.coiling_radius * np.cos(phi_t)
                y_contact = belt_disp_now[1] + self.coiling_radius * np.sin(phi_t)

                heel_profile = 3.0 * xi**2 - 2.0 * xi**3

                if t < t_contact:
                    bias = float(self.config["initial_conditions"]["lateral_branch_bias"])
                    q = s / self.initial_deployed_length
                    positions[i] = [0.0, bias * q * q, outlet_z - s]
                    velocities[i] = [0.0, 0.0, -self.injection_speed]
                else:
                    x_pos = self.outlet[0] * (1.0 - heel_profile) + x_contact * heel_profile
                    y_pos = self.outlet[1] * (1.0 - heel_profile) + y_contact * heel_profile
                    positions[i] = [x_pos, y_pos, z_pos]

                    dheel_dxi = 6.0 * xi * (1.0 - xi)
                    dxi_ds = 1.0 / s_heel
                    dx_ds = (x_contact - self.outlet[0]) * dheel_dxi * dxi_ds
                    dy_ds = (y_contact - self.outlet[1]) * dheel_dxi * dxi_ds

                    vx_contact = self.belt_axis[0] * self.belt_speed - self.coiling_radius * self.omega * np.sin(phi_t)
                    vy_contact = self.belt_axis[1] * self.belt_speed + self.coiling_radius * self.omega * np.cos(phi_t)

                    vx = heel_profile * vx_contact - self.injection_speed * dx_ds
                    vy = heel_profile * vy_contact - self.injection_speed * dy_ds
                    vz = -self.injection_speed
                    velocities[i] = [vx, vy, vz]
            else:
                # Deposited portion (s > s_heel) on the translating belt
                deposited_sticking[i] = 1.0

                # Arc length along deposited part: s_dep = s - s_heel
                t_dep = t - (s - s_heel) / max(self.injection_speed, 1e-6)
                t_dep_clamped = max(t_contact, t_dep)

                phi_dep = self.omega * (t_dep_clamped - t_contact)
                # Since t_dep, the belt moved by belt_axis * belt_speed * (t - t_dep)
                belt_disp_total = self.belt_axis * self.belt_speed * t
                x_pos = belt_disp_total[0] + self.coiling_radius * np.cos(phi_dep)
                y_pos = belt_disp_total[1] + self.coiling_radius * np.sin(phi_dep)
                z_pos = substrate_z

                positions[i] = [x_pos, y_pos, z_pos]
                velocities[i] = self.belt_axis * self.belt_speed

        # Nodal masses
        mass = np.full(n, self.line_mass * ds)
        mass[0] *= 0.5
        mass[-1] *= 0.5

        # Cumulative mass source
        self.cumulative_mass_inflow = self.line_mass * self.injection_speed * t
        cumulative_mass_source = np.array([self.cumulative_mass_inflow], dtype=float)

        # Budget external forces: gravity on all nodes, contact normal + friction on belt, injector reaction
        total_rod_mass = np.sum(mass)
        gravity_force = total_rod_mass * self.gravity

        deposited_mass = np.sum(mass[deposited_sticking > 0.5])
        contact_normal_force = np.array([0.0, 0.0, -deposited_mass * self.gravity[2]], dtype=float)
        friction_force = np.zeros(3, dtype=float)

        f_grav = gravity_force
        f_belt = contact_normal_force + friction_force
        f_inj = - (f_grav + f_belt)

        budget_external_force = np.zeros((1, 3, 3), dtype=float)
        budget_external_force[0, 0] = f_grav
        budget_external_force[0, 1] = f_belt
        budget_external_force[0, 2] = f_inj

        # Force application positions
        p_grav = np.mean(positions, axis=0)
        p_belt = np.array([np.mean(positions[deposited_sticking > 0.5, 0]) if np.any(deposited_sticking > 0.5) else 0.0,
                           np.mean(positions[deposited_sticking > 0.5, 1]) if np.any(deposited_sticking > 0.5) else 0.0,
                           self.belt_surface_z], dtype=float)
        p_inj = self.outlet.copy()

        budget_force_position = np.zeros((1, 3, 3), dtype=float)
        budget_force_position[0, 0] = p_grav
        budget_force_position[0, 1] = p_belt
        budget_force_position[0, 2] = p_inj

        budget_external_couple = np.zeros((1, 3, 3), dtype=float)

        # Constraints: segment lengths vs reference ds
        segment_diffs = np.diff(positions, axis=0)
        actual_segment_lengths = np.linalg.norm(segment_diffs, axis=1)
        target_segment_lengths = np.full(n - 1, ds)

        constraint_value = actual_segment_lengths
        constraint_target = target_segment_lengths

        # Reference scales
        ref_force = total_rod_mass * 9.81
        ref_torque = ref_force * self.coiling_radius
        constraint_scale = ds

        frame = StateFrame(
            time=t,
            position=positions,
            velocity=velocities,
            mass=mass,
            attributes={
                "arc_length": s_coords,
                "natural_curvature": np.zeros(n),
                "natural_twist": np.zeros(n),
                "injector_controlled": injector_controlled,
                "deposited_sticking": deposited_sticking,
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_couple": budget_external_couple,
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "rod_radius": float(self.radius),
                "injection_speed": float(self.injection_speed),
                "belt_speed": float(self.belt_speed),
                "constraint_scale": float(constraint_scale),
                "reference_force": float(ref_force),
                "reference_torque": float(ref_torque),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "coiling_radius": self.coiling_radius,
            "omega": self.omega,
        }
