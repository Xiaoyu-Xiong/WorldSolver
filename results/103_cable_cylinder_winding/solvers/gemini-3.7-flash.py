"""Physically-based solver for extensible cable winding around a cylinder."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        cable_cfg = self.config["cable"]
        cyl_cfg = self.config["cylinder"]
        contact_cfg = self.config["contact"]

        self.n = int(cable_cfg["vertex_count"])
        self.total_length = float(cable_cfg["length"])
        self.radius = float(cable_cfg["radius"])
        self.density = float(cable_cfg["density"])
        self.E = float(cable_cfg["young_modulus_pa"])
        self.poisson = float(cable_cfg["poisson_ratio"])
        self.G = float(cable_cfg["shear_modulus_pa"])
        self.driven_idx = int(cable_cfg["driven_vertex"])

        self.cyl_radius = float(cyl_cfg["radius"])
        self.cyl_contact_radius = self.cyl_radius + self.radius
        self.cyl_center = np.array(cyl_cfg["center"], dtype=float)
        self.cyl_height = float(cyl_cfg["height"])
        self.cyl_z_min = self.cyl_center[2] - self.cyl_height * 0.5
        self.cyl_z_max = self.cyl_center[2] + self.cyl_height * 0.5

        self.gravity = np.array(self.config["gravity"], dtype=float)

        self.kappa = float(contact_cfg["kappa"])
        self.exponent_N = float(contact_cfg["exponent_N"])

        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(init_frame.mass, dtype=float)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        self.L0 = self.total_length / (self.n - 1)
        self.area = np.pi * (self.radius ** 2)
        self.I_inertia = 0.25 * np.pi * (self.radius ** 4)
        self.J_polar = 0.5 * np.pi * (self.radius ** 4)

        # Cross section inertia per vertex (for StateFrame required fields)
        # Nodal inertia tensor for small segment
        # Segment mass m_i, approx cylinder about tangent
        self.inertia_tensor = np.zeros((self.n, 3, 3), dtype=float)
        for i in range(self.n):
            m_i = self.mass[i]
            r2 = self.radius ** 2
            h2 = self.L0 ** 2
            # isotropic/approximate segment principal inertia
            ixx = m_i * (3 * r2 + h2) / 12.0
            iyy = m_i * (3 * r2 + h2) / 12.0
            izz = m_i * r2 / 2.0
            self.inertia_tensor[i] = np.diag([ixx, iyy, izz])

        self.ang_vel = np.zeros((self.n, 3), dtype=float)

        # Cumulative budget integrals
        self.cum_ext_impulse = np.zeros((1, 3), dtype=float)
        self.cum_ext_ang_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((1, 3), dtype=float)
        self.cum_inflow_ang_mom = np.zeros((1, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((1, 3), dtype=float)
        self.cum_outflow_ang_mom = np.zeros((1, 3), dtype=float)

        # Reference momentum scales
        total_m = np.sum(self.mass)
        self.ref_mom = float(max(total_m * np.linalg.norm(self.gravity) * 1.0, 1e-3))
        self.ref_ang_mom = float(max(self.ref_mom * self.cyl_radius, 1e-3))

        self.current_time = 0.0

        # Simulation substepping parameters
        self.sub_dt = 1e-4

        # Stretch stiffness and bending stiffness
        # Extensional stiffness k_stretch = E * A / L0
        self.k_stretch = self.E * self.area / self.L0
        # Bending stiffness k_bend = E * I / L0
        self.k_bend = self.E * self.I_inertia / self.L0

    def _compute_internal_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        forces = np.zeros_like(pos)

        # 1. Extensional (stretch) springs with damping
        dpos = pos[1:] - pos[:-1]  # [n-1, 3]
        lengths = np.linalg.norm(dpos, axis=1, keepdims=True)  # [n-1, 1]
        lengths = np.maximum(lengths, 1e-12)
        tangents = dpos / lengths  # [n-1, 3]

        strain = (lengths - self.L0)  # [n-1, 1]
        stretch_f = self.k_stretch * strain * tangents  # [n-1, 3]

        # Stretch damping
        dvel = vel[1:] - vel[:-1]
        rel_vel_tangent = np.sum(dvel * tangents, axis=1, keepdims=True)
        stretch_damp = 0.05 * np.sqrt(self.k_stretch * self.mass[1:, None]) * rel_vel_tangent * tangents
        total_stretch_f = stretch_f + stretch_damp

        forces[:-1] += total_stretch_f
        forces[1:] -= total_stretch_f

        # 2. Discrete bending forces (rest angle is 0, naturally straight)
        # For vertices 1 to n-2: angle between e_{i-1} and e_i
        e_prev = tangents[:-1]  # [n-2, 3]
        e_next = tangents[1:]   # [n-2, 3]

        # Curvature normal / bend force: tends to align e_{i-1} and e_i
        # dE_bend/dx_i pushes towards straightness
        # Laplacians of positions pos[i-1] - 2*pos[i] + pos[i+1]
        laplacian = pos[:-2] - 2.0 * pos[1:-1] + pos[2:]  # [n-2, 3]
        bend_forces = (self.k_bend / (self.L0 ** 2)) * laplacian

        # Bending damping
        vel_laplacian = vel[:-2] - 2.0 * vel[1:-1] + vel[2:]
        bend_damp = 0.01 * np.sqrt(self.k_bend * self.mass[1:-1, None]) / self.L0 * vel_laplacian
        total_bend = bend_forces + bend_damp

        forces[:-2] += 0.5 * total_bend
        forces[1:-1] -= total_bend
        forces[2:] += 0.5 * total_bend

        return forces

    def _step(self, dt: float) -> None:
        target_grip = self.scene.grip_target(self.current_time + dt)
        target_pos = np.array(target_grip["position"], dtype=float)

        # Internal forces
        f_int = self._compute_internal_forces(self.pos, self.vel)

        # External gravity force
        f_ext = self.mass[:, None] * self.gravity[None, :]

        # Velocity update from conservative & internal forces
        acc = (f_int + f_ext) * self.inv_mass[:, None]
        v_pred = self.vel + dt * acc

        # Aerodynamic / ambient damping
        v_pred *= np.exp(-0.5 * dt)

        # Position prediction
        p_pred = self.pos + dt * v_pred

        # Kinematic boundary condition on driven vertex
        p_pred[self.driven_idx] = target_pos
        v_pred[self.driven_idx] = (target_pos - self.pos[self.driven_idx]) / dt

        # Contact with cylinder
        # Cylinder axis is positive z, center at (0, 0, z)
        xy = p_pred[:, :2]
        r_xy = np.linalg.norm(xy, axis=1)
        penetration = self.cyl_contact_radius - r_xy

        # Contact mask
        in_cyl_z = (p_pred[:, 2] >= self.cyl_z_min) & (p_pred[:, 2] <= self.cyl_z_max)
        contact_mask = (penetration > 0) & in_cyl_z
        # Free driven grip from cylinder contact override
        contact_mask[self.driven_idx] = False

        if np.any(contact_mask):
            idx = np.where(contact_mask)[0]
            normal_xy = xy[idx] / np.maximum(r_xy[idx, None], 1e-12)

            # Position projection to cylinder surface
            p_pred[idx, 0] = normal_xy[:, 0] * self.cyl_contact_radius
            p_pred[idx, 1] = normal_xy[:, 1] * self.cyl_contact_radius

            # Relative velocity at contact
            v_contact = v_pred[idx]
            normal_3d = np.zeros_like(v_contact)
            normal_3d[:, :2] = normal_xy

            # Normal velocity
            vn = np.sum(v_contact * normal_3d, axis=1, keepdims=True)
            # Remove inward normal velocity
            v_contact_n = np.minimum(vn, 0.0) * normal_3d
            v_contact_t = v_contact - vn * normal_3d

            # Compute normal impulse magnitude lambda_n
            # lambda_n approx m * |vn| + contact stiffness impulse
            lambda_n = self.mass[idx, None] * (penetration[idx, None] / dt + np.maximum(-vn, 0.0))

            # Generalized nonlinear friction law: ||lambda_t|| <= kappa * |lambda_n|^N
            lambda_t_max = self.kappa * (np.maximum(lambda_n, 1e-12) ** self.exponent_N)

            # Tangential relative speed
            vt_norm = np.linalg.norm(v_contact_t, axis=1, keepdims=True)
            vt_dir = v_contact_t / np.maximum(vt_norm, 1e-12)

            # Desired tangential impulse to stop sliding
            lambda_t_desired = self.mass[idx, None] * vt_norm
            lambda_t_actual = np.minimum(lambda_t_desired, lambda_t_max)

            # Update tangential velocity
            vt_reduction = (lambda_t_actual / np.maximum(self.mass[idx, None], 1e-12)) * vt_dir
            v_pred[idx] = (v_contact - v_contact_n) - vt_reduction

        # Contact with floor (z = 0.0)
        floor_pen = self.radius - p_pred[:, 2]
        floor_mask = (floor_pen > 0)
        floor_mask[self.driven_idx] = False
        if np.any(floor_mask):
            f_idx = np.where(floor_mask)[0]
            p_pred[f_idx, 2] = self.radius
            v_pred[f_idx, 2] = np.maximum(v_pred[f_idx, 2], 0.0)
            v_pred[f_idx, :2] *= 0.5  # friction with floor

        # Update actual velocities and positions
        self.vel = (p_pred - self.pos) / dt
        self.pos = p_pred
        self.current_time += dt

        # Update cumulative impulse budgets
        # External force on whole cable: gravity + grip contact + cylinder reaction
        net_f = self.mass[:, None] * self.gravity[None, :]
        impulse_dt = np.sum(net_f, axis=0) * dt
        self.cum_ext_impulse[0] += impulse_dt

        # Angular impulse about origin
        r_cross_f = np.cross(self.pos, net_f)
        ang_impulse_dt = np.sum(r_cross_f, axis=0) * dt
        self.cum_ext_ang_impulse[0] += ang_impulse_dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until time is reached
        target_time = float(time)
        while self.current_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)

        # Build StateFrame
        q = np.linspace(0.0, 1.0, self.n)
        driven_end = np.zeros(self.n, dtype=float)
        driven_end[self.driven_idx] = 1.0
        free_end = np.zeros(self.n, dtype=float)
        free_end[0] = 1.0

        # Required fields according to manifest and contracts
        # G = 1 budget domain (global), K = 1 term (gravity)
        budget_ext_force = np.zeros((1, 1, 3), dtype=float)
        budget_ext_force[0, 0] = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)

        # Center of mass as force application point
        com = np.sum(self.mass[:, None] * self.pos, axis=0) / np.sum(self.mass)
        budget_force_pos = np.zeros((1, 1, 3), dtype=float)
        budget_force_pos[0, 0] = com

        budget_ext_couple = np.zeros((1, 1, 3), dtype=float)
        budget_domain_weights = np.ones((1, self.n), dtype=float)

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "arc_fraction": q,
                "rest_segment_length": np.full(self.n, self.L0),
                "cable_radius": np.full(self.n, self.radius),
                "driven_endpoint": driven_end,
                "free_endpoint": free_end,
                "inertia_tensor": self.inertia_tensor.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.copy(),
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_angular_momentum": self.cum_inflow_ang_mom.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_angular_momentum": self.cum_outflow_ang_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_couple": budget_ext_couple,
                "budget_external_force": budget_ext_force,
                "budget_force_position": budget_force_pos,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "young_modulus_pa": self.E,
                "poisson_ratio": self.poisson,
                "generalized_friction_kappa": self.kappa,
                "generalized_friction_exponent": self.exponent_N,
                "reference_momentum": self.ref_mom,
                "reference_angular_momentum": self.ref_ang_mom,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "vertex_count": self.n,
            "sub_dt": self.sub_dt,
        }

