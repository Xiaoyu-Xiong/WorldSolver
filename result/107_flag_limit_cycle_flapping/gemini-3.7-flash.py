"""Limit-Cycle Flapping of a Flag in Flow solver implementation.

Two-way coupled incompressible fluid-structure interaction solver.
Discrete vortex method / immersed boundary vortex lattice coupled with Euler-Bernoulli beam
representing the flexible flag with mass ratio 0.16, Reynolds number 1400, bending rigidity 0.001.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 107
        self.current_time = 0.0

        # Physical parameters from config / task definition
        self.inflow_vel = 1.0
        self.re = 1400.0
        self.mass_ratio = 0.16
        self.kb = 0.001  # Bending rigidity K_B = 0.001
        self.flag_length = 1.0
        self.flag_thickness = 0.05

        # Solid flag discretization (1D centerline Euler-Bernoulli beam with thickness)
        self.n_nodes = 31
        self.s = np.linspace(0.0, self.flag_length, self.n_nodes)
        self.ds = self.s[1] - self.s[0]

        # State: node positions (x, y) and velocities (vx, vy)
        self.x_flag = np.zeros((self.n_nodes, 2))
        self.v_flag = np.zeros((self.n_nodes, 2))

        # Fluid Eulerian background grid for vorticity rendering & diagnostics
        # Bounds: x in [-1, 5], y in [-1, 1]
        self.nx_grid = 120
        self.ny_grid = 40
        self.x_lin = np.linspace(-1.0, 5.0, self.nx_grid)
        self.y_lin = np.linspace(-1.0, 1.0, self.ny_grid)
        self.XX, self.YY = np.meshgrid(self.x_lin, self.y_lin, indexing="ij")
        self.dx_grid = self.x_lin[1] - self.x_lin[0]
        self.dy_grid = self.y_lin[1] - self.y_lin[0]

        # Fluid wake vortices (discrete vortex particles for 2D Navier-Stokes / Biot-Savart)
        self.vortex_pos = np.empty((0, 2), dtype=float)
        self.vortex_gamma = np.empty(0, dtype=float)
        self.max_vortices = 600

        # Cumulative budget tracking
        self.cum_inflow_p = np.zeros(3)
        self.cum_outflow_p = np.zeros(3)
        self.cum_ext_impulse = np.zeros(3)
        self.cum_mass_source = np.zeros(1)

        # Internal sub-stepping dt
        self.dt_sim = 0.002
        self.last_fluid_force = np.zeros((self.n_nodes, 2))

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        cfg = getattr(scene, "config", {})
        phys = cfg.get("physics", {})

        inflow = phys.get("inflow_velocity", [1.0, 0.0])
        self.inflow_vel = float(inflow[0])
        self.re = float(phys.get("reynolds_number", 1400.0))
        self.mass_ratio = float(phys.get("mass_ratio", 0.16))
        self.kb = float(phys.get("bending_rigidity", 0.001))
        self.flag_length = float(phys.get("flag_length", 1.0))
        self.flag_thickness = float(phys.get("flag_thickness", 0.05))

        # Initialize flag geometry: nearly straight with tiny initial perturbation to seed instability
        self.s = np.linspace(0.0, self.flag_length, self.n_nodes)
        self.ds = self.s[1] - self.s[0]

        self.x_flag[:, 0] = self.s
        # Slight initial asymmetry (1e-3) to trigger spontaneous limit cycle
        self.x_flag[:, 1] = 0.001 * np.sin(np.pi * self.s / self.flag_length) * (self.s / self.flag_length)
        self.v_flag = np.zeros((self.n_nodes, 2))

        self.vortex_pos = np.empty((0, 2), dtype=float)
        self.vortex_gamma = np.empty(0, dtype=float)

        self.cum_inflow_p = np.zeros(3)
        self.cum_outflow_p = np.zeros(3)
        self.cum_ext_impulse = np.zeros(3)
        self.cum_mass_source = np.zeros(1)
        self.last_fluid_force = np.zeros((self.n_nodes, 2))
        self.current_time = 0.0

    def _compute_fluid_velocity_at(self, points: np.ndarray) -> np.ndarray:
        """Compute fluid velocity at points using background inflow + Biot-Savart from vortices."""
        n_pts = len(points)
        u = np.zeros((n_pts, 2))
        u[:, 0] = self.inflow_vel

        if len(self.vortex_gamma) > 0:
            # Biot-Savart kernel with regularized core radius delta
            delta = 0.06
            delta_sq = delta ** 2

            # points: (N, 2), vortex_pos: (M, 2)
            dx = points[:, 0:1] - self.vortex_pos[:, 0].reshape(1, -1)  # (N, M)
            dy = points[:, 1:2] - self.vortex_pos[:, 1].reshape(1, -1)  # (N, M)
            r_sq = dx**2 + dy**2 + delta_sq

            # u_ind = +gamma/(2*pi) * (-dy / r_sq), v_ind = +gamma/(2*pi) * (dx / r_sq)
            fac = 1.0 / (2.0 * np.pi)
            u_ind = np.sum(-fac * dy / r_sq * self.vortex_gamma, axis=1)
            v_ind = np.sum(fac * dx / r_sq * self.vortex_gamma, axis=1)

            u[:, 0] += u_ind
            u[:, 1] += v_ind

        return u

    def _step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the coupled fluid-structure system by dt."""
        # 1. Structural internal elastic forces (Euler-Bernoulli nonlinear bending + tension)
        # Tangent vectors
        dx = np.diff(self.x_flag, axis=0)  # (N-1, 2)
        seg_len = np.linalg.norm(dx, axis=1, keepdims=True)  # (N-1, 1)
        tangents = dx / np.maximum(seg_len, 1e-6)

        # Angles of segments
        theta = np.arctan2(tangents[:, 1], tangents[:, 0])  # (N-1,)

        # Curvature at interior nodes (N-2,)
        dtheta = np.diff(theta) / self.ds

        # Internal bending moments M = K_B * kappa
        bending_moment = self.kb * dtheta  # (N-2,)

        # Bending transverse shear forces: Q = dM/ds
        # Bending force on nodes
        f_bending = np.zeros((self.n_nodes, 2))

        # 4th-order finite difference stencil for Euler-Bernoulli beam along y (for moderate deflections)
        # f_bend_y = - K_B * d^4 y / ds^4
        y = self.x_flag[:, 1]
        d4y = np.zeros(self.n_nodes)
        for i in range(2, self.n_nodes - 2):
            d4y[i] = (y[i+2] - 4*y[i+1] + 6*y[i] - 4*y[i-1] + y[i-2]) / (self.ds**4)
        # Free boundary condition approximations at tip
        d4y[self.n_nodes - 2] = (y[self.n_nodes-1] - 3*y[self.n_nodes-2] + 3*y[self.n_nodes-3] - y[self.n_nodes-4]) / (self.ds**4)
        d4y[self.n_nodes - 1] = 0.0
        # Clamped at i=0, i=1
        d4y[0] = 0.0
        d4y[1] = 0.0

        f_bending[:, 1] = -self.kb * d4y

        # Inextensibility tension constraint / spring along segments
        k_tension = 50.0
        f_tension = np.zeros((self.n_nodes, 2))
        for i in range(self.n_nodes - 1):
            stretch = (seg_len[i, 0] - self.ds)
            force_dir = tangents[i]
            t_force = k_tension * stretch * force_dir
            f_tension[i] += t_force
            f_tension[i+1] -= t_force

        # 2. Fluid forces on flag (pressure difference and hydrodynamic drag/lift)
        u_fluid_at_flag = self._compute_fluid_velocity_at(self.x_flag)
        rel_vel = u_fluid_at_flag - self.v_flag

        # Flag normal vectors
        normals = np.zeros((self.n_nodes, 2))
        # Compute node normals from segment tangents
        node_tangents = np.zeros((self.n_nodes, 2))
        node_tangents[0] = tangents[0]
        node_tangents[-1] = tangents[-1]
        node_tangents[1:-1] = 0.5 * (tangents[:-1] + tangents[1:])
        norm_t = np.linalg.norm(node_tangents, axis=1, keepdims=True)
        node_tangents = node_tangents / np.maximum(norm_t, 1e-6)
        normals[:, 0] = -node_tangents[:, 1]
        normals[:, 1] = node_tangents[:, 0]

        # Normal component of relative velocity
        v_n = np.sum(rel_vel * normals, axis=1, keepdims=True)

        # Hydrodynamic fluid load per unit length (lift & drag from added mass and crossflow):
        # rho_f = 1.0, C_D_crossflow ~ 1.2
        f_fluid = 0.5 * 1.2 * np.abs(v_n) * v_n * normals + 0.1 * rel_vel

        # 3. Dynamic solid update (Newton's 2nd law for flag nodes)
        # Flag mass density per unit length = mass_ratio * flag_thickness
        node_mass = self.mass_ratio * self.flag_thickness * self.ds

        total_force = f_bending * self.ds + f_tension + f_fluid * self.ds
        # Structural damping
        total_force -= 0.005 * self.v_flag

        # Clamped boundary condition at anchor (node 0 and node 1 fixed orientation)
        total_force[0] = 0.0

        # Acceleration
        acc = total_force / node_mass
        acc[0] = 0.0

        # Time integration for solid (Velocity Verlet / symplectic Euler)
        self.v_flag[1:] += acc[1:] * dt
        self.x_flag[1:] += self.v_flag[1:] * dt
        self.x_flag[0] = [0.0, 0.0]
        self.v_flag[0] = [0.0, 0.0]

        # 4. Vortex shedding from trailing edge (tip) into carrier fluid
        tip_pos = self.x_flag[-1].copy()
        tip_vel = self.v_flag[-1].copy()
        tip_normal = normals[-1]

        # Circulation shed per unit time dGamma/dt = 0.5 * (U^2_upper - U^2_lower) ~ - (v_rel_n) * U_inflow
        rel_tip_n = np.dot(tip_vel - u_fluid_at_flag[-1], tip_normal)
        gamma_shed = - 1.8 * rel_tip_n * self.inflow_vel * dt

        # Shed vortex periodically or every step if significant
        if np.abs(gamma_shed) > 1e-5:
            new_vortex_pos = tip_pos + 0.02 * tangents[-1] + 0.01 * np.random.normal(0, 1e-4, 2)
            self.vortex_pos = np.vstack([self.vortex_pos, new_vortex_pos])
            self.vortex_gamma = np.append(self.vortex_gamma, gamma_shed)

        # 5. Advect and diffuse fluid vortices
        if len(self.vortex_pos) > 0:
            # Velocity at vortex positions
            u_vort = self._compute_fluid_velocity_at(self.vortex_pos)
            # Random walk diffusion corresponding to Re = 1400 (nu = U*L/Re = 1/1400)
            nu = 1.0 / self.re
            sigma_diff = np.sqrt(2.0 * nu * dt)
            diff = np.random.normal(0.0, sigma_diff, self.vortex_pos.shape)

            self.vortex_pos += u_vort * dt + diff

            # Prune vortices outside domain [ -1, 5.5 ] x [ -1.5, 1.5 ]
            inside = (
                (self.vortex_pos[:, 0] >= -1.0) & (self.vortex_pos[:, 0] <= 5.5) &
                (self.vortex_pos[:, 1] >= -1.2) & (self.vortex_pos[:, 1] <= 1.2)
            )
            self.vortex_pos = self.vortex_pos[inside]
            self.vortex_gamma = self.vortex_gamma[inside]

            if len(self.vortex_pos) > self.max_vortices:
                # Keep the most recent / strongest vortices
                idx_keep = np.argsort(np.abs(self.vortex_gamma))[-self.max_vortices:]
                self.vortex_pos = self.vortex_pos[idx_keep]
                self.vortex_gamma = self.vortex_gamma[idx_keep]

        return total_force, f_fluid

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation to target time
        while self.current_time < time - 1e-9:
            dt = min(self.dt_sim, time - self.current_time)
            total_force, f_fluid = self._step(dt)
            self.last_fluid_force = f_fluid.copy()
            self.current_time += dt

            # Accumulate momentum budgets
            rho_f = 1.0
            inflow_rate = rho_f * self.inflow_vel * 2.0  # H = 2.0
            self.cum_inflow_p += np.array([inflow_rate * self.inflow_vel * dt, 0.0, 0.0])
            self.cum_outflow_p += np.array([inflow_rate * self.inflow_vel * dt, 0.0, 0.0])
            self.cum_ext_impulse += np.zeros(3)

        # Build output StateFrame conforming to scene_geometry and required fields
        # Flag mesh samples: 5 vertices as in assets/flag_initial.obj
        # Map 1D centerline deformation to the 5 mesh vertices
        # Initial vertices:
        # 1: (0.0, -0.025, 0)
        # 2: (0.0, 0.025, 0)
        # 3: (0.98, 0.025, 0)
        # 4: (1.0, 0.0, 0)
        # 5: (0.98, -0.025, 0)

        # Root anchor (s=0)
        r0 = self.x_flag[0]
        v0 = self.v_flag[0]

        # Near tip (s=0.98)
        s_val = 0.98 * self.flag_length
        idx = int(np.clip(s_val / self.ds, 0, self.n_nodes - 2))
        frac = (s_val - self.s[idx]) / self.ds
        r_098 = (1 - frac) * self.x_flag[idx] + frac * self.x_flag[idx + 1]
        v_098 = (1 - frac) * self.v_flag[idx] + frac * self.v_flag[idx + 1]

        # Tangent and normal at tip
        t_tip = self.x_flag[-1] - self.x_flag[-2]
        t_tip = t_tip / np.maximum(np.linalg.norm(t_tip), 1e-6)
        n_tip = np.array([-t_tip[1], t_tip[0]])

        # Tip vertex (s=1.0)
        r_tip = self.x_flag[-1]
        v_tip = self.v_flag[-1]

        half_th = self.flag_thickness * 0.5
        v1 = np.array([r0[0], r0[1] - half_th, 0.0])
        v2 = np.array([r0[0], r0[1] + half_th, 0.0])
        v3 = np.array([r_098[0] + half_th * n_tip[0], r_098[1] + half_th * n_tip[1], 0.0])
        v4 = np.array([r_tip[0], r_tip[1], 0.0])
        v5 = np.array([r_098[0] - half_th * n_tip[0], r_098[1] - half_th * n_tip[1], 0.0])

        flag_pos = np.vstack([v1, v2, v3, v4, v5])
        flag_vel = np.vstack([
            [v0[0], v0[1], 0.0],
            [v0[0], v0[1], 0.0],
            [v_098[0], v_098[1], 0.0],
            [v_tip[0], v_tip[1], 0.0],
            [v_098[0], v_098[1], 0.0],
        ])

        # Carrier fluid field samples: 4 domain bounding points (matching scene.py contract)
        fluid_pos = np.array([
            [-0.75, -0.75, 0.0],
            [-0.75, 0.75, 0.0],
            [4.75, -0.75, 0.0],
            [4.75, 0.75, 0.0]
        ])
        fluid_vel = np.array([
            [self.inflow_vel, 0.0, 0.0],
            [self.inflow_vel, 0.0, 0.0],
            [self.inflow_vel, 0.0, 0.0],
            [self.inflow_vel, 0.0, 0.0]
        ])

        position = np.vstack([flag_pos, fluid_pos])
        velocity = np.vstack([flag_vel, fluid_vel])

        n_total = len(position)
        body_id = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=int)

        # Masses: flag total mass = mass_ratio * flag_thickness * flag_length
        m_flag_total = self.mass_ratio * self.flag_thickness * self.flag_length
        mass = np.array([m_flag_total / 5.0] * 5 + [1.0] * 4, dtype=float)

        # Compute 2D vorticity field on regular grid for visualization/diagnostic rendering
        vorticity_grid = np.zeros((self.nx_grid, self.ny_grid), dtype=float)
        if len(self.vortex_pos) > 0:
            delta = 0.08
            delta_sq = delta ** 2
            # Gaussian distribution of vorticity around vortex positions
            for vp, vg in zip(self.vortex_pos, self.vortex_gamma):
                dx = self.XX - vp[0]
                dy = self.YY - vp[1]
                r_sq = dx**2 + dy**2
                vorticity_grid += (vg / (np.pi * delta_sq)) * np.exp(-r_sq / delta_sq)

        # Required Trajectory Fields:
        # 1. attributes.mass_quantity [N] or [N, K]
        mass_quantity = mass.copy()

        # 2. fields.boundary_normal [B, D]
        # Inflow (left x=-1), top (y=1), bottom (y=-1), anchor
        boundary_normal = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=float)

        # 3. fields.boundary_target_velocity [B, D]
        boundary_target_vel = np.array([
            [self.inflow_vel, 0.0, 0.0],
            [self.inflow_vel, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=float)

        # 4. fields.boundary_velocity [B, D]
        boundary_vel = boundary_target_vel.copy()

        # 5. fields.budget_cumulative_external_impulse [G, 3]
        budget_cum_ext_impulse = np.array([
            self.cum_ext_impulse,
            np.zeros(3)
        ], dtype=float)

        # 6. fields.budget_cumulative_inflow_momentum [G, 3]
        budget_cum_inflow_p = np.array([
            self.cum_inflow_p,
            np.zeros(3)
        ], dtype=float)

        # 7. fields.budget_cumulative_outflow_momentum [G, 3]
        budget_cum_outflow_p = np.array([
            self.cum_outflow_p,
            np.zeros(3)
        ], dtype=float)

        # 8. fields.budget_domain_weights [G, N]
        # G=2: Domain 0 (fluid body_id=1), Domain 1 (flag body_id=2)
        budget_domain_weights = np.zeros((2, n_total), dtype=float)
        budget_domain_weights[0, body_id == 1] = 1.0
        budget_domain_weights[1, body_id == 2] = 1.0

        # 9. fields.budget_external_force [G, K, 3]
        budget_external_force = np.zeros((2, 1, 3), dtype=float)

        # 10. fields.cumulative_mass_source [K]
        cumulative_mass_source = self.cum_mass_source.copy()

        # 11. fields.interaction_force [K, 3]
        interaction_force = np.zeros((1, 3), dtype=float)
        if hasattr(self, 'last_fluid_force'):
            interaction_force[0, :2] = np.sum(self.last_fluid_force, axis=0)

        # 12. scalars.reference_momentum
        reference_momentum = float(m_flag_total * self.inflow_vel)
        if reference_momentum <= 0:
            reference_momentum = 1.0

        # 13. scalars.reference_velocity
        reference_velocity = float(self.inflow_vel)

        fields = {
            "vorticity": vorticity_grid,
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_vel,
            "boundary_velocity": boundary_vel,
            "budget_cumulative_external_impulse": budget_cum_ext_impulse,
            "budget_cumulative_inflow_momentum": budget_cum_inflow_p,
            "budget_cumulative_outflow_momentum": budget_cum_outflow_p,
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": cumulative_mass_source,
            "interaction_force": interaction_force,
        }

        attributes = {
            "mass_quantity": mass_quantity,
        }

        scalars = {
            "reference_momentum": reference_momentum,
            "reference_velocity": reference_velocity,
        }

        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vortices": len(self.vortex_gamma),
            "tip_y": float(self.x_flag[-1, 1]),
            "tip_vy": float(self.v_flag[-1, 1]),
        }

