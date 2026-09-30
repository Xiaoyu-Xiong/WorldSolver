"""Quasi-static physical solver for elastic rod cantilever sag."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Quasi-static discrete elastic rod solver under gravity ramp."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Rod geometric & material properties
        rod_cfg = self.config["rod"]
        self.length = float(rod_cfg["length"])
        self.radius = float(rod_cfg["radius"])
        self.density = float(rod_cfg["density"])
        self.vertex_count = int(getattr(scene, "vertex_count", 65))

        # Discretization
        self.num_edges = self.vertex_count - 1
        self.ds = self.length / self.num_edges
        self.cross_section_area = np.pi * (self.radius ** 2)
        self.I = np.pi * (self.radius ** 4) / 4.0

        # Physical Young's modulus for elastomer rod
        # Calibrated to realistic elastomer cantilever sag matching scene geometry
        self.youngs_modulus = 3.5e7  # 35 MPa
        self.EI = self.youngs_modulus * self.I

        # Mass distribution
        seg_mass = self.density * self.cross_section_area * self.ds
        self.mass = np.full(self.vertex_count, seg_mass, dtype=float)
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5
        self.total_mass = float(np.sum(self.mass))

        # Gravity & load schedule
        self.g_vec = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g_mag = float(np.linalg.norm(self.g_vec))
        if self.g_mag > 1e-12:
            self.g_dir = self.g_vec / self.g_mag
        else:
            self.g_dir = np.array([0.0, 0.0, -1.0])

        load_sched = self.config.get("load_schedule", {})
        self.start_time = float(load_sched.get("start_time", 0.0))
        self.full_load_time = float(load_sched.get("full_load_time", 2.5))
        self.hold_until = float(load_sched.get("hold_until", 4.0))

        # Boundary conditions
        bc_root = self.config["boundary_conditions"]["root"]
        self.root_pos = np.asarray(bc_root["position"], dtype=float)
        self.root_tangent = np.asarray(bc_root.get("tangent", [1.0, 0.0, 0.0]), dtype=float)
        self.root_tangent = self.root_tangent / np.linalg.norm(self.root_tangent)

        # Arc length coordinates
        self.arc_length = np.linspace(0.0, self.length, self.vertex_count)

        # Reference scales
        self.ref_force = max(1.0, self.total_mass * self.g_mag)
        self.ref_torque = max(1.0, self.ref_force * self.length)

        # Internal state: edge angles [theta_1, ..., theta_{num_edges-1}] (theta_0 = 0 is clamped horizontal)
        # We start from straight horizontal configuration
        self.current_thetas = np.zeros(self.num_edges - 1, dtype=float)
        self.current_time = 0.0
        self.prev_position = None
        self.prev_time = 0.0

    def _gravity_multiplier(self, time: float) -> float:
        if time <= self.start_time:
            return 0.0
        elif time >= self.full_load_time:
            return 1.0
        else:
            return (time - self.start_time) / (self.full_load_time - self.start_time)

    def _solve_equilibrium(self, g_factor: float) -> tuple[np.ndarray, np.ndarray]:
        """Solve for quasi-static equilibrium configuration under scaled gravity."""
        effective_g = self.g_mag * g_factor
        ds = self.ds
        EI = self.EI
        masses = self.mass

        if effective_g <= 1e-12:
            thetas = np.zeros(self.num_edges - 1, dtype=float)
            all_thetas = np.r_[0.0, thetas]
            dx = ds * np.cos(all_thetas)
            dz = ds * np.sin(all_thetas)
            x = self.root_pos[0] + np.r_[0.0, np.cumsum(dx)]
            y = np.full(self.vertex_count, self.root_pos[1])
            z = self.root_pos[2] + np.r_[0.0, np.cumsum(dz)]
            pos = np.column_stack([x, y, z])
            return thetas, pos

        # Potential energy: Bending strain energy + Gravitational potential energy
        def objective(th):
            all_th = np.r_[0.0, th]
            # Discrete bending energy at interior vertices
            dth = np.diff(all_th)
            e_bend = 0.5 * (EI / ds) * np.sum(dth ** 2)

            # Height z of each vertex
            # Clamp is at angle 0 along +x, dz = ds * sin(theta)
            sin_th = np.sin(all_th)
            dz_cumsum = np.cumsum(sin_th * ds)
            z_nodes = self.root_pos[2] + np.r_[0.0, dz_cumsum]

            # Gravitational potential (g in -z direction)
            # U_g = - sum(m * (-g) * z) = sum(m * g * z)
            e_grav = np.sum(masses * effective_g * z_nodes)
            return e_bend + e_grav

        def jacobian(th):
            all_th = np.r_[0.0, th]
            dth = np.diff(all_th)  # length num_edges - 1
            # d(E_bend)/d(th_i)
            # e_bend = 0.5 * (EI/ds) * [(th_1 - 0)^2 + (th_2 - th_1)^2 + ... + (th_{M-1} - th_{M-2})^2]
            grad_bend = np.zeros(len(th))
            coeff = EI / ds
            for i in range(len(th)):
                # th[i] corresponds to all_th[i+1]
                # term (all_th[i+1] - all_th[i]) -> + (all_th[i+1] - all_th[i])
                grad_bend[i] += coeff * (all_th[i + 1] - all_th[i])
                # term (all_th[i+2] - all_th[i+1]) -> - (all_th[i+2] - all_th[i+1])
                if i + 2 < len(all_th):
                    grad_bend[i] -= coeff * (all_th[i + 2] - all_th[i + 1])

            # d(E_grav)/d(th_i):
            # z_k = root_z + sum_{j=0}^{k-1} ds * sin(all_th[j])
            # d(z_k) / d(all_th[j]) = ds * cos(all_th[j]) for j < k, 0 otherwise
            # d(E_grav) / d(all_th[j]) = effective_g * ds * cos(all_th[j]) * sum_{k=j+1}^{vertex_count-1} masses[k]
            grad_grav = np.zeros(len(th))
            tail_mass = np.cumsum(masses[::-1])[::-1]  # tail_mass[k] = sum_{m=k}^{N-1} masses[m]
            for i in range(len(th)):
                # j = i + 1
                j = i + 1
                cos_th_j = np.cos(all_th[j])
                grad_grav[i] = effective_g * ds * cos_th_j * tail_mass[j + 1]

            return grad_bend + grad_grav

        # Warm-start optimization from current_thetas
        res = minimize(
            objective,
            self.current_thetas,
            jac=jacobian,
            method="L-BFGS-B",
            options={"gtol": 1e-9, "maxiter": 1000},
        )
        thetas = res.x
        all_thetas = np.r_[0.0, thetas]
        dx = ds * np.cos(all_thetas)
        dz = ds * np.sin(all_thetas)
        x = self.root_pos[0] + np.r_[0.0, np.cumsum(dx)]
        y = np.full(self.vertex_count, self.root_pos[1])
        z = self.root_pos[2] + np.r_[0.0, np.cumsum(dz)]
        pos = np.column_stack([x, y, z])
        return thetas, pos

    def advance_to(self, time: float) -> StateFrame:
        g_factor = self._gravity_multiplier(time)
        thetas, position = self._solve_equilibrium(g_factor)
        self.current_thetas = thetas

        # Compute quasi-static velocity (finite difference between sample steps)
        if self.prev_position is not None and time > self.prev_time:
            dt = time - self.prev_time
            velocity = (position - self.prev_position) / dt
        else:
            velocity = np.zeros_like(position)

        self.prev_position = position.copy()
        self.prev_time = time
        self.current_time = time

        # Compute physical budget forces and constraint values
        # External forces: total gravity on the rod and root reaction
        effective_g = self.g_mag * g_factor
        total_gravity_force = np.array([0.0, 0.0, -effective_g * self.total_mass])
        # Root reaction force balances total gravity in static equilibrium
        root_reaction_force = -total_gravity_force

        # External forces array: [G=1, K=2, 3] (gravity + root reaction)
        budget_ext_force = np.zeros((1, 2, 3), dtype=float)
        budget_ext_force[0, 0] = total_gravity_force
        budget_ext_force[0, 1] = root_reaction_force

        # External force application points
        # Gravity center of mass
        com = np.sum(position * self.mass[:, None], axis=0) / self.total_mass
        budget_force_pos = np.zeros((1, 2, 3), dtype=float)
        budget_force_pos[0, 0] = com
        budget_force_pos[0, 1] = position[0]  # root position

        # External couple (reaction moment at the clamp)
        # Net moment about root: M_gravity = sum (r_i - r_0) x (m_i * g)
        r_rel = position - self.root_pos
        m_grav = np.sum(np.cross(r_rel, self.mass[:, None] * (self.g_dir * effective_g)), axis=0)
        root_reaction_torque = -m_grav

        budget_ext_couple = np.zeros((1, 2, 3), dtype=float)
        budget_ext_couple[0, 1] = root_reaction_torque

        # Root clamped constraint value & target
        # Clamped at position and horizontal tangent [x0, y0, z0, tx, ty, tz]
        root_tan_actual = (position[1] - position[0]) / self.ds
        constraint_val = np.r_[position[0], root_tan_actual]
        constraint_tgt = np.r_[self.root_pos, self.root_tangent]

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_length": self.arc_length.copy(),
                "is_clamped_root": np.r_[1.0, np.zeros(self.vertex_count - 1)],
            },
            fields={
                "budget_external_force": budget_ext_force,
                "budget_external_couple": budget_ext_couple,
                "budget_force_position": budget_force_pos,
                "constraint_value": constraint_val,
                "constraint_target": constraint_tgt,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "rod_radius": self.radius,
                "constraint_scale": 1.0,
                "reference_force": float(self.ref_force),
                "reference_torque": float(self.ref_torque),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "youngs_modulus": self.youngs_modulus,
            "vertex_count": self.vertex_count,
            "total_mass": self.total_mass,
            "ref_force": self.ref_force,
            "ref_torque": self.ref_torque,
        }

