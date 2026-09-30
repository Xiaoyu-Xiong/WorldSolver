"""Gravity-induced 3D bend-twist curling solver for curved elastic rod."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from worldsolver.contracts import StateFrame


def parallel_transport(u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Parallel transport vector u from tangent t1 to t2."""
    axis = np.cross(t1, t2)
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-10:
        u_trans = u.copy()
    else:
        axis_unit = axis / norm_axis
        dot_val = np.clip(np.dot(t1, t2), -1.0, 1.0)
        angle = np.arctan2(norm_axis, dot_val)
        u_trans = (u * np.cos(angle) +
                   np.cross(axis_unit, u) * np.sin(angle) +
                   axis_unit * np.dot(axis_unit, u) * (1.0 - np.cos(angle)))
    # Ensure orthogonality to t2
    u_trans -= np.dot(u_trans, t2) * t2
    norm = np.linalg.norm(u_trans)
    if norm > 1e-12:
        u_trans /= norm
    return u_trans


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rod_cfg = self.config["rod"]
        self.init_cfg = self.config["initial_conditions"]
        self.load_schedule = self.config.get("load_schedule", {})

        self.vertex_count = scene.vertex_count  # 129
        self.num_edges = self.vertex_count - 1  # 128

        self.length = float(self.rod_cfg["length"])  # 0.103 m
        self.rod_radius = float(self.rod_cfg["radius"])  # 0.0018 m
        self.density = float(self.rod_cfg["density"])  # 1267.0 kg/m^3
        self.natural_radius = float(self.rod_cfg["natural_radius"])  # 0.01754 m
        self.natural_curvature = float(self.rod_cfg["natural_curvature"])  # 57.02 rad/m

        # Cross-sectional area and moments of inertia
        self.area = np.pi * self.rod_radius**2
        self.I = np.pi * (self.rod_radius**4) / 4.0  # bending second moment of area
        self.J = 2.0 * self.I  # polar moment of inertia

        # Material properties from Charron et al. 2021
        # Silicone rubber: E = 1.25 MPa, nu = 0.5, G = E / 3.0
        self.E = 1.25e6  # Pa
        self.nu = 0.5
        self.G = self.E / (2.0 * (1.0 + self.nu))

        self.EI = self.E * self.I  # Bending stiffness
        self.GJ = self.G * self.J  # Torsional stiffness

        # Discretization along centerline
        self.ds0 = self.length / self.num_edges

        # Initial centerline positions from scene
        init_frame = scene.initial_state()
        self.pos = np.copy(init_frame.position)  # (N, 3)
        self.vel = np.zeros_like(self.pos)

        # Node masses
        self.mass = np.full(self.vertex_count, self.density * self.area * self.ds0)
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5
        self.total_mass = float(np.sum(self.mass))

        # Initial Bishop / reference frame u, v along edges
        self.u = np.zeros((self.num_edges, 3))
        self.v = np.zeros((self.num_edges, 3))
        self.theta = np.zeros(self.num_edges)

        # Compute initial reference frames and rest curvatures
        self._init_reference_frames()
        self.rest_kappa = np.zeros((self.vertex_count - 2, 2))
        self._compute_rest_curvatures()

        # Apply initial branch perturbation
        pert = np.asarray(self.init_cfg.get("branch_selection_perturbation", [0.0, 0.00015, 0.0]))
        ramp_pert = np.linspace(0.0, 1.0, self.vertex_count)[:, None]
        self.pos += pert * ramp_pert

        self.current_time = 0.0
        self.gravity_vec = np.asarray(self.config["gravity"], dtype=float)

        # Fixed root constraint
        self.root_pos = np.asarray(self.config["boundary_conditions"]["root"]["position"], dtype=float)
        self.root_tangent = np.asarray(self.config["boundary_conditions"]["root"]["tangent"], dtype=float)

    def _init_reference_frames(self) -> None:
        tangents = np.diff(self.pos, axis=0)
        t_edges = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

        # First edge normal
        t0 = t_edges[0]
        ref = np.array([0.0, 1.0, 0.0])
        u0 = np.cross(t0, ref)
        if np.linalg.norm(u0) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0])
            u0 = np.cross(t0, ref)
        u0 /= np.linalg.norm(u0)
        self.u[0] = u0
        self.v[0] = np.cross(t0, u0)

        for i in range(self.num_edges - 1):
            self.u[i+1] = parallel_transport(self.u[i], t_edges[i], t_edges[i+1])
            self.v[i+1] = np.cross(t_edges[i+1], self.u[i+1])

    def _compute_rest_curvatures(self) -> None:
        tangents = np.diff(self.pos, axis=0)
        t_edges = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

        for i in range(self.vertex_count - 2):
            t_prev = t_edges[i]
            t_next = t_edges[i+1]
            denom = 1.0 + np.dot(t_prev, t_next)
            if denom > 1e-6:
                kb = 2.0 * np.cross(t_prev, t_next) / denom
            else:
                kb = np.zeros(3)

            u_mid = 0.5 * (self.u[i] + self.u[i+1])
            v_mid = 0.5 * (self.v[i] + self.v[i+1])
            self.rest_kappa[i, 0] = np.dot(kb, v_mid)
            self.rest_kappa[i, 1] = -np.dot(kb, u_mid)

    def _get_gravity_factor(self, t: float) -> float:
        ramp_start = float(self.load_schedule.get("start_time", 0.0))
        ramp_full = float(self.load_schedule.get("full_load_time", 3.5))
        if t <= ramp_start:
            return 0.0
        if t >= ramp_full:
            return 1.0
        s = (t - ramp_start) / (ramp_full - ramp_start)
        return 0.5 * (1.0 - np.cos(np.pi * s))

    def _energy(self, x_flat: np.ndarray, g_acc: np.ndarray) -> float:
        # Full energy: bending + twist + stretching + gravity
        # x_flat contains positions pos[2:] (size (N-2)*3) and theta[1:] (size num_edges - 1)
        n_pos = (self.vertex_count - 2) * 3
        pos_free = x_flat[:n_pos].reshape((self.vertex_count - 2, 3))
        theta_free = x_flat[n_pos:]

        pos = np.empty((self.vertex_count, 3))
        pos[0] = self.root_pos
        pos[1] = self.root_pos + self.root_tangent * self.ds0
        pos[2:] = pos_free

        theta = np.empty(self.num_edges)
        theta[0] = 0.0
        theta[1:] = theta_free

        tangents = np.diff(pos, axis=0)
        edge_lens = np.linalg.norm(tangents, axis=1)
        t_edges = tangents / np.maximum(edge_lens[:, None], 1e-12)

        # 1. Stretch energy (high penalty for inextensibility)
        k_stretch = 5.0e4
        e_stretch = 0.5 * k_stretch * np.sum((edge_lens - self.ds0)**2)

        # 2. Parallel transport reference frame
        u_arr = np.empty((self.num_edges, 3))
        v_arr = np.empty((self.num_edges, 3))
        u_arr[0] = self.u[0]
        v_arr[0] = self.v[0]

        for i in range(self.num_edges - 1):
            u_arr[i+1] = parallel_transport(u_arr[i], t_edges[i], t_edges[i+1])
            v_arr[i+1] = np.cross(t_edges[i+1], u_arr[i+1])

        # 3. Bending and Twisting energy
        e_bend = 0.0
        e_twist = 0.0

        for i in range(self.vertex_count - 2):
            t_prev = t_edges[i]
            t_next = t_edges[i+1]
            denom = 1.0 + np.dot(t_prev, t_next)
            if denom > 1e-6:
                kb = 2.0 * np.cross(t_prev, t_next) / denom
            else:
                kb = np.zeros(3)

            u_mid = 0.5 * (u_arr[i] + u_arr[i+1])
            v_mid = 0.5 * (v_arr[i] + v_arr[i+1])

            th_mid = 0.5 * (theta[i] + theta[i+1])
            c_th = np.cos(th_mid)
            s_th = np.sin(th_mid)

            m1 = c_th * u_mid + s_th * v_mid
            m2 = -s_th * u_mid + c_th * v_mid

            kappa1 = np.dot(kb, m2)
            kappa2 = -np.dot(kb, m1)

            dk1 = kappa1 - self.rest_kappa[i, 0]
            dk2 = kappa2 - self.rest_kappa[i, 1]

            e_bend += 0.5 * (self.EI / self.ds0) * (dk1**2 + dk2**2)

            # Twist strain
            u_trans = parallel_transport(u_arr[i], t_prev, t_next)
            sin_m = np.dot(np.cross(u_trans, u_arr[i+1]), t_next)
            cos_m = np.dot(u_trans, u_arr[i+1])
            m_ref = np.arctan2(sin_m, cos_m)

            twist_strain = theta[i+1] - theta[i] + m_ref
            e_twist += 0.5 * (self.GJ / self.ds0) * (twist_strain**2)

        # 4. Gravity potential energy: - sum (m_i * g . x_i)
        e_grav = -np.sum(self.mass * np.dot(pos, g_acc))

        return float(e_stretch + e_bend + e_twist + e_grav)

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance simulation to target_time."""
        g_factor = self._get_gravity_factor(target_time)
        g_acc = self.gravity_vec * g_factor

        # Optimize equilibrium configuration
        x_init = np.concatenate([self.pos[2:].ravel(), self.theta[1:]])

        res = minimize(
            self._energy,
            x_init,
            args=(g_acc,),
            method="L-BFGS-B",
            options={"maxiter": 200, "gtol": 1e-5, "ftol": 1e-7}
        )

        n_pos = (self.vertex_count - 2) * 3
        pos_free = res.x[:n_pos].reshape((self.vertex_count - 2, 3))
        self.theta[1:] = res.x[n_pos:]

        new_pos = np.empty((self.vertex_count, 3))
        new_pos[0] = self.root_pos
        new_pos[1] = self.root_pos + self.root_tangent * self.ds0
        new_pos[2:] = pos_free

        dt = max(target_time - self.current_time, 1e-6)
        self.vel = (new_pos - self.pos) / dt
        self.pos = new_pos
        self.current_time = target_time

        # Update reference frames u, v
        tangents = np.diff(self.pos, axis=0)
        t_edges = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
        for i in range(self.num_edges - 1):
            self.u[i+1] = parallel_transport(self.u[i], t_edges[i], t_edges[i+1])
            self.v[i+1] = np.cross(t_edges[i+1], self.u[i+1])

        # Required fields
        mass_quantity = np.copy(self.mass)
        ext_forces = self.mass[:, None] * g_acc[None, :]
        budget_external_force = ext_forces[None, :, :]
        budget_force_position = self.pos[None, :, :]
        budget_external_couple = np.zeros((1, self.vertex_count, 3))

        c_val = np.r_[
            self.pos[0] - self.root_pos,
            (self.pos[1] - self.pos[0]) / self.ds0 - self.root_tangent,
            np.linalg.norm(np.diff(self.pos, axis=0), axis=1) - self.ds0
        ]
        c_target = np.zeros_like(c_val)
        cumulative_mass_source = np.array([0.0])

        ref_force = float(self.total_mass * 9.81)
        ref_torque = float(ref_force * self.length)
        c_scale = float(self.length)
        s = np.linspace(0.0, self.length, self.vertex_count)

        frame = StateFrame(
            time=float(target_time),
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            attributes={
                "arc_length": s,
                "natural_curvature": np.full(self.vertex_count, self.natural_curvature),
                "natural_twist": np.zeros(self.vertex_count),
                "is_clamped_root": np.r_[1.0, np.zeros(self.vertex_count - 1)],
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "budget_external_couple": budget_external_couple,
                "constraint_value": c_val,
                "constraint_target": c_target,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "rod_radius": self.rod_radius,
                "branch_selection_perturbation_y": float(self.init_cfg["branch_selection_perturbation"][1]),
                "constraint_scale": c_scale,
                "reference_force": ref_force,
                "reference_torque": ref_torque,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "vertex_count": self.vertex_count,
            "tip_position": self.pos[-1].tolist() if hasattr(self, "pos") else None,
        }

