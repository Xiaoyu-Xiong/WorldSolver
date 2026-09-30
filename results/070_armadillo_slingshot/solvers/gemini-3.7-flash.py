"""Deterministic solver for 070_armadillo_slingshot.

Simulates 3D elastodynamics on the Armadillo solid mesh using Projective Dynamics /
corotational linear elasticity with fixed grasp boundary conditions, kinematic pull,
and release recoil.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config = None
        self.points_rest = None
        self.tetrahedra = None
        self.num_nodes = 0
        self.num_elements = 0
        self.dt = 0.002
        self.current_time = 0.0

        # State
        self.x = None
        self.v = None
        self.mass = None
        self.inv_mass = None

        # Attributes
        self.fixed_grasp = None
        self.pulled_patch = None
        self.free_mask = None

        # Boundary condition params
        self.pull_start_s = 0.0
        self.pull_peak_s = 0.9
        self.release_s = 1.0
        self.target_start = np.array([0.0, 0.0, 1.25], dtype=np.float64)
        self.target_peak = np.array([0.0, 0.0, 3.65], dtype=np.float64)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Elasticity precomputations
        self.invDm = None
        self.element_volumes = None
        self.element_mu = 0.0
        self.element_lambda = 0.0
        self.element_weight = 0.0
        self.element_active = None

        # Pre-factored solver system
        self.solver_factor = None
        self.A_mat = None
        self.free_indices = None
        self.fixed_indices = None
        self.B_list = []  # for projective projection

        # Budget tracking
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)
        self.reference_momentum = 1.0

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config

        init_frame = scene.initial_state()
        self.points_rest = np.asarray(init_frame.position, dtype=np.float64).copy()
        self.num_nodes = len(self.points_rest)

        # Load solid mesh
        data = np.load("assets/armadillo_solid.npz")
        self.tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int32)
        self.num_elements = len(self.tetrahedra)

        # Boundary condition masks
        self.fixed_grasp = np.asarray(init_frame.attributes["fixed_grasp"], dtype=np.float64) > 0.5
        self.pulled_patch = np.asarray(init_frame.attributes["pulled_patch"], dtype=np.float64) > 0.5

        # Timeline
        timeline = self.config.get("event_timeline", {})
        self.pull_start_s = float(timeline.get("pull_start_s", 0.0))
        self.pull_peak_s = float(timeline.get("pull_peak_s", 0.9))
        self.release_s = float(timeline.get("release_s", 1.0))

        # Gravity
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Boundary condition parameters
        bcs = self.config.get("boundary_conditions", {})
        temp_pull = bcs.get("temporary_pull", {})
        self.target_start = np.asarray(temp_pull.get("target_start", [0.0, 0.0, 1.25]), dtype=np.float64)
        self.target_peak = np.asarray(temp_pull.get("target_peak", [0.0, 0.0, 3.65]), dtype=np.float64)

        # Material parameters
        mat = self.config.get("material_reference", {})
        density = float(mat.get("density_kg_m3", 1000.0))
        E = float(mat.get("youngs_modulus_pa", 45000.0))
        nu = float(mat.get("poisson_ratio", 0.45))

        # Lame parameters
        self.element_mu = E / (2.0 * (1.0 + nu))
        self.element_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Precompute tet geometry
        Dm = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for i in range(3):
            Dm[:, :, i] = self.points_rest[self.tetrahedra[:, i + 1]] - self.points_rest[self.tetrahedra[:, 0]]
        detDm = np.linalg.det(Dm)
        self.element_volumes = np.abs(detDm) / 6.0
        self.element_active = np.abs(detDm) > 1e-12

        self.invDm = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for e in range(self.num_elements):
            if self.element_active[e]:
                self.invDm[e] = np.linalg.inv(Dm[e])

        # Compute lump mass per node from density and tet volumes
        node_mass = np.zeros(self.num_nodes, dtype=np.float64)
        for e in range(self.num_elements):
            if self.element_active[e]:
                m_e = self.element_volumes[e] * density / 4.0
                node_mass[self.tetrahedra[e]] += m_e

        # Fallback for any zero-mass nodes
        mean_mass = np.mean(node_mass[node_mass > 0]) if np.any(node_mass > 0) else 1.0
        node_mass[node_mass <= 0] = mean_mass
        self.mass = node_mass
        self.inv_mass = 1.0 / self.mass

        # Reference momentum scale
        total_mass = np.sum(self.mass)
        self.reference_momentum = max(total_mass * np.linalg.norm(self.gravity) * 1.0, 1.0)

        # Time stepping
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.substeps = 10
        self.dt = self.sample_dt / self.substeps

        # State initialization
        self.x = self.points_rest.copy()
        self.v = np.zeros_like(self.x)
        self.current_time = 0.0

        # Build Projective Dynamics / Corotational stiffness
        # For each active element e:
        # deformation gradient F = Ds * invDm
        # Ds = [x1-x0, x2-x0, x3-x0] = x * G_e
        # G_e is 4x3 matrix such that x_elem @ G_e = Ds
        # Specifically, G_e = [-1 -1 -1; 1 0 0; 0 1 0; 0 0 1]
        # Then F = x_elem @ (G_e @ invDm) = x_elem @ K_e, where K_e = G_e @ invDm is 4x3.
        # Strain energy potential per element: 0.5 * w_e * ||F - R_e||_F^2
        # where w_e = 2 * mu * vol_e (corotational energy).
        # Derivative w.r.t x_elem: w_e * (x_elem @ K_e - R_e) @ K_e.T = w_e * x_elem @ (K_e @ K_e.T) - w_e * R_e @ K_e.T
        # Thus element stiffness matrix S_e = w_e * (K_e @ K_e.T) is a 4x4 scalar matrix (acting on each xyz coord).
        # Right hand side from element: w_e * (K_e @ R_e.T) which is 4x3.

        G = np.array([[-1.0, -1.0, -1.0],
                      [ 1.0,  0.0,  0.0],
                      [ 0.0,  1.0,  0.0],
                      [ 0.0,  0.0,  1.0]], dtype=np.float64)

        self.K_list = np.zeros((self.num_elements, 4, 3), dtype=np.float64)
        self.weight_list = np.zeros(self.num_elements, dtype=np.float64)

        I_idx = []
        J_idx = []
        V_idx = []

        for e in range(self.num_elements):
            if not self.element_active[e]:
                continue
            Ke = G @ self.invDm[e]  # 4x3
            self.K_list[e] = Ke
            we = 2.0 * self.element_mu * self.element_volumes[e]
            self.weight_list[e] = we
            Se = we * (Ke @ Ke.T)  # 4x4
            elem_nodes = self.tetrahedra[e]
            for i in range(4):
                ni = elem_nodes[i]
                for j in range(4):
                    nj = elem_nodes[j]
                    I_idx.append(ni)
                    J_idx.append(nj)
                    V_idx.append(Se[i, j])

        # Global stiffness matrix L (scalar Laplacian for 1 coordinate)
        self.L = sp.coo_matrix((V_idx, (I_idx, J_idx)), shape=(self.num_nodes, self.num_nodes)).tocsr()
        self.M = sp.diags(self.mass)

    def _get_target_pull_pos(self, t: float) -> np.ndarray:
        if t <= self.pull_start_s:
            return self.target_start.copy()
        elif t >= self.pull_peak_s:
            return self.target_peak.copy()
        else:
            s = (t - self.pull_start_s) / (self.pull_peak_s - self.pull_start_s)
            # smoothstep
            s = s * s * (3.0 - 2.0 * s)
            return (1.0 - s) * self.target_start + s * self.target_peak

    def _step(self, dt: float, current_t: float) -> None:
        """One implicit/projective dynamics time step."""
        is_pulling = (current_t < self.release_s)

        # Determine Dirichlet boundary conditions for this step
        is_fixed = self.fixed_grasp.copy()
        target_pos = self.points_rest.copy()

        if is_pulling:
            target_pull = self._get_target_pull_pos(current_t + dt)
            # Center of pulled patch at rest
            pulled_rest_center = np.mean(self.points_rest[self.pulled_patch], axis=0)
            offset = target_pull - pulled_rest_center
            target_pos[self.pulled_patch] = self.points_rest[self.pulled_patch] + offset
            is_fixed = is_fixed | self.pulled_patch

        free_mask = ~is_fixed
        free_indices = np.where(free_mask)[0]
        fixed_indices = np.where(is_fixed)[0]

        # External forces (gravity)
        f_ext = self.mass[:, None] * self.gravity[None, :]

        # Inertial / predicted positions
        # x_pred = x + dt * v + dt^2 * inv_mass * f_ext
        damping = 0.9995  # light numerical stability damping
        v_pred = self.v * damping + dt * self.gravity[None, :]
        x_pred = self.x + dt * v_pred

        # Enforce fixed constraints on prediction
        x_pred[is_fixed] = target_pos[is_fixed]
        x_curr = x_pred.copy()

        # System matrix for free vertices:
        # (M + dt^2 * L)_ff * x_f = M_ff * x_pred_f + dt^2 * (b(R)_f - L_fc * x_c)
        # We assemble the left-hand matrix and factorize / solve with conjugate gradient or pre-factor
        A_ff = (self.M + (dt**2) * self.L)[free_indices, :][:, free_indices].tocsr()
        L_fc = self.L[free_indices, :][:, fixed_indices]

        # Precompute right hand side inertia part
        rhs_inertia = self.mass[free_indices, None] * x_pred[free_indices] - (dt**2) * (L_fc @ target_pos[fixed_indices])

        # Projective Dynamics local-global solver iterations
        num_pd_iters = 8
        for _ in range(num_pd_iters):
            # Local step: compute best rotation R_e for each element
            # F_e = x_elem @ K_e (4x3 @ 3x3 not directly; x_elem is 4x3, K_e is 4x3 => F_e = x_elem.T @ K_e is 3x3)
            # Actually: Ds = [x1-x0, x2-x0, x3-x0] (3x3), F = Ds @ invDm
            # local RHS contribution per element:
            # b_e = we * (Ke @ R_e.T) where Ke is 4x3, R_e is 3x3 => b_e is 4x3
            rhs_b = np.zeros((self.num_nodes, 3), dtype=np.float64)

            # Vectorized local step
            nodes = self.tetrahedra  # (Ne, 4)
            x_elems = x_curr[nodes]  # (Ne, 4, 3)
            # Ds = x1-x0, x2-x0, x3-x0
            Ds = np.stack([x_elems[:, 1] - x_elems[:, 0],
                           x_elems[:, 2] - x_elems[:, 0],
                           x_elems[:, 3] - x_elems[:, 0]], axis=2)  # (Ne, 3, 3)

            # F = Ds @ invDm
            F = np.matmul(Ds, self.invDm)  # (Ne, 3, 3)

            # SVD of F for active elements to get closest rotation R = U @ V.T
            # For speed, compute SVD on active elements
            active_idx = np.where(self.element_active)[0]
            U, S, Vt = np.linalg.svd(F[active_idx])
            # Handle reflections
            det_UV = np.linalg.det(np.matmul(U, Vt))
            Vt[:, 2, :] *= det_UV[:, None]
            R = np.matmul(U, Vt)  # (N_act, 3, 3)

            # Accumulate rhs_b: for each elem, b_e = we * (Ke @ R.T) (4x3)
            # Ke is (N_act, 4, 3), R is (N_act, 3, 3)
            Ke_act = self.K_list[active_idx]  # (N_act, 4, 3)
            we_act = self.weight_list[active_idx, None, None]  # (N_act, 1, 1)
            b_act = we_act * np.matmul(Ke_act, R.transpose(0, 2, 1))  # (N_act, 4, 3)

            # Scatter to rhs_b
            for i in range(4):
                np.add.at(rhs_b, nodes[active_idx, i], b_act[:, i, :])

            # Global step: solve A_ff @ x_f = rhs_inertia + dt^2 * rhs_b[free_indices]
            rhs_total = rhs_inertia + (dt**2) * rhs_b[free_indices]

            # Solve for x, y, z coordinates using CG
            for c in range(3):
                x_sol, _ = spla.cg(A_ff, rhs_total[:, c], x0=x_curr[free_indices, c], rtol=1e-5, maxiter=25)
                x_curr[free_indices, c] = x_sol

        # Update velocities and positions
        self.v = (x_curr - self.x) / dt
        self.v[is_fixed] = 0.0
        self.x = x_curr.copy()

        # Update budget impulses
        # Total external impulse on system = gravity impulse + reaction impulse at fixed boundaries
        net_f_ext = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
        self.cumulative_external_impulse[0] += net_f_ext * dt

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation in fixed sub-steps until current_time >= time
        eps = 1e-9
        while self.current_time < time - eps:
            dt_step = min(self.dt, time - self.current_time)
            self._step(dt_step, self.current_time)
            self.current_time += dt_step

        # Compute element deformation gradients
        nodes = self.tetrahedra
        x_elems = self.x[nodes]
        Ds = np.stack([x_elems[:, 1] - x_elems[:, 0],
                       x_elems[:, 2] - x_elems[:, 0],
                       x_elems[:, 3] - x_elems[:, 0]], axis=2)
        F = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        active_idx = np.where(self.element_active)[0]
        F[active_idx] = np.matmul(Ds[active_idx], self.invDm[active_idx])
        # Inactive elements get identity
        inactive_idx = np.where(~self.element_active)[0]
        F[inactive_idx] = np.eye(3, dtype=np.float64)

        # Budget fields
        G = 1  # 1 domain (global)
        N = self.num_nodes
        K = 1  # 1 material
        domain_weights = np.ones((G, N), dtype=np.float64)

        # External forces on the domain
        gravity_force = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
        budget_external_force = gravity_force.reshape(G, K, 3)

        attributes = {
            "mass_quantity": self.mass.copy(),
            "fixed_grasp": self.fixed_grasp.astype(np.float64),
            "pulled_patch": self.pulled_patch.astype(np.float64),
        }

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
            "budget_domain_weights": domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "element_deformation_gradient": F,
        }

        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "density_kg_m3": 1000.0,
        }

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "current_time": self.current_time,
        }

