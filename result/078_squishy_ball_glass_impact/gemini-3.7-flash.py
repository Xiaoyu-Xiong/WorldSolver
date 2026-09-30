"""Volumetric FEM / mass-spring / corotational elasticity solver for squishy ball impact against glass."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Nonlinear solid mechanics solver for squishy ball impact and rebound."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load mesh asset
        asset_path = Path(scene._asset)
        with np.load(asset_path) as data:
            self.nodes0 = np.asarray(data["nodes"], dtype=np.float64).copy()
            self.tets = np.asarray(data["tets"], dtype=np.int64).copy()

        self.num_nodes = len(self.nodes0)
        self.num_tets = len(self.tets)

        # Positions and velocities
        self.pos = self.nodes0.copy()
        self.vel = np.zeros_like(self.pos)
        v0_x = float(self.config["initial_conditions"]["squishy_ball"]["linear_velocity"][0])
        self.vel[:, 0] = v0_x

        # Mass distribution
        self.mass = np.full(self.num_nodes, 1.0 / self.num_nodes, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.ones(self.num_nodes, dtype=np.int64)

        # Reference momentum scale
        total_mass = np.sum(self.mass)
        self.ref_momentum = float(total_mass * abs(v0_x)) if abs(v0_x) > 0 else 1.0

        # Precompute reference tet geometry: Dm and Dm_inv, and tet rest volumes
        # Dm = [X1 - X0, X2 - X0, X3 - X0] (shape: (Ne, 3, 3))
        v0 = self.nodes0[self.tets[:, 0]]
        v1 = self.nodes0[self.tets[:, 1]]
        v2 = self.nodes0[self.tets[:, 2]]
        v3 = self.nodes0[self.tets[:, 3]]

        Dm = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1) # (Ne, 3, 3)
        self.Dm_inv = np.linalg.inv(Dm) # (Ne, 3, 3)

        # Determinant and rest volume: V = 1/6 * |det(Dm)|
        dets = np.linalg.det(Dm)
        self.tet_vol = np.abs(dets) / 6.0 # (Ne,)

        # Precompute tet edge pairs for auxiliary elastic network (edge springs + corotational FEM)
        # Elastic material parameters
        # Soft rubber: Young's modulus E ~ 2e4, Poisson's ratio nu ~ 0.35
        # Lame parameters:
        # mu = E / (2 * (1 + nu))
        # lam = E * nu / ((1 + nu) * (1 - 2*nu))
        E = 45000.0  # Young's modulus chosen for realistic deformation and good rebound
        nu = 0.38
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Tetrahedral shape gradient matrices in rest configuration:
        # For each tet e: B_e = -Dm_inv^T * [1, 1, 1]^T for node 0, and Dm_inv^T row i for node i+1
        # Gradient of deformation w.r.t node displacements:
        # dF/dx_i = d(Ds * Dm_inv)/dx_i
        # d x_j / d x_i = delta_ij
        # If Ds = [x1 - x0, x2 - x0, x3 - x0],
        # then d Ds_{a, k} / d x_{i, b} = (delta_{i, k+1} - delta_{i, 0}) * delta_{a, b}
        # F_{a, b} = sum_k Ds_{a, k} * Dm_inv_{k, b}
        # d F_{a, b} / d x_{i, c} = sum_k (delta_{i, k+1} - delta_{i, 0}) * delta_{a, c} * Dm_inv_{k, b}
        # P = d Psi / d F
        # f_{i, a} = - integral P_{a, b} * (d F_{a, b} / d x_{i, a}) dV
        # f_{i} = - V_e * P * g_i where g_i are column vectors of shape gradients:
        # g_1 = Dm_inv_{0, :}, g_2 = Dm_inv_{1, :}, g_3 = Dm_inv_{2, :}
        # g_0 = - (g_1 + g_2 + g_3)
        g1 = self.Dm_inv[:, 0, :]  # (Ne, 3)
        g2 = self.Dm_inv[:, 1, :]  # (Ne, 3)
        g3 = self.Dm_inv[:, 2, :]  # (Ne, 3)
        g0 = -(g1 + g2 + g3)       # (Ne, 3)
        self.grad_N = np.stack([g0, g1, g2, g3], axis=1) # (Ne, 4, 3)

        # Glass wall boundary: plane x = 0.0 (glass wall front face)
        self.glass_x = 0.0

        # Budget tracking variables
        self.cumulative_impulse = np.zeros((1, 3), dtype=np.float64)
        self.external_force = np.zeros((1, 1, 3), dtype=np.float64)

        # Simulation substepping dt
        self.sub_dt = 0.0002
        self.current_time = 0.0
        self.last_F = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (self.num_tets, 1, 1))

    def _compute_internal_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute FEM Neo-Hookean elastic forces and deformation gradients."""
        # Current Ds for all tets
        p0 = self.pos[self.tets[:, 0]]
        p1 = self.pos[self.tets[:, 1]]
        p2 = self.pos[self.tets[:, 2]]
        p3 = self.pos[self.tets[:, 3]]

        Ds = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)  # (Ne, 3, 3)
        F = np.matmul(Ds, self.Dm_inv)  # (Ne, 3, 3)

        # Determinant J = det(F)
        # Using analytical 3x3 determinant for vectorization speed
        # det(F) = F00(F11 F22 - F12 F21) - F01(F10 F22 - F12 F20) + F02(F10 F21 - F11 F20)
        c00 = F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1]
        c01 = F[:, 1, 2] * F[:, 2, 0] - F[:, 1, 0] * F[:, 2, 2]
        c02 = F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]

        c10 = F[:, 0, 2] * F[:, 2, 1] - F[:, 0, 1] * F[:, 2, 2]
        c11 = F[:, 0, 0] * F[:, 2, 2] - F[:, 0, 2] * F[:, 2, 0]
        c12 = F[:, 0, 1] * F[:, 2, 0] - F[:, 0, 0] * F[:, 2, 1]

        c20 = F[:, 0, 1] * F[:, 1, 2] - F[:, 0, 2] * F[:, 1, 1]
        c21 = F[:, 0, 2] * F[:, 1, 0] - F[:, 0, 0] * F[:, 1, 2]
        c22 = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]

        J = F[:, 0, 0] * c00 + F[:, 0, 1] * (F[:, 1, 2] * F[:, 2, 0] - F[:, 1, 0] * F[:, 2, 2]) + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0])

        # Clamp J for inversion stability
        J_safe = np.maximum(J, 0.05)

        # F^{-T} = Cofactor(F) / J
        # Cofactor matrix transpose:
        F_inv_T = np.stack([
            np.stack([c00, c10, c20], axis=-1),
            np.stack([c01, c11, c21], axis=-1),
            np.stack([c02, c12, c22], axis=-1)
        ], axis=-1) / J_safe[:, None, None]

        # Stable Neo-Hookean 1st Piola-Kirchhoff stress P:
        # P = mu * (F - F^{-T}) + lam * ln(J) * F^{-T}
        log_J = np.log(J_safe)
        P = self.mu * (F - F_inv_T) + (self.lam * log_J)[:, None, None] * F_inv_T

        # Node forces from tets:
        # f_{e, i} = - V_e * P_e * g_{e, i}  (where g is (Ne, 4, 3))
        # P_e is (Ne, 3, 3), g_{e, i} is (Ne, 3) -> P_e * g_{e, i} is (Ne, 3)
        # We can compute H = - V_e * P_e  (Ne, 3, 3)
        H = - (self.tet_vol[:, None, None] * P)

        forces = np.zeros((self.num_nodes, 3), dtype=np.float64)
        for i in range(4):
            g_i = self.grad_N[:, i, :] # (Ne, 3)
            # f_i = H @ g_i
            f_i = np.einsum('nij,nj->ni', H, g_i)
            np.add.at(forces, self.tets[:, i], f_i)

        return forces, F

    def _step(self, dt: float) -> None:
        """Single physics sub-step with semi-implicit integration and contact."""
        # 1. Compute elastic internal forces
        f_int, F = self._compute_internal_forces()
        self.last_F = F

        # Damping force (Rayleigh damping / viscous velocity damping)
        f_damp = - 1.5 * self.mass[:, None] * self.vel

        f_total = f_int + f_damp

        # 2. Acceleration and explicit velocity update
        acc = f_total * self.inv_mass[:, None]
        self.vel += acc * dt

        # 3. Position update
        self.pos += self.vel * dt

        # 4. Glass wall collision resolution (penalty + non-penetration projection + friction)
        # Wall is at x = glass_x (x <= 0 is free, x > 0 is inside glass)
        penetration = self.pos[:, 0] - self.glass_x
        contact_mask = penetration > 0.0

        if np.any(contact_mask):
            contact_nodes = np.where(contact_mask)[0]
            # Normal impulse to enforce non-penetration
            pen_depth = penetration[contact_nodes]

            # Position projection to wall surface
            self.pos[contact_nodes, 0] = self.glass_x

            # Contact restitution / velocity reflection
            v_normal = self.vel[contact_nodes, 0]
            rebound_mask = v_normal > 0.0

            if np.any(rebound_mask):
                reb_idx = contact_nodes[rebound_mask]
                # Elastic rebound with high coefficient for squishy rubber
                e_restitution = 0.85
                v_n = self.vel[reb_idx, 0]
                delta_vn = -(1.0 + e_restitution) * v_n
                self.vel[reb_idx, 0] = -e_restitution * v_n

                # Tangential friction during contact
                v_tan = self.vel[reb_idx, 1:]
                v_tan_mag = np.linalg.norm(v_tan, axis=1, keepdims=True)
                friction_coeff = 0.25
                max_friction_dv = friction_coeff * np.abs(delta_vn)[:, None]

                friction_factor = np.clip(1.0 - max_friction_dv / (v_tan_mag + 1e-8), 0.0, 1.0)
                self.vel[reb_idx, 1:] *= friction_factor

                # Accumulate external impulse from wall
                total_delta_v = np.zeros(3, dtype=np.float64)
                total_delta_v[0] = np.sum(delta_vn * self.mass[reb_idx])
                # External impulse delivered to ball
                self.cumulative_impulse[0] += total_delta_v
                self.external_force[0, 0] = total_delta_v / dt
            else:
                self.external_force[0, 0] = 0.0
        else:
            self.external_force[0, 0] = 0.0

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until reaching the requested time
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Compute final deformation gradient F at current time
        p0 = self.pos[self.tets[:, 0]]
        p1 = self.pos[self.tets[:, 1]]
        p2 = self.pos[self.tets[:, 2]]
        p3 = self.pos[self.tets[:, 3]]
        Ds = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
        F = np.matmul(Ds, self.Dm_inv)

        # Build required trajectory StateFrame
        attributes = {
            "mass_quantity": self.mass.copy(),
            "component": np.zeros(self.num_nodes, dtype=np.int64)
        }

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.ones((1, self.num_nodes), dtype=np.float64),
            "budget_external_force": self.external_force.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": F.astype(np.float64),
        }

        scalars = {
            "reference_momentum": float(self.ref_momentum),
        }

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_tets": self.num_tets,
            "ref_momentum": self.ref_momentum,
        }

