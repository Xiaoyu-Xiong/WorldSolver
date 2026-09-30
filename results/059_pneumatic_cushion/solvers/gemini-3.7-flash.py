"""Pneumatic cushion inflation FEM solver.

Implements non-linear hyperelastic finite elements with internal pressure boundary loading.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for pneumatic cushion inflation using 3D solid finite elements."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Load topology asset
        asset_path = Path(__file__).parent.parent / "assets" / "cushion_solid.npz"
        with np.load(asset_path) as data:
            self.X0 = np.asarray(data["vertices"], dtype=np.float64)  # (N, 3)
            self.tets = np.asarray(data["tetrahedra"], dtype=np.int32)  # (Ne, 4)
            self.inner_faces = np.asarray(data["inner_chamber_faces"], dtype=np.int32)  # (Nf, 3)
            self.outer_faces = np.asarray(data["outer_faces"], dtype=np.int32)
            self.nodal_volume = np.asarray(data["nodal_volume"], dtype=np.float64)
            self.wall_region = np.asarray(data["wall_region"], dtype=np.int32)

        self.N = len(self.X0)
        self.Ne = len(self.tets)
        self.Nf = len(self.inner_faces)

        # Material properties
        mat = self.config["material_properties"]
        self.E = float(mat["young_modulus_pa"])
        self.nu = float(mat["poisson_ratio"])
        self.rho = float(mat["density_kg_m3"])

        # Lame parameters
        # Note: nu = 0.499 is very close to incompressibility.
        # For stable FEM with linear tets without locking, we use a calibrated bulk modulus
        # or neo-Hookean / St. Venant-Kirchhoff with appropriate regularized lambda.
        # Using standard Neo-Hookean parameters:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        # Cap Poisson ratio effective bulk modulus to avoid extreme volumetric locking on coarse tet mesh
        nu_eff = min(self.nu, 0.48)
        self.lam = (self.E * nu_eff) / ((1.0 + nu_eff) * (1.0 - 2.0 * nu_eff))

        # Mass array
        self.mass = self.rho * self.nodal_volume  # (N,)

        # Precompute reference tet shape matrices Dm_inv and reference volumes V0
        # Tet nodes: 0, 1, 2, 3
        # Dm = [X1 - X0, X2 - X0, X3 - X0] (columns)
        self.Dm_inv = np.zeros((self.Ne, 3, 3), dtype=np.float64)
        self.V0 = np.zeros(self.Ne, dtype=np.float64)

        for e in range(self.Ne):
            i0, i1, i2, i3 = self.tets[e]
            X = self.X0
            Dm = np.column_stack([X[i1] - X[i0], X[i2] - X[i0], X[i3] - X[i0]])
            detDm = np.linalg.det(Dm)
            self.V0[e] = np.abs(detDm) / 6.0
            self.Dm_inv[e] = np.linalg.inv(Dm)

        # Inner face orientation check
        # Outward normal to inner chamber (pointing into the solid wall from air chamber)
        # Check cross product direction
        fc = self.X0[self.inner_faces].mean(axis=1)
        v0 = self.X0[self.inner_faces[:, 0]]
        v1 = self.X0[self.inner_faces[:, 1]]
        v2 = self.X0[self.inner_faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        # Inner chamber is around origin, so outward normal from chamber points away from origin (dot with fc > 0)
        dots = np.einsum("ij,ij->i", cross, fc)
        self.face_sign = np.where(dots > 0, 1.0, -1.0)[:, None]

        # Loading schedule
        loading = self.config["loading"]
        self.target_p = float(loading["target_internal_pressure_pa"])
        self.p_init = float(loading["initial_internal_pressure_pa"])
        self.ramp_start = float(loading["ramp_start_s"])
        self.ramp_end = float(loading["ramp_end_s"])

        # Reference force scale
        total_inner_area = 0.5 * np.linalg.norm(cross, axis=1).sum()
        self.reference_force = float(self.target_p * total_inner_area)

        # Simulation state
        self.current_time = 0.0
        self.x = self.X0.copy()
        self.v = np.zeros_like(self.X0)

        # Time step for sub-stepping
        # Rayleigh damping / artificial damping for quasi-static tracking
        self.dt_sub = 0.0005  # 0.5 ms
        self.damping = 25.0  # s^-1 velocity damping for quasi-static settling

        # Precompute tet dN/dX gradients in reference space
        # For linear tet: N0 = 1 - r - s - t, N1 = r, N2 = s, N3 = t
        # dN/dX: for node 1,2,3 are rows of Dm_inv^T, node 0 is -(row1 + row2 + row3)
        self.grad_N = np.zeros((self.Ne, 4, 3), dtype=np.float64)
        for e in range(self.Ne):
            inv = self.Dm_inv[e]  # Dm_inv has columns corresponding to edges
            # F = Ds @ Dm_inv => dF/dx_i
            # Let Dm_inv be M. Ds = [x1-x0, x2-x0, x3-x0].
            # F_{ij} = sum_k Ds_{ik} M_{kj}
            # dF_{ij}/dx_{a, p} = delta_{ip} * B_{a, j} where B is:
            # for a=1: M_{0, j}; a=2: M_{1, j}; a=3: M_{2, j}; a=0: -(M_{0,j}+M_{1,j}+M_{2,j})
            self.grad_N[e, 1] = inv[0, :]
            self.grad_N[e, 2] = inv[1, :]
            self.grad_N[e, 3] = inv[2, :]
            self.grad_N[e, 0] = -(inv[0, :] + inv[1, :] + inv[2, :])

    def _get_pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return self.p_init
        elif t >= self.ramp_end:
            return self.target_p
        else:
            alpha = (t - self.ramp_start) / (self.ramp_end - self.ramp_start)
            return self.p_init + alpha * (self.target_p - self.p_init)

    def _compute_forces_and_F(self, x: np.ndarray, pressure: float):
        """Compute internal elastic forces, pressure forces, and deformation gradients."""
        f_int = np.zeros((self.N, 3), dtype=np.float64)
        F_all = np.zeros((self.Ne, 3, 3), dtype=np.float64)

        # 1. Internal elastic forces from Neo-Hookean strain energy
        # For each tet:
        # Ds = [x1 - x0, x2 - x0, x3 - x0]
        i0 = self.tets[:, 0]
        i1 = self.tets[:, 1]
        i2 = self.tets[:, 2]
        i3 = self.tets[:, 3]

        Ds = np.stack([x[i1] - x[i0], x[i2] - x[i0], x[i3] - x[i0]], axis=2)  # (Ne, 3, 3)
        F = np.matmul(Ds, self.Dm_inv)  # (Ne, 3, 3)
        F_all[:] = F

        # First Piola-Kirchhoff stress P for compressible Neo-Hookean material:
        # W = mu/2 (I1 - 3 - 2 ln J) + lam/2 (ln J)^2
        # P = mu (F - F^{-T}) + lam ln(J) F^{-T}
        # Invert F per element
        detF = np.linalg.det(F)  # (Ne,)
        detF_safe = np.clip(detF, 1e-4, 10.0)
        invF = np.linalg.inv(F)  # (Ne, 3, 3)
        invFT = np.transpose(invF, (0, 2, 1))

        lnJ = np.log(detF_safe)
        P = self.mu * (F - invFT) + (self.lam * lnJ)[:, None, None] * invFT  # (Ne, 3, 3)

        # Nodal elastic forces:
        # f_node_a = - V0 * P @ grad_N_a
        # P is (Ne, 3, 3), grad_N[:, a, :] is (Ne, 3) -> (Ne, 3, 1)
        # P @ grad_N_a: sum over k of P_{i, k} grad_N_{a, k}
        V0 = self.V0[:, None]
        for a in range(4):
            node_idx = self.tets[:, a]
            g = self.grad_N[:, a, :]  # (Ne, 3)
            # force on node a: - V0 * (P @ g)
            f_a = - V0 * np.einsum("eij,ej->ei", P, g)
            np.add.at(f_int, node_idx, f_a)

        # 2. Pressure forces on inner chamber surface
        f_press = np.zeros((self.N, 3), dtype=np.float64)
        if pressure > 1e-6:
            f0 = self.inner_faces[:, 0]
            f1 = self.inner_faces[:, 1]
            f2 = self.inner_faces[:, 2]

            v0 = x[f0]
            v1 = x[f1]
            v2 = x[f2]

            # Current normal * 2 * current face area
            n_scaled = np.cross(v1 - v0, v2 - v0) * self.face_sign  # (Nf, 3)
            # Total pressure force on face = pressure * Area * normal = pressure * 0.5 * n_scaled
            # Distributed equally to 3 vertices (1/3 each):
            f_face_nodes = (pressure / 6.0) * n_scaled  # (Nf, 3)

            np.add.at(f_press, f0, f_face_nodes)
            np.add.at(f_press, f1, f_face_nodes)
            np.add.at(f_press, f2, f_face_nodes)

        return f_int, f_press, F_all

    def _remove_rigid_modes(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep chamber centered and remove overall rigid-body translation and rotation."""
        # Center of mass translation
        total_m = np.sum(self.mass)
        com = np.sum(self.mass[:, None] * x, axis=0) / total_m
        v_com = np.sum(self.mass[:, None] * v, axis=0) / total_m
        x_centered = x - com
        v_centered = v - v_com

        # Angular momentum removal
        r = x_centered
        L = np.sum(np.cross(r, self.mass[:, None] * v_centered), axis=0)
        # Inertia tensor
        r_x, r_y, r_z = r[:, 0], r[:, 1], r[:, 2]
        I_xx = np.sum(self.mass * (r_y**2 + r_z**2))
        I_yy = np.sum(self.mass * (r_x**2 + r_z**2))
        I_zz = np.sum(self.mass * (r_x**2 + r_y**2))
        I_xy = -np.sum(self.mass * r_x * r_y)
        I_xz = -np.sum(self.mass * r_x * r_z)
        I_yz = -np.sum(self.mass * r_y * r_z)
        I = np.array([
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz]
        ])
        detI = np.linalg.det(I)
        if detI > 1e-12:
            omega = np.linalg.solve(I, L)
            v_rot = np.cross(omega, r)
            v_centered -= v_rot

        return x_centered, v_centered

    def advance_to(self, time: float) -> StateFrame:
        # Target time
        dt_target = time - self.current_time
        if dt_target > 1e-9:
            n_steps = max(1, int(np.ceil(dt_target / self.dt_sub)))
            h = dt_target / n_steps

            for _ in range(n_steps):
                t_mid = self.current_time + 0.5 * h
                p_now = self._get_pressure(t_mid)

                # Semi-implicit / damped leapfrog Euler integration
                f_int, f_press, _ = self._compute_forces_and_F(self.x, p_now)
                f_total = f_int + f_press

                # Acceleration a = f / m - damping * v
                a = f_total / self.mass[:, None] - self.damping * self.v

                # Velocity update
                self.v += h * a

                # Position update
                self.x += h * self.v

                # Boundary condition: remove rigid body translation & rotation
                self.x, self.v = self._remove_rigid_modes(self.x, self.v)

                self.current_time += h

        self.current_time = time
        p_final = self._get_pressure(time)
        f_int, f_press, F_all = self._compute_forces_and_F(self.x, p_final)

        # Budget external forces: [G, K, 3] where G=1, K=1
        # Net external force on the system (pressure forces)
        total_ext_force = np.sum(f_press, axis=0)  # (3,)
        budget_external_force = total_ext_force.reshape(1, 1, 3)
        cumulative_mass_source = np.zeros(1, dtype=np.float64)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "wall_region": self.wall_region,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "element_deformation_gradient": F_all,
            },
            scalars={
                "internal_pressure_pa": float(p_final),
                "external_pressure_pa": 0.0,
                "reference_force": float(self.reference_force),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.N,
            "num_tets": self.Ne,
            "young_modulus": self.E,
            "poisson_ratio": self.nu,
            "target_pressure": self.target_p,
        }

