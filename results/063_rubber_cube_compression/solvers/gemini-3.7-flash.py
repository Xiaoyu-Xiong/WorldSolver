"""Physical FEM solver for rubber cube compression."""
from __future__ import annotations

import numpy as np
import numba
from worldsolver.contracts import StateFrame


@numba.njit(fastmath=True)
def compute_fem_forces_and_F(
    x: np.ndarray,
    Dm_inv: np.ndarray,
    tet_vol: np.ndarray,
    tetrahedra: np.ndarray,
    mu: float,
    bulk_k: float,
    N: int,
):
    """
    Compute elastic forces on nodes and deformation gradient F on each tet
    using a Neo-Hookean / St. Venant-Kirchhoff style hyperelastic energy.
    Energy density: W = 0.5 * mu * (tr(C) - 3) + 0.5 * bulk_k * (J - 1)^2
    or Neo-Hookean with log(J): W = 0.5 * mu * (tr(C) - 3 - 2*ln(J)) + 0.5 * bulk_k * (J - 1)^2
    First Piola-Kirchhoff stress P = dW/dF = mu * (F - F^{-T}) + bulk_k * (J - 1) * J * F^{-T}
    """
    forces = np.zeros((N, 3), dtype=np.float64)
    num_tets = tetrahedra.shape[0]
    F_all = np.zeros((num_tets, 3, 3), dtype=np.float64)

    for e in range(num_tets):
        i0 = tetrahedra[e, 0]
        i1 = tetrahedra[e, 1]
        i2 = tetrahedra[e, 2]
        i3 = tetrahedra[e, 3]

        # Ds = [x1-x0, x2-x0, x3-x0]
        ds00 = x[i1, 0] - x[i0, 0]
        ds10 = x[i1, 1] - x[i0, 1]
        ds20 = x[i1, 2] - x[i0, 2]

        ds01 = x[i2, 0] - x[i0, 0]
        ds11 = x[i2, 1] - x[i0, 1]
        ds21 = x[i2, 2] - x[i0, 2]

        ds02 = x[i3, 0] - x[i0, 0]
        ds12 = x[i3, 1] - x[i0, 1]
        ds22 = x[i3, 2] - x[i0, 2]

        # F = Ds * Dm_inv
        inv00 = Dm_inv[e, 0, 0]
        inv01 = Dm_inv[e, 0, 1]
        inv02 = Dm_inv[e, 0, 2]
        inv10 = Dm_inv[e, 1, 0]
        inv11 = Dm_inv[e, 1, 1]
        inv12 = Dm_inv[e, 1, 2]
        inv20 = Dm_inv[e, 2, 0]
        inv21 = Dm_inv[e, 2, 1]
        inv22 = Dm_inv[e, 2, 2]

        f00 = ds00 * inv00 + ds01 * inv10 + ds02 * inv20
        f01 = ds00 * inv01 + ds01 * inv11 + ds02 * inv21
        f02 = ds00 * inv02 + ds01 * inv12 + ds02 * inv22

        f10 = ds10 * inv00 + ds11 * inv10 + ds12 * inv20
        f11 = ds10 * inv01 + ds11 * inv11 + ds12 * inv21
        f12 = ds10 * inv02 + ds11 * inv12 + ds12 * inv22

        f20 = ds20 * inv00 + ds21 * inv10 + ds22 * inv20
        f21 = ds20 * inv01 + ds21 * inv11 + ds22 * inv21
        f22 = ds20 * inv02 + ds21 * inv12 + ds22 * inv22

        F_all[e, 0, 0] = f00
        F_all[e, 0, 1] = f01
        F_all[e, 0, 2] = f02
        F_all[e, 1, 0] = f10
        F_all[e, 1, 1] = f11
        F_all[e, 1, 2] = f12
        F_all[e, 2, 0] = f20
        F_all[e, 2, 1] = f21
        F_all[e, 2, 2] = f22

        # det(F) = J
        detF = (
            f00 * (f11 * f22 - f12 * f21)
            - f01 * (f10 * f22 - f12 * f20)
            + f02 * (f10 * f21 - f11 * f20)
        )
        if detF < 1e-4:
            detF = 1e-4

        # Inverse transpose of F: invFT = (cofactor(F))^T / detF
        c00 = f11 * f22 - f12 * f21
        c01 = -(f10 * f22 - f12 * f20)
        c02 = f10 * f21 - f11 * f20
        c10 = -(f01 * f22 - f02 * f21)
        c11 = f00 * f22 - f02 * f20
        c12 = -(f00 * f21 - f01 * f20)
        c20 = f01 * f12 - f02 * f11
        c21 = -(f00 * f12 - f02 * f10)
        c22 = f00 * f11 - f01 * f10

        invFT00 = c00 / detF
        invFT01 = c10 / detF
        invFT02 = c20 / detF
        invFT10 = c01 / detF
        invFT11 = c11 / detF
        invFT12 = c21 / detF
        invFT20 = c02 / detF
        invFT21 = c12 / detF
        invFT22 = c22 / detF

        # First Piola-Kirchhoff stress P = mu * (F - invFT) + bulk_k * (detF - 1) * detF * invFT
        # Or standard Neo-Hookean: P = mu * (F - invFT) + bulk_k * ln(detF) * invFT
        vol = tet_vol[e]
        lnJ = np.log(detF)
        p_hydro = bulk_k * lnJ

        p00 = mu * (f00 - invFT00) + p_hydro * invFT00
        p01 = mu * (f01 - invFT01) + p_hydro * invFT01
        p02 = mu * (f02 - invFT02) + p_hydro * invFT02

        p10 = mu * (f10 - invFT10) + p_hydro * invFT10
        p11 = mu * (f11 - invFT11) + p_hydro * invFT11
        p12 = mu * (f12 - invFT12) + p_hydro * invFT12

        p20 = mu * (f20 - invFT20) + p_hydro * invFT20
        p21 = mu * (f21 - invFT21) + p_hydro * invFT21
        p22 = mu * (f22 - invFT22) + p_hydro * invFT22

        # Force matrix H = -vol * P * Dm_inv^T
        # H = [h0, h1, h2] for nodes 1, 2, 3; node 0 force is -h0 - h1 - h2
        # H = -vol * P * (Dm_inv)^T
        # Dm_inv^T:
        # [inv00, inv10, inv20]
        # [inv01, inv11, inv21]
        # [inv02, inv12, inv22]
        h00 = -vol * (p00 * inv00 + p01 * inv01 + p02 * inv02)
        h01 = -vol * (p00 * inv10 + p01 * inv11 + p02 * inv12)
        h02 = -vol * (p00 * inv20 + p01 * inv21 + p02 * inv22)

        h10 = -vol * (p10 * inv00 + p11 * inv01 + p12 * inv02)
        h11 = -vol * (p10 * inv10 + p11 * inv11 + p12 * inv12)
        h12 = -vol * (p10 * inv20 + p11 * inv21 + p12 * inv22)

        h20 = -vol * (p20 * inv00 + p21 * inv01 + p22 * inv02)
        h21 = -vol * (p20 * inv10 + p21 * inv11 + p22 * inv12)
        h22 = -vol * (p20 * inv20 + p21 * inv21 + p22 * inv22)

        # Node 1 force += col 0 of H
        forces[i1, 0] += h00
        forces[i1, 1] += h10
        forces[i1, 2] += h20

        # Node 2 force += col 1 of H
        forces[i2, 0] += h01
        forces[i2, 1] += h11
        forces[i2, 2] += h21

        # Node 3 force += col 2 of H
        forces[i3, 0] += h02
        forces[i3, 1] += h12
        forces[i3, 2] += h22

        # Node 0 force += - (h col 0 + h col 1 + h col 2)
        forces[i0, 0] -= (h00 + h01 + h02)
        forces[i0, 1] -= (h10 + h11 + h12)
        forces[i0, 2] -= (h20 + h21 + h22)

    return forces, F_all


@numba.njit(fastmath=True)
def integrate_substeps(
    pos: np.ndarray,
    vel: np.ndarray,
    inv_mass: np.ndarray,
    Dm_inv: np.ndarray,
    tet_vol: np.ndarray,
    tetrahedra: np.ndarray,
    mu: float,
    bulk_k: float,
    damping_coeff: float,
    bottom_fixed_mask: np.ndarray,
    top_force_per_node: np.ndarray,
    top_mask: np.ndarray,
    substeps: int,
    dt_sub: float,
):
    N = pos.shape[0]
    for _ in range(substeps):
        # Compute internal elastic forces
        f_int, _ = compute_fem_forces_and_F(
            pos, Dm_inv, tet_vol, tetrahedra, mu, bulk_k, N
        )

        # Total forces = elastic forces + damping + external top traction
        for i in range(N):
            if bottom_fixed_mask[i]:
                vel[i, 0] = 0.0
                vel[i, 1] = 0.0
                vel[i, 2] = 0.0
                continue

            f_x = f_int[i, 0] - damping_coeff * vel[i, 0]
            f_y = f_int[i, 1] - damping_coeff * vel[i, 1]
            f_z = f_int[i, 2] - damping_coeff * vel[i, 2]

            if top_mask[i]:
                f_x += top_force_per_node[0]
                f_y += top_force_per_node[1]
                f_z += top_force_per_node[2]

            im = inv_mass[i]
            vel[i, 0] += f_x * im * dt_sub
            vel[i, 1] += f_y * im * dt_sub
            vel[i, 2] += f_z * im * dt_sub

            # Position update (semi-implicit Euler)
            pos[i, 0] += vel[i, 0] * dt_sub
            pos[i, 1] += vel[i, 1] * dt_sub
            pos[i, 2] += vel[i, 2] * dt_sub

            # Constraint: bottom face fixed at z = 0
            # (handled by zeroing vel and not moving)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Mesh geometry
        self.vertices = scene.vertices.copy()
        self.tetrahedra = scene.tetrahedra.copy()
        self.faces = scene.faces.copy()
        self.num_nodes = len(self.vertices)
        self.num_tets = len(self.tetrahedra)

        # Node classification
        self.bottom_mask = np.isclose(self.vertices[:, 2], 0.0)
        self.top_mask = np.isclose(self.vertices[:, 2], 0.2)
        self.num_top_nodes = np.sum(self.top_mask)

        # Cube physical dimensions and mass
        # Dimension: 0.2 x 0.2 x 0.2 m, volume = 0.008 m^3
        # Density for rubber ~ 1000 kg/m^3 -> Total mass = 8.0 kg
        # Lumped mass per node
        total_volume = 0.2 * 0.2 * 0.2
        density = 1000.0  # kg/m^3
        total_mass = density * total_volume  # 8.0 kg

        # Material parameters:
        # Young's modulus E = 200 kPa (0.2 MPa), Poisson's ratio nu = 0.45 (near-incompressible rubber)
        # Mu = E / (2 * (1 + nu)) = 200e3 / (2 * 1.45) = 68965.5 Pa
        # Bulk modulus K = E / (3 * (1 - 2*nu)) = 200e3 / (3 * 0.1) = 666666.7 Pa
        # With 50 kPa compressive traction on top, nominal strain ~ 50 / 200 = 25%
        self.E = 2.0e5  # Pa
        self.nu = 0.45
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.bulk_k = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        # Damping to reach quasi-static equilibrium smoothly
        self.damping_coeff = 25.0

        # Precompute reference shape matrices and volumes for each tet
        self.Dm_inv = np.zeros((self.num_tets, 3, 3), dtype=np.float64)
        self.tet_vol = np.zeros(self.num_tets, dtype=np.float64)
        node_mass = np.zeros(self.num_nodes, dtype=np.float64)

        for e in range(self.num_tets):
            i0, i1, i2, i3 = self.tetrahedra[e]
            x0 = self.vertices[i0]
            x1 = self.vertices[i1]
            x2 = self.vertices[i2]
            x3 = self.vertices[i3]

            Dm = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
            vol = np.linalg.det(Dm) / 6.0
            if vol < 0:
                # Swap vertices to ensure positive orientation
                self.tetrahedra[e, 1], self.tetrahedra[e, 2] = self.tetrahedra[e, 2], self.tetrahedra[e, 1]
                i1, i2 = self.tetrahedra[e, 1], self.tetrahedra[e, 2]
                Dm = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
                vol = np.linalg.det(Dm) / 6.0

            self.tet_vol[e] = vol
            self.Dm_inv[e] = np.linalg.inv(Dm)

            m_tet = vol * density
            node_mass[i0] += m_tet / 4.0
            node_mass[i1] += m_tet / 4.0
            node_mass[i2] += m_tet / 4.0
            node_mass[i3] += m_tet / 4.0

        self.mass = node_mass
        self.inv_mass = 1.0 / np.maximum(node_mass, 1e-12)

        # Dynamic state
        self.pos = self.vertices.copy()
        self.vel = np.zeros_like(self.vertices)
        self.current_time = 0.0

        # Peak traction and schedule
        self.peak_traction = 50000.0  # Pa
        self.top_area = 0.2 * 0.2     # 0.04 m^2
        self.total_peak_force = self.peak_traction * self.top_area  # 2000 N

        # Warm-up JIT
        f_dummy, F_dummy = compute_fem_forces_and_F(
            self.pos, self.Dm_inv, self.tet_vol, self.tetrahedra, self.mu, self.bulk_k, self.num_nodes
        )

    def _get_traction_scale(self, t: float) -> float:
        """Schedule: 0.0 to 0.25 -> 0.0; 0.25 to 1.5 -> linear 0.0 to 1.0; 1.5 to 2.0 -> 1.0"""
        if t <= 0.25:
            return 0.0
        elif t >= 1.5:
            return 1.0
        else:
            return (t - 0.25) / (1.5 - 0.25)

    def advance_to(self, target_time: float) -> StateFrame:
        # Time-stepping
        dt_target = target_time - self.current_time
        if dt_target > 1e-9:
            # Substep size ~ 1e-4 s for numerical stability
            dt_sub = 1e-4
            num_substeps = max(1, int(np.round(dt_target / dt_sub)))
            actual_dt_sub = dt_target / num_substeps

            for step in range(num_substeps):
                t_mid = self.current_time + (step + 0.5) * actual_dt_sub
                scale = self._get_traction_scale(t_mid)
                applied_traction = self.peak_traction * scale

                # Total vertical downward force on top face: F_z = - applied_traction * top_area
                # Distributed among top nodes
                force_per_top_node = np.array([0.0, 0.0, -applied_traction * self.top_area / self.num_top_nodes], dtype=np.float64)

                integrate_substeps(
                    self.pos,
                    self.vel,
                    self.inv_mass,
                    self.Dm_inv,
                    self.tet_vol,
                    self.tetrahedra,
                    self.mu,
                    self.bulk_k,
                    self.damping_coeff,
                    self.bottom_mask,
                    force_per_top_node,
                    self.top_mask,
                    1,
                    actual_dt_sub,
                )

            self.current_time = target_time

        # Compute current deformation gradient F for all elements
        _, F_elements = compute_fem_forces_and_F(
            self.pos, self.Dm_inv, self.tet_vol, self.tetrahedra, self.mu, self.bulk_k, self.num_nodes
        )

        scale_now = self._get_traction_scale(self.current_time)
        applied_traction_now = self.peak_traction * scale_now
        total_applied_force_z = -applied_traction_now * self.top_area

        # Required fields:
        # 1. attributes.mass_quantity: [N] or [N, K]
        # 2. fields.budget_external_force: [G, K, 3]
        #    G=1, K=1, [1, 1, 3] -> external downward force + reaction
        # 3. fields.cumulative_mass_source: [K] -> [1]
        # 4. fields.element_deformation_gradient: [Ne, 3, 3]
        # 5. scalars.reference_force: scalar
        budget_external_force = np.zeros((1, 1, 3), dtype=np.float64)
        budget_external_force[0, 0, 2] = total_applied_force_z

        cumulative_mass_source = np.zeros((1,), dtype=np.float64)
        reference_force = float(self.total_peak_force)

        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.vertices.copy(),
                "bottom_face": self.bottom_mask.astype(float),
                "top_face": self.top_mask.astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "element_deformation_gradient": F_elements,
            },
            scalars={
                "applied_traction_pa": float(applied_traction_now),
                "reference_force": reference_force,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_nodes": self.num_nodes,
            "num_tets": self.num_tets,
        }

