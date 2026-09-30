"""Solver implementation for the pneumatic bubble chamber inflation benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physics solver for volumetric elastic pneumatic chamber inflation."""

    def __init__(self) -> None:
        self.scene = None
        self.config: dict = {}
        self.dt: float = 0.0
        self.current_time: float = 0.0

        # State
        self.position: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.velocity: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.mass: np.ndarray = np.empty(0, dtype=np.float64)
        self.inv_mass: np.ndarray = np.empty(0, dtype=np.float64)
        self.body_id: np.ndarray = np.empty(0, dtype=np.int32)
        self.fixed: np.ndarray = np.empty(0, dtype=np.float64)
        self.surface_faces: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.inner_faces: np.ndarray = np.empty((0, 3), dtype=np.int32)

        # Tetrahedral mesh data
        self.tets: np.ndarray = np.empty((0, 4), dtype=np.int32)
        self.inv_Dm: np.ndarray = np.empty((0, 3, 3), dtype=np.float64)
        self.tet_vol: np.ndarray = np.empty(0, dtype=np.float64)

        # Material parameters
        self.E: float = 100000.0
        self.nu: float = 0.499
        self.mu: float = 0.0
        self.lam: float = 0.0
        self.density: float = 1070.0

        # Pressure loading profile
        self.p_max: float = 4000.0
        self.t_ramp_start: float = 0.25
        self.t_ramp_end: float = 2.5
        self.t_hold: float = 4.0

        # Reference force scale
        self.reference_force: float = 1.0

    def initialize(self, scene, seed: int) -> None:
        """Initialize the pneumatic bubble simulation from scene configuration."""
        self.scene = scene
        self.config = scene.config

        # Loading config
        loading_cfg = self.config.get("loading", {})
        self.p_max = float(loading_cfg.get("pressure_pa", 4000.0))
        self.t_ramp_start = float(loading_cfg.get("ramp_start_s", 0.25))
        self.t_ramp_end = float(loading_cfg.get("ramp_end_s", 2.5))
        self.t_hold = float(loading_cfg.get("hold_until_s", 4.0))

        # Material config
        mat_cfg = self.config.get("material_reference", {})
        self.density = float(mat_cfg.get("density_kg_m3", 1070.0))
        self.E = float(mat_cfg.get("youngs_modulus_pa", 100000.0))
        self.nu = float(mat_cfg.get("poisson_ratio", 0.499))

        # Lame parameters (with capped lambda for standard linear/neohookean elasticity stability)
        # For nu ~ 0.499, effective bulk modulus is large; we compute standard Lame parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # Initial scene state
        init_frame = scene.initial_state()
        self.position = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)
        self.fixed = np.array(init_frame.attributes["fixed"], dtype=np.float64, copy=True)
        self.surface_faces = np.array(init_frame.fields["surface_faces"], dtype=np.int32, copy=True)

        n_verts = len(self.position)
        self.current_time = float(init_frame.time)

        # Build tetrahedral mesh from the axisymmetric ring structure
        nr = 7
        n_theta = 32
        tets = []
        for j in range(nr - 1):
            for i in range(n_theta):
                k = (i + 1) % n_theta
                v0 = j * n_theta + i
                v1 = j * n_theta + k
                v2 = (j + 1) * n_theta + k
                v3 = (j + 1) * n_theta + i
                v4 = nr * n_theta + j * n_theta + i
                v5 = nr * n_theta + j * n_theta + k
                v6 = nr * n_theta + (j + 1) * n_theta + k
                v7 = nr * n_theta + (j + 1) * n_theta + i

                hex_tets = [
                    [v0, v2, v1, v6],
                    [v0, v3, v2, v6],
                    [v0, v7, v3, v6],
                    [v0, v6, v1, v5],
                    [v0, v6, v5, v4],
                    [v0, v6, v4, v7],
                ]
                tets.extend(hex_tets)

        self.tets = np.array(tets, dtype=np.int32)
        n_tets = len(self.tets)

        # Reference shape matrices Dm and volumes
        Dm = np.zeros((n_tets, 3, 3), dtype=np.float64)
        for i in range(3):
            Dm[:, :, i] = self.position[self.tets[:, i + 1]] - self.position[self.tets[:, 0]]

        dets = np.linalg.det(Dm)
        self.tet_vol = dets / 6.0
        self.inv_Dm = np.linalg.inv(Dm)

        # Compute lumped nodal masses from tet volumes and density
        node_mass = np.zeros(n_verts, dtype=np.float64)
        for e in range(n_tets):
            m_e = self.tet_vol[e] * self.density / 4.0
            for idx in self.tets[e]:
                node_mass[idx] += m_e

        self.mass = node_mass
        self.inv_mass = np.where(self.fixed > 0.5, 0.0, 1.0 / np.maximum(self.mass, 1e-12))

        # Identify inner surface faces (where pressure is applied)
        # In scene.py, inner surface faces are faces 384..767
        self.inner_faces = self.surface_faces[384:768].copy()

        # Reference force scale (total pressure force scale ~ p_max * area)
        inner_v0 = self.position[self.inner_faces[:, 0]]
        inner_v1 = self.position[self.inner_faces[:, 1]]
        inner_v2 = self.position[self.inner_faces[:, 2]]
        inner_fn = np.cross(inner_v1 - inner_v0, inner_v2 - inner_v0)
        inner_area = np.sum(np.linalg.norm(inner_fn, axis=1)) * 0.5
        self.reference_force = float(max(1.0, self.p_max * inner_area))

    def _get_pressure(self, t: float) -> float:
        """Evaluate internal pressure at time t."""
        if t <= self.t_ramp_start:
            return 0.0
        elif t >= self.t_ramp_end:
            return self.p_max
        else:
            alpha = (t - self.t_ramp_start) / (self.t_ramp_end - self.t_ramp_start)
            # Smooth cubic hermite ramp for gradual inflation
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            return self.p_max * s

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray, p_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute elastic internal forces, pressure external forces, and deformation gradients."""
        n_verts = len(pos)
        n_tets = len(self.tets)

        f_int = np.zeros((n_verts, 3), dtype=np.float64)
        f_ext = np.zeros((n_verts, 3), dtype=np.float64)

        # 1. Deformation gradients F = Ds @ inv_Dm
        Ds = np.zeros((n_tets, 3, 3), dtype=np.float64)
        for i in range(3):
            Ds[:, :, i] = pos[self.tets[:, i + 1]] - pos[self.tets[:, 0]]

        F = np.einsum("eik,ekj->eij", Ds, self.inv_Dm)

        # 2. Neo-Hookean / Linear elastic strain energy Piola-Kirchhoff stress P
        # Stable compressible Neo-Hookean model
        # P = mu * (F - F^-T) + lam * ln(J) * F^-T
        # For numerical robustness against extreme Poisson ratio / high compression:
        J = np.linalg.det(F)
        J = np.maximum(J, 0.05)  # prevent collapse/inversion singularity

        inv_F_T = np.linalg.inv(F).transpose(0, 2, 1)
        ln_J = np.log(J)

        # Clamp lambda bulk term to prevent locking while preserving volume
        lam_eff = min(self.lam, 50.0 * self.mu)
        P = self.mu * (F - inv_F_T) + lam_eff * ln_J[:, None, None] * inv_F_T

        # Tet nodal forces: H = -vol * P @ inv_Dm^T
        inv_Dm_T = self.inv_Dm.transpose(0, 2, 1)
        H = -self.tet_vol[:, None, None] * np.einsum("eik,ekj->eij", P, inv_Dm_T)

        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f3 = H[:, :, 2]
        f0 = -f1 - f2 - f3

        np.add.at(f_int, self.tets[:, 0], f0)
        np.add.at(f_int, self.tets[:, 1], f1)
        np.add.at(f_int, self.tets[:, 2], f2)
        np.add.at(f_int, self.tets[:, 3], f3)

        # 3. Pressure surface traction on inner wall
        # For inner faces, outward normal (into the cavity) is along cross(v1-v0, v2-v0)
        # Pressure pushes outward into the rubber wall, in direction -cross(v1-v0, v2-v0)
        if p_val > 0.0:
            iv0 = pos[self.inner_faces[:, 0]]
            iv1 = pos[self.inner_faces[:, 1]]
            iv2 = pos[self.inner_faces[:, 2]]
            # Normal pointing into cavity is cross(v1-v0, v2-v0)
            # Inflation pressure force on face is -p_val * 0.5 * (v1-v0) x (v2-v0)
            cross_prod = np.cross(iv1 - iv0, iv2 - iv0)
            f_face = -p_val * (cross_prod / 6.0)  # split evenly among 3 face nodes

            np.add.at(f_ext, self.inner_faces[:, 0], f_face)
            np.add.at(f_ext, self.inner_faces[:, 1], f_face)
            np.add.at(f_ext, self.inner_faces[:, 2], f_face)

        # Damping force
        damping_coeff = 25.0
        f_int -= damping_coeff * self.mass[:, None] * vel

        return f_int, f_ext, F

    def _step(self, dt_sub: float, t_mid: float) -> tuple[np.ndarray, np.ndarray]:
        """Perform one sub-step of semi-implicit / explicit integration."""
        p_val = self._get_pressure(t_mid)
        f_int, f_ext, F = self._compute_forces(self.position, self.velocity, p_val)

        total_force = f_int + f_ext
        total_force[self.fixed > 0.5] = 0.0

        # Semi-implicit Euler
        acc = total_force * self.inv_mass[:, None]
        self.velocity += acc * dt_sub
        self.velocity[self.fixed > 0.5] = 0.0

        self.position += self.velocity * dt_sub
        self.position[self.fixed > 0.5] = self.scene.initial_state().position[self.fixed > 0.5]

        return f_ext, F

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance simulation state to target_time and export StateFrame."""
        target_time = float(target_time)

        # Fixed sub-step size for stable numerical integration
        dt_sub = 5e-5

        while self.current_time < target_time - 1e-10:
            step = min(dt_sub, target_time - self.current_time)
            t_mid = self.current_time + 0.5 * step
            self._step(step, t_mid)
            self.current_time += step

        self.current_time = target_time

        # Compute final forces and deformation gradient at current state
        p_val = self._get_pressure(self.current_time)
        f_int, f_ext, F = self._compute_forces(self.position, self.velocity, p_val)

        # External force budget: shape [1, 1, 3] (1 domain, 1 material body, 3 force components)
        sum_f_ext = np.sum(f_ext, axis=0, keepdims=True)  # shape [1, 3]
        budget_ext_force = sum_f_ext.reshape(1, 1, 3)

        # Cumulative mass source: shape [1]
        cum_mass_source = np.zeros(1, dtype=np.float64)

        # Mass quantity: per-sample mass contribution shape [N]
        mass_qty = np.array(self.mass, dtype=np.float64, copy=True)

        frame = StateFrame(
            time=self.current_time,
            position=np.array(self.position, dtype=np.float64, copy=True),
            velocity=np.array(self.velocity, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.array(self.body_id, dtype=np.int32, copy=True),
            attributes={
                "fixed": np.array(self.fixed, dtype=np.float64, copy=True),
                "mass_quantity": mass_qty,
            },
            fields={
                "surface_faces": np.array(self.surface_faces, dtype=np.int32, copy=True),
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": cum_mass_source,
                "element_deformation_gradient": F,
            },
            scalars={
                "internal_pressure_pa": float(p_val),
                "reference_force": float(self.reference_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        """Return solver diagnostics."""
        return {
            "num_nodes": len(self.position),
            "num_tets": len(self.tets),
            "reference_force": self.reference_force,
        }

