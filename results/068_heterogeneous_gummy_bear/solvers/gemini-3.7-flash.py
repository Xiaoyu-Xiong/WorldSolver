"""Physically-based solver for heterogeneous gummy bear simulation.

Models the deformable gummy bear consisting of soft outer material and stiff core regions.
Uses a coupled mass-spring / shape-matching elastic formulation with ground contact.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _load_obj(filename: str):
    path = Path(__file__).resolve().parent.parent / "assets" / filename
    verts = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load mesh geometries
        v_out, f_out = _load_obj("gummy_bear_outer.obj")
        v_stiff, f_stiff = _load_obj("gummy_bear_stiff.obj")

        self.n_out = len(v_out)
        self.n_stiff = len(v_stiff)
        self.N = self.n_out + self.n_stiff

        # Initial state from scene
        init_frame = scene.initial_state()
        self.x0 = np.copy(init_frame.position)  # rest reference configuration (N, 3)
        self.x = np.copy(init_frame.position)   # current positions (N, 3)
        self.v = np.copy(init_frame.velocity)   # current velocities (N, 3)
        self.body_id = np.copy(init_frame.body_id)

        # Physical parameters
        init_cond = self.config.get("initial_conditions", {})
        self.gravity = np.array(init_cond.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.density = float(init_cond.get("density_kg_m3", 1000.0))
        self.soft_E = float(init_cond.get("soft_youngs_modulus_pa", 6.0e4))
        self.stiff_E = float(init_cond.get("stiff_youngs_modulus_pa", 1.0e14))
        self.poisson = float(init_cond.get("poisson_ratio", 0.45))

        # Particle mass allocation based on total estimated volume
        # Approximate bear volume ~ 0.005 m^3, total mass ~ 5 kg
        total_mass = 5.0
        self.mass = np.full(self.N, total_mass / self.N, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass

        # Subdomains / bodies: 0 = soft_bear_exterior, 1 = stiff_bear_interior
        self.G = 2
        self.K = 2

        # Build springs: surface mesh edges + cross springs + interface coupling
        # Edges from meshes
        edges = set()
        for f in f_out:
            for i in range(3):
                u, w = int(f[i]), int(f[(i + 1) % 3])
                edges.add((min(u, w), max(u, w)))

        for f in f_stiff:
            for i in range(3):
                u, w = int(f[i]) + self.n_out, int(f[(i + 1) % 3]) + self.n_out
                edges.add((min(u, w), max(u, w)))

        # Internal cross-springs (k-nearest neighbors) within soft and stiff
        # Soft volume cross springs
        dist_out = np.linalg.norm(self.x0[:self.n_out, None, :] - self.x0[None, :self.n_out, :], axis=-1)
        for i in range(self.n_out):
            knn = np.argsort(dist_out[i])[1:12]
            for j in knn:
                edges.add((min(i, int(j)), max(i, int(j))))

        # Stiff volume cross springs
        dist_stiff = np.linalg.norm(self.x0[self.n_out:, None, :] - self.x0[None, self.n_out:, :], axis=-1)
        for i in range(self.n_stiff):
            knn = np.argsort(dist_stiff[i])[1:15]
            for j in knn:
                u, w = i + self.n_out, int(j) + self.n_out
                edges.add((min(u, w), max(u, w)))

        # Interface bonding springs: couple nearby vertices between soft and stiff
        dist_inter = np.linalg.norm(self.x0[:self.n_out, None, :] - self.x0[None, self.n_out:, :], axis=-1)
        for j in range(self.n_stiff):
            closest_soft = np.argsort(dist_inter[:, j])[:6]
            for i in closest_soft:
                edges.add((min(int(i), j + self.n_out), max(int(i), j + self.n_out)))

        edge_list = np.array(list(edges), dtype=np.int32)
        self.spring_i = edge_list[:, 0]
        self.spring_j = edge_list[:, 1]

        p_i = self.x0[self.spring_i]
        p_j = self.x0[self.spring_j]
        self.rest_len = np.linalg.norm(p_i - p_j, axis=-1)

        # Spring stiffness based on Young's modulus
        is_stiff_i = self.spring_i >= self.n_out
        is_stiff_j = self.spring_j >= self.n_out

        # Base stiffness factors
        self.spring_k = np.zeros(len(edge_list), dtype=np.float64)
        for idx in range(len(edge_list)):
            if is_stiff_i[idx] and is_stiff_j[idx]:
                self.spring_k[idx] = 2.0e5
            elif is_stiff_i[idx] or is_stiff_j[idx]:
                self.spring_k[idx] = 1.0e5
            else:
                self.spring_k[idx] = 3.0e3

        self.spring_damping = 0.05

        # Stiff core rest shape matching setup (for rigid core preservation)
        self.stiff_indices = np.arange(self.n_out, self.N)
        self.stiff_x0 = self.x0[self.stiff_indices]
        self.stiff_com0 = np.mean(self.stiff_x0, axis=0)
        self.stiff_q0 = self.stiff_x0 - self.stiff_com0

        # Ground boundary
        self.ground_z = 0.0
        self.ground_k = 5.0e4
        self.ground_damp = 50.0
        self.friction_coeff = 0.4

        # Simulation time-stepping
        self.current_time = 0.0
        self.dt = 0.001  # sub-step size

        # Budget accumulators
        self.cum_external_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_inflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_outflow_momentum = np.zeros((self.G, 3), dtype=np.float64)

        # Reference momentum scale
        init_p = np.sum(np.abs(self.mass[:, None] * self.v))
        self.ref_momentum = float(max(init_p, 10.0))

        # Domain weights [G, N]
        self.domain_weights = np.zeros((self.G, self.N), dtype=np.float64)
        self.domain_weights[0, :self.n_out] = 1.0
        self.domain_weights[1, self.n_out:] = 1.0

    def _compute_stiff_matching_targets(self):
        """Compute optimal rigid transformation for stiff core particles."""
        x_stiff = self.x[self.stiff_indices]
        com = np.mean(x_stiff, axis=0)
        q = x_stiff - com

        # Covariance matrix
        H = self.stiff_q0.T @ q
        # Polar decomposition via SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        goal_pos = com + (self.stiff_q0 @ R.T)
        return goal_pos

    def _step(self, dt: float):
        # 1. External forces (gravity)
        f_ext = self.mass[:, None] * self.gravity[None, :]  # (N, 3)

        # 2. Internal spring forces
        xi = self.x[self.spring_i]
        xj = self.x[self.spring_j]
        dij = xj - xi
        dist = np.linalg.norm(dij, axis=-1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        dir_ij = dij / dist_safe

        delta = dist.squeeze(-1) - self.rest_len
        rel_v = self.v[self.spring_j] - self.v[self.spring_i]
        proj_v = np.sum(rel_v * dir_ij, axis=-1)

        f_mag = self.spring_k * delta + self.spring_damping * self.spring_k * proj_v
        f_spring = f_mag[:, None] * dir_ij

        f_total = np.copy(f_ext)
        np.add.at(f_total, self.spring_i, f_spring)
        np.add.at(f_total, self.spring_j, -f_spring)

        # 3. Shape matching restoring forces for stiff core
        goal_stiff = self._compute_stiff_matching_targets()
        k_shape = 5.0e4
        f_shape = k_shape * (goal_stiff - self.x[self.stiff_indices])
        f_total[self.stiff_indices] += f_shape

        # 4. Ground contact force
        penetration = self.ground_z - self.x[:, 2]
        contact = penetration > 0
        f_contact = np.zeros_like(self.x)
        if np.any(contact):
            # Normal spring-damper contact
            f_n = self.ground_k * penetration[contact] - self.ground_damp * self.v[contact, 2]
            f_n = np.maximum(f_n, 0.0)
            f_contact[contact, 2] = f_n

            # Coulomb friction
            v_tan = self.v[contact, :2]
            v_speed = np.linalg.norm(v_tan, axis=-1, keepdims=True)
            v_safe = np.maximum(v_speed, 1e-6)
            f_fric = - (v_tan / v_safe) * np.minimum(self.friction_coeff * f_n[:, None], self.ground_damp * v_speed)
            f_contact[contact, :2] = f_fric

        f_total += f_contact

        # Accumulate external impulse (gravity + contact)
        ext_f_soft = f_ext[:self.n_out] + f_contact[:self.n_out]
        ext_f_stiff = f_ext[self.n_out:] + f_contact[self.n_out:]
        impulse_soft = np.sum(ext_f_soft, axis=0) * dt
        impulse_stiff = np.sum(ext_f_stiff, axis=0) * dt
        self.cum_external_impulse[0] += impulse_soft
        self.cum_external_impulse[1] += impulse_stiff

        # Numerical integration (Semi-implicit Euler with damping)
        damping = 0.9995
        a = f_total * self.inv_mass[:, None]
        self.v = (self.v + a * dt) * damping
        self.x += self.v * dt

        # Hard ground constraint
        below = self.x[:, 2] < self.ground_z
        if np.any(below):
            self.x[below, 2] = self.ground_z
            self.v[below, 2] = np.maximum(self.v[below, 2], 0.0)
            self.v[below, :2] *= 0.9

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute estimated deformation gradient F for each domain element."""
        # 2 elements: soft exterior and stiff core
        F = np.zeros((2, 3, 3), dtype=np.float64)

        # Soft region deformation gradient
        x_soft = self.x[:self.n_out]
        x0_soft = self.x0[:self.n_out]
        p_soft = x_soft - np.mean(x_soft, axis=0)
        q_soft = x0_soft - np.mean(x0_soft, axis=0)
        A_soft = p_soft.T @ q_soft
        A0_soft = q_soft.T @ q_soft
        try:
            F[0] = A_soft @ np.linalg.inv(A0_soft)
        except np.linalg.LinAlgError:
            F[0] = np.eye(3)

        # Stiff region deformation gradient
        x_stiff = self.x[self.n_out:]
        x0_stiff = self.x0[self.n_out:]
        p_stiff = x_stiff - np.mean(x_stiff, axis=0)
        q_stiff = x0_stiff - np.mean(x0_stiff, axis=0)
        A_stiff = p_stiff.T @ q_stiff
        A0_stiff = q_stiff.T @ q_stiff
        try:
            F[1] = A_stiff @ np.linalg.inv(A0_stiff)
        except np.linalg.LinAlgError:
            F[1] = np.eye(3)

        return F

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until reaching the target sample time
        while self.current_time < time - 1e-9:
            step_dt = min(self.dt, time - self.current_time)
            self._step(step_dt)
            self.current_time += step_dt

        # Compute budget external forces for this frame [G=2, K=2, 3]
        f_ext_g = np.zeros((self.G, self.K, 3), dtype=np.float64)
        # Term 0: gravity, Term 1: boundary contact
        f_ext_g[0, 0] = np.sum(self.mass[:self.n_out, None] * self.gravity[None, :], axis=0)
        f_ext_g[1, 0] = np.sum(self.mass[self.n_out:, None] * self.gravity[None, :], axis=0)

        penetration = np.maximum(self.ground_z - self.x[:, 2], 0.0)
        contact_force_z = self.ground_k * penetration
        f_ext_g[0, 1, 2] = np.sum(contact_force_z[:self.n_out])
        f_ext_g[1, 1, 2] = np.sum(contact_force_z[self.n_out:])

        # Interaction forces between domains [K=2, 3]
        interaction_f = np.zeros((self.K, 3), dtype=np.float64)
        # Interface spring forces between soft and stiff
        inter_mask = (self.spring_i < self.n_out) & (self.spring_j >= self.n_out)
        if np.any(inter_mask):
            xi = self.x[self.spring_i[inter_mask]]
            xj = self.x[self.spring_j[inter_mask]]
            dij = xj - xi
            dist = np.linalg.norm(dij, axis=-1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            dir_ij = dij / dist_safe
            delta = dist.squeeze(-1) - self.rest_len[inter_mask]
            f_mag = self.spring_k[inter_mask] * delta
            f_inter = np.sum(f_mag[:, None] * dir_ij, axis=0)
            interaction_f[0] = f_inter
            interaction_f[1] = -f_inter

        # Deformation gradient [Ne=2, 3, 3]
        F = self._compute_deformation_gradient()

        frame = StateFrame(
            time=time,
            position=np.copy(self.x),
            velocity=np.copy(self.v),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "mass_quantity": np.copy(self.mass),
                "material_region": self.body_id.astype(np.float64),
            },
            fields={
                "budget_cumulative_external_impulse": np.copy(self.cum_external_impulse),
                "budget_cumulative_inflow_momentum": np.copy(self.cum_inflow_momentum),
                "budget_cumulative_outflow_momentum": np.copy(self.cum_outflow_momentum),
                "budget_domain_weights": np.copy(self.domain_weights),
                "budget_external_force": np.copy(f_ext_g),
                "cumulative_mass_source": np.zeros(self.K, dtype=np.float64),
                "element_deformation_gradient": F,
                "interaction_force": np.copy(interaction_f),
            },
            scalars={
                "reference_momentum": float(self.ref_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "particle_count": self.N,
        }

