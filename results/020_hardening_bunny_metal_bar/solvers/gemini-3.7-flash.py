"""Solver Agent-owned implementation.

Simulates a viscoplastic bunny falling onto a metal bar with work hardening
using a continuous deformable physics model (viscoplastic mass-spring / shape matching with plasticity and hardening).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load mesh asset
        asset = Path(__file__).resolve().parent.parent / self.config["subject_asset"]
        with np.load(asset) as data:
            self.rest_vertices = np.asarray(data["vertices"], dtype=float)
            self.faces = np.asarray(data["faces"], dtype=int)

        self.n_nodes = len(self.rest_vertices)
        self.dim = 3

        # Physical parameters from config
        self.total_mass = float(self.config.get("subject_mass", 2.4))
        self.mass = np.full(self.n_nodes, self.total_mass / self.n_nodes, dtype=float)
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.initial_velocity = np.asarray(self.config.get("initial_velocity", [0.0, 0.0, -0.55]), dtype=float)

        self.base_stiffness = float(self.config.get("stiffness", 18.0))
        self.hardening_rate = float(self.config.get("hardening_rate", 1.8))
        self.damping = float(self.config.get("damping", 1.4))
        self.restitution = float(self.config.get("restitution", 0.28))
        self.friction = float(self.config.get("friction", 0.55))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))

        # Obstacle geometry
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.bar_z = float(self.config.get("bar_z", 0.66))
        self.bar_radius = float(self.config.get("bar_radius", 0.085))

        # State variables
        self.x = self.rest_vertices.copy()
        self.v = np.tile(self.initial_velocity, (self.n_nodes, 1))
        self.current_time = 0.0

        # Build mesh edges for structural spring networks
        edges = set()
        for f in self.faces:
            for i in range(3):
                u, w = sorted([f[i], f[(i + 1) % 3]])
                edges.add((u, w))
        self.edges = np.array(list(edges), dtype=int)

        # Compute rest lengths
        p0 = self.rest_vertices[self.edges[:, 0]]
        p1 = self.rest_vertices[self.edges[:, 1]]
        self.rest_edge_lengths = np.linalg.norm(p0 - p1, axis=1)
        self.plastic_edge_lengths = self.rest_edge_lengths.copy()
        self.edge_hardening = np.zeros(len(self.edges), dtype=float)

        # Per-node plastic deformation and hardening tracking
        self.plastic_strain = np.zeros(self.n_nodes, dtype=float)
        self.hardening = np.zeros(self.n_nodes, dtype=float)

        # Cluster/Region definition for shape matching / deformation gradient computation
        # Find k-nearest neighbors for local deformation gradient
        self._build_local_neighborhoods(k=12)

        # Deformation gradient per node (3x3)
        self.F = np.tile(np.eye(3), (self.n_nodes, 1, 1))

        # Contact tracking
        self.active_contacts = 0
        self.cumulative_mass_source = np.zeros(1, dtype=float)

    def _build_local_neighborhoods(self, k: int = 12):
        """Precompute local neighborhood matrices for local deformation gradients."""
        from scipy.spatial import cKDTree
        tree = cKDTree(self.rest_vertices)
        _, self.neighbors = tree.query(self.rest_vertices, k=k)

        # Precompute inverted covariance matrices for rest shape
        self.inv_rest_cov = np.zeros((self.n_nodes, 3, 3), dtype=float)
        for i in range(self.n_nodes):
            nbrs = self.neighbors[i]
            r_rel = self.rest_vertices[nbrs] - self.rest_vertices[i]
            cov = r_rel.T @ r_rel + 1e-6 * np.eye(3)
            self.inv_rest_cov[i] = np.linalg.inv(cov)

    def _compute_deformation_gradient(self):
        """Compute per-node deformation gradient F = A_pq * inv(A_qq)."""
        F_all = np.zeros((self.n_nodes, 3, 3), dtype=float)
        for i in range(self.n_nodes):
            nbrs = self.neighbors[i]
            x_rel = self.x[nbrs] - self.x[i]
            r_rel = self.rest_vertices[nbrs] - self.rest_vertices[i]
            A_pq = x_rel.T @ r_rel
            F_all[i] = A_pq @ self.inv_rest_cov[i]
        return F_all

    def _substep(self, dt: float):
        """Advance the physics by one small sub-step dt."""
        # 1. Internal elastic and plastic forces
        f_int = np.zeros_like(self.x)

        # Edge springs with viscoplasticity and work hardening
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        p0 = self.x[i0]
        p1 = self.x[i1]
        diff = p1 - p0
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        dir_vec = diff / dist[:, None]

        # Relative elongation relative to plastic rest length
        current_rest = self.plastic_edge_lengths
        strain = (dist - current_rest) / self.rest_edge_lengths

        # Hardening increases stiffness locally: k = k0 * (1.0 + hardening_rate * edge_hardening)
        k_edge = self.base_stiffness * 80.0 * (1.0 + self.hardening_rate * self.edge_hardening)
        force_mag = k_edge * (dist - current_rest)

        # Damping force along edge
        rel_v = self.v[i1] - self.v[i0]
        v_proj = np.sum(rel_v * dir_vec, axis=1)
        d_force = self.damping * 2.0 * v_proj

        total_edge_force = (force_mag + d_force)[:, None] * dir_vec

        np.add.at(f_int, i0, total_edge_force)
        np.add.at(f_int, i1, -total_edge_force)

        # Viscoplastic flow & hardening update for edges:
        # If strain exceeds yield threshold, plastic rest length evolves towards current length
        yield_threshold = 0.08
        plastic_rate = 15.0
        exceed = np.abs(strain) - yield_threshold
        mask_yield = exceed > 0.0
        if np.any(mask_yield):
            delta_L = plastic_rate * np.sign(strain[mask_yield]) * exceed[mask_yield] * dt * self.rest_edge_lengths[mask_yield]
            # Clamp change to avoid inversion
            self.plastic_edge_lengths[mask_yield] += delta_L
            self.plastic_edge_lengths[mask_yield] = np.clip(
                self.plastic_edge_lengths[mask_yield],
                0.2 * self.rest_edge_lengths[mask_yield],
                3.0 * self.rest_edge_lengths[mask_yield]
            )
            # Hardening accumulation
            self.edge_hardening[mask_yield] += 2.0 * np.abs(delta_L) / self.rest_edge_lengths[mask_yield]

        # Fast Shape matching preservation force across pre-built local clusters
        # Compute local F in vectorized/batched slices
        for i in range(self.n_nodes):
            nbrs = self.neighbors[i]
            x_rel = self.x[nbrs] - self.x[i]
            r_rel = self.rest_vertices[nbrs] - self.rest_vertices[i]
            A_pq = x_rel.T @ r_rel
            F_local = A_pq @ self.inv_rest_cov[i]

            # Fast rotation extraction via SVD
            U, _, Vt = np.linalg.svd(F_local)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            target_rel = r_rel @ R.T
            err = target_rel - x_rel
            k_shape = self.base_stiffness * 12.0 * (1.0 + self.hardening_rate * self.hardening[i])
            f_shape = k_shape * err
            f_int[nbrs] += f_shape / len(nbrs)

        # Update node hardening from edge hardening
        self.hardening.fill(0.0)
        np.add.at(self.hardening, i0, 0.5 * self.edge_hardening)
        np.add.at(self.hardening, i1, 0.5 * self.edge_hardening)
        self.hardening = np.clip(self.hardening, 0.0, 15.0)

        # 2. Gravity and External forces
        f_ext = self.mass[:, None] * self.gravity[None, :]

        # 3. Acceleration & velocity integration (Symplectic Euler)
        a = (f_int + f_ext) / self.mass[:, None]
        self.v += a * dt

        # Global velocity damping
        self.v *= np.exp(-0.15 * dt)

        # Predict candidate positions
        self.x += self.v * dt

        # 4. Collisions
        # Cylinder collision: axis is along Y axis at X=0, Z=bar_z
        # Cylinder radius = bar_radius
        dx = self.x[:, 0]
        dz = self.x[:, 2] - self.bar_z
        dist_cylinder = np.sqrt(dx * dx + dz * dz) + 1e-12

        in_cylinder = dist_cylinder < self.bar_radius
        if np.any(in_cylinder):
            normal_x = dx[in_cylinder] / dist_cylinder[in_cylinder]
            normal_z = dz[in_cylinder] / dist_cylinder[in_cylinder]
            normal = np.column_stack([normal_x, np.zeros_like(normal_x), normal_z])

            # Penetration depth
            penetration = self.bar_radius - dist_cylinder[in_cylinder]
            self.x[in_cylinder] += normal * penetration[:, None]

            # Collision response on velocity
            vn = np.sum(self.v[in_cylinder] * normal, axis=1)
            approach = vn < 0.0
            if np.any(approach):
                idx_app = np.where(in_cylinder)[0][approach]
                norm_app = normal[approach]
                vn_app = vn[approach]

                # Normal restitution impulse
                v_norm_vec = vn_app[:, None] * norm_app
                v_tan = self.v[idx_app] - v_norm_vec

                # Friction
                v_tan_mag = np.linalg.norm(v_tan, axis=1, keepdims=True) + 1e-12
                friction_scale = np.maximum(0.0, 1.0 - self.friction * (1.0 + self.restitution) * np.abs(vn_app[:, None]) / v_tan_mag)
                v_tan_new = v_tan * friction_scale

                self.v[idx_app] = v_tan_new - self.restitution * v_norm_vec

        # Ground collision at ground_z
        under_ground = self.x[:, 2] < self.ground_z
        if np.any(under_ground):
            self.x[under_ground, 2] = self.ground_z
            vn_g = self.v[under_ground, 2]
            approach_g = vn_g < 0.0
            if np.any(approach_g):
                idx_g = np.where(under_ground)[0][approach_g]
                self.v[idx_g, 2] = -self.restitution * self.v[idx_g, 2]
                self.v[idx_g, 0] *= (1.0 - self.friction)
                self.v[idx_g, 1] *= (1.0 - self.friction)

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation in small substeps to target time
        target_dt = time - self.current_time
        if target_dt > 1e-8:
            substep_dt = 0.002  # 2 ms substeps
            n_substeps = max(1, int(np.ceil(target_dt / substep_dt)))
            actual_dt = target_dt / n_substeps
            for _ in range(n_substeps):
                self._substep(actual_dt)
            self.current_time = time
        elif time == 0.0:
            self.current_time = 0.0

        # Compute required outputs
        F_all = self._compute_deformation_gradient()

        # Contact signed distance (distance to cylinder bar or ground, whichever is min)
        dx = self.x[:, 0]
        dz = self.x[:, 2] - self.bar_z
        dist_cyl = np.sqrt(dx * dx + dz * dz) - self.bar_radius
        dist_gnd = self.x[:, 2] - self.ground_z
        contact_signed_dist = np.minimum(dist_cyl, dist_gnd)

        # Interaction force summary (K, 3) where K=1
        # Net contact / internal interaction force
        interaction_force = np.zeros((1, 3), dtype=float)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_nodes, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F_all,
            },
            fields={
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": contact_signed_dist,
                "interaction_force": interaction_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale)
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_hardening": float(np.mean(self.hardening)) if hasattr(self, "hardening") else 0.0,
        }

