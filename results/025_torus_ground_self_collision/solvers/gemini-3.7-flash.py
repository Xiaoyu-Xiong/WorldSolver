"""BenchmarkSolver implementation for torus ground self collision."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Load geometry
        asset = Path(__file__).resolve().parent.parent / self.config["subject_asset"]
        with np.load(asset) as data:
            self.ref_pos = np.asarray(data["vertices"], dtype=float).copy()
            self.faces = np.asarray(data["faces"], dtype=int).copy()

        self.num_vertices = len(self.ref_pos)
        self.pos = self.ref_pos.copy()

        init_v = np.asarray(self.config.get("initial_velocity", [0.0, 0.0, -0.25]), dtype=float)
        self.vel = np.tile(init_v, (self.num_vertices, 1))

        self.total_mass = float(self.config.get("subject_mass", 1.8))
        self.mass = np.full(self.num_vertices, self.total_mass / self.num_vertices, dtype=float)
        self.inv_mass = 1.0 / self.mass

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))
        self.friction = float(self.config.get("friction", 0.58))
        self.restitution = float(self.config.get("restitution", 0.2))

        # Build edge springs: surface edges + bending edges + volumetric cross springs
        # Extract unique triangle edges
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))
        self.surface_edges = np.array(list(edges_set), dtype=int)

        # Find additional internal/bending connections based on rest distance
        # For a hollow mesh or torus surface, connect nearby vertices across the tube
        # Build KDTree on rest positions to find cross-tube and bending neighbors
        tree = cKDTree(self.ref_pos)
        minor_r = float(self.config.get("minor_radius", 0.17))
        # Search radius up to 2 * minor_radius * 1.15 to connect opposite sides of tube
        pairs = tree.query_pairs(r=minor_r * 2.2, output_type="ndarray")
        # Combine surface edges and internal pairs
        all_edges_set = set(edges_set)
        for p in pairs:
            u, v = int(p[0]), int(p[1])
            if u > v:
                u, v = v, u
            all_edges_set.add((u, v))

        all_edges_arr = np.array(list(all_edges_set), dtype=int)
        self.edge_i = all_edges_arr[:, 0]
        self.edge_j = all_edges_arr[:, 1]
        p_i = self.ref_pos[self.edge_i]
        p_j = self.ref_pos[self.edge_j]
        self.edge_rest_len = np.linalg.norm(p_i - p_j, axis=1)

        # Filter out zero length edges if any
        valid = self.edge_rest_len > 1e-6
        self.edge_i = self.edge_i[valid]
        self.edge_j = self.edge_j[valid]
        self.edge_rest_len = self.edge_rest_len[valid]

        # Edge stiffnesses: structural edges vs cross-tube internal edges
        is_surface = np.array([(u, v) in edges_set for u, v in zip(self.edge_i, self.edge_j)], dtype=bool)
        base_stiffness = float(self.config.get("stiffness", 13.0)) * 300.0
        self.edge_k = np.where(is_surface, base_stiffness, base_stiffness * 0.8)

        # Find 1-ring neighbors for computing per-vertex deformation gradient
        self.neighbors = [[] for _ in range(self.num_vertices)]
        for u, v in self.surface_edges:
            self.neighbors[u].append(v)
            self.neighbors[v].append(u)

        # Precompute rest shape matrices for per-vertex deformation gradient estimation
        self.rest_cov_inv = np.zeros((self.num_vertices, 3, 3), dtype=float)
        for i in range(self.num_vertices):
            nbrs = self.neighbors[i]
            if len(nbrs) >= 3:
                dx0 = self.ref_pos[nbrs] - self.ref_pos[i]  # [M, 3]
                cov0 = dx0.T @ dx0 + np.eye(3) * 1e-6
                self.rest_cov_inv[i] = np.linalg.inv(cov0)
            else:
                self.rest_cov_inv[i] = np.eye(3)

        self.current_time = 0.0
        self.sub_dt = 0.0005  # fine substepping for stable explicit integration
        self.damping = float(self.config.get("damping", 1.2))

        # Collision parameters
        self.self_collision_radius = float(self.config.get("contact_length_scale", 0.04)) * 0.8
        self.self_collision_stiffness = 25000.0

    def _step(self, dt: float) -> None:
        # 1. External forces (gravity)
        forces = np.tile(self.gravity * self.mass[:, None], (1, 1))

        # 2. Elastic spring forces
        pos_i = self.pos[self.edge_i]
        pos_j = self.pos[self.edge_j]
        disp = pos_j - pos_i
        dist = np.linalg.norm(disp, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        dir_vec = disp / dist_safe

        # Spring force: F_ij = k * (dist - rest_len) * dir
        delta_len = dist - self.edge_rest_len[:, None]
        f_spring = self.edge_k[:, None] * delta_len * dir_vec

        # Relative velocity for spring damping
        vel_i = self.vel[self.edge_i]
        vel_j = self.vel[self.edge_j]
        rel_vel = vel_j - vel_i
        rel_vel_proj = np.sum(rel_vel * dir_vec, axis=1, keepdims=True)
        f_damp = (self.damping * 20.0) * rel_vel_proj * dir_vec

        total_f_edge = f_spring + f_damp

        np.add.at(forces, self.edge_i, total_f_edge)
        np.add.at(forces, self.edge_j, -total_f_edge)

        # 3. Global velocity damping (air drag / internal dissipation)
        forces -= self.damping * 0.5 * self.vel * self.mass[:, None]

        # 4. Integrate velocity
        acc = forces * self.inv_mass[:, None]
        self.vel += acc * dt

        # 5. Integrate positions
        self.pos += self.vel * dt

        # 6. Self-collision penalty / impulse
        # Check spatial hashing / KDTree for nearby non-connected vertices
        tree = cKDTree(self.pos)
        r_col = self.self_collision_radius
        pairs = tree.query_pairs(r=r_col, output_type="ndarray")
        if len(pairs) > 0:
            p_u = pairs[:, 0]
            p_v = pairs[:, 1]
            # Exclude direct topological neighbors
            # Quick check rest distance: if rest distance is also small, it's a neighbor
            rest_d = np.linalg.norm(self.ref_pos[p_u] - self.ref_pos[p_v], axis=1)
            non_neighbor = rest_d > (r_col * 1.5)
            p_u = p_u[non_neighbor]
            p_v = p_v[non_neighbor]

            if len(p_u) > 0:
                pos_u = self.pos[p_u]
                pos_v = self.pos[p_v]
                disp_uv = pos_v - pos_u
                dist_uv = np.linalg.norm(disp_uv, axis=1, keepdims=True)
                dist_uv_safe = np.maximum(dist_uv, 1e-8)
                n_uv = disp_uv / dist_uv_safe
                penetration = r_col - dist_uv
                # Repulsion position correction
                corr = 0.5 * penetration * n_uv
                np.add.at(self.pos, p_u, -corr)
                np.add.at(self.pos, p_v, corr)

                # Relative velocity damping along normal
                v_rel = np.sum((self.vel[p_v] - self.vel[p_u]) * n_uv, axis=1, keepdims=True)
                # If approaching, apply impulse
                approaching = v_rel < 0
                impulse = -0.5 * v_rel * n_uv * approaching
                np.add.at(self.vel, p_u, -impulse)
                np.add.at(self.vel, p_v, impulse)

        # 7. Ground collision & friction
        penetration_mask = self.pos[:, 2] < self.ground_z
        if np.any(penetration_mask):
            # Positional projection to ground
            self.pos[penetration_mask, 2] = self.ground_z

            # Normal velocity reflection with restitution
            v_n = self.vel[penetration_mask, 2]
            moving_down = v_n < 0
            if np.any(moving_down):
                down_indices = np.where(penetration_mask)[0][moving_down]
                self.vel[down_indices, 2] = -self.restitution * self.vel[down_indices, 2]

                # Friction in tangential directions
                v_t = self.vel[down_indices, :2]
                v_t_norm = np.linalg.norm(v_t, axis=1, keepdims=True)
                normal_impulse = (1.0 + self.restitution) * np.abs(v_n[moving_down, None])
                friction_limit = self.friction * normal_impulse
                friction_scale = np.maximum(0.0, 1.0 - friction_limit / np.maximum(v_t_norm, 1e-8))
                self.vel[down_indices, :2] *= friction_scale

    def _compute_deformation_gradient(self) -> np.ndarray:
        F = np.tile(np.eye(3), (self.num_vertices, 1, 1))
        for i in range(self.num_vertices):
            nbrs = self.neighbors[i]
            if len(nbrs) >= 3:
                dx0 = self.ref_pos[nbrs] - self.ref_pos[i]  # [M, 3]
                dx = self.pos[nbrs] - self.pos[i]          # [M, 3]
                # F = (dx.T @ dx0) @ inv(dx0.T @ dx0)
                cov = dx.T @ dx0
                F[i] = cov @ self.rest_cov_inv[i]
        return F

    def advance_to(self, target_time: float) -> StateFrame:
        # Step simulation forward to target_time
        while self.current_time + 1e-9 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Construct required StateFrame
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        gap = self.pos[:, 2] - self.ground_z
        F = self._compute_deformation_gradient()

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": gravity_force,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_vertices,
            "num_edges": len(self.edge_i),
            "simulated_time": self.current_time,
        }

