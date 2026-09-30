"""Cloth Sphere Drape Solver with Strain Limiting and Contact Handling.

Recreates the Figure 15(c) C-IPC benchmark: soft cloth falling onto a fixed sphere,
draping with folds/wrinkles under principal strain limits [0.9, 1.1] and sphere/ground contact.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.meta = scene.metadata()
        init_frame = scene.initial_state()

        self.time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        cloth_cfg = self.config.get("cloth", {})
        self.grid_shape = tuple(cloth_cfg.get("grid_shape", [31, 31]))
        self.n_rows, self.n_cols = self.grid_shape
        self.n_verts = self.pos.shape[0]

        # Geometry & physics params
        self.side_length = float(cloth_cfg.get("side_length", 2.8))
        self.thickness = float(cloth_cfg.get("thickness", 0.012))
        self.contact_offset = 0.5 * self.thickness
        self.damping = float(cloth_cfg.get("damping", 1.35))
        self.friction = float(cloth_cfg.get("friction", 0.22))

        # Strain bounds
        self.strain_lower = float(cloth_cfg.get("strain_lower", 0.9))
        self.strain_upper = float(cloth_cfg.get("strain_upper", 1.1))

        # Sphere obstacle
        sphere_cfg = self.config.get("sphere", {})
        self.sphere_center = np.array(sphere_cfg.get("center", [0.0, 0.0, 0.64]), dtype=np.float64)
        self.sphere_radius = float(sphere_cfg.get("radius", 0.64))

        # Ground obstacle
        ground_cfg = self.config.get("ground", {})
        self.ground_height = float(ground_cfg.get("height", 0.0))

        # Gravity
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Solver params
        solver_cfg = self.config.get("solver", {})
        self.dt_substep = float(solver_cfg.get("substep", 0.004))
        self.constraint_iterations = int(solver_cfg.get("constraint_iterations", 15))

        # Topology and reference state
        self.faces = np.array(self.meta["topology"]["faces"], dtype=np.int32)
        self.num_faces = len(self.faces)

        # Build 2D rest coordinates in UV / planar space for deformation gradient
        # Grid covers [-side_length/2, side_length/2] in x and y
        xs = np.linspace(-0.5 * self.side_length, 0.5 * self.side_length, self.n_cols)
        ys = np.linspace(-0.5 * self.side_length, 0.5 * self.side_length, self.n_rows)
        gx, gy = np.meshgrid(xs, ys)
        self.rest_uv = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # [N, 2]

        # Precompute rest inverse matrices for triangle deformation gradient
        # For each triangle (i0, i1, i2), D_m = [uv1 - uv0, uv2 - uv0] in R^(2x2)
        i0 = self.faces[:, 0]
        i1 = self.faces[:, 1]
        i2 = self.faces[:, 2]
        Dm = np.stack([self.rest_uv[i1] - self.rest_uv[i0], self.rest_uv[i2] - self.rest_uv[i0]], axis=-1)  # [F, 2, 2]
        self.Dm_inv = np.linalg.inv(Dm)  # [F, 2, 2]

        # Distance constraints: structural, shear, and bending springs
        edges_set = set()
        edges_list = []
        rest_lens = []

        # Structural edges from grid
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                idx = r * self.n_cols + c
                if c + 1 < self.n_cols:
                    edges_set.add((min(idx, idx + 1), max(idx, idx + 1)))
                if r + 1 < self.n_rows:
                    edges_set.add((min(idx, idx + self.n_cols), max(idx, idx + self.n_cols)))

        # Add all triangle edges (includes shear)
        for f in self.faces:
            for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                edges_set.add((min(a, b), max(a, b)))

        # Bending edges (2-hop grid neighbors)
        bend_edges_set = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                idx = r * self.n_cols + c
                if c + 2 < self.n_cols:
                    bend_edges_set.add((min(idx, idx + 2), max(idx, idx + 2)))
                if r + 2 < self.n_rows:
                    bend_edges_set.add((min(idx, idx + 2 * self.n_cols), max(idx, idx + 2 * self.n_cols)))

        # Convert to numpy arrays
        edges_arr = np.array(list(edges_set), dtype=np.int32)
        dist_rest = np.linalg.norm(self.pos[edges_arr[:, 0]] - self.pos[edges_arr[:, 1]], axis=1)
        self.edges = edges_arr
        self.edge_rest_lens = dist_rest

        bend_arr = np.array(list(bend_edges_set), dtype=np.int32)
        bend_dist_rest = np.linalg.norm(self.pos[bend_arr[:, 0]] - self.pos[bend_arr[:, 1]], axis=1)
        self.bend_edges = bend_arr
        self.bend_rest_lens = bend_dist_rest

        # Material properties
        # Low membrane stiffness as described: soft cloth (0.01x) with strain limiting
        self.k_stretch = 0.6
        self.k_bend = 0.05

        # Conserved scalars / fields
        self.contact_length_scale = float(self.thickness)
        self.cumulative_mass_source = np.array([0.0], dtype=np.float64)

    def _compute_contact_distance(self) -> np.ndarray:
        # Distance from each vertex to sphere surface: ||p - c|| - R
        diff = self.pos - self.sphere_center
        dists = np.linalg.norm(diff, axis=1) - self.sphere_radius
        return dists

    def _compute_deformation_gradient(self) -> np.ndarray:
        # Per-triangle deformation gradient F = [Ds] @ Dm_inv
        # Ds = [x1 - x0, x2 - x0] in R^(3x2)
        i0 = self.faces[:, 0]
        i1 = self.faces[:, 1]
        i2 = self.faces[:, 2]
        Ds = np.stack([self.pos[i1] - self.pos[i0], self.pos[i2] - self.pos[i0]], axis=-1)  # [F, 3, 2]
        F_2d = np.matmul(Ds, self.Dm_inv)  # [F, 3, 2]

        # Normal to deformed face
        face_normals = np.cross(Ds[:, :, 0], Ds[:, :, 1])  # [F, 3]
        fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
        fn_norm = np.maximum(fn_norm, 1e-12)
        n_hat = face_normals / fn_norm  # [F, 3]

        # Full 3x3 deformation gradient F = [F_col0, F_col1, n_hat]
        F = np.stack([F_2d[:, :, 0], F_2d[:, :, 1], n_hat], axis=-1)  # [F, 3, 3]
        return F

    def _substep(self, dt: float) -> None:
        # 1. External forces (gravity) and velocity damping
        acc = np.broadcast_to(self.gravity, self.pos.shape)
        self.vel += acc * dt
        # Velocity damping
        damping_factor = max(0.0, 1.0 - self.damping * dt)
        self.vel *= damping_factor

        # 2. Position prediction
        pos_prev = self.pos.copy()
        pos_pred = self.pos + self.vel * dt

        # 3. Constraint projection (PBD iterations)
        for _ in range(self.constraint_iterations):
            # A. Sphere and ground collision projections
            # Sphere collision
            diff = pos_pred - self.sphere_center
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            normal = diff / dist_safe
            min_dist = self.sphere_radius + self.contact_offset
            penetration = min_dist - dist
            in_sphere = (penetration > 0).ravel()
            if np.any(in_sphere):
                # Correct position out of sphere
                pos_pred[in_sphere] += normal[in_sphere] * penetration[in_sphere]

            # Ground collision
            min_z = self.ground_height + self.contact_offset
            under_ground = pos_pred[:, 2] < min_z
            if np.any(under_ground):
                pos_pred[under_ground, 2] = min_z

            # B. Strain limiting & stretch constraints on edges
            i0 = self.edges[:, 0]
            i1 = self.edges[:, 1]
            p0 = pos_pred[i0]
            p1 = pos_pred[i1]
            d = p1 - p0
            current_len = np.linalg.norm(d, axis=1, keepdims=True)
            current_len_safe = np.maximum(current_len, 1e-10)
            dir_vec = d / current_len_safe

            # Target strain range [strain_lower, strain_upper]
            L0 = self.edge_rest_lens[:, None]
            min_len = L0 * self.strain_lower
            max_len = L0 * self.strain_upper

            # Elastic stretch toward rest length + hard clamp to [min_len, max_len]
            # First calculate desired length: soft elasticity within [min_len, max_len]
            target_len = np.clip(current_len + self.k_stretch * (L0 - current_len), min_len, max_len)
            diff_len = current_len - target_len

            w0 = self.inv_mass[i0, None]
            w1 = self.inv_mass[i1, None]
            w_sum = w0 + w1
            w_sum_safe = np.maximum(w_sum, 1e-10)

            delta = (diff_len / w_sum_safe) * dir_vec
            corr0 = w0 * delta
            corr1 = -w1 * delta

            # Accumulate corrections
            np.add.at(pos_pred, i0, corr0)
            np.add.at(pos_pred, i1, corr1)

            # C. Bending constraints (softer)
            if len(self.bend_edges) > 0:
                bi0 = self.bend_edges[:, 0]
                bi1 = self.bend_edges[:, 1]
                bp0 = pos_pred[bi0]
                bp1 = pos_pred[bi1]
                bd = bp1 - bp0
                b_cur_len = np.linalg.norm(bd, axis=1, keepdims=True)
                b_cur_safe = np.maximum(b_cur_len, 1e-10)
                b_dir = bd / b_cur_safe
                bL0 = self.bend_rest_lens[:, None]
                b_diff = self.k_bend * (b_cur_len - bL0)

                bw0 = self.inv_mass[bi0, None]
                bw1 = self.inv_mass[bi1, None]
                bw_sum = np.maximum(bw0 + bw1, 1e-10)

                b_delta = (b_diff / bw_sum) * b_dir
                np.add.at(pos_pred, bi0, bw0 * b_delta)
                np.add.at(pos_pred, bi1, -bw1 * b_delta)

        # Final collision correction
        diff = pos_pred - self.sphere_center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        normal = diff / dist_safe
        min_dist = self.sphere_radius + self.contact_offset
        penetration = min_dist - dist
        in_sphere = (penetration > 0).ravel()
        if np.any(in_sphere):
            pos_pred[in_sphere] += normal[in_sphere] * penetration[in_sphere]

        under_ground = pos_pred[:, 2] < (self.ground_height + self.contact_offset)
        if np.any(under_ground):
            pos_pred[under_ground, 2] = self.ground_height + self.contact_offset

        # 4. Update velocity and apply friction at contact
        self.vel = (pos_pred - pos_prev) / dt
        self.pos = pos_pred

        # Friction for colliding vertices
        if np.any(in_sphere):
            norm_sphere = normal[in_sphere]  # [M, 3]
            v_sphere = self.vel[in_sphere]  # [M, 3]
            v_dot_n = np.sum(v_sphere * norm_sphere, axis=1, keepdims=True)
            v_normal = v_dot_n * norm_sphere
            v_tangent = v_sphere - v_normal
            # Apply tangential friction damping
            fric_factor = max(0.0, 1.0 - self.friction)
            self.vel[in_sphere] = v_normal + fric_factor * v_tangent

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        while self.time + 1e-8 < target_time:
            dt = min(self.dt_substep, target_time - self.time)
            self._substep(dt)
            self.time += dt

        self.time = target_time
        return self._make_state_frame()

    def _make_state_frame(self) -> StateFrame:
        F = self._compute_deformation_gradient()
        contact_sd = self._compute_contact_distance()

        frame = StateFrame(
            self.time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        contact_sd = self._compute_contact_distance()
        return {
            "time": self.time,
            "min_contact_dist": float(np.min(contact_sd)),
            "max_z": float(np.max(self.pos[:, 2])),
            "min_z": float(np.min(self.pos[:, 2])),
        }

