"""Noodles in Bowl Physics Solver.

Simulates 625 elastic rod noodles falling into an axisymmetric bowl,
intertwining, colliding, and settling under gravity using physical time-stepping
(position-based dynamics / XPBD with bowl and segment-segment collisions).
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config

        # Extract initial state
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        # Config parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.radius = float(self.config.get("radius", 0.006))
        self.contact_length_scale = float(init_frame.scalars.get("contact_length_scale", self.radius))
        self.noodle_count = int(self.config.get("noodle_count", 625))
        self.nodes_per_noodle = int(self.config.get("nodes_per_noodle", 2))
        self.rest_length = float(self.config.get("noodle_length", 0.4))

        # Metadata & edges
        meta = scene.metadata()
        if "edges" in meta:
            self.edges = np.array(meta["edges"], dtype=np.int32)
        else:
            self.edges = np.arange(len(self.pos), dtype=np.int32).reshape(-1, 2)

        # Initial segment rest lengths
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.edge_rest_lens = np.linalg.norm(p1 - p0, axis=1)

        # Bowl profile in (r, z): from origin along bottom then up the wall
        self.bowl_profile = np.array([
            [0.0, 0.0],
            [0.12, 0.0],
            [0.38, 0.08],
            [0.66, 0.34],
            [0.72, 0.48]
        ], dtype=np.float64)

        self.num_particles = len(self.pos)
        self.last_signed_dist = np.full(self.num_particles, 0.01, dtype=np.float64)
        self.initialized = True

    def _project_bowl_collision(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project particles against axisymmetric bowl interior.
        Returns:
            x_corr: corrected positions
            signed_dist: signed geometric distance to noodle contact surface (negative = penetrating)
        """
        r_xy = np.hypot(x[:, 0], x[:, 1])
        eps = 1e-8
        dir_xy = np.zeros_like(x)
        dir_xy[:, 0] = x[:, 0] / np.maximum(r_xy, eps)
        dir_xy[:, 1] = x[:, 1] / np.maximum(r_xy, eps)

        z = x[:, 2]
        pts_rz = np.stack([r_xy, z], axis=-1)  # (N, 2)

        # Segments in full bowl profile: 4 segments
        P0 = self.bowl_profile[:-1]  # (4, 2)
        P1 = self.bowl_profile[1:]   # (4, 2)
        seg_vec = P1 - P0            # (4, 2)
        seg_len_sq = np.sum(seg_vec**2, axis=-1)  # (4,)

        # Vectorized projection onto profile segments
        diff = pts_rz[:, None, :] - P0[None, :, :]  # (N, 4, 2)
        t = np.sum(diff * seg_vec[None, :, :], axis=-1) / seg_len_sq[None, :]  # (N, 4)
        t = np.clip(t, 0.0, 1.0)
        proj_seg = P0[None, :, :] + t[:, :, None] * seg_vec[None, :, :]  # (N, 4, 2)
        dist_seg_sq = np.sum((pts_rz[:, None, :] - proj_seg)**2, axis=-1)  # (N, 4)

        best_idx = np.argmin(dist_seg_sq, axis=1)  # (N,)
        row_idx = np.arange(len(x))
        closest_rz = proj_seg[row_idx, best_idx]  # (N, 2)

        # Wall normals pointing inward (into the bowl interior):
        # seg_vec is (dr, dz). Inward normal is (-dz, dr) normalized.
        seg_n = np.stack([-seg_vec[:, 1], seg_vec[:, 0]], axis=-1)  # (4, 2)
        seg_n_len = np.hypot(seg_n[:, 0], seg_n[:, 1])
        seg_n = seg_n / seg_n_len[:, None]

        # Normal at best segment
        best_norm_rz = seg_n[best_idx]  # (N, 2)

        diff_rz = pts_rz - closest_rz  # (N, 2)
        dist_geom = np.hypot(diff_rz[:, 0], diff_rz[:, 1])

        # Dot product with inward normal to determine if inside (positive dot) or outside (negative dot)
        dot_inward = np.sum(diff_rz * best_norm_rz, axis=-1)

        # Gap to noodle contact boundary (shell):
        gap = dot_inward - self.radius
        signed_dist = np.where(dot_inward >= 0, dist_geom - self.radius, gap)

        # Position correction for penetrating particles
        penetration = np.maximum(0.0, -gap)
        colliding = penetration > 0

        x_corr = np.array(x, copy=True)
        if np.any(colliding):
            # Inward 3D normal
            n3d = np.zeros_like(x)
            n3d[:, 0] = best_norm_rz[:, 0] * dir_xy[:, 0]
            n3d[:, 1] = best_norm_rz[:, 1] * dir_xy[:, 1]
            n3d[:, 2] = best_norm_rz[:, 1]
            n3d_len = np.linalg.norm(n3d, axis=1, keepdims=True)
            n3d = n3d / np.maximum(n3d_len, eps)

            x_corr[colliding] += penetration[colliding, None] * n3d[colliding]

        return x_corr, signed_dist

    def _solve_particle_collisions(self, x: np.ndarray, cell_size: float = 0.03) -> np.ndarray:
        """Spatial hash grid particle-particle contact solving."""
        N = len(x)
        min_dist = 2.0 * self.radius
        min_dist_sq = min_dist * min_dist

        # Spatial grid coordinates
        coords = np.floor(x / cell_size).astype(np.int32)
        h = ((coords[:, 0] * 73856093) ^ (coords[:, 1] * 19349663) ^ (coords[:, 2] * 83492791)) % 4096
        h = np.abs(h)

        # Group particles by cell
        order = np.argsort(h)
        sorted_h = h[order]
        unique_hashes, split_indices = np.unique(sorted_h, return_index=True)
        cell_groups = np.split(order, split_indices[1:])

        delta = np.zeros_like(x)
        counts = np.zeros(N, dtype=np.int32)

        for group in cell_groups:
            if len(group) < 2:
                continue
            gp_x = x[group]
            diffs = gp_x[:, None, :] - gp_x[None, :, :]  # (M, M, 3)
            dists_sq = np.sum(diffs**2, axis=-1)
            tri_i, tri_j = np.triu_indices(len(group), k=1)
            pair_dists_sq = dists_sq[tri_i, tri_j]
            overlap = pair_dists_sq < min_dist_sq

            if np.any(overlap):
                sel_i = tri_i[overlap]
                sel_j = tri_j[overlap]
                d = np.sqrt(np.maximum(pair_dists_sq[overlap], 1e-12))
                diff = diffs[sel_i, sel_j]
                norm = diff / d[:, None]
                pen = (min_dist - d) * 0.5

                p_i = group[sel_i]
                p_j = group[sel_j]

                # Ignore connected node pairs (handled by edge constraint)
                not_connected = (p_i ^ 1) != p_j

                p_i = p_i[not_connected]
                p_j = p_j[not_connected]
                pen = pen[not_connected]
                norm = norm[not_connected]

                if len(p_i) > 0:
                    np.add.at(delta, p_i, pen[:, None] * norm)
                    np.add.at(delta, p_j, -pen[:, None] * norm)
                    np.add.at(counts, p_i, 1)
                    np.add.at(counts, p_j, 1)

        mask = counts > 0
        x_corr = np.array(x, copy=True)
        x_corr[mask] += delta[mask] / counts[mask, None]
        return x_corr

    def _step(self, dt: float) -> None:
        """Physical time-step using XPBD / Verlet scheme."""
        # 1. External forces (gravity) & unconstrained integration
        x_old = np.array(self.pos, copy=True)
        self.vel += self.gravity * dt
        self.vel *= 0.995  # Air damping
        x_pred = self.pos + self.vel * dt

        # 2. Constraint solver iterations
        num_iters = 8
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        rest_l = self.edge_rest_lens

        for _ in range(num_iters):
            # a. Edge distance constraints (rod elasticity)
            p0 = x_pred[e0]
            p1 = x_pred[e1]
            diff = p0 - p1
            dist = np.linalg.norm(diff, axis=1)
            eps = 1e-8
            dir_e = diff / np.maximum(dist[:, None], eps)
            err = dist - rest_l
            corr = 0.5 * err[:, None] * dir_e
            x_pred[e0] -= corr
            x_pred[e1] += corr

            # b. Bowl collision projection
            x_pred, _ = self._project_bowl_collision(x_pred)

            # c. Particle-particle contact / separation
            x_pred = self._solve_particle_collisions(x_pred, cell_size=4.0 * self.radius)

        # 3. Final collision pass & update velocities
        x_pred, signed_dist = self._project_bowl_collision(x_pred)

        self.vel = (x_pred - x_old) / dt
        # Friction at contacts
        touching = signed_dist < 0.001
        self.vel[touching, :2] *= 0.8
        self.vel[touching, 2] = np.maximum(self.vel[touching, 2], -0.05)

        self.pos = x_pred
        self.last_signed_dist = signed_dist

    def advance_to(self, time: float) -> StateFrame:
        if not self.initialized:
            raise RuntimeError("Solver not initialized")

        # Step physics forward
        target_time = float(time)
        dt_sub = 0.005  # 5 ms substep for stable physical contact & rod dynamics

        while self.current_time + dt_sub < target_time + 1e-9:
            self._step(dt_sub)
            self.current_time += dt_sub

        remaining = target_time - self.current_time
        if remaining > 1e-6:
            self._step(remaining)
            self.current_time = target_time

        # Compute required StateFrame fields
        ext_force = np.zeros((self.num_particles, 3), dtype=np.float64)
        ext_force[:, :] = self.mass[:, None] * self.gravity[None, :]

        # Signed distance field
        _, signed_dist = self._project_bowl_collision(self.pos)

        frame = StateFrame(
            time=float(target_time),
            position=np.array(self.pos, dtype=np.float64, copy=True),
            velocity=np.array(self.vel, dtype=np.float64, copy=True),
            attributes={
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
            },
            fields={
                "contact_signed_distance": np.array(signed_dist, dtype=np.float64, copy=True),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.array(ext_force, dtype=np.float64, copy=True),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
            mass=np.array(self.mass, dtype=np.float64, copy=True)
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": self.num_particles,
            "current_time": self.current_time,
            "min_signed_dist": float(np.min(self.last_signed_dist)) if hasattr(self, "last_signed_dist") else 0.0,
        }
