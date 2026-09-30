"""Physical deformable solver for forty elastic tori falling into a pile."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        # Scene initial state
        init_frame = scene.initial_state()
        self.time = 0.0
        self.current_time = 0.0

        self.num_bodies = 40
        self.verts_per_body = 288
        self.total_verts = self.num_bodies * self.verts_per_body

        self.pos = np.copy(init_frame.position)
        self.vel = np.zeros_like(self.pos)
        self.mass = np.copy(init_frame.mass)
        self.body_id = np.copy(init_frame.body_id)
        self.orientation = np.copy(init_frame.orientation)

        # Parse torus faces for deformation gradient and volume conservation
        faces_list = []
        with open("assets/torus.obj", "r") as f:
            for line in f:
                if line.startswith("f "):
                    parts = line.strip().split()[1:]
                    faces_list.append([int(p.split("/")[0]) - 1 for p in parts])
        self.faces_single = np.array(faces_list, dtype=np.int32)  # (576, 3)
        self.num_faces_single = len(self.faces_single)

        # Rest local vertices of a single torus
        u = np.arange(24) * (2.0 * math.pi / 24)
        v = np.arange(12) * (2.0 * math.pi / 12)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        radial = 0.46 + 0.18 * np.cos(vv)
        self.local_rest = np.column_stack((
            (radial * np.cos(uu)).ravel(),
            (radial * np.sin(uu)).ravel(),
            (0.18 * np.sin(vv)).ravel(),
        ))  # (288, 3)

        # Internal springs / structural lattice inside each torus
        # 1. Surface edge springs
        edges = set()
        for tri in self.faces_single:
            for i in range(3):
                j = (i + 1) % 3
                i0, i1 = min(tri[i], tri[j]), max(tri[i], tri[j])
                edges.add((i0, i1))

        # 2. Cross-tube strut springs (connecting opposite sides across tube)
        # Grid indices: u in [0..23], v in [0..11] -> idx = u * 12 + v
        for ui in range(24):
            for vi in range(12):
                idx1 = ui * 12 + vi
                # Opposite on tube: vi + 6
                idx2 = ui * 12 + ((vi + 6) % 12)
                edges.add((min(idx1, idx2), max(idx1, idx2)))
                # Cross-section diagonals
                idx3 = ((ui + 1) % 24) * 12 + ((vi + 6) % 12)
                edges.add((min(idx1, idx3), max(idx1, idx3)))
                # Major ring struts (opposite across major ring)
                idx4 = ((ui + 12) % 24) * 12 + vi
                edges.add((min(idx1, idx4), max(idx1, idx4)))

        edge_arr = np.array(list(edges), dtype=np.int32)
        self.edges_single = edge_arr
        rest_diff = self.local_rest[edge_arr[:, 0]] - self.local_rest[edge_arr[:, 1]]
        self.edge_rest_lens = np.linalg.norm(rest_diff, axis=1)

        # Precompute inverted 2D reference coordinates for each face triangle
        # For deformation gradient F in tangent plane / 3D
        self.Dm_inv = []
        for tri in self.faces_single:
            v0, v1, v2 = self.local_rest[tri[0]], self.local_rest[tri[1]], self.local_rest[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-8:
                normal /= n_len
            # Form 3x3 reference matrix [e1, e2, normal]
            Dm = np.column_stack((e1, e2, normal))
            try:
                Dm_inv = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                Dm_inv = np.eye(3)
            self.Dm_inv.append(Dm_inv)
        self.Dm_inv = np.array(self.Dm_inv)  # (576, 3, 3)

        # All elements across all 40 tori
        self.total_elements = self.num_bodies * self.num_faces_single
        self.all_faces = np.zeros((self.total_elements, 3), dtype=np.int32)
        for b in range(self.num_bodies):
            self.all_faces[b * self.num_faces_single : (b + 1) * self.num_faces_single] = (
                self.faces_single + b * self.verts_per_body
            )

        # Posts configuration (12 cylinders)
        self.posts_xy = []
        self.post_radius = 0.13
        for i in range(12):
            angle = 2.0 * math.pi * i / 12.0
            x = 2.35 * math.cos(angle)
            y = 2.35 * math.sin(angle)
            self.posts_xy.append((x, y))
        self.posts_xy = np.array(self.posts_xy)  # (12, 2)
        self.posts_z_max = 2.8

        # Spatial subdivision for contact / particle radius
        # Approximate vertex collision sphere radius
        self.v_radius = 0.055

        # Budget tracking
        self.cumulative_external_impulse = np.zeros((self.num_bodies, 3), dtype=float)
        self.budget_external_force = np.zeros((self.num_bodies, 1, 3), dtype=float)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.body_mass = 0.90  # total per torus

        # Budget domain weights (40 bodies x 11520 vertices)
        self.domain_weights = np.zeros((self.num_bodies, self.total_verts), dtype=float)
        for b in range(self.num_bodies):
            self.domain_weights[b, b * self.verts_per_body : (b + 1) * self.verts_per_body] = 1.0

        # Substep dt
        self.sub_dt = 0.001

    def _step_substep(self, dt: float) -> None:
        pos = self.pos
        vel = self.vel
        n_bodies = self.num_bodies
        vpb = self.verts_per_body

        # 1. External forces (Gravity)
        g = self.gravity
        vel += g * dt

        # 2. Internal elastic forces (Position-based / Shape matching + Springs)
        # Apply shape matching + spring relaxation for fast stable elastodynamics
        # Shape matching per torus: preserves volume & rest shape with elastic stiffness
        alpha_sm = 0.45  # stiffness
        for b in range(n_bodies):
            idx_start = b * vpb
            idx_end = idx_start + vpb
            p = pos[idx_start:idx_end]
            v = vel[idx_start:idx_end]

            # Current center of mass
            cm = np.mean(p, axis=0)
            p_rel = p - cm
            r_rel = self.local_rest - np.mean(self.local_rest, axis=0)

            # Covariance matrix A = sum (p_rel * r_rel^T)
            A = r_rel.T @ p_rel
            # Optimal rotation via SVD
            U, S, Vt = np.linalg.svd(A)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            # Target matching positions
            p_target = cm + r_rel @ R.T

            # Update velocity and position towards target
            pos_corr = alpha_sm * (p_target - p)
            p += pos_corr
            v += pos_corr / dt

            # Spring edge constraints for local deformation details
            for edge_idx, (i0, i1) in enumerate(self.edges_single):
                d = p[i1] - p[i0]
                dist = np.linalg.norm(d)
                if dist > 1e-8:
                    rest_len = self.edge_rest_lens[edge_idx]
                    diff = (dist - rest_len) / dist
                    corr = 0.3 * diff * d
                    p[i0] += 0.5 * corr
                    p[i1] -= 0.5 * corr
                    v[i0] += 0.5 * corr / dt
                    v[i1] -= 0.5 * corr / dt

            pos[idx_start:idx_end] = p
            vel[idx_start:idx_end] = v

        # 3. Collision with Floor (z = 0)
        contact_impulse = np.zeros((n_bodies, 3), dtype=float)
        restitution = 0.1
        mu_friction = 0.4

        below_floor = pos[:, 2] < self.v_radius
        if np.any(below_floor):
            floor_indices = np.where(below_floor)[0]
            for idx in floor_indices:
                pen = self.v_radius - pos[idx, 2]
                pos[idx, 2] = self.v_radius
                vz = vel[idx, 2]
                if vz < 0:
                    imp_z = -(1.0 + restitution) * vz * self.mass[idx]
                    b_id = self.body_id[idx]
                    contact_impulse[b_id, 2] += imp_z
                    vel[idx, 2] = -restitution * vz
                    # Friction
                    v_horiz = vel[idx, :2]
                    v_norm = np.linalg.norm(v_horiz)
                    if v_norm > 1e-6:
                        fric_scale = max(0.0, 1.0 - mu_friction * abs(imp_z) / (self.mass[idx] * v_norm))
                        vel[idx, :2] *= fric_scale

        # 4. Collision with Vertical Posts
        for px, py in self.posts_xy:
            dx = pos[:, 0] - px
            dy = pos[:, 1] - py
            dist_sq = dx * dx + dy * dy
            r_col = self.post_radius + self.v_radius
            in_post = (dist_sq < r_col * r_col) & (pos[:, 2] <= self.posts_z_max) & (pos[:, 2] >= 0.0)
            if np.any(in_post):
                post_indices = np.where(in_post)[0]
                for idx in post_indices:
                    d = math.sqrt(dx[idx] * dx[idx] + dy[idx] * dy[idx])
                    if d > 1e-6:
                        nx = dx[idx] / d
                        ny = dy[idx] / d
                        pen = r_col - d
                        pos[idx, 0] += nx * pen
                        pos[idx, 1] += ny * pen
                        vn = vel[idx, 0] * nx + vel[idx, 1] * ny
                        if vn < 0:
                            b_id = self.body_id[idx]
                            imp_n = -(1.0 + restitution) * vn * self.mass[idx]
                            contact_impulse[b_id, 0] += imp_n * nx
                            contact_impulse[b_id, 1] += imp_n * ny
                            vel[idx, 0] -= (1.0 + restitution) * vn * nx
                            vel[idx, 1] -= (1.0 + restitution) * vn * ny
                            # Friction along post tangent & vertical
                            vel[idx, 2] *= 0.98

        # 5. Inter-body Torus-Torus Contact Collision
        # Fast bounding sphere check per torus pair
        centers = np.zeros((n_bodies, 3), dtype=float)
        for b in range(n_bodies):
            centers[b] = np.mean(pos[b * vpb : (b + 1) * vpb], axis=0)

        # Torus bounding radius ~ 0.46 + 0.18 = 0.64
        R_bound = 0.68
        for b1 in range(n_bodies):
            for b2 in range(b1 + 1, n_bodies):
                d_cnt = centers[b1] - centers[b2]
                if np.sum(d_cnt * d_cnt) < (2.0 * R_bound) ** 2:
                    # Detailed particle-particle collision between body b1 and b2
                    p1 = pos[b1 * vpb : (b1 + 1) * vpb]
                    p2 = pos[b2 * vpb : (b2 + 1) * vpb]
                    v1 = vel[b1 * vpb : (b1 + 1) * vpb]
                    v2 = vel[b2 * vpb : (b2 + 1) * vpb]

                    diff = p1[:, None, :] - p2[None, :, :]  # (288, 288, 3)
                    dist_sq = np.sum(diff * diff, axis=-1)  # (288, 288)
                    min_dist = 2.0 * self.v_radius
                    close = dist_sq < (min_dist * min_dist)
                    if np.any(close):
                        i_idxs, j_idxs = np.where(close)
                        for i_c, j_c in zip(i_idxs, j_idxs):
                            d_vec = p1[i_c] - p2[j_c]
                            d_len = np.linalg.norm(d_vec)
                            if d_len < 1e-6:
                                d_vec = np.array([0.0, 0.0, 1.0])
                                d_len = 1e-6
                            n_col = d_vec / d_len
                            overlap = min_dist - d_len
                            corr = 0.5 * overlap * n_col
                            p1[i_c] += corr
                            p2[j_c] -= corr

                            rel_v = v1[i_c] - v2[j_c]
                            vn = np.dot(rel_v, n_col)
                            if vn < 0:
                                impulse = -(1.0 + restitution) * 0.5 * vn * n_col
                                v1[i_c] += impulse
                                v2[j_c] -= impulse
                                contact_impulse[b1] += impulse * self.mass[0]
                                contact_impulse[b2] -= impulse * self.mass[0]

        # 6. Euler integration update
        pos += vel * dt
        # Slight damping for physical realism / air resistance
        vel *= (1.0 - 0.002)

        # Track external forces & cumulative impulse
        self.budget_external_force[:, 0, :] = self.body_mass * self.gravity + contact_impulse / dt
        self.cumulative_external_impulse += contact_impulse + (self.body_mass * self.gravity) * dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Advance simulation in small substeps up to target_time
        while self.current_time < target_time - 1e-8:
            step = min(self.sub_dt, target_time - self.current_time)
            self._step_substep(step)
            self.current_time += step

        # Compute element deformation gradients F for all 40 * 576 elements
        # F = Ds * Dm_inv
        Ne = self.total_elements
        F = np.zeros((Ne, 3, 3), dtype=float)

        for b in range(self.num_bodies):
            p = self.pos[b * self.verts_per_body : (b + 1) * self.verts_per_body]
            b_offset = b * self.num_faces_single
            for f_idx, tri in enumerate(self.faces_single):
                v0, v1, v2 = p[tri[0]], p[tri[1]], p[tri[2]]
                e1 = v1 - v0
                e2 = v2 - v0
                normal = np.cross(e1, e2)
                n_len = np.linalg.norm(normal)
                if n_len > 1e-8:
                    normal /= n_len
                Ds = np.column_stack((e1, e2, normal))
                Dm_inv = self.Dm_inv[f_idx]
                F[b_offset + f_idx] = Ds @ Dm_inv

        # Constraint values: volume / area ratio per body
        constraint_val = np.ones(self.num_bodies, dtype=float)
        constraint_tgt = np.ones(self.num_bodies, dtype=float)
        for b in range(self.num_bodies):
            # Compute determinant of deformation gradient average
            dets = np.linalg.det(F[b * self.num_faces_single : (b + 1) * self.num_faces_single])
            constraint_val[b] = float(np.mean(dets))

        # Compute torus orientations
        orientations = np.zeros((self.total_verts, 4), dtype=float)
        for b in range(self.num_bodies):
            p = self.pos[b * self.verts_per_body : (b + 1) * self.verts_per_body]
            cm = np.mean(p, axis=0)
            p_rel = p - cm
            r_rel = self.local_rest - np.mean(self.local_rest, axis=0)
            A = r_rel.T @ p_rel
            U, S, Vt = np.linalg.svd(A)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            # Convert R to quaternion [x, y, z, w]
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                S_q = math.sqrt(tr + 1.0) * 2
                qw = 0.25 * S_q
                qx = (R[2, 1] - R[1, 2]) / S_q
                qy = (R[0, 2] - R[2, 0]) / S_q
                qz = (R[1, 0] - R[0, 1]) / S_q
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S_q = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S_q
                qx = 0.25 * S_q
                qy = (R[0, 1] + R[1, 0]) / S_q
                qz = (R[0, 2] + R[2, 0]) / S_q
            elif R[1, 1] > R[2, 2]:
                S_q = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S_q
                qx = (R[0, 1] + R[1, 0]) / S_q
                qy = 0.25 * S_q
                qz = (R[1, 2] + R[2, 1]) / S_q
            else:
                S_q = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S_q
                qx = (R[0, 2] + R[2, 0]) / S_q
                qy = (R[1, 2] + R[2, 1]) / S_q
                qz = 0.25 * S_q
            q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            q = np.array([qx / q_norm, qy / q_norm, qz / q_norm, qw / q_norm])
            orientations[b * self.verts_per_body : (b + 1) * self.verts_per_body] = q

        frame = StateFrame(
            time=float(target_time),
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            orientation=orientations,
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "mass_quantity": np.copy(self.mass),
            },
            fields={
                "budget_cumulative_external_impulse": np.copy(self.cumulative_external_impulse),
                "budget_cumulative_inflow_momentum": np.zeros((self.num_bodies, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((self.num_bodies, 3), dtype=float),
                "budget_domain_weights": np.copy(self.domain_weights),
                "budget_external_force": np.copy(self.budget_external_force),
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": F,
            },
            scalars={
                "constraint_scale": 1.0,
                "reference_momentum": float(self.body_mass * 9.81 * 1.0),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_bodies": self.num_bodies,
            "total_verts": self.total_verts,
        }

