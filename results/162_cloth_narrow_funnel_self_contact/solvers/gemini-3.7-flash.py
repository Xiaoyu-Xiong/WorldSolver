"""Physically-based cloth solver for narrow funnel simulation with self-contact and boundary collisions."""
from __future__ import annotations

import json
import os
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = None
        self.current_time = 0.0

        # State
        self.pos = None
        self.vel = None
        self.mass = None
        self.inv_mass = None

        # Cloth topology and geometry
        self.nx = 17
        self.ny = 17
        self.dx = 0.0
        self.dy = 0.0
        self.thickness = 0.015
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.boundary_mask = None

        # Constraints / Springs
        self.struct_edges = None
        self.struct_rest_lens = None
        self.shear_edges = None
        self.shear_rest_lens = None
        self.bend_edges = None
        self.bend_rest_lens = None

        # Funnel geometry
        self.funnel_tris_a = None
        self.funnel_tris_b = None
        self.funnel_tris_c = None
        self.funnel_normals = None

        # Ground
        self.ground_z = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        np.random.seed(self.seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-12)

        cloth_cfg = scene.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        sx, sy = map(float, cloth_cfg["size"])
        self.dx = sx / (self.nx - 1)
        self.dy = sy / (self.ny - 1)
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        if "boundary_vertex" in init_frame.attributes:
            self.boundary_mask = np.array(init_frame.attributes["boundary_vertex"], dtype=np.int32)
        else:
            self.boundary_mask = np.zeros(len(self.pos), dtype=np.int32)

        # Build constraints from grid topology
        struct_e, shear_e, bend_e = [], [], []

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                # Structural (horizontal and vertical)
                if i + 1 < self.nx:
                    struct_e.append([idx, j * self.nx + (i + 1)])
                if j + 1 < self.ny:
                    struct_e.append([idx, (j + 1) * self.nx + i])
                # Shear (diagonals)
                if i + 1 < self.nx and j + 1 < self.ny:
                    shear_e.append([idx, (j + 1) * self.nx + (i + 1)])
                if i > 0 and j + 1 < self.ny:
                    shear_e.append([idx, (j + 1) * self.nx + (i - 1)])
                # Bending (step 2)
                if i + 2 < self.nx:
                    bend_e.append([idx, j * self.nx + (i + 2)])
                if j + 2 < self.ny:
                    bend_e.append([idx, (j + 2) * self.nx + i])

        self.struct_edges = np.array(struct_e, dtype=np.int32)
        self.struct_rest_lens = np.linalg.norm(self.pos[self.struct_edges[:, 0]] - self.pos[self.struct_edges[:, 1]], axis=1)

        self.shear_edges = np.array(shear_e, dtype=np.int32)
        self.shear_rest_lens = np.linalg.norm(self.pos[self.shear_edges[:, 0]] - self.pos[self.shear_edges[:, 1]], axis=1)

        self.bend_edges = np.array(bend_e, dtype=np.int32)
        self.bend_rest_lens = np.linalg.norm(self.pos[self.bend_edges[:, 0]] - self.pos[self.bend_edges[:, 1]], axis=1)

        # Load funnel geometry
        funnel_path = os.path.join(os.path.dirname(__file__), "..", "assets", "funnel.obj")
        if not os.path.exists(funnel_path):
            funnel_path = "assets/funnel.obj"

        verts, faces = [], []
        with open(funnel_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append(list(map(float, line.split()[1:4])))
                elif line.startswith("f "):
                    faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])

        funnel_v = np.array(verts, dtype=np.float64)
        funnel_f = np.array(faces, dtype=np.int32)

        self.funnel_tris_a = funnel_v[funnel_f[:, 0]]
        self.funnel_tris_b = funnel_v[funnel_f[:, 1]]
        self.funnel_tris_c = funnel_v[funnel_f[:, 2]]

        fn = np.cross(self.funnel_tris_b - self.funnel_tris_a, self.funnel_tris_c - self.funnel_tris_a)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
        self.funnel_normals = np.where(fn_norm > 1e-12, fn / np.maximum(fn_norm, 1e-12), np.array([0.0, 0.0, 1.0]))

    def _project_funnel(self, points):
        """Vectorized projection of points onto all funnel triangles.
        Returns: (min_dist, closest_point, closest_face_normal)
        """
        N = points.shape[0]
        P = points[:, None, :]
        A = self.funnel_tris_a[None, :, :]
        B = self.funnel_tris_b[None, :, :]
        C = self.funnel_tris_c[None, :, :]

        ab = B - A
        ac = C - A
        ap = P - A

        d1 = np.sum(ab * ap, axis=-1)
        d2 = np.sum(ac * ap, axis=-1)
        bp = P - B
        d3 = np.sum(ab * bp, axis=-1)
        d4 = np.sum(ac * bp, axis=-1)
        cp = P - C
        d5 = np.sum(ab * cp, axis=-1)
        d6 = np.sum(ac * cp, axis=-1)

        m_a = (d1 <= 0) & (d2 <= 0)
        m_b = (d3 >= 0) & (d4 <= d3)
        m_c = (d6 >= 0) & (d5 <= d6)

        vc = d1 * d4 - d3 * d2
        m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        v_ab = np.where(m_ab, d1 / np.maximum(d1 - d3, 1e-12), 0.0)[:, :, None]

        vb = d5 * d2 - d1 * d6
        m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        w_ac = np.where(m_ac, d2 / np.maximum(d2 - d6, 1e-12), 0.0)[:, :, None]

        va = d3 * d6 - d5 * d4
        m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        w_bc = np.where(m_bc, (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-12), 0.0)[:, :, None]

        denom = 1.0 / np.maximum(va + vb + vc, 1e-12)
        v_face = (vb * denom)[:, :, None]
        w_face = (vc * denom)[:, :, None]

        proj = np.where(m_a[:, :, None], A,
               np.where(m_b[:, :, None], B,
               np.where(m_c[:, :, None], C,
               np.where(m_ab[:, :, None], A + v_ab * ab,
               np.where(m_ac[:, :, None], A + w_ac * ac,
               np.where(m_bc[:, :, None], B + w_bc * (C - B),
               A + v_face * ab + w_face * ac))))))

        diff = P - proj
        dist_sq = np.sum(diff * diff, axis=-1)
        min_idx = np.argmin(dist_sq, axis=1)

        closest_p = proj[np.arange(N), min_idx]
        closest_dist = np.sqrt(dist_sq[np.arange(N), min_idx])
        closest_norm = self.funnel_normals[min_idx]

        return closest_dist, closest_p, closest_norm

    def _solve_distance_constraints(self, p, edges, rest_lens, stiffness):
        i1 = edges[:, 0]
        i2 = edges[:, 1]

        p1 = p[i1]
        p2 = p[i2]
        delta = p1 - p2
        dist = np.linalg.norm(delta, axis=1)

        mask = dist > 1e-9
        dir_vec = np.zeros_like(delta)
        dir_vec[mask] = delta[mask] / dist[mask, None]

        w1 = self.inv_mass[i1]
        w2 = self.inv_mass[i2]
        w_sum = w1 + w2
        w_sum = np.maximum(w_sum, 1e-12)

        correction_mag = stiffness * (dist - rest_lens) / w_sum
        corr = dir_vec * correction_mag[:, None]

        np.add.at(p, i1, -corr * w1[:, None])
        np.add.at(p, i2, corr * w2[:, None])

    def _solve_self_collisions(self, p, r_contact):
        # Spatial hash / grid for self-collision candidate pruning
        cell_size = 2.0 * r_contact
        coords = np.floor(p / cell_size).astype(np.int64)

        # Simple spatial bucketing
        keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
        sorted_indices = np.argsort(keys)

        p_sorted = p[sorted_indices]
        N = len(p)

        # Check adjacent elements in sorted list
        window = 16
        for offset in range(1, window + 1):
            if offset >= N:
                break
            idx_a = sorted_indices[:-offset]
            idx_b = sorted_indices[offset:]

            # Avoid direct grid neighbors (cloth mesh neighbors)
            row_a, col_a = idx_a // self.nx, idx_a % self.nx
            row_b, col_b = idx_b // self.nx, idx_b % self.nx
            grid_dist = np.maximum(np.abs(row_a - row_b), np.abs(col_a - col_b))

            mask = grid_dist > 1
            if not np.any(mask):
                continue

            ia = idx_a[mask]
            ib = idx_b[mask]

            delta = p[ia] - p[ib]
            dist_sq = np.sum(delta * delta, axis=1)
            diam = 2.0 * r_contact
            collide = dist_sq < (diam * diam)
            if np.any(collide):
                c_ia = ia[collide]
                c_ib = ib[collide]
                c_delta = delta[collide]
                c_dist = np.sqrt(np.maximum(dist_sq[collide], 1e-12))

                c_dir = c_delta / c_dist[:, None]
                penetration = diam - c_dist

                w1 = self.inv_mass[c_ia]
                w2 = self.inv_mass[c_ib]
                w_sum = np.maximum(w1 + w2, 1e-12)

                corr = 0.5 * penetration / w_sum
                np.add.at(p, c_ia, c_dir * (corr * w1)[:, None])
                np.add.at(p, c_ib, -c_dir * (corr * w2)[:, None])

    def _step_substep(self, dt: float):
        # 1. External forces (gravity) and velocity damping
        damping = 0.999
        v_pred = self.vel * damping + self.gravity * dt
        p_pred = self.pos + v_pred * dt

        # 2. Constraint projections (PBD iterations)
        iters = 12
        r_contact = self.thickness * 0.5

        for _ in range(iters):
            # Distance constraints
            self._solve_distance_constraints(p_pred, self.struct_edges, self.struct_rest_lens, 0.98)
            self._solve_distance_constraints(p_pred, self.shear_edges, self.shear_rest_lens, 0.85)
            self._solve_distance_constraints(p_pred, self.bend_edges, self.bend_rest_lens, 0.40)

            # Ground collision: z >= ground_z + r_contact
            ground_penetration = (self.ground_z + r_contact) - p_pred[:, 2]
            ground_hit = ground_penetration > 0
            if np.any(ground_hit):
                p_pred[ground_hit, 2] = self.ground_z + r_contact

            # Funnel collision
            closest_dist, closest_p, closest_n = self._project_funnel(p_pred)

            # Check if point is inside or near funnel surface
            # Funnel normal points outwards from the funnel wall surface
            diff = p_pred - closest_p
            signed_dist = np.sum(diff * closest_n, axis=1)

            # Contact threshold
            funnel_hit = signed_dist < r_contact
            if np.any(funnel_hit):
                # Push along normal
                push_dist = r_contact - signed_dist[funnel_hit]
                p_pred[funnel_hit] += closest_n[funnel_hit] * push_dist[:, None]

            # Self-collision handling
            self._solve_self_collisions(p_pred, r_contact)

        # 3. Update velocity and position
        new_vel = (p_pred - self.pos) / dt

        # Friction at contacts
        # Re-check funnel contact for friction
        closest_dist, closest_p, closest_n = self._project_funnel(p_pred)
        diff = p_pred - closest_p
        signed_dist = np.sum(diff * closest_n, axis=1)
        in_contact = (signed_dist <= (r_contact * 1.5)) | (p_pred[:, 2] <= (self.ground_z + r_contact * 1.5))

        if np.any(in_contact):
            # Apply tangential friction
            friction_coeff = 0.2
            v_contact = new_vel[in_contact]
            n_contact = closest_n[in_contact]
            v_normal = np.sum(v_contact * n_contact, axis=1, keepdims=True) * n_contact
            v_tangent = v_contact - v_normal
            new_vel[in_contact] = v_normal + (1.0 - friction_coeff) * v_tangent

        self.vel = new_vel
        self.pos = p_pred

    def _compute_deformation_gradient(self):
        """Compute per-vertex 2D deformation gradient F = [dpos/dx, dpos/dy] projected onto local sheet."""
        N = len(self.pos)
        grid_pos = self.pos.reshape(self.ny, self.nx, 3)

        d_dx = np.zeros((self.ny, self.nx, 3))
        d_dy = np.zeros((self.ny, self.nx, 3))

        d_dx[:, 1:-1, :] = (grid_pos[:, 2:, :] - grid_pos[:, :-2, :]) / (2.0 * self.dx)
        d_dx[:, 0, :] = (grid_pos[:, 1, :] - grid_pos[:, 0, :]) / self.dx
        d_dx[:, -1, :] = (grid_pos[:, -1, :] - grid_pos[:, -2, :]) / self.dx

        d_dy[1:-1, :, :] = (grid_pos[2:, :, :] - grid_pos[:-2, :, :]) / (2.0 * self.dy)
        d_dy[0, :, :] = (grid_pos[1, :, :] - grid_pos[0, :, :]) / self.dy
        d_dy[-1, :, :] = (grid_pos[-1, :, :] - grid_pos[-2, :, :]) / self.dy

        d_dx_flat = d_dx.reshape(N, 3)
        d_dy_flat = d_dy.reshape(N, 3)

        # Construct 2x2 deformation gradient tensor
        # F = [[F11, F12], [F21, F22]]
        F11 = np.linalg.norm(d_dx_flat, axis=1)
        F22 = np.linalg.norm(d_dy_flat, axis=1)
        F12 = np.sum(d_dx_flat * d_dy_flat, axis=1) / np.maximum(F22, 1e-12)
        F21 = np.zeros_like(F11)

        F = np.zeros((N, 2, 2), dtype=np.float64)
        F[:, 0, 0] = F11
        F[:, 0, 1] = F12
        F[:, 1, 0] = F21
        F[:, 1, 1] = F22

        return F

    def _compute_contact_signed_distance(self):
        """Compute signed geometric gap at active and near-contact samples; negative means penetration."""
        r_contact = self.thickness * 0.5
        closest_dist, closest_p, closest_n = self._project_funnel(self.pos)
        diff = self.pos - closest_p
        funnel_gap = np.sum(diff * closest_n, axis=1) - r_contact

        ground_gap = (self.pos[:, 2] - self.ground_z) - r_contact

        return np.minimum(funnel_gap, ground_gap)

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        dt_sub = 0.002  # fine substep for stability and collision accuracy

        while self.current_time < target_time - 1e-9:
            dt_step = min(dt_sub, target_time - self.current_time)
            self._step_substep(dt_step)
            self.current_time += dt_step

        self.current_time = target_time

        N = len(self.pos)
        external_force = self.mass[:, None] * self.gravity[None, :]
        deformation_gradient = self._compute_deformation_gradient()
        contact_signed_dist = self._compute_contact_signed_distance()

        frame = StateFrame(
            float(self.current_time),
            np.array(self.pos, dtype=np.float64, copy=True),
            np.array(self.vel, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.zeros(N, dtype=np.int32),
            attributes={
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
                "boundary_vertex": np.array(self.boundary_mask, dtype=np.int32, copy=True),
                "deformation_gradient": deformation_gradient,
            },
            fields={
                "external_force": external_force,
                "contact_signed_distance": contact_signed_dist,
                "cumulative_mass_source": np.zeros(N, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.pos) if self.pos is not None else 0,
        }

