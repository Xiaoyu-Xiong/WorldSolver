"""Solver Agent-owned implementation for Stiff Sheet Drop on a Sphere."""
from __future__ import annotations
import json
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.current_time = 0.0

        cloth_cfg = self.config["cloth"]
        self.grid_shape = tuple(map(int, cloth_cfg["grid_shape"]))
        self.nx, self.ny = self.grid_shape
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.total_mass = float(cloth_cfg.get("total_mass", 0.55))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Obstacles
        self.sphere_radius = 0.62
        self.sphere_center = np.array([0.0, 0.0, 0.68], dtype=float)
        self.ground_z = 0.0  # Top of ground box (z=-0.05 + 0.05)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.positions = np.asarray(init_frame.position, dtype=float).copy()
        self.velocities = np.asarray(init_frame.velocity, dtype=float).copy()
        self.num_vertices = len(self.positions)
        self.masses = np.asarray(init_frame.mass, dtype=float).copy()
        self.inv_masses = np.where(self.masses > 0, 1.0 / self.masses, 0.0)

        # Initial reference positions in 2D material coords
        sx, sy = map(float, cloth_cfg["size"])
        x_ref = np.linspace(-sx * 0.5, sx * 0.5, self.nx)
        y_ref = np.linspace(-sy * 0.5, sy * 0.5, self.ny)
        xx_ref, yy_ref = np.meshgrid(x_ref, y_ref, indexing="xy")
        self.ref_pos_2d = np.column_stack((xx_ref.ravel(), yy_ref.ravel()))

        # Load topology
        try:
            with open("assets/cloth_topology.json", "r") as f:
                topo = json.load(f)
                self.faces = np.array(topo["faces"], dtype=np.int32)
        except Exception:
            # Fallback grid triangulation
            faces = []
            for j in range(self.ny - 1):
                for i in range(self.nx - 1):
                    v0 = j * self.nx + i
                    v1 = v0 + 1
                    v2 = (j + 1) * self.nx + i
                    v3 = v2 + 1
                    faces.append([v0, v1, v3])
                    faces.append([v0, v3, v2])
            self.faces = np.array(faces, dtype=np.int32)

        # Precompute triangle inverse material space matrices
        self.num_faces = len(self.faces)
        self.inv_Dm = np.zeros((self.num_faces, 2, 2))
        for f_idx, (i0, i1, i2) in enumerate(self.faces):
            u0, u1, u2 = self.ref_pos_2d[i0], self.ref_pos_2d[i1], self.ref_pos_2d[i2]
            Dm = np.column_stack((u1 - u0, u2 - u0))
            if np.abs(np.linalg.det(Dm)) > 1e-12:
                self.inv_Dm[f_idx] = np.linalg.inv(Dm)
            else:
                self.inv_Dm[f_idx] = np.eye(2)

        # Build vertex to faces mapping for deformation gradient interpolation
        self.vertex_faces = [[] for _ in range(self.num_vertices)]
        for f_idx, (i0, i1, i2) in enumerate(self.faces):
            self.vertex_faces[i0].append(f_idx)
            self.vertex_faces[i1].append(f_idx)
            self.vertex_faces[i2].append(f_idx)

        # Build distance / stretching springs
        edges_set = set()
        # Structural & shear edges
        for i0, i1, i2 in self.faces:
            for pair in [(i0, i1), (i1, i2), (i2, i0)]:
                e = tuple(sorted(pair))
                edges_set.add(e)

        # Bending edges (2-hop grid neighbors)
        bending_edges_set = set()
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i + 2 < self.nx:
                    bending_edges_set.add(tuple(sorted((idx, idx + 2))))
                if j + 2 < self.ny:
                    bending_edges_set.add(tuple(sorted((idx, (j + 2) * self.nx + i))))
                if i + 2 < self.nx and j + 2 < self.ny:
                    bending_edges_set.add(tuple(sorted((idx, (j + 2) * self.nx + i + 2))))
                if i - 2 >= 0 and j + 2 < self.ny:
                    bending_edges_set.add(tuple(sorted((idx, (j + 2) * self.nx + i - 2))))

        self.stretch_edges = np.array(list(edges_set), dtype=np.int32)
        p0 = self.positions[self.stretch_edges[:, 0]]
        p1 = self.positions[self.stretch_edges[:, 1]]
        self.stretch_rest_len = np.linalg.norm(p0 - p1, axis=1)

        self.bend_edges = np.array(list(bending_edges_set), dtype=np.int32)
        bp0 = self.positions[self.bend_edges[:, 0]]
        bp1 = self.positions[self.bend_edges[:, 1]]
        self.bend_rest_len = np.linalg.norm(bp0 - bp1, axis=1)

        # Simulation parameters
        self.substep_dt = 0.001  # 1 ms sub-stepping
        self.stretch_compliance = 1e-7  # High in-plane stiffness (stiff sheet)
        self.bend_compliance = 1e-4     # Moderate bending stiffness for sharp creases
        self.damping = 0.998
        self.pbd_iterations = 15

    def _substep(self, dt: float) -> None:
        # Semi-implicit Euler / XPBD integration
        # Gravity
        ext_acc = self.gravity
        self.velocities += ext_acc * dt
        self.velocities *= (self.damping ** (dt / 0.001))

        prev_pos = self.positions.copy()
        self.positions += self.velocities * dt

        effective_radius = self.sphere_radius + self.thickness * 0.5
        effective_ground = self.ground_z + self.thickness * 0.5

        # PBD / XPBD Constraint projection iterations
        alpha_s = self.stretch_compliance / (dt * dt)
        alpha_b = self.bend_compliance / (dt * dt)

        for _ in range(self.pbd_iterations):
            # 1. Stretch constraints
            idx0 = self.stretch_edges[:, 0]
            idx1 = self.stretch_edges[:, 1]
            p0 = self.positions[idx0]
            p1 = self.positions[idx1]
            diff = p0 - p1
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            dir_vec = diff / dist_safe
            C = dist - self.stretch_rest_len[:, None]

            w0 = self.inv_masses[idx0, None]
            w1 = self.inv_masses[idx1, None]
            w_sum = w0 + w1

            delta_lambda = -C / (w_sum + alpha_s)
            corr0 = w0 * delta_lambda * dir_vec
            corr1 = -w1 * delta_lambda * dir_vec

            np.add.at(self.positions, idx0, corr0)
            np.add.at(self.positions, idx1, corr1)

            # 2. Bend constraints
            b_idx0 = self.bend_edges[:, 0]
            b_idx1 = self.bend_edges[:, 1]
            bp0 = self.positions[b_idx0]
            bp1 = self.positions[b_idx1]
            b_diff = bp0 - bp1
            b_dist = np.linalg.norm(b_diff, axis=1, keepdims=True)
            b_dist_safe = np.maximum(b_dist, 1e-8)
            b_dir_vec = b_diff / b_dist_safe
            b_C = b_dist - self.bend_rest_len[:, None]

            bw0 = self.inv_masses[b_idx0, None]
            bw1 = self.inv_masses[b_idx1, None]
            bw_sum = bw0 + bw1

            b_delta_lambda = -b_C / (bw_sum + alpha_b)
            b_corr0 = bw0 * b_delta_lambda * b_dir_vec
            b_corr1 = -bw1 * b_delta_lambda * b_dir_vec

            np.add.at(self.positions, b_idx0, b_corr0)
            np.add.at(self.positions, b_idx1, b_corr1)

            # 3. Collision constraints: Sphere
            disp_sphere = self.positions - self.sphere_center
            dist_sphere = np.linalg.norm(disp_sphere, axis=1, keepdims=True)
            dist_sphere_safe = np.maximum(dist_sphere, 1e-8)
            penetration_mask = (dist_sphere < effective_radius).ravel()
            if np.any(penetration_mask):
                normal = disp_sphere[penetration_mask] / dist_sphere_safe[penetration_mask]
                self.positions[penetration_mask] = self.sphere_center + normal * effective_radius

            # 4. Collision constraints: Ground
            ground_mask = self.positions[:, 2] < effective_ground
            if np.any(ground_mask):
                self.positions[ground_mask, 2] = effective_ground

        # Update velocities
        self.velocities = (self.positions - prev_pos) / dt

        # Collision velocity post-processing (friction & non-penetration)
        disp_sphere = self.positions - self.sphere_center
        dist_sphere = np.linalg.norm(disp_sphere, axis=1, keepdims=True)
        dist_sphere_safe = np.maximum(dist_sphere, 1e-8)
        contact_sphere = (dist_sphere <= effective_radius + 1e-4).ravel()
        if np.any(contact_sphere):
            norm_s = disp_sphere[contact_sphere] / dist_sphere_safe[contact_sphere]
            vn = np.sum(self.velocities[contact_sphere] * norm_s, axis=1, keepdims=True)
            vn_clamp = np.minimum(vn, 0.0)
            self.velocities[contact_sphere] -= vn_clamp * norm_s
            # Coulomb friction
            vt = self.velocities[contact_sphere] - np.sum(self.velocities[contact_sphere] * norm_s, axis=1, keepdims=True) * norm_s
            self.velocities[contact_sphere] -= 0.6 * vt

        contact_ground = self.positions[:, 2] <= effective_ground + 1e-4
        if np.any(contact_ground):
            self.velocities[contact_ground, 2] = np.maximum(self.velocities[contact_ground, 2], 0.0)
            self.velocities[contact_ground, 0:2] *= 0.5

    def _compute_deformation_gradients(self) -> np.ndarray:
        # Per-triangle deformation gradient F = Ds * inv(Dm) (shape [M, 3, 2]) or [N, 2, 2]
        # Scene initial state expects shape [N, 2, 2] or element-wise
        # Let's compute per-vertex deformation gradient (2x2) in the sheet tangent space
        i0 = self.faces[:, 0]
        i1 = self.faces[:, 1]
        i2 = self.faces[:, 2]
        p0 = self.positions[i0]
        p1 = self.positions[i1]
        p2 = self.positions[i2]

        # 3D edge vectors
        e1 = p1 - p0  # [M, 3]
        e2 = p2 - p0  # [M, 3]

        # Local 2D coordinates on triangle plane
        n = np.cross(e1, e2)
        n_len = np.linalg.norm(n, axis=1, keepdims=True)
        n_len_safe = np.maximum(n_len, 1e-8)
        n = n / n_len_safe

        # Tangent basis u along e1, v = n x u
        e1_len = np.linalg.norm(e1, axis=1, keepdims=True)
        e1_len_safe = np.maximum(e1_len, 1e-8)
        t_u = e1 / e1_len_safe
        t_v = np.cross(n, t_u)

        # Deformed coords in 2D plane:
        # x0 = (0, 0), x1 = (e1_len, 0), x2 = (e2.tu, e2.tv)
        Ds_00 = e1_len.ravel()
        Ds_01 = np.sum(e2 * t_u, axis=1)
        Ds_10 = np.zeros_like(Ds_00)
        Ds_11 = np.sum(e2 * t_v, axis=1)

        # Ds = [[Ds_00, Ds_01], [Ds_10, Ds_11]]
        # F = Ds * inv(Dm)
        invDm = self.inv_Dm  # [M, 2, 2]
        F_faces = np.zeros((self.num_faces, 2, 2))
        F_faces[:, 0, 0] = Ds_00 * invDm[:, 0, 0] + Ds_01 * invDm[:, 1, 0]
        F_faces[:, 0, 1] = Ds_00 * invDm[:, 0, 1] + Ds_01 * invDm[:, 1, 1]
        F_faces[:, 1, 0] = Ds_10 * invDm[:, 0, 0] + Ds_11 * invDm[:, 1, 0]
        F_faces[:, 1, 1] = Ds_10 * invDm[:, 0, 1] + Ds_11 * invDm[:, 1, 1]

        # Average to vertices
        F_vertices = np.zeros((self.num_vertices, 2, 2))
        vertex_counts = np.zeros(self.num_vertices)
        for f_idx, (v0, v1, v2) in enumerate(self.faces):
            F_vertices[v0] += F_faces[f_idx]
            F_vertices[v1] += F_faces[f_idx]
            F_vertices[v2] += F_faces[f_idx]
            vertex_counts[v0] += 1
            vertex_counts[v1] += 1
            vertex_counts[v2] += 1

        vertex_counts = np.maximum(vertex_counts[:, None, None], 1.0)
        F_vertices /= vertex_counts
        return F_vertices

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation forward to reach requested time
        while self.current_time < time - 1e-9:
            dt = min(self.substep_dt, time - self.current_time)
            self._substep(dt)
            self.current_time += dt

        self.current_time = time

        # Compute signed distance to nearest obstacle (sphere or ground)
        disp_sphere = self.positions - self.sphere_center
        dist_sphere = np.linalg.norm(disp_sphere, axis=1) - self.sphere_radius
        dist_ground = self.positions[:, 2] - self.ground_z
        contact_signed_dist = np.minimum(dist_sphere, dist_ground)

        # External force = mass * gravity
        ext_force = self.masses[:, None] * self.gravity[None, :]

        # Deformation gradient [N, 2, 2]
        F = self._compute_deformation_gradients()

        # Build StateFrame
        frame = StateFrame(
            time=float(time),
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.masses.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int32),
            attributes={
                "mass_quantity": self.masses.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": contact_signed_dist,
                "cumulative_mass_source": np.zeros(self.num_vertices, dtype=float),
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_vertices,
            "simulated_time": self.current_time,
        }

