"""Physical cloth simulation solver for cloth wrinkle wake from a moving sphere.

Recreates the classic Narain et al. 2012 Figure 2 scenario:
A sphere moves horizontally under / against a draped cloth sheet, lifting it,
creating a contact bulge and trailing wrinkle wake with anisotropic folds.
"""
from __future__ import annotations

import json
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load topology
        try:
            with open("assets/cloth_topology.json", "r") as f:
                top_data = json.load(f)
                self.faces = np.array(top_data["faces"], dtype=np.int32)
        except Exception:
            self.faces = None

        # Grid configuration
        cloth_cfg = self.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.sx, self.sy = map(float, cloth_cfg["size"])
        self.cloth_center = np.asarray(cloth_cfg["center"], dtype=float)
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.total_mass = float(cloth_cfg.get("total_mass", 0.32))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)
        self.boundary_vertex = np.array(init_frame.attributes.get("boundary_vertex", np.zeros(len(self.pos), dtype=np.int32)), copy=True)
        self.num_particles = len(self.pos)

        # Fix pinned boundary: 4 corners or top/bottom edges
        # In Narain et al. 2012, cloth is held at corners or edges
        # Let's pin the 4 corners so cloth drapes naturally and stays suspended
        self.pinned = np.zeros(self.num_particles, dtype=bool)
        # Pin four corners: (0, 0), (nx-1, 0), (0, ny-1), (nx-1, ny-1)
        # grid indexing: index = j * nx + i (since xx, yy created with meshgrid(x, y, indexing="xy"))
        corners = [0, self.nx - 1, (self.ny - 1) * self.nx, self.ny * self.nx - 1]
        for c in corners:
            if c < self.num_particles:
                self.pinned[c] = True

        # Pinned particles have zero inverse mass in solver
        self.inv_mass_eff = self.inv_mass.copy()
        self.inv_mass_eff[self.pinned] = 0.0

        # Reference rest positions in 2D material coords and 3D
        self.rest_pos_2d = np.zeros((self.num_particles, 2), dtype=np.float64)
        x = np.linspace(-self.sx * 0.5, self.sx * 0.5, self.nx)
        y = np.linspace(-self.sy * 0.5, self.sy * 0.5, self.ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.rest_pos_2d[:, 0] = xx.ravel()
        self.rest_pos_2d[:, 1] = yy.ravel()
        self.rest_pos_3d = self.pos.copy()

        # Build spring network (structural, shear, bending)
        spring_list = []
        spring_lens = []
        spring_types = [] # 1: structural, 2: shear, 3: bend

        def add_spring(i1, i2, stype):
            if i1 < self.num_particles and i2 < self.num_particles:
                p1 = self.rest_pos_3d[i1]
                p2 = self.rest_pos_3d[i2]
                l0 = float(np.linalg.norm(p1 - p2))
                spring_list.append((i1, i2))
                spring_lens.append(l0)
                spring_types.append(stype)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                # Structural (horizontal & vertical)
                if i + 1 < self.nx:
                    add_spring(idx, j * self.nx + (i + 1), 1)
                if j + 1 < self.ny:
                    add_spring(idx, (j + 1) * self.nx + i, 1)
                # Shear (diagonals)
                if i + 1 < self.nx and j + 1 < self.ny:
                    add_spring(idx, (j + 1) * self.nx + (i + 1), 2)
                if i - 1 >= 0 and j + 1 < self.ny:
                    add_spring(idx, (j + 1) * self.nx + (i - 1), 2)
                # Bending (2-hop)
                if i + 2 < self.nx:
                    add_spring(idx, j * self.nx + (i + 2), 3)
                if j + 2 < self.ny:
                    add_spring(idx, (j + 2) * self.nx + i, 3)

        self.spring_indices = np.array(spring_list, dtype=np.int32)
        self.spring_rest_lens = np.array(spring_lens, dtype=np.float64)
        self.spring_types = np.array(spring_types, dtype=np.int32)

        # Build face topology if not loaded
        if self.faces is None:
            faces_list = []
            for j in range(self.ny - 1):
                for i in range(self.nx - 1):
                    p0 = j * self.nx + i
                    p1 = j * self.nx + (i + 1)
                    p2 = (j + 1) * self.nx + i
                    p3 = (j + 1) * self.nx + (i + 1)
                    faces_list.append([p0, p1, p3])
                    faces_list.append([p0, p3, p2])
            self.faces = np.array(faces_list, dtype=np.int32)

        # Compute per-face reference triangle inversed Dm matrices for deformation gradient
        # Dm = [X1 - X0, X2 - X0] in 2D
        # F = Ds * Dm^{-1}
        self.face_inv_Dm = []
        for face in self.faces:
            i0, i1, i2 = face
            X0 = self.rest_pos_2d[i0]
            X1 = self.rest_pos_2d[i1]
            X2 = self.rest_pos_2d[i2]
            Dm = np.column_stack([X1 - X0, X2 - X0])
            try:
                inv_Dm = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                inv_Dm = np.eye(2)
            self.face_inv_Dm.append(inv_Dm)
        self.face_inv_Dm = np.array(self.face_inv_Dm, dtype=np.float64)

        # Map each vertex to incident faces for vertex deformation gradient
        self.vertex_faces = [[] for _ in range(self.num_particles)]
        for f_idx, face in enumerate(self.faces):
            for v in face:
                self.vertex_faces[v].append(f_idx)

        # Sphere kinematics parameters
        # In scene_geometry: radius = 0.62, initial translation = [0.0, 0.0, 0.68]
        # Cloth center is [0.0, 0.0, 1.22]. Sphere center z = 0.68 -> top of sphere is 0.68 + 0.62 = 1.30
        # which is 0.08m above cloth rest plane (1.22), so sphere pushes cloth up into a bulge!
        # The sphere moves across the cloth to generate a wake.
        # Path: starts around x = -1.2, moves to x = +1.2 over duration (7.0s)
        self.sphere_radius = 0.62
        self.sphere_start = np.array([-1.2, 0.0, 0.68], dtype=np.float64)
        self.sphere_end = np.array([1.2, 0.0, 0.68], dtype=np.float64)
        self.total_duration = float(self.config.get("duration", 7.0))

        # Ground collision
        # ground box translation: [0, 0, -0.05], half_extents: [2.4, 2.0, 0.05] -> top at z = 0.0
        self.ground_z = 0.0

        # Physical simulation constants
        self.current_time = 0.0
        self.sim_substeps = 20
        self.stiffness_struct = 1.0
        self.stiffness_shear = 0.8
        self.stiffness_bend = 0.05
        self.damping = 0.996

    def _get_sphere_pos(self, t: float) -> np.ndarray:
        # Smooth movement or linear movement across the sheet
        alpha = np.clip(t / max(self.total_duration, 1e-5), 0.0, 1.0)
        # Smoothstep
        s = alpha * alpha * (3.0 - 2.0 * alpha)
        return self.sphere_start + s * (self.sphere_end - self.sphere_start)

    def _step_substep(self, dt: float, current_t: float) -> None:
        # Gravity and external forces
        ext_acc = self.gravity.copy()

        # PBD / Verlet scheme
        # 1. Apply external forces & predict positions
        p_prev = self.pos.copy()

        # Add gravity to velocity for unpinned particles
        unpinned = ~self.pinned
        self.vel[unpinned] += ext_acc * dt
        self.vel *= self.damping

        p = self.pos + self.vel * dt
        # Keep pinned fixed
        p[self.pinned] = self.pos[self.pinned]

        # Sphere position at intermediate time
        sph_pos = self._get_sphere_pos(current_t + dt)
        sph_r = self.sphere_radius + self.thickness * 0.5

        # 2. Constraint solver iterations
        num_solver_iters = 10

        # Pre-extract arrays
        i1 = self.spring_indices[:, 0]
        i2 = self.spring_indices[:, 1]
        l0 = self.spring_rest_lens
        w1 = self.inv_mass_eff[i1]
        w2 = self.inv_mass_eff[i2]
        w_sum = w1 + w2
        valid_springs = w_sum > 1e-12

        # Compliance / stiffness per spring type
        stiffness_arr = np.where(self.spring_types == 1, self.stiffness_struct,
                        np.where(self.spring_types == 2, self.stiffness_shear, self.stiffness_bend))

        for _ in range(num_solver_iters):
            # Distance constraints (springs)
            delta = p[i1] - p[i2]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-9)
            dir_vec = delta / dist_safe

            C = dist.ravel() - l0
            corr_mag = (C / np.maximum(w_sum, 1e-12)) * stiffness_arr
            corr = dir_vec * corr_mag[:, None] * valid_springs[:, None]

            # Apply corrections accumulated
            np.add.at(p, i1, -corr * w1[:, None])
            np.add.at(p, i2, corr * w2[:, None])

            # Re-enforce pinned
            p[self.pinned] = self.pos[self.pinned]

            # Sphere contact constraint
            d_sph = p - sph_pos
            dist_sq = np.sum(d_sph**2, axis=1)
            penetration_mask = (dist_sq < sph_r**2) & (~self.pinned)
            if np.any(penetration_mask):
                d_norm = np.sqrt(np.maximum(dist_sq[penetration_mask], 1e-12))
                push_dir = d_sph[penetration_mask] / d_norm[:, None]
                p[penetration_mask] = sph_pos + push_dir * sph_r

            # Ground contact constraint
            ground_mask = (p[:, 2] < self.ground_z + self.thickness * 0.5) & (~self.pinned)
            p[ground_mask, 2] = self.ground_z + self.thickness * 0.5

        # 3. Update velocity and positions
        self.vel = (p - p_prev) / dt
        self.vel[self.pinned] = 0.0
        self.pos = p

    def advance_to(self, time: float) -> StateFrame:
        # Step physics up to requested time
        dt_target = time - self.current_time
        if dt_target > 1e-7:
            # Subdivide into smaller sub-steps
            substep_dt = dt_target / self.sim_substeps
            for step_i in range(self.sim_substeps):
                t_sub = self.current_time + step_i * substep_dt
                self._step_substep(substep_dt, t_sub)
            self.current_time = time
        elif dt_target < -1e-7:
            # Re-initialize if called out of order
            pass

        # Compute output fields
        # 1. External force: mass * gravity (for unpinned), 0 for pinned
        external_force = np.zeros_like(self.pos)
        external_force[~self.pinned] = self.mass[~self.pinned, None] * self.gravity[None, :]

        # 2. Signed contact distance to sphere (and ground)
        sph_pos = self._get_sphere_pos(self.current_time)
        sph_r = self.sphere_radius
        d_to_sph_center = np.linalg.norm(self.pos - sph_pos, axis=1)
        # Signed distance to sphere surface: >0 outside, <0 inside penetration
        contact_dist_sphere = d_to_sph_center - sph_r
        # Signed distance to ground surface
        contact_dist_ground = self.pos[:, 2] - self.ground_z
        contact_signed_distance = np.minimum(contact_dist_sphere, contact_dist_ground)

        # 3. Deformation gradient: per-element or per-vertex [N, 2, 2]
        # In scene.py baseline, deformation_gradient shape is [N, 2, 2]
        # Let's compute deformation gradient F in 2D material coords for each vertex (averaged from incident faces)
        # For a triangle with material coords X0, X1, X2 and spatial coords x0, x1, x2:
        # Ds = [x1 - x0, x2 - x0] (projected onto element tangent plane or 2D)
        # Dm = [X1 - X0, X2 - X0]
        # F = Ds * Dm^{-1}
        # In 2D material coords, F is 2x2.
        F_faces = np.zeros((len(self.faces), 2, 2), dtype=np.float64)
        for f_idx, face in enumerate(self.faces):
            i0, i1, i2 = face
            x0 = self.pos[i0]
            x1 = self.pos[i1]
            x2 = self.pos[i2]

            # Form local 2D basis on deformed triangle
            e1 = x1 - x0
            l_e1 = np.linalg.norm(e1)
            if l_e1 > 1e-9:
                u = e1 / l_e1
            else:
                u = np.array([1.0, 0.0, 0.0])

            e2 = x2 - x0
            normal = np.cross(u, e2)
            l_n = np.linalg.norm(normal)
            if l_n > 1e-9:
                v = np.cross(normal / l_n, u)
            else:
                v = np.array([0.0, 1.0, 0.0])

            Ds_2d = np.array([
                [np.dot(e1, u), np.dot(e2, u)],
                [np.dot(e1, v), np.dot(e2, v)]
            ], dtype=np.float64)

            inv_Dm = self.face_inv_Dm[f_idx]
            F_faces[f_idx] = Ds_2d @ inv_Dm

        # Average face F to vertices
        F_vertices = np.zeros((self.num_particles, 2, 2), dtype=np.float64)
        for v in range(self.num_particles):
            f_list = self.vertex_faces[v]
            if len(f_list) > 0:
                F_vertices[v] = np.mean(F_faces[f_list], axis=0)
            else:
                F_vertices[v] = np.eye(2)

        # 4. Cumulative mass source: [K] or [N], scene.py uses np.zeros(len(position))
        cum_mass_source = np.zeros(self.num_particles, dtype=np.float64)

        # Construct StateFrame exactly matching scene.py and required fields
        frame = StateFrame(
            float(self.current_time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": F_vertices,
            },
            fields={
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cum_mass_source,
                "external_force": external_force,
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
            "num_particles": self.num_particles,
            "sphere_radius": self.sphere_radius,
        }

