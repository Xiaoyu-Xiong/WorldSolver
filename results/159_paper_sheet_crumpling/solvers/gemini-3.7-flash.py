"""Paper Sheet Crumpling Physics Solver.

Implements a thin-sheet crumpling simulation with:
- Position-based / projective dynamics with stretching, bending (hinges), and plastic creasing
- Moving kinematic boundaries (compressing side walls)
- Ground collision and sheet self-contact
- Plastic crease accumulation for realistic paper crumpling behavior
"""
from __future__ import annotations
import json
import os
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.pos = init_frame.position.copy().astype(np.float64)
        self.vel = np.zeros_like(self.pos, dtype=np.float64)
        self.mass = init_frame.mass.copy().astype(np.float64)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)
        self.num_nodes = len(self.pos)
        self.current_time = 0.0

        cloth_cfg = self.config.get("cloth", {})
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Rest coordinates
        self.rest_pos = self.pos.copy()

        # Load topology
        mesh_topo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "cloth_topology.json")
        if os.path.exists(mesh_topo_path):
            with open(mesh_topo_path, "r") as f:
                topo = json.load(f)
                self.faces = np.array(topo.get("faces", []), dtype=np.int32)
        else:
            # Generate regular grid faces (17x17)
            nx, ny = map(int, cloth_cfg.get("grid_shape", [17, 17]))
            faces = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    v0 = j * nx + i
                    v1 = j * nx + i + 1
                    v2 = (j + 1) * nx + i
                    v3 = (j + 1) * nx + i + 1
                    faces.append([v0, v1, v3])
                    faces.append([v0, v3, v2])
            self.faces = np.array(faces, dtype=np.int32)

        # Build edge springs
        edges_set = set()
        for f in self.faces:
            edges_set.add(tuple(sorted([f[0], f[1]])))
            edges_set.add(tuple(sorted([f[1], f[2]])))
            edges_set.add(tuple(sorted([f[2], f[0]])))

        self.edges = np.array(list(edges_set), dtype=np.int32)
        p0 = self.rest_pos[self.edges[:, 0]]
        p1 = self.rest_pos[self.edges[:, 1]]
        self.edge_rest_lens = np.linalg.norm(p0 - p1, axis=1)

        # Build bending hinges (dihedrals between adjacent triangles)
        # Map directed edge -> triangle
        edge_to_face = {}
        for f_idx, f in enumerate(self.faces):
            for i in range(3):
                e = (f[i], f[(i + 1) % 3])
                e_sorted = tuple(sorted(e))
                if e_sorted not in edge_to_face:
                    edge_to_face[e_sorted] = []
                edge_to_face[e_sorted].append((f_idx, f[(i + 2) % 3]))

        hinges = []
        rest_dihedrals = []
        for e_sorted, adj in edge_to_face.items():
            if len(adj) == 2:
                # e_sorted = (v0, v1), wing vertices = v2, v3
                v0, v1 = e_sorted
                v2 = adj[0][1]
                v3 = adj[1][1]
                hinges.append([v0, v1, v2, v3])
                # Compute rest angle
                p_v0 = self.rest_pos[v0]
                p_v1 = self.rest_pos[v1]
                p_v2 = self.rest_pos[v2]
                p_v3 = self.rest_pos[v3]
                e_vec = p_v1 - p_v0
                e_len = np.linalg.norm(e_vec) + 1e-12
                e_hat = e_vec / e_len
                n1 = np.cross(p_v1 - p_v0, p_v2 - p_v0)
                n2 = np.cross(p_v3 - p_v0, p_v1 - p_v0)
                n1_len = np.linalg.norm(n1) + 1e-12
                n2_len = np.linalg.norm(n2) + 1e-12
                n1 = n1 / n1_len
                n2 = n2 / n2_len
                cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
                sin_theta = np.dot(np.cross(n1, n2), e_hat)
                angle = np.arctan2(sin_theta, cos_theta)
                rest_dihedrals.append(angle)

        self.hinges = np.array(hinges, dtype=np.int32) if len(hinges) > 0 else np.zeros((0, 4), dtype=np.int32)
        self.plastic_crease_angles = np.array(rest_dihedrals, dtype=np.float64) if len(rest_dihedrals) > 0 else np.zeros((0,), dtype=np.float64)

        # Precompute material 2D coordinates for deformation gradient computation
        # For each triangle, define local 2D frame (D_m)
        self.Dm_inv = []
        for f in self.faces:
            x0 = self.rest_pos[f[0]]
            x1 = self.rest_pos[f[1]]
            x2 = self.rest_pos[f[2]]
            e1 = x1 - x0
            e2 = x2 - x0
            # Since paper is initially in XY plane, take (x,y)
            Dm = np.column_stack([e1[:2], e2[:2]])
            try:
                self.Dm_inv.append(np.linalg.inv(Dm))
            except np.linalg.LinAlgError:
                self.Dm_inv.append(np.eye(2))
        self.Dm_inv = np.array(self.Dm_inv, dtype=np.float64)

        # Node-to-triangle mapping for vertex deformation gradient
        self.node_faces = [[] for _ in range(self.num_nodes)]
        for f_idx, f in enumerate(self.faces):
            for v in f:
                self.node_faces[v].append(f_idx)

        # Simulation parameters
        self.sub_dt = 0.001  # 1 ms sub-stepping
        self.stretch_stiffness = 0.95
        self.bending_stiffness = 0.4
        self.yield_angle = 0.35  # rad, plastic threshold for creasing
        self.plastic_hardening = 0.8
        self.damping = 0.9995

        # Wall kinematics parameters
        # Total duration = 7.0s
        # Walls compress from initial x = +/- 1.55 to final x = +/- 0.45
        self.initial_wall_x = 1.55
        self.final_wall_x = 0.42
        self.wall_compress_duration = 5.5

        # Initial small perturbation to break symmetry for natural crumpling
        np.random.seed(self.seed)
        noise = (np.random.rand(self.num_nodes) - 0.5) * 0.008
        self.pos[:, 2] += noise

        # State cache
        self.last_contact_dists = np.array([self.thickness], dtype=np.float64)
        self.last_ext_forces = np.zeros_like(self.pos, dtype=np.float64)

    def _get_wall_positions(self, t: float) -> tuple[float, float, float]:
        """Returns (wall_left_x, wall_right_x, wall_vx)."""
        s = np.clip(t / self.wall_compress_duration, 0.0, 1.0)
        # Smooth cosine-based compression profile
        smooth_s = 0.5 * (1.0 - np.cos(np.pi * s))
        curr_x = self.initial_wall_x - (self.initial_wall_x - self.final_wall_x) * smooth_s
        wall_left_x = -curr_x
        wall_right_x = curr_x
        if t < self.wall_compress_duration:
            ds = (np.pi / (2.0 * self.wall_compress_duration)) * np.sin(np.pi * s)
            vx = - (self.initial_wall_x - self.final_wall_x) * ds
        else:
            vx = 0.0
        return wall_left_x, wall_right_x, vx

    def _step(self, dt: float, current_t: float):
        # 1. External forces (gravity) & velocity prediction
        ext_force = self.mass[:, None] * self.gravity[None, :]
        self.last_ext_forces = ext_force.copy()

        self.vel += (ext_force * self.inv_mass[:, None]) * dt
        self.vel *= self.damping
        new_pos = self.pos + self.vel * dt

        wall_left_x, wall_right_x, wall_vx = self._get_wall_positions(current_t + dt)
        ground_z = 0.0
        contact_dists = []

        # 2. Position-Based Dynamics iterations
        pbd_iters = 12
        for it in range(pbd_iters):
            # A. Distance (stretching) constraints
            p0 = new_pos[self.edges[:, 0]]
            p1 = new_pos[self.edges[:, 1]]
            diff = p0 - p1
            dist = np.linalg.norm(diff, axis=1) + 1e-12
            err = (dist - self.edge_rest_lens) / dist
            corr = 0.5 * self.stretch_stiffness * diff * err[:, None]

            w0 = self.inv_mass[self.edges[:, 0]]
            w1 = self.inv_mass[self.edges[:, 1]]
            w_sum = w0 + w1 + 1e-12
            c0 = corr * (w0 / w_sum)[:, None]
            c1 = -corr * (w1 / w_sum)[:, None]

            np.add.at(new_pos, self.edges[:, 0], -c0)
            np.add.at(new_pos, self.edges[:, 1], -c1)

            # B. Bending / Creasing hinge constraints
            if len(self.hinges) > 0:
                h_v0 = self.hinges[:, 0]
                h_v1 = self.hinges[:, 1]
                h_v2 = self.hinges[:, 2]
                h_v3 = self.hinges[:, 3]

                p_v0 = new_pos[h_v0]
                p_v1 = new_pos[h_v1]
                p_v2 = new_pos[h_v2]
                p_v3 = new_pos[h_v3]

                e_vec = p_v1 - p_v0
                e_len = np.linalg.norm(e_vec, axis=1, keepdims=True) + 1e-12
                e_hat = e_vec / e_len

                n1 = np.cross(p_v1 - p_v0, p_v2 - p_v0)
                n2 = np.cross(p_v3 - p_v0, p_v1 - p_v0)
                n1_len = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12
                n2_len = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
                n1_hat = n1 / n1_len
                n2_hat = n2 / n2_len

                cos_theta = np.clip(np.sum(n1_hat * n2_hat, axis=1), -1.0, 1.0)
                sin_theta = np.sum(np.cross(n1_hat, n2_hat) * e_hat, axis=1)
                curr_theta = np.arctan2(sin_theta, cos_theta)

                # Plastic creasing update: if deformation exceeds yield angle, update plastic rest angle
                d_theta = curr_theta - self.plastic_crease_angles
                plastic_mask = np.abs(d_theta) > self.yield_angle
                if np.any(plastic_mask):
                    excess = d_theta[plastic_mask] - np.sign(d_theta[plastic_mask]) * self.yield_angle
                    self.plastic_crease_angles[plastic_mask] += excess * (1.0 - self.plastic_hardening)

                # Restore towards plastic_crease_angles
                d_angle = curr_theta - self.plastic_crease_angles
                bend_corr = 0.5 * self.bending_stiffness * d_angle[:, None]

                # Approximate normal impulse on wing vertices
                np.add.at(new_pos, h_v2, -bend_corr * n1_hat * 0.25)
                np.add.at(new_pos, h_v3, -bend_corr * n2_hat * 0.25)

            # C. Kinematic boundary collision (Moving Left / Right walls)
            # Wall left is at x <= wall_left_x (penetration if new_pos[:, 0] < wall_left_x + thickness*0.5)
            left_bound = wall_left_x + self.thickness * 0.5
            penetrate_left = new_pos[:, 0] < left_bound
            if np.any(penetrate_left):
                new_pos[penetrate_left, 0] = left_bound

            # Wall right is at x >= wall_right_x
            right_bound = wall_right_x - self.thickness * 0.5
            penetrate_right = new_pos[:, 0] > right_bound
            if np.any(penetrate_right):
                new_pos[penetrate_right, 0] = right_bound

            # Ground collision (z >= ground_z + thickness*0.5)
            bot_bound = ground_z + self.thickness * 0.5
            penetrate_bot = new_pos[:, 2] < bot_bound
            if np.any(penetrate_bot):
                new_pos[penetrate_bot, 2] = bot_bound

            # Side boundary confines in y-direction to keep sheet within container
            max_y = 1.3
            new_pos[:, 1] = np.clip(new_pos[:, 1], -max_y, max_y)

        # Sheet self-contact repulsion (spatial hash / pairwise for small mesh)
        # N = 289 is small enough for vector-distance filtering
        p_coords = new_pos
        diff_all = p_coords[:, None, :] - p_coords[None, :, :]  # [N, N, 3]
        dists_sq = np.sum(diff_all ** 2, axis=-1)  # [N, N]
        np.fill_diagonal(dists_sq, 1e9)  # ignore self

        # Ignore direct topological neighbors from self-collision
        for e in self.edges:
            dists_sq[e[0], e[1]] = 1e9
            dists_sq[e[1], e[0]] = 1e9

        min_sep = self.thickness * 1.5
        min_sep_sq = min_sep ** 2
        close_i, close_j = np.where(dists_sq < min_sep_sq)

        # Only take unique pairs i < j
        pair_mask = close_i < close_j
        ci = close_i[pair_mask]
        cj = close_j[pair_mask]

        if len(ci) > 0:
            d_pair = np.sqrt(dists_sq[ci, cj])
            overlap = min_sep - d_pair
            n_pair = diff_all[ci, cj] / (d_pair[:, None] + 1e-12)
            shift = 0.4 * overlap[:, None] * n_pair
            np.add.at(new_pos, ci, shift)
            np.add.at(new_pos, cj, -shift)
            contact_dists.extend((d_pair - self.thickness).tolist())

        # Collect contact distances for reporting
        left_gaps = new_pos[:, 0] - left_bound
        right_gaps = right_bound - new_pos[:, 0]
        bot_gaps = new_pos[:, 2] - bot_bound
        min_wall_gaps = np.minimum(np.minimum(left_gaps, right_gaps), bot_gaps)
        active_contacts = min_wall_gaps[min_wall_gaps < self.thickness * 3.0]
        if len(active_contacts) > 0:
            contact_dists.extend(active_contacts.tolist())

        if len(contact_dists) == 0:
            self.last_contact_dists = np.array([self.thickness], dtype=np.float64)
        else:
            self.last_contact_dists = np.array(contact_dists, dtype=np.float64)

        # 3. Update velocity and position
        self.vel = (new_pos - self.pos) / dt
        self.pos = new_pos

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute per-vertex 2D deformation gradient F."""
        num_faces = len(self.faces)
        face_F = np.zeros((num_faces, 2, 2), dtype=np.float64)
        for i, f in enumerate(self.faces):
            x0 = self.pos[f[0]]
            x1 = self.pos[f[1]]
            x2 = self.pos[f[2]]
            e1 = x1 - x0
            e2 = x2 - x0
            # Current 2D tangent space components (in plane of deformed triangle)
            l1 = np.linalg.norm(e1) + 1e-12
            t1 = e1 / l1
            n = np.cross(e1, e2)
            n_len = np.linalg.norm(n) + 1e-12
            t2 = np.cross(n / n_len, t1)
            # Coordinates of edge vectors in local 2D tangent frame
            Ds = np.column_stack([
                [l1, 0.0],
                [np.dot(e2, t1), np.dot(e2, t2)]
            ])
            face_F[i] = Ds @ self.Dm_inv[i]

        # Average deformation gradient onto vertices
        node_F = np.zeros((self.num_nodes, 2, 2), dtype=np.float64)
        for v in range(self.num_nodes):
            adj = self.node_faces[v]
            if len(adj) > 0:
                node_F[v] = np.mean(face_F[adj], axis=0)
            else:
                node_F[v] = np.eye(2)
        return node_F

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation forward to reach requested time
        while self.current_time + 1e-7 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt, self.current_time)
            self.current_time += dt

        self.current_time = time

        # Compute deformation gradient
        def_grad = self._compute_deformation_gradients()

        # Build StateFrame
        frame = StateFrame(
            float(self.current_time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_nodes, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": def_grad
            },
            fields={
                "contact_signed_distance": self.last_contact_dists.copy(),
                "cumulative_mass_source": np.zeros(self.num_nodes, dtype=np.float64),
                "external_force": self.last_ext_forces.copy()
            },
            scalars={
                "contact_length_scale": float(self.thickness)
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_faces": len(self.faces),
            "num_hinges": len(self.hinges),
            "current_time": self.current_time
        }

