"""BenchmarkSolver implementation for 033_reef_knot_ribbons."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Load initial state
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True) if init_frame.mass is not None else np.ones(len(self.pos))
        self.body_id = np.array(init_frame.body_id, dtype=np.int64, copy=True) if init_frame.body_id is not None else np.zeros(len(self.pos), dtype=np.int64)

        self.num_particles = len(self.pos)
        self.N_ribbon = self.num_particles // 2  # 145 per ribbon

        # Load mesh topology
        root = Path(__file__).resolve().parent.parent / "assets"
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        faces_single = np.array(meta["topology"]["faces"], dtype=np.int64)
        grid_shape = meta["topology"]["grid_shape"]  # [29, 5]
        self.Nu = grid_shape[0]  # 29
        self.Nv = grid_shape[1]  # 5

        # Ribbon 0 and Ribbon 1 faces
        faces_r0 = faces_single.copy()
        faces_r1 = faces_single + self.N_ribbon
        self.faces = np.vstack([faces_r0, faces_r1])

        # Identify endpoints/boundaries to pull
        # Ribbon 1 (body 0): runs along x roughly, from u=0 to u=28
        # Ribbon 2 (body 1): runs along y roughly, from u=0 to u=28
        # Ribbon 1 ends: u=0 (indices 0*5 + v = 0..4? Wait, let's check grid indexing)
        # In grid_shape [29, 5], index = v * Nu + u or u * Nv + v?
        # Let's verify grid indexing from faces: face (0, 1, 30), (0, 30, 29).
        # Notice 29 is the row stride: node (u, v) -> v * 29 + u.
        # Node (u, 0) is index u. Node (u, 1) is index 29 + u.
        # So u runs 0..28, v runs 0..4.
        # End 0: u = 0 -> indices [0, 29, 58, 87, 116]
        # End 1: u = 28 -> indices [28, 57, 86, 115, 144]
        self.r0_end_minus = np.array([0, 29, 58, 87, 116], dtype=np.int64)
        self.r0_end_plus = np.array([28, 57, 86, 115, 144], dtype=np.int64)

        self.r1_end_minus = self.r0_end_minus + self.N_ribbon
        self.r1_end_plus = self.r0_end_plus + self.N_ribbon

        self.pull_speed = float(self.config.get("pull_speed", 0.32))
        self.thickness = float(self.config.get("ribbon", {}).get("thickness", 0.006))
        self.contact_length_scale = self.thickness
        self.strain_upper = float(self.config.get("ribbon", {}).get("strain_upper", 1.134))
        self.friction = float(self.config.get("ribbon", {}).get("friction", 0.02))

        # Build springs/edges for cloth simulation
        edges = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i+1)%3]
                if u > v:
                    u, v = v, u
                edges.add((u, v))

        # Also add bending/shear springs across the grid for realistic ribbon mechanics
        for r_offset in [0, self.N_ribbon]:
            for v in range(self.Nv):
                for u in range(self.Nu):
                    idx = r_offset + v * self.Nu + u
                    # Along u (length)
                    if u + 1 < self.Nu:
                        idx_u = r_offset + v * self.Nu + (u + 1)
                        edges.add((min(idx, idx_u), max(idx, idx_u)))
                    if u + 2 < self.Nu: # bending spring along length
                        idx_u2 = r_offset + v * self.Nu + (u + 2)
                        edges.add((min(idx, idx_u2), max(idx, idx_u2)))
                    # Along v (width)
                    if v + 1 < self.Nv:
                        idx_v = r_offset + (v + 1) * self.Nu + u
                        edges.add((min(idx, idx_v), max(idx, idx_v)))
                    if v + 2 < self.Nv: # bending spring along width
                        idx_v2 = r_offset + (v + 2) * self.Nu + u
                        edges.add((min(idx, idx_v2), max(idx, idx_v2)))
                    # Diagonals (shear)
                    if u + 1 < self.Nu and v + 1 < self.Nv:
                        idx_d1 = r_offset + (v + 1) * self.Nu + (u + 1)
                        edges.add((min(idx, idx_d1), max(idx, idx_d1)))
                    if u + 1 < self.Nu and v - 1 >= 0:
                        idx_d2 = r_offset + (v - 1) * self.Nu + (u + 1)
                        edges.add((min(idx, idx_d2), max(idx, idx_d2)))

        self.edges = np.array(list(edges), dtype=np.int64)
        # Compute rest lengths
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Store initial rest positions for deformation gradient calculation
        self.pos_rest = self.pos.copy()

        # Store fixed / pulled node masks
        self.is_pulled = np.zeros(self.num_particles, dtype=bool)
        self.is_pulled[self.r0_end_minus] = True
        self.is_pulled[self.r0_end_plus] = True
        self.is_pulled[self.r1_end_minus] = True
        self.is_pulled[self.r1_end_plus] = True

        # Pull velocities for each end
        # Ribbon 0 ends pulled along -x and +x
        # Ribbon 1 ends pulled along -y and +y
        self.pull_vel = np.zeros_like(self.pos)
        self.pull_vel[self.r0_end_minus] = np.array([-self.pull_speed, 0.0, 0.0])
        self.pull_vel[self.r0_end_plus] = np.array([ self.pull_speed, 0.0, 0.0])
        self.pull_vel[self.r1_end_minus] = np.array([0.0, -self.pull_speed, 0.0])
        self.pull_vel[self.r1_end_plus] = np.array([0.0,  self.pull_speed, 0.0])

        # Cumulative mass source
        self.cumulative_mass_source = np.array([0.0, 0.0], dtype=np.float64)

        # Material coordinates per element for deformation gradient
        # Triangles in material space (using UV or rest positions)
        self.rest_tri_inv = []
        for face in self.faces:
            # Face particles
            r0 = self.pos_rest[face[0]]
            r1 = self.pos_rest[face[1]]
            r2 = self.pos_rest[face[2]]
            # 3D edge vectors
            v1 = r1 - r0
            v2 = r2 - r0
            # Construct 2D local coordinate basis
            e1 = v1 / (np.linalg.norm(v1) + 1e-12)
            norm_v = np.cross(e1, v2)
            e2 = np.cross(norm_v, e1)
            e2 = e2 / (np.linalg.norm(e2) + 1e-12)
            # Local 2D rest coordinates
            u1 = np.array([np.dot(v1, e1), np.dot(v1, e2)])
            u2 = np.array([np.dot(v2, e1), np.dot(v2, e2)])
            Dm = np.column_stack([u1, u2])
            try:
                Dm_inv = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                Dm_inv = np.eye(2)
            self.rest_tri_inv.append((Dm_inv, e1, e2))

    def _step(self, dt: float) -> None:
        """Sub-step physics integration with PBD / XPBD and collision impulses."""
        # 1. Update prescribed boundary nodes
        self.pos[self.is_pulled] += self.pull_vel[self.is_pulled] * dt
        self.vel[self.is_pulled] = self.pull_vel[self.is_pulled]

        # 2. Integrate free nodes (gravity is [0,0,0], damping applied)
        free_mask = ~self.is_pulled
        damping = 0.985
        self.vel[free_mask] *= damping
        pos_prev = self.pos.copy()
        self.pos[free_mask] += self.vel[free_mask] * dt

        # 3. Position-Based Dynamics (PBD) Constraint Projection
        num_solver_iters = 15
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]

        w0 = np.where(self.is_pulled[e0], 0.0, 1.0 / self.mass[e0])
        w1 = np.where(self.is_pulled[e1], 0.0, 1.0 / self.mass[e1])
        w_sum = w0 + w1
        valid_springs = w_sum > 0

        # Collision parameters
        collision_dist = self.thickness * 1.5  # contact threshold

        for _ in range(num_solver_iters):
            # Distance / strain constraints
            dp = self.pos[e1] - self.pos[e0]
            dist = np.linalg.norm(dp, axis=1) + 1e-12
            # Allow slight stretch up to strain_upper, but enforce stiffness
            # Rest distance:
            c = dist - self.rest_lengths
            # Strict strain upper limit
            max_len = self.rest_lengths * self.strain_upper
            c = np.where(dist > max_len, dist - max_len, c * 0.8)

            corr = (c / (w_sum + 1e-12))[:, None] * (dp / dist[:, None])

            # Apply corrections
            np.add.at(self.pos, e0, np.where(valid_springs[:, None], (w0 / (w_sum + 1e-12))[:, None] * corr, 0.0))
            np.add.at(self.pos, e1, np.where(valid_springs[:, None], -(w1 / (w_sum + 1e-12))[:, None] * corr, 0.0))

            # Re-enforce prescribed positions
            self.pos[self.is_pulled] = pos_prev[self.is_pulled] + self.pull_vel[self.is_pulled] * dt

            # Inter-ribbon and self collision handling (particle-particle repulsion)
            # Sample subset of node pairs between body 0 and body 1 or spatial hash
            # Since total nodes = 290, all pairs is only 290*290/2 = 42,000, very fast!
            p = self.pos
            diff = p[:, None, :] - p[None, :, :]  # [N, N, 3]
            dists = np.linalg.norm(diff, axis=-1) + 1e-12  # [N, N]

            # Mask out same node or immediate mesh neighbors
            mask = dists < collision_dist
            np.fill_diagonal(mask, False)
            # Remove pairs that are adjacent on same ribbon
            for u, v in self.edges:
                mask[u, v] = False
                mask[v, u] = False

            pair_i, pair_j = np.where(mask)
            # Keep only i < j to avoid double counting
            valid_pairs = pair_i < pair_j
            pi = pair_i[valid_pairs]
            pj = pair_j[valid_pairs]

            if len(pi) > 0:
                p_dist = dists[pi, pj]
                penetration = collision_dist - p_dist
                norm_dir = diff[pi, pj] / p_dist[:, None]

                # Mass weights
                wpi = np.where(self.is_pulled[pi], 0.0, 1.0 / self.mass[pi])
                wpj = np.where(self.is_pulled[pj], 0.0, 1.0 / self.mass[pj])
                w_tot = wpi + wpj + 1e-12

                # Positional push
                delta_pi = norm_dir * (penetration * (wpi / w_tot))[:, None]
                delta_pj = -norm_dir * (penetration * (wpj / w_tot))[:, None]

                np.add.at(self.pos, pi, delta_pi)
                np.add.at(self.pos, pj, delta_pj)

                # Friction correction
                if self.friction > 0:
                    rel_vel = self.vel[pi] - self.vel[pj]
                    rel_tan = rel_vel - np.sum(rel_vel * norm_dir, axis=-1, keepdims=True) * norm_dir
                    fric_corr = rel_tan * min(self.friction, 0.5)
                    self.vel[pi] -= fric_corr * 0.5
                    self.vel[pj] += fric_corr * 0.5

            # Re-enforce pulled ends
            self.pos[self.is_pulled] = pos_prev[self.is_pulled] + self.pull_vel[self.is_pulled] * dt

        # 4. Update velocities
        self.vel[free_mask] = (self.pos[free_mask] - pos_prev[free_mask]) / dt
        self.vel[self.is_pulled] = self.pull_vel[self.is_pulled]

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute deformation gradient F for each particle or element.
        Shape required: [N, D, D] = [290, 3, 3]."""
        # We can compute F per node by averaging incident face deformation gradients
        F_nodes = np.zeros((self.num_particles, 3, 3), dtype=np.float64)
        node_counts = np.zeros(self.num_particles, dtype=np.float64)

        for idx, face in enumerate(self.faces):
            r0 = self.pos[face[0]]
            r1 = self.pos[face[1]]
            r2 = self.pos[face[2]]

            v1 = r1 - r0
            v2 = r2 - r0

            Dm_inv, e1, e2 = self.rest_tri_inv[idx]

            # Local 2D coordinates in deformed state
            e1_cur = v1 / (np.linalg.norm(v1) + 1e-12)
            norm_cur = np.cross(e1_cur, v2)
            e2_cur = np.cross(norm_cur, e1_cur)
            e2_cur = e2_cur / (np.linalg.norm(e2_cur) + 1e-12)

            u1_cur = np.array([np.dot(v1, e1_cur), np.dot(v1, e2_cur)])
            u2_cur = np.array([np.dot(v2, e1_cur), np.dot(v2, e2_cur)])
            Ds = np.column_stack([u1_cur, u2_cur])

            # 2D deformation gradient
            F_2d = Ds @ Dm_inv

            # Embed into 3D
            F_3d = np.eye(3, dtype=np.float64)
            F_3d[0:2, 0:2] = F_2d

            for node in face:
                F_nodes[node] += F_3d
                node_counts[node] += 1.0

        # Average
        mask = node_counts > 0
        F_nodes[mask] /= node_counts[mask, None, None]
        F_nodes[~mask] = np.eye(3, dtype=np.float64)
        return F_nodes

    def _compute_contact_signed_distance(self) -> np.ndarray:
        """Compute signed distance at contact / near-contact samples.
        Required shape: [C] = [290]."""
        p = self.pos
        diff = p[:, None, :] - p[None, :, :]  # [N, N, 3]
        dists = np.linalg.norm(diff, axis=-1)  # [N, N]

        # Mask out self-particles and adjacent ribbon neighbors
        np.fill_diagonal(dists, 1e9)
        for u, v in self.edges:
            dists[u, v] = 1e9
            dists[v, u] = 1e9

        # For each particle, find closest non-neighbor particle distance
        min_dists = np.min(dists, axis=1)
        # Signed distance: gap minus thickness
        signed_dist = min_dists - self.thickness
        return signed_dist

    def advance_to(self, time: float) -> StateFrame:
        """Advance the simulation to the requested sample time."""
        target_time = float(time)
        dt = 0.001  # Internal sub-step for physical stability

        while self.current_time < target_time - 1e-7:
            step_dt = min(dt, target_time - self.current_time)
            self._step(step_dt)
            self.current_time += step_dt

        self.current_time = target_time

        # Build required StateFrame output fields
        F_nodes = self._compute_deformation_gradient()
        contact_dist = self._compute_contact_signed_distance()

        attributes = {
            "deformation_gradient": F_nodes,
            "mass_quantity": self.mass.copy()
        }

        fields = {
            "contact_signed_distance": contact_dist,
            "cumulative_mass_source": self.cumulative_mass_source.copy()
        }

        scalars = {
            "contact_length_scale": float(self.contact_length_scale)
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.num_particles,
        }

