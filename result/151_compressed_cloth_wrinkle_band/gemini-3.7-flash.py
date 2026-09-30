"""Compressed Cloth Wrinkle Band physics solver."""
from __future__ import annotations
import numpy as np
import json
import os
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.time = 0.0
        self.substep_dt = 1e-3  # 1 ms sub-step for cloth PBD/XPBD

        # Mesh topology and geometry
        self.num_vertices = 0
        self.nx = 17
        self.ny = 17
        self.pos = None
        self.prev_pos = None
        self.vel = None
        self.inv_mass = None
        self.mass = None

        # Grid indexing
        self.left_boundary_idx = None
        self.right_boundary_idx = None

        # Kinematic grips
        self.initial_grip_left_x = -1.5
        self.initial_grip_right_x = 1.5
        self.target_compression_rate = 0.12  # rate at which each grip moves inward (m/s)
        self.max_compression = 0.45  # maximum inward displacement per grip

        # Constraints
        self.distance_constraints = []  # (i, j, rest_len)
        self.bending_constraints = []   # (i, j, rest_len) or dihedral

        # Faces and rest state
        self.faces = None
        self.rest_pos_2d = None
        self.inv_Dm = None

        # Physical parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.damping = 0.998
        self.thickness = 0.015

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64)
        self.prev_pos = np.array(self.pos, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.num_vertices = len(self.pos)

        # Cloth config
        cloth_cfg = scene.config.get("cloth", {})
        self.nx, self.ny = map(int, cloth_cfg.get("grid_shape", [17, 17]))
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Load topology
        topo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "cloth_topology.json")
        if os.path.exists(topo_path):
            with open(topo_path, "r") as f:
                topo_data = json.load(f)
                self.faces = np.array(topo_data.get("faces", []), dtype=np.int32)
        else:
            # Fallback grid triangulation
            faces = []
            for j in range(self.ny - 1):
                for i in range(self.nx - 1):
                    v0 = j * self.nx + i
                    v1 = j * self.nx + (i + 1)
                    v2 = (j + 1) * self.nx + i
                    v3 = (j + 1) * self.nx + (i + 1)
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
            self.faces = np.array(faces, dtype=np.int32)

        # Invert masses for boundary vs interior
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        # Boundary grip indices: left column (i=0) and right column (i=nx-1)
        # Note: grid is laid out with x along columns (nx) and y along rows (ny):
        # idx = j * nx + i
        left_idxs = []
        right_idxs = []
        for j in range(self.ny):
            left_idxs.append(j * self.nx + 0)
            right_idxs.append(j * self.nx + (self.nx - 1))
        self.left_boundary_idx = np.array(left_idxs, dtype=np.int32)
        self.right_boundary_idx = np.array(right_idxs, dtype=np.int32)

        # Pin grips kinematically (inv_mass = 0 for boundary vertices)
        self.inv_mass[self.left_boundary_idx] = 0.0
        self.inv_mass[self.right_boundary_idx] = 0.0

        # Rest 2D coordinates in XY plane
        self.rest_pos_2d = self.pos[:, :2].copy()

        # Build distance constraints (springs)
        constraints = set()
        # Structural & shear from faces
        for tri in self.faces:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                if a > b:
                    a, b = b, a
                constraints.add((a, b))

        # Bending springs (2-hop neighbors)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i + 2 < self.nx:
                    idx2 = j * self.nx + (i + 2)
                    constraints.add((min(idx, idx2), max(idx, idx2)))
                if j + 2 < self.ny:
                    idx2 = (j + 2) * self.nx + i
                    constraints.add((min(idx, idx2), max(idx, idx2)))

        dist_list = []
        for a, b in constraints:
            rest_l = np.linalg.norm(self.pos[a] - self.pos[b])
            dist_list.append((a, b, rest_l))
        self.distance_constraints = dist_list

        # Add slight initial out-of-plane perturbation to promote natural wrinkle nucleation
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if idx not in self.left_boundary_idx and idx not in self.right_boundary_idx:
                    # Sinusoidal perturbation matching wrinkle mode
                    x_frac = (i / (self.nx - 1) - 0.5) * np.pi
                    y_frac = (j / (self.ny - 1) - 0.5) * np.pi
                    pert = 0.005 * np.cos(3.0 * y_frac) * np.cos(x_frac)
                    self.pos[idx, 2] += pert
        self.prev_pos = self.pos.copy()

        # Precompute triangle inverse material coordinate matrices for deformation gradient
        self.inv_Dm = []
        for tri in self.faces:
            X0 = self.rest_pos_2d[tri[0]]
            X1 = self.rest_pos_2d[tri[1]]
            X2 = self.rest_pos_2d[tri[2]]
            Dm = np.column_stack([X1 - X0, X2 - X0])
            try:
                inv_D = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                inv_D = np.eye(2)
            self.inv_Dm.append(inv_D)
        self.inv_Dm = np.array(self.inv_Dm)

    def _grip_position(self, t: float):
        # Kinematic displacement of grips inward
        # Inward motion at constant speed up to max_compression
        disp = min(self.target_compression_rate * t, self.max_compression)
        left_x = self.initial_grip_left_x + disp
        right_x = self.initial_grip_right_x - disp
        return left_x, right_x

    def _substep(self, dt: float, current_time: float):
        # 1. Apply kinematic boundary conditions to pinned vertices
        left_x, right_x = self._grip_position(current_time + dt)

        # Update grip positions
        for idx in self.left_boundary_idx:
            self.pos[idx, 0] = left_x
        for idx in self.right_boundary_idx:
            self.pos[idx, 0] = right_x

        # 2. Integrate unconstrained positions (semi-implicit Euler / Verlet)
        unpinned = (self.inv_mass > 0)

        # Forces: gravity
        self.vel[unpinned] += self.gravity * dt
        self.vel[unpinned] *= self.damping

        self.prev_pos = self.pos.copy()
        self.pos[unpinned] += self.vel[unpinned] * dt

        # 3. Position-Based Dynamics (PBD) constraint projections
        iterations = 15
        for _ in range(iterations):
            # Enforce distance constraints
            for a, b, rest_len in self.distance_constraints:
                w1 = self.inv_mass[a]
                w2 = self.inv_mass[b]
                w_sum = w1 + w2
                if w_sum == 0:
                    continue

                diff = self.pos[a] - self.pos[b]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    continue

                # Unilateral or bilateral: cloth can resist stretch strongly, resist compression weakly
                # To simulate thin cloth wrinkling, stretch stiffness is high (1.0), compression allows buckling
                corr = (dist - rest_len) / dist
                corr_vec = diff * corr

                # Stretch resistance (strong)
                if dist > rest_len:
                    stiffness = 0.95
                    if w1 > 0:
                        self.pos[a] -= (w1 / w_sum) * corr_vec * stiffness
                    if w2 > 0:
                        self.pos[b] += (w2 / w_sum) * corr_vec * stiffness
                else:
                    # Weak compression resistance to allow wrinkle band formation
                    stiffness = 0.05
                    if w1 > 0:
                        self.pos[a] -= (w1 / w_sum) * corr_vec * stiffness
                    if w2 > 0:
                        self.pos[b] += (w2 / w_sum) * corr_vec * stiffness

            # Enforce boundary positions
            for idx in self.left_boundary_idx:
                self.pos[idx, 0] = left_x
            for idx in self.right_boundary_idx:
                self.pos[idx, 0] = right_x

            # Ground contact constraint (z >= thickness * 0.5)
            # Ground top is at z = 0.0 (half-extent 0.05, center -0.05)
            ground_z = 0.0
            contact_mask = self.pos[:, 2] < (ground_z + self.thickness * 0.5)
            self.pos[contact_mask, 2] = ground_z + self.thickness * 0.5

        # 4. Update velocities from position change
        self.vel = (self.pos - self.prev_pos) / dt
        self.vel[~unpinned] = 0.0

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward to target_time
        while self.time < target_time - 1e-7:
            dt = min(self.substep_dt, target_time - self.time)
            self._substep(dt, self.time)
            self.time += dt

        # Build trajectory output frame
        # 1. Deformation gradient F per vertex (2x2)
        # Compute triangle deformation gradients and map to vertices
        F_tri = np.zeros((len(self.faces), 2, 2), dtype=np.float64)
        for k, tri in enumerate(self.faces):
            x0 = self.pos[tri[0], :2]
            x1 = self.pos[tri[1], :2]
            x2 = self.pos[tri[2], :2]
            Ds = np.column_stack([x1 - x0, x2 - x0])
            F_tri[k] = Ds @ self.inv_Dm[k]

        # Accumulate to vertices
        F_vert = np.zeros((self.num_vertices, 2, 2), dtype=np.float64)
        vert_tri_count = np.zeros(self.num_vertices, dtype=np.float64)
        for k, tri in enumerate(self.faces):
            for v in tri:
                F_vert[v] += F_tri[k]
                vert_tri_count[v] += 1.0

        vert_tri_count = np.maximum(vert_tri_count, 1.0)[:, None, None]
        F_vert /= vert_tri_count

        # 2. Contact signed distance
        # Distance to ground plane (ground surface at z = 0)
        # and distance to grips
        contact_sd = self.pos[:, 2] - 0.0

        # 3. External force (gravity on dynamic vertices + contact force)
        ext_force = np.zeros((self.num_vertices, 3), dtype=np.float64)
        ext_force[:, :] = self.gravity[None, :] * self.mass[:, None]
        # Pinned boundary has reaction forces
        ext_force[self.left_boundary_idx] = 0.0
        ext_force[self.right_boundary_idx] = 0.0

        # Boundary vertex indicator
        boundary = np.zeros(self.num_vertices, dtype=np.int32)
        boundary[self.left_boundary_idx] = 1
        boundary[self.right_boundary_idx] = 1

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": F_vert,
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(self.num_vertices, dtype=np.float64),
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
            "time": float(self.time),
            "num_vertices": int(self.num_vertices),
            "mean_z": float(np.mean(self.pos[:, 2])) if self.pos is not None else 0.0,
        }

