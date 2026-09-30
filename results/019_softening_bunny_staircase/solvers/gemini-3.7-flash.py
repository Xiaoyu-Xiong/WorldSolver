"""Physically-based viscoplastic / softening deformable body solver for bunny staircase."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.pos = None
        self.vel = None
        self.rest_pos = None
        self.faces = None
        self.edges = None
        self.rest_lengths = None
        self.plastic_rest_lengths = None
        self.plastic_strain = None
        self.softening_factor = None
        self.boxes = []
        self.mass = None
        self.inv_mass = None
        self.stiffness = 22.0
        self.softening_rate = 0.34
        self.damping = 2.3
        self.restitution = 0.12
        self.friction = 0.72
        self.contact_length_scale = 0.04
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.deformation_gradient = None
        self.contact_signed_distance = None
        self.interaction_force = None
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)
        self.dt_sub = 0.001
        self.num_vertices = 0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        cfg = scene.config
        self.stiffness = float(cfg.get("stiffness", 22.0))
        self.softening_rate = float(cfg.get("softening_rate", 0.34))
        self.damping = float(cfg.get("damping", 2.3))
        self.restitution = float(cfg.get("restitution", 0.12))
        self.friction = float(cfg.get("friction", 0.72))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Load mesh
        asset_path = Path(__file__).resolve().parent.parent / cfg["subject_asset"]
        with np.load(asset_path) as data:
            self.pos = np.asarray(data["vertices"], dtype=np.float64).copy()
            self.faces = np.asarray(data["faces"], dtype=np.int64).copy()

        self.rest_pos = self.pos.copy()
        self.num_vertices = len(self.pos)

        # Initial velocity
        init_vel = np.asarray(cfg.get("initial_velocity", [0.65, 0.0, -0.15]), dtype=np.float64)
        self.vel = np.tile(init_vel, (self.num_vertices, 1))

        # Masses
        total_mass = float(cfg.get("subject_mass", 2.4))
        self.mass = np.full(self.num_vertices, total_mass / self.num_vertices, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass

        # Extract unique edges and volumetric cross-springs / nearest-neighbor connections
        # 1. Surface edges from faces
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                edges_set.add((min(u, v), max(u, v)))

        # 2. Internal structural springs using k-nearest neighbors in rest configuration
        # to ensure 3D volumetric integrity and viscoplastic drooping
        from scipy.spatial import cKDTree
        tree = cKDTree(self.rest_pos)
        # Query nearest neighbors within a radius
        k_neighbors = 14
        _, idxs = tree.query(self.rest_pos, k=k_neighbors)
        for i in range(self.num_vertices):
            for j in idxs[i, 1:]:
                edges_set.add((min(i, int(j)), max(i, int(j))))

        self.edges = np.array(list(edges_set), dtype=np.int64)
        diff = self.rest_pos[self.edges[:, 0]] - self.rest_pos[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(diff, axis=1)
        self.plastic_rest_lengths = self.rest_lengths.copy()
        self.plastic_strain = np.zeros(len(self.edges), dtype=np.float64)
        self.softening_factor = np.ones(len(self.edges), dtype=np.float64)

        # Load collision geometry from scene_geometry.yaml
        sg_path = Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
        with open(sg_path) as f:
            sg = yaml.safe_load(f)

        self.boxes = []
        for obj in sg.get("objects", []):
            if obj.get("collision_enabled", False) and obj.get("geometry", {}).get("type") == "box":
                he = np.array(obj["geometry"]["half_extents"], dtype=np.float64)
                trans = np.array(obj["transform"]["translation"], dtype=np.float64)
                box_min = trans - he
                box_max = trans + he
                self.boxes.append((box_min, box_max, he, trans))

        self.current_time = 0.0
        self.deformation_gradient = np.tile(np.eye(3, dtype=np.float64), (self.num_vertices, 1, 1))
        self.contact_signed_distance = self._compute_sdf(self.pos)
        self.interaction_force = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)

    def _compute_sdf(self, pos: np.ndarray) -> np.ndarray:
        # Distance to ground / stairs
        # For each point, compute min signed distance across all boxes
        # Point inside box: negative distance to closest boundary
        # Point outside: positive Euclidean distance to box surface
        min_sd = np.full(len(pos), 1e9, dtype=np.float64)
        for box_min, box_max, _, _ in self.boxes:
            d_min = pos - box_min  # > 0 inside min bound
            d_max = box_max - pos  # > 0 inside max bound
            # Penetration depths into each face
            q = np.maximum(box_min - pos, np.maximum(pos - box_max, 0.0))
            outside_dist = np.linalg.norm(q, axis=1)
            # Inside distance: negative of min margin to boundary
            inside_margin = np.minimum(d_min, d_max)
            min_inside_margin = np.min(inside_margin, axis=1) # positive if completely inside

            is_inside = np.all(inside_margin > 0, axis=1)
            sd = np.where(is_inside, -min_inside_margin, outside_dist)
            min_sd = np.minimum(min_sd, sd)
        return min_sd

    def _compute_deformation_gradients(self) -> np.ndarray:
        # Compute local deformation gradient F for each vertex by fitting local neighborhood transformation
        # x_i - x_j approx F (X_i - X_j)
        # Using SVD / covariance of rest to current relative positions
        # For efficiency, compute per-vertex F using local neighbor edges
        num_v = self.num_vertices
        F = np.tile(np.eye(3, dtype=np.float64), (num_v, 1, 1))

        # Vectorized or simple local calculation
        p0 = self.edges[:, 0]
        p1 = self.edges[:, 1]
        dX = self.rest_pos[p1] - self.rest_pos[p0]
        dx = self.pos[p1] - self.pos[p0]

        # Accumulate covariance matrices
        # We can construct covariance C_XX and C_xX for each vertex
        # To keep it fast and robust:
        Cxx = np.zeros((num_v, 3, 3), dtype=np.float64)
        CxX = np.zeros((num_v, 3, 3), dtype=np.float64)

        for k in range(3):
            for m in range(3):
                val_XX = dX[:, k] * dX[:, m]
                val_xX = dx[:, k] * dX[:, m]
                np.add.at(Cxx[:, k, m], p0, val_XX)
                np.add.at(Cxx[:, k, m], p1, val_XX)
                np.add.at(CxX[:, k, m], p0, val_xX)
                np.add.at(CxX[:, k, m], p1, val_xX)

        # Invert Cxx and compute F = CxX @ inv(Cxx)
        eps = 1e-6
        for i in range(num_v):
            A = Cxx[i] + eps * np.eye(3)
            try:
                F[i] = CxX[i] @ np.linalg.inv(A)
            except Exception:
                F[i] = np.eye(3)
        return F

    def _step(self, dt: float) -> None:
        # 1. Internal spring forces with work-softening viscoplasticity
        # Edge indices
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]

        p0 = self.pos[i0]
        p1 = self.pos[i1]
        diff = p1 - p0
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-8)
        dir_vec = diff / dist[:, None]

        # Strain relative to plastic rest length
        strain = (dist - self.plastic_rest_lengths) / self.rest_lengths

        # Viscoplastic update: plastic flow when strain exceeds yield threshold
        yield_strain = 0.08
        plastic_rate = 12.0 * self.softening_rate
        excess_strain = np.abs(strain) - yield_strain
        flow_mask = excess_strain > 0
        if np.any(flow_mask):
            delta_Lp = np.sign(strain[flow_mask]) * excess_strain[flow_mask] * self.rest_lengths[flow_mask] * (1.0 - np.exp(-plastic_rate * dt))
            self.plastic_rest_lengths[flow_mask] += delta_Lp
            self.plastic_strain[flow_mask] += np.abs(delta_Lp) / self.rest_lengths[flow_mask]

            # Work softening: stiffness decreases as plastic work accumulates
            self.softening_factor[flow_mask] = np.maximum(0.08, 1.0 / (1.0 + 3.5 * self.softening_rate * self.plastic_strain[flow_mask]))

        # Effective stiffness
        k_eff = self.stiffness * 180.0 * self.softening_factor

        # Elastic spring forces
        force_mag = k_eff * (dist - self.plastic_rest_lengths)
        spring_forces = force_mag[:, None] * dir_vec

        # Damping forces along springs
        v_diff = self.vel[i1] - self.vel[i0]
        v_rel_norm = np.sum(v_diff * dir_vec, axis=1)
        damping_force_mag = self.damping * 4.0 * v_rel_norm
        spring_forces += damping_force_mag[:, None] * dir_vec

        # Accumulate forces on vertices
        net_forces = np.zeros_like(self.pos)
        np.add.at(net_forces, i0, spring_forces)
        np.add.at(net_forces, i1, -spring_forces)

        # Add gravity
        net_forces += self.mass[:, None] * self.gravity[None, :]

        # Global velocity damping
        self.vel *= (1.0 - 0.1 * self.damping * dt)

        # Semi-implicit Euler step
        self.vel += (net_forces * self.inv_mass[:, None]) * dt
        self.pos += self.vel * dt

        # 2. Collision resolution with stairs and ground
        total_contact_force = np.zeros(3, dtype=np.float64)

        for box_min, box_max, _, _ in self.boxes:
            # Check if vertices penetrate box
            in_x = (self.pos[:, 0] >= box_min[0]) & (self.pos[:, 0] <= box_max[0])
            in_y = (self.pos[:, 1] >= box_min[1]) & (self.pos[:, 1] <= box_max[1])
            in_z = (self.pos[:, 2] >= box_min[2]) & (self.pos[:, 2] <= box_max[2])
            colliding = in_x & in_y & in_z

            if not np.any(colliding):
                continue

            idx = np.where(colliding)[0]
            pts = self.pos[idx]

            # Distances to the 6 faces
            d_left   = pts[:, 0] - box_min[0]
            d_right  = box_max[0] - pts[:, 0]
            d_front  = pts[:, 1] - box_min[1]
            d_back   = box_max[1] - pts[:, 1]
            d_bottom = pts[:, 2] - box_min[2]
            d_top    = box_max[2] - pts[:, 2]

            d_all = np.stack([d_left, d_right, d_front, d_back, d_bottom, d_top], axis=1)
            min_face = np.argmin(d_all, axis=1)

            # Normal vectors for faces: [-x, +x, -y, +y, -z, +z]
            normals = np.array([
                [-1.0,  0.0,  0.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [ 0.0,  0.0,  1.0]
            ], dtype=np.float64)

            for k, f_idx in enumerate(min_face):
                v_idx = idx[k]
                n = normals[f_idx]
                pen = d_all[k, f_idx]

                # Push vertex out of collision
                self.pos[v_idx] += n * (pen + 1e-4)

                # Velocity resolution
                v = self.vel[v_idx]
                vn = np.dot(v, n)
                if vn < 0:
                    v_normal = vn * n
                    v_tangent = v - v_normal

                    # Normal impulse with restitution
                    v_normal_new = -self.restitution * v_normal

                    # Coulomb friction
                    tangent_speed = np.linalg.norm(v_tangent)
                    if tangent_speed > 1e-8:
                        friction_factor = max(0.0, 1.0 - self.friction * (1.0 + self.restitution) * abs(vn) / tangent_speed)
                        v_tangent_new = v_tangent * friction_factor
                    else:
                        v_tangent_new = np.zeros(3)

                    delta_v = (v_normal_new + v_tangent_new) - v
                    self.vel[v_idx] = v_normal_new + v_tangent_new

                    # Contact force impulse
                    impulse = delta_v * self.mass[v_idx]
                    total_contact_force += impulse / dt

        self.interaction_force = total_contact_force.reshape(1, 3)

    def advance_to(self, target_time: float) -> StateFrame:
        # Substep until reaching target_time
        while self.current_time + 1e-9 < target_time:
            dt = min(self.dt_sub, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time

        # Compute trajectory state outputs
        self.contact_signed_distance = self._compute_sdf(self.pos)
        self.deformation_gradient = self._compute_deformation_gradients()

        gravity = self.gravity
        external_force = self.mass[:, None] * gravity[None, :]

        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self.deformation_gradient.copy(),
            },
            fields={
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": external_force,
                "contact_signed_distance": self.contact_signed_distance.copy(),
                "interaction_force": self.interaction_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_plastic_strain": float(np.mean(self.plastic_strain)) if self.plastic_strain is not None else 0.0,
            "mean_softening": float(np.mean(self.softening_factor)) if self.softening_factor is not None else 1.0,
        }

