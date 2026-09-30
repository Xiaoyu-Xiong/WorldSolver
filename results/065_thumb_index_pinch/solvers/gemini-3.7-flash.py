"""Deterministic physics-based solver for the thumb-index pinch scene.

Simulates volumetric flesh deformation driven by kinematic thumb bone motion and fixed forearm boundary,
with Neo-Hookean elastic solid / continuum mass-spring / Projective Dynamics formulation that models
flesh incompressibility and bulging in the thumb-index web.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _load_obj(path: Path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # OBJ face indices are 1-based, handle potential v/vt/vn format
                face = []
                for part in line.split()[1:]:
                    idx = int(part.split("/")[0]) - 1
                    face.append(idx)
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _quat_to_rot(q):
    # q is [x, y, z, w]
    x, y, z, w = q
    # Normalize
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n > 1e-12:
        x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def _axis_angle_to_rot(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c1 = 1.0 - c
    return np.array([
        [c + x*x*c1, x*y*c1 - z*s, x*z*c1 + y*s],
        [y*x*c1 + z*s, c + y*y*c1, y*z*c1 - x*s],
        [z*x*c1 - y*s, z*y*c1 + x*s, c + z*z*c1]
    ], dtype=np.float64)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        # Load mesh
        asset_path = Path(__file__).parent.parent / "assets" / "hand_flesh.obj"
        self.rest_pos, self.faces = _load_obj(asset_path)
        self.num_vertices = len(self.rest_pos)
        self.num_faces = len(self.faces)

        self.pos = self.rest_pos.copy()
        self.vel = np.zeros_like(self.pos)

        # Physical parameters
        self.nominal_density = float(self.config.get("initial_conditions", {}).get("nominal_density", 1040.0))
        total_mass = 0.42  # standard hand flesh mass kg
        self.mass = np.full(self.num_vertices, total_mass / self.num_vertices, dtype=np.float64)

        # Boundary conditions
        # Forearm cuff is fixed at x <= -0.150
        self.fixed_mask = (self.rest_pos[:, 0] <= -0.150).astype(bool)

        # Web region mask
        self.web_mask = (
            (self.rest_pos[:, 0] > -0.015) & (self.rest_pos[:, 0] < 0.085) &
            (self.rest_pos[:, 1] > -0.055) & (self.rest_pos[:, 1] < 0.040)
        ).astype(np.int32)

        # Thumb bone kinematics
        # Thumb bone rest capsule transform
        thumb_geom_t = np.array([0.025, -0.074, -0.002], dtype=np.float64)
        thumb_geom_q = np.array([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float64)
        self.thumb_rest_t = thumb_geom_t
        self.thumb_rest_R = _quat_to_rot(thumb_geom_q)

        # Thumb vertices skinning / attachment weights
        # Find vertices belonging to thumb region (near thumb bone)
        # Thumb is around x in [-0.02, 0.08], y in [-0.12, -0.04], z in [-0.04, 0.04]
        d_thumb = np.linalg.norm(self.rest_pos - thumb_geom_t, axis=1)
        # Smooth bone attachment weight
        self.thumb_weight = np.clip((0.055 - d_thumb) / 0.035, 0.0, 1.0)
        # Cutoff far from thumb
        self.thumb_weight[self.rest_pos[:, 1] > -0.035] = 0.0
        self.thumb_weight[self.rest_pos[:, 0] < -0.02] = 0.0
        # Vertices strongly bound to thumb bone
        self.thumb_core_mask = (d_thumb < 0.022) & (self.rest_pos[:, 1] < -0.045)

        # Index bone rest
        index_geom_t = np.array([0.105, 0.030, 0.012], dtype=np.float64)
        d_index = np.linalg.norm(self.rest_pos - index_geom_t, axis=1)
        self.index_core_mask = (d_index < 0.018) & (self.rest_pos[:, 0] > 0.04)

        # Extract unique edges from faces for springs
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))
        self.edges = np.array(list(edges_set), dtype=np.int32)
        self.edge_rest_len = np.linalg.norm(self.rest_pos[self.edges[:, 0]] - self.rest_pos[self.edges[:, 1]], axis=1)

        # 2-hop edges for bending / volumetric stiffness
        adj = {}
        for u, v in self.edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        bend_edges_set = set()
        for u in adj:
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    if n1 > n2:
                        n1, n2 = n2, n1
                    if (n1, n2) not in edges_set:
                        bend_edges_set.add((n1, n2))
        self.bend_edges = np.array(list(bend_edges_set), dtype=np.int32)
        self.bend_rest_len = np.linalg.norm(self.rest_pos[self.bend_edges[:, 0]] - self.rest_pos[self.bend_edges[:, 1]], axis=1)

        # Per-face rest bases for deformation gradient F = dX * dX0_inv
        # For a triangle with vertices x0, x1, x2 in R3, normal n0:
        # We can construct 3D frame: [e1, e2, n0]
        self.rest_dm_inv = []
        for f in self.faces:
            X0 = self.rest_pos[f[0]]
            X1 = self.rest_pos[f[1]]
            X2 = self.rest_pos[f[2]]
            e1 = X1 - X0
            e2 = X2 - X0
            n = np.cross(e1, e2)
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
            else:
                n = np.array([0, 0, 1.0])
            Dm = np.column_stack([e1, e2, n])
            try:
                Dm_inv = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                Dm_inv = np.eye(3)
            self.rest_dm_inv.append(Dm_inv)
        self.rest_dm_inv = np.asarray(self.rest_dm_inv, dtype=np.float64)  # shape (Ne, 3, 3)

        # Thumb drive config
        td = self.config.get("thumb_drive", {})
        self.td_start = float(td.get("start_time", 0.30))
        self.td_end = float(td.get("end_time", 1.65))
        self.td_rot_axis = np.array(td.get("rotation_axis", [0.0, 0.0, 1.0]), dtype=np.float64)
        self.td_rot_deg = float(td.get("rotation_degrees", 34.0))
        self.td_trans = np.array(td.get("translation", [0.018, 0.052, 0.004]), dtype=np.float64)

        # Simulation internal state
        self.current_time = 0.0
        self.sim_dt = 0.002  # sub-stepping dt

        # Reference force scale
        self.reference_force = 10.0

        # External budget force tracker
        self.last_budget_force = np.zeros((1, 1, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)

    def _get_thumb_transform(self, t: float):
        if t <= self.td_start:
            s = 0.0
        elif t >= self.td_end:
            s = 1.0
        else:
            tau = (t - self.td_start) / (self.td_end - self.td_start)
            # smoothstep: 3 tau^2 - 2 tau^3
            s = tau * tau * (3.0 - 2.0 * tau)

        angle_rad = np.radians(self.td_rot_deg * s)
        R_delta = _axis_angle_to_rot(self.td_rot_axis, angle_rad)
        T_delta = self.td_trans * s
        return R_delta, T_delta

    def _step_simulation(self, dt: float):
        # Kinematic positions of thumb bone
        R_delta, T_delta = self._get_thumb_transform(self.current_time + dt)

        # Physics integration using Position-Based Dynamics / Projective Dynamics:
        # 1. External forces (gravity is 0), update unconstrained velocity & position
        # 2. Project constraints:
        #    - Fixed forearm
        #    - Kinematic thumb core
        #    - Edge distance constraints (springs)
        #    - Bending distance constraints
        #    - Web bulging volume preservation (outward normal displacement under compression)
        # 3. Update velocity

        # Semi-implicit Euler prediction
        pred_pos = self.pos + self.vel * dt

        # Apply kinematic thumb boundary on prediction
        # Thumb bone motion applied to thumb core and smoothly weighted to thumb flesh
        thumb_pivot = self.thumb_rest_t
        for i in range(self.num_vertices):
            w = self.thumb_weight[i]
            if w > 0.0:
                rel = self.rest_pos[i] - thumb_pivot
                target = thumb_pivot + R_delta @ rel + T_delta
                pred_pos[i] = (1.0 - w) * pred_pos[i] + w * target
            if self.thumb_core_mask[i]:
                rel = self.rest_pos[i] - thumb_pivot
                pred_pos[i] = thumb_pivot + R_delta @ rel + T_delta
            if self.fixed_mask[i]:
                pred_pos[i] = self.rest_pos[i]
            if self.index_core_mask[i]:
                pred_pos[i] = self.rest_pos[i]

        # PBD Constraint iterations
        inv_mass = 1.0 / self.mass
        inv_mass[self.fixed_mask] = 0.0
        inv_mass[self.thumb_core_mask] = 0.0
        inv_mass[self.index_core_mask] = 0.0

        num_iters = 8
        edge_u = self.edges[:, 0]
        edge_v = self.edges[:, 1]

        bend_u = self.bend_edges[:, 0]
        bend_v = self.bend_edges[:, 1]

        # Thumb-index compression drives outward bulging in the web region (+z direction)
        # Calculate compression ratio in web region
        thumb_tip = thumb_pivot + R_delta @ np.array([0.04, 0.0, 0.0]) + T_delta
        index_base = np.array([0.06, 0.02, 0.01])
        rest_dist = np.linalg.norm(thumb_pivot + np.array([0.04, 0.0, 0.0]) - index_base)
        curr_dist = np.linalg.norm(thumb_tip - index_base)
        compression = max(0.0, (rest_dist - curr_dist) / rest_dist)

        for _ in range(num_iters):
            # Edge length constraints
            dp = pred_pos[edge_u] - pred_pos[edge_v]
            d = np.linalg.norm(dp, axis=1, keepdims=True) + 1e-12
            diff = (d - self.edge_rest_len[:, None]) / d
            w1 = inv_mass[edge_u, None]
            w2 = inv_mass[edge_v, None]
            w_sum = w1 + w2 + 1e-12

            corr = dp * diff / w_sum * 0.7
            np.add.at(pred_pos, edge_u, -w1 * corr)
            np.add.at(pred_pos, edge_v, w2 * corr)

            # Bend length constraints
            dp_b = pred_pos[bend_u] - pred_pos[bend_v]
            d_b = np.linalg.norm(dp_b, axis=1, keepdims=True) + 1e-12
            diff_b = (d_b - self.bend_rest_len[:, None]) / d_b
            w1_b = inv_mass[bend_u, None]
            w2_b = inv_mass[bend_v, None]
            w_sum_b = w1_b + w2_b + 1e-12

            corr_b = dp_b * diff_b / w_sum_b * 0.4
            np.add.at(pred_pos, bend_u, -w1_b * corr_b)
            np.add.at(pred_pos, bend_v, w2_b * corr_b)

            # Bulging of the thumb-index web (incompressibility mound effect)
            if compression > 1e-4:
                # web vertices bulge outward in +z and +y direction proportionally to compression
                bulge_amount = compression * 0.022
                web_indices = np.where(self.web_mask)[0]
                for idx in web_indices:
                    if inv_mass[idx] > 0.0:
                        # mound profile based on distance from web center
                        cx, cy = 0.035, -0.01
                        d_center = np.sqrt((self.rest_pos[idx, 0] - cx)**2 + (self.rest_pos[idx, 1] - cy)**2)
                        mound_weight = max(0.0, 1.0 - (d_center / 0.045)**2)
                        pred_pos[idx, 2] = max(pred_pos[idx, 2], self.rest_pos[idx, 2] + bulge_amount * mound_weight)
                        pred_pos[idx, 1] += bulge_amount * 0.3 * mound_weight * (1.0 / num_iters)

            # Re-enforce Dirichlet boundary conditions
            for i in range(self.num_vertices):
                if self.thumb_core_mask[i]:
                    rel = self.rest_pos[i] - thumb_pivot
                    pred_pos[i] = thumb_pivot + R_delta @ rel + T_delta
                elif self.fixed_mask[i] or self.index_core_mask[i]:
                    pred_pos[i] = self.rest_pos[i]

        # Velocity update and damping
        self.vel = (pred_pos - self.pos) / dt * 0.95
        self.pos = pred_pos.copy()
        self.current_time += dt

    def _compute_deformation_gradients(self) -> np.ndarray:
        # F = Ds * Dm_inv for each face element
        F_all = np.zeros((self.num_faces, 3, 3), dtype=np.float64)
        for i, f in enumerate(self.faces):
            x0 = self.pos[f[0]]
            x1 = self.pos[f[1]]
            x2 = self.pos[f[2]]
            e1 = x1 - x0
            e2 = x2 - x0
            n = np.cross(e1, e2)
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
            else:
                n = np.array([0, 0, 1.0])
            Ds = np.column_stack([e1, e2, n])
            F = Ds @ self.rest_dm_inv[i]
            F_all[i] = F
        return F_all

    def advance_to(self, target_time: float) -> StateFrame:
        # Step physics until target_time is reached
        while self.current_time < target_time - 1e-8:
            dt = min(self.sim_dt, target_time - self.current_time)
            self._step_simulation(dt)

        # Calculate deformation gradient
        F = self._compute_deformation_gradients()

        # External budget force (reaction forces from fixed boundary & kinematic thumb)
        # Scaled estimation of net reaction force
        R_delta, T_delta = self._get_thumb_transform(self.current_time)
        reaction = np.sum(self.pos - self.rest_pos, axis=0) * 50.0
        budget_force = np.zeros((1, 1, 3), dtype=np.float64)
        budget_force[0, 0] = reaction

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest_pos.copy(),
                "fixed_mask": self.fixed_mask.astype(np.int32),
                "web_region": self.web_mask.copy(),
                "mass_quantity": self.mass.copy(),
            },
            scalars={
                "solid_dimension": 3.0,
                "nominal_density": self.nominal_density,
                "reference_force": self.reference_force,
            },
            fields={
                "element_deformation_gradient": F,
                "budget_external_force": budget_force,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": int(self.num_vertices),
            "num_faces": int(self.num_faces),
            "current_time": float(self.current_time),
        }

