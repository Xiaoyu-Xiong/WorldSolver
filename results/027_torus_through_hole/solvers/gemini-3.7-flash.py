"""BenchmarkSolver implementation for torus through hole."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load subject mesh
        asset_path = Path(self.config.get("subject_asset", "assets/subject.npz"))
        if not asset_path.is_absolute():
            asset_path = Path(scene.__module__ if hasattr(scene, "__file__") else ".").resolve().parent / asset_path
            if not asset_path.exists():
                asset_path = Path("assets/subject.npz").resolve()

        with np.load(asset_path) as data:
            self.rest_verts = np.asarray(data["vertices"], dtype=np.float64)
            self.faces = np.asarray(data["faces"], dtype=np.int64)

        self.num_verts = len(self.rest_verts)
        self.x = self.rest_verts.copy()

        # Initial velocities
        v_init = np.asarray(self.config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.v = np.tile(v_init, (self.num_verts, 1))

        # Mass
        total_mass = float(self.config.get("subject_mass", 1.6))
        self.mass = np.full(self.num_verts, total_mass / self.num_verts, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass

        # Gravity & constants
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.hole_z = float(self.config.get("hole_z", 0.62))
        self.hole_radius = float(self.config.get("hole_radius", 0.5))
        self.push_speed = float(self.config.get("push_speed", 0.48))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))

        # Physical parameters
        self.stiffness = float(self.config.get("stiffness", 11.0))
        self.damping = float(self.config.get("damping", 1.6))
        self.friction = float(self.config.get("friction", 0.48))
        self.restitution = float(self.config.get("restitution", 0.04))

        # Build mesh topology: edges (springs) and bending hinges / dihedral / 2-ring springs
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                edges_set.add(e)
        self.edges = np.array(list(edges_set), dtype=np.int64)

        # Edge rest lengths
        p0 = self.rest_verts[self.edges[:, 0]]
        p1 = self.rest_verts[self.edges[:, 1]]
        self.edge_rest_lens = np.linalg.norm(p0 - p1, axis=1)

        # Volume / shape preservation using 2-hop / diagonal springs
        # Build adjacency for 2-hop springs to provide shear and bending stiffness
        adj = [set() for _ in range(self.num_verts)]
        for e in self.edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])

        bending_edges_set = set()
        for i in range(self.num_verts):
            nbrs = list(adj[i])
            for idx1 in range(len(nbrs)):
                for idx2 in range(idx1 + 1, len(nbrs)):
                    u, w = nbrs[idx1], nbrs[idx2]
                    e = tuple(sorted([u, w]))
                    if e not in edges_set:
                        bending_edges_set.add(e)

        self.bending_edges = np.array(list(bending_edges_set), dtype=np.int64)
        bp0 = self.rest_verts[self.bending_edges[:, 0]]
        bp1 = self.rest_verts[self.bending_edges[:, 1]]
        self.bending_rest_lens = np.linalg.norm(bp0 - bp1, axis=1)

        # Pusher parameters
        self.pusher_radius = 0.48
        self.pusher_height = 0.18
        self.pusher_init_z = 1.88  # center z from scene_geometry.yaml

        # Internal torus volume / cross-section preservation
        # For a torus (R=0.42, r=0.17), ring center in rest coordinates
        rest_xy = self.rest_verts[:, :2]
        rest_r_xy = np.linalg.norm(rest_xy, axis=1)
        # major circle center for each vertex
        self.rest_phi = np.arctan2(self.rest_verts[:, 1], self.rest_verts[:, 0])
        self.major_R = 0.42
        self.minor_r = 0.17

        # Precompute per-vertex rest neighbor offsets for deformation gradient estimation
        self.vert_nbrs = [np.array(list(adj[i]), dtype=np.int64) for i in range(self.num_verts)]
        self.rest_nbr_inv_cov = []
        for i in range(self.num_verts):
            nbrs = self.vert_nbrs[i]
            dX0 = self.rest_verts[nbrs] - self.rest_verts[i]  # [K, 3]
            cov = dX0.T @ dX0 + 1e-6 * np.eye(3)
            self.rest_nbr_inv_cov.append(np.linalg.inv(cov))

        self.current_time = 0.0
        self.sub_dt = 0.0005  # sub-step size
        self.initialized = True

    def _compute_deformation_gradient(self) -> np.ndarray:
        F = np.zeros((self.num_verts, 3, 3), dtype=np.float64)
        for i in range(self.num_verts):
            nbrs = self.vert_nbrs[i]
            dX0 = self.rest_verts[nbrs] - self.rest_verts[i]
            dx = self.x[nbrs] - self.x[i]
            # dx ~ F @ dX0 => F = (dx.T @ dX0) @ (dX0.T @ dX0)^{-1}
            A = dx.T @ dX0
            F[i] = A @ self.rest_nbr_inv_cov[i]
        return F

    def _compute_contact_distance(self) -> np.ndarray:
        # Distance to ground
        dist_ground = self.x[:, 2] - self.ground_z

        # Distance to hole rim / torus shape of the plate
        # Hole plate torus shape is around z = hole_z, radius = hole_radius
        # Rigid hole boundary is at r_xy = 0.5 for |z - hole_z| <= plate_thickness
        r_xy = np.hypot(self.x[:, 0], self.x[:, 1])
        # Outside the hole in the plate region (e.g. z around 0.39 to 0.85 and r_xy >= 0.5)
        dist_hole = r_xy - self.hole_radius

        # Combined contact signed distance
        gap = np.minimum(dist_ground, dist_hole)
        return gap

    def _step(self, dt: float, current_sim_time: float) -> None:
        # Pusher position at time t
        # Pusher starts at z = 1.88 (center), bottom at 1.88 - 0.09 = 1.79
        # Pusher moves down at push_speed until it pushes the torus through the hole
        pusher_center_z = self.pusher_init_z - self.push_speed * current_sim_time
        pusher_bottom_z = pusher_center_z - self.pusher_height / 2.0
        pusher_top_z = pusher_center_z + self.pusher_height / 2.0
        pusher_vel_z = -self.push_speed

        # Integrate forces using symplectic Euler / PBD hybrid for high stability
        # External forces
        ext_f = self.mass[:, None] * self.gravity[None, :]  # gravity

        # Explicit velocity update from external forces and damping
        self.v += (ext_f * self.inv_mass[:, None]) * dt
        # Velocity damping
        self.v *= np.exp(-self.damping * dt)

        # Position prediction
        x_pred = self.x + self.v * dt

        # Position-based constraint projection (XPBD / PBD iterations)
        # 1. Edge length constraints (stretching)
        # 2. Bending edge constraints (bending/shearing)
        # 3. Collision constraints (hole plate, pusher, ground)
        num_iters = 8
        alpha_stretch = 0.95
        alpha_bend = 0.60

        for _ in range(num_iters):
            # --- Edge stretch constraints ---
            p0 = x_pred[self.edges[:, 0]]
            p1 = x_pred[self.edges[:, 1]]
            disp = p0 - p1
            dist = np.linalg.norm(disp, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-8)
            diff = (dist - self.edge_rest_lens[:, None]) / dist

            # Corrections
            w0 = self.inv_mass[self.edges[:, 0], None]
            w1 = self.inv_mass[self.edges[:, 1], None]
            w_sum = w0 + w1
            corr = (disp * diff) / w_sum * alpha_stretch

            np.add.at(x_pred, self.edges[:, 0], -w0 * corr)
            np.add.at(x_pred, self.edges[:, 1], w1 * corr)

            # --- Bending constraints ---
            bp0 = x_pred[self.bending_edges[:, 0]]
            bp1 = x_pred[self.bending_edges[:, 1]]
            bdisp = bp0 - bp1
            bdist = np.linalg.norm(bdisp, axis=1, keepdims=True)
            bdist = np.maximum(bdist, 1e-8)
            bdiff = (bdist - self.bending_rest_lens[:, None]) / bdist

            bw0 = self.inv_mass[self.bending_edges[:, 0], None]
            bw1 = self.inv_mass[self.bending_edges[:, 1], None]
            bw_sum = bw0 + bw1
            bcorr = (bdisp * bdiff) / bw_sum * alpha_bend

            np.add.at(x_pred, self.bending_edges[:, 0], -bw0 * bcorr)
            np.add.at(x_pred, self.bending_edges[:, 1], bw1 * bcorr)

            # --- Collision with Rigid Hole Plate ---
            # The plate has a circular hole of radius R_hole = self.hole_radius (0.5)
            # The plate is centered around hole_z = 0.62 with rounded/filleted rim.
            # Torus cross-section profile of the plate rim: center circle at (r=0.73, z=0.62) with minor radius 0.23
            # or simply: a plate spanning z in [0.39, 0.85] with inner radius 0.50
            # For vertices with r_xy >= 0.5 and z in [0.39, 0.85], push them out towards r_xy < 0.5 or outside z range.
            # The rim is a smooth rounded torus around r = 0.73, z = 0.62, minor radius = 0.23
            # Closest point on the inner rim surface (a torus tube of R_tube = 0.73, r_tube = 0.23):
            px = x_pred[:, 0]
            py = x_pred[:, 1]
            pz = x_pred[:, 2]
            r_xy = np.hypot(px, py)
            cos_theta = np.where(r_xy > 1e-8, px / r_xy, 1.0)
            sin_theta = np.where(r_xy > 1e-8, py / r_xy, 0.0)

            # Rim torus center circle has R_rim = 0.73, Z_rim = 0.62, r_tube = 0.23
            # Rim inner boundary is at R_rim - r_tube = 0.50, z = 0.62
            R_rim = 0.73
            Z_rim = self.hole_z
            r_tube = 0.23

            # Distance of vertex (r_xy, pz) to rim circle (R_rim, Z_rim)
            dr = r_xy - R_rim
            dz = pz - Z_rim
            dist_to_rim_core = np.hypot(dr, dz)

            # Penetration into the solid plate / rim:
            # Inside the tube: dist_to_rim_core < r_tube
            inside_rim = dist_to_rim_core < r_tube
            if np.any(inside_rim):
                idx = np.where(inside_rim)[0]
                d_core = np.maximum(dist_to_rim_core[idx], 1e-6)
                # Normal points outward from the solid plate interior towards the hole/air
                # Outside the tube is air, inside the tube is plate body
                norm_r = -dr[idx] / d_core
                norm_z = -dz[idx] / d_core

                penetration = r_tube - d_core
                # Push vertex out
                corr_r = penetration * norm_r
                corr_z = penetration * norm_z

                x_pred[idx, 0] += corr_r * cos_theta[idx]
                x_pred[idx, 1] += corr_r * sin_theta[idx]
                x_pred[idx, 2] += corr_z

            # Also ensure plate body for r_xy >= R_rim and |pz - Z_rim| <= r_tube is blocked
            inside_plate_bulk = (r_xy >= R_rim) & (np.abs(pz - Z_rim) < r_tube)
            if np.any(inside_plate_bulk):
                idx_b = np.where(inside_plate_bulk)[0]
                # Push to top or bottom of plate
                top_dist = (Z_rim + r_tube) - pz[idx_b]
                bot_dist = pz[idx_b] - (Z_rim - r_tube)
                push_top = top_dist < bot_dist

                z_new = np.where(push_top, Z_rim + r_tube, Z_rim - r_tube)
                x_pred[idx_b, 2] = z_new

            # --- Collision with Pusher (Cylinder) ---
            # Pusher is a cylinder of radius 0.48, height 0.18, moving downwards
            # pusher_center_z, pusher_bottom_z, pusher_top_z
            # Pusher penetrates when r_xy <= pusher_radius and pusher_bottom_z <= pz <= pusher_top_z
            px_cur = x_pred[:, 0]
            py_cur = x_pred[:, 1]
            pz_cur = x_pred[:, 2]
            r_xy_cur = np.hypot(px_cur, py_cur)

            inside_pusher = (r_xy_cur <= self.pusher_radius) & (pz_cur >= pusher_bottom_z) & (pz_cur <= pusher_top_z)
            if np.any(inside_pusher):
                idx_p = np.where(inside_pusher)[0]
                # Push downward to pusher_bottom_z
                x_pred[idx_p, 2] = pusher_bottom_z
                # Give downward velocity
                self.v[idx_p, 2] = np.minimum(self.v[idx_p, 2], pusher_vel_z)

            # --- Collision with Ground ---
            # Ground at z = ground_z
            below_ground = x_pred[:, 2] < self.ground_z
            if np.any(below_ground):
                idx_g = np.where(below_ground)[0]
                x_pred[idx_g, 2] = self.ground_z
                # Zero / reflect velocity with restitution and friction
                if self.v[idx_g, 2].size > 0:
                    self.v[idx_g, 2] = np.maximum(-self.v[idx_g, 2] * self.restitution, 0.0)
                    self.v[idx_g, 0] *= (1.0 - self.friction)
                    self.v[idx_g, 1] *= (1.0 - self.friction)

        # Update velocity from position change
        new_v = (x_pred - self.x) / dt

        # Friction against plate contact
        # If vertex was in contact with plate rim, damp tangential velocity
        dr_new = np.hypot(x_pred[:, 0], x_pred[:, 1]) - R_rim
        dz_new = x_pred[:, 2] - Z_rim
        near_plate = np.hypot(dr_new, dz_new) <= r_tube + 0.01
        new_v[near_plate] *= (1.0 - self.friction * 0.5)

        self.v = new_v
        self.x = x_pred

    def advance_to(self, target_time: float) -> StateFrame:
        if not self.initialized:
            raise RuntimeError("BenchmarkSolver is not initialized.")

        # If target_time is current_time (e.g. initial frame t=0)
        if target_time > self.current_time:
            time_remaining = target_time - self.current_time
            while time_remaining > 1e-8:
                step_dt = min(self.sub_dt, time_remaining)
                self._step(step_dt, self.current_time)
                self.current_time += step_dt
                time_remaining = target_time - self.current_time

        self.current_time = target_time

        # Compute output fields
        F = self._compute_deformation_gradient()
        contact_dist = self._compute_contact_distance()
        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(target_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_verts, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext_force,
                "contact_signed_distance": contact_dist,
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_verts": self.num_verts,
            "current_time": self.current_time,
            "mean_z": float(np.mean(self.x[:, 2])),
        }

