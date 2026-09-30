"""Cloth Edge Point Contact Physics Solver.

Simulates square cloth falling onto a sharp edge (box obstacle) and a sharp point (cone obstacle).
Uses mass-spring / PBD cloth dynamics with stretch, shear, and bending constraints,
and exact geometric continuous/discrete collision resolution against the sharp edge, sharp cone,
and ground plane.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.positions = None
        self.rest_positions = None
        self.velocities = None
        self.mass = None
        self.inv_mass = None
        self.faces = None
        self.uv = None

        # Grid parameters
        self.nx = 41
        self.ny = 41
        self.thickness = 0.008
        self.cloth_thickness_offset = 0.008
        self.friction_coef = 0.35
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Obstacle geometries
        # Sharp edge (box)
        self.edge_center = np.array([-0.38, 0.0, 0.3], dtype=np.float64)
        self.edge_half = np.array([0.025, 0.68, 0.3], dtype=np.float64)

        # Sharp point (cone)
        self.point_tip = np.array([0.42, 0.0, 0.68], dtype=np.float64)
        self.point_height = 0.58
        self.point_base_radius = 0.24

        # Ground
        self.ground_center = np.array([0.0, 0.0, -0.04], dtype=np.float64)
        self.ground_half = np.array([1.35, 1.15, 0.04], dtype=np.float64)

        # Constraints
        self.edges_stretch = None
        self.edges_shear = None
        self.edges_bend = None

        # Simulation state
        self.current_time = 0.0
        self.dt_sub = 0.001  # sub-step time
        self.damping = 0.9995

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        init_frame = scene.initial_state()

        self.positions = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.rest_positions = self.positions.copy()
        self.velocities = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        self.faces = np.array(scene._faces, dtype=np.int32)
        self.uv = np.array(scene._uv, dtype=np.float64)

        cloth_cfg = scene.config.get("cloth", {})
        self.nx, self.ny = map(int, cloth_cfg.get("grid_shape", [41, 41]))
        self.thickness = float(cloth_cfg.get("thickness", 0.008))

        friction_cfg = scene.config.get("friction", {})
        self.friction_coef = float(friction_cfg.get("coefficient", 0.35))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        edge_cfg = scene.config.get("sharp_edge", {})
        self.edge_center = np.array(edge_cfg.get("center", [-0.38, 0.0, 0.3]), dtype=np.float64)
        self.edge_half = np.array(edge_cfg.get("half_extents", [0.025, 0.68, 0.3]), dtype=np.float64)

        point_cfg = scene.config.get("sharp_point", {})
        self.point_tip = np.array(point_cfg.get("tip", [0.42, 0.0, 0.68]), dtype=np.float64)
        self.point_height = float(point_cfg.get("height", 0.58))
        self.point_base_radius = float(point_cfg.get("base_radius", 0.24))

        ground_cfg = scene.config.get("ground", {})
        self.ground_center = np.array(ground_cfg.get("center", [0.0, 0.0, -0.04]), dtype=np.float64)
        self.ground_half = np.array(ground_cfg.get("half_extents", [1.35, 1.15, 0.04]), dtype=np.float64)

        contact_cfg = scene.config.get("contact", {})
        self.cloth_thickness_offset = float(contact_cfg.get("thickness_offset", 0.008))

        self._build_constraints()
        self.current_time = 0.0

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny

        stretch_list = []
        shear_list = []
        bend_list = []

        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                # Structural (stretch)
                if ix + 1 < nx:
                    stretch_list.append((idx, idx + 1))
                if iy + 1 < ny:
                    stretch_list.append((idx, idx + nx))

                # Shear (diagonal)
                if ix + 1 < nx and iy + 1 < ny:
                    stretch_list.append((idx, idx + nx + 1))
                if ix + 1 < nx and iy - 1 >= 0:
                    stretch_list.append((idx, idx - nx + 1))

                # Bending (distance 2)
                if ix + 2 < nx:
                    bend_list.append((idx, idx + 2))
                if iy + 2 < ny:
                    bend_list.append((idx, idx + 2 * nx))

        def make_edge_array(edge_pairs):
            if not edge_pairs:
                return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
            arr = np.array(edge_pairs, dtype=np.int32)
            p0 = self.rest_positions[arr[:, 0]]
            p1 = self.rest_positions[arr[:, 1]]
            dists = np.linalg.norm(p1 - p0, axis=-1)
            return arr, dists

        self.edges_stretch, self.rest_len_stretch = make_edge_array(stretch_list)
        self.edges_bend, self.rest_len_bend = make_edge_array(bend_list)

    def _solve_obstacle_collision(self, pos: np.ndarray, prev_pos: np.ndarray, dt: float) -> np.ndarray:
        """Resolve collisions with sharp edge (box), sharp point (cone), and ground."""
        r = self.thickness * 0.5 + self.cloth_thickness_offset * 0.5

        # 1. Ground collision (box / plane)
        # Ground top is at center[2] + half[2] = -0.04 + 0.04 = 0.0
        ground_top = self.ground_center[2] + self.ground_half[2]
        ground_min_x = self.ground_center[0] - self.ground_half[0]
        ground_max_x = self.ground_center[0] + self.ground_half[0]
        ground_min_y = self.ground_center[1] - self.ground_half[1]
        ground_max_y = self.ground_center[1] + self.ground_half[1]

        in_ground_xy = (
            (pos[:, 0] >= ground_min_x) & (pos[:, 0] <= ground_max_x) &
            (pos[:, 1] >= ground_min_y) & (pos[:, 1] <= ground_max_y)
        )
        penetrate_ground = in_ground_xy & (pos[:, 2] < ground_top + r)
        pos[penetrate_ground, 2] = ground_top + r

        # 2. Sharp edge collision (box obstacle)
        # Center: [-0.38, 0.0, 0.3], half: [0.025, 0.68, 0.3]
        # Box bounds: [-0.405, -0.68, 0.0] to [-0.355, 0.68, 0.6]
        b_min = self.edge_center - self.edge_half
        b_max = self.edge_center + self.edge_half

        # SDF for box obstacle with collision margin r
        # For each point, find closest point on box surface if penetrated or close
        bx_min, by_min, bz_min = b_min[0] - r, b_min[1] - r, b_min[2] - r
        bx_max, by_max, bz_max = b_max[0] + r, b_max[1] + r, b_max[2] + r

        inside_box = (
            (pos[:, 0] >= bx_min) & (pos[:, 0] <= bx_max) &
            (pos[:, 1] >= by_min) & (pos[:, 1] <= by_max) &
            (pos[:, 2] >= bz_min) & (pos[:, 2] <= bz_max)
        )

        if np.any(inside_box):
            idx = np.where(inside_box)[0]
            p = pos[idx]

            # Distances to each face
            d_left = p[:, 0] - bx_min
            d_right = bx_max - p[:, 0]
            d_front = p[:, 1] - by_min
            d_back = by_max - p[:, 1]
            d_bottom = p[:, 2] - bz_min
            d_top = bz_max - p[:, 2]

            dists = np.stack([d_left, d_right, d_front, d_back, d_bottom, d_top], axis=-1)
            min_face = np.argmin(dists, axis=-1)

            # Project to the closest face with friction
            for i, f in enumerate(min_face):
                pt = p[i]
                old_pt = prev_pos[idx[i]]
                if f == 0:
                    pt[0] = bx_min
                elif f == 1:
                    pt[0] = bx_max
                elif f == 2:
                    pt[1] = by_min
                elif f == 3:
                    pt[1] = by_max
                elif f == 4:
                    pt[2] = bz_min
                elif f == 5:
                    pt[2] = bz_max

                # Apply friction
                tangent_disp = pt - old_pt
                if f in (0, 1):
                    pt[1] = old_pt[1] + (1.0 - self.friction_coef) * tangent_disp[1]
                    pt[2] = old_pt[2] + (1.0 - self.friction_coef) * tangent_disp[2]
                elif f in (2, 3):
                    pt[0] = old_pt[0] + (1.0 - self.friction_coef) * tangent_disp[0]
                    pt[2] = old_pt[2] + (1.0 - self.friction_coef) * tangent_disp[2]
                elif f in (4, 5):
                    pt[0] = old_pt[0] + (1.0 - self.friction_coef) * tangent_disp[0]
                    pt[1] = old_pt[1] + (1.0 - self.friction_coef) * tangent_disp[1]
                pos[idx[i]] = pt

        # 3. Sharp point collision (cone obstacle)
        # Cone tip at point_tip = [0.42, 0.0, 0.68], height H = 0.58, base_radius R = 0.24
        # Base is at z = tip_z - H = 0.68 - 0.58 = 0.10.
        # Inside cone: for z in [tip_z - H, tip_z], local_z = tip_z - z >= 0.
        # Radius at height: r_cone(z) = local_z * (R / H).
        tip = self.point_tip
        H = self.point_height
        R = self.point_base_radius
        slope = R / H

        # Transform points to local coordinates relative to tip
        rel = pos - tip
        # Cone extends downwards along -Z
        # local z_down = -rel[:, 2]
        z_down = -rel[:, 2]

        # Consider points in the vertical range of the cone plus margin
        in_cone_z = (z_down >= -r) & (z_down <= H + r)
        if np.any(in_cone_z):
            idx = np.where(in_cone_z)[0]
            p_rel = rel[idx]
            zd = z_down[idx]

            # Horizontal distance from cone axis
            rho = np.linalg.norm(p_rel[:, :2], axis=-1)

            # Expected radius with margin
            # Normal to cone surface points outward: normal = (cos_alpha * er, sin_alpha * ez_up)
            # where tan_alpha = slope = R / H.
            cos_cone = H / np.sqrt(H * H + R * R)
            sin_cone = R / np.sqrt(H * H + R * R)

            # Distance to cone surface:
            # Cone surface equation: rho = zd * slope (for 0 <= zd <= H)
            # Signed distance approximately: d = (rho - zd * slope) * cos_cone
            # Tip sphere margin around tip (zd < 0):
            dist_to_tip = np.linalg.norm(p_rel, axis=-1)

            for k in range(len(idx)):
                i_vert = idx[k]
                pt_rel = p_rel[k]
                curr_zd = zd[k]
                curr_rho = rho[k]

                # Check collision with cone tip (sphere of radius r at tip)
                if curr_zd < 0:
                    d_tip = dist_to_tip[k]
                    if d_tip < r:
                        if d_tip > 1e-7:
                            n = pt_rel / d_tip
                        else:
                            n = np.array([0.0, 0.0, 1.0])
                        pos[i_vert] = tip + n * r
                        # Friction at tip
                        old_pt = prev_pos[i_vert]
                        pos[i_vert][:2] = old_pt[:2] + (1.0 - self.friction_coef) * (pos[i_vert][:2] - old_pt[:2])
                elif curr_zd <= H:
                    # Along cone body
                    r_target = curr_zd * slope + r / cos_cone
                    if curr_rho < r_target:
                        if curr_rho > 1e-7:
                            n_xy = pt_rel[:2] / curr_rho
                            # Move outward along surface normal
                            # Normal vector in 3D: (cos_cone * n_xy, sin_cone)
                            # Push point along normal
                            push_dist = (r_target - curr_rho) * cos_cone
                            pos[i_vert][:2] = tip[:2] + n_xy * (curr_rho + push_dist * cos_cone)
                            pos[i_vert][2] = tip[2] - curr_zd + push_dist * sin_cone
                        else:
                            # Directly on axis, push in arbitrary x direction
                            pos[i_vert][0] = tip[0] + r_target

                        # Friction
                        old_pt = prev_pos[i_vert]
                        pos[i_vert] = old_pt + (1.0 - self.friction_coef) * (pos[i_vert] - old_pt)

        return pos

    def _step(self, dt: float) -> None:
        """Advance physical simulation by dt using PBD / Verlet."""
        # 1. Apply gravity & external forces
        self.velocities += self.gravity * dt
        self.velocities *= self.damping

        prev_positions = self.positions.copy()
        pred_positions = self.positions + self.velocities * dt

        # 2. PBD constraint solver iterations
        num_iters = 12
        edges = self.edges_stretch
        rest_len = self.rest_len_stretch

        edges_bend = self.edges_bend
        rest_bend = self.rest_len_bend

        for _ in range(num_iters):
            # Distance constraints (stretch & shear)
            if len(edges) > 0:
                p0 = pred_positions[edges[:, 0]]
                p1 = pred_positions[edges[:, 1]]
                delta = p1 - p0
                dists = np.linalg.norm(delta, axis=-1, keepdims=True)
                dists = np.maximum(dists, 1e-8)
                err = (dists - rest_len[:, None]) / dists
                corr = 0.5 * 0.85 * delta * err

                # Accumulate corrections
                np.add.at(pred_positions, edges[:, 0], corr)
                np.add.at(pred_positions, edges[:, 1], -corr)

            # Bending constraints
            if len(edges_bend) > 0:
                p0 = pred_positions[edges_bend[:, 0]]
                p1 = pred_positions[edges_bend[:, 1]]
                delta = p1 - p0
                dists = np.linalg.norm(delta, axis=-1, keepdims=True)
                dists = np.maximum(dists, 1e-8)
                err = (dists - rest_bend[:, None]) / dists
                corr = 0.5 * 0.4 * delta * err

                np.add.at(pred_positions, edges_bend[:, 0], corr)
                np.add.at(pred_positions, edges_bend[:, 1], -corr)

            # Obstacle collisions inside loop
            pred_positions = self._solve_obstacle_collision(pred_positions, prev_positions, dt)

        # 3. Update velocity and positions
        self.velocities = (pred_positions - prev_positions) / dt
        self.positions = pred_positions

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward with sub-stepping
        while self.current_time < target_time - 1e-7:
            step_dt = min(self.dt_sub, target_time - self.current_time)
            self._step(step_dt)
            self.current_time += step_dt

        self.current_time = target_time

        # Calculate StateFrame trajectory fields
        N = len(self.positions)

        # 1. Deformation gradient: approximate per vertex
        # Compute local stretch tensor or gradient relative to rest positions
        # Rest grid spacing
        cloth_side = 1.8
        dx = cloth_side / (self.nx - 1)
        dy = cloth_side / (self.ny - 1)

        pos_grid = self.positions.reshape((self.ny, self.nx, 3))
        def_grad = np.tile(np.eye(3), (N, 1, 1))

        # Finite difference for deformation gradient
        # dX/du, dX/dv
        dpos_du = np.zeros_like(pos_grid)
        dpos_dv = np.zeros_like(pos_grid)

        dpos_du[:, 1:-1] = (pos_grid[:, 2:] - pos_grid[:, :-2]) / (2.0 * dx)
        dpos_du[:, 0] = (pos_grid[:, 1] - pos_grid[:, 0]) / dx
        dpos_du[:, -1] = (pos_grid[:, -1] - pos_grid[:, -2]) / dx

        dpos_dv[1:-1, :] = (pos_grid[2:, :] - pos_grid[:-2, :]) / (2.0 * dy)
        dpos_dv[0, :] = (pos_grid[1, :] - pos_grid[0, :]) / dy
        dpos_dv[-1, :] = (pos_grid[-1, :] - pos_grid[-2, :]) / dy

        # Normal
        normals = np.cross(dpos_du, dpos_dv)
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.where(norm_len > 1e-8, normals / np.maximum(norm_len, 1e-8), np.array([0.0, 0.0, 1.0]))

        F_flat = def_grad.copy()
        dpos_du_flat = dpos_du.reshape((N, 3))
        dpos_dv_flat = dpos_dv.reshape((N, 3))
        normals_flat = normals.reshape((N, 3))

        F_flat[:, :, 0] = dpos_du_flat
        F_flat[:, :, 1] = dpos_dv_flat
        F_flat[:, :, 2] = normals_flat

        # 2. Contact signed distance
        # Minimum signed distance to ground, edge box, and sharp point cone
        r = self.thickness * 0.5 + self.cloth_thickness_offset * 0.5

        # Ground distance
        ground_top = self.ground_center[2] + self.ground_half[2]
        d_ground = self.positions[:, 2] - ground_top

        # Edge box distance (approximate box signed distance)
        b_min = self.edge_center - self.edge_half
        b_max = self.edge_center + self.edge_half
        d_box_max = np.maximum(b_min - self.positions, self.positions - b_max)
        d_box_out = np.linalg.norm(np.maximum(d_box_max, 0.0), axis=-1)
        d_box_in = np.minimum(np.max(d_box_max, axis=-1), 0.0)
        d_edge = d_box_out + d_box_in

        # Point cone distance
        rel_cone = self.positions - self.point_tip
        zd = -rel_cone[:, 2]
        rho = np.linalg.norm(rel_cone[:, :2], axis=-1)
        slope = self.point_base_radius / self.point_height
        cos_cone = self.point_height / np.sqrt(self.point_height**2 + self.point_base_radius**2)
        d_cone_body = (rho - zd * slope) * cos_cone
        d_cone_tip = np.linalg.norm(rel_cone, axis=-1)
        d_cone = np.where(zd < 0, d_cone_tip, d_cone_body)

        contact_signed_dist = np.minimum(np.minimum(d_ground, d_edge), d_cone)

        frame = StateFrame(
            target_time,
            self.positions.copy(),
            self.velocities.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F_flat,
            },
            fields={
                "contact_signed_distance": contact_signed_dist,
                "cumulative_mass_source": np.asarray([0.0], dtype=np.float64),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.positions) if self.positions is not None else 0,
        }

