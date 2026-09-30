"""Physics solver for 161_pinned_denim_gravity_sag.

Simulates inextensible cloth sag under gravity with pinned upper edge/corners.
Uses Position Based Dynamics (PBD) / Extended PBD with stretch, shear, and bending constraints
plus ground contact handling to model realistic heavy denim drape.
"""
from __future__ import annotations

import json
import os
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.position = None
        self.velocity = None
        self.mass = None
        self.inv_mass = None
        self.fixed_mask = None
        self.edges = None
        self.rest_lengths = None
        self.faces = None
        self.bend_edges = None
        self.bend_rest_lengths = None
        self.total_mass = 0.32
        self.thickness = 0.015
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.external_force = None
        self.ground_z = 0.0  # Top of ground box: z_center (-0.05) + half_extent (0.05) = 0.0
        self.substeps = 25
        self.pbd_iterations = 35
        self.damping = 0.998
        self.rest_positions_2d = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.position = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.total_mass = float(scene.config.get("cloth", {}).get("total_mass", 0.32))
        self.thickness = float(scene.config.get("cloth", {}).get("thickness", 0.015))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Ground level from scene_geometry: box at z=-0.05, half_extents z=0.05 -> top at 0.0
        self.ground_z = 0.0

        cloth_cfg = scene.config.get("cloth", {})
        grid_shape = cloth_cfg.get("grid_shape", [17, 17])
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        N = len(self.position)

        # In scene.py:
        # x = np.linspace(-sx * .5, sx * .5, nx)
        # y = np.linspace(-sy * .5, sy * .5, ny)
        # xx, yy = np.meshgrid(x, y, indexing="xy")
        # position = [xx.ravel(), 0, yy.ravel()] + center
        # The top row corresponds to yy == y[-1] (maximum z coordinates).
        # Pin the top edge (or top corners/edge). Pinned denim drape: top row fixed.
        # Top row in xx, yy with meshgrid(x, y):
        # row index j from 0 to ny-1, column index i from 0 to nx-1
        # index = j * nx + i
        # Top row is j = ny - 1.
        self.fixed_mask = np.zeros(N, dtype=bool)
        top_row_indices = np.arange((ny - 1) * nx, ny * nx)
        self.fixed_mask[top_row_indices] = True

        # Compute inverse masses (0 for fixed vertices)
        self.inv_mass = np.where(self.fixed_mask, 0.0, 1.0 / self.mass)

        # Load topology faces if available, otherwise generate grid topology
        topo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "cloth_topology.json")
        if os.path.exists(topo_path):
            with open(topo_path, "r") as f:
                topo_data = json.load(f)
            self.faces = np.array(topo_data["faces"], dtype=np.int32)
        else:
            faces = []
            for j in range(ny - 1):
                for i in range(nx - 1):
                    p0 = j * nx + i
                    p1 = j * nx + (i + 1)
                    p2 = (j + 1) * nx + (i + 1)
                    p3 = (j + 1) * nx + i
                    faces.append([p0, p1, p2])
                    faces.append([p0, p2, p3])
            self.faces = np.array(faces, dtype=np.int32)

        # Extract structural and shear edges from faces
        edge_set = set()
        for face in self.faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            edge_set.add(tuple(sorted((i0, i1))))
            edge_set.add(tuple(sorted((i1, i2))))
            edge_set.add(tuple(sorted((i2, i0))))

        edges_list = list(edge_set)
        self.edges = np.array(edges_list, dtype=np.int32)
        diff = self.position[self.edges[:, 0]] - self.position[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(diff, axis=1)

        # Extract bending / cross-springs (2-hop grid neighbors)
        bend_set = set()
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i + 2 < nx:
                    bend_set.add(tuple(sorted((idx, j * nx + (i + 2)))))
                if j + 2 < ny:
                    bend_set.add(tuple(sorted((idx, (j + 2) * nx + i))))
                if i + 1 < nx and j + 1 < ny:
                    bend_set.add(tuple(sorted((idx, (j + 1) * nx + (i + 1)))))
                if i - 1 >= 0 and j + 1 < ny:
                    bend_set.add(tuple(sorted((idx, (j + 1) * nx + (i - 1)))))

        # Remove edges already in structural edge_set
        bend_list = [e for e in bend_set if e not in edge_set]
        if len(bend_list) > 0:
            self.bend_edges = np.array(bend_list, dtype=np.int32)
            b_diff = self.position[self.bend_edges[:, 0]] - self.position[self.bend_edges[:, 1]]
            self.bend_rest_lengths = np.linalg.norm(b_diff, axis=1)
        else:
            self.bend_edges = np.empty((0, 2), dtype=np.int32)
            self.bend_rest_lengths = np.empty(0, dtype=np.float64)

        # 2D Rest positions for deformation gradient calculation
        sx, sy = float(cloth_cfg.get("size", [3.0, 2.5])[0]), float(cloth_cfg.get("size", [3.0, 2.5])[1])
        x = np.linspace(-sx * 0.5, sx * 0.5, nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.rest_positions_2d = np.column_stack((xx.ravel(), yy.ravel()))

        # Add tiny seed-based lateral disturbance to break symmetry on free vertices during motion
        self.lateral_perturbation = np.sin(np.pi * (xx.ravel() - x[0]) / (x[-1] - x[0])) * 0.003
        self.lateral_perturbation[self.fixed_mask] = 0.0
        self.first_step = True

    def _step(self, dt: float) -> None:
        if self.first_step:
            self.position[:, 1] += self.lateral_perturbation
            self.first_step = False

        sub_dt = dt / self.substeps
        N = len(self.position)

        for _ in range(self.substeps):
            # External forces: gravity
            ext_acc = self.gravity

            # Semi-implicit Euler prediction
            # Apply acceleration only to free vertices
            free_idx = ~self.fixed_mask
            self.velocity[free_idx] += ext_acc * sub_dt
            self.velocity[free_idx] *= self.damping

            pos_pred = self.position.copy()
            pos_pred[free_idx] += self.velocity[free_idx] * sub_dt
            # Enforce fixed pins
            pos_pred[self.fixed_mask] = self.position[self.fixed_mask]

            # PBD Constraint projection
            inv_m = self.inv_mass
            edges_0 = self.edges[:, 0]
            edges_1 = self.edges[:, 1]
            inv_m0 = inv_m[edges_0]
            inv_m1 = inv_m[edges_1]
            w_sum = inv_m0 + inv_m1
            valid_edges = w_sum > 1e-12

            e0_v = edges_0[valid_edges]
            e1_v = edges_1[valid_edges]
            w_sum_v = w_sum[valid_edges]
            inv_m0_v = inv_m0[valid_edges]
            inv_m1_v = inv_m1[valid_edges]
            rest_len_v = self.rest_lengths[valid_edges]

            has_bend = len(self.bend_edges) > 0
            if has_bend:
                b_edges_0 = self.bend_edges[:, 0]
                b_edges_1 = self.bend_edges[:, 1]
                b_inv_m0 = inv_m[b_edges_0]
                b_inv_m1 = inv_m[b_edges_1]
                b_w_sum = b_inv_m0 + b_inv_m1
                b_valid = b_w_sum > 1e-12
                b0_v = b_edges_0[b_valid]
                b1_v = b_edges_1[b_valid]
                b_w_sum_v = b_w_sum[b_valid]
                b_inv_m0_v = b_inv_m0[b_valid]
                b_inv_m1_v = b_inv_m1[b_valid]
                b_rest_len_v = self.bend_rest_lengths[b_valid]

            for _ in range(self.pbd_iterations):
                # Structural / shear edge constraints (high stiffness for inextensible denim)
                delta = pos_pred[e0_v] - pos_pred[e1_v]
                cur_len = np.linalg.norm(delta, axis=1, keepdims=True)
                cur_len_safe = np.maximum(cur_len, 1e-9)
                corr = (cur_len - rest_len_v[:, None]) * (delta / cur_len_safe)

                # Vectorized scatter add corrections
                c0 = corr * (inv_m0_v[:, None] / w_sum_v[:, None])
                c1 = corr * (inv_m1_v[:, None] / w_sum_v[:, None])

                np.add.at(pos_pred, e0_v, -c0)
                np.add.at(pos_pred, e1_v, c1)

                # Bending constraints (moderate stiffness)
                if has_bend:
                    b_delta = pos_pred[b0_v] - pos_pred[b1_v]
                    b_cur_len = np.linalg.norm(b_delta, axis=1, keepdims=True)
                    b_cur_len_safe = np.maximum(b_cur_len, 1e-9)
                    b_corr = 0.3 * (b_cur_len - b_rest_len_v[:, None]) * (b_delta / b_cur_len_safe)

                    bc0 = b_corr * (b_inv_m0_v[:, None] / b_w_sum_v[:, None])
                    bc1 = b_corr * (b_inv_m1_v[:, None] / b_w_sum_v[:, None])

                    np.add.at(pos_pred, b0_v, -bc0)
                    np.add.at(pos_pred, b1_v, bc1)

                # Enforce fixed pins
                pos_pred[self.fixed_mask] = self.position[self.fixed_mask]

                # Ground contact constraint (floor at z = ground_z + thickness * 0.5)
                min_z = self.ground_z + self.thickness * 0.5
                pos_pred[:, 2] = np.maximum(pos_pred[:, 2], min_z)

            # Update velocity and position
            self.velocity = (pos_pred - self.position) / sub_dt
            self.velocity[self.fixed_mask] = 0.0
            self.position = pos_pred

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute per-vertex 2x2 deformation gradient tensor in surface material coordinates."""
        cloth_cfg = self.scene.config.get("cloth", {})
        grid_shape = cloth_cfg.get("grid_shape", [17, 17])
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        N = len(self.position)

        F = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], N, axis=0)

        # Estimate tangent basis vectors in current 3D deformation vs 2D rest configuration
        # For structured grid: dX/du, dX/dv
        pos_grid = self.position.reshape((ny, nx, 3))
        rest_grid = self.rest_positions_2d.reshape((ny, nx, 2))

        # Forward / backward / central difference in 2D material coords
        du = np.zeros((ny, nx, 3), dtype=np.float64)
        dv = np.zeros((ny, nx, 3), dtype=np.float64)

        # Along x (u): columns
        du[:, 1:-1] = (pos_grid[:, 2:] - pos_grid[:, :-2]) * 0.5
        du[:, 0] = pos_grid[:, 1] - pos_grid[:, 0]
        du[:, -1] = pos_grid[:, -1] - pos_grid[:, -2]

        # Along y (v): rows
        dv[1:-1, :] = (pos_grid[2:, :] - pos_grid[:-2, :]) * 0.5
        dv[0, :] = pos_grid[1, :] - pos_grid[0, :]
        dv[-1, :] = pos_grid[-1, :] - pos_grid[-2, :]

        # Rest metric
        du_rest = np.zeros((ny, nx, 2), dtype=np.float64)
        dv_rest = np.zeros((ny, nx, 2), dtype=np.float64)
        du_rest[:, 1:-1] = (rest_grid[:, 2:] - rest_grid[:, :-2]) * 0.5
        du_rest[:, 0] = rest_grid[:, 1] - rest_grid[:, 0]
        du_rest[:, -1] = rest_grid[:, -1] - rest_grid[:, -2]

        dv_rest[1:-1, :] = (rest_grid[2:, :] - rest_grid[:-2, :]) * 0.5
        dv_rest[0, :] = rest_grid[1, :] - rest_grid[0, :]
        dv_rest[-1, :] = rest_grid[-1, :] - rest_grid[-2, :]

        # Approximate stretch ratio in u and v directions
        len_du_rest = np.linalg.norm(du_rest, axis=-1, keepdims=True)
        len_dv_rest = np.linalg.norm(dv_rest, axis=-1, keepdims=True)
        len_du = np.linalg.norm(du, axis=-1, keepdims=True)
        len_dv = np.linalg.norm(dv, axis=-1, keepdims=True)

        lambda_u = (len_du / np.maximum(len_du_rest, 1e-9)).reshape(N)
        lambda_v = (len_dv / np.maximum(len_dv_rest, 1e-9)).reshape(N)

        F[:, 0, 0] = lambda_u
        F[:, 1, 1] = lambda_v

        return F

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        dt = time - self.current_time
        if dt > 1e-9:
            self._step(dt)
            self.current_time = time

        N = len(self.position)

        # External force: gravity force on cloth particles = mass * gravity
        external_force = self.mass[:, None] * self.gravity[None, :]

        # Signed distance to ground contact: gap = z - ground_top_z - half_thickness
        contact_signed_distance = self.position[:, 2] - self.ground_z - 0.5 * self.thickness

        # Boundary vertex attribute
        cloth_cfg = self.scene.config.get("cloth", {})
        grid_shape = cloth_cfg.get("grid_shape", [17, 17])
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        boundary = np.zeros(N, dtype=np.int32)
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary[j * nx + i] = 1

        deformation_gradient = self._compute_deformation_gradient()

        frame = StateFrame(
            float(self.current_time),
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": deformation_gradient,
            },
            fields={
                "cumulative_mass_source": np.zeros(N, dtype=np.float64),
                "external_force": external_force,
                "contact_signed_distance": contact_signed_distance,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.position) if self.position is not None else 0,
        }

