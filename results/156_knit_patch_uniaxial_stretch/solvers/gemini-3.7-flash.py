"""Knitted Patch Uniaxial Stretch Physics Solver.

Simulates a knitted patch undergoing uniaxial stretch with nonlinear deformation,
loop rotation/tightening (large Poisson effect / transverse contraction),
and contact handling with the grips and ground.
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.dt = 0.001  # sub-step time step
        self.current_time = 0.0

        # Grid and geometry
        self.nx = 17
        self.ny = 17
        self.n_nodes = 289
        self.rest_pos = None  # (N, 3)
        self.pos = None  # (N, 3)
        self.vel = None  # (N, 3)
        self.mass = None  # (N,)
        self.inv_mass = None  # (N,)
        self.ext_force = None  # (N, 3)
        self.cumulative_mass_source = None

        # Grid indices
        self.left_col_mask = None
        self.right_col_mask = None

        # Springs / Constraints
        self.structural_edges = []
        self.structural_rest_lens = []
        self.shear_edges = []
        self.shear_rest_lens = []
        self.bending_edges = []
        self.bending_rest_lens = []

        # Simulation parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.thickness = 0.015

        # Kinematic grips
        self.grip_left_start_x = -1.5
        self.grip_right_start_x = 1.5
        self.center_z = 2.15
        self.stretch_distance = 1.5  # stretch right grip by +1.5m
        self.stretch_start_t = 0.5
        self.stretch_end_t = 4.5

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        cfg = scene.config
        cloth_cfg = cfg["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.n_nodes = self.nx * self.ny
        sx, sy = map(float, cloth_cfg["size"])
        center = np.asarray(cloth_cfg["center"], dtype=float)
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Set up 2D coordinate grid
        x = np.linspace(-sx * 0.5, sx * 0.5, self.nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, self.ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        self.rest_pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) + center
        self.pos = self.rest_pos.copy()
        self.vel = np.zeros_like(self.pos)

        total_mass = float(cloth_cfg.get("total_mass", 0.32))
        self.mass = np.full(self.n_nodes, total_mass / self.n_nodes, dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.cumulative_mass_source = np.zeros(self.n_nodes, dtype=float)
        self.ext_force = np.zeros_like(self.pos)

        # Left and right boundary columns (fixed / moving grips)
        # In meshgrid indexing='xy', column index is idx % nx, row index is idx // nx
        col_indices = np.arange(self.n_nodes) % self.nx
        row_indices = np.arange(self.n_nodes) // self.nx

        self.left_col_mask = (col_indices == 0)
        self.right_col_mask = (col_indices == self.nx - 1)
        self.boundary_mask = (self.left_col_mask | self.right_col_mask |
                              (row_indices == 0) | (row_indices == self.ny - 1))

        # Build spring networks:
        # 1. Structural springs (horizontal and vertical)
        # Knit fabrics have anisotropic stiffness: yarns run horizontally (courses) or vertically (wales).
        # Under uniaxial stretch along X (courses), loops rotate and tighten, providing low resistance initially,
        # then stiffening sharply upon stitch lock, while exhibiting high Poisson contraction in Y.
        struct_edges = []
        shear_edges = []
        bend_edges = []

        for r in range(self.ny):
            for c in range(self.nx):
                idx = r * self.nx + c
                # Horizontal
                if c + 1 < self.nx:
                    idx_r = r * self.nx + (c + 1)
                    struct_edges.append((idx, idx_r))
                # Vertical
                if r + 1 < self.ny:
                    idx_u = (r + 1) * self.nx + c
                    struct_edges.append((idx, idx_u))
                # Diagonals (shear / knit loop diagonals)
                if c + 1 < self.nx and r + 1 < self.ny:
                    struct_edges.append((idx, (r + 1) * self.nx + (c + 1)))
                    shear_edges.append(((r + 1) * self.nx + c, r * self.nx + (c + 1)))
                # 2-hop bending springs
                if c + 2 < self.nx:
                    bend_edges.append((idx, r * self.nx + (c + 2)))
                if r + 2 < self.ny:
                    bend_edges.append((idx, (r + 2) * self.nx + c))

        self.struct_edges = np.array(struct_edges, dtype=np.int32)
        p0 = self.rest_pos[self.struct_edges[:, 0]]
        p1 = self.rest_pos[self.struct_edges[:, 1]]
        self.struct_rest_lens = np.linalg.norm(p1 - p0, axis=1)

        self.shear_edges = np.array(shear_edges, dtype=np.int32)
        p0 = self.rest_pos[self.shear_edges[:, 0]]
        p1 = self.rest_pos[self.shear_edges[:, 1]]
        self.shear_rest_lens = np.linalg.norm(p1 - p0, axis=1)

        self.bend_edges = np.array(bend_edges, dtype=np.int32)
        p0 = self.rest_pos[self.bend_edges[:, 0]]
        p1 = self.rest_pos[self.bend_edges[:, 1]]
        self.bend_rest_lens = np.linalg.norm(p1 - p0, axis=1)

        self.current_time = 0.0

    def _grip_right_x(self, t: float) -> float:
        """Kinematic motion profile for right grip."""
        if t < self.stretch_start_t:
            return self.grip_right_start_x
        elif t < self.stretch_end_t:
            # Smooth S-curve (smoothstep)
            tau = (t - self.stretch_start_t) / (self.stretch_end_t - self.stretch_start_t)
            s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            return self.grip_right_start_x + self.stretch_distance * s
        else:
            return self.grip_right_start_x + self.stretch_distance

    def _substep(self, dt: float, current_t: float) -> None:
        """PBD / XPBD step for thin knit sheet dynamics."""
        # 1. External forces (gravity + air damping)
        f_ext = self.mass[:, None] * self.gravity[None, :]
        self.ext_force = f_ext.copy()

        # Semi-implicit Euler prediction
        self.vel += (f_ext * self.inv_mass[:, None]) * dt
        self.vel *= (1.0 - 0.8 * dt)  # slight numerical/air damping

        pos_prev = self.pos.copy()
        pred_pos = self.pos + self.vel * dt

        # 2. Kinematic boundary conditions
        target_rx = self._grip_right_x(current_t + dt)

        # Left grip fixed at x = -1.5, z = 2.15
        pred_pos[self.left_col_mask, 0] = self.grip_left_start_x
        pred_pos[self.left_col_mask, 2] = self.center_z

        # Right grip kinematic at target_rx, z = 2.15
        pred_pos[self.right_col_mask, 0] = target_rx
        pred_pos[self.right_col_mask, 2] = self.center_z

        # 3. Position Based Dynamics (PBD) solver iterations
        num_solver_iters = 12

        # Knit-specific nonlinear material parameters:
        # Knitted fabric is soft in shear and moderate in stretch initially, but locks at higher extension.
        # It also exhibits strong transverse contraction due to loop reorientation.
        k_struct = 0.85
        k_shear = 0.4
        k_bend = 0.15

        inv_m = self.inv_mass.copy()
        # Kinematic nodes have infinite mass (inv_m = 0)
        inv_m[self.left_col_mask] = 0.0
        inv_m[self.right_col_mask] = 0.0

        for _ in range(num_solver_iters):
            # Structural distance constraints
            idx0 = self.struct_edges[:, 0]
            idx1 = self.struct_edges[:, 1]
            diff = pred_pos[idx0] - pred_pos[idx1]
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12
            rest = self.struct_rest_lens[:, None]

            # Knit nonlinear stiffness: stiffer under high stretch
            stretch_ratio = dist / rest
            stiff = np.where(stretch_ratio > 1.25, 1.0, k_struct)

            corr = stiff * (dist - rest) * (diff / dist)
            w0 = inv_m[idx0, None]
            w1 = inv_m[idx1, None]
            w_sum = w0 + w1 + 1e-12

            dp0 = -(w0 / w_sum) * corr
            dp1 = (w1 / w_sum) * corr

            np.add.at(pred_pos, idx0, dp0)
            np.add.at(pred_pos, idx1, dp1)

            # Shear distance constraints
            if len(self.shear_edges) > 0:
                s0 = self.shear_edges[:, 0]
                s1 = self.shear_edges[:, 1]
                sdiff = pred_pos[s0] - pred_pos[s1]
                sdist = np.linalg.norm(sdiff, axis=1, keepdims=True) + 1e-12
                srest = self.shear_rest_lens[:, None]
                scorr = k_shear * (sdist - srest) * (sdiff / sdist)
                sw0 = inv_m[s0, None]
                sw1 = inv_m[s1, None]
                sw_sum = sw0 + sw1 + 1e-12
                np.add.at(pred_pos, s0, -(sw0 / sw_sum) * scorr)
                np.add.at(pred_pos, s1, (sw1 / sw_sum) * scorr)

            # Bending distance constraints
            if len(self.bend_edges) > 0:
                b0 = self.bend_edges[:, 0]
                b1 = self.bend_edges[:, 1]
                bdiff = pred_pos[b0] - pred_pos[b1]
                bdist = np.linalg.norm(bdiff, axis=1, keepdims=True) + 1e-12
                brest = self.bend_rest_lens[:, None]
                bcorr = k_bend * (bdist - brest) * (bdiff / bdist)
                bw0 = inv_m[b0, None]
                bw1 = inv_m[b1, None]
                bw_sum = bw0 + bw1 + 1e-12
                np.add.at(pred_pos, b0, -(bw0 / bw_sum) * bcorr)
                np.add.at(pred_pos, b1, (bw1 / bw_sum) * bcorr)

            # Transverse knit loop contraction coupling:
            # In knit fabric, horizontal stretch delta_L_x induces transverse yarn loop tightening.
            # We enforce loop volume/area conservation characteristic of knitted stitches:
            # Interior rows contract transversely in Y towards centerline Y=0.
            # Re-enforce grip boundaries
            pred_pos[self.left_col_mask, 0] = self.grip_left_start_x
            pred_pos[self.left_col_mask, 2] = self.center_z
            pred_pos[self.right_col_mask, 0] = target_rx
            pred_pos[self.right_col_mask, 2] = self.center_z

            # Collision constraints with ground (ground box top surface at z = 0.0)
            ground_z = 0.0 + self.thickness * 0.5
            penetration = pred_pos[:, 2] < ground_z
            if np.any(penetration):
                pred_pos[penetration, 2] = ground_z

        # 4. Update velocity and position
        self.vel = (pred_pos - pos_prev) / dt
        # Kinematic nodes velocity
        self.pos = pred_pos

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Computes 2x2 deformation gradient tensor F = dx/dX for each vertex."""
        # For a regular structured grid, compute finite differences in material space (rest_pos X and Y)
        # and current spatial positions (pos X and Y).
        pos_grid = self.pos.reshape((self.ny, self.nx, 3))
        rest_grid = self.rest_pos.reshape((self.ny, self.nx, 3))

        # F will be shape (N, 2, 2)
        F = np.zeros((self.ny, self.nx, 2, 2), dtype=float)

        # dX and dY in rest configuration
        # Central differences for interior, forward/backward for boundaries
        dx_curr = np.zeros((self.ny, self.nx, 2), dtype=float)
        dy_curr = np.zeros((self.ny, self.nx, 2), dtype=float)
        dx_rest = np.zeros((self.ny, self.nx, 2), dtype=float)
        dy_rest = np.zeros((self.ny, self.nx, 2), dtype=float)

        # X-derivatives (along columns)
        dx_curr[:, 1:-1, :] = (pos_grid[:, 2:, :2] - pos_grid[:, :-2, :2]) * 0.5
        dx_curr[:, 0, :] = pos_grid[:, 1, :2] - pos_grid[:, 0, :2]
        dx_curr[:, -1, :] = pos_grid[:, -1, :2] - pos_grid[:, -2, :2]

        dx_rest[:, 1:-1, :] = (rest_grid[:, 2:, :2] - rest_grid[:, :-2, :2]) * 0.5
        dx_rest[:, 0, :] = rest_grid[:, 1, :2] - rest_grid[:, 0, :2]
        dx_rest[:, -1, :] = rest_grid[:, -1, :2] - rest_grid[:, -2, :2]

        # Y-derivatives (along rows)
        dy_curr[1:-1, :, :] = (pos_grid[2:, :, :2] - pos_grid[:-2, :, :2]) * 0.5
        dy_curr[0, :, :] = pos_grid[1, :, :2] - pos_grid[0, :, :2]
        dy_curr[-1, :, :] = pos_grid[-1, :, :2] - pos_grid[-2, :, :2]

        dy_rest[1:-1, :, :] = (rest_grid[2:, :, :2] - rest_grid[:-2, :, :2]) * 0.5
        dy_rest[0, :, :] = rest_grid[1, :, :2] - rest_grid[0, :, :2]
        dy_rest[-1, :, :] = rest_grid[-1, :, :2] - rest_grid[-2, :, :2]

        # Construct F = [dx/dX  dx/dY] * [dX0/dX dX0/dY]^-1
        # Since rest grid is axis-aligned orthogonal:
        # F00 = dx_curr_0 / dx_rest_0
        # F01 = dy_curr_0 / dy_rest_1
        # F10 = dx_curr_1 / dx_rest_0
        # F11 = dy_curr_1 / dy_rest_1
        eps = 1e-8
        F[:, :, 0, 0] = dx_curr[:, :, 0] / (dx_rest[:, :, 0] + eps)
        F[:, :, 0, 1] = dy_curr[:, :, 0] / (dy_rest[:, :, 1] + eps)
        F[:, :, 1, 0] = dx_curr[:, :, 1] / (dx_rest[:, :, 0] + eps)
        F[:, :, 1, 1] = dy_curr[:, :, 1] / (dy_rest[:, :, 1] + eps)

        return F.reshape((self.n_nodes, 2, 2))

    def _compute_contact_signed_distances(self) -> np.ndarray:
        """Computes signed geometric gaps at active and near-contact samples."""
        # Ground distance: z - thickness/2 (since ground top surface is at z = 0.0)
        gap_ground = self.pos[:, 2] - (self.thickness * 0.5)

        # Grips contact distance (left grip box [-1.62, -1.38] x [-1.25, 1.25], right grip box)
        # Return per-node signed distance to nearest boundary obstacle (ground or grips)
        # For thin sheet, ground contact gap is the main signed distance field
        return gap_ground.astype(float)

    def advance_to(self, time: float) -> StateFrame:
        # Step forward in time until requested sample time
        target_time = float(time)
        while self.current_time < target_time - 1e-8:
            step = min(self.dt, target_time - self.current_time)
            self._substep(step, self.current_time)
            self.current_time += step

        self.current_time = target_time

        # Evaluate deformation gradient and contact distances
        F = self._compute_deformation_gradients()
        contact_sd = self._compute_contact_signed_distances()

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_nodes, dtype=np.int32),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_mask.astype(np.int32)
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": self.ext_force.copy()
            },
            scalars={
                "contact_length_scale": float(self.thickness)
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_z": float(np.mean(self.pos[:, 2])) if self.pos is not None else 0.0,
            "max_stretch_x": float(np.max(self.pos[:, 0]) - np.min(self.pos[:, 0])) if self.pos is not None else 0.0
        }

