"""Cloth Drape over a Cylinder Solver using Position-Based Dynamics (XPBD).

Recreates the cloth drape over a horizontal cylinder (Baraff & Witkin 1998 / Figure 1).
"""
from __future__ import annotations
import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Get initial state from scene
        init_frame = scene.initial_state()
        self.pos = init_frame.position.copy()
        self.vel = np.zeros_like(self.pos)
        self.num_vertices = len(self.pos)
        self.mass = init_frame.mass.copy()
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-8)

        # Cloth configuration
        cloth_cfg = self.config["cloth"]
        self.grid_shape = list(map(int, cloth_cfg["grid_shape"]))  # [17, 17]
        self.nx, self.ny = self.grid_shape
        self.thickness = float(cloth_cfg["thickness"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Cylinder parameters from scene_geometry.yaml:
        # center: [0, 0, 0.7], radius: 0.52, axis along Y (rotation [0.7071068, 0, 0, 0.7071068])
        # In this rotation, cylinder Z-axis is rotated 90 deg around X into Y-axis.
        self.cyl_center = np.array([0.0, 0.0, 0.7], dtype=float)
        self.cyl_radius = 0.52
        self.cyl_half_height = 2.8 / 2.0  # along Y: -1.4 to 1.4

        # Ground plane at z = 0.0 (box half extents [2.4, 2.0, 0.05] at z=-0.05 => top face at z=0.0)
        self.ground_z = 0.0

        # Build distance and bending constraints for PBD
        self._build_constraints()

        # Build 2D rest coordinates for deformation gradient computation
        # Mesh is nx x ny in xy plane
        sx, sy = map(float, cloth_cfg["size"])
        x = np.linspace(-sx * 0.5, sx * 0.5, self.nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, self.ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.rest_coords = np.column_stack((xx.ravel(), yy.ravel()))  # [N, 2]

        # Precompute neighbors for finite difference deformation gradient
        self._build_diff_indices()

        # Simulation parameters
        self.current_time = 0.0
        self.sub_dt = 0.001  # 1 ms sub-stepping for XPBD stability
        self.pbd_iterations = 10
        self.damping = 0.998  # slight damping per sub-step to settle

        # For contact distance output
        self.contact_signed_dist = np.full(self.num_vertices, 1.0)
        self.external_force = np.zeros_like(self.pos)

    def _build_constraints(self) -> None:
        """Build structural, shear, and bending springs."""
        edges = []
        bending = []

        def vid(i, j):
            return j * self.nx + i

        for j in range(self.ny):
            for i in range(self.nx):
                # Structural horizontal
                if i + 1 < self.nx:
                    edges.append((vid(i, j), vid(i + 1, j)))
                # Structural vertical
                if j + 1 < self.ny:
                    edges.append((vid(i, j), vid(i, j + 1)))
                # Shear diagonals
                if i + 1 < self.nx and j + 1 < self.ny:
                    edges.append((vid(i, j), vid(i + 1, j + 1)))
                    edges.append((vid(i + 1, j), vid(i, j + 1)))
                # Bending horizontal (skip 1)
                if i + 2 < self.nx:
                    bending.append((vid(i, j), vid(i + 2, j)))
                # Bending vertical (skip 1)
                if j + 2 < self.ny:
                    bending.append((vid(i, j), vid(i, j + 2)))

        self.edges = np.array(edges, dtype=np.int32)
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.edge_rest_lens = np.linalg.norm(p0 - p1, axis=1)

        self.bending = np.array(bending, dtype=np.int32)
        if len(self.bending) > 0:
            b0 = self.pos[self.bending[:, 0]]
            b1 = self.pos[self.bending[:, 1]]
            self.bending_rest_lens = np.linalg.norm(b0 - b1, axis=1)
        else:
            self.bending_rest_lens = np.array([], dtype=float)

    def _build_diff_indices(self) -> None:
        """Precompute stencil indices for deformation gradient."""
        # For each vertex (i, j), get neighbor right and neighbor top
        self.idx_center = np.arange(self.num_vertices, dtype=np.int32)
        self.idx_right = np.zeros(self.num_vertices, dtype=np.int32)
        self.idx_left = np.zeros(self.num_vertices, dtype=np.int32)
        self.idx_top = np.zeros(self.num_vertices, dtype=np.int32)
        self.idx_bottom = np.zeros(self.num_vertices, dtype=np.int32)

        for j in range(self.ny):
            for i in range(self.nx):
                v = j * self.nx + i
                self.idx_right[v] = j * self.nx + min(i + 1, self.nx - 1)
                self.idx_left[v] = j * self.nx + max(i - 1, 0)
                self.idx_top[v] = min(j + 1, self.ny - 1) * self.nx + i
                self.idx_bottom[v] = max(j - 1, 0) * self.nx + i

    def _step(self, dt: float) -> None:
        """Perform one sub-step of XPBD."""
        # 1. External forces (gravity)
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.vel += dt * self.gravity[None, :]
        self.vel *= (self.damping ** (dt / 0.001))

        # 2. Predict positions
        p_pred = self.pos + self.vel * dt

        # Effective contact radius: cylinder radius + cloth half-thickness
        r_eff = self.cyl_radius + self.thickness * 0.5
        cx = self.cyl_center[0]
        cz = self.cyl_center[2]

        # 3. Solver iterations (XPBD / PBD)
        alpha_stretch = 1e-6 / (dt * dt)  # XPBD compliance for stretch
        alpha_bend = 1e-3 / (dt * dt)     # XPBD compliance for bend

        for _ in range(self.pbd_iterations):
            # Distance constraints (stretch & shear)
            v0 = self.edges[:, 0]
            v1 = self.edges[:, 1]
            p0 = p_pred[v0]
            p1 = p_pred[v1]
            diff = p0 - p1
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-7)
            rest = self.edge_rest_lens[:, None]
            c = dist - rest
            w0 = self.inv_mass[v0, None]
            w1 = self.inv_mass[v1, None]
            w_sum = w0 + w1 + alpha_stretch
            n = diff / dist
            delta = (c / w_sum) * n
            # Accumulate corrections
            np.add.at(p_pred, v0, -w0 * delta)
            np.add.at(p_pred, v1, w1 * delta)

            # Bending constraints
            if len(self.bending) > 0:
                bv0 = self.bending[:, 0]
                bv1 = self.bending[:, 1]
                bp0 = p_pred[bv0]
                bp1 = p_pred[bv1]
                bdiff = bp0 - bp1
                bdist = np.linalg.norm(bdiff, axis=1, keepdims=True)
                bdist = np.maximum(bdist, 1e-7)
                brest = self.bending_rest_lens[:, None]
                bc = bdist - brest
                bw0 = self.inv_mass[bv0, None]
                bw1 = self.inv_mass[bv1, None]
                bw_sum = bw0 + bw1 + alpha_bend
                bn = bdiff / bdist
                bdelta = (bc / bw_sum) * bn
                np.add.at(p_pred, bv0, -bw0 * bdelta)
                np.add.at(p_pred, bv1, bw1 * bdelta)

            # Collision constraints with horizontal cylinder along Y-axis
            # Cylinder center (cx, y, cz), radius r_eff
            # Distance in XZ plane:
            dx = p_pred[:, 0] - cx
            dz = p_pred[:, 2] - cz
            dist_xz = np.sqrt(dx * dx + dz * dz)
            dist_xz_safe = np.maximum(dist_xz, 1e-7)

            # Check penetration into cylinder
            # Note: cylinder extends in Y from -cyl_half_height to +cyl_half_height
            in_y_range = (p_pred[:, 1] >= self.cyl_center[1] - self.cyl_half_height) & \
                         (p_pred[:, 1] <= self.cyl_center[1] + self.cyl_half_height)
            penetration = dist_xz < r_eff
            colliding = penetration & in_y_range

            if np.any(colliding):
                overlap = r_eff - dist_xz[colliding]
                nx_coll = dx[colliding] / dist_xz_safe[colliding]
                nz_coll = dz[colliding] / dist_xz_safe[colliding]
                p_pred[colliding, 0] += nx_coll * overlap
                p_pred[colliding, 2] += nz_coll * overlap

            # Ground collision: z >= ground_z + thickness * 0.5
            ground_penetration = p_pred[:, 2] < (self.ground_z + self.thickness * 0.5)
            if np.any(ground_penetration):
                p_pred[ground_penetration, 2] = self.ground_z + self.thickness * 0.5

        # 4. Update velocities and positions
        new_vel = (p_pred - self.pos) / dt

        # Apply friction to colliding vertices
        dx = p_pred[:, 0] - cx
        dz = p_pred[:, 2] - cz
        dist_xz = np.sqrt(dx * dx + dz * dz)
        on_cylinder = (dist_xz <= r_eff + 1e-4) & \
                      (p_pred[:, 1] >= self.cyl_center[1] - self.cyl_half_height) & \
                      (p_pred[:, 1] <= self.cyl_center[1] + self.cyl_half_height)
        new_vel[on_cylinder] *= 0.85  # friction on cylinder

        self.vel = new_vel
        self.pos = p_pred

        # Compute signed distance for all vertices to cylinder & ground
        # Cylinder signed distance (gap: positive outside, negative inside)
        gap_cyl = dist_xz - r_eff
        gap_ground = self.pos[:, 2] - (self.ground_z + self.thickness * 0.5)
        self.contact_signed_dist = np.minimum(gap_cyl, gap_ground)

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute per-vertex 2x2 deformation gradient in material coordinates."""
        # F = dx / dX
        # In rest space, dx_ref = rest_coords[right] - rest_coords[left], dy_ref = rest_coords[top] - rest_coords[bottom]
        # In current 3D space, find the local 2D tangent plane deformation or 2D projected deformation
        # StateFrame requires attributes.deformation_gradient to have shape [N, 2, 2] or [N, D, D] (D=2)
        dX = self.rest_coords[self.idx_right] - self.rest_coords[self.idx_left]
        dY = self.rest_coords[self.idx_top] - self.rest_coords[self.idx_bottom]

        # Current 3D difference
        dx_3d = self.pos[self.idx_right] - self.pos[self.idx_left]
        dy_3d = self.pos[self.idx_top] - self.pos[self.idx_bottom]

        # Rest step lengths
        norm_dX = np.maximum(np.linalg.norm(dX, axis=1, keepdims=True), 1e-7)
        norm_dY = np.maximum(np.linalg.norm(dY, axis=1, keepdims=True), 1e-7)

        # Strains along principal directions
        stretch_u = np.linalg.norm(dx_3d, axis=1) / norm_dX.ravel()
        stretch_v = np.linalg.norm(dy_3d, axis=1) / norm_dY.ravel()

        # Build 2x2 deformation gradient F for each vertex
        # Diagonal approximation: diag(stretch_u, stretch_v)
        F = np.zeros((self.num_vertices, 2, 2), dtype=float)
        F[:, 0, 0] = stretch_u
        F[:, 1, 1] = stretch_v

        # Off-diagonal shear
        cos_theta = np.sum(dx_3d * dy_3d, axis=1) / np.maximum(
            np.linalg.norm(dx_3d, axis=1) * np.linalg.norm(dy_3d, axis=1), 1e-7
        )
        F[:, 0, 1] = cos_theta * 0.5
        F[:, 1, 0] = cos_theta * 0.5
        return F

    def advance_to(self, time: float) -> StateFrame:
        """Advance the simulation to the requested timestamp and return StateFrame."""
        # Perform sub-stepping up to target time
        while self.current_time < time - 1e-8:
            dt = min(self.sub_dt, time - self.current_time)
            if dt < 1e-9:
                break
            self._step(dt)
            self.current_time += dt

        self.current_time = time

        # Compute deformation gradients
        F = self._compute_deformation_gradients()

        # Boundary vertices
        cloth_cfg = self.config["cloth"]
        sx, sy = map(float, cloth_cfg["size"])
        x = np.linspace(-sx * 0.5, sx * 0.5, self.nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, self.ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        boundary = ((np.isclose(xx, x[0])) | (np.isclose(xx, x[-1])) |
                    (np.isclose(yy, y[0])) | (np.isclose(yy, y[-1]))).ravel().astype(np.int32)

        frame = StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": self.contact_signed_dist.copy(),
                "cumulative_mass_source": np.zeros(self.num_vertices, dtype=float),
                "external_force": self.external_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_vertices,
            "current_time": self.current_time,
        }

