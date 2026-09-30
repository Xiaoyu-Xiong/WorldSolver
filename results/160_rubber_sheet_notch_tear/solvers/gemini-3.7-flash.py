"""Benchmark solver for rubber sheet notch tear simulation."""
from __future__ import annotations

import json
import os
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        cloth_cfg = self.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.sx, self.sy = map(float, cloth_cfg["size"])
        self.center = np.asarray(cloth_cfg["center"], dtype=float)
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.total_mass = float(cloth_cfg.get("total_mass", 0.32))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=float).copy() # [N, 3]
        self.vel = np.zeros_like(self.pos) # [N, 3]
        self.N = len(self.pos)
        self.mass = np.array(init_frame.mass, dtype=float).copy() # [N]
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-8)

        # Material 2D reference coordinates (in XZ plane centered at center)
        # x in [-sx/2, sx/2], z in [-sy/2, sy/2]
        self.ref_pos_2d = np.zeros((self.N, 2), dtype=float)
        self.ref_pos_2d[:, 0] = self.pos[:, 0] - self.center[0]
        self.ref_pos_2d[:, 1] = self.pos[:, 2] - self.center[2]

        # Mesh topology
        topology_path = os.path.join(os.path.dirname(__file__), "..", "assets", "cloth_topology.json")
        if os.path.exists(topology_path):
            with open(topology_path, "r") as f:
                top_data = json.load(f)
            self.faces = np.array(top_data["faces"], dtype=int)
        else:
            # Fallback standard grid triangulation
            faces_list = []
            for j in range(self.ny - 1):
                for i in range(self.nx - 1):
                    p0 = j * self.nx + i
                    p1 = j * self.nx + (i + 1)
                    p2 = (j + 1) * self.nx + i
                    p3 = (j + 1) * self.nx + (i + 1)
                    faces_list.append([p0, p1, p3])
                    faces_list.append([p0, p3, p2])
            self.faces = np.array(faces_list, dtype=int)

        # Extract unique edges from faces
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))
        self.edges = np.array(list(edges_set), dtype=int)

        # Reference edge lengths
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.edge_rest_len = np.linalg.norm(p0 - p1, axis=1)

        # Edge active mask (can tear if strained past threshold)
        self.edge_active = np.ones(len(self.edges), dtype=bool)

        # Identify boundary grips (left and right columns)
        # Grid indexing: grid_shape is [17, 17], meshgrid(x, y, indexing="xy")
        # x changes fast along row (nx=17), y changes along column (ny=17)
        # vertex index = j * nx + i
        # Left boundary: i == 0; Right boundary: i == nx - 1
        x_coords = self.ref_pos_2d[:, 0]
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        self.is_left_grip = np.isclose(x_coords, min_x, atol=1e-4)
        self.is_right_grip = np.isclose(x_coords, max_x, atol=1e-4)
        self.is_grip = self.is_left_grip | self.is_right_grip

        # Grip kinematics parameters
        self.grip_speed = 0.25 # Separation speed in X direction (m/s)

        # Initial grip positions
        self.grip_left_x0 = self.pos[self.is_left_grip, 0].copy()
        self.grip_right_x0 = self.pos[self.is_right_grip, 0].copy()

        # Material properties
        self.stiffness = 2500.0  # Spring stiffness
        self.damping = 4.0       # Velocity damping
        self.tear_threshold = 1.35 # Strain ratio (current_len / rest_len) to tear an edge

        # Internal simulation clock
        self.current_time = 0.0
        self.sub_dt = 0.0005 # 0.5 ms sub-stepping for stability

        # Stored deformation gradients for output
        self.F_per_vertex = np.repeat(np.eye(2)[None, :, :], self.N, axis=0)

    def _step(self, dt: float) -> None:
        # 1. Update kinematic grip positions and fix their velocities
        # Tensile loading: left grip moves in -X, right grip moves in +X
        disp = self.grip_speed * self.current_time
        self.pos[self.is_left_grip, 0] = self.grip_left_x0 - disp
        self.pos[self.is_right_grip, 0] = self.grip_right_x0 + disp
        self.vel[self.is_left_grip] = np.array([-self.grip_speed, 0.0, 0.0])
        self.vel[self.is_right_grip] = np.array([self.grip_speed, 0.0, 0.0])

        # 2. Gravity and external forces
        ext_forces = self.mass[:, None] * self.gravity[None, :]

        # 3. Internal spring forces
        active_mask = self.edge_active
        active_edges = self.edges[active_mask]
        active_rest_len = self.edge_rest_len[active_mask]

        if len(active_edges) > 0:
            idx0 = active_edges[:, 0]
            idx1 = active_edges[:, 1]
            dp = self.pos[idx1] - self.pos[idx0]
            dist = np.linalg.norm(dp, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            dir_vec = dp / dist_safe

            # Strain check for tearing (progressive crack propagation)
            strain_ratio = (dist[:, 0] / np.maximum(active_rest_len, 1e-8))
            # Don't tear grip boundary edges
            edge_is_grip = self.is_grip[idx0] & self.is_grip[idx1]
            tear_cond = (strain_ratio > self.tear_threshold) & (~edge_is_grip)
            if np.any(tear_cond):
                active_indices = np.where(active_mask)[0]
                self.edge_active[active_indices[tear_cond]] = False

            # Spring force: Hookean with damping
            spring_force_mag = self.stiffness * (dist - active_rest_len[:, None])
            # Relative velocity along edge
            dv = self.vel[idx1] - self.vel[idx0]
            damp_force_mag = 0.5 * self.damping * np.sum(dv * dir_vec, axis=1, keepdims=True)

            f_spring = (spring_force_mag + damp_force_mag) * dir_vec

            # Accumulate internal forces onto nodes
            np.add.at(ext_forces, idx0, f_spring)
            np.add.at(ext_forces, idx1, -f_spring)

        # Global aerodynamic / material damping
        ext_forces -= 0.5 * self.damping * self.vel

        # 4. Integrate unconstrained dynamic vertices
        dynamic_mask = ~self.is_grip
        acc = ext_forces[dynamic_mask] * self.inv_mass[dynamic_mask, None]
        self.vel[dynamic_mask] += acc * dt
        self.pos[dynamic_mask] += self.vel[dynamic_mask] * dt

        # Ground collision (ground top surface at z = 0.0)
        ground_z = 0.0
        penetration = ground_z + self.thickness * 0.5 - self.pos[:, 2]
        contact_mask = penetration > 0.0
        if np.any(contact_mask):
            self.pos[contact_mask, 2] = ground_z + self.thickness * 0.5
            self.vel[contact_mask, 2] = np.maximum(0.0, self.vel[contact_mask, 2])
            # Ground friction
            self.vel[contact_mask, 0] *= 0.9
            self.vel[contact_mask, 1] *= 0.9

        self.current_time += dt

    def _compute_deformation_gradients(self) -> np.ndarray:
        # Approximate 2D in-plane deformation gradient F = dx / dX at each vertex
        F = np.repeat(np.eye(2)[None, :, :], self.N, axis=0)

        # Per active face deformation gradient
        # Reference coordinates in (X, Z) plane
        for f in self.faces:
            # Check if all edges in face are intact
            i0, i1, i2 = f[0], f[1], f[2]
            X0 = self.ref_pos_2d[i0]
            X1 = self.ref_pos_2d[i1]
            X2 = self.ref_pos_2d[i2]

            # Current in-plane coordinates (x, z)
            x0 = np.array([self.pos[i0, 0] - self.center[0], self.pos[i0, 2] - self.center[2]])
            x1 = np.array([self.pos[i1, 0] - self.center[0], self.pos[i1, 2] - self.center[2]])
            x2 = np.array([self.pos[i2, 0] - self.center[0], self.pos[i2, 2] - self.center[2]])

            dX = np.column_stack((X1 - X0, X2 - X0))
            dx = np.column_stack((x1 - x0, x2 - x0))

            detX = np.linalg.det(dX)
            if abs(detX) > 1e-10:
                F_elem = dx @ np.linalg.inv(dX)
                # Scatter to vertices
                for v in f:
                    F[v] = 0.5 * (F[v] + F_elem)

        return F

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation in sub-steps to reach target time
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)

        self.current_time = time

        # Compute deformation gradients
        F = self._compute_deformation_gradients()

        # Signed distance to ground (ground top surface at z = 0)
        # gap = z - thickness/2 - 0.0
        gap = self.pos[:, 2] - self.thickness * 0.5

        # External forces (gravity on dynamic cloth)
        ext_f = self.mass[:, None] * self.gravity[None, :]

        # Boundary vertex indicator
        boundary = self.is_grip.astype(np.int32)

        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(self.N, dtype=float),
                "external_force": ext_f,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        active_edge_count = int(np.sum(self.edge_active))
        return {
            "time": float(self.current_time),
            "active_edges": active_edge_count,
            "total_edges": len(self.edges),
        }

