"""Mass-spring cloth solver with contact projection for the edge/point scene.

This is a lightweight, physically real baseline: explicit time integration of a
mass-spring network, in-plane membrane/stretch forces, bending resistance, and
repulsive/projective collision handling against the box, sharp edge, sharp
cone, and the ground.  It does not use Genesis, keyframes, or precomputed
animation.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Public platform interface
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically-based cloth solver for the 031_cloth_edge_point_contact scene."""

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed

        # Copy configuration from the scene
        config = scene.config
        self.config = config
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])

        # Physics constants
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.friction_coeff = float(config["friction"]["coefficient"])
        self.contact_offset = float(config["contact"]["thickness_offset"])

        # Cloth parameters
        cloth_cfg = config["cloth"]
        self.cloth_name = cloth_cfg["name"]
        self.thickness = float(cloth_cfg["thickness"])
        self.side_length = float(cloth_cfg["side_length"])
        self.grid_shape = tuple(map(int, cloth_cfg["grid_shape"]))

        # Scene object geometry
        self.edge_center = np.asarray(config["sharp_edge"]["center"], dtype=float)
        self.edge_half = np.asarray(config["sharp_edge"]["half_extents"], dtype=float)
        self.point_tip = np.asarray(config["sharp_point"]["tip"], dtype=float)
        self.point_height = float(config["sharp_point"]["height"])
        self.point_radius = float(config["sharp_point"]["base_radius"])
        self.ground_center = np.asarray(config["ground"]["center"], dtype=float)
        self.ground_half = np.asarray(config["ground"]["half_extents"], dtype=float)

        # Initial state from the scene (positions, faces, uv)
        initial_frame = scene.initial_state()
        self.positions = np.asarray(initial_frame.position, dtype=float).copy()
        self.velocities = np.asarray(initial_frame.velocity, dtype=float).copy()
        self.mass = np.asarray(initial_frame.mass, dtype=float).copy()
        self.uv = np.asarray(initial_frame.attributes["uv"], dtype=float).copy()
        self.faces = np.asarray(scene._faces, dtype=np.int32)

        self.n_vertices = self.positions.shape[0]
        self.n_faces = len(self.faces)

        # Reference positions in the cloth plane (2D material coordinates)
        self.ref_positions = self.positions.copy()

        # Build edge graph from the mesh for spring forces
        edges = set()
        for f in self.faces:
            edges.add((min(f[0], f[1]), max(f[0], f[1])))
            edges.add((min(f[1], f[2]), max(f[1], f[2])))
            edges.add((min(f[2], f[0]), max(f[2], f[0])))
        self.edges = np.asarray(list(edges), dtype=np.int32)
        self.rest_lengths = np.linalg.norm(
            self.positions[self.edges[:, 0]] - self.positions[self.edges[:, 1]], axis=1
        )

        # Bending edges: shared interior edges between two triangles
        self.bend_edges: list[tuple[int, int, int, int]] = []
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi, f in enumerate(self.faces):
            e0 = (min(f[0], f[1]), max(f[0], f[1]))
            e1 = (min(f[1], f[2]), max(f[1], f[2]))
            e2 = (min(f[2], f[0]), max(f[2], f[0]))
            for e in (e0, e1, e2):
                edge_to_faces.setdefault(e, []).append(fi)
        for (i, j), fcs in edge_to_faces.items():
            if len(fcs) == 2:
                f0, f1 = fcs
                # Opposite vertices
                opp0 = [v for v in self.faces[f0] if v not in (i, j)][0]
                opp1 = [v for v in self.faces[f1] if v not in (i, j)][0]
                self.bend_edges.append((i, j, opp0, opp1))
        self.bend_edges = np.asarray(self.bend_edges, dtype=np.int32)
        if self.bend_edges.size == 0:
            self.bend_edges = np.zeros((0, 4), dtype=np.int32)

        # Per-vertex adjacent faces for deformation gradient averaging
        self.vertex_faces: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for fi, f in enumerate(self.faces):
            for v in f:
                self.vertex_faces[v].append(fi)

        # Time integration
        self.time = 0.0
        self._dt = self.sample_dt / 10.0  # internal sub-step (~1.67 ms)
        if self._dt <= 0.0:
            self._dt = self.sample_dt
        self._cached_frame: StateFrame | None = None

    def _deformation_gradient_per_face(self) -> NDArray[np.float64]:
        """Compute a 3x3 deformation gradient for each triangle."""
        F = np.zeros((self.n_faces, 3, 3), dtype=float)
        for idx, f in enumerate(self.faces):
            p0 = self.positions[f[0]]
            p1 = self.positions[f[1]]
            p2 = self.positions[f[2]]
            e1 = p1 - p0
            e2 = p2 - p0
            r0 = self.ref_positions[f[0]]
            r1 = self.ref_positions[f[1]]
            r2 = self.ref_positions[f[2]]
            re1 = r1 - r0
            re2 = r2 - r0
            X = np.column_stack([re1[:2], re2[:2]])
            x = np.column_stack([e1[:2], e2[:2]])
            det = np.linalg.det(X)
            if det > 1e-12:
                F2d = x @ np.linalg.inv(X)
            else:
                F2d = np.eye(2)
            Fmat = np.eye(3)
            Fmat[:2, :2] = F2d
            F[idx] = Fmat
        return F

    def _state_frame(self) -> StateFrame:
        """Build a StateFrame from the current simulation state."""
        face_F = self._deformation_gradient_per_face()
        deformation_gradient = np.zeros((self.n_vertices, 3, 3), dtype=float)
        for v in range(self.n_vertices):
            faces = self.vertex_faces[v]
            if faces:
                deformation_gradient[v] = np.mean(face_F[faces], axis=0)
            else:
                deformation_gradient[v] = np.eye(3)

        # Contact signed distance: minimum gap to any obstacle, positive outside.
        distances = self._signed_distance(self.positions)

        return StateFrame(
            time=self.time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation_gradient,
            },
            fields={
                "contact_signed_distance": distances,
                "cumulative_mass_source": np.asarray([0.0]),
            },
            scalars={
                "contact_length_scale": self.thickness,
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("Cannot advance backwards in time")
        if time == self.time and self._cached_frame is not None:
            return self._cached_frame

        # Sub-step forward until we reach the requested sample time
        target = float(time)
        while self.time < target - 1e-12:
            step = min(self._dt, target - self.time)
            self._step(step)
            self.time += step

        self._cached_frame = self._state_frame()
        return self._cached_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "solver": "mass_spring_cloth_contact_projection",
            "internal_dt": self._dt,
            "vertices": self.n_vertices,
            "faces": len(self.faces),
        }

    # --------------------------------------------------------------------- #
    # Internal physics
    # --------------------------------------------------------------------- #
    def _step(self, dt: float) -> None:
        # Compute forces
        forces = self._compute_forces()

        # Symplectic Euler integration
        self.velocities += dt * forces / self.mass[:, None]
        self.positions += dt * self.velocities

        # Collision projection / friction
        self._handle_collisions(dt)

    def _compute_forces(self) -> NDArray[np.float64]:
        forces = np.zeros_like(self.positions)

        # Gravity
        forces += self.mass[:, None] * self.gravity

        # Global air/velocity damping (light)
        forces -= 0.15 * self.velocities

        # In-plane membrane / stretch forces via edge springs
        p0 = self.positions[self.edges[:, 0]]
        p1 = self.positions[self.edges[:, 1]]
        diff = p1 - p0
        lengths = np.linalg.norm(diff, axis=1)
        lengths = np.where(lengths < 1e-12, 1e-12, lengths)
        strain = (lengths - self.rest_lengths) / self.rest_lengths
        k = 2.0e4  # stretch stiffness, N/m
        magnitudes = k * strain
        forces_per_edge = magnitudes[:, None] * (diff / lengths[:, None])

        np.add.at(forces, self.edges[:, 0], forces_per_edge)
        np.add.at(forces, self.edges[:, 1], -forces_per_edge)

        # Damping proportional to relative velocity along edges
        v0 = self.velocities[self.edges[:, 0]]
        v1 = self.velocities[self.edges[:, 1]]
        rel_v = v1 - v0
        rel_dir = diff / lengths[:, None]
        dot = np.einsum("ij,ij->i", rel_v, rel_dir)
        damping = 0.5 * dot
        damp_force = damping[:, None] * rel_dir
        np.add.at(forces, self.edges[:, 0], damp_force)
        np.add.at(forces, self.edges[:, 1], -damp_force)

        # Bending resistance (opposite-vertex springs)
        if len(self.bend_edges):
            i, j, k, l = self.bend_edges.T
            pk = self.positions[k]
            pl = self.positions[l]
            diff_b = pk - pl
            f_b = 2.0e2 * diff_b  # weak bend spring
            np.add.at(forces, k, f_b)
            np.add.at(forces, l, -f_b)

        return forces

    def _signed_distance_to_box(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed distance to an axis-aligned box (negative inside)."""
        local = p - self.edge_center
        d = np.abs(local) - self.edge_half
        inside = np.all(d <= 0, axis=1)
        outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside_dist = np.min(-d, axis=1)
        dist = np.where(inside, inside_dist, outside_dist)
        return dist

    def _signed_distance_to_ground(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        local = p - self.ground_center
        xy_d = np.abs(local[:, :2]) - self.ground_half[:2]
        inside_xy = np.all(xy_d <= 0, axis=1)
        z_top = self.ground_center[2] + self.ground_half[2]
        dist = p[:, 2] - z_top
        outside = ~inside_xy
        d_xy = np.maximum(xy_d, 0.0)
        dist_out = np.sqrt(d_xy[:, 0] ** 2 + d_xy[:, 1] ** 2 + dist ** 2)
        dist = np.where(outside, dist_out, dist)
        return dist

    def _signed_distance_to_cone(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed distance to the sharp cone (negative inside)."""
        local = p - self.point_tip
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        h = self.point_height
        r = self.point_radius
        rho = np.sqrt(x * x + y * y)
        t = -z / h
        t = np.clip(t, 0.0, 1.0)
        z_surf = -h * t
        r_surf = r * (1.0 - t)
        dz = z - z_surf
        d_rho = rho - r_surf
        dist = np.sqrt(d_rho ** 2 + dz ** 2)
        inside = (z <= 0.0) & (z >= -h) & (rho < r_surf)
        dist = np.where(inside, -dist, dist)
        return dist

    def _signed_distance(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Minimum signed distance over all collision obstacles."""
        d_edge = self._signed_distance_to_box(p)
        d_ground = self._signed_distance_to_ground(p)
        d_cone = self._signed_distance_to_cone(p)
        d = np.minimum(np.minimum(d_edge, d_ground), d_cone)
        return d

    def _handle_collisions(self, dt: float) -> None:
        # Box (sharp edge)
        self._project_box(self.edge_center, self.edge_half)
        # Ground
        self._project_ground()
        # Cone
        self._project_cone()

    def _project_box(self, center: NDArray[np.float64], half: NDArray[np.float64]) -> None:
        local = self.positions - center
        inside = np.all(np.abs(local) < half, axis=1)
        if not np.any(inside):
            return
        d_pos = half - local
        d_neg = local + half
        push_pos = np.min(d_pos, axis=1)
        push_neg = np.min(d_neg, axis=1)
        axis_pos = np.argmin(d_pos, axis=1)
        axis_neg = np.argmin(d_neg, axis=1)
        axis = np.where(push_pos < push_neg, axis_pos, axis_neg)
        sign = np.where(push_pos < push_neg, 1.0, -1.0)

        for i in np.where(inside)[0]:
            ax = int(axis[i])
            if sign[i] > 0.0:
                target = half[ax]
            else:
                target = -half[ax]
            self.positions[i, ax] = center[ax] + target + self.contact_offset * sign[i]
            # Remove normal velocity
            self.velocities[i, ax] = 0.0

    def _project_ground(self) -> None:
        z_top = self.ground_center[2] + self.ground_half[2]
        penetration = z_top + self.contact_offset - self.positions[:, 2]
        contact = penetration > 0.0
        if not np.any(contact):
            return
        self.positions[contact, 2] += penetration[contact]
        v = self.velocities[contact]
        v[:, 2] = 0.0
        self.velocities[contact, :2] *= np.exp(-self.friction_coeff * 0.5)

    def _project_cone(self) -> None:
        local = self.positions - self.point_tip
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        h = self.point_height
        r = self.point_radius
        rho = np.sqrt(x * x + y * y)
        mask = (z <= 0.0) & (z >= -h)
        if not np.any(mask):
            return
        t = -z / h
        r_surf = r * (1.0 - t)
        inside = mask & (rho < r_surf + self.contact_offset)
        if not np.any(inside):
            return
        rad_dir = np.zeros((self.n_vertices, 3), dtype=float)
        rad_dir[inside, :2] = local[inside, :2] / (rho[inside, None] + 1e-12)
        pen = (r_surf[inside] + self.contact_offset - rho[inside]) / (1.0 + (r / h) ** 2)
        pen = np.maximum(pen, 0.0)
        self.positions[inside] += rad_dir[inside] * pen[:, None]
        v_local = self.velocities[inside]
        radial = np.einsum("ij,ij->i", v_local, rad_dir[inside])
        self.velocities[inside] -= (radial[:, None] * rad_dir[inside])
