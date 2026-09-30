"""Mass-spring cloth solver with position-based stretch constraints.

Implements a lightweight physically-based solver for a paper-like sheet that is
compressed between two inward-moving walls.  The solver uses an explicit
integration of gravity, damping, bending and self-contact forces, followed by
position-based projection of in-plane stretch constraints and collision
projection against the kinematic walls and ground.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for the paper-sheet crumpling benchmark."""

    def __init__(self) -> None:
        pass

    def initialize(self, scene: Any, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.scene = scene
        self.config = scene.config

        # Physical parameters from the frozen configuration.
        self.gravity = np.asarray(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        cloth_cfg = self.config["cloth"]
        self.thickness = float(cloth_cfg["thickness"])
        self.total_mass = float(cloth_cfg["total_mass"])
        self.nx = int(cloth_cfg["grid_shape"][0])
        self.ny = int(cloth_cfg["grid_shape"][1])

        # Initial state from the scaffold.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.N = self.position.shape[0]
        self.vertex_mass = self.total_mass / self.N

        # Deterministic geometric imperfection to break symmetry and seed buckling.
        # This is a tiny out-of-plane displacement that still leaves the sheet
        # ``mostly flat'' as required by the task description.
        imperfection = 0.002 * (
            np.sin(2.0 * self.position[:, 0])
            * np.cos(3.0 * self.position[:, 1])
        )
        self.position[:, 2] += imperfection

        # Build cloth mesh, edges, bending hinges and adjacency data.
        self._build_mesh()

        # Kinematic wall motion.
        self.duration = float(self.config["duration"])
        self.wall_half_x = 0.12
        self.wall_left_start = -1.55
        self.wall_left_end = -0.30
        self.wall_right_start = 1.55
        self.wall_right_end = 0.30

        # Sub-stepping and numerical parameters.
        self.max_dt = 1.0 / 1200.0
        self.pbd_iterations = 3
        self.damping_coeff = 0.5
        self.bending_stiffness = 0.05
        self.contact_stiffness = 20.0
        self.contact_radius = 2.0 * self.thickness

        self.current_time = 0.0

    def _build_mesh(self) -> None:
        nx, ny = self.nx, self.ny

        # Triangulate the regular grid (x varies fastest, matching scene.py).
        self.triangles: list[tuple[int, int, int]] = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v00 = i * ny + j
                v10 = (i + 1) * ny + j
                v01 = i * ny + (j + 1)
                v11 = (i + 1) * ny + (j + 1)
                self.triangles.append((v00, v10, v11))
                self.triangles.append((v00, v11, v01))

        # Unique edges and rest lengths.
        self.edges: list[tuple[int, int]] = []
        self.edge_rest: list[float] = []
        seen_edges: set[tuple[int, int]] = set()
        for (i, j, k) in self.triangles:
            for a, b in ((i, j), (j, k), (k, i)):
                key = tuple(sorted((a, b)))
                if key not in seen_edges:
                    seen_edges.add(key)
                    self.edges.append(key)
                    self.edge_rest.append(
                        float(np.linalg.norm(self.position[a] - self.position[b]))
                    )

        # Bending hinges: for each interior edge, the two opposite vertices.
        self.bending_hinges: list[tuple[int, int, int, int]] = []
        edge_tris: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for ti, (i, j, k) in enumerate(self.triangles):
            for a, b in ((i, j), (j, k), (k, i)):
                key = tuple(sorted((a, b)))
                edge_tris.setdefault(key, []).append((ti, a, b))
        for key, tris in edge_tris.items():
            if len(tris) == 2:
                (t1, a1, b1), (t2, a2, b2) = tris
                shared = {a1, b1} & {a2, b2}
                if len(shared) != 2:
                    continue
                a, b = sorted(shared)
                tri1 = set(self.triangles[t1])
                tri2 = set(self.triangles[t2])
                opp_c = (tri1 - {a, b}).pop()
                opp_d = (tri2 - {a, b}).pop()
                self.bending_hinges.append((a, b, opp_c, opp_d))

        # Vertex adjacency for self-contact exclusion.
        self.adjacent: list[set[int]] = [set() for _ in range(self.N)]
        for a, b in self.edges:
            self.adjacent[a].add(b)
            self.adjacent[b].add(a)

        self.adjacency_matrix = np.zeros((self.N, self.N), dtype=bool)
        for a in range(self.N):
            for b in self.adjacent[a]:
                self.adjacency_matrix[a, b] = True

        # Rest material coordinates are the initial in-plane (x, y) positions.
        self.rest_xy = self.position[:, :2].copy()

    # --------------------------------------------------------------------- #
    # Kinematic wall helpers
    # --------------------------------------------------------------------- #
    def _wall_centers(self, t: float) -> tuple[float, float]:
        s = t / self.duration
        if s <= 0.0:
            s = 0.0
        elif s >= 1.0:
            s = 1.0
        # Smooth step for gentle start/stop.
        s = s * s * (3.0 - 2.0 * s)
        left = self.wall_left_start + (self.wall_left_end - self.wall_left_start) * s
        right = self.wall_right_start + (self.wall_right_end - self.wall_right_start) * s
        return left, right

    # --------------------------------------------------------------------- #
    # Force computation
    # --------------------------------------------------------------------- #
    def _compute_forces(self) -> np.ndarray:
        forces = np.zeros_like(self.position)

        # Gravity.
        forces += self.vertex_mass * self.gravity

        # Velocity damping.
        forces -= self.damping_coeff * self.velocity

        # Bending resistance using a dihedral-angle spring approximation.
        p = self.position
        for a, b, c, d in self.bending_hinges:
            pa, pb, pc, pd = p[a], p[b], p[c], p[d]
            n1 = np.cross(pb - pa, pc - pa)
            n2 = np.cross(pd - pa, pb - pa)
            n1_norm = float(np.linalg.norm(n1))
            n2_norm = float(np.linalg.norm(n2))
            if n1_norm < 1e-12 or n2_norm < 1e-12:
                continue
            n1u = n1 / n1_norm
            n2u = n2 / n2_norm
            cos_theta = float(np.clip(np.dot(n1u, n2u), -1.0, 1.0))
            theta = math.acos(cos_theta)
            f_mag = self.bending_stiffness * theta
            f_c = -f_mag * n1u
            f_d = -f_mag * n2u
            f_a = -0.5 * (f_c + f_d)
            f_b = f_a
            forces[a] += f_a
            forces[b] += f_b
            forces[c] += f_c
            forces[d] += f_d

        # Self-contact repulsion.
        forces = self._add_self_contact_forces(forces)
        return forces

    def _add_self_contact_forces(self, forces: np.ndarray) -> np.ndarray:
        if self.contact_radius <= 0.0 or self.contact_stiffness <= 0.0:
            return forces
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return forces

        tree = cKDTree(self.position)
        pairs = tree.query_pairs(r=self.contact_radius, output_type="ndarray")
        if pairs.size == 0:
            return forces

        i = pairs[:, 0]
        j = pairs[:, 1]
        mask = ~self.adjacency_matrix[i, j]
        i = i[mask]
        j = j[mask]
        if i.size == 0:
            return forces

        dx = self.position[j] - self.position[i]
        dist = np.linalg.norm(dx, axis=1)
        valid = (dist < self.contact_radius) & (dist > 1e-12)
        i = i[valid]
        j = j[valid]
        dx = dx[valid]
        dist = dist[valid]
        if i.size == 0:
            return forces

        n = dx / dist[:, None]
        f_mag = self.contact_stiffness * (self.contact_radius - dist)
        f = f_mag[:, None] * n
        np.add.at(forces, i, f)
        np.add.at(forces, j, -f)
        return forces

    # --------------------------------------------------------------------- #
    # Constraints
    # --------------------------------------------------------------------- #
    def _project_stretch(self) -> None:
        for _ in range(self.pbd_iterations):
            for (i, j), rest_len in zip(self.edges, self.edge_rest):
                dx = self.position[i] - self.position[j]
                dist = float(np.linalg.norm(dx))
                if dist < 1e-12:
                    continue
                correction = 0.5 * (1.0 - rest_len / dist) * dx
                self.position[i] -= correction
                self.position[j] += correction

    def _project_walls_ground(self, left_c: float, right_c: float) -> None:
        left_inner = left_c + self.wall_half_x
        right_inner = right_c - self.wall_half_x
        for i in range(self.N):
            # Left wall: keep cloth to the right of the wall's inner face.
            if self.position[i, 0] < left_inner:
                self.position[i, 0] = left_inner
                if self.velocity[i, 0] < 0.0:
                    self.velocity[i, 0] = 0.0
            # Right wall: keep cloth to the left of the wall's inner face.
            if self.position[i, 0] > right_inner:
                self.position[i, 0] = right_inner
                if self.velocity[i, 0] > 0.0:
                    self.velocity[i, 0] = 0.0
            # Ground plane at z = 0.
            if self.position[i, 2] < 0.0:
                self.position[i, 2] = 0.0
                if self.velocity[i, 2] < 0.0:
                    self.velocity[i, 2] = -0.2 * self.velocity[i, 2]

    # --------------------------------------------------------------------- #
    # Trajectory field helpers
    # --------------------------------------------------------------------- #
    def _deformation_gradient(self) -> np.ndarray:
        F = np.zeros((self.N, 2, 2), dtype=float)
        p = self.position
        r = self.rest_xy
        for i, j, k in self.triangles:
            e1 = r[j] - r[i]
            e2 = r[k] - r[i]
            A = np.column_stack((e1, e2))
            try:
                inv_A = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            E1 = p[j, :2] - p[i, :2]
            E2 = p[k, :2] - p[i, :2]
            B = np.column_stack((E1, E2))
            F_tri = B @ inv_A
            F[i] += F_tri
            F[j] += F_tri
            F[k] += F_tri
        counts = np.zeros(self.N, dtype=int)
        for i, j, k in self.triangles:
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        for v in range(self.N):
            if counts[v] > 0:
                F[v] /= counts[v]
        return F

    def _contact_signed_distance(self, left_c: float, right_c: float) -> np.ndarray:
        left_inner = left_c + self.wall_half_x
        right_inner = right_c - self.wall_half_x
        sd_left = self.position[:, 0] - left_inner
        sd_right = right_inner - self.position[:, 0]
        sd_ground = self.position[:, 2]
        return np.concatenate([sd_left, sd_right, sd_ground])

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        if target_time > self.current_time:
            dt = target_time - self.current_time
            n_sub = max(1, int(math.ceil(dt / self.max_dt)))
            sub_dt = dt / n_sub
            for s in range(n_sub):
                t_end = self.current_time + (s + 1) * sub_dt
                left_c, right_c = self._wall_centers(t_end)
                # Force evaluation at the beginning of the sub-step.
                forces = self._compute_forces()
                self.velocity += sub_dt * forces / self.vertex_mass
                self.position += sub_dt * self.velocity
                # Enforce inextensibility and collisions.
                self._project_stretch()
                self._project_walls_ground(left_c, right_c)
            self.current_time = target_time

        return self._make_frame(target_time)

    def _make_frame(self, time: float) -> StateFrame:
        left_c, right_c = self._wall_centers(time)
        F = self._deformation_gradient()
        contact_sd = self._contact_signed_distance(left_c, right_c)
        external_force = np.tile(self.vertex_mass * self.gravity, (self.N, 1))

        frame = StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.array([0.0], dtype=float),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self.thickness,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "paper_sheet_mass_spring_pbd",
            "vertices": self.N,
            "triangles": len(self.triangles),
            "edges": len(self.edges),
            "hinges": len(self.bending_hinges),
        }
