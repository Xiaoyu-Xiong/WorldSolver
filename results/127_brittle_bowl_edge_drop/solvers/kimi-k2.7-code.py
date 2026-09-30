"""Brittle bowl edge-drop solver.

A particle-spring thin-shell solver with explicit time integration, penalty
contact against a rigid floor, and a simple damage-based brittle fracture
model.  The bowl is represented by the vertices and triangles of the supplied
``bowl.obj`` asset; each vertex is a particle and each mesh edge is a
stretch spring.  When a spring's tensile strain exceeds a material threshold
for long enough (damage accumulation) the spring is removed, allowing the mesh
graph to split into distinct fragments.  Output conforms to the StateFrame and
trajectory field contracts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically-based brittle-bowl solver."""

    # Material / numerical parameters.  These are intentionally simple; the
    # evaluation is qualitative and the bowl scale is set by the scaffold.
    K_ELASTIC = 1.5e6          # edge spring stiffness [N/m]
    K_CONTACT = 1.5e6          # floor penalty stiffness [N/m]
    DAMPING = 0.4              # global velocity damping [kg/s]
    MU_FRICTION = 0.35         # dynamic friction coefficient
    CONTACT_MARGIN = 0.015     # [m] over which the penalty contact is active
    STRAIN_THRESHOLD = 0.004   # tensile strain at which damage begins
    DAMAGE_RATE = 800.0        # damage accumulation speed [1/s]
    DT = 1e-4                  # fixed sub-step [s]

    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.config = scene.config
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        sampling = self.config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])

        # Initial state from the deterministic scaffold.
        self.frame0 = scene.initial_state()
        self.position = self.frame0.position.copy()
        self.velocity = self.frame0.velocity.copy()
        self.mass = self.frame0.mass.copy()
        self.N = len(self.position)

        # Reference (rest) configuration is the initial scaffold pose.
        self.rest_position = self.position.copy()

        # Load mesh connectivity from the bowl asset.  The vertex ordering in
        # bowl.obj matches the topology used by scene.py, so we can use the
        # scaffold-provided positions directly with these faces.
        self.faces = self._parse_obj_faces(
            Path(__file__).resolve().parent.parent / "assets" / "bowl.obj"
        )

        # Build unique, undirected edge list.
        edge_set: set[Tuple[int, int]] = set()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                edge_set.add((u, v) if u < v else (v, u))
        self.edges = np.array(sorted(edge_set), dtype=np.int64)  # [E, 2]
        self.E = self.edges.shape[0]

        self.rest_length = np.linalg.norm(
            self.rest_position[self.edges[:, 1]] - self.rest_position[self.edges[:, 0]],
            axis=1,
        )
        self.active_edge = np.ones(self.E, dtype=bool)
        self.edge_damage = np.zeros(self.E, dtype=float)

        # Neighbor lists for per-particle deformation gradient and body ids.
        self.neighbors: List[List[int]] = [[] for _ in range(self.N)]
        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        # Simulation clock.
        self.time = 0.0

        # Cached scalar constants.
        self._contact_length_scale = self.spacing

        # Diagnostic counters.
        self._broken_count = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("advance_to time must be non-decreasing")

        while self.time < time:
            sub_dt = min(self.DT, time - self.time)
            self._step(sub_dt)
            self.time += sub_dt

        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "particles": self.N,
            "edges": self.E,
            "broken_edges": int(self._broken_count),
            "time": float(self.time),
        }

    # --------------------------------------------------------------------- #
    # Time stepping
    # --------------------------------------------------------------------- #
    def _step(self, dt: float) -> None:
        # Gravity + damping.
        force = self.mass[:, None] * self.gravity - self.DAMPING * self.velocity

        # Elastic stretch forces on still-intact edges.
        if self.active_edge.any():
            i = self.edges[self.active_edge, 0]
            j = self.edges[self.active_edge, 1]
            dx = self.position[j] - self.position[i]
            dist = np.linalg.norm(dx, axis=1)
            dist[dist < 1e-12] = 1e-12

            rest = self.rest_length[self.active_edge]
            strain = (dist - rest) / rest

            # Damage accumulation: tensile strain beyond the brittle limit.
            tensile_excess = np.maximum(strain - self.STRAIN_THRESHOLD, 0.0)
            self.edge_damage[self.active_edge] += tensile_excess * self.DAMAGE_RATE * dt

            # Break edges whose damage has reached unity.
            broken_now = self.active_edge.copy()
            broken_now[self.active_edge] = self.edge_damage[self.active_edge] >= 1.0
            if broken_now.any():
                self.active_edge &= ~broken_now
                self._broken_count += int(broken_now.sum())

            # Recompute active subset after possible breaks.
            i = self.edges[self.active_edge, 0]
            j = self.edges[self.active_edge, 1]
            dx = self.position[j] - self.position[i]
            dist = np.linalg.norm(dx, axis=1)
            dist[dist < 1e-12] = 1e-12
            rest = self.rest_length[self.active_edge]

            f_mag = self.K_ELASTIC * (dist - rest)
            direction = dx / dist[:, None]
            f_ij = f_mag[:, None] * direction

            np.add.at(force, i, f_ij)
            np.add.at(force, j, -f_ij)

        # Floor contact (penalty + friction, with a small compliance margin).
        force += self._contact_force(self.position, self.velocity)

        # Symplectic Euler update.
        self.velocity += dt * force / self.mass[:, None]
        self.position += dt * self.velocity

        # Hard non-penetration constraint.
        below = self.position[:, 2] < 0.0
        if below.any():
            self.position[below, 2] = 0.0
            self.velocity[below, 2] = np.maximum(self.velocity[below, 2], 0.0)

    def _contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Penalty + friction force against the rigid floor at z=0."""
        f = np.zeros_like(x)
        d = x[:, 2]  # signed distance to the floor plane without margin
        penetration = self.CONTACT_MARGIN - d
        contact = penetration > 0.0
        if not contact.any():
            return f

        indices = np.nonzero(contact)[0]
        v_z = v[indices, 2]
        mass_c = self.mass[indices]

        # Damping tuned near critical for the contact spring.
        c_contact = 2.0 * np.sqrt(self.K_CONTACT * mass_c)
        f_n = self.K_CONTACT * penetration[indices] - c_contact * v_z
        f_n = np.maximum(f_n, 0.0)

        f[indices, 2] += f_n

        # Tangential Coulomb friction.
        v_t = v[indices].copy()
        v_t[:, 2] = 0.0
        v_t_norm = np.linalg.norm(v_t, axis=1)
        moving = v_t_norm > 1e-12
        if moving.any():
            m_idx = indices[moving]
            v_t_m = v_t[moving]
            v_t_n = v_t_norm[moving]
            f_t_max = self.MU_FRICTION * f_n[moving]
            # Do not reverse tangential motion in a single sub-step.
            f_t_mag = np.minimum(f_t_max, mass_c[moving] * v_t_n / self.DT)
            f_t = -(f_t_mag / v_t_n)[:, None] * v_t_m
            f[m_idx, :2] += f_t[:, :2]

        return f

    # --------------------------------------------------------------------- #
    # Frame construction
    # --------------------------------------------------------------------- #
    def _build_frame(self, time: float) -> StateFrame:
        # External force at the sample instant: gravity plus floor contact.
        external_force = (
            self.mass[:, None] * self.gravity
            + self._contact_force(self.position, self.velocity)
        )

        # Per-particle signed gap to the floor (negative => penetration).
        contact_signed_distance = self.position[:, 2].copy()

        # Cumulative mass source: none, so scalar zero.
        cumulative_mass_source = np.zeros(1, dtype=float)

        # Connected components of the still-intact edge graph give fragment ids.
        body_id = self._compute_body_ids()

        # Deformation gradient computed from local neighbor reference/current data.
        deformation_gradient = self._compute_deformation_gradient()

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation_gradient,
                "particle_radius": np.full(self.N, 0.45 * self.spacing, dtype=float),
                "material_label": np.zeros(self.N, dtype=np.int32),
            },
            fields={
                "external_force": external_force,
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "contact_length_scale": self._contact_length_scale,
                "grid_spacing": self.spacing,
                "characteristic_length": self.spacing,
                "reference_velocity": 1.0,
            },
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _compute_body_ids(self) -> np.ndarray:
        """Union-find connected components over unbroken edges."""
        parent = np.arange(self.N, dtype=np.int32)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for idx in np.nonzero(self.active_edge)[0]:
            i, j = self.edges[idx]
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        body = np.empty(self.N, dtype=np.int32)
        for i in range(self.N):
            body[i] = find(i)
        return body

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Per-particle least-squares deformation gradient F = dx/dX."""
        F = np.tile(np.eye(3, dtype=float)[None], (self.N, 1, 1))
        x0 = self.rest_position
        x1 = self.position

        for i in range(self.N):
            # Use only still-intact neighbors to respect the fractured state.
            nbrs = []
            for j in self.neighbors[i]:
                # Find edge index quickly?  The neighbor list is small enough
                # that a linear scan over edges is acceptable once per frame.
                a, b = (i, j) if i < j else (j, i)
                idx = np.where((self.edges[:, 0] == a) & (self.edges[:, 1] == b))[0]
                if idx.size and self.active_edge[idx[0]]:
                    nbrs.append(j)

            if len(nbrs) < 3:
                continue

            dX = x0[nbrs] - x0[i]
            dx = x1[nbrs] - x1[i]
            # F^T = pinv(dX) @ dx  =>  F = (pinv(dX) @ dx).T
            F_i = np.linalg.pinv(dX) @ dx
            F[i] = F_i.T

        return F

    @staticmethod
    def _parse_obj_faces(path: Path) -> np.ndarray:
        """Return a [F,3] 0-indexed triangle array from a minimal .obj file."""
        faces: List[Tuple[int, int, int]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("f "):
                    continue
                parts = line.split()
                # Handle "f v/vt/vn v v" by taking the first integer of each part.
                a = int(parts[1].split("/")[0]) - 1
                b = int(parts[2].split("/")[0]) - 1
                c = int(parts[3].split("/")[0]) - 1
                faces.append((a, b, c))
        return np.asarray(faces, dtype=np.int64)
