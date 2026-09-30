"""Cloth solver for the moving-sphere wrinkle-wake scene.

Implements an explicit mass-spring cloth with stretch/bending resistance,
gravity, velocity damping, and stiff repulsive contact against a kinematic
sphere.  The sphere moves horizontally beneath the sheet while rising into
contact, producing a travelling contact bulge and trailing wrinkles.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically-based mass-spring cloth solver."""

    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

        # Public scene contract
        self.scene = scene
        self.config = scene.config
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        cloth_cfg = self.config["cloth"]
        self.thickness = float(cloth_cfg["thickness"])
        self.total_mass = float(cloth_cfg["total_mass"])

        # Initial state from the scene scaffold
        frame0 = scene.initial_state()
        self.time = float(frame0.time)
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=int).copy()

        # Boundary mask for pinned vertices
        self.boundary = np.asarray(frame0.attributes.get("boundary_vertex", np.zeros(len(self.position))), dtype=bool)

        # Load cloth topology
        topo_path = Path("assets/cloth_topology.json")
        if not topo_path.is_file():
            raise FileNotFoundError("Cloth topology asset not found")
        with topo_path.open() as fh:
            topo = json.load(fh)
        self.faces = np.asarray(topo["faces"], dtype=int)

        # Reference material coordinates are the initial in-plane positions
        self.ref_pos = self.position.copy()
        self.q = self.ref_pos[:, :2].copy()

        # Build unique edge list and rest lengths
        edge_set = set()
        for a, b, c in self.faces:
            for e in ((a, b), (b, c), (c, a)):
                edge_set.add((min(e), max(e)))
        self.edges = np.array(list(edge_set), dtype=int)

        self.edge_rest = np.linalg.norm(
            self.ref_pos[self.edges[:, 1]] - self.ref_pos[self.edges[:, 0]], axis=1
        )

        # Bending pairs: opposite vertices across each interior edge
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi, (a, b, c) in enumerate(self.faces):
            for e in ((a, b), (b, c), (c, a)):
                key = (min(e), max(e))
                edge_to_faces.setdefault(key, []).append(fi)

        bend_pairs = []
        for (i, j), f_list in edge_to_faces.items():
            if len(f_list) == 2:
                f1, f2 = f_list
                # third vertex of each face
                face1 = self.faces[f1]
                face2 = self.faces[f2]
                k1 = next(v for v in face1 if v not in (i, j))
                k2 = next(v for v in face2 if v not in (i, j))
                bend_pairs.append((k1, k2))
        if bend_pairs:
            self.bend = np.array(bend_pairs, dtype=int)
            self.bend_rest = np.linalg.norm(
                self.ref_pos[self.bend[:, 1]] - self.ref_pos[self.bend[:, 0]], axis=1
            )
        else:
            self.bend = np.zeros((0, 2), dtype=int)
            self.bend_rest = np.zeros(0, dtype=float)

        # Per-face reference inverses for the 2D deformation gradient
        q0 = self.q[self.faces[:, 0]]
        q1 = self.q[self.faces[:, 1]]
        q2 = self.q[self.faces[:, 2]]
        # Columns are edge vectors in material space
        Q = np.stack([q1 - q0, q2 - q0], axis=2)  # (F,2,2)
        self.Qinv = np.linalg.inv(Q)
        # Reference areas (used for averaging per-vertex F)
        self.ref_area = 0.5 * np.abs(np.linalg.det(Q))

        # Simulation parameters
        self.k_stretch = 2.0e4
        self.k_bend = 2.0e3
        self.k_contact = 2.0e4
        self.c_damp = 0.1 * np.sqrt(self.k_stretch * (self.total_mass / len(self.position)))
        self.c_bend = 0.1 * np.sqrt(self.k_bend * (self.total_mass / len(self.position)))
        self.c_contact = 80.0
        self.global_damping = 0.8

        self.dt_sub = 1.0e-4  # internal fixed sub-step

        # Cumulative mass source is zero for this scene
        self.cumulative_mass_source = np.zeros(1, dtype=float)

    # ------------------------------------------------------------------ #
    # Kinematic sphere motion
    # ------------------------------------------------------------------ #
    def _sphere_center(self, t: float) -> np.ndarray:
        """Prescribed sphere center: horizontal sweep with a slow rise."""
        x = -0.5 + 0.35 * t
        z = 0.68 + 0.20 * min(t / 2.0, 1.0)
        return np.array([x, 0.0, z], dtype=float)

    # ------------------------------------------------------------------ #
    # Physics step
    # ------------------------------------------------------------------ #
    def _spring_forces(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        pairs: np.ndarray,
        rest: np.ndarray,
        stiffness: float,
        damping: float,
    ) -> np.ndarray:
        """Return nodal forces for a set of springs (i,j)."""
        if len(pairs) == 0:
            return np.zeros_like(position)
        i, j = pairs[:, 0], pairs[:, 1]
        dx = position[j] - position[i]
        l = np.linalg.norm(dx, axis=1, keepdims=True)
        l = np.where(l < 1e-12, 1e-12, l)
        dir_vec = dx / l
        dv = velocity[j] - velocity[i]
        rel_v = np.sum(dv * dir_vec, axis=1, keepdims=True)
        strain = l.squeeze(-1) - rest
        f_mag = stiffness * strain[:, None] + damping * rel_v
        f_vec = f_mag * dir_vec

        force = np.zeros_like(position)
        np.add.at(force, i, f_vec)
        np.add.at(force, j, -f_vec)
        return force

    def _integrate(self, t_target: float) -> None:
        """Advance the cloth state from self.time to t_target with sub-stepping."""
        n_steps = max(1, int(math.ceil((t_target - self.time) / self.dt_sub)))
        dt = (t_target - self.time) / n_steps if n_steps > 0 else self.dt_sub

        for _ in range(n_steps):
            self._substep(dt)
        self.time = t_target

    def _substep(self, dt: float) -> None:
        position = self.position
        velocity = self.velocity

        # Stretch forces
        force = self._spring_forces(
            position, velocity, self.edges, self.edge_rest,
            self.k_stretch, self.c_damp,
        )

        # Bending forces
        force += self._spring_forces(
            position, velocity, self.bend, self.bend_rest,
            self.k_bend, self.c_bend,
        )

        # Gravity
        force += self.mass[:, None] * self.gravity

        # Global velocity damping
        force -= self.global_damping * velocity * self.mass[:, None]

        # Contact with moving sphere
        c = self._sphere_center(self.time)
        r = 0.62
        margin = self.thickness  # signed distance zero near the outer surface
        diff = position - c
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < 1e-12, 1e-12, dist)
        normal = diff / dist[:, None]
        penetration = r + margin - dist

        # Active contacts (negative signed distance means penetration)
        active = penetration > 0.0
        if np.any(active):
            p = penetration[active]
            n = normal[active]
            v_rel = velocity[active]
            v_n = np.sum(v_rel * n, axis=1)
            f_mag = self.k_contact * p + self.c_contact * v_n
            f_contact = (f_mag[:, None] * n)
            force[active] += f_contact

        # Symplectic Euler update
        a = force / self.mass[:, None]
        velocity_new = velocity + a * dt
        position_new = position + velocity_new * dt

        # Enforce non-penetration by projection and damping normal velocity
        diff2 = position_new - c
        dist2 = np.linalg.norm(diff2, axis=1)
        active2 = dist2 < (r + margin)
        if np.any(active2):
            n2 = diff2[active2] / dist2[active2, None]
            v_n2 = np.sum(velocity_new[active2] * n2, axis=1)
            # Push out along normal
            depth = (r + margin) - dist2[active2]
            position_new[active2] += (depth[:, None] + 1e-4) * n2
            # Remove inward normal velocity
            velocity_new[active2] -= np.minimum(v_n2, 0.0)[:, None] * n2

        # Pin boundary vertices
        if self.boundary.any():
            position_new[self.boundary] = self.ref_pos[self.boundary]
            velocity_new[self.boundary] = 0.0

        self.position = position_new
        self.velocity = velocity_new

    # ------------------------------------------------------------------ #
    # Trajectory output helpers
    # ------------------------------------------------------------------ #
    def _deformation_gradient(self) -> np.ndarray:
        """Per-vertex 2x2 in-plane deformation gradient."""
        p0 = self.position[self.faces[:, 0]]
        p1 = self.position[self.faces[:, 1]]
        p2 = self.position[self.faces[:, 2]]
        # Columns are edge vectors in world space, keep x-y components
        P = np.stack(
            [p1[:, :2] - p0[:, :2], p2[:, :2] - p0[:, :2]],
            axis=2,
        )  # (F,2,2)
        # F_tri = P @ Qinv, shape (F,2,2)
        F_tri = np.einsum("fij,fjk->fik", P, self.Qinv)

        n = len(self.position)
        vertex_F = np.zeros((n, 2, 2), dtype=float)
        counts = np.zeros(n, dtype=int)
        for f, (a, b, c) in enumerate(self.faces):
            for v in (a, b, c):
                vertex_F[v] += F_tri[f]
                counts[v] += 1
        counts = np.maximum(counts, 1)
        vertex_F /= counts[:, None, None]
        return vertex_F

    def _signed_distance(self) -> np.ndarray:
        c = self._sphere_center(self.time)
        r = 0.62
        margin = self.thickness
        dist = np.linalg.norm(self.position - c, axis=1)
        return dist - (r + margin)

    def _external_force(self) -> np.ndarray:
        return self.mass[:, None] * self.gravity

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        if time > self.time:
            self._integrate(time)
        elif time < self.time:
            # The runner advances monotonically; reset would break determinism.
            # If called out of order, re-initialize is too expensive; keep state.
            pass

        deformation = self._deformation_gradient()
        signed = self._signed_distance()
        external = self._external_force()

        return StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": deformation,
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.astype(np.int32),
            },
            fields={
                "contact_signed_distance": signed,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": self.thickness,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_cloth",
            "vertices": len(self.position),
            "faces": len(self.faces),
            "edges": len(self.edges),
            "bend_springs": len(self.bend),
            "substep": self.dt_sub,
        }
