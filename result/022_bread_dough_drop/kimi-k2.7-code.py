"""Mass-spring viscoplastic solver for the bread-dough drop scene.

The dough is modelled as a network of Hookean springs over the subject
mesh edges.  A simple per-edge plastic rest-length update gives the
viscoplastic flow that lets the dough spread on impact and leave a final
footprint, while the elastic part provides the small amount of recovery
observed in real dough.  Gravity, Rayleigh damping, Coulomb friction and
a near-inelastic ground collision complete the model.

All integration is explicit time-stepped with a fixed sub-step chosen
relative to the output sample rate for stability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene: Any, seed: int) -> None:
        # Initial state from the frozen scene.
        initial = scene.initial_state()
        self.config = scene.config

        self.time = float(initial.time)
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int64).copy()
        self.n = self.pos.shape[0]

        # Physics parameters.
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.stiffness = float(self.config["stiffness"])
        self.damping = float(self.config["damping"])
        self.restitution = float(self.config["restitution"])
        self.friction = float(self.config["friction"])
        self.plasticity = float(self.config["plasticity"])
        self.ground_z = float(self.config["ground_z"])
        self.contact_length_scale = float(self.config["contact_length_scale"])

        # Sample rate / sub-stepping.
        sample_dt = float(self.config["sample_dt"])
        target_substep = 0.001
        self.substeps = max(1, int(np.ceil(sample_dt / target_substep)))
        self.dt_sub = sample_dt / self.substeps

        # Load the subject asset to obtain connectivity and rest geometry.
        asset_path = Path(self.config["subject_asset"])
        with np.load(asset_path) as data:
            # Build an undirected edge list from the available topology.
            self.edges = self._build_edges(data)
            asset_vertices = np.asarray(data["vertices"], dtype=float)

        # Rest configuration uses the initial (possibly identical) positions.
        self.rest_pos = self.pos.copy()

        # Edge rest lengths from the initial configuration.
        i, j = self.edges[:, 0], self.edges[:, 1]
        dx = self.rest_pos[j] - self.rest_pos[i]
        self.edge_rest_length = np.linalg.norm(dx, axis=1)
        # Plastic rest length models accumulated flow; starts at elastic rest.
        self.plastic_length = self.edge_rest_length.copy()

        # Neighbor graph for per-vertex deformation gradients.
        self.neighbors: list[list[int]] = [[] for _ in range(self.n)]
        for a, b in self.edges:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        self.neighbors = [list(set(nbrs)) for nbrs in self.neighbors]

        # Pre-allocate force buffer.
        self._force = np.zeros((self.n, 3), dtype=float)

        # Cumulative source term (no mass is added or removed).
        self.cumulative_mass_source = np.zeros(1, dtype=float)

        self._initialized = True

    @staticmethod
    def _build_edges(data: Any) -> np.ndarray:
        """Return a [E,2] array of unique undirected edges from the asset."""
        edge_set: set[tuple[int, int]] = set()

        if "tets" in data or "elements" in data:
            key = "tets" if "tets" in data else "elements"
            elements = np.asarray(data[key], dtype=int)
            if elements.ndim == 2 and elements.shape[1] == 4:
                for tet in elements:
                    a, b, c, d = tet
                    for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                        edge_set.add((min(u, v), max(u, v)))

        if "faces" in data:
            faces = np.asarray(data["faces"], dtype=int)
            if faces.ndim == 2 and faces.shape[1] == 3:
                for f in faces:
                    a, b, c = f
                    for u, v in ((a, b), (b, c), (c, a)):
                        edge_set.add((min(u, v), max(u, v)))

        if not edge_set:
            # No explicit topology: build a k-NN graph from the vertices.
            verts = np.asarray(data["vertices"], dtype=float)
            k = min(8, max(2, verts.shape[0] - 1))
            tree = cKDTree(verts)
            _, idx = tree.query(verts, k=k + 1)
            for i, nbrs in enumerate(idx):
                for j in nbrs[1:]:
                    if i < j:
                        edge_set.add((i, j))
                    elif j < i:
                        edge_set.add((j, i))

        if not edge_set:
            raise ValueError("Could not build any connectivity for the subject mesh.")

        return np.asarray(sorted(edge_set), dtype=int)

    # --------------------------------------------------------------------- #
    # Integration
    # --------------------------------------------------------------------- #
    def _compute_forces(self) -> np.ndarray:
        """Compute per-vertex internal + external forces (excluding contact)."""
        self._force.fill(0.0)

        # Elastic spring forces using the plastic rest length.
        i, j = self.edges[:, 0], self.edges[:, 1]
        dx = self.pos[j] - self.pos[i]
        dist = np.linalg.norm(dx, axis=1)
        # Guard against degenerate edges.
        dist = np.where(dist < 1e-12, 1.0, dist)
        direction = dx / dist[:, None]
        delta = dist - self.plastic_length
        mag = self.stiffness * delta

        # f_i +=  m * u, f_j -= m * u
        forces = mag[:, None] * direction
        np.add.at(self._force, i, forces)
        np.add.at(self._force, j, -forces)

        # Gravity.
        self._force += self.mass[:, None] * self.gravity

        # Rayleigh velocity damping (damping coefficient in s^-1).
        self._force -= self.damping * self.mass[:, None] * self.vel

        return self._force

    def _apply_ground_contact(self) -> None:
        """Resolve ground collision with restitution and Coulomb friction."""
        z = self.pos[:, 2]
        below = z < self.ground_z
        if not np.any(below):
            return

        # Position correction: snap to ground.
        self.pos[below, 2] = self.ground_z

        # Normal velocity reflection / damping.
        v_z = self.vel[:, 2]
        hitting = below & (v_z < 0.0)
        self.vel[hitting, 2] = -self.restitution * v_z[hitting]

        # Tangential Coulomb friction.
        v_t = self.vel[below, :2]
        v_t_norm = np.linalg.norm(v_t, axis=1)
        if np.any(v_t_norm > 0.0):
            scale = np.maximum(0.0, 1.0 - self.friction * np.abs(v_t_norm) / (v_t_norm + 1e-12))
            scale = np.minimum(scale, 1.0)
            self.vel[below, :2] = (v_t.T * scale).T

    def _plastic_flow(self, dt: float) -> None:
        """Update plastic rest lengths towards current edge lengths."""
        i, j = self.edges[:, 0], self.edges[:, 1]
        dx = self.pos[j] - self.pos[i]
        current_len = np.linalg.norm(dx, axis=1)
        # Relax plastic rest length toward current length.
        self.plastic_length += self.plasticity * dt * (current_len - self.plastic_length)
        # Keep a small minimum rest length to avoid singular springs.
        np.maximum(self.plastic_length, 1e-9, out=self.plastic_length)

    def _step(self, dt: float) -> None:
        """Advance by dt with explicit semi-implicit Euler."""
        forces = self._compute_forces()
        acc = forces / self.mass[:, None]

        self.vel += dt * acc
        self.pos += dt * self.vel

        # Ground contact resolves penetration and applies friction.
        self._apply_ground_contact()

        # Viscoplastic flow: rest shape adapts to current deformation.
        self._plastic_flow(dt)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt_sub, target - self.time)
            self._step(dt)
            self.time += dt
        return self._frame()

    # --------------------------------------------------------------------- #
    # Frame export
    # --------------------------------------------------------------------- #
    def _compute_deformation_gradients(self) -> np.ndarray:
        """Per-vertex deformation gradient from local neighborhood mapping."""
        F = np.tile(np.eye(3, dtype=float)[None, :, :], (self.n, 1, 1))
        for i in range(self.n):
            nbrs = self.neighbors[i]
            if len(nbrs) < 3:
                continue
            X = self.rest_pos[nbrs] - self.rest_pos[i]
            x = self.pos[nbrs] - self.pos[i]
            # Solve x_j = F X_j in least-squares sense.
            Ft, *_ = np.linalg.lstsq(X, x, rcond=None)
            F[i] = Ft.T
        return F

    def _frame(self) -> StateFrame:
        contact_gap = self.pos[:, 2] - self.ground_z
        external_force = self.mass[:, None] * self.gravity

        frame = StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self._compute_deformation_gradients(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": contact_gap,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": external_force,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, Any]:
        return {
            "solver": "mass_spring_viscoplastic",
            "samples": self.n,
            "edges": len(self.edges) if self._initialized else 0,
            "substep_dt": self.dt_sub if self._initialized else 0.0,
        }
