"""Physically-based mass-spring solver for the hard-sphere-on-ground scene.

The solver treats the subject mesh as a particle cloud connected by a nearest-neighbor
spring graph.  It advances with explicit symplectic Euler substeps, handles ground
contact via position projection and velocity impulses with restitution/friction,
and exports the required StateFrame trajectory fields.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._time = 0.0
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        # Deterministic; seed is kept for reproducibility even though the
        # integration is fully deterministic.
        if seed is not None:
            np.random.seed(seed)

        # Prefer the scene config, but fall back to the workspace config file.
        self.cfg: dict[str, Any] = {}
        if hasattr(scene, "config"):
            self.cfg.update(scene.config)
        try:
            from worldsolver.config import load_yaml  # type: ignore

            cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
            if cfg_path.is_file():
                self.cfg.update(load_yaml(cfg_path))
        except Exception:
            pass

        frame = scene.initial_state()
        self.x = np.asarray(frame.position, dtype=float).copy()
        self.v = np.asarray(frame.velocity, dtype=float).copy()
        self.m = np.asarray(frame.mass, dtype=float).copy()
        self.n = self.x.shape[0]

        if frame.body_id is None:
            self.body_id: np.ndarray | None = None
        else:
            self.body_id = np.asarray(frame.body_id, dtype=np.int64).copy()

        self.gravity = np.asarray(
            self.cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.k = float(self.cfg.get("stiffness", 44.0))
        self.c_damp = float(self.cfg.get("damping", 0.75))
        self.restitution = float(self.cfg.get("restitution", 0.62))
        self.friction = float(self.cfg.get("friction", 0.42))
        self.contact_length_scale = float(self.cfg.get("contact_length_scale", 0.04))
        sample_dt = float(self.cfg.get("sample_dt", 1.0 / 30.0))

        # Ten substeps per output frame; small enough for the stiff spring model.
        self._dt = sample_dt / 10.0
        if self._dt <= 0.0:
            self._dt = 1e-4

        # Rest configuration for the elastic model.
        self.rest_x = self.x.copy()

        # Build the elastic graph (FEM elements if provided, otherwise k-nearest).
        self._load_topology()

        # Precompute neighbor data for per-vertex deformation gradients.
        self._prepare_deformation_gradient()

        self._time = 0.0
        self._initialized = True

    def _load_topology(self) -> None:
        """Load element/face connectivity from the subject asset, or build a
        nearest-neighbor spring graph when no connectivity is supplied."""
        asset_path = (
            Path(__file__).resolve().parent.parent
            / self.cfg.get("subject_asset", "assets/subject.npz")
        )

        elements: np.ndarray | None = None
        faces: np.ndarray | None = None
        try:
            with np.load(asset_path) as data:
                if "elements" in data:
                    elements = data["elements"]
                elif "tets" in data:
                    elements = data["tets"]
                elif "cells" in data:
                    elements = data["cells"]
                elif "faces" in data:
                    faces = data["faces"]
                elif "triangles" in data:
                    faces = data["triangles"]
        except Exception:
            pass

        edges: set[tuple[int, int]] = set()

        if elements is not None and elements.ndim == 2 and elements.shape[1] >= 3:
            elems = np.asarray(elements, dtype=int)
            for e in elems:
                for a, b in itertools.combinations(e, 2):
                    if 0 <= a < self.n and 0 <= b < self.n and a != b:
                        edges.add((min(a, b), max(a, b)))
        elif faces is not None and faces.ndim == 2 and faces.shape[1] == 3:
            tri = np.asarray(faces, dtype=int)
            for f in tri:
                a, b, c = f
                for p, q in ((a, b), (b, c), (c, a)):
                    if 0 <= p < self.n and 0 <= q < self.n and p != q:
                        edges.add((min(p, q), max(p, q)))

        if not edges:
            edges = self._build_neighbor_graph()

        if not edges:
            # Degenerate case: no internal elasticity.
            self.edges = np.empty((0, 2), dtype=int)
            self.rest_lengths = np.empty((0,), dtype=float)
        else:
            self.edges = np.array(sorted(edges), dtype=int)
            self.rest_lengths = np.linalg.norm(
                self.rest_x[self.edges[:, 0]] - self.rest_x[self.edges[:, 1]], axis=1
            )

    def _build_neighbor_graph(self) -> set[tuple[int, int]]:
        """Build an undirected spring graph from rest-pose neighbors."""
        if self.n < 2:
            return set()

        tree = cKDTree(self.rest_x)
        dd, _ = tree.query(self.rest_x, k=2)
        avg_nn = float(np.mean(dd[:, 1]))
        if avg_nn <= 0.0:
            avg_nn = 1e-3

        cutoff = 2.5 * avg_nn
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            return set()

        pairs = np.sort(pairs, axis=1)
        return set(map(tuple, pairs.tolist()))

    def _prepare_deformation_gradient(self) -> None:
        """Precompute rest-neighbor offsets and regularized inverse matrices for a
        least-squares per-vertex deformation gradient."""
        k = min(8, max(3, self.n - 1))
        if self.n < 4:
            self._fg_neighbors = np.zeros((self.n, k), dtype=int)
            self._fg_Q = np.zeros((self.n, k, 3), dtype=float)
            self._fg_Binv = np.tile(np.eye(3), (self.n, 1, 1))
            return

        tree = cKDTree(self.rest_x)
        _, nbrs = tree.query(self.rest_x, k=k + 1)
        nbrs = np.asarray(nbrs[:, 1:], dtype=int)
        self._fg_neighbors = nbrs

        Q = np.zeros((self.n, k, 3), dtype=float)
        Binv = np.zeros((self.n, 3, 3), dtype=float)
        for i in range(self.n):
            qi = self.rest_x[nbrs[i]] - self.rest_x[i]
            Q[i] = qi
            B = qi.T @ qi
            # Regularize to give a well-defined identity in the normal direction
            # for thin shells.
            reg = 1e-6 * np.trace(B)
            if reg > 0.0:
                B = B + reg * np.eye(3)
            try:
                Binv[i] = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                Binv[i] = np.eye(3)
        self._fg_Q = Q
        self._fg_Binv = Binv

    def _deformation_gradient(self) -> np.ndarray:
        """Compute per-vertex deformation gradient F = P B^{-1}."""
        dX = self.x[self._fg_neighbors] - self.x[:, None, :]
        P = np.einsum("nki,nkj->nij", dX, self._fg_Q)
        F = P @ self._fg_Binv
        if not np.isfinite(F).all():
            F = np.tile(np.eye(3), (self.n, 1, 1))
        return F

    def _elastic_forces(self) -> np.ndarray:
        """Spring+dashpot forces for the current state."""
        f = np.zeros_like(self.x)
        if self.edges.shape[0] == 0:
            return f

        i = self.edges[:, 0]
        j = self.edges[:, 1]

        r = self.x[i] - self.x[j]
        d = np.linalg.norm(r, axis=1)
        safe = d > 1e-12
        d = np.where(safe, d, 1.0)
        dir_vec = r / d[:, None]

        # Linear-elastic spring based on engineering strain.
        strain = (d - self.rest_lengths) / self.rest_lengths
        fs = -(self.k * strain)[:, None] * dir_vec

        # Damping along the spring direction.
        v_ij = self.v[i] - self.v[j]
        rel_v = np.einsum("ij,ij->i", v_ij, dir_vec)
        fd = -(self.c_damp * rel_v)[:, None] * dir_vec

        edge_force = fs + fd
        np.add.at(f, i, edge_force)
        np.subtract.at(f, j, edge_force)
        return f

    def _apply_contact(self) -> None:
        """Hard ground contact: position projection, restitution, and friction."""
        gap = self.x[:, 2] - self.ground_z
        penetrating = gap < 0.0
        if not np.any(penetrating):
            return

        idx = np.nonzero(penetrating)[0]

        # Eliminate penetration for the rendered state.
        self.x[idx, 2] = self.ground_z

        # Normal impulse.
        v_z = self.v[idx, 2].copy()
        moving_down = v_z < 0.0
        v_z[moving_down] = -self.restitution * v_z[moving_down]
        self.v[idx, 2] = v_z

        # Coulomb friction on the tangential component.
        v_tan = self.v[idx, :2]
        mag = np.linalg.norm(v_tan, axis=1)
        # Normal impulse magnitude ~ (1+e) * |v_n|; scale tangential velocity.
        reduction = self.friction * (1.0 + self.restitution) * np.abs(self.v[idx, 2])
        scale = np.ones_like(mag)
        nonzero = mag > 1e-12
        scale[nonzero] = np.maximum(0.0, 1.0 - reduction[nonzero] / mag[nonzero])
        self.v[idx, :2] *= scale[:, None]

    def _step(self, dt: float) -> None:
        """One symplectic Euler substep."""
        f = self.m[:, None] * self.gravity
        f += self._elastic_forces()

        self.v += dt * f / self.m[:, None]
        self.x += dt * self.v
        self._apply_contact()

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            time = self._time
        target = time

        while self._time < target - 1e-12:
            dt = min(self._dt, target - self._time)
            self._step(dt)
            self._time += dt

        F = self._deformation_gradient()
        gap = self.x[:, 2] - self.ground_z
        external_force = self.m[:, None] * self.gravity

        body_id = self.body_id
        if body_id is None:
            body_id = np.zeros(self.n, dtype=np.int64)

        frame = StateFrame(
            time=float(target),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
                "contact_signed_distance": gap,
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {}
