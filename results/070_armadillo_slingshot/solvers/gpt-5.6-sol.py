"""Deterministic tetrahedral elastodynamics for the Armadillo slingshot.

This uses a compliant position based discretisation of a volumetric solid:
tetrahedral edges provide shear/stretch resistance and signed cell volumes
provide the nearly-incompressible part of the material response.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small, deterministic XPBD-style solid mechanics solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.cfg = scene.config
        self.time = 0.0
        self.rest = np.asarray(initial.position, dtype=np.float64).copy()
        self.x = self.rest.copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.fixed = np.asarray(initial.attributes["fixed_grasp"]) > 0.5
        self.pulled = np.asarray(initial.attributes["pulled_patch"]) > 0.5
        self.n = len(self.x)

        root = Path(__file__).resolve().parents[1]
        with np.load(root / "assets" / "armadillo_solid.npz") as solid:
            self.tets = np.asarray(solid["tetrahedra"], dtype=np.int64)

        # Cell edges discretise strain.  The asset has overlapping anatomical
        # volumes, so short cross-component links weld them into one body.
        raw_edges = np.concatenate(
            [self.tets[:, [a, b]] for a, b in
             ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=0)
        raw_edges.sort(axis=1)
        edges = np.unique(raw_edges, axis=0)
        parent = np.arange(self.n, dtype=np.int64)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = int(parent[a])
            return a

        for a, b in edges:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        component = np.asarray([find(i) for i in range(self.n)])
        weld = [(int(a), int(b))
                for a, b in cKDTree(self.rest).query_pairs(
                    0.135, output_type="ndarray")
                if component[a] != component[b]]
        if weld:
            edges = np.unique(
                np.vstack((edges, np.asarray(weld, dtype=np.int64))), axis=0)

        edge_vec = self.rest[edges[:, 1]] - self.rest[edges[:, 0]]
        edge_len = np.linalg.norm(edge_vec, axis=1)
        good = edge_len > 1.0e-9
        self.edges = edges[good]
        self.edge_rest = edge_len[good]
        self.edge_count = np.maximum(
            np.bincount(self.edges.ravel(), minlength=self.n).astype(float), 1.0)

        dm = np.stack((
            self.rest[self.tets[:, 1]] - self.rest[self.tets[:, 0]],
            self.rest[self.tets[:, 2]] - self.rest[self.tets[:, 0]],
            self.rest[self.tets[:, 3]] - self.rest[self.tets[:, 0]],
        ), axis=2)
        self.dm_inv = np.linalg.pinv(dm, rcond=1.0e-10)
        self.rest_volume = np.linalg.det(dm) / 6.0
        self.live_tets = np.flatnonzero(np.abs(self.rest_volume) > 1.0e-11)
        self.tet_count = np.maximum(np.bincount(
            self.tets[self.live_tets].ravel(), minlength=self.n).astype(float), 1.0)

        self.gravity = np.asarray(self.cfg["gravity"], dtype=np.float64)
        pull = self.cfg["boundary_conditions"]["temporary_pull"]
        self.target_start = np.asarray(pull["target_start"], dtype=np.float64)
        self.target_peak = np.asarray(pull["target_peak"], dtype=np.float64)
        self.pull_anchor = (np.mean(self.rest[self.pulled], axis=0)
                            if np.any(self.pulled) else self.rest.mean(axis=0))
        self.pull_peak_time = float(self.cfg["event_timeline"]["pull_peak_s"])
        self.release_time = float(self.cfg["event_timeline"]["release_s"])
        self.max_step = 0.0025
        self.iterations = 5
        self.edge_stiffness = 0.72
        self.volume_stiffness = 0.38
        self.velocity_damping = 0.10
        self.cumulative_impulse = np.zeros(3, dtype=np.float64)
        self.last_external_force = np.zeros((1, 2, 3), dtype=np.float64)
        self.steps = 0

    def _pull_displacement(self, t: float) -> np.ndarray:
        if t >= self.release_time:
            return np.zeros(3, dtype=np.float64)
        u = min(max(t / self.pull_peak_time, 0.0), 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        target = self.target_start + smooth * (self.target_peak - self.target_start)
        return target - self.pull_anchor

    def _project_edges(self, movable: np.ndarray) -> None:
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = self.x[j] - self.x[i]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1.0e-12)
        amount = 0.5 * self.edge_stiffness * (length - self.edge_rest) / safe
        c = amount[:, None] * d
        corr = np.zeros_like(self.x)
        np.add.at(corr, i, c)
        np.add.at(corr, j, -c)
        corr /= self.edge_count[:, None]
        self.x[movable] += corr[movable]

    def _project_volumes(self, movable: np.ndarray) -> None:
        ids = self.live_tets
        tet = self.tets[ids]
        p0, p1, p2, p3 = (self.x[tet[:, k]] for k in range(4))
        e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
        volume = np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0
        g1 = np.cross(e2, e3) / 6.0
        g2 = np.cross(e3, e1) / 6.0
        g3 = np.cross(e1, e2) / 6.0
        g0 = -(g1 + g2 + g3)
        grads = (g0, g1, g2, g3)
        denom = sum(np.einsum("ij,ij->i", g, g) for g in grads) + 1.0e-18
        lam = -self.volume_stiffness * (volume - self.rest_volume[ids]) / denom
        max_grad = np.maximum.reduce([np.linalg.norm(g, axis=1) for g in grads])
        cap = 0.018 / np.maximum(max_grad, 1.0e-12)
        lam = np.clip(lam, -cap, cap)
        corr = np.zeros_like(self.x)
        for k, g in enumerate(grads):
            np.add.at(corr, tet[:, k], lam[:, None] * g)
        corr /= self.tet_count[:, None]
        self.x[movable] += corr[movable]

    def _substep(self, h: float, new_time: float) -> None:
        old_x = self.x.copy()
        momentum_before = np.sum(self.mass[:, None] * self.v, axis=0)
        self.v += h * self.gravity
        vcm = np.sum(self.mass[:, None] * self.v, axis=0) / np.sum(self.mass)
        decay = np.exp(-self.velocity_damping * h)
        self.v = vcm + decay * (self.v - vcm)
        self.x += h * self.v

        pull_active = new_time < self.release_time - 1.0e-12
        constrained = self.fixed | (self.pulled if pull_active else False)
        movable = ~constrained
        pull_position = self.rest[self.pulled] + self._pull_displacement(new_time)
        self.x[self.fixed] = self.rest[self.fixed]
        if pull_active:
            self.x[self.pulled] = pull_position
        for _ in range(self.iterations):
            self._project_edges(movable)
            self._project_volumes(movable)
            self.x[self.fixed] = self.rest[self.fixed]
            if pull_active:
                self.x[self.pulled] = pull_position

        self.v = (self.x - old_x) / h
        self.v[self.fixed] = 0.0
        momentum_after = np.sum(self.mass[:, None] * self.v, axis=0)
        impulse = momentum_after - momentum_before
        self.cumulative_impulse += impulse
        gravity_force = np.sum(self.mass) * self.gravity
        reaction_force = impulse / h - gravity_force
        self.last_external_force[0, 0] = gravity_force
        self.last_external_force[0, 1] = reaction_force
        self.steps += 1

    def _deformation_gradients(self) -> np.ndarray:
        tet = self.tets
        ds = np.stack((
            self.x[tet[:, 1]] - self.x[tet[:, 0]],
            self.x[tet[:, 2]] - self.x[tet[:, 0]],
            self.x[tet[:, 3]] - self.x[tet[:, 0]],
        ), axis=2)
        return np.matmul(ds, self.dm_inv)

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            h = min(self.max_step, target - self.time)
            self._substep(h, self.time + h)
            self.time += h
        self.time = target
        frame = StateFrame(
            target, self.x.copy(), self.v.copy(), mass=self.mass.copy(),
            attributes={
                "fixed_grasp": self.fixed.astype(np.float64),
                "pulled_patch": self.pulled.astype(np.float64),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse":
                    self.cumulative_impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum":
                    np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum":
                    np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights":
                    np.ones((1, self.n), dtype=np.float64),
                "budget_external_force": self.last_external_force.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": self._deformation_gradients(),
            },
            scalars={
                "density_kg_m3": 1000.0,
                "reference_momentum": float(np.sum(self.mass) * 5.0),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "tetrahedral compliant position based dynamics",
            "substeps": self.steps,
            "node_count": self.n,
            "tetrahedron_count": len(self.tets),
            "edge_constraint_count": len(self.edges),
        }
