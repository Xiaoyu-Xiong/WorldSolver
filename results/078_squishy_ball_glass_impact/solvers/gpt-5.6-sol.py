"""Deterministic deformable-solid solver for the squishy ball impact."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        initial = scene.initial_state()
        with np.load(Path(scene._asset)) as data:
            self.tets = np.asarray(data["tets"], dtype=np.int64)
            self.rest = np.asarray(data["rest_position"], dtype=np.float64).copy()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.n = len(self.x)
        pairs = np.concatenate([self.tets[:, [0, 1]], self.tets[:, [0, 2]], self.tets[:, [0, 3]],
                                self.tets[:, [1, 2]], self.tets[:, [1, 3]], self.tets[:, [2, 3]]], axis=0)
        pairs.sort(axis=1)
        self.edges = np.unique(pairs, axis=0)
        a, b = self.edges[:, 0], self.edges[:, 1]
        self.rest_len = np.maximum(np.linalg.norm(self.rest[a] - self.rest[b], axis=1), 1e-8)
        # A very soft elastic lattice gives the long compression/recovery time
        # of a gel toy; dashpots act only in equal-and-opposite edge pairs.
        self.k_edge, self.edge_damping = 0.15, 0.006
        self.gravity = np.zeros(3, dtype=np.float64)
        self.contact_k, self.contact_damping, self.wall_x = 180.0, 24.0, 0.0
        self.time = 0.0
        self._impulse = np.zeros(3, dtype=np.float64)
        self._last_external = np.zeros(3, dtype=np.float64)
        r = self.rest
        dm = np.stack((r[self.tets[:, 1]] - r[self.tets[:, 0]], r[self.tets[:, 2]] - r[self.tets[:, 0]],
                       r[self.tets[:, 3]] - r[self.tets[:, 0]]), axis=2)
        # A few scaffold tetrahedra are intentionally flat/sliver elements;
        # pseudoinverse keeps their reported gradients finite and deterministic.
        self.dm_inv = np.linalg.pinv(dm, rcond=1e-10)
        self.reference_momentum = max(float(np.linalg.norm(self.mass[:, None] * self.v, axis=1).sum()), 1e-6)

    def _forces(self):
        a, b = self.edges[:, 0], self.edges[:, 1]
        d = self.x[b] - self.x[a]
        length = np.linalg.norm(d, axis=1)
        direction = d / np.maximum(length, 1e-9)[:, None]
        extension = length - self.rest_len
        rate = np.sum((self.v[b] - self.v[a]) * direction, axis=1)
        fe = direction * (self.k_edge * extension + self.edge_damping * rate)[:, None]
        force = np.zeros_like(self.x)
        np.add.at(force, a, fe)
        np.add.at(force, b, -fe)
        external = self.mass[:, None] * self.gravity
        penetration = np.maximum(self.x[:, 0] - self.wall_x, 0.0)
        # Scale penalty by nodal mass: these are acceleration-level material
        # parameters, avoiding mesh-resolution-dependent contact impulses.
        external[:, 0] += self.mass * (
            -self.contact_k * penetration - self.contact_damping * np.maximum(self.v[:, 0], 0.0)
        )
        force += external
        self._last_external = external.sum(axis=0)
        return force, external

    def _step(self, dt):
        force, external = self._forces()
        self.v += dt * force / self.mass[:, None]
        self.x += dt * self.v
        hit = self.x[:, 0] > self.wall_x
        correction_impulse = np.zeros(3, dtype=np.float64)
        if np.any(hit):
            # The projection is the impulse limit of the same unilateral
            # contact law, so include its momentum change in the budget.
            correction_impulse[0] = np.sum(
                -self.mass[hit] * np.maximum(self.v[hit, 0], 0.0)
            )
            self.x[hit, 0] = self.wall_x
            self.v[hit, 0] = np.minimum(self.v[hit, 0], 0.0)
        self._impulse += external.sum(axis=0) * dt + correction_impulse
        self._last_external = external.sum(axis=0) + correction_impulse / dt
        self.time += dt

    def _deformation_gradient(self):
        t = self.tets
        ds = np.stack((self.x[t[:, 1]] - self.x[t[:, 0]], self.x[t[:, 2]] - self.x[t[:, 0]],
                       self.x[t[:, 3]] - self.x[t[:, 0]]), axis=2)
        return np.einsum("eij,ejk->eik", ds, self.dm_inv)

    def _make_frame(self, time):
        ext_terms = np.zeros((1, 2, 3), dtype=np.float64)
        ext_terms[0, 0] = self.mass.sum() * self.gravity
        ext_terms[0, 1] = self._last_external - ext_terms[0, 0]
        fields = {
            "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": ext_terms,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": self._deformation_gradient(),
        }
        frame = StateFrame(time=float(time), position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
                           body_id=np.ones(self.n, dtype=np.int64),
                           attributes={"component": np.zeros(self.n, dtype=np.int64), "mass_quantity": self.mass.copy()},
                           fields=fields, scalars={"reference_momentum": self.reference_momentum})
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing times")
        while self.time < target - 1e-12:
            self._step(min(0.001, target - self.time))
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"nodes": int(self.n), "tetrahedra": int(len(self.tets)), "edges": int(len(self.edges)),
                "integrated_time": float(self.time)}
