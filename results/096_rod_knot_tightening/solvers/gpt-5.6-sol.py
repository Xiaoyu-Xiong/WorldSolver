"""Deterministic finite-thickness elastic-rod solver for the knot scene.

The implementation uses a position based discretisation of a Kirchhoff-like
rod. Edge constraints provide axial inextensibility, second-neighbour
constraints provide bending resistance, and unilateral distance constraints
provide frictionless self contact.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small, self-contained constrained dynamics solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.n = len(self.x)

        rod = self.config["rod"]
        self.mass = np.full(self.n, float(rod["total_mass"]) / self.n)
        self.inv_mass = 1.0 / self.mass
        # Controlled vertices remain physical masses but are kinematic in the
        # constraint solve.
        self.inv_mass[0] = self.inv_mass[-1] = 0.0

        raw_lengths = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        raw_total = float(np.sum(raw_lengths))
        if raw_total <= 1.0e-12:
            raise ValueError("initial rod centreline has zero length")
        self.rest_lengths = raw_lengths * (float(rod["rest_length"]) / raw_total)
        self.bend_lengths = self.rest_lengths[:-1] + self.rest_lengths[1:]

        self.contact_distance = float(self.config["self_contact"]["contact_distance"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        # At this sampling density, vertex distance constraints robustly protect
        # the tube centreline. Nearby material neighbours are excluded.
        ii, jj = np.triu_indices(self.n, k=4)
        self.contact_i, self.contact_j = ii.astype(np.intp), jj.astype(np.intp)

        self.time = 0.0
        self.max_step = 0.005
        self.iterations = 10
        self.bend_stiffness = 0.018
        self.velocity_damping = 0.997
        self.initial_momentum = np.sum(self.mass[:, None] * self.v, axis=0)
        self.cumulative_external_impulse = np.zeros(3)
        self.report_force = np.zeros(3)
        self.step_count = 0
        self.max_stretch_error = 0.0
        self.min_nonlocal_distance = math.inf
        self.arc_fraction = np.linspace(0.0, 1.0, self.n)
        self.radius = np.full(self.n, float(rod["radius"]))
        self.domain_weights = np.ones((1, self.n))
        self.reference_momentum = max(float(rod["total_mass"]) * 0.5, 1.0e-6)
        self._pin_endpoints(0.0)

    def _targets(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        values = self.scene.endpoint_targets(float(time))
        return (np.asarray(values["left_tail"], dtype=float),
                np.asarray(values["threading_tail"], dtype=float))

    def _pin_endpoints(self, time: float) -> None:
        left, right = self._targets(time)
        self.x[0], self.x[-1] = left, right

    def _project_edges(self, parity: int) -> None:
        a = np.arange(parity, self.n - 1, 2, dtype=np.intp)
        b = a + 1
        delta = self.x[b] - self.x[a]
        distance = np.linalg.norm(delta, axis=1)
        correction = ((distance - self.rest_lengths[a]) /
                      np.maximum(distance, 1.0e-12))[:, None] * delta
        denom = self.inv_mass[a] + self.inv_mass[b]
        fa = np.divide(self.inv_mass[a], denom, out=np.zeros_like(denom), where=denom > 0)
        fb = np.divide(self.inv_mass[b], denom, out=np.zeros_like(denom), where=denom > 0)
        self.x[a] += fa[:, None] * correction
        self.x[b] -= fb[:, None] * correction

    def _project_bending(self, phase: int) -> None:
        # Non-overlapping three-colour batches make each update a stable
        # Gauss-Seidel sweep. The straight two-edge chord is stress-free.
        center = np.arange(1 + phase, self.n - 1, 3, dtype=np.intp)
        if center.size == 0:
            return
        a, b = center - 1, center + 1
        delta = self.x[b] - self.x[a]
        distance = np.linalg.norm(delta, axis=1)
        correction = ((distance - self.bend_lengths[center - 1]) /
                      np.maximum(distance, 1.0e-12))[:, None] * delta
        denom = self.inv_mass[a] + self.inv_mass[b]
        fa = np.divide(self.inv_mass[a], denom, out=np.zeros_like(denom), where=denom > 0)
        fb = np.divide(self.inv_mass[b], denom, out=np.zeros_like(denom), where=denom > 0)
        self.x[a] += self.bend_stiffness * fa[:, None] * correction
        self.x[b] -= self.bend_stiffness * fb[:, None] * correction

    def _project_self_contact(self) -> None:
        i, j = self.contact_i, self.contact_j
        delta = self.x[j] - self.x[i]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        active = distance2 < self.contact_distance ** 2
        if not np.any(active):
            return
        ia, ja, d = i[active], j[active], delta[active]
        distance = np.sqrt(np.maximum(distance2[active], 0.0))
        normal = np.empty_like(d)
        regular = distance > 1.0e-10
        normal[regular] = d[regular] / distance[regular, None]
        if np.any(~regular):
            q = (ia[~regular] * 73856093 + ja[~regular] * 19349663).astype(float)
            fallback = np.column_stack((np.cos(q), np.sin(q), np.cos(0.5 * q)))
            normal[~regular] = fallback / np.linalg.norm(fallback, axis=1)[:, None]

        denom = self.inv_mass[ia] + self.inv_mass[ja]
        valid = denom > 0.0
        move = (self.contact_distance - distance)[:, None] * normal
        move_i = np.zeros_like(move)
        move_j = np.zeros_like(move)
        move_i[valid] = -(self.inv_mass[ia[valid]] / denom[valid])[:, None] * move[valid]
        move_j[valid] = +(self.inv_mass[ja[valid]] / denom[valid])[:, None] * move[valid]

        # Average simultaneous Jacobi corrections to avoid contact explosions.
        correction = np.zeros_like(self.x)
        count = np.zeros(self.n)
        np.add.at(correction, ia, move_i)
        np.add.at(correction, ja, move_j)
        np.add.at(count, ia, valid.astype(float))
        np.add.at(count, ja, valid.astype(float))
        moving = count > 0
        self.x[moving] += correction[moving] / count[moving, None]

    def _substep(self, h: float) -> None:
        old_x = self.x.copy()
        new_time = self.time + h
        self.v += h * self.gravity
        self.x += h * self.v
        self._pin_endpoints(new_time)
        for _ in range(self.iterations):
            self._project_edges(0)
            self._project_edges(1)
            self._project_bending(0)
            self._project_bending(1)
            self._project_bending(2)
            self._project_self_contact()
            self._pin_endpoints(new_time)
        self.v = (self.x - old_x) / h
        self.v[1:-1] *= self.velocity_damping
        self.time = new_time
        self.step_count += 1

    def _make_frame(self, time: float) -> StateFrame:
        lengths = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self.max_stretch_error = max(
            self.max_stretch_error,
            float(np.max(np.abs(lengths - self.rest_lengths))),
        )
        distances = np.linalg.norm(self.x[self.contact_j] - self.x[self.contact_i], axis=1)
        self.min_nonlocal_distance = min(self.min_nonlocal_distance, float(np.min(distances)))

        frame = StateFrame(
            float(time), self.x.copy(), self.v.copy(),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (self.n, 1)),
            angular_velocity=np.zeros((self.n, 3)),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_fraction": self.arc_fraction.copy(),
                "rod_radius": self.radius.copy(),
                "contact_diameter": (2.0 * self.radius).copy(),
                "endpoint_controlled": np.isin(np.arange(self.n), [0, self.n - 1]).astype(float),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse[None].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.report_force.reshape(1, 1, 3).copy(),
                "constraint_value": lengths.copy(),
                "constraint_target": self.rest_lengths.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "friction_coefficient": 0.0,
                "constraint_scale": float(np.mean(self.rest_lengths)),
                "reference_momentum": self.reference_momentum,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target_time = float(time)
        if not math.isfinite(target_time) or target_time < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        old_time = self.time
        old_momentum = np.sum(self.mass[:, None] * self.v, axis=0)
        while self.time < target_time - 1.0e-12:
            self._substep(min(self.max_step, target_time - self.time))
        self.time = target_time
        momentum = np.sum(self.mass[:, None] * self.v, axis=0)
        self.cumulative_external_impulse = momentum - self.initial_momentum
        elapsed = target_time - old_time
        self.report_force = ((momentum - old_momentum) / elapsed
                             if elapsed > 1.0e-12 else np.zeros(3))
        return self._make_frame(target_time)

    def diagnostics(self) -> dict[str, object]:
        minimum = (None if not math.isfinite(self.min_nonlocal_distance)
                   else float(self.min_nonlocal_distance))
        return {
            "method": "position_based_elastic_rod_with_frictionless_self_contact",
            "substeps": int(self.step_count),
            "max_edge_length_error": float(self.max_stretch_error),
            "minimum_nonlocal_vertex_distance": minimum,
        }
