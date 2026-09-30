"""Deterministic elastoplastic shell solver for the hollow ball drop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring shell with plastic rest lengths and ductile bond failure."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._x = np.asarray(initial.position, dtype=float).copy()
        self._v = np.asarray(initial.velocity, dtype=float).copy()
        self._x0 = self._x.copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._n = len(self._x)
        self._time = 0.0
        self._gravity = np.asarray(scene.config["gravity"], dtype=float)
        self._spacing = float(scene.config["sampling"]["spacing"])
        self._drag = 0.08
        self._steps = 0
        self._failed_count = 0

        asset = Path(__file__).resolve().parents[1] / "assets" / "hollow_ball_topology.json"
        topology = json.loads(asset.read_text(encoding="utf-8"))
        faces = np.asarray(topology["faces"], dtype=np.int64)

        # Unique in-surface bonds from the authoritative triangles.
        edge_blocks = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        )
        edge_blocks.sort(axis=1)
        membrane = np.unique(edge_blocks, axis=0)
        membrane_length = np.linalg.norm(
            self._x0[membrane[:, 1]] - self._x0[membrane[:, 0]], axis=1
        )
        membrane = membrane[membrane_length > 1.0e-6]

        # The topology has separate inner and outer skins. Couple corresponding
        # material directions to give the hollow surface physical thickness.
        used = np.unique(faces.reshape(-1))
        center = np.mean(self._x0[used], axis=0)
        local = self._x0[used] - center
        radius = np.linalg.norm(local, axis=1)
        inner = used[radius < 0.56]
        outer = used[radius >= 0.56]
        inner_dir = local[radius < 0.56] / np.maximum(radius[radius < 0.56, None], 1.0e-12)
        outer_dir = local[radius >= 0.56] / np.maximum(radius[radius >= 0.56, None], 1.0e-12)
        similarity = inner_dir @ outer_dir.T
        ties = {(int(i), int(outer[j])) for i, j in zip(inner, np.argmax(similarity, axis=1))}
        ties.update((int(inner[i]), int(j)) for i, j in zip(np.argmax(similarity, axis=0), outer))
        thickness = np.asarray(sorted(ties), dtype=np.int64)

        self._edges = np.concatenate((membrane, thickness), axis=0)
        self._edge_i = self._edges[:, 0]
        self._edge_j = self._edges[:, 1]
        rest_vector = self._x0[self._edge_j] - self._x0[self._edge_i]
        self._rest0 = np.linalg.norm(rest_vector, axis=1)
        self._rest = self._rest0.copy()
        self._alive = np.ones(len(self._edges), dtype=bool)
        self._plastic = np.zeros(len(self._edges), dtype=float)

        is_thickness = np.arange(len(self._edges)) >= len(membrane)
        self._stiffness = np.where(is_thickness, 4600.0, 2800.0)
        effective_mass = 0.5 * (self._mass[self._edge_i] + self._mass[self._edge_j])
        self._damping = 0.18 * 2.0 * np.sqrt(self._stiffness * effective_mass)

        # A shallow material imperfection localizes the tear. It changes bond
        # strength only; motion still follows the integrated force balance.
        midpoint = 0.5 * (self._x0[self._edge_i] + self._x0[self._edge_j]) - center
        weak = (
            (midpoint[:, 0] > 0.08)
            & (midpoint[:, 1] < 0.18)
            & (midpoint[:, 2] < -0.08)
            & (~is_thickness)
        )
        # Thresholds are reached only after the shell has spent time crushed;
        # the weak band then fails before the rest of the still-load-bearing skin.
        self._failure_plastic_strain = np.where(weak, 0.09, 0.24)
        self._prepare_deformation_bases(faces)

    def _prepare_deformation_bases(self, faces: np.ndarray) -> None:
        """Choose two well-separated material neighbors per rendered vertex."""
        neighbors = [set() for _ in range(self._n)]
        for a, b, c in faces:
            neighbors[a].update((int(b), int(c)))
            neighbors[b].update((int(a), int(c)))
            neighbors[c].update((int(a), int(b)))

        self._basis_valid = np.zeros(self._n, dtype=bool)
        self._basis_n1 = np.zeros(self._n, dtype=np.int64)
        self._basis_n2 = np.zeros(self._n, dtype=np.int64)
        self._basis_inverse = np.repeat(np.eye(3)[None, :, :], self._n, axis=0)
        self._basis_normal_length = np.ones(self._n, dtype=float)
        self._basis_area = np.ones(self._n, dtype=float)

        for vertex, adjacent in enumerate(neighbors):
            candidates = list(adjacent)
            if len(candidates) < 2:
                continue
            offsets = self._x0[candidates] - self._x0[vertex]
            best = None
            best_area = 0.0
            for left in range(len(candidates) - 1):
                crosses = np.cross(offsets[left], offsets[left + 1 :])
                areas = np.linalg.norm(crosses, axis=1)
                right_offset = int(np.argmax(areas))
                if areas[right_offset] > best_area:
                    best_area = float(areas[right_offset])
                    best = (left, left + 1 + right_offset)
            if best is None or best_area < 1.0e-10:
                continue
            n1, n2 = candidates[best[0]], candidates[best[1]]
            d1 = self._x0[n1] - self._x0[vertex]
            d2 = self._x0[n2] - self._x0[vertex]
            normal = np.cross(d1, d2) / best_area
            normal_length = np.sqrt(np.linalg.norm(d1) * np.linalg.norm(d2))
            basis = np.column_stack((d1, d2, normal * normal_length))
            self._basis_valid[vertex] = True
            self._basis_n1[vertex] = n1
            self._basis_n2[vertex] = n2
            self._basis_inverse[vertex] = np.linalg.inv(basis)
            self._basis_normal_length[vertex] = normal_length
            self._basis_area[vertex] = best_area

    def _substep(self, dt: float) -> None:
        i, j = self._edge_i, self._edge_j
        delta = self._x[j] - self._x[i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length[:, None], 1.0e-12)
        strain = (length - self._rest) / np.maximum(self._rest, 1.0e-9)

        # Rate-dependent return mapping with bounded permanent rest lengths.
        excess = np.maximum(np.abs(strain) - 0.065, 0.0)
        plastic_increment = 8.0 * excess * np.sign(strain) * self._rest * dt * self._alive
        proposed_rest = np.clip(
            self._rest + plastic_increment, 0.58 * self._rest0, 1.72 * self._rest0
        )
        actual_increment = proposed_rest - self._rest
        self._rest = proposed_rest
        self._plastic += np.abs(actual_increment) / np.maximum(self._rest0, 1.0e-9)

        # Failure follows substantial irreversible tensile deformation.
        newly_failed = self._alive & (strain > 0.14) & (
            self._plastic > self._failure_plastic_strain
        )
        self._alive[newly_failed] = False
        self._failed_count += int(np.count_nonzero(newly_failed))

        relative_speed = np.einsum("ij,ij->i", self._v[j] - self._v[i], direction)
        magnitude = (
            self._stiffness * (length - self._rest) + self._damping * relative_speed
        ) * self._alive
        magnitude = np.clip(magnitude, -2.0e5, 2.0e5)
        spring_force = magnitude[:, None] * direction

        force = self._mass[:, None] * (self._gravity - self._drag * self._v)
        np.add.at(force, i, spring_force)
        np.add.at(force, j, -spring_force)
        self._v += dt * force / self._mass[:, None]
        self._x += dt * self._v

        # Rigid z=0 floor: dissipative unilateral contact and frictional slip.
        penetrating = self._x[:, 2] < 0.0
        if np.any(penetrating):
            impact = penetrating & (self._v[:, 2] < 0.0)
            self._x[penetrating, 2] = 0.0
            incoming = np.abs(self._v[impact, 2]).copy()
            self._v[impact, 2] *= -0.025
            tangential = self._v[impact, :2]
            tangential_speed = np.linalg.norm(tangential, axis=1)
            reduction = np.maximum(
                0.0, 1.0 - 0.22 * incoming / np.maximum(tangential_speed, 1.0e-12)
            )
            self._v[impact, :2] *= reduction[:, None]
            resting = penetrating & (np.abs(self._v[:, 2]) < 0.025)
            self._v[resting, 2] = 0.0

        # Guard against a singular failed-bond event without altering normal motion.
        self._v = np.nan_to_num(self._v, nan=0.0, posinf=40.0, neginf=-40.0)
        np.clip(self._v, -40.0, 40.0, out=self._v)
        self._x = np.nan_to_num(self._x, nan=0.0, posinf=8.0, neginf=-8.0)
        np.clip(self._x, -8.0, 8.0, out=self._x)
        self._steps += 1

    def _deformation_gradient(self) -> np.ndarray:
        gradient = np.repeat(np.eye(3)[None, :, :], self._n, axis=0)
        ids = np.flatnonzero(self._basis_valid)
        if len(ids) == 0:
            return gradient
        d1 = self._x[self._basis_n1[ids]] - self._x[ids]
        d2 = self._x[self._basis_n2[ids]] - self._x[ids]
        cross = np.cross(d1, d2)
        area = np.linalg.norm(cross, axis=1)
        normal = cross / np.maximum(area[:, None], 1.0e-12)
        area_ratio = np.clip(np.sqrt(area / self._basis_area[ids]), 0.18, 3.0)
        third = normal * (self._basis_normal_length[ids] * area_ratio)[:, None]
        current_basis = np.stack((d1, d2, third), axis=2)
        gradient[ids] = current_basis @ self._basis_inverse[ids]
        return np.nan_to_num(gradient, nan=0.0, posinf=6.0, neginf=-6.0)

    def _frame(self, time: float) -> StateFrame:
        external_force = self._mass[:, None] * (self._gravity - self._drag * self._v)
        frame = StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "particle_radius": np.full(self._n, 0.45 * self._spacing),
                "material_label": self._body_id.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._x[:, 2].copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self._spacing,
                "grid_spacing": self._spacing,
                "characteristic_length": max(
                    float(np.ptp(self._x0, axis=0).max()), self._spacing
                ),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1.0e-10:
            raise ValueError("advance_to requires finite, monotonically increasing times")
        maximum_dt = 1.0 / 600.0
        while self._time < target - 1.0e-12:
            dt = min(maximum_dt, target - self._time)
            self._substep(dt)
            self._time += dt
        self._time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped_elastoplastic_mass_spring_shell",
            "substeps": int(self._steps),
            "failed_bonds": int(self._failed_count),
            "remaining_bonds": int(np.count_nonzero(self._alive)),
        }
