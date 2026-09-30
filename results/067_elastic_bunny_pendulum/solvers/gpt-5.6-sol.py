"""Deterministic elastic-body solver for the suspended bunny scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _load_faces(path: Path) -> np.ndarray:
    faces = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("f "):
                ids = [int(word.split("/", 1)[0]) - 1 for word in line.split()[1:]]
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
    if not faces:
        raise ValueError("Bunny asset has no faces")
    return np.asarray(faces, dtype=np.int64)

class BenchmarkSolver:
    """Lumped-mass, corotational position-based solver with a clamped crown."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._time = float(initial.time)
        self._x = np.asarray(initial.position, dtype=np.float64).copy()
        self._v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self._rest = self._x.copy()
        self._mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self._fixed = np.asarray(initial.attributes["fixed"], dtype=np.float64) > 0.5
        self._dynamic = ~self._fixed
        self._gravity = np.asarray(scene.metadata().get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        asset = Path(__file__).resolve().parent.parent / "assets" / "bunny.obj"
        self._faces = _load_faces(asset)
        raw_edges = np.concatenate((self._faces[:, (0, 1)], self._faces[:, (1, 2)], self._faces[:, (2, 0)]), axis=0)
        raw_edges.sort(axis=1)
        self._edges = np.unique(raw_edges, axis=0)
        # The regular subsample keeps the broad mesh connectivity while bounding
        # per-step work for the high-resolution source surface.
        stride = max(1, len(self._edges) // 14000)
        self._edges = self._edges[::stride]
        self._edge_i, self._edge_j = self._edges[:, 0], self._edges[:, 1]
        rest_delta = self._rest[self._edge_j] - self._rest[self._edge_i]
        self._edge_length = np.linalg.norm(rest_delta, axis=1)
        self._inv_mass = np.zeros_like(self._mass)
        self._inv_mass[self._dynamic] = 1.0 / self._mass[self._dynamic]

        fit_weight = self._mass.copy()
        fit_weight[self._fixed] *= 80.0
        self._fit_weight = fit_weight / fit_weight.sum()
        self._rest_center = np.sum(self._fit_weight[:, None] * self._rest, axis=0)
        self._rest_offset = self._rest - self._rest_center

        ne = min(768, len(self._faces))
        chosen = np.linspace(0, len(self._faces) - 1, ne, dtype=np.int64)
        self._element_faces = self._faces[chosen]
        tri = self._rest[self._element_faces]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        safe = np.maximum(norm, 1.0e-12)
        self._element_normal_scale = np.sqrt(safe)
        e3 = normal / safe[:, None] * self._element_normal_scale[:, None]
        self._element_dm_inverse = np.linalg.pinv(np.stack((e1, e2, e3), axis=2))

        self._total_mass = float(self._mass.sum())
        self._fixed_mass = float(self._mass[self._fixed].sum())
        self._cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self._last_support_force = -self._fixed_mass * self._gravity
        self._step_count = 0
        extent = np.ptp(self._rest, axis=0)
        length_scale = max(float(np.linalg.norm(extent)), 1.0e-6)
        self._reference_momentum = max(self._total_mass * np.sqrt(np.linalg.norm(self._gravity) * length_scale), 1.0e-8)

    @staticmethod
    def _proper_rotation(covariance: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation

    def _constraint_sweep(self, position: np.ndarray) -> None:
        delta = position[self._edge_j] - position[self._edge_i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1.0e-12)[:, None]
        error = length - self._edge_length
        wi, wj = self._inv_mass[self._edge_i], self._inv_mass[self._edge_j]
        denom = wi + wj
        multiplier = np.zeros_like(error)
        active = denom > 0.0
        multiplier[active] = -error[active] / denom[active]
        edge_push = multiplier[:, None] * direction
        correction = np.zeros_like(position)
        np.add.at(correction, self._edge_i, -wi[:, None] * edge_push)
        np.add.at(correction, self._edge_j, wj[:, None] * edge_push)
        position += 0.12 * correction

        center = np.sum(self._fit_weight[:, None] * position, axis=0)
        current_offset = position - center
        covariance = (current_offset * self._fit_weight[:, None]).T @ self._rest_offset
        rotation = self._proper_rotation(covariance)
        target = center + self._rest_offset @ rotation.T
        position[self._dynamic] += 0.055 * (target[self._dynamic] - position[self._dynamic])
        position[self._fixed] = self._rest[self._fixed]

    def _substep(self, h: float) -> np.ndarray:
        old_position = self._x.copy()
        predicted_velocity = self._v.copy()
        predicted_velocity[self._dynamic] += h * self._gravity
        position = self._x + h * predicted_velocity
        position[self._fixed] = self._rest[self._fixed]
        for _ in range(3):
            self._constraint_sweep(position)
        new_velocity = (position - old_position) / h
        new_velocity[self._fixed] = 0.0
        impulse = np.sum(self._mass[self._dynamic, None] * (new_velocity[self._dynamic] - predicted_velocity[self._dynamic]), axis=0)
        self._x, self._v = position, new_velocity
        self._step_count += 1
        return impulse

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1.0e-12:
            raise ValueError("advance_to requires finite, monotonically increasing times")
        interval = target - self._time
        support_impulse = np.zeros(3, dtype=np.float64)
        if interval > 1.0e-12:
            substeps = max(1, int(np.ceil(interval / 0.01)))
            h = interval / substeps
            for _ in range(substeps):
                support_impulse += self._substep(h) - self._fixed_mass * self._gravity * h
            self._cumulative_external_impulse[0] += self._total_mass * self._gravity * interval + support_impulse
            self._last_support_force = support_impulse / interval
            self._time = target

        gravity_force = self._total_mass * self._gravity
        external_force = np.stack((gravity_force, self._last_support_force), axis=0)[None, :, :]
        n = len(self._x)
        frame = StateFrame(
            time=self._time,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={"fixed": self._fixed.astype(np.float64), "mass_quantity": self._mass.copy()},
            fields={
                "budget_cumulative_external_impulse": self._cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": np.ones((1, n), dtype=np.float64),
                "budget_external_force": external_force,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": self._deformation_gradients(),
            },
            scalars={"density_kg_m3": 1000.0, "reference_momentum": self._reference_momentum},
        )
        frame.validate()
        return frame

    def _deformation_gradients(self) -> np.ndarray:
        tri = self._x[self._element_faces]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        e3 = normal / np.maximum(norm, 1.0e-12)[:, None] * self._element_normal_scale[:, None]
        return np.stack((e1, e2, e3), axis=2) @ self._element_dm_inverse

    def diagnostics(self) -> dict[str, object]:
        return {"method": "corotational position-based elastic solid", "internal_steps": self._step_count,
                "fixed_vertices": int(np.count_nonzero(self._fixed)), "elastic_edges": int(len(self._edges)),
                "diagnostic_elements": int(len(self._element_faces))}
