"""Deterministic particle/rod solver for the noodles-in-bowl scene."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-implicit point-mass rods with projected finite contact."""

    def initialize(self, scene, seed: int) -> None:
        self.config = dict(scene.config)
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.n = int(self.position.shape[0])
        self.gravity = np.asarray(self.config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.radius = float(self.config.get("radius", 0.006))
        self.rest_length = float(self.config.get("noodle_length", 0.4))
        bowl = self.config.get("bowl", {})
        self.rim_radius = float(bowl.get("rim_radius", 0.72))
        self.bowl_height = float(bowl.get("height", 0.48))
        self.mass = (np.asarray(initial.mass, dtype=float).copy()
                     if initial.mass is not None else
                     np.asarray(initial.attributes.get("mass_quantity",
                                                       np.full(self.n, 0.00125)), dtype=float).copy())
        if self.mass.ndim != 1 or len(self.mass) != self.n:
            self.mass = np.full(self.n, 0.00125, dtype=float)
        self.mass = np.maximum(self.mass, 1.0e-9)
        self.inv_mass = 1.0 / self.mass
        self.mass_quantity = self.mass.copy()
        self.edges = np.arange(self.n, dtype=int).reshape(-1, 2)
        self.time = 0.0
        self.seed = int(seed)
        self._last_contact = np.full(self.n, 0.01, dtype=float)

    def _inner_radius(self, z: np.ndarray) -> np.ndarray:
        """Piecewise-linear interior profile from the frozen bowl geometry."""
        return np.interp(np.asarray(z), (0.0, 0.08, 0.34, self.bowl_height),
                         (0.12, 0.38, 0.66, self.rim_radius))

    def _bowl_project(self) -> None:
        p = self.position
        radial = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        below = p[:, 2] < self.radius
        p[below, 2] = self.radius
        # The container is open above its rim.
        low = (p[:, 2] >= self.radius) & (p[:, 2] <= self.bowl_height)
        limit = np.maximum(self._inner_radius(np.clip(p[:, 2], 0.0, self.bowl_height)) - self.radius, 1e-5)
        outside = low & (radial > limit)
        if np.any(outside):
            scale = limit[outside] / np.maximum(radial[outside], 1e-12)
            p[outside, 0] *= scale
            p[outside, 1] *= scale

    def _rod_project(self) -> None:
        for _ in range(3):
            a, b = self.edges[:, 0], self.edges[:, 1]
            delta = self.position[b] - self.position[a]
            length = np.linalg.norm(delta, axis=1)
            good = length > 1.0e-10
            unit = np.zeros_like(delta)
            unit[good] = delta[good] / length[good, None]
            wa, wb = self.inv_mass[a], self.inv_mass[b]
            correction = ((length - self.rest_length) / np.maximum(wa + wb, 1e-12))[:, None] * unit
            self.position[a] += wa[:, None] * correction
            self.position[b] -= wb[:, None] * correction

    def _self_contact(self) -> None:
        """Repel nearby noodle nodes with a deterministic spatial hash."""
        cell_size = 2.2 * self.radius
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        keys = np.floor(self.position / cell_size).astype(np.int64)
        for i, key in enumerate(keys):
            cells[(int(key[0]), int(key[1]), int(key[2]))].append(i)
        minimum = 2.0 * self.radius
        minimum2 = minimum * minimum
        for key, members in cells.items():
            kx, ky, kz = key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        other = cells.get((kx + dx, ky + dy, kz + dz))
                        if not other:
                            continue
                        for i in members:
                            for j in other:
                                if j <= i:
                                    continue
                                delta = self.position[j] - self.position[i]
                                d2 = float(np.dot(delta, delta))
                                if d2 >= minimum2:
                                    continue
                                if d2 < 1.0e-16:
                                    direction = (np.array((1.0, 0.0, 0.0)) if ((i + j) & 1) == 0
                                                 else np.array((0.0, 1.0, 0.0)))
                                    distance = 0.0
                                else:
                                    distance = math.sqrt(d2)
                                    direction = delta / distance
                                amount = 0.5 * (minimum - distance)
                                wi, wj = self.inv_mass[i], self.inv_mass[j]
                                total = wi + wj
                                self.position[i] -= direction * amount * wi / total
                                self.position[j] += direction * amount * wj / total

    def _contact_distances(self) -> np.ndarray:
        p = self.position
        radial = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        bowl_gap = np.full(self.n, self.rim_radius, dtype=float)
        low = p[:, 2] <= self.bowl_height
        profile = self._inner_radius(np.clip(p[:, 2], 0.0, self.bowl_height))
        bowl_gap[low] = profile[low] - self.radius - radial[low]
        bowl_gap = np.minimum(bowl_gap, p[:, 2] - self.radius)
        nearest = np.full(self.n, np.inf, dtype=float)
        cell_size = 2.2 * self.radius
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        keys = np.floor(p / cell_size).astype(np.int64)
        for i, key in enumerate(keys):
            cells[(int(key[0]), int(key[1]), int(key[2]))].append(i)
        for key, members in cells.items():
            kx, ky, kz = key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for i in members:
                            for j in cells.get((kx + dx, ky + dy, kz + dz), ()):
                                if i != j:
                                    gap = float(np.linalg.norm(p[j] - p[i])) - 2.0 * self.radius
                                    if gap < nearest[i]:
                                        nearest[i] = gap
        nearest[~np.isfinite(nearest)] = self.rim_radius
        return np.minimum(bowl_gap, nearest)

    def _substep(self, h: float) -> None:
        old = self.position.copy()
        self.velocity += self.gravity[None, :] * h
        self.velocity *= math.exp(-1.15 * h)
        self.position += self.velocity * h
        self._bowl_project()
        self._self_contact()
        self._rod_project()
        self._bowl_project()
        self.velocity = (self.position - old) / h
        self.velocity *= math.exp(-0.35 * h)
        self.velocity = np.clip(self.velocity, -25.0, 25.0)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time:
            target = self.time
        while self.time < target - 1.0e-12:
            h = min(0.008, target - self.time)
            self._substep(h)
            self.time += h
        contact = self._contact_distances()
        self._last_contact = contact
        frame = StateFrame(
            self.time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass_quantity.copy()},
            fields={
                "contact_signed_distance": contact.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.radius)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_position_projection",
            "nodes": self.n,
            "rods": int(len(self.edges)),
            "time": float(self.time),
            "min_contact_gap": float(np.min(self._last_contact)),
        }
