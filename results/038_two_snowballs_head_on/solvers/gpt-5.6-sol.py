"""Deterministic particle-based head-on snowball collision solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, float).copy()
        self.velocity = np.asarray(initial.velocity, float).copy()
        self.mass = np.asarray(initial.mass, float).copy()
        self.body_id = np.asarray(initial.body_id, np.int64).copy()
        self.reference_position = self.position.copy()
        self.time = 0.0
        self.steps = 0
        self.gravity = np.asarray(self.cfg["gravity"], float)
        snow = self.cfg["snowballs"]
        self.radius = float(snow["radius"])
        self.floor_z = float(self.cfg["floor"]["top_z"])
        self.floor_xy = np.asarray(self.cfg["floor"]["half_extents"], float)[:2]
        off, height = float(snow["center_offset_x"]), float(snow["center_height"])
        self.initial_centers = np.array([[-off, 0., height], [off, 0., height]])
        self.local_reference = self.reference_position - self.initial_centers[self.body_id]
        rng = np.random.default_rng(int(seed))
        self.failure_threshold = 0.10 + 0.055 * rng.random(len(self.position))
        self.damage = np.zeros(len(self.position), float)
        self.last_overlap = 0.0

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-10:
            raise ValueError("advance_to requires finite non-decreasing time")
        while self.time < target - 1e-12:
            h = min(0.004, target - self.time)
            self._step(h)
            self.time += h
        return self._frame(target)

    def _centers(self):
        return np.array([np.average(self.position[self.body_id == b], axis=0,
                                    weights=self.mass[self.body_id == b]) for b in (0, 1)])

    def _step(self, h):
        n = len(self.position)
        acc = np.broadcast_to(self.gravity, (n, 3)).copy()
        centers = self._centers()
        target = centers[self.body_id] + self.local_reference
        k = 115.0 * np.square(1.0 - self.damage)
        acc += (k[:, None] * (target - self.position) - 7.5 * self.velocity) / self.mass[:, None]
        delta = centers[1] - centers[0]
        dist = float(np.linalg.norm(delta))
        normal = delta / dist if dist > 1e-9 else np.array([1., 0., 0.])
        overlap = max(0.0, 2.0 * self.radius - dist)
        self.last_overlap = overlap
        if overlap > 0.0:
            vc = np.array([np.average(self.velocity[self.body_id == b], axis=0,
                                       weights=self.mass[self.body_id == b]) for b in (0, 1)])
            closing = max(0.0, -float(np.dot(vc[1] - vc[0], normal)))
            force_mag = 1850.0 * overlap + 95.0 * closing
            for b, sign in ((0, -1.), (1, 1.)):
                mask = self.body_id == b
                radial = self.position[mask] - centers[b]
                axial = radial @ normal
                # The contact patch is the side pointing toward the other
                # ball: opposite to the direction in which this ball is
                # accelerated.
                facing = np.maximum(0., -sign * axial)
                transverse2 = np.sum(radial * radial, axis=1) - axial * axial
                weights = facing * np.exp(-transverse2 / (0.32 * self.radius) ** 2)
                if weights.sum() < 1e-12: weights = np.ones(mask.sum())
                acc[mask] += ((sign * force_mag * weights / weights.sum())[:, None] * normal) / self.mass[mask, None]
                stress = (overlap / self.radius) * (0.35 + 0.65 * weights / (weights.max() + 1e-12))
                ids = np.flatnonzero(mask)
                self.damage[ids] += h * 2.8 * np.maximum(stress - self.failure_threshold[ids], 0.)
        self.velocity += h * acc
        self.position += h * self.velocity
        below = self.position[:, 2] < self.floor_z
        if np.any(below):
            self.position[below, 2] = self.floor_z
            vz = self.velocity[below, 2]
            self.velocity[below, 2] = np.where(vz < 0., -0.24 * vz, vz)
            self.velocity[below, :2] *= 0.82
        for axis, limit in ((0, self.floor_xy[0]), (1, self.floor_xy[1])):
            lo, hi = self.position[:, axis] < -limit, self.position[:, axis] > limit
            if np.any(lo): self.position[lo, axis], self.velocity[lo, axis] = -limit, np.abs(self.velocity[lo, axis]) * 0.35
            if np.any(hi): self.position[hi, axis], self.velocity[hi, axis] = limit, -np.abs(self.velocity[hi, axis]) * 0.35
        self.damage = np.clip(self.damage, 0., 1.)
        self.steps += 1

    def _frame(self, time):
        centers = self._centers()
        rel, local = self.position - centers[self.body_id], self.local_reference
        denom = np.sum(local * local, axis=1) + 0.015 ** 2
        F = np.tile(np.eye(3), (len(self.position), 1, 1))
        F += np.einsum("ni,nj->nij", rel - local, local) / denom[:, None, None]
        F = np.clip(F, -2., 3.)
        massq = self.mass.copy()
        frame = StateFrame(time=float(time), position=self.position.copy(), velocity=self.velocity.copy(),
                           mass=massq, body_id=self.body_id.copy(),
                           attributes={"mass_quantity": massq, "deformation_gradient": F,
                                       "initial_component_id": self.body_id.copy()},
                           fields={"external_force": massq[:, None] * self.gravity[None, :],
                                   "cumulative_mass_source": np.zeros(1),
                                   "floor_signed_distance": self.position[:, 2] - self.floor_z},
                           scalars={"initial_total_mass": float(self.mass.sum()),
                                    "initial_mass_each": float(self.cfg["snowballs"]["total_mass_each"]),
                                    "particle_spacing": float(self.cfg["snowballs"]["particle_spacing"])})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler_particle_snow", "particle_count": int(len(self.position)),
                "substeps": int(self.steps), "max_damage": float(self.damage.max()),
                "max_contact_overlap": float(self.last_overlap)}
