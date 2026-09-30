"""Particle solver for repeated topological merging."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.scene = scene
        self.seed = int(seed)
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self.base_attributes = {k: np.asarray(v, dtype=np.float64).copy() for k, v in frame.attributes.items()}
        self.n = len(self.position)
        self.radius = float(np.median(self.base_attributes.get("particle_radius", [0.112])))
        self.spacing = float(frame.scalars.get("particle_spacing", 0.205))
        self.rest_density = float(frame.scalars.get("rest_density", 1050.0))
        self.gravity = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.time = 0.0
        self.deformation_gradient = np.broadcast_to(np.eye(3), (self.n, 3, 3)).copy()
        self.interaction = np.zeros((self.n, 3), dtype=np.float64)
        self.external = self.mass[:, None] * self.gravity[None, :]
        self.cumulative_source = np.zeros(1, dtype=np.float64)
        self.initialized = True

    def _forces(self):
        n = self.n
        force = np.zeros((n, 3), dtype=np.float64)
        grad = np.zeros((n, 3, 3), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        cell_size = self.spacing * 1.75
        cells = defaultdict(list)
        keys = np.floor(self.position / cell_size).astype(np.int64)
        for i, key in enumerate(keys):
            cells[(int(key[0]), int(key[1]), int(key[2]))].append(i)
        cutoff = self.spacing * 1.72
        inv_r0 = 1.0 / self.spacing
        stiffness, pair_damping = 1150.0, 38.0
        for i in range(n):
            ci, pi, vi = keys[i], self.position[i], self.velocity[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((int(ci[0] + dx), int(ci[1] + dy), int(ci[2] + dz)), ()):
                            if j <= i:
                                continue
                            dvec = self.position[j] - pi
                            dist2 = float(np.dot(dvec, dvec))
                            if dist2 <= 1.0e-12 or dist2 >= cutoff * cutoff:
                                continue
                            dist = float(np.sqrt(dist2))
                            unit = dvec / dist
                            extension = dist - self.spacing
                            k = stiffness if extension < 0.0 else stiffness * 0.52
                            reln = float(np.dot(self.velocity[j] - vi, unit))
                            fmag = float(np.clip(k * extension + pair_damping * reln, -420.0, 260.0))
                            fij = fmag * unit
                            force[i] += fij
                            force[j] -= fij
                            dv = self.velocity[j] - vi
                            outer = np.outer(dv, dvec) * (inv_r0 * inv_r0)
                            grad[i] += outer
                            grad[j] += outer
                            counts[i] += 1.0
                            counts[j] += 1.0
        mask = counts > 0
        grad[mask] /= counts[mask, None, None]
        return force, grad

    def _step(self, dt: float) -> None:
        pair_force, grad = self._forces()
        self.external = self.mass[:, None] * self.gravity[None, :]
        self.velocity += dt * (self.external + pair_force) / self.mass[:, None]
        self.velocity *= np.exp(-0.42 * dt)
        self.position += dt * self.velocity
        floor = self.radius
        below = self.position[:, 2] < floor
        if np.any(below):
            self.position[below, 2] = floor
            vz = self.velocity[below, 2]
            vz[vz < 0.0] *= -0.12
            self.velocity[below, 2] = vz
            self.velocity[below, :2] *= 0.84
            pair_force[below, 2] += np.maximum(0.0, -self.mass[below] * self.gravity[2])
        self.deformation_gradient += dt * np.einsum("nij,njk->nik", grad, self.deformation_gradient)
        self.deformation_gradient = np.clip(self.deformation_gradient, -3.0, 3.0)
        self.interaction = pair_force

    def _frame(self) -> StateFrame:
        attrs = dict(self.base_attributes)
        attrs["deformation_gradient"] = self.deformation_gradient.copy()
        attrs["mass_quantity"] = self.mass.copy()
        fields = {
            "cumulative_mass_source": self.cumulative_source.copy(),
            "external_force": self.external.copy(),
            "interaction_force": self.interaction.copy(),
        }
        out = StateFrame(time=float(self.time), position=self.position.copy(), velocity=self.velocity.copy(),
                         mass=self.mass.copy(), body_id=self.body_id.copy(), attributes=attrs, fields=fields,
                         scalars={"gravity_x": float(self.gravity[0]), "gravity_y": float(self.gravity[1]),
                                  "gravity_z": float(self.gravity[2]), "rest_density": self.rest_density,
                                  "particle_spacing": self.spacing})
        out.validate()
        return out

    def advance_to(self, time: float):
        if not self.initialized:
            raise RuntimeError("BenchmarkSolver.initialize must be called first")
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite non-decreasing time")
        while self.time + 1.0e-12 < target:
            dt = min(1.0 / 120.0, target - self.time)
            self._step(dt)
            self.time += dt
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        if not self.initialized:
            return {}
        return {"solver": "hashed_cohesive_viscoplastic_particles", "particle_count": int(self.n),
                "simulated_time": float(self.time)}
