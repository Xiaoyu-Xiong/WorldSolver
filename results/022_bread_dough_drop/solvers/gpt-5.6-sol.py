"""Deterministic edge-spring viscoplastic solver for the bread dough drop."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        self.x0 = np.asarray(scene.vertices, dtype=float).copy()
        self.x = self.x0.copy()
        self.n = len(self.x)
        self.gravity = np.asarray(self.cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.mass = np.full(self.n, float(self.cfg.get("subject_mass", 1.0)) / max(self.n, 1), dtype=float)
        self.v = np.tile(np.asarray(self.cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float), (self.n, 1))
        asset = Path(__file__).resolve().parents[1] / self.cfg["subject_asset"]
        with np.load(asset) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
        pairs = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
        pairs.sort(axis=1)
        self.edges = np.unique(pairs, axis=0)
        i, j = self.edges[:, 0], self.edges[:, 1]
        self.rest_length = np.maximum(np.linalg.norm(self.x0[j] - self.x0[i], axis=1), 1.0e-7)
        self.spring_k = 140.0 * max(float(self.cfg.get("stiffness", 5.0)), 0.05)
        self.edge_damping = max(0.15, 0.16 * float(self.cfg.get("damping", 3.4)))
        self.plasticity = float(np.clip(self.cfg.get("plasticity", 0.86), 0.0, 1.0))
        self.restitution = float(np.clip(self.cfg.get("restitution", 0.02), 0.0, 1.0))
        self.friction = max(0.0, float(self.cfg.get("friction", 0.82)))
        self.contact_k, self.contact_damping = 1800.0, 7.0
        self.contact_radius = max(0.001, 0.35 * float(self.cfg.get("contact_length_scale", 0.04)))
        self.neighbors = [[] for _ in range(self.n)]
        for a, b in self.edges:
            self.neighbors[int(a)].append(int(b))
            self.neighbors[int(b)].append(int(a))
        self.local_inv = np.zeros((self.n, 3, 3), dtype=float)
        eye = np.eye(3)
        for a in range(self.n):
            dr = self.x0[self.neighbors[a]] - self.x0[a]
            self.local_inv[a] = np.linalg.inv(dr.T @ dr + 1.0e-8 * eye) if len(dr) else eye
        self.time = 0.0
        self._last_frame = self._make_frame(0.0)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time:
            raise ValueError("advance_to requires a finite nondecreasing time")
        while self.time + 1.0e-12 < target:
            dt = min(1.0 / 2400.0, target - self.time)
            self._step(dt)
            self.time += dt
        self._last_frame = self._make_frame(target)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi-implicit-euler-edge-spring-plasticity", "particle_count": int(self.n), "edge_count": int(len(self.edges)), "time": float(self.time)}

    def _forces(self):
        force = self.mass[:, None] * self.gravity[None, :]
        i, j = self.edges[:, 0], self.edges[:, 1]
        delta = self.x[j] - self.x[i]
        length = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(length[:, None], 1.0e-9)
        rate = np.sum((self.v[j] - self.v[i]) * unit, axis=1)
        ef = (self.spring_k * (length - self.rest_length) + self.edge_damping * rate)[:, None] * unit
        np.add.at(force, i, ef)
        np.add.at(force, j, -ef)
        gap = self.x[:, 2] - self.ground_z
        active = gap < self.contact_radius
        if np.any(active):
            depth = self.contact_radius - gap[active]
            normal = np.maximum(self.contact_k * depth - self.contact_damping * self.v[active, 2], 0.0)
            force[active, 2] += normal
            tangent = self.v[active, :2]
            speed = np.linalg.norm(tangent, axis=1)
            drag = np.minimum(self.friction * normal / np.maximum(speed, 1.0e-8), self.contact_damping)
            force[active, :2] -= drag[:, None] * tangent
        return force, gap

    def _step(self, dt):
        force, _ = self._forces()
        self.v += dt * force / self.mass[:, None]
        self.x += dt * self.v
        gap = self.x[:, 2] - self.ground_z
        hit = gap < 0.0
        if np.any(hit):
            self.x[hit, 2] = self.ground_z
            downward = self.v[hit, 2] < 0.0
            self.v[hit, 2] = np.where(downward, -self.restitution * self.v[hit, 2], self.v[hit, 2])
            self.v[hit, :2] *= max(0.0, 1.0 - min(1.0, self.friction * dt * 5.0))
        i, j = self.edges[:, 0], self.edges[:, 1]
        current = np.linalg.norm(self.x[j] - self.x[i], axis=1)
        strain = (current - self.rest_length) / np.maximum(self.rest_length, 1.0e-7)
        active = np.abs(strain) > 0.045
        if np.any(active):
            relax = min(1.0, self.plasticity * dt * 3.2)
            self.rest_length[active] = (1.0 - relax) * self.rest_length[active] + relax * current[active]
        self.v *= np.exp(-0.035 * dt)

    def _deformation_gradient(self):
        F = np.empty((self.n, 3, 3), dtype=float)
        for a in range(self.n):
            nb = self.neighbors[a]
            if nb:
                dr = self.x0[nb] - self.x0[a]
                dx = self.x[nb] - self.x[a]
                F[a] = (dx.T @ dr) @ self.local_inv[a]
            else:
                F[a] = np.eye(3)
        return np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)

    def _make_frame(self, time):
        _, gap = self._forces()
        frame = StateFrame(
            time=float(time), position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": self._deformation_gradient()},
            fields={"cumulative_mass_source": np.zeros(1, dtype=float), "external_force": self.mass[:, None] * self.gravity[None, :], "contact_signed_distance": np.asarray(gap, dtype=float)},
            scalars={"contact_length_scale": float(self.cfg.get("contact_length_scale", 0.04))},
        )
        frame.validate()
        return frame
