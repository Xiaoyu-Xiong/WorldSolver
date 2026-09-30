"""Deterministic viscoplastic bunny solver for the staircase scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = dict(scene.config)
        self.x0 = np.asarray(scene.vertices, dtype=float).copy()
        self.x = self.x0.copy()
        self.n = len(self.x)
        self.v = np.tile(np.asarray(self.cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float), (self.n, 1))
        self.mass = np.full(self.n, float(self.cfg.get("subject_mass", 1.0)) / self.n)
        self.g = np.asarray(self.cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.k0 = float(self.cfg.get("stiffness", 22.0))
        self.damping = float(self.cfg.get("damping", 2.3))
        self.soft_rate = float(self.cfg.get("softening_rate", 0.34))
        self.restitution = float(self.cfg.get("restitution", 0.12))
        self.friction = float(self.cfg.get("friction", 0.72))
        self.contact_scale = float(self.cfg.get("contact_length_scale", 0.04))
        self.time = 0.0
        self.substep = min(0.004, float(self.cfg.get("sample_dt", 1.0)) / 8.0)
        asset = Path(__file__).resolve().parents[1] / self.cfg["subject_asset"]
        with np.load(asset) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
        pairs = np.concatenate((faces[:, :2], faces[:, 1:3], faces[:, ::2]), axis=0)
        pairs.sort(axis=1)
        self.edges = np.unique(pairs, axis=0)
        self.ei, self.ej = self.edges[:, 0], self.edges[:, 1]
        self.rest = np.linalg.norm(self.x0[self.ej] - self.x0[self.ei], axis=1).clip(1e-5)
        self.edge_damage = np.zeros(len(self.edges))
        self.damage = np.zeros(self.n)
        self.prev_contact = np.zeros((self.n, 3))
        self.boxes = [
            (np.array([0.0, 0.0, -0.04]), np.array([3.0, 1.6, 0.04])),
            (np.array([-0.75, 0.0, 0.96]), np.array([0.36, 1.05, 0.14])),
            (np.array([-0.15, 0.0, 0.68]), np.array([0.36, 1.05, 0.14])),
            (np.array([0.45, 0.0, 0.40]), np.array([0.36, 1.05, 0.14])),
            (np.array([1.05, 0.0, 0.12]), np.array([0.36, 1.05, 0.14])),
        ]
        self.initialized = True

    def _contacts(self, x, v):
        force = np.zeros_like(x)
        gap = np.full(self.n, np.inf)
        for center, half in self.boxes:
            q = np.abs(x - center) - half
            gap = np.minimum(gap, np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0))
            inside = (np.abs(x[:, 0] - center[0]) <= half[0] + self.contact_scale) & (np.abs(x[:, 1] - center[1]) <= half[1] + self.contact_scale)
            top = center[2] + half[2]
            penetration = np.maximum(top - x[:, 2], 0.0) * inside
            normal = np.clip(850.0 * penetration - 18.0 * np.minimum(v[:, 2], 0.0), 0.0, 700.0)
            force[:, 2] += normal
            force[:, :2] -= self.friction * normal[:, None] * np.tanh(v[:, :2] / 0.08)
            zinside = (x[:, 2] >= center[2] - half[2] - self.contact_scale) & (x[:, 2] <= top + self.contact_scale)
            for side, sign in ((center[0] - half[0], 1.0), (center[0] + half[0], -1.0)):
                pen = np.maximum((side - x[:, 0]) * sign, 0.0) * zinside
                rf = np.clip(420.0 * pen - 10.0 * v[:, 0] * sign, 0.0, 450.0)
                force[:, 0] += sign * rf
        return force, gap

    def _step(self, dt):
        a, b = self.ei, self.ej
        d = self.x[b] - self.x[a]
        length = np.linalg.norm(d, axis=1).clip(1e-6)
        nvec = d / length[:, None]
        rate = np.sum((self.v[b] - self.v[a]) * nvec, axis=1)
        strain = (length - self.rest) / np.maximum(self.rest, 1e-5)
        excess = np.maximum(np.abs(strain) - 0.075, 0.0)
        self.edge_damage = np.clip(self.edge_damage + (0.9 * excess + 0.1 * self.edge_damage) * dt, 0.0, 5.0)
        self.rest *= np.exp(np.clip(strain, -0.12, 0.12) * (0.8 * excess) * dt)
        k = self.k0 * np.exp(-self.soft_rate * (self.edge_damage + 0.35 * self.damage[a] + 0.35 * self.damage[b]))
        ef = (k * (length - self.rest) + self.damping * rate)[:, None] * nvec
        f = np.zeros_like(self.x)
        np.add.at(f, a, ef)
        np.add.at(f, b, -ef)
        cf, _ = self._contacts(self.x, self.v)
        f += cf + self.mass[:, None] * self.g
        work = np.maximum(0.0, -np.sum(cf * self.v, axis=1)) * dt
        self.damage = np.clip(self.damage + self.soft_rate * (work / (self.mass * 8.0) + excess.mean() * dt), 0.0, 5.0)
        self.prev_contact = cf
        self.v += dt * f / self.mass[:, None]
        self.v *= 1.0 / (1.0 + 0.015 * dt)
        self.x += dt * self.v
        for center, half in self.boxes:
            inside = (np.abs(self.x[:, 0] - center[0]) <= half[0] + self.contact_scale) & (np.abs(self.x[:, 1] - center[1]) <= half[1] + self.contact_scale)
            hit = inside & (self.x[:, 2] < center[2] + half[2])
            self.x[hit, 2] = center[2] + half[2]
            down = hit & (self.v[:, 2] < 0.0)
            self.v[down, 2] *= -self.restitution
            self.v[hit, :2] *= max(0.0, 1.0 - 0.18 * self.friction)

    def _frame(self):
        rv = self.x0[self.ej] - self.x0[self.ei]
        cv = self.x[self.ej] - self.x[self.ei]
        num = np.zeros((self.n, 3, 3)); den = np.zeros((self.n, 3, 3))
        for idx, (i, j) in enumerate(self.edges):
            r, c = rv[idx], cv[idx]
            num[i] += np.outer(c, r); num[j] += np.outer(c, r)
            den[i] += np.outer(r, r); den[j] += np.outer(r, r)
        F = np.empty_like(num)
        for i in range(self.n):
            F[i] = num[i] @ np.linalg.inv(den[i] + 1e-5 * np.eye(3))
        F = np.nan_to_num(F, nan=0.0, posinf=3.0, neginf=-3.0)
        F += np.eye(3)[None] * (np.linalg.norm(F, axis=(1, 2)) < 1e-8)[:, None, None]
        _, gap = self._contacts(self.x, self.v)
        return StateFrame(time=float(self.time), position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(), body_id=np.zeros(self.n, dtype=np.int64), attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": F}, fields={"cumulative_mass_source": np.zeros(1), "interaction_force": np.sum(self.prev_contact, axis=0, keepdims=True), "contact_signed_distance": gap}, scalars={"contact_length_scale": self.contact_scale})

    def advance_to(self, time: float):
        if not self.initialized:
            raise RuntimeError("solver is not initialized")
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-9:
            raise ValueError("advance_to requires finite nondecreasing time")
        while self.time < target - 1e-10:
            dt = min(self.substep, target - self.time)
            self._step(dt)
            self.time += dt
        frame = self._frame()
        frame.validate()
        return frame

    def diagnostics(self):
        return {"solver": "deterministic_viscoplastic_mass_spring", "vertex_count": int(self.n), "edge_count": int(len(self.edges))}
