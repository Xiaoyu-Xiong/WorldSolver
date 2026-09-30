"""Deterministic time-stepped deformable sphere and ground contact solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.cfg = {}
        self.time = 0.0
        self.position = np.empty((0, 3), float)
        self.velocity = np.empty((0, 3), float)
        self.rest = np.empty((0, 3), float)
        self.mass = np.empty(0, float)
        self.gravity = np.zeros(3, float)
        self.ground_z = 0.0
        self.restitution = 0.62
        self.friction = 0.42
        self.contact_length = 0.04
        self.steps = 0

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.cfg = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, float).copy()
        self.velocity = np.asarray(initial.velocity, float).copy()
        center = self.position.mean(axis=0)
        self.rest = self.position - center[None, :]
        n = len(self.position)
        self.mass = np.full(n, float(self.cfg.get("subject_mass", 2.0)) / max(n, 1))
        self.gravity = np.asarray(self.cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.restitution = float(self.cfg.get("restitution", 0.62))
        self.friction = float(self.cfg.get("friction", 0.42))
        self.contact_length = float(self.cfg.get("contact_length_scale", 0.04))
        self.time = 0.0
        self.steps = 0

    def _substep(self, dt: float) -> None:
        n = len(self.position)
        center = np.average(self.position, axis=0, weights=self.mass)
        center_v = np.average(self.velocity, axis=0, weights=self.mass)
        rel = self.position - center[None, :]
        rel_v = self.velocity - center_v[None, :]
        k = float(self.cfg.get("stiffness", 44.0))
        c = float(self.cfg.get("damping", 0.75))
        acc = np.broadcast_to(self.gravity, (n, 3)).copy()
        acc += k * (self.rest - rel) - c * rel_v
        penetration = self.ground_z - self.position[:, 2]
        active = penetration > 0.0
        if np.any(active):
            omega, zeta = 85.0, 0.9
            acc[active, 2] += omega * omega * penetration[active]
            acc[active, 2] -= 2.0 * zeta * omega * np.minimum(self.velocity[active, 2], 0.0)
            acc[active, :2] -= self.friction * omega * self.velocity[active, :2]
        self.velocity += dt * acc
        self.position += dt * self.velocity
        below = self.position[:, 2] < self.ground_z
        if np.any(below):
            ids = np.flatnonzero(below)
            self.position[ids, 2] = self.ground_z
            incoming = self.velocity[ids, 2] < 0.0
            self.velocity[ids[incoming], 2] *= -self.restitution
            settled = ids[np.abs(self.velocity[ids, 2]) < 1e-5]
            self.velocity[settled, 2] = 0.0
        self.time += dt
        self.steps += 1

    def _frame(self) -> StateFrame:
        n = len(self.position)
        center = np.average(self.position, axis=0, weights=self.mass)
        rel = self.position - center[None, :]
        rn = np.linalg.norm(self.rest, axis=1)
        cn = np.linalg.norm(rel, axis=1)
        stretch = np.ones(n)
        mask = rn > 1e-10
        stretch[mask] = np.clip(cn[mask] / rn[mask], 0.25, 2.0)
        deformation = np.zeros((n, 3, 3))
        deformation[:, 0, 0] = stretch
        deformation[:, 1, 1] = stretch
        deformation[:, 2, 2] = stretch
        frame = StateFrame(
            time=float(self.time), position=self.position.copy(), velocity=self.velocity.copy(),
            mass=self.mass.copy(), body_id=np.zeros(n, dtype=np.int64),
            attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": deformation},
            fields={
                "contact_signed_distance": self.position[:, 2] - self.ground_z,
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": self.contact_length},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0 or target < self.time - 1e-10:
            raise ValueError("advance_to requires a finite nondecreasing time")
        max_dt = 1.0 / 600.0
        while self.time < target - 1e-12:
            self._substep(min(max_dt, target - self.time))
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler", "substep_max": 1.0 / 600.0, "substeps": self.steps}
