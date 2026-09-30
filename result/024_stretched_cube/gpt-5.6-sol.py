"""Numpy deformable-cube solver for the stretched-cube benchmark.

The subject asset is a triangulated surface. Its edges are treated as a
mass-spring discretisation of an elastic shell, with prescribed end-grip
motion during the stretch stage and a short inward impulse afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic semi-implicit spring solver with kinematic end grips."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg
        self.x0 = np.asarray(scene.vertices, dtype=float).copy()
        self.x = self.x0.copy()
        self.n = len(self.x)
        self.v = np.tile(np.asarray(cfg.get("initial_velocity", (0.0, 0.0, 0.0)), dtype=float), (self.n, 1))
        self.mass = np.full(self.n, float(cfg["subject_mass"]) / max(self.n, 1), dtype=float)
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, 0.0)), dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_scale = float(cfg.get("contact_length_scale", 0.04))
        self.time = 0.0
        self.step_count = 0

        with np.load(Path(__file__).resolve().parents[1] / cfg["subject_asset"]) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
        edge_pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
        edge_pairs.sort(axis=1)
        self.edges = np.unique(edge_pairs, axis=0)
        self.rest_edge = self.x0[self.edges[:, 1]] - self.x0[self.edges[:, 0]]
        self.rest_len = np.maximum(np.linalg.norm(self.rest_edge, axis=1), 1.0e-8)

        xmin, xmax = self.x0[:, 0].min(), self.x0[:, 0].max()
        tol = max(1.0e-9, 0.25 * float(np.min(self.rest_len)))
        self.left = np.flatnonzero(self.x0[:, 0] <= xmin + tol)
        self.right = np.flatnonzero(self.x0[:, 0] >= xmax - tol)
        self.center_x = 0.5 * (xmin + xmax)
        self.half_width = 0.5 * (xmax - xmin)

        cov = np.zeros((self.n, 3, 3), dtype=float)
        outer = self.rest_edge[:, :, None] * self.rest_edge[:, None, :]
        np.add.at(cov, self.edges[:, 0], outer)
        np.add.at(cov, self.edges[:, 1], outer)
        self.cov_inv = np.linalg.pinv(cov + np.eye(3)[None] * 1.0e-10)

        self.edge_k = float(cfg.get("stiffness", 28.0))
        self.damping = max(0.0, float(cfg.get("damping", 0.42)))
        self.stretch_ratio = max(1.0, float(cfg.get("stretch_ratio", 1.0)))
        self.stretch_end = max(1.0e-6, float(cfg.get("stretch_end", 1.0)))
        self.pulse_time = float(cfg.get("pulse_time", 1.5))
        self.base_dt = min(float(cfg.get("sample_dt", 1.0 / 30.0)) / 4.0, 1.0 / 240.0)
        self._last_force = np.zeros_like(self.x)
        self._last_external = np.zeros_like(self.x)

    def _stretch_scale(self, t: float) -> tuple[float, float]:
        if t <= 0.0:
            return 1.0, 0.0
        if t >= self.stretch_end:
            return self.stretch_ratio, 0.0
        u = np.clip(t / self.stretch_end, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        dsdu = 6.0 * u * (1.0 - u)
        return 1.0 + (self.stretch_ratio - 1.0) * smooth, (self.stretch_ratio - 1.0) * dsdu / self.stretch_end

    def _forces(self, t: float) -> np.ndarray:
        external = self.mass[:, None] * self.gravity[None, :]
        f = external.copy()
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = self.x[j] - self.x[i]
        length = np.linalg.norm(d, axis=1)
        direction = d / np.maximum(length[:, None], 1.0e-9)
        pair = self.edge_k * (length - self.rest_len)[:, None] * direction
        np.add.at(f, i, pair)
        np.add.at(f, j, -pair)

        width = 0.075
        pulse = np.exp(-((t - self.pulse_time) / width) ** 2) if t >= self.pulse_time else 0.0
        if pulse > 1.0e-12:
            sign = np.sign(self.x[:, 0] - self.center_x)
            pulse_force = -sign * (0.70 * float(self.cfg.get("stiffness", 28.0)) * pulse)
            f[:, 0] += pulse_force
            external[:, 0] += pulse_force

        penetration = self.ground_z - self.x[:, 2]
        active = penetration > 0.0
        if np.any(active):
            f[active, 2] += 1400.0 * penetration[active] - 18.0 * self.v[active, 2]
        self._last_external = external
        return f

    def _integrate(self, dt: float) -> None:
        t_next = self.time + dt
        force = self._forces(t_next)
        self.v += dt * force / self.mass[:, None]
        self.v *= np.exp(-self.damping * dt)
        self.x += dt * self.v

        scale, scale_rate = self._stretch_scale(t_next)
        target_x = self.center_x + scale * (self.x0[:, 0] - self.center_x)
        target_vx = scale_rate * (self.x0[:, 0] - self.center_x)
        constrained = np.concatenate((self.left, self.right))
        self.x[constrained, 0] = target_x[constrained]
        self.v[constrained, 0] = target_vx[constrained]
        self.x[constrained, 1:] = self.x0[constrained, 1:]
        self.v[constrained, 1:] = 0.0
        self.time = t_next
        self.step_count += 1
        self._last_force = force

    def _deformation_gradient(self) -> np.ndarray:
        i, j = self.edges[:, 0], self.edges[:, 1]
        current = self.x[j] - self.x[i]
        contribution = current[:, :, None] * self.rest_edge[:, None, :]
        accum = np.zeros((self.n, 3, 3), dtype=float)
        np.add.at(accum, i, contribution)
        np.add.at(accum, j, contribution)
        return np.einsum("nij,njk->nik", accum, self.cov_inv)

    def _contact_distance(self) -> np.ndarray:
        d_ground = self.x[:, 2] - self.ground_z
        distances = [d_ground]
        for cx in (-0.72, 0.72):
            q = np.abs(self.x - np.array((cx, 0.0, 0.86))) - np.array((0.12, 0.55, 0.55))
            outside = np.maximum(q, 0.0)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside_dist = np.minimum(np.max(q, axis=1), 0.0)
            distances.append(outside_dist + inside_dist)
        return np.min(np.stack(distances, axis=1), axis=1)

    def _frame(self, requested_time: float) -> StateFrame:
        mass = self.mass.copy()
        frame = StateFrame(
            time=float(requested_time), position=self.x.copy(), velocity=self.v.copy(),
            mass=mass, body_id=np.zeros(self.n, dtype=np.int64),
            attributes={"mass_quantity": mass.copy(), "deformation_gradient": self._deformation_gradient()},
            fields={"cumulative_mass_source": np.zeros(1, dtype=float),
                    "external_force": self._last_external.copy(),
                    "contact_signed_distance": self._contact_distance()},
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to time must be finite and non-negative")
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._integrate(min(self.base_dt, target - self.time))
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"method": "semi_implicit_edge_springs", "integration_substep": float(self.base_dt),
                "edge_count": int(len(self.edges)), "step_count": int(self.step_count)}
