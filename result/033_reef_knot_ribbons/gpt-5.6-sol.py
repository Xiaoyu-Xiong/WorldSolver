"""Deterministic cloth dynamics for the reef-knot ribbon scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small explicit mass-spring/penalty-contact solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.x0 = np.asarray(initial.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body = np.asarray(initial.body_id, dtype=int).copy()
        self.n = len(self.x)
        self.shape = tuple(int(q) for q in scene.config["ribbon"]["grid_shape"])
        self.nr = int(np.prod(self.shape))
        self.thickness = float(scene.config["ribbon"]["thickness"])
        self.strain_upper = float(scene.config["ribbon"]["strain_upper"])
        self.mu = float(scene.config["ribbon"]["friction"])
        self.pull_speed = float(scene.config.get("pull_speed", 0.0))
        self.current_time = 0.0
        self.internal_dt = min(0.004, float(scene.config.get("sample_dt", 1.0)) / 4.0)
        self._build_edges()
        self._initial_attrs = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self._ref_d1 = np.zeros((self.n, 3), dtype=float)
        self._ref_d2 = np.zeros((self.n, 3), dtype=float)
        for b in range(2):
            sl = slice(b * self.nr, (b + 1) * self.nr)
            q = self.x0[sl].reshape(self.shape[1], self.shape[0], 3)
            self._ref_d1[sl] = np.gradient(q, axis=1, edge_order=1).reshape(-1, 3)
            self._ref_d2[sl] = np.gradient(q, axis=0, edge_order=1).reshape(-1, 3)

    def _build_edges(self) -> None:
        rows, cols = self.shape[1], self.shape[0]
        edges = []
        for b in range(2):
            off = b * self.nr
            for r in range(rows):
                for c in range(cols):
                    i = off + r * cols + c
                    if c + 1 < cols:
                        edges.append((i, i + 1, 1.0))
                    if r + 1 < rows:
                        edges.append((i, i + cols, 1.0))
                    if c + 1 < cols and r + 1 < rows:
                        edges.append((i, i + cols + 1, 0.72))
                    if c > 0 and r + 1 < rows:
                        edges.append((i, i + cols - 1, 0.72))
        self.ei = np.asarray([e[0] for e in edges], dtype=int)
        self.ej = np.asarray([e[1] for e in edges], dtype=int)
        self.erest = np.asarray([np.linalg.norm(self.x0[i] - self.x0[j]) for i, j, _ in edges])
        self.ediag = np.asarray([e[2] for e in edges])
        self.endpoints = np.array([0, self.shape[0] - 1, self.nr, self.nr + self.shape[0] - 1], dtype=int)

    def _target_endpoints(self, t: float) -> np.ndarray:
        target = self.x0[self.endpoints].copy()
        amount = self.pull_speed * max(0.0, t)
        target[0] -= np.array([amount, 0.0, 0.0])
        target[1] += np.array([amount, 0.0, 0.0])
        target[2] -= np.array([0.0, amount, 0.0])
        target[3] += np.array([0.0, amount, 0.0])
        return target

    def _forces(self, t: float) -> np.ndarray:
        f = np.zeros_like(self.x)
        d = self.x[self.ej] - self.x[self.ei]
        length = np.linalg.norm(d, axis=1)
        nrm = d / np.maximum(length[:, None], 1.0e-10)
        stretch = length - self.erest
        relv = self.v[self.ej] - self.v[self.ei]
        damp = np.sum(relv * nrm, axis=1)
        fs = (190.0 * self.ediag * stretch + 2.4 * damp)[:, None] * nrm
        np.add.at(f, self.ei, fs)
        np.add.at(f, self.ej, -fs)
        for b in range(2):
            off = b * self.nr
            q = self.x[off : off + self.nr].reshape(self.shape[1], self.shape[0], 3)
            vv = self.v[off : off + self.nr].reshape(self.shape[1], self.shape[0], 3)
            lap = np.zeros_like(q)
            lap[:, 1:-1] += q[:, :-2] - 2.0 * q[:, 1:-1] + q[:, 2:]
            lap[1:-1, :] += q[:-2, :] - 2.0 * q[1:-1, :] + q[2:, :]
            vlap = np.zeros_like(vv)
            vlap[:, 1:-1] += vv[:, :-2] - 2.0 * vv[:, 1:-1] + vv[:, 2:]
            vlap[1:-1, :] += vv[:-2, :] - 2.0 * vv[1:-1, :] + vv[2:, :]
            f[off : off + self.nr] += 1.3 * lap.reshape(-1, 3) + 0.16 * vlap.reshape(-1, 3)
        radius = max(2.2 * self.thickness, 0.009)
        for a in range(self.n):
            candidates = np.arange(self.nr, self.n) if self.body[a] == 0 else np.arange(0, self.nr)
            delta = self.x[a] - self.x[candidates]
            dist = np.linalg.norm(delta, axis=1)
            hit = dist < radius
            if not np.any(hit):
                continue
            ids = candidates[hit]
            dd = delta[hit]
            dl = np.maximum(dist[hit], 1.0e-8)
            nn = dd / dl[:, None]
            penetration = radius - dl
            rv = self.v[a] - self.v[ids]
            vn = np.sum(rv * nn, axis=1)
            normal_force = 105.0 * penetration - 1.5 * np.minimum(vn, 0.0)
            tang = rv - vn[:, None] * nn
            ft = np.minimum(self.mu * np.maximum(normal_force, 0.0), 0.8)[:, None] * tang / np.maximum(np.linalg.norm(tang, axis=1)[:, None], 1.0e-8)
            pair_force = normal_force[:, None] * nn - ft
            f[a] += np.sum(pair_force, axis=0)
            np.add.at(f, ids, -pair_force)
        return f

    def _project_constraints(self, t: float) -> None:
        maxlen = self.erest * self.strain_upper
        for _ in range(2):
            d = self.x[self.ej] - self.x[self.ei]
            length = np.linalg.norm(d, axis=1)
            over = length > maxlen
            if not np.any(over):
                break
            ids = np.flatnonzero(over)
            nn = d[ids] / np.maximum(length[ids, None], 1.0e-10)
            corr = (length[ids] - maxlen[ids])[:, None] * nn
            np.add.at(self.x, self.ei[ids], 0.5 * corr)
            np.add.at(self.x, self.ej[ids], -0.5 * corr)
        targets = self._target_endpoints(t)
        old = self.x[self.endpoints].copy()
        self.x[self.endpoints] = targets
        self.v[self.endpoints] = (targets - old) / max(self.internal_dt, 1.0e-8)

    def _step(self, h: float, t: float) -> None:
        force = self._forces(t)
        self.v += h * force / np.maximum(self.mass[:, None], 1.0e-9)
        self.v *= np.exp(-2.0 * h)
        self.x += h * self.v
        self._project_constraints(t + h)
        self.v = np.clip(self.v, -8.0, 8.0)

    def _deformation_gradient(self) -> np.ndarray:
        out = np.tile(np.eye(3), (self.n, 1, 1))
        for b in range(2):
            sl = slice(b * self.nr, (b + 1) * self.nr)
            q = self.x[sl].reshape(self.shape[1], self.shape[0], 3)
            d1 = np.gradient(q, axis=1, edge_order=1).reshape(-1, 3)
            d2 = np.gradient(q, axis=0, edge_order=1).reshape(-1, 3)
            e1, e2 = self._ref_d1[sl], self._ref_d2[sl]
            for k in range(self.nr):
                n1 = e1[k] / max(np.linalg.norm(e1[k]), 1.0e-9)
                n2 = e2[k] / max(np.linalg.norm(e2[k]), 1.0e-9)
                out[b * self.nr + k] += np.outer(d1[k] - e1[k], n1) + np.outer(d2[k] - e2[k], n2)
        return out

    def _contact_gaps(self) -> np.ndarray:
        a, b = self.x[: self.nr], self.x[self.nr :]
        dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return np.minimum(np.min(dist, axis=1), np.min(dist, axis=0)) - 2.0 * self.thickness

    def _frame(self, time: float) -> StateFrame:
        attrs = {"mass_quantity": self.mass.copy(), "deformation_gradient": self._deformation_gradient()}
        if "uv" in self._initial_attrs:
            attrs["uv"] = self._initial_attrs["uv"].copy()
        frame = StateFrame(
            float(time), self.x.copy(), self.v.copy(), mass=self.mass.copy(), body_id=self.body.copy(),
            attributes=attrs,
            fields={"contact_signed_distance": self._contact_gaps(), "cumulative_mass_source": np.zeros(1, dtype=float)},
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1.0e-9:
            raise ValueError("advance_to requires nondecreasing finite time")
        while self.current_time < target - 1.0e-10:
            h = min(self.internal_dt, target - self.current_time)
            self._step(h, self.current_time)
            self.current_time += h
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "explicit_mass_spring_contact", "nodes": int(self.n), "internal_dt": float(self.internal_dt)}
