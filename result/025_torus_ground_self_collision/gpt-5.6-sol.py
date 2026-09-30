"""Self-contained deformable torus solver."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.cfg = scene.config
        self.rest = np.asarray(scene.vertices, dtype=float).copy()
        self.n = len(self.rest)
        self.x = self.rest.copy()
        iv = np.asarray(self.cfg.get("initial_velocity", [0., 0., 0.]), float)
        self.v = np.repeat(iv[None, :], self.n, axis=0)
        self.m = np.full(self.n, float(self.cfg.get("subject_mass", 1.0)) / max(self.n, 1))
        self.g = np.asarray(self.cfg.get("gravity", [0., 0., -9.81]), float)
        self.ground_z = float(self.cfg.get("ground_z", 0.0))
        self.k = float(self.cfg.get("stiffness", 13.0)); self.c = float(self.cfg.get("damping", 1.2))
        self.restitution = float(self.cfg.get("restitution", 0.2)); self.friction = float(self.cfg.get("friction", 0.58))
        self.contact_scale = float(self.cfg.get("contact_length_scale", 0.04))
        self.t = 0.0; self.contact_count = 0
        d2 = np.sum((self.rest[:, None, :] - self.rest[None, :, :])**2, axis=2); np.fill_diagonal(d2, np.inf)
        kk = min(10, max(3, self.n - 1)); nbr = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        edges = set()
        for i in range(self.n):
            for j0 in nbr[i]:
                j = int(j0); edges.add((i, j) if i < j else (j, i))
        es = sorted(edges); self.ei = np.array([a for a, b in es], dtype=np.int64); self.ej = np.array([b for a, b in es], dtype=np.int64)
        dr = self.rest[self.ej] - self.rest[self.ei]; self.rest_len = np.linalg.norm(dr, axis=1); self.edge_set = edges
        self._last_frame = self._make_frame()

    def _deformation_gradient(self):
        F = np.empty((self.n, 3, 3)); I = np.eye(3)
        for i in range(self.n):
            q = np.where((self.ei == i) | (self.ej == i))[0]
            if q.size < 3: F[i] = I; continue
            A = np.zeros((3, 3)); B = np.zeros((3, 3))
            for e in q:
                j = int(self.ej[e] if self.ei[e] == i else self.ei[e]); dr = self.rest[j] - self.rest[i]; dx = self.x[j] - self.x[i]
                A += np.outer(dr, dr); B += np.outer(dx, dr)
            F[i] = B @ np.linalg.inv(A + 1e-8 * I)
            if not np.isfinite(F[i]).all(): F[i] = I
        return F

    def _make_frame(self):
        frame = StateFrame(time=float(self.t), position=self.x.copy(), velocity=self.v.copy(), mass=self.m.copy(), body_id=np.zeros(self.n, dtype=np.int64), attributes={"mass_quantity": self.m.copy(), "deformation_gradient": self._deformation_gradient()}, fields={"cumulative_mass_source": np.zeros(1), "external_force": self.m[:, None] * self.g[None, :], "contact_signed_distance": (self.x[:, 2] - self.ground_z).astype(float)}, scalars={"contact_length_scale": self.contact_scale})
        frame.validate(); return frame

    def _self_contacts(self, threshold):
        cell = max(threshold, 1e-3); keys = np.floor(self.x / cell).astype(np.int64); buckets = {}
        for i, key in enumerate(keys): buckets.setdefault(tuple(key), []).append(i)
        for key, ids in buckets.items():
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for oz in (-1, 0, 1):
                        for i in ids:
                            for j in buckets.get((key[0]+ox, key[1]+oy, key[2]+oz), ()):
                                if j <= i or (i, j) in self.edge_set: continue
                                d = self.x[j] - self.x[i]; r = np.linalg.norm(d)
                                if 1e-9 < r < threshold: yield i, j, d / r, r

    def _step(self, h):
        force = self.m[:, None] * self.g[None, :]
        if self.ei.size:
            d = self.x[self.ej] - self.x[self.ei]; l = np.linalg.norm(d, axis=1); nvec = d / np.maximum(l[:, None], 1e-9)
            relv = self.v[self.ej] - self.v[self.ei]; fs = self.k * (l - self.rest_len) + self.c * np.sum(relv * nvec, axis=1); ef = fs[:, None] * nvec
            np.add.at(force, self.ei, ef); np.add.at(force, self.ej, -ef)
        pen = self.ground_z - self.x[:, 2]; active = pen > 0
        if np.any(active):
            fn = np.maximum(180*self.k*pen[active] - 8*self.v[active, 2], 0); force[active, 2] += fn
            force[active, :2] -= self.friction * fn[:, None] * np.tanh(self.v[active, :2] / 0.05); self.contact_count += int(active.sum())
        thresh = max(0.11, 1.25*self.contact_scale)
        for i, j, nvec, r in self._self_contacts(thresh):
            fn = 35*self.k*(thresh-r) + 2*self.c*max(-np.dot(self.v[j]-self.v[i], nvec), 0); f = fn*nvec; force[i] -= f; force[j] += f
        self.v += h * force / self.m[:, None]; self.v *= np.exp(-0.08*h); self.x += h*self.v
        below = self.x[:, 2] < self.ground_z
        if np.any(below):
            self.x[below, 2] = self.ground_z; idx = np.where(below)[0]; down = self.v[idx, 2] < 0; self.v[idx[down], 2] *= -self.restitution

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0: raise ValueError("time must be finite and non-negative")
        if target < self.t: return self._last_frame
        while self.t < target - 1e-12:
            h = min(0.004, target - self.t); self._step(h); self.t += h
        self._last_frame = self._make_frame(); return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {"time": float(self.t), "contact_events": int(self.contact_count), "particle_count": int(self.n)}
