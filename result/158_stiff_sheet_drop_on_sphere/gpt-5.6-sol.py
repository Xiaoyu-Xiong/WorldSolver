"""Self-contained stiff-sheet drop solver using position based dynamics."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.radius = 0.62
        self.sphere_center = np.array([0.0, 0.0, 0.68], dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.floor_z = 0.0
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.inv_mass = 1.0 / np.maximum(self.mass, 1.0e-12)
        self.rest = self.x.copy()
        self.time = 0.0
        self.seed = int(seed)
        self.step_limit = min(1.0 / 240.0, float(cfg.get("sample_dt", 1.0 / 30.0)) / 4.0)

        def vid(i, j):
            return j * self.nx + i

        structural, diagonal, bending = [], [], []
        for j in range(self.ny):
            for i in range(self.nx):
                a = vid(i, j)
                if i + 1 < self.nx:
                    structural.append((a, vid(i + 1, j)))
                if j + 1 < self.ny:
                    structural.append((a, vid(i, j + 1)))
                if i + 1 < self.nx and j + 1 < self.ny:
                    diagonal.extend(((a, vid(i + 1, j + 1)), (vid(i + 1, j), vid(i, j + 1))))
                if i + 2 < self.nx:
                    bending.append((a, vid(i + 2, j)))
                if j + 2 < self.ny:
                    bending.append((a, vid(i, j + 2)))
        self.structural = np.asarray(structural, dtype=np.int32)
        self.diagonal = np.asarray(diagonal, dtype=np.int32)
        self.bending = np.asarray(bending, dtype=np.int32)
        self.structural_rest = self._pair_lengths(self.structural)
        self.diagonal_rest = self._pair_lengths(self.diagonal)
        self.bending_rest = self._pair_lengths(self.bending)
        self._last_external = self.mass[:, None] * self.gravity[None, :]

    def _pair_lengths(self, pairs):
        if len(pairs) == 0:
            return np.empty(0, dtype=float)
        return np.linalg.norm(self.rest[pairs[:, 0]] - self.rest[pairs[:, 1]], axis=1)

    def _project_pairs(self, pairs, rest, strength):
        if len(pairs) == 0:
            return
        a, b = pairs[:, 0], pairs[:, 1]
        d = self.x[a] - self.x[b]
        lengths = np.linalg.norm(d, axis=1)
        safe = np.maximum(lengths, 1.0e-10)
        corr = (strength * (lengths - rest) / safe)[:, None] * d
        wa, wb = self.inv_mass[a], self.inv_mass[b]
        denom = wa + wb + 1.0e-12
        np.add.at(self.x, a, -corr * (wa / denom)[:, None])
        np.add.at(self.x, b, corr * (wb / denom)[:, None])

    def _project_boundaries(self):
        rel = self.x - self.sphere_center
        dist = np.linalg.norm(rel, axis=1)
        normal = rel / np.maximum(dist[:, None], 1.0e-10)
        gap = dist - (self.radius + 0.5 * self.thickness)
        hit = gap < 0.0
        if np.any(hit):
            self.x[hit] -= gap[hit, None] * normal[hit]
        below = self.x[:, 2] < 0.5 * self.thickness
        self.x[below, 2] = 0.5 * self.thickness

    def _project_self_contact(self):
        cell = max(2.5 * self.thickness, 0.025)
        keys = np.floor(self.x / cell).astype(np.int32)
        buckets = {}
        for k, key in enumerate(keys):
            buckets.setdefault(tuple(int(q) for q in key), []).append(k)
        min_dist = 0.9 * self.thickness
        min_sq = min_dist * min_dist
        for key, members in buckets.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        other = buckets.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if other is None:
                            continue
                        for i in members:
                            for j in other:
                                ii, jj = i % self.nx, j % self.nx
                                if j <= i or (abs(ii - jj) <= 1 and abs(i // self.nx - j // self.nx) <= 1):
                                    continue
                                d = self.x[i] - self.x[j]
                                ds = float(np.dot(d, d))
                                if 1.0e-14 < ds < min_sq:
                                    length = np.sqrt(ds)
                                    n = d / length
                                    amount = 0.5 * (min_dist - length)
                                    wi, wj = self.inv_mass[i], self.inv_mass[j]
                                    self.x[i] += n * amount * wi / (wi + wj)
                                    self.x[j] -= n * amount * wj / (wi + wj)

    def _substep(self, h: float) -> None:
        old = self.x.copy()
        self.v += self.gravity[None, :] * h
        self.x += self.v * h
        for _ in range(5):
            self._project_pairs(self.structural, self.structural_rest, 0.98)
            self._project_pairs(self.diagonal, self.diagonal_rest, 0.72)
            self._project_pairs(self.bending, self.bending_rest, 0.16)
            self._project_boundaries()
            self._project_self_contact()
        if len(self.bending):
            ba, bb = self.bending[:, 0], self.bending[:, 1]
            bd = np.linalg.norm(self.x[ba] - self.x[bb], axis=1)
            near = np.minimum(np.linalg.norm(self.x[ba] - self.sphere_center, axis=1),
                              np.linalg.norm(self.x[bb] - self.sphere_center, axis=1)) < self.radius + 0.12
            plastic = near & (bd < 0.90 * self.bending_rest)
            self.bending_rest[plastic] = 0.985 * self.bending_rest[plastic] + 0.015 * bd[plastic]
        self.v = (self.x - old) / max(h, 1.0e-9)
        rel = self.x - self.sphere_center
        dist = np.linalg.norm(rel, axis=1)
        normal = rel / np.maximum(dist[:, None], 1.0e-10)
        contact = dist <= self.radius + self.thickness * 0.55
        if np.any(contact):
            vn = np.sum(self.v[contact] * normal[contact], axis=1)
            self.v[contact] -= np.minimum(vn, 0.0)[:, None] * normal[contact]
            self.v[contact] *= 0.78
        floor_hit = self.x[:, 2] <= 0.5 * self.thickness + 1.0e-5
        self.v[floor_hit, 2] = np.maximum(self.v[floor_hit, 2], 0.0)
        self.v *= 0.997
        self._last_external = self.mass[:, None] * self.gravity[None, :]
        self.time += h

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        if target < self.time - 1.0e-10:
            target = self.time
        while self.time + 1.0e-10 < target:
            self._substep(min(self.step_limit, target - self.time))
        return self._frame(target)

    def _deformation_gradient(self):
        f = np.zeros((self.n, 2, 2), dtype=float)
        for j in range(self.ny):
            for i in range(self.nx):
                k = j * self.nx + i
                ii, im = min(i + 1, self.nx - 1), max(i - 1, 0)
                jj, jm = min(j + 1, self.ny - 1), max(j - 1, 0)
                dx0 = self.rest[j * self.nx + ii] - self.rest[j * self.nx + im]
                dx = self.x[j * self.nx + ii] - self.x[j * self.nx + im]
                dy0 = self.rest[jj * self.nx + i] - self.rest[jm * self.nx + i]
                dy = self.x[jj * self.nx + i] - self.x[jm * self.nx + i]
                nx0, ny0 = max(np.linalg.norm(dx0), 1e-8), max(np.linalg.norm(dy0), 1e-8)
                f[k, 0, 0] = np.dot(dx, dx0) / (nx0 * nx0)
                f[k, 1, 0] = np.dot(dx, dy0) / (nx0 * ny0)
                f[k, 0, 1] = np.dot(dy, dx0) / (nx0 * ny0)
                f[k, 1, 1] = np.dot(dy, dy0) / (ny0 * ny0)
        return np.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0)

    def _frame(self, requested_time: float) -> StateFrame:
        rel = self.x - self.sphere_center
        sphere_gap = np.linalg.norm(rel, axis=1) - (self.radius + 0.5 * self.thickness)
        ground_gap = self.x[:, 2] - 0.5 * self.thickness
        frame = StateFrame(float(requested_time), self.x.copy(), self.v.copy(), mass=self.mass.copy(),
                           body_id=np.zeros(self.n, dtype=np.int32),
                           attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": self._deformation_gradient()},
                           fields={"contact_signed_distance": np.concatenate((sphere_gap, ground_gap)),
                                   "cumulative_mass_source": np.zeros(1, dtype=float),
                                   "external_force": self._last_external.copy()},
                           scalars={"contact_length_scale": self.thickness})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        gap = np.linalg.norm(self.x - self.sphere_center, axis=1) - self.radius
        return {"time": float(self.time), "min_sphere_gap": float(np.min(gap)), "vertex_count": self.n}
