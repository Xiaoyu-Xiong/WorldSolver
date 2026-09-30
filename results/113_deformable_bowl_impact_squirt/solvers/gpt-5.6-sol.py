"""Deterministic coupled particle/spring solver for the deformable bowl scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.pos = np.asarray(initial.position, float).copy()
        self.vel = np.asarray(initial.velocity, float).copy()
        self.mass = np.asarray(initial.mass, float).copy()
        self.body = np.asarray(initial.body_id, np.int32).copy()
        self.n = len(self.pos)
        self.bowl = np.flatnonzero(self.body == 1)
        self.liquid = np.flatnonzero(self.body == 2)
        self.water = np.flatnonzero(self.body == 3)
        self.t = 0.0
        self.g = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), float)
        faces = np.asarray(scene.faces, np.int64)
        edges = set()
        for tri in faces:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges.add(tuple(sorted((int(a), int(b)))))
        self.edges = np.asarray(sorted(edges), np.int64)
        bp = self.pos[self.bowl]
        self.rest_len = np.linalg.norm(bp[self.edges[:, 0]] - bp[self.edges[:, 1]], axis=1)
        self.rest_pos = self.pos.copy()
        self.external_force = np.zeros((self.n, 3), float)
        self.interaction = np.zeros((3, 3), float)
        self.cum_impulse = np.zeros((3, 3), float)
        self.cum_out = np.zeros((3, 3), float)

    def _fluid_pairs(self, indices, force, h=0.18):
        if len(indices) == 0:
            return
        cell = np.floor(self.pos[indices] / h).astype(np.int64)
        buckets = {}
        for k, c in enumerate(cell):
            buckets.setdefault(tuple(c.tolist()), []).append(k)
        for k, c in enumerate(cell):
            i = int(indices[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for q in buckets.get((int(c[0] + dx), int(c[1] + dy), int(c[2] + dz)), ()):
                            if q <= k:
                                continue
                            j = int(indices[q]); dvec = self.pos[j] - self.pos[i]
                            d2 = float(np.dot(dvec, dvec))
                            if d2 < 1e-10 or d2 >= h * h:
                                continue
                            d = np.sqrt(d2); nh = dvec / d
                            aij = 34.0 * (h - d) * nh + 1.8 * (self.vel[j] - self.vel[i]) * (h - d) / h
                            fij = self.mass[i] * aij
                            force[i] -= fij; force[j] += fij

    def _step(self, dt):
        force = self.mass[:, None] * self.g
        bp, bv = self.pos[self.bowl], self.vel[self.bowl]
        for e, r0 in zip(self.edges, self.rest_len):
            a, b = int(e[0]), int(e[1]); dvec = bp[b] - bp[a]; length = np.linalg.norm(dvec)
            if length < 1e-8:
                continue
            nh = dvec / length
            fs = (1450.0 * (length - r0) + 7.0 * np.dot(bv[b] - bv[a], nh)) * nh
            ia, ib = self.bowl[a], self.bowl[b]; force[ia] += fs; force[ib] -= fs
        fluids = np.concatenate((self.liquid, self.water))
        self._fluid_pairs(fluids, force)
        if len(self.bowl) and len(fluids):
            shellp = self.pos[self.bowl]
            for j in fluids:
                d2 = np.sum((shellp - self.pos[j]) ** 2, axis=1); q = int(np.argmin(d2))
                d = np.sqrt(max(float(d2[q]), 1e-12))
                if d < 0.15:
                    nh = (self.pos[j] - shellp[q]) / d
                    acontact = 115.0 * (0.15 - d) * nh - 2.5 * np.dot(self.vel[j] - self.vel[self.bowl[q]], nh) * nh
                    fj = self.mass[j] * acontact
                    force[j] += fj; force[self.bowl[q]] -= fj; self.interaction[0] += fj
        for i in fluids:
            p, v = self.pos[i], self.vel[i]
            for axis, lo, hi in ((0, -2.18, 2.18), (1, -1.48, 1.52)):
                if p[axis] < lo:
                    dv = np.zeros(3); dv[axis] = 70.0 * (lo - p[axis]) - 5.0 * v[axis]; force[i] += self.mass[i] * dv
                elif p[axis] > hi:
                    dv = np.zeros(3); dv[axis] = -70.0 * (p[axis] - hi) - 5.0 * v[axis]; force[i] += self.mass[i] * dv
            if p[2] < 0.06:
                dvz = 90.0 * (0.06 - p[2]) - 7.0 * v[2]; force[i, 2] += self.mass[i] * dvz
        for i in self.bowl:
            if self.pos[i, 2] < 0.035:
                fz = self.mass[i] * (2400.0 * (0.035 - self.pos[i, 2]) - 32.0 * self.vel[i, 2])
                force[i, 2] += fz; self.interaction[1, 2] += fz
        acc = force / np.maximum(self.mass[:, None], 1e-12)
        self.vel += dt * acc; self.vel *= np.exp(-0.18 * dt); self.pos += dt * self.vel
        self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=5.0, neginf=-5.0)
        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=30.0, neginf=-30.0)
        self.external_force = force.copy()
        for row, bid in enumerate((1, 2, 3)):
            mask = self.body == bid; self.cum_impulse[row] += dt * np.sum(force[mask], axis=0)
        rim = 2.30
        out = self.liquid[self.pos[self.liquid, 2] > rim + 0.08] if len(self.liquid) else np.empty(0, int)
        if len(out):
            self.cum_out[1] += np.sum(self.mass[out, None] * self.vel[out], axis=0)

    def _frame(self, time):
        weights = np.vstack([self.body == 1, self.body == 2, self.body == 3]).astype(float)
        normals = np.array([[0., 0., 1.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        fields = {
            "boundary_normal": normals,
            "boundary_target_velocity": np.zeros_like(normals),
            "boundary_velocity": np.zeros_like(normals),
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": self.cum_out.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": np.sum(self.external_force, axis=0)[None, None, :],
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": self.interaction.copy(),
        }
        frame = StateFrame(float(time), self.pos.copy(), self.vel.copy(), mass=self.mass.copy(), body_id=self.body.copy(),
                           attributes={"mass_quantity": self.mass.copy()}, fields=fields,
                           scalars={"reference_momentum": 1.0, "reference_velocity": 1.0, "gravity_z": float(self.g[2]), "characteristic_length": 1.0})
        frame.validate(); return frame

    def advance_to(self, time: float):
        target = float(time)
        if target >= self.t:
            while self.t < target - 1e-12:
                dt = min(0.006, target - self.t); self._step(dt); self.t += dt
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "coupled_mass_spring_particle", "particle_count": int(self.n), "substep_max": 0.006}
