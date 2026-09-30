"""Deterministic mass-spring thin-sheet solver for the crumpling scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring shell advanced with symplectic Euler and contact projection."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.count = self.nx * self.ny
        self.thickness = float(cloth["thickness"])
        self.radius = 0.5 * self.thickness
        self.mass = np.full(self.count, float(cloth["total_mass"]) / self.count)
        self.inv_mass = 1.0 / self.mass
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        sx, sy = (float(v) for v in cloth["size"])
        self.dx, self.dy = sx / (self.nx - 1), sy / (self.ny - 1)
        cx, cy, cz = (float(v) for v in cloth["center"])
        xs = np.linspace(cx - sx * 0.5, cx + sx * 0.5, self.nx)
        ys = np.linspace(cy - sy * 0.5, cy + sy * 0.5, self.ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        ix = np.arange(self.nx, dtype=float)[None, :]
        iy = np.arange(self.ny, dtype=float)[:, None]
        ripple = 0.0045 * (np.sin(0.73 * ix + 1.17 * iy + 0.001 * int(seed))
                           + 0.45 * np.sin(1.91 * ix - 0.61 * iy))
        self.position = np.column_stack((xx.ravel(), yy.ravel(), (cz + ripple).ravel()))
        self.velocity = np.zeros_like(self.position)
        self.time, self.steps = 0.0, 0
        boundary = ((np.arange(self.nx)[None, :] == 0)
                    | (np.arange(self.nx)[None, :] == self.nx - 1)
                    | (np.arange(self.ny)[:, None] == 0)
                    | (np.arange(self.ny)[:, None] == self.ny - 1))
        self.boundary_vertex = boundary.ravel().astype(np.int32)
        self.body_id = np.zeros(self.count, dtype=np.int32)

        ei, ej, ek, ec, kinds = [], [], [], [], []
        def add(di, dj, stiffness, damping, kind):
            for j in range(self.ny):
                for i in range(self.nx):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        ei.append(j * self.nx + i); ej.append(nj * self.nx + ni)
                        ek.append(stiffness); ec.append(damping); kinds.append(kind)
        add(1, 0, 95.0, 0.035, 0); add(0, 1, 95.0, 0.035, 0)
        add(1, 1, 42.0, 0.025, 0); add(1, -1, 42.0, 0.025, 0)
        add(2, 0, 1.15, 0.008, 1); add(0, 2, 1.15, 0.008, 1)
        self.edge_i, self.edge_j = np.asarray(ei, dtype=np.int32), np.asarray(ej, dtype=np.int32)
        self.edge_k, self.edge_c = np.asarray(ek), np.asarray(ec)
        self.edge_kind = np.asarray(kinds, dtype=np.int8)
        di = (self.edge_j % self.nx) - (self.edge_i % self.nx)
        dj = (self.edge_j // self.nx) - (self.edge_i // self.nx)
        self.rest_length = np.sqrt((di * self.dx) ** 2 + (dj * self.dy) ** 2)
        self.initial_rest_length = self.rest_length.copy()
        a, b = np.triu_indices(self.count, k=1)
        ai, aj, bi, bj = a % self.nx, a // self.nx, b % self.nx, b // self.nx
        keep = (np.abs(ai - bi) > 1) | (np.abs(aj - bj) > 1)
        self.self_i, self.self_j = a[keep].astype(np.int32), b[keep].astype(np.int32)
        self.self_distance = 1.8 * self.thickness
        self.max_dt = 1.0 / 600.0
        self._external_force = self.mass[:, None] * self.gravity[None, :]

    @staticmethod
    def _smoothstep(u):
        u = min(1.0, max(0.0, u)); return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)

    def _wall_state(self, time):
        start, finish, travel = 0.25, 5.0, 0.90
        if time <= start: return 1.55, 0.0
        if time >= finish: return 1.55 - travel, 0.0
        u = (time - start) / (finish - start); s, ds = self._smoothstep(u)
        return 1.55 - travel * s, travel * ds / (finish - start)

    def _spring_forces(self):
        d = self.position[self.edge_j] - self.position[self.edge_i]
        length = np.linalg.norm(d, axis=1); direction = d / np.maximum(length, 1e-12)[:, None]
        rel = self.velocity[self.edge_j] - self.velocity[self.edge_i]
        normal_speed = np.einsum("ij,ij->i", rel, direction)
        magnitude = self.edge_k * (length - self.rest_length) + self.edge_c * normal_speed
        pair = magnitude[:, None] * direction
        force = self._external_force.copy()
        np.add.at(force, self.edge_i, pair); np.add.at(force, self.edge_j, -pair)
        bend = (self.edge_kind == 1) & (length < 0.78 * self.rest_length)
        if np.any(bend):
            floor = 0.42 * self.initial_rest_length[bend]
            self.rest_length[bend] += 0.018 * (np.maximum(length[bend], floor) - self.rest_length[bend])
        return force

    def _apply_boundaries(self, time):
        p, v = self.position, self.velocity
        hit = p[:, 2] < self.radius
        if np.any(hit):
            p[hit, 2] = self.radius; falling = hit & (v[:, 2] < 0.0)
            v[falling, 2] *= -0.035; v[hit, 0:2] *= 0.965
        center, inward = self._wall_state(time); limit = center - 0.12 - self.radius
        eligible = ((p[:, 2] >= -self.radius) & (p[:, 2] <= 2.6 + self.radius)
                    & (np.abs(p[:, 1]) <= 1.5 + self.radius))
        left, right = eligible & (p[:, 0] < -limit), eligible & (p[:, 0] > limit)
        if np.any(left):
            p[left, 0] = -limit; v[left, 0] = np.maximum(v[left, 0], inward); v[left, 1:] *= 0.985
        if np.any(right):
            p[right, 0] = limit; v[right, 0] = np.minimum(v[right, 0], -inward); v[right, 1:] *= 0.985

    def _apply_self_contact(self):
        d = self.position[self.self_i] - self.position[self.self_j]
        distance = np.linalg.norm(d, axis=1); active = distance < self.self_distance
        if not np.any(active): return
        ii, jj, dist = self.self_i[active], self.self_j[active], distance[active]
        normal = d[active] / np.maximum(dist, 1e-12)[:, None]
        normal[dist < 1e-10] = np.array([0.0, 0.0, 1.0])
        correction = 0.5 * (self.self_distance - dist)[:, None] * normal
        accumulated = np.zeros_like(self.position)
        np.add.at(accumulated, ii, correction); np.add.at(accumulated, jj, -correction)
        norm = np.linalg.norm(accumulated, axis=1)
        scale = np.minimum(1.0, 0.55 * self.self_distance / np.maximum(norm, 1e-12))
        self.position += accumulated * scale[:, None]
        closing = np.einsum("ij,ij->i", self.velocity[ii] - self.velocity[jj], normal)
        impulse = np.maximum(-closing, 0.0)[:, None] * 0.5 * normal
        np.add.at(self.velocity, ii, impulse); np.add.at(self.velocity, jj, -impulse)

    def _substep(self, dt):
        force = self._spring_forces()
        self.velocity += dt * force * self.inv_mass[:, None]
        self.velocity *= np.exp(-0.32 * dt)
        speed = np.linalg.norm(self.velocity, axis=1); fast = speed > 15.0
        if np.any(fast): self.velocity[fast] *= (15.0 / speed[fast])[:, None]
        self.position += dt * self.velocity
        next_time = self.time + dt
        self._apply_boundaries(next_time); self._apply_self_contact(); self._apply_boundaries(next_time)
        self.time, self.steps = next_time, self.steps + 1

    def _deformation_gradient(self):
        grid = self.position.reshape(self.ny, self.nx, 3)
        tx = np.gradient(grid, self.dx, axis=1, edge_order=2)
        ty = np.gradient(grid, self.dy, axis=0, edge_order=2)
        metric = np.empty((self.ny, self.nx, 2, 2))
        metric[..., 0, 0] = np.einsum("ijk,ijk->ij", tx, tx)
        metric[..., 0, 1] = metric[..., 1, 0] = np.einsum("ijk,ijk->ij", tx, ty)
        metric[..., 1, 1] = np.einsum("ijk,ijk->ij", ty, ty)
        values, vectors = np.linalg.eigh(metric)
        values = np.sqrt(np.maximum(values, 1e-10))
        stretch = np.einsum("...ik,...k,...jk->...ij", vectors, values, vectors)
        return stretch.reshape(self.count, 2, 2)

    @staticmethod
    def _box_sdf(points, center, half):
        q = np.abs(points - center[None, :]) - half[None, :]
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

    def _contact_distances(self):
        wall_x, _ = self._wall_state(self.time)
        ground = self._box_sdf(self.position, np.array([0., 0., -.05]), np.array([2.4, 2., .05])) - self.radius
        left = self._box_sdf(self.position, np.array([-wall_x, 0., 1.3]), np.array([.12, 1.5, 1.3])) - self.radius
        right = self._box_sdf(self.position, np.array([wall_x, 0., 1.3]), np.array([.12, 1.5, 1.3])) - self.radius
        d = self.position[self.self_i] - self.position[self.self_j]
        gap = np.linalg.norm(d, axis=1) - self.self_distance
        return np.concatenate((ground, left, right, gap[gap < 2.5 * self.self_distance]))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time + 1e-12 < target:
            self._substep(min(self.max_dt, target - self.time))
        self.time = target
        frame = StateFrame(
            time=target, position=self.position.copy(), velocity=self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass.copy(), "boundary_vertex": self.boundary_vertex.copy(),
                        "deformation_gradient": self._deformation_gradient()},
            fields={"contact_signed_distance": self._contact_distances(),
                    "cumulative_mass_source": np.zeros(1), "external_force": self._external_force.copy()},
            scalars={"contact_length_scale": self.thickness})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "symplectic mass-spring shell with projected contact",
                "substeps": int(self.steps), "particle_count": int(self.count),
                "spring_count": int(len(self.edge_i)), "simulated_time": float(self.time)}
