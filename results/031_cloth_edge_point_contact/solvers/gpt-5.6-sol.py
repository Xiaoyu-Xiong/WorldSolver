"""Self-contained spring-mass cloth solver with unilateral contacts."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.thickness = float(cloth["thickness"])
        self.mass = np.full(self.n, float(cloth["total_mass"]) / self.n)
        self.inv_mass = 1.0 / self.mass
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.rest = self.x.copy()
        self.time = 0.0
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.friction = float(cfg["friction"]["coefficient"])
        self.edge_center = np.asarray(cfg["sharp_edge"]["center"], dtype=float)
        self.edge_half = np.asarray(cfg["sharp_edge"]["half_extents"], dtype=float)
        self.point_tip = np.asarray(cfg["sharp_point"]["tip"], dtype=float)
        self.point_h = float(cfg["sharp_point"]["height"])
        self.point_r = float(cfg["sharp_point"]["base_radius"])
        self.ground_z = float(cfg["ground"]["center"][2]) + float(cfg["ground"]["half_extents"][2])
        edges, lengths = [], []

        def add(a, b):
            edges.append((a, b))
            lengths.append(np.linalg.norm(self.rest[b] - self.rest[a]))

        for iy in range(self.ny):
            for ix in range(self.nx):
                a = iy * self.nx + ix
                if ix + 1 < self.nx:
                    add(a, a + 1)
                if iy + 1 < self.ny:
                    add(a, a + self.nx)
                if ix + 1 < self.nx and iy + 1 < self.ny:
                    add(a, a + self.nx + 1)
                if ix > 0 and iy + 1 < self.ny:
                    add(a, a + self.nx - 1)
                if ix + 2 < self.nx:
                    add(a, a + 2)
                if iy + 2 < self.ny:
                    add(a, a + 2 * self.nx)
        self.edges = np.asarray(edges, dtype=np.int32)
        self.rest_len = np.asarray(lengths, dtype=float)
        shortest = np.min(self.rest_len)
        self.constraint_stiffness = np.where(self.rest_len > 1.5 * shortest, 0.48, 0.94)
        self.air_damp = 0.42
        self._last_frame = self._make_frame(0.0)

    def _cone(self, p):
        q = p - self.point_tip
        z, rho = q[:, 2], np.linalg.norm(q[:, :2], axis=1)
        slope = self.point_r / max(self.point_h, 1e-9)
        side = (rho - slope * (-z)) / np.sqrt(1.0 + slope * slope)
        side_n = np.column_stack((q[:, 0] / np.maximum(rho, 1e-8), q[:, 1] / np.maximum(rho, 1e-8), np.full(len(p), slope)))
        side_n /= np.maximum(np.linalg.norm(side_n, axis=1), 1e-8)[:, None]
        active = (z <= 0.0) & (z >= -self.point_h)
        tq = np.maximum(np.linalg.norm(q, axis=1), 1e-8)
        tip_n = q / tq[:, None]
        return np.where(active, side - self.thickness, tq - self.thickness), np.where(active[:, None], side_n, tip_n)

    def _project_contacts(self):
        zmin = self.ground_z + self.thickness
        hit = self.x[:, 2] < zmin
        self.x[hit, 2] = zmin
        if np.any(hit):
            self.v[hit, 2] = np.maximum(self.v[hit, 2], 0.0)
            self.v[hit, :2] *= max(0.0, 1.0 - 0.12 * self.friction)
        lo = self.edge_center - self.edge_half - self.thickness
        hi = self.edge_center + self.edge_half + self.thickness
        inside = np.all((self.x >= lo) & (self.x <= hi), axis=1)
        for idx in np.flatnonzero(inside):
            q = self.x[idx]
            pen = np.minimum(q - lo, hi - q)
            ax = int(np.argmin(pen))
            sign = -1.0 if q[ax] - lo[ax] < hi[ax] - q[ax] else 1.0
            self.x[idx, ax] = lo[ax] if sign < 0 else hi[ax]
            vn = self.v[idx, ax] * sign
            if vn < 0.0:
                self.v[idx, ax] -= vn * sign
            self.v[idx] *= max(0.0, 1.0 - 0.08 * self.friction)
        gap, normal = self._cone(self.x)
        hit = gap < 0.0
        if np.any(hit):
            self.x[hit] -= gap[hit, None] * normal[hit]
            vn = np.sum(self.v[hit] * normal[hit], axis=1)
            self.v[hit] -= np.minimum(vn, 0.0)[:, None] * normal[hit]
            self.v[hit] *= max(0.0, 1.0 - 0.06 * self.friction)

    def _step(self, dt):
        if dt <= 0.0:
            return
        nsub = max(1, int(np.ceil(dt / 0.0034)))
        h = dt / nsub
        for _ in range(nsub):
            old = self.x.copy()
            self.v += h * self.gravity
            self.v *= np.exp(-self.air_damp * h)
            self.x += h * self.v
            a, b = self.edges[:, 0], self.edges[:, 1]
            for _iteration in range(4):
                d = self.x[b] - self.x[a]
                length = np.linalg.norm(d, axis=1)
                corr = (0.5 * self.constraint_stiffness * (length - self.rest_len) / np.maximum(length, 1e-9))[:, None] * d
                delta = np.zeros_like(self.x)
                np.add.at(delta, a, corr)
                np.add.at(delta, b, -corr)
                self.x += delta
                self._project_contacts()
            self.v = (self.x - old) / h
            self._project_contacts()

    def _make_frame(self, time):
        X = self.rest.reshape(self.ny, self.nx, 3)
        Y = self.x.reshape(self.ny, self.nx, 3)
        F = np.tile(np.eye(3), (self.n, 1, 1))
        dx = np.linalg.norm(X[0, 1] - X[0, 0])
        dy = np.linalg.norm(X[1, 0] - X[0, 0])
        du, dv = np.zeros_like(Y), np.zeros_like(Y)
        du[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / (2 * dx)
        du[:, 0] = (Y[:, 1] - Y[:, 0]) / dx
        du[:, -1] = (Y[:, -1] - Y[:, -2]) / dx
        dv[1:-1] = (Y[2:] - Y[:-2]) / (2 * dy)
        dv[0] = (Y[1] - Y[0]) / dy
        dv[-1] = (Y[-1] - Y[-2]) / dy
        F[:, :, 0], F[:, :, 1] = du.reshape(self.n, 3), dv.reshape(self.n, 3)
        normal = np.cross(F[:, :, 0], F[:, :, 1])
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-8)[:, None]
        F[:, :, 2] = normal
        edge_gap = np.maximum.reduce([
            self.edge_center[0] - self.edge_half[0] - self.thickness - self.x[:, 0],
            self.x[:, 0] - self.edge_center[0] - self.edge_half[0] - self.thickness,
            np.zeros(self.n),
        ])
        ground_gap = self.x[:, 2] - self.ground_z - self.thickness
        cone_gap, _ = self._cone(self.x)
        contact = np.minimum(np.minimum(ground_gap, cone_gap), edge_gap)
        frame = StateFrame(
            float(time), self.x.copy(), self.v.copy(), mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": F},
            fields={"contact_signed_distance": contact.copy(), "cumulative_mass_source": np.asarray([0.0])},
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing time")
        self._step(target - self.time)
        self.time = target
        self._last_frame = self._make_frame(target)
        return self._last_frame

    def diagnostics(self):
        return {"time": float(self.time), "particle_count": int(self.n), "max_speed": float(np.max(np.linalg.norm(self.v, axis=1)))}
