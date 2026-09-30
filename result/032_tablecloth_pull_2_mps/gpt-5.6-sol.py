"""Deterministic spring-mass tablecloth and rigid-body solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.t = 0.0
        self.p = self.v = self.p0 = self.v0 = None
        self.mass = self.body_id = None
        self.orientation = self.angular_velocity = None
        self.attributes = {}
        self.base_fields = {}
        self.base_scalars = {}
        self.scene = None
        self.cfg = {}
        self._external = None
        self._strain = 0.0

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.t = float(initial.time)
        self.p = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.p0, self.v0 = self.p.copy(), self.v.copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.orientation = np.asarray(initial.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(initial.angular_velocity, dtype=float).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self.base_fields = {k: np.asarray(v).copy() for k, v in initial.fields.items()}
        self.base_scalars = {k: float(v) for k, v in initial.scalars.items()}
        self.ncloth = int(np.count_nonzero(self.body_id == 0))
        self.nr = len(self.p) - self.ncloth
        self.grid_nx, self.grid_ny = 17, 13
        cloth = self.p[: self.ncloth]
        self.dx = float(np.linalg.norm(cloth[1] - cloth[0]))
        self.dy = float(np.linalg.norm(cloth[self.grid_nx] - cloth[0]))
        self._edges = []
        for j in range(self.grid_ny):
            for i in range(self.grid_nx):
                a = j * self.grid_nx + i
                if i + 1 < self.grid_nx:
                    self._edges.append((a, a + 1, self.dx, 620.0))
                if j + 1 < self.grid_ny:
                    self._edges.append((a, a + self.grid_nx, self.dy, 620.0))
                if i + 1 < self.grid_nx and j + 1 < self.grid_ny:
                    rest = float(np.hypot(self.dx, self.dy))
                    self._edges.extend(((a, a + self.grid_nx + 1, rest, 260.0),
                                        (a + 1, a + self.grid_nx, rest, 260.0)))
                if i + 2 < self.grid_nx:
                    self._edges.append((a, a + 2, 2.0 * self.dx, 90.0))
                if j + 2 < self.grid_ny:
                    self._edges.append((a, a + 2 * self.grid_nx, 2.0 * self.dy, 90.0))
        self.pull_speed = float(self.cfg.get("pull_speed", 2.0))
        self.pull_duration = float(self.cfg.get("pull_duration", 0.9))
        self.table_top = float(self.cfg.get("table_top", 0.78))
        self.gravity = np.asarray(self.cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._external = self.mass[:, None] * self.gravity[None, :]

    def _rigid_geometry(self, k: int) -> tuple[float, float, float, float]:
        return ((0.085, 0.085, 0.36, 0.42), (0.070, 0.070, 0.15, 0.38),
                (0.240, 0.240, 0.0175, 0.50), (0.190, 0.110, 0.035, 0.48),
                (0.160, 0.025, 0.012, 0.35), (0.190, 0.022, 0.012, 0.35))[k]

    def _step(self, h: float) -> None:
        force = self.mass[:, None] * self.gravity[None, :]
        max_strain = 0.0
        for a, b, rest, stiffness in self._edges:
            d = self.p[b] - self.p[a]
            length = float(np.linalg.norm(d))
            if length < 1.0e-9:
                continue
            direction = d / length
            extension = length - rest
            relv = float(np.dot(self.v[b] - self.v[a], direction))
            max_strain = max(max_strain, abs(extension) / max(rest, 1.0e-6))
            fvec = np.clip(stiffness * extension + 2.2 * relv, -35.0, 35.0) * direction
            force[a] += fvec
            force[b] -= fvec
        self._strain = max_strain

        for r in range(self.nr):
            idx = self.ncloth + r
            rx, ry, half_h, mu = self._rigid_geometry(r)
            rp = self.p[idx]
            near = [j for j in range(self.ncloth)
                    if abs(self.p[j, 0] - rp[0]) < rx and abs(self.p[j, 1] - rp[1]) < ry]
            if near:
                jj = np.asarray(near, dtype=int)
                cloth_vel = np.mean(self.v[jj], axis=0)
                gap = rp[2] - half_h - (self.table_top + 0.004)
                contact = np.clip(1.0 - abs(gap) / 0.06, 0.0, 1.0)
                drag = mu * self.mass[idx] * 9.81 * contact
                force[idx, :2] += drag * (cloth_vel[:2] - self.v[idx, :2])
                bottom = rp[2] - half_h - 0.004
                for j in jj:
                    penetration = bottom - self.p[j, 2]
                    if penetration > 0.0:
                        normal = min(18.0 * penetration + 0.25 * max(0.0, -self.v[j, 2]), 18.0)
                        force[j, 2] += normal
                        force[idx, 2] -= normal * (self.mass[idx] / max(self.mass[j], 1.0e-6)) * 0.15

        acceleration = np.clip(force / self.mass[:, None], -1800.0, 1800.0)
        self.v += h * acceleration
        self.v *= np.exp(-1.8 * h)
        self.v = np.clip(self.v, -22.0, 22.0)
        self.p += h * self.v

        for i in range(self.ncloth):
            x, y = self.p[i, :2]
            floor = self.table_top + 0.004 if abs(x) <= 1.25 and abs(y) <= 0.90 else 0.015
            if self.p[i, 2] < floor:
                self.p[i, 2] = floor
                if self.v[i, 2] < 0.0:
                    self.v[i, 2] *= -0.12
                self.v[i, :2] *= 0.94
        for r in range(self.nr):
            i = self.ncloth + r
            _, _, half_h, mu = self._rigid_geometry(r)
            support = (self.table_top + half_h if abs(self.p[i, 0]) <= 1.25 and abs(self.p[i, 1]) <= 0.90
                       else half_h + 0.015)
            if self.p[i, 2] < support:
                self.p[i, 2] = support
                if self.v[i, 2] < 0.0:
                    self.v[i, 2] *= -0.08
                self.v[i, :2] *= max(0.0, 1.0 - 3.0 * mu * h)

        # Prescribed hand motion of the right cloth edge.
        edge = np.arange(self.grid_nx - 1, self.ncloth, self.grid_nx)
        self.p[edge, 0] = self.p0[edge, 0] + self.pull_speed * min(self.t + h, self.pull_duration)
        self.p[edge, 1] = self.p0[edge, 1]
        self.v[edge, 0] = self.pull_speed if self.t + h < self.pull_duration else 0.0
        self.v[edge, 1:] = 0.0

    def _contact_distances(self) -> np.ndarray:
        out = np.zeros(len(self.p), dtype=float)
        for i in range(self.ncloth):
            x, y, z = self.p[i]
            floor = self.table_top + 0.004 if abs(x) <= 1.25 and abs(y) <= 0.90 else 0.015
            gap = z - floor
            for r in range(self.nr):
                j = self.ncloth + r
                rx, ry, hh, _ = self._rigid_geometry(r)
                if abs(x - self.p[j, 0]) < rx and abs(y - self.p[j, 1]) < ry:
                    gap = min(gap, z - (self.p[j, 2] - hh - 0.004))
            out[i] = gap
        for r in range(self.nr):
            i = self.ncloth + r
            _, _, hh, _ = self._rigid_geometry(r)
            support = self.table_top + hh if abs(self.p[i, 0]) <= 1.25 and abs(self.p[i, 1]) <= 0.90 else hh + 0.015
            out[i] = self.p[i, 2] - support
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def _frame(self) -> StateFrame:
        fields = {k: np.asarray(v).copy() for k, v in self.base_fields.items()}
        fields["contact_signed_distance"] = self._contact_distances()
        fields["constraint_value"] = np.asarray(
            [self.p[self.ncloth - 1, 0] - self.p0[self.ncloth - 1, 0], self._strain], dtype=float)
        fields["constraint_target"] = np.asarray(self.base_fields.get("constraint_target", [0.0, 0.0]), dtype=float).copy()
        fields["cumulative_mass_source"] = np.asarray(self.base_fields.get("cumulative_mass_source", [0.0]), dtype=float).copy()
        fields["external_force"] = np.asarray(self._external, dtype=float).copy()
        attrs = {k: np.asarray(v).copy() for k, v in self.attributes.items()}
        attrs["mass_quantity"] = np.asarray(self.mass, dtype=float).copy()
        return StateFrame(float(self.t), self.p.copy(), self.v.copy(), orientation=self.orientation.copy(),
                          angular_velocity=self.angular_velocity.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(),
                          attributes=attrs, fields=fields, scalars=self.base_scalars.copy())

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.t - 1.0e-9:
            raise ValueError("advance_to requires a finite nondecreasing time")
        while self.t < target - 1.0e-10:
            h = min(0.0025, target - self.t)
            self._step(h)
            self.t += h
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "semi_implicit_cloth_rigid", "cloth_vertices": int(self.ncloth) if self.p is not None else 0,
                "rigid_bodies": int(self.nr) if self.p is not None else 0, "time": float(self.t)}
