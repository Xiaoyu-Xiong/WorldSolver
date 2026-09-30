"""Deterministic mass-spring solver for the knitted legwarmer scene."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        cloth = self.config["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.radius = 0.42
        self.thickness = float(cloth["thickness"])
        self.mass = np.full(self.n, float(cloth["total_mass"]) / self.n)
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.rest = self.position.copy()
        self.time = float(initial.time)
        self._last_contact = np.zeros(self.n, dtype=float)
        self._last_force = self.mass[:, None] * self.gravity[None, :]
        self._build_edges()
        probe = np.zeros_like(self.position)
        self._contact(self.time, probe)
        self._last_contact = self._contact(self.time, probe)

    def _build_edges(self) -> None:
        edges: list[tuple[int, int, float, float]] = []
        self.dx = float(np.linalg.norm(self.rest[self.ny] - self.rest[0]))
        self.ds = 2.0 * self.radius * math.sin(math.pi / self.ny)
        self.dv_span = 2.0 * self.radius * math.sin(2.0 * math.pi / self.ny)
        for i in range(self.nx):
            for j in range(self.ny):
                a = i * self.ny + j
                b = i * self.ny + ((j + 1) % self.ny)
                edges.append((a, b, float(np.linalg.norm(self.rest[b] - self.rest[a])), 105.0))
                if i + 1 < self.nx:
                    b = (i + 1) * self.ny + j
                    edges.append((a, b, float(np.linalg.norm(self.rest[b] - self.rest[a])), 145.0))
                    b = (i + 1) * self.ny + ((j + 1) % self.ny)
                    edges.append((a, b, float(np.linalg.norm(self.rest[b] - self.rest[a])), 42.0))
        self.edges = np.asarray(edges, dtype=float)

    def _bend_angle(self, t: float) -> float:
        phase = min(max(t / max(float(self.config["duration"]), 1e-6), 0.0), 1.0)
        return 1.12 * math.sin(math.pi * phase) ** 2

    def _segments(self, t: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
        a = self._bend_angle(t) * 0.5
        knee = np.array([0.0, 0.0, 1.10])
        upper_dir = np.array([-math.cos(a), 0.0, math.sin(a)])
        lower_dir = np.array([math.cos(a), 0.0, math.sin(a)])
        upper_end = knee + 1.28 * upper_dir
        lower_end = knee + 1.28 * lower_dir
        segs = [(knee, upper_end, 0.34), (knee, lower_end, 0.31)]
        # Rotate the frozen foot transform with the lower limb.  At t=0 its
        # center is exactly (1.55, -0.10, 0.92) and its capsule axis is x.
        foot_center = knee + np.array([
            1.55 * math.cos(a) + 0.18 * math.sin(a),
            -0.10,
            1.55 * math.sin(a) - 0.18 * math.cos(a),
        ])
        segs.append((foot_center - 0.36 * lower_dir, foot_center + 0.36 * lower_dir, 0.25))
        return segs

    @staticmethod
    def _point_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
        ab = b - a
        den = float(np.dot(ab, ab))
        u = 0.0 if den < 1e-12 else float(np.dot(point - a, ab) / den)
        u = min(1.0, max(0.0, u))
        dvec = point - (a + u * ab)
        d = float(np.linalg.norm(dvec))
        if d < 1e-10:
            dvec, d = np.array([0.0, 1.0, 0.0]), 1e-10
        return d, dvec / d

    def _contact(self, t: float, force: np.ndarray) -> np.ndarray:
        segs = self._segments(t)
        knee = np.array([0.0, 0.0, 1.10])
        distances = np.empty(self.n, dtype=float)
        kn, kt, margin = 220.0, 0.45, self.thickness * 1.15
        for pidx, p in enumerate(self.position):
            best_d, best_n = p[2], np.array([0.0, 0.0, 1.0])
            for aa, bb, rr in segs:
                raw, normal = self._point_segment(p, aa, bb)
                d = raw - rr
                if d < best_d:
                    best_d, best_n = d, normal
            d_k = float(np.linalg.norm(p - knee) - 0.36)
            if d_k < best_d:
                vec = p - knee
                norm = float(np.linalg.norm(vec))
                best_d = d_k
                best_n = vec / norm if norm > 1e-10 else np.array([0.0, 1.0, 0.0])
            distances[pidx] = best_d
            if best_d < margin:
                penetration = margin - best_d
                rel = self.velocity[pidx]
                vn = float(np.dot(rel, best_n))
                normal_force = kn * penetration - 0.60 * min(vn, 0.0)
                tangential = rel - vn * best_n
                friction = -kt * tangential
                fmag, cap = float(np.linalg.norm(friction)), 0.55 * max(normal_force, 0.0)
                if fmag > cap > 0.0:
                    friction *= cap / fmag
                force[pidx] += normal_force * best_n + friction
                if best_d < 0.0:
                    self.position[pidx] += (-best_d + 1e-4) * best_n
                    vn_now = float(np.dot(self.velocity[pidx], best_n))
                    if vn_now < 0.0:
                        self.velocity[pidx] -= 0.75 * vn_now * best_n
        return distances

    def _step(self, dt: float, t: float) -> None:
        force = self.mass[:, None] * self.gravity[None, :]
        for aa_f, bb_f, rest_len, stiffness in self.edges:
            aa, bb = int(aa_f), int(bb_f)
            delta = self.position[bb] - self.position[aa]
            length = float(np.linalg.norm(delta))
            if length < 1e-9:
                continue
            direction = delta / length
            relv = self.velocity[bb] - self.velocity[aa]
            fs = stiffness * (length - rest_len) + 0.10 * float(np.dot(relv, direction))
            f = fs * direction
            force[aa] += f
            force[bb] -= f
        kb = 5.0
        for i in range(self.nx - 2):
            for j in range(self.ny):
                a, b, c = i * self.ny + j, (i + 1) * self.ny + j, (i + 2) * self.ny + j
                curvature = self.position[a] - 2.0 * self.position[b] + self.position[c]
                force[a] -= kb * curvature
                force[b] += 2.0 * kb * curvature
                force[c] -= kb * curvature
        self._last_contact = self._contact(t, force)
        self._last_force = force.copy()
        self.velocity += dt * force / self.mass[:, None]
        self.velocity *= math.exp(-1.8 * dt)
        self.position += dt * self.velocity
        self._end_coupling(t, dt)

    def _end_coupling(self, t: float, dt: float) -> None:
        segs = self._segments(t)
        for ring, (a, b, _radius) in ((0, segs[0]), (self.nx - 1, segs[1])):
            center = b
            axis = (b - a) / np.linalg.norm(b - a)
            side = np.array([0.0, 1.0, 0.0])
            binormal = np.cross(axis, side)
            ang = np.arange(self.ny) * 2.0 * math.pi / self.ny
            target = center + self.radius * (np.cos(ang)[:, None] * side + np.sin(ang)[:, None] * binormal)
            ids = slice(ring * self.ny, (ring + 1) * self.ny)
            self.velocity[ids] += (dt * 1500.0) * (target - self.position[ids])

    def _deformation_gradient(self) -> np.ndarray:
        out = np.zeros((self.n, 2, 2), dtype=float)
        for i in range(self.nx):
            im, ip = max(i - 1, 0), min(i + 1, self.nx - 1)
            den = (ip - im) * self.dx if ip != im else self.dx
            for j in range(self.ny):
                jm, jp = (j - 1) % self.ny, (j + 1) % self.ny
                du = (self.position[ip * self.ny + j] - self.position[im * self.ny + j]) / den
                dv = (self.position[i * self.ny + jp] - self.position[i * self.ny + jm]) / self.dv_span
                ang = 2.0 * math.pi * j / self.ny
                et = np.array([0.0, -math.sin(ang), math.cos(ang)])
                out[i * self.ny + j] = ((np.dot(du, [1.0, 0.0, 0.0]), np.dot(dv, [1.0, 0.0, 0.0])),
                                         (np.dot(du, et), np.dot(dv, et)))
        return out

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("time must be finite and non-negative")
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing times")
        while self.time < target - 1e-12:
            dt = min(1.0 / 600.0, target - self.time)
            self._step(dt, self.time + dt)
            self.time += dt
        # Resolve any last substep penetration, then measure the exported state.
        probe = np.zeros_like(self.position)
        self._contact(target, probe)
        self._last_contact = self._contact(target, probe)
        frame = StateFrame(
            self.time, self.position.copy(), self.velocity.copy(), mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={"mass_quantity": self.mass.copy(), "deformation_gradient": self._deformation_gradient(),
                        "boundary_vertex": np.array([1 if i in (0, self.nx - 1) else 0 for i in range(self.nx) for _ in range(self.ny)], dtype=np.int32)},
            fields={"contact_signed_distance": self._last_contact.copy(), "cumulative_mass_source": np.zeros(1),
                    "external_force": (self.mass[:, None] * self.gravity[None, :]).copy()},
            scalars={"contact_length_scale": self.thickness})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler_mass_spring", "vertex_count": int(self.n)}
