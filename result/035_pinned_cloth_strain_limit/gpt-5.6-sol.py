"""Deterministic position-based cloth solver for the sphere drape scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small PBD cloth model with strain limits and sphere/ground contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        cfg = scene.config
        cloth = cfg["cloth"]
        state = scene.initial_state()
        self.x = np.asarray(state.position, dtype=np.float64).copy()
        self.v = np.asarray(state.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(state.mass, dtype=np.float64).copy()
        self.rest_x = self.x.copy()
        self.uv = np.asarray(state.attributes["uv"], dtype=np.float64).copy()
        self.rows, self.cols = (int(v) for v in cloth["grid_shape"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.base_dt = float(cfg["solver"]["substep"])
        self.iterations = int(cfg["solver"]["constraint_iterations"])
        self.lower = float(cloth.get("target_strain_lower", cloth["strain_lower"]))
        self.upper = float(cloth.get("target_strain_upper", cloth["strain_upper"]))
        self.damping = float(cloth["damping"])
        self.friction = float(cloth["friction"])
        self.thickness = float(cloth["thickness"])
        self.contact_offset = 0.5 * self.thickness
        self.sphere_center = np.asarray(cfg["sphere"]["center"], dtype=np.float64)
        self.sphere_radius = float(cfg["sphere"]["radius"])
        self.ground_height = float(cfg["ground"]["height"])

        ids = np.arange(self.rows * self.cols, dtype=np.int64).reshape(self.rows, self.cols)
        self.edge_i = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel(),
                                      ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()))
        self.edge_j = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel(),
                                      ids[1:, 1:].ravel(), ids[1:, :-1].ravel()))
        self.edge_rest = np.linalg.norm(self.rest_x[self.edge_j] - self.rest_x[self.edge_i], axis=1)
        self.bend_i = np.concatenate((ids[:, :-2].ravel(), ids[:-2, :].ravel()))
        self.bend_j = np.concatenate((ids[:, 2:].ravel(), ids[2:, :].ravel()))
        self.bend_rest = np.linalg.norm(self.rest_x[self.bend_j] - self.rest_x[self.bend_i], axis=1)
        self.time = 0.0
        self.step_count = 0
        self._last_min_gap = float(np.min(self._contact_distance(self.x)))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while target - self.time > 1e-12:
            dt = min(self.base_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        frame = StateFrame(self.time, self.x.copy(), self.v.copy(), mass=self.mass.copy(),
                           attributes={"deformation_gradient": self._deformation_gradient(),
                                       "mass_quantity": self.mass.copy(), "uv": self.uv.copy()},
                           fields={"contact_signed_distance": self._contact_distance(self.x),
                                   "cumulative_mass_source": np.zeros(1)},
                           scalars={"contact_length_scale": self.thickness})
        frame.validate()
        return frame

    def _step(self, dt: float) -> None:
        old_x = self.x.copy()
        free_v = self.v * math.exp(-self.damping * dt) + self.gravity * dt
        p = old_x + free_v * dt
        for _ in range(self.iterations):
            self._project_strain(p)
            self._project_bending(p, 0.018)
            self._project_contacts(p)
        new_v = (p - old_x) / dt
        self._apply_contact_velocity(p, new_v, free_v)
        speed = np.linalg.norm(new_v, axis=1)
        fast = speed > 20.0
        if np.any(fast):
            new_v[fast] *= (20.0 / speed[fast])[:, None]
        self.x, self.v = p, new_v
        self.step_count += 1
        self._last_min_gap = float(np.min(self._contact_distance(p)))

    def _project_strain(self, p: np.ndarray) -> None:
        d = p[self.edge_j] - p[self.edge_i]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1e-12)
        target = np.clip(length / self.edge_rest, self.lower, self.upper)
        violation = length - target * self.edge_rest
        active = np.abs(violation) > 1e-12
        if np.any(active):
            corr = (0.5 * violation[active] / safe[active])[:, None] * d[active]
            self._scatter_pair_corrections(p, self.edge_i[active], self.edge_j[active], corr, 0.96)

    def _project_bending(self, p: np.ndarray, strength: float) -> None:
        d = p[self.bend_j] - p[self.bend_i]
        length = np.linalg.norm(d, axis=1)
        corr = (0.5 * (length - self.bend_rest) / np.maximum(length, 1e-12))[:, None] * d
        self._scatter_pair_corrections(p, self.bend_i, self.bend_j, corr, strength)

    @staticmethod
    def _scatter_pair_corrections(p, i, j, corr, relaxation):
        delta = np.zeros_like(p)
        count = np.zeros(len(p))
        np.add.at(delta, i, corr)
        np.add.at(delta, j, -corr)
        np.add.at(count, i, 1.0)
        np.add.at(count, j, 1.0)
        used = count > 0
        p[used] += relaxation * delta[used] / count[used, None]

    def _project_contacts(self, p: np.ndarray) -> None:
        radius = self.sphere_radius + self.contact_offset
        radial = p - self.sphere_center
        distance = np.linalg.norm(radial, axis=1)
        inside = distance < radius
        if np.any(inside):
            normal = radial[inside] / np.maximum(distance[inside, None], 1e-12)
            p[inside] = self.sphere_center + radius * normal
        floor = self.ground_height + self.contact_offset
        p[p[:, 2] < floor, 2] = floor

    def _apply_contact_velocity(self, p, velocity, free_velocity) -> None:
        radius = self.sphere_radius + self.contact_offset
        radial = p - self.sphere_center
        distance = np.linalg.norm(radial, axis=1)
        on = distance <= radius + 2e-7
        if np.any(on):
            n = radial[on] / np.maximum(distance[on, None], 1e-12)
            vel = velocity[on]
            vn = np.sum(vel * n, axis=1)
            vel -= np.minimum(vn, 0.0)[:, None] * n
            tangent = vel - np.sum(vel * n, axis=1)[:, None] * n
            ts = np.linalg.norm(tangent, axis=1)
            impact = np.maximum(-np.sum(free_velocity[on] * n, axis=1), 0.0)
            scale = np.maximum(0.0, 1.0 - self.friction * impact / np.maximum(ts, 1e-12))
            velocity[on] = vel - tangent + tangent * scale[:, None]
        floor = self.ground_height + self.contact_offset
        on_floor = p[:, 2] <= floor + 2e-7
        if np.any(on_floor):
            velocity[on_floor, 2] = np.maximum(velocity[on_floor, 2], 0.0)
            tangent = velocity[on_floor, :2]
            ts = np.linalg.norm(tangent, axis=1)
            impact = np.maximum(-free_velocity[on_floor, 2], 0.0)
            scale = np.maximum(0.0, 1.0 - self.friction * impact / np.maximum(ts, 1e-12))
            velocity[on_floor, :2] = tangent * scale[:, None]

    def _contact_distance(self, p: np.ndarray) -> np.ndarray:
        sphere = np.linalg.norm(p - self.sphere_center, axis=1) - self.sphere_radius - self.contact_offset
        floor = p[:, 2] - self.ground_height - self.contact_offset
        return np.minimum(sphere, floor)

    def _deformation_gradient(self) -> np.ndarray:
        c = self.x.reshape(self.rows, self.cols, 3)
        r = self.rest_x.reshape(self.rows, self.cols, 3)
        li, ri = np.maximum(np.arange(self.cols) - 1, 0), np.minimum(np.arange(self.cols) + 1, self.cols - 1)
        di, ui = np.maximum(np.arange(self.rows) - 1, 0), np.minimum(np.arange(self.rows) + 1, self.rows - 1)
        cu, cv = (c[:, ri] - c[:, li]).reshape(-1, 3), (c[ui, :] - c[di, :]).reshape(-1, 3)
        ru, rv = (r[:, ri] - r[:, li]).reshape(-1, 3), (r[ui, :] - r[di, :]).reshape(-1, 3)
        q1 = ru / np.maximum(np.linalg.norm(ru, axis=1, keepdims=True), 1e-12)
        n0 = np.cross(ru, rv); n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-12)
        q2 = np.cross(n0, q1)
        a = np.stack((np.sum(q1 * ru, 1), np.sum(q1 * rv, 1), np.sum(q2 * ru, 1), np.sum(q2 * rv, 1)), 1).reshape(-1, 2, 2)
        ds = np.stack((cu, cv), axis=2)
        basis = np.stack((q1, q2), axis=2)
        tang = ds @ np.linalg.inv(a) @ np.swapaxes(basis, 1, 2)
        normal = np.cross(cu, cv); nl = np.linalg.norm(normal, axis=1, keepdims=True)
        normal /= np.maximum(nl, 1e-12)
        bad = nl[:, 0] <= 1e-12
        normal[bad] = n0[bad]
        return tang + normal[:, :, None] * n0[:, None, :]

    def diagnostics(self) -> dict[str, object]:
        return {"time": float(self.time), "substeps": int(self.step_count),
                "minimum_contact_gap": float(self._last_min_gap),
                "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all())}
