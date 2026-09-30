"""Deterministic position-based cloth solver for the inclined scarf scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring thin sheet with rigid contact, friction, and self contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.zeros(len(self.x), dtype=np.int32)
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.thickness = float(cfg["cloth"]["thickness"])
        self.nx, self.ny = map(int, cfg["cloth"]["grid_shape"])
        self.rest_x = self.x.copy()
        self.time = 0.0
        self.steps = 0

        ids = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        structural = []
        structural.extend(zip(ids[:, :-1].ravel(), ids[:, 1:].ravel()))
        structural.extend(zip(ids[:-1, :].ravel(), ids[1:, :].ravel()))
        shear = []
        shear.extend(zip(ids[:-1, :-1].ravel(), ids[1:, 1:].ravel()))
        shear.extend(zip(ids[1:, :-1].ravel(), ids[:-1, 1:].ravel()))
        bending = []
        bending.extend(zip(ids[:, :-2].ravel(), ids[:, 2:].ravel()))
        bending.extend(zip(ids[:-2, :].ravel(), ids[2:, :].ravel()))
        self.constraints = (
            self._make_constraints(structural, 0.42),
            self._make_constraints(shear, 0.25),
            self._make_constraints(bending, 0.075),
        )

        angle = np.deg2rad(35.0)
        c, s = np.cos(angle), np.sin(angle)
        incline_rotation = np.array(
            ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)),
            dtype=np.float64,
        )
        self.boxes = (
            (np.array((0.0, 0.0, -0.05)), np.array((2.4, 2.0, 0.05)),
             np.eye(3), 0.92),
            (np.array((0.0, 0.0, 0.8)), np.array((2.2, 1.1, 0.08)),
             incline_rotation, 0.20),
        )

    def _make_constraints(self, pairs, stiffness):
        pairs = np.asarray(list(pairs), dtype=np.int32).reshape(-1, 2)
        delta = self.rest_x[pairs[:, 0]] - self.rest_x[pairs[:, 1]]
        return pairs, np.linalg.norm(delta, axis=1), float(stiffness)

    def _solve_distances(self, pairs, rest, stiffness):
        i, j = pairs[:, 0], pairs[:, 1]
        delta = self.x[i] - self.x[j]
        length = np.linalg.norm(delta, axis=1)
        direction = np.zeros_like(delta)
        valid = length > 1.0e-12
        direction[valid] = delta[valid] / length[valid, None]
        wi, wj = self.inv_mass[i], self.inv_mass[j]
        amount = -stiffness * (length - rest) / (wi + wj)
        correction = amount[:, None] * direction
        np.add.at(self.x, i, wi[:, None] * correction)
        np.add.at(self.x, j, -wj[:, None] * correction)

    @staticmethod
    def _box_sdf(points, center, half, rotation):
        local = (points - center) @ rotation
        q = np.abs(local) - half
        return (np.linalg.norm(np.maximum(q, 0.0), axis=1) +
                np.minimum(np.max(q, axis=1), 0.0))

    def _project_box(self, center, half, rotation):
        radius = 0.5 * self.thickness
        local = (self.x - center) @ rotation
        expanded = half + radius
        q = np.abs(local) - expanded
        # Include points already lying on the expanded surface so the final
        # pass retains their contact normals for the friction impulse.
        inside = np.all(q <= 1.0e-10, axis=1)
        indices = np.flatnonzero(inside)
        normals = np.zeros_like(self.x)
        if indices.size:
            face = np.argmax(q[indices], axis=1)
            signs = np.sign(local[indices, face])
            signs[signs == 0.0] = 1.0
            local[indices, face] = signs * expanded[face]
            self.x[indices] = local[indices] @ rotation.T + center
            local_normals = np.zeros((indices.size, 3), dtype=np.float64)
            local_normals[np.arange(indices.size), face] = signs
            normals[indices] = local_normals @ rotation.T
        return inside, normals

    def _self_contact(self):
        separation = 2.2 * self.thickness
        coords = np.floor(self.x / separation).astype(np.int64)
        buckets = {}
        for index, values in enumerate(coords):
            buckets.setdefault(tuple(values), []).append(index)
        offsets = tuple((a, b, c) for a in (-1, 0, 1)
                        for b in (-1, 0, 1) for c in (-1, 0, 1))
        pair_i, pair_j = [], []
        for i, values in enumerate(coords):
            row_i, col_i = divmod(i, self.nx)
            key = tuple(values)
            for offset in offsets:
                near = (key[0] + offset[0], key[1] + offset[1],
                        key[2] + offset[2])
                for j in buckets.get(near, ()):
                    if j <= i:
                        continue
                    row_j, col_j = divmod(j, self.nx)
                    if abs(row_i - row_j) <= 1 and abs(col_i - col_j) <= 1:
                        continue
                    pair_i.append(i)
                    pair_j.append(j)
        if not pair_i:
            return
        i = np.asarray(pair_i, dtype=np.int32)
        j = np.asarray(pair_j, dtype=np.int32)
        delta = self.x[i] - self.x[j]
        distance = np.linalg.norm(delta, axis=1)
        active = (distance < separation) & (distance > 1.0e-10)
        if not np.any(active):
            return
        i, j = i[active], j[active]
        direction = delta[active] / distance[active, None]
        wi, wj = self.inv_mass[i], self.inv_mass[j]
        amount = (separation - distance[active]) / (wi + wj)
        correction = amount[:, None] * direction
        np.add.at(self.x, i, wi[:, None] * correction)
        np.add.at(self.x, j, -wj[:, None] * correction)

    def _substep(self, dt):
        previous = self.x.copy()
        self.v += dt * self.gravity
        self.v *= np.exp(-0.10 * dt)
        self.x += dt * self.v

        for iteration in range(5):
            for pairs, rest, stiffness in self.constraints:
                self._solve_distances(pairs, rest, stiffness)
            for center, half, rotation, _ in self.boxes:
                self._project_box(center, half, rotation)
            if iteration >= 2:
                self._self_contact()

        contact_data = []
        for center, half, rotation, friction in self.boxes:
            mask, normals = self._project_box(center, half, rotation)
            contact_data.append((mask, normals, friction))

        self.v = (self.x - previous) / dt
        for mask, normals, friction in contact_data:
            if not np.any(mask):
                continue
            vn = np.sum(self.v * normals, axis=1)
            inward = mask & (vn < 0.0)
            self.v[inward] -= vn[inward, None] * normals[inward]
            indices = np.flatnonzero(mask)
            normal_part = np.sum(self.v[indices] * normals[indices], axis=1)
            tangent = self.v[indices] - normal_part[:, None] * normals[indices]
            speed = np.linalg.norm(tangent, axis=1)
            reduction = np.maximum(
                0.0, 1.0 - friction * 9.81 * dt / np.maximum(speed, 1.0e-12)
            )
            self.v[indices] = (normal_part[:, None] * normals[indices] +
                               tangent * reduction[:, None])
        self.steps += 1

    def _deformation_gradient(self):
        points = self.x.reshape(self.ny, self.nx, 3)
        du = np.empty_like(points)
        dv = np.empty_like(points)
        du[:, 1:-1] = 0.5 * (points[:, 2:] - points[:, :-2])
        du[:, 0] = points[:, 1] - points[:, 0]
        du[:, -1] = points[:, -1] - points[:, -2]
        dv[1:-1] = 0.5 * (points[2:] - points[:-2])
        dv[0] = points[1] - points[0]
        dv[-1] = points[-1] - points[-2]
        rest_du = np.linalg.norm(self.rest_x[1] - self.rest_x[0])
        rest_dv = np.linalg.norm(self.rest_x[self.nx] - self.rest_x[0])
        a = du.reshape(-1, 3) / rest_du
        b = dv.reshape(-1, 3) / rest_dv
        g11 = np.maximum(np.sum(a * a, axis=1), 1.0e-12)
        g12 = np.sum(a * b, axis=1)
        g22 = np.maximum(np.sum(b * b, axis=1), 1.0e-12)
        f11 = np.sqrt(g11)
        f12 = g12 / f11
        f22 = np.sqrt(np.maximum(g22 - f12 * f12, 1.0e-12))
        result = np.zeros((len(self.x), 2, 2), dtype=np.float64)
        result[:, 0, 0] = f11
        result[:, 0, 1] = f12
        result[:, 1, 1] = f22
        return result

    def _contact_distances(self):
        radius = 0.5 * self.thickness
        distances = [self._box_sdf(self.x, center, half, rotation) - radius
                     for center, half, rotation, _ in self.boxes]
        return np.minimum.reduce(distances)

    def _frame(self, time):
        frame = StateFrame(
            float(time), self.x.copy(), self.v.copy(), mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_distances(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")
        max_dt = 1.0 / 300.0
        while self.time < target - 1.0e-12:
            dt = min(max_dt, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_knit_sheet",
            "substeps": int(self.steps),
            "simulated_time": float(self.time),
            "particle_count": int(len(self.x)),
        }
