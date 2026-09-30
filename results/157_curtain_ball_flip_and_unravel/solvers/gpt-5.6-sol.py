"""Deterministic thin-cloth solver for the curtain/ball scene.

The sheet is discretized as lumped masses joined by stretch, shear, and
bending constraints. Symplectic prediction followed by iterative position
projection provides a robust time integrator for the very light cloth. The
ball follows rigid-body projectile dynamics and acts as a one-way kinematic
contact boundary, as specified by the frozen scene geometry.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based cloth with sphere, ground, and self contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The discretization is intentionally deterministic.
        self.scene = scene
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.thickness = float(cloth["thickness"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.mass = np.full(
            self.n, float(cloth["total_mass"]) / self.n, dtype=np.float64
        )
        self.inv_mass = 1.0 / self.mass

        sx, sz = map(float, cloth["size"])
        center = np.asarray(cloth["center"], dtype=np.float64)
        gx = np.linspace(-0.5 * sx, 0.5 * sx, self.nx)
        gz = np.linspace(-0.5 * sz, 0.5 * sz, self.ny)
        xx, zz = np.meshgrid(gx, gz, indexing="xy")
        self.rest = np.column_stack(
            (xx.ravel(), np.zeros(self.n), zz.ravel())
        ) + center
        self.dx = sx / (self.nx - 1)
        self.dz = sz / (self.ny - 1)

        # Resolve the frozen plane/ball overlap onto the camera-facing
        # hemisphere, yielding the requested initially wrapped connected sheet.
        self.ball_radius = 0.62
        self.ball_center = np.array([0.0, 0.0, 0.68], dtype=np.float64)
        self.ball_velocity = np.array([0.0, -1.55, 2.35], dtype=np.float64)
        self.x = self.rest.copy()
        contact_radius = self.ball_radius + 0.5 * self.thickness
        radial2 = (
            (self.x[:, (0, 2)] - self.ball_center[(0, 2)]) ** 2
        ).sum(axis=1)
        wrapped = radial2 < contact_radius * contact_radius
        self.x[wrapped, 1] = (
            self.ball_center[1]
            - np.sqrt(
                np.maximum(
                    contact_radius * contact_radius - radial2[wrapped], 0.0
                )
            )
        )
        self.v = np.zeros_like(self.x)

        # Pin the complete upper edge to the curtain rod.
        self.pinned = np.arange(
            (self.ny - 1) * self.nx, self.n, dtype=np.int64
        )
        self.pin_position = self.rest[self.pinned].copy()
        self.inv_mass[self.pinned] = 0.0

        self.struct_i, self.struct_j = self._grid_pairs(((1, 0), (0, 1)))
        self.shear_i, self.shear_j = self._grid_pairs(((1, 1), (-1, 1)))
        self.bend_i, self.bend_j = self._grid_pairs(((2, 0), (0, 2)))
        self.struct_l = self._rest_lengths(self.struct_i, self.struct_j)
        self.shear_l = self._rest_lengths(self.shear_i, self.shear_j)
        self.bend_l = self._rest_lengths(self.bend_i, self.bend_j)

        # Exclude vertices sharing a local mesh neighborhood from self-contact.
        ids = np.arange(self.n)
        col = ids % self.nx
        row = ids // self.nx
        self.self_excluded = (
            (np.abs(row[:, None] - row[None, :]) <= 1)
            & (np.abs(col[:, None] - col[None, :]) <= 1)
        )
        np.fill_diagonal(self.self_excluded, True)

        self.time = 0.0
        self.steps = 0
        self.max_penetration = 0.0
        self.substep = 1.0 / 240.0

    def _grid_pairs(self, offsets):
        first, second = [], []
        for dj, di in offsets:
            for row in range(self.ny):
                for col in range(self.nx):
                    other_row, other_col = row + di, col + dj
                    if (
                        0 <= other_row < self.ny
                        and 0 <= other_col < self.nx
                    ):
                        first.append(row * self.nx + col)
                        second.append(other_row * self.nx + other_col)
        return (
            np.asarray(first, dtype=np.int64),
            np.asarray(second, dtype=np.int64),
        )

    def _rest_lengths(self, i, j):
        return np.linalg.norm(self.rest[j] - self.rest[i], axis=1)

    def _project_springs(self, i, j, rest_length, stiffness):
        delta = self.x[j] - self.x[i]
        length = np.linalg.norm(delta, axis=1)
        weight = self.inv_mass[i] + self.inv_mass[j]
        valid = (length > 1.0e-12) & (weight > 1.0e-12)
        scale = np.zeros_like(length)
        scale[valid] = (
            stiffness
            * (length[valid] - rest_length[valid])
            / (length[valid] * weight[valid])
        )
        correction = delta * scale[:, None]
        accumulated = np.zeros_like(self.x)
        np.add.at(
            accumulated, i, correction * self.inv_mass[i, None]
        )
        np.add.at(
            accumulated, j, -correction * self.inv_mass[j, None]
        )
        self.x += accumulated

    def _sphere_contact(self, old_x, ball_old):
        offset = self.x - self.ball_center
        distance = np.linalg.norm(offset, axis=1)
        radius = self.ball_radius + 0.5 * self.thickness
        hit = (distance < radius) & (self.inv_mass > 0.0)
        if not np.any(hit):
            return
        normal = offset[hit] / np.maximum(
            distance[hit, None], 1.0e-12
        )
        centered = distance[hit] <= 1.0e-12
        normal[centered] = np.array([0.0, -1.0, 0.0])
        depth = radius - distance[hit]
        self.max_penetration = max(
            self.max_penetration, float(np.max(depth))
        )
        self.x[hit] += depth[:, None] * normal

        # Coulomb-like positional friction relative to the moving sphere.
        cloth_move = self.x[hit] - old_x[hit]
        ball_move = self.ball_center - ball_old
        relative = cloth_move - ball_move
        tangent = (
            relative
            - (relative * normal).sum(axis=1)[:, None] * normal
        )
        tangent_norm = np.linalg.norm(tangent, axis=1)
        limit = 0.32 * depth
        fraction = np.minimum(
            1.0, limit / np.maximum(tangent_norm, 1.0e-12)
        )
        self.x[hit] -= tangent * fraction[:, None]

    def _ground_contact(self, old_x):
        floor = 0.5 * self.thickness
        hit = (self.x[:, 2] < floor) & (self.inv_mass > 0.0)
        if not np.any(hit):
            return
        depth = floor - self.x[hit, 2]
        self.max_penetration = max(
            self.max_penetration, float(np.max(depth))
        )
        self.x[hit, 2] = floor
        slide = self.x[hit, :2] - old_x[hit, :2]
        slide_length = np.linalg.norm(slide, axis=1)
        friction = np.minimum(
            1.0,
            0.45 * depth / np.maximum(slide_length, 1.0e-12),
        )
        self.x[hit, :2] -= slide * friction[:, None]

    def _self_contact(self):
        # Simultaneous mass-weighted repulsion separates multilayer folds.
        delta = self.x[:, None, :] - self.x[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        separation = 1.65 * self.thickness
        candidates = np.triu(
            (~self.self_excluded)
            & (distance2 < separation * separation),
            k=1,
        )
        a, b = np.nonzero(candidates)
        if len(a) == 0:
            return
        direction = self.x[a] - self.x[b]
        distance = np.linalg.norm(direction, axis=1)
        valid = distance > 1.0e-10
        direction[valid] /= distance[valid, None]
        if np.any(~valid):
            direction[~valid] = np.array([0.0, 1.0, 0.0])
        weight = self.inv_mass[a] + self.inv_mass[b]
        depth = separation - distance
        correction = direction * (depth / weight)[:, None]
        accumulated = np.zeros_like(self.x)
        np.add.at(
            accumulated, a, correction * self.inv_mass[a, None]
        )
        np.add.at(
            accumulated, b, -correction * self.inv_mass[b, None]
        )
        self.x += accumulated

    def _advance_ball(self, h):
        old = self.ball_center.copy()
        self.ball_velocity += self.gravity * h
        self.ball_center += self.ball_velocity * h
        minimum_z = self.ball_radius + 0.5 * self.thickness
        if self.ball_center[2] < minimum_z:
            self.ball_center[2] = minimum_z
            if self.ball_velocity[2] < 0.0:
                self.ball_velocity[2] *= -0.18
            self.ball_velocity[:2] *= np.exp(-1.0 * h)
            if abs(self.ball_velocity[2]) < 0.06:
                self.ball_velocity[2] = 0.0
        return old

    def _step(self, h):
        old_x = self.x.copy()
        ball_old = self._advance_ball(h)

        # Symplectic Euler prediction with mild air drag.
        self.v += self.gravity[None, :] * h
        self.v *= np.exp(-0.22 * h)
        self.x += self.v * h

        for _ in range(6):
            self._project_springs(
                self.struct_i, self.struct_j, self.struct_l, 0.72
            )
            self._project_springs(
                self.shear_i, self.shear_j, self.shear_l, 0.50
            )
            self._project_springs(
                self.bend_i, self.bend_j, self.bend_l, 0.055
            )
            self._sphere_contact(old_x, ball_old)
            self._ground_contact(old_x)
            self.x[self.pinned] = self.pin_position

        self._self_contact()
        self._sphere_contact(old_x, ball_old)
        self._ground_contact(old_x)
        self.x[self.pinned] = self.pin_position
        self.v = (self.x - old_x) / h
        self.v[self.pinned] = 0.0
        np.clip(self.v, -35.0, 35.0, out=self.v)
        self.steps += 1

    def _deformation_gradient(self):
        p = self.x.reshape(self.ny, self.nx, 3)
        du = np.empty_like(p)
        dv = np.empty_like(p)
        du[:, 1:-1] = (
            p[:, 2:] - p[:, :-2]
        ) / (2.0 * self.dx)
        du[:, 0] = (p[:, 1] - p[:, 0]) / self.dx
        du[:, -1] = (p[:, -1] - p[:, -2]) / self.dx
        dv[1:-1] = (
            p[2:] - p[:-2]
        ) / (2.0 * self.dz)
        dv[0] = (p[1] - p[0]) / self.dz
        dv[-1] = (p[-1] - p[-2]) / self.dz
        jacobian = np.stack(
            (du.reshape(self.n, 3), dv.reshape(self.n, 3)),
            axis=2,
        )
        metric = np.einsum(
            "nki,nkj->nij", jacobian, jacobian
        )
        eigenvalue, eigenvector = np.linalg.eigh(metric)
        stretch = np.sqrt(
            np.clip(eigenvalue, 1.0e-12, 1.0e6)
        )
        return np.einsum(
            "nik,nk,njk->nij", eigenvector, stretch, eigenvector
        )

    def _signed_contact_distance(self):
        sphere_gap = (
            np.linalg.norm(self.x - self.ball_center, axis=1)
            - self.ball_radius
            - 0.5 * self.thickness
        )
        ground_gap = self.x[:, 2] - 0.5 * self.thickness
        delta = self.x[:, None, :] - self.x[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        distance[self.self_excluded] = np.inf
        self_gap = (
            np.min(distance, axis=1) - 1.65 * self.thickness
        )
        return np.minimum(
            np.minimum(sphere_gap, ground_gap), self_gap
        )

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            h = min(self.substep, target - self.time)
            self._step(h)
            self.time += h
        self.time = target

        external_force = (
            self.mass[:, None] * self.gravity[None, :]
        )
        boundary = np.zeros(self.n, dtype=np.int32)
        boundary[self.pinned] = 1
        frame = StateFrame(
            target,
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": (
                    self._deformation_gradient()
                ),
            },
            fields={
                "contact_signed_distance": (
                    self._signed_contact_distance()
                ),
                "cumulative_mass_source": np.zeros(
                    1, dtype=np.float64
                ),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self.thickness
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "position_based_dynamics",
            "substeps": self.steps,
            "simulated_time": self.time,
            "maximum_resolved_penetration": (
                self.max_penetration
            ),
            "ball_center": self.ball_center.tolist(),
        }
