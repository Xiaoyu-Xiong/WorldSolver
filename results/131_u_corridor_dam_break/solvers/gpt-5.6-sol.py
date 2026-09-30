"""Deterministic particle solver for the U-corridor dam-break scene.

The liquid is advanced with a position-based incompressibility projection. All
motion comes from time integration of gravity, particle density constraints,
viscosity, and rigid boundary contact.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small position-based fluid solver with collocated grid diagnostics."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.config = scene.config
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.time = 0.0

        self.spacing = float(scene.config["sampling"]["spacing"])
        self.particle_radius = 0.42 * self.spacing
        self.kernel_radius = 2.0 * self.spacing
        self.gravity = np.asarray(scene.config["gravity"], dtype=np.float64)
        self.base_dt = 1.0 / 180.0
        self.projection_iterations = 4
        self.viscosity = 0.055

        # Inner faces of the visible floor and U-shaped walls.
        radius = self.particle_radius
        self.x_min, self.x_max = -1.05 + radius, 1.05 - radius
        self.y_min, self.y_max = -1.62 + radius, 1.45 - radius
        self.z_min = radius
        self.gate_y = -0.64 - radius
        self.gate_speed = 6.0
        self.gate_travel = 1.32

        self.identity = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], len(self.position), axis=0
        )
        self.characteristic_length = float(
            initial.scalars.get(
                "characteristic_length",
                max(np.ptp(self.position, axis=0).max(), self.spacing),
            )
        )
        self.reference_velocity = float(
            np.sqrt(np.linalg.norm(self.gravity) * self.characteristic_length)
        )

        pairs, _, weight = self._pairs(self.position)
        number_density = np.ones(len(self.position), dtype=np.float64)
        if len(pairs):
            np.add.at(number_density, pairs[:, 0], weight)
            np.add.at(number_density, pairs[:, 1], weight)
        self.rest_number_density = max(
            float(np.percentile(number_density, 95)), 1.0
        )

        self._build_diagnostic_grid()
        self._build_boundary_samples()
        self._last_max_speed = 0.0
        self._substeps = 0

    def _pairs(self, position):
        pairs = cKDTree(position).query_pairs(
            self.kernel_radius, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.empty((0, 2), np.intp), np.empty(0), np.empty(0)
        pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        delta = position[pairs[:, 0]] - position[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        q = np.minimum(distance / self.kernel_radius, 1.0)
        weight = (1.0 - q) ** 3
        return pairs, distance, weight

    def _project_density(self, position) -> None:
        """Apply one nonlinear incompressibility projection in-place."""
        pairs, distance, weight = self._pairs(position)
        if len(pairs) == 0:
            return
        count = len(position)
        density = np.ones(count, dtype=np.float64)
        np.add.at(density, pairs[:, 0], weight)
        np.add.at(density, pairs[:, 1], weight)
        constraint = np.maximum(
            density / self.rest_number_density - 1.0, 0.0
        )
        if float(np.max(constraint)) < 1.0e-5:
            return

        delta = position[pairs[:, 0]] - position[pairs[:, 1]]
        inv_distance = 1.0 / np.maximum(distance, 1.0e-9)
        dweight = -3.0 * (1.0 - distance / self.kernel_radius) ** 2
        dweight /= self.kernel_radius
        gradient = (
            dweight[:, None]
            * delta
            * inv_distance[:, None]
            / self.rest_number_density
        )

        gradient_sum = np.zeros_like(position)
        np.add.at(gradient_sum, pairs[:, 0], gradient)
        np.add.at(gradient_sum, pairs[:, 1], -gradient)
        denominator = np.einsum("ij,ij->i", gradient_sum, gradient_sum)
        pair_gradient_sq = np.einsum("ij,ij->i", gradient, gradient)
        np.add.at(denominator, pairs[:, 0], pair_gradient_sq)
        np.add.at(denominator, pairs[:, 1], pair_gradient_sq)
        lagrange = -constraint / (denominator + 1.0e-4)

        pair_correction = (
            (lagrange[pairs[:, 0]] + lagrange[pairs[:, 1]])[:, None]
            * gradient
        )
        correction = np.zeros_like(position)
        np.add.at(correction, pairs[:, 0], pair_correction)
        np.add.at(correction, pairs[:, 1], -pair_correction)
        length = np.linalg.norm(correction, axis=1)
        limiter = np.minimum(
            1.0, (0.24 * self.spacing) / np.maximum(length, 1.0e-12)
        )
        position += correction * limiter[:, None]

    def _constrain_boundaries(self, position, next_time: float) -> None:
        position[:, 0] = np.clip(position[:, 0], self.x_min, self.x_max)
        position[:, 1] = np.clip(position[:, 1], self.y_min, self.y_max)
        position[:, 2] = np.maximum(position[:, 2], self.z_min)

        # The kinematic gate rises vertically, progressively opening underneath.
        gate_bottom = min(self.gate_speed * next_time, self.gate_travel)
        gate_top = gate_bottom + 1.10
        near_gate = (position[:, 1] > self.gate_y) & (
            position[:, 1] < self.gate_y + 0.10
        )
        blocked_height = (
            position[:, 2] + self.particle_radius > gate_bottom
        ) & (position[:, 2] - self.particle_radius < gate_top)
        position[near_gate & blocked_height, 1] = self.gate_y

    def _step(self, dt: float) -> None:
        old_position = self.position.copy()
        self.velocity += self.gravity * dt
        predicted = self.position + self.velocity * dt
        next_time = self.time + dt

        for _ in range(self.projection_iterations):
            self._constrain_boundaries(predicted, next_time)
            self._project_density(predicted)
        self._constrain_boundaries(predicted, next_time)

        velocity = (predicted - old_position) / dt
        pairs, _, weight = self._pairs(predicted)
        if len(pairs):
            smooth = np.zeros_like(velocity)
            normalization = np.zeros(len(velocity), dtype=np.float64)
            difference = velocity[pairs[:, 1]] - velocity[pairs[:, 0]]
            np.add.at(smooth, pairs[:, 0], weight[:, None] * difference)
            np.add.at(smooth, pairs[:, 1], -weight[:, None] * difference)
            np.add.at(normalization, pairs[:, 0], weight)
            np.add.at(normalization, pairs[:, 1], weight)
            velocity += (
                self.viscosity
                * smooth
                / np.maximum(normalization[:, None], 1.0)
            )

        # Contact impulses remove outward normal velocity with slight rebound.
        eps = 1.0e-8
        masks = (
            (predicted[:, 0] <= self.x_min + eps) & (velocity[:, 0] < 0.0),
            (predicted[:, 0] >= self.x_max - eps) & (velocity[:, 0] > 0.0),
            (predicted[:, 1] <= self.y_min + eps) & (velocity[:, 1] < 0.0),
            (predicted[:, 1] >= self.y_max - eps) & (velocity[:, 1] > 0.0),
            (predicted[:, 2] <= self.z_min + eps) & (velocity[:, 2] < 0.0),
        )
        for mask, component in zip(masks, (0, 0, 1, 1, 2)):
            velocity[mask, component] *= -0.04

        speed = np.linalg.norm(velocity, axis=1)
        velocity *= np.minimum(
            1.0, 8.0 / np.maximum(speed, 1.0e-12)
        )[:, None]
        velocity *= 0.9995
        if not (
            np.isfinite(predicted).all() and np.isfinite(velocity).all()
        ):
            predicted = old_position
            velocity.fill(0.0)
        self.position = predicted
        self.velocity = velocity
        self.time = next_time
        self._substeps += 1
        self._last_max_speed = float(
            np.max(np.linalg.norm(velocity, axis=1))
        )

    def _build_diagnostic_grid(self) -> None:
        shape = tuple(
            int(value)
            for value in self.config["sampling"]["field_grid_shape"]
        )
        x = np.linspace(-1.68, 1.68, shape[0])
        y = np.linspace(-1.68, 1.68, shape[1])
        z = np.linspace(0.0, 1.75, shape[2])
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        self.grid_shape = shape
        self.grid_points = np.column_stack(
            (gx.ravel(), gy.ravel(), gz.ravel())
        )
        self.active_mask = (
            (np.abs(gx) <= 1.05)
            & (gy >= -1.62)
            & (gy <= 1.45)
            & (gz >= 0.0)
        )

    def _build_boundary_samples(self) -> None:
        points, normals, gate_flags = [], [], []

        def add(surface_points, normal, is_gate=False):
            points.extend(surface_points)
            normals.extend([normal] * len(surface_points))
            gate_flags.extend([is_gate] * len(surface_points))

        xs = np.linspace(-1.0, 1.0, 11)
        ys = np.linspace(-1.55, 1.42, 16)
        zs = np.linspace(0.08, 1.08, 6)
        add(
            [(x, y, 0.0) for x in xs for y in ys],
            (0.0, 0.0, 1.0),
        )
        add(
            [(-1.05, y, z) for y in ys for z in zs],
            (1.0, 0.0, 0.0),
        )
        add(
            [(1.05, y, z) for y in ys for z in zs],
            (-1.0, 0.0, 0.0),
        )
        add(
            [(x, 1.45, z) for x in xs for z in zs],
            (0.0, -1.0, 0.0),
        )
        add(
            [(x, -0.64, z) for x in xs for z in zs],
            (0.0, -1.0, 0.0),
            True,
        )
        self.boundary_points_base = np.asarray(points, dtype=np.float64)
        self.boundary_normals = np.asarray(normals, dtype=np.float32)
        self.boundary_gate = np.asarray(gate_flags, dtype=bool)

    def _interpolate_velocity(self, query_points, radius):
        tree = cKDTree(self.position)
        distance, index = tree.query(
            query_points, k=4, distance_upper_bound=radius
        )
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        valid = np.isfinite(distance) & (index < len(self.position))
        safe_index = np.minimum(index, len(self.position) - 1)
        weight = np.where(
            valid,
            (1.0 - np.minimum(distance / radius, 1.0)) ** 2,
            0.0,
        )
        total = np.sum(weight, axis=1)
        sampled = np.sum(
            weight[:, :, None] * self.velocity[safe_index], axis=1
        )
        sampled /= np.maximum(total[:, None], 1.0e-12)
        sampled[total == 0.0] = 0.0
        return sampled

    def _fields(self):
        tree = cKDTree(self.position)
        nearest, _ = tree.query(self.grid_points, k=1)
        level_set = (
            nearest - 1.15 * self.spacing
        ).reshape(self.grid_shape)
        grid_velocity = self._interpolate_velocity(
            self.grid_points, 2.5 * self.spacing
        ).reshape(self.grid_shape + (3,))
        grid_velocity[~self.active_mask] = 0.0

        boundary_points = self.boundary_points_base.copy()
        gate_offset = min(self.gate_speed * self.time, self.gate_travel)
        boundary_points[self.boundary_gate, 2] += gate_offset
        target = np.zeros_like(boundary_points)
        if self.gate_speed * self.time < self.gate_travel:
            target[self.boundary_gate, 2] = self.gate_speed
        boundary_velocity = self._interpolate_velocity(
            boundary_points, 2.2 * self.spacing
        )
        normal_error = np.sum(
            (boundary_velocity - target) * self.boundary_normals, axis=1
        )
        boundary_velocity -= normal_error[:, None] * self.boundary_normals
        boundary_velocity[self.boundary_gate] = target[self.boundary_gate]

        return {
            "active_mask": self.active_mask.copy(),
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_target_velocity": target.astype(np.float32),
            "boundary_velocity": boundary_velocity.astype(np.float32),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "liquid_level_set": level_set.astype(np.float32),
            "velocity_field": grid_velocity.astype(np.float32),
        }

    def advance_to(self, time: float):
        target_time = float(time)
        if target_time < self.time - 1.0e-9:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target_time - 1.0e-10:
            dt = min(self.base_dt, target_time - self.time)
            self._step(dt)
        self.time = target_time

        frame = StateFrame(
            target_time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(
                    len(self.position),
                    self.particle_radius,
                    dtype=np.float64,
                ),
                "material_label": self.body_id.copy(),
                "deformation_gradient": self.identity.copy(),
            },
            fields=self._fields(),
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_incompressible_particles",
            "substeps": int(self._substeps),
            "particle_count": int(len(self.position)),
            "max_speed": float(self._last_max_speed),
            "simulated_time": float(self.time),
        }
