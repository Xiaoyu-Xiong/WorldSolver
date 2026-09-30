"""Deterministic particle solver for the breaking-wave scene."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Compact weakly-compressible particle solver with rigid contacts."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.position)
        self.time = 0.0
        config = scene.config
        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.spacing = float(config["sampling"]["spacing"])
        self.grid_shape = tuple(int(v) for v in config["sampling"]["field_grid_shape"])
        self.radius = 0.43 * self.spacing
        self.rest_distance = 0.94 * self.spacing
        self.interaction_radius = 1.42 * self.spacing
        self.max_step = 0.0045
        self.pressure_stiffness = 6200.0
        self.pair_damping = 2.5
        self.kinematic_viscosity = 1.8
        self.surface_cohesion = 7.0
        self.linear_drag = 0.025
        self.restitution = 0.035
        self.gate_speed = 2.8
        self.gate_open_time = 0.40
        self.step_count = 0
        self.max_observed_speed = 0.0
        self.velocity[:, 0] = 0.06
        self._make_diagnostic_grid()
        self._make_boundary_samples()

    def _make_diagnostic_grid(self) -> None:
        nx, ny, nz = self.grid_shape
        dx = self.spacing
        x0 = -1.35
        y0 = -0.5 * (ny - 1) * dx
        z0 = -0.5 * dx
        x = x0 + dx * np.arange(nx)
        y = y0 + dx * np.arange(ny)
        z = z0 + dx * np.arange(nz)
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        self.grid_points = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        self.grid_solid = (
            (self.grid_points[:, 0] >= 0.25)
            & (self.grid_points[:, 0] <= 2.65)
            & (np.abs(self.grid_points[:, 1]) <= 0.84)
            & (self.grid_points[:, 2] <= 0.60)
        ) | (self.grid_points[:, 2] < 0.0)

    def _make_boundary_samples(self) -> None:
        points, normals, kinds = [], [], []

        def add(point, normal, kind="static"):
            points.append(point)
            normals.append(normal)
            kinds.append(kind)

        for x in np.linspace(-3.0, 3.0, 17):
            add((x, 0.0, 0.0), (0.0, 0.0, 1.0))
        for x in np.linspace(-2.9, 2.9, 13):
            add((x, -0.92, 0.45), (0.0, 1.0, 0.0))
            add((x, 0.92, 0.45), (0.0, -1.0, 0.0))
        for z in np.linspace(0.08, 1.02, 8):
            add((3.06, 0.0, z), (-1.0, 0.0, 0.0))
        for x in np.linspace(0.32, 2.58, 12):
            add((x, 0.0, 0.60), (0.0, 0.0, 1.0))
        for z in np.linspace(0.07, 0.55, 7):
            add((0.25, 0.0, z), (-1.0, 0.0, 0.0))
        for z in np.linspace(0.08, 1.02, 8):
            add((-0.49, 0.0, z), (-1.0, 0.0, 0.0), "gate")
        self.boundary_points = np.asarray(points, dtype=np.float64)
        self.boundary_normal = np.asarray(normals, dtype=np.float64)
        self.gate_samples = np.asarray([k == "gate" for k in kinds])

    def _particle_acceleration(self) -> np.ndarray:
        acceleration = np.broadcast_to(self.gravity, self.position.shape).copy()
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(self.interaction_radius, output_type="ndarray")
        if len(pairs) == 0:
            acceleration -= self.linear_drag * self.velocity
            return acceleration
        i, j = pairs[:, 0], pairs[:, 1]
        delta = self.position[i] - self.position[j]
        distance = np.linalg.norm(delta, axis=1)
        valid = distance > 1.0e-9
        i, j, delta, distance = i[valid], j[valid], delta[valid], distance[valid]
        direction = delta / distance[:, None]
        overlap = self.rest_distance - distance
        pressure = self.pressure_stiffness * np.maximum(overlap, 0.0)
        cohesion = -self.surface_cohesion * np.maximum(-overlap, 0.0)
        weight = np.maximum(0.0, 1.0 - distance / self.interaction_radius)
        relative_velocity = self.velocity[i] - self.velocity[j]
        normal_speed = np.einsum("ij,ij->i", relative_velocity, direction)
        magnitude = pressure + cohesion - self.pair_damping * weight * normal_speed
        pair_acceleration = magnitude[:, None] * direction
        np.add.at(acceleration, i, pair_acceleration)
        np.add.at(acceleration, j, -pair_acceleration)
        viscous = (self.kinematic_viscosity * weight)[:, None] * (self.velocity[j] - self.velocity[i])
        np.add.at(acceleration, i, viscous)
        np.add.at(acceleration, j, -viscous)
        acceleration -= self.linear_drag * self.velocity
        return acceleration

    @staticmethod
    def _reflect_component(values, velocity, axis, wall, lower, restitution):
        hit = values < wall if lower else values > wall
        if not np.any(hit):
            return
        values[hit] = wall
        inward = velocity[hit, axis] < 0.0 if lower else velocity[hit, axis] > 0.0
        indices = np.flatnonzero(hit)[inward]
        velocity[indices, axis] *= -restitution

    def _collide(self, previous: np.ndarray, new_time: float) -> None:
        p, v, r = self.position, self.velocity, self.radius
        self._reflect_component(p[:, 0], v, 0, -3.14 + r, True, self.restitution)
        self._reflect_component(p[:, 0], v, 0, 3.06 - r, False, self.restitution)
        self._reflect_component(p[:, 1], v, 1, -0.92 + r, True, self.restitution)
        self._reflect_component(p[:, 1], v, 1, 0.92 - r, False, self.restitution)
        self._reflect_component(p[:, 2], v, 2, r, True, self.restitution)
        lo = np.array([0.25 - r, -0.84 - r, -r])
        hi = np.array([2.65 + r, 0.84 + r, 0.60 + r])
        inside = np.all((p > lo) & (p < hi), axis=1)
        for idx in np.flatnonzero(inside):
            if previous[idx, 0] <= lo[0]:
                axis, wall, sign = 0, lo[0], -1.0
            elif previous[idx, 0] >= hi[0]:
                axis, wall, sign = 0, hi[0], 1.0
            elif previous[idx, 2] >= hi[2]:
                axis, wall, sign = 2, hi[2], 1.0
            else:
                distances = np.array([p[idx, 0] - lo[0], hi[0] - p[idx, 0], hi[2] - p[idx, 2]])
                face = int(np.argmin(distances))
                axis, wall, sign = ((0, lo[0], -1.0), (0, hi[0], 1.0), (2, hi[2], 1.0))[face]
            p[idx, axis] = wall
            if v[idx, axis] * sign < 0.0:
                v[idx, axis] *= -self.restitution
        if new_time < self.gate_open_time:
            gate_bottom = self.gate_speed * new_time
            front = -0.49 - r
            blocked = ((previous[:, 0] <= front) & (p[:, 0] > front) &
                       (p[:, 2] + r >= gate_bottom) & (np.abs(p[:, 1]) <= 0.72 + r))
            p[blocked, 0] = front
            impact = blocked & (v[:, 0] > 0.0)
            v[impact, 0] *= -self.restitution
            v[blocked, 2] = np.maximum(v[blocked, 2], self.gate_speed * 0.08)

    def _step(self, dt: float) -> None:
        previous = self.position.copy()
        self.velocity += dt * self._particle_acceleration()
        speed = np.linalg.norm(self.velocity, axis=1)
        too_fast = speed > 12.0
        if np.any(too_fast):
            self.velocity[too_fast] *= (12.0 / speed[too_fast])[:, None]
        self.position += dt * self.velocity
        self._collide(previous, self.time + dt)
        self.time += dt
        self.step_count += 1
        self.max_observed_speed = max(self.max_observed_speed, float(np.max(speed)))

    def _sample_fields(self):
        tree = cKDTree(self.position)
        distance, indices = tree.query(self.grid_points, k=min(8, self.n))
        if distance.ndim == 1:
            distance, indices = distance[:, None], indices[:, None]
        weights = np.exp(-0.5 * (distance / (0.9 * self.spacing)) ** 2)
        weight_sum = np.sum(weights, axis=1)
        velocity = np.sum(weights[:, :, None] * self.velocity[indices], axis=1)
        velocity /= np.maximum(weight_sum[:, None], 1.0e-12)
        far = distance[:, 0] > 2.2 * self.spacing
        velocity[far] = 0.0
        level_set = distance[:, 0] - 0.88 * self.spacing
        level_set[self.grid_solid] = np.maximum(level_set[self.grid_solid], 0.5 * self.spacing)
        active = (level_set <= 0.0) & ~self.grid_solid
        shape = self.grid_shape
        return active.reshape(shape), level_set.reshape(shape), velocity.reshape(shape + (3,))

    def _frame(self, requested_time: float) -> StateFrame:
        active, level_set, velocity_field = self._sample_fields()
        boundary_target = np.zeros_like(self.boundary_normal)
        if requested_time < self.gate_open_time:
            boundary_target[self.gate_samples, 2] = self.gate_speed
        frame = StateFrame(
            time=float(requested_time), position=self.position.copy(), velocity=self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass.copy(), "particle_radius": np.full(self.n, self.radius),
                        "material_label": self.body_id.copy()},
            fields={"active_mask": active, "boundary_normal": self.boundary_normal.copy(),
                    "boundary_target_velocity": boundary_target, "boundary_velocity": boundary_target.copy(),
                    "cumulative_mass_source": np.zeros(1), "liquid_level_set": level_set,
                    "velocity_field": velocity_field},
            scalars={"characteristic_length": 2.25, "grid_spacing": self.spacing,
                     "reference_velocity": float(np.sqrt(9.81 * 1.05)), "contact_length_scale": self.spacing})
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time + 1.0e-12 < target:
            self._step(min(self.max_step, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"method": "short-range particle hydrodynamics", "particle_count": int(self.n),
                "internal_steps": int(self.step_count), "maximum_speed": float(self.max_observed_speed),
                "mass_initial": float(np.sum(self.mass)), "mass_final": float(np.sum(self.mass))}
