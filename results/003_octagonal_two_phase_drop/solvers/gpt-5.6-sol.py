"""Deterministic two-phase particle-in-cell fluid solver.

The simulation uses the scene particles as volume-carrying material points. At
every substep their mass, volume and momentum are transferred to a Cartesian
grid, a weakly-compressible pressure force and gravity are integrated there,
and velocities are transferred back with a FLIP/PIC blend. Because pressure
is computed from volume while acceleration uses physical mixture density, the
four-to-one phase density ratio produces buoyant separation without scripted
motion.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A compact, deterministic weakly-compressible free-surface solver."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        config = scene.config

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.phase_id = np.asarray(initial.attributes["phase_id"], dtype=np.int32).copy()
        self.density = np.asarray(initial.attributes["density"], dtype=np.float64).copy()
        self.release_time = np.asarray(
            initial.attributes["release_time"], dtype=np.float64
        ).copy()
        self.volume = np.asarray(
            initial.attributes["target_volume"], dtype=np.float64
        ).copy()
        self.rest_position = self.position.copy()

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.time = 0.0
        self.step_count = 0
        self.max_substep = 0.004

        # A 10 cm grid resolves the 6.5 cm material-point lattice while keeping
        # the full falling columns and diagnostic fields inexpensive.
        self.dx = 0.10
        self.origin = np.array([-1.2, -1.2, 0.0], dtype=np.float64)
        self.grid_shape = (25, 25, 45)
        self.cell_volume = self.dx ** 3
        self.bulk_modulus = 2.4e4
        self.rest_fill = 0.90
        self.max_pressure = 1.2e5
        self.flip_fraction = 0.92

        # Inner collision surface of the OBJ, inset by one particle radius.
        self.floor_z = 0.115
        self.rim_z = 0.62
        self.axis_limit = 1.025
        self.diagonal_limit = 1.45
        self.restitution = 0.04
        self.wall_friction = 0.10

        n = len(self.position)
        self.mass_quantity = np.zeros((n, 2), dtype=np.float64)
        self.mass_quantity[np.arange(n), self.phase_id] = self.mass
        self.cumulative_mass_source = np.zeros(2, dtype=np.float64)
        self.characteristic_length = 2.12132

    @staticmethod
    def _shift_zero(array: np.ndarray, axis: int, amount: int) -> np.ndarray:
        """Shift by one cell, filling the introduced edge with zero."""
        result = np.zeros_like(array)
        source = [slice(None)] * array.ndim
        target = [slice(None)] * array.ndim
        if amount > 0:
            source[axis] = slice(None, -amount)
            target[axis] = slice(amount, None)
        else:
            source[axis] = slice(-amount, None)
            target[axis] = slice(None, amount)
        result[tuple(target)] = array[tuple(source)]
        return result

    def _particle_stencil(self, particle_indices: np.ndarray):
        """Yield grid indices and trilinear weights for active particles."""
        p = self.position[particle_indices]
        grid_p = (p - self.origin) / self.dx
        base = np.floor(grid_p).astype(np.int32)
        upper = np.asarray(self.grid_shape, dtype=np.int32) - 2
        base = np.clip(base, 0, upper)
        fraction = np.clip(grid_p - base, 0.0, 1.0)

        for ox in (0, 1):
            wx = fraction[:, 0] if ox else 1.0 - fraction[:, 0]
            for oy in (0, 1):
                wy = fraction[:, 1] if oy else 1.0 - fraction[:, 1]
                for oz in (0, 1):
                    wz = fraction[:, 2] if oz else 1.0 - fraction[:, 2]
                    yield (
                        base[:, 0] + ox,
                        base[:, 1] + oy,
                        base[:, 2] + oz,
                        wx * wy * wz,
                    )

    def _deposit(self, active: np.ndarray):
        grid_mass = np.zeros(self.grid_shape, dtype=np.float64)
        grid_volume = np.zeros(self.grid_shape, dtype=np.float64)
        momentum = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            return grid_mass, grid_volume, momentum, indices

        for ix, iy, iz, weight in self._particle_stencil(indices):
            key = (ix, iy, iz)
            weighted_mass = weight * self.mass[indices]
            np.add.at(grid_mass, key, weighted_mass)
            np.add.at(grid_volume, key, weight * self.volume[indices])
            for component in range(3):
                np.add.at(
                    momentum[..., component],
                    key,
                    weighted_mass * self.velocity[indices, component],
                )
        return grid_mass, grid_volume, momentum, indices

    def _interpolate(self, field: np.ndarray, indices: np.ndarray) -> np.ndarray:
        result = np.zeros((len(indices), field.shape[-1]), dtype=np.float64)
        weight_sum = np.zeros(len(indices), dtype=np.float64)
        for ix, iy, iz, weight in self._particle_stencil(indices):
            result += weight[:, None] * field[ix, iy, iz]
            weight_sum += weight
        result /= np.maximum(weight_sum[:, None], 1.0e-12)
        return result

    def _smoothed_fill(self, grid_volume: np.ndarray) -> np.ndarray:
        fill = grid_volume / self.cell_volume
        total = 2.0 * fill
        for axis in range(3):
            total += self._shift_zero(fill, axis, 1)
            total += self._shift_zero(fill, axis, -1)
        return total / 8.0

    def _collide_basin(self, old_position: np.ndarray, active: np.ndarray) -> None:
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            return
        p = self.position[indices]
        v = self.velocity[indices]

        below = p[:, 2] < self.floor_z
        if np.any(below):
            p[below, 2] = self.floor_z
            downward = below & (v[:, 2] < 0.0)
            v[downward, 2] *= -self.restitution
            v[below, :2] *= 1.0 - self.wall_friction

        # Confinement begins below the lip; crossing detection prevents
        # high-speed material points from tunneling through its top edge.
        confined = (p[:, 2] <= self.rim_z + 0.035) | (
            (old_position[indices, 2] > self.rim_z)
            & (p[:, 2] <= self.rim_z)
        )

        for axis in (0, 1):
            outside = confined & (np.abs(p[:, axis]) > self.axis_limit)
            if not np.any(outside):
                continue
            sign = np.sign(p[outside, axis])
            p[outside, axis] = sign * self.axis_limit
            rows = np.flatnonzero(outside)
            normal_speed = v[rows, axis] * sign
            moving_out = normal_speed > 0.0
            v[rows[moving_out], axis] *= -self.restitution
            tangent = 1 - axis
            v[rows, tangent] *= 1.0 - self.wall_friction

        diagonal = np.abs(p[:, 0]) + np.abs(p[:, 1])
        outside = confined & (diagonal > self.diagonal_limit)
        if np.any(outside):
            rows = np.flatnonzero(outside)
            sx = np.where(p[rows, 0] >= 0.0, 1.0, -1.0)
            sy = np.where(p[rows, 1] >= 0.0, 1.0, -1.0)
            excess = (diagonal[rows] - self.diagonal_limit) * 0.5
            p[rows, 0] -= sx * excess
            p[rows, 1] -= sy * excess
            normals = np.column_stack((sx, sy)) / math.sqrt(2.0)
            normal_speed = np.sum(v[rows, :2] * normals, axis=1)
            moving_out = normal_speed > 0.0
            if np.any(moving_out):
                rr = rows[moving_out]
                v[rr, :2] -= (
                    (1.0 + self.restitution)
                    * normal_speed[moving_out, None]
                    * normals[moving_out]
                )
            v[rows, 2] *= 1.0 - 0.5 * self.wall_friction

        self.position[indices] = p
        self.velocity[indices] = v

    def _step(self, dt: float) -> None:
        active = self.release_time <= self.time + 1.0e-12
        grid_mass, grid_volume, momentum, indices = self._deposit(active)
        occupied = grid_mass > 1.0e-12
        old_grid_velocity = np.zeros_like(momentum)
        old_grid_velocity[occupied] = momentum[occupied] / grid_mass[occupied, None]

        fill = self._smoothed_fill(grid_volume)
        compression = np.maximum(fill / self.rest_fill - 1.0, 0.0)
        pressure = self.bulk_modulus * (2.0 * compression + compression * compression)
        pressure = np.minimum(pressure, self.max_pressure)
        pressure[~occupied] = 0.0

        grad_p = np.stack(np.gradient(pressure, self.dx, edge_order=1), axis=-1)
        mixture_density = np.zeros(self.grid_shape, dtype=np.float64)
        has_volume = grid_volume > 1.0e-12
        mixture_density[has_volume] = grid_mass[has_volume] / grid_volume[has_volume]

        acceleration = np.zeros_like(momentum)
        acceleration[occupied] = self.gravity
        pressure_acceleration = np.zeros_like(momentum)
        pressure_acceleration[has_volume] = (
            -grad_p[has_volume] / mixture_density[has_volume, None]
        )
        magnitude = np.linalg.norm(pressure_acceleration, axis=-1)
        scale = np.minimum(1.0, 180.0 / np.maximum(magnitude, 1.0e-12))
        acceleration += pressure_acceleration * scale[..., None]

        new_grid_velocity = old_grid_velocity + dt * acceleration
        new_grid_velocity[~occupied] = 0.0

        if len(indices):
            pic_velocity = self._interpolate(new_grid_velocity, indices)
            grid_delta = self._interpolate(
                new_grid_velocity - old_grid_velocity, indices
            )
            flip_velocity = self.velocity[indices] + grid_delta
            self.velocity[indices] = (
                self.flip_fraction * flip_velocity
                + (1.0 - self.flip_fraction) * pic_velocity
            )
            # Mild drag represents unresolved viscosity and damps grid ringing.
            self.velocity[indices] *= math.exp(-0.10 * dt)
            speed = np.linalg.norm(self.velocity[indices], axis=1)
            speed_scale = np.minimum(1.0, 12.0 / np.maximum(speed, 1.0e-12))
            self.velocity[indices] *= speed_scale[:, None]

        old_position = self.position.copy()
        self.position[indices] += dt * self.velocity[indices]
        self._collide_basin(old_position, active)

        inactive = ~active
        self.position[inactive] = self.rest_position[inactive]
        self.velocity[inactive] = 0.0
        self.time += dt
        self.step_count += 1

    def _diagnostic_fields(self):
        active = self.release_time <= self.time + 1.0e-10
        grid_mass, grid_volume, momentum, _ = self._deposit(active)
        occupied_mass = grid_mass > 1.0e-12
        velocity_field = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        velocity_field[occupied_mass] = (
            momentum[occupied_mass] / grid_mass[occupied_mass, None]
        )

        fill = self._smoothed_fill(grid_volume)
        inside = fill > 0.10
        if np.any(inside):
            outside_distance = distance_transform_edt(~inside) * self.dx
            inside_distance = distance_transform_edt(inside) * self.dx
            level_set = outside_distance - inside_distance
        else:
            level_set = np.full(self.grid_shape, self.characteristic_length)
        return (
            inside.astype(np.float32),
            level_set.astype(np.float32),
            velocity_field.astype(np.float32),
        )

    def _frame(self) -> StateFrame:
        active_mask, liquid_level_set, velocity_field = self._diagnostic_fields()
        frame = StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase_id": self.phase_id.copy(),
                "density": self.density.copy(),
                "release_time": self.release_time.copy(),
                "target_volume": self.volume.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "liquid_level_set": liquid_level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "gravity_z": float(self.gravity[2]),
                "heavy_phase_volume": float(np.sum(self.volume[self.phase_id == 0])),
                "light_phase_volume": float(np.sum(self.volume[self.phase_id == 1])),
                "light_release_time": float(np.min(self.release_time[self.phase_id == 1])),
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.dx),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")

        while self.time < target - 1.0e-12:
            dt = min(self.max_substep, target - self.time)
            future_releases = self.release_time[
                (self.release_time > self.time + 1.0e-12)
                & (self.release_time < self.time + dt - 1.0e-12)
            ]
            if len(future_releases):
                dt = float(np.min(future_releases) - self.time)
            self._step(dt)
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "weakly_compressible_flip_pic",
            "time": float(self.time),
            "substeps": int(self.step_count),
            "grid_spacing": float(self.dx),
            "finite": bool(
                np.all(np.isfinite(self.position))
                and np.all(np.isfinite(self.velocity))
            ),
        }
