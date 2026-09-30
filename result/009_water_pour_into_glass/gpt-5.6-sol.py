"""Deterministic particle-in-cell solver for the pouring-water benchmark.

The liquid is represented by Lagrangian marker particles. Momentum is advanced
on a uniform Eulerian grid, where a pressure projection supplies the
incompressibility constraint, and is transferred back with a damped FLIP/PIC
update. The inlet is a prescribed flux boundary; every emitted marker carries
its share of the integrated source mass.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A compact, deterministic free-surface PIC/FLIP solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.source_region = np.asarray(
            initial.attributes.get("source_region", np.zeros(len(self.position))),
            dtype=np.float64,
        ).copy()

        ic = self.config["initial_conditions"]
        domain = self.config["domain"]
        self.rho = float(ic["liquid_density"])
        self.particle_spacing = float(ic["particle_spacing"])
        self.glass_radius = float(ic["glass_inner_radius"])
        self.glass_height = float(ic["glass_inner_height"])
        self.jet_radius = float(ic["jet_radius"])
        self.jet_speed = float(ic["jet_speed"])
        self.jet_start = np.asarray(ic["jet_start"], dtype=np.float64)
        jet_end = np.asarray(ic["jet_end"], dtype=np.float64)
        self.jet_direction = jet_end - self.jet_start
        self.jet_direction /= np.linalg.norm(self.jet_direction)
        self.source_start = float(ic["source_start_time"])
        self.source_end = float(ic["source_end_time"])
        self.flow_rate = float(ic["target_inflow_rate_m3_s"])
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)

        # Fine enough to resolve the jet while leaving time for pressure solves.
        self.grid_spacing = 0.024
        self.bounds_min = np.asarray(domain["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.grid_shape = tuple(
            np.ceil((bounds_max - self.bounds_min) / self.grid_spacing).astype(int)
        )
        self.current_time = 0.0
        self.cumulative_source_mass = 0.0
        self.emission_batch = 0
        self.internal_steps = 0
        self.initial_mass = float(np.sum(self.mass))
        self.last_grid_velocity = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        self.last_fluid_mask = np.zeros(self.grid_shape, dtype=bool)
        self._update_diagnostic_grid()

        # Fixed samples of the inner wall. Normals point out of the liquid and
        # target velocities are zero for the static glass.
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        normals = []
        for _z in (0.04, 0.20, 0.38, 0.56, 0.74):
            normals.extend((math.cos(a), math.sin(a), 0.0) for a in angles)
        normals.extend((0.0, 0.0, -1.0) for _ in range(32))
        self.boundary_normal = np.asarray(normals, dtype=np.float64)
        self.boundary_target = np.zeros_like(self.boundary_normal)

    def _grid_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.floor((positions - self.bounds_min) / self.grid_spacing).astype(np.int64)
        shape = np.asarray(self.grid_shape, dtype=np.int64)
        valid = np.all((idx >= 0) & (idx < shape), axis=1)
        return idx, valid

    def _particles_to_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mass-weighted nearest-cell deposit followed by compact extrapolation."""
        idx, valid = self._grid_indices(self.position)
        shape = self.grid_shape
        size = int(np.prod(shape))
        flat = np.ravel_multi_index(idx[valid].T, shape)
        particle_mass = self.mass[valid]
        mass_grid = np.bincount(flat, weights=particle_mass, minlength=size).reshape(shape)
        momentum = np.empty(shape + (3,), dtype=np.float64)
        for axis in range(3):
            momentum[..., axis] = np.bincount(
                flat,
                weights=particle_mass * self.velocity[valid, axis],
                minlength=size,
            ).reshape(shape)

        # Smooth numerator and denominator together, which also extrapolates
        # velocity a short distance into the free surface.
        weight = gaussian_filter(mass_grid, 0.72, mode="constant")
        grid_velocity = np.zeros_like(momentum)
        for axis in range(3):
            smooth_momentum = gaussian_filter(momentum[..., axis], 0.72, mode="constant")
            np.divide(
                smooth_momentum,
                weight,
                out=grid_velocity[..., axis],
                where=weight > 1.0e-12,
            )
        cell_mass = self.rho * self.grid_spacing ** 3
        fluid = weight > 0.035 * cell_mass
        fluid = binary_dilation(fluid, iterations=1)
        return grid_velocity, fluid, mass_grid

    @staticmethod
    def _shift(array: np.ndarray, axis: int, amount: int) -> np.ndarray:
        """Shift with zero (free-surface pressure) rather than periodic wrap."""
        result = np.zeros_like(array)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if amount > 0:
            src[axis] = slice(None, -amount)
            dst[axis] = slice(amount, None)
        else:
            src[axis] = slice(-amount, None)
            dst[axis] = slice(None, amount)
        result[tuple(dst)] = array[tuple(src)]
        return result

    def _project(self, velocity: np.ndarray, fluid: np.ndarray) -> np.ndarray:
        """Solve the constant-density pressure Poisson equation by Jacobi steps."""
        h = self.grid_spacing
        divergence = np.zeros(self.grid_shape, dtype=np.float64)
        for axis in range(3):
            divergence += (
                self._shift(velocity[..., axis], axis, -1)
                - self._shift(velocity[..., axis], axis, 1)
            ) / (2.0 * h)
        divergence *= fluid

        pressure_dt_over_rho = np.zeros(self.grid_shape, dtype=np.float64)
        for _ in range(18):
            neighbor_sum = np.zeros_like(pressure_dt_over_rho)
            for axis in range(3):
                neighbor_sum += self._shift(pressure_dt_over_rho, axis, 1)
                neighbor_sum += self._shift(pressure_dt_over_rho, axis, -1)
            pressure_dt_over_rho = np.where(
                fluid, (neighbor_sum - h * h * divergence) / 6.0, 0.0
            )

        projected = velocity.copy()
        for axis in range(3):
            gradient = (
                self._shift(pressure_dt_over_rho, axis, -1)
                - self._shift(pressure_dt_over_rho, axis, 1)
            ) / (2.0 * h)
            projected[..., axis] -= gradient * fluid
        projected[~binary_dilation(fluid, iterations=1)] = 0.0
        np.clip(projected, -5.0, 5.0, out=projected)
        return projected

    def _sample_grid(self, grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Trilinear cell-centered grid interpolation."""
        coordinate = (positions - self.bounds_min) / self.grid_spacing - 0.5
        base = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - base
        result = np.zeros((len(positions), 3), dtype=np.float64)
        shape = np.asarray(self.grid_shape, dtype=np.int64)
        for ox in (0, 1):
            wx = (1.0 - fraction[:, 0]) if ox == 0 else fraction[:, 0]
            for oy in (0, 1):
                wy = (1.0 - fraction[:, 1]) if oy == 0 else fraction[:, 1]
                for oz in (0, 1):
                    wz = (1.0 - fraction[:, 2]) if oz == 0 else fraction[:, 2]
                    index = base + np.array((ox, oy, oz), dtype=np.int64)
                    valid = np.all((index >= 0) & (index < shape), axis=1)
                    if np.any(valid):
                        result[valid] += (
                            (wx * wy * wz)[valid, None]
                            * grid[index[valid, 0], index[valid, 1], index[valid, 2]]
                        )
        return result

    def _emit(self, dt: float, interval_start: float) -> None:
        active_dt = max(
            0.0,
            min(interval_start + dt, self.source_end)
            - max(interval_start, self.source_start),
        )
        if active_dt <= 0.0:
            return
        volume = self.flow_rate * active_dt
        nominal_volume = self.particle_spacing ** 3
        count = max(1, int(round(volume / nominal_volume)))

        # A rotating Vogel disk gives a uniform deterministic nozzle section.
        index = np.arange(count, dtype=np.float64) + 0.5
        radius = self.jet_radius * 0.88 * np.sqrt(index / count)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angle = golden_angle * index + 0.71 * self.emission_batch
        side = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        cross = np.cross(self.jet_direction, side)
        cross /= np.linalg.norm(cross)
        new_position = (
            self.jet_start[None, :]
            + (radius * np.cos(angle))[:, None] * side[None, :]
            + (radius * np.sin(angle))[:, None] * cross[None, :]
        )
        # Small zero-mean inlet perturbations seed resolved eddies after impact.
        swirl = 0.025 * np.sin(angle)
        new_velocity = np.repeat(
            (self.jet_speed * self.jet_direction)[None, :], count, axis=0
        )
        new_velocity += swirl[:, None] * side[None, :]

        new_mass = np.full(count, self.rho * volume / count, dtype=np.float64)
        self.position = np.vstack((self.position, new_position))
        self.velocity = np.vstack((self.velocity, new_velocity))
        self.mass = np.concatenate((self.mass, new_mass))
        self.source_region = np.concatenate((self.source_region, np.ones(count)))
        self.cumulative_source_mass += self.rho * volume
        self.emission_batch += 1

    def _collide_with_glass(self) -> None:
        p = self.position
        v = self.velocity
        half_h = 0.48 * self.particle_spacing

        # Axisymmetric inner profile: tapered base and nearly vertical wall.
        local_radius = np.where(
            p[:, 2] < 0.03,
            0.225 + (0.25 - 0.225) * np.clip(p[:, 2] / 0.03, 0.0, 1.0),
            0.25 + 0.005 * np.clip((p[:, 2] - 0.75) / 0.03, 0.0, 1.0),
        )
        allowed = local_radius - half_h
        radial = np.linalg.norm(p[:, :2], axis=1)
        hit_side = (p[:, 2] < self.glass_height) & (radial > allowed)
        if np.any(hit_side):
            normal = p[hit_side, :2] / np.maximum(radial[hit_side, None], 1.0e-12)
            p[hit_side, :2] = normal * allowed[hit_side, None]
            normal_speed = np.sum(v[hit_side, :2] * normal, axis=1)
            outward = normal_speed > 0.0
            side_velocity = v[hit_side, :2]
            side_velocity[outward] -= 1.08 * normal_speed[outward, None] * normal[outward]
            v[hit_side, :2] = 0.985 * side_velocity

        hit_bottom = p[:, 2] < half_h
        if np.any(hit_bottom):
            p[hit_bottom, 2] = half_h
            falling = v[hit_bottom, 2] < 0.0
            indices = np.flatnonzero(hit_bottom)
            v[indices[falling], 2] *= -0.04
            v[hit_bottom, :2] *= 0.975

    def _substep(self, dt: float) -> None:
        self._emit(dt, self.current_time)
        old_grid, fluid, _mass_grid = self._particles_to_grid()
        grid_velocity = old_grid + dt * self.gravity
        grid_velocity = self._project(grid_velocity, fluid)

        old_sample = self._sample_grid(old_grid, self.position)
        new_sample = self._sample_grid(grid_velocity, self.position)
        flip_velocity = self.velocity + (new_sample - old_sample)
        # Mostly-FLIP retains turbulent motion; PIC suppresses energy growth.
        self.velocity = 0.88 * flip_velocity + 0.12 * new_sample
        speed = np.linalg.norm(self.velocity, axis=1)
        excessive = speed > 5.0
        if np.any(excessive):
            self.velocity[excessive] *= (5.0 / speed[excessive])[:, None]
        self.position += dt * self.velocity
        self._collide_with_glass()
        self.current_time += dt
        self.internal_steps += 1
        self.last_grid_velocity = grid_velocity
        self.last_fluid_mask = fluid

    def _update_diagnostic_grid(self) -> None:
        velocity, fluid, _mass_grid = self._particles_to_grid()
        self.last_grid_velocity = velocity
        self.last_fluid_mask = fluid

    def _make_frame(self, time: float) -> StateFrame:
        fluid = self.last_fluid_mask
        outside_distance = distance_transform_edt(~fluid) * self.grid_spacing
        inside_distance = distance_transform_edt(fluid) * self.grid_spacing
        level_set = outside_distance - inside_distance
        boundary_velocity = np.zeros_like(self.boundary_normal)
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "source_region": self.source_region.copy(),
            },
            fields={
                "active_mask": fluid.astype(np.float64),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target.copy(),
                "boundary_velocity": boundary_velocity,
                "cumulative_mass_source": np.asarray(
                    [self.cumulative_source_mass], dtype=np.float64
                ),
                "liquid_level_set": level_set.astype(np.float64),
                "velocity_field": self.last_grid_velocity.copy(),
            },
            scalars={
                "characteristic_length": self.glass_radius * 2.0,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": self.jet_speed,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        remaining = target - self.current_time
        if remaining > 1.0e-12:
            # CFL <= about 0.75 for the prescribed 2.25 m/s jet.
            count = max(1, int(math.ceil(remaining / 0.0075)))
            dt = remaining / count
            for _ in range(count):
                self._substep(dt)
            self.current_time = target
        else:
            self._update_diagnostic_grid()
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "particle-in-cell FLIP with pressure projection",
            "particle_count": int(len(self.position)),
            "internal_steps": int(self.internal_steps),
            "initial_mass_kg": self.initial_mass,
            "cumulative_source_mass_kg": float(self.cumulative_source_mass),
            "final_mass_kg": float(np.sum(self.mass)),
            "grid_shape": list(self.grid_shape),
        }
