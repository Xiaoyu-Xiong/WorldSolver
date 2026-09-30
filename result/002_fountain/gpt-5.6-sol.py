"""Particle-based free-surface fountain solver.

This is a compact weakly-compressible SPH model. Particles carry liquid mass,
and a recirculating pump condition accelerates liquid in the nozzle throat.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic WCSPH integration with basin and nozzle boundaries."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.config = scene.config
        ic = self.config["initial_conditions"]

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.source_region = np.asarray(
            initial.attributes.get("source_region", np.zeros(len(self.mass))),
            dtype=np.float64,
        ).copy()
        self.time = 0.0
        self.steps = 0

        self.rho0 = float(ic["liquid_density"])
        self.spacing = float(ic["particle_spacing"])
        self.pool_radius = float(ic["pool_radius"])
        self.surface_z = float(ic["pool_surface_z"])
        self.nozzle_radius = float(ic["nozzle_radius"])
        self.nozzle_exit = float(ic["nozzle_exit_z"])
        self.jet_speed = float(ic["jet_speed"])
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)

        # Wendland C2 support and stable explicit integration parameters.
        self.h_sph = 0.78 * self.spacing
        self.support = 2.0 * self.h_sph
        self.sound_speed = 18.0
        self.viscosity_rate = 7.0
        self.internal_dt = 0.002
        self.particle_radius = 0.40 * self.spacing

        rng = np.random.default_rng(int(seed))
        self.noise_phase = rng.uniform(0.0, 2.0 * math.pi, len(self.mass))

        self.grid_spacing = 0.10
        self.grid_shape = (30, 30, 32)
        gx = -1.5 + (np.arange(self.grid_shape[0]) + 0.5) * self.grid_spacing
        gy = -1.5 + (np.arange(self.grid_shape[1]) + 0.5) * self.grid_spacing
        gz = (np.arange(self.grid_shape[2]) + 0.5) * self.grid_spacing
        mesh = np.meshgrid(gx, gy, gz, indexing="ij")
        self.grid_points = np.column_stack([a.ravel() for a in mesh])

        self.boundary_normal = self._make_boundary_normals()
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

    def _make_boundary_normals(self) -> np.ndarray:
        """Inward normals at representative side-wall and floor samples."""
        angle = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        side = np.column_stack((-np.cos(angle), -np.sin(angle), np.zeros_like(angle)))
        floor = np.tile(np.array([[0.0, 0.0, 1.0]]), (25, 1))
        return np.vstack((side, floor))

    def _sph_acceleration(self) -> np.ndarray:
        n = len(self.position)
        acceleration = np.broadcast_to(self.gravity, (n, 3)).copy()
        pairs = cKDTree(self.position).query_pairs(self.support, output_type="ndarray")

        alpha = 21.0 / (16.0 * math.pi * self.h_sph ** 3)
        density = self.mass * alpha
        if pairs.size == 0:
            return acceleration

        i = pairs[:, 0]
        j = pairs[:, 1]
        delta = self.position[i] - self.position[j]
        distance = np.linalg.norm(delta, axis=1)
        valid = distance > 1.0e-10
        i, j, delta, distance = i[valid], j[valid], delta[valid], distance[valid]
        if len(i) == 0:
            return acceleration

        q = distance / self.h_sph
        taper = np.maximum(0.0, 1.0 - 0.5 * q)
        kernel = alpha * taper ** 4 * (2.0 * q + 1.0)
        np.add.at(density, i, self.mass[j] * kernel)
        np.add.at(density, j, self.mass[i] * kernel)
        density = np.maximum(density, 0.35 * self.rho0)

        # Limited tensile pressure supplies free-surface cohesion; compression
        # remains the dominant pressure response.
        pressure = self.sound_speed ** 2 * (density - self.rho0)
        pressure = np.maximum(pressure, -1800.0)
        minus_dkernel = alpha * (5.0 * q / self.h_sph) * taper ** 3
        direction = delta / distance[:, None]
        coefficient = self.mass[j] * (
            pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2
        ) * minus_dkernel
        pair_acceleration = coefficient[:, None] * direction
        np.add.at(acceleration, i, pair_acceleration)
        np.add.at(
            acceleration,
            j,
            -pair_acceleration * (self.mass[i] / self.mass[j])[:, None],
        )

        # Kernel-normalized velocity diffusion supplies SPH viscosity.
        weight_sum = np.bincount(i, weights=kernel, minlength=n)
        weight_sum += np.bincount(j, weights=kernel, minlength=n)
        velocity_sum = np.zeros((n, 3), dtype=np.float64)
        np.add.at(velocity_sum, i, kernel[:, None] * self.velocity[j])
        np.add.at(velocity_sum, j, kernel[:, None] * self.velocity[i])
        mask = weight_sum > 0.0
        neighbor_velocity = self.velocity.copy()
        neighbor_velocity[mask] = velocity_sum[mask] / weight_sum[mask, None]
        acceleration += self.viscosity_rate * (neighbor_velocity - self.velocity)

        magnitude = np.linalg.norm(acceleration, axis=1)
        too_large = magnitude > 500.0
        acceleration[too_large] *= (500.0 / magnitude[too_large])[:, None]
        return acceleration

    def _apply_nozzle(self, dt: float, step_time: float) -> None:
        radial2 = np.sum(self.position[:, :2] ** 2, axis=1)
        in_throat = (
            (radial2 < self.nozzle_radius ** 2)
            & (self.position[:, 2] < self.nozzle_exit + 0.10)
            & (self.position[:, 2] > 0.025)
        )
        ids = np.flatnonzero(in_throat)
        if len(ids) == 0:
            return

        # Deterministic inlet turbulence seeds sheet and droplet breakup.
        phase = self.noise_phase[ids] + 8.0 * step_time
        target = np.zeros((len(ids), 3), dtype=np.float64)
        target[:, 0] = 0.13 * np.cos(phase)
        target[:, 1] = 0.13 * np.sin(phase)
        profile = np.clip(1.0 - radial2[ids] / self.nozzle_radius ** 2, 0.0, 1.0)
        target[:, 2] = self.jet_speed * (0.82 + 0.18 * profile)
        blend = 1.0 - math.exp(-dt / 0.012)
        self.velocity[ids] += blend * (target - self.velocity[ids])

    def _collide(self) -> None:
        radius = np.linalg.norm(self.position[:, :2], axis=1)
        wall = self.pool_radius + 0.05 - self.particle_radius
        outside = radius > wall
        if np.any(outside):
            normal = self.position[outside, :2] / radius[outside, None]
            self.position[outside, :2] = normal * wall
            vn = np.sum(self.velocity[outside, :2] * normal, axis=1)
            moving_out = vn > 0.0
            corrected = self.velocity[outside, :2]
            corrected[moving_out] -= 1.25 * vn[moving_out, None] * normal[moving_out]
            corrected *= 0.92
            self.velocity[outside, :2] = corrected

        floor_z = self.particle_radius
        below = self.position[:, 2] < floor_z
        self.position[below, 2] = floor_z
        downward = below & (self.velocity[:, 2] < 0.0)
        self.velocity[downward, 2] *= -0.12
        self.velocity[below, :2] *= 0.94

        # The visible metal nozzle is a solid annulus around its open throat.
        radius = np.linalg.norm(self.position[:, :2], axis=1)
        in_metal = (
            (self.position[:, 2] < 0.34 + self.particle_radius)
            & (radius >= self.nozzle_radius)
            & (radius < 0.24 + self.particle_radius)
        )
        if np.any(in_metal):
            normal = self.position[in_metal, :2] / np.maximum(radius[in_metal, None], 1e-12)
            shell = 0.24 + self.particle_radius
            self.position[in_metal, :2] = normal * shell
            vn = np.sum(self.velocity[in_metal, :2] * normal, axis=1)
            into_shell = vn < 0.0
            local_velocity = self.velocity[in_metal, :2]
            local_velocity[into_shell] -= 1.2 * vn[into_shell, None] * normal[into_shell]
            local_velocity *= 0.9
            self.velocity[in_metal, :2] = local_velocity

    def _substep(self, dt: float) -> None:
        acceleration = self._sph_acceleration()
        self.velocity += dt * acceleration
        self._apply_nozzle(dt, self.time + 0.5 * dt)
        speed = np.linalg.norm(self.velocity, axis=1)
        fast = speed > 12.0
        self.velocity[fast] *= (12.0 / speed[fast])[:, None]
        self.position += dt * self.velocity
        self._collide()
        self.time += dt
        self.steps += 1

    def _grid_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree = cKDTree(self.position)
        distance, nearest = tree.query(self.grid_points, k=1)
        level_set = distance - 0.90 * self.spacing
        velocity = self.velocity[nearest].copy()
        velocity[distance > 1.8 * self.spacing] = 0.0
        active = level_set <= 0.75 * self.grid_spacing
        shape = self.grid_shape
        return (
            active.reshape(shape).astype(np.uint8),
            level_set.reshape(shape),
            velocity.reshape(shape + (3,)),
        )

    def _frame(self, requested_time: float) -> StateFrame:
        active, level_set, grid_velocity = self._grid_fields()
        frame = StateFrame(
            time=float(requested_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "source_region": self.source_region.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "active_mask": active,
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": np.zeros_like(self.boundary_normal),
                "cumulative_mass_source": np.array([0.0], dtype=np.float64),
                "liquid_level_set": level_set,
                "velocity_field": grid_velocity,
            },
            scalars={
                "rest_density": self.rho0,
                "particle_spacing": self.spacing,
                "characteristic_length": self.pool_radius,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": self.jet_speed,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-9:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time < target - 1.0e-12:
            self._substep(min(self.internal_dt, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "weakly_compressible_sph",
            "steps": self.steps,
            "particle_count": len(self.mass),
            "simulated_time": self.time,
            "total_mass": float(np.sum(self.mass)),
        }
