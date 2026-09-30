"""Finite-difference buoyant smoke solver for the benchmark scene.

The implementation uses a collocated three-dimensional grid, semi-Lagrangian
advection, a variable-domain pressure projection, and solid-cell boundary
conditions. It intentionally has no dependency on an external simulator.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic incompressible smoke simulation with a spherical obstacle."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.base = scene.initial_state()
        self.time = 0.0
        self.steps = 0

        shape = tuple(int(v) for v in self.config["sampling"]["field_grid_shape"])
        self.shape = shape
        self.lower = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.upper = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        self.spacing = (self.upper - self.lower) / (np.asarray(shape) - 1.0)

        axes = [np.linspace(self.lower[d], self.upper[d], shape[d]) for d in range(3)]
        self.x, self.y, self.z = np.meshgrid(*axes, indexing="ij")
        self.index_grid = np.asarray(
            np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                indexing="ij",
            ),
            dtype=np.float64,
        )

        self.sphere_center = np.array([0.0, 0.0, 1.35], dtype=np.float64)
        self.sphere_radius = 0.48
        rx = self.x - self.sphere_center[0]
        ry = self.y - self.sphere_center[1]
        rz = self.z - self.sphere_center[2]
        self.radius_grid = np.sqrt(rx * rx + ry * ry + rz * rz)

        # Grid endpoints represent enclosure walls; the sphere is a solid region.
        self.fluid = self.radius_grid >= self.sphere_radius
        self.fluid[[0, -1], :, :] = False
        self.fluid[:, [0, -1], :] = False
        self.fluid[:, :, [0, -1]] = False

        initial_fields = self.base.fields
        self.smoke = np.asarray(
            initial_fields["transported_scalar"], dtype=np.float64
        ).copy()
        self.smoke[~self.fluid] = 0.0
        self.velocity = np.zeros(shape + (3,), dtype=np.float64)

        radial2 = self.x * self.x + self.y * self.y
        source_profile = np.exp(-((radial2 / (0.27 ** 2)) ** 2))
        source_height = (self.z >= 0.07) & (self.z <= 0.30)
        self.scalar_source = 3.2 * source_profile * source_height * self.fluid

        self._prepare_pressure_stencil()
        self.boundary_normal = self._make_boundary_normals()
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

        self.position = np.asarray(self.base.position, dtype=np.float64).copy()
        self.particle_velocity = np.zeros_like(self.position)
        self.mass = np.asarray(self.base.mass).copy()
        self.body_id = np.asarray(self.base.body_id).copy()
        self.attributes = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in self.base.attributes.items()
        }
        self.cumulative_mass_source = np.zeros(len(self.position), dtype=np.float64)
        self.scalars = dict(self.base.scalars)
        self.scalars["grid_spacing"] = float(self.config["sampling"]["spacing"])
        self.scalars["reference_velocity"] = 1.0

    def _prepare_pressure_stencil(self) -> None:
        self.neighbor_masks = []
        denominator = np.zeros(self.shape, dtype=np.float64)
        for axis, h in enumerate(self.spacing):
            plus = self.fluid & np.roll(self.fluid, -1, axis=axis)
            minus = self.fluid & np.roll(self.fluid, 1, axis=axis)
            edge_plus = [slice(None)] * 3
            edge_minus = [slice(None)] * 3
            edge_plus[axis] = -1
            edge_minus[axis] = 0
            plus[tuple(edge_plus)] = False
            minus[tuple(edge_minus)] = False
            weight = 1.0 / (h * h)
            denominator += weight * (
                plus.astype(np.float64) + minus.astype(np.float64)
            )
            self.neighbor_masks.append((plus, minus, weight))
        self.pressure_denominator = np.where(denominator > 0.0, denominator, 1.0)

    @staticmethod
    def _fibonacci_normals(count: int) -> np.ndarray:
        i = np.arange(count, dtype=np.float64)
        z = 1.0 - 2.0 * (i + 0.5) / count
        angle = i * (math.pi * (3.0 - math.sqrt(5.0)))
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack(
            (radius * np.cos(angle), radius * np.sin(angle), z)
        )

    def _make_boundary_normals(self) -> np.ndarray:
        # Normals point from each stationary wall into its adjacent fluid.
        wall_normals = []
        for normal, count in (
            ((1.0, 0.0, 0.0), 24), ((-1.0, 0.0, 0.0), 24),
            ((0.0, 1.0, 0.0), 32), ((0.0, -1.0, 0.0), 32),
            ((0.0, 0.0, 1.0), 24), ((0.0, 0.0, -1.0), 24),
        ):
            wall_normals.append(np.tile(np.asarray(normal), (count, 1)))
        # Sphere normals point from the obstacle into the fluid.
        wall_normals.append(self._fibonacci_normals(96))
        return np.concatenate(wall_normals, axis=0).astype(np.float64)

    def _sample_grid(self, field: np.ndarray, positions: np.ndarray) -> np.ndarray:
        indices = (positions - self.lower) / self.spacing
        coords = np.vstack([indices[:, d] for d in range(3)])
        if field.ndim == 3:
            return map_coordinates(
                field, coords, order=1, mode="nearest", prefilter=False
            )
        return np.column_stack([
            map_coordinates(
                field[..., d], coords, order=1, mode="nearest", prefilter=False
            )
            for d in range(field.shape[-1])
        ])

    def _advect(
        self, field: np.ndarray, velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        departure = self.index_grid.copy()
        for d in range(3):
            departure[d] -= dt * velocity[..., d] / self.spacing[d]
            np.clip(
                departure[d], 0.0, self.shape[d] - 1.0, out=departure[d]
            )
        if field.ndim == 3:
            result = map_coordinates(
                field, departure, order=1, mode="nearest", prefilter=False
            )
            result[~self.fluid] = 0.0
            return result
        result = np.empty_like(field)
        for d in range(3):
            result[..., d] = map_coordinates(
                field[..., d], departure, order=1, mode="nearest", prefilter=False
            )
        result[~self.fluid] = 0.0
        return result

    def _derivative(self, field: np.ndarray, axis: int) -> np.ndarray:
        plus_mask, minus_mask, _ = self.neighbor_masks[axis]
        plus = np.where(plus_mask, np.roll(field, -1, axis=axis), field)
        minus = np.where(minus_mask, np.roll(field, 1, axis=axis), field)
        result = (plus - minus) / (2.0 * self.spacing[axis])
        result[~self.fluid] = 0.0
        return result

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        result = np.zeros_like(field)
        for axis, (plus_mask, minus_mask, weight) in enumerate(self.neighbor_masks):
            if field.ndim == 4:
                pm = plus_mask[..., None]
                mm = minus_mask[..., None]
            else:
                pm, mm = plus_mask, minus_mask
            plus = np.where(pm, np.roll(field, -1, axis=axis), field)
            minus = np.where(mm, np.roll(field, 1, axis=axis), field)
            result += weight * (plus + minus - 2.0 * field)
        result[~self.fluid] = 0.0
        return result

    def _apply_boundaries(self, velocity: np.ndarray) -> None:
        velocity[~self.fluid] = 0.0

        # Retain tangential flow next to the sphere but remove normal flow, making
        # the rising plume slide around rather than enter the rigid surface.
        shell_width = 1.35 * float(np.max(self.spacing))
        shell = self.fluid & (
            self.radius_grid < self.sphere_radius + shell_width
        )
        rx = self.x[shell] - self.sphere_center[0]
        ry = self.y[shell] - self.sphere_center[1]
        rz = self.z[shell] - self.sphere_center[2]
        normal = np.column_stack((rx, ry, rz))
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1.0e-12)
        local = velocity[shell]
        velocity[shell] = (
            local - np.sum(local * normal, axis=1)[:, None] * normal
        )

    def _vorticity_confinement(
        self, velocity: np.ndarray, dt: float
    ) -> None:
        du_dy = self._derivative(velocity[..., 0], 1)
        du_dz = self._derivative(velocity[..., 0], 2)
        dv_dx = self._derivative(velocity[..., 1], 0)
        dv_dz = self._derivative(velocity[..., 1], 2)
        dw_dx = self._derivative(velocity[..., 2], 0)
        dw_dy = self._derivative(velocity[..., 2], 1)
        omega = np.stack(
            (dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy), axis=-1
        )
        magnitude = np.linalg.norm(omega, axis=-1)
        grad = np.stack(
            [self._derivative(magnitude, d) for d in range(3)], axis=-1
        )
        grad /= np.maximum(
            np.linalg.norm(grad, axis=-1)[..., None], 1.0e-9
        )
        force = np.cross(grad, omega)
        velocity += dt * 0.20 * float(np.mean(self.spacing)) * force

    def _project(self, velocity: np.ndarray, dt: float) -> None:
        divergence = sum(
            self._derivative(velocity[..., d], d) for d in range(3)
        )
        rhs = divergence / max(dt, 1.0e-12)
        rhs[~self.fluid] = 0.0
        if np.any(self.fluid):
            rhs[self.fluid] -= np.mean(rhs[self.fluid])

        pressure = np.zeros(self.shape, dtype=np.float64)
        for _ in range(32):
            neighbor_sum = np.zeros_like(pressure)
            for axis, (plus_mask, minus_mask, weight) in enumerate(
                self.neighbor_masks
            ):
                neighbor_sum += weight * (
                    np.where(
                        plus_mask, np.roll(pressure, -1, axis=axis), 0.0
                    )
                    + np.where(
                        minus_mask, np.roll(pressure, 1, axis=axis), 0.0
                    )
                )
            pressure = (neighbor_sum - rhs) / self.pressure_denominator
            pressure[~self.fluid] = 0.0
            pressure[self.fluid] -= np.mean(pressure[self.fluid])

        for d in range(3):
            velocity[..., d] -= dt * self._derivative(pressure, d)
        self._apply_boundaries(velocity)

    def _advance_particles(self, dt: float) -> None:
        particle_velocity = self._sample_grid(self.velocity, self.position)
        self.position += dt * particle_velocity
        np.clip(
            self.position, self.lower + 1.0e-5, self.upper - 1.0e-5,
            out=self.position,
        )

        delta = self.position - self.sphere_center
        distance = np.linalg.norm(delta, axis=1)
        inside = distance < self.sphere_radius + 1.0e-4
        if np.any(inside):
            direction = delta[inside]
            length = np.linalg.norm(direction, axis=1)
            degenerate = length < 1.0e-12
            direction[degenerate] = np.array([1.0, 0.0, 0.0])
            length[degenerate] = 1.0
            direction /= length[:, None]
            self.position[inside] = self.sphere_center + (
                self.sphere_radius + 1.0e-4
            ) * direction
            normal_speed = np.sum(
                particle_velocity[inside] * direction, axis=1
            )
            particle_velocity[inside] -= normal_speed[:, None] * direction
        self.particle_velocity = particle_velocity

    def _step(self, dt: float) -> None:
        self.velocity = self._advect(self.velocity, self.velocity, dt)
        self.velocity += dt * 0.003 * self._laplacian(self.velocity)

        # Smoke is lighter than ambient gas, so gravity creates upward buoyancy.
        self.velocity[..., 2] += dt * 2.35 * self.smoke
        self.velocity *= math.exp(-0.055 * dt)
        self._vorticity_confinement(self.velocity, dt)
        self._apply_boundaries(self.velocity)
        self._project(self.velocity, dt)

        speed = np.linalg.norm(self.velocity, axis=-1)
        limiter = np.minimum(1.0, 3.5 / np.maximum(speed, 1.0e-12))
        self.velocity *= limiter[..., None]

        self.smoke = self._advect(self.smoke, self.velocity, dt)
        self.smoke += dt * (
            0.0022 * self._laplacian(self.smoke) + self.scalar_source
        )
        self.smoke *= math.exp(-0.035 * dt)
        np.clip(self.smoke, 0.0, 4.0, out=self.smoke)
        self.smoke[~self.fluid] = 0.0

        self._advance_particles(dt)
        self.time += dt
        self.steps += 1

        if (
            not np.isfinite(self.smoke).all()
            or not np.isfinite(self.velocity).all()
        ):
            np.nan_to_num(
                self.smoke, copy=False, nan=0.0, posinf=4.0, neginf=0.0
            )
            np.nan_to_num(
                self.velocity, copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )

    def _frame(self, requested_time: float) -> StateFrame:
        fields = {
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "active_mask": self.fluid.copy(),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": np.zeros_like(self.boundary_normal),
            "scalar_source": self.scalar_source.copy(),
            "transported_scalar": self.smoke.copy(),
            "velocity_field": self.velocity.copy(),
        }
        frame = StateFrame(
            float(requested_time),
            self.position.copy(),
            self.particle_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.attributes.items()
            },
            fields=fields,
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-9:
            raise ValueError("BenchmarkSolver cannot integrate backward in time")

        # The fixed cap keeps force integration and pressure correction smooth.
        max_dt = 1.0 / 30.0
        while self.time < target - 1.0e-10:
            self._step(min(max_dt, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_lagrangian_incompressible_smoke",
            "grid_shape": list(self.shape),
            "step_count": int(self.steps),
            "simulated_time": float(self.time),
            "max_smoke": float(np.max(self.smoke)),
            "max_speed": float(np.max(np.linalg.norm(self.velocity, axis=-1))),
        }
