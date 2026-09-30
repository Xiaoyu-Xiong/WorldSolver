"""Deterministic particle solver for the arch-obstacle dam break.

The liquid is advanced with a weakly-compressible SPH discretisation. Static
geometry is handled as analytic boxes, while the dam gate is lifted during the
first part of the simulation. The implementation has no engine dependency.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small, deterministic weakly-compressible SPH solver."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.config = scene.config
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.position)

        self.spacing = float(self.config["sampling"]["spacing"])
        self.rho0 = float(self.config["sampling"]["density"])
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)
        self.radius = 0.43 * self.spacing
        self.h = 1.30 * self.spacing
        self.max_step = 0.0025
        self.sound_speed = 12.0
        self.kinematic_viscosity = 0.004
        self.cohesion = 13.0
        self.max_acceleration = 260.0
        self.max_velocity = 11.0

        self.poly6 = 315.0 / (64.0 * math.pi * self.h**9)
        self.spiky_gradient = 45.0 / (math.pi * self.h**6)
        self.viscosity_laplacian = 45.0 / (math.pi * self.h**6)

        self.characteristic_length = float(
            initial.scalars.get("characteristic_length", self.spacing)
        )
        self.reference_velocity = float(
            initial.scalars.get("reference_velocity", 1.0)
        )

        shape = tuple(int(v) for v in self.config["sampling"]["field_grid_shape"])
        self.grid_shape = shape
        ix, iy, iz = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing="ij",
        )
        origin = np.array([-2.76, -1.05, -0.07], dtype=np.float64)
        self.grid_points = origin + self.spacing * np.column_stack(
            (ix.ravel(), iy.ravel(), iz.ravel())
        )
        self.active_mask = self._make_active_mask(self.grid_points).reshape(shape)

        tank_normals = np.array(
            [[0, 0, 1], [1, 0, 0], [-1, 0, 0],
             [0, 1, 0], [0, -1, 0]], dtype=np.float64
        )
        face_normals = np.array(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
             [0, 0, -1], [0, 0, 1]], dtype=np.float64
        )
        self.boundary_normal = np.vstack(
            (tank_normals, face_normals, face_normals, face_normals,
             face_normals)
        )
        self.zero_boundary_velocity = np.zeros_like(self.boundary_normal)

        self.deformation_gradient = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], self.n, axis=0
        )
        self._steps = 0
        self._last_max_speed = 0.0

    def _make_active_mask(self, points: np.ndarray) -> np.ndarray:
        p = points
        active = (
            (p[:, 0] >= -3.14) & (p[:, 0] <= 3.14)
            & (p[:, 1] >= -1.34) & (p[:, 1] <= 1.34)
            & (p[:, 2] >= 0.0) & (p[:, 2] <= 2.35)
        )
        for lo, hi in self._arch_boxes(expand=0.0):
            active &= ~np.all((p >= lo) & (p <= hi), axis=1)
        return active

    @staticmethod
    def _arch_boxes(expand: float = 0.0):
        e = float(expand)
        return (
            (np.array([-0.97-e, -0.70-e, -e]),
             np.array([-0.53+e, 0.70+e, 1.40+e])),
            (np.array([0.53-e, -0.70-e, -e]),
             np.array([0.97+e, 0.70+e, 1.40+e])),
            (np.array([-0.95-e, -0.70-e, 1.22-e]),
             np.array([0.95+e, 0.70+e, 1.58+e])),
        )

    def _accelerations(self) -> np.ndarray:
        """Evaluate symmetric pressure, viscosity, and cohesion forces."""
        x = self.position
        v = self.velocity
        diff = x[:, None, :] - x[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff)
        near = (r2 > 1.0e-14) & (r2 < self.h * self.h)
        r = np.sqrt(np.maximum(r2, 1.0e-24))

        q2 = np.maximum(self.h * self.h - r2, 0.0)
        density = self.poly6 * ((q2**3) @ self.mass)
        density = np.maximum(density, 0.25 * self.rho0)

        pressure = self.sound_speed**2 * np.maximum(
            density - 0.96*self.rho0, 0.0
        )
        inv_rho2 = 1.0 / (density * density)
        pair_pressure = -self.mass[None, :] * (
            pressure[:, None] * inv_rho2[:, None]
            + pressure[None, :] * inv_rho2[None, :]
        )

        unit = np.zeros_like(diff)
        unit[near] = diff[near] / r[near, None]
        grad_w = (
            -self.spiky_gradient * np.maximum(self.h-r, 0.0)**2
        )[:, :, None] * unit
        acceleration = np.sum(pair_pressure[:, :, None] * grad_w, axis=1)

        lap_w = self.viscosity_laplacian * np.maximum(self.h-r, 0.0)
        lap_w *= near
        visc_weight = self.mass[None, :] * lap_w / density[None, :]
        acceleration += self.kinematic_viscosity * np.sum(
            visc_weight[:, :, None] * (v[None, :, :] - v[:, None, :]), axis=1
        )

        stretch = np.maximum(r - 0.92*self.spacing, 0.0)
        cohesive = near * stretch * (r > self.spacing)
        acceleration += np.sum(
            -self.cohesion * cohesive[:, :, None] * unit, axis=1
        )
        acceleration += self.gravity[None, :]

        amag = np.linalg.norm(acceleration, axis=1)
        limited = amag > self.max_acceleration
        if np.any(limited):
            acceleration[limited] *= (
                self.max_acceleration / amag[limited]
            )[:, None]
        return acceleration

    @staticmethod
    def _collide_box(position, velocity, lo, hi, wall_velocity,
                     restitution=0.035, tangential_keep=0.985):
        inside = np.all((position > lo) & (position < hi), axis=1)
        indices = np.flatnonzero(inside)
        if len(indices) == 0:
            return
        p = position[indices]
        to_low = p - lo
        to_high = hi - p
        distances = np.concatenate((to_low, to_high), axis=1)
        face = np.argmin(distances, axis=1)
        penetration = distances[np.arange(len(indices)), face]
        normals = np.zeros_like(p)
        for axis in range(3):
            low = face == axis
            high = face == axis + 3
            normals[low, axis] = -1.0
            normals[high, axis] = 1.0
        position[indices] += normals * (penetration + 1.0e-6)[:, None]

        wv = np.broadcast_to(np.asarray(wall_velocity, dtype=np.float64), p.shape)
        relative = velocity[indices] - wv
        vn = np.sum(relative * normals, axis=1)
        hitting = vn < 0.0
        relative[hitting] -= (
            (1.0 + restitution) * vn[hitting, None] * normals[hitting]
        )
        normal_part = np.sum(relative * normals, axis=1)[:, None] * normals
        relative = normal_part + tangential_keep * (relative - normal_part)
        velocity[indices] = wv + relative

    def _resolve_boundaries(self, step_time: float) -> None:
        p, v, rad = self.position, self.velocity, self.radius

        bounds = ((0, -3.14+rad, 1.0), (0, 3.14-rad, -1.0),
                  (1, -1.34+rad, 1.0), (1, 1.34-rad, -1.0),
                  (2, rad, 1.0))
        for axis, value, inward_sign in bounds:
            if inward_sign > 0:
                hit = p[:, axis] < value
            else:
                hit = p[:, axis] > value
            if not np.any(hit):
                continue
            p[hit, axis] = value
            vn = inward_sign * v[hit, axis]
            incoming = vn < 0.0
            selected = np.flatnonzero(hit)
            v[selected[incoming], axis] *= -0.04
            tangent = [a for a in range(3) if a != axis]
            v[np.ix_(selected, tangent)] *= 0.985

        zero = np.zeros(3, dtype=np.float64)
        for lo, hi in self._arch_boxes(expand=rad):
            self._collide_box(p, v, lo, hi, zero)

        gate_speed = 4.0
        gate_centre_z = 0.55 + gate_speed * step_time
        if gate_centre_z - 0.55 - rad < 1.75:
            lo = np.array([-1.59-rad, -0.65-rad,
                           gate_centre_z-0.55-rad])
            hi = np.array([-1.51+rad, 0.65+rad,
                           gate_centre_z+0.55+rad])
            self._collide_box(
                p, v, lo, hi, np.array([0.0, 0.0, gate_speed]),
                restitution=0.02, tangential_keep=0.99,
            )

        # A box corner projection can choose the downward obstacle face; the
        # floor constraint has precedence where obstacle and ground meet.
        below = p[:, 2] < rad
        if np.any(below):
            p[below, 2] = rad
            falling = below & (v[:, 2] < 0.0)
            v[falling, 2] *= -0.04

    def _substep(self, dt: float) -> None:
        acceleration = self._accelerations()
        self.velocity += dt * acceleration
        speed = np.linalg.norm(self.velocity, axis=1)
        fast = speed > self.max_velocity
        if np.any(fast):
            self.velocity[fast] *= (self.max_velocity / speed[fast])[:, None]
        self.position += dt * self.velocity
        self._resolve_boundaries(self.time + dt)
        self.time += dt
        self._steps += 1
        self._last_max_speed = float(np.max(np.linalg.norm(self.velocity, axis=1)))

    def _diagnostic_fields(self) -> dict[str, np.ndarray]:
        tree = cKDTree(self.position)
        k = min(4, self.n)
        distance, index = tree.query(self.grid_points, k=k)
        if k == 1:
            distance = distance[:, None]
            index = index[:, None]
        weights = 1.0 / np.maximum(distance, 0.30*self.spacing)**2
        weights /= np.sum(weights, axis=1, keepdims=True)
        grid_velocity = np.sum(
            weights[:, :, None] * self.velocity[index], axis=1
        )
        grid_velocity[~self.active_mask.ravel()] = 0.0
        level_set = distance[:, 0] - 0.78*self.spacing
        boundary_target = self.zero_boundary_velocity.copy()
        if 4.0*self.time - self.radius < 1.75:
            boundary_target[-6:, 2] = 4.0
        return {
            "active_mask": self.active_mask.copy(),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_target.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "liquid_level_set": level_set.reshape(self.grid_shape),
            "velocity_field": grid_velocity.reshape(self.grid_shape + (3,)),
        }

    def _frame(self, requested_time: float) -> StateFrame:
        frame = StateFrame(
            time=float(requested_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.n, self.radius, dtype=np.float64),
                "material_label": self.body_id.copy(),
                "deformation_gradient": self.deformation_gradient.copy(),
            },
            fields=self._diagnostic_fields(),
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        remaining = target - self.time
        if remaining > 1.0e-12:
            steps = max(1, int(math.ceil(remaining / self.max_step)))
            dt = remaining / steps
            for _ in range(steps):
                self._substep(dt)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "weakly_compressible_sph",
            "particle_count": int(self.n),
            "substeps": int(self._steps),
            "max_speed": float(self._last_max_speed),
            "simulated_time": float(self.time),
        }
