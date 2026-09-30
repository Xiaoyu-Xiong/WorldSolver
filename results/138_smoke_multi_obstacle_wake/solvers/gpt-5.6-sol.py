"""Deterministic grid based smoke solver for the benchmark scene.

The implementation is a compact ``stable fluids`` style solver. Smoke and
cell-centred velocity are semi-Lagrangian advected, buoyancy and viscosity are
applied, and a pressure projection enforces incompressibility around the two
solid obstacles and the enclosing walls.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A small, self-contained incompressible smoke simulation."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        cfg = scene.config

        self.shape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.lo = np.array([-3.0, -1.2, 0.0], dtype=float)
        self.hi = np.array([3.0, 1.2, 3.8], dtype=float)
        self.spacing3 = (self.hi - self.lo) / (np.asarray(self.shape) - 1)
        axes = [np.linspace(self.lo[a], self.hi[a], self.shape[a]) for a in range(3)]
        self.X, self.Y, self.Z = np.meshgrid(*axes, indexing="ij")
        self.index_grid = np.stack(np.meshgrid(
            np.arange(self.shape[0], dtype=float),
            np.arange(self.shape[1], dtype=float),
            np.arange(self.shape[2], dtype=float), indexing="ij"), axis=-1)

        # Fluid cells are all domain cells not occupied by either rigid box.
        active = np.ones(self.shape, dtype=bool)
        self.obstacles = [
            (np.array([-0.65, 0.0, 1.25]), np.array([0.38, 0.38, 0.38])),
            (np.array([0.65, 0.0, 2.15]), np.array([0.42, 0.42, 0.42])),
        ]
        for center, half in self.obstacles:
            inside = ((np.abs(self.X - center[0]) <= half[0]) &
                      (np.abs(self.Y - center[1]) <= half[1]) &
                      (np.abs(self.Z - center[2]) <= half[2]))
            active &= ~inside
        self.active = active

        self.scalar = np.asarray(initial.fields["transported_scalar"], dtype=float).copy()
        self.scalar[~active] = 0.0
        self.velocity = np.zeros(self.shape + (3,), dtype=float)

        # Finite-area emitter immediately above the floor. The source field is
        # a rate; a smooth radial profile avoids grid-aligned injection.
        radial2 = self.X * self.X + self.Y * self.Y
        source_profile = np.clip(1.0 - radial2 / (0.34 ** 2), 0.0, 1.0)
        vertical_profile = np.clip((0.34 - self.Z) / 0.24, 0.0, 1.0)
        self.scalar_source = 2.6 * source_profile * vertical_profile * active

        self.time = 0.0
        self.max_step = 1.0 / 60.0
        self.buoyancy = 2.35
        self.viscosity = 0.0015
        self.confinement = 0.16

        self.position = np.asarray(initial.position, dtype=float).copy()
        self.particle_velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = None if initial.mass is None else np.asarray(initial.mass).copy()
        self.body_id = None if initial.body_id is None else np.asarray(initial.body_id).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self.fixed_scalars = dict(initial.scalars)

        # Six enclosure faces and six faces of each obstacle. All are static.
        axis_normals = np.array([
            [-1., 0., 0.], [1., 0., 0.], [0., -1., 0.],
            [0., 1., 0.], [0., 0., -1.], [0., 0., 1.],
        ])
        self.boundary_normal = np.tile(axis_normals, (3, 1))
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

        # Masks for fluid cells adjoining either a box or the domain exterior.
        self.solid_neighbor = []
        for axis in range(3):
            pair = []
            for direction in (-1, 1):
                neighbor_active = self._shift(active, axis, direction, False)
                pair.append(active & ~neighbor_active)
            self.solid_neighbor.append(pair)
        self._apply_boundaries(self.velocity)

    @staticmethod
    def _shift(array, axis: int, direction: int, fill):
        """Return the adjacent-cell value with a specified exterior value."""
        out = np.empty_like(array)
        src = [slice(None)] * array.ndim
        dst = [slice(None)] * array.ndim
        if direction > 0:
            src[axis] = slice(1, None)
            dst[axis] = slice(None, -1)
            out[tuple(dst)] = array[tuple(src)]
            edge = [slice(None)] * array.ndim
            edge[axis] = -1
        else:
            src[axis] = slice(None, -1)
            dst[axis] = slice(1, None)
            out[tuple(dst)] = array[tuple(src)]
            edge = [slice(None)] * array.ndim
            edge[axis] = 0
        out[tuple(edge)] = fill
        return out

    def _sample_grid(self, field, coords):
        """Trilinear interpolation at fractional grid-index coordinates."""
        c = np.empty_like(coords, dtype=float)
        for a, n in enumerate(self.shape):
            c[..., a] = np.clip(coords[..., a], 0.0, n - 1.000001)
        i0 = np.floor(c).astype(np.int32)
        f = c - i0
        i1 = np.minimum(i0 + 1, np.asarray(self.shape) - 1)

        result = np.zeros(c.shape[:-1] + field.shape[3:], dtype=float)
        for bx in (0, 1):
            ix = i1[..., 0] if bx else i0[..., 0]
            wx = f[..., 0] if bx else 1.0 - f[..., 0]
            for by in (0, 1):
                iy = i1[..., 1] if by else i0[..., 1]
                wy = f[..., 1] if by else 1.0 - f[..., 1]
                for bz in (0, 1):
                    iz = i1[..., 2] if bz else i0[..., 2]
                    wz = f[..., 2] if bz else 1.0 - f[..., 2]
                    weight = wx * wy * wz
                    value = field[ix, iy, iz]
                    if value.ndim > weight.ndim:
                        weight = weight[..., None]
                    result += weight * value
        return result

    def _advect(self, field, dt: float):
        displacement = self.velocity * (dt / self.spacing3)
        back = self.index_grid - displacement
        nearest = np.rint(np.clip(back, 0, np.asarray(self.shape) - 1)).astype(np.int32)
        endpoint_active = self.active[nearest[..., 0], nearest[..., 1], nearest[..., 2]]
        back = np.where(endpoint_active[..., None], back, self.index_grid)
        out = self._sample_grid(field, back)
        out[~self.active] = 0.0
        return out

    def _laplacian(self, field):
        lap = np.zeros_like(field)
        for axis, h in enumerate(self.spacing3):
            plus = self._shift(field, axis, 1, 0.0)
            minus = self._shift(field, axis, -1, 0.0)
            lap += (plus + minus - 2.0 * field) / (h * h)
        return lap

    def _apply_boundaries(self, velocity):
        velocity[~self.active] = 0.0
        wall_shell = np.zeros(self.shape, dtype=bool)
        for axis in range(3):
            hit = self.solid_neighbor[axis][0] | self.solid_neighbor[axis][1]
            velocity[..., axis][hit] = 0.0
            wall_shell |= hit
        # Tangential damping represents the viscous no-slip boundary layer.
        velocity[wall_shell] *= 0.94
        return velocity

    def _curl(self, velocity):
        du_dx, du_dy, du_dz = np.gradient(velocity[..., 0], *self.spacing3,
                                           edge_order=2)
        dv_dx, dv_dy, dv_dz = np.gradient(velocity[..., 1], *self.spacing3,
                                           edge_order=2)
        dw_dx, dw_dy, dw_dz = np.gradient(velocity[..., 2], *self.spacing3,
                                           edge_order=2)
        del du_dx, dv_dy, dw_dz
        return np.stack((dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy), axis=-1)

    def _project(self, dt: float):
        u = self.velocity
        divergence = np.zeros(self.shape, dtype=float)
        for axis, h in enumerate(self.spacing3):
            comp = u[..., axis]
            plus = self._shift(comp, axis, 1, 0.0)
            minus = self._shift(comp, axis, -1, 0.0)
            divergence += (plus - minus) / (2.0 * h)
        divergence[~self.active] = 0.0

        pressure = np.zeros(self.shape, dtype=float)
        rhs = divergence / dt
        denominator = np.zeros(self.shape, dtype=float)
        neighbor_masks = []
        for axis, h in enumerate(self.spacing3):
            coeff = 1.0 / (h * h)
            minus_ok = self._shift(self.active, axis, -1, False)
            plus_ok = self._shift(self.active, axis, 1, False)
            neighbor_masks.append((minus_ok, plus_ok, coeff))
            denominator += coeff * (minus_ok + plus_ok)
        denominator = np.maximum(denominator, 1.0)

        for _ in range(28):
            neighbor_sum = np.zeros(self.shape, dtype=float)
            for axis, (minus_ok, plus_ok, coeff) in enumerate(neighbor_masks):
                pm = self._shift(pressure, axis, -1, 0.0)
                pp = self._shift(pressure, axis, 1, 0.0)
                neighbor_sum += coeff * (pm * minus_ok + pp * plus_ok)
            candidate = (neighbor_sum - rhs) / denominator
            pressure = np.where(self.active, 0.72 * candidate + 0.28 * pressure, 0.0)

        for axis, h in enumerate(self.spacing3):
            pm = self._shift(pressure, axis, -1, 0.0)
            pp = self._shift(pressure, axis, 1, 0.0)
            minus_ok, plus_ok, _ = neighbor_masks[axis]
            pm = np.where(minus_ok, pm, pressure)
            pp = np.where(plus_ok, pp, pressure)
            u[..., axis] -= dt * (pp - pm) / (2.0 * h)
        self._apply_boundaries(u)

    def _advance_particles(self, dt: float):
        coords = (self.position - self.lo) / self.spacing3
        pv = self._sample_grid(self.velocity, coords)
        proposed = self.position + dt * pv
        proposed = np.minimum(np.maximum(proposed, self.lo), self.hi)
        for center, half in self.obstacles:
            rel = proposed - center
            inside = np.all(np.abs(rel) < half, axis=1)
            if np.any(inside):
                local = rel[inside]
                clearance = half - np.abs(local)
                face_axis = np.argmin(clearance, axis=1)
                rows = np.arange(len(local))
                signs = np.where(local[rows, face_axis] >= 0.0, 1.0, -1.0)
                local[rows, face_axis] = signs * half[face_axis]
                proposed[inside] = center + local
                selected = pv[inside]
                selected[rows, face_axis] = 0.0
                pv[inside] = selected
        self.position = proposed
        self.particle_velocity = pv

    def _step(self, dt: float):
        self.velocity = self._advect(self.velocity, dt)
        self.scalar = self._advect(self.scalar, dt)
        self.velocity += self.viscosity * dt * self._laplacian(self.velocity)
        self.scalar += dt * self.scalar_source
        self.scalar *= np.exp(-0.018 * dt)
        np.clip(self.scalar, 0.0, 1.5, out=self.scalar)
        self.scalar[~self.active] = 0.0
        self.velocity[..., 2] += dt * self.buoyancy * self.scalar

        # Vorticity confinement counteracts excessive numerical diffusion and
        # keeps the alternating obstacle wakes resolved on this coarse grid.
        curl = self._curl(self.velocity)
        magnitude = np.linalg.norm(curl, axis=-1)
        grad_mag = np.stack(np.gradient(magnitude, *self.spacing3,
                                        edge_order=2), axis=-1)
        norm = np.linalg.norm(grad_mag, axis=-1, keepdims=True)
        normal = grad_mag / np.maximum(norm, 1.0e-9)
        self.velocity += (dt * self.confinement * float(np.min(self.spacing3)) *
                          np.cross(normal, curl))

        # The emitter is a stationary inflow boundary supplying upward momentum.
        emitter = self.scalar_source > 0.35
        self.velocity[..., 2][emitter] = np.maximum(
            self.velocity[..., 2][emitter], 0.72)
        self.velocity[..., 0][emitter] += (
            dt * 0.055 * np.tanh(self.X[emitter] / 0.12))

        self._apply_boundaries(self.velocity)
        self._project(dt)
        self._advance_particles(dt)
        self.time += dt

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time + 1.0e-12 < target:
            dt = min(self.max_step, target - self.time)
            speed = float(np.max(np.abs(self.velocity)))
            if speed > 1.0e-8:
                dt = min(dt, 0.72 * float(np.min(self.spacing3)) / speed)
            self._step(dt)
        self.time = target

        fields = {
            "active_mask": self.active.copy(),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": np.zeros_like(self.boundary_target_velocity),
            "scalar_source": self.scalar_source.copy(),
            "transported_scalar": self.scalar.copy(),
            "velocity_field": self.velocity.copy(),
        }
        scalars = {
            "characteristic_length": float(self.fixed_scalars.get(
                "characteristic_length", 0.76)),
            "grid_spacing": float(self.fixed_scalars.get("grid_spacing", 0.18)),
            "reference_velocity": float(self.fixed_scalars.get(
                "reference_velocity", 1.0)),
        }
        frame = StateFrame(
            target, self.position.copy(), self.particle_velocity.copy(),
            mass=None if self.mass is None else self.mass.copy(),
            body_id=None if self.body_id is None else self.body_id.copy(),
            attributes={k: v.copy() for k, v in self.attributes.items()},
            fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "3d_semi_lagrangian_incompressible_smoke",
            "grid_shape": self.shape,
            "simulation_time": float(self.time),
            "active_cells": int(np.count_nonzero(self.active)),
        }
