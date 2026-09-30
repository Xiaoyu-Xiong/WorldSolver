"""Deterministic finite-difference smoke solver for the benchmark scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Stable incompressible-flow solver on the scene's voxel grid."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        cfg = scene.config
        self.shape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.nx, self.ny, self.nz = self.shape
        self.x = np.linspace(-3.0, 3.0, self.nx)
        self.y = np.linspace(-1.2, 1.2, self.ny)
        self.z = np.linspace(0.0, 3.8, self.nz)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.dz = float(self.z[1] - self.z[0])
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        low = ((np.abs(self.X + 0.65) <= 0.38) & (np.abs(self.Y) <= 0.38) &
               (np.abs(self.Z - 1.25) <= 0.38))
        high = ((np.abs(self.X - 0.65) <= 0.42) & (np.abs(self.Y) <= 0.42) &
                (np.abs(self.Z - 2.15) <= 0.42))
        self.solid = (self.Z <= 0.5 * self.dz) | low | high
        self.fluid = ~self.solid

        self.velocity = np.zeros(self.shape + (3,), dtype=np.float64)
        self.scalar = np.asarray(initial.fields["transported_scalar"], dtype=np.float64).copy()
        self.scalar[self.solid] = 0.0
        radius2 = self.Y * self.Y + (self.Z - 1.0) ** 2
        self.inlet_profile = np.exp(-radius2 / (2.0 * 0.27 ** 2))
        self.inlet_patch = ((self.X <= self.x[1] + 1.0e-12) &
                            (radius2 <= 0.40 ** 2) & self.fluid)
        self.source = np.zeros(self.shape, dtype=np.float64)
        self.source[self.inlet_patch] = 3.2 * self.inlet_profile[self.inlet_patch]

        self._wx = 1.0 / (self.dx * self.dx)
        self._wy = 1.0 / (self.dy * self.dy)
        self._wz = 1.0 / (self.dz * self.dz)
        den = np.zeros(self.shape, dtype=np.float64)
        den[1:] += self._wx * self.fluid[:-1]
        den[:-1] += self._wx * self.fluid[1:]
        den[:, 1:] += self._wy * self.fluid[:, :-1]
        den[:, :-1] += self._wy * self.fluid[:, 1:]
        den[:, :, 1:] += self._wz * self.fluid[:, :, :-1]
        den[:, :, :-1] += self._wz * self.fluid[:, :, 1:]
        self._pressure_denom = np.maximum(den, 1.0)

        self.time = 0.0
        self.base_dt = 1.0 / 60.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.particle_velocity = np.zeros_like(self.position)
        self.inlet_particle_position = self.position.copy()
        self.mass = np.asarray(initial.mass).copy()
        self.body_id = np.asarray(initial.body_id).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self.characteristic_length = float(initial.scalars["characteristic_length"])
        self.grid_spacing = float(initial.scalars["grid_spacing"])
        self.reference_velocity = float(initial.scalars["reference_velocity"])
        self.boundary_normal = self._make_boundary_normals()
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)
        self.last_divergence = 0.0
        self.steps = 0
        self._enforce_velocity_boundaries(self.velocity, 0.0)

    def _make_boundary_normals(self) -> np.ndarray:
        normals = [[0.0, 0.0, 1.0]] * 63
        for _ in range(2):
            for normal in ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                           (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                           (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)):
                normals.extend([normal] * 16)
        return np.asarray(normals, dtype=np.float64)

    def _sample_grid(self, field, px, py, pz):
        fx = np.clip((px - self.x[0]) / self.dx, 0.0, self.nx - 1.000001)
        fy = np.clip((py - self.y[0]) / self.dy, 0.0, self.ny - 1.000001)
        fz = np.clip((pz - self.z[0]) / self.dz, 0.0, self.nz - 1.000001)
        i0, j0, k0 = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp), np.floor(fz).astype(np.intp)
        i1, j1, k1 = np.minimum(i0 + 1, self.nx - 1), np.minimum(j0 + 1, self.ny - 1), np.minimum(k0 + 1, self.nz - 1)
        ax, ay, az = fx - i0, fy - j0, fz - k0
        c00 = field[i0, j0, k0] * (1 - ax) + field[i1, j0, k0] * ax
        c10 = field[i0, j1, k0] * (1 - ax) + field[i1, j1, k0] * ax
        c01 = field[i0, j0, k1] * (1 - ax) + field[i1, j0, k1] * ax
        c11 = field[i0, j1, k1] * (1 - ax) + field[i1, j1, k1] * ax
        return (c00 * (1 - ay) + c10 * ay) * (1 - az) + (c01 * (1 - ay) + c11 * ay) * az

    def _advect(self, field, velocity, dt):
        result = self._sample_grid(field, self.X - dt * velocity[..., 0],
                                   self.Y - dt * velocity[..., 1], self.Z - dt * velocity[..., 2])
        result[self.solid] = 0.0
        return result

    @staticmethod
    def _laplacian(a, dx, dy, dz):
        lap = np.zeros_like(a)
        lap[1:-1] += (a[2:] - 2 * a[1:-1] + a[:-2]) / (dx * dx)
        lap[:, 1:-1] += (a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]) / (dy * dy)
        lap[:, :, 1:-1] += (a[:, :, 2:] - 2 * a[:, :, 1:-1] + a[:, :, :-2]) / (dz * dz)
        return lap

    def _curl(self, v):
        dwdy = np.gradient(v[..., 2], self.dy, axis=1, edge_order=2)
        dvdz = np.gradient(v[..., 1], self.dz, axis=2, edge_order=2)
        dudz = np.gradient(v[..., 0], self.dz, axis=2, edge_order=2)
        dwdx = np.gradient(v[..., 2], self.dx, axis=0, edge_order=2)
        dvdx = np.gradient(v[..., 1], self.dx, axis=0, edge_order=2)
        dudy = np.gradient(v[..., 0], self.dy, axis=1, edge_order=2)
        return np.stack((dwdy - dvdz, dudz - dwdx, dvdx - dudy), axis=-1)

    def _project(self, v, dt):
        div = (np.gradient(v[..., 0], self.dx, axis=0, edge_order=2) +
               np.gradient(v[..., 1], self.dy, axis=1, edge_order=2) +
               np.gradient(v[..., 2], self.dz, axis=2, edge_order=2))
        div[self.solid] = 0.0
        rhs = div / dt
        p = np.zeros(self.shape, dtype=np.float64)
        for _ in range(18):
            total = np.zeros_like(p)
            total[1:] += self._wx * p[:-1] * self.fluid[:-1]
            total[:-1] += self._wx * p[1:] * self.fluid[1:]
            total[:, 1:] += self._wy * p[:, :-1] * self.fluid[:, :-1]
            total[:, :-1] += self._wy * p[:, 1:] * self.fluid[:, 1:]
            total[:, :, 1:] += self._wz * p[:, :, :-1] * self.fluid[:, :, :-1]
            total[:, :, :-1] += self._wz * p[:, :, 1:] * self.fluid[:, :, 1:]
            p = (total - rhs) / self._pressure_denom
            p[self.solid] = 0.0
            p[0] = 0.0
            p[-1] = 0.0
        v[..., 0] -= dt * np.gradient(p, self.dx, axis=0, edge_order=2)
        v[..., 1] -= dt * np.gradient(p, self.dy, axis=1, edge_order=2)
        v[..., 2] -= dt * np.gradient(p, self.dz, axis=2, edge_order=2)
        self.last_divergence = float(np.sqrt(np.mean(div[self.fluid] ** 2)))
        return v

    def _enforce_velocity_boundaries(self, v, time):
        v[self.solid] = 0.0
        adjacent = np.zeros(self.shape, dtype=bool)
        adjacent[1:] |= self.solid[:-1]; adjacent[:-1] |= self.solid[1:]
        adjacent[:, 1:] |= self.solid[:, :-1]; adjacent[:, :-1] |= self.solid[:, 1:]
        adjacent[:, :, 1:] |= self.solid[:, :, :-1]; adjacent[:, :, :-1] |= self.solid[:, :, 1:]
        v[adjacent & self.fluid] *= 0.18
        patch = self.Y[0] ** 2 + (self.Z[0] - 1.0) ** 2 <= 0.40 ** 2
        phase = 2 * np.pi * 0.72 * time
        profile = self.inlet_profile[0]
        u = 0.20 + 1.25 * profile
        vv = 0.075 * profile * np.sin(phase + 2.1 * (self.Z[0] - 1.0))
        ww = 0.060 * profile * np.sin(0.63 * phase + 2.7 * self.Y[0])
        for i in (0, 1):
            v[i, ..., 0] = np.where(patch, u, 0.20)
            v[i, ..., 1] = np.where(patch, vv, 0.0)
            v[i, ..., 2] = np.where(patch, ww, 0.0)
        v[-1] = v[-2]
        v[:, 0, 1] = np.minimum(v[:, 1, 1], 0.0)
        v[:, -1, 1] = np.maximum(v[:, -2, 1], 0.0)
        v[:, :, -1, 2] = np.maximum(v[:, :, -2, 2], 0.0)
        v[self.solid] = 0.0

    def _step_particles(self, dt):
        p = self.position
        vel0 = np.column_stack([self._sample_grid(self.velocity[..., d], p[:, 0], p[:, 1], p[:, 2]) for d in range(3)])
        mid = p + 0.5 * dt * vel0
        mid[:, 0] = np.clip(mid[:, 0], self.x[0], self.x[-1])
        mid[:, 1] = np.clip(mid[:, 1], self.y[0], self.y[-1])
        mid[:, 2] = np.clip(mid[:, 2], self.z[0] + self.dz, self.z[-1])
        vel1 = np.column_stack([self._sample_grid(self.velocity[..., d], mid[:, 0], mid[:, 1], mid[:, 2]) for d in range(3)])
        p += dt * vel1
        exited = p[:, 0] >= self.x[-1] - 0.02
        if np.any(exited):
            p[exited] = self.inlet_particle_position[exited]
        p[:, 0] = np.clip(p[:, 0], self.x[0] + 0.01, self.x[-1] - 0.01)
        p[:, 1] = np.clip(p[:, 1], self.y[0] + 0.01, self.y[-1] - 0.01)
        p[:, 2] = np.clip(p[:, 2], self.z[0] + self.dz, self.z[-1] - 0.01)
        self.particle_velocity = vel1

    def _step(self, dt):
        old_v = self.velocity
        v = np.empty_like(old_v)
        for d in range(3):
            v[..., d] = self._advect(old_v[..., d], old_v, dt)
        viscosity = 0.0025
        for d in range(3):
            v[..., d] += dt * viscosity * self._laplacian(v[..., d], self.dx, self.dy, self.dz)
        v[..., 2] += dt * 0.16 * self.scalar
        omega = self._curl(v)
        magnitude = np.sqrt(np.sum(omega * omega, axis=-1))
        grad_mag = np.stack(np.gradient(magnitude, self.dx, self.dy, self.dz, edge_order=2), axis=-1)
        direction = grad_mag / (np.sqrt(np.sum(grad_mag * grad_mag, axis=-1))[..., None] + 1.0e-10)
        v += dt * (0.26 * min(self.dx, self.dy, self.dz)) * np.cross(direction, omega)
        self._enforce_velocity_boundaries(v, self.time + dt)
        self.velocity = self._project(v, dt)
        self._enforce_velocity_boundaries(self.velocity, self.time + dt)
        np.clip(self.velocity, -4.0, 4.0, out=self.velocity)
        scalar = self._advect(self.scalar, self.velocity, dt)
        scalar += dt * 0.0015 * self._laplacian(scalar, self.dx, self.dy, self.dz)
        scalar += dt * self.source
        inlet_value = 1.15 * self.inlet_profile[:2]
        inlet_patch = self.inlet_patch[:2]
        scalar[:2][inlet_patch] = np.maximum(scalar[:2][inlet_patch], inlet_value[inlet_patch])
        scalar[self.solid] = 0.0
        self.scalar = np.clip(scalar, 0.0, 1.5)
        self._step_particles(dt)
        self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-9:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.base_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        fields = {
            "active_mask": self.fluid.copy(),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": np.zeros_like(self.boundary_normal),
            "scalar_source": self.source.copy(),
            "transported_scalar": self.scalar.copy(),
            "velocity_field": self.velocity.copy(),
            "cumulative_mass_source": np.zeros(len(self.position), dtype=np.float64),
        }
        frame = StateFrame(target, self.position.copy(), self.particle_velocity.copy(),
                           mass=self.mass.copy(), body_id=self.body_id.copy(),
                           attributes={k: v.copy() for k, v in self.attributes.items()},
                           fields=fields,
                           scalars={"contact_length_scale": self.grid_spacing,
                                    "grid_spacing": self.grid_spacing,
                                    "characteristic_length": self.characteristic_length,
                                    "reference_velocity": self.reference_velocity})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "semi_lagrangian_projection_3d", "steps": int(self.steps),
                "time": float(self.time), "rms_preprojection_divergence": float(self.last_divergence),
                "max_velocity": float(np.max(np.linalg.norm(self.velocity, axis=-1))),
                "smoke_mass_grid": float(np.sum(self.scalar) * self.dx * self.dy * self.dz)}
