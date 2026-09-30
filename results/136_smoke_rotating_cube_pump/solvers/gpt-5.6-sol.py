"""Deterministic Eulerian smoke solver for the rotating-cube scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.shape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.lo = np.array([-3.0, -1.2, 0.0])
        self.hi = np.array([3.0, 1.2, 3.8])
        self.d = (self.hi - self.lo) / np.asarray(self.shape, float)
        axes = [self.lo[i] + (np.arange(self.shape[i]) + .5) * self.d[i] for i in range(3)]
        self.X, self.Y, self.Z = np.meshgrid(*axes, indexing="ij")
        self.grid_pos = np.column_stack((self.X.ravel(), self.Y.ravel(), self.Z.ravel()))
        initial = scene.initial_state()
        self.scalar = np.asarray(initial.fields["transported_scalar"], float).copy()
        self.velocity = np.zeros(self.shape + (3,), float)
        self.particle_position = np.asarray(initial.position, float).copy()
        self.particle_velocity = np.zeros_like(self.particle_position)
        self.mass = np.asarray(initial.mass, float).copy()
        self.body_id = np.asarray(initial.body_id, np.int32).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self.spacing = float(cfg["sampling"]["spacing"])
        self.characteristic_length = float(initial.scalars.get("characteristic_length", 1.0))
        self.current_time = 0.0
        self.internal_dt = min(.018, float(cfg.get("sample_dt", 1.0)) * .6)
        self.omega = 2.65
        self.cube_center = np.array([0.0, 0.0, 1.55])
        self.cube_half = np.array([.45, .45, .45])
        self.source = 3.0 * (((self.X * self.X + self.Y * self.Y) <= .34 ** 2) & (self.Z <= .22)).astype(float)
        self.active = np.ones(self.shape, dtype=bool)

    def _cube_geometry(self, time):
        a = self.omega * time
        ca, sa = np.cos(a), np.sin(a)
        rx = ca * self.X + sa * self.Y
        ry = -sa * self.X + ca * self.Y
        rz = self.Z - self.cube_center[2]
        inside = (np.abs(rx) <= .45) & (np.abs(ry) <= .45) & (np.abs(rz) <= .45)
        dist = np.maximum.reduce((np.abs(rx) - .45, np.abs(ry) - .45, np.abs(rz) - .45))
        return inside, dist, a

    def _sample(self, field, points):
        q = np.asarray(points, float)
        a = (q - self.lo) / self.d - .5
        i0 = np.floor(a).astype(np.int64)
        f = a - i0
        i0 = np.clip(i0, 0, np.asarray(self.shape) - 1)
        i1 = np.minimum(i0 + 1, np.asarray(self.shape) - 1)
        i, j, k = i0[:, 0], i0[:, 1], i0[:, 2]
        ii, jj, kk = i1[:, 0], i1[:, 1], i1[:, 2]
        wx, wy, wz = f[:, 0], f[:, 1], f[:, 2]
        c000, c100 = field[i, j, k], field[ii, j, k]
        c010, c110 = field[i, jj, k], field[ii, jj, k]
        c001, c101 = field[i, j, kk], field[ii, j, kk]
        c011, c111 = field[i, jj, kk], field[ii, jj, kk]
        c00 = c000 * (1 - wx) + c100 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        return c0 * (1 - wz)[..., None] + c1 * wz[..., None] if field.ndim == 4 else c0 * (1 - wz) + c1 * wz

    def _laplacian(self, a):
        p = np.pad(a, ((1, 1),) * 3 + ((0, 0),) * (a.ndim - 3), mode="edge")
        lap = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] - 2 * a) / self.d[0] ** 2
        lap += (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] - 2 * a) / self.d[1] ** 2
        lap += (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 2 * a) / self.d[2] ** 2
        return lap

    def _project(self, vel, dt, solid):
        div = np.gradient(vel[..., 0], self.d[0], axis=0) + np.gradient(vel[..., 1], self.d[1], axis=1) + np.gradient(vel[..., 2], self.d[2], axis=2)
        pfield = np.zeros(self.shape)
        inv = 1.0 / (2 / self.d[0] ** 2 + 2 / self.d[1] ** 2 + 2 / self.d[2] ** 2)
        rhs = div / max(dt, 1e-6)
        for _ in range(8):
            p = np.pad(pfield, 1, mode="edge")
            pfield = ((p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) / self.d[0] ** 2 + (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) / self.d[1] ** 2 + (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / self.d[2] ** 2 - rhs) * inv
            pfield[solid] = 0
        grad = np.stack(np.gradient(pfield, self.d[0], self.d[1], self.d[2], edge_order=1), axis=-1)
        return vel - dt * grad

    def _step(self, dt, time):
        solid, distance, _ = self._cube_geometry(time)
        back = self.grid_pos - dt * self.velocity.reshape(-1, 3)
        adv_v = self._sample(self.velocity, back).reshape(self.shape + (3,))
        adv_s = self._sample(self.scalar, back).reshape(self.shape)
        buoy = np.stack((np.zeros(self.shape), np.zeros(self.shape), 5.8 * self.scalar), axis=-1)
        vel = adv_v + dt * buoy + dt * .075 * self._laplacian(adv_v)
        adv_s += dt * (self.source - .012 * adv_s) + dt * .018 * self._laplacian(adv_s)
        rx, ry = self.X, self.Y
        rigid = np.stack((-self.omega * ry, self.omega * rx, np.zeros(self.shape)), axis=-1)
        shell = np.clip(1 - np.maximum(distance, 0) / .24, 0, 1)
        vel += dt * (2.8 * shell[..., None]) * (rigid - vel)
        vel[solid] = rigid[solid]
        vel = self._project(vel, dt, solid)
        vel[solid] = rigid[solid]
        vel[0, :, :, 0] = vel[-1, :, :, 0] = 0
        vel[:, 0, :, 1] = vel[:, -1, :, 1] = 0
        vel[:, :, 0, 2] = np.maximum(vel[:, :, 0, 2], 0)
        vel[:, :, -1, 2] = np.minimum(vel[:, :, -1, 2], 0)
        adv_s[solid] = 0
        adv_s = np.maximum(adv_s, 0)
        adv_s[0] = adv_s[1]; adv_s[-1] = adv_s[-2]
        adv_s[:, 0] = adv_s[:, 1]; adv_s[:, -1] = adv_s[:, -2]
        adv_s[:, :, 0] = 0; adv_s[:, :, -1] = adv_s[:, :, -2]
        self.velocity = np.nan_to_num(np.clip(vel, -8, 8))
        self.scalar = np.nan_to_num(np.clip(adv_s, 0, 8))
        self.particle_velocity = self._sample(self.velocity, self.particle_position)
        self.particle_position += dt * self.particle_velocity
        self.particle_position = np.clip(self.particle_position, self.lo + 1e-4, self.hi - 1e-4)

    def _boundary_fields(self, time):
        _, _, angle = self._cube_geometry(time)
        local = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        ca, sa = np.cos(angle), np.sin(angle)
        normals_cube = local.copy()
        normals_cube[:, 0] = ca * local[:, 0] - sa * local[:, 1]
        normals_cube[:, 1] = sa * local[:, 0] + ca * local[:, 1]
        cube_points = self.cube_center + normals_cube * .45
        cube_target = np.cross(np.tile([0., 0., self.omega], (6, 1)), cube_points - self.cube_center)
        wall_normals = local.copy()
        wall_points = np.array([[self.hi[0] - self.d[0] / 2, 0, 1.5], [self.lo[0] + self.d[0] / 2, 0, 1.5], [0, self.hi[1] - self.d[1] / 2, 1.5], [0, self.lo[1] + self.d[1] / 2, 1.5], [0, 0, self.hi[2] - self.d[2] / 2], [0, 0, self.lo[2] + self.d[2] / 2]])
        points = np.vstack((wall_points, cube_points))
        normals = np.vstack((wall_normals, normals_cube))
        target = np.vstack((np.zeros((6, 3)), cube_target))
        sampled = self._sample(self.velocity, points)
        sampled[:6] = 0; sampled[6:] = cube_target
        return normals, target, sampled

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1e-9:
            raise ValueError("advance_to requires finite nondecreasing time")
        while self.current_time < target - 1e-10:
            dt = min(self.internal_dt, target - self.current_time)
            self._step(dt, self.current_time + dt)
            self.current_time += dt
        normals, target_velocity, boundary_velocity = self._boundary_fields(self.current_time)
        frame = StateFrame(self.current_time, self.particle_position.copy(), self.particle_velocity.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(), attributes=self.attributes, fields={"active_mask": self.active.copy(), "boundary_normal": normals, "boundary_target_velocity": target_velocity, "boundary_velocity": boundary_velocity, "scalar_source": self.source.copy(), "transported_scalar": self.scalar.copy(), "velocity_field": self.velocity.copy()}, scalars={"characteristic_length": self.characteristic_length, "grid_spacing": self.spacing, "reference_velocity": 1.0})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "semi_lagrangian_buoyant_smoke", "grid_shape": list(self.shape), "internal_dt": float(self.internal_dt), "max_scalar": float(np.max(self.scalar)), "max_speed": float(np.max(np.linalg.norm(self.velocity, axis=-1)))}
