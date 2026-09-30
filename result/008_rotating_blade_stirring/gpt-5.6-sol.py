"""Self-contained rotating-blade liquid solver.

The implementation uses an Eulerian velocity grid with semi-Lagrangian
advection and pressure projection, while liquid samples are advected by the
grid and corrected against the tank and moving blade.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self._time = 0.0
        self._position = None
        self._velocity = None
        self._body_id = None
        self._mass = None
        self._orientation = None
        self._angular_velocity = None
        self._u = None
        self._active = None
        self._grid = None
        self._axes = None
        self._h = 0.05
        self._half = 0.8
        self._depth = 0.3
        self._rho = 1000.0
        self._nu = 1.0e-6
        self._spacing = 0.025
        self._wall_positions = None
        self._wall_normals = None
        self._last_substeps = 0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        initial = scene.initial_state()
        self._time = 0.0
        self._position = np.asarray(initial.position, dtype=float).copy()
        self._velocity = np.asarray(initial.velocity, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int64).copy()
        self._mass = np.asarray(initial.attributes["mass_quantity"], dtype=float).copy()
        self._orientation = np.asarray(initial.orientation, dtype=float).copy()
        self._angular_velocity = np.asarray(initial.angular_velocity, dtype=float).copy()
        self._u = np.asarray(initial.fields["velocity_field"], dtype=float).copy()
        self._active = np.asarray(initial.fields["active_mask"], dtype=float).copy()
        self._h = float(initial.scalars["grid_spacing"])
        self._half = float(scene.config["tank"]["interior_half_width"])
        self._depth = float(scene.config["liquid"]["depth"])
        self._rho = float(scene.config["liquid"]["density"])
        self._nu = float(scene.config["liquid"]["kinematic_viscosity"])
        self._spacing = float(scene.config["liquid"]["particle_spacing"])
        self._wall_positions, self._wall_normals = scene.initial_tank_samples()
        domain = scene.config["domain"]
        axes = [
            float(lo) + (np.arange(int(n)) + 0.5) * self._h
            for lo, n in zip(domain["bounds_min"], domain["reference_grid"], strict=True)
        ]
        self._axes = axes
        self._grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    @staticmethod
    def _trilerp(values: np.ndarray, points: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
        """Trilinear interpolation with clamped coordinates."""
        shape = values.shape[:3]
        out = np.zeros((len(points), values.shape[3]), dtype=float)
        q = np.asarray(points, dtype=float)
        idx0, frac = [], []
        for d, axis in enumerate(axes):
            t = (q[:, d] - axis[0]) / (axis[1] - axis[0])
            t = np.clip(t, 0.0, shape[d] - 1.000001)
            i = np.floor(t).astype(np.int64)
            idx0.append(i)
            frac.append(t - i)
        i, j, k = idx0
        fx, fy, fz = frac
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            ii = np.minimum(i + di, shape[0] - 1)
            for dj, wy in ((0, 1.0 - fy), (1, fy)):
                jj = np.minimum(j + dj, shape[1] - 1)
                for dk, wz in ((0, 1.0 - fz), (1, fz)):
                    kk = np.minimum(k + dk, shape[2] - 1)
                    out += (wx * wy * wz)[:, None] * values[ii, jj, kk]
        return out

    def _grid_step(self, dt: float, time: float) -> None:
        """Advance velocity using advection, body forces, and projection."""
        old = self._u
        back = self._grid - old * dt
        u = self._trilerp(old, back.reshape(-1, 3), self._axes).reshape(old.shape)
        u += np.asarray(self.scene.config["gravity"], dtype=float) * dt

        pose = self.scene.blade_kinematics(time + 0.5 * dt)
        rel = self._grid - pose["position"]
        target = np.cross(np.broadcast_to(pose["angular_velocity"], rel.shape), rel)
        sd = self.scene.blade_signed_distance(
            self._grid.reshape(-1, 3), time + 0.5 * dt
        ).reshape(u.shape[:3])
        # Diffuse immersed-boundary forcing transfers blade momentum to liquid.
        influence = np.exp(-np.square(np.maximum(sd, 0.0) / 0.10))
        influence *= (sd < 0.14) * (self._grid[..., 2] < self._depth + 0.18)
        blend = np.clip(dt * 15.0 * influence, 0.0, 0.65)[..., None]
        u = u * (1.0 - blend) + target * blend

        lap = (
            np.roll(u, 1, 0) + np.roll(u, -1, 0)
            + np.roll(u, 1, 1) + np.roll(u, -1, 1)
            + np.roll(u, 1, 2) + np.roll(u, -1, 2) - 6.0 * u
        ) / (self._h * self._h)
        u += self._nu * dt * lap
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0

        active = self._active > 0.5
        div = np.zeros(active.shape, dtype=float)
        div[1:-1, 1:-1, 1:-1] = (
            u[2:, 1:-1, 1:-1, 0] - u[:-2, 1:-1, 1:-1, 0]
            + u[1:-1, 2:, 1:-1, 1] - u[1:-1, :-2, 1:-1, 1]
            + u[1:-1, 1:-1, 2:, 2] - u[1:-1, 1:-1, :-2, 2]
        ) / (2.0 * self._h)
        pressure = np.zeros_like(div)
        for _ in range(8):
            neighbor_sum = (
                np.roll(pressure, 1, 0) + np.roll(pressure, -1, 0)
                + np.roll(pressure, 1, 1) + np.roll(pressure, -1, 1)
                + np.roll(pressure, 1, 2) + np.roll(pressure, -1, 2)
            )
            updated = np.zeros_like(pressure)
            updated[active] = (
                neighbor_sum[active] - div[active] * self._h * self._h / max(dt, 1.0e-6)
            ) / 6.0
            pressure = updated
        grad = np.zeros_like(u)
        grad[1:-1, 1:-1, 1:-1, 0] = (
            pressure[2:, 1:-1, 1:-1] - pressure[:-2, 1:-1, 1:-1]
        ) / (2.0 * self._h)
        grad[1:-1, 1:-1, 1:-1, 1] = (
            pressure[1:-1, 2:, 1:-1] - pressure[1:-1, :-2, 1:-1]
        ) / (2.0 * self._h)
        grad[1:-1, 1:-1, 1:-1, 2] = (
            pressure[1:-1, 1:-1, 2:] - pressure[1:-1, 1:-1, :-2]
        ) / (2.0 * self._h)
        u -= dt * grad
        u[~active] *= 0.995
        self._u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    def _particle_step(self, dt: float, time: float) -> None:
        fluid = self._body_id == 0
        p = self._position[fluid]
        v = self._velocity[fluid]
        grid_v = self._trilerp(self._u, p, self._axes)
        v += (grid_v - v) * np.clip(dt / 0.075, 0.0, 1.0)
        v += np.asarray(self.scene.config["gravity"], dtype=float) * (0.10 * dt)
        p += v * dt

        pose = self.scene.blade_kinematics(time)
        center = pose["position"]
        rotation = pose["rotation_matrix"]
        local = (p - center) @ rotation
        half = np.asarray(self.scene.config["blade"]["half_extents"], dtype=float)
        penetration = half - np.abs(local)
        inside = np.all(penetration >= 0.0, axis=1)
        near = self.scene.blade_signed_distance(p, time) < 0.5 * self._spacing
        hit = inside | near
        if np.any(hit):
            local_hit = local[hit]
            depths = half - np.abs(local_hit)
            axis = np.argmin(depths, axis=1)
            signs = np.where(local_hit[np.arange(len(local_hit)), axis] >= 0.0, 1.0, -1.0)
            normal_local = np.zeros_like(local_hit)
            normal_local[np.arange(len(local_hit)), axis] = signs
            push = np.maximum(depths[np.arange(len(local_hit)), axis], 0.0) + 1.0e-4
            local_hit += normal_local * push[:, None]
            p[hit] = local_hit @ rotation.T + center
            normal = normal_local @ rotation.T
            target = np.cross(
                np.broadcast_to(pose["angular_velocity"], p[hit].shape), p[hit] - center
            )
            normal_speed = np.sum(v[hit] * normal, axis=1)
            v[hit] += np.maximum(-normal_speed, 0.0)[:, None] * normal
            v[hit] += 0.45 * (target - v[hit])

        lim = self._half - 0.5 * self._spacing
        for axis in (0, 1):
            low = p[:, axis] < -lim
            high = p[:, axis] > lim
            p[low, axis] = -lim
            p[high, axis] = lim
            v[low | high, axis] = 0.0
        floor = p[:, 2] < 0.5 * self._spacing
        p[floor, 2] = 0.5 * self._spacing
        v[floor, 2] = np.maximum(v[floor, 2], 0.0)
        self._position[fluid] = p
        self._velocity[fluid] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def _surface_level_set(self) -> tuple[np.ndarray, np.ndarray]:
        p = self._position[self._body_id == 0]
        nx, ny, _ = self._u.shape[:3]
        surface = np.full((nx, ny), -np.inf, dtype=float)
        ix = np.floor((p[:, 0] + self._half) / self._h).astype(int)
        iy = np.floor((p[:, 1] + self._half) / self._h).astype(int)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.maximum.at(surface, (ix[valid], iy[valid]), p[valid, 2])
        surface[~np.isfinite(surface)] = self._depth
        for _ in range(2):
            neighbor_max = np.max(
                np.stack(
                    [surface, np.roll(surface, 1, 0), np.roll(surface, -1, 0),
                     np.roll(surface, 1, 1), np.roll(surface, -1, 1)], axis=0
                ), axis=0
            )
            surface = 0.9 * surface + 0.1 * neighbor_max
        return self._grid[..., 2] - surface[..., None], surface

    def _frame(self, time: float) -> StateFrame:
        pose = self.scene.blade_kinematics(time)
        blade_id = int(self.scene.config["blade"]["body_id"])
        blade = self._body_id == blade_id
        self._position[blade] = pose["position"]
        self._velocity[blade] = pose["velocity"]
        self._orientation[blade] = pose["orientation"]
        self._angular_velocity[blade] = pose["angular_velocity"]

        phi, surface = self._surface_level_set()
        blade_sd = self.scene.blade_signed_distance(
            self._grid.reshape(-1, 3), time
        ).reshape(phi.shape)
        active = (
            (phi < -self._h) & (self._grid[..., 2] > self._h)
            & (np.abs(self._grid[..., 0]) < self._half - self._h)
            & (np.abs(self._grid[..., 1]) < self._half - self._h)
            & (blade_sd > self._h)
        )
        self._active = active.astype(float)
        self._u[blade_sd < 0.0] = 0.0

        samples = self.scene.blade_surface_samples(time)
        # Keep the diagnostic tensor shape stable for trajectory storage.  These
        # are the blade patches below the nominal liquid line; local splashes
        # affect their measured velocity through the evolving grid state.
        wetted = samples["position"][:, 2] < self._depth
        blade_position = samples["position"][wetted]
        blade_velocity = self._trilerp(self._u, blade_position, self._axes)

        frame = StateFrame(
            time=float(time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            orientation=self._orientation.copy(),
            angular_velocity=self._angular_velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "particle_volume": self._mass / self._rho,
                "particle_radius": np.where(self._body_id == 0, 0.5 * self._spacing, 0.0),
            },
            fields={
                "velocity_field": self._u.copy(),
                "active_mask": active.astype(float),
                "liquid_level_set": phi,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "boundary_normal": self._wall_normals.copy(),
                "boundary_velocity": np.zeros_like(self._wall_positions),
                "boundary_target_velocity": np.zeros_like(self._wall_positions),
                "blade_boundary_normal": samples["normal"][wetted],
                "blade_boundary_velocity": blade_velocity,
                "blade_boundary_target_velocity": samples["target_velocity"][wetted],
                "grid_origin": np.asarray(self.scene.config["domain"]["bounds_min"], dtype=float),
            },
            scalars={
                "grid_spacing": self._h,
                "characteristic_length": float(2.0 * self.scene.config["blade"]["half_extents"][0]),
                "reference_velocity": float(
                    abs(self.scene.config["blade"]["angular_speed"])
                    * self.scene.config["blade"]["half_extents"][0]
                ),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1.0e-12:
            raise ValueError("requested time must be finite and nondecreasing")
        remaining = target - self._time
        self._last_substeps = 0
        while remaining > 1.0e-12:
            dt = min(0.01, remaining)
            mid = self._time + 0.5 * dt
            self._grid_step(dt, mid)
            self._particle_step(dt, mid)
            self._time += dt
            remaining = target - self._time
            self._last_substeps += 1
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        if self._body_id is None:
            return {"solver": "semi_lagrangian_grid_particle_blade_coupling"}
        fluid = self._body_id == 0
        return {
            "solver": "semi_lagrangian_grid_particle_blade_coupling",
            "particle_count": int(np.count_nonzero(fluid)),
            "substeps_last_advance": int(self._last_substeps),
            "escaped_particle_count": int(np.count_nonzero(self._position[fluid, 2] > 1.2)),
        }
