"""Deterministic two-dimensional free-surface dam-break solver.

The liquid is represented by Lagrangian samples and advanced with a hybrid
PIC/FLIP method on a staggered Eulerian grid. Each substep transfers particle
momentum to the grid, applies gravity, projects the liquid velocity to be
divergence free, transfers it back, and advects the particles.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A stable 2-D particle-in-cell incompressible liquid solver."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()

        # Inner surfaces of the frozen tank. Only x and z are simulated.
        self.xmin = -3.0
        self.zmin = 0.05
        self.dx = 0.075
        self.nx = 80
        self.nz = 48
        self.xmax = self.xmin + self.nx * self.dx
        self.zmax = self.zmin + self.nz * self.dx
        self.rho = 1000.0
        self.gravity = 9.81
        self.max_dt = 0.005
        self.pic_fraction = 0.12
        self.particle_radius = 0.025
        self.time = 0.0
        self.step_count = 0
        self.pressure_iterations = 42

        self._u = np.zeros((self.nx + 1, self.nz), dtype=np.float64)
        self._w = np.zeros((self.nx, self.nz + 1), dtype=np.float64)
        self._fluid = np.zeros((self.nx, self.nz), dtype=bool)

        normals = [np.tile((1.0, 0.0), (self.nz, 1))]
        normals.append(np.tile((-1.0, 0.0), (self.nz, 1)))
        normals.append(np.tile((0.0, 1.0), (self.nx, 1)))
        self.boundary_normal = np.concatenate(normals, axis=0)
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)
        self.boundary_velocity = np.zeros_like(self.boundary_normal)

        self.seed = int(seed)
        self._update_diagnostic_grid()

    def _scatter(self, values: np.ndarray, ox: float, oz: float,
                 shape: tuple[int, int]):
        """Bilinearly scatter particle values to an offset regular grid."""
        gx = (self.position[:, 0] - self.xmin) / self.dx - ox
        gz = (self.position[:, 2] - self.zmin) / self.dx - oz
        i0 = np.floor(gx).astype(np.int64)
        k0 = np.floor(gz).astype(np.int64)
        fx = gx - i0
        fz = gz - k0
        grid = np.zeros(shape, dtype=np.float64)
        weight = np.zeros(shape, dtype=np.float64)
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            ii = i0 + di
            for dk, wz in ((0, 1.0 - fz), (1, fz)):
                kk = k0 + dk
                ww = wx * wz
                valid = ((ii >= 0) & (ii < shape[0]) &
                         (kk >= 0) & (kk < shape[1]))
                np.add.at(weight, (ii[valid], kk[valid]), ww[valid])
                np.add.at(grid, (ii[valid], kk[valid]),
                          ww[valid] * values[valid])
        occupied = weight > 1.0e-12
        grid[occupied] /= weight[occupied]
        return grid, weight

    def _sample(self, grid: np.ndarray, ox: float, oz: float) -> np.ndarray:
        """Bilinearly sample an offset grid at all particle positions."""
        gx = (self.position[:, 0] - self.xmin) / self.dx - ox
        gz = (self.position[:, 2] - self.zmin) / self.dx - oz
        i0 = np.floor(gx).astype(np.int64)
        k0 = np.floor(gz).astype(np.int64)
        fx = gx - i0
        fz = gz - k0
        result = np.zeros(len(gx), dtype=np.float64)
        total = np.zeros(len(gx), dtype=np.float64)
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            ii = i0 + di
            for dk, wz in ((0, 1.0 - fz), (1, fz)):
                kk = k0 + dk
                ww = wx * wz
                valid = ((ii >= 0) & (ii < grid.shape[0]) &
                         (kk >= 0) & (kk < grid.shape[1]))
                result[valid] += ww[valid] * grid[ii[valid], kk[valid]]
                total[valid] += ww[valid]
        valid_total = total > 1.0e-12
        result[valid_total] /= total[valid_total]
        return result

    @staticmethod
    def _extrapolate(grid: np.ndarray, valid: np.ndarray,
                     layers: int = 3) -> np.ndarray:
        """Extend velocities a few cells into air for particle sampling."""
        out = grid.copy()
        known = valid.copy()
        for _ in range(layers):
            sums = np.zeros_like(out)
            counts = np.zeros(out.shape, dtype=np.int16)
            sums[1:, :] += out[:-1, :] * known[:-1, :]
            counts[1:, :] += known[:-1, :]
            sums[:-1, :] += out[1:, :] * known[1:, :]
            counts[:-1, :] += known[1:, :]
            sums[:, 1:] += out[:, :-1] * known[:, :-1]
            counts[:, 1:] += known[:, :-1]
            sums[:, :-1] += out[:, 1:] * known[:, 1:]
            counts[:, :-1] += known[:, 1:]
            fill = (~known) & (counts > 0)
            out[fill] = sums[fill] / counts[fill]
            known[fill] = True
        return out

    def _mark_liquid(self, center_weight: np.ndarray) -> np.ndarray:
        fluid = center_weight > 0.045
        neighbors = np.zeros_like(center_weight, dtype=np.int16)
        neighbors[1:, :] += fluid[:-1, :]
        neighbors[:-1, :] += fluid[1:, :]
        neighbors[:, 1:] += fluid[:, :-1]
        neighbors[:, :-1] += fluid[:, 1:]
        fluid |= neighbors >= 3
        return fluid

    def _project(self, u: np.ndarray, w: np.ndarray,
                 fluid: np.ndarray, dt: float):
        """Pressure-project staggered velocity in occupied liquid cells."""
        divergence = ((u[1:, :] - u[:-1, :]) +
                      (w[:, 1:] - w[:, :-1])) / self.dx
        rhs = self.rho * self.dx * self.dx * divergence / dt

        # Solid sides/floor are Neumann; air and the open top are p=0.
        diagonal = np.full((self.nx, self.nz), 4.0, dtype=np.float64)
        diagonal[0, :] -= 1.0
        diagonal[-1, :] -= 1.0
        diagonal[:, 0] -= 1.0
        pressure = np.zeros_like(rhs)
        for _ in range(self.pressure_iterations):
            neighbor_sum = np.zeros_like(pressure)
            neighbor_sum[1:, :] += pressure[:-1, :] * fluid[:-1, :]
            neighbor_sum[:-1, :] += pressure[1:, :] * fluid[1:, :]
            neighbor_sum[:, 1:] += pressure[:, :-1] * fluid[:, :-1]
            neighbor_sum[:, :-1] += pressure[:, 1:] * fluid[:, 1:]
            updated = (neighbor_sum - rhs) / diagonal
            pressure = np.where(fluid, np.maximum(updated, 0.0), 0.0)

        scale = dt / (self.rho * self.dx)
        left_p = np.where(fluid[:-1, :], pressure[:-1, :], 0.0)
        right_p = np.where(fluid[1:, :], pressure[1:, :], 0.0)
        active_u = fluid[:-1, :] | fluid[1:, :]
        interior_u = u[1:self.nx, :]
        gradient_u = right_p - left_p
        interior_u[active_u] -= scale * gradient_u[active_u]

        lower_p = np.where(fluid[:, :-1], pressure[:, :-1], 0.0)
        upper_p = np.where(fluid[:, 1:], pressure[:, 1:], 0.0)
        active_w = fluid[:, :-1] | fluid[:, 1:]
        interior_w = w[:, 1:self.nz]
        gradient_w = upper_p - lower_p
        interior_w[active_w] -= scale * gradient_w[active_w]

        top_active = fluid[:, -1]
        w[:, self.nz][top_active] += scale * pressure[:, -1][top_active]
        u[0, :] = 0.0
        u[-1, :] = 0.0
        w[:, 0] = 0.0
        return u, w

    def _step(self, dt: float) -> None:
        vx = self.velocity[:, 0]
        vz = self.velocity[:, 2]
        u, u_weight = self._scatter(vx, 0.0, 0.5,
                                    (self.nx + 1, self.nz))
        w, w_weight = self._scatter(vz, 0.5, 0.0,
                                    (self.nx, self.nz + 1))
        _, center_weight = self._scatter(
            np.ones(len(vx)), 0.5, 0.5, (self.nx, self.nz))
        fluid = self._mark_liquid(center_weight)

        u = self._extrapolate(u, u_weight > 1.0e-12)
        w = self._extrapolate(w, w_weight > 1.0e-12)
        u_old = u.copy()
        w_old = w.copy()

        active_w = np.zeros_like(w, dtype=bool)
        active_w[:, 1:self.nz] = fluid[:, :-1] | fluid[:, 1:]
        active_w[:, self.nz] = fluid[:, -1]
        w[active_w] -= self.gravity * dt
        u, w = self._project(u, w, fluid, dt)

        active_u = np.zeros_like(u, dtype=bool)
        active_u[1:self.nx, :] = fluid[:-1, :] | fluid[1:, :]
        active_w[:, 1:self.nz] = fluid[:, :-1] | fluid[:, 1:]
        u = self._extrapolate(u, active_u)
        w = self._extrapolate(w, active_w)
        u[0, :] = u[-1, :] = 0.0
        w[:, 0] = 0.0

        pic_x = self._sample(u, 0.0, 0.5)
        pic_z = self._sample(w, 0.5, 0.0)
        flip_x = vx + self._sample(u - u_old, 0.0, 0.5)
        flip_z = vz + self._sample(w - w_old, 0.5, 0.0)
        a = self.pic_fraction
        self.velocity[:, 0] = (1.0 - a) * flip_x + a * pic_x
        self.velocity[:, 2] = (1.0 - a) * flip_z + a * pic_z
        self.velocity[:, 1] = 0.0

        # Mild viscosity dissipates late recirculation.
        self.velocity[:, (0, 2)] *= math.exp(-0.055 * dt)
        speed = np.linalg.norm(self.velocity[:, (0, 2)], axis=1)
        excessive = speed > 12.0
        if np.any(excessive):
            self.velocity[excessive, 0] *= 12.0 / speed[excessive]
            self.velocity[excessive, 2] *= 12.0 / speed[excessive]

        self.position[:, 0] += dt * self.velocity[:, 0]
        self.position[:, 2] += dt * self.velocity[:, 2]

        lo_x = self.xmin + self.particle_radius
        hi_x = self.xmax - self.particle_radius
        lo_z = self.zmin + self.particle_radius
        hit_left = self.position[:, 0] < lo_x
        hit_right = self.position[:, 0] > hi_x
        hit_floor = self.position[:, 2] < lo_z
        self.position[hit_left, 0] = lo_x
        self.position[hit_right, 0] = hi_x
        self.position[hit_floor, 2] = lo_z
        self.velocity[hit_left & (self.velocity[:, 0] < 0.0), 0] *= -0.04
        self.velocity[hit_right & (self.velocity[:, 0] > 0.0), 0] *= -0.04
        self.velocity[hit_floor & (self.velocity[:, 2] < 0.0), 2] *= -0.02
        self.velocity[hit_floor, 0] *= 0.985

        upper_guard = self.zmax - self.particle_radius
        too_high = self.position[:, 2] > upper_guard
        self.position[too_high, 2] = upper_guard
        self.velocity[too_high & (self.velocity[:, 2] > 0.0), 2] *= -0.02

        self._u = u
        self._w = w
        self._fluid = fluid
        self.time += dt
        self.step_count += 1

    def _update_diagnostic_grid(self) -> None:
        vx, weight = self._scatter(
            self.velocity[:, 0], 0.5, 0.5, (self.nx, self.nz))
        vz, _ = self._scatter(
            self.velocity[:, 2], 0.5, 0.5, (self.nx, self.nz))
        fluid = self._mark_liquid(weight)
        velocity = np.stack((vx, vz), axis=-1)
        velocity[~fluid, :] = 0.0
        distance_out = distance_transform_edt(~fluid) * self.dx
        distance_in = distance_transform_edt(fluid) * self.dx
        self._diagnostic_mask = fluid.astype(np.float64)
        self._diagnostic_level_set = distance_out - distance_in
        self._diagnostic_velocity = velocity

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")

        while self.time < target - 1.0e-12:
            speed = np.linalg.norm(self.velocity[:, (0, 2)], axis=1)
            max_speed = float(np.max(speed))
            wave_speed = math.sqrt(self.gravity * self.dx)
            cfl_dt = 0.38 * self.dx / (max_speed + wave_speed + 1.0e-9)
            dt = min(self.max_dt, cfl_dt, target - self.time)
            self._step(dt)
        self.time = target
        self._update_diagnostic_grid()

        n = len(self.position)
        characteristic_length = 2.4
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={"phase_id": np.ones(n, dtype=np.int32)},
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "liquid_interior_mask": self._diagnostic_mask.copy(),
                "liquid_level_set": self._diagnostic_level_set.copy(),
                "velocity_field": self._diagnostic_velocity.copy(),
            },
            scalars={
                "characteristic_length": characteristic_length,
                "grid_spacing": self.dx,
                "liquid_phase_id": 1.0,
                "reference_velocity": math.sqrt(
                    self.gravity * characteristic_length),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        kinetic = 0.5 * np.sum(
            self.mass * (self.velocity[:, 0] ** 2 +
                         self.velocity[:, 2] ** 2))
        return {
            "method": "hybrid PIC/FLIP staggered-grid projection",
            "substeps": int(self.step_count),
            "simulation_time": float(self.time),
            "kinetic_energy": float(kinetic),
            "particle_count": int(len(self.position)),
        }
