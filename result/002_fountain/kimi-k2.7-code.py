"""Particle-grid weakly-compressible fluid solver for the fountain benchmark.

Implements a simple grid-based pressure force plus Lagrangian particle
advection.  Boundaries are handled by geometric reflection against the basin
floor/wall and the central nozzle.  A sustained inflow at the nozzle exit
carries mass and upward momentum into the domain.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        # Deterministic seeding
        np.random.seed(seed)
        self.seed = seed

        # Pull configuration; fall back to reading the scaffold directly.
        try:
            config = scene.config
        except AttributeError:
            import yaml

            with open("config.yaml", "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        self.config = config

        # Physical constants from the frozen configuration
        ic = config["initial_conditions"]
        self.rho0: float = float(ic["liquid_density"])
        self.dx: float = float(ic["particle_spacing"])
        self.pool_radius: float = float(ic["pool_radius"])
        self.pool_surface_z: float = float(ic["pool_surface_z"])
        self.nozzle_radius: float = float(ic["nozzle_radius"])
        self.nozzle_exit_z: float = float(ic["nozzle_exit_z"])
        self.jet_speed: float = float(ic["jet_speed"])
        self.gravity = np.array(config["gravity"], dtype=np.float64)

        # Geometry helpers
        self.basin_radius: float = 1.50
        self.basin_wall_top: float = 0.52
        self.nozzle_obstacle_radius: float = 0.24
        self.nozzle_height: float = 0.34

        # Particle mass consistent with the initial lattice
        self.particle_mass: float = self.rho0 * self.dx**3

        # Diagnostic grid covering the particle domain declared in scene_geometry
        self.bounds_min = np.array([-1.5, -1.5, 0.04], dtype=np.float64)
        self.bounds_max = np.array([1.5, 1.5, 3.20], dtype=np.float64)
        self.grid_shape = tuple(
            np.ceil((self.bounds_max - self.bounds_min) / self.dx).astype(int)
        )
        self.cell_volume: float = self.dx**3

        # Time stepping
        # CFL: dx/c with c = sqrt(B/rho0) ≈ 31 m/s gives dt < 2.7e-3 s.
        self.dt: float = 5.0e-4
        self.B: float = 1.0e6  # Tait EOS pressure scale (Pa)
        self.gamma: float = 7.0
        self.restitution: float = 0.2

        # Inflow bookkeeping
        source_area = math.pi * self.nozzle_radius**2
        self.mass_flux: float = self.rho0 * self.jet_speed * source_area
        self.mass_to_emit: float = 0.0
        self.cumulative_mass_source: float = 0.0

        # Scalar normalizations for trajectory output
        self.characteristic_length: float = self.nozzle_radius
        self.reference_velocity: float = self.jet_speed
        self.grid_spacing: float = self.dx

        # State from the benchmark scene
        init = scene.initial_state()
        self.position = init.position.copy()
        self.velocity = init.velocity.copy()
        self.mass = init.mass.copy()
        self.source_region = np.asarray(
            init.attributes.get("source_region", np.zeros(len(self.position))),
            dtype=np.float64,
        ).copy()
        self.time: float = 0.0

        # Static boundary samples for trajectory export
        self._build_boundary_samples()

    def _build_boundary_samples(self) -> None:
        pts: list[list[float]] = []
        nrm: list[list[float]] = []

        # Basin floor (z = 0), normal points into the liquid
        n_rings = 5
        n_ang_floor = 16
        for i in range(1, n_rings + 1):
            r = self.basin_radius * i / n_rings
            for th in np.linspace(0.0, 2.0 * math.pi, n_ang_floor, endpoint=False):
                pts.append([r * math.cos(th), r * math.sin(th), 0.0])
                nrm.append([0.0, 0.0, 1.0])
        pts.append([0.0, 0.0, 0.0])
        nrm.append([0.0, 0.0, 1.0])

        # Basin wall (r = R, 0 <= z <= wall_top)
        n_z = 8
        n_ang_wall = 24
        for z in np.linspace(0.0, self.basin_wall_top, n_z):
            for th in np.linspace(0.0, 2.0 * math.pi, n_ang_wall, endpoint=False):
                c = math.cos(th)
                s = math.sin(th)
                pts.append([self.basin_radius * c, self.basin_radius * s, z])
                nrm.append([-c, -s, 0.0])

        # Central nozzle wall (r = R_n, 0 <= z <= nozzle_height)
        n_z_noz = 6
        n_ang_noz = 16
        for z in np.linspace(0.0, self.nozzle_height, n_z_noz):
            for th in np.linspace(0.0, 2.0 * math.pi, n_ang_noz, endpoint=False):
                c = math.cos(th)
                s = math.sin(th)
                pts.append([self.nozzle_obstacle_radius * c, self.nozzle_obstacle_radius * s, z])
                nrm.append([c, s, 0.0])

        self.boundary_points = np.array(pts, dtype=np.float64)
        self.boundary_normals = np.array(nrm, dtype=np.float64)
        self.boundary_target_velocity = np.zeros_like(self.boundary_points)
        self.boundary_velocity = np.zeros_like(self.boundary_points)

    def advance_to(self, time: float) -> StateFrame:
        while self.time < time - 1e-12:
            self._step()
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "particle_count": int(len(self.position)),
            "final_time": float(self.time),
            "cumulative_mass_source_kg": float(self.cumulative_mass_source),
        }

    # ------------------------------------------------------------------
    # Internal time-stepping
    # ------------------------------------------------------------------
    def _step(self) -> None:
        # Pressure acceleration from a cell-centered density field
        a = self._pressure_acceleration()

        # Integrate: explicit Euler for velocity, then position
        self.velocity += (a + self.gravity) * self.dt
        self.position += self.velocity * self.dt

        # Enforce boundary conditions
        self._apply_boundaries()

        # Sustained inflow from the nozzle
        self._emit_inflow()

        self.time += self.dt

    def _pressure_acceleration(self) -> np.ndarray:
        n = len(self.position)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        cell = self._positions_to_cells(self.position)

        mass_grid = np.zeros(self.grid_shape, dtype=np.float64)
        np.add.at(mass_grid, (cell[:, 0], cell[:, 1], cell[:, 2]), self.mass)

        rho = mass_grid / self.cell_volume
        eta = rho / self.rho0
        # Tait equation of state, clamped to zero (no tensile pressure)
        p = (self.B / self.gamma) * np.maximum(0.0, eta**self.gamma - 1.0)

        # Pressure gradient on the grid
        grad_p_x, grad_p_y, grad_p_z = np.gradient(p, self.dx)

        # Sample back to particles from their containing cell
        gp = np.stack(
            [
                grad_p_x[cell[:, 0], cell[:, 1], cell[:, 2]],
                grad_p_y[cell[:, 0], cell[:, 1], cell[:, 2]],
                grad_p_z[cell[:, 0], cell[:, 1], cell[:, 2]],
            ],
            axis=1,
        )

        a = -gp / self.rho0
        return a

    def _apply_boundaries(self) -> None:
        # Floor at z = 0
        below = self.position[:, 2] < 0.0
        if np.any(below):
            self.position[below, 2] = 0.0
            self.velocity[below, 2] = -self.velocity[below, 2] * self.restitution

        # Basin cylindrical wall
        r = np.hypot(self.position[:, 0], self.position[:, 1])
        mask = (r > self.basin_radius) & (self.position[:, 2] < self.basin_wall_top)
        if np.any(mask):
            theta = np.arctan2(self.position[mask, 1], self.position[mask, 0])
            self.position[mask, 0] = self.basin_radius * np.cos(theta)
            self.position[mask, 1] = self.basin_radius * np.sin(theta)

            vx, vy = self.velocity[mask, 0], self.velocity[mask, 1]
            vr = vx * np.cos(theta) + vy * np.sin(theta)
            vt = -vx * np.sin(theta) + vy * np.cos(theta)
            vr = -vr * self.restitution
            self.velocity[mask, 0] = vr * np.cos(theta) - vt * np.sin(theta)
            self.velocity[mask, 1] = vr * np.sin(theta) + vt * np.cos(theta)

        # Central nozzle obstacle (solid cylinder r <= R_n, 0 <= z <= h)
        r = np.hypot(self.position[:, 0], self.position[:, 1])
        mask = (
            (r < self.nozzle_obstacle_radius)
            & (self.position[:, 2] >= 0.0)
            & (self.position[:, 2] <= self.nozzle_height)
        )
        if np.any(mask):
            theta = np.arctan2(self.position[mask, 1], self.position[mask, 0])
            self.position[mask, 0] = self.nozzle_obstacle_radius * np.cos(theta)
            self.position[mask, 1] = self.nozzle_obstacle_radius * np.sin(theta)

            vx, vy = self.velocity[mask, 0], self.velocity[mask, 1]
            vr = vx * np.cos(theta) + vy * np.sin(theta)
            vt = -vx * np.sin(theta) + vy * np.cos(theta)
            # Bounce back toward the cylinder axis
            vr = -vr * self.restitution
            self.velocity[mask, 0] = vr * np.cos(theta) - vt * np.sin(theta)
            self.velocity[mask, 1] = vr * np.sin(theta) + vt * np.cos(theta)

        # Safety: keep particles inside the diagnostic box for indexing
        for dim in (0, 1, 2):
            lo = self.bounds_min[dim]
            hi = self.bounds_max[dim]
            too_low = self.position[:, dim] < lo
            too_high = self.position[:, dim] > hi
            if np.any(too_low):
                self.position[too_low, dim] = 2.0 * lo - self.position[too_low, dim]
                self.velocity[too_low, dim] *= -self.restitution
            if np.any(too_high):
                self.position[too_high, dim] = 2.0 * hi - self.position[too_high, dim]
                self.velocity[too_high, dim] *= -self.restitution

    def _emit_inflow(self) -> None:
        self.mass_to_emit += self.mass_flux * self.dt
        n_new = int(self.mass_to_emit // self.particle_mass)
        if n_new <= 0:
            return

        # Uniform sampling over the nozzle disk
        u = np.random.rand(n_new)
        r = self.nozzle_radius * np.sqrt(u)
        theta = 2.0 * math.pi * np.random.rand(n_new)

        new_pos = np.empty((n_new, 3), dtype=np.float64)
        new_pos[:, 0] = r * np.cos(theta)
        new_pos[:, 1] = r * np.sin(theta)
        new_pos[:, 2] = self.nozzle_exit_z

        new_vel = np.zeros((n_new, 3), dtype=np.float64)
        new_vel[:, 2] = self.jet_speed

        new_mass = np.full(n_new, self.particle_mass, dtype=np.float64)
        new_region = np.ones(n_new, dtype=np.float64)

        self.position = np.vstack([self.position, new_pos])
        self.velocity = np.vstack([self.velocity, new_vel])
        self.mass = np.concatenate([self.mass, new_mass])
        self.source_region = np.concatenate([self.source_region, new_region])

        self.cumulative_mass_source += n_new * self.particle_mass
        self.mass_to_emit -= n_new * self.particle_mass

    def _positions_to_cells(self, pos: np.ndarray) -> np.ndarray:
        cell = np.floor((pos - self.bounds_min) / self.dx).astype(int)
        shape_arr = np.array(self.grid_shape)
        return np.clip(cell, 0, shape_arr - 1)

    # ------------------------------------------------------------------
    # StateFrame construction
    # ------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        active_mask, level_set, velocity_field = self._compute_grid_fields()

        return StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "source_region": self.source_region.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "active_mask": active_mask,
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
                "boundary_normal": self.boundary_normals,
                "boundary_target_velocity": self.boundary_target_velocity,
                "boundary_velocity": self.boundary_velocity,
                "cumulative_mass_source": np.array(
                    [self.cumulative_mass_source], dtype=np.float64
                ),
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.grid_spacing),
                "reference_velocity": float(self.reference_velocity),
            },
        )

    def _compute_grid_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass_grid = np.zeros(self.grid_shape, dtype=np.float64)
        momentum_grid = np.zeros(self.grid_shape + (3,), dtype=np.float64)

        if len(self.position) > 0:
            cell = self._positions_to_cells(self.position)
            np.add.at(mass_grid, (cell[:, 0], cell[:, 1], cell[:, 2]), self.mass)
            for d in range(3):
                np.add.at(
                    momentum_grid[..., d],
                    (cell[:, 0], cell[:, 1], cell[:, 2]),
                    self.mass * self.velocity[:, d],
                )

        velocity_field = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        mask = mass_grid > 0.0
        if np.any(mask):
            for d in range(3):
                velocity_field[..., d][mask] = (
                    momentum_grid[..., d][mask] / mass_grid[mask]
                )

        active_mask = mask.astype(np.float64)

        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(~mask)
        level_set = (dist_out.astype(np.float64) - dist_in.astype(np.float64)) * self.dx

        return active_mask, level_set, velocity_field
