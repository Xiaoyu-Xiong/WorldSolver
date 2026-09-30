"""Physics solver for the U-corridor dam-break scene.

Implements a weakly-compressible SPH fluid simulation with rigid corridor
boundaries and reconstructs the required diagnostic fields on the frozen
diagnostic grid.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame
import yaml


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene
        self.frame0 = scene.initial_state()

        with Path("config.yaml").open("r") as f:
            self.config = yaml.safe_load(f)

        self.gravity = np.array(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )

        # Particle state
        self.position = self.frame0.position.copy()
        self.velocity = self.frame0.velocity.copy()
        self.mass = self.frame0.mass.copy()
        self.body_id = (
            self.frame0.body_id.copy()
            if self.frame0.body_id is not None
            else np.zeros(len(self.position), dtype=int)
        )
        self.base_attributes = {
            k: np.asarray(v).copy() for k, v in self.frame0.attributes.items()
        }

        # Physical / numerical parameters
        self.rho0 = float(self.config["sampling"]["density"])
        self.spacing = float(self.config["sampling"]["spacing"])
        self.h = 1.35 * self.spacing  # smoothing length
        self.c0 = 30.0
        self.gamma = 7.0
        self.alpha = 0.08
        self.beta = 0.0
        self.eta2 = 0.01 * self.h**2
        self.B = self.c0**2 * self.rho0 / self.gamma
        self.dt_max = 5.0e-4
        self.time = 0.0

        self.r_particle = float(
            np.asarray(self.frame0.attributes["particle_radius"]).mean()
        )

        # Corridor inner bounds
        self.x_min, self.x_max = -1.15, 1.15
        self.y_min, self.y_max = -1.55, 1.55
        self.z_floor = 0.0
        self.wall_height = 1.1

        # Diagnostic grid
        self.field_shape = tuple(self.config["field_grid_shape"])
        self.grid_origin = np.array([-1.25, -1.75, 0.0], dtype=float)
        self.grid_extent = np.array([2.5, 3.4, 1.2], dtype=float)
        self.grid_spacing_vec = self.grid_extent / np.array(self.field_shape, dtype=float)

        self._init_boundary_samples()
        self._init_grid_points()

    def _init_boundary_samples(self) -> None:
        samples: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        def add_samples(points: np.ndarray, normal: list[float], target: list[float]) -> None:
            points = np.asarray(points, dtype=float)
            n = len(points)
            samples.append(points)
            normals.append(np.tile(np.array(normal, dtype=float), (n, 1)))
            targets.append(np.tile(np.array(target, dtype=float), (n, 1)))

        # Floor
        xf = np.linspace(-1.15, 1.15, 12)
        yf = np.linspace(-1.55, 1.55, 12)
        Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
        floor_pts = np.column_stack((Xf.ravel(), Yf.ravel(), np.full(Xf.size, 0.01)))
        add_samples(floor_pts, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])

        # Left wall
        yw = np.linspace(-1.55, 1.55, 12)
        zw = np.linspace(0.05, 1.05, 6)
        Yw, Zw = np.meshgrid(yw, zw, indexing="ij")
        left_pts = np.column_stack((np.full(Yw.size, -1.14), Yw.ravel(), Zw.ravel()))
        add_samples(left_pts, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Right wall
        right_pts = np.column_stack((np.full(Yw.size, 1.14), Yw.ravel(), Zw.ravel()))
        add_samples(right_pts, [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # End wall
        xw2 = np.linspace(-1.15, 1.15, 12)
        zw2 = np.linspace(0.05, 1.05, 6)
        Xw2, Zw2 = np.meshgrid(xw2, zw2, indexing="ij")
        end_pts = np.column_stack((Xw2.ravel(), np.full(Xw2.size, 1.54), Zw2.ravel()))
        add_samples(end_pts, [0.0, -1.0, 0.0], [0.0, 0.0, 0.0])

        # Gate (kinematic, lifted upward)
        xg = np.linspace(-0.85, 0.85, 10)
        zg = np.linspace(0.05, 1.05, 5)
        Xg, Zg = np.meshgrid(xg, zg, indexing="ij")
        gate_pts = np.column_stack((Xg.ravel(), np.full(Xg.size, -0.59), Zg.ravel()))
        add_samples(gate_pts, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

        self.boundary_samples = np.concatenate(samples, axis=0)
        self.boundary_normals = np.concatenate(normals, axis=0)
        self.boundary_target_velocity = np.concatenate(targets, axis=0)
        self.boundary_velocity = self.boundary_target_velocity.copy()

    def _init_grid_points(self) -> None:
        nx, ny, nz = self.field_shape
        ix = np.arange(nx) + 0.5
        iy = np.arange(ny) + 0.5
        iz = np.arange(nz) + 0.5
        cx = self.grid_origin[0] + ix * self.grid_spacing_vec[0]
        cy = self.grid_origin[1] + iy * self.grid_spacing_vec[1]
        cz = self.grid_origin[2] + iz * self.grid_spacing_vec[2]
        self.grid_coords = (cx, cy, cz)
        X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        self.grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # --------------------------------------------------------------------- #
    # SPH kernels and forces
    # --------------------------------------------------------------------- #
    def _kernel(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return W(r) and dW/dr for the 3D cubic spline kernel."""
        q = r / self.h
        alpha = 1.0 / (math.pi * self.h**3)
        W = np.zeros_like(r)
        dW = np.zeros_like(r)

        m1 = q <= 1.0
        W[m1] = alpha * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
        dW[m1] = alpha * (-3.0 * q[m1] + 2.25 * q[m1] ** 2) / self.h

        m2 = (q > 1.0) & (q <= 2.0)
        W[m2] = alpha * 0.25 * (2.0 - q[m2]) ** 3
        dW[m2] = alpha * (-0.75) * (2.0 - q[m2]) ** 2 / self.h
        return W, dW

    def _compute_acceleration(self) -> np.ndarray:
        x = self.position
        v = self.velocity

        dx = x[:, None, :] - x[None, :, :]
        r = np.linalg.norm(dx, axis=2)

        W, dWdr = self._kernel(r)
        density = np.sum(self.mass[None, :] * W, axis=1)
        density = np.maximum(density, 0.1 * self.rho0)

        pressure = self.B * ((density / self.rho0) ** self.gamma - 1.0)
        pressure = np.maximum(pressure, 0.0)

        term_p = (
            pressure[:, None] / density[:, None] ** 2
            + pressure[None, :] / density[None, :] ** 2
        )

        dv = v[:, None, :] - v[None, :, :]
        dot = np.einsum("ijk,ijk->ij", dx, dv)
        mu = self.h * dot / (r**2 + self.eta2)
        mu[dot >= 0.0] = 0.0

        rho_avg = 0.5 * (density[:, None] + density[None, :])
        pi = (-self.alpha * self.c0 * mu + self.beta * mu * mu) / rho_avg

        term = term_p + pi

        with np.errstate(divide="ignore", invalid="ignore"):
            gradW = dWdr[:, :, None] * dx / r[:, :, None]
        gradW[r == 0] = 0.0

        force = -np.sum(self.mass[None, :, None] * term[:, :, None] * gradW, axis=1)
        acc = force / self.mass[:, None] + self.gravity[None, :]
        return acc

    def _enforce_boundaries(self) -> None:
        r = self.r_particle

        # Floor
        mask = self.position[:, 2] < self.z_floor + r
        self.position[mask, 2] = self.z_floor + r
        self.velocity[mask, 2] = np.abs(self.velocity[mask, 2]) * 0.5

        # Left wall
        mask = self.position[:, 0] < self.x_min + r
        self.position[mask, 0] = self.x_min + r
        self.velocity[mask, 0] = np.abs(self.velocity[mask, 0]) * 0.5

        # Right wall
        mask = self.position[:, 0] > self.x_max - r
        self.position[mask, 0] = self.x_max - r
        self.velocity[mask, 0] = -np.abs(self.velocity[mask, 0]) * 0.5

        # Back wall (reservoir end)
        mask = self.position[:, 1] < self.y_min + r
        self.position[mask, 1] = self.y_min + r
        self.velocity[mask, 1] = np.abs(self.velocity[mask, 1]) * 0.5

        # End wall
        mask = self.position[:, 1] > self.y_max - r
        self.position[mask, 1] = self.y_max - r
        self.velocity[mask, 1] = -np.abs(self.velocity[mask, 1]) * 0.5

        # Soft ceiling to keep particles from escaping the simulation domain
        mask = self.position[:, 2] > self.wall_height + 0.5
        self.position[mask, 2] = self.wall_height + 0.5
        self.velocity[mask, 2] = -np.abs(self.velocity[mask, 2]) * 0.3

    def _step(self, dt: float) -> None:
        acc = self._compute_acceleration()
        self.velocity += dt * acc
        self.position += dt * self.velocity
        self._enforce_boundaries()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            raise ValueError("Target time must be >= current time")

        if target == self.time:
            return self._build_frame(target)

        dt_total = target - self.time
        n_steps = max(1, int(np.ceil(dt_total / self.dt_max)))
        dt = dt_total / n_steps
        for _ in range(n_steps):
            self._step(dt)

        self.time = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "sph_particles": len(self.position),
            "sph_smoothing_length": self.h,
            "sph_dt_max": self.dt_max,
        }

    # --------------------------------------------------------------------- #
    # Frame construction
    # --------------------------------------------------------------------- #
    def _build_frame(self, time: float) -> StateFrame:
        # Level set as signed distance to union of particle spheres
        tree = cKDTree(self.position)
        dist, _ = tree.query(self.grid_points)
        liquid_level_set = (dist - self.r_particle).reshape(self.field_shape)

        # Velocity field: Gaussian-weighted particle-to-grid interpolation
        diff = self.grid_points[:, None, :] - self.position[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        q = r / self.h
        weights = np.exp(-q * q)
        weights[r > 2.0 * self.h] = 0.0
        denom = np.sum(weights, axis=1)[:, None]
        denom[denom == 0.0] = 1.0
        v_flat = np.sum(weights[:, :, None] * self.velocity[None, :, :], axis=1) / denom
        velocity_field = v_flat.reshape(self.field_shape + (3,))

        # Active mask: cells lying inside the corridor volume
        gx, gy, gz = self.grid_coords
        X = gx[:, None, None] * np.ones((1, len(gy), len(gz)))
        Y = gy[None, :, None] * np.ones((len(gx), 1, len(gz)))
        Z = gz[None, None, :] * np.ones((len(gx), len(gy), 1))
        active_mask = (
            (X >= self.x_min)
            & (X <= self.x_max)
            & (Y >= self.y_min)
            & (Y <= self.y_max)
            & (Z >= self.z_floor)
            & (Z <= self.wall_height)
        )

        attributes = {
            k: np.asarray(v).copy() for k, v in self.base_attributes.items()
        }
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "active_mask": active_mask.astype(bool),
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": self.boundary_velocity.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "liquid_level_set": liquid_level_set,
            "velocity_field": velocity_field,
        }

        scalars = dict(self.frame0.scalars)
        scalars["grid_spacing"] = float(self.spacing)
        scalars["characteristic_length"] = float(
            scalars.get("characteristic_length", self.spacing)
        )
        scalars["reference_velocity"] = float(
            scalars.get("reference_velocity", 1.0)
        )

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
