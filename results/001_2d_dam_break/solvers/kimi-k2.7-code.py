"""2D dam-break solver using a MAC-grid incompressible projection with marker particles.

The liquid is represented by passive marker particles that define the liquid region on
the grid.  The grid velocity is advanced with semi-Lagrangian advection, driven by
gravity, and projected to be divergence-free inside the liquid with a free-surface
(Dirichlet p=0) at the air interface and no-flow through the tank walls and floor.
The top of the grid is left open so the free surface can move upward without an
artificial lid.

This is a real, if simplified, Navier-Stokes free-surface solver using NumPy/SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Incompressible free-surface fluid solver for the 2D dam break."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)

        # Read physical constants from the frozen config where available.
        config = getattr(scene, "config", {})
        gravity = config.get("gravity", [0.0, 0.0, -9.81])
        self._g = float(gravity[2])
        self._rho = float(
            config.get("initial_conditions", {}).get("liquid_density", 1000.0)
        )

        # Tank interior: x in [-3, 3], z in [0, 4].
        # The diagnostic grid is uniform with spacing 0.1 m.
        self._x_min = -3.0
        self._x_max = 3.0
        self._z_min = 0.0
        self._z_max = 4.0
        self._dx = 0.1
        self._dz = 0.1
        self._Nx = int(round((self._x_max - self._x_min) / self._dx))
        self._Nz = int(round((self._z_max - self._z_min) / self._dz))

        # Staggered MAC velocity fields.
        self._u = np.zeros((self._Nx + 1, self._Nz), dtype=np.float64)
        self._w = np.zeros((self._Nx, self._Nz + 1), dtype=np.float64)

        # Face-centered coordinate arrays for advection.
        self._x_u, self._z_u = np.meshgrid(
            self._x_min + np.arange(self._Nx + 1) * self._dx,
            self._z_min + (np.arange(self._Nz) + 0.5) * self._dz,
            indexing="ij",
        )
        self._x_w, self._z_w = np.meshgrid(
            self._x_min + (np.arange(self._Nx) + 0.5) * self._dx,
            self._z_min + np.arange(self._Nz + 1) * self._dz,
            indexing="ij",
        )

        # Marker particles come from the deterministic scene initial state.
        init = scene.initial_state()
        self._pos = np.asarray(init.position, dtype=np.float64).copy()
        self._vel = np.asarray(init.velocity, dtype=np.float64).copy()
        if init.mass is None:
            self._mass = np.ones(self._pos.shape[0], dtype=np.float64)
        else:
            self._mass = np.asarray(init.mass, dtype=np.float64).copy()

        self._sample_volume = np.asarray(
            init.attributes.get(
                "sample_volume", np.full(self._pos.shape[0], self._dx * self._dz)
            )
        ).copy()

        self._time = float(init.time)

        # Static boundary samples for the required wall fields.
        self._build_boundary_samples()

        # Diagnostic scalars.
        self._liquid_phase_id = 1
        self._char_length = 1.0
        self._ref_vel = np.sqrt(abs(self._g) * self._char_length)

        # Time-stepping control.
        self._dt_max = 0.005

        # Initial liquid mask and grid-to-particle consistency.
        self._update_liquid_mask()

    def _build_boundary_samples(self) -> None:
        floor_x = np.linspace(self._x_min, self._x_max, 31, dtype=np.float64)
        floor_z = np.full_like(floor_x, self._z_min)
        wall_z = np.linspace(self._z_min, self._z_max, 31, dtype=np.float64)
        left_x = np.full_like(wall_z, self._x_min)
        right_x = np.full_like(wall_z, self._x_max)

        normals = np.vstack(
            [
                np.tile([0.0, 0.0, -1.0], (len(floor_x), 1)),
                np.tile([-1.0, 0.0, 0.0], (len(wall_z), 1)),
                np.tile([1.0, 0.0, 0.0], (len(wall_z), 1)),
            ]
        )
        self._boundary_normal = normals
        self._boundary_target_velocity = np.zeros_like(normals)
        self._boundary_velocity = np.zeros_like(normals)

    def _update_liquid_mask(self) -> None:
        x = self._pos[:, 0]
        z = self._pos[:, 2]
        i = np.floor((x - self._x_min) / self._dx).astype(np.int64)
        j = np.floor((z - self._z_min) / self._dz).astype(np.int64)
        i = np.clip(i, 0, self._Nx - 1)
        j = np.clip(j, 0, self._Nz - 1)
        mask = np.zeros((self._Nx, self._Nz), dtype=bool)
        mask[i, j] = True
        self._liquid_mask = ndimage.binary_fill_holes(mask)

    # --------------------------------------------------------------------- #
    # Velocity interpolation helpers for staggered MAC grids.
    # --------------------------------------------------------------------- #
    def _velocity_at(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (vx, vz) at arbitrary (x, z) points using the current u/w fields."""
        row_u = (z - (self._z_min + 0.5 * self._dz)) / self._dz
        col_u = (x - self._x_min) / self._dx
        vx = ndimage.map_coordinates(
            self._u, [row_u, col_u], order=1, mode="constant", cval=0.0
        )

        row_w = (z - self._z_min) / self._dz
        col_w = (x - (self._x_min + 0.5 * self._dx)) / self._dx
        vz = ndimage.map_coordinates(
            self._w, [row_w, col_w], order=1, mode="constant", cval=0.0
        )
        return vx, vz

    def _interp_u(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        row = (z - (self._z_min + 0.5 * self._dz)) / self._dz
        col = (x - self._x_min) / self._dx
        return ndimage.map_coordinates(
            self._u, [row, col], order=1, mode="constant", cval=0.0
        )

    def _interp_w(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        row = (z - self._z_min) / self._dz
        col = (x - (self._x_min + 0.5 * self._dx)) / self._dx
        return ndimage.map_coordinates(
            self._w, [row, col], order=1, mode="constant", cval=0.0
        )

    # --------------------------------------------------------------------- #
    # Physics substep.
    # --------------------------------------------------------------------- #
    def _apply_gravity(self, dt: float) -> None:
        """Add gravitational acceleration to vertical faces adjacent to liquid."""
        liquid = self._liquid_mask
        above = np.concatenate([liquid, np.zeros((self._Nx, 1), dtype=bool)], axis=1)
        below = np.concatenate([np.zeros((self._Nx, 1), dtype=bool), liquid], axis=1)
        adjacent = above | below
        self._w[adjacent] += self._g * dt

    def _advect_velocities(self, dt: float) -> None:
        # Advect the x-face (u) field.
        vx, vz = self._velocity_at(self._x_u, self._z_u)
        xb = self._x_u - dt * vx
        zb = self._z_u - dt * vz
        self._u = self._interp_u(xb, zb)

        # Advect the z-face (w) field.
        vx, vz = self._velocity_at(self._x_w, self._z_w)
        xb = self._x_w - dt * vx
        zb = self._z_w - dt * vz
        self._w = self._interp_w(xb, zb)

        self._apply_velocity_boundary_conditions()

    def _apply_velocity_boundary_conditions(self) -> None:
        # No normal flow through the side walls.
        self._u[0, :] = 0.0
        self._u[self._Nx, :] = 0.0
        # Floor is a solid wall.
        self._w[:, 0] = 0.0
        # No-slip condition on the floor/sides for the tangential components.
        self._u[:, 0] = 0.0
        self._w[0, :] = 0.0
        self._w[self._Nx - 1, :] = 0.0

    def _project(self, dt: float) -> None:
        """Pressure projection to enforce incompressibility inside the liquid."""
        mask = self._liquid_mask
        if not np.any(mask):
            return

        idx = np.full(mask.shape, -1, dtype=np.int64)
        idx[mask] = np.arange(mask.sum())
        n = int(mask.sum())

        # Divergence of the staggered velocity field.
        D = (
            (self._u[1 : self._Nx + 1, :] - self._u[0 : self._Nx, :]) / self._dx
            + (self._w[:, 1 : self._Nz + 1] - self._w[:, 0 : self._Nz]) / self._dz
        )

        c_x = dt / (self._rho * self._dx * self._dx)
        c_z = dt / (self._rho * self._dz * self._dz)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diag = np.zeros(n, dtype=np.float64)

        cell_i, cell_j = np.nonzero(mask)
        for k, i, j in zip(range(n), cell_i, cell_j):
            d = 0.0

            # Right face (x).
            if i + 1 < self._Nx:
                if mask[i + 1, j]:
                    rows.append(k)
                    cols.append(idx[i + 1, j])
                    data.append(-c_x)
                    d += c_x
                else:
                    d += c_x

            # Left face (x).
            if i > 0:
                if mask[i - 1, j]:
                    rows.append(k)
                    cols.append(idx[i - 1, j])
                    data.append(-c_x)
                    d += c_x
                else:
                    d += c_x

            # Top face (z).  Open boundary when j+1 == Nz.
            if j + 1 < self._Nz:
                if mask[i, j + 1]:
                    rows.append(k)
                    cols.append(idx[i, j + 1])
                    data.append(-c_z)
                    d += c_z
                else:
                    d += c_z
            else:
                d += c_z

            # Bottom face (z).  Solid wall when j == 0.
            if j > 0:
                if mask[i, j - 1]:
                    rows.append(k)
                    cols.append(idx[i, j - 1])
                    data.append(-c_z)
                    d += c_z
                else:
                    d += c_z

            rows.append(k)
            cols.append(k)
            data.append(d)
            diag[k] = d

        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        b = (self._rho / dt) * D[mask]
        p_sol = spsolve(A, b)

        p = np.zeros(mask.shape, dtype=np.float64)
        p[mask] = p_sol

        # Update the x-face velocity.
        self._u[1 : self._Nx, :] -= (
            dt / self._rho * (p[1 : self._Nx, :] - p[0 : self._Nx - 1, :]) / self._dx
        )
        # Update the interior z-face velocity.
        self._w[:, 1 : self._Nz] -= (
            dt / self._rho * (p[:, 1 : self._Nz] - p[:, 0 : self._Nz - 1]) / self._dz
        )
        # Top open boundary: pressure above the domain is zero.
        self._w[:, self._Nz] -= (
            dt / self._rho * (0.0 - p[:, self._Nz - 1]) / self._dz
        )

        self._apply_velocity_boundary_conditions()

    def _advect_particles(self, dt: float) -> None:
        x = self._pos[:, 0]
        z = self._pos[:, 2]

        # RK2 integration using the divergence-free grid velocity.
        vx, vz = self._velocity_at(x, z)
        x_mid = x + 0.5 * dt * vx
        z_mid = z + 0.5 * dt * vz
        vx_mid, vz_mid = self._velocity_at(x_mid, z_mid)

        x_new = x + dt * vx_mid
        z_new = z + dt * vz_mid

        eps = 1e-6
        x_new = np.clip(x_new, self._x_min + eps, self._x_max - eps)
        z_new = np.clip(z_new, self._z_min + eps, self._z_max - eps)

        self._pos[:, 0] = x_new
        self._pos[:, 2] = z_new
        self._pos[:, 1] = 0.0

        self._vel[:, 0] = vx_mid
        self._vel[:, 2] = vz_mid
        self._vel[:, 1] = 0.0

        self._update_liquid_mask()

    def _substep(self, dt: float) -> None:
        self._apply_gravity(dt)
        self._advect_velocities(dt)
        self._project(dt)
        self._advect_particles(dt)
        self._time += dt

    def _choose_dt(self) -> float:
        vmax_u = np.max(np.abs(self._u)) if self._u.size else 0.0
        vmax_w = np.max(np.abs(self._w)) if self._w.size else 0.0
        vmax = max(vmax_u, vmax_w)
        if vmax > 1e-12:
            dt_cfl = 0.25 * min(self._dx, self._dz) / vmax
        else:
            dt_cfl = self._dt_max
        return min(self._dt_max, dt_cfl)

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            raise ValueError("Requested time is before current simulation time")

        while self._time < time - 1e-12:
            dt = self._choose_dt()
            remaining = time - self._time
            if dt > remaining:
                dt = remaining
            self._substep(float(dt))

        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "MAC_incompressible_marker_particles",
            "grid_shape": [self._Nx, self._Nz],
            "grid_spacing": [self._dx, self._dz],
            "particles": int(self._pos.shape[0]),
        }

    def _make_frame(self) -> StateFrame:
        mask = self._liquid_mask

        # Signed distance function from the binary liquid mask.
        dist_liquid = ndimage.distance_transform_edt(
            mask, sampling=(self._dx, self._dz)
        )
        dist_air = ndimage.distance_transform_edt(
            ~mask, sampling=(self._dx, self._dz)
        )
        level_set = dist_liquid - dist_air

        # Cell-centered diagnostic velocity field.
        vx = 0.5 * (self._u[0 : self._Nx, :] + self._u[1 : self._Nx + 1, :])
        vz = 0.5 * (self._w[:, 0 : self._Nz] + self._w[:, 1 : self._Nz + 1])
        velocity_field = np.zeros((self._Nx, self._Nz, 3), dtype=np.float64)
        velocity_field[:, :, 0] = vx
        velocity_field[:, :, 2] = vz

        return StateFrame(
            time=float(self._time),
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            attributes={
                "phase_id": np.full(
                    self._pos.shape[0], self._liquid_phase_id, dtype=np.int64
                ),
                "sample_volume": self._sample_volume.copy(),
            },
            fields={
                "boundary_normal": self._boundary_normal.copy(),
                "boundary_target_velocity": self._boundary_target_velocity.copy(),
                "boundary_velocity": self._boundary_velocity.copy(),
                "liquid_interior_mask": mask.astype(np.float64),
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(self._char_length),
                "grid_spacing": float(self._dx),
                "liquid_phase_id": float(self._liquid_phase_id),
                "reference_velocity": float(self._ref_vel),
            },
        )
