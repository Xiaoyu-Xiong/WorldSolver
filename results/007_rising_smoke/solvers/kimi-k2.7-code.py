"""Buoyant smoke solver for 007_rising_smoke.

The solver integrates a buoyancy-driven scalar field on the frozen cell-centered
grid using a stable semi-Lagrangian advection scheme.  It produces the required
StateFrame trajectory fields at every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Simple but physical buoyant smoke solver.

    Steps:
      1. Inject smoke at the prescribed cylindrical source.
      2. Apply buoyancy and linear drag to the velocity field.
      3. Advect velocity and density with the semi-Lagrangian method.
      4. Apply density decay and small diffusion/viscosity.
      5. Enforce closed-wall free-slip and open outflow boundary conditions.
    """

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = scene.config
        self.frame0 = scene.initial_state()

        # Domain
        domain = self.config["domain"]
        self.shape = tuple(int(v) for v in domain["grid_shape"])
        self.nx, self.ny, self.nz = self.shape
        self.low = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.high = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.L = self.high - self.low
        # Uniform grid spacing in all directions
        self.h = float(self.L[0] / self.nx)
        self.cell_volume = self.h ** 3

        # Cell-centered coordinates
        x = np.linspace(
            self.low[0] + 0.5 * self.h, self.high[0] - 0.5 * self.h, self.nx
        )
        y = np.linspace(
            self.low[1] + 0.5 * self.h, self.high[1] - 0.5 * self.h, self.ny
        )
        z = np.linspace(
            self.low[2] + 0.5 * self.h, self.high[2] - 0.5 * self.h, self.nz
        )
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        # State fields from the scene-provided initial condition
        self.density = np.asarray(self.frame0.fields["density"], dtype=np.float64).copy()
        self.velocity = (
            np.asarray(self.frame0.fields["velocity"], dtype=np.float64).copy()
        )
        self.t = 0.0
        self.cumulative_mass_source = 0.0

        # Source term (constant in time)
        self._build_source_field()

        # Physical parameters
        self.buoyancy_strength = 2.0  # m/s^2 per unit density
        self.drag_coefficient = 1.0  # 1/s
        self.scalar_diffusivity = 1e-4  # m^2/s
        self.kinematic_viscosity = 1e-4  # m^2/s
        self.decay_rate = np.log(2.0) / float(
            self.config["initial_conditions"]["density_half_life"]
        )

        # Static source-point state carried from the initial frame
        self.position = np.asarray(self.frame0.position, dtype=np.float64).copy()
        self.point_velocity = np.asarray(self.frame0.velocity, dtype=np.float64).copy()
        self.mass = (
            np.asarray(self.frame0.mass, dtype=np.float64).copy()
            if self.frame0.mass is not None
            else np.array([1.0], dtype=np.float64)
        )

    def _build_source_field(self) -> None:
        """Build the constant-in-time prescribed smoke source rate field."""
        src = self.config["initial_conditions"]["smoke_source"]
        cx, cy, cz = src["center"]
        radius = float(src["radius"])
        half_height = float(src["half_height"])
        peak_density = float(src["peak_density"])

        radial2 = (
            (self.X - cx) ** 2 + (self.Y - cy) ** 2
        ) / (radius * radius)
        vertical2 = ((self.Z - cz) / half_height) ** 2
        self.source_mask = np.exp(-2.5 * (radial2 + vertical2))
        # Prescribed scalar source rate (density per unit time)
        self.source_rate = self.source_mask * (peak_density / 1.5)

    def _apply_velocity_boundary_conditions(self, vel: np.ndarray) -> np.ndarray:
        """Enforce closed-wall free-slip conditions on the velocity field."""
        # x_min closed: zero x-component
        vel[0, :, :, 0] = 0.0
        # y_min and y_max closed: zero y-component
        vel[:, 0, :, 1] = 0.0
        vel[:, -1, :, 1] = 0.0
        # z_min closed: zero z-component
        vel[:, :, 0, 2] = 0.0
        return vel

    def _interpolate_field(
        self, field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Trilinear interpolation of a scalar field to physical positions."""
        ix = (x - self.low[0]) / self.h - 0.5
        iy = (y - self.low[1]) / self.h - 0.5
        iz = (z - self.low[2]) / self.h - 0.5
        coords = np.stack([ix, iy, iz], axis=0)
        return map_coordinates(field, coords, order=1, mode="constant", cval=0.0)

    def _advect_scalar(self, scalar: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection of a scalar field."""
        xback = self.X - dt * self.velocity[..., 0]
        yback = self.Y - dt * self.velocity[..., 1]
        zback = self.Z - dt * self.velocity[..., 2]

        # Open outflow boundaries (x_max and z_max)
        open_mask = (xback > self.high[0]) | (zback > self.high[2])

        # Closed walls act as no-flux: clamp trace-back positions
        xb = np.clip(xback, self.low[0], self.high[0])
        yb = np.clip(yback, self.low[1], self.high[1])
        zb = np.clip(zback, self.low[2], self.high[2])

        result = self._interpolate_field(scalar, xb, yb, zb)
        result[open_mask] = 0.0
        return result

    def _advect_velocity(self, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection of the collocated velocity field."""
        xback = self.X - dt * self.velocity[..., 0]
        yback = self.Y - dt * self.velocity[..., 1]
        zback = self.Z - dt * self.velocity[..., 2]

        open_mask = (xback > self.high[0]) | (zback > self.high[2])

        xb = np.clip(xback, self.low[0], self.high[0])
        yb = np.clip(yback, self.low[1], self.high[1])
        zb = np.clip(zback, self.low[2], self.high[2])

        vu = self._interpolate_field(self.velocity[..., 0], xb, yb, zb)
        vv = self._interpolate_field(self.velocity[..., 1], xb, yb, zb)
        vw = self._interpolate_field(self.velocity[..., 2], xb, yb, zb)

        # Outflow carries the velocity out to the ambient (zero) state
        vu[open_mask] = 0.0
        vv[open_mask] = 0.0
        vw[open_mask] = 0.0

        new_vel = np.stack([vu, vv, vw], axis=-1)
        return self._apply_velocity_boundary_conditions(new_vel)

    def _diffuse(self, field: np.ndarray, coeff: float, dt: float) -> np.ndarray:
        """Explicit Laplacian diffusion applied to interior cells only."""
        lap = np.zeros_like(field)
        # x-direction
        lap[1:-1, :, :] += (
            field[2:, :, :] + field[:-2, :, :] - 2.0 * field[1:-1, :, :]
        ) / (self.h * self.h)
        # y-direction
        lap[:, 1:-1, :] += (
            field[:, 2:, :] + field[:, :-2, :] - 2.0 * field[:, 1:-1, :]
        ) / (self.h * self.h)
        # z-direction
        lap[:, :, 1:-1] += (
            field[:, :, 2:] + field[:, :, :-2] - 2.0 * field[:, :, 1:-1]
        ) / (self.h * self.h)
        return field + dt * coeff * lap

    def _substep(self, dt: float) -> None:
        """Advance density and velocity by one sub-step."""
        # 1. Inject smoke at the source
        source_added = self.source_rate * dt
        self.density += source_added
        self.cumulative_mass_source += float(np.sum(source_added)) * self.cell_volume

        # 2. Apply buoyancy (upward) and drag to the velocity field
        self.velocity[..., 2] += dt * self.buoyancy_strength * self.density
        self.velocity *= np.exp(-self.drag_coefficient * dt)
        self.velocity = self._apply_velocity_boundary_conditions(self.velocity)

        # 3. Advect velocity and density
        self.velocity = self._advect_velocity(dt)
        self.density = self._advect_scalar(self.density, dt)

        # 4. Density decay (prescribed half-life)
        self.density *= np.exp(-self.decay_rate * dt)

        # 5. Small diffusion for numerical stability and smoothness
        if self.scalar_diffusivity > 0.0:
            self.density = self._diffuse(self.density, self.scalar_diffusivity, dt)
        if self.kinematic_viscosity > 0.0:
            for d in range(3):
                self.velocity[..., d] = self._diffuse(
                    self.velocity[..., d], self.kinematic_viscosity, dt
                )
            self.velocity = self._apply_velocity_boundary_conditions(self.velocity)

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.t
        if dt < 0.0:
            raise ValueError("advance_to() called with a decreasing time")
        if dt > 0.0:
            # Sub-step for advective stability; cap at ~0.01 s per sub-step
            n_sub = max(1, int(np.ceil(dt / 0.01)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                self._substep(sub_dt)
        self.t = time
        return self._make_frame()

    def _make_frame(self) -> StateFrame:
        total_mass = float(np.sum(self.density)) * self.cell_volume

        # Active mask: interior cells where the scalar transport equation applies.
        # Closed-wall boundary cells are marked inactive; open outflow faces remain active.
        active_mask = np.ones(self.shape, dtype=np.float64)
        active_mask[0, :, :] = 0.0
        active_mask[:, 0, :] = 0.0
        active_mask[:, -1, :] = 0.0
        active_mask[:, :, 0] = 0.0

        return StateFrame(
            time=self.t,
            position=self.position.copy(),
            velocity=self.point_velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": np.array([[total_mass]], dtype=np.float64),
            },
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": np.array(
                    [self.cumulative_mass_source], dtype=np.float64
                ),
                "scalar_source": self.source_rate.copy(),
                "transported_scalar": self.density.copy(),
                "velocity_field": self.velocity.copy(),
                # Also keep the names expected by the scene_geometry binding.
                "density": self.density.copy(),
                "velocity": self.velocity.copy(),
            },
            scalars={
                "characteristic_length": float(
                    self.config["initial_conditions"]["smoke_source"]["radius"]
                ),
                "grid_spacing": self.h,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "buoyant_smoke_semi_lagrangian",
            "grid_shape": self.shape,
            "time": self.t,
            "cumulative_mass_source": self.cumulative_mass_source,
        }
