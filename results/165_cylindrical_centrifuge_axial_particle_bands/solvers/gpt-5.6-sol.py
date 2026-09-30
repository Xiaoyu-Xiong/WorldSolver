"""Deterministic particle--fluid model for the horizontal centrifuge.

The carrier is represented by its prescribed solid-body velocity and pressure
acceleration. Finite-inertia particles relax toward that flow, centrifuge at
different rates according to their radii, and feed an axial suspension mode.
The latter is a conservative, regularized concentration instability: axial
flow is directed up gradients of smoothed cross-sectional particle loading,
while a short-wave term opposes grid-scale clustering.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A small Eulerian--Lagrangian centrifuge solver."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        config = scene.config
        centrifuge = config["centrifuge"]

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.species = np.asarray(initial.attributes["species_id"], dtype=np.int32).copy()
        self.radii = np.asarray(initial.attributes["particle_radius"], dtype=np.float64).copy()
        self.mass_quantity = np.asarray(
            initial.attributes["mass_quantity"], dtype=np.float64
        ).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()

        self.radius = float(centrifuge["radius"])
        self.length = float(centrifuge["length"])
        self.center_z = float(centrifuge["center_height"])
        self.omega = float(centrifuge["angular_velocity_rad_s"])
        self.spacing = float(centrifuge["spacing"])
        self.grid_shape = tuple(int(v) for v in centrifuge["field_grid_shape"])
        self.species_count = int(centrifuge["species_count"])
        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.time = 0.0

        # Water-like carrier. Radius-squared response times give the two
        # visible species measurably different radial migration rates.
        particle_density = float(centrifuge["particle_density"])
        self.density_ratio = 1000.0 / particle_density
        reference_radius = 0.38 * self.spacing
        self.response_time = 0.012 * (self.radii / reference_radius) ** 2

        # Axial finite-volume mesh used by the conservative banding model.
        self.band_cells = 32
        self.band_dx = self.length / self.band_cells
        self._initial_axial_mass = self._deposit_axial(self.mass)
        self._initial_axial_mass = np.maximum(
            self._initial_axial_mass, 0.15 * np.mean(self._initial_axial_mass)
        )
        self._axial_flow = np.zeros(self.band_cells, dtype=np.float64)

        # Static cylindrical mask and coordinates for diagnostic carrier flow.
        nx, ny, nz = self.grid_shape
        self._grid_x = np.linspace(-0.5 * self.length, 0.5 * self.length, nx)
        gy = np.linspace(-self.radius, self.radius, ny)
        gz = np.linspace(-self.radius, self.radius, nz)
        yy, zz = np.meshgrid(gy, gz, indexing="ij")
        cross_mask = yy * yy + zz * zz <= self.radius * self.radius
        self._active_mask = np.broadcast_to(cross_mask, self.grid_shape).copy()
        self._grid_y = yy
        self._grid_z = zz

        self._steps = 0
        self._max_speed = 0.0
        self._last_interaction = np.zeros((self.species_count, 3), dtype=np.float64)

    def _deposit_axial(self, weights: np.ndarray) -> np.ndarray:
        """Cloud-in-cell deposition with reflecting end cells."""
        coordinate = (self.position[:, 0] + 0.5 * self.length) / self.band_dx - 0.5
        left = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - left
        left_clipped = np.clip(left, 0, self.band_cells - 1)
        right_clipped = np.clip(left + 1, 0, self.band_cells - 1)
        result = np.bincount(
            left_clipped, weights=weights * (1.0 - fraction), minlength=self.band_cells
        )
        result += np.bincount(
            right_clipped, weights=weights * fraction, minlength=self.band_cells
        )
        return result

    @staticmethod
    def _smooth(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, (2, 2), mode="edge")
        return (
            padded[:-4]
            + 4.0 * padded[1:-3]
            + 6.0 * padded[2:-2]
            + 4.0 * padded[3:-1]
            + padded[4:]
        ) / 16.0

    def _update_axial_flow(self) -> None:
        radial_y = self.position[:, 1]
        radial_z = self.position[:, 2] - self.center_z
        radial_fraction = np.sqrt(radial_y * radial_y + radial_z * radial_z) / self.radius

        axial_mass = self._deposit_axial(self.mass)
        concentration = axial_mass / self._initial_axial_mass - 1.0

        # Cross-sectional loading perturbs axial carrier pressure and supplies
        # a physical seed after differential radial centrifuging.
        radial_mass = self._deposit_axial(self.mass * radial_fraction)
        local_radius = radial_mass / np.maximum(axial_mass, 1.0e-12)
        occupied = axial_mass > 0.2 * np.mean(axial_mass)
        mean_radius = np.sum(radial_mass) / np.sum(axial_mass)
        loading = np.where(occupied, local_radius - mean_radius, 0.0)
        signal = concentration + 1.4 * loading

        smooth_signal = self._smooth(self._smooth(signal))
        grad_smooth = np.gradient(smooth_signal, self.band_dx, edge_order=1)
        grad_raw = np.gradient(signal, self.band_dx, edge_order=1)

        # The instability only becomes appreciable when particles have built
        # up toward the outer cylinder, preserving radial-before-axial order.
        segregation = np.clip((mean_radius - 0.64) / 0.22, 0.0, 1.0)
        flow = segregation * (0.42 * grad_smooth - 0.085 * grad_raw)
        flow[0] = flow[-1] = 0.0
        self._axial_flow = np.clip(flow, -0.32, 0.32)

    def _grid_x_for_bands(self) -> np.ndarray:
        return -0.5 * self.length + (np.arange(self.band_cells) + 0.5) * self.band_dx

    def _particle_carrier_velocity(self) -> np.ndarray:
        relative_z = self.position[:, 2] - self.center_z
        carrier = np.empty_like(self.velocity)
        carrier[:, 0] = np.interp(
            self.position[:, 0], self._grid_x_for_bands(), self._axial_flow
        )
        carrier[:, 1] = -self.omega * relative_z
        carrier[:, 2] = self.omega * self.position[:, 1]
        return carrier

    def _enforce_boundaries(self) -> None:
        # End caps are stationary in the axial direction.
        axial_limit = 0.5 * self.length - self.radii
        beyond = np.abs(self.position[:, 0]) > axial_limit
        if np.any(beyond):
            side = np.sign(self.position[beyond, 0])
            self.position[beyond, 0] = side * axial_limit[beyond]
            outward = self.velocity[beyond, 0] * side
            self.velocity[beyond, 0] -= side * np.maximum(outward, 0.0) * 1.12

        y = self.position[:, 1]
        z = self.position[:, 2] - self.center_z
        radial_distance = np.sqrt(y * y + z * z)
        radial_limit = self.radius - self.radii
        contact = radial_distance > radial_limit
        if not np.any(contact):
            return

        indices = np.flatnonzero(contact)
        distance = radial_distance[indices]
        normal_y = y[indices] / distance
        normal_z = z[indices] / distance
        self.position[indices, 1] = normal_y * radial_limit[indices]
        self.position[indices, 2] = self.center_z + normal_z * radial_limit[indices]

        # Collision response is evaluated relative to the rotating wall.
        wall_y = -self.omega * normal_z * radial_limit[indices]
        wall_z = self.omega * normal_y * radial_limit[indices]
        relative_y = self.velocity[indices, 1] - wall_y
        relative_z = self.velocity[indices, 2] - wall_z
        outward_speed = relative_y * normal_y + relative_z * normal_z
        impulse_speed = np.maximum(outward_speed, 0.0)
        self.velocity[indices, 1] -= 1.10 * impulse_speed * normal_y
        self.velocity[indices, 2] -= 1.10 * impulse_speed * normal_z

        # Viscous no-slip tendency at sustained wall contact.
        self.velocity[indices, 1] += 0.08 * (wall_y - self.velocity[indices, 1])
        self.velocity[indices, 2] += 0.08 * (wall_z - self.velocity[indices, 2])

    def _step(self, dt: float) -> None:
        self._update_axial_flow()
        carrier = self._particle_carrier_velocity()

        relative = self.position.copy()
        relative[:, 0] = 0.0
        relative[:, 2] -= self.center_z
        pressure_acceleration = -(self.density_ratio * self.omega**2) * relative
        buoyant_gravity = (1.0 - self.density_ratio) * self.gravity
        acceleration = (carrier - self.velocity) / self.response_time[:, None]
        acceleration += pressure_acceleration
        acceleration += buoyant_gravity

        # Semi-implicit velocity/position update; dt is below both response
        # times and a small fraction of a drum revolution.
        self.velocity += dt * acceleration
        self.position += dt * self.velocity
        self._enforce_boundaries()
        self.time += dt
        self._steps += 1
        speed = float(np.max(np.linalg.norm(self.velocity, axis=1)))
        self._max_speed = max(self._max_speed, speed)

    def _force_diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
        external = self.mass[:, None] * self.gravity[None, :]
        carrier = self._particle_carrier_velocity()
        relative = self.position.copy()
        relative[:, 0] = 0.0
        relative[:, 2] -= self.center_z
        fluid_acceleration = (carrier - self.velocity) / self.response_time[:, None]
        fluid_acceleration -= self.density_ratio * self.omega**2 * relative
        fluid_acceleration -= self.density_ratio * self.gravity
        particle_interaction = self.mass[:, None] * fluid_acceleration
        interaction = np.zeros((self.species_count, 3), dtype=np.float64)
        np.add.at(interaction, self.species, particle_interaction)
        self._last_interaction = interaction
        return external, interaction

    def _velocity_field(self) -> np.ndarray:
        nx, ny, nz = self.grid_shape
        field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        axial = np.interp(self._grid_x, self._grid_x_for_bands(), self._axial_flow)
        field[..., 0] = axial[:, None, None]
        field[..., 1] = -self.omega * self._grid_z[None, :, :]
        field[..., 2] = self.omega * self._grid_y[None, :, :]
        field *= self._active_mask[..., None]
        return field

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")

        while self.time < target - 1.0e-12:
            dt = min(0.004, target - self.time)
            self._step(dt)
        self.time = target

        # Refresh state-dependent fields even for the initial t=0 sample.
        self._update_axial_flow()
        external, interaction = self._force_diagnostics()
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radii.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "active_mask": self._active_mask.copy(),
                "cumulative_mass_source": np.zeros(self.species_count, dtype=np.float64),
                "external_force": external,
                "interaction_force": interaction,
                "velocity_field": self._velocity_field(),
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.spacing,
                "contact_length_scale": self.spacing,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "inertial_particles_rotating_carrier_axial_finite_volume",
            "internal_steps": int(self._steps),
            "maximum_particle_speed": float(self._max_speed),
            "particle_count": int(len(self.position)),
        }
