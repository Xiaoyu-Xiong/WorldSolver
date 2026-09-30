"""Benchmark solver for the cylindrical centrifuge axial particle-bands scene.

The cylinder is treated as static in the rotating frame attached to the drum.
Particles are advanced with a stable explicit time step using:

* gravity
* centrifugal and Coriolis fictitious forces from the prescribed rotation
* viscous Stokes drag from the co-rotating carrier fluid
* a simple hydrodynamic coupling that drives axial concentration fluctuations
  toward species-rich bands
* reflective cylinder-wall and end-cap constraints

All required StateFrame fields are produced at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Initial frame from the benchmark scene
        frame0 = scene.initial_state()

        # Per-particle state
        self.time = 0.0
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.body_id = frame0.body_id.copy()
        self.species = frame0.attributes["species_id"].copy()
        self.radius = frame0.attributes["particle_radius"].copy()
        self.mass_quantity = frame0.attributes["mass_quantity"].copy()

        # Scene geometry / physics parameters
        c = self.cfg["centrifuge"]
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.z0 = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.h = float(c["spacing"])
        self.rho_p = float(c["particle_density"])
        self.species_count = int(c["species_count"])
        self.g = np.array(self.cfg["gravity"], dtype=float)
        self.grid_shape = tuple(c["field_grid_shape"])

        # Viscous Stokes drag coefficient per particle (1/s)
        eta = 50.0  # Pa s -- viscous carrier fluid
        self.drag = (6.0 * np.pi * eta * self.radius) / self.mass

        # Axial hydrodynamic mobility: larger particles are slightly more mobile
        self.mobility = np.where(self.species == 0, 0.08, 0.12)  # m/s per unit gradient
        self.tau_axial = 0.05  # s, relaxation time to local drift velocity

        # 1-D axial bins for the hydrodynamic concentration field
        self.nx_bins = 40
        self.x_edges = np.linspace(-self.L / 2.0, self.L / 2.0, self.nx_bins + 1)
        self.dx = self.L / self.nx_bins

        # Reuse the frozen scalar / field skeleton
        self.active_mask = frame0.fields["active_mask"].copy()
        self.velocity_field = frame0.fields["velocity_field"].copy()
        self.cumulative_mass_source = frame0.fields["cumulative_mass_source"].copy()
        self.grid_spacing = float(frame0.scalars["grid_spacing"])
        self.characteristic_length = float(frame0.scalars["characteristic_length"])
        self.contact_length_scale = float(
            frame0.scalars.get("contact_length_scale", self.grid_spacing)
        )

        # Time-stepping control
        self.dt = 0.001  # s
        self._last_time = 0.0
        self._cached_frame: StateFrame | None = None

        # Caches for force diagnostics (updated each advance_to call)
        self._external_force = np.zeros_like(self.position)
        self._interaction_force = np.zeros((self.species_count, 3), dtype=float)

    def advance_to(self, t: float) -> StateFrame:
        t = float(t)
        if t <= self.time:
            return self._make_frame()

        elapsed = t - self.time
        n_steps = max(1, int(np.ceil(elapsed / self.dt)))
        contact_impulse = np.zeros((self.species_count, 3), dtype=float)

        for i in range(n_steps):
            # Last sub-step is shortened to land exactly on the requested time
            sub_dt = self.dt if i < n_steps - 1 else elapsed - i * self.dt
            if sub_dt <= 0.0:
                continue
            contact_impulse += self._step(sub_dt)

        self.time = t
        self._interaction_force = contact_impulse / elapsed if elapsed > 0.0 else np.zeros((self.species_count, 3))
        self._cached_frame = self._make_frame()
        return self._cached_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_dt": self.dt,
            "particle_count": self.position.shape[0],
            "species_count": self.species_count,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _make_frame(self) -> StateFrame:
        # Recompute external force at the final state for the trajectory sample
        self._external_force = self.mass[:, None] * self._acceleration(
            self.position, self.velocity, include_axial=True
        )

        # Diagnostic velocity field: deposit particle velocities onto the grid
        self.velocity_field = self._deposit_velocity_field(self.position, self.velocity)

        attributes = {
            "species_id": self.species.copy(),
            "particle_radius": self.radius.copy(),
            "mass_quantity": self.mass_quantity.copy(),
        }

        fields = {
            "active_mask": self.active_mask.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "external_force": self._external_force.copy(),
            "interaction_force": self._interaction_force.copy(),
            "velocity_field": self.velocity_field.copy(),
        }

        scalars = {
            "grid_spacing": self.grid_spacing,
            "characteristic_length": self.characteristic_length,
            "contact_length_scale": self.contact_length_scale,
        }

        return StateFrame(
            self.time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def _step(self, dt: float) -> np.ndarray:
        """Advance one sub-step and return the species-summed contact impulse."""
        # Acceleration including axial hydrodynamic coupling
        acc = self._acceleration(self.position, self.velocity, include_axial=True)

        # Store velocity before boundary correction
        self.velocity = self.velocity + acc * dt
        v_before = self.velocity.copy()

        # Advect positions
        self.position = self.position + self.velocity * dt

        # Enforce cylindrical and end-cap constraints
        self._enforce_boundaries()

        # Contact impulse from the wall (change in momentum due to boundary correction)
        impulse_per_particle = self.mass[:, None] * (self.velocity - v_before)
        contact_impulse = np.zeros((self.species_count, 3), dtype=float)
        for k in range(self.species_count):
            mask = self.species == k
            if np.any(mask):
                contact_impulse[k] = impulse_per_particle[mask].sum(axis=0)

        return contact_impulse

    def _acceleration(self, pos: np.ndarray, vel: np.ndarray, *, include_axial: bool = True) -> np.ndarray:
        """Return per-particle acceleration (excluding wall contact)."""
        N = pos.shape[0]
        acc = np.zeros((N, 3), dtype=float)

        # Gravity
        acc += self.g

        # Centrifugal acceleration in the rotating frame: omega^2 * r_perp
        r_vec = np.zeros_like(pos)
        r_vec[:, 1] = pos[:, 1]
        r_vec[:, 2] = pos[:, 2] - self.z0
        acc += (self.omega ** 2) * r_vec

        # Coriolis acceleration: -2 * omega x v  (omega along +x)
        acc[:, 1] += -2.0 * self.omega * vel[:, 2]
        acc[:, 2] += 2.0 * self.omega * vel[:, 1]

        # Stokes drag from the co-rotating fluid (fluid at rest in rotating frame)
        acc -= self.drag[:, None] * vel

        # Axial hydrodynamic coupling: drive particles toward species-rich axial regions
        if include_axial:
            c = self._axial_concentration()
            # gradients for each species along x
            grad = np.zeros((self.species_count, self.nx_bins), dtype=float)
            for k in range(self.species_count):
                grad[k] = np.gradient(c[k], self.dx)

            bin_idx = np.clip(
                np.floor((pos[:, 0] - (-self.L / 2.0)) / self.dx).astype(int),
                0,
                self.nx_bins - 1,
            )
            s = self.species
            other = 1 - s
            grad_self = grad[s, bin_idx]
            grad_other = grad[other, bin_idx]

            u_d = self.mobility * (grad_self - grad_other)
            acc[:, 0] += (u_d - vel[:, 0]) / self.tau_axial

        return acc

    def _axial_concentration(self) -> np.ndarray:
        """Return smoothed, normalized 1-D concentration field for each species."""
        c = np.zeros((self.species_count, self.nx_bins), dtype=float)
        for k in range(self.species_count):
            mask = self.species == k
            if np.any(mask):
                counts, _ = np.histogram(self.position[mask, 0], bins=self.x_edges)
                c[k] = counts

        # Smooth to obtain a well-defined concentration gradient
        for k in range(self.species_count):
            if c[k].sum() > 0.0:
                c[k] = gaussian_filter1d(c[k], sigma=1.5, mode="reflect")

        # Normalize so the mean per species per bin is unity
        total = c.sum()
        if total > 0.0:
            c = c * (self.species_count * self.nx_bins / total)
        return c

    def _enforce_boundaries(self) -> None:
        """Project particles inside the cylinder and reflect wall-normal velocity."""
        # Cylindrical wall: radial constraint
        dy = self.position[:, 1]
        dz = self.position[:, 2] - self.z0
        r = np.sqrt(dy * dy + dz * dz)
        max_r = self.R - self.radius

        out = r > max_r
        if np.any(out):
            n_y = dy[out] / r[out]
            n_z = dz[out] / r[out]
            self.position[out, 1] = n_y * max_r[out]
            self.position[out, 2] = self.z0 + n_z * max_r[out]

            # Damp the wall-normal velocity component
            v_n = self.velocity[out, 1] * n_y + self.velocity[out, 2] * n_z
            restitution = 0.2
            self.velocity[out, 1] -= (1.0 + restitution) * v_n * n_y
            self.velocity[out, 2] -= (1.0 + restitution) * v_n * n_z

        # End caps
        x_min = -self.L / 2.0
        x_max = self.L / 2.0
        hit_left = self.position[:, 0] < x_min
        hit_right = self.position[:, 0] > x_max
        if np.any(hit_left):
            self.position[hit_left, 0] = x_min
            self.velocity[hit_left, 0] = -0.2 * self.velocity[hit_left, 0]
        if np.any(hit_right):
            self.position[hit_right, 0] = x_max
            self.velocity[hit_right, 0] = -0.2 * self.velocity[hit_right, 0]

    def _deposit_velocity_field(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Deposit particle velocities onto the diagnostic Cartesian grid."""
        nx, ny, nz = self.grid_shape
        field = np.zeros((nx, ny, nz, 3), dtype=float)
        counts = np.zeros((nx, ny, nz), dtype=float)

        # Domain that contains the cylinder
        x_min, x_max = -self.L / 2.0, self.L / 2.0
        y_min, y_max = -self.R, self.R
        z_min, z_max = self.z0 - self.R, self.z0 + self.R

        ix = np.clip(np.floor((pos[:, 0] - x_min) / (x_max - x_min) * nx).astype(int), 0, nx - 1)
        iy = np.clip(np.floor((pos[:, 1] - y_min) / (y_max - y_min) * ny).astype(int), 0, ny - 1)
        iz = np.clip(np.floor((pos[:, 2] - z_min) / (z_max - z_min) * nz).astype(int), 0, nz - 1)

        np.add.at(counts, (ix, iy, iz), 1.0)
        for d in range(3):
            np.add.at(field[..., d], (ix, iy, iz), vel[:, d])

        mask = counts > 0.0
        if np.any(mask):
            field[mask] /= counts[mask, None]
        return field
