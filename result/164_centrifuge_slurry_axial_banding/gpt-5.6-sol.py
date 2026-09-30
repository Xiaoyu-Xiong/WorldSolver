"""Deterministic particle/continuum model for a rotating granular slurry.

The carrier liquid is represented on a diagnostic Eulerian grid and by its
local rigid-body velocity when coupling to the grains. Grains are inertial
particles subject to gravity, buoyancy, viscous relaxation, size-dependent
shear migration, and impacts with the rotating cylinder. Axial segregation is
a conservative spinodal constitutive law measured from particle concentration.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Vectorized fixed-step slurry solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        c = self.config["centrifuge"]
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.species = np.asarray(initial.attributes["species_id"], dtype=np.int32).copy()
        self.radii = np.asarray(initial.attributes["particle_radius"], dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.radius = float(c["radius"])
        self.length = float(c["length"])
        self.center_z = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.spacing = float(c["spacing"])
        self.grid_shape = tuple(int(v) for v in c["field_grid_shape"])
        self.species_count = int(c["species_count"])
        self.particle_density = float(c["particle_density"])
        self.fluid_density = 1000.0
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)
        self.time = 0.0
        self.max_step = 0.005

        r0 = float(np.min(self.radii))
        size_ratio = self.radii / max(r0, 1.0e-12)
        self.response_time = 0.040 * size_ratio * size_ratio
        self.nx = self.grid_shape[0]
        self.dx = self.length / self.nx
        self.ch_a = 1.0
        self.ch_b = 1.0
        self.ch_kappa = 0.010
        self.ch_mobility = 0.16

        count = len(self.mass)
        self.mass_quantity = np.zeros((count, self.species_count), dtype=np.float64)
        self.mass_quantity[np.arange(count), self.species] = self.mass
        self.cumulative_mass_source = np.zeros(self.species_count, dtype=np.float64)
        self.active_mask, self.grid_y, self.grid_z = self._make_grid_geometry()
        self.last_phi = np.zeros(self.nx, dtype=np.float64)
        self.last_mu_gradient = np.zeros(self.nx, dtype=np.float64)
        self.last_interaction = np.zeros((self.species_count, 3), dtype=np.float64)
        self.step_count = 0

    @staticmethod
    def _reflective_smooth(values: np.ndarray, passes: int = 2) -> np.ndarray:
        out = np.asarray(values, dtype=np.float64).copy()
        for _ in range(passes):
            padded = np.pad(out, 1, mode="edge")
            out = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
        return out

    def _axial_state(self) -> tuple[np.ndarray, np.ndarray]:
        indices = np.floor((self.position[:, 0] + 0.5 * self.length) / self.dx).astype(np.int64)
        indices = np.clip(indices, 0, self.nx - 1)
        n0 = np.bincount(indices[self.species == 0], minlength=self.nx).astype(np.float64)
        n1 = np.bincount(indices[self.species == 1], minlength=self.nx).astype(np.float64)
        n0 = self._reflective_smooth(n0)
        n1 = self._reflective_smooth(n1)
        phi = (n1 - n0) / np.maximum(n1 + n0, 1.0)
        pp = np.pad(phi, 1, mode="edge")
        laplacian = (pp[:-2] - 2.0 * pp[1:-1] + pp[2:]) / (self.dx * self.dx)
        mu = -self.ch_a * phi + self.ch_b * phi**3 - self.ch_kappa * laplacian
        mp = np.pad(mu, 1, mode="edge")
        gradient = (mp[2:] - mp[:-2]) / (2.0 * self.dx)
        gradient[0] = (mu[1] - mu[0]) / self.dx
        gradient[-1] = (mu[-1] - mu[-2]) / self.dx
        return phi, gradient

    def _make_grid_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, ny, nz = self.grid_shape
        ys = np.linspace(-self.radius, self.radius, ny)
        zs_rel = np.linspace(-self.radius, self.radius, nz)
        yy, zz = np.meshgrid(ys, zs_rel, indexing="ij")
        cross_mask = yy * yy + zz * zz <= self.radius * self.radius
        mask = np.broadcast_to(cross_mask, self.grid_shape).copy()
        return mask, yy, zz

    def _step(self, dt: float) -> None:
        phi, mu_gradient = self._axial_state()
        xbin = np.floor((self.position[:, 0] + 0.5 * self.length) / self.dx).astype(np.int64)
        xbin = np.clip(xbin, 0, self.nx - 1)
        rel_y = self.position[:, 1]
        rel_z = self.position[:, 2] - self.center_z
        radial_distance = np.sqrt(rel_y * rel_y + rel_z * rel_z)
        safe_radius = np.maximum(radial_distance, 1.0e-10)
        radial_y = rel_y / safe_radius
        radial_z = rel_z / safe_radius

        # Local rotating-carrier velocity plus conservative axial species flux.
        carrier = np.empty_like(self.velocity)
        signs = 2.0 * self.species.astype(np.float64) - 1.0
        carrier[:, 0] = -signs * self.ch_mobility * mu_gradient[xbin]
        carrier[:, 1] = -self.omega * rel_z
        carrier[:, 2] = self.omega * rel_y

        buoyancy_accel = -(self.fluid_density / self.particle_density) * self.gravity
        migration_strength = 0.070 * self.omega**2 * radial_distance / self.radius
        other_accel = np.broadcast_to(self.gravity + buoyancy_accel, self.velocity.shape).copy()
        other_accel[:, 1] += signs * migration_strength * radial_y
        other_accel[:, 2] += signs * migration_strength * radial_z
        decay = np.exp(-dt / self.response_time)[:, None]
        tau = self.response_time[:, None]
        old_velocity = self.velocity.copy()
        self.velocity = carrier + (old_velocity - carrier) * decay + other_accel * tau * (1.0 - decay)
        self.position += dt * self.velocity

        # Inelastic end-cap collision.
        cap = 0.5 * self.length - self.radii
        high = self.position[:, 0] > cap
        low = self.position[:, 0] < -cap
        if np.any(high):
            self.position[high, 0] = cap[high]
            self.velocity[high, 0] = -0.20 * np.maximum(self.velocity[high, 0], 0.0)
        if np.any(low):
            self.position[low, 0] = -cap[low]
            self.velocity[low, 0] = -0.20 * np.minimum(self.velocity[low, 0], 0.0)

        # Side-wall collision in the rotating wall frame.
        rel_y = self.position[:, 1]
        rel_z = self.position[:, 2] - self.center_z
        rr = np.sqrt(rel_y * rel_y + rel_z * rel_z)
        allowed = self.radius - self.radii
        outside = rr > allowed
        if np.any(outside):
            idx = np.flatnonzero(outside)
            ny = rel_y[idx] / rr[idx]
            nz = rel_z[idx] / rr[idx]
            self.position[idx, 1] = ny * allowed[idx]
            self.position[idx, 2] = self.center_z + nz * allowed[idx]
            wall_y = -self.omega * nz * allowed[idx]
            wall_z = self.omega * ny * allowed[idx]
            relative_y = self.velocity[idx, 1] - wall_y
            relative_z = self.velocity[idx, 2] - wall_z
            outward_speed = relative_y * ny + relative_z * nz
            hit = outward_speed > 0.0
            if np.any(hit):
                j = idx[hit]
                impulse_speed = 1.15 * outward_speed[hit]
                self.velocity[j, 1] -= impulse_speed * ny[hit]
                self.velocity[j, 2] -= impulse_speed * nz[hit]
            self.velocity[idx, 1] = wall_y + 0.82 * (self.velocity[idx, 1] - wall_y)
            self.velocity[idx, 2] = wall_z + 0.82 * (self.velocity[idx, 2] - wall_z)
            self.velocity[idx, 0] *= 0.97

        coupling_accel = (carrier - old_velocity) / tau + buoyancy_accel
        coupling_accel[:, 1] += signs * migration_strength * radial_y
        coupling_accel[:, 2] += signs * migration_strength * radial_z
        particle_coupling = self.mass[:, None] * coupling_accel
        self.last_interaction.fill(0.0)
        for k in range(self.species_count):
            self.last_interaction[k] = particle_coupling[self.species == k].sum(axis=0)
        self.last_phi = phi
        self.last_mu_gradient = mu_gradient
        self.time += dt
        self.step_count += 1

    def _velocity_field(self) -> np.ndarray:
        field = np.zeros(self.grid_shape + (3,), dtype=np.float64)
        field[..., 1] = np.broadcast_to(-self.omega * self.grid_z, self.grid_shape)
        field[..., 2] = np.broadcast_to(self.omega * self.grid_y, self.grid_shape)
        field[..., 0] = np.broadcast_to(
            (-0.12 * self.ch_mobility * self.last_mu_gradient)[:, None, None], self.grid_shape
        )
        field[~self.active_mask] = 0.0
        return field

    def _frame(self, requested_time: float) -> StateFrame:
        external_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(requested_time), self.position.copy(), self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radii.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": external_force,
                "interaction_force": self.last_interaction.copy(),
                "velocity_field": self._velocity_field(),
            },
            scalars={
                "grid_spacing": self.spacing,
                "characteristic_length": 2.0 * self.radius,
                "contact_length_scale": self.spacing,
                "angular_velocity_rad_s": self.omega,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-10:
            raise ValueError("sample times must be nondecreasing")
        while self.time < target - 1.0e-12:
            self._step(min(self.max_step, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "inertial_particles_rotating_carrier_spinodal_banding",
            "time": float(self.time),
            "steps": int(self.step_count),
            "particle_count": int(len(self.mass)),
            "mass_total": float(np.sum(self.mass)),
        }
