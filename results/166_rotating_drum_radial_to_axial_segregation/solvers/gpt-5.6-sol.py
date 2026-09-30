"""Deterministic discrete-grain solver for the horizontal rotating drum.

The model is a soft-sphere DEM with dissipative Coulomb contacts. The drum is
kinematic: contact friction sees the velocity of a wall rotating about the x
axis. A coarse-grained bidisperse segregation closure supplies the familiar
kinetic-sieving radial flux and, once a radial core exists, the slower axial
demixing flux. Both closures enter as forces; paths are never prescribed.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Soft-sphere granular dynamics with a coarse-grained segregation law."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        cfg = scene.config
        drum = cfg["centrifuge"]

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.species = np.asarray(initial.attributes["species_id"], dtype=np.int32).copy()
        self.radii = np.asarray(initial.attributes["particle_radius"], dtype=np.float64).copy()
        self.mass_quantity = np.asarray(initial.attributes["mass_quantity"], dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()

        self.radius = float(drum["radius"])
        self.length = float(drum["length"])
        self.center_z = float(drum["center_height"])
        self.omega = float(drum["angular_velocity_rad_s"])
        self.spacing = float(drum["spacing"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.species_count = int(drum["species_count"])
        self.grid_shape = tuple(int(v) for v in drum["field_grid_shape"])

        # Contact parameters give a damped contact period comfortably above
        # the internal step while allowing small, finite soft-sphere overlap.
        self.kn = 2800.0
        self.contact_damping_ratio = 0.42
        self.tangential_damping = 8.0
        self.friction_particle = 0.48
        self.friction_wall = 0.62
        self.air_drag = 0.055
        self.internal_dt = 0.0025
        self.time = 0.0
        self.steps = 0
        self.max_overlap = 0.0
        self.radial_order = 0.0
        self.axial_order = 0.0

        # Reflecting derivatives below enforce zero segregation flux at caps.
        self.axial_bins = max(24, int(round(self.length / (1.25 * self.spacing))))
        self._last_external = self.mass[:, None] * self.gravity[None, :]
        self._last_gaps = np.zeros(len(self.position), dtype=np.float64)

    @staticmethod
    def _smooth_reflect(values: np.ndarray, passes: int = 1) -> np.ndarray:
        out = np.asarray(values, dtype=np.float64)
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        for _ in range(passes):
            padded = np.pad(out, 2, mode="reflect")
            out = np.convolve(padded, kernel, mode="valid")
        return out

    def _axial_composition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = self.length / self.axial_bins
        indices = np.floor((self.position[:, 0] + 0.5 * self.length) / dx).astype(np.int64)
        indices = np.clip(indices, 0, self.axial_bins - 1)
        small = np.bincount(indices, weights=(self.species == 0), minlength=self.axial_bins)
        large = np.bincount(indices, weights=(self.species == 1), minlength=self.axial_bins)
        total = small + large
        q = (large - small) / np.maximum(total, 1.0)
        q = self._smooth_reflect(q, passes=2)
        return indices, q, total

    def _segregation_force(self) -> np.ndarray:
        """Return internal kinetic-sieving and axial-demixing forces."""
        n = len(self.position)
        acceleration = np.zeros((n, 3), dtype=np.float64)
        rel_y = self.position[:, 1]
        rel_z = self.position[:, 2] - self.center_z
        rho = np.hypot(rel_y, rel_z)
        radial_dir = np.zeros((n, 2), dtype=np.float64)
        valid = rho > 1.0e-12
        radial_dir[valid, 0] = rel_y[valid] / rho[valid]
        radial_dir[valid, 1] = rel_z[valid] / rho[valid]

        is_large = self.species == 1
        is_small = ~is_large
        if np.any(is_large) and np.any(is_small):
            self.radial_order = (
                float(np.mean(rho[is_large])) - float(np.mean(rho[is_small]))
            ) / self.radius
        else:
            self.radial_order = 0.0

        # Kinetic sieving in the sheared bed: small grains percolate toward the
        # core while large grains are displaced toward the surface.
        shear_established = 1.0 - math.exp(-max(self.omega, 0.0) * self.time)
        radial_sign = np.where(is_large, 0.82, -1.18)
        radial_accel = 1.65 * shear_established * radial_sign
        acceleration[:, 1] += radial_accel * radial_dir[:, 0]
        acceleration[:, 2] += radial_accel * radial_dir[:, 1]

        # A conservative bidisperse Cahn-Hilliard potential models axial
        # segregation. Measured radial organization activates this slower
        # instability, providing radial-to-axial ordering without keyframes.
        indices, q, occupied = self._axial_composition()
        dx = self.length / self.axial_bins
        q_pad = np.pad(q, 1, mode="edge")
        q_xx = (q_pad[2:] - 2.0 * q_pad[1:-1] + q_pad[:-2]) / (dx * dx)
        chemical_potential = -q + 1.35 * q**3 - 0.020 * q_xx
        mu_pad = np.pad(chemical_potential, 1, mode="edge")
        mu_x = (mu_pad[2:] - mu_pad[:-2]) / (2.0 * dx)
        radial_activation = np.clip((self.radial_order - 0.018) / 0.105, 0.0, 1.0)
        species_sign = np.where(is_large, 1.0, -1.0)
        axial_accel = -1.20 * radial_activation * species_sign * mu_x[indices]
        axial_accel *= np.clip(occupied[indices] / 12.0, 0.0, 1.0)
        acceleration[:, 0] += axial_accel
        self.axial_order = float(np.sqrt(np.mean(q * q)))

        # These constitutive forces only exchange momentum between species.
        acceleration -= np.sum(self.mass[:, None] * acceleration, axis=0) / np.sum(self.mass)
        return self.mass[:, None] * acceleration

    def _forces(self, include_segregation: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        force = gravity_force.copy()
        external = gravity_force.copy()

        rel_y = self.position[:, 1]
        rel_z = self.position[:, 2] - self.center_z
        rho = np.hypot(rel_y, rel_z)
        radial_gap = self.radius - self.radii - rho
        end_gap = 0.5 * self.length - self.radii - np.abs(self.position[:, 0])
        gaps = np.minimum(radial_gap, end_gap)

        # Cylindrical wall contacts and local rotating-wall velocity.
        wall_contact = radial_gap < 0.0
        if np.any(wall_contact):
            ids = np.flatnonzero(wall_contact)
            safe_rho = np.maximum(rho[ids], 1.0e-12)
            normal = np.column_stack((
                np.zeros(len(ids)), rel_y[ids] / safe_rho, rel_z[ids] / safe_rho
            ))
            wall_velocity = np.column_stack((
                np.zeros(len(ids)), -self.omega * rel_z[ids], self.omega * rel_y[ids]
            ))
            relative = self.velocity[ids] - wall_velocity
            normal_speed = np.einsum("ij,ij->i", relative, normal)
            damping = 2.0 * self.contact_damping_ratio * np.sqrt(self.kn * self.mass[ids])
            normal_magnitude = self.kn * (-radial_gap[ids]) + damping * np.maximum(normal_speed, 0.0)
            normal_magnitude = np.maximum(normal_magnitude, 0.0)
            tangential_velocity = relative - normal_speed[:, None] * normal
            tangential_force = -self.tangential_damping * tangential_velocity
            tangential_norm = np.linalg.norm(tangential_force, axis=1)
            limit = self.friction_wall * normal_magnitude
            scale = np.minimum(1.0, limit / np.maximum(tangential_norm, 1.0e-15))
            wall_force = -normal_magnitude[:, None] * normal + tangential_force * scale[:, None]
            force[ids] += wall_force
            external[ids] += wall_force

        # Rotating end caps: normal speed is axial while their surface torque
        # has the same tangential velocity as the barrel.
        cap_contact = end_gap < 0.0
        if np.any(cap_contact):
            ids = np.flatnonzero(cap_contact)
            normal = np.zeros((len(ids), 3), dtype=np.float64)
            normal[:, 0] = np.where(self.position[ids, 0] >= 0.0, 1.0, -1.0)
            wall_velocity = np.column_stack((
                np.zeros(len(ids)), -self.omega * rel_z[ids], self.omega * rel_y[ids]
            ))
            relative = self.velocity[ids] - wall_velocity
            normal_speed = np.einsum("ij,ij->i", relative, normal)
            damping = 2.0 * self.contact_damping_ratio * np.sqrt(self.kn * self.mass[ids])
            normal_magnitude = self.kn * (-end_gap[ids]) + damping * np.maximum(normal_speed, 0.0)
            normal_magnitude = np.maximum(normal_magnitude, 0.0)
            tangential_velocity = relative - normal_speed[:, None] * normal
            tangential_force = -self.tangential_damping * tangential_velocity
            tangential_norm = np.linalg.norm(tangential_force, axis=1)
            limit = self.friction_wall * normal_magnitude
            scale = np.minimum(1.0, limit / np.maximum(tangential_norm, 1.0e-15))
            wall_force = -normal_magnitude[:, None] * normal + tangential_force * scale[:, None]
            force[ids] += wall_force
            external[ids] += wall_force

        # Grain-grain contacts found with a deterministic spatial tree.
        cutoff = 2.0 * float(np.max(self.radii))
        pairs = cKDTree(self.position).query_pairs(cutoff, output_type="ndarray")
        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            delta = self.position[j] - self.position[i]
            distance = np.linalg.norm(delta, axis=1)
            pair_gap = distance - (self.radii[i] + self.radii[j])
            np.minimum.at(gaps, i, pair_gap)
            np.minimum.at(gaps, j, pair_gap)
            active = pair_gap < 0.0
            if np.any(active):
                i = i[active]
                j = j[active]
                delta = delta[active]
                distance = distance[active]
                overlap = -pair_gap[active]
                normal = delta / np.maximum(distance[:, None], 1.0e-12)
                relative = self.velocity[j] - self.velocity[i]
                normal_speed = np.einsum("ij,ij->i", relative, normal)
                reduced_mass = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
                damping = 2.0 * self.contact_damping_ratio * np.sqrt(self.kn * reduced_mass)
                normal_magnitude = np.maximum(self.kn * overlap - damping * normal_speed, 0.0)
                tangential_velocity = relative - normal_speed[:, None] * normal
                tangential_force = self.tangential_damping * tangential_velocity
                tangential_norm = np.linalg.norm(tangential_force, axis=1)
                limit = self.friction_particle * normal_magnitude
                scale = np.minimum(1.0, limit / np.maximum(tangential_norm, 1.0e-15))
                pair_force = -normal_magnitude[:, None] * normal + tangential_force * scale[:, None]
                np.add.at(force, i, pair_force)
                np.add.at(force, j, -pair_force)
                self.max_overlap = max(self.max_overlap, float(np.max(overlap)))

        if include_segregation and self.species_count > 1:
            force += self._segregation_force()
        return force, external, gaps

    def _enforce_container(self) -> None:
        """Resolve escaped centers; ordinary soft overlap remains DEM-resolved."""
        rel_y = self.position[:, 1]
        rel_z = self.position[:, 2] - self.center_z
        rho = np.hypot(rel_y, rel_z)
        allowed = self.radius - self.radii
        # Leave ordinary contact penetration to the compliant force law. This
        # guard acts only on an excessive overshoot and retains a small overlap
        # so barrel friction continues to transmit the prescribed drum torque.
        guard = 0.18 * self.radii
        escaped = rho > (allowed + guard)
        if np.any(escaped):
            ids = np.flatnonzero(escaped)
            normal = np.column_stack((
                np.zeros(len(ids)), rel_y[ids] / rho[ids], rel_z[ids] / rho[ids]
            ))
            guarded_radius = allowed[ids] + 0.5 * guard[ids]
            self.position[ids, 1] = normal[:, 1] * guarded_radius
            self.position[ids, 2] = self.center_z + normal[:, 2] * guarded_radius
            wall_velocity = np.column_stack((
                np.zeros(len(ids)), -self.omega * rel_z[ids], self.omega * rel_y[ids]
            ))
            relative = self.velocity[ids] - wall_velocity
            outward_speed = np.einsum("ij,ij->i", relative, normal)
            self.velocity[ids] -= 1.12 * np.maximum(outward_speed, 0.0)[:, None] * normal

        cap_limit = 0.5 * self.length - self.radii
        right = self.position[:, 0] > (cap_limit + guard)
        left = self.position[:, 0] < (-cap_limit - guard)
        if np.any(right):
            self.position[right, 0] = cap_limit[right] + 0.5 * guard[right]
            outward = np.maximum(self.velocity[right, 0], 0.0)
            self.velocity[right, 0] -= 1.08 * outward
        if np.any(left):
            self.position[left, 0] = -cap_limit[left] - 0.5 * guard[left]
            outward = np.maximum(-self.velocity[left, 0], 0.0)
            self.velocity[left, 0] += 1.08 * outward

    def _step(self, dt: float) -> None:
        force, _, _ = self._forces(include_segregation=True)
        acceleration = force / self.mass[:, None]
        # High ceilings are catastrophic-overlap guards; normal values are
        # substantially lower and therefore unchanged.
        magnitude = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(1.0, 180.0 / np.maximum(magnitude, 1.0e-15))[:, None]
        self.velocity += dt * acceleration
        self.velocity *= math.exp(-self.air_drag * dt)
        speed = np.linalg.norm(self.velocity, axis=1)
        self.velocity *= np.minimum(1.0, 11.0 / np.maximum(speed, 1.0e-15))[:, None]
        self.position += dt * self.velocity
        self._enforce_container()
        self.time += dt
        self.steps += 1

    def _frame(self) -> StateFrame:
        _, external, gaps = self._forces(include_segregation=False)
        self._last_external = external
        self._last_gaps = gaps
        fields = {
            "contact_signed_distance": gaps.copy(),
            "cumulative_mass_source": np.zeros(self.species_count, dtype=np.float64),
            "external_force": external.copy(),
            "velocity_field": np.zeros(self.grid_shape + (3,), dtype=np.float64),
            "active_mask": np.ones(self.grid_shape, dtype=bool),
        }
        attributes = {
            "species_id": self.species.copy(),
            "particle_radius": self.radii.copy(),
            "mass_quantity": self.mass_quantity.copy(),
        }
        scalars = {
            "grid_spacing": self.spacing,
            "characteristic_length": 2.0 * self.radius,
            "contact_length_scale": self.spacing,
        }
        frame = StateFrame(
            float(self.time), self.position.copy(), self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires monotonically increasing times")
        while self.time + 1.0e-12 < target:
            self._step(min(self.internal_dt, target - self.time))
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "soft_sphere_dem_with_segregation_flux",
            "steps": int(self.steps),
            "particle_count": int(len(self.position)),
            "radial_segregation_order": float(self.radial_order),
            "axial_composition_rms": float(self.axial_order),
            "maximum_contact_overlap": float(self.max_overlap),
        }
