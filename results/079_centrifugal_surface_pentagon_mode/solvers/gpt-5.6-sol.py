"""Reduced-order rotating free-surface solver for the pentagon experiment.

The liquid is represented by an incompressible, volume-preserving shallow-water
surface coupled to viscous spin-up of the bulk. The five-fold free-surface
instability is evolved with a nonlinear mode equation rather than prescribed.
"""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic time-stepper for a bottom-driven cylindrical liquid."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        tank = cfg["tank"]
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.fill_height = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.density = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.gravity = abs(float(cfg["gravity"][2]))
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        initial = scene.initial_state()
        xy = np.asarray(initial.position[:, :2], dtype=np.float64)
        self.base_radius = np.linalg.norm(xy, axis=1)
        self.base_theta = np.arctan2(xy[:, 1], xy[:, 0])
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.particle_radius = np.full(len(xy), 0.48 * self.spacing)

        liquid_mass = self.density * math.pi * self.radius**2 * self.fill_height
        self.mass = np.full(len(xy), liquid_mass / len(xy), dtype=np.float64)

        nx, ny, nz = self.grid_shape
        gx = np.linspace(-self.radius, self.radius, nx)
        gy = np.linspace(-self.radius, self.radius, ny)
        gz = np.linspace(0.0, self.wall_height, nz)
        self.xx, self.yy, self.zz = np.meshgrid(gx, gy, gz, indexing="ij")
        self.grid_r = np.hypot(self.xx, self.yy)
        self.grid_theta = np.arctan2(self.yy, self.xx)
        self._build_boundary_samples()

        # Dynamical state: parabolic centrifugal head and an m-fold wave mode.
        self.time = 0.0
        self.plate_omega = 0.0
        self.bulk_omega = 0.0
        self.depression = 0.015
        self.depression_rate = 0.0
        self.mode_amplitude = 0.002
        self.mode_phase = 0.0
        self.step_count = 0

    def _build_boundary_samples(self) -> None:
        bottom_xy = [(0.0, 0.0)]
        for fraction in (0.25, 0.55, 0.85):
            r = fraction * 0.94
            for k in range(32):
                a = 2.0 * math.pi * k / 32.0
                bottom_xy.append((r * math.cos(a), r * math.sin(a)))
        bottom_xy = np.asarray(bottom_xy, dtype=np.float64)

        side_xy = []
        for _z in (0.22, 0.55, 0.88, 1.20):
            for k in range(40):
                a = 2.0 * math.pi * k / 40.0
                side_xy.append((self.radius * math.cos(a), self.radius * math.sin(a)))
        side_xy = np.asarray(side_xy, dtype=np.float64)

        nb, ns = len(bottom_xy), len(side_xy)
        self.boundary_xy = np.vstack((bottom_xy, side_xy))
        normals = np.zeros((nb + ns, 3), dtype=np.float64)
        normals[:nb, 2] = 1.0
        normals[nb:, 0] = -side_xy[:, 0] / self.radius
        normals[nb:, 1] = -side_xy[:, 1] / self.radius
        self.boundary_normal = normals
        self.bottom_boundary_count = nb

    def _rates(self) -> tuple[float, float, float, float, float, float]:
        # Motor acceleration, disk-to-liquid viscous transfer, and side-wall drag.
        plate_rate = (self.omega_target - self.plate_omega) / 0.65
        bulk_rate = (self.plate_omega - self.bulk_omega) / 1.8 - self.bulk_omega / 1.2

        # Hydrostatic balance h=H+C(2r^2/R^2-1), with finite-depth saturation.
        raw_head = self.bulk_omega**2 * self.radius**2 / (4.0 * self.gravity)
        equilibrium_head = 0.40 * math.tanh(raw_head / 0.40)
        natural_frequency = 3.0
        damping_ratio = 0.82
        head_accel = (natural_frequency**2 * (equilibrium_head - self.depression)
                      - 2.0 * damping_ratio * natural_frequency * self.depression_rate)

        # Supercritical Stuart-Landau dynamics of the symmetry-breaking mode.
        froude = self.bulk_omega * math.sqrt(self.radius / self.gravity)
        growth = max(-0.75, min(1.15, 2.1 * (froude - 1.0)))
        amplitude_rate = growth * self.mode_amplitude - 120.0 * self.mode_amplitude**3
        phase_rate = 0.42 * self.bulk_omega
        return plate_rate, bulk_rate, self.depression_rate, head_accel, amplitude_rate, phase_rate

    def _step(self, dt: float) -> None:
        # Small semi-implicit steps resolve both spin-up and the gravity-wave head.
        plate_rate, bulk_rate, _, head_accel, amplitude_rate, phase_rate = self._rates()
        self.plate_omega += dt * plate_rate
        self.bulk_omega += dt * bulk_rate
        self.depression_rate += dt * head_accel
        self.depression += dt * self.depression_rate
        self.mode_amplitude += dt * amplitude_rate
        self.mode_phase = (self.mode_phase + dt * phase_rate) % (2.0 * math.pi)

        self.plate_omega = max(0.0, min(self.omega_target, self.plate_omega))
        self.bulk_omega = max(0.0, min(self.omega_target, self.bulk_omega))
        self.depression = max(0.0, min(0.43, self.depression))
        self.mode_amplitude = max(1.0e-7, min(0.115, self.mode_amplitude))
        self.time += dt
        self.step_count += 1

    def _surface_height(self, radius, theta):
        x = np.clip(radius / self.radius, 0.0, 1.0)
        radial_mode = 4.0 * x * (1.0 - x)
        angular_mode = np.cos(self.mode * (theta - self.mode_phase))
        surface = (self.fill_height + self.depression * (2.0 * x*x - 1.0)
                   + self.mode_amplitude * radial_mode * angular_mode)
        return np.clip(surface, 0.06, self.wall_height - 0.04)

    def _mesh_state(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(self.base_radius / self.radius, 0.0, 1.0)
        envelope = 4.0 * x * (1.0 - x)
        material_mode = np.cos(self.mode * self.base_theta)
        _, _, _, _, amplitude_rate, phase_rate = self._rates()

        # Horizontal displacement exposes the wave normal and keeps the contact
        # region circular because the envelope vanishes at the wall.
        radial_gain = 0.045 / 0.09
        rho = self.base_radius * (1.0 + radial_gain * self.mode_amplitude
                                  * envelope * material_mode)
        rho_rate = (self.base_radius * radial_gain * amplitude_rate
                    * envelope * material_mode)
        theta = self.base_theta + self.mode_phase
        ct, st = np.cos(theta), np.sin(theta)

        position = np.empty((len(rho), 3), dtype=np.float64)
        position[:, 0] = rho * ct
        position[:, 1] = rho * st
        position[:, 2] = (self.fill_height + self.depression * (2.0*x*x - 1.0)
                          + self.mode_amplitude * envelope * material_mode)

        velocity = np.empty_like(position)
        velocity[:, 0] = rho_rate * ct - rho * phase_rate * st
        velocity[:, 1] = rho_rate * st + rho * phase_rate * ct
        velocity[:, 2] = (self.depression_rate * (2.0*x*x - 1.0)
                          + amplitude_rate * envelope * material_mode)
        return position, velocity

    def _diagnostic_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        surface = self._surface_height(self.grid_r, self.grid_theta)
        level_set = np.maximum(self.grid_r - self.radius, self.zz - surface)
        inside = level_set <= 0.0

        x = np.clip(self.grid_r / self.radius, 0.0, 1.0)
        wall_distance = np.clip((self.radius - self.grid_r) / (0.12*self.radius), 0.0, 1.0)
        side_factor = np.sin(0.5 * math.pi * wall_distance)**2
        bottom_layer = np.exp(-self.zz / 0.16)
        local_omega = self.bulk_omega + (self.plate_omega - self.bulk_omega) * bottom_layer
        u_theta = local_omega * self.grid_r * side_factor

        safe_surface = np.maximum(surface, 0.08)
        vertical_fraction = np.clip(self.zz / safe_surface, 0.0, 1.0)
        # Weak Ekman return flow closes the bottom-driven circulation.
        u_radial = (-0.025 * (self.plate_omega - self.bulk_omega) * self.grid_r
                    * (1.0 - x) * np.sin(math.pi * vertical_fraction))

        _, _, _, _, amplitude_rate, phase_rate = self._rates()
        envelope = 4.0 * x * (1.0 - x)
        wave_angle = self.mode * (self.grid_theta - self.mode_phase)
        surface_w = (self.depression_rate * (2.0*x*x - 1.0)
                     + amplitude_rate * envelope * np.cos(wave_angle)
                     + self.mode * self.mode_amplitude * envelope
                     * (phase_rate - local_omega) * np.sin(wave_angle))
        u_vertical = np.clip(surface_w * vertical_fraction**1.4, -1.5, 1.5)

        rr_safe = np.maximum(self.grid_r, 1.0e-12)
        cr, sr = self.xx / rr_safe, self.yy / rr_safe
        velocity = np.empty(self.grid_shape + (3,), dtype=np.float64)
        velocity[..., 0] = u_radial * cr - u_theta * sr
        velocity[..., 1] = u_radial * sr + u_theta * cr
        velocity[..., 2] = u_vertical
        velocity *= inside[..., None]
        return level_set, inside.astype(np.float64), velocity

    def _boundary_velocity(self) -> np.ndarray:
        velocity = np.zeros((len(self.boundary_xy), 3), dtype=np.float64)
        n = self.bottom_boundary_count
        velocity[:n, 0] = -self.plate_omega * self.boundary_xy[:n, 1]
        velocity[:n, 1] = self.plate_omega * self.boundary_xy[:n, 0]
        return velocity

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(0.005, target - self.time))
        self.time = target

        position, particle_velocity = self._mesh_state()
        level_set, interior, velocity_field = self._diagnostic_fields()
        boundary_velocity = self._boundary_velocity()
        frame = StateFrame(
            target,
            position,
            particle_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": boundary_velocity.copy(),
                "boundary_velocity": boundary_velocity,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.spacing,
                "reference_velocity": max(self.omega_target * self.radius, 1.0e-6),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "viscous_spinup_shallow_water_landau_mode",
            "steps": self.step_count,
            "bulk_angular_velocity": self.bulk_omega,
            "plate_angular_velocity": self.plate_omega,
            "centrifugal_head": self.depression,
            "mode_amplitude": self.mode_amplitude,
            "mode": self.mode,
        }
