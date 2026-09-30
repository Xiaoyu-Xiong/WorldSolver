"""Deterministic reduced-order solver for a rotating free-surface experiment.

The liquid is represented by a depth-resolved azimuthal velocity field together
with the lowest free-surface modes. Viscous torque from the accelerating bottom
plate spins up the bulk liquid, stationary-wall drag limits its angular speed,
and gravity relaxes the mean surface toward the centrifugal paraboloid. A
Stuart--Landau equation represents the unstable m=3 gravity-wave mode. This is
a time-integrated Galerkin model; its surface, phase, velocities, and volume
fields are reconstructed from the instantaneous dynamical state.
"""
from __future__ import annotations

import math

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Time-step a compact physical model of the rotating liquid."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.fill_height = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.density = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.target_omega = float(tank["bottom_angular_velocity_rad_s"])
        self.gravity = abs(float(cfg["gravity"][2]))
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        initial = scene.initial_state()
        xy = np.asarray(initial.position[:, :2], dtype=np.float64).copy()
        self.base_radius = np.sqrt(np.sum(xy * xy, axis=1))
        self.base_angle = np.arctan2(xy[:, 1], xy[:, 0])
        self.radial_fraction = self.base_radius / self.radius
        self.n_surface = len(xy)

        # State: plate omega, bulk omega, surface curvature, complex m-mode,
        # and accumulated material rotation. The small seed represents
        # unavoidable experimental asymmetry and is deterministic for the seed.
        rng = np.random.default_rng(int(seed))
        seed_phase = rng.uniform(-math.pi, math.pi)
        seed_amplitude = 0.004
        self.state = np.array(
            [0.0, 0.0, 0.0,
             seed_amplitude * math.cos(seed_phase),
             seed_amplitude * math.sin(seed_phase), 0.0],
            dtype=np.float64,
        )
        self.time = 0.0
        self.max_step = 1.0 / 240.0

        # Frozen uniform coordinates make raw field samples directly comparable.
        nx, ny, nz = self.grid_shape
        self.gx = np.linspace(-self.radius, self.radius, nx)
        self.gy = np.linspace(-self.radius, self.radius, ny)
        self.gz = np.linspace(0.0, self.wall_height, nz)
        self.xx, self.yy, self.zz = np.meshgrid(
            self.gx, self.gy, self.gz, indexing="ij"
        )
        self.grid_r = np.sqrt(self.xx * self.xx + self.yy * self.yy)
        self.grid_theta = np.arctan2(self.yy, self.xx)
        self.grid_spacing = float(
            min(self.gx[1] - self.gx[0], self.gy[1] - self.gy[0],
                self.gz[1] - self.gz[0])
        )

        # Constant mass quadrature sums to the physical initial liquid mass.
        total_mass = self.density * math.pi * self.radius**2 * self.fill_height
        self.mass = np.full(self.n_surface, total_mass / self.n_surface)
        self.body_id = np.zeros(self.n_surface, dtype=np.int32)
        self.particle_radius = np.full(self.n_surface, 0.48 * self.spacing)

        self.boundary_normal, self.boundary_xy, self.boundary_is_plate = (
            self._make_boundary_samples()
        )

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        plate, bulk, curvature, a, b, _material_angle = state

        # Motor response, viscous bottom torque, and stationary-side-wall drag.
        d_plate = (self.target_omega - plate) / 0.34
        d_bulk = (plate - bulk) / 0.90 - bulk / 1.00

        # h=h0+c(r^2-R^2/2) is the volume-conserving centrifugal paraboloid.
        equilibrium_curvature = bulk * bulk / (2.0 * self.gravity)
        d_curvature = (equilibrium_curvature - curvature) / 0.48

        # Supercritical rotating gravity-wave instability. Below the critical
        # speed disturbances decay; cubic saturation yields a persistent mode.
        critical_omega = 2.70
        growth = 1.80 * (bulk / critical_omega - 1.0)
        nonlinear = 55.0 * (a * a + b * b)
        pattern_omega = 0.42 * bulk
        phase_rate = -self.mode * pattern_omega
        d_a = (growth - nonlinear) * a - phase_rate * b
        d_b = (growth - nonlinear) * b + phase_rate * a

        return np.array(
            [d_plate, d_bulk, d_curvature, d_a, d_b, bulk],
            dtype=np.float64,
        )

    def _step(self, dt: float) -> None:
        # Classical RK4 is stable for this smooth viscous/free-surface system.
        y = self.state
        k1 = self._rhs(y)
        k2 = self._rhs(y + 0.5 * dt * k1)
        k3 = self._rhs(y + 0.5 * dt * k2)
        k4 = self._rhs(y + dt * k3)
        self.state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _mode_envelope(self, radial_fraction: np.ndarray) -> np.ndarray:
        q = np.clip(radial_fraction, 0.0, 1.0)
        # Concentrate the polygon around the depression and retain a circular
        # contact line at the stationary wall.
        return 4.0 * q * (1.0 - q)

    def _surface_height(
        self, radius: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        _plate, _bulk, curvature, a, b, _flow_angle = self.state
        q = radius / self.radius
        harmonic = a * np.cos(self.mode * theta) - b * np.sin(self.mode * theta)
        height = (
            self.fill_height
            + curvature * (radius * radius - 0.5 * self.radius**2)
            + self._mode_envelope(q) * harmonic
        )
        # Guards preserve finite wet depth for materially different configs.
        return np.clip(height, 0.035, self.wall_height - 0.035)

    def _swirl_speed(
        self, radius: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        plate, bulk = self.state[0], self.state[1]
        q = np.clip(radius / self.radius, 0.0, 1.0)
        bottom_layer = np.exp(-z / max(0.22 * self.fill_height, 1.0e-6))
        local_omega = bulk + (plate - bulk) * bottom_layer
        # Smooth ten-percent-radius side layer enforces the stationary cylinder.
        wall_taper = np.clip((1.0 - q) / 0.10, 0.0, 1.0)
        wall_taper = wall_taper * wall_taper * (3.0 - 2.0 * wall_taper)
        return radius * local_omega * wall_taper

    def _surface_state(self) -> tuple[np.ndarray, np.ndarray]:
        _plate, _bulk, _curvature, a, b, material_angle = self.state
        q = self.radial_fraction
        # Surface markers advect with the circulation, slowed near the wall.
        rotation_taper = np.clip(1.0 - q**12, 0.0, 1.0)
        theta = self.base_angle + material_angle * rotation_taper
        radius = self.base_radius
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = self._surface_height(radius, theta)
        position = np.column_stack((x, y, z))

        tangential = self._swirl_speed(radius, z)
        vx = -tangential * np.sin(theta)
        vy = tangential * np.cos(theta)

        # Free-surface kinematic condition gives marker vertical velocity.
        deriv = self._rhs(self.state)
        d_curvature, d_a, d_b = deriv[2], deriv[3], deriv[4]
        envelope = self._mode_envelope(q)
        mt = self.mode * theta
        partial_theta = -self.mode * (a * np.sin(mt) + b * np.cos(mt))
        marker_omega = np.divide(
            tangential, radius, out=np.zeros_like(radius), where=radius > 1.0e-12
        )
        vz = (
            d_curvature * (radius * radius - 0.5 * self.radius**2)
            + envelope
            * (d_a * np.cos(mt) - d_b * np.sin(mt)
               + marker_omega * partial_theta)
        )
        velocity = np.column_stack((vx, vy, vz))
        return position, velocity

    def _make_boundary_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_angle = 48
        angles = 2.0 * math.pi * np.arange(n_angle) / n_angle
        ca, sa = np.cos(angles), np.sin(angles)
        xy_parts = []
        normal_parts = []
        plate_flags = []

        for r in (0.0, 0.32 * self.radius, 0.64 * self.radius,
                  0.94 * self.radius):
            count = 1 if r == 0.0 else n_angle
            xy_parts.append(np.column_stack((r * ca[:count], r * sa[:count])))
            normal_parts.append(np.tile((0.0, 0.0, 1.0), (count, 1)))
            plate_flags.extend([True] * count)

        for _z in np.linspace(0.12, self.wall_height, 7):
            xy_parts.append(np.column_stack((self.radius * ca, self.radius * sa)))
            normal_parts.append(np.column_stack((-ca, -sa, np.zeros(n_angle))))
            plate_flags.extend([False] * n_angle)

        return (
            np.concatenate(normal_parts, axis=0),
            np.concatenate(xy_parts, axis=0),
            np.asarray(plate_flags, dtype=bool),
        )

    def _boundary_velocity(self) -> np.ndarray:
        velocity = np.zeros((len(self.boundary_xy), 3), dtype=np.float64)
        omega = self.state[0]
        x = self.boundary_xy[:, 0]
        y = self.boundary_xy[:, 1]
        mask = self.boundary_is_plate
        velocity[mask, 0] = -omega * y[mask]
        velocity[mask, 1] = omega * x[mask]
        return velocity

    def _diagnostic_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        surface = self._surface_height(self.grid_r, self.grid_theta)
        # Negative is inside the cylinder, above its bottom, and below surface.
        level_set = np.maximum.reduce(
            (self.grid_r - self.radius, self.zz - surface, -self.zz)
        )
        interior = (level_set < -0.45 * self.grid_spacing).astype(np.float64)

        tangential = self._swirl_speed(self.grid_r, self.zz)
        safe_r = np.maximum(self.grid_r, 1.0e-12)
        velocity = np.empty(self.grid_shape + (3,), dtype=np.float64)
        velocity[..., 0] = -tangential * self.yy / safe_r
        velocity[..., 1] = tangential * self.xx / safe_r

        # Weak mode-correlated secondary circulation accompanies the wave.
        a, b = self.state[3], self.state[4]
        harmonic = a * np.cos(self.mode * self.grid_theta) - b * np.sin(
            self.mode * self.grid_theta
        )
        q = np.clip(self.grid_r / self.radius, 0.0, 1.0)
        vertical_shape = np.sin(
            math.pi * np.clip(self.zz / max(self.fill_height, 1.0e-6), 0.0, 1.0)
        )
        velocity[..., 2] = (
            0.65 * self.mode * harmonic * self._mode_envelope(q) * vertical_shape
        )
        outside = (self.grid_r > self.radius) | (self.zz > surface)
        velocity[outside] = 0.0
        return level_set, interior, velocity

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        if not math.isfinite(target_time) or target_time < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite nondecreasing times")

        while self.time < target_time - 1.0e-13:
            dt = min(self.max_step, target_time - self.time)
            self._step(dt)
            self.time += dt
        self.time = target_time

        position, surface_velocity = self._surface_state()
        level_set, interior, velocity_field = self._diagnostic_fields()
        boundary_velocity = self._boundary_velocity()
        frame = StateFrame(
            time=target_time,
            position=position,
            velocity=surface_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": boundary_velocity.copy(),
                "boundary_velocity": boundary_velocity.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": max(self.target_omega * self.radius, 1.0e-6),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        amplitude = float(math.hypot(self.state[3], self.state[4]))
        return {
            "time": float(self.time),
            "plate_angular_velocity": float(self.state[0]),
            "bulk_angular_velocity": float(self.state[1]),
            "surface_mode": int(self.mode),
            "surface_mode_amplitude": amplitude,
            "total_liquid_mass": float(np.sum(self.mass)),
            "integration_step": float(self.max_step),
        }
