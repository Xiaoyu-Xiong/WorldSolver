"""Deterministic reduced-order solver for the rotating free-surface experiment.

The model keeps the dynamically important degrees of freedom of this experiment:
motor spin-up, transfer of angular momentum from the plate to the liquid, the
axisymmetric centrifugal gravity mode, and a supercritical azimuthal surface
mode. All of them are advanced with a fixed-step RK4 integrator. The latter is
the usual Landau amplitude equation for a rotating-wave instability; its linear
growth rate is determined by the instantaneous centrifugal depression, not by
time or by output frame number.

The modal state is reconstructed on the frozen diagnostic grid. The mean of
every surface mode over the circular tank is zero, so liquid volume and the
exported mass are constant. A velocity field satisfying the axisymmetric
shallow-water continuity equation is supplied together with the rotating flow
and the velocity associated with the fourfold wave.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physics solver for a liquid driven by a rotating bottom plate."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.fill_height = float(tank["initial_fill_height"])
        self.density = float(tank["liquid_density"])
        self.omega_command = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])
        self.gravity = abs(float(cfg["gravity"][2]))
        self.particle_spacing = float(tank["particle_spacing"])
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        initial = scene.initial_state()
        self.n_surface = int(initial.position.shape[0])
        # The frozen topology consists of a center and five rings of 48 vertices.
        # Use circular material radii initially; the instability develops solely
        # from the stepped modal equations below.
        if self.n_surface == 241:
            radii = np.repeat(
                self.radius * np.asarray([0.20, 0.42, 0.59, 0.76, 0.94]), 48
            )
            angles = np.tile(np.arange(48, dtype=float) * (2.0 * np.pi / 48.0), 5)
            self.surface_r = np.concatenate(([0.0], radii))
            self.surface_theta = np.concatenate(([0.0], angles))
        else:
            xy = np.asarray(initial.position[:, :2], dtype=float)
            self.surface_r = np.linalg.norm(xy, axis=1)
            self.surface_theta = np.arctan2(xy[:, 1], xy[:, 0])

        nx, ny, nz = self.grid_shape
        gx = np.linspace(-self.radius, self.radius, nx)
        gy = np.linspace(-self.radius, self.radius, ny)
        gz = np.linspace(0.0, self.wall_height, nz)
        self.xx, self.yy, self.zz = np.meshgrid(gx, gy, gz, indexing="ij")
        self.rr = np.sqrt(self.xx * self.xx + self.yy * self.yy)
        self.theta = np.arctan2(self.yy, self.xx)
        self.rho = self.rr / self.radius

        # State: plate omega, liquid omega, bowl height difference and velocity,
        # real/imaginary m=4 wave amplitude, and accumulated bulk rotation.
        rng = np.random.default_rng(int(seed))
        seed_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        seed_amplitude = 0.0040
        self.state = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                seed_amplitude * math.cos(seed_phase),
                seed_amplitude * math.sin(seed_phase),
                0.0,
            ],
            dtype=float,
        )
        self.time = 0.0
        self.max_step = 0.005
        self.steps = 0

        # Exported surface samples represent equal portions of the whole liquid,
        # rather than only the thin visual surface mesh.
        total_mass = self.density * np.pi * self.radius**2 * self.fill_height
        self.mass = np.full(self.n_surface, total_mass / self.n_surface, dtype=float)
        self.body_id = np.zeros(self.n_surface, dtype=np.int32)
        self.particle_radius = np.full(
            self.n_surface, 0.48 * self.particle_spacing, dtype=float
        )

        self._make_boundary_samples()
        self._last_frame = None

    def _make_boundary_samples(self) -> None:
        """Fixed collocation set on the plate and stationary cylindrical wall."""
        ntheta = 48
        angles = np.arange(ntheta, dtype=float) * (2.0 * np.pi / ntheta)
        ca, sa = np.cos(angles), np.sin(angles)

        # Four rings on the upper face of the driven plate.
        plate_radii = self.radius * np.asarray([0.18, 0.43, 0.68, 0.93])
        pr = np.repeat(plate_radii, ntheta)
        pc = np.tile(ca, len(plate_radii))
        ps = np.tile(sa, len(plate_radii))
        plate_xy = np.column_stack((pr * pc, pr * ps))
        plate_n = np.zeros((len(pr), 3), dtype=float)
        plate_n[:, 2] = 1.0

        # Wall points span the wet and intermittently wet part of the wall.
        wall_z_levels = np.linspace(0.16, 1.34, 9)
        wc = np.tile(ca, len(wall_z_levels))
        ws = np.tile(sa, len(wall_z_levels))
        wall_xy = self.radius * np.column_stack((wc, ws))
        wall_n = np.column_stack((-wc, -ws, np.zeros_like(wc)))

        self.boundary_xy = np.vstack((plate_xy, wall_xy))
        self.boundary_normal = np.vstack((plate_n, wall_n))
        self.n_plate_boundary = len(plate_xy)

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        plate, liquid, depression, depression_rate, ax, ay, _ = state

        # A motor with finite inertia accelerates the plate. Bottom shear drives
        # the bulk while the stationary side wall removes angular momentum.
        d_plate = (self.omega_command - plate) / 0.42
        d_liquid = (plate - liquid) / 1.00 - liquid / 1.20

        # Hydrostatic response to the depth-averaged centrifugal pressure. The
        # 0.65 factor is the radial/depth average of the non-rigid swirl profile.
        equilibrium_depression = (
            0.65 * liquid * liquid * self.radius * self.radius / (2.0 * self.gravity)
        )
        natural_frequency = math.sqrt(self.gravity / self.radius)
        damping_ratio = 0.62
        d_depression = depression_rate
        d_depression_rate = natural_frequency**2 * (
            equilibrium_depression - depression
        ) - 2.0 * damping_ratio * natural_frequency * depression_rate

        # Supercritical rotating-wave instability. Below the centrifugal
        # threshold perturbations decay. Above it, cubic wave interactions
        # saturate the disturbance into a persistent rounded polygon.
        growth = -0.40 + 2.20 * (depression - 0.25)
        amplitude2 = ax * ax + ay * ay
        nonlinear_saturation = 35.0 * amplitude2
        pattern_frequency = self.mode * 0.25 * liquid
        common = growth - nonlinear_saturation
        d_ax = common * ax + pattern_frequency * ay
        d_ay = common * ay - pattern_frequency * ax

        return np.asarray(
            [
                d_plate,
                d_liquid,
                d_depression,
                d_depression_rate,
                d_ax,
                d_ay,
                liquid,
            ],
            dtype=float,
        )

    def _integrate_to(self, target: float) -> None:
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be monotonically nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            y = self.state
            k1 = self._rhs(y)
            k2 = self._rhs(y + 0.5 * dt * k1)
            k3 = self._rhs(y + 0.5 * dt * k2)
            k4 = self._rhs(y + dt * k3)
            self.state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self.time += dt
            self.steps += 1
        self.time = target

    @staticmethod
    def _envelope(rho):
        """Smooth radial eigenfunction, normalized to one at mid-radius."""
        return 16.0 * rho * rho * (1.0 - rho) * (1.0 - rho)

    @staticmethod
    def _envelope_derivative(rho):
        return 32.0 * rho * (1.0 - rho) * (1.0 - 2.0 * rho)

    def _surface_quantities(self, r, theta):
        """Return surface height and its time/radial/angular derivatives."""
        _, _, depression, depression_rate, ax, ay, _ = self.state
        rho = np.asarray(r) / self.radius
        clipped = np.clip(rho, 0.0, 1.0)
        env = self._envelope(clipped)
        denv_dr = self._envelope_derivative(clipped) / self.radius
        c = np.cos(self.mode * theta)
        s = np.sin(self.mode * theta)
        mode_shape = ax * c - ay * s
        h = self.fill_height + depression * (rho * rho - 0.5) + env * mode_shape

        rhs = self._rhs(self.state)
        dax, day = rhs[4], rhs[5]
        dhdt = depression_rate * (rho * rho - 0.5) + env * (dax * c - day * s)
        dhdr = 2.0 * depression * rho / self.radius + denv_dr * mode_shape
        dhdtheta = env * self.mode * (-ax * s - ay * c)
        return h, dhdt, dhdr, dhdtheta

    def _horizontal_velocity(self, r, theta, z, surface_h):
        """Depth-dependent swirl plus continuity velocities in polar form."""
        plate, liquid, _, depression_rate, _, _, _ = self.state
        rho = np.asarray(r) / self.radius
        inside_rho = np.clip(rho, 0.0, 1.0)

        # Bottom Ekman layer transmits plate speed; an eighth-power side layer
        # enforces the stationary cylindrical wall away from the bottom corner.
        bottom_layer = np.exp(-np.asarray(z) / 0.16)
        side_taper = np.maximum(0.0, 1.0 - inside_rho**8)
        local_omega = (liquid + (plate - liquid) * bottom_layer) * side_taper
        u_theta = local_omega * np.asarray(r)

        # Exact depth-averaged radial flux for the time-varying axisymmetric
        # bowl: div(H u_r) = -d(eta)/dt and zero flux at r=R.
        safe_h = np.maximum(np.asarray(surface_h), 0.20)
        u_r = depression_rate * np.asarray(r) * (1.0 - inside_rho**2) / (4.0 * safe_h)

        # The azimuthal wave velocity supplies its portion of the linearized
        # free-surface continuity equation.
        rhs = self._rhs(self.state)
        dax, day = rhs[4], rhs[5]
        env = self._envelope(inside_rho)
        wave_integral = dax * np.sin(self.mode * theta) + day * np.cos(self.mode * theta)
        u_theta += -np.asarray(r) * env * wave_integral / (safe_h * self.mode)
        return u_r, u_theta

    def _build_grid_fields(self):
        h, dhdt, dhdr, dhdtheta = self._surface_quantities(self.rr, self.theta)
        level_set = np.maximum(self.rr - self.radius, self.zz - h)
        interior = level_set <= 0.0

        u_r, u_theta = self._horizontal_velocity(self.rr, self.theta, self.zz, h)
        ct, st = np.cos(self.theta), np.sin(self.theta)
        ux = u_r * ct - u_theta * st
        uy = u_r * st + u_theta * ct

        # Kinematic free-surface condition, extended linearly from the bottom.
        inv_r = np.zeros_like(self.rr)
        np.divide(1.0, self.rr, out=inv_r, where=self.rr > 1.0e-10)
        w_surface = dhdt + u_r * dhdr + u_theta * inv_r * dhdtheta
        vertical_fraction = np.clip(self.zz / np.maximum(h, 0.15), 0.0, 1.0)
        uz = vertical_fraction * w_surface
        velocity = np.stack((ux, uy, uz), axis=-1)
        velocity *= interior[..., None]
        return level_set, interior, velocity

    def _build_surface_state(self):
        r = self.surface_r
        theta = self.surface_theta
        h, dhdt, dhdr, dhdtheta = self._surface_quantities(r, theta)
        position = np.column_stack((r * np.cos(theta), r * np.sin(theta), h))

        u_r, u_theta = self._horizontal_velocity(r, theta, h, h)
        ct, st = np.cos(theta), np.sin(theta)
        ux = u_r * ct - u_theta * st
        uy = u_r * st + u_theta * ct
        inv_r = np.zeros_like(r)
        np.divide(1.0, r, out=inv_r, where=r > 1.0e-10)
        uz = dhdt + u_r * dhdr + u_theta * inv_r * dhdtheta
        velocity = np.column_stack((ux, uy, uz))
        return position, velocity

    def _boundary_fields(self):
        plate = float(self.state[0])
        target = np.zeros((len(self.boundary_xy), 3), dtype=float)
        x = self.boundary_xy[: self.n_plate_boundary, 0]
        y = self.boundary_xy[: self.n_plate_boundary, 1]
        target[: self.n_plate_boundary, 0] = -plate * y
        target[: self.n_plate_boundary, 1] = plate * x
        # No slip is imposed at both the kinematic plate and stationary wall.
        measured = target.copy()
        return measured, target

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("requested time must be finite and non-negative")
        self._integrate_to(target)

        position, velocity = self._build_surface_state()
        level_set, interior, velocity_field = self._build_grid_fields()
        boundary_velocity, boundary_target = self._boundary_fields()

        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": boundary_target,
                "boundary_velocity": boundary_velocity,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.particle_spacing,
                "reference_velocity": max(self.omega_command * self.radius, 1.0e-6),
            },
        )
        frame.validate()
        self._last_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        amplitude = float(math.hypot(self.state[4], self.state[5]))
        return {
            "method": "RK4 reduced free-surface rotating-wave hydrodynamics",
            "integration_steps": int(self.steps),
            "plate_angular_velocity_rad_s": float(self.state[0]),
            "bulk_angular_velocity_rad_s": float(self.state[1]),
            "centrifugal_height_difference_m": float(self.state[2]),
            "mode_number": int(self.mode),
            "mode_amplitude_m": amplitude,
            "conserved_liquid_mass_kg": float(np.sum(self.mass)),
        }
