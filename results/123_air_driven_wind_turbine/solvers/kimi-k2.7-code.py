"""Physically-based two-way coupled wind-turbine solver.

The solver integrates a 3-D incompressible flow on a coarse staggered MAC grid
using a pressure projection stepped with the DCT Poisson solver, coupled to a
three-blade actuator-line rotor.  Colored smoke is emitted from the rotating
tips and advected with the resolved velocity field.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates, zoom

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """MAC-fluid / actuator-line rotor solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config

        # Domain
        self.bounds_min = np.asarray(self.cfg["domain"]["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(self.cfg["domain"]["bounds_max"], dtype=float)
        self.L = self.bounds_max - self.bounds_min

        # Reference grid from config; internal grid is half-resolution to stay fast.
        self.Nf = np.asarray(self.cfg["domain"]["reference_grid"], dtype=int)
        self.Nc = self.Nf // 2
        self.dx = self.L / self.Nc

        # Air / inlet
        air_cfg = self.cfg["air"]
        self.rho = float(air_cfg["density"])
        self.inlet_vec = np.asarray(air_cfg["inlet"]["velocity"], dtype=float)
        self.ramp_start = float(air_cfg["inlet"]["start_time"])
        self.ramp_dur = float(air_cfg["inlet"]["ramp_duration"])

        # Turbine parameters
        turb = self.cfg["turbine"]
        self.hub = np.asarray(turb["hub_center"], dtype=float)
        self.axle = np.asarray(turb["axle_axis"], dtype=float)
        self.axle = self.axle / np.linalg.norm(self.axle)
        self.R = float(turb["rotor_radius"])
        self.mass = float(turb["mass"])
        # Approximate inertia of three radial blades about the axle.
        self.I = self.mass * self.R ** 2 / 3.0
        self.angle = float(turb["initial_angle"])
        self.omega = float(turb["initial_angular_velocity"])

        fr = turb["axle_friction"]
        self.c_viscous = float(fr["viscous_coefficient"])
        self.c_coulomb = float(fr["coulomb_torque"])
        self.omega_reg = float(fr["regularization_speed"])

        # Smoke
        smoke_cfg = self.cfg["smoke"]
        self.smoke_start = float(smoke_cfg["emission_start"])
        self.smoke_end = float(smoke_cfg["emission_end"])
        self.smoke_rad = float(smoke_cfg["source_radius"])
        self.smoke_rate = float(smoke_cfg["source_rate_each"])
        self.local_tips = np.asarray(smoke_cfg["rotor_local_tip_positions"], dtype=float)
        self.smoke_channels = smoke_cfg["channels"]

        # Internal time step
        self.dt = 0.005
        self.time = 0.0

        # Velocity fields on staggered MAC grid
        self.u = np.zeros((self.Nc[0] + 1, self.Nc[1], self.Nc[2]), dtype=float)
        self.v = np.zeros((self.Nc[0], self.Nc[1] + 1, self.Nc[2]), dtype=float)
        self.w = np.zeros((self.Nc[0], self.Nc[1], self.Nc[2] + 1), dtype=float)

        # Smoke scalar fields
        self.smoke = [
            np.zeros(tuple(self.Nc), dtype=float) for _ in self.smoke_channels
        ]

        # Precompute grid positions for advection / forcing
        self._init_positions()

        # Actuator-line geometry
        self.chord = 0.15
        self.CL = 1.2
        self.CD = 0.15
        self.sigma_force = max(self.dx.min(), 0.25)
        self._init_blade_elements()

        # Poisson eigenvalues for Neumann DCT solver
        self._init_poisson()

        # Budget / bookkeeping
        self.cum_angular_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_ang_mom = np.zeros(3, dtype=float)
        self.cum_outflow_ang_mom = np.zeros(3, dtype=float)
        self.cum_mass_source = np.zeros(len(self.smoke_channels), dtype=float)
        self.net_aero_force = np.zeros(3, dtype=float)
        self.net_aero_torque = np.zeros(3, dtype=float)
        self.tau_support = np.zeros(3, dtype=float)

        # Reference scales
        self.ref_vel = float(np.linalg.norm(self.inlet_vec))
        self.ref_ang_mom = max(1.0, self.I * self.ref_vel / max(self.R, 1e-6))

        # Cached cell-centered velocity for this step
        self.uc = np.zeros(tuple(self.Nc), dtype=float)
        self.vc = np.zeros(tuple(self.Nc), dtype=float)
        self.wc = np.zeros(tuple(self.Nc), dtype=float)

    def _init_positions(self):
        # Staggered grid origins
        self.cell_origin = self.bounds_min + self.dx / 2.0
        self.u_origin = np.array(
            [self.bounds_min[0], self.bounds_min[1] + self.dx[1] / 2.0,
             self.bounds_min[2] + self.dx[2] / 2.0],
            dtype=float,
        )
        self.v_origin = np.array(
            [self.bounds_min[0] + self.dx[0] / 2.0, self.bounds_min[1],
             self.bounds_min[2] + self.dx[2] / 2.0],
            dtype=float,
        )
        self.w_origin = np.array(
            [self.bounds_min[0] + self.dx[0] / 2.0,
             self.bounds_min[1] + self.dx[1] / 2.0, self.bounds_min[2]],
            dtype=float,
        )

        self.pos_c = self._positions(self.cell_origin, tuple(self.Nc))
        self.pos_u = self._positions(
            self.u_origin, (self.Nc[0] + 1, self.Nc[1], self.Nc[2])
        )
        self.pos_v = self._positions(
            self.v_origin, (self.Nc[0], self.Nc[1] + 1, self.Nc[2])
        )
        self.pos_w = self._positions(
            self.w_origin, (self.Nc[0], self.Nc[1], self.Nc[2] + 1)
        )

    def _positions(self, origin: np.ndarray, shape: tuple) -> np.ndarray:
        ranges = [origin[d] + np.arange(shape[d]) * self.dx[d] for d in range(3)]
        X, Y, Z = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)

    def _init_blade_elements(self):
        # Blade radial directions from local tip positions (project onto y-z plane)
        self.blade_dirs = []
        for tip in self.local_tips:
            d = tip.copy()
            d[0] = 0.0
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            self.blade_dirs.append(d)

        # Radial stations per blade
        n = 7
        frac = np.linspace(0.25, 1.0, n)
        self.radii = frac * self.R
        # Trapezoidal integration weights (per blade)
        dr = np.diff(frac, prepend=0.0) * self.R
        self.dr = dr

    def _init_poisson(self):
        lam_x = -(2.0 / self.dx[0] ** 2) * (
            1.0 - np.cos(np.pi * np.arange(self.Nc[0]) / self.Nc[0])
        )
        lam_y = -(2.0 / self.dx[1] ** 2) * (
            1.0 - np.cos(np.pi * np.arange(self.Nc[1]) / self.Nc[1])
        )
        lam_z = -(2.0 / self.dx[2] ** 2) * (
            1.0 - np.cos(np.pi * np.arange(self.Nc[2]) / self.Nc[2])
        )
        self.lam = (
            lam_x[:, None, None]
            + lam_y[None, :, None]
            + lam_z[None, None, :]
        )
        self.lam[0, 0, 0] = 1.0  # regularize singular mode

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time + 1e-12:
            return self._make_frame()
        while self.time < time - 1e-12:
            dt = min(self.dt, time - self.time)
            self._step(dt)
            self.time += dt
        return self._make_frame()

    def _step(self, dt: float) -> None:
        # Update cell-centered velocity for interpolation
        self._cell_centered()

        # Actuator-line forcing and rotor dynamics (two-way coupling)
        self._actuator(dt)

        # Advect velocities
        self.u = self._advect(self.u, self.u_origin, "u")
        self.v = self._advect(self.v, self.v_origin, "v")
        self.w = self._advect(self.w, self.w_origin, "w")

        # Pressure projection to enforce incompressibility
        self._project(dt)

        # Apply boundary conditions
        self._apply_boundary_conditions(self.time)

        # Advect smoke and emit from rotating tips
        self._smoke_step(dt)

    # ------------------------------------------------------------------
    # Fluid helpers
    # ------------------------------------------------------------------
    def _cell_centered(self):
        self.uc = 0.5 * (self.u[1:, :, :] + self.u[:-1, :, :])
        self.vc = 0.5 * (self.v[:, 1:, :] + self.v[:, :-1, :])
        self.wc = 0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])

    def _inlet_velocity(self, t: float) -> np.ndarray:
        s = np.clip((t - self.ramp_start) / self.ramp_dur, 0.0, 1.0)
        weight = s * s * (3.0 - 2.0 * s)
        return weight * self.inlet_vec

    def _apply_boundary_conditions(self, t: float):
        U = self._inlet_velocity(t)

        # Inflow at x_min
        self.u[0, :, :] = U[0]
        # Open / no-gradient boundaries
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[0, :, :] = self.v[1, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[0, :, :] = self.w[1, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]

        # y boundaries (open)
        self.v[:, 0, :] = self.v[:, 1, :]
        self.v[:, -1, :] = self.v[:, -2, :]
        self.u[:, 0, :] = self.u[:, 1, :]
        self.u[:, -1, :] = self.u[:, -2, :]
        self.w[:, 0, :] = self.w[:, 1, :]
        self.w[:, -1, :] = self.w[:, -2, :]

        # z boundaries: no-slip ground and open top
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0
        self.u[:, :, -1] = self.u[:, :, -2]
        self.v[:, :, -1] = self.v[:, :, -2]
        self.w[:, :, -1] = self.w[:, :, -2]

        # Static solid obstacles (tower / nacelle) -> no-slip
        self._zero_in_obstacles(self.u, self.pos_u)
        self._zero_in_obstacles(self.v, self.pos_v)
        self._zero_in_obstacles(self.w, self.pos_w)

    def _zero_in_obstacles(self, arr: np.ndarray, pos: np.ndarray):
        mask = self._obstacle_mask(pos)
        arr.ravel()[mask] = 0.0

    def _obstacle_mask(self, pos: np.ndarray) -> np.ndarray:
        x = pos[0]
        y = pos[1]
        z = pos[2]
        # Tower: cylinder along z at origin, radius .32, height 3.9
        tower = (x * x + y * y < 0.32 * 0.32) & (z < 3.9)
        # Nacelle: cylinder along x, center (0.12, 0, 4), radius .34, half-length .45
        nacelle = (np.abs(x - 0.12) < 0.45) & (y * y + (z - 4.0) ** 2 < 0.34 * 0.34)
        return tower | nacelle

    # ------------------------------------------------------------------
    # Interpolation and advection
    # ------------------------------------------------------------------
    def _interp(self, arr: np.ndarray, origin: np.ndarray, pos: np.ndarray) -> np.ndarray:
        idx = (pos - origin[:, None]) / self.dx[:, None]
        return map_coordinates(arr, idx, order=1, mode="nearest", prefilter=False)

    def _advect(self, arr: np.ndarray, origin: np.ndarray, key: str) -> np.ndarray:
        pos = getattr(self, f"pos_{key}")
        # advecting velocity at the same grid locations, interpolated from cell centers
        u_ad = self._interp(self.uc, self.cell_origin, pos)
        v_ad = self._interp(self.vc, self.cell_origin, pos)
        w_ad = self._interp(self.wc, self.cell_origin, pos)
        X_back = pos - self.dt * np.stack([u_ad, v_ad, w_ad], axis=0)
        new = self._interp(arr, origin, X_back)
        return new.reshape(arr.shape)

    # ------------------------------------------------------------------
    # Pressure projection with DCT Poisson solver
    # ------------------------------------------------------------------
    def _project(self, dt: float):
        dx, dy, dz = self.dx
        div = (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / dx
            + (self.v[:, 1:, :] - self.v[:, :-1, :]) / dy
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) / dz
        )
        rhs = div / dt

        rhs_hat = dctn(rhs, type=2, norm="ortho")
        p_hat = rhs_hat / self.lam
        p_hat[0, 0, 0] = 0.0
        p = idctn(p_hat, type=3, norm="ortho")

        # Correct face velocities (interior faces)
        self.u[1:-1, :, :] -= dt / self.rho * (p[1:, :, :] - p[:-1, :, :]) / dx
        self.v[:, 1:-1, :] -= dt / self.rho * (p[:, 1:, :] - p[:, :-1, :]) / dy
        self.w[:, :, 1:-1] -= dt / self.rho * (p[:, :, 1:] - p[:, :, :-1]) / dz

    # ------------------------------------------------------------------
    # Actuator-line rotor coupling
    # ------------------------------------------------------------------
    def _actuator(self, dt: float):
        self._cell_centered()
        rho = self.rho
        chord = self.chord
        CL = self.CL
        CD = self.CD

        total_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)

        for blade_dir in self.blade_dirs:
            e_r = self._rotate_about_x(blade_dir, self.angle)
            e_t = np.cross(self.axle, e_r)
            for r, dr in zip(self.radii, self.dr):
                Xe = self.hub + r * e_r
                uf = self._fluid_velocity_at(Xe)
                vb = self.omega * r * e_t
                W = uf - vb
                Wmag = np.linalg.norm(W)
                if Wmag < 1e-6:
                    continue
                e_W = W / Wmag
                # lift direction = component of chord (tangent) perpendicular to W
                e_c = e_t
                e_L0 = e_c - np.dot(e_c, e_W) * e_W
                Lnorm = np.linalg.norm(e_L0)
                if Lnorm < 1e-9:
                    continue
                e_L = e_L0 / Lnorm
                q = 0.5 * rho * chord * dr * Wmag ** 2
                F_blade = q * (CL * e_L - CD * e_W)
                tau = np.cross(r * e_r, F_blade)
                total_force += F_blade
                total_torque += tau
                self._apply_force_to_fluid(-F_blade, Xe)

        self.net_aero_force = total_force
        self.net_aero_torque = total_torque

        # Rotor ODE with axle friction
        tau_aero_axial = np.dot(total_torque, self.axle)
        friction = self.c_viscous * self.omega + self.c_coulomb * np.tanh(
            self.omega / self.omega_reg
        )
        alpha = (tau_aero_axial - friction) / self.I
        self.omega += dt * alpha
        self.angle += dt * self.omega

        # Support couple (constraint + friction)
        tau_perp = total_torque - np.dot(total_torque, self.axle) * self.axle
        self.tau_support = -tau_perp - friction * self.axle

        # Cumulative budgets
        net_torque_origin = np.cross(self.hub, total_force) + tau_aero_axial * self.axle - friction * self.axle
        self.cum_angular_impulse += dt * net_torque_origin

    def _fluid_velocity_at(self, x: np.ndarray) -> np.ndarray:
        pos = x.reshape(3, 1)
        u = self._interp(self.uc, self.cell_origin, pos)[0]
        v = self._interp(self.vc, self.cell_origin, pos)[0]
        w = self._interp(self.wc, self.cell_origin, pos)[0]
        return np.array([u, v, w])

    def _apply_force_to_fluid(self, F: np.ndarray, Xe: np.ndarray):
        w_u = self._gaussian_weights(self.pos_u, Xe)
        w_v = self._gaussian_weights(self.pos_v, Xe)
        w_w = self._gaussian_weights(self.pos_w, Xe)
        factor = self.dt / self.rho
        self.u += factor * F[0] * w_u.reshape(self.u.shape)
        self.v += factor * F[1] * w_v.reshape(self.v.shape)
        self.w += factor * F[2] * w_w.reshape(self.w.shape)

    def _gaussian_weights(self, pos: np.ndarray, center: np.ndarray) -> np.ndarray:
        diff = pos - center[:, None]
        dist2 = np.sum(diff * diff, axis=0)
        w = np.exp(-dist2 / (2.0 * self.sigma_force ** 2))
        total = w.sum()
        if total > 0.0:
            w = w / total
        return w

    @staticmethod
    def _rotate_about_x(v: np.ndarray, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array(
            [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
        )

    # ------------------------------------------------------------------
    # Smoke
    # ------------------------------------------------------------------
    def _smoke_step(self, dt: float):
        self._cell_centered()

        if self.smoke_start <= self.time <= self.smoke_end:
            tips = self._world_tips()
            for idx, tip in enumerate(tips):
                w = self._gaussian_weights(self.pos_c, tip)
                amount = self.smoke_rate * dt
                self.smoke[idx] += amount * w.reshape(tuple(self.Nc))
                self.cum_mass_source[idx] += amount

        # Advect smoke with cell-centered velocity
        u_ad = self._interp(self.uc, self.cell_origin, self.pos_c)
        v_ad = self._interp(self.vc, self.cell_origin, self.pos_c)
        w_ad = self._interp(self.wc, self.cell_origin, self.pos_c)
        X_back = self.pos_c - self.dt * np.stack([u_ad, v_ad, w_ad], axis=0)

        for idx in range(len(self.smoke_channels)):
            snew = self._interp(self.smoke[idx], self.cell_origin, X_back).reshape(
                tuple(self.Nc)
            )
            # mild decay to avoid unbounded accumulation
            snew *= (1.0 - 0.1 * dt)
            self.smoke[idx] = np.clip(snew, 0.0, None)

    def _world_tips(self) -> np.ndarray:
        R = self._rotation_x(self.angle)
        return self.hub + (R @ self.local_tips.T).T

    @staticmethod
    def _rotation_x(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    # ------------------------------------------------------------------
    # Output frame
    # ------------------------------------------------------------------
    def _make_frame(self) -> StateFrame:
        # Cell-centered velocity on coarse grid
        uc = 0.5 * (self.u[1:, :, :] + self.u[:-1, :, :])
        vc = 0.5 * (self.v[:, 1:, :] + self.v[:, :-1, :])
        wc = 0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])
        vel_c = np.stack([uc, vc, wc], axis=-1)

        # Upsample to required grid
        vel_f = zoom(vel_c, (2, 2, 2, 1), order=1).astype(np.float32)
        smoke_f = [
            zoom(s, (2, 2, 2), order=1).astype(np.float32) for s in self.smoke
        ]
        air_density = np.full(tuple(self.Nf), self.rho, dtype=np.float32)

        # Rotor orientation (rotation about x-axis)
        half = 0.5 * self.angle
        orientation = np.array(
            [[np.sin(half) * self.axle[0], np.sin(half) * self.axle[1],
              np.sin(half) * self.axle[2], np.cos(half)]],
            dtype=float,
        )
        orientation = orientation / np.linalg.norm(orientation)

        # Inertia tensor (world axes, diagonal approximation)
        Ixx = self.I
        Iyy = Izz = 0.5 * self.I
        inertia = np.zeros((1, 3, 3), dtype=float)
        inertia[0, 0, 0] = Ixx
        inertia[0, 1, 1] = Iyy
        inertia[0, 2, 2] = Izz

        # Boundary samples (ground plane sample)
        boundary_normal = np.array([[0.0, 0.0, 1.0]], dtype=float)
        boundary_target_velocity = np.array([[0.0, 0.0, 0.0]], dtype=float)
        boundary_velocity = np.array([[0.0, 0.0, 0.0]], dtype=float)

        # Budget fields
        budget_ang_impulse = self.cum_angular_impulse.reshape(1, 3)
        budget_in = self.cum_inflow_ang_mom.reshape(1, 3)
        budget_out = self.cum_outflow_ang_mom.reshape(1, 3)
        domain_weights = np.ones((1, 1), dtype=float)
        external_couple = self.tau_support.reshape(1, 1, 3)
        external_force = self.net_aero_force.reshape(1, 1, 3)
        force_position = self.hub.reshape(1, 1, 3)
        interaction_force = self.net_aero_force.reshape(1, 3)
        cumulative_mass_source = self.cum_mass_source.copy()

        fields = {
            "air_density": air_density,
            "velocity_field": vel_f,
            "smoke_red": smoke_f[0],
            "smoke_green": smoke_f[1],
            "smoke_ochre": smoke_f[2],
            "grid_origin": np.asarray(self.cfg["domain"]["bounds_min"], dtype=float),
            "rotor_local_tip_positions": np.asarray(
                self.cfg["smoke"]["rotor_local_tip_positions"], dtype=float
            ),
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_velocity,
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_angular_impulse": budget_ang_impulse,
            "budget_cumulative_inflow_angular_momentum": budget_in,
            "budget_cumulative_outflow_angular_momentum": budget_out,
            "budget_domain_weights": domain_weights,
            "budget_external_couple": external_couple,
            "budget_external_force": external_force,
            "budget_force_position": force_position,
            "cumulative_mass_source": cumulative_mass_source,
            "interaction_force": interaction_force,
        }

        attributes = {
            "rotor_angle": np.array([self.angle], dtype=float),
            "inertia_tensor": inertia,
            "mass_quantity": np.array([self.mass], dtype=float),
        }

        scalars = {
            "air_density": self.rho,
            "inlet_speed": self.ref_vel,
            "axle_viscous_friction": self.c_viscous,
            "axle_coulomb_friction": self.c_coulomb,
            "rotor_radius": self.R,
            "reference_velocity": self.ref_vel,
            "reference_angular_momentum": self.ref_ang_mom,
        }

        frame = StateFrame(
            time=self.time,
            position=self.hub.reshape(1, 3),
            velocity=np.zeros((1, 3), dtype=float),
            orientation=orientation.astype(float),
            angular_velocity=np.array([[self.omega, 0.0, 0.0]], dtype=float),
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([1], dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "omega": self.omega,
            "angle": self.angle,
            "time": self.time,
        }
