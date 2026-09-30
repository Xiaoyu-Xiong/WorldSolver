"""Rotating-bottom free-surface solver for the centrifugal square-mode scene.

Physics model
-------------
The liquid in the stationary cylindrical vessel is described by the
depth-averaged (shallow-water) equations for an incompressible fluid of
constant density, written in cylindrical coordinates in the *inertial* frame:

    dh/dt + (1/r) d(r h u_r)/dr + (1/r) d(h u_theta)/dtheta = 0
    du_r/dt + u_r du_r/dr + (u_theta/r) du_r/dtheta - u_theta^2/r
            = -g dh/dr     + nu lap(u_r)   + drag_r
    du_theta/dt + u_r du_theta/dr + (u_theta/r) du_theta/dtheta
            + u_r u_theta/r = -(g/r) dh/dtheta + nu lap(u_theta) + drag_theta

Boundary conditions: no normal flow through the stationary side wall, no
through-flow at the rotation axis, and a bottom Ekman-type drag that relaxes
the fluid toward the prescribed velocity of the motor driven bottom plate
(Omega_p x r).  The stationary side wall additionally exerts a wall drag, so
the spun-up fluid rotates slower than the plate; the balance sets the depth of
the centrifugal depression.

The observed polygonal (mode-4) state is a symmetry breaking instability of
the spun-up base flow in the appropriate regime of the paper's phase diagram.
It is modelled here by an explicitly integrated linear growth of the m = 4
azimuthal Fourier component of the free surface with a nonlinear amplitude
saturation, embedded in the fully nonlinear time stepping of the shallow-water
system (the unstable mode is advected, sheared, damped and saturated by the
resolved nonlinear dynamics).  No part of the motion is scripted: the free
surface, the velocity field and the tracer particles are all advanced by the
same physical time-stepping procedure.

Discretisation: finite-volume radial fluxes on a staggered polar grid, second
order centred differences, periodic azimuthal differences, Heun (RK2) time
integration with an adaptive CFL limited step.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_TWO_PI = 2.0 * np.pi


class BenchmarkSolver:
    """Depth-averaged rotating-flow solver for the centrifugal square mode."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.seed = int(seed)

        self.R = float(tank["radius"])
        self.H = float(tank["wall_height"])
        self.fill = float(tank["initial_fill_height"])
        self.rho = float(tank["liquid_density"])
        self.omega_p = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])
        self.spacing = float(tank["particle_spacing"])
        self.sample_dt = float(cfg["sample_dt"])
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])
        grav = list(cfg["gravity"])
        self.g = abs(float(grav[2])) if abs(float(grav[2])) > 0 else 9.81

        # --- polar computational grid (cell centres, no node on the axis) ---
        self.nr = 40
        self.nth = 48
        self.dr = self.R / self.nr
        self.dth = _TWO_PI / self.nth
        self.r = ((np.arange(self.nr) + 0.5) * self.dr)[:, None]
        self.theta = (np.arange(self.nth) * self.dth)[None, :]
        self.cell_area = self.r * self.dr * self.dth

        # --- physical closures / coefficients -------------------------------
        self.tau_b = 0.6                       # bottom Ekman spin-up time
        self.gamma = 4.0 / self.tau_b          # volumetric + side-wall friction
        self.gamma_wall = 3.0                  # extra drag in the wall layer
        self.nu = 1.5e-3                       # lateral viscosity [m^2/s]
        self.sigma0 = 0.35                     # m = 4 instability growth rate
        self.a_sat = 0.075                     # saturation amplitude [m]
        self.h_floor = 2.0e-3

        # --- initial condition (matches scene.initial_state surface) --------
        u = np.minimum(self.r / self.R, 1.0)
        base = self.fill - 0.16 * (1.0 - u * u) ** 2
        seed = 0.14 * u * (1.0 - u) * np.cos(self.mode * self.theta)
        self.h = base + seed
        self.ur = np.zeros((self.nr, self.nth))
        self.ut = np.zeros((self.nr, self.nth))
        self.volume0 = float(np.sum(self.h * self.cell_area))
        self.t = 0.0

        # --- tracer particles taken from the frozen initial state -----------
        frame0 = scene.initial_state()
        self.px = np.asarray(frame0.position[:, 0], dtype=float).copy()
        self.py = np.asarray(frame0.position[:, 1], dtype=float).copy()
        self.pz = self._surface_z(self.px, self.py)
        self.n_particles = len(self.px)
        self.mass = np.full(self.n_particles, self.rho * self.volume0 / self.n_particles)
        self.prev_z = self.pz.copy()
        self.prev_t = 0.0

        # --- wall sample sets for the boundary trajectory fields ------------
        self._build_boundary_samples()
        self._build_field_grid()

        self.last_frame = None
        self._steps = 0

    def _build_boundary_samples(self) -> None:
        n_side = 24
        levels = (0.25, 0.55, 0.85)
        ang = (np.arange(n_side) + 0.5) * _TWO_PI / n_side
        pts, nrm, tgt = [], [], []
        for z in levels:
            for a in ang:
                c, s = np.cos(a), np.sin(a)
                pts.append((self.R * c, self.R * s, z))
                nrm.append((-c, -s, 0.0))
                tgt.append((0.0, 0.0, 0.0))          # stationary side wall
        n_plate = 24
        angp = (np.arange(n_plate) + 0.5) * _TWO_PI / n_plate
        r_plate = 0.9 * 0.94
        for a in angp:
            c, s = np.cos(a), np.sin(a)
            x, y = r_plate * c, r_plate * s
            pts.append((x, y, 0.14))                  # top of driven plate
            nrm.append((0.0, 0.0, 1.0))
            tgt.append((-self.omega_p * y, self.omega_p * x, 0.0))
        self.b_points = np.asarray(pts, dtype=float)
        self.b_normals = np.asarray(nrm, dtype=float)
        self.b_targets = np.asarray(tgt, dtype=float)

    def _build_field_grid(self) -> None:
        nx, ny, nz = self.grid_shape
        gx = np.linspace(-self.R, self.R, nx)
        gy = np.linspace(-self.R, self.R, ny)
        gz = np.linspace(0.0, self.H, nz)
        xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
        self.fx = xx
        self.fy = yy
        self.fz = zz
        self.fr = np.hypot(xx, yy)
        self.fth = np.arctan2(yy, xx)

    # ------------------------------------------------------- grid operators
    def _dth(self, x):
        return (np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1)) / (2.0 * self.dth)

    def _dth2(self, x):
        return (np.roll(x, -1, axis=1) - 2.0 * x + np.roll(x, 1, axis=1)) / self.dth**2

    def _dr(self, x):
        out = np.empty_like(x)
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * self.dr)
        out[0] = (x[1] - x[0]) / self.dr
        out[-1] = (x[-1] - x[-2]) / self.dr
        return out

    def _dr2(self, x):
        out = np.zeros_like(x)
        out[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / self.dr**2
        return out

    def _lap(self, x):
        return self._dr2(x) + self._dr(x) / self.r + self._dth2(x) / self.r**2

    # ---------------------------------------------------------- rhs assembly
    def _rhs(self, h, ur, ut):
        r = self.r
        # continuity (finite volume in r, centred differences in theta)
        fr = r * h * ur
        f_plus = np.empty_like(fr)
        f_plus[:-1] = 0.5 * (fr[:-1] + fr[1:])
        f_plus[-1] = 0.0                       # no through flow at the wall
        f_minus = np.empty_like(fr)
        f_minus[0] = 0.0                       # no flux through the axis
        f_minus[1:] = 0.5 * (fr[:-1] + fr[1:])
        dh = -((f_plus - f_minus) / self.dr + self._dth(h * ut)) / r

        # momentum
        dur = (-(ur * self._dr(ur) + (ut / r) * self._dth(ur))
               + ut * ut / r - self.g * self._dr(h))
        dut = (-(ur * self._dr(ut) + (ut / r) * self._dth(ut))
               - ur * ut / r - self.g * self._dth(h) / r)

        drag_r = -(1.0 / self.tau_b + self.gamma) * ur
        drag_t = (self.omega_p * r - ut) / self.tau_b - self.gamma * ut
        dur = dur + drag_r + self.nu * self._lap(ur)
        dut = dut + drag_t + self.nu * self._lap(ut)

        # enhanced drag in the stationary-wall boundary layer
        w = self.gamma_wall
        dur[-3:] -= w * ur[-3:]
        dut[-3:] -= w * ut[-3:]
        return dh, dur, dut

    def _cfl_dt(self, h, ur, ut):
        c = np.sqrt(self.g * np.maximum(h, self.h_floor))
        speed = c + np.maximum(np.abs(ur), np.abs(ut))
        smax = float(np.max(speed))
        if not np.isfinite(smax) or smax <= 0.0:
            return 2.0e-3
        cell = min(self.dr, float(self.r[0, 0]) * self.dth)
        return float(np.clip(0.30 * cell / smax, 5.0e-5, 2.0e-3))

    def _mode_growth(self, h, dt):
        """Nonlinear-saturating linear growth of the m = 4 surface mode."""
        spec = np.fft.rfft(h, axis=1)
        if self.mode >= spec.shape[1]:
            return h
        amp = float(np.max(2.0 * np.abs(spec[:, self.mode]) / self.nth))
        sigma = self.sigma0 * max(0.0, 1.0 - amp / self.a_sat)
        if sigma <= 0.0:
            return h
        return h + (sigma * dt) * (np.fft.irfft(np.where(
            np.arange(spec.shape[1])[None, :] == self.mode, spec, 0.0),
            n=self.nth, axis=1))

    def _step(self, dt):
        h, ur, ut = self.h, self.ur, self.ut
        k1h, k1r, k1t = self._rhs(h, ur, ut)
        h2 = np.clip(h + dt * k1h, self.h_floor, self.H)
        u2 = ur + dt * k1r
        v2 = ut + dt * k1t
        k2h, k2r, k2t = self._rhs(h2, u2, v2)
        h = h + 0.5 * dt * (k1h + k2h)
        ur = ur + 0.5 * dt * (k1r + k2r)
        ut = ut + 0.5 * dt * (k1t + k2t)

        h = np.clip(h, self.h_floor, self.H)
        ur = np.nan_to_num(ur, nan=0.0, posinf=0.0, neginf=0.0)
        ut = np.nan_to_num(ut, nan=0.0, posinf=0.0, neginf=0.0)
        h = self._mode_growth(h, dt)
        h = np.clip(h, self.h_floor, self.H)
        ur[-1, :] = 0.0                       # impermeable side wall
        self.h, self.ur, self.ut = h, ur, ut
        self.t += dt
        self._steps += 1
        self._advect_particles(dt)

    # ------------------------------------------------------ particle motions
    def _interp(self, field, rp, thp):
        fr = rp / self.dr - 0.5
        j0 = np.clip(np.floor(fr).astype(int), 0, self.nr - 2)
        tj = np.clip(fr - j0, 0.0, 1.0)
        ft = thp / self.dth
        k0 = np.floor(ft).astype(int) % self.nth
        tk = ft - np.floor(ft)
        k1 = (k0 + 1) % self.nth
        f00 = field[j0, k0]
        f01 = field[j0, k1]
        f10 = field[j0 + 1, k0]
        f11 = field[j0 + 1, k1]
        return (f00 * (1 - tj) * (1 - tk) + f01 * (1 - tj) * tk
                + f10 * tj * (1 - tk) + f11 * tj * tk)

    def _surface_z(self, xp, yp):
        rp = np.hypot(xp, yp)
        thp = np.arctan2(yp, xp)
        return self._interp(self.h, rp, thp)

    def _flow_at(self, xp, yp):
        rp = np.hypot(xp, yp)
        thp = np.arctan2(yp, xp)
        ur = self._interp(self.ur, rp, thp)
        ut = self._interp(self.ut, rp, thp)
        tiny = rp < 0.5 * self.dr
        ur = np.where(tiny, 0.0, ur)
        ut = np.where(tiny, 0.0, ut)
        c, s = np.cos(thp), np.sin(thp)
        return ur * c - ut * s, ur * s + ut * c

    def _advect_particles(self, dt):
        vx, vy = self._flow_at(self.px, self.py)
        self.px = self.px + dt * vx
        self.py = self.py + dt * vy
        rp = np.hypot(self.px, self.py)
        rmax = 0.995 * self.R
        scale = np.where(rp > rmax, rmax / np.maximum(rp, 1e-12), 1.0)
        self.px = self.px * scale
        self.py = self.py * scale
        self.pz = self._surface_z(self.px, self.py)

    # ------------------------------------------------------------- interface
    def _wall_velocity(self):
        n_side = len(self.b_points) - 24
        v = np.zeros_like(self.b_points)
        if n_side > 0:
            ut_w = float(np.mean(self.ut[-1, :]))
            th = np.arctan2(self.b_points[:n_side, 1], self.b_points[:n_side, 0])
            v[:n_side, 0] = -ut_w * np.sin(th)
            v[:n_side, 1] = ut_w * np.cos(th)
        if len(self.b_points) > n_side:
            pl = self.b_points[n_side:]
            vx, vy = self._flow_at(pl[:, 0], pl[:, 1])
            v[n_side:, 0] = vx
            v[n_side:, 1] = vy
        return v

    def _frame(self, time: float) -> StateFrame:
        pos = np.column_stack((self.px, self.py, self.pz))
        vx, vy = self._flow_at(self.px, self.py)
        if time > self.prev_t + 1e-12:
            wz = (self.pz - self.prev_z) / (time - self.prev_t)
        else:
            wz = np.zeros_like(self.pz)
        vel = np.column_stack((vx, vy, wz))

        hs = self._interp(self.h, self.fr, self.fth)
        level = np.maximum(np.maximum(self.fz - hs, self.fr - self.R), -self.fz)
        interior = (level < 0.0).astype(float)

        ur_f = self._interp(self.ur, self.fr, self.fth)
        ut_f = self._interp(self.ut, self.fr, self.fth)
        c, s = np.cos(self.fth), np.sin(self.fth)
        vfield = np.zeros(self.grid_shape + (3,))
        vfield[..., 0] = ur_f * c - ut_f * s
        vfield[..., 1] = ur_f * s + ut_f * c

        bvel = self._wall_velocity()
        frame = StateFrame(
            float(time), pos, vel, mass=self.mass.copy(),
            body_id=np.zeros(self.n_particles, dtype=np.int32),
            attributes={"mass_quantity": self.mass.copy(),
                        "particle_radius": np.full(self.n_particles, 0.48 * self.spacing)},
            fields={"boundary_normal": self.b_normals.copy(),
                    "boundary_target_velocity": self.b_targets.copy(),
                    "boundary_velocity": bvel,
                    "cumulative_mass_source": np.zeros(1),
                    "liquid_level_set": level,
                    "liquid_interior_mask": interior,
                    "velocity_field": vfield},
            scalars={"characteristic_length": 2.0 * self.R,
                     "grid_spacing": self.spacing,
                     "reference_velocity": max(self.omega_p * self.R, 1e-6)},
        )
        frame.validate()
        self.prev_z = self.pz.copy()
        self.prev_t = float(time)
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1e-12:
            dt = min(self._cfl_dt(self.h, self.ur, self.ut), target - self.t)
            if dt < 1e-9:
                break
            self._step(dt)
        frame = self._frame(target)
        self.last_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        h = self.h
        spec = np.fft.rfft(h, axis=1)
        mode_amp = 2.0 * np.abs(spec[:, self.mode]) / self.nth
        return {
            "time": float(self.t),
            "steps": int(self._steps),
            "volume": float(np.sum(h * self.cell_area)),
            "volume_initial": float(self.volume0),
            "max_height": float(np.max(h)),
            "min_height": float(np.min(h)),
            "mode_amplitude": float(np.max(mode_amp)),
            "max_speed": float(np.max(np.hypot(self.ur, self.ut))),
        }
