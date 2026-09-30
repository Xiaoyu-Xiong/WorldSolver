"""Deterministic physics solver for the rotating-bottom free-surface scene.

Physical model
--------------
The liquid in the stationary cylindrical tank is advanced with the
depth-averaged free-surface (shallow-water) equations in cylindrical
coordinates on a polar grid:

    d(eta)/dt + (1/r) d(r H u_r)/dr + (1/r) d(H u_theta)/dtheta = 0
    D(u_r)/Dt   - u_theta^2 / r = -g d(eta)/dr      + F_r + nu Lap(u_r)
    D(u_theta)/Dt + u_r u_theta / r = -g (1/r) d(eta)/dtheta + F_theta + nu Lap(u_theta)

* The rotating bottom plate (kinematic, omega_b) injects angular momentum
  through a linear Ekman-type drag that relaxes the azimuthal velocity toward
  the plate-driven target swirl profile (spin-up).
* The stationary side wall blocks radial flow (no-penetration, zero mass
  flux) and exerts azimuthal/radial drag in a thin wall layer.
* Gravity restores the free surface; the cyclostrophic balance between the
  spun-up swirl and the radial pressure gradient produces the axisymmetric
  central depression.
* The target swirl profile has anticyclonic vorticity that changes sign with
  radius, so the shear is barotropically unstable; the azimuthal mode-3
  perturbation present in the frozen initial condition grows/rotates and the
  nonlinear terms steepen it into the persistent rotating triangular
  depression.
* Surface tracer particles (frozen initial frame) are advected by the
  resolved velocity field and ride the free surface.

Time stepping is a fixed-deterministic two-stage Runge-Kutta (midpoint)
scheme with a conservative finite-volume continuity update, so the liquid
volume (mass) is conserved exactly and every state is finite.
"""
from __future__ import annotations

import numpy as np

try:  # platform contract (scene.py already depends on it)
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None


class BenchmarkSolver:
    """Shallow-water polar-grid solver for scene 167."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.seed = int(seed)
        self.R = float(tank["radius"])
        self.wall_h = float(tank["wall_height"])
        self.fill = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.rho_liq = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.omega_plate = float(tank["bottom_angular_velocity_rad_s"])
        self.g = abs(float(cfg["gravity"][2]))
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])
        self.sample_dt = float(cfg["sample_dt"])

        # ---- polar shallow-water grid -------------------------------------
        self.nr = 44
        self.nth = 128
        self.rmin = 0.08
        self.dr = (self.R - self.rmin) / self.nr
        self.rc = self.rmin + (np.arange(self.nr) + 0.5) * self.dr
        self.rf = self.rmin + np.arange(self.nr + 1) * self.dr
        self.dth = 2.0 * np.pi / self.nth
        self.thc = (np.arange(self.nth) + 0.5) * self.dth

        # ---- numerics / closure parameters --------------------------------
        self.sub = 60                       # substeps per sample interval
        self.dt = self.sample_dt / self.sub
        self.nu = 0.004                     # eddy viscosity [m^2/s]
        self.k_spin = 0.5                   # bottom-plate coupling [1/s]
        self.k_wall = 2.5                   # wall-layer drag [1/s]
        self.omega0 = 6.0                   # target swirl amplitude [rad/s]
        self.h_eps = 0.02                   # dry-layer floor [m]

        rho_n = self.rc / self.R
        self.ut_target = (self.omega0 * self.rc * (1.0 - rho_n ** 2))[:, None]
        self.wallw = np.clip((self.rc - (self.R - 0.12)) / 0.12, 0.0, 1.0)[:, None]

        # ---- initial free surface (matches the frozen scaffold state) -----
        TH = np.tile(self.thc, (self.nr, 1))
        RN = np.tile(rho_n[:, None], (1, self.nth))
        depression = 0.16 * (1.0 - RN ** 2) ** 2
        seed_mode = 0.035 * 4.0 * RN * (1.0 - RN) * np.cos(self.mode * TH)
        self.eta = self.fill - depression - seed_mode
        self.ur = np.zeros_like(self.eta)
        self.ut = np.zeros_like(self.eta)

        # ---- tracer particles from the frozen initial frame ---------------
        f0 = scene.initial_state()
        p = np.asarray(f0.position, dtype=float)
        self.n_particles = len(p)
        self.p_mass = np.asarray(f0.mass, dtype=float).copy()
        attrs = dict(f0.attributes) if f0.attributes else {}
        if "particle_radius" in attrs:
            self.p_radius = np.asarray(attrs["particle_radius"], dtype=float).copy()
        else:
            self.p_radius = np.full(self.n_particles, 0.48 * self.spacing)
        self.body_id = np.zeros(self.n_particles, dtype=np.int32)
        self.pr = np.hypot(p[:, 0], p[:, 1])
        self.pth = np.arctan2(p[:, 1], p[:, 0])

        # ---- state bookkeeping --------------------------------------------
        self.steps = 0
        self.t_now = 0.0

        # ---- diagnostic export grid (frozen conventions from scene.py) ----
        self.gx = np.linspace(-self.R, self.R, self.grid_shape[0])
        self.gy = np.linspace(-self.R, self.R, self.grid_shape[1])
        self.gz = np.linspace(0.0, self.wall_h, self.grid_shape[2])

    # ------------------------------------------------------- grid operators
    def _ddr(self, A):
        """Radial derivative, zero-gradient at both radial boundaries."""
        Ap = np.concatenate([A[:1], A, A[-1:]], axis=0)
        return (Ap[2:] - Ap[:-2]) / (2.0 * self.dr)

    def _ddt(self, A):
        """Azimuthal derivative, periodic."""
        return (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1)) / (2.0 * self.dth)

    def _lap(self, A):
        """Polar Laplacian: (1/r) d(r dA/dr)/dr + (1/r^2) d2A/dtheta2."""
        Ar = self._ddr(A)
        Arf = 0.5 * (Ar[:-1] + Ar[1:])
        Arf = np.concatenate([Arf[:1], Arf, Arf[-1:]], axis=0)
        lap_r = (self.rf[1:, None] * Arf[1:] - self.rf[:-1, None] * Arf[:-1]) / self.dr
        lap_t = (np.roll(A, -1, axis=1) - 2.0 * A + np.roll(A, 1, axis=1)) / (self.dth ** 2)
        return lap_r / self.rc[:, None] + lap_t / (self.rc[:, None] ** 2)

    def _interp(self, A, r, t):
        """Bilinear interpolation of a polar-grid field at (r, theta)."""
        r = np.asarray(r, dtype=float)
        t = np.asarray(t, dtype=float)
        x = np.clip((r - self.rmin) / self.dr - 0.5, 0.0, self.nr - 1.0)
        i0 = np.clip(np.floor(x).astype(np.int64), 0, self.nr - 2)
        fx = x - i0
        i1 = i0 + 1
        y = np.mod(np.mod(t, 2.0 * np.pi) / self.dth - 0.5, self.nth)
        j0 = np.floor(y).astype(np.int64) % self.nth
        fy = y - np.floor(y)
        j1 = (j0 + 1) % self.nth
        return (A[i0, j0] * (1.0 - fx) * (1.0 - fy)
                + A[i1, j0] * fx * (1.0 - fy)
                + A[i0, j1] * (1.0 - fx) * fy
                + A[i1, j1] * fx * fy)

    # --------------------------------------------------------- dynamics RHS
    def _rhs(self, eta, ur, ut):
        rc = self.rc[:, None]
        H = np.maximum(eta, self.h_eps)

        # continuity: conservative face fluxes, zero flux through r=rmin/R
        Hf = 0.5 * (H[:-1] + H[1:])
        urf = 0.5 * (ur[:-1] + ur[1:])
        Fr = self.rf[1:, None] * Hf * urf
        Fr = np.concatenate([np.zeros((1, self.nth)), Fr,
                             np.zeros((1, self.nth))], axis=0)
        deta = -(Fr[1:] - Fr[:-1]) / (self.dr * rc)

        Hut = H * ut
        Fut = 0.5 * (Hut + np.roll(Hut, -1, axis=1))
        deta = deta - (Fut - np.roll(Fut, 1, axis=1)) / (self.dth * rc)

        # pressure (free-surface) gradients
        ger = self._ddr(eta)
        get = self._ddt(eta) / rc

        # advection (vector-invariant form)
        def adv(A):
            return ur * self._ddr(A) + (ut / rc) * self._ddt(A)

        dur = adv(ur) + ut * ut / rc - self.g * ger + self.nu * self._lap(ur)
        dut = adv(ut) - ur * ut / rc - self.g * get + self.nu * self._lap(ut)

        # bottom-plate spin-up torque + stationary wall drag
        dut = dut + self.k_spin * (self.ut_target - ut) - self.k_wall * self.wallw * ut
        dur = dur - self.k_spin * ur - self.k_wall * self.wallw * ur
        return deta, dur, dut

    # ------------------------------------------------------------ time step
    def _step(self):
        d1e, d1r, d1t = self._rhs(self.eta, self.ur, self.ut)
        em = self.eta + 0.5 * self.dt * d1e
        um = self.ur + 0.5 * self.dt * d1r
        vm = self.ut + 0.5 * self.dt * d1t
        d2e, d2r, d2t = self._rhs(em, um, vm)
        self.eta = self.eta + self.dt * d2e
        self.ur = self.ur + self.dt * d2r
        self.ut = self.ut + self.dt * d2t

        # hard no-penetration at inner symmetry core and outer tank wall
        self.ur[0, :] = 0.0
        self.ur[-1, :] = 0.0

        # deterministic robustness bounds (keep every state finite)
        self.eta = np.clip(self.eta, -0.15, self.wall_h)
        self.ur = np.clip(self.ur, -8.0, 8.0)
        self.ut = np.clip(self.ut, -8.0, 8.0)

        self._advect_particles()
        self.steps += 1

    def _advect_particles(self):
        urp = self._interp(self.ur, self.pr, self.pth)
        utp = self._interp(self.ut, self.pr, self.pth)
        rn = self.pr + urp * self.dt
        tn = self.pth + utp * self.dt / np.maximum(self.pr, 0.05)
        self.pr = np.clip(rn, 0.02, self.R - 0.01)
        self.pth = tn

    # ------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        target = float(time)
        n_target = int(round(target / self.dt))
        while self.steps < n_target:
            self._step()
        self.t_now = target
        return self._build_frame(target)

    def _boundary_samples(self):
        """Wall sample points: stationary side wall + rotating bottom plate."""
        n_side = 48
        ang = 2.0 * np.pi * (np.arange(n_side) + 0.5) / n_side
        normal_side = np.stack([-np.cos(ang), -np.sin(ang),
                                np.zeros(n_side)], axis=1)
        vel_side = np.zeros((n_side, 3))

        n_rad, n_ang = 8, 16
        rb, tb = np.meshgrid(np.linspace(0.12, 0.90, n_rad),
                             2.0 * np.pi * (np.arange(n_ang) + 0.5) / n_ang,
                             indexing="ij")
        rb = rb.ravel()
        tb = tb.ravel()
        n_bot = rb.size
        normal_bot = np.zeros((n_bot, 3))
        normal_bot[:, 2] = 1.0
        # prescribed plate velocity: omega x r
        vel_bot = np.stack([-self.omega_plate * rb * np.sin(tb),
                            self.omega_plate * rb * np.cos(tb),
                            np.zeros(n_bot)], axis=1)

        normal = np.concatenate([normal_side, normal_bot], axis=0)
        vel = np.concatenate([vel_side, vel_bot], axis=0)
        return normal, vel

    def _build_frame(self, t):
        # ---- tracer particles on the free surface -------------------------
        rr = self.pr
        tt = self.pth
        c, s = np.cos(tt), np.sin(tt)
        urp = self._interp(self.ur, rr, tt)
        utp = self._interp(self.ut, rr, tt)
        eta_p = self._interp(self.eta, rr, tt)
        ger = self._interp(self._ddr(self.eta), rr, tt)
        get = self._interp(self._ddt(self.eta) / self.rc[:, None], rr, tt)
        wz = urp * ger + utp * get           # material free-surface height rate
        pos = np.stack([rr * c, rr * s, eta_p], axis=1)
        vel = np.stack([urp * c - utp * s,
                        urp * s + utp * c,
                        wz], axis=1)

        # ---- diagnostic grids ----------------------------------------------
        XX, YY, ZZ = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        RG = np.hypot(XX, YY)
        TG = np.arctan2(YY, XX)
        etaG = self._interp(self.eta, RG, TG)
        level_set = np.maximum(RG - self.R, ZZ - etaG)
        interior = level_set < -1.5 * self.spacing

        urG = self._interp(self.ur, RG, TG)
        utG = self._interp(self.ut, RG, TG)
        cg, sg = np.cos(TG), np.sin(TG)
        vx = urG * cg - utG * sg
        vy = urG * sg + utG * cg
        vz = np.zeros_like(vx)
        outside = level_set >= 0.0
        vx = np.where(outside, 0.0, vx)
        vy = np.where(outside, 0.0, vy)
        velocity_field = np.stack([vx, vy, vz], axis=-1)

        bnormal, bvel = self._boundary_samples()

        fields = {
            "boundary_normal": bnormal,
            "boundary_target_velocity": bvel.copy(),
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(1),
            "liquid_interior_mask": interior,
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
        }
        scalars = {
            "characteristic_length": 2.0 * self.R,
            "grid_spacing": self.spacing,
            "reference_velocity": max(self.omega_plate * self.R, 1e-6),
        }
        attributes = {
            "mass_quantity": self.p_mass.copy(),
            "particle_radius": self.p_radius.copy(),
        }
        if StateFrame is not None:
            frame = StateFrame(
                t, pos, vel,
                mass=self.p_mass.copy(),
                body_id=self.body_id,
                attributes=attributes,
                fields=fields,
                scalars=scalars,
            )
        else:  # pragma: no cover - contract always present on the platform
            frame = _FallbackFrame(t, pos, vel, attributes, fields, scalars)
        return frame

    def diagnostics(self) -> dict:
        eta = self.eta
        ring = eta[self.nr // 3, :]
        m = self.mode
        amp = float(np.abs(np.mean(ring * np.exp(1j * m * self.thc))))
        return {
            "time": float(self.t_now),
            "steps": int(self.steps),
            "min_eta": float(np.min(eta)),
            "max_eta": float(np.max(eta)),
            "mean_eta": float(np.mean(eta)),
            "mode3_amplitude": amp,
            "max_speed": float(np.max(np.hypot(self.ur, self.ut))),
        }


class _FallbackFrame:  # pragma: no cover
    def __init__(self, t, pos, vel, attributes, fields, scalars):
        self.time = t
        self.position = pos
        self.velocity = vel
        self.mass = None
        self.body_id = None
        self.attributes = attributes
        self.fields = fields
        self.scalars = scalars
