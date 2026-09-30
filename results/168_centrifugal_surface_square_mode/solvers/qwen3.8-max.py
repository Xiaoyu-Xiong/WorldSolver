"""Physically-based reduced-order solver for the rotating-fluid polygon scene.

Model (all quantities are computed, not scripted):

1. Base azimuthal flow ``v(r,t)``: depth-averaged angular-momentum balance
   ``dv/dt = nu_t * (v_rr + v_r/r - v/r^2) + C_d*|slip|*slip/h - C_w*(r/R)^6*v/h``
   with quadratic bottom drag against the spinning plate (slip = Omega(t)*r - v),
   a stationary-side-wall friction layer, turbulent momentum diffusion, and
   regularity v(0)=0 / no-penetration at the wall.  Omega(t) ramps smoothly to
   the configured target rate (spin-up transient).

2. Free surface ``zeta(r,t)``: quasi-static radial momentum balance
   ``g dzeta/dr = v^2/r`` (centrifugal depression), integrated and offset by a
   bisection that enforces exact liquid-volume conservation.  Depth is clipped
   at h_min (dry/deep-depression core) and at the wall height.

3. Azimuthal mode-4 perturbation: Landau amplitude equation
   ``dA/dt = sigma*A - beta*A^3`` with growth rate ``sigma = k_s*(S - S_c)``
   where ``S`` is the area-weighted rotational Froude number ``v/sqrt(g*zeta)``
   of the *computed* base flow (centrifugal symmetry-breaking instability).
   The pattern phase precesses with the area-weighted mean fluid angular
   velocity (``dphi/dt = c_p * <v/r>``), as observed for rotating polygons.
   Surface: ``zeta_tot = zeta(r) + A * 4(rho)(1-rho) * cos(4(theta-phi))``.

4. Particle carriers: the 241 surface samples are advected by the physical
   velocity field: azimuthal ``v(r_p)``, radial velocity from depth-integrated
   incompressible continuity ``u_r = -(1/(r h)) int_0^r r' dzeta/dt dr'``, and
   vertical velocity from the material derivative of the surface elevation.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.scene = scene
        self.seed = int(seed)
        self.R = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.H0 = float(tank["initial_fill_height"])
        self.h_sp = float(tank["particle_spacing"])
        self.mode = int(tank["target_mode"])
        self.Omega_T = float(tank["bottom_angular_velocity_rad_s"])
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])
        grav = np.asarray(cfg["gravity"], dtype=float)
        self.g = float(abs(grav[2])) if grav[2] != 0 else 9.81

        init = scene.initial_state()
        p = np.asarray(init.position, dtype=float)
        self.N = len(p)
        self.mass = np.asarray(init.attributes["mass_quantity"], dtype=float).copy()
        self.p_rad = np.asarray(init.attributes["particle_radius"], dtype=float).copy()
        self.r_p = np.hypot(p[:, 0], p[:, 1]).copy()
        self.th_p = np.arctan2(p[:, 1], p[:, 0]).copy()
        self.z_p = p[:, 2].copy()
        self.w_p = np.zeros(self.N)

        # radial grid for the base flow
        self.nr = 96
        self.dr = self.R / self.nr
        self.r_c = (np.arange(self.nr) + 0.5) * self.dr
        rho = self.r_c / self.R

        # physical parameters
        self.C_d = 0.016      # quadratic bottom-plate drag coefficient
        self.C_w = 3.0        # stationary side-wall layer drag coefficient
        self.nu_t = 5.0e-3    # turbulent momentum diffusivity (m^2/s)
        self.h_min = 0.02     # minimum wetted depth (m)
        self.k_s = 2.5        # instability sensitivity (1/s)
        self.S_c = 0.40       # critical rotational Froude number
        self.beta = 55.0      # Landau saturation coefficient
        self.c_p = 0.55       # pattern precession fraction of mean rotation
        self.T_ramp = 2.5     # plate spin-up time (s)
        self.dt_base = 0.005

        # initial base flow consistent with the scene's sculpted depression:
        # zeta = H0 - 0.16(1-rho^2)^2  =>  v = rho*sqrt(0.64*g*(1-rho^2))
        self.v = np.sqrt(0.64 * self.g) * rho * np.sqrt(np.maximum(1.0 - rho * rho, 0.0))
        self.zeta = self.H0 - 0.16 * (1.0 - rho * rho) ** 2
        self.V0 = float(np.sum(self.zeta * 2.0 * np.pi * self.r_c * self.dr))
        self.zeta_prev = self.zeta.copy()
        self.dzdt = np.zeros(self.nr)
        self.dzdr = self.v ** 2 / (self.g * self.r_c)
        self.u_r = np.zeros(self.nr)

        # mode-4 state (matches the seeded perturbation amplitude in scene.py)
        self.A = 0.035
        self.phi = 0.0
        self.dA = 0.0
        self.omega_p = 0.0
        self.t = 0.0

        self._precompute_grids()
        self._precompute_boundary()
        self.scalars = {
            "grid_spacing": self.h_sp,
            "characteristic_length": 2.0 * self.R,
            "reference_velocity": max(self.Omega_T * self.R, 1e-6),
        }

    def _precompute_grids(self):
        nx, ny, nz = self.grid_shape
        gx = np.linspace(-self.R, self.R, nx)
        gy = np.linspace(-self.R, self.R, ny)
        gz = np.linspace(0.0, self.wall_height, nz)
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        self.gx, self.gz_axis = gx, gz
        self.rr_g = np.hypot(xx, yy)
        self.th_g = np.arctan2(yy, xx)
        rho_g = np.minimum(self.rr_g / self.R, 1.0)
        self.f_g = 4.0 * rho_g * (1.0 - rho_g)                 # mode shape
        self.fp_g = 4.0 * (1.0 - 2.0 * rho_g) / self.R         # d f / dr
        self.cos_g, self.sin_g = np.cos(self.th_g), np.sin(self.th_g)
        self.zz_g = gz[None, None, :]

    def _precompute_boundary(self):
        # rotating bottom-plate top samples (plate top at z = 0.14)
        rb = np.array([0.12, 0.30, 0.48, 0.66, 0.84])
        ab = np.arange(16) * (2.0 * np.pi / 16.0)
        RB, TB = np.meshgrid(rb, ab, indexing="ij")
        RB = RB.ravel(); TB = TB.ravel()
        xb = RB * np.cos(TB); yb = RB * np.sin(TB)
        pos_b = np.stack([xb, yb, np.full_like(xb, 0.14)], axis=1)
        nor_b = np.tile(np.array([0.0, 0.0, 1.0]), (len(xb), 1))
        # stationary side-wall samples (normal points into the liquid)
        aw = np.arange(24) * (2.0 * np.pi / 24.0)
        zw = np.array([0.25, 0.55, 0.85, 1.15])
        AW, ZW = np.meshgrid(aw, zw, indexing="ij")
        AW = AW.ravel(); ZW = ZW.ravel()
        xw = self.R * np.cos(AW); yw = self.R * np.sin(AW)
        pos_w = np.stack([xw, yw, ZW], axis=1)
        nor_w = np.stack([-np.cos(AW), -np.sin(AW), np.zeros_like(AW)], axis=1)
        self.b_pos = np.concatenate([pos_b, pos_w], axis=0)
        self.b_normal = np.concatenate([nor_b, nor_w], axis=0)
        self.b_is_bottom = np.concatenate(
            [np.ones(len(xb), bool), np.zeros(len(xw), bool)])
        self.b_r = np.concatenate([RB, np.full(len(xw), self.R)])
        self.b_ct = np.concatenate([np.cos(TB), np.cos(AW)])
        self.b_st = np.concatenate([np.sin(TB), np.sin(AW)])
        self.B = len(self.b_pos)

    # ---------------------------------------------------------------- physics
    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _omega_plate(self, t: float) -> float:
        return self.Omega_T * self._smoothstep(t / self.T_ramp)

    def _update_surface(self, dt: float, first: bool) -> None:
        """Quasi-static free surface from radial centrifugal balance + volume."""
        r = self.r_c
        integrand = self.v ** 2 / (self.g * r)          # dzeta/dr
        I = np.cumsum(integrand) * self.dr
        w = 2.0 * np.pi * r * self.dr

        def vol(zc: float) -> float:
            return float(np.sum(np.clip(zc + I, self.h_min, self.wall_height) * w))

        lo, hi = -3.0, 4.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if vol(mid) < self.V0:
                lo = mid
            else:
                hi = mid
        zc = 0.5 * (lo + hi)
        zeta_new = np.clip(zc + I, self.h_min, self.wall_height)
        if first:
            self.dzdt = np.zeros(self.nr)
        else:
            self.dzdt = (zeta_new - self.zeta_prev) / dt
        self.zeta = zeta_new
        self.dzdr = integrand
        h = np.maximum(self.zeta, self.h_min)
        J = np.cumsum(r * self.dzdt) * self.dr           # int_0^r r' dzeta/dt dr'
        self.u_r = -J / (r * h)
        self.u_r[-1] = 0.0                               # impermeable side wall

    def _step(self, dt: float) -> None:
        t = self.t
        Omega = self._omega_plate(t)
        r, v = self.r_c, self.v
        h = np.maximum(self.zeta, self.h_min)

        # momentum diffusion (ghost: odd about axis -> v(0)=0; wall -> v=0)
        ve = np.empty(self.nr + 2)
        ve[1:-1] = v
        ve[0] = -v[0]
        ve[-1] = 0.0
        lap = ((ve[2:] - 2.0 * v + ve[:-2]) / self.dr ** 2
               + (ve[2:] - ve[:-2]) / (2.0 * r * self.dr)
               - v / r ** 2)
        v_new = v + dt * self.nu_t * lap

        # quadratic bottom drag toward the plate velocity
        slip = Omega * r - v_new
        v_new = v_new + dt * self.C_d * slip * np.abs(slip) / h

        # stationary side-wall friction layer (semi-implicit for stiffness)
        kw = self.C_w * (r / self.R) ** 6 / h
        v_new = v_new / (1.0 + dt * kw)
        self.v = np.maximum(v_new, 0.0)

        self.zeta_prev = self.zeta
        self._update_surface(dt, first=False)

        # mode-4 instability of the computed base flow (rotational Froude)
        zeta_safe = np.maximum(self.zeta, self.h_min)
        S = float(np.sum(r * self.v / np.sqrt(self.g * zeta_safe)) / np.sum(r))
        sigma = float(np.clip(self.k_s * (S - self.S_c), -0.5, 1.5))
        self.dA = sigma * self.A - self.beta * self.A ** 3
        self.A = float(np.clip(self.A + dt * self.dA, 0.0, 0.14))
        om_f = float(np.sum(self.v * r) / np.sum(r * r))  # mean fluid rotation
        self.omega_p = self.c_p * om_f
        self.phi += dt * self.omega_p

        # advect surface-marker particles with the physical velocity field
        ur_p = np.interp(self.r_p, r, self.u_r)
        v_p = np.interp(self.r_p, r, self.v)
        self.r_p = np.clip(self.r_p + dt * ur_p, 0.0, self.R - 0.015)
        self.th_p += dt * v_p / np.maximum(self.r_p, 0.05)

        zeta_p = np.interp(self.r_p, r, self.zeta)
        dzdr_p = np.interp(self.r_p, r, self.dzdr)
        dzdt_p = np.interp(self.r_p, r, self.dzdt)
        ur_p = np.interp(self.r_p, r, self.u_r)
        v_p = np.interp(self.r_p, r, self.v)
        rho_p = self.r_p / self.R
        f = 4.0 * rho_p * (1.0 - rho_p)
        fp = 4.0 * (1.0 - 2.0 * rho_p) / self.R
        c4 = np.cos(self.mode * (self.th_p - self.phi))
        s4 = np.sin(self.mode * (self.th_p - self.phi))
        ztot = zeta_p + self.A * f * c4
        self.z_p = np.maximum(ztot, self.h_min)
        # material derivative of the surface elevation
        self.w_p = (dzdt_p
                    + self.dA * f * c4
                    + self.A * f * self.mode * self.omega_p * s4
                    + ur_p * (dzdr_p + self.A * fp * c4)
                    + (v_p / np.maximum(self.r_p, 0.05)) * (-self.mode * self.A * f * s4))
        self.t += dt

    # ----------------------------------------------------------------- output
    def advance_to(self, time: float):
        time = float(time)
        if time < self.t:
            time = self.t
        while self.t < time - 1e-9:
            self._step(min(self.dt_base, time - self.t))
        return self._frame(time)

    def _frame(self, time: float) -> StateFrame:
        r = self.r_c
        ur = np.interp(self.r_p, r, self.u_r)
        vt = np.interp(self.r_p, r, self.v)
        ct, st = np.cos(self.th_p), np.sin(self.th_p)
        position = np.stack([self.r_p * ct, self.r_p * st, self.z_p], axis=1)
        velocity = np.stack([ur * ct - vt * st,
                             ur * st + vt * ct,
                             self.w_p], axis=1)

        # free-surface elevation on the diagnostic grid
        zeta_g = np.interp(self.rr_g.ravel(), r, self.zeta).reshape(self.rr_g.shape)
        c4g = np.cos(self.mode * (self.th_g - self.phi))
        s4g = np.sin(self.mode * (self.th_g - self.phi))
        zeta_s = zeta_g + self.A * self.f_g * c4g

        level_set = np.maximum(self.rr_g[..., None] - self.R,
                               self.zz_g - zeta_s[..., None])
        interior = level_set < -1.5 * self.h_sp

        # diagnostic velocity grid
        ur_g = np.interp(self.rr_g.ravel(), r, self.u_r).reshape(self.rr_g.shape)
        vt_g = np.interp(self.rr_g.ravel(), r, self.v).reshape(self.rr_g.shape)
        dzdt_g = np.interp(self.rr_g.ravel(), r, self.dzdt).reshape(self.rr_g.shape)
        dzdr_g = np.interp(self.rr_g.ravel(), r, self.dzdr).reshape(self.rr_g.shape)
        w_s = (dzdt_g
               + self.dA * self.f_g * c4g
               + self.A * self.f_g * self.mode * self.omega_p * s4g
               + ur_g * (dzdr_g + self.A * self.fp_g * c4g)
               + (vt_g / np.maximum(self.rr_g, 0.05)) * (-self.mode * self.A * self.f_g * s4g))
        ux = ur_g * self.cos_g - vt_g * self.sin_g
        uy = ur_g * self.sin_g + vt_g * self.cos_g
        wet = self.zz_g < zeta_s[..., None]
        zfrac = np.clip(self.zz_g / np.maximum(zeta_s[..., None], self.h_min), 0.0, 1.0)
        vf = np.zeros(self.grid_shape + (3,))
        vf[..., 0] = np.where(wet, ux[..., None], 0.0)
        vf[..., 1] = np.where(wet, uy[..., None], 0.0)
        vf[..., 2] = np.where(wet, w_s[..., None] * zfrac, 0.0)

        # boundary samples
        Omega_now = self._omega_plate(self.t)
        btarget = np.zeros((self.B, 3))
        bb = self.b_is_bottom
        btarget[bb, 0] = -Omega_now * self.b_pos[bb, 1]
        btarget[bb, 1] = Omega_now * self.b_pos[bb, 0]
        ur_b = np.interp(self.b_r, r, self.u_r)
        vt_b = np.interp(self.b_r, r, self.v)
        # fluid-side velocity sampled just inside the boundary layers
        vt_b = np.where(bb, 0.85 * vt_b, 0.05 * vt_b)
        ur_b = np.where(bb, ur_b, 0.0)
        bvel = np.zeros((self.B, 3))
        bvel[:, 0] = ur_b * self.b_ct - vt_b * self.b_st
        bvel[:, 1] = ur_b * self.b_st + vt_b * self.b_ct

        fields = {
            "boundary_normal": self.b_normal.copy(),
            "boundary_target_velocity": btarget,
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(1),
            "liquid_interior_mask": interior,
            "liquid_level_set": level_set,
            "velocity_field": vf,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.p_rad.copy(),
        }
        frame = StateFrame(
            float(time), position, velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes=attributes, fields=fields,
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "model": "reduced-order rotating shallow-water base flow with "
                     "Landau mode-4 centrifugal surface instability",
            "mode_amplitude_m": float(self.A),
            "pattern_phase_rad": float(self.phi % (2.0 * np.pi)),
            "pattern_rate_rad_s": float(self.omega_p),
            "final_time_s": float(self.t),
            "liquid_volume_m3": float(self.V0),
        }
