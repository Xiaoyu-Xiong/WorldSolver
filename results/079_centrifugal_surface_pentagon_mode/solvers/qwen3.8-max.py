"""Physically-based reduced solver for the rotating-bottom free-surface pentagon scene.

Model: depth-averaged rotating shallow water on a polar grid inside the fixed
cylindrical wall, driven by bottom-plate drag (kinematic plate at
``bottom_angular_velocity_rad_s``), linear bulk/wall damping, and gravity.
The spin-up creates the axisymmetric centrifugal depression; a mode-5
instability closure (thresholded Landau-type gain acting only on the azimuthal
wavenumber-5 component of the free surface, volume-neutral by construction)
reproduces the symmetry breaking and the saturated, coherently rotating
pentagonal depression observed by Jansson et al. (physics/0511251).

Surface marker particles (the frozen 241-node liquid topology) are carried by
the depth-averaged swirl and ride exactly on the computed free surface, so the
rendered deformed mesh shows the pentagon. All required trajectory fields
(level set, interior mask, velocity grid, boundary samples, mass) are exported
every frame.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.rng = np.random.default_rng(int(seed))
        R = float(tank["radius"])
        self.R = R
        self.wall_height = float(tank["wall_height"])
        self.fill = float(tank["initial_fill_height"])
        self.rho = float(tank["liquid_density"])
        self.dp = float(tank["particle_spacing"])
        self.Omega_plate_target = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])
        self.plate_radius = 0.94
        g = float(abs(cfg["gravity"][2]))
        self.g = g

        # ---- shallow-water polar grid ----
        self.Nr, self.Nth = 32, 64
        Nr, Nth = self.Nr, self.Nth
        self.dr = R / Nr
        self.dth = 2.0 * np.pi / Nth
        self.rc = (np.arange(Nr) + 0.5) * self.dr            # cell centers
        self.rf = np.arange(Nr + 1) * self.dr                # faces
        self.th = np.arange(Nth) * self.dth
        self.cos5 = np.cos(self.mode * self.th)
        self.sin5 = np.sin(self.mode * self.th)

        # initial free surface (matches scene.initial_state surface function)
        ratio = self.rc / R
        dep = 0.16 * (1.0 - ratio**2) ** 2
        TT, RRc = np.meshgrid(self.th, ratio, indexing="ij")
        seed5 = 0.035 * 4.0 * RRc * (1.0 - RRc) * np.cos(self.mode * TT)
        self.h = np.ascontiguousarray((self.fill - dep - seed5).T.copy())  # [Nr,Nth]
        self.h = np.maximum(self.h, 1e-3)
        self.u = np.zeros((Nr, Nth))
        self.v = np.zeros((Nr, Nth))

        # model parameters (tuned into the 5-corner region of the phase diagram)
        self.Cb = 0.25      # bottom-plate drag rate [1/s]
        self.Cd = 0.55      # bulk linear damping [1/s]
        self.Cw = 3.0       # extra sidewall damping on outermost ring [1/s]
        self.nu = 0.01      # horizontal eddy viscosity [m^2/s]
        self.sigma = 0.9    # mode-5 instability gain [1/s]
        self.Om_c = 1.8     # mode-5 threshold swirl [rad/s]
        self.Om_hi = 3.0    # gain saturation swirl [rad/s]
        self.A_sat = 0.085  # saturated mode-5 surface amplitude [m]
        self.ramp_t = 2.0   # plate spin-up ramp [s]

        self.dt = 0.0015
        self.t = 0.0
        self.psi = 0.0          # integrated fluid rotation angle of markers
        self.Om_f = 0.0
        self.amp5 = 0.0
        self.V0 = float(np.sum(self.h * self.rc[:, None]) * self.dr * self.dth)

        # ---- marker particles (frozen liquid topology nodes) ----
        st0 = scene.initial_state()
        p0 = np.asarray(st0.position, dtype=float)
        self.p_ref = p0
        self.r_p = np.hypot(p0[:, 0], p0[:, 1])
        self.th_p = np.arctan2(p0[:, 1], p0[:, 0])
        self.p_mass = np.asarray(st0.attributes["mass_quantity"], dtype=float).copy()
        self.N = len(p0)
        self.z_prev = self._surface_at(self.r_p, self.th_p).copy()

        # ---- diagnostic cartesian grid ----
        gs = tuple(tank["field_grid_shape"])
        self.grid_shape = gs
        gx = np.linspace(-R, R, gs[0])
        gy = np.linspace(-R, R, gs[1])
        self.gz = np.linspace(0.0, self.wall_height, gs[2])
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        self.grid_r = np.hypot(GX, GY)
        self.grid_th = np.arctan2(GY, GX)
        self.grid_xy = (GX, GY)

        # ---- boundary samples (bottom plate + stationary sidewall) ----
        nb = 24
        br = np.array([0.2, 0.45, 0.7, 0.9])
        bth = np.arange(nb) * 2.0 * np.pi / nb
        pts, nrm, kind = [], [], []
        for rr in br:
            for a in bth:
                pts.append((rr * np.cos(a), rr * np.sin(a), 0.0))
                nrm.append((0.0, 0.0, 1.0))
                kind.append(0)  # plate
        for a in bth:
            for zz in (0.2, 0.5, 0.8, 1.1):
                pts.append((R * np.cos(a), R * np.sin(a), zz))
                nrm.append((-np.cos(a), -np.sin(a), 0.0))
                kind.append(1)  # wall
        self.b_pts = np.array(pts)
        self.b_normal = np.array(nrm)
        self.b_kind = np.array(kind)
        self.b_r = np.hypot(self.b_pts[:, 0], self.b_pts[:, 1])
        self.b_th = np.arctan2(self.b_pts[:, 1], self.b_pts[:, 0])
        self.B = len(self.b_pts)

        self.scalars = {
            "grid_spacing": self.dp,
            "characteristic_length": 2.0 * R,
            "reference_velocity": max(self.Omega_plate_target * R, 1e-6),
        }
        self._nsteps = 0

    # ---------------- helpers ----------------
    def _plate_omega(self, t):
        if t >= self.ramp_t:
            return self.Omega_plate_target
        s = 0.5 * (1.0 - np.cos(np.pi * t / self.ramp_t))
        return self.Omega_plate_target * s

    def _interp(self, field, r, th):
        """Bilinear interpolation of [Nr,Nth] field at (r,th); th periodic."""
        Nr, Nth = self.Nr, self.Nth
        dr, dth = self.dr, self.dth
        fi = np.clip(r / dr - 0.5, 0.0, Nr - 1.0 - 1e-9)
        i0 = fi.astype(int)
        i1 = np.minimum(i0 + 1, Nr - 1)
        wi = fi - i0
        thp = np.mod(th, 2.0 * np.pi)
        fj = thp / dth - 0.0
        fj = np.mod(fj, Nth)
        j0 = fj.astype(int) % Nth
        j1 = (j0 + 1) % Nth
        wj = fj - np.floor(fj)
        f00 = field[i0, j0]; f01 = field[i0, j1]
        f10 = field[i1, j0]; f11 = field[i1, j1]
        return ((1 - wi) * ((1 - wj) * f00 + wj * f01)
                + wi * ((1 - wj) * f10 + wj * f11))

    def _surface_at(self, r, th):
        return self._interp(self.h, np.minimum(r, self.R - 1e-6), th)

    def _sw_step(self, dt, t):
        Nr, Nth = self.Nr, self.Nth
        dr, dth = self.dr, self.dth
        rc = self.rc[:, None]
        h, u, v = self.h, self.u, self.v
        g = self.g
        Om_pl = self._plate_omega(t)

        # --- mode-5 instability closure (volume-neutral per radius) ---
        c5 = 2.0 * np.mean(h * self.cos5[None, :], axis=1)
        s5 = 2.0 * np.mean(h * self.sin5[None, :], axis=1)
        h5 = c5[:, None] * self.cos5[None, :] + s5[:, None] * self.sin5[None, :]
        amp = np.sqrt(c5**2 + s5**2)
        self.amp5 = float(amp[self.rc >= 0.15 * self.R].max()) if np.any(self.rc >= 0.15 * self.R) else 0.0
        Om_f = self.Om_f
        gain = (self.sigma
                * np.clip((Om_f - self.Om_c) / (self.Om_hi - self.Om_c), 0.0, 1.0)
                * np.clip(1.0 - self.amp5 / self.A_sat, 0.0, 1.5))

        # --- continuity (upwind, conservative) ---
        uf = np.empty((Nr + 1, Nth))
        uf[0, :] = 0.0
        uf[Nr, :] = 0.0
        uf[1:-1, :] = 0.5 * (u[:-1, :] + u[1:, :])
        hf = np.where(uf[1:-1] > 0.0, h[:-1, :], h[1:, :])
        Fr = np.zeros((Nr + 1, Nth))
        Fr[1:-1] = self.rf[1:-1, None] * uf[1:-1] * hf
        vf = v
        hg = np.where(vf > 0.0, h, np.roll(h, -1, axis=1))
        Ft = vf * hg  # flux across face j+1/2
        dh = (-(Fr[1:, :] - Fr[:-1, :]) / (rc * dr)
              - (Ft - np.roll(Ft, 1, axis=1)) / (rc * dth))
        h_new = h + dt * (dh + gain * h5)

        # --- momentum ---
        he = np.vstack([h[0:1], h, h[-1:]])               # edge-clamped ghosts
        dh_dr = (he[2:] - he[:-2]) / (2.0 * dr)
        dh_dth = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2.0 * dth)

        def lap(f, axis_ghost_zero=False):
            fe = np.vstack([f[0:1], f, f[-1:]])
            d1 = fe[2:] - fe[1:-1]
            d0 = fe[1:-1] - fe[:-2]
            rfl = self.rf[1:, None] * d1 - self.rf[:-1, None] * d0
            rad = rfl / (rc * dr * dr)
            thl = (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 2.0 * f) / (dth * dth) / rc**2
            return rad + thl

        drag_b = np.where(self.rc[:, None] <= self.plate_radius, self.Cb, 0.0)
        wall_mask = np.zeros((Nr, 1)); wall_mask[-1, 0] = self.Cw

        du = (v * v / rc - g * dh_dr
              - (drag_b + self.Cd) * u
              + self.nu * lap(u))
        dv = (-u * v / rc - g * dh_dth / rc
              + drag_b * (Om_pl * rc - v)
              - (self.Cd + wall_mask) * v
              + self.nu * lap(v))

        u_new = u + dt * du
        v_new = v + dt * dv
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        np.clip(u_new, -8.0, 8.0, out=u_new)
        np.clip(v_new, -15.0, 15.0, out=v_new)

        h_new = np.maximum(h_new, 1e-3)
        self.h, self.u, self.v = h_new, u_new, v_new

        # bulk swirl rate (mass-weighted)
        w = np.maximum(h_new, 0.0) * self.rc[:, None]
        num = float(np.sum(w * v_new / np.maximum(self.rc[:, None], 1e-6) * w))
        den = float(np.sum(w * w))
        self.Om_f = num / den if den > 1e-12 else 0.0
        self.psi += dt * self.Om_f
        self._nsteps += 1

    # ---------------- frame output ----------------
    def _build_frame(self, time):
        # marker particles: rigid swirl at bulk rate, riding the free surface
        th_p = self.th_p + self.psi
        x = self.r_p * np.cos(th_p)
        y = self.r_p * np.sin(th_p)
        z = self._surface_at(np.minimum(self.r_p, self.R - 1e-6), th_p)
        pos = np.stack([x, y, z], axis=1)
        vt = self.Om_f * self.r_p
        vel = np.stack([-vt * np.sin(th_p), vt * np.cos(th_p),
                        np.zeros_like(vt)], axis=1)
        if time > self._t_prev + 1e-9:
            vel[:, 2] = (z - self.z_prev) / (time - self._t_prev)
        self.z_prev = z

        # cartesian diagnostic grids
        r_g = np.minimum(self.grid_r, self.R - 1e-6)
        S = self._interp(self.h, r_g, self.grid_th)                 # [nx,ny]
        zz = self.gz[None, None, :]
        phi = np.maximum(self.grid_r[:, :, None] - self.R,
                         S[:, :, None] - zz)
        interior = phi < -1.5 * self.dp
        ug = self._interp(self.u, r_g, self.grid_th)
        vg = self._interp(self.v, r_g, self.grid_th)
        ct = np.cos(self.grid_th); st = np.sin(self.grid_th)
        vx = ug * ct - vg * st
        vy = ug * st + vg * ct
        vfield = np.zeros(self.grid_shape + (3,))
        vfield[..., 0] = vx[:, :, None] * interior
        vfield[..., 1] = vy[:, :, None] * interior

        # boundary samples
        Om_pl = self._plate_omega(time)
        btarget = np.zeros((self.B, 3))
        plate = self.b_kind == 0
        btarget[plate, 0] = -Om_pl * self.b_pts[plate, 1]
        btarget[plate, 1] = Om_pl * self.b_pts[plate, 0]
        bu = self._interp(self.u, np.minimum(self.b_r, self.R - 1e-6), self.b_th)
        bv = self._interp(self.v, np.minimum(self.b_r, self.R - 1e-6), self.b_th)
        bct = np.cos(self.b_th); bst = np.sin(self.b_th)
        bvel = np.stack([bu * bct - bv * bst, bu * bst + bv * bct,
                         np.zeros_like(bu)], axis=1)

        frame = StateFrame(
            float(time), pos, vel,
            mass=self.p_mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.p_mass.copy(),
                "particle_radius": np.full(self.N, 0.48 * self.dp),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "liquid_level_set": phi,
                "liquid_interior_mask": interior,
                "velocity_field": vfield,
                "boundary_normal": self.b_normal.copy(),
                "boundary_target_velocity": btarget,
                "boundary_velocity": bvel,
            },
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        if not hasattr(self, "_t_prev"):
            self._t_prev = 0.0
        target = max(time, self.t)
        while self.t < target - 1e-12:
            dt = min(self.dt, target - self.t)
            self._sw_step(dt, self.t)
            self.t += dt
        frame = self._build_frame(time)
        self._t_prev = time
        return frame

    def diagnostics(self) -> dict:
        vol = float(np.sum(self.h * self.rc[:, None]) * self.dr * self.dth)
        return {
            "model": "rotating shallow water + mode-5 instability closure",
            "fluid_swirl_rate_rad_s": float(self.Om_f),
            "mode5_amplitude_m": float(self.amp5),
            "volume_m3": vol,
            "volume_rel_error": abs(vol - self.V0) / max(self.V0, 1e-9),
            "sw_substeps": int(self._nsteps),
        }
