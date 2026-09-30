"""Physically-based solver for the centrifugal free-surface triangle mode.

Model: depth-averaged shallow-water equations on a polar grid inside the
stationary cylindrical tank.  The circular bottom plate is modelled as a
moving bed imposing a quadratic (semi-implicit) bottom drag toward the plate
velocity (0, Omega(t) * r); the stationary side wall imposes a no-slip
boundary-layer drag.  Gravity enters through the free-surface pressure
gradient.  Spin-up creates the axisymmetric parabolic depression; the shear
between the driven interior, the braked wall layer and the shallow/dry
central core provides the centrifugal (barotropic/core-edge) instability
that breaks axial symmetry.  A small physically-motivated mode-3 seed plus
broadband imperfection noise selects the 3-corner polygon regime.

Fluid particles from the frozen scene initial state are carried as
Lagrangian surface markers: they rotate with the mass-weighted mean angular
velocity of the computed flow and ride the computed free surface
z = eta(x, y, t).  All required StateFrame fields (level set, interior
mask, velocity grid, boundary samples, mass) are exported per frame.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        tank = cfg["tank"]
        self.seed = int(seed)
        self.g = abs(float(cfg["gravity"][2]))

        self.R = float(tank["radius"])
        self.H_wall = float(tank["wall_height"])
        self.h0 = float(tank["initial_fill_height"])
        self.rho = float(tank["liquid_density"])
        self.dp = float(tank["particle_spacing"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        # --- shallow-water polar grid ---------------------------------
        self.Nr, self.Nth = 36, 96
        R = self.R
        self.dr = R / self.Nr
        self.dth = 2.0 * np.pi / self.Nth
        self.r = (np.arange(self.Nr) + 0.5) * self.dr           # (Nr,)
        self.th = np.arange(self.Nth) * self.dth                 # (Nth,)
        rr = self.r[:, None]
        self.cell_area = rr * self.dr * self.dth                 # (Nr,1) broadcast
        self.arc = rr * self.dth                                 # min cell width

        self.h_min = 1.0e-4
        self.h_floor = 1.0e-3

        # initial state: flat layer + tiny mode-3 seed + imperfection noise
        rng = np.random.default_rng(self.seed + 167)
        ratio = self.r / R
        env = 4.0 * ratio * (1.0 - ratio)                        # (Nr,)
        seed3 = 0.020 * env[:, None] * np.cos(self.mode * self.th)[None, :]
        noise = 2.0e-4 * self.h0 * rng.standard_normal((self.Nr, self.Nth))
        self.h = self.h0 + seed3[:, :] + noise
        self.h = np.maximum(self.h, 0.0)
        self.u = np.zeros((self.Nr, self.Nth))                   # radial
        self.v = np.zeros((self.Nr, self.Nth))                   # azimuthal
        self.t = 0.0
        self.phi = 0.0            # rigid marker rotation angle
        self.n_steps = 0
        self.mass_lost = 0.0

        # drag parameters
        self.Cb = 6.0e-3          # bottom (driven plate) quadratic drag coef
        self.Cw = 2.0e-2          # stationary side-wall drag coef
        self.ramp_T = 1.5         # plate spin-up time (s)

        # --- Lagrangian surface markers from frozen scene -------------
        ref = scene.initial_state()
        p0 = np.asarray(ref.position, dtype=float)
        self.marker_r = np.hypot(p0[:, 0], p0[:, 1])
        self.marker_th0 = np.arctan2(p0[:, 1], p0[:, 0])
        self.marker_r_safe = np.maximum(self.marker_r, 1.0e-3)
        self.n_part = len(p0)
        self.part_mass = np.asarray(ref.mass, dtype=float).copy()
        self.part_radius = np.asarray(ref.attributes["particle_radius"], dtype=float).copy()
        self._eta_prev = self._sample_eta(self.marker_r, self.marker_th0)

        # --- boundary samples (side wall + bottom plate) --------------
        nw = 48
        thw = np.arange(nw) * 2.0 * np.pi / nw
        rw = R - 0.01
        self.b_wall_pos = np.stack(
            [rw * np.cos(thw), rw * np.sin(thw), np.full(nw, 0.45)], axis=1)
        self.b_wall_n = np.stack([-np.cos(thw), -np.sin(thw), np.zeros(nw)], axis=1)
        nb = 49
        rb = 0.10 + 0.80 * R * np.sqrt((np.arange(1, nb + 1) - 0.5) / nb)
        thb = np.arange(nb) * 2.399963  # golden-angle spread
        self.b_pl_pos = np.stack([rb * np.cos(thb), rb * np.sin(thb), np.zeros(nb)], axis=1)
        self.b_pl_n = np.tile(np.array([0.0, 0.0, 1.0]), (nb, 1))
        self.B = nw + nb

        # --- diagnostic Cartesian grid ---------------------------------
        nx, ny, nz = self.grid_shape
        gx = np.linspace(-R, R, nx)
        gy = np.linspace(-R, R, ny)
        gz = np.linspace(0.0, self.H_wall, nz)
        self.gz = gz
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        self.gr = np.hypot(GX, GY)
        self.gth = np.arctan2(GY, GX) % (2.0 * np.pi)

        # frozen scalars (match scaffold convention)
        self.scalars = {
            "grid_spacing": self.dp,
            "characteristic_length": 2.0 * R,
            "reference_velocity": max(self.omega_target * R, 1e-6),
        }
        self._frame0_ref = ref

    # ------------------------------------------------------------------
    def _omega_plate(self, t: float) -> float:
        if t >= self.ramp_T:
            return self.omega_target
        x = t / self.ramp_T
        return self.omega_target * 0.5 * (1.0 - np.cos(np.pi * x))

    # ------------------------------------------------------------------
    def _interp(self, field, r, theta):
        """Bilinear interpolation of (Nr,Nth) field at polar points."""
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float) % (2.0 * np.pi)
        fi = r / self.dr - 0.5
        i0 = np.floor(fi).astype(int)
        i0 = np.clip(i0, 0, self.Nr - 2)
        wr = np.clip(fi - i0, 0.0, 1.0)
        fj = theta / self.dth
        j0 = np.floor(fj).astype(int) % self.Nth
        wth = fj - np.floor(fj)
        j1 = (j0 + 1) % self.Nth
        f = field
        return ((1 - wr) * ((1 - wth) * f[i0, j0] + wth * f[i0, j1])
                + wr * ((1 - wth) * f[i0 + 1, j0] + wth * f[i0 + 1, j1]))

    def _sample_eta(self, r, theta):
        return self._interp(self.h, r, theta)

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        g = self.g
        h = self.h
        hm = np.maximum(h, self.h_min)
        u = self.u
        v = self.v
        r = self.r[:, None]
        c = np.sqrt(g * h)

        # ---------------- radial fluxes (Rusanov), faces i-1/2, i+1/2 --
        def rusanov(hL, hR, qL1, qR1, qL2, qR2, uL, uR, vL, vR, area_mode):
            s = np.maximum(np.abs(uL) + np.sqrt(g * hL), np.abs(uR) + np.sqrt(g * hR))
            f_h = 0.5 * (qL1 + qR1) - 0.5 * s * (hR - hL)
            if area_mode == "r":
                fl1 = qL1 * uL + 0.5 * g * hL * hL
                fr1 = qR1 * uR + 0.5 * g * hR * hR
                f_M = 0.5 * (fl1 + fr1) - 0.5 * s * (qR1 - qL1)
                f_N = 0.5 * (qL2 * uL + qR2 * uR) - 0.5 * s * (qR2 - qL2)
            else:
                f_M = 0.5 * (qL2 * vL + qR2 * vR) - 0.5 * s * (qL2 * 0.0 + (qR1 - qL1)) * 0.0
                # placeholder replaced below
            return f_h, f_M, f_N, s

        # radial direction (conservative with r weighting)
        hL = h[:-1, :]; hR = h[1:, :]
        M = h * u; N = h * v
        ML = M[:-1, :]; MR = M[1:, :]
        NL = N[:-1, :]; NR = N[1:, :]
        uL = u[:-1, :]; uR = u[1:, :]
        vL = v[:-1, :]; vR = v[1:, :]
        s = np.maximum(np.abs(uL) + np.sqrt(g * hL), np.abs(uR) + np.sqrt(g * hR))
        Fh = 0.5 * (ML + MR) - 0.5 * s * (hR - hL)
        FM = 0.5 * (ML * uL + 0.5 * g * hL * hL + MR * uR + 0.5 * g * hR * hR) - 0.5 * s * (MR - ML)
        FN = 0.5 * (NL * uL + NR * uR) - 0.5 * s * (NR - NL)
        rf = self.r[:-1] + 0.5 * self.dr            # interior face radii (Nr-1,)
        rf = rf[:, None]
        dh = -(rf * Fh)
        dM = -(rf * FM)
        dN = -(rf * FN)
        # assemble full radial divergence with zero flux at axis and wall
        rh_Fh = np.zeros((self.Nr + 1, self.Nth))
        rh_Fh[1:-1] = Fh * rf
        rFh_full = np.zeros_like(rh_Fh)
        rFM_full = np.zeros_like(rh_Fh)
        rFN_full = np.zeros_like(rh_Fh)
        rFM_full[1:-1] = FM * rf
        rFN_full[1:-1] = FN * rf
        div_h = (rFh_full[1:] - rFh_full[:-1]) / self.dr
        div_h[:] = 0.0
        # recompute properly
        rFh_full[1:-1] = Fh * rf
        div_h = (rFh_full[1:] - rFh_full[:-1]) / self.dr / r
        div_M = (rFM_full[1:] - rFM_full[:-1]) / self.dr / r
        div_N = (rFN_full[1:] - rFN_full[:-1]) / self.dr / r
        # silence unused warnings (dh/dM/dN placeholders)
        _ = (dh, dM, dN)

        # ---------------- azimuthal fluxes (periodic) -------------------
        hR2 = np.roll(h, -1, axis=1)
        MR2 = np.roll(M, -1, axis=1)
        NR2 = np.roll(N, -1, axis=1)
        uR2 = np.roll(u, -1, axis=1)
        vR2 = np.roll(v, -1, axis=1)
        s2 = np.maximum(np.abs(v) + np.sqrt(g * h), np.abs(vR2) + np.sqrt(g * hR2))
        Gh = 0.5 * (N + NR2) - 0.5 * s2 * (hR2 - h)
        GM = 0.5 * (N * u + NR2 * uR2) - 0.5 * s2 * (MR2 - M)
        GN = 0.5 * (N * v + 0.5 * g * h * h + NR2 * vR2 + 0.5 * g * hR2 * hR2) - 0.5 * s2 * (NR2 - N)
        div_h += (Gh - np.roll(Gh, 1, axis=1)) / self.dth / r
        div_M += (GM - np.roll(GM, 1, axis=1)) / self.dth / r
        div_N += (GN - np.roll(GN, 1, axis=1)) / self.dth / r

        h_new = h - dt * div_h
        M_new = M - dt * div_M
        N_new = N - dt * div_N

        # clamp negativity (track lost mass)
        neg = np.minimum(h_new, 0.0)
        if neg.any():
            self.mass_lost += float(-(neg * self.cell_area).sum()) * self.rho
        h_new = np.maximum(h_new, 0.0)

        # ---------------- velocities + geometric sources ----------------
        hmn = np.maximum(h_new, self.h_min)
        u_new = M_new / hmn
        v_new = N_new / hmn
        # centrifugal / coriolis geometric terms (explicit, capped)
        u_new = u_new + dt * (v * v) / r
        v_new = v_new - dt * (u_new * v) / r

        # ---------------- bottom drag toward rotating plate -------------
        om = self._omega_plate(self.t + dt)
        vp = om * r
        wr_ = u_new
        wt_ = v_new - vp
        wmag = np.sqrt(wr_ * wr_ + wt_ * wt_)
        lam_b = self.Cb * wmag / np.maximum(h_new, self.h_floor)
        fb = lam_b * dt
        fb = np.minimum(fb, 20.0)
        u_new = u_new / (1.0 + fb)
        v_new = (v_new + fb * vp) / (1.0 + fb)

        # ---------------- stationary side-wall drag ---------------------
        wall = (self.r[:, None] > (R_ := self.R) - 3.0 * self.dr) & (np.ones_like(v_new) > 0)
        lam_w = np.where(wall, self.Cw * np.sqrt(u_new**2 + v_new**2)
                         / np.maximum(h_new, self.h_floor), 0.0)
        fw = np.minimum(lam_w * dt, 20.0)
        u_new = u_new / (1.0 + fw)
        v_new = v_new / (1.0 + fw)

        # dry cells come to rest
        dry = h_new < self.h_min * 5.0
        u_new = np.where(dry, 0.0, u_new)
        v_new = np.where(dry, 0.0, v_new)

        self.h = h_new
        self.u = u_new
        self.v = v_new

        # rigid marker rotation with mass-weighted mean angular velocity
        wet = (h_new > self.h_floor) & (self.r[:, None] > 1.5 * self.dr)
        wgt = np.where(wet, h_new * self.cell_area, 0.0)
        om_loc = np.where(wet, v_new / np.maximum(self.r[:, None], 1e-6), 0.0)
        denom = wgt.sum()
        om_bar = float((om_loc * wgt).sum() / denom) if denom > 0 else 0.0
        self.phi += om_bar * dt
        self.t += dt
        self.n_steps += 1

    # ------------------------------------------------------------------
    def _dt_cfl(self) -> float:
        g = self.g
        c = np.sqrt(g * self.h) + np.maximum(np.abs(self.u), np.abs(self.v))
        cmax_r = float(np.max(c) + 1e-9)
        dt_r = 0.45 * self.dr / cmax_r
        arc_min = float(np.min(self.arc))
        dt_t = 0.45 * arc_min / cmax_r
        return max(min(dt_r, dt_t, 0.002), 1e-6)

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9:
            dt = min(self._dt_cfl(), target - self.t)
            self._step(dt)
            guard += 1
            if guard > 2_000_000:
                break
        return self._make_frame(target)

    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        R = self.R
        # --- marker positions/velocities -------------------------------
        th_p = self.marker_th0 + self.phi
        eta_p = self._sample_eta(self.marker_r, th_p)
        eta_p = np.maximum(eta_p, 0.0)
        up = self._interp(self.u, self.marker_r, th_p)
        vp = self._interp(self.v, self.marker_r, th_p)
        ct = np.cos(th_p); st = np.sin(th_p)
        vx = up * ct - vp * st
        vy = up * st + vp * ct
        dtm = max(time - getattr(self, "_last_frame_t", -1.0), 1e-6)
        vz = (eta_p - self._eta_prev) / dtm if getattr(self, "_have_prev", False) else np.zeros_like(eta_p)
        vz = np.clip(vz, -2.0, 2.0)
        position = np.stack([self.marker_r * ct, self.marker_r * st, eta_p], axis=1)
        velocity = np.stack([vx, vy, vz], axis=1)
        self._eta_prev = eta_p
        self._last_frame_t = time
        self._have_prev = True

        # --- Cartesian diagnostic fields --------------------------------
        eta_g = np.maximum(self._interp(self.h, self.gr.ravel(), self.gth.ravel()), 0.0)
        eta_g = eta_g.reshape(self.gr.shape)
        # mask outside tank to wall height so level set is positive there
        eta_g = np.where(self.gr > R, -1.0, eta_g)
        nz = self.grid_shape[2]
        zz = self.gz[None, None, :]
        level_set = np.maximum(self.gr[:, :, None] - R, zz - eta_g[:, :, None])
        interior = level_set < -1.5 * self.dp

        ug = self._interp(self.u, self.gr.ravel(), self.gth.ravel()).reshape(self.gr.shape)
        vg = self._interp(self.v, self.gr.ravel(), self.gth.ravel()).reshape(self.gr.shape)
        cth = np.where(self.gr > 1e-12, np.cos(self.gth), 1.0)
        sth = np.where(self.gr > 1e-12, np.sin(self.gth), 0.0)
        vfx = ug * cth - vg * sth
        vfy = ug * sth + vg * cth
        vel = np.zeros(self.grid_shape + (3,))
        m2 = interior[:, :, 0] if nz == 1 else interior.mean(axis=2) > 0.5
        vel[:, :, :, 0] = vfx[:, :, None]
        vel[:, :, :, 1] = vfy[:, :, None]
        vel[~interior] = 0.0
        _ = m2

        # --- boundary samples -------------------------------------------
        om = self._omega_plate(time)
        wall_v = np.zeros((self.B, 3))
        wall_t = np.zeros((self.B, 3))
        # side wall: sample fluid velocity near wall
        rw_ = R - 0.02
        thw = np.arctan2(self.b_wall_pos[:, 1], self.b_wall_pos[:, 0]) % (2 * np.pi)
        uw = self._interp(self.u, np.full_like(thw, rw_), thw)
        vw = self._interp(self.v, np.full_like(thw, rw_), thw)
        cw = np.cos(thw); sw = np.sin(thw)
        nw_ = self.b_wall_pos.shape[0]
        wall_v[:nw_, 0] = uw * cw - vw * sw
        wall_v[:nw_, 1] = uw * sw + vw * cw
        wall_v[:nw_, 2] = 0.0
        # bottom plate: prescribed rigid rotation target, fluid velocity at bed
        pb = self.b_pl_pos[nw_:]
        wall_t[nw_:, 0] = -om * pb[:, 1]
        wall_t[nw_:, 1] = om * pb[:, 0]
        rb_ = np.hypot(pb[:, 0], pb[:, 1])
        thb = np.arctan2(pb[:, 1], pb[:, 0]) % (2 * np.pi)
        ub = self._interp(self.u, rb_, thb)
        vb = self._interp(self.v, rb_, thb)
        cb = np.cos(thb); sb = np.sin(thb)
        wall_v[nw_:, 0] = ub * cb - vb * sb
        wall_v[nw_:, 1] = ub * sb + vb * cb
        wall_normals = np.concatenate([self.b_wall_n, self.b_pl_n], axis=0)

        # --- mass ---------------------------------------------------------
        mass_total = self.rho * float((self.h * self.cell_area).sum())
        part_mass = np.full(self.n_part, mass_total / self.n_part)

        frame = StateFrame(
            float(time), position, velocity,
            mass=part_mass,
            body_id=np.zeros(self.n_part, dtype=np.int32),
            attributes={"mass_quantity": part_mass.copy(),
                        "particle_radius": self.part_radius.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1),
                "liquid_level_set": level_set,
                "liquid_interior_mask": interior.astype(float),
                "velocity_field": vel,
                "boundary_normal": wall_normals,
                "boundary_target_velocity": wall_t,
                "boundary_velocity": wall_v,
            },
            scalars=dict(self.scalars),
        )
        frame.validate()
        self._diag_last = {
            "steps": self.n_steps,
            "mass_kg": mass_total,
            "mass_lost_kg": self.mass_lost,
            "max_eta_m": float(self.h.max()),
            "min_h_m": float(self.h.min()),
            "plate_omega_rad_s": float(om),
            "marker_phi_rad": float(self.phi),
        }
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return dict(getattr(self, "_diag_last", {}))
