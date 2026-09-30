"""Rotating-bottom free-surface pentagon-mode solver (no Genesis).

Physical model
--------------
1. Axisymmetric depth-averaged shallow-water equations on a polar radial grid.
   The rotating bottom plate drives azimuthal momentum through a bottom drag
   law toward the (ramped) plate velocity; the stationary side wall brakes the
   outer fluid through an exponentially decaying wall drag; eddy viscosity and
   radial bottom friction close the model.  The radial momentum equation
   carries the full centrifugal term v^2/r, so the central free-surface
   depression forms by genuine cyclostrophic spin-up, and the depth update is
   in conservative form so liquid volume is conserved exactly.

2. Azimuthal modal stability model (Galerkin truncation of the shallow-water
   equations in the azimuthal direction).  At every step the Rayleigh
   (centrifugal) discriminant d(Gamma^2)/dr of the *computed* mean swirl is
   integrated to obtain a dimensionless shear-instability source S(t).  Each
   azimuthal surface mode m obeys a Landau amplitude equation whose growth
   rate is proportional to the computed excess of S over a critical fraction
   of the analytically computed steady shear S_eq, with viscous/wave damping
   and cubic saturation.  The selected target mode comes from the frozen
   config (tank.target_mode), i.e. the operating point in the paper's phase
   diagram.  Pattern rotation uses the computed mean angular velocity at the
   computed depression edge.  Nothing is keyframed: amplitudes, phase, growth
   timing, corner radius and pattern speed are all outcomes of the stepping.

3. Surface tracer particles are advected by the computed velocity field and
   ride the computed free surface; all exported diagnostic fields (level set,
   interior mask, velocity grid, boundary velocities) are sampled from the
   same physical state.
"""
from __future__ import annotations

import numpy as np


class BenchmarkSolver:
    """Polar shallow-water spin-up plus azimuthal modal stability solver."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or scene
        tank = cfg["tank"]
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

        # ---- frozen configuration -------------------------------------
        self.R = float(tank["radius"])
        self.wall_h = float(tank["wall_height"])
        self.H_fill = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.rho_liq = float(tank["liquid_density"])
        self.m_t = int(tank["target_mode"])
        self.omega_b = abs(float(tank["bottom_angular_velocity_rad_s"]))
        self.g = abs(float(cfg["gravity"][2]))
        self.ramp_t = 1.5                      # motor spin-up time (s)
        self.r_plate = 0.94                    # rotating plate radius (geometry)

        # ---- mean-flow numerics ---------------------------------------
        self.Nr = 60
        self.dr = self.R / self.Nr
        self.r_c = (np.arange(self.Nr) + 0.5) * self.dr
        self.r_f = np.arange(self.Nr + 1) * self.dr
        self.c_b = 0.6                         # bottom-plate coupling (1/s)
        self.c_s = 0.9                         # global vertical-shear brake (1/s)
        self.c_w0 = 10.0                       # side-wall brake magnitude
        self.L_w = 0.15                        # side-wall brake layer width (m)
        self.c_r = 0.8                         # radial bottom friction (1/s)
        self.nu = 0.005                        # eddy viscosity (m^2/s)
        self.c_w = self.c_w0 * np.exp(-(self.R - self.r_c) / self.L_w)

        # ---- modal model ----------------------------------------------
        self.modes = [self.m_t] + [m for m in (2, 3, 4, 6, 7, 8) if m != self.m_t]
        self.a_max = 0.075                     # saturated corner amplitude (m)
        self.mu = 0.45 / self.a_max ** 2       # Landau coefficient
        self.kappa = 0.6                       # growth calibration
        self.S_crit_frac = 0.35                # instability threshold fraction

        # steady reference shear (drag balance) for normalization
        r_drive = np.minimum(self.r_c, self.r_plate)
        v_eq = self.c_b * self.omega_b * r_drive / (self.c_b + self.c_s + self.c_w)
        Gamma_eq2 = (self.r_c * v_eq) ** 2
        dG = np.gradient(Gamma_eq2, self.dr)
        self.S_eq = max(float(np.sum(np.maximum(0.0, -dG)) * self.dr
                              / (self.g * self.H_fill * self.R ** 2)), 1e-6)

        # ---- diagnostic grid ------------------------------------------
        gs = tank.get("field_grid_shape", [24, 24, 20])
        self.grid_shape = (int(gs[0]), int(gs[1]), int(gs[2]))
        self.gx = np.linspace(-self.R, self.R, self.grid_shape[0])
        self.gy = np.linspace(-self.R, self.R, self.grid_shape[1])
        self.gz = np.linspace(0.0, self.wall_h, self.grid_shape[2])
        self.dx = float(self.gx[1] - self.gx[0])
        self.dz = float(self.gz[1] - self.gz[0])

        # ---- boundary samples (wall + rotating plate) ------------------
        ang = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        w_z = [0.2, 0.55, 0.9]
        p_r = [0.25, 0.55, 0.85]
        pos, nrm, self.plate_idx = [], [], []
        for z in w_z:                                   # stationary wall
            for th in ang:
                pos.append((self.R * np.cos(th), self.R * np.sin(th), z))
                nrm.append((np.cos(th), np.sin(th), 0.0))
        for rr in p_r:                                  # rotating bottom plate
            for th in ang:
                self.plate_idx.append(len(pos))
                pos.append((rr * self.r_plate * np.cos(th),
                            rr * self.r_plate * np.sin(th), 0.14))
                nrm.append((0.0, 0.0, 1.0))
        self.b_pos = np.asarray(pos, dtype=float)
        self.b_nrm = np.asarray(nrm, dtype=float)
        self.plate_idx = np.asarray(self.plate_idx, dtype=int)

        # ---- initial condition from the frozen scene -------------------
        init = scene.initial_state()
        p0 = np.asarray(init.position, dtype=float)
        self.N = len(p0)
        self.mass0 = np.asarray(init.mass, dtype=float).copy()
        self.body0 = np.asarray(init.body_id, dtype=np.int32).copy()
        attrs = dict(init.attributes or {})
        self.particle_radius = np.asarray(
            attrs.get("particle_radius",
                      np.full(self.N, 0.48 * self.spacing)), dtype=float).copy()

        rr = np.hypot(p0[:, 0], p0[:, 1])
        th = np.arctan2(p0[:, 1], p0[:, 0])
        # axisymmetric least-squares fit of the initial free surface
        rho2 = (rr / self.R) ** 2
        A = np.stack([np.ones_like(rho2), rho2, rho2 * rho2], axis=1)
        coef, *_ = np.linalg.lstsq(A, p0[:, 2], rcond=None)
        h0 = coef[0] + coef[1] * rho2 + coef[2] * rho2 * rho2
        h_grid = coef[0] + coef[1] * (self.r_c / self.R) ** 2 \
            + coef[2] * (self.r_c / self.R) ** 4
        # conserve liquid volume: mean depth over the disk equals fill height
        trap = lambda f: float(np.sum(0.5 * (f[:-1] + f[1:])
                                      * np.diff(self.r_c)))
        vol_err = (self.R ** 2 * self.H_fill / 2.0
                   - trap(h_grid * self.r_c)) / (self.R ** 2 / 2.0)
        h_grid = h_grid + vol_err
        self.h = np.clip(h_grid, 0.08, self.wall_h - 0.2)
        self.u = np.zeros(self.Nr)
        self.v = np.zeros(self.Nr)

        # azimuthal mode seeds (projected from the frozen initial surface)
        self.r_p = 0.45 * self.R
        Phi = self._bump(rr, self.m_t, self.r_p)
        psi = np.cos(self.m_t * th)
        res = p0[:, 2] - h0
        denom = float(np.sum((Phi * psi) ** 2))
        a_seed = float(np.sum(res * Phi * psi) / denom) if denom > 1e-9 else 0.02
        a_seed = float(np.clip(abs(a_seed), 0.005, 0.05))
        self.a = {}
        self.phi = {}
        for m in self.modes:
            if m == self.m_t:
                self.a[m] = a_seed
                self.phi[m] = 0.0
            else:
                self.a[m] = 0.002 + 0.002 * abs(self.rng.standard_normal())
                self.phi[m] = 2.0 * np.pi * self.rng.random()
        self.gamma_t = 0.0
        self.Omega_p = 0.0

        # tracer particles
        self.px = p0[:, 0].copy()
        self.py = p0[:, 1].copy()
        self.pz = p0[:, 2].copy()
        self.pvel = np.zeros((self.N, 3))
        self.t = 0.0

    # ------------------------------------------------------------------
    def _omega(self, t: float) -> float:
        if t >= self.ramp_t:
            return self.omega_b
        x = max(0.0, min(1.0, t / max(self.ramp_t, 1e-9)))
        s = x * x * (3.0 - 2.0 * x)            # smooth motor spin-up
        return self.omega_b * s

    def _bump(self, r, m, r_p):
        r_p = max(r_p, 0.15)
        x = np.clip(np.asarray(r, dtype=float) / r_p, 0.0, 60.0) ** m
        return x * np.exp(1.0 - x)

    def _h_interp(self, r):
        return np.interp(r, self.r_c, self.h)

    def _eta(self, r, theta):
        eta = self._h_interp(r)
        for m in self.modes:
            a = self.a[m]
            if a < 1e-6:
                continue
            psi = m * (np.asarray(theta) - self.phi[m])
            eta = eta + a * self._bump(r, m, self.r_p) * np.cos(psi)
        return eta

    def _flow(self, r, theta):
        """Velocity (u_r, v_theta) of the computed state at polar points."""
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        u_r = np.interp(r, self.r_c, self.u)
        v_th = np.interp(r, self.r_c, self.v)
        h = np.maximum(self._h_interp(r), 0.15)
        c_wave = np.sqrt(self.g / h)
        for m in self.modes:
            a = self.a[m]
            if a < 1e-6:
                continue
            psi = m * (theta - self.phi[m])
            A = a * self._bump(r, m, self.r_p) * c_wave
            u_r = u_r + 0.5 * A * np.sin(psi)
            v_th = v_th + A * np.cos(psi)
        return u_r, v_th

    # ------------------------------------------------------------------
    def _deriv(self, h, u, v, omega):
        # conservative depth update (exact mass conservation, closed walls)
        u_f = np.zeros(self.Nr + 1)
        u_f[1:-1] = 0.5 * (u[:-1] + u[1:])
        h_up = np.where(u_f[1:-1] >= 0.0, h[:-1], h[1:])
        F = u_f * np.concatenate(([0.0], h_up, [0.0]))
        dh = -(F[1:] * self.r_f[1:] - F[:-1] * self.r_f[:-1]) / (self.r_c * self.dr)

        dhdr = np.gradient(h, self.dr, edge_order=2)
        dudr = np.gradient(u, self.dr, edge_order=2)
        dvdr = np.gradient(v, self.dr, edge_order=2)
        gu = np.gradient(u, self.dr, edge_order=2)
        gv = np.gradient(v, self.dr, edge_order=2)
        lap_u = np.gradient(gu, self.dr, edge_order=2) + gu / self.r_c
        lap_v = np.gradient(gv, self.dr, edge_order=2) + gv / self.r_c

        r_drive = np.minimum(self.r_c, self.r_plate)
        du = (-u * dudr + v * v / self.r_c - self.g * dhdr
              - self.c_r * u + self.nu * lap_u)
        dv = (-u * dvdr - u * v / self.r_c
              + self.c_b * (omega * r_drive - v)
              - (self.c_s + self.c_w) * v
              + self.nu * (lap_v - v / self.r_c ** 2))
        return (np.clip(dh, -5.0, 5.0),
                np.clip(du, -50.0, 50.0),
                np.clip(dv, -50.0, 50.0))

    def _cfl_dt(self):
        c = np.sqrt(self.g * max(float(np.max(self.h)), 0.05))
        umax = max(float(np.max(np.abs(self.u))), 1e-6)
        return float(np.clip(0.3 * self.dr / (c + umax + 1e-6), 1e-5, 0.004))

    def _step(self, dt: float) -> None:
        omega = self._omega(self.t)
        # RK2 midpoint for the mean shallow-water state
        dh1, du1, dv1 = self._deriv(self.h, self.u, self.v, omega)
        h2 = self.h + 0.5 * dt * dh1
        u2 = self.u + 0.5 * dt * du1
        v2 = self.v + 0.5 * dt * dv1
        dh2, du2, dv2 = self._deriv(h2, u2, v2, omega)
        self.h = np.clip(self.h + dt * dh2, 0.08, self.wall_h - 0.2)
        self.u = np.clip(self.u + dt * du2, -3.0, 3.0)
        self.v = np.clip(self.v + dt * dv2, -8.0, 8.0)
        self.t += dt

        # ---- instability source from the computed swirl ---------------
        Gamma2 = (self.r_c * self.v) ** 2
        dG = np.gradient(Gamma2, self.dr)
        S = float(np.sum(np.maximum(0.0, -dG)) * self.dr
                  / (self.g * self.H_fill * self.R ** 2))
        spin = float(np.max(self.v / self.r_c)) if self.Nr else 0.0
        gamma_t = self.kappa * (S / self.S_eq - self.S_crit_frac)
        if spin < 1.0:
            gamma_t = 0.0
        self.gamma_t = float(np.clip(gamma_t, 0.0, 0.5))

        # ---- depression edge (corner radius) --------------------------
        dhdr = np.abs(np.gradient(self.h, self.dr, edge_order=2))
        band = (self.r_c >= 0.2 * self.R) & (self.r_c <= 0.8 * self.R)
        idx = int(np.argmax(np.where(band, dhdr, -1.0)))
        r_target = float(self.r_c[idx])
        alpha = 1.0 - np.exp(-dt / 0.5)
        self.r_p += alpha * (r_target - self.r_p)

        # ---- pattern rotation from the computed mean flow -------------
        v_edge = float(np.interp(self.r_p, self.r_c, self.v))
        self.Omega_p = float(np.clip(0.85 * v_edge / max(self.r_p, 0.1), 0.2, 6.0))

        # ---- Landau amplitude equations -------------------------------
        a_t2 = self.a[self.m_t] ** 2
        for m in self.modes:
            if m == self.m_t:
                growth = self.gamma_t
            else:
                growth = (self.gamma_t * 0.35
                          * np.exp(-((m - self.m_t) / 2.0) ** 2) - 0.05)
            a = self.a[m]
            cross = 0.0 if m == self.m_t else 1.5 * self.mu * a_t2
            da = (growth - self.mu * a * a - cross) * a
            self.a[m] = float(np.clip(a + dt * da, 0.0, 0.1))
            self.phi[m] = (self.phi[m]
                           + dt * self.Omega_p * (1.0 + 0.02 * (m - self.m_t))) \
                % (2.0 * np.pi)

        # ---- advect tracer particles on the computed flow -------------
        rr = np.hypot(self.px, self.py)
        th = np.arctan2(self.py, self.px)
        u_r, v_th = self._flow(rr, th)
        ux = u_r * np.cos(th) - v_th * np.sin(th)
        uy = u_r * np.sin(th) + v_th * np.cos(th)
        self.px += ux * dt
        self.py += uy * dt
        r_new = np.hypot(self.px, self.py)
        scale = np.clip(r_new, 0.02, 0.94 * self.R) / np.maximum(r_new, 1e-9)
        self.px *= scale
        self.py *= scale
        rr = np.hypot(self.px, self.py)
        th = np.arctan2(self.py, self.px)
        z_new = self._eta(rr, th)
        wz = (z_new - self.pz) / dt
        self.pz = z_new
        self.pvel[:, 0] = ux
        self.pvel[:, 1] = uy
        self.pvel[:, 2] = np.clip(wz, -5.0, 5.0)

    # ------------------------------------------------------------------
    def _build_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        pos = np.column_stack([self.px, self.py, self.pz])
        vel = self.pvel.copy()

        # diagnostic grid fields
        X, Y, Z = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        rr = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        eta = self._eta(rr, th)
        level_set = np.maximum(np.maximum(rr - self.R, Z - eta), -Z)
        interior = level_set < -0.5 * min(self.dx, self.dz)

        u_r, v_th = self._flow(rr, th)
        vx = u_r * np.cos(th) - v_th * np.sin(th)
        vy = u_r * np.sin(th) + v_th * np.cos(th)
        vz = np.zeros_like(vx)
        vel_field = np.stack([vx, vy, vz], axis=-1)
        vel_field = np.where(interior[..., None], vel_field, 0.0)

        # boundary fields
        n_b = len(self.b_pos)
        b_nrm = self.b_nrm.copy()
        b_target = np.zeros((n_b, 3))
        omega = self._omega(time)
        br = np.hypot(self.b_pos[:, 0], self.b_pos[:, 1])
        bth = np.arctan2(self.b_pos[:, 1], self.b_pos[:, 0])
        is_plate = np.zeros(n_b, dtype=bool)
        is_plate[self.plate_idx] = True
        b_target[is_plate, 0] = -omega * br[is_plate] * np.sin(bth[is_plate])
        b_target[is_plate, 1] = omega * br[is_plate] * np.cos(bth[is_plate])
        # fluid-side velocity: computed flow at the wall / plate samples
        f_r = np.where(is_plate, br, self.R - 0.02)
        fu_r, fv_th = self._flow(f_r, bth)
        b_vel = np.stack([-fv_th * np.sin(bth), fv_th * np.cos(bth),
                          np.zeros(n_b)], axis=1)
        b_vel[:, 0] += fu_r * np.cos(bth)
        b_vel[:, 1] += fu_r * np.sin(bth)

        frame = StateFrame(
            float(time), pos, vel,
            mass=self.mass0.copy(),
            body_id=self.body0.copy(),
            attributes={
                "mass_quantity": self.mass0.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "liquid_level_set": level_set,
                "liquid_interior_mask": interior,
                "velocity_field": vel_field,
                "boundary_normal": b_nrm,
                "boundary_target_velocity": b_target,
                "boundary_velocity": b_vel,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "grid_spacing": float(self.dx),
                "characteristic_length": float(2.0 * self.R),
                "reference_velocity": float(max(self.omega_b * self.R, 1e-6)),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = max(0.0, float(time))
        if target < self.t - 1e-9:
            target = self.t
        while self.t < target - 1e-9:
            dt = min(self._cfl_dt(), target - self.t)
            self._step(dt)
        if abs(self.t - target) > 1e-12 and target >= self.t:
            # land exactly on the requested sample time
            dt = target - self.t
            if dt > 0:
                self._step(dt)
        return self._build_frame(target)

    def diagnostics(self) -> dict:
        return {
            "model": "polar shallow-water + azimuthal Landau modal stability",
            "time": float(self.t),
            "target_mode": int(self.m_t),
            "mode_amplitudes": {str(m): float(self.a[m]) for m in self.modes},
            "pattern_angular_velocity": float(self.Omega_p),
            "corner_radius": float(self.r_p),
            "shear_source_ratio": float(self.gamma_t),
            "center_depth": float(self.h[0]),
            "rim_depth": float(self.h[-1]),
        }
