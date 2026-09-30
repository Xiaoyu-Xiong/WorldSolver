"""Rotating-bottom free-surface solver (self-implemented shallow-water model).

Physical model
-------------
The liquid in the stationary cylinder is modelled with the depth-averaged
(shallow-water) equations for depth ``h(x, y, t)`` and depth-averaged
horizontal momentum ``(hu, hv)``:

    d_t h  + d_x (h u)          + d_y (h v)          = 0
    d_t hu + d_x (h u^2 + g h^2/2) + d_y (h u v)     = S_x
    d_t hv + d_x (h u v)        + d_y (h v^2 + g h^2/2) = S_y

Sources ``S`` contain

* the driven bottom plate: a quadratic no-slip bottom stress relaxing the
  depth-averaged velocity toward the plate velocity ``Omega(t) e_z x r``,
  scaled by ``1 / h``.  The ``1/h`` factor is the physical mechanism that
  makes the azimuthal free-surface pattern unstable: where the surface is
  locally depressed the plate couples more strongly, the fluid spins faster,
  the centrifugal push deepens the depression further.  This is what turns
  the axisymmetric centrifugal dip into a rotating polygon.
* the stationary side wall: Brinkman/volume penalization of the wall-normal
  velocity (no through-flow) plus a quadratic tangential wall stress
  (no-slip drag) inside a thin, strictly axisymmetric wall layer, so the
  wall introduces no grid-aligned azimuthal bias.
* horizontal eddy viscosity (Laplacian), which damps short azimuthal scales
  and keeps the low-order pattern smooth.

Numerics: conservative finite volumes on a uniform Cartesian grid, MUSCL
reconstruction with a minmod limiter, Rusanov (local Lax-Friedrichs) fluxes,
two-stage SSP Runge-Kutta time integration with a CFL-limited step, reflective
ghost states on the solid mask, semi-implicit (unconditionally stable)
treatment of the friction/penalization sources and exact global volume
renormalization.  Everything is deterministic: no random numbers are used.

Only NumPy/SciPy are used; no external physics engine.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is optional; a NumPy fallback interpolator is provided.
    from scipy.ndimage import map_coordinates as _map_coordinates
except Exception:  # pragma: no cover
    _map_coordinates = None

from worldsolver.contracts import StateFrame


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod slope limiter."""
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        tank = dict(cfg["tank"])
        self.cfg = cfg
        self.tank = tank
        self.seed = int(seed)

        self.g = float(abs(cfg["gravity"][2])) or 9.81
        self.R = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.H0 = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.rho = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.plate_radius = 0.94
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        # --- solver grid (cell centred, uniform, covers the whole tank) -----
        self.n = 112
        self.half = 1.0
        self.dx = 2.0 * self.half / self.n
        xc = -self.half + (np.arange(self.n) + 0.5) * self.dx
        self.xc = xc
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        self.X, self.Y = X, Y
        rr = np.sqrt(X * X + Y * Y)
        self.rr = rr
        eps = 1e-12
        self.cos_t = X / np.maximum(rr, eps)
        self.sin_t = Y / np.maximum(rr, eps)

        # solid mask sits slightly outside the physical wall; the penalization
        # layer below stops the flow before it can feel this staircase.
        self.solid = rr > (self.R + 0.02)
        self.wet = ~self.solid
        self.plate_mask = (rr <= self.plate_radius) & self.wet

        # axisymmetric wall layer
        r_in, r_out = self.R - 0.075, self.R
        self.chi = np.clip((rr - r_in) / (r_out - r_in), 0.0, 1.0)
        self.chi = np.where(self.solid, 1.0, self.chi)

        # --- numerical / physical constants --------------------------------
        self.cfl = 0.35
        self.dt_max = 2.0e-3
        self.dt_min = 5.0e-5
        self.h_floor = 0.02          # depth floor for velocity reconstruction
        self.h_drag = 0.06           # depth floor inside friction denominators
        self.u_max = 15.0
        self.c_bottom = 0.05         # plate friction coefficient
        self.c_wall = 0.022          # side-wall tangential friction
        self.wall_bl = 0.07          # wall boundary-layer thickness [m]
        self.tau_normal = 3.0e-3     # wall-normal penalization time [s]
        self.nu = 8.0e-4             # horizontal eddy viscosity [m^2/s]
        self.omega_ramp = 1.2        # plate spin-up time [s]

        # --- initial condition (matches the frozen scene surface) ----------
        theta = np.arctan2(Y, X)
        ratio = np.clip(rr / self.R, 0.0, 1.0)
        depression = 0.16 * (1.0 - ratio * ratio) ** 2
        seeded = 0.035 * 4.0 * ratio * (1.0 - ratio) * np.cos(self.mode * theta)
        eta0 = self.H0 - depression - seeded
        self.h = np.where(self.wet, np.maximum(eta0, 1.0e-3), 0.0)
        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)
        self.volume0 = float(self.h.sum() * self.dx * self.dx)
        self.t = 0.0
        self.steps = 0

        # --- mesh / particle bookkeeping from the frozen scene -------------
        init = scene.initial_state()
        pos0 = np.asarray(init.position, dtype=float)
        self.n_pts = int(pos0.shape[0])
        self.vx = pos0[:, 0].copy()
        self.vy = pos0[:, 1].copy()
        init_mass = init.mass
        if init_mass is None:
            init_mass = np.full(self.n_pts, self.rho * self.spacing ** 3)
        self.pmass = np.asarray(init_mass, dtype=float).copy()
        prad = None
        if init.attributes is not None:
            prad = init.attributes.get("particle_radius")
        if prad is None:
            prad = np.full(self.n_pts, 0.48 * self.spacing)
        self.prad = np.asarray(prad, dtype=float).copy()
        self.body_id = np.zeros(self.n_pts, dtype=np.int32)
        self.total_mass = float(self.pmass.sum())

        # --- diagnostic grid ------------------------------------------------
        nx, ny, nz = self.grid_shape
        self.gx = np.linspace(-self.R, self.R, nx)
        self.gy = np.linspace(-self.R, self.R, ny)
        self.gz = np.linspace(0.0, self.wall_height, nz)
        GX, GY = np.meshgrid(self.gx, self.gy, indexing="ij")
        self.GX, self.GY = GX, GY
        self.g_rr = np.sqrt(GX * GX + GY * GY)
        self.GZ = self.gz[None, None, :]

        # --- boundary samples ----------------------------------------------
        n_ang = 64
        ang = 2.0 * np.pi * np.arange(n_ang) / n_ang
        ca, sa = np.cos(ang), np.sin(ang)
        wall_z = [0.25 * self.H0, 0.75 * self.H0]
        wpts, wnrm = [], []
        for zc in wall_z:
            wpts.append(np.stack([self.R * ca, self.R * sa, np.full(n_ang, zc)], 1))
            wnrm.append(np.stack([-ca, -sa, np.zeros(n_ang)], 1))
        n_ang2 = 48
        ang2 = 2.0 * np.pi * np.arange(n_ang2) / n_ang2
        ca2, sa2 = np.cos(ang2), np.sin(ang2)
        ppts, pnrm = [], []
        for frac in (0.45, 0.85):
            rp = frac * self.plate_radius
            ppts.append(np.stack([rp * ca2, rp * sa2, np.zeros(n_ang2)], 1))
            pnrm.append(np.stack([np.zeros(n_ang2), np.zeros(n_ang2),
                                  np.ones(n_ang2)], 1))
        self.b_wall = np.concatenate(wpts, 0)
        self.b_plate = np.concatenate(ppts, 0)
        self.bnd_pts = np.concatenate([self.b_wall, self.b_plate], 0)
        self.bnd_nrm = np.concatenate(wnrm + pnrm, 0)
        self.n_wall = self.b_wall.shape[0]
        self.n_bnd = self.bnd_pts.shape[0]

        self._prev_time = None
        self._prev_z = None
        self._max_speed = 0.0

    # ------------------------------------------------------------ kinematics
    def _omega(self, t: float) -> float:
        s = min(max(t / self.omega_ramp, 0.0), 1.0)
        return self.omega_target * (s * s * (3.0 - 2.0 * s))

    # --------------------------------------------------------------- physics
    def _primitive(self, h, hu, hv):
        hs = np.maximum(h, self.h_floor)
        u = np.clip(hu / hs, -self.u_max, self.u_max)
        v = np.clip(hv / hs, -self.u_max, self.u_max)
        dry = h < 1.0e-4
        u = np.where(dry, 0.0, u)
        v = np.where(dry, 0.0, v)
        return u, v

    def _face_states(self, W, axis):
        """MUSCL reconstructed left/right face states with reflective ghosts."""
        d = np.diff(W, axis=axis + 1)
        pad = list(W.shape)
        pad[axis + 1] = 1
        z = np.zeros(pad)
        back = np.concatenate([z, d], axis=axis + 1)
        fwd = np.concatenate([d, z], axis=axis + 1)
        s = _minmod(back, fwd)
        if axis == 0:
            WL = W[:, :-1, :] + 0.5 * s[:, :-1, :]
            WR = W[:, 1:, :] - 0.5 * s[:, 1:, :]
            cL = W[:, :-1, :]
            cR = W[:, 1:, :]
            sL = self.solid[:-1, :]
            sR = self.solid[1:, :]
            mom = 1
        else:
            WL = W[:, :, :-1] + 0.5 * s[:, :, :-1]
            WR = W[:, :, 1:] - 0.5 * s[:, :, 1:]
            cL = W[:, :, :-1]
            cR = W[:, :, 1:]
            sL = self.solid[:, :-1]
            sR = self.solid[:, 1:]
            mom = 2

        WL = WL.copy()
        WR = WR.copy()
        # wall on the right: mirror the left cell
        m = sR & (~sL)
        if m.any():
            for k in range(3):
                WR[k] = np.where(m, cL[k] if k != mom else -cL[k], WR[k])
        # wall on the left: mirror the right cell
        m = sL & (~sR)
        if m.any():
            for k in range(3):
                WL[k] = np.where(m, cR[k] if k != mom else -cR[k], WL[k])
        both = sL & sR
        return WL, WR, both

    def _rusanov(self, WL, WR, axis):
        hL = np.maximum(WL[0], 0.0)
        hR = np.maximum(WR[0], 0.0)
        dryL = hL < 1.0e-4
        dryR = hR < 1.0e-4
        huL = np.where(dryL, 0.0, WL[1])
        hvL = np.where(dryL, 0.0, WL[2])
        huR = np.where(dryR, 0.0, WR[1])
        hvR = np.where(dryR, 0.0, WR[2])
        hsL = np.maximum(hL, self.h_floor)
        hsR = np.maximum(hR, self.h_floor)
        uL = np.clip(huL / hsL, -self.u_max, self.u_max)
        vL = np.clip(hvL / hsL, -self.u_max, self.u_max)
        uR = np.clip(huR / hsR, -self.u_max, self.u_max)
        vR = np.clip(hvR / hsR, -self.u_max, self.u_max)
        huL = hL * uL
        hvL = hL * vL
        huR = hR * uR
        hvR = hR * vR
        p_l = 0.5 * self.g * hL * hL
        p_r = 0.5 * self.g * hR * hR
        if axis == 0:
            FL = (huL, huL * uL + p_l, huL * vL)
            FR = (huR, huR * uR + p_r, huR * vR)
            aL = np.abs(uL) + np.sqrt(self.g * hL)
            aR = np.abs(uR) + np.sqrt(self.g * hR)
        else:
            FL = (hvL, hvL * uL, hvL * vL + p_l)
            FR = (hvR, hvR * uR, hvR * vR + p_r)
            aL = np.abs(vL) + np.sqrt(self.g * hL)
            aR = np.abs(vR) + np.sqrt(self.g * hR)
        a = np.maximum(aL, aR)
        UL = (hL, huL, hvL)
        UR = (hR, huR, hvR)
        out = [0.5 * (FL[k] + FR[k]) - 0.5 * a * (UR[k] - UL[k]) for k in range(3)]
        return np.stack(out, 0)

    def _rhs(self, W):
        n = self.n
        # x direction
        WL, WR, both = self._face_states(W, 0)
        Fx = self._rusanov(WL, WR, 0)
        Fx = np.where(both[None, :, :], 0.0, Fx)
        Fxf = np.zeros((3, n + 1, n))
        Fxf[:, 1:-1, :] = Fx
        dW = -(Fxf[:, 1:, :] - Fxf[:, :-1, :]) / self.dx
        # y direction
        WL, WR, both = self._face_states(W, 1)
        Fy = self._rusanov(WL, WR, 1)
        Fy = np.where(both[None, :, :], 0.0, Fy)
        Fyf = np.zeros((3, n, n + 1))
        Fyf[:, :, 1:-1] = Fy
        dW += -(Fyf[:, :, 1:] - Fyf[:, :, :-1]) / self.dx
        dW = np.where(self.solid[None, :, :], 0.0, dW)
        return dW

    def _apply_sources(self, dt: float) -> None:
        h, hu, hv = self.h, self.hu, self.hv
        u, v = self._primitive(h, hu, hv)
        om = self._omega(self.t)

        # rotating bottom plate: quadratic no-slip stress ~ 1/h
        ub = -om * self.Y
        vb = om * self.X
        du = u - ub
        dv = v - vb
        sp = np.sqrt(du * du + dv * dv)
        k = self.c_bottom * sp / np.maximum(h, self.h_drag)
        fac = 1.0 / (1.0 + dt * np.where(self.plate_mask, k, 0.0))
        u = np.where(self.plate_mask, ub + du * fac, u)
        v = np.where(self.plate_mask, vb + dv * fac, v)

        # stationary side wall: normal penalization + tangential friction
        ur = u * self.cos_t + v * self.sin_t
        ut = -u * self.sin_t + v * self.cos_t
        ur = ur / (1.0 + dt * self.chi / self.tau_normal)
        kt = self.chi * self.c_wall * np.abs(ut) / self.wall_bl
        ut = ut / (1.0 + dt * kt)
        u = ur * self.cos_t - ut * self.sin_t
        v = ur * self.sin_t + ut * self.cos_t

        # horizontal eddy viscosity
        u = np.where(self.solid, 0.0, u)
        v = np.where(self.solid, 0.0, v)
        lap_u = (np.roll(u, 1, 0) + np.roll(u, -1, 0)
                 + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u) / (self.dx ** 2)
        lap_v = (np.roll(v, 1, 0) + np.roll(v, -1, 0)
                 + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4.0 * v) / (self.dx ** 2)
        u = u + dt * self.nu * lap_u
        v = v + dt * self.nu * lap_v
        u = np.clip(np.where(self.solid, 0.0, u), -self.u_max, self.u_max)
        v = np.clip(np.where(self.solid, 0.0, v), -self.u_max, self.u_max)

        self.hu = h * u
        self.hv = h * v
        self._max_speed = float(np.max(np.sqrt(u * u + v * v)))

    def _sanitize(self) -> None:
        h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)
        h = np.where(self.solid, 0.0, np.maximum(h, 0.0))
        tot = float(h.sum()) * self.dx * self.dx
        if tot > 1.0e-9:
            h = h * (self.volume0 / tot)
        else:  # pragma: no cover - defensive
            h = np.where(self.wet, self.volume0 / (self.wet.sum() * self.dx ** 2), 0.0)
        self.h = h
        u, v = self._primitive(h, np.nan_to_num(self.hu), np.nan_to_num(self.hv))
        self.hu = np.where(self.solid, 0.0, h * u)
        self.hv = np.where(self.solid, 0.0, h * v)

    def _cfl_dt(self) -> float:
        u, v = self._primitive(self.h, self.hu, self.hv)
        c = np.sqrt(self.g * np.maximum(self.h, 0.0))
        speed = np.maximum(np.abs(u), np.abs(v)) + c
        smax = float(np.max(np.where(self.wet, speed, 0.0)))
        smax = max(smax, 1.0e-3, self._omega(self.t) * self.R * 0.25)
        dt = self.cfl * self.dx / smax
        return float(min(max(dt, self.dt_min), self.dt_max))

    def _step(self, dt: float) -> None:
        W = np.stack([self.h, self.hu, self.hv], 0)
        W1 = W + dt * self._rhs(W)
        W1[0] = np.maximum(W1[0], 0.0)
        W2 = 0.5 * (W + W1 + dt * self._rhs(W1))
        self.h, self.hu, self.hv = W2[0], W2[1], W2[2]
        self._sanitize()
        self._apply_sources(dt)
        self._sanitize()
        self.t += dt
        self.steps += 1

    # --------------------------------------------------------- interpolation
    @staticmethod
    def _extend(field: np.ndarray, valid: np.ndarray, iters: int = 4) -> np.ndarray:
        f = field.astype(float).copy()
        vd = valid.copy()
        for _ in range(iters):
            s = np.zeros_like(f)
            c = np.zeros_like(f)
            src = np.where(vd, f, 0.0)
            vf = vd.astype(float)
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                s += np.roll(src, sh, axis=ax)
                c += np.roll(vf, sh, axis=ax)
            newv = (~vd) & (c > 0.0)
            f = np.where(newv, s / np.maximum(c, 1.0), f)
            vd = vd | newv
        return f

    def _sample(self, field: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
        gi = (np.asarray(xq, dtype=float) + self.half) / self.dx - 0.5
        gj = (np.asarray(yq, dtype=float) + self.half) / self.dx - 0.5
        gi = np.clip(gi, 0.0, self.n - 1.0)
        gj = np.clip(gj, 0.0, self.n - 1.0)
        if _map_coordinates is not None:
            out = _map_coordinates(field, np.stack([gi.ravel(), gj.ravel()], 0),
                                   order=1, mode="nearest")
            return np.asarray(out, dtype=float).reshape(np.shape(gi))
        i0 = np.floor(gi).astype(int)
        j0 = np.floor(gj).astype(int)
        i1 = np.minimum(i0 + 1, self.n - 1)
        j1 = np.minimum(j0 + 1, self.n - 1)
        fx = gi - i0
        fy = gj - j0
        return ((1 - fx) * (1 - fy) * field[i0, j0]
                + fx * (1 - fy) * field[i1, j0]
                + (1 - fx) * fy * field[i0, j1]
                + fx * fy * field[i1, j1])

    # ------------------------------------------------------------------ frame
    def advance_to(self, time: float):
        t_target = float(time)
        guard = 0
        while self.t < t_target - 1.0e-12 and guard < 200000:
            dt = min(self._cfl_dt(), t_target - self.t)
            if dt <= 0.0:
                break
            self._step(dt)
            guard += 1
        self.t = t_target
        return self._make_frame(t_target)

    def _make_frame(self, t: float) -> StateFrame:
        u, v = self._primitive(self.h, self.hu, self.hv)
        h_ext = self._extend(self.h, self.wet, 5)
        u_ext = self._extend(u, self.wet, 5)
        v_ext = self._extend(v, self.wet, 5)

        # --- free-surface mesh vertices ------------------------------------
        z = self._sample(h_ext, self.vx, self.vy)
        z = np.clip(np.nan_to_num(z, nan=self.H0), 0.01, self.wall_height)
        pu = np.clip(np.nan_to_num(self._sample(u_ext, self.vx, self.vy)),
                     -self.u_max, self.u_max)
        pv = np.clip(np.nan_to_num(self._sample(v_ext, self.vx, self.vy)),
                     -self.u_max, self.u_max)
        if self._prev_z is not None and self._prev_time is not None \
                and t > self._prev_time + 1.0e-9:
            pw = np.clip((z - self._prev_z) / (t - self._prev_time), -5.0, 5.0)
        else:
            pw = np.zeros_like(z)
        self._prev_z = z.copy()
        self._prev_time = t

        position = np.stack([self.vx, self.vy, z], 1)
        velocity = np.stack([pu, pv, pw], 1)

        # --- diagnostic grids ----------------------------------------------
        nx, ny, nz = self.grid_shape
        eta_g = np.clip(self._sample(h_ext, self.GX, self.GY), 0.0, self.wall_height)
        ug = self._sample(u_ext, self.GX, self.GY)
        vg = self._sample(v_ext, self.GX, self.GY)
        outside = self.g_rr > self.R
        eta_g = np.where(outside, 0.0, eta_g)
        ug = np.where(outside, 0.0, ug)
        vg = np.where(outside, 0.0, vg)

        radial = (self.g_rr - self.R)[:, :, None] * np.ones((1, 1, nz))
        level_set = np.maximum(radial, self.GZ - eta_g[:, :, None])
        level_set = np.nan_to_num(level_set, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        interior = level_set < -1.5 * self.spacing

        below = (self.GZ <= eta_g[:, :, None]) & (~outside[:, :, None])
        dudx = np.gradient(ug, self.gx, axis=0)
        dvdy = np.gradient(vg, self.gy, axis=1)
        div = np.nan_to_num(dudx + dvdy)
        wg = -np.clip(div, -20.0, 20.0)[:, :, None] * self.GZ
        vel_field = np.zeros((nx, ny, nz, 3))
        vel_field[..., 0] = np.where(below, ug[:, :, None], 0.0)
        vel_field[..., 1] = np.where(below, vg[:, :, None], 0.0)
        vel_field[..., 2] = np.where(below, np.clip(wg, -5.0, 5.0), 0.0)
        vel_field = np.nan_to_num(vel_field)

        # --- boundary samples ----------------------------------------------
        om = self._omega(t)
        b_vel = np.zeros((self.n_bnd, 3))
        b_tgt = np.zeros((self.n_bnd, 3))
        # side wall: stationary, no-slip -> target zero; reported velocity is
        # the residual wall-normal slip left by the penalization.
        wx = self.b_wall[:, 0]
        wy = self.b_wall[:, 1]
        rw = np.maximum(np.hypot(wx, wy), 1e-12)
        cw, sw = wx / rw, wy / rw
        uw = self._sample(u_ext, wx * 0.995, wy * 0.995)
        vw = self._sample(v_ext, wx * 0.995, wy * 0.995)
        urad = np.clip(np.nan_to_num(uw * cw + vw * sw), -self.u_max, self.u_max)
        b_vel[:self.n_wall, 0] = urad * cw
        b_vel[:self.n_wall, 1] = urad * sw
        # rotating plate: no-slip with the driven plate
        px = self.b_plate[:, 0]
        py = self.b_plate[:, 1]
        b_tgt[self.n_wall:, 0] = -om * py
        b_tgt[self.n_wall:, 1] = om * px
        b_vel[self.n_wall:, 0] = -om * py
        b_vel[self.n_wall:, 1] = om * px
        b_vel = np.nan_to_num(b_vel)

        frame = StateFrame(
            time=float(t),
            position=np.ascontiguousarray(np.nan_to_num(position)),
            velocity=np.ascontiguousarray(np.nan_to_num(velocity)),
            mass=self.pmass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.pmass.copy(),
                "particle_radius": self.prad.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "liquid_level_set": level_set,
                "liquid_interior_mask": interior,
                "velocity_field": vel_field,
                "boundary_normal": np.ascontiguousarray(self.bnd_nrm.astype(float)),
                "boundary_target_velocity": np.ascontiguousarray(b_tgt),
                "boundary_velocity": np.ascontiguousarray(b_vel),
            },
            scalars={
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(2.0 * self.R),
                "reference_velocity": float(max(self.omega_target * self.R, 1e-6)),
            },
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        u, v = self._primitive(self.h, self.hu, self.hv)
        wet = self.wet
        h = self.h
        # mean angular velocity of the liquid (mass weighted)
        rr = np.maximum(self.rr, 1e-9)
        ut = -u * self.sin_t + v * self.cos_t
        w = np.where(wet & (h > 0.02), h, 0.0)
        denom = float(w.sum()) or 1.0
        omega_mean = float((w * ut / rr).sum() / denom)
        return {
            "method": "shallow_water_MUSCL_Rusanov_SSPRK2",
            "grid": [int(self.n), int(self.n)],
            "steps": int(self.steps),
            "time": float(self.t),
            "plate_omega": float(self._omega(self.t)),
            "liquid_omega_mean": omega_mean,
            "min_depth": float(np.min(np.where(wet, h, np.inf))),
            "max_depth": float(np.max(h)),
            "volume": float(h.sum() * self.dx * self.dx),
            "total_mass": float(self.total_mass),
            "max_speed": float(self._max_speed),
        }
