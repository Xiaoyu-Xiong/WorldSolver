"""Two-way coupled shallow-water / rigid-hull solver for 106_frigate_wave_response.

Physical model
--------------
* Free-surface water: depth-averaged shallow-water equations (SWE) on a uniform
  Eulerian grid, advanced with an explicit conservative finite-volume scheme
  (upwind advection, centered pressure gradient, seabed/side-wall boundaries,
  Dirichlet wave inlet at x_min, sponge outflow at x_max).
* Rigid frigate hull: 2-DOF rigid body (heave z, pitch about y -- the only
  degrees of freedom left free by the frozen config).  Buoyancy and the
  Froude-Krylov wave moment come from integrating the *resolved* water surface
  over the hull stations; the hull exerts form drag on the water column, which
  deflects the flow around the hull.  This is the two-way coupling.
* Water markers (Lagrangian samples): advected every substep by the resolved
  SWE velocity field plus a depth-linear vertical velocity w = (z/eta)*deta/dt;
  the impermeable hull projection pushes markers out of the solid.
All budgets (external impulses, boundary momentum fluxes, interaction forces)
are accumulated from the actual forces and fluxes of each substep.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Implements the Solver contract (initialize / advance_to / diagnostics)."""

    def __init__(self) -> None:
        self.t = 0.0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg

        # ---- frozen physical parameters -----------------------------------
        g = 9.81
        self.g = g
        self.rho = float(cfg["fluid"]["density"])
        self.wl = float(cfg["fluid"]["still_water_level"])
        wav = cfg["incoming_waves"]
        self.amp = float(wav["amplitude"])
        self.lam = float(wav["wavelength"])
        self.period = float(wav["period"])
        self.k_wave = 2.0 * np.pi / self.lam
        self.omega = 2.0 * np.pi / self.period
        self.u0 = float(cfg["fluid"]["initial_velocity"][0])
        ship = cfg["frigate"]
        self.ship_body = int(ship["body_id"])
        self.M = float(ship["mass"])
        self.Iy = 40000.0          # pitch inertia of the hull (kg m^2)
        self.cz = 25000.0          # heave radiation damping (N s/m)
        self.cth = 150000.0        # pitch radiation damping (N m s)
        self.cd = 1.2              # hull form-drag coefficient
        self.dt_sub = 0.005

        # ---- frozen hull geometry (analytic hull used by the scaffold) ----
        self.L = 4.35              # half length
        self.beam_c = 0.92         # half-beam coefficient
        self.keel0 = -1.12         # keel height at midship (local)
        self.keel_rise = 0.16      # keel rise toward the ends
        self.deck = 0.42           # deck height (local)

        # ---- initial state from the frozen scene --------------------------
        init = scene.initial_state()
        pos = np.array(init.position, dtype=np.float64)
        self.mass = np.array(init.mass, dtype=np.float64)
        body_id = np.array(init.body_id, dtype=np.int64)
        self.body_id = body_id
        self.n = len(pos)
        ship_mask = body_id == self.ship_body
        self.ship_mask = ship_mask
        fluid_mask = ~ship_mask
        self.fluid_mask = fluid_mask
        self.xf = pos[fluid_mask].copy()
        n_f = self.xf.shape[0]
        self.Mw = float(self.mass[fluid_mask].sum())
        self.vf = np.tile(
            np.asarray(cfg["fluid"]["initial_velocity"], dtype=np.float64), (n_f, 1)
        )
        self.phase = np.where(ship_mask, 0.0, 1.0)
        self.ship_visual = np.where(ship_mask, 1.0, 0.0)
        self.mass_quantity = self.mass.copy()

        # ---- SWE grid ------------------------------------------------------
        self.x0 = float(cfg["fluid"]["domain_min"][0])
        self.x1 = float(cfg["fluid"]["domain_max"][0])
        self.y0 = float(cfg["fluid"]["domain_min"][1])
        self.y1 = float(cfg["fluid"]["domain_max"][1])
        self.dx = 0.30
        self.dy = 0.30
        self.nx = int(round((self.x1 - self.x0) / self.dx))
        self.ny = int(round((self.y1 - self.y0) / self.dy))
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.xc2, self.yc2 = np.meshgrid(self.xc, self.yc, indexing="ij")
        eta0 = self.wl + self.amp * np.sin(self.k_wave * self.xc)
        self.h = np.repeat(eta0[:, None], self.ny, axis=1).copy()
        self.u = np.full_like(self.h, self.u0)
        self.v = np.zeros_like(self.h)
        self.hmin = 0.05
        self.nu = 0.02             # eddy viscosity (m^2/s)
        self.cf = 0.02             # seabed friction (1/s)

        # ---- hull rigid state (only heave/pitch are free) ------------------
        self.zc = float(ship["initial_center"][2])
        self.zcd = 0.0
        self.th = 0.0
        self.thd = 0.0

        # ---- hull station quadrature ---------------------------------------
        dxq = 0.10
        self.xs = np.arange(-self.L + 0.05, self.L, dxq)
        self.dxq = dxq
        self.ws = self.beam_c * np.clip(1.0 - np.abs(self.xs) / self.L, 0.0, 1.0) ** 0.35
        self.ks = self.keel0 + self.keel_rise * np.abs(self.xs) / self.L

        # ---- boundary sample points ----------------------------------------
        pts, nrm, kind = [], [], []
        for yy in np.linspace(-5.5, 5.5, 13):
            pts.append((self.x0 + 0.05, yy, 0.8))
            nrm.append((-1.0, 0.0, 0.0))
            kind.append(0)  # wave inlet
            pts.append((self.x1 - 0.05, yy, 0.8))
            nrm.append((1.0, 0.0, 0.0))
            kind.append(1)  # open outlet
        for xx in np.linspace(-14.0, 14.0, 15):
            pts.append((xx, self.y1 - 0.05, 0.8))
            nrm.append((0.0, 1.0, 0.0))
            kind.append(2)  # port wall
            pts.append((xx, self.y0 + 0.05, 0.8))
            nrm.append((0.0, -1.0, 0.0))
            kind.append(2)  # starboard wall
        for xx in np.linspace(-14.0, 14.0, 9):
            for yy in np.linspace(-5.0, 5.0, 5):
                pts.append((xx, yy, 0.05))
                nrm.append((0.0, 0.0, -1.0))
                kind.append(3)  # seabed
        self.bpts = np.asarray(pts, dtype=np.float64)
        self.bnrm = np.asarray(nrm, dtype=np.float64)
        self.bkind = np.asarray(kind, dtype=np.int64)
        self.bx = self.bpts[:, 0].copy()
        self.by = self.bpts[:, 1].copy()
        self.bk_in = self.bkind == 0
        self.bk_out = self.bkind == 1
        self.bk_wall = self.bkind == 2

        # ---- budget accumulators -------------------------------------------
        self.cum_impulse = np.zeros((3, 3))     # per-domain external impulse
        self.cum_in = np.zeros(3)               # inflow momentum (water/global)
        self.cum_out = np.zeros(3)              # outflow momentum
        self.last_u_in = self.u0
        self.last_eta_in = self.wl

        # initial force evaluation (frame at t=0)
        f_hull, f_bnd, _f_water = self._forces()
        self.F_hull = f_hull          # water-on-hull interaction force
        self.F_bnd = f_bnd            # boundary traction on the water
        self._init_budget_weights()

        self.reference_velocity = float(self.lam / self.period)
        self.reference_momentum = float(self.Mw * self.reference_velocity)
        self.t = 0.0
        self._frames = 0

    # ---------------------------------------------------------- budget setup
    def _init_budget_weights(self) -> None:
        w = np.zeros((3, self.n), dtype=np.float32)
        w[0, self.fluid_mask] = 1.0     # water domain
        w[1, self.ship_mask] = 1.0      # ship domain
        w[2, :] = 1.0                   # global domain
        self.budget_weights = w

    # ------------------------------------------------------------- utilities
    def _sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample of a cell-centered field at (x, y)."""
        gx = (x - self.x0) / self.dx - 0.5
        gy = (y - self.y0) / self.dy - 0.5
        i = np.clip(np.floor(gx).astype(np.int64), 0, self.nx - 2)
        j = np.clip(np.floor(gy).astype(np.int64), 0, self.ny - 2)
        tx = np.clip(gx - i, 0.0, 1.0)
        ty = np.clip(gy - j, 0.0, 1.0)
        return (
            field[i, j] * (1.0 - tx) * (1.0 - ty)
            + field[i + 1, j] * tx * (1.0 - ty)
            + field[i, j + 1] * (1.0 - tx) * ty
            + field[i + 1, j + 1] * tx * ty
        )

    def _inlet_state(self, t: float):
        """Prescribed incoming wave at the x_min boundary."""
        eta_in = self.wl + self.amp * np.sin(-self.k_wave * 15.0 - self.omega * t)
        u_in = self.u0 + (eta_in - self.wl) * np.sqrt(self.g / self.wl)
        return eta_in, u_in

    # ------------------------------------------------------ coupled forces
    def _forces(self):
        """Buoyancy/wave moment on the hull, hull-on-water drag, boundary traction.

        Returns (F_hull_water_on_hull [3], F_bnd_traction [3], F_drag [3]).
        """
        c = np.cos(self.th)
        s = np.sin(self.th)
        h = self.h
        u = self.u
        v = self.v
        rho = self.rho
        g = self.g

        # ---- buoyancy / Froude-Krylov moment from hull stations ------------
        xw = c * self.xs + s * self.ks
        xw_c = np.clip(xw, self.x0 + 0.05, self.x1 - 0.05)
        eta_s = self._sample(h, xw_c, np.zeros_like(xw_c))
        zl_surf = (eta_s - self.zc + s * self.xs) / max(c, 0.5)
        d = np.clip(zl_surf - self.ks, 0.0, self.deck - self.ks)
        dv = 2.0 * self.ws * d * self.dxq
        vol = float(dv.sum())
        fb = rho * g * vol
        zmid = self.ks + 0.5 * d
        xc_w = c * self.xs + s * zmid
        tau = -rho * g * float((dv * xc_w).sum())

        # ---- hull form drag on the water column (cell based) ---------------
        xl = c * self.xc2 - s * (self.wl - self.zc)
        in_len = np.abs(xl) < self.L
        wloc = np.where(in_len, self.beam_c * np.clip(1.0 - np.abs(xl) / self.L, 0.0, 1.0) ** 0.35, 0.0)
        wy = np.clip(self.yc2 + 0.5 * self.dy, -wloc, wloc) - np.clip(
            self.yc2 - 0.5 * self.dy, -wloc, wloc
        )
        kloc = self.keel0 + self.keel_rise * np.abs(xl) / self.L
        z_keel = self.zc - s * xl + c * kloc
        d_imm = np.clip(h - z_keel, 0.0, np.maximum(self.deck - kloc, 0.0))
        a_proj = np.where(wy > 0.0, d_imm * wy, 0.0)
        speed = np.sqrt(u * u + v * v)
        fdrag_x = 0.5 * rho * self.cd * float((a_proj * speed * u).sum())
        fdrag_y = 0.5 * rho * self.cd * float((a_proj * speed * v).sum())

        # ---- boundary traction on the water --------------------------------
        h_in = float(h[0, :].mean())
        h_out = float(h[-1, :].mean())
        width = self.y1 - self.y0
        fx_p = 0.5 * rho * g * (h_in * h_in - h_out * h_out) * width
        fz_p = rho * g * float(h.sum()) * self.dx * self.dy  # seabed support

        f_hull = np.array(
            [fdrag_x, fdrag_y, fb - self.cz * self.zcd], dtype=np.float64
        )
        f_bnd = np.array([fx_p, 0.0, fz_p], dtype=np.float64)
        self._tau_b = tau
        return f_hull, f_bnd, np.array([fdrag_x, fdrag_y, 0.0])

    # ------------------------------------------------------------- substep
    def _substep(self, dt: float) -> None:
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        g = self.g
        h_prev = self.h.copy()

        # ---- boundary forcing at mid-time ----------------------------------
        eta_in, u_in = self._inlet_state(self.t + 0.5 * dt)
        self.last_eta_in, self.last_u_in = eta_in, u_in

        # ---- face velocities ------------------------------------------------
        uf = np.empty((nx + 1, ny))
        uf[1:nx] = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        uf[0] = u_in
        uf[nx] = self.u[-1, :]
        vf = np.zeros((nx, ny + 1))
        vf[:, 1:ny] = 0.5 * (self.v[:, :-1] + self.v[:, 1:])

        # ---- upwind volume fluxes -------------------------------------------
        hf = np.empty((nx + 1, ny))
        hf[1:nx] = np.where(uf[1:nx] > 0.0, h_prev[:-1, :], h_prev[1:, :])
        hf[0] = eta_in
        hf[nx] = h_prev[-1, :]
        vfw = np.zeros((nx, ny + 1))
        vfw[:, 1:ny] = np.where(vf[:, 1:ny] > 0.0, h_prev[:, :-1], h_prev[:, 1:])
        vfw[:, 0] = h_prev[:, 0]
        vfw[:, ny] = h_prev[:, -1]

        # ---- continuity ------------------------------------------------------
        h_new = self.h - dt * ((hf[1:, :] - hf[:-1, :]) / dx + (vfw[:, 1:] - vfw[:, :-1]) / dy)

        # ---- momentum advection (donor cell) --------------------------------
        def advect(s):
            sx = np.empty((nx + 1, ny))
            sx[1:nx] = uf[1:nx] * np.where(uf[1:nx] > 0.0, s[:-1, :], s[1:, :])
            sx[0] = uf[0] * s[0, :]
            sx[nx] = uf[nx] * s[-1, :]
            sy = np.zeros((nx, ny + 1))
            sy[:, 1:ny] = vf[:, 1:ny] * np.where(vf[:, 1:ny] > 0.0, s[:, :-1], s[:, 1:])
            return -((sx[1:, :] - sx[:-1, :]) / dx + (sy[:, 1:] - sy[:, :-1]) / dy)

        gx = np.gradient(h_new, dx, axis=0, edge_order=2)
        gy = np.gradient(h_new, dy, axis=1, edge_order=2)

        def lap(s):
            l = np.zeros_like(s)
            l[1:-1, :] = (s[2:, :] - 2.0 * s[1:-1, :] + s[:-2, :]) / (dx * dx)
            l[:, 1:-1] += (s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]) / (dy * dy)
            return l

        u_new = self.u + dt * (advect(self.u) - g * gx + self.nu * lap(self.u))
        v_new = self.v + dt * (advect(self.v) - g * gy + self.nu * lap(self.v))

        # ---- hull form drag (implicit) on the water -------------------------
        c = np.cos(self.th)
        s = np.sin(self.th)
        xl = c * self.xc2 - s * (self.wl - self.zc)
        in_len = np.abs(xl) < self.L
        wloc = np.where(
            in_len, self.beam_c * np.clip(1.0 - np.abs(xl) / self.L, 0.0, 1.0) ** 0.35, 0.0
        )
        wy = np.clip(self.yc2 + 0.5 * dy, -wloc, wloc) - np.clip(self.yc2 - 0.5 * dy, -wloc, wloc)
        kloc = self.keel0 + self.keel_rise * np.abs(xl) / self.L
        z_keel = self.zc - s * xl + c * kloc
        d_imm = np.clip(h_new - z_keel, 0.0, np.maximum(self.deck - kloc, 0.0))
        a_proj = np.where(wy > 0.0, d_imm * wy, 0.0)
        speed = np.sqrt(u_new * u_new + v_new * v_new)
        k_drag = (
            0.5
            * self.cd
            * a_proj
            * speed
            / (np.maximum(h_new, self.hmin) * dx * dy)
        )
        u_new = u_new / (1.0 + dt * k_drag)
        v_new = v_new / (1.0 + dt * k_drag)

        # ---- seabed friction -------------------------------------------------
        u_new = u_new / (1.0 + dt * self.cf)
        v_new = v_new / (1.0 + dt * self.cf)

        # ---- boundary conditions ---------------------------------------------
        h_new[0, :] = eta_in
        u_new[0, :] = u_in
        v_new[0, :] = 0.0
        v_new[:, 0] = 0.0
        v_new[:, -1] = 0.0
        ns = 10
        ramp = (((np.arange(ns) + 1) / ns) ** 2) * 4.0
        h_new[-ns:, :] = h_new[-ns:, :] + dt * ramp[:, None] * (self.wl - h_new[-ns:, :])
        u_new[-ns:, :] = u_new[-ns:, :] + dt * ramp[:, None] * (self.u0 - u_new[-ns:, :])
        v_new[-ns:, :] = v_new[-ns:, :] * (1.0 - np.clip(dt * ramp[:, None], 0.0, 0.95))
        h_new = np.clip(h_new, self.hmin, 4.0)

        # ---- boundary momentum fluxes (water domain) --------------------------
        q_in = self.rho * hf[0, :] * uf[0, :] * dy      # + into domain at inlet
        q_out = self.rho * hf[nx, :] * uf[nx, :] * dy   # + out of domain at outlet
        u_b = u_new[-1, :]
        in_in = q_in > 0.0     # mass entering through the inlet face
        in_out = q_out < 0.0   # mass entering through the outlet face
        pin_x = float(
            np.where(in_in, q_in * u_in, 0.0).sum()
            + np.where(in_out, -q_out * u_b, 0.0).sum()
        )
        pout_x = float(
            np.where(~in_in, -q_in * u_in, 0.0).sum()
            + np.where(~in_out, q_out * u_b, 0.0).sum()
        )
        pin = np.array([pin_x, 0.0, 0.0])
        pout = np.array([pout_x, 0.0, 0.0])
        self.cum_in += pin * dt
        self.cum_out += pout * dt

        # ---- commit fluid state ----------------------------------------------
        self.h = h_new
        self.u = u_new
        self.v = v_new

        # ---- rigid hull response ---------------------------------------------
        f_hull, f_bnd, _ = self._forces()
        fz = f_hull[2] - self.M * self.g
        tau = self._tau_b - self.cth * self.thd
        self.zcd += (fz / self.M) * dt
        self.thd += (tau / self.Iy) * dt
        self.zc += self.zcd * dt
        self.th += self.thd * dt
        if self.th > 0.4:
            self.th = 0.4
            self.thd = min(self.thd, 0.0)
        elif self.th < -0.4:
            self.th = -0.4
            self.thd = max(self.thd, 0.0)
        if self.zc > 3.4:
            self.zc = 3.4
            self.zcd = min(self.zcd, 0.0)
        elif self.zc < 0.6:
            self.zc = 0.6
            self.zcd = max(self.zcd, 0.0)

        # recompute interaction force with the updated hull pose
        f_hull, f_bnd, _ = self._forces()

        # ---- budget accumulation ----------------------------------------------
        f_gw = np.array([0.0, 0.0, -self.g * self.Mw])
        f_gs = np.array([0.0, 0.0, -self.g * self.M])
        f_water = f_gw - f_hull + f_bnd
        f_ship = f_gs + f_hull
        f_glob = f_gw + f_gs + f_bnd
        self.cum_impulse += dt * np.stack([f_water, f_ship, f_glob])
        self.F_hull = f_hull
        self.F_bnd = f_bnd

        # ---- Lagrangian water markers -----------------------------------------
        self._advance_markers(dt, (self.h - h_prev) / dt)
        self.t += dt

    # ------------------------------------------------------- marker advection
    def _advance_markers(self, dt: float, det: np.ndarray) -> None:
        x = self.xf[:, 0]
        y = self.xf[:, 1]
        z = self.xf[:, 2]
        xc = np.clip(x, self.x0 + 0.01, self.x1 - 0.01)
        yc = np.clip(y, self.y0 + 0.01, self.y1 - 0.01)

        u_p = self._sample(self.u, xc, yc)
        v_p = self._sample(self.v, xc, yc)
        eta_p = np.maximum(self._sample(self.h, xc, yc), 0.06)
        w_p = self._sample(det, xc, yc)
        frac = np.clip(z / eta_p, 0.0, 1.0)
        w_p = np.clip(frac * w_p, -3.0, 3.0)

        xn = x + u_p * dt
        yn = y + v_p * dt
        zn = z + w_p * dt

        # ---- impermeable hull projection --------------------------------------
        c = np.cos(self.th)
        s = np.sin(self.th)
        dxp = xn
        dzp = zn - self.zc
        xl = c * dxp - s * dzp
        zl = s * dxp + c * dzp
        yl = yn
        ax = np.abs(xl)
        wloc = self.beam_c * np.clip(1.0 - ax / self.L, 0.0, 1.0) ** 0.35
        kloc = self.keel0 + self.keel_rise * ax / self.L
        inside = (ax < self.L) & (np.abs(yl) < wloc) & (zl > kloc) & (zl < self.deck)
        if np.any(inside):
            pen_y = wloc - np.abs(yl)
            pen_x = self.L - ax
            pen_b = zl - kloc
            pen_t = self.deck - zl
            pen = np.minimum(np.minimum(pen_y, pen_x), np.minimum(pen_b, pen_t))
            eps = 0.01
            # side (y) escape
            new_y = np.sign(yl) * (wloc + eps)
            # bottom escape
            zl_b = kloc - eps
            x_b = c * xl + s * zl_b
            z_b = self.zc - s * xl + c * zl_b
            # top escape
            zl_t = self.deck + eps
            x_t = c * xl + s * zl_t
            z_t = self.zc - s * xl + c * zl_t
            # end escape
            x_e = np.sign(xl) * (self.L + eps)
            x_x = c * x_e + s * zl
            z_x = self.zc - s * x_e + c * zl

            m_y = (pen_y <= pen) & (pen_y <= pen_x) & (pen_y <= pen_b) & (pen_y <= pen_t)
            m_b = (~m_y) & (pen_b <= pen) & (pen_b <= pen_x) & (pen_b <= pen_t)
            m_t = (~m_y) & (~m_b) & (pen_t <= pen) & (pen_t <= pen_x)
            m_x = ~(m_y | m_b | m_t)
            xn = np.where(m_y, xn, np.where(m_b, x_b, np.where(m_t, x_t, np.where(m_x, x_x, xn))))
            yn = np.where(m_y, new_y, yn)
            zn = np.where(m_y, zn, np.where(m_b, z_b, np.where(m_t, z_t, np.where(m_x, z_x, zn))))

        # ---- keep markers inside the water column / tank ----------------------
        eta_n = np.maximum(self._sample(self.h, np.clip(xn, self.x0 + 0.01, self.x1 - 0.01),
                                        np.clip(yn, self.y0 + 0.01, self.y1 - 0.01)), 0.06)
        zn = np.clip(zn, 0.02, eta_n - 0.01)

        # side walls: free-slip reflection
        yh = self.y1 - 0.10
        yl_ = self.y0 + 0.10
        high = yn > yh
        low = yn < yl_
        yn = np.where(high, 2.0 * yh - yn, yn)
        yn = np.where(low, 2.0 * yl_ - yn, yn)
        yn = np.clip(yn, self.y0 + 0.02, self.y1 - 0.02)

        # open outlet: recycle markers through the wave inlet
        out = xn > self.x1 - 0.05
        if np.any(out):
            xn = np.where(out, xn - (self.x1 - self.x0), xn)
            eta_r = np.maximum(
                self._sample(self.h, np.clip(xn, self.x0 + 0.01, self.x1 - 0.01),
                             np.clip(yn, self.y0 + 0.01, self.y1 - 0.01)), 0.06)
            zn = np.where(out, np.clip(zn, 0.02, eta_r - 0.01), zn)
        xn = np.where(xn < self.x0 + 0.02, self.x0 + 0.02, xn)

        self.xf[:, 0] = xn
        self.xf[:, 1] = yn
        self.xf[:, 2] = zn
        self.vf = np.column_stack((u_p, v_p, w_p))

    # ------------------------------------------------------------- frame build
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        nf = self.xf.shape[0]
        pos = np.empty((n, 3), dtype=np.float32)
        vel = np.empty((n, 3), dtype=np.float32)
        ori = np.zeros((n, 4), dtype=np.float32)
        ori[:, 3] = 1.0
        ang = np.zeros((n, 3), dtype=np.float32)

        sh = self.th * 0.5
        q = np.array([0.0, np.sin(sh), 0.0, np.cos(sh)], dtype=np.float64)
        pos[self.ship_mask] = np.array([0.0, 0.0, self.zc])
        vel[self.ship_mask] = np.array([0.0, 0.0, self.zcd])
        ori[self.ship_mask] = q
        ang[self.ship_mask] = np.array([0.0, self.thd, 0.0])
        pos[self.fluid_mask] = self.xf.astype(np.float32)
        vel[self.fluid_mask] = self.vf.astype(np.float32)

        mass = self.mass.astype(np.float32)

        # ---- boundary samples --------------------------------------------------
        bvel = np.zeros((len(self.bpts), 3))
        btgt = np.zeros((len(self.bpts), 3))
        u_in = self.last_u_in
        btgt[self.bk_in] = (u_in, 0.0, 0.0)
        bvel[self.bk_in] = (u_in, 0.0, 0.0)
        u_out = float(self.u[-1, :].mean())
        bvel[self.bk_out] = (u_out, 0.0, 0.0)
        btgt[self.bk_out] = (self.u0, 0.0, 0.0)
        uw = self._sample(self.u, self.bx[self.bk_wall], self.by[self.bk_wall])
        bvel[self.bk_wall, 0] = uw

        # ---- budgets ------------------------------------------------------------
        f_hull = self.F_hull
        f_bnd = self.F_bnd
        f_gw = np.array([0.0, 0.0, -self.g * self.Mw])
        f_gs = np.array([0.0, 0.0, -self.g * self.M])
        f_water = f_gw - f_hull + f_bnd
        f_ship = f_gs + f_hull
        f_glob = f_gw + f_gs + f_bnd
        budget_ext = np.stack([f_water, f_ship, f_glob])[:, None, :]  # [G,K,3]

        cum_in = np.stack([self.cum_in, np.zeros(3), self.cum_in])
        cum_out = np.stack([self.cum_out, np.zeros(3), self.cum_out])

        attributes = {
            "phase": self.phase.astype(np.float32),
            "ship_visual": self.ship_visual.astype(np.float32),
            "mass_quantity": mass.copy(),
        }
        fields = {
            "boundary_normal": self.bnrm.copy(),
            "boundary_target_velocity": btgt,
            "boundary_velocity": bvel,
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": cum_in,
            "budget_cumulative_outflow_momentum": cum_out,
            "budget_domain_weights": self.budget_weights.copy(),
            "budget_external_force": budget_ext,
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": f_hull[None, :].copy(),
        }
        scalars = {
            "still_water_level": float(self.wl),
            "reference_velocity": float(self.reference_velocity),
            "reference_momentum": float(self.reference_momentum),
        }
        self._frames += 1
        return StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=ori,
            angular_velocity=ang,
            mass=mass,
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ---------------------------------------------------------- Solver contract
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.dt_sub, time - self.t)
            self._substep(dt)
        if abs(self.t - time) < 1e-9:
            self.t = time
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "coupled_shallow_water_rigid_hull",
            "time": float(self.t),
            "frames": int(self._frames),
            "hull_heave": float(self.zc),
            "hull_pitch": float(self.th),
            "water_mass_kg": float(self.Mw),
            "marker_count": int(self.xf.shape[0]),
        }
