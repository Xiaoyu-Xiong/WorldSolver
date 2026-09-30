"""Incompressible Navier-Stokes solver for wall-confined flow past a cylinder.

Physical model
--------------
2D incompressible Navier-Stokes in primitive variables

    du/dt + (u . grad) u = -grad p / rho + nu * lap(u),      div u = 0

discretized with a staggered (MAC) finite-volume grid and advanced with a
fractional-step (Chorin) projection method:

  1. explicit advection (energy-conserving central flux form) + explicit
     viscous diffusion  ->  intermediate velocity u*
  2. pressure Poisson solve  lap(p) = div(u*) / dt  on the fluid cells with
     Neumann (no-flux) conditions on the channel walls, the inlet and the
     immersed cylinder, and a Dirichlet (p = 0) condition on the open outlet
  3. correction  u = u* - dt * grad p   (discretely divergence free)

The circular obstacle is an immersed solid: every face touching a solid cell
is a no-flux/no-slip face held at zero velocity, so incompressibility is
satisfied exactly on the fluid cells while the viscous stencil enforces
no-slip on the cylinder.  A passive dye scalar is transported conservatively
with the same divergence-free face velocities (MUSCL/minmod flux limiter plus
Fickian diffusion) using an alternating inlet stripe pattern so that the
alternating von Karman wake is directly visible.

Only NumPy/SciPy are used.  Everything is deterministic for a fixed seed.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)

        dom = cfg["domain"]
        self.x0 = float(dom["bounds_min"][0])
        self.x1 = float(dom["bounds_max"][0])
        self.y0 = float(dom["bounds_min"][1])
        self.y1 = float(dom["bounds_max"][1])
        self.Lx = self.x1 - self.x0
        self.Ly = self.y1 - self.y0

        fl = cfg["fluid"]
        self.rho = float(fl["density"])
        self.nu = float(fl["kinematic_viscosity"])
        self.spacing = float(fl["nominal_particle_spacing"])   # output grid 0.1

        ic = cfg["initial_conditions"]
        self.u_center = float(ic["centerline_speed"])
        self.u_mean = float(ic["mean_speed"])
        self.half_height = 0.5 * self.Ly

        obs = cfg["obstacle"]
        self.cx = float(obs["center"][0])
        self.cy = float(obs["center"][1])
        self.radius = float(obs["radius"])

        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        # ---------------------------------------------------------- grids
        self.refine = 3                       # internal cells per output cell
        self.h = self.spacing / self.refine   # 0.033333...
        self.nx = int(round(self.Lx / self.h))
        self.ny = int(round(self.Ly / self.h))
        h = self.h
        nx, ny = self.nx, self.ny

        # cell centers / face coordinates
        self.xc = self.x0 + (np.arange(nx) + 0.5) * h
        self.yc = self.y0 + (np.arange(ny) + 0.5) * h
        self.xf = self.x0 + np.arange(nx + 1) * h
        self.yf = self.y0 + np.arange(ny + 1) * h

        # solid (cylinder) cell mask, index order [i, j] = [x, y]
        XC, YC = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.XC, self.YC = XC, YC
        r2 = (XC - self.cx) ** 2 + (YC - self.cy) ** 2
        self.solid = r2 <= self.radius ** 2
        self.fluid = ~self.solid

        # blocked faces (touching a solid cell) and channel walls
        self.blocked_u = np.zeros((nx + 1, ny), dtype=bool)
        self.blocked_u[1:nx, :] = self.solid[:-1, :] | self.solid[1:, :]
        self.blocked_v = np.zeros((nx, ny + 1), dtype=bool)
        self.blocked_v[:, 1:ny] = self.solid[:, :-1] | self.solid[:, 1:]
        self.blocked_v[:, 0] = True          # bottom wall: no through-flow
        self.blocked_v[:, ny] = True         # top wall

        self.open_u = ~self.blocked_u[1:nx, :]      # pressure-coupled x faces
        self.open_v = ~self.blocked_v[:, 1:ny]      # pressure-coupled y faces

        # inlet profile on u faces (parabolic, no-slip at walls)
        self.u_in_face = self.u_center * np.maximum(
            0.0, 1.0 - (self.yc / self.half_height) ** 2
        )

        # ---------------------------------------------------------- state
        self.u = np.zeros((nx + 1, ny))
        self.u[:, :] = self.u_in_face[None, :]
        self.v = np.zeros((nx, ny + 1))

        # small deterministic transverse seed limited to the wake region; this
        # only perturbs the (unstable) symmetric base flow, it does not script
        # the shedding, whose frequency comes out of the momentum equations.
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal((nx, ny + 1))
        wake_v = ((XC[:, 1:] > self.cx) & (np.abs(YC[:, 1:]) < 0.8)).astype(float)
        self.v[:, 1:ny] += 0.005 * self.u_center * noise[:, 1:ny] * wake_v

        self.u[self.blocked_u] = 0.0
        self.v[self.blocked_v] = 0.0

        # dye: alternating inlet stripes concentrated near the channel center
        self.stripe_width = 0.55
        self.dye_sigma = 0.34
        stripe = np.floor((XC - self.x0) / self.stripe_width).astype(np.int64)
        c0 = np.where(XC < self.cx + 0.45, (stripe % 2) * 2.0 - 1.0, 0.0)
        c0 *= np.exp(-0.5 * (YC / self.dye_sigma) ** 2)
        c0[self.solid] = 0.0
        self.c = c0
        self.kappa = 0.002

        # ---------------------------------------------------------- Poisson
        self._build_poisson()
        self.t = 0.0
        self.dt = self.sample_dt / 7.0
        self._project(self.dt)     # make the initial field divergence free

        # ---------------------------------------------------------- output
        self._build_output_grid()
        self._build_boundary_samples()
        self._steps = 0

    # ---------------------------------------------------------------- Poisson
    def _build_poisson(self) -> None:
        nx, ny, h = self.nx, self.ny, self.h
        idx = np.full((nx, ny), -1, dtype=np.int64)
        nf = int(self.fluid.sum())
        idx[self.fluid] = np.arange(nf, dtype=np.int64)
        self.idx = idx
        self.nfluid = nf

        rows = []
        cols = []
        vals = []
        diag = np.zeros(nf)

        m = self.open_u
        a = idx[0:nx - 1, :][m]
        b = idx[1:nx, :][m]
        rows.append(a); cols.append(b); vals.append(np.ones(a.size))
        rows.append(b); cols.append(a); vals.append(np.ones(a.size))
        np.add.at(diag, a, -1.0)
        np.add.at(diag, b, -1.0)

        m = self.open_v
        a = idx[:, 0:ny - 1][m]
        b = idx[:, 1:ny][m]
        rows.append(a); cols.append(b); vals.append(np.ones(a.size))
        rows.append(b); cols.append(a); vals.append(np.ones(a.size))
        np.add.at(diag, a, -1.0)
        np.add.at(diag, b, -1.0)

        # open outlet: p = 0 at the last x face (Dirichlet on the face)
        out = idx[nx - 1, :]
        out = out[out >= 0]
        np.add.at(diag, out, -2.0)

        rows.append(np.arange(nf, dtype=np.int64))
        cols.append(np.arange(nf, dtype=np.int64))
        vals.append(diag)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals) / (h * h)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(nf, nf))
        self.lu = spla.splu(A)

    def _divergence(self) -> np.ndarray:
        h = self.h
        return (self.u[1:, :] - self.u[:-1, :]) / h + (
            self.v[:, 1:] - self.v[:, :-1]
        ) / h

    def _project(self, dt: float) -> None:
        nx, ny, h = self.nx, self.ny, self.h
        div = self._divergence()
        rhs = div[self.fluid] / dt
        pv = self.lu.solve(rhs)
        p = np.zeros((nx, ny))
        p[self.fluid] = pv
        self.p = p

        gx = np.zeros((nx + 1, ny))
        gx[1:nx, :] = np.where(self.open_u, (p[1:, :] - p[:-1, :]) / h, 0.0)
        gx[nx, :] = -2.0 * p[nx - 1, :] / h        # outlet Dirichlet p_face = 0
        self.u -= dt * gx
        self.u[self.blocked_u] = 0.0
        self.u[0, :] = self.u_in_face             # prescribed inlet

        gy = np.zeros((nx, ny + 1))
        gy[:, 1:ny] = np.where(self.open_v, (p[:, 1:] - p[:, :-1]) / h, 0.0)
        self.v -= dt * gy
        self.v[self.blocked_v] = 0.0

    # ------------------------------------------------------------- time step
    def _body_force_v(self, t: float) -> np.ndarray:
        """Short-lived localized transverse disturbance that seeds the wake
        instability (switched off well before the measurement window)."""
        if t >= 0.45:
            return 0.0
        nx, ny = self.nx, self.ny
        X = self.XC[:, 1:]           # v-face positions (interior in y)
        Y = self.y0 + np.arange(1, ny)[None, :] * self.h
        Y = np.broadcast_to(Y, (nx, ny - 1))
        g = np.exp(-(((X - (self.cx + 0.55)) ** 2 + (Y - 0.22) ** 2) /
                     (2.0 * 0.16 ** 2)))
        f = np.zeros((nx, ny + 1))
        f[:, 1:ny] = 1.4 * g
        return f

    def _step(self, dt: float) -> None:
        nx, ny, h = self.nx, self.ny, self.h
        u, v = self.u, self.v
        nu = self.nu

        # ---- convective fluxes (central, flux form: conserves energy) ----
        uc = 0.5 * (u[:-1, :] + u[1:, :])            # (nx, ny) cell centered
        vc = 0.5 * (v[:, :-1] + v[:, 1:])            # (nx, ny)

        u_cor = np.zeros((nx + 1, ny + 1))
        u_cor[:, 1:ny] = 0.5 * (u[:, :-1] + u[:, 1:])   # 0 on both walls
        v_cor = np.zeros((nx + 1, ny + 1))
        v_cor[1:nx, :] = 0.5 * (v[:-1, :] + v[1:, :])
        v_cor[0, :] = 0.0                                # inlet: v = 0
        v_cor[nx, :] = v[nx - 1, :]                      # outflow
        F = u_cor * v_cor

        adv_u = np.zeros((nx + 1, ny))
        adv_u[1:nx, :] = (uc[1:, :] ** 2 - uc[:-1, :] ** 2) / h
        adv_u += (F[:, 1:] - F[:, :-1]) / h

        adv_v = np.zeros((nx, ny + 1))
        adv_v[:, 1:ny] = (vc[:, 1:] ** 2 - vc[:, :-1] ** 2) / h
        adv_v += (F[1:, :] - F[:-1, :]) / h

        # ---- viscous terms -------------------------------------------------
        ue = np.empty((nx + 1, ny + 2))
        ue[:, 1:-1] = u
        ue[:, 0] = -u[:, 0]           # no-slip bottom wall
        ue[:, -1] = -u[:, -1]         # no-slip top wall
        lap_u = (ue[:, 2:] - 2.0 * ue[:, 1:-1] + ue[:, :-2]) / (h * h)
        lap_u[1:nx, :] += (u[2:, :] - 2.0 * u[1:nx, :] + u[:nx - 1, :]) / (h * h)

        ve = np.empty((nx + 2, ny + 1))
        ve[1:-1, :] = v
        ve[0, :] = -v[0, :]           # inlet: v = 0 on the face
        ve[-1, :] = v[-1, :]          # outflow
        lap_v = (ve[2:, :] - 2.0 * ve[1:-1, :] + ve[:-2, :]) / (h * h)
        lap_v[:, 1:ny] += (v[:, 2:] - 2.0 * v[:, 1:ny] + v[:, :ny - 1]) / (h * h)

        # ---- explicit predictor -------------------------------------------
        us = u + dt * (-adv_u + nu * lap_u)
        vs = v + dt * (-adv_v + nu * lap_v + self._body_force_v(self.t))

        us[0, :] = self.u_in_face                    # prescribed inlet
        us[nx, :] = us[nx - 1, :]                    # zero-gradient outflow
        us[self.blocked_u] = 0.0
        vs[self.blocked_v] = 0.0

        self.u, self.v = us, vs
        self._project(dt)

        # ---- passive dye transport (uses the divergence-free faces) -------
        self._advect_scalar(dt)
        self.t += dt

    # ------------------------------------------------------- scalar transport
    def _inlet_dye(self, t: float) -> np.ndarray:
        k = int(np.floor(-self.u_center * t / self.stripe_width))
        s = float((k % 2) * 2 - 1)
        return s * np.exp(-0.5 * (self.yc / self.dye_sigma) ** 2)

    @staticmethod
    def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(a * b <= 0.0, 0.0,
                        np.where(np.abs(a) < np.abs(b), a, b))

    def _advect_scalar(self, dt: float) -> None:
        nx, ny, h = self.nx, self.ny, self.h
        c = self.c
        u, v = self.u, self.v

        cin = self._inlet_dye(self.t)

        # x direction ------------------------------------------------------
        cxg = np.empty((nx + 2, ny))
        cxg[1:-1, :] = c
        cxg[0, :] = cin
        cxg[-1, :] = c[-1, :]
        sx = self._minmod(cxg[1:-1, :] - cxg[:-2, :], cxg[2:, :] - cxg[1:-1, :])

        Fx = np.zeros((nx + 1, ny))
        cL = c[:-1, :] + 0.5 * sx[:-1, :]
        cR = c[1:, :] - 0.5 * sx[1:, :]
        uin = u[1:nx, :]
        Fx[1:nx, :] = np.where(uin >= 0.0, uin * cL, uin * cR)
        Fx[1:nx, :] -= self.kappa * (c[1:, :] - c[:-1, :]) / h
        Fx[1:nx, :] = np.where(self.open_u, Fx[1:nx, :], 0.0)
        Fx[0, :] = np.where(u[0, :] >= 0.0, u[0, :] * cin, u[0, :] * c[0, :])
        Fx[nx, :] = np.where(u[nx, :] >= 0.0, u[nx, :] * c[nx - 1, :], 0.0)

        # y direction ------------------------------------------------------
        cyg = np.empty((nx, ny + 2))
        cyg[:, 1:-1] = c
        cyg[:, 0] = c[:, 0]
        cyg[:, -1] = c[:, -1]
        sy = self._minmod(cyg[:, 1:-1] - cyg[:, :-2], cyg[:, 2:] - cyg[:, 1:-1])

        Fy = np.zeros((nx, ny + 1))
        cB = c[:, :-1] + 0.5 * sy[:, :-1]
        cT = c[:, 1:] - 0.5 * sy[:, 1:]
        vin = v[:, 1:ny]
        Fy[:, 1:ny] = np.where(vin >= 0.0, vin * cB, vin * cT)
        Fy[:, 1:ny] -= self.kappa * (c[:, 1:] - c[:, :-1]) / h
        Fy[:, 1:ny] = np.where(self.open_v, Fy[:, 1:ny], 0.0)

        cnew = c - dt * ((Fx[1:, :] - Fx[:-1, :]) / h +
                         (Fy[:, 1:] - Fy[:, :-1]) / h)
        cnew[self.solid] = 0.0
        self.c = np.clip(cnew, -1.5, 1.5)

    # ------------------------------------------------------------- diagnostics
    def _build_output_grid(self) -> None:
        sp_ = self.spacing
        xs = np.arange(self.x0 + 0.5 * sp_, self.x1, sp_)
        ys = np.arange(self.y0 + 0.5 * sp_, self.y1, sp_)
        self.xs_out, self.ys_out = xs, ys
        self.nxo, self.nyo = len(xs), len(ys)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")     # shape (nyo, nxo)
        self.xx_out, self.yy_out = xx, yy
        d2 = (xx - self.cx) ** 2 + (yy - self.cy) ** 2
        self.keep = d2 > self.radius ** 2
        self.position = np.column_stack(
            (xx[self.keep], yy[self.keep], np.zeros(int(self.keep.sum())))
        )
        self.n_particles = len(self.position)

        # equation-applicability mask: fluid cells whose collocated stencil
        # does not cross the immersed cylinder or the domain boundary
        active = self.keep & (d2 > (self.radius + 1.2 * sp_) ** 2)
        active[0, :] = False
        active[-1, :] = False
        active[:, 0] = False
        active[:, -1] = False
        self.active_mask = active.astype(np.uint8)

        w = self.fluid.astype(float)
        self.blk_w = w.reshape(self.nxo, self.refine,
                               self.nyo, self.refine).sum(axis=(1, 3))
        self.blk_w_safe = np.where(self.blk_w > 0.0, self.blk_w, 1.0)
        self.blk_valid = self.blk_w > 0.0

    def _coarsen(self, field_ij: np.ndarray) -> np.ndarray:
        """Fluid-weighted block average of an internal [nx,ny] field ->
        output [nyo,nxo] array."""
        w = self.fluid.astype(float)
        acc = (field_ij * w).reshape(self.nxo, self.refine,
                                    self.nyo, self.refine).sum(axis=(1, 3))
        out = np.where(self.blk_valid, acc / self.blk_w_safe, 0.0)
        return out.T

    def _cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        uc = np.where(self.fluid, uc, 0.0)
        vc = np.where(self.fluid, vc, 0.0)
        return uc, vc

    def _cell_vorticity(self, uc: np.ndarray, vc: np.ndarray) -> np.ndarray:
        nx, ny, h = self.nx, self.ny, self.h
        ue = np.empty((nx + 2, ny + 2))
        ve = np.empty((nx + 2, ny + 2))
        ue[1:-1, 1:-1] = uc
        ve[1:-1, 1:-1] = vc
        ue[0, 1:-1] = 2.0 * self.u_in_face - uc[0, :]
        ve[0, 1:-1] = -vc[0, :]
        ue[-1, 1:-1] = uc[-1, :]
        ve[-1, 1:-1] = vc[-1, :]
        ue[:, 0] = -ue[:, 1]
        ue[:, -1] = -ue[:, -2]
        ve[:, 0] = -ve[:, 1]
        ve[:, -1] = -ve[:, -2]
        dvdx = (ve[2:, 1:-1] - ve[:-2, 1:-1]) / (2.0 * h)
        dudy = (ue[1:-1, 2:] - ue[1:-1, :-2]) / (2.0 * h)
        return dvdx - dudy

    def _extended_velocity(self, uc: np.ndarray, vc: np.ndarray):
        nx, ny = self.nx, self.ny
        Ux = np.empty((nx + 2, ny + 2))
        Uy = np.empty((nx + 2, ny + 2))
        Ux[1:-1, 1:-1] = uc
        Uy[1:-1, 1:-1] = vc
        Ux[0, 1:-1] = 2.0 * self.u_in_face - uc[0, :]
        Uy[0, 1:-1] = -vc[0, :]
        Ux[-1, 1:-1] = uc[-1, :]
        Uy[-1, 1:-1] = vc[-1, :]
        Ux[:, 0] = -Ux[:, 1]
        Ux[:, -1] = -Ux[:, -2]
        Uy[:, 0] = -Uy[:, 1]
        Uy[:, -1] = -Uy[:, -2]
        return Ux, Uy

    def _build_boundary_samples(self) -> None:
        # cylinder surface samples
        nth = 72
        th = np.arange(nth) * (2.0 * np.pi / nth)
        cx_pts = self.cx + self.radius * np.cos(th)
        cy_pts = self.cy + self.radius * np.sin(th)
        cyl_n = np.column_stack((np.cos(th), np.sin(th)))   # outward into fluid

        # channel wall samples
        nw = 60
        xw = np.linspace(self.x0 + 0.25, self.x1 - 0.25, nw)
        low_pts = np.column_stack((xw, np.full(nw, self.y0)))
        up_pts = np.column_stack((xw, np.full(nw, self.y1)))
        low_n = np.column_stack((np.zeros(nw), np.ones(nw)))
        up_n = np.column_stack((np.zeros(nw), -np.ones(nw)))

        self.bnd_pts = np.vstack((np.column_stack((cx_pts, cy_pts)),
                                  low_pts, up_pts))
        self.bnd_normal = np.vstack((cyl_n, low_n, up_n))
        self.bnd_target = np.zeros_like(self.bnd_pts)   # every wall is static

    def _sample_boundary_velocity(self, uc, vc) -> np.ndarray:
        Ux, Uy = self._extended_velocity(uc, vc)
        h = self.h
        xe0 = self.x0 - 0.5 * h
        ye0 = self.y0 - 0.5 * h
        gx = (self.bnd_pts[:, 0] - xe0) / h
        gy = (self.bnd_pts[:, 1] - ye0) / h
        i0 = np.clip(np.floor(gx).astype(int), 0, self.nx)
        j0 = np.clip(np.floor(gy).astype(int), 0, self.ny)
        tx = np.clip(gx - i0, 0.0, 1.0)
        ty = np.clip(gy - j0, 0.0, 1.0)
        out = np.zeros((len(self.bnd_pts), 2))
        for k, A in enumerate((Ux, Uy)):
            a00 = A[i0, j0]
            a10 = A[i0 + 1, j0]
            a01 = A[i0, j0 + 1]
            a11 = A[i0 + 1, j0 + 1]
            out[:, k] = ((1 - tx) * (1 - ty) * a00 + tx * (1 - ty) * a10 +
                         (1 - tx) * ty * a01 + tx * ty * a11)
        return out

    # ----------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        eps = 1e-9
        guard = 0
        while self.t < target - eps and guard < 2_000_000:
            dt = min(self.dt, target - self.t)
            if dt <= eps:
                break
            self._step(dt)
            guard += 1
            self._steps += 1
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        uc, vc = self._cell_velocity()
        vort = self._cell_vorticity(uc, vc)
        div = self._divergence()

        vx_o = self._coarsen(uc)
        vy_o = self._coarsen(vc)
        vort_o = self._coarsen(vort)
        div_o = self._coarsen(div)
        dye_o = self._coarsen(self.c)

        mask2d = self.keep
        vx_o = np.where(mask2d, vx_o, 0.0)
        vy_o = np.where(mask2d, vy_o, 0.0)
        vort_o = np.where(mask2d, vort_o, 0.0)
        div_o = np.where(mask2d, div_o, 0.0)
        dye_o = np.where(mask2d, dye_o, 0.0)

        vel3 = np.zeros(vx_o.shape + (3,))
        vel3[..., 0] = vx_o
        vel3[..., 1] = vy_o
        vel2 = np.stack((vx_o, vy_o), axis=-1)

        pvx = vx_o[mask2d]
        pvy = vy_o[mask2d]
        velocity = np.column_stack((pvx, pvy, np.zeros_like(pvx)))
        mass = np.full(self.n_particles, self.rho * self.spacing ** 2)

        bnd_vel = self._sample_boundary_velocity(uc, vc)

        frame = StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0),
            mass=mass,
            attributes={
                "dye": np.nan_to_num(dye_o[mask2d]),
                "vorticity": np.nan_to_num(vort_o[mask2d]),
                "divergence": np.nan_to_num(div_o[mask2d]),
                "velocity_magnitude": np.sqrt(pvx ** 2 + pvy ** 2),
            },
            fields={
                "velocity": np.nan_to_num(vel3),
                "velocity_field": np.nan_to_num(vel2),
                "active_mask": self.active_mask,
                "vorticity": np.nan_to_num(vort_o),
                "divergence": np.nan_to_num(div_o),
                "dye": np.nan_to_num(dye_o),
                "transported_scalar": np.nan_to_num(dye_o),
                "scalar_source": np.zeros_like(dye_o),
                "boundary_velocity": np.nan_to_num(bnd_vel),
                "boundary_target_velocity": self.bnd_target.copy(),
                "boundary_normal": self.bnd_normal.copy(),
            },
            scalars={
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(2.0 * self.radius),
                "reference_velocity": float(self.u_mean),
                "cylinder_radius": float(self.radius),
                "mean_inlet_speed": float(self.u_mean),
                "centerline_speed": float(self.u_center),
                "kinematic_viscosity": float(self.nu),
                "dynamic_viscosity": float(self.rho * self.nu),
                "reynolds_number": float(self.u_mean * 2.0 * self.radius / self.nu),
                "expected_strouhal_number": float(
                    self.cfg["diagnostics"]["expected_strouhal_number"]
                ),
                "internal_grid_spacing": float(self.h),
                "time_step": float(self.dt),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC staggered projection (Chorin) incompressible NSE",
            "advection": "central flux form (momentum), MUSCL-minmod (scalar)",
            "pressure_solver": "sparse LU of the masked 5-point Poisson operator",
            "internal_grid": [int(self.nx), int(self.ny)],
            "internal_spacing": float(self.h),
            "time_step": float(self.dt),
            "steps_taken": int(self._steps),
            "reynolds_number": float(self.u_mean * 2.0 * self.radius / self.nu),
            "immersed_boundary": "no-slip staircase cylinder + no-slip channel walls",
        }
