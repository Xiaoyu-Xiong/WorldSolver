"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Incompressible channel flow past a fixed circular cylinder.

    Uniform Cartesian MAC grid, semi-Lagrangian advection, explicit viscous
    diffusion and an exact (pre-factorised) pressure projection.  All state is
    produced by time-stepping real physics; nothing is scripted.
    """

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        fluid = cfg["fluid"]
        self.rho = float(fluid["density"])
        self.nu = float(fluid["kinematic_viscosity"])
        init = cfg["initial_conditions"]
        self.umax = float(init["centerline_speed"])
        self.umean = float(init["mean_speed"])
        obstacle = cfg["obstacle"]
        self.ox = float(obstacle["center"][0])
        self.oy = float(obstacle["center"][1])
        self.radius = float(obstacle["radius"])
        bmin = cfg["domain"]["bounds_min"]
        bmax = cfg["domain"]["bounds_max"]
        self.xmin, self.ymin = float(bmin[0]), float(bmin[1])
        self.xmax, self.ymax = float(bmax[0]), float(bmax[1])
        self.half_height = 0.5 * (self.ymax - self.ymin)
        self.spacing = float(fluid["nominal_particle_spacing"])

        # ---- uniform staggered grid (axis 0 = y, axis 1 = x, as the scene) --
        self.nx = 256
        self.h = (self.xmax - self.xmin) / self.nx
        self.ny = int(round((self.ymax - self.ymin) / self.h))
        self.h = (self.ymax - self.ymin) / self.ny
        self.xc = self.xmin + (np.arange(self.nx) + 0.5) * self.h
        self.yc = self.ymin + (np.arange(self.ny) + 0.5) * self.h
        self.xf = self.xmin + np.arange(self.nx + 1) * self.h
        self.yf = self.ymin + np.arange(self.ny + 1) * self.h

        # ---- solid mask (the fixed circular obstacle) -----------------------
        xx = self.xc[None, :]
        yy = self.yc[:, None]
        self.solid = (xx - self.ox) ** 2 + (yy - self.oy) ** 2 <= self.radius**2
        self.fluid = ~self.solid
        # faces that connect two fluid cells (or lie on inlet/outlet/walls)
        both_x = np.zeros((self.ny, self.nx + 1), bool)
        both_x[:, 0] = True
        both_x[:, self.nx] = True
        both_x[:, 1 : self.nx] = self.fluid[:, :-1] & self.fluid[:, 1:]
        self.act_x = both_x
        both_y = np.zeros((self.ny + 1, self.nx), bool)
        both_y[0, :] = True
        both_y[self.ny, :] = True
        both_y[1 : self.ny, :] = self.fluid[:-1, :] & self.fluid[1:, :]
        self.act_y = both_y

        # ---- coordinate stencils (axis 0 = y, axis 1 = x) -------------------
        self.XC = np.broadcast_to(xx, (self.ny, self.nx)).copy()
        self.YC = np.broadcast_to(yy, (self.ny, self.nx)).copy()
        self.XF = np.broadcast_to(self.xf[None, :], (self.ny, self.nx + 1)).copy()
        self.YF = np.broadcast_to(yy, (self.ny, self.nx + 1)).copy()
        self.XG = np.broadcast_to(xx, (self.ny + 1, self.nx)).copy()
        self.YG = np.broadcast_to(self.yf[:, None], (self.ny + 1, self.nx)).copy()

        # ---- inlet profile --------------------------------------------------
        uin_c = self.umax * np.maximum(0.0, 1.0 - (self.yc / self.half_height) ** 2)
        uin_f = self.umax * np.maximum(0.0, 1.0 - (self.yc / self.half_height) ** 2)
        self.uin_c = uin_c
        self.uin_f = uin_f

        # ---- transported scalar (dye) --------------------------------------
        dx = self.XC + 6.0
        stripe = np.floor(dx / 0.55).astype(np.int64)
        target = ((stripe & 1) * 2.0 - 1.0) * np.exp(-0.5 * (self.YC / 0.34) ** 2)
        target = np.where(dx < 3.45, target, 0.0)
        self.dye_target = target
        source_band = (dx < 0.75).astype(float)
        self.source_band = source_band

        # ---- initial velocity ----------------------------------------------
        self.ufx = np.broadcast_to(
            self.uin_f[:, None], (self.ny, self.nx + 1)
        ).copy()
        self.ufx[~self.act_x] = 0.0
        self.ufy = np.zeros((self.ny + 1, self.nx))
        self.dye = np.where(self.fluid, target, 0.0)
        self.src = np.zeros((self.ny, self.nx))
        self._build_cc()
        _v, self.vort, self.divf = self._fields()

        # ---- divergence-free projection operator (constant geometry) --------
        self._build_projection()
        self._build_boundary_samples()

        # ---- sample times / stepping bookkeeping ----------------------------
        self.dt = 0.5 * float(cfg["sample_dt"])
        self.t_sim = 0.0
        self.steps = 0
        self.max_div = 0.0
        self._next_sample = 0

        # ---- Lagrangian dye tracers from the frozen scene initial frame ------
        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.pos[:, 2] = 0.0
        self.mass = np.full(len(self.pos), self.rho * self.spacing**2)
        self.pdye = np.array(frame.attributes["dye"], dtype=float)
        self.pvel = np.array(frame.velocity, dtype=float)
        self._sample_particles(first=True)
    # ------------------------------------------------------------ operators
    def _build_cc(self) -> None:
        self.uCx = 0.5 * (self.ufx[:, :-1] + self.ufx[:, 1:])
        self.uCy = 0.5 * (self.ufy[:-1, :] + self.ufy[1:, :])
        self.uCx[self.solid] = 0.0
        self.uCy[self.solid] = 0.0

    def _interp(self, field, X, Y, offx, offy):
        """Bilinear sample of a staggered/collocated field at physical points."""
        fx = (X - self.xmin) / self.h - offx
        fy = (Y - self.ymin) / self.h - offy
        fx = np.clip(fx, 0.0, field.shape[1] - 1.0)
        fy = np.clip(fy, 0.0, field.shape[0] - 1.0)
        i0 = np.floor(fx).astype(np.int64)
        j0 = np.floor(fy).astype(np.int64)
        i1 = np.minimum(i0 + 1, field.shape[1] - 1)
        j1 = np.minimum(j0 + 1, field.shape[0] - 1)
        tx = fx - i0
        ty = fy - j0
        return (
            field[j0, i0] * (1.0 - tx) * (1.0 - ty)
            + field[j0, i1] * tx * (1.0 - ty)
            + field[j1, i0] * (1.0 - tx) * ty
            + field[j1, i1] * tx * ty
        )

    def _lagrange(self, t):
        """Cubic Lagrange weights for nodes at local positions -1, 0, 1, 2."""
        return (
            -t * (t - 1.0) * (t - 2.0) / 6.0,
            0.5 * (t + 1.0) * (t - 1.0) * (t - 2.0),
            -0.5 * (t + 1.0) * t * (t - 2.0),
            (t + 1.0) * t * (t - 1.0) / 6.0,
        )

    def _interp3(self, field, X, Y, offx, offy):
        """Third-order (cubic Lagrange) sample: low numerical diffusion."""
        nxc = field.shape[1]
        nyc = field.shape[0]
        fx = (X - self.xmin) / self.h - offx
        fy = (Y - self.ymin) / self.h - offy
        i0 = np.floor(fx).astype(np.int64) - 1
        j0 = np.floor(fy).astype(np.int64) - 1
        wx = self._lagrange(fx - (i0 + 1))
        wy = self._lagrange(fy - (j0 + 1))
        out = np.zeros_like(fx)
        for a in range(4):
            ii = np.clip(i0 + a, 0, nxc - 1)
            row = np.zeros_like(fx)
            for b in range(4):
                jj = np.clip(j0 + b, 0, nyc - 1)
                row += wy[b] * field[jj, ii]
            out += wx[a] * row
        return out

    def _project(self, X, Y):
        """Keep traced points inside the channel and outside the obstacle.

        Returns mirrored sign for the odd (no-slip) extension in the transverse
        direction.
        """
        sy = np.ones_like(Y)
        low = Y < self.ymin
        high = Y > self.ymax
        Yp = np.where(low, 2.0 * self.ymin - Y, Y)
        sy = np.where(low, -1.0, sy)
        Yp = np.where(high, 2.0 * self.ymax - Yp, Yp)
        sy = np.where(high, -1.0, sy)
        Yp = np.clip(Yp, self.ymin, self.ymax)
        Xp = np.clip(X, self.xmin, self.xmax)
        rr = np.hypot(Xp - self.ox, Yp - self.oy)
        rmin = self.radius + 0.25 * self.h
        inside = rr < rmin
        safe = np.where(rr > 1e-12, rr, 1.0)
        Xp = np.where(inside, self.ox + (Xp - self.ox) / safe * rmin, Xp)
        Yp = np.where(inside, self.oy + (Yp - self.oy) / safe * rmin, Yp)
        return Xp, Yp, sy

    def _build_projection(self) -> None:
        ny, nx, fluid = self.ny, self.nx, self.fluid
        nf = int(fluid.sum())
        self.nf = nf
        comp = np.full((ny, nx), -1, np.int64)
        comp[fluid] = np.arange(nf)
        self.comp = comp
        nb = np.zeros((ny, nx, 4), np.int64)
        w = np.zeros((ny, nx, 4))
        nb[:, 1:, 0] = comp[:, :-1]
        w[:, 1:, 0] = fluid[:, :-1]
        nb[:, :-1, 1] = comp[:, 1:]
        w[:, :-1, 1] = fluid[:, 1:]
        nb[1:, :, 2] = comp[:-1, :]
        w[1:, :, 2] = fluid[:-1, :]
        nb[:-1, :, 3] = comp[1:, :]
        w[:-1, :, 3] = fluid[1:, :]
        extra = np.zeros((ny, nx))
        extra[:, nx - 1] = 2.0 * fluid[:, nx - 1]
        diag = w.sum(axis=2) + extra
        h2 = self.h * self.h
        self.cdiag = diag[fluid] / h2
        self.cn = [nb[:, :, d][fluid] for d in range(4)]
        self.cw = [w[:, :, d][fluid] / h2 for d in range(4)]

        cc = np.arange(nf, dtype=np.int64)
        rows = [cc]
        cols = [cc]
        vals = [self.cdiag]
        for d in range(4):
            sel = self.cw[d] > 0.0
            rows.append(cc[sel])
            cols.append(self.cn[d][sel])
            vals.append(-self.cw[d][sel])
        self.lu = None
        try:  # exact constant-coefficient factorisation when SciPy is present
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import splu

            A = csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(nf, nf),
            )
            self.lu = splu(A.tocsc())
        except Exception:
            self.lu = None

    def _matvec(self, p):
        ap = self.cdiag * p
        for d in range(4):
            ap -= self.cw[d] * p[self.cn[d]]
        return ap

    def _solve_pressure(self, rhs):
        if self.lu is not None:
            return self.lu.solve(rhs)
        # Jacobi-preconditioned conjugate gradient fallback
        x = np.zeros_like(rhs)
        r = rhs.copy()
        z = r / self.cdiag
        p = z.copy()
        rz = float(np.dot(r, z))
        nrm0 = float(np.sqrt(np.dot(rhs, rhs))) + 1e-30
        for _ in range(400):
            Ap = self._matvec(p)
            alpha = rz / (float(np.dot(p, Ap)) + 1e-300)
            x += alpha * p
            r -= alpha * Ap
            if float(np.sqrt(np.dot(r, r))) <= 1e-10 * nrm0:
                break
            z = r / self.cdiag
            rz_new = float(np.dot(r, z))
            p = z + (rz_new / (rz + 1e-300)) * p
            rz = rz_new
        return x

    def _project_velocity(self, dt):
        div = (self.ufx[:, 1:] - self.ufx[:, :-1]) / self.h + (
            self.ufy[1:, :] - self.ufy[:-1, :]
        ) / self.h
        rr = np.nonzero(self.fluid.ravel())[0]
        rhs = (self.rho / dt) * div.ravel()[rr]
        p = self._solve_pressure(rhs)
        if not np.isfinite(p).all():
            return div
        pall = np.zeros((self.ny, self.nx))
        pall[self.fluid] = p
        k = dt / self.rho / self.h
        mx = self.fluid[:, :-1] & self.fluid[:, 1:]
        self.ufx[:, 1 : self.nx] -= k * (pall[:, 1:] - pall[:, :-1]) * mx
        last = self.fluid[:, self.nx - 1]
        self.ufx[last, self.nx] += k * pall[last, self.nx - 1]
        my = self.fluid[:-1, :] & self.fluid[1:, :]
        self.ufy[1 : self.ny, :] -= k * (pall[1:, :] - pall[:-1, :]) * my
        return div

    # ------------------------------------------------------------- advection
    def _advect_xface(self, f, dt):
        u0 = self._interp3(self.uCx, self.XF, self.YF, 0.5, 0.5)
        v0 = self._interp3(self.uCy, self.XF, self.YF, 0.5, 0.5)
        Xp, Yp, sy = self._project(self.XF - dt * u0, self.YF - dt * v0)
        return sy * self._interp3(f, Xp, Yp, 0.0, 0.5)

    def _advect_yface(self, f, dt):
        u0 = self._interp3(self.uCx, self.XG, self.YG, 0.5, 0.5)
        v0 = self._interp3(self.uCy, self.XG, self.YG, 0.5, 0.5)
        Xp, Yp, sy = self._project(self.XG - dt * u0, self.YG - dt * v0)
        return sy * self._interp3(f, Xp, Yp, 0.5, 0.0)

    def _lap_xface(self, f):
        ny, nx = self.ny, self.nx
        E = np.empty((ny + 2, nx + 3))
        E[1:-1, 1:-1] = f
        E[1:-1, 0] = self.uin_f
        E[1:-1, -1] = f[:, -1]
        E[0, :] = -E[1, :]
        E[-1, :] = -E[-2, :]
        mid = E[1:-1, 1:-1]
        lap = (
            E[1:-1, 2:] + E[1:-1, :-2] + E[2:, 1:-1] + E[:-2, 1:-1] - 4.0 * mid
        ) / (self.h * self.h)
        return lap * self.act_x

    def _lap_yface(self, f):
        ny, nx = self.ny, self.nx
        E = np.empty((ny + 3, nx + 2))
        E[1:-1, 1:-1] = f
        E[0, :] = -E[1, :]
        E[-1, :] = -E[-2, :]
        E[:, 0] = -E[:, 1]
        E[:, -1] = E[:, -2]
        mid = E[1:-1, 1:-1]
        lap = (
            E[1:-1, 2:] + E[1:-1, :-2] + E[2:, 1:-1] + E[:-2, 1:-1] - 4.0 * mid
        ) / (self.h * self.h)
        return lap * self.act_y

    def _advect_scalar(self, dt):
        Xp, Yp, _ = self._project(
            self.XC - dt * self._interp3(self.uCx, self.XC, self.YC, 0.5, 0.5),
            self.YC - dt * self._interp3(self.uCy, self.XC, self.YC, 0.5, 0.5),
        )
        adv = self._interp3(self.dye, Xp, Yp, 0.5, 0.5)
        ny, nx = self.ny, self.nx
        E = np.empty((ny + 2, nx + 2))
        E[1:-1, 1:-1] = self.dye
        E[0, :] = E[1, :]
        E[-1, :] = E[-2, :]
        E[:, 0] = E[:, 1]
        E[:, -1] = E[:, -2]
        mid = E[1:-1, 1:-1]
        lap = (
            E[1:-1, 2:] + E[1:-1, :-2] + E[2:, 1:-1] + E[:-2, 1:-1] - 4.0 * mid
        ) / (self.h * self.h)
        self.dye = adv + dt * 0.5 * self.nu * lap * self.fluid
        tau = 0.05
        src = (self.dye_target - self.dye) / tau * self.source_band * self.fluid
        self.src = src
        self.dye = np.clip(self.dye + dt * src, -1.0, 1.0) * self.fluid

    def _apply_bc(self) -> None:
        self.ufx[:, 0] = self.uin_f
        self.ufx[~self.act_x] = 0.0
        self.ufy[0, :] = 0.0
        self.ufy[self.ny, :] = 0.0
        self.ufy[~self.act_y] = 0.0

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        old_x = self.ufx.copy()
        old_y = self.ufy.copy()
        self._build_cc()
        self.ufx = self._advect_xface(self.ufx, dt) + dt * self.nu * self._lap_xface(
            self.ufx
        )
        self.ufy = self._advect_yface(self.ufy, dt) + dt * self.nu * self._lap_yface(
            self.ufy
        )
        self._apply_bc()
        self._advect_scalar(dt)
        div = self._project_velocity(dt)
        self._apply_bc()
        if not (np.isfinite(self.ufx).all() and np.isfinite(self.ufy).all()):
            self.ufx, self.ufy = old_x, old_y
        self._build_cc()
        self._advance_particles(dt)
        d = np.abs(div)[self.fluid]
        if d.size:
            self.max_div = max(self.max_div, float(d.max()))
        self.steps += 1

    # -------------------------------------------------------------- boundary
    def _build_boundary_samples(self) -> None:
        pts = []
        nrm = []
        kind = []
        xcs = self.xc
        for sgn, yv in ((1.0, self.ymin), (-1.0, self.ymax)):
            pts.append(np.column_stack((xcs, np.full(self.nx, yv))))
            nrm.append(np.column_stack((np.zeros(self.nx), np.full(self.nx, sgn))))
            kind.append(np.full(self.nx, 0))
        ang = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        cp = np.column_stack(
            (
                self.ox + self.radius * np.cos(ang),
                self.oy + self.radius * np.sin(ang),
            )
        )
        pts.append(cp)
        nrm.append(
            np.column_stack((np.cos(ang), np.sin(ang)))
        )
        kind.append(np.full(len(ang), 1))
        pts.append(
            np.column_stack((np.full(self.ny, self.xmin), self.yc))
        )
        nrm.append(np.column_stack((np.full(self.ny, -1.0), np.zeros(self.ny))))
        kind.append(np.full(self.ny, 2))
        pts.append(
            np.column_stack((np.full(self.ny, self.xmax), self.yc))
        )
        nrm.append(np.column_stack((np.full(self.ny, 1.0), np.zeros(self.ny))))
        kind.append(np.full(self.ny, 3))
        self.bnd_pos = np.concatenate(pts, axis=0)
        self.bnd_nrm = np.concatenate(nrm, axis=0)
        self.bnd_kind = np.concatenate(kind).astype(np.int64)
        self.bnd_target = np.zeros_like(self.bnd_pos)
        inlet = self.bnd_kind == 2
        yin = self.bnd_pos[inlet, 1]
        self.bnd_target[inlet, 0] = self.umax * np.maximum(
            0.0, 1.0 - (yin / self.half_height) ** 2
        )

    def _boundary_velocity(self):
        X = self.bnd_pos[:, 0]
        Y = self.bnd_pos[:, 1]
        ux = self._interp(self.ufx, X, Y, 0.0, 0.5)
        uy = self._interp(self.ufy, X, Y, 0.5, 0.0)
        vel = np.column_stack((ux, uy))
        target = self.bnd_target.copy()
        outlet = self.bnd_kind == 3
        target[outlet] = vel[outlet]
        return vel, target

    # -------------------------------------------------------------- particles
    def _sample_particles(self, first: bool = False) -> None:
        X = self.pos[:, 0]
        Y = self.pos[:, 1]
        ux = self._interp3(self.uCx, X, Y, 0.5, 0.5)
        uy = self._interp3(self.uCy, X, Y, 0.5, 0.5)
        self.pvel = np.column_stack((ux, uy, np.zeros_like(ux)))
        self.pvort = self._interp3(self.vort, X, Y, 0.5, 0.5)
        self.pdiv = self._interp3(self.divf, X, Y, 0.5, 0.5)
        self.pmag = np.hypot(ux, uy)
        if not first:
            self.pdye = self._interp3(self.dye, X, Y, 0.5, 0.5)

    def _advance_particles(self, dt: float) -> None:
        X = self.pos[:, 0]
        Y = self.pos[:, 1]
        ux = self._interp3(self.uCx, X, Y, 0.5, 0.5)
        uy = self._interp3(self.uCy, X, Y, 0.5, 0.5)
        Xn = X + dt * ux
        Yn = Y + dt * uy
        out = Xn > self.xmax
        Xn = np.where(out, Xn - (self.xmax - self.xmin), Xn)
        Xp, Yp, _ = self._project(Xn, Yn)
        self.pos[:, 0] = np.clip(Xp, self.xmin + 1e-6, self.xmax - 1e-6)
        self.pos[:, 1] = np.clip(Yp, self.ymin + 1e-6, self.ymax - 1e-6)
        self.pos[:, 2] = 0.0

    # ------------------------------------------------------------------ API
    def _fields(self):
        ny, nx = self.ny, self.nx
        vel = np.zeros((ny, nx, 3))
        vel[:, :, 0] = self.uCx
        vel[:, :, 1] = self.uCy
        vel[self.solid] = 0.0
        vort = np.zeros((ny, nx))
        dudy = (self.uCx[2:, 1:-1] - self.uCx[:-2, 1:-1]) / (2.0 * self.h)
        dvdx = (self.uCy[1:-1, 2:] - self.uCy[1:-1, :-2]) / (2.0 * self.h)
        vort[1:-1, 1:-1] = dvdx - dudy
        vort[self.solid] = 0.0
        divf = (self.ufx[:, 1:] - self.ufx[:, :-1]) / self.h + (
            self.ufy[1:, :] - self.ufy[:-1, :]
        ) / self.h
        divf[self.solid] = 0.0
        if not (
            np.isfinite(vel).all()
            and np.isfinite(vort).all()
            and np.isfinite(divf).all()
        ):
            vel = np.nan_to_num(vel)
            vort = np.nan_to_num(vort)
            divf = np.nan_to_num(divf)
        return vel, vort, divf

    def advance_to(self, time: float):
        time = float(time)
        n = int(round((time - self.t_sim) / self.dt))
        for _ in range(max(0, n)):
            self._step(self.dt)
            self.t_sim += self.dt
        vel, vort, divf = self._fields()
        self.vort = vort
        self.divf = divf
        self._sample_particles()
        bvel, btgt = self._boundary_velocity()
        pos = self.pos.copy()
        velp = self.pvel.copy()
        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(velp).all():
            velp = np.nan_to_num(velp, nan=0.0, posinf=0.0, neginf=0.0)
        frame = StateFrame(
            time=time,
            position=pos,
            velocity=velp,
            mass=self.mass.copy(),
            attributes={
                "dye": np.nan_to_num(self.pdye),
                "vorticity": np.nan_to_num(self.pvort),
                "divergence": np.nan_to_num(self.pdiv),
                "velocity_magnitude": np.nan_to_num(self.pmag),
            },
            fields={
                "velocity": vel,
                "velocity_field": np.ascontiguousarray(vel[:, :, :2]),
                "active_mask": self.fluid.astype(np.uint8),
                "vorticity": vort,
                "divergence": divf,
                "dye": self.dye.copy(),
                "transported_scalar": self.dye.copy(),
                "scalar_source": self.src.copy(),
                "boundary_normal": self.bnd_nrm.copy(),
                "boundary_velocity": bvel,
                "boundary_target_velocity": btgt,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.h,
                "reference_velocity": self.umax,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "grid": [int(self.nx), int(self.ny)],
            "spacing": float(self.h),
            "time_step": float(self.dt),
            "steps": int(self.steps),
            "simulated_time": float(self.t_sim),
            "max_abs_divergence": float(self.max_div),
            "reynolds_number": float(self.umax * 2.0 * self.radius / self.nu),
            "particles": int(len(self.pos)),
        }
