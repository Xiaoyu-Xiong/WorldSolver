"""Channel cylinder vortex-street solver.

2D incompressible Navier-Stokes in vorticity-streamfunction form:

    d(omega)/dt + u.grad(omega) = nu * lap(omega)
    lap(psi) = -omega,   u = (dpsi/dy, -dpsi/dx)

on a cell-centered uniform internal grid (h = 0.05 m) over
x in [-6, 10], y in [-1.5, 1.5].  A fixed no-slip circular cylinder
(center (-3, 0), R = 0.42) is treated by masking: interior cells carry
psi = psi_c and are excluded from the Poisson unknowns; surface vorticity is
generated with a Thom-type no-slip formula.  Channel walls are no-slip
(Thom formula), the inlet prescribes the parabolic profile (centerline
1.2 m/s, mean 0.8 m/s), and the outlet is open (psi_xx = 0, convective
vorticity outflow).  A passive dye scalar is advected with first-order
upwind fluxes plus small diffusion; a prescribed inlet source re-injects the
scene's streamwise stripe pattern so the alternating von Karman wake stays
visible.  The Poisson operator is constant in time, so it is factorized once
(scipy splu) and reused.

Diagnostics are emitted on the scene's 0.1 m cell-centered grid
(2x2 block averages of the internal grid), matching the array conventions
of the frozen scene initial_state.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from worldsolver.contracts import StateFrame

H = 0.05                 # internal grid spacing [m]
DT = 0.0025              # internal time step [s]
X0, X1 = -6.0, 10.0
Y0, Y1 = -1.5, 1.5
CX, CY, RAD = -3.0, 0.0, 0.42
HALF_H = 1.5
KAPPA_DYE = 0.002        # dye diffusivity [m^2/s]


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        self._seed = int(seed)
        try:
            self.nu = float(cfg["fluid"]["kinematic_viscosity"])
        except Exception:
            self.nu = 0.012
        try:
            self.uc = float(cfg["initial_conditions"]["centerline_speed"])
        except Exception:
            self.uc = 1.2
        try:
            cx, cy, _ = cfg["obstacle"]["center"]
            rad = float(cfg["obstacle"]["radius"])
            self.cx, self.cy, self.rad = float(cx), float(cy), rad
        except Exception:
            self.cx, self.cy, self.rad = CX, CY, RAD
        uc, nu = self.uc, self.nu
        self._dt = DT
        self._t = 0.0

        nx = int(round((X1 - X0) / H))
        ny = int(round((Y1 - Y0) / H))
        self.nx, self.ny = nx, ny
        self.xc = X0 + (np.arange(nx) + 0.5) * H
        self.yc = Y0 + (np.arange(ny) + 0.5) * H
        XX, YY = np.meshgrid(self.xc, self.yc, indexing="xy")
        self.XX, self.YY = XX, YY

        dist = np.hypot(XX - self.cx, YY - self.cy)
        self.fluid = dist > self.rad
        self.solid = ~self.fluid
        idx = -np.ones((ny, nx), dtype=np.int64)
        idx[self.fluid] = np.arange(int(self.fluid.sum()))
        self.idx = idx
        n = int(self.fluid.sum())

        # Inlet streamfunction (integral of parabolic profile from bottom wall)
        f = lambda y: uc * (y - y**3 / (3 * HALF_H**2))
        self.psi_in = f(self.yc) - f(Y0)
        self.psi_bot = 0.0
        self.psi_top = float(f(Y1) - f(Y0))       # total flux = 2.4
        self.psi_c = 0.5 * (self.psi_bot + self.psi_top)

        # ---- Poisson matrix: lap(psi) = -omega with Dirichlet ghosts -------
        fi, fj = np.where(self.fluid)
        k = idx[fj, fi]
        inv_h2 = 1.0 / H**2
        R_l, C_l, V_l = [], [], []
        # diagonal
        R_l.append(k); C_l.append(k); V_l.append(np.full(n, -4.0 * inv_h2))
        # neighbors that are fluid cells
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii = fi + di
            jj = fj + dj
            inside = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            nb = inside & self.fluid[np.clip(jj, 0, ny - 1), np.clip(ii, 0, nx - 1)]
            R_l.append(k[nb])
            C_l.append(idx[np.clip(jj, 0, ny - 1), np.clip(ii, 0, nx - 1)][nb])
            V_l.append(np.full(int(nb.sum()), inv_h2))
        # Outlet rows (i = nx-1): ghost psi_g = 2 psi_N - psi_W (psi_xx = 0)
        # -> add +2/h^2 to diagonal and -1/h^2 at west neighbor (COO sums).
        out = fi == nx - 1
        ko = k[out]
        kw = idx[fj[out], fi[out] - 1]
        R_l.append(ko); C_l.append(ko); V_l.append(np.full(len(ko), 2.0 * inv_h2))
        R_l.append(ko); C_l.append(kw); V_l.append(np.full(len(ko), -1.0 * inv_h2))
        A = sp.coo_matrix(
            (np.concatenate(V_l), (np.concatenate(R_l), np.concatenate(C_l))),
            shape=(n, n),
        ).tocsr()
        self._lu = spla.splu(A.tocsc())
        self._n = n

        # Constant ghost contributions to the Poisson RHS.
        g = np.zeros(n)
        g[fi == 0] -= inv_h2 * self.psi_in[fj[fi == 0]]
        g[fj == 0] -= inv_h2 * self.psi_bot
        g[fj == ny - 1] -= inv_h2 * self.psi_top
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii = fi + di
            jj = fj + dj
            inb = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            sol = inb & self.solid[np.clip(jj, 0, ny - 1), np.clip(ii, 0, nx - 1)]
            g[k[sol]] -= inv_h2 * self.psi_c
        self._g_const = g
        self._fi, self._fj, self._fk = fi, fj, k

        # ---- Cylinder surface band (Thom-type no-slip vorticity) ----------
        d = dist - self.rad
        band = self.fluid & (d < 0.7 * H)
        self._band = band
        delta = np.clip(d[band], 0.35 * H, 1.3 * H)
        self._band_coeff = 2.0 / delta**2

        # ---- Inlet vorticity Dirichlet: curl of the parabolic profile -----
        self._om_in = 2.0 * uc * self.yc / HALF_H**2

        # ---- Initial fields ------------------------------------------------
        om = np.zeros((ny, nx))
        om[self.fluid] = np.interp(YY[self.fluid], self.yc, self._om_in)
        rng = np.random.default_rng(self._seed)
        region = self.fluid & (np.abs(XX - (self.cx + self.rad + 0.5)) < 0.9) & (np.abs(YY) < 1.2)
        om[region] += 0.05 * rng.standard_normal((ny, nx))[region]
        self.om = om
        self.psi = np.full((ny, nx), self.psi_c)
        self.psi[self.fluid] = 0.0
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.u[self.fluid] = uc * np.maximum(0.0, 1.0 - (YY[self.fluid] / HALF_H) ** 2)

        # ---- Dye: scene's alternating inlet stripes (frozen pattern) -------
        stripe_f = np.floor((XX - X0) / 0.55).astype(np.int32)
        dye = np.where(XX < X0 + 3.45, ((stripe_f & 1) * 2 - 1), 0.0)
        dye = dye * np.exp(-0.5 * (YY / 0.34) ** 2)
        dye[self.solid] = 0.0
        self.dye = dye
        self.src = np.zeros((ny, nx))
        self._pat_s = 0.0  # advected pattern coordinate for inlet re-injection

        # ---- Output grid (0.1 m, matches scene initial_state) --------------
        self._xs01 = np.arange(X0 + 0.05, X1, 0.1)
        self._ys01 = np.arange(Y0 + 0.05, Y1, 0.1)
        XX2, YY2 = np.meshgrid(self._xs01, self._ys01, indexing="xy")
        self._XX2, self._YY2 = XX2, YY2
        self._keep = (XX2 - self.cx) ** 2 + (YY2 - self.cy) ** 2 > self.rad**2

        # ---- Static boundary samples ---------------------------------------
        pts, nrm, tgt = [], [], []
        xw = np.arange(X0, X1 + 1e-9, 0.2)
        pts.append(np.column_stack([xw, np.full_like(xw, Y0)]))
        nrm.append(np.tile([0.0, 1.0], (len(xw), 1)))
        tgt.append(np.zeros((len(xw), 2)))
        pts.append(np.column_stack([xw, np.full_like(xw, Y1)]))
        nrm.append(np.tile([0.0, -1.0], (len(xw), 1)))
        tgt.append(np.zeros((len(xw), 2)))
        th = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        pts.append(np.column_stack([self.cx + self.rad * np.cos(th),
                                    self.cy + self.rad * np.sin(th)]))
        nrm.append(np.column_stack([np.cos(th), np.sin(th)]))
        tgt.append(np.zeros((len(th), 2)))
        yi = np.linspace(Y0 + 0.05, Y1 - 0.05, 30)
        ui = uc * np.maximum(0.0, 1.0 - (yi / HALF_H) ** 2)
        pts.append(np.column_stack([np.full_like(yi, X0), yi]))
        nrm.append(np.tile([1.0, 0.0], (len(yi), 1)))
        tgt.append(np.column_stack([ui, np.zeros_like(ui)]))
        self.b_pts = np.vstack(pts)
        self.b_nrm = np.vstack(nrm).astype(float)
        self.b_tgt = np.vstack(tgt).astype(float)

        # Shedding probe (downstream of cylinder, off centerline)
        self._probe_sig: list[float] = []
        self._probe_t: list[float] = []
        self._probe_x = self.cx + self.rad + 1.5
        self._probe_y = 0.45

    # ------------------------------------------------------------- internals
    def _solve_psi(self):
        rhs = -self.om[self.fluid] + self._g_const
        vals = self._lu.solve(rhs)
        psi = np.full((self.ny, self.nx), self.psi_c)
        psi[self.fluid] = vals
        self.psi = psi

    def _velocity(self):
        psi = self.psi
        ny, nx = self.ny, self.nx
        pe = np.empty((ny + 2, nx + 2))
        pe[1:-1, 1:-1] = psi
        pe[0, 1:-1] = self.psi_bot
        pe[-1, 1:-1] = self.psi_top
        pe[1:-1, 0] = self.psi_in
        pe[1:-1, -1] = 2.0 * psi[:, -1] - psi[:, -2]
        pe[0, 0] = self.psi_bot; pe[0, -1] = self.psi_bot
        pe[-1, 0] = self.psi_top; pe[-1, -1] = self.psi_top
        u = (pe[2:, 1:-1] - pe[:-2, 1:-1]) / (2 * H)
        v = -(pe[1:-1, 2:] - pe[1:-1, :-2]) / (2 * H)
        u[self.solid] = 0.0
        v[self.solid] = 0.0
        self.u, self.v = u, v

    def _apply_om_bc(self):
        c = 8.0 / H**2  # Thom formula with wall half a cell away
        # walls (no-slip)
        self.om[:, 0] = np.where(self.fluid[:, 0], c * (self.psi_bot - self.psi[:, 0]), 0.0)
        self.om[:, -1] = np.where(self.fluid[:, -1], c * (self.psi_top - self.psi[:, -1]), 0.0)
        # inlet: curl of the prescribed parabolic profile
        self.om[0, :] = np.where(self.fluid[0, :], self._om_in, 0.0)
        # cylinder surface band
        self.om[self._band] = self._band_coeff * (self.psi_c - self.psi[self._band])
        self.om[self.solid] = 0.0

    def _step(self, dt):
        self._solve_psi()
        self._velocity()
        self._apply_om_bc()
        u, v, om = self.u, self.v, self.om
        nu, ny, nx = self.nu, self.ny, self.nx

        # ---- vorticity transport (explicit, central differences) ----
        # padded omega: west ghost = inlet Dirichlet, east ghost = extrapolated,
        # north/south ghosts = wall Thom values.
        omp = np.empty((ny + 2, nx + 2))
        omp[1:-1, 1:-1] = om
        omp[1:-1, 0] = self._om_in
        omp[1:-1, -1] = om[:, -1]
        omp[0, 1:-1] = np.where(self.fluid[0, :], 8.0 / H**2 * (self.psi_bot - self.psi[0, :]), 0.0)
        omp[-1, 1:-1] = np.where(self.fluid[-1, :], 8.0 / H**2 * (self.psi_top - self.psi[-1, :]), 0.0)
        omp[0, 0] = omp[1, 0]; omp[0, -1] = omp[1, -1]
        omp[-1, 0] = omp[-2, 0]; omp[-1, -1] = omp[-2, -1]

        dudx = (omp[1:-1, 2:] - omp[1:-1, :-2]) / (2 * H)
        dudy = (omp[2:, 1:-1] - omp[:-2, 1:-1]) / (2 * H)
        lap = (omp[2:, 1:-1] + omp[:-2, 1:-1] + omp[1:-1, 2:] + omp[1:-1, :-2]
               - 4.0 * om) / H**2
        adv = u * dudx + v * dudy
        new_om = om.copy()
        upd = om + dt * (-adv + nu * lap)
        interior = self.fluid.copy()
        interior[:, 0] = False
        interior[:, -1] = False
        new_om[interior] = upd[interior]
        # inlet column stays Dirichlet
        new_om[0, :] = np.where(self.fluid[0, :], self._om_in, 0.0)
        # outlet column: convective outflow d(om)/dt + u d(om)/dx = 0
        ue = np.maximum(u[:, -1], 0.05)
        domdx = (om[:, -1] - om[:, -2]) / H
        new_om[:, -1] = np.where(self.fluid[:, -1], om[:, -1] - dt * ue * domdx, 0.0)
        new_om[self.solid] = 0.0
        self.om = new_om

        # ---- dye: upwind fluxes + diffusion + inlet source ----
        c = self.dye
        fx = np.zeros((ny, nx + 1))
        fx[:, 1:-1] = (np.maximum(u[:, 1:-1], 0.0) * c[:, :-1]
                       + np.minimum(u[:, 1:-1], 0.0) * c[:, 1:])
        fy = np.zeros((ny + 1, nx))
        fy[1:-1, :] = (np.maximum(v[1:-1, :], 0.0) * c[:-1, :]
                       + np.minimum(v[1:-1, :], 0.0) * c[1:, :])
        advd = (fx[:, 1:] - fx[:, :-1]) / H + (fy[1:, :] - fy[:-1, :]) / H
        cp = np.pad(c, 1, mode="edge")
        lapd = (cp[1:-1, 2:] + cp[1:-1, :-2] + cp[2:, 1:-1] + cp[:-2, 1:-1]
                - 4.0 * c) / H**2
        # inlet source: relax first column toward the frozen stripe pattern
        # advected into the domain at the centerline speed.
        self._pat_s = (self._pat_s + dt * self.uc) % 1.1
        xv = X0 + self._pat_s
        sign = 1.0 if (int((xv - X0) / 0.55) & 1) else -1.0
        target = sign * np.exp(-0.5 * (self.yc / 0.34) ** 2)
        src = np.zeros((ny, nx))
        src[:, 0] = np.where(self.fluid[:, 0], 20.0 * (target - c[:, 0]), 0.0)
        self.src = src
        newc = c + dt * (-advd + KAPPA_DYE * lapd + src)
        newc[self.solid] = 0.0
        self.dye = newc

        self._t += dt

    # ---------------------------------------------------------------- output
    def _coarsen(self, a):
        return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2]
                       + a[0::2, 1::2] + a[1::2, 1::2])

    def _frame(self, t):
        u = self._coarsen(self.u)
        v = self._coarsen(self.v)
        om = self._coarsen(self.om)
        dye = self._coarsen(self.dye)
        src01 = self._coarsen(self.src)
        keep = self._keep
        u[~keep] = 0.0; v[~keep] = 0.0; om[~keep] = 0.0
        dye[~keep] = 0.0; src01[~keep] = 0.0
        ny, nx = keep.shape
        vel3 = np.zeros((ny, nx, 3)); vel3[..., 0] = u; vel3[..., 1] = v
        vel2 = np.zeros((ny, nx, 2)); vel2[..., 0] = u; vel2[..., 1] = v

        XX2, YY2 = self._XX2, self._YY2
        nk = int(keep.sum())
        pos = np.column_stack([XX2[keep], YY2[keep], np.zeros(nk)])
        vel = np.column_stack([u[keep], v[keep], np.zeros(nk)])
        mass = np.full(nk, 1000.0 * 0.1**2)

        # boundary velocity: bilinear interpolation of the diagnostic grid
        gx = np.clip((self.b_pts[:, 0] - (X0 + 0.05)) / 0.1, 0, nx - 1 - 1e-6)
        gy = np.clip((self.b_pts[:, 1] - (Y0 + 0.05)) / 0.1, 0, ny - 1 - 1e-6)
        i0 = gx.astype(int); j0 = gy.astype(int)
        i1 = np.minimum(i0 + 1, nx - 1); j1 = np.minimum(j0 + 1, ny - 1)
        tx = gx - i0; ty = gy - j0
        def bilerp(fq):
            return ((1 - tx) * (1 - ty) * fq[j0, i0] + tx * (1 - ty) * fq[j0, i1]
                    + (1 - tx) * ty * fq[j1, i0] + tx * ty * fq[j1, i1])
        bvel = np.column_stack([bilerp(u), bilerp(v)])

        # shedding probe on the diagnostic grid
        jp = int(np.argmin(np.abs(self._ys01 - self._probe_y)))
        ip = int(np.argmin(np.abs(self._xs01 - self._probe_x)))
        self._probe_sig.append(float(om[jp, ip]))
        self._probe_t.append(float(t))

        return StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=mass,
            attributes={
                "dye": dye[keep],
                "vorticity": om[keep],
                "divergence": np.zeros(nk),
                "velocity_magnitude": np.hypot(u[keep], v[keep]),
            },
            fields={
                "velocity": vel3,
                "velocity_field": vel2,
                "active_mask": keep.astype(np.uint8),
                "vorticity": om,
                "divergence": np.zeros((ny, nx)),
                "dye": dye,
                "transported_scalar": dye,
                "scalar_source": src01,
                "boundary_normal": self.b_nrm,
                "boundary_target_velocity": self.b_tgt,
                "boundary_velocity": bvel,
            },
            scalars={
                "characteristic_length": 2.0 * self.rad,
                "grid_spacing": 0.1,
                "reference_velocity": float(self.uc),
                "cylinder_radius": float(self.rad),
                "mean_inlet_speed": (2.0 / 3.0) * float(self.uc),
                "expected_strouhal_number": 0.2,
            },
        )

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        gap = time - self._t
        if gap > 1e-12:
            n = max(1, int(np.ceil(gap / DT)))
            dt = gap / n
            for _ in range(n):
                self._step(dt)
        frame = self._frame(time)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        out: dict[str, object] = {
            "reynolds_number": float(self.uc * 2.0 * self.rad / self.nu),
            "time": float(self._t),
        }
        try:
            sig = np.asarray(self._probe_sig, dtype=float)
            ts = np.asarray(self._probe_t, dtype=float)
            if len(sig) > 200:
                half = len(sig) // 2
                s = sig[half:]
                t2 = ts[half:]
                m = float(s.mean())
                zc = np.where(np.diff(np.sign(s - m)) != 0)[0]
                if len(zc) >= 4:
                    periods = np.diff(t2[zc]) * 2.0
                    per = float(np.median(periods))
                    f = 1.0 / per
                    out["shedding_frequency_hz"] = f
                    out["shedding_period_s"] = per
                    out["strouhal_number"] = f * 2.0 * self.rad / self.uc
        except Exception:
            pass
        return out
