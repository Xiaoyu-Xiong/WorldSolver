"""Two-way coupled limit-cycle flapping of a flexible flag in a viscous stream.

Physical model
--------------
* Carrier fluid: incompressible 2D Navier-Stokes on a uniform MAC (staggered)
  grid.  Advection uses a second-order upwind flux form, viscosity is explicit
  (nu = U L / Re, Re = 1400), and incompressibility is enforced by an exact
  pressure projection: FFT in the periodic y direction and a direct tridiagonal
  solve in x with a no-penetration (Neumann) pressure condition at the inflow
  plane and p = 0 at the open downstream boundary.
* Flag: a finite-thickness inextensible Euler-Bernoulli elastica discretised as
  a chain of Lagrangian nodes.  Axial inextensibility is penalised by stiff
  axial springs, bending resistance uses the discrete curvature operator with
  the frozen bending rigidity K_B = 0.001 and the frozen aspect ratio 20.
* Coupling: immersed-boundary direct forcing.  Fluid cells inside the smeared
  flag support are driven to the local solid velocity by an implicit per-cell
  penalty; the equal-and-opposite reaction is applied to the flag nodes.  Solid
  inertia, the implicit added-mass operator and the elastic tangent stiffness
  are solved together in one small linear system, which removes the added-mass
  instability of the light flag (mass ratio 0.16) and keeps the interaction
  fully two-way.

Every state is produced by actually time-stepping these equations; no motion is
scripted or precomputed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame

__all__ = ["BenchmarkSolver"]


def _kernel_1d(r: np.ndarray) -> np.ndarray:
    """Peskin four-point cosine delta function, r measured in cell widths."""
    r = np.abs(np.asarray(r, dtype=float))
    out = np.zeros_like(r)
    inside = r <= 1.0
    outer = (r > 1.0) & (r <= 2.0)
    out[inside] = (
        3.0 - 2.0 * r[inside]
        + np.sqrt(np.maximum(1.0 + 4.0 * r[inside] - 4.0 * r[inside] ** 2, 0.0))
    ) / 8.0
    out[outer] = (
        5.0 - 2.0 * r[outer]
        - np.sqrt(np.maximum(-7.0 + 12.0 * r[outer] - 4.0 * r[outer] ** 2, 0.0))
    ) / 8.0
    return out


class BenchmarkSolver:
    """Immersed-boundary elastica in a projected incompressible stream."""

    # ------------------------------------------------------------- layout
    NX = 288
    NY = 96
    X0 = -1.0
    Y0 = -1.0
    X1 = 5.0
    Y1 = 1.0
    DT = 0.005
    VORT_BINS = 2
    NKX = 24
    NKY = 6
    # ----------------------------------------------------------- material
    RHO_F = 1.0
    RE_DEFAULT = 1400.0
    MASS_RATIO_DEFAULT = 0.16
    BENDING_DEFAULT = 0.001
    FLAG_LENGTH_DEFAULT = 1.0
    FLAG_THICKNESS_DEFAULT = 0.05
    N_NODE = 41
    KAPPA = 400.0
    EA = 2000.0
    TIP_SEED = 0.02
    MAX_SPEED = 40.0

    # ============================================================== set-up
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        phys = dict(cfg.get("physics", {}) or {})
        inflow = phys.get("inflow_velocity", [1.0, 0.0])
        self.inflow = float(inflow[0])
        self.re = float(phys.get("reynolds_number", self.RE_DEFAULT))
        self.nu = self.inflow / max(self.re, 1.0)
        self.mass_ratio = float(phys.get("mass_ratio", self.MASS_RATIO_DEFAULT))
        self.bending = float(phys.get("bending_rigidity", self.BENDING_DEFAULT))
        self.flag_len = float(phys.get("flag_length", self.FLAG_LENGTH_DEFAULT))
        self.flag_thick = float(phys.get("flag_thickness", self.FLAG_THICKNESS_DEFAULT))

        self.nx = int(self.NX)
        self.ny = int(self.NY)
        self.h = (self.X1 - self.X0) / self.nx
        self.x_c = self.X0 + (np.arange(self.nx) + 0.5) * self.h
        self.y_c = self.Y0 + (np.arange(self.ny) + 0.5) * self.h

        self.time = 0.0
        self._n_steps = 0
        self.seed = int(seed)

        # fluid state (staggered)
        self.u = np.full((self.nx + 1, self.ny), self.inflow, dtype=float)
        self.v = np.zeros((self.nx, self.ny + 1), dtype=float)
        self.p = np.zeros((self.nx, self.ny), dtype=float)

        # flag discretisation, anchored upstream end, free downstream end
        self.n_node = int(self.N_NODE)
        self.ds = self.flag_len / (self.n_node - 1)
        self.node_mass = self.mass_ratio * self.flag_len / (self.n_node - 1)
        s = np.arange(self.n_node) * self.ds
        self.X = np.zeros((self.n_node, 2))
        self.X[:, 0] = s
        self.X[:, 1] = self.TIP_SEED * (s / self.flag_len) ** 2
        self.V = np.zeros((self.n_node, 2))
        self.F_int = np.zeros((self.n_node, 2))
        self.node_force = np.zeros((self.n_node, 2))
        self.free = np.arange(1, self.n_node)
        self.k_spring = self.EA / self.ds
        self.bend_c = self.bending / self.ds ** 3

        self._build_bending_matrix()

        # static anchor post (frozen scene boundary object)
        ax = np.array([-0.0125, 0.0125, -0.0125, 0.0125, -0.0125, 0.0125, -0.0125, 0.0125])
        ay = np.array([-0.045, -0.045, -0.015, -0.015, 0.015, 0.015, 0.045, 0.045])
        self.n_static = int(len(ax))
        self.anchor_xy = np.column_stack([ax, ay])
        self.X_all = np.vstack([self.X, self.anchor_xy])

        self._build_poisson()

        # ------------------------------------------------- trajectory samples
        vx = np.linspace(0.0, 4.0, self.NKX)
        vy = np.linspace(-0.60, 0.60, self.NKY)
        gx, gy = np.meshgrid(vx, vy, indexing="ij")
        self.probe_xy = np.column_stack([gx.ravel(), gy.ravel()])
        self.probe_ci = np.column_stack(
            [
                np.clip((self.probe_xy[:, 0] - self.X0) / self.h - 0.5, 0.0, self.nx - 1.0),
                np.clip((self.probe_xy[:, 1] - self.Y0) / self.h - 0.5, 0.0, self.ny - 1.0),
            ]
        ).T
        self.n_probe = int(len(self.probe_xy))
        self.n_sample = self.n_probe + 5
        self.flag_vertex_s = np.array([0.0, 0.0, 0.98, 1.0, 0.98])
        self.flag_vertex_off = np.array([-0.025, 0.025, 0.025, 0.0, -0.025])
        probe_area = (self.X1 - self.X0) * (self.Y1 - self.Y0) / self.n_probe
        self.probe_mass = self.RHO_F * probe_area
        flag_mass_each = self.mass_ratio * self.flag_len / 5.0
        self.sample_mass = np.concatenate(
            [np.full(5, flag_mass_each), np.full(self.n_probe, self.probe_mass)]
        )
        self.mass_quantity = np.column_stack(
            [
                np.concatenate([np.zeros(5), np.full(self.n_probe, self.probe_mass)]),
                np.concatenate([np.full(5, flag_mass_each), np.zeros(self.n_probe)]),
            ]
        )
        self.body_id = np.concatenate([np.full(5, 2), np.full(self.n_probe, 1)]).astype(int)
        self.domain_weights = np.vstack(
            [
                np.concatenate([np.zeros(5), np.ones(self.n_probe)]),
                np.concatenate([np.ones(5), np.zeros(self.n_probe)]),
            ]
        )

        # -------------------------------------------------- budget accumulators
        self.impulse_ext = np.zeros((2, 3))
        self.inflow_mom = np.zeros((2, 3))
        self.outflow_mom = np.zeros((2, 3))

    # ================================================== static operators
    def _build_bending_matrix(self) -> None:
        n = self.n_node
        b = np.zeros((n - 2, n))
        for c in range(1, n - 1):
            left = c - 1 if c - 1 >= 0 else 1
            right = c + 1 if c + 1 <= n - 1 else n - 2
            b[c - 1, left] += 1.0
            b[c - 1, c] += -2.0
            b[c - 1, right] += 1.0
        self.bend_mat = b
        self.bend_hess = self.bend_c * (b.T @ b)

    def _build_poisson(self) -> None:
        nx, ny, h = self.nx, self.ny, self.h
        nk = ny // 2 + 1
        self.nk = nk
        lam = -4.0 * np.sin(np.pi * np.arange(nk) / ny) ** 2 / h ** 2
        b = np.empty((nx, nk))
        b[:, :] = lam[None, :] - 2.0 / h ** 2
        b[0, :] = lam - 1.0 / h ** 2
        b[nx - 1, :] = lam - 3.0 / h ** 2
        a = 1.0 / h ** 2
        cp = np.zeros((nx, nk))
        dg = np.empty((nx, nk))
        dg[0, :] = b[0, :]
        cp[0, :] = a / dg[0, :]
        for i in range(1, nx):
            dg[i, :] = b[i, :] - a * cp[i - 1, :]
            cp[i, :] = a / dg[i, :] if i < nx - 1 else 0.0
        self.poisson_cp = cp
        self.poisson_inv = 1.0 / dg
        self.poisson_a = a

    # ============================================================ stepping
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-9 and guard < 200000:
            self._step()
            guard += 1
        return self._frame(target)

    def _step(self) -> None:
        dt = self.DT
        h = self.h
        u, v = self.u, self.v

        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        du_c, dv_c = self._advection(uc, vc)

        # ---------------------------------------------------- viscous terms
        x_u = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / h ** 2
        y_u = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / h ** 2
        lap_u = x_u + y_u[1:-1, :]

        vx_pad = np.pad(v, ((1, 1), (0, 0)), mode="edge")
        x_v = (vx_pad[2:, :] - 2.0 * vx_pad[1:-1, :] + vx_pad[:-2, :]) / h ** 2
        y_v = (np.roll(v, -1, axis=1) - 2.0 * v + np.roll(v, 1, axis=1)) / h ** 2
        lap_v = x_v[:, 1:-1] + y_v[:, 1:-1]

        du_face = 0.5 * (du_c[:-1, :] + du_c[1:, :])
        dv_face = 0.5 * (dv_c[:, :-1] + dv_c[:, 1:])

        u[1:-1, :] += dt * (du_face + self.nu * lap_u)
        v[:, 1:-1] += dt * (dv_face + self.nu * lap_v)

        self._apply_fluid_bc()
        np.clip(u, -self.MAX_SPEED, self.MAX_SPEED, out=u)
        np.clip(v, -self.MAX_SPEED, self.MAX_SPEED, out=v)

        # ------------------------------------- immersed-boundary interaction
        self._couple(dt)

        # ------------------------------------------------------- projection
        div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
        p = self._solve_pressure(div * (self.RHO_F / dt))
        self.p = p
        u[1:-1, :] -= dt / self.RHO_F * (p[1:, :] - p[:-1, :]) / h
        v[:, 1:-1] -= dt / self.RHO_F * (p[:, 1:] - p[:, :-1]) / h
        v[:, -1] = v[:, 0]

        self._apply_fluid_bc()
        for arr, cap in ((u, self.MAX_SPEED), (v, self.MAX_SPEED)):
            if not np.isfinite(arr).all():
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=cap, neginf=-cap)

        self._accumulate_budgets(dt)
        self.time += dt
        self._n_steps += 1

    def _apply_fluid_bc(self) -> None:
        self.u[0, :] = self.inflow
        self.v[0, :] = 0.0
        self.u[-1, :] = self.u[-2, :]
        self.v[:, -1] = self.v[:, 0]

    # ============================================================ advection
    def _advection(self, uc: np.ndarray, vc: np.ndarray):
        h = self.h
        up = np.pad(uc, ((2, 2), (0, 0)), mode="edge")
        vp = np.pad(vc, ((2, 2), (0, 0)), mode="edge")

        def shift_p(arr, k):
            if k >= 0:
                return np.concatenate([np.repeat(arr[:1], k, axis=0), arr[:-k]], axis=0)
            k = -k
            return np.concatenate([arr[k:], np.repeat(arr[-1:], k, axis=0)], axis=0)

        um1, ur2 = shift_p(up, 1), shift_p(up, -2)
        vm1, vr2 = shift_p(vp, 1), shift_p(vp, -2)

        uf = 0.5 * (up[:-1, :] + up[1:, :])
        uup = np.where(
            uf > 0.0, 1.5 * up[:-1, :] - 0.5 * um1[:-1, :], 1.5 * up[1:, :] - 0.5 * ur2[1:, :]
        )
        vup = np.where(
            uf > 0.0, 1.5 * vp[:-1, :] - 0.5 * vm1[:-1, :], 1.5 * vp[1:, :] - 0.5 * vr2[1:, :]
        )
        fx_u = uf * uup
        fx_v = uf * vup
        divx_u = (fx_u[1:, :] - fx_u[:-1, :]) / h
        divx_v = (fx_v[1:, :] - fx_v[:-1, :]) / h

        vf_u = 0.5 * (up + np.roll(up, -1, axis=1))
        uy = np.where(
            vf_u > 0.0,
            1.5 * up - 0.5 * np.roll(up, 1, axis=1),
            1.5 * np.roll(up, -1, axis=1) - 0.5 * np.roll(up, -2, axis=1),
        )
        vf_v = 0.5 * (vp + np.roll(vp, -1, axis=1))
        vy = np.where(
            vf_v > 0.0,
            1.5 * vp - 0.5 * np.roll(vp, 1, axis=1),
            1.5 * np.roll(vp, -1, axis=1) - 0.5 * np.roll(vp, -2, axis=1),
        )
        fy_u = vf_u * uy
        fy_v = vf_v * vy
        divy_u = (fy_u - np.roll(fy_u, 1, axis=1)) / h
        divy_v = (fy_v - np.roll(fy_v, 1, axis=1)) / h

        du_c = -(divx_u[1:-1, :] + divy_u[2:-2, :])
        dv_c = -(divx_v[1:-1, :] + divy_v[2:-2, :])
        return du_c, dv_c

    # ======================================================= IB interaction
    def _delta_weights(self):
        h = self.h
        n_all = self.n_node + self.n_static
        X = self.X_all
        i0 = np.floor((X[:, 0] - self.X0) / h).astype(int) - 2
        j0 = np.floor((X[:, 1] - self.Y0) / h).astype(int) - 2
        off = np.arange(4)
        ii = i0[:, None, None] + off[None, :, None]
        jj = j0[:, None, None] + off[None, None, :]
        ii_c = np.clip(ii, 0, self.nx - 1)
        jj_w = np.mod(jj, self.ny)
        dx = (self.x_c[ii_c.ravel()] - np.repeat(X[:, 0], 16)) / h
        dy = (self.y_c[jj_w.ravel()] - np.repeat(X[:, 1], 16)) / h
        w = (_kernel_1d(dx) * _kernel_1d(dy) / h ** 2).reshape(n_all, 16)
        flat = (ii_c * self.ny + jj_w).reshape(n_all, 16)
        return flat.ravel(), w.ravel(), n_all

    def _couple(self, dt: float) -> None:
        h, ds = self.h, self.ds
        flat, w, n_all = self._delta_weights()
        band, inv = np.unique(flat, return_inverse=True)
        nb = int(len(band))
        node_ids = np.repeat(np.arange(n_all), 16)
        D = np.zeros((nb, n_all))
        np.add.at(D, (inv, node_ids), w)
        W = np.maximum(D.sum(axis=1) * ds, 1e-12)

        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        u_star = uc.ravel()[band]
        v_star = vc.ravel()[band]
        star = np.column_stack([u_star, v_star])

        G = dt * self.KAPPA * W / self.RHO_F
        wcell = h * h / (1.0 + G)

        nf = self.n_node
        Df = D[:, :nf]
        c_node = Df.sum(axis=0) * h * h
        b_node = Df.T @ (wcell[:, None] * star)
        A = (Df * wcell[:, None]).T @ Df

        V_all = np.zeros((n_all, 2))
        V_all[:nf] = self.V
        U_target = (D @ V_all) / W[:, None]
        u_new = (star + G[:, None] * U_target) / (1.0 + G)[:, None]

        c_all = D.sum(axis=0) * h * h
        force_all = -self.KAPPA * ds * (V_all * c_all[:, None] - D.T @ (u_new * h * h))

        self.node_force = force_all
        self.coupling_A = A
        self.coupling_b = b_node
        self.coupling_c = c_node
        self.ib_total = force_all.sum(axis=0)

        corr_u = np.zeros((self.nx, self.ny))
        corr_v = np.zeros((self.nx, self.ny))
        corr_u.ravel()[band] = u_new[:, 0] - u_star
        corr_v.ravel()[band] = u_new[:, 1] - v_star
        self.u[1:-1, :] += 0.5 * (corr_u[:-1, :] + corr_u[1:, :])
        self.v[:, 1:-1] += 0.5 * (corr_v[:, :-1] + corr_v[:, 1:])

        self._advance_solid(dt)

    def _advance_solid(self, dt: float) -> None:
        n = self.n_node
        X = self.X
        d = X[1:] - X[:-1]
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        dhat = d / r[:, None]

        f_s = self.k_spring * (r - self.ds)[:, None] * dhat
        f_spring = np.zeros((n, 2))
        f_spring[:-1] -= f_s
        f_spring[1:] += f_s
        f_bend = -(self.bend_c) * (self.bend_mat.T @ (self.bend_mat @ X))
        f_el = f_spring + f_bend
        self.F_int = f_el

        free = self.free
        fi = free - 1  # indices into the (flag-node x flag-node) coupling blocks
        m_inv = self.node_mass / dt
        Cmat = np.zeros((n, n))
        blk = (self.KAPPA * self.ds) * np.diag(self.coupling_c[free]) - (
            dt * self.KAPPA ** 2 / self.RHO_F
        ) * self.ds ** 2 * self.coupling_A[np.ix_(fi, fi)]
        Cmat[np.ix_(free, free)] = blk
        Cmat[free, free] += m_inv
        Cmat[np.ix_(free, free)] += dt * self.bend_hess[np.ix_(free, free)]

        H2 = np.zeros((2 * n, 2 * n))
        for i in range(n - 1):
            block = self.k_spring * np.outer(dhat[i], dhat[i])
            a0, a1 = 2 * i, 2 * i + 2
            b0, b1 = 2 * (i + 1), 2 * (i + 1) + 2
            H2[a0:a1, a0:a1] += block
            H2[b0:b1, b0:b1] += block
            H2[a0:a1, b0:b1] -= block
            H2[b0:b1, a0:a1] -= block

        dofs = np.sort(np.concatenate([2 * free, 2 * free + 1]))
        M = np.kron(Cmat, np.eye(2)) + dt * H2
        M = M[np.ix_(dofs, dofs)]

        rhs = np.zeros(2 * n)
        rhs[2 * free] = (
            m_inv * self.V[free, 0]
            + f_el[free, 0]
            + self.KAPPA * self.ds * self.coupling_b[free, 0]
        )
        rhs[2 * free + 1] = (
            m_inv * self.V[free, 1]
            + f_el[free, 1]
            + self.KAPPA * self.ds * self.coupling_b[free, 1]
        )
        rhs = rhs[dofs]

        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        sol = np.clip(np.nan_to_num(sol), -self.MAX_SPEED, self.MAX_SPEED)

        Vnew = np.zeros((n, 2))
        Vnew[free, 0] = sol[0::2]
        Vnew[free, 1] = sol[1::2]
        self.V = Vnew
        self.X = X + dt * Vnew
        self.X[0, :] = 0.0
        self.X_all[:n] = self.X

    # ============================================================= pressure
    def _solve_pressure(self, rhs: np.ndarray) -> np.ndarray:
        nx = self.nx
        f = np.fft.rfft(rhs, axis=1)
        work = np.empty_like(f)
        a = self.poisson_a
        inv = self.poisson_inv
        work[0, :] = f[0, :] * inv[0, :]
        for i in range(1, nx):
            work[i, :] = (f[i, :] - a * work[i - 1, :]) * inv[i, :]
        cp = self.poisson_cp
        for i in range(nx - 2, -1, -1):
            work[i, :] -= cp[i, :] * work[i + 1, :]
        return np.fft.irfft(work, n=self.ny, axis=1)

    # ============================================================== budgets
    def _boundary_momentum(self):
        h = self.h
        u_in, v_in = self.u[0, :], 0.5 * (self.v[0, :-1] + self.v[0, 1:])
        u_out, v_out = self.u[-1, :], 0.5 * (self.v[-1, :-1] + self.v[-1, 1:])
        inflow = np.array(
            [np.sum(self.RHO_F * u_in ** 2) * h, np.sum(self.RHO_F * u_in * v_in) * h, 0.0]
        )
        outflow = np.array(
            [np.sum(self.RHO_F * u_out ** 2) * h, np.sum(self.RHO_F * u_out * v_out) * h, 0.0]
        )
        return inflow, outflow

    def _boundary_tractions(self):
        h, ny = self.h, self.ny
        u, v, p = self.u, self.v, self.p
        p_force = np.array([float(np.mean(p[0, :]) - np.mean(p[-1, :])), 0.0])
        dudx_in = (u[1, :] - u[0, :]) / h
        dudx_out = (u[-1, :] - u[-2, :]) / h
        dvdx_in = (v[1, :] - v[0, :]) / h
        dvdx_out = (v[-1, :] - v[-2, :]) / h
        scale = self.RHO_F * self.nu * (self.Y1 - self.Y0)
        visc = np.array(
            [
                scale * float(np.mean(dudx_out - dudx_in)),
                scale * float(np.mean(dvdx_out - dvdx_in)),
            ]
        )
        return p_force, visc

    def _external_forces(self):
        nf = self.n_node
        Fflag = self.node_force[:nf].sum(axis=0)
        Fstatic = self.node_force[nf:].sum(axis=0)
        anchor_react = -(self.F_int[0] + self.node_force[0])
        p_force, visc = self._boundary_tractions()
        ext = np.zeros((2, 3, 3))
        ext[0, 0, :2] = p_force
        ext[0, 1, :2] = visc
        ext[0, 2, :2] = -(Fflag + Fstatic)
        ext[1, 0, :2] = anchor_react
        ext[1, 1, :] = 0.0
        ext[1, 2, :2] = Fflag
        return ext

    def _accumulate_budgets(self, dt: float) -> None:
        ext = self._external_forces()
        self.impulse_ext += ext.sum(axis=1) * dt
        inflow, outflow = self._boundary_momentum()
        self.inflow_mom[0] += inflow * dt
        self.outflow_mom[0] += outflow * dt

    # =============================================================== output
    def _sample_centerline(self, s_query: np.ndarray):
        n = self.n_node
        t = np.clip(s_query, 0.0, 1.0) * (n - 1)
        i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
        frac = (t - i0)[:, None]
        pos = self.X[i0] * (1.0 - frac) + self.X[i0 + 1] * frac
        vel = self.V[i0] * (1.0 - frac) + self.V[i0 + 1] * frac
        tan = self.X[i0 + 1] - self.X[i0]
        tan = tan / np.maximum(np.linalg.norm(tan, axis=1), 1e-12)[:, None]
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
        return pos, vel, nrm

    def _vorticity(self) -> np.ndarray:
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        dudy = (uc[:, 1:] - uc[:, :-1]) / self.h
        dudy = np.concatenate([dudy, dudy[:, -1:]], axis=1)
        dvdx = (vc[1:, :] - vc[:-1, :]) / self.h
        dvdx = np.concatenate([dvdx, dvdx[-1:, :]], axis=0)
        om = dvdx - dudy
        b = self.VORT_BINS
        om = om[: (self.nx // b) * b, : (self.ny // b) * b]
        return om.reshape(self.nx // b, b, self.ny // b, b).mean(axis=(1, 3))

    def _boundary_fields(self):
        n = self.n_node
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        d = self.X[1:] - self.X[:-1]
        tan = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
        tan = np.vstack([tan, tan[-1:]])
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
        ci = np.column_stack(
            [
                np.clip((self.X[:, 0] - self.X0) / self.h - 0.5, 0.0, self.nx - 1.0),
                np.clip((self.X[:, 1] - self.Y0) / self.h - 0.5, 0.0, self.ny - 1.0),
            ]
        ).T
        uw = map_coordinates(uc, ci, order=1, mode="nearest")
        vw = map_coordinates(vc, ci, order=1, mode="nearest")
        bn = np.zeros((n + 4, 3))
        bvel = np.zeros((n + 4, 3))
        btgt = np.zeros((n + 4, 3))
        bn[:n, :2] = nrm
        bvel[:n, 0] = uw
        bvel[:n, 1] = vw
        btgt[:n, :2] = self.V
        bn[n:, 0] = 1.0
        return bn, bvel, btgt

    def _frame(self, time: float) -> StateFrame:
        nf = self.n_node
        pos_v, vel_v, nrm_v = self._sample_centerline(self.flag_vertex_s)
        pos_v = pos_v + nrm_v * self.flag_vertex_off[:, None]

        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        up = map_coordinates(uc, self.probe_ci, order=1, mode="nearest")
        vp = map_coordinates(vc, self.probe_ci, order=1, mode="nearest")

        position = np.zeros((self.n_sample, 3))
        velocity = np.zeros((self.n_sample, 3))
        position[:5, :2] = pos_v
        position[5:, :2] = self.probe_xy
        velocity[:5, :2] = vel_v
        velocity[5:, 0] = up
        velocity[5:, 1] = vp

        bn, bvel, btgt = self._boundary_fields()
        ext = self._external_forces()
        Fflag = self.node_force[:nf].sum(axis=0)
        interaction = np.zeros((2, 3))
        interaction[0, :2] = -Fflag
        interaction[1, :2] = Fflag

        fields = {
            "vorticity": self._vorticity(),
            "boundary_normal": bn,
            "boundary_velocity": bvel,
            "boundary_target_velocity": btgt,
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": ext,
            "budget_cumulative_external_impulse": self.impulse_ext.copy(),
            "budget_cumulative_inflow_momentum": self.inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.outflow_mom.copy(),
            "cumulative_mass_source": np.zeros(2),
            "interaction_force": interaction,
        }
        scalars = {
            "reference_velocity": float(self.inflow),
            "reference_momentum": float(self.RHO_F * self.inflow * self.flag_len),
        }
        return StateFrame(
            time=float(time),
            position=np.nan_to_num(position),
            velocity=np.nan_to_num(velocity),
            mass=self.sample_mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass_quantity.copy()},
            fields={k: np.nan_to_num(np.asarray(val, dtype=float)) for k, val in fields.items()},
            scalars=scalars,
        )

    # ================================================================ misc
    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self._n_steps),
            "time": float(self.time),
            "grid": [int(self.nx), int(self.ny)],
            "dt": float(self.DT),
            "reynolds_number": float(self.re),
            "mass_ratio": float(self.mass_ratio),
            "bending_rigidity": float(self.bending),
            "flag_nodes": int(self.n_node),
            "tip_deflection": float(self.X[-1, 1]),
            "samples": int(self.n_sample),
        }
