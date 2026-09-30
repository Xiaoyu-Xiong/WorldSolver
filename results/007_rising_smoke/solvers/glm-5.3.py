"""Eulerian buoyant-smoke solver for scene 007_rising_smoke.

Physically real time-stepping on a staggered (MAC) grid:

  * semi-Lagrangian advection of velocity, smoke density and temperature,
  * buoyancy as the only external force (thermal lift minus smoke weight),
  * vorticity confinement to preserve the rolling vortices,
  * prescribed emitter source terms (density/temperature/velocity inflow),
  * exact incompressibility via a pressure projection whose Poisson operator
    is a Kronecker sum of three small symmetric tridiagonal operators
    (closed free-slip walls -> homogeneous Neumann, open outflow faces ->
    zero boundary pressure), solved directly by separable
    eigen-decomposition (dense 1D eigenbases + tensor contraction).

No Genesis; NumPy/SciPy only. Deterministic: no randomness in the update.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# grid sampling helpers
# ---------------------------------------------------------------------------


def _sample_trilinear(field: np.ndarray, x, y, z, origin, inv_h):
    """Trilinear sample of a 3D field whose first node sits at ``origin``."""
    nx, ny, nz = field.shape
    gx = np.clip((x - origin[0]) * inv_h[0], 0.0, nx - 1.0 - 1.0e-7)
    gy = np.clip((y - origin[1]) * inv_h[1], 0.0, ny - 1.0 - 1.0e-7)
    gz = np.clip((z - origin[2]) * inv_h[2], 0.0, nz - 1.0 - 1.0e-7)
    i0 = gx.astype(np.int64)
    j0 = gy.astype(np.int64)
    k0 = gz.astype(np.int64)
    fx = gx - i0
    fy = gy - j0
    fz = gz - k0
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)
    gxf, gyf, gzf = 1.0 - fx, 1.0 - fy, 1.0 - fz
    return (
        field[i0, j0, k0] * gxf * gyf * gzf
        + field[i1, j0, k0] * fx * gyf * gzf
        + field[i0, j1, k0] * gxf * fy * gzf
        + field[i1, j1, k0] * fx * fy * gzf
        + field[i0, j0, k1] * gxf * gyf * fz
        + field[i1, j0, k1] * fx * gyf * fz
        + field[i0, j1, k1] * gxf * fy * fz
        + field[i1, j1, k1] * fx * fy * fz
    )


class _MACGrid:
    """Staggered grid bookkeeping for the frozen voxel domain."""

    def __init__(self, bounds_min, bounds_max, shape):
        self.lo = np.asarray(bounds_min, dtype=np.float64)
        self.hi = np.asarray(bounds_max, dtype=np.float64)
        self.shape = tuple(int(v) for v in shape)
        self.nx, self.ny, self.nz = self.shape
        self.h = (self.hi - self.lo) / np.asarray(self.shape, dtype=np.float64)
        self.inv_h = 1.0 / self.h
        self.dV = float(np.prod(self.h))

        cx = self.lo[0] + (np.arange(self.nx) + 0.5) * self.h[0]
        cy = self.lo[1] + (np.arange(self.ny) + 0.5) * self.h[1]
        cz = self.lo[2] + (np.arange(self.nz) + 0.5) * self.h[2]
        self.Xc, self.Yc, self.Zc = np.meshgrid(cx, cy, cz, indexing="ij")

        fxp = self.lo[0] + np.arange(self.nx + 1) * self.h[0]
        fyp = self.lo[1] + (np.arange(self.ny) + 0.5) * self.h[1]
        fzp = self.lo[2] + (np.arange(self.nz) + 0.5) * self.h[2]
        self.Xf, self.Yf, self.Zf = np.meshgrid(fxp, fyp, fzp, indexing="ij")

        gxp = self.lo[0] + (np.arange(self.nx) + 0.5) * self.h[0]
        gyp = self.lo[1] + np.arange(self.ny + 1) * self.h[1]
        gzp = self.lo[2] + (np.arange(self.nz) + 0.5) * self.h[2]
        self.Xg, self.Yg, self.Zg = np.meshgrid(gxp, gyp, gzp, indexing="ij")

        hxp = self.lo[0] + (np.arange(self.nx) + 0.5) * self.h[0]
        hyp = self.lo[1] + (np.arange(self.ny) + 0.5) * self.h[1]
        hzp = self.lo[2] + np.arange(self.nz + 1) * self.h[2]
        self.Xh, self.Yh, self.Zh = np.meshgrid(hxp, hyp, hzp, indexing="ij")

        # origins of the staggered component grids (first node location)
        self.o_c = self.lo + 0.5 * self.h
        self.o_u = np.array([self.lo[0], self.o_c[1], self.o_c[2]])
        self.o_v = np.array([self.o_c[0], self.lo[1], self.o_c[2]])
        self.o_w = np.array([self.o_c[0], self.o_c[1], self.lo[2]])

    def cell_velocity(self, u, v, w):
        uc = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        return uc, vc, wc

    def divergence(self, u, v, w):
        return (
            np.diff(u, axis=0) * self.inv_h[0]
            + np.diff(v, axis=1) * self.inv_h[1]
            + np.diff(w, axis=2) * self.inv_h[2]
        )


# ---------------------------------------------------------------------------
# separable Poisson solver for the frozen boundary conditions
# ---------------------------------------------------------------------------


def _axis_operator(n: int, closed_low: bool, closed_high: bool) -> np.ndarray:
    """1D second-difference operator on cell centers along one axis.

    Closed (free-slip wall) end: ghost value mirrors the boundary value, so
    the wall-neighbor term vanishes (diagonal relaxes from -2 to -1).
    Open (outflow) end: the boundary-face pressure is zero, i.e. the ghost
    cell value is zero, keeping the diagonal at -2.
    """
    L = (
        np.diag(-2.0 * np.ones(n))
        + np.diag(np.ones(n - 1), 1)
        + np.diag(np.ones(n - 1), -1)
    )
    if closed_low:
        L[0, 0] += 1.0
    if closed_high:
        L[-1, -1] += 1.0
    return L


class _SeparablePoisson:
    """Direct solver for A p = rhs, A = (Lx (+) Ly (+) Lz) / h^2.

    The 3D operator is the Kronecker sum of the per-axis 1D operators, so it
    diagonalizes in the tensor product of the (small, dense) 1D eigenbases.
    """

    def __init__(self, g: _MACGrid):
        if not np.allclose(g.h, g.h[0]):
            raise ValueError("separable solver requires a uniform grid")
        # closed free-slip: x_min, y_min, y_max, z_min; open outflow: x_max, z_max
        Lx = _axis_operator(g.nx, True, False)
        Ly = _axis_operator(g.ny, True, True)
        Lz = _axis_operator(g.nz, True, False)
        self.lx, self.Vx = np.linalg.eigh(Lx)
        self.ly, self.Vy = np.linalg.eigh(Ly)
        self.lz, self.Vz = np.linalg.eigh(Lz)
        h2 = float(g.h[0] * g.h[0])
        lam = (
            self.lx[:, None, None]
            + self.ly[None, :, None]
            + self.lz[None, None, :]
        ) / h2
        # strictly negative definite because of the Dirichlet (open) faces
        self.inv_lam = 1.0 / np.where(np.abs(lam) < 1.0e-14, -1.0e-14, lam)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        c = np.einsum("ia,ajk->ijk", self.Vx.T, rhs, optimize=True)
        c = np.einsum("jb,ibk->ijk", self.Vy.T, c, optimize=True)
        c = np.einsum("kc,ijc->ijk", self.Vz.T, c, optimize=True)
        c = c * self.inv_lam
        p = np.einsum("ia,ajk->ijk", self.Vx, c, optimize=True)
        p = np.einsum("jb,ibk->ijk", self.Vy, p, optimize=True)
        p = np.einsum("kc,ijc->ijk", self.Vz, p, optimize=True)
        return p


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class BenchmarkSolver:
    """Incompressible buoyant-smoke solver on the frozen voxel grid."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        dom = cfg["domain"]
        init = cfg["initial_conditions"]
        src = init["smoke_source"]

        self.g = _MACGrid(dom["bounds_min"], dom["bounds_max"], dom["grid_shape"])
        self.sample_dt = float(cfg["sample_dt"])
        self.dt = self.sample_dt if self.sample_dt > 0.0 else 0.02
        self.poisson = _SeparablePoisson(self.g)

        # ---- initial fields from the frozen scene initial state ------------
        frame0 = scene.initial_state()
        d0 = np.asarray(frame0.fields["density"], dtype=np.float64)
        v0 = np.asarray(frame0.fields["velocity"], dtype=np.float64)
        if d0.shape != self.g.shape:
            raise ValueError("initial density grid does not match domain grid")
        self.d = d0.copy()

        center = np.asarray(src["center"], dtype=np.float64)
        radius = float(src["radius"])
        half_h = float(src["half_height"])
        radial2 = (
            (self.g.Xc - center[0]) ** 2 + (self.g.Yc - center[1]) ** 2
        ) / (radius * radius)
        vertical2 = (self.g.Zc - center[2]) ** 2 / (half_h * half_h)
        envelope = np.exp(-2.5 * (radial2 + vertical2))
        self.envelope = np.where(d0 > 1.0e-6, np.clip(envelope, 0.0, 1.0), 0.0)
        self.T = float(src["temperature_excess"]) * envelope
        self.T = np.where(d0 > 1.0e-6, self.T, 0.0)

        # ---- staggered face velocities --------------------------------------
        self.u = np.zeros((self.g.nx + 1, self.g.ny, self.g.nz), dtype=np.float64)
        self.v = np.zeros((self.g.nx, self.g.ny + 1, self.g.nz), dtype=np.float64)
        self.w = np.zeros((self.g.nx, self.g.ny, self.g.nz + 1), dtype=np.float64)
        self.u[1:-1, :, :] = 0.5 * (v0[:-1, :, :, 0] + v0[1:, :, :, 0])
        self.v[:, 1:-1, :] = 0.5 * (v0[:, :-1, :, 1] + v0[:, 1:, :, 1])
        self.w[:, :, 1:-1] = 0.5 * (v0[:, :, :-1, 2] + v0[:, :, 1:, 2])
        self._apply_wall_bc()

        # ---- prescribed emitter ---------------------------------------------
        self.d_target = float(src["peak_density"]) * envelope
        self.T_target = float(src["temperature_excess"]) * envelope
        self.src_vel = np.asarray(src["initial_velocity"], dtype=np.float64)
        self.emit_rate = 8.0  # relaxation rate toward the emitter profile [1/s]

        # ---- model parameters ------------------------------------------------
        forcing = cfg.get("forcing", {})
        buoy_dir = np.asarray(
            forcing.get("buoyancy_direction", [0.0, 0.0, 1.0]), dtype=np.float64
        )
        sign = 1.0 if buoy_dir[2] >= 0.0 else -1.0
        self.kappa_T = 1.1 * sign    # thermal buoyancy acceleration [m/s^2]
        self.kappa_d = 0.05 * sign   # smoke weight [m/s^2]
        self.cooling = 0.35          # temperature excess decay rate [1/s]
        self.vort_eps = 0.28         # vorticity confinement coefficient
        half_life = float(init.get("density_half_life", 6.0))
        self.density_decay = _LN2 / max(half_life, 1.0e-6)

        # ---- bookkeeping ------------------------------------------------------
        self.time = 0.0
        self.steps = 0
        self.added_mass = 0.0
        self.removed_mass = 0.0
        self.scalar_source = np.zeros(self.g.shape, dtype=np.float64)
        self.div_residual = 0.0
        self.max_speed = 0.0

    # -- boundary conditions ---------------------------------------------------

    def _apply_wall_bc(self):
        """Closed free-slip walls: zero normal velocity at wall faces."""
        self.u[0, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0

    # -- physics ----------------------------------------------------------------

    def _apply_forces(self, dt):
        g = self.g
        uc, vc, wc = g.cell_velocity(self.u, self.v, self.w)
        # buoyancy (only external force): f_z = kappa_T * T - kappa_d * d
        fx = np.zeros(g.shape)
        fy = np.zeros(g.shape)
        fz = self.kappa_T * self.T - self.kappa_d * self.d
        # vorticity confinement (fluid-internal): f = eps * h * (N x omega)
        if self.vort_eps > 0.0:
            dwdx = np.gradient(wc, g.h[0], axis=0)
            dwdy = np.gradient(wc, g.h[1], axis=1)
            dudz = np.gradient(uc, g.h[2], axis=2)
            dvdz = np.gradient(vc, g.h[2], axis=2)
            dvdx = np.gradient(vc, g.h[0], axis=0)
            dudy = np.gradient(uc, g.h[1], axis=1)
            owx = dwdy - dvdz
            owy = dudz - dwdx
            owz = dvdx - dudy
            eta = np.sqrt(owx * owx + owy * owy + owz * owz)
            if float(eta.max()) > 1.0e-12:
                etax = np.gradient(eta, g.h[0], axis=0)
                etay = np.gradient(eta, g.h[1], axis=1)
                etaz = np.gradient(eta, g.h[2], axis=2)
                norm = np.sqrt(etax * etax + etay * etay + etaz * etaz) + 1.0e-12
                nx_, ny_, nz_ = etax / norm, etay / norm, etaz / norm
                hs = float(np.mean(g.h))
                fx = fx + self.vort_eps * hs * (ny_ * owz - nz_ * owy)
                fy = fy + self.vort_eps * hs * (nz_ * owx - nx_ * owz)
                fz = fz + self.vort_eps * hs * (nx_ * owy - ny_ * owx)
        # distribute cell-centered forces onto the staggered faces
        self.u[1:-1, :, :] += dt * 0.5 * (fx[:-1, :, :] + fx[1:, :, :])
        self.v[:, 1:-1, :] += dt * 0.5 * (fy[:, :-1, :] + fy[:, 1:, :])
        self.w[:, :, 1:-1] += dt * 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])

    def _advect_velocity(self, dt):
        g = self.g
        uc, vc, wc = g.cell_velocity(self.u, self.v, self.w)
        # x-faces
        vx = _sample_trilinear(uc, g.Xf, g.Yf, g.Zf, g.o_c, g.inv_h)
        vy = _sample_trilinear(vc, g.Xf, g.Yf, g.Zf, g.o_c, g.inv_h)
        vz = _sample_trilinear(wc, g.Xf, g.Yf, g.Zf, g.o_c, g.inv_h)
        u_new = _sample_trilinear(
            self.u, g.Xf - dt * vx, g.Yf - dt * vy, g.Zf - dt * vz, g.o_u, g.inv_h
        )
        # y-faces
        vx = _sample_trilinear(uc, g.Xg, g.Yg, g.Zg, g.o_c, g.inv_h)
        vy = _sample_trilinear(vc, g.Xg, g.Yg, g.Zg, g.o_c, g.inv_h)
        vz = _sample_trilinear(wc, g.Xg, g.Yg, g.Zg, g.o_c, g.inv_h)
        v_new = _sample_trilinear(
            self.v, g.Xg - dt * vx, g.Yg - dt * vy, g.Zg - dt * vz, g.o_v, g.inv_h
        )
        # z-faces
        vx = _sample_trilinear(uc, g.Xh, g.Yh, g.Zh, g.o_c, g.inv_h)
        vy = _sample_trilinear(vc, g.Xh, g.Yh, g.Zh, g.o_c, g.inv_h)
        vz = _sample_trilinear(wc, g.Xh, g.Yh, g.Zh, g.o_c, g.inv_h)
        w_new = _sample_trilinear(
            self.w, g.Xh - dt * vx, g.Yh - dt * vy, g.Zh - dt * vz, g.o_w, g.inv_h
        )
        self.u, self.v, self.w = u_new, v_new, w_new

    def _project(self, dt):
        g = self.g
        div = g.divergence(self.u, self.v, self.w)
        p = self.poisson.solve(div / dt)
        # subtract the pressure gradient at interior faces
        self.u[1:-1, :, :] -= dt * np.diff(p, axis=0) * g.inv_h[0]
        self.v[:, 1:-1, :] -= dt * np.diff(p, axis=1) * g.inv_h[1]
        self.w[:, :, 1:-1] -= dt * np.diff(p, axis=2) * g.inv_h[2]
        # open outflow faces: boundary-face pressure is zero
        self.u[-1, :, :] += dt * p[-1, :, :] * g.inv_h[0]
        self.w[:, :, -1] += dt * p[:, :, -1] * g.inv_h[2]
        self._apply_wall_bc()
        self.div_residual = float(np.abs(g.divergence(self.u, self.v, self.w)).max())

    def _advect_scalars(self, dt):
        g = self.g
        uc, vc, wc = g.cell_velocity(self.u, self.v, self.w)
        bx = g.Xc - dt * uc
        by = g.Yc - dt * vc
        bz = g.Zc - dt * wc
        self.d = _sample_trilinear(self.d, bx, by, bz, g.o_c, g.inv_h)
        self.T = _sample_trilinear(self.T, bx, by, bz, g.o_c, g.inv_h)

    def _sources_and_decay(self, dt):
        g = self.g
        # density half-life decay (prescribed removal)
        keep = math.exp(-self.density_decay * dt)
        decayed = float(self.d.sum()) * (1.0 - keep) * g.dV
        self.removed_mass += decayed
        self.d *= keep
        # temperature excess relaxation toward ambient
        self.T *= math.exp(-self.cooling * dt)
        # emitter injection (prescribed source)
        blend = min(1.0, self.emit_rate * dt)
        inj_d = np.clip(self.d_target - self.d, 0.0, None) * blend * self.envelope
        inj_T = np.clip(self.T_target - self.T, 0.0, None) * blend * self.envelope
        self.d += inj_d
        self.T += inj_T
        self.scalar_source = inj_d / dt
        self.added_mass += float(inj_d.sum()) * g.dV * dt
        # emitter inflow velocity (prescribed inflow at the emitter disk)
        iblend = blend * self.envelope
        self.u[1:-1, :, :] += iblend * (self.src_vel[0] - self.u[1:-1, :, :])
        self.v[:, 1:-1, :] += iblend * (self.src_vel[1] - self.v[:, 1:-1, :])
        self.w[:, :, 1:-1] += iblend * (self.src_vel[2] - self.w[:, :, 1:-1])
        # mass leaving through the open outflow boundaries
        d_face_x = 0.5 * (self.d[-1, :, :] + self.d[-2, :, :])
        out_x = (
            float(np.sum(d_face_x * np.clip(self.u[-1, :, :], 0.0, None)))
            * g.h[1] * g.h[2] * dt
        )
        d_face_z = 0.5 * (self.d[:, :, -1] + self.d[:, :, -2])
        out_z = (
            float(np.sum(d_face_z * np.clip(self.w[:, :, -1], 0.0, None)))
            * g.h[0] * g.h[1] * dt
        )
        self.removed_mass += out_x + out_z

    def _step(self, dt):
        self._apply_wall_bc()
        self._apply_forces(dt)
        self._advect_velocity(dt)
        self._apply_wall_bc()
        self._project(dt)
        self._advect_scalars(dt)
        self._sources_and_decay(dt)
        self._apply_wall_bc()
        self.time += dt
        self.steps += 1

    # -- state frames -------------------------------------------------------------

    def advance_to(self, time: float):
        target = float(time)
        eps = 1.0e-9
        guard = 0
        while self.time < target - eps:
            dt = min(self.dt, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 100000:
                raise RuntimeError("advance_to failed to reach target time")
        return self._make_frame(target)

    def _make_frame(self, time: float):
        g = self.g
        uc, vc, wc = g.cell_velocity(self.u, self.v, self.w)
        vel = np.stack([uc, vc, wc], axis=-1)
        self.max_speed = float(np.sqrt(uc * uc + vc * vc + wc * wc).max())
        total_mass = float(self.d.sum()) * g.dV
        wsum = self.d + 1.0e-12
        denom = float(wsum.sum())
        if total_mass > 1.0e-12:
            com = np.array(
                [
                    float((g.Xc * wsum).sum() / denom),
                    float((g.Yc * wsum).sum() / denom),
                    float((g.Zc * wsum).sum() / denom),
                ]
            )
            mean_v = np.array(
                [
                    float((uc * wsum).sum() / denom),
                    float((vc * wsum).sum() / denom),
                    float((wc * wsum).sum() / denom),
                ]
            )
        else:
            com = np.asarray(
                self.cfg["initial_conditions"]["smoke_source"]["center"],
                dtype=np.float64,
            )
            mean_v = np.zeros(3)
        fields = {
            "density": self.d.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "velocity_field": vel.astype(np.float32),
            "transported_scalar": self.d.astype(np.float32),
            "scalar_source": self.scalar_source.astype(np.float32),
            "active_mask": np.ones(g.shape, dtype=np.float32),
            "cumulative_mass_source": np.asarray(
                [self.added_mass - self.removed_mass], dtype=np.float64
            ),
            "temperature": self.T.astype(np.float32),
        }
        scalars = {
            "grid_spacing": float(np.mean(g.h)),
            "characteristic_length": float(g.hi[2] - g.lo[2]),
            "ambient_density": float(
                self.cfg["initial_conditions"].get("ambient_density", 0.0)
            ),
            "reference_velocity": 1.0,
        }
        attributes = {
            "mass_quantity": np.asarray([total_mass], dtype=np.float64),
            "density": np.asarray([float(self.d.mean())], dtype=np.float64),
            "temperature_excess": np.asarray([float(self.T.max())], dtype=np.float64),
        }
        frame = StateFrame(
            time=float(time),
            position=com.reshape(1, 3).copy(),
            velocity=mean_v.reshape(1, 3).copy(),
            mass=np.asarray([max(total_mass, 0.0)], dtype=np.float64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mac_semi_lagrangian_buoyant_smoke",
            "pressure_solver": "separable_eigen_direct",
            "steps": int(self.steps),
            "time": float(self.time),
            "max_divergence_residual": float(self.div_residual),
            "max_speed": float(self.max_speed),
        }
