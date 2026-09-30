"""Physically-based solver for the rising-smoke scene.

The model is an incompressible (constant-density) Navier-Stokes fluid carrying a
transported smoke scalar.  It is stepped on a staggered marker-and-cell (MAC)
grid with a semi-Lagrangian advection scheme, explicit buoyancy forcing driven by
the smoke density, vorticity confinement to preserve the rolled-up vortices, an
exponential density decay (the configured half-life) and a pressure projection
that enforces a divergence-free velocity field.

Everything is implemented from scratch on top of NumPy / SciPy; no external
simulation engine is imported or invoked.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical / numerical constants (frozen for the scene)
# ---------------------------------------------------------------------------
_BUOYANCY_ACCEL = 2.5          # buoyant acceleration per unit smoke density [m/s^2]
_VORTICITY_EPSILON = 0.10      # vorticity confinement strength (Fedkiw et al.)
_SOURCE_RATE = 0.15            # prescribed scalar source rate [1/s]
_DEFAULT_HALF_LIFE = 6.0       # smoke density half life [s]
_CG_RTOL = 2.0e-4              # pressure Poisson relative tolerance
_CG_MAXITER = 80               # pressure Poisson iteration cap
_MAX_SUBSTEPS = 4              # internal sub-stepping cap per output interval
_TARGET_CFL = 0.8              # target Courant number for sub-stepping
_SPEED_CLAMP = 25.0            # safety clamp on velocity magnitude [m/s]
_CHARACTERISTIC_LENGTH = 1.0   # frozen normalization length [m]


class BenchmarkSolver:
    """Smoke solver: buoyant, incompressible, vortex-preserving flow."""

    # ------------------------------------------------------------------
    # construction / initialization
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = getattr(scene, "config", {}) or {}
        initial = cfg.get("initial_conditions", {}) or {}
        source = initial.get("smoke_source", {}) or {}

        domain = cfg["domain"]
        self.nx, self.ny, self.nz = (int(v) for v in domain["grid_shape"])
        nx, ny, nz = self.nx, self.ny, self.nz
        self.low = np.asarray(domain["bounds_min"], dtype=np.float64)
        high = np.asarray(domain["bounds_max"], dtype=np.float64)
        extent = high - self.low
        self.h = float(extent[0] / nx)
        self._h2 = self.h * self.h
        self.cell_volume = self.h ** 3

        # cell-center coordinate axes
        self.xc = self.low[0] + (np.arange(nx, dtype=np.float64) + 0.5) * self.h
        self.yc = self.low[1] + (np.arange(ny, dtype=np.float64) + 0.5) * self.h
        self.zc = self.low[2] + (np.arange(nz, dtype=np.float64) + 0.5) * self.h
        self._X, self._Y, self._Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self._I = np.arange(nx, dtype=np.float64)[:, None, None]
        self._J = np.arange(ny, dtype=np.float64)[None, :, None]
        self._K = np.arange(nz, dtype=np.float64)[None, None, :]

        # ---- frozen initial condition supplied by the scene -------------
        frame0 = scene.initial_state()
        d0 = np.array(frame0.fields["density"], dtype=np.float64, copy=True)
        v0 = np.array(frame0.fields["velocity"], dtype=np.float64, copy=True)
        if d0.shape != (nx, ny, nz):
            d0 = np.ascontiguousarray(d0.reshape(nx, ny, nz).astype(np.float64))
        if v0.shape != (nx, ny, nz, 3):
            v0 = np.ascontiguousarray(v0.reshape(nx, ny, nz, 3).astype(np.float64))
        np.clip(d0, 0.0, None, out=d0)

        self.d = d0
        self._build_faces_from_centers(v0)
        self._apply_boundary_conditions()

        # ---- pressure operator (constant coefficients) ------------------
        self._build_poisson_operator()
        self._p = np.zeros((nx, ny, nz), dtype=np.float64)

        # ---- prescribed source region (static emitter) ------------------
        src_center = np.asarray(source.get("center", [-2.25, 0.0, 0.32]), dtype=np.float64)
        src_radius = float(source.get("radius", 0.38))
        src_half = float(source.get("half_height", 0.24))
        rad2 = ((self._X - src_center[0]) ** 2 + (self._Y - src_center[1]) ** 2) / (src_radius ** 2)
        zt = np.abs(self._Z - src_center[2]) / max(src_half, 1e-9)
        profile = np.clip(1.0 - rad2, 0.0, None) * np.clip(1.0 - zt ** 2, 0.0, None)
        self._source = np.ascontiguousarray((profile * _SOURCE_RATE).astype(np.float32))
        self._active = np.ascontiguousarray(np.ones((nx, ny, nz), dtype=np.float32))
        self._source_mass_rate = float(self._source.sum()) * self.cell_volume

        self._half_life = float(initial.get("density_half_life", _DEFAULT_HALF_LIFE))
        self._decay_scale = math.log(2.0) / max(self._half_life, 1e-9)

        # ---- bookkeeping ------------------------------------------------
        self._t = 0.0
        self._mass_added = 0.0
        self._mass_removed = 0.0
        self._mass_outflow = 0.0
        self._initial_mass = float(self.d.sum() * self.cell_volume)
        uc, vc, wc = self._cell_velocity()
        self._com, self._mean_vel = self._reduced_state(uc, vc, wc)
        self._initialized = True

    # ------------------------------------------------------------------
    # grid helpers
    # ------------------------------------------------------------------
    def _build_faces_from_centers(self, vc: np.ndarray) -> None:
        """Interpolate a cell-centered velocity onto the staggered MAC faces."""
        nx, ny, nz = self.nx, self.ny, self.nz
        u = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        v = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        w = np.zeros((nx, ny, nz + 1), dtype=np.float64)
        u[1:nx] = 0.5 * (vc[:-1, :, :, 0] + vc[1:, :, :, 0])
        u[0] = 0.0
        u[nx] = vc[nx - 1, :, :, 0]
        v[:, 1:ny] = 0.5 * (vc[:, :-1, :, 1] + vc[:, 1:, :, 1])
        v[:, 0] = 0.0
        v[:, ny] = 0.0
        w[:, :, 1:nz] = 0.5 * (vc[:, :, :-1, 2] + vc[:, :, 1:, 2])
        w[:, :, 0] = 0.0
        w[:, :, nz] = vc[:, :, nz - 1, 2]
        self.u, self.v, self.w = u, v, w

    def _cell_velocity(self):
        uc = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        vc = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def _apply_boundary_conditions(self) -> None:
        """Free-slip closed walls and zero-gradient open outflow boundaries."""
        nx, ny, nz = self.nx, self.ny, self.nz
        self.u[0, :, :] = 0.0                     # x_min: closed, free slip
        self.u[nx, :, :] = self.u[nx - 1, :, :]   # x_max: open outflow
        self.v[:, 0, :] = 0.0                     # y_min: closed, free slip
        self.v[:, ny, :] = 0.0                    # y_max: closed, free slip
        self.w[:, :, 0] = 0.0                     # z_min: closed floor
        self.w[:, :, nz] = self.w[:, :, nz - 1]   # z_max: open outflow

    # ------------------------------------------------------------------
    # interpolation
    # ------------------------------------------------------------------
    def _sample(self, f: np.ndarray, xi: np.ndarray, yj: np.ndarray, zk: np.ndarray) -> np.ndarray:
        """Trilinear interpolation on the cell-centered grid (clamped at walls)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        i0 = np.floor(xi).astype(np.int64)
        j0 = np.floor(yj).astype(np.int64)
        k0 = np.floor(zk).astype(np.int64)
        fx = xi - i0
        fy = yj - j0
        fz = zk - k0
        np.clip(i0, 0, nx - 2, out=i0)
        np.clip(j0, 0, ny - 2, out=j0)
        np.clip(k0, 0, nz - 2, out=k0)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        c000 = gx * gy * gz
        c100 = fx * gy * gz
        c010 = gx * fy * gz
        c110 = fx * fy * gz
        c001 = gx * gy * fz
        c101 = fx * gy * fz
        c011 = gx * fy * fz
        c111 = fx * fy * fz
        return (
            f[i0, j0, k0] * c000 + f[i1, j0, k0] * c100
            + f[i0, j1, k0] * c010 + f[i1, j1, k0] * c110
            + f[i0, j0, k1] * c001 + f[i1, j0, k1] * c101
            + f[i0, j1, k1] * c011 + f[i1, j1, k1] * c111
        )

    def _grad_axis(self, f: np.ndarray, axis: int) -> np.ndarray:
        """Second-order central differences with one-sided edges, /h."""
        g = np.empty_like(f)
        if axis == 0:
            g[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) * 0.5
            g[0, :, :] = f[1, :, :] - f[0, :, :]
            g[-1, :, :] = f[-1, :, :] - f[-2, :, :]
        elif axis == 1:
            g[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) * 0.5
            g[:, 0, :] = f[:, 1, :] - f[:, 0, :]
            g[:, -1, :] = f[:, -1, :] - f[:, -2, :]
        else:
            g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) * 0.5
            g[:, :, 0] = f[:, :, 1] - f[:, :, 0]
            g[:, :, -1] = f[:, :, -1] - f[:, :, -2]
        g /= self.h
        return g

    # ------------------------------------------------------------------
    # pressure projection
    # ------------------------------------------------------------------
    def _build_poisson_operator(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        diagx = np.zeros(nx)
        diagx[1:] += 1.0          # neighbour at i-1 (closed / interior)
        diagx[:-1] += 1.0         # neighbour at i+1
        diagx[-1] += 1.0          # Dirichlet p=0 outside the open x_max face
        diagy = np.zeros(ny)
        diagy[1:] += 1.0          # closed free-slip wall (Neumann)
        diagy[:-1] += 1.0         # closed free-slip wall (Neumann)
        diagz = np.zeros(nz)
        diagz[1:] += 1.0
        diagz[:-1] += 1.0
        diagz[-1] += 1.0          # Dirichlet p=0 outside the open z_max face
        self._diag3 = (
            diagx[:, None, None] + diagy[None, :, None] + diagz[None, None, :]
        ) / self._h2
        n = nx * ny * nz
        self._poisson = LinearOperator(
            (n, n), matvec=self._poisson_matvec, dtype=np.float64
        )

    def _poisson_matvec(self, p: np.ndarray) -> np.ndarray:
        out = self._diag3 * p
        out[1:, :, :] -= p[:-1, :, :]
        out[:-1, :, :] -= p[1:, :, :]
        out[:, 1:, :] -= p[:, :-1, :]
        out[:, :-1, :] -= p[:, 1:, :]
        out[:, :, 1:] -= p[:, :, :-1]
        out[:, :, :-1] -= p[:, :, 1:]
        out /= self._h2
        return out

    def _divergence(self):
        u, v, w = self.u, self.v, self.w
        return (
            (u[1:, :, :] - u[:-1, :, :])
            + (v[:, 1:, :] - v[:, :-1, :])
            + (w[:, :, 1:] - w[:, :, :-1])
        ) / self.h

    def _project(self, dt: float) -> None:
        if not (dt > 0.0):
            return
        div = self._divergence()
        rhs = (-div / dt).ravel()
        x0 = self._p.ravel()
        try:
            p, _info = cg(self._poisson, rhs, x0=x0, rtol=_CG_RTOL, maxiter=_CG_MAXITER)
        except Exception:
            p = np.array(x0, copy=True)
        if not np.isfinite(p).all():
            p = np.zeros_like(rhs)
        P = p.reshape(self.nx, self.ny, self.nz)
        nx, ny, nz, h = self.nx, self.ny, self.nz, self.h
        self.u[1:nx, :, :] -= dt * (P[1:, :, :] - P[:-1, :, :]) / h
        self.v[:, 1:ny, :] -= dt * (P[:, 1:, :] - P[:, :-1, :]) / h
        self.w[:, :, 1:nz] -= dt * (P[:, :, 1:] - P[:, :, :-1]) / h
        self._p = P
        self._apply_boundary_conditions()

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _advect_velocity(self, dt: float) -> None:
        uc, vc, wc = self._cell_velocity()
        xi = self._I - dt * uc / self.h
        yj = self._J - dt * vc / self.h
        zk = self._K - dt * wc / self.h
        nuc = self._sample(uc, xi, yj, zk)
        nvc = self._sample(vc, xi, yj, zk)
        nwc = self._sample(wc, xi, yj, zk)
        nx, ny, nz = self.nx, self.ny, self.nz
        u = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        v = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        w = np.zeros((nx, ny, nz + 1), dtype=np.float64)
        u[1:nx] = 0.5 * (nuc[:-1] + nuc[1:])
        u[0] = 0.0
        u[nx] = nuc[nx - 1]
        v[:, 1:ny] = 0.5 * (nvc[:, :-1] + nvc[:, 1:])
        v[:, 0] = 0.0
        v[:, ny] = 0.0
        w[:, :, 1:nz] = 0.5 * (nwc[:, :, :-1] + nwc[:, :, 1:])
        w[:, :, 0] = 0.0
        w[:, :, nz] = nwc[:, :, nz - 1]
        self.u, self.v, self.w = u, v, w

    def _apply_forces(self, dt: float) -> None:
        """Buoyancy (only external force) plus vorticity confinement."""
        uc, vc, wc = self._cell_velocity()
        # vorticity of the cell-centered velocity
        dwdy = self._grad_axis(wc, 1)
        dvdz = self._grad_axis(vc, 2)
        dudz = self._grad_axis(uc, 2)
        dwdx = self._grad_axis(wc, 0)
        dvdx = self._grad_axis(vc, 0)
        dudy = self._grad_axis(uc, 1)
        wx = dwdy - dvdz
        wy = dwdx - dudz
        wz = dvdx - dudy
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx = self._grad_axis(mag, 0)
        gy = self._grad_axis(mag, 1)
        gz = self._grad_axis(mag, 2)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1.0e-12
        gx /= norm
        gy /= norm
        gz /= norm
        eps_h = _VORTICITY_EPSILON * self.h
        fx = eps_h * (gy * wz - gz * wy)
        fy = eps_h * (gz * wx - gx * wz)
        fz = eps_h * (gx * wy - gy * wx) + _BUOYANCY_ACCEL * self.d

        nx, ny, nz = self.nx, self.ny, self.nz
        self.u[1:nx, :, :] += dt * 0.5 * (fx[:-1, :, :] + fx[1:, :, :])
        self.v[:, 1:ny, :] += dt * 0.5 * (fy[:, :-1, :] + fy[:, 1:, :])
        self.w[:, :, 1:nz] += dt * 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])

    def _advect_scalar(self, dt: float) -> None:
        uc, vc, wc = self._cell_velocity()
        xi = self._I - dt * uc / self.h
        yj = self._J - dt * vc / self.h
        zk = self._K - dt * wc / self.h
        d_new = self._sample(self.d, xi, yj, zk)
        np.clip(d_new, 0.0, None, out=d_new)
        self.d = d_new

        mass_before = float(self.d.sum() * self.cell_volume)
        decay = 1.0 - math.exp(-self._decay_scale * dt)
        if decay > 0.0:
            self.d *= (1.0 - decay)
            self._mass_removed += mass_before * decay
        if _SOURCE_RATE > 0.0:
            added = self._source_mass_rate * dt
            self.d += dt * self._source
            self._mass_added += added

    def _sanitize(self) -> None:
        for arr in (self.d, self.u, self.v, self.w):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.d, 0.0, None, out=self.d)
        np.clip(self.u, -_SPEED_CLAMP, _SPEED_CLAMP, out=self.u)
        np.clip(self.v, -_SPEED_CLAMP, _SPEED_CLAMP, out=self.v)
        np.clip(self.w, -_SPEED_CLAMP, _SPEED_CLAMP, out=self.w)
        self._apply_boundary_conditions()

    def _step(self, dt: float) -> None:
        self._advect_velocity(dt)
        self._apply_forces(dt)
        self._apply_boundary_conditions()
        self._project(dt)
        self._advect_scalar(dt)
        self._sanitize()

    def advance_to(self, time: float):
        target = float(time)
        if target < self._t:
            target = self._t
        remaining = target - self._t
        if remaining > 1.0e-12:
            speed = max(
                float(np.abs(self.u).max()),
                float(np.abs(self.v).max()),
                float(np.abs(self.w).max()),
                1.0e-6,
            )
            cfl_dt = _TARGET_CFL * self.h / speed
            nsub = int(math.ceil(remaining / cfl_dt - 1.0e-9))
            nsub = max(1, min(nsub, _MAX_SUBSTEPS))
            dt = remaining / nsub
            for _ in range(nsub):
                self._step(dt)
            self._t = target
        return self._frame(target)

    # ------------------------------------------------------------------
    # state reduction and frame export
    # ------------------------------------------------------------------
    def _reduced_state(self, uc, vc, wc):
        d = self.d
        total = float(d.sum())
        if total > 1.0e-15:
            cx = float((d * self._X).sum() / total)
            cy = float((d * self._Y).sum() / total)
            cz = float((d * self._Z).sum() / total)
            vx = float((d * uc).sum() / total)
            vy = float((d * vc).sum() / total)
            vz = float((d * wc).sum() / total)
        else:
            cx, cy, cz = (float(v) for v in self._com)
            vx, vy, vz = (float(v) for v in self._mean_vel)
        return np.array([cx, cy, cz], dtype=np.float64), np.array([vx, vy, vz], dtype=np.float64)

    def _frame(self, time: float) -> StateFrame:
        uc, vc, wc = self._cell_velocity()
        com, mean_vel = self._reduced_state(uc, vc, wc)
        self._com, self._mean_vel = com, mean_vel
        mass = float(self.d.sum() * self.cell_volume)

        d32 = np.ascontiguousarray(self.d.astype(np.float32))
        vel32 = np.ascontiguousarray(np.stack((uc, vc, wc), axis=-1).astype(np.float32))
        cumulative = np.array([self._mass_added - self._mass_removed], dtype=np.float64)

        fields = {
            "density": d32,
            "transported_scalar": d32,
            "velocity": vel32,
            "velocity_field": vel32,
            "scalar_source": self._source,
            "active_mask": self._active,
            "cumulative_mass_source": cumulative,
        }
        frame = StateFrame(
            float(time),
            com.reshape(1, 3).copy(),
            mean_vel.reshape(1, 3).copy(),
            mass=np.asarray([mass], dtype=np.float64),
            attributes={"mass_quantity": np.asarray([mass], dtype=np.float64)},
            fields=fields,
            scalars={
                "characteristic_length": float(_CHARACTERISTIC_LENGTH),
                "grid_spacing": float(self.h),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        uc, vc, wc = self._cell_velocity()
        speed = np.sqrt(uc * uc + vc * vc + wc * wc)
        div = self._divergence()
        return {
            "time": float(self._t),
            "total_mass": float(self.d.sum() * self.cell_volume),
            "mass_added": float(self._mass_added),
            "mass_removed": float(self._mass_removed),
            "max_speed": float(speed.max()) if speed.size else 0.0,
            "max_divergence": float(np.abs(div).max()) if div.size else 0.0,
            "grid_spacing": float(self.h),
        }
