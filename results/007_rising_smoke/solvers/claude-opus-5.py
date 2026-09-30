"""Buoyant smoke solver: incompressible flow on a MAC grid.

Physical model (Fedkiw/Stam "Visual Simulation of Smoke" family, implemented
from scratch with NumPy/SciPy only):

    d_t u + (u . grad) u = -grad p + f_buoy + f_conf + nu lap u
    div u = 0
    d_t d + div(u d) = -k_d d          (smoke density, prescribed decay)
    d_t T + div(u T) = -k_T T          (temperature excess, cooling)
    f_buoy = (beta * T - alpha * d) * z_hat

Discretization
    * staggered (MAC) velocity, cell-centered scalars, uniform spacing h
    * semi-Lagrangian advection of velocity (unconditionally stable)
    * conservative MUSCL/minmod flux advection of scalars, so the only mass
      changes are the prescribed decay sink and the open-boundary flux
    * exact pressure projection by eigen-decomposition of the separable
      discrete Poisson operator (Neumann at closed free-slip walls, p = 0 at
      open outflow walls); nonsingular, direct, deterministic
    * explicit vorticity confinement to keep the plume vortices persistent
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from worldsolver.contracts import StateFrame

def _sample_trilinear(field: np.ndarray, ix: np.ndarray, iy: np.ndarray,
                      iz: np.ndarray) -> np.ndarray:
    """Trilinear gather of ``field`` at continuous index coordinates (clamped)."""
    nx, ny, nz = field.shape
    fx = np.clip(ix, 0.0, nx - 1.0)
    fy = np.clip(iy, 0.0, ny - 1.0)
    fz = np.clip(iz, 0.0, nz - 1.0)
    i0 = np.floor(fx).astype(np.intp)
    j0 = np.floor(fy).astype(np.intp)
    k0 = np.floor(fz).astype(np.intp)
    np.minimum(i0, nx - 2 if nx > 1 else 0, out=i0)
    np.minimum(j0, ny - 2 if ny > 1 else 0, out=j0)
    np.minimum(k0, nz - 2 if nz > 1 else 0, out=k0)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)
    tx = fx - i0
    ty = fy - j0
    tz = fz - k0
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 + (c100 - c000) * tx
    c10 = c010 + (c110 - c010) * tx
    c01 = c001 + (c101 - c001) * tx
    c11 = c011 + (c111 - c011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod slope limiter (component-wise)."""
    return np.where(a * b <= 0.0, 0.0,
                    np.where(np.abs(a) < np.abs(b), a, b))


def _apply_axis_matrix(data: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Contract ``matrix`` with ``data`` along ``axis`` (matrix @ data_axis)."""
    moved = np.moveaxis(data, axis, -1)
    result = moved @ matrix.T
    return np.moveaxis(result, -1, axis)


def _laplace_1d(n: int, h: float, low_open: bool, high_open: bool):
    """Symmetric tridiagonal 1-D Laplacian.

    Closed (free-slip) wall  -> homogeneous Neumann, dp/dn = 0.
    Open outflow wall        -> Dirichlet p = 0 on the wall via a mirrored
                                ghost value p_ghost = -p_interior.
    """
    diag = np.full(n, -2.0, dtype=np.float64)
    off = np.ones(max(n - 1, 0), dtype=np.float64)
    if n == 1:
        diag[0] = 0.0
        if low_open:
            diag[0] -= 2.0
        if high_open:
            diag[0] -= 2.0
    else:
        diag[0] = -3.0 if low_open else -1.0
        diag[-1] = -3.0 if high_open else -1.0
    return diag / (h * h), off / (h * h)


class BenchmarkSolver:
    """Physically stepped rising-smoke solver."""

    # buoyancy / forcing coefficients (SI-consistent, tuned for this domain)
    ALPHA_DENSITY = 0.40      # smoke weight, m/s^2 per unit density
    BETA_TEMPERATURE = 1.00   # thermal buoyancy, m/s^2 per unit temperature
    CONFINEMENT_EPS = 1.6     # vorticity confinement strength
    VISCOSITY = 5.0e-5        # small physical viscosity, m^2/s
    TEMPERATURE_HALF_LIFE = 3.0
    # Semi-Lagrangian momentum advection is stable beyond CFL 1; the
    # conservative scalar sweep subdivides itself to stay under CFL 0.5.
    CFL_TARGET = 0.9
    MAX_SUBSTEPS = 4

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        domain = config["domain"]
        initial = config["initial_conditions"]
        source = initial["smoke_source"]

        self.shape = tuple(int(v) for v in domain["grid_shape"])
        nx, ny, nz = self.shape
        self.low = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.high = np.asarray(domain["bounds_max"], dtype=np.float64)
        extent = self.high - self.low
        spacing = extent / np.asarray(self.shape, dtype=np.float64)
        self.spacing_vec = spacing
        self.h = float(np.mean(spacing))
        self.cell_volume = float(spacing[0] * spacing[1] * spacing[2])

        bc = domain.get("boundary_conditions", {}) or {}

        def _is_open(key: str) -> bool:
            return "open" in str(bc.get(key, "closed_free_slip")).lower()

        self.open_x_min = _is_open("x_min")
        self.open_x_max = _is_open("x_max")
        self.open_y_min = _is_open("y_min")
        self.open_y_max = _is_open("y_max")
        self.open_z_min = _is_open("z_min")
        self.open_z_max = _is_open("z_max")

        forcing = config.get("forcing", {}) or {}
        buoy_dir = np.asarray(forcing.get("buoyancy_direction", [0.0, 0.0, 1.0]),
                              dtype=np.float64)
        norm = float(np.linalg.norm(buoy_dir))
        self.buoyancy_sign = float(np.sign(buoy_dir[2])) if norm > 0 else 1.0
        if self.buoyancy_sign == 0.0:
            self.buoyancy_sign = 1.0

        half_life = float(initial.get("density_half_life", 0.0) or 0.0)
        self.density_decay_rate = np.log(2.0) / half_life if half_life > 0 else 0.0
        self.temperature_decay_rate = np.log(2.0) / float(self.TEMPERATURE_HALF_LIFE)

        # ---- initial fields taken verbatim from the frozen scene ----
        frame0 = scene.initial_state()
        density = np.array(frame0.fields["density"], dtype=np.float64).reshape(self.shape)
        velocity_cc = np.array(frame0.fields["velocity"], dtype=np.float64).reshape(
            self.shape + (3,))
        self.density = np.ascontiguousarray(density)
        peak = float(source["peak_density"])
        envelope = np.clip(self.density / peak, 0.0, 1.0) if peak > 0 else np.zeros(self.shape)
        self.temperature = np.ascontiguousarray(
            float(source["temperature_excess"]) * envelope)

        # cell-centered -> MAC faces (interior faces only; walls handled by BC)
        self.u = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        self.v = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        self.w = np.zeros((nx, ny, nz + 1), dtype=np.float64)
        self.u[1:nx, :, :] = 0.5 * (velocity_cc[:-1, :, :, 0] + velocity_cc[1:, :, :, 0])
        self.v[:, 1:ny, :] = 0.5 * (velocity_cc[:, :-1, :, 1] + velocity_cc[:, 1:, :, 1])
        self.w[:, :, 1:nz] = 0.5 * (velocity_cc[:, :, :-1, 2] + velocity_cc[:, :, 1:, 2])
        if self.open_x_min:
            self.u[0, :, :] = velocity_cc[0, :, :, 0]
        if self.open_x_max:
            self.u[nx, :, :] = velocity_cc[-1, :, :, 0]
        if self.open_y_min:
            self.v[:, 0, :] = velocity_cc[:, 0, :, 1]
        if self.open_y_max:
            self.v[:, ny, :] = velocity_cc[:, -1, :, 1]
        if self.open_z_min:
            self.w[:, :, 0] = velocity_cc[:, :, 0, 2]
        if self.open_z_max:
            self.w[:, :, nz] = velocity_cc[:, :, -1, 2]

        self._build_poisson()
        self._apply_wall_bc()
        self.project()

        # ---- bookkeeping ----
        self.time = 0.0
        self.mass0 = float(self.density.sum()) * self.cell_volume
        self.cumulative_source = 0.0     # added minus removed since frame 0
        self.step_count = 0
        self.max_speed_seen = 0.0
        self.scalar_source = np.zeros(self.shape, dtype=np.float64)
        self.active_mask = np.ones(self.shape, dtype=np.float64)
        self.source_center = np.asarray(source["center"], dtype=np.float64)
        self.characteristic_length = 2.0 * float(source["radius"])
        self.ambient_density = float(initial.get("ambient_density", 0.0) or 0.0)

        # index-space coordinates of the MAC face centers (precomputed once)
        ax = np.arange(nx + 1, dtype=np.float64)
        ay = np.arange(ny + 1, dtype=np.float64)
        az = np.arange(nz + 1, dtype=np.float64)
        cx = np.arange(nx, dtype=np.float64) + 0.5
        cy = np.arange(ny, dtype=np.float64) + 0.5
        cz = np.arange(nz, dtype=np.float64) + 0.5
        self._u_idx = np.meshgrid(ax, cy, cz, indexing="ij")
        self._v_idx = np.meshgrid(cx, ay, cz, indexing="ij")
        self._w_idx = np.meshgrid(cx, cy, az, indexing="ij")

    # --------------------------------------------------------------- Poisson
    def _build_poisson(self) -> None:
        nx, ny, nz = self.shape
        hx, hy, hz = (float(s) for s in self.spacing_vec)
        dx, ox = _laplace_1d(nx, hx, self.open_x_min, self.open_x_max)
        dy, oy = _laplace_1d(ny, hy, self.open_y_min, self.open_y_max)
        dz, oz = _laplace_1d(nz, hz, self.open_z_min, self.open_z_max)
        self._lx, self._qx = eigh_tridiagonal(dx, ox)
        self._ly, self._qy = eigh_tridiagonal(dy, oy)
        self._lz, self._qz = eigh_tridiagonal(dz, oz)
        denom = (self._lx[:, None, None] + self._ly[None, :, None]
                 + self._lz[None, None, :])
        scale = max(abs(float(denom.min())), 1.0)
        tiny = 1.0e-12 * scale
        self._eig_denom = np.where(np.abs(denom) < tiny, np.inf, denom)
        self._qx_t = np.ascontiguousarray(self._qx.T)
        self._qy_t = np.ascontiguousarray(self._qy.T)
        self._qz_t = np.ascontiguousarray(self._qz.T)

    def _solve_poisson(self, rhs: np.ndarray) -> np.ndarray:
        """Direct spectral solve of lap(p) = rhs with the frozen wall types."""
        spectral = _apply_axis_matrix(rhs, self._qx_t, 0)
        spectral = _apply_axis_matrix(spectral, self._qy_t, 1)
        spectral = _apply_axis_matrix(spectral, self._qz_t, 2)
        spectral /= self._eig_denom
        out = _apply_axis_matrix(spectral, self._qx, 0)
        out = _apply_axis_matrix(out, self._qy, 1)
        return _apply_axis_matrix(out, self._qz, 2)

    def _apply_wall_bc(self) -> None:
        """No-penetration at closed free-slip walls (tangential slip is free)."""
        nx, ny, nz = self.shape
        if not self.open_x_min:
            self.u[0, :, :] = 0.0
        if not self.open_x_max:
            self.u[nx, :, :] = 0.0
        if not self.open_y_min:
            self.v[:, 0, :] = 0.0
        if not self.open_y_max:
            self.v[:, ny, :] = 0.0
        if not self.open_z_min:
            self.w[:, :, 0] = 0.0
        if not self.open_z_max:
            self.w[:, :, nz] = 0.0

    def divergence(self) -> np.ndarray:
        hx, hy, hz = (float(s) for s in self.spacing_vec)
        return ((self.u[1:, :, :] - self.u[:-1, :, :]) / hx
                + (self.v[:, 1:, :] - self.v[:, :-1, :]) / hy
                + (self.w[:, :, 1:] - self.w[:, :, :-1]) / hz)

    def project(self) -> None:
        """Make the MAC velocity discretely divergence free."""
        nx, ny, nz = self.shape
        hx, hy, hz = (float(s) for s in self.spacing_vec)
        p = self._solve_poisson(self.divergence())
        self.u[1:nx, :, :] -= (p[1:, :, :] - p[:-1, :, :]) / hx
        self.v[:, 1:ny, :] -= (p[:, 1:, :] - p[:, :-1, :]) / hy
        self.w[:, :, 1:nz] -= (p[:, :, 1:] - p[:, :, :-1]) / hz
        if self.open_x_min:
            self.u[0, :, :] -= 2.0 * p[0, :, :] / hx
        if self.open_x_max:
            self.u[nx, :, :] += 2.0 * p[-1, :, :] / hx
        if self.open_y_min:
            self.v[:, 0, :] -= 2.0 * p[:, 0, :] / hy
        if self.open_y_max:
            self.v[:, ny, :] += 2.0 * p[:, -1, :] / hy
        if self.open_z_min:
            self.w[:, :, 0] -= 2.0 * p[:, :, 0] / hz
        if self.open_z_max:
            self.w[:, :, nz] += 2.0 * p[:, :, -1] / hz
        self._apply_wall_bc()
        self.pressure = p

    # ------------------------------------------------------- MAC sampling
    def cell_velocity(self) -> np.ndarray:
        """Cell-centered velocity (average of opposite MAC faces)."""
        out = np.empty(self.shape + (3,), dtype=np.float64)
        out[..., 0] = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        out[..., 1] = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        out[..., 2] = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return out

    def _sample_u(self, gx, gy, gz):
        return _sample_trilinear(self.u, gx, gy - 0.5, gz - 0.5)

    def _sample_v(self, gx, gy, gz):
        return _sample_trilinear(self.v, gx - 0.5, gy, gz - 0.5)

    def _sample_w(self, gx, gy, gz):
        return _sample_trilinear(self.w, gx - 0.5, gy - 0.5, gz)

    def _advect_velocity(self, dt: float) -> None:
        """First-order semi-Lagrangian advection of the MAC velocity."""
        hx, hy, hz = (float(s) for s in self.spacing_vec)
        u_old, v_old, w_old = self.u, self.v, self.w

        gx, gy, gz = self._u_idx
        bx = gx - dt * u_old / hx
        by = gy - dt * self._sample_v(gx, gy, gz) / hy
        bz = gz - dt * self._sample_w(gx, gy, gz) / hz
        u_new = _sample_trilinear(u_old, bx, by - 0.5, bz - 0.5)

        gx, gy, gz = self._v_idx
        bx = gx - dt * self._sample_u(gx, gy, gz) / hx
        by = gy - dt * v_old / hy
        bz = gz - dt * self._sample_w(gx, gy, gz) / hz
        v_new = _sample_trilinear(v_old, bx - 0.5, by, bz - 0.5)

        gx, gy, gz = self._w_idx
        bx = gx - dt * self._sample_u(gx, gy, gz) / hx
        by = gy - dt * self._sample_v(gx, gy, gz) / hy
        bz = gz - dt * w_old / hz
        w_new = _sample_trilinear(w_old, bx - 0.5, by - 0.5, bz)

        self.u, self.v, self.w = u_new, v_new, w_new
        self._apply_wall_bc()

    # -------------------------------------------------- conservative scalars
    def _flux_axis(self, field: np.ndarray, face_velocity: np.ndarray,
                   axis: int, dt: float):
        """One conservative MUSCL/minmod flux sweep along ``axis``.

        Returns the updated field and the net mass (per unit density-volume)
        that left the domain through the two boundary faces of this axis.
        """
        f = np.moveaxis(field, axis, 0)
        vel = np.moveaxis(face_velocity, axis, 0)
        h = float(self.spacing_vec[axis])
        ambient = self.ambient_density

        pad = np.empty((f.shape[0] + 2,) + f.shape[1:], dtype=np.float64)
        pad[1:-1] = f
        pad[0] = ambient
        pad[-1] = ambient
        slope = np.zeros_like(pad)
        slope[1:-1] = _minmod(pad[1:-1] - pad[:-2], pad[2:] - pad[1:-1])

        left = pad[:-1] + 0.5 * slope[:-1]     # state on the low side of face
        right = pad[1:] - 0.5 * slope[1:]      # state on the high side
        upwind = np.where(vel >= 0.0, left, right)
        flux = vel * upwind

        new = f - (dt / h) * (flux[1:] - flux[:-1])
        outflow = float(flux[-1].sum() - flux[0].sum())
        area = self.cell_volume / h
        boundary_mass = outflow * dt * area
        return np.moveaxis(new, 0, axis), boundary_mass

    def _advect_scalars(self, dt: float) -> float:
        """Advect density and temperature in flux form; return mass lost."""
        removed = 0.0
        for axis, vel in ((0, self.u), (1, self.v), (2, self.w)):
            self.density, lost = self._flux_axis(self.density, vel, axis, dt)
            removed += lost
            self.temperature, _ = self._flux_axis(self.temperature, vel, axis, dt)
        negative = float(np.minimum(self.density, 0.0).sum()) * self.cell_volume
        if negative < 0.0:
            np.maximum(self.density, 0.0, out=self.density)
            removed += negative
        np.maximum(self.temperature, 0.0, out=self.temperature)
        return removed

    # ---------------------------------------------------------------- forces
    def _apply_buoyancy(self, dt: float) -> None:
        """f = sign * (beta * T - alpha * d) on the vertical MAC faces."""
        nz = self.shape[2]
        force_cc = self.buoyancy_sign * (self.BETA_TEMPERATURE * self.temperature
                                         - self.ALPHA_DENSITY * self.density)
        self.w[:, :, 1:nz] += dt * 0.5 * (force_cc[:, :, :-1] + force_cc[:, :, 1:])
        if self.open_z_min:
            self.w[:, :, 0] += dt * force_cc[:, :, 0]
        if self.open_z_max:
            self.w[:, :, nz] += dt * force_cc[:, :, -1]

    def _apply_confinement(self, dt: float) -> None:
        """Vorticity confinement: restores vorticity lost to numerical damping."""
        eps = float(self.CONFINEMENT_EPS)
        if eps <= 0.0:
            return
        nx, ny, nz = self.shape
        hx, hy, hz = (float(s) for s in self.spacing_vec)
        vel = self.cell_velocity()
        du_dy, du_dz = np.gradient(vel[..., 0], hy, hz, axis=(1, 2))
        dv_dx, dv_dz = np.gradient(vel[..., 1], hx, hz, axis=(0, 2))
        dw_dx, dw_dy = np.gradient(vel[..., 2], hx, hy, axis=(0, 1))
        wx = dw_dy - dv_dz
        wy = du_dz - dw_dx
        wz = dv_dx - du_dy
        magnitude = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx, gy, gz = np.gradient(magnitude, hx, hy, hz)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz)
        inv = np.where(norm > 1.0e-12, 1.0 / np.maximum(norm, 1.0e-30), 0.0)
        nxv, nyv, nzv = gx * inv, gy * inv, gz * inv
        scale = eps * self.h
        fx = scale * (nyv * wz - nzv * wy)
        fy = scale * (nzv * wx - nxv * wz)
        fz = scale * (nxv * wy - nyv * wx)
        self.u[1:nx, :, :] += dt * 0.5 * (fx[:-1, :, :] + fx[1:, :, :])
        self.v[:, 1:ny, :] += dt * 0.5 * (fy[:, :-1, :] + fy[:, 1:, :])
        self.w[:, :, 1:nz] += dt * 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])

    def _apply_viscosity(self, dt: float) -> None:
        nu = float(self.VISCOSITY)
        if nu <= 0.0:
            return
        coefficient = nu * dt / (self.h * self.h)
        for name in ("u", "v", "w"):
            field = getattr(self, name)
            pad = np.pad(field, 1, mode="edge")
            laplacian = (pad[2:, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1]
                         + pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1]
                         + pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2]
                         - 6.0 * field)
            setattr(self, name, field + coefficient * laplacian)

    # ------------------------------------------------------------- time step
    def _max_face_speed(self) -> float:
        speeds = (float(np.abs(self.u).max()) if self.u.size else 0.0,
                  float(np.abs(self.v).max()) if self.v.size else 0.0,
                  float(np.abs(self.w).max()) if self.w.size else 0.0)
        return max(speeds)

    def _stable_dt(self, remaining: float) -> float:
        speed = self._max_face_speed()
        self.max_speed_seen = max(self.max_speed_seen, speed)
        limit = remaining
        if speed > 1.0e-9:
            limit = min(limit, self.CFL_TARGET * self.h / speed)
        floor = remaining / float(self.MAX_SUBSTEPS)
        return float(max(min(remaining, limit), min(floor, remaining)))

    def _step(self, dt: float) -> None:
        """One physical time step of the buoyant, incompressible smoke system."""
        # transport of the conserved scalars with the divergence-free velocity
        speed = self._max_face_speed()
        sub = int(np.ceil(speed * dt / (0.5 * self.h))) if speed > 0.0 else 1
        sub = int(min(max(sub, 1), 16))
        for _ in range(sub):
            self.cumulative_source -= self._advect_scalars(dt / sub)

        # prescribed decay (density loss) and thermal cooling
        if self.density_decay_rate > 0.0:
            factor = float(np.exp(-self.density_decay_rate * dt))
            before = float(self.density.sum())
            self.density *= factor
            self.cumulative_source -= (before - float(self.density.sum())) * self.cell_volume
        if self.temperature_decay_rate > 0.0:
            self.temperature *= float(np.exp(-self.temperature_decay_rate * dt))

        # momentum: advection, diffusion, buoyancy, confinement, projection
        self._advect_velocity(dt)
        self._apply_viscosity(dt)
        self._apply_buoyancy(dt)
        self._apply_confinement(dt)
        self._apply_wall_bc()
        self.project()

        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while target - self.time > 1.0e-12 and guard < 100000:
            dt = self._stable_dt(target - self.time)
            if dt <= 0.0:
                break
            self._step(dt)
            guard += 1
        self.time = target
        return self._make_frame(target)

    # ---------------------------------------------------------- frame export
    def _cell_axes(self):
        cached = getattr(self, "_axes_cache", None)
        if cached is None:
            cached = tuple(
                self.low[i] + (np.arange(self.shape[i], dtype=np.float64) + 0.5)
                * float(self.spacing_vec[i]) for i in range(3))
            self._axes_cache = cached
        return cached

    def _make_frame(self, time: float) -> StateFrame:
        density = np.nan_to_num(self.density, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(density, 0.0, out=density)
        temperature = np.nan_to_num(self.temperature, nan=0.0, posinf=0.0, neginf=0.0)
        velocity_cc = np.nan_to_num(self.cell_velocity(), nan=0.0,
                                    posinf=0.0, neginf=0.0)

        mass_total = float(density.sum()) * self.cell_volume
        ax, ay, az = self._cell_axes()
        if mass_total > 1.0e-12:
            weight = density
            total = float(weight.sum())
            cx = float((weight.sum(axis=(1, 2)) * ax).sum() / total)
            cy = float((weight.sum(axis=(0, 2)) * ay).sum() / total)
            cz = float((weight.sum(axis=(0, 1)) * az).sum() / total)
            centroid = np.array([cx, cy, cz], dtype=np.float64)
            mean_velocity = np.array(
                [float((weight * velocity_cc[..., c]).sum() / total) for c in range(3)],
                dtype=np.float64)
        else:
            centroid = self.source_center.astype(np.float64).copy()
            mean_velocity = np.zeros(3, dtype=np.float64)

        scalar_source = (-self.density_decay_rate * density).astype(np.float32)
        density32 = density.astype(np.float32)

        frame = StateFrame(
            time=float(time),
            position=centroid.reshape(1, 3),
            velocity=mean_velocity.reshape(1, 3),
            mass=np.asarray([mass_total], dtype=np.float64),
            attributes={
                "mass_quantity": np.asarray([mass_total], dtype=np.float64),
                "density": np.asarray([float(density.max())], dtype=np.float64),
                "temperature_excess": np.asarray([float(temperature.max())],
                                                 dtype=np.float64),
            },
            fields={
                "density": density32,
                "temperature": temperature.astype(np.float32),
                "transported_scalar": density32,
                "scalar_source": scalar_source,
                "active_mask": self.active_mask.astype(np.float32),
                "velocity": velocity_cc.astype(np.float32),
                "velocity_field": velocity_cc.astype(np.float32),
                "cumulative_mass_source": np.asarray([self.cumulative_source],
                                                     dtype=np.float64),
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.h),
                "ambient_density": float(self.ambient_density),
                "total_mass": mass_total,
                "initial_mass": float(self.mass0),
                "max_divergence": float(np.abs(self.divergence()).max()),
                "max_speed": float(np.abs(velocity_cc).max()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC staggered incompressible solver, semi-Lagrangian momentum "
                      "advection, conservative MUSCL scalar transport, spectral "
                      "pressure projection, vorticity confinement",
            "grid_shape": list(self.shape),
            "grid_spacing": float(self.h),
            "steps": int(self.step_count),
            "final_time": float(self.time),
            "initial_mass": float(self.mass0),
            "final_mass": float(self.density.sum()) * self.cell_volume,
            "cumulative_mass_source": float(self.cumulative_source),
            "max_speed_seen": float(self.max_speed_seen),
            "buoyancy_alpha": float(self.ALPHA_DENSITY),
            "buoyancy_beta": float(self.BETA_TEMPERATURE),
            "confinement_epsilon": float(self.CONFINEMENT_EPS),
            "viscosity": float(self.VISCOSITY),
        }
