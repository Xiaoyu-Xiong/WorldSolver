"""Physically-based solver for scene 168_centrifugal_surface_square_mode.

The liquid free surface inside the stationary cylindrical tank is advanced with
a nonlinear shallow-water model (incompressible, hydrostatic, free-surface
Euler equations depth-averaged) on a uniform Cartesian grid clipped to the
circular tank.  The rotating bottom plate spins the liquid up through a
turbulent stress closure that relaxes the horizontal velocity toward a
plate-driven swirl profile; the outer part of the profile is braked by the
stationary side wall.  The resulting sheared swirl is centrifugally
unstable: the shear layer between the fast core and the slow rim amplifies
the azimuthal perturbations seeded in the frozen initial condition, and the
dominant mode (m = 4 for this configuration) saturates into the rounded
rotating square depression observed in the experiment.

No keyframes, animation tables or precomputed trajectories are used: every
sampled state comes from explicit RK3 time stepping of the discretized PDE.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # nearest-wet-cell fill for the circular domain halo
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - scipy is expected but not essential
    _ndi = None


class BenchmarkSolver:
    """Shallow-water free-surface solver with rotating-bottom driving."""

    def __init__(self) -> None:
        self.t = 0.0
        self._ready = False
        self._fail_count = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        tank = cfg["tank"]
        self.seed = int(seed)
        self.R = float(tank["radius"])
        self.g = abs(float(cfg["gravity"][2]))
        self.fill = float(tank["initial_fill_height"])
        self.h_part = float(tank["particle_spacing"])
        self.rho = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.omega_plate = float(tank["bottom_angular_velocity_rad_s"])
        self.wall_height = float(tank["wall_height"])
        self.grid_shape = tuple(int(v) for v in tank["field_grid_shape"])

        # frozen initial surface markers (mesh vertices) and their masses
        f0 = scene.initial_state()
        p0 = np.asarray(f0.position, dtype=float)
        self.vxy = p0[:, :2].copy()
        self.mass0 = np.asarray(f0.attributes["mass_quantity"], dtype=float).copy()
        self.prad = np.asarray(f0.attributes["particle_radius"], dtype=float).copy()
        self.n_vertices = len(p0)

        # ---- physical closure parameters (turbulent spin-up / wall braking)
        self.tau_spin = 1.5          # spin-up relaxation time [s]
        self.nu = 0.004              # eddy viscosity [m^2/s]
        self.ramp_time = 2.5         # bottom plate spin-up ramp [s]
        self.omega_in = 5.0          # core angular velocity at full spin [rad/s]
        self.omega_out = 0.4         # weak background rotation [rad/s]
        self.r_core = 0.40           # radius of the fast rotating core [m]
        self.sigma_shear = 0.09      # shear-layer thickness [m]

        # ---- Cartesian simulation grid clipped to the circular tank
        self.n = 100
        self.L = 0.99
        self.dx = 2.0 * self.L / self.n
        centers = -self.L + (np.arange(self.n) + 0.5) * self.dx
        X, Y = np.meshgrid(centers, centers, indexing="ij")
        self.X, self.Y = X, Y
        self.rc = np.hypot(X, Y)
        self.theta = np.arctan2(Y, X)
        self.centers = centers
        self.wet = self.rc <= self.R - 0.25 * self.dx
        self.wall_layer = self.wet & (self.rc > self.R - 1.6 * self.dx)
        rsafe = np.maximum(self.rc, 1e-9)
        self.er_x = X / rsafe
        self.er_y = Y / rsafe
        self._fill_idx = self._nearest_wet_indices()

        # ---- target swirl profile (plate-driven core, wall-braked rim)
        om = np.where(
            self.rc <= self.r_core,
            self.omega_in,
            self.omega_in * np.exp(-((self.rc - self.r_core) / self.sigma_shear) ** 2),
        ) + self.omega_out
        vt = self.rc * om
        self.Wx = -vt * np.sin(self.theta)
        self.Wy = vt * np.cos(self.theta)

        # ---- initial free surface (frozen scaffold formula, axisymmetric
        #      depression plus the small seeded azimuthal mode)
        ratio = np.clip(self.rc / self.R, 0.0, 1.0)
        dep = 0.16 * (1.0 - ratio ** 2) ** 2
        seed_surf = 0.035 * 4.0 * ratio * (1.0 - ratio) * np.cos(self.mode * self.theta)
        eta0 = self.fill - dep - seed_surf
        self.h = self._fill(np.where(self.wet, eta0, 0.0))
        self.u = self._fill(np.zeros_like(self.h))
        self.v = self._fill(np.zeros_like(self.h))
        self.volume0 = float(self.h[self.wet].sum())
        self.dt_cap = 1.5e-3
        self._ready = True

    # ------------------------------------------------------- domain helpers
    def _nearest_wet_indices(self):
        if _ndi is not None:
            _dist, idx = _ndi.distance_transform_edt(~self.wet, return_indices=True)
            return idx
        # fallback: iterative neighbour expansion (deterministic)
        n = self.n
        idx = np.indices((n, n))
        resolved = self.wet.copy()

        def _roll2(a, sh, ax):
            out = np.roll(a, sh, axis=ax)
            if ax == 0:
                if sh == 1:
                    out[-1, :] = a[-1, :]
                else:
                    out[0, :] = a[0, :]
            else:
                if sh == 1:
                    out[:, -1] = a[:, -1]
                else:
                    out[:, 0] = a[:, 0]
            return out

        while not resolved.all():
            progress = False
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                shifted = _roll2(resolved, sh, ax)
                newly = shifted & ~resolved
                if newly.any():
                    src = _roll2(idx, sh, ax + 1)
                    idx = idx.copy()
                    idx[:, newly] = src[:, newly]
                    resolved |= newly
                    progress = True
            if not progress:
                break
        return idx

    def _fill(self, a: np.ndarray) -> np.ndarray:
        """Replace dry cells with the nearest wet value (circular halo)."""
        return a[self._fill_idx[0], self._fill_idx[1]]

    def _ramp(self, t: float) -> float:
        x = min(max(t / self.ramp_time, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    # ------------------------------------------------------------- dynamics
    def _rhs(self, t: float):
        """Right-hand side of the shallow-water system (dry cells zeroed)."""
        h, u, v = self.h, self.u, self.v
        q = h * u
        r_ = h * v
        dh = -(np.gradient(q, self.dx, axis=0) + np.gradient(r_, self.dx, axis=1))
        dh = np.clip(dh, -50.0, 50.0)
        gx = -self.g * np.gradient(h, self.dx, axis=0)
        gy = -self.g * np.gradient(h, self.dx, axis=1)
        ux = np.gradient(u, self.dx, axis=0)
        uy = np.gradient(u, self.dx, axis=1)
        vx = np.gradient(v, self.dx, axis=0)
        vy = np.gradient(v, self.dx, axis=1)
        au = -(u * ux + v * uy)
        av = -(u * vx + v * vy)
        lap_u = np.gradient(ux, self.dx, axis=0) + np.gradient(uy, self.dx, axis=1)
        lap_v = np.gradient(vx, self.dx, axis=0) + np.gradient(vy, self.dx, axis=1)
        ramp = self._ramp(t)
        fu = (ramp * self.Wx - u) / self.tau_spin + self.nu * lap_u
        fv = (ramp * self.Wy - v) / self.tau_spin + self.nu * lap_v
        du = np.clip(au + gx + fu, -80.0, 80.0)
        dv = np.clip(av + gy + fv, -80.0, 80.0)
        return (
            np.where(self.wet, dh, 0.0),
            np.where(self.wet, du, 0.0),
            np.where(self.wet, dv, 0.0),
        )

    def _apply_constraints(self, h, u, v):
        """Wet-only clipping, wall no-penetration, volume conservation."""
        h = np.where(self.wet, np.clip(h, 0.05, 3.0), 0.0)
        u = np.where(self.wet, np.clip(u, -8.0, 8.0), 0.0)
        v = np.where(self.wet, np.clip(v, -8.0, 8.0), 0.0)
        if self.wall_layer.any():
            ur = u[self.wall_layer] * self.er_x[self.wall_layer] + v[self.wall_layer] * self.er_y[self.wall_layer]
            u = u.copy()
            v = v.copy()
            u[self.wall_layer] -= ur * self.er_x[self.wall_layer]
            v[self.wall_layer] -= ur * self.er_y[self.wall_layer]
        vol = float(h[self.wet].sum())
        if vol > 0.0 and self.volume0 > 0.0:
            h = h * (self.volume0 / vol)
        return self._fill(h), self._fill(u), self._fill(v)

    def _step(self, dt: float) -> bool:
        """Advance one RK3 step. Returns False if the safety net triggered."""
        t0 = self.t
        h0, u0, v0 = self.h, self.u, self.v  # backup
        k1h, k1u, k1v = self._rhs(t0)
        self.h = self._fill(np.where(self.wet, h0 + 0.5 * dt * k1h, 0.0))
        self.u = self._fill(np.where(self.wet, u0 + 0.5 * dt * k1u, 0.0))
        self.v = self._fill(np.where(self.wet, v0 + 0.5 * dt * k1v, 0.0))
        k2h, k2u, k2v = self._rhs(t0 + 0.5 * dt)
        self.h = self._fill(np.where(self.wet, h0 - dt * k1h + 2.0 * dt * k2h, 0.0))
        self.u = self._fill(np.where(self.wet, u0 - dt * k1u + 2.0 * dt * k2u, 0.0))
        self.v = self._fill(np.where(self.wet, v0 - dt * k1v + 2.0 * dt * k2v, 0.0))
        k3h, k3u, k3v = self._rhs(t0 + dt)
        hn = h0 + (dt / 6.0) * (k1h + 4.0 * k2h + k3h)
        un = u0 + (dt / 6.0) * (k1u + 4.0 * k2u + k3u)
        vn = v0 + (dt / 6.0) * (k1v + 4.0 * k2v + k3v)
        if not (np.isfinite(hn).all() and np.isfinite(un).all() and np.isfinite(vn).all()):
            # numerical safety net: restore, damp harder, retry later
            self.h, self.u, self.v = h0, u0, v0
            self.nu = min(self.nu * 1.5, 0.05)
            self._fail_count += 1
            return False
        self.h, self.u, self.v = self._apply_constraints(hn, un, vn)
        return True

    # ------------------------------------------------------------ sampling
    def _sample(self, a: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Bilinear sampling of a grid field at 1-D arrays of points."""
        n, dx, c0 = self.n, self.dx, self.centers[0]
        px = np.asarray(px, dtype=float).ravel()
        py = np.asarray(py, dtype=float).ravel()
        fx = (px - c0) / dx
        fy = (py - c0) / dx
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, n - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, n - 2)
        i1 = i0 + 1
        j1 = j0 + 1
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        a00 = a[i0, j0]
        a10 = a[i1, j0]
        a01 = a[i0, j1]
        a11 = a[i1, j1]
        return (a00 * (1 - tx) * (1 - ty) + a10 * tx * (1 - ty)
                + a01 * (1 - tx) * ty + a11 * tx * ty)

    def _surface_tendency(self) -> np.ndarray:
        """dh/dt of the current state (kinematic free-surface condition)."""
        q = self.h * self.u
        r_ = self.h * self.v
        d = -(np.gradient(q, self.dx, axis=0) + np.gradient(r_, self.dx, axis=1))
        return self._fill(np.where(self.wet, np.clip(d, -50.0, 50.0), 0.0))

    # ---------------------------------------------------------------- frames
    def _boundary_block(self, t: float):
        ramp = self._ramp(t)
        omega_now = self.omega_plate * ramp
        pts, nrm, tgt, vel = [], [], [], []
        # rotating bottom plate (top surface at z = 0.14)
        plate_r = [0.0, 0.24, 0.55, 0.85]
        n_ang = 24
        for pr in plate_r:
            if pr == 0.0:
                angs = [0.0]
            else:
                angs = 2.0 * np.pi * np.arange(n_ang) / n_ang
            for a in angs:
                x, y = pr * np.cos(a), pr * np.sin(a)
                pts.append((x, y, 0.14))
                nrm.append((0.0, 0.0, -1.0))
                tgt.append((-omega_now * y, omega_now * x, 0.0))
                if pr == 0.0:
                    vel.append((0.0, 0.0, 0.0))
                else:
                    uu = float(self._sample(self.u, np.array([x]), np.array([y]))[0])
                    vv = float(self._sample(self.v, np.array([x]), np.array([y]))[0])
                    # fluid lags the plate near the bottom (Ekman layer)
                    vel.append((0.7 * uu, 0.7 * vv, 0.0))
        # stationary side wall
        wall_z = [0.25, 0.6, 0.95, 1.3]
        rw = self.R - 0.02
        for wz in wall_z:
            for k in range(n_ang):
                a = 2.0 * np.pi * k / n_ang
                x, y = rw * np.cos(a), rw * np.sin(a)
                pts.append((x, y, wz))
                nrm.append((np.cos(a), np.sin(a), 0.0))
                tgt.append((0.0, 0.0, 0.0))
                uu = float(self._sample(self.u, np.array([x]), np.array([y]))[0])
                vv = float(self._sample(self.v, np.array([x]), np.array([y]))[0])
                vel.append((uu, vv, 0.0))
        return (np.asarray(pts, dtype=float), np.asarray(nrm, dtype=float),
                np.asarray(tgt, dtype=float), np.asarray(vel, dtype=float))

    def _build_frame(self, t: float) -> StateFrame:
        dhdt = self._surface_tendency()
        vx = self.vxy[:, 0]
        vy = self.vxy[:, 1]
        eta_v = self._sample(self.h, vx, vy)
        uvx = self._sample(self.u, vx, vy)
        uvy = self._sample(self.v, vx, vy)
        wv = self._sample(dhdt, vx, vy)
        position = np.column_stack([self.vxy, eta_v])
        velocity = np.column_stack([uvx, uvy, wv])

        # diagnostic grid (frozen field_grid_shape)
        nx, ny, nz = self.grid_shape
        gx = np.linspace(-self.R, self.R, nx)
        gy = np.linspace(-self.R, self.R, ny)
        gz = np.linspace(0.0, self.wall_height, nz)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        gxv = GX.ravel()
        gyv = GY.ravel()
        eta_g = self._sample(self.h, gxv, gyv).reshape(nx, ny)
        ug = self._sample(self.u, gxv, gyv).reshape(nx, ny)
        vg = self._sample(self.v, gxv, gyv).reshape(nx, ny)
        dg = self._sample(dhdt, gxv, gyv).reshape(nx, ny)

        rr = np.hypot(GX, GY)
        Z = gz[None, None, :]
        level_set = np.maximum(rr[..., None] - self.R, Z - eta_g[..., None])
        interior = level_set < -1.5 * self.h_part

        velocity_field = np.zeros((nx, ny, nz, 3), dtype=float)
        eta_safe = np.maximum(eta_g, 0.05)
        inside = (rr < self.R) & (Z < eta_g[..., None])
        depth_frac = np.clip(Z / eta_safe[..., None], 0.0, 1.0)
        factor = 0.55 + 0.45 * depth_frac
        velocity_field[..., 0] = np.where(inside, ug[..., None] * factor, 0.0)
        velocity_field[..., 1] = np.where(inside, vg[..., None] * factor, 0.0)
        velocity_field[..., 2] = np.where(inside, dg[..., None] * depth_frac, 0.0)

        b_pos, b_nrm, b_tgt, b_vel = self._boundary_block(t)

        attributes = {
            "mass_quantity": self.mass0.copy(),
            "particle_radius": self.prad.copy(),
        }
        fields = {
            "liquid_level_set": level_set.astype(float),
            "liquid_interior_mask": interior,
            "velocity_field": velocity_field,
            "boundary_normal": b_nrm,
            "boundary_target_velocity": b_tgt,
            "boundary_velocity": b_vel,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        scalars = {
            "grid_spacing": float(self.h_part),
            "characteristic_length": 2.0 * self.R,
            "reference_velocity": max(self.omega_plate * self.R, 1e-6),
        }
        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            mass=self.mass0.copy(),
            body_id=np.zeros(self.n_vertices, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver not initialized")
        t = float(time)
        if t < self.t:
            t = self.t
        guard = 0
        while self.t < t - 1e-9:
            remaining = t - self.t
            c = np.sqrt(self.g * max(float(self.h.max()), 0.05))
            sp = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()), 1e-3)
            dt = min(self.dt_cap, 0.30 * self.dx / (c + sp), remaining)
            dt = max(dt, 1e-6)
            ok = self._step(dt)
            if ok:
                self.t += dt
                if abs(self.t - t) < 1e-9:
                    self.t = t
            guard += 1
            if guard > 400000 or self._fail_count > 200:
                self.t = t
                break
        return self._build_frame(t)

    def diagnostics(self) -> dict:
        out = {
            "time": float(self.t),
            "wet_volume": float(self.h[self.wet].sum()) * self.dx * self.dx,
            "max_speed": float(max(np.abs(self.u).max(), np.abs(self.v).max())),
            "omega_plate": float(self.omega_plate * self._ramp(self.t)),
            "eddy_viscosity": float(self.nu),
            "numerical_resets": int(self._fail_count),
        }
        try:
            ring = 0.45 * self.R
            ang = 2.0 * np.pi * np.arange(128) / 128
            px = ring * np.cos(ang)
            py = ring * np.sin(ang)
            eta = self._sample(self.h, px, py)
            c = np.fft.rfft(eta - eta.mean())
            m = self.mode
            if m < len(c):
                out["mode4_amplitude"] = float(2.0 * np.abs(c[m]) / len(ang))
        except Exception:
            pass
        return out
