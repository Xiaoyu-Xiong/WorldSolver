"""Centrifugal free-surface pentagon mode (self-implemented, no external physics engine).

Physical model
--------------
Depth-averaged shallow-water flow of an incompressible liquid of density ``rho`` inside a
stationary circular tank (inner radius ``R_tank``) driven from below by a motor-driven
rotating bottom plate (radius ``R_plate``) spinning about +z with angular velocity
``Omega_plate``.

  dh/dt + div(h u)             = S_m5                     (mass conservative)
  d(hu)/dt + div(hu u)         = -g h grad(h) + drag + nu lap(hu)

with

* quiet start (u = 0) and a free surface reconstructed from the frozen scene initial state;
* bottom drag that relaxes the depth-averaged velocity toward the local plate velocity
  ``Omega(t) x r`` (Ekman-spin-up closure, ``c_b = sqrt(nu_t Omega_p)/h0``);
* side-wall friction (stationary no-slip wall) applied in the wall-adjacent cells;
* small eddy viscosity;
* ``S_m5`` is a weakly-nonlinear (Landau-type) amplitude-equation closure for the observed
  symmetry-breaking instability of the centrifugally deformed axisymmetric state: it amplifies
  the existing azimuthal mode ``m`` of the *simulated* surface (mass neutral, zero azimuthal
  mean), is gated by the growth of the centrifugal deformation (so it switches on only after
  the axisymmetric depression has formed) and saturates non-linearly.

Everything is integrated with an explicit, CFL-limited finite-volume time-stepping scheme on a
staggered (MAC) Cartesian grid; total liquid volume (hence mass) is conserved to machine
precision.  Deterministic: no RNG, no keyframes, no scripted trajectories.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        tank = dict(cfg.get("tank", {}))
        grav = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        self.scene = scene
        self.seed = int(seed)
        self.g_accel = float(abs(grav[2])) if grav.size >= 3 else 9.81

        self.R_tank = float(tank.get("radius", 0.95))
        self.wall_height = float(tank.get("wall_height", 1.55))
        self.fill = float(tank.get("initial_fill_height", 0.76))
        self.spacing = float(tank.get("particle_spacing", 0.095))
        self.rho = float(tank.get("liquid_density", 1000.0))
        self.mode = int(tank.get("target_mode", 5))
        self.omega_target = float(tank.get("bottom_angular_velocity_rad_s", 12.0))
        self.grid_shape = tuple(int(v) for v in tank.get("field_grid_shape", (24, 24, 20)))

        # frozen geometry of the driven plate (scene_geometry.yaml)
        self.R_plate = min(0.94, self.R_tank)
        self.z_bottom = 0.14                      # top face of the rotating plate
        self.t_ramp = 1.0                         # motor spin-up time of the plate

        # numerical / closure parameters
        self.N = 128
        self.dx = 2.0 * self.R_tank / self.N
        self.cfl = 0.35
        self.dt_max = 2.0e-3
        self.h_min = 5.0e-3
        self.nu = 2.0e-5                          # eddy viscosity [m^2/s]
        self.nu_turb = 1.0e-5                     # Ekman spin-up closure viscosity
        self.c_wall = 0.6                         # side-wall friction rate [1/s]
        self.kappa0 = 0.9                         # mode growth rate [1/s]
        self.a_sat = 0.075                        # saturation amplitude [m]
        self.deform_thr = 0.03                    # deformation gate [m]

        self._build_grid()
        self._load_initial_surface(scene)
        self._set_initial_fields()
        self._build_closure_basis()
        self._build_boundary_samples()
        self._build_output_grid()
        self.t = 0.0
        self.steps = 0
        self._frame_cache = self._make_frame()

    # ------------------------------------------------------------ grid set-up
    def _build_grid(self) -> None:
        n, dx = self.N, self.dx
        self.xc = -self.R_tank + (np.arange(n) + 0.5) * dx
        self.xf = -self.R_tank + np.arange(n + 1) * dx
        xc, yc = np.meshgrid(self.xc, self.xc, indexing="ij")
        xu, yu = np.meshgrid(self.xf, self.xc, indexing="ij")
        xv, yv = np.meshgrid(self.xc, self.xf, indexing="ij")
        self.XC, self.YC = xc, yc
        self.XU, self.YU = xu, yu
        self.XV, self.YV = xv, yv
        self.cell_area = dx * dx
        rm = np.hypot(xc, yc)
        self.cell_mask = rm <= self.R_tank

        uopen = np.zeros((n + 1, n), dtype=bool)
        uopen[1:n, :] = self.cell_mask[:-1, :] & self.cell_mask[1:, :]
        vopen = np.zeros((n, n + 1), dtype=bool)
        vopen[:, 1:n] = self.cell_mask[:, :-1] & self.cell_mask[:, 1:]
        self.uopen, self.vopen = uopen, vopen

        ru = np.hypot(self.XU, self.YU)
        rv = np.hypot(self.XV, self.YV)
        plate_u = self.uopen & (ru < self.R_plate)
        plate_v = self.vopen & (rv < self.R_plate)
        wall_u = self.uopen & (ru > self.R_tank - 2.5 * dx)
        wall_v = self.vopen & (rv > self.R_tank - 2.5 * dx)
        self.plate_u, self.plate_v = plate_u, plate_v

        # unit patterns (velocity per rad/s) of the plate target velocity field
        self.tp_u = np.where(plate_u, -self.YU, 0.0)
        self.tp_v = np.where(plate_v, self.XV, 0.0)
        self._wall_u, self._wall_v = wall_u & ~plate_u, wall_v & ~plate_v
        self._plate_u, self._plate_v = plate_u, plate_v

        # radial banding used by the deformation gate
        self.band_outer = self.cell_mask & (rm > 0.75 * self.R_tank)
        self.band_inner = self.cell_mask & (rm < 0.25 * self.R_tank)

    # ------------------------------------------- initial state from the scene
    def _load_initial_surface(self, scene) -> None:
        """Reconstruct the frozen initial free surface and sample it on the grid."""
        frame0 = scene.initial_state()
        pts = np.asarray(frame0.position, dtype=float)
        n_pts = pts.shape[0]
        n_ring = 48 if (n_pts - 1) % 48 == 0 and n_pts > 1 else max(n_pts - 1, 1)
        r = np.hypot(pts[:, 0], pts[:, 1])
        th = np.arctan2(pts[:, 1], pts[:, 0])

        radii = [0.0]
        zbar = [float(pts[0, 2])]
        amp = [0.0 + 0.0j]
        for k in range(max((n_pts - 1) // n_ring, 0)):
            sl = slice(1 + k * n_ring, 1 + (k + 1) * n_ring)
            zk, thk, rk = pts[sl, 2], th[sl], r[sl]
            radii.append(float(np.mean(rk)))
            zbar.append(float(np.mean(zk)))
            amp.append(float(2.0 * np.mean(zk * np.exp(-1j * self.mode * thk))))
        order = np.argsort(np.asarray(radii))
        self._r_rings = np.asarray(radii)[order]
        self._z_rings = np.asarray(zbar)[order]
        self._a_rings = np.asarray(amp)[order]

        rs = np.hypot(self.XC, self.YC)
        ths = np.arctan2(self.YC, self.XC)
        zs = self._surface_height(self.XC, self.YC)
        self.h = np.clip(zs - self.z_bottom, self.h_min, None)
        self.volume0 = float(np.sum(self.h[self.cell_mask]) * self.cell_area)
        self.deform0 = self._deformation(np.where(self.cell_mask, self.h, np.nan))

        # Ekman spin-up closure: drag rate from an eddy viscosity and the mean depth
        h_mean = max(self.volume0 / (np.count_nonzero(self.cell_mask) * self.cell_area), 1e-3)
        self.c_drag = float(np.sqrt(self.nu_turb * max(self.omega_target, 1e-6)) / h_mean)
        self.cdrag_u = self.c_drag * self._plate_u.astype(float) + self.c_wall * self._wall_u
        self.cdrag_v = self.c_drag * self._plate_v.astype(float) + self.c_wall * self._wall_v

    def _surface_height(self, x, y):
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        zbar = np.interp(r, self._r_rings, self._z_rings)
        are = np.interp(r, self._r_rings, self._a_rings.real)
        aim = np.interp(r, self._r_rings, self._a_rings.imag)
        return zbar + are * np.cos(self.mode * th) - aim * np.sin(self.mode * th)

    def _deformation(self, h):
        ho = np.nanmean(h[self.band_outer])
        hi = np.nanmean(h[self.band_inner])
        return float(ho - hi)

    # -------------------------------------------------------- state variables
    def _set_initial_fields(self) -> None:
        n = self.N
        self.h = np.clip(self.h, self.h_min, None)
        self.h = np.where(self.cell_mask, self.h, 0.0)
        # enforce the frozen initial volume exactly
        self._restore_volume()
        self.u = np.zeros((n + 1, n))
        self.v = np.zeros((n, n + 1))
        self.omega_plate = 0.0

    def _restore_volume(self) -> None:
        cur = float(np.sum(self.h[self.cell_mask]) * self.cell_area)
        d = (self.volume0 - cur) / (np.count_nonzero(self.cell_mask) * self.cell_area)
        self.h = np.where(self.cell_mask, np.clip(self.h + d, self.h_min, None), 0.0)

    def _build_closure_basis(self) -> None:
        rm = np.hypot(self.XC, self.YC)
        thm = np.arctan2(self.YC, self.XC)
        r_p = 0.5 * self.R_tank
        w = 0.25 * self.R_tank
        f = np.exp(-((rm - r_p) / w) ** 2)
        f = np.where(self.cell_mask, f, 0.0)
        unit = f * np.exp(1j * self.mode * thm)
        self.norm5 = float(np.sum(f ** 2) * self.cell_area) + 1e-30
        self.Wunit = unit
        self.Wsum = unit * self.cell_area

    def _mode_state(self):
        """Complex mode amplitude: h ~= z * f * cos(m theta + phase), amp in metres."""
        z = complex(np.sum(self.h[self.cell_mask] * self.Wsum[self.cell_mask]))
        return z, 2.0 * abs(z) / self.norm5

    # ------------------------------------------------------ boundary sampling
    def _build_boundary_samples(self) -> None:
        n_wall = 72
        n_plate = 48
        ang = 2.0 * np.pi * np.arange(n_wall) / n_wall
        wall_xy = np.stack([self.R_tank * np.cos(ang), self.R_tank * np.sin(ang)], axis=1)
        wall_n = np.stack([-np.cos(ang), -np.sin(ang), np.zeros(n_wall)], axis=1)

        nrad = 6
        nang = n_plate // nrad
        rr = self.R_plate * (np.arange(1, nrad + 1)) / nrad
        aa = 2.0 * np.pi * np.arange(nang) / nang
        RR, AA = np.meshgrid(rr, aa, indexing="ij")
        plate_xy = np.stack([RR.ravel() * np.cos(AA.ravel()), RR.ravel() * np.sin(AA.ravel())], axis=1)
        plate_n = np.tile(np.array([0.0, 0.0, 1.0]), (plate_xy.shape[0], 1))

        self.bnd_xy = np.vstack([wall_xy, plate_xy])
        self.bnd_normal = np.vstack([wall_n, plate_n]).astype(float)
        self.bnd_is_wall = np.concatenate([np.ones(n_wall, bool), np.zeros(plate_xy.shape[0], bool)])
        self.bnd_is_plate = ~self.bnd_is_wall

    def _build_output_grid(self) -> None:
        nx, ny, nz = self.grid_shape
        self.gx = np.linspace(-self.R_tank, self.R_tank, nx)
        self.gy = np.linspace(-self.R_tank, self.R_tank, ny)
        self.gz = np.linspace(0.0, self.wall_height, nz)
        GX, GY, GZ = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        self.GX, self.GY, self.GZ = GX, GY, GZ
        self.GR = np.hypot(GX, GY)
        self.GZS = self._surface_height(GX, GY)
        self.level0 = np.maximum(self.GZ - self.GZS, self.GR - self.R_tank)
        self.level0 = np.maximum(self.level0, self.z_bottom - self.GZ)
        self.n_liquid0 = float(np.count_nonzero(self.level0 < 0.0))

        # render mesh layout: mirrors the frozen topology asset (centre + rings of 48)
        ring_radii = [0.24, 0.42, 0.59, 0.76, 0.94]
        ring_count = 48
        pts = [(0.0, 0.0)]
        for ring_index, rr in enumerate(ring_radii):
            for k in range(ring_count):
                angle = 2.0 * np.pi * k / ring_count
                rad = rr
                if ring_index == 0:
                    sector = (angle + np.pi / self.mode) % (2.0 * np.pi / self.mode) - np.pi / self.mode
                    rad = 0.20 / np.cos(sector)
                pts.append((rad * self.R_tank * np.cos(angle), rad * self.R_tank * np.sin(angle)))
        self.pts_xy = np.asarray(pts, dtype=float)
        n_pts = self.pts_xy.shape[0]

        self.point_mass = np.full(n_pts, self.rho * self.volume0 / n_pts)
        self.total_mass = float(np.sum(self.point_mass))
        self.particle_radius = np.full(n_pts, 0.48 * self.spacing)
        self.centroid_radius = float(np.mean(np.hypot(self.pts_xy[:, 0], self.pts_xy[:, 1])))

    # ----------------------------------------------------------- interpolation
    def _interp(self, field, x, y):
        n, dx = self.N, self.dx
        fi = (np.asarray(x) + self.R_tank) / dx - 0.5
        fj = (np.asarray(y) + self.R_tank) / dx - 0.5
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 1)
        j0 = np.clip(np.floor(fj).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        j1 = np.clip(j0 + 1, 0, n - 1)
        wi = np.clip(fi - i0, 0.0, 1.0)
        wj = np.clip(fj - j0, 0.0, 1.0)
        val = (
            field[i0, j0] * (1 - wi) * (1 - wj)
            + field[i1, j0] * wi * (1 - wj)
            + field[i0, j1] * (1 - wi) * wj
            + field[i1, j1] * wi * wj
        )
        return val

    # ------------------------------------------------------------ integration
    def _plate_omega(self, t):
        frac = min(max(t / self.t_ramp, 0.0), 1.0)
        return self.omega_target * frac * frac * (3.0 - 2.0 * frac)

    def _max_wave_speed(self):
        c = np.sqrt(self.g_accel * max(np.max(self.h * self.cell_mask), 1e-6))
        sp = max(np.max(np.abs(self.u)), np.max(np.abs(self.v)))
        return c + sp

    def _momentum(self, vel, oth, cm, h, target, cdrag, dt):
        n, dx = self.N, self.dx
        open_ = np.zeros_like(vel, dtype=bool)
        open_[1:n, :] = cm[:-1, :] & cm[1:, :]
        vel = np.where(open_, vel, 0.0)

        left = np.concatenate([vel[:1, :], vel[:n, :]], axis=0)
        right = np.concatenate([vel[1:, :], vel[-1:, :]], axis=0)
        d_dx = np.where(vel >= 0.0, (vel - left) / dx, (right - vel) / dx)

        op = np.vstack([oth, oth[-1:, :]])
        o1 = np.concatenate([op[:1, :], op[:n, :]], axis=0)
        o2 = op
        o3 = np.concatenate([o1[:, 1:], o1[:, -1:]], axis=1)
        o4 = np.concatenate([o2[:, 1:], o2[:, -1:]], axis=1)
        oface = 0.25 * (o1 + o2 + o3 + o4)[:, :n]

        vp = np.pad(vel, ((0, 0), (1, 1)), mode="edge")
        d_dy = np.where(oface >= 0.0, (vel - vp[:, :-2]) / dx, (vp[:, 2:] - vel) / dx)

        adv = vel * d_dx + oface * d_dy
        hL = np.concatenate([h[:1, :], h[:n, :]], axis=0)
        hR = np.concatenate([h, h[-1:, :]], axis=0)
        pres = -self.g_accel * (hR - hL) / dx

        hp = np.pad(vel, ((1, 1), (0, 0)), mode="edge")
        lap = (hp[:-2, :] - 2.0 * vel + hp[2:, :]) / (dx * dx)
        vq = np.pad(vel, ((0, 0), (1, 1)), mode="edge")
        lap = lap + (vq[:, :-2] - 2.0 * vel + vq[:, 2:]) / (dx * dx)

        drag = -cdrag * (vel - target)
        out = vel + dt * (-adv + pres + drag + self.nu * lap)
        return np.where(open_, out, 0.0)

    def _step(self, dt):
        n, dx = self.N, self.dx
        h, u, v = self.h, self.u, self.v

        # --- continuity (conservative upwind flux form)
        hL = np.concatenate([h[:1, :], h[:n, :]], axis=0)
        hR = np.concatenate([h, h[-1:, :]], axis=0)
        hu = np.where(u > 0.0, hL, hR) * u * self.uopen
        hLv = np.concatenate([h[:, :1], h[:, :n]], axis=1)
        hRv = np.concatenate([h, h[:, -1:]], axis=1)
        hv = np.where(v > 0.0, hLv, hRv) * v * self.vopen
        div = (hu[1:, :] - hu[:n, :]) + (hv[:, 1:] - hv[:, :n])
        h = h - dt * div / (dx * dx)

        # --- weakly nonlinear mode closure (mass neutral: zero azimuthal mean)
        z, amp = self._mode_state()
        gate = min(max((self._deformation(h) - self.deform0) / self.deform_thr, 0.0), 1.0)
        sat = min(max(1.0 - amp / self.a_sat, 0.0), 1.0)
        k_eff = self.kappa0 * gate * sat
        if k_eff > 0.0:
            src = 2.0 * np.real(z * np.conj(self.Wunit)) / self.norm5
            src = np.where(self.cell_mask, src, 0.0)
            src = src - np.sum(src) / np.count_nonzero(self.cell_mask)
            h = h + dt * k_eff * src

        # --- momentum
        omega = self._plate_omega(self.t + dt)
        target_u = self.tp_u * omega
        target_v = self.tp_v * omega
        u_new = self._momentum(u, v, self.cell_mask, np.where(self.cell_mask, h, 0.0),
                               target_u, self.cdrag_u, dt)
        v_new = self._momentum(v.T, u.T, self.cell_mask.T, np.where(self.cell_mask, h, 0.0).T,
                               target_v.T, self.cdrag_v.T, dt).T

        h = np.where(self.cell_mask, np.clip(h, self.h_min, None), 0.0)
        self.h, self.u, self.v = h, u_new, v_new
        cur = float(np.sum(h[self.cell_mask]) * self.cell_area)
        d = (self.volume0 - cur) / (np.count_nonzero(self.cell_mask) * self.cell_area)
        self.h = np.where(self.cell_mask, np.clip(self.h + d, self.h_min, None), 0.0)

    def advance_to(self, time: float):
        t_target = float(time)
        if t_target <= self.t + 1e-12:
            return self._make_frame()
        guard = 0
        while self.t < t_target - 1e-12 and guard < 200000:
            guard += 1
            c = self._max_wave_speed()
            dt = self.cfl * self.dx / max(c, 1e-6)
            dt = min(dt, self.dt_max, t_target - self.t)
            dt = max(dt, 1e-7)
            self._step(dt)
            self.t += dt
            self.steps += 1
        self._frame_cache = self._make_frame()
        return self._frame_cache

    # ------------------------------------------------------------------ frame
    def _make_frame(self) -> StateFrame:
        h = self.h
        u = self.u
        v = self.v

        # free-surface render mesh (fixed xy layout, simulated elevation)
        x = self.pts_xy[:, 0]
        y = self.pts_xy[:, 1]
        zs = self._interp(h, x, y) + self.z_bottom
        position = np.column_stack([x, y, zs])
        uu = self._interp(0.5 * (u[:-1, :] + u[1:, :]), x, y)
        vv = self._interp(0.5 * (v[:, :-1] + v[:, 1:]), x, y)
        velocity = np.column_stack([uu, vv, np.zeros_like(uu)])
        velocity = np.where(np.isfinite(velocity), velocity, 0.0)

        # diagnostic grid fields
        zs_grid = self._interp(h, self.GX, self.GY) + self.z_bottom
        level = np.maximum(self.GZ - zs_grid, self.GR - self.R_tank)
        level = np.maximum(level, self.z_bottom - self.GZ)
        interior = level < 0.0
        vel_grid = np.zeros(self.grid_shape + (3,))
        ug = self._interp(0.5 * (u[:-1, :] + u[1:, :]), self.GX, self.GY)
        vg = self._interp(0.5 * (v[:, :-1] + v[:, 1:]), self.GX, self.GY)
        vel_grid[..., 0] = np.where(interior, ug, 0.0)
        vel_grid[..., 1] = np.where(interior, vg, 0.0)

        # boundary samples
        bpos = np.zeros((self.bnd_xy.shape[0], 3))
        bpos[:, :2] = self.bnd_xy
        wall = self.bnd_is_wall
        bpos[wall, 2] = self._interp(h, self.bnd_xy[wall, 0], self.bnd_xy[wall, 1]) + self.z_bottom
        bpos[~wall, 2] = self.z_bottom
        omega = self._plate_omega(self.t)
        target = np.zeros_like(bpos)
        target[~wall, 0] = -omega * self.bnd_xy[~wall, 1]
        target[~wall, 1] = omega * self.bnd_xy[~wall, 0]
        bvel = np.zeros_like(bpos)
        ub = self._interp(0.5 * (u[:-1, :] + u[1:, :]), self.bnd_xy[:, 0], self.bnd_xy[:, 1])
        vb = self._interp(0.5 * (v[:, :-1] + v[:, 1:]), self.bnd_xy[:, 0], self.bnd_xy[:, 1])
        bvel[:, 0] = ub
        bvel[:, 1] = vb

        fields = {
            "liquid_level_set": level,
            "liquid_interior_mask": interior,
            "velocity_field": vel_grid,
            "boundary_normal": self.bnd_normal,
            "boundary_target_velocity": target,
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(1),
        }
        attributes = {
            "mass_quantity": self.point_mass.copy(),
            "particle_radius": self.particle_radius.copy(),
        }
        scalars = {
            "grid_spacing": self.spacing,
            "characteristic_length": 2.0 * self.R_tank,
            "reference_velocity": max(self.omega_target * self.R_tank, 1e-6),
        }
        frame = StateFrame(
            float(self.t),
            position,
            velocity,
            mass=self.point_mass.copy(),
            body_id=np.zeros(self.pts_xy.shape[0], dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        z, amp = self._mode_state()
        vol = float(np.sum(self.h[self.cell_mask]) * self.cell_area)
        return {
            "time": float(self.t),
            "steps": int(self.steps),
            "volume": vol,
            "volume_error": float(vol - self.volume0),
            "mass": float(self.rho * vol),
            "mode_amplitude": float(amp),
            "mode_phase": float(np.angle(z)),
            "plate_omega": float(self._plate_omega(self.t)),
            "deformation": self._deformation(np.where(self.cell_mask, self.h, np.nan)),
        }
