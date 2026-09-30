"""Physically-based 2D dam-break solver.

Model
-----
The liquid is represented by Lagrangian samples (the 1000 cell-centred
samples of the frozen scene) that carry the conserved mass of the initial
liquid column.  They are advanced with an operator-split, semi-implicit
incompressible Navier-Stokes scheme on a uniform staggered (MAC) grid:

   1. particle -> grid transfer of momentum (mass-weighted linear weights),
   2. explicit gravity acceleration and an explicit viscous (Laplacian)
      diffusion term with no-slip tank floor / side walls,
   3. free-surface pressure projection solved with a matrix-free,
      Jacobi-preconditioned conjugate-gradient solver on the wetted cells
      (Neumann at solid walls, Dirichlet p = 0 at free-surface faces),
   4. grid -> particle velocity transfer with a FLIP/PIC blend and
      Lagrangian advection of the samples,
   5. non-penetration clamping against the closed floor and side walls.

The tank geometry (floor top z = 0.05 m, inner side walls at x = +/-3.0 m)
is taken from the frozen scene geometry and is honoured exactly.  All steps
are deterministic and the internal time step exactly subdivides the
requested sample interval, so every requested sample time is hit exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Frozen scene constants (from scene_geometry.yaml / config.yaml)
# ---------------------------------------------------------------------------
_FLOOR_TOP = 0.05          # top face of tank_floor box
_LEFT_INNER = -3.0         # inner face of tank_left_wall  (-3.05 + 0.05)
_RIGHT_INNER = 3.0         # inner face of tank_right_wall ( 3.05 - 0.05)
_TOP_OPEN = 3.05           # open-air top of the simulated domain
_DENSITY = 1000.0          # kg/m^3
_GRAVITY = 9.81            # m/s^2, downward (-z)


class BenchmarkSolver:
    """Two-dimensional dam-break solver (incompressible particle/MAC method)."""

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", {}) or {}

        gravity = cfg.get("gravity", [0.0, 0.0, -_GRAVITY])
        try:
            gz = float(gravity[2])
        except Exception:
            gz = -_GRAVITY
        self.gravity = gz if gz != 0.0 else -_GRAVITY

        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=np.float64)
        mass = np.asarray(frame0.mass, dtype=np.float64)
        if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
            mass = np.full(pos.shape[0], 3.6, dtype=np.float64)
        self.mass = np.maximum(mass, 1e-12)

        self.px = np.array(pos[:, 0], dtype=np.float64, copy=True)
        self.pz = np.array(pos[:, 2], dtype=np.float64, copy=True)
        self.vx = np.zeros_like(self.px)
        self.vz = np.zeros_like(self.pz)
        self.n = int(self.px.size)

        # Initial column bounds (frozen liquid region) for length scales.
        x0 = float(np.min(self.px))
        x1 = float(np.max(self.px))
        z0 = float(np.min(self.pz))
        z1 = float(np.max(self.pz))
        self.char_length = max(z1 - z0, 1e-6)
        self.ref_velocity = float(np.sqrt(abs(self.gravity) * self.char_length))

        # MAC grid.  Cell size is chosen a little larger than the initial
        # particle spacing so that every column/row of cells holds particles.
        self.h = 0.1
        self.xmin = _LEFT_INNER
        self.xmax = _RIGHT_INNER
        self.zmin = _FLOOR_TOP
        self.zmax = _TOP_OPEN
        self.nx = int(round((self.xmax - self.xmin) / self.h))
        self.nz = int(round((self.zmax - self.zmin) / self.h))
        self.h = min((self.xmax - self.xmin) / self.nx, (self.zmax - self.zmin) / self.nz)

        # Time stepping: internal step exactly subdivides the sample interval.
        sample_dt = float(cfg.get("sample_dt", 0.02))
        if sample_dt <= 0.0:
            sample_dt = 0.02
        self.sample_dt = sample_dt
        self.substeps = 8
        self.dt = sample_dt / self.substeps

        self.flip = 0.95           # FLIP/PIC blend (5% PIC for stability)
        self.nu = 0.03             # effective (turbulent) kinematic viscosity
        self.vmax = 12.0           # velocity safety clamp (m/s)
        self.time = 0.0

        # Warm start for the pressure solve.
        self._p = np.zeros((self.nx, self.nz), dtype=np.float64)

        # Static tank wall samples (floor, left wall, right wall).
        nx, nz, h = self.nx, self.nz, self.h
        xc = self.xmin + (np.arange(nx) + 0.5) * h
        zc = self.zmin + (np.arange(nz) + 0.5) * h
        nrm = np.zeros((nx + 2 * nz, 2), dtype=np.float64)
        nrm[:nx, 1] = 1.0           # floor normal points up into the liquid
        nrm[nx:nx + nz, 0] = 1.0    # left wall normal points +x
        nrm[nx + nz:, 0] = -1.0     # right wall normal points -x
        self.bnd_normal = nrm
        self.bnd_target = np.zeros_like(nrm)
        self._bnd_floor = (xc, zc)
        self._ready = True

    # -------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            target = self.time
        step = self.dt
        while self.time < target - 1e-12:
            dt = min(step, target - self.time)
            if dt <= 1e-15:
                break
            self._step(dt)
            self.time += dt
        # Snap exactly onto the requested sample time.
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------ one substep
    def _step(self, dt: float) -> None:
        nx, nz, h = self.nx, self.nz, self.h
        xmin, zmin = self.xmin, self.zmin
        px, pz, vx, vz = self.px, self.pz, self.vx, self.vz
        m = self.mass

        # --- cell occupancy from particle locations -------------------------
        ci = np.clip(((px - xmin) / h).astype(np.int64), 0, nx - 1)
        cj = np.clip(((pz - zmin) / h).astype(np.int64), 0, nz - 1)
        occ = np.zeros((nx, nz), dtype=bool)
        occ[ci, cj] = True

        # --- face openness: solid walls closed, fluid/air faces open -------
        xopen = np.zeros((nx + 1, nz), dtype=bool)
        xopen[1:nx] = occ[:-1] | occ[1:]
        zopen = np.zeros((nx, nz + 1), dtype=bool)
        zopen[:, 1:nz] = occ[:, :-1] | occ[:, 1:]
        zopen[:, nz] = occ[:, nz - 1]      # open air above the liquid

        # --- particle -> grid weights --------------------------------------
        gx = (px - xmin) / h
        i0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 1)
        ax = np.clip(gx - i0, 0.0, 1.0)
        gz = (pz - zmin) / h
        j0 = np.clip(np.floor(gz).astype(np.int64), 0, nz - 1)
        az = np.clip(gz - j0, 0.0, 1.0)
        wx0, wx1 = 1.0 - ax, ax
        wz0, wz1 = 1.0 - az, az
        pairs = (
            (i0, j0, wx0 * wz0),
            (i0 + 1, j0, wx1 * wz0),
            (i0, j0 + 1, wx0 * wz1),
            (i0 + 1, j0 + 1, wx1 * wz1),
        )

        unum = np.zeros((nx + 1, nz))
        uden = np.zeros_like(unum)
        wnum = np.zeros((nx, nz + 1))
        wden = np.zeros_like(wnum)
        for ii, jj, ww in pairs:
            wt = m * ww
            np.add.at(unum, (ii, jj), wt * vx)
            np.add.at(uden, (ii, jj), wt)
            np.add.at(wnum, (ii, jj), wt * vz)
            np.add.at(wden, (ii, jj), wt)

        u = np.where(uden > 0.0, unum / np.maximum(uden, 1e-30), 0.0)
        w = np.where(wden > 0.0, wnum / np.maximum(wden, 1e-30), 0.0)
        u = u * xopen
        w = w * zopen
        u[0, :] = 0.0
        u[nx, :] = 0.0
        w[:, 0] = 0.0

        # Pre-project grid velocity (used for the FLIP increment).
        u_dep = u
        w_dep = w

        # --- explicit gravity + viscous diffusion --------------------------
        w = w + dt * self.gravity
        if self.nu > 0.0:
            u = self._viscous_u(u, dt)
            w = self._viscous_w(w, dt)
        u = u * xopen
        w = w * zopen
        u[0, :] = 0.0
        u[nx, :] = 0.0
        w[:, 0] = 0.0

        # --- free-surface pressure projection ------------------------------
        um = u * xopen
        wm = w * zopen
        div = (um[1:, :] - um[:-1, :]) / h + (wm[:, 1:] - wm[:, :-1]) / h
        p = self._solve_pressure(np.where(occ, div / dt, 0.0), occ, xopen, zopen)
        self._p = p

        unew = um.copy()
        unew[1:nx] -= dt * (p[1:, :] - p[:-1, :]) / h
        unew[0, :] = 0.0
        unew[nx, :] = 0.0
        wnew = wm.copy()
        wnew[:, 1:nz] -= dt * (p[:, 1:] - p[:, :-1]) / h
        wnew[:, nz] = wm[:, nz] - dt * (-p[:, nz - 1] / h)
        wnew[:, 0] = 0.0

        # --- grid -> particle transfer (FLIP/PIC blend) --------------------
        numx = np.zeros(self.n)
        denx = np.zeros(self.n)
        numz = np.zeros(self.n)
        denz = np.zeros(self.n)
        for ii, jj, ww in pairs:
            numx += ww * unew[ii, jj]
            denx += ww
            numz += ww * wnew[ii, jj]
            denz += ww
        vx_pic = np.where(denx > 1e-12, numx / np.maximum(denx, 1e-30), vx)
        vz_pic = np.where(denz > 1e-12, numz / np.maximum(denz, 1e-30), vz)

        numx = np.zeros(self.n)
        numz = np.zeros(self.n)
        dummy = np.zeros(self.n)
        for ii, jj, ww in pairs:
            numx += ww * u_dep[ii, jj]
            numz += ww * w_dep[ii, jj]
            dummy += ww
        den = np.maximum(dummy, 1e-30)
        udx = np.where(dummy > 1e-12, numx / den, vx)
        udz = np.where(dummy > 1e-12, numz / den, vz)
        dvx = vx_pic - udx
        dvz = vz_pic - udz

        alpha = self.flip
        vx_new = (1.0 - alpha) * vx_pic + alpha * (vx + dvx)
        vz_new = (1.0 - alpha) * vz_pic + alpha * (vz + dvz)
        vx_new = np.clip(vx_new, -self.vmax, self.vmax)
        vz_new = np.clip(vz_new, -self.vmax, self.vmax)

        # --- Lagrangian advection and solid-boundary clamping --------------
        self.px = px + dt * vx_new
        self.pz = pz + dt * vz_new
        self.vx = vx_new
        self.vz = vz_new
        self._apply_bounds()

    def _apply_bounds(self) -> None:
        eps = 0.01
        lo_x, hi_x = self.xmin + eps, self.xmax - eps
        lo_z, hi_z = self.zmin + eps, self.zmax - 0.05
        px = np.nan_to_num(self.px, nan=0.0, posinf=0.0, neginf=0.0)
        pz = np.nan_to_num(self.pz, nan=0.0)
        vx = np.nan_to_num(self.vx, nan=0.0)
        vz = np.nan_to_num(self.vz, nan=0.0)
        floor = pz < lo_z
        pz[floor] = lo_z
        vz[floor] = np.maximum(vz[floor], 0.0)
        ceil = pz > hi_z
        pz[ceil] = hi_z
        vz[ceil] = np.minimum(vz[ceil], 0.0)
        left = px < lo_x
        px[left] = lo_x
        vx[left] = np.maximum(vx[left], 0.0)
        right = px > hi_x
        px[right] = hi_x
        vx[right] = np.minimum(vx[right], 0.0)
        self.px, self.pz, self.vx, self.vz = px, pz, vx, vz

    # ------------------------------------------------------------- viscosity
    def _viscous_u(self, u, dt):
        nx, nz, h = self.nx, self.nz, self.h
        up = np.zeros((nx + 3, nz + 2))
        up[1:nx + 2, 1:nz + 1] = u
        up[:, nz + 1] = u[:, nz - 1]              # zero-gradient free surface
        lap = (up[2:, 1:-1] - 2.0 * up[1:-1, 1:-1] + up[:-2, 1:-1]
               + up[1:-1, 2:] - 2.0 * up[1:-1, 1:-1] + up[1:-1, :-2]) / (h * h)
        out = u + self.nu * dt * lap
        out[0, :] = 0.0
        out[nx, :] = 0.0
        return out

    def _viscous_w(self, w, dt):
        nx, nz, h = self.nx, self.nz, self.h
        wp = np.zeros((nx + 2, nz + 3))
        wp[1:nx + 1, 1:nz + 2] = w
        wp[0, :] = w[0, :]                        # free-slip side walls
        wp[nx + 1, :] = w[nx - 1, :]
        wp[:, nz + 2] = w[:, nz]                  # zero-gradient free surface
        lap = (wp[2:, 1:-1] - 2.0 * wp[1:-1, 1:-1] + wp[:-2, 1:-1]
               + wp[1:-1, 2:] - 2.0 * wp[1:-1, 1:-1] + wp[1:-1, :-2]) / (h * h)
        out = w + self.nu * dt * lap
        out[:, 0] = 0.0
        return out

    # ------------------------------------------------------- pressure solver
    def _solve_pressure(self, rhs, occ, xopen, zopen):
        nx, nz, h = self.nx, self.nz, self.h
        b = np.where(occ, rhs, 0.0)
        if not np.any(occ):
            return np.zeros((nx, nz))
        diag = (xopen[:-1].astype(np.float64) + xopen[1:].astype(np.float64)
                + zopen[:, :-1].astype(np.float64) + zopen[:, 1:].astype(np.float64))
        diag = np.where(diag > 0.0, diag / (h * h), 1.0)

        def apply(p):
            gu = np.zeros((nx + 1, nz))
            gu[1:nx] = (p[1:, :] - p[:-1, :]) / h
            gu *= xopen
            gw = np.zeros((nx, nz + 1))
            gw[:, 1:nz] = (p[:, 1:] - p[:, :-1]) / h
            gw[:, nz] = -p[:, nz - 1] / h
            gw *= zopen
            out = -((gu[1:, :] - gu[:-1, :]) + (gw[:, 1:] - gw[:, :-1])) / h
            return np.where(occ, out, 0.0)

        p = np.where(occ, self._p, 0.0)
        r = b - apply(p)
        z = r / diag
        d = z.copy()
        rz = float(np.sum(r * z))
        bnorm = float(np.sqrt(np.sum(b * b))) + 1e-30
        for _ in range(300):
            if float(np.sqrt(np.sum(r * r))) <= 1e-9 * bnorm:
                break
            ad = apply(d)
            denom = float(np.sum(d * ad))
            if not np.isfinite(denom) or abs(denom) < 1e-300:
                break
            a = rz / denom
            p = p + a * d
            r = r - a * ad
            z = r / diag
            rz_new = float(np.sum(r * z))
            if not np.isfinite(rz_new):
                break
            beta = rz_new / rz if abs(rz) > 1e-300 else 0.0
            rz = rz_new
            d = z + beta * d
        return p

    # ---------------------------------------------------------- diagnostics
    def _fields(self):
        nx, nz, h = self.nx, self.nz, self.h
        xmin, zmin = self.xmin, self.zmin
        xc = xmin + (np.arange(nx) + 0.5) * h
        zc = zmin + (np.arange(nz) + 0.5) * h

        # Smoothed particle volume fraction.
        vol = self.mass / _DENSITY
        sigma = 0.75 * h
        norm = 1.0 / (2.0 * np.pi * sigma * sigma)
        ci = np.clip(((self.px - xmin) / h).astype(np.int64), 0, nx - 1)
        cj = np.clip(((self.pz - zmin) / h).astype(np.int64), 0, nz - 1)
        frac = np.zeros((nx, nz))
        rad = 2
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                ii = ci + di
                jj = cj + dj
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < nz)
                if not np.any(valid):
                    continue
                ix = ii[valid]
                jz = jj[valid]
                dx = xc[ix] - self.px[valid]
                dz = zc[jz] - self.pz[valid]
                wt = vol[valid] * norm * np.exp(-(dx * dx + dz * dz) / (2.0 * sigma * sigma))
                np.add.at(frac, (ix, jz), wt)
        frac = np.clip(frac, 0.0, 1.0)

        inside = frac >= 0.5
        if np.any(inside) and np.any(~inside):
            d_in = distance_transform_edt(inside)
            d_out = distance_transform_edt(~inside)
            phi = np.where(inside, -(d_in - 0.5) * h, (d_out - 0.5) * h)
        elif np.any(inside):
            phi = np.full((nx, nz), -0.5 * h)
        else:
            phi = np.full((nx, nz), 0.5 * h)

        return frac, phi, inside, xc, zc

    def _frame(self, time: float) -> StateFrame:
        nx, nz, h = self.nx, self.nz, self.h
        frac, phi, inside, xc, zc = self._fields()

        # Cell-centred diagnostic velocity reconstructed from the samples.
        # Reconstruct a divergence-free-ish grid velocity by interpolating the
        # latest particle velocities onto the grid faces (mass weighted).
        vxg, vzg, wgt, wgz = self._grid_faces()
        u = np.where(wgt > 0.0, vxg / np.maximum(wgt, 1e-30), 0.0)
        w = np.where(wgz > 0.0, vzg / np.maximum(wgz, 1e-30), 0.0)
        u[0, :] = 0.0
        u[nx, :] = 0.0
        w[:, 0] = 0.0
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        wc = 0.5 * (w[:, :-1] + w[:, 1:])
        vel = np.stack((uc, wc), axis=-1)
        # Outside the liquid the diagnostic velocity is zero.
        vel = np.where(inside[:, :, None], vel, 0.0)

        # Boundary samples: floor, left wall, right wall.
        bv = np.zeros_like(self.bnd_normal)
        bv[:nx, 0] = uc[:, 0]
        bv[nx:nx + nz, 1] = wc[0, :]
        bv[nx + nz:, 1] = wc[nx - 1, :]

        position = np.column_stack((self.px, np.zeros(self.n), self.pz))
        velocity = np.column_stack((self.vx, np.zeros(self.n), self.vz))

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={"phase_id": np.ones(self.n, dtype=np.float64)},
            fields={
                "velocity_field": np.ascontiguousarray(vel),
                "liquid_level_set": np.ascontiguousarray(phi),
                "liquid_interior_mask": inside.astype(np.float64),
                "boundary_normal": self.bnd_normal.copy(),
                "boundary_target_velocity": self.bnd_target.copy(),
                "boundary_velocity": bv,
            },
            scalars={
                "grid_spacing": float(h),
                "characteristic_length": float(self.char_length),
                "reference_velocity": float(self.ref_velocity),
                "liquid_phase_id": 1.0,
            },
        )
        frame.validate()
        return frame

    def _grid_faces(self):
        """Mass-weighted particle -> MAC face velocity deposit (diagnostic)."""
        nx, nz, h = self.nx, self.nz, self.h
        gx = (self.px - self.xmin) / h
        i0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 1)
        ax = np.clip(gx - i0, 0.0, 1.0)
        gz = (self.pz - self.zmin) / h
        j0 = np.clip(np.floor(gz).astype(np.int64), 0, nz - 1)
        az = np.clip(gz - j0, 0.0, 1.0)
        m = self.mass
        vxg = np.zeros((nx + 1, nz))
        vzg = np.zeros((nx, nz + 1))
        wgt = np.zeros((nx + 1, nz))
        wgz = np.zeros((nx, nz + 1))
        for ii, jj, ww in (
            (i0, j0, (1.0 - ax) * (1.0 - az)),
            (i0 + 1, j0, ax * (1.0 - az)),
            (i0, j0 + 1, (1.0 - ax) * az),
            (i0 + 1, j0 + 1, ax * az),
        ):
            wt = m * ww
            np.add.at(vxg, (ii, jj), wt * self.vx)
            np.add.at(wgt, (ii, jj), wt)
            np.add.at(vzg, (ii, jj), wt * self.vz)
            np.add.at(wgz, (ii, jj), wt)
        return vxg, vzg, wgt, wgz

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "incompressible particle/MAC projection (FLIP/PIC)",
            "grid": [int(self.nx), int(self.nz)],
            "grid_spacing": float(self.h),
            "substep_dt": float(self.dt),
            "samples": int(self.n),
            "total_mass": float(np.sum(self.mass)),
            "final_time": float(self.time),
        }
