"""Weakly-compressible two-phase SPH solver for the octagonal two-phase drop.

Physics
-------
A heavy blue liquid (rho = 1000 kg/m^3) is released above an open octagonal
basin at t = 0 and falls, impacts the floor and pools.  At the configured
release time a light pink liquid (rho = 250 kg/m^3, 4x lighter) is released
above the pool, impacts, spreads and buoyantly rearranges until the phases
separate with the light liquid floating on the heavy one.

Method: weakly compressible SPH with a cubic-spline kernel,
  * density summation rho_i = sum_j m_j W_ij,
  * Tait equation of state p = p0 ((rho/rho0)^7 - 1) per phase,
  * symmetric momentum equation -m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W,
  * Monaghan artificial viscosity (handles impact) + XSPH smoothing,
  * gravity, impermeable octagonal wall/floor via penalty force plus hard
    projection clamp, open top,
  * staged activation: light-phase particles are kinematically held at their
    spawn cloud until their release time, then join the dynamics.

All required trajectory fields are exported every sample:
  attributes.mass_quantity [N,2], fields.active_mask,
  fields.cumulative_mass_source [2], fields.liquid_level_set,
  fields.velocity_field, scalars.characteristic_length, scalars.grid_spacing.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_SQRT2 = float(np.sqrt(2.0))


def _lattice_ellipsoid(center, radii, spacing):
    center = np.asarray(center, float)
    radii = np.asarray(radii, float)
    axes = [np.arange(-r + spacing / 2.0, r, spacing) for r in radii]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = grid[np.sum((grid / radii) ** 2, axis=1) <= 1.0]
    return pts + center


def _even_subset(points, count):
    if len(points) >= count:
        idx = np.linspace(0, len(points) - 1, count).astype(int)
        return np.asarray(points, float)[idx]
    return np.asarray(points, float)


def _cloud(count, center, radii):
    """Uniform lattice cloud of exactly `count` points inside the ellipsoid.

    The lattice spacing is refined until the fully occupied lattice contains at
    least `count` points; a deterministic even subset then yields exactly
    `count`.  The spacing is chosen close to (nominal_volume / count)^(1/3) so
    the SPH density summation reproduces the phase reference density at t=0.
    """
    center = np.asarray(center, float)
    radii = np.asarray(radii, float)
    ell_vol = (4.0 / 3.0) * math.pi * float(np.prod(radii))
    s0 = (ell_vol / max(count, 1)) ** (1.0 / 3.0)
    pts = _lattice_ellipsoid(center, radii, s0)
    s = s0
    for k in range(1, 60):
        if len(pts) >= count:
            break
        s = s0 * (1.0 - 0.01 * k)
        pts = _lattice_ellipsoid(center, radii, s)
    return _even_subset(pts, count)


class BenchmarkSolver:
    """Two-phase WCSPH solver for the staged octagonal drop."""

    # ---- frozen basin geometry (from assets/octagonal_basin.obj) ----
    R_IN = 1.060660   # inner apothem of the octagon (m)
    Z_FLOOR = 0.08    # basin floor height (m)
    Z_TOP = 0.62      # rim height, open above (m)

    # ---- numerics ----
    C0 = 30.0         # artificial sound speed (m/s)
    GAMMA = 7.0
    DT = 8.0e-4       # physics time step (s)
    ALPHA_V = 0.08    # artificial viscosity, linear
    BETA_V = 0.15     # artificial viscosity, quadratic
    XSPH_EPS = 0.15   # XSPH advection smoothing
    V_CAP = 15.0      # velocity safety cap (m/s)
    K_PEN = 2.5e5     # boundary penalty stiffness (1/s^2)
    C_PEN = 200.0     # boundary penalty damping (1/s)

    # ---- diagnostic grid ----
    DX = 0.06
    GX = np.arange(-1.2, 1.2 + 1e-9, 0.06)
    GZ = np.arange(0.0, 0.78 + 1e-9, 0.06)
    H_G = 1.8 * DX

    def __init__(self):
        self._t = 0.0
        self._steps = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        att = frame0.attributes
        self.phase = np.asarray(att["phase_id"], float)
        self.rho0 = np.asarray(att["density"], float)
        self.rel_t = np.asarray(att["release_time"], float)
        self.vol = np.asarray(att["target_volume"], float)
        self.mass = np.asarray(frame0.mass, float)
        self.n = len(self.phase)
        self.n_heavy = int(np.count_nonzero(self.phase < 0.5))
        self.n_light = self.n - self.n_heavy
        light = self.phase > 0.5
        self.release_time = float(self.rel_t[light].min()) if light.any() else 0.0
        self.light_mass_total = float(self.mass[light].sum())

        # Rebuild both drop clouds as uniform ellipsoidal lattices matching the
        # authored particle_domain bounds so the SPH reference densities are
        # reproduced by density summation at t = 0.
        blue = _cloud(self.n_heavy, (0.0, 0.0, 2.15), (0.43, 0.43, 0.50))
        pink = _cloud(self.n_light, (0.05, 0.0, 3.62), (0.37, 0.37, 0.40))
        self.pos = np.vstack((blue, pink))
        self.vel = np.zeros_like(self.pos)
        self.vel[: self.n_heavy, 2] = -0.35
        self.v_half = self.vel.copy()
        self.active = self.rel_t <= 1e-12

        dp = (float(self.vol[0].mean())) ** (1.0 / 3.0)
        self.h = 1.9 * dp
        self.inv_h = 1.0 / self.h
        self.sigma_k = 1.0 / (math.pi * self.h ** 3)
        self.p0 = self.rho0 * self.C0 ** 2 / self.GAMMA

        self._t = 0.0
        self._steps = 0
        self._init_grid()

    def _init_grid(self):
        nx, nz = len(self.GX), len(self.GZ)
        self.grid_shape = (nx, nx, nz)
        cx, cy, cz = np.meshgrid(self.GX, self.GX, self.GZ, indexing="ij")
        self.gxc = cx.ravel()
        self.gyc = cy.ravel()
        self.gzc = cz.ravel()
        s_oct = self.R_IN - np.maximum.reduce(
            [np.abs(cx), np.abs(cy),
             np.abs(cx + cy) / _SQRT2, np.abs(cx - cy) / _SQRT2]
        )
        mask = (
            (s_oct >= -0.5 * self.DX)
            & (cz >= self.Z_FLOOR - 0.5 * self.DX)
            & (cz <= self.Z_TOP + 0.5 * self.DX)
        )
        self.mask3 = mask
        self.mask_flat = mask.ravel()
        self.active_mask = mask.astype(np.float64)
        self.cum_source = np.zeros(2)

    # ----------------------------------------------------------------- kernel
    def _w(self, r):
        q = r * self.inv_h
        return self.sigma_k * np.where(q <= 1.0, 1.0 - 1.5 * q * q + 0.75 * q ** 3, 0.0)

    def _dwdq(self, r):
        q = r * self.inv_h
        return np.where(q <= 1.0, q * (-4.5 + 2.25 * q), 0.0)

    def _oct_s(self, x, y):
        return self.R_IN - np.maximum.reduce(
            [np.abs(x), np.abs(y),
             np.abs(x + y) / _SQRT2, np.abs(x - y) / _SQRT2]
        )

    # -------------------------------------------------------------- dynamics
    def _accelerate(self):
        """One force evaluation + leapfrog step of size DT (activates releases)."""
        dt = self.DT
        if not self.active.all():
            newly = (~self.active) & (self.rel_t <= self._t + 1e-9)
            if newly.any():
                self.active |= newly
                self.v_half[newly] = self.vel[newly]

        act = self.active
        idx = np.flatnonzero(act)
        pos = self.pos
        vel = self.vel
        m = self.mass
        v = self.vol

        tree = cKDTree(pos[idx])
        pairs = tree.query_pairs(2.0 * self.h, output_type="ndarray")
        if pairs.size:
            I = idx[pairs[:, 0]]
            J = idx[pairs[:, 1]]
            dvec = pos[I] - pos[J]
            r = np.sqrt((dvec * dvec).sum(1)) + 1e-12
        else:
            I = J = np.empty(0, dtype=np.intp)
            dvec = np.empty((0, 3))
            r = np.empty(0)

        # --- density summation ---
        rho = self.rho0.copy()
        if I.size:
            w = self._w(r)
            rho += np.bincount(I, weights=m[J] * w, minlength=self.n)
            rho += np.bincount(J, weights=m[I] * w, minlength=self.n)

        ratio = rho / self.rho0
        pres = (self.p0 / self.GAMMA) * (ratio ** self.GAMMA - 1.0)
        pres = np.maximum(pres, -0.05 * self.p0) * act

        a = np.zeros_like(pos)
        a[act, 2] = -9.81

        if I.size:
            # --- Monaghan artificial viscosity ---
            dvv = vel[I] - vel[J]
            vdotr = (dvv * dvec).sum(1)
            Pi = np.zeros_like(r)
            neg = vdotr < 0.0
            if neg.any():
                mu = self.h * vdotr[neg] / (r[neg] ** 2 + 0.01 * self.h ** 2)
                mup = np.minimum(mu, 10.0 * self.C0)
                rho_bar = 0.5 * (rho[I[neg]] + rho[J[neg]])
                Pi[neg] = (-self.ALPHA_V * self.C0 * mup
                           + self.BETA_V * mup * mup) / rho_bar
            # --- pressure + viscosity gradient ---
            gsc = self._dwdq(r) * self.sigma_k * self.inv_h / r
            coef = pres[I] / rho[I] ** 2 + pres[J] / rho[J] ** 2 - Pi
            fij = -(m[J] * coef * gsc)[:, None] * dvec
            np.add.at(a, I, fij)
            np.add.at(a, J, -fij * (m[I] / m[J])[:, None])

            # --- XSPH velocity smoothing ---
            rho_bar_all = 0.5 * (rho[I] + rho[J])
            wv = (m[J] / rho_bar_all) * w
            dvel = vel[J] - vel[I]
            vx = np.zeros_like(pos)
            for k in range(3):
                vx[:, k] = np.bincount(I, weights=wv * dvel[:, k], minlength=self.n)
            vel[act] += self.XSPH_EPS * vx[act]

        # --- boundary penalty: floor + octagonal walls (below rim) ---
        z = pos[:, 2]
        inside = z < self.Z_TOP
        x = pos[:, 0]
        y = pos[:, 1]
        s_oct = self._oct_s(x, y)
        pen_f = np.where(inside & (z < self.Z_FLOOR), self.Z_FLOOR - z, 0.0)
        pen_w = np.where(inside & (s_oct < 0.0), -s_oct, 0.0)
        # floor
        sel = pen_f > 0.0
        if sel.any():
            a[sel, 2] += self.K_PEN * pen_f[sel] - self.C_PEN * np.minimum(vel[sel, 2], 0.0)
        # wall (radial inward normal approximation)
        sel = pen_w > 0.0
        if sel.any():
            rr = np.hypot(x[sel], y[sel])
            rr = np.where(rr < 1e-9, 1e-9, rr)
            nxn = x[sel] / rr
            nyn = y[sel] / rr
            vn = vel[sel, 0] * nxn + vel[sel, 1] * nyn
            f = self.K_PEN * pen_w[sel] - self.C_PEN * np.minimum(vn, 0.0)
            a[sel, 0] += f * nxn
            a[sel, 1] += f * nyn

        # --- leapfrog kick-drift ---
        np.clip(a, -5.0e4, 5.0e4, out=a)
        self.v_half += dt * a
        np.clip(self.v_half, -self.V_CAP, self.V_CAP, out=self.v_half)
        pos += dt * self.v_half
        self.vel = 0.5 * (self.vel + self.v_half)

        # --- hard projection clamp (impermeable basin) ---
        self._clamp()

        self._t += dt
        self._steps += 1

    def _clamp(self):
        pos = self.pos
        vel = self.vel
        z = pos[:, 2]
        low = z < self.Z_FLOOR
        if low.any():
            pos[low, 2] = self.Z_FLOOR
            np.maximum(vel[low, 2], 0.0, out=vel[low, 2])
        inside = pos[:, 2] < self.Z_TOP
        s = self._oct_s(pos[:, 0], pos[:, 1])
        bad = inside & (s < -1.0e-4)
        if bad.any():
            x = pos[bad, 0]
            y = pos[bad, 1]
            rmax = np.maximum.reduce(
                [np.abs(x), np.abs(y),
                 np.abs(x + y) / _SQRT2, np.abs(x - y) / _SQRT2]
            )
            scale = self.R_IN / np.maximum(rmax, 1e-9)
            pos[bad, 0] = x * scale
            pos[bad, 1] = y * scale
            rr = np.maximum(np.hypot(x, y), 1e-9)
            nxn = x / rr
            nyn = y / rr
            vn = vel[bad, 0] * nxn + vel[bad, 1] * nyn
            out = vn > 0.0
            if out.any():
                vel[bad, 0] -= np.where(out, vn, 0.0) * nxn
                vel[bad, 1] -= np.where(out, vn, 0.0) * nyn
        np.clip(pos[:, 0], -1.25, 1.25, out=pos[:, 0])
        np.clip(pos[:, 1], -1.25, 1.25, out=pos[:, 1])
        np.clip(pos[:, 2], -0.05, 5.0, out=pos[:, 2])

    # ------------------------------------------------------- diagnostic grid
    def _grid_fields(self):
        h = self.H_G
        sigma = 1.0 / (math.pi * h ** 3)
        nx = len(self.GX)
        nz = len(self.GZ)
        off = np.array([(i, j, k) for i in (-1, 0, 1)
                        for j in (-1, 0, 1) for k in (-1, 0, 1)])
        mcell = int(round(h / self.DX))
        ncells = nx * nx * nz
        num = np.zeros(ncells)
        wsum = np.zeros(ncells)
        mv = [np.zeros(ncells) for _ in range(3)]
        idx = np.flatnonzero(self.active)
        for k in range(3):
            mv[k][...] = 0.0
        if idx.size:
            P = self.pos[idx]
            M = self.mass[idx]
            R = self.rho0[idx]
            V = self.vel[idx]
            bi = np.rint((P[:, 0] - self.GX[0]) / self.DX).astype(int)
            bj = np.rint((P[:, 1] - self.GX[0]) / self.DX).astype(int)
            bk = np.rint((P[:, 2] - self.GZ[0]) / self.DX).astype(int)
            for di in range(-mcell, mcell + 1):
                ii = bi + di
                ok_i = (ii >= 0) & (ii < nx)
                if not ok_i.any():
                    continue
                for dj in range(-mcell, mcell + 1):
                    jj = bj + dj
                    ok_j = ok_i & (jj >= 0) & (jj < nx)
                    if not ok_j.any():
                        continue
                    for dk in range(-mcell, mcell + 1):
                        kk = bk + dk
                        ok = ok_j & (kk >= 0) & (kk < nz)
                        if not ok.any():
                            continue
                        flat = (ii[ok] * nx + jj[ok]) * nz + kk[ok]
                        gx = self.GX[ii[ok]]
                        gy = self.GX[jj[ok]]
                        gz = self.GZ[kk[ok]]
                        p = P[ok]
                        rx = gx - p[:, 0]
                        ry = gy - p[:, 1]
                        rz = gz - p[:, 2]
                        d2 = rx * rx + ry * ry + rz * rz
                        sel = d2 < (h * h)
                        if not sel.any():
                            continue
                        flat = flat[sel]
                        q = np.sqrt(d2[sel]) * (1.0 / h)
                        w = sigma * (1.0 - 1.5 * q * q + 0.75 * q ** 3)
                        mw = M[ok][sel] / R[ok][sel] * w
                        num += np.bincount(flat, weights=w, minlength=ncells)
                        wsum += np.bincount(flat, weights=mw, minlength=ncells)
                        vv = V[ok][sel]
                        for c in range(3):
                            mv[c] += np.bincount(flat, weights=mw * vv[:, c],
                                                 minlength=ncells)

        dp = (float(self.vol[0].mean())) ** (1.0 / 3.0)
        n_ref = 1.0 / dp ** 3
        chi = num / n_ref
        level = np.zeros(ncells)
        level[self.mask_flat] = (3.0 * self.DX) * (
            np.minimum(chi[self.mask_flat], 3.0) - 0.5
        )
        vel_field = np.zeros((ncells, 3))
        np.divide(mv[0], wsum, out=vel_field[:, 0], where=wsum > 1e-12)
        np.divide(mv[1], wsum, out=vel_field[:, 1], where=wsum > 1e-12)
        np.divide(mv[2], wsum, out=vel_field[:, 2], where=wsum > 1e-12)
        vel_field[~self.mask_flat] = 0.0
        return (vel_field.reshape(self.grid_shape + (3,)),
                level.reshape(self.grid_shape))

    # ----------------------------------------------------------------- frame
    def _make_frame(self):
        vel_field, level = self._grid_fields()
        if self._t >= self.release_time - 1e-9:
            self.cum_source = np.array([0.0, self.light_mass_total])
        else:
            self.cum_source = np.zeros(2)
        mass_quantity = np.zeros((self.n, 2))
        mass_quantity[:, 0] = np.where(self.phase < 0.5, self.mass, 0.0)
        mass_quantity[:, 1] = np.where(self.phase > 0.5, self.mass, 0.0)
        frame = StateFrame(
            time=float(self._t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.phase.astype(np.int32),
            attributes={
                "phase_id": self.phase.copy(),
                "density": self.rho0.copy(),
                "release_time": self.rel_t.copy(),
                "target_volume": self.vol.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "active_mask": self.active_mask,
                "cumulative_mass_source": self.cum_source.copy(),
                "liquid_level_set": level,
                "velocity_field": vel_field,
            },
            scalars={
                "characteristic_length": float(self.R_IN),
                "grid_spacing": float(self.DX),
                "liquid_phase_id": 0.0,
                "reference_velocity": math.sqrt(9.81 * 1.8),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float):
        time = float(time)
        if time < self._t - 1e-9:
            raise ValueError("advance_to called with a time in the past")
        guard = 0
        while self._t < time - 1e-7:
            self._accelerate()
            guard += 1
            if guard > 200000:
                break
        self._t = time
        return self._make_frame()

    def diagnostics(self) -> dict:
        return {
            "method": "two-phase WCSPH (Tait EOS, Monaghan viscosity, XSPH)",
            "particles": int(self.n),
            "heavy_particles": self.n_heavy,
            "light_particles": self.n_light,
            "steps": self._steps,
            "dt": self.DT,
            "time": float(self._t),
            "smoothing_length": float(self.h),
        }
