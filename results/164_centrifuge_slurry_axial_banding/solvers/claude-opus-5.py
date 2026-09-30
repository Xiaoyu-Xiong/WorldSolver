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
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical / numerical closure constants
# ---------------------------------------------------------------------------
RHO_FLUID = 1000.0        # carrier liquid density [kg/m^3]
MU_SLURRY = 8.0           # effective (hindered) suspension viscosity [Pa s]
MU_AIR = 1.0e-4           # drag in the drained air core [Pa s]
K_N = 1.5e4               # particle-particle normal stiffness [N/m]
K_W = 3.0e4               # particle-wall normal stiffness [N/m]
ZETA_P = 0.35             # contact damping ratio (particle-particle)
ZETA_W = 0.40             # contact damping ratio (particle-wall)
MU_F = 0.12               # lubricated inter-particle friction
MU_W = 0.35               # wall friction (spins the slurry up)
CT_MAX = 400.0            # tangential viscous regularisation [N s/m]
DT_MAX = 5.0e-4           # integration step ceiling [s]
REBUILD_EVERY = 10        # neighbour-list rebuild cadence [steps]
SKIN = 0.05               # neighbour-list skin [m]
LIQ_EVERY = 20            # liquid (Darcy) update cadence [steps]
SPIN_UP = 0.8             # drum angular-velocity ramp time [s]
V_CLIP = 12.0             # velocity safety clip [m/s]

# Cahn-Hilliard style axial segregation closure -----------------------------
SEG_A = 1.0               # de-mixing (anti-diffusive) coefficient
SEG_B = 1.0               # quartic saturation coefficient
SEG_KAPPA = 0.0127        # gradient penalty -> band wavelength ~1 m
SEG_PRESS = 2.0e3         # particle-pressure scale [Pa per unit potential]
SEG_CLIP = 4.0            # clip on the reduced chemical-potential gradient
PACK_MIN = 1.0            # contacts needed before axial migration activates
PACK_SPAN = 4.0


class BenchmarkSolver:
    """Soft-sphere DEM slurry in a rotating horizontal drum.

    Physics integrated in time:

    * prescribed drum rotation about the +x axis (ramped from rest),
    * gravity plus the pressure-gradient (buoyancy + centrifugal) force of a
      co-rotating carrier liquid whose free surface is the iso-surface of
      ``Phi = g z - 0.5 w^2 r^2`` matched to the local liquid volume,
    * Stokes drag against the local liquid velocity (swirl + Darcy axial flow),
    * linear spring-dashpot + regularised Coulomb contacts between particles,
      against the cylindrical wall and against the end caps,
    * a resolved, momentum-conserving axial migration flux driven by the
      gradient of a composition-dependent particle chemical potential; this is
      the de-mixing instability that produces the axial bands.
    """

    # -- setup --------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        cent = dict(cfg.get("centrifuge", {}))
        self.radius = float(cent.get("radius", 0.92))
        self.length = float(cent.get("length", 4.8))
        self.zc = float(cent.get("center_height", 1.05))
        self.omega0 = float(cent.get("angular_velocity_rad_s", 5.5))
        self.fill = float(cent.get("fill_fraction", 0.68))
        self.rho_p = float(cent.get("particle_density", 1450.0))
        self.h = float(cent.get("spacing", 0.12))
        self.species_count = max(1, int(cent.get("species_count", 2)))
        grav = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.gvec = grav
        self.g = float(abs(grav[2])) if grav[2] != 0.0 else 9.81
        self.sample_dt = float(cfg.get("sample_dt", 0.04))
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.n = int(self.pos.shape[0])
        self.rad = np.asarray(frame0.attributes["particle_radius"], dtype=float).copy()
        self.species = np.asarray(frame0.attributes["species_id"], dtype=np.int32).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.vol = self.mass / self.rho_p
        self.mass_quantity = np.asarray(
            frame0.attributes["mass_quantity"], dtype=float
        ).copy()
        self.body_id = np.zeros(self.n, dtype=np.int32)
        self.sgn = (2.0 * self.species.astype(float) - 1.0)   # small -1, large +1
        self.drag_b_liq = 6.0 * np.pi * MU_SLURRY * self.rad
        self.drag_b_air = 6.0 * np.pi * MU_AIR * self.rad
        self.scalars = {
            "grid_spacing": float(self.h),
            "characteristic_length": float(2.0 * self.radius),
            "contact_length_scale": float(self.h),
        }

        shape = tuple(int(v) for v in cent.get("field_grid_shape", (40, 18, 18)))
        self.gshape = shape
        self.nx, self.ny, self.nz = shape
        self.x0 = -0.5 * self.nx * self.h
        self.y0 = -0.5 * self.ny * self.h
        self.z0 = self.zc - 0.5 * self.nz * self.h
        self._build_grid()
        self._build_liquid()
        self._build_disk_samples()

        self.time = 0.0
        self.step_count = 0
        self.omega = 0.0
        self.contact_count = np.zeros(self.n, dtype=float)
        self.f_ext = np.zeros((self.n, 3), dtype=float)
        self.f_int = np.zeros((self.species_count, 3), dtype=float)
        self._phi_cache_omega = None
        self._phi_sorted = None
        self._rebuild_pairs()
        self._update_liquid()
        self.f_ext[:] = self.mass[:, None] * self.gvec[None, :]

    # -- diagnostic grid ----------------------------------------------------
    def _build_grid(self) -> None:
        h = self.h
        xc = self.x0 + (np.arange(self.nx) + 0.5) * h
        yc = self.y0 + (np.arange(self.ny) + 0.5) * h
        zc = self.z0 + (np.arange(self.nz) + 0.5) * h
        self.gx_c, self.gy_c, self.gz_c = np.meshgrid(xc, yc, zc, indexing="ij")
        self.g_zrel = self.gz_c - self.zc
        self.g_r2 = self.gy_c ** 2 + self.g_zrel ** 2
        self.g_inside = self.g_r2 <= self.radius ** 2
        self.n_cells = int(self.nx * self.ny * self.nz)

    def _build_liquid(self) -> None:
        """Axial liquid bookkeeping (bins along the drum axis)."""
        self.nb = self.nx
        self.dxb = self.length / self.nb
        area = np.pi * self.radius ** 2
        self.bin_volume = area * self.dxb
        v_slurry = self.fill * area * self.length
        v_solid_tot = float(self.vol.sum())
        self.v_liquid_total = max(1e-6, v_slurry - v_solid_tot)
        self.v_liq = np.full(self.nb, self.v_liquid_total / self.nb)
        self.u_axial = np.zeros(self.nb)
        self.phi_free = np.zeros(self.nb)
        self.d_bin = np.full(self.nb, 2.0 * float(self.rad.mean()))
        self.v_sol_bin = np.zeros(self.nb)

    def _build_disk_samples(self, m: int = 180) -> None:
        R = self.radius
        s = np.linspace(-R, R, m)
        yy, zz = np.meshgrid(s, s, indexing="ij")
        inside = yy ** 2 + zz ** 2 <= R ** 2
        self.sy = yy[inside]
        self.sz = zz[inside]
        self.disk_area = np.pi * R ** 2
        self.n_samples = int(self.sy.size)
        self.sample_area = self.disk_area / max(1, self.n_samples)

    # -- kinematics helpers -------------------------------------------------
    def _omega_at(self, t: float) -> float:
        u = min(1.0, max(0.0, t / SPIN_UP)) if SPIN_UP > 0.0 else 1.0
        return self.omega0 * (u * u * (3.0 - 2.0 * u))

    def _potential(self, y, zrel, omega):
        """Effective potential of the co-rotating liquid, lab frame."""
        return self.g * zrel - 0.5 * omega * omega * (y * y + zrel * zrel)

    def _phi_sorted_for(self, omega: float):
        if self._phi_cache_omega is not None and abs(omega - self._phi_cache_omega) < 1e-9:
            return self._phi_sorted
        phi = self._potential(self.sy, self.sz, omega)
        self._phi_sorted = np.sort(phi)
        self._phi_cache_omega = float(omega)
        return self._phi_sorted

    def _free_surface_potentials(self, omega: float) -> np.ndarray:
        """Per-bin iso-potential level that encloses the local mixture volume."""
        phi_sorted = self._phi_sorted_for(omega)
        v_mix = self.v_liq + self.v_sol_bin
        frac = np.clip(v_mix / max(1e-9, self.bin_volume), 1e-4, 1.0)
        idx = np.clip((frac * (self.n_samples - 1)).astype(np.int64), 0,
                      self.n_samples - 1)
        return phi_sorted[idx]
    def _bin_index(self) -> np.ndarray:
        idx = np.floor((self.pos[:, 0] + 0.5 * self.length) / self.dxb).astype(np.int64)
        return np.clip(idx, 0, self.nb - 1)

    # -- neighbour list -----------------------------------------------------
    def _rebuild_pairs(self) -> None:
        cutoff = 2.0 * float(self.rad.max()) + SKIN
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            self.pi = np.zeros(0, dtype=np.int64)
            self.pj = np.zeros(0, dtype=np.int64)
        else:
            self.pi = np.ascontiguousarray(pairs[:, 0].astype(np.int64))
            self.pj = np.ascontiguousarray(pairs[:, 1].astype(np.int64))
        self.pair_rsum = self.rad[self.pi] + self.rad[self.pj]
        mi = self.mass[self.pi]
        mj = self.mass[self.pj]
        meff = mi * mj / np.maximum(mi + mj, 1e-12)
        self.pair_cn = 2.0 * ZETA_P * np.sqrt(K_N * meff)

    # -- carrier liquid (axial Darcy redistribution) ------------------------
    def _update_liquid(self, dt_liq: float = 0.0) -> None:
        idx = self._bin_index()
        self.v_sol_bin = np.bincount(idx, weights=self.vol, minlength=self.nb)
        dnum = np.bincount(idx, weights=self.vol * 2.0 * self.rad, minlength=self.nb)
        dmean = 2.0 * float(self.rad.mean())
        self.d_bin = np.where(self.v_sol_bin > 1e-9,
                              dnum / np.maximum(self.v_sol_bin, 1e-12), dmean)
        self.phi_free = self._free_surface_potentials(self.omega)
        if dt_liq <= 0.0:
            return
        v_mix = np.maximum(self.v_liq + self.v_sol_bin, 1e-9)
        eps = np.clip(1.0 - self.v_sol_bin / v_mix, 0.2, 0.95)
        perm = self.d_bin ** 2 * eps ** 3 / (180.0 * np.maximum((1.0 - eps) ** 2, 1e-6))
        a_mix = v_mix / self.dxb
        kf = 0.5 * (perm[:-1] + perm[1:])
        af = 0.5 * (a_mix[:-1] + a_mix[1:])
        dphi = (self.phi_free[1:] - self.phi_free[:-1]) / self.dxb
        q = -(kf / MU_SLURRY) * af * RHO_FLUID * dphi
        q = np.clip(q, -0.5 * af, 0.5 * af)
        dv = np.zeros(self.nb)
        dv[:-1] -= q * dt_liq
        dv[1:] += q * dt_liq
        v_new = self.v_liq + dv
        vmin = 0.02 * self.v_liquid_total / self.nb
        cap = np.maximum(self.bin_volume - self.v_sol_bin, vmin)
        v_new = np.clip(v_new, vmin, cap)
        total = float(v_new.sum())
        if total > 1e-9:
            v_new *= self.v_liquid_total / total
        self.v_liq = v_new
        qc = np.zeros(self.nb)
        if self.nb > 2:
            qc[1:-1] = 0.5 * (q[:-1] + q[1:])
            qc[0] = q[0]
            qc[-1] = q[-1]
        self.u_axial = np.clip(qc / np.maximum(a_mix * eps, 1e-6), -0.3, 0.3)
    # -- contacts -----------------------------------------------------------
    def _pair_forces(self) -> np.ndarray:
        f = np.zeros((self.n, 3), dtype=float)
        self.contact_count = np.zeros(self.n, dtype=float)
        if self.pi.size == 0:
            return f
        d = self.pos[self.pj] - self.pos[self.pi]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        overlap = self.pair_rsum - dist
        act = overlap > 0.0
        if not np.any(act):
            return f
        i = self.pi[act]
        j = self.pj[act]
        dist_a = np.maximum(dist[act], 1e-9)
        nrm = d[act] / dist_a[:, None]
        ov = overlap[act]
        dv = self.vel[self.pj[act]] - self.vel[self.pi[act]]
        vn = np.einsum("ij,ij->i", dv, nrm)
        fn = np.maximum(K_N * ov - self.pair_cn[act] * vn, 0.0)
        fvec = -fn[:, None] * nrm            # force on i (repulsive, along -n)
        vt = dv - vn[:, None] * nrm
        vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        ft_mag = np.minimum(CT_MAX * vt_mag, MU_F * fn)
        safe = vt_mag > 1e-9
        ft = np.zeros_like(vt)
        ft[safe] = (ft_mag[safe] / vt_mag[safe])[:, None] * vt[safe]
        fi = fvec + ft                       # total force on particle i
        for k in range(3):
            f[:, k] += np.bincount(i, weights=fi[:, k], minlength=self.n)
            f[:, k] -= np.bincount(j, weights=fi[:, k], minlength=self.n)
        ones = np.ones(i.size)
        self.contact_count = (np.bincount(i, weights=ones, minlength=self.n)
                              + np.bincount(j, weights=ones, minlength=self.n))
        return f

    def _boundary_forces(self) -> np.ndarray:
        f = np.zeros((self.n, 3), dtype=float)
        y = self.pos[:, 1]
        zr = self.pos[:, 2] - self.zc
        r = np.sqrt(y * y + zr * zr)
        pen = r + self.rad - self.radius
        cw = 2.0 * ZETA_W * np.sqrt(K_W * self.mass)
        act = pen > 0.0
        if np.any(act):
            rs = np.maximum(r[act], 1e-9)
            ny = y[act] / rs
            nz = zr[act] / rs
            vn = self.vel[act, 1] * ny + self.vel[act, 2] * nz
            fn = np.maximum(K_W * pen[act] - cw[act] * vn, 0.0)
            f[act, 1] -= fn * ny
            f[act, 2] -= fn * nz
            # wall material velocity: rotation about +x through the drum axis
            wy = -self.omega * zr[act]
            wz = self.omega * y[act]
            rvx = self.vel[act, 0]
            rvy = self.vel[act, 1] - wy
            rvz = self.vel[act, 2] - wz
            rvn = rvy * ny + rvz * nz
            ty = rvy - rvn * ny
            tz = rvz - rvn * nz
            tmag = np.sqrt(rvx * rvx + ty * ty + tz * tz)
            ftm = np.minimum(CT_MAX * tmag, MU_W * fn)
            scale = np.where(tmag > 1e-9, ftm / np.maximum(tmag, 1e-9), 0.0)
            f[act, 0] -= scale * rvx
            f[act, 1] -= scale * ty
            f[act, 2] -= scale * tz
        # end caps
        half = 0.5 * self.length
        for sign in (-1.0, 1.0):
            pen_x = self.rad - (half - sign * self.pos[:, 0])
            ax = pen_x > 0.0
            if not np.any(ax):
                continue
            vn = sign * self.vel[ax, 0]
            fn = np.maximum(K_W * pen_x[ax] - cw[ax] * vn, 0.0)
            f[ax, 0] -= sign * fn
            wy = -self.omega * (self.pos[ax, 2] - self.zc)
            wz = self.omega * self.pos[ax, 1]
            rvy = self.vel[ax, 1] - wy
            rvz = self.vel[ax, 2] - wz
            tmag = np.sqrt(rvy * rvy + rvz * rvz)
            ftm = np.minimum(CT_MAX * tmag, MU_W * fn)
            scale = np.where(tmag > 1e-9, ftm / np.maximum(tmag, 1e-9), 0.0)
            f[ax, 1] -= scale * rvy
            f[ax, 2] -= scale * rvz
        return f
    # -- composition-driven axial migration (de-mixing instability) ---------
    def _segregation_force(self, idx: np.ndarray, subm: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n, 3), dtype=float)
        if self.species_count < 2:
            return f
        small = self.species == 0
        large = ~small
        v_s = np.bincount(idx[small], weights=self.vol[small], minlength=self.nb)
        v_l = np.bincount(idx[large], weights=self.vol[large], minlength=self.nb)
        tot = v_s + v_l
        c = np.where(tot > 1e-9, (v_l - v_s) / np.maximum(tot, 1e-12), 0.0)
        # light smoothing of the coarse-grained composition field
        cs = c.copy()
        cs[1:-1] = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
        lap = np.zeros(self.nb)
        lap[1:-1] = (cs[2:] - 2.0 * cs[1:-1] + cs[:-2]) / (self.dxb ** 2)
        mu = -SEG_A * cs + SEG_B * cs ** 3 - SEG_KAPPA * lap
        grad = np.zeros(self.nb)
        grad[1:-1] = (mu[2:] - mu[:-2]) / (2.0 * self.dxb)
        grad = np.clip(grad, -SEG_CLIP, SEG_CLIP)
        pack = np.clip((self.contact_count - PACK_MIN) / PACK_SPAN, 0.0, 1.0)
        gate = pack * subm
        f[:, 0] = -SEG_PRESS * self.vol * self.sgn * grad[idx] * gate
        # remove net momentum injection: an internal migration flux
        mtot = float(self.mass.sum())
        if mtot > 1e-12:
            f[:, 0] -= self.mass * (float(f[:, 0].sum()) / mtot)
        return f

    # -- one explicit / semi-implicit step ----------------------------------
    def _step(self, dt: float) -> None:
        if self.step_count % REBUILD_EVERY == 0:
            self._rebuild_pairs()
        self.omega = self._omega_at(self.time)
        if self.step_count % LIQ_EVERY == 0:
            self._update_liquid(dt * LIQ_EVERY)

        idx = self._bin_index()
        y = self.pos[:, 1]
        zr = self.pos[:, 2] - self.zc
        r2 = y * y + zr * zr
        w2 = self.omega * self.omega
        phi = self.g * zr - 0.5 * w2 * r2
        gy = -w2 * y
        gz = self.g - w2 * zr
        gmag = np.sqrt(gy * gy + gz * gz) + 1.0e-6
        depth = (self.phi_free[idx] - phi) / gmag
        subm = np.clip(0.5 + depth / (2.0 * self.rad), 0.0, 1.0)

        # gravity + liquid pressure gradient (buoyancy incl. centrifugal part)
        f_body = self.mass[:, None] * self.gvec[None, :]
        coef = RHO_FLUID * self.vol * subm
        f_body[:, 1] += coef * gy
        f_body[:, 2] += coef * gz

        f_wall = self._boundary_forces()
        f_pair = self._pair_forces()
        f_seg = self._segregation_force(idx, subm)

        # local liquid velocity: solid-body swirl + Darcy axial flow
        u = np.empty_like(self.pos)
        u[:, 0] = self.u_axial[idx]
        u[:, 1] = -self.omega * zr
        u[:, 2] = self.omega * y
        bdrag = self.drag_b_air + subm * (self.drag_b_liq - self.drag_b_air)

        f_other = f_body + f_wall + f_pair + f_seg
        m = self.mass[:, None]
        denom = 1.0 + (dt * bdrag / self.mass)[:, None]
        self.vel = (self.vel + dt * (f_other / m + (bdrag / self.mass)[:, None] * u)) / denom
        np.clip(self.vel, -V_CLIP, V_CLIP, out=self.vel)
        f_drag = bdrag[:, None] * (u - self.vel)
        self.pos = self.pos + dt * self.vel
        self._confine()

        self.f_ext = f_body + f_wall + f_drag
        self.f_drag = f_drag
        self.subm = subm
        f_i = f_pair + f_seg
        fint = np.zeros((self.species_count, 3), dtype=float)
        for s in range(self.species_count):
            sel = self.species == s
            if np.any(sel):
                fint[s] = f_i[sel].sum(axis=0)
        self.f_int = fint
        self.time += dt
        self.step_count += 1

    def _confine(self) -> None:
        y = self.pos[:, 1]
        zr = self.pos[:, 2] - self.zc
        r = np.sqrt(y * y + zr * zr)
        rmax = self.radius - self.rad
        out = r > rmax
        if np.any(out):
            rs = np.maximum(r[out], 1e-9)
            ny = y[out] / rs
            nz = zr[out] / rs
            self.pos[out, 1] = ny * rmax[out]
            self.pos[out, 2] = self.zc + nz * rmax[out]
            vr = self.vel[out, 1] * ny + self.vel[out, 2] * nz
            vr_out = np.maximum(vr, 0.0)
            self.vel[out, 1] -= vr_out * ny
            self.vel[out, 2] -= vr_out * nz
        xmax = 0.5 * self.length - self.rad
        hi = self.pos[:, 0] > xmax
        if np.any(hi):
            self.pos[hi, 0] = xmax[hi]
            self.vel[hi, 0] = np.minimum(self.vel[hi, 0], 0.0)
        lo = self.pos[:, 0] < -xmax
        if np.any(lo):
            self.pos[lo, 0] = -xmax[lo]
            self.vel[lo, 0] = np.maximum(self.vel[lo, 0], 0.0)
    # -- diagnostic grid fields --------------------------------------------
    def _grid_fields(self):
        h = self.h
        ix = np.clip(((self.pos[:, 0] - self.x0) / h).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((self.pos[:, 1] - self.y0) / h).astype(np.int64), 0, self.ny - 1)
        iz = np.clip(((self.pos[:, 2] - self.z0) / h).astype(np.int64), 0, self.nz - 1)
        flat = (ix * self.ny + iy) * self.nz + iz
        wsum = np.bincount(flat, weights=self.mass, minlength=self.n_cells)
        vfield = np.zeros((self.n_cells, 3), dtype=float)
        for k in range(3):
            vfield[:, k] = np.bincount(flat, weights=self.mass * self.vel[:, k],
                                       minlength=self.n_cells)
        occupied = wsum > 1e-12
        vfield[occupied] /= wsum[occupied][:, None]
        vfield = vfield.reshape(self.gshape + (3,))
        occ = occupied.reshape(self.gshape)

        phi_cell = self.g * self.g_zrel - 0.5 * self.omega ** 2 * self.g_r2
        phi_free_cell = self.phi_free[np.clip(
            ((self.gx_c + 0.5 * self.length) / self.dxb).astype(np.int64),
            0, self.nb - 1)]
        submerged = self.g_inside & (phi_cell <= phi_free_cell)
        fill_cells = submerged & (~occ)
        if np.any(fill_cells):
            vfield[..., 0][fill_cells] = self.u_axial[np.clip(
                ((self.gx_c[fill_cells] + 0.5 * self.length) / self.dxb).astype(np.int64),
                0, self.nb - 1)]
            vfield[..., 1][fill_cells] = -self.omega * self.g_zrel[fill_cells]
            vfield[..., 2][fill_cells] = self.omega * self.gy_c[fill_cells]
        active = submerged | occ
        return vfield, active

    def _frame(self) -> StateFrame:
        vfield, active = self._grid_fields()
        f_drag = getattr(self, "f_drag", np.zeros((self.n, 3)))
        subm = getattr(self, "subm", np.zeros(self.n))
        frame = StateFrame(
            time=float(max(0.0, self.time)),
            position=np.ascontiguousarray(self.pos, dtype=float),
            velocity=np.ascontiguousarray(self.vel, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.rad.copy(),
                "mass_quantity": self.mass_quantity.astype(np.float32),
                "submerged_fraction": subm.astype(np.float32),
                "contact_count": self.contact_count.astype(np.float32),
                "drag_force": f_drag.astype(np.float32),
            },
            fields={
                "external_force": self.f_ext.astype(np.float32),
                "interaction_force": self.f_int.astype(float),
                "cumulative_mass_source": np.zeros(self.species_count, dtype=float),
                "velocity_field": vfield.astype(np.float32),
                "active_mask": active.astype(bool),
                "liquid_volume_axial": self.v_liq.astype(np.float32),
                "axial_liquid_velocity": self.u_axial.astype(np.float32),
            },
            scalars=dict(self.scalars, angular_velocity=float(self.omega)),
        )
        frame.validate()
        return frame

    # -- driver -------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time:
            span = target - self.time
            nsub = max(1, int(np.ceil(span / DT_MAX)))
            dt = span / nsub
            for _ in range(nsub):
                self._step(dt)
            self.time = target
        self.omega = self._omega_at(self.time)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "particles": int(self.n),
            "steps": int(self.step_count),
            "dt_max": float(DT_MAX),
            "omega_rad_s": float(self.omega),
            "liquid_volume": float(self.v_liq.sum()),
            "max_speed": float(np.abs(self.vel).max()) if self.n else 0.0,
            "method": "soft-sphere DEM + co-rotating liquid pressure gradient"
                      " + Darcy axial coupling + Cahn-Hilliard axial de-mixing",
        }

