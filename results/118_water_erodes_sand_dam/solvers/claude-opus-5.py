"""Two-way coupled sand-water solver for the sand-dam infiltration/breach scene.

Physical model (implemented from scratch with NumPy/SciPy only):

* Both phases are carried by one particle set and interact through a shared,
  volume-fraction based weakly compressible SPH pressure field, so the evolving
  granular skeleton changes the resolved water flow while water pressure and
  Darcy drag change the sand motion (two-way momentum exchange).
* Sand is a cohesive granular skeleton: SPH pressure + pairwise elastic bonds
  (peridynamic style, irreversible damage) + regularized Coulomb friction. Bond
  strength/stiffness and inter-granular friction are reduced by the locally
  simulated pore saturation, so wetted sand yields where water infiltrated.
* Saturation is a transported field: capillary uptake from the resolved local
  water volume fraction plus SPH diffusion into the dam interior.
* Water is weakly compressible SPH with artificial (physical) viscosity.
* Static boundaries (flume floor, reservoir end wall, side walls, downstream
  retaining wall) are analytic half spaces with penalty + friction contact and
  kernel completion so hydrostatic pressure stays consistent at the walls.

No sand mass is ever created or deleted; erosion is pure transport. Time
stepping is a fixed step semi-implicit (symplectic) Euler scheme, so exported
budget impulses are the exact impulses applied by the integrator.
"""
from __future__ import annotations

import math
import time as _wall

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

SAND_ID = 1
WATER_ID = 2

_W_COEF = 21.0 / (2.0 * math.pi)      # Wendland C2 (3D) normalization
_DW_COEF = 210.0 / math.pi            # |dW/dr| normalization


class BenchmarkSolver:
    """Particle solver for the coupled sand-dam erosion benchmark."""

    # --- kernel / discretization ---
    H_SAND = 0.130
    H_WATER = 0.180
    SKIN = 0.025
    REBUILD_EVERY = 8
    DT_NOMINAL = 1.0e-3
    DT_MAX = 1.6e-3
    WALL_BUDGET_S = 1300.0

    # --- equations of state ---
    C_WATER = 35.0
    C_SAND = 32.0
    P_MAX = 1.5e5

    # --- dissipation ---
    ETA_WATER = 10.0
    ETA_SAND = 90.0
    CT_SAND = 600.0

    # --- cohesive sand skeleton ---
    BOND_RADIUS_FACTOR = 1.55
    K_BOND = 1.5e4
    C_BOND = 20.0
    EPS_BOND = 0.06
    BOND_CRUSH = -0.40
    SAT_BOND_WEAKEN = 0.85
    SAT_STIFF_WEAKEN = 0.60
    MU_SAND = 0.62
    MU_WALL_SAND = 0.55
    MU_WALL_WATER = 0.02
    SAT_FRICTION_WEAKEN = 0.55

    # --- coupling / transport ---
    K_DRAG = 8.0e3
    D_SAT = 0.012
    K_WET = 2.0
    K_DRAIN = 0.10
    SAT_REF_FRAC = 0.25
    SAT_EVERY = 4

    # --- contact / safety ---
    K_WALL = 5.0e4
    V_MAX = 30.0
    DAMP_FRACTION = 0.5

    # inner faces of the frozen static boundaries: (axis, inward sign, offset)
    WALLS = (
        (2, 1.0, 0.0),       # flume floor top face
        (0, 1.0, -3.93),     # reservoir end wall inner face
        (0, -1.0, 4.43),     # downstream retaining wall inner face
        (1, 1.0, -0.95),     # front side wall inner face
        (1, -1.0, 0.95),     # back side wall inner face
    )

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._t0_wall = _wall.perf_counter()
        self.scene = scene
        self.seed = int(seed)
        cfg = getattr(scene, "config", {}) or {}
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.duration = float(cfg.get("duration", 10.0))

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.N = int(self.pos.shape[0])
        self.mass = np.array(frame0.mass, dtype=np.float64).reshape(self.N)
        self.mat = np.array(frame0.body_id, dtype=np.int32).reshape(self.N)
        attrs = dict(frame0.attributes)
        self.vol = np.array(attrs["particle_volume"], dtype=np.float64).reshape(self.N)
        self.sat = np.array(attrs["sand_saturation"], dtype=np.float64).reshape(self.N)
        self.porosity = np.array(attrs["porosity"], dtype=np.float64).reshape(self.N)

        self.is_sand = self.mat == SAND_ID
        self.is_water = ~self.is_sand
        self.sand_idx = np.flatnonzero(self.is_sand)
        self.water_idx = np.flatnonzero(self.is_water)

        self.rho0 = self.mass / np.maximum(self.vol, 1e-12)
        self.dx = np.cbrt(self.vol)
        self.h = np.where(self.is_sand, self.H_SAND, self.H_WATER)
        self.eta = np.where(self.is_sand, self.ETA_SAND, self.ETA_WATER)
        cs = np.where(self.is_sand, self.C_SAND, self.C_WATER)
        self.k_eos = cs * cs
        self.d_wall = 0.5 * self.dx
        self.c_wall = 0.7 * np.sqrt(self.K_WALL * self.mass)
        self.mu_wall0 = np.where(self.is_sand, self.MU_WALL_SAND, self.MU_WALL_WATER)
        self.inv_mass = 1.0 / self.mass

        # rest volume fraction of each ideal lattice (removes systematic bias)
        dx_sand = float(np.median(self.dx[self.is_sand])) if self.sand_idx.size else 0.085
        dx_water = float(np.median(self.dx[self.is_water])) if self.water_idx.size else 0.10
        n0_sand = self._lattice_fraction(dx_sand, self.H_SAND)
        n0_water = self._lattice_fraction(dx_water, self.H_WATER)
        self.n0 = np.where(self.is_sand, n0_sand, n0_water)
        self.self_n = self.vol * (_W_COEF / self.h ** 3)
        self.dx_sand = dx_sand
        self.sat_ref = self.SAT_REF_FRAC * n0_water

        self._build_bonds(dx_sand)
        self._init_budget()

        self._search_radius = float(max(self.H_SAND, self.H_WATER) + self.SKIN)
        self._steps_since_build = 10 ** 9
        self._pos_at_build = self.pos.copy()
        self._t = 0.0
        self._dt_max = self.DT_MAX
        self._n_steps = 0
        self._n_builds = 0
        self._sat_counter = 0
        self._max_speed = 0.0
        self._broken_bonds = 0

        self._build_neighbors()
        # populate instantaneous external-force terms for the t = 0 frame
        self._forces(self.DT_NOMINAL, update_sat=False)

    def _lattice_fraction(self, spacing: float, h: float) -> float:
        """Kernel volume sum over an ideal lattice (analytic value is ~1)."""
        m = int(math.ceil(h / spacing)) + 1
        a = np.arange(-m, m + 1, dtype=np.float64) * spacing
        gx, gy, gz = np.meshgrid(a, a, a, indexing="ij")
        r = np.sqrt(gx * gx + gy * gy + gz * gz)
        q = np.minimum(r / h, 1.0)
        w = (_W_COEF / h ** 3) * (1.0 - q) ** 4 * (1.0 + 4.0 * q)
        return float(spacing ** 3 * w.sum())

    def _build_bonds(self, dx_sand: float) -> None:
        rng = np.random.default_rng(self.seed)
        if self.sand_idx.size < 2:
            self.bi = np.zeros(0, dtype=np.int64)
            self.bj = np.zeros(0, dtype=np.int64)
            self.b_r0 = np.zeros(0)
            self.b_state = np.zeros(0)
            self.b_strength = np.zeros(0)
            return
        tree = cKDTree(self.pos[self.sand_idx])
        pairs = tree.query_pairs(
            self.BOND_RADIUS_FACTOR * dx_sand, output_type="ndarray"
        ).reshape(-1, 2)
        self.bi = self.sand_idx[pairs[:, 0]].astype(np.int64)
        self.bj = self.sand_idx[pairs[:, 1]].astype(np.int64)
        d = self.pos[self.bi] - self.pos[self.bj]
        self.b_r0 = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        self.b_state = np.ones(self.bi.size, dtype=np.float64)
        # deterministic material heterogeneity, so the breach localizes physically
        self.b_strength = np.clip(np.exp(rng.normal(0.0, 0.25, self.bi.size)), 0.5, 1.8)

    def _init_budget(self) -> None:
        # G = 3 frozen domains: combined system, sand body, water body
        w = np.zeros((3, self.N), dtype=np.float32)
        w[0, :] = 1.0
        w[1, self.sand_idx] = 1.0
        w[2, self.water_idx] = 1.0
        self.domain_weights = w
        self.cum_impulse = np.zeros((3, 3), dtype=np.float64)
        self.cum_inflow = np.zeros((3, 3), dtype=np.float64)
        self.cum_outflow = np.zeros((3, 3), dtype=np.float64)
        self.mass_source = np.zeros(2, dtype=np.float64)
        self.mass_quantity = np.zeros((self.N, 2), dtype=np.float64)
        self.mass_quantity[self.sand_idx, 0] = self.mass[self.sand_idx]
        self.mass_quantity[self.water_idx, 1] = self.mass[self.water_idx]
        self._ext_terms = np.zeros((3, 3, 3), dtype=np.float64)
        self._interaction = np.zeros((4, 3), dtype=np.float64)
        self.sand_mass_total = float(self.mass[self.sand_idx].sum())
        self.water_mass_total = float(self.mass[self.water_idx].sum())
        v_scale = math.sqrt(2.0 * abs(float(self.gravity[2])) * 1.75)
        self.reference_momentum = float(self.mass.sum() * v_scale)

    # ------------------------------------------------------- neighbour search
    def _build_neighbors(self) -> None:
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self._search_radius, output_type="ndarray").reshape(-1, 2)
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        self.pi, self.pj = i, j
        hij = 0.5 * (self.h[i] + self.h[j])
        self.hij = hij
        self.inv_hij = 1.0 / hij
        self.wcoef = _W_COEF / hij ** 3
        self.dwcoef = _DW_COEF / hij ** 5
        self.vol_ij = self.vol[i] * self.vol[j]
        self.eta_ij = 0.5 * (self.eta[i] + self.eta[j])
        si, sj = self.is_sand[i], self.is_sand[j]
        self.pair_ss = si & sj
        self.pair_ww = (~si) & (~sj)
        self.pair_mix = si ^ sj
        # +1 when i is the sand member of a mixed pair, -1 when j is
        self.mix_sign = np.where(si, 1.0, -1.0) * self.pair_mix
        self.mix_idx = np.flatnonzero(self.pair_mix)
        self.ss_idx = np.flatnonzero(self.pair_ss)
        self._steps_since_build = 0
        self._pos_at_build = self.pos.copy()
        self._n_builds += 1

    def _maybe_rebuild(self) -> None:
        if self._steps_since_build >= self._rebuild_every_now():
            self._build_neighbors()
            return
        d = self.pos - self._pos_at_build
        if float(np.abs(d).max()) > 0.5 * self.SKIN:
            self._build_neighbors()

    def _rebuild_every_now(self) -> int:
        return self.REBUILD_EVERY

    @staticmethod
    def _accumulate(n: int, idx_i, idx_j, f):
        """Scatter symmetric pair forces (+f on i, -f on j) onto particles."""
        out = np.zeros((n, 3), dtype=np.float64)
        for c in range(3):
            out[:, c] = np.bincount(idx_i, weights=f[:, c], minlength=n)
            out[:, c] -= np.bincount(idx_j, weights=f[:, c], minlength=n)
        return out

    # ------------------------------------------------------------ force model
    def _forces(self, dt: float, update_sat: bool = True):
        """Assemble all forces; returns (total, boundary, cross_phase_on_sand)."""
        n = self.N
        i, j = self.pi, self.pj
        pos, vel = self.pos, self.vel

        dpos = pos[i] - pos[j]
        r2 = np.einsum("ij,ij->i", dpos, dpos)
        r = np.sqrt(r2)
        q = r * self.inv_hij
        active = q < 1.0
        idx = np.flatnonzero(active)

        ia, ja = i[idx], j[idx]
        ra = np.maximum(r[idx], 1.0e-9)
        qa = q[idx]
        one_q = 1.0 - qa
        one_q3 = one_q * one_q * one_q
        w = self.wcoef[idx] * one_q3 * one_q * (1.0 + 4.0 * qa)
        # |dW/dr| / r  for the Wendland C2 kernel: dW/dr = -coef * q * (1-q)^3
        dw_over_r = -self.dwcoef[idx] * one_q3
        volij = self.vol_ij[idx]

        # --- shared volume-fraction density and weakly compressible pressure ---
        wv_i = np.bincount(ia, weights=w * self.vol[ja], minlength=n)
        wv_j = np.bincount(ja, weights=w * self.vol[ia], minlength=n)
        nden = wv_i + wv_j + self.self_n
        nden += self._wall_completion()
        ratio = nden / self.n0
        press = np.minimum(self.k_eos * self.rho0 * np.maximum(ratio - 1.0, 0.0), self.P_MAX)

        # symmetric pressure force (momentum conserving), pair force on i
        pf = (press[ia] + press[ja]) * volij * dw_over_r
        fpair = pf[:, None] * dpos[idx]
        f_press_mag = np.abs(pf) * ra

        # --- viscous / drag dissipation (also momentum conserving) ---
        dvel = vel[ia] - vel[ja]
        gam = self.eta_ij[idx] * volij * (-dw_over_r)
        mix = self.pair_mix[idx]
        if self.K_DRAG > 0.0:
            gam = gam + (self.K_DRAG * volij * w) * mix
        # per-pair implicitness limit keeps the explicit damping unconditionally stable
        m_min = np.minimum(self.mass[ia], self.mass[ja])
        gam = np.minimum(gam, self.DAMP_FRACTION * m_min / max(dt, 1e-9))
        fpair -= gam[:, None] * dvel

        # --- regularized Coulomb friction between sand grains ---
        ss = self.pair_ss[idx]
        sel = np.flatnonzero(ss)
        if sel.size:
            isel, jsel = ia[sel], ja[sel]
            nrm = dpos[idx][sel] / ra[sel][:, None]
            dv = dvel[sel]
            vn = np.einsum("ij,ij->i", dv, nrm)
            vt = dv - vn[:, None] * nrm
            vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt)) + 1.0e-9
            sat_pair = 0.5 * (self.sat[isel] + self.sat[jsel])
            mu = self.MU_SAND * (1.0 - self.SAT_FRICTION_WEAKEN * sat_pair)
            f_lim = mu * f_press_mag[sel]
            f_visc = self.CT_SAND * self.vol_ij[idx][sel] * (-dw_over_r[sel]) * vt_mag
            m_min_s = np.minimum(self.mass[isel], self.mass[jsel])
            f_cap = 0.25 * m_min_s * vt_mag / max(dt, 1e-9)
            ft = np.minimum(np.minimum(f_lim, f_visc), f_cap)
            fpair[sel] -= (ft / vt_mag)[:, None] * vt

        f_pair_total = self._accumulate(n, ia, ja, fpair)

        # cross-phase (water <-> sand) part of the pair forces, signed onto sand
        cross = np.zeros((n, 3), dtype=np.float64)
        msel = np.flatnonzero(mix)
        if msel.size:
            cross = self._accumulate(n, ia[msel], ja[msel], fpair[msel])
        # split the exchange into the sand-side action and the water-side reaction
        cross_sand = np.where(self.is_sand[:, None], cross, 0.0)
        cross_water = np.where(self.is_sand[:, None], 0.0, cross)

        # --- saturation transport: capillary uptake + SPH diffusion ---
        if update_sat:
            self._update_saturation(dt, ia, ja, w, idx)

        # --- cohesive bonds of the granular skeleton ---
        f_bond = self._bond_forces(dt)

        # --- static boundary contact (external traction on each domain) ---
        f_bound = self._wall_forces(dt)

        f_grav = self.mass[:, None] * self.gravity[None, :]
        total = f_pair_total + f_bond + f_bound + f_grav

        # ---- budget bookkeeping: per-domain external force terms [G,K,3] ----
        ext = self._ext_terms
        ext[0, 0] = self.mass.sum() * self.gravity
        ext[1, 0] = self.sand_mass_total * self.gravity
        ext[2, 0] = self.water_mass_total * self.gravity
        ext[0, 1] = f_bound.sum(axis=0)
        ext[1, 1] = f_bound[self.sand_idx].sum(axis=0)
        ext[2, 1] = f_bound[self.water_idx].sum(axis=0)
        cs = cross_sand.sum(axis=0)
        cw = cross_water.sum(axis=0)
        ext[0, 2] = 0.0        # internal pair in the combined domain
        ext[1, 2] = cs         # water -> sand traction
        ext[2, 2] = cw         # sand -> water reaction

        inter = self._interaction
        inter[0] = cs          # force on sand from water
        inter[1] = cw          # equal and opposite reaction on water
        inter[2] = np.where(self.is_sand[:, None], f_pair_total - cross_sand
                            + f_bond, 0.0).sum(axis=0)
        inter[3] = np.where(self.is_sand[:, None], 0.0,
                            f_pair_total - cross_water).sum(axis=0)
        return total, f_bound

    def _bond_forces(self, dt: float):
        """Elastic-cohesive sand bonds with saturation dependent damage."""
        out = np.zeros((self.N, 3), dtype=np.float64)
        if self.bi.size == 0:
            return out
        live = np.flatnonzero(self.b_state > 0.0)
        if live.size == 0:
            return out
        bi, bj = self.bi[live], self.bj[live]
        d = self.pos[bi] - self.pos[bj]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r = np.maximum(r, 1.0e-9)
        nrm = d / r[:, None]
        r0 = self.b_r0[live]
        strain = (r - r0) / r0

        sat_pair = 0.5 * (self.sat[bi] + self.sat[bj])
        # pore pressure softens the wetted skeleton (Pradhana et al. cohesion loss)
        k_eff = self.K_BOND * (1.0 - self.SAT_STIFF_WEAKEN * sat_pair)
        dv = self.vel[bi] - self.vel[bj]
        vn = np.einsum("ij,ij->i", dv, nrm)
        fmag = -(k_eff * r0 * strain) - self.C_BOND * vn
        # keep the explicit bond damping from over-shooting at large dt
        m_min = np.minimum(self.mass[bi], self.mass[bj])
        cap = 0.4 * m_min * np.abs(vn) / max(dt, 1e-9) + np.abs(k_eff * r0 * strain)
        fmag = np.clip(fmag, -cap, cap)
        fb = fmag[:, None] * nrm
        out += self._accumulate(self.N, bi, bj, fb)

        eps_c = self.EPS_BOND * self.b_strength[live] * (
            1.0 - self.SAT_BOND_WEAKEN * sat_pair
        )
        broken = (strain > eps_c) | (strain < self.BOND_CRUSH)
        if np.any(broken):
            self.b_state[live[broken]] = 0.0
            self._broken_bonds += int(broken.sum())
        return out

    def _wall_completion(self):
        """Kernel-volume completion of the frozen static walls."""
        add = np.zeros(self.N, dtype=np.float64)
        for axis, sign, off in self.WALLS:
            g = sign * (self.pos[:, axis] - off)
            x = np.clip(g / self.h, 0.0, 1.0)
            # smooth 0.5 -> 0 fraction of the kernel support cut off by the plane
            add += 0.5 * (1.0 - x) ** 2 * (1.0 + 2.0 * x) * self.n0
        return np.minimum(add, 0.9 * self.n0)

    def _wall_forces(self, dt: float):
        """Penalty + Coulomb friction contact against the static boundaries."""
        f = np.zeros((self.N, 3), dtype=np.float64)
        mu_wall = self.mu_wall0 * (1.0 - self.SAT_FRICTION_WEAKEN * self.sat)
        for axis, sign, off in self.WALLS:
            g = sign * (self.pos[:, axis] - off)
            pen = self.d_wall - g
            hit = np.flatnonzero(pen > 0.0)
            if hit.size == 0:
                continue
            vn = sign * self.vel[hit, axis]
            fn = self.K_WALL * pen[hit] - self.c_wall[hit] * vn
            fn = np.maximum(fn, 0.0)
            f[hit, axis] += sign * fn
            # tangential Coulomb friction, capped for explicit stability
            vt = self.vel[hit].copy()
            vt[:, axis] = 0.0
            vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt)) + 1.0e-9
            f_lim = mu_wall[hit] * fn
            f_cap = 0.3 * self.mass[hit] * vt_mag / max(dt, 1e-9)
            ft = np.minimum(f_lim, f_cap)
            f[hit] -= (ft / vt_mag)[:, None] * vt
        return f

    def _update_saturation(self, dt: float, ia, ja, w, idx) -> None:
        """Capillary uptake from the resolved water phase + SPH pore diffusion."""
        self._sat_counter += 1
        if self._sat_counter % self.SAT_EVERY != 0:
            return
        dt_eff = dt * self.SAT_EVERY
        n = self.N
        vol_w = np.where(self.is_water, self.vol, 0.0)
        # local resolved water volume fraction seen by every sample
        wf = np.bincount(ia, weights=w * vol_w[ja], minlength=n)
        wf += np.bincount(ja, weights=w * vol_w[ia], minlength=n)
        target = np.clip(wf / max(self.sat_ref, 1e-9), 0.0, 1.0)

        sat = self.sat
        rate = np.where(target > sat, self.K_WET, self.K_DRAIN) * (target - sat)

        # diffusion of pore water through the granular skeleton (sand-sand pairs)
        sel = self.ss_idx
        sel = sel[np.isin(sel, idx, assume_unique=False)] if False else None
        ss_mask = self.pair_ss[idx]
        m = np.flatnonzero(ss_mask)
        if m.size:
            im, jm = ia[m], ja[m]
            flux = w[m] * (sat[jm] - sat[im])
            lap = np.bincount(im, weights=flux, minlength=n)
            lap += np.bincount(jm, weights=-flux, minlength=n)
            rate = rate + self.D_SAT * lap

        sat_new = sat + dt_eff * rate
        sat_new[self.water_idx] = 1.0
        self.sat = np.clip(sat_new, 0.0, 1.0)
# __APPEND__

