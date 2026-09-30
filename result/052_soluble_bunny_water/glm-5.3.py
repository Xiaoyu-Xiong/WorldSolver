"""Physically-based particle solver for 052_soluble_bunny_water.

Coupled multiphysics model (implemented from scratch, no Genesis):

* Liquid (clear water, mixture) : Position-Based Fluids (Macklin & Muller 2013)
  density constraints with poly6/spiky kernels, XSPH viscosity, floor
  no-penetration, open-side outflow and prescribed pipe inflow.
* Soluble bunny                 : PBD mass-spring elastic solid (deformable),
  velocity-damping, viscous drag coupling with the liquid, floor friction.
* Dissolution                   : wetted solid mass is transferred as solute to
  neighboring liquid particles (kernel weighted), advected and diffused with
  the liquid (conservative Fickian pair fluxes), and fully dissolved chunks
  convert to liquid carrying their remaining mass as solute.

All required StateFrame trajectory fields are exported every sample.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
GRAVITY = np.array([0.0, 0.0, -9.81])
H = 0.16              # SPH smoothing radius [m]
H2 = H * H
POLY6 = 315.0 / (64.0 * np.pi * H ** 9)
SPIKY = -45.0 / (np.pi * H ** 6)
W0 = POLY6 * H2 ** 3  # self kernel value

LIQ_DT = 0.005        # physics substep [s]
PBF_ITERS = 3
PBF_RELAX = 0.5
PBF_EPS = 1.0e-2
MAX_DP = 0.3 * H
XSPH_ALPHA = 0.03
BUNNY_DAMP = 0.5
SPRING_ITERS = 2
SPRING_STIFF = 0.7

DISSOLVE_RATE = 2.0       # 1/s at full wetness
WET_SAT = 6.0             # water neighbors counting as fully wet
CONVERT_FRAC = 0.05       # solid fraction at which a chunk turns liquid
MIN_BUNNY = 30            # keep at least this many label-1 samples (binding)
DIFFUSION = 0.010         # solute pair-diffusion coefficient
DRAG_RATE = 4.0           # liquid->bunny drag coupling [1/s]

FLOOR_Z = 0.0
FLOOR_X = (-3.6, 4.0)
FLOOR_Y = (-2.7, 2.7)
OUT_X = (-3.5, 3.9)
OUT_Y = 2.65

EMIT_DT = 0.05
EMIT_X = -0.90
EMIT_R = 0.30
EMIT_CY = 0.0
EMIT_CZ = 1.30
EMIT_SPACING = 0.085
EMIT_SPEED = 2.4

# diagnostic grid
GX = (-1.05, 2.60)
GY = (-1.35, 1.35)
GZ = (0.0, 1.95)
GRID_DX = 0.13
MIX_BOUNDS = ((-0.55, 2.45), (-1.25, 1.25), (0.04, 1.65))


def _poly6(r2: np.ndarray) -> np.ndarray:
    w = np.zeros_like(r2)
    m = r2 < H2
    w[m] = POLY6 * (H2 - r2[m]) ** 3
    return w


def _spike_coef(r: np.ndarray) -> np.ndarray:
    """Coefficient c with grad_xi W = c * (xi - xj)."""
    r = np.maximum(r, 1e-6)
    c = np.zeros_like(r)
    m = r < H
    c[m] = SPIKY * (H - r[m]) ** 2 / r[m]
    return c


class BenchmarkSolver:
    def __init__(self):
        self.initialized = False

    # ------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.rng = np.random.default_rng(int(seed) + 1)
        frame = scene.initial_state()
        pos = np.array(frame.position, dtype=float)
        vel = np.array(frame.velocity, dtype=float)
        label = np.array(frame.attributes["material_label"], dtype=float)
        conc = np.array(frame.attributes["red_solute_concentration"], dtype=float)
        mass = np.array(frame.mass, dtype=float)

        n = len(pos)
        self.n = n
        self.pos = pos
        self.vel = vel
        self.label = label
        self.M = mass.copy()
        self.msolute = mass * conc
        self.mwater = mass * (1.0 - conc)
        self.msolid = np.where(label == 1.0, mass, 0.0)
        self.mwater = np.where(label == 1.0, 0.0, self.mwater)
        self.ref_pos = pos.copy()
        self.m0 = self.msolid.copy()

        # static receiving layer (initial label-3 lattice) never moves
        self.static = label == 3.0

        # initial bunny elastic / neighborhood bonds
        bun = np.flatnonzero(label == 1.0)
        self.spring_a = np.empty(0, dtype=np.int64)
        self.spring_b = np.empty(0, dtype=np.int64)
        self.spring_rest = np.empty(0)
        self.spring_alive = np.empty(0, dtype=bool)
        self.mls_a = np.empty(0, dtype=np.int64)
        self.mls_b = np.empty(0, dtype=np.int64)
        self.mls_alive = np.empty(0, dtype=bool)
        if len(bun):
            tree = cKDTree(pos[bun])
            # spring bonds (short range)
            pairs = tree.query_pairs(H, output_type="ndarray")
            if len(pairs):
                a = bun[pairs[:, 0]]
                b = bun[pairs[:, 1]]
                self.spring_a, self.spring_b = a, b
                self.spring_rest = np.linalg.norm(pos[a] - pos[b], axis=1)
                self.spring_alive = np.ones(len(a), dtype=bool)
            # MLS neighborhood for deformation gradients (wider)
            pairs = tree.query_pairs(0.26, output_type="ndarray")
            if len(pairs):
                self.mls_a = bun[pairs[:, 0]]
                self.mls_b = bun[pairs[:, 1]]
                self.mls_alive = np.ones(len(self.mls_a), dtype=bool)

        # rest densities measured from the initial same-label lattice
        self.rho0 = np.full(n, 1000.0)
        tree = cKDTree(pos)
        for lab in (1.0, 2.0, 3.0):
            idx = np.flatnonzero(label == lab)
            if not len(idx):
                continue
            neigh = tree.query_ball_point(pos[idx], H)
            rho = np.full(len(idx), mass[idx] * W0)
            for k, lst in enumerate(neigh):
                lst = np.asarray(lst, dtype=np.int64)
                lst = lst[(lst != idx[k]) & (label[lst] == lab)]
                if len(lst):
                    d2 = np.sum((pos[idx[k]] - pos[lst]) ** 2, axis=1)
                    rho[k] += np.sum(mass[lst] * _poly6(d2))
            self.rho0[idx] = rho
        self.rho0_water = float(np.mean(self.rho0[label == 2.0]))
        self.last_rho = self.rho0.copy()

        # bookkeeping
        self.t = 0.0
        self.dt = LIQ_DT
        self.next_emit = EMIT_DT
        inflow_cfg = cfg.get("initial_conditions", {})
        self.inflow_speed = float(inflow_cfg.get("inflow_speed_m_s", EMIT_SPEED))
        self.inflow_end = float(inflow_cfg.get("inflow_end_s", 5.5))
        self.inflow_start = float(inflow_cfg.get("inflow_start_s", 0.0))
        self.cum_inflow = 0.0
        self.cum_out_water = 0.0
        self.cum_out_solute = 0.0
        self.cum_out_solid = 0.0
        self.dissolved_total = 0.0

        # per-frame accumulators
        gshape = self._grid_shape()
        self.release_buf = np.zeros(gshape)
        self.imp_water_on_bunny = np.zeros(3)
        self.imp_floor_on_bunny = np.zeros(3)
        self.last_frame_t = 0.0

        self.initial_bunny_mass = float(np.sum(self.msolid))
        self.initial_solute = self.initial_bunny_mass
        self.grid_spacing = GRID_DX
        self.initialized = True

    # ------------------------------------------------------------- grid
    def _grid_shape(self):
        nx = int(np.floor((GX[1] - GX[0]) / GRID_DX))
        ny = int(np.floor((GY[1] - GY[0]) / GRID_DX))
        nz = int(np.floor((GZ[1] - GZ[0]) / GRID_DX))
        self.nx, self.ny, self.nz = nx, ny, nz
        return nx * ny * nz

    def _cell_index(self, p):
        ix = np.clip(((p[:, 0] - GX[0]) / GRID_DX).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((p[:, 1] - GY[0]) / GRID_DX).astype(np.int64), 0, self.ny - 1)
        iz = np.clip(((p[:, 2] - GZ[0]) / GRID_DX).astype(np.int64), 0, self.nz - 1)
        inside = ((p[:, 0] >= GX[0]) & (p[:, 0] < GX[0] + self.nx * GRID_DX)
                  & (p[:, 1] >= GY[0]) & (p[:, 1] < GY[0] + self.ny * GRID_DX)
                  & (p[:, 2] >= GZ[0]) & (p[:, 2] < GZ[0] + self.nz * GRID_DX))
        return (ix * self.ny + iy) * self.nz + iz, inside

    # ------------------------------------------------------------- physics
    def _emit(self):
        ys = np.arange(EMIT_CY - EMIT_R, EMIT_CY + EMIT_R + 1e-9, EMIT_SPACING)
        zs = np.arange(EMIT_CZ - EMIT_R, EMIT_CZ + EMIT_R + 1e-9, EMIT_SPACING)
        gy, gz = np.meshgrid(ys, zs, indexing="ij")
        keep = (gy - EMIT_CY) ** 2 + (gz - EMIT_CZ) ** 2 <= EMIT_R ** 2
        gy, gz = gy[keep], gz[keep]
        m = len(gy)
        if m == 0:
            return
        newp = np.stack([np.full(m, EMIT_X), gy, gz], axis=1)
        newv = np.stack([np.full(m, self.inflow_speed), np.zeros(m), np.zeros(m)], axis=1)
        mass = self.rho0_water * EMIT_SPACING ** 3
        self.pos = np.vstack((self.pos, newp))
        self.vel = np.vstack((self.vel, newv))
        self.M = np.append(self.M, np.full(m, mass))
        self.msolid = np.append(self.msolid, np.zeros(m))
        self.mwater = np.append(self.mwater, np.full(m, mass))
        self.msolute = np.append(self.msolute, np.zeros(m))
        self.label = np.append(self.label, np.full(m, 2.0))
        self.rho0 = np.append(self.rho0, np.full(m, self.rho0_water))
        self.ref_pos = np.vstack((self.ref_pos, newp))
        self.static = np.append(self.static, np.zeros(m, dtype=bool))
        self.m0 = np.append(self.m0, np.zeros(m))
        self.last_rho = np.append(self.last_rho, np.full(m, self.rho0_water))
        self.cum_inflow += m * mass

    def _neighbors(self, liquid_idx):
        """Pair arrays (i, j): i liquid sample, j any sample within H (j != i)."""
        if len(liquid_idx) == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64))
        tree = cKDTree(self.pos)
        lists = tree.query_ball_point(self.pos[liquid_idx], H)
        counts = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
        i_arr = np.repeat(liquid_idx, counts)
        j_arr = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) \
            if counts.sum() else np.empty(0, np.int64)
        if len(i_arr):
            good = i_arr != j_arr
            i_arr, j_arr = i_arr[good], j_arr[good]
        return i_arr, j_arr

    def _substep(self, dt: float) -> None:
        pos, vel, M = self.pos, self.vel, self.M
        n = len(pos)
        label = self.label
        dyn = ~self.static                     # anything that may move
        liquid = np.isin(label, (2.0, 3.0))
        dyn_liq = liquid & dyn
        bunny = (label == 1.0) & (self.msolid > 0.0)

        liq_idx = np.flatnonzero(dyn_liq)
        i_arr, j_arr = self._neighbors(liq_idx)

        # ---- external gravity
        vel[dyn] += GRAVITY[None, :] * dt

        # ---- liquid <-> bunny drag coupling (momentum conserving)
        if len(i_arr):
            wet = bunny[j_arr]
            if np.any(wet):
                wi, wj = i_arr[wet], j_arr[wet]
                d = pos[wi] - pos[wj]
                r2 = np.sum(d * d, axis=1)
                w = _poly6(r2)
                wsum = np.bincount(wj, weights=w, minlength=n)
                wnorm = w / np.maximum(wsum[wj], 1e-12)
                dv = vel[wi] - vel[wj]           # liquid velocity relative to bunny
                accb = np.zeros((n, 3))          # drag acceleration on bunny samples
                for k in range(3):
                    accb[:, k] = DRAG_RATE * np.bincount(
                        wj, weights=wnorm * dv[:, k], minlength=n)
                impj = M[:, None] * accb * dt    # per-sample impulse (bunny rows only)
                vel[wj] += accb[wj] * dt
                # equal and opposite reaction on the liquid neighbors
                for k in range(3):
                    vel[wi, k] -= np.bincount(
                        wi, weights=wnorm * impj[wj, k], minlength=n)[wi] / M[wi]
                self.imp_water_on_bunny += impj.sum(axis=0)

        # ---- XSPH viscosity for dynamic liquid
        if len(i_arr):
            liqmask = liquid[j_arr]
            if np.any(liqmask):
                ii, jj = i_arr[liqmask], j_arr[liqmask]
                d = pos[ii] - pos[jj]
                r2 = np.sum(d * d, axis=1)
                w = _poly6(r2)
                coef = XSPH_ALPHA * w * M[jj] / (0.5 * (self.rho0[ii] + self.rho0[jj]))
                coef = np.minimum(coef, 0.05)
                for k in range(3):
                    vel[ii, k] += np.bincount(ii, weights=coef * (vel[jj, k] - vel[ii, k]),
                                              minlength=n)[ii]

        # ---- bunny internal velocity damping (fixed initial bonds)
        alive = self.spring_alive
        if np.any(alive) and np.any(bunny):
            a, b = self.spring_a[alive], self.spring_b[alive]
            ok = bunny[a] & bunny[b]
            a, b = a[ok], b[ok]
            if len(a):
                dv = vel[a] - vel[b]
                fac = min(1.0, BUNNY_DAMP * dt)
                for k in range(3):
                    da = np.bincount(a, weights=fac * dv[:, k], minlength=n)
                    db = np.bincount(b, weights=fac * dv[:, k], minlength=n)
                    vel[a, k] -= da[a]
                    vel[b, k] += db[b]

        # ---- predict
        x_prev = pos.copy()
        pos[dyn] += vel[dyn] * dt

        # ---- bunny PBD springs
        for _ in range(SPRING_ITERS):
            alive = self.spring_alive
            if not np.any(alive):
                break
            a, b = self.spring_a[alive], self.spring_b[alive]
            ok = bunny[a] & bunny[b]
            a, b, rest = a[ok], b[ok], self.spring_rest[alive][ok]
            if not len(a):
                break
            d = pos[b] - pos[a]
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            corr = (SPRING_STIFF * (r - rest) / r)[:, None] * d
            ca = np.zeros((n, 3))
            cb = np.zeros((n, 3))
            np.add.at(ca, a, 0.5 * corr)     # stretched: a moves toward b
            np.add.at(cb, b, -0.5 * corr)
            mag = np.linalg.norm(ca, axis=1) + np.linalg.norm(cb, axis=1)
            ca *= np.where(mag > MAX_DP, MAX_DP / np.maximum(mag, 1e-12), 1.0)[:, None]
            cb *= np.where(mag > MAX_DP, MAX_DP / np.maximum(mag, 1e-12), 1.0)[:, None]
            pos += np.where(bunny[:, None], ca + cb, 0.0)

        # ---- PBF density constraints for dynamic liquid
        if len(liq_idx):
            lam = np.zeros(n)
            for _ in range(PBF_ITERS):
                if len(i_arr) == 0:
                    break
                d = pos[i_arr] - pos[j_arr]
                r2 = np.sum(d * d, axis=1)
                w = _poly6(r2)
                rho = M[liq_idx] * W0 \
                    + np.bincount(i_arr, weights=M[j_arr] * w, minlength=n)[liq_idx]
                self.last_rho[liq_idx] = rho
                C = np.maximum(rho / self.rho0[liq_idx] - 1.0, 0.0)
                r = np.sqrt(r2)
                coef = _spike_coef(r)
                gw = coef[:, None] * d                     # dW/dx_i
                gi = np.zeros((n, 3))
                mj_gw = M[j_arr][:, None] * gw / self.rho0[i_arr][:, None]
                for k in range(3):
                    gi[:, k] = np.bincount(i_arr, weights=mj_gw[:, k], minlength=n)
                # |sum_k grad C|^2 + sum |grad|^2
                sum2 = np.bincount(i_arr, weights=np.sum(mj_gw ** 2, axis=1), minlength=n)
                denom = np.zeros(n)
                denom[liq_idx] = (np.sum(gi[liq_idx] ** 2, axis=1) + sum2[liq_idx]
                                  + PBF_EPS)
                lam = np.zeros(n)
                lam[liq_idx] = -C / denom[liq_idx]
                dpx = np.zeros((n, 3))
                lamsum = (lam[i_arr] + np.where(liquid[j_arr], lam[j_arr], 0.0))[:, None]
                contrib = lamsum * M[j_arr][:, None] * gw / self.rho0[i_arr][:, None]
                for k in range(3):
                    dpx[:, k] = np.bincount(i_arr, weights=contrib[:, k], minlength=n)
                step = PBF_RELAX * dpx[liq_idx]
                mag = np.linalg.norm(step, axis=1)
                scale = np.where(mag > MAX_DP, MAX_DP / np.maximum(mag, 1e-12), 1.0)
                pos[liq_idx] += step * scale[:, None]
                # floor no-penetration inside iterations
                below = pos[:, 2] < FLOOR_Z
                if np.any(below):
                    imp = np.zeros((n, 3))
                    zold = pos[below, 2]
                    imp[below, 2] = M[below] * (FLOOR_Z - zold) / dt
                    self.imp_floor_on_bunny += imp[bunny & below].sum(axis=0)
                    pos[below, 2] = FLOOR_Z

        # ---- floor clamp for every dynamic sample
        below = (pos[:, 2] < FLOOR_Z) & dyn
        if np.any(below):
            imp = np.zeros((n, 3))
            imp[below, 2] = M[below] * (FLOOR_Z - pos[below, 2]) / dt
            self.imp_floor_on_bunny += imp[bunny & below].sum(axis=0)
            pos[below, 2] = FLOOR_Z

        # ---- velocities from positions (PBD)
        vel[dyn] = (pos[dyn] - x_prev[dyn]) / dt
        contact = (pos[:, 2] <= FLOOR_Z + 1e-9) & dyn
        if np.any(contact):
            vel[contact, 2] = np.maximum(vel[contact, 2], 0.0)
            fric = float(np.exp(-2.0 * dt))   # viscous ground friction per second
            vel[contact, 0] *= fric
            vel[contact, 1] *= fric

        # ---- dissolution: wetted solid -> solute in neighboring liquid
        if len(i_arr):
            wetb = bunny[j_arr]
            if np.any(wetb):
                wi, wj = i_arr[wetb], j_arr[wetb]
                d = pos[wi] - pos[wj]
                r2 = np.sum(d * d, axis=1)
                w = _poly6(r2)
                nwet = np.bincount(wj, weights=np.ones(len(wj)), minlength=n)
                wsum = np.bincount(wj, weights=w, minlength=n)
                rate = DISSOLVE_RATE * np.minimum(1.0, nwet / WET_SAT)
                frac = 1.0 - np.exp(-rate * dt)
                dm = np.zeros(n)
                dm[wj] = self.msolid[wj] * frac[wj]
                dm = np.maximum(dm, 0.0)
                share = w / np.maximum(wsum[wj], 1e-12)
                got = np.bincount(wi, weights=share * dm[wj], minlength=n)
                # grid source of released solute
                cells, inside = self._cell_index(pos[wi])
                ok = inside & (dm[wj] > 0.0)
                if np.any(ok):
                    self.release_buf += np.bincount(cells[ok],
                                                    weights=(share * dm[wj])[ok],
                                                    minlength=self.release_buf.size)
                self.msolute += got
                self.M += got
                self.msolid -= dm
                self.M -= dm
                self.dissolved_total += float(dm.sum())
                np.clip(self.msolid, 0.0, None, out=self.msolid)

        # ---- conservative solute diffusion between liquid samples
        if len(i_arr):
            both = liquid[i_arr] & liquid[j_arr]
            if np.any(both):
                ii, jj = i_arr[both], j_arr[both]
                d = pos[ii] - pos[jj]
                r2 = np.sum(d * d, axis=1)
                w = _poly6(r2)
                ci = self.msolute[ii] / np.maximum(M[ii], 1e-12)
                cj = self.msolute[jj] / np.maximum(M[jj], 1e-12)
                flux = DIFFUSION * dt * w * M[ii] * M[jj] * (cj - ci)
                delta = np.zeros(n)
                delta += np.bincount(ii, weights=flux, minlength=n)
                delta -= np.bincount(jj, weights=flux, minlength=n)
                self.msolute += delta
                np.clip(self.msolute, 0.0, None, out=self.msolute)

        # ---- conversion of nearly dissolved chunks to liquid
        n_bunny = int(np.count_nonzero(self.label == 1.0))
        conv = np.flatnonzero((self.label == 1.0)
                              & (self.msolid <= CONVERT_FRAC * np.maximum(self.m0, 1e-12)))
        for c in conv:
            if n_bunny <= MIN_BUNNY:
                break
            self.msolute[c] += self.msolid[c]
            self.msolid[c] = 0.0
            p = self.pos[c]
            inside_mix = (MIX_BOUNDS[0][0] <= p[0] <= MIX_BOUNDS[0][1]
                          and MIX_BOUNDS[1][0] <= p[1] <= MIX_BOUNDS[1][1]
                          and MIX_BOUNDS[2][0] <= p[2] <= MIX_BOUNDS[2][1])
            self.label[c] = 3.0 if inside_mix else 2.0
            self.mwater[c] = 0.0
            n_bunny -= 1
            dead = (self.spring_a == c) | (self.spring_b == c)
            self.spring_alive[dead] = False
            deadm = (self.mls_a == c) | (self.mls_b == c)
            self.mls_alive[deadm] = False

        # ---- open side outflow
        out = ((self.pos[:, 0] < OUT_X[0]) | (self.pos[:, 0] > OUT_X[1])
               | (np.abs(self.pos[:, 1]) > OUT_Y))
        out &= ~self.static
        if np.any(out):
            idx = np.flatnonzero(out)
            self.cum_out_solid += float(self.msolid[idx].sum())
            self.cum_out_water += float(self.mwater[idx].sum())
            self.cum_out_solute += float(self.msolute[idx].sum())
            keep = ~out
            newidx = np.cumsum(keep) - 1
            sa_ok = keep[self.spring_a] & keep[self.spring_b]
            self.spring_a = newidx[self.spring_a[sa_ok]]
            self.spring_b = newidx[self.spring_b[sa_ok]]
            self.spring_rest = self.spring_rest[sa_ok]
            self.spring_alive = self.spring_alive[sa_ok]
            ma_ok = keep[self.mls_a] & keep[self.mls_b]
            self.mls_a = newidx[self.mls_a[ma_ok]]
            self.mls_b = newidx[self.mls_b[ma_ok]]
            self.mls_alive = self.mls_alive[ma_ok]
            for arr in ("pos", "vel", "M", "msolid", "mwater", "msolute",
                        "label", "rho0", "ref_pos", "m0", "last_rho"):
                setattr(self, arr, getattr(self, arr)[keep])
            self.static = self.static[keep]

        # ---- prescribed pipe inflow
        if (self.t + dt) >= self.next_emit - 1e-9 and self.t < self.inflow_end - 1e-9:
            self._emit()
            self.next_emit += EMIT_DT

        self.t += dt

    # ------------------------------------------------------------- frames
    def _deformation_gradients(self, bunny_mask):
        n = len(self.pos)
        F = np.tile(np.eye(3), (n, 1, 1))
        alive = self.mls_alive & bunny_mask[self.mls_a] & bunny_mask[self.mls_b]
        if not np.any(alive):
            return F
        a, b = self.mls_a[alive], self.mls_b[alive]
        dc = self.pos[a] - self.pos[b]
        dr = self.ref_pos[a] - self.ref_pos[b]
        w = np.ones(len(a))
        cnt = np.bincount(a, minlength=n)
        A = np.zeros((n, 3, 3))
        B = np.zeros((n, 3, 3))
        for k in range(3):
            for l in range(3):
                A[:, k, l] = np.bincount(a, weights=dc[:, k] * dr[:, l] * w, minlength=n)
                B[:, k, l] = np.bincount(a, weights=dr[:, k] * dr[:, l] * w, minlength=n)
        idx = np.flatnonzero(bunny_mask & (cnt >= 4))
        for i in idx:
            Bm = B[i] + 1e-9 * np.eye(3) * max(np.trace(B[i]), 1e-6)
            try:
                F[i] = A[i] @ np.linalg.inv(Bm)
            except np.linalg.LinAlgError:
                F[i] = np.eye(3)
        if not np.all(np.isfinite(F)):
            F[~np.isfinite(F)] = np.eye(3).reshape(1, 3, 3)
        return F

    def _build_frame(self, t: float) -> StateFrame:
        pos, vel, M = self.pos, self.vel, self.M
        n = len(pos)
        label = self.label
        liquid = np.isin(label, (2.0, 3.0))
        bunny = label == 1.0
        conc = np.where(M > 0, self.msolute / np.maximum(M, 1e-12), 0.0)
        conc = np.where(bunny, np.where(M > 0, self.msolid / np.maximum(M, 1e-12), 0.0), conc)
        solid_frac = np.where(M > 0, self.msolid / np.maximum(M, 1e-12), 0.0)

        mass_quantity = np.stack((self.msolid, self.mwater, self.msolute), axis=1)
        F = self._deformation_gradients(bunny)

        # diagnostic grid fields
        cells, inside = self._cell_index(pos)
        ng = self.release_buf.size
        liq_in = liquid & inside
        wsum = np.bincount(cells[liq_in], weights=M[liq_in], minlength=ng)
        vfield = np.zeros((self.nx, self.ny, self.nz, 3))
        if np.any(liq_in):
            w = M[liq_in]
            for k in range(3):
                vk = np.bincount(cells[liq_in], weights=w * vel[liq_in, k], minlength=ng)
                vk = np.where(wsum > 0, vk / np.maximum(wsum, 1e-12), 0.0)
                vfield[..., k] = vk.reshape(self.nx, self.ny, self.nz)
        active = (np.bincount(cells[liq_in], minlength=ng) > 0).astype(float)
        active = active.reshape(self.nx, self.ny, self.nz)
        csum = np.bincount(cells[liq_in], weights=M[liq_in] * conc[liq_in], minlength=ng)
        tscalar = np.where(wsum > 0, csum / np.maximum(wsum, 1e-12), 0.0)
        tscalar = tscalar.reshape(self.nx, self.ny, self.nz)

        interval = max(t - self.last_frame_t, 0.0)
        cell_vol = GRID_DX ** 3
        if interval > 1e-9:
            src = (self.release_buf / (interval * cell_vol)).reshape(self.nx, self.ny, self.nz)
            interaction = np.stack((
                self.imp_water_on_bunny / interval,
                -self.imp_water_on_bunny / interval,
                self.imp_floor_on_bunny / interval))
        else:
            src = np.zeros((self.nx, self.ny, self.nz))
            interaction = np.zeros((3, 3))
        cum_source = np.array([0.0,
                               self.cum_inflow - self.cum_out_water,
                               -self.cum_out_solute])

        attributes = {
            "material_label": label.copy(),
            "red_solute_concentration": conc,
            "solid_phase_fraction": solid_frac,
            "density": np.where(liquid, self.last_rho, self.rho0),
            "mass_quantity": mass_quantity,
            "deformation_gradient": F,
        }
        fields = {
            "velocity_field": vfield,
            "active_mask": active,
            "transported_scalar": tscalar,
            "scalar_source": src,
            "cumulative_mass_source": cum_source,
            "interaction_force": interaction,
        }
        scalars = {
            "grid_spacing": float(GRID_DX),
            "initial_red_solute_mass": float(self.initial_solute),
            "initial_bunny_mass": float(self.initial_bunny_mass),
            "inflow_speed": float(self.inflow_speed),
            "remaining_bunny_solid_mass": float(self.msolid.sum()),
            "dissolved_solute_mass": float(self.msolute.sum()),
            "cumulative_inflow_mass": float(self.cum_inflow),
            "cumulative_outflow_mass": float(self.cum_out_water + self.cum_out_solute),
        }
        self.release_buf = np.zeros(ng)
        self.imp_water_on_bunny = np.zeros(3)
        self.imp_floor_on_bunny = np.zeros(3)
        self.last_frame_t = t

        frame = StateFrame(
            time=float(t), position=pos.copy(), velocity=vel.copy(),
            mass=M.copy(), body_id=label.astype(np.int64),
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- api
    def advance_to(self, time: float):
        t = float(time)
        if not self.initialized:
            raise RuntimeError("solver not initialized")
        guard = 0
        while self.t < t - 1e-9:
            dt = min(self.dt, t - self.t)
            self._substep(dt)
            guard += 1
            if guard > 100000:
                raise RuntimeError("substep guard triggered")
        self.t = t
        return self._build_frame(t)

    def diagnostics(self) -> dict[str, object]:
        solid = float(self.msolid.sum()) if self.initialized else float("nan")
        return {
            "remaining_bunny_solid_mass_kg": solid,
            "dissolved_mass_kg": float(self.dissolved_total),
            "cumulative_inflow_kg": float(self.cum_inflow),
            "cumulative_outflow_kg": float(self.cum_out_water + self.cum_out_solute),
            "particle_count": int(len(self.pos)),
            "final_time": float(self.t),
        }
