"""Physically-based particle solver for wettable (insoluble) sand under water pouring.

Model
-----
* Sand: bonded discrete-element pile.  Bonds are built once from the initial
  compact conical configuration (rest length = initial gap) so the dry pile is
  stress free at t = 0.  Each bond carries a normal spring (compression /
  tension limited by a wetness dependent cohesion), viscous normal damping and
  an accumulated tangential (frictional) spring capped by Coulomb friction.
  The sand phase never evolves (solid_fraction = 1) and never dissolves: its
  mass is constant.
* Water: weakly-compressible particle fluid (pairwise short range pressure +
  XSPH-like viscosity) falling from the two spouts, colliding with the tray and
  with the sand grains.
* Coupling: water grains penetrating the sand packing are slowed by a
  permeability drag, push the grains (impact -> crater / pile deformation) and
  transfer water mass into a per-grain ``wetness`` scalar (wetting
  concentration transfer only).  Bound water mass is tracked so that the total
  water phase mass is conserved.  Wetness spreads through the pile by
  capillary diffusion along grain bonds -> wetting penetration front.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        cfg = scene.config
        self.dt = 0.0025
        self.duration = float(cfg.get("duration", 6.0))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        g = cfg.get("gravity", (0.0, 0.0, -9.81))
        self.gravity = np.asarray(g, dtype=float)

        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=float).copy()
        vel = np.asarray(frame0.velocity, dtype=float).copy()
        mass = np.asarray(frame0.mass, dtype=float).copy()
        body = np.asarray(frame0.body_id).astype(int)
        conc = np.asarray(frame0.attributes["water_concentration"], dtype=float).copy()

        sm = body == 1
        wm = body == 2
        self.sp = pos[sm]
        self.sv = vel[sm]
        self.smass = mass[sm]
        self.n_sand = int(sm.sum())
        self.wp = pos[wm]
        self.wv = vel[wm]
        self.wc = conc[wm]
        self.wmass0 = float(mass[wm].mean()) if wm.any() else 2.75e-4

        self.wet = np.zeros(self.n_sand)
        self.w_max = 0.05

        # --- geometry / material parameters -----------------------------
        self.sand_r = 0.030
        self.water_r = 0.020
        self.spacing = 0.075
        self.floor_z = 0.0
        self.wall_x = 1.47
        self.wall_y = 1.07
        self.rest = 0.2
        self.friction = 0.55

        self.kn = 22.0
        self.kt = 0.55 * self.kn
        m_eff = 0.5 * float(self.smass.mean())
        self.cn = 0.25 * 2.0 * np.sqrt(self.kn * m_eff)
        self.coh_dry = 0.0035
        self.coh_wet = 0.030

        self.h_ww = 0.095
        self.kp1 = 0.020
        self.kp2 = 0.140
        self.visc = 0.22
        self.r_ws = 0.075
        self.k_ws = 0.030
        self.drag_ws = 26.0
        self.ka = 14.0
        self.diff_wet = 0.55

        # --- spouts / emission ------------------------------------------
        self.spout_x = (-0.58, 0.58)
        self.spout_y = 0.08
        self.spout_z = 1.75
        self.emit_rate = 70.0          # particles per second per spout
        self.pour_end = 5.2
        self.max_water = 1500
        self.emit_acc = 0.0

        # --- bonds -------------------------------------------------------
        self._build_bonds()

        # --- bookkeeping -------------------------------------------------
        self.t = 0.0
        self.emitted_mass = 0.0
        self.grid_dx = 0.15
        self.gx0 = np.array([-1.65, -1.20, -0.05])
        self.gdim = (22, 16, 16)
        self.src_flat = np.zeros(self.gdim[0] * self.gdim[1] * self.gdim[2])
        self.last_interaction = np.zeros((3, 3))

    def _build_bonds(self) -> None:
        cut = 1.42 * self.spacing
        tree = cKDTree(self.sp)
        pairs = tree.query_pairs(cut, output_type="ndarray")
        if len(pairs) == 0:
            pairs = np.zeros((0, 2), dtype=int)
        self.bi = pairs[:, 0].astype(np.int64)
        self.bj = pairs[:, 1].astype(np.int64)
        d = self.sp[self.bj] - self.sp[self.bi]
        self.d0 = np.linalg.norm(d, axis=1)
        self.d0 = np.maximum(self.d0, 1e-6)
        self.ts = np.zeros((len(self.bi), 3))

    # ------------------------------------------------------------ stepping
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.dt, time - self.t)
            self._substep(dt)
            self.t += dt
        return self._frame(self.t)

    def _emit(self, dt: float) -> None:
        if self.t >= self.pour_end or len(self.wp) >= self.max_water:
            return
        self.emit_acc += self.emit_rate * dt
        n = int(self.emit_acc)
        if n <= 0:
            return
        self.emit_acc -= n
        for k in range(n):
            side = self.spout_x[k % 2]
            jitter = self.rng.normal(0.0, 0.012, size=3)
            p = np.array([side, self.spout_y, self.spout_z]) + jitter
            v = np.array([-np.sign(side) * 0.38, 0.0, -1.85]) + self.rng.normal(0.0, 0.05, 3)
            self.wp = np.vstack((self.wp, p))
            self.wv = np.vstack((self.wv, v))
            self.wc = np.append(self.wc, 1.0)
            self.emitted_mass += self.wmass0
            self._deposit_source(p, self.wmass0)

    def _cell_index(self, pts: np.ndarray) -> np.ndarray:
        idx = np.floor((pts - self.gx0) / self.grid_dx).astype(np.int64)
        idx = np.clip(idx, 0, np.asarray(self.gdim) - 1)
        nx, ny, _ = self.gdim
        return (idx[:, 0] * ny + idx[:, 1]) * self.gdim[2] + idx[:, 2]

    def _deposit_source(self, pts, mass) -> None:
        pts = np.atleast_2d(pts)
        mass = np.atleast_1d(mass)
        flat = self._cell_index(pts)
        np.add.at(self.src_flat, flat, mass)

    def _substep(self, dt: float) -> None:
        self._emit(dt)
        n_w = len(self.wp)
        acc_s = np.zeros_like(self.sp)
        f_ss = np.zeros((len(self.bi), 3))
        f_ws_sum = np.zeros(3)
        f_ww_sum = np.zeros(3)

        # ---------------- sand-sand bonded contacts ----------------------
        if len(self.bi):
            dvec = self.sp[self.bj] - self.sp[self.bi]
            dist = np.linalg.norm(dvec, axis=1)
            dist = np.maximum(dist, 1e-9)
            nhat = dvec / dist[:, None]
            gap = dist - self.d0
            vrel = self.sv[self.bj] - self.sv[self.bi]
            vn = np.einsum("ij,ij->i", vrel, nhat)
            wbar = 0.5 * (self.wet[self.bi] + self.wet[self.bj]) / self.w_max
            coh = self.coh_dry + (self.coh_wet - self.coh_dry) * np.clip(wbar, 0.0, 1.0)
            s = -self.kn * gap - self.cn * vn          # >0 push apart, <0 tension
            s = np.maximum(s, -coh)
            vt = vrel - vn[:, None] * nhat
            # accumulate tangential spring, project out normal drift
            self.ts += vt * dt
            self.ts -= np.einsum("ij,ij->i", self.ts, nhat)[:, None] * nhat
            ft = -self.kt * self.ts
            comp = np.maximum(s, 0.0)
            ft_max = self.friction * comp + coh
            ft_norm = np.linalg.norm(ft, axis=1)
            scale = np.where(ft_norm > 1e-12, np.minimum(1.0, ft_max / np.maximum(ft_norm, 1e-12)), 0.0)
            ft *= scale[:, None]
            self.ts *= scale[:, None]
            f_ss = (s[:, None] * nhat) + ft            # force on grain j
            np.add.at(acc_s, self.bj, f_ss)
            np.add.at(acc_s, self.bi, -f_ss)

        # ---------------- water-water pressure + viscosity ---------------
        dv = np.zeros_like(self.wv) if n_w else None
        if n_w > 1:
            wtree = cKDTree(self.wp)
            pairs = wtree.query_pairs(self.h_ww, output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                dvec = self.wp[j] - self.wp[i]
                dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
                nhat = dvec / dist[:, None]
                q = np.clip(1.0 - dist / self.h_ww, 0.0, 1.0)
                mag = self.kp1 * q + self.kp2 * q * q * q
                f = mag[:, None] * nhat
                np.add.at(dv, j, f)
                np.add.at(dv, i, -f)
                f_ww_sum = f.sum(axis=0)
                # XSPH-like viscosity
                rel = self.wv[i] - self.wv[j]
                vis = self.visc * q[:, None] * rel
                np.add.at(dv, i, -vis)
                np.add.at(dv, j, vis)

        # ---------------- water-sand coupling ----------------------------
        remove = np.zeros(n_w, dtype=bool)
        if n_w:
            stree = cKDTree(self.sp)
            k = min(8, max(1, self.n_sand))
            dist, idx = stree.query(self.wp, k=k)
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            near = dist < self.r_ws
            wgt = np.clip(1.0 - dist / self.r_ws, 0.0, 1.0) * near
            if wgt.any():
                rows, cols = np.nonzero(wgt)
                sj = idx[rows, cols]
                dvec = self.sp[sj] - self.wp[rows]
                dd = np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
                nhat = dvec / dd[:, None]
                qq = wgt[rows, cols]
                f = (self.k_ws * qq)[:, None] * nhat          # pushes water away
                np.add.at(dv, rows, f)
                np.add.at(acc_s, sj, -f)
                f_ws_sum = -f.sum(axis=0)
                # permeability drag on water inside the packing
                drag = np.clip(self.drag_ws * qq.sum(axis=1) * dt, 0.0, 0.9)
                self.wv *= (1.0 - drag)[:, None]

            # ---- wetting: transfer free water mass into grain wetness ----
            rt = 1.25 * self.r_ws
            close = dist < rt
            if close.any():
                cw = np.clip(1.0 - dist / rt, 0.0, 1.0) * close
                capacity_room = np.clip(1.0 - self.wet[idx] / self.w_max, 0.0, 1.0)
                w_pair = cw * capacity_room
                row_sum = w_pair.sum(axis=1)
                avail = np.clip(self.wc, 0.0, 1.0) * self.wmass0
                q_row = np.minimum(self.ka * dt * avail, avail)
                q_row = np.where(row_sum > 1e-12, q_row, 0.0)
                q_pair = q_row[:, None] * np.where(row_sum[:, None] > 1e-12,
                                                   w_pair / np.maximum(row_sum[:, None], 1e-12), 0.0)
                # respect per-grain saturation capacity
                want = np.zeros(self.n_sand)
                np.add.at(want, idx.ravel(), q_pair.ravel())
                cap = np.clip(self.w_max - self.wet, 0.0, self.w_max) * self.smass
                shrink = np.where(want > 1e-15, np.minimum(1.0, cap / np.maximum(want, 1e-15)), 0.0)
                q_pair *= shrink[idx]
                gained = np.zeros(self.n_sand)
                np.add.at(gained, idx.ravel(), q_pair.ravel())
                self.wet = np.clip(self.wet + gained / self.smass, 0.0, self.w_max)
                lost = q_pair.sum(axis=1)
                self.wc = np.clip(self.wc - lost / self.wmass0, 0.0, 1.0)
                remove = self.wc < 0.02

        # ---------------- integration ------------------------------------
        self.sv += (acc_s / self.smass[:, None] + self.gravity) * dt
        self.sp += self.sv * dt
        self._boundaries(self.sp, self.sv, self.sand_r)

        if n_w:
            self.wv += (dv / self.wmass0 + self.gravity) * dt
            self.wp += self.wv * dt
            self._boundaries(self.wp, self.wv, self.water_r)

        # ---------------- wetness capillary diffusion --------------------
        if len(self.bi):
            wi = self.wet[self.bi]
            wj = self.wet[self.bj]
            flux = self.diff_wet * dt * (wj - wi)
            np.add.at(self.wet, self.bi, flux)
            np.add.at(self.wet, self.bj, -flux)
            self.wet = np.clip(self.wet, 0.0, self.w_max)

        self.last_interaction = np.vstack((
            f_ss.sum(axis=0),
            f_ws_sum,
            f_ww_sum,
        ))

        # ---------------- absorb fully drained water grains --------------
        if n_w and remove.any():
            keep = ~remove
            self.wp = self.wp[keep]
            self.wv = self.wv[keep]
            self.wc = self.wc[keep]

    def _boundaries(self, p: np.ndarray, v: np.ndarray, r: float) -> None:
        if not len(p):
            return
        low = p[:, 2] < self.floor_z + r
        p[low, 2] = self.floor_z + r
        vz = v[low, 2]
        v[low, 2] = np.where(vz < 0.0, -self.rest * vz, vz)
        v[low, 0] *= 1.0 - 0.06
        v[low, 1] *= 1.0 - 0.06
        for axis, lim, sgn in ((0, self.wall_x, 1.0), (0, self.wall_x, -1.0),
                               (1, self.wall_y, 1.0), (1, self.wall_y, -1.0)):
            if sgn > 0:
                hit = p[:, axis] > lim - r
                p[hit, axis] = lim - r
                va = v[hit, axis]
                v[hit, axis] = np.where(va > 0.0, -self.rest * va, va)
            else:
                hit = p[:, axis] < -lim + r
                p[hit, axis] = -lim + r
                va = v[hit, axis]
                v[hit, axis] = np.where(va < 0.0, -self.rest * va, va)

    # -------------------------------------------------------------- output
    def _frame(self, time: float):
        n_s, n_w = self.n_sand, len(self.wp)
        n = n_s + n_w
        position = np.vstack((self.sp, self.wp)) if n_w else self.sp.copy()
        velocity = np.vstack((self.sv, self.wv)) if n_w else self.sv.copy()
        body_id = np.concatenate((np.ones(n_s, dtype=np.int32), np.full(n_w, 2, dtype=np.int32)))
        mass = np.concatenate((self.smass, np.full(n_w, self.wmass0 * np.clip(self.wc, 0.0, 1.0))))
        bound = self.wet * self.smass
        free_w = self.wmass0 * np.clip(self.wc, 0.0, 1.0) if n_w else np.zeros(0)
        mass_quantity = np.zeros((n, 2))
        mass_quantity[:n_s, 0] = self.smass + bound
        if n_w:
            mass_quantity[n_s:, 1] = free_w
        solid = np.concatenate((np.ones(n_s), np.zeros(n_w)))
        conc = np.concatenate((self.wet / self.w_max, np.clip(self.wc, 0.0, 1.0) if n_w else np.zeros(0)))
        wet = np.concatenate((self.wet, np.full(n_w, self.w_max)))

        # ---- diagnostic grid ----
        nx, ny, nz = self.gdim
        cell_vol = self.grid_dx ** 3
        scalar = np.zeros(nx * ny * nz)
        mask = np.zeros(nx * ny * nz)
        vel_grid = np.zeros((nx * ny * nz, 3))
        wmass_pts = np.concatenate((bound, free_w))
        flat = self._cell_index(position)
        np.add.at(scalar, flat, wmass_pts)
        np.add.at(mask, flat, 1.0)
        cell_mass = np.zeros(nx * ny * nz)
        np.add.at(vel_grid, flat, mass[:, None] * velocity)
        np.add.at(cell_mass, flat, mass)
        vel_grid /= np.maximum(cell_mass, 1e-12)[:, None]
        source = self.src_flat / max(cell_vol * max(self.sample_dt, 1e-9), 1e-12)
        self.src_flat[:] = 0.0
        scalar = scalar / cell_vol
        mask = np.minimum(mask, 1.0)

        # ---- contact signed distances ----
        csd = []
        if len(self.bi):
            dvec = self.sp[self.bj] - self.sp[self.bi]
            dist = np.linalg.norm(dvec, axis=1)
            gap = dist - self.d0
            sel = gap < 0.25 * self.spacing
            csd.append(gap[sel])
        if n_w:
            stree = cKDTree(self.sp)
            dd, _ = stree.query(self.wp, k=1)
            csd.append(np.atleast_1d(dd) - self.r_ws)
        contact = np.concatenate(csd) if csd else np.zeros(1)

        sand_mass = float((self.smass + bound).sum())
        water_mass = float(bound.sum() + free_w.sum())
        fields = {
            "active_mask": mask.reshape(self.gdim),
            "contact_signed_distance": contact,
            "cumulative_mass_source": np.array([0.0, self.emitted_mass]),
            "interaction_force": self.last_interaction.copy(),
            "scalar_source": source.reshape(self.gdim),
            "transported_scalar": scalar.reshape(self.gdim),
            "velocity_field": vel_grid.reshape(self.gdim + (3,)),
        }
        attributes = {
            "material_id": body_id.copy(),
            "solid_phase_fraction": solid,
            "water_concentration": conc,
            "wetness": wet,
            "mass_quantity": mass_quantity,
        }
        scalars = {
            "contact_length_scale": float(self.spacing),
            "grid_spacing": float(self.grid_dx),
            "sand_mass": sand_mass,
            "water_mass": water_mass,
            "free_water_mass": float(free_w.sum()),
            "bound_water_mass": float(bound.sum()),
            "mean_wetness": float(self.wet.mean()),
            "wet_sand_fraction": float(np.mean(self.wet > 1e-4)),
            "n_water": float(n_w),
        }
        frame = StateFrame(float(time), position, velocity, mass=mass, body_id=body_id,
                           attributes=attributes, fields=fields, scalars=scalars)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "bonded-DEM sand + particle water with wetness transfer",
            "sand_particles": int(self.n_sand),
            "water_particles": int(len(self.wp)),
            "bonds": int(len(self.bi)),
            "emitted_water_mass_kg": float(self.emitted_mass),
            "mean_wetness": float(self.wet.mean()),
            "final_time": float(self.t),
        }
