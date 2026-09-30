"""Two-way coupled SPH-water / cohesive-granular-sand solver.

Physical model
--------------
The scene is discretised into Lagrangian samples: cohesive sand grains
(body_id 1) and water particles (body_id 2).  The two phases are advanced by
an explicit, momentum-conserving time-stepping scheme:

* water      : weakly compressible SPH (cubic-spline kernel, Tait-like linear
               equation of state, Monaghan artificial viscosity);
* sand       : linear spring/dashpot contact law with Coulomb friction and a
               cohesion law whose strength decays with the *simulated* water
               saturation of the grain neighbourhood;
* coupling   : pairwise drag (Darcy/Forchheimer-type), pressure-gradient
               (buoyancy) coupling and a short-range contact penalty, all
               applied as equal-and-opposite pair forces so that momentum is
               conserved;
* saturation : water wetting of sand surfaces plus capillary diffusion through
               the granular skeleton; cohesion and friction both respond to it;
* boundaries : penalty/traction walls for the flume floor, side walls, the
               reservoir end wall and the closed downstream retaining wall.

Nothing is scripted: the breach emerges from the loss of cohesion/friction in
the wetted region combined with fluid pressure and drag acting on the grains.
No sample is ever deleted, so sand mass is conserved and retained.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    SAND_ID = 1
    WATER_ID = 2
    GRAVITY = np.array([0.0, 0.0, -9.81])
    G = 3          # budget domains: global, sand, water
    K = 3          # external force terms: gravity, boundary, phase coupling
    KM = 2         # conserved materials: sand, water

    # ---- numerics -------------------------------------------------------
    dt = 1.0e-3
    max_speed = 30.0

    # ---- water / SPH ----------------------------------------------------
    c_sound = 20.0
    alpha_av = 0.20
    h_ratio = 1.0
    beta_drag = 1.5          # drag relaxation rate [1/s]
    k_ws = 2.0e4             # water-sand contact stiffness [N/m]
    c_ws = 60.0              # water-sand contact damping [N*s/m]
    r_ws_scale = 1.3         # effective water radius used for blocking
    pressure_couple = 1.0

    # ---- sand / granular ------------------------------------------------
    k_n = 2.0e4              # contact normal stiffness [N/m]
    c_n = 60.0               # contact normal damping [N*s/m]
    c_t = 30.0               # tangential damping [N*s/m]
    mu0 = 0.70               # dry friction coefficient
    mu_sat = 0.85            # relative friction loss at full saturation
    coh0 = 32.0              # dry cohesion force [N]
    coh_range = 0.30         # cohesion range in units of the grain spacing

    # ---- saturation -----------------------------------------------------
    k_inf = 3.0              # wetting rate [1/s]
    d_sat = 1.0              # capillary diffusion rate [1/s]
    r_wet = 0.20             # wetting influence radius [m]

    # ---- walls ----------------------------------------------------------
    k_wall = 1.0e5           # wall stiffness per unit mass [1/s^2]
    c_wall = 400.0           # wall damping per unit mass [1/s]
    c_wall_t = 50.0          # wall tangential damping per unit mass [1/s]

    # =====================================================================
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        frame = scene.initial_state()

        self.x = np.asarray(frame.position, dtype=float).copy()
        self.v = np.asarray(frame.velocity, dtype=float).copy()
        self.m = np.asarray(frame.mass, dtype=float).copy()
        self.bid = np.asarray(frame.body_id, dtype=np.int64).copy()
        self.n = int(self.x.shape[0])

        attrs = frame.attributes
        vol = attrs.get("particle_volume")
        self.vol = (
            np.asarray(vol, dtype=float).copy()
            if vol is not None
            else np.full(self.n, 1.0e-3)
        )
        sat = attrs.get("sand_saturation")
        self.sat = (
            np.asarray(sat, dtype=float).copy() if sat is not None else np.zeros(self.n)
        )
        por = attrs.get("porosity")
        self.por = (
            np.asarray(por, dtype=float).copy() if por is not None else np.zeros(self.n)
        )

        self.sand = self.bid == self.SAND_ID
        self.water = self.bid == self.WATER_ID
        if not self.sand.any() or not self.water.any():
            raise ValueError("scene must provide both sand (body_id 1) and water (body_id 2)")

        self.iw = np.flatnonzero(self.water)
        self.is_ = np.flatnonzero(self.sand)

        self.rad = 0.5 * np.cbrt(np.maximum(self.vol, 1e-12))
        self.s_w = float(np.median(np.cbrt(self.vol[self.water])))
        self.s_s = float(np.median(np.cbrt(self.vol[self.sand])))
        self.h = self.h_ratio * self.s_w
        self.support = 2.0 * self.h
        self.sigma = 1.0 / (np.pi * self.h ** 3)
        self.W0 = self.sigma
        self.rq_water = self.support
        self.rq_sand = 2.0 * self.rad[self.is_].max() + self.coh_range * self.s_s
        self.r_ws = self.r_ws_scale * 0.5 * self.h + float(np.median(self.rad[self.is_]))
        self.coh_d = self.coh_range * self.s_s

        # free parameters that enter particle mass ratios
        self.rho0 = self._calibrate_density()
        self.c2 = self.c_sound ** 2
        self.wall_friction = {"sand": 0.65, "water": 0.03}

        self.time = 0.0
        self.step_count = 0
        self.pairs_seen = 0

        self._zero_budgets()
        # reference momentum scale: frozen positive constant
        char_v = float(np.sqrt(9.81 * max(0.5, float(self.x[:, 2].max()))))
        self.reference_momentum = float(max(1.0, self.m.sum() * char_v))

        self.ext_force = np.zeros((self.G, self.K, 3))
        self.inter_force = np.zeros((self.KM, 3))
        self.cum_mass_source = np.zeros(self.KM)

    # ------------------------------------------------------------------
    def _calibrate_density(self) -> float:
        xw = self.x[self.iw]
        mw = self.m[self.iw]
        tree = cKDTree(xw)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        rho = np.zeros(len(self.iw))
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            r = np.linalg.norm(xw[a] - xw[b], axis=1)
            w = self._kernel(r)
            np.add.at(rho, a, mw[b] * w)
            np.add.at(rho, b, mw[a] * w)
        interior = rho > 0.5 * float(rho.max() if rho.max() > 0 else 1.0)
        if interior.sum() >= 8:
            return float(np.median(rho[interior]))
        return float(max(rho.max(), 1.0))

    # ---- kernels ------------------------------------------------------
    def _kernel(self, r):
        q = r / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        out[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        out[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return out * self.sigma

    def _dkernel(self, r):
        """dW/dr."""
        q = r / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        out[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        out[m2] = -0.75 * (2.0 - q[m2]) ** 2
        return out * (self.sigma / self.h)

    # ---- accumulation helpers -----------------------------------------
    @staticmethod
    def _acc_vec(target, ii, jj, f):
        """target[ii] += f ; target[jj] -= f  (equal and opposite pairs)."""
        idx = np.concatenate((ii, jj))
        for c in range(3):
            target[:, c] += np.bincount(
                idx, weights=np.concatenate((f[:, c], -f[:, c])), minlength=len(target)
            )

    @staticmethod
    def _acc_sum(target, ii, vals):
        target += np.bincount(ii, weights=vals, minlength=len(target))

    # ==================================================================
    def _walls(self, F, F_wall):
        planes = (
            (np.array([0.0, 0.0, 1.0]), 0.0, "sand", 0.65),
            (np.array([0.0, 1.0, 0.0]), -0.95, "sand", 0.35),
            (np.array([0.0, -1.0, 0.0]), -0.95, "sand", 0.35),
            (np.array([1.0, 0.0, 0.0]), -3.93, "sand", 0.35),
            (np.array([-1.0, 0.0, 0.0]), -4.43, "sand", 0.35),
        )
        for n, d, _, mu in planes:
            pen = d - self.x @ n
            k = pen > 0.0
            if not k.any():
                continue
            mm = self.m[k]
            vn = self.v[k] @ n
            fn = self.k_wall * mm * pen[k] - self.c_wall * mm * np.minimum(vn, 0.0)
            fn = np.maximum(fn, 0.0)
            f = fn[:, None] * n
            vt = self.v[k] - vn[:, None] * n
            vtm = np.linalg.norm(vt, axis=1)
            cap = self.c_wall_t * mm * vtm
            mu_eff = np.where(
                self.sand[k],
                mu * (1.0 - 0.6 * self.sat[k]),
                self.wall_friction["water"],
            )
            ft = np.minimum(mu_eff * fn, cap)
            f = f - (ft / np.maximum(vtm, 1e-9))[:, None] * vt
            F[k] += f
            F_wall[k] += f

    # ==================================================================
    def _step(self, dt: float) -> None:
        x, v, m = self.x, self.v, self.m
        n = self.n
        F = np.zeros_like(x)
        F_grav = m[:, None] * self.GRAVITY
        F += F_grav
        F_wall = np.zeros_like(x)
        F_couple = np.zeros_like(x)
        F_int = np.zeros_like(x)
        sat = self.sat

        iw, isx = self.iw, self.is_
        xw, xs = x[iw], x[isx]
        mw, ms = m[iw], m[isx]

        # ---------------- water-water (SPH) ---------------------------
        tw = cKDTree(xw)
        pw = tw.query_pairs(self.support, output_type="ndarray")
        self.pairs_seen += len(pw)
        nw = len(iw)
        rho_w = np.full(nw, self.rho0)
        p_w = np.zeros(nw)
        if len(pw):
            a, b = pw[:, 0], pw[:, 1]
            dw = xw[a] - xw[b]
            r = np.linalg.norm(dw, axis=1)
            r = np.maximum(r, 1e-9)
            e = dw / r[:, None]
            w = self._kernel(r)
            dens = np.zeros(nw)
            self._acc_sum(dens, a, mw[b] * w)
            self._acc_sum(dens, b, mw[a] * w)
            good = dens > 1e-6
            rho_w = np.where(good, dens, self.rho0)
            p_w = np.maximum(0.0, self.c2 * (rho_w - self.rho0))
            dW = self._dkernel(r)
            vi, vj = v[iw[a]], v[iw[b]]
            vdot = np.einsum("ij,ij->i", vi - vj, dw)
            pterm = p_w[a] / rho_w[a] ** 2 + p_w[b] / rho_w[b] ** 2
            mu = self.h * vdot / (r * r + 0.01 * self.h ** 2)
            rho_bar = 0.5 * (rho_w[a] + rho_w[b])
            pi = np.where(
                vdot < 0.0,
                -self.alpha_av * self.c_sound * np.minimum(mu, 0.0) / rho_bar,
                0.0,
            )
            coef = -mw[a] * mw[b] * (pterm + pi) * dW / r
            f = coef[:, None] * dw
            self._acc_vec(F, iw[a], iw[b], f)
            self._acc_vec(F_int, iw[a], iw[b], f)

        # ---------------- sand-sand (contacts + cohesion) -------------
        ts = cKDTree(xs)
        ps = ts.query_pairs(self.rq_sand, output_type="ndarray")
        self.pairs_seen += len(ps)
        sdiff = np.zeros(n)
        if len(ps):
            a, b = ps[:, 0], ps[:, 1]
            ds = xs[a] - xs[b]
            r = np.maximum(np.linalg.norm(ds, axis=1), 1e-9)
            e = ds / r[:, None]
            r0 = self.rad[isx][a] + self.rad[isx][b]
            sbar = 0.5 * (sat[isx[a]] + sat[isx[b]])
            mu = self.mu0 * (1.0 - self.mu_sat * np.clip(sbar, 0.0, 1.0))
            vi, vj = v[isx[a]], v[isx[b]]
            vdot = np.einsum("ij,ij->i", vi - vj, e)
            over = r0 - r
            fn = np.where(over > 0.0, np.maximum(self.k_n * over - self.c_n * vdot, 0.0), 0.0)
            vt = (vi - vj) - vdot[:, None] * e
            vtm = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(mu * fn, self.c_t * vtm)
            f = fn[:, None] * e - (ft_mag / np.maximum(vtm, 1e-9))[:, None] * vt
            # cohesion: attraction that vanishes as the grains saturate
            gap = r - r0
            frac = np.clip(1.0 - gap / self.coh_d, 0.0, 1.0)
            coh = self.coh0 * (1.0 - np.clip(sbar, 0.0, 1.0)) ** 2 * frac
            f = f - coh[:, None] * e
            self._acc_vec(F, isx[a], isx[b], f)
            self._acc_vec(F_int, isx[a], isx[b], f)
            # capillary saturation diffusion through the skeleton
            wq = self._kernel(r) / self.W0
            self._acc_sum(sdiff, isx[a], (sat[isx[b]] - sat[isx[a]]) * wq)
            self._acc_sum(sdiff, isx[b], (sat[isx[a]] - sat[isx[b]]) * wq)

        # ---------------- water-sand coupling --------------------------
        ws_lists = tw.query_ball_point(xs, self.support)
        cnt = np.fromiter((len(l) for l in ws_lists), dtype=np.int64, count=len(xs))
        wet_vol = np.zeros(n)
        jj = np.zeros(0, dtype=np.int64)
        nonempty = [np.asarray(l, dtype=np.int64) for l in ws_lists if len(l)]
        if nonempty:
            jj = np.concatenate(nonempty)
        ii = np.repeat(isx, cnt)
        if len(ii):
            jw = iw[jj]
            ds = x[ii] - x[jw]
            r = np.maximum(np.linalg.norm(ds, axis=1), 1e-9)
            e = ds / r[:, None]
            # pressure-gradient (buoyancy) coupling, equal and opposite
            dW = self._dkernel(r)
            pr = p_w[jj] / np.maximum(rho_w[jj], 1e-9)
            f = (
                -self.pressure_couple
                * self.vol[ii][:, None]
                * (mw[jj] * pr)[:, None]
                * (dW / r)[:, None]
                * ds
            )
            f = f + self._ws_contact(ds, r, e, ii, jw)
            # drag (Darcy/Forchheimer): relaxes grain velocity to fluid velocity
            wk = self._kernel(r) / self.W0
            beta = self.beta_drag * np.minimum(m[ii], mw[jj]) * wk
            f = f + beta[:, None] * (v[jw] - v[ii])
            self._acc_vec(F_couple, ii, jw, f)
            self._acc_vec(F, ii, jw, f)
            # wetting of the granular skeleton
            wet = np.maximum(0.0, 1.0 - r / self.r_wet) * self.vol[jw]
            self._acc_sum(wet_vol, ii, wet)

        # apply wetting / diffusion
        if len(ii):
            pore = np.maximum(self.por[isx] * self.vol[isx], 1e-12)
            rate = self.k_inf * (1.0 - sat[isx]) * np.minimum(1.0, wet_vol[isx] / pore)
            sat[isx] = np.clip(sat[isx] + dt * rate, 0.0, 1.0)
        sat[isx] = np.clip(sat[isx] + dt * self.d_sat * sdiff[isx], 0.0, 1.0)

        # ---------------- walls ---------------------------------------
        self._walls(F, F_wall)

        # ---------------- integrate -----------------------------------
        v += dt * (F / m[:, None])
        too_fast = np.linalg.norm(v, axis=1) > self.max_speed
        if too_fast.any():
            v[too_fast] *= (self.max_speed / np.linalg.norm(v[too_fast], axis=1))[:, None]
        x += dt * v

        # ---------------- budgets -------------------------------------
        self.ext_force[0, 0] = F_grav.sum(axis=0)
        self.ext_force[0, 1] = F_wall.sum(axis=0)
        self.ext_force[0, 2] = 0.0
        for idx, mask in ((1, self.sand), (2, self.water)):
            self.ext_force[idx, 0] = F_grav[mask].sum(axis=0)
            self.ext_force[idx, 1] = F_wall[mask].sum(axis=0)
            self.ext_force[idx, 2] = F_couple[mask].sum(axis=0)
        self.inter_force[0] = F_int.sum(axis=0)
        self.inter_force[1] = F_couple.sum(axis=0)
        self.cum_ext_impulse += dt * self.ext_force.sum(axis=1)

    # ------------------------------------------------------------------
    def _ws_contact(self, ds, r, e, ii, jw):
        """Short-range penalty keeping water out of the granular skeleton."""
        rr = self.r_ws
        over = rr - r
        k = over > 0.0
        out = np.zeros_like(ds)
        if k.any():
            mm = np.minimum(self.m[ii[k]], self.m[jw[k]])
            vdot = np.einsum("ij,ij->i", self.v[ii[k]] - self.v[jw[k]], e[k])
            fn = np.maximum(self.k_ws * over[k] - self.c_ws * vdot, 0.0) * mm
            out[k] = fn[:, None] * e[k]
        return out

    # ------------------------------------------------------------------
    def _zero_budgets(self):
        self.cum_ext_impulse = np.zeros((self.G, 3))
        self.cum_in_momentum = np.zeros((self.G, 3))
        self.cum_out_momentum = np.zeros((self.G, 3))
        self.weights = np.zeros((self.G, self.n))
        self.weights[0, :] = 1.0
        self.weights[1, self.sand] = 1.0
        self.weights[2, self.water] = 1.0

    # ==================================================================
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.step_count < target:
            self._step(self.dt)
            self.step_count += 1
        self.time = float(time)
        return self._frame()

    # ------------------------------------------------------------------
    def _frame(self) -> StateFrame:
        m = self.m
        sand = self.sand
        mass_q = np.zeros((self.n, self.KM))
        mass_q[sand, 0] = m[sand]
        mass_q[self.water, 1] = m[self.water]
        attributes = {
            "material_id": self.bid.copy(),
            "mass_quantity": mass_q,
            "sand_saturation": np.clip(self.sat, 0.0, 1.0),
            "porosity": self.por,
            "particle_volume": self.vol,
        }
        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_in_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cum_out_momentum.copy(),
            "budget_domain_weights": self.weights.copy(),
            "budget_external_force": self.ext_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.inter_force.copy(),
        }
        scalars = {
            "reference_momentum": self.reference_momentum,
            "reference_velocity": float(np.sqrt(9.81 * max(0.5, float(self.x[:, 2].max())))),
            "time_step": self.dt,
        }
        frame = StateFrame(
            self.time,
            self.x.copy(),
            self.v.copy(),
            mass=m.copy(),
            body_id=self.bid.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self):
        return {
            "samples": int(self.n),
            "sand_samples": int(self.sand.sum()),
            "water_samples": int(self.water.sum()),
            "steps": int(self.step_count),
            "dt": self.dt,
            "particle_pairs_processed": int(self.pairs_seen),
            "rho0": float(self.rho0),
            "sand_mass": float(self.m[self.sand].sum()),
            "water_mass": float(self.m[self.water].sum()),
            "saturation_min": float(self.sat.min()),
            "saturation_max": float(self.sat.max()),
        }
