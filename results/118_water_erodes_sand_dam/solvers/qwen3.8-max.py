"""Two-way coupled sand-dam / water breach solver.

Physics model (particle based, Lagrangian, mass conserving):

* Sand (body_id 1): cohesive granular particles.  Pairwise central repulsion
  (Hertz-like linear spring), velocity-dependent tangential (frictional)
  damping capped by mu*F_normal, and a cohesion attraction whose strength
  depends on the *simulated* local water saturation S:
      C_ij = (1 - 0.5*(S_i+S_j))**2   (dry = strong, saturated ~ 0).
  Saturation S is a dynamic per-sample state advanced by local water contact
  (wetting) and sand-sand capillary diffusion -- never scripted in time.

* Water (body_id 2): weakly compressible particle fluid.  Pairwise repulsion
  (equation of state), normal + tangential viscosity, small global damping.

* Coupling: sand-water mixed pairs exchange momentum through a symmetric
  drag  F = c_d * w(r) * (v_j - v_i)  (equal and opposite -> two way), plus
  a soft mixed repulsion.  Water forces therefore move/erode sand, and the
  evolving sand skeleton (including the growing breach) deflects/resists the
  resolved water flow.  No solid mass is ever deleted.

* Boundaries: penalty contact against the frozen flume planes (floor z=0,
  side walls y=+-0.95, reservoir wall x=-3.93, retaining wall x=4.43) with
  normal damping and Coulomb-capped floor friction.

Integration: symplectic Euler at fixed dt with a uniform-grid half-stencil
neighbour search rebuilt every substep.  All budget fields (external force
terms, cumulative impulses, interaction forces, mass sources) are integrated
at substep resolution and exported per frame.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    SAND_ID = 1
    WATER_ID = 2

    # ---- numerics -------------------------------------------------------
    DT = 0.0025            # substep
    CELL = 0.20            # uniform grid cell (>= max interaction radius)
    V_MAX = 6.0            # velocity clamp (safety)

    # ---- interaction radii / stiffness per pair type (0 ss, 1 ww, 2 sw) --
    H_RAD = np.array([0.15, 0.18, 0.17])
    K_REP = np.array([1.2e4, 1.0e4, 3.0e3])

    K_COH = 1.0e3          # dry sand cohesion bond force [N]
    MU_SAND = 0.65         # sand friction cap
    C_T_SAND = 12.0        # sand tangential damping
    C_VN_WAT = 6.0         # water normal viscosity
    C_VT_WAT = 1.5         # water tangential viscosity
    C_DRAG = 10.0          # sand-water drag coefficient (two way)

    K_WALL = 4.0e4
    C_WALL = 300.0
    C_FLOOR_T = 150.0
    MU_FLOOR = 0.65

    C_DAMP_SAND = 1.0      # global linear damping rates [1/s]
    C_DAMP_WAT = 0.4

    K_WET = 1.2            # saturation rate from water contact [1/s]
    WET_REF = 5.0          # kernel-weight normalisation for full wetting
    K_DIFF = 0.4           # saturation diffusion between sand [1/s]

    GRAV = np.array([0.0, 0.0, -9.81])

    # domain interior (frozen flume planes)
    X_MIN, X_MAX = -3.93, 4.43
    Y_MIN, Y_MAX = -0.95, 0.95

    # half stencil offsets (13 of 26; self cell handled by sorted shifts)
    _OFFS = np.array(
        [(dx, dy, dz)
         for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
         if (dx, dy, dz) != (0, 0, 0)
         and (dx > 0 or (dx == 0 and (dy > 0 or (dy == 0 and dz > 0))))],
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        self.scene = scene
        self.seed = int(seed)

        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.body_id = np.array(frame0.body_id)
        self.n = len(self.pos)

        self.is_sand = self.body_id == self.SAND_ID
        self.is_water = ~self.is_sand
        self.sat = np.array(frame0.attributes["sand_saturation"], dtype=np.float64)
        self.porosity = np.array(frame0.attributes["porosity"], dtype=np.float64)
        self.pvol = np.array(frame0.attributes["particle_volume"], dtype=np.float64)

        self.m_sand_tot = float(self.mass[self.is_sand].sum())
        self.m_water_tot = float(self.mass[self.is_water].sum())
        m_tot = self.m_sand_tot + self.m_water_tot
        # frozen positive momentum scale (NOT the initial net momentum)
        self.reference_momentum = float(m_tot * np.sqrt(9.81 * 1.75))

        self.fg_sand = self.m_sand_tot * self.GRAV
        self.fg_water = self.m_water_tot * self.GRAV

        # budget domains: 0 = sand, 1 = water, 2 = combined
        self.cum_impulse = np.zeros((3, 3))
        self.fb_sand_last = np.zeros(3)
        self.fb_water_last = np.zeros(3)
        self.drag_sand_last = np.zeros(3)

        self.domain_weights = np.stack([
            self.is_sand.astype(np.float64),
            self.is_water.astype(np.float64),
            np.ones(self.n),
        ])

        self.t = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------------
    # neighbour search (uniform grid, 13-offset half stencil + self cell)
    def _build_pairs(self):
        pos = self.pos
        cs = self.CELL
        ix = np.floor((pos[:, 0] + 4.6) / cs).astype(np.int64)
        iy = np.floor((pos[:, 1] + 1.2) / cs).astype(np.int64)
        iz = np.floor((pos[:, 2] + 0.2) / cs).astype(np.int64)
        key = (ix * 64 + iy) * 64 + iz

        order = np.argsort(key, kind="stable")
        skey = key[order]
        uniq, first, cnt = np.unique(skey, return_index=True, return_counts=True)

        p_i = []
        p_j = []

        # self-cell pairs: within each equal-key run take (a, a+k)
        maxc = int(cnt.max()) if len(cnt) else 0
        nn = len(skey)
        for k in range(1, maxc + 1):
            a = np.arange(nn - k)
            m = skey[a] == skey[a + k]
            if not m.any():
                continue
            sel = a[m]
            p_i.append(order[sel])
            p_j.append(order[sel + k])

        # 13 forward neighbour cells
        n = self.n
        ar_n = np.arange(n, dtype=np.int64)
        nuniq = len(uniq)
        for d in self._OFFS:
            dkey = (d[0] * 64 + d[1]) * 64 + d[2]
            nkey = key + dkey
            idx = np.searchsorted(uniq, nkey)
            valid = idx < nuniq
            idxc = np.where(valid, idx, 0)
            valid &= uniq[idxc] == nkey
            c = np.where(valid, cnt[idxc], 0).astype(np.int64)
            total = int(c.sum())
            if total == 0:
                continue
            f = np.where(valid, first[idxc], 0).astype(np.int64)
            seg = np.cumsum(c) - c
            rep_f = np.repeat(f, c)
            within = np.arange(total, dtype=np.int64) - np.repeat(seg, c)
            p_i.append(np.repeat(ar_n, c))
            p_j.append(order[rep_f + within])

        if not p_i:
            e = np.empty(0, dtype=np.int64)
            return e, e
        return np.concatenate(p_i), np.concatenate(p_j)

    # ------------------------------------------------------------------
    def _pair_forces(self, dt):
        """Returns (force[N,3], wall_force[N,3], drag_on_sand[3]) and
        advances the saturation state."""
        n = self.n
        pos, vel = self.pos, self.vel
        pi, pj = self._build_pairs()
        force = np.zeros((n, 3))
        drag_sand = np.zeros(3)

        if len(pi):
            d = pos[pj] - pos[pi]
            r2 = np.einsum("ij,ij->i", d, d)
            isand = self.is_sand
            both_sand = isand[pi] & isand[pj]
            both_wat = (~isand[pi]) & (~isand[pj])
            tt = np.where(both_sand, 0, np.where(both_wat, 1, 2)).astype(np.int64)
            hp = self.H_RAD[tt]
            act = (r2 < hp * hp) & (r2 > 1e-12)
            if act.any():
                ia = pi[act]
                ja = pj[act]
                ta = tt[act]
                dv = d[act]
                ra = np.sqrt(r2[act])
                rhat = dv / ra[:, None]
                w = 1.0 - ra / self.H_RAD[ta]
                vrel = vel[ja] - vel[ia]
                vn = np.einsum("ij,ij->i", vrel, rhat)
                vt = vrel - vn[:, None] * rhat

                fn = self.K_REP[ta] * w                # repulsion magnitude
                F = -fn[:, None] * rhat                # on i (pushes apart)

                sat = self.sat
                m0 = ta == 0                            # sand-sand
                m1 = ta == 1                            # water-water
                m2 = ta == 2                            # sand-water

                # cohesion (saturation dependent)
                cij = np.zeros(len(ta))
                sbar = 0.5 * (sat[ia] + sat[ja])
                cij[m0] = (1.0 - sbar[m0]) ** 2
                coh = self.K_COH * cij * w
                F[m0] += coh[m0, None] * rhat[m0]

                # water normal viscosity
                F[m1] += (self.C_VN_WAT * vn[m1])[:, None] * rhat[m1]

                # tangential damping / friction
                c_t = np.where(m0, self.C_T_SAND, np.where(m1, self.C_VT_WAT, 0.0))
                ft = c_t[:, None] * vt
                if m0.any():
                    ft_n = np.linalg.norm(ft[m0], axis=1)
                    cap = self.MU_SAND * (fn[m0] + coh[m0])
                    sc = np.ones_like(ft_n)
                    bad = ft_n > 1e-9
                    sc[bad] = np.minimum(1.0, cap[bad] / ft_n[bad])
                    ft[m0] *= sc[:, None]
                F += ft

                # two-way sand-water drag
                if m2.any():
                    fd = (self.C_DRAG * w[m2])[:, None] * vrel[m2]
                    F[m2] += fd
                    sand_on_i = self.is_sand[ia[m2]]
                    signed = np.where(sand_on_i[:, None], fd, -fd)
                    drag_sand = signed.sum(axis=0)

                for ax in range(3):
                    force[:, ax] += np.bincount(ia, weights=F[:, ax], minlength=n)
                    force[:, ax] -= np.bincount(ja, weights=F[:, ax], minlength=n)

                # ---- saturation update (wetting + diffusion) ----
                sidx = np.where(self.is_sand[ia], ia, ja)
                wet = np.bincount(sidx[m2], weights=w[m2], minlength=n)
                ds = dt * self.K_WET * np.minimum(wet / self.WET_REF, 1.0) \
                    * (1.0 - sat) * self.is_sand
                if m0.any():
                    i0 = ia[m0]
                    j0 = ja[m0]
                    w0 = w[m0]
                    sdiff = w0 * (sat[j0] - sat[i0])
                    acc = (np.bincount(i0, weights=sdiff, minlength=n)
                           - np.bincount(j0, weights=sdiff, minlength=n))
                    ds += dt * self.K_DIFF * acc * self.is_sand
                sat += ds
                np.clip(sat, 0.0, 1.0, out=sat)
                sat[~self.is_sand] = 1.0

        # ---- boundary penalty forces (frozen flume planes) ----
        fb = np.zeros((n, 3))
        kw, cw = self.K_WALL, self.C_WALL

        pen = np.maximum(0.0, -pos[:, 2])
        f = kw * pen + cw * np.maximum(0.0, -vel[:, 2])
        f = np.where(pen > 0.0, f, 0.0)
        fb[:, 2] += f
        # floor friction (Coulomb capped tangential damping)
        cont = pen > 1e-9
        vh = vel[:, :2]
        sp = np.linalg.norm(vh, axis=1)
        fm = np.minimum(self.C_FLOOR_T * sp, self.MU_FLOOR * f)
        fm = np.where(cont & (sp > 1e-6), fm, 0.0)
        fb[:, :2] -= (fm / np.maximum(sp, 1e-9))[:, None] * vh

        pen = np.maximum(0.0, self.Y_MIN - pos[:, 1])
        f = kw * pen + cw * np.maximum(0.0, -vel[:, 1])
        fb[:, 1] += np.where(pen > 0.0, f, 0.0)
        pen = np.maximum(0.0, pos[:, 1] - self.Y_MAX)
        f = kw * pen + cw * np.maximum(0.0, vel[:, 1])
        fb[:, 1] -= np.where(pen > 0.0, f, 0.0)
        pen = np.maximum(0.0, self.X_MIN - pos[:, 0])
        f = kw * pen + cw * np.maximum(0.0, -vel[:, 0])
        fb[:, 0] += np.where(pen > 0.0, f, 0.0)
        pen = np.maximum(0.0, pos[:, 0] - self.X_MAX)
        f = kw * pen + cw * np.maximum(0.0, vel[:, 0])
        fb[:, 0] -= np.where(pen > 0.0, f, 0.0)

        force += fb

        # gravity + global damping
        force += self.mass[:, None] * self.GRAV
        damp = np.where(self.is_sand, self.C_DAMP_SAND, self.C_DAMP_WAT)
        force -= (damp * self.mass)[:, None] * vel

        return force, fb, drag_sand

    # ------------------------------------------------------------------
    def _substep(self, dt):
        force, fb, drag_sand = self._pair_forces(dt)

        acc = force / self.mass[:, None]
        self.vel += acc * dt
        sp = np.linalg.norm(self.vel, axis=1)
        over = sp > self.V_MAX
        if over.any():
            self.vel[over] *= (self.V_MAX / sp[over])[:, None]
        self.pos += self.vel * dt

        # hard containment (safety net behind the penalty walls)
        p, v = self.pos, self.vel
        lo = p[:, 0] < self.X_MIN + 1e-3
        p[lo, 0] = self.X_MIN + 1e-3
        v[lo, 0] = np.maximum(v[lo, 0], 0.0)
        hi = p[:, 0] > self.X_MAX - 1e-3
        p[hi, 0] = self.X_MAX - 1e-3
        v[hi, 0] = np.minimum(v[hi, 0], 0.0)
        lo = p[:, 1] < self.Y_MIN + 1e-3
        p[lo, 1] = self.Y_MIN + 1e-3
        v[lo, 1] = np.maximum(v[lo, 1], 0.0)
        hi = p[:, 1] > self.Y_MAX - 1e-3
        p[hi, 1] = self.Y_MAX - 1e-3
        v[hi, 1] = np.minimum(v[hi, 1], 0.0)
        lo = p[:, 2] < 1e-3
        p[lo, 2] = 1e-3
        v[lo, 2] = np.maximum(v[lo, 2], 0.0)
        hi = p[:, 2] > 3.2
        p[hi, 2] = 3.2
        v[hi, 2] = np.minimum(v[hi, 2], 0.0)

        # budget accumulation at substep resolution
        fb_s = fb[self.is_sand].sum(axis=0)
        fb_w = fb[self.is_water].sum(axis=0)
        self.cum_impulse[0] += (self.fg_sand + fb_s + drag_sand) * dt
        self.cum_impulse[1] += (self.fg_water + fb_w - drag_sand) * dt
        self.cum_impulse[2] += (self.fg_sand + self.fg_water + fb_s + fb_w) * dt
        self.fb_sand_last = fb_s
        self.fb_water_last = fb_w
        self.drag_sand_last = drag_sand

        self.t += dt
        self.n_steps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time > self.t + 1e-12:
            n_sub = max(1, int(round((time - self.t) / self.DT)))
            dt = (time - self.t) / n_sub
            for _ in range(n_sub):
                self._substep(dt)
        return self._make_frame(time)

    def _make_frame(self, time):
        drag = self.drag_sand_last
        fb_s, fb_w = self.fb_sand_last, self.fb_water_last
        ext = np.zeros((3, 3, 3))
        ext[0, 0] = self.fg_sand
        ext[0, 1] = fb_s
        ext[0, 2] = drag
        ext[1, 0] = self.fg_water
        ext[1, 1] = fb_w
        ext[1, 2] = -drag
        ext[2, 0] = self.fg_sand + self.fg_water
        ext[2, 1] = fb_s + fb_w
        ext[2, 2] = 0.0

        interaction = np.stack([drag, -drag])
        zeros_g3 = np.zeros((3, 3))

        frame = StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "material_id": self.body_id.astype(np.float64).copy(),
                "sand_saturation": self.sat.copy(),
                "porosity": self.porosity.copy(),
                "particle_volume": self.pvol.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": ext,
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3.copy(),
                "cumulative_mass_source": np.zeros(2),
                "interaction_force": interaction,
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "initial_sand_mass": self.m_sand_tot,
                "initial_water_mass": self.m_water_tot,
                "sand_mass": float(self.mass[self.is_sand].sum()),
                "water_mass": float(self.mass[self.is_water].sum()),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        sand = self.is_sand
        return {
            "substeps": int(self.n_steps),
            "n_particles": int(self.n),
            "sand_particles": int(sand.sum()),
            "water_particles": int((~sand).sum()),
            "max_sand_saturation": float(self.sat[sand].max()) if sand.any() else 0.0,
            "mean_sand_saturation": float(self.sat[sand].mean()) if sand.any() else 0.0,
            "sand_centroid_x": float(self.pos[sand, 0].mean()) if sand.any() else 0.0,
            "sand_mass": float(self.mass[sand].sum()),
        }
