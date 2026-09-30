"""Physically-based particle solver for the egg-frying scene (051).

Model summary
-------------
* Lagrangian particles (positions taken from the frozen scene scaffold) carry
  fixed per-material masses and one-hot material concentrations that never mix
  (concentration diffusion disabled).
* Momentum: pairwise contact springs (repulsion at rest spacing), a weak
  cohesion band, normal damping (viscosity rising with cooked fraction),
  gravity, hot-pan contact forces (normal penalty, adhesion/wetting,
  friction growing as the egg solidifies) and a soft cylindrical rim.
* Thermal: pan-contact Newtonian heating (443 K surface), pairwise heat
  conduction, weak ambient cooling.
* Phase change: first-order coagulation kinetics phi' = k(T)(1-phi) with
  material-specific onset temperature; cooked same-material neighbours form
  elastic bonds (network locking) -> phase front propagates from the pan up.
* Phase diffusion (enabled) smooths solid_phase_fraction between
  same-material neighbours only; concentrations remain untouched.

The white pool relaxes from its initially thick deposited shape toward a
near-monolayer film at its rest spacing, spreading to ~0.12 m radius, while
the cohesive yolk keeps its dome; both cook from the hot boundary upward.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_G = 9.81


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        ic = cfg["initial_conditions"]
        bc = cfg["boundary_conditions"]
        frame0 = scene.initial_state()

        self.seed = int(seed)
        self.dt = 1.0e-3
        self.t = 0.0

        self.pos = np.array(frame0.position, dtype=float, copy=True)
        self.vel = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.body = np.array(frame0.body_id, dtype=np.int64, copy=True)
        self.n = len(self.pos)

        # Symmetry-breaking jitter from the seed (physical: cracking churn).
        rng = np.random.default_rng(self.seed)
        self.vel += rng.normal(0.0, 0.008, size=self.vel.shape)
        self.vel[:, 2] *= 0.0

        self.is_white = self.body == 1
        self.is_yolk = self.body == 2

        # material concentrations: one-hot, frozen (no concentration diffusion)
        conc0 = np.asarray(frame0.attributes["material_concentration"], float)
        self.conc = conc0.copy()
        self.material_label = np.asarray(
            frame0.attributes["material_label"], float).copy()
        self.phi = np.zeros(self.n)                 # solid_phase_fraction
        self.temp = np.full(self.n, float(ic["egg_temperature_k"]))
        self.phase_diffusion = bool(ic.get("phase_diffusion_enabled", True))

        self.t_amb = float(ic["ambient_temperature_k"])
        self.t_pan = float(bc["pan_surface_temperature_k"])
        self.z_top = float(bc["pan_contact_height_m"])
        self.r_escape = float(bc["pan_no_escape_radius_m"])

        # per-material particle model parameters
        self.r0_w, self.r0_y = 0.00773, 0.0060
        self.rc_w, self.rc_y = 1.8 * self.r0_w, 1.8 * self.r0_y
        self.r0 = np.where(self.is_white, self.r0_w, self.r0_y)
        self.rc = np.where(self.is_white, self.rc_w, self.rc_y)
        self.kc = np.where(self.is_white, 0.25, 2.0)   # cohesion stiffness
        self.cn = np.full(self.n, 0.0036)              # normal damping
        self.kn = 8.0                                  # contact stiffness
        self.kb, self.cbd = 10.0, 0.01                 # bond spring/damping

        # thermal / kinetic parameters
        self.kth = 1.5          # pairwise conduction rate
        self.pdiff = 0.10       # phase-diffusion rate (same material only)
        self.hpan = 2.0         # pan contact heat rate [1/s]
        self.hair = 0.03        # ambient cooling rate [1/s]
        self.onset = np.where(self.is_white, 337.0, 340.0)
        self.width = np.where(self.is_white, 20.0, 18.0)
        self.kcook = np.where(self.is_white, 0.5, 0.4)

        # pan interaction gaps
        self.heat_gap = 0.8 * self.r0
        self.fric_gap = 0.5 * self.r0
        self.adh_gap = 1.2 * self.r0
        self.adh = np.where(self.is_white, 12.0, 2.0)
        self.kn_wall, self.cn_wall = 80.0, 0.03

        # bonds (formed on coagulation)
        self.bi = np.empty(0, dtype=np.int64)
        self.bj = np.empty(0, dtype=np.int64)
        self.bl = np.empty(0, dtype=float)
        self.bkey = np.empty(0, dtype=np.int64)
        self.bond_cap = 25 * self.n
        self._step_count = 0

        # hashing grid (cell = largest cutoff)
        self.cell = self.rc_w
        self.ox, self.oy, self.oz = -0.215, -0.215, -0.012
        gx, gy, gz = 0.43, 0.43, 0.165
        self.GX = int(np.ceil(gx / self.cell)) + 1
        self.GY = int(np.ceil(gy / self.cell)) + 1
        self.GZ = int(np.ceil(gz / self.cell)) + 1
        offs = np.array([(i, j, k) for i in (-1, 0, 1)
                         for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
        self.dkey27 = (offs[:, 0] * self.GY + offs[:, 1]) * self.GZ + offs[:, 2]

        # diagnostic raster grid
        self.gdx = 0.005
        self.gxmin, self.gymin = -0.165, -0.145
        self.gnx, self.gny = 66, 58
        self.contact_length_scale = float(self.r0_w)

        self._heated = np.zeros(self.n, dtype=bool)
        self._pan_rate = np.zeros(self.n)
        self.Fint = np.zeros((self.n, 3))

    # ------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        t_target = float(time)
        while self.t < t_target - 1.0e-9:
            h = min(self.dt, t_target - self.t)
            self._step(h)
        return self._frame(t_target)

    # ------------------------------------------------------------ neighbors
    def _pairs(self):
        pos = self.pos
        n = self.n
        ci = np.clip(np.floor((pos[:, 0] - self.ox) / self.cell).astype(np.int64), 0, self.GX - 1)
        cj = np.clip(np.floor((pos[:, 1] - self.oy) / self.cell).astype(np.int64), 0, self.GY - 1)
        ck = np.clip(np.floor((pos[:, 2] - self.oz) / self.cell).astype(np.int64), 0, self.GZ - 1)
        key = (ci * self.GY + cj) * self.GZ + ck
        order = np.argsort(key, kind="stable")
        skeys = key[order]
        cand = (key[:, None] + self.dkey27[None, :]).ravel()
        lo = np.searchsorted(skeys, cand, "left")
        hi = np.searchsorted(skeys, cand, "right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty((0, 3)), np.empty(0)
        p = np.repeat(np.arange(n, dtype=np.int64), cnt)
        lo_rep = np.repeat(lo, cnt)
        off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        q = order[lo_rep + off]
        m = p < q
        p, q = p[m], q[m]
        dv = pos[q] - pos[p]
        d2 = np.einsum("ij,ij->i", dv, dv)
        rcp = 0.5 * (self.rc[p] + self.rc[q])
        keep = d2 < rcp * rcp
        p, q, dv = p[keep], q[keep], dv[keep]
        d = np.sqrt(d2[keep]) + 1.0e-15
        return p, q, dv, d

    def _accumulate(self, out, p, q, vals):
        for ax in range(vals.shape[1]):
            out[:, ax] += (np.bincount(p, weights=vals[:, ax], minlength=self.n)
                           - np.bincount(q, weights=vals[:, ax], minlength=self.n))

    def _step(self, h: float):
        n = self.n
        pos, vel = self.pos, self.vel
        mass = self.mass
        phi, temp = self.phi, self.temp

        p, q, dv, d = self._pairs()
        F = np.zeros((n, 3))
        self.Fint[:] = 0.0

        if len(p):
            rcp = 0.5 * (self.rc[p] + self.rc[q])
            r0p = 0.5 * (self.r0[p] + self.r0[q])
            nvec = dv / d[:, None]
            # --- contact / cohesion / damping forces -------------------
            ovl = np.maximum(r0p - d, 0.0)
            span = np.maximum(rcp - r0p, 1.0e-9)
            w_att = np.where((d > r0p) & (d < rcp),
                             4.0 * np.maximum(d - r0p, 0.0) * np.maximum(rcp - d, 0.0) / (span * span),
                             0.0)
            kcp = 0.5 * (self.kc[p] + self.kc[q])
            phim = 0.5 * (phi[p] + phi[q])
            cnp = 0.5 * (self.cn[p] + self.cn[q]) * (1.0 + 15.0 * phim * phim)
            vrel = vel[q] - vel[p]
            vn = np.einsum("ij,ij->i", vrel, nvec)
            f = -self.kn * ovl + kcp * w_att + cnp * vn
            Fp = f[:, None] * nvec
            self._accumulate(F, p, q, Fp)
            self._accumulate(self.Fint, p, q, Fp)

            # --- heat conduction & phase diffusion ----------------------
            wd = 1.0 - d / rcp
            flux = self.kth * wd * (temp[q] - temp[p])
            temp[:] += h * (np.bincount(p, weights=flux, minlength=n)
                            - np.bincount(q, weights=flux, minlength=n))
            if self.phase_diffusion:
                same = (self.body[p] == self.body[q]).astype(float)
                pflux = self.pdiff * wd * same * (phi[q] - phi[p])
                phi[:] += h * (np.bincount(p, weights=pflux, minlength=n)
                               - np.bincount(q, weights=pflux, minlength=n))

            # --- bond formation on coagulation --------------------------
            self._step_count += 1
            if self._step_count % 4 == 0 and len(self.bi) < self.bond_cap:
                same = self.body[p] == self.body[q]
                cand = (same & (phi[p] > 0.55) & (phi[q] > 0.55)
                        & (d < 1.25 * r0p))
                if cand.any():
                    pc, qc, dc = p[cand], q[cand], d[cand]
                    k64 = pc * np.int64(n) + qc
                    if len(self.bkey):
                        fresh = ~np.isin(k64, self.bkey)
                    else:
                        fresh = np.ones(len(k64), dtype=bool)
                    room = self.bond_cap - len(self.bi)
                    take = np.flatnonzero(fresh)[:room]
                    if len(take):
                        self.bi = np.concatenate((self.bi, pc[take]))
                        self.bj = np.concatenate((self.bj, qc[take]))
                        self.bl = np.concatenate((self.bl, dc[take]))
                        self.bkey = np.concatenate((self.bkey, k64[take]))

        # --- bond (solid network) forces --------------------------------
        if len(self.bi):
            bi, bj = self.bi, self.bj
            dvb = pos[bj] - pos[bi]
            db = np.sqrt(np.einsum("ij,ij->i", dvb, dvb)) + 1.0e-15
            nb = dvb / db[:, None]
            vnb = np.einsum("ij,ij->i", vel[bj] - vel[bi], nb)
            fb = self.kb * (db - self.bl) + self.cbd * vnb
            Fb = fb[:, None] * nb
            self._accumulate(F, bi, bj, Fb)
            self._accumulate(self.Fint, bi, bj, Fb)

        # --- pan contact, adhesion, friction, heating --------------------
        F[:, 2] -= mass * _G
        g = pos[:, 2] - self.z_top
        pen = np.maximum(-g, 0.0)
        touching = pen > 0.0
        near = g < self.fric_gap
        F[:, 2] += self.kn_wall * pen
        velz = vel[:, 2]
        F[:, 2] -= np.where(touching, self.cn_wall * np.minimum(velz, 0.0), 0.0)
        # wetting adhesion toward the pan surface
        adh_zone = (g >= 0.0) & (g < self.adh_gap)
        F[:, 2] -= mass * np.where(adh_zone, self.adh, 0.0)
        # friction: liquid slips, cooked sticks
        gam = (0.5 + 40.0 * phi) * np.where(near, 1.0, 0.0)
        F[:, 0] -= mass * gam * vel[:, 0]
        F[:, 1] -= mass * gam * vel[:, 1]

        # heating from the hot pan where the particle is in the contact layer
        heated = g < self.heat_gap
        self._heated = heated
        rate = np.where(heated, self.hpan * (self.t_pan - temp), 0.0)
        self._pan_rate = rate
        temp += h * rate
        temp += h * self.hair * (self.t_amb - temp)
        np.clip(temp, 250.0, 470.0, out=temp)

        # coagulation kinetics (phase front from pan upward)
        c = np.clip((temp - self.onset) / self.width, 0.0, 1.0)
        phi += h * self.kcook * c * (1.0 - phi)
        np.clip(phi, 0.0, 1.0, out=phi)

        # --- integrate ----------------------------------------------------
        acc = F / mass[:, None]
        vel += acc * h
        sp = np.linalg.norm(vel, axis=1)
        np.multiply(vel, np.minimum(1.0, 1.0 / np.maximum(sp, 1.0e-12))[:, None], out=vel)
        pos += vel * h

        # hard projections: pan floor and no-escape rim
        below = pos[:, 2] < self.z_top
        if below.any():
            pos[below, 2] = self.z_top
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)
        r = np.hypot(pos[:, 0], pos[:, 1])
        out = r > self.r_escape
        if out.any():
            s = self.r_escape / np.maximum(r[out], 1.0e-12)
            pos[out, 0] *= s
            pos[out, 1] *= s
            vel[out, 0] *= 0.5
            vel[out, 1] *= 0.5
        self.t += h

    # ---------------------------------------------------------------- frame
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        pos = self.pos
        g = pos[:, 2] - self.z_top

        # rasterized diagnostic fields on a fixed 2-D grid
        dx, nx, ny = self.gdx, self.gnx, self.gny
        ix = np.floor((pos[:, 0] - self.gxmin) / dx).astype(np.int64)
        iy = np.floor((pos[:, 1] - self.gymin) / dx).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flat = (ix[ok] * ny + iy[ok])
        w = self.mass[ok]
        L = nx * ny
        mw = np.bincount(flat, weights=w, minlength=L)
        mt = np.bincount(flat, weights=w * self.temp[ok], minlength=L)
        mx = np.bincount(flat, weights=w * self.vel[ok, 0], minlength=L)
        my = np.bincount(flat, weights=w * self.vel[ok, 1], minlength=L)
        ms = np.bincount(flat, weights=w * self._pan_rate[ok], minlength=L)
        mw2 = mw.reshape(nx, ny)
        filled = mw2 > 1.0e-15
        active = filled.astype(float)
        tscalar = np.where(filled, mt.reshape(nx, ny) / np.maximum(mw2, 1e-30), self.t_amb)
        ssource = np.where(filled, ms.reshape(nx, ny) / np.maximum(mw2, 1e-30), 0.0)
        vfield = np.zeros((nx, ny, 2))
        vfield[:, :, 0] = np.where(filled, mx.reshape(nx, ny) / np.maximum(mw2, 1e-30), 0.0)
        vfield[:, :, 1] = np.where(filled, my.reshape(nx, ny) / np.maximum(mw2, 1e-30), 0.0)

        # signed distance at near-contact samples (negative = penetration)
        sel = g < 0.02
        csd = np.array(g[sel], dtype=float)

        interaction = np.stack((
            self.Fint[self.is_white].sum(axis=0),
            self.Fint[self.is_yolk].sum(axis=0),
        ))

        mass_quantity = self.mass[:, None] * self.conc

        frame = StateFrame(
            time=float(time),
            position=pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.astype(np.int32),
            attributes={
                "material_concentration": self.conc.copy(),
                "material_label": self.material_label.copy(),
                "liquid_phase_fraction": 1.0 - self.phi,
                "solid_phase_fraction": self.phi.copy(),
                "temperature_k": self.temp.copy(),
                "pan_contact_distance": np.array(g),
                "mass_quantity": mass_quantity,
            },
            fields={
                "active_mask": active,
                "transported_scalar": tscalar,
                "scalar_source": ssource,
                "velocity_field": vfield,
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(2),
                "interaction_force": interaction,
            },
            scalars={
                "total_egg_mass_kg": float(self.mass.sum()),
                "pan_temperature_k": float(self.t_pan),
                "grid_spacing": float(self.gdx),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        w = self.is_white
        rw = np.hypot(self.pos[w, 0] - self.pos[w, 0].mean(),
                      self.pos[w, 1] - self.pos[w, 1].mean())
        ry = np.linalg.norm(self.pos[self.is_yolk] - self.pos[self.is_yolk].mean(0), axis=1)
        return {
            "time": float(self.t),
            "white_spread_radius_p98": float(np.percentile(rw, 98.0)),
            "yolk_radius_p98": float(np.percentile(ry, 98.0)),
            "white_solid_fraction_mean": float(self.phi[w].mean()),
            "yolk_solid_fraction_mean": float(self.phi[self.is_yolk].mean()),
            "max_temperature_k": float(self.temp.max()),
            "total_mass_kg": float(self.mass.sum()),
            "n_bonds": int(len(self.bi)),
        }
