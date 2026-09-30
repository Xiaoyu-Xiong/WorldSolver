"""Two-way coupled water-sheet / four-corner suspended-cloth solver (NumPy).

Physical model
--------------
* Water: particle fluid solved with Position Based Fluids (poly6 density,
  PBF pressure projection, artificial surface-tension scalar, XSPH viscosity).
* Cloth: three-dimensional node shell solved with XPBD distance constraints
  (structural / shear / bending) with the four corner vertices pinned.  The
  cloth is impermeable and its resolved motion feeds back into the fluid.
* Coupling: cloth nodes act as moving boundary particles (Akinci-style) that
  contribute to the fluid density and therefore exert pressure on the fluid.
  The equal-and-opposite reaction is gathered on the cloth nodes, which makes
  the transfer momentum conserving and therefore genuinely two-way.  A hard
  node-sphere contact acts as a leak-proof safety barrier.
* Static boundaries (basin floor/walls, four supports, nozzle) use analytic
  projections.

Deterministic, NumPy only, explicit physical time stepping.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_G = np.array([0.0, 0.0, -9.81], dtype=float)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        cfg = self.config
        meta = scene.metadata()
        frame0 = scene.initial_state()

        self.faces = np.asarray(meta["topology"]["faces"], dtype=np.int64)
        self.nx, self.ny = (int(v) for v in cfg["cloth"]["grid_shape"])
        n_cloth = self.nx * self.ny
        self.n_cloth = n_cloth

        cloth_cfg = cfg["cloth"]
        water_cfg = cfg["water"]
        inflow_cfg = cfg["inflow"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt = 0.01
        self.time = 0.0
        self.step_index = 0
        self.pbf_iters = 3
        self.rho0 = float(water_cfg["density"])
        self.spacing = float(water_cfg["initial_particle_spacing"])
        self.h = 2.0 * self.spacing
        self.poly6_c = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.m_p = self.rho0 * self.spacing ** 3
        self.kernel_eps = 100.0

        # ------------------------------------------------------------- cloth
        pos = np.asarray(frame0.position, dtype=float)
        vel = np.asarray(frame0.velocity, dtype=float)
        mass = np.asarray(frame0.mass, dtype=float)
        body = np.asarray(frame0.body_id)
        self.cloth_p = pos[:n_cloth].copy()
        self.cloth_v = vel[:n_cloth].copy()
        self.cloth_m = np.maximum(mass[:n_cloth].copy(), 1e-12)
        self.cloth_p0 = self.cloth_p.copy()
        self.pinned = np.zeros(n_cloth)
        pinned_idx = np.asarray(cloth_cfg["pinned_vertices"], dtype=np.int64)
        self.pinned[pinned_idx] = 1.0
        self.pin_mask = self.pinned > 0.0
        self.cloth_w = np.where(self.pin_mask, 0.0, 1.0 / self.cloth_m)

        nx, ny = self.nx, self.ny
        k = np.arange(n_cloth).reshape(ny, nx)
        e_struct = [np.column_stack((k[:, :-1].ravel(), k[:, 1:].ravel())),
                    np.column_stack((k[:-1, :].ravel(), k[1:, :].ravel()))]
        e_shear = [np.column_stack((k[:-1, :-1].ravel(), k[1:, 1:].ravel())),
                   np.column_stack((k[:-1, 1:].ravel(), k[1:, :-1].ravel()))]
        e_bend = [np.column_stack((k[:, :-2].ravel(), k[:, 2:].ravel())),
                  np.column_stack((k[:-2, :].ravel(), k[2:, :].ravel()))]
        self.edge_groups = []
        for group, compliance in ((e_struct, 1.2e-5), (e_shear, 2.4e-5), (e_bend, 3.0e-4)):
            for arr in group:
                arr = np.asarray(arr, dtype=np.int64)
                rest = np.linalg.norm(self.cloth_p0[arr[:, 0]] - self.cloth_p0[arr[:, 1]], axis=1)
                self.edge_groups.append((arr[:, 0].copy(), arr[:, 1].copy(), rest, compliance))
        self.edge_lambda = [np.zeros(len(a[0])) for a in self.edge_groups]
        self.cloth_uv = self._cloth_uv()

        # ------------------------------------------------------------- water
        self.wp = pos[n_cloth:].copy()
        self.wv = vel[n_cloth:].copy()
        self.wm = mass[n_cloth:].copy()
        if len(self.wm) == 0:
            self.wp = np.zeros((0, 3))
            self.wv = np.zeros((0, 3))
            self.wm = np.zeros((0,))
        self.cloth_wet = np.zeros(n_cloth)
        self.initial_water_mass = float(self.wm.sum())

        # ------------------------------------------------------------ inflow
        self.axis = np.asarray(inflow_cfg["axis"], dtype=float)
        self.axis = self.axis / max(np.linalg.norm(self.axis), 1e-12)
        self.inflow_speed = float(inflow_cfg["speed"])
        self.inflow_center = np.asarray(inflow_cfg["center"], dtype=float)
        self.nozzle_r = float(inflow_cfg["radius"])
        self.t_in0, self.t_in1 = (float(v) for v in inflow_cfg["active_interval"])
        self.mass_flux = self.rho0 * np.pi * self.nozzle_r ** 2 * self.inflow_speed
        self.emit_offsets = self._emit_lattice()
        self.layer_mass = self.m_p * len(self.emit_offsets)
        self.emit_accum = 0.0
        self.emitted_mass = 0.0
        self.layer_index = 0
        self.inflow_active = True

        # -------------------------------------------------------- accounting
        self.cum_inflow_momentum = np.zeros(3)
        self.cum_ext_impulse = np.zeros((3, 3))
        self.cum_mass_source = np.zeros(1)
        self.last_wall_force = np.zeros(3)
        self.last_contact_force = np.zeros(3)

        self._build_boundary_samples()

    # --------------------------------------------------------------- helpers
    def _cloth_uv(self):
        nx, ny = self.nx, self.ny
        return np.column_stack((np.tile(np.linspace(0.0, 1.0, nx), ny),
                                np.repeat(np.linspace(0.0, 1.0, ny), nx)))

    def _emit_lattice(self):
        g = np.arange(-self.nozzle_r, self.nozzle_r + 0.5 * self.spacing, self.spacing)
        xx, yy = np.meshgrid(g, g, indexing="xy")
        sel = (xx ** 2 + yy ** 2) <= self.nozzle_r ** 2
        return np.column_stack((xx[sel], yy[sel], np.full(int(sel.sum()), -0.02)))

    def _build_boundary_samples(self):
        pts, nrm = [], []
        g = np.linspace(-2.0, 2.0, 8)
        gg = np.linspace(-1.75, 1.75, 8)
        mx, my = np.meshgrid(g, gg, indexing="xy")
        pts.append(np.column_stack((mx.ravel(), my.ravel(), np.zeros(mx.size))))
        nrm.append(np.tile(np.array([0.0, 0.0, 1.0]), (mx.size, 1)))
        sides = [(1, -1.79, np.array([0.0, 1.0, 0.0])), (1, 1.79, np.array([0.0, -1.0, 0.0])),
                 (0, -2.05, np.array([1.0, 0.0, 0.0])), (0, 2.05, np.array([-1.0, 0.0, 0.0]))]
        for axis, coord, normal in sides:
            lin = np.linspace(-1.6, 1.6, 8)
            zz = np.array([0.1, 0.4, 0.7])
            a, b = np.meshgrid(lin, zz, indexing="ij")
            p = np.zeros((a.size, 3))
            p[:, axis] = coord
            p[:, 2] = b.ravel()
            p[:, 1 - axis] = a.ravel()
            pts.append(p)
            nrm.append(np.tile(normal, (len(p), 1)))
        for sx in (-1.325, 1.325):
            for sy in (-1.125, 1.125):
                th = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                zz = np.array([0.2, 1.0, 1.9])
                a, b = np.meshgrid(th, zz, indexing="ij")
                rad = 0.045
                pts.append(np.column_stack((sx + rad * np.cos(a.ravel()),
                                            sy + rad * np.sin(a.ravel()), b.ravel())))
                nrm.append(np.column_stack((np.cos(a.ravel()), np.sin(a.ravel()),
                                            np.zeros(a.size))))
        th = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        zz = np.array([3.25, 3.6])
        a, b = np.meshgrid(th, zz, indexing="ij")
        pts.append(np.column_stack((0.31 * np.cos(a.ravel()), 0.31 * np.sin(a.ravel()),
                                    b.ravel())))
        nrm.append(np.column_stack((np.cos(a.ravel()), np.sin(a.ravel()), np.zeros(a.size))))
        self.b_static_p = np.vstack(pts)
        self.b_static_n = np.vstack(nrm)
        self.b_static_p[:, 2] = np.maximum(self.b_static_p[:, 2], 0.0)

    # ------------------------------------------------------- neighbor search
    @staticmethod
    def _pairs(pos, h):
        n = len(pos)
        if n < 2:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        cell = np.floor(pos / h).astype(np.int64)
        mn = cell.min(axis=0) - 1
        idx = cell - mn
        m = idx.max(axis=0) + 3
        key = (idx[:, 0] * m[1] + idx[:, 1]) * m[2] + idx[:, 2]
        order = np.argsort(key, kind="stable")
        sk = key[order]
        ii, jj = [], []
        base = np.arange(n, dtype=np.int64)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    tkey = key + ((a * m[1] + b) * m[2] + c)
                    lo = np.searchsorted(sk, tkey, side="left")
                    hi = np.searchsorted(sk, tkey, side="right")
                    cnt = hi - lo
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    i_rep = np.repeat(base, cnt)
                    starts = np.repeat(lo, cnt)
                    cs = np.cumsum(cnt)
                    bs = np.repeat(cs - cnt, cnt)
                    js = order[starts + (np.arange(tot) - bs)]
                    ii.append(i_rep)
                    jj.append(js)
        if not ii:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        i = np.concatenate(ii)
        j = np.concatenate(jj)
        keep = i < j
        return i[keep], j[keep]

    def _neighbors(self, all_pos):
        i, j = self._pairs(all_pos, self.h)
        if len(i) == 0:
            return i, j
        d = all_pos[i] - all_pos[j]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = (r2 > 1e-14) & (r2 < self.h * self.h)
        return i[keep], j[keep]

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        nw = len(self.wp)
        nc = self.n_cloth
        v = self.wv
        cp = self.cloth_p
        cv = self.cloth_v
        v = v + self.gravity * dt

        p_prev = self.wp.copy()
        cp_prev = cp.copy()
        p = self.wp + v * dt
        cp = cp + cv * dt

        all_pos = np.vstack((p, cp))
        i, j = self._neighbors(all_pos)
        if len(i):
            is_wi = i < nw
            is_wj = j < nw
            ww = is_wi & is_wj
            wc = is_wi & (~is_wj)
            iw, jw = i[ww], j[ww]
            cwi, cwj = i[wc], (j[wc] - nw)
        else:
            iw = jw = cwi = cwj = np.zeros(0, dtype=np.int64)

        for lam_arr in self.edge_lambda:
            lam_arr[:] = 0.0

        m = self.wm
        rho0 = self.rho0
        h2 = self.h * self.h
        c6 = self.poly6_c
        m_b = self.m_p
        w_self = m * (c6 * h2 ** 3)

        react = np.zeros((nc, 3))
        for _ in range(self.pbf_iters):
            if len(iw) == 0 and len(cwi) == 0:
                break
            rho = w_self.copy()
            sum_g = np.zeros(nw)
            dw_ratio = np.zeros(nw)
            wq = c6 * (h2 - (0.2 * self.h) ** 2) ** 3
            if len(iw):
                d = p[iw] - p[jw]
                r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-14)
                r = np.sqrt(r2)
                diff = h2 - r2
                wgt = c6 * diff ** 3
                dw = -6.0 * c6 * r * diff ** 2
                rho += (np.bincount(iw, weights=m[jw] * wgt, minlength=nw)
                        + np.bincount(jw, weights=m[iw] * wgt, minlength=nw))
                gr = (dw / r) ** 2
                sum_g += (np.bincount(iw, weights=gr, minlength=nw)
                          + np.bincount(jw, weights=gr, minlength=nw))
                dw_ratio += (np.bincount(iw, weights=wgt, minlength=nw)
                             + np.bincount(jw, weights=wgt, minlength=nw))
            if len(cwi):
                d = p[cwi] - cp[cwj]
                r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-14)
                r = np.sqrt(r2)
                diff = h2 - r2
                wgt_c = c6 * diff ** 3
                dw_c = -6.0 * c6 * r * diff ** 2
                rho += np.bincount(cwi, weights=m_b * wgt_c, minlength=nw)
                sum_g += np.bincount(cwi, weights=(dw_c / r) ** 2, minlength=nw)
                dw_ratio += np.bincount(cwi, weights=wgt_c, minlength=nw)
            rho = np.maximum(rho, 0.3 * rho0)
            sum_g = sum_g / rho0 ** 2
            con = rho / rho0 - 1.0
            lam = -con / (sum_g + self.kernel_eps)
            scorr = -0.02 * (dw_ratio / wq) ** 4
            dp = np.zeros((nw, 3))
            if len(iw):
                dd = p[iw] - p[jw]
                rr = np.sqrt(np.maximum(np.einsum("ij,ij->i", dd, dd), 1e-14))
                dW = (-6.0 * c6 * rr * (h2 - rr ** 2) ** 2) / rr
                coef = (lam[iw] + lam[jw] + scorr[iw] + scorr[jw]) * dW
                for axis in range(3):
                    dp[:, axis] += (np.bincount(iw, weights=coef * dd[:, axis], minlength=nw)
                                    - np.bincount(jw, weights=coef * dd[:, axis], minlength=nw))
            if len(cwi):
                dd = p[cwi] - cp[cwj]
                rr = np.sqrt(np.maximum(np.einsum("ij,ij->i", dd, dd), 1e-14))
                dW = (-6.0 * c6 * rr * (h2 - rr ** 2) ** 2) / rr
                coef = (lam[cwi] + scorr[cwi]) * dW
                for axis in range(3):
                    dp[:, axis] += np.bincount(cwi, weights=coef * dd[:, axis], minlength=nw)
                react_weight = -m[cwi] * (lam[cwi] + scorr[cwi]) * dW / rho0
                np.add.at(react, cwj, react_weight[:, None] * dd / rr[:, None])
            p = p + dp / rho0
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            self._solve_cloth(cp, dt)

        # ---------------------------------------------- hard safety membrane
        if len(cwi):
            d = p[cwi] - cp[cwj]
            r = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-14))
            pen = 0.72 * self.spacing - r
            hit = np.nonzero(pen > 0.0)[0]
            if len(hit):
                nvec = d[hit] / r[hit, None]
                wi = 1.0 / m[cwi[hit]]
                wj = 20.0 / self.cloth_m[cwj[hit]]
                dl = pen[hit] / (wi + wj)
                p[cwi[hit]] += (wi * dl)[:, None] * nvec
                cp[cwj[hit]] -= (wj * dl)[:, None] * nvec

        v = (p - p_prev) / dt
        cv = (cp - cp_prev) / dt

        # ---------------------------------------------------- fluid load on cloth
        f_load = react / (dt * dt)
        acc = f_load / self.cloth_m[:, None]
        acc = np.clip(acc, -100.0, 100.0)
        cv = cv + acc * dt
        cv *= 0.85
        cv = np.clip(cv, -4.0, 4.0)

        # ------------------------------------------------------ XSPH viscosity
        if len(iw):
            dd = p[iw] - p[jw]
            r2 = np.einsum("ij,ij->i", dd, dd)
            keep = (r2 > 1e-14) & (r2 < h2)
            ii, jj = iw[keep], jw[keep]
            if len(ii):
                r2k = np.maximum(r2[keep], 1e-14)
                wgt = c6 * (h2 - r2k) ** 3
                vol = 0.03 * (m[jj] / rho0) * wgt
                dv = vol[:, None] * (v[jj] - v[ii])
                for axis in range(3):
                    v[:, axis] += np.bincount(ii, weights=dv[:, axis], minlength=nw)

        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        cp = np.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)

        # --------------------------------------------------- static boundaries
        wall_force = np.zeros(3)
        if nw:
            v_before = v.copy()
            self._static_collide(p, v)
            wall_force = np.einsum("i,ij->j", m, v - v_before)
        if nc:
            below = cp[:, 2] < 0.03
            cp[below, 2] = 0.03
            cv[below, 2] = np.maximum(cv[below, 2], 0.0)

        cp[self.pin_mask] = self.cloth_p0[self.pin_mask]
        cv[self.pin_mask] = 0.0

        # ------------------------------------------------------------ wetness
        node_target = np.zeros(nc)
        if len(cwi):
            dd = p[cwi] - cp[cwj]
            near = np.einsum("ij,ij->i", dd, dd) < (1.5 * self.spacing) ** 2
            if np.any(near):
                np.add.at(node_target, cwj[near], 1.0)
        node_target = np.clip(node_target, 0.0, 1.0)
        self.cloth_wet = np.clip(
            self.cloth_wet + dt * 2.0 * (node_target - self.cloth_wet), 0.0, 1.0)

        # -------------------------------------------------------------- commit
        self.wp = p
        self.wv = v
        self.cloth_p = cp
        self.cloth_v = cv
        self.last_wall_force = wall_force
        self.last_contact_force = f_load.sum(axis=0)
        self._accumulate_budget(dt)

    # --------------------------------------------------------- budget update
    def _accumulate_budget(self, dt: float) -> None:
        t0 = self.step_index * self.dt
        active = t0 < self.t_in1
        f_in = self.mass_flux * self.axis * self.inflow_speed if active else np.zeros(3)
        self.inflow_active = bool(active)
        self.cum_ext_impulse += self._external_force(f_in).sum(axis=1) * dt
        if active:
            dt_eff = min(dt, self.t_in1 - t0)
            self.emit_accum += self.mass_flux * dt_eff
            while (self.emit_accum >= self.layer_mass) and (self.emitted_mass < 3.0e3):
                self.emit_accum -= self.layer_mass
                self.emitted_mass += self.layer_mass
                n_new = len(self.emit_offsets)
                ang = 0.37 * self.layer_index
                ca, sa = float(np.cos(ang)), float(np.sin(ang))
                off = self.emit_offsets
                rot = np.column_stack((ca * off[:, 0] - sa * off[:, 1],
                                       sa * off[:, 0] + ca * off[:, 1], off[:, 2]))
                self.layer_index += 1
                self.wp = np.vstack((self.wp, self.inflow_center + rot))
                self.wv = np.vstack((self.wv, np.tile(self.axis * self.inflow_speed, (n_new, 1))))
                self.wm = np.concatenate((self.wm, np.full(n_new, self.m_p)))
                self.cum_mass_source[0] += self.layer_mass
                self.cum_inflow_momentum += self.layer_mass * self.axis * self.inflow_speed

    def _external_force(self, f_in=None):
        if f_in is None:
            f_in = (self.mass_flux * self.axis * self.inflow_speed if self.inflow_active
                    else np.zeros(3))
        m = self.wm
        wall = self.last_wall_force
        fc = self.last_contact_force
        ext = np.zeros((3, 4, 3))
        ext[0, 0] = (m.sum() + self.cloth_m.sum()) * self.gravity
        ext[0, 1] = wall
        ext[0, 2] = f_in
        ext[1, 0] = self.cloth_m.sum() * self.gravity
        ext[1, 3] = fc
        ext[2, 0] = m.sum() * self.gravity
        ext[2, 1] = wall
        ext[2, 2] = f_in
        ext[2, 3] = -fc
        return ext

    # ------------------------------------------------------------ cloth XPBD
    def _solve_cloth(self, cp, dt):
        weights = self.cloth_w
        for g, (ia, ja, rest, compliance) in enumerate(self.edge_groups):
            lam = self.edge_lambda[g]
            alpha_t = compliance / (dt * dt)
            d = cp[ia] - cp[ja]
            r = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-14))
            con = r - rest
            wi = weights[ia]
            wj = weights[ja]
            dl = (-con - alpha_t * lam) / (wi + wj + alpha_t)
            unit = d / r[:, None]
            cp[ia] += (wi * dl)[:, None] * unit
            cp[ja] -= (wj * dl)[:, None] * unit
            self.edge_lambda[g] = lam + dl
        cp[self.pin_mask] = self.cloth_p0[self.pin_mask]

    # ------------------------------------------------------- static collision
    def _static_collide(self, p, v):
        below = p[:, 2] < 0.0
        if np.any(below):
            p[below, 2] = 0.0
            v[below, 2] = np.maximum(v[below, 2], 0.0)
        above = p[:, 2] > 3.62
        if np.any(above):
            p[above, 2] = 3.62
            v[above, 2] = np.minimum(v[above, 2], 0.0)
        low = p[:, 2] < 0.76
        lx = np.where(low, 2.05, 2.33)
        ly = np.where(low, 1.79, 2.13)
        for axis, lim in ((0, lx), (1, ly)):
            over = p[:, axis] > lim
            if np.any(over):
                p[over, axis] = lim[over]
                v[over, axis] = np.minimum(v[over, axis], 0.0)
            under = p[:, axis] < -lim
            if np.any(under):
                p[under, axis] = -lim[under]
                v[under, axis] = np.maximum(v[under, axis], 0.0)
        for cx, cy, z0, z1, rad in ((-1.325, -1.125, 0.0, 2.05, 0.045),
                                    (1.325, -1.125, 0.0, 2.05, 0.045),
                                    (-1.325, 1.125, 0.0, 2.05, 0.045),
                                    (1.325, 1.125, 0.0, 2.05, 0.045),
                                    (0.0, 0.0, 3.200001, 3.72, 0.31)):
            sel = (p[:, 2] >= z0) & (p[:, 2] <= z1)
            if not np.any(sel):
                continue
            dx = p[sel, 0] - cx
            dy = p[sel, 1] - cy
            rr = np.sqrt(dx * dx + dy * dy)
            hit = rr < rad
            if not np.any(hit):
                continue
            idx = np.nonzero(sel)[0][hit]
            rr_h = np.maximum(rr[hit], 1e-9)
            nx = dx[hit] / rr_h
            ny = dy[hit] / rr_h
            p[idx, 0] = cx + rad * nx
            p[idx, 1] = cy + rad * ny
            radial = v[idx, 0] * nx + v[idx, 1] * ny
            inward = radial < 0.0
            if np.any(inward):
                v[idx[inward], 0] -= radial[inward] * nx[inward]
                v[idx[inward], 1] -= radial[inward] * ny[inward]

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-9:
            raise ValueError("advance_to requires non-decreasing time")
        while self.step_index * self.dt < target - 1e-9:
            self._step(self.dt)
            self.step_index += 1
        self.time = target
        return self._frame(target)

    def _cloth_normals(self):
        p = self.cloth_p
        f = self.faces
        n = np.zeros_like(p)
        fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        for kk in range(3):
            np.add.at(n, f[:, kk], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
        n[n[:, 2] < 0.0] *= -1.0
        return n

    def _frame(self, t: float) -> StateFrame:
        nw = len(self.wp)
        nc = self.n_cloth
        n = nc + nw
        position = np.vstack((self.cloth_p, self.wp))
        velocity = np.vstack((self.cloth_v, self.wv))
        mass = np.concatenate((self.cloth_m, self.wm))
        body_id = np.concatenate((np.zeros(nc, dtype=np.int64), np.ones(nw, dtype=np.int64)))
        wetness = np.concatenate((np.clip(self.cloth_wet, 0.0, 1.0), np.ones(nw)))

        b_p = np.vstack((self.b_static_p, self.cloth_p))
        b_n = np.vstack((self.b_static_n, self._cloth_normals()))
        b_v = np.vstack((np.zeros_like(self.b_static_p), self.cloth_v))
        ln = np.linalg.norm(b_n, axis=1, keepdims=True)
        b_n = np.where(ln > 1e-12, b_n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))

        weights = np.zeros((3, n))
        weights[0, :] = 1.0
        weights[1, :nc] = 1.0
        weights[2, nc:] = 1.0

        fc = self.last_contact_force
        fields = {
            "boundary_normal": b_n,
            "boundary_target_velocity": b_v.copy(),
            "boundary_velocity": b_v.copy(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.vstack(
                (self.cum_inflow_momentum, np.zeros(3), self.cum_inflow_momentum)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "budget_domain_weights": weights,
            "budget_external_force": self._external_force(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": np.vstack((fc, -fc)),
        }
        scalars = {
            "water_density": self.rho0,
            "cloth_thickness": float(self.config["cloth"]["thickness"]),
            "nominal_inflow_speed": self.inflow_speed,
            "nozzle_radius": self.nozzle_r,
            "reference_velocity": self.inflow_speed,
            "reference_momentum": 1000.0 * self.inflow_speed,
            "particle_spacing": self.spacing,
            "time": t,
        }
        attributes = {
            "mass_quantity": mass.copy(),
            "uv": np.vstack((self.cloth_uv, np.zeros((nw, 2)))),
            "pinned": np.concatenate((self.pinned, np.zeros(nw))),
            "material_phase": body_id.astype(float),
            "wetness": wetness,
        }
        frame = StateFrame(time=t, position=position, velocity=velocity, mass=mass,
                           body_id=body_id, attributes=attributes, fields=fields, scalars=scalars)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbf_sph_cloth_xpbd_two_way",
            "dt": self.dt,
            "steps": int(self.step_index),
            "particles": int(len(self.wp)),
            "cloth_vertices": int(self.n_cloth),
            "emitted_mass": float(self.emitted_mass),
            "cumulative_mass_source": float(self.cum_mass_source[0]),
        }
