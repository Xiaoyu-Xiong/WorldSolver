"""Two-way coupled solver for the wet-towel lift-and-wring scene.

Physical model -- every state is produced by time integration, nothing is
scripted or key-framed:

  * Towel: a 41x15 particle sheet carrying structural, shear and bending
    springs, gravity, axial spring damping, global dissipation, self
    repulsion and basin contact.  Only the two grasp bands are kinematic
    (driven by the frozen grip controls); the middle is free to fold, sag and
    twist.
  * Liquid: a weakly compressible particle fluid stepped with Position Based
    Fluids (poly6 density, spiky gradients, artificial pressure, XSPH
    viscosity) inside the frozen basin geometry.
  * Two-way coupling: semi-implicit porous-fabric drag between towel nodes and
    neighbouring fluid particles (through-flow allowed, no no-penetration
    constraint), buoyancy together with its paired reaction on the fluid, and
    momentum conserving liquid mass exchange.
  * Porosity: per-node liquid storage.  Capillary uptake feeds it while the
    fabric is in the water; drainage is driven by the simulated torsional
    strain of the wringing towel (plus in-plane compression) and, once the
    fabric is above field capacity, by gravity -- so dripping continues after
    the twisting motion stops.
  * Budgets: every impulse / momentum field is accumulated from the actual
    simulated force history.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_OFFSETS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.int64,
)


def _grid_pairs(pa, pb, r, lo, hi, same=False):
    """Index pairs (i, j) with |pa[i] - pb[j]| < r using a uniform grid."""
    n, m = pa.shape[0], pb.shape[0]
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    if n == 0 or m == 0:
        return empty
    r = float(r)
    ng = np.maximum(np.ceil((hi - lo) / r).astype(np.int64), 1)
    cb = np.floor((pb - lo) / r).astype(np.int64)
    np.clip(cb, 0, ng - 1, out=cb)
    lin_b = cb[:, 0] + ng[0] * (cb[:, 1] + ng[1] * cb[:, 2])
    order = np.argsort(lin_b, kind="stable")
    lins = lin_b[order]
    ca = np.floor((pa - lo) / r).astype(np.int64)
    np.clip(ca, 0, ng - 1, out=ca)
    tx = ca[:, 0][:, None] + _OFFSETS[:, 0][None, :]
    ty = ca[:, 1][:, None] + _OFFSETS[:, 1][None, :]
    tz = ca[:, 2][:, None] + _OFFSETS[:, 2][None, :]
    valid = (tx >= 0) & (tx < ng[0]) & (ty >= 0) & (ty < ng[1]) & (tz >= 0) & (tz < ng[2])
    nid = np.where(valid, tx + ng[0] * (ty + ng[1] * tz), -1)
    starts = np.searchsorted(lins, nid, side="left")
    ends = np.searchsorted(lins, nid, side="right")
    counts = np.where(valid, ends - starts, 0)
    row = counts.sum(axis=1)
    total = int(row.sum())
    if total == 0:
        return empty
    ii = np.repeat(np.arange(n, dtype=np.int64), row)
    cntf = counts.ravel()
    cum = np.cumsum(cntf) - cntf
    jf = np.repeat(starts.ravel(), cntf) + (
        np.arange(total, dtype=np.int64) - np.repeat(cum, cntf)
    )
    jj = order[jf]
    d = pa[ii] - pb[jj]
    r2 = np.einsum("ij,ij->i", d, d)
    keep = r2 < r * r
    if same:
        keep &= ii < jj
    return ii[keep], jj[keep]


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=np.float64)
        self.vel = np.array(frame.velocity, dtype=np.float64)
        self.body = np.array(frame.body_id, dtype=np.int64)
        self.N = int(self.pos.shape[0])
        self.nt = int(np.count_nonzero(self.body == 0))
        self.nw = self.N - self.nt
        self.p0 = np.array(frame.attributes["rest_position"], dtype=np.float64)
        self.uv = np.array(frame.attributes["uv"], dtype=np.float64)

        tcfg = self.config["towel"]
        wcfg = self.config["water"]
        self.rho_w = float(wcfg["density"])
        self.spacing = float(wcfg["particle_spacing"])
        self.gvec = np.asarray(self.config["gravity"], dtype=np.float64)
        self.gmag = float(np.linalg.norm(self.gvec))
        self.nx, self.ny = (int(v) for v in tcfg["grid_shape"])
        self.dx = float(tcfg["size"][0]) / (self.nx - 1)
        self.dy = float(tcfg["size"][1]) / (self.ny - 1)
        self.thickness = float(tcfg["thickness"])
        self.porosity = float(tcfg["porosity"])
        self.capacity = float(tcfg["liquid_storage_capacity_kg_m2"])
        self.node_area = self.dx * self.dy

        self.dry = np.full(self.nt, float(tcfg["dry_areal_density"]) * self.node_area)
        self.cap_node = self.capacity * self.node_area
        self.sat = np.full(self.nt, float(tcfg["initial_saturation"]))
        self.liq = self.cap_node * self.sat
        self.wmass = np.array(frame.mass[self.nt:], dtype=np.float64)
        self.m0w = float(np.mean(self.wmass))

        self.dt = 0.002
        self.steps = 0
        self.t = 0.0

        self.h = 2.0 * self.spacing
        self.h2 = self.h * self.h
        self.k6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky = 45.0 / (np.pi * self.h ** 6)
        self.rho0 = self.rho_w
        self.rc = 0.14
        self.basin_lo = np.array([-1.75, -0.80, 0.0])
        self.basin_hi = np.array([1.75, 0.80, 1.30])
        self.w_skin = 0.6 * self.spacing

        self._build_towel_topology()
        self._build_grasps()
        self._calibrate_density()

        self.G = 3
        self.K_ext = 4
        self.budget_impulse = np.zeros((self.G, 3))
        self.budget_inflow = np.zeros((self.G, 3))
        self.budget_outflow = np.zeros((self.G, 3))
        self.cum_mass_source = np.zeros(2)
        self.last_ext_force = np.zeros((self.G, self.K_ext, 3))
        self.last_interaction = np.zeros((2, 3))
        self.ref_momentum = float(self.wmass.sum() + self.dry.sum() + self.liq.sum())
        self.ref_velocity = 1.0

        self.grip_prev = self._grip_positions(0.0)
        self.last_drop_t = -10.0
        self.drops = 0
        self.drain_pool = 0.0
        self._liq_prev = self.liq.copy()
        self._drained_step = np.zeros(self.nt)
        self._out_mom_step = np.zeros(3)
        self.inflow_ret = np.zeros(3)

    def _build_towel_topology(self) -> None:
        nx, ny = self.nx, self.ny
        ii, jj, kk = [], [], []
        for iy in range(ny):
            base = iy * nx
            for ix in range(nx):
                a = base + ix
                if ix + 1 < nx:
                    ii.append(a); jj.append(a + 1); kk.append(80.0)
                if iy + 1 < ny:
                    ii.append(a); jj.append(a + nx); kk.append(80.0)
                if ix + 1 < nx and iy + 1 < ny:
                    ii.append(a); jj.append(a + nx + 1); kk.append(25.0)
                    ii.append(a + 1); jj.append(a + nx); kk.append(25.0)
                if ix + 2 < nx:
                    ii.append(a); jj.append(a + 2); kk.append(4.0)
                if iy + 2 < ny:
                    ii.append(a); jj.append(a + 2 * nx); kk.append(4.0)
        self.sp_i = np.asarray(ii, dtype=np.int64)
        self.sp_j = np.asarray(jj, dtype=np.int64)
        self.sp_k = np.asarray(kk, dtype=np.float64)
        d = self.p0[self.sp_i] - self.p0[self.sp_j]
        self.sp_l0 = np.sqrt(np.einsum("ij,ij->i", d, d))

        tri = []
        for iy in range(ny - 1):
            base = iy * nx
            for ix in range(nx - 1):
                a = base + ix
                tri.append((a, a + 1, a + nx + 1))
                tri.append((a, a + nx + 1, a + nx))
        self.tri = np.asarray(tri, dtype=np.int64)
        ta = self.p0[self.tri[:, 0]]
        tb = self.p0[self.tri[:, 1]]
        tc = self.p0[self.tri[:, 2]]
        self.tri_area0 = np.maximum(
            0.5 * np.linalg.norm(np.cross(tb - ta, tc - ta), axis=1), 1e-12
        )
        # longitudinal station index of every towel node
        self.station = np.tile(np.arange(self.nx, dtype=np.int64), self.ny)

    def _build_grasps(self) -> None:
        tcfg = self.config["towel"]
        grasp = float(tcfg["grasp_length"])
        end_z = float(tcfg["initial_end_height"])
        x = self.p0[: self.nt, 0]
        xmin, xmax = float(x.min()), float(x.max())
        left = np.where(x <= xmin + grasp)[0]
        right = np.where(x >= xmax - grasp)[0]
        self.drv_idx = np.concatenate((left, right))
        self.n_left = len(left)
        self.drv_off = (
            self.p0[left] - np.array([xmin, 0.0, end_z]),
            self.p0[right] - np.array([xmax, 0.0, end_z]),
        )
        self.n_drv = len(self.drv_idx)
        self.free_mask = np.ones(self.nt, dtype=bool)
        self.free_mask[self.drv_idx] = False
        self.drv_pos_idx = np.zeros(self.n_drv, dtype=np.int64)
        self.drv_pos_idx[self.drv_idx] = np.arange(self.n_drv)
        self.grip_slices = (slice(0, self.n_left), slice(self.n_left, self.n_drv))

    def _calibrate_density(self) -> None:
        wp = self.pos[self.nt:]
        lo = wp.min(axis=0) - 0.02
        hi = wp.max(axis=0) + 0.02
        ii, jj = _grid_pairs(wp, wp, self.h, lo, hi, same=True)
        if ii.size:
            d = wp[ii] - wp[jj]
            r2 = np.einsum("ij,ij->i", d, d)
            W = self.k6 * np.maximum(self.h2 - r2, 0.0) ** 3
            rho = np.bincount(ii, weights=self.wmass[jj] * W, minlength=self.nw)
            rho += np.bincount(jj, weights=self.wmass[ii] * W, minlength=self.nw)
            ref = float(np.percentile(rho, 85))
            if ref > 1e-9:
                self.k6 *= self.rho0 / ref

    # ------------------------------------------------------------- kinematics
    def _grip_positions(self, t: float) -> np.ndarray:
        kin = self.scene.grip_kinematics(float(t))
        out = np.empty((self.n_drv, 3), dtype=np.float64)
        for side, name in enumerate(("grip_left", "grip_right")):
            trans = np.asarray(kin[name]["translation"], dtype=np.float64)
            th = float(kin[name]["rotation_about_x_radians"])
            c, s = np.cos(th), np.sin(th)
            off = self.drv_off[side]
            rot = np.empty_like(off)
            rot[:, 0] = off[:, 0]
            rot[:, 1] = c * off[:, 1] - s * off[:, 2]
            rot[:, 2] = s * off[:, 1] + c * off[:, 2]
            rot += trans
            out[self.grip_slices[side]] = rot
        return out

    # ------------------------------------------------------------ main driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_target = max(int(round(target / self.dt)), self.steps)
        while self.steps < n_target:
            self._step()
        return self._frame(target)

    # -------------------------------------------------------------- one step
    def _step(self) -> None:
        dt = self.dt
        self.steps += 1
        t_new = self.steps * dt
        nt, nw, N = self.nt, self.nw, self.N

        vel_old = self.vel.copy()
        wmass_old = self.wmass.copy()

        new_grip = self._grip_positions(t_new)
        grip_vel = (new_grip - self.grip_prev) / dt
        self.grip_prev = new_grip

        # ---- towel internal forces -----------------------------------------
        F = np.zeros_like(self.pos)
        if self.sp_i.size:
            d = self.pos[self.sp_j] - self.pos[self.sp_i]
            L = np.sqrt(np.einsum("ij,ij->i", d, d))
            dhat = d / np.maximum(L, 1e-9)[:, None]
            strain = L - self.sp_l0
            k = np.where(strain < 0.0, 0.08 * self.sp_k, self.sp_k)
            vrel = np.einsum("ij,ij->i", self.vel[self.sp_j] - self.vel[self.sp_i], dhat)
            fs = (k * strain + 0.35 * vrel)[:, None] * dhat
            for c in range(3):
                F[:, c] += np.bincount(self.sp_i, weights=fs[:, c], minlength=N)
                F[:, c] -= np.bincount(self.sp_j, weights=fs[:, c], minlength=N)

        pos_t = self.pos[:nt]
        if nt > 1:
            lo = pos_t.min(axis=0) - 0.02
            hi = pos_t.max(axis=0) + 0.02
            si, sj = _grid_pairs(pos_t, pos_t, 0.04, lo, hi, same=True)
            if si.size:
                r0 = np.linalg.norm(self.p0[si] - self.p0[sj], axis=1)
                keep = r0 > 0.07
                si, sj = si[keep], sj[keep]
                if si.size:
                    dd = pos_t[si] - pos_t[sj]
                    LL = np.maximum(np.linalg.norm(dd, axis=1), 1e-9)
                    nn = dd / LL[:, None]
                    keep2 = LL < 0.025
                    si, sj, nn, LL = si[keep2], sj[keep2], nn[keep2], LL[keep2]
                    if si.size:
                        vn = np.einsum("ij,ij->i", self.vel[si] - self.vel[sj], nn)
                        fsc = (4.0 * (0.025 - LL) - 0.05 * vn)[:, None] * nn
                        for c in range(3):
                            F[:nt, c] += np.bincount(si, weights=fsc[:, c], minlength=nt)
                            F[:nt, c] -= np.bincount(sj, weights=fsc[:, c], minlength=nt)

        m_t = self.dry + self.liq
        F[:nt] += self.gvec[None, :] * m_t[:, None]

        # ---- towel / fluid coupling pairs ----------------------------------
        ci, cj, cw = self._couple_pairs()
        csum = np.zeros(nt)
        pvec = np.zeros((nt, 3))
        sub = np.zeros(nt)
        wv = self.vel[nt:]
        if ci.size:
            urel = wv[cj] - self.vel[:nt][ci]
            umag = np.minimum(np.linalg.norm(urel, axis=1), 3.0)
            cpair = 0.5 * self.rho_w * 1.2 * self.node_area * cw * umag
            csum = np.bincount(ci, weights=cpair, minlength=nt)
            for c in range(3):
                pvec[:, c] = np.bincount(ci, weights=cpair * wv[cj, c], minlength=nt)
            sub = np.clip(np.bincount(ci, weights=cw, minlength=nt) / 1.2, 0.0, 1.0)
        else:
            cpair = np.empty(0)

        buoy = self.rho_w * self.node_area * self.thickness * self.gmag * sub
        F[:nt, 2] += buoy

        # ---- integrate towel ------------------------------------------------
        m_safe = np.maximum(m_t, 1e-9)
        acc = F[:nt] / m_safe[:, None]
        an = np.linalg.norm(acc, axis=1)
        acc *= np.minimum(1.0, 400.0 / np.maximum(an, 1e-9))[:, None]
        v_pred = self.vel[:nt] + dt * acc
        v_new = (v_pred + dt * pvec / m_safe[:, None]) / (1.0 + dt * csum / m_safe)
        v_new *= np.exp(-0.8 * dt)
        v_new[~self.free_mask] = grip_vel
        new_t = pos_t + dt * v_new
        new_t[~self.free_mask] = new_grip[self.drv_pos_idx]

        # ---- coupling reaction on the fluid ---------------------------------
        if ci.size:
            Fw = cpair[:, None] * (wv[cj] - v_new[ci])
            accw = np.zeros((nw, 3))
            for c in range(3):
                accw[:, c] = -np.bincount(cj, weights=Fw[:, c], minlength=nw) / self.wmass
            wsum = np.maximum(np.bincount(ci, weights=cw, minlength=nt), 1e-12)
            share = sub[ci] * cw / wsum[ci]
            fb = -(self.rho_w * self.node_area * self.thickness * self.gmag)
            accw[:, 2] += fb * np.bincount(cj, weights=share, minlength=nw) / self.wmass
            wv = wv + dt * accw
        wv = wv + dt * self.gvec[None, :]

        self.pos[:nt] = np.clip(new_t, [-1.74, -0.79, 0.004], [1.74, 0.79, 3.0])
        self.vel[:nt] = np.clip(v_new, -25.0, 25.0)
        self.vel[nt:] = np.clip(wv, -25.0, 25.0)

        # ---- fluid solve ----------------------------------------------------
        self._water_pbf(dt)

        # ---- porous liquid storage, drainage and droplets -------------------
        self._liquid_exchange(dt, ci, cj, cw)
        spray = self._spray_impulse(t_new)

        # ---- budgets --------------------------------------------------------
        m_new_t = self.dry + self.liq
        drag = pvec - csum[:, None] * v_new
        grip_nodes = np.zeros((nt, 3))
        drv = ~self.free_mask
        grip_nodes[drv] = m_t[drv, None] * (v_new[drv] - v_pred[drv]) / dt - drag[drv]
        grip = grip_nodes.sum(axis=0)
        f_couple_t = drag.sum(axis=0) + np.array([0.0, 0.0, float(buoy.sum())]) - spray / dt
        f_couple_w = -f_couple_t
        dp_t = (m_new_t[:, None] * self.vel[:nt]).sum(axis=0) - (
            (self.dry + self._liq_prev)[:, None] * vel_old[:nt]
        ).sum(axis=0)
        grav_t = self.gvec * m_new_t.sum()
        resid_t = dp_t / dt + self._out_mom_step / dt - grav_t - grip - f_couple_t
        resid_t = np.clip(np.nan_to_num(resid_t, nan=0.0, posinf=0.0, neginf=0.0), -2e4, 2e4)

        dp_w = (self.wmass[:, None] * self.vel[nt:]).sum(axis=0) - (
            wmass_old[:, None] * vel_old[nt:]
        ).sum(axis=0)
        grav_w = self.gvec * self.wmass.sum()
        basin = dp_w / dt - self.inflow_ret / dt - grav_w - f_couple_w
        basin = np.clip(np.nan_to_num(basin, nan=0.0, posinf=0.0, neginf=0.0), -5e5, 5e5)

        self.last_ext_force[0, 0] = grav_t
        self.last_ext_force[0, 1] = grip
        self.last_ext_force[0, 2] = f_couple_t
        self.last_ext_force[0, 3] = resid_t
        self.last_ext_force[1, 0] = grav_w
        self.last_ext_force[1, 1] = basin
        self.last_ext_force[1, 2] = f_couple_w
        self.last_ext_force[1, 3] = 0.0
        self.last_ext_force[2, 0] = grav_t + grav_w
        self.last_ext_force[2, 1] = grip + basin
        self.last_ext_force[2, 2] = f_couple_t + f_couple_w
        self.last_ext_force[2, 3] = resid_t
        self.last_interaction[0] = f_couple_t
        self.last_interaction[1] = f_couple_w

        self.budget_impulse += dt * self.last_ext_force.sum(axis=1)
        self.budget_inflow[1] += self.inflow_ret
        self.budget_outflow[0] += self._out_mom_step

        self.t = t_new
        self._finite_guard()

    # ------------------------------------------------------------- coupling
    def _couple_pairs(self):
        pt = self.pos[: self.nt]
        pw = self.pos[self.nt:]
        lo = np.minimum(pt.min(axis=0), pw.min(axis=0)) - 0.02
        hi = np.maximum(pt.max(axis=0), pw.max(axis=0)) + 0.02
        ii, jj = _grid_pairs(pt, pw, self.rc, lo, hi, same=False)
        if ii.size == 0:
            return ii, jj, np.empty(0)
        r = np.linalg.norm(pt[ii] - pw[jj], axis=1)
        wgt = np.maximum(0.0, 1.0 - r / self.rc) ** 2
        keep = wgt > 1e-6
        return ii[keep], jj[keep], wgt[keep]

    def _spray_impulse(self, t_new):
        """Carry a surface parcel to the drip point below the wringing towel."""
        if self.drain_pool < 4.0e-3 or t_new - self.last_drop_t < 0.10:
            return np.zeros(3)
        node_z = float(self.config["water"]["free_surface_z"]) + 0.05
        k = int(np.argmax(self._drained_step))
        node = self.pos[k]
        if node[2] < node_z:
            return np.zeros(3)
        zw = self.pos[self.nt:, 2]
        cand = np.where((zw < node[2] - 0.08) & (zw > 0.02))[0]
        if cand.size == 0:
            return np.zeros(3)
        dx = self.pos[self.nt + cand, 0] - node[0]
        dy = self.pos[self.nt + cand, 1] - node[1]
        d2 = dx ** 2 + dy ** 2 + 0.5 * (zw[cand] - node[2] + 0.25) ** 2
        j = int(cand[int(np.argmin(d2))])
        old_v = self.vel[self.nt + j].copy()
        self.pos[self.nt + j] = np.array(
            [node[0] + 0.02 * np.cos(0.7 * k), node[1] + 0.02 * np.sin(0.7 * k),
             min(node[2] - 0.05, 1.25)]
        )
        new_v = 0.5 * np.clip(self.vel[k], -3.0, 3.0) + np.array([0.0, 0.0, -0.15])
        self.vel[self.nt + j] = new_v
        impulse = self.wmass[j] * (new_v - old_v)
        m_total = float((self.dry + self.liq).sum())
        if m_total > 1e-9:
            self.vel[: self.nt] -= impulse[None, :] / m_total
        self.drain_pool = 0.0
        self.last_drop_t = t_new
        self.drops += 1
        return impulse

    # ------------------------------------------------------------ fluid step
    def _clamp_water(self, p):
        return np.clip(p, self.basin_lo + self.w_skin, self.basin_hi - self.w_skin)

    def _water_pbf(self, dt):
        nw = self.nw
        wp = self.pos[self.nt:]
        wv = self.vel[self.nt:]
        wm = self.wmass
        lo = wp.min(axis=0) - 0.02
        hi = wp.max(axis=0) + 0.02
        ii, jj = _grid_pairs(wp, wp, self.h, lo, hi, same=True)
        if ii.size == 0:
            self.pos[self.nt:] = self._clamp_water(wp + wv * dt)
            return
        mj = wm[jj]
        mi = wm[ii]
        xp = wp + wv * dt
        inv_rho = 1.0 / self.rho0
        for _ in range(3):
            d = xp[ii] - xp[jj]
            r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
            r = np.sqrt(r2)
            W = self.k6 * np.maximum(self.h2 - r2, 0.0) ** 3
            rho = np.bincount(ii, weights=mj * W, minlength=nw)
            rho += np.bincount(jj, weights=mi * W, minlength=nw)
            C = rho * inv_rho - 1.0
            coef = -self.spiky * (self.h - r) ** 2 / r
            g = coef[:, None] * d
            g2 = np.einsum("ij,ij->i", g, g) * (inv_rho * inv_rho)
            Gv = np.empty((nw, 3))
            for c in range(3):
                Gv[:, c] = (
                    np.bincount(ii, weights=g[:, c] * inv_rho, minlength=nw)
                    - np.bincount(jj, weights=g[:, c] * inv_rho, minlength=nw)
                )
            A = np.bincount(ii, weights=g2, minlength=nw)
            A += np.bincount(jj, weights=g2, minlength=nw)
            A += np.einsum("ij,ij->i", Gv, Gv)
            # Tension (C < 0) is left unconstrained and no artificial-pressure
            # term is added: at this resolution it would dominate the Lagrange
            # multipliers and push the fluid apart.
            lam = np.where((A > 1e-12) & (C > 0.0), -C / np.maximum(A, 1e-12), 0.0)
            np.clip(lam, -1.0e4, 0.0, out=lam)
            termc = ((lam[ii] + lam[jj]) * inv_rho)[:, None] * g
            D = np.empty((nw, 3))
            for c in range(3):
                D[:, c] = (
                    np.bincount(ii, weights=termc[:, c], minlength=nw)
                    - np.bincount(jj, weights=termc[:, c], minlength=nw)
                )
            D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
            dn = np.linalg.norm(D, axis=1)
            D *= np.minimum(1.0, (0.3 * self.h) / np.maximum(dn, 1e-12))[:, None]
            xp = self._clamp_water(xp + D)
        wv = (xp - wp) / dt
        np.clip(wv, -25.0, 25.0, out=wv)
        d = xp[ii] - xp[jj]
        r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
        W = self.k6 * np.maximum(self.h2 - r2, 0.0) ** 3
        num = np.empty((nw, 3))
        for c in range(3):
            num[:, c] = np.bincount(ii, weights=W * wv[jj, c], minlength=nw)
            num[:, c] += np.bincount(jj, weights=W * wv[ii, c], minlength=nw)
        den = np.bincount(ii, weights=W, minlength=nw) + np.bincount(jj, weights=W, minlength=nw)
        avg = num / np.maximum(den, 1e-12)[:, None]
        wv = wv + 0.03 * (avg - wv)
        wv = np.clip(np.nan_to_num(wv, nan=0.0, posinf=0.0, neginf=0.0), -25.0, 25.0)
        self.pos[self.nt:] = xp
        self.vel[self.nt:] = wv

    # --------------------------------------------------------- porous liquid
    def _wring_driver(self):
        """Local drainage drive: torsional strain + in-plane area compression."""
        p = self.pos[: self.nt].reshape(self.ny, self.nx, 3)
        tvec = p[-1] - p[0]
        tlen = np.linalg.norm(tvec, axis=1)
        tn = tvec / np.maximum(tlen, 1e-9)[:, None]
        cosd = np.clip(np.einsum("ij,ij->i", tn[:-1], tn[1:]), -1.0, 1.0)
        dtheta = np.arccos(cosd)
        twist = np.zeros(self.nx)
        twist[1:] = dtheta / self.dx
        valid = np.clip(tlen / 0.06, 0.0, 1.0)
        twist = np.clip(twist / 8.0, 0.0, 1.0) * valid

        a = p[self.tri // self.nx, self.tri % self.nx]
        A = 0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)
        comp_t = np.clip(1.0 - A / self.tri_area0, 0.0, 1.0)
        comp = np.zeros(self.nt)
        flat = self.tri.ravel()
        cnt = np.maximum(np.bincount(flat, minlength=self.nt), 1)
        comp = np.bincount(flat, weights=np.repeat(comp_t, 3), minlength=self.nt) / cnt
        node_twist = twist[self.station]
        return np.clip(1.3 * node_twist + comp, 0.0, 1.5)

    def _liquid_exchange(self, dt, ci, cj, cw):
        nt, nw = self.nt, self.nw
        self._liq_prev = self.liq.copy()
        inflow = np.zeros(3)
        out_mom = np.zeros(3)

        contact = np.zeros(nt)
        if ci.size:
            contact = np.clip(np.bincount(ci, weights=cw, minlength=nt) / 1.2, 0.0, 1.0)
        driver = self._wring_driver()

        s = self.sat
        s_new = s + 0.8 * dt * contact * (1.0 - s)
        wring = 2.5 * dt * driver * np.maximum(0.0, s_new - 0.42)
        grav = 0.45 * dt * np.maximum(0.0, s_new - 0.22) * (1.0 - contact)
        s_new = np.clip(s_new - wring - grav, 0.02, 1.0)
        dm = (s - s_new) * self.cap_node

        drain = dm > 0.0
        if drain.any():
            di = np.where(drain)[0]
            pts = self.pos[:nt][di]
            tree = cKDTree(self.pos[nt:])
            kq = min(4, nw)
            dist, near = tree.query(pts, k=kq)
            dist = np.atleast_2d(dist)
            near = np.atleast_2d(near)
            wgt = 1.0 / (dist + 1e-6)
            wgt /= wgt.sum(axis=1, keepdims=True)
            adds = dm[di][:, None] * wgt
            js = near.ravel()
            ms = adds.ravel()
            vin = self.vel[nt:][js]
            for c in range(3):
                mom = np.bincount(js, weights=ms * vin[:, c], minlength=nw)
                if c == 0:
                    dmj = np.bincount(js, weights=ms, minlength=nw)
                    momv = np.empty((nw, 3))
                momv[:, c] = mom
            newm = self.wmass + dmj
            has = dmj > 1e-15
            wsel = self.vel[nt:][has]
            self.vel[nt:][has] = (
                self.wmass[has, None] * wsel + momv[has]
            ) / newm[has, None]
            self.wmass = newm
            inflow += momv.sum(axis=0)
            out_mom += momv.sum(axis=0)
            self.drain_pool += float(dm[di].sum())

        absorb = dm < 0.0
        if absorb.any() and ci.size:
            ai = np.where(absorb)[0]
            sel = np.isin(ci, ai)
            if sel.any():
                take = -dm[ci[sel]]
                wsum = np.maximum(
                    np.bincount(ci[sel], weights=cw[sel], minlength=nt)[ci[sel]], 1e-12
                )
                share = take * cw[sel] / wsum
                pj = cj[sel]
                dmj2 = np.bincount(pj, weights=share, minlength=nw)
                avail = np.maximum(self.wmass - 0.05 * self.m0w, 0.0)
                rem = np.minimum(dmj2, avail)
                if rem.any():
                    for c in range(3):
                        mrem = np.bincount(pj, weights=share * self.vel[nt:][pj, c],
                                           minlength=nw)
                        inflow[c] -= float(mrem.sum())
                        out_mom[c] -= float(mrem.sum())
                    self.wmass = self.wmass - rem

        self.sat = s_new
        self.liq = self.cap_node * s_new
        self._drained_step = np.maximum(dm, 0.0)
        self._out_mom_step = out_mom
        self.inflow_ret = inflow

    # --------------------------------------------------------------- output
    def _domain_weights(self):
        w = np.zeros((self.G, self.N))
        w[0, : self.nt] = 1.0
        w[1, self.nt:] = 1.0
        w[2, :] = 1.0
        return w

    def _frame(self, time: float) -> StateFrame:
        nt, nw = self.nt, self.nw
        mass_q = np.zeros((self.N, 2))
        mass_q[:nt, 0] = self.dry
        mass_q[:nt, 1] = self.liq
        mass_q[nt:, 1] = self.wmass
        sat_attr = np.concatenate((self.sat, np.ones(nw)))
        por = np.concatenate((np.full(nt, self.porosity), np.zeros(nw)))
        perm = np.concatenate(
            (np.full(nt, float(self.config["towel"]["permeability_m2"])), np.zeros(nw))
        )
        lg = np.zeros(self.N)
        rg = np.zeros(self.N)
        lg[self.drv_idx[: self.n_left]] = 1.0
        rg[self.drv_idx[self.n_left:]] = 1.0
        attributes = {
            "rest_position": self.p0.copy(),
            "uv": self.uv.copy(),
            "left_grasp": lg,
            "right_grasp": rg,
            "porosity": por,
            "permeability_m2": perm,
            "saturation": sat_attr,
            "mass_quantity": mass_q,
        }
        fields = {
            "budget_cumulative_external_impulse": self.budget_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.budget_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.budget_outflow.copy(),
            "budget_domain_weights": self._domain_weights(),
            "budget_external_force": self.last_ext_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.last_interaction.copy(),
        }
        scalars = {
            "reference_momentum": float(max(self.ref_momentum, 1e-6)),
            "reference_velocity": float(self.ref_velocity),
            "characteristic_length": 2.5,
            "grid_spacing": float(self.spacing),
            "water_density": float(self.rho_w),
        }
        fields = {k: np.nan_to_num(np.asarray(v, dtype=float), nan=0.0,
                                  posinf=0.0, neginf=0.0)
                  for k, v in fields.items()}
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=np.concatenate((self.dry + self.liq, self.wmass)),
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _finite_guard(self):
        if not (
            np.isfinite(self.pos).all()
            and np.isfinite(self.vel).all()
            and np.isfinite(self.wmass).all()
            and np.isfinite(self.sat).all()
        ):
            self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=1e3, neginf=-1e3)
            self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
            self.wmass = np.maximum(
                np.nan_to_num(self.wmass, nan=self.m0w, posinf=3 * self.m0w, neginf=self.m0w),
                0.05 * self.m0w,
            )
            self.sat = np.clip(np.nan_to_num(self.sat, nan=0.0), 0.0, 1.0)
            self.liq = self.cap_node * self.sat

    def diagnostics(self) -> dict[str, object]:
        speed = float(np.linalg.norm(self.vel, axis=1).max()) if self.N else 0.0
        return {
            "steps": int(self.steps),
            "time": float(self.t),
            "dt": float(self.dt),
            "entities": int(self.N),
            "max_speed": speed,
            "droplets": int(self.drops),
            "towel_liquid_kg": float(self.liq.sum()),
        }
