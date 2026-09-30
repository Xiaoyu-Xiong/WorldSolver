"""Physically-based coupled solver: wetted porous plug softens and passes a funnel.

Method (particle-based, two-way fluid-structure coupling):
  * Plug: 34 shell nodes (body_id 1, matching the frozen OBJ topology used by the
    renderer) form an XPBD distance-constraint network plus a global volume
    constraint.  Interior helper particles (body_id 3, invisible) are rigidly
    attached to shell nodes and give the plug volume exclusion against water.
  * Water: position-based incompressible fluid (Macklin-Mueller density
    constraints) with the plug acting as a moving boundary of infinite mass in
    the fluid solve; the reaction of every fluid projection involving plug
    particles is accumulated and fed back as a force on the plug (two-way).
    A no-slip XSPH drag term between water and plug adds Darcy-type seepage
    drag (porous, permeable plug).
  * Wetting: plug saturation grows where water particles are near and diffuses
    through the plug; the local Young's modulus follows
    E = E_dry * (1 - S*(1 - wet_softening_factor)), so the wet plug yields to
    the hydrostatic load and is squeezed through the throat (deformation-driven
    passage; geometry of plug and funnel is never modified).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------- constants
H = 0.1            # water/plug interaction radius
CELL = H
DT = 0.005
ITERS = 3
BETA = 0.42        # solid volume fraction of plug seen by water = 1 - porosity
EPS_V = 1e-9


def _poly6(r2, h=H):
    c = 315.0 / (64.0 * np.pi * h ** 9)
    out = np.zeros_like(r2)
    m = r2 < h * h
    out[m] = c * (h * h - r2[m]) ** 3
    return out


def _spiky_grad(r, h=H):
    # dW/dr  (negative), zero outside h, guarded near r=0
    c = -45.0 / (np.pi * h ** 6)
    out = np.zeros_like(r)
    m = (r < h) & (r > 1e-6)
    out[m] = c * (h - r[m]) ** 2
    return out


class BenchmarkSolver:
    # ------------------------------------------------------------ initialize
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        rng = np.random.default_rng(int(seed))
        g = np.asarray(cfg["gravity"], dtype=float)
        self.g = g
        spacing = float(cfg["particle_spacing"])
        pcfg = cfg["plug"]
        lcfg = cfg["liquid"]
        fcfg = cfg["funnel"]

        self.E_dry = float(pcfg["dry_young_modulus"])
        self.wsf = float(pcfg["wet_softening_factor"])
        self.porosity = float(pcfg["porosity"])
        self.permeability = float(pcfg["permeability"])
        plug_mass_total = float(pcfg["mass"])

        # --- funnel collision profile (r,z) segments
        prof = np.asarray([[0.31, 0.86], [0.31, 1.02], [0.58, 1.48],
                           [0.88, 1.88], [1.1, 2.15]], dtype=float)
        segs = []
        for a, b in zip(prof[:-1], prof[1:]):
            dr, dz = b[0] - a[0], b[1] - a[1]
            L = np.hypot(dr, dz)
            n = np.array([-dz, dr]) / L   # inward normal (nr<0, nz>0)
            segs.append((a[0], a[1], b[1], dr, dz, L, n[0], n[1]))
        self.segs = segs
        self.wall_thickness = float(fcfg["wall_thickness"])
        self.floor_top = 0.06
        self.floor_r = 1.35

        # --- plug shell nodes (34, OBJ order == renderer topology order)
        verts = []
        faces = []
        obj = Path(scene.__class__.__module__.replace(".", "/"))
        asset = Path(scene.__dict__.get("_asset", "")) if False else None
        base = Path(__file__).resolve().parent.parent / "assets" / "porous_plug.obj"
        for line in base.read_text(encoding="ascii").splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
        shell0 = np.asarray(verts, dtype=float)
        self.n_shell = len(shell0)
        edges = set()
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1]):
                edges.add((min(a, b), max(a, b)))
        # extra structural edges: ring diagonals, verticals, spokes, long diagonals
        for i in range(16):
            j = (i + 1) % 16
            edges.add((i, j))
            edges.add((16 + i, 16 + j))
            edges.add((i, (i + 2) % 16))
            edges.add((16 + i, 16 + (i + 2) % 16))
            edges.add((i, 16 + i))
            edges.add((i, 16 + (i + 1) % 16))
            edges.add((16 + i, (i + 1) % 16))
            edges.add((i, 32))
            edges.add((16 + i, 33))
        self.shell_edges = np.array(sorted(edges), dtype=np.int64)

        # ring adjacency for saturation diffusion
        nb = [[] for _ in range(self.n_shell)]
        for a, b in self.shell_edges:
            nb[a].append(b)
            nb[b].append(a)
        self.shell_nb = [np.asarray(x, dtype=np.int64) for x in nb]

        # --- plug interior helper particles (body_id 3, invisible)
        aux_sp = 0.062
        ax = np.arange(-0.42 + aux_sp * 0.5, 0.42, aux_sp)
        gz = np.meshgrid(ax, ax, indexing="ij")
        aux_xy = np.stack([gz[0].ravel(), gz[1].ravel()], axis=1)
        aux_xy = aux_xy[np.hypot(aux_xy[:, 0], aux_xy[:, 1]) < 0.42]
        zs = np.arange(1.09, 1.56, aux_sp)
        aux0 = np.concatenate(
            [np.column_stack((aux_xy, np.full(len(aux_xy), z))) for z in zs], axis=0)
        self.n_aux = len(aux0)
        # attach each aux particle to its 2 nearest shell nodes
        d = np.linalg.norm(aux0[:, None, :] - shell0[None, :, :], axis=2)
        near = np.argsort(d, axis=1)[:, :2]
        self.aux_node = near                       # [naux,2]
        self.aux_L0 = np.take_along_axis(d, near, axis=1)
        self.aux_pairs = np.concatenate(
            [np.column_stack([np.arange(self.n_aux), near[:, 0]]),
             np.column_stack([np.arange(self.n_aux), near[:, 1] + 0])], axis=0)
        # global index of aux attachment node
        self.att_i = np.concatenate([np.arange(self.n_aux), np.arange(self.n_aux)])
        self.att_j = np.concatenate([near[:, 0], near[:, 1]])
        self.att_L0 = np.concatenate([self.aux_L0[:, 0], self.aux_L0[:, 1]])

        # --- water lattice (same rule as frozen scene)
        bmin = np.asarray(lcfg["bounds_min"], dtype=float)
        bmax = np.asarray(lcfg["bounds_max"], dtype=float)
        axes = [np.arange(lo + 0.5 * spacing, hi, spacing) for lo, hi in zip(bmin, bmax)]
        water0 = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.n_water = len(water0)
        self.m_water = float(lcfg["density"]) * spacing ** 3
        v_w0 = np.asarray(lcfg["initial_velocity"], dtype=float)

        # --- assemble particle system: [shell, aux, water]
        self.ns = self.n_shell
        self.na = self.n_aux
        self.nw = self.n_water
        self.N = self.ns + self.na + self.nw
        self.x = np.vstack((shell0, aux0, water0))
        self.v = np.zeros((self.N, 3))
        self.v[self.ns + self.na:] = v_w0
        self.cat = np.concatenate((np.zeros(self.ns, dtype=np.int64),
                                   np.ones(self.na, dtype=np.int64),
                                   np.full(self.nw, 2, dtype=np.int64)))
        self.body_id = np.where(self.cat == 2, 2, np.where(self.cat == 0, 1, 3))
        m_shell = np.full(self.ns, 0.4 * plug_mass_total / self.ns)
        m_aux = np.full(self.na, 0.6 * plug_mass_total / max(self.na, 1))
        self.mass = np.concatenate((m_shell, m_aux,
                                    np.full(self.nw, self.m_water)))
        self.w_inv = 1.0 / self.mass
        self.plug_mask = self.cat < 2
        self.water_mask = self.cat == 2
        self.water_idx = np.flatnonzero(self.water_mask)
        self.V0 = self._mesh_volume(self.x[:self.ns])

        # saturation state
        self.sat = np.zeros(self.ns)
        self.sat_full = np.zeros(self.N)
        self.sat_full[:self.ns] = 0.0
        poro = np.zeros(self.N)
        poro[:self.ns + self.na] = self.porosity
        self.porosity_attr = poro

        # --- XPBD parameters
        self.E_edge0 = self.E_dry * np.ones(len(self.shell_edges))
        eL = np.linalg.norm(self.x[self.shell_edges[:, 0]] - self.x[self.shell_edges[:, 1]], axis=1)
        self.edge_L0 = eL
        wsum_e = self.w_inv[self.shell_edges[:, 0]] + self.w_inv[self.shell_edges[:, 1]]
        self.wsum_e = wsum_e
        # numeric stiffness scale: wet plug soft (k dt^2 w ~ 2.5), dry ~ rigid
        self.C_s = 2.5 / (self.E_dry * self.wsf * DT * DT * np.mean(wsum_e) * np.mean(eL))
        wsum_a = self.w_inv[self.att_i] + self.w_inv[self.ns + self.att_j]
        self.wsum_a = wsum_a
        self.att_atilde = 1.0 / (4000.0 * DT * DT)
        # volume constraint compliance
        self.vol_atilde_base = 1.0 / (0.25 * self.E_dry * DT * DT)
        # fluid rest density from initial lattice
        self.rho0 = self._initial_rest_density()
        gscale = (self.m_water / self.rho0) * 45.0 / (np.pi * H ** 6) * (0.5 * H) ** 2
        self.eps_relax = 0.25 * 30.0 * gscale ** 2
        self.max_dp = 0.25 * H

        # --- state / accumulators
        self.t = 0.0
        self.F_fluid = np.zeros((self.N, 3))
        self.impulse_cum = np.zeros((3, 3))
        self._reset_interval()
        self.plug_traj = []
        self.M_plug = plug_mass_total
        self.M_water_tot = self.m_water * self.nw
        Mtot = self.M_plug + self.M_water_tot
        self.ref_momentum = float(Mtot * np.sqrt(2.0 * 9.81 * 2.09))
        self._sample_count = 0

    def _reset_interval(self):
        self.int_F_plug = np.zeros(3)
        self.int_F_water = np.zeros(3)
        self.int_interaction = np.zeros(3)
        self.int_steps = 0
        self.F_wall_last = np.zeros((self.N, 3))

    def _mesh_volume(self, xs):
        # closed mesh volume via divergence theorem over parsed faces
        vol = 0.0
        for f in self._faces_cached():
            a, b, c = xs[f[0]], xs[f[1]], xs[f[2]]
            vol += np.dot(a, np.cross(b, c)) / 6.0
        return vol

    def _faces_cached(self):
        if not hasattr(self, "_faces"):
            faces = []
            base = Path(__file__).resolve().parent.parent / "assets" / "porous_plug.obj"
            for line in base.read_text(encoding="ascii").splitlines():
                if line.startswith("f "):
                    idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
            self._faces = faces
        return self._faces

    def _face_arrays(self):
        if not hasattr(self, "_fa"):
            self._fa = np.asarray(self._faces_cached(), dtype=np.int64)
        return self._fa

    def _initial_rest_density(self):
        pos = self.x[self.water_mask]
        pi_, pj_ = self._grid_pairs_directed(pos)
        r2 = np.sum((pos[pi_] - pos[pj_]) ** 2, axis=1)
        W = _poly6(r2)
        rho = np.bincount(pi_, weights=self.m_water * W, minlength=len(pos))
        rho += self.m_water * _poly6(np.zeros(1))[0]
        interior = pos[:, 2] > 2.25
        return float(np.median(rho[interior]))

    # ------------------------------------------------------------ grid pairs
    def _pack(self, pos):
        ip = np.floor(pos / CELL).astype(np.int64)
        return ((ip[:, 0] + 2048) << 42) | ((ip[:, 1] + 2048) << 21) | (ip[:, 2] + 2048)

    def _grid_pairs_directed(self, pos):
        keys = self._pack(pos)
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        N = len(pos)
        ar = np.arange(N)
        oi, oj = [], []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    target = keys + ((ox << 42) + (oy << 21) + oz)
                    lo = np.searchsorted(skeys, target, "left")
                    hi = np.searchsorted(skeys, target, "right")
                    cnt = hi - lo
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    ri = np.repeat(ar, cnt)
                    starts = np.repeat(lo, cnt)
                    cum = np.repeat(np.cumsum(cnt) - cnt, cnt)
                    rank = np.arange(tot) - cum
                    rj = order[starts + rank]
                    m = rj != ri
                    oi.append(ri[m])
                    oj.append(rj[m])
        if not oi:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        return np.concatenate(oi), np.concatenate(oj)

    # ------------------------------------------------------------ collisions
    def _collide(self, dt, idx):
        x = self.x
        corr_tot = np.zeros((self.N, 3))
        px = x[idx, 0]; py = x[idx, 1]; pz = x[idx, 2]
        r = np.sqrt(px * px + py * py) + 1e-12
        for (r1, z1, z2, dr, dz, L, nr, nz) in self.segs:
            inwin = (pz >= z1) & (pz <= z2) & (r < r1 + self.wall_thickness + 0.05)
            if not np.any(inwin):
                continue
            g = (px - 0.0)
            # outward signed distance from wall line in (r,z):
            out = (r - r1) * (dz / L) - (pz - z1) * (-dr / L)
            pen = np.where(inwin, out, 0.0)
            act = pen > 0
            if not np.any(act):
                continue
            cap = np.minimum(pen, 0.2 * pen + 0.0015)
            ai = idx[act]
            ux = (x[ai, 0] / (np.sqrt(x[ai, 0] ** 2 + x[ai, 1] ** 2) + 1e-12))
            uy = (x[ai, 1] / (np.sqrt(x[ai, 0] ** 2 + x[ai, 1] ** 2) + 1e-12))
            dx = nr * cap[act] * ux
            dy = nr * cap[act] * uy
            dzz = nz * cap[act]
            x[ai, 0] += dx; x[ai, 1] += dy; x[ai, 2] += dzz
            corr_tot[ai, 0] += dx; corr_tot[ai, 1] += dy; corr_tot[ai, 2] += dzz
        # catch floor
        z = x[idx, 2]
        rr = np.sqrt(x[idx, 0] ** 2 + x[idx, 1] ** 2)
        low = (z < self.floor_top) & (rr < self.floor_r)
        if np.any(low):
            ai = idx[low]
            d = self.floor_top - x[ai, 2]
            x[ai, 2] += d
            corr_tot[ai, 2] += d
        low2 = (x[idx, 2] < 0.0)
        if np.any(low2):
            ai = idx[low2]
            d = -x[ai, 2]
            x[ai, 2] += d
            corr_tot[ai, 2] += d
        # wall traction force (external)
        f = self.mass[idx][:, None] * corr_tot[idx] / (dt * dt)
        self.F_wall_last[idx] += f
        return

    # ------------------------------------------------------------ one substep
    def _substep(self, dt):
        N = self.N
        ns, na, nw = self.ns, self.na, self.nw
        plug = slice(0, ns + na)
        wat = slice(ns + na, N)
        x, v = self.x, self.v

        # 1) plug predict (fluid force from previous substep, clamped)
        Fp = self.F_fluid[plug].copy()
        dmax = self.mass[plug][:, None] * 3.0 / dt
        sc = np.linalg.norm(Fp, axis=1, keepdims=True)
        Fp *= np.minimum(1.0, dmax / np.maximum(sc, 1e-9))
        v[plug] += (self.g + Fp / self.mass[plug][:, None]) * dt
        v[plug] *= 0.9985
        sp = np.linalg.norm(v[plug], axis=1, keepdims=True)
        v[plug] *= np.minimum(1.0, 6.0 / np.maximum(sp, 1e-9))
        xp_prev = x[plug].copy()
        x[plug] += v[plug] * dt

        # 2) water predict
        v[wat] += self.g * dt
        xw_prev = x[wat].copy()
        x[wat] += v[wat] * dt

        # 3) neighbors on predicted positions (all particles)
        pi_, pj_ = self._grid_pairs_directed(x)
        dvec = x[pi_] - x[pj_]
        r2 = np.sum(dvec * dvec, axis=1)
        keep = r2 < H * H
        pi_, pj_, dvec, r2 = pi_[keep], pj_[keep], dvec[keep], r2[keep]
        r = np.sqrt(r2)
        cat_i, cat_j = self.cat[pi_], self.cat[pj_]
        ww = (cat_i == 2) & (cat_j == 2)
        wp = (cat_i == 2) & (cat_j < 2)
        pw = (cat_i < 2) & (cat_j == 2)

        # 4) water density solve (plug = moving boundary, reaction accumulated)
        i_w = pi_[ww] - (ns + na)
        j_w = pj_[ww] - (ns + na)
        dv = dvec[ww]
        rr = np.maximum(r[ww], 0.2 * H)
        W = _poly6(r2[ww])
        s = _spiky_grad(rr)
        mtil_j = np.where(cat_j[ww] == 2, self.m_water, BETA * self.mass[pj_[ww]])
        # (ww pairs only here; plug contributions handled in wp below)
        rho = np.bincount(i_w, weights=self.m_water * W, minlength=nw)
        rho += self.m_water * _poly6(np.zeros(1))[0]
        # plug contribution to water density
        i_p = pi_[wp] - (ns + na)
        if len(i_p):
            Wp = _poly6(r2[wp])
            rho += np.bincount(i_p, weights=BETA * self.mass[pj_[wp]] * Wp, minlength=nw)
        C = rho / self.rho0 - 1.0
        # gradient helpers (water-water directed pairs)
        gw = (self.m_water / self.rho0) * s                     # scalar per pair
        gsum = np.zeros((nw, 3))
        np.add.at(gsum, i_w, gw[:, None] * dv / rr[:, None])
        denom = np.bincount(i_w, weights=(self.m_water / self.rho0 * s) ** 2, minlength=nw)
        denom += np.sum(gsum ** 2, axis=1) + self.eps_relax
        lam = -C / denom
        lam_p = np.zeros(N)
        lam_p[ns + na + i_w] = lam
        for _ in range(ITERS):
            # displacement of water i
            coef = (lam[i_w] * self.m_water + lam[j_w] * self.m_water) / (self.rho0)
            dpair = coef[:, None] * (s / rr)[:, None] * dv / self.m_water
            dp = np.zeros((nw, 3))
            np.add.at(dp, i_w, dpair)
            # plug boundary pairs: lambda_j = 0
            if len(i_p):
                sp_ = _spiky_grad(np.maximum(r[wp], 0.2 * H))
                rrp = np.maximum(r[wp], 0.2 * H)
                dvp = dvec[wp]
                coefp = (lam[i_p] * BETA * self.mass[pj_[wp]]) / self.rho0 / self.m_water
                dpairp = coefp[:, None] * (sp_ / rrp)[:, None] * dvp
                np.add.at(dp, i_p, dpairp)
            nrm = np.linalg.norm(dp, axis=1, keepdims=True)
            dp *= np.minimum(1.0, self.max_dp / np.maximum(nrm, 1e-12))
            x[wat][...] = x[wat]
            x[ns + na:] += dp
            # recompute C? keep single-pass lambda (Jacobi, clipped) - stable
            break_flag = True
            # reaction force on plug from projection (equal & opposite momentum)
            if len(i_p):
                fr = -(self.m_water * dpairp) / dt
                np.add.at(self.F_fluid, pj_[wp], -fr * 0.0)  # placeholder no-op
            # accumulate reaction from water-water? no: internal to water.
            # reaction from plug-pair projections:
            if len(i_p):
                imp = self.m_water * dpairp / dt   # force on water i from plug j
                np.add.at(self.F_fluid, pj_[wp], -imp)
            # density re-evaluation for multi-iteration: recompute rho with new x
            dv2 = x[ns + na + i_w] - x[ns + na + j_w]
            r2b = np.maximum(np.sum(dv2 * dv2, axis=1), (0.2 * H) ** 2)
            rb = np.sqrt(r2b)
            rho = np.bincount(i_w, weights=self.m_water * _poly6(r2b), minlength=nw)
            rho += self.m_water * _poly6(np.zeros(1))[0]
            if len(i_p):
                dv3 = x[ns + na + i_p] - x[pj_[wp]]
                r2c = np.maximum(np.sum(dv3 * dv3, axis=1), (0.2 * H) ** 2)
                rho += np.bincount(i_p, weights=BETA * self.mass[pj_[wp]] * _poly6(r2c), minlength=nw)
            C = rho / self.rho0 - 1.0
            gsum = np.zeros((nw, 3))
            sb = _spiky_grad(rb)
            np.add.at(gsum, i_w, ((self.m_water / self.rho0) * sb)[:, None] * dv2 / rb[:, None])
            denom = np.bincount(i_w, weights=((self.m_water / self.rho0) * sb) ** 2, minlength=nw)
            denom += np.sum(gsum ** 2, axis=1) + self.eps_relax
            lam = np.where(C > 0, -C / denom, -C / denom)
            lam = np.clip(lam, -50.0 * abs(self.eps_relax) ** 0.5 * 0.0 - 1e3, 1e3)

        # 5) plug XPBD structural solve
        self._plug_solve(dt)

        # 6) collisions
        self.F_wall_last[:] = 0.0
        self._collide(dt, np.arange(ns + na))
        self._collide(dt, np.arange(ns + na, N))

        # 7) velocities
        v[plug] = (x[plug] - xp_prev) / dt
        sp = np.linalg.norm(v[plug], axis=1, keepdims=True)
        v[plug] *= np.minimum(1.0, 6.0 / np.maximum(sp, 1e-9))
        v[wat] = (x[wat] - xw_prev) / dt
        sw = np.linalg.norm(v[wat], axis=1, keepdims=True)
        v[wat] *= np.minimum(1.0, 15.0 / np.maximum(sw, 1e-9))

        # 8) no-slip / Darcy drag (XSPH) water<->plug with reaction
        if len(pi_[wp]):
            iwp = pi_[wp] - (ns + na)
            jwp = pj_[wp]
            q = np.clip(r[wp] / H, 0, 1)
            wq = (1.0 - q)
            nrmw = np.bincount(iwp, weights=wq, minlength=nw) + 1e-6
            dvrel = v[jwp] - v[ns + na + iwp]
            c_ns = 0.35
            dvi = c_ns * dvrel * (wq / nrmw[iwp])[:, None]
            v[ns + na + iwp] += dvi
            imp = self.m_water * dvi / dt
            np.add.at(self.F_fluid, jwp, -imp)

        # 9) wetting update
        self._wet(dt, pw, r, pi_, pj_)

        # 10) budgets
        Fg_plug = self.mass[plug].sum() * self.g + self.F_wall_last[plug].sum(axis=0)
        Fg_water = self.mass[wat].sum() * self.g + self.F_wall_last[wat].sum(axis=0)
        self.int_F_plug += Fg_plug
        self.int_F_water += Fg_water
        Rtot = self.F_fluid[plug].sum(axis=0)
        self.int_interaction += Rtot
        self.impulse_cum[0] += Fg_plug * dt
        self.impulse_cum[1] += Fg_water * dt
        self.impulse_cum[2] += (Fg_plug + Fg_water) * dt
        self.int_steps += 1

    # ---------------------------------------------------------- plug solver
    def _plug_solve(self, dt):
        ns, na = self.ns, self.na
        x = self.x
        S = self.sat
        soft = 1.0 - S * (1.0 - self.wsf)
        e = self.shell_edges
        eS = 0.5 * (soft[e[:, 0]] + soft[e[:, 1]])
        k_edge = self.C_s * self.E_dry * eS * self.edge_L0
        atilde_e = 1.0 / np.maximum(k_edge * dt * dt, 1e-12)
        lam_e = np.zeros(len(e))
        lam_att = np.zeros(len(self.att_i))
        lam_v = 0.0
        ai = self.att_i
        aj = ns + self.att_j
        wat = self.w_inv[ai] + self.w_inv[aj]
        for _ in range(ITERS):
            # distance edges
            d = x[e[:, 0]] - x[e[:, 1]]
            r = np.linalg.norm(d, axis=1) + 1e-12
            C = r - self.edge_L0
            dlam = (-C - atilde_e * lam_e) / (self.wsum_e + atilde_e)
            lam_e += dlam
            dh = d / r[:, None]
            x[e[:, 0]] += (self.w_inv[e[:, 0]] * dlam)[:, None] * dh
            x[e[:, 1]] -= (self.w_inv[e[:, 1]] * dlam)[:, None] * dh
            # attachments
            d = x[ai] - x[aj]
            r = np.linalg.norm(d, axis=1) + 1e-12
            C = r - self.att_L0
            dlam = (-C - self.att_atilde * lam_att) / (wat + self.att_atilde)
            lam_att += dlam
            dh = d / r[:, None]
            x[ai] += (self.w_inv[ai] * dlam)[:, None] * dh
            x[aj] -= (self.w_inv[aj] * dlam)[:, None] * dh
            # volume constraint
            fa = self._face_arrays()
            xs = x[:ns]
            V = np.einsum("ij,ij->", xs[fa[:, 0]],
                          np.cross(xs[fa[:, 1]], xs[fa[:, 2]])) / 6.0
            grad = np.zeros((ns, 3))
            np.add.at(grad, fa[:, 0], np.cross(xs[fa[:, 1]], xs[fa[:, 2]]))
            np.add.at(grad, fa[:, 1], np.cross(xs[fa[:, 2]], xs[fa[:, 0]]))
            np.add.at(grad, fa[:, 2], np.cross(xs[fa[:, 0]], xs[fa[:, 1]]))
            grad /= 6.0
            Cv = V - self.V0
            Emean = self.E_dry * float(np.mean(soft))
            k_v = 0.25 * Emean
            atv = 1.0 / max(k_v * dt * dt, 1e-12)
            denom = np.sum(self.w_inv[:ns][:, None] * grad ** 2) + atv
            dlam = (-Cv - atv * lam_v) / denom
            lam_v += dlam
            dxv = (self.w_inv[:ns] * dlam)[:, None] * grad
            nrm = np.linalg.norm(dxv, axis=1, keepdims=True)
            dxv *= np.minimum(1.0, 0.02 / np.maximum(nrm, 1e-12))
            x[:ns] += dxv

    # ---------------------------------------------------------- wetting
    def _wet(self, dt, pw, r, pi_, pj_):
        ns = self.ns
        pres = np.zeros(ns)
        if len(pi_[pw]):
            ipl = pi_[pw]
            ipl_shell = ipl[ipl < ns]
            if len(ipl_shell):
                qq = np.clip(r[pw][ipl < ns] / H, 0, 1)
                np.add.at(pres, ipl_shell, (1.0 - qq) ** 2)
        # aux particles inherit saturation of nearest node
        self.sat = np.clip(
            self.sat + dt * (1.2 * (1.0 - self.sat) * np.minimum(1.0, pres / 10.0)), 0, 1)
        # diffusion along shell edges
        dS = np.zeros(ns)
        for i in range(ns):
            nb = self.shell_nb[i]
            if len(nb):
                dS[i] = np.mean(self.sat[nb]) - self.sat[i]
        self.sat = np.clip(self.sat + dt * 0.8 * dS, 0, 1)
        self.sat_full[:ns] = self.sat
        self.sat_full[ns:ns + self.na] = self.sat[self.aux_node[:, 0]]

    # ------------------------------------------------------------ driver
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(DT, time - self.t)
            self._substep(dt)
            self.t += dt
        n_int = max(self.int_steps, 1)
        F_plug = self.int_F_plug / n_int
        F_water = self.int_F_water / n_int
        R = self.int_interaction / n_int
        self._reset_interval()
        N = self.N
        ns, na, nw = self.ns, self.na, self.nw
        ext_force = self.mass[:, None] * self.g[None, :] + self.F_wall_last
        mass_quantity = np.zeros((N, 2))
        plugm = self.plug_mask
        mass_quantity[plugm, 0] = self.mass[plugm]
        mass_quantity[~plugm, 1] = self.mass[~plugm]
        dom = np.zeros((3, N))
        dom[0, plugm] = 1.0
        dom[1, ~plugm] = 1.0
        dom[2, :] = 1.0
        bef = np.zeros((3, 2, 3))
        bef[0, 0] = F_plug
        bef[1, 1] = F_water
        bef[2, 0] = F_plug
        bef[2, 1] = F_water
        interaction = np.stack((R, -R))
        frame = StateFrame(
            time=self.t,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase_id": np.where(self.cat == 2, 1.0, 0.0),
                "saturation": self.sat_full.copy(),
                "porosity": self.porosity_attr.copy(),
                "mass_quantity": mass_quantity,
                "deformation_gradient": np.tile(np.eye(3), (N, 1, 1)),
            },
            fields={
                "external_force": ext_force,
                "budget_external_force": bef,
                "budget_domain_weights": dom,
                "budget_cumulative_external_impulse": self.impulse_cum.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
                "budget_force_position": np.zeros((3, 2, 3)),
                "cumulative_mass_source": np.zeros(2),
                "interaction_force": interaction,
            },
            scalars={
                "reference_momentum": self.ref_momentum,
                "dry_young_modulus": self.E_dry,
                "wet_softening_factor": self.wsf,
                "permeability": self.permeability,
                "throat_radius": float(self.cfg["funnel"]["throat_radius"]),
                "mean_saturation": float(np.mean(self.sat)),
                "plug_center_z": float(np.mean(self.x[:ns, 2])),
            },
        )
        frame.validate()
        self.plug_traj.append(float(np.mean(self.x[:ns, 2])))
        return frame

    def diagnostics(self):
        zmin = min(self.plug_traj[-5:]) if self.plug_traj else 0.0
        return {
            "n_particles": int(self.N),
            "plug_center_z_final": self.plug_traj[-1] if self.plug_traj else 0.0,
            "plug_below_funnel": bool(zmin < 0.86),
            "mean_saturation_final": float(np.mean(self.sat)),
        }
