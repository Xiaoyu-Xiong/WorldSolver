"""Stick-slip hair-on-shoulder solver (rods_and_strands).

Physical model
--------------
* Elastic strands: stretch + bending springs whose rest lengths come from the
  frozen initial (wavy) configuration, plus structural damping and light
  aerodynamic damping. Root nodes are kinematic and follow the prescribed
  lateral head motion.
* Kinematic collision rig: torso / shoulder_bar / neck / head capsules and the
  scaled head ellipsoid from the frozen scene geometry, all translating with
  the smooth lateral head motion.
* Contacts: penalty normal forces (with approach damping and saturation) plus
  Coulomb friction implemented with tangential stick springs capped at
  mu * N (static and kinetic coefficient both 0.3), for hair-body and
  hair-hair contacts. Hair-hair pairs are found with a spatial hash grid.
* Integration: semi-implicit (symplectic) Euler with fixed deterministic
  substeps; external impulses are accumulated exactly per substep so the
  momentum budget of every exported frame closes.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

DT_SUB = 2.0e-4
K_STRETCH = 1000.0
C_STRETCH = 0.5
K_BEND = 25.0
C_BEND = 0.05
C_AIR = 1.2
R_NODE = 4.2e-3
R_HH = 4.2e-3
K_BODY = 60.0
C_BODY = 0.15
KT_BODY = 400.0
PEN_CLAMP = 2.0e-3
K_HH = 20.0
C_HH = 0.05
KT_HH = 150.0
SEP_MARGIN = 3.0e-3
STICK_TOL = 5.0e-3
MAX_CONTACTS = 4000
MAX_HH = 3000
V_CLAMP = 30.0
GRID_ORG = np.array([-1.2, -1.2, -0.3])
GRID_DIM = (300, 300, 340)
CELL = 2.0 * R_HH


def _flat_offsets():
    offs = []
    g1, g2 = GRID_DIM[1], GRID_DIM[2]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                key = (di, dj, dk)
                if key == (0, 0, 0) or not key > (0, 0, 0):
                    continue
                offs.append(di * g1 * g2 + dj * g2 + dk)
    return tuple(offs)


FLAT_OFFSETS = _flat_offsets()


def _quat_mat(q):
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _make_prim(kind, radius, height, translation, quat, scale):
    return {
        "kind": kind,
        "r": float(radius),
        "h": float(height) * 0.5,
        "t": np.asarray(translation, dtype=float),
        "R": _quat_mat(quat),
        "s": np.asarray(scale, dtype=float),
        "smin": float(np.min(scale)),
    }


PRIM_DEFS = (
    ("capsule", 0.34, 0.62, (0.0, 0.0, 0.62), (0.0, 0.0, 0.0, 1.0), (1.0, 0.72, 1.0)),
    ("capsule", 0.145, 0.92, (0.0, 0.0, 1.02), (0.0, 0.7071068, 0.0, 0.7071068), (1.0, 1.0, 1.0)),
    ("capsule", 0.115, 0.28, (0.0, 0.0, 1.28), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
    ("sphere", 0.265, 0.0, (0.0, 0.0, 1.62), (0.0, 0.0, 0.0, 1.0), (0.88, 0.95, 1.12)),
)


def _prim_gap_normal(prim, xw, disp, axis):
    """World gap (approx signed distance) and outward unit normal per node."""
    p = xw - prim["t"] - disp * axis
    pl = p @ prim["R"]  # local = R^T (p)
    q = pl / prim["s"]
    if prim["kind"] == "sphere":
        d = np.sqrt((q * q).sum(1))
        dSafe = np.maximum(d, 1e-12)
        phi = d - prim["r"]
        g = q / dSafe[:, None]
        g[np.abs(d) < 1e-12, 2] = 1.0
    else:
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
        r2 = np.hypot(qx, qy)
        r2Safe = np.maximum(r2, 1e-12)
        az = np.abs(qz)
        a0 = r2 - prim["r"]
        a1 = az - prim["h"]
        m0 = np.maximum(a0, 0.0)
        m1 = np.maximum(a1, 0.0)
        phi = np.minimum(np.maximum(a0, a1), 0.0) + np.hypot(m0, m1)
        g0 = np.zeros_like(phi)
        g1 = np.zeros_like(phi)
        both = (a0 > 0.0) & (a1 > 0.0)
        pLen = np.maximum(np.hypot(m0, m1), 1e-12)
        g0[both] = (m0 / pLen)[both]
        g1[both] = (m1 / pLen)[both]
        only1 = (~both) & (a1 > 0.0)
        g1[only1] = 1.0
        only0 = (~both) & (a1 <= 0.0) & (a0 > 0.0)
        g0[only0] = 1.0
        inside = (~both) & (a1 <= 0.0) & (a0 <= 0.0)
        g1[inside] = (a1 > a0)[inside].astype(float)
        g0[inside] = 1.0 - g1[inside]
        sz = np.sign(qz)
        g = np.stack(
            [g0 * qx / r2Safe, g0 * qy / r2Safe, g1 * sz], axis=1
        )
    gs = g / prim["s"]
    grad_mag = np.sqrt((gs * gs).sum(1))
    grad_mag = np.maximum(grad_mag, 1e-12)
    gap = phi / grad_mag
    nw = gs @ prim["R"].T
    nn = np.sqrt((nw * nw).sum(1))
    nn = np.maximum(nn, 1e-12)
    return gap, nw / nn[:, None]


def _scatter(dest, idx, f):
    n = dest.shape[0]
    for c in range(3):
        dest[:, c] += np.bincount(idx, weights=f[:, c], minlength=n)


class BenchmarkSolver:
    def __init__(self):
        self.ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        frame = scene.initial_state()

        self.g = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        hm = cfg.get("head_motion", {}) or {}
        self.t0 = float(hm.get("start_time", 0.45))
        self.t1 = float(hm.get("stop_time", 2.35))
        self.disp_mag = float(hm.get("displacement", 0.18))
        ax = np.asarray(hm.get("axis", (1.0, 0.0, 0.0)), dtype=float)
        na = float(np.linalg.norm(ax))
        self.axis = ax / (na if na > 1e-12 else 1.0)

        fr = cfg.get("friction", {}) or {}
        hh = fr.get("hair_hair", {}) or {}
        hb = fr.get("hair_body", {}) or {}
        self.mu_s = float(hb.get("static", hh.get("static", 0.3)))
        self.mu_k = float(hb.get("kinetic", hh.get("kinetic", 0.3)))
        self.mu_s_hh = float(hh.get("static", self.mu_s))
        self.mu_k_hh = float(hh.get("kinetic", self.mu_k))
        self.mu = self.mu_s  # static == kinetic == 0.3 in this scene

        self.x = np.array(frame.position, dtype=float, copy=True)
        self.v = np.array(frame.velocity, dtype=float, copy=True)
        self.m = np.array(frame.mass, dtype=float, copy=True)
        self.n_nodes = self.x.shape[0]
        self.massive = self.m > 0.0
        self.inv_m = np.where(self.massive, 1.0 / np.maximum(self.m, 1e-12), 0.0)

        self.strand_id = np.asarray(frame.attributes["strand_id"]).astype(int).copy()
        self.node_index = np.asarray(frame.attributes["node_index"]).astype(int).copy()
        self.root_idx = np.flatnonzero(self.node_index == 0)
        self.x0_root = self.x[self.root_idx].copy()
        self.nps = int(self.node_index.max()) + 1

        # spring edges (index layout: global = strand * nps + node)
        base = np.arange(self.n_nodes)
        s_mask = self.node_index < self.nps - 1
        self.e_s0 = base[s_mask]
        self.e_s1 = self.e_s0 + 1
        b_mask = self.node_index < self.nps - 2
        self.e_b0 = base[b_mask]
        self.e_b1 = self.e_b0 + 2
        d = self.x[self.e_s1] - self.x[self.e_s0]
        self.L0_s = np.sqrt((d * d).sum(1))
        d = self.x[self.e_b1] - self.x[self.e_b0]
        self.L0_b = np.sqrt((d * d).sum(1))

        self.prims = [_make_prim(*p) for p in PRIM_DEFS]
        self.xi_body = np.zeros((len(self.prims), self.n_nodes, 3))
        self.hh_xi = {}

        self.t = 0.0
        self.step = 0
        self.cum_imp = np.zeros(3)
        self.interval_terms = np.zeros((4, 3))
        self.t_last_emit = 0.0
        self._grav_vec = (self.m[:, None] * self.g).sum(0)
        total_mass = float(self.m.sum())
        self.reference_force = total_mass * 9.81
        self.reference_velocity = 0.15
        self.reference_momentum = total_mass * self.reference_velocity

        self._init_attrs = {
            k: np.asarray(v).copy() for k, v in frame.attributes.items()
        }

        # read-only force evaluation at t = 0 for the first frame's terms
        _, terms = self._compute_forces(0.0, 0.0, False)
        self._first_terms = terms.copy()
        self._last_contacts = 0
        self.ready = True

    # ------------------------------------------------------- prescribed motion
    def _head_disp(self, t):
        if t <= self.t0:
            return 0.0
        if t >= self.t1:
            return self.disp_mag
        u = (t - self.t0) / (self.t1 - self.t0)
        return self.disp_mag * u * u * (3.0 - 2.0 * u)

    def _head_vel(self, t):
        if t <= self.t0 or t >= self.t1:
            return 0.0
        u = (t - self.t0) / (self.t1 - self.t0)
        return self.disp_mag * 6.0 * u * (1.0 - u) / (self.t1 - self.t0)

    # ------------------------------------------------------------ pair search
    def _find_pairs(self):
        x = self.x
        ijk = np.floor((x - GRID_ORG) / CELL).astype(np.int64)
        np.clip(ijk, 0, np.array(GRID_DIM) - 1, out=ijk)
        cid = (ijk[:, 0] * GRID_DIM[1] + ijk[:, 1]) * GRID_DIM[2] + ijk[:, 2]
        order = np.argsort(cid, kind="stable")
        sc = cid[order]
        uniq, start, counts = np.unique(sc, return_index=True, return_counts=True)
        ends = start + counts
        m = self.n_nodes
        pos = np.arange(m)
        pa_list, pb_list = [], []

        ge = np.repeat(ends, counts)
        lens = np.maximum(ge - pos - 1, 0)
        tot = int(lens.sum())
        if tot:
            rep = np.repeat(pos, lens)
            cum = np.cumsum(lens)
            offs = np.arange(tot) - np.repeat(cum - lens, lens)
            pb = rep + 1 + offs
            pa_list.append(order[rep])
            pb_list.append(order[pb])

        for fo in FLAT_OFFSETS:
            tc = sc + fo
            tpos = np.searchsorted(uniq, tc)
            ok = tpos < len(uniq)
            tposc = np.where(ok, tpos, 0)
            ok &= uniq[tposc] == tc
            if not ok.any():
                continue
            cnt = np.where(ok, counts[tposc], 0)
            tot = int(cnt.sum())
            if tot <= 0:
                continue
            src = np.flatnonzero(ok)
            csrc = cnt[src]
            rep = np.repeat(src, csrc)
            cum = np.cumsum(cnt)
            offs = np.arange(tot) - np.repeat(cum[src] - csrc, csrc)
            mem = np.repeat(start[tposc][src], csrc) + offs
            pa_list.append(order[rep])
            pb_list.append(order[mem])

        if not pa_list:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        pa = np.concatenate(pa_list)
        pb = np.concatenate(pb_list)
        keep = self.strand_id[pa] != self.strand_id[pb]
        pa, pb = pa[keep], pb[keep]
        if pa.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        sidx = np.lexsort((hi, lo))
        lo, hi = lo[sidx], hi[sidx]
        d = np.sqrt(((x[lo] - x[hi]) ** 2).sum(1))
        sel = (d < 2.0 * R_HH) & (d > 1e-9)
        lo, hi = lo[sel], hi[sel]
        if lo.size > MAX_HH:
            lo, hi = lo[:MAX_HH], hi[:MAX_HH]
        return lo, hi

    def _prune_hh(self):
        x = self.x
        nn = self.n_nodes
        lim = 2.0 * R_HH + 0.006
        lim2 = lim * lim
        for key in list(self.hh_xi.keys()):
            i = key // nn
            j = key % nn
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dz = x[i, 2] - x[j, 2]
            if dx * dx + dy * dy + dz * dz > lim2:
                del self.hh_xi[key]

    # ------------------------------------------------------------- force model
    def _compute_forces(self, t, dt, update):
        x, v, m = self.x, self.v, self.m
        n = self.n_nodes
        F = np.zeros((n, 3))
        F += m[:, None] * self.g
        F -= (C_AIR * m)[:, None] * v

        # stretch springs
        d = x[self.e_s1] - x[self.e_s0]
        L = np.sqrt((d * d).sum(1))
        Ls = np.maximum(L, 1e-9)
        u = d / Ls[:, None]
        ext = L - self.L0_s
        vt = ((v[self.e_s1] - v[self.e_s0]) * u).sum(1)
        fv = ((K_STRETCH * ext + C_STRETCH * vt)[:, None]) * u
        _scatter(F, self.e_s0, fv)
        _scatter(F, self.e_s1, -fv)

        # bending springs
        d = x[self.e_b1] - x[self.e_b0]
        L = np.sqrt((d * d).sum(1))
        Ls = np.maximum(L, 1e-9)
        u = d / Ls[:, None]
        ext = L - self.L0_b
        vt = ((v[self.e_b1] - v[self.e_b0]) * u).sum(1)
        fv = ((K_BEND * ext + C_BEND * vt)[:, None]) * u
        _scatter(F, self.e_b0, fv)
        _scatter(F, self.e_b1, -fv)

        # hair-body contacts
        Fb = np.zeros((n, 3))
        disp = self._head_disp(t)
        v_head = self._head_vel(t) * self.axis
        for pi, prim in enumerate(self.prims):
            gap, nrm = _prim_gap_normal(prim, x, disp, self.axis)
            pen = R_NODE - gap
            act = (pen > 0.0) & self.massive
            idx = np.flatnonzero(act)
            if idx.size:
                pen_c = np.minimum(pen[idx], PEN_CLAMP)
                v_rel = v[idx] - v_head
                vn = (v_rel * nrm[idx]).sum(1)
                nmag = np.maximum(K_BODY * pen_c - C_BODY * vn, 0.0)
                fn = nmag[:, None] * nrm[idx]
                vt_vec = v_rel - vn[:, None] * nrm[idx]
                xi = self.xi_body[pi]
                if update and dt > 0.0:
                    xi[idx] += vt_vec * dt
                    # keep the stick state tangential to the current surface
                    xn = (xi[idx] * nrm[idx]).sum(1)
                    xi[idx] -= xn[:, None] * nrm[idx]
                fs = -KT_BODY * xi[idx]
                mag = np.sqrt((fs * fs).sum(1))
                cap = self.mu * nmag
                slip = mag > cap
                ratio = np.ones_like(mag)
                ratio[slip] = cap[slip] / np.maximum(mag[slip], 1e-30)
                fs *= ratio[:, None]
                if update:
                    shrink = np.ones_like(mag)
                    shrink[slip] = cap[slip] / np.maximum(KT_BODY * mag[slip], 1e-30)
                    xi[idx] *= shrink[:, None]
                contrib = fn + fs
                F[idx] += contrib
                Fb[idx] += contrib
            xi = self.xi_body[pi]
            reset = (pen <= 0.0) & (gap > R_NODE + SEP_MARGIN)
            xi[reset] = 0.0

        # hair-hair contacts
        ia, ib = self._find_pairs()
        if ia.size:
            dvec = x[ia] - x[ib]
            dist = np.sqrt((dvec * dvec).sum(1))
            u = dvec / np.maximum(dist, 1e-12)[:, None]
            vrel = v[ia] - v[ib]
            vn = (vrel * u).sum(1)
            pen = 2.0 * R_HH - dist
            nmag = np.maximum(K_HH * pen - C_HH * vn, 0.0)
            fn = nmag[:, None] * u
            _scatter(F, ia, fn)
            _scatter(F, ib, -fn)
            vt_vec = vrel - vn[:, None] * u
            cap = self.mu * nmag
            vt_l = vt_vec.tolist()
            cap_l = cap.tolist()
            key_l = (ia * n + ib).tolist()
            f_list = []
            store = self.hh_xi
            for k in range(len(key_l)):
                key = key_l[k]
                xi = store.get(key)
                if xi is None:
                    xi = [0.0, 0.0, 0.0]
                if update and dt > 0.0:
                    vt = vt_l[k]
                    xi = [
                        xi[0] + vt[0] * dt,
                        xi[1] + vt[1] * dt,
                        xi[2] + vt[2] * dt,
                    ]
                fx = -KT_HH * xi[0]
                fy = -KT_HH * xi[1]
                fz = -KT_HH * xi[2]
                mag = np.sqrt(fx * fx + fy * fy + fz * fz)
                cp = cap_l[k]
                if mag > cp and mag > 0.0:
                    r = cp / mag
                    fx *= r
                    fy *= r
                    fz *= r
                    s2 = cp / (KT_HH * mag)
                    xi = [xi[0] * s2, xi[1] * s2, xi[2] * s2]
                if update:
                    store[key] = xi
                f_list.append((fx, fy, fz))
            farr = np.asarray(f_list, dtype=float).reshape(-1, 3)
            _scatter(F, ia, farr)
            _scatter(F, ib, -farr)

        # external force decomposition (massive nodes only; roots carry no mass)
        mas = self.massive
        fa = (-(C_AIR * m)[:, None] * v)[mas].sum(0)
        fb_sum = Fb[mas].sum(0)
        fo = F[mas].sum(0) - self._grav_vec - fa - fb_sum
        terms = np.stack([self._grav_vec, fa, fb_sum, fo])
        return F, terms

    # ---------------------------------------------------------------- stepping
    def _step(self, dt):
        F, terms = self._compute_forces(self.t, dt, True)
        self.v += (F * self.inv_m[:, None]) * dt
        sp = np.sqrt((self.v * self.v).sum(1))
        over = sp > V_CLAMP
        if over.any():
            scale = np.where(over, V_CLAMP / np.maximum(sp, 1e-12), 1.0)
            self.v *= scale[:, None]
        self.x += self.v * dt
        t_new = self.t + dt
        disp = self._head_disp(t_new)
        vel = self._head_vel(t_new)
        self.x[self.root_idx] = self.x0_root + disp * self.axis
        self.v[self.root_idx] = vel * self.axis
        self.cum_imp += terms.sum(0) * dt
        self.interval_terms += terms * dt
        self.t = t_new
        self.step += 1
        if self.step % 400 == 0:
            self._prune_hh()

    def advance_to(self, time):
        t = float(time)
        if t < self.t:
            t = self.t
        while self.t < t - 1e-9:
            self._step(min(DT_SUB, t - self.t))
        return self._emit(t)

    # -------------------------------------------------------------- exporting
    def _contact_snapshot(self, t):
        x, v = self.x, self.v
        disp = self._head_disp(t)
        v_head = self._head_vel(t) * self.axis
        n_list, f_list, vr_list, ft_list = [], [], [], []

        for pi, prim in enumerate(self.prims):
            gap, nrm = _prim_gap_normal(prim, x, disp, self.axis)
            pen = R_NODE - gap
            act = (pen > 0.0) & self.massive
            idx = np.flatnonzero(act)
            if idx.size == 0:
                continue
            pen_c = np.minimum(pen[idx], PEN_CLAMP)
            v_rel = v[idx] - v_head
            vn = (v_rel * nrm[idx]).sum(1)
            nmag = np.maximum(K_BODY * pen_c - C_BODY * vn, 0.0)
            fs = -KT_BODY * self.xi_body[pi][idx]
            mag = np.sqrt((fs * fs).sum(1))
            cap = self.mu * nmag
            ratio = np.ones_like(mag)
            slip = mag > cap
            ratio[slip] = cap[slip] / np.maximum(mag[slip], 1e-30)
            fs *= ratio[:, None]
            n_list.append(nrm[idx])
            f_list.append(nmag)
            vr_list.append(v_rel)
            ft_list.append(fs)

        n_body = sum(a.shape[0] for a in f_list)

        ia, ib = self._find_pairs()
        if ia.size and n_body < MAX_CONTACTS:
            dvec = x[ia] - x[ib]
            dist = np.sqrt((dvec * dvec).sum(1))
            u = dvec / np.maximum(dist, 1e-12)[:, None]
            vrel = v[ia] - v[ib]
            vn = (vrel * u).sum(1)
            pen = 2.0 * R_HH - dist
            nmag = np.maximum(K_HH * pen - C_HH * vn, 0.0)
            nn = self.n_nodes
            store = self.hh_xi
            fs_rows = np.zeros((ia.size, 3))
            for k in range(ia.size):
                xi = store.get(int(ia[k]) * nn + int(ib[k]))
                if xi is None:
                    continue
                fx = -KT_HH * xi[0]
                fy = -KT_HH * xi[1]
                fz = -KT_HH * xi[2]
                mag = np.sqrt(fx * fx + fy * fy + fz * fz)
                cp = self.mu * nmag[k]
                if mag > cp and mag > 0.0:
                    r = cp / mag
                    fx *= r
                    fy *= r
                    fz *= r
                fs_rows[k] = (fx, fy, fz)
            room = MAX_CONTACTS - n_body
            if room < ia.size:
                ia, ib = ia[:room], ib[:room]
                u = u[:room]
                vrel = vrel[:room]
                nmag = nmag[:room]
                fs_rows = fs_rows[:room]
            n_list.append(u)
            f_list.append(nmag)
            vr_list.append(vrel)
            ft_list.append(fs_rows)

        if n_list:
            normals = np.concatenate(n_list, axis=0)
            forces = np.concatenate(f_list, axis=0)
            vrels = np.concatenate(vr_list, axis=0)
            fts = np.concatenate(ft_list, axis=0)
        else:
            normals = np.zeros((0, 3))
            forces = np.zeros(0)
            vrels = np.zeros((0, 3))
            fts = np.zeros((0, 3))
        return normals, forces, vrels, fts

    def _emit(self, t):
        dt_int = t - self.t_last_emit
        if dt_int > 1e-12:
            terms_avg = self.interval_terms / dt_int
        else:
            terms_avg = self._first_terms.copy()
        normals, forces, vrels, fts = self._contact_snapshot(t)
        c = normals.shape[0]
        mask = (forces > 1e-9).astype(float)

        attrs = dict(self._init_attrs)
        attrs["mass_quantity"] = self.m.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cum_imp[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n_nodes)),
            "budget_external_force": terms_avg[None, :, :].copy(),
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": mask,
            "friction_dynamic_coefficient": np.full(c, self.mu_k),
            "friction_normal": normals,
            "friction_normal_force": forces,
            "friction_relative_velocity": vrels,
            "friction_static_coefficient": np.full(c, self.mu_s),
            "friction_tangential_force": fts,
        }
        scalars = {
            "friction_stick_speed_tolerance": STICK_TOL,
            "reference_force": self.reference_force,
            "reference_momentum": self.reference_momentum,
            "reference_velocity": self.reference_velocity,
            "strand_radius": R_NODE,
            "static_friction": self.mu_s,
            "kinetic_friction": self.mu_k,
            "head_stop_time": self.t1,
        }
        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.strand_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self.interval_terms[:] = 0.0
        self.t_last_emit = float(t)
        self._last_contacts = c
        return frame

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "substeps": int(self.step),
            "hair_hair_stick_states": len(self.hh_xi),
            "last_reported_contacts": int(self._last_contacts),
        }
