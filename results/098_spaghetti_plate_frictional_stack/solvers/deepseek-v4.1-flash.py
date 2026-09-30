"""Frictional stacking of 31 flexible spaghetti strands in a shallow plate.

Physical model
--------------
Each strand is a discrete elastic rod: 16 nodes joined by 15 inextensible
distance constraints, with bending resistance supplied by second-neighbour
(i, i+2) distance constraints that penalise deviation from the straight rest
shape.  Contact uses sphere samples of radius R placed every 8 mm along every
rod (6 samples per edge plus the terminal node); strand-strand pairs are found
with a uniform spatial hash, and the frozen plate is the axisymmetric container
whose inner meridian profile is used as a piecewise-linear surface.

Numerical method
----------------
A fixed-step position-based (XPBD-style) scheme: predict x + v dt under gravity,
then Gauss-Seidel project distance, bending and unilateral contact constraints
for a fixed number of iterations, then recover v = (x - x_prev)/dt.  Coulomb dry
friction is imposed at the position level by clamping the tangential contact
correction to mu times the normal correction generated in the same substep
(cone-projected friction), which reproduces stick-slip and lets the stack settle
to rest.  Every step is deterministic and finite; no randomness is used.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAV = np.array([0.0, 0.0, -9.81])
_G = 9.81
_MU = 0.3
_DT = 0.001
_ITERS = 4
_KB = 0.15
_DAMP = 0.2
_MAXV = 25.0
_R = 0.006

# Frozen plate inner meridian profile (radius, z), plus the rim top annulus
# implied by wall_thickness = 0.025.
_PROFILE = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.4, 0.018],
        [0.52, 0.075],
        [0.57, 0.125],
        [0.595, 0.125],
    ],
    dtype=float,
)


def _scatter_add(target: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """Accumulate vals[idx] into target (both [n,3]); bincount scatter."""
    n = target.shape[0]
    for c in range(3):
        target[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        body = np.asarray(frame.body_id, dtype=np.int64)
        node = np.asarray(frame.attributes["node_index"], dtype=np.int64)
        self.S = int(body.max()) + 1
        self.N = int(node.max()) + 1
        self.nnode = self.S * self.N
        gidx = body * self.N + node
        self.body_of_node = np.repeat(np.arange(self.S, dtype=np.int64), self.N)
        self.pos = np.zeros((self.nnode, 3), dtype=float)
        self.mass = np.zeros(self.nnode, dtype=float)
        self.pos[gidx] = np.asarray(frame.position, dtype=float)
        self.mass[gidx] = np.asarray(frame.mass, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.inv_m = 1.0 / self.mass
        self.seg = 0.72 / (self.N - 1)
        self._t = 0.0
        self._last_dt = _DT
        self._build_samples()
        self._build_budget()
        # external force record at t = 0: gravity only
        self.ext_force = np.zeros((self.S + 1, 3, 3), dtype=float)
        self.ext_force[: self.S, 0, :] = self.g_body
        self.ext_force[self.S, 0, :] = self.g_total
        self._pairs = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        self._ss_T0 = np.zeros((0, 3), dtype=float)
        self._plate_P0 = np.zeros((self.M, 3), dtype=float)
        self._reset_records()

    # ------------------------------------------------------------------ setup
    def _build_samples(self) -> None:
        rows = []
        n_frac = 6
        for e in range(self.N - 1):
            for k in range(n_frac):
                rows.append((e, k / n_frac))
        rows.append((self.N - 2, 1.0))
        arr = np.asarray(rows, dtype=float)
        a_loc = arr[:, 0].astype(np.int64)
        s = arr[:, 1]
        self.sw0 = np.tile(1.0 - s, self.S)
        self.sw1 = np.tile(s, self.S)
        self.sa = (np.arange(self.S, dtype=np.int64)[:, None] * self.N + a_loc[None, :]).ravel()
        self.sb = (np.arange(self.S, dtype=np.int64)[:, None] * self.N + (a_loc + 1)[None, :]).ravel()
        self.M = self.sa.shape[0]
        im = self.inv_m
        inv_eff = self.sw0**2 * im[self.sa] + self.sw1**2 * im[self.sb]
        self.inv_eff = inv_eff
        self.ka = self.sw0 * im[self.sa] / inv_eff
        self.kb = self.sw1 * im[self.sb] / inv_eff
        self.cutoff = 2.0 * _R + 0.004

    def _build_budget(self) -> None:
        mass_total = float(self.mass.sum())
        self.ref_velocity = float(np.sqrt(_G * 0.72))
        self.ref_force = float(mass_total * _G)
        self.ref_momentum = float(mass_total * self.ref_velocity)
        self.weights = np.zeros((self.S + 1, self.nnode), dtype=float)
        for b in range(self.S):
            self.weights[b, self.body_of_node == b] = 1.0
        self.weights[self.S, :] = 1.0
        gnode = self.mass[:, None] * _GRAV[None, :]
        self.g_body = np.zeros((self.S, 3), dtype=float)
        for c in range(3):
            self.g_body[:, c] = np.bincount(
                self.body_of_node, weights=gnode[:, c], minlength=self.S
            )
        self.g_total = self.g_body.sum(axis=0)
        self.cum = np.zeros((self.S + 1, 3), dtype=float)

    def _reset_records(self) -> None:
        self._pl_nc = np.zeros(self.M, dtype=float)
        self._pl_ft = np.zeros((self.M, 3), dtype=float)
        self._pl_n = np.zeros((self.M, 3), dtype=float)
        self._pl_left = np.zeros(self.M, dtype=float)
        self.node_corr_plate = np.zeros((self.nnode, 3), dtype=float)
        P = self._pairs[0].shape[0]
        self._ss_nc = np.zeros(P, dtype=float)
        self._ss_ft = np.zeros((P, 3), dtype=float)
        self._ss_n = np.zeros((P, 3), dtype=float)
        self._ss_left = np.zeros(P, dtype=float)
        self.node_corr_ss = np.zeros((self.nnode, 3), dtype=float)

    # ------------------------------------------------------------- geometry
    def _sample_pos(self, flat: np.ndarray) -> np.ndarray:
        return self.sw0[:, None] * flat[self.sa] + self.sw1[:, None] * flat[self.sb]

    def _sample_vel(self) -> np.ndarray:
        return self.sw0[:, None] * self.vel[self.sa] + self.sw1[:, None] * self.vel[self.sb]

    def _build_pairs(self, P: np.ndarray, cutoff: float):
        cell = cutoff
        c = np.floor(P / cell).astype(np.int64)
        key = (c[:, 0] + 512) * 1048576 + (c[:, 1] + 512) * 1024 + (c[:, 2] + 512)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uk, first = np.unique(ks, return_index=True)
        starts = first.astype(np.int64)
        ends = np.append(first[1:], len(ks)).astype(np.int64)
        cnts = ends - starts
        iis = []
        jjs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = key + (dx * 1048576 + dy * 1024 + dz)
                    loc = np.searchsorted(uk, nk)
                    np.clip(loc, 0, len(uk) - 1, out=loc)
                    valid = uk[loc] == nk
                    if not valid.any():
                        continue
                    src = np.nonzero(valid)[0]
                    st = starts[loc[src]]
                    ct = cnts[loc[src]]
                    tot = int(ct.sum())
                    if tot == 0:
                        continue
                    iis.append(np.repeat(src, ct))
                    rep_st = np.repeat(st, ct)
                    offs = np.arange(tot) - np.repeat(np.cumsum(ct) - ct, ct)
                    jjs.append(order[rep_st + offs])
        if not iis:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        ii = np.concatenate(iis)
        jj = np.concatenate(jjs)
        keep0 = ii < jj
        ii = ii[keep0]
        jj = jj[keep0]
        d = P[ii] - P[jj]
        dd = np.einsum("ij,ij->i", d, d)
        keep = dd < cutoff * cutoff
        return ii[keep], jj[keep]

    # ----------------------------------------------------------- constraints
    def _solve_distance(self) -> None:
        pos = self.pos.reshape(self.S, self.N, 3)
        d = pos[:, 1:, :] - pos[:, :-1, :]
        dist = np.maximum(np.linalg.norm(d, axis=2), 1e-9)
        n = d / dist[..., None]
        err = (dist - self.seg) / dist
        ia = self.inv_m[:, :-1]
        ib = self.inv_m[:, 1:]
        w = ia + ib
        pos[:, :-1, :] += (ia / w)[..., None] * err[..., None] * n
        pos[:, 1:, :] -= (ib / w)[..., None] * err[..., None] * n

    def _solve_bending(self) -> None:
        if self.N < 3:
            return
        pos = self.pos.reshape(self.S, self.N, 3)
        d = pos[:, 2:, :] - pos[:, :-2, :]
        dist = np.maximum(np.linalg.norm(d, axis=2), 1e-9)
        n = d / dist[..., None]
        err = (dist - 2.0 * self.seg) / dist * _KB
        ia = self.inv_m[:, :-2]
        ib = self.inv_m[:, 2:]
        w = ia + ib
        pos[:, :-2, :] += (ia / w)[..., None] * err[..., None] * n
        pos[:, 2:, :] -= (ib / w)[..., None] * err[..., None] * n

    def _solve_plate(self, dt: float) -> None:
        P = self._sample_pos(self.pos)
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(P[:, 1], P[:, 0])
        z = P[:, 2]
        best_d = np.full(self.M, np.inf)
        best_s = np.zeros(self.M)
        best_nr = np.zeros(self.M)
        best_nz = np.ones(self.M)
        for k in range(len(_PROFILE) - 1):
            r0, z0 = _PROFILE[k]
            r1, z1 = _PROFILE[k + 1]
            dr = r1 - r0
            dz = z1 - z0
            L2 = dr * dr + dz * dz
            t = np.clip(((r - r0) * dr + (z - z0) * dz) / L2, 0.0, 1.0)
            cpr = r0 + t * dr
            cpz = z0 + t * dz
            nr = -dz
            nz = dr
            ln = np.hypot(nr, nz)
            nr = nr / ln
            nz = nz / ln
            drr = r - cpr
            dzz = z - cpz
            dist = np.hypot(drr, dzz)
            signed = drr * nr + dzz * nz
            better = dist < best_d
            best_d[better] = dist[better]
            best_s[better] = signed[better]
            best_nr[better] = nr
            best_nz[better] = nz
        pen = _R - best_s
        act = (pen > 0.0) & (r < 0.62)
        if not np.any(act):
            return
        idx = np.nonzero(act)[0]
        ca = np.cos(th[idx])
        sa = np.sin(th[idx])
        n3 = np.empty((idx.size, 3), dtype=float)
        n3[:, 0] = best_nr[idx] * ca
        n3[:, 1] = best_nr[idx] * sa
        n3[:, 2] = best_nz[idx]
        penv = pen[idx]
        corr = penv[:, None] * n3
        ia = self.sa[idx]
        ib = self.sb[idx]
        cai = self.ka[idx, None] * corr
        cbi = self.kb[idx, None] * corr
        _scatter_add(self.pos, ia, cai)
        _scatter_add(self.pos, ib, cbi)
        _scatter_add(self.node_corr_plate, ia, cai)
        _scatter_add(self.node_corr_plate, ib, cbi)
        self._pl_nc[idx] += penv
        self._pl_n[idx] = n3
        # Coulomb friction: clamp tangential drift to mu * normal correction.
        Pn = self._sample_pos(self.pos)
        dvec = Pn[idx] - self._plate_P0[idx]
        dn = np.einsum("ij,ij->i", dvec, n3)
        tang = dvec - dn[:, None] * n3
        tmag = np.linalg.norm(tang, axis=1)
        budget = _MU * penv + self._pl_left[idx]
        use = np.minimum(tmag, budget)
        sc = np.where(tmag > 1e-12, use / np.maximum(tmag, 1e-12), 0.0)
        cvec = -tang * sc[:, None]
        fa = self.ka[idx, None] * cvec
        fb = self.kb[idx, None] * cvec
        _scatter_add(self.pos, ia, fa)
        _scatter_add(self.pos, ib, fb)
        _scatter_add(self.node_corr_plate, ia, fa)
        _scatter_add(self.node_corr_plate, ib, fb)
        self._pl_left[idx] = budget - use
        self._pl_ft[idx] += cvec

    def _solve_ss(self, dt: float) -> None:
        ii, jj = self._pairs
        if ii.size == 0:
            return
        P = self._sample_pos(self.pos)
        d = P[ii] - P[jj]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n = d / dist[:, None]
        overlap = 2.0 * _R - dist
        act = (overlap > 0.0) | (self._ss_left > 0.0)
        if not np.any(act):
            return
        ia = ii[act]
        ja = jj[act]
        nn = n[act]
        ov = np.maximum(overlap[act], 0.0)
        w = self.inv_eff[ia] + self.inv_eff[ja]
        ci = (self.inv_eff[ia] / w)[:, None] * nn * ov[:, None]
        cj = -ci
        _scatter_add(self.pos, self.sa[ia], self.ka[ia, None] * ci)
        _scatter_add(self.pos, self.sb[ia], self.kb[ia, None] * ci)
        _scatter_add(self.pos, self.sa[ja], self.ka[ja, None] * cj)
        _scatter_add(self.pos, self.sb[ja], self.kb[ja, None] * cj)
        _scatter_add(self.node_corr_ss, self.sa[ia], self.ka[ia, None] * ci)
        _scatter_add(self.node_corr_ss, self.sb[ia], self.kb[ia, None] * ci)
        _scatter_add(self.node_corr_ss, self.sa[ja], self.ka[ja, None] * cj)
        _scatter_add(self.node_corr_ss, self.sb[ja], self.kb[ja, None] * cj)
        self._ss_nc[act] += ov
        self._ss_n[act] = nn
        # Coulomb friction on tangential relative drift since substep start.
        P2 = self._sample_pos(self.pos)
        d2 = P2[ia] - P2[ja]
        dn2 = np.einsum("ij,ij->i", d2, nn)
        tang = d2 - dn2[:, None] * nn
        T0a = self._ss_T0[act]
        d0n = np.einsum("ij,ij->i", T0a, nn)
        t0 = T0a - d0n[:, None] * nn
        drift = tang - t0
        dmag = np.linalg.norm(drift, axis=1)
        budget = _MU * ov + self._ss_left[act]
        use = np.minimum(dmag, budget)
        sc = np.where(dmag > 1e-12, use / np.maximum(dmag, 1e-12), 0.0)
        crel = -drift * sc[:, None]
        c2i = (self.inv_eff[ia] / w)[:, None] * crel
        c2j = -c2i
        _scatter_add(self.pos, self.sa[ia], self.ka[ia, None] * c2i)
        _scatter_add(self.pos, self.sb[ia], self.kb[ia, None] * c2i)
        _scatter_add(self.pos, self.sa[ja], self.ka[ja, None] * c2j)
        _scatter_add(self.pos, self.sb[ja], self.kb[ja, None] * c2j)
        _scatter_add(self.node_corr_ss, self.sa[ia], self.ka[ia, None] * c2i)
        _scatter_add(self.node_corr_ss, self.sb[ia], self.kb[ia, None] * c2i)
        _scatter_add(self.node_corr_ss, self.sa[ja], self.ka[ja, None] * c2j)
        _scatter_add(self.node_corr_ss, self.sb[ja], self.kb[ja, None] * c2j)
        self._ss_left[act] = budget - use
        self._ss_ft[act] += crel

    # ------------------------------------------------------------- dynamics
    def _substep(self, dt: float) -> None:
        self._last_dt = dt
        P0 = self._sample_pos(self.pos)
        self._pairs = self._build_pairs(P0, self.cutoff)
        ii, jj = self._pairs
        self._ss_T0 = (P0[ii] - P0[jj]) if ii.size else np.zeros((0, 3))
        self._plate_P0 = P0
        self._reset_records()
        self.vel += _GRAV[None, :] * dt
        self.vel *= max(0.0, 1.0 - _DAMP * dt)
        sp = np.linalg.norm(self.vel, axis=1)
        scale = np.where(sp > _MAXV, _MAXV / np.maximum(sp, 1e-12), 1.0)
        self.vel *= scale[:, None]
        prev = self.pos.copy()
        self.pos += self.vel * dt
        for _ in range(_ITERS):
            self._solve_distance()
            self._solve_bending()
            self._solve_plate(dt)
            self._solve_ss(dt)
        self.vel = (self.pos - prev) / dt
        self._accumulate_budgets()

    def _accumulate_budgets(self) -> None:
        dt = max(self._last_dt, 1e-9)
        Fp = np.zeros((self.S, 3), dtype=float)
        Fs = np.zeros((self.S, 3), dtype=float)
        imp_p = self.mass[:, None] * self.node_corr_plate / dt
        imp_s = self.mass[:, None] * self.node_corr_ss / dt
        for c in range(3):
            Fp[:, c] = np.bincount(self.body_of_node, weights=imp_p[:, c], minlength=self.S) / dt
            Fs[:, c] = np.bincount(self.body_of_node, weights=imp_s[:, c], minlength=self.S) / dt
        F = np.zeros((self.S + 1, 3, 3), dtype=float)
        F[: self.S, 0, :] = self.g_body
        F[self.S, 0, :] = self.g_total
        # Body domains: gravity plus both contact contributions.  The combined
        # strand domain sees the plate as external (a boundary traction) while
        # strand-strand pairs are internal and cancel, so only term 1 survives.
        F[: self.S, 1, :] = Fp
        F[: self.S, 2, :] = Fs
        F[self.S, 1, :] = Fp.sum(axis=0)
        self.ext_force = F
        self.cum += F.sum(axis=1) * dt

    # ---------------------------------------------------------------- frames
    def _contact_fields(self):
        dt = max(self._last_dt, 1e-9)
        vs = self._sample_vel()
        n_parts = []
        fn_parts = []
        v_parts = []
        ft_parts = []
        coef_parts = []
        idx = np.nonzero(self._pl_nc > 0.0)[0]
        if idx.size:
            n3 = self._pl_n[idx]
            inv = self.inv_eff[idx]
            n_parts.append(n3)
            fn_parts.append(self._pl_nc[idx] / (inv * dt * dt))
            v_parts.append(vs[idx])
            ft_parts.append(self._pl_ft[idx] / (inv[:, None] * dt * dt))
            coef_parts.append(np.full(idx.size, _MU))
        ii, jj = self._pairs
        if ii.size:
            act = np.nonzero(self._ss_nc > 0.0)[0]
            if act.size:
                n3 = self._ss_n[act]
                w = self.inv_eff[ii[act]] + self.inv_eff[jj[act]]
                n_parts.append(n3)
                fn_parts.append(self._ss_nc[act] / (w * dt * dt))
                v_parts.append(vs[ii[act]] - vs[jj[act]])
                ft_parts.append(self._ss_ft[act] / (w[:, None] * dt * dt))
                coef_parts.append(np.full(act.size, _MU))
        if not n_parts:
            nrm = np.array([[0.0, 0.0, 1.0]])
            fn = np.zeros(1)
            vr = np.zeros((1, 3))
            ft = np.zeros((1, 3))
            mask = np.zeros(1)
            coef = np.full(1, _MU)
        else:
            nrm = np.concatenate(n_parts, axis=0)
            fn = np.concatenate(fn_parts, axis=0)
            vr = np.concatenate(v_parts, axis=0)
            ft = np.concatenate(ft_parts, axis=0)
            coef = np.concatenate(coef_parts, axis=0)
            mask = (fn > 0.0).astype(float)
        return nrm, fn, vr, ft, coef, mask

    def _make_frame(self, time: float) -> StateFrame:
        nrm, fn, vr, ft, coef, mask = self._contact_fields()
        node_index = np.tile(np.arange(self.N, dtype=np.int64), self.S)
        is_end = ((node_index == 0) | (node_index == self.N - 1)).astype(np.int64)
        fields = {
            "budget_cumulative_external_impulse": self.cum.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((self.S + 1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((self.S + 1, 3)),
            "budget_domain_weights": self.weights.copy(),
            "budget_external_force": self.ext_force.copy(),
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": mask,
            "friction_dynamic_coefficient": coef.copy(),
            "friction_normal": nrm.copy(),
            "friction_normal_force": fn.copy(),
            "friction_relative_velocity": vr.copy(),
            "friction_static_coefficient": coef.copy(),
            "friction_tangential_force": ft.copy(),
        }
        scalars = {
            "friction_stick_speed_tolerance": 0.01,
            "reference_force": float(self.ref_force),
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
            "strand_radius": float(_R),
            "strand_length": 0.72,
        }
        attributes = {
            "strand_id": self.body_of_node.copy(),
            "node_index": node_index,
            "is_endpoint": is_end,
            "mass_quantity": self.mass.copy(),
        }
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_of_node.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        guard = 0
        while time - self._t > 1e-9:
            dt = min(_DT, time - self._t)
            if dt <= 1e-12:
                break
            self._substep(dt)
            self._t += dt
            guard += 1
            if guard > 200000:
                break
        self._t = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "xpbd_mass_spring_rod",
            "strands": int(self.S),
            "nodes_per_strand": int(self.N),
            "substep_dt": float(_DT),
            "friction_coefficient": float(_MU),
            "sim_time": float(self._t),
        }
