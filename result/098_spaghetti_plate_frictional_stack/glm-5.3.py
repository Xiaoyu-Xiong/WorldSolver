"""Frictional stacking of flexible spaghetti strands in a plate.

Custom impulse-based elastic-rod dynamics implemented with NumPy only
(the `genesis` package is neither imported nor used).

Model
-----
* each strand is a chain of 16 lumped-mass nodes (masses taken from the
  frozen initial frame) with
  - Hookean stretch springs + damping along the strand,
  - a discrete bending energy E = kb/2 * sum |x[i-1]-2x[i]+x[i+1]|^2
    (isometric-bending approximation of a straight elastic rod);
* gravity is uniform and downward;
* contacts (strand-strand and strand-plate) are resolved with velocity-level
  impulses plus a small Baumgarte penetration bias, iterated with a relaxed
  Jacobi scheme;
* dry friction is an exact Coulomb cone: tangential impulses are capped by
  mu * lambda_n with mu = 0.3 for both static and kinetic friction;
* the plate is a static axisymmetric shallow dish with a raised rim whose
  inner profile is frozen in the scene geometry.

The time integration is semi-implicit (symplectic) Euler with a fixed
substep, which is unconditionally stable for the impulse contacts and
stable for the chosen stiffnesses, and fully deterministic.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

# ---- frozen physical constants -------------------------------------------
MU = 0.3                     # dry friction (static and kinetic)
GRAVITY = np.array([0.0, 0.0, -9.81])

# ---- numerics -------------------------------------------------------------
DT = 4.0e-4                  # fixed substep
N_CONTACT_ITER = 8           # relaxed-Jacobi contact iterations
K_STRETCH = 350.0            # N/m  stretch stiffness
C_STRETCH = 0.15             # N*s/m stretch damping
KB = 90.0                    # N/m  discrete bending stiffness (EI ~ KB*h^3)
CD = 0.01                    # N*s/m discrete bending damping
DRAG = 0.3                   # 1/s   mild aerodynamic drag (settling aid)
BETA = 0.25                  # Baumgarte penetration bias
V_CLAMP = 12.0               # m/s   safety velocity clamp
RELAX = 0.9                  # per-node Jacobi relaxation factor

# ---- plate geometry (frozen inner profile) --------------------------------
PLATE_R = np.array([0.0, 0.10, 0.40, 0.52, 0.57])
PLATE_Z = np.array([0.0, 0.00, 0.018, 0.075, 0.125])
PLATE_SL_R = np.array([0.10, 0.40, 0.52, 0.57])
PLATE_SL = np.array([0.06, 0.475, 1.0])
RIM_R = 0.57
RIM_Z = 0.125

STRAND_R = 0.006
CONTACT_DIST = 2.0 * STRAND_R


def _plate_surface(r):
    """Inner-surface height and radial slope of the axisymmetric plate."""
    z = np.interp(r, PLATE_R, PLATE_Z)
    s = np.interp(r, PLATE_SL_R, PLATE_SL, left=0.0, right=0.0)
    return z, s


class BenchmarkSolver:
    """Impulse-based rod solver for the spaghetti-plate frictional stack."""

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = getattr(scene, "config", {})
        self.seed = int(seed)

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body = np.array(frame0.body_id, dtype=int)
        self.node_index = np.array(frame0.attributes["node_index"], dtype=int)
        self.is_endpoint = np.array(frame0.attributes["is_endpoint"], dtype=float)
        self.base_scalars = dict(frame0.scalars)

        self.N = self.pos.shape[0]
        self.M = int(cfg.get("initial_conditions", {}).get(
            "nodes_per_strand", 16)) or 16
        self.S = self.N // self.M
        self.h = float(cfg.get("initial_conditions", {}).get(
            "strand_length", 0.72)) / (self.M - 1)

        self.minv = 1.0 / self.mass
        self.minv3 = self.minv[:, None]
        self.grav_f = self.mass[:, None] * GRAVITY[None, :]
        self.mass_per_strand = self.mass.reshape(self.S, self.M).sum(axis=1)
        self.total_mass = float(self.mass.sum())

        # ---- collision probe points (nodes + 3 interior points/segment) ----
        loc = []
        for n in range(self.M):
            loc.append((n, 1.0, n, 0.0))
        for n in range(self.M - 1):
            for f in (0.25, 0.5, 0.75):
                loc.append((n, 1.0 - f, n + 1, f))
        loc = np.asarray(loc, dtype=float)
        per = loc.shape[0]
        strand_off = np.repeat(np.arange(self.S) * self.M, per)
        self.p_na = loc[:, 0].astype(int) + strand_off
        self.p_wa = np.tile(loc[:, 1], self.S)
        self.p_nb = loc[:, 2].astype(int) + strand_off
        self.p_wb = np.tile(loc[:, 3], self.S)
        self.p_body = np.repeat(np.arange(self.S), per)
        self.p_minv = (self.p_wa ** 2 * self.minv[self.p_na]
                       + self.p_wb ** 2 * self.minv[self.p_nb])

        # ---- budget domains: global + one per strand -----------------------
        self.G = self.S + 1
        w = np.zeros((self.G, self.N))
        w[0, :] = 1.0
        for s in range(self.S):
            w[s + 1, s * self.M:(s + 1) * self.M] = 1.0
        self.domain_w = w
        self.cum_ext = np.zeros((self.G, 3))
        self.plate_imp_strand = np.zeros((self.S, 3))
        self.cross_imp_strand = np.zeros((self.S, 3))
        self.plate_imp_total = np.zeros(3)
        self._snapshot = None

        self.step = 0
        self.t = 0.0
        self._emitted = False

    # ------------------------------------------------------------------
    def _neighbor_pairs(self, P):
        """Undirected candidate pairs (i<j) of probe points within one cell."""
        n = len(P)
        c = np.floor(P / CONTACT_DIST).astype(np.int64)
        c -= c.min(axis=0)
        dims = c.max(axis=0) + 3
        d1, d2 = int(dims[1]), int(dims[2])
        key = (c[:, 0] * d1 + c[:, 1]) * d2 + c[:, 2]
        order = np.argsort(key, kind="stable")
        skey = key[order]

        offs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    if (dx, dy, dz) < (0, 0, 0):
                        continue
                    offs.append((dx * d1 + dy) * d2 + dz)

        ii_all, jj_all = [], []
        arange_n = np.arange(n)
        for koff in offs + [0]:
            tkey = key + koff
            left = np.searchsorted(skey, tkey, "left")
            right = np.searchsorted(skey, tkey, "right")
            cnt = right - left
            tot = int(cnt.sum())
            if tot == 0:
                continue
            q = np.repeat(arange_n, cnt)
            starts = np.repeat(left, cnt)
            base = np.repeat(np.cumsum(cnt) - cnt, cnt)
            seq = np.arange(tot) - base
            m = order[starts + seq]
            if koff == 0:
                keep = q < m
                q, m = q[keep], m[keep]
            ii_all.append(q)
            jj_all.append(m)
        if not ii_all:
            return (np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        return np.concatenate(ii_all), np.concatenate(jj_all)

    # ------------------------------------------------------------------
    def _plate_contacts(self, xp):
        """Node contacts with the static axisymmetric plate (dish + rim wall)."""
        x, y, z = xp[:, 0], xp[:, 1], xp[:, 2]
        r = np.hypot(x, y)
        zsurf, slope = _plate_surface(r)
        invn = 1.0 / np.sqrt(1.0 + slope * slope)
        gap = (z - zsurf) * invn
        pen = STRAND_R - gap
        m1 = pen > 0.0

        penw = r - (RIM_R - STRAND_R)
        m2 = (penw > 0.0) & (z < RIM_Z) & (r > 1e-9)

        idx = np.concatenate([np.nonzero(m1)[0], np.nonzero(m2)[0]])
        if idx.size == 0:
            return (np.zeros(0, dtype=int), np.zeros((0, 3)),
                    np.zeros(0), np.zeros(0, dtype=bool))
        safe = np.maximum(r[idx], 1e-12)
        ux = np.where(r[idx] > 1e-9, x[idx] / safe, 0.0)
        uy = np.where(r[idx] > 1e-9, y[idx] / safe, 0.0)
        sl = slope[idx]
        iv = invn[idx]
        nrm = np.empty((idx.size, 3))
        floor_mask = m1[idx]
        # dish surface normal
        nrm[:, 0] = -sl * ux * iv
        nrm[:, 1] = -sl * uy * iv
        nrm[:, 2] = iv
        # rim wall normal (inward radial) overrides for wall contacts
        wrows = ~floor_mask
        if wrows.any():
            nrm[wrows, 0] = -ux[wrows]
            nrm[wrows, 1] = -uy[wrows]
            nrm[wrows, 2] = 0.0
        pen_all = np.where(floor_mask, pen[idx], penw[idx])
        return idx, nrm, pen_all, np.ones(idx.size, dtype=bool)

    # ------------------------------------------------------------------
    def _substep(self, dt):
        S, M, N = self.S, self.M, self.N
        x, v = self.pos, self.vel
        X = x.reshape(S, M, 3)
        V = v.reshape(S, M, 3)

        # ---- internal elastic forces ----
        d = X[:, 1:] - X[:, :-1]
        ln = np.linalg.norm(d, axis=2)
        np.maximum(ln, 1e-12, out=ln)
        u = d / ln[..., None]
        ext = ln - self.h
        vrel = np.sum((V[:, 1:] - V[:, :-1]) * u, axis=2)
        fseg = (-(K_STRETCH * ext + C_STRETCH * vrel))[..., None] * u
        Fs = np.zeros((S, M, 3))
        Fs[:, 1:] += fseg
        Fs[:, :-1] -= fseg

        dd = X[:, :-2] - 2.0 * X[:, 1:-1] + X[:, 2:]
        ddp = np.zeros((S, M, 3))
        ddp[:, 1:-1] = dd
        ddv = V[:, :-2] - 2.0 * V[:, 1:-1] + V[:, 2:]
        ddvp = np.zeros((S, M, 3))
        ddvp[:, 1:-1] = ddv

        def st(a):
            o = np.zeros_like(a)
            o[:, 1:-1] = a[:, :-2] - 2.0 * a[:, 1:-1] + a[:, 2:]
            o[:, 0] = a[:, 1]
            o[:, -1] = a[:, -2]
            return o

        Fb = -KB * st(ddp) - CD * st(ddvp)
        Ftot = self.grav_f + (Fs + Fb).reshape(N, 3)

        # ---- semi-implicit Euler ----
        v += Ftot * self.minv3 * dt
        v *= 1.0 - DRAG * dt
        sp = np.linalg.norm(v, axis=1)
        fast = sp > V_CLAMP
        if fast.any():
            v[fast] *= (V_CLAMP / sp[fast])[:, None]

        xp = x + v * dt

        # ---- contact detection ----
        P = self.p_wa[:, None] * xp[self.p_na] + self.p_wb[:, None] * xp[self.p_nb]
        ii, jj = self._neighbor_pairs(P)
        na_l, wa_l, nb_l, wb_l, nc_l, wc_l, nd_l, wd_l = [], [], [], [], [], [], [], []
        nvec_l, pen_l, meff_l, plate_l = [], [], [], []

        if ii.size:
            dvec = P[ii] - P[jj]
            dist = np.linalg.norm(dvec, axis=1)
            ok = (dist < CONTACT_DIST) & (dist > 1e-9)
            ok &= self.p_body[ii] != self.p_body[jj]
            if ok.any():
                i = ii[ok]
                j = jj[ok]
                nrm = dvec[ok] / dist[ok][:, None]
                nvec_l.append(nrm)
                pen_l.append(CONTACT_DIST - dist[ok])
                meff_l.append(1.0 / (self.p_minv[i] + self.p_minv[j]))
                na_l.append(self.p_na[i]); wa_l.append(self.p_wa[i])
                nb_l.append(self.p_nb[i]); wb_l.append(self.p_wb[i])
                nc_l.append(self.p_na[j]); wc_l.append(self.p_wa[j])
                nd_l.append(self.p_nb[j]); wd_l.append(self.p_wb[j])
                plate_l.append(np.zeros(i.size, dtype=bool))

        pidx, pnrm, ppen, _ = self._plate_contacts(xp)
        if pidx.size:
            nvec_l.append(pnrm)
            pen_l.append(ppen)
            meff_l.append(self.mass[pidx])
            na_l.append(pidx); wa_l.append(np.ones(pidx.size))
            nb_l.append(pidx); wb_l.append(np.zeros(pidx.size))
            nc_l.append(pidx); wc_l.append(np.zeros(pidx.size))
            nd_l.append(pidx); wd_l.append(np.zeros(pidx.size))
            plate_l.append(np.ones(pidx.size, dtype=bool))

        if nvec_l:
            c_n = np.concatenate(nvec_l)
            c_pen = np.concatenate(pen_l)
            c_meff = np.concatenate(meff_l)
            c_na = np.concatenate(na_l); c_wa = np.concatenate(wa_l)
            c_nb = np.concatenate(nb_l); c_wb = np.concatenate(wb_l)
            c_nc = np.concatenate(nc_l); c_wc = np.concatenate(wc_l)
            c_nd = np.concatenate(nd_l); c_wd = np.concatenate(wd_l)
            is_plate = np.concatenate(plate_l)
            C = c_n.shape[0]
        else:
            C = 0

        # ---- contact / friction solve (relaxed Jacobi) ----
        J_acc = np.zeros((max(C, 1), 3))
        if C:
            slot_idx = np.concatenate([c_na, c_nb, c_nc, c_nd])
            slot_w = np.concatenate([c_wa, c_wb, c_wc, c_wd])
            cnt = np.bincount(slot_idx, weights=(slot_w > 0).astype(float),
                              minlength=N)
            omega = 1.0 / (1.0 + RELAX * cnt)

            lam = np.zeros(C)
            Jn_acc = np.zeros(C)
            Jt_acc = np.zeros((C, 3))
            for it in range(N_CONTACT_ITER):
                vA = c_wa[:, None] * v[c_na] + c_wb[:, None] * v[c_nb]
                vB = c_wc[:, None] * v[c_nc] + c_wd[:, None] * v[c_nd]
                vr = vA - vB
                vn = np.sum(vr * c_n, axis=1)
                jn = np.where(vn < 0.0, -c_meff * vn, 0.0)
                if it == 0:
                    jn = jn + c_meff * (BETA * c_pen / dt)
                lam += jn
                vt = vr - vn[:, None] * c_n
                vtn = np.linalg.norm(vt, axis=1)
                jt_mag = np.minimum(c_meff * vtn, MU * np.maximum(lam, 0.0))
                scale = np.where(vtn > 1e-12, jt_mag / np.maximum(vtn, 1e-12), 0.0)
                jt = -scale[:, None] * vt
                J = jn[:, None] * c_n + jt
                Jn_acc += jn
                Jt_acc += jt
                J_acc[:C] += J
                for idx_a, w_a, sgn in ((c_na, c_wa, 1.0), (c_nb, c_wb, 1.0),
                                        (c_nc, c_wc, -1.0), (c_nd, c_wd, -1.0)):
                    contrib = (sgn * w_a)[:, None] * J
                    contrib *= omega[idx_a][:, None] * self.minv[idx_a][:, None]
                    np.add.at(v, idx_a, contrib)

            # ---- external impulse bookkeeping ----
            Jc = J_acc[:C]
            plate_rows = is_plate
            strand_rows = ~plate_rows
            plate_imp = np.zeros((N, 3))
            if plate_rows.any():
                np.add.at(plate_imp, c_na[plate_rows], Jc[plate_rows])
            self.plate_imp_strand = plate_imp.reshape(S, M, 3).sum(axis=1)
            self.plate_imp_total = self.plate_imp_strand.sum(axis=0)

            cross = np.zeros((S, 3))
            if strand_rows.any():
                sa = c_na[strand_rows] // M
                sb = c_nc[strand_rows] // M
                np.add.at(cross, sa, Jc[strand_rows])
                np.add.at(cross, sb, -Jc[strand_rows])
            self.cross_imp_strand = cross

            # ---- sample-time contact snapshot ----
            self._snapshot = {
                "na": c_na.copy(), "wa": c_wa.copy(),
                "nb": c_nb.copy(), "wb": c_wb.copy(),
                "nc": c_nc.copy(), "wc": c_wc.copy(),
                "nd": c_nd.copy(), "wd": c_wd.copy(),
                "n": c_n.copy(), "lam": Jn_acc.copy(), "Jt": Jt_acc.copy(),
            }
        else:
            self.plate_imp_strand = np.zeros((S, 3))
            self.plate_imp_total = np.zeros(3)
            self.cross_imp_strand = np.zeros((S, 3))
            self._snapshot = None

        # ---- cumulative external impulse per budget domain ----
        self.cum_ext[0] += (self.total_mass * GRAVITY * dt
                            + self.plate_imp_total)
        self.cum_ext[1:] += (self.mass_per_strand[:, None] * GRAVITY * dt
                             + self.plate_imp_strand
                             + self.cross_imp_strand)

        # ---- position update ----
        x += v * dt
        np.clip(x[:, 2], -0.2, 2.0, out=x[:, 2])

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        target = int(round(t / DT))
        if target < self.step:
            target = self.step
        while self.step < target:
            self._substep(DT)
            self.step += 1
            self.t = self.step * DT
        return self._build_frame(self.t)

    # ------------------------------------------------------------------
    def _build_frame(self, t):
        v = self.vel
        snap = self._snapshot
        if snap is not None and snap["lam"].size:
            vA = (snap["wa"][:, None] * v[snap["na"]]
                  + snap["wb"][:, None] * v[snap["nb"]])
            vB = (snap["wc"][:, None] * v[snap["nc"]]
                  + snap["wd"][:, None] * v[snap["nd"]])
            vr = vA - vB
            C = snap["lam"].shape[0]
            normals = snap["n"]
            normal_force = snap["lam"] / DT
            tangential = snap["Jt"] / DT
            active = np.ones(C)
        else:
            C = 0
            vr = np.zeros((0, 3))
            normals = np.zeros((0, 3))
            normal_force = np.zeros(0)
            tangential = np.zeros((0, 3))
            active = np.zeros(0)

        G = self.G
        ext_f = np.zeros((G, 1, 3))
        ext_f[0, 0] = self.total_mass * GRAVITY + self.plate_imp_total / DT
        ext_f[1:, 0] = (self.mass_per_strand[:, None] * GRAVITY
                        + self.plate_imp_strand / DT
                        + self.cross_imp_strand / DT)

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((G, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((G, 3)),
            "budget_domain_weights": self.domain_w.copy(),
            "budget_external_force": ext_f,
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": active,
            "friction_dynamic_coefficient": np.full(C, MU),
            "friction_normal": normals,
            "friction_normal_force": normal_force,
            "friction_relative_velocity": vr,
            "friction_static_coefficient": np.full(C, MU),
            "friction_tangential_force": tangential,
        }

        scalars = dict(self.base_scalars)
        scalars.update({
            "friction_stick_speed_tolerance": 1.0e-3,
            "reference_force": float(self.total_mass * 9.81),
            "reference_momentum": float(self.total_mass * 1.0),
            "reference_velocity": 1.0,
        })

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "strand_id": self.body.copy(),
                "node_index": self.node_index,
                "is_endpoint": self.is_endpoint,
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "time": self.t,
            "substeps": self.step,
            "substep_dt": DT,
            "nodes": self.N,
            "strands": self.S,
            "method": "semi-implicit Euler + relaxed-Jacobi Coulomb impulses",
        }
