"""Physics solver: strand collisions on a rapidly rotating hairy ball.

Elastic hair strands (finite stretching stiffness, bending stiffness about a
naturally curled rest shape) are rooted on a kinematically prescribed rotating
sphere.  Dynamics use substepped XPBD (extended position based dynamics) for
the internal rod constraints, projective/impulse contacts against the sphere
and between strands, and position-level Coulomb friction with mu = 0.2 for
both static and kinetic friction.  Implemented from scratch with NumPy only.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

MU = 0.2                 # static = kinetic friction (frozen)
DRAG = 0.5               # linear air-drag coefficient [1/s]
SUBSTEPS_PER_SAMPLE = 40 # dt_sub = 5e-4 s at sample_dt = 0.02 s
CONSTRAINT_ITERS = 5
CONTACT_ITERS = 3
CONTACT_H = 0.0035       # strand-strand contact distance [m]
REPORT_MARGIN = 1.0e-3   # reporting margin for persistent contacts [m]
VMAX = 30.0              # velocity clamp [m/s]
MAX_CORR = 0.004         # max sphere-depenetration correction per substep [m]
MAX_PAIRS = 400000


# ----------------------------------------------------------------------------
# Quaternion / rotation utilities
# ----------------------------------------------------------------------------
def _quat_from_rotvec(rv):
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1.0e-14:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rv / theta
    s = np.sin(0.5 * theta)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * theta)])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + bw * ax + ay * bz - az * by,
        aw * by + bw * ay + az * bx - ax * bz,
        aw * bz + bw * az + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _rotvec_from_quat(q):
    v = np.asarray(q[:3], dtype=float)
    w = float(q[3])
    nv = float(np.linalg.norm(v))
    if nv < 1.0e-14:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(nv, w)
    return v * (angle / nv)


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


class _SphereRotation:
    """Smooth (C1) prescribed rotation from axis-angle keyframes."""

    def __init__(self, keyframes):
        self.times = []
        self.rv = []
        for k in keyframes:
            self.times.append(float(k["time"]))
            axis = np.asarray(k["axis"], dtype=float)
            na = float(np.linalg.norm(axis))
            if na > 1.0e-12:
                axis = axis / na
            angle = float(k["angle"]) * np.pi / 180.0
            self.rv.append(axis * angle)
        self.times = np.asarray(self.times, dtype=float)
        self.rv = np.asarray(self.rv, dtype=float)

    def rotvec(self, t):
        if t <= self.times[0]:
            return self.rv[0]
        if t >= self.times[-1]:
            return self.rv[-1]
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        u = (t - t0) / (t1 - t0)
        u = min(max(u, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep: zero angular rate at keys
        return (1.0 - s) * self.rv[i] + s * self.rv[i + 1]

    def quat(self, t):
        return _quat_from_rotvec(self.rotvec(t))

    def omega(self, t, eps=1.0e-3):
        q = _quat_mul(self.quat(t + eps), _quat_conj(self.quat(t - eps)))
        return _rotvec_from_quat(q) / (2.0 * eps)


class BenchmarkSolver:
    """Substepped XPBD strand solver with frictional sphere/strand contacts."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        p = cfg["physical_parameters"]

        self.S = int(p["strand_count"])
        self.K = int(p["segments_per_strand"])
        self.Kn = self.K + 1
        self.N = self.S * self.Kn
        self.g = np.asarray(cfg["gravity"], dtype=float)

        self.X = np.asarray(frame0.position, dtype=float).reshape(self.S, self.Kn, 3).copy()
        self.V = np.zeros_like(self.X)
        mf = np.asarray(frame0.mass, dtype=float).copy()
        self.mf = mf
        self.m = mf.reshape(self.S, self.Kn)
        self.w = np.where(self.m > 0.0, 1.0 / np.maximum(self.m, 1e-30), 0.0)
        self.wf = self.w.reshape(-1).copy()
        self.nonroot = self.wf > 0.0
        self.strand_flat = np.asarray(frame0.attributes["strand_id"], dtype=float).copy()
        self.root_flat = np.asarray(frame0.attributes["root_mask"], dtype=float).copy()
        self.m_total = float(mf.sum())

        # Rest geometry from the natural curled configuration.
        d = self.X[:, 1:, :] - self.X[:, :-1, :]
        self.L0 = np.sqrt((d * d).sum(-1))                      # [S,K]
        d2 = self.X[:, 2:, :] - self.X[:, :-2, :]
        self.D0 = np.sqrt((d2 * d2).sum(-1))                    # [S,K-1]

        r = float(p["strand_radius"])
        area = np.pi * r * r
        E_s = float(p["stretch_young_modulus"])
        E_b = float(p["bending_young_modulus"])
        EA = E_s * area
        EI = E_b * np.pi * r ** 4 / 4.0
        self.alpha_stretch = self.L0 / EA                       # 1/k, k = EA/L0
        self.alpha_bend = self.L0[:, :-1] ** 2 / (2.0 * EI)     # soft bend compliance

        self.R_ball = float(p["sphere_radius"])
        self.R_c = self.R_ball + r                              # sphere contact radius
        self.h = CONTACT_H
        self.mu = MU

        self.dt_sample = float(cfg["sample_dt"])
        self.dt_sub = self.dt_sample / SUBSTEPS_PER_SAMPLE
        self.steps = 0
        self.t_last = 0.0

        self.rot = _SphereRotation(cfg["prescribed_sphere_rotation"]["keyframes_axis_angle_degrees"])
        self.roots0 = self.X[:, 0, :].copy()

        # Momentum-budget accumulators.
        self.cum_ext = np.zeros(3)      # cumulative external impulse since t=0
        self.int_Jg = np.zeros(3)       # interval gravity impulse
        self.int_Jdrag = np.zeros(3)    # interval air-drag impulse
        self.int_Jsph = np.zeros(3)     # interval net sphere-contact impulse
        self.int_Jroot = np.zeros(3)    # interval root-attachment impulse
        self.int_Jsph_node = np.zeros((self.N, 3))
        self.int_Jss_node = np.zeros((self.N, 3))
        self.last_pair_count = 0
        self.reset_count = 0

        weight = self.m_total * float(np.linalg.norm(self.g))
        self.ref_force = weight if weight > 0 else 1e-3
        self.ref_momentum = max(self.m_total * 1.0, 1e-9)
        self.ref_velocity = 1.0

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.steps * self.dt_sub,
            "substeps": self.steps,
            "dt_sub": self.dt_sub,
            "last_pair_count": int(self.last_pair_count),
            "nan_resets": int(self.reset_count),
        }

    # ------------------------------------------------------------------
    def _detect_pairs(self, Pf, h):
        """Neighbor pairs (i<j cells) of nodes from different strands within h."""
        C = np.floor(Pf / h).astype(np.int64)
        np.clip(C, -512, 511, out=C)
        key = ((C[:, 0] + 1024) * 2048 + (C[:, 1] + 1024)) * 2048 + (C[:, 2] + 1024)
        order = np.argsort(key, kind="stable")
        Ks = key[order]
        N = Pf.shape[0]
        ar = np.arange(N)
        hh = h * h
        strand = self.strand_flat
        outa, outb = [], []
        offsets = [(0, 0, 0)]
        offsets += [(1, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        offsets += [(0, 1, dz) for dz in (-1, 0, 1)]
        offsets += [(0, 0, 1)]
        for (ox, oy, oz) in offsets:
            k2 = ((C[:, 0] + ox + 1024) * 2048 + (C[:, 1] + oy + 1024)) * 2048 + (C[:, 2] + oz + 1024)
            lo = np.searchsorted(Ks, k2, side="left")
            hi = np.searchsorted(Ks, k2, side="right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            i_idx = np.repeat(ar, cnt)
            starts = np.repeat(lo, cnt)
            bases = np.repeat(np.cumsum(cnt) - cnt, cnt)
            j_idx = order[starts + (np.arange(tot) - bases)]
            if ox == 0 and oy == 0 and oz == 0:
                sel = i_idx < j_idx
                i_idx = i_idx[sel]
                j_idx = j_idx[sel]
            if i_idx.size == 0:
                continue
            dv = Pf[j_idx] - Pf[i_idx]
            sel = (dv * dv).sum(1) < hh
            if not sel.any():
                continue
            i_idx = i_idx[sel]
            j_idx = j_idx[sel]
            sel = strand[i_idx] != strand[j_idx]
            if sel.any():
                outa.append(i_idx[sel])
                outb.append(j_idx[sel])
        if not outa:
            e = np.empty(0, dtype=np.int64)
            return e, e
        ia = np.concatenate(outa)
        ib = np.concatenate(outb)
        if ia.size > MAX_PAIRS:
            ia = ia[:MAX_PAIRS]
            ib = ib[:MAX_PAIRS]
        return ia, ib

    # ------------------------------------------------------------------
    def _substep(self) -> None:
        dt = self.dt_sub
        t_now = self.steps * dt
        t_end = t_now + dt
        S, Kn = self.S, self.Kn
        X = self.X
        Xprev = X.copy()
        Vf = self.V.reshape(-1, 3)
        P_old = (self.mf[:, None] * Vf).sum(0)

        # External dynamics: gravity and linear air drag.
        V = self.V + self.g[None, None, :] * dt
        Vf = V.reshape(-1, 3)
        Jdrag = -DRAG * dt * (self.mf[:, None] * Vf).sum(0)
        V *= (1.0 - DRAG * dt)

        Xp = X + V * dt
        q = self.rot.quat(t_end)
        Rm = _quat_to_mat(q)
        roots_world = self.roots0 @ Rm.T
        Xp[:, 0, :] = roots_world
        om = self.rot.omega(0.5 * (t_now + t_end))

        w = self.w
        alph_s = self.alpha_stretch / (dt * dt)
        alph_b = self.alpha_bend / (dt * dt)
        lam = np.zeros((S, self.K))
        lam2 = np.zeros((S, self.K - 1))

        # --- internal elastic constraints (XPBD) -------------------------
        for _ in range(CONSTRAINT_ITERS):
            d = Xp[:, 1:, :] - Xp[:, :-1, :]
            dist = np.sqrt((d * d).sum(-1))
            np.maximum(dist, 1.0e-12, out=dist)
            C = dist - self.L0
            wsum = w[:, :-1] + w[:, 1:]
            dlam = (-C - alph_s * lam) / (wsum + alph_s)
            lam += dlam
            dhat = d / dist[..., None]
            Xp[:, :-1, :] -= (w[:, :-1] * dlam)[..., None] * dhat
            Xp[:, 1:, :] += (w[:, 1:] * dlam)[..., None] * dhat

            d2 = Xp[:, 2:, :] - Xp[:, :-2, :]
            dist2 = np.sqrt((d2 * d2).sum(-1))
            np.maximum(dist2, 1.0e-12, out=dist2)
            C2 = dist2 - self.D0
            wsum2 = w[:, :-2] + w[:, 2:]
            dlam2 = (-C2 - alph_b * lam2) / (wsum2 + alph_b)
            lam2 += dlam2
            dhat2 = d2 / dist2[..., None]
            Xp[:, :-2, :] -= (w[:, :-2] * dlam2)[..., None] * dhat2
            Xp[:, 2:, :] += (w[:, 2:] * dlam2)[..., None] * dhat2

        # --- contacts -----------------------------------------------------
        N = self.N
        Pf = Xp.reshape(N, 3)
        Xf_prev = Xprev.reshape(N, 3)
        ia, ib = self._detect_pairs(Pf, self.h)
        self.last_pair_count = int(ia.size)
        n_pairs = int(ia.size)

        corr_sph = np.zeros((N, 3))
        contacted = np.zeros(N, dtype=bool)
        pair_ncorr = np.zeros(n_pairs)
        dxa = np.zeros((n_pairs, 3))
        dxb = np.zeros((n_pairs, 3))
        wf = self.wf

        for _ in range(CONTACT_ITERS):
            # sphere: project penetrating nodes outward along the radial normal
            dn = np.sqrt((Pf * Pf).sum(1))
            mask = self.nonroot & (dn < self.R_c)
            idx = np.flatnonzero(mask)
            if idx.size:
                nvec = Pf[idx] / np.maximum(dn[idx], 1.0e-12)[:, None]
                # Cap the per-substep outward correction so deep initial
                # penetrations resolve over several substeps without shocks.
                c = np.minimum(self.R_c - dn[idx], MAX_CORR)[:, None]
                Pf[idx] += nvec * c
                corr_sph[idx] += nvec * c
                contacted[idx] = True
            # strand-strand: mass-weighted repulsive projection
            if n_pairs:
                dv = Pf[ib] - Pf[ia]
                dist = np.sqrt((dv * dv).sum(1))
                np.maximum(dist, 1.0e-12, out=dist)
                ov = self.h - dist
                act = ov > 0.0
                if act.any():
                    ia_a = ia[act]
                    ib_a = ib[act]
                    nhat = dv[act] / dist[act][:, None]
                    ovo = ov[act]
                    wa = wf[ia_a]
                    wb = wf[ib_a]
                    ws = wa + wb
                    fa = np.divide(wa, ws, out=np.zeros_like(wa), where=ws > 0.0)
                    fb = np.divide(wb, ws, out=np.zeros_like(wb), where=ws > 0.0)
                    ca = -nhat * (ovo * fa)[:, None]
                    cb = nhat * (ovo * fb)[:, None]
                    Pf[ia_a] += ca
                    Pf[ib_a] += cb
                    pair_ncorr[act] += 0.5 * ovo
                    dxa[act] += ca
                    dxb[act] += cb

        # sphere friction (position-level Coulomb, static = kinetic = mu)
        idx = np.flatnonzero(contacted)
        if idx.size:
            p = Pf[idx]
            dn = np.sqrt((p * p).sum(1))
            nvec = p / np.maximum(dn, 1.0e-12)[:, None]
            vs = np.cross(np.broadcast_to(om, p.shape), p)
            rel = (Pf[idx] - Xf_prev[idx]) - vs * dt
            reln = (rel * nvec).sum(1)
            relt = rel - reln[:, None] * nvec
            rt = np.sqrt((relt * relt).sum(1))
            cn = np.sqrt((corr_sph[idx] * corr_sph[idx]).sum(1))
            fr = np.minimum(1.0, self.mu * cn / np.maximum(rt, 1.0e-12))
            dfric = -relt * fr[:, None]
            Pf[idx] += dfric
            corr_sph[idx] += dfric

        # strand-strand friction
        if n_pairs:
            dv = Pf[ib] - Pf[ia]
            dist = np.sqrt((dv * dv).sum(1))
            np.maximum(dist, 1.0e-12, out=dist)
            nhat = dv / dist[:, None]
            rel = (Pf[ia] - Xf_prev[ia]) - (Pf[ib] - Xf_prev[ib])
            reln = (rel * nhat).sum(1)
            relt = rel - reln[:, None] * nhat
            rt = 0.5 * np.sqrt((relt * relt).sum(1))
            fr = np.minimum(1.0, self.mu * pair_ncorr / np.maximum(rt, 1.0e-12))
            da = -relt * 0.5 * fr[:, None]
            db = relt * 0.5 * fr[:, None]
            ma = (wf[ia] > 0.0)[:, None]
            mb = (wf[ib] > 0.0)[:, None]
            da = da * ma
            db = db * mb
            Pf[ia] += da
            Pf[ib] += db
            dxa += da
            dxb += db

        # --- commit -------------------------------------------------------
        Xp = Pf.reshape(S, Kn, 3)
        Xp[:, 0, :] = roots_world
        Vnew = (Xp - Xprev) / dt
        Vnew[:, 0, :] = np.cross(np.broadcast_to(om, roots_world.shape), roots_world)
        sp = np.sqrt((Vnew * Vnew).sum(-1))
        fac = np.minimum(1.0, VMAX / np.maximum(sp, 1.0e-12))
        Vnew *= fac[..., None]

        if not (np.isfinite(Xp).all() and np.isfinite(Vnew).all()):
            # Safety fallback: freeze this substep deterministically.
            self.reset_count += 1
            self.steps += 1
            return

        Jsph_node = self.mf[:, None] * corr_sph
        Jss_node = np.zeros((N, 3))
        if n_pairs:
            for k in range(3):
                Jss_node[:, k] = np.bincount(ia, weights=self.mf[ia] * dxa[:, k], minlength=N)
                Jss_node[:, k] += np.bincount(ib, weights=self.mf[ib] * dxb[:, k], minlength=N)

        P_new = (self.mf[:, None] * Vnew.reshape(N, 3)).sum(0)
        Jg = self.m_total * self.g * dt
        # Root attachment / pinned-constraint impulse: momentum residual after
        # gravity, drag, and the external sphere contact are accounted for.
        # (Strand-strand impulses are internal; contacts against pinned roots
        # transmit to the sphere and are part of this residual.)
        Jroot = (P_new - P_old) - Jg - Jdrag - Jsph_node.sum(0)

        self.cum_ext += Jg + Jdrag + Jsph_node.sum(0) + Jroot
        self.int_Jg += Jg
        self.int_Jdrag += Jdrag
        self.int_Jsph += Jsph_node.sum(0)
        self.int_Jroot += Jroot
        self.int_Jsph_node += Jsph_node
        self.int_Jss_node += Jss_node

        self.X = Xp
        self.V = Vnew
        self.steps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t_target = float(time)
        if t_target < 0.0:
            t_target = 0.0
        target_steps = int(round(t_target / self.dt_sub))
        while self.steps < target_steps:
            self._substep()

        Xf = self.X.reshape(-1, 3).copy()
        Vf = self.V.reshape(-1, 3).copy()
        N = self.N
        dt_int = t_target - self.t_last
        inv = (1.0 / dt_int) if dt_int > 1.0e-9 else 0.0
        om = self.rot.omega(t_target)

        # ---- friction contact reporting ---------------------------------
        dn = np.sqrt((Xf * Xf).sum(1))
        smask = self.nonroot & (dn < self.R_c + REPORT_MARGIN)
        sidx = np.flatnonzero(smask)
        ia, ib = self._detect_pairs(Xf, self.h + REPORT_MARGIN)
        Cn = int(sidx.size) + int(ia.size)

        normals = np.zeros((Cn, 3))
        fnormal = np.zeros(Cn)
        ftang = np.zeros((Cn, 3))
        vrel = np.zeros((Cn, 3))

        if sidx.size:
            p = Xf[sidx]
            nvec = p / np.maximum(dn[sidx], 1.0e-12)[:, None]
            J = self.int_Jsph_node[sidx]
            jn = (J * nvec).sum(1)
            fnormal[: sidx.size] = np.maximum(jn, 0.0) * inv
            ftang[: sidx.size] = (J - jn[:, None] * nvec) * inv
            vrel[: sidx.size] = Vf[sidx] - np.cross(np.broadcast_to(om, p.shape), p)
            normals[: sidx.size] = nvec
        if ia.size:
            off = sidx.size
            dv = Xf[ia] - Xf[ib]
            dist = np.sqrt((dv * dv).sum(1))
            np.maximum(dist, 1.0e-12, out=dist)
            nvec = dv / dist[:, None]  # points from other (ib) to acting (ia)
            Ja = self.int_Jss_node[ia]
            Jb = self.int_Jss_node[ib]
            jn = 0.5 * ((Ja - Jb) * nvec).sum(1)
            fnormal[off:] = np.maximum(jn, 0.0) * inv
            ftang[off:] = (Ja - ((Ja * nvec).sum(1))[:, None] * nvec) * inv
            vrel[off:] = Vf[ia] - Vf[ib]
            normals[off:] = nvec

        # ---- momentum-budget external force terms ------------------------
        f_ext = np.zeros((1, 4, 3))
        f_ext[0, 0] = self.int_Jg * inv
        f_ext[0, 1] = self.int_Jsph * inv
        f_ext[0, 2] = self.int_Jdrag * inv
        f_ext[0, 3] = self.int_Jroot * inv

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, N)),
            "budget_external_force": f_ext,
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": np.ones(Cn),
            "friction_dynamic_coefficient": np.full(Cn, self.mu),
            "friction_normal": normals,
            "friction_normal_force": fnormal,
            "friction_relative_velocity": vrel,
            "friction_static_coefficient": np.full(Cn, self.mu),
            "friction_tangential_force": ftang,
        }
        scalars = {
            "friction_stick_speed_tolerance": 1.0e-3,
            "reference_force": float(self.ref_force),
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
        }
        frame = StateFrame(
            time=t_target,
            position=Xf,
            velocity=Vf,
            mass=self.mf.copy(),
            attributes={
                "strand_id": self.strand_flat.copy(),
                "root_mask": self.root_flat.copy(),
                "mass_quantity": self.mf.copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()

        # Reset per-interval accumulators.
        self.t_last = t_target
        self.int_Jg[:] = 0.0
        self.int_Jdrag[:] = 0.0
        self.int_Jsph[:] = 0.0
        self.int_Jroot[:] = 0.0
        self.int_Jsph_node[:] = 0.0
        self.int_Jss_node[:] = 0.0
        return frame
