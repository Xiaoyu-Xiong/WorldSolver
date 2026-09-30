"""Coiling reversal of a naturally curved elastic rod (scene 092).

Self-contained discrete-Cosserat-rod solver (no external physics engine):

* state: node positions + node material frames (quaternions, xyzw);
* damped semi-implicit Euler time stepping (quasi-static regime);
* inextensibility via stiff axial springs; bending/torsion of a rod with
  nonzero natural curvature (natural curvature direction is the material
  1-axis; the inlet material orientation is held fixed by the injector);
* elastic position forces from finite differences of the discrete
  bending/twist energy; twist torques in closed form;
* hard substrate contact with a sticking deposited condition;
* continuous injection: the first edge rest length grows at the injection
  speed and new nodes are inserted below the fixed inlet, carrying the
  parallel-transported inlet material frame.

The coiling direction of the deposit is set purely by rod mechanics (the
interplay of natural curvature and twist accumulated between the fixed
inlet frame and the stuck deposition front), so a coiling reversal can
emerge from continued injection without any scripted motion.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

DT = 1.0e-3            # physical substep (s)
NODE_SPACING = 0.0125  # rest length of injected edges (m)
STICK_DWELL = 0.02     # contact dwell before a node sticks (s)
FD_H = 1.0e-6          # finite-difference step (m)
V_CAP = 2.0            # velocity cap (m/s)
DAMP_TIME = 0.05       # translational damping time (s)
TWIST_RELAX = 0.02     # overdamped twist relaxation time (s)


# ---------------- quaternion / vector helpers (xyzw, vectorized) ----------------

def _unit2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def _quat_mul(p, q):
    px, py, pz, pw = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * py - pz * pz,
    ], axis=-1)


def _quat_conj(q):
    r = q.copy()
    r[:, :3] *= -1.0
    return r


def _quat_normalize(q):
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def _quat_rotate(q, v):
    qv = q[:, :3]
    w = q[:, 3]
    t = 2.0 * np.cross(qv, v)
    return v + w[:, None] * t + np.cross(qv, t)


def _min_rotation(a, b):
    """Quaternion rotating unit vectors a onto b, per row."""
    a = _unit2(a)
    b = _unit2(b)
    c = np.sum(a * b, axis=1)
    ax = np.cross(a, b)
    s = np.sqrt(np.clip(0.5 * (1.0 + c), 0.0, 1.0))
    xyz = np.zeros_like(a)
    ok = s > 1e-9
    if np.any(ok):
        xyz[ok] = ax[ok] / (2.0 * s[ok][:, None])
    bad = ~ok
    if np.any(bad):
        aa = a[bad]
        helper = np.zeros_like(aa)
        j = np.argmin(np.abs(aa), axis=1)
        helper[np.arange(len(j)), j] = 1.0
        xyz[bad] = _unit2(np.cross(aa, helper))
        s[bad] = 0.0
    q = np.concatenate([xyz, s[:, None]], axis=1)
    return _quat_normalize(q)


def _axis_angle_quat(axis, angle):
    u = _unit2(axis)
    h = 0.5 * angle
    return np.concatenate([u * np.sin(h)[:, None], np.cos(h)[:, None]], axis=1)


def _quat_to_mats(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m1 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1)
    m2 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=1)
    m3 = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)
    return m1, m2, m3


# ---------------- solver ----------------

class BenchmarkSolver:
    def __init__(self):
        self.initialized = False

    # ---- setup ----

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        md = scene.metadata()
        rod = md["rod"]
        inj = md["injector"]

        self.g = np.asarray(md["gravity"], dtype=float)
        self.gnorm = float(np.linalg.norm(self.g))
        self.r = float(rod["radius"])
        rho = float(rod["density"])
        E = float(rod["young_modulus"])
        nu = float(rod["poisson_ratio"])
        self.k0 = float(rod["natural_curvature"])
        self.natural_radius = float(rod["natural_radius"])
        self.lg = float(rod["gravito_bending_length"])
        self.L_avail = float(rod["total_available_length"])
        I = math.pi * self.r ** 4 / 4.0
        self.B = E * I
        self.C = self.B / (1.0 + nu)
        self.rhoA = rho * math.pi * self.r ** 2

        self.v_inj = float(inj["injection_speed"])
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.inlet_axis = _unit2(np.asarray(inj["axis"], dtype=float).reshape(1, 3))[0]
        self.t_inj_end = float(inj["end_time"])
        # inlet material frame: m1 = (1,0,0), m3 = inlet axis -> rotation pi about x
        self.inlet_q = np.array([1.0, 0.0, 0.0, 0.0])

        # rod state
        self.x = np.array(frame0.position, dtype=float)
        n = len(self.x)
        self.rest = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self.L0 = float(self.rest.sum())
        self.v = np.zeros((n, 3))
        q = np.empty((n, 4))
        q[0] = self.inlet_q
        sh = self._edge_dirs(self.x)
        for i in range(1, n):
            qr = _min_rotation(sh[i - 1][None, :], sh[i][None, :])[0]
            q[i] = _quat_mul(qr[None, :], q[i - 1][None, :])[0]
        self.q = _quat_normalize(q)

        dep = np.asarray(frame0.attributes["is_initially_deposited"], dtype=float) > 0.5
        self.stuck = dep.copy()
        self.stuck_pos = self.x.copy()
        self.stuck_q = self.q.copy()
        self.is_init = dep.copy()
        self.timers = np.zeros(n)
        self.omega = np.zeros((n, 3))
        self._remass()

        self.k_s = 100.0
        self.mu_tw = 1.0 / (TWIST_RELAX * self.C / NODE_SPACING)
        self.dt = DT
        self.t = 0.0
        self.frozen = False
        self.scalars0 = {k: float(v) for k, v in frame0.scalars.items()}
        self._reset_accum()
        self._save_good()
        self.initialized = True

    def _edge_dirs(self, x):
        d = x[1:] - x[:-1]
        l = np.linalg.norm(d, axis=1)
        s = d / np.maximum(l, 1e-12)[:, None]
        return np.vstack([s, s[-1]])

    def _remass(self):
        rest = self.rest
        n = len(rest) + 1
        w = np.empty(n)
        w[0] = rest[0] / 2.0
        w[-1] = rest[-1] / 2.0
        if n > 2:
            w[1:-1] = 0.5 * (rest[:-1] + rest[1:])
        self.mass = self.rhoA * w

    def _reset_accum(self):
        self.acc_contact = np.zeros(3)
        self.acc_cpos = np.zeros(3)
        self.acc_cw = 0.0
        self.acc_inj = np.zeros(3)
        self.acc_couple = np.zeros(3)
        self.window = 0.0

    def _save_good(self):
        self._good = (
            self.x.copy(), self.v.copy(), self.q.copy(), self.omega.copy(),
            self.rest.copy(), self.stuck.copy(), self.is_init.copy(),
            self.stuck_pos.copy(), self.stuck_q.copy(), self.timers.copy(),
            self.t,
        )

    def _restore_good(self):
        (self.x, self.v, self.q, self.omega, self.rest, self.stuck,
         self.is_init, self.stuck_pos, self.stuck_q, self.timers, self.t) = self._good
        for arr in (self.x, self.v, self.q, self.omega, self.rest, self.stuck,
                    self.is_init, self.stuck_pos, self.stuck_q, self.timers):
            arr[...] = np.asarray(arr).copy()
        self._remass()

    # ---- elastic energies ----

    def _eb(self, xl, xm, xr, m1m):
        d1 = xm - xl
        d2 = xr - xm
        l1 = np.linalg.norm(d1, axis=1)
        l2 = np.linalg.norm(d2, axis=1)
        tl = d1 / np.maximum(l1, 1e-12)[:, None]
        tr = d2 / np.maximum(l2, 1e-12)[:, None]
        w = 0.5 * (l1 + l2)
        kap = np.cross(tl, tr) / np.maximum(w, 1e-9)[:, None]
        diff = kap - self.k0 * m1m
        return 0.5 * self.B * w * np.sum(diff * diff, axis=1)

    def _et(self, xl, xm, xr, m1l, m1r):
        d1 = xm - xl
        d2 = xr - xm
        l1 = np.linalg.norm(d1, axis=1)
        sl = d1 / np.maximum(l1, 1e-12)[:, None]
        sr = d2 / np.maximum(np.linalg.norm(d2, axis=1), 1e-12)[:, None]
        qpt = _min_rotation(sl, sr)
        mpt = _quat_rotate(qpt, m1l)
        cr = np.cross(mpt, m1r)
        psi = np.arctan2(np.sum(cr * sr, axis=1), np.sum(mpt * m1r, axis=1))
        return 0.5 * self.C / np.maximum(l1, 1e-9) * psi * psi

    def _fd_forces(self, x, m1):
        n = len(x)
        F = np.zeros((n, 3))
        if n < 3:
            return F
        xl = x[:-2]
        xm = x[1:-1]
        xr = x[2:]
        m1m = m1[1:-1]
        m1l = m1[:-2]
        m1r = m1[1:-1]
        for c in range(3):
            for sgn in (1.0, -1.0):
                d = np.zeros(3)
                d[c] = sgn * FD_H
                acc = np.zeros(n)
                acc[1:n - 1] += self._eb(xl, xm + d, xr, m1m)
                acc[0:n - 2] += self._eb(xl + d, xm, xr, m1m)
                acc[2:n] += self._eb(xl, xm, xr + d, m1m)
                acc[0:n - 2] += self._et(xl + d, xm, xr, m1l, m1r)
                acc[1:n - 1] += self._et(xl, xm + d, xr, m1l, m1r)
                acc[2:n] += self._et(xl, xm, xr + d, m1l, m1r)
                F[:, c] -= sgn * acc / (2.0 * FD_H)
        return F

    def _torques(self, x, q, se):
        n = len(x)
        m1, m2, m3 = _quat_to_mats(q)
        le = np.linalg.norm(x[1:] - x[:-1], axis=1)
        sen = np.vstack([se, se[-1]])
        # twist angle between consecutive node frames
        qpt = _min_rotation(sen[:-1], sen[1:])
        mpt = _quat_rotate(qpt, m1[:-1])
        cr = np.cross(mpt, m1[1:])
        psi = np.arctan2(np.sum(cr * sen[1:], axis=1), np.sum(mpt * m1[1:], axis=1))
        a = self.C * psi / np.maximum(le, 1e-9)
        tau = np.zeros(n)
        tau[:n - 1] += a
        tau[1:] -= a
        # bending-twist coupling at interior nodes
        if n >= 3:
            wj = 0.5 * (le[:-1] + le[1:])
            kap = np.cross(se[:-1], se[1:]) / np.maximum(wj, 1e-9)[:, None]
            diff = kap - self.k0 * m1[1:-1]
            tau[1:n - 1] += self.B * wj * self.k0 * np.sum(diff * m2[1:-1], axis=1)
        return tau

    # ---- injection ----

    def _insert_node(self):
        x0 = self.x[0]
        d01 = self.x[1] - x0
        L01 = float(np.linalg.norm(d01))
        if L01 < 1e-9:
            dhat = self.inlet_axis.copy()
            L01 = 1e-9
        else:
            dhat = d01 / L01
        frac = NODE_SPACING if L01 > 1.5 * NODE_SPACING else max(0.25 * L01, 1e-4)
        p = x0 + dhat * frac
        qn = _quat_mul(_min_rotation(self.inlet_axis[None, :], dhat[None, :]),
                       self.inlet_q[None, :])[0]
        qn = _quat_normalize(qn[None, :])[0]
        r0 = self.rest[0]
        self.x = np.insert(self.x, 1, p, axis=0)
        self.v = np.insert(self.v, 1, self.v[1], axis=0)
        self.q = np.insert(self.q, 1, qn, axis=0)
        self.omega = np.insert(self.omega, 1, 0.0, axis=0)
        self.stuck = np.insert(self.stuck, 1, False)
        self.is_init = np.insert(self.is_init, 1, False)
        self.timers = np.insert(self.timers, 1, 0.0)
        self.stuck_pos = np.insert(self.stuck_pos, 1, p, axis=0)
        self.stuck_q = np.insert(self.stuck_q, 1, qn, axis=0)
        self.rest = np.insert(self.rest, 0, frac)
        self.rest[1] = max(r0 - frac, 1e-4)
        self._remass()

    # ---- time stepping ----

    def _substep(self, dt):
        if self.frozen:
            return
        self._save_good()
        x = self.x
        q = self.q
        n = len(x)
        m1, m2, m3 = _quat_to_mats(q)

        d = x[1:] - x[:-1]
        le = np.linalg.norm(d, axis=1)
        se = d / np.maximum(le, 1e-12)[:, None]

        # external + elastic + axial forces
        F = self.mass[:, None] * self.g[None, :]
        Fs = (self.k_s * (le - self.rest))[:, None] * se
        F[:-1] += Fs
        F[1:] -= Fs
        F += self._fd_forces(x, m1)
        tau = self._torques(x, q, se)

        v_free = self.v + (F / self.mass[:, None]) * dt
        v_free *= max(0.0, 1.0 - dt / DAMP_TIME)
        sp = np.linalg.norm(v_free, axis=1)
        v_free *= np.minimum(1.0, V_CAP / np.maximum(sp, 1e-12))[:, None]

        x_prev = x.copy()
        q_before = q.copy()
        x_new = x + v_free * dt

        # overdamped twist update about local tangents
        sen = np.vstack([se, se[-1]])
        dtheta = np.clip(self.mu_tw * tau * dt, -0.05, 0.05)
        dtheta[0] = 0.0
        dtheta[self.stuck] = 0.0
        q_new = _quat_normalize(_quat_mul(_axis_angle_quat(sen, dtheta), q))

        # constraints: inlet pin, stuck nodes, substrate
        x_new[0] = self.outlet
        x_new[self.stuck] = self.stuck_pos[self.stuck]
        low = x_new[:, 2] < self.r
        low[0] = False
        low[self.stuck] = False
        x_new[low, 2] = self.r

        # frame-tangent alignment (free nodes only)
        d2 = x_new[1:] - x_new[:-1]
        s2 = d2 / np.maximum(np.linalg.norm(d2, axis=1), 1e-12)[:, None]
        s2n = np.vstack([s2, s2[-1]])
        _, _, m3n = _quat_to_mats(q_new)
        qr = _min_rotation(m3n, s2n)
        ident = np.zeros_like(qr)
        ident[:, 3] = 1.0
        pin = self.stuck.copy()
        pin[0] = True
        qr[pin] = ident[pin]
        q_new = _quat_normalize(_quat_mul(qr, q_new))
        q_new[0] = self.inlet_q
        q_new[self.stuck] = self.stuck_q[self.stuck]

        v_new = (x_new - x_prev) / dt
        sp2 = np.linalg.norm(v_new, axis=1)
        v_new *= np.minimum(1.0, V_CAP / np.maximum(sp2, 1e-12))[:, None]

        # constraint impulse bookkeeping (external budget forces)
        J = self.mass[:, None] * (v_new - v_free)
        touch = x_new[:, 2] <= self.r + 2.0e-5
        touch[0] = False
        contact = self.stuck | touch
        if np.any(contact):
            Jc = J[contact]
            self.acc_contact += Jc.sum(axis=0)
            wJ = np.linalg.norm(Jc, axis=1)
            self.acc_cpos += (wJ[:, None] * x_new[contact]).sum(axis=0)
            self.acc_cw += float(wJ.sum())
        self.acc_inj += J[0]
        self.acc_couple += (-tau[0]) * m3[0] * dt
        self.window += dt

        # sticking deposition
        touch2 = touch & ~self.stuck
        self.timers[touch2] += dt
        self.timers[~touch2] = 0.0
        newly = self.timers >= STICK_DWELL
        if np.any(newly):
            self.stuck[newly] = True
            self.stuck_pos[newly] = x_new[newly]
            self.stuck_q[newly] = q_new[newly]
            self.timers[newly] = 0.0

        # angular velocity of node frames over the substep
        rel = _quat_mul(q_new, _quat_conj(q_before))
        rel[rel[:, 3] < 0.0] *= -1.0
        w_ = np.clip(rel[:, 3], 0.0, 1.0)
        vec = rel[:, :3]
        s_ = np.linalg.norm(vec, axis=1)
        ang = 2.0 * np.arctan2(s_, w_)
        ax_ = vec / np.maximum(s_, 1e-12)[:, None]
        omega_new = ax_ * (ang / max(dt, 1e-12))[:, None]

        # commit
        self.x = x_new
        self.v = v_new
        self.q = q_new
        self.omega = omega_new
        self.t += dt

        # injection (feed grows first edge, split when long enough)
        try:
            Ltot = float(self.rest.sum())
            if self.t < self.t_inj_end and Ltot < self.L_avail - 1e-9:
                add = min(self.v_inj * dt, self.L_avail - Ltot)
                self.rest[0] += add
                guard = 0
                while self.rest[0] >= 2.0 * NODE_SPACING - 1e-12 and guard < 4:
                    self._insert_node()
                    guard += 1
        except Exception:
            self._restore_good()
            self.frozen = True
            return

        if not (np.isfinite(self.x).all() and np.isfinite(self.q).all()
                and np.isfinite(self.v).all() and np.isfinite(self.rest).all()
                and np.abs(self.x).max() < 10.0):
            self._restore_good()
            self.frozen = True

    # ---- output ----

    def _build_frame(self, t_s):
        n = len(self.x)
        x = self.x.copy()
        v = self.v.copy()
        q = _quat_normalize(self.q.copy())
        edge = np.linalg.norm(x[1:] - x[:-1], axis=1)
        arc = np.concatenate([[0.0], np.cumsum(edge)])
        Ltot = float(self.rest.sum())
        minj = float(self.mass.sum())
        com = (self.mass[:, None] * x).sum(axis=0) / max(minj, 1e-12)
        g_force = minj * self.g
        window = self.window
        if window > 1e-12:
            contact_force = self.acc_contact / window
            inj_force = self.acc_inj / window
            inj_couple = self.acc_couple / window
        else:
            contact_force = np.zeros(3)
            inj_force = np.zeros(3)
            inj_couple = np.zeros(3)
        if self.acc_cw > 1e-15:
            contact_point = self.acc_cpos / self.acc_cw
        elif np.any(self.stuck):
            contact_point = self.x[self.stuck].mean(axis=0)
        else:
            contact_point = self.outlet.copy()

        budget_force = np.stack([g_force, contact_force, inj_force], axis=0)[:, None, :]
        budget_pos = np.stack([com, contact_point, self.outlet], axis=0)[:, None, :]
        budget_couple = np.zeros((3, 1, 3))
        budget_couple[2, 0, :] = inj_couple

        L_target = self.L0 + max(0.0, min(self.v_inj * t_s, self.L_avail - self.L0))
        constraint_value = np.array([x[0, 0], x[0, 1], x[0, 2], Ltot])
        constraint_target = np.array([self.outlet[0], self.outlet[1],
                                      self.outlet[2], L_target])
        cum_source = np.array([(Ltot - self.L0) * self.rhoA])

        inj_ctrl = np.zeros(n)
        inj_ctrl[0] = 1.0
        attributes = {
            "mass_quantity": self.mass.copy(),
            "arc_coordinate": arc,
            "is_injector_controlled": inj_ctrl,
            "is_initially_deposited": self.is_init.astype(float),
            "is_sticking": self.stuck.astype(float),
            "rod_radius": np.full(n, self.r),
            "natural_curvature": np.full(n, self.k0),
            "natural_curvature_direction": np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)),
        }
        fields = {
            "budget_external_couple": budget_couple,
            "budget_external_force": budget_force,
            "budget_force_position": budget_pos,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": cum_source,
        }
        scalars = dict(self.scalars0)
        scalars["constraint_scale"] = float(self.natural_radius) if self.natural_radius > 0.0 else 0.0478
        scalars["reference_force"] = float(self.rhoA * max(self.gnorm, 1e-6) * self.lg)
        scalars["reference_torque"] = float(self.B * self.k0)

        frame = StateFrame(
            time=float(t_s),
            position=x,
            velocity=v,
            orientation=q,
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        self._reset_accum()
        return frame

    # ---- contract API ----

    def advance_to(self, time):
        if not self.initialized:
            raise RuntimeError("solver not initialized")
        target = float(time)
        if target < self.t:
            target = self.t
        while self.t < target - 1e-9 and not self.frozen:
            step = min(self.dt, target - self.t)
            self._substep(step)
        return self._build_frame(max(target, self.t))

    def diagnostics(self):
        return {
            "time": float(self.t),
            "num_nodes": int(len(self.x)),
            "num_stuck": int(np.count_nonzero(self.stuck)),
            "deployed_length": float(self.rest.sum()),
            "injected_mass": float((self.rest.sum() - self.L0) * self.rhoA),
            "frozen_fallback": bool(self.frozen),
        }
