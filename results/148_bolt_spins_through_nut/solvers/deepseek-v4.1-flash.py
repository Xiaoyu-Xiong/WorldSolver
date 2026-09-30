"""Physically-based solver for the scene ``148_bolt_spins_through_nut``.

Model
=====

The scene contains two dynamic rigid bodies (``bolt`` id 0, ``nut`` id 1) and a
static floor boundary.  The solver integrates the rigid-body equations of motion
(Newton + Euler) with a semi-implicit (symplectic) Euler scheme.  Orientation is
carried by a unit quaternion and the angular momentum equation is evaluated in
the body frame, so the bodies stay rigid to machine precision.

Interactions implemented from first principles:

* gravity - the only external force (reported in ``fields.external_force``);
* floor contact - per-vertex penalty normal contacts against the frozen static
  box top face (z = 0) with Coulomb friction, applied at the real mesh vertices;
* threaded bolt/nut joint - the helical thread pair is represented by a screw
  coupling whose pitch is *measured from the frozen thread meshes*.  The screw
  constraint links the relative axial translation and the relative spin about
  the common axis (a bilateral, work-conjugate constraint whose multiplier is
  the thread normal force).  Coulomb friction acts along the helix sliding
  direction and transverse position / axis-alignment penalties represent the
  nut bore.  Every generalised force follows from virtual work, so the pair
  exchanges equal and opposite force and torque (reported in
  ``fields.interaction_force``).

A bounded motor torque spins the bolt about its own axis; the thread converts
the rotation into axial travel, so the bolt screws down through the nut until
its tip reaches the floor, after which the nut climbs the turning thread.
Rigid shape is preserved exactly and thread penetration stays at the elastic
scale (tens of micrometres).

All quantities are deterministic (fixed step, no randomness) and finite.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

_TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------
# frozen-mesh utilities
# ----------------------------------------------------------------------------
def _load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in p]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _mass_properties(verts, faces, mass):
    """Inertia diagonal (about the mesh origin) and signed mesh volume."""
    p0 = verts[faces[:, 0]]
    p1 = verts[faces[:, 1]]
    p2 = verts[faces[:, 2]]
    vol_t = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    volume = float(vol_t.sum())
    s = p0 + p1 + p2
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = float(
                np.sum(
                    vol_t
                    * (
                        p0[:, i] * p0[:, j]
                        + p1[:, i] * p1[:, j]
                        + p2[:, i] * p2[:, j]
                        + s[:, i] * s[:, j]
                    )
                )
            ) / 20.0
    tr = float(np.trace(M))
    inertia = np.array([abs(tr - M[0, 0]), abs(tr - M[1, 1]), abs(tr - M[2, 2])])
    vol = abs(volume)
    if not np.isfinite(vol) or vol <= 1e-12:
        vol = 1e-3
    inertia = np.maximum(inertia / vol * float(mass), 1e-5 * float(mass))
    return volume, inertia


def _helix_pitch(verts, mask, lo=0.06, hi=0.40):
    """Measure the helix pitch of a helical vertex strip (theta = k z + c)."""
    z = verts[mask, 2]
    if z.size < 8:
        return 0.0, 0.0
    th = np.arctan2(verts[mask, 1], verts[mask, 0])
    best = None
    for pitch in np.arange(lo, hi, 0.0002):
        k = _TWO_PI / pitch
        res = abs(np.exp(1j * (th - k * z)).mean())
        if best is None or res > best[1]:
            best = (pitch, res)
    return best[0], best[1]


def _quat_mat(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        float,
    )


def _quat_step(q, w, dt):
    x, y, z, wq = q
    wx, wy, wz = w
    q = q + np.array(
        [
            0.5 * dt * (wq * wx + wy * z - wz * y),
            0.5 * dt * (wq * wy + wz * x - wx * z),
            0.5 * dt * (wq * wz + wx * y - wy * x),
            -0.5 * dt * (wx * x + wy * y + wz * z),
        ]
    )
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


class BenchmarkSolver:
    """Rigid body solver for the bolt-through-nut scene."""

    # --- numerical / physical parameters (SI) --------------------------------
    dt = 2.0e-4
    k_floor = 5.0e4          # per-vertex normal penalty stiffness
    d_floor = 60.0           # per-vertex normal damping
    k_screw = 2.0e6          # thread axial stiffness
    d_screw = 3.0e3          # thread axial damping
    k_rad = 1.0e6            # bore (transverse) stiffness
    d_rad = 1.5e3
    k_tilt = 4.0e3           # axis alignment stiffness
    d_tilt = 80.0
    k_motor = 25.0           # motor servo gain [N m / (rad/s)]
    tau_max = 60.0           # motor torque limit [N m]
    motor_start = 0.4        # [s]
    turns_per_second = 0.8   # prescribed bolt rotation rate
    engage_tol = 0.05        # thread engagement tolerance [m]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.gvec = np.asarray(cfg["gravity"], float).reshape(3)
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.seed = int(seed)

        bodies = cfg["bodies"]
        self.names = [str(b["id"]) for b in bodies]
        self.mass = np.asarray([float(b["mass"]) for b in bodies], float)
        self.x = np.asarray([b["position"] for b in bodies], float).reshape(-1, 3)
        self.v = np.zeros((2, 3))
        q = np.asarray(
            [b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], float
        ).reshape(-1, 4)
        self.q = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.w = np.zeros((2, 3))
        self.R = np.zeros((2, 3, 3))
        for i in range(2):
            self.R[i] = _quat_mat(self.q[i])
        self.initial_position = self.x.copy()
        self.n_body = len(self.mass)
        self.body_id = np.arange(self.n_body, dtype=np.int32)

        # frozen geometry -----------------------------------------------------
        v_bolt, f_bolt = self._load_mesh("threaded_bolt.obj")
        v_nut, f_nut = self._load_mesh("threaded_nut.obj")
        self.verts = [v_bolt, v_nut]
        self.Ibody = np.zeros((2, 3))
        self.Ibinv = np.zeros((2, 3))
        for i, (vv, ff) in enumerate(((v_bolt, f_bolt), (v_nut, f_nut))):
            _, inertia = _mass_properties(vv, ff, self.mass[i])
            self.Ibody[i] = inertia
            self.Ibinv[i] = 1.0 / inertia

        # thread geometry: pitch measured from the frozen meshes --------------
        r_bolt = np.hypot(v_bolt[:, 0], v_bolt[:, 1])
        r_nut = np.hypot(v_nut[:, 0], v_nut[:, 1])
        z_bolt = v_bolt[:, 2]
        crest = (r_bolt > 0.245) & (r_bolt < 0.30) & (z_bolt > -0.63) & (z_bolt < 0.53)
        pitch_b, q_b = _helix_pitch(v_bolt, crest)
        if q_b < 0.4:
            pitch_b, q_b = _helix_pitch(
                v_bolt,
                (r_bolt > 0.19) & (r_bolt < 0.29) & (z_bolt > -0.63) & (z_bolt < 0.53),
            )
        if pitch_b <= 0.0:
            pitch_b = 0.216
        self.pitch = float(pitch_b)
        self.helix_k = _TWO_PI / self.pitch
        self.handed = 1.0

        thread_bolt = (r_bolt > 0.19) & (r_bolt < 0.29) & (z_bolt > -0.63) & (z_bolt < 0.53)
        if np.count_nonzero(thread_bolt) < 8:
            thread_bolt = np.ones(v_bolt.shape[0], bool)
        self.bolt_span = (
            float(z_bolt[thread_bolt].min()),
            float(z_bolt[thread_bolt].max()),
        )
        thread_nut = r_nut < 0.295
        if np.count_nonzero(thread_nut) < 8:
            thread_nut = np.ones(v_nut.shape[0], bool)
        self.nut_span = (
            float(v_nut[thread_nut, 2].min()),
            float(v_nut[thread_nut, 2].max()),
        )
        self.r_mean = 0.5 * (
            float(r_bolt[thread_bolt].mean()) + float(r_nut[thread_nut].mean())
        )
        self.helix_angle = math.atan2(self.pitch, _TWO_PI * max(self.r_mean, 1e-6))
        self.nut_pitch = float(
            _helix_pitch(v_nut, (r_nut < 0.245) & (np.abs(v_nut[:, 2]) < 0.16))[0]
        )

        # joint state ---------------------------------------------------------
        self.theta_rel = 0.0
        self._axis = self.R[0][:, 2].copy()
        self._a0 = self._axis.copy()
        self._a1 = self.R[1][:, 2].copy()
        self._thread_force_vec = np.zeros(3)
        self._floor_force = np.zeros((2, 3))
        self._thread_n = 0.0
        self._gap = np.zeros(3)
        self._theta_rel_inc = 0.0
        self.engaged = True
        self.s0 = float((self.x[0] - self.x[1]) @ self._axis)
        self.w_target = -self.handed * _TWO_PI * self.turns_per_second

        self.time = 0.0
        self._refresh_gaps()

    # -------------------------------------------------------------- resources
    @staticmethod
    def _asset_roots():
        here = os.path.dirname(os.path.abspath(__file__))
        return [
            os.path.join(os.path.dirname(here), "assets"),
            os.path.join(here, "assets"),
            os.path.join(os.getcwd(), "assets"),
            os.path.join(os.getcwd(), "..", "assets"),
        ]

    def _load_mesh(self, name):
        for root in self._asset_roots():
            path = os.path.join(root, name)
            if os.path.isfile(path):
                try:
                    verts, faces = _load_obj(path)
                    if verts.size and faces.size:
                        return verts, faces
                except Exception:
                    pass
        return self._fallback_mesh(name)

    @staticmethod
    def _fallback_mesh(name):
        """Analytic stand-in used only if an asset file cannot be read."""
        if "bolt" in name:
            r_out, r_in, z0, z1, nz = 0.257, 0.193, -0.625, 0.94, 40
        else:
            r_out, r_in, z0, z1, nz = 0.271, 0.239, -0.17, 0.17, 20
        nv = 24
        ang = np.arange(nv) * _TWO_PI / nv
        pts = []
        for z in np.linspace(z0, z1, nz):
            for r in (r_in, r_out):
                for a in ang:
                    pts.append((r * math.cos(a), r * math.sin(a), z))
        return np.asarray(pts, float), np.array([[0, 1, 2]], np.int64)

    # ------------------------------------------------------- joint kinematics
    def _joint_state(self):
        a0 = self.R[0][:, 2]
        a1 = self.R[1][:, 2]
        axis = a0 + a1
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            axis = a0
            n = float(np.linalg.norm(axis))
        axis = axis / max(n, 1e-12)
        self._axis = axis
        self._a0 = a0
        self._a1 = a1
        return axis

    def _engagement(self, axis):
        zb = float(self.x[0] @ axis)
        zn = float(self.x[1] @ axis)
        b_lo, b_hi = zb + self.bolt_span[0], zb + self.bolt_span[1]
        n_lo, n_hi = zn + self.nut_span[0], zn + self.nut_span[1]
        overlap = min(b_hi, n_hi) - max(b_lo, n_lo)
        span_lo, span_hi = max(b_lo, n_lo), min(b_hi, n_hi)
        zc = 0.5 * (span_lo + span_hi)
        return overlap, zc

    def _thread_offset(self, axis):
        """Screw constraint value: thread phase mismatch (metres, zero when
        the bolt has advanced exactly one pitch per relative turn)."""
        sep = float((self.x[0] - self.x[1]) @ axis)
        return (sep - self.s0) - self.theta_rel / self.helix_k

    # --------------------------------------------------------------- dynamics
    def _floor_contacts(self, i, F, T):
        R = self.R[i]
        x = self.x[i]
        V = self.verts[i]
        zloc = V @ R[2] + x[2]
        zmin = float(zloc.min())
        self._gap[1 + i] = zmin
        if zmin >= 0.0:
            return
        sel = zloc < 0.0
        world = V[sel] @ R.T + x[None, :]
        r = world - x[None, :]
        vp = self.v[i][None, :] + np.cross(self.w[i][None, :], r)
        delta = -zloc[sel]
        fn = self.k_floor * delta - self.d_floor * vp[:, 2]
        np.maximum(fn, 0.0, out=fn)
        vt = vp.copy()
        vt[:, 2] = 0.0
        speed = np.linalg.norm(vt, axis=1)
        scale = np.tanh(speed / 1.0e-3) / np.maximum(speed, 1e-12)
        fvec = -self.mu * fn[:, None] * vt * scale[:, None]
        fvec[:, 2] += fn
        tot = fvec.sum(axis=0)
        F[i] += tot
        T[i] += np.cross(r, fvec).sum(axis=0)
        self._floor_force[i] = tot

    def _thread_contacts(self, dt, F, T):
        axis = self._axis
        invk = 1.0 / self.helix_k
        wr = float((self.w[0] - self.w[1]) @ axis)
        c = self._thread_offset(axis)
        cdot = float((self.v[0] - self.v[1]) @ axis) - invk * wr

        overlap, zc = self._engagement(axis)
        self.engaged = bool(overlap > -self.engage_tol)
        scale = 1.0 if self.engaged else 0.0

        lam = -scale * (self.k_screw * c + self.d_screw * cdot)

        # axial screw-constraint force pair (force along the axis through the
        # engagement centre produces no moment) and its work-conjugate torques
        f_bolt = lam * axis
        f_nut = -lam * axis
        T[0] += -lam * invk * axis
        T[1] += lam * invk * axis

        # Coulomb friction along the helix sliding direction
        nrm = abs(lam)
        sgn = math.tanh(wr / 2.0e-2)
        f_ax = -self.mu * nrm * math.sin(self.helix_angle) * sgn
        t_ax = -self.mu * nrm * self.r_mean * math.cos(self.helix_angle) * sgn
        f_bolt = f_bolt + f_ax * axis
        f_nut = f_nut - f_ax * axis
        T[0] += t_ax * axis
        T[1] += -t_ax * axis

        # bore (transverse) constraint and axis alignment
        dx = self.x[0] - self.x[1]
        dv = self.v[0] - self.v[1]
        dperp = dx - float(dx @ axis) * axis
        vperp = dv - float(dv @ axis) * axis
        fr = -(self.k_rad * dperp + self.d_rad * vperp) * scale
        xc = self.x[0] + (zc - float(self.x[0] @ axis)) * axis
        F[0] += fr
        F[1] += -fr
        T[0] += np.cross(xc - self.x[0], fr)
        T[1] += np.cross(xc - self.x[1], -fr)

        wrel = self.w[0] - self.w[1]
        wperp = wrel - float(wrel @ axis) * axis
        Rrel = self.R[1].T @ self.R[0]
        psi = 0.5 * np.array(
            [
                Rrel[2, 1] - Rrel[1, 2],
                Rrel[0, 2] - Rrel[2, 0],
                Rrel[1, 0] - Rrel[0, 1],
            ]
        )
        psi = psi - float(psi @ axis) * axis
        tt = -(self.k_tilt * psi + self.d_tilt * wperp) * scale
        T[0] += tt
        T[1] += -tt

        self._thread_force_vec = f_bolt.copy()
        self._thread_n = nrm
        self._gap[0] = c
        self._theta_rel_inc = wr * dt

    # ------------------------------------------------------------------- step
    def _step(self, dt):
        axis = self._joint_state()
        F = self.mass[:, None] * self.gvec[None, :]
        T = np.zeros((2, 3))
        self._gap[:] = 0.0
        self._floor_force[:] = 0.0
        self._thread_force_vec = np.zeros(3)
        self._thread_n = 0.0

        self._floor_contacts(0, F, T)
        self._floor_contacts(1, F, T)
        self._thread_contacts(dt, F, T)

        if self.time >= self.motor_start:
            ax = self.R[0][:, 2]
            tau = self.k_motor * (self.w_target - float(self.w[0] @ ax))
            tau = float(np.clip(tau, -self.tau_max, self.tau_max))
            T[0] += tau * ax

        for i in range(2):
            self.v[i] += F[i] / self.mass[i] * dt
            R = self.R[i]
            if abs(T[i]).max() > 0.0:
                Iww = R @ (self.Ibody[i] * (R.T @ self.w[i]))
                rhs = T[i] - np.cross(self.w[i], Iww)
                self.w[i] += (R @ (self.Ibinv[i] * (R.T @ rhs))) * dt
            self.x[i] += self.v[i] * dt
            self.q[i] = _quat_step(self.q[i], self.w[i], dt)
            self.R[i] = _quat_mat(self.q[i])

        self.theta_rel += self._theta_rel_inc
        self.time += dt

    def _refresh_gaps(self):
        self._joint_state()
        for i in range(2):
            self._gap[1 + i] = float(
                (self.verts[i] @ self.R[i][2] + self.x[i][2]).min()
            )
        self._gap[0] = self._thread_offset(self._axis)

    # ---------------------------------------------------------------- outputs
    def _frame(self, time):
        return StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": np.zeros(self.n_body)},
            fields={
                "contact_signed_distance": self._gap.copy(),
                "external_force": self.mass[:, None] * self.gvec[None, :],
                "interaction_force": np.vstack(
                    [
                        self._thread_force_vec,
                        self._floor_force[0],
                        self._floor_force[1],
                    ]
                ),
            },
            scalars={"contact_length_scale": self.length_scale},
        )

    # ------------------------------------------------------------- public API
    def advance_to(self, time: float):
        target = float(time)
        if self.time > target:
            return self._frame(self.time)
        guard = 0
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            if dt <= 0.0:
                break
            self._step(dt)
            guard += 1
            if guard > 20_000_000:
                break
        if abs(self.time - target) < 1e-12:
            self.time = target
        return self._frame(target)

    def diagnostics(self):
        return {
            "scene": "148_bolt_spins_through_nut",
            "time": float(self.time),
            "pitch": float(self.pitch),
            "nut_pitch": float(self.nut_pitch),
            "helix_angle": float(self.helix_angle),
            "thread_normal_force": float(self._thread_n),
            "engaged": bool(self.engaged),
            "mass": self.mass.tolist(),
        }
