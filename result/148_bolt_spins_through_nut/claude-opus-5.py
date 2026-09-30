"""Rigid-body solver: bolt spins through a nut (helical thread contact).

Physical model
--------------
Two rigid bodies (threaded bolt, matching nut) are integrated with a
semi-implicit (symplectic) Euler scheme in full 6-DOF.  The frozen assets are
ideal single-start helical threads, so the bolt/nut thread interface is modelled
analytically instead of by mesh-mesh proximity queries:

* The thread of the nut is a helical groove of pitch ``P`` whose centre surface
  in the nut frame satisfies ``z = (P/2pi) * theta`` (mod ``P``).  The thread of
  the bolt is the same helix rigidly attached to the bolt.  Engagement of the
  two helices therefore depends on a single scalar residual

      psi = dz_axial - (P/2pi) * (dphi_bolt - dphi_nut)

  measured from the initial (interlocked, overlap-free) configuration.  The
  angular coordinate cancels exactly, which is the classical screw constraint.
* Each engaged turn/angle sample carries two one-sided flank contacts (lower and
  upper flank) with signed axial gaps ``c +- psi_wrapped``, where ``c`` is the
  half of the axial clearance between tooth thickness and groove width.  Each
  contact uses a penalty normal force with normal damping and regularised
  Coulomb friction driven by the true relative surface velocity, so the axial
  travel of the bolt is produced by real contact forces (it is never scripted).
* Radial (bore) contacts at two axial stations model the crest/root radial
  clearance and keep the axes aligned while resisting tilt.
* Additional one-sided contacts: bolt head underside vs. nut top face, bolt tip
  vs. floor, nut bottom vs. floor.

Actuation: the spin of the bolt about its own axis is prescribed (motor drive,
smoothly ramped); all other five degrees of freedom of the bolt are fully
dynamic.  The nut is held by an external fixture modelled as a stiff PD
force/torque, reported as part of ``fields.external_force``.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

TWO_PI = 2.0 * np.pi

# --- geometry fitted from the frozen assets (meters) -------------------------
PITCH = 0.2165          # helical lead of both threads
R_PITCH = 0.247         # pitch-circle radius of the thread interface
NUT_HALF_H = 0.17       # nut half height
AX_CLEAR = 0.005        # half of the axial flank clearance
RAD_CLEAR = 0.006       # radial crest/root clearance
BOLT_TIP_Z = -0.625     # shank tip in bolt frame
BOLT_HEAD_Z = 0.70      # head underside in bolt frame
BOLT_SHANK_TOP = 0.70   # top of the threaded shank in bolt frame
ENG_RAMP = 0.04         # smoothing length of axial thread engagement
FLOOR_TOP = 0.0         # top face of the static floor box

N_ANG = 8               # angular thread samples
N_TURN = 2              # turns sampled per angle
N_FLANK_SLOT = N_ANG * N_TURN * 2
N_BORE = 2
N_CONTACT = N_FLANK_SLOT + N_BORE + 3   # + head/face, bolt-floor, nut-floor

DT = 2.5e-4             # integration step
SPIN_RATE = 2.8         # prescribed |omega| about the bolt axis (rad/s)
SPIN_RAMP = 0.5         # smooth ramp-up time (s)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_err(q_target: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Small-rotation vector taking ``q`` onto ``q_target`` (world frame)."""
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
    d = _quat_mul(q_target, q_inv)
    if d[3] < 0.0:
        d = -d
    return 2.0 * d[:3]


def _wrap(value: float, period: float) -> float:
    return value - period * np.round(value / period)


class BenchmarkSolver:
    """Self-contained rigid-body solver for the bolt/nut screw interaction."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact = cfg["contact"]
        self.mu = float(contact["friction"])
        self.restitution = float(contact["restitution"])
        self.length_scale = float(contact["length_scale"])
        self.duration = float(cfg["duration"])

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)          # (2,3)
        self.q = np.array(frame0.orientation, dtype=float)       # (2,4) xyzw
        self.v = np.zeros((2, 3))
        self.w = np.zeros((2, 3))
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.x0 = self.x.copy()
        self.q0 = self.q.copy()

        # principal inertia (uniform-density approximations of the assets)
        bolt_len, bolt_r = 1.565, 0.25
        nut_h, nut_r = 0.34, 0.45
        mb, mn = float(self.mass[0]), float(self.mass[1])
        self.inertia = np.array([
            [mb * (3.0 * bolt_r ** 2 + bolt_len ** 2) / 12.0,
             mb * (3.0 * bolt_r ** 2 + bolt_len ** 2) / 12.0,
             0.5 * mb * bolt_r ** 2],
            [mn * (3.0 * nut_r ** 2 + nut_h ** 2) / 12.0,
             mn * (3.0 * nut_r ** 2 + nut_h ** 2) / 12.0,
             0.5 * mn * nut_r ** 2],
        ])
        self.inertia_inv = 1.0 / self.inertia

        # thread contact sample layout (fixed slots, nut-local)
        ang = np.arange(N_ANG) * (TWO_PI / N_ANG)
        base = np.mod((PITCH / TWO_PI) * ang, PITCH)
        zloc = np.concatenate([base - PITCH, base])            # (N_ANG*N_TURN,)
        theta = np.tile(ang, N_TURN)
        self.slot_theta = theta
        self.slot_zloc = zloc
        # axial engagement fraction inside the nut height (static part)
        self.slot_weight_nut = np.clip(
            (NUT_HALF_H - np.abs(zloc)) / ENG_RAMP, 0.0, 1.0)
        self.slot_local = np.stack(
            [R_PITCH * np.cos(theta), R_PITCH * np.sin(theta), zloc], axis=1)
        # flanks: +1 lower flank (pushes bolt along +axis), -1 upper flank
        self.flank_sign = np.concatenate(
            [np.ones(N_ANG * N_TURN), -np.ones(N_ANG * N_TURN)])
        self.flank_local = np.vstack([self.slot_local, self.slot_local])
        self.flank_zloc = np.concatenate([zloc, zloc])
        self.flank_weight_nut = np.concatenate(
            [self.slot_weight_nut, self.slot_weight_nut])

        # penalty parameters
        self.k_n = 1.0e5
        self.c_n = 2.0e2
        self.k_r = 6.0e4
        self.c_r = 1.5e2
        self.k_face = 8.0e4
        self.c_face = 4.0e2
        self.k_floor = 1.0e5
        self.c_floor = 6.0e2
        # nut fixture (external hold)
        self.k_h, self.c_h = 2.0e4, 4.4e2
        self.k_th, self.c_th = 3.0e3, 6.0e1

        # helical engagement reference (initial state is interlocked, no overlap)
        a0 = _quat_to_mat(self.q[1])[:, 2]
        self.z_ax0 = float((self.x[0] - self.x[1]) @ a0)
        self.phi = np.zeros(2)

        self.time = 0.0
        self.dt = DT
        self._steps = 0
        self._max_pen = 0.0
        self._motor_torque = 0.0
        self._diag_psi = 0.0

    # --------------------------------------------------------------- dynamics
    def _spin_target(self, t: float) -> float:
        s = np.clip(t / SPIN_RAMP, 0.0, 1.0)
        ramp = s * s * (3.0 - 2.0 * s)      # smoothstep
        return -SPIN_RATE * ramp            # negative z-spin drives descent

    def _forces(self):
        x, q, v, w = self.x, self.q, self.v, self.w
        Rb = _quat_to_mat(q[0])
        Rn = _quat_to_mat(q[1])
        axis = Rn[:, 2]
        axis_b = Rb[:, 2]

        ext = np.zeros((2, 3))
        ext[0] = self.mass[0] * self.gravity
        ext[1] = self.mass[1] * self.gravity
        # external fixture holding the nut in place
        ext[1] = ext[1] + self.k_h * (self.x0[1] - x[1]) - self.c_h * v[1]
        torque_hold = (self.k_th * _quat_err(self.q0[1], q[1])
                       - self.c_th * w[1])

        force = ext.copy()
        torque = np.zeros((2, 3))
        torque[1] += torque_hold

        inter = np.zeros((N_CONTACT, 3))
        dist = np.zeros(N_CONTACT)

        d = x[0] - x[1]
        z_ax = float(d @ axis)
        psi = (z_ax - self.z_ax0) - (PITCH / TWO_PI) * (
            (self.phi[0] - 0.0) - (self.phi[1] - 0.0))
        e = _wrap(psi, PITCH)
        self._diag_psi = e

        # ---- helical flank contacts -------------------------------------
        gap = AX_CLEAR - self.flank_sign * e
        # engagement: slot must lie inside the nut AND be covered by the
        # threaded part of the bolt shank (both smoothly ramped)
        lo = self.flank_zloc - (z_ax + BOLT_TIP_Z)
        hi = (z_ax + BOLT_SHANK_TOP) - self.flank_zloc
        w_bolt = (np.clip(lo / ENG_RAMP, 0.0, 1.0)
                  * np.clip(hi / ENG_RAMP, 0.0, 1.0))
        weight = self.flank_weight_nut * w_bolt
        cp = x[1] + self.flank_local @ Rn.T         # world contact points
        # normal opposes the sign of the flank penetration
        n = -self.flank_sign[:, None] * axis[None, :]
        r_b = cp - x[0]
        r_n = cp - x[1]
        v_rel = ((v[0] + np.cross(w[0], r_b)) - (v[1] + np.cross(w[1], r_n)))
        vn = np.einsum("ij,ij->i", v_rel, n)
        pen = np.maximum(0.0, -gap)
        active = pen > 0.0
        fn = np.where(active,
                      weight * np.maximum(
                          0.0, self.k_n * pen - self.c_n * vn),
                      0.0)
        vt = v_rel - vn[:, None] * n
        vt_mag = np.linalg.norm(vt, axis=1)
        ft = -(self.mu * fn / (vt_mag + 0.02))[:, None] * vt
        f_slot = fn[:, None] * n + ft
        dist[:N_FLANK_SLOT] = gap
        inter[:N_FLANK_SLOT] = f_slot
        tot = f_slot.sum(axis=0)
        force[0] += tot
        force[1] -= tot
        torque[0] += np.cross(r_b, f_slot).sum(axis=0)
        torque[1] -= np.cross(r_n, f_slot).sum(axis=0)
        if active.any():
            self._max_pen = max(self._max_pen, float(pen.max()))

        idx = N_FLANK_SLOT
        # ---- radial bore contacts at two axial stations -----------------
        denom = float(axis_b @ axis)
        if abs(denom) < 1e-6:
            denom = 1e-6
        for zs in (-0.13, 0.13):
            s = (zs - z_ax) / denom
            pt = x[0] + axis_b * s
            rel = pt - x[1]
            off = rel - (rel @ axis) * axis
            mag = float(np.linalg.norm(off))
            gap_r = RAD_CLEAR - mag
            dist[idx] = gap_r
            if gap_r < 0.0 and mag > 1e-9:
                nr = -off / mag
                r_b1 = pt - x[0]
                r_n1 = pt - x[1]
                vr = ((v[0] + np.cross(w[0], r_b1))
                      - (v[1] + np.cross(w[1], r_n1)))
                vnr = float(vr @ nr)
                fnr = max(0.0, self.k_r * (-gap_r) - self.c_r * vnr)
                vtr = vr - vnr * nr
                ftr = -(self.mu * fnr / (np.linalg.norm(vtr) + 0.02)) * vtr
                fr = fnr * nr + ftr
                inter[idx] = fr
                force[0] += fr
                force[1] -= fr
                torque[0] += np.cross(r_b1, fr)
                torque[1] -= np.cross(r_n1, fr)
            idx += 1

        # ---- bolt head underside vs nut top face ------------------------
        pt = x[0] + Rb @ np.array([0.0, 0.0, BOLT_HEAD_Z])
        gap_f = float((pt - x[1]) @ axis) - NUT_HALF_H
        dist[idx] = gap_f
        if gap_f < 0.0:
            r_b1 = pt - x[0]
            r_n1 = pt - x[1]
            vr = ((v[0] + np.cross(w[0], r_b1))
                  - (v[1] + np.cross(w[1], r_n1)))
            vnf = float(vr @ axis)
            fnf = max(0.0, self.k_face * (-gap_f) - self.c_face * vnf)
            vtf = vr - vnf * axis
            ftf = -(self.mu * fnf / (np.linalg.norm(vtf) + 0.02)) * vtf
            ff = fnf * axis + ftf
            inter[idx] = ff
            force[0] += ff
            force[1] -= ff
            torque[0] += np.cross(r_b1, ff)
            torque[1] -= np.cross(r_n1, ff)
        idx += 1

        # ---- floor contacts (bolt tip, nut bottom) ----------------------
        up = np.array([0.0, 0.0, 1.0])
        for bi, local in ((0, np.array([0.0, 0.0, BOLT_TIP_Z])),
                          (1, np.array([0.0, 0.0, -NUT_HALF_H]))):
            R = Rb if bi == 0 else Rn
            pt = x[bi] + R @ local
            gap_g = float(pt[2] - FLOOR_TOP)
            dist[idx] = gap_g
            if gap_g < 0.0:
                r_b1 = pt - x[bi]
                vr = v[bi] + np.cross(w[bi], r_b1)
                vng = float(vr @ up)
                fng = max(0.0, self.k_floor * (-gap_g) - self.c_floor * vng)
                vtg = vr - vng * up
                ftg = -(self.mu * fng / (np.linalg.norm(vtg) + 0.02)) * vtg
                fg = fng * up + ftg
                inter[idx] = fg
                force[bi] += fg
                torque[bi] += np.cross(r_b1, fg)
                self._max_pen = max(self._max_pen, -gap_g)
            idx += 1

        return force, torque, ext, inter, dist

    def _step(self, dt: float) -> None:
        force, torque, _, _, _ = self._forces()

        # linear
        self.v += (force / self.mass[:, None]) * dt
        self.x += self.v * dt

        # angular (body-frame Euler equations)
        for i in range(2):
            R = _quat_to_mat(self.q[i])
            wb = R.T @ self.w[i]
            tb = R.T @ torque[i]
            Ib = self.inertia[i]
            dwb = self.inertia_inv[i] * (tb - np.cross(wb, Ib * wb))
            wb = wb + dwb * dt
            self.w[i] = R @ wb

        # prescribed bolt spin about its own axis (motor drive)
        axis_b = _quat_to_mat(self.q[0])[:, 2]
        w_spin = float(self.w[0] @ axis_b)
        w_goal = self._spin_target(self.time + dt)
        self._motor_torque = self.inertia[0, 2] * (w_goal - w_spin) / dt
        self.w[0] = self.w[0] + (w_goal - w_spin) * axis_b

        # orientation update (world-frame angular velocity)
        for i in range(2):
            om = self.w[i]
            dq = 0.5 * _quat_mul(np.array([om[0], om[1], om[2], 0.0]), self.q[i])
            qn = self.q[i] + dq * dt
            nrm = float(np.linalg.norm(qn))
            self.q[i] = qn / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
            self.phi[i] += float(self.w[i] @ _quat_to_mat(self.q[i])[:, 2]) * dt

        self.time += dt
        self._steps += 1

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
        self.time = target

        _, _, ext, inter, dist = self._forces()
        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": np.asarray(dist, dtype=float),
                "external_force": np.asarray(ext, dtype=float),
                "interaction_force": np.asarray(inter, dtype=float),
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self._steps),
            "dt": float(self.dt),
            "contacts": int(N_CONTACT),
            "pitch": float(PITCH),
            "max_penetration": float(self._max_pen),
            "thread_phase_residual": float(self._diag_psi),
            "motor_torque": float(self._motor_torque),
            "axial_travel": float(self.x0[0, 2] - self.x[0, 2]),
        }
