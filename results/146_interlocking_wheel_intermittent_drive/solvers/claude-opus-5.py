"""Interlocking wheel intermittent drive (Geneva mechanism) solver.

Self-contained rigid-body model; no external physics engine is used.

Physical model
=============
Both wheels turn on frozen, fixed axles whose axis is the world ``y`` axis, so
the mechanism lives in the world ``x-z`` plane (up axis ``z``).  Angles below
are measured as ``atan2(z, x)`` about the corresponding axle, positive
counter-clockwise in that view.

Bodies (state bindings from ``scene_geometry.yaml``):

* ``body_id 0`` ``driver``    -- rigid wheel pinned at ``(-0.85, 0, 1)``.
  Its rotation is *prescribed* (``omega_d`` constant), i.e. an ideal motor
  holds the crank speed.  Its centre of mass therefore never translates.
* ``body_id 1`` ``follower``  -- slotted rigid wheel pinned at ``(0.85, 0, 1)``.
  Its angle is the single unknown degree of freedom of the system.  It is
  driven *only* by contact with the pin plus bearing friction, so it must
  dwell between engagements.
* ``body_id 2`` ``drive_pin`` -- sphere of radius ``0.14`` rigidly mounted on
  the driver at the frozen crank radius ``a = |p_pin - p_driver| = 0.65`` m.

Geneva registration
-------------------
With the frozen axle spacing ``c = 1.70`` m and crank radius ``a = 0.65`` m,

    a / c = 0.38235 = sin(22.5 deg)  (to four digits),

so the frozen dimensions are those of a *registered* external Geneva drive
with ``n = 360 / (2 * 22.5) = 8`` slots:

* the slot mouth lies on the tangency circle ``r_out = sqrt(c^2 - a^2)``, where
  the crank velocity is parallel to the slot axis, so the pin enters and leaves
  each slot with (essentially) zero follower angular velocity -- no impact
  spike, no residual spin;
* the pin bottoms out at ``r = c - a`` at mid engagement;
* engagement lasts ``180 * (1 - 2 / n) = 135 deg`` of crank rotation and the
  follower is indexed by exactly ``360 / n = 45 deg``; the remaining
  ``225 deg`` is dwell.

Numerics
--------
One dynamic generalized coordinate (the follower angle) is integrated with a
semi-implicit / symplectic Euler scheme at a fixed micro step ``2e-4 s``,
which is ~30x below the penalty-contact stability limit
``2 / sqrt(k_theta / I)``.  Pin-vs-slot-wall contact uses a linear penalty
spring plus a damper calibrated from the frozen restitution, and a regularized
Coulomb friction law with the frozen friction coefficient.  The channel half
width is ``pin_radius + contact.length_scale``, so the frozen contact length
scale is the running clearance and steady-state penetration stays at the
1e-4 m level (two orders below that clearance): the rigid shapes stay
separated.

All reported forces follow from Newton's laws for the resulting motion, so the
exported external / interaction force pairs are consistent with the exported
positions and velocities.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_MICRO_DT = 2.0e-4


def _quat_about_y(view_angle: float) -> tuple[float, float, float, float]:
    """xyzw quaternion whose rotation advances ``atan2(z, x)`` by view_angle."""
    half = -0.5 * float(view_angle)
    return (0.0, math.sin(half), 0.0, math.cos(half))


class BenchmarkSolver:
    """Deterministic Geneva-drive rigid-body solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.seed = int(seed)

        bodies = list(cfg["bodies"])
        self.n_bodies = len(bodies)
        self.body_id = np.arange(self.n_bodies, dtype=np.int32)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.p0 = np.asarray([b["position"] for b in bodies], dtype=float)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        contact = dict(cfg["contact"])
        self.mu = float(contact["friction"])
        self.restitution = float(contact["restitution"])
        self.length_scale = float(contact["length_scale"])

        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])
        self.micro_dt = _MICRO_DT

        # ---- frozen planar geometry (x-z plane, rotation axis = y) ----------
        self.driver_c = np.array([self.p0[0][0], self.p0[0][2]], dtype=float)
        self.follower_c = np.array([self.p0[1][0], self.p0[1][2]], dtype=float)
        self.driver_y = float(self.p0[0][1])
        self.follower_y = float(self.p0[1][1])
        self.pin_y = float(self.p0[2][1])
        pin_off = np.array([self.p0[2][0], self.p0[2][2]]) - self.driver_c
        self.crank = float(math.hypot(pin_off[0], pin_off[1]))          # a
        self.pin_phase = float(math.atan2(pin_off[1], pin_off[0]))      # 0 rad
        self.centre_dist = float(np.linalg.norm(self.follower_c - self.driver_c))
        self.pin_radius = 0.14
        self.driver_rim = 0.64        # frozen driver mesh outer radius
        self.follower_rim = 0.70      # frozen follower mesh outer radius
        self.floor_top = 0.0          # frozen floor slab top face

        # ---- registered Geneva slot layout ---------------------------------
        ratio = min(max(self.crank / self.centre_dist, 1e-6), 0.999999)
        self.n_slots = int(round(math.pi / math.asin(ratio)))
        self.n_slots = max(3, min(self.n_slots, 12))
        self.slot_pitch = 2.0 * math.pi / self.n_slots
        self.tangency = math.sqrt(max(self.centre_dist**2 - self.crank**2, 1e-9))
        self.slot_r_out = self.tangency
        self.slot_r_in = max(self.centre_dist - self.crank - 0.05, 0.05)
        self.slot_half_w = self.pin_radius + self.length_scale
        self.n_walls = 2 * self.n_slots

        # At t = 0 the pin sits on the line joining the axles, i.e. exactly on a
        # slot axis of the ``k * slot_pitch`` family, at the slot bottom: the
        # frozen identity orientations already have both wheels aligned on an
        # engagement slot.
        pin_bearing0 = math.atan2(
            self.p0[2][2] - self.follower_c[1], self.p0[2][0] - self.follower_c[0]
        )
        k0 = round(pin_bearing0 / self.slot_pitch)
        self.slot_phase = pin_bearing0 - k0 * self.slot_pitch
        self.slot_angles = tuple(
            self.slot_phase + i * self.slot_pitch for i in range(self.n_slots)
        )

        # ---- inertias, prescribed drive, bearing friction -------------------
        self.I_driver = 0.5 * float(self.mass[0]) * self.driver_rim**2
        self.I_follower = 0.5 * float(self.mass[1]) * self.follower_rim**2
        self.omega_driver = math.pi     # prescribed crank rate: 2 s / revolution
        self.bearing_visc = 0.05        # N*m*s  viscous bearing drag
        self.bearing_coulomb = 0.15     # N*m    dry bearing / detent torque

        # ---- penalty contact parameters -------------------------------------
        self.k_n = 4.0e4
        m_eff = max(self.I_follower / max(self.slot_r_out**2, 1e-6), 1e-3)
        e = min(max(self.restitution, 1e-3), 0.999)
        log_e = math.log(e)
        zeta = -log_e / math.sqrt(math.pi * math.pi + log_e * log_e)
        self.c_n = 2.0 * zeta * math.sqrt(self.k_n * m_eff)
        self.v_eps = 0.02               # friction regularization velocity

        # ---- integration state ----------------------------------------------
        self.t = 0.0
        self.alpha_f = 0.0              # follower angle (view convention)
        self.omega_f = 0.0
        self._contact_force = np.zeros(3)     # on the pin, world frame
        self._contact_torque = 0.0            # on the follower, about +view
        self._max_penetration = 0.0
        self._max_omega_f = 0.0
        self._steps = 0

    # ------------------------------------------------------------- kinematics
    def _drive_rate(self, t: float) -> float:
        """Prescribed crank rate with a short smooth motor spin-up.

        The ramp lets the mechanism start from rest in the frozen aligned
        configuration ("just before the driver enters an engagement slot")
        instead of being kicked impulsively at t = 0.
        """
        ramp_t = 0.25
        if t >= ramp_t:
            return self.omega_driver
        s = max(0.0, t) / ramp_t
        return self.omega_driver * (s * s * (3.0 - 2.0 * s))

    def _pin_kinematics(self, theta_d: float, omega_d: float):
        ang = self.pin_phase + theta_d
        c, s = math.cos(ang), math.sin(ang)
        pos = np.array(
            [self.driver_c[0] + self.crank * c, self.driver_c[1] + self.crank * s]
        )
        vel = np.array([-self.crank * omega_d * s, self.crank * omega_d * c])
        acc = np.array(
            [-self.crank * omega_d * omega_d * c, -self.crank * omega_d * omega_d * s]
        )
        return pos, vel, acc

    # ------------------------------------------------------------ pin/slot set
    def _slot_gaps(self, pin_pos, pin_vel, alpha_f, omega_f):
        """Signed pin-vs-slot-wall distances plus the resulting contact loads.

        Returns ``(gaps, force_on_pin_xz, torque_on_follower)`` where ``gaps``
        holds ``2 * n_slots`` signed distances (negative = penetration) ordered
        slot-major, ``[-wall, +wall]`` per slot.
        """
        d = pin_pos - self.follower_c
        gaps = []
        f_pin = np.zeros(2)
        tau_f = 0.0
        rp = self.pin_radius
        for i in range(self.n_slots):
            axis_ang = self.slot_angles[i] + alpha_f
            ca, sa = math.cos(axis_ang), math.sin(axis_ang)
            e = np.array([ca, sa])                 # along the slot, outward
            n = np.array([-sa, ca])                # slot lateral normal
            u = float(d @ e)
            v = float(d @ n)
            du = max(self.slot_r_in - u, u - self.slot_r_out, 0.0)
            for s in (-1.0, 1.0):
                # gap from the pin surface to the wall plane at lateral s * W
                gp = (self.slot_half_w - s * v) - rp
                if du > 0.0:
                    gap = math.hypot(max(gp, 0.0), du) if gp > 0.0 else du
                else:
                    gap = gp
                gaps.append(gap)
                if du > 0.0 or gp >= 0.0:
                    continue
                # ---- active penalty contact on this wall -------------------
                depth = -gp
                # relative velocity of pin material point w.r.t. follower
                v_fol = omega_f * np.array([-d[1], d[0]])
                v_rel = pin_vel - v_fol
                vn = float(v_rel @ n) * s          # closing rate along +s
                fn = self.k_n * depth + self.c_n * vn
                if fn <= 0.0:
                    continue
                vt = float(v_rel @ e)
                ft = -self.mu * fn * vt / (abs(vt) + self.v_eps)
                force = -s * fn * n + ft * e       # on the pin
                f_pin = f_pin + force
                fr = -force                        # on the follower
                tau_f += d[0] * fr[1] - d[1] * fr[0]
        return gaps, f_pin, tau_f

    # ------------------------------------------------------------ time stepper
    def _micro_step(self, dt: float) -> None:
        """One semi-implicit Euler step of the single follower dof."""
        omega_d = self._drive_rate(self.t)
        pin_pos, pin_vel, _ = self._pin_kinematics(self.theta_d, omega_d)
        _, f_pin, tau_c = self._slot_gaps(
            pin_pos, pin_vel, self.alpha_f, self.omega_f
        )

        tau = tau_c - self.bearing_visc * self.omega_f
        w = self.omega_f
        if abs(w) < 1.0e-4 and abs(tau) <= self.bearing_coulomb:
            w_new = 0.0                            # static bearing friction
        else:
            if abs(w) < 1.0e-9:
                dry = -self.bearing_coulomb * (1.0 if tau > 0.0 else -1.0)
            else:
                dry = -self.bearing_coulomb * (1.0 if w > 0.0 else -1.0)
            w_new = w + dt * (tau + dry) / self.I_follower
            if w != 0.0 and w_new * w < 0.0 and abs(tau) <= self.bearing_coulomb:
                w_new = 0.0                        # friction cannot reverse it
        self.omega_f = w_new
        self.alpha_f += dt * self.omega_f

        # prescribed driver: midpoint rate for second-order angle accuracy
        omega_mid = self._drive_rate(self.t + 0.5 * dt)
        self.theta_d += dt * omega_mid
        self.t += dt

        self._contact_force = f_pin
        self._contact_torque = tau_c
        self._steps += 1
        self._max_omega_f = max(self._max_omega_f, abs(self.omega_f))
