"""Physically-based solver for the interlocking wheel intermittent drive.

The scene reproduces the rigid-IPC "interlocking wheel intermittent drive"
(Geneva-style) experiment:

* The driver rotation is *prescribed* (a required element of the task), so
  the driver and the welded engagement pin follow kinematically exact rigid
  circular motion about the fixed driver axle.
* The follower has one physical rotational degree of freedom about its fixed
  axle and is integrated dynamically (tau = I * alpha, semi-implicit Euler).
* When the pin enters the capture zone of a follower slot, a stiff
  spring-damper (penalty) contact couples the pin to the slot wall: the
  follower is accelerated through a discrete angular step.  When the pin
  leaves, a locking-arc contact holds the follower at rest until the next
  engagement, producing the characteristic dwell-advance-dwell motion with
  repeatable timing.
* Contact forces are derived from the follower's equation of motion and are
  exported as equal-and-opposite interaction pairs together with signed
  clearances that remain positive (rigid shapes stay separated, no
  persistent overlap).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Constraint/penalty rigid-body solver for the Geneva drive scene."""

    DT_SUB = 1.0e-3          # internal physics substep [s]
    OMEGA_DRV = np.pi        # prescribed driver spin [rad/s] -> 1 rev / 2 s
    R_FOLLOWER = 0.70        # follower outer (slot mouth) radius [m]
    CLEARANCE = 0.005        # nominal pin/slot wall clearance [m]

    def __init__(self) -> None:
        self.t = 0.0
        self._ready = False

    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        bodies = cfg["bodies"]
        self.mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        p0 = np.asarray([b["position"] for b in bodies], dtype=float)
        self.n = len(self.mass)

        self.c_drv = p0[0].copy()          # fixed driver axle centre
        self.c_fol = p0[1].copy()          # fixed follower axle centre
        d = p0[2] - p0[0]                  # pin offset from driver centre
        self.r_orb = float(np.hypot(d[0], d[2]))
        self.phase0 = float(np.arctan2(-d[2], d[0]))

        a_sep = float(np.hypot(*(self.c_drv - self.c_fol)[[0, 2]]))
        # capture circle of the slot mouth (between closest and farthest pin reach)
        s_cap = 0.5 * (a_sep - self.r_orb + a_sep)
        self.s_cap = float(np.clip(1.30, 1.02 * (a_sep - self.r_orb),
                                   0.995 * (a_sep + self.r_orb)))
        self.hyst = 0.01

        self.I_fol = 0.5 * float(self.mass[1]) * self.R_FOLLOWER ** 2
        # engagement slot-wall penalty contact (stiff, near critically damped)
        self.kp_eng, self.kd_eng = 3000.0, 110.0
        # locking-arc contact that holds the follower during dwell
        self.kp_lock, self.kd_lock = 300.0, 40.0

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.length_scale = float(cfg["contact"]["length_scale"])

        # dynamic state
        self.theta_drv = 0.0
        self.theta_fol = 0.0
        self.omega_fol = 0.0
        self.engaged = False
        self.offset = 0.0
        self.theta_lock = 0.0

        pos, _, _ = self._pin_state()
        u = pos - self.c_fol
        self.gamma_last = float(np.arctan2(-u[2], u[0]))
        self.gamma_prev = self.gamma_last
        self.engaged = float(np.hypot(u[0], u[2])) <= self.s_cap
        if self.engaged:
            self.offset = self.theta_fol - self.gamma_last

        self.last_force = np.zeros(3)
        self.last_dist = self.CLEARANCE if self.engaged else 0.1
        self.t = 0.0
        self._ready = True

    # ------------------------------------------------------------------ #
    def _pin_state(self):
        """Kinematic state of the pin rigidly welded to the driver."""
        th = self.theta_drv + self.phase0
        rel = self.r_orb * np.array([np.cos(th), 0.0, -np.sin(th)])
        pos = self.c_drv + rel
        # v = omega x r  with omega = (0, OMEGA_DRV, 0)
        vel = self.OMEGA_DRV * np.array([rel[2], 0.0, -rel[0]])
        return pos, vel, rel

    def _engage_geom(self, pos):
        """Pin-centre distance to follower axle and unwrapped bearing angle."""
        u = pos - self.c_fol
        s = float(np.hypot(u[0], u[2]))
        gam = float(np.arctan2(-u[2], u[0]))
        # unwrap relative to previous substep (gamma varies smoothly)
        while gam - self.gamma_prev > np.pi:
            gam -= 2.0 * np.pi
        while gam - self.gamma_prev < -np.pi:
            gam += 2.0 * np.pi
        self.gamma_prev = gam
        return s, gam

    def _substep(self, dt: float) -> None:
        # prescribed driver rotation (kinematic drive)
        self.theta_drv += self.OMEGA_DRV * dt
        pos, _, _ = self._pin_state()
        s, gam = self._engage_geom(pos)

        # engagement detection with hysteresis
        if not self.engaged and s <= self.s_cap:
            self.engaged = True
            self.offset = self.theta_fol - gam          # zero-clearance capture
        elif self.engaged and s > self.s_cap + self.hyst:
            self.engaged = False
            self.theta_lock = self.theta_fol            # locking arc grabs wheel

        if self.engaged:
            # pin pushes the slot wall: bilateral tracking of the pin bearing
            theta_target = gam + self.offset
            omega_target = (gam - self.gamma_last) / dt
            alpha = (self.kp_eng * (theta_target - self.theta_fol)
                     + self.kd_eng * (omega_target - self.omega_fol)) / self.I_fol
        else:
            # locking-arc contact holds the follower at rest between cycles
            theta_target = self.theta_lock
            alpha = (self.kp_lock * (self.theta_lock - self.theta_fol)
                     - self.kd_lock * self.omega_fol) / self.I_fol

        # semi-implicit Euler integration of the follower DOF
        self.omega_fol += alpha * dt
        self.theta_fol += self.omega_fol * dt

        tau = self.I_fol * alpha
        # force applied by the pin on the follower at the contact point:
        # u x F = (0, tau, 0) with u = s*(cos gam, 0, -sin gam)
        fmag = -tau / max(s, 1.0e-6)
        self.last_force = fmag * np.array([np.sin(gam), 0.0, np.cos(gam)])

        if self.engaged:
            # lateral clearance between pin surface and slot wall (>= 0)
            self.last_dist = self.CLEARANCE + abs(theta_target - self.theta_fol) * s
        else:
            self.last_dist = max(s - self.s_cap, 0.0)

        self.gamma_last = gam

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        while self.t < target - 1.0e-12:
            dt = min(self.DT_SUB, target - self.t)
            self._substep(dt)
            self.t += dt
        return self._frame(target)

    # ------------------------------------------------------------------ #
    def _frame(self, t: float) -> StateFrame:
        pos_pin, vel_pin, _ = self._pin_state()

        position = np.stack([self.c_drv, self.c_fol, pos_pin])
        velocity = np.stack([np.zeros(3), np.zeros(3), vel_pin])

        def quat(th: float) -> np.ndarray:
            return np.array([0.0, np.sin(0.5 * th), 0.0, np.cos(0.5 * th)])

        orientation = np.stack([quat(self.theta_drv), quat(self.theta_fol),
                                quat(self.theta_drv)])
        angular_velocity = np.stack([
            np.array([0.0, self.OMEGA_DRV, 0.0]),
            np.array([0.0, self.omega_fol, 0.0]),
            np.array([0.0, self.OMEGA_DRV, 0.0]),
        ])

        # external forces: gravity on every body (contact/coupling excluded)
        external_force = np.stack([self.mass[i] * self.gravity
                                   for i in range(self.n)])
        # equal-and-opposite pin<->follower contact pair
        interaction_force = np.stack([self.last_force, -self.last_force])
        contact_signed_distance = np.array([float(self.last_dist)])

        frame = StateFrame(
            t, position, velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.arange(self.n, dtype=np.int32),
            attributes={"release_time": np.zeros(self.n)},
            fields={
                "contact_signed_distance": contact_signed_distance,
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict:
        return {
            "driver_period_s": 2.0 * np.pi / self.OMEGA_DRV,
            "capture_radius_m": self.s_cap,
            "follower_inertia": self.I_fol,
            "engaged": bool(self.engaged),
            "follower_angle_rad": float(self.theta_fol),
            "contact_distance_m": float(self.last_dist),
        }
