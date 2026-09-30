"""Physically-based solver for the friction ball loop (Kaufman et al. Fig 1(c)).

Model
-----
Single rigid sphere (m=1 kg, r=0.24 m, I=2/5 m r^2) under gravity, in
impulse-based contact with:

* ``loop_track``: exact torus SDF.  loop.stl is a torus revolved about the
  y-axis; ring centerline = circle of radius R=1.0 in the x-z plane centered
  at C=(0,0,1.2), tube radius t=0.2.  Signed distance for point p:
      q = sqrt(x^2 + (z-1.2)^2);  d = sqrt((q-1)^2 + y^2)
      s_track = d - t - r_ball
  The ball center rides the inner rail at q-distance rho = R - t - r = 0.56
  from C, contacting the tube's inner surface all the way around.
* ``ground``: plane, top at z=0, s_ground = z - r_ball.

Contact resolution each substep (sequential impulses, a few iterations):
  - positional projection out of penetration,
  - inelastic normal impulse (restitution 0),
  - Coulomb friction at the contact point: full stick (rolling) impulse
    capped by mu * j_n with mu = 0.18 from the config,
  - rolling resistance: f = mu_r * N opposing the tangential center-of-mass
    motion while in rolling contact (rubber ball on metal; mu_r = 0.016).

The rolling resistance dissipates ~ (5/7)*mu_r*oint v^2 dtheta of energy per
pass.  With v0=5.5 m/s (u0 = v^2/2 = 15.1 J/kg) the loop-top margin on the
first pass stays positive (ball remains on the track), while the second pass
arrives at the top with insufficient centripetal demand satisfied
(N -> 0 just before the top) and the ball detaches and falls -- the target
phenomenon of the scene.

All required StateFrame fields are exported per sample.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# ---- frozen scene constants -------------------------------------------------
R_RING = 1.0        # torus major radius (m)
T_TUBE = 0.2        # torus tube radius (m)
C_LOOP = (0.0, 0.0, 1.2)   # ring center (world)
R_BALL = 0.24
MASS = 1.0
INERTIA = 0.4 * MASS * R_BALL * R_BALL   # solid sphere
MU_SLIDE = 0.18     # config friction_coefficient (Coulomb cap)
MU_ROLL = 0.016     # rolling-resistance coefficient (rubber on metal)
RAIL_RHO = R_RING - T_TUBE - R_BALL      # 0.56 ball-center rail radius

H_SUB = 5.0e-4      # physics substep (s)
CONTACT_MARGIN = 2.0e-4
ITER = 4


def _track_sdf(x, y, z):
    """Signed distance from ball *center* to torus surface minus ball radius."""
    qx = math.hypot(x, z - C_LOOP[2])          # distance from y-axis (ring plane radius)
    qx = max(qx, 1e-12)
    d = math.hypot(qx - R_RING, y)             # distance from ring centerline
    s = d - T_TUBE - R_BALL
    # outward unit normal (gradient of torus sdf)
    w = (qx - R_RING) / d if d > 1e-12 else 0.0
    nx = w * (x / qx)
    ny = (y / d) if d > 1e-12 else 0.0
    nz = w * ((z - C_LOOP[2]) / qx)
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nrm > 1e-12:
        nx /= nrm; ny /= nrm; nz /= nrm
    else:
        nx, ny, nz = 0.0, 0.0, 1.0
    return s, (nx, ny, nz)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        g = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = (float(g[0]), float(g[1]), float(g[2]))
        self.mu = float(cfg.get("friction_coefficient", MU_SLIDE))

        frame0 = scene.initial_state()
        p = np.asarray(frame0.position, dtype=float).reshape(-1, 3)[0]
        v = np.asarray(frame0.velocity, dtype=float).reshape(-1, 3)[0]
        q = np.asarray(frame0.orientation, dtype=float).reshape(-1, 4)[0]
        w = np.asarray(frame0.angular_velocity, dtype=float).reshape(-1, 3)[0]
        self.p = (float(p[0]), float(p[1]), float(p[2]))
        self.v = (float(v[0]), float(v[1]), float(v[2]))
        self.q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.w = (float(w[0]), float(w[1]), float(w[2]))

        self.t = 0.0
        # diagnostics bookkeeping
        self.theta_prev = math.atan2(self.p[0], C_LOOP[2] - self.p[2])
        self.theta_total = 0.0
        self.laps = 0
        self.detach_time = None
        self.was_attached = True
        self.min_track_gap = 1e9
        self._last = None  # cached last-sample diagnostics

    # ------------------------------------------------------------------ step
    def _substep(self, h: float) -> None:
        gx, gy, gz = self.gravity
        x, y, z = self.p
        vx, vy, vz = self.v
        wx, wy, wz = self.w

        # gravity (only external force)
        vx += gx * h; vy += gy * h; vz += gz * h

        # integrate position
        x += vx * h; y += vy * h; z += vz * h

        jn_track = 0.0
        jn_ground = 0.0
        s_t = 1.0
        s_g = 1.0
        n_t = (0.0, 0.0, 1.0)

        for _ in range(ITER):
            # ---- torus (loop track) ----
            s_t, n_t = _track_sdf(x, y, z)
            if s_t < CONTACT_MARGIN:
                if s_t < 0.0:
                    x -= n_t[0] * s_t; y -= n_t[1] * s_t; z -= n_t[2] * s_t
                nx, ny, nz = n_t
                # contact point offset from center
                rcx, rcy, rcz = -R_BALL * nx, -R_BALL * ny, -R_BALL * nz
                # contact point velocity
                cvx = vx + (wy * rcz - wz * rcy)
                cvy = vy + (wz * rcx - wx * rcz)
                cvz = vz + (wx * rcy - wy * rcx)
                vn = cvx * nx + cvy * ny + cvz * nz
                if vn < 0.0:
                    jn = -(1.0 + 0.0) * MASS * vn   # restitution 0
                    vx += (jn / MASS) * nx; vy += (jn / MASS) * ny; vz += (jn / MASS) * nz
                    jn_track += jn
                # tangential slip at contact -> friction (stick capped by mu*jn)
                cvx = vx + (wy * rcz - wz * rcy)
                cvy = vy + (wz * rcx - wx * rcz)
                cvz = vz + (wx * rcy - wy * rcx)
                vn = cvx * nx + cvy * ny + cvz * nz
                tx, ty, tz = cvx - vn * nx, cvy - vn * ny, cvz - vn * nz
                tnorm = math.sqrt(tx * tx + ty * ty + tz * tz)
                if tnorm > 1e-12:
                    j_stick = (2.0 / 7.0) * MASS * tnorm  # impulse to enforce rolling
                    j_cap = self.mu * jn_track
                    jf = min(j_stick, max(j_cap, 0.0))
                    fx, fy, fz = -jf * tx / tnorm, -jf * ty / tnorm, -jf * tz / tnorm
                    vx += fx / MASS; vy += fy / MASS; vz += fz / MASS
                    # torque: dw = I^-1 (rc x f)
                    cx = rcy * fz - rcz * fy
                    cy = rcz * fx - rcx * fz
                    cz = rcx * fy - rcy * fx
                    wx += cx / INERTIA; wy += cy / INERTIA; wz += cz / INERTIA

            # ---- ground ----
            s_g = z - R_BALL
            if s_g < CONTACT_MARGIN:
                if s_g < 0.0:
                    z -= s_g
                nx, ny, nz = 0.0, 0.0, 1.0
                rcx, rcy, rcz = 0.0, 0.0, -R_BALL
                cvx = vx + (wy * rcz - wz * rcy)
                cvy = vy + (wz * rcx - wx * rcz)
                cvz = vz + (wx * rcy - wy * rcx)
                vn = cvz
                if vn < 0.0:
                    jn = -MASS * vn
                    vz += jn / MASS
                    jn_ground += jn
                cvx = vx + (wy * rcz - wz * rcy)
                cvy = vy + (wz * rcx - wx * rcz)
                cvz = vz + (wx * rcy - wy * rcx)
                vn = cvz
                tx, ty, tz = cvx - vn * nx, cvy - vn * ny, cvz - vn * nz
                tnorm = math.sqrt(tx * tx + ty * ty + tz * tz)
                if tnorm > 1e-12:
                    j_stick = (2.0 / 7.0) * MASS * tnorm
                    j_cap = self.mu * jn_ground
                    jf = min(j_stick, max(j_cap, 0.0))
                    fx, fy, fz = -jf * tx / tnorm, -jf * ty / tnorm, -jf * tz / tnorm
                    vx += fx / MASS; vy += fy / MASS; vz += fz / MASS
                    cx = rcy * fz - rcz * fy
                    cy = rcz * fx - rcx * fz
                    cz = rcx * fy - rcy * fx
                    wx += cx / INERTIA; wy += cy / INERTIA; wz += cz / INERTIA

        # ---- rolling resistance (dissipation while in rolling contact) ----
        jn_total = jn_track + jn_ground
        if jn_total > 0.0:
            # tangential (to contact normal) center-of-mass velocity
            nx, ny, nz = n_t if jn_track >= jn_ground else (0.0, 0.0, 1.0)
            vn = vx * nx + vy * ny + vz * nz
            tvx, tvy, tvz = vx - vn * nx, vy - vn * ny, vz - vn * nz
            tn = math.sqrt(tvx * tvx + tvy * tvy + tvz * tvz)
            if tn > 1e-9:
                # rolling-resistance impulse; capped so it never reverses motion
                jr = min(MU_ROLL * jn_total, 0.9 * (7.0 / 10.0) * MASS * tn)
                # apply deceleration at COM; stick friction re-syncs spin next substep
                scale = jr / (MASS * tn)
                vx -= tvx * scale
                vy -= tvy * scale
                vz -= tvz * scale

        self.p = (x, y, z)
        self.v = (vx, vy, vz)

        # integrate orientation: qd = 0.5 q (x) (w,0)
        qx, qy, qz, qw = self.q
        dx = 0.5 * (wx * qw + wy * qz - wz * qy)
        dy = 0.5 * (wy * qw + wz * qx - wx * qz)
        dz = 0.5 * (wz * qw + wx * qy - wy * qx)
        dw = 0.5 * (-wx * qx - wy * qy - wz * qz)
        qx += dx * h; qy += dy * h; qz += dz * h; qw += dw * h
        qn = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if qn > 1e-12:
            qx /= qn; qy /= qn; qz /= qn; qw /= qn
        self.q = (qx, qy, qz, qw)
        self.w = (wx, wy, wz)

        # ---- diagnostics bookkeeping ----
        x, y, z = self.p
        th = math.atan2(x, C_LOOP[2] - z)
        dth = th - self.theta_prev
        while dth > math.pi:
            dth -= 2.0 * math.pi
        while dth < -math.pi:
            dth += 2.0 * math.pi
        self.theta_prev = th
        self.theta_total += dth
        self.laps = int(abs(self.theta_total) // (2.0 * math.pi))
        attached = s_t < 0.01
        if self.was_attached and not attached and self.detach_time is None:
            self.detach_time = self.t + h
        self.was_attached = attached
        self.min_track_gap = min(self.min_track_gap, s_t)

    # ------------------------------------------------------------- advance
    def advance_to(self, time: float):
        time = float(time)
        target = max(time, self.t)
        while self.t < target - 1e-12:
            h = min(H_SUB, target - self.t)
            self._substep(h)
            self.t += h
        self.t = time if time >= self.t else self.t
        return self._frame(time)

    # ---------------------------------------------------------------- frame
    def _frame(self, time: float) -> StateFrame:
        x, y, z = self.p
        vx, vy, vz = self.v
        s_t, n_t = _track_sdf(x, y, z)
        s_g = z - R_BALL
        gx, gy, gz = self.gravity

        position = np.array([[x, y, z]], dtype=float)
        velocity = np.array([[vx, vy, vz]], dtype=float)
        orientation = np.array([self.q], dtype=float)
        angular_velocity = np.array([self.w], dtype=float)
        mass = np.array([MASS], dtype=float)
        body_id = np.array([0], dtype=int)

        # contact point slip magnitude -> rolling indicator
        rc = (-R_BALL * n_t[0], -R_BALL * n_t[1], -R_BALL * n_t[2])
        cvx = vx + (self.w[1] * rc[2] - self.w[2] * rc[1])
        cvy = vy + (self.w[2] * rc[0] - self.w[0] * rc[2])
        cvz = vz + (self.w[0] * rc[1] - self.w[1] * rc[0])
        vn = cvx * n_t[0] + cvy * n_t[1] + cvz * n_t[2]
        slip = math.sqrt(max(0.0, cvx * cvx + cvy * cvy + cvz * cvz - vn * vn))
        rolling = 1.0 if (s_t < 0.01 and slip < 0.05) else 0.0

        frame = StateFrame(
            time,
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={"rolling": np.array([rolling])},
            fields={
                # loop-rail contact constraint: signed gap to the track surface
                "constraint_value": np.array([s_t], dtype=float),
                "constraint_target": np.array([0.0], dtype=float),
                # signed geometric gaps (negative = penetration)
                "contact_signed_distance": np.array([s_t, s_g], dtype=float),
                "external_force": np.array([[MASS * gx, MASS * gy, MASS * gz]], dtype=float),
            },
            scalars={
                "constraint_scale": RAIL_RHO,
                "contact_length_scale": T_TUBE,
                "gravity_z": gz,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = math.sqrt(sum(c * c for c in self.v))
        ke = 0.5 * MASS * speed * speed + 0.5 * INERTIA * sum(c * c for c in self.w)
        return {
            "time": self.t,
            "detach_time": self.detach_time,
            "laps_completed": self.laps,
            "theta_total": self.theta_total,
            "speed": speed,
            "kinetic_energy": ke,
            "min_track_gap": self.min_track_gap,
        }
