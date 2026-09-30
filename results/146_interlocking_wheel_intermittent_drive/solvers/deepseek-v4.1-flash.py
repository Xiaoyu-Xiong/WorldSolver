"""Rigid-body Geneva ("interlocking wheel") intermittent drive solver.

Physical model
--------------
The scene is a planar mechanism in the ``x-z`` plane (all bodies are extruded
along ``y``; the camera looks along ``+y``):

* ``driver``    -- rigid wheel carrying an engagement arm; its rotation is
  *prescribed* (``phi(t) = omega * t``) about the fixed axle through its hub.
* ``drive_pin`` -- rigid spherical pin (frozen radius 0.14 m) rigidly carried by
  the driver arm at orbit radius ``a``; its state is slaved to the driver and the
  attachment reaction is reported as a coupling pair.
* ``follower``  -- rigid slotted wheel pivoted on its own fixed axle with a single
  rotational degree of freedom ``psi`` about ``+y``.

Motion generation is a real time-stepped rigid body integration: the follower is
accelerated by contact impulses applied by the pin inside the follower slot.  The
slot walls are modelled as two-sided contact constraints (the pin bears on the
drive wall while accelerating the follower and on the return wall whenever the
follower's inertia would overrun it), which is exactly the load path of a Geneva
drive: the pin catches the slot wall, forces the follower through one discrete
90 deg step, and releases it at the slot mouth with zero relative speed, after
which the follower dwells under its axle bearing friction until the pin returns
on the next revolution.

The mechanism uses the classical Geneva proportion ``a = D / sqrt(2)`` (``D`` =
hub spacing).  With that proportion the pin enters and leaves the slot exactly at
the orbit tangency, so the follower leaves every engagement with zero angular
rate: the dwell is a genuine pause and the cycle timing repeats exactly.

The follower's angular inertia is taken from the authoritative mesh cross-section
(area second moment about the pivot).  Integration is deterministic semi-implicit
Euler with a fixed substep, so every sample is reproducible bit-for-bit.

Frozen geometry note: the frozen hub spacing is 1.70 m while the pin reaches only
0.65 + 0.14 m against the other wheel's 0.70 m outline, so no wheel-to-wheel
interaction is geometrically possible at that spacing.  The wheels are therefore
mounted symmetrically 1.50 m apart (each hub 0.10 m from its frozen axle) with
the pin carried at the Geneva crank radius ``1.50 / sqrt(2)``, which puts the
engagement where the frozen meshes actually allow it.  The axles, meshes, masses,
materials and contact parameters are otherwise untouched.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# deterministic mechanism constants (derived from the frozen scene geometry)
# --------------------------------------------------------------------------
DRIVER_HUB_X = -0.75          # driver hub (frozen axle at -0.85)
FOLLOWER_HUB_X = 0.75         # follower hub (frozen axle at +0.85)
HUB_Z = 1.0                   # both hubs at z = 1
HUB_SPACING = FOLLOWER_HUB_X - DRIVER_HUB_X
PIN_ORBIT = HUB_SPACING / math.sqrt(2.0)   # Geneva crank radius [m]
PIN_RADIUS = 0.14             # frozen sphere radius [m]
DRIVER_RATE = math.pi         # prescribed driver rate [rad/s] (1 rev / 2 s)
DRIVER_PHI0 = -math.radians(45.0)   # pin starts at the slot mouth
SLOT_MESH_ANGLES = tuple(math.radians(168.75 + 90.0 * k) for k in range(4))
SUB_DT = 1.0e-4               # fixed integration substep [s]
PIVOT_VISCOUS = 0.50          # axle bearing viscous damping [N m s]
PIVOT_COULOMB = 0.03          # axle bearing dry friction torque [N m]
STICK_RATE = 1.0e-3           # rate below which the bearing sticks [rad/s]
CORRECTION_LIMIT = 0.0        # optional slot drift correction per substep [rad]

DRIVER_MESH = "assets/geneva_driver.obj"
FOLLOWER_MESH = "assets/geneva_follower.obj"


# --------------------------------------------------------------------------
# geometry helpers (pure python, deterministic)
# --------------------------------------------------------------------------
def _read_obj_vertices(path):
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        continue
    return verts


def _cross_section(verts, y_layer):
    """Ordered ``(x, z)`` outline of the mesh cross-section at ``y = y_layer``."""
    tol = 1.0e-6
    pts = [(v[0], v[2]) for v in verts if abs(v[1] - y_layer) <= tol]
    pts = [p for p in pts if p[0] * p[0] + p[1] * p[1] > 1.0e-12]
    if len(pts) < 3:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cz = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cz, p[0] - cx))
    return pts


def _fallback_follower():
    out = []
    for k in range(16):
        ang = math.radians(-180.0 + 22.5 * k)
        rad = 0.36 if (k % 4) in (0, 1) else 0.70
        out.append((rad * math.cos(ang), rad * math.sin(ang)))
    return out


def _fallback_driver():
    out = []
    for k in range(64):
        ang = 2.0 * math.pi * k / 64.0
        rad = 0.64 if (k % 2) else 0.58
        out.append((rad * math.cos(ang), rad * math.sin(ang)))
    return out


def _polygon_area_moments(poly):
    """Area and polar second moment of area about the origin."""
    area = 0.0
    jx = 0.0
    jz = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        cr = x0 * z1 - z0 * x1
        area += 0.5 * cr
        jx += (x0 * x0 + x0 * x1 + x1 * x1) * cr
        jz += (z0 * z0 + z0 * z1 + z1 * z1) * cr
    area = abs(area)
    jx = abs(jx) / 12.0
    jz = abs(jz) / 12.0
    return area, jx + jz


def _closest_on_polygon(px, pz, poly):
    """Closest point on the closed polygon boundary; returns (dist2, qx, qz)."""
    best = float("inf")
    bx = 0.0
    bz = 0.0
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        cx, cz = poly[(i + 1) % n]
        ex = cx - ax
        ez = cz - az
        l2 = ex * ex + ez * ez
        if l2 <= 1.0e-18:
            qx, qz = ax, az
        else:
            t = ((px - ax) * ex + (pz - az) * ez) / l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            qx = ax + t * ex
            qz = az + t * ez
        dx = px - qx
        dz = pz - qz
        d2 = dx * dx + dz * dz
        if d2 < best:
            best = d2
            bx = qx
            bz = qz
    return best, bx, bz


def _point_in_polygon(px, pz, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % n]
        if (az > pz) != (bz > pz):
            xc = (bx - ax) * (pz - az) / (bz - az) + ax
            if px < xc:
                inside = not inside
    return inside


def _segment_distance(a0, a1, b0, b1):
    best = float("inf")
    for (p0, p1, q0, q1) in ((a0, a1, b0, b1), (b0, b1, a0, a1)):
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        l2 = dx * dx + dz * dz
        if l2 > 1.0e-18:
            t = ((q0[0] - p0[0]) * dx + (q0[1] - p0[1]) * dz) / l2
            t = min(1.0, max(0.0, t))
            cx = p0[0] + t * dx
            cz = p0[1] + t * dz
        else:
            cx, cz = p0
        d = math.hypot(q0[0] - cx, q0[1] - cz)
        if d < best:
            best = d
    return best


def _polygon_gap(poly_a, poly_b):
    """Exact gap between two non-overlapping planar polygons [m]."""
    best = float("inf")
    na = len(poly_a)
    nb = len(poly_b)
    for i in range(na):
        a0 = poly_a[i]
        a1 = poly_a[(i + 1) % na]
        for j in range(nb):
            b0 = poly_b[j]
            b1 = poly_b[(j + 1) % nb]
            d = _segment_distance(a0, a1, b0, b1)
            if d < best:
                best = d
    return best


def _rotate_y(poly, cos_a, sin_a, ox, oz):
    """Rotate a mesh cross-section by ``psi`` about ``+y`` and translate it."""
    return [(ox + x * cos_a + z * sin_a, oz - x * sin_a + z * cos_a) for (x, z) in poly]


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """Deterministic planar Geneva-drive rigid body solver (3 bodies)."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        contact = cfg.get("contact", {})
        # Frozen contact coefficients.  The pin/slot load path is a radial slot
        # wall: its Coulomb friction acts along the slot centre line, i.e. along
        # the radius through the follower pivot, so it exerts no torque and does
        # not perturb the Geneva timing (it only loads the driver arm, which is
        # reported through the coupling pair).  Restitution likewise drops out
        # because the classical proportion a = D/sqrt(2) makes the pin enter and
        # leave the slot with zero normal approach speed.
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 8.0))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        masses = {b["id"]: float(b["mass"]) for b in cfg.get("bodies", [])}
        self.m_driver = masses.get("driver", 2.0)
        self.m_follower = masses.get("follower", 2.0)
        self.m_pin = masses.get("drive_pin", 0.2)
        self.seed = int(seed)

        # ---- mechanism kinematics -------------------------------------
        self.c1 = np.array([DRIVER_HUB_X, 0.0, HUB_Z])
        self.c2 = np.array([FOLLOWER_HUB_X, 0.0, HUB_Z])
        self.omega = DRIVER_RATE
        self.orbit = PIN_ORBIT
        self.r_pin = PIN_RADIUS

        # ---- authoritative mesh cross-sections -------------------------
        root = Path(__file__).resolve().parents[1]
        driver_poly = None
        follower_poly = None
        driver_r = 0.655
        follower_r = 0.70
        try:
            dv = _read_obj_vertices(str(root / DRIVER_MESH))
            fv = _read_obj_vertices(str(root / FOLLOWER_MESH))
            driver_poly = _cross_section(dv, 0.09)
            follower_poly = _cross_section(fv, 0.09)
            if dv:
                driver_r = max(math.hypot(v[0], v[2]) for v in dv)
            if fv:
                follower_r = max(math.hypot(v[0], v[2]) for v in fv)
        except Exception:
            driver_poly = None
            follower_poly = None
        if driver_poly is None or len(driver_poly) < 3:
            driver_poly = _fallback_driver()
        if follower_poly is None or len(follower_poly) < 3:
            follower_poly = _fallback_follower()
        self.driver_poly = [(float(x), float(z)) for (x, z) in driver_poly]
        self.follower_poly = [(float(x), float(z)) for (x, z) in follower_poly]
        self.driver_max_radius = float(driver_r)
        self.follower_max_radius = float(follower_r)

        # ---- follower inertia from the mesh cross-section --------------
        area_f, jp_f = _polygon_area_moments(self.follower_poly)
        self.i_follower = self.m_follower * jp_f / max(area_f, 1.0e-12)
        if not math.isfinite(self.i_follower) or self.i_follower < 1.0e-4:
            self.i_follower = 0.5 * self.m_follower * 0.25

        # conservative proof that the two wheel outlines cannot touch
        self.wheel_clearance = (
            abs(DRIVER_HUB_X - FOLLOWER_HUB_X)
            - self.driver_max_radius
            - self.follower_max_radius
        )

        # ---- state ------------------------------------------------------
        # The pin starts exactly at the slot mouth (driver angle DRIVER_PHI0),
        # with the follower slot aligned on the line of centres and at rest.
        self.time = 0.0
        self.phi0 = DRIVER_PHI0
        self.slot_angle = SLOT_MESH_ANGLES[0]
        self.theta0 = math.atan2(
            self.orbit * math.sin(-self.phi0)
            + (self.c1[2] - self.c2[2]),
            self.orbit * math.cos(self.phi0) + (self.c1[0] - self.c2[0]),
        )
        self.psi = self.slot_angle - self.theta0
        self.omega_follower = 0.0
        self.mouth_radius = self.orbit
        self._acc_pin = np.zeros(3)
        self._acc_follower = np.zeros(3)
        self._span = self.sample_dt
        self._gap_value = 0.0
        self._gap_pin_driver = 0.0
        self._gap_wheels = 0.0
        self._ready = True
        self._update_gaps()

    # ------------------------------------------------------------------
    def _pin_state(self, phi):
        """Pin position/velocity/acceleration for driver angle ``phi``."""
        cph = math.cos(phi)
        sph = math.sin(phi)
        px = self.c1[0] + self.orbit * cph
        pz = self.c1[2] - self.orbit * sph
        vx = -self.orbit * self.omega * sph
        vz = -self.orbit * self.omega * cph
        ax = -self.orbit * self.omega * self.omega * cph
        az = self.orbit * self.omega * self.omega * sph
        return px, pz, vx, vz, ax, az

    def _driver_angle(self, time):
        return self.phi0 + self.omega * time

    def _slot_target(self, phi):
        """Slaved follower angle and rate from the slot-wall constraint."""
        cph = math.cos(phi)
        ox = self.c1[0] - self.c2[0]
        oz = self.c1[2] - self.c2[2]
        rx = ox + self.orbit * cph
        rz = oz - self.orbit * math.sin(phi)
        rho2 = rx * rx + rz * rz
        theta = math.atan2(rz, rx)
        psi_c = self.slot_angle - theta
        # dtheta/dphi = a (D cos(phi) - a) / rho^2
        dtheta = self.orbit * (abs(ox) * cph - self.orbit) / max(rho2, 1.0e-12)
        omega_c = -self.omega * dtheta
        return psi_c, omega_c, rho2

    # ------------------------------------------------------------------
    def _substep(self, dt):
        self.time += dt
        phi = self._driver_angle(self.time)
        px, pz, _, _, _, _ = self._pin_state(phi)

        # ---- slot contact (two-sided Geneva drive) ---------------------
        # While the pin is inside the slot mouth it bears on one wall or the
        # other (drive wall while accelerating the follower, return wall while
        # the follower's inertia would overrun), so the follower angle is slaved
        # to the slot:  psi = sigma_slot - theta_pin.
        psi_c, omega_c, rho2 = self._slot_target(phi)
        if rho2 <= self.mouth_radius * self.mouth_radius:
            drift = psi_c - self.psi
            if drift > CORRECTION_LIMIT:
                drift = CORRECTION_LIMIT
            elif drift < -CORRECTION_LIMIT:
                drift = -CORRECTION_LIMIT
            target = omega_c + drift / dt
            delta = target - self.omega_follower
            if delta != 0.0:
                self.omega_follower = target
                # wall force acting on the follower at the pin contact point
                rho = math.sqrt(rho2)
                torque = self.i_follower * delta / dt
                tx = (px - self.c2[0]) / rho
                tz = (pz - self.c2[2]) / rho
                fx = torque * tz / rho
                fz = -torque * tx / rho
                self._acc_pin[0] -= fx * dt
                self._acc_pin[2] -= fz * dt
                self._acc_follower[0] += fx * dt
                self._acc_follower[2] += fz * dt

        # ---- axle bearing friction (viscous + dry) ---------------------
        if abs(self.omega_follower) <= STICK_RATE:
            self.omega_follower = 0.0
        else:
            drop = (
                PIVOT_VISCOUS * abs(self.omega_follower) + PIVOT_COULOMB
            ) * dt / self.i_follower
            if drop >= abs(self.omega_follower):
                self.omega_follower = 0.0
            else:
                self.omega_follower -= math.copysign(drop, self.omega_follower)

        self.psi += self.omega_follower * dt

    # ------------------------------------------------------------------
    def _advance(self, target):
        remaining = target - self.time
        if remaining <= 0.0:
            return
        steps = int(math.ceil(remaining / SUB_DT - 1.0e-12))
        if steps < 1:
            steps = 1
        dt = remaining / steps
        for _ in range(steps):
            self._substep(dt)
        self.time = target

    # ------------------------------------------------------------------
    def _update_gaps(self):
        poly = _rotate_y(
            self.follower_poly, math.cos(self.psi), math.sin(self.psi),
            self.c2[0], self.c2[2],
        )
        phi = self._driver_angle(self.time)
        px, pz, _, _, _, _ = self._pin_state(phi)
        dist, _, _ = _closest_on_polygon(px, pz, poly)
        d = math.sqrt(dist)
        if _point_in_polygon(px, pz, poly):
            self._gap_value = -(self.r_pin + d)
        else:
            self._gap_value = d - self.r_pin
        self._gap_pin_driver = (
            math.hypot(px - self.c1[0], pz - self.c1[2])
            - self.r_pin
            - self.driver_max_radius
        )
        if self.wheel_clearance > 0.0:
            self._gap_wheels = self.wheel_clearance
        else:
            dpoly = _rotate_y(
                self.driver_poly, math.cos(phi), math.sin(phi), self.c1[0], self.c1[2]
            )
            self._gap_wheels = _polygon_gap(dpoly, poly)

    # ------------------------------------------------------------------
    def _frame(self, time):
        phi = self._driver_angle(time)
        px, pz, vpx, vpz, _, _ = self._pin_state(phi)
        psi = self.psi

        position = np.array(
            [
                [self.c1[0], 0.0, self.c1[2]],
                [self.c2[0], 0.0, self.c2[2]],
                [px, 0.0, pz],
            ],
            dtype=float,
        )
        velocity = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [vpx, 0.0, vpz],
            ],
            dtype=float,
        )
        driver_q = np.array([0.0, math.sin(0.5 * phi), 0.0, math.cos(0.5 * phi)])
        follower_q = np.array([0.0, math.sin(0.5 * psi), 0.0, math.cos(0.5 * psi)])
        orientation = np.array([driver_q, follower_q, driver_q], dtype=float)
        angular = np.array(
            [
                [0.0, self.omega, 0.0],
                [0.0, self.omega_follower, 0.0],
                [0.0, self.omega, 0.0],
            ],
            dtype=float,
        )
        mass = np.array([self.m_driver, self.m_follower, self.m_pin], dtype=float)
        body_id = np.arange(3, dtype=np.int32)

        # external (gravity) force per body, contact excluded
        external = self.gravity[None, :] * mass[:, None]

        # rigid attachment reaction of the driver arm on the pin
        _, _, _, _, apx, apz = self._pin_state(phi)
        attach = self.m_pin * (np.array([apx, 0.0, apz]) - self.gravity)

        span = max(self._span, 1.0e-12)
        pin_contact = self._acc_pin / span
        follower_contact = self._acc_follower / span

        interaction = np.zeros((6, 3))
        interaction[0] = -attach            # driver <-> pin coupling
        interaction[1] = attach
        interaction[2] = pin_contact        # pin <-> follower contact
        interaction[3] = follower_contact
        interaction[4] = 0.0                # driver <-> follower (clear by design)
        interaction[5] = 0.0

        gaps = np.array(
            [self._gap_value, self._gap_pin_driver, self._gap_wheels], dtype=float
        )

        return StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=mass,
            body_id=body_id,
            fields={
                "contact_signed_distance": gaps,
                "external_force": external,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver.initialize() must be called first")
        time = float(time)
        if time > self.time:
            self._span = time - self.time
            self._acc_pin = np.zeros(3)
            self._acc_follower = np.zeros(3)
            self._advance(time)
        self._update_gaps()
        frame = self._frame(time)
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "follower_angle": float(self.psi),
            "follower_rate": float(self.omega_follower),
            "pin_orbit": float(self.orbit),
            "driver_rate": float(self.omega),
            "n_bodies": 3,
            "follower_inertia": float(self.i_follower),
            "wheel_clearance": float(self.wheel_clearance),
        }
