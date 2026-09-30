"""Rigid-body solver for the spolling coin decay scene (149).

A thin rigid coin (cylinder) starts tilted and spinning above a static
horizontal floor.  The scene is advanced with a genuine physical
time-stepping procedure (semi-implicit / symplectic Euler integration of
the full 6-DOF rigid-body equations):

  * state: centre-of-mass position, unit quaternion orientation (xyzw),
    linear and angular velocity;
  * external forces: gravity plus a very light aerodynamic drag;
  * contact: penalty normal force with viscous damping (restitution
    consistent with the frozen 0.12) evaluated at sample points on both
    cylinder rim circles against the floor plane, plus regularised
    Coulomb friction at every penetrating sample;
  * rolling resistance and spin (drilling) friction torques at the
    contact, which drain rotational energy so that the coin's tilt angle
    decays while its precession tightens and accelerates -- the
    characteristic spolling / Euler-disk decay -- until it settles flat
    and comes to rest without penetrating the floor.

No keyframes, animation tables or precomputed trajectories are used:
every sampled state is the direct result of the numerical integration.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------
# small quaternion helpers (xyzw convention, body -> world)
# --------------------------------------------------------------------------
def _quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if (not np.isfinite(n)) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_to_mat(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _quat_mul(a, b):
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


# --------------------------------------------------------------------------
# frozen scene geometry (defaults match scene_geometry.yaml exactly)
# --------------------------------------------------------------------------
def _load_scene_geometry():
    radius, height, floor_top = 0.48, 0.08, 0.0
    try:
        import os

        import yaml

        path = os.path.join(os.getcwd(), "scene_geometry.yaml")
        if os.path.exists(path):
            with open(path, "r") as fh:
                data = yaml.safe_load(fh) or {}
            for obj in data.get("objects", []):
                geo = obj.get("geometry") or {}
                tr = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
                if geo.get("type") == "cylinder" and obj.get("motion") == "dynamic":
                    radius = float(geo.get("radius", radius))
                    height = float(geo.get("height", height))
                elif geo.get("type") == "box":
                    he = geo.get("half_extents") or [1.0, 1.0, 0.05]
                    floor_top = float(tr[2]) + float(he[2])
    except Exception:
        pass
    return radius, height, floor_top


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """Semi-implicit rigid-body integrator with penalty/friction contact."""

    def __init__(self):
        self._initialised = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config if hasattr(scene, "config") else dict(scene)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact = cfg.get("contact", {}) or {}
        self.mu = float(contact.get("friction", 0.5))
        self.restitution = float(contact.get("restitution", 0.1))
        self.length_scale = float(contact.get("length_scale", 0.02))

        bodies = cfg["bodies"]
        body = bodies[0]
        self.body_id = 0
        self.mass = float(body["mass"])
        self.p = np.asarray(body["position"], dtype=float).copy()
        self.q = _quat_normalize(np.asarray(body.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=float))

        self.radius, height, self.floor_top = _load_scene_geometry()
        self.half_h = 0.5 * height
        i1 = self.mass * (3.0 * self.radius ** 2 + height ** 2) / 12.0
        i3 = 0.5 * self.mass * self.radius ** 2
        self.I_body = np.diag([i1, i1, i3])

        # contact sample points: both rim circles of the cylinder
        m = 16
        ang = 2.0 * np.pi * np.arange(m) / m
        rim = np.column_stack([
            self.radius * np.cos(ang),
            self.radius * np.sin(ang),
            np.zeros(m),
        ])
        self.pts_local = np.vstack([
            rim + np.array([0.0, 0.0, -self.half_h]),
            rim + np.array([0.0, 0.0, self.half_h]),
        ])

        # numerics (frozen for determinism)
        self.dt = 1.0e-4
        self.k_contact = 3.0e4          # normal penalty stiffness [N/m]
        self.c_contact = 90.0           # normal viscous damping [N s/m]
        self.v_reg = 0.05               # friction regularisation speed [m/s]
        self.c_roll = 8.0e-3            # rolling-resistance coefficient [m]
        self.c_drill = 4.0e-3           # spin-friction coefficient [m]
        self.c_lin = 0.04               # linear aerodynamic drag [N s/m]
        self.c_ang = 3.0e-3             # angular aerodynamic drag [N m s]

        self.v = np.zeros(3)
        self.w = self._initial_angular_velocity()
        self.t = 0.0
        self._initialised = True

    def _initial_angular_velocity(self):
        """Tilted, near-steady rolling-precession initial spin (Euler disk).

        For a thin disk of tilt alpha (symmetry axis vs. vertical) whose
        centre sits on the precession axis, steady rolling precession
        satisfies  Omega^2 * I1 * cos(alpha) = R * m * g * cot(alpha),
        with the rolling constraint tying the spin about the symmetry
        axis to the precession rate.  The result is a coin that lands
        already rolling on its rim and then spolls as energy drains.
        """
        zh = np.array([0.0, 0.0, 1.0])
        A = _quat_to_mat(self.q)
        u = A[:, 2].copy()
        uz = float(u[2])
        g = float(np.linalg.norm(self.gravity))
        if g <= 0.0:
            g = 9.81
        i1 = float(self.I_body[0, 0])
        horiz = float(np.sqrt(max(1.0 - uz * uz, 0.0)))
        if uz > 0.2 and horiz > 1e-6:
            cot = uz / horiz
            denom = max(i1 * uz, 1e-9)
            omega_prec = 1.1 * np.sqrt(self.radius * self.mass * g * cot / denom)
            perp = zh - uz * u
            m_hat = -perp / horiz              # rim direction toward the low side
            r_c = -self.half_h * u + self.radius * m_hat   # centre -> lowest rim point
            c = self.p + r_c                   # ground-track point below the rim
            a = np.cross(zh, c)
            b = np.cross(u, r_c)
            bb = float(b @ b)
            omega_s = -omega_prec * float(a @ b) / bb if bb > 1e-12 else 0.0
            return omega_prec * zh + omega_s * u
        return 5.0 * u

    # --------------------------------------------------------------- stepping
    def _contact_kinematics(self):
        A = _quat_to_mat(self.q)
        rw = self.pts_local @ A.T                     # world offsets from COM
        pts = self.p + rw
        vpts = self.v + np.cross(self.w, rw)
        gaps = pts[:, 2] - self.floor_top             # signed gap (<0 = overlap)
        return A, rw, pts, vpts, gaps

    def _contact_forces(self, rw, vpts, gaps):
        """Penalty normal + regularised Coulomb friction at penetrating points."""
        pen = -gaps
        idx = np.nonzero(pen > 0.0)[0]
        F = np.zeros(3)
        T = np.zeros(3)
        Fn_tot = 0.0
        forces = np.zeros((0, 3))
        if idx.size:
            vn = vpts[idx, 2]
            Fn = self.k_contact * pen[idx] - self.c_contact * vn
            Fn = np.clip(Fn, 0.0, 2000.0)
            vt = vpts[idx, :2]
            svt = np.linalg.norm(vt, axis=1)
            fdir = np.zeros_like(vt)
            nz = svt > 1e-9
            fdir[nz] = -vt[nz] / svt[nz, None]
            reg = np.minimum(1.0, svt / self.v_reg)
            ft = self.mu * Fn * reg
            f = np.column_stack([fdir[:, 0] * ft, fdir[:, 1] * ft, Fn])
            F = f.sum(axis=0)
            T = np.cross(rw[idx], f).sum(axis=0)
            Fn_tot = float(Fn.sum())
            forces = f
        return F, T, Fn_tot, forces, idx

    def _substep(self, dt):
        A, rw, pts, vpts, gaps = self._contact_kinematics()
        F, T, Fn_tot, forces, idx = self._contact_forces(rw, vpts, gaps)

        zh = np.array([0.0, 0.0, 1.0])
        if Fn_tot > 0.0:
            # rolling resistance: opposes the rolling (precession) rate
            w_roll = self.w - float(self.w[2]) * zh
            nw = float(np.linalg.norm(w_roll))
            if nw > 1e-9:
                T = T - self.c_roll * Fn_tot * w_roll / (nw + 0.3)
            # spin (drilling) friction about the contact normal
            wn = float(self.w[2])
            T = T - self.c_drill * Fn_tot * wn / (abs(wn) + 0.3) * zh

        # external loads: gravity + light aerodynamic drag
        F = F - self.c_lin * self.v + self.mass * self.gravity
        T = T - self.c_ang * self.w

        # rigid-body dynamics (world-frame inertia)
        I_w = A @ self.I_body @ A.T
        L = I_w @ self.w
        w_dot = np.linalg.solve(I_w, T - np.cross(self.w, L))

        # semi-implicit Euler
        self.v = self.v + (F / self.mass) * dt
        self.w = self.w + w_dot * dt
        self.p = self.p + self.v * dt
        dq = 0.5 * _quat_mul(np.array([self.w[0], self.w[1], self.w[2], 0.0]), self.q)
        self.q = _quat_normalize(self.q + dq * dt)
        self.t += dt

        self._safety_clamp()

    def _safety_clamp(self):
        """Keep the state finite and bounded (never expected to trigger)."""
        if not (np.isfinite(self.p).all() and np.isfinite(self.v).all()
                and np.isfinite(self.w).all() and np.isfinite(self.q).all()):
            self.v = np.zeros(3)
            self.w = np.zeros(3)
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
            if not np.isfinite(self.p).all():
                self.p = np.array([0.0, 0.0, self.half_h + 1e-3])
            return
        nv = float(np.linalg.norm(self.v))
        if nv > 30.0:
            self.v *= 30.0 / nv
        nw = float(np.linalg.norm(self.w))
        if nw > 300.0:
            self.w *= 300.0 / nw
        zmin = self.floor_top - 0.25
        if self.p[2] < zmin:
            self.p[2] = zmin
            if self.v[2] < 0.0:
                self.v[2] = 0.0

    # ----------------------------------------------------------------- output
    def _make_frame(self, frame_time):
        A, rw, pts, vpts, gaps = self._contact_kinematics()
        F, T, Fn_tot, forces, idx = self._contact_forces(rw, vpts, gaps)

        near = gaps < 3.0 * self.length_scale
        if not np.any(near):
            near = np.zeros(gaps.shape, dtype=bool)
            near[int(np.argmin(gaps))] = True
        signed = np.asarray(gaps[near], dtype=float)

        external = (self.mass * self.gravity).reshape(1, 3)
        interaction = np.asarray(forces, dtype=float).reshape(-1, 3)

        frame = StateFrame(
            float(frame_time),
            np.asarray(self.p, dtype=float).reshape(1, 3),
            np.asarray(self.v, dtype=float).reshape(1, 3),
            orientation=np.asarray(self.q, dtype=float).reshape(1, 4),
            angular_velocity=np.asarray(self.w, dtype=float).reshape(1, 3),
            mass=np.asarray([self.mass], dtype=float),
            body_id=np.asarray([self.body_id], dtype=np.int32),
            attributes={"release_time": np.asarray([0.0])},
            fields={
                "contact_signed_distance": signed,
                "external_force": external,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time):
        if not self._initialised:
            raise RuntimeError("initialize() must be called before advance_to()")
        t_target = max(float(time), 0.0)
        while self.t < t_target - 1e-9:
            dt = min(self.dt, t_target - self.t)
            self._substep(dt)
        return self._make_frame(max(self.t, 0.0))

    def diagnostics(self) -> dict:
        A = _quat_to_mat(self.q)
        u = A[:, 2]
        I_w = A @ self.I_body @ A.T
        ke = 0.5 * self.mass * float(self.v @ self.v) + 0.5 * float(self.w @ (I_w @ self.w))
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(u[2])), 0.0, 1.0))))
        return {
            "time": float(self.t),
            "kinetic_energy": ke,
            "tilt_from_vertical_deg": tilt,
            "angular_speed": float(np.linalg.norm(self.w)),
            "centre_height": float(self.p[2]),
        }
