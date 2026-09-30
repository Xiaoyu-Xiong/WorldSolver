"""Physically-based solver for the spolling coin decay scene (Euler's disk).

A single rigid thin coin (cylinder) falls onto a rigid horizontal floor with an
initial spin about its symmetry axis.  Frictional rim contact, gravity and a
small rolling-resistance torque reproduce the characteristic wobbling
precession whose rate increases as the tilt angle decays, ending at rest.

Method: semi-implicit Euler integration of the full rigid-body equations
(translational + quaternion rotational) with a penalty (spring-damper) normal
contact model and regularized Coulomb friction applied at sample points on the
two rim circles of the cylinder.  Fixed small substeps guarantee stability and
non-penetration.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_to_matrix(q):
    """xyzw quaternion -> 3x3 rotation matrix (body -> world)."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_deriv(q, omega):
    """dq/dt = 0.5 * (omega as pure quaternion) * q, xyzw convention."""
    x, y, z, w = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        wx * w + wy * z - wz * y,
        wy * w + wz * x - wx * z,
        wz * w + wx * y - wy * x,
        -wx * x - wy * y - wz * z,
    ])


class BenchmarkSolver:
    # --- geometry / material -------------------------------------------------
    R = 0.48          # coin radius
    H = 0.08          # coin thickness
    N_RIM = 24        # sample points per rim circle

    # --- numerics ------------------------------------------------------------
    SUB_DT = 2.0e-4   # integration substep
    K_N = 1.0e5       # normal penalty stiffness
    C_N = 250.0       # normal penalty damping
    FRIC_EPS = 1.0e-3 # regularization of Coulomb friction
    SPIN0 = 45.0      # initial spin about coin symmetry axis (rad/s)
    C_ROLL = 1.5e-3   # rolling-resistance coefficient (torque = -C*Fn*omega)

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.mass = float(cfg["bodies"][0]["mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.55))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.duration = float(cfg.get("duration", 8.0))

        # Body-frame inertia of a solid cylinder about its centre.
        m, R, H = self.mass, self.R, self.H
        self.I_body = np.diag([
            m * (3 * R * R + H * H) / 12.0,
            m * (3 * R * R + H * H) / 12.0,
            m * R * R / 2.0,
        ])

        # Rim contact sample points (body frame): both edge circles.
        phi = np.arange(self.N_RIM) * (2 * np.pi / self.N_RIM)
        c = R * np.cos(phi)
        s = R * np.sin(phi)
        zc = np.full(self.N_RIM, H / 2.0)
        self.P_body = np.vstack([
            np.column_stack([c, s, zc]),
            np.column_stack([c, s, -zc]),
        ])  # [2*N_RIM, 3]
        self.n_pts = self.P_body.shape[0]

        # Initial state from scene definition.
        q0 = np.asarray(cfg["bodies"][0].get("orientation", [0.0, 0.0, 0.0, 1.0]),
                        dtype=float)
        q0 = q0 / np.linalg.norm(q0)
        self.q = q0.copy()
        self.pos = np.asarray(cfg["bodies"][0]["position"], dtype=float).copy()
        self.vel = np.zeros(3)
        # Spin rapidly about the coin's own symmetry axis (body +z).
        Rmat = _quat_to_matrix(self.q)
        self.omega = Rmat @ np.array([0.0, 0.0, self.SPIN0])
        # Tiny seeding tilt rate so precession phase is well defined.
        self.omega += Rmat @ np.array([0.5, 0.0, 0.0])

        self.t = 0.0
        # Cached contact diagnostics for output fields.
        self._signed_dist = np.zeros(4)
        self._inter_force = np.zeros((4, 3))
        self._update_contact_outputs(Rmat)

    # ------------------------------------------------------------------ physics
    def _contact_forces(self, Rmat):
        """Penalty normal + regularized Coulomb friction at rim sample points.

        Returns (total_force, total_torque_world, signed_distances, point_forces).
        """
        P = self.P_body @ Rmat.T            # world offsets r_i
        pts = self.pos + P                   # world contact points
        vels = self.vel + np.cross(np.broadcast_to(self.omega, P.shape), P)
        d = pts[:, 2]                        # signed distance to floor plane z=0
        vn = vels[:, 2]

        fn = np.where(d < 0.0, -self.K_N * d - self.C_N * vn, 0.0)
        fn = np.maximum(fn, 0.0)

        vt = vels.copy()
        vt[:, 2] = 0.0
        speed = np.sqrt((vt * vt).sum(axis=1) + self.FRIC_EPS ** 2)
        scale = self.mu * fn / speed
        ft = -vt * scale[:, None]

        forces = ft.copy()
        forces[:, 2] += fn                   # total point force (world)

        F = forces.sum(axis=0)
        tau = np.cross(P, forces).sum(axis=0)

        # Rolling resistance torque (only while load is carried).
        fn_tot = fn.sum()
        if fn_tot > 1e-6:
            tau -= self.C_ROLL * fn_tot * self.omega

        return F, tau, d, forces

    def _update_contact_outputs(self, Rmat):
        P = self.P_body @ Rmat.T
        pts = self.pos + P
        d = pts[:, 2]
        order = np.argsort(d)[:4]
        sd = np.zeros(4)
        sd[: len(order)] = d[order]
        self._signed_dist = sd
        # placeholder, refined after the step once forces are known
        self._inter_force = np.zeros((4, 3))

    def _step(self, dt):
        Rmat = _quat_to_matrix(self.q)
        F, tau, d, forces = self._contact_forces(Rmat)

        F_ext = self.mass * self.gravity + F

        # Translational update (semi-implicit).
        self.vel += dt * F_ext / self.mass
        self.pos += dt * self.vel

        # Rotational update in world frame.
        I_w = Rmat @ self.I_body @ Rmat.T
        omega_dot = np.linalg.solve(I_w, tau - np.cross(self.omega, I_w @ self.omega))
        self.omega += dt * omega_dot
        self.q = self.q + dt * _quat_deriv(self.q, self.omega)
        self.q /= np.linalg.norm(self.q)

        # Contact output fields: 4 lowest signed distances, 4 strongest forces.
        P = self.P_body @ Rmat.T
        pts = self.pos + P
        dd = pts[:, 2]
        order = np.argsort(dd)[:4]
        sd = np.zeros(4)
        sd[: len(order)] = dd[order]
        self._signed_dist = sd

        mag = np.linalg.norm(forces, axis=1)
        f_order = np.argsort(mag)[::-1][:4]
        inter = np.zeros((4, 3))
        keep = mag[f_order] > 1e-9
        inter[np.arange(4)[keep]] = forces[f_order[keep]]
        self._inter_force = inter

    def _advance(self, target):
        while self.t < target - 1e-12:
            dt = min(self.SUB_DT, target - self.t)
            self._step(dt)
            self.t += dt

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t:
            self._advance(target)
        ext = (self.mass * self.gravity).reshape(1, 3)
        frame = StateFrame(
            time=target,
            position=self.pos.reshape(1, 3).copy(),
            velocity=self.vel.reshape(1, 3).copy(),
            orientation=self.q.reshape(1, 4).copy(),
            angular_velocity=self.omega.reshape(1, 3).copy(),
            mass=np.array([self.mass]),
            body_id=np.array([0], dtype=np.int32),
            attributes={},
            fields={
                "contact_signed_distance": self._signed_dist.copy(),
                "external_force": ext,
                "interaction_force": self._inter_force.copy(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        tilt = float(np.degrees(np.arccos(
            np.clip(abs(_quat_to_matrix(self.q)[2, 2]), -1.0, 1.0))))
        return {
            "time": self.t,
            "tilt_deg_from_floor_normal": tilt,
            "spin_rate": float(np.linalg.norm(self.omega)),
            "height": float(self.pos[2]),
            "min_signed_distance": float(self._signed_dist[0]),
        }
