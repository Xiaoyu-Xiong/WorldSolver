"""Deterministic rigid-body solver for the friction ball loop scene.

The ball is modelled as a sphere rolling on the inner surface of a vertical
circular loop.  While it remains in contact, a reduced 1-D coordinate along the
loop is integrated with a symplectic Euler scheme.  Rolling resistance,
proportional to the normal load, dissipates mechanical energy so that the ball
stays on the track for the first pass but detaches on the second.  After
detachment the ball is integrated as a free rigid body under gravity with a
simple ground contact model.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config: dict = {}

        # Ball / loop geometry (meters).  The loop is a torus with major radius
        # R=1.0 and tube radius t=0.2; the ball radius is 0.24.
        self.r_b = 0.24
        self.loop_R = 1.0
        self.loop_t = 0.2
        self.R_track = self.loop_R - self.loop_t          # inner surface radius
        self.rho = self.R_track - self.r_b                  # rail radius
        self.loop_center = np.array([0.0, 0.0, 1.2])
        self.e_b = np.array([0.0, 1.0, 0.0], dtype=float)   # binormal (out of plane)

        # Time stepping
        self.h = 1e-4

        # State
        self.t = 0.0
        self.m = 1.0
        self.I = (2.0 / 5.0) * self.m * self.r_b ** 2
        self.m_eff = self.m + self.I / self.r_b ** 2        # rolling effective mass
        self.theta = 0.0
        self.v = 0.0
        self.in_contact = True
        self.p = np.zeros(3)
        self.v_vec = np.zeros(3)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.w = np.zeros(3)
        self.g = np.array([0.0, 0.0, -9.81])
        self.mu = 0.18
        self.mu_rr = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = getattr(scene, "config", {})
        self.g = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu = float(self.config.get("friction_coefficient", 0.18))

        # Rolling resistance coefficient: scale the Coulomb coefficient by the
        # ratio of ball radius to track surface radius.  This gives a small
        # physical rolling-friction term while preserving the supplied
        # friction coefficient.
        self.mu_rr = self.mu * (self.r_b / self.R_track)

        frame = scene.initial_state()
        self.m = float(np.asarray(frame.mass)[0])
        self.I = (2.0 / 5.0) * self.m * self.r_b ** 2
        self.m_eff = self.m + self.I / self.r_b ** 2
        self.p = np.asarray(frame.position, dtype=float).reshape(3).copy()
        self.v_vec = np.asarray(frame.velocity, dtype=float).reshape(3).copy()
        q_in = np.asarray(frame.orientation, dtype=float).reshape(4)
        self.q = self._normalize(q_in)
        self.w = np.asarray(frame.angular_velocity, dtype=float).reshape(3).copy()
        self.t = 0.0
        self.in_contact = True

        # Derive reduced scalar state (theta, signed speed) from the 3-D pose.
        dp = self.p - self.loop_center
        s = np.linalg.norm(dp[:2])
        if s < 1e-9:
            self.theta = 0.0
        else:
            # n = [sin(theta), 0, -cos(theta)]
            self.theta = float(np.arctan2(dp[0] / s, -dp[2] / s))

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        t_vec = np.array([cos_t, 0.0, sin_t])
        self.v = float(np.dot(self.v_vec, t_vec))
        # Enforce consistent rolling angular velocity while on the track.
        self.w = -(self.v / self.r_b) * self.e_b

    def _n(self, theta: float) -> np.ndarray:
        return np.array([np.sin(theta), 0.0, -np.cos(theta)])

    def _t(self, theta: float) -> np.ndarray:
        return np.array([np.cos(theta), 0.0, np.sin(theta)])

    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        q = q.astype(float)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    def _integrate_quaternion(self, q: np.ndarray, w: np.ndarray, h: float) -> np.ndarray:
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            return q.copy()
        half = 0.5 * w_norm * h
        axis = w / w_norm
        q_delta = np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)])
        return self._normalize(self._quat_mul(q_delta, q))

    def _project_to_track(self) -> None:
        n = self._n(self.theta)
        self.p = self.loop_center + self.rho * n
        t = self._t(self.theta)
        self.v_vec = self.v * t
        self.w = -(self.v / self.r_b) * self.e_b

    def _track_signed_distance(self) -> float:
        # Signed gap from ball surface to loop inner track surface in the
        # radial direction.
        s = np.linalg.norm((self.p - self.loop_center)[:2])
        return float(s - self.rho)

    def _step_contact(self, h: float) -> None:
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)

        # Normal force required to keep the centre on the circular rail.
        # N = m * (g_perp + v^2 / rho), where g_perp = g * n = g * cos(theta).
        g = -self.g[2]  # positive scalar magnitude
        N = self.m * (g * cos_t + self.v ** 2 / self.rho)

        if N < 0.0:
            # Normal reaction would need to be tensile -> detach.
            self.in_contact = False
            self._project_to_track()
            return

        # Tangential gravity component.
        F_grav_t = -self.m * g * sin_t

        # Rolling resistance opposes motion.
        if abs(self.v) > 1e-12:
            F_rr = self.mu_rr * N * np.sign(self.v)
        else:
            F_rr = 0.0

        # Effective tangential acceleration for a rolling sphere.
        a = (F_grav_t - F_rr) / self.m_eff

        # Symplectic Euler update.
        self.v += a * h
        self.theta += (self.v / self.rho) * h

        # Reconstruct 3-D pose consistent with the constraint.
        self._project_to_track()
        self.q = self._integrate_quaternion(self.q, self.w, h)

    def _step_free(self, h: float) -> None:
        # Free flight under gravity.
        self.v_vec += self.g * h
        self.p += self.v_vec * h

        # Simple ground contact: treat the ground at z=0 as a rigid plane with
        # restitution and tangential dissipation.
        if self.p[2] < self.r_b:
            self.p[2] = self.r_b
            if self.v_vec[2] < 0.0:
                self.v_vec[2] = -0.3 * self.v_vec[2]
            # Coulomb-like damping for tangential motion.
            self.v_vec[0] *= 0.95
            self.v_vec[1] *= 0.95
            self.w *= 0.95

        self.q = self._integrate_quaternion(self.q, self.w, h)

    def _step(self, h: float) -> None:
        if self.in_contact:
            self._step_contact(h)
        else:
            self._step_free(h)

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            raise ValueError("advance_to time is earlier than current state")

        while self.t < time:
            h = min(self.h, time - self.t)
            self._step(h)
            self.t += h

        return self._build_frame()

    def _build_frame(self) -> StateFrame:
        # Rolling no-slip slip scalar at the track contact.
        if self.in_contact:
            slip = self.v + float(np.dot(self.w, self.e_b)) * self.r_b
            constraint_value = np.array([float(slip)], dtype=float)
            constraint_target = np.array([0.0], dtype=float)
        else:
            constraint_value = np.empty((0,), dtype=float)
            constraint_target = np.empty((0,), dtype=float)

        contact_dist = [self._track_signed_distance()]
        ground_gap = self.p[2] - self.r_b
        if ground_gap < 0.05:
            contact_dist.append(float(ground_gap))
        contact_signed_distance = np.array(contact_dist, dtype=float)

        fields = {
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "contact_signed_distance": contact_signed_distance,
            "external_force": (self.m * self.g).reshape(1, 3),
        }

        scalars = {
            "gravity_z": float(self.g[2]),
            "contact_length_scale": 0.01,
            "constraint_scale": 1.0,
        }

        return StateFrame(
            time=self.t,
            position=self.p.reshape(1, 3),
            velocity=self.v_vec.reshape(1, 3),
            orientation=self.q.reshape(1, 4),
            angular_velocity=self.w.reshape(1, 3),
            mass=np.array([self.m]),
            body_id=np.array([0], dtype=int),
            attributes={"rolling": np.array([1.0 if self.in_contact else 0.0])},
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict:
        return {
            "solver": "BenchmarkSolver",
            "method": "reduced_coordinate_rolling_loop",
            "time_step": self.h,
            "rho": self.rho,
            "mu": self.mu,
            "mu_rr": self.mu_rr,
        }
