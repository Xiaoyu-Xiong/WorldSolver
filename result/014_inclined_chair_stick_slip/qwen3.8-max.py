"""Physically-based rigid-body solver for the inclined chair stick-slip scene.

Model
-----
A single rigid chair (mass from the frozen scene) rests on four compliant
plastic feet on a fixed inclined plane.  Contact is handled with a penalty
method in the normal direction and a regularized elastic-bristle (LuGre /
Haessig-Friedland style) model in the tangential direction with a Stribeck
friction curve mu(v) = mu_k + (mu_s - mu_k) exp(-|v|/v_st).  The bristle
force is clamped to the per-foot friction cone; when the cone is exceeded the
foot slips and the bristle snaps back onto the cone surface.  The coupling of
the tangential bristle compliance, the Stribeck velocity weakening, the
rocking (pitch) mode of the tall chair and the uneven normal load split
between the downhill and uphill feet produces the stick-slip chatter
(sliding with repeated stick/slip transitions) observed in the reference
paper (Otten et al., "Staggered Projections for Frictional Contact in
Multibody Systems", Fig. 5 right).

Integration: semi-implicit Euler at fixed dt = 1e-4 s with quaternion
orientation propagation.  Everything is deterministic and drift-free
(orientation renormalized each step, cone-limited forces).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
G = 9.81
G_VEC = np.array([0.0, 0.0, -G])

# Frozen incline plane (from scene_geometry.yaml): box center, half-thickness,
# orientation quaternion (x, y, z, w).
INCLINE_CENTER = np.array([0.0, 0.0, 0.35])
INCLINE_HALF_T = 0.08
INCLINE_Q = np.array([0.0, 0.1564, 0.0, 0.9877])

# Foot contact points in the chair body frame (centers of the four leg
# discs read from chair.stl; local +x points downhill on the incline).
FOOT_OFFSETS = np.array([
    [0.5835, -0.42, -1.12685],   # downhill-left foot
    [0.5835, 0.42, -1.12685],    # downhill-right foot
    [-0.3465, -0.42, -1.12685],  # uphill-left foot
    [-0.3465, 0.42, -1.12685],   # uphill-right foot
])
N_CONTACTS = FOOT_OFFSETS.shape[0]

# Body-frame inertia (kg m^2), estimated by lumping seat/back/leg masses of
# the chair.stl geometry about its center of mass (products of inertia small
# and neglected for a symmetric plastic chair).
I_BODY = np.diag([4.0, 4.5, 2.2])
I_BODY_INV = np.diag([1.0 / 4.0, 1.0 / 4.5, 1.0 / 2.2])

# Contact / friction parameters (plastic on painted incline).
MU_S = 0.36        # static friction coefficient
MU_K = 0.26        # kinetic friction coefficient
V_STRIB = 0.10     # Stribeck velocity (m/s)
K_N = 2.0e5        # normal penalty stiffness per foot (N/m)
C_N = 40.0         # normal penalty damping per foot (N s/m)
K_T = 4.0e4        # tangential bristle stiffness per foot (N/m)
C_T = 10.0         # tangential viscous regularization (N s/m)
CONTACT_TOL = 0.0  # strict penalty contact (dist < 0)

DT = 1.0e-4        # integration substep (s)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_deriv(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    ox, oy, oz = omega
    return 0.5 * np.array([
        w * ox + y * oz - z * oy,
        w * oy + z * ox - x * oz,
        w * oz + x * oy - y * ox,
        -x * ox - y * oy - z * oz,
    ])


class BenchmarkSolver:
    """Rigid chair on an incline with bristle-friction stick-slip contact."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        init = scene.initial_state()
        self._mass = float(np.asarray(init.mass)[0])
        self._p = np.array(init.position[0], dtype=float)
        self._v = np.zeros(3)
        self._q = np.array(init.orientation[0], dtype=float)
        self._q /= np.linalg.norm(self._q)
        self._omega = np.zeros(3)

        # Frozen incline plane: normal and a point on the top surface.
        R_inc = _quat_to_rot(INCLINE_Q)
        self._n = R_inc @ np.array([0.0, 0.0, 1.0])
        self._plane_pt = INCLINE_CENTER + INCLINE_HALF_T * self._n

        # Tiny physical imperfections that seed the rocking/chatter:
        # a small nose-down (downhill) pitch rate and near-noise downhill
        # tangential velocity.  Downhill unit tangent:
        d = G_VEC - np.dot(G_VEC, self._n) * self._n
        self._downhill = d / np.linalg.norm(d)
        R0 = _quat_to_rot(self._q)
        self._omega = R0 @ np.array([0.0, 0.05, 0.0])
        self._v = 0.01 * self._downhill

        # Per-foot tangential bristle deflection (world frame, in plane).
        self._z = np.zeros((N_CONTACTS, 3))
        self._t = 0.0

        # Diagnostics / last-sample field values (initial geometric state).
        R0i = _quat_to_rot(self._q)
        feet0 = self._p + (R0i @ FOOT_OFFSETS.T).T
        self._dists = np.array([
            float(np.dot(feet0[i] - self._plane_pt, self._n))
            for i in range(N_CONTACTS)
        ])
        self._residual = np.zeros(N_CONTACTS)
        self._ext_force = self._mass * G_VEC
        self._n_slip_events = 0
        self._slipping = np.zeros(N_CONTACTS, dtype=bool)
        self._min_dist = np.inf
        self._max_speed = 0.0

        self._constraint_scale = MU_S * self._mass * G
        self._contact_length_scale = INCLINE_HALF_T

    # ------------------------------------------------------------------ dynamics
    def _contact_forces(self, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (total_force, total_torque, distances, cone_residual)."""
        R = _quat_to_rot(self._q)
        r_world = (R @ FOOT_OFFSETS.T).T  # [4,3] lever arms
        total_f = self._mass * G_VEC
        total_tau = np.zeros(3)
        dists = np.empty(N_CONTACTS)
        residual = np.zeros(N_CONTACTS)

        for i in range(N_CONTACTS):
            x_f = self._p + r_world[i]
            u_f = self._v + np.cross(self._omega, r_world[i])
            dist = float(np.dot(x_f - self._plane_pt, self._n))
            dists[i] = dist
            v_n = float(np.dot(u_f, self._n))
            u_t = u_f - v_n * self._n
            speed_t = float(np.linalg.norm(u_t))

            pen = -dist if dist < CONTACT_TOL else 0.0
            if pen > 0.0:
                n_force = K_N * pen - C_N * v_n
                if n_force < 0.0:
                    n_force = 0.0
            else:
                n_force = 0.0

            if n_force > 0.0:
                mu = MU_K + (MU_S - MU_K) * np.exp(-speed_t / V_STRIB)
                f_max = mu * n_force
                # bristle trial step (stuck dynamics)
                z_trial = self._z[i] + h * u_t
                f_t = -K_T * z_trial - C_T * u_t
                f_mag = float(np.linalg.norm(f_t))
                slipping = f_mag > f_max
                if slipping:
                    scale = f_max / f_mag
                    f_t = f_t * scale
                    # keep bristle consistent with the clamped force
                    self._z[i] = -(f_t + C_T * u_t) / K_T
                else:
                    self._z[i] = z_trial
                if slipping and not self._slipping[i]:
                    self._n_slip_events += 1
                self._slipping[i] = slipping
                residual[i] = float(np.linalg.norm(f_t)) - MU_S * n_force
                f_c = n_force * self._n + f_t
            else:
                # airborne foot: no contact, bristle relaxes quickly
                self._z[i] *= np.exp(-h / 0.005)
                self._slipping[i] = False
                residual[i] = 0.0
                f_c = np.zeros(3)

            total_f = total_f + f_c
            total_tau = total_tau + np.cross(r_world[i], f_c)

        return total_f, total_tau, dists, residual

    def _step(self, h: float) -> None:
        f_total, tau, dists, residual = self._contact_forces(h)

        # linear
        self._v = self._v + (h / self._mass) * f_total
        self._p = self._p + h * self._v

        # angular (body-frame inertia)
        R = _quat_to_rot(self._q)
        I_w = R @ I_BODY @ R.T
        omega_body = R.T @ self._omega
        tau_body = R.T @ tau - np.cross(omega_body, I_BODY @ omega_body)
        alpha_body = I_BODY_INV @ tau_body
        self._omega = self._omega + h * (R @ alpha_body)

        self._q = self._q + h * _quat_deriv(self._q, self._omega)
        norm = np.linalg.norm(self._q)
        self._q /= norm

        self._t += h
        self._dists = dists
        self._residual = residual
        self._min_dist = min(self._min_dist, float(dists.min()))
        self._max_speed = max(self._max_speed, float(np.linalg.norm(self._v)))

    def _integrate_to(self, t: float) -> None:
        while self._t < t - 1e-12:
            self._step(min(DT, t - self._t))

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        if t < self._t:
            t = self._t  # monotone safeguard
        self._integrate_to(t)

        position = self._p.reshape(1, 3).copy()
        velocity = self._v.reshape(1, 3).copy()
        orientation = self._q.reshape(1, 4).copy()
        angular_velocity = self._omega.reshape(1, 3).copy()
        mass = np.array([self._mass])
        body_id = np.array([0], dtype=np.int64)

        dists = np.nan_to_num(self._dists, nan=1.0)
        residual = np.nan_to_num(self._residual, nan=0.0)

        fields = {
            "constraint_value": residual.astype(float),
            "constraint_target": np.zeros(N_CONTACTS),
            "contact_signed_distance": dists.astype(float),
            "external_force": self._ext_force.reshape(1, 3).copy(),
        }
        scalars = {
            "constraint_scale": float(self._constraint_scale),
            "contact_length_scale": float(self._contact_length_scale),
        }
        return StateFrame(
            float(self._t),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self._seed,
            "substep_dt": DT,
            "time": self._t,
            "slip_events": int(self._n_slip_events),
            "min_signed_distance": float(self._min_dist),
            "max_speed": float(self._max_speed),
            "mu_s": MU_S,
            "mu_k": MU_K,
            "v_stribeck": V_STRIB,
        }
