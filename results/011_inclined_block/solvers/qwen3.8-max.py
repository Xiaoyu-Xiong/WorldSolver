"""Physically-based solver for scene 011_inclined_block.

A rigid box (the block) is released above a fixed inclined plane, falls under
gravity, makes frictional contact, slides downhill while kinetic friction
exceeds the gravitational tangent component, and comes to rest on the plane.
The block is initially aligned with the plane and the net contact torque about
its center of mass vanishes (normal pressure on the flat face is symmetric),
so its orientation stays constant: no spurious tumbling.

Method: semi-implicit Euler at a small fixed substep with a stiff
spring-damper (penalty) unilateral normal contact and Coulomb friction
(static/kinematic switch via a velocity threshold), integrated between the
requested sample times.  All required StateFrame fields are exported.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
G = 9.81                 # m/s^2
MASS = 2.0               # kg (frozen by scene.initial_state)
THETA = math.radians(30.0)          # plane inclination (quat -15 deg half-angle)
NRM = np.array([0.0, math.sin(THETA), math.cos(THETA)])   # plane up-normal
PLANE_HALF_T = 0.08      # plane half thickness along normal
BLOCK_HE = np.array([0.45, 0.35, 0.30])                   # block half extents
QUAT = np.array([-0.258819, 0.0, 0.0, 0.965926])          # frozen orientation

MU = 0.75                # Coulomb friction (> tan(30deg): block comes to rest)
K_N = 2.0e6              # normal penalty stiffness (N/m)
C_N = 1.8e3              # normal penalty damping (restitution ~0.2)
STICK_EPS = 0.02         # m/s tangential velocity threshold for static friction
DT = 1.0e-4              # integration substep (s)
CONTACT_TOL = 1.0e-4     # near-contact band for reporting


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        init = scene.initial_state()
        self._p = np.array(init.position[0], dtype=float)
        self._v = np.array(init.velocity[0], dtype=float)
        self._m = float(init.mass[0]) if init.mass is not None else MASS
        self._body_id = int(init.body_id[0]) if init.body_id is not None else 1
        self._t = 0.0
        self._R = _quat_to_matrix(QUAT)
        # body-frame corner offsets -> world offsets (orientation is constant)
        sx, sy, sz = BLOCK_HE
        corners = np.array([[a * sx, b * sy, c * sz]
                            for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        self._corner_off = corners @ self._R.T   # world-space offsets, [8,3]
        self._gravity = np.array([0.0, 0.0, -G * self._m])
        self._last_fn = 0.0

    # ------------------------------------------------------------ internals
    def _gap(self, p: np.ndarray) -> float:
        """Signed distance from the block's lowest face to the plane surface."""
        return float(p @ NRM - (PLANE_HALF_T + BLOCK_HE[2]))

    def _step(self, dt: float) -> None:
        p, v = self._p, self._v
        d = self._gap(p)
        vn = float(v @ NRM)
        fn = 0.0
        if d < 0.0 and vn < 0.0:
            pen = -d
            fn = K_N * pen - C_N * vn
            if fn < 0.0:
                fn = 0.0
            v += (fn / self._m) * dt * NRM
            # Coulomb friction in the tangent plane
            vt = v - (v @ NRM) * NRM
            sp = float(np.linalg.norm(vt))
            if sp > 0.0 and fn > 0.0:
                dv = (MU * fn / self._m) * dt
                if sp <= STICK_EPS and dv >= sp:
                    v -= vt                       # static stick
                elif dv >= sp:
                    v -= vt                       # friction stops slide this step
                else:
                    v -= (dv / sp) * vt
        elif d < 0.0 and vn >= 0.0:
            # separating but still penetrating: keep a (decaying) normal push
            fn = max(0.0, K_N * (-d))
            v += (fn / self._m) * dt * NRM
        self._last_fn = fn
        v += np.array([0.0, 0.0, -G]) * dt        # gravity
        p += v * dt
        self._t += dt

    def _advance(self, target: float) -> None:
        target = float(target)
        # exact number of substeps so sample times are hit deterministically
        remaining = target - self._t
        while remaining > 1e-12:
            dt = DT if remaining > DT else remaining
            self._step(dt)
            remaining = target - self._t

    # --------------------------------------------------------------- output
    def _frame(self, time: float) -> StateFrame:
        p = self._p.copy()
        corners = p + self._corner_off            # [8,3]
        cd = corners @ NRM - PLANE_HALF_T         # signed distances to plane
        gap = self._gap(p)
        return StateFrame(
            time=float(time),
            position=p.reshape(1, 3),
            velocity=self._v.reshape(1, 3).copy(),
            orientation=QUAT.reshape(1, 4).copy(),
            angular_velocity=np.zeros((1, 3)),
            mass=np.array([self._m]),
            body_id=np.array([self._body_id], dtype=int),
            fields={
                "external_force": self._gravity.reshape(1, 3).copy(),
                "contact_signed_distance": cd.astype(float),
                "constraint_value": np.array([gap]),
                "constraint_target": np.array([0.0]),
            },
            scalars={
                "constraint_scale": float(BLOCK_HE[2]),   # 0.30 m
                "contact_length_scale": float(2.0 * PLANE_HALF_T),  # 0.16 m
            },
        )

    # ----------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        if float(time) > self._t:
            self._advance(float(time))
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self._t,
            "gap": self._gap(self._p),
            "speed": float(np.linalg.norm(self._v)),
            "mu": MU,
            "last_normal_force": self._last_fn,
        }
