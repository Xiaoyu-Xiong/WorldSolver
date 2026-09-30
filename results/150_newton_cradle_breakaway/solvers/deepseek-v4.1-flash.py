"""Newton cradle breakaway solver (self-contained, no external physics engine).

Physical model
--------------
Each ball is a spherical pendulum: the ball centre is rigidly attached to a
fixed pivot by an inextensible, massless string, so its generalised coordinate
is the in-plane swing angle ``theta``.  The row of balls lies along the x axis
and every ball swings in the x-z plane, which is the plane of the frozen
camera view.

Dynamics:               theta_ddot = -(g / L) sin(theta) + (t . F) / (m L)
with ``t = (cos theta, 0, sin theta)`` the unit tangent of the circular path.

Contacts between neighbouring balls are resolved with a velocity-level
sequential-impulse (projected Gauss-Seidel) sweep driven by a one-sided normal
impulse plus Coulomb friction, exactly as rigid-contact solvers do.  The
impulse sweeps the contact chain once per substep, which is the discrete
counterpart of the finite-speed impulse wave travelling through the row: the
released ball stops, the interior balls barely move, and the opposite end ball
breaks away.  Penetration is removed by an impulse-free positional projection
so contacts never show persistent overlap.

The frozen ``contact.restitution`` is the coefficient of a two-body contact.
For a simultaneous multi-contact impact the dissipation of the whole contact
group is distributed over the contacts of that group rather than compounded
pair by pair (per-contact exponential compounding would be unphysical for an
already-closed chain), i.e. for a group of ``n_c`` active contacts
``e_group = 1 - (1 - e) / n_c``, which reduces to the frozen value for any
isolated pair collision.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen-scene fallbacks (used only if the scene does not expose geometry).
_FALLBACK_RADIUS = 0.23
_FALLBACK_PIVOT_Z = 2.05
_FALLBACK_THETA0 = math.radians(30.0)
_DT = 1.0e-4  # integrator substep, divides the 0.02 s sample interval
_GROUP_TOL = 1.0e-3  # metres: neighbours closer than this belong to one group
_POSITION_ITERATIONS = 4
_POSITION_BETA = 0.8
_MAX_OMEGA = 60.0


class BenchmarkSolver:
    """Deterministic rigid-pendulum Newton cradle with impulse contact waves."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        geometry = config.get("scene_geometry") or {}
        bodies = list(config.get("bodies", []))

        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._gravity = gravity.astype(float).copy()
        self._g = float(abs(gravity[2])) if gravity.shape == (3,) else 9.81

        contact = dict(config.get("contact", {}) or {})
        self._restitution = float(contact.get("restitution", 0.12))
        self._friction = float(contact.get("friction", 0.55))
        self._length_scale = float(contact.get("length_scale", 0.02))

        n = len(bodies)
        self._n = n
        self._mass = np.asarray([float(b.get("mass", 1.0)) for b in bodies], dtype=float)
        pos = np.asarray([b.get("position", [0.0, 0.0, 0.0]) for b in bodies], dtype=float)
        self._rest_x = pos[:, 0].copy()
        self._rest_y = pos[:, 1].copy()
        rest_z = float(np.mean(pos[:, 2]))
        self._body_id = np.arange(n, dtype=np.int32)

        radius = np.full(n, _FALLBACK_RADIUS, dtype=float)
        pivot_z = _FALLBACK_PIVOT_Z
        for item in geometry.get("objects", []):
            ident = str(item.get("id", ""))
            spec = item.get("geometry", {}) or {}
            if ident.startswith("ball_"):
                try:
                    idx = int(ident.split("_")[-1])
                except ValueError:
                    continue
                if 0 <= idx < n and spec.get("radius"):
                    radius[idx] = float(spec["radius"])
            elif ident == "cradle_top":
                half = spec.get("half_extents")
                trans = (item.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0])
                if half and len(half) == 3:
                    pivot_z = float(trans[2]) - float(half[2])
        self._radius = radius

        self._pivot_x = self._rest_x.copy()
        self._pivot_y = self._rest_y.copy()
        self._pivot_z = float(pivot_z)
        length = self._pivot_z - rest_z
        self._L = float(length) if length > 1.0e-6 else 1.0
        if self._L <= 0.0:
            self._L = 1.0

        self._theta = np.zeros(n, dtype=float)
        self._omega = np.zeros(n, dtype=float)
        start = int(np.argmin(self._rest_x))
        self._theta[start] = -_FALLBACK_THETA0
        self._theta0_index = start

        self._time = 0.0
        self._steps = 0
        self._accum_impulse = np.zeros((max(n - 1, 0), 3), dtype=float)
        self._interval_dt = 0.0
        self._ready = True

    # ------------------------------------------------------------- kinematics
    def _positions(self, theta: np.ndarray) -> np.ndarray:
        p = np.empty((self._n, 3), dtype=float)
        p[:, 0] = self._pivot_x + self._L * np.sin(theta)
        p[:, 1] = self._pivot_y
        p[:, 2] = self._pivot_z - self._L * np.cos(theta)
        return p

    def _tangents(self, theta: np.ndarray) -> np.ndarray:
        t = np.empty((self._n, 3), dtype=float)
        t[:, 0] = np.cos(theta)
        t[:, 1] = 0.0
        t[:, 2] = np.sin(theta)
        return t

    def _velocities(self, omega: np.ndarray, tan: np.ndarray) -> np.ndarray:
        return (self._L * omega)[:, None] * tan

    # ------------------------------------------------------------- simulation
    def _apply_impulse(self, omega, vel, tan, index, impulse) -> None:
        ti = tan[index]
        scalar = float(np.dot(impulse, ti))
        omega[index] += scalar / (self._mass[index] * self._L)
        vel[index] += (scalar / self._mass[index]) * ti

    def _step(self, h: float) -> None:
        theta = self._theta
        omega = self._omega
        mass = self._mass
        length = self._L

        omega -= (self._g / length) * np.sin(theta) * h
        np.clip(omega, -_MAX_OMEGA, _MAX_OMEGA, out=omega)

        p = self._positions(theta)
        tan = self._tangents(theta)
        vel = self._velocities(omega, tan)

        gaps = np.empty(self._n - 1, dtype=float)
        normals = np.zeros((self._n - 1, 3), dtype=float)
        for k in range(self._n - 1):
            d = p[k + 1] - p[k]
            dist = float(math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2 + float(d[2]) ** 2))
            if dist > 1.0e-12:
                normals[k] = d / dist
            else:
                normals[k, 0] = 1.0
            gaps[k] = dist - (self._radius[k] + self._radius[k + 1])

        impulse = np.zeros((self._n - 1, 3), dtype=float)
        approach = np.empty(self._n - 1, dtype=float)
        for k in range(self._n - 1):
            approach[k] = float(np.dot(vel[k + 1] - vel[k], normals[k]))

        active = gaps < _GROUP_TOL
        k = 0
        while k < self._n - 1:
            if not active[k]:
                k += 1
                continue
            start = k
            while k < self._n - 1 and active[k]:
                k += 1
            stop = k - 1
            count = stop - start + 1
            restitution = 1.0 - (1.0 - self._restitution) / float(max(1, count))
            order = sorted(range(start, stop + 1), key=lambda c: (approach[c], c))
            for c in order:
                i, j = c, c + 1
                nrm = normals[c]
                ti, tj = tan[i], tan[j]
                ni = float(np.dot(nrm, ti))
                nj = float(np.dot(nrm, tj))
                wi = ni * ni / mass[i]
                wj = nj * nj / mass[j]
                denom = wi + wj
                if denom <= 1.0e-12:
                    continue
                vrel = float(np.dot(vel[j] - vel[i], nrm))
                if vrel >= 0.0:
                    continue
                lam = -(1.0 + restitution) * vrel / denom
                if lam <= 0.0:
                    continue
                vec = lam * nrm
                self._apply_impulse(omega, vel, tan, i, -vec)
                self._apply_impulse(omega, vel, tan, j, vec)
                impulse[c] += vec
                # Coulomb friction opposing the tangential slip of the pair.
                rel = vel[j] - vel[i]
                tang = rel - float(np.dot(rel, nrm)) * nrm
                speed = float(math.sqrt(float(tang[0]) ** 2 + float(tang[1]) ** 2
                                        + float(tang[2]) ** 2))
                if speed > 1.0e-12:
                    u = tang / speed
                    wu = (float(np.dot(u, ti)) ** 2) / mass[i] + \
                         (float(np.dot(u, tj)) ** 2) / mass[j]
                    if wu > 1.0e-12:
                        lam_t = -speed / wu
                        limit = self._friction * lam
                        if lam_t < -limit:
                            lam_t = -limit
                        fvec = lam_t * u
                        self._apply_impulse(omega, vel, tan, i, -fvec)
                        self._apply_impulse(omega, vel, tan, j, fvec)
                        impulse[c] += fvec

        theta += omega * h

        # Impulse-free positional projection: remove any residual overlap while
        # keeping every ball exactly on its circular pendulum path.
        for _ in range(_POSITION_ITERATIONS):
            p = self._positions(theta)
            penetration = False
            for k in range(self._n - 1):
                d = p[k + 1] - p[k]
                dist = float(math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2
                                       + float(d[2]) ** 2))
                if dist <= 1.0e-12:
                    continue
                gap = dist - (self._radius[k] + self._radius[k + 1])
                if gap >= 0.0:
                    continue
                penetration = True
                nrm = d / dist
                ti, tj = tan[k], tan[k + 1]
                ni = float(np.dot(nrm, ti))
                nj = float(np.dot(nrm, tj))
                wi = ni * ni / mass[k]
                wj = nj * nj / mass[k + 1]
                denom = wi + wj
                if denom <= 1.0e-12:
                    continue
                corr = _POSITION_BETA * (-gap) / denom
                theta[k] -= corr * ni / (mass[k] * length)
                theta[k + 1] += corr * nj / (mass[k + 1] * length)
                p[k] = self._positions(theta[k:k + 1])[0]
                p[k + 1] = self._positions(theta[k + 1:k + 2])[0]
            if not penetration:
                break

        np.clip(theta, -0.5 * math.pi, 0.5 * math.pi, out=theta)
        self._accum_impulse += impulse
        self._interval_dt += h
        self._steps += 1

    # ------------------------------------------------------------------ output
    def _frame(self, time: float) -> StateFrame:
        theta = self._theta
        tan = self._tangents(theta)
        position = self._positions(theta)
        velocity = self._velocities(self._omega, tan)

        orientation = np.zeros((self._n, 4), dtype=float)
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((self._n, 3), dtype=float)

        gaps = np.empty(self._n - 1, dtype=float)
        for k in range(self._n - 1):
            d = position[k + 1] - position[k]
            dist = float(math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2 + float(d[2]) ** 2))
            gaps[k] = dist - (self._radius[k] + self._radius[k + 1])

        external_force = self._mass[:, None] * self._gravity[None, :]
        # Paired contact forces, averaged over the sampling interval so that an
        # impact occurring between two sample times is still recorded.
        dt = self._interval_dt if self._interval_dt > 0.0 else _DT
        interaction_force = self._accum_impulse / dt
        self._accum_impulse = np.zeros_like(self._accum_impulse)
        self._interval_dt = 0.0

        return StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self._mass.copy(),
            body_id=self._body_id,
            attributes={},
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": gaps,
            },
            scalars={"contact_length_scale": self._length_scale},
        )

    # --------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        if target < self._time - 1.0e-9:
            target = self._time
        steps = int(math.floor((target - self._time) / _DT + 1.0e-9))
        for _ in range(steps):
            self._step(_DT)
        self._time += steps * _DT
        remainder = target - self._time
        if remainder > 1.0e-12:
            self._step(remainder)
            self._time += remainder
        if self._time < 0.0:
            self._time = 0.0
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pendulum_impulse_chain",
            "substep": _DT,
            "steps": int(self._steps),
            "sim_time": float(self._time),
            "bodies": int(self._n),
            "pendulum_length": float(self._L),
            "displaced_body_index": int(self._theta0_index),
            "initial_angle": float(self._theta[self._theta0_index]),
        }
