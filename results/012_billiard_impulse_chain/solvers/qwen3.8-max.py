"""Physically-based solver for the billiard impulse chain scene.

Model
-----
Six rigid spheres (one incoming cue ball + a five-ball chain) resting on a
horizontal felt plane (top of the ``ground`` box, z = 0).  The dynamics are
integrated with a small fixed substep using semi-implicit Euler plus an
impulse-based (sequential, single-pass-per-substep) collision response:

* gravity acts on every ball;
* the plane is a unilateral constraint: penetration is projected out and the
  normal velocity is removed, recording the supporting normal impulse;
* balls in plane contact obey a rolling-without-slipping constraint
  (omega is slaved to the linear velocity) and experience rolling resistance
  ``mu_r`` from the public config, giving the deceleration
  ``mu_r * g / (1 + I/(m r^2))``;
* ball-ball contacts are resolved with a restitution impulse
  (e = 0.95, inelastic below a small threshold to avoid jitter) using the
  rolling-constrained effective mass ``m (1 + I/(m r^2))``.  Because the
  substep is much smaller than the time a ball needs to cross the 1 cm gap
  between chain neighbours, the impulse propagates ball-by-ball as a contact
  wave: the cue ball stops, intermediate balls stay (almost) at rest and the
  far ball is launched and rolls away - the Newton's-cradle outcome of the
  reference figure, with energy loss at every contact.

Every StateFrame exports the required raw trajectory fields:
``fields.contact_signed_distance`` (all ball-ball gaps and ball-plane gaps,
negative = penetration), ``fields.external_force`` (gravity, per body),
``fields.interaction_force`` (time-averaged paired contact forces: force on
the first body of each ball-ball pair and the plane reaction on each ball),
``scalars.contact_length_scale`` (frozen ball diameter), plus
``state.body_id``/``state.mass``/``state.position``/``state.velocity`` (and
orientation/angular velocity for completeness).
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    SUBSTEP = 2.0e-4          # s, resolves the contact wave ball-by-ball
    RESTITUTION = 0.95        # ball-ball normal restitution
    REST_THRESHOLD = 0.05     # m/s, below this use inelastic response
    CONTACT_MARGIN = 2.0e-3   # m, near-contact band for force bookkeeping
    GROUND_Z = 0.0            # top surface of the ground box

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.pos = np.array(frame0.position, dtype=float).copy()
        self.vel = np.array(frame0.velocity, dtype=float).copy()
        self.n = int(self.pos.shape[0])

        if frame0.mass is None:
            self.mass = np.ones(self.n)
        else:
            self.mass = np.array(frame0.mass, dtype=float).copy()
        if frame0.body_id is None:
            self.body_id = np.arange(self.n, dtype=int)
        else:
            self.body_id = np.asarray(frame0.body_id).astype(int).copy()
        if frame0.angular_velocity is None:
            self.omega = np.zeros((self.n, 3))
        else:
            self.omega = np.array(frame0.angular_velocity, dtype=float).copy()
        self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))

        gravity = config.get("gravity", (0.0, 0.0, -9.81))
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.mu_roll = float(config.get("rolling_friction", 0.015))
        self.radius = self._sphere_radius(config)
        self.inertia = 0.4 * self.mass * self.radius ** 2  # solid spheres

        self.pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.n_pair = len(self.pairs)
        self.n_contacts = self.n_pair + self.n

        self.t = 0.0
        self._frame_dt = 1.0 / 60.0
        self._substeps = 0
        self._max_penetration = 0.0
        self.contact_length_scale = 2.0 * self.radius
        self._reset_accumulators()
        # Static support reaction so the t=0 frame already carries the weight
        # balance through the interaction-force field.
        self.ground_imp[:, 2] = self.mass * abs(self.gravity[2]) * self._frame_dt

    @staticmethod
    def _sphere_radius(config: dict) -> float:
        geom = config.get("scene_geometry") or {}
        for obj in geom.get("objects", []) or []:
            spec = obj.get("geometry") or {}
            if spec.get("type") == "sphere" and spec.get("radius"):
                return float(spec["radius"])
        return 0.2

    # -------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-12:
            self._reset_accumulators()
            frame_dt = target - self.t
            self._frame_dt = frame_dt
            remaining = frame_dt
            while remaining > 1e-12:
                h = self.SUBSTEP if remaining > self.SUBSTEP else remaining
                self._substep(h)
                remaining -= h
            self.t = target
        return self._make_frame(target)

    def _reset_accumulators(self) -> None:
        self.pair_imp = np.zeros(self.n_pair)
        self.pair_nrm = np.zeros((self.n_pair, 3))
        self.ground_imp = np.zeros((self.n, 3))

    def _substep(self, h: float) -> None:
        self._substeps += 1
        r = self.radius
        m = self.mass
        pos, vel = self.pos, self.vel

        # --- external force: gravity -------------------------------------
        vel += self.gravity * h

        # --- free flight integration --------------------------------------
        pos += vel * h

        # --- plane contact: projection, normal impulse, rolling friction ---
        rolling = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            gap = pos[i, 2] - (self.GROUND_Z + r)
            if gap <= 1e-9:
                rolling[i] = True
                if gap < 0.0:
                    pos[i, 2] -= gap  # remove penetration exactly
                if vel[i, 2] < 0.0:
                    self.ground_imp[i, 2] += -m[i] * vel[i, 2]
                    vel[i, 2] = 0.0
                # Rolling resistance: torque mu_r * N * r opposing the spin,
                # with the rolling constraint converting it into a linear
                # deceleration mu_r * g / (1 + I / (m r^2)).
                speed = math.hypot(vel[i, 0], vel[i, 1])
                if speed > 1e-12:
                    normal_acc = abs(self.gravity[2])
                    decel = self.mu_roll * normal_acc / (1.0 + 0.4)
                    dv = min(decel * h, speed)
                    ux, uy = vel[i, 0] / speed, vel[i, 1] / speed
                    vel[i, 0] -= dv * ux
                    vel[i, 1] -= dv * uy
                    self.ground_imp[i, 0] -= m[i] * dv * ux
                    self.ground_imp[i, 1] -= m[i] * dv * uy

        # --- ball-ball contacts (single sequential-impulse pass) ----------
        min_dist = 2.0 * r
        for k, (i, j) in enumerate(self.pairs):
            dx = pos[j] - pos[i]
            dist = float(np.sqrt(dx @ dx))
            if dist >= min_dist + self.CONTACT_MARGIN:
                continue
            if dist > 1e-12:
                nrm = dx / dist
            else:  # degenerate: separate along x
                nrm = np.array([1.0, 0.0, 0.0])
                dist = 0.0
            pen = min_dist - dist
            if pen > 0.0:
                self._max_penetration = max(self._max_penetration, pen)
                wi = 1.0 / m[i]
                wj = 1.0 / m[j]
                corr = nrm * (pen / (wi + wj))
                pos[i] -= corr * wi
                pos[j] += corr * wj
            v_rel = float((vel[j] - vel[i]) @ nrm)
            if v_rel < 0.0:
                e = self.RESTITUTION if -v_rel > self.REST_THRESHOLD else 0.0
                # rolling-constrained effective inverse masses
                inv_i = 1.0 / (m[i] * 1.4) if rolling[i] else 1.0 / m[i]
                inv_j = 1.0 / (m[j] * 1.4) if rolling[j] else 1.0 / m[j]
                J = -(1.0 + e) * v_rel / (inv_i + inv_j)
                vel[i] -= (J * inv_i) * nrm
                vel[j] += (J * inv_j) * nrm
                self.pair_imp[k] += J
                self.pair_nrm[k] = nrm

        # --- rolling constraint on spin + orientation integration ---------
        for i in range(self.n):
            if rolling[i]:
                self.omega[i, 0] = -vel[i, 1] / r
                self.omega[i, 1] = vel[i, 0] / r
                self.omega[i, 2] = 0.0
            w = self.omega[i]
            angle = float(np.sqrt(w @ w)) * h
            if angle > 1e-15:
                s = math.sin(0.5 * angle) / angle * h  # = sin(a/2)/|w|
                dq = np.array([w[0] * s, w[1] * s, w[2] * s, math.cos(0.5 * angle)])
                q = self.quat[i]
                self.quat[i] = _quat_mul(dq, q)
                self.quat[i] /= np.linalg.norm(self.quat[i])

    # --------------------------------------------------------------- output
    def _make_frame(self, time: float) -> StateFrame:
        dt = max(self._frame_dt, 1e-12)
        r = self.radius
        signed = np.empty(self.n_contacts)
        forces = np.zeros((self.n_contacts, 3))

        for k, (i, j) in enumerate(self.pairs):
            dx = self.pos[j] - self.pos[i]
            dist = float(np.sqrt(dx @ dx))
            signed[k] = dist - 2.0 * r
            if dist > 1e-12:
                nrm = dx / dist
            else:
                nrm = np.array([1.0, 0.0, 0.0])
            if self.pair_imp[k] != 0.0:
                nrm = self.pair_nrm[k]
            # force applied to the first body of the pair
            forces[k] = -(self.pair_imp[k] / dt) * nrm

        for i in range(self.n):
            idx = self.n_pair + i
            signed[idx] = self.pos[i, 2] - (self.GROUND_Z + r)
            forces[idx] = self.ground_imp[i] / dt

        external = self.mass[:, None] * self.gravity[None, :]

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": signed,
                "external_force": external,
                "interaction_force": forces,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "impulse-based rigid spheres, rolling constraint + rolling resistance",
            "substeps": int(self._substeps),
            "max_penetration": float(self._max_penetration),
            "restitution": self.RESTITUTION,
            "rolling_friction": self.mu_roll,
        }


def _quat_mul(a, b):
    """Hamilton product of quaternions in (x, y, z, w) order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])
