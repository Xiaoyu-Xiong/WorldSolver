"""Deterministic rigid-body solver for the circular domino scene.

The dominoes are integrated as rigid rectangular pendula about their lower
forward edges. This is the usual reduced-coordinate model for a domino while
it is supported by a rough floor. Neighbouring dominoes interact through a
compliant unilateral contact law; no fall times or poses are prescribed.
"""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw,
         aw * bw - ax * bx - ay * by - az * bz],
        dtype=float,
    )


class BenchmarkSolver:
    """Small, self-contained time-stepper for the circular domino chain."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        initial = scene.initial_state()

        self.initial_position = np.asarray(initial.position, dtype=float).copy()
        self.initial_orientation = np.asarray(initial.orientation, dtype=float).copy()
        qnorm = np.linalg.norm(self.initial_orientation, axis=1)
        self.initial_orientation /= np.maximum(qnorm[:, None], 1.0e-15)
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.mass)

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.friction = float(cfg.get("contact", {}).get("friction", 0.55))
        self.restitution = float(cfg.get("contact", {}).get("restitution", 0.12))
        self.contact_scale = float(cfg.get("contact", {}).get("length_scale", 0.02))

        # Frozen box dimensions are 0.14 x 0.40 x 1.10 m. The toppling
        # section uses the thin dimension in the direction of chain travel.
        self.half_thickness = 0.07
        self.half_height = 0.55
        self.height = 2.0 * self.half_height

        xy = self.initial_position[:, :2]
        radius = np.linalg.norm(xy, axis=1)
        self.radial = xy / np.maximum(radius[:, None], 1.0e-15)
        self.tangent = np.column_stack((-self.radial[:, 1], self.radial[:, 0]))
        theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        if self.n > 1:
            steps = np.diff(np.r_[theta, theta[0] + 2.0 * math.pi])
            self.spacing = float(np.mean(radius) * np.mean(steps))
        else:
            self.spacing = 1.0

        full_t = 2.0 * self.half_thickness
        full_h = self.height
        self.inertia = self.mass * (full_t * full_t + full_h * full_h) / 3.0

        self.angle = np.zeros(self.n, dtype=float)
        self.omega = np.zeros(self.n, dtype=float)
        self.alpha_accel = np.zeros(self.n, dtype=float)
        self.time = 0.0

        # A short top force is the physical trigger, rather than an imposed
        # velocity or prescribed rotation.
        self.trigger_force = 6.0
        self.trigger_duration = 0.085
        self.contact_stiffness = 2600.0
        self.contact_damping = 13.0
        self.hinge_damping = 0.030 + 0.025 * self.friction
        self.internal_dt = 0.001

        self._last_gap = np.full(self.n, self.spacing - 2.0 * self.half_thickness)
        self._last_lambda = np.zeros(self.n, dtype=float)
        self._last_external = self._external_forces(0.0)
        self._steps = 0

    def _contact_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return neighbour gaps and their two generalized derivatives."""
        a = self.angle
        b = np.roll(a, -1)
        h = self.half_thickness
        height = self.height

        # From sender's forward top corner to receiver's back-bottom corner
        # in the locally unwrapped (arc length, height) plane.
        r_s = self.spacing - 2.0 * h * np.cos(b) - height * np.sin(a)
        r_z = 2.0 * h * np.sin(b) - height * np.cos(a)
        n_s = np.cos(b)
        n_z = -np.sin(b)
        gap = r_s * n_s + r_z * n_z

        dg_sender = -height * (np.cos(a) * n_s + np.sin(a) * n_z)
        drs_b = 2.0 * h * np.sin(b)
        drz_b = 2.0 * h * np.cos(b)
        dns_b = -np.sin(b)
        dnz_b = -np.cos(b)
        dg_receiver = drs_b * n_s + drz_b * n_z + r_s * dns_b + r_z * dnz_b
        return gap, dg_sender, dg_receiver, b

    def _external_forces(self, t: float) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :]
        if t < self.trigger_duration and self.n:
            force[0, :2] += self.trigger_force * self.tangent[0]
        return force

    def _substep(self, dt: float) -> None:
        a = self.angle
        w = self.omega
        h = self.half_thickness
        zc = self.half_height
        gmag = max(0.0, -float(self.gravity[2]))

        # z_cm = h sin(a) + zc cos(a) about the supporting edge.
        torque = self.mass * gmag * (zc * np.sin(a) - h * np.cos(a))
        torque -= self.hinge_damping * w

        if self.time < self.trigger_duration and self.n:
            torque[0] += self.trigger_force * self.height * max(0.0, math.cos(a[0]))

        gap, dg_i, dg_j, _ = self._contact_kinematics()
        gap_rate = dg_i * w + dg_j * np.roll(w, -1)
        lam = np.maximum(0.0, -self.contact_stiffness * gap - self.contact_damping * gap_rate)

        # Contact virtual work is lambda * delta(gap).
        torque += lam * dg_i
        torque += np.roll(lam * dg_j, 1)

        accel = torque / self.inertia
        w += dt * accel
        a += dt * w

        # Unilateral rough-floor support at the broad base and broad face.
        below = a < 0.0
        if np.any(below):
            a[below] = 0.0
            w[below] = np.maximum(0.0, w[below])
        limit = 0.5 * math.pi
        hit_floor = a > limit
        if np.any(hit_floor):
            a[hit_floor] = limit
            incoming = w[hit_floor]
            w[hit_floor] = np.where(incoming > 0.20,
                                    -self.restitution * incoming, 0.0)
        settled = ((a < 1.0e-7) | (a > limit - 1.0e-7)) & (np.abs(w) < 0.08)
        w[settled] = 0.0

        self.alpha_accel = accel
        self._last_gap = gap
        self._last_lambda = lam
        self.time += dt
        self._steps += 1

    def _state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = self.angle
        h = self.half_thickness
        zc = self.half_height
        ds = (-h * np.cos(a) + zc * np.sin(a)) + h
        dz = h * np.sin(a) + zc * np.cos(a) - zc
        dds = (h * np.sin(a) + zc * np.cos(a)) * self.omega
        ddz = (h * np.cos(a) - zc * np.sin(a)) * self.omega

        position = self.initial_position.copy()
        position[:, :2] += self.tangent * ds[:, None]
        position[:, 2] += dz
        velocity = np.zeros((self.n, 3), dtype=float)
        velocity[:, :2] = self.tangent * dds[:, None]
        velocity[:, 2] = ddz

        orientation = np.empty((self.n, 4), dtype=float)
        angular_velocity = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            half = -0.5 * a[i]
            axis = self.radial[i]
            delta = np.array([axis[0] * math.sin(half),
                              axis[1] * math.sin(half),
                              0.0, math.cos(half)])
            orientation[i] = _quat_mul(delta, self.initial_orientation[i])
            orientation[i] /= max(np.linalg.norm(orientation[i]), 1.0e-15)
            angular_velocity[i, :2] = -self.omega[i] * axis
        return position, velocity, orientation, angular_velocity

    def _interaction_forces(self) -> np.ndarray:
        """Neighbour contact forces followed by supporting floor reactions."""
        _, _, _, receiver_angle = self._contact_kinematics()
        pair_force = np.zeros((self.n, 3), dtype=float)
        receiver = np.roll(np.arange(self.n), -1)
        horizontal = self._last_lambda * np.cos(receiver_angle)
        pair_force[:, :2] = self.tangent[receiver] * horizontal[:, None]
        pair_force[:, 2] = -self._last_lambda * np.sin(receiver_angle)

        floor_force = np.zeros((self.n, 3), dtype=float)
        floor_force[:, 2] = self.mass * max(0.0, -float(self.gravity[2]))
        return np.vstack((pair_force, floor_force))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")

        while self.time < target - 1.0e-12:
            self._substep(min(self.internal_dt, target - self.time))
        self.time = target

        position, velocity, orientation, angular_velocity = self._state_arrays()
        external = self._external_forces(target)
        self._last_external = external
        contact_distance = np.concatenate((self._last_gap.copy(), np.zeros(self.n)))
        frame = StateFrame(
            target,
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": contact_distance,
                "external_force": external.copy(),
                "interaction_force": self._interaction_forces(),
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_reduced_rigid_body",
            "internal_steps": int(self._steps),
            "time": float(self.time),
            "maximum_angle": float(np.max(self.angle)) if self.n else 0.0,
            "active_contacts": int(np.count_nonzero(self._last_lambda > 0.0)),
        }
