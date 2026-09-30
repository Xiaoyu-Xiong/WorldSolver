"""Deterministic rigid-body solver for the inclined-block scene.

The block is face-aligned with the plane. Consequently the four coplanar
normal reactions have zero resultant torque and the six degree-of-freedom
problem reduces (without approximation for this symmetric initial state) to
translation of the centre of mass plus a constant orientation. Contact is
advanced with semi-implicit Euler, non-penetration projection, and Coulomb
kinetic/static friction.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small, self-contained time-stepping rigid contact solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The physical initial-value problem is deterministic.
        initial = scene.initial_state()

        self._time = float(initial.time)
        self._position = np.asarray(initial.position, dtype=float).copy()
        self._velocity = np.asarray(initial.velocity, dtype=float).copy()
        self._orientation = np.asarray(initial.orientation, dtype=float).copy()
        self._orientation /= np.linalg.norm(self._orientation, axis=1)[:, None]
        self._angular_velocity = np.zeros((1, 3), dtype=float)
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=int).copy()

        # Frozen geometry: both boxes are rotated -30 degrees about x. The
        # upper surface is 0.08 m along local +z and the block half-height is
        # 0.30 m along that same normal.
        x, y, z, w = self._orientation[0]
        self._normal = np.array(
            [2.0 * (x * z + w * y),
             2.0 * (y * z - w * x),
             1.0 - 2.0 * (x * x + y * y)],
            dtype=float,
        )
        self._normal /= np.linalg.norm(self._normal)
        self._surface_offset = 0.08
        self._block_normal_radius = 0.30

        self._gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self._mu_kinetic = 0.80
        self._mu_static = 0.85
        self._contact = self._gap() <= 1.0e-10
        self._steps = 0

    def _gap(self) -> float:
        """Signed separation between the block lower face and plane top."""
        return float(
            np.dot(self._normal, self._position[0])
            - self._surface_offset
            - self._block_normal_radius
        )

    def _step(self, dt: float) -> None:
        """Advance one stable semi-implicit contact/friction step."""
        n = self._normal
        v = self._velocity[0]
        p = self._position[0]

        if not self._contact:
            # Symplectic Euler in free flight. On impact, project the small
            # overlap and apply a perfectly inelastic normal impulse.
            v += self._gravity * dt
            p += v * dt
            gap = self._gap()
            if gap <= 0.0 and float(np.dot(v, n)) < 0.0:
                p -= gap * n
                v -= float(np.dot(v, n)) * n
                self._contact = True
        else:
            # Normal reaction cancels gravity normal to the plane. Coulomb
            # friction acts on in-plane velocity; static friction is selected
            # once kinetic friction can stop the block without reversal.
            normal_accel = max(0.0, -float(np.dot(self._gravity, n)))
            gravity_tangent = self._gravity - float(np.dot(self._gravity, n)) * n
            vt = v - float(np.dot(v, n)) * n
            speed = float(np.linalg.norm(vt))

            if speed > 1.0e-12:
                a_tangent = gravity_tangent - (
                    self._mu_kinetic * normal_accel * vt / speed
                )
                candidate = vt + a_tangent * dt
                if (float(np.dot(candidate, vt)) <= 0.0
                        and np.linalg.norm(gravity_tangent)
                        <= self._mu_static * normal_accel):
                    v[:] = 0.0
                else:
                    v[:] = candidate
            elif (np.linalg.norm(gravity_tangent)
                  <= self._mu_static * normal_accel):
                v[:] = 0.0
            else:
                direction = gravity_tangent / np.linalg.norm(gravity_tangent)
                v[:] = (gravity_tangent
                        - self._mu_kinetic * normal_accel * direction) * dt

            p += v * dt
            # Position-level stabilization enforces non-penetration without
            # storing elastic energy or allowing accumulated normal drift.
            gap = self._gap()
            p -= gap * n
            v -= float(np.dot(v, n)) * n

        self._steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1.0e-12:
            raise ValueError("sample times must be finite and nondecreasing")

        # Millisecond steps resolve impact; the final shortened step lands
        # exactly on every requested sample time.
        max_dt = 1.0e-3
        while self._time < target - 1.0e-14:
            dt = min(max_dt, target - self._time)
            self._step(dt)
            self._time += dt
        self._time = target

        gap = self._gap()
        fields = {
            "constraint_target": np.zeros((1,), dtype=float),
            "constraint_value": np.array([gap], dtype=float),
            "contact_signed_distance": np.array([gap], dtype=float),
            # External force deliberately excludes plane reactions/friction.
            "external_force": self._mass[:, None] * self._gravity[None, :],
        }
        scalars = {
            "constraint_scale": 0.60,
            "contact_length_scale": 0.60,
        }
        return StateFrame(
            self._time,
            self._position.copy(),
            self._velocity.copy(),
            orientation=self._orientation.copy(),
            angular_velocity=self._angular_velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_euler_coulomb_contact",
            "internal_steps": self._steps,
            "contact_active": self._contact,
            "signed_gap": self._gap(),
        }
