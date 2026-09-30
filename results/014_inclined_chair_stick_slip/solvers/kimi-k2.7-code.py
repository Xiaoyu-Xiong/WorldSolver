"""Physical time-stepping solver for the inclined-chair stick-slip scene.

The chair is modelled as a single rigid body whose four leg tips interact with the
inclined plane through compliant normal contacts and Coulomb friction.  Spatial
variation of the friction coefficient along the slope produces repeated stick-slip
cycles, while the under-damped normal compliance generates visible chatter.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit semi-implicit Euler solver for a rigid chair on an incline."""

    def __init__(self) -> None:
        pass

    # --------------------------------------------------------------------- #
    # Solver protocol
    # --------------------------------------------------------------------- #
    def initialize(self, scene: Any, seed: int) -> None:
        self.seed = seed

        # Physical constants and body parameters.
        self.mass = 10.8
        self.g = 9.81
        self.gravity = np.array([0.0, 0.0, -self.g])

        # Initial state from the scene.
        init = scene.initial_state()
        self.time = 0.0
        self.position = init.position[0].astype(float, copy=True)
        self.velocity = init.velocity[0].astype(float, copy=True)
        self.orientation = init.orientation[0].astype(float, copy=True)
        self.orientation /= np.linalg.norm(self.orientation)
        self.angular_velocity = init.angular_velocity[0].astype(float, copy=True)
        self.body_id = int(init.body_id[0]) if init.body_id is not None else 0

        # Fixed world-space plane normal (the incline does not move).
        self.plane_normal = self._rotate_vector(self.orientation, np.array([0.0, 0.0, 1.0]))
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        # The actual top surface of the inclined box, used as the contact reference.
        box_center = np.array([0.0, 0.0, 0.35])
        self.plane_anchor = box_center + 0.08 * self.plane_normal

        # COM-to-foot distance in the normal direction.
        self.h = float(np.dot(self.position - self.plane_anchor, self.plane_normal))
        if self.h < 0.1:
            self.h = 0.5

        # Body-frame foot offsets: x = downhill, y = lateral, z = normal (into chair).
        a = 0.25
        b = 0.25
        self.foot_local = np.array(
            [
                [a, b, -self.h],
                [a, -b, -self.h],
                [-a, b, -self.h],
                [-a, -b, -self.h],
            ],
            dtype=float,
        )

        # Contact parameters (SI units).  Stiffness chosen so static compression is
        # sub-millimetre; damping is slightly under-critical to give visible chatter.
        self.k_n = 1e5
        self.c_n = 600.0

        # Friction coefficients with deterministic spatial roughness along the slope.
        # The average static coefficient straddles tan(theta), giving repeated
        # stick-slip as the chair slides over high- and low-friction patches.
        self.mu_base = 0.30
        self.mu_amp = 0.10
        self.lambda_rough = 0.20
        self.mu_dynamic_offset = 0.08
        self.v_tol = 1e-3

        # Approximate body inertia as a uniform box with half-extents (a, b, h).
        hx, hy, hz = a, b, self.h
        ix = self.mass / 3.0 * (hy * hy + hz * hz)
        iy = self.mass / 3.0 * (hx * hx + hz * hz)
        iz = self.mass / 3.0 * (hx * hx + hy * hy)
        self.inertia_body = np.array([ix, iy, iz])

        # Time integration.
        self.substep = 1e-4

        # Cached contact data for the most recent substep, used to populate fields.
        self._last_signed_distance = np.zeros(4, dtype=float)
        self._last_constraint_value = np.zeros((4, 2), dtype=float)
        self._last_external_force = self.mass * self.gravity

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            raise ValueError("advance_to must be called with non-decreasing times")

        while self.time < target:
            dt = min(self.substep, target - self.time)
            self._step(dt)
            self.time += dt

        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep": self.substep,
            "contact_stiffness": self.k_n,
            "contact_damping": self.c_n,
            "mu_base": self.mu_base,
        }

    # --------------------------------------------------------------------- #
    # Integration helpers
    # --------------------------------------------------------------------- #
    def _body_axes(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world-space body axes (x, y, z) for the given quaternion."""
        ex = self._rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        ey = self._rotate_vector(q, np.array([0.0, 1.0, 0.0]))
        ez = self._rotate_vector(q, np.array([0.0, 0.0, 1.0]))
        return ex, ey, ez

    def _rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by unit quaternion q (xyzw)."""
        qxyz = q[:3]
        qw = q[3]
        t = 2.0 * np.cross(qxyz, v)
        return v + qw * t + np.cross(qxyz, t)

    def _quat_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ]
        )

    def _inertia_world(self, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> np.ndarray:
        """Diagonal body inertia transformed to world coordinates."""
        ix, iy, iz = self.inertia_body
        return (
            ix * np.outer(ex, ex)
            + iy * np.outer(ey, ey)
            + iz * np.outer(ez, ez)
        )

    def _friction_coefficients(self, s: float) -> tuple[float, float]:
        """Static and dynamic friction coefficients at downhill coordinate s."""
        mu_s = self.mu_base + self.mu_amp * math.sin(2.0 * math.pi * s / self.lambda_rough)
        mu_s = max(0.05, mu_s)
        mu_d = max(0.05, mu_s - self.mu_dynamic_offset)
        return mu_s, mu_d

    def _step(self, dt: float) -> None:
        # Body axes and foot offsets in world coordinates.
        ex, ey, ez = self._body_axes(self.orientation)
        R = np.stack([ex, ey, ez], axis=1)  # world = R @ local
        r_world = self.foot_local @ R.T  # shape (4, 3)

        # Signed distances of the four foot tips to the plane.
        foot_pos = self.position + r_world
        signed_distance = (foot_pos - self.plane_anchor) @ self.plane_normal

        # Velocities of the foot tips.
        v_foot = self.velocity + np.cross(self.angular_velocity, r_world)
        v_normal = np.dot(v_foot, self.plane_normal)
        v_tangent = v_foot - v_normal[:, None] * self.plane_normal

        # Normal contact forces (unilateral spring-damper).
        f_n_mag = np.maximum(0.0, -self.k_n * signed_distance - self.c_n * v_normal)
        total_normal = float(np.sum(f_n_mag))

        # Tangential friction.
        # Use the COM downhill coordinate to sample the spatially varying friction.
        downhill = self._rotate_vector(self.orientation, np.array([1.0, 0.0, 0.0]))
        downhill /= np.linalg.norm(downhill)
        s_coord = float(np.dot(self.position - self.plane_anchor, downhill))
        mu_s, mu_d = self._friction_coefficients(s_coord)

        # Net non-frictional tangential force (mainly gravity component along slope).
        f_normal_total = self.plane_normal * total_normal
        f_non_fric = self.mass * self.gravity + f_normal_total
        f_t_non = f_non_fric - np.dot(f_non_fric, self.plane_normal) * self.plane_normal

        # COM tangential velocity.
        v_t_com = self.velocity - np.dot(self.velocity, self.plane_normal) * self.plane_normal
        v_t_mag = np.linalg.norm(v_t_com)

        if v_t_mag > self.v_tol:
            # Sliding: kinetic friction opposes the COM tangential velocity.
            f_fric_total = -mu_d * total_normal * (v_t_com / v_t_mag)
        else:
            # Sticking: try to cancel the tangential force.
            needed = -f_t_non
            needed_mag = np.linalg.norm(needed)
            if total_normal > 0.0 and needed_mag <= mu_s * total_normal:
                f_fric_total = needed
            else:
                # Friction insufficient; kinetic friction opposes the driving force.
                if needed_mag > 0.0:
                    f_fric_total = -mu_d * total_normal * (needed / needed_mag)
                else:
                    f_fric_total = np.zeros(3)

        # Distribute total friction to the four feet proportionally to their normal load.
        if total_normal > 0.0:
            weights = f_n_mag / total_normal
            f_fric = f_fric_total[None, :] * weights[:, None]
        else:
            f_fric = np.zeros((4, 3))

        # Total force and torque on the body.
        total_force = f_non_fric + f_fric_total
        contact_forces = f_n_mag[:, None] * self.plane_normal[None, :] + f_fric
        torque = np.sum(np.cross(r_world, contact_forces), axis=0)

        # Add a tiny angular damping to suppress unphysical high-frequency spin.
        torque -= 0.1 * self.angular_velocity

        # Linear acceleration.
        accel = total_force / self.mass

        # Angular acceleration using the world-frame inertia tensor.
        I_world = self._inertia_world(ex, ey, ez)
        I_omega = I_world @ self.angular_velocity
        alpha = np.linalg.solve(I_world, torque - np.cross(self.angular_velocity, I_omega))

        # Semi-implicit Euler update.
        self.velocity += accel * dt
        self.angular_velocity += alpha * dt
        self.position += self.velocity * dt

        # Quaternion update with angular velocity in world frame.
        omega_quat = np.array([self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2], 0.0])
        q_dot = 0.5 * self._quat_mul(omega_quat, self.orientation)
        self.orientation += q_dot * dt
        self.orientation /= np.linalg.norm(self.orientation)

        # Cache contact fields for the trajectory writer.
        self._last_signed_distance = signed_distance.astype(float)

        # Constraint channels per contact:
        #   channel 0 : friction-cone residual  |f_t| - mu_s * f_n
        #   channel 1 : tangential speed magnitude
        f_t_mag = np.linalg.norm(v_tangent, axis=1)
        residual = f_t_mag - mu_s * f_n_mag
        self._last_constraint_value = np.stack([residual, f_t_mag], axis=1)

        self._last_external_force = self.mass * self.gravity

    def _build_frame(self) -> StateFrame:
        return StateFrame(
            time=self.time,
            position=self.position[None, :].copy(),
            velocity=self.velocity[None, :].copy(),
            orientation=self.orientation[None, :].copy(),
            angular_velocity=self.angular_velocity[None, :].copy(),
            mass=np.array([self.mass]),
            body_id=np.array([self.body_id], dtype=int),
            fields={
                "constraint_target": np.zeros_like(self._last_constraint_value),
                "constraint_value": self._last_constraint_value.copy(),
                "contact_signed_distance": self._last_signed_distance.copy(),
                "external_force": self._last_external_force[None, :].copy(),
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": 0.25,
            },
        )
