"""Deterministic rigid-body solver for the spolling coin decay scene.

This solver implements a small-time-step 3D rigid body integrator with a
single-point contact impulse model between a cylinder (the coin) and a
horizontal floor.  It uses only NumPy; no external physics engines are used.
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physics solver implementing the worldsolver Solver protocol."""

    def __init__(self) -> None:
        self._scene = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Quaternion helpers (quaternions stored as xyzw)
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

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
            ],
            dtype=float,
        )

    def _quat_rotate(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        p_q = np.array([p[0], p[1], p[2], 0.0], dtype=float)
        return self._quat_mul(q, self._quat_mul(p_q, self._quat_conj(q)))[:3]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene

        init = scene.initial_state()
        init.validate()

        # There is a single dynamic body in this scene.
        self.mass = float(init.mass[0])
        self.x = np.asarray(init.position[0], dtype=float).copy()
        self.v = np.asarray(init.velocity[0], dtype=float).copy()
        self.q = np.asarray(init.orientation[0], dtype=float).copy()
        self.q /= np.linalg.norm(self.q)
        self.omega = np.asarray(init.angular_velocity[0], dtype=float).copy()

        # Geometry from the scene specification (cylinder).
        self.radius = 0.48
        self.half_height = 0.04

        # Inertia tensor in principal (body) frame for a solid cylinder.
        m = self.mass
        R = self.radius
        h = 2.0 * self.half_height
        self.Ix = m * (3.0 * R * R + h * h) / 12.0
        self.Iz = 0.5 * m * R * R

        # Start the coin spinning rapidly around its symmetry axis.
        local_z = self._quat_rotate(self.q, np.array([0.0, 0.0, 1.0]))
        spin_rate = 22.0  # rad/s, deterministic
        self.omega += spin_rate * local_z

        # Contact / material parameters from configuration.
        metadata = scene.metadata()
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.mu = float(metadata.get("friction", 0.55))
        self.restitution = 0.12
        self.contact_length_scale = 0.02

        # Fixed time step for physics (0.0005 s).
        self.dt_phys = 0.0005

        self.t = float(init.time)
        self._last_time = self.t
        self._contact_impulse = np.zeros(3, dtype=float)

        # Preserve any scene-provided attributes (e.g. release_time).
        self._attributes = {k: np.asarray(v).copy() for k, v in init.attributes.items()}

        self._initialized = True

    def diagnostics(self) -> dict:
        return {"substeps_per_sample": int(round(0.02 / self.dt_phys)) if self._initialized else 0}

    # ------------------------------------------------------------------
    # Per-substep integration
    # ------------------------------------------------------------------
    def _inertia_inv_world(self, q: np.ndarray) -> np.ndarray:
        """Return the inverse inertia tensor in world coordinates."""
        ex = self._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        ey = self._quat_rotate(q, np.array([0.0, 1.0, 0.0]))
        ez = self._quat_rotate(q, np.array([0.0, 0.0, 1.0]))
        inv_Ix = 1.0 / self.Ix
        inv_Iz = 1.0 / self.Iz
        return (
            inv_Ix * np.outer(ex, ex)
            + inv_Ix * np.outer(ey, ey)
            + inv_Iz * np.outer(ez, ez)
        )

    def _closest_point_info(self, x: np.ndarray, q: np.ndarray):
        """Compute signed gap and contact-point offset for the coin vs floor."""
        # Local axes in world coordinates.
        local_x = self._quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        local_y = self._quat_rotate(q, np.array([0.0, 1.0, 0.0]))
        local_z = self._quat_rotate(q, np.array([0.0, 0.0, 1.0]))

        # Bottom face center offset (points from COM toward floor along local_z).
        bottom_offset = -self.half_height * np.sign(local_z[2]) * local_z

        # Transverse projection of world-up onto the coin's local xy-plane.
        d = np.array([local_x[2], local_y[2]], dtype=float)
        norm_d = np.linalg.norm(d)

        if norm_d > 1e-12:
            rim_dir_xy = -d / norm_d
            rim_dir = rim_dir_xy[0] * local_x + rim_dir_xy[1] * local_y
        else:
            rim_dir = local_x

        # Contact point is the lowest point of the cylinder rim.
        r = bottom_offset + self.radius * rim_dir
        phi = x[2] + r[2]  # signed gap to z=0 floor

        return phi, r, local_x, local_y, local_z

    def _step(self, h: float) -> None:
        # 1. Apply gravity to linear velocity.
        self.v += h * self.gravity

        # 2. Contact detection and impulse response.
        phi, r, *_ = self._closest_point_info(self.x, self.q)
        n = np.array([0.0, 0.0, 1.0], dtype=float)
        Iinv = self._inertia_inv_world(self.q)

        u = self.v + np.cross(self.omega, r)
        v_n = float(np.dot(u, n))

        # Normal impulse to prevent penetration and model restitution.
        r_cross_n = np.cross(r, n)
        Dn = 1.0 / self.mass + np.dot(r_cross_n, Iinv @ r_cross_n)

        # Baumgarte position correction coefficient.
        beta = 0.2
        j_n = (-(1.0 + self.restitution) * v_n - beta * min(phi, 0.0) / h) / Dn

        impulse_total = np.zeros(3, dtype=float)
        if j_n > 0.0:
            j_vec = j_n * n
            self.v += j_vec / self.mass
            self.omega += Iinv @ np.cross(r, j_vec)
            impulse_total += j_vec

        # 3. Friction impulse (Coulomb, dynamic).
        u = self.v + np.cross(self.omega, r)
        v_t = u - n * np.dot(u, n)
        v_t_mag = np.linalg.norm(v_t)

        if v_t_mag > 1e-8:
            t_dir = v_t / v_t_mag
            r_cross_t = np.cross(r, t_dir)
            Dt = 1.0 / self.mass + np.dot(r_cross_t, Iinv @ r_cross_t)
            j_t_max = self.mu * abs(j_n)
            j_t = min(j_t_max, v_t_mag / Dt)
            if j_t > 0.0:
                jf_vec = -j_t * t_dir
                self.v += jf_vec / self.mass
                self.omega += Iinv @ np.cross(r, jf_vec)
                impulse_total += jf_vec

        # Accumulate contact impulse for field export.
        self._contact_impulse += impulse_total

        # 4. Light velocity/angular damping for energy decay.
        damping = math.exp(-0.02 * h)
        self.v *= damping
        self.omega *= damping

        # 5. Position integration.
        self.x += h * self.v

        # 6. Orientation integration with renormalisation.
        omega_quat = np.array([self.omega[0], self.omega[1], self.omega[2], 0.0], dtype=float)
        q_dot = 0.5 * self._quat_mul(omega_quat, self.q)
        self.q += h * q_dot
        norm = np.linalg.norm(self.q)
        if norm > 1e-12:
            self.q /= norm

        # 7. Hard positional correction to avoid persistent floor penetration.
        phi_after, _, *_ = self._closest_point_info(self.x, self.q)
        if phi_after < 0.0:
            self.x[2] -= phi_after

        self.t += h

    def _build_frame(self) -> StateFrame:
        dt = self.t - self._last_time
        if dt > 0.0:
            interaction_force = self._contact_impulse / dt
        else:
            interaction_force = np.zeros(3, dtype=float)

        # Current signed contact gap (coin vs floor).
        phi, *_ = self._closest_point_info(self.x, self.q)
        contact_signed = np.array([phi], dtype=float)

        external_force = self.mass * self.gravity
        external_force = external_force.reshape(1, 3)
        interaction_force = interaction_force.reshape(1, 3)

        frame = StateFrame(
            time=float(self.t),
            position=self.x.reshape(1, 3).copy(),
            velocity=self.v.reshape(1, 3).copy(),
            orientation=self.q.reshape(1, 4).copy(),
            angular_velocity=self.omega.reshape(1, 3).copy(),
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([0], dtype=np.int32),
            attributes={k: v.copy() for k, v in self._attributes.items()},
            fields={
                "contact_signed_distance": contact_signed,
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            return self._build_frame()

        dt = time - self.t
        self._contact_impulse = np.zeros(3, dtype=float)

        if dt > 0.0:
            n_sub = max(1, int(round(dt / self.dt_phys)))
            h = dt / n_sub
            for _ in range(n_sub):
                self._step(h)

        self._last_time = self.t
        return self._build_frame()
