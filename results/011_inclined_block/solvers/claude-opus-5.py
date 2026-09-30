"""Rigid-body solver for the inclined-block scene.

Physics
-------
A single rigid box (mass 2 kg, half extents 0.45 x 0.35 x 0.30 m) is released
slightly above a static inclined slab (30 degrees about the world x axis, i.e.
downhill along +y / -z).  The block free-falls the frozen initial gap, lands
flat on the incline, slides while kinetic friction dissipates the tangential
landing velocity and, because the friction coefficient exceeds tan(30 deg),
transitions into a permanent resting state without tumbling.

Numerics
--------
* Semi-implicit (symplectic) Euler for linear/angular momentum, fixed dt.
* Box-vs-halfspace collision detection at the eight body corners against the
  incline's oriented top face (restricted to the slab's finite extents).
* Velocity-level sequential-impulse (projected Gauss-Seidel) contact solver
  with speculative contacts, Baumgarte stabilization and a Coulomb friction
  cone coupled to the accumulated normal impulse.
* Fully deterministic: no randomness, integer step accounting, fixed dt.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

# --- fixed numerical parameters -------------------------------------------
SUBSTEP_DT = 2.0e-4          # integrator step [s]
SOLVER_ITERATIONS = 16       # PGS sweeps per step
CONTACT_MARGIN = 1.0e-2      # speculative contact activation distance [m]
BAUMGARTE_BETA = 0.25        # penetration recovery fraction per step
PENETRATION_SLOP = 1.0e-5    # tolerated penetration [m]
MAX_BIAS_SPEED = 1.0         # cap on penetration recovery speed [m/s]
RESTITUTION = 0.0            # inelastic landing (no spurious bouncing)
FRICTION_MU = 0.8            # > tan(30 deg) = 0.577 -> block comes to rest


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a unit quaternion in xyzw order."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


class BenchmarkSolver:
    """Sequential-impulse rigid-body solver honoring the frozen scene."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        geometry = config.get("scene_geometry") or {}
        objects = {item["id"]: item for item in geometry.get("objects", [])}

        plane = objects.get("inclined_plane", {})
        block = objects.get("inclined_block", {})

        plane_geom = plane.get("geometry", {})
        self.plane_half = np.asarray(
            plane_geom.get("half_extents", [3.2, 2.0, 0.08]), dtype=float
        )
        plane_tf = plane.get("transform", {})
        self.plane_center = np.asarray(plane_tf.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        plane_quat = _normalize_quat(
            np.asarray(plane_tf.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        )
        self.plane_rotation = _quat_to_matrix(plane_quat)
        # Incline top face: outward normal is the slab's local +z axis and the
        # face plane satisfies normal . x = offset.
        self.plane_normal = self.plane_rotation[:, 2].copy()
        self.plane_tangent_u = self.plane_rotation[:, 0].copy()
        self.plane_tangent_v = self.plane_rotation[:, 1].copy()
        self.plane_offset = float(
            self.plane_normal @ self.plane_center + self.plane_half[2]
        )

        block_geom = block.get("geometry", {})
        self.half_extents = np.asarray(
            block_geom.get("half_extents", [0.45, 0.35, 0.3]), dtype=float
        )

        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float)[0].copy()
        self.velocity = np.asarray(initial.velocity, dtype=float)[0].copy()
        if initial.orientation is not None:
            self.orientation = _normalize_quat(
                np.asarray(initial.orientation, dtype=float)[0].copy()
            )
        else:
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if initial.angular_velocity is not None:
            self.angular_velocity = np.asarray(initial.angular_velocity, dtype=float)[0].copy()
        else:
            self.angular_velocity = np.zeros(3, dtype=float)
        self.mass = float(np.asarray(initial.mass, dtype=float)[0]) if initial.mass is not None else 2.0
        if initial.body_id is not None:
            self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        else:
            binding = block.get("state_binding", {}) or {}
            self.body_id = np.array([int(binding.get("value", 1))], dtype=int)

        # Frozen reference orientation used by the no-tumbling constraint.
        self.reference_orientation = self.orientation.copy()

        # Solid box inertia about the center of mass, body frame.
        hx, hy, hz = self.half_extents
        self.inertia_body = (self.mass / 3.0) * np.array(
            [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy], dtype=float
        )
        self.inertia_body_inv = 1.0 / self.inertia_body

        # Body-frame corners; the four bottom corners (local -z) come first so
        # the exported contact sample ordering is fixed for every frame.
        signs = []
        for sz in (-1.0, 1.0):
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    signs.append((sx, sy, sz))
        self.corner_signs = np.asarray(signs, dtype=float)
        self.corners_body = self.corner_signs * self.half_extents
        self.contact_sample_count = 4  # exported bottom-face contact samples

        self.dt = float(SUBSTEP_DT)
        self.step_index = 0
        self.current_time = 0.0
        self.max_penetration = 0.0
        self.contact_steps = 0

    # ------------------------------------------------------------ mechanics
    def _inertia_world_inv(self, rotation: np.ndarray) -> np.ndarray:
        return (rotation * self.inertia_body_inv) @ rotation.T

    def _world_corners(self, rotation: np.ndarray) -> np.ndarray:
        return self.position + self.corners_body @ rotation.T

    def _signed_distances(self, corners: np.ndarray) -> np.ndarray:
        return corners @ self.plane_normal - self.plane_offset

    def _on_slab(self, corners: np.ndarray) -> np.ndarray:
        local = corners - self.plane_center
        u = np.abs(local @ self.plane_tangent_u)
        v = np.abs(local @ self.plane_tangent_v)
        return (u <= self.plane_half[0]) & (v <= self.plane_half[1])

    def _step(self) -> None:
        dt = self.dt
        rotation = _quat_to_matrix(self.orientation)
        inv_inertia = self._inertia_world_inv(rotation)
        inv_mass = 1.0 / self.mass

        # 1) unconstrained velocity update (gravity is the only external load)
        velocity = self.velocity + dt * GRAVITY
        omega = self.angular_velocity.copy()

        # 2) collision detection at the box corners
        corners = self._world_corners(rotation)
        distances = self._signed_distances(corners)
        active = np.nonzero((distances < CONTACT_MARGIN) & self._on_slab(corners))[0]

        if active.size:
            self.contact_steps += 1
            normal = self.plane_normal
            arms = corners[active] - self.position
            gaps = distances[active]
            # effective normal masses and per-contact bias velocities
            normal_mass = np.empty(active.size, dtype=float)
            bias = np.empty(active.size, dtype=float)
            cross_terms = np.empty((active.size, 3), dtype=float)
            for k in range(active.size):
                r = arms[k]
                rn = np.cross(r, normal)
                cross_terms[k] = rn
                normal_mass[k] = inv_mass + rn @ (inv_inertia @ rn)
                gap = gaps[k]
                if gap < -PENETRATION_SLOP:
                    bias[k] = min(-(BAUMGARTE_BETA / dt) * (gap + PENETRATION_SLOP), MAX_BIAS_SPEED)
                    if -gap > self.max_penetration:
                        self.max_penetration = float(-gap)
                elif gap > 0.0:
                    bias[k] = -gap / dt  # speculative: close the gap at most
                else:
                    bias[k] = 0.0

            # restitution target from the approach velocity before solving
            approach = np.empty(active.size, dtype=float)
            for k in range(active.size):
                approach[k] = normal @ (velocity + np.cross(omega, arms[k]))
            target = np.where(approach < -0.05, -RESTITUTION * approach, 0.0) + bias

            impulse_n = np.zeros(active.size, dtype=float)
            impulse_t = np.zeros((active.size, 3), dtype=float)

            for iteration in range(SOLVER_ITERATIONS):
                order = range(active.size) if iteration % 2 == 0 else range(active.size - 1, -1, -1)
                for k in order:
                    r = arms[k]
                    # --- normal constraint
                    point_velocity = velocity + np.cross(omega, r)
                    un = normal @ point_velocity
                    delta = -(un - target[k]) / normal_mass[k]
                    new_n = impulse_n[k] + delta
                    if new_n < 0.0:
                        new_n = 0.0
                    delta = new_n - impulse_n[k]
                    impulse_n[k] = new_n
                    if delta != 0.0:
                        p = delta * normal
                        velocity += inv_mass * p
                        omega += inv_inertia @ np.cross(r, p)

                    # --- Coulomb friction inside the accumulated normal cone
                    if impulse_n[k] <= 0.0:
                        if np.any(impulse_t[k]):
                            p = -impulse_t[k]
                            velocity += inv_mass * p
                            omega += inv_inertia @ np.cross(r, p)
                            impulse_t[k] = 0.0
                        continue
                    point_velocity = velocity + np.cross(omega, r)
                    tangential = point_velocity - (normal @ point_velocity) * normal
                    speed = float(np.linalg.norm(tangential))
                    if speed > 1e-14:
                        direction = tangential / speed
                        tm = inv_mass + (
                            np.cross(r, direction) @ (inv_inertia @ np.cross(r, direction))
                        )
                        trial = impulse_t[k] - (speed / tm) * direction
                    else:
                        trial = impulse_t[k].copy()
                    limit = FRICTION_MU * impulse_n[k]
                    magnitude = float(np.linalg.norm(trial))
                    if magnitude > limit:
                        trial = trial * (limit / magnitude) if magnitude > 1e-14 else trial * 0.0
                    dp = trial - impulse_t[k]
                    impulse_t[k] = trial
                    if np.any(dp):
                        velocity += inv_mass * dp
                        omega += inv_inertia @ np.cross(r, dp)

        # 3) state integration
        self.velocity = velocity
        self.angular_velocity = omega
        self.position = self.position + dt * velocity
        spin = _quat_multiply(np.array([omega[0], omega[1], omega[2], 0.0]), self.orientation)
        self.orientation = _normalize_quat(self.orientation + (0.5 * dt) * spin)
        self.step_index += 1
        self.current_time = self.step_index * dt

    # -------------------------------------------------------------- outputs
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        target_steps = int(round(target / self.dt))
        while self.step_index < target_steps:
            self._step()
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        rotation = _quat_to_matrix(self.orientation)
        corners = self._world_corners(rotation)
        distances = self._signed_distances(corners)
        contact_gap = np.ascontiguousarray(
            distances[: self.contact_sample_count], dtype=float
        )

        position = self.position.reshape(1, 3).copy()
        velocity = self.velocity.reshape(1, 3).copy()
        orientation = self.orientation.reshape(1, 4).copy()
        angular_velocity = self.angular_velocity.reshape(1, 3).copy()
        external_force = (self.mass * GRAVITY).reshape(1, 3).copy()

        # No-tumbling constraint: the block's orientation must stay locked to
        # the frozen incline orientation while it slides and comes to rest.
        constraint_value = orientation.copy()
        constraint_target = self.reference_orientation.reshape(1, 4).copy()

        frame = StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=np.array([self.mass], dtype=float),
            body_id=self.body_id.copy(),
            attributes={},
            fields={
                "external_force": external_force,
                "contact_signed_distance": contact_gap,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
            },
            scalars={
                "contact_length_scale": float(self.plane_half[2]),
                "constraint_scale": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        rotation = _quat_to_matrix(self.orientation)
        corners = self._world_corners(rotation)
        return {
            "method": "sequential_impulse_rigid_body",
            "substep_dt": self.dt,
            "solver_iterations": SOLVER_ITERATIONS,
            "friction_mu": FRICTION_MU,
            "restitution": RESTITUTION,
            "steps_taken": int(self.step_index),
            "contact_steps": int(self.contact_steps),
            "max_penetration": float(self.max_penetration),
            "final_position": [float(v) for v in self.position],
            "final_speed": float(np.linalg.norm(self.velocity)),
            "final_angular_speed": float(np.linalg.norm(self.angular_velocity)),
            "final_min_gap": float(np.min(self._signed_distances(corners))),
        }
