"""Physically-based rigid-body solver for the ``011_inclined_block`` scene.

Physical model
--------------
The scene contains one static inclined plane (a thin box tilted 30 degrees about
the world x axis) and one dynamic rigid block that starts slightly above the
plane surface, out of contact.  The block is a genuine rigid body: it has mass,
a full box inertia tensor, a quaternion orientation and an angular velocity.

The simulation performs an actual fixed-step time integration:

* free fall under uniform gravity while the block is out of contact;
* vertex/half-space contact detection between the block's eight corners and the
  plane's upper face (the plane is treated as a bounded half space);
* a velocity-level projected Gauss-Seidel (sequential impulse) contact solve
  with a speculative non-penetration constraint (restitution 0) and Coulomb
  friction with a static/kinetic coefficient larger than ``tan(theta)`` so that
  the block settles into a static resting state instead of creeping;
* quaternion integration by the exponential map and a small positional
  projection that removes residual penetration without adding kinetic energy.

All quantities are deterministic: no randomness, no keyframes, no precomputed
motion tables.  Re-running the same configuration yields identical samples.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------- #
# Frozen scene constants (taken verbatim from the authoritative scene geometry)
# --------------------------------------------------------------------------- #
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

PLANE_HALF_EXTENTS = np.array([3.2, 2.0, 0.08], dtype=float)
PLANE_TRANSLATION = np.zeros(3, dtype=float)
PLANE_QUAT_XYZW = np.array([-0.258819, 0.0, 0.0, 0.965926], dtype=float)

BLOCK_HALF_EXTENTS = np.array([0.45, 0.35, 0.3], dtype=float)
BLOCK_MASS = 2.0
BLOCK_BODY_ID = 1

# Contact / integration parameters.
FRICTION_COEFFICIENT = 0.8        # static = kinetic, safely above tan(30 deg)
CONTACT_MARGIN = 0.02             # m, detection distance for near-contact verts
POSITION_SLOP = 1.0e-4            # m, tolerated penetration kept by the solver
SOLVER_ITERATIONS = 24            # PGS sweeps per micro-step
SIM_DT = 5.0e-4                   # s, fixed micro-step (40 per 0.02 s sample)

# Frozen normalization scales reported in ``scalars``.
CONSTRAINT_SCALE = float(BLOCK_HALF_EXTENTS[2])          # 0.30 m
CONTACT_LENGTH_SCALE = float(PLANE_HALF_EXTENTS[2])      # 0.08 m (plane thickness)


# --------------------------------------------------------------------------- #
# Small quaternion helpers (scalar-first xyzw convention)
# --------------------------------------------------------------------------- #
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix for the unit quaternion ``q`` in (x, y, z, w) order."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n <= 1.0e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance ``q`` by a constant body angular velocity ``omega`` over ``dt``."""
    speed = float(np.linalg.norm(omega))
    if speed * dt < 1.0e-12:
        out = np.array(q, dtype=float, copy=True)
    else:
        axis = omega / speed
        half = 0.5 * speed * dt
        s = np.sin(half)
        dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=float)
        out = _quat_mul(dq, q)
    norm = float(np.linalg.norm(out))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return out / norm


class BenchmarkSolver:
    """Deterministic rigid-body simulation of a block on an inclined plane."""

    def __init__(self) -> None:
        self.dt = SIM_DT
        self.duration = 6.0
        self.mass = BLOCK_MASS

        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)

        self._steps = 0
        self._time = 0.0
        self._initialized = False

        # Static plane frame.
        self._plane_rot = _quat_to_matrix(PLANE_QUAT_XYZW)
        self._plane_normal = self._plane_rot[:, 2].copy()
        self._plane_center = PLANE_TRANSLATION.copy()

        # Block inertia tensor (solid box) in the body frame.
        hx, hy, hz = (float(v) for v in BLOCK_HALF_EXTENTS)
        m = self.mass
        inertia = np.array(
            [
                m / 3.0 * (hy * hy + hz * hz),
                m / 3.0 * (hx * hx + hz * hz),
                m / 3.0 * (hx * hx + hy * hy),
            ],
            dtype=float,
        )
        self._inv_inertia_body = np.diag(1.0 / inertia)

        # Body-frame corner vertices.
        self._local_vertices = np.array(
            [
                [sx * hx, sy * hy, sz * hz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Contract API
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        self.duration = float(config.get("duration", 6.0))

        frame = scene.initial_state()
        position = np.asarray(frame.position, dtype=float).reshape(-1, 3)
        velocity = np.asarray(frame.velocity, dtype=float).reshape(-1, 3)
        self.position = position[0].copy()
        self.velocity = velocity[0].copy() if velocity.size else np.zeros(3, dtype=float)

        orientation = getattr(frame, "orientation", None)
        if orientation is not None:
            quat = np.asarray(orientation, dtype=float).reshape(-1, 4)[0]
            norm = float(np.linalg.norm(quat))
            self.quaternion = quat / norm if norm > 1.0e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.quaternion = np.array(PLANE_QUAT_XYZW, copy=True)

        angular = getattr(frame, "angular_velocity", None)
        if angular is not None and np.asarray(angular).size:
            self.angular_velocity = np.asarray(angular, dtype=float).reshape(-1, 3)[0].copy()
        else:
            self.angular_velocity = np.zeros(3, dtype=float)

        mass = getattr(frame, "mass", None)
        if mass is not None and np.asarray(mass).size:
            value = float(np.asarray(mass, dtype=float).reshape(-1)[0])
            if value > 0.0:
                self.mass = value

        self._steps = 0
        self._time = 0.0
        self._initialized = True

    def advance_to(self, time: float):
        if not self._initialized:  # defensive: never expected by the runner
            self._initialized = True
        target_time = float(time)
        target_steps = int(round(target_time / self.dt))
        if target_steps < self._steps:
            target_steps = self._steps
        while self._steps < target_steps:
            self._step(self.dt)
            self._steps += 1
        self._time = self._steps * self.dt
        return self._make_frame(target_time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene": "011_inclined_block",
            "steps": int(self._steps),
            "time": float(self._time),
            "dt": float(self.dt),
            "friction": float(FRICTION_COEFFICIENT),
            "mass": float(self.mass),
            "speed": float(np.linalg.norm(self.velocity)),
            "angular_speed": float(np.linalg.norm(self.angular_velocity)),
        }

    # ------------------------------------------------------------------ #
    # Contact geometry
    # ------------------------------------------------------------------ #
    def _vertex_geometry(self, position: np.ndarray, rotation: np.ndarray):
        """Return world vertices and their signed gap to the plane top face."""
        world = self._local_vertices @ rotation.T + position
        rel = world - self._plane_center
        gap = rel @ self._plane_normal - float(PLANE_HALF_EXTENTS[2])
        local = rel @ self._plane_rot
        inside = (np.abs(local[:, 0]) <= PLANE_HALF_EXTENTS[0]) & (
            np.abs(local[:, 1]) <= PLANE_HALF_EXTENTS[1]
        )
        return world, gap, inside

    # ------------------------------------------------------------------ #
    # One micro-step of rigid body dynamics
    # ------------------------------------------------------------------ #
    def _step(self, dt: float) -> None:
        mass = self.mass
        position = self.position
        velocity = self.velocity + GRAVITY * dt
        omega = self.angular_velocity
        rotation = _quat_to_matrix(self.quaternion)
        inv_inertia = rotation @ self._inv_inertia_body @ rotation.T

        normal = self._plane_normal
        world, gap, inside = self._vertex_geometry(position, rotation)
        active = np.nonzero(inside & (gap < CONTACT_MARGIN))[0]

        if active.size:
            contacts = world[active] - position
            gaps = gap[active]
            count = active.size
            lambda_n = np.zeros(count, dtype=float)
            lambda_t = np.zeros((count, 3), dtype=float)

            for _ in range(SOLVER_ITERATIONS):
                for k in range(count):
                    r = contacts[k]
                    r_cross_n = np.cross(r, normal)
                    k_n = 1.0 / mass + float(normal @ np.cross(inv_inertia @ r_cross_n, r))
                    if k_n <= 1.0e-12:
                        continue
                    point_velocity = velocity + np.cross(omega, r)
                    v_n = float(point_velocity @ normal)
                    # Speculative (restitution-free) non-penetration target.
                    rhs = 0.0 if gaps[k] <= 0.0 else -float(gaps[k]) / dt
                    if v_n < rhs:
                        delta = (rhs - v_n) / k_n
                        updated = max(0.0, lambda_n[k] + delta)
                        delta = updated - lambda_n[k]
                        lambda_n[k] = updated
                        if delta != 0.0:
                            impulse = delta * normal
                            velocity = velocity + impulse / mass
                            omega = omega + inv_inertia @ np.cross(r, impulse)

                    # Coulomb friction: clamp the accumulated tangential impulse.
                    point_velocity = velocity + np.cross(omega, r)
                    v_n = float(point_velocity @ normal)
                    tangent_velocity = point_velocity - v_n * normal
                    speed = float(np.linalg.norm(tangent_velocity))
                    if speed <= 1.0e-12:
                        continue
                    tangent = tangent_velocity / speed
                    r_cross_t = np.cross(r, tangent)
                    k_t = 1.0 / mass + float(tangent @ np.cross(inv_inertia @ r_cross_t, r))
                    if k_t <= 1.0e-12:
                        continue
                    candidate = lambda_t[k] - (speed / k_t) * tangent
                    limit = FRICTION_COEFFICIENT * lambda_n[k]
                    magnitude = float(np.linalg.norm(candidate))
                    if magnitude > limit:
                        candidate = candidate * (limit / magnitude) if magnitude > 0.0 else candidate
                    impulse = candidate - lambda_t[k]
                    lambda_t[k] = candidate
                    if np.any(impulse != 0.0):
                        velocity = velocity + impulse / mass
                        omega = omega + inv_inertia @ np.cross(r, impulse)

        # Integrate position / orientation.
        new_position = position + velocity * dt
        new_quaternion = _integrate_quaternion(self.quaternion, omega, dt)
        new_rotation = _quat_to_matrix(new_quaternion)

        # Positional projection: lift the block out of residual penetration.
        world_new, gap_new, inside_new = self._vertex_geometry(new_position, new_rotation)
        penetrating = inside_new & (gap_new < -POSITION_SLOP)
        if np.any(penetrating):
            deepest = float(gap_new[penetrating].min())
            new_position = new_position + normal * (-(deepest) - POSITION_SLOP)

        # Sanitize: guarantee finite, bounded state at every step.
        if not (
            np.isfinite(new_position).all()
            and np.isfinite(velocity).all()
            and np.isfinite(omega).all()
            and np.isfinite(new_quaternion).all()
        ):
            new_position = position
            velocity = np.zeros(3, dtype=float)
            omega = np.zeros(3, dtype=float)
            new_quaternion = self.quaternion

        self.position = new_position
        self.velocity = velocity
        self.angular_velocity = omega
        self.quaternion = new_quaternion

    # ------------------------------------------------------------------ #
    # Frame assembly
    # ------------------------------------------------------------------ #
    def _signed_contact_distance(self) -> float:
        rotation = _quat_to_matrix(self.quaternion)
        _, gap, inside = self._vertex_geometry(self.position, rotation)
        if np.any(inside):
            return float(gap[inside].min())
        return float(gap.min())

    def _make_frame(self, time: float) -> StateFrame:
        distance = self._signed_contact_distance()
        external_force = (self.mass * GRAVITY).reshape(1, 3)
        return StateFrame(
            float(time),
            self.position.reshape(1, 3).copy(),
            self.velocity.reshape(1, 3).copy(),
            orientation=self.quaternion.reshape(1, 4).copy(),
            angular_velocity=self.angular_velocity.reshape(1, 3).copy(),
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([BLOCK_BODY_ID], dtype=np.int64),
            fields={
                "constraint_value": np.array([[distance]], dtype=float),
                "constraint_target": np.zeros((1, 1), dtype=float),
                "contact_signed_distance": np.array([distance], dtype=float),
                "external_force": external_force.astype(float, copy=True),
            },
            scalars={
                "constraint_scale": float(CONSTRAINT_SCALE),
                "contact_length_scale": float(CONTACT_LENGTH_SCALE),
            },
        )
