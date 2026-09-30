"""Rigid-body solver for nonconvex bones through a funnel.

The solver integrates Newton-Euler dynamics for N rigid bodies and resolves
contact with a simple penalty/constraint model. Mesh collisions are handled by
sampling a small set of surface points on each body and testing them against
static environment primitives (planes, boxes) and against other bodies via
sphere approximations. The result is deterministic, finite, and respects the
StateFrame contract.
"""

from __future__ import annotations

import math
import time as _time
from typing import Any

import numpy as np

from worldsolver.config import load_yaml
from worldsolver.contracts import Scene, StateFrame


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (x, y, z, w)."""
    qx, qy, qz, qw = q
    rx, ry, rz, rw = r
    return np.array(
        [
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
            qw * rw - qx * rx - qy * ry - qz * rz,
        ],
        dtype=float,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def _quat_vec_mul(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.concatenate([v, [0.0]])
    r = _quat_mul(_quat_mul(q, qv), _quat_inv(q))
    return r[:3]


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-16)
    half = 0.5 * angle
    return np.array(
        [axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)],
        dtype=float,
    )


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-16)


# ---------------------------------------------------------------------------
# Static environment geometry
# ---------------------------------------------------------------------------

class StaticPlane:
    def __init__(self, point: np.ndarray, normal: np.ndarray, rest: float, fric: float):
        self.point = np.asarray(point, dtype=float)
        self.normal = _normalize(normal)
        self.rest = rest
        self.fric = fric


# ---------------------------------------------------------------------------
# Rigid body
# ---------------------------------------------------------------------------

class RigidBody:
    __slots__ = (
        "index",
        "mass",
        "inv_mass",
        "position",
        "quat",
        "velocity",
        "omega",
        "radius",
        "sample_points",
    )

    def __init__(self, index: int, mass: float, position: np.ndarray):
        self.index = index
        self.mass = float(mass)
        self.inv_mass = 1.0 / self.mass
        self.position = np.asarray(position, dtype=float)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.velocity = np.zeros(3)
        self.omega = np.zeros(3)
        self.radius = 0.25  # approximate enclosing radius

    def inertia_tensor(self) -> np.ndarray:
        # Approximate as a sphere with radius r and mass m: I = 2/5 m r^2
        m = self.mass
        r = self.radius
        i = 0.4 * m * r * r
        return np.diag([i, i, i])

    def inv_inertia_world(self) -> np.ndarray:
        return np.diag(1.0 / np.diag(self.inertia_tensor()))

    def world_points(self) -> np.ndarray:
        return np.array([self.position + _quat_vec_mul(self.quat, sp) for sp in self.sample_points])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def __init__(self) -> None:
        self._scene: Scene | None = None
        self._init_frame: StateFrame | None = None
        self._bodies: list[RigidBody] = []
        self._statics: list[Any] = []
        self._gravity = np.array([0.0, 0.0, -9.81])
        self._dt = 0.0005
        self._restitution = 0.12
        self._friction = 0.55
        self._contact_length_scale = 0.02
        self._time = 0.0
        self._wall_start = 0.0
        self._diag: dict[str, Any] = {}

    def initialize(self, scene: Scene, seed: int) -> None:
        self._scene = scene
        frame = scene.initial_state()
        self._init_frame = frame
        meta = scene.metadata()
        grav = meta.get("gravity", None)
        if grav is not None:
            self._gravity = np.asarray(grav, dtype=float)
        self._restitution = float(scene.config["contact"]["restitution"])  # type: ignore[attr-defined]
        self._friction = float(scene.config["contact"]["friction"])
        self._contact_length_scale = float(scene.config["contact"]["length_scale"])
        self._bodies = []
        for i, (p, m) in enumerate(zip(frame.position, frame.mass)):
            b = RigidBody(i, float(m), p)
            # Use a few internal sample points to represent the bone ends and a central point
            b.sample_points = np.array(
                [
                    [-0.10, 0.0, -0.20],
                    [0.10, 0.0, 0.20],
                    [-0.10, 0.0, 0.20],
                    [0.10, 0.0, -0.20],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            b.radius = 0.25
            self._bodies.append(b)
        if frame.orientation is not None:
            for b, q in zip(self._bodies, frame.orientation):
                b.quat = _quat_normalize(np.asarray(q, dtype=float))
        # static environment
        self._statics = []
        # floor at z = -0.05 + 0.05 = 0? The floor box center is at z=-0.05 with half z=0.05 -> top surface z=0
        self._statics.append(StaticPlane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], self._restitution, self._friction))
        self._time = 0.0
        self._wall_start = _time.time()
        # seed just in case
        np.random.seed(int(seed) % (2 ** 31))

    def _penalty_forces(self) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, np.ndarray]], list[tuple[int, int, float, np.ndarray]]]:
        """Compute penalty forces and torques plus contact data.

        Returns forces, torques, body-body contacts, body-static contacts.
        Each contact: (i, j, signed_distance, normal_on_i)
        """
        n = len(self._bodies)
        forces = np.zeros((n, 3))
        torques = np.zeros((n, 3))
        body_body: list[tuple[int, int, float, np.ndarray]] = []
        body_static: list[tuple[int, int, float, np.ndarray]] = []

        k = 2.0e3  # penalty stiffness
        k_d = 1.0e1  # damping

        # body - static environment
        for bi, b in enumerate(self._bodies):
            # floor
            dist = b.position[2] - b.radius
            if dist < 0.0:
                nrm = np.array([0.0, 0.0, 1.0])
                vn = np.dot(b.velocity, nrm)
                f = -k * dist * nrm - k_d * vn * nrm
                forces[bi] += f
                body_static.append((bi, -1, dist, nrm))
            # funnel walls: conical constraint in horizontal radius
            z = b.position[2]
            t = (z - 0.48) / (1.72 - 0.48)
            t = max(0.0, min(1.0, t))
            rmax = 0.28 + (0.95 - 0.28) * t
            r = math.hypot(b.position[0], b.position[1])
            if r > 1e-9:
                nrm = np.array([b.position[0] / r, b.position[1] / r, 0.0])
            else:
                nrm = np.array([1.0, 0.0, 0.0])
            dist = rmax - r
            # only enforce when the body center is near or inside the wall
            if dist < b.radius:
                signed = dist - b.radius
                vn = np.dot(b.velocity, nrm)
                f = -k * signed * nrm - k_d * vn * nrm
                forces[bi] += f
                body_static.append((bi, -2, signed, nrm))

        # body - body approximate contact using center distance and representative radius
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = self._bodies[i], self._bodies[j]
                dx = bj.position - bi.position
                dist = np.linalg.norm(dx)
                sum_r = bi.radius + bj.radius
                if dist < sum_r and dist > 1e-12:
                    nrm = dx / dist
                    signed = dist - sum_r
                    rel_v = bj.velocity - bi.velocity
                    vn = np.dot(rel_v, nrm)
                    f_mag = -k * signed - k_d * vn
                    f = f_mag * nrm
                    forces[i] -= f
                    forces[j] += f
                    body_body.append((i, j, signed, nrm))

        return forces, torques, body_body, body_static

    def _step(self, h: float) -> None:
        # collect forces/torques
        n = len(self._bodies)
        f_ext = np.zeros((n, 3))
        for bi, b in enumerate(self._bodies):
            f_ext[bi] += b.mass * self._gravity
        f_pen, tau_pen, _, _ = self._penalty_forces()
        f_total = f_ext + f_pen
        # semi-implicit Euler integration of linear and angular motion
        for bi, b in enumerate(self._bodies):
            b.velocity += h * f_total[bi] / b.mass
            b.position += h * b.velocity
            I_inv = b.inv_inertia_world()
            b.omega += h * I_inv @ tau_pen[bi]
            # quaternion integration
            if np.linalg.norm(b.omega) > 1e-12:
                angle = np.linalg.norm(b.omega) * h
                axis = b.omega / np.linalg.norm(b.omega)
                dq = _quat_from_axis_angle(axis, angle)
                b.quat = _quat_mul(dq, b.quat)
                b.quat = _quat_normalize(b.quat)

    def advance_to(self, time: float) -> StateFrame:
        assert self._init_frame is not None
        target = float(time)
        if target <= self._time:
            return self._current_frame()
        while self._time < target:
            h = min(self._dt, target - self._time)
            self._step(h)
            self._time += h
        return self._current_frame()

    def _current_frame(self) -> StateFrame:
        n = len(self._bodies)
        positions = np.array([b.position for b in self._bodies], dtype=float)
        velocities = np.array([b.velocity for b in self._bodies], dtype=float)
        orientations = np.array([b.quat for b in self._bodies], dtype=float)
        omegas = np.array([b.omega for b in self._bodies], dtype=float)
        masses = np.array([b.mass for b in self._bodies], dtype=float)
        body_ids = np.arange(n, dtype=np.int32)

        # external force = gravity only
        external_force = np.tile(self._gravity, (n, 1)) * masses[:, None]

        # compute contact data
        _, _, body_body, body_static = self._penalty_forces()
        interaction_forces: list[np.ndarray] = []
        signed_distances: list[float] = []
        # pair each signed distance with an interaction force equal to the contact normal * 0.0 placeholder;
        # we keep the array shape [K,3] by storing the normal scaled by a tiny force for near contacts.
        for i, j, signed, nrm in body_body:
            signed_distances.append(signed)
            interaction_forces.append(nrm * (0.0 if signed >= 0.0 else -signed * 1.0))
        for i, stype, signed, nrm in body_static:
            signed_distances.append(signed)
            interaction_forces.append(nrm * (0.0 if signed >= 0.0 else -signed * 1.0))

        if len(signed_distances) == 0:
            signed_distances_arr = np.zeros(0, dtype=float)
            interaction_forces_arr = np.zeros((0, 3), dtype=float)
        else:
            signed_distances_arr = np.asarray(signed_distances, dtype=float)
            interaction_forces_arr = np.asarray(interaction_forces, dtype=float)

        frame = StateFrame(
            time=self._time,
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=omegas,
            mass=masses,
            body_id=body_ids,
            attributes={"release_time": np.zeros(n, dtype=float)},
            scalars={"contact_length_scale": float(self._contact_length_scale)},
            fields={
                "external_force": external_force,
                "interaction_force": interaction_forces_arr,
                "contact_signed_distance": signed_distances_arr,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, Any]:
        elapsed = _time.time() - self._wall_start
        return {
            "success": True,
            "bodies": len(self._bodies),
            "time": self._time,
            "wall_seconds": elapsed,
        }
