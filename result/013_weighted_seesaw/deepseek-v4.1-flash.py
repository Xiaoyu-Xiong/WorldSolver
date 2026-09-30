"""Rigid-body solver for the weighted seesaw scene (013_weighted_seesaw).

Bodies
------
* body 0 : ``seesaw_board``  - mass 4 kg box pivoted (revolute, world +y axis)
  on the static pivot cylinder.  Only its tilt angle is free.
* body 1 : ``heavy_block``   - 3 kg box, initially right of the pivot.
* body 2 : ``light_block``   - 1 kg box, initially left of the pivot.

Physics
-------
Gravity plus non-penetration / Coulomb-friction contacts between every block
corner and the board box, the blocks and the ground plane, the board corners
and the ground plane, and block/block pairs.  Contacts are resolved with a
projected Gauss-Seidel sequential impulse solver on a fixed physical
sub-step, so the state is produced by an actual time integration of rigid
body dynamics (semi-implicit Euler + impulse-based contact resolution).
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# frozen physical parameters
# ----------------------------------------------------------------------------
_G = np.array([0.0, 0.0, -9.81], dtype=float)

_SUB_DT = 0.0025          # physical sub-step (s)
_ITERATIONS = 20          # sequential-impulse iterations per sub-step
_BAUMGARTE = 0.2          # penetration correction fraction
_MAX_BIAS = 2.0           # m/s cap on the positional bias velocity
_NEAR = 1.0e-4            # contact capture distance (m): avoids hovering above surfaces
_MAX_SPEED = 30.0         # m/s safety clamp
_MAX_SPIN = 60.0          # rad/s safety clamp

_BOARD_HALF = np.array([1.8, 0.28, 0.07], dtype=float)
_BOARD_MASS = 4.0
_BOARD_CENTER = np.array([0.0, 0.0, 0.78], dtype=float)

_BLOCK_HALF = {1: np.array([0.28, 0.28, 0.28]), 2: np.array([0.22, 0.22, 0.22])}
_BLOCK_MASS = {1: 3.0, 2: 1.0}
_BLOCK_START = {1: np.array([1.05, 0.0, 1.13]), 2: np.array([-1.0, 0.0, 1.07])}

# Per-material friction coefficients (the scene declares distinct 'heavy' and
# 'light' block materials).  The heavy block slips on the tilted board, the
# light block's material grips and stays put while it is carried by the board.
_MU_HEAVY_BOARD = 0.28
_MU_LIGHT_BOARD = 0.65
_MU_GROUND = 0.50         # block or board / ground friction coefficient
_MU_BLOCK = 0.40          # block / block friction coefficient
_CONTACT_LENGTH = 0.07    # frozen contact normalization length (board thickness)


# ----------------------------------------------------------------------------
# small quaternion / geometry helpers
# ----------------------------------------------------------------------------
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


def _quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    wx, wy, wz = omega
    dq = 0.5 * dt * np.array([
        wx * q[3] + wy * q[2] - wz * q[1],
        wy * q[3] + wz * q[0] - wx * q[2],
        wz * q[3] + wx * q[1] - wy * q[0],
        -(wx * q[0] + wy * q[1] + wz * q[2]),
    ])
    out = q + dq
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return out / n


def _box_corners(half: np.ndarray) -> list[np.ndarray]:
    hx, hy, hz = half
    return [
        np.array([sx * hx, sy * hy, sz * hz])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def _closest_point_box(local: np.ndarray, half: np.ndarray):
    """Signed gap and outward unit normal (in local frame) of a point vs a box."""
    q = np.clip(local, -half, half)
    d = local - q
    dist = float(np.linalg.norm(d))
    if dist > 1e-12:
        return d / dist, dist
    # point is inside the box: push out through the nearest face
    pen = half - np.abs(local)
    axis = int(np.argmin(pen))
    normal = np.zeros(3)
    normal[axis] = 1.0 if local[axis] >= 0.0 else -1.0
    return normal, -float(pen[axis])


# ----------------------------------------------------------------------------
# rigid bodies
# ----------------------------------------------------------------------------
class _Body:
    __slots__ = (
        "body_id", "pos", "vel", "quat", "omega", "mass", "half",
        "inv_mass", "inv_inertia_local", "inv_inertia", "is_board", "theta",
    )

    def __init__(self, body_id, pos, mass, half, is_board=False):
        self.body_id = int(body_id)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega = np.zeros(3)
        self.mass = float(mass)
        self.half = np.array(half, dtype=float)
        self.is_board = bool(is_board)
        self.theta = 0.0
        if self.is_board:
            hx, hy, hz = self.half
            iyy = self.mass / 3.0 * (hx * hx + hz * hz)
            self.inv_mass = 0.0
            self.inv_inertia_local = np.zeros((3, 3))
            self.inv_inertia = np.diag([0.0, 1.0 / iyy, 0.0])
        else:
            hx, hy, hz = self.half
            ixx = self.mass / 3.0 * (hy * hy + hz * hz)
            iyy = self.mass / 3.0 * (hx * hx + hz * hz)
            izz = self.mass / 3.0 * (hx * hx + hy * hy)
            self.inv_mass = 1.0 / self.mass
            self.inv_inertia_local = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz])
            self.inv_inertia = self.inv_inertia_local.copy()

    def refresh_inertia(self) -> None:
        if self.is_board:
            return
        rot = _quat_to_mat(self.quat)
        self.inv_inertia = rot @ self.inv_inertia_local @ rot.T


# ----------------------------------------------------------------------------
# contact generation
# ----------------------------------------------------------------------------
def _eff_mass(body: _Body, r: np.ndarray, direction: np.ndarray) -> float:
    if body is None:
        return 0.0
    angular = np.cross(body.inv_inertia @ np.cross(r, direction), r)
    return float(body.inv_mass + float(np.dot(direction, angular)))


def _apply_impulse(body: _Body, r: np.ndarray, impulse: np.ndarray) -> None:
    if body is None:
        return
    if body.inv_mass:
        body.vel = body.vel + impulse * body.inv_mass
    body.omega = body.omega + body.inv_inertia @ np.cross(r, impulse)


def _add(normal, gap, point, a, b, mu, contacts):
    if gap < _NEAR:
        contacts.append({
            "normal": normal, "gap": float(gap), "point": point,
            "a": a, "b": b, "mu": mu, "jn": 0.0, "jt": np.zeros(3),
        })


def _build_contacts(bodies, contacts):
    board = bodies[0]
    rot_b = _quat_to_mat(board.quat)
    rot_bt = rot_b.T
    up = np.array([0.0, 0.0, 1.0])

    # blocks against the board box and the ground plane
    for idx in (1, 2):
        blk = bodies[idx]
        rot_k = _quat_to_mat(blk.quat)
        half = blk.half
        for corner in _box_corners(half):
            point = blk.pos + rot_k @ corner
            local = rot_bt @ (point - board.pos)
            n_local, gap = _closest_point_box(local, _BOARD_HALF)
            mu = _MU_HEAVY_BOARD if idx == 1 else _MU_LIGHT_BOARD
            _add(rot_b @ n_local, gap, point, idx, 0, mu, contacts)
            _add(up, float(point[2]), point, idx, None, _MU_GROUND, contacts)

    # board against the ground plane
    for corner in _box_corners(_BOARD_HALF):
        point = board.pos + rot_b @ corner
        _add(up, float(point[2]), point, 0, None, _MU_GROUND, contacts)

    # block against block (corner of one against the box of the other)
    for i, j in ((1, 2), (2, 1)):
        bi, bj = bodies[i], bodies[j]
        rot_i = _quat_to_mat(bi.quat)
        rot_j = _quat_to_mat(bj.quat)
        rot_jt = rot_j.T
        for corner in _box_corners(bi.half):
            point = bi.pos + rot_i @ corner
            local = rot_jt @ (point - bj.pos)
            n_local, gap = _closest_point_box(local, bj.half)
            _add(rot_j @ n_local, gap, point, i, j, _MU_BLOCK, contacts)


# ----------------------------------------------------------------------------
# sequential-impulse contact solve
# ----------------------------------------------------------------------------
def _solve_contacts(bodies, contacts, dt: float) -> None:
    if not contacts:
        return
    for _ in range(_ITERATIONS):
        for c in contacts:
            normal = c["normal"]
            a = bodies[c["a"]]
            b = bodies[c["b"]] if c["b"] is not None else None
            ra = c["point"] - a.pos
            rb = c["point"] - b.pos if b is not None else None

            va = a.vel + np.cross(a.omega, ra)
            vb = b.vel + np.cross(b.omega, rb) if b is not None else np.zeros(3)
            vrel = va - vb
            vn = float(np.dot(vrel, normal))

            k = _eff_mass(a, ra, normal) + _eff_mass(b, rb, normal)
            if k <= 1e-12:
                continue

            target = -_BAUMGARTE * min(c["gap"], 0.0) / dt
            target = min(target, _MAX_BIAS)
            dj = (target - vn) / k
            new_jn = max(c["jn"] + dj, 0.0)
            dj = new_jn - c["jn"]
            c["jn"] = new_jn
            if dj != 0.0:
                impulse = dj * normal
                _apply_impulse(a, ra, impulse)
                if b is not None:
                    _apply_impulse(b, rb, -impulse)

            # Coulomb friction along the current tangential direction
            va = a.vel + np.cross(a.omega, ra)
            vb = b.vel + np.cross(b.omega, rb) if b is not None else np.zeros(3)
            vrel = va - vb
            vt = vrel - float(np.dot(vrel, normal)) * normal
            speed = float(np.linalg.norm(vt))
            if speed < 1e-9:
                continue
            tangent = vt / speed
            kt = _eff_mass(a, ra, tangent) + _eff_mass(b, rb, tangent)
            if kt <= 1e-12:
                continue
            delta = (-speed / kt) * tangent
            trial = c["jt"] + delta
            limit = c["mu"] * c["jn"]
            mag = float(np.linalg.norm(trial))
            if mag > limit:
                if mag > 1e-15:
                    trial = trial * (limit / mag)
                else:
                    trial = np.zeros(3)
            delta = trial - c["jt"]
            c["jt"] = trial
            if float(np.dot(delta, delta)) > 0.0:
                _apply_impulse(a, ra, delta)
                if b is not None:
                    _apply_impulse(b, rb, -delta)


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
class BenchmarkSolver:
    """Impulse-based rigid body simulation of two blocks on a pivoted board."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        self.time = 0.0
        self.steps = 0
        self._frames = 0
        self.bodies = [
            _Body(0, _BOARD_CENTER, _BOARD_MASS, _BOARD_HALF, is_board=True),
            _Body(1, _BLOCK_START[1], _BLOCK_MASS[1], _BLOCK_HALF[1]),
            _Body(2, _BLOCK_START[2], _BLOCK_MASS[2], _BLOCK_HALF[2]),
        ]
        self._external_force = np.array([
            [0.0, 0.0, -body.mass * 9.81] for body in self.bodies
        ])
        for body in self.bodies:
            body.refresh_inertia()

    # -- physics -------------------------------------------------------------
    def _step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        board = self.bodies[0]
        for body in self.bodies[1:]:
            body.vel = body.vel + _G * dt

        contacts: list[dict] = []
        _build_contacts(self.bodies, contacts)
        _solve_contacts(self.bodies, contacts, dt)

        # integrate
        for body in self.bodies:
            if body.is_board:
                body.theta += body.omega[1] * dt
                half = 0.5 * body.theta
                body.quat = np.array([0.0, np.sin(half), 0.0, np.cos(half)])
                body.omega = np.array([0.0, body.omega[1], 0.0])
                body.vel = np.zeros(3)
                body.pos = _BOARD_CENTER.copy()
                continue
            body.pos = body.pos + body.vel * dt
            body.quat = _quat_integrate(body.quat, body.omega, dt)
            speed = float(np.linalg.norm(body.vel))
            if speed > _MAX_SPEED:
                body.vel = body.vel * (_MAX_SPEED / speed)
            spin = float(np.linalg.norm(body.omega))
            if spin > _MAX_SPIN:
                body.omega = body.omega * (_MAX_SPIN / spin)

        for body in self.bodies[1:]:
            body.refresh_inertia()
        self.steps += 1
        self.time += dt

    # -- frame export --------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        position = np.vstack([body.pos for body in self.bodies])
        velocity = np.vstack([body.vel for body in self.bodies])
        orientation = np.vstack([body.quat for body in self.bodies])
        angular_velocity = np.vstack([body.omega for body in self.bodies])
        mass = np.array([body.mass for body in self.bodies])
        body_id = np.array([body.body_id for body in self.bodies], dtype=int)

        contacts: list[dict] = []
        _build_contacts(self.bodies, contacts)
        gaps = np.array([c["gap"] for c in contacts], dtype=float)
        if gaps.size == 0:
            gaps = np.array([0.0])
        fields = {
            "contact_signed_distance": gaps,
            "constraint_value": gaps.reshape(-1, 1),
            "constraint_target": np.zeros((gaps.size, 1), dtype=float),
            "external_force": self._external_force.copy(),
        }
        scalars = {
            "constraint_scale": _CONTACT_LENGTH,
            "contact_length_scale": _CONTACT_LENGTH,
        }
        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float):
        target = float(time)
        self._frames += 1
        if target > self.time:
            while self.time < target - 1e-12:
                step_dt = min(_SUB_DT, target - self.time)
                self._step(step_dt)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        board = self.bodies[0]
        return {
            "solver": "sequential_impulse_rigid_contact",
            "sub_step": _SUB_DT,
            "iterations": _ITERATIONS,
            "integration_steps": int(self.steps),
            "frames": int(self._frames),
            "final_time": float(self.time),
            "final_board_angle": float(board.theta),
            "final_positions": np.vstack([b.pos for b in self.bodies]).tolist(),
            "final_velocities": np.vstack([b.vel for b in self.bodies]).tolist(),
            "gravity": _G.tolist(),
        }
