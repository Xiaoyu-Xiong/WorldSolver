"""Self-contained rigid-body solver for the spolling coin decay scene.

The scene is a single thin rigid cylinder (the coin) released tilted above a
static box floor with gravity.  The solver implements its own rigid-body
engine with the base scientific stack only:

* semi-implicit (symplectic) Euler time stepping with sub-stepping,
* quaternion attitude integration (exponential map),
* contact detection against the frozen floor box through a set of surface
  sample points on the coin (rim rings), using signed gaps,
* sequential-impulse contact resolution with speculative normal constraints
  (no tunnelling), restitution and Coulomb friction at the frozen friction
  coefficient,
* a positional projection pass that removes residual overlap so contacts
  never stay persistently penetrating,
* a static-friction rest criterion that lets the decaying motion settle into
  an exact state of rest.

Every requested sample frame exports the raw physical fields demanded by the
trajectory contract (state and diagnostics), never a scripted animation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

_UP = np.array([0.0, 0.0, 1.0], dtype=float)


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------
def _normalize_quaternion(q: Any) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
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


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + bw * ax + ay * bz - az * by,
            aw * by + bw * ay + az * bx - ax * bz,
            aw * bz + bw * az + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _rotation_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-14:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = rotation_vector / angle
    half = 0.5 * angle
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def _integrate_quaternion(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    rate = float(np.linalg.norm(omega))
    if rate < 1e-14:
        return q
    axis = omega / rate
    half = 0.5 * rate * h
    delta = np.concatenate([axis * math.sin(half), [math.cos(half)]])
    return _normalize_quaternion(_quaternion_multiply(delta, q))


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    """Deterministic rigid-body solver for one coin decaying on a rigid floor."""

    # numerical settings (units: SI)
    SUBSTEPS = 10            # sub-steps per exported sample interval
    SOLVER_ITERATIONS = 8    # sequential-impulse iterations per sub-step
    PROJECTION_ITERATIONS = 3
    SLOP = 1e-4              # tolerated overlap (m)
    RESTITUTION_SPEED = 0.4  # below this approach speed restitution is off (m/s)
    REST_LINEAR = 1e-3       # rest criterion (m/s)
    REST_ANGULAR = 3e-3      # rest criterion (rad/s)
    REST_ENERGY = 1e-6       # rest criterion (J)
    REST_HOLD = 0.15         # rest must persist this long (s)

    def initialize(self, scene: Any, seed: int) -> None:
        config = getattr(scene, "config", None)
        self._config: dict[str, Any] = dict(config) if isinstance(config, dict) else {}
        self._seed = int(seed)

        gravity = self._config.get("gravity", [0.0, 0.0, -9.81])
        self._gravity = np.asarray(gravity, dtype=float).reshape(3)
        contact = self._config.get("contact", {}) or {}
        self._friction = float(contact.get("friction", 0.5))
        self._restitution = float(contact.get("restitution", 0.0))
        self._length_scale = float(contact.get("length_scale", 0.02))
        self._sample_dt = float(self._config.get("sample_dt", 0.02))
        self._substep = self._sample_dt / self.SUBSTEPS
        self._margin = float(np.clip(self._length_scale, 0.005, 0.05))

        # ---- frozen geometry -------------------------------------------------
        geometry = self._config.get("scene_geometry", {}) or {}
        coin_obj: dict[str, Any] | None = None
        floor_obj: dict[str, Any] | None = None
        for raw in geometry.get("objects", []) or []:
            obj = dict(raw)
            kind = str(dict(obj.get("geometry", {}) or {}).get("type", ""))
            if obj.get("motion") == "dynamic" and kind == "cylinder" and coin_obj is None:
                coin_obj = obj
            elif obj.get("semantic") == "boundary" and kind == "box" and floor_obj is None:
                floor_obj = obj

        coin_geom = dict((coin_obj or {}).get("geometry", {}) or {})
        radius = float(coin_geom.get("radius", 0.48))
        height = float(coin_geom.get("height", 0.08))
        if radius <= 0.0:
            radius = 0.48
        if height <= 0.0:
            height = 0.08

        self._floor_top = 0.0
        self._floor_hx = 3.2
        self._floor_hy = 2.2
        if floor_obj is not None:
            floor_geom = dict(floor_obj.get("geometry", {}) or {})
            half = np.asarray(floor_geom.get("half_extents", [3.2, 2.2, 0.05]), dtype=float).reshape(3)
            shift = np.asarray(
                dict(floor_obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3)
            self._floor_top = float(shift[2] + abs(half[2]))
            self._floor_hx = float(abs(half[0]))
            self._floor_hy = float(abs(half[1]))

        # ---- frozen initial state -------------------------------------------
        initial = scene.initial_state()
        position = np.asarray(initial.position, dtype=float).reshape(-1, 3)
        velocity = np.asarray(initial.velocity, dtype=float).reshape(-1, 3)
        self._position = position[0].copy()
        self._velocity = velocity[0].copy()
        orientation = getattr(initial, "orientation", None)
        if orientation is not None:
            self._orientation = _normalize_quaternion(np.asarray(orientation, dtype=float).reshape(-1, 4)[0])
        else:
            self._orientation = np.array([0.0, 0.0, 0.0, 1.0])
        angular = getattr(initial, "angular_velocity", None)
        if angular is not None:
            self._omega = np.asarray(angular, dtype=float).reshape(-1, 3)[0].copy()
        else:
            self._omega = np.zeros(3)
        mass = getattr(initial, "mass", None)
        self._mass = float(np.asarray(mass, dtype=float).reshape(-1)[0]) if mass is not None else 0.5
        if not np.isfinite(self._mass) or self._mass <= 0.0:
            self._mass = 0.5
        body_id = getattr(initial, "body_id", None)
        self._body_id = (
            np.asarray(body_id).reshape(-1).astype(np.int32)
            if body_id is not None
            else np.array([0], dtype=np.int32)
        )

        # solid-cylinder inertia about the centre of mass (body frame, z = axis)
        i_axis = 0.5 * self._mass * radius * radius
        i_perp = self._mass * (3.0 * radius * radius + height * height) / 12.0
        self._inertia_body = np.array([i_perp, i_perp, i_axis], dtype=float)

        # ---- surface sample points (body frame) ------------------------------
        samples = 32
        angles = 2.0 * math.pi * np.arange(samples, dtype=float) / float(samples)
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        points = []
        for z in (0.5 * height, -0.5 * height):
            layer = np.zeros((samples, 3), dtype=float)
            layer[:, :2] = ring
            layer[:, 2] = z
            points.append(layer)
        self._body_points = np.concatenate(points, axis=0)

        # ---- simulation bookkeeping -----------------------------------------
        self._time = 0.0
        self._impulse_accum = np.zeros_like(self._body_points)
        self._interval = self._sample_dt
        self._rest_hold = 0.0
        self._frozen = False
        self._max_penetration = 0.0
        self._min_gap_seen = float("inf")
        self._initial_energy = self._mechanical_energy()
        self._last_energy = self._initial_energy
        self._max_energy = self._initial_energy

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _world_offsets(self, orientation: np.ndarray | None = None) -> np.ndarray:
        rot = _quaternion_matrix(self._orientation if orientation is None else orientation)
        return self._body_points @ rot.T

    def _gaps(self, offsets: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = centre[2] + offsets[:, 2] - self._floor_top
        valid = (np.abs(centre[0] + offsets[:, 0]) <= self._floor_hx) & (
            np.abs(centre[1] + offsets[:, 1]) <= self._floor_hy
        )
        return np.where(valid, z, np.inf), valid

    def _inverse_inertia(self, orientation: np.ndarray) -> np.ndarray:
        rot = _quaternion_matrix(orientation)
        world = rot @ np.diag(self._inertia_body) @ rot.T
        return np.linalg.inv(world)

    def _mechanical_energy(self) -> float:
        transl = 0.5 * self._mass * float(self._velocity @ self._velocity)
        rot = _quaternion_matrix(self._orientation)
        world = rot @ np.diag(self._inertia_body) @ rot.T
        spin = 0.5 * float(self._omega @ (world @ self._omega))
        potential = self._mass * float(-self._gravity[2]) * (self._position[2] - self._floor_top)
        return transl + spin + potential

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        if self._frozen or h <= 0.0:
            return
        mass = self._mass
        position = self._position
        velocity = self._velocity
        orientation = self._orientation
        omega = self._omega

        rot = _quaternion_matrix(orientation)
        world_inertia = rot @ np.diag(self._inertia_body) @ rot.T
        inv_inertia = np.linalg.inv(world_inertia)

        # external (gravity) and torque-free gyroscopic evolution
        velocity = velocity + h * self._gravity
        omega = omega + h * (inv_inertia @ (-np.cross(omega, world_inertia @ omega)))

        # ---- contact detection ------------------------------------------------
        offsets = self._body_points @ rot.T
        gaps, valid = self._gaps(offsets, position)
        active = np.nonzero(valid & (gaps < self._margin))[0]

        if active.size:
            r = offsets[active]
            gap = gaps[active]
            rn = np.cross(r, _UP)
            inv_rn = (inv_inertia @ rn.T).T
            kn = 1.0 / mass + np.einsum("ij,ij->i", np.cross(inv_rn, r), np.tile(_UP, (r.shape[0], 1)))

            point_velocity = velocity + np.cross(omega, r)
            pre_normal = point_velocity @ _UP
            # speculative constraint: a point that is still separated may only
            # approach fast enough to reach the tolerated gap at the end of the
            # step; contacts at/inside the slop may not approach at all.
            target = np.where(gap > self.SLOP, -(gap - self.SLOP) / h, 0.0)
            # restitution only for impacts that really occur inside this step
            predicted = (pre_normal * h) < (-(gap - self.SLOP))
            bounce = predicted & (pre_normal < -self.RESTITUTION_SPEED)
            if np.any(bounce):
                target = np.where(
                    bounce, np.maximum(target, -self._restitution * pre_normal), target
                )

            lam = np.zeros(active.size)
            fric = np.zeros((active.size, 3))
            for _ in range(self.SOLVER_ITERATIONS):
                for k in range(active.size):
                    rk = r[k]
                    normal_velocity = (velocity + np.cross(omega, rk))[2]
                    delta = (target[k] - normal_velocity) / kn[k]
                    updated = lam[k] + delta
                    if updated < 0.0:
                        updated = 0.0
                    delta = updated - lam[k]
                    lam[k] = updated
                    if delta != 0.0:
                        impulse = delta * _UP
                        velocity = velocity + impulse / mass
                        omega = omega + inv_inertia @ np.cross(rk, impulse)

                    tangential = (velocity + np.cross(omega, rk))
                    tangential = tangential - tangential[2] * _UP
                    speed = float(np.linalg.norm(tangential))
                    if speed > 1e-10:
                        tangent = tangential / speed
                        rotation = np.cross(rk, tangent)
                        kt = 1.0 / mass + float(tangent @ (inv_inertia @ rotation))
                        if kt > 1e-12:
                            magnitude = min(self._friction * lam[k], speed / kt)
                            impulse = -magnitude * tangent
                            velocity = velocity + impulse / mass
                            omega = omega + inv_inertia @ np.cross(rk, impulse)
                            fric[k] += impulse

            self._impulse_accum[active] += lam[:, None] * _UP + fric

        # ---- position / attitude integration ---------------------------------
        position = position + h * velocity
        orientation = _integrate_quaternion(orientation, omega, h)

        # ---- positional projection (remove residual overlap) -----------------
        for _ in range(self.PROJECTION_ITERATIONS):
            offsets = self._body_points @ _quaternion_matrix(orientation).T
            gaps, valid = self._gaps(offsets, position)
            penetrating = np.nonzero(valid & (gaps < 0.0))[0]
            if penetrating.size == 0:
                break
            inv_inertia = self._inverse_inertia(orientation)
            for k in penetrating:
                rk = offsets[k]
                rn = np.cross(rk, _UP)
                kn = 1.0 / mass + float(_UP @ np.cross(inv_inertia @ rn, rk))
                if kn <= 1e-12:
                    continue
                correction = min(0.6 * (-gaps[k]), 1e-2) / kn
                impulse = correction * _UP
                position = position + impulse / mass
                rotation = inv_inertia @ np.cross(rk, impulse)
                orientation = _normalize_quaternion(
                    _quaternion_multiply(_rotation_quaternion(rotation), orientation)
                )

        # ---- safety clamps and rest detection --------------------------------
        speed = float(np.linalg.norm(velocity))
        spin = float(np.linalg.norm(omega))
        if speed > 1e3 or not np.isfinite(speed):
            velocity = np.zeros(3)
        if spin > 1e3 or not np.isfinite(spin):
            omega = np.zeros(3)
        if not np.all(np.isfinite(position)):
            position = np.zeros(3)
        orientation = _normalize_quaternion(orientation)

        self._position = position
        self._velocity = velocity
        self._orientation = orientation
        self._omega = omega

        energy = self._mechanical_energy()
        self._last_energy = energy
        self._max_energy = max(self._max_energy, energy)

        offsets = self._body_points @ _quaternion_matrix(orientation).T
        gaps, valid = self._gaps(offsets, position)
        finite = gaps[np.isfinite(gaps)]
        if finite.size:
            self._max_penetration = max(self._max_penetration, float(-finite.min()))
            self._min_gap_seen = min(self._min_gap_seen, float(finite.min()))

        touching = bool(finite.size) and float(finite.min()) < 5e-3
        if touching and speed < self.REST_LINEAR and spin < self.REST_ANGULAR and energy < self.REST_ENERGY:
            self._rest_hold += h
        else:
            self._rest_hold = 0.0
        if self._rest_hold >= self.REST_HOLD:
            self._frozen = True
            self._velocity = np.zeros(3)
            self._omega = np.zeros(3)

    # ------------------------------------------------------------------
    # frame construction
    # ------------------------------------------------------------------
    def _contact_report(self) -> tuple[np.ndarray, np.ndarray]:
        offsets = self._world_offsets()
        gaps, valid = self._gaps(offsets, self._position)
        finite = np.isfinite(gaps)
        if np.any(finite):
            order = np.argsort(np.where(finite, gaps, np.inf))
            ordered = order[np.isfinite(gaps[order])]
            near = ordered[gaps[ordered] <= self._margin]
            selected = near if near.size else ordered[:1]
        else:
            selected = np.array([int(np.argmin(np.where(finite, gaps, np.inf)))], dtype=int)
        signed = np.asarray(gaps[selected], dtype=float).reshape(-1)
        signed[~np.isfinite(signed)] = float(self._margin)
        return selected, signed

    def _interaction_forces(self, selected: np.ndarray | None) -> np.ndarray:
        if self._frozen:
            offsets = self._world_offsets()
            gaps, valid = self._gaps(offsets, self._position)
            finite = np.isfinite(gaps)
            count = int(np.count_nonzero(finite & (gaps <= self._margin)))
            if count <= 0:
                count = 1
            support = -self._mass * self._gravity / float(count)
            return np.tile(support.reshape(1, 3), (count, 1))
        rows = self._impulse_accum
        magnitude = np.linalg.norm(rows, axis=1)
        keep = np.nonzero(magnitude > 0.0)[0]
        if keep.size == 0:
            return np.zeros((1, 3), dtype=float)
        interval = self._interval if self._interval > 1e-12 else self._sample_dt
        return rows[keep] / interval

    def _make_frame(self, time: float) -> StateFrame:
        selected, signed = self._contact_report()
        interaction = self._interaction_forces(selected)
        external = self._mass * self._gravity
        frame = StateFrame(
            time=float(time),
            position=self._position.reshape(1, 3).copy(),
            velocity=self._velocity.reshape(1, 3).copy(),
            orientation=self._orientation.reshape(1, 4).copy(),
            angular_velocity=self._omega.reshape(1, 3).copy(),
            mass=np.array([self._mass], dtype=float),
            body_id=self._body_id.copy(),
            fields={
                "external_force": external.reshape(1, 3).astype(float),
                "interaction_force": np.asarray(interaction, dtype=float).reshape(-1, 3),
                "contact_signed_distance": np.asarray(signed, dtype=float).reshape(-1),
            },
            scalars={"contact_length_scale": float(self._length_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # public contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            target = self._time
        if target < self._time:
            target = self._time
        remaining = target - self._time
        if remaining > 1e-12:
            steps = max(1, int(round(remaining / self._substep)))
            h = remaining / float(steps)
            self._impulse_accum[:] = 0.0
            for _ in range(steps):
                self._step(h)
            self._interval = remaining
            self._time = target
        else:
            self._impulse_accum[:] = 0.0
            self._interval = self._sample_dt
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": str(self._config.get("scene_id", "")),
            "seed": self._seed,
            "frozen": bool(self._frozen),
            "max_penetration": float(self._max_penetration),
            "min_signed_gap": float(self._min_gap_seen) if np.isfinite(self._min_gap_seen) else 0.0,
            "initial_energy": float(self._initial_energy),
            "final_energy": float(self._last_energy),
            "max_energy": float(self._max_energy),
            "final_position": [float(v) for v in self._position],
        }
