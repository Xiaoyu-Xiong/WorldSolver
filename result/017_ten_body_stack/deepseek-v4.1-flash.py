"""Rigid-body physics solver for the ten-body stack scene (017).

The stack is simulated with a real time-stepping scheme: semi-implicit
integration of rigid-body motion (position + quaternion orientation) with a
sequential-impulse (projected Gauss-Seidel) contact solver, Coulomb friction,
Baumgarte velocity-level penetration correction and speculative contacts.

Only NumPy is used; no external physics engine is imported.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
_DEFAULT_BODY_HALF = (0.45, 0.45, 0.1)
_DEFAULT_GROUND_HALF = (4.0, 4.0, 0.05)
_DEFAULT_GROUND_TRANSLATION = (0.0, 0.0, -0.05)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from xyzw quaternions, shape (M,3,3)."""
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=1),
        ],
        axis=1,
    )


def _integrate_quat(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Exponential-map-free first order quaternion integration + renormalization."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    dqx = 0.5 * (wx * qw + wy * qz - wz * qy)
    dqy = 0.5 * (-wx * qz + wy * qw + wz * qx)
    dqz = 0.5 * (wx * qy - wy * qx + wz * qw)
    dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
    out = np.stack(
        [qx + dqx * dt, qy + dqy * dt, qz + dqz * dt, qw + dqw * dt], axis=1
    )
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


class BenchmarkSolver:
    """Deterministic rigid-body stack simulation."""

    def __init__(self) -> None:
        self._time = 0.0
        self._initialized = False
        self._diagnostics: dict[str, object] = {}

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _parse_geometry(config: dict):
        geometry = config.get("scene_geometry") or {}
        objects = geometry.get("objects") or []
        ground_half = np.asarray(_DEFAULT_GROUND_HALF, dtype=float)
        ground_translation = np.asarray(_DEFAULT_GROUND_TRANSLATION, dtype=float)
        half_by_id: dict[int, np.ndarray] = {}
        for obj in objects:
            geom = obj.get("geometry") or {}
            half = geom.get("half_extents")
            if not half:
                continue
            half = np.asarray(half, dtype=float)
            binding = obj.get("state_binding") or {}
            selector = binding.get("selector")
            if selector == "body_id":
                half_by_id[int(binding.get("value"))] = half
            elif obj.get("motion") == "static" or obj.get("semantic") == "boundary":
                transform = obj.get("transform") or {}
                translation = transform.get("translation")
                ground_half = half
                if translation is not None:
                    ground_translation = np.asarray(translation, dtype=float)
        return ground_half, ground_translation, half_by_id

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = getattr(scene, "config", {}) or {}
        gravity = config.get("gravity", _DEFAULT_GRAVITY)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

        initial = scene.initial_state()
        position = np.asarray(initial.position, dtype=float)
        velocity = np.asarray(initial.velocity, dtype=float)
        n = int(position.shape[0])
        self.n_bodies = n
        self.mass = (
            np.asarray(initial.mass, dtype=float).copy()
            if initial.mass is not None
            else np.ones(n)
        )
        self.body_id = (
            np.asarray(initial.body_id, dtype=int).copy()
            if initial.body_id is not None
            else np.arange(1, n + 1, dtype=int)
        )
        if initial.orientation is not None:
            orientation = np.asarray(initial.orientation, dtype=float).copy()
            orientation /= np.maximum(
                np.linalg.norm(orientation, axis=1, keepdims=True), 1e-12
            )
        else:
            orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        angular = (
            np.asarray(initial.angular_velocity, dtype=float).copy()
            if initial.angular_velocity is not None
            else np.zeros((n, 3))
        )

        ground_half, ground_translation, half_by_id = self._parse_geometry(config)
        half = np.zeros((n, 3))
        for i, bid in enumerate(self.body_id):
            half[i] = half_by_id.get(int(bid), np.asarray(_DEFAULT_BODY_HALF, float))
        self.half = half
        self.contact_length_scale = float(2.0 * np.mean(half[:, 2]))

        inv_inertia_local = np.zeros((n, 3))
        for i in range(n):
            hx, hy, hz = half[i]
            inertia = (self.mass[i] / 3.0) * np.array(
                [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]
            )
            inv_inertia_local[i] = 1.0 / np.maximum(inertia, 1e-12)
        self.inv_inertia_local = inv_inertia_local

        # Augmented state: index n is the static ground plane.
        m = n + 1
        self.inv_mass = np.zeros(m)
        self.inv_mass[:n] = 1.0 / np.maximum(self.mass, 1e-12)
        self.xi = np.zeros((m, 3))
        self.xi[:n] = position
        self.xi[n] = ground_translation
        self.vi = np.zeros((m, 3))
        self.vi[:n] = velocity
        self.wi = np.zeros((m, 3))
        self.wi[:n] = angular
        self.qi = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (m, 1))
        self.qi[:n] = orientation
        self.inv_inertia_aug = np.zeros((m, 3))
        self.inv_inertia_aug[:n] = inv_inertia_local
        self.half_aug = np.zeros((m, 3))
        self.half_aug[:n] = half
        self.half_aug[n] = ground_half

        # Interfaces: ground->body_1 then body_k -> body_{k+1}.
        self.upper_index = np.arange(n)
        self.lower_index = np.concatenate(([n], np.arange(n - 1)))
        self.corner_signs = np.array(
            [[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]
        )

        self.accum_normal = np.zeros((n, 4))
        self.accum_tangent = np.zeros((n, 4, 3))
        self.last_signed_distance = np.zeros((n, 4))
        self.last_interaction_force = np.zeros((n, 3))

        self.dt_sub = 0.001
        self.iterations = 10
        self.contact_margin = 1.5e-3
        self.baumgarte = 0.25
        self.slop = 2.0e-4
        self.friction = 0.6
        self.restitution = 0.0

        # Populate the reported geometric gaps for the initial frame.
        self.last_signed_distance = self._contacts()[2]
        self.last_interaction_force = np.zeros((n, 3))
        self._initialized = True
        self._diagnostics = {
            "scene_id": config.get("scene_id", "unknown"),
            "bodies": int(n),
            "dt_substep": float(self.dt_sub),
            "solver": "sequential_impulse",
        }

    # ---------------------------------------------------------------- contacts
    def _contacts(self):
        upper = self.upper_index
        lower = self.lower_index
        rot = _quat_to_mat(self.qi)
        local = self.half_aug[upper][:, None, :] * self.corner_signs[None, :, :]
        points = self.xi[upper][:, None, :] + np.einsum(
            "nij,ncj->nci", rot[upper], local
        )
        normals = rot[lower][:, :, 2]
        plane_points = (
            self.xi[lower] + self.half_aug[lower][:, 2][:, None] * rot[lower][:, :, 2]
        )
        signed = np.einsum(
            "nci,ni->nc", points - plane_points[:, None, :], normals
        )
        r_upper = points - self.xi[upper][:, None, :]
        r_lower = points - self.xi[lower][:, None, :]
        inv_inertia_upper = np.einsum(
            "nij,nj,nkj->nik", rot[upper], self.inv_inertia_aug[upper], rot[upper]
        )
        inv_inertia_lower = np.einsum(
            "nij,nj,nkj->nik", rot[lower], self.inv_inertia_aug[lower], rot[lower]
        )
        normal_mass = self._effective_mass(
            r_upper, r_lower, inv_inertia_upper, inv_inertia_lower, normals[:, None, :]
        )
        return (
            points,
            normals,
            signed,
            r_upper,
            r_lower,
            inv_inertia_upper,
            inv_inertia_lower,
            normal_mass,
        )

    def _effective_mass(self, r_upper, r_lower, inv_inertia_upper, inv_inertia_lower, direction):
        cross_upper = np.cross(r_upper, direction)
        cross_lower = np.cross(r_lower, direction)
        term_upper = np.cross(
            np.einsum("nij,ncj->nci", inv_inertia_upper, cross_upper), r_upper
        )
        term_lower = np.cross(
            np.einsum("nij,ncj->nci", inv_inertia_lower, cross_lower), r_lower
        )
        return (
            self.inv_mass[self.upper_index][:, None]
            + self.inv_mass[self.lower_index][:, None]
            + np.einsum("nci,nci->nc", term_upper, direction)
            + np.einsum("nci,nci->nc", term_lower, direction)
        )

    def _contact_velocity(self, r_upper, r_lower):
        vel_upper = self.vi[self.upper_index][:, None, :] + np.cross(
            self.wi[self.upper_index][:, None, :], r_upper
        )
        vel_lower = self.vi[self.lower_index][:, None, :] + np.cross(
            self.wi[self.lower_index][:, None, :], r_lower
        )
        return vel_upper - vel_lower

    def _apply_impulse(self, impulse, r_upper, r_lower, inv_inertia_upper, inv_inertia_lower):
        summed = impulse.sum(axis=1)
        upper = self.upper_index
        lower = self.lower_index
        self.vi[upper] += summed * self.inv_mass[upper][:, None]
        self.wi[upper] += np.einsum(
            "nij,nj->ni", inv_inertia_upper, np.cross(r_upper, impulse).sum(axis=1)
        )
        self.vi[lower] -= summed * self.inv_mass[lower][:, None]
        self.wi[lower] += np.einsum(
            "nij,nj->ni", inv_inertia_lower, np.cross(r_lower, impulse).sum(axis=1)
        )

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        n = self.n_bodies
        self.vi[:n] += self.gravity * dt
        (
            _points,
            normals,
            signed,
            r_upper,
            r_lower,
            inv_inertia_upper,
            inv_inertia_lower,
            normal_mass,
        ) = self._contacts()
        normal_mass = np.maximum(normal_mass, 1e-12)
        normal_dir = normals[:, None, :]
        active = signed < self.contact_margin
        self.accum_normal *= active
        self.accum_tangent *= active[:, :, None]

        warm = self.accum_normal[:, :, None] * normal_dir + self.accum_tangent
        self._apply_impulse(warm, r_upper, r_lower, inv_inertia_upper, inv_inertia_lower)

        bias = self.baumgarte * np.maximum(0.0, -signed - self.slop) / dt
        target_velocity = np.where(signed > 0.0, -signed / dt, bias)

        for _ in range(self.iterations):
            relative = self._contact_velocity(r_upper, r_lower)
            normal_speed = np.einsum("nci,nci->nc", relative, normal_dir)
            delta = (target_velocity - normal_speed) / normal_mass
            new_accum = np.maximum(0.0, self.accum_normal + delta)
            delta_normal = new_accum - self.accum_normal
            self.accum_normal = new_accum
            self._apply_impulse(
                delta_normal[:, :, None] * normal_dir,
                r_upper,
                r_lower,
                inv_inertia_upper,
                inv_inertia_lower,
            )

            relative = self._contact_velocity(r_upper, r_lower)
            normal_speed = np.einsum("nci,nci->nc", relative, normal_dir)
            tangential = relative - normal_speed[:, :, None] * normal_dir
            speed = np.linalg.norm(tangential, axis=2)
            direction = np.where(
                speed[:, :, None] > 1e-12,
                tangential / np.maximum(speed[:, :, None], 1e-12),
                np.array([1.0, 0.0, 0.0]),
            )
            tangent_mass = np.maximum(
                self._effective_mass(
                    r_upper, r_lower, inv_inertia_upper, inv_inertia_lower, direction
                ),
                1e-12,
            )
            projected = np.einsum("nci,nci->nc", tangential, direction)
            delta_tangent = (-projected / tangent_mass)[:, :, None] * direction
            candidate = self.accum_tangent + delta_tangent
            limit = self.friction * self.accum_normal
            magnitude = np.linalg.norm(candidate, axis=2)
            scale = np.where(magnitude > limit, limit / np.maximum(magnitude, 1e-12), 1.0)
            candidate = candidate * scale[:, :, None]
            delta_tangent = candidate - self.accum_tangent
            self.accum_tangent = candidate
            self._apply_impulse(
                delta_tangent, r_upper, r_lower, inv_inertia_upper, inv_inertia_lower
            )

        self._integrate(dt)

        total = self.accum_normal[:, :, None] * normal_dir + self.accum_tangent
        per_upper = total.sum(axis=1)
        force = np.zeros((n, 3))
        force[self.upper_index] += per_upper
        dynamic_lower = self.lower_index < n
        np.add.at(force, self.lower_index[dynamic_lower], -per_upper[dynamic_lower])
        self.last_signed_distance = signed
        self.last_interaction_force = force / dt

    def _integrate(self, dt: float) -> None:
        n = self.n_bodies
        self.xi[:n] += self.vi[:n] * dt
        self.qi[:n] = _integrate_quat(self.qi[:n], self.wi[:n], dt)

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        if not self._initialized:
            raise RuntimeError("BenchmarkSolver.initialize must be called first")
        if target < self._time - 1e-9:
            raise ValueError("advance_to requires non-decreasing sample times")
        steps = int(round((target - self._time) / self.dt_sub))
        for _ in range(max(0, steps)):
            self._step(self.dt_sub)
        self._time = target
        return self._frame()

    def _frame(self) -> StateFrame:
        n = self.n_bodies
        return StateFrame(
            time=float(self._time),
            position=self.xi[:n].copy(),
            velocity=self.vi[:n].copy(),
            orientation=self.qi[:n].copy(),
            angular_velocity=self.wi[:n].copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": self.last_signed_distance.reshape(-1).copy(),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "interaction_force": self.last_interaction_force.copy(),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)
