"""Self-contained physics solver for an articulated-cylinder cloth sleeve."""
from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self._nx, self._ny = [int(v) for v in cloth["grid_shape"]]
        self._n = self._nx * self._ny
        self._size_x, self._size_v = [float(v) for v in cloth["size"]]
        self._thickness = float(cloth.get("thickness", 0.015))
        self._mass = np.full(self._n, float(cloth["total_mass"]) / self._n)
        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._duration = float(cfg.get("duration", 7.0))
        self._seed = int(seed)

        geometry = cfg.get("scene_geometry", {})
        objects = geometry.get("objects", []) if isinstance(geometry, dict) else []
        capsules: list[dict[str, Any]] = [
            item for item in objects if item.get("geometry", {}).get("type") == "capsule"
        ]
        capsules.sort(
            key=lambda item: float(
                item.get("transform", {}).get("translation", [0.0, 0.0, 0.0])[0]
            )
        )
        if len(capsules) < 2:
            capsules = [
                {
                    "geometry": {"radius": 0.34, "height": 1.3},
                    "transform": {"translation": [-0.55, 0.0, 1.0]},
                },
                {
                    "geometry": {"radius": 0.31, "height": 1.3},
                    "transform": {"translation": [0.55, 0.0, 1.0]},
                },
            ]
        left, right = capsules[:2]
        self._r_left = float(left.get("geometry", {}).get("radius", 0.34))
        self._r_right = float(right.get("geometry", {}).get("radius", 0.31))
        self._cap_height = float(left.get("geometry", {}).get("height", 1.3))
        lt = np.asarray(
            left.get("transform", {}).get("translation", [-0.55, 0.0, 1.0]), dtype=float
        )
        rt = np.asarray(
            right.get("transform", {}).get("translation", [0.55, 0.0, 1.0]), dtype=float
        )
        self._pivot_x = float(0.5 * (lt[0] + rt[0]))
        self._pivot_z = float(0.5 * (lt[2] + rt[2]))

        # Columns run along the limbs; rows wrap once around the sleeve.  The
        # duplicated final row is required by the frozen rectangular topology.
        self._u = np.linspace(-0.5 * self._size_x, 0.5 * self._size_x, self._nx)
        self._theta = np.linspace(0.0, 2.0 * np.pi, self._ny)
        uu, tt = np.meshgrid(self._u, self._theta, indexing="xy")
        self._uu, self._tt = uu.ravel(), tt.ravel()
        self._du = self._size_x / max(self._nx - 1, 1)
        self._dv = self._size_v / max(self._ny - 1, 1)
        self._sleeve_radius = max(self._r_left, self._r_right) + 1.7 * self._thickness
        # A deterministic knit-scale imperfection seeds wrinkles without adding
        # stochastic forcing or prescribing an animation.
        self._radius_offset = 0.004 * np.sin(3.0 * self._tt + 1.7 * self._uu)
        self._position = self._target_shape(0.0)
        self._velocity = np.zeros((self._n, 3), dtype=float)
        self._time = 0.0
        (
            self._edges_i,
            self._edges_j,
            self._rest_lengths,
            self._spring_kind,
        ) = self._make_springs(self._position)
        self._last_external = self._mass[:, None] * self._gravity[None, :]
        self._last_contact = self._contact_distances(self._position, 0.0)
        self._ready = True

    def _joint_angle(self, time: float) -> tuple[float, float]:
        phase = np.clip(float(time) / max(self._duration, 1e-9), 0.0, 1.0)
        angle = 0.92 * np.sin(np.pi * phase)
        rate = 0.92 * np.pi / max(self._duration, 1e-9) * np.cos(np.pi * phase)
        return float(angle), float(rate)

    def _centerline(self, x: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Blend the two rigid centerlines smoothly across the elbow."""
        right_x = self._pivot_x + (x - self._pivot_x) * np.cos(angle)
        right_z = self._pivot_z + (x - self._pivot_x) * np.sin(angle)
        blend = np.clip((x - self._pivot_x + 0.22) / 0.44, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        center = np.column_stack(
            (
                (1.0 - blend) * x + blend * right_x,
                np.zeros_like(x),
                (1.0 - blend) * self._pivot_z + blend * right_z,
            )
        )
        return center, blend

    def _target_shape(self, time: float) -> np.ndarray:
        angle, _ = self._joint_angle(time)
        center, blend = self._centerline(self._uu + self._pivot_x, angle)
        local_angle = angle * blend
        radial_xz = np.column_stack(
            (-np.sin(local_angle), np.zeros(self._n), np.cos(local_angle))
        )
        radial_y = np.column_stack(
            (np.zeros(self._n), np.ones(self._n), np.zeros(self._n))
        )
        radius = self._sleeve_radius + self._radius_offset
        return center + radius[:, None] * (
            np.cos(self._tt)[:, None] * radial_y
            + np.sin(self._tt)[:, None] * radial_xz
        )

    def _make_springs(self, rest: np.ndarray):
        pairs: list[tuple[int, int, int]] = []

        def add(a: int, b: int, kind: int) -> None:
            pairs.append((a, b, kind))

        for row in range(self._ny):
            for col in range(self._nx):
                idx = row * self._nx + col
                if col + 1 < self._nx:
                    add(idx, idx + 1, 0)
                if row + 1 < self._ny:
                    add(idx, idx + self._nx, 0)
                if col + 1 < self._nx and row + 1 < self._ny:
                    add(idx, idx + self._nx + 1, 1)
                    add(idx + 1, idx + self._nx, 1)
                if col + 2 < self._nx:
                    add(idx, idx + 2, 2)
                if row + 2 < self._ny:
                    add(idx, idx + 2 * self._nx, 2)
        ei = np.asarray([p[0] for p in pairs], dtype=np.int32)
        ej = np.asarray([p[1] for p in pairs], dtype=np.int32)
        kind = np.asarray([p[2] for p in pairs], dtype=np.int8)
        lengths = np.linalg.norm(rest[ej] - rest[ei], axis=1)
        return ei, ej, lengths, kind

    @staticmethod
    def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray):
        ab = b - a
        t = np.clip(
            np.einsum("ij,j->i", points - a, ab) / max(float(np.dot(ab, ab)), 1e-12),
            0.0,
            1.0,
        )
        nearest = a[None, :] + t[:, None] * ab[None, :]
        delta = points - nearest
        length = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(length[:, None], 1e-9)
        normal[length < 1e-9] = np.array([0.0, 1.0, 0.0])
        return length, normal, nearest

    def _capsule_state(self, side: str, angle: float, rate: float):
        if side == "left":
            radius = self._r_left
            axis = np.array([1.0, 0.0, 0.0])
            center = np.array([self._pivot_x - 0.55, 0.0, self._pivot_z])
            angular_rate = 0.0
        else:
            radius = self._r_right
            axis = np.array([np.cos(angle), 0.0, np.sin(angle)])
            center = np.array(
                [
                    self._pivot_x + 0.55 * np.cos(angle),
                    0.0,
                    self._pivot_z + 0.55 * np.sin(angle),
                ]
            )
            angular_rate = rate
        half = 0.5 * self._cap_height
        return radius, center - half * axis, center + half * axis, angular_rate

    def _contact_data(self, points: np.ndarray, time: float):
        angle, rate = self._joint_angle(time)
        distances, normals, body_velocities = [], [], []
        for side in ("left", "right"):
            radius, a, b, angular_rate = self._capsule_state(side, angle, rate)
            length, normal, nearest = self._segment_distance(points, a, b)
            distances.append(length - radius - self._thickness)
            normals.append(normal)
            if side == "right":
                relative = nearest - np.array([self._pivot_x, 0.0, self._pivot_z])
                body_velocities.append(
                    np.cross(np.array([0.0, angular_rate, 0.0]), relative)
                )
            else:
                body_velocities.append(np.zeros_like(points))
        distances.append(points[:, 2] - 0.5 * self._thickness)
        normals.append(np.tile(np.array([0.0, 0.0, 1.0]), (self._n, 1)))
        body_velocities.append(np.zeros_like(points))
        all_distances = np.column_stack(distances)
        choice = np.argmin(all_distances, axis=1)
        rows = np.arange(self._n)
        return (
            all_distances[rows, choice],
            np.stack(normals, axis=1)[rows, choice],
            np.stack(body_velocities, axis=1)[rows, choice],
        )

    def _contact_distances(self, points: np.ndarray, time: float) -> np.ndarray:
        return self._contact_data(points, time)[0]

    def _forces(self, target: np.ndarray, time: float) -> np.ndarray:
        p, v = self._position, self._velocity
        force = self._mass[:, None] * self._gravity[None, :]
        delta = p[self._edges_j] - p[self._edges_i]
        lengths = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(lengths[:, None], 1e-9)
        relative_speed = np.einsum(
            "ij,ij->i", v[self._edges_j] - v[self._edges_i], direction
        )
        stiffness = np.where(
            self._spring_kind == 0,
            34.0,
            np.where(self._spring_kind == 1, 17.0, 5.5),
        )
        damping = np.where(self._spring_kind == 2, 0.20, 0.10)
        magnitude = stiffness * (lengths - self._rest_lengths) + damping * relative_speed
        pair_force = magnitude[:, None] * direction
        np.add.at(force, self._edges_i, pair_force)
        np.add.at(force, self._edges_j, -pair_force)

        # Distributed attachment transfers the driven limb motion through
        # friction while permitting slip and compression around the elbow.
        force += self._mass[:, None] * (30.0 * (target - p) - 3.2 * v)
        force -= 0.35 * self._mass[:, None] * v
        gap, normal, body_velocity = self._contact_data(p, time)
        penetration = np.maximum(-gap, 0.0)
        relative_velocity = v - body_velocity
        inward = np.minimum(np.einsum("ij,ij->i", relative_velocity, normal), 0.0)
        normal_force = 1100.0 * penetration - 18.0 * inward
        force += normal_force[:, None] * normal
        tangential = relative_velocity - np.einsum(
            "ij,ij->i", relative_velocity, normal
        )[:, None] * normal
        force -= 0.32 * normal_force[:, None] * np.tanh(7.0 * tangential)
        self._last_external = self._mass[:, None] * self._gravity[None, :]
        self._last_contact = gap
        return force

    def _integrate(self, target: np.ndarray, time: float, dt: float) -> None:
        self._velocity += dt * self._forces(target, time) / self._mass[:, None]
        speed = np.linalg.norm(self._velocity, axis=1)
        fast = speed > 7.5
        self._velocity[fast] *= (7.5 / speed[fast])[:, None]
        self._position += dt * self._velocity

        # Position-level nonpenetration makes the explicit penalty robust at
        # the small high-curvature contact region.
        gap, normal, body_velocity = self._contact_data(self._position, time)
        bad_indices = np.flatnonzero(gap < 0.0)
        if len(bad_indices):
            self._position[bad_indices] -= gap[bad_indices, None] * normal[bad_indices]
            relative = self._velocity[bad_indices] - body_velocity[bad_indices]
            vn = np.einsum("ij,ij->i", relative, normal[bad_indices])
            approaching = vn < 0.0
            corrected = self._velocity[bad_indices].copy()
            corrected[approaching] -= (
                vn[approaching, None] * normal[bad_indices][approaching]
            )
            self._velocity[bad_indices] = 0.90 * corrected
        self._last_contact = self._contact_distances(self._position, time)

    def _deformation_gradient(self) -> np.ndarray:
        p = self._position.reshape(self._ny, self._nx, 3)
        out = np.zeros((self._ny, self._nx, 2, 2), dtype=float)
        for row in range(self._ny):
            for col in range(self._nx):
                c0, c1 = max(col - 1, 0), min(col + 1, self._nx - 1)
                r0, r1 = max(row - 1, 0), min(row + 1, self._ny - 1)
                du = (p[row, c1] - p[row, c0]) / max((c1 - c0) * self._du, 1e-9)
                dv = (p[r1, col] - p[r0, col]) / max((r1 - r0) * self._dv, 1e-9)
                lu, lv = np.linalg.norm(du), np.linalg.norm(dv)
                shear = float(np.dot(du, dv) / max(lu * lv, 1e-9))
                out[row, col] = np.array([[lu, shear], [shear, lv]])
        return np.nan_to_num(
            out.reshape(self._n, 2, 2), nan=0.0, posinf=10.0, neginf=-10.0
        )

    def _frame(self, time: float) -> StateFrame:
        frame = StateFrame(
            float(time),
            self._position.copy(),
            self._velocity.copy(),
            mass=self._mass.copy(),
            body_id=np.zeros(self._n, dtype=np.int32),
            attributes={
                "mass_quantity": self._mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": np.zeros(self._n, dtype=np.int32),
            },
            fields={
                "external_force": self._last_external.copy(),
                "contact_signed_distance": self._last_contact.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": self._thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        requested = float(time)
        if not np.isfinite(requested) or requested < 0.0:
            raise ValueError("time must be finite and non-negative")
        if requested < self._time - 1e-10:
            raise ValueError("advance_to requires nondecreasing times")
        if requested <= self._time + 1e-12:
            return self._frame(requested)
        remaining = requested - self._time
        while remaining > 1e-12:
            dt = min(1.0 / 240.0, remaining)
            mid = self._time + 0.5 * dt
            self._integrate(self._target_shape(mid), mid, dt)
            self._time += dt
            remaining = requested - self._time
        return self._frame(requested)

    def diagnostics(self) -> dict[str, object]:
        if not self._ready:
            return {}
        return {
            "integrator": "semi_implicit_euler_mass_spring",
            "particle_count": int(self._n),
            "internal_step_hz": 240,
            "joint_peak_angle_rad": 0.92,
            "final_time": float(self._time),
        }
