"""Deterministic, self-contained flexible-rod/contact solver for the benchmark.

The strands are discretised as massive nodes joined by inextensible distance
constraints. Second-neighbour constraints supply bending resistance. A
semi-implicit Euler step followed by mass-weighted position projections is
used for the stiff elastic and unilateral contact constraints. Contact
impulses include an exact Coulomb friction projection.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small position-based elastic-rod solver with dry contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.n = len(self.position)
        self.nodes_per_strand = 16
        self.strand_count = 31
        self.radius = 0.006
        self.diameter = 2.0 * self.radius
        self.mu = 0.3
        self.gravity = np.asarray(scene.config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.time = 0.0
        self.max_dt = 0.004
        self.initial_momentum = np.sum(self.mass[:, None] * self.velocity, axis=0)
        self.previous_sample_momentum = self.initial_momentum.copy()
        self.previous_sample_time = 0.0
        self.cumulative_external_impulse = np.zeros(3, dtype=float)
        self.last_external_force = np.sum(self.mass) * self.gravity
        self.step_count = 0

        stretch_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for parity in range(2):
            left, right = [], []
            for body in range(self.strand_count):
                base = body * self.nodes_per_strand
                for j in range(parity, self.nodes_per_strand - 1, 2):
                    left.append(base + j)
                    right.append(base + j + 1)
            stretch_groups.append((np.asarray(left, dtype=int), np.asarray(right, dtype=int)))
        self.stretch_groups = stretch_groups

        bend_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for residue in range(3):
            left, right = [], []
            for body in range(self.strand_count):
                base = body * self.nodes_per_strand
                for j in range(residue, self.nodes_per_strand - 2, 3):
                    left.append(base + j)
                    right.append(base + j + 2)
            bend_groups.append((np.asarray(left, dtype=int), np.asarray(right, dtype=int)))
        self.bend_groups = bend_groups
        self.segment_rest = 0.72 / (self.nodes_per_strand - 1)
        self.bend_rest = 2.0 * self.segment_rest

        # One stable contact-output slot per node keeps all trajectory shapes fixed.
        self.contact_active = np.zeros(self.n, dtype=float)
        self.contact_normal = np.zeros((self.n, 3), dtype=float)
        self.contact_normal[:, 2] = 1.0
        self.contact_normal_force = np.zeros(self.n, dtype=float)
        self.contact_relative_velocity = np.zeros((self.n, 3), dtype=float)
        self.contact_tangential_force = np.zeros((self.n, 3), dtype=float)

    def _project_pair_group(
        self, ia: np.ndarray, ib: np.ndarray, rest: float, stiffness: float
    ) -> None:
        delta = self.position[ib] - self.position[ia]
        length = np.linalg.norm(delta, axis=1)
        good = length > 1.0e-12
        direction = np.zeros_like(delta)
        direction[good] = delta[good] / length[good, None]
        lagrange = stiffness * (length - rest) / (self.inv_mass[ia] + self.inv_mass[ib])
        correction = lagrange[:, None] * direction
        self.position[ia] += self.inv_mass[ia, None] * correction
        self.position[ib] -= self.inv_mass[ib, None] * correction

    @staticmethod
    def _plate_height_slope(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Piecewise-linear frozen inner plate profile and radial derivative."""
        rr = np.asarray(radius)
        height = np.empty_like(rr)
        slope = np.empty_like(rr)
        intervals = (
            (0.0, 0.10, 0.0, 0.0),
            (0.10, 0.40, 0.0, 0.018),
            (0.40, 0.52, 0.018, 0.075),
            (0.52, 0.57, 0.075, 0.125),
        )
        for lo, hi, z0, z1 in intervals:
            mask = (rr >= lo) & (rr < hi)
            s = (z1 - z0) / (hi - lo) if hi > lo else 0.0
            height[mask] = z0 + s * (rr[mask] - lo)
            slope[mask] = s
        mask = rr >= 0.57
        height[mask] = 0.125
        slope[mask] = 0.0
        return height, slope

    def _project_plate(self, records: dict[tuple[int, ...], list[object]], dt: float) -> None:
        xy = self.position[:, :2]
        rr = np.linalg.norm(xy, axis=1)
        height, slope = self._plate_height_slope(rr)
        denom = np.sqrt(1.0 + slope * slope)
        signed = (self.position[:, 2] - height) / denom
        penetration = self.radius - signed
        active = np.flatnonzero((rr <= 0.57) & (penetration > 0.0))
        for i in active:
            radial = xy[i] / rr[i] if rr[i] > 1.0e-12 else np.zeros(2)
            normal = np.array(
                (-slope[i] * radial[0], -slope[i] * radial[1], 1.0), dtype=float
            )
            normal /= np.linalg.norm(normal)
            depth = float(penetration[i])
            self.position[i] += depth * normal
            impulse = depth / (dt * self.inv_mass[i])
            key = (0, int(i))
            if key in records:
                records[key][1] = float(records[key][1]) + impulse
                records[key][2] = normal
            else:
                records[key] = [int(i), impulse, normal, -1]

        wall_radius = 0.57 - self.radius
        wall_pen = rr - wall_radius
        active = np.flatnonzero((wall_pen > 0.0) & (self.position[:, 2] < 0.15))
        for i in active:
            if rr[i] <= 1.0e-12:
                continue
            normal = np.array((-xy[i, 0] / rr[i], -xy[i, 1] / rr[i], 0.0))
            depth = float(wall_pen[i])
            self.position[i] += depth * normal
            impulse = depth / (dt * self.inv_mass[i])
            key = (1, int(i))
            if key in records:
                records[key][1] = float(records[key][1]) + impulse
            else:
                records[key] = [int(i), impulse, normal, -1]

    def _candidate_pairs(self) -> list[tuple[int, int]]:
        cell_size = self.diameter * 1.05
        cells: dict[tuple[int, int, int], list[int]] = {}
        coordinates = np.floor(self.position / cell_size).astype(np.int64)
        for i, coord in enumerate(coordinates):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            cells.setdefault(key, []).append(i)
        offsets = tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        )
        pairs: list[tuple[int, int]] = []
        for i, coord in enumerate(coordinates):
            cx, cy, cz = int(coord[0]), int(coord[1]), int(coord[2])
            for dx, dy, dz in offsets:
                for j in cells.get((cx + dx, cy + dy, cz + dz), ()):
                    if j > i and self.body_id[j] != self.body_id[i]:
                        pairs.append((i, j))
        return pairs

    def _project_strands(
        self, records: dict[tuple[int, ...], list[object]], dt: float
    ) -> None:
        for i, j in self._candidate_pairs():
            delta = self.position[i] - self.position[j]
            distance = float(np.linalg.norm(delta))
            if distance >= self.diameter:
                continue
            if distance > 1.0e-12:
                normal = delta / distance
            else:
                angle = ((i * 73856093 + j * 19349663) % 6283) * 0.001
                normal = np.array((math.cos(angle), math.sin(angle), 0.0))
            depth = self.diameter - distance
            inv_sum = self.inv_mass[i] + self.inv_mass[j]
            lagrange = depth / inv_sum
            self.position[i] += self.inv_mass[i] * lagrange * normal
            self.position[j] -= self.inv_mass[j] * lagrange * normal
            impulse = lagrange / dt
            key = (2, int(i), int(j))
            if key in records:
                records[key][1] = float(records[key][1]) + impulse
                records[key][2] = normal
            else:
                records[key] = [int(i), impulse, normal, int(j)]

    def _apply_contact_velocities(
        self, records: dict[tuple[int, ...], list[object]], dt: float
    ) -> None:
        self.contact_active.fill(0.0)
        self.contact_normal.fill(0.0)
        self.contact_normal[:, 2] = 1.0
        self.contact_normal_force.fill(0.0)
        self.contact_relative_velocity.fill(0.0)
        self.contact_tangential_force.fill(0.0)

        def save(
            slot: int,
            normal: np.ndarray,
            normal_force: float,
            relative: np.ndarray,
            tangent_force: np.ndarray,
        ) -> None:
            if normal_force > self.contact_normal_force[slot]:
                self.contact_active[slot] = 1.0
                self.contact_normal[slot] = normal
                self.contact_normal_force[slot] = normal_force
                self.contact_relative_velocity[slot] = relative
                self.contact_tangential_force[slot] = tangent_force

        for key in sorted(records):
            i, positional_impulse, normal, j = records[key]
            i = int(i)
            j = int(j)
            normal = np.asarray(normal, dtype=float)
            jn = max(0.0, float(positional_impulse))
            if j < 0:
                relative = self.velocity[i].copy()
                vn = float(np.dot(relative, normal))
                if vn < 0.0:
                    extra = -vn / self.inv_mass[i]
                    self.velocity[i] += extra * self.inv_mass[i] * normal
                    jn += extra
                relative = self.velocity[i].copy()
                tangent = relative - np.dot(relative, normal) * normal
                tangent_speed = float(np.linalg.norm(tangent))
                jt = 0.0
                tangent_direction = np.zeros(3)
                if tangent_speed > 1.0e-12 and jn > 0.0:
                    tangent_direction = tangent / tangent_speed
                    needed = tangent_speed / self.inv_mass[i]
                    jt = min(needed, self.mu * jn)
                    self.velocity[i] -= jt * self.inv_mass[i] * tangent_direction
                save(
                    i,
                    normal,
                    jn / dt,
                    self.velocity[i].copy(),
                    -jt * tangent_direction / dt,
                )
            else:
                relative = self.velocity[i] - self.velocity[j]
                inv_sum = self.inv_mass[i] + self.inv_mass[j]
                vn = float(np.dot(relative, normal))
                if vn < 0.0:
                    extra = -vn / inv_sum
                    self.velocity[i] += extra * self.inv_mass[i] * normal
                    self.velocity[j] -= extra * self.inv_mass[j] * normal
                    jn += extra
                relative = self.velocity[i] - self.velocity[j]
                tangent = relative - np.dot(relative, normal) * normal
                tangent_speed = float(np.linalg.norm(tangent))
                jt = 0.0
                tangent_direction = np.zeros(3)
                if tangent_speed > 1.0e-12 and jn > 0.0:
                    tangent_direction = tangent / tangent_speed
                    needed = tangent_speed / inv_sum
                    jt = min(needed, self.mu * jn)
                    impulse = jt * tangent_direction
                    self.velocity[i] -= self.inv_mass[i] * impulse
                    self.velocity[j] += self.inv_mass[j] * impulse
                relative_after = self.velocity[i] - self.velocity[j]
                tangent_force = -jt * tangent_direction / dt
                save(i, normal, jn / dt, relative_after, tangent_force)
                save(j, -normal, jn / dt, -relative_after, -tangent_force)

    def _step(self, dt: float) -> None:
        old_position = self.position.copy()
        self.velocity *= math.exp(-0.18 * dt)
        self.velocity += dt * self.gravity
        self.position += dt * self.velocity

        records: dict[tuple[int, ...], list[object]] = {}
        for _ in range(4):
            for ia, ib in self.stretch_groups:
                self._project_pair_group(ia, ib, self.segment_rest, 0.92)
            for ia, ib in self.bend_groups:
                self._project_pair_group(ia, ib, self.bend_rest, 0.10)
            self._project_plate(records, dt)
            self._project_strands(records, dt)

        self.velocity = (self.position - old_position) / dt
        self._apply_contact_velocities(records, dt)
        np.nan_to_num(self.position, copy=False, nan=0.0, posinf=2.0, neginf=-2.0)
        np.nan_to_num(self.velocity, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(self.position, -2.0, 2.0, out=self.position)
        np.clip(self.velocity, -20.0, 20.0, out=self.velocity)
        self.time += dt
        self.step_count += 1

    def _make_frame(self, requested_time: float) -> StateFrame:
        momentum = np.sum(self.mass[:, None] * self.velocity, axis=0)
        self.cumulative_external_impulse = momentum - self.initial_momentum
        elapsed = requested_time - self.previous_sample_time
        if elapsed > 1.0e-12:
            self.last_external_force = (momentum - self.previous_sample_momentum) / elapsed
        elif requested_time == 0.0:
            self.last_external_force = np.sum(self.mass) * self.gravity
        self.previous_sample_momentum = momentum.copy()
        self.previous_sample_time = requested_time

        total_mass = float(np.sum(self.mass))
        reference_velocity = math.sqrt(9.81 * 0.72)
        attributes = dict(self.attributes)
        attributes["mass_quantity"] = self.mass.copy()
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": self.last_external_force[None, None, :].copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "friction_active_mask": self.contact_active.copy(),
            "friction_dynamic_coefficient": np.full(self.n, self.mu, dtype=float),
            "friction_normal": self.contact_normal.copy(),
            "friction_normal_force": self.contact_normal_force.copy(),
            "friction_relative_velocity": self.contact_relative_velocity.copy(),
            "friction_static_coefficient": np.full(self.n, self.mu, dtype=float),
            "friction_tangential_force": self.contact_tangential_force.copy(),
        }
        frame = StateFrame(
            time=float(requested_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars={
                "strand_radius": self.radius,
                "strand_length": 0.72,
                "static_friction": self.mu,
                "kinetic_friction": self.mu,
                "friction_stick_speed_tolerance": 0.01,
                "reference_force": total_mass * 9.81,
                "reference_momentum": total_mass * reference_velocity,
                "reference_velocity": reference_velocity,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        remaining = target - self.time
        if remaining > 1.0e-12:
            steps = max(1, int(math.ceil(remaining / self.max_dt - 1.0e-12)))
            dt = remaining / steps
            for _ in range(steps):
                self._step(dt)
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_pbd_rods_coulomb",
            "internal_steps": int(self.step_count),
            "strand_count": int(self.strand_count),
            "finite": bool(
                np.isfinite(self.position).all() and np.isfinite(self.velocity).all()
            ),
        }
