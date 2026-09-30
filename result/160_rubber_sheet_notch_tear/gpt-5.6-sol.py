"""Deterministic thin-shell solver for the notched rubber sheet scene.

The sheet is discretized as a triangular mass-spring membrane. Fracture is
represented by irreversible removal of cohesive mesh edges ahead of the notch;
the crack tip follows a rate-dependent Griffith-like strain threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit, substepped cloth dynamics with cohesive edge fracture."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model is deterministic; no stochastic forcing is used.
        self.scene = scene
        cfg = scene.config
        cloth = cfg["cloth"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.contact_scale = self.thickness
        self.nx, self.nz = map(int, cloth["grid_shape"])

        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.zeros_like(self.position)
        self.rest_position = self.position.copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.zeros(len(self.position), dtype=np.int32)
        self.time = 0.0

        asset = Path(__file__).resolve().parent.parent / "assets" / "cloth_topology.json"
        topology = json.loads(asset.read_text(encoding="utf-8"))
        self.faces = np.asarray(topology["faces"], dtype=np.int32)
        edge_set: set[tuple[int, int]] = set()
        for a, b, c in self.faces:
            edge_set.add(tuple(sorted((int(a), int(b)))))
            edge_set.add(tuple(sorted((int(b), int(c)))))
            edge_set.add(tuple(sorted((int(c), int(a)))))
        self.edges = np.asarray(sorted(edge_set), dtype=np.int32)
        delta = self.rest_position[self.edges[:, 1]] - self.rest_position[self.edges[:, 0]]
        self.rest_length = np.linalg.norm(delta, axis=1)
        self.edge_active = np.ones(len(self.edges), dtype=bool)

        z_values = np.unique(self.rest_position[:, 2])
        center_z = float(cloth["center"][2])
        lower = z_values[z_values <= center_z].max()
        upper = z_values[z_values > center_z].min()
        self.crack_z = 0.5 * (lower + upper)

        # Low-stiffness two-ring springs regularize curvature without preventing folds.
        edge_lookup = {tuple(edge) for edge in self.edges.tolist()}
        bend: list[tuple[int, int]] = []
        for j in range(self.nz):
            for i in range(self.nx - 2):
                a, m, b = j * self.nx + i, j * self.nx + i + 1, j * self.nx + i + 2
                if (a, m) in edge_lookup and (m, b) in edge_lookup:
                    bend.append((a, b))
        for j in range(self.nz - 2):
            for i in range(self.nx):
                a, m, b = j * self.nx + i, (j + 1) * self.nx + i, (j + 2) * self.nx + i
                if (a, m) in edge_lookup and (m, b) in edge_lookup:
                    bend.append((a, b))
        self.bend_edges = np.asarray(bend, dtype=np.int32).reshape((-1, 2))
        if len(self.bend_edges):
            d = self.rest_position[self.bend_edges[:, 1]] - self.rest_position[self.bend_edges[:, 0]]
            self.bend_rest = np.linalg.norm(d, axis=1)
            bend_z_a = self.rest_position[self.bend_edges[:, 0], 2]
            bend_z_b = self.rest_position[self.bend_edges[:, 1], 2]
            self.bend_crosses_crack = ((bend_z_a < self.crack_z) & (bend_z_b >= self.crack_z)) | (
                (bend_z_b < self.crack_z) & (bend_z_a >= self.crack_z)
            )
            self.bend_mid_x = 0.5 * (
                self.rest_position[self.bend_edges[:, 0], 0]
                + self.rest_position[self.bend_edges[:, 1], 0]
            )
        else:
            self.bend_rest = np.empty(0, dtype=float)
            self.bend_crosses_crack = np.empty(0, dtype=bool)
            self.bend_mid_x = np.empty(0, dtype=float)

        self.left = np.flatnonzero(np.isclose(self.rest_position[:, 0], self.rest_position[:, 0].min()))
        self.right = np.flatnonzero(np.isclose(self.rest_position[:, 0], self.rest_position[:, 0].max()))
        self.pinned = np.zeros(len(self.position), dtype=bool)
        self.pinned[self.left] = True
        self.pinned[self.right] = True

        # Existing topology determines the first intact cohesive bond and crack tip.
        za = self.rest_position[self.edges[:, 0], 2]
        zb = self.rest_position[self.edges[:, 1], 2]
        self.cohesive = ((za < self.crack_z) & (zb >= self.crack_z)) | (
            (zb < self.crack_z) & (za >= self.crack_z)
        )
        cohesive_x = 0.5 * (
            self.rest_position[self.edges[:, 0], 0] + self.rest_position[self.edges[:, 1], 0]
        )
        present = cohesive_x[self.cohesive]
        self.crack_front = float(present.min() - 1.0e-6) if len(present) else -0.75
        self.crack_start = self.crack_front
        self.crack_limit = float(self.rest_position[:, 0].max())
        self.broken_count = 0

        self.spring_k = 42.0
        self.spring_damping = 0.045
        self.bend_k = 0.65
        self.bend_damping = 0.012
        self.max_step = 0.0015
        self.grip_travel = 0.42
        self.last_external_force = self._external_force(0.0)

    @staticmethod
    def _smoothstep(x: float) -> tuple[float, float]:
        """Return cubic smoothstep and its derivative with respect to x."""
        if x <= 0.0:
            return 0.0, 0.0
        if x >= 1.0:
            return 1.0, 0.0
        return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x)

    def _grip_motion(self, time: float) -> tuple[float, float]:
        start, duration = 0.45, 4.25
        value, derivative = self._smoothstep((time - start) / duration)
        return self.grip_travel * value, self.grip_travel * derivative / duration

    def _external_force(self, time: float) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :]
        # A small transverse pressure imperfection lets real membrane forces wrinkle.
        x = self.rest_position[:, 0]
        z = self.rest_position[:, 2]
        phase = np.sin(np.pi * (x - x.min()) / max(float(np.ptp(x)), 1.0e-12))
        side = np.tanh((z - self.crack_z) / 0.12)
        pressure = 2.2e-4 * phase * side * (1.0 + 0.25 * np.sin(2.1 * time))
        force[:, 1] += pressure
        return force

    @staticmethod
    def _add_spring_forces(
        force: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        edges: np.ndarray,
        rest: np.ndarray,
        stiffness: float,
        damping: float,
        active: np.ndarray | None = None,
    ) -> None:
        if active is not None:
            edges = edges[active]
            rest = rest[active]
        if len(edges) == 0:
            return
        a, b = edges[:, 0], edges[:, 1]
        delta = position[b] - position[a]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length[:, None], 1.0e-12)
        relative_speed = np.einsum("ij,ij->i", velocity[b] - velocity[a], direction)
        magnitude = stiffness * (length - rest) + damping * relative_speed
        pair_force = direction * magnitude[:, None]
        np.add.at(force, a, pair_force)
        np.add.at(force, b, -pair_force)

    def _update_fracture(self, dt: float) -> None:
        active_idx = np.flatnonzero(self.edge_active)
        if not len(active_idx):
            return
        e = self.edges[active_idx]
        length = np.linalg.norm(self.position[e[:, 1]] - self.position[e[:, 0]], axis=1)
        strain = length / self.rest_length[active_idx] - 1.0
        midpoint_x = 0.5 * (
            self.rest_position[e[:, 0], 0] + self.rest_position[e[:, 1], 0]
        )
        midpoint_z = 0.5 * (
            self.rest_position[e[:, 0], 2] + self.rest_position[e[:, 1], 2]
        )
        process_zone = (
            (midpoint_x >= self.crack_front - 0.20)
            & (midpoint_x <= self.crack_front + 0.48)
            & (np.abs(midpoint_z - self.crack_z) < 0.34)
        )
        driving_strain = float(np.max(strain[process_zone])) if np.any(process_zone) else 0.0
        toughness = 0.065
        if driving_strain > toughness:
            self.crack_front = min(
                self.crack_limit + 0.05,
                self.crack_front + 9.0 * (driving_strain - toughness) * dt,
            )

        edge_mid_x = 0.5 * (
            self.rest_position[self.edges[:, 0], 0]
            + self.rest_position[self.edges[:, 1], 0]
        )
        new_break = self.edge_active & self.cohesive & (edge_mid_x <= self.crack_front)
        # A high terminal strain is a mesh-independent safety fracture criterion.
        overstrained = np.zeros(len(self.edges), dtype=bool)
        overstrained[active_idx] = strain > 0.72
        new_break |= self.edge_active & overstrained
        count = int(np.count_nonzero(new_break))
        if count:
            self.edge_active[new_break] = False
            self.broken_count += count

    def _apply_grips(self, time: float) -> None:
        displacement, speed = self._grip_motion(time)
        for indices, sign in ((self.left, -1.0), (self.right, 1.0)):
            self.position[indices] = self.rest_position[indices]
            self.position[indices, 0] += sign * displacement
            self.velocity[indices] = 0.0
            self.velocity[indices, 0] = sign * speed

    def _box_collision(self, center: np.ndarray, half: np.ndarray) -> None:
        rel = self.position - center
        inside = np.all(np.abs(rel) < half + 0.5 * self.thickness, axis=1) & ~self.pinned
        for idx in np.flatnonzero(inside):
            penetration = half + 0.5 * self.thickness - np.abs(rel[idx])
            axis = int(np.argmin(penetration))
            normal_sign = 1.0 if rel[idx, axis] >= 0.0 else -1.0
            self.position[idx, axis] = center[axis] + normal_sign * (
                half[axis] + 0.5 * self.thickness
            )
            normal_velocity = self.velocity[idx, axis] * normal_sign
            if normal_velocity < 0.0:
                self.velocity[idx, axis] -= 1.05 * normal_velocity * normal_sign
            tangential = [q for q in range(3) if q != axis]
            self.velocity[idx, tangential] *= 0.92

    def _resolve_contacts(self, time: float) -> None:
        floor = 0.5 * self.thickness
        hit = (self.position[:, 2] < floor) & ~self.pinned
        self.position[hit, 2] = floor
        downward = hit & (self.velocity[:, 2] < 0.0)
        self.velocity[downward, 2] *= -0.04
        self.velocity[hit, :2] *= 0.86

        displacement, _ = self._grip_motion(time)
        half = np.asarray([0.12, 0.18, 1.25], dtype=float)
        self._box_collision(np.asarray([-1.5 - displacement, 0.0, 1.6]), half)
        self._box_collision(np.asarray([1.5 + displacement, 0.0, 1.6]), half)

    def _substep(self, dt: float) -> None:
        next_time = self.time + dt
        force = self._external_force(next_time)
        external_force = force.copy()
        self._add_spring_forces(
            force, self.position, self.velocity, self.edges, self.rest_length,
            self.spring_k, self.spring_damping, self.edge_active,
        )
        bend_active = ~(self.bend_crosses_crack & (self.bend_mid_x <= self.crack_front))
        self._add_spring_forces(
            force, self.position, self.velocity, self.bend_edges, self.bend_rest,
            self.bend_k, self.bend_damping, bend_active,
        )
        free = ~self.pinned
        self.velocity[free] += dt * force[free] / self.mass[free, None]
        self.velocity[free] *= np.exp(-0.10 * dt)
        self.position[free] += dt * self.velocity[free]
        self._apply_grips(next_time)
        self._resolve_contacts(next_time)
        self._update_fracture(dt)
        self.last_external_force = external_force
        self.time = next_time

    def _deformation_gradient(self) -> np.ndarray:
        """Least-squares in-plane deformation gradient at each mesh vertex."""
        result = np.repeat(np.eye(2, dtype=float)[None, :, :], len(self.position), axis=0)
        neighbors: list[list[int]] = [[] for _ in range(len(self.position))]
        for edge in self.edges[self.edge_active]:
            a, b = map(int, edge)
            neighbors[a].append(b)
            neighbors[b].append(a)
        for i, adjacent in enumerate(neighbors):
            if len(adjacent) < 2:
                continue
            ids = np.asarray(adjacent, dtype=np.int32)
            rest_delta = self.rest_position[ids][:, [0, 2]] - self.rest_position[i, [0, 2]]
            current_delta = self.position[ids][:, [0, 2]] - self.position[i, [0, 2]]
            if np.linalg.matrix_rank(rest_delta) >= 2:
                mapping, _, _, _ = np.linalg.lstsq(rest_delta, current_delta, rcond=None)
                result[i] = mapping.T
        return result

    @staticmethod
    def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        q = np.abs(points - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _contact_distance(self) -> np.ndarray:
        displacement, _ = self._grip_motion(self.time)
        half = np.asarray([0.12, 0.18, 1.25], dtype=float) + 0.5 * self.thickness
        left = self._box_sdf(
            self.position, np.asarray([-1.5 - displacement, 0.0, 1.6]), half
        )
        right = self._box_sdf(
            self.position, np.asarray([1.5 + displacement, 0.0, 1.6]), half
        )
        ground = self.position[:, 2] - 0.5 * self.thickness
        return np.minimum(ground, np.minimum(left, right))

    def _frame(self, requested_time: float) -> StateFrame:
        incident = np.zeros(len(self.position), dtype=float)
        broken = ~self.edge_active
        np.add.at(incident, self.edges[broken, 0], 1.0)
        np.add.at(incident, self.edges[broken, 1], 1.0)
        frame = StateFrame(
            float(requested_time), self.position.copy(), self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.pinned.astype(np.int32),
                "fracture_damage": incident,
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_distance(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.last_external_force.copy(),
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time + 1.0e-12 < target:
            self._substep(min(self.max_step, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped triangular mass-spring membrane with cohesive fracture",
            "substep": self.max_step,
            "active_edges": int(np.count_nonzero(self.edge_active)),
            "broken_edges": int(self.broken_count),
            "crack_front_x": float(self.crack_front),
            "mass": float(np.sum(self.mass)),
        }
