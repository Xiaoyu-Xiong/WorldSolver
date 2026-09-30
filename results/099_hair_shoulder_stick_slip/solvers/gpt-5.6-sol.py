"""Deterministic elastic-strand solver for the hair/shoulder scene.

The implementation uses a small-step position based discretization of elastic
rods. Segment and two-edge constraints provide extension and bending
resistance, while unilateral projections and impulse-limited Coulomb friction
handle the moving body and neighboring hair layers.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.scene = scene
        self.config = scene.config
        self.time = 0.0
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id).copy()
        self.attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.n = len(self.x)
        self.strand_id = np.asarray(self.attributes["strand_id"], dtype=int)
        self.node_index = np.asarray(self.attributes["node_index"], dtype=int)
        self.root_mask = self.node_index == 0
        self.dynamic_mask = ~self.root_mask
        self.inv_mass = np.zeros(self.n, dtype=float)
        self.inv_mass[self.dynamic_mask] = 1.0 / self.mass[self.dynamic_mask]
        self.root_initial = self.x[self.root_mask].copy()

        self.nodes_per_strand = int(self.node_index.max()) + 1
        self.strand_count = int(self.strand_id.max()) + 1
        base = np.arange(self.strand_count, dtype=int) * self.nodes_per_strand
        self.edge_groups = []
        for parity in range(2):
            local = np.arange(parity, self.nodes_per_strand - 1, 2)
            a = (base[:, None] + local[None, :]).ravel()
            self.edge_groups.append((a, a + 1))
        self.bend_groups = []
        for color in range(3):
            local = np.arange(color, self.nodes_per_strand - 2, 3)
            a = (base[:, None] + local[None, :]).ravel()
            self.bend_groups.append((a, a + 2))

        all_a = (base[:, None] + np.arange(self.nodes_per_strand - 1)).ravel()
        self.edge_rest_grid = np.linalg.norm(
            self.x[all_a + 1] - self.x[all_a], axis=1
        ).reshape(self.strand_count, self.nodes_per_strand - 1)
        bend_a = (base[:, None] + np.arange(self.nodes_per_strand - 2)).ravel()
        self.bend_rest_grid = np.linalg.norm(
            self.x[bend_a + 2] - self.x[bend_a], axis=1
        ).reshape(self.strand_count, self.nodes_per_strand - 2)

        # Fixed neighboring-layer pairs give persistent, resolvable contact
        # identities while still turning on only at actual geometric contact.
        hp_i = []
        hp_j = []
        for layer in range(5):
            for ring in range(14):
                s0 = layer * 14 + ring
                s1 = (layer + 1) * 14 + ring
                for node in range(1, self.nodes_per_strand):
                    hp_i.append(s0 * self.nodes_per_strand + node)
                    hp_j.append(s1 * self.nodes_per_strand + node)
        self.hp_i = np.asarray(hp_i, dtype=int)
        self.hp_j = np.asarray(hp_j, dtype=int)
        self.hair_radius = float(initial.scalars.get("strand_radius", 0.0042))
        self.contact_distance = 2.0 * self.hair_radius
        self.mu = float(self.config["friction"]["hair_body"]["static"])

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        motion = self.config["head_motion"]
        self.motion_axis = np.asarray(motion["axis"], dtype=float)
        self.motion_axis /= max(np.linalg.norm(self.motion_axis), 1.0e-12)
        self.motion_distance = float(motion["displacement"])
        self.motion_start = float(motion["start_time"])
        self.motion_stop = float(motion["stop_time"])

        self.max_step = 0.0025
        self.constraint_iterations = 5
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=float)
        self.last_average_force = np.sum(self.mass[:, None] * self.gravity, axis=0)
        self.last_body_normal = np.zeros((self.n, 3), dtype=float)
        self.last_body_normal[:, 2] = 1.0
        self.last_body_nf = np.zeros(self.n, dtype=float)
        self.last_body_rel = np.zeros((self.n, 3), dtype=float)
        self.last_body_ft = np.zeros((self.n, 3), dtype=float)
        pcount = len(self.hp_i)
        self.last_hair_normal = np.zeros((pcount, 3), dtype=float)
        self.last_hair_normal[:, 2] = 1.0
        self.last_hair_nf = np.zeros(pcount, dtype=float)
        self.last_hair_rel = np.zeros((pcount, 3), dtype=float)
        self.last_hair_ft = np.zeros((pcount, 3), dtype=float)
        total_mass = float(np.sum(self.mass))
        self.reference_velocity = max(
            self.motion_distance / max(self.motion_stop - self.motion_start, 1.0e-6),
            0.5,
        )
        self.reference_force = max(total_mass * np.linalg.norm(self.gravity), 1.0e-6)
        self.reference_momentum = max(total_mass * self.reference_velocity, 1.0e-6)
        self.step_count = 0

    def _motion(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        if time <= self.motion_start:
            return np.zeros(3), np.zeros(3)
        if time >= self.motion_stop:
            return self.motion_axis * self.motion_distance, np.zeros(3)
        duration = self.motion_stop - self.motion_start
        u = (time - self.motion_start) / duration
        smooth = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        ds_du = 30.0 * u * u * (1.0 - u) * (1.0 - u)
        return (
            self.motion_axis * (self.motion_distance * smooth),
            self.motion_axis * (self.motion_distance * ds_du / duration),
        )

    @staticmethod
    def _capsule_data(
        points: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        half: float,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = points - center
        along = np.clip(local @ axis, -half, half)
        closest = center + along[:, None] * axis
        delta = points - closest
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1.0e-12)
        bad = distance < 1.0e-10
        if np.any(bad):
            normal[bad] = np.array([0.0, -1.0, 0.0])
        return distance - radius, normal, closest + radius * normal

    @staticmethod
    def _ellipsoid_data(
        points: np.ndarray, center: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = points - center
        q = np.linalg.norm(local / radii, axis=1)
        qsafe = np.maximum(q, 1.0e-12)
        surface = center + local / qsafe[:, None]
        grad = (surface - center) / (radii * radii)
        normal = grad / np.maximum(np.linalg.norm(grad, axis=1)[:, None], 1.0e-12)
        bad = q < 1.0e-10
        if np.any(bad):
            normal[bad] = np.array([0.0, -1.0, 0.0])
            surface[bad] = center + np.array([0.0, -radii[1], 0.0])
        signed = (q - 1.0) * np.min(radii)
        return signed, normal, surface

    def _body_candidates(
        self, points: np.ndarray, time: float
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        shift, _ = self._motion(time)
        skin = self.hair_radius
        return [
            self._capsule_data(
                points,
                np.array([0.0, 0.0, 1.02]) + shift,
                np.array([1.0, 0.0, 0.0]),
                0.46,
                0.145 + skin,
            ),
            self._capsule_data(
                points,
                np.array([0.0, 0.0, 1.28]) + shift,
                np.array([0.0, 0.0, 1.0]),
                0.14,
                0.115 + skin,
            ),
            self._ellipsoid_data(
                points,
                np.array([0.0, 0.0, 1.62]) + shift,
                np.array([0.265 * 0.88, 0.265 * 0.95, 0.265 * 1.12]) + skin,
            ),
            self._ellipsoid_data(
                points,
                np.array([0.0, 0.0, 0.62]) + shift,
                np.array([0.34, 0.34 * 0.72, 0.34 + 0.31]) + skin,
            ),
        ]

    def _project_bodies(self, time: float, maximum_penetration: np.ndarray) -> None:
        for signed, _normal, surface in self._body_candidates(self.x, time):
            hit = (signed < 0.0) & self.dynamic_mask
            if np.any(hit):
                maximum_penetration[hit] = np.maximum(
                    maximum_penetration[hit], -signed[hit]
                )
                self.x[hit] = surface[hit]

    def _solve_distances(
        self, a: np.ndarray, b: np.ndarray, rest: np.ndarray, stiffness: float
    ) -> None:
        delta = self.x[b] - self.x[a]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length[:, None], 1.0e-12)
        weight = self.inv_mass[a] + self.inv_mass[b]
        scale = stiffness * (length - rest) / np.maximum(weight, 1.0e-30)
        correction = scale[:, None] * direction
        self.x[a] += self.inv_mass[a, None] * correction
        self.x[b] -= self.inv_mass[b, None] * correction

    def _project_hair_contacts(self, maximum_penetration: np.ndarray) -> None:
        delta = self.x[self.hp_i] - self.x[self.hp_j]
        distance = np.linalg.norm(delta, axis=1)
        hit = distance < self.contact_distance
        if not np.any(hit):
            return
        normal = delta / np.maximum(distance[:, None], 1.0e-12)
        normal[distance < 1.0e-10] = np.array([0.0, 1.0, 0.0])
        depth = np.maximum(self.contact_distance - distance, 0.0)
        maximum_penetration[:] = np.maximum(maximum_penetration, depth)
        weight = self.inv_mass[self.hp_i] + self.inv_mass[self.hp_j]
        corr = 0.55 * depth[:, None] * normal / np.maximum(weight[:, None], 1.0e-30)
        move_i = self.inv_mass[self.hp_i, None] * corr
        move_j = -self.inv_mass[self.hp_j, None] * corr
        np.add.at(self.x, self.hp_i[hit], move_i[hit])
        np.add.at(self.x, self.hp_j[hit], move_j[hit])

    def _body_velocity_contact(
        self, time: float, dt: float, penetration: np.ndarray
    ) -> None:
        candidates = self._body_candidates(self.x, time)
        gaps = np.stack([item[0] for item in candidates], axis=0)
        which = np.argmin(gaps, axis=0)
        min_gap = np.min(gaps, axis=0)
        normals = np.empty((self.n, 3), dtype=float)
        for ci, item in enumerate(candidates):
            select = which == ci
            normals[select] = item[1][select]
        active = (min_gap < 7.5e-4) & self.dynamic_mask
        _, body_velocity = self._motion(time)
        relative = self.v - body_velocity
        vn = np.einsum("ij,ij->i", relative, normals)
        inward_speed = np.maximum(-vn, 0.0)
        normal_impulse = self.mass * inward_speed + self.mass * penetration / dt
        normal_impulse[~active] = 0.0
        self.v += inward_speed[:, None] * normals * active[:, None]

        relative = self.v - body_velocity
        tangent = relative - np.einsum("ij,ij->i", relative, normals)[:, None] * normals
        tangent_speed = np.linalg.norm(tangent, axis=1)
        desired = self.mass * tangent_speed
        friction_impulse = np.minimum(desired, self.mu * normal_impulse)
        direction = -tangent / np.maximum(tangent_speed[:, None], 1.0e-12)
        impulse_vector = friction_impulse[:, None] * direction
        self.v[self.dynamic_mask] += (
            impulse_vector[self.dynamic_mask] / self.mass[self.dynamic_mask, None]
        )

        self.last_body_normal = normals
        self.last_body_nf = normal_impulse / dt
        self.last_body_rel = relative
        self.last_body_ft = impulse_vector / dt

    def _hair_velocity_contact(self, dt: float, penetration: np.ndarray) -> None:
        i = self.hp_i
        j = self.hp_j
        delta = self.x[i] - self.x[j]
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1.0e-12)
        normal[distance < 1.0e-10] = np.array([0.0, 1.0, 0.0])
        active = distance < self.contact_distance + 5.0e-4
        relative = self.v[i] - self.v[j]
        vn = np.einsum("ij,ij->i", relative, normal)
        inv_sum = self.inv_mass[i] + self.inv_mass[j]
        effective_mass = 1.0 / np.maximum(inv_sum, 1.0e-30)
        normal_impulse = effective_mass * (
            np.maximum(-vn, 0.0) + penetration / dt
        )
        normal_impulse[~active] = 0.0

        impulse_n = normal_impulse[:, None] * normal
        np.add.at(self.v, i[active], (self.inv_mass[i, None] * impulse_n)[active])
        np.add.at(self.v, j[active], (-self.inv_mass[j, None] * impulse_n)[active])

        relative = self.v[i] - self.v[j]
        tangent = relative - np.einsum("ij,ij->i", relative, normal)[:, None] * normal
        tangent_speed = np.linalg.norm(tangent, axis=1)
        desired_impulse = effective_mass * tangent_speed
        friction_impulse = np.minimum(desired_impulse, self.mu * normal_impulse)
        tangent_direction = -tangent / np.maximum(tangent_speed[:, None], 1.0e-12)
        impulse = friction_impulse[:, None] * tangent_direction
        np.add.at(self.v, i[active], (self.inv_mass[i, None] * impulse)[active])
        np.add.at(self.v, j[active], (-self.inv_mass[j, None] * impulse)[active])

        self.last_hair_normal = normal
        self.last_hair_nf = normal_impulse / dt
        self.last_hair_rel = relative
        self.last_hair_ft = impulse / dt

    def _substep(self, dt: float) -> np.ndarray:
        old_x = self.x.copy()
        momentum_before = np.sum(self.mass[:, None] * self.v, axis=0)
        end_time = self.time + dt

        self.v[self.dynamic_mask] += self.gravity * dt
        self.v[self.dynamic_mask] *= np.exp(-0.32 * dt)
        self.x[self.dynamic_mask] += self.v[self.dynamic_mask] * dt
        shift, root_velocity = self._motion(end_time)
        self.x[self.root_mask] = self.root_initial + shift

        body_penetration = np.zeros(self.n, dtype=float)
        hair_penetration = np.zeros(len(self.hp_i), dtype=float)
        for _ in range(self.constraint_iterations):
            for parity, (a, b) in enumerate(self.edge_groups):
                rest = self.edge_rest_grid[:, parity::2].ravel()
                self._solve_distances(a, b, rest, 0.92)
            for color, (a, b) in enumerate(self.bend_groups):
                rest = self.bend_rest_grid[:, color::3].ravel()
                self._solve_distances(a, b, rest, 0.075)
            self._project_hair_contacts(hair_penetration)
            self._project_bodies(end_time, body_penetration)
            self.x[self.root_mask] = self.root_initial + shift

        self.v = (self.x - old_x) / dt
        self.v[self.root_mask] = root_velocity
        self._body_velocity_contact(end_time, dt, body_penetration)
        self._hair_velocity_contact(dt, hair_penetration)
        self.v[self.root_mask] = root_velocity
        np.nan_to_num(self.x, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.nan_to_num(self.v, copy=False, nan=0.0, posinf=50.0, neginf=-50.0)
        np.clip(self.v, -50.0, 50.0, out=self.v)

        momentum_after = np.sum(self.mass[:, None] * self.v, axis=0)
        actual_external_impulse = momentum_after - momentum_before
        self.cumulative_external_impulse[0] += actual_external_impulse
        self.time = end_time
        self.step_count += 1
        return actual_external_impulse

    def _make_frame(self, time: float) -> StateFrame:
        active = np.concatenate(
            (self.last_body_nf > 0.0, self.last_hair_nf > 0.0)
        ).astype(float)
        normal = np.concatenate((self.last_body_normal, self.last_hair_normal), axis=0)
        normal_force = np.concatenate((self.last_body_nf, self.last_hair_nf))
        relative_velocity = np.concatenate(
            (self.last_body_rel, self.last_hair_rel), axis=0
        )
        tangential_force = np.concatenate(
            (self.last_body_ft, self.last_hair_ft), axis=0
        )
        contact_count = len(active)
        attributes = {key: value.copy() for key, value in self.attributes.items()}
        attributes["mass_quantity"] = self.mass.copy()
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": self.last_average_force.reshape(1, 1, 3).copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "friction_active_mask": active,
            "friction_dynamic_coefficient": np.full(contact_count, self.mu),
            "friction_normal": normal,
            "friction_normal_force": normal_force,
            "friction_relative_velocity": relative_velocity,
            "friction_static_coefficient": np.full(contact_count, self.mu),
            "friction_tangential_force": tangential_force,
        }
        scalars = {
            "strand_radius": self.hair_radius,
            "static_friction": self.mu,
            "kinetic_friction": self.mu,
            "head_stop_time": self.motion_stop,
            "friction_stick_speed_tolerance": 0.015,
            "reference_force": self.reference_force,
            "reference_momentum": self.reference_momentum,
            "reference_velocity": self.reference_velocity,
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to requires nondecreasing sample times")
        interval = target - self.time
        interval_impulse = np.zeros(3, dtype=float)
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            interval_impulse += self._substep(dt)
        self.time = target
        if interval > 1.0e-12:
            self.last_average_force = interval_impulse / interval
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_elastic_rods_with_coulomb_impulses",
            "substeps": self.step_count,
            "particle_count": self.n,
            "hair_contact_candidates": len(self.hp_i),
            "finite": bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.v))),
        }
