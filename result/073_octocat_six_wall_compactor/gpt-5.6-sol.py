"""Deterministic elastic-solid solver for the six-wall compactor scene.

The Octocat's closed triangle surface is treated as the boundary of an elastic
solid. Stretching is discretized by springs on every mesh edge and resistance
to bulk compression is supplied by the signed-volume energy of the closed
surface. A semi-implicit time integrator advances those forces, followed by
inelastic unilateral contact with the six moving planes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit elastodynamics with moving-plane contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.rest_x = self.x.copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).reshape(-1).copy()
        self.inv_mass = 1.0 / np.maximum(self.mass, 1.0e-15)
        self.n = len(self.x)
        self.time = 0.0

        self.faces = self._load_faces()
        edge_pairs = np.concatenate(
            (self.faces[:, (0, 1)], self.faces[:, (1, 2)], self.faces[:, (2, 0)]), axis=0
        )
        edge_pairs.sort(axis=1)
        self.edges = np.unique(edge_pairs, axis=0)
        self.edge_i = self.edges[:, 0]
        self.edge_j = self.edges[:, 1]
        rest_delta = self.x[self.edge_j] - self.x[self.edge_i]
        self.rest_length = np.maximum(np.linalg.norm(rest_delta, axis=1), 1.0e-9)
        self.rest_volume = self._signed_volume(self.x)
        if abs(self.rest_volume) < 1.0e-10:
            self.rest_volume = 1.0

        mean_mass = float(np.mean(self.mass))
        self.edge_stiffness = 0.040
        self.edge_damping = 0.32 * np.sqrt(self.edge_stiffness * mean_mass * 0.5)
        self.bulk_modulus = 1.8
        self.max_step = 0.0025
        self.contact_friction_rate = 5.0
        self.self_contact_radius = 0.018
        self.self_contact_interval = 4
        self.gravity = np.asarray(scene.config.get("gravity", (0.0, 0.0, 0.0)), dtype=np.float64)

        # Report deformation gradients on a fixed, uniformly distributed subset
        # of the surface elements to bound trajectory size.
        ne = min(len(self.faces), 8192)
        element_ids = np.linspace(0, len(self.faces) - 1, ne, dtype=np.int64)
        self.element_faces = self.faces[element_ids]
        tri = self.x[self.element_faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        normal_len = np.maximum(np.linalg.norm(normal, axis=1), 1.0e-12)
        self.element_thickness = np.sqrt(normal_len)
        e3 = normal / normal_len[:, None] * self.element_thickness[:, None]
        dm = np.stack((e1, e2, e3), axis=2)
        self.element_inv_dm = np.linalg.inv(dm)
        self.element_rest_normal = normal / normal_len[:, None]

        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.last_external_force = np.zeros(3, dtype=np.float64)
        self.step_count = 0

    def _load_faces(self) -> np.ndarray:
        mesh_path = Path(__file__).resolve().parent.parent / "assets" / "octocat.obj"
        faces: list[tuple[int, int, int]] = []
        with mesh_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("f "):
                    fields = line.split()[1:]
                    indices = [int(field.split("/", 1)[0]) - 1 for field in fields]
                    for k in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[k], indices[k + 1]))
        return np.asarray(faces, dtype=np.int64)

    def _sum_by_vertex(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        result = np.empty((self.n, 3), dtype=np.float64)
        for axis in range(3):
            result[:, axis] = np.bincount(indices, weights=values[:, axis], minlength=self.n)
        return result

    def _signed_volume(self, positions: np.ndarray) -> float:
        tri = positions[self.faces]
        return float(
            np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
        )

    def _elastic_forces(self) -> np.ndarray:
        delta = self.x[self.edge_j] - self.x[self.edge_i]
        length = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-12)
        direction = delta / length[:, None]
        relative_speed = np.einsum(
            "ij,ij->i", self.v[self.edge_j] - self.v[self.edge_i], direction
        )
        magnitude = self.edge_stiffness * (length - self.rest_length)
        magnitude += self.edge_damping * relative_speed
        edge_force = magnitude[:, None] * direction
        force = self._sum_by_vertex(self.edge_i, edge_force)
        force -= self._sum_by_vertex(self.edge_j, edge_force)

        # Gradient of the translation-invariant signed-volume energy.
        tri = self.x[self.faces]
        current_volume = float(
            np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
        )
        pressure = -self.bulk_modulus * (current_volume - self.rest_volume) / abs(self.rest_volume)
        g0 = np.cross(tri[:, 1], tri[:, 2]) / 6.0
        g1 = np.cross(tri[:, 2], tri[:, 0]) / 6.0
        g2 = np.cross(tri[:, 0], tri[:, 1]) / 6.0
        volume_gradient = self._sum_by_vertex(self.faces[:, 0], g0)
        volume_gradient += self._sum_by_vertex(self.faces[:, 1], g1)
        volume_gradient += self._sum_by_vertex(self.faces[:, 2], g2)
        force += pressure * volume_gradient
        return force

    def _apply_wall_contact(self, half_extent: float, wall_speed: float, dt: float) -> np.ndarray:
        impulse = np.zeros(3, dtype=np.float64)
        tangential_factor = max(0.0, 1.0 - self.contact_friction_rate * dt)
        for axis in range(3):
            high = self.x[:, axis] > half_extent
            if np.any(high):
                before = self.v[high].copy()
                self.x[high, axis] = half_extent
                self.v[high, axis] = np.minimum(self.v[high, axis], wall_speed)
                tangent_axes = [a for a in range(3) if a != axis]
                self.v[np.ix_(high, tangent_axes)] *= tangential_factor
                impulse += np.sum(self.mass[high, None] * (self.v[high] - before), axis=0)

            low = self.x[:, axis] < -half_extent
            if np.any(low):
                before = self.v[low].copy()
                self.x[low, axis] = -half_extent
                self.v[low, axis] = np.maximum(self.v[low, axis], -wall_speed)
                tangent_axes = [a for a in range(3) if a != axis]
                self.v[np.ix_(low, tangent_axes)] *= tangential_factor
                impulse += np.sum(self.mass[low, None] * (self.v[low] - before), axis=0)
        return impulse

    def _apply_self_contact(self, dt: float) -> None:
        """Resolve sparse nonlocal vertex contacts with symmetric PBD impulses."""
        pairs = cKDTree(self.x).query_pairs(self.self_contact_radius, output_type="ndarray")
        if len(pairs) == 0:
            return
        # Nearby vertices in material coordinates belong to the same patch and
        # are already governed by elasticity, rather than self-contact.
        rest_delta = self.rest_x[pairs[:, 1]] - self.rest_x[pairs[:, 0]]
        nonlocal_pair = np.einsum("ij,ij->i", rest_delta, rest_delta) > (
            2.5 * self.self_contact_radius
        ) ** 2
        pairs = pairs[nonlocal_pair]
        if len(pairs) == 0:
            return

        delta = self.x[pairs[:, 1]] - self.x[pairs[:, 0]]
        distance = np.linalg.norm(delta, axis=1)
        direction = np.empty_like(delta)
        safe = distance > 1.0e-10
        direction[safe] = delta[safe] / distance[safe, None]
        fallback = rest_delta[nonlocal_pair]
        fallback_length = np.maximum(np.linalg.norm(fallback, axis=1), 1.0e-12)
        direction[~safe] = fallback[~safe] / fallback_length[~safe, None]
        penetration = np.maximum(self.self_contact_radius - distance, 0.0)
        pair_shift = 0.25 * penetration[:, None] * direction

        displacement = self._sum_by_vertex(pairs[:, 1], pair_shift)
        displacement -= self._sum_by_vertex(pairs[:, 0], pair_shift)
        self.x += displacement
        # A modest restitution update makes separating tentacles retain outward
        # momentum without destabilizing simultaneous multi-contact regions.
        self.v += 0.20 * displacement / max(dt * self.self_contact_interval, 1.0e-12)

    def _step(self, dt: float) -> np.ndarray:
        old_half_extent = float(self.scene.inner_half_extent(self.time))
        new_time = self.time + dt
        half_extent = float(self.scene.inner_half_extent(new_time))
        wall_speed = (half_extent - old_half_extent) / dt
        force = self._elastic_forces()
        force += self.mass[:, None] * self.gravity[None, :]
        self.v += dt * force * self.inv_mass[:, None]
        self.x += dt * self.v
        contact_impulse = self._apply_wall_contact(half_extent, wall_speed, dt)
        if self.step_count % self.self_contact_interval == 0:
            self._apply_self_contact(dt)
            # Self-contact near a corner may push a vertex back against a wall.
            contact_impulse += self._apply_wall_contact(half_extent, wall_speed, dt)
        gravity_impulse = dt * float(np.sum(self.mass)) * self.gravity
        self.time = new_time
        self.step_count += 1
        return contact_impulse + gravity_impulse

    def _deformation_gradient(self) -> np.ndarray:
        tri = self.x[self.element_faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        normal_len = np.linalg.norm(normal, axis=1)
        safe = normal_len > 1.0e-12
        unit_normal = self.element_rest_normal.copy()
        unit_normal[safe] = normal[safe] / normal_len[safe, None]
        e3 = unit_normal * self.element_thickness[:, None]
        ds = np.stack((e1, e2, e3), axis=2)
        return np.einsum("nij,njk->nik", ds, self.element_inv_dm).astype(np.float32)

    def _make_frame(self, time: float) -> StateFrame:
        frame = StateFrame(float(time), self.x.copy(), self.v.copy(), mass=self.mass.copy())
        frame.attributes["mass_quantity"] = self.mass.copy()
        frame.fields["budget_cumulative_external_impulse"] = self.cumulative_external_impulse[None, :].copy()
        frame.fields["budget_cumulative_inflow_momentum"] = np.zeros((1, 3), dtype=np.float64)
        frame.fields["budget_cumulative_outflow_momentum"] = np.zeros((1, 3), dtype=np.float64)
        frame.fields["budget_domain_weights"] = np.ones((1, self.n), dtype=np.float32)
        # One total external-force term for the single conserved material.
        frame.fields["budget_external_force"] = self.last_external_force[None, None, :].copy()
        frame.fields["cumulative_mass_source"] = np.zeros(1, dtype=np.float64)
        frame.fields["element_deformation_gradient"] = self._deformation_gradient()
        frame.scalars["reference_momentum"] = 0.5
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        interval_start = self.time
        interval_impulse = np.zeros(3, dtype=np.float64)
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            interval_impulse += self._step(dt)
        self.time = target
        elapsed = target - interval_start
        if elapsed > 0.0:
            self.last_external_force = interval_impulse / elapsed
            self.cumulative_external_impulse += interval_impulse
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_surface_elastodynamics_with_bulk_volume_energy",
            "vertex_count": int(self.n),
            "edge_count": int(len(self.edges)),
            "surface_triangle_count": int(len(self.faces)),
            "reported_element_count": int(len(self.element_faces)),
            "substeps": int(self.step_count),
            "simulated_time": float(self.time),
        }
