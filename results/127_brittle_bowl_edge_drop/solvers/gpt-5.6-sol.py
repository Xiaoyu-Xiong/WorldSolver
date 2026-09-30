"""Deterministic brittle-shell dynamics for the bowl drop scene.

The bowl is discretized at the supplied mesh vertices. Triangle edges are
cohesive axial bonds; their elastic strain is evaluated every substep and a
bond is irreversibly removed when its heterogeneous brittle strength is
exceeded. Symplectic Euler advances the particles, with penalty contact
against the floor and short-range contact between separated pieces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Breakable mass-spring shell with frictional contact."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.reference = self.x.copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.inv_mass = 1.0 / self.mass
        self.n = len(self.x)
        self.gravity = np.asarray(scene.config["gravity"], dtype=np.float64)
        self.spacing = float(scene.config["sampling"]["spacing"])
        self.radius = 0.16 * self.spacing
        self.time = 0.0
        self.dt_max = 1.0 / 900.0

        topology_path = Path(__file__).resolve().parent.parent / "assets" / "bowl_topology.json"
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        faces = np.asarray(topology["faces"], dtype=np.int64)
        all_edges = np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
        )
        self.edges = np.unique(np.sort(all_edges, axis=1), axis=0)
        self.e0 = self.reference[self.edges[:, 1]] - self.reference[self.edges[:, 0]]
        self.rest = np.linalg.norm(self.e0, axis=1)
        self.active = np.ones(len(self.edges), dtype=bool)
        self.broken_pairs: set[tuple[int, int]] = set()

        # A Weibull-like deterministic strength spread supplies flaws without
        # prescribing where or when a crack travels. Bonds near the initially
        # low rim are weaker, so impact stress nucleates at the contact edge.
        rng = np.random.default_rng(int(seed))
        midpoint = 0.5 * (
            self.reference[self.edges[:, 0]] + self.reference[self.edges[:, 1]]
        )
        impact_x = float(np.max(self.reference[:, 0]))
        distance_from_impact = np.linalg.norm(
            midpoint[:, (0, 1)] - np.array([impact_x, 0.0]), axis=1
        )
        spatial_toughness = 0.85 + 1.45 * np.clip(distance_from_impact / 1.25, 0.0, 1.0)
        flaw_factor = 0.72 + 0.48 * rng.weibull(2.6, len(self.edges))
        # Very short edges join the two sides of the shell and should not be
        # lost solely to their small denominator in the strain measure.
        thickness_factor = np.where(self.rest < 0.035, 2.4, 1.0)
        self.break_strain = np.clip(
            0.020 * spatial_toughness * flaw_factor * thickness_factor,
            0.014,
            0.105,
        )

        mean_mass = float(np.mean(self.mass))
        self.spring_k = 1.55e5
        self.spring_damping = 0.11 * 2.0 * np.sqrt(
            self.spring_k * 0.5 * mean_mass
        )
        self.floor_k = mean_mass * 520.0**2
        self.floor_damping = 0.72 * 2.0 * np.sqrt(self.floor_k * mean_mass)
        self.contact_k = mean_mass * 310.0**2
        self.contact_damping = 0.38 * 2.0 * np.sqrt(
            self.contact_k * 0.5 * mean_mass
        )
        self.friction = 0.48

        # Pairs close in the rest mesh are material neighbors, not collision
        # pairs. Excluding them prevents the two shell layers from repelling
        # one another before fracture.
        initial_tree = cKDTree(self.reference)
        self.material_pairs = {
            (int(i), int(j))
            for i, j in initial_tree.query_pairs(2.15 * self.radius, output_type="set")
        }
        self.broken_count = 0
        self.contact_count = 0
        self.first_impact_time: float | None = None
        self.max_penetration = 0.0

    def _spring_forces(self) -> np.ndarray:
        force = np.zeros((self.n, 3), dtype=np.float64)
        if not np.any(self.active):
            return force

        active_indices = np.flatnonzero(self.active)
        edge = self.edges[active_indices]
        delta = self.x[edge[:, 1]] - self.x[edge[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        safe_length = np.maximum(length, 1.0e-12)
        direction = delta / safe_length[:, None]
        strain = (length - self.rest[active_indices]) / self.rest[active_indices]

        failed = np.abs(strain) > self.break_strain[active_indices]
        if np.any(failed):
            failed_indices = active_indices[failed]
            self.active[failed_indices] = False
            self.broken_pairs.update(
                (int(i), int(j)) for i, j in self.edges[failed_indices]
            )
            self.broken_count += int(len(failed_indices))
            keep = ~failed
            edge = edge[keep]
            direction = direction[keep]
            length = length[keep]
            active_indices = active_indices[keep]

        if len(edge) == 0:
            return force
        relative_velocity = self.v[edge[:, 1]] - self.v[edge[:, 0]]
        normal_speed = np.einsum("ij,ij->i", relative_velocity, direction)
        magnitude = (
            self.spring_k * (length - self.rest[active_indices])
            + self.spring_damping * normal_speed
        )
        edge_force = magnitude[:, None] * direction
        np.add.at(force, edge[:, 0], edge_force)
        np.add.at(force, edge[:, 1], -edge_force)
        return force

    def _floor_forces(self, force: np.ndarray) -> None:
        # The frozen box has its top at z=0 and broad enough lateral extents
        # for the complete observed impact and settling sequence.
        inside = (np.abs(self.x[:, 0]) <= 3.2) & (np.abs(self.x[:, 1]) <= 1.4)
        penetration = self.radius - self.x[:, 2]
        contact = inside & (penetration > 0.0)
        if not np.any(contact):
            return
        indices = np.flatnonzero(contact)
        normal_force = (
            self.floor_k * penetration[indices]
            - self.floor_damping * np.minimum(self.v[indices, 2], 0.0)
        )
        normal_force = np.maximum(normal_force, 0.0)
        force[indices, 2] += normal_force

        tangential_velocity = self.v[indices, :2]
        tangential_speed = np.linalg.norm(tangential_velocity, axis=1)
        moving = tangential_speed > 1.0e-10
        friction_force = np.zeros_like(tangential_velocity)
        friction_force[moving] = (
            -tangential_velocity[moving] / tangential_speed[moving, None]
            * (self.friction * normal_force[moving])[:, None]
        )
        # Regularize Coulomb friction so one explicit step cannot reverse the
        # tangential velocity.
        stop_force = self.mass[indices] * tangential_speed / self._step_dt
        friction_norm = np.linalg.norm(friction_force, axis=1)
        scale = np.minimum(1.0, stop_force / np.maximum(friction_norm, 1.0e-12))
        force[indices, :2] += friction_force * scale[:, None]

        self.contact_count += int(len(indices))
        self.max_penetration = max(self.max_penetration, float(np.max(penetration[indices])))
        if self.first_impact_time is None:
            self.first_impact_time = self.time

    def _fragment_contact_forces(self, force: np.ndarray) -> None:
        if self.broken_count == 0:
            return
        diameter = 2.0 * self.radius
        pairs = cKDTree(self.x).query_pairs(diameter, output_type="ndarray")
        if len(pairs) == 0:
            return
        keep = np.fromiter(
            (
                (int(i), int(j)) not in self.material_pairs
                or (int(i), int(j)) in self.broken_pairs
                for i, j in pairs
            ),
            dtype=bool,
            count=len(pairs),
        )
        pairs = pairs[keep]
        if len(pairs) == 0:
            return

        delta = self.x[pairs[:, 1]] - self.x[pairs[:, 0]]
        distance = np.linalg.norm(delta, axis=1)
        valid = distance > 1.0e-10
        pairs = pairs[valid]
        delta = delta[valid]
        distance = distance[valid]
        if len(pairs) == 0:
            return
        normal = delta / distance[:, None]
        relative_velocity = self.v[pairs[:, 1]] - self.v[pairs[:, 0]]
        separating_speed = np.einsum("ij,ij->i", relative_velocity, normal)
        magnitude = self.contact_k * (diameter - distance)
        magnitude -= self.contact_damping * np.minimum(separating_speed, 0.0)
        pair_force = np.maximum(magnitude, 0.0)[:, None] * normal
        np.add.at(force, pairs[:, 0], -pair_force)
        np.add.at(force, pairs[:, 1], pair_force)

    def _substep(self, dt: float) -> None:
        self._step_dt = dt
        force = self.mass[:, None] * self.gravity[None, :]
        force += self._spring_forces()
        self._floor_forces(force)
        self._fragment_contact_forces(force)

        self.v += dt * force * self.inv_mass[:, None]
        # This finite guard only protects trajectory validity in an extreme
        # collision; ordinary scene velocities remain far below the limit.
        speed = np.linalg.norm(self.v, axis=1)
        excessive = speed > 25.0
        if np.any(excessive):
            self.v[excessive] *= (25.0 / speed[excessive])[:, None]
        self.x += dt * self.v
        self.time += dt

    def _deformation_gradient(self) -> np.ndarray:
        """Least-squares material gradient from each surviving bond star."""
        covariance = np.zeros((self.n, 3, 3), dtype=np.float64)
        reference_covariance = np.zeros_like(covariance)
        active_edges = self.edges[self.active]
        active_rest = self.e0[self.active]
        current = self.x[active_edges[:, 1]] - self.x[active_edges[:, 0]]
        a = current[:, :, None] * active_rest[:, None, :]
        b = active_rest[:, :, None] * active_rest[:, None, :]
        np.add.at(covariance, active_edges[:, 0], a)
        np.add.at(covariance, active_edges[:, 1], a)
        np.add.at(reference_covariance, active_edges[:, 0], b)
        np.add.at(reference_covariance, active_edges[:, 1], b)

        gradient = np.repeat(np.eye(3)[None, :, :], self.n, axis=0)
        rank = np.linalg.matrix_rank(reference_covariance, tol=1.0e-10)
        for i in np.flatnonzero(rank >= 2):
            gradient[i] = covariance[i] @ np.linalg.pinv(
                reference_covariance[i], rcond=1.0e-8
            )
            # A surface neighborhood may constrain only two directions. Add
            # its normal mapping instead of leaving a zero singular axis.
            if rank[i] == 2:
                u, _, vh = np.linalg.svd(gradient[i])
                gradient[i] += np.outer(u[:, 2], vh[2])
        return gradient

    def _frame(self, requested_time: float) -> StateFrame:
        external_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(requested_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.n, self.radius),
                "material_label": np.zeros(self.n, dtype=np.int32),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self.x[:, 2].copy() - self.radius,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": max(
                    float(np.ptp(self.reference, axis=0).max()), self.spacing
                ),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.dt_max, target - self.time)
            self._substep(dt)
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "breakable_edge_springs_symplectic_euler",
            "substep": float(self.dt_max),
            "broken_bonds": int(self.broken_count),
            "remaining_bonds": int(np.count_nonzero(self.active)),
            "floor_contact_samples": int(self.contact_count),
            "first_impact_time": self.first_impact_time,
            "maximum_floor_penetration": float(self.max_penetration),
        }
