"""Deterministic elastic-rod network solver for the branching tree scene.

The tree is discretized as mass-lumped beam nodes. Axial strain and changes
of curvature at every parent/node/child triple define a linearized elastic
energy about the supplied reference configuration. Backward Euler advances
the stiff elastic and Rayleigh-damping forces; prescribed wind drag and
gravity are evaluated at each physical substep.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Implicit time stepper for a connected, fixed-root elastic rod tree."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.x0 = np.asarray(initial.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.n = len(self.x)

        asset = Path(__file__).resolve().parents[1] / self.cfg["rod_network"]["topology_asset"]
        topology = json.loads(asset.read_text(encoding="utf-8"))
        self.edges = np.asarray(topology["edges"], dtype=np.int64)
        self.level = np.asarray(topology["vertex_level"], dtype=np.int64)
        self.branch = np.asarray(topology["branch_id"], dtype=np.int64)
        self.lumped_length = np.asarray(topology["lumped_length"], dtype=float)

        rod_cfg = self.cfg["rod_network"]
        self.radius = np.asarray(rod_cfg["radius_by_level"], dtype=float)[self.level]
        density = np.asarray(rod_cfg["linear_density_by_level"], dtype=float)[self.level]
        self.mass = density * np.pi * self.radius**2 * self.lumped_length

        # Rigid-junction masses remain lumped because incident rod motion resolves rotation.
        junctions = np.asarray(self.cfg["rigid_junctions"]["vertex_indices"], dtype=np.int64)
        junction_mass = np.asarray(self.cfg["rigid_junctions"]["mass_by_level"], dtype=float)
        self.mass[junctions] += junction_mass[self.level[junctions]]
        self.root = int(self.cfg["rigid_junctions"]["fixed_root_vertex"])
        self.is_junction = np.zeros(self.n, dtype=float)
        self.is_junction[junctions] = 1.0

        self.parent = np.full(self.n, -1, dtype=np.int64)
        self.children: list[list[int]] = [[] for _ in range(self.n)]
        adjacency: list[list[int]] = [[] for _ in range(self.n)]
        for a, b in self.edges:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        queue = [self.root]
        for p in queue:
            for child in adjacency[p]:
                if child == self.parent[p]:
                    continue
                self.parent[child] = p
                self.children[p].append(child)
                queue.append(child)

        self.local_rest_tangent = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            if i == self.root:
                j = self.children[i][0]
                d = self.x0[j] - self.x0[i]
            else:
                d = self.x0[i] - self.x0[self.parent[i]]
            self.local_rest_tangent[i] = d / np.linalg.norm(d)

        self.K = self._assemble_stiffness(young_modulus=1.2e7)
        self.M = np.diag(np.repeat(self.mass, 3))
        # Stiffness-proportional damping is pairwise/internal and therefore does
        # not inject or remove net linear momentum from the complete tree domain.
        self.C = 0.003 * self.K
        root_dofs = np.arange(3 * self.root, 3 * self.root + 3)
        self.free = np.setdiff1d(np.arange(3 * self.n), root_dofs)
        self._inverse_cache: dict[float, np.ndarray] = {}

        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        wind_cfg = self.cfg["wind"]
        self.wind_direction = np.asarray(wind_cfg["direction"], dtype=float)
        self.wind_direction /= np.linalg.norm(self.wind_direction)
        self.wind_times = np.asarray([k["time"] for k in wind_cfg["keyframes"]], dtype=float)
        self.wind_speeds = np.asarray([k["speed"] for k in wind_cfg["keyframes"]], dtype=float)
        self.phase_offsets = np.asarray(wind_cfg["branch_phase_offsets"], dtype=float)
        self.gust_frequency = float(wind_cfg["gust_frequency_hz"])
        self.reference_height = max(float(np.max(self.x0[:, 2])), 1.0)

        self.time = 0.0
        self.max_step = 0.005
        self.cumulative_impulse = np.zeros((1, 3), dtype=float)
        self.cumulative_angular_impulse = np.zeros((1, 3), dtype=float)
        self._last_total_force = None
        self._last_total_torque = None
        self._last_external_terms = None
        self._last_force_positions = None
        self._update_external_bookkeeping(0.0, integrate=False)

        total_mass = float(np.sum(self.mass))
        peak_speed = float(np.max(self.wind_speeds))
        self.reference_momentum = max(total_mass * peak_speed, 1.0e-6)
        self.reference_angular_momentum = max(
            self.reference_momentum * self.reference_height, 1.0e-6
        )
        self._step_count = 0

    def _add_block(self, matrix: np.ndarray, i: int, j: int, block: np.ndarray) -> None:
        matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block

    def _assemble_stiffness(self, young_modulus: float) -> np.ndarray:
        """Assemble the Hessian of axial-strain and curvature-change energies."""
        matrix = np.zeros((3 * self.n, 3 * self.n), dtype=float)
        eye = np.eye(3)
        for ia, ib in self.edges:
            i, j = int(ia), int(ib)
            rest = self.x0[j] - self.x0[i]
            length = float(np.linalg.norm(rest))
            tangent = rest / length
            area = np.pi * min(self.radius[i], self.radius[j]) ** 2
            block = (young_modulus * area / length) * np.outer(tangent, tangent)
            self._add_block(matrix, i, i, block)
            self._add_block(matrix, j, j, block)
            self._add_block(matrix, i, j, -block)
            self._add_block(matrix, j, i, -block)

        # Difference of adjacent unit tangents is the discrete curvature measure.
        for node in range(self.n):
            parent = int(self.parent[node])
            if parent < 0:
                continue
            lp = float(np.linalg.norm(self.x0[node] - self.x0[parent]))
            for child in self.children[node]:
                lc = float(np.linalg.norm(self.x0[child] - self.x0[node]))
                radius = min(self.radius[parent], self.radius[node], self.radius[child])
                second_moment = np.pi * radius**4 / 4.0
                bend_weight = young_modulus * second_moment / (0.5 * (lp + lc))
                indices = (parent, node, child)
                coefficients = (1.0 / lp, -(1.0 / lp + 1.0 / lc), 1.0 / lc)
                for a, ca in zip(indices, coefficients):
                    for b, cb in zip(indices, coefficients):
                        self._add_block(matrix, a, b, bend_weight * ca * cb * eye)
        return 0.5 * (matrix + matrix.T)

    def _wind_speed(self, time: float) -> float:
        return float(np.interp(time, self.wind_times, self.wind_speeds))

    def _external_forces(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        wind_velocity = self._wind_speed(time) * self.wind_direction
        wind_force = np.zeros((self.n, 3), dtype=float)

        # Quadratic cross-flow drag on each mass-lumped cylindrical rod segment.
        for i in range(self.n):
            if i == self.root:
                j = self.children[i][0]
                segment = self.x[j] - self.x[i]
            else:
                segment = self.x[i] - self.x[self.parent[i]]
            length = float(np.linalg.norm(segment))
            tangent = segment / max(length, 1.0e-9)
            relative = wind_velocity - self.v[i]
            cross_flow = relative - tangent * float(np.dot(relative, tangent))
            cross_speed = float(np.linalg.norm(cross_flow))
            phase = self.phase_offsets[self.branch[i] % len(self.phase_offsets)]
            gust = 1.0 + 0.22 * np.sin(2.0 * np.pi * (self.gust_frequency * time - phase))
            height_weight = 0.35 + 0.65 * np.clip(
                self.x0[i, 2] / self.reference_height, 0.0, 1.0
            )
            wind_force[i] = (
                0.022
                * (2.0 * self.radius[i])
                * self.lumped_length[i]
                * height_weight
                * gust
                * cross_speed
                * cross_flow
            )
        return gravity_force, wind_force

    def _constraint_reaction(
        self, gravity_force: np.ndarray, wind_force: np.ndarray
    ) -> np.ndarray:
        displacement = (self.x - self.x0).reshape(-1)
        velocity = self.v.reshape(-1)
        internal = (-self.K @ displacement - self.C @ velocity).reshape(self.n, 3)
        return -(gravity_force[self.root] + wind_force[self.root] + internal[self.root])

    def _external_terms(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        gravity_force, wind_force = self._external_forces(time)
        reaction = self._constraint_reaction(gravity_force, wind_force)
        # One term per application point preserves the angular-momentum budget.
        terms = np.zeros((1, 2 * self.n + 1, 3), dtype=float)
        positions = np.zeros_like(terms)
        terms[0, : self.n] = gravity_force
        terms[0, self.n : 2 * self.n] = wind_force
        terms[0, -1] = reaction
        positions[0, : self.n] = self.x
        positions[0, self.n : 2 * self.n] = self.x
        positions[0, -1] = self.x[self.root]
        return terms, positions

    def _update_external_bookkeeping(
        self, time: float, integrate: bool, dt: float = 0.0
    ) -> None:
        terms, positions = self._external_terms(time)
        total_force = np.sum(terms, axis=1)
        total_torque = np.sum(np.cross(positions, terms), axis=1)
        if integrate and self._last_total_force is not None:
            self.cumulative_impulse += 0.5 * dt * (self._last_total_force + total_force)
            self.cumulative_angular_impulse += 0.5 * dt * (
                self._last_total_torque + total_torque
            )
        self._last_total_force = total_force
        self._last_total_torque = total_torque
        self._last_external_terms = terms
        self._last_force_positions = positions

    def _step(self, dt: float) -> None:
        new_time = self.time + dt
        gravity_force, wind_force = self._external_forces(new_time)
        external = (gravity_force + wind_force).reshape(-1)
        displacement = (self.x - self.x0).reshape(-1)
        old_velocity = self.v.reshape(-1)

        key = round(float(dt), 12)
        inverse = self._inverse_cache.get(key)
        if inverse is None:
            lhs = self.M + dt * self.C + dt * dt * self.K
            inverse = np.linalg.inv(lhs[np.ix_(self.free, self.free)])
            self._inverse_cache[key] = inverse
        rhs = self.M @ old_velocity - dt * (self.K @ displacement) + dt * external
        new_velocity = np.zeros(3 * self.n, dtype=float)
        new_velocity[self.free] = inverse @ rhs[self.free]
        self.v = new_velocity.reshape(self.n, 3)
        self.x += dt * self.v
        self.x[self.root] = self.x0[self.root]
        self.v[self.root] = 0.0
        self.time = new_time
        self._update_external_bookkeeping(new_time, integrate=True, dt=dt)
        self._step_count += 1

    @staticmethod
    def _rotation_quaternion(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
        dot = float(np.clip(np.dot(rest, current), -1.0, 1.0))
        cross = np.cross(rest, current)
        quat = np.array([cross[0], cross[1], cross[2], 1.0 + dot], dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm < 1.0e-10:
            axis = np.cross(rest, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1.0e-8:
                axis = np.cross(rest, np.array([0.0, 1.0, 0.0]))
            axis /= np.linalg.norm(axis)
            return np.array([axis[0], axis[1], axis[2], 0.0])
        return quat / norm

    def _orientations(self) -> tuple[np.ndarray, np.ndarray]:
        orientation = np.zeros((self.n, 4), dtype=float)
        angular_velocity = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            if i == self.root:
                j = self.children[i][0]
                segment = self.x[j] - self.x[i]
                relative_velocity = self.v[j] - self.v[i]
            else:
                p = int(self.parent[i])
                segment = self.x[i] - self.x[p]
                relative_velocity = self.v[i] - self.v[p]
            length = max(float(np.linalg.norm(segment)), 1.0e-9)
            tangent = segment / length
            orientation[i] = self._rotation_quaternion(self.local_rest_tangent[i], tangent)
            angular_velocity[i] = np.cross(tangent, relative_velocity / length)
        angular_velocity[self.root] = 0.0
        return orientation, angular_velocity

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(self.max_step, target - self.time))
        self.time = target

        orientation, angular_velocity = self._orientations()
        attributes = {
            "is_rod_vertex": np.ones(self.n, dtype=float),
            "is_rigid_junction": self.is_junction.copy(),
            "is_fixed_root": (np.arange(self.n) == self.root).astype(float),
            "branch_id": self.branch.astype(float),
            "branch_level": self.level.astype(float),
            "rod_radius": self.radius.copy(),
            "inertia_tensor": np.zeros((self.n, 3, 3), dtype=float),
            "mass_quantity": self.mass.copy(),
        }
        zero_budget = np.zeros((1, 3), dtype=float)
        fields = {
            "budget_cumulative_external_angular_impulse": self.cumulative_angular_impulse.copy(),
            "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
            "budget_cumulative_inflow_angular_momentum": zero_budget.copy(),
            "budget_cumulative_inflow_momentum": zero_budget.copy(),
            "budget_cumulative_outflow_angular_momentum": zero_budget.copy(),
            "budget_cumulative_outflow_momentum": zero_budget.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_couple": np.zeros_like(self._last_external_terms),
            "budget_external_force": self._last_external_terms.copy(),
            "budget_force_position": self._last_force_positions.copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars={
                "wind_speed": self._wind_speed(target),
                "gravity_magnitude": float(np.linalg.norm(self.gravity)),
                "reference_momentum": self.reference_momentum,
                "reference_angular_momentum": self.reference_angular_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mass-lumped elastic rods with implicit Euler",
            "substeps": self._step_count,
            "simulated_time": self.time,
            "max_displacement": float(np.max(np.linalg.norm(self.x - self.x0, axis=1))),
        }
