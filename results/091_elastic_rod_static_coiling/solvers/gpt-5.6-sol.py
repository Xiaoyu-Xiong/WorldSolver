"""Deterministic discrete-elastic-rod solver for the static coiling scene.

The mechanical state is kept on material nodes at the 25 mm resolution of the
supplied topology. Nodes are introduced at the injector as material is fed.
Stretch is removed with mass-weighted inextensibility projections, while the
Euler--Bernoulli bending energy is integrated explicitly. Contact with the
table is unilateral and becomes a permanent sticking constraint on impact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based time integrator for a growing, inextensible elastic rod."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rod = cfg["rod"]
        injector = cfg["injector"]

        asset = Path(__file__).resolve().parents[1] / rod["topology_asset"]
        topology = json.loads(asset.read_text(encoding="utf-8"))
        supplied = np.asarray(topology["vertices"], dtype=float)

        self.ds = float(np.median(np.diff(np.asarray(topology["arc_coordinate"], dtype=float))))
        self.initial_length = float(rod["initial_deployed_length"])
        self.available_length = float(rod["total_available_length"])
        self.feed_speed = float(injector["injection_speed"])
        self.feed_start = float(injector["start_time"])
        self.feed_end = float(injector["end_time"])
        self.outlet = np.asarray(injector["outlet_position"], dtype=float)
        self.radius = float(rod["radius"])
        self.surface_z = float(cfg["substrate"]["surface_z"]) + self.radius
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        area = np.pi * self.radius**2
        second_moment = np.pi * self.radius**4 / 4.0
        self.linear_density = float(rod["density"]) * area
        self.bending_stiffness = float(rod["young_modulus"]) * second_moment

        # Material indices run from the initially deposited end toward the
        # injector. Unused nodes remain collapsed at the outlet.
        self.n_material = int(np.ceil(self.available_length / self.ds)) + 1
        self.x = np.repeat(self.outlet[None, :], self.n_material, axis=0)
        self.v = np.zeros_like(self.x)
        initial_nodes = len(supplied)
        self.x[:initial_nodes] = supplied[::-1]

        # A microscopic deterministic imperfection selects one of the two
        # equivalent winding directions without imposing continuing motion.
        rng = np.random.default_rng(int(seed))
        phase = rng.uniform(0.15, 0.35)
        s0 = np.linspace(0.0, 1.0, initial_nodes)
        self.x[:initial_nodes, 1] += 2.0e-6 * np.sin(np.pi * s0) * np.sin(
            2.0 * np.pi * s0 + phase
        )

        self.stuck = np.zeros(self.n_material, dtype=bool)
        self.stuck[0] = True
        self.stuck_position = self.x.copy()
        self.stuck_position[0, 2] = self.surface_z
        self.x[0] = self.stuck_position[0]

        self.time = 0.0
        self.internal_dt = 0.005
        self.constraint_sweeps = 12
        self.velocity_damping = 7.0
        self.current_mass, self.top_index, self.segment_rest = self._mass_and_activation(0.0)

        self.output_count = len(supplied)
        self.previous_output_position: np.ndarray | None = None
        self.previous_output_time: float | None = None
        self.previous_momentum: np.ndarray | None = None
        self.previous_angular_momentum: np.ndarray | None = None
        self.step_count = 0

    def _deployed_length(self, time: float) -> float:
        feeding_time = max(0.0, min(time, self.feed_end) - self.feed_start)
        return min(self.available_length, self.initial_length + self.feed_speed * feeding_time)

    def _mass_and_activation(self, time: float) -> tuple[np.ndarray, int, np.ndarray]:
        length = self._deployed_length(time)
        full = min(int(np.floor((length + 1.0e-12) / self.ds)), self.n_material - 1)
        remainder = max(0.0, length - full * self.ds)
        rest = np.zeros(self.n_material - 1, dtype=float)
        rest[:full] = self.ds
        if remainder > 1.0e-10 and full < len(rest):
            rest[full] = remainder
            top = full + 1
        else:
            top = full

        segment_mass = self.linear_density * rest
        mass = np.zeros(self.n_material, dtype=float)
        mass[:-1] += 0.5 * segment_mass
        mass[1:] += 0.5 * segment_mass
        return mass, top, rest

    def _project_lengths(self, q: np.ndarray, inverse_mass: np.ndarray, top: int) -> None:
        """Alternating disjoint-pair Gauss--Seidel projections."""
        for _ in range(self.constraint_sweeps):
            for parity in (0, 1):
                left = np.arange(parity, top, 2, dtype=int)
                if left.size == 0:
                    continue
                right = left + 1
                delta = q[right] - q[left]
                distance = np.linalg.norm(delta, axis=1)
                valid = distance > 1.0e-12
                weight_sum = inverse_mass[left] + inverse_mass[right]
                valid &= weight_sum > 0.0
                if not np.any(valid):
                    continue
                li = left[valid]
                ri = right[valid]
                unit = delta[valid] / distance[valid, None]
                error = distance[valid] - self.segment_rest[li]
                share_left = inverse_mass[li] / weight_sum[valid]
                share_right = inverse_mass[ri] / weight_sum[valid]
                q[li] += (share_left * error)[:, None] * unit
                q[ri] -= (share_right * error)[:, None] * unit

            q[top] = self.outlet
            if np.any(self.stuck[: top + 1]):
                fixed = np.flatnonzero(self.stuck[: top + 1])
                q[fixed] = self.stuck_position[fixed]

    def _substep(self, dt: float, new_time: float) -> None:
        old_top = self.top_index
        new_mass, top, rest = self._mass_and_activation(new_time)
        self.current_mass = new_mass
        self.segment_rest = rest

        if top > old_top:
            for index in range(old_top + 1, top + 1):
                self.x[index] = self.outlet
                self.v[index] = 0.0
        self.top_index = top

        active = slice(0, top + 1)
        force = np.zeros_like(self.x)
        force[active] = self.current_mass[active, None] * self.gravity

        # Gradient of B/(2 ds^3) sum |x[i-1]-2x[i]+x[i+1]|^2.
        # Triplets containing the still-growing partial segment are omitted.
        full_segments = int(np.count_nonzero(rest >= self.ds * (1.0 - 1.0e-9)))
        if full_segments >= 2:
            d2 = (
                self.x[: full_segments - 1]
                - 2.0 * self.x[1:full_segments]
                + self.x[2 : full_segments + 1]
            )
            bend = (self.bending_stiffness / self.ds**3) * d2
            force[: full_segments - 1] -= bend
            force[1:full_segments] += 2.0 * bend
            force[2 : full_segments + 1] -= bend

        movable = self.current_mass > 1.0e-14
        movable[self.stuck] = False
        movable[top] = False
        acceleration = np.zeros_like(self.x)
        acceleration[movable] = force[movable] / self.current_mass[movable, None]
        self.v[movable] = (self.v[movable] + dt * acceleration[movable]) * np.exp(
            -self.velocity_damping * dt
        )
        self.v[~movable] = 0.0

        old_x = self.x.copy()
        q = self.x.copy()
        q[movable] += dt * self.v[movable]
        q[top] = self.outlet

        inverse_mass = np.zeros(self.n_material, dtype=float)
        inverse_mass[movable] = 1.0 / self.current_mass[movable]
        self._project_lengths(q, inverse_mass, top)

        contact = (~self.stuck[: top + 1]) & (q[: top + 1, 2] <= self.surface_z)
        contact[top] = False
        if np.any(contact):
            indices = np.flatnonzero(contact)
            q[indices, 2] = self.surface_z
            self.stuck[indices] = True
            self.stuck_position[indices] = q[indices]

        self.x[: top + 1] = q[: top + 1]
        self.v[: top + 1] = (q[: top + 1] - old_x[: top + 1]) / dt
        self.v[self.stuck] = 0.0
        self.v[top] = 0.0
        self.x[top] = self.outlet
        self.x[top + 1 :] = self.outlet
        self.v[top + 1 :] = 0.0
        self.step_count += 1

    def _sample_rod(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the full material interval onto the frozen 21-node topology."""
        length = self._deployed_length(self.time)
        coordinates = np.linspace(length, 0.0, self.output_count)
        # The newest segment has a continuously growing rest length, so its
        # material coordinate is the deployed length (not the next full-node
        # multiple of ds). This also makes the first sample exactly the outlet.
        grid = np.concatenate(([0.0], np.cumsum(self.segment_rest)))
        grid = grid[: self.top_index + 1]
        coordinates = np.clip(coordinates, 0.0, grid[-1])

        position = np.empty((self.output_count, 3), dtype=float)
        material_velocity = np.empty_like(position)
        for axis in range(3):
            position[:, axis] = np.interp(
                coordinates, grid, self.x[: self.top_index + 1, axis]
            )
            material_velocity[:, axis] = np.interp(
                coordinates, grid, self.v[: self.top_index + 1, axis]
            )

        if self.previous_output_position is not None and self.previous_output_time is not None:
            elapsed = self.time - self.previous_output_time
            velocity = (
                (position - self.previous_output_position) / elapsed
                if elapsed > 0.0
                else material_velocity
            )
        else:
            velocity = material_velocity
        return position, velocity, coordinates

    @staticmethod
    def _orientations(position: np.ndarray) -> np.ndarray:
        tangent = np.empty_like(position)
        tangent[0] = position[0] - position[1]
        tangent[-1] = position[-2] - position[-1]
        tangent[1:-1] = position[:-2] - position[2:]
        norm = np.linalg.norm(tangent, axis=1)
        tangent /= np.maximum(norm[:, None], 1.0e-12)

        source = np.array([0.0, 0.0, -1.0])
        cross = np.cross(np.repeat(source[None, :], len(tangent), axis=0), tangent)
        dot = tangent @ source
        quat = np.zeros((len(tangent), 4), dtype=float)
        regular = dot > -0.999999
        quat[regular, :3] = cross[regular]
        quat[regular, 3] = 1.0 + dot[regular]
        quat[~regular, 0] = 1.0
        quat /= np.maximum(np.linalg.norm(quat, axis=1)[:, None], 1.0e-12)
        return quat

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.internal_dt, target - self.time)
            new_time = self.time + dt
            self._substep(dt, new_time)
            self.time = new_time
        self.time = target

        position, velocity, coordinates = self._sample_rod()
        length = self._deployed_length(self.time)
        spacing = length / (self.output_count - 1)
        mass = np.full(self.output_count, self.linear_density * spacing, dtype=float)
        mass[[0, -1]] *= 0.5

        momentum = np.sum(mass[:, None] * velocity, axis=0)
        angular_momentum = np.sum(np.cross(position, mass[:, None] * velocity), axis=0)
        total_mass = float(np.sum(mass))
        center = np.sum(mass[:, None] * position, axis=0) / max(total_mass, 1.0e-15)
        gravity_force = total_mass * self.gravity

        if self.previous_momentum is not None and self.previous_output_time is not None:
            elapsed = self.time - self.previous_output_time
            if elapsed > 0.0:
                required_force = (momentum - self.previous_momentum) / elapsed
                required_torque = (
                    angular_momentum - self.previous_angular_momentum
                ) / elapsed
            else:
                required_force = np.zeros(3)
                required_torque = np.zeros(3)
        else:
            required_force = np.zeros(3)
            required_torque = np.zeros(3)

        reaction = required_force - gravity_force
        stuck_active = np.flatnonzero(self.stuck[: self.top_index + 1])
        support_point = (
            np.mean(self.stuck_position[stuck_active], axis=0)
            if stuck_active.size
            else self.outlet.copy()
        )
        force_terms = np.asarray([[gravity_force, reaction]], dtype=float)
        force_points = np.asarray([[center, support_point]], dtype=float)
        force_torque = np.cross(center, gravity_force) + np.cross(support_point, reaction)
        free_couple = required_torque - force_torque
        couple_terms = np.asarray([[np.zeros(3), free_couple]], dtype=float)

        orientation = self._orientations(position)
        initial_mass = self.linear_density * self.initial_length
        frame = StateFrame(
            time=self.time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros_like(position),
            mass=mass,
            body_id=np.zeros(self.output_count, dtype=np.int64),
            attributes={
                "arc_coordinate": coordinates,
                "is_injector_controlled": (np.arange(self.output_count) == 0).astype(float),
                "is_initial_contact": (
                    np.arange(self.output_count) == self.output_count - 1
                ).astype(float),
                "rod_radius": np.full(self.output_count, self.radius),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_force": force_terms,
                "budget_force_position": force_points,
                "budget_external_couple": couple_terms,
                "constraint_value": position[0:1].copy(),
                "constraint_target": self.outlet[None, :].copy(),
                "cumulative_mass_source": np.asarray(
                    [self.linear_density * (length - self.initial_length)], dtype=float
                ),
            },
            scalars={
                "constraint_scale": max(self.initial_length, 1.0e-6),
                "reference_force": max(
                    self.linear_density
                    * self.available_length
                    * np.linalg.norm(self.gravity),
                    1.0e-9,
                ),
                "reference_torque": max(
                    self.bending_stiffness / self.radius, 1.0e-9
                ),
                "injection_speed": self.feed_speed,
                "natural_curvature": 0.0,
                "substrate_speed": 0.0,
            },
        )
        frame.validate()

        self.previous_output_position = position.copy()
        self.previous_output_time = self.time
        self.previous_momentum = momentum.copy()
        self.previous_angular_momentum = angular_momentum.copy()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "growing discrete elastic rod with sticking PBD contact",
            "internal_steps": int(self.step_count),
            "material_nodes": int(self.n_material),
            "deployed_length": float(self._deployed_length(self.time)),
            "stuck_material_nodes": int(np.count_nonzero(self.stuck)),
        }
