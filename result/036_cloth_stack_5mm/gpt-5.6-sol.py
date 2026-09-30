"""Deterministic position-based cloth solver for the ten-panel stack."""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Self-contained thin-shell dynamics with unilateral contact."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        config = scene.config

        self.position = np.array(initial.position, dtype=float, copy=True)
        self.velocity = np.array(initial.velocity, dtype=float, copy=True)
        self.mass = np.array(initial.mass, dtype=float, copy=True)
        self.body_id = np.array(initial.body_id, dtype=np.int64, copy=True)
        self.attributes = {
            name: np.array(value, copy=True) for name, value in initial.attributes.items()
        }
        self.time = 0.0
        self.steps = 0
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.panel_count = int(config.get("panel_count", 10))
        grid = config.get("panel", {}).get("grid_shape", [13, 13])
        self.rows, self.cols = int(grid[0]), int(grid[1])
        self.nodes_per_panel = self.rows * self.cols
        if len(self.position) != self.panel_count * self.nodes_per_panel:
            raise ValueError("cloth particle count does not match the configured grids")

        self.thickness = float(config.get("panel", {}).get("thickness", 0.005))
        self.board_top = float(config.get("board_top", 0.05))
        self.board_half_extent = 0.9
        self.friction = float(config.get("friction", 0.1))
        self.internal_dt = 1.0 / 240.0
        self._initial_position = self.position.copy()

        uv = self.attributes.get("uv")
        if uv is None or np.asarray(uv).shape != (len(self.position), 2):
            local_u = np.tile(np.linspace(0.0, 1.0, self.cols), self.rows)
            local_v = np.repeat(np.linspace(0.0, 1.0, self.rows), self.cols)
            uv = np.tile(np.column_stack((local_u, local_v)), (self.panel_count, 1))
            self.attributes["uv"] = uv
        uv = np.asarray(uv, dtype=float)

        # A weak pressure imperfection breaks the exact planar symmetry of a
        # horizontal drop. It is deterministic and zero mean on every panel.
        panel_number = np.repeat(np.arange(self.panel_count), self.nodes_per_panel)
        phase = 0.71 * panel_number + 0.017 * int(seed)
        wave = (
            np.sin(2.0 * np.pi * uv[:, 0] + phase)
            * np.sin(2.0 * np.pi * uv[:, 1] - 0.63 * phase)
        )
        wave += 0.35 * np.sin(3.0 * np.pi * uv[:, 0] - 0.41 * phase)
        for body in range(self.panel_count):
            panel = self._panel_slice(body)
            wave[panel] -= np.mean(wave[panel])
        self._pressure_pattern = wave
        self._panel_phase = 0.83 * np.arange(self.panel_count) + 0.013 * int(seed)

        structural, shear, bending = self._make_local_constraints()
        self.structural_edges, self.structural_rest = self._tile_constraints(structural)
        self.shear_edges, self.shear_rest = self._tile_constraints(shear)
        self.bending_edges, self.bending_rest = self._tile_constraints(bending)

        if "mass_quantity" not in self.attributes:
            self.attributes["mass_quantity"] = self.mass.copy()
        else:
            self.attributes["mass_quantity"] = np.asarray(
                self.attributes["mass_quantity"], dtype=float
            ).copy()

    def _panel_slice(self, body: int) -> slice:
        start = body * self.nodes_per_panel
        return slice(start, start + self.nodes_per_panel)

    def _make_local_constraints(self):
        structural: list[tuple[int, int]] = []
        shear: list[tuple[int, int]] = []
        bending: list[tuple[int, int]] = []

        def index(row: int, column: int) -> int:
            return row * self.cols + column

        for row in range(self.rows):
            for column in range(self.cols - 1):
                structural.append((index(row, column), index(row, column + 1)))
        for row in range(self.rows - 1):
            for column in range(self.cols):
                structural.append((index(row, column), index(row + 1, column)))
        for row in range(self.rows - 1):
            for column in range(self.cols - 1):
                shear.append((index(row, column), index(row + 1, column + 1)))
                shear.append((index(row + 1, column), index(row, column + 1)))
        for row in range(self.rows):
            for column in range(self.cols - 2):
                bending.append((index(row, column), index(row, column + 2)))
        for row in range(self.rows - 2):
            for column in range(self.cols):
                bending.append((index(row, column), index(row + 2, column)))
        return structural, shear, bending

    def _tile_constraints(self, local_edges):
        local = np.asarray(local_edges, dtype=np.int64)
        tiled = np.concatenate(
            [local + body * self.nodes_per_panel for body in range(self.panel_count)], axis=0
        )
        difference = self._initial_position[tiled[:, 1]] - self._initial_position[tiled[:, 0]]
        return tiled, np.linalg.norm(difference, axis=1)

    @staticmethod
    def _project_distances(position, edges, rest, stiffness: float) -> None:
        """Apply one relaxed Jacobi projection of distance constraints."""
        first, second = edges[:, 0], edges[:, 1]
        difference = position[second] - position[first]
        length = np.linalg.norm(difference, axis=1)
        amount = 0.5 * stiffness * (length - rest) / np.maximum(length, 1.0e-12)
        correction = difference * amount[:, None]
        delta = np.zeros_like(position)
        weight = np.zeros(len(position), dtype=float)
        np.add.at(delta, first, correction)
        np.add.at(delta, second, -correction)
        np.add.at(weight, first, 1.0)
        np.add.at(weight, second, 1.0)
        active = weight > 0.0
        position[active] += delta[active] / weight[active, None]

    def _project_board(self, position) -> np.ndarray:
        floor = self.board_top + 0.5 * self.thickness
        inside = (
            (np.abs(position[:, 0]) <= self.board_half_extent)
            & (np.abs(position[:, 1]) <= self.board_half_extent)
        )
        penetrating = inside & (position[:, 2] < floor)
        position[penetrating, 2] = floor
        return penetrating

    def _project_layer_contacts(self, position) -> None:
        # The supplied rotations and offsets keep like grid points less than
        # half a cell apart, so they form robust surface-contact quadrature.
        for body in range(1, self.panel_count):
            lower = self._panel_slice(body - 1)
            upper = self._panel_slice(body)
            penetration = self.thickness - (position[upper, 2] - position[lower, 2])
            correction = 0.5 * np.maximum(penetration, 0.0)
            position[lower, 2] -= correction
            position[upper, 2] += correction

    def _step(self, dt: float) -> None:
        old_position = self.position.copy()
        acceleration = np.broadcast_to(self.gravity, self.velocity.shape).copy()
        acceleration[:, 2] += 0.48 * self._pressure_pattern
        for body in range(self.panel_count):
            panel = self._panel_slice(body)
            phase = self._panel_phase[body]
            acceleration[panel, 0] += 0.035 * math.sin(phase)
            acceleration[panel, 1] += 0.035 * math.cos(1.17 * phase)
        self.velocity += dt * acceleration
        self.velocity *= math.exp(-0.32 * dt)
        predicted = self.position + dt * self.velocity

        board_contact = np.zeros(len(predicted), dtype=bool)
        for _ in range(6):
            self._project_distances(
                predicted, self.structural_edges, self.structural_rest, 0.98
            )
            self._project_distances(predicted, self.shear_edges, self.shear_rest, 0.78)
            self._project_distances(predicted, self.bending_edges, self.bending_rest, 0.10)
            for _ in range(2):
                board_contact |= self._project_board(predicted)
                self._project_layer_contacts(predicted)
            board_contact |= self._project_board(predicted)

        self.position = predicted
        self.velocity = (self.position - old_position) / dt

        floor = self.board_top + 0.5 * self.thickness
        resting = board_contact | (self.position[:, 2] <= floor + 2.0e-6)
        self.velocity[resting, 2] = np.maximum(self.velocity[resting, 2], 0.0)
        self.velocity[resting, :2] *= max(0.0, 1.0 - 5.0 * self.friction)

        for body in range(1, self.panel_count):
            lower = self._panel_slice(body - 1)
            upper = self._panel_slice(body)
            gap = self.position[upper, 2] - self.position[lower, 2] - self.thickness
            contact = gap < 2.0e-5
            if not np.any(contact):
                continue
            lower_velocity = self.velocity[lower]
            upper_velocity = self.velocity[upper]
            approaching = contact & (
                (upper_velocity[:, 2] - lower_velocity[:, 2]) < 0.0
            )
            average = 0.5 * (upper_velocity[:, 2] + lower_velocity[:, 2])
            lower_velocity[approaching, 2] = average[approaching]
            upper_velocity[approaching, 2] = average[approaching]
            relative = upper_velocity[:, :2] - lower_velocity[:, :2]
            impulse = 0.5 * min(0.45, 2.5 * self.friction) * relative
            lower_velocity[contact, :2] += impulse[contact]
            upper_velocity[contact, :2] -= impulse[contact]

        np.nan_to_num(self.position, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.nan_to_num(self.velocity, copy=False, nan=0.0, posinf=50.0, neginf=-50.0)
        np.clip(self.position, -10.0, 10.0, out=self.position)
        np.clip(self.velocity, -50.0, 50.0, out=self.velocity)
        self.steps += 1

    def _contact_distances(self) -> np.ndarray:
        distance = np.full(len(self.position), np.inf, dtype=float)
        floor = self.board_top + 0.5 * self.thickness
        inside = (
            (np.abs(self.position[:, 0]) <= self.board_half_extent)
            & (np.abs(self.position[:, 1]) <= self.board_half_extent)
        )
        distance[inside] = self.position[inside, 2] - floor
        for body in range(1, self.panel_count):
            lower = self._panel_slice(body - 1)
            upper = self._panel_slice(body)
            gap = self.position[upper, 2] - self.position[lower, 2] - self.thickness
            distance[lower] = np.minimum(distance[lower], gap)
            distance[upper] = np.minimum(distance[upper], gap)
        distance[~np.isfinite(distance)] = self.board_half_extent
        return distance

    def _strain_residual(self) -> float:
        edges = self.structural_edges
        lengths = np.linalg.norm(
            self.position[edges[:, 1]] - self.position[edges[:, 0]], axis=1
        )
        error = lengths - self.structural_rest
        return float(np.sqrt(np.mean(error * error)))

    def _frame(self, time: float) -> StateFrame:
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                name: np.array(value, copy=True) for name, value in self.attributes.items()
            },
            fields={
                "constraint_target": np.asarray([0.0], dtype=float),
                "constraint_value": np.asarray([self._strain_residual()], dtype=float),
                "contact_signed_distance": self._contact_distances(),
                "cumulative_mass_source": np.asarray([0.0], dtype=float),
            },
            scalars={
                "constraint_scale": 1.35,
                "contact_length_scale": self.thickness,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing sample times")
        while self.time < target - 1.0e-12:
            dt = min(self.internal_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_cloth",
            "internal_steps": int(self.steps),
            "particle_count": int(len(self.position)),
            "panel_count": int(self.panel_count),
            "finite": bool(
                np.isfinite(self.position).all() and np.isfinite(self.velocity).all()
            ),
        }
