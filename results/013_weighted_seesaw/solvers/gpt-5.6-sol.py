"""Deterministic rigid-body solver for the weighted seesaw scene.

The mechanism is planar in x-z. The board is constrained by an ideal revolute
joint through its centre, and the boxes have translational and rotational
degrees of freedom. Contact uses semi-implicit Euler and sequential impulses.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    _GRAVITY = 9.81
    _STEP = 0.001
    _BOARD_CENTER = np.array([0.0, 0.78], dtype=float)  # x, z
    _BOARD_HALF = np.array([1.8, 0.07], dtype=float)
    _BLOCK_HALF = np.array([[0.28, 0.28], [0.22, 0.22]], dtype=float)

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.time = 0.0
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()

        # Planar block state is ordered heavy, light. Positive rotation about
        # +y lowers the right-hand end of the board.
        self.block_pos = np.asarray(initial.position[1:], dtype=float)[:, [0, 2]].copy()
        self.block_vel = np.asarray(initial.velocity[1:], dtype=float)[:, [0, 2]].copy()
        self.block_angle = np.zeros(2, dtype=float)
        self.block_omega = np.zeros(2, dtype=float)
        self.board_angle = 0.0
        self.board_omega = 0.0

        length, thickness = 2.0 * self._BOARD_HALF
        self.board_inertia = self.mass[0] * (length**2 + thickness**2) / 12.0
        widths = 2.0 * self._BLOCK_HALF[:, 0]
        heights = 2.0 * self._BLOCK_HALF[:, 1]
        self.block_inertia = self.mass[1:] * (widths**2 + heights**2) / 12.0

        # Geometric angle at which a lower board corner reaches the floor.
        half_length, half_thickness = self._BOARD_HALF
        radius = math.hypot(half_length, half_thickness)
        phase = math.atan2(half_thickness, half_length)
        self.board_limit = math.asin(self._BOARD_CENTER[1] / radius) - phase
        self.steps = 0
        self.max_penetration = 0.0

    @staticmethod
    def _axes(angle: float) -> tuple[np.ndarray, np.ndarray]:
        """World directions of local +x and +z in the x-z plane."""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c, -s]), np.array([s, c])

    @staticmethod
    def _point_velocity(linear: np.ndarray, omega: float, radius: np.ndarray) -> np.ndarray:
        # omega * (+y) cross (radius.x, 0, radius.z)
        return linear + omega * np.array([radius[1], -radius[0]])

    @staticmethod
    def _moment_arm(radius: np.ndarray, direction: np.ndarray) -> float:
        # y component of radius cross direction in the x-z plane.
        return radius[1] * direction[0] - radius[0] * direction[1]

    def _corners(self, index: int) -> list[np.ndarray]:
        tangent, normal = self._axes(self.block_angle[index])
        hx, hz = self._BLOCK_HALF[index]
        centre = self.block_pos[index]
        return [
            centre + sx * hx * tangent + sz * hz * normal
            for sx in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    def _apply_board_contact(self, index: int, point: np.ndarray, gap: float, dt: float) -> None:
        tangent, normal = self._axes(self.board_angle)
        block_radius = point - self.block_pos[index]
        board_radius = point - self._BOARD_CENTER
        point_v = self._point_velocity(self.block_vel[index], self.block_omega[index], block_radius)
        board_v = self._point_velocity(np.zeros(2), self.board_omega, board_radius)
        relative = point_v - board_v

        cb = self._moment_arm(block_radius, normal)
        cs = self._moment_arm(board_radius, normal)
        inverse_mass = 1.0 / self.mass[index + 1]
        normal_mass = inverse_mass + cb**2 / self.block_inertia[index] + cs**2 / self.board_inertia
        bias = 0.18 * max(-gap - 2.0e-5, 0.0) / dt
        normal_speed = float(np.dot(relative, normal))
        bounce = -0.04 * normal_speed if normal_speed < -0.5 else 0.0
        impulse_n = max(0.0, (max(bias, bounce) - normal_speed) / normal_mass)
        if impulse_n <= 0.0:
            return

        impulse = impulse_n * normal
        self.block_vel[index] += impulse * inverse_mass
        self.block_omega[index] += impulse_n * cb / self.block_inertia[index]
        self.board_omega -= impulse_n * cs / self.board_inertia

        point_v = self._point_velocity(self.block_vel[index], self.block_omega[index], block_radius)
        board_v = self._point_velocity(np.zeros(2), self.board_omega, board_radius)
        tangent_speed = float(np.dot(point_v - board_v, tangent))
        tb = self._moment_arm(block_radius, tangent)
        ts = self._moment_arm(board_radius, tangent)
        tangent_mass = inverse_mass + tb**2 / self.block_inertia[index] + ts**2 / self.board_inertia
        impulse_t = -tangent_speed / tangent_mass
        friction_limit = 0.28 * impulse_n
        impulse_t = float(np.clip(impulse_t, -friction_limit, friction_limit))
        friction = impulse_t * tangent
        self.block_vel[index] += friction * inverse_mass
        self.block_omega[index] += impulse_t * tb / self.block_inertia[index]
        self.board_omega -= impulse_t * ts / self.board_inertia

    def _apply_ground_contact(self, index: int, point: np.ndarray, gap: float, dt: float) -> None:
        normal = np.array([0.0, 1.0])
        tangent = np.array([1.0, 0.0])
        radius = point - self.block_pos[index]
        inverse_mass = 1.0 / self.mass[index + 1]
        point_v = self._point_velocity(self.block_vel[index], self.block_omega[index], radius)
        cn = self._moment_arm(radius, normal)
        normal_mass = inverse_mass + cn**2 / self.block_inertia[index]
        bias = 0.20 * max(-gap - 2.0e-5, 0.0) / dt
        normal_speed = float(np.dot(point_v, normal))
        bounce = -0.08 * normal_speed if normal_speed < -0.7 else 0.0
        impulse_n = max(0.0, (max(bias, bounce) - normal_speed) / normal_mass)
        if impulse_n <= 0.0:
            return
        self.block_vel[index] += impulse_n * normal * inverse_mass
        self.block_omega[index] += impulse_n * cn / self.block_inertia[index]

        point_v = self._point_velocity(self.block_vel[index], self.block_omega[index], radius)
        ct = self._moment_arm(radius, tangent)
        tangent_mass = inverse_mass + ct**2 / self.block_inertia[index]
        impulse_t = -float(np.dot(point_v, tangent)) / tangent_mass
        impulse_t = float(np.clip(impulse_t, -0.48 * impulse_n, 0.48 * impulse_n))
        self.block_vel[index] += impulse_t * tangent * inverse_mass
        self.block_omega[index] += impulse_t * ct / self.block_inertia[index]

    def _solve_contacts(self, dt: float) -> None:
        for _ in range(10):
            board_tangent, board_normal = self._axes(self.board_angle)
            for index in range(2):
                # Do not let a fallen box contact the board from underneath.
                centre_height = float(np.dot(self.block_pos[index] - self._BOARD_CENTER, board_normal))
                if centre_height > 0.0:
                    for point in self._corners(index):
                        local_x = float(np.dot(point - self._BOARD_CENTER, board_tangent))
                        gap = float(np.dot(point - self._BOARD_CENTER, board_normal) - self._BOARD_HALF[1])
                        if abs(local_x) <= self._BOARD_HALF[0] + 1.0e-6 and gap <= 8.0e-4:
                            self._apply_board_contact(index, point, gap, dt)

                for point in self._corners(index):
                    if point[1] <= 8.0e-4:
                        self._apply_ground_contact(index, point, float(point[1]), dt)

    def _project_positions(self) -> None:
        # Remove the small overlap tolerated by the velocity solver.
        for _ in range(2):
            tangent, normal = self._axes(self.board_angle)
            for index in range(2):
                centre_height = float(np.dot(self.block_pos[index] - self._BOARD_CENTER, normal))
                if centre_height > 0.0:
                    eligible = []
                    for point in self._corners(index):
                        local_x = float(np.dot(point - self._BOARD_CENTER, tangent))
                        if abs(local_x) <= self._BOARD_HALF[0]:
                            eligible.append(float(np.dot(point - self._BOARD_CENTER, normal) - self._BOARD_HALF[1]))
                    if eligible and min(eligible) < 0.0:
                        depth = -min(eligible)
                        self.max_penetration = max(self.max_penetration, depth)
                        self.block_pos[index] += depth * normal

                ground_gap = min(point[1] for point in self._corners(index))
                if ground_gap < 0.0:
                    self.max_penetration = max(self.max_penetration, -ground_gap)
                    self.block_pos[index, 1] -= ground_gap

    def _substep(self, dt: float) -> None:
        # Symplectic Euler: external forces update momenta before positions.
        self.block_vel[:, 1] -= self._GRAVITY * dt
        self.board_omega *= math.exp(-0.06 * dt)
        self.block_omega *= math.exp(-0.025 * dt)

        self.block_pos += self.block_vel * dt
        self.block_angle += self.block_omega * dt
        self.board_angle += self.board_omega * dt

        # Board-floor contact is an angular unilateral constraint.
        if self.board_angle > self.board_limit:
            self.board_angle = self.board_limit
            if self.board_omega > 0.0:
                self.board_omega *= -0.03
        elif self.board_angle < -self.board_limit:
            self.board_angle = -self.board_limit
            if self.board_omega < 0.0:
                self.board_omega *= -0.03

        self._solve_contacts(dt)
        self._project_positions()
        self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("sample times must be finite and nondecreasing")
        while self.time + 1.0e-12 < target:
            dt = min(self._STEP, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame()

    def _box_board_signed_distance(self, index: int) -> float:
        """SAT signed separation of a block and the finite oriented board."""
        board_t, board_n = self._axes(self.board_angle)
        block_t, block_n = self._axes(self.block_angle[index])
        delta = self.block_pos[index] - self._BOARD_CENTER
        separations = []
        for axis in (board_t, board_n, block_t, block_n):
            board_radius = (
                self._BOARD_HALF[0] * abs(float(np.dot(board_t, axis)))
                + self._BOARD_HALF[1] * abs(float(np.dot(board_n, axis)))
            )
            block_radius = (
                self._BLOCK_HALF[index, 0] * abs(float(np.dot(block_t, axis)))
                + self._BLOCK_HALF[index, 1] * abs(float(np.dot(block_n, axis)))
            )
            separations.append(abs(float(np.dot(delta, axis))) - board_radius - block_radius)
        return max(separations)

    def _frame(self) -> StateFrame:
        position = np.zeros((3, 3), dtype=float)
        velocity = np.zeros((3, 3), dtype=float)
        position[0] = [self._BOARD_CENTER[0], 0.0, self._BOARD_CENTER[1]]
        position[1:, 0] = self.block_pos[:, 0]
        position[1:, 2] = self.block_pos[:, 1]
        velocity[1:, 0] = self.block_vel[:, 0]
        velocity[1:, 2] = self.block_vel[:, 1]

        angles = np.array([self.board_angle, self.block_angle[0], self.block_angle[1]])
        orientation = np.zeros((3, 4), dtype=float)
        orientation[:, 1] = np.sin(0.5 * angles)
        orientation[:, 3] = np.cos(0.5 * angles)
        angular_velocity = np.zeros((3, 3), dtype=float)
        angular_velocity[:, 1] = [self.board_omega, self.block_omega[0], self.block_omega[1]]

        ground_gaps = [min(point[1] for point in self._corners(i)) for i in range(2)]
        board_t, board_n = self._axes(self.board_angle)
        board_corners = [
            self._BOARD_CENTER + sx * self._BOARD_HALF[0] * board_t + sz * self._BOARD_HALF[1] * board_n
            for sx in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        contact_distance = np.array(
            [
                self._box_board_signed_distance(0),
                self._box_board_signed_distance(1),
                ground_gaps[0],
                ground_gaps[1],
                min(point[1] for point in board_corners),
            ],
            dtype=float,
        )

        external_force = np.zeros((3, 3), dtype=float)
        external_force[:, 2] = -self.mass * self._GRAVITY
        hinge_position = np.array([[self._BOARD_CENTER[0], 0.0, self._BOARD_CENTER[1]]], dtype=float)
        return StateFrame(
            time=self.time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "constraint_target": hinge_position.copy(),
                "constraint_value": position[0:1].copy(),
                "contact_signed_distance": contact_distance,
                "external_force": external_force,
            },
            scalars={"constraint_scale": 1.0, "contact_length_scale": 0.14},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "planar_sequential_impulse_rigid_body",
            "substeps": self.steps,
            "maximum_corrected_penetration": float(self.max_penetration),
        }
