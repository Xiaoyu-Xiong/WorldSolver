"""Planar rigid-body model of a frictionally jammed articulated mechanism.

The links are full rigid bodies in the x-z plane. Adjacent links are joined by
pin constraints represented by stiff, damped forces, while the supplied guide
geometry supplies unilateral penalty contact with regularized Coulomb
friction. A smooth inward load on the end links compresses the zig-zag chain;
the resulting guide normal forces generate enough friction to lock it.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic substepped rigid-link/contact solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model contains no stochastic terms.
        config = scene.config
        initial = scene.initial_state()

        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.mass)
        self.theta = np.zeros(self.n, dtype=float)
        self.omega = np.zeros(self.n, dtype=float)
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.mu = float(config["contact"]["friction"])
        self.length_scale = float(config["contact"]["length_scale"])
        self.time = 0.0
        self.steps = 0

        # x-z silhouette of the supplied extruded L-shaped mesh.
        self.local_outline = np.asarray(
            [
                [-0.24, -0.16],
                [0.24, -0.16],
                [0.24, 0.30],
                [0.08, 0.30],
                [0.08, 0.00],
                [-0.24, 0.00],
            ],
            dtype=float,
        )

        # Pin locations are initial midpoints between neighboring centers.
        # Material-frame storage permits genuine articulation by rotation.
        self.anchor_a = np.empty((self.n - 1, 2), dtype=float)
        self.anchor_b = np.empty((self.n - 1, 2), dtype=float)
        for j in range(self.n - 1):
            delta = self.position[j + 1, (0, 2)] - self.position[j, (0, 2)]
            self.anchor_a[j] = 0.5 * delta
            self.anchor_b[j] = -0.5 * delta

        # Conservative inertia and stiffness values for the supplied mesh.
        self.inertia = self.mass * 0.036
        self.joint_stiffness = 1900.0
        self.joint_damping = 34.0
        self.contact_stiffness = 3200.0
        self.contact_damping = 42.0
        self.linear_drag = 0.22
        self.angular_drag = 0.035
        self.max_step = 0.0005

        # Inner faces of the frozen lower and upper guide boxes.
        self.guide_min_x = -2.25
        self.guide_max_x = 2.25
        self.lower_surface = 0.20
        self.upper_surface = 0.94
        self._last_external = np.zeros((self.n, 3), dtype=float)
        self._last_interaction = np.zeros(
            (4 * self.n + 2 * (self.n - 1), 3), dtype=float
        )
        self._last_gaps = np.zeros(2 * self.n, dtype=float)

        # Populate every requested field at t=0 without advancing the state.
        self._evaluate_forces(self.time)

    @staticmethod
    def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
        """Rotate material x-z vectors about the positive world y axis."""
        c = math.cos(angle)
        s = math.sin(angle)
        result = np.empty_like(points)
        result[..., 0] = c * points[..., 0] + s * points[..., 1]
        result[..., 1] = -s * points[..., 0] + c * points[..., 1]
        return result

    @staticmethod
    def _point_velocity(
        linear: np.ndarray, omega: float, arm: np.ndarray
    ) -> np.ndarray:
        return linear + omega * np.asarray([arm[1], -arm[0]])

    @staticmethod
    def _torque(arm: np.ndarray, force: np.ndarray) -> float:
        # y component of arm cross force in the x-z plane.
        return arm[1] * force[0] - arm[0] * force[1]

    def _driver_force(self, time: float) -> float:
        # Smoothly engage the load, then maintain it to demonstrate locking.
        ramp = min(max(time / 1.25, 0.0), 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        return 13.0 * ramp

    def _evaluate_forces(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Assemble physical force and torque at the current configuration."""
        planar_velocity = self.velocity[:, (0, 2)]
        force = self.mass[:, None] * self.gravity[(0, 2)]
        force -= self.linear_drag * planar_velocity
        torque = -self.angular_drag * self.omega

        drive = self._driver_force(time)
        force[0, 0] += drive
        force[-1, 0] -= drive

        external = np.zeros((self.n, 3), dtype=float)
        external[:, 0] = force[:, 0]
        external[:, 2] = force[:, 1]

        interactions: list[np.ndarray] = []

        # Damped pin-joint forces. Each interaction is exported as the equal
        # and opposite pair applied to its two bodies.
        for j in range(self.n - 1):
            ra = self._rotate(self.anchor_a[j], self.theta[j])
            rb = self._rotate(self.anchor_b[j], self.theta[j + 1])
            pa = self.position[j, (0, 2)] + ra
            pb = self.position[j + 1, (0, 2)] + rb
            va = self._point_velocity(planar_velocity[j], self.omega[j], ra)
            vb = self._point_velocity(
                planar_velocity[j + 1], self.omega[j + 1], rb
            )
            joint_force = (
                self.joint_stiffness * (pb - pa)
                + self.joint_damping * (vb - va)
            )
            force[j] += joint_force
            force[j + 1] -= joint_force
            torque[j] += self._torque(ra, joint_force)
            torque[j + 1] += self._torque(rb, -joint_force)
            pair = np.asarray(
                [joint_force[0], 0.0, joint_force[1]], dtype=float
            )
            interactions.extend((pair, -pair))

        gaps = np.empty(2 * self.n, dtype=float)
        contact_pairs: list[np.ndarray] = []

        for i in range(self.n):
            rotated = self._rotate(self.local_outline, self.theta[i])
            world = rotated + self.position[i, (0, 2)]

            low_index = int(np.argmin(world[:, 1]))
            high_index = int(np.argmax(world[:, 1]))
            low_gap = world[low_index, 1] - self.lower_surface
            high_gap = self.upper_surface - world[high_index, 1]
            gaps[2 * i] = low_gap
            gaps[2 * i + 1] = high_gap

            for is_lower, index, gap in (
                (True, low_index, low_gap),
                (False, high_index, high_gap),
            ):
                arm = rotated[index]
                point = world[index]
                contact_force = np.zeros(2, dtype=float)

                # The finite guide extent is honored.
                if (
                    self.guide_min_x <= point[0] <= self.guide_max_x
                    and gap < 0.0
                ):
                    point_velocity = self._point_velocity(
                        planar_velocity[i], self.omega[i], arm
                    )
                    normal_sign = 1.0 if is_lower else -1.0
                    normal_speed = normal_sign * point_velocity[1]
                    normal_magnitude = max(
                        0.0,
                        -self.contact_stiffness * gap
                        - self.contact_damping * normal_speed,
                    )
                    contact_force[1] = normal_sign * normal_magnitude
                    # Smooth static/kinetic Coulomb approximation. Its steep
                    # origin arrests creeping once the chain wedges.
                    contact_force[0] = (
                        -self.mu
                        * normal_magnitude
                        * math.tanh(point_velocity[0] / 0.012)
                    )
                    force[i] += contact_force
                    torque[i] += self._torque(arm, contact_force)

                pair = np.asarray(
                    [contact_force[0], 0.0, contact_force[1]], dtype=float
                )
                contact_pairs.extend((pair, -pair))

        interactions.extend(contact_pairs)
        self._last_external = external
        self._last_interaction = np.asarray(interactions, dtype=float)
        self._last_gaps = gaps
        return force, torque

    def _step(self, dt: float) -> None:
        force, torque = self._evaluate_forces(self.time)

        # Symplectic Euler is stable for dissipative contact with this small
        # fixed maximum substep.
        self.velocity[:, 0] += dt * force[:, 0] / self.mass
        self.velocity[:, 2] += dt * force[:, 1] / self.mass
        self.omega += dt * torque / self.inertia
        self.position[:, 0] += dt * self.velocity[:, 0]
        self.position[:, 2] += dt * self.velocity[:, 2]
        self.theta += dt * self.omega

        # The scene and forces are planar; explicitly eliminate roundoff drift.
        self.position[:, 1] = 0.0
        self.velocity[:, 1] = 0.0
        self.time += dt
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")

        while self.time + 1.0e-12 < target:
            self._step(min(self.max_step, target - self.time))
        self.time = target

        # Report forces at the exact sample state/time.
        self._evaluate_forces(target)

        half = 0.5 * self.theta
        orientation = np.zeros((self.n, 4), dtype=float)
        orientation[:, 1] = np.sin(half)
        orientation[:, 3] = np.cos(half)
        angular_velocity = np.zeros((self.n, 3), dtype=float)
        angular_velocity[:, 1] = self.omega

        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": self._last_gaps.copy(),
                "external_force": self._last_external.copy(),
                "interaction_force": self._last_interaction.copy(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "planar_rigid_body_penalty_contact",
            "substeps": int(self.steps),
            "final_time": float(self.time),
            "maximum_speed": float(
                np.max(np.linalg.norm(self.velocity, axis=1))
            ),
            "maximum_joint_error": float(self._maximum_joint_error()),
            "minimum_contact_gap": float(np.min(self._last_gaps)),
        }

    def _maximum_joint_error(self) -> float:
        error = 0.0
        for j in range(self.n - 1):
            ra = self._rotate(self.anchor_a[j], self.theta[j])
            rb = self._rotate(self.anchor_b[j], self.theta[j + 1])
            pa = self.position[j, (0, 2)] + ra
            pb = self.position[j + 1, (0, 2)] + rb
            error = max(error, float(np.linalg.norm(pb - pa)))
        return error
