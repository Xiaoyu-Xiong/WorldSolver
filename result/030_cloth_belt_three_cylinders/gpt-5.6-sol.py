"""Physics solver for the three-cylinder closed cloth belt benchmark.

This implementation deliberately has no dependency on a third-party simulator. The
belt is a particle cloth advanced with semi-implicit Euler and iterative position
constraints (a standard PBD discretisation). Cylinder contact is imposed at every
substep and a regularised Coulomb impulse couples contacting material to the
prescribed cylinder surface velocity.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic particle-cloth solver with kinematic cylinder contact."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model is intentionally deterministic.
        self.cfg = scene.config
        initial = scene.initial_state()

        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int64).copy()
        self.orientation = np.asarray(initial.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(initial.angular_velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()

        self.cloth_count = int(np.count_nonzero(self.body_id == 0))
        self.rings = 120
        self.width_nodes = int(self.cfg["belt"]["width_nodes"])
        if self.cloth_count != self.rings * self.width_nodes:
            raise ValueError("cloth topology does not match the declared 120 x width grid")

        self.time = 0.0
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)
        self.radius = float(self.cfg["cylinders"]["radius"])
        self.thickness = float(self.cfg["belt"]["thickness"])
        self.clearance = float(self.cfg["belt"]["centerline_clearance"])
        self.contact_radius = self.radius + self.clearance
        self.mu = float(self.cfg["friction"]["coefficient"])
        self.centers_initial = np.asarray(
            [[x, 0.0, z] for x, z in self.cfg["cylinders"]["initial_centers_xz"]],
            dtype=float,
        )
        self.centers_final = np.asarray(
            [[x, 0.0, z] for x, z in self.cfg["cylinders"]["final_centers_xz"]],
            dtype=float,
        )
        self.base_cylinder_orientation = self.orientation[-3:].copy()

        # A 240 Hz physical step is small compared with the 60 Hz recording step,
        # while constraint projection removes the severe explicit cloth stiffness.
        self.max_step = 1.0 / 240.0
        self.constraint_iterations = 7
        self.linear_damping = 0.35
        self._last_contact_count = 0
        self._steps = 0

        cloth = self.position[: self.cloth_count]
        self._constraint_sets = self._make_constraints(cloth)
        self._reference_basis_inverse = self._make_reference_basis_inverse(cloth)

        mass_per_node = float(self.cfg["belt"]["total_mass"]) / self.cloth_count
        self.mass[:] = 0.0
        self.mass[: self.cloth_count] = mass_per_node
        self.inv_mass = np.full(self.cloth_count, 1.0 / mass_per_node, dtype=float)

        self._set_kinematics(0.0)

    def _make_constraints(self, x: np.ndarray):
        """Build fixed material stretch, shear, and bending constraint arrays."""
        r, w = self.rings, self.width_nodes
        grid = np.arange(r * w, dtype=np.int64).reshape(r, w)

        ring_a = grid.reshape(-1)
        ring_b = np.roll(grid, -1, axis=0).reshape(-1)
        width_a = grid[:, :-1].reshape(-1)
        width_b = grid[:, 1:].reshape(-1)
        shear_a = np.concatenate((grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1)))
        shear_b = np.concatenate(
            (
                np.roll(grid, -1, axis=0)[:, 1:].reshape(-1),
                np.roll(grid, -1, axis=0)[:, :-1].reshape(-1),
            )
        )
        bend_ring_a = grid.reshape(-1)
        bend_ring_b = np.roll(grid, -2, axis=0).reshape(-1)
        bend_width_a = grid[:, :-2].reshape(-1)
        bend_width_b = grid[:, 2:].reshape(-1)

        prestrain = float(self.cfg["belt"].get("prestrain", 0.0))
        material_scale = 1.0 / (1.0 + max(prestrain, 0.0))

        def constraint(a, b, stiffness, shorten=True):
            rest = np.linalg.norm(x[b] - x[a], axis=1)
            if shorten:
                rest = rest * material_scale
            return a, b, rest, float(stiffness)

        return (
            constraint(ring_a, ring_b, 0.92),
            constraint(width_a, width_b, 0.96, shorten=False),
            constraint(shear_a, shear_b, 0.32),
            constraint(bend_ring_a, bend_ring_b, 0.16),
            constraint(bend_width_a, bend_width_b, 0.22, shorten=False),
        )

    def _make_reference_basis_inverse(self, x: np.ndarray) -> np.ndarray:
        grid = x.reshape(self.rings, self.width_nodes, 3)
        along = np.roll(grid, -1, axis=0) - np.roll(grid, 1, axis=0)
        across = np.empty_like(grid)
        across[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
        across[:, 0] = grid[:, 1] - grid[:, 0]
        across[:, -1] = grid[:, -1] - grid[:, -2]
        normal = np.cross(along, across)
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-12)
        basis = np.stack((along, across, normal), axis=-1).reshape(-1, 3, 3)
        return np.linalg.inv(basis)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(1.0, max(0.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _cylinder_motion(self, time: float):
        lift0, lift1 = map(float, self.cfg["timeline"]["lift"])
        if lift1 > lift0:
            lift_fraction = self._smoothstep((time - lift0) / (lift1 - lift0))
        else:
            lift_fraction = 1.0
        centers = self.centers_initial + lift_fraction * (
            self.centers_final - self.centers_initial
        )

        forward0, forward1 = map(float, self.cfg["timeline"]["forward"])
        reverse0, reverse1 = map(float, self.cfg["timeline"]["reverse"])
        speed = float(self.cfg["cylinders"]["outer_angular_speed"])
        forward_elapsed = min(max(time - forward0, 0.0), forward1 - forward0)
        reverse_elapsed = min(max(time - reverse0, 0.0), reverse1 - reverse0)
        phase = speed * (forward_elapsed - reverse_elapsed)

        if forward0 <= time < forward1:
            drive_speed = speed
        elif reverse0 <= time < reverse1:
            drive_speed = -speed
        else:
            drive_speed = 0.0
        signs = np.asarray(self.cfg["cylinders"]["outer_rotation_signs"], dtype=float)
        omega = np.asarray([signs[0] * drive_speed, signs[1] * drive_speed, 0.0])
        angle = np.asarray([signs[0] * phase, signs[1] * phase, 0.0])
        return centers, omega, angle

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.asarray(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )

    def _set_kinematics(self, time: float) -> None:
        centers, omega, angle = self._cylinder_motion(time)
        self.position[-3:] = centers
        self.angular_velocity[-3:] = 0.0
        self.angular_velocity[-3:, 1] = omega
        self.velocity[-3:] = 0.0

        lift0, lift1 = map(float, self.cfg["timeline"]["lift"])
        if lift0 < time < lift1:
            u = (time - lift0) / (lift1 - lift0)
            derivative = 6.0 * u * (1.0 - u) / (lift1 - lift0)
            self.velocity[-1] = derivative * (
                self.centers_final[-1] - self.centers_initial[-1]
            )

        for k in range(3):
            half = 0.5 * angle[k]
            spin = np.asarray([0.0, math.sin(half), 0.0, math.cos(half)])
            q = self._quat_multiply(spin, self.base_cylinder_orientation[k])
            self.orientation[-3 + k] = q / max(np.linalg.norm(q), 1.0e-12)

    def _project_distance_constraints(self, x: np.ndarray) -> None:
        # Jacobi accumulation makes the result independent of constraint order.
        for a, b, rest, stiffness in self._constraint_sets:
            delta = x[b] - x[a]
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length[:, None], 1.0e-12)
            wa = self.inv_mass[a]
            wb = self.inv_mass[b]
            magnitude = stiffness * (length - rest) / np.maximum(wa + wb, 1.0e-12)
            correction = magnitude[:, None] * direction

            accumulated = np.zeros_like(x)
            weights = np.zeros(self.cloth_count, dtype=float)
            np.add.at(accumulated, a, wa[:, None] * correction)
            np.add.at(accumulated, b, -wb[:, None] * correction)
            np.add.at(weights, a, 1.0)
            np.add.at(weights, b, 1.0)
            x += accumulated / np.maximum(weights[:, None], 1.0)

    def _project_cylinders(
        self, x: np.ndarray, centers: np.ndarray, normal_motion: np.ndarray
    ) -> np.ndarray:
        """Project penetrations and return accumulated outward displacement."""
        correction_total = np.zeros(self.cloth_count, dtype=float)
        for center in centers:
            radial = x[:, (0, 2)] - center[(0, 2)]
            distance = np.linalg.norm(radial, axis=1)
            penetration = self.contact_radius - distance
            active = penetration > 0.0
            if not np.any(active):
                continue
            normal2 = radial[active] / np.maximum(distance[active, None], 1.0e-12)
            push = penetration[active, None] * normal2
            x[active, 0] += push[:, 0]
            x[active, 2] += push[:, 1]
            correction_total[active] += penetration[active]
        normal_motion += correction_total
        return correction_total

    def _apply_contact_friction(
        self,
        x: np.ndarray,
        old_x: np.ndarray,
        centers: np.ndarray,
        center_velocity: np.ndarray,
        omega: np.ndarray,
        normal_motion: np.ndarray,
        dt: float,
    ) -> int:
        """Apply a capped, position-level Coulomb impulse at near contacts."""
        count = 0
        near_band = 0.004
        for k, center in enumerate(centers):
            radial2 = x[:, (0, 2)] - center[(0, 2)]
            distance = np.linalg.norm(radial2, axis=1)
            gap = distance - self.contact_radius
            active = gap < near_band
            if not np.any(active):
                continue
            count += int(np.count_nonzero(active))
            ids = np.flatnonzero(active)
            normal = np.zeros((len(ids), 3), dtype=float)
            normal[:, 0] = radial2[ids, 0] / np.maximum(distance[ids], 1.0e-12)
            normal[:, 2] = radial2[ids, 1] / np.maximum(distance[ids], 1.0e-12)

            relative = x[ids] - old_x[ids]
            rel_center = x[ids] - center
            surface_velocity = np.empty_like(rel_center)
            surface_velocity[:, 0] = omega[k] * rel_center[:, 2]
            surface_velocity[:, 1] = 0.0
            surface_velocity[:, 2] = -omega[k] * rel_center[:, 0]
            surface_velocity += center_velocity[k]
            desired = surface_velocity * dt - relative
            desired -= np.sum(desired * normal, axis=1)[:, None] * normal

            # The normal PBD impulse is correction/dt. A tiny proximity term
            # regularises static high-tension contact when penetration is zero.
            normal_budget = normal_motion[ids] + np.maximum(near_band - gap[ids], 0.0)
            max_tangent = self.mu * normal_budget
            desired_length = np.linalg.norm(desired, axis=1)
            scale = np.minimum(1.0, max_tangent / np.maximum(desired_length, 1.0e-12))
            x[ids] += desired * scale[:, None]
        return count

    def _substep(self, dt: float) -> None:
        old_x = self.position[: self.cloth_count].copy()
        old_centers, _, _ = self._cylinder_motion(self.time)
        new_time = self.time + dt
        centers, omega, _ = self._cylinder_motion(new_time)
        center_velocity = (centers - old_centers) / dt

        cloth_velocity = self.velocity[: self.cloth_count]
        cloth_velocity += dt * self.gravity
        cloth_velocity *= math.exp(-self.linear_damping * dt)
        x = old_x + dt * cloth_velocity

        normal_motion = np.zeros(self.cloth_count, dtype=float)
        for _ in range(self.constraint_iterations):
            self._project_distance_constraints(x)
            self._project_cylinders(x, centers, normal_motion)

        self._last_contact_count = self._apply_contact_friction(
            x, old_x, centers, center_velocity, omega, normal_motion, dt
        )
        # Friction can move a point slightly inward on a curved surface.
        self._project_cylinders(x, centers, normal_motion)

        self.position[: self.cloth_count] = x
        self.velocity[: self.cloth_count] = (x - old_x) / dt
        # A conservative speed cap is only a numerical guard for degenerate input.
        speed = np.linalg.norm(self.velocity[: self.cloth_count], axis=1)
        excessive = speed > 20.0
        if np.any(excessive):
            self.velocity[: self.cloth_count][excessive] *= (
                20.0 / speed[excessive]
            )[:, None]

        self.time = new_time
        self._set_kinematics(self.time)
        self._steps += 1

    def _deformation_gradient(self) -> np.ndarray:
        x = self.position[: self.cloth_count].reshape(self.rings, self.width_nodes, 3)
        along = np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)
        across = np.empty_like(x)
        across[:, 1:-1] = x[:, 2:] - x[:, :-2]
        across[:, 0] = x[:, 1] - x[:, 0]
        across[:, -1] = x[:, -1] - x[:, -2]
        normal = np.cross(along, across)
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-12)
        current_basis = np.stack((along, across, normal), axis=-1).reshape(-1, 3, 3)
        cloth_f = np.einsum("nij,njk->nik", current_basis, self._reference_basis_inverse)
        result = np.tile(np.eye(3), (len(self.position), 1, 1))
        result[: self.cloth_count] = cloth_f
        return result

    def _contact_distances(self) -> np.ndarray:
        centers = self.position[-3:]
        cloth = self.position[: self.cloth_count]
        gaps = []
        physical_surface = self.radius + 0.5 * self.thickness
        for center in centers:
            radial = cloth[:, (0, 2)] - center[(0, 2)]
            gaps.append(np.linalg.norm(radial, axis=1) - physical_surface)
        return np.min(np.stack(gaps, axis=1), axis=1)

    def _frame(self, requested_time: float) -> StateFrame:
        frame = StateFrame(
            float(requested_time),
            self.position.copy(),
            self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_distances(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time + 1.0e-12 < target:
            self._substep(min(self.max_step, target - self.time))
        # Eliminate harmless floating accumulation from the externally visible time.
        self.time = target
        self._set_kinematics(target)
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_euler_pbd_cloth_with_coulomb_contact",
            "substeps": self._steps,
            "last_contact_samples": self._last_contact_count,
            "cloth_vertices": self.cloth_count,
        }
