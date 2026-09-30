"""Self contained granular dynamics solver for the sand impact scene.

The sand is represented by frictional, slightly compliant grains. Contacts are
integrated with an impulse/PBD (non-smooth DEM) scheme. The model conserves
contact momentum, includes Coulomb friction and inelastic yielding, and permits
grains to separate into a ballistic spray.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic discrete-element sand and rigid-body integrator."""

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.scene = scene
        self.time = 0.0
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self.n_sand = int(np.count_nonzero(self.body_id == 0))
        self.ball = self.n_sand
        self.gravity = np.asarray(scene.config["gravity"], dtype=np.float64)
        self.spacing = float(scene.config["sampling"]["spacing"])
        self.grain_radius = 0.485 * self.spacing
        self.ball_radius = 0.54
        self.inv_mass = 1.0 / self.mass
        self.material_label = self.body_id.copy()
        self.initial_position = self.position.copy()
        self.cumulative_mass_source = np.zeros(len(self.position), dtype=np.float64)
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.contact_distance = np.zeros(len(self.position), dtype=np.float64)

        # Fixed material neighbours recover a mesh-free deformation gradient.
        x0 = self.initial_position[: self.n_sand]
        tree = cKDTree(x0)
        neighbour_lists = tree.query_ball_point(x0, 1.48 * self.spacing)
        self.ref_neighbours: list[np.ndarray] = []
        self.ref_inverse: list[np.ndarray] = []
        for i, values in enumerate(neighbour_lists):
            ids = np.asarray([j for j in values if j != i], dtype=np.int32)
            q = x0[ids] - x0[i]
            moment = q.T @ q
            self.ref_neighbours.append(ids)
            self.ref_inverse.append(np.linalg.pinv(moment, rcond=1.0e-8))

        self.deformation_gradient = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], len(self.position), axis=0
        )
        self.steps = 0
        self.max_speed = 0.0
        self._update_contact_distance()

    @staticmethod
    def _safe_normal(delta: np.ndarray, fallback_index: int = 0) -> tuple[np.ndarray, float]:
        distance = float(np.linalg.norm(delta))
        if distance > 1.0e-12:
            return delta / distance, distance
        axis = np.zeros(3, dtype=np.float64)
        axis[fallback_index % 3] = 1.0
        return axis, 0.0

    def _particle_contacts(self, dt: float) -> None:
        """Resolve frictional grain contacts with momentum-conserving impulses."""
        x = self.position[: self.n_sand]
        v = self.velocity[: self.n_sand]
        diameter = 2.0 * self.grain_radius
        pairs = cKDTree(x).query_pairs(diameter, output_type="ndarray")
        if pairs.size == 0:
            return

        restitution = 0.035
        friction = 0.58
        for i, j in pairs:
            delta = x[j] - x[i]
            normal, distance = self._safe_normal(delta, int(i + j))
            penetration = diameter - distance
            if penetration <= 0.0:
                continue
            wi, wj = self.inv_mass[i], self.inv_mass[j]
            wsum = wi + wj

            # Incomplete overlap recovery represents plastic compaction.
            correction = 0.78 * max(penetration - 0.002 * self.spacing, 0.0)
            x[i] -= normal * correction * wi / wsum
            x[j] += normal * correction * wj / wsum

            relative = v[j] - v[i]
            vn = float(np.dot(relative, normal))
            bias = 0.10 * max(penetration - 0.01 * self.spacing, 0.0) / dt
            normal_impulse = max((-(1.0 + restitution) * vn + bias) / wsum, 0.0)
            impulse = normal_impulse * normal

            tangent = relative - vn * normal
            tangent_speed = float(np.linalg.norm(tangent))
            if tangent_speed > 1.0e-10:
                desired = tangent_speed / wsum
                tangent_impulse = min(desired, friction * normal_impulse)
                impulse -= tangent_impulse * tangent / tangent_speed

            v[i] -= impulse * wi
            v[j] += impulse * wj

    def _ball_contacts(self, dt: float, contact_impulse: np.ndarray) -> None:
        center = self.position[self.ball]
        ball_velocity = self.velocity[self.ball]
        x = self.position[: self.n_sand]
        v = self.velocity[: self.n_sand]
        reach = self.ball_radius + self.grain_radius
        candidates = cKDTree(x).query_ball_point(center, reach)
        friction = 0.42
        restitution = 0.08
        wb = self.inv_mass[self.ball]

        for i in candidates:
            normal, distance = self._safe_normal(x[i] - center, i)
            penetration = reach - distance
            if penetration <= 0.0:
                continue
            wi = self.inv_mass[i]
            wsum = wi + wb
            correction = 0.92 * max(penetration - 0.001 * self.spacing, 0.0)
            x[i] += normal * correction * wi / wsum
            center -= normal * correction * wb / wsum

            relative = v[i] - ball_velocity
            vn = float(np.dot(relative, normal))
            bias = 0.14 * max(penetration - 0.006 * self.spacing, 0.0) / dt
            normal_impulse = max((-(1.0 + restitution) * vn + bias) / wsum, 0.0)
            impulse = normal_impulse * normal

            tangent = relative - vn * normal
            tangent_speed = float(np.linalg.norm(tangent))
            if tangent_speed > 1.0e-10:
                tangent_impulse = min(tangent_speed / wsum, friction * normal_impulse)
                impulse -= tangent_impulse * tangent / tangent_speed

            v[i] += impulse * wi
            ball_velocity -= impulse * wb
            contact_impulse[i] += impulse
            contact_impulse[self.ball] -= impulse

        self.position[self.ball] = center
        self.velocity[self.ball] = ball_velocity

    def _floor_contacts(self, dt: float, contact_impulse: np.ndarray) -> None:
        for i in range(self.n_sand):
            penetration = self.grain_radius - self.position[i, 2]
            if penetration <= 0.0:
                continue
            self.position[i, 2] += penetration
            old_velocity = self.velocity[i].copy()
            if self.velocity[i, 2] < 0.0:
                self.velocity[i, 2] *= -0.025
            horizontal = self.velocity[i, :2]
            speed = float(np.linalg.norm(horizontal))
            if speed > 0.0:
                drop = min(speed, 0.72 * 9.81 * dt)
                self.velocity[i, :2] *= (speed - drop) / speed
                if speed < 0.035 and abs(self.velocity[i, 2]) < 0.04:
                    self.velocity[i, :2] = 0.0
            contact_impulse[i] += self.mass[i] * (self.velocity[i] - old_velocity)

        i = self.ball
        penetration = self.ball_radius - self.position[i, 2]
        if penetration > 0.0:
            self.position[i, 2] += penetration
            old_velocity = self.velocity[i].copy()
            if self.velocity[i, 2] < 0.0:
                self.velocity[i, 2] *= -0.04
            self.velocity[i, :2] *= 0.9
            contact_impulse[i] += self.mass[i] * (self.velocity[i] - old_velocity)

    def _step(self, dt: float) -> None:
        old_velocity = self.velocity.copy()
        self.velocity += self.gravity[None, :] * dt
        self.velocity[: self.n_sand] *= np.exp(-0.055 * dt)
        self.position += self.velocity * dt

        contact_impulse = np.zeros_like(self.position)
        for _ in range(3):
            self._particle_contacts(dt)
            self._ball_contacts(dt, contact_impulse)
            self._floor_contacts(dt, contact_impulse)

        resting = (
            (self.position[: self.n_sand, 2] <= self.grain_radius + 2.0e-4)
            & (np.linalg.norm(self.velocity[: self.n_sand], axis=1) < 0.025)
        )
        self.velocity[: self.n_sand][resting] = 0.0

        # Gravity is the prescribed external load. Contact pairs are deliberately
        # excluded from this exported field by contract (their equal/opposite
        # impulses are nevertheless applied above). The floor is represented by
        # its signed-distance contact field rather than a prescribed body load.
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.steps += 1
        self.max_speed = max(self.max_speed, float(np.max(np.linalg.norm(self.velocity, axis=1))))

    def _update_contact_distance(self) -> None:
        center = self.position[self.ball]
        sand = self.position[: self.n_sand]
        floor_gap = sand[:, 2] - self.grain_radius
        sphere_gap = np.linalg.norm(sand - center[None, :], axis=1) - (
            self.ball_radius + self.grain_radius
        )
        self.contact_distance[: self.n_sand] = np.minimum(floor_gap, sphere_gap)
        self.contact_distance[self.ball] = center[2] - self.ball_radius

    def _update_deformation_gradient(self) -> None:
        current = self.position[: self.n_sand]
        reference = self.initial_position[: self.n_sand]
        for i in range(self.n_sand):
            ids = self.ref_neighbours[i]
            if len(ids) < 3:
                self.deformation_gradient[i] = np.eye(3)
                continue
            q = reference[ids] - reference[i]
            p = current[ids] - current[i]
            raw = (p.T @ q) @ self.ref_inverse[i]
            if not np.all(np.isfinite(raw)):
                raw = np.eye(3)
            try:
                u, singular, vh = np.linalg.svd(raw)
                singular = np.clip(singular, 0.42, 2.35)
                projected = (u * singular[None, :]) @ vh
                if np.linalg.det(projected) < 0.0:
                    u[:, -1] *= -1.0
                    projected = (u * singular[None, :]) @ vh
                self.deformation_gradient[i] = projected
            except np.linalg.LinAlgError:
                self.deformation_gradient[i] = np.eye(3)
        self.deformation_gradient[self.ball] = np.eye(3)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")

        max_dt = 1.0 / 240.0
        while self.time < target - 1.0e-12:
            dt = min(max_dt, target - self.time)
            self._step(dt)
            self.time += dt

        self.time = target
        self._update_contact_distance()
        self._update_deformation_gradient()

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.concatenate(
                (np.full(self.n_sand, 0.45 * self.spacing), np.asarray([self.ball_radius]))
            ),
            "material_label": self.material_label.copy(),
            "deformation_gradient": self.deformation_gradient.copy(),
        }
        fields = {
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "external_force": self.external_force.copy(),
            "contact_signed_distance": self.contact_distance.copy(),
        }
        characteristic = max(float(np.ptp(self.initial_position, axis=0).max()), self.spacing)
        frame = StateFrame(
            target,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": characteristic,
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "frictional_nonsmooth_dem",
            "internal_steps": int(self.steps),
            "maximum_speed": float(self.max_speed),
            "sand_particles": int(self.n_sand),
        }
