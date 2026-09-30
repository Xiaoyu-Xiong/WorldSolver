"""Deterministic soft-body solver for the squeeze-out scene.

The five bodies are represented by affine elastic ellipsoids. Their centres
obey Newton's equations, while logarithmic principal strains provide a small,
robust deformable-solid model. Contact pressure compresses the body in the
contact-normal direction and volumetric elasticity produces transverse bulge.
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Affine soft solids advanced with a fixed-step symplectic scheme."""

    _RADII = np.array([
        [0.45, 0.35, 0.33], [0.33, 0.30, 0.55], [0.45, 0.39, 0.46],
        [0.34, 0.32, 0.56], [0.42, 0.33, 0.30],
    ], dtype=float)
    _PAIR_COUNT = 10

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        initial = scene.initial_state()
        self._initial_position = np.asarray(initial.position, dtype=float).copy()
        self._initial_velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()

        config = scene.config
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        bc = config.get("boundary_conditions", {})
        self.chamber_x = float(bc.get("chamber_inner_half_width", 0.72))
        self.chamber_y = float(bc.get("chamber_inner_half_depth", 0.52))
        self.outlet_x = float(bc.get("outlet_half_width", 0.34))
        self.outlet_y = float(bc.get("outlet_half_depth", 0.13))
        motion = bc.get("plate_motion", {})
        self.plate_start_time = float(motion.get("start_time", 0.5))
        self.plate_end_time = float(motion.get("end_time", 5.8))
        self.plate_start_z = float(motion.get("start_translation", [0, 0, 4.85])[2])
        self.plate_end_z = float(motion.get("end_translation", [0, 0, 0.58])[2])
        self.reference_momentum = float(np.sum(self.mass) * math.sqrt(9.81))
        self.domain_weights = np.vstack((np.eye(5, dtype=float), np.ones((1, 5))))
        self._reset()

    def _reset(self) -> None:
        self.time = 0.0
        self.position = self._initial_position.copy()
        self.velocity = self._initial_velocity.copy()
        self.log_stretch = np.zeros((5, 3), dtype=float)
        self.stretch_velocity = np.zeros((5, 3), dtype=float)
        self.cumulative_external_impulse = np.zeros((6, 3), dtype=float)
        self.step_count = 0

    def _plate_state(self, time: float) -> tuple[float, float]:
        if time <= self.plate_start_time:
            return self.plate_start_z, 0.0
        if time >= self.plate_end_time:
            return self.plate_end_z, 0.0
        duration = self.plate_end_time - self.plate_start_time
        velocity = (self.plate_end_z - self.plate_start_z) / duration
        return self.plate_start_z + velocity * (time - self.plate_start_time), velocity

    def _tube_limits(self, z: float) -> tuple[float, float, float, float]:
        funnel_top, outlet_top = 1.47, 0.05
        if z >= funnel_top:
            return self.chamber_x, self.chamber_y, 0.0, 0.0
        if z <= outlet_top:
            return self.outlet_x, self.outlet_y, 0.0, 0.0
        span = funnel_top - outlet_top
        alpha = (z - outlet_top) / span
        sx = (self.chamber_x - self.outlet_x) / span
        sy = (self.chamber_y - self.outlet_y) / span
        return self.outlet_x + alpha * (self.chamber_x - self.outlet_x), \
            self.outlet_y + alpha * (self.chamber_y - self.outlet_y), sx, sy

    @staticmethod
    def _contact_magnitude(penetration: float, normal_speed: float, stiffness: float) -> float:
        return min(5000.0, max(0.0, stiffness * penetration - 24.0 * normal_speed))

    def _forces(self, time: float):
        radii = self._RADII * np.exp(self.log_stretch)
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        static_force = np.zeros((5, 3), dtype=float)
        pusher_force = np.zeros((5, 3), dtype=float)
        total = gravity_force.copy()
        compression = np.zeros((5, 3), dtype=float)
        pair_forces = np.zeros((self._PAIR_COUNT, 3), dtype=float)

        pair_index = 0
        for i in range(5):
            for j in range(i + 1, 5):
                delta = self.position[i] - self.position[j]
                distance = float(np.linalg.norm(delta))
                normal = delta / distance if distance > 1.0e-10 else np.array([1.0, 0.0, 0.0])
                ri = float(np.sqrt(np.sum((radii[i] * normal) ** 2)))
                rj = float(np.sqrt(np.sum((radii[j] * normal) ** 2)))
                penetration = ri + rj - distance
                if penetration > 0.0:
                    relative_speed = float(np.dot(self.velocity[i] - self.velocity[j], normal))
                    magnitude = self._contact_magnitude(penetration, relative_speed, 1050.0)
                    force = magnitude * normal
                    total[i] += force
                    total[j] -= force
                    pair_forces[pair_index] = force
                    directional_load = magnitude * normal * normal
                    compression[i] += directional_load
                    compression[j] += directional_load
                pair_index += 1

        def add_static_contact(i: int, penetration: float, inward: np.ndarray,
                               stiffness: float) -> None:
            if penetration <= 0.0:
                return
            normal = inward / max(float(np.linalg.norm(inward)), 1.0e-12)
            normal_speed = float(np.dot(self.velocity[i], normal))
            magnitude = self._contact_magnitude(penetration, normal_speed, stiffness)
            force = magnitude * normal
            total[i] += force
            static_force[i] += force
            compression[i] += magnitude * normal * normal

        for i in range(5):
            x, y, z = self.position[i]
            rx, ry, _ = radii[i]
            if -0.18 <= z <= 5.10:
                hx, hy, sx, sy = self._tube_limits(float(z))
                add_static_contact(i, x + rx - hx, np.array([-1.0, 0.0, sx]), 2200.0)
                add_static_contact(i, -x + rx - hx, np.array([1.0, 0.0, sx]), 2200.0)
                add_static_contact(i, y + ry - hy, np.array([0.0, -1.0, sy]), 2400.0)
                add_static_contact(i, -y + ry - hy, np.array([0.0, 1.0, sy]), 2400.0)
            if z < 0.25:
                add_static_contact(i, x + rx - 1.25, np.array([-1.0, 0.0, 0.0]), 2600.0)
                add_static_contact(i, -x + rx - 1.25, np.array([1.0, 0.0, 0.0]), 2600.0)
                add_static_contact(i, y + ry - 0.90, np.array([0.0, -1.0, 0.0]), 2600.0)
                floor_penetration = -1.99 - (z - radii[i, 2])
                add_static_contact(i, floor_penetration, np.array([0.0, 0.0, 1.0]), 3200.0)

        plate_z, plate_velocity = self._plate_state(time)
        plate_bottom = plate_z - 0.08
        for i in range(5):
            if abs(self.position[i, 0]) <= 0.76 and abs(self.position[i, 1]) <= 0.56:
                penetration = self.position[i, 2] + radii[i, 2] - plate_bottom
                if penetration > 0.0:
                    normal = np.array([0.0, 0.0, -1.0])
                    plate_v = np.array([0.0, 0.0, plate_velocity])
                    relative_speed = float(np.dot(self.velocity[i] - plate_v, normal))
                    magnitude = self._contact_magnitude(penetration, relative_speed, 2800.0)
                    force = magnitude * normal
                    total[i] += force
                    pusher_force[i] += force
                    compression[i, 2] += magnitude

        strain_sum = np.sum(self.log_stretch, axis=1, keepdims=True)
        elastic_acceleration = (-28.0 * self.log_stretch - 105.0 * strain_sum
                                - 9.0 * self.stretch_velocity)
        contact_acceleration = -0.16 * compression / (
            self.mass[:, None] * np.maximum(self._RADII, 0.20))
        stretch_acceleration = elastic_acceleration + contact_acceleration
        external_terms = np.stack((gravity_force, static_force, pusher_force), axis=1)
        return total, stretch_acceleration, external_terms, pair_forces

    def _step(self, dt: float) -> None:
        total_force, stretch_accel, external_terms, _ = self._forces(self.time)
        self.velocity += dt * total_force / self.mass[:, None]
        self.position += dt * self.velocity
        self.stretch_velocity += dt * stretch_accel
        self.log_stretch += dt * self.stretch_velocity

        low_hit = self.log_stretch < -1.25
        high_hit = self.log_stretch > 1.05
        self.log_stretch = np.clip(self.log_stretch, -1.25, 1.05)
        self.stretch_velocity[low_hit & (self.stretch_velocity < 0.0)] = 0.0
        self.stretch_velocity[high_hit & (self.stretch_velocity > 0.0)] = 0.0

        individual_external = np.sum(external_terms, axis=1)
        self.cumulative_external_impulse[:5] += dt * individual_external
        self.cumulative_external_impulse[5] += dt * np.sum(individual_external, axis=0)
        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < -1.0e-12:
            raise ValueError("sample time must be finite and nonnegative")
        if target < self.time - 1.0e-10:
            self._reset()
        while self.time < target - 1.0e-12:
            self._step(min(0.001, target - self.time))
        self.time = target

        _, _, individual_terms, pair_forces = self._forces(self.time)
        budget_terms = np.zeros((6, 3, 3), dtype=float)
        budget_terms[:5] = individual_terms
        budget_terms[5] = np.sum(individual_terms, axis=0)
        deformation_gradient = np.zeros((5, 3, 3), dtype=float)
        diagonal = np.exp(self.log_stretch)
        deformation_gradient[:, 0, 0] = diagonal[:, 0]
        deformation_gradient[:, 1, 1] = diagonal[:, 1]
        deformation_gradient[:, 2, 2] = diagonal[:, 2]

        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.arange(5, dtype=np.int64),
            attributes={
                "reference_position": self._initial_position.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((6, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((6, 3), dtype=float),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": budget_terms,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": deformation_gradient,
                "interaction_force": pair_forces,
            },
            scalars={
                "gravity_z": float(self.gravity[2]),
                "reference_momentum": self.reference_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        stretches = np.exp(self.log_stretch)
        return {
            "method": "affine_elastic_ellipsoids_symplectic_euler",
            "internal_steps": int(self.step_count),
            "simulated_time": float(self.time),
            "minimum_principal_stretch": float(np.min(stretches)),
            "maximum_principal_stretch": float(np.max(stretches)),
        }
