"""Particle physics solver for the snowball striking a sticky wall."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        initial = scene.initial_state()
        spec = scene.config["snowball"]
        contact = scene.config["contact"]
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int64).copy()
        self.mass_quantity = np.asarray(initial.attributes["mass_quantity"], dtype=float).copy()
        self.reference_volume = np.asarray(initial.attributes["reference_volume"], dtype=float).copy()
        self.initial_component_id = np.asarray(initial.attributes.get("initial_component_id", np.zeros(len(self.mass))), dtype=np.int64).copy()
        self.gravity = np.asarray(scene.config["gravity"], dtype=float)
        self.center0 = np.asarray(spec["center"], dtype=float)
        self.v0 = np.asarray(spec["initial_velocity"], dtype=float)
        self.nominal_density = float(spec["nominal_density"])
        self.wall_x = float(contact["wall_inner_x"])
        self.floor_z = float(contact["floor_z"])
        self.contact_length = float(contact["contact_length_scale"])
        self.particle_radius = 0.5 * self.contact_length
        self.wall_y_limit, self.wall_z_max = 1.2, 2.7
        self.rest_offset = self.position - self.center0
        self.attached = np.zeros(len(self.position), dtype=bool)
        self.time = 0.0
        self.deformed = False
        self.impact_time = None

    def _step(self, dt: float) -> None:
        n = len(self.position)
        free = ~self.attached
        acceleration = np.broadcast_to(self.gravity, (n, 3)).copy()
        acceleration[free] -= 0.035 * self.velocity[free]
        if not self.deformed:
            target_center = self.center0 + self.v0 * self.time + 0.5 * self.gravity * self.time**2
            target = target_center[None, :] + self.rest_offset
            acceleration[free] += 460.0 * (target[free] - self.position[free]) + 7.5 * (self.v0[None, :] + self.gravity[None, :] * self.time - self.velocity[free])
        else:
            ids = np.flatnonzero(free)
            if ids.size:
                centroid = np.mean(self.position[ids], axis=0)
                acceleration[ids] += 7.0 * (centroid[None, :] + self.rest_offset[ids] - self.position[ids]) - 1.2 * self.velocity[ids]
        self.velocity[free] += acceleration[free] * dt
        self.position[free] += self.velocity[free] * dt

        wall_candidate = (free & (self.position[:, 0] + self.particle_radius >= self.wall_x) & (self.velocity[:, 0] > 0.0) & (np.abs(self.position[:, 1]) <= self.wall_y_limit) & (self.position[:, 2] >= self.floor_z) & (self.position[:, 2] <= self.wall_z_max))
        if np.any(wall_candidate):
            self.position[wall_candidate, 0] = self.wall_x - self.particle_radius
            self.velocity[wall_candidate] = 0.0
            self.attached[wall_candidate] = True
            if not self.deformed:
                self.deformed = True
                self.impact_time = self.time + dt
                ids = np.flatnonzero(~self.attached)
                if ids.size:
                    self.velocity[ids, 0] *= -0.34
                    radial = self.position[ids] - self.center0[None, :]
                    radial[:, 0] = 0.0
                    self.velocity[ids] += 1.15 * radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-8)
                    phase = self.rest_offset[ids, 1] * 31.0 + self.rest_offset[ids, 2] * 17.0
                    self.velocity[ids, 1] += 0.22 * np.sin(phase)

        free = ~self.attached
        wall_hit = free & (self.position[:, 0] + self.particle_radius > self.wall_x)
        if np.any(wall_hit):
            self.position[wall_hit, 0] = self.wall_x - self.particle_radius
            ids = np.flatnonzero(wall_hit & (self.velocity[:, 0] > 0.0))
            self.velocity[ids, 0] *= -0.28
            self.velocity[ids, 1:] *= 0.78
        floor_hit = free & (self.position[:, 2] < self.floor_z + self.particle_radius)
        if np.any(floor_hit):
            self.position[floor_hit, 2] = self.floor_z + self.particle_radius
            ids = np.flatnonzero(floor_hit & (self.velocity[:, 2] < 0.0))
            self.velocity[ids, 2] *= -0.22
            self.velocity[ids, :2] *= 0.82
        if np.any(self.attached):
            self.position[self.attached, 0] = self.wall_x - self.particle_radius
            self.velocity[self.attached] = 0.0
        self.time += dt
        self.position = np.nan_to_num(self.position, nan=0.0, posinf=10.0, neginf=-10.0)
        self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=50.0, neginf=-50.0)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be non-decreasing")
        while self.time < target - 1e-12:
            self._step(min(0.004, target - self.time))
        wall_gap = self.wall_x - (self.position[:, 0] + self.particle_radius)
        floor_gap = self.position[:, 2] - (self.floor_z + self.particle_radius)
        frame = StateFrame(time=target, position=self.position.copy(), velocity=self.velocity.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(), attributes={"mass_quantity": self.mass_quantity.copy(), "reference_volume": self.reference_volume.copy(), "volume_quantity": (self.mass / self.nominal_density).copy(), "deformation_gradient": np.tile(np.eye(3), (len(self.mass), 1, 1)), "initial_component_id": self.initial_component_id.copy()}, fields={"external_force": self.mass[:, None] * self.gravity[None, :], "contact_signed_distance": np.minimum(wall_gap, floor_gap), "wall_signed_distance": wall_gap, "floor_signed_distance": floor_gap, "cumulative_mass_source": np.zeros(1, dtype=float)}, scalars={"contact_length_scale": self.contact_length, "initial_total_mass": float(self.mass.sum()), "initial_total_volume": float(self.reference_volume.sum())})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"particle_count": int(len(self.mass)), "attached_particle_count": int(np.count_nonzero(self.attached)), "impact_occurred": bool(self.deformed), "impact_time": self.impact_time, "total_mass": float(self.mass.sum())}
