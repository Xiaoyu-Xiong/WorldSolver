"""Deterministic solver-neutral head-on snowball collision scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        snow = self.config["snowballs"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "symmetric_head_on_snowball_fracture",
            "gravity": list(self.config["gravity"]),
            "floor_top_z": float(self.config["floor"]["top_z"]),
            "snowball_radius": float(snow["radius"]),
            "impact_speed_each": float(snow["impact_speed"]),
            "initial_total_momentum": [0.0, 0.0, 0.0],
        }

    def _local_sphere(self) -> np.ndarray:
        snow = self.config["snowballs"]
        radius = float(snow["radius"])
        spacing = float(snow["particle_spacing"])
        count = int(np.ceil(2.0 * radius / spacing))
        axis = (np.arange(count + 1, dtype=float) - count / 2.0) * spacing
        points = np.stack(
            np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        return points[np.einsum("ij,ij->i", points, points) <= radius * radius]

    def initial_state(self) -> StateFrame:
        snow = self.config["snowballs"]
        local = self._local_sphere()
        offset = float(snow["center_offset_x"])
        height = float(snow["center_height"])
        centers = np.array([[-offset, 0.0, height], [offset, 0.0, height]])
        position = np.concatenate((local + centers[0], local + centers[1]), axis=0)
        per_ball = len(local)
        body_id = np.repeat(np.array([0, 1], dtype=np.int64), per_ball)
        speed = float(snow["impact_speed"])
        velocity = np.zeros_like(position)
        velocity[:per_ball, 0] = speed
        velocity[per_ball:, 0] = -speed
        particle_mass = float(snow["total_mass_each"]) / per_ball
        mass = np.full(len(position), particle_mass, dtype=float)
        volume = mass / float(snow["nominal_density"])
        gravity = np.asarray(self.config["gravity"], dtype=float)
        floor_z = float(self.config["floor"]["top_z"])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass.copy(),
                "reference_volume": volume,
                "deformation_gradient": np.tile(np.eye(3), (len(position), 1, 1)),
                "initial_component_id": body_id.copy(),
            },
            fields={
                "external_force": mass[:, None] * gravity[None, :],
                "floor_signed_distance": position[:, 2] - floor_z,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "initial_total_mass": float(mass.sum()),
                "initial_mass_each": float(snow["total_mass_each"]),
                "initial_total_volume": float(volume.sum()),
                "particle_spacing": float(snow["particle_spacing"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
