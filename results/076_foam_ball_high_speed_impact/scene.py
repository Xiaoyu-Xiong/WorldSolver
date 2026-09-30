"""Deterministic, solver-neutral scene for the foam-ball impact benchmark."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Expose the undeformed volumetric ball samples and prescribed initial motion."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        ball = self.config["foam_ball"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "high_speed_volumetric_elastic_impact",
            "youngs_modulus": float(ball["youngs_modulus"]),
            "poisson_ratio": float(ball["poisson_ratio"]),
            "density": float(ball["density"]),
        }

    def initial_state(self) -> StateFrame:
        ball = self.config["foam_ball"]
        radius = float(ball["radius"])
        center = np.asarray(ball["initial_center"], dtype=np.float64)
        spacing = radius / 5.0
        axis = np.arange(-radius, radius + spacing * 0.5, spacing)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        local = grid[np.einsum("ij,ij->i", grid, grid) <= radius * radius + 1e-15]
        position = local + center
        velocity = np.broadcast_to(
            np.asarray(ball["initial_velocity"], dtype=np.float64), position.shape
        ).copy()
        total_mass = float(ball["density"]) * (4.0 / 3.0) * np.pi * radius**3
        mass = np.full(len(position), total_mass / len(position), dtype=np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"rest_position": position.copy()},
            scalars={"radius": radius, "total_mass": total_mass},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1, dtype=np.float64).tolist()
