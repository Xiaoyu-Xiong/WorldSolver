"""Scaffold Agent-owned scene definition."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    spring_vertex_count = 128

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "Wilberforce vertical-torsional energy exchange",
            "gravity": [0.0, 0.0, -9.81],
            "spring_body": "helical_spring",
            "weight_body": "suspended_weight",
        }

    def initial_state(self) -> StateFrame:
        # Sample the visible spring as a helical fiber and bind the last sample to the weight.
        count = self.spring_vertex_count
        phase = np.linspace(0.0, 10.5 * 2.0 * np.pi, count)
        top_z, bottom_z = 2.55, 0.99
        radius = 0.16
        spring_position = np.column_stack(
            (
                radius * np.cos(phase),
                radius * np.sin(phase),
                np.linspace(top_z, bottom_z, count),
            )
        )
        weight_position = np.array([[0.0, 0.0, 0.78]], dtype=float)
        position = np.vstack((spring_position, weight_position))

        velocity = np.zeros_like(position)
        velocity[:, 2] = -0.22
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (count + 1, 1))
        angular_velocity = np.zeros_like(position)
        angular_velocity[:count, 2] = 1.8
        angular_velocity[count, 2] = 0.0
        mass = np.r_[np.full(count, 0.12 / count), 1.0]
        body_id = np.r_[np.zeros(count, dtype=int), np.array([1], dtype=int)]
        frame = StateFrame(0.0, position, velocity, orientation=orientation,
                           angular_velocity=angular_velocity,
                           mass=mass, body_id=body_id)
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
