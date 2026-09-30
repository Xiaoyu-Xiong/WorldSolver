"""Scaffold Agent-owned scene definition."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        n = 10
        z = 0.15 + 0.205 * np.arange(n, dtype=float)
        x = np.array([0.015, -0.012, 0.01, -0.008, 0.006, -0.005, 0.004, -0.003, 0.002, 0.0])
        position = np.column_stack((x, np.zeros(n), z))
        velocity = np.zeros((n, 3), dtype=float)
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        return StateFrame(0.0, position, velocity, orientation=orientation,
                          angular_velocity=np.zeros((n, 3)),
                          mass=np.ones(n), body_id=np.arange(1, n + 1, dtype=int))

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
