"""Deterministic solver-neutral inclined-block scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        position = np.array([[-1.35, 0.0, 1.05]], dtype=float)
        velocity = np.zeros((1, 3), dtype=float)
        orientation = np.array([[-0.258819, 0.0, 0.0, 0.965926]], dtype=float)
        return StateFrame(
            0.0, position, velocity, orientation=orientation,
            angular_velocity=np.zeros((1, 3)), mass=np.array([2.0]),
            body_id=np.array([1], dtype=int),
        )

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
