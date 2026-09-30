"""Deterministic, solver-neutral initial state for the weighted seesaw."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        # Board, heavy block, and light block; gravity and contact are solver-owned.
        position = np.array([
            [0.0, 0.0, 0.78],
            [1.05, 0.0, 1.13],
            [-1.0, 0.0, 1.07],
        ], dtype=float)
        velocity = np.zeros((3, 3), dtype=float)
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
        return StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros((3, 3), dtype=float),
            mass=np.array([4.0, 3.0, 1.0], dtype=float),
            body_id=np.array([0, 1, 2], dtype=int),
        )

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
