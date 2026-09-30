"""Deterministic solver-neutral initial state for an inclined plastic chair."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        # Single rigid chair (seat + back + four legs merged into chair.stl),
        # at rest on the incline with all four feet in contact and the seat
        # plane parallel to the inclined surface. Position is the center of mass.
        position = np.array([[0.0354, 0.0, 1.6074]])
        return StateFrame(
            0.0,
            position,
            np.zeros((1, 3)),
            orientation=np.array([[0.0, 0.1564, 0.0, 0.9877]]),
            angular_velocity=np.zeros((1, 3)),
            mass=np.array([10.8]),
            body_id=np.array([0]),
        )

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
