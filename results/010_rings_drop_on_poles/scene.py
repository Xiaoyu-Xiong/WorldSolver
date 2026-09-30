"""Deterministic initial state for the rings-over-poles benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        # Staggered layers give rings varied pole contacts while preserving determinism.
        positions = []
        for layer in range(8):
            z = 3.0 + 0.34 * layer
            for row in range(4):
                for col in range(3):
                    positions.append([(col - 1) * 0.62 + (0.10 if layer % 2 else 0.0),
                                      (row - 1.5) * 0.62, z])
        pos = np.asarray(positions, dtype=float)
        n = len(pos)
        return StateFrame(0.0, pos, np.zeros_like(pos),
                          orientation=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
                          angular_velocity=np.zeros((n, 3)), mass=np.ones(n))

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
