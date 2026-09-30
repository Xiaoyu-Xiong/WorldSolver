from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"],
                "phenomenon": "two_way_coupled_limit_cycle_flapping"}

    def initial_state(self) -> StateFrame:
        # Four flag corner samples followed by four carrier-fluid samples.
        flag = np.array([[0.0, -0.025, 0.0], [0.0, 0.025, 0.0],
                         [1.0, -0.025, 0.0], [1.0, 0.025, 0.0]])
        fluid = np.array([[-0.75, -0.75, 0.0], [-0.75, 0.75, 0.0],
                          [4.75, -0.75, 0.0], [4.75, 0.75, 0.0]])
        position = np.vstack([flag, fluid])
        velocity = np.vstack([np.zeros((4, 3)), np.tile([1.0, 0.0, 0.0], (4, 1))])
        body_id = np.array([2, 2, 2, 2, 1, 1, 1, 1], dtype=int)
        return StateFrame(0.0, position, velocity, mass=np.ones(8), body_id=body_id,
                          fields={"vorticity": np.zeros((4, 4))})

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
