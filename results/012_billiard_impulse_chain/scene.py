"""Deterministic solver-neutral billiard impulse-chain scene."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict): self.config = config
    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}
    def initial_state(self) -> StateFrame:
        positions = np.array([[-2.0,0,0.2],[-0.4,0,0.2],[0.01,0,0.2],[0.42,0,0.2],[0.83,0,0.2],[1.24,0,0.2]], float)
        velocities = np.zeros((6,3)); velocities[0,0] = 2.2
        return StateFrame(0.0, positions, velocities, mass=np.ones(6), body_id=np.arange(6,dtype=int), angular_velocity=np.zeros((6,3)))
    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
