"""Deterministic, solver-neutral scene contract for the squeeze-out benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial body state and sampling schedule; dynamics are solver-owned."""

    _POSITIONS = np.array([
        [-0.25, -0.12, 4.13],
        [0.22, 0.12, 3.34],
        [-0.18, 0.10, 2.55],
        [0.18, -0.10, 1.72],
        [-0.20, -0.02, 0.98],
    ], dtype=float)

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "dimension": 3,
            "soft_body_count": 5,
            "kinematic_object": "pusher_plate",
            "outlet_center_z": 0.0,
        }

    def initial_state(self) -> StateFrame:
        frame = StateFrame(
            time=0.0,
            position=self._POSITIONS.copy(),
            velocity=np.zeros_like(self._POSITIONS),
            mass=np.array([0.62, 0.54, 0.70, 0.58, 0.50], dtype=float),
            body_id=np.arange(5, dtype=np.int64),
            attributes={"reference_position": self._POSITIONS.copy()},
            scalars={"gravity_z": -9.81},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
