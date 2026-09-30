"""Deterministic, solver-neutral scene contract for the squishy-ball impact."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial geometry samples and timing; deformation is solver-owned."""

    def __init__(self, config: dict):
        self.config = config
        self._asset = Path(__file__).parent / "assets" / "squishy_ball_volume.npz"

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "description": "Volumetric tendril-covered toy moving toward a fixed glass wall",
        }

    def initial_state(self) -> StateFrame:
        with np.load(self._asset) as data:
            position = np.asarray(data["nodes"], dtype=np.float64).copy()
        velocity = np.zeros_like(position)
        velocity[:, 0] = float(
            self.config["initial_conditions"]["squishy_ball"]["linear_velocity"][0]
        )
        # Lumped positive masses are deliberately uniform; total mass is normalized.
        mass = np.full(len(position), 1.0 / len(position), dtype=np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=np.ones(len(position), dtype=np.int64),
            attributes={"component": np.zeros(len(position), dtype=np.int64)},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
