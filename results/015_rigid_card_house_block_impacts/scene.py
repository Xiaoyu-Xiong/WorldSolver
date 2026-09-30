"""Deterministic solver-neutral rigid card-house scene definition."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial rigid-body state and prescribed sources for the benchmark."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "gravity": list(self.config["gravity"]),
            "friction_coefficient": float(
                self.config["contact_parameters"]["friction_coefficient"]
            ),
            "restitution": float(self.config["contact_parameters"]["restitution"]),
            "impact_block_release_times": list(
                self.config["initial_conditions"]["impact_block_release_times"]
            ),
        }

    def initial_state(self) -> StateFrame:
        positions = np.array(
            [
                [-0.925, 0.0, 0.397], [-0.475, 0.0, 0.397],
                [0.475, 0.0, 0.397], [0.925, 0.0, 0.397],
                [-0.7, 0.0, 0.81], [0.7, 0.0, 0.81],
                [-0.925, 0.0, 1.222], [-0.475, 0.0, 1.222],
                [0.475, 0.0, 1.222], [0.925, 0.0, 1.222],
                [-0.7, 0.0, 2.25], [0.7, 0.0, 2.25], [0.0, 0.0, 2.65],
            ],
            dtype=float,
        )
        q_minus = [0.0, -0.5, 0.0, 0.8660254]
        q_plus = [0.0, 0.5, 0.0, 0.8660254]
        q_identity = [0.0, 0.0, 0.0, 1.0]
        orientations = np.array(
            [q_minus, q_plus, q_minus, q_plus, q_identity, q_identity,
             q_minus, q_plus, q_minus, q_plus,
             q_identity, q_identity, q_identity],
            dtype=float,
        )
        masses = np.array([0.08] * 10 + [0.35, 0.35, 0.50], dtype=float)
        frame = StateFrame(
            time=0.0,
            position=positions,
            velocity=np.zeros((13, 3), dtype=float),
            orientation=orientations,
            angular_velocity=np.zeros((13, 3), dtype=float),
            mass=masses,
            body_id=np.arange(13, dtype=int),
            attributes={
                "release_time": np.array(
                    [0.0] * 10
                    + self.config["initial_conditions"]["impact_block_release_times"],
                    dtype=float,
                ),
                "is_impact_block": np.array([0.0] * 10 + [1.0, 1.0, 1.0]),
            },
            scalars={
                "friction_coefficient": float(
                    self.config["contact_parameters"]["friction_coefficient"]
                ),
                "restitution": float(
                    self.config["contact_parameters"]["restitution"]
                ),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
