"""Deterministic solver-neutral scene for rod meandering on a moving belt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        cfg = self.config
        return {
            "scene_id": cfg["scene_id"],
            "seed": cfg["seed"],
            "phenomenon": "elastic rod meandering followed by speed-matched decay",
            "gravity": cfg["gravity"],
            "regime": cfg["regime"],
            "rod": cfg["rod"],
            "injector": cfg["injector"],
            "belt": cfg["belt"],
            "initial_conditions": cfg["initial_conditions"],
            "event_sequence": cfg["event_sequence"],
        }

    def initial_state(self) -> StateFrame:
        path = Path(__file__).parent / self.config["rod"]["topology_asset"]
        data = json.loads(path.read_text(encoding="utf-8"))
        position = np.asarray(data["vertices"], dtype=float)
        n = len(position)
        radius = float(self.config["rod"]["radius"])
        density = float(self.config["rod"]["density"])
        lengths = np.linalg.norm(np.diff(position, axis=0), axis=1)
        lumped_length = np.empty(n)
        lumped_length[[0, -1]] = [lengths[0] / 2.0, lengths[-1] / 2.0]
        lumped_length[1:-1] = (lengths[:-1] + lengths[1:]) / 2.0
        deposited = np.asarray(data["deposited_mask"], dtype=float)
        velocity = np.zeros((n, 3))
        velocity[deposited > 0.5] = np.asarray(self.config["belt"]["travel_direction"]) * float(self.config["belt"]["initial_speed"])
        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros((n, 3)),
            mass=density * np.pi * radius**2 * lumped_length,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": np.asarray(data["arc_coordinate"], dtype=float),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initially_deposited": deposited,
                "is_sticking_to_belt": deposited.copy(),
                "rod_radius": np.full(n, radius),
                "natural_curvature": np.zeros(n),
            },
            scalars={
                "injection_speed": float(self.config["injector"]["injection_speed"]),
                "initial_belt_speed": float(self.config["belt"]["initial_speed"]),
                "matched_belt_speed": float(self.config["belt"]["matched_speed"]),
                "speed_switch_time": float(self.config["belt"]["speed_switch_time"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
