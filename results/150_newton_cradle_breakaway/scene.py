"""Deterministic solver-neutral rigid-body scaffold."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "variant": self.config["variant"], "gravity": list(self.config["gravity"]),
                "friction": float(self.config["contact"]["friction"])}

    def initial_state(self) -> StateFrame:
        bodies = self.config["bodies"]
        position = np.asarray([b["position"] for b in bodies], dtype=float)
        orientation = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies], dtype=float)
        mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        frame = StateFrame(
            0.0, position, np.zeros_like(position), orientation=orientation,
            angular_velocity=np.zeros_like(position), mass=mass,
            body_id=np.arange(len(bodies), dtype=np.int32),
            attributes={"release_time": np.asarray([b.get("release_time", 0.) for b in bodies])},
            scalars={"contact_length_scale": float(self.config["contact"]["length_scale"])})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1) * dt).tolist()
