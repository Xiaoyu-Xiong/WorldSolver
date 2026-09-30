"""Deterministic solver-neutral scene for the Figure 11 octopus squash test."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _obj_vertices() -> np.ndarray:
    path = Path(__file__).parent / "assets" / "octopus.obj"
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vertices, dtype=float)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "gravity drop, cylinder squash, and elastic recovery",
            "gravity": list(self.config["gravity"]),
            "solid_dimension": 3,
            "rest_topology_asset": "assets/octopus.obj",
            "kinematic_object": "upper_cylinder",
            "kinematic_keyframes": self.config["upper_cylinder_motion"]["keyframes"],
            "event_intervals": self.config["timeline"],
            "material_intent": {"elastic": True, "near_incompressible": True},
        }

    def initial_state(self) -> StateFrame:
        position = _obj_vertices()
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=np.full(len(position), 12.0 / len(position)),
            body_id=np.zeros(len(position), dtype=np.int32),
            attributes={"rest_position": position.copy()},
            scalars={"gravity_z": -9.81, "initial_clearance": 0.42},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
