"""Deterministic, solver-neutral six-wall Octocat compactor scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        c = self.config["compactor"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "three-dimensional elastic self-contact under six-wall compression and release",
            "deformable_body": {"id": "octocat", "volumetric_solid": True, "tentacle_count": 8},
            "kinematic_walls": ["wall_left", "wall_right", "wall_front", "wall_back", "wall_bottom", "wall_top"],
            "compactor_schedule": c,
            "wall_pose_contract": "wall_transforms(time) returns centers for the six frozen wall objects",
        }

    def _smoothstep(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def inner_half_extent(self, time: float) -> float:
        c = self.config["compactor"]
        open_h, closed_h = float(c["open_inner_half_extent"]), float(c["closed_inner_half_extent"])
        inward, withdraw = c["inward_interval"], c["withdrawal_interval"]
        if time < inward[0]:
            return open_h
        if time <= inward[1]:
            return open_h + (closed_h - open_h) * self._smoothstep((time - inward[0]) / (inward[1] - inward[0]))
        if time < withdraw[0]:
            return closed_h
        if time <= withdraw[1]:
            return closed_h + (open_h - closed_h) * self._smoothstep((time - withdraw[0]) / (withdraw[1] - withdraw[0]))
        return open_h

    def wall_transforms(self, time: float) -> dict[str, dict[str, list[float]]]:
        c = self.config["compactor"]
        center = self.inner_half_extent(time) + 0.5 * float(c["wall_thickness"])
        return {
            "wall_left": {"translation": [-center, 0.0, 0.0]},
            "wall_right": {"translation": [center, 0.0, 0.0]},
            "wall_front": {"translation": [0.0, -center, 0.0]},
            "wall_back": {"translation": [0.0, center, 0.0]},
            "wall_bottom": {"translation": [0.0, 0.0, -center]},
            "wall_top": {"translation": [0.0, 0.0, center]},
        }

    def initial_state(self) -> StateFrame:
        mesh_path = Path(__file__).resolve().parent / "assets" / "octocat.obj"
        vertices = np.asarray(trimesh.load_mesh(mesh_path, process=False).vertices, dtype=float)
        velocity = np.zeros_like(vertices)
        mass = np.full(len(vertices), 1.0 / len(vertices))
        frame = StateFrame(0.0, vertices, velocity, mass=mass)
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
