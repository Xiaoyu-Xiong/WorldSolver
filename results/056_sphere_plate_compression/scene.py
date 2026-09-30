"""Deterministic solver-neutral scene for symmetric plate compression."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        self._vertices = self._load_obj_vertices(
            Path(__file__).resolve().parent / "assets" / "soft_sphere.obj"
        )

    @staticmethod
    def _load_obj_vertices(path: Path) -> np.ndarray:
        vertices = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
        return np.asarray(vertices, dtype=float)

    def metadata(self) -> dict:
        sphere = self.config["sphere"]
        plates = self.config["plates"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "incompressible elastic sphere compressed between opposed plates",
            "material_class": sphere["material_class"],
            "spatial_dimensions": 3,
            "gravity": list(self.config["gravity"]),
            "kinematic_boundaries": {
                "lower_plate": {
                    "initial_inner_face_z": plates["initial_inner_faces_z"][0],
                    "final_inner_face_z": plates["final_inner_faces_z"][0],
                    "velocity": [0.0, 0.0, plates["speed_each"]],
                    "active_interval": [plates["motion_start"], plates["motion_end"]],
                },
                "upper_plate": {
                    "initial_inner_face_z": plates["initial_inner_faces_z"][1],
                    "final_inner_face_z": plates["final_inner_faces_z"][1],
                    "velocity": [0.0, 0.0, -plates["speed_each"]],
                    "active_interval": [plates["motion_start"], plates["motion_end"]],
                },
            },
            "minimum_positive_gap": plates["minimum_positive_gap"],
        }

    def initial_state(self) -> StateFrame:
        center = np.asarray(self.config["sphere"]["center"], dtype=float)
        position = self._vertices + center
        count = len(position)
        mass = np.full(count, float(self.config["sphere"]["mass"]) / count)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            body_id=np.zeros(count, dtype=np.int64),
            attributes={
                "rest_position": position.copy(),
                "deformation_gradient": np.tile(np.eye(3), (count, 1, 1)),
                "rest_volume_share": np.full(
                    count, float(self.config["sphere"]["rest_volume"]) / count
                ),
            },
            scalars={
                "rest_volume": float(self.config["sphere"]["rest_volume"]),
                "initial_plate_gap": 1.7013016168,
                "minimum_positive_gap": float(self.config["plates"]["minimum_positive_gap"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
