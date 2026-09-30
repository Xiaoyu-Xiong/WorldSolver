"""Deterministic high-tension belt scaffold with prescribed kinematic drivers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        data = json.loads((Path(__file__).resolve().parent / "assets" / "closed_cloth_belt_topology.json").read_text(encoding="utf-8"))
        self._cloth = np.asarray(data["vertices"], dtype=float)
        self._faces = np.asarray(data["faces"], dtype=np.int32)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "gravity": self.config["gravity"],
            "topology": {"faces": self._faces.tolist(), "grid_shape": [120, 9]},
            "timeline": self.config["timeline"],
            "cylinders": self.config["cylinders"],
            "belt_thickness": self.config["belt"]["thickness"],
            "coulomb_friction": self.config["friction"]["coefficient"],
        }

    def initial_state(self) -> StateFrame:
        cylinder_xz = np.asarray(self.config["cylinders"]["initial_centers_xz"], dtype=float)
        cylinder_position = np.asarray([[x, 0.0, z] for x, z in cylinder_xz], dtype=float)
        position = np.vstack([self._cloth, cylinder_position])
        count = len(self._cloth)
        velocity = np.zeros_like(position)
        mass = np.r_[np.full(count, self.config["belt"]["total_mass"] / count), np.zeros(3)]
        body_id = np.r_[np.zeros(count, dtype=int), np.asarray([1, 2, 3], dtype=int)]
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (len(position), 1))
        orientation[-3:] = [0.707106781, 0.0, 0.0, 0.707106781]
        angular_velocity = np.zeros_like(position)
        uv = np.zeros((len(position), 2))
        ring_count, width_nodes = 120, 9
        uv[:count] = np.asarray([[i / ring_count, j / (width_nodes - 1)] for i in range(ring_count) for j in range(width_nodes)])
        frame = StateFrame(
            0.0, position, velocity, orientation=orientation,
            angular_velocity=angular_velocity, mass=mass, body_id=body_id,
            attributes={"uv": uv, "mass_quantity": mass.copy()},
            fields={"cumulative_mass_source": np.zeros(1)},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()

