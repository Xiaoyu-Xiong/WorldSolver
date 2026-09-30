"""Deterministic solver-neutral thumb/index web pinch scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


ASSET = Path(__file__).parent / "assets" / "hand_flesh.obj"


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
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
            "phenomenon": "outward volumetric bulging of thumb-index web under pinch",
            "solid_dimension": 3,
            "rest_topology_asset": "assets/hand_flesh.obj",
            "gravity": self.config["gravity"],
            "fixed_region": {"axis": "x", "maximum": -0.150},
            "driven_bones": {
                "index_bone": {"motion": "held in initial pose"},
                "thumb_bone": self.config["thumb_drive"],
            },
            "material_intent": {
                "flesh": "isotropic nearly incompressible deformable solid",
                "observable": "web mound rises toward camera while thumb approaches index base",
            },
        }

    def initial_state(self) -> StateFrame:
        position = _load_obj_vertices(ASSET)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=np.full(len(position), 0.42 / len(position)),
            attributes={
                "rest_position": position.copy(),
                "fixed_mask": (position[:, 0] <= -0.150).astype(np.int32),
                "web_region": ((position[:, 0] > -0.015) & (position[:, 0] < 0.085)
                               & (position[:, 1] > -0.055) & (position[:, 1] < 0.040)).astype(np.int32),
            },
            scalars={"solid_dimension": 3.0, "nominal_density": 1040.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
