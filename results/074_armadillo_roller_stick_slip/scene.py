"""Deterministic solver-neutral scene contract for the armadillo roller test."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=float)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"], "seed": self.config["seed"],
            "phenomenon": "high-friction roller stick-slip of a volumetric elastic solid",
            "dynamic_body_ids": {"armadillo": 1},
            "roller_axis": [0.0, 1.0, 0.0],
        }

    def initial_state(self) -> StateFrame:
        path = Path(__file__).resolve().parent / "assets" / "armadillo.obj"
        position = _obj_vertices(path) + np.array([-0.72, 0.0, 0.0])
        n = len(position)
        frame = StateFrame(
            time=0.0, position=position, velocity=np.zeros((n, 3)),
            mass=np.full(n, 2.4 / n), body_id=np.ones(n, dtype=np.int64),
            attributes={"rest_position": position.copy()},
            scalars={"roller_friction": 0.5, "reference_youngs_modulus_pa": 5.0e5},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        count = int(round(float(self.config["duration"]) / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
