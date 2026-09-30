"""Deterministic, solver-neutral elbow-flexion scene contract."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _obj_vertices(path: Path) -> np.ndarray:
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
            "seed": self.config["seed"],
            "phenomenon": "three-dimensional flesh bulging during elbow flexion",
            "deformable_object": "arm_flesh",
            "driven_bones": ["humerus_bone", "forearm_bone"],
            "prescribed_motion": self.config["boundary_conditions"]["forearm_bone"],
            "rest_mesh": "assets/arm_flesh.obj",
            "units": "meters",
        }

    def initial_state(self) -> StateFrame:
        positions = _obj_vertices(Path(__file__).parent / "assets" / "arm_flesh.obj")
        # Equal nodal masses encode the requested total rest-volume density without
        # implying any constitutive law or discretization in the downstream solver.
        mass = np.full(len(positions), 3.05 / len(positions), dtype=float)
        frame = StateFrame(
            time=0.0,
            position=positions,
            velocity=np.zeros_like(positions),
            mass=mass,
            attributes={"rest_position": positions.copy()},
            scalars={"flesh_density_kg_m3": 1050.0, "elbow_flexion_degrees": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
