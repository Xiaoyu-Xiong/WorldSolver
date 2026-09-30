"""Deterministic solver-neutral scene contract for rubber-cube compression."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        asset = Path(__file__).resolve().parent / "assets" / "rubber_cube_mesh.npz"
        with np.load(asset) as mesh:
            self.vertices = np.asarray(mesh["vertices"], dtype=float)
            self.tetrahedra = np.asarray(mesh["tetrahedra"], dtype=np.int64)
            self.faces = np.asarray(mesh["faces"], dtype=np.int64)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "solid_dimension": 3,
            "topology": "volumetric_tetrahedral",
        }

    def initial_state(self) -> StateFrame:
        n = len(self.vertices)
        frame = StateFrame(
            time=0.0,
            position=self.vertices.copy(),
            velocity=np.zeros((n, 3), dtype=float),
            mass=np.full(n, 1.0 / n, dtype=float),
            attributes={
                "rest_position": self.vertices.copy(),
                "bottom_face": np.isclose(self.vertices[:, 2], 0.0).astype(float),
                "top_face": np.isclose(self.vertices[:, 2], 0.2).astype(float),
            },
            scalars={"applied_traction_pa": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
