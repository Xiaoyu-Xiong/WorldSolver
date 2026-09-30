"""Deterministic, solver-neutral elastic rod scene contract."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial volumetric-rod samples and prescribed top-grip metadata.

    The downstream solver owns deformation and contact evolution.  The vertex
    attributes identify the driven cap and leave every other rod vertex free.
    """

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "elastic_rod_ground_stick_slip",
            "drag_direction": [1.0, 0.0, 0.0],
            "ground_friction_coefficient": 0.35,
            "boundary_conditions": self.config["boundary_conditions"],
        }

    @staticmethod
    def _vertices() -> np.ndarray:
        rings, sides, radius, height = 10, 6, 0.055, 1.45
        verts = []
        for k in range(rings):
            z = radius + height * k / (rings - 1)
            for j in range(sides):
                a = 2.0 * np.pi * j / sides
                verts.append([radius * np.cos(a), radius * np.sin(a), z])
        return np.asarray(verts, dtype=float)

    def initial_state(self) -> StateFrame:
        x = self._vertices()
        top = (x[:, 2] > 1.45).astype(float)
        frame = StateFrame(
            0.0,
            x,
            np.zeros_like(x),
            mass=np.full(len(x), 0.018 / len(x)),
            attributes={"driven_top": top, "rod_vertex": np.ones(len(x))},
            scalars={"rod_radius": 0.055, "rod_height": 1.45},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
