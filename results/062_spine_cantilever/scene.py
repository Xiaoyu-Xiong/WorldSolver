"""Deterministic solver-neutral scene for the hanging rubber spine benchmark."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "gravity": list(self.config["gravity"]),
            "material": dict(self.config["material"]),
            "fixed_region": dict(self.config["boundary_conditions"]["fixed_region"]),
            "phenomenon": "gravity sag of a three-dimensional segmented soft cantilever",
        }

    @staticmethod
    def _vertices() -> np.ndarray:
        path = Path(__file__).resolve().parent / "assets" / "spine_arm.obj"
        vertices = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
        return np.asarray(vertices, dtype=float)

    def initial_state(self) -> StateFrame:
        position = self._vertices()
        n = len(position)
        fixed = (position[:, 0] <= float(
            self.config["boundary_conditions"]["fixed_region"]["x_max"]
        )).astype(float)
        density = float(self.config["material"]["density"])
        # Uniform nodal quadrature weights; total mass matches the mesh envelope closely.
        envelope_volume = 0.075 * 0.18 * 1.18
        mass = np.full(n, density * envelope_volume / n, dtype=float)
        gravity = np.asarray(self.config["gravity"], dtype=float)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "fixed": fixed,
                "external_force": mass[:, None] * gravity[None, :],
                "rest_position": position.copy(),
            },
            scalars={"gravity_x": gravity[0], "gravity_y": gravity[1], "gravity_z": gravity[2]},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
