"""Deterministic, solver-neutral pneumatic cushion scene contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Expose the uninflated solid wall mesh and prescribed loading metadata."""

    def __init__(self, config: dict):
        self.config = config
        self._asset = Path(__file__).parent / "assets" / "cushion_solid.npz"

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "finite-thickness sealed cushion inflation",
            "topology_asset": "assets/cushion_solid.npz",
            "pressure_surface": "inner_chamber_wall",
        }

    def initial_state(self) -> StateFrame:
        with np.load(self._asset) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            nodal_volume = np.asarray(data["nodal_volume"], dtype=np.float64)
            wall_region = np.asarray(data["wall_region"], dtype=np.int32)
        density = float(self.config["material_properties"]["density_kg_m3"])
        frame = StateFrame(
            time=0.0,
            position=vertices.copy(),
            velocity=np.zeros_like(vertices),
            mass=density * nodal_volume,
            attributes={"wall_region": wall_region},
            scalars={
                "internal_pressure_pa": float(
                    self.config["loading"]["initial_internal_pressure_pa"]
                ),
                "external_pressure_pa": float(
                    self.config["loading"]["external_pressure_pa"]
                ),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
