"""Deterministic solver-neutral scene contract for a chambered soft finger."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        pressure = self.config["pressure"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "pressure_driven_bending_of_chambered_soft_finger",
            "geometry_dimension": 3,
            "deformable_object": "soft_finger",
            "fixed_base_x_range_m": self.config["initial_conditions"]["base_support_region_x"],
            "pressure_boundary": {
                "target_pa": pressure["chamber_pressure_pa"],
                "ramp_start_s": pressure["ramp_start_s"],
                "ramp_end_s": pressure["ramp_end_s"],
                "hold_until_s": pressure["hold_until_s"],
                "applies_to": "six_enclosed_internal_chambers",
            },
            "rest_configuration": "straight, unpressurized, base supported",
        }

    def initial_state(self) -> StateFrame:
        asset = Path(__file__).parent / "assets" / "pneumatic_finger.obj"
        vertices = []
        for line in asset.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
        position = np.asarray(vertices, dtype=float)
        velocity = np.zeros_like(position)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=np.full(len(position), 0.001, dtype=float),
            attributes={
                "fixed_base": (position[:, 0] <= 0.030).astype(float),
                "rest_position": position.copy(),
            },
            scalars={"chamber_pressure_pa": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
