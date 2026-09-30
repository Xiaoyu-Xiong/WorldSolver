"""Deterministic, solver-neutral scene contract for the pin-cushion experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        self._root = Path(__file__).resolve().parent

    def metadata(self) -> dict:
        bc = self.config["boundary_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "units": "meters",
            "up_axis": "z",
            "phenomenon": "volumetric elastic ball compressed between codimensional segment arrays",
            "gravity": list(self.config["gravity"]),
            "deformable_body": {
                "object_id": "soft_ball",
                "rest_topology_asset": "assets/soft_ball_volume.json",
                "dimensionality": 3,
                "density": float(self.config["initial_conditions"]["ball_density"]),
            },
            "kinematic_motion": {
                "object_id": "upper_pin_array",
                "type": "piecewise_linear_translation",
                "axis": list(bc["upper_pin_array"]["axis"]),
                "keyframes": [dict(k) for k in bc["upper_pin_array"]["keyframes"]],
            },
            "event_sequence": dict(self.config["event_sequence"]),
        }

    def initial_state(self) -> StateFrame:
        data = json.loads((self._root / "assets" / "soft_ball_volume.json").read_text())
        position = np.asarray(data["vertices"], dtype=float)
        velocity = np.broadcast_to(
            np.asarray(self.config["initial_conditions"]["ball_velocity"], dtype=float),
            position.shape,
        ).copy()
        density = float(self.config["initial_conditions"]["ball_density"])
        tetrahedra = np.asarray(data["tetrahedra"], dtype=int)
        mass = np.zeros(len(position), dtype=float)
        for tet in tetrahedra:
            p = position[tet]
            volume = abs(np.linalg.det(np.stack((p[1]-p[0], p[2]-p[0], p[3]-p[0]), axis=1))) / 6.0
            mass[tet] += density * volume / 4.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"rest_position": position.copy()},
            scalars={"density": density},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
