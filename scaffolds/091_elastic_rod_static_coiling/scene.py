"""Deterministic solver-neutral scene for quasi-static elastic-rod coiling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        cfg = self.config
        return {
            "scene_id": cfg["scene_id"],
            "seed": cfg["seed"],
            "phenomenon": "static coiling of a naturally straight elastic rod",
            "gravity": cfg["gravity"],
            "rod": cfg["rod"],
            "injector": cfg["injector"],
            "substrate": cfg["substrate"],
            "initial_conditions": cfg["initial_conditions"],
            "event_sequence": [
                "continued_downward_injection",
                "suspended_heel_bends_sideways",
                "contact_point_orbits_consistently",
                "successive_nearly_circular_coils_stick_to_substrate",
            ],
        }

    def initial_state(self) -> StateFrame:
        asset = Path(__file__).parent / self.config["rod"]["topology_asset"]
        data = json.loads(asset.read_text(encoding="utf-8"))
        position = np.asarray(data["vertices"], dtype=float)
        n = len(position)
        radius = float(self.config["rod"]["radius"])
        density = float(self.config["rod"]["density"])
        segment_lengths = np.linalg.norm(np.diff(position, axis=0), axis=1)
        lumped = np.empty(n, dtype=float)
        lumped[0], lumped[-1] = segment_lengths[0] / 2, segment_lengths[-1] / 2
        lumped[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2
        mass = density * np.pi * radius**2 * lumped
        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros((n, 3), dtype=float),
            orientation=orientation,
            angular_velocity=np.zeros((n, 3), dtype=float),
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": np.asarray(data["arc_coordinate"], dtype=float),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initial_contact": (np.arange(n) == n - 1).astype(float),
                "rod_radius": np.full(n, radius),
            },
            scalars={
                "injection_speed": float(self.config["injector"]["injection_speed"]),
                "natural_curvature": 0.0,
                "substrate_speed": 0.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
