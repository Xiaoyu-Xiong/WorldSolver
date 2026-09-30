"""Deterministic solver-neutral branching elastic-rod scene."""
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
            "scene_id": cfg["scene_id"], "seed": cfg["seed"],
            "phenomenon": "wind-driven, branch-dependent bending and twisting",
            "gravity": cfg["gravity"], "wind": cfg["wind"],
            "rod_network": cfg["rod_network"],
            "rigid_junctions": cfg["rigid_junctions"],
            "boundary_conditions": cfg["initial_conditions"],
            "event_sequence": ["wind_ramp", "first_gust_peak", "partial_recovery",
                               "strongest_gust", "asynchronous_branch_ringing"],
        }

    def initial_state(self) -> StateFrame:
        asset = Path(__file__).parent / self.config["rod_network"]["topology_asset"]
        data = json.loads(asset.read_text(encoding="utf-8"))
        position = np.asarray(data["vertices"], dtype=float)
        n = len(position)
        levels = np.asarray(data["vertex_level"], dtype=int)
        junction_indices = np.asarray(self.config["rigid_junctions"]["vertex_indices"], dtype=int)
        is_junction = np.zeros(n, dtype=float); is_junction[junction_indices] = 1.0
        radii = np.asarray(self.config["rod_network"]["radius_by_level"], dtype=float)[levels]
        density = np.asarray(self.config["rod_network"]["linear_density_by_level"], dtype=float)[levels]
        mass = density * np.pi * radii ** 2 * np.asarray(data["lumped_length"], dtype=float)
        orientation = np.zeros((n, 4), dtype=float); orientation[:, 3] = 1.0
        frame = StateFrame(
            0.0, position, np.zeros((n, 3)), orientation=orientation,
            angular_velocity=np.zeros((n, 3)), mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "is_rod_vertex": np.ones(n), "is_rigid_junction": is_junction,
                "is_fixed_root": (np.arange(n) == 0).astype(float),
                "branch_id": np.asarray(data["branch_id"], dtype=float),
                "branch_level": levels.astype(float), "rod_radius": radii,
            },
            scalars={"wind_speed": 0.0, "gravity_magnitude": 9.81},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
