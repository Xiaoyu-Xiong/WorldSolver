"""Deterministic solver-neutral scene for a gravity-loaded rod cantilever."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    vertex_count = 65

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "quasi-static gravitational sag of a clamped-free elastic rod",
            "gravity": list(self.config["gravity"]),
            "simulation_mode": self.config["simulation_mode"],
            "dynamic_object": "cantilever_rod",
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        initial = self.config["initial_conditions"]
        start = np.asarray(initial["centerline_start"], dtype=float)
        end = np.asarray(initial["centerline_end"], dtype=float)
        position = np.linspace(start, end, self.vertex_count)
        velocity = np.zeros_like(position)
        segment_length = float(rod["length"]) / (self.vertex_count - 1)
        area = np.pi * float(rod["radius"]) ** 2
        mass = np.full(self.vertex_count, float(rod["density"]) * area * segment_length)
        mass[[0, -1]] *= 0.5
        arc_length = np.linspace(0.0, float(rod["length"]), self.vertex_count)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "arc_length": arc_length,
                "is_clamped_root": np.r_[1.0, np.zeros(self.vertex_count - 1)],
            },
            scalars={"rod_radius": float(rod["radius"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
