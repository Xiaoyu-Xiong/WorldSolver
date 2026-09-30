"""Deterministic solver-neutral scene for localized helical rod buckling."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    vertex_count = 129

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "localized helical buckling under twist and axial shortening",
            "dynamic_object": "twisted_rod",
            "loading_sequence": ["apply_end_twist", "incremental_clamp_shortening", "hold"],
            "obstacle_forcing": False,
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        initial = self.config["initial_conditions"]
        length = float(rod["length"])
        s = np.linspace(0.0, length, self.vertex_count)
        x = s - 0.5 * length
        # Smooth zero-displacement/zero-slope seed at both clamps selects a reproducible branch.
        q = s / length
        window = np.sin(np.pi * q) ** 2
        amp = float(initial["branch_selection_perturbation_amplitude"])
        position = np.column_stack((x, amp * window * np.sin(2.0 * np.pi * q), np.zeros_like(x)))
        velocity = np.zeros_like(position)
        segment_length = length / (self.vertex_count - 1)
        mass = np.full(self.vertex_count, float(rod["density"]) * np.pi * float(rod["radius"])**2 * segment_length)
        mass[[0, -1]] *= 0.5
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "arc_length": s,
                "natural_curvature": np.zeros(self.vertex_count),
                "natural_twist": np.zeros(self.vertex_count),
                "endpoint_clamp": np.r_[1.0, np.zeros(self.vertex_count - 2), 2.0],
            },
            scalars={
                "rod_radius": float(rod["radius"]),
                "target_relative_twist_rad": float(rod["source_imposed_end_twist_rad"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
