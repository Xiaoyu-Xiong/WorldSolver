"""Deterministic solver-neutral Bend-Twist benchmark scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    vertex_count = 129

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        rod = self.config["rod"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "gravity-induced 3D bend-twist curling of a naturally curved rod",
            "gravity": list(self.config["gravity"]),
            "simulation_mode": self.config["simulation_mode"],
            "dynamic_object": "curved_rod",
            "natural_curvature": float(rod["natural_curvature"]),
            "natural_twist": 0.0,
            "expected_equilibrium_branch": "three_dimensional",
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        initial = self.config["initial_conditions"]
        length = float(rod["length"])
        radius = float(rod["natural_radius"])
        s = np.linspace(0.0, length, self.vertex_count)
        angle = s / radius
        root = np.asarray(initial["root_position"], dtype=float)

        # Twist-free natural circle in x-z, parametrized so t(0) = -e_z.
        position = np.column_stack((
            radius * (1.0 - np.cos(angle)),
            np.zeros_like(s),
            -radius * np.sin(angle),
        )) + root
        velocity = np.zeros_like(position)

        segment_length = length / (self.vertex_count - 1)
        area = np.pi * float(rod["radius"]) ** 2
        mass = np.full(self.vertex_count, float(rod["density"]) * area * segment_length)
        mass[[0, -1]] *= 0.5
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "arc_length": s,
                "natural_curvature": np.full(self.vertex_count, 1.0 / radius),
                "natural_twist": np.zeros(self.vertex_count),
                "is_clamped_root": np.r_[1.0, np.zeros(self.vertex_count - 1)],
            },
            scalars={
                "rod_radius": float(rod["radius"]),
                "branch_selection_perturbation_y": float(initial["branch_selection_perturbation"][1]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
