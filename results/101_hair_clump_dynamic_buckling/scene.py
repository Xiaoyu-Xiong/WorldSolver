"""Deterministic solver-neutral scene for dynamic buckling of a hair clump."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

NODE_COUNT = 49


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        hair = self.config["hair_clump"]
        drive = self.config["root_motion"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "dynamic_buckling_of_vertically_shaken_hair_clump",
            "gravity": self.config["gravity"],
            "physical_strand_count": 1,
            "render_only_strand_count": 0,
            "material_length_m": float(hair["material_length"]),
            "material_length_measure": "centerline_arclength",
            "inextensible": bool(hair["inextensible"]),
            "dissipation": {
                "internal_gamma_kg_m3_s_inv": float(hair["internal_damping_gamma"]),
                "ambient_drag_nu_kg_m_s_inv": float(hair["ambient_drag_nu"]),
                "ambient_air_velocity": hair["ambient_air_velocity"],
            },
            "boundary_conditions": {"root": "clamped_to_oscillating_root", "tip": "free"},
            "root_motion": drive,
            "event_windows": {
                "hanging_hold": [0.0, float(drive["start_time"])],
                "fast_vertical_oscillation": [float(drive["start_time"]), float(self.config["duration"])],
            },
            "expected_response": "lower-clump lateral lag and a conspicuous transient local bend",
        }

    def initial_state(self) -> StateFrame:
        hair = self.config["hair_clump"]
        length = float(hair["material_length"])
        root = np.asarray(self.config["root_motion"]["rest_position"], dtype=float)
        u = np.linspace(0.0, 1.0, NODE_COUNT)
        position = np.zeros((NODE_COUNT, 3), dtype=float)
        # A minute, smooth side-view imperfection selects a reproducible buckling side.
        position[:, 0] = float(hair["lateral_imperfection_amplitude"]) * np.sin(np.pi * u) ** 2
        position[:, 2] = -length * u
        polygon_length = np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
        position *= length / polygon_length
        position += root
        radius = float(hair["effective_radius"])
        line_density = float(hair["density"]) * np.pi * radius**2
        mass = np.full(NODE_COUNT, line_density * length / (NODE_COUNT - 1))
        mass[[0, -1]] *= 0.5
        mass[0] = 0.0
        index = np.arange(NODE_COUNT, dtype=int)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "node_index": index,
                "is_root": (index == 0).astype(int),
                "is_free_tip": (index == NODE_COUNT - 1).astype(int),
                "kinematic_root": (index == 0).astype(int),
            },
            scalars={
                "material_length": length,
                "internal_damping_gamma": float(hair["internal_damping_gamma"]),
                "ambient_drag_nu": float(hair["ambient_drag_nu"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
