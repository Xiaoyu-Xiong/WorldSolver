"""Deterministic solver-neutral scene for a vertically driven curly hair clump."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

NODE_COUNT = 65


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        guide = self.config["physical_guide"]
        drive = self.config["root_motion"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "curly_hair_vertical_oscillation",
            "gravity": self.config["gravity"],
            "physical_strand_count": 1,
            "render_only_strand_count": 0,
            "material_length_m": float(guide["material_length"]),
            "material_length_measure": "centerline_arclength",
            "inextensible": bool(guide["inextensible"]),
            "cross_section_radius_m": float(guide["radius"]),
            "density_kg_m3": float(guide["density"]),
            "natural_shape": {
                "type": "circular_helix",
                "radius_m": float(guide["natural_helix_radius"]),
                "pitch_per_turn_m": float(guide["natural_helix_pitch_per_turn"]),
            },
            "dissipation": {
                "internal_gamma_kg_m3_s_inv": float(guide["internal_damping_gamma"]),
                "ambient_drag_nu_kg_m_s_inv": float(guide["ambient_drag_nu"]),
                "ambient_air_velocity": [0.0, 0.0, 0.0],
            },
            "boundary_conditions": {
                "root": "clamped_to_oscillating_root",
                "tip": guide["tip_boundary"],
            },
            "root_motion": drive,
            "event_windows": {
                "suspended_before_drive": [0.0, float(drive["start_time"])],
                "repeated_vertical_oscillation": [float(drive["start_time"]), float(self.config["duration"])],
            },
        }

    def initial_state(self) -> StateFrame:
        guide = self.config["physical_guide"]
        length = float(guide["material_length"])
        radius = float(guide["natural_helix_radius"])
        pitch = float(guide["natural_helix_pitch_per_turn"])
        root = np.asarray(self.config["root_motion"]["rest_position"], dtype=float)
        # Solve for turns against the polygonal centerline so its discrete
        # arclength (the solver-facing material length) is exact.
        turns = length / np.sqrt((2.0 * np.pi * radius) ** 2 + pitch**2)
        theta = np.linspace(0.0, 2.0 * np.pi * turns, NODE_COUNT)
        position = np.empty((NODE_COUNT, 3), dtype=float)
        position[:, 0] = radius * (1.0 - np.cos(theta))
        position[:, 1] = radius * np.sin(theta)
        position[:, 2] = -(pitch / (2.0 * np.pi)) * theta
        polygon_length = np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
        position *= length / polygon_length
        position += root
        line_density = float(guide["density"]) * np.pi * float(guide["radius"]) ** 2
        mass = np.full(NODE_COUNT, line_density * length / (NODE_COUNT - 1))
        mass[[0, -1]] *= 0.5
        mass[0] = 0.0
        node_index = np.arange(NODE_COUNT, dtype=int)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "node_index": node_index,
                "is_root": (node_index == 0).astype(int),
                "is_free_tip": (node_index == NODE_COUNT - 1).astype(int),
                "kinematic_root": (node_index == 0).astype(int),
            },
            scalars={
                "material_length": length,
                "internal_damping_gamma": float(guide["internal_damping_gamma"]),
                "ambient_drag_nu": float(guide["ambient_drag_nu"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
