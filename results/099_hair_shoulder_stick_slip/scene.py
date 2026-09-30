"""Deterministic, solver-neutral wavy-hair/shoulder contact scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

STRAND_COUNT = 84
NODES_PER_STRAND = 22
STRAND_RADIUS = 0.0042


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        motion = self.config["head_motion"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "hair_shoulder_hair_hair_stick_slip",
            "strand_count": STRAND_COUNT,
            "nodes_per_strand": NODES_PER_STRAND,
            "strand_radius": STRAND_RADIUS,
            "gravity": [0.0, 0.0, -9.81],
            "roots_attached_to_head": True,
            "rest_shape": "wavy",
            "contact": {
                "hair_hair": {"static_friction": 0.3, "kinetic_friction": 0.3},
                "hair_body": {"static_friction": 0.3, "kinetic_friction": 0.3},
            },
            "kinematic_body_motion": {
                "object_ids": ["head", "neck", "shoulder_bar", "torso"],
                "type": "smooth_lateral_translation_then_hold",
                "axis": motion["axis"],
                "displacement": float(motion["displacement"]),
                "start_time": float(motion["start_time"]),
                "stop_time": float(motion["stop_time"]),
                "hold_until": float(motion["settle_end_time"]),
                "root_nodes_follow_head_motion": True,
            },
            "event_windows": {
                "loading_and_contact": [0.45, 1.55],
                "stick_slip_and_continuing_motion": [1.55, 2.35],
                "head_stopped_draped_outcome": [2.35, 4.8],
            },
        }

    def initial_state(self) -> StateFrame:
        pos = np.empty((STRAND_COUNT * NODES_PER_STRAND, 3), dtype=float)
        strand_id = np.repeat(np.arange(STRAND_COUNT, dtype=int), NODES_PER_STRAND)
        node_index = np.tile(np.arange(NODES_PER_STRAND, dtype=int), STRAND_COUNT)
        for sid in range(STRAND_COUNT):
            ring = sid % 14
            layer = sid // 14
            az = 2.0 * np.pi * ring / 14.0 + 0.055 * layer
            root_r = 0.185 + 0.010 * layer
            root_z = 1.64 + 0.18 * np.cos(0.78 * az)
            root = np.array([root_r * np.cos(az), root_r * np.sin(az), root_z])
            length = 0.62 + 0.055 * layer + 0.035 * np.sin(1.7 * az)
            s = np.linspace(0.0, 1.0, NODES_PER_STRAND)
            outward = np.array([np.cos(az), np.sin(az), 0.0])
            tangent = np.array([-np.sin(az), np.cos(az), 0.0])
            wave = (0.023 + 0.004 * layer) * np.sin(5.0 * np.pi * s + 0.73 * sid)
            curve = root + (0.045 * s)[:, None] * outward + wave[:, None] * tangent
            curve[:, 2] -= length * (0.84 * s + 0.16 * s * s)
            # Bias camera-side/back strands toward both shoulders so head motion
            # immediately produces body and neighboring-layer contacts.
            curve[:, 0] += 0.10 * np.sin(az) * s * s
            sl = slice(sid * NODES_PER_STRAND, (sid + 1) * NODES_PER_STRAND)
            pos[sl] = curve
        segment_length = 0.72 / (NODES_PER_STRAND - 1)
        mass = np.full(len(pos), 0.006 * segment_length)
        mass[node_index == 0] = 0.0
        frame = StateFrame(
            time=0.0,
            position=pos,
            velocity=np.zeros_like(pos),
            mass=mass,
            body_id=strand_id,
            attributes={
                "strand_id": strand_id.copy(),
                "node_index": node_index,
                "is_root": (node_index == 0).astype(int),
                "kinematic_root": (node_index == 0).astype(int),
            },
            scalars={
                "strand_radius": STRAND_RADIUS,
                "static_friction": 0.3,
                "kinetic_friction": 0.3,
                "head_stop_time": 2.35,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
