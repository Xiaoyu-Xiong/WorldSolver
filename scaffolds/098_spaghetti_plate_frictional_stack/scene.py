"""Deterministic, solver-neutral spaghetti packing benchmark scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


STRAND_COUNT = 31
NODES_PER_STRAND = 16
STRAND_LENGTH = 0.72
STRAND_RADIUS = 0.006


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "frictional_stacking_of_flexible_rods",
            "strand_count": STRAND_COUNT,
            "nodes_per_strand": NODES_PER_STRAND,
            "strand_length": STRAND_LENGTH,
            "strand_radius": STRAND_RADIUS,
            "gravity": [0.0, 0.0, -9.81],
            "free_ends": True,
            "initial_velocity": [0.0, 0.0, 0.0],
            "rest_shape": "straight",
            "bending_resistance": True,
            "contact": {
                "strand_strand": {"static_friction": 0.3, "kinetic_friction": 0.3},
                "strand_plate": {"static_friction": 0.3, "kinetic_friction": 0.3},
            },
            "plate": {
                "motion": "static",
                "shape": "shallow_axisymmetric_container_with_raised_rim",
            },
        }

    def initial_state(self) -> StateFrame:
        # A compact 7 x 5 staggered bundle (four corner slots omitted), centered
        # over the plate. Spacing exceeds the 12 mm strand diameter.
        slots = []
        for row in range(5):
            for col in range(7):
                if (row, col) in {(0, 0), (0, 6), (4, 0), (4, 6)}:
                    continue
                y = (col - 3) * 0.014
                z = 0.62 + row * 0.014
                slots.append((y, z))
        x = np.linspace(-0.5 * STRAND_LENGTH, 0.5 * STRAND_LENGTH, NODES_PER_STRAND)
        position = np.empty((STRAND_COUNT * NODES_PER_STRAND, 3), dtype=float)
        body_id = np.repeat(np.arange(STRAND_COUNT, dtype=int), NODES_PER_STRAND)
        node_index = np.tile(np.arange(NODES_PER_STRAND, dtype=int), STRAND_COUNT)
        for strand_id, (y, z) in enumerate(slots):
            sl = slice(strand_id * NODES_PER_STRAND, (strand_id + 1) * NODES_PER_STRAND)
            position[sl, 0] = x
            position[sl, 1] = y
            position[sl, 2] = z
        segment_length = STRAND_LENGTH / (NODES_PER_STRAND - 1)
        linear_density = 0.010
        mass = np.full(len(position), linear_density * segment_length, dtype=float)
        mass[node_index == 0] *= 0.5
        mass[node_index == NODES_PER_STRAND - 1] *= 0.5
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            body_id=body_id,
            attributes={
                "strand_id": body_id.copy(),
                "node_index": node_index,
                "is_endpoint": ((node_index == 0) | (node_index == NODES_PER_STRAND - 1)).astype(int),
            },
            scalars={
                "strand_radius": STRAND_RADIUS,
                "strand_length": STRAND_LENGTH,
                "static_friction": 0.3,
                "kinetic_friction": 0.3,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()

