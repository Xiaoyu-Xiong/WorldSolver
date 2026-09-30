"""Deterministic solver-neutral scene for frictionless spaghetti spreading."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "zero-friction packing and lateral spreading of flexible rods",
            "gravity": self.config["gravity"],
            "strand_count": self.config["strand_count"],
            "strand": self.config["strand"],
            "plate": self.config["plate"],
            "contact": self.config["contact"],
            "initial_conditions": self.config["initial_conditions"],
            "event_windows": self.config["event_windows"],
            "required_outcome": self.config["required_outcome"],
        }

    def initial_state(self) -> StateFrame:
        strand = self.config["strand"]
        count = int(self.config["strand_count"])
        points_per_strand = int(strand["vertex_count"])
        length = float(strand["length"])
        radius = float(strand["radius"])
        spacing = float(self.config["initial_conditions"]["center_spacing"])
        center = np.asarray(self.config["initial_conditions"]["center"], dtype=float)

        # A compact 6-row hexagonal bundle (6+5+5+5+5+5 = 31). Every
        # centerline is straight and parallel to x; spacing exceeds diameter.
        row_counts = [6, 5, 5, 5, 5, 5]
        centers = []
        for row, row_count in enumerate(row_counts):
            y_values = (np.arange(row_count) - 0.5 * (row_count - 1)) * spacing
            z = (row - 2.5) * spacing * np.sqrt(3.0) / 2.0
            centers.extend((float(y), float(z)) for y in y_values)
        if len(centers) != count:
            raise ValueError("authored bundle layout must contain exactly 31 strands")

        x = np.linspace(-0.5 * length, 0.5 * length, points_per_strand)
        positions = []
        strand_ids = []
        arc_fractions = []
        endpoint_flags = []
        for strand_id, (y, z) in enumerate(centers):
            curve = np.column_stack((x, np.full_like(x, y), np.full_like(x, z))) + center
            positions.append(curve)
            strand_ids.extend([strand_id] * points_per_strand)
            arc_fractions.extend(np.linspace(0.0, 1.0, points_per_strand))
            flags = np.zeros(points_per_strand)
            flags[[0, -1]] = 1.0
            endpoint_flags.extend(flags)

        position = np.vstack(positions)
        n = len(position)
        mass_each = float(strand["total_mass_each"])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=np.full(n, mass_each / points_per_strand),
            body_id=np.asarray(strand_ids, dtype=int),
            attributes={
                "strand_id": np.asarray(strand_ids, dtype=float),
                "arc_fraction": np.asarray(arc_fractions),
                "rod_radius": np.full(n, radius),
                "contact_diameter": np.full(n, 2.0 * radius),
                "free_endpoint": np.asarray(endpoint_flags),
            },
            scalars={
                "strand_strand_static_friction": 0.0,
                "strand_strand_kinetic_friction": 0.0,
                "strand_plate_static_friction": 0.0,
                "strand_plate_kinetic_friction": 0.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
