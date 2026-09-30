"""Deterministic, solver-neutral initial state for the 2D hourglass."""
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
            "events": self.config["events"],
            "observables": self.config["observables"],
        }

    def initial_state(self) -> StateFrame:
        count = int(self.config["initial_conditions"]["sand_particle_count"])
        radius = float(self.config["initial_conditions"]["particle_radius"])
        density = float(self.config["initial_conditions"]["particle_density"])
        rng = np.random.default_rng(int(self.config["seed"]))

        # A jittered thin-slab lattice provides a dense, readable charge in the upper bulb.
        spacing = 2.05 * radius
        candidates = []
        slab_half_depth = float(self.config["boundary_conditions"].get("slab_half_depth", 0.055))
        for z in np.arange(1.08 + radius, 1.54 - radius, spacing):
            half_width = min(0.39, 0.038 + 0.92 * (z - 0.72))
            for y in np.arange(-slab_half_depth + radius, slab_half_depth - radius, spacing):
                for x in np.arange(-half_width + radius, half_width - radius, spacing):
                    candidates.append((x, y, z))
        if len(candidates) < count:
            raise ValueError("Configured particle count exceeds the deterministic fill region")
        position = np.asarray(candidates[:count], dtype=float)
        position[:, 0] += rng.uniform(-0.10, 0.10, count) * radius
        position[:, 1] = rng.uniform(-0.35, 0.35, count) * radius
        position[:, 2] += rng.uniform(-0.08, 0.08, count) * radius
        velocity = np.zeros_like(position)
        grain_volume = 4.0 * np.pi * radius**3 / 3.0
        mass = np.full(count, density * grain_volume, dtype=float)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"particle_radius": np.full(count, radius)},
            scalars={"initial_bulk_particle_volume": count * grain_volume},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
