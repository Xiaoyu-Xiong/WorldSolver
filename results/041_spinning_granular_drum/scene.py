"""Deterministic scaffold for a spinning granular drum scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial samples and frozen timing for a half-filled rotating drum."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["seed"])

    def metadata(self) -> dict:
        drum = self.config["initial_conditions"]["drum"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.seed,
            "units": "meters",
            "up_axis": "z",
            "gravity": list(self.config.get("gravity", [0.0, 0.0, -9.81])),
            "phenomenon": "half_filled_spinning_granular_drum_avalanche",
            "source": "Hybrid Grains Fig.17, Section 5.8",
            "drum_angular_velocity_rad_s": float(drum["angular_velocity_rad_s"]),
            "scaffold_visual_intent": (
                "A dark rotating drum with bright rim markers lifts tan grains up the wall; "
                "tracer grains make the avalanche layer and non-rigid granular motion legible."
            ),
        }

    def initial_state(self) -> StateFrame:
        initial = self.config["initial_conditions"]
        drum = initial["drum"]
        grains = initial["grains"]
        center = np.asarray(drum["center"], dtype=np.float64)
        radius = float(drum["radius"])
        depth = float(drum["depth"])
        grain_radius = float(grains["radius"])
        spacing = float(grains["spacing"])

        # Drum axis is y; the lower half of the circular x-z cross-section is filled.
        xs = np.arange(-radius + grain_radius, radius - grain_radius + 1e-12, spacing)
        ys = np.arange(-0.5 * depth + grain_radius, 0.5 * depth - grain_radius + 1e-12, spacing)
        zs = np.arange(-radius + grain_radius, -0.03, spacing)
        samples = []
        for layer_index, y in enumerate(ys):
            offset = 0.5 * spacing if layer_index % 2 else 0.0
            for z in zs:
                for x in xs + offset:
                    if x * x + z * z <= (radius - grain_radius) ** 2:
                        samples.append((x, y, z))

        position = np.asarray(samples, dtype=np.float64)
        position += center
        count = len(position)
        density = float(grains["bulk_density"])
        particle_volume = (4.0 * np.pi * grain_radius**3) / 3.0
        rng = np.random.default_rng(self.seed)
        tracer = (rng.random(count) < float(grains.get("tracer_fraction", 0.06))).astype(np.float64)

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros((count, 3), dtype=np.float64),
            mass=np.full(count, density * particle_volume, dtype=np.float64),
            body_id=np.ones(count, dtype=np.int32),
            attributes={
                "radius": np.full(count, grain_radius, dtype=np.float64),
                "particle_radius": np.full(count, grain_radius, dtype=np.float64),
                "material_label": np.ones(count, dtype=np.int32),
                "tracer_grain": tracer,
                "initial_volume": np.full(count, particle_volume, dtype=np.float64),
            },
            scalars={
                "drum_radius": radius,
                "drum_depth": depth,
                "drum_angular_velocity_rad_s": float(drum["angular_velocity_rad_s"]),
                "initial_particle_count": float(count),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        steps = int(round(duration / dt))
        return (np.arange(steps + 1, dtype=np.float64) * dt).tolist()
