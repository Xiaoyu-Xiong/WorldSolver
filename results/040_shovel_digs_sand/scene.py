"""Deterministic, solver-neutral initial state for a shovel crossing a sand bed."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Create a level granular bed; scripted shovel motion is declared in config.yaml."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["seed"])

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.seed,
            "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
            "phenomenon": "kinematic shovel displacement, lateral granular flow, and stable mound deposition",
            "kinematic_motion": self.config["kinematic_motion"],
            "diagnostics": self.config["diagnostics"],
        }

    def initial_state(self) -> StateFrame:
        initial = self.config["initial_conditions"]
        spacing = float(initial["sand_particle_spacing"])
        lo = np.asarray(initial["sand_bounds_min"], dtype=float)
        hi = np.asarray(initial["sand_bounds_max"], dtype=float)
        axes = [np.arange(lo[i] + 0.5 * spacing, hi[i], spacing) for i in range(3)]
        position = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        # Small seeded sub-cell offsets suppress a visibly crystalline bed without
        # altering its level free surface or deterministic bulk volume.
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(-0.08 * spacing, 0.08 * spacing, position.shape)
        jitter[:, 2] *= 0.35
        position += jitter

        particle_volume = spacing**3
        density = float(initial["sand_bulk_density"])
        mass = np.full(len(position), density * particle_volume)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            body_id=np.ones(len(position), dtype=np.int32),
            attributes={
                "initial_volume": np.full(len(position), particle_volume),
                "material_label": np.ones(len(position), dtype=np.int32),
            },
            scalars={
                "initial_sand_volume": float(len(position) * particle_volume),
                "initial_sand_mass": float(mass.sum()),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
