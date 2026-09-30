"""Deterministic solver-neutral foam pull-apart scene for Figure 7."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "units": "meters",
            "up_axis": "z",
            "gravity": self.config["gravity"],
            "phenomenon": "foam_tendril_pull_apart_with_tearing_enabled",
            "tearing_intent": "A necked bridge must separate into two fragments instead of persisting as a nonphysical thread.",
            "scaffold_visual_intent": (
                "Render as bright white shaving foam between a dark upper plate and gray floor; "
                "the hourglass-like silhouette and final dark tear gap must be visually legible."
            ),
        }

    @staticmethod
    def _radius(z: np.ndarray) -> np.ndarray:
        """Axisymmetric Figure-7 silhouette: lower dollop, neck, upper adhered cap."""
        lower = 0.08 + 0.48 * np.sin(np.pi * np.clip(z / 0.72, 0.0, 1.0)) ** 0.72
        neck = 0.075 + 0.055 * np.abs(z - 0.93) / 0.23
        upper = 0.10 + 0.47 * np.sin(np.pi * np.clip((z - 1.15) / 0.69, 0.0, 1.0)) ** 0.62
        return np.where(z <= 0.70, lower, np.where(z < 1.16, neck, upper))

    def initial_state(self) -> StateFrame:
        spacing = float(self.config["initial_conditions"]["particle_spacing"])
        xs = np.arange(-0.60, 0.6001, spacing)
        ys = np.arange(-0.54, 0.5401, spacing)
        zs = np.arange(0.035, 1.8201, spacing)
        x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
        radius = self._radius(z)
        # Slight fore-aft compression reproduces the compact photographic dollop.
        inside = x * x + (y / 0.88) ** 2 <= radius * radius
        position = np.column_stack((x[inside], y[inside], z[inside])).astype(np.float64)

        velocity = np.zeros_like(position)
        blend = np.clip((position[:, 2] - 0.68) / 0.58, 0.0, 1.0)
        velocity[:, 2] = (-0.03 * (1.0 - blend) + 0.20 * blend)

        density = float(self.config["initial_conditions"]["bulk_density"])
        mass = np.full(len(position), density * spacing**3, dtype=np.float64)
        neck = ((position[:, 2] >= 0.78) & (position[:, 2] <= 1.12)).astype(np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=np.zeros(len(position), dtype=np.int32),
            attributes={
                "particle_radius": np.full(len(position), 0.026),
                "foam_material": np.ones(len(position)),
                "tear_candidate": neck,
            },
            scalars={"explicit_tearing_enabled": 1.0, "initial_particle_count": float(len(position))},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=np.float64) * dt).tolist()
