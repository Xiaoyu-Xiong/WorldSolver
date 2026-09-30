"""Deterministic, solver-neutral pouring-sand scene contract."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial granular reservoir and established stream shown in Figure 9."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        initial = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "units": "meters",
            "up_axis": "z",
            "gravity": list(self.config["gravity"]),
            "particle_spacing": float(initial["particle_spacing"]),
            "nominal_grain_radius": float(initial["nominal_grain_radius"]),
            "phenomenon": "gravity-driven discharge, pile growth, runout, and repose",
        }

    @staticmethod
    def _lattice_in_cylinder(spacing: float, radius: float, z0: float, z1: float) -> np.ndarray:
        xs = np.arange(-radius + spacing / 2, radius, spacing)
        ys = np.arange(-radius + spacing / 2, radius, spacing)
        zs = np.arange(z0 + spacing / 2, z1, spacing)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid[np.sum(grid[:, :2] ** 2, axis=1) <= (radius - spacing / 2) ** 2]

    def initial_state(self) -> StateFrame:
        initial = self.config["initial_conditions"]
        h = float(initial["particle_spacing"])

        # Broad upper reservoir plus conical funnel fill and a pre-existing thin stream.
        reservoir = self._lattice_in_cylinder(h, 0.312, 1.68, 2.25)
        funnel_box = self._lattice_in_cylinder(h, 0.30, 1.47, 1.73)
        allowed_radius = 0.065 + (funnel_box[:, 2] - 1.51) * (0.24 / 0.22)
        funnel = funnel_box[np.linalg.norm(funnel_box[:, :2], axis=1) <= allowed_radius]
        stream = self._lattice_in_cylinder(h, 0.060, 0.56, 1.51)
        position = np.concatenate((reservoir, funnel, stream), axis=0)

        # Seeded sub-spacing offsets avoid an artificially perfect granular lattice.
        rng = np.random.default_rng(int(self.config["seed"]))
        position += rng.uniform(-0.08 * h, 0.08 * h, size=position.shape)
        velocity = np.zeros_like(position)
        stream_start = len(reservoir) + len(funnel)
        velocity[stream_start:, 2] = -0.55

        particle_volume = h ** 3
        mass = np.full(len(position), float(initial["bulk_density"]) * particle_volume)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "nominal_radius": np.full(len(position), float(initial["nominal_grain_radius"])),
                "material_phase": np.zeros(len(position), dtype=float),
            },
            scalars={"initial_total_mass": float(np.sum(mass))},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
