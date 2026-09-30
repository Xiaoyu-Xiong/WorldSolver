"""Deterministic, solver-neutral initial scene for the octagonal two-phase drop."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _ellipsoid_lattice(center, radii, spacing):
    """Return a stable Cartesian sampling strictly inside an ellipsoid."""
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    axes = [np.arange(-r + spacing / 2, r, spacing) for r in radii]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid[np.sum((grid / radii) ** 2, axis=1) <= 1.0]
    return points + center


def _even_subset(points, count):
    """Deterministically retain spatially distributed samples in source order."""
    indices = np.linspace(0, len(points) - 1, count, dtype=int)
    return points[indices]


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "staged incompressible two-phase drop and buoyant separation",
            "phase_attribute": "phase_id",
            "density_attribute": "density",
            "release_time_attribute": "release_time",
            "target_volume_attribute": "target_volume",
            "container": "open octagonal basin",
            "density_ratio_heavy_to_light": 4.0,
        }

    def initial_state(self) -> StateFrame:
        # Sample counts retain the paper's 130:70 phase ratio at scaffold scale.
        blue = _even_subset(
            _ellipsoid_lattice((0.0, 0.0, 2.15), (0.43, 0.43, 0.58), 0.065), 1300
        )
        pink = _even_subset(
            _ellipsoid_lattice((0.05, 0.0, 3.62), (0.37, 0.37, 0.46), 0.065), 700
        )
        position = np.vstack((blue, pink))
        n_blue, n_pink = len(blue), len(pink)
        phase_id = np.concatenate((np.zeros(n_blue), np.ones(n_pink)))
        density = np.where(phase_id == 0, 1000.0, 250.0)
        release_time = np.where(phase_id == 0, 0.0, 1.35)
        phase_volume = np.where(phase_id == 0, 0.36 / n_blue, 0.20 / n_pink)
        velocity = np.zeros_like(position)
        velocity[:n_blue, 2] = -0.35
        mass = density * phase_volume
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=phase_id.astype(np.int32),
            attributes={
                "phase_id": phase_id,
                "density": density,
                "release_time": release_time,
                "target_volume": phase_volume,
            },
            scalars={
                "gravity_z": -9.81,
                "heavy_phase_volume": 0.36,
                "light_phase_volume": 0.20,
                "light_release_time": 1.35,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
