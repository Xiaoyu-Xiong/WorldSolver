"""Deterministic, solver-neutral scene contract for an inflated rubber tube."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """A finite-thickness hollow tube with fixed rims and cavity pressure data."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        bc = self.config["boundary_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "inflation_of_end_constrained_rubber_tube",
            "spatial_dimensions": 3,
            "geometry": "closed finite-thickness hollow volumetric wall",
            "fixed_rim_attribute": bc["fixed_rims"]["node_attribute"],
            "pressure_surface_attribute": bc["cavity_pressure"]["surface_attribute"],
            "pressure_pa": float(bc["cavity_pressure"]["pressure_pa"]),
        }

    def initial_state(self) -> StateFrame:
        g = self.config["geometry"]
        n_theta = int(g["circumferential_segments"])
        n_z = int(g["axial_segments"])
        inner = float(g["inner_radius"])
        outer = inner + float(g["wall_thickness"])
        half_height = 0.5 * float(g["height"])

        theta = 2.0 * np.pi * np.arange(n_theta, dtype=np.float64) / n_theta
        z = np.linspace(-half_height, half_height, n_z + 1, dtype=np.float64)
        tt, zz = np.meshgrid(theta, z, indexing="xy")
        def layer(radius: float) -> np.ndarray:
            return np.column_stack((radius * np.cos(tt).ravel(),
                                    radius * np.sin(tt).ravel(), zz.ravel()))
        position = np.vstack((layer(outer), layer(inner)))
        count = position.shape[0]
        surface_size = n_theta * (n_z + 1)
        fixed = np.isclose(np.abs(position[:, 2]), half_height).astype(np.int32)
        pressure_surface = np.zeros(count, dtype=np.int32)
        pressure_surface[surface_size:] = 1
        region = np.concatenate((np.zeros(surface_size, dtype=np.int32),
                                 np.ones(surface_size, dtype=np.int32)))

        density = float(self.config["material"]["density"])
        wall_volume = np.pi * (outer**2 - inner**2) * (2.0 * half_height)
        mass = np.full(count, density * wall_volume / count, dtype=np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "fixed_rim": fixed,
                "pressure_surface": pressure_surface,
                "surface_region": region,
                "rest_position": position.copy(),
            },
            scalars={
                "cavity_pressure_pa": float(self.config["boundary_conditions"]["cavity_pressure"]["pressure_pa"]),
                "inner_radius": inner,
                "outer_radius": outer,
                "wall_thickness": outer - inner,
            },
        )
        frame.validate()
        return frame

    def pressure_at(self, time: float) -> float:
        """Prescribed load history; evolution of the solid remains solver-owned."""
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and nonnegative")
        p = self.config["boundary_conditions"]["cavity_pressure"]
        start, end = float(p["ramp_start"]), float(p["ramp_end"])
        fraction = np.clip((time - start) / (end - start), 0.0, 1.0)
        return float(p["pressure_pa"]) * float(fraction)

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return (np.arange(int(round(duration / dt)) + 1, dtype=np.float64) * dt).tolist()
