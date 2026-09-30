"""Deterministic solver-neutral initial state for the buoyant smoke box."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        initial = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "closed-box buoyant smoke plume",
            "power_particle_count": initial["power_particle_count"],
            "passive_marker_count": initial["passive_marker_count"],
            "reference_passive_marker_count": initial["reference_passive_marker_count"],
            "boundary_condition": self.config["domain"]["boundary_condition"],
        }

    @staticmethod
    def _source_markers(count: int, seed: int, radius: float, height: float) -> np.ndarray:
        """Low-discrepancy cylindrical seed: uniform, reproducible, and unclumped."""
        i = np.arange(count, dtype=np.float64) + 0.5
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        turns = (i * golden_angle + (seed % 997) * 0.001) % (2.0 * np.pi)
        radial = radius * np.sqrt(((i * 0.7548776662466927) % 1.0))
        z = 0.035 + height * ((i * 0.5698402909980532) % 1.0)
        return np.column_stack((radial * np.cos(turns), radial * np.sin(turns), z))

    def initial_state(self) -> StateFrame:
        initial = self.config["initial_conditions"]
        count = int(initial["passive_marker_count"])
        position = self._source_markers(
            count,
            int(self.config["seed"]),
            float(initial["source_radius"]),
            float(initial["source_height"]),
        )
        velocity = np.zeros_like(position)
        velocity[:] = np.asarray(initial["initial_velocity"], dtype=np.float64)

        shape = tuple(int(v) for v in self.config["field_layout"]["grid_shape"])
        bounds_min = np.asarray(self.config["domain"]["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(self.config["domain"]["bounds_max"], dtype=np.float64)
        axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(bounds_min, bounds_max, shape)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        r2 = x * x + y * y
        smoke_density = np.exp(-r2 / (2.0 * 0.18**2)) * np.exp(-((z - 0.13) ** 2) / (2.0 * 0.10**2))
        smoke_density[z > 0.34] = 0.0
        temperature = np.exp(-r2 / (2.0 * 0.20**2)) * np.exp(-((z - 0.10) ** 2) / (2.0 * 0.09**2))
        velocity_field = np.zeros(shape + (3,), dtype=np.float64)

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=np.full(count, 1.0 / count, dtype=np.float64),
            attributes={"marker_kind": np.ones(count, dtype=np.int32)},
            fields={
                "smoke_density": smoke_density.astype(np.float32),
                "velocity_field": velocity_field,
                "temperature": temperature.astype(np.float32),
            },
            scalars={"divergence_l2": 0.0, "divergence_max": 0.0, "relative_volume_error": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        steps = int(round(duration / dt))
        return (np.arange(steps + 1, dtype=np.float64) * dt).tolist()
