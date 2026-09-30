"""Deterministic, solver-neutral rising-smoke scene definition."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        domain = self.config["domain"]
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "phenomenon": "buoyant smoke rising and rolling into persistent vortices",
                "domain_bounds": [domain["bounds_min"], domain["bounds_max"]],
                "grid_shape": domain["grid_shape"], "field_layout": "cell_centered_xyz",
                "required_fields": ["density", "velocity"]}

    def initial_state(self) -> StateFrame:
        domain = self.config["domain"]
        initial = self.config["initial_conditions"]
        source = initial["smoke_source"]
        shape = tuple(int(v) for v in domain["grid_shape"])
        low = np.asarray(domain["bounds_min"], dtype=np.float64)
        high = np.asarray(domain["bounds_max"], dtype=np.float64)
        spacing = (high - low) / np.asarray(shape, dtype=np.float64)
        axes = [low[i] + (np.arange(shape[i]) + 0.5) * spacing[i] for i in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        center = np.asarray(source["center"], dtype=np.float64)
        radial2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) / float(source["radius"]) ** 2
        vertical2 = (z - center[2]) ** 2 / float(source["half_height"]) ** 2
        density = float(source["peak_density"]) * np.exp(-2.5 * (radial2 + vertical2))
        density[density < 1.0e-4] = 0.0
        velocity = np.zeros(shape + (3,), dtype=np.float64)
        source_velocity = np.asarray(source["initial_velocity"], dtype=np.float64)
        envelope = np.clip(density / float(source["peak_density"]), 0.0, 1.0)
        velocity += envelope[..., None] * source_velocity
        amplitude = float(initial["perturbation_amplitude"])
        phase = (int(self.config["seed"]) % 360) * np.pi / 180.0
        velocity[..., 0] += amplitude * envelope * np.sin(2.0 * np.pi * y / (high[1] - low[1]) + phase)
        velocity[..., 1] += amplitude * envelope * np.sin(2.0 * np.pi * x / (high[0] - low[0]) - phase)
        frame = StateFrame(0.0, center.reshape(1, 3).copy(), source_velocity.reshape(1, 3).copy(),
                           mass=np.asarray([1.0]),
                           attributes={"density": np.asarray([source["peak_density"]]),
                                       "temperature_excess": np.asarray([source["temperature_excess"]])},
                           fields={"density": density, "velocity": velocity},
                           scalars={"ambient_density": float(initial["ambient_density"])})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(int(round(duration / dt)) + 1, dtype=np.float64) * dt).tolist()
