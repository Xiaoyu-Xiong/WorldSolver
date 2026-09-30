"""Deterministic, solver-neutral initial scene for the 2D dam break."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """A quiescent liquid column in an open-top, closed-sided tank."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "spatial_dimensions": 2,
            "gravity": [0.0, 0.0, -9.81],
            "initial_condition": "rectangular liquid column at rest",
            "boundary_condition": "closed floor and side walls; open air above",
        }

    def initial_state(self) -> StateFrame:
        # Cell-centred regular sampling makes the StateFrame reproducible and
        # leaves a clear gap from collision boundaries. The constant y plane
        # is the solver-neutral representation of a strictly 2D liquid.
        nx, nz = 25, 40
        x_edges = np.linspace(-2.75, -1.25, nx + 1, dtype=np.float64)
        z_edges = np.linspace(0.10, 2.50, nz + 1, dtype=np.float64)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        z = 0.5 * (z_edges[:-1] + z_edges[1:])
        xx, zz = np.meshgrid(x, z, indexing="xy")
        position = np.column_stack(
            (xx.ravel(), np.zeros(xx.size, dtype=np.float64), zz.ravel())
        )
        velocity = np.zeros_like(position)
        dx, dz = x_edges[1] - x_edges[0], z_edges[1] - z_edges[0]
        mass = np.full(position.shape[0], 1000.0 * dx * dz, dtype=np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"sample_volume": np.full(position.shape[0], dx * dz)},
            scalars={"density": 1000.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=np.float64) * dt).tolist()
