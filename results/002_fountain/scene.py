"""Deterministic, solver-neutral initial state for the fountain benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """A calm circular pool with a centered, already-rising free-surface jet."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        ic = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "units": "meters",
            "up_axis": "z",
            "phenomenon": "free_surface_fountain",
            "gravity": self.config["gravity"],
            "liquid_density": ic["liquid_density"],
            "source": {
                "center": [0.0, 0.0, ic["nozzle_exit_z"]],
                "radius": ic["nozzle_radius"],
                "speed": ic["jet_speed"],
                "mode": "sustained_upward_inflow",
            },
        }

    @staticmethod
    def _lattice(bounds, spacing):
        axes = [
            np.arange(lo + 0.5 * spacing, hi, spacing, dtype=np.float64)
            for lo, hi in bounds
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([axis.ravel() for axis in grid])

    def initial_state(self) -> StateFrame:
        ic = self.config["initial_conditions"]
        h = float(ic["particle_spacing"])
        pool_radius = float(ic["pool_radius"])
        surface_z = float(ic["pool_surface_z"])
        pool = self._lattice(
            [(-pool_radius, pool_radius), (-pool_radius, pool_radius),
             (0.04, surface_z)], h
        )
        pool = pool[np.sum(pool[:, :2] ** 2, axis=1) <= (pool_radius - 0.5 * h) ** 2]

        jet_radius = float(ic["nozzle_radius"])
        exit_z = float(ic["nozzle_exit_z"])
        jet_top = exit_z + float(ic["jet_height"])
        jet = self._lattice(
            [(-jet_radius, jet_radius), (-jet_radius, jet_radius),
             (exit_z, jet_top)], h
        )
        jet = jet[np.sum(jet[:, :2] ** 2, axis=1) <= (jet_radius - 0.25 * h) ** 2]

        position = np.vstack((pool, jet))
        velocity = np.zeros_like(position)
        velocity[len(pool):, 2] = float(ic["jet_speed"])
        density = float(ic["liquid_density"])
        mass = np.full(len(position), density * h ** 3, dtype=np.float64)
        phase = np.concatenate((
            np.zeros(len(pool), dtype=np.float64),
            np.ones(len(jet), dtype=np.float64),
        ))
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"source_region": phase},
            scalars={"rest_density": density, "particle_spacing": h},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=np.float64) * dt).tolist()
