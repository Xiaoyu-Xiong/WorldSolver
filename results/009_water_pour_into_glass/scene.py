"""Deterministic solver-neutral setup for water poured into a clear glass."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _lattice(bounds: list[tuple[float, float]], spacing: float) -> np.ndarray:
    axes = [
        np.arange(lo + 0.5 * spacing, hi, spacing, dtype=np.float64)
        for lo, hi in bounds
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([component.ravel() for component in grid])


class BenchmarkScene:
    """A shallow fill and continuous oblique jet, matching paper Figures 1 and 7."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        ic = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "units": "meters",
            "up_axis": "z",
            "phenomenon": "turbulent water-air mixing during pouring",
            "gravity": self.config["gravity"],
            "reference_grid": self.config["domain"]["reference_grid"],
            "liquid_density": ic["liquid_density"],
            "source": {
                "mode": "sustained_prescribed_inflow",
                "start": ic["jet_start"],
                "end": ic["jet_end"],
                "radius": ic["jet_radius"],
                "speed": ic["jet_speed"],
                "active_interval": [ic["source_start_time"], ic["source_end_time"]],
                "target_flow_rate_m3_s": ic["target_inflow_rate_m3_s"],
            },
            "volume_accounting": "initial volume plus integrated source flux",
            "air_phase": "ambient open air; entrainment occurs at the resolved free surface",
        }

    def initial_state(self) -> StateFrame:
        ic = self.config["initial_conditions"]
        h = float(ic["particle_spacing"])
        radius = float(ic["glass_inner_radius"])

        pool = _lattice(
            [(-radius, radius), (-radius, radius), (0.018, float(ic["initial_water_depth"]))],
            h,
        )
        pool = pool[np.sum(pool[:, :2] ** 2, axis=1) <= (radius - 0.55 * h) ** 2]

        start = np.asarray(ic["jet_start"], dtype=np.float64)
        end = np.asarray(ic["jet_end"], dtype=np.float64)
        direction = (end - start) / np.linalg.norm(end - start)
        length = np.linalg.norm(end - start)
        axial = np.arange(0.5 * h, length, h, dtype=np.float64)
        # y and a perpendicular vector span each circular jet cross-section.
        side = np.array([0.0, 1.0, 0.0])
        cross = np.cross(direction, side)
        cross /= np.linalg.norm(cross)
        offsets = np.arange(-ic["jet_radius"] + 0.5 * h, ic["jet_radius"], h)
        disk = [(a, b) for a in offsets for b in offsets
                if a * a + b * b <= (ic["jet_radius"] - 0.2 * h) ** 2]
        jet = np.vstack([
            start + s * direction + a * side + b * cross
            for s in axial for a, b in disk
        ])

        position = np.vstack((pool, jet))
        velocity = np.zeros_like(position)
        velocity[len(pool):] = float(ic["jet_speed"]) * direction
        density = float(ic["liquid_density"])
        mass = np.full(len(position), density * h ** 3, dtype=np.float64)
        source_region = np.concatenate((np.zeros(len(pool)), np.ones(len(jet))))
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"source_region": source_region},
            scalars={
                "rest_density": density,
                "particle_spacing": h,
                "initial_liquid_volume": float(ic["initial_liquid_volume_m3"]),
                "target_inflow_rate": float(ic["target_inflow_rate_m3_s"]),
                "source_end_time": float(ic["source_end_time"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return (np.arange(int(round(duration / dt)) + 1, dtype=np.float64) * dt).tolist()
