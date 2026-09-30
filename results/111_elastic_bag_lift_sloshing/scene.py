"""Deterministic solver-neutral lifted elastic-bag scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial state and controls for two-way cloth/liquid coupling."""

    def __init__(self, config: dict):
        self.config = config
        asset = np.load(Path(__file__).resolve().parent / "assets" / "elastic_bag.npz")
        source_vertices = np.asarray(asset["vertices"], dtype=float)
        self._bag_faces = np.asarray(asset["faces"], dtype=np.int32)
        self._rim_indices = np.asarray(asset["rim_indices"], dtype=np.int32)
        self._bag_vertices = source_vertices.copy()

    def metadata(self) -> dict:
        lift = self.config["boundary_conditions"]["rim_lift"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "up_axis": "z",
            "phenomenon": "two-way coupled lifting, bouncing, sloshing, and spill from an elastic bag",
            "coupled_bodies": {"elastic_bag": 0, "water": 1},
            "topology": {"bag_faces": self._bag_faces.tolist()},
            "controlled_vertex_sets": {"rim": self._rim_indices.tolist()},
            "rim_control": dict(lift),
            "coupling_requirement": (
                "Only rim vertices follow the lift control. Fluid traction changes all other shell "
                "motion, and the evolving impermeable shell changes the resolved interior and exterior flow."
            ),
        }

    @staticmethod
    def _pool_points(low: list[float], high: list[float], surface_z: float, spacing: float) -> np.ndarray:
        axes = [
            np.arange(low[0] + 0.5 * spacing, high[0], spacing),
            np.arange(low[1] + 0.5 * spacing, high[1], spacing),
            np.arange(low[2] + 0.5 * spacing, surface_z, spacing),
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([component.ravel() for component in grid])

    def initial_state(self) -> StateFrame:
        bag = self._bag_vertices.copy()
        pool_cfg = self.config["initial_conditions"]["pool"]
        tank_cfg = self.config["boundary_conditions"]["tank"]
        spacing = float(pool_cfg["particle_spacing"])
        water = self._pool_points(
            tank_cfg["interior_min"], tank_cfg["interior_max"],
            float(pool_cfg["free_surface_z"]), spacing,
        )
        position = np.vstack((bag, water))
        velocity = np.zeros_like(position)
        body_id = np.r_[np.zeros(len(bag), dtype=np.int32), np.ones(len(water), dtype=np.int32)]

        bag_mass = float(self.config["initial_conditions"]["bag"]["total_mass"])
        density = float(pool_cfg["rest_density"])
        mass = np.r_[np.full(len(bag), bag_mass / len(bag)), np.full(len(water), density * spacing**3)]
        controlled = np.zeros(len(position), dtype=float)
        controlled[self._rim_indices] = 1.0
        particle_radius = np.r_[np.full(len(bag), 0.008), np.full(len(water), 0.55 * spacing)]

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "rim_controlled": controlled,
                "material_phase": body_id.astype(float),
                "particle_radius": particle_radius,
            },
            scalars={
                "water_rest_density": density,
                "bag_total_mass": bag_mass,
                "bag_thickness": float(self.config["initial_conditions"]["bag"]["thickness"]),
                "initial_free_surface_z": float(pool_cfg["free_surface_z"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1, dtype=float) * dt).tolist()
