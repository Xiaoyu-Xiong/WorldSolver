"""Deterministic solver-neutral elastic-bag and water initial scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial samples and declarative controls for two-way cloth/liquid coupling."""

    def __init__(self, config: dict):
        self.config = config
        asset = np.load(Path(__file__).resolve().parent / "assets" / "elastic_bag.npz")
        self._bag_vertices = np.asarray(asset["vertices"], dtype=float)
        self._bag_faces = np.asarray(asset["faces"], dtype=np.int32)
        self._rim_indices = np.asarray(asset["rim_indices"], dtype=np.int32)

    def metadata(self) -> dict:
        inflow = self.config["boundary_conditions"]["nozzle_inflow"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "up_axis": "z",
            "phenomenon": "two-way coupled filling and exterior overflow of an elastic cloth bag",
            "coupled_bodies": {"elastic_bag": 0, "water": 1},
            "topology": {"bag_faces": self._bag_faces.tolist()},
            "fixed_vertex_sets": {"rim": self._rim_indices.tolist()},
            "inflow": {
                "center": list(inflow["center"]),
                "direction": list(inflow["direction"]),
                "radius": float(inflow["radius"]),
                "speed": float(inflow["speed"]),
                "active_interval": list(inflow["active_interval"]),
            },
            "coupling_requirement": (
                "Fluid traction loads the impermeable dynamic shell, and the evolving shell "
                "is the moving boundary of the resolved liquid domain."
            ),
        }

    @staticmethod
    def _initial_jet(radius: float, z_min: float, z_max: float, spacing: float) -> np.ndarray:
        axis = np.arange(-radius + spacing / 2, radius, spacing)
        z = np.arange(z_min + spacing / 2, z_max, spacing)
        xx, yy, zz = np.meshgrid(axis, axis, z, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return points[points[:, 0] ** 2 + points[:, 1] ** 2 <= (radius - spacing / 3) ** 2]

    def initial_state(self) -> StateFrame:
        bag = self._bag_vertices.copy()
        water = self._initial_jet(radius=0.14, z_min=2.13, z_max=2.47, spacing=0.055)
        position = np.vstack((bag, water))
        velocity = np.zeros_like(position)
        velocity[len(bag):, 2] = -2.3
        body_id = np.r_[np.zeros(len(bag), dtype=np.int32), np.ones(len(water), dtype=np.int32)]

        bag_mass = float(self.config["initial_conditions"]["bag"]["total_mass"])
        water_density = float(self.config["initial_conditions"]["water"]["rest_density"])
        mass = np.r_[np.full(len(bag), bag_mass / len(bag)),
                     np.full(len(water), water_density * 0.055 ** 3)]
        pinned = np.zeros(len(position), dtype=float)
        pinned[self._rim_indices] = 1.0
        phase = body_id.astype(float)

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={"pinned": pinned, "material_phase": phase},
            scalars={
                "water_rest_density": water_density,
                "bag_total_mass": bag_mass,
                "bag_thickness": float(self.config["initial_conditions"]["bag"]["thickness"]),
                "nozzle_radius": 0.14,
                "nominal_inflow_speed": 2.3,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1, dtype=float) * dt).tolist()
