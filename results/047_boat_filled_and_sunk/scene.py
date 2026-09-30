"""Deterministic, solver-neutral initial state for the fill-and-sink scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "thin-shell boat filled by water, sinking, leakage, bottom impact",
            "coupled_materials": ["clear_water", "boat_red"],
            "interface_region": "water free surface and open boat hull",
            "scaffold_visual_intent": (
                "The red open hull, blue inlet stream, cyan internal water level, external water "
                "surface, and bottom impact must all be visible from the camera."
            ),
        }

    @staticmethod
    def _water_points() -> np.ndarray:
        spacing = 0.18
        x = np.arange(-2.16, 2.17, spacing)
        y = np.arange(-1.56, 1.57, spacing)
        z = np.arange(0.09, 1.57, spacing)
        pool = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        # A short initialized jet makes the transfer region readable at frame zero.
        jx = np.arange(-0.12, 0.13, 0.12)
        jy = np.arange(-0.12, 0.13, 0.12)
        jz = np.arange(2.48, 3.36, 0.12)
        jet = np.stack(np.meshgrid(jx, jy, jz, indexing="ij"), axis=-1).reshape(-1, 3)
        return np.vstack((pool, jet))

    def initial_state(self) -> StateFrame:
        water = self._water_points()
        boat_center = np.asarray(self.config["boat"]["initial_center"], dtype=float)[None, :]
        position = np.vstack((boat_center, water))
        velocity = np.zeros_like(position)
        velocity[1:, 2] = np.where(water[:, 2] > 2.4, -2.6, 0.0)
        mass = np.full(len(position), 1000.0 * 0.18**3)
        mass[0] = float(self.config["boat"]["mass"])
        body_id = np.ones(len(position), dtype=np.int32)
        body_id[0] = 0
        water_region = np.ones(len(position), dtype=np.int32)
        water_region[0] = 0
        water_region[1:][water[:, 2] > 2.4] = 3
        orientation = np.zeros((len(position), 4))
        orientation[:, 3] = 1.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros_like(position),
            mass=mass,
            body_id=body_id,
            attributes={"phase_id": np.ones(len(position), dtype=np.int32), "water_region": water_region},
            scalars={"initial_water_mass_in_hull": 0.0, "initial_floating_height": 1.82},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
