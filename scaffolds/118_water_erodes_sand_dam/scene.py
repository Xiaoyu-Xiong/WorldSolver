"""Deterministic, solver-neutral initial state for a coupled sand-dam breach."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    SAND_ID = 1
    WATER_ID = 2

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "materials": ["cohesive_porous_sand", "water"],
            "flow_direction": "+x",
            "initial_breach": False,
            "scripted_failure": False,
            "sand_mass_deletion_allowed": False,
            "required_coupling": {
                "water_to_sand": "fluid forces alter sand deformation and transport",
                "sand_to_water": "evolving porous skeleton and breach alter water flow",
                "wetting_response": "sand cohesion responds to simulated saturation",
            },
            "attributes": {
                "material_id": "1 sand, 2 water",
                "sand_saturation": "dynamic state: 0 dry to 1 saturated",
                "porosity": "initial solid pore fraction descriptor",
            },
        }

    @staticmethod
    def _lattice(bounds_min, bounds_max, spacing):
        lo, hi = np.asarray(bounds_min), np.asarray(bounds_max)
        axes = [np.arange(lo[i] + spacing / 2, hi[i], spacing) for i in range(3)]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([a.ravel() for a in grid])

    @classmethod
    def _sand_dam(cls, spacing=0.085):
        candidates = cls._lattice((-0.95, -0.94, 0.0), (-0.05, 0.94, 1.75), spacing)
        # Trapezoidal cross-section: broad foot and narrower crest, with no opening.
        z = candidates[:, 2]
        center_x = -0.50
        half_depth = 0.45 - 0.22 * (z / 1.75)
        return candidates[np.abs(candidates[:, 0] - center_x) <= half_depth]

    @classmethod
    def _water(cls, spacing=0.10):
        # The pool has a short air gap to the intact upstream dam face at t=0.
        pool = cls._lattice((-3.82, -0.90, 0.03), (-1.08, 0.90, 1.42), spacing)
        # A compact falling sheet makes the source water supply visible.
        sheet = cls._lattice((-3.40, -0.58, 1.55), (-2.76, 0.58, 2.48), spacing)
        return pool, sheet

    def initial_state(self) -> StateFrame:
        sand = self._sand_dam()
        pool, supply = self._water()
        water = np.vstack((pool, supply))
        position = np.vstack((sand, water))
        velocity = np.zeros_like(position)
        velocity[len(sand) + len(pool):, 2] = -1.15
        body_id = np.concatenate((
            np.full(len(sand), self.SAND_ID, dtype=np.int32),
            np.full(len(water), self.WATER_ID, dtype=np.int32),
        ))
        particle_volume = np.concatenate((
            np.full(len(sand), 0.085 ** 3),
            np.full(len(water), 0.10 ** 3),
        ))
        mass = np.concatenate((
            particle_volume[:len(sand)] * 1650.0,
            particle_volume[len(sand):] * 1000.0,
        ))
        saturation = np.zeros(len(position))
        saturation[len(sand):] = 1.0
        porosity = np.zeros(len(position))
        porosity[:len(sand)] = 0.38
        frame = StateFrame(
            0.0, position, velocity, mass=mass, body_id=body_id,
            attributes={
                "material_id": body_id.copy(),
                "sand_saturation": saturation,
                "porosity": porosity,
                "particle_volume": particle_volume,
            },
            scalars={
                "initial_sand_mass": float(mass[body_id == self.SAND_ID].sum()),
                "initial_water_mass": float(mass[body_id == self.WATER_ID].sum()),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
