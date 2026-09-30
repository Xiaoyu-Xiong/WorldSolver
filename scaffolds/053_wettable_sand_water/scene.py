"""Deterministic solver-neutral scene contract for wettable, insoluble sand."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "materials": ["initially_dry_wettable_sand", "water"],
            "sand_phase_evolution": False,
            "interaction": "wetting_only",
            "initial_wetness": "all sand particles start dry; no left/right wetness split",
            "rendered_scalar_attributes": {
                "wetness": "0 dry sand, 0.05 saturated wettable sand; only sand particles should develop this after water contact",
                "water_concentration": "water material fraction used for wetting transport",
            },
            "required_outcomes": ["wetting_penetration", "pile_deformation", "phase_mass", "runoff"],
        }

    @staticmethod
    def _sand_pile(spacing: float = 0.075) -> np.ndarray:
        points = []
        z = spacing * 0.5
        while z <= 1.05:
            radius = 0.78 * (1.0 - z / 1.12)
            for y in np.arange(-radius, radius + spacing * 0.5, spacing):
                for x in np.arange(-radius, radius + spacing * 0.5, spacing):
                    if x * x + (1.08 * y) ** 2 <= radius * radius:
                        points.append((x, y + 0.12, z))
            z += spacing
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _water_streams(spacing: float = 0.065) -> tuple[np.ndarray, np.ndarray]:
        points, velocity = [], []
        for side in (-1.0, 1.0):
            for k in range(9):
                z = 1.42 + k * spacing
                center_x = side * (0.28 + 0.035 * k)
                for dx in (-0.5, 0.5):
                    for dy in (-0.5, 0.5):
                        points.append((center_x + dx * spacing, 0.08 + dy * spacing, z))
                        velocity.append((-side * 0.38, 0.0, -1.85))
        return np.asarray(points), np.asarray(velocity)

    def initial_state(self) -> StateFrame:
        sand = self._sand_pile()
        water, water_velocity = self._water_streams()
        position = np.vstack((sand, water))
        velocity = np.vstack((np.zeros_like(sand), water_velocity))
        body_id = np.concatenate((np.ones(len(sand), dtype=np.int32), np.full(len(water), 2, dtype=np.int32)))
        solid = np.concatenate((np.ones(len(sand)), np.zeros(len(water))))
        concentration = np.concatenate((np.zeros(len(sand)), np.ones(len(water))))
        mass = np.concatenate((np.full(len(sand), 0.00063), np.full(len(water), 0.000275)))
        wetness = np.concatenate((np.zeros(len(sand)), np.ones(len(water))))
        frame = StateFrame(0.0, position, velocity, mass=mass, body_id=body_id, attributes={"material_id": body_id.copy(), "solid_phase_fraction": solid, "water_concentration": concentration, "wetness": wetness}, scalars={"sand_mass": float(mass[body_id == 1].sum()), "water_mass": float(mass[body_id == 2].sum())})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
