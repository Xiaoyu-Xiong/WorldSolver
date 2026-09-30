"""Deterministic solver-neutral initial state for Water Collapses Towers."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Defines samples only; the downstream agent chooses all physics methods."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "incoming water momentum collapses two mixed-density towers",
            "coupled_materials": {
                "water": "water_density",
                "light_rigid": "light_cylinders",
                "heavy_rigid": "heavy_cubes",
            },
            "tower_failure_window_s": [0.8, 3.0],
            "scaffold_visual_intent": (
                "Left and right towers use separate high-contrast palettes, and light cylinders "
                "are distinct from heavy cubes; incoming water is saturated blue."
            ),
        }

    @staticmethod
    def _water_samples() -> np.ndarray:
        # Regular deterministic volume samples; spacing leaves a dry contact gap.
        x = np.arange(-3.72, -1.34, 0.16)
        y = np.arange(-1.62, 1.63, 0.16)
        z = np.arange(0.08, 1.21, 0.16)
        return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    @staticmethod
    def _tower(center_x: float, center_y: float, cylinder_group: int, cube_group: int):
        offsets = np.array([
            [-0.27, -0.27], [0.0, -0.27], [0.27, -0.27],
            [-0.27, 0.0], [0.0, 0.0], [0.27, 0.0],
            [-0.27, 0.27], [0.0, 0.27], [0.27, 0.27],
        ])
        pos = np.column_stack((offsets[:, 0] + center_x, offsets[:, 1] + center_y,
                               np.full(9, 0.925)))
        # Checkerboard mixture makes both geometry and density difference readable.
        is_cube = np.array([False, True, False, True, False, True, False, True, False])
        groups = np.where(is_cube, cube_group, cylinder_group)
        density = np.where(is_cube, 1500.0, 600.0)
        volume = np.where(is_cube, 0.23 * 0.23 * 1.85,
                          np.pi * 0.115**2 * 1.85)
        return pos, groups, density * volume

    def initial_state(self) -> StateFrame:
        water = self._water_samples()
        left = self._tower(0.15, 0.70, 20, 21)
        right = self._tower(0.65, -0.70, 30, 31)
        position = np.vstack((water, left[0], right[0])).astype(float)
        velocity = np.zeros_like(position)
        velocity[:len(water), 0] = 3.2
        groups = np.concatenate((np.full(len(water), 10), left[1], right[1])).astype(float)
        water_cell_mass = np.full(len(water), 1000.0 * 0.16**3)
        mass = np.concatenate((water_cell_mass, left[2], right[2]))
        density = np.concatenate((np.full(len(water), 1000.0),
                                  mass[len(water):] / np.concatenate((
                                      np.where(left[1] % 10 == 1, 0.23**2 * 1.85, np.pi * 0.115**2 * 1.85),
                                      np.where(right[1] % 10 == 1, 0.23**2 * 1.85, np.pi * 0.115**2 * 1.85)))))
        n = len(position)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
            angular_velocity=np.zeros((n, 3)),
            mass=mass,
            body_id=np.concatenate((np.zeros(len(water), dtype=int), np.arange(1, 19))),
            attributes={"scene_group": groups, "density": density},
            fields={"density": density.copy()},
            scalars={"water_rest_density": 1000.0, "initial_water_speed": 3.2},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
