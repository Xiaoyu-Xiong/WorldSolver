"""Solver-neutral shallow-water initial state with an explicit octagonal obstacle."""

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
            "phenomenon": "step-height shallow-water wave reflecting from an octagonal obstacle",
            "field_layout": "cell_centered_xy",
            "density_field_quantity": "water height in meters, not material mass density",
            "obstacle_asset": self.config["obstacle"]["asset"],
        }

    def initial_state(self) -> StateFrame:
        cfg = self.config
        domain, obstacle, initial = cfg["domain"], cfg["obstacle"], cfg["initial_conditions"]
        nx, ny = domain["grid_shape"]
        low, high = np.asarray(domain["bounds_min"]), np.asarray(domain["bounds_max"])
        dx, dy = (high[:2] - low[:2]) / np.array([nx, ny])
        x = low[0] + (np.arange(nx) + 0.5) * dx
        y = low[1] + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        theta = np.arange(8) * np.pi / 4 + obstacle["rotation_radians"]
        vertices = np.asarray(obstacle["center"]) + obstacle["circumradius"] * np.column_stack(
            (np.cos(theta), np.sin(theta))
        )
        solid = np.ones((nx, ny), dtype=bool)
        obstacle_normals = []
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            edge = b - a
            solid &= edge[0] * (yy - a[1]) - edge[1] * (xx - a[0]) >= 0
            obstacle_normals.extend([np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)] * 8)
        active = ~solid
        height = np.where(
            xx > initial["step_x"], initial["high_water_height"], initial["low_water_height"]
        )
        height[solid] = 0.0
        velocity_field = np.zeros((nx, ny, 2))
        velocity_field[active] = initial["initial_velocity"]
        position = np.column_stack((xx[active], yy[active], 0.5 * height[active]))
        velocity = np.column_stack((velocity_field[active], np.zeros(active.sum())))
        volume = height[active] * dx * dy
        mass = cfg["fluid"]["density"] * volume
        normals = np.vstack(
            (
                np.tile([1.0, 0.0], (ny, 1)),
                np.tile([-1.0, 0.0], (ny, 1)),
                np.tile([0.0, 1.0], (nx, 1)),
                np.tile([0.0, -1.0], (nx, 1)),
                obstacle_normals,
            )
        )
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"mass_quantity": mass.copy(), "water_height": height[active].copy()},
            fields={
                "density_field": height,
                "velocity_field": velocity_field,
                "active_mask": active.astype(np.uint8),
                "continuity_source": np.zeros_like(height),
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": normals,
                "boundary_target_velocity": np.zeros_like(normals),
                "boundary_velocity": np.zeros_like(normals),
            },
            scalars={
                "grid_spacing": float(dx),
                "reference_velocity": float(np.sqrt(abs(cfg["gravity"][2]) * height.max())),
                "initial_water_volume": float(volume.sum()),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1) * dt).tolist()
