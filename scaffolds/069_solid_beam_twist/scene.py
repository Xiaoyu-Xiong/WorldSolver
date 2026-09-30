"""Deterministic solver-neutral scene contract for a twisted solid beam."""
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
            "phenomenon": "distributed_twist_of_volumetric_solid",
            "beam_axis": "x",
            "boundary_conditions": self.config["boundary_conditions"],
        }

    def initial_state(self) -> StateFrame:
        beam = self.config["beam"]
        nx, ny, nz = (int(v) for v in beam["volumetric_grid_shape"])
        length, width, height = (float(v) for v in beam["dimensions"])
        center = np.asarray(beam["center"], dtype=float)
        x = np.linspace(-0.5 * length, 0.5 * length, nx)
        y = np.linspace(-0.5 * width, 0.5 * width, ny)
        z = np.linspace(-0.5 * height, 0.5 * height, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        position = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())) + center
        velocity = np.zeros_like(position)
        cell_volume = (length / (nx - 1)) * (width / (ny - 1)) * (height / (nz - 1))
        mass = np.full(len(position), float(beam["density"]) * cell_volume)
        half_length = 0.5 * length
        band = float(self.config["boundary_conditions"]["end_face_band"])
        end_region = np.zeros(len(position), dtype=float)
        end_region[position[:, 0] <= center[0] - half_length + band] = -1.0
        end_region[position[:, 0] >= center[0] + half_length - band] = 1.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"end_region": end_region, "rest_position": position.copy()},
            scalars={"relative_end_rotation_degrees": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
