"""Deterministic solver-neutral snowball / sticky-wall scene contract."""
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
            "phenomenon": "snowball_wall_sticky",
            "gravity": list(self.config["gravity"]),
            "contact_mode": self.config["contact"]["mode"],
            "wall_inner_x": float(self.config["contact"]["wall_inner_x"]),
            "floor_z": float(self.config["contact"]["floor_z"]),
        }

    def _particles(self) -> np.ndarray:
        spec = self.config["snowball"]
        center = np.asarray(spec["center"], dtype=float)
        radius, spacing = float(spec["radius"]), float(spec["particle_spacing"])
        count = int(np.ceil(2.0 * radius / spacing))
        axis = (np.arange(count + 1, dtype=float) - count / 2.0) * spacing
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        points = grid[np.einsum("ij,ij->i", grid, grid) <= radius * radius]
        return points + center

    def initial_state(self) -> StateFrame:
        position = self._particles()
        n = len(position)
        spec, contact = self.config["snowball"], self.config["contact"]
        velocity = np.tile(np.asarray(spec["initial_velocity"], dtype=float), (n, 1))
        mass = np.full(n, float(spec["total_mass"]) / n, dtype=float)
        volume = mass / float(spec["nominal_density"])
        gravity = np.asarray(self.config["gravity"], dtype=float)
        wall_gap = float(contact["wall_inner_x"]) - position[:, 0]
        floor_gap = position[:, 2] - float(contact["floor_z"])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "reference_volume": volume,
                "deformation_gradient": np.tile(np.eye(3), (n, 1, 1)),
                "initial_component_id": np.zeros(n, dtype=np.int64),
            },
            fields={
                "external_force": mass[:, None] * gravity[None, :],
                "wall_signed_distance": wall_gap,
                "floor_signed_distance": floor_gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "contact_length_scale": float(contact["contact_length_scale"]),
                "initial_total_mass": float(mass.sum()),
                "initial_total_volume": float(volume.sum()),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
