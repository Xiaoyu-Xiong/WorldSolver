"""Deterministic solver-neutral deformable-solid scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        asset = Path(__file__).resolve().parent / config["subject_asset"]
        with np.load(asset) as data:
            self.vertices = np.asarray(data["vertices"], dtype=float)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "scenario": self.config["scenario"],
            "gravity": list(self.config["gravity"]),
        }

    def initial_state(self) -> StateFrame:
        n = len(self.vertices)
        velocity = np.tile(np.asarray(self.config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float), (n, 1))
        total_mass = float(self.config["subject_mass"])
        mass = np.full(n, total_mass / n, dtype=float)
        gravity = np.asarray(self.config["gravity"], dtype=float)
        ground_z = float(self.config.get("ground_z", 0.0))
        gap = np.maximum(self.vertices[:, 2] - ground_z, 0.0)
        frame = StateFrame(
            time=0.0,
            position=self.vertices.copy(),
            velocity=velocity,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": np.tile(np.eye(3), (n, 1, 1)),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": mass[:, None] * gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.config["contact_length_scale"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
