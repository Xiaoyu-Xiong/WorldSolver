"""Deterministic solver-neutral sampled-material scene."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "family": self.config["scene_family"], "variant": self.config["variant"],
                "gravity": list(self.config["gravity"])}

    @staticmethod
    def _bowl_vertices():
        asset = Path(__file__).resolve().parent / "assets" / "bowl_topology.json"
        payload = json.loads(asset.read_text(encoding="utf-8"))
        points = np.asarray(payload["vertices"], dtype=float)
        angle = np.deg2rad(58.0)
        rotation = np.asarray([[np.cos(angle), 0.0, np.sin(angle)],
                               [0.0, 1.0, 0.0],
                               [-np.sin(angle), 0.0, np.cos(angle)]])
        return points @ rotation.T + np.asarray([0.25, 0.0, 0.76])

    def initial_state(self) -> StateFrame:
        sampling = self.config["sampling"]
        spacing, density = float(sampling["spacing"]), float(sampling["density"])
        position = self._bowl_vertices()
        mass = np.full(len(position), density * spacing ** 3)
        labels = np.zeros(len(position), dtype=np.int32)
        fields = {"cumulative_mass_source": np.zeros(len(position))}
        if self.config["scene_family"] == "gas":
            shape = tuple(sampling["field_grid_shape"])
            scalar = np.zeros(shape)
            scalar[:, :, :max(2, shape[2] // 5)] = .7
            fields.update({"transported_scalar": scalar, "scalar_source": np.zeros(shape),
                           "active_mask": np.ones(shape, dtype=bool),
                           "velocity_field": np.zeros(shape + (3,))})
        frame = StateFrame(
            0.0, position, np.zeros_like(position), mass=mass, body_id=labels,
            attributes={"mass_quantity": mass.copy(),
                        "particle_radius": np.full(len(position), .45 * spacing),
                        "material_label": labels,
                        "deformation_gradient": np.repeat(np.eye(3)[None], len(position), axis=0)},
            fields=fields,
            scalars={"contact_length_scale": spacing, "grid_spacing": spacing,
                     "characteristic_length": max(float(np.ptp(position, axis=0).max()), spacing),
                     "reference_velocity": 1.0})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1) * dt).tolist()
