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
    def _block(bounds, spacing):
        lo, hi = np.asarray(bounds[:3], float), np.asarray(bounds[3:], float)
        axes = [np.arange(lo[i] + spacing * .5, hi[i], spacing) for i in range(3)]
        if any(len(a) == 0 for a in axes):
            return ((lo + hi) * .5)[None, :]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    @staticmethod
    def _hollow_ball(spacing):
        topology = json.loads(
            (Path(__file__).resolve().parent / "assets" / "hollow_ball_topology.json").read_text(
                encoding="utf-8"
            )
        )
        return np.asarray(topology["vertices"], dtype=float) + np.asarray([0.0, 0.0, 1.95])

    def initial_state(self) -> StateFrame:
        sampling = self.config["sampling"]
        spacing, density = float(sampling["spacing"]), float(sampling["density"])
        clouds = [self._hollow_ball(spacing)]
        position = np.concatenate(clouds)
        mass = np.full(len(position), density * spacing ** 3)
        labels = np.concatenate([np.full(len(c), i, np.int32) for i, c in enumerate(clouds)])
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
