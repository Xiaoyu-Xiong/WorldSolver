"""Deterministic solver-neutral sampled-material scene."""
from __future__ import annotations
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

    def initial_state(self) -> StateFrame:
        sampling = self.config["sampling"]
        spacing, density = float(sampling["spacing"]), float(sampling["density"])
        bed = self._block([-1.45, -1.0, 0.08, 1.45, 1.0, 0.42], spacing)
        # Sand remains a granular cloud; the impactor is one rigid body, not a
        # second cloud of particles.  This distinction is essential for reading
        # the crater/spray experiment and for rigid-contact scoring.
        ball_center = np.asarray([[0.0, 0.0, 1.85]], dtype=float)
        clouds = [bed, ball_center]
        position = np.concatenate(clouds)
        sand_mass = np.full(len(bed), density * spacing ** 3)
        ball_mass = np.asarray([density * (4.0 / 3.0) * np.pi * 0.54 ** 3])
        mass = np.concatenate((sand_mass, ball_mass))
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
