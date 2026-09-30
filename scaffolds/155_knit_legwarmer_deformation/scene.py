"""Deterministic solver-neutral thin-sheet scaffold."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "variant": self.config["variant"], "gravity": list(self.config["gravity"]),
                "grid_shape": list(self.config["cloth"]["grid_shape"])}

    def initial_state(self) -> StateFrame:
        cloth = self.config["cloth"]
        nx, ny = map(int, cloth["grid_shape"])
        sx, _ = map(float, cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        x = np.linspace(-sx * .5, sx * .5, ny)
        angles = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        radius = 0.42
        position = np.asarray([[axial, radius * np.cos(angle), radius * np.sin(angle)]
                               for axial in x for angle in angles], dtype=float) + center
        mass = np.full(len(position), float(cloth["total_mass"]) / len(position))
        boundary = np.zeros(len(position), dtype=np.int32)
        boundary[:nx] = 1
        boundary[-nx:] = 1
        frame = StateFrame(
            0.0, position, np.zeros_like(position), mass=mass,
            body_id=np.zeros(len(position), dtype=np.int32),
            attributes={"mass_quantity": mass.copy(), "boundary_vertex": boundary,
                        "deformation_gradient": np.repeat(np.eye(2)[None], len(position), axis=0)},
            fields={"cumulative_mass_source": np.zeros(len(position))},
            scalars={"contact_length_scale": float(cloth["thickness"])})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1) * dt).tolist()
