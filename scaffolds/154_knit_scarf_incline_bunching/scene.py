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
        sx, sy = map(float, cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        x = np.linspace(-sx * .5, sx * .5, nx)
        y = np.linspace(-sy * .5, sy * .5, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        if cloth.get("plane", "xy") == "xz":
            position = np.column_stack((xx.ravel(), np.zeros(xx.size), yy.ravel())) + center
        else:
            position = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) + center
        tilt = np.deg2rad(float(cloth.get("tilt_about_y_deg", 0.0)))
        if abs(tilt) > 1e-12:
            # Optional rigid placement for inclined-plane test scenes.  The cloth
            # remains a regular sheet; only its scaffold pose is changed.
            c, s = np.cos(tilt), np.sin(tilt)
            rel = position - center
            rel = np.column_stack((c * rel[:, 0] + s * rel[:, 2], rel[:, 1],
                                   -s * rel[:, 0] + c * rel[:, 2]))
            position = rel + center
        mass = np.full(len(position), float(cloth["total_mass"]) / len(position))
        boundary = ((np.isclose(xx, x[0])) | (np.isclose(xx, x[-1])) |
                    (np.isclose(yy, y[0])) | (np.isclose(yy, y[-1]))).ravel().astype(np.int32)
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
