from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"]}

    def initial_state(self) -> StateFrame:
        # Closed icosahedral sphere vertices, in the same order as the OBJ asset.
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ]
        pos = np.asarray(verts, dtype=float) / np.sqrt(1.0 + phi**2)
        pos[:, 2] += 2.25
        vel = np.zeros_like(pos)
        mass = np.full(len(pos), 4.0 / len(pos), dtype=float)
        return StateFrame(0.0, pos, vel, mass=mass)

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
