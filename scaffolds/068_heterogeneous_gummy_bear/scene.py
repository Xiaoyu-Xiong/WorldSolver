"""Deterministic, solver-neutral heterogeneous gummy-bear scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _obj_vertices(name: str) -> np.ndarray:
    path = Path(__file__).resolve().parent / "assets" / name
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "dimension": 3,
            "phenomenon": "bonded heterogeneous deformable solid released into bending and turning motion",
            "materials": {
                "soft_bear_exterior": {"youngs_modulus_pa": 6.0e4, "density_kg_m3": 1000.0, "poisson_ratio": 0.45},
                "stiff_bear_interior": {"youngs_modulus_pa": 1.0e14, "density_kg_m3": 1000.0, "poisson_ratio": 0.45},
            },
            "bonded_interfaces": [{"objects": ["soft_bear_exterior", "stiff_bear_interior"], "condition": "no_relative_slip_or_separation"}],
            "initial_condition": "rest geometry with prescribed smooth velocity; unconstrained release at t=0",
            "required_behavior": "both regions share gross translation and rotation while the soft shell changes shape substantially more",
        }

    def initial_state(self) -> StateFrame:
        outer = _obj_vertices("gummy_bear_outer.obj")
        core = _obj_vertices("gummy_bear_stiff.obj")
        position = np.vstack((outer, core))

        # Smooth pre-release motion: downward curl at the ends plus a depthwise
        # turn. The identical spatial field on both regions makes the intended
        # bonded gross motion explicit without prescribing a solver.
        x = position[:, 0]
        z = position[:, 2]
        velocity = np.empty_like(position)
        velocity[:, 0] = 0.30 * (z - 1.15)
        velocity[:, 1] = 1.20 * x + 0.22
        velocity[:, 2] = -1.45 * x - 0.38 * x * x

        body_id = np.concatenate((np.zeros(len(outer), dtype=np.int32), np.ones(len(core), dtype=np.int32)))
        material_region = body_id.astype(np.float64)
        mass = np.full(len(position), 1.0 / len(position), dtype=np.float64)
        frame = StateFrame(0.0, position, velocity, mass=mass, body_id=body_id,
                           attributes={"material_region": material_region})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
