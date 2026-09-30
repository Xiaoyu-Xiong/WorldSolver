"""Scaffold Agent-owned scene definition.

Self-supporting parabolic arch assembled from rigid voussoir blocks.
The scene reproduces the observable phenomenon from Rojas et al.,
"Non-Smooth Newton Methods for Deformable Multi-Body Dynamics"
(Figure 14 top): a parabolic arch of rigid blocks that supports itself
through frictional contact and maintains a stable load path under gravity.

Geometry is solver-neutral. Each block is a mating wedge (voussoir) whose
radial contact faces are shared with its neighbours so the assembled arch
forms a continuous compressive load path. No solver, integrator, or contact
model is prescribed here.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

# Block centroids (world translation, meters) and masses (kg) for the 15
# voussoir blocks, ordered from the -x springing block to the +x springing
# block with the keystone at the centre. These values are produced by the
# deterministic arch generator and must match scene_geometry.yaml exactly.
_BLOCKS: list[tuple[float, float, float, float]] = [
    (-1.910879, 0.0, 0.290781, 27.115),
    (-1.725044, 0.0, 0.661629, 42.566),
    (-1.521962, 0.0, 1.023225, 57.631),
    (-1.296525, 0.0, 1.371116, 72.125),
    (-1.041099, 0.0, 1.697086, 85.707),
    (-0.744536, 0.0, 1.984451, 97.679),
    (-0.395025, 0.0, 2.197916, 106.573),
    (0.0, 0.0, 2.280175, 110.0),
    (0.395025, 0.0, 2.197916, 106.573),
    (0.744536, 0.0, 1.984451, 97.679),
    (1.041099, 0.0, 1.697086, 85.707),
    (1.296525, 0.0, 1.371116, 72.125),
    (1.521962, 0.0, 1.023225, 57.631),
    (1.725044, 0.0, 0.661629, 42.566),
    (1.910879, 0.0, 0.290781, 27.115),
]


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "self_supporting_parabolic_arch",
            "num_bodies": len(_BLOCKS),
            "friction_coefficient": 0.6,
            "gravity": [0.0, 0.0, -9.81],
        }

    def initial_state(self) -> StateFrame:
        n = len(_BLOCKS)
        position = np.array([[b[0], b[1], b[2]] for b in _BLOCKS], dtype=float)
        velocity = np.zeros((n, 3), dtype=float)
        # Identity orientation (xyzw); block shape already encodes the wedge.
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        angular_velocity = np.zeros((n, 3), dtype=float)
        mass = np.array([b[3] for b in _BLOCKS], dtype=float)
        body_id = np.arange(n, dtype=int)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
