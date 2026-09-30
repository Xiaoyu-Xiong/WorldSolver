"""Deterministic solver-neutral water/cloth curtain scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    CLOTH_NY, CLOTH_NZ = 31, 41

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "two-way coupled water stream loading an impermeable cloth curtain",
            "cloth_pinned_vertex_indices": list(range(self.CLOTH_NY)),
            "inflow": {"center": [-1.255, 0.0, 1.72], "radius": 0.20,
                       "velocity": [3.2, 0.0, 0.0], "active_interval": [0.0, 4.0]},
            "scaffold_visual_intent": (
                "The cloth topology asset contains vertices and faces for rendering; red cloth and "
                "cyan-blue water must remain visually separated at the impact interface."
            ),
        }

    @staticmethod
    def _lattice(lo, hi, spacing):
        axes = [np.arange(a, b + 0.25 * spacing, spacing) for a, b in zip(lo, hi)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def initial_state(self) -> StateFrame:
        # Curtain lies in the y-z plane; a slight catenary at the pinned top edge
        # avoids a perfectly symmetric response while remaining deterministic.
        y = np.linspace(-0.92, 0.92, self.CLOTH_NY)
        z = np.linspace(2.72, 0.43, self.CLOTH_NZ)
        yy, zz = np.meshgrid(y, z, indexing="xy")
        xx = 0.06 * (yy / 0.92) ** 2
        cloth = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        # A shallow retained volume plus an already established cylindrical jet.
        pool = self._lattice((-1.48, -1.27, 0.08), (1.95, 1.27, 0.29), 0.105)
        jet_box = self._lattice((-1.28, -0.19, 1.54), (-0.08, 0.19, 1.90), 0.075)
        jet = jet_box[(jet_box[:, 1] ** 2 + (jet_box[:, 2] - 1.72) ** 2) <= 0.20 ** 2]
        water = np.vstack((pool, jet))

        position = np.vstack((cloth, water))
        velocity = np.zeros_like(position)
        velocity[len(cloth) + len(pool):, 0] = 3.2
        body_id = np.r_[np.zeros(len(cloth), dtype=np.int32),
                        np.ones(len(water), dtype=np.int32)]
        mass = np.r_[np.full(len(cloth), 0.42 / len(cloth)),
                     np.full(len(water), 1000.0 * 0.105 ** 3)]
        pinned = np.zeros(len(position), dtype=np.float64)
        pinned[:self.CLOTH_NY] = 1.0
        material_phase = body_id.astype(np.float64)
        frame = StateFrame(
            time=0.0, position=position, velocity=velocity, mass=mass, body_id=body_id,
            attributes={"pinned": pinned, "material_phase": material_phase},
            scalars={"water_density": 1000.0, "cloth_total_mass": 0.42,
                     "cloth_thickness": 0.0015, "nozzle_radius": 0.20,
                     "nominal_inflow_speed": 3.2},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
