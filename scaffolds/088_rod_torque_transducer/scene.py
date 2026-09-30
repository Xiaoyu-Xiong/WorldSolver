"""Deterministic solver-neutral torque-transducer scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "rod-mediated torque transmission between unequal rigid bodies",
            "rod": self.config["rod"],
            "endpoint_attachments": {"start": "driven_body", "end": "receiver_body"},
            "rigid_assemblies": {
                "driven_body": [1, 3],
                "receiver_body": [2, 4],
            },
            "orientation_marks": "yellow radial bars welded to each body",
            "excitation": self.config["excitation"],
            "constraints": self.config["boundary_conditions"],
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        n = int(rod["vertex_count"])
        start = np.asarray(self.config["initial_conditions"]["rod_start"], dtype=float)
        end = np.asarray(self.config["initial_conditions"]["rod_end"], dtype=float)
        rod_position = np.linspace(start, end, n)
        driven = np.asarray(self.config["rigid_bodies"]["driven"]["center"], dtype=float)
        receiver = np.asarray(self.config["rigid_bodies"]["receiver"]["center"], dtype=float)
        positions = np.vstack((rod_position, driven, receiver,
                               driven + [0.0, 0.0, 0.075],
                               receiver + [0.0, 0.0, 0.0475]))
        count = len(positions)
        orientations = np.zeros((count, 4), dtype=float)
        orientations[:, 3] = 1.0
        segment = float(rod["length"]) / (n - 1)
        rod_mass = np.full(n, float(rod["density"]) * np.pi * float(rod["radius"])**2 * segment)
        masses = np.r_[rod_mass,
                       float(self.config["rigid_bodies"]["driven"]["mass"]),
                       float(self.config["rigid_bodies"]["receiver"]["mass"]), 0.0, 0.0]
        body_id = np.r_[np.zeros(n, dtype=int), [1, 2, 3, 4]]
        component = np.r_[np.zeros(n), [1.0, 2.0, 3.0, 4.0]]
        arc = np.r_[np.linspace(0.0, float(rod["length"]), n), np.zeros(4)]
        frame = StateFrame(
            time=0.0,
            position=positions,
            velocity=np.zeros((count, 3)),
            orientation=orientations,
            angular_velocity=np.zeros((count, 3)),
            mass=masses,
            body_id=body_id,
            attributes={
                "component_id": component,
                "arc_length": arc,
                "material_twist": np.zeros(count),
                "is_driven": np.r_[np.zeros(n), [1.0, 0.0, 1.0, 0.0]],
            },
            scalars={"applied_torque": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
