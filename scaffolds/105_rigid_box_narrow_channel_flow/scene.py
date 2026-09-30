"""Deterministic solver-neutral initial state for the narrow-channel experiment."""
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
            "dimension": 2,
            "coupling": "two_way_fluid_rigid",
            "fluid_drive_target": "fluid",
        }

    def initial_state(self) -> StateFrame:
        fluid = self.config["fluid"]
        box = self.config["rigid_box"]
        spacing = float(fluid["marker_spacing"])
        xs = np.arange(fluid["domain_min"][0] + spacing / 2,
                       fluid["domain_max"][0], spacing)
        zs = np.arange(fluid["domain_min"][2] + spacing / 2,
                       fluid["domain_max"][2], spacing)
        xx, zz = np.meshgrid(xs, zs, indexing="xy")
        markers = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))

        # Remove markers initially inside the tilted rigid box, retaining both side gaps.
        angle = np.deg2rad(float(box["initial_angle_degrees"]))
        rel = markers[:, [0, 2]] - np.asarray(box["initial_center"])[[0, 2]]
        local_x = np.cos(angle) * rel[:, 0] - np.sin(angle) * rel[:, 1]
        local_z = np.sin(angle) * rel[:, 0] + np.cos(angle) * rel[:, 1]
        keep = (np.abs(local_x) > box["size"][0] / 2) | (np.abs(local_z) > box["size"][2] / 2)
        markers = markers[keep]

        position = np.vstack((np.asarray(box["initial_center"], dtype=float), markers))
        velocity = np.zeros_like(position)
        velocity[0] = np.asarray(box["initial_velocity"], dtype=float)
        n = len(position)
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        orientation[0] = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
        angular_velocity = np.zeros((n, 3))
        angular_velocity[0] = np.asarray(box["initial_angular_velocity"], dtype=float)
        mass = np.full(n, float(fluid["density"]) * spacing * spacing * 0.08)
        mass[0] = float(box["mass"])
        body_id = np.zeros(n, dtype=np.int64)
        body_id[0] = int(box["body_id"])
        phase = np.ones(n, dtype=float)
        phase[0] = 0.0
        return StateFrame(
            time=0.0, position=position, velocity=velocity,
            orientation=orientation, angular_velocity=angular_velocity,
            mass=mass, body_id=body_id, attributes={"phase": phase},
            fields={"fluid_velocity": np.zeros((224, 64, 3), dtype=float)},
            scalars={"fluid_drive": 0.0},
        )

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
