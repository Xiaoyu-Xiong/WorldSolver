"""Deterministic solver-neutral Wilberforce pendulum release scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        cfg = self.config
        return {
            "scene_id": cfg["scene_id"],
            "seed": cfg["seed"],
            "phenomenon": "weakly coupled vertical-torsional energy exchange",
            "gravity": cfg["gravity"],
            "preparation": cfg["initial_conditions"]["preparation"],
            "fixed_upper_end": {"body_id": 0, "vertex_count": cfg["spring"]["upper_clamped_vertex_count"]},
            "load_attachment": {"spring_body_id": 0, "load_body_id": 1, "at_last_spring_vertex": True},
            "load_principal_inertia": cfg["load"]["principal_inertia"],
            "coupled_mode_regime": cfg["coupled_mode_regime"],
            "event_sequence": ["vertical_dominant", "rotation_dominant", "vertical_revival"],
        }

    def initial_state(self) -> StateFrame:
        cfg, spring, load = self.config, self.config["spring"], self.config["load"]
        n = int(spring["vertex_count"])
        theta = np.linspace(0.0, 2.0 * np.pi * float(spring["coil_count"]), n)
        dz = float(cfg["initial_conditions"]["upward_displacement"])
        z = np.linspace(float(spring["equilibrium_top_z"]), float(spring["equilibrium_bottom_z"]), n) + dz
        position = np.column_stack((float(spring["coil_radius"]) * np.cos(theta),
                                    float(spring["coil_radius"]) * np.sin(theta), z))
        load_center = np.array([[0.0, 0.0, z[-1] - 0.15]])
        marker_center = np.array([[0.0, 0.0, load_center[0, 2] + 0.08]])
        position = np.vstack((position, load_center, marker_center))
        velocity = np.zeros_like(position)
        orientation = np.zeros((n + 2, 4)); orientation[:, 3] = 1.0
        angular_velocity = np.zeros_like(position)
        segment_mass = float(spring["total_mass"]) / n
        mass = np.r_[np.full(n, segment_mass), float(load["mass"]), 0.0]
        body_id = np.r_[np.zeros(n, dtype=np.int64), 1, 2]
        frame = StateFrame(
            0.0, position, velocity, orientation=orientation,
            angular_velocity=angular_velocity, mass=mass, body_id=body_id,
            attributes={
                "arc_parameter": np.r_[np.linspace(0.0, 1.0, n), 1.0, 1.0],
                "is_upper_clamped": np.r_[np.arange(n) < int(spring["upper_clamped_vertex_count"]), 0, 0].astype(float),
                "is_orientation_marker": np.r_[np.zeros(n + 1), 1.0],
            },
            scalars={"equilibrium_upward_release_offset": dz},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
