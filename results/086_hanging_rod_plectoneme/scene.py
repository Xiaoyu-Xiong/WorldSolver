"""Deterministic solver-neutral hanging-rod plectoneme scene."""
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
            "phenomenon": "twist-driven plectoneme formation in a hanging elastic rod",
            "gravity": self.config["gravity"],
            "ambient_drag": self.config["ambient_drag"],
            "rod": self.config["rod"],
            "end_controls": self.config["end_controls"],
            "event_windows": self.config["event_windows"],
            "required_outcome": self.config["required_outcome"],
        }

    def _centerline(self) -> np.ndarray:
        """Planar, twist-free symmetric hanging curve with fixed endpoints."""
        rod = self.config["rod"]
        controls = self.config["end_controls"]
        n = int(rod["vertex_count"])
        left = np.asarray(controls["left"]["position"], dtype=float)
        right = np.asarray(controls["right"]["position"], dtype=float)
        u = np.linspace(-1.0, 1.0, n)
        # A smooth parabola gives the intended gravity-sagged starting layout.
        x = left[0] + (u + 1.0) * 0.5 * (right[0] - left[0])
        y = np.zeros(n)
        z = left[2] - 2.52 * (1.0 - u * u)
        p = np.column_stack((x, y, z))
        p[0], p[-1] = left, right
        return p

    def initial_state(self) -> StateFrame:
        p = self._centerline()
        n = len(p)
        rod = self.config["rod"]
        radius = float(rod["radius"])
        mass = np.full(n, float(rod["mass"]) / (n - 1))
        mass[[0, -1]] *= 0.5
        endpoint = np.zeros(n)
        endpoint[[0, -1]] = 1.0
        frame = StateFrame(
            time=0.0,
            position=p,
            velocity=np.zeros_like(p),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            angular_velocity=np.zeros_like(p),
            mass=mass,
            attributes={
                "arc_fraction": np.linspace(0.0, 1.0, n),
                "rod_radius": np.full(n, radius),
                "contact_diameter": np.full(n, 2.0 * radius),
                "endpoint_controlled": endpoint,
                "material_twist_turns": np.zeros(n),
            },
            scalars={"rest_length": float(rod["total_length"])},
        )
        frame.validate()
        return frame

    def end_control_targets(self, time: float) -> dict:
        """Fixed endpoint positions and progressively opposed axial rotations."""
        controls = self.config["end_controls"]
        ramp = controls["twist_ramp"]
        span = float(ramp["ramp_end"] - ramp["ramp_start"])
        u = np.clip((float(time) - float(ramp["ramp_start"])) / span, 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        relative_angle = 2.0 * np.pi * float(ramp["final_relative_turns"]) * u
        return {
            "left": {
                "position": controls["left"]["position"],
                "rotation_axis": controls["left"]["outward_axis"],
                "angle_radians": -0.5 * relative_angle,
            },
            "right": {
                "position": controls["right"]["position"],
                "rotation_axis": controls["right"]["outward_axis"],
                "angle_radians": 0.5 * relative_angle,
            },
        }

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
