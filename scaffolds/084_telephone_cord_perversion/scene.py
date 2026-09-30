"""Deterministic, solver-neutral telephone-cord perversion scene contract."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        controls = self.config["endpoint_controls"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "persistent junction of opposite-handed helices",
            "rod": self.config["rod"],
            "gravity": self.config["gravity"],
            "endpoint_controls": controls,
            "boundary_schedule": {
                "controlled_vertices": [0, self.config["rod"]["vertex_count"] - 1],
                "position_interpolation": "smoothstep",
                "orientations_fixed_in_world": True,
                "axial_spin_permitted": False,
            },
            "required_outcome": {
                "left_portion_handedness": "opposite_to_right",
                "transition_location_fraction": 0.5,
                "transition_persistent_through": self.config["duration"],
            },
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        n = int(rod["vertex_count"])
        s = np.linspace(0.0, 1.0, n)
        left = np.asarray(self.config["endpoint_controls"]["left"]["initial_position"])
        right = np.asarray(self.config["endpoint_controls"]["right"]["initial_position"])
        position = left[None, :] * (1.0 - s[:, None]) + right[None, :] * s[:, None]
        # The paper's prepared state is almost flat, not perfectly straight or stress-free.
        amp = float(rod["initial_preparation"]["centerline_wave_amplitude"])
        waves = int(rod["initial_preparation"]["centerline_wave_count"])
        position[:, 2] += amp * np.sin(2.0 * np.pi * waves * s) * np.sin(np.pi * s)
        velocity = np.zeros_like(position)
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        mass = np.full(n, float(rod["mass"]) / n)
        attributes = {
            "arc_fraction": s,
            "material_axis_1": np.tile(np.array([0.0, 1.0, 0.0]), (n, 1)),
            "cross_section_width": np.full(n, rod["cross_section"]["width"]),
            "cross_section_thickness": np.full(n, rod["cross_section"]["thickness"]),
            "natural_helix_radius": np.full(n, rod["natural_shape"]["radius"]),
            "natural_helix_pitch": np.full(n, rod["natural_shape"]["pitch_per_turn"]),
            "endpoint_controlled": np.isin(np.arange(n), [0, n - 1]).astype(float),
        }
        frame = StateFrame(0.0, position, velocity, orientation=orientation,
                           angular_velocity=np.zeros_like(position), mass=mass,
                           attributes=attributes)
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()

    def endpoint_targets(self, time: float) -> dict:
        """Prescribed quasi-static targets; downstream solvers may query this helper."""
        ctl = self.config["endpoint_controls"]
        motion = ctl["motion"]
        u = np.clip((time - motion["move_start"]) /
                    (motion["move_end"] - motion["move_start"]), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        result = {}
        for side in ("left", "right"):
            start = np.asarray(ctl[side]["initial_position"], dtype=float)
            end = np.asarray(ctl[side]["final_position"], dtype=float)
            result[side] = {
                "position": ((1.0 - u) * start + u * end).tolist(),
                "orientation_xyzw": ctl[side]["orientation_xyzw"],
            }
        return result
