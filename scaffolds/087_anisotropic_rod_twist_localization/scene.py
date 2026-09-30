"""Deterministic, solver-neutral V-shaped anisotropic rod scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        rod = self.config["rod"]
        bc = self.config["boundary_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "twist localization and transmission across a V-shaped bend",
            "dynamic_object": "anisotropic_rod",
            "reference_shape": "v_shaped",
            "inextensible": True,
            "cross_section": {
                "width": float(rod["cross_section_width"]),
                "thickness": float(rod["cross_section_thickness"]),
                "orientation_marking": "continuous_red_surface_stripe",
            },
            "endpoint_constraints": bc,
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        n = int(rod["vertex_count"])
        apex = int(rod["apex_vertex"])
        control = np.asarray(self.config["initial_conditions"]["centerline_points"], dtype=float)
        left = np.linspace(control[0], control[1], apex + 1)
        right = np.linspace(control[1], control[2], n - apex)[1:]
        position = np.vstack((left, right))
        velocity = np.zeros_like(position)

        # A twist-free material frame. Quaternions are identity because the
        # initial orientation marking lies in the common x-z centerline plane.
        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0
        segment_lengths = np.linalg.norm(np.diff(position, axis=0), axis=1)
        vertex_lengths = np.empty(n)
        vertex_lengths[0] = 0.5 * segment_lengths[0]
        vertex_lengths[-1] = 0.5 * segment_lengths[-1]
        vertex_lengths[1:-1] = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
        area = float(rod["cross_section_width"]) * float(rod["cross_section_thickness"])
        mass = float(rod["density"]) * area * vertex_lengths
        arc_length = np.r_[0.0, np.cumsum(segment_lengths)]
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "arc_length": arc_length,
                "material_twist": np.zeros(n),
                "cross_section_width": np.full(n, float(rod["cross_section_width"])),
                "cross_section_thickness": np.full(n, float(rod["cross_section_thickness"])),
                "is_apex": (np.arange(n) == apex).astype(float),
                "is_driven_end": (np.arange(n) == 0).astype(float),
                "is_fixed_end": (np.arange(n) == n - 1).astype(float),
            },
            scalars={"driven_end_angle": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
