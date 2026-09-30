"""Deterministic solver-neutral scene for Michell buckling of a closed ring."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        initial = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "Michell instability of a pre-twisted elastic ring",
            "dynamic_object": "pretwisted_ring",
            "loading_sequence": ["release_compatible_preloaded_closed_ring", "free_relaxation"],
            "supports_or_shape_imposing_geometry": False,
            "continuous_twist_injection": False,
            "material_frame_winding_number": int(initial["material_frame_winding_number"]),
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        initial = self.config["initial_conditions"]
        n = int(rod["vertex_count"])
        radius = float(rod["radius"])
        phase = 2.0 * np.pi * np.arange(n) / n
        mode = int(initial["out_of_plane_mode"])
        amplitude = float(initial["out_of_plane_amplitude"])

        # Periodic two-lobed seed preserves exact closure while selecting the
        # alternating up/down branch visible in the source experiment.
        position = np.column_stack((
            radius * np.cos(phase),
            radius * np.sin(phase),
            amplitude * np.sin(mode * phase),
        ))
        position += np.asarray(initial["center"], dtype=float)
        velocity = np.zeros_like(position)

        segment_length = float(rod["length"]) / n
        line_mass = float(rod["density"]) * np.pi * float(rod["cross_section_radius"]) ** 2
        mass = np.full(n, line_mass * segment_length)
        winding = int(initial["material_frame_winding_number"])
        material_angle = winding * phase

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "arc_length": segment_length * np.arange(n),
                "closed_curve_index": np.arange(n, dtype=float),
                "natural_curvature": np.zeros(n),
                "natural_twist": np.zeros(n),
                "material_frame_angle": material_angle,
                "stored_twist_density": np.full(n, float(initial["stored_total_twist_rad"]) / float(rod["length"])),
            },
            scalars={
                "rod_radius": float(rod["cross_section_radius"]),
                "stored_total_twist_rad": float(initial["stored_total_twist_rad"]),
                "material_frame_winding_number": float(winding),
                "closed_ring": 1.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
