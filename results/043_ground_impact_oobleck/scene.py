"""Deterministic solver-neutral scene for the ground-impact oobleck benchmark."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """A sampled spherical body released toward a horizontal contact floor."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        initial = self.config["initial_conditions"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "units": "meters",
            "up_axis": "z",
            "gravity": list(self.config["gravity"]),
            "material_semantic": "oobleck_shear_thickening_granular_continuum",
            "required_behavior": (
                "The right-hand yellow-green oobleck body rebounds elastically after the first "
                "high-speed ground impact, then flows and spreads after the slower second impact. "
                "The left warm-coral body is a visual comparison matching the source figure layout."
            ),
            "initial_shape": initial.get("body_shape", "squat_bunny_like_blob"),
            "scaffold_visual_intent": (
                "Render a side-by-side Figure-13-style comparison: warm-coral viscoplastic body on "
                "the left, saturated yellow-green oobleck target on the right, both on a warm gray "
                "floor so impact patches, rebound gaps, and final spread are distinguishable."
            ),
            "sphere_radius": float(initial["sphere_radius"]),
            "contact_floor_z": float(
                self.config["boundary_conditions"]["ground"]["plane_z"]
            ),
            "event_windows": dict(self.config["event_windows"]),
        }

    def initial_state(self) -> StateFrame:
        initial = self.config["initial_conditions"]
        velocity = np.asarray(initial["velocity"], dtype=np.float64)
        radius = float(initial["sphere_radius"])
        spacing = float(initial["particle_spacing"])

        # Cell-centered cubic sampling, clipped to the analytic sphere.  The
        # ordering is stable and contains no solver-dependent randomness.
        axis = np.arange(-radius + 0.5 * spacing, radius, spacing)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        offsets = grid.reshape(-1, 3)
        offsets = offsets[np.einsum("ij,ij->i", offsets, offsets) <= radius**2]
        centers = initial.get("comparison_centers", {})
        left_center = np.asarray(centers.get("viscoplastic_left", [-0.48, 0.0, 1.62]), dtype=np.float64)
        right_center = np.asarray(centers.get("oobleck_right", [0.48, 0.0, 1.62]), dtype=np.float64)
        left_position = offsets + left_center
        right_position = offsets + right_center
        position = np.vstack((left_position, right_position))
        velocities = np.broadcast_to(velocity, position.shape).copy()
        body_id = np.concatenate((
            np.ones(len(left_position), dtype=np.int32),
            np.full(len(right_position), 2, dtype=np.int32),
        ))

        density = float(initial["density"])
        particle_mass = density * (4.0 * np.pi * radius**3 / 3.0) / len(offsets)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocities,
            mass=np.full(len(position), particle_mass, dtype=np.float64),
            body_id=body_id,
            attributes={
                "particle_radius": np.full(len(position), 0.5 * spacing),
                "rest_position": position.copy(),
                "material_label": body_id.copy(),
            },
            scalars={
                "total_mass": particle_mass * len(position),
                "initial_kinetic_energy": (
                    0.5 * particle_mass * len(position) * float(velocity @ velocity)
                ),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=np.float64) * dt).tolist()
