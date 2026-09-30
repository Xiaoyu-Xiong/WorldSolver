"""Deterministic, solver-neutral rotating hairy-ball scene contract."""
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
            "phenomenon": "frictional strand-strand and strand-sphere collisions",
            "prescribed_sphere_rotation": self.config["prescribed_sphere_rotation"],
        }

    def initial_state(self) -> StateFrame:
        p = self.config["physical_parameters"]
        count, segs = int(p["strand_count"]), int(p["segments_per_strand"])
        ball_r, length = float(p["sphere_radius"]), float(p["strand_length"])
        curl_r, pitch = float(p["natural_helix_radius"]), float(p["natural_helix_pitch"])
        points = np.empty((count * (segs + 1), 3), dtype=float)
        strand_id = np.repeat(np.arange(count, dtype=float), segs + 1)
        root_mask = np.zeros(count * (segs + 1), dtype=float)
        root_mask[::segs + 1] = 1.0
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for strand in range(count):
            # Equal-area Fibonacci roots strictly over the upper hemisphere.
            z = (strand + 0.5) / count
            phi = strand * golden
            radial = np.sqrt(max(0.0, 1.0 - z * z))
            normal = np.array([radial * np.cos(phi), radial * np.sin(phi), z])
            root = ball_r * normal
            tangent = np.array([-np.sin(phi), np.cos(phi), 0.0])
            binormal = np.cross(normal, tangent)
            base = strand * (segs + 1)
            for j in range(segs + 1):
                s = length * j / segs
                # Hanging initial pose with subtle source-specified helical rest curl.
                fade = np.sin(np.pi * j / (2.0 * segs))
                phase = 2.0 * np.pi * s / pitch
                curl = curl_r * fade * ((np.cos(phase) - 1.0) * tangent + np.sin(phase) * binormal)
                points[base + j] = root + np.array([0.0, 0.0, -s]) + curl
        area = np.pi * float(p["strand_radius"]) ** 2
        vertex_mass = np.full(len(points), float(p["density"]) * area * length / segs)
        vertex_mass[root_mask > 0.5] = 0.0
        frame = StateFrame(
            time=0.0,
            position=points,
            velocity=np.zeros_like(points),
            mass=vertex_mass,
            attributes={"strand_id": strand_id, "root_mask": root_mask},
            scalars={
                "strand_radius": float(p["strand_radius"]),
                "static_friction": 0.2,
                "kinetic_friction": 0.2,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
