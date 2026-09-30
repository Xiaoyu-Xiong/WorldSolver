"""Deterministic solver-neutral initial scene for repeated topological merging."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# Layered composition based on Figure 6: a broad low cluster and staggered arrivals.
# Each tuple is (x, y, z, sx, sy, sz, yaw_degrees, color_index).
BLOCKS = (
    (-1.55, -0.80, 0.53, 0.88, 0.66, 0.48, -18, 0),
    (-0.45, -1.05, 0.50, 0.86, 0.68, 0.46,  12, 1),
    ( 0.70, -0.95, 0.52, 0.90, 0.64, 0.48, -10, 2),
    ( 1.65, -0.52, 0.51, 0.78, 0.64, 0.46,  22, 3),
    (-1.35,  0.35, 0.56, 0.82, 0.62, 0.50,  28, 4),
    (-0.25,  0.28, 0.53, 0.90, 0.68, 0.48, -15, 5),
    ( 0.92,  0.26, 0.54, 0.84, 0.66, 0.49,  17, 6),
    ( 1.55,  0.88, 0.52, 0.76, 0.62, 0.46, -25, 0),
    (-0.90,  1.08, 0.55, 0.82, 0.62, 0.48,   7, 2),
    ( 0.18,  1.12, 0.54, 0.86, 0.64, 0.48, -20, 3),
    (-1.05, -0.38, 1.48, 0.80, 0.62, 0.48,  20, 1),
    ( 0.05, -0.45, 1.52, 0.88, 0.66, 0.50, -12, 4),
    ( 1.10, -0.18, 1.50, 0.82, 0.63, 0.48,  26, 5),
    (-0.62,  0.58, 1.55, 0.86, 0.66, 0.50, -27, 6),
    ( 0.55,  0.67, 1.54, 0.86, 0.64, 0.49,  11, 0),
    (-1.15, -0.05, 2.55, 0.76, 0.60, 0.46, -16, 3),
    ( 0.00, -0.20, 2.72, 0.84, 0.62, 0.48,  23, 2),
    ( 1.10,  0.18, 2.92, 0.78, 0.60, 0.47, -19, 1),
    (-0.55,  0.65, 3.32, 0.80, 0.62, 0.48,   9, 5),
    ( 0.62,  0.72, 3.72, 0.78, 0.60, 0.46, -24, 4),
    (-1.05, -0.12, 4.45, 0.76, 0.60, 0.46,  18, 6),
    ( 0.18, -0.25, 5.18, 0.80, 0.60, 0.47, -10, 0),
    ( 0.92,  0.36, 6.18, 0.76, 0.58, 0.45,  22, 3),
    (-0.38,  0.20, 7.38, 0.78, 0.60, 0.46, -21, 2),
)


def _rounded_block(center, scale, yaw, spacing=0.205):
    """Regular interior samples clipped to a rounded-superellipsoid silhouette."""
    axes = [np.arange(-s + spacing * 0.5, s, spacing) for s in scale]
    local = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    # Power four gives flat faces and strongly rounded edges, like the paper pieces.
    keep = np.sum((np.abs(local) / np.asarray(scale)) ** 4, axis=1) <= 1.0
    local = local[keep]
    a = np.deg2rad(yaw)
    rotation = np.array(((np.cos(a), -np.sin(a), 0.0),
                         (np.sin(a),  np.cos(a), 0.0),
                         (0.0, 0.0, 1.0)))
    return local @ rotation.T + np.asarray(center)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "repeated_topological_merge",
            "initial_group_count": len(BLOCKS),
            "up_axis": "z",
        }

    def initial_state(self) -> StateFrame:
        clouds, ids, colors = [], [], []
        for body_id, (x, y, z, sx, sy, sz, yaw, color) in enumerate(BLOCKS):
            points = _rounded_block((x, y, z), (sx, sy, sz), yaw)
            clouds.append(points)
            ids.append(np.full(len(points), body_id, dtype=np.int32))
            colors.append(np.full(len(points), color, dtype=np.float64))
        position = np.concatenate(clouds)
        body_id = np.concatenate(ids)
        color_id = np.concatenate(colors)
        particle_volume = 0.205 ** 3
        density = float(self.config["initial_conditions"]["rest_density"])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=np.full(len(position), density * particle_volume),
            body_id=body_id,
            attributes={
                "initial_group_id": body_id.astype(np.float64),
                "color_id": color_id,
                "particle_radius": np.full(len(position), 0.112),
            },
            scalars={
                "gravity_x": 0.0,
                "gravity_y": 0.0,
                "gravity_z": -9.81,
                "particle_spacing": 0.205,
                "rest_density": density,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
