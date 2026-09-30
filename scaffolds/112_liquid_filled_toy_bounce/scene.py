"""Deterministic solver-neutral initial state for the liquid-filled toy bounce."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


RINGS = (
    (-0.78, 0.34), (-0.68, 0.54), (-0.56, 0.62), (-0.42, 0.72),
    (-0.12, 0.84), (0.22, 0.82), (0.47, 0.68), (0.60, 0.46),
    (0.68, 0.25), (0.82, 0.24), (0.92, 0.34), (1.02, 0.31),
)
NTHETA = 16
CENTER_Z = 1.36


def shell_vertices() -> np.ndarray:
    points = []
    for z, radius in RINGS:
        for j in range(NTHETA):
            a = 2.0 * np.pi * j / NTHETA
            # The source toy is broad in x and shallow in the viewing-depth direction.
            points.append((radius * np.cos(a), 0.64 * radius * np.sin(a), CENTER_Z + z))
    points.extend(((0.0, 0.0, CENTER_Z + RINGS[0][0]),
                   (0.0, 0.0, CENTER_Z + RINGS[-1][0])))
    return np.asarray(points, dtype=float)


def cavity_points(spacing: float = 0.105) -> np.ndarray:
    values = np.arange(-0.78, 0.781, spacing)
    pts = []
    for x in values:
        for y in np.arange(-0.48, 0.481, spacing):
            for z in np.arange(-0.50, 0.501, spacing):
                q = (x / 0.76) ** 2 + (y / 0.46) ** 2 + (z / 0.53) ** 2
                if q <= 0.88:
                    pts.append((x, y, CENTER_Z + z))
    return np.asarray(pts, dtype=float)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "two_way_coupled_closed_liquid_deformable_bounce",
            "notes": "Markers are passive liquid samples, not shell attachments.",
        }

    def initial_state(self) -> StateFrame:
        shell = shell_vertices()
        liquid = cavity_points()
        red = liquid[(liquid[:, 0] < -0.14) & (liquid[:, 2] > 1.08) & (liquid[:, 2] < 1.72)][::2]
        yellow = liquid[(liquid[:, 0] > 0.14) & (liquid[:, 2] > 1.08) & (liquid[:, 2] < 1.72)][1::2]
        pos = np.vstack((shell, liquid, red, yellow))
        body_id = np.concatenate((
            np.full(len(shell), 1, dtype=np.int32),
            np.full(len(liquid), 2, dtype=np.int32),
            np.full(len(red), 3, dtype=np.int32),
            np.full(len(yellow), 4, dtype=np.int32),
        ))
        vel = np.zeros_like(pos)
        vel[:, 2] = -1.15
        mass = np.concatenate((
            np.full(len(shell), 0.65 / len(shell)),
            np.full(len(liquid), 1.65 / len(liquid)),
            np.zeros(len(red)), np.zeros(len(yellow)),
        ))
        return StateFrame(
            0.0, pos, vel, mass=mass, body_id=body_id,
            attributes={"passive_marker": np.isin(body_id, (3, 4)).astype(float)},
            scalars={"gravity_z": -9.81, "cavity_is_closed": 1.0},
        )

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
