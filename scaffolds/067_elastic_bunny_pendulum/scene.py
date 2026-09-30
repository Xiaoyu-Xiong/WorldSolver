"""Deterministic, solver-neutral suspended elastic bunny scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


_ASSET = Path(__file__).resolve().parent / "assets" / "bunny.obj"
_ATTACHMENT = np.array([0.0, 0.0, 2.85], dtype=np.float64)


def _vertices(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
    if not points:
        raise ValueError(f"Bunny asset has no vertices: {path}")
    return np.asarray(points, dtype=np.float64)


def _initial_pose(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale and rigidly displace the bunny while keeping its crown patch fixed."""
    lo, hi = rest.min(axis=0), rest.max(axis=0)
    centered = rest - 0.5 * (lo + hi)
    centered *= 1.18 / (hi[2] - lo[2])

    # A 48-degree rotation about camera y produces the displaced release pose.
    angle = np.deg2rad(48.0)
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
         [-np.sin(angle), 0.0, np.cos(angle)]],
        dtype=np.float64,
    )
    posed = centered @ rotation.T
    crown = posed[:, 2] >= np.quantile(posed[:, 2], 0.992)
    posed += _ATTACHMENT - posed[crown].mean(axis=0)
    return posed, crown


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "gravity-driven swinging of a suspended 3D elastic bunny",
            "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
            "attachment_attribute": "fixed",
        }

    def initial_state(self) -> StateFrame:
        position, crown = _initial_pose(_vertices(_ASSET))
        n = len(position)
        # Equal nodal masses represent the paper's uniform density without imposing a discretization.
        mass = np.full(n, 1.0 / n, dtype=np.float64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={"fixed": crown.astype(np.float64)},
            scalars={"density_kg_m3": 1000.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=np.float64) * dt).tolist()
