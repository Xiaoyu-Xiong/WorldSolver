"""Deterministic solver-neutral scene contract for the soft-torus water entry."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from worldsolver.contracts import StateFrame


MAJOR_RADIUS = 0.255
MINOR_RADIUS = 0.105
TORUS_CENTER = np.array([0.0, 0.0, 0.66], dtype=float)


def torus_surface(n_major: int = 48, n_minor: int = 20) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for i in range(n_major):
        u = 2.0 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2.0 * math.pi * j / n_minor
            r = MAJOR_RADIUS + MINOR_RADIUS * math.cos(v)
            vertices.append([r * math.cos(u), r * math.sin(u), MINOR_RADIUS * math.sin(v)])
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            faces.extend([[a, b, c], [a, c, d]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


def water_samples(spacing: float = 0.04) -> np.ndarray:
    x = np.arange(-0.48 + spacing / 2, 0.48, spacing)
    y = np.arange(-0.48 + spacing / 2, 0.48, spacing)
    z = np.arange(spacing / 2, 0.40, spacing)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "two_way_coupled_deformable_water_entry_and_floating",
            "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
        }

    def initial_state(self) -> StateFrame:
        torus, _ = torus_surface()
        torus = torus + TORUS_CENTER
        water = water_samples()
        position = np.vstack([torus, water])
        velocity = np.zeros_like(position)
        velocity[: len(torus), 2] = -0.15
        body_id = np.concatenate([
            np.full(len(torus), 1, dtype=np.int32),
            np.full(len(water), 2, dtype=np.int32),
        ])
        torus_volume = 2.0 * math.pi**2 * MAJOR_RADIUS * MINOR_RADIUS**2
        mass = np.concatenate([
            np.full(len(torus), 650.0 * torus_volume / len(torus)),
            np.full(len(water), 1000.0 * 0.04**3),
        ])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={"rest_position": position.copy()},
            scalars={"water_free_surface_z": 0.40},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()


def write_torus_asset(path: Path) -> None:
    vertices, faces = torus_surface()
    lines = ["# Deterministic soft torus rest surface; meters"]
    lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    write_torus_asset(Path(__file__).parent / "assets" / "soft_torus.obj")
