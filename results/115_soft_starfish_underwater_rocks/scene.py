"""Deterministic solver-neutral scene contract for coupled starfish settling."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from worldsolver.contracts import StateFrame


STAR_CENTER = np.array([0.0, 0.0, 0.58], dtype=float)
WATER_LEVEL = 0.40


def starfish_surface(n_theta: int = 100, n_radial: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Closed, gently domed five-arm surface in local coordinates."""
    vertices = [[0.0, 0.0, 0.030], [0.0, 0.0, -0.030]]
    rings = []
    for j in range(1, n_radial + 1):
        s = j / n_radial
        ring = []
        for i in range(n_theta):
            a = 2.0 * math.pi * i / n_theta
            outer = 0.205 + 0.145 * math.cos(5.0 * a)
            r = s * outer
            half_t = 0.010 + 0.031 * (1.0 - s) ** 0.65
            top = len(vertices)
            vertices.append([r * math.cos(a), r * math.sin(a), half_t])
            vertices.append([r * math.cos(a), r * math.sin(a), -half_t])
            ring.append((top, top + 1))
        rings.append(ring)
    faces = []
    first = rings[0]
    for i in range(n_theta):
        k = (i + 1) % n_theta
        faces.extend([[0, first[i][0], first[k][0]], [1, first[k][1], first[i][1]]])
    for j in range(n_radial - 1):
        inner, outer = rings[j], rings[j + 1]
        for i in range(n_theta):
            k = (i + 1) % n_theta
            it, ib = inner[i]
            kt, kb = inner[k]
            ot, ob = outer[i]
            qt, qb = outer[k]
            faces.extend([[it, ot, qt], [it, qt, kt], [ib, kb, qb], [ib, qb, ob]])
    last = rings[-1]
    for i in range(n_theta):
        k = (i + 1) % n_theta
        t0, b0 = last[i]
        t1, b1 = last[k]
        faces.extend([[t0, b0, b1], [t0, b1, t1]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def water_samples(spacing: float = 0.04) -> np.ndarray:
    x = np.arange(-0.48 + spacing / 2, 0.48, spacing)
    y = np.arange(-0.48 + spacing / 2, 0.48, spacing)
    z = np.arange(0.08 + spacing / 2, WATER_LEVEL, spacing)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "two_way_coupled_soft_starfish_water_entry_and_rock_settling",
            "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
            "free_surface_z": WATER_LEVEL,
            "arms_unconstrained": True,
        }

    def initial_state(self) -> StateFrame:
        star, _ = starfish_surface()
        star = star + STAR_CENTER
        water = water_samples()
        position = np.vstack([star, water])
        velocity = np.zeros_like(position)
        velocity[:len(star), 2] = -0.10
        body_id = np.concatenate([
            np.full(len(star), 1, dtype=np.int32),
            np.full(len(water), 2, dtype=np.int32),
        ])
        # Lumped masses communicate density regime without selecting a solver.
        star_volume = 0.0115
        mass = np.concatenate([
            np.full(len(star), 1080.0 * star_volume / len(star)),
            np.full(len(water), 1000.0 * 0.04 ** 3),
        ])
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={"rest_position": position.copy()},
            scalars={"water_free_surface_z": WATER_LEVEL},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()


def write_starfish_asset(path: Path) -> None:
    vertices, faces = starfish_surface()
    lines = ["# Closed five-arm starfish rest surface; meters"]
    lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    write_starfish_asset(Path(__file__).parent / "assets" / "starfish.obj")
