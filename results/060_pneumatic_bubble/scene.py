"""Deterministic solver-neutral definition of the pneumatic bubble chamber."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def chamber_mesh(n_theta: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed surface of a thick, open-necked axisymmetric chamber wall."""
    z = np.array([0.0, 0.018, 0.070, 0.145, 0.225, 0.270, 0.292])
    outer_r = np.array([0.058, 0.066, 0.061, 0.054, 0.047, 0.041, 0.040])
    inner_r = np.array([0.043, 0.052, 0.049, 0.042, 0.035, 0.029, 0.028])
    theta = np.arange(n_theta) * (2.0 * np.pi / n_theta)
    rings = []
    for radii in (outer_r, inner_r):
        for radius, height in zip(radii, z):
            rings.append(np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(n_theta, height))))
    vertices = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []
    nr = len(z)
    def quad(a: int, b: int, c: int, d: int, reverse: bool = False) -> None:
        faces.extend(((a, c, b), (a, d, c)) if reverse else ((a, b, c), (a, c, d)))
    for side in range(2):
        base = side * nr * n_theta
        for j in range(nr - 1):
            for i in range(n_theta):
                k = (i + 1) % n_theta
                quad(base+j*n_theta+i, base+j*n_theta+k, base+(j+1)*n_theta+k, base+(j+1)*n_theta+i, side == 1)
    inner = nr * n_theta
    for i in range(n_theta):
        k = (i + 1) % n_theta
        quad((nr-1)*n_theta+i, (nr-1)*n_theta+k, inner+(nr-1)*n_theta+k, inner+(nr-1)*n_theta+i)
        quad(i, inner+i, inner+k, k)
    fixed = np.zeros(len(vertices), dtype=np.float64)
    ring_index = np.tile(np.repeat(np.arange(nr), n_theta), 2)
    fixed[(ring_index == 0) | (ring_index == nr-1)] = 1.0
    return vertices, np.asarray(faces, dtype=np.int32), fixed


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"], "seed": self.config["seed"],
            "phenomenon": "pressure-driven inflation of a volumetric rubber chamber",
            "pressure_pa": float(self.config["loading"]["pressure_pa"]),
        }

    def initial_state(self) -> StateFrame:
        position, faces, fixed = chamber_mesh()
        n = len(position)
        frame = StateFrame(
            time=0.0, position=position, velocity=np.zeros_like(position),
            mass=np.full(n, 0.012 / n), body_id=np.zeros(n, dtype=np.int32),
            attributes={"fixed": fixed}, fields={"surface_faces": faces},
            scalars={"internal_pressure_pa": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()


def write_asset(path: Path) -> None:
    vertices, faces, fixed = chamber_mesh()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=vertices, faces=faces, fixed=fixed)


if __name__ == "__main__":
    write_asset(Path(__file__).with_name("assets") / "chamber_wall.npz")
