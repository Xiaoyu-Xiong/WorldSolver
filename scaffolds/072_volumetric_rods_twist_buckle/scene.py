"""Deterministic solver-neutral contract for seven twisting volumetric rods."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _local_surface(length: float, radius: float, nx: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangulated rest surface of a solid rod aligned with +x."""
    x = np.linspace(-0.5 * length, 0.5 * length, nx + 1)
    theta = 2.0 * np.pi * np.arange(nt) / nt
    vertices = np.array([[xx, radius * np.cos(t), radius * np.sin(t)] for xx in x for t in theta])
    faces: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(nt):
            a, b = i * nt + j, i * nt + (j + 1) % nt
            c, d = (i + 1) * nt + j, (i + 1) * nt + (j + 1) % nt
            faces.extend(((a, c, d), (a, d, b)))
    left, right = len(vertices), len(vertices) + 1
    vertices = np.vstack((vertices, [-0.5 * length, 0.0, 0.0], [0.5 * length, 0.0, 0.0]))
    for j in range(nt):
        faces.extend(((left, (j + 1) % nt, j), (right, nx * nt + j, nx * nt + (j + 1) % nt)))
    return vertices, np.asarray(faces, dtype=np.int32)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        bc = self.config["boundary_conditions"]
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "phenomenon": "twist_induced_buckling_of_separate_volumetric_rods",
                "spatial_dimensions": 3, "rod_count": 7,
                "driven_end_angular_speeds_deg_s": [bc["left_end_angular_speed_deg_s"], bc["right_end_angular_speed_deg_s"]]}

    def initial_state(self) -> StateFrame:
        g, bc = self.config["geometry"], self.config["boundary_conditions"]
        local, _ = _local_surface(float(g["rod_length"]), float(g["rod_radius"]), int(g["axial_segments"]), int(g["radial_segments"]))
        ring_r = float(g["bundle_ring_radius"])
        offsets = [(0.0, 0.0)] + [(ring_r * np.cos(k * np.pi / 3), ring_r * np.sin(k * np.pi / 3)) for k in range(6)]
        positions = np.vstack([local + [0.0, y, z] for y, z in offsets])
        n = len(local)
        body_id = np.repeat(np.arange(7, dtype=np.int32), n)
        end_len = float(bc["driven_end_length"])
        left = np.tile(local[:, 0] <= -0.5 * float(g["rod_length"]) + end_len, 7).astype(np.int32)
        right = np.tile(local[:, 0] >= 0.5 * float(g["rod_length"]) - end_len, 7).astype(np.int32)
        density = float(self.config["material"]["density"])
        mass = np.full(len(positions), density * np.pi * float(g["rod_radius"])**2 * float(g["rod_length"]) / n)
        frame = StateFrame(0.0, positions, np.zeros_like(positions), mass=mass, body_id=body_id,
                           attributes={"rest_position": positions.copy(), "driven_left": left, "driven_right": right})
        frame.validate()
        return frame

    def end_rotation_radians(self, time: float) -> tuple[float, float]:
        bc = self.config["boundary_conditions"]
        t = np.clip(float(time), float(bc["start_time"]), float(bc["stop_time"])) - float(bc["start_time"])
        return tuple(np.deg2rad(float(bc[k])) * t for k in ("left_end_angular_speed_deg_s", "right_end_angular_speed_deg_s"))

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration / dt) + 1, dtype=float) * dt).tolist()


def _write_asset(path: Path) -> None:
    vertices, faces = _local_surface(6.0, 0.09, 48, 10)
    lines = ["# deterministic closed rod surface; coordinates in meters"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    _write_asset(Path(__file__).parent / "assets" / "rod_surface.obj")
