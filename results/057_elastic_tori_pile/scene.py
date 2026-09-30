"""Deterministic solver-neutral scene for forty falling elastic solid tori."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from worldsolver.contracts import StateFrame


COUNT = 40
MAJOR_RADIUS = 0.46
TUBE_RADIUS = 0.18
VERTICES_U = 24
VERTICES_V = 12


def _unit_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return np.r_[axis * math.sin(angle / 2.0), math.cos(angle / 2.0)]


def _rotate(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    xyz, w = q[:3], q[3]
    return points + 2.0 * np.cross(xyz, np.cross(xyz, points) + w * points)


def _torus_vertices() -> np.ndarray:
    u = np.arange(VERTICES_U) * (2.0 * math.pi / VERTICES_U)
    v = np.arange(VERTICES_V) * (2.0 * math.pi / VERTICES_V)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = MAJOR_RADIUS + TUBE_RADIUS * np.cos(vv)
    return np.column_stack((
        (radial * np.cos(uu)).ravel(),
        (radial * np.sin(uu)).ravel(),
        (TUBE_RADIUS * np.sin(vv)).ravel(),
    ))


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "object_count": COUNT,
            "phenomenon": "gravity-driven pile of incompressible elastic volumetric tori",
            "gravity": [0.0, 0.0, -9.81],
            "rest_topology_asset": "assets/torus.obj",
            "material_intent": {"solid_dimension": 3, "incompressible": True},
        }

    def initial_state(self) -> StateFrame:
        local = _torus_vertices()
        positions, body_ids, orientations = [], [], []
        # Five rings per tier and eight tiers. Repeated azimuths keep a clear
        # 0.65 m vertical surface gap; tilt varies openings without intersections.
        for body in range(COUNT):
            tier, slot = divmod(body, 5)
            azimuth = 2.0 * math.pi * slot / 5.0 + 0.11 * tier
            center = np.array([
                1.22 * math.cos(azimuth),
                1.22 * math.sin(azimuth),
                3.00 + 0.66 * tier,
            ])
            tilt_axis = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
            tilt = math.radians((-1 if body % 2 else 1) * (8 + 3 * (body % 4)))
            q = _unit_quaternion(tilt_axis, tilt)
            world = _rotate(local, q) + center
            positions.append(world)
            body_ids.append(np.full(len(local), body, dtype=np.int32))
            orientations.append(np.repeat(q[None, :], len(local), axis=0))
        position = np.vstack(positions)
        body_id = np.concatenate(body_ids)
        vertex_mass = np.full(len(position), 0.90 / len(local), dtype=float)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            orientation=np.vstack(orientations),
            mass=vertex_mass,
            body_id=body_id,
            attributes={"rest_position": position.copy()},
            scalars={"gravity_z": -9.81, "torus_count": float(COUNT)},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()

