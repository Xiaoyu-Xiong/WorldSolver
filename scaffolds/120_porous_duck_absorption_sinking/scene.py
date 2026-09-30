"""Deterministic solver-neutral porous-duck initial state."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _grid(lo, hi, spacing):
    axes = [np.arange(a, b + spacing * 0.2, spacing, dtype=np.float64)
            for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _duck_contains(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    body = ((x + .12) / .60) ** 2 + (y / .23) ** 2 + ((z - 1.05) / .34) ** 2 <= 1
    head = ((x - .28) / .30) ** 2 + (y / .20) ** 2 + ((z - 1.48) / .28) ** 2 <= 1
    neck = ((x - .13) / .23) ** 2 + (y / .19) ** 2 + ((z - 1.30) / .30) ** 2 <= 1
    tail = ((x + .61) / .24) ** 2 + (y / .18) ** 2 + ((z - 1.25) / .25) ** 2 <= 1
    beak = (x >= .48) & (x <= .70) & (abs(y) <= .13) & (z >= 1.40) & (z <= 1.52)
    return body | head | neck | tail | beak


def _obj_vertices(path):
    vertices = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "phenomenon": "two_way_coupled_porous_duck_absorption_and_sinking",
            "permeability_switch": self.config["permeability_control"],
            "coupling_requirement": (
                "Conserved free liquid enters pore space after the switch; fluid forces alter "
                "duck motion/deformation and the evolving duck alters resolved fluid flow."
            ),
            "supports": "static tank contact only",
            "prescribed_duck_motion": False,
        }

    def initial_state(self) -> StateFrame:
        spacing = 0.065
        duck_surface = _obj_vertices(Path(__file__).parent / "assets" / "duck.obj")
        candidates = _grid((-.72, -.22, .70), (.70, .22, 1.66), spacing)
        porous = candidates[_duck_contains(candidates)]
        water = _grid((-1.57, -.67, .03), (1.57, .67, .79), spacing)
        # Remove free-water samples initially occupied by the floating duck.
        water = water[~_duck_contains(water)]
        ns, npore, nw = len(duck_surface), len(porous), len(water)
        position = np.concatenate((duck_surface, porous, water))
        body_id = np.concatenate((np.full(ns, 1, np.int32),
                                  np.full(npore, 3, np.int32),
                                  np.full(nw, 2, np.int32)))
        phase = np.concatenate((np.zeros(ns + npore), np.ones(nw)))
        saturation = np.concatenate((np.zeros(ns + npore), np.ones(nw)))
        cell_volume = spacing ** 3
        dry_density = float(self.config["duck"]["dry_bulk_density_kg_m3"])
        mass = np.concatenate((np.full(ns, .003),
                               np.full(npore, dry_density * cell_volume),
                               np.full(nw, 1000.0 * cell_volume)))
        frame = StateFrame(
            0.0, position, np.zeros_like(position), mass=mass, body_id=body_id,
            attributes={
                "phase": phase,
                "saturation": saturation,
                "porosity": np.where(np.isin(body_id, [1, 3]), .62, 0.0),
                "permeability_m2": np.zeros(len(position)),
                "pore_water_mass": np.zeros(len(position)),
                "pore_pressure": np.zeros(len(position)),
                "solid_velocity": np.zeros_like(position),
            },
            scalars={
                "gravity_z": -9.81,
                "permeability_switch_time": 2.0,
                "initial_water_surface_z": .82,
                "initial_free_water_mass": float(nw * 1000.0 * cell_volume),
                "initial_pore_water_mass": 0.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
