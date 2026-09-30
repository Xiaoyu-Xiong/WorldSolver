"""Deterministic solver-neutral wet porous plug and funnel scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vertices, dtype=float)


def _filled_lattice(bounds_min, bounds_max, spacing: float) -> np.ndarray:
    axes = [np.arange(lo + 0.5 * spacing, hi, spacing) for lo, hi in zip(bounds_min, bounds_max)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        asset = Path(__file__).resolve().parent / "assets" / "porous_plug.obj"
        self.plug_vertices = _obj_vertices(asset)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "scenario": "two_way_coupled_wetting_softening_and_funnel_passage",
            "gravity": list(self.config["gravity"]),
            "controls": "gravity_and_initial_liquid_release_only",
            "required_coupling": "fluid_forces_affect_solid_and_solid_motion_affects_fluid",
        }

    def initial_state(self) -> StateFrame:
        spacing = float(self.config["particle_spacing"])
        liquid_cfg = self.config["liquid"]
        water = _filled_lattice(liquid_cfg["bounds_min"], liquid_cfg["bounds_max"], spacing)
        plug = self.plug_vertices.copy()
        position = np.vstack((plug, water))
        velocity = np.zeros_like(position)
        velocity[len(plug):] = np.asarray(liquid_cfg["initial_velocity"], dtype=float)
        body_id = np.concatenate((np.ones(len(plug), dtype=np.int64), np.full(len(water), 2, dtype=np.int64)))
        plug_mass = np.full(len(plug), float(self.config["plug"]["mass"]) / len(plug))
        water_mass = np.full(len(water), float(liquid_cfg["density"]) * spacing ** 3)
        phase = np.concatenate((np.zeros(len(plug)), np.ones(len(water))))
        saturation = np.concatenate((np.full(len(plug), float(self.config["plug"]["initial_saturation"])), np.ones(len(water))))
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=np.concatenate((plug_mass, water_mass)),
            body_id=body_id,
            attributes={
                "phase_id": phase,
                "saturation": saturation,
                "porosity": np.concatenate((np.full(len(plug), float(self.config["plug"]["porosity"])), np.zeros(len(water)))),
                "deformation_gradient": np.tile(np.eye(3), (len(position), 1, 1)),
            },
            fields={"external_force": np.concatenate((plug_mass, water_mass))[:, None] * np.asarray(self.config["gravity"])[None, :]},
            scalars={
                "dry_young_modulus": float(self.config["plug"]["dry_young_modulus"]),
                "wet_softening_factor": float(self.config["plug"]["wet_softening_factor"]),
                "permeability": float(self.config["plug"]["permeability"]),
                "throat_radius": float(self.config["funnel"]["throat_radius"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
