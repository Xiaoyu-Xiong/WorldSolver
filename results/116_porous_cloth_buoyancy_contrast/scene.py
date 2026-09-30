"""Deterministic initial state for the porous-cloth buoyancy comparison."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Declare released porous cloth and quiescent water without solver choices."""

    def __init__(self, config: dict):
        self.config = config
        self._positions, self._faces, self._uv, self._body_id = self._make_cloth()

    def _make_cloth(self):
        cloth = self.config["cloth"]
        nx, ny = map(int, cloth["grid_shape"])
        width, depth = map(float, cloth["size"])
        x = np.linspace(-0.5 * width, 0.5 * width, nx)
        y = np.linspace(-0.5 * depth, 0.5 * depth, ny)
        local = np.asarray(
            [[px, py, 0.025 * np.sin(2.0 * np.pi * px / width) * np.cos(np.pi * py / depth)]
             for py in y for px in x], dtype=float
        )
        uv = np.asarray([[ix / (nx - 1), iy / (ny - 1)]
                         for iy in range(ny) for ix in range(nx)], dtype=float)
        faces = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                b, c, d = a + 1, a + nx, a + nx + 1
                faces.extend(((a, b, d), (a, d, c)))
        positions, body_ids, uvs = [], [], []
        for name in ("light", "neutral", "heavy"):
            center = np.asarray(cloth["initial_centers"][name], dtype=float)
            positions.append(local + center)
            body_ids.append(np.full(len(local), cloth["regimes"][name]["body_id"], dtype=np.int32))
            uvs.append(uv)
        return (np.vstack(positions), np.asarray(faces, dtype=np.int32),
                np.vstack(uvs), np.concatenate(body_ids))

    def _make_water(self):
        water = self.config["water"]
        spacing = 0.18
        low = np.asarray(water["initial_bounds_min"], dtype=float)
        high = np.asarray(water["initial_bounds_max"], dtype=float)
        axes = [np.arange(lo + 0.5 * spacing, hi, spacing) for lo, hi in zip(low, high)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3), spacing

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "gravity": self.config["gravity"],
            "tank": self.config["tank"],
            "water": self.config["water"],
            "cloth": self.config["cloth"],
            "coupling_requirements": self.config["coupling_requirements"],
            "cloth_topology": {"faces": self._faces.tolist(), "local_indexing_per_body": True},
        }

    def initial_state(self) -> StateFrame:
        cloth = self.config["cloth"]
        area_per_vertex = np.prod(np.asarray(cloth["size"], dtype=float)) / len(self._faces)
        densities = np.asarray([cloth["regimes"][name]["density"]
                                for name in ("light", "neutral", "heavy")], dtype=float)
        mass_by_body = densities * float(cloth["thickness"]) * area_per_vertex
        cloth_mass = mass_by_body[self._body_id - 1]
        water_position, water_spacing = self._make_water()
        water_mass = np.full(
            len(water_position),
            float(self.config["water"]["density"]) * water_spacing**3,
        )
        position = np.vstack((water_position, self._positions))
        body_id = np.concatenate((np.zeros(len(water_position), dtype=np.int32), self._body_id))
        mass = np.concatenate((water_mass, cloth_mass))
        velocity = np.vstack((
            np.tile(np.asarray(self.config["water"]["initial_velocity"], dtype=float), (len(water_position), 1)),
            np.tile(np.asarray(cloth["initial_velocity"], dtype=float), (len(cloth_mass), 1)),
        ))
        water_count = len(water_position)
        uv = np.vstack((np.zeros((water_count, 2)), self._uv))
        cloth_density = densities[self._body_id - 1]
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "uv": uv,
                "material_density": np.concatenate((
                    np.full(water_count, float(self.config["water"]["density"])),
                    cloth_density,
                )),
                "porosity": np.concatenate((
                    np.ones(water_count),
                    np.full(len(cloth_mass), float(cloth["porosity"])),
                )),
                "permeability_m2": np.concatenate((
                    np.zeros(water_count),
                    np.full(len(cloth_mass), float(cloth["permeability_m2"])),
                )),
                "saturation": np.ones(len(mass)),
            },
            fields={
                "water_initial_bounds": np.asarray([
                    self.config["water"]["initial_bounds_min"],
                    self.config["water"]["initial_bounds_max"],
                ], dtype=float),
                "water_initial_velocity": np.asarray([self.config["water"]["initial_velocity"]], dtype=float),
            },
            scalars={"water_density": float(self.config["water"]["density"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
