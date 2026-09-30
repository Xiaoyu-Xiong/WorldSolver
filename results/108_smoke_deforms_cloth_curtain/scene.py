"""Deterministic solver-neutral contract for a smoke-driven hanging curtain."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        self._positions, self._faces, self._uv = self._make_cloth()

    def _make_cloth(self):
        cloth = self.config["cloth"]
        nx, nz = map(int, cloth["grid_shape"])
        ys = np.linspace(-0.5 * cloth["width"], 0.5 * cloth["width"], nx)
        zs = np.linspace(cloth["top_height"], cloth["top_height"] - cloth["height"], nz)
        yy, zz = np.meshgrid(ys, zs, indexing="xy")
        position = np.column_stack((np.zeros(yy.size), yy.ravel(), zz.ravel()))
        faces = []
        for j in range(nz - 1):
            for i in range(nx - 1):
                a = j * nx + i
                faces.extend(((a, a + nx, a + nx + 1), (a, a + nx + 1, a + 1)))
        uv = np.column_stack((np.tile(np.linspace(0.0, 1.0, nx), nz),
                              np.repeat(np.linspace(0.0, 1.0, nz), nx)))
        return position, np.asarray(faces, dtype=np.int32), uv

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "two-way coupled smoke flow and impermeable hanging cloth",
            "topology": {"faces": self._faces.copy(), "grid_shape": tuple(self.config["cloth"]["grid_shape"])},
            "pinned_vertices": tuple(self.config["cloth"]["pinned_vertices"]),
            "free_edges": ["left", "right", "bottom"],
            "field_layout": "cell_centered_xyz",
            "required_fields": ["smoke_density", "fluid_velocity"],
            "coupling_requirements": dict(self.config["coupling_requirements"]),
        }

    def initial_state(self) -> StateFrame:
        domain, jet = self.config["domain"], self.config["jet"]
        shape = tuple(map(int, domain["grid_shape"]))
        low, high = np.asarray(domain["bounds_min"], float), np.asarray(domain["bounds_max"], float)
        spacing = (high - low) / np.asarray(shape)
        axes = [low[k] + (np.arange(shape[k]) + 0.5) * spacing[k] for k in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        smoke = np.zeros(shape, dtype=np.float32)
        velocity_field = np.zeros(shape + (3,), dtype=np.float32)
        # A short smoke plug already inside the nozzle; prescribed emission starts later.
        plug = ((x >= low[0]) & (x <= jet["center"][0] + 0.18) &
                ((y - jet["center"][1]) ** 2 + (z - jet["center"][2]) ** 2 <= jet["radius"] ** 2))
        smoke[plug] = float(jet["smoke_density"])
        mass = np.full(len(self._positions),
                       float(self.config["cloth"]["surface_density"]) *
                       float(self.config["cloth"]["width"]) * float(self.config["cloth"]["height"]) /
                       len(self._positions))
        frame = StateFrame(
            time=0.0,
            position=self._positions.copy(),
            velocity=np.tile(np.asarray(self.config["cloth"]["initial_velocity"], float), (len(self._positions), 1)),
            mass=mass,
            body_id=np.ones(len(self._positions), dtype=np.int32),
            attributes={"uv": self._uv.copy(), "pinned": np.isin(np.arange(len(self._positions)), self.config["cloth"]["pinned_vertices"]).astype(float)},
            fields={"smoke_density": smoke, "fluid_velocity": velocity_field},
            scalars={"jet_start_time": float(jet["start_time"]), "jet_speed": float(jet["speed"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(int(round(duration / dt)) + 1, dtype=float) * dt).tolist()
