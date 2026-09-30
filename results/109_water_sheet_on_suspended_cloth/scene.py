"""Deterministic solver-neutral four-corner suspended cloth and water scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config
        self._cloth, self._faces, self._uv = self._make_cloth()

    def _make_cloth(self):
        cfg = self.config["cloth"]
        nx, ny = map(int, cfg["grid_shape"])
        width, depth = map(float, cfg["size"])
        center = np.asarray(cfg["center"], dtype=float)
        xs = np.linspace(-0.5 * width, 0.5 * width, nx)
        ys = np.linspace(-0.5 * depth, 0.5 * depth, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        position = center + np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                faces.extend(((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)))
        uv = np.column_stack((np.tile(np.linspace(0.0, 1.0, nx), ny),
                              np.repeat(np.linspace(0.0, 1.0, ny), nx)))
        return position, np.asarray(faces, dtype=np.int32), uv

    @staticmethod
    def _lattice(lo, hi, spacing):
        axes = [np.arange(a, b + 0.25 * spacing, spacing) for a, b in zip(lo, hi)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "two-way coupled water sheets over cloth suspended at four corners",
            "topology": {"faces": self._faces.tolist(), "grid_shape": tuple(self.config["cloth"]["grid_shape"])},
            "pinned_vertices": tuple(self.config["cloth"]["pinned_vertices"]),
            "free_edges": ["north", "south", "east", "west"],
            "inflow": dict(self.config["inflow"]),
            "boundary_conditions": dict(self.config["boundary_conditions"]),
            "coupling_requirements": dict(self.config["coupling_requirements"]),
        }

    def initial_state(self) -> StateFrame:
        water_cfg, inflow = self.config["water"], self.config["inflow"]
        spacing = float(water_cfg["initial_particle_spacing"])
        pool = self._lattice((-2.00, -1.68, 0.06),
                             (2.00, 1.68, float(water_cfg["initial_receiving_depth"])), spacing)
        c = np.asarray(inflow["center"], dtype=float)
        jet_box = self._lattice((c[0] - inflow["radius"], c[1] - inflow["radius"], inflow["initial_column_bottom"]),
                                (c[0] + inflow["radius"], c[1] + inflow["radius"], c[2]), spacing)
        jet = jet_box[((jet_box[:, 0] - c[0]) ** 2 + (jet_box[:, 1] - c[1]) ** 2) <= inflow["radius"] ** 2]
        water = np.vstack((pool, jet))
        position = np.vstack((self._cloth, water))
        velocity = np.zeros_like(position)
        velocity[len(self._cloth) + len(pool):] = np.asarray(inflow["axis"], float) * float(inflow["speed"])
        body_id = np.r_[np.zeros(len(self._cloth), dtype=np.int32), np.ones(len(water), dtype=np.int32)]
        mass = np.r_[np.full(len(self._cloth), float(self.config["cloth"]["total_mass"]) / len(self._cloth)),
                     np.full(len(water), float(water_cfg["density"]) * spacing ** 3)]
        pinned = np.zeros(len(position), dtype=float)
        pinned[np.asarray(self.config["cloth"]["pinned_vertices"], dtype=int)] = 1.0
        phase = body_id.astype(float)
        frame = StateFrame(time=0.0, position=position, velocity=velocity, mass=mass, body_id=body_id,
                           attributes={"uv": np.vstack((self._uv, np.zeros((len(water), 2)))),
                                       "pinned": pinned, "material_phase": phase},
                           scalars={"water_density": float(water_cfg["density"]),
                                    "cloth_thickness": float(self.config["cloth"]["thickness"]),
                                    "nominal_inflow_speed": float(inflow["speed"]),
                                    "nozzle_radius": float(inflow["radius"])})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(int(round(duration / dt)) + 1, dtype=float) * dt).tolist()
