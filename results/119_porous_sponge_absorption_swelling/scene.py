"""Deterministic, solver-neutral initial state for sponge imbibition."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _grid(bounds_min, bounds_max, spacing):
    axes = [np.arange(lo, hi + 0.25 * spacing, spacing, dtype=np.float64)
            for lo, hi in zip(bounds_min, bounds_max)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _box_surface(bounds_min, bounds_max, shape):
    axes = [np.linspace(lo, hi, count, dtype=np.float64)
            for lo, hi, count in zip(bounds_min, bounds_max, shape)]
    points = []
    for i, x in enumerate(axes[0]):
        for j, y in enumerate(axes[1]):
            for k, z in enumerate(axes[2]):
                if i in (0, shape[0] - 1) or j in (0, shape[1] - 1) or k in (0, shape[2] - 1):
                    points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"], "seed": self.config["seed"],
            "phenomenon": "two_way_poro_elasto_capillary_swelling",
            "gravity": [0.0, 0.0, -9.81], "external_actuation": False,
            "support": "floor_only", "particle_spacing": 0.06,
            "presentation_surface_shape": [17, 13, 13],
            "reference_particle_size": 0.03, "initial_porosity": 0.9,
            "intrinsic_permeability": 3.0e-8,
            "darcy_diagnostic": {"required_interval": [1.5, 5.5],
                                  "include_solid_velocity": True,
                                  "regime": "low_pore_reynolds_number"},
        }

    def initial_state(self) -> StateFrame:
        surface = _box_surface((-1.25, -.65, 0.0), (.25, .65, 1.3), (17, 13, 13))
        porous = _grid((-1.22, -.62, .03), (.22, .62, 1.27), .06)
        water = _grid((.28, -.72, .03), (1.72, .72, 1.11), .06)
        position = np.concatenate((surface, water, porous))
        ns, nw, npore = len(surface), len(water), len(porous)
        body_id = np.concatenate((np.full(ns, 1, np.int32), np.full(nw, 2, np.int32),
                                  np.full(npore, 3, np.int32)))
        saturation = np.concatenate((np.zeros(ns), np.ones(nw), np.zeros(npore)))
        frame = StateFrame(
            0.0, position, np.zeros_like(position),
            mass=np.concatenate((np.full(ns, .64 / ns), np.full(nw, .216), np.full(npore, .035))),
            body_id=body_id,
            attributes={
                "phase": np.concatenate((np.zeros(ns), np.ones(nw), np.zeros(npore))),
                "saturation": saturation,
                "porosity": np.where(body_id == 3, .9, 0.0),
                "pore_pressure": np.zeros(len(position)),
                "darcy_flux": np.zeros((len(position), 3)),
                "solid_velocity": np.zeros((len(position), 3)),
            },
            scalars={"gravity_z": -9.81, "fluid_volume_initial": nw * .06**3},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
