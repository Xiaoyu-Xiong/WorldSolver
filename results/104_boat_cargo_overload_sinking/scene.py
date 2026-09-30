"""Deterministic initial state for the coupled boat, cargo, and water scene."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict): self.config = config
    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"], "gravity": self.config["gravity"], "coupling": "two_way_fluid_rigid"}
    def initial_state(self) -> StateFrame:
        ic = self.config["initial_conditions"]; h = float(ic["particle_spacing"])
        depth = float(self.config["fluid"]["initial_water_depth"])
        xs = np.arange(-2.3 + h/2, 2.3, h); ys = np.arange(-1.3 + h/2, 1.3, h); zs = np.arange(h/2, depth, h)
        g = np.meshgrid(xs, ys, zs, indexing="ij"); water = np.column_stack([a.ravel() for a in g])
        rigid = np.asarray([ic["boat_position"], ic["light_sphere_position"], ic["heavy_sphere_position"]], dtype=float)
        pos = np.vstack([water, rigid]); vel = np.zeros_like(pos)
        n = len(water); body_id = np.concatenate([np.zeros(n, dtype=int), np.arange(1,4)])
        mass = np.concatenate([np.full(n, 1000*h**3), [ic["boat_mass"], 4/3*np.pi*ic["sphere_radius"]**3*ic["light_density"], 4/3*np.pi*ic["sphere_radius"]**3*ic["heavy_density"]]])
        frame = StateFrame(0.0, pos, vel, mass=mass, body_id=body_id, attributes={"phase": body_id.copy()}); frame.validate(); return frame
    def sample_times(self) -> list[float]:
        count = int(round(self.config["duration"] / self.config["sample_dt"]))
        return (np.arange(count + 1, dtype=float) * float(self.config["sample_dt"])).tolist()
