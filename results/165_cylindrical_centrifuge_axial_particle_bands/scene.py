"""Deterministic horizontal-centrifuge initial scene."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        c = self.config["centrifuge"]
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "family": self.config["family"], "axis": [1.0, 0.0, 0.0],
                "angular_velocity_rad_s": float(c["angular_velocity_rad_s"]),
                "fill_fraction": float(c["fill_fraction"])}

    def initial_state(self) -> StateFrame:
        c = self.config["centrifuge"]
        radius, length = float(c["radius"]), float(c["length"])
        h = float(c["spacing"])
        xs = np.arange(-length/2+h, length/2, h)
        ys = np.arange(-radius+h, radius, h)
        zs = np.arange(-radius+h, radius, h)
        pts = []
        fill = float(c["fill_fraction"])
        z_limit = -radius + min(1.0, fill) * 2.0 * radius
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                for iz, z in enumerate(zs):
                    if y*y + z*z < (radius-h*.55)**2 and (fill >= .999 or z <= z_limit):
                        pts.append((x, y, z + float(c["center_height"])))
        position = np.asarray(pts, dtype=float)
        count = len(position)
        species_count = int(c["species_count"])
        rng = np.random.default_rng(int(self.config["seed"]))
        species = rng.integers(0, species_count, count, dtype=np.int32) if species_count > 1 else np.zeros(count, np.int32)
        radii = np.where(species == 0, .38*h, .52*h)
        mass = float(c["particle_density"]) * (4*np.pi/3) * radii**3
        mass_quantity = np.zeros((count, species_count))
        mass_quantity[np.arange(count), species] = mass
        shape = tuple(c["field_grid_shape"])
        frame = StateFrame(
            0.0, position, np.zeros_like(position), mass=mass,
            body_id=np.zeros(count, dtype=np.int32),
            attributes={"species_id": species, "particle_radius": radii,
                        "mass_quantity": mass_quantity},
            fields={"cumulative_mass_source": np.zeros(species_count),
                    "velocity_field": np.zeros(shape + (3,)),
                    "active_mask": np.ones(shape, dtype=bool)},
            scalars={"grid_spacing": h, "characteristic_length": 2*radius,
                     "contact_length_scale": h})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration/dt)+1)*dt).tolist()
