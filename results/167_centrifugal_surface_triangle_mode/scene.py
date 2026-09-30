"""Deterministic rotating-bottom free-surface cylinder scaffold."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        tank = self.config["tank"]
        return {"scene_id": self.config["scene_id"], "seed": int(self.config["seed"]),
                "target_azimuthal_mode": int(tank["target_mode"]),
                "bottom_angular_velocity_rad_s": float(tank["bottom_angular_velocity_rad_s"]),
                "stationary_side_wall": True, "gravity": list(self.config["gravity"])}

    def initial_state(self) -> StateFrame:
        tank = self.config["tank"]
        radius, height = float(tank["radius"]), float(tank["initial_fill_height"])
        h = float(tank["particle_spacing"])
        ring_count = 48
        ring_radii = [0.24, 0.42, 0.59, 0.76, 0.94]
        mode = int(tank["target_mode"])
        def surface_z(ratio, angle):
            depression = 0.16 * (1.0 - ratio * ratio) ** 2
            seeded_mode = 0.035 * 4.0 * ratio * (1.0 - ratio) * np.cos(mode * angle)
            return height - depression - seeded_mode
        pts = [(0.0, 0.0, surface_z(0.0, 0.0))]
        for ring_index, ring_radius in enumerate(ring_radii):
            for k in range(ring_count):
                angle = 2.0 * np.pi * k / ring_count
                # Exact regular-polygon polar radius for the dry-core boundary.
                if ring_index == 0:
                    sector = (angle + np.pi / mode) % (2.0 * np.pi / mode) - np.pi / mode
                    ring_radius = 0.20 / np.cos(sector)
                pts.append((
                    ring_radius * radius * np.cos(angle),
                    ring_radius * radius * np.sin(angle),
                    surface_z(ring_radius, angle),
                ))
        position = np.asarray(pts, dtype=float)
        volume = h**3
        mass = np.full(len(position), float(tank["liquid_density"]) * volume)
        grid_shape = tuple(tank["field_grid_shape"])
        level_set = np.empty(grid_shape, dtype=float)
        gx = np.linspace(-radius, radius, grid_shape[0])
        gy = np.linspace(-radius, radius, grid_shape[1])
        gz = np.linspace(0.0, float(tank["wall_height"]), grid_shape[2])
        xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
        level_set[:] = np.maximum(np.sqrt(xx*xx+yy*yy)-radius, zz-height)
        interior = level_set < -1.5*h
        frame = StateFrame(
            0.0, position, np.zeros_like(position), mass=mass,
            body_id=np.zeros(len(position), dtype=np.int32),
            attributes={"mass_quantity": mass.copy(), "particle_radius": np.full(len(position), .48*h)},
            fields={"cumulative_mass_source": np.zeros(1), "liquid_level_set": level_set,
                    "liquid_interior_mask": interior,
                    "velocity_field": np.zeros(grid_shape+(3,))},
            scalars={"grid_spacing": h, "characteristic_length": 2*radius,
                     "reference_velocity": max(float(tank["bottom_angular_velocity_rad_s"])*radius, 1e-6)})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return (np.arange(round(duration/dt)+1)*dt).tolist()
