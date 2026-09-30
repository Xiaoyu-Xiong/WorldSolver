"""Deterministic solver-neutral frigate and incoming-wave initial state."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.config["seed"],
            "coupling": "two_way_fluid_rigid",
            "dynamic_body": "frigate_hull",
            "free_response": ["heave", "pitch"],
            "prescribed_solid_motion": False,
        }

    def initial_state(self) -> StateFrame:
        fluid = self.config["fluid"]
        wave = self.config["incoming_waves"]
        ship = self.config["frigate"]
        h = float(fluid["marker_spacing"])
        lo = np.asarray(fluid["domain_min"], dtype=float)
        hi = np.asarray(fluid["domain_max"], dtype=float)
        xs = np.arange(lo[0] + 0.5 * h, hi[0], h)
        ys = np.arange(lo[1] + 0.5 * h, hi[1], h)
        zs = np.arange(lo[2] + 0.5 * h, float(fluid["still_water_level"]) + float(wave["amplitude"]) + h, h)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        surface = float(fluid["still_water_level"]) + float(wave["amplitude"]) * np.sin(
            2.0 * np.pi * xx / float(wave["wavelength"]) + float(wave["phase_radians"])
        )
        wet = zz <= surface
        markers = np.column_stack((xx[wet], yy[wet], zz[wet]))

        # Start with an empty impermeable hull volume. The downstream solver resolves
        # the evolving free surface and pressure/velocity exchange at this boundary.
        center = np.asarray(ship["initial_center"], dtype=float)
        q = markers - center
        half_width = 0.92 * np.clip(1.0 - np.abs(q[:, 0]) / 4.35, 0.0, 1.0) ** 0.35
        hull_inside = (
            (np.abs(q[:, 0]) < 4.35)
            & (np.abs(q[:, 1]) < half_width)
            & (q[:, 2] > -1.12 + 0.16 * np.abs(q[:, 0]) / 4.35)
            & (q[:, 2] < 0.42)
        )
        markers = markers[~hull_inside]

        position = np.vstack((center, markers))
        velocity = np.tile(np.asarray(fluid["initial_velocity"], dtype=float), (len(position), 1))
        velocity[0] = np.asarray(ship["initial_velocity"], dtype=float)
        n = len(position)
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        orientation[0] = np.asarray(ship["initial_orientation_xyzw"], dtype=float)
        angular_velocity = np.zeros((n, 3), dtype=float)
        angular_velocity[0] = np.asarray(ship["initial_angular_velocity"], dtype=float)
        mass = np.full(n, float(fluid["density"]) * h ** 3)
        mass[0] = float(ship["mass"])
        body_id = np.zeros(n, dtype=np.int64)
        body_id[0] = int(ship["body_id"])
        phase = np.ones(n, dtype=float)
        phase[0] = 0.0
        ship_visual = np.zeros(n, dtype=float)
        ship_visual[0] = 1.0
        return StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={"phase": phase, "ship_visual": ship_visual},
            scalars={"still_water_level": float(fluid["still_water_level"])},
        )

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
