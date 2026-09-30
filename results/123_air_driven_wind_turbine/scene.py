"""Solver-neutral wind-turbine initial scene and boundary controls."""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        turbine = self.config["turbine"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "two_way_air_driven_wind_turbine_helical_wake",
            "spatial_dimension": 3,
            "reference": "Lyu et al. 2021, Figure 20, PDF page 13",
            "dynamic_body_id": int(turbine["rotor_body_id"]),
            "coupling": "two_way_fluid_rigid",
            "rotor_control": "unpowered_dynamic_revolute_joint",
            "required_outputs": [
                "air_density",
                "velocity_field",
                "smoke_red",
                "smoke_green",
                "smoke_ochre",
                "rotor_angle",
                "rotor_angular_velocity",
            ],
        }

    def inlet_velocity(self, time: float) -> np.ndarray:
        """Prescribed inflow control with a short smooth startup ramp."""
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and nonnegative")
        inlet = self.config["air"]["inlet"]
        ramp = float(inlet["ramp_duration"])
        start = float(inlet["start_time"])
        s = np.clip((time - start) / ramp, 0.0, 1.0)
        weight = s * s * (3.0 - 2.0 * s)
        return weight * np.asarray(inlet["velocity"], dtype=float)

    def initial_state(self) -> StateFrame:
        domain = self.config["domain"]
        turbine = self.config["turbine"]
        shape = tuple(int(value) for value in domain["reference_grid"])
        density = float(self.config["air"]["density"])
        hub = np.asarray(turbine["hub_center"], dtype=float)

        position = hub.reshape(1, 3)
        velocity = np.zeros((1, 3), dtype=float)
        orientation = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        angular_velocity = np.zeros((1, 3), dtype=float)
        body_id = np.array([int(turbine["rotor_body_id"])], dtype=np.int64)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=np.array([float(turbine["mass"])], dtype=float),
            body_id=body_id,
            attributes={
                "rotor_angle": np.array([float(turbine["initial_angle"])]),
            },
            fields={
                "air_density": np.full(shape, density, dtype=np.float32),
                "velocity_field": np.zeros((*shape, 3), dtype=np.float32),
                "smoke_red": np.zeros(shape, dtype=np.float32),
                "smoke_green": np.zeros(shape, dtype=np.float32),
                "smoke_ochre": np.zeros(shape, dtype=np.float32),
                "grid_origin": np.asarray(domain["bounds_min"], dtype=float),
                "rotor_local_tip_positions": np.asarray(
                    self.config["smoke"]["rotor_local_tip_positions"], dtype=float
                ),
            },
            scalars={
                "air_density": density,
                "inlet_speed": float(np.linalg.norm(self.config["air"]["inlet"]["velocity"])),
                "axle_viscous_friction": float(
                    turbine["axle_friction"]["viscous_coefficient"]
                ),
                "axle_coulomb_friction": float(
                    turbine["axle_friction"]["coulomb_torque"]
                ),
                "rotor_radius": float(turbine["rotor_radius"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        duration = float(self.config["duration"])
        dt = float(self.config["sample_dt"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
