"""Deterministic solver-neutral scene for a buoyant cube dropped into water."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "up_axis": "z",
            "coupled_bodies": {"water": 1, "cube": 2},
            "scaffold_visual_intent": (
                "The bright cube starts above the water and in front of the tank centerline; "
                "the front wall is rendered as a cutaway so the cube, splash, and wake are visible."
            ),
        }

    def initial_state(self) -> StateFrame:
        water_cfg = self.config["initial_conditions"]["water"]
        cube_cfg = self.config["initial_conditions"]["cube"]

        # Regular samples define the initial liquid volume; they do not prescribe a solver.
        spacing = 0.12
        lo = np.asarray(water_cfg["fill_bounds_min"], dtype=float)
        hi = np.asarray(water_cfg["fill_bounds_max"], dtype=float)
        axes = [np.arange(lo[i] + spacing / 2, hi[i], spacing) for i in range(3)]
        grid = np.meshgrid(*axes, indexing="ij")
        water_position = np.stack(grid, axis=-1).reshape(-1, 3)
        water_velocity = np.repeat(
            np.asarray(water_cfg["initial_velocity"], dtype=float)[None, :],
            len(water_position), axis=0,
        )

        cube_position = np.asarray(cube_cfg["initial_position"], dtype=float)[None, :]
        cube_velocity = np.asarray(cube_cfg["initial_velocity"], dtype=float)[None, :]
        position = np.vstack((water_position, cube_position))
        velocity = np.vstack((water_velocity, cube_velocity))
        body_id = np.concatenate((np.ones(len(water_position), dtype=np.int32), [2]))

        water_volume = spacing ** 3
        cube_volume = float(cube_cfg["side_length"]) ** 3
        mass = np.concatenate((
            np.full(len(water_position), float(water_cfg["rest_density"]) * water_volume),
            [float(cube_cfg["density"]) * cube_volume],
        ))
        phase = np.concatenate((np.zeros(len(water_position)), [1.0]))
        orientation = np.zeros((len(position), 4), dtype=float)
        orientation[:, 3] = 1.0
        orientation[-1] = np.asarray(cube_cfg["initial_orientation_xyzw"], dtype=float)

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=np.zeros_like(position),
            mass=mass,
            body_id=body_id,
            attributes={"phase": phase},
            scalars={"water_rest_density": float(water_cfg["rest_density"]),
                     "cube_density": float(cube_cfg["density"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
