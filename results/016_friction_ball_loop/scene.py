"""Deterministic solver-neutral scene definition for the friction ball loop."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {"scene_id": self.config["scene_id"], "seed": self.config["seed"],
                "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
                "friction_coefficient": self.config.get("friction_coefficient", 0.18)}

    def initial_state(self) -> StateFrame:
        # Ball rides the inner rail of the vertical loop ring (loop.stl at
        # [0,0,1.2], major R=1.0, minor t=0.2). Ball-center rail radius is
        # rho = R - t - r_b = 0.56, so the ball contacts the ring's inner
        # surface (R - t = 0.8) exactly all the way around.
        #
        # Initial pose: bottom of the rail circle, already rolling tangent.
        # Speed v is above the loop-contact minimum for a rolling ball
        # (v_bottom^2 >= (27/7) g rho), so the first pass stays on the
        # track; friction then dissipates energy so a later pass fails.
        r_b = 0.24
        v = 5.5
        position = np.array([[0.0, 0.0, 0.64]], dtype=float)  # loop bottom, on rail
        velocity = np.array([[v, 0.0, 0.0]], dtype=float)     # +x, up the right side
        orientation = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        angular_velocity = np.array([[0.0, -v / r_b, 0.0]], dtype=float)
        frame = StateFrame(0.0, position, velocity, orientation=orientation,
                          angular_velocity=angular_velocity, mass=np.array([1.0]),
                          body_id=np.array([0], dtype=int),
                          attributes={"rolling": np.array([1.0])},
                          scalars={"gravity_z": -9.81})
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
