"""Deterministic solver-neutral scene for tightening a frictionless rod knot."""
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
            "phenomenon": "threading and tightening a frictionless elastic rod knot",
            "rod": self.config["rod"],
            "self_contact": self.config["self_contact"],
            "gravity": self.config["gravity"],
            "endpoint_manipulation": self.config["endpoint_manipulation"],
            "event_windows": self.config["event_windows"],
            "required_outcome": self.config["required_outcome"],
        }

    @staticmethod
    def _catmull_rom(points: np.ndarray, count: int) -> np.ndarray:
        """Sample a smooth open curve through authored knot-preparation points."""
        p = np.vstack((points[0], points, points[-1]))
        spans = len(points) - 1
        u = np.linspace(0.0, spans, count)
        i = np.minimum(np.floor(u).astype(int), spans - 1)
        t = (u - i)[:, None]
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        return 0.5 * ((2*p1) + (-p0+p2)*t +
                      (2*p0-5*p1+4*p2-p3)*t*t +
                      (-p0+3*p1-3*p2+p3)*t*t*t)

    def _centerline(self) -> np.ndarray:
        # Left tail enters a large upright loop; the raised free end is poised
        # in front of the loop aperture for the prescribed threading motion.
        controls = np.array([
            [-2.65,  0.00,  0.00], [-1.78,  0.00,  0.00],
            [-1.10, -0.03,  0.12], [-0.72,  0.02,  0.72],
            [-0.05,  0.12,  1.10], [ 0.68,  0.04,  0.72],
            [ 0.88, -0.05,  0.05], [ 0.42, -0.12, -0.48],
            [-0.34, -0.05, -0.58], [-0.92,  0.04, -0.14],
            [-0.62,  0.20,  0.36], [ 0.02,  0.32,  0.42],
            [ 0.74,  0.02,  0.44], [ 1.70, -0.42,  0.46],
        ], dtype=float)
        return self._catmull_rom(controls, int(self.config["rod"]["vertex_count"]))

    def initial_state(self) -> StateFrame:
        position = self._centerline()
        n = len(position)
        radius = float(self.config["rod"]["radius"])
        frame = StateFrame(
            0.0,
            position,
            np.zeros_like(position),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            angular_velocity=np.zeros_like(position),
            mass=np.full(n, float(self.config["rod"]["total_mass"]) / n),
            attributes={
                "arc_fraction": np.linspace(0.0, 1.0, n),
                "rod_radius": np.full(n, radius),
                "contact_diameter": np.full(n, 2.0 * radius),
                "endpoint_controlled": np.isin(np.arange(n), [0, n-1]).astype(float),
            },
            scalars={"friction_coefficient": 0.0},
        )
        frame.validate()
        return frame

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x*x*(3.0-2.0*x)

    def _waypoint_position(self, waypoints: list[dict], time: float) -> list[float]:
        if time <= float(waypoints[0]["time"]):
            return list(waypoints[0]["position"])
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            if time <= float(b["time"]):
                u = self._smoothstep((time-float(a["time"])) /
                                     (float(b["time"])-float(a["time"])))
                pa, pb = np.asarray(a["position"]), np.asarray(b["position"])
                return ((1.0-u)*pa + u*pb).tolist()
        return list(waypoints[-1]["position"])

    def endpoint_targets(self, time: float) -> dict:
        """Kinematic boundary targets; the downstream solver enforces rod physics."""
        controls = self.config["endpoint_manipulation"]
        return {
            "left_tail": self._waypoint_position(controls["left_tail"]["waypoints"], time),
            "threading_tail": self._waypoint_position(controls["threading_tail"]["waypoints"], time),
        }

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration/dt)+1).tolist()
