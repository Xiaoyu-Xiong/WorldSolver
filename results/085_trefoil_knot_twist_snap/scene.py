"""Deterministic solver-neutral open-trefoil twist-instability scene."""
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
            "phenomenon": "twist-induced dynamic snap of an open trefoil knot",
            "rod": self.config["rod"],
            "gravity": self.config["gravity"],
            "clamps": self.config["clamps"],
            "event_windows": self.config["event_windows"],
            "required_outcome": self.config["required_outcome"],
        }

    def _centerline(self) -> np.ndarray:
        """Loose trefoil from a smooth knot core blended into horizontal tails."""
        n = int(self.config["rod"]["vertex_count"])
        s = np.linspace(0.0, 1.0, n)
        # Open traversal of a standard trefoil, oriented with its large loop upright.
        t = 2.0 * np.pi * (0.08 + 0.84 * s)
        x = 0.72 * (np.sin(t) + 2.0 * np.sin(2.0 * t))
        z = 0.92 + 0.55 * (np.cos(t) - 2.0 * np.cos(2.0 * t))
        y = 0.30 * np.sin(3.0 * t)
        core = np.column_stack((x, y, z))
        left = np.asarray(self.config["clamps"]["left"]["position"], dtype=float)
        right = np.asarray(self.config["clamps"]["right"]["position"], dtype=float)
        tail = 0.16
        for mask, endpoint, join_s in ((s < tail, left, tail), (s > 1-tail, right, 1-tail)):
            idx = np.flatnonzero(mask)
            join = int(round(join_s * (n - 1)))
            if idx.size:
                u = s[idx] / tail if join_s < 0.5 else (s[idx] - (1-tail)) / tail
                core[idx] = endpoint + u[:, None] * (core[join] - endpoint) if join_s < 0.5 else core[join] + u[:, None] * (endpoint - core[join])
        core[0], core[-1] = left, right
        return core

    def initial_state(self) -> StateFrame:
        p = self._centerline()
        n = len(p)
        radius = float(self.config["rod"]["radius"])
        frame = StateFrame(
            0.0, p, np.zeros_like(p),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            angular_velocity=np.zeros_like(p),
            mass=np.full(n, float(self.config["rod"]["mass"]) / n),
            attributes={
                "arc_fraction": np.linspace(0.0, 1.0, n),
                "rod_radius": np.full(n, radius),
                "contact_diameter": np.full(n, 2.0 * radius),
                "endpoint_controlled": np.isin(np.arange(n), [0, n-1]).astype(float),
                "material_twist_turns": np.zeros(n),
            },
        )
        frame.validate()
        return frame

    def clamp_targets(self, time: float) -> dict:
        """Kinematic end targets, including the driven clamp's axial rotation."""
        c = self.config["clamps"]
        ramp = c["twist_ramp"]
        u = np.clip((time-ramp["ramp_start"]) / (ramp["ramp_end"]-ramp["ramp_start"]), 0.0, 1.0)
        u = u*u*(3.0-2.0*u)
        angle = 2.0*np.pi*float(ramp["final_relative_turns"])*u
        return {
            "left": {"position": c["left"]["position"], "orientation_xyzw": c["left"]["orientation_xyzw"]},
            "right": {"position": c["right"]["position"], "axis": c["right"]["rotation_axis"], "angle_radians": float(angle)},
        }

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration/dt)+1).tolist()
