"""Deterministic solver-neutral cable winding scene."""
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
            "phenomenon": "pitch-controlled winding of an extensible cable around a cylinder",
            "gravity": list(self.config["gravity"]),
            "cable": dict(self.config["cable"]),
            "cylinder": dict(self.config["cylinder"]),
            "contact": dict(self.config["contact"]),
            "winding": dict(self.config["winding"]),
            "initial_conditions": dict(self.config["initial_conditions"]),
            "event_sequence": list(self.config["event_sequence"]),
            "required_outcome": self.config["required_outcome"],
        }

    def initial_state(self) -> StateFrame:
        cable = self.config["cable"]
        initial = self.config["initial_conditions"]
        n = int(cable["vertex_count"])
        free = np.asarray(initial["free_end"], dtype=float)
        driven = np.asarray(initial["driven_end"], dtype=float)
        q = np.linspace(0.0, 1.0, n)

        # A slightly bowed, entirely unwrapped cable. The bow prevents an
        # artificial perfectly symmetric first contact but prescribes no coil.
        position = (1.0 - q[:, None]) * free + q[:, None] * driven
        position[:, 1] += 0.006 * np.sin(np.pi * q)
        segment_rest = float(cable["length"]) / (n - 1)
        area = np.pi * float(cable["radius"]) ** 2
        nodal_mass = np.full(n, float(cable["density"]) * area * segment_rest)
        nodal_mass[[0, -1]] *= 0.5
        controlled = np.zeros(n)
        controlled[-1] = 1.0

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=nodal_mass,
            attributes={
                "arc_fraction": q,
                "rest_segment_length": np.full(n, segment_rest),
                "cable_radius": np.full(n, float(cable["radius"])),
                "driven_endpoint": controlled,
                "free_endpoint": (np.arange(n) == 0).astype(float),
            },
            scalars={
                "young_modulus_pa": float(cable["young_modulus_pa"]),
                "poisson_ratio": float(cable["poisson_ratio"]),
                "generalized_friction_kappa": float(self.config["contact"]["kappa"]),
                "generalized_friction_exponent": float(self.config["contact"]["exponent_N"]),
            },
        )
        frame.validate()
        return frame

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def grip_target(self, time: float) -> dict:
        """Prescribed end pose only; downstream code supplies all dynamics."""
        winding = self.config["winding"]
        initial = np.asarray(self.config["initial_conditions"]["driven_end"], dtype=float)
        t = float(np.clip(time, 0.0, float(self.config["duration"])))
        ta = float(winding["approach_end_time"])
        tw = float(winding["winding_end_time"])
        radius = float(winding["centerline_orbit_radius"])
        angle0 = float(winding["start_angle_rad"])
        start = np.array([radius * np.cos(angle0), radius * np.sin(angle0),
                          float(winding["start_height"])])

        if t < ta:
            u = self._smoothstep(t / ta)
            position = (1.0 - u) * initial + u * start
            angle = angle0
        else:
            u = self._smoothstep((t - ta) / (tw - ta))
            angle = angle0 + 2.0 * np.pi * float(winding["turns"]) * u
            z = (1.0 - u) * float(winding["start_height"]) + u * float(winding["end_height"])
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), z])

        tangent = np.array([-np.sin(angle), np.cos(angle),
                            -float(winding["desired_pitch"]) / (2.0 * np.pi)])
        tangent /= np.linalg.norm(tangent)
        return {"position": position.tolist(), "tangent": tangent.tolist(),
                "controlled_vertex": int(self.config["cable"]["driven_vertex"])}

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
