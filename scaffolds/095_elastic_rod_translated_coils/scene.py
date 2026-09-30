"""Deterministic solver-neutral translated elastic-rod coiling scene."""
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
            "phenomenon": "same-handed translated coiling of an elastic rod on a moving belt",
            "gravity": list(self.config["gravity"]),
            "simulation_mode": self.config["simulation_mode"],
            "rod": dict(self.config["rod"]),
            "injector": dict(self.config["injector"]),
            "belt": dict(self.config["belt"]),
            "event_sequence": list(self.config["event_sequence"]),
            "required_outcome": dict(self.config["required_outcome"]),
        }

    def initial_state(self) -> StateFrame:
        rod = self.config["rod"]
        injector = self.config["injector"]
        initial = self.config["initial_conditions"]
        n = int(rod["vertex_count"])
        length = float(rod["initial_deployed_length"])
        s = np.linspace(0.0, length, n)
        outlet = np.asarray(injector["outlet"], dtype=float)

        # A nearly vertical pre-contact heel. The smooth, minute y offset only
        # selects one winding branch; no deposited coil is encoded initially.
        q = s / length
        bias = float(initial["lateral_branch_bias"])
        position = outlet + np.column_stack((np.zeros(n), bias*q*q, -s))
        tip_z = float(self.config["belt"]["surface_z"]) + float(initial["lower_tip_clearance"])
        position[:, 2] += tip_z - position[-1, 2]

        segment_length = length / (n - 1)
        line_mass = float(rod["density"]) * np.pi * float(rod["radius"])**2
        mass = np.full(n, line_mass * segment_length)
        mass[[0, -1]] *= 0.5
        controlled = np.zeros(n)
        controlled[0] = 1.0
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=mass,
            attributes={
                "arc_length": s,
                "natural_curvature": np.zeros(n),
                "natural_twist": np.zeros(n),
                "injector_controlled": controlled,
                "deposited_sticking": np.zeros(n),
            },
            scalars={
                "rod_radius": float(rod["radius"]),
                "injection_speed": float(injector["injection_speed"]),
                "belt_speed": float(self.config["belt"]["speed"]),
            },
        )
        frame.validate()
        return frame

    def boundary_targets(self, time: float) -> dict:
        """Prescribed feed and belt kinematics for downstream solver use."""
        t = float(np.clip(time, 0.0, float(self.config["duration"])))
        injector = self.config["injector"]
        belt = self.config["belt"]
        feed_elapsed = np.clip(t - float(injector["feed_start"]), 0.0,
                               float(injector["feed_end"])-float(injector["feed_start"]))
        return {
            "injector": {
                "outlet": list(injector["outlet"]),
                "axis": list(injector["axis"]),
                "speed": float(injector["injection_speed"]),
                "supplied_length": float(injector["injection_speed"]) * float(feed_elapsed),
            },
            "moving_belt": {
                "velocity": (float(belt["speed"]) * np.asarray(belt["translation_axis"])).tolist(),
                "surface_z": float(belt["surface_z"]),
                "contact_condition": "sticking",
            },
        }

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
