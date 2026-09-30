"""Deterministic, solver-neutral initial scene for sponge wetting and wringing."""
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
            "units": "meters",
            "up_axis": "z",
            "dynamic_bodies": {"sponge": 1, "water": 2},
            "kinematic_bodies": {"left_grip": 10, "right_grip": 11},
            "required_presentation_attributes": ["saturation"],
            "event_sequence": self.config["event_sequence"],
            "controls": self.config["controls"],
            "coupling_requirements": self.config["coupling_requirements"],
        }

    @staticmethod
    def _grid(lo, hi, spacing):
        axes = [np.arange(lo[i], hi[i] + spacing * 0.25, spacing) for i in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def initial_state(self) -> StateFrame:
        # Volumetric sponge samples start completely dry and at rest.
        cross_section = [(-0.08, 0.025), (0.08, 0.025), (0.08, 0.105), (-0.08, 0.105)]
        sponge = np.array([[x, y, z] for x in np.linspace(-0.18, 0.18, 9) for y, z in cross_section])

        # The running stream starts below the nozzle but remains above the sponge,
        # preserving a clearly dry t=0 state before first contact.
        z = np.arange(0.175, 0.525, 0.012)
        cross = np.array([[0.0, 0.0], [0.008, 0.0], [-0.008, 0.0], [0.0, 0.008], [0.0, -0.008]])
        water = np.array([[-0.07 + dx, 0.04 + dy, zz] for zz in z for dx, dy in cross])

        grips = np.array([[-0.205, 0.0, 0.065], [0.205, 0.0, 0.065]], dtype=float)
        position = np.vstack((sponge, water, grips)).astype(float)
        velocity = np.zeros_like(position)
        velocity[len(sponge):len(sponge) + len(water), 2] = -1.0
        body_id = np.concatenate((
            np.ones(len(sponge), dtype=np.int32),
            np.full(len(water), 2, dtype=np.int32),
            np.array([10, 11], dtype=np.int32),
        ))
        mass = np.concatenate((
            np.full(len(sponge), 0.0124 / len(sponge)),
            np.full(len(water), 0.00095),
            np.full(len(grips), 0.02),
        ))
        saturation = np.concatenate((np.zeros(len(sponge)), np.ones(len(water)), np.zeros(len(grips))))
        porosity = np.concatenate((np.full(len(sponge), 0.72), np.zeros(len(water) + len(grips))))
        phase = np.concatenate((np.zeros(len(sponge)), np.ones(len(water)), np.zeros(len(grips))))

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "saturation": saturation,
                "porosity": porosity,
                "phase": phase,
            },
            scalars={"gravity_z": -9.81},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
