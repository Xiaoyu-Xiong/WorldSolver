"""Deterministic solver-neutral initial state for Soluble Bunny Water."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Declares coupled material samples; downstream code chooses the solver."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "deformable soluble bunny dissolves into pipe water",
            "coupled_materials": {"soluble_red": 1, "clear_water": 2, "red_water_mixture": 3},
            "conserved_quantity": "red_solute_mass",
            "rendered_scalar_attributes": {
                "red_solute_concentration": "0 clear water, 1 red soluble bunny material",
                "solid_phase_fraction": "1 solid bunny, 0 liquid water; may decrease at dissolving interfaces",
            },
            "event_sequence_s": {"impact": 0.35, "deformation": 1.0, "mixing": 2.0, "mostly_dissolved": 7.0},
            "scaffold_visual_intent": (
                "A saturated pink soluble bunny, transparent blue pipe stream, and red-pink dissolved "
                "plume must be separately visible; red water is driven by the exported "
                "red_solute_concentration scalar, not by a fixed red material name."
            ),
        }

    @staticmethod
    def _bunny_samples() -> np.ndarray:
        # Interior lattice approximates the same analytic bunny used by assets/bunny.obj.
        parts = [
            ((0, 0, .72), (.48, .34, .58)), ((.26, 0, .60), (.43, .37, .45)),
            ((-.28, 0, 1.18), (.32, .29, .34)), ((-.51, 0, 1.13), (.20, .24, .18)),
            ((.67, 0, .88), (.18, .18, .20)), ((-.18, -.23, .22), (.32, .20, .17)),
            ((-.18, .23, .22), (.32, .20, .17)), ((-.38, -.12, 1.65), (.13, .14, .50)),
            ((-.16, .13, 1.66), (.13, .14, .48)),
        ]
        x = np.arange(-.78, .89, .105); y = np.arange(-.50, .51, .105); z = np.arange(.07, 2.14, .105)
        p = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.zeros(len(p), dtype=bool)
        for center, radius in parts:
            inside |= np.sum(((p - center) / radius) ** 2, axis=1) <= 1.0
        p = p[inside]; p[:, 0] += .35
        return p

    @staticmethod
    def _water_samples() -> np.ndarray:
        x = np.arange(-.90, .30, .085); y = np.arange(-.24, .25, .085); z = np.arange(.92, 1.50, .085)
        return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    def initial_state(self) -> StateFrame:
        bunny = self._bunny_samples(); water = self._water_samples()
        # A tiny zero-concentration receiving layer makes the declared final mixing domain bindable.
        x = np.arange(-.55, 2.21, .22); y = np.arange(-1.20, 1.21, .22)
        base = np.stack(np.meshgrid(x, y, [.09], indexing="ij"), axis=-1).reshape(-1, 3)
        position = np.vstack((bunny, water, base)).astype(float)
        velocity = np.zeros_like(position); velocity[len(bunny):len(bunny)+len(water), 0] = 2.4
        labels = np.concatenate((np.ones(len(bunny)), np.full(len(water), 2.0), np.full(len(base), 3.0)))
        concentration = np.concatenate((np.ones(len(bunny)), np.zeros(len(water) + len(base))))
        phase = np.concatenate((np.ones(len(bunny)), np.zeros(len(water) + len(base))))
        spacing = .105
        density = np.where(labels == 1, 1180.0, 1000.0)
        mass = density * spacing ** 3
        frame = StateFrame(
            0.0, position, velocity, mass=mass, body_id=labels.astype(int),
            attributes={"material_label": labels, "red_solute_concentration": concentration,
                        "solid_phase_fraction": phase, "density": density},
            fields={"initial_red_solute_concentration": concentration.copy()},
            scalars={"initial_red_solute_mass": float(np.sum(mass * concentration)),
                     "initial_bunny_mass": float(np.sum(mass[:len(bunny)])),
                     "inflow_speed": 2.4},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt, duration = float(self.config["sample_dt"]), float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
