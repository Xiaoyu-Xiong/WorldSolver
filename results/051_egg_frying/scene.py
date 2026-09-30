"""Deterministic solver-neutral initial scene for frying an egg on a hot pan."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _ellipsoid_lattice(center, radii, spacing):
    """Return a regular interior sampling without prescribing solver topology."""
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    axes = [np.arange(c - r, c + r + spacing * 0.25, spacing) for c, r in zip(center, radii)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    local = (grid - center) / radii
    return grid[np.sum(local * local, axis=1) <= 1.0]


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "thermal phase-front evolution and spreading of unmixed egg materials",
            "coupled_materials": ["egg_white", "egg_yolk"],
            "hot_boundary": "hot_pan_base",
            "concentration_diffusion_enabled": False,
            "phase_diffusion_enabled": True,
            "rendered_scalar_attributes": {
                "solid_phase_fraction": "0 raw/liquid, 1 cooked/solid phase for diagnostic color overlay",
                "temperature_k": "thermal front diagnostic if solver chooses temperature coloring",
            },
            "scaffold_visual_intent": (
                "Use dark pan, orange yolk, translucent-to-opaque egg white, and pale eggshell pieces; "
                "also export a solid_phase_fraction attribute so the raw-to-cooked phase front is "
                "visible as a scalar color overlay rather than only a material name."
            ),
        }

    def initial_state(self) -> StateFrame:
        # A low, broad albumen pool and an offset domed yolk match Fig. 1's deposited egg.
        white = _ellipsoid_lattice((0.012, 0.006, 0.035), (0.076, 0.062, 0.017), 0.007)
        yolk = _ellipsoid_lattice((-0.026, -0.003, 0.052), (0.034, 0.031, 0.032), 0.006)
        position = np.vstack((white, yolk))
        n_white, n_yolk = len(white), len(yolk)
        body_id = np.concatenate((np.full(n_white, 1), np.full(n_yolk, 2))).astype(np.int32)
        masses = np.concatenate((np.full(n_white, 0.034 / n_white), np.full(n_yolk, 0.018 / n_yolk)))

        # Concentrations are one-hot and conserved: phase may evolve independently.
        concentration = np.zeros((n_white + n_yolk, 2), dtype=float)
        concentration[:n_white, 0] = 1.0
        concentration[n_white:, 1] = 1.0
        temperature = np.full(n_white + n_yolk, float(self.config["initial_conditions"]["egg_temperature_k"]))
        liquid_phase = np.ones(n_white + n_yolk, dtype=float)
        solid_phase = np.zeros(n_white + n_yolk, dtype=float)
        material_label = body_id.astype(float)
        pan_distance = position[:, 2] - float(self.config["boundary_conditions"]["pan_contact_height_m"])

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            mass=masses,
            body_id=body_id,
            attributes={
                "material_concentration": concentration,
                "material_label": material_label,
                "liquid_phase_fraction": liquid_phase,
                "solid_phase_fraction": solid_phase,
                "temperature_k": temperature,
                "pan_contact_distance": pan_distance,
            },
            scalars={
                "total_egg_mass_kg": float(np.sum(masses)),
                "pan_temperature_k": float(self.config["initial_conditions"]["pan_temperature_k"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return [min(i * dt, duration) for i in range(count + 1)]
