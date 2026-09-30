"""Deterministic solver-neutral scene for a sealed thin-plate jet test."""
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
            "phenomenon": "upward jet redirected laterally beneath an impermeable thin plate",
            "coupled_materials": ["coupled_jet_gas", "coupled_thin_solid"],
            "dimensionality": 2,
            "interface_height": 1.80,
            "upper_chamber_threshold": 1.85,
            "target_leaked_mass": 0.0,
            "scaffold_visual_intent": (
                "Render injected_fraction, not ambient density, so the upper chamber starts visually clear "
                "and only the lower inlet jet appears before plate deflection."
            ),
        }

    def initial_state(self) -> StateFrame:
        nx, ny, nz = 121, 3, 73
        x = np.linspace(-3.0, 3.0, nx)
        y = np.linspace(-0.18, 0.18, ny)
        z = np.linspace(0.0, 3.6, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        position = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        density = np.ones((nx, ny, nz), dtype=float)
        pressure = np.ones_like(density)
        velocity_grid = np.zeros((nx, ny, nz, 3), dtype=float)

        # Seed only the inlet and a short lower-chamber core. The upper chamber
        # starts perfectly quiescent and contains no injected tracer mass.
        core = (np.abs(xx) <= 0.21) & (zz <= 0.45)
        taper = np.clip(1.0 - (np.abs(xx) / 0.24) ** 4, 0.0, 1.0)
        velocity_grid[..., 2] = np.where(core, 2.4 * taper, 0.0)
        density[core] = 1.18
        pressure[core] = 1.08
        injected_fraction = np.where(core, 1.0, 0.0)

        velocity = velocity_grid.reshape((-1, 3))
        mass = density.ravel() * (6.0 / (nx - 1)) * (0.36 / (ny - 1)) * (3.6 / (nz - 1))
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={"injected_fraction": injected_fraction.ravel()},
            fields={
                "density": density,
                "pressure": pressure,
                "velocity": velocity_grid,
                "injected_fraction": injected_fraction,
            },
            scalars={
                "upper_chamber_injected_mass": 0.0,
                "lower_chamber_injected_mass": float(np.sum(mass * injected_fraction.ravel() * (zz.ravel() < 1.80))),
                "pressure_jump_across_plate": 0.0,
                "maximum_speed": 2.4,
                "finite_state": 1.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return [round(i * dt, 12) for i in range(count + 1)]
