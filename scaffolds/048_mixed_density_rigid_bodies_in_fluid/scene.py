"""Deterministic solver-neutral mixed-density rigid-body/fluid scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial samples and metadata; downstream code supplies the coupled dynamics."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "buoyancy and settling across wide rigid-body density contrast",
            "fluid_density": float(self.config["fluid"]["density"]),
            "density_ratios": list(self.config["rigid_bodies"]["density_ratios"]),
            "scaffold_visual_intent": (
                "Fluid markers are intentionally sparse and pale; each rigid body has a distinct "
                "high-contrast color so upward buoyant and downward settling trajectories are trackable."
            ),
        }

    def initial_state(self) -> StateFrame:
        fluid = self.config["fluid"]
        lo = np.asarray(fluid["bounds_min"], dtype=float)
        hi = np.asarray(fluid["bounds_max"], dtype=float)
        h = float(fluid["marker_spacing"])
        axes = [np.arange(lo[i] + 0.5 * h, hi[i], h) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        # Seeded sub-cell offsets reveal flow while preserving a reproducible fill.
        rng = np.random.default_rng(int(self.config["seed"]))
        grid += rng.uniform(-0.16 * h, 0.16 * h, size=grid.shape)

        centers = np.array([
            [-0.92, 0.00, 2.14], [-0.88, 0.06, 1.58], [0.08, 0.00, 1.72],
            [0.05, 0.00, 0.78], [0.32, -0.02, 0.39], [0.95, 0.00, 0.62],
        ], dtype=float)
        radii = np.array([0.27, 0.29, 0.36, 0.31, 0.29, 0.25])
        keep = np.ones(len(grid), dtype=bool)
        for center, radius in zip(centers, radii):
            keep &= np.linalg.norm(grid - center, axis=1) > radius
        markers = grid[keep]

        position = np.vstack((markers, centers))
        velocity = np.zeros_like(position)
        velocity[:len(markers)] = np.asarray(fluid["initial_velocity"], dtype=float)
        body_id = np.concatenate((np.zeros(len(markers), dtype=np.int32), np.arange(1, 7, dtype=np.int32)))

        rho_f = float(fluid["density"])
        ratios = np.asarray(self.config["rigid_bodies"]["density_ratios"], dtype=float)
        volumes = np.array([
            4.0 / 3.0 * np.pi * 0.25**3,
            np.pi * 0.22**2 * 0.20 + 4.0 / 3.0 * np.pi * 0.22**3,
            8.0 * 0.29 * 0.20 * 0.18,
            np.pi * 0.24**2 * 0.18 + 4.0 / 3.0 * np.pi * 0.24**3,
            8.0 * 0.20**3,
            4.0 / 3.0 * np.pi * 0.23**3,
        ])
        mass = np.concatenate((np.full(len(markers), rho_f * h**3), rho_f * ratios * volumes))
        sample_density = np.concatenate((np.full(len(markers), rho_f), rho_f * ratios))
        phase = np.concatenate((np.zeros(len(markers)), np.ones(6)))

        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            angular_velocity=np.zeros_like(position),
            mass=mass,
            body_id=body_id,
            attributes={
                "density": sample_density,
                "density_ratio": sample_density / rho_f,
                "coupled_phase": phase,
                "marker_volume": np.concatenate((np.full(len(markers), h**3), volumes)),
            },
            scalars={"fluid_density": rho_f, "initial_total_momentum": 0.0},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, int(round(duration / dt)) + 1).tolist()
