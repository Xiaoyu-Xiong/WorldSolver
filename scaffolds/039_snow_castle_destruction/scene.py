"""Deterministic, solver-neutral initial state for snow-castle destruction."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Build a crenellated particle castle and an incoming rigid striker sample."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["seed"])

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": self.seed,
            "gravity": self.config.get("gravity", [0.0, 0.0, -9.81]),
            "phenomenon": "projectile-triggered fracture, compaction, and collapse of snow castle",
            "scaffold_visual_intent": (
                "The orange striker must visibly hit the front-left side of a warm off-white "
                "crenellated snow castle, then remain on screen long enough for post-impact "
                "fracture and deposition to be judged."
            ),
        }

    @staticmethod
    def _inside_castle(p: np.ndarray) -> np.ndarray:
        """Reference-inspired stepped castle with towers, battlements, and gate void."""
        x, y, z = p.T
        # Main keep and lower enclosing wall.
        solid = ((abs(x) <= 0.72) & (abs(y) <= 0.66) & (z <= 2.18))
        solid |= ((abs(x) <= 1.28) & (abs(y) <= 0.92) & (z <= 1.30))

        # Four cylindrical corner towers, slightly taller than the curtain walls.
        for cx in (-1.05, 1.05):
            for cy in (-0.72, 0.72):
                solid |= (((x - cx) ** 2 + (y - cy) ** 2 <= 0.36 ** 2) & (z <= 1.72))

        # Raised central tower and its crown blocks.
        solid |= ((abs(x) <= 0.42) & (abs(y) <= 0.42) & (z <= 2.72))

        # Merlons around the main wall and central tower make the silhouette readable.
        for cx in (-1.20, -0.72, -0.24, 0.24, 0.72, 1.20):
            solid |= ((abs(x - cx) <= 0.13) & (abs(y) <= 0.94) & (z >= 1.28) & (z <= 1.58))
        for cy in (-0.82, -0.30, 0.30, 0.82):
            solid |= ((abs(y - cy) <= 0.13) & (abs(x) <= 1.30) & (z >= 1.28) & (z <= 1.58))
        for cx in (-0.32, 0.0, 0.32):
            solid |= ((abs(x - cx) <= 0.11) & (abs(y) <= 0.44) & (z >= 2.70) & (z <= 3.04))
        for cy in (-0.32, 0.0, 0.32):
            solid |= ((abs(y - cy) <= 0.11) & (abs(x) <= 0.44) & (z >= 2.70) & (z <= 3.04))

        # Arched front gate on the camera-facing (negative-y) wall.
        gate = (abs(x) < 0.25) & (y < -0.58) & (z < 0.64)
        gate |= ((x * x + (z - 0.62) ** 2 < 0.25 ** 2) & (y < -0.58) & (z < 0.86))
        return solid & ~gate

    def _castle_particles(self) -> np.ndarray:
        h = float(self.config.get("initial_conditions", {}).get("snow_particle_spacing", 0.075))
        xs = np.arange(-1.32, 1.32 + 0.5 * h, h)
        ys = np.arange(-0.98, 0.98 + 0.5 * h, h)
        zs = np.arange(0.5 * h, 3.08, h)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        points = grid[self._inside_castle(grid)]
        rng = np.random.default_rng(self.seed)
        points += rng.uniform(-0.10 * h, 0.10 * h, size=points.shape)
        return points

    def initial_state(self) -> StateFrame:
        snow = self._castle_particles()
        initial = self.config.get("initial_conditions", {})
        ball = np.array([initial.get("striker_start", [-2.85, 0.05, 0.82])], dtype=float)
        position = np.vstack((snow, ball))
        velocity = np.zeros_like(position)
        velocity[-1] = np.asarray(
            self.config.get("initial_conditions", {}).get("striker_velocity", [7.5, 0.0, 0.0]),
            dtype=float,
        )
        spacing = float(self.config.get("initial_conditions", {}).get("snow_particle_spacing", 0.075))
        density = float(self.config.get("initial_conditions", {}).get("snow_bulk_density", 420.0))
        snow_particle_mass = density * spacing ** 3
        mass = np.full(len(position), snow_particle_mass, dtype=float)
        mass[-1] = 18.0
        body_id = np.ones(len(position), dtype=np.int32)
        body_id[-1] = 2
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={"initial_volume": np.full(len(position), spacing ** 3)},
            scalars={"initial_total_mass": float(mass.sum())},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return np.linspace(0.0, duration, count + 1).tolist()
