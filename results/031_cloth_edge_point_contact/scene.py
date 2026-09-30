"""Deterministic square-cloth scene with sharp edge and point contacts."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial cloth samples and fixed collision-feature metadata."""

    def __init__(self, config: dict):
        self.config = config
        self._positions, self._faces, self._uv = self._make_cloth()

    def _make_cloth(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cloth = self.config["cloth"]
        nx, ny = map(int, cloth["grid_shape"])
        side = float(cloth["side_length"])
        center = np.asarray(cloth["initial_center"], dtype=float)
        x = np.linspace(-0.5 * side, 0.5 * side, nx)
        y = np.linspace(-0.5 * side, 0.5 * side, ny)
        positions = np.asarray(
            [[center[0] + px, center[1] + py, center[2]] for py in y for px in x],
            dtype=float,
        )
        uv = np.asarray(
            [[ix / (nx - 1), iy / (ny - 1)] for iy in range(ny) for ix in range(nx)],
            dtype=float,
        )
        faces: list[tuple[int, int, int]] = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                b, c, d = a + 1, a + nx, a + nx + 1
                if (ix + iy) % 2 == 0:
                    faces.extend(((a, b, d), (a, d, c)))
                else:
                    faces.extend(((a, b, c), (b, d, c)))
        return positions, np.asarray(faces, dtype=np.int32), uv

    def metadata(self) -> dict:
        cloth = self.config["cloth"]
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "gravity": list(map(float, self.config["gravity"])),
            "cloth_name": cloth["name"],
            "topology": {
                "faces": self._faces.tolist(),
                "grid_shape": list(map(int, cloth["grid_shape"])),
            },
            "sharp_edge": self.config["sharp_edge"],
            "sharp_point": self.config["sharp_point"],
            "ground": self.config["ground"],
            "coulomb_friction": float(self.config["friction"]["coefficient"]),
            "cloth_thickness": float(cloth["thickness"]),
        }

    def initial_state(self) -> StateFrame:
        count = len(self._positions)
        mass = np.full(count, float(self.config["cloth"]["total_mass"]) / count)
        velocity = np.tile(
            np.asarray(self.config["cloth"]["initial_velocity"], dtype=float),
            (count, 1),
        )
        deformation_gradient = np.tile(np.eye(3), (count, 1, 1))
        frame = StateFrame(
            0.0,
            self._positions.copy(),
            velocity,
            mass=mass,
            attributes={
                "uv": self._uv.copy(),
                "mass_quantity": mass.copy(),
                "deformation_gradient": deformation_gradient,
            },
            fields={
                "contact_signed_distance": np.full(count, 0.6),
                "cumulative_mass_source": np.asarray([0.0]),
            },
            scalars={"contact_length_scale": float(self.config["cloth"]["thickness"])},
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
