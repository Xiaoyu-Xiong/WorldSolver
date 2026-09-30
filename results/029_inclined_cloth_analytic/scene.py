"""Deterministic, solver-neutral inclined-cloth scene contract."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial samples and topology for a square cloth parallel to a rigid ramp."""

    def __init__(self, config: dict):
        self.config = config
        self._positions, self._faces, self._uv = self._make_cloth()

    def _make_cloth(self):
        cloth = self.config["cloth"]
        nx, ny = map(int, cloth["grid_shape"])
        side = float(cloth["side_length"])
        angle = np.deg2rad(float(self.config["incline"]["angle_degrees"]))
        # Local ramp axes: u points uphill, v crosses the slope, n points outward.
        u = np.array([np.cos(angle), 0.0, np.sin(angle)])
        v = np.array([0.0, 1.0, 0.0])
        n = np.array([-np.sin(angle), 0.0, np.cos(angle)])
        # Anchor the cloth above the ramp's top face at the same downslope-coordinate
        # center.  The box transform is its mid-plane, so include its half-thickness.
        ramp_center = np.asarray(self.config["incline"]["plane_translation"], dtype=float)
        ramp_half_thickness = float(self.config["incline"]["plane_half_thickness"])
        center = ramp_center + 0.48 * u + (
            ramp_half_thickness + float(cloth["initial_normal_gap"])
        ) * n
        uv = np.array([(i / (nx - 1), j / (ny - 1)) for j in range(ny) for i in range(nx)])
        positions = center + (uv[:, 0:1] - 0.5) * side * u + (uv[:, 1:2] - 0.5) * side * v
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b, c, d = a + 1, a + nx, a + nx + 1
                faces.extend(((a, b, d), (a, d, c)))
        return positions, np.asarray(faces, dtype=np.int32), uv

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "cloth_name": "inclined_cloth",
            "topology": {"faces": self._faces.copy(), "grid_shape": tuple(self.config["cloth"]["grid_shape"])},
            "gravity": tuple(self.config["gravity"]),
            "coulomb_friction": float(self.config["friction"]["coefficient"]),
            "contact_pair": ("inclined_cloth", "inclined_plane"),
            "diagnostics": self.config["diagnostics"],
        }

    def initial_state(self) -> StateFrame:
        count = len(self._positions)
        velocity = np.tile(np.asarray(self.config["cloth"]["initial_velocity"], dtype=float), (count, 1))
        mass = np.full(count, float(self.config["cloth"]["total_mass"]) / count)
        frame = StateFrame(
            0.0,
            self._positions.copy(),
            velocity,
            mass=mass,
            attributes={"uv": self._uv.copy(), "cloth_vertex": np.ones(count)},
            scalars={
                "friction_coefficient": float(self.config["friction"]["coefficient"]),
                "constraint_residual": 0.0,
                "analytical_displacement_error": 0.0,
                "analytical_velocity_error": 0.0,
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        return np.linspace(0.0, duration, round(duration / dt) + 1).tolist()
