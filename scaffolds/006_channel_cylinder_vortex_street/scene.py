"""Deterministic solver-neutral channel-flow scene contract."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    """Initial samples and observables for flow past a fixed circular cylinder."""

    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "wall_confined_cylinder_vortex_street",
            "coordinate_convention": "flow_+x_cross_channel_y_span_z",
        }

    def initial_state(self) -> StateFrame:
        spacing = float(self.config["fluid"]["nominal_particle_spacing"])
        radius = float(self.config["obstacle"]["radius"])
        cx, cy, _ = self.config["obstacle"]["center"]

        # Cell-centered planar samples for an undistorted scientific top view.
        xs = np.arange(-6.0 + 0.5 * spacing, 10.0, spacing)
        ys = np.arange(-1.5 + 0.5 * spacing, 1.5, spacing)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        keep = (xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2
        x, y = xx[keep], yy[keep]
        z = np.zeros_like(x)
        position = np.column_stack((x, y, z))

        centerline = float(self.config["initial_conditions"]["centerline_speed"])
        ux = centerline * np.maximum(0.0, 1.0 - (y / 1.5) ** 2)
        velocity = np.column_stack((ux, np.zeros_like(ux), np.zeros_like(ux)))
        grid_velocity = np.zeros(xx.shape + (3,))
        grid_velocity[..., 0] = centerline * np.maximum(0.0, 1.0 - (yy / 1.5) ** 2)
        grid_velocity[~keep] = 0.0
        grid_vorticity = 2.0 * centerline * yy / 1.5**2
        grid_vorticity[~keep] = 0.0

        # Alternating narrow inlet stripes are passive labels. Their downstream
        # deformation makes alternating shedding and advection immediately visible.
        stripe = np.floor((x + 6.0) / 0.55).astype(np.int32)
        dye = np.where(x < -2.55, ((stripe & 1) * 2 - 1), 0).astype(float)
        near_center = np.exp(-0.5 * (y / 0.34) ** 2)
        dye *= near_center
        grid_dye = np.zeros_like(xx)
        grid_dye[keep] = dye

        n = len(position)
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=velocity,
            mass=np.full(n, float(self.config["fluid"]["density"]) * spacing**2),
            attributes={
                "dye": dye,
                "vorticity": grid_vorticity[keep],
                "divergence": np.zeros(n),
                "velocity_magnitude": ux.copy(),
            },
            fields={
                "velocity": grid_velocity,
                "velocity_field": grid_velocity[..., :2].copy(),
                "active_mask": keep.astype(np.uint8),
                "vorticity": grid_vorticity,
                "divergence": np.zeros((len(ys), len(xs))),
                "dye": grid_dye,
            },
            scalars={
                "cylinder_radius": radius,
                "mean_inlet_speed": float(self.config["initial_conditions"]["mean_speed"]),
                "expected_strouhal_number": float(self.config["diagnostics"]["expected_strouhal_number"]),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        dt = float(self.config["sample_dt"])
        duration = float(self.config["duration"])
        count = int(round(duration / dt))
        return (np.arange(count + 1, dtype=float) * dt).tolist()
