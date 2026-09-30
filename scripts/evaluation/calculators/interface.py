from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series, rms

_DEFAULT_SPATIAL_DIMENSIONS = 3


def _spatial_dimensions(parameters: dict[str, Any]) -> int:
    dimensions = int(parameters.get("spatial_dimensions", _DEFAULT_SPATIAL_DIMENSIONS))
    if dimensions not in {2, 3}:
        raise ValueError("level-set kinematics requires spatial_dimensions 2 or 3")
    return dimensions


class LevelSetKinematicCalculator:
    id = "interface.level_set_kinematic.v1"
    metric_name = "kinematic_relative_l2"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Level-set kinematics requires at least two frames")
        dimensions = _spatial_dimensions(parameters)
        band_cells = float(parameters.get("interface_band_cells", 1.5))
        level_sets = np.stack(input_series(frames, inputs, "level_set")).astype(float)
        velocities = np.stack(input_series(frames, inputs, "velocity_field")).astype(float)
        spacings = np.asarray(input_series(frames, inputs, "grid_spacing"), dtype=float)
        times = np.asarray([frame.time for frame in frames], dtype=float)
        if level_sets.ndim != dimensions + 1:
            raise ValueError("level_set must have shape [T,nx,ny] or [T,nx,ny,nz]")
        if any(size < 2 for size in level_sets.shape[1:]):
            raise ValueError("level_set must have at least two cells per axis")
        if (
            velocities.shape[:-1] != level_sets.shape
            or velocities.shape[-1] != dimensions
        ):
            raise ValueError("velocity_field must have shape [T,nx,ny,2] or [T,nx,ny,nz,3]")
        if np.any(spacings <= 0):
            raise ValueError("grid_spacing must be positive")
        temporal = np.gradient(level_sets, times, axis=0, edge_order=1)
        relative = []
        for index in range(len(frames)):
            gradients = np.gradient(
                level_sets[index],
                *([float(spacings[index])] * dimensions),
                edge_order=1,
            )
            advection = sum(
                velocities[index, ..., axis] * gradients[axis]
                for axis in range(dimensions)
            )
            band = np.abs(level_sets[index]) <= band_cells * spacings[index]
            residual = temporal[index][band] + advection[band]
            temporal_rms = rms(temporal[index][band])
            advection_rms = rms(advection[band])
            value_rms = rms(residual)
            relative.append(
                value_rms
                / max(temporal_rms + advection_rms, np.finfo(float).eps)
            )
        return CalculatorResult(np.asarray(relative))
