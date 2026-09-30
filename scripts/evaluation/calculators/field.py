from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series, rms

_DEFAULT_SPATIAL_DIMENSIONS = 3


def _spatial_dimensions(parameters: dict[str, Any]) -> int:
    dimensions = int(parameters.get("spatial_dimensions", _DEFAULT_SPATIAL_DIMENSIONS))
    if dimensions not in {2, 3}:
        raise ValueError("field calculators require spatial_dimensions 2 or 3")
    return dimensions


def _require_grid_and_velocity(
    scalar_grid: np.ndarray,
    velocity: np.ndarray,
    *,
    scalar_name: str,
    dimensions: int,
) -> None:
    if scalar_grid.ndim != dimensions:
        raise ValueError(f"{scalar_name} must have shape [nx,ny] or [nx,ny,nz]")
    if any(size < 2 for size in scalar_grid.shape):
        raise ValueError(f"{scalar_name} must have at least two cells per axis")
    if velocity.shape[:-1] != scalar_grid.shape:
        raise ValueError(f"velocity_field is not collocated with {scalar_name}")
    if velocity.shape[-1] != dimensions:
        raise ValueError("velocity_field must have shape [nx,ny,2] or [nx,ny,nz,3]")


class DivergenceCalculator:
    id = "field.divergence.v1"
    metric_name = "divergence_relative_l2"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        dimensions = _spatial_dimensions(parameters)
        velocity_series = input_series(frames, inputs, "velocity_field")
        mask_series = input_series(frames, inputs, "active_mask")
        spacing_series = input_series(frames, inputs, "grid_spacing")
        length_series = input_series(frames, inputs, "characteristic_length")
        relative = []
        for velocity, mask, spacing, length in zip(
            velocity_series,
            mask_series,
            spacing_series,
            length_series,
            strict=True,
        ):
            velocity = np.asarray(velocity, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            _require_grid_and_velocity(
                mask,
                velocity,
                scalar_name="active_mask",
                dimensions=dimensions,
            )
            spacing = float(spacing)
            if spacing <= 0:
                raise ValueError("grid_spacing must be positive")
            divergence = np.zeros(mask.shape, dtype=float)
            for axis in range(dimensions):
                divergence += np.gradient(
                    velocity[..., axis], spacing, axis=axis, edge_order=1
                )
            selected_divergence = divergence[mask]
            selected_velocity = velocity[mask, :dimensions]
            div_rms = rms(selected_divergence)
            speed_rms = rms(np.linalg.norm(selected_velocity, axis=-1))
            relative.append(
                div_rms * float(length) / max(speed_rms, np.finfo(float).eps)
            )
        return CalculatorResult(np.asarray(relative))
