from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series, rms

_DEFAULT_SPATIAL_DIMENSIONS = 3


class BoundaryVelocityCalculator:
    id = "boundary.velocity_residual.v1"
    metric_name = "normal_velocity_relative_rms"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if parameters.get("condition", "no_penetration") != "no_penetration":
            raise ValueError("v1 supports the no_penetration condition")
        dimensions = int(parameters.get("spatial_dimensions", _DEFAULT_SPATIAL_DIMENSIONS))
        if dimensions not in {2, 3}:
            raise ValueError("boundary velocity requires spatial_dimensions 2 or 3")
        velocity_series = input_series(frames, inputs, "boundary_velocity")
        target_series = input_series(frames, inputs, "boundary_target_velocity")
        normal_series = input_series(frames, inputs, "boundary_normal")
        reference_series = input_series(frames, inputs, "reference_velocity")
        relative = []
        for velocity, target, normal, reference in zip(
            velocity_series,
            target_series,
            normal_series,
            reference_series,
            strict=True,
        ):
            velocity = np.asarray(velocity, dtype=float)
            target = np.asarray(target, dtype=float)
            normal = np.asarray(normal, dtype=float)
            if velocity.shape != target.shape or velocity.shape != normal.shape:
                raise ValueError("Boundary sample arrays must have identical shapes")
            if velocity.ndim != 2 or velocity.shape[-1] != dimensions:
                raise ValueError("Boundary sample arrays must have shape [B,2] or [B,3]")
            normal_length = np.linalg.norm(normal, axis=-1)
            if np.any(normal_length <= np.finfo(float).eps):
                raise ValueError("boundary_normal contains a zero vector")
            unit_normal = normal / normal_length[:, None]
            residual = np.sum((velocity - target) * unit_normal, axis=-1)
            value_rms = rms(residual)
            relative.append(value_rms / max(abs(float(reference)), np.finfo(float).eps))
        return CalculatorResult(np.asarray(relative))
