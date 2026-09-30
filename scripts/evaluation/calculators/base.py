from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from worldsolver.contracts import StateFrame, frame_value
from worldsolver.contracts.scene_geometry import SceneGeometry


@dataclass(frozen=True, slots=True)
class CalculatorResult:
    normalized_errors: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    # Optional explicit alignment for interval laws or sparse applicable samples.
    # None preserves the original one-result-per-trajectory-frame contract.
    frame_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.normalized_errors, dtype=float)
        if values.ndim != 1:
            raise ValueError("Calculator normalized_errors must have shape [T]")
        if values.size == 0:
            raise ValueError("Calculator normalized_errors cannot be empty")
        if not np.isfinite(values).all():
            raise ValueError("Calculator normalized_errors contain NaN or infinity")
        if np.any(values < 0.0):
            raise ValueError("Calculator normalized_errors must be non-negative")
        object.__setattr__(self, "normalized_errors", values)
        if self.frame_indices is not None:
            indices = np.asarray(self.frame_indices)
            if indices.shape != values.shape or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(
                    "Calculator frame_indices must be an integer array aligned to errors"
                )
            if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
                raise ValueError(
                    "Calculator frame_indices must be nonnegative and strictly increasing"
                )
            object.__setattr__(self, "frame_indices", indices)


class Calculator(Protocol):
    id: str
    metric_name: str
    metric_unit: str

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult: ...


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    result_dir: Path
    scene_geometry: SceneGeometry | None
    config_path: Path | None = None


class ContextualCalculator(Protocol):
    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult: ...


def input_series(frames: list[StateFrame], inputs: dict[str, str], name: str) -> list[Any]:
    try:
        path = inputs[name]
    except KeyError as exc:
        raise ValueError(f"Calculator input {name!r} is not bound") from exc
    return [frame_value(frame, path) for frame in frames]


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Calculator received an empty evaluation region")
    return float(np.sqrt(np.mean(np.square(array))))
