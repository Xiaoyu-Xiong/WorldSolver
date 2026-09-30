from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series


class ExtensiveQuantityCalculator:
    id = "balance.extensive_quantity.v1"
    metric_name = "relative_budget_residual"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if parameters.get("source_mode", "none") != "none":
            raise ValueError("v1 pilot supports only a frozen zero-source budget")
        masses = input_series(frames, inputs, "mass")
        phase_ids = input_series(frames, inputs, "phase_id")
        selected_ids = input_series(frames, inputs, "selected_phase_id")
        quantity = np.asarray(
            [
                np.asarray(mass, dtype=float)[np.asarray(phase) == selected].sum()
                for mass, phase, selected in zip(masses, phase_ids, selected_ids, strict=True)
            ],
            dtype=float,
        )
        initial = float(quantity[0])
        absolute = np.abs(quantity - initial)
        relative = absolute / max(abs(initial), np.finfo(float).eps)
        return CalculatorResult(relative, {"initial_quantity": initial})
