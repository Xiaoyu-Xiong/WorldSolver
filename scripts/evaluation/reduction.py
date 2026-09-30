from __future__ import annotations

from typing import Any

import numpy as np


def reduce_values(values, reduction: dict[str, Any]) -> float:
    """Reduce one normalized-error value per frame to a trajectory measurement."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Reduction input must have shape [T]")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Reduction input is empty or non-finite")
    kind = reduction["type"]
    if kind == "maximum":
        return float(np.max(array))
    if kind == "minimum":
        return float(np.min(array))
    if kind == "mean":
        return float(np.mean(array))
    if kind == "rms":
        return float(np.sqrt(np.mean(np.square(array))))
    if kind == "final":
        return float(array[-1])
    raise ValueError(f"Unknown reduction: {kind}")
