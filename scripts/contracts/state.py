from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


@dataclass(slots=True)
class StateFrame:
    """Solver-neutral state sampled at one simulation time."""

    time: float
    position: Array
    velocity: Array
    orientation: Array | None = None
    angular_velocity: Array | None = None
    mass: Array | None = None
    body_id: NDArray[np.integer] | None = None
    attributes: Mapping[str, Array] = field(default_factory=dict)
    fields: Mapping[str, Array] = field(default_factory=dict)
    scalars: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("StateFrame.time must be finite and non-negative")
        for name in ("position", "velocity"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite [N,3] array")
        if self.position.shape != self.velocity.shape:
            raise ValueError("position and velocity shapes differ")
        n = len(self.position)
        optional_shapes = {
            "orientation": (n, 4),
            "angular_velocity": (n, 3),
            "mass": (n,),
            "body_id": (n,),
        }
        for name, shape in optional_shapes.items():
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value is not None and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if not all(np.isfinite(v) for v in self.scalars.values()):
            raise ValueError("scalars contain NaN or infinity")
        for name, value in self.attributes.items():
            array = np.asarray(value)
            if array.ndim < 1 or array.shape[0] != n:
                raise ValueError(f"attribute {name!r} must have N values")
            if not np.isfinite(array).all():
                raise ValueError(f"attribute {name!r} contains NaN or infinity")
        for name, value in self.fields.items():
            array = np.asarray(value)
            if array.ndim < 1 or not np.isfinite(array).all():
                raise ValueError(f"field {name!r} must be a finite non-scalar array")
