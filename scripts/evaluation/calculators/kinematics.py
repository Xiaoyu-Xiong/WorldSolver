from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series


class CenterOfMassConsistencyCalculator:
    """Check dx_com = integral(v_com dt) for each persistent rigid body."""

    id = "trajectory.center_of_mass_consistency.v1"
    metric_name = "center_of_mass_kinematic_relative_residual"
    metric_unit = "dimensionless"

    @staticmethod
    def _body_states(position, velocity, mass, body_id):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        mass = np.asarray(mass, dtype=float)
        body_id = np.asarray(body_id)
        if position.ndim != 2 or position.shape[1] != 3 or not len(position):
            raise ValueError("position must have nonempty shape [N,3]")
        if velocity.shape != position.shape:
            raise ValueError("velocity must match position shape [N,3]")
        if mass.shape != (len(position),) or body_id.shape != (len(position),):
            raise ValueError("mass and body_id must have shape [N]")
        if not all(np.isfinite(value).all() for value in (position, velocity, mass, body_id)):
            raise ValueError("Rigid body state must be finite")
        if np.any(mass < 0.0) or np.any(body_id != np.floor(body_id)):
            raise ValueError("mass must be nonnegative and body_id must be integral")

        states = {}
        for identifier in np.unique(body_id):
            selected = body_id == identifier
            total_mass = float(np.sum(mass[selected]))
            if total_mass <= 0.0:
                raise ValueError("Each rigid body must have positive total mass")
            weights = mass[selected] / total_mass
            states[int(identifier)] = (
                np.sum(weights[:, None] * position[selected], axis=0),
                np.sum(weights[:, None] * velocity[selected], axis=0),
            )
        return states

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Center-of-mass consistency requires at least two frames")
        times = np.asarray([frame.time for frame in frames], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise ValueError("Trajectory times must be finite and strictly increasing")
        floor = float(parameters.get("displacement_floor", 1e-8))
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("displacement_floor must be finite and positive")

        series = [
            input_series(frames, inputs, name)
            for name in ("position", "velocity", "mass", "body_id")
        ]
        errors = np.zeros(len(frames))
        maximum_residual = 0.0
        previous = None
        identifiers = set()
        for index, values in enumerate(zip(*series, strict=True)):
            current = self._body_states(*values)
            if previous is None:
                identifiers = set(current)
            else:
                if set(current) != identifiers:
                    raise ValueError("Center-of-mass consistency requires stable body_id sets")
                dt = times[index] - times[index - 1]
                for identifier in identifiers:
                    old_position, old_velocity = previous[identifier]
                    position, velocity = current[identifier]
                    observed = position - old_position
                    # Trapezoidal quadrature is exact for constant acceleration; unresolved
                    # impacts introduce sampling error, so this is not a force-balance test.
                    expected = 0.5 * (old_velocity + velocity) * dt
                    residual = float(np.linalg.norm(observed - expected))
                    scale = max(
                        float(np.linalg.norm(observed)), float(np.linalg.norm(expected)), floor
                    )
                    errors[index] = max(errors[index], residual / scale)
                    maximum_residual = max(maximum_residual, residual)
            previous = current

        return CalculatorResult(
            errors,
            {
                "body_count": len(identifiers),
                "interval_count": len(frames) - 1,
                "quadrature": "trapezoidal",
                "body_reduction": "maximum",
                "initial_frame_error": "zero (no preceding interval)",
                "displacement_floor_meters": floor,
                "maximum_position_residual_meters": maximum_residual,
            },
        )
