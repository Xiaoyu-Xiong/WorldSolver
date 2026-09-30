from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, EvaluationContext, input_series, rms

_EPS = np.finfo(float).eps
_DEFAULT_SPATIAL_DIMENSIONS = 3


def _spatial_dimensions(parameters: dict[str, Any], calculator_name: str) -> int:
    dimensions = int(parameters.get("spatial_dimensions", _DEFAULT_SPATIAL_DIMENSIONS))
    if dimensions not in {2, 3}:
        raise ValueError(f"{calculator_name} requires spatial_dimensions 2 or 3")
    return dimensions


def _require_scalar_velocity_mask(
    scalar: np.ndarray,
    velocity: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    *,
    scalar_name: str,
    dimensions: int,
) -> None:
    if scalar.ndim != dimensions + 1:
        raise ValueError(f"{scalar_name} must have shape [T,nx,ny] or [T,nx,ny,nz]")
    if any(size < 2 for size in scalar.shape[1:]):
        raise ValueError(f"{scalar_name} must have at least two cells per axis")
    if source.shape != scalar.shape or mask.shape != scalar.shape:
        raise ValueError(f"{scalar_name}, source_field, and active_mask must be collocated")
    if velocity.shape[:-1] != scalar.shape or velocity.shape[-1] != dimensions:
        raise ValueError("velocity_field must have shape [T,nx,ny,2] or [T,nx,ny,nz,3]")


class SourceAwareBalanceCalculator:
    """Balance an extensive per-sample quantity against an exported cumulative source."""

    id = "balance.source_aware.v1"
    metric_name = "relative_budget_residual"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        quantities = input_series(frames, inputs, "quantity")
        sources = input_series(frames, inputs, "cumulative_source")
        totals = np.stack(
            [np.atleast_1d(np.asarray(value, dtype=float).sum(axis=0)) for value in quantities]
        )
        source_totals = np.stack(
            [np.atleast_1d(np.asarray(value, dtype=float)) for value in sources]
        )
        expected = totals[0] + source_totals - source_totals[0]
        residual = np.abs(totals - expected)
        scale = np.maximum(np.maximum(np.abs(expected), abs(totals[0])), _EPS)
        relative = residual / scale
        frame_errors = np.max(relative.reshape(len(frames), -1), axis=1)
        return CalculatorResult(
            frame_errors,
            {"initial_quantity": np.asarray(totals[0]).tolist()},
        )


class LinearMomentumBalanceCalculator:
    """Check d/dt sum(m v) = sum(f_external) from consecutive StateFrames."""

    id = "dynamics.linear_momentum_balance.v1"
    metric_name = "momentum_balance_relative_l2"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Momentum balance requires at least two frames")
        masses = input_series(frames, inputs, "mass")
        velocities = input_series(frames, inputs, "velocity")
        forces = input_series(frames, inputs, "external_force")
        momentum = np.stack(
            [
                np.sum(np.asarray(m, dtype=float)[:, None] * np.asarray(v, dtype=float), axis=0)
                for m, v in zip(masses, velocities, strict=True)
            ]
        )
        total_force = np.stack(
            [np.asarray(value, dtype=float).reshape(-1, 3).sum(axis=0) for value in forces]
        )
        times = np.asarray([frame.time for frame in frames], dtype=float)
        momentum_rate = np.gradient(momentum, times, axis=0, edge_order=1)
        residual = np.linalg.norm(momentum_rate - total_force, axis=-1)
        scale = np.maximum(
            np.linalg.norm(momentum_rate, axis=-1) + np.linalg.norm(total_force, axis=-1),
            _EPS,
        )
        return CalculatorResult(residual / scale)


class ContactPenetrationCalculator:
    """Measure negative signed contact gaps relative to a frozen length scale."""

    id = "contact.penetration.v1"
    metric_name = "penetration_relative_max"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        gaps = input_series(frames, inputs, "signed_distance")
        scales = input_series(frames, inputs, "length_scale")
        relative_max = []
        for gap, scale in zip(gaps, scales, strict=True):
            penetration = np.maximum(-np.asarray(gap, dtype=float), 0.0)
            denominator = max(abs(float(scale)), _EPS)
            relative_max.append(
                0.0 if penetration.size == 0 else float(np.max(penetration)) / denominator
            )
        return CalculatorResult(np.asarray(relative_max))


class TargetConstraintCalculator:
    """Compare sampled physical constraint values with frozen target values."""

    id = "constraint.target.v1"
    metric_name = "constraint_relative_rms"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        values = input_series(frames, inputs, "value")
        targets = input_series(frames, inputs, "target")
        scales = input_series(frames, inputs, "scale")
        relative = [
            rms(np.asarray(value, dtype=float) - np.asarray(target, dtype=float))
            / max(abs(float(scale)), _EPS)
            for value, target, scale in zip(values, targets, scales, strict=True)
        ]
        return CalculatorResult(np.asarray(relative))


class RigidShapeCalculator:
    """Remove translation and rotation per body and measure shape drift from frame zero."""

    id = "rigid.shape_preservation.v1"
    metric_name = "shape_relative_rms"
    metric_unit = "dimensionless"

    @staticmethod
    def _aligned_error(reference: np.ndarray, current: np.ndarray) -> float:
        reference_centered = reference - reference.mean(axis=0)
        current_centered = current - current.mean(axis=0)
        covariance = current_centered.T @ reference_centered
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned = current_centered @ rotation
        scale = rms(np.linalg.norm(reference_centered, axis=-1))
        return rms(aligned - reference_centered) / max(scale, _EPS)

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        positions = input_series(frames, inputs, "position")
        body_ids = input_series(frames, inputs, "body_id")
        reference_position = np.asarray(positions[0], dtype=float)
        reference_ids = np.asarray(body_ids[0])
        unique_ids = np.unique(reference_ids)
        errors = []
        for position, ids in zip(positions, body_ids, strict=True):
            position = np.asarray(position, dtype=float)
            ids = np.asarray(ids)
            body_errors = []
            for body_id in unique_ids:
                reference = reference_position[reference_ids == body_id]
                current = position[ids == body_id]
                if len(reference) != len(current) or len(reference) < 3:
                    raise ValueError("Each rigid body needs the same >=3 samples in every frame")
                body_errors.append(self._aligned_error(reference, current))
            errors.append(max(body_errors))
        return CalculatorResult(np.asarray(errors))


class RigidImpulseTransferCalculator:
    """Measure whether an incoming rigid body transfers impulse to the far body."""

    id = "rigid.impulse_transfer.v1"
    metric_name = "impulse_transfer_relative_error"
    metric_unit = "dimensionless"

    @staticmethod
    def _body_state(
        position: np.ndarray,
        velocity: np.ndarray,
        mass: np.ndarray,
        body_id: np.ndarray,
    ) -> dict[int, dict[str, np.ndarray | float]]:
        ids = np.asarray(body_id)
        positions = np.asarray(position, dtype=float)
        velocities = np.asarray(velocity, dtype=float)
        masses = np.asarray(mass, dtype=float)
        if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position and velocity must have shape [N,3]")
        if masses.shape != (positions.shape[0],) or ids.shape != (positions.shape[0],):
            raise ValueError("mass and body_id must have shape [N]")

        result: dict[int, dict[str, np.ndarray | float]] = {}
        for raw_id in np.unique(ids):
            mask = ids == raw_id
            body_mass = float(np.sum(masses[mask]))
            if body_mass <= _EPS:
                raise ValueError("Rigid impulse transfer requires positive body masses")
            momentum = np.sum(masses[mask, None] * velocities[mask], axis=0)
            centroid = np.sum(masses[mask, None] * positions[mask], axis=0) / body_mass
            result[int(raw_id)] = {
                "mass": body_mass,
                "momentum": momentum,
                "velocity": momentum / body_mass,
                "centroid": centroid,
            }
        return result

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Rigid impulse transfer requires at least two frames")
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        masses = input_series(frames, inputs, "mass")
        body_ids = input_series(frames, inputs, "body_id")

        initial = self._body_state(
            np.asarray(positions[0]),
            np.asarray(velocities[0]),
            np.asarray(masses[0]),
            np.asarray(body_ids[0]),
        )
        if len(initial) < 3:
            raise ValueError("Rigid impulse transfer requires at least three rigid bodies")
        speed_by_id = {
            body_id: float(np.linalg.norm(state["velocity"]))
            for body_id, state in initial.items()
        }
        incoming_id = max(speed_by_id, key=speed_by_id.get)
        incoming_velocity = np.asarray(initial[incoming_id]["velocity"], dtype=float)
        incoming_speed = float(np.linalg.norm(incoming_velocity))
        if incoming_speed <= _EPS:
            raise ValueError("Rigid impulse transfer requires a moving incoming body")

        axis = incoming_velocity / incoming_speed
        incoming_initial_momentum = max(
            float(np.dot(np.asarray(initial[incoming_id]["momentum"], dtype=float), axis)),
            _EPS,
        )
        initial_ids = set(initial)
        far_id = max(
            (body_id for body_id in initial if body_id != incoming_id),
            key=lambda body_id: float(
                np.dot(np.asarray(initial[body_id]["centroid"], dtype=float), axis)
            ),
        )
        intermediate_ids = initial_ids - {incoming_id, far_id}
        min_far_transfer = float(parameters.get("min_far_transfer_fraction", 0.25))
        incoming_weight = float(parameters.get("incoming_residual_weight", 0.5))
        lateral_weight = float(parameters.get("lateral_leakage_weight", 0.25))
        if min_far_transfer <= 0:
            raise ValueError("min_far_transfer_fraction must be positive")

        errors = []
        far_transfer_values = []
        intermediate_values = []
        incoming_values = []
        lateral_values = []
        for position, velocity, mass, ids in zip(
            positions, velocities, masses, body_ids, strict=True
        ):
            state = self._body_state(
                np.asarray(position),
                np.asarray(velocity),
                np.asarray(mass),
                np.asarray(ids),
            )
            if set(state) != initial_ids:
                raise ValueError("Rigid impulse transfer requires stable body_id set")

            far_momentum = np.asarray(state[far_id]["momentum"], dtype=float)
            far_transfer = max(float(np.dot(far_momentum, axis)), 0.0) / incoming_initial_momentum
            launch_deficit = max(0.0, min_far_transfer - far_transfer) / min_far_transfer

            intermediate_residual = sum(
                abs(float(np.dot(np.asarray(state[body_id]["momentum"], dtype=float), axis)))
                for body_id in intermediate_ids
            ) / incoming_initial_momentum
            incoming_residual = abs(
                float(np.dot(np.asarray(state[incoming_id]["momentum"], dtype=float), axis))
            ) / incoming_initial_momentum
            lateral_leakage = sum(
                float(
                    np.linalg.norm(
                        np.asarray(item["momentum"], dtype=float)
                        - np.dot(np.asarray(item["momentum"], dtype=float), axis) * axis
                    )
                )
                for item in state.values()
            ) / incoming_initial_momentum

            errors.append(
                launch_deficit
                + intermediate_residual
                + incoming_weight * incoming_residual
                + lateral_weight * lateral_leakage
            )
            far_transfer_values.append(far_transfer)
            intermediate_values.append(intermediate_residual)
            incoming_values.append(incoming_residual)
            lateral_values.append(lateral_leakage)

        return CalculatorResult(
            np.asarray(errors),
            {
                "incoming_body_id": incoming_id,
                "far_body_id": far_id,
                "intermediate_body_ids": sorted(intermediate_ids),
                "initial_impulse_axis": axis.tolist(),
                "initial_projected_momentum": incoming_initial_momentum,
                "final_far_transfer_fraction": far_transfer_values[-1],
                "final_intermediate_residual_fraction": intermediate_values[-1],
                "final_incoming_residual_fraction": incoming_values[-1],
                "final_lateral_leakage_fraction": lateral_values[-1],
                "min_far_transfer_fraction": min_far_transfer,
            },
        )


class RigidResidualKineticEnergyCalculator:
    """Measure how much kinetic energy remains after a rigid-body event settles."""

    id = "rigid.residual_kinetic_energy.v1"
    metric_name = "residual_kinetic_energy_fraction"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        masses = input_series(frames, inputs, "mass")
        velocities = input_series(frames, inputs, "velocity")
        energies = []
        for mass, velocity in zip(masses, velocities, strict=True):
            mass_array = np.asarray(mass, dtype=float)
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.ndim != 2 or velocity_array.shape[1] != 3:
                raise ValueError("velocity must have shape [N,3]")
            if mass_array.shape != (velocity_array.shape[0],):
                raise ValueError("mass must have shape [N]")
            if np.any(mass_array < 0):
                raise ValueError("mass must be non-negative")
            speed_squared = np.sum(np.square(velocity_array), axis=1)
            energies.append(float(0.5 * np.sum(mass_array * speed_squared)))

        energy_array = np.asarray(energies, dtype=float)
        peak = max(float(np.max(energy_array)), _EPS)
        return CalculatorResult(
            energy_array / peak,
            {
                "peak_kinetic_energy": peak,
                "initial_kinetic_energy": float(energy_array[0]),
                "final_kinetic_energy": float(energy_array[-1]),
                "final_residual_fraction": float(energy_array[-1] / peak),
            },
        )


class RigidKineticEventSettlingCalculator:
    """Measure that a rigid-body event becomes active and later settles."""

    id = "rigid.kinetic_event_settling.v1"
    metric_name = "kinetic_event_settling_error"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        masses = input_series(frames, inputs, "mass")
        velocities = input_series(frames, inputs, "velocity")
        min_peak_speed = float(parameters.get("min_peak_mass_weighted_rms_speed", 0.05))
        if min_peak_speed <= 0:
            raise ValueError("min_peak_mass_weighted_rms_speed must be positive")

        speeds = []
        for mass, velocity in zip(masses, velocities, strict=True):
            mass_array = np.asarray(mass, dtype=float)
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.ndim != 2 or velocity_array.shape[1] != 3:
                raise ValueError("velocity must have shape [N,3]")
            if mass_array.shape != (velocity_array.shape[0],):
                raise ValueError("mass must have shape [N]")
            total_mass = float(np.sum(mass_array))
            if total_mass <= _EPS:
                raise ValueError("Rigid kinetic event settling requires positive total mass")
            speed_squared = np.sum(np.square(velocity_array), axis=1)
            speeds.append(float(np.sqrt(np.sum(mass_array * speed_squared) / total_mass)))

        speed_array = np.asarray(speeds, dtype=float)
        peak_speed = float(np.max(speed_array))
        activation_deficit = max(0.0, min_peak_speed - peak_speed) / min_peak_speed
        residual = speed_array / max(peak_speed, _EPS)
        return CalculatorResult(
            activation_deficit + residual,
            {
                "initial_mass_weighted_rms_speed": float(speed_array[0]),
                "peak_mass_weighted_rms_speed": peak_speed,
                "final_mass_weighted_rms_speed": float(speed_array[-1]),
                "final_residual_speed_fraction": float(residual[-1]),
                "activation_deficit": activation_deficit,
                "min_peak_mass_weighted_rms_speed": min_peak_speed,
            },
        )


class RigidLoopContactProgressCalculator:
    """Measure loop traversal progress followed by an out-of-loop fall."""

    id = "rigid.loop_contact_progress.v1"
    metric_name = "loop_contact_progress_error"
    metric_unit = "dimensionless"

    @staticmethod
    def _body_centroid(position: np.ndarray, body_id: np.ndarray, target_body_id: int | None) -> np.ndarray:
        positions = np.asarray(position, dtype=float)
        ids = np.asarray(body_id)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position must have shape [N,3]")
        if ids.shape != (positions.shape[0],):
            raise ValueError("body_id must have shape [N]")
        if target_body_id is None:
            if len(np.unique(ids)) != 1:
                raise ValueError("loop contact progress needs body_id when multiple bodies exist")
            target_body_id = int(ids[0])
        mask = ids == target_body_id
        if not np.any(mask):
            raise ValueError(f"body_id {target_body_id} is missing from a frame")
        return np.mean(positions[mask], axis=0)

    @staticmethod
    def _positive_unwrapped_angles(raw_angles: np.ndarray, positions: np.ndarray) -> np.ndarray:
        forward = np.unwrap(raw_angles)
        backward = np.unwrap(-raw_angles)
        initial_delta = positions[min(1, len(positions) - 1), 0] - positions[0, 0]
        if initial_delta >= 0:
            return forward - forward[0]
        return backward - backward[0]

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Loop contact progress requires at least two frames")
        positions = input_series(frames, inputs, "position")
        body_ids = input_series(frames, inputs, "body_id")

        target_body_id = parameters.get("body_id")
        if target_body_id is not None:
            target_body_id = int(target_body_id)
        centroids = np.stack(
            [
                self._body_centroid(np.asarray(position), np.asarray(ids), target_body_id)
                for position, ids in zip(positions, body_ids, strict=True)
            ]
        )
        plane_axes = parameters.get("plane_axes", [0, 2])
        if len(plane_axes) != 2:
            raise ValueError("plane_axes must contain exactly two coordinate axes")
        horizontal_axis, vertical_axis = [int(axis) for axis in plane_axes]
        if horizontal_axis == vertical_axis or any(axis not in {0, 1, 2} for axis in plane_axes):
            raise ValueError("plane_axes must contain distinct axes from 0, 1, 2")

        center = parameters.get("loop_center")
        if center is None:
            horizontal_mid = 0.5 * (
                float(np.min(centroids[:, horizontal_axis]))
                + float(np.max(centroids[:, horizontal_axis]))
            )
            vertical_mid = 0.5 * (
                float(np.min(centroids[:, vertical_axis]))
                + float(np.max(centroids[:, vertical_axis]))
            )
            center_point = np.zeros(3, dtype=float)
            center_point[horizontal_axis] = horizontal_mid
            center_point[vertical_axis] = vertical_mid
        else:
            center_point = np.asarray(center, dtype=float)
            if center_point.shape != (3,):
                raise ValueError("loop_center must have shape [3]")

        radius = float(parameters.get("loop_radius", 0.0))
        relative = centroids - center_point
        planar = relative[:, [horizontal_axis, vertical_axis]]
        radial = np.linalg.norm(planar, axis=1)
        if radius <= 0.0:
            contact_band = parameters.get("contact_band_fraction", 0.35)
            near = radial <= float(np.max(radial)) * (1.0 + float(contact_band))
            radius = float(np.median(radial[near])) if np.any(near) else float(np.median(radial))
        if radius <= _EPS:
            raise ValueError("Loop contact progress requires a positive loop_radius")

        horizontal = relative[:, horizontal_axis]
        vertical_from_top_axis = -relative[:, vertical_axis]
        raw_angles = np.arctan2(horizontal, vertical_from_top_axis)
        progress = np.maximum.accumulate(
            self._positive_unwrapped_angles(raw_angles, centroids)
        )
        max_progress = float(np.max(progress))
        max_height = float(np.max(centroids[:, vertical_axis]))
        final_height = float(centroids[-1, vertical_axis])
        final_radial = float(radial[-1])

        min_first_top_fraction = float(parameters.get("min_first_top_fraction", 0.85))
        min_complete_pass_revolutions = float(parameters.get("min_complete_pass_revolutions", 1.0))
        min_second_attempt_revolutions = float(parameters.get("min_second_attempt_revolutions", 1.15))
        min_final_fall_fraction = float(parameters.get("min_final_fall_fraction", 0.45))
        min_detach_fraction = float(parameters.get("min_detach_fraction", 0.25))
        if min_complete_pass_revolutions <= 0.0:
            raise ValueError("min_complete_pass_revolutions must be positive")
        if min_second_attempt_revolutions <= min_complete_pass_revolutions:
            raise ValueError("min_second_attempt_revolutions must exceed min_complete_pass_revolutions")

        required_top_height = center_point[vertical_axis] + min_first_top_fraction * radius
        first_top_deficit = max(0.0, required_top_height - max_height) / radius
        required_first_pass_progress = min_complete_pass_revolutions * 2.0 * np.pi
        first_pass_deficit = (
            max(0.0, required_first_pass_progress - max_progress) / required_first_pass_progress
        )
        required_progress = min_second_attempt_revolutions * 2.0 * np.pi
        second_attempt_deficit = max(0.0, required_progress - max_progress) / required_progress
        required_fall_height = center_point[vertical_axis] - min_final_fall_fraction * radius
        final_fall_deficit = max(0.0, final_height - required_fall_height) / radius
        final_detach_deficit = max(0.0, min_detach_fraction * radius - abs(final_radial - radius)) / (
            min_detach_fraction * radius
        )

        error = (
            0.2 * first_top_deficit
            + 0.25 * first_pass_deficit
            + 0.4 * second_attempt_deficit
            + 0.1 * final_fall_deficit
            + 0.05 * final_detach_deficit
        )
        return CalculatorResult(
            np.full(len(frames), error, dtype=float),
            {
                "body_id": target_body_id,
                "loop_center": center_point.tolist(),
                "loop_radius": radius,
                "angular_progress_revolutions": max_progress / (2.0 * np.pi),
                "max_height": max_height,
                "final_height": final_height,
                "final_radial_distance": final_radial,
                "first_top_deficit": first_top_deficit,
                "first_pass_deficit": first_pass_deficit,
                "second_attempt_deficit": second_attempt_deficit,
                "final_fall_deficit": final_fall_deficit,
                "final_detach_deficit": final_detach_deficit,
            },
        )


class WilberforceModeEnergyExchangeCalculator:
    """Measure vertical/torsional mode exchange in a Wilberforce pendulum."""

    id = "wilberforce.mode_energy_exchange.v1"
    metric_name = "mode_energy_exchange_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        resolved = dict(parameters)
        if "body_id" not in resolved:
            if context.scene_geometry is None:
                raise ValueError("Wilberforce calculator requires scene geometry")
            requested_id = str(resolved.get("object_id", "pendulum_body"))
            keywords = tuple(
                str(value).lower()
                for value in resolved.get(
                    "object_keywords", ["pendulum", "weight", "mass"]
                )
            )
            candidates = [
                item
                for item in context.scene_geometry.objects
                if item.id == requested_id
                or (
                    item.motion == "dynamic"
                    and item.semantic == "rigid"
                    and any(keyword in item.id.lower() for keyword in keywords)
                )
            ]
            if len(candidates) != 1:
                raise ValueError("suspended Wilberforce body was not found uniquely")
            binding = candidates[0].state_binding
            if binding is None or binding.selector != "body_id":
                raise ValueError("suspended Wilberforce body requires a body_id state binding")
            resolved["body_id"] = int(binding.value)
        return self.compute(frames, inputs, resolved)

    @staticmethod
    def _body_centroid_series(
        values: list[Any],
        body_ids: list[Any],
        target_body_id: int,
    ) -> np.ndarray:
        series = []
        for value, ids in zip(values, body_ids, strict=True):
            array = np.asarray(value, dtype=float)
            id_array = np.asarray(ids)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError("position/velocity/angular_velocity must have shape [N,3]")
            if id_array.shape != (array.shape[0],):
                raise ValueError("body_id must have shape [N]")
            mask = id_array == target_body_id
            if not np.any(mask):
                raise ValueError(f"body_id {target_body_id} is missing from a frame")
            series.append(np.mean(array[mask], axis=0))
        return np.stack(series)

    @staticmethod
    def _peak_fraction(values: np.ndarray, start: int, end: int, peak: float) -> float:
        if end <= start:
            return 0.0
        return float(np.max(values[start:end]) / max(peak, _EPS))

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 6:
            raise ValueError("Wilberforce mode exchange requires at least six frames")
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        angular_velocities = input_series(frames, inputs, "angular_velocity")
        body_ids = input_series(frames, inputs, "body_id")

        body_id = int(parameters.get("body_id", 2))
        vertical_axis = int(parameters.get("vertical_axis", 2))
        torsion_axis = int(parameters.get("torsion_axis", 2))
        if vertical_axis not in {0, 1, 2} or torsion_axis not in {0, 1, 2}:
            raise ValueError("vertical_axis and torsion_axis must be 0, 1, or 2")

        position = self._body_centroid_series(positions, body_ids, body_id)
        velocity = self._body_centroid_series(velocities, body_ids, body_id)
        angular_velocity = self._body_centroid_series(angular_velocities, body_ids, body_id)
        times = np.asarray([frame.time for frame in frames], dtype=float)
        duration = max(float(times[-1] - times[0]), _EPS)

        length_scale = float(
            parameters.get(
                "vertical_length_scale",
                max(float(np.ptp(position[:, vertical_axis])), 1.0),
            )
        )
        speed_scale = float(parameters.get("vertical_speed_scale", length_scale / duration))
        angular_speed_scale = float(parameters.get("angular_speed_scale", 1.0))
        if length_scale <= 0 or speed_scale <= 0 or angular_speed_scale <= 0:
            raise ValueError("mode exchange scales must be positive")

        vertical_displacement = position[:, vertical_axis] - float(position[0, vertical_axis])
        vertical_proxy = np.square(vertical_displacement / length_scale) + np.square(
            velocity[:, vertical_axis] / speed_scale
        )
        torsional_proxy = np.square(angular_velocity[:, torsion_axis] / angular_speed_scale)

        vertical_peak = float(np.max(vertical_proxy))
        torsional_peak = float(np.max(torsional_proxy))
        min_vertical_peak = float(parameters.get("min_vertical_peak", 1.0e-4))
        min_torsional_peak = float(parameters.get("min_torsional_peak", 1.0e-4))
        vertical_activation_deficit = max(0.0, min_vertical_peak - vertical_peak) / min_vertical_peak
        torsional_activation_deficit = (
            max(0.0, min_torsional_peak - torsional_peak) / min_torsional_peak
        )

        count = len(frames)
        first_end = max(1, count // 3)
        second_end = max(first_end + 1, (2 * count) // 3)
        early_vertical = self._peak_fraction(vertical_proxy, 0, first_end, vertical_peak)
        middle_vertical = self._peak_fraction(vertical_proxy, first_end, second_end, vertical_peak)
        late_vertical = self._peak_fraction(vertical_proxy, second_end, count, vertical_peak)
        early_torsion = self._peak_fraction(torsional_proxy, 0, first_end, torsional_peak)
        middle_torsion = self._peak_fraction(torsional_proxy, first_end, second_end, torsional_peak)
        late_torsion = self._peak_fraction(torsional_proxy, second_end, count, torsional_peak)

        min_dominance_margin = float(parameters.get("min_dominance_margin", 0.15))
        min_late_recovery_fraction = float(parameters.get("min_late_recovery_fraction", 0.35))
        early_vertical_deficit = max(
            0.0, early_torsion + min_dominance_margin - early_vertical
        )
        middle_torsion_deficit = max(
            0.0, middle_vertical + min_dominance_margin - middle_torsion
        )
        late_vertical_recovery_deficit = max(
            0.0, min_late_recovery_fraction - late_vertical
        ) / max(min_late_recovery_fraction, _EPS)
        late_torsion_decay_deficit = max(
            0.0, late_torsion - late_vertical + min_dominance_margin
        )

        error = (
            0.1 * vertical_activation_deficit
            + 0.15 * torsional_activation_deficit
            + 0.15 * early_vertical_deficit
            + 0.25 * middle_torsion_deficit
            + 0.1 * late_vertical_recovery_deficit
            + 0.25 * late_torsion_decay_deficit
        )
        return CalculatorResult(
            np.full(count, error, dtype=float),
            {
                "body_id": body_id,
                "vertical_axis": vertical_axis,
                "torsion_axis": torsion_axis,
                "vertical_peak_proxy": vertical_peak,
                "torsional_peak_proxy": torsional_peak,
                "vertical_activation_deficit": vertical_activation_deficit,
                "torsional_activation_deficit": torsional_activation_deficit,
                "early_vertical_fraction": early_vertical,
                "middle_vertical_fraction": middle_vertical,
                "late_vertical_fraction": late_vertical,
                "early_torsion_fraction": early_torsion,
                "middle_torsion_fraction": middle_torsion,
                "late_torsion_fraction": late_torsion,
                "early_vertical_deficit": early_vertical_deficit,
                "middle_torsion_deficit": middle_torsion_deficit,
                "late_vertical_recovery_deficit": late_vertical_recovery_deficit,
                "late_torsion_decay_deficit": late_torsion_decay_deficit,
            },
        )


class RigidCentroidDriftCalculator:
    """Measure drift of rigid-body centroids from their initial locations."""

    id = "rigid.centroid_drift.v1"
    metric_name = "centroid_drift_relative_max"
    metric_unit = "dimensionless"

    @staticmethod
    def _centroids(position: np.ndarray, body_id: np.ndarray) -> dict[int, np.ndarray]:
        positions = np.asarray(position, dtype=float)
        ids = np.asarray(body_id)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position must have shape [N,3]")
        if ids.shape != (positions.shape[0],):
            raise ValueError("body_id must have shape [N]")
        return {
            int(raw_id): np.mean(positions[ids == raw_id], axis=0)
            for raw_id in np.unique(ids)
        }

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        positions = input_series(frames, inputs, "position")
        body_ids = input_series(frames, inputs, "body_id")
        reference = self._centroids(np.asarray(positions[0]), np.asarray(body_ids[0]))
        reference_ids = set(reference)
        reference_points = np.stack([reference[body_id] for body_id in sorted(reference)])
        bbox = np.ptp(reference_points, axis=0)
        default_scale = max(float(np.linalg.norm(bbox)), _EPS)
        scale = float(parameters.get("length_scale", default_scale))
        if scale <= 0:
            raise ValueError("length_scale must be positive")

        errors = []
        for position, ids in zip(positions, body_ids, strict=True):
            current = self._centroids(np.asarray(position), np.asarray(ids))
            if set(current) != reference_ids:
                raise ValueError("Rigid centroid drift requires stable body_id set")
            errors.append(
                max(
                    float(np.linalg.norm(current[body_id] - reference[body_id])) / scale
                    for body_id in reference_ids
                )
            )
        return CalculatorResult(
            np.asarray(errors),
            {
                "length_scale": scale,
                "final_max_centroid_drift": float(errors[-1]),
            },
        )


class DeformationJacobianCalculator:
    id = "deformation.jacobian.v1"
    metric_name = "nonpositive_jacobian_fraction"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        gradients = input_series(frames, inputs, "deformation_gradient")
        negative_fraction = []
        for gradient in gradients:
            array = np.asarray(gradient, dtype=float)
            if array.ndim != 3 or array.shape[-1] != array.shape[-2]:
                raise ValueError("deformation_gradient must have shape [N,D,D]")
            jacobian = np.linalg.det(array)
            negative_fraction.append(float(np.mean(jacobian <= 0.0)))
        return CalculatorResult(np.asarray(negative_fraction))


class StrainLimitCalculator:
    id = "deformation.strain_limit.v1"
    metric_name = "strain_limit_relative_max"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        gradients = input_series(frames, inputs, "deformation_gradient")
        lower = float(parameters.get("lower", 0.0))
        upper = float(parameters["upper"])
        if upper <= lower:
            raise ValueError("strain upper limit must be greater than lower limit")
        violations = []
        for gradient in gradients:
            singular = np.linalg.svd(np.asarray(gradient, dtype=float), compute_uv=False)
            violation = np.maximum(lower - singular, 0.0) + np.maximum(singular - upper, 0.0)
            violations.append(float(np.max(violation)) / max(upper - lower, _EPS))
        return CalculatorResult(np.asarray(violations))


class ContinuityCalculator:
    """Evaluate partial_t rho + div(rho u) - source on a collocated grid."""

    id = "field.continuity.v1"
    metric_name = "continuity_relative_l2"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Continuity requires at least two frames")
        dimensions = _spatial_dimensions(parameters, "Continuity")
        density = np.stack(input_series(frames, inputs, "density_field")).astype(float)
        velocity = np.stack(input_series(frames, inputs, "velocity_field")).astype(float)
        source = np.stack(input_series(frames, inputs, "source_field")).astype(float)
        mask = np.stack(input_series(frames, inputs, "active_mask")).astype(bool)
        spacing = np.asarray(input_series(frames, inputs, "grid_spacing"), dtype=float)
        times = np.asarray([frame.time for frame in frames], dtype=float)
        _require_scalar_velocity_mask(
            density,
            velocity,
            source,
            mask,
            scalar_name="density_field",
            dimensions=dimensions,
        )
        if np.any(spacing <= 0):
            raise ValueError("grid_spacing must be positive")
        temporal = np.gradient(density, times, axis=0, edge_order=1)
        values = []
        for index in range(len(frames)):
            divergence = np.zeros_like(density[index])
            for axis in range(dimensions):
                flux = density[index] * velocity[index, ..., axis]
                divergence += np.gradient(flux, spacing[index], axis=axis, edge_order=1)
            selected = mask[index]
            residual = temporal[index][selected] + divergence[selected] - source[index][selected]
            scale = (
                rms(temporal[index][selected])
                + rms(divergence[selected])
                + rms(source[index][selected])
            )
            values.append(rms(residual) / max(scale, _EPS))
        return CalculatorResult(np.asarray(values))


class ScalarTransportCalculator:
    """Evaluate partial_t q + u dot grad(q) - source on a collocated grid."""

    id = "field.scalar_transport.v1"
    metric_name = "transport_relative_l2"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("Scalar transport requires at least two frames")
        dimensions = _spatial_dimensions(parameters, "Scalar transport")
        scalar = np.stack(input_series(frames, inputs, "scalar_field")).astype(float)
        velocity = np.stack(input_series(frames, inputs, "velocity_field")).astype(float)
        source = np.stack(input_series(frames, inputs, "source_field")).astype(float)
        mask = np.stack(input_series(frames, inputs, "active_mask")).astype(bool)
        spacing = np.asarray(input_series(frames, inputs, "grid_spacing"), dtype=float)
        times = np.asarray([frame.time for frame in frames], dtype=float)
        _require_scalar_velocity_mask(
            scalar,
            velocity,
            source,
            mask,
            scalar_name="scalar_field",
            dimensions=dimensions,
        )
        if np.any(spacing <= 0):
            raise ValueError("grid_spacing must be positive")
        temporal = np.gradient(scalar, times, axis=0, edge_order=1)
        values = []
        for index in range(len(frames)):
            gradients = np.gradient(
                scalar[index],
                *([float(spacing[index])] * dimensions),
                edge_order=1,
            )
            advection = sum(
                velocity[index, ..., axis] * gradients[axis]
                for axis in range(dimensions)
            )
            selected = mask[index]
            residual = temporal[index][selected] + advection[selected] - source[index][selected]
            scale = (
                rms(temporal[index][selected])
                + rms(advection[selected])
                + rms(source[index][selected])
            )
            values.append(rms(residual) / max(scale, _EPS))
        return CalculatorResult(np.asarray(values))


class InteractionForceBalanceCalculator:
    """Check that exported internal coupling/contact forces sum to zero."""

    id = "coupling.force_balance.v1"
    metric_name = "interaction_force_relative_residual"
    metric_unit = "dimensionless"

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        force_series = input_series(frames, inputs, "interaction_force")
        values = []
        for force in force_series:
            array = np.asarray(force, dtype=float).reshape(-1, 3)
            residual = np.linalg.norm(array.sum(axis=0))
            scale = np.linalg.norm(array, axis=-1).sum()
            values.append(residual / max(scale, _EPS))
        return CalculatorResult(np.asarray(values))
