from __future__ import annotations

import json
from typing import Any

import numpy as np

from worldsolver.config import load_yaml
from worldsolver.contracts import StateFrame, frame_value
from worldsolver.contracts.scene_geometry import SceneObject

from .base import CalculatorResult, EvaluationContext, input_series

_EPS = np.finfo(float).eps


def _quat_matrix(value: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= _EPS:
        raise ValueError("orientation quaternion must be nonzero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _objects(context: EvaluationContext) -> list[SceneObject]:
    if context.scene_geometry is None:
        raise ValueError("calculator requires result_dir/scene_geometry.yaml")
    return context.scene_geometry.objects


def _find_object(
    context: EvaluationContext,
    requested_id: str,
    *keywords: str,
) -> SceneObject:
    objects = _objects(context)
    for item in objects:
        if item.id == requested_id:
            return item
    lowered = tuple(word.lower() for word in keywords if word)
    matches = [item for item in objects if any(word in item.id.lower() for word in lowered)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"scene object {requested_id!r} was not found uniquely")


def _body_mask(frame: StateFrame, item: SceneObject) -> np.ndarray:
    binding = item.state_binding
    if binding is None:
        raise ValueError(f"scene object {item.id!r} has no state binding")
    if binding.selector == "all":
        return np.ones(len(frame.position), dtype=bool)
    if binding.selector == "body_id":
        if frame.body_id is None:
            raise ValueError("state.body_id is required by scene geometry")
        return np.asarray(frame.body_id) == binding.value
    if binding.selector == "attribute":
        if binding.attribute_name not in frame.attributes:
            raise ValueError(f"missing state attribute {binding.attribute_name!r}")
        return np.asarray(frame.attributes[binding.attribute_name]) == binding.value
    raise ValueError(f"state binding {binding.selector!r} cannot select point samples")


def _centroid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=float)[mask]
    if selected.size == 0:
        raise ValueError("bound scene object has no trajectory samples")
    return selected.mean(axis=0)


def _hinge(value: float, low: float | None, high: float | None, scale: float) -> float:
    error = 0.0
    if low is not None:
        error = max(error, low - value)
    if high is not None:
        error = max(error, value - high)
    return error / max(scale, _EPS)


class InclinedPlaneReferenceCalculator:
    """Compare cloth centroid observables with a gravity+Coulomb analytic trajectory."""

    id = "trajectory.reference_curve.v1"
    metric_name = "analytic_trajectory_relative_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("reference-curve comparison requires at least two frames")
        plane = _find_object(
            context, str(parameters.get("plane_object_id", "incline_plane")), "incline", "plane"
        )
        if plane.geometry.type not in {"box", "plane"}:
            raise ValueError("inclined-plane reference requires a box or plane boundary")
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        centers = np.stack([np.asarray(value, dtype=float).mean(axis=0) for value in positions])
        center_velocity = np.stack(
            [np.asarray(value, dtype=float).mean(axis=0) for value in velocities]
        )
        rotation = _quat_matrix(plane.transform.rotation_xyzw)
        normal = rotation @ np.asarray([0.0, 0.0, 1.0])
        if plane.geometry.type == "box":
            half_z = float(plane.geometry.half_extents[2] * plane.transform.scale[2])
            plane_point = np.asarray(plane.transform.translation, dtype=float) + normal * half_z
        else:
            local_normal = np.asarray(plane.geometry.normal, dtype=float)
            normal = rotation @ (local_normal / max(np.linalg.norm(local_normal), _EPS))
            plane_point = np.asarray(plane.transform.translation, dtype=float) + normal * float(
                plane.geometry.offset or 0.0
            )

        times = np.asarray([frame.time for frame in frames], dtype=float)
        tau = times - times[0]
        gravity = np.asarray(parameters.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        mu = float(parameters.get("friction_coefficient", 0.2))
        initial_gap = max(float(np.dot(centers[0] - plane_point, normal)), 0.0)
        normal_acceleration = float(np.dot(gravity, normal))
        if normal_acceleration >= -_EPS:
            contact_time = 0.0
        else:
            contact_time = float(np.sqrt(2.0 * initial_gap / -normal_acceleration))
        contact_time = min(contact_time, float(tau[-1]))
        gravity_tangent = gravity - normal_acceleration * normal
        tangent_magnitude = float(np.linalg.norm(gravity_tangent))
        friction = mu * abs(normal_acceleration)
        slide_acceleration = (
            gravity_tangent * max(0.0, 1.0 - friction / tangent_magnitude)
            if tangent_magnitude > _EPS
            else np.zeros(3)
        )
        initial_velocity = center_velocity[0]
        contact_position = (
            centers[0] + initial_velocity * contact_time + 0.5 * gravity * contact_time**2
        )
        contact_velocity = initial_velocity + gravity * contact_time
        contact_velocity -= np.dot(contact_velocity, normal) * normal
        expected_position = []
        expected_velocity = []
        for value in tau:
            if value <= contact_time:
                expected_position.append(
                    centers[0] + initial_velocity * value + 0.5 * gravity * value**2
                )
                expected_velocity.append(initial_velocity + gravity * value)
            else:
                slide_time = value - contact_time
                expected_position.append(
                    contact_position
                    + contact_velocity * slide_time
                    + 0.5 * slide_acceleration * slide_time**2
                )
                expected_velocity.append(contact_velocity + slide_acceleration * slide_time)
        expected_position = np.asarray(expected_position)
        expected_velocity = np.asarray(expected_velocity)
        observable = str(parameters.get("observable", "speed"))
        if observable == "speed":
            actual = np.linalg.norm(center_velocity, axis=1)
            expected = np.linalg.norm(expected_velocity, axis=1)
            scale = max(
                float(np.max(np.abs(expected))), tangent_magnitude * max(tau[-1], 1e-3), 1e-3
            )
        elif observable == "relative_centroid_height":
            actual = centers[:, 2] - centers[0, 2]
            expected = expected_position[:, 2] - expected_position[0, 2]
            scale = max(float(np.ptp(expected)), initial_gap, 1e-3)
        else:
            raise ValueError("observable must be speed or relative_centroid_height")
        return CalculatorResult(
            np.abs(actual - expected) / scale,
            {
                "observable": observable,
                "plane_object_id": plane.id,
                "contact_time_seconds": contact_time,
                "friction_coefficient": mu,
                "reference_scale": scale,
            },
        )


class OscillationFrequencyPhaseCalculator:
    id = "oscillation.frequency_phase.v1"
    metric_name = "oscillation_frequency_phase_error"
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
            item = _find_object(
                context,
                str(resolved.get("object_id", "pendulum_body")),
                *resolved.get("object_keywords", ["pendulum", "weight", "mass"]),
            )
            binding = item.state_binding
            if binding is None or binding.selector != "body_id":
                raise ValueError("oscillating body requires a body_id state binding")
            resolved["body_id"] = int(binding.value)
        return self.compute(frames, inputs, resolved)

    def compute(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
    ) -> CalculatorResult:
        if len(frames) < 8:
            raise ValueError("oscillation analysis requires at least eight frames")
        positions = input_series(frames, inputs, "position")
        angular = input_series(frames, inputs, "angular_velocity")
        ids = input_series(frames, inputs, "body_id")
        target = int(parameters.get("body_id", 2))
        vertical = []
        torsional = []
        for position, omega, body_ids in zip(positions, angular, ids, strict=True):
            mask = np.asarray(body_ids) == target
            if not np.any(mask):
                raise ValueError(f"body_id={target} has no samples")
            vertical.append(float(np.asarray(position)[mask, 2].mean()))
            torsional.append(float(np.asarray(omega)[mask, 2].mean()))
        vertical = np.asarray(vertical) - np.mean(vertical)
        torsional = np.asarray(torsional) - np.mean(torsional)
        dt = float(np.median(np.diff([frame.time for frame in frames])))
        if dt <= 0:
            raise ValueError("trajectory times must increase")

        def dominant(signal: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
            frequencies = np.fft.rfftfreq(len(signal), dt)
            if len(spectrum) <= 1 or np.max(spectrum[1:]) <= _EPS:
                return 0.0
            return float(frequencies[1 + np.argmax(spectrum[1:])])

        f_vertical = dominant(vertical)
        f_torsional = dominant(torsional)
        target_cycles = float(parameters.get("minimum_cycles", 1.5))
        duration = frames[-1].time - frames[0].time
        cycles = min(f_vertical, f_torsional) * duration
        frequency_error = abs(f_vertical - f_torsional) / max(f_vertical, f_torsional, _EPS)
        activation_error = float(
            np.linalg.norm(torsional) <= 1e-8 or np.linalg.norm(vertical) <= 1e-8
        )
        error = max(
            frequency_error, max(0.0, target_cycles - cycles) / target_cycles, activation_error
        )
        return CalculatorResult(
            np.full(len(frames), error),
            {
                "body_id": target,
                "vertical_frequency_hz": f_vertical,
                "torsional_frequency_hz": f_torsional,
                "observed_common_cycles": cycles,
            },
        )


class RodAttachmentInextensibilityCalculator:
    id = "rod.attachment_inextensibility.v1"
    metric_name = "rod_attachment_and_edge_strain"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        spring = _find_object(
            context, str(parameters.get("rod_object_id", "helical_spring")), "spring"
        )
        body = _find_object(
            context, str(parameters.get("attached_object_id", "pendulum_body")), "pendulum"
        )
        topology_path = spring.geometry.topology_asset
        if not topology_path:
            raise ValueError("rod object requires a topology_asset with edges")
        with open(topology_path, encoding="utf-8") as topology_file:
            edges = np.asarray(json.load(topology_file)["edges"], dtype=int)
        positions = input_series(frames, inputs, "position")
        first_rod = np.asarray(positions[0], dtype=float)[_body_mask(frames[0], spring)]
        if edges.ndim != 2 or edges.shape[1] != 2 or np.max(edges) >= len(first_rod):
            raise ValueError("rod topology edges do not match bound trajectory samples")
        rest_lengths = np.linalg.norm(first_rod[edges[:, 1]] - first_rod[edges[:, 0]], axis=1)
        first_body = _centroid(positions[0], _body_mask(frames[0], body))
        degrees = np.bincount(edges.ravel(), minlength=len(first_rod))
        endpoint_indices = np.flatnonzero(degrees == 1)
        attached_index = int(
            endpoint_indices[
                np.argmin(np.linalg.norm(first_rod[endpoint_indices] - first_body, axis=1))
            ]
        )
        initial_attachment = float(np.linalg.norm(first_rod[attached_index] - first_body))
        length_scale = max(float(np.median(rest_lengths)), initial_attachment, 1e-6)
        errors = []
        for frame, position in zip(frames, positions, strict=True):
            rod = np.asarray(position, dtype=float)[_body_mask(frame, spring)]
            body_center = _centroid(position, _body_mask(frame, body))
            lengths = np.linalg.norm(rod[edges[:, 1]] - rod[edges[:, 0]], axis=1)
            strain = float(np.max(np.abs(lengths - rest_lengths) / np.maximum(rest_lengths, 1e-6)))
            attachment = (
                abs(float(np.linalg.norm(rod[attached_index] - body_center)) - initial_attachment)
                / length_scale
            )
            errors.append(max(strain, attachment))
        return CalculatorResult(
            np.asarray(errors),
            {
                "rod_object_id": spring.id,
                "attached_object_id": body.id,
                "attached_vertex": attached_index,
            },
        )


def _cantilever_config(context: EvaluationContext) -> tuple[np.ndarray, float, float]:
    config_path = context.config_path or context.result_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError("cantilever calculator requires config.yaml")
    config = load_yaml(config_path)
    gravity = np.asarray(config["gravity"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= _EPS:
        raise ValueError("cantilever config gravity must be a nonzero three-vector")
    length = float(config["rod"]["length"])
    full_load_time = float(config["load_schedule"].get("full_load_time", 0.0))
    if length <= 0.0 or full_load_time < 0.0:
        raise ValueError("cantilever config has invalid rod length or load schedule")
    return gravity, length, full_load_time


def _cantilever_state(frame: StateFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(frame.position, dtype=float)
    mass = np.asarray(frame.mass, dtype=float) if frame.mass is not None else None
    if position.ndim != 2 or position.shape[1] != 3 or mass is None:
        raise ValueError("cantilever balance requires state.position and state.mass")
    if mass.shape != (len(position),) or np.any(mass < 0.0) or not np.isfinite(mass).all():
        raise ValueError("cantilever state.mass must be a finite nonnegative [N] array")
    if mass.sum() <= 0.0:
        raise ValueError("cantilever state.mass must have positive total mass")
    return position, mass, position[0]


def _cantilever_load(time: float, full_load_time: float) -> float:
    if full_load_time <= 0.0:
        return 1.0
    return float(np.clip(time / full_load_time, 0.0, 1.0))


class RodCantileverInextensibilityCalculator:
    """Measure rod strain from the submitted centerline, not a self-reported field."""

    id = "rod.cantilever_inextensibility.v1"
    metric_name = "rod_actual_edge_strain"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        if parameters:
            raise ValueError(f"Unsupported parameters: {sorted(parameters)}")
        _, length, _ = _cantilever_config(context)
        positions = input_series(frames, inputs, "position")
        first = np.asarray(positions[0], dtype=float)
        if first.ndim != 2 or first.shape[1] != 3 or len(first) < 2:
            raise ValueError("cantilever position must have shape [N,3] with N>=2")
        reference = length / (len(first) - 1)
        errors = []
        for position in positions:
            current = np.asarray(position, dtype=float)
            if current.shape != first.shape:
                raise ValueError("cantilever sample count changed across frames")
            edge_lengths = np.linalg.norm(np.diff(current, axis=0), axis=1)
            errors.append(float(np.max(np.abs(edge_lengths - reference) / reference)))
        return CalculatorResult(np.asarray(errors), {"reference_edge_length": reference})


class _RodCantileverStaticBalanceCalculator:
    angular = False

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        if parameters:
            raise ValueError(f"Unsupported parameters: {sorted(parameters)}")
        gravity, length, full_load_time = _cantilever_config(context)
        gravity_hat = gravity / np.linalg.norm(gravity)
        errors = []
        for frame in frames:
            position, mass, root = _cantilever_state(frame)
            force = np.asarray(frame_value(frame, inputs["external_force"]), dtype=float)
            points = np.asarray(frame_value(frame, inputs["force_position"]), dtype=float)
            couple = np.asarray(frame_value(frame, inputs["external_couple"]), dtype=float)
            if force.ndim != 3 or force.shape[0] != 1 or force.shape[2] != 3:
                raise ValueError("cantilever external_force must have shape [1,K,3]")
            if points.shape != force.shape or couple.shape != force.shape:
                raise ValueError("cantilever force positions and couples must match external_force")
            scale_name = "reference_torque" if self.angular else "reference_force"
            scale = float(frame_value(frame, inputs[scale_name]))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{scale_name} must be a positive finite scalar")
            force = force[0]
            points = points[0]
            couple = couple[0]
            load = _cantilever_load(frame.time, full_load_time)
            expected_gravity = mass.sum() * gravity * load
            net_force_error = np.linalg.norm(force.sum(axis=0)) / scale
            if load <= 1.0e-12:
                source_error = np.linalg.norm(force) / scale
                gravity_mask = np.zeros(len(force), dtype=bool)
                reaction_mask = np.zeros(len(force), dtype=bool)
            else:
                projection = force @ gravity_hat
                gravity_mask = projection >= 0.0
                reaction_mask = projection < 0.0
                source_error = max(
                    np.linalg.norm(force[gravity_mask].sum(axis=0) - expected_gravity) / scale,
                    np.linalg.norm(force[reaction_mask].sum(axis=0) + expected_gravity) / scale,
                )
                reaction_points = points[reaction_mask]
                root_error = (
                    min(np.linalg.norm(reaction_points - root, axis=1)) / length
                    if len(reaction_points)
                    else 1.0
                )
                source_error = max(source_error, root_error)
            if not self.angular:
                errors.append(max(net_force_error, source_error))
                continue

            actual_gravity_torque = np.cross(
                position - root, mass[:, None] * gravity * load
            ).sum(axis=0)
            reported_torque = np.cross(points - root, force) + couple
            total_torque_error = np.linalg.norm(reported_torque.sum(axis=0)) / scale
            if load <= 1.0e-12:
                torque_source_error = np.linalg.norm(reported_torque) / scale
            else:
                gravity_torque = reported_torque[gravity_mask].sum(axis=0)
                reaction_torque = reported_torque[reaction_mask].sum(axis=0)
                torque_source_error = max(
                    np.linalg.norm(gravity_torque - actual_gravity_torque) / scale,
                    np.linalg.norm(reaction_torque + actual_gravity_torque) / scale,
                    np.linalg.norm(couple[gravity_mask].sum(axis=0)) / scale,
                )
            errors.append(max(net_force_error, source_error, total_torque_error, torque_source_error))
        return CalculatorResult(
            np.asarray(errors),
            {
                "check": "actual centerline, mass-weighted gravity, force locations, and clamp reaction",
                "gravity": gravity.tolist(),
            },
        )


class RodCantileverStaticForceBalanceCalculator(_RodCantileverStaticBalanceCalculator):
    id = "rod.cantilever_force_balance.v1"
    metric_name = "rod_actual_static_force_balance"
    metric_unit = "dimensionless"


class RodCantileverStaticMomentBalanceCalculator(_RodCantileverStaticBalanceCalculator):
    id = "rod.cantilever_moment_balance.v1"
    metric_name = "rod_actual_static_moment_balance"
    metric_unit = "dimensionless"
    angular = True


class KinematicContinuityCalculator:
    """Check that a bound object's position increments agree with its velocity trace."""

    id = "trajectory.kinematic_continuity.v1"
    metric_name = "trajectory_kinematic_continuity_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        if len(frames) < 2:
            raise ValueError("kinematic continuity requires at least two frames")
        requested_id = str(parameters.get("object_id", "tracked_object"))
        item = _find_object(context, requested_id, *parameters.get("object_keywords", []))
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        errors = [0.0]
        maximum_residual = 0.0
        for (
            previous,
            current,
            previous_position,
            current_position,
            previous_velocity,
            current_velocity,
        ) in zip(
            frames[:-1],
            frames[1:],
            positions[:-1],
            positions[1:],
            velocities[:-1],
            velocities[1:],
            strict=True,
        ):
            dt = float(current.time - previous.time)
            if dt <= 0.0:
                raise ValueError("trajectory times must increase")
            previous_points = np.asarray(previous_position, dtype=float)[_body_mask(previous, item)]
            current_points = np.asarray(current_position, dtype=float)[_body_mask(current, item)]
            previous_speeds = np.asarray(previous_velocity, dtype=float)[_body_mask(previous, item)]
            current_speeds = np.asarray(current_velocity, dtype=float)[_body_mask(current, item)]
            if previous_points.shape != current_points.shape:
                raise ValueError("bound object sample count changed between trajectory frames")
            observed_step = current_points - previous_points
            expected_step = 0.5 * (previous_speeds + current_speeds) * dt
            residual = float(np.sqrt(np.mean(np.sum((observed_step - expected_step) ** 2, axis=1))))
            observed_scale = float(np.sqrt(np.mean(np.sum(observed_step**2, axis=1))))
            expected_scale = float(np.sqrt(np.mean(np.sum(expected_step**2, axis=1))))
            scale = max(observed_scale, expected_scale, 1e-8)
            maximum_residual = max(maximum_residual, residual)
            errors.append(residual / scale)
        return CalculatorResult(
            np.asarray(errors),
            {
                "object_id": item.id,
                "maximum_position_residual": maximum_residual,
            },
        )


def _cylinder_data(
    context: EvaluationContext, parameters: dict[str, Any]
) -> list[tuple[str, np.ndarray, np.ndarray, float, float]]:
    requested = parameters.get("cylinder_object_ids")
    objects = _objects(context)
    if requested:
        cylinders = [_find_object(context, str(item), str(item)) for item in requested]
    else:
        cylinders = [
            item for item in objects if item.geometry.type == "cylinder" and item.motion == "static"
        ]
    if not cylinders:
        raise ValueError("no static cylinder objects found")
    result = []
    for item in cylinders:
        if item.geometry.type != "cylinder":
            raise ValueError(f"{item.id!r} is not a cylinder")
        rotation = _quat_matrix(item.transform.rotation_xyzw)
        axis = rotation @ np.asarray([0.0, 0.0, 1.0])
        scale = np.asarray(item.transform.scale, dtype=float)
        radius = float(item.geometry.radius * np.mean(scale[:2]))
        height = float(item.geometry.height * scale[2])
        result.append(
            (item.id, np.asarray(item.transform.translation, dtype=float), axis, radius, height)
        )
    return result


class ContactCoverageCalculator:
    id = "contact.coverage.v1"
    metric_name = "multi_object_contact_coverage_deficit"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        positions = input_series(frames, inputs, "position")
        cylinders = _cylinder_data(context, parameters)
        tolerance_fraction = float(parameters.get("contact_band_radius_fraction", 0.18))
        minimum = float(parameters.get("minimum_vertex_fraction", 0.01))
        errors = []
        final_coverages = {}
        for frame_position in positions:
            points = np.asarray(frame_position, dtype=float)
            coverages = []
            for object_id, center, axis, radius, height in cylinders:
                delta = points - center
                axial = delta @ axis
                radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
                contact = (np.abs(radial - radius) <= tolerance_fraction * radius) & (
                    np.abs(axial) <= height / 2
                )
                coverage = float(np.mean(contact))
                coverages.append(coverage)
                final_coverages[object_id] = coverage
            errors.append(
                float(np.mean([max(0.0, minimum - value) / minimum for value in coverages]))
            )
        return CalculatorResult(np.asarray(errors), {"final_contact_fractions": final_coverages})


class LoopWrapCalculator:
    id = "geometry.loop_wrap.v1"
    metric_name = "closed_loop_wrap_deficit"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        from scipy.spatial import ConvexHull

        positions = input_series(frames, inputs, "position")
        cylinders = _cylinder_data(context, parameters)
        reference_scale = None
        errors = []
        final_inside = {}
        for frame_position in positions:
            points = np.asarray(frame_position, dtype=float)[:, :2]
            if len(points) < 3:
                raise ValueError("loop-wrap analysis requires at least three cloth points")
            hull = ConvexHull(points)
            polygon = points[hull.vertices]
            area = float(hull.volume)
            if reference_scale is None:
                reference_scale = max(area, 1e-6)
            inside_values = []
            for object_id, center, _axis, _radius, _height in cylinders:
                edges = np.roll(polygon, -1, axis=0) - polygon
                offsets = center[:2] - polygon
                cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
                inside = bool(np.all(cross >= -1e-8) or np.all(cross <= 1e-8))
                inside_values.append(inside)
                final_inside[object_id] = inside
            area_error = abs(area - reference_scale) / reference_scale
            errors.append(max(float(np.mean(np.logical_not(inside_values))), area_error))
        return CalculatorResult(np.asarray(errors), {"final_cylinder_inside_loop": final_inside})


def _bound_cylinder_pose(
    frame: StateFrame, item: SceneObject
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cylinder center and world-space axis from its bound pose."""
    if item.geometry.type != "cylinder":
        raise ValueError(f"{item.id!r} is not a cylinder")
    if item.state_binding is None:
        center = np.asarray(item.transform.translation, dtype=float)
        orientation = np.asarray(item.transform.rotation_xyzw, dtype=float)
    else:
        mask = _body_mask(frame, item)
        center = _centroid(frame.position, mask)
        if frame.orientation is None:
            raise ValueError("state.orientation is required for bound cylinders")
        orientation = np.asarray(frame.orientation, dtype=float)[mask][0]
    axis = _quat_matrix(orientation) @ np.asarray([0.0, 0.0, 1.0])
    axis /= max(float(np.linalg.norm(axis)), _EPS)
    return center, axis


def _belt_grid(
    frame: StateFrame, item: SceneObject
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a finite-width belt and return points, centerline, and its state mask."""
    mask = _body_mask(frame, item)
    points = np.asarray(frame.position, dtype=float)[mask]
    shape = item.geometry.grid_shape
    if shape is None or len(shape) != 2:
        raise ValueError("belt triangle_mesh requires grid_shape=[ring_count,width_nodes]")
    ring_count, width_nodes = map(int, shape)
    if ring_count < 6 or width_nodes < 2 or len(points) != ring_count * width_nodes:
        raise ValueError("belt grid_shape does not match its bound trajectory samples")
    return points, points.reshape(ring_count, width_nodes, 3).mean(axis=1), mask


def _cylinder_gap(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(points, dtype=float) - center
    axial = delta @ axis
    radial_vector = delta - axial[:, None] * axis
    radial = np.linalg.norm(radial_vector, axis=1)
    gap = radial - radius
    within_axis = np.abs(axial) <= half_height
    return gap, radial_vector, within_axis


class BeltGeometricContactCalculator:
    """Measure belt-cylinder contact from geometry rather than solver-reported gaps."""

    id = "belt.geometric_contact.v1"
    metric_name = "geometry_derived_belt_contact_error"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        belt = _find_object(
            context, str(parameters.get("belt_object_id", "closed_cloth_belt")), "belt"
        )
        cylinder_ids = parameters.get(
            "cylinder_object_ids", ["cylinder_left", "cylinder_right", "cylinder_top"]
        )
        cylinders = [_find_object(context, str(value), str(value)) for value in cylinder_ids]
        positions = input_series(frames, inputs, "position")
        body_ids = input_series(frames, inputs, "body_id")
        thickness = float(parameters.get("thickness", 0.006))
        band = float(parameters.get("contact_band", 0.035))
        minimum = float(parameters.get("minimum_contact_fraction", 0.025))
        if thickness <= 0 or band <= 0 or minimum <= 0:
            raise ValueError("belt contact scales must be positive")
        errors = []
        final_fractions: dict[str, float] = {}
        for frame, position, ids in zip(frames, positions, body_ids, strict=True):
            frame.position = np.asarray(position, dtype=float)
            frame.body_id = np.asarray(ids)
            points, _centerline, _mask = _belt_grid(frame, belt)
            penetration = 0.0
            deficits = []
            for item in cylinders:
                center, axis = _bound_cylinder_pose(frame, item)
                radius = float(item.geometry.radius)
                half_height = float(item.geometry.height) / 2.0
                gap, _radial, within = _cylinder_gap(
                    points, center, axis, radius, half_height
                )
                if np.any(within):
                    penetration = max(
                        penetration,
                        float(np.max(np.maximum(-gap[within], 0.0))) / thickness,
                    )
                fraction = float(np.mean(within & (np.abs(gap) <= band)))
                final_fractions[item.id] = fraction
                deficits.append(max(0.0, minimum - fraction) / minimum)
            errors.append(max(penetration, float(np.mean(deficits))))
        return CalculatorResult(
            np.asarray(errors), {"final_contact_fractions": final_fractions}
        )


def _cyclic_free_runs(free: np.ndarray) -> list[np.ndarray]:
    count = len(free)
    if count == 0 or not np.any(free):
        return []
    if np.all(free):
        return [np.arange(count)]
    start = int(np.flatnonzero(~free)[0])
    runs: list[list[int]] = []
    current: list[int] = []
    for offset in range(1, count + 1):
        index = (start + offset) % count
        if free[index]:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    return [np.asarray(run, dtype=int) for run in runs]


class BeltTensionedTriangleCalculator:
    """Check that non-contact belt spans form three straight, taut sides."""

    id = "belt.tensioned_triangle.v1"
    metric_name = "tensioned_triangle_shape_error"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        belt = _find_object(
            context, str(parameters.get("belt_object_id", "closed_cloth_belt")), "belt"
        )
        cylinders = [
            _find_object(context, str(value), str(value))
            for value in parameters.get(
                "cylinder_object_ids",
                ["cylinder_left", "cylinder_right", "cylinder_top"],
            )
        ]
        positions = input_series(frames, inputs, "position")
        body_ids = input_series(frames, inputs, "body_id")
        start_time = float(parameters.get("evaluation_start_time", 0.0))
        band = float(parameters.get("contact_band", 0.05))
        tolerance = float(parameters.get("maximum_span_deviation", 0.035))
        required_spans = int(parameters.get("minimum_span_count", 3))
        if band <= 0 or tolerance <= 0 or required_spans < 1:
            raise ValueError("belt triangle parameters are invalid")
        errors = []
        final_deviations: list[float] = []
        for frame, position, ids in zip(frames, positions, body_ids, strict=True):
            frame.position = np.asarray(position, dtype=float)
            frame.body_id = np.asarray(ids)
            if frame.time < start_time:
                errors.append(0.0)
                continue
            _points, centerline, _mask = _belt_grid(frame, belt)
            distance_columns = []
            for item in cylinders:
                center, axis = _bound_cylinder_pose(frame, item)
                gap, _radial, within = _cylinder_gap(
                    centerline,
                    center,
                    axis,
                    float(item.geometry.radius),
                    float(item.geometry.height) / 2.0,
                )
                distance_columns.append(np.where(within, np.abs(gap), np.inf))
            distances = np.stack(distance_columns, axis=1)
            free = np.min(distances, axis=1) > band
            runs = [run for run in _cyclic_free_runs(free) if len(run) >= 3]
            deviations = []
            for run in runs:
                span = centerline[run]
                centered = span - span.mean(axis=0)
                _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
                orthogonal = centered - (centered @ direction)[:, None] * direction
                rms_deviation = float(np.sqrt(np.mean(np.sum(orthogonal**2, axis=1))))
                deviations.append(rms_deviation)
            span_deficit = max(0.0, required_spans - len(deviations)) / required_spans
            straightness = (
                max(max(deviations) - tolerance, 0.0) / tolerance
                if deviations
                else 1.0
            )
            errors.append(max(span_deficit, straightness))
            final_deviations = deviations
        return CalculatorResult(
            np.asarray(errors),
            {
                "final_span_count": len(final_deviations),
                "final_span_rms_deviations": final_deviations,
            },
        )


class BeltSurfaceVelocityTransferCalculator:
    """Compare belt velocity with the analytic surface velocity of driven cylinders."""

    id = "belt.surface_velocity_transfer.v1"
    metric_name = "belt_cylinder_slip_excess"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        belt = _find_object(
            context, str(parameters.get("belt_object_id", "closed_cloth_belt")), "belt"
        )
        cylinders = [
            _find_object(context, str(value), str(value))
            for value in parameters.get(
                "cylinder_object_ids", ["cylinder_left", "cylinder_right"]
            )
        ]
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        angular = input_series(frames, inputs, "angular_velocity")
        body_ids = input_series(frames, inputs, "body_id")
        band = float(parameters.get("contact_band", 0.05))
        minimum_speed = float(parameters.get("minimum_surface_speed", 0.1))
        allowed_slip = float(parameters.get("maximum_slip_ratio", 0.3))
        errors = []
        slip_history = []
        for frame, position, velocity, omega, ids in zip(
            frames, positions, velocities, angular, body_ids, strict=True
        ):
            frame.position = np.asarray(position, dtype=float)
            frame.velocity = np.asarray(velocity, dtype=float)
            frame.angular_velocity = np.asarray(omega, dtype=float)
            frame.body_id = np.asarray(ids)
            points, _centerline, belt_mask = _belt_grid(frame, belt)
            belt_velocity = np.asarray(frame.velocity)[belt_mask]
            driven_slips = []
            for item in cylinders:
                center, axis = _bound_cylinder_pose(frame, item)
                gap, radial, within = _cylinder_gap(
                    points,
                    center,
                    axis,
                    float(item.geometry.radius),
                    float(item.geometry.height) / 2.0,
                )
                contact = within & (np.abs(gap) <= band)
                cylinder_mask = _body_mask(frame, item)
                cylinder_omega = np.asarray(frame.angular_velocity)[cylinder_mask].mean(axis=0)
                surface_speed = float(np.linalg.norm(cylinder_omega) * item.geometry.radius)
                if surface_speed < minimum_speed:
                    continue
                if not np.any(contact):
                    driven_slips.append(1.0 + allowed_slip)
                    continue
                surface_velocity = np.cross(
                    np.broadcast_to(cylinder_omega, radial[contact].shape), radial[contact]
                )
                tangent = np.cross(
                    np.broadcast_to(axis, radial[contact].shape), radial[contact]
                )
                tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], _EPS)
                cylinder_tangent_speed = np.sum(surface_velocity * tangent, axis=1)
                belt_tangent_speed = np.sum(belt_velocity[contact] * tangent, axis=1)
                slip = float(
                    np.mean(
                        np.abs(belt_tangent_speed - cylinder_tangent_speed)
                        / np.maximum(np.abs(cylinder_tangent_speed), minimum_speed)
                    )
                )
                driven_slips.append(slip)
            if driven_slips:
                slip_value = float(np.mean(driven_slips))
                error = max(0.0, slip_value - allowed_slip) / allowed_slip
            else:
                slip_value = 0.0
                error = 0.0
            errors.append(error)
            slip_history.append(slip_value)
        return CalculatorResult(
            np.asarray(errors),
            {
                "maximum_slip_ratio": float(np.max(slip_history)),
                "final_slip_ratio": float(slip_history[-1]),
            },
        )


class BeltDriveReversalCalculator:
    """Track signed tangential transport of fixed material rings across drive phases."""

    id = "belt.drive_reversal.v1"
    metric_name = "forward_pause_reverse_transport_error"
    metric_unit = "dimensionless"

    @staticmethod
    def _interval_mask(times: np.ndarray, interval: list[float]) -> np.ndarray:
        if len(interval) != 2 or interval[1] <= interval[0]:
            raise ValueError("drive interval must be [start,end] with end > start")
        return (times >= float(interval[0])) & (times < float(interval[1]))

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        belt = _find_object(
            context, str(parameters.get("belt_object_id", "closed_cloth_belt")), "belt"
        )
        positions = input_series(frames, inputs, "position")
        velocities = input_series(frames, inputs, "velocity")
        body_ids = input_series(frames, inputs, "body_id")
        times = np.asarray([frame.time for frame in frames], dtype=float)
        signed_speeds = []
        marker_indices = None
        for frame, position, velocity, ids in zip(
            frames, positions, velocities, body_ids, strict=True
        ):
            frame.position = np.asarray(position, dtype=float)
            frame.velocity = np.asarray(velocity, dtype=float)
            frame.body_id = np.asarray(ids)
            _points, centerline, belt_mask = _belt_grid(frame, belt)
            shape = belt.geometry.grid_shape
            ring_count, width_nodes = map(int, shape)
            center_velocity = np.asarray(frame.velocity)[belt_mask].reshape(
                ring_count, width_nodes, 3
            ).mean(axis=1)
            if marker_indices is None:
                marker_indices = np.linspace(0, ring_count, 6, endpoint=False, dtype=int)
            previous = centerline[(marker_indices - 1) % ring_count]
            following = centerline[(marker_indices + 1) % ring_count]
            tangent = following - previous
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], _EPS)
            signed_speeds.append(
                float(np.mean(np.sum(center_velocity[marker_indices] * tangent, axis=1)))
            )
        speeds = np.asarray(signed_speeds)
        forward = self._interval_mask(times, parameters["forward_interval"])
        pause = self._interval_mask(times, parameters["pause_interval"])
        reverse = self._interval_mask(times, parameters["reverse_interval"])
        if min(np.count_nonzero(forward), np.count_nonzero(pause), np.count_nonzero(reverse)) < 2:
            raise ValueError("each belt drive phase requires at least two trajectory samples")
        forward_displacement = float(np.trapezoid(speeds[forward], times[forward]))
        reverse_displacement = float(np.trapezoid(speeds[reverse], times[reverse]))
        minimum = float(parameters.get("minimum_phase_displacement", 0.2))
        maximum_pause_fraction = float(parameters.get("maximum_pause_speed_fraction", 0.2))
        if minimum <= 0 or maximum_pause_fraction <= 0:
            raise ValueError("belt reversal thresholds must be positive")
        forward_deficit = max(0.0, minimum - abs(forward_displacement)) / minimum
        reverse_deficit = max(0.0, minimum - abs(reverse_displacement)) / minimum
        sign_error = float(forward_displacement * reverse_displacement >= 0.0)
        drive_speed = max(
            float(np.mean(np.abs(speeds[forward]))),
            float(np.mean(np.abs(speeds[reverse]))),
            _EPS,
        )
        pause_fraction = float(np.mean(np.abs(speeds[pause])) / drive_speed)
        pause_error = max(0.0, pause_fraction - maximum_pause_fraction) / maximum_pause_fraction
        symmetry_error = abs(abs(forward_displacement) - abs(reverse_displacement)) / max(
            abs(forward_displacement), abs(reverse_displacement), minimum
        )
        error = max(
            forward_deficit,
            reverse_deficit,
            sign_error,
            pause_error,
            symmetry_error,
        )
        return CalculatorResult(
            np.full(len(frames), error),
            {
                "forward_displacement": forward_displacement,
                "reverse_displacement": reverse_displacement,
                "pause_speed_fraction": pause_fraction,
                "marker_ring_indices": marker_indices.tolist(),
            },
        )


def _feature_distance(points: np.ndarray, item: SceneObject) -> np.ndarray:
    center = np.asarray(item.transform.translation, dtype=float)
    local = (points - center) @ _quat_matrix(item.transform.rotation_xyzw)
    scale = np.asarray(item.transform.scale, dtype=float)
    geometry = item.geometry
    if geometry.type == "sphere":
        return np.abs(np.linalg.norm(local, axis=1) - float(geometry.radius * np.mean(scale)))
    if geometry.type in {"cylinder", "capsule"}:
        radius = float(geometry.radius * np.mean(scale[:2]))
        half = float(geometry.height * scale[2] / 2)
        radial = np.linalg.norm(local[:, :2], axis=1)
        return np.sqrt(
            np.square(np.maximum(radial - radius, 0.0))
            + np.square(np.maximum(np.abs(local[:, 2]) - half, 0.0))
        )
    if geometry.type == "box":
        half = np.asarray(geometry.half_extents, dtype=float) * scale
        return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
    if geometry.type == "plane":
        normal = np.asarray(geometry.normal, dtype=float)
        return np.abs(local @ normal - float(geometry.offset or 0.0))
    return np.linalg.norm(local, axis=1)


class FeatureContactCoverageCalculator:
    id = "contact.feature_coverage.v1"
    metric_name = "edge_point_contact_event_deficit"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        edge = _find_object(context, str(parameters.get("edge_object_id", "sharp_edge")), "edge")
        point = _find_object(
            context, str(parameters.get("point_object_id", "sharp_point")), "point", "tip"
        )
        tolerance = float(parameters.get("contact_distance", 0.04))
        minimum = int(parameters.get("minimum_contact_samples", 1))
        positions = input_series(frames, inputs, "position")
        seen_edge = False
        seen_point = False
        errors = []
        for value in positions:
            points = np.asarray(value, dtype=float)
            edge_count = int(np.count_nonzero(_feature_distance(points, edge) <= tolerance))
            point_count = int(np.count_nonzero(_feature_distance(points, point) <= tolerance))
            seen_edge |= edge_count >= minimum
            seen_point |= point_count >= minimum
            errors.append(1.0 - 0.5 * (seen_edge + seen_point))
        return CalculatorResult(
            np.asarray(errors), {"edge_object_id": edge.id, "point_object_id": point.id}
        )


class LocalDeformationSignatureCalculator:
    id = "geometry.local_deformation_signature.v1"
    metric_name = "edge_point_local_shape_deficit"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        edge = _find_object(context, str(parameters.get("edge_object_id", "sharp_edge")), "edge")
        point = _find_object(
            context, str(parameters.get("point_object_id", "sharp_point")), "point", "tip"
        )
        radius = float(parameters.get("neighborhood_radius", 0.15))
        minimum = int(parameters.get("minimum_neighborhood_samples", 3))
        positions = input_series(frames, inputs, "position")
        errors = []
        signatures = {}
        for value in positions:
            points = np.asarray(value, dtype=float)
            local_errors = []
            for label, item, target_anisotropy in (("edge", edge, 2.0), ("point", point, 1.0)):
                selected = points[_feature_distance(points, item) <= radius]
                if len(selected) < minimum:
                    local_errors.append(1.0)
                    continue
                eigenvalues = np.linalg.eigvalsh(np.cov(selected, rowvar=False))
                anisotropy = float(eigenvalues[-1] / max(eigenvalues[-2], 1e-10))
                if label == "edge":
                    local_errors.append(
                        max(0.0, target_anisotropy - anisotropy) / target_anisotropy
                    )
                else:
                    local_errors.append(max(0.0, anisotropy - 4.0) / 4.0)
                signatures[label] = anisotropy
            errors.append(float(np.mean(local_errors)))
        return CalculatorResult(np.asarray(errors), {"final_neighborhood_anisotropy": signatures})


def _rigid_objects(context: EvaluationContext) -> list[SceneObject]:
    result = [
        item
        for item in _objects(context)
        if item.semantic == "rigid" and item.motion == "dynamic" and item.state_binding is not None
    ]
    if not result:
        raise ValueError("scene has no bound dynamic rigid objects")
    return result


class RigidPoseTrajectoryCalculator:
    id = "rigid.pose_trajectory.v1"
    metric_name = "rigid_pose_envelope_error"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        positions = input_series(frames, inputs, "position")
        orientations = input_series(frames, inputs, "orientation")
        objects = _rigid_objects(context)
        max_displacement = float(parameters.get("maximum_displacement", 0.75))
        max_tilt = np.deg2rad(float(parameters.get("maximum_tilt_degrees", 70.0)))
        references = {}
        for item in objects:
            mask = _body_mask(frames[0], item)
            references[item.id] = (
                _centroid(positions[0], mask),
                np.asarray(orientations[0])[mask][0],
            )
        errors = []
        final_pose = {}
        for frame, position, orientation in zip(frames, positions, orientations, strict=True):
            body_errors = []
            for item in objects:
                mask = _body_mask(frame, item)
                center = _centroid(position, mask)
                quaternion = np.asarray(orientation, dtype=float)[mask][0]
                reference_center, reference_quaternion = references[item.id]
                displacement = float(np.linalg.norm(center - reference_center))
                q0 = reference_quaternion / max(np.linalg.norm(reference_quaternion), _EPS)
                q1 = quaternion / max(np.linalg.norm(quaternion), _EPS)
                tilt = 2.0 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0))
                body_errors.append(
                    max(
                        _hinge(displacement, None, max_displacement, max_displacement),
                        _hinge(tilt, None, max_tilt, max_tilt),
                    )
                )
                final_pose[item.id] = {
                    "displacement": displacement,
                    "rotation_radians": float(tilt),
                }
            errors.append(float(np.mean(body_errors)))
        return CalculatorResult(np.asarray(errors), {"final_pose": final_pose})


class ObjectRetentionCalculator:
    id = "outcome.object_retention.v1"
    metric_name = "table_retention_outcome_error"
    metric_unit = "dimensionless"

    def compute_with_context(self, frames, inputs, parameters, context) -> CalculatorResult:
        table = _find_object(
            context, str(parameters.get("table_object_id", "table")), "table", "top"
        )
        if table.geometry.type != "box":
            raise ValueError("object retention requires a box table object")
        objects = _rigid_objects(context)
        positions = input_series(frames, inputs, "position")
        final = frames[-1]
        center = np.asarray(table.transform.translation, dtype=float)
        half = np.asarray(table.geometry.half_extents, dtype=float) * np.asarray(
            table.transform.scale
        )
        retained = {}
        displaced = {}
        for item in objects:
            initial_center = _centroid(positions[0], _body_mask(frames[0], item))
            final_center = _centroid(positions[-1], _body_mask(final, item))
            on_table = bool(
                abs(final_center[0] - center[0]) <= half[0]
                and abs(final_center[1] - center[1]) <= half[1]
                and final_center[2]
                >= center[2] + half[2] - float(parameters.get("height_tolerance", 0.15))
            )
            retained[item.id] = on_table
            displaced[item.id] = float(np.linalg.norm(final_center - initial_center))
        retention_fraction = float(np.mean(list(retained.values())))
        displacement_threshold = float(parameters.get("displacement_threshold", 0.10))
        displaced_fraction = float(
            np.mean([value >= displacement_threshold for value in displaced.values()])
        )
        error = max(
            _hinge(
                retention_fraction,
                float(parameters.get("minimum_retention_fraction", 0.25)),
                float(parameters.get("maximum_retention_fraction", 0.9)),
                1.0,
            ),
            _hinge(
                displaced_fraction,
                float(parameters.get("minimum_displaced_fraction", 0.1)),
                float(parameters.get("maximum_displaced_fraction", 0.9)),
                1.0,
            ),
        )
        return CalculatorResult(
            np.full(len(frames), error),
            {
                "retained": retained,
                "displacements": displaced,
                "retention_fraction": retention_fraction,
                "displaced_fraction": displaced_fraction,
            },
        )
