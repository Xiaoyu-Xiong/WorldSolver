from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame
from worldsolver.contracts.scene_geometry import SceneObject

from .base import CalculatorResult, EvaluationContext, input_series


_EPS = np.finfo(float).eps


def _quaternion_matrix(value: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=float)
    norm = max(float(np.linalg.norm([x, y, z, w])), _EPS)
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _find_subject(context: EvaluationContext) -> SceneObject:
    if context.scene_geometry is None:
        raise ValueError("calculator requires scene_geometry.yaml")
    subjects = [
        item
        for item in context.scene_geometry.objects
        if item.semantic == "deformable" and item.motion == "dynamic"
    ]
    if len(subjects) != 1:
        raise ValueError("calculator requires exactly one dynamic deformable subject")
    return subjects[0]


def _find_object(context: EvaluationContext, object_id: str, keyword: str) -> SceneObject:
    if context.scene_geometry is None:
        raise ValueError("calculator requires scene_geometry.yaml")
    exact = [item for item in context.scene_geometry.objects if item.id == object_id]
    if exact:
        return exact[0]
    matches = [item for item in context.scene_geometry.objects if keyword in item.id]
    if len(matches) != 1:
        raise ValueError(f"scene object {object_id!r} was not found uniquely")
    return matches[0]


def _subject_positions(frames: list[StateFrame], inputs: dict[str, str]) -> np.ndarray:
    positions = np.stack(input_series(frames, inputs, "position")).astype(float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("position must have shape [T,N,3]")
    return positions


def _topology_edges(subject: SceneObject) -> np.ndarray:
    asset = subject.geometry.topology_asset or subject.geometry.asset
    if not asset:
        raise ValueError("deformable calculator requires a topology asset")
    path = Path(asset)
    if path.suffix.casefold() == ".npz":
        with np.load(path) as data:
            if "edges" in data:
                edges = np.asarray(data["edges"], dtype=int)
            elif "faces" in data:
                faces = np.asarray(data["faces"], dtype=int)
                edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
            else:
                raise ValueError("topology asset has neither faces nor edges")
    elif path.suffix.casefold() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "edges" in data:
            edges = np.asarray(data["edges"], dtype=int)
        elif "faces" in data:
            faces = np.asarray(data["faces"], dtype=int)
            edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        else:
            raise ValueError("topology asset has neither faces nor edges")
    else:
        raise ValueError("deformable calculator currently requires npz or json topology")
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) > 6000:
        edges = edges[np.linspace(0, len(edges) - 1, 6000, dtype=int)]
    return edges


def _topology_components(subject: SceneObject, vertex_count: int) -> list[np.ndarray]:
    edges = _topology_edges(subject)
    if np.any(edges < 0) or np.any(edges >= vertex_count):
        raise ValueError("topology indices do not match trajectory material points")
    parent = np.arange(vertex_count, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for raw_left, raw_right in edges:
        left = find(int(raw_left))
        right = find(int(raw_right))
        if left != right:
            parent[right] = left
    groups: dict[int, list[int]] = {}
    for index in range(vertex_count):
        groups.setdefault(find(index), []).append(index)
    components = [np.asarray(values, dtype=int) for values in groups.values()]
    components.sort(key=lambda values: int(values[0]))
    if len(components) < 2:
        raise ValueError("topological merge calculator requires multiple initial components")
    return components


def _component_pair_patch_counts(
    positions: np.ndarray,
    components: list[np.ndarray],
    contact_band: float,
) -> dict[tuple[int, int], np.ndarray]:
    if contact_band <= 0.0:
        raise ValueError("contact_band must be positive")
    pairs = {
        (left, right): np.zeros(len(positions), dtype=int)
        for left in range(len(components))
        for right in range(left + 1, len(components))
    }
    for frame_index, points in enumerate(positions):
        trees = [cKDTree(points[indices]) for indices in components]
        for (left, right), counts in pairs.items():
            neighborhoods = trees[left].query_ball_tree(trees[right], contact_band)
            close_left = sum(bool(neighbors) for neighbors in neighborhoods)
            close_right = len({neighbor for neighbors in neighborhoods for neighbor in neighbors})
            counts[frame_index] = min(close_left, close_right)
    return pairs


def _persistent_activation(mask: np.ndarray, frames: int) -> int | None:
    if frames <= 0:
        raise ValueError("persistence_frames must be positive")
    run = 0
    for index, active in enumerate(np.asarray(mask, dtype=bool)):
        run = run + 1 if active else 0
        if run >= frames:
            return index
    return None


def _edge_strain_signal(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    initial = positions[0]
    rest_lengths = np.linalg.norm(initial[edges[:, 1]] - initial[edges[:, 0]], axis=1)
    valid = rest_lengths > 1e-9
    if not np.any(valid):
        raise ValueError("topology contains no non-degenerate edges")
    edges = edges[valid]
    rest_lengths = rest_lengths[valid]
    values = []
    for points in positions:
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        values.append(float(np.sqrt(np.mean(np.square((lengths - rest_lengths) / rest_lengths)))))
    return np.asarray(values)


def _window_indices(count: int, interval: list[float] | tuple[float, float]) -> np.ndarray:
    start = int(np.floor(float(interval[0]) * max(count - 1, 1)))
    stop = int(np.ceil(float(interval[1]) * max(count - 1, 1))) + 1
    return np.arange(max(0, start), min(count, max(start + 1, stop)))


def _box_signed_distance(points: np.ndarray, item: SceneObject) -> np.ndarray:
    if item.geometry.type != "box":
        raise ValueError("box distance requires box geometry")
    center = np.asarray(item.transform.translation, dtype=float)
    rotation = _quaternion_matrix(item.transform.rotation_xyzw)
    scale = np.asarray(item.transform.scale, dtype=float)
    local = (np.asarray(points) - center) @ rotation
    half = np.asarray(item.geometry.half_extents, dtype=float) * scale
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _cylinder_signed_distance(points: np.ndarray, item: SceneObject) -> np.ndarray:
    if item.geometry.type != "cylinder":
        raise ValueError("cylinder distance requires cylinder geometry")
    center = np.asarray(item.transform.translation, dtype=float)
    rotation = _quaternion_matrix(item.transform.rotation_xyzw)
    scale = np.asarray(item.transform.scale, dtype=float)
    local = (np.asarray(points) - center) @ rotation
    radial = np.linalg.norm(local[:, :2], axis=1) - float(item.geometry.radius) * float(
        np.mean(scale[:2])
    )
    axial = np.abs(local[:, 2]) - 0.5 * float(item.geometry.height) * float(scale[2])
    q = np.column_stack((radial, axial))
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _rigid_aligned_residual(initial: np.ndarray, current: np.ndarray) -> np.ndarray:
    reference_center = initial.mean(axis=0)
    current_center = current.mean(axis=0)
    reference = initial - reference_center
    moving = current - current_center
    u, _, vt = np.linalg.svd(moving.T @ reference)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = moving @ rotation + reference_center
    return aligned - initial


class StairContactEventCountCalculator:
    id = "impact.contact_event_count.v1"
    metric_name = "stair_contact_coverage_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        if context.scene_geometry is None:
            raise ValueError("calculator requires scene geometry")
        stairs = [
            item
            for item in context.scene_geometry.objects
            if item.geometry.type == "box" and "stair" in item.id
        ]
        if not stairs:
            raise ValueError("no staircase boxes found")
        band = float(parameters.get("contact_band", 0.06))
        minimum = int(parameters.get("minimum_distinct_contacts", min(3, len(stairs))))
        touched = []
        minimum_gaps = {}
        for stair in stairs:
            gap = min(float(np.min(np.abs(_box_signed_distance(points, stair)))) for points in positions)
            minimum_gaps[stair.id] = gap
            if gap <= band:
                touched.append(stair.id)
        error = max(0.0, minimum - len(touched)) / max(minimum, 1)
        return CalculatorResult(
            np.full(len(frames), error),
            {
                "distinct_stair_contacts": len(touched),
                "contacted_stairs": touched,
                "minimum_required_contacts": minimum,
                "minimum_surface_gaps": minimum_gaps,
            },
        )


class ResidualDeformationGrowthCalculator:
    id = "deformation.residual_growth.v1"
    metric_name = "residual_deformation_growth_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        signal = _edge_strain_signal(positions, _topology_edges(_find_subject(context)))
        early_ids = _window_indices(len(frames), parameters.get("early_interval", [0.25, 0.55]))
        late_ids = _window_indices(len(frames), parameters.get("late_interval", [0.75, 1.0]))
        early = float(np.percentile(signal[early_ids], 35))
        late = float(np.percentile(signal[late_ids], 35))
        minimum = float(parameters.get("minimum_final_residual", 0.025))
        increment = float(parameters.get("minimum_growth_increment", 0.01))
        ratio = float(parameters.get("minimum_growth_ratio", 1.1))
        target = max(minimum, early + increment, early * ratio)
        activation_error = max(0.0, minimum - late) / max(minimum, _EPS)
        growth_error = max(0.0, target - late) / max(target, _EPS)
        error = activation_error + growth_error
        return CalculatorResult(
            np.full(len(frames), error),
            {
                "early_residual_edge_strain": early,
                "late_residual_edge_strain": late,
                "target_late_residual_edge_strain": target,
                "peak_edge_strain": float(np.max(signal)),
            },
        )


class DroopProgressionCalculator:
    id = "deformation.droop_progression.v1"
    metric_name = "aligned_droop_progression_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        del context
        positions = _subject_positions(frames, inputs)
        initial = positions[0]
        height = max(float(np.ptp(initial[:, 2])), 1e-6)
        droop_quantile = float(parameters.get("droop_quantile", 0.1))
        if not 0.0 < droop_quantile < 0.5:
            raise ValueError("droop_quantile must lie between zero and one half")
        droop = []
        for points in positions:
            residual = _rigid_aligned_residual(initial, points)
            downward_tail = float(np.quantile(residual[:, 2], droop_quantile))
            droop.append(max(0.0, -downward_tail) / height)
        droop = np.asarray(droop)
        early_ids = _window_indices(len(frames), parameters.get("early_interval", [0.25, 0.55]))
        late_ids = _window_indices(len(frames), parameters.get("late_interval", [0.75, 1.0]))
        early = float(np.percentile(droop[early_ids], 50))
        late = float(np.percentile(droop[late_ids], 50))
        minimum = float(parameters.get("minimum_final_droop", 0.025))
        increment = float(parameters.get("minimum_droop_increment", 0.01))
        activation_error = max(0.0, minimum - late) / max(minimum, _EPS)
        growth_error = max(0.0, early + increment - late) / max(early + increment, minimum, _EPS)
        return CalculatorResult(
            np.full(len(frames), activation_error + growth_error),
            {
                "early_aligned_droop_fraction": early,
                "late_aligned_droop_fraction": late,
                "peak_aligned_droop_fraction": float(np.max(droop)),
                "droop_quantile": droop_quantile,
            },
        )


class BarContactActivationCalculator:
    id = "impact.bar_contact_activation.v1"
    metric_name = "bar_contact_activation_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        velocities = np.stack(input_series(frames, inputs, "velocity")).astype(float)
        bar = _find_object(context, str(parameters.get("bar_object_id", "metal_bar")), "bar")
        gaps = np.asarray([float(np.min(np.abs(_cylinder_signed_distance(points, bar)))) for points in positions])
        contact_index = int(np.argmin(gaps))
        band = float(parameters.get("contact_band", 0.05))
        speed_index = max(0, contact_index - 1)
        impact_speed = float(np.sqrt(np.mean(np.sum(np.square(velocities[speed_index]), axis=1))))
        minimum_speed = float(parameters.get("minimum_impact_speed", 0.15))
        contact_error = max(0.0, float(gaps[contact_index]) - band) / max(band, _EPS)
        speed_error = max(0.0, minimum_speed - impact_speed) / max(minimum_speed, _EPS)
        return CalculatorResult(
            np.full(len(frames), contact_error + speed_error),
            {
                "contact_frame_index": contact_index,
                "minimum_bar_surface_gap": float(gaps[contact_index]),
                "precontact_rms_speed": impact_speed,
            },
        )


class PeakResidualRecoveryCalculator:
    id = "deformation.peak_to_residual_recovery.v1"
    metric_name = "peak_to_residual_recovery_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        signal = _edge_strain_signal(positions, _topology_edges(_find_subject(context)))
        start = int(float(parameters.get("evaluation_start_fraction", 0.1)) * len(signal))
        peak = float(np.max(signal[start:]))
        final_ids = _window_indices(len(frames), parameters.get("final_interval", [0.8, 1.0]))
        residual = float(np.percentile(signal[final_ids], 35))
        recovery = max(0.0, 1.0 - residual / max(peak, _EPS))
        minimum_peak = float(parameters.get("minimum_peak_strain", 0.04))
        minimum_residual = float(parameters.get("minimum_residual_strain", 0.008))
        minimum_recovery = float(parameters.get("minimum_recovery_fraction", 0.18))
        peak_error = max(0.0, minimum_peak - peak) / max(minimum_peak, _EPS)
        residual_error = max(0.0, minimum_residual - residual) / max(minimum_residual, _EPS)
        recovery_error = max(0.0, minimum_recovery - recovery) / max(minimum_recovery, _EPS)
        return CalculatorResult(
            np.full(len(frames), peak_error + residual_error + recovery_error),
            {
                "peak_edge_strain": peak,
                "final_residual_edge_strain": residual,
                "recovery_fraction": recovery,
            },
        )


class LocalizedJiggleCalculator:
    id = "oscillation.localized_jiggle.v1"
    metric_name = "localized_jiggle_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        bar = _find_object(context, str(parameters.get("bar_object_id", "metal_bar")), "bar")
        per_frame_gap = np.stack([np.abs(_cylinder_signed_distance(points, bar)) for points in positions])
        contact_index, _ = np.unravel_index(int(np.argmin(per_frame_gap)), per_frame_gap.shape)
        count = positions.shape[1]
        patch_count = max(8, int(float(parameters.get("patch_fraction", 0.12)) * count))
        patch = np.argsort(per_frame_gap[contact_index])[: min(patch_count, count)]
        signal = positions[:, patch, 2].mean(axis=1) - positions[:, :, 2].mean(axis=1)
        post = signal[contact_index:]
        if len(post) < 12:
            raise ValueError("localized jiggle requires at least twelve post-contact frames")
        index = np.arange(len(post), dtype=float)
        trend = np.polyval(np.polyfit(index, post, 1), index)
        oscillation = post - trend
        characteristic = max(float(np.ptp(positions[0, :, 2])), 1e-6)
        amplitude = float(np.max(np.abs(oscillation))) / characteristic
        deadband = max(float(np.std(oscillation)) * 0.08, 1e-10)
        signs = np.sign(np.where(np.abs(oscillation) >= deadband, oscillation, 0.0))
        signs = signs[signs != 0.0]
        crossings = int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0)) if len(signs) > 1 else 0
        split = max(4, len(oscillation) // 2)
        early_rms = float(np.sqrt(np.mean(np.square(oscillation[:split]))))
        late_rms = float(np.sqrt(np.mean(np.square(oscillation[split:]))))
        minimum_amplitude = float(parameters.get("minimum_relative_amplitude", 0.005))
        minimum_crossings = int(parameters.get("minimum_zero_crossings", 2))
        maximum_late_ratio = float(parameters.get("maximum_late_rms_ratio", 1.25))
        amplitude_error = max(0.0, minimum_amplitude - amplitude) / max(minimum_amplitude, _EPS)
        crossing_error = max(0.0, minimum_crossings - crossings) / max(minimum_crossings, 1)
        decay_ratio = late_rms / max(early_rms, _EPS)
        decay_error = max(0.0, decay_ratio - maximum_late_ratio) / max(maximum_late_ratio, _EPS)
        return CalculatorResult(
            np.full(len(frames), amplitude_error + crossing_error + decay_error),
            {
                "contact_frame_index": int(contact_index),
                "patch_vertex_count": int(len(patch)),
                "relative_oscillation_amplitude": amplitude,
                "zero_crossings": crossings,
                "early_rms": early_rms,
                "late_rms": late_rms,
                "late_to_early_rms_ratio": decay_ratio,
            },
        )


class PersistentMergeProgressCalculator:
    id = "topology.persistent_merge_progress.v1"
    metric_name = "persistent_merge_progress_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        components = _topology_components(_find_subject(context), positions.shape[1])
        persistence = int(parameters.get("persistence_frames", 3))
        minimum_merges = int(parameters.get("minimum_merge_events", max(1, len(components) - 2)))
        maximum_groups = int(parameters.get("maximum_final_groups", 2))
        if not 1 <= maximum_groups < len(components):
            raise ValueError("maximum_final_groups must be between one and the initial group count")
        if not 1 <= minimum_merges < len(components):
            raise ValueError("minimum_merge_events must be smaller than the initial group count")
        patch_counts = _component_pair_patch_counts(
            positions, components, float(parameters.get("contact_band", 0.05))
        )
        activations = {
            pair: _persistent_activation(counts > 0, persistence)
            for pair, counts in patch_counts.items()
        }
        parent = np.arange(len(components), dtype=int)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        merge_count = np.zeros(len(frames), dtype=int)
        group_count = np.full(len(frames), len(components), dtype=int)
        merged = 0
        groups = len(components)
        events_by_frame: dict[int, list[tuple[int, int]]] = {}
        for pair, activation in activations.items():
            if activation is not None:
                events_by_frame.setdefault(activation, []).append(pair)
        for frame_index in range(len(frames)):
            for left, right in sorted(events_by_frame.get(frame_index, [])):
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parent[right_root] = left_root
                    merged += 1
                    groups -= 1
            merge_count[frame_index] = merged
            group_count[frame_index] = groups
        merge_error = np.maximum(0.0, minimum_merges - merge_count) / max(minimum_merges, 1)
        group_denominator = max(len(components) - maximum_groups, 1)
        group_error = np.maximum(0.0, group_count - maximum_groups) / group_denominator
        event_details = [
            {"components": [left, right], "activation_frame": int(activation)}
            for (left, right), activation in sorted(activations.items())
            if activation is not None
        ]
        return CalculatorResult(
            merge_error + group_error,
            {
                "initial_component_count": len(components),
                "persistent_union_events": int(merge_count[-1]),
                "final_group_count": int(group_count[-1]),
                "minimum_required_merges": minimum_merges,
                "maximum_allowed_final_groups": maximum_groups,
                "activated_component_pairs": event_details,
            },
        )


class MergeBridgeContinuityCalculator:
    id = "topology.merge_bridge_continuity.v1"
    metric_name = "merge_bridge_continuity_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        components = _topology_components(_find_subject(context), positions.shape[1])
        minimum_patch = int(parameters.get("minimum_patch_vertices", 4))
        persistence = int(parameters.get("persistence_frames", 3))
        minimum_bridges = int(parameters.get("minimum_supported_bridges", 3))
        minimum_late = float(parameters.get("minimum_late_persistence", 0.5))
        if minimum_patch <= 0 or minimum_bridges <= 0:
            raise ValueError("minimum bridge and patch counts must be positive")
        if not 0.0 <= minimum_late <= 1.0:
            raise ValueError("minimum_late_persistence must lie between zero and one")
        patch_counts = _component_pair_patch_counts(
            positions, components, float(parameters.get("contact_band", 0.05))
        )
        late = _window_indices(len(frames), parameters.get("late_interval", [0.65, 1.0]))
        supported: list[tuple[tuple[int, int], int, float, int]] = []
        for pair, counts in patch_counts.items():
            mask = counts >= minimum_patch
            activation = _persistent_activation(mask, persistence)
            if activation is not None:
                supported.append(
                    (pair, activation, float(np.mean(mask[late])), int(np.max(counts)))
                )
        late_values = sorted((item[2] for item in supported), reverse=True)
        selected = late_values[:minimum_bridges]
        selected.extend([0.0] * (minimum_bridges - len(selected)))
        late_support = float(np.mean(selected))
        count_error = max(0.0, minimum_bridges - len(supported)) / minimum_bridges
        persistence_error = max(0.0, minimum_late - late_support) / max(minimum_late, _EPS)
        return CalculatorResult(
            np.full(len(frames), count_error + persistence_error),
            {
                "initial_component_count": len(components),
                "supported_bridge_count": len(supported),
                "minimum_required_bridges": minimum_bridges,
                "selected_late_persistence": late_support,
                "minimum_late_persistence": minimum_late,
                "bridges": [
                    {
                        "components": list(pair),
                        "activation_frame": int(activation),
                        "late_persistence": late_fraction,
                        "maximum_patch_vertices_per_side": maximum_patch,
                    }
                    for pair, activation, late_fraction, maximum_patch in supported
                ],
            },
        )


class TopologyTemporalCoherenceCalculator:
    id = "topology.temporal_coherence.v1"
    metric_name = "topology_temporal_coherence_error"
    metric_unit = "dimensionless"

    def compute_with_context(
        self,
        frames: list[StateFrame],
        inputs: dict[str, str],
        parameters: dict[str, Any],
        context: EvaluationContext,
    ) -> CalculatorResult:
        positions = _subject_positions(frames, inputs)
        edges = _topology_edges(_find_subject(context))
        initial = positions[0]
        scale = max(float(np.linalg.norm(np.ptp(initial, axis=0))), 1e-8)
        rest_lengths = np.linalg.norm(initial[edges[:, 1]] - initial[edges[:, 0]], axis=1)
        valid = rest_lengths > 1e-9
        if not np.any(valid):
            raise ValueError("topology contains no non-degenerate edges")
        edges = edges[valid]
        rest_lengths = rest_lengths[valid]
        maximum_step = float(parameters.get("maximum_step_fraction", 0.25))
        maximum_edge_step = float(parameters.get("maximum_edge_step_log_change", 0.5))
        maximum_broken = float(parameters.get("maximum_broken_edge_fraction", 0.05))
        maximum_stretch = float(parameters.get("maximum_edge_stretch", 4.0))
        maximum_extent = float(parameters.get("maximum_extent_ratio", 3.0))
        if min(maximum_step, maximum_edge_step, maximum_broken, maximum_stretch, maximum_extent) <= 0:
            raise ValueError("temporal coherence thresholds must be positive")
        errors = np.zeros(len(frames), dtype=float)
        step_fractions = np.zeros(len(frames), dtype=float)
        edge_step_changes = np.zeros(len(frames), dtype=float)
        broken_fractions = np.zeros(len(frames), dtype=float)
        extent_ratios = np.zeros(len(frames), dtype=float)
        previous_lengths = rest_lengths
        for frame_index, points in enumerate(positions):
            lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
            stretch = lengths / np.maximum(rest_lengths, 1e-9)
            broken_fractions[frame_index] = float(np.mean(stretch > maximum_stretch))
            extent_ratios[frame_index] = float(np.linalg.norm(np.ptp(points, axis=0))) / scale
            if frame_index:
                displacement = np.linalg.norm(points - positions[frame_index - 1], axis=1)
                step_fractions[frame_index] = float(np.percentile(displacement, 99)) / scale
                log_change = np.abs(
                    np.log(np.maximum(lengths, 1e-9) / np.maximum(previous_lengths, 1e-9))
                )
                edge_step_changes[frame_index] = float(np.percentile(log_change, 99))
            previous_lengths = lengths
            errors[frame_index] = (
                max(0.0, step_fractions[frame_index] - maximum_step) / maximum_step
                + max(0.0, edge_step_changes[frame_index] - maximum_edge_step)
                / maximum_edge_step
                + max(0.0, broken_fractions[frame_index] - maximum_broken) / maximum_broken
                + max(0.0, extent_ratios[frame_index] - maximum_extent) / maximum_extent
            )
        return CalculatorResult(
            errors,
            {
                "characteristic_length": scale,
                "maximum_step_fraction_observed": float(np.max(step_fractions)),
                "maximum_edge_step_log_change_observed": float(np.max(edge_step_changes)),
                "maximum_broken_edge_fraction_observed": float(np.max(broken_fractions)),
                "maximum_extent_ratio_observed": float(np.max(extent_ratios)),
            },
        )
