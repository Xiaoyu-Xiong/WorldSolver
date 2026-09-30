from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldsolver.config import load_yaml
from worldsolver.contracts import frame_value
from worldsolver.contracts.scene_geometry import load_scene_geometry
from worldsolver.evaluation.aggregation import aggregate_evaluation
from worldsolver.evaluation.calculators import CALCULATORS
from worldsolver.evaluation.calculators.base import EvaluationContext
from worldsolver.evaluation.reduction import reduce_values
from worldsolver.evaluation.scoring import attach_scores
from worldsolver.evaluation.visual import VISUAL_JUDGE_FRAME_COUNT, evaluate_visual_criteria
from worldsolver.trajectory import TrajectoryReader, validate_trajectory


class Evaluator:
    def evaluate(
        self,
        result_dir: Path,
        metric_path: Path,
        *,
        geometry_path: Path | None = None,
        config_path: Path | None = None,
        visual_model_config: dict[str, Any] | None = None,
        require_video: bool = True,
    ) -> dict[str, Any]:
        metric = load_yaml(metric_path)
        trajectory_path = result_dir / "trajectory.h5"
        validation = validate_trajectory(trajectory_path)
        frames = list(TrajectoryReader(trajectory_path))
        context = EvaluationContext(
            result_dir=result_dir,
            scene_geometry=load_scene_geometry(geometry_path or result_dir / "scene_geometry.yaml"),
            config_path=config_path,
        )
        diagnostics_path = result_dir / "diagnostics.json"
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.is_file()
            else {}
        )
        video_path = result_dir / "video.mp4"
        execution_criterion = metric["criteria"]["execution"][0]
        video_readable = video_path.is_file() and video_path.stat().st_size > 0
        execution_pass = diagnostics.get("success") is True and (
            video_readable or not require_video
        )
        execution = {
            "criterion_id": execution_criterion["id"],
            "domain": "execution",
            "score": 1.0 if execution_pass else 0.0,
            "passed": execution_pass,
            "details": {
                "diagnostics_success": diagnostics.get("success") is True,
                "trajectory_frames": validation.frame_count,
                "trajectory_bytes": validation.bytes,
                "video_readable": video_readable if require_video else None,
            },
        }
        physics = [
            self._evaluate_physics_criterion(criterion, frames, context)
            for criterion in metric["criteria"]["physics"]
        ]
        if visual_model_config is None:
            visual = [
                {
                    "criterion_id": criterion["id"],
                    "domain": "visual",
                    "score": None,
                    "passed": None,
                    "status": "pending_visual_evaluator",
                }
                for criterion in metric["criteria"]["visual"]
            ]
        else:
            try:
                visual = evaluate_visual_criteria(
                    video_path,
                    metric["criteria"]["visual"],
                    visual_model_config,
                    result_dir,
                    frame_count=_visual_judge_frame_count(result_dir),
                )
            except (KeyError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                visual = [
                    {
                        "criterion_id": criterion["id"],
                        "domain": "visual",
                        "score": None,
                        "passed": None,
                        "status": "failed_visual_evaluator",
                        "details": {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                    for criterion in metric["criteria"]["visual"]
                ]
        report = aggregate_evaluation(
            execution,
            physics,
            visual,
        )
        report["task_id"] = metric["task_id"]
        attach_scores(report)
        (result_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    @staticmethod
    def _evaluate_physics_criterion(
        criterion: dict[str, Any],
        frames,
        context: EvaluationContext,
    ) -> dict[str, Any]:
        binding = criterion["numeric_binding"]
        calculator_id = binding["calculator"]
        try:
            missing_fields = sorted(
                {
                    path
                    for path in binding["inputs"].values()
                    if any(_frame_is_missing(frame, path) for frame in frames)
                }
            )
            if missing_fields:
                return _measurement_error(
                    criterion,
                    calculator_id,
                    binding,
                    f"Missing calculator input fields: {missing_fields}",
                    missing_fields=missing_fields,
                )
            calculator = CALCULATORS[calculator_id]
            if hasattr(calculator, "compute_with_context"):
                calculated = calculator.compute_with_context(
                    frames,
                    binding["inputs"],
                    binding.get("parameters", {}),
                    context,
                )
            else:
                calculated = calculator.compute(
                    frames,
                    binding["inputs"],
                    binding.get("parameters", {}),
                )
            values = calculated.normalized_errors
            if calculated.frame_indices is None:
                if len(values) != len(frames):
                    raise ValueError(
                        "Calculator must return one normalized error per trajectory frame"
                    )
                frame_indices = np.arange(len(frames))
            else:
                frame_indices = calculated.frame_indices
                if np.any(frame_indices >= len(frames)):
                    raise ValueError("Calculator frame_indices are outside the trajectory")
            value = reduce_values(values, binding["reduction"])
            return {
                "criterion_id": criterion["id"],
                "criterion": criterion["criterion"],
                "domain": "physics",
                "profile": criterion["profile"],
                "dimension": criterion["dimension"],
                "calculator": calculator_id,
                "metric": calculator.metric_name,
                "unit": calculator.metric_unit,
                "reduction": binding["reduction"],
                "normalized_error": value,
                "details": {
                    **calculated.details,
                    "sample_count": int(values.size),
                    "worst_frame_index": int(frame_indices[np.argmax(values)]),
                    **(
                        {"evaluated_frame_indices": frame_indices.tolist()}
                        if calculated.frame_indices is not None
                        else {}
                    ),
                },
            }
        except (KeyError, ValueError, FloatingPointError) as exc:
            return _measurement_error(
                criterion, calculator_id, binding, str(exc)
            )


def _frame_is_missing(frame, path: str) -> bool:
    try:
        frame_value(frame, path)
    except KeyError:
        return True
    return False


def _measurement_error(
    criterion: dict[str, Any],
    calculator_id: str,
    binding: dict[str, Any],
    error: str,
    *,
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    calculator = CALCULATORS.get(calculator_id)
    details: dict[str, Any] = {"error": error}
    if missing_fields:
        details["missing_fields"] = missing_fields
    return {
        "criterion_id": criterion["id"],
        "criterion": criterion["criterion"],
        "domain": "physics",
        "profile": criterion["profile"],
        "dimension": criterion["dimension"],
        "calculator": calculator_id,
        "metric": getattr(calculator, "metric_name", None),
        "unit": getattr(calculator, "metric_unit", None),
        "reduction": binding["reduction"],
        "normalized_error": None,
        "status": "measurement_error",
        "details": details,
    }


def _visual_judge_frame_count(result_dir: Path) -> int:
    config_path = result_dir / "config.yaml"
    if not config_path.is_file():
        return VISUAL_JUDGE_FRAME_COUNT
    config = load_yaml(config_path)
    value = config.get("visual_judge_frame_count", VISUAL_JUDGE_FRAME_COUNT)
    return max(1, int(value))


def scene_evaluate_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--scene-geometry", type=Path)
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args()
    Evaluator().evaluate(
        arguments.result_dir,
        arguments.metric,
        geometry_path=arguments.scene_geometry,
        config_path=arguments.config,
    )
    return 0
