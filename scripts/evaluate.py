from __future__ import annotations

import argparse
import io
import json
import shutil
import statistics
import sys
import tempfile
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.config import load_yaml  # noqa: E402
from scripts.evaluation.aggregation import aggregate_evaluation  # noqa: E402
from scripts.evaluation.evaluator import Evaluator  # noqa: E402
from scripts.evaluation.scoring import attach_scores, compact_evaluation  # noqa: E402
from scripts.evaluation.visual import evaluate_visual_criteria  # noqa: E402
from scripts.rendering import render_trajectory  # noqa: E402
from scripts.runtime.sandbox import clean_environment, run_monitored  # noqa: E402


VISUAL_FRAME_COUNT = 20
VISUAL_ATTEMPTS = 5
SOLVER_TIMEOUT_SECONDS = 600


class SolverTimeoutError(RuntimeError):
    pass


def reject_scripted_animation(solver_path: Path) -> None:
    source = solver_path.read_text(encoding="utf-8")
    forbidden = ("keyframe", "key frame", "hermite", "precomputed trajectory")
    try:
        code_only = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
                code_only.append(token.string)
        haystack = " ".join(code_only).casefold()
    except (tokenize.TokenError, IndentationError, UnicodeDecodeError):
        haystack = source.casefold()
    found = [term for term in forbidden if term in haystack]
    if found:
        raise RuntimeError(
            f"Scripted animation is forbidden in solver submissions: {found}"
        )


def run_solver(workspace: Path, output: Path, usage_path: Path) -> None:
    environment = clean_environment(str(sys.executable))
    environment["PYTHONPATH"] = str(ROOT)
    usage = run_monitored(
        [
            sys.executable,
            "-c",
            (
                "from scripts.runtime.runner import scene_project_main; "
                "raise SystemExit(scene_project_main())"
            ),
            "--output-dir",
            str(output),
        ],
        cwd=workspace,
        timeout_seconds=SOLVER_TIMEOUT_SECONDS,
        output_path=usage_path,
        env=environment,
    )
    if usage.timed_out:
        raise SolverTimeoutError(
            f"Solver execution exceeded the {SOLVER_TIMEOUT_SECONDS}-second limit."
        )
    if usage.exit_code != 0:
        raise RuntimeError(f"Solver process exited with code {usage.exit_code}.")


def write_failed_evaluation(output: Path, metric_path: Path) -> dict:
    metric = load_yaml(metric_path)
    report = {
        "execution": 0,
        "visual_fidelity": {
            "criteria": [
                {"dimension": item["dimension"], "score": 0.0}
                for item in metric["criteria"]["visual"]
            ],
            "score": 0.0,
        },
        "physical_plausibility": {
            "criteria": [
                {"dimension": item["dimension"], "score": 0.0}
                for item in metric["criteria"]["physics"]
            ],
            "score": 0.0,
        },
        "total_score": 0.0,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def load_visual_judge() -> dict | None:
    path = Path(__file__).with_name("visual_judge.yaml")
    if not path.is_file():
        return None
    config = load_yaml(path)
    required = ("model", "base_url", "api_key")
    values = {
        key: str(config.get(key) or "").strip()
        for key in required
    }
    if not values["base_url"] and not values["api_key"]:
        return None
    missing = [key for key in required if not values[key]]
    if missing:
        raise ValueError(
            f"scripts/visual_judge.yaml is missing values for: {', '.join(missing)}"
        )
    return config


def evaluate_visual_five_times(
    output: Path,
    metric_path: Path,
    judge_config: dict,
    base_report: dict,
) -> dict:
    visual_criteria = load_yaml(metric_path)["criteria"]["visual"]
    attempts = [
        evaluate_visual_criteria(
            output / "video.mp4",
            visual_criteria,
            judge_config,
            frame_count=VISUAL_FRAME_COUNT,
        )
        for _ in range(VISUAL_ATTEMPTS)
    ]

    visual = []
    for criterion in visual_criteria:
        criterion_attempts = [
            next(item for item in attempt if item["criterion_id"] == criterion["id"])
            for attempt in attempts
        ]
        scores = [float(item["score"]) for item in criterion_attempts]
        visual.append(
            {
                "criterion_id": criterion["id"],
                "domain": "visual",
                "score": statistics.mean(scores),
                "details": {
                    "attempt_count": VISUAL_ATTEMPTS,
                    "attempt_std": statistics.pstdev(scores),
                    "frame_count": VISUAL_FRAME_COUNT,
                    "attempts": [
                        {
                            "score": item["score"],
                            "evidence": item.get("details", {}).get("evidence", ""),
                        }
                        for item in criterion_attempts
                    ],
                },
            }
        )

    report = aggregate_evaluation(
        base_report["criteria"]["execution"][0],
        base_report["criteria"]["physics"],
        visual,
    )
    report["task_id"] = base_report["task_id"]
    attach_scores(report)
    public_report = compact_evaluation(report, load_yaml(metric_path))
    (output / "evaluation.json").write_text(
        json.dumps(public_report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run one published WorldSolver submission and evaluate it."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="Run the solver and physical calculators without rendering or visual judging.",
    )
    args = parser.parse_args()

    scaffold_root = ROOT / "scaffolds" / args.task_id
    solver = ROOT / "result" / args.task_id / f"{args.model}.py"
    metric = ROOT / "task" / args.task_id / "metric.yaml"

    required = [
        scaffold_root / "scene.py",
        scaffold_root / "config.yaml",
        scaffold_root / "scene_geometry.yaml",
        scaffold_root / "render_spec.yaml",
        solver,
        metric,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("missing required files:\n" + "\n".join(f"  {path}" for path in missing))

    output = ROOT / "outputs" / args.task_id / args.model
    output.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "evaluation.json",
        "diagnostics.json",
        "trajectory.h5.tmp",
        "visual_attempts.json",
        "visual_judge_trace.jsonl",
        "visual_judge_raw.txt",
    ):
        (output / stale_name).unlink(missing_ok=True)
    if args.physical_only:
        for stale_name in ("video.mp4",):
            (output / stale_name).unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="worldsolver-") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(scaffold_root, workspace)
        (workspace / "solver").mkdir()
        shutil.copy2(solver, workspace / "solver" / "solver.py")

        try:
            reject_scripted_animation(workspace / "solver" / "solver.py")
            run_solver(workspace, output, Path(temporary) / "resource_usage.json")
            if not args.physical_only:
                render_trajectory(
                    output / "trajectory.h5",
                    workspace / "config.yaml",
                    output / "video.mp4",
                    render_spec_path=workspace / "render_spec.yaml",
                    geometry_path=workspace / "scene_geometry.yaml",
                )
            report = Evaluator().evaluate(
                output,
                metric,
                geometry_path=workspace / "scene_geometry.yaml",
                config_path=workspace / "config.yaml",
                require_video=not args.physical_only,
            )

            if not args.physical_only:
                judge_config = load_visual_judge()
                if judge_config is None:
                    print("Visual judge is not configured; visual evaluation remains pending.")
                else:
                    evaluate_visual_five_times(output, metric, judge_config, report)
                    print(
                        f"Visual evaluation complete: {VISUAL_FRAME_COUNT} frames x "
                        f"{VISUAL_ATTEMPTS} attempts."
                    )
        except Exception as exc:
            write_failed_evaluation(output, metric)
            for incomplete_name in (
                "trajectory.h5",
                "trajectory.h5.tmp",
                "video.mp4",
            ):
                (output / incomplete_name).unlink(missing_ok=True)
            print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            (output / "diagnostics.json").unlink(missing_ok=True)

    if (output / "trajectory.h5").is_file():
        print(f"Wrote {output / 'trajectory.h5'}")
    if (output / "video.mp4").is_file():
        print(f"Wrote {output / 'video.mp4'}")
    print(f"Wrote {output / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
