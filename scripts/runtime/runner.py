from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from worldsolver.config import load_yaml
from worldsolver.contracts import Scene, Solver, validate_frame_bindings
from worldsolver.contracts.scene_geometry import load_scene_geometry
from worldsolver.trajectory import TrajectoryWriter, validate_trajectory


def _load_symbol(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"worldsolver_submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: stdlib machinery (e.g. @dataclass) resolves the
    # defining module by name through sys.modules and crashes if it is missing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _diagnostic_json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Diagnostic value is not JSON serializable: {type(value).__name__}")


def execute_scene_project(workspace: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, object] = {"success": False}
    try:
        config = load_yaml(workspace / "config.yaml")
        geometry_path = workspace / "scene_geometry.yaml"
        if geometry_path.is_file():
            config["scene_geometry"] = load_yaml(geometry_path)
        scene = _load_symbol(workspace / "scene.py", "BenchmarkScene")(config)
        solver = _load_symbol(workspace / "solver" / "solver.py", "BenchmarkSolver")()
        if not isinstance(scene, Scene) or not isinstance(solver, Solver):
            raise TypeError("Submission does not implement the Scene/Solver Contract")
        solver.initialize(scene, int(config.get("seed", 0)))
        geometry = load_scene_geometry(geometry_path)
        sample_times = scene.sample_times()
        trajectory_path = output_dir / "trajectory.h5"
        with TrajectoryWriter(trajectory_path, scene.metadata()) as writer:
            for time in sample_times:
                frame = solver.advance_to(float(time))
                if geometry is not None:
                    validate_frame_bindings(frame, geometry)
                writer.append(frame)
        validate_trajectory(
            trajectory_path,
            expected_times=sample_times,
        )
        diagnostics.update(solver.diagnostics())
        diagnostics["success"] = True
    except Exception as exc:
        diagnostics.update({"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        (output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, default=_diagnostic_json_default),
            encoding="utf-8",
        )


def scene_project_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    execute_scene_project(Path.cwd(), arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(scene_project_main())
