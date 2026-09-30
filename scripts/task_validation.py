from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldsolver.config import ROOT, TASKS_ROOT, load_yaml
from worldsolver.contracts.scene_geometry import load_scene_geometry
from worldsolver.rendering.render_spec import validate_render_spec

SCENE_ID = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")
CRITERION_ID = re.compile(r"^[a-z][a-z0-9_]{2,126}$")
DEFAULT_VISUAL_JUDGE_FRAME_COUNT = 20


class SourceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    anchor: str = Field(min_length=1)


class SceneRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_elements: list[str] = Field(min_length=1)
    initial_state: list[str] = Field(min_length=1)
    required_events: list[str] = Field(min_length=1)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source: SourceDefinition
    goal: str = Field(min_length=1)
    scene_requirements: SceneRequirements
    camera_requirements: list[str] = Field(min_length=1)
    parameter_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_id(self):
        if not SCENE_ID.fullmatch(self.id):
            raise ValueError("id must be lower_snake_case with 3-64 characters")
        return self


class MetricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    criterion: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_id(self):
        if not CRITERION_ID.fullmatch(self.id):
            raise ValueError(
                "criterion id must be lower_snake_case with 3-127 characters"
            )
        return self


class VisualMetricCriterion(MetricCriterion):
    dimension: str = Field(min_length=1)


class ReductionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern=r"^(maximum|rms|mean|final|minimum)$")


class NumericBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calculator: str = Field(min_length=1)
    inputs: dict[str, str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reduction: ReductionDefinition

    @model_validator(mode="after")
    def spatial_dimensions_are_supported(self):
        dimensions = self.parameters.get("spatial_dimensions")
        if dimensions is not None and int(dimensions) not in {2, 3}:
            raise ValueError("spatial_dimensions must be 2 or 3")
        return self


class ApplicabilityDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires: list[str] = Field(min_length=1)
    excludes: list[str] = Field(default_factory=list)


class PhysicsMetricCriterion(MetricCriterion):
    profile: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    applicability: ApplicabilityDefinition
    numeric_binding: NumericBinding


class MetricCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution: list[MetricCriterion] = Field(min_length=1, max_length=1)
    visual: list[VisualMetricCriterion] = Field(min_length=1, max_length=4)
    physics: list[PhysicsMetricCriterion] = Field(min_length=1, max_length=6)


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=4)
    task_id: str
    criteria: MetricCriteria

    @model_validator(mode="after")
    def unique_criterion_ids(self):
        groups = self.criteria.model_dump()
        identifiers = [item["id"] for entries in groups.values() for item in entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("criterion ids must be unique within a task")
        return self


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    position: list[float] = Field(min_length=3, max_length=3)
    look_at: list[float] = Field(min_length=3, max_length=3)
    up: list[float] = Field(min_length=3, max_length=3)
    width: int = Field(ge=64, le=8192)
    height: int = Field(ge=64, le=8192)
    fps: int = Field(ge=1, le=240)


class GeneratedScaffoldConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    scene_id: str
    seed: int
    duration: float = Field(gt=0, le=3600)
    sample_dt: float = Field(gt=0)
    visual_judge_frame_count: int = Field(default=DEFAULT_VISUAL_JUDGE_FRAME_COUNT, ge=1)
    camera: CameraConfig

    @model_validator(mode="after")
    def sampling_is_bounded(self):
        if self.duration / self.sample_dt > 1_000_000:
            raise ValueError("duration/sample_dt exceeds one million frames")
        return self


def _tasks_dir(root: Path) -> Path:
    if root == ROOT:
        return TASKS_ROOT
    return root / "task"


def task_ids(root: Path = ROOT) -> list[str]:
    tasks_dir = _tasks_dir(root)
    return sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir() and (path / "task.yaml").is_file()
    )


def validate_task(scene_id: str, root: Path = ROOT) -> dict[str, Any]:
    if not SCENE_ID.fullmatch(scene_id):
        raise ValueError(f"Invalid scene id: {scene_id}")
    return validate_task_directory(_tasks_dir(root) / scene_id, scene_id)


def validate_task_directory(
    task_dir: Path, scene_id: str | None = None
) -> dict[str, Any]:
    scene_id = scene_id or task_dir.name
    if not SCENE_ID.fullmatch(scene_id):
        raise ValueError(f"Invalid scene id: {scene_id}")
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Unknown scene: {scene_id}")

    task = TaskDefinition.model_validate(load_yaml(task_dir / "task.yaml"))
    metric = MetricDefinition.model_validate(load_yaml(task_dir / "metric.yaml"))
    if task.id != scene_id:
        raise ValueError("task.yaml id must match its directory name")
    if metric.task_id != scene_id:
        raise ValueError("metric.yaml task_id must match its directory name")
    return {"task": task.model_dump(), "metric": metric.model_dump()}


def validate_all_tasks(root: Path = ROOT) -> list[str]:
    identifiers = task_ids(root)
    if not identifiers:
        raise ValueError(f"No tasks found under {_tasks_dir(root)}")
    for scene_id in identifiers:
        validate_task(scene_id, root)
    return identifiers


def validate_registry(root: Path = ROOT) -> list[str]:
    """Backward-compatible name for callers; tasks are discovered by directory."""
    return validate_all_tasks(root)


def validate_scaffold_config(scene_id: str, value: dict[str, Any]) -> dict[str, Any]:
    config = GeneratedScaffoldConfig.model_validate(value)
    if config.scene_id != scene_id:
        raise ValueError("Generated config scene_id must match the task")
    return config.model_dump()


def validate_scaffold_render_spec(
    path: Path, geometry_path: Path | None = None
) -> dict[str, Any]:
    return validate_render_spec(path, load_scene_geometry(geometry_path))
