from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldsolver.config import load_yaml

GeometryType = Literal[
    "plane",
    "box",
    "sphere",
    "cylinder",
    "capsule",
    "axisymmetric_container",
    "triangle_mesh",
    "heightfield",
    "curve_network",
    "particle_domain",
    "voxel_grid",
]


class TransformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3
    )
    rotation_xyzw: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3)

    @model_validator(mode="after")
    def finite_non_degenerate(self):
        values = self.translation + self.rotation_xyzw + self.scale
        if not np.isfinite(values).all() or any(value <= 0.0 for value in self.scale):
            raise ValueError("transform values must be finite and scale must be positive")
        if np.linalg.norm(self.rotation_xyzw) < 1e-9:
            raise ValueError("rotation quaternion must be nonzero")
        return self


class GeometrySpec(BaseModel):
    """Authoritative physical shape. Renderers may tessellate it but never replace it."""

    model_config = ConfigDict(extra="forbid")

    type: GeometryType
    half_extents: list[float] | None = None
    radius: float | None = Field(default=None, gt=0.0)
    height: float | None = Field(default=None, gt=0.0)
    normal: list[float] | None = None
    offset: float | None = None
    inner_profile: list[list[float]] | None = None
    wall_thickness: float | None = Field(default=None, gt=0.0)
    asset: str | None = None
    topology_asset: str | None = None
    grid_shape: list[int] | None = None
    bounds_min: list[float] | None = None
    bounds_max: list[float] | None = None

    @model_validator(mode="after")
    def required_parameters(self):
        if self.type == "box" and (self.half_extents is None or len(self.half_extents) != 3):
            raise ValueError("box requires half_extents[3]")
        if self.type in {"sphere", "cylinder", "capsule"} and self.radius is None:
            raise ValueError(f"{self.type} requires radius")
        if self.type in {"cylinder", "capsule"} and self.height is None:
            raise ValueError(f"{self.type} requires height")
        if self.type == "plane" and (self.normal is None or len(self.normal) != 3):
            raise ValueError("plane requires normal[3]")
        if self.type == "axisymmetric_container":
            if self.inner_profile is None or len(self.inner_profile) < 3:
                raise ValueError(
                    "axisymmetric_container requires at least three [radius,z] samples"
                )
            profile = np.asarray(self.inner_profile, dtype=float)
            if profile.shape[1:] != (2,) or np.any(profile[:, 0] <= 0.0):
                raise ValueError("inner_profile must contain positive [radius,z] samples")
            if np.any(np.diff(profile[:, 1]) <= 0.0):
                raise ValueError("inner_profile heights must be strictly increasing")
            if self.wall_thickness is None:
                raise ValueError("axisymmetric_container requires wall_thickness")
        if self.type in {"triangle_mesh", "heightfield", "curve_network"} and not (
            self.asset or self.topology_asset
        ):
            raise ValueError(f"{self.type} requires an asset or topology_asset")
        if self.type in {"particle_domain", "voxel_grid"}:
            if self.bounds_min is None or self.bounds_max is None:
                raise ValueError(f"{self.type} requires bounds_min and bounds_max")
        return self


class StateBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selector: Literal["all", "body_id", "attribute", "field"] = "all"
    attribute_name: str | None = None
    value: int | float | None = None

    @model_validator(mode="after")
    def selector_has_value(self):
        if self.selector in {"body_id", "attribute"} and self.value is None:
            raise ValueError("body_id and attribute selectors require value")
        if self.selector in {"attribute", "field"} and not self.attribute_name:
            raise ValueError("attribute and field selectors require attribute_name")
        if self.selector not in {"attribute", "field"} and self.attribute_name is not None:
            raise ValueError("attribute_name is only valid for attribute and field selectors")
        return self


class SceneObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    semantic: Literal[
        "boundary",
        "rigid",
        "deformable",
        "cloth",
        "fiber",
        "granular",
        "liquid",
        "gas",
        "emitter",
        "sensor",
    ]
    motion: Literal["static", "kinematic", "dynamic", "field"]
    geometry: GeometrySpec
    transform: TransformSpec = Field(default_factory=TransformSpec)
    state_binding: StateBinding | None = None
    material_id: str = "default"
    collision_enabled: bool = True

    @model_validator(mode="after")
    def dynamic_objects_are_bound(self):
        if self.motion in {"dynamic", "field"} and self.state_binding is None:
            raise ValueError(f"dynamic object {self.id!r} requires state_binding")
        return self


class SceneGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    units: Literal["meters"] = "meters"
    up_axis: Literal["z"] = "z"
    objects: list[SceneObject] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_objects_and_bindings(self):
        identifiers = [item.id for item in self.objects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scene object ids must be unique")
        body_ids = [
            item.state_binding.value
            for item in self.objects
            if item.state_binding is not None and item.state_binding.selector == "body_id"
        ]
        if len(body_ids) != len(set(body_ids)):
            raise ValueError("body_id bindings must be unique")
        all_bindings = [
            item.id
            for item in self.objects
            if item.state_binding is not None and item.state_binding.selector == "all"
        ]
        dynamic_count = sum(item.motion in {"dynamic", "field"} for item in self.objects)
        if all_bindings and dynamic_count > 1:
            raise ValueError(
                "selector 'all' is only allowed when the scene has one dynamic/field object"
            )
        return self


def load_scene_geometry(path: Path | None) -> SceneGeometry | None:
    if path is None or not path.is_file():
        return None
    geometry = SceneGeometry.model_validate(load_yaml(path))
    base = path.parent.resolve()
    for item in geometry.objects:
        for field_name in ("asset", "topology_asset"):
            value = getattr(item.geometry, field_name)
            if not value:
                continue
            candidate = Path(value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (base / candidate).resolve()
            )
            try:
                resolved.relative_to(base)
            except ValueError as exc:
                raise ValueError(f"Scene asset escapes scaffold: {value}") from exc
            if not resolved.is_file():
                raise ValueError(f"Scene asset does not exist: {value}")
            setattr(item.geometry, field_name, str(resolved))
    return geometry


def validate_frame_bindings(frame, geometry: SceneGeometry) -> None:
    """Ensure a solver's StateFrame can be partitioned exactly as the frozen scene declares."""
    for item in geometry.objects:
        binding = item.state_binding
        if binding is None:
            continue
        if binding.selector == "body_id":
            if frame.body_id is None:
                raise ValueError(
                    f"StateFrame.body_id is required by scene object {item.id!r}"
                )
            if not np.any(np.asarray(frame.body_id) == binding.value):
                raise ValueError(
                    f"StateFrame has no samples for body_id={binding.value!r} "
                    f"required by scene object {item.id!r}"
                )
        elif binding.selector == "attribute":
            if binding.attribute_name not in frame.attributes:
                raise ValueError(
                    f"StateFrame attribute {binding.attribute_name!r} is required by "
                    f"scene object {item.id!r}"
                )
            values = np.asarray(frame.attributes[binding.attribute_name])
            if not np.any(values == binding.value):
                raise ValueError(
                    f"StateFrame attribute {binding.attribute_name!r} has no samples with "
                    f"value {binding.value!r} required by scene object {item.id!r}"
                )
        elif binding.selector == "field" and binding.attribute_name not in frame.fields:
            raise ValueError(
                f"StateFrame field {binding.attribute_name!r} is required by "
                f"scene object {item.id!r}"
            )
