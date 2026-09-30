from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldsolver.config import load_yaml
from worldsolver.contracts.scene_geometry import SceneGeometry

from .adapters import adapter_for

VisualProfile = Literal[
    "rigid_contact",
    "deformable_surface",
    "cloth_shell",
    "fiber_rods",
    "granular_particles",
    "liquid_in_container",
    "gas_volume",
    "multiphysics",
]

RepresentationMethod = Literal[
    "rigid_mesh",
    "deformed_mesh",
    "shell_surface",
    "curve_tubes",
    "particle_instances",
    "reconstructed_surface",
    "volume",
    "diagnostic_points",
]


class VisualBinding(BaseModel):
    """Presentation of an authoritative scene object; geometry remains scene-owned."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    method: RepresentationMethod
    material: str = "default"
    visible: bool = True
    particle_radius_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    surface_pitch: float | None = Field(default=None, gt=0.0)
    surface_smoothing: float = Field(default=0.9, ge=0.0, le=4.0)
    subdivision: int = Field(default=0, ge=0, le=3)
    # Explicit, opt-in presentation controls; legacy bindings retain their defaults.
    color_rgb: list[float] | None = Field(default=None, min_length=3, max_length=3)
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    wireframe: bool = False
    scalar_field: str | None = None
    scalar_attribute: str | None = None
    scalar_range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    scalar_color_low: list[float] | None = Field(default=None, min_length=3, max_length=3)
    scalar_color_high: list[float] | None = Field(default=None, min_length=3, max_length=3)
    scalar_grid: bool = False
    field_axes: list[int] = Field(default_factory=lambda: [0, 1])
    max_points: int | None = Field(default=None, gt=0)
    volume_mode: Literal["points", "slices"] = "points"
    volume_axis: int = Field(default=1, ge=0, le=2)
    density_scale: float = Field(default=4.0, gt=0.0)

    @model_validator(mode="after")
    def valid_presentation(self):
        if self.color_rgb is not None and (
            not np.isfinite(self.color_rgb).all() or any(v < 0.0 or v > 1.0 for v in self.color_rgb)
        ):
            raise ValueError("color_rgb must contain three finite values in [0,1]")
        if self.scalar_field and self.scalar_attribute:
            raise ValueError("choose scalar_field or scalar_attribute, not both")
        if self.scalar_field or self.scalar_attribute:
            if self.method not in {"diagnostic_points", "deformed_mesh", "shell_surface"}:
                raise ValueError("scalar colors require diagnostic points or a deforming surface")
            if self.scalar_field and self.method != "diagnostic_points":
                raise ValueError("deforming surfaces require a per-vertex scalar_attribute")
            if self.scalar_range is None:
                raise ValueError("scalar colors require a fixed scalar_range")
        custom_scalar_colors = (self.scalar_color_low, self.scalar_color_high)
        if any(color is not None for color in custom_scalar_colors):
            if not all(color is not None for color in custom_scalar_colors):
                raise ValueError("custom scalar colors require both low and high colors")
            if not (self.scalar_field or self.scalar_attribute):
                raise ValueError("custom scalar colors require a scalar field or attribute")
            for color in custom_scalar_colors:
                invalid = not np.isfinite(color).all() or any(
                    value < 0.0 or value > 1.0 for value in color
                )
                if invalid:
                    raise ValueError("scalar colors must contain three finite values in [0,1]")
        if self.scalar_grid and (not self.scalar_field or len(self.field_axes) != 2):
            raise ValueError("scalar_grid requires a two-dimensional scalar_field")
        if self.scalar_range is not None and (
            not np.isfinite(self.scalar_range).all() or self.scalar_range[0] >= self.scalar_range[1]
        ):
            raise ValueError("scalar_range must be finite and increasing")
        if not self.field_axes or len(set(self.field_axes)) != len(self.field_axes):
            raise ValueError("field_axes must be distinct coordinate axes")
        if any(axis not in (0, 1, 2) for axis in self.field_axes):
            raise ValueError("field_axes entries must be 0, 1 or 2")
        if self.wireframe and self.method != "rigid_mesh":
            raise ValueError("wireframe requires rigid_mesh")
        if self.volume_mode == "slices" and self.method != "volume":
            raise ValueError("volume slices require volume method")
        return self


class RenderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    profile: VisualProfile
    environment: Literal["studio_neutral", "studio_dark", "outdoor_soft"] = "studio_neutral"
    camera: Literal["frozen_config"] = "frozen_config"
    bindings: list[VisualBinding] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_bindings(self):
        ids = [item.object_id for item in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("render bindings must have unique object_id values")
        return self


METHOD_BY_SEMANTIC = {
    "boundary": {"rigid_mesh"},
    "rigid": {"rigid_mesh"},
    "deformable": {"deformed_mesh"},
    "cloth": {"shell_surface"},
    "fiber": {"curve_tubes"},
    "granular": {"particle_instances", "reconstructed_surface", "diagnostic_points"},
    "liquid": {"reconstructed_surface", "diagnostic_points"},
    "gas": {"volume"},
    "emitter": {"rigid_mesh"},
    "sensor": {"rigid_mesh"},
}


def load_render_spec(path: Path | None) -> RenderSpec | None:
    if path is None or not path.is_file():
        return None
    return RenderSpec.model_validate(load_yaml(path))


def validate_render_spec(path: Path, geometry: SceneGeometry | None = None) -> dict:
    spec = load_render_spec(path)
    if spec is None:
        raise ValueError(f"Missing render specification: {path}")
    if geometry is not None:
        objects = {item.id: item for item in geometry.objects}
        for binding in spec.bindings:
            try:
                scene_object = objects[binding.object_id]
            except KeyError as exc:
                raise ValueError(f"Unknown render object_id: {binding.object_id}") from exc
            allowed = METHOD_BY_SEMANTIC[scene_object.semantic]
            if binding.method not in allowed:
                raise ValueError(
                    f"{binding.method!r} cannot represent {scene_object.semantic!r} "
                    f"object {binding.object_id!r}; expected one of {sorted(allowed)}"
                )
            adapter = adapter_for(binding.method)
            if scene_object.geometry.type not in adapter.geometry_kinds:
                raise ValueError(
                    f"{binding.method!r} does not support geometry "
                    f"{scene_object.geometry.type!r} for {binding.object_id!r}"
                )
        rendered = {item.object_id for item in spec.bindings if item.visible}
        required = {
            item.id
            for item in geometry.objects
            if item.semantic not in {"sensor"}
            and (item.collision_enabled or item.motion != "static")
        }
        missing = sorted(required - rendered)
        if missing:
            raise ValueError(f"Physical scene objects are missing visible bindings: {missing}")
    return spec.model_dump()
