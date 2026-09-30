from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .shared_law_fields import SHARED_LAW_FIELDS
from .state import StateFrame


@dataclass(frozen=True, slots=True)
class TrajectoryFieldDefinition:
    path: str
    shape: str
    unit: str
    description: str


# Public canonical raw-state vocabulary used by executable metrics. Solver
# submissions export physical samples and source terms, never scored residuals.
STANDARD_FIELDS = {
    item.path: item
    for item in (
        TrajectoryFieldDefinition("state.mass", "[N]", "kilogram", "Per-sample mass."),
        TrajectoryFieldDefinition("state.position", "[N,3]", "meter", "Per-sample position."),
        TrajectoryFieldDefinition(
            "state.velocity", "[N,3]", "meter_per_second", "Per-sample velocity."
        ),
        TrajectoryFieldDefinition(
            "state.orientation",
            "[N,4]",
            "dimensionless",
            "Per-sample orientation quaternion in xyzw order.",
        ),
        TrajectoryFieldDefinition(
            "state.angular_velocity",
            "[N,3]",
            "radian_per_second",
            "Per-sample angular velocity.",
        ),
        TrajectoryFieldDefinition(
            "state.body_id", "[N]", "dimensionless", "Per-sample rigid body id."
        ),
        TrajectoryFieldDefinition(
            "attributes.phase_id", "[N]", "dimensionless", "Per-sample phase id."
        ),
        TrajectoryFieldDefinition(
            "scalars.liquid_phase_id", "scalar", "dimensionless", "Selected liquid phase id."
        ),
        TrajectoryFieldDefinition(
            "fields.velocity_field",
            "[nx,ny,2] or [nx,ny,nz,3]",
            "meter_per_second",
            "Diagnostic velocity grid.",
        ),
        TrajectoryFieldDefinition(
            "fields.liquid_interior_mask",
            "[nx,ny] or [nx,ny,nz]",
            "dimensionless",
            "Liquid interior mask.",
        ),
        TrajectoryFieldDefinition(
            "scalars.grid_spacing", "scalar", "meter", "Uniform diagnostic-grid spacing."
        ),
        TrajectoryFieldDefinition(
            "scalars.characteristic_length", "scalar", "meter", "Frozen normalization length."
        ),
        TrajectoryFieldDefinition(
            "fields.boundary_velocity", "[B,D]", "meter_per_second", "Velocity at wall samples."
        ),
        TrajectoryFieldDefinition(
            "fields.boundary_target_velocity",
            "[B,D]",
            "meter_per_second",
            "Prescribed wall velocity.",
        ),
        TrajectoryFieldDefinition(
            "fields.boundary_normal", "[B,D]", "dimensionless", "Unit wall normals."
        ),
        TrajectoryFieldDefinition(
            "fields.blade_boundary_velocity",
            "[B_blade,3]",
            "meter_per_second",
            "Measured fluid-side velocity at currently wetted moving-blade samples; "
            "do not substitute the prescribed target velocity.",
        ),
        TrajectoryFieldDefinition(
            "fields.blade_boundary_target_velocity",
            "[B_blade,3]",
            "meter_per_second",
            "Prescribed blade surface velocity omega(t) cross (sample_position-center), "
            "collocated with blade_boundary_velocity; separate from stationary wall samples.",
        ),
        TrajectoryFieldDefinition(
            "fields.blade_boundary_normal",
            "[B_blade,3]",
            "dimensionless",
            "Current rotated outward unit normals at the same wetted blade samples "
            "as blade_boundary_velocity and blade_boundary_target_velocity.",
        ),
        TrajectoryFieldDefinition(
            "scalars.reference_velocity", "scalar", "meter_per_second", "Velocity scale."
        ),
        TrajectoryFieldDefinition(
            "fields.liquid_level_set", "[nx,ny] or [nx,ny,nz]", "meter", "Signed liquid level set."
        ),
        TrajectoryFieldDefinition(
            "attributes.mass_quantity",
            "[N] or [N,K]",
            "kilogram",
            "Per-sample mass contribution, optionally split into K conserved materials.",
        ),
        TrajectoryFieldDefinition(
            "attributes.volume_quantity",
            "[N]",
            "meter_cubed",
            "Per-sample current volume contribution.",
        ),
        TrajectoryFieldDefinition(
            "attributes.deformation_gradient",
            "[N,D,D]",
            "dimensionless",
            "Per-element deformation gradient in material coordinates.",
        ),
        TrajectoryFieldDefinition(
            "fields.element_deformation_gradient",
            "[Ne,3,3]",
            "dimensionless",
            "Per-element deformation gradient in material coordinates; element order is fixed across frames.",
        ),
        TrajectoryFieldDefinition(
            "fields.cumulative_mass_source",
            "[K]",
            "kilogram",
            "Cumulative prescribed mass added minus removed since the initial frame.",
        ),
        TrajectoryFieldDefinition(
            "fields.cumulative_volume_source",
            "[K]",
            "meter_cubed",
            "Cumulative prescribed volume added minus removed since the initial frame.",
        ),
        TrajectoryFieldDefinition(
            "fields.external_force",
            "[N,3]",
            "newton",
            "Per-sample external force; excludes internal contact and coupling pairs.",
        ),
        TrajectoryFieldDefinition(
            "fields.interaction_force",
            "[K,3]",
            "newton",
            "All paired internal contact or multiphysics interaction forces.",
        ),
        TrajectoryFieldDefinition(
            "fields.contact_signed_distance",
            "[C]",
            "meter",
            "Signed geometric gap at active and near-contact samples; negative means penetration.",
        ),
        TrajectoryFieldDefinition(
            "fields.constraint_value",
            "[C,...]",
            "various",
            "Sampled physical value governed by the case-selected constraint.",
        ),
        TrajectoryFieldDefinition(
            "fields.constraint_target",
            "[C,...]",
            "various",
            "Frozen or prescribed target in the same coordinates and units as constraint_value.",
        ),
        TrajectoryFieldDefinition(
            "scalars.contact_length_scale",
            "scalar",
            "meter",
            "Frozen contact normalization length such as thickness or mean element size.",
        ),
        TrajectoryFieldDefinition(
            "scalars.constraint_scale",
            "scalar",
            "various",
            "Frozen nonzero normalization scale for the selected constraint value.",
        ),
        TrajectoryFieldDefinition(
            "fields.density_field",
            "[nx,ny] or [nx,ny,nz]",
            "kilogram_per_meter_cubed",
            "Collocated density or shallow-water height field.",
        ),
        TrajectoryFieldDefinition(
            "fields.continuity_source",
            "[nx,ny] or [nx,ny,nz]",
            "kilogram_per_meter_cubed_per_second",
            "Prescribed source in the continuity equation, zero where absent.",
        ),
        TrajectoryFieldDefinition(
            "fields.active_mask",
            "[nx,ny] or [nx,ny,nz]",
            "dimensionless",
            "Mask of cells on which the selected field equation applies.",
        ),
        TrajectoryFieldDefinition(
            "fields.transported_scalar",
            "[nx,ny] or [nx,ny,nz]",
            "various",
            "Collocated raw scalar being advected or diffused.",
        ),
        TrajectoryFieldDefinition(
            "fields.scalar_source",
            "[nx,ny] or [nx,ny,nz]",
            "various_per_second",
            "Prescribed local source for the transported scalar, zero where absent.",
        ),
    )
}


for _path, _shape, _unit, _description in SHARED_LAW_FIELDS:
    if _path in STANDARD_FIELDS:
        raise ValueError(f"Duplicate shared trajectory field: {_path}")
    STANDARD_FIELDS[_path] = TrajectoryFieldDefinition(_path, _shape, _unit, _description)


def collect_trajectory_input_paths(metric: dict[str, Any]) -> tuple[str, ...]:
    paths: set[str] = set()
    for criterion in metric.get("criteria", {}).get("physics", []):
        for path in criterion.get("numeric_binding", {}).get("inputs", {}).values():
            if path.startswith(("state.", "attributes.", "fields.", "scalars.")):
                paths.add(path)
    return tuple(sorted(paths))


def frame_value(frame: StateFrame, path: str):
    namespace, separator, name = path.partition(".")
    if not separator or not name:
        raise ValueError(f"Invalid trajectory field path: {path}")
    if namespace == "state":
        if not hasattr(frame, name):
            raise KeyError(path)
        value = getattr(frame, name)
    elif namespace in {"attributes", "fields", "scalars"}:
        value = getattr(frame, namespace).get(name)
    else:
        raise ValueError(f"Not a trajectory field path: {path}")
    if value is None:
        raise KeyError(path)
    return value


def validate_required_trajectory_fields(frame: StateFrame, required_paths: tuple[str, ...]) -> None:
    missing: list[str] = []
    for path in required_paths:
        try:
            frame_value(frame, path)
        except KeyError:
            missing.append(path)
    if missing:
        raise ValueError(f"StateFrame is missing required trajectory fields: {missing}")
