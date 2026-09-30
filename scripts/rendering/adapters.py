from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:
    method: str
    geometry_kinds: frozenset[str]
    dynamic: bool


ADAPTERS = {
    item.method: item
    for item in (
        AdapterDescriptor(
            "rigid_mesh",
            frozenset(
                {
                    "plane",
                    "box",
                    "sphere",
                    "cylinder",
                    "capsule",
                    "axisymmetric_container",
                    "triangle_mesh",
                    "heightfield",
                }
            ),
            True,
        ),
        AdapterDescriptor("deformed_mesh", frozenset({"triangle_mesh"}), True),
        AdapterDescriptor("shell_surface", frozenset({"triangle_mesh"}), True),
        AdapterDescriptor("curve_tubes", frozenset({"curve_network"}), True),
        AdapterDescriptor("particle_instances", frozenset({"particle_domain"}), True),
        AdapterDescriptor("reconstructed_surface", frozenset({"particle_domain"}), True),
        AdapterDescriptor("volume", frozenset({"voxel_grid"}), True),
        AdapterDescriptor("diagnostic_points", frozenset({"particle_domain"}), True),
    )
}


def adapter_for(method: str) -> AdapterDescriptor:
    try:
        return ADAPTERS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown visual adapter: {method}") from exc
