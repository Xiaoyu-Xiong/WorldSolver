"""Fixed, independently evaluated elastic energies for material.elastic_law.v1.

See elastic_model_registry.md beside this module for parameter ordering, energy
definitions, force conventions, masks, and numerical differentiation limits.
No solver-provided target, callback, or constitutive expression is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series


@dataclass(frozen=True)
class ElasticModel:
    name: str
    representation: str
    parameter_names: tuple[str, ...]


# These numeric IDs and parameter orders are part of the v1 contract.
MODEL_REGISTRY = MappingProxyType(
    {
        1: ElasticModel("neo_hookean", "volume", ("lambda", "mu")),
        2: ElasticModel("stvk", "volume", ("lambda", "mu")),
        3: ElasticModel("linear", "volume", ("lambda", "mu")),
        101: ElasticModel("cosserat_rod", "rod", ("GA1", "GA2", "EA", "EI1", "EI2", "GJ")),
        201: ElasticModel(
            "stvk_hinge_shell", "shell", ("young_modulus", "poisson_ratio", "thickness")
        ),
    }
)

_STEP = np.finfo(float).eps ** 0.2  # Five-point derivative: roundoff vs O(h^4).
_BRANCH_MARGIN = 0.01  # Larger than twice the rotational differentiation step.
_GEOMETRY_TOL = 1e-12  # Dimensionless shape degeneracy threshold.
_COMMON = ("material_model_id", "material_parameters", "element_material_id", "active_mask")
_VOLUME = ("deformation_gradient", "first_piola_stress", "reference_stress")
_SURFACE = ("position", "reference_position", "elastic_force", "reference_force", "topology")
_ROD = ("orientation", "reference_orientation", "elastic_couple", "reference_torque")


def _array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a finite real numeric array")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")


def _indices(array: np.ndarray, size: int, name: str) -> np.ndarray:
    if np.any(array != np.floor(array)) or np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} must contain integral indices in [0, {size})")
    return array.astype(np.intp)


def _length(vector: np.ndarray) -> float:
    # hypot avoids the avoidable overflow/underflow of sum(vector**2).
    return float(np.hypot.reduce(vector))


def _unit(vector: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    length = _length(vector)
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"{name} is degenerate")
    return vector / length, length


def _rotations(quaternions: np.ndarray, n: int, name: str) -> np.ndarray:
    _shape(quaternions, (n, 4), name)
    norms = np.hypot.reduce(quaternions, axis=1)
    if not np.allclose(norms, 1, rtol=0, atol=1e-8):
        raise ValueError(f"{name} must contain unit xyzw quaternions")
    if n == 0:
        return np.empty((0, 3, 3))
    return Rotation.from_quat(quaternions).as_matrix()


def _rotation_log(matrix: np.ndarray) -> np.ndarray:
    vector = Rotation.from_matrix(matrix).as_rotvec()
    if _length(vector) >= np.pi - _BRANCH_MARGIN:
        raise ValueError("Rod relative rotation is too close to the pi log branch; refine the mesh")
    return vector


def _rod_strain(
    x: np.ndarray, rotations: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    relative = _rotation_log(rotations[0].T @ rotations[1])
    midpoint = rotations[0] @ Rotation.from_rotvec(0.5 * relative).as_matrix()
    return midpoint.T @ (x[1] - x[0]) / length, relative / length


def _negative_gradient(energy, x: np.ndarray, length: float) -> np.ndarray:
    """Local, translation-recentered five-point differentiation, O(local DOFs)."""
    force = np.empty_like(x)
    h = _STEP * length
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Element is outside the representable differentiation scale")
    for i in range(len(x)):
        for j in range(3):
            values = []
            for offset in (-2, -1, 1, 2):
                shifted = x.copy()
                shifted[i, j] += offset * h
                values.append(energy(shifted))
            a, b, c, d = values
            force[i, j] = ((d - a) - 8 * (c - b)) / (12 * h)
    return force


def _rod_response(x, reference, rotations, rest_rotations, stiffness):
    _, length = _unit(reference[1] - reference[0], "Reference rod edge")
    _unit(x[1] - x[0], "Current rod edge")
    # Evaluate the energy gradient by the chain rule. Differentiating total
    # energy would subtract large, unstressed shear-mode probe energies from
    # tiny twist energies on slender/long elements, losing the twist couple.
    local = x - x[0]
    gamma0, kappa0 = _rod_strain(reference - reference[0], rest_rotations, length)
    gamma, kappa = _rod_strain(local, rotations, length)
    strain = np.concatenate((gamma - gamma0, kappa - kappa0))
    energy_gradient = length * stiffness * strain
    midpoint = rotations[0] @ Rotation.from_rotvec(0.5 * length * kappa).as_matrix()
    tension = midpoint @ (stiffness[:3] * strain[:3])
    force = np.array([tension, -tension])
    couple = np.empty((2, 3))
    for i in range(2):
        for j in range(3):
            values = []
            for offset in (-2, -1, 1, 2):
                shifted = rotations.copy()
                increment = np.zeros(3)
                increment[j] = offset * _STEP
                # Left multiplication gives the derivative w.r.t. WORLD rotation.
                shifted[i] = Rotation.from_rotvec(increment).as_matrix() @ rotations[i]
                values.append(np.concatenate(_rod_strain(local, shifted, length)))
            a, b, c, d = values
            negative_strain_derivative = ((d - a) - 8 * (c - b)) / (12 * _STEP)
            couple[i, j] = np.dot(energy_gradient, negative_strain_derivative)
    return force, couple, float(0.5 * np.dot(energy_gradient, strain))


def _triangle_geometry(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    first, l1 = _unit(x[1] - x[0], "Triangle edge")
    second, l2 = _unit(x[2] - x[0], "Triangle edge")
    normal, sine = _unit(np.cross(first, second), "Triangle area")
    if sine <= _GEOMETRY_TOL:
        raise ValueError("Triangle is degenerate or too ill-conditioned")
    return normal, 0.5 * l1 * l2 * sine, min(l1, l2, _length(x[2] - x[1]))


def _membrane_response(x: np.ndarray, reference: np.ndarray, material: np.ndarray):
    normal, area, _ = _triangle_geometry(reference)
    _triangle_geometry(x)
    e1, _ = _unit(reference[1] - reference[0], "Reference triangle edge")
    basis = np.column_stack((e1, np.cross(normal, e1)))
    dm = basis.T @ (reference[1:] - reference[0]).T
    inv = np.linalg.inv(dm)
    f = (x[1:] - x[0]).T @ inv
    strain = 0.5 * (f.T @ f - np.eye(2))
    young, nu, thickness = material[:3]
    mu = young / (2 * (1 + nu))
    lam = young * nu / (1 - nu * nu)  # Plane-stress Lame coefficient.
    second_piola = 2 * mu * strain + lam * np.trace(strain) * np.eye(2)
    derivative = area * thickness * f @ second_piola @ inv.T
    force = np.vstack((np.sum(derivative, axis=1), -derivative.T))
    energy = area * thickness * (mu * np.sum(strain * strain) + 0.5 * lam * np.trace(strain) ** 2)
    return force, float(energy)


def _hinges(topology: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    edges: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for triangle, (a, b, c) in enumerate(topology):
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            edges.setdefault(tuple(sorted((int(i), int(j)))), []).append((triangle, i, j, k))
    hinges = []
    for adjacent in edges.values():
        if len(adjacent) > 2:
            raise ValueError("Shell topology is nonmanifold")
        if len(adjacent) == 2:
            left, i, j, k = adjacent[0]
            right, u, v, ell = adjacent[1]
            if u != j or v != i:
                raise ValueError("Shell triangles must have consistent winding")
            hinges.append((left, right, np.array([i, j, k, ell], dtype=np.intp)))
    return hinges


def _dihedral(x: np.ndarray) -> float:
    tangent, _ = _unit(x[1] - x[0], "Hinge edge")
    n1, _, _ = _triangle_geometry(x[[0, 1, 2]])
    n2, _, _ = _triangle_geometry(x[[1, 0, 3]])
    return float(np.arctan2(np.dot(tangent, np.cross(n1, n2)), np.dot(n1, n2)))


def _hinge_response(x, reference, left_material, right_material):
    _, area1, scale1 = _triangle_geometry(reference[[0, 1, 2]])
    _, area2, scale2 = _triangle_geometry(reference[[1, 0, 3]])
    edge = _length(reference[1] - reference[0])
    theta0 = _dihedral(reference)
    bending = []
    for material in (left_material, right_material):
        young, nu, thickness = material[:3]
        bending.append(young * thickness**3 / (12 * (1 - nu * nu)))
    stiffness = 0.5 * sum(bending) * edge**2 / (area1 + area2)

    def energy(points):
        delta = _dihedral(points) - theta0
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        if abs(delta) >= np.pi - _BRANCH_MARGIN:
            raise ValueError("Shell fold is too close to the pi bending branch; refine the mesh")
        return 0.5 * stiffness * delta**2

    local = x - x[0]
    # The minimum current/rest altitude scale keeps probes local on distorted meshes.
    _, a1, s1 = _triangle_geometry(local[[0, 1, 2]])
    _, a2, s2 = _triangle_geometry(local[[1, 0, 3]])
    current_edge = _length(local[1] - local[0])
    scale = min(scale1, scale2, s1, s2, 2 * a1 / current_edge, 2 * a2 / current_edge)
    return _negative_gradient(energy, local, scale), float(energy(local))


def _volume_stress(f: np.ndarray, model: int, material: np.ndarray) -> np.ndarray:
    if np.linalg.det(f) <= 0:
        raise ValueError("Active deformation_gradient must have positive determinant")
    lam, mu = material[:2]
    if model == 1:
        inverse_transpose = np.linalg.inv(f).T
        return mu * (f - inverse_transpose) + lam * np.log(np.linalg.det(f)) * inverse_transpose
    if model == 2:
        strain = 0.5 * (f.T @ f - np.eye(3))
        return f @ (2 * mu * strain + lam * np.trace(strain) * np.eye(3))
    strain = 0.5 * (f + f.T) - np.eye(3)
    return 2 * mu * strain + lam * np.trace(strain) * np.eye(3)


def _material_table(data: dict[str, np.ndarray], representation: str):
    models = data["material_model_id"]
    table = data["material_parameters"]
    if models.ndim != 1 or not models.size or np.any(models != np.floor(models)):
        raise ValueError("material_model_id must be nonempty integral [Me]")
    if table.ndim != 2 or table.shape[0] != len(models):
        raise ValueError("material_parameters must be a rectangular [Me,Qp] array")
    for model, row in zip(models, table, strict=True):
        spec = MODEL_REGISTRY.get(int(model))
        if spec is None or spec.representation != representation:
            raise ValueError(f"Unsupported {representation} material_model_id {model}")
        width = len(spec.parameter_names)
        if len(row) < width or np.any(row[width:] != 0):
            raise ValueError(
                f"{spec.name} requires {width} parameters followed only by zero padding"
            )
        if representation == "volume":
            lam, mu = row[:2]
            if mu <= 0 or lam + 2 * mu / 3 <= 0:
                raise ValueError("Volume shear and bulk moduli must be positive")
        elif representation == "rod":
            if np.any(row[:6] <= 0):
                raise ValueError("All six rod stiffnesses must be positive")
        else:
            young, nu, thickness = row[:3]
            if young <= 0 or thickness <= 0 or not -1 < nu < 0.5:
                raise ValueError("Shell requires E>0, h>0 and -1<nu<0.5")
    return models.astype(np.intp), table


def _rms(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Elastic residual must be finite and nonempty")
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


class ElasticLawCalculator:
    id = "material.elastic_law.v1"
    metric_name = "elastic_constitutive_relative_residual"
    metric_unit = "dimensionless"

    def compute(
        self, frames: list[StateFrame], inputs: dict[str, str], parameters: dict[str, Any]
    ) -> CalculatorResult:
        if not frames:
            raise ValueError("Elastic law requires at least one frame")
        times = _array([frame.time for frame in frames], "Frame times")
        if times.ndim != 1 or np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("Frame times must be finite, nonnegative and strictly increasing")
        if set(parameters) - {"representation", "reference_source"}:
            raise ValueError(
                "Unsupported elastic parameters; only the fixed model registry is allowed"
            )
        representation = parameters.get("representation")
        if representation not in ("volume", "rod", "shell"):
            raise ValueError("representation must be volume, rod, or shell")
        if parameters.get("reference_source") != "frozen_material_model":
            raise ValueError("reference_source must be frozen_material_model")
        names = _COMMON + (_VOLUME if representation == "volume" else _SURFACE)
        if representation == "rod":
            names += _ROD
        series = {name: input_series(frames, inputs, name) for name in names}
        frozen: dict[str, np.ndarray] = {}
        persistent_materials: dict[tuple[int, ...], int] = {}
        errors, counts, node_counts, energies = [], [], [], []
        valid_frame_indices = []
        volume_pool_size = 0
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                for frame_index in range(len(frames)):
                    data = {
                        name: _array(values[frame_index], name) for name, values in series.items()
                    }
                    for name in (
                        "material_model_id",
                        "material_parameters",
                        "reference_position",
                        "reference_orientation",
                        "reference_force",
                        "reference_stress",
                        "reference_torque",
                    ):
                        if name not in data:
                            continue
                        value = data[name]
                        if name in ("reference_force", "reference_stress", "reference_torque"):
                            if value.shape != () or value <= 0:
                                raise ValueError(f"{name} must be a finite positive scalar")
                        if name in ("reference_position", "reference_orientation"):
                            width = 3 if name == "reference_position" else 4
                            if value.ndim != 2 or value.shape[1] != width:
                                raise ValueError(f"{name} must have shape [N,{width}]")
                            if name in frozen:
                                old = frozen[name]
                                if (
                                    len(value) < len(old)
                                    or value[: len(old)].tobytes() != old.tobytes()
                                ):
                                    raise ValueError(
                                        f"{name} is append-only; existing rows must remain frozen"
                                    )
                            frozen[name] = value.copy()
                        else:
                            if name in frozen and not np.array_equal(frozen[name], value):
                                raise ValueError(f"{name} must remain frozen across frames")
                            frozen.setdefault(name, value.copy())
                    models, table = _material_table(data, representation)
                    material_ids = data["element_material_id"]
                    if material_ids.ndim != 1:
                        raise ValueError("element_material_id must have shape [Ne]")
                    ne = len(material_ids)
                    material_ids = _indices(material_ids, len(models), "element_material_id")
                    active = data["active_mask"]
                    _shape(active, (ne,), "active_mask")
                    if np.any((active != 0) & (active != 1)):
                        raise ValueError("active_mask must contain only 0 or 1")
                    active = active.astype(bool)
                    counts.append(int(np.sum(active)))
                    if representation == "volume":
                        f = data["deformation_gradient"]
                        _shape(f, (ne, 3, 3), "deformation_gradient (N must match Ne explicitly)")
                        actual = data["first_piola_stress"]
                        _shape(actual, (ne, 3, 3), "first_piola_stress")
                        if ne < volume_pool_size:
                            raise ValueError(
                                "Volume element index pool is append-only; mask deleted rows"
                            )
                        volume_pool_size = ne
                        for i, material in enumerate(material_ids):
                            key = (i,)
                            if (
                                key in persistent_materials
                                and persistent_materials[key] != material
                            ):
                                raise ValueError(
                                    "Persistent element material assignment must remain frozen"
                                )
                            persistent_materials[key] = int(material)
                        if not np.any(active):
                            continue
                        expected = np.array(
                            [
                                _volume_stress(
                                    f[i], models[material_ids[i]], table[material_ids[i]]
                                )
                                for i in np.flatnonzero(active)
                            ]
                        )
                        errors.append(_rms((actual[active] - expected) / data["reference_stress"]))
                    else:
                        response = self._discrete_frame(
                            data, representation, material_ids, table, active, persistent_materials
                        )
                        if response is None:
                            continue
                        error, nodes, energy = response
                        errors.append(error)
                        node_counts.append(nodes)
                        energies.append(energy)
                    valid_frame_indices.append(frame_index)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            raise ValueError("Elastic calculation is nonfinite or geometrically singular") from exc
        if not errors:
            raise ValueError("Elastic dataset has no active evaluable frames")
        details: dict[str, Any] = {
            "representation": representation,
            "model_registry_version": "material.elastic_law.v1",
            "model_ids": frozen["material_model_id"].astype(int).tolist(),
            "active_element_counts": counts,
            "skipped_frame_indices": [i for i, count in enumerate(counts) if count == 0],
            "reduction": "Cartesian component RMS; rod force and couple concatenated",
        }
        if node_counts:
            details.update(
                evaluated_node_counts=node_counts,
                elastic_energy=energies,
                derivative=(
                    "analytic membrane/rod forces; five-point rod strain and hinge energy gradients"
                ),
                differentiation_relative_step=_STEP,
            )
        return CalculatorResult(
            np.asarray(errors), details, frame_indices=np.asarray(valid_frame_indices, dtype=int)
        )

    @staticmethod
    def _discrete_frame(data, representation, materials, table, active, persistent_materials):
        x, reference = data["position"], data["reference_position"]
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("position must have shape [N,3]")
        n = len(x)
        _shape(reference, (n, 3), "reference_position")
        _shape(data["elastic_force"], (n, 3), "elastic_force")
        topology = data["topology"]
        _shape(topology, (len(active), 2 if representation == "rod" else 3), "topology")
        topology = _indices(topology, n, "topology")
        seen = set()
        for element, material in zip(topology, materials, strict=True):
            key = tuple(sorted(int(i) for i in element))
            if len(set(key)) != len(key) or key in seen:
                raise ValueError("Topology has repeated vertices or duplicate elements")
            seen.add(key)
            if key in persistent_materials and persistent_materials[key] != material:
                raise ValueError("Persistent element material assignment must remain frozen")
            persistent_materials[key] = int(material)
        eligible = np.zeros(n, dtype=bool)
        eligible[topology[active].ravel()] = True
        eligible[topology[~active].ravel()] = False
        force = np.zeros((n, 3))
        energy = 0.0
        if representation == "rod":
            rotations = _rotations(data["orientation"], n, "orientation")
            rest_rotations = _rotations(data["reference_orientation"], n, "reference_orientation")
            _shape(data["elastic_couple"], (n, 3), "elastic_couple")
            if not np.any(active):
                return None
            couple = np.zeros((n, 3))
            for i in np.flatnonzero(active):
                nodes = topology[i]
                f, c, u = _rod_response(
                    x[nodes],
                    reference[nodes],
                    rotations[nodes],
                    rest_rotations[nodes],
                    table[materials[i], :6],
                )
                force[nodes] += f
                couple[nodes] += c
                energy += u
        else:
            hinges = _hinges(topology)
            if not np.any(active):
                return None
            for i in np.flatnonzero(active):
                nodes = topology[i]
                f, u = _membrane_response(x[nodes], reference[nodes], table[materials[i]])
                force[nodes] += f
                energy += u
            for left, right, nodes in hinges:
                if not (active[left] and active[right]):
                    # The unknown/excluded hinge can exert force on all four vertices.
                    eligible[nodes] = False
                    continue
                f, u = _hinge_response(
                    x[nodes], reference[nodes], table[materials[left]], table[materials[right]]
                )
                force[nodes] += f
                energy += u
        if not np.any(eligible):
            raise ValueError("No evaluable nodes with complete active elastic neighborhoods")
        residual = (
            (data["elastic_force"][eligible] - force[eligible]) / data["reference_force"]
        ).ravel()
        if representation == "rod":
            torque_residual = (
                (data["elastic_couple"][eligible] - couple[eligible]) / data["reference_torque"]
            ).ravel()
            residual = np.concatenate((residual, torque_residual))
        if not np.isfinite(energy):
            raise ValueError("Elastic energy must be finite")
        return _rms(residual), int(np.sum(eligible)), float(energy)
