"""Constitutive checks for the 055--154 numeric bindings (SI quantities).

Only raw, collocated physical quantities are accepted, never a solver's target or
residual. Scalars really are rank zero; reference scales, viscosity, spacing,
density and gravity are frozen across the trajectory. StateFrame has no unit or
provenance metadata: the caller must supply the declared SI units and authenticate
the platform's material data and masks. Binary masks are applicability selections,
not residual-dependent weights; they may change with contact/phase evolution.

Grid masks certify *evaluation centers with valid stencils*, not the availability
of neighboring data. Centered differences use the supplied immediate neighbors,
including unscored boundary/ghost values. The outermost grid layer is never scored
(there is no boundary-condition or periodicity binding). No mask erosion, wrapping,
one-sided extrapolation, or extra solver-provided gradient is used. Cross-interface
or source-contaminated stencils must be excluded by the platform mask.

Inactive frames are validated but contribute no artificial zero error. Coulomb
reports a maximum per active frame; Newtonian reports a spatial RMS per active
frame. Darcy reports the maximum of flux RMS and any valid adjacent storage-rate
RMS at each active endpoint (including initial flux). Thus neither signed samples
nor flux/storage errors cancel. Missing intervals contribute no artificial zero.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

from .base import CalculatorResult, input_series


def _arithmetic_guard(method):
    @wraps(method)
    def guarded(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return method(*args, **kwargs)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("Non-finite or ill-conditioned constitutive arithmetic") from exc

    return guarded


def _parameters(parameters, defaults):
    unknown = set(parameters) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported parameters: {sorted(unknown)}")
    result = defaults | parameters
    for key, default in defaults.items():
        value = result[key]
        if isinstance(default, bool):
            valid = type(value) is bool and value == default
        elif isinstance(default, int):
            valid = type(value) is int and value in (2, 3)
        else:
            valid = isinstance(value, str) and value == default
        if not valid:
            raise ValueError(f"Unsupported {key}: {value!r}")
    return result


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _series(frames, inputs, names, *, minimum=1):
    if len(frames) < minimum:
        raise ValueError(f"At least {minimum} frame(s) required")
    times = [_array(frame.time, "time") for frame in frames]
    if any(t.shape != () or t < 0 for t in times):
        raise ValueError("time must be a finite nonnegative scalar")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Frame times must be strictly increasing")
    if set(inputs) != set(names):
        raise ValueError(
            f"Invalid input bindings; missing={sorted(set(names) - set(inputs))}, "
            f"unknown={sorted(set(inputs) - set(names))}"
        )
    data = {}
    for name in names:
        try:
            values = input_series(frames, inputs, name)
        except KeyError as exc:
            raise ValueError(f"Missing required input {name}: {exc}") from exc
        data[name] = [_array(value, name) for value in values]
    return data


def _frozen_scalar(data, name):
    values = data[name]
    if any(v.shape != () or v <= 0 for v in values):
        raise ValueError(f"{name} must be a positive scalar in the declared units")
    if any(v != values[0] for v in values[1:]):
        raise ValueError(f"{name} must remain frozen across frames")
    return float(values[0])


def _shape(value, shape, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def _mask(value, shape):
    _shape(value, shape, "active_mask")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("active_mask must be binary (0 or 1)")
    return value.astype(bool)


def _grid(mask, dimensions):
    if mask.ndim != dimensions or any(n < 3 for n in mask.shape):
        raise ValueError(f"Grid must have {dimensions} axes, each with at least three cells")
    active = _mask(mask, mask.shape)
    interior = (slice(1, -1),) * dimensions
    return mask.shape, interior, active[interior]


def _derivative(value, axis, dimensions, spacing):
    high = [slice(1, -1)] * dimensions
    low = high.copy()
    high[axis] = slice(2, None)
    low[axis] = slice(None, -2)
    return (value[tuple(high)] - value[tuple(low)]) / (2 * spacing)


def _rms(value):
    # Scaling avoids squaring a large dimensional quantity or adding an SI floor.
    peak = float(np.max(np.abs(value)))
    return 0.0 if peak == 0 else peak * float(np.sqrt(np.mean((value / peak) ** 2)))


def _result(errors, counts, *, frame_indices, **details):
    if not errors:
        raise ValueError("No valid samples in the dataset")
    return CalculatorResult(
        np.asarray(errors),
        {"valid_sample_count": int(sum(counts)), "valid_sample_counts": counts, **details},
        frame_indices=np.asarray(frame_indices, dtype=int),
    )


class CoulombLawCalculator:
    id = "contact.coulomb_law.v1"
    metric_name = "coulomb_law_normalized_error"
    metric_unit = "dimensionless"

    @_arithmetic_guard
    def compute(
        self, frames: list[StateFrame], inputs: dict[str, str], parameters: dict[str, Any]
    ) -> CalculatorResult:
        _parameters(parameters, {"contact_regime": "sustained_dry", "require_active_samples": True})
        names = (
            "normal_force",
            "tangential_force",
            "relative_velocity",
            "normal",
            "static_coefficient",
            "dynamic_coefficient",
            "active_mask",
            "stick_speed_tolerance",
            "reference_force",
            "reference_velocity",
        )
        data = _series(frames, inputs, names)
        tolerance = _frozen_scalar(data, "stick_speed_tolerance")
        force_scale = _frozen_scalar(data, "reference_force")
        speed_scale = _frozen_scalar(data, "reference_velocity")
        errors, counts, indices = [], [], []
        sticking_count = sliding_count = 0
        for i in range(len(frames)):
            f = {name: values[i] for name, values in data.items()}
            fn = f["normal_force"]
            if fn.ndim != 1:
                raise ValueError("normal_force must have shape [C]")
            count = len(fn)
            for name in ("tangential_force", "relative_velocity", "normal"):
                _shape(f[name], (count, 3), name)
            for name in ("static_coefficient", "dynamic_coefficient"):
                _shape(f[name], (count,), name)
            active = _mask(f["active_mask"], (count,))
            static, dynamic = f["static_coefficient"], f["dynamic_coefficient"]
            if np.any(fn < 0):
                raise ValueError("normal_force must be nonnegative compression")
            if np.any(static < 0) or np.any(dynamic < 0) or np.any(dynamic > static):
                raise ValueError("Coefficients must satisfy 0 <= dynamic <= static")
            normals = f["normal"]
            if not np.allclose(np.linalg.norm(normals, axis=-1), 1, rtol=0, atol=1e-10):
                raise ValueError("normal must contain unit vectors")
            counts.append(int(active.sum()))
            if not active.any():
                continue
            # Normalize forces before projections, limits and residual norms;
            # squaring dimensional forces can underflow at valid small SI scales.
            n, fn = normals[active], fn[active] / force_scale
            ft = f["tangential_force"][active] / force_scale
            velocity = f["relative_velocity"][active]
            vt = velocity - np.sum(velocity * n, axis=-1)[:, None] * n
            # hypot also preserves tiny speeds for the physical stick/slide test.
            speed = np.hypot.reduce(vt, axis=-1)
            sticking = speed <= tolerance
            sliding = ~sticking
            sticking_count += int(sticking.sum())
            sliding_count += int(sliding.sum())
            normal_leak = np.sum(ft * n, axis=-1)
            tangential = ft - normal_leak[:, None] * n
            residual = np.abs(normal_leak)
            residual[sticking] = np.maximum.reduce(
                (
                    residual[sticking],
                    np.maximum(
                        np.hypot.reduce(tangential[sticking], axis=-1)
                        - static[active][sticking] * fn[sticking],
                        0,
                    ),
                    speed[sticking] / speed_scale,
                )
            )
            target = (
                -dynamic[active][sliding, None]
                * fn[sliding, None]
                * (vt[sliding] / speed[sliding, None])
            )
            residual[sliding] = np.hypot.reduce(ft[sliding] - target, axis=-1)
            # Friction power F_t . v_t must be nonpositive, including near-stick cases.
            power_violation = np.maximum(np.sum(ft * (vt / speed_scale), axis=-1), 0)
            errors.append(float(np.max(np.maximum(residual, power_violation))))
            indices.append(i)
        return _result(
            errors,
            counts,
            frame_indices=indices,
            sticking_sample_count=sticking_count,
            sliding_sample_count=sliding_count,
        )


class NewtonianStressCalculator:
    id = "fluid.newtonian_stress.v1"
    metric_name = "newtonian_stress_normalized_rms"
    metric_unit = "dimensionless"

    @_arithmetic_guard
    def compute(
        self, frames: list[StateFrame], inputs: dict[str, str], parameters: dict[str, Any]
    ) -> CalculatorResult:
        representation = parameters.get("representation", "grid")
        if representation == "incompressible_slender_thread":
            _parameters(parameters, {"representation": representation})
            return self._thread(frames, inputs)
        p = _parameters(parameters, {"representation": "grid", "spatial_dimensions": 3})
        dimensions = p["spatial_dimensions"]
        names = (
            "dynamic_viscosity",
            "velocity_field",
            "viscous_stress",
            "active_mask",
            "grid_spacing",
            "reference_stress",
        )
        data = _series(frames, inputs, names)
        mu = _frozen_scalar(data, "dynamic_viscosity")
        spacing = _frozen_scalar(data, "grid_spacing")
        scale = _frozen_scalar(data, "reference_stress")
        errors, counts, indices, dissipation_errors = [], [], [], []
        grid_shape = None
        for i in range(len(frames)):
            shape, interior, active = _grid(data["active_mask"][i], dimensions)
            if grid_shape is not None and shape != grid_shape:
                raise ValueError("Newtonian requires a fixed collocated grid across frames")
            grid_shape = shape
            velocity, stress = data["velocity_field"][i], data["viscous_stress"][i]
            _shape(velocity, shape + (dimensions,), "velocity_field")
            _shape(stress, shape + (dimensions, dimensions), "viscous_stress")
            counts.append(int(active.sum()))
            if not active.any():
                continue
            gradient = np.stack(
                [_derivative(velocity, axis, dimensions, spacing) for axis in range(dimensions)],
                axis=-1,
            )[active]
            strain = 0.5 * (gradient + gradient.swapaxes(-1, -2))
            actual = stress[interior][active]
            residual = np.linalg.norm((actual - 2 * mu * strain) / scale, axis=(-2, -1))
            # tau:D [Pa/s], normalized by reference stress times local ||D|| [1/s].
            rate = np.linalg.norm(strain, axis=(-2, -1))
            direction = np.divide(
                strain,
                rate[:, None, None],
                out=np.zeros_like(strain),
                where=rate[:, None, None] > 0,
            )
            violation = np.maximum(-np.sum((actual / scale) * direction, axis=(-2, -1)), 0)
            errors.append(_rms(np.maximum(residual, violation)))
            dissipation_errors.append(_rms(violation))
            indices.append(i)
        return _result(
            errors,
            counts,
            frame_indices=indices,
            representation="grid",
            dissipation_normalized_errors=dissipation_errors,
            stencil="centered_interior; mask certifies valid evaluation stencils",
        )

    def _thread(self, frames, inputs):
        names = (
            "dynamic_viscosity",
            "position",
            "velocity",
            "edges",
            "cross_section_area",
            "viscous_axial_force",
            "active_mask",
            "reference_force",
        )
        data = _series(frames, inputs, names)
        mu = _frozen_scalar(data, "dynamic_viscosity")
        scale = _frozen_scalar(data, "reference_force")
        errors, counts, indices, dissipation_errors = [], [], [], []
        for i in range(len(frames)):
            position, velocity, edges = (
                data[name][i] for name in ("position", "velocity", "edges")
            )
            if position.ndim != 2 or position.shape[-1] != 3:
                raise ValueError("position must have shape [N,3]")
            _shape(velocity, position.shape, "velocity")
            if edges.ndim != 2 or edges.shape[-1] != 2:
                raise ValueError("edges must have shape [Nv,2]")
            if (
                np.any(edges != np.floor(edges))
                or np.any(edges < 0)
                or np.any(edges >= len(position))
            ):
                raise ValueError("edges must contain integral, in-range node indices")
            edges = edges.astype(np.intp)
            count = len(edges)
            area, actual = data["cross_section_area"][i], data["viscous_axial_force"][i]
            _shape(area, (count,), "cross_section_area")
            _shape(actual, (count,), "viscous_axial_force")
            active = _mask(data["active_mask"][i], (count,))
            # Deleted/inactive segments may have zero area or coincident endpoints.
            if np.any(area < 0) or np.any(area[active] <= 0):
                raise ValueError(
                    "cross_section_area must be positive on active edges, nonnegative elsewhere"
                )
            counts.append(int(active.sum()))
            if not active.any():
                continue
            edge = edges[active]
            delta = position[edge[:, 1]] - position[edge[:, 0]]
            length = np.linalg.norm(delta, axis=-1)
            if np.any(length <= 0):
                raise ValueError("Active thread edges must have positive length")
            dv = velocity[edge[:, 1]] - velocity[edge[:, 0]]
            rate = np.sum((delta / length[:, None]) * dv, axis=-1) / length
            target = 3 * mu * area[active] * rate
            residual = np.abs((actual[active] - target) / scale)
            violation = np.maximum(-(actual[active] / scale) * np.sign(rate), 0)
            errors.append(_rms(np.maximum(residual, violation)))
            dissipation_errors.append(_rms(violation))
            indices.append(i)
        return _result(
            errors,
            counts,
            frame_indices=indices,
            representation="incompressible_slender_thread",
            dissipation_normalized_errors=dissipation_errors,
        )


class DarcyBalanceCalculator:
    id = "porous.darcy_balance.v1"
    metric_name = "darcy_balance_normalized_rms"
    metric_unit = "dimensionless"

    @_arithmetic_guard
    def compute(
        self, frames: list[StateFrame], inputs: dict[str, str], parameters: dict[str, Any]
    ) -> CalculatorResult:
        p = _parameters(
            parameters, {"spatial_dimensions": 3, "flux_definition": "relative_to_solid"}
        )
        if p["spatial_dimensions"] != 3:
            raise ValueError("Darcy binding requires spatial_dimensions=3")
        names = (
            "porosity",
            "saturation",
            "pressure",
            "solid_velocity",
            "relative_flux",
            "permeability",
            "source",
            "active_mask",
            "grid_spacing",
            "dynamic_viscosity",
            "fluid_density",
            "gravity",
            "reference_flux",
            "reference_storage_rate",
        )
        data = _series(frames, inputs, names, minimum=2)
        spacing = _frozen_scalar(data, "grid_spacing")
        mu = _frozen_scalar(data, "dynamic_viscosity")
        density = _frozen_scalar(data, "fluid_density")
        flux_scale = _frozen_scalar(data, "reference_flux")
        storage_scale = _frozen_scalar(data, "reference_storage_rate")
        for gravity in data["gravity"]:
            _shape(gravity, (1, 3), "gravity")
            if not np.array_equal(gravity, data["gravity"][0]):
                raise ValueError("gravity must remain frozen across frames")
        gravity = data["gravity"][0][0]
        flux_errors, flux_counts, flux_indices = [], [], []
        endpoint_errors = {}
        storage_errors, interval_counts, interval_indices = [], [], []
        previous = None
        grid_shape = None
        for i, frame in enumerate(frames):
            f = {name: values[i] for name, values in data.items()}
            shape, interior, active = _grid(f["active_mask"], 3)
            if grid_shape is not None and shape != grid_shape:
                raise ValueError("Darcy requires a fixed collocated grid across frames")
            grid_shape = shape
            for name in ("porosity", "saturation", "pressure", "source"):
                _shape(f[name], shape, name)
            for name in ("solid_velocity", "relative_flux"):
                _shape(f[name], shape + (3,), name)
            _shape(f["permeability"], shape + (3, 3), "permeability")
            for name in ("porosity", "saturation"):
                if np.any(f[name] < 0) or np.any(f[name] > 1):
                    raise ValueError(f"{name} must lie in [0,1]")
            permeability = f["permeability"]
            # Relative tensor tolerance, never an absolute SI floor: tiny physical
            # permeabilities must not make a negative or asymmetric K acceptable.
            magnitude = np.max(np.abs(permeability), axis=(-2, -1))
            unit_k = np.divide(
                permeability,
                magnitude[..., None, None],
                out=np.zeros_like(permeability),
                where=magnitude[..., None, None] > 0,
            )
            if np.any(np.max(np.abs(unit_k - unit_k.swapaxes(-1, -2)), axis=(-2, -1)) > 1e-12):
                raise ValueError("permeability must be symmetric")
            if np.any(np.linalg.eigvalsh(0.5 * (unit_k + unit_k.swapaxes(-1, -2))) < -1e-12):
                raise ValueError("permeability must be positive semidefinite")
            storage = f["porosity"] * f["saturation"]
            transport = storage[..., None] * f["solid_velocity"] + f["relative_flux"]
            divergence = sum(
                _derivative(transport[..., axis], axis, 3, spacing) for axis in range(3)
            )
            rhs = f["source"][interior] - divergence
            flux_counts.append(int(active.sum()))
            if active.any():
                gradient = np.stack(
                    [_derivative(f["pressure"], axis, 3, spacing) for axis in range(3)], axis=-1
                )[active]
                target = -np.einsum(
                    "nij,nj->ni",
                    permeability[interior][active] / mu,
                    gradient - density * gravity,
                )
                error = np.linalg.norm(
                    (f["relative_flux"][interior][active] - target) / flux_scale, axis=-1
                )
                flux_errors.append(_rms(error))
                flux_indices.append(i)
                endpoint_errors[i] = flux_errors[-1]
            if previous is not None:
                old_time, old_storage, old_rhs, old_active = previous
                selected = active & old_active
                interval_counts.append(int(selected.sum()))
                if selected.any():
                    # Adjacent endpoints only. Divide the integrated balance by dt
                    # before normalizing: reference_storage_rate has units 1/s.
                    dt = float(frame.time) - old_time
                    rate = (storage[interior][selected] - old_storage[selected]) / dt
                    error = (rate - 0.5 * (rhs[selected] + old_rhs[selected])) / storage_scale
                    storage_errors.append(_rms(error))
                    interval_indices.append([i - 1, i])
                    endpoint_errors[i] = max(endpoint_errors[i], storage_errors[-1])
            previous = (float(frame.time), storage[interior], rhs, active)
        if not storage_errors:
            raise ValueError("No valid samples in adjacent Darcy storage intervals")
        return _result(
            list(endpoint_errors.values()),
            flux_counts + interval_counts,
            frame_indices=list(endpoint_errors),
            flux_normalized_errors=flux_errors,
            storage_normalized_errors=storage_errors,
            flux_valid_sample_counts=flux_counts,
            storage_valid_sample_counts=interval_counts,
            flux_valid_sample_count=sum(flux_counts),
            storage_valid_sample_count=sum(interval_counts),
            flux_frame_indices=flux_indices,
            storage_interval_indices=interval_indices,
            error_order="active endpoints: max(flux RMS, available adjacent storage RMS)",
            stencil="centered_interior; mask certifies valid evaluation stencils",
        )
