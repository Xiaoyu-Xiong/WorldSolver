"""Shared integrated mechanical laws; no scene-specific target trajectories.

Every budget row describes an independent physical control volume. The exporter
must supply forces/fluxes for that same row, including inter-body forces when
only one body is selected. Row identities must remain stable across frames;
particle counts/order may change. Normalization scales are frozen SI inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame, frame_value

from .base import CalculatorResult


def _times(frames: list[StateFrame], minimum: int) -> np.ndarray:
    if len(frames) < minimum:
        raise ValueError(f"Requires at least {minimum} frames")
    t = np.asarray([f.time for f in frames], dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Frame times must be finite, nonnegative and strictly increasing")
    return t


def _array(frame: StateFrame, inputs: dict[str, str], name: str) -> np.ndarray:
    if name not in inputs:
        raise ValueError(f"Calculator input {name!r} is not bound")
    raw = np.asarray(frame_value(frame, inputs[name]))
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    a = np.asarray(raw, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def _shape(a: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    return a


def _scale(frames: list[StateFrame], inputs: dict[str, str], name: str) -> float:
    values = [_array(f, inputs, name) for f in frames]
    if any(a.ndim != 0 or a <= 0 for a in values):
        raise ValueError(f"{name} must be a positive scalar")
    if any(a != values[0] for a in values[1:]):
        raise ValueError(f"{name} must remain frozen across frames")
    return float(values[0])


def _parameters(parameters: dict[str, Any], *, energy: bool = False) -> str:
    allowed = (
        {"integration", "window", "potential_reference"}
        if energy
        else {"mode", "integration", "window", "origin"}
    )
    if set(parameters) - allowed:
        raise ValueError(f"Unsupported parameters: {sorted(set(parameters) - allowed)}")
    if parameters.get("integration", "trapezoidal") != "trapezoidal":
        raise ValueError("Only trapezoidal integration is supported")
    if parameters.get("window", "adjacent_frames") != "adjacent_frames":
        raise ValueError("Only adjacent_frames windows are supported")
    mode = parameters.get("mode", "dynamic")
    if mode not in {"dynamic", "quasistatic"}:
        raise ValueError("mode must be dynamic or quasistatic")
    return mode


def _origin(parameters: dict[str, Any], key: str) -> np.ndarray:
    origin = np.asarray(parameters.get(key, [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError(f"{key} must be a finite fixed three-vector")
    # Exported cumulative angular fluxes have the fixed-origin contract used by YAML.
    if np.any(origin != 0):
        raise ValueError(f"v1 requires {key} [0,0,0]")
    return origin


def _samples(frame: StateFrame, inputs: dict[str, str]):
    m = _array(frame, inputs, "mass")
    if m.ndim != 1 or np.any(m < 0):
        raise ValueError("mass must be a nonnegative [N] array")
    v = _shape(_array(frame, inputs, "velocity"), (len(m), 3), "velocity")
    w = _array(frame, inputs, "domain_weights")
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] != len(m):
        raise ValueError("domain_weights must have shape [G,N] with G>0")
    if np.any((w < 0) | (w > 1)):
        raise ValueError("domain_weights must be in [0,1]")
    return m, v, w


def _spin(frame: StateFrame, inputs: dict[str, str], n: int):
    omega = _shape(_array(frame, inputs, "angular_velocity"), (n, 3), "angular_velocity")
    inertia = _shape(_array(frame, inputs, "inertia_tensor"), (n, 3, 3), "inertia_tensor")
    norm = np.max(np.abs(inertia), axis=(1, 2)) if n else np.empty(0)
    tolerance = 1e-12 * np.maximum(norm, np.finfo(float).tiny)
    if np.any(np.max(np.abs(inertia - inertia.swapaxes(-1, -2)), axis=(1, 2)) > tolerance):
        raise ValueError("inertia_tensor must be symmetric in world coordinates")
    if np.any(np.linalg.eigvalsh(inertia) < -tolerance[:, None]):
        raise ValueError("inertia_tensor must be positive semidefinite")
    spin = np.einsum("nij,nj->ni", inertia, omega)
    return omega, spin


def _force(frame: StateFrame, inputs: dict[str, str]) -> np.ndarray:
    force = _array(frame, inputs, "external_force")
    if force.ndim != 3 or force.shape[0] == 0 or force.shape[-1] != 3:
        raise ValueError("external_force must have shape [G,K,3] with G>0")
    return force


class MomentumBudgetCalculator:
    id = "dynamics.momentum_budget.v1"
    metric_name = "linear_momentum_budget_residual"
    metric_unit = "dimensionless"
    angular = False

    def compute(self, frames, inputs, parameters):
        mode = _parameters(parameters)
        if not self.angular and "origin" in parameters:
            raise ValueError("origin is only supported for angular momentum")
        origin = _origin(parameters, "origin") if self.angular else np.zeros(3)
        times = _times(frames, 1 if mode == "quasistatic" else 2)
        scale_name = (
            ("reference_torque" if self.angular else "reference_force")
            if mode == "quasistatic"
            else ("reference_angular_momentum" if self.angular else "reference_momentum")
        )
        scale = _scale(frames, inputs, scale_name)
        rates, quantities, transfers, selected_mass = [], [], [], []
        groups = None
        for frame in frames:
            force = _force(frame, inputs)
            g = force.shape[0]
            if groups is not None and g != groups:
                raise ValueError("Budget domain row count must remain stable")
            groups = g
            if self.angular:
                point = _shape(
                    _array(frame, inputs, "force_position"), force.shape, "force_position"
                )
                couple = _shape(
                    _array(frame, inputs, "external_couple"), force.shape, "external_couple"
                )
                rate = (np.cross(point - origin, force) + couple).sum(axis=1)
            else:
                rate = force.sum(axis=1)
            rates.append(rate)
            if mode == "quasistatic":
                if force.shape[1] == 0:
                    raise ValueError("Quasistatic equilibrium requires sampled force terms")
                continue
            m, v, weights = _samples(frame, inputs)
            if weights.shape[0] != g:
                raise ValueError("Force and state budget domains do not align")
            selected_mass.append(weights @ m)
            q = m[:, None] * v
            if self.angular:
                x = _shape(_array(frame, inputs, "position"), v.shape, "position")
                _, spin = _spin(frame, inputs, len(m))
                q = np.cross(x - origin, q) + spin
            quantities.append(weights @ q)
            suffix = "angular_momentum" if self.angular else "momentum"
            impulse = (
                "cumulative_external_angular_impulse"
                if self.angular
                else "cumulative_external_impulse"
            )
            terms = [
                _shape(_array(frame, inputs, name), (g, 3), name)
                for name in (impulse, f"cumulative_inflow_{suffix}", f"cumulative_outflow_{suffix}")
            ]
            # Difference each cumulative stream before combining: a small new
            # impulse must not disappear beside a large historic flux offset.
            transfers.append(terms)
        if mode == "quasistatic":
            errors = np.hypot.reduce(np.asarray(rates) / scale, axis=-1).max(axis=1)
            return CalculatorResult(errors, {"mode": mode, "sample_times": times.tolist()})
        if np.any(np.asarray(selected_mass).sum(axis=0) <= 0):
            raise ValueError("Each budget domain needs positive sampled mass at some frame")
        dt = np.diff(times)[:, None, None]
        rates = np.asarray(rates)
        increments = np.diff(np.asarray(transfers), axis=0)
        residual = np.diff(np.asarray(quantities), axis=0) - (
            0.5 * dt * (rates[:-1] + rates[1:])
            + increments[:, 0]
            + increments[:, 1]
            - increments[:, 2]
        )
        errors = np.hypot.reduce(residual / scale, axis=-1).max(axis=1)
        return CalculatorResult(
            errors,
            {
                "mode": mode,
                "interval_end_times": times[1:].tolist(),
                "domain_count": groups,
                "normalization_scale": scale,
            },
            frame_indices=np.arange(1, len(frames)),
        )


class AngularMomentumBudgetCalculator(MomentumBudgetCalculator):
    id = "dynamics.angular_momentum_budget.v1"
    metric_name = "angular_momentum_budget_residual"
    angular = True


class WorkBalanceCalculator:
    id = "energy.work_balance.v1"
    metric_name = "mechanical_work_energy_budget_residual"
    metric_unit = "dimensionless"

    def compute(self, frames, inputs, parameters):
        _parameters(parameters, energy=True)
        origin = _origin(parameters, "potential_reference")
        times = _times(frames, 2)
        scale = _scale(frames, inputs, "reference_energy")
        energies, powers, transfers, masks, selected_mass = [], [], [], [], []
        frozen_gravity, groups = None, None
        for frame in frames:
            m, v, w = _samples(frame, inputs)
            g = w.shape[0]
            if groups is not None and groups != g:
                raise ValueError("Energy domain row count must remain stable")
            groups = g
            selected_mass.append(w @ m)
            x = _shape(_array(frame, inputs, "position"), v.shape, "position")
            omega, spin = _spin(frame, inputs, len(m))
            elastic = _shape(_array(frame, inputs, "elastic_energy"), m.shape, "elastic_energy")
            gravity = _shape(_array(frame, inputs, "gravity"), (1, 3), "gravity")[0]
            if frozen_gravity is not None and not np.array_equal(gravity, frozen_gravity):
                raise ValueError("gravity must remain frozen across frames")
            frozen_gravity = gravity
            e = 0.5 * (m * np.sum(v * v, axis=1) + np.sum(omega * spin, axis=1))
            e += elastic - m * ((x - origin) @ gravity)
            energies.append(w @ e)
            power = _array(frame, inputs, "external_power")
            dissipation = _array(frame, inputs, "dissipation_power")
            for name, array in (("external_power", power), ("dissipation_power", dissipation)):
                if array.ndim != 2 or array.shape[0] != g:
                    raise ValueError(f"{name} must have shape [G,K]")
            if np.any(dissipation < 0):
                raise ValueError("dissipation_power must be nonnegative, not fitted from residual")
            powers.append(power.sum(axis=1) - dissipation.sum(axis=1))
            incoming = _shape(
                _array(frame, inputs, "cumulative_inflow_energy"), (g,), "cumulative_inflow_energy"
            )
            outgoing = _shape(
                _array(frame, inputs, "cumulative_outflow_energy"),
                (g,),
                "cumulative_outflow_energy",
            )
            transfers.append([incoming, outgoing])
            mask = _shape(
                _array(frame, inputs, "active_interval_mask"), (g,), "active_interval_mask"
            )
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError("active_interval_mask must be binary")
            masks.append(mask.astype(bool))
        if np.any(np.asarray(selected_mass).sum(axis=0) <= 0):
            raise ValueError("Each energy domain needs positive sampled mass at some frame")
        powers = np.asarray(powers)
        increments = np.diff(np.asarray(transfers), axis=0)
        residual = np.diff(np.asarray(energies), axis=0) - (
            0.5 * np.diff(times)[:, None] * (powers[:-1] + powers[1:])
            + increments[:, 0]
            - increments[:, 1]
        )
        # Frame i's mask describes (i-1,i]. Frame zero has no preceding interval.
        active = np.asarray(masks)[1:]
        if np.any(~np.any(active, axis=0)):
            raise ValueError("Each energy domain requires at least one active interval")
        errors, ends, indices = [], [], []
        for index, selection in enumerate(active):
            if np.any(selection):
                errors.append(float(np.max(np.abs(residual[index, selection]))) / scale)
                ends.append(float(times[index + 1]))
                indices.append(index + 1)
        return CalculatorResult(
            np.asarray(errors),
            {
                "interval_end_times": ends,
                "active_domain_intervals": int(active.sum()),
                "normalization_scale": scale,
                "note": "Elastic energy and dissipation require independently audited exports.",
            },
            frame_indices=np.asarray(indices, dtype=int),
        )
