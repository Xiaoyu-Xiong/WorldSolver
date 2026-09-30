"""Deterministic reduced continuum solver for the frozen soft-ball scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Time-step a homogeneous, nearly incompressible soft ball."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._scene = scene
        self._n = int(initial.position.shape[0])
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        if self._mass.shape != (self._n,):
            self._mass = np.full(self._n, 4.0 / self._n, dtype=float)
        self._total_mass = float(np.sum(self._mass))
        self._x0 = np.asarray(initial.position, dtype=float).copy()
        self._center0 = np.average(self._x0, axis=0, weights=self._mass)
        self._rel0 = self._x0 - self._center0
        self._zmin = float(np.min(self._rel0[:, 2]))

        self._g = 9.81
        self._shape_stiffness = 55.0
        self._shape_damping = 8.0
        self._contact_stiffness = 2800.0
        self._contact_damping = 70.0
        self._step = 0.001
        mode0 = np.column_stack((-0.5 * self._rel0[:, 0],
                                 -0.5 * self._rel0[:, 1], self._rel0[:, 2]))
        self._shape_inertia = max(float(np.sum(self._mass * np.sum(mode0 * mode0, axis=1))), 1e-6)

        self._time = 0.0
        self._center_z = float(self._center0[2])
        self._center_vz = 0.0
        self._q = 0.0
        self._qdot = 0.0
        self._impulse = np.zeros(3, dtype=float)
        self._last_contact_force = 0.0

    def _shape(self) -> tuple[float, float]:
        q = float(np.clip(self._q, -1.15, 0.65))
        return float(np.exp(q)), float(np.exp(-0.5 * q))

    def _advance_substep(self, h: float) -> None:
        sz, _ = self._shape()
        bottom = self._center_z + self._zmin * sz
        bottom_velocity = self._center_vz + self._zmin * sz * self._qdot
        penetration = max(0.0, -bottom)
        contact_force = max(0.0, float(self._contact_stiffness * penetration
                                       - self._contact_damping * bottom_velocity))
        gravity_force = -self._total_mass * self._g
        self._center_vz += h * (gravity_force + contact_force) / self._total_mass
        contact_q_force = contact_force * self._zmin * sz
        qdd = (contact_q_force - self._shape_stiffness * self._q
               - self._shape_damping * self._qdot) / self._shape_inertia
        self._qdot += h * qdd
        self._center_z += h * self._center_vz
        self._q = float(np.clip(self._q + h * self._qdot, -1.15, 0.65))

        sz_after, _ = self._shape()
        residual = self._center_z + self._zmin * sz_after
        if residual < 0.0:
            self._center_z -= residual
            vz_before = self._center_vz
            vbottom = self._center_vz + self._zmin * sz_after * self._qdot
            if vbottom < 0.0:
                self._center_vz -= vbottom
                self._impulse[2] += self._total_mass * (self._center_vz - vz_before)
        self._impulse[2] += h * (gravity_force + contact_force)
        self._last_contact_force = contact_force

    def _state(self, time: float) -> StateFrame:
        sz, sr = self._shape()
        deformation = np.array([[sr, 0.0, 0.0], [0.0, sr, 0.0], [0.0, 0.0, sz]], dtype=float)
        center = self._center0.copy()
        center[2] = self._center_z
        position = center[None, :] + self._rel0 @ deformation.T
        radial_rate = -0.5 * self._qdot * sr
        vertical_rate = self._qdot * sz
        velocity = np.empty_like(position)
        velocity[:, 0] = radial_rate * self._rel0[:, 0]
        velocity[:, 1] = radial_rate * self._rel0[:, 1]
        velocity[:, 2] = self._center_vz + vertical_rate * self._rel0[:, 2]
        below = position[:, 2] < 0.0
        if np.any(below):
            position[below, 2] = 0.0
            velocity[below, 2] = np.maximum(velocity[below, 2], 0.0)

        element_f = np.repeat(deformation[None, :, :], 20, axis=0)
        external_force = np.array([[[0.0, 0.0, -self._total_mass * self._g],
                                    [0.0, 0.0, self._last_contact_force]]], dtype=float)
        fields = {
            "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self._n), dtype=float),
            "budget_external_force": external_force,
            "constraint_target": np.array([1.0], dtype=float),
            "constraint_value": np.array([float(np.linalg.det(deformation))], dtype=float),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "element_deformation_gradient": element_f,
        }
        scalars = {"constraint_scale": 1.0,
                   "reference_momentum": max(1.0, self._total_mass * np.sqrt(self._g))}
        return StateFrame(float(time), position, velocity, mass=self._mass.copy(),
                          fields=fields, scalars=scalars)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1e-10:
            raise ValueError("advance_to requires finite nondecreasing times")
        while self._time < target - 1e-12:
            h = min(self._step, target - self._time)
            self._advance_substep(h)
            self._time += h
        self._time = target
        return self._state(target)

    def diagnostics(self) -> dict[str, object]:
        return {"model": "volume-preserving axial reduced continuum",
                "internal_step": self._step, "mesh_vertices": self._n}
