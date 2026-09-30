"""Deterministic constrained elastic-rod solver for the hair scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.cfg = None
        self.pos = None
        self.vel = None
        self.mass = None
        self.rest_lengths = None
        self.root_segment = None
        self._time = 0.0
        self._impulse = np.zeros(3, dtype=float)
        self._last_external_force = np.zeros((3, 3), dtype=float)
        self._step_count = 0

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.mass[0] = 0.0
        self.rest_lengths = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.root_segment = self.pos[1] - self.pos[0]
        self._time = 0.0
        self._impulse[:] = 0.0
        gravity_force = np.sum(
            self.mass[:, None] * np.asarray(self.cfg["gravity"], dtype=float)[None, :], axis=0
        )
        self._last_external_force[0] = gravity_force
        self._last_external_force[1] = 0.0
        self._last_external_force[2] = -gravity_force
        self._step_count = 0

    def _root_kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        drive = self.cfg["root_motion"]
        root0 = np.asarray(drive["rest_position"], dtype=float)
        axis = np.asarray(drive["axis"], dtype=float)
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        start = float(drive["start_time"])
        if t <= start:
            return root0.copy(), np.zeros(3), np.zeros(3)
        omega = 2.0 * np.pi * float(drive["frequency"])
        phase = omega * (t - start) + float(drive.get("phase", 0.0))
        amp = float(drive["amplitude"])
        s, c = np.sin(phase), np.cos(phase)
        return root0 + axis * (amp * s), axis * (amp * omega * c), axis * (-amp * omega * omega * s)

    def _forces(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.pos)
        gravity = np.asarray(self.cfg["gravity"], dtype=float)
        hair = self.cfg["hair_clump"]
        drag_coeff = float(hair["ambient_drag_nu"])
        air = np.asarray(hair.get("ambient_air_velocity", [0.0, 0.0, 0.0]), dtype=float)
        force = self.mass[:, None] * gravity[None, :]
        force -= drag_coeff * (self.vel - air[None, :])
        ds = max(float(np.mean(self.rest_lengths)), 1.0e-6)
        effective_EI = 2.0e-8
        bend_coeff = effective_EI / (ds ** 3)
        gamma = float(hair["internal_damping_gamma"])
        bend_damp = gamma / (ds ** 3)
        for i in range(1, n - 1):
            curvature = self.pos[i - 1] - 2.0 * self.pos[i] + self.pos[i + 1]
            rate = self.vel[i - 1] - 2.0 * self.vel[i] + self.vel[i + 1]
            qforce = bend_coeff * curvature + bend_damp * rate
            force[i - 1] -= qforce
            force[i] += 2.0 * qforce
            force[i + 1] -= qforce
        force[0] = 0.0
        return force, force.sum(axis=0)

    def _step(self, h: float, target_time: float) -> None:
        root_new, root_vel, _ = self._root_kinematics(target_time)
        force, _ = self._forces()
        old_momentum = np.sum(self.mass[:, None] * self.vel, axis=0)
        accel = np.zeros_like(self.pos)
        accel[1:] = force[1:] / self.mass[1:, None]
        old_pos = self.pos.copy()
        predicted = self.pos + self.vel * h + accel * (h * h)
        predicted[0] = root_new
        predicted[1] = root_new + self.root_segment
        inv_mass = np.zeros_like(self.mass)
        inv_mass[2:] = 1.0 / self.mass[2:]
        for _ in range(6):
            predicted[0] = root_new
            predicted[1] = root_new + self.root_segment
            for j, length in enumerate(self.rest_lengths):
                d = predicted[j + 1] - predicted[j]
                dist = float(np.linalg.norm(d))
                if dist < 1.0e-12:
                    continue
                correction = (dist - length) * d / dist
                wi, wj = inv_mass[j], inv_mass[j + 1]
                total = wi + wj
                if total <= 0.0:
                    continue
                predicted[j] += correction * (wi / total)
                predicted[j + 1] -= correction * (wj / total)
        self.pos = predicted
        self.vel = (self.pos - old_pos) / h
        self.pos[0] = root_new
        self.pos[1] = root_new + self.root_segment
        self.vel[0] = root_vel
        self.vel[1] = root_vel
        self.vel[2:] *= 0.9995
        new_momentum = np.sum(self.mass[:, None] * self.vel, axis=0)
        gravity_force = np.sum(
            self.mass[:, None] * np.asarray(self.cfg["gravity"], dtype=float)[None, :], axis=0
        )
        hair = self.cfg["hair_clump"]
        air = np.asarray(hair.get("ambient_air_velocity", [0.0, 0.0, 0.0]), dtype=float)
        drag_force = np.sum(
            -float(hair["ambient_drag_nu"]) * (self.vel - air[None, :]), axis=0
        )
        reaction_force = (new_momentum - old_momentum) / h - gravity_force - drag_force
        self._last_external_force[0] = gravity_force
        self._last_external_force[1] = drag_force
        self._last_external_force[2] = reaction_force
        self._impulse += new_momentum - old_momentum
        self._step_count += 1

    def _frame(self, time: float) -> StateFrame:
        n = len(self.pos)
        root_target, _, _ = self._root_kinematics(time)
        total_mass = float(np.sum(self.mass))
        fields = {
            "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, n), dtype=float),
            "budget_external_force": self._last_external_force[None, :, :].copy(),
            "constraint_target": root_target[None, :].copy(),
            "constraint_value": self.pos[0][None, :].copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        frame = StateFrame(
            time=float(time), position=self.pos.copy(), velocity=self.vel.copy(),
            mass=self.mass.copy(), attributes={"mass_quantity": self.mass.copy()},
            fields=fields,
            scalars={
                "constraint_scale": max(float(self.cfg["root_motion"]["amplitude"]), 1.0e-8),
                "reference_momentum": max(
                    total_mass * float(self.cfg["root_motion"]["amplitude"])
                    * 2.0 * np.pi * float(self.cfg["root_motion"]["frequency"]), 1.0e-8
                ),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1.0e-10:
            raise ValueError("advance_to requires finite nondecreasing time")
        while self._time < target - 1.0e-12:
            h = min(0.0015, target - self._time)
            self._step(h, self._time + h)
            self._time += h
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_constrained_elastic_rod", "node_count": int(len(self.pos)) if self.pos is not None else 0, "internal_steps": int(self._step_count)}
