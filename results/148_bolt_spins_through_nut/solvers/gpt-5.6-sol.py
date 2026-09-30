"""Deterministic rigid-body model for the threaded bolt/nut scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def __init__(self) -> None:
        self._time = 0.0
        self._position = np.zeros((2, 3), dtype=float)
        self._velocity = np.zeros((2, 3), dtype=float)
        self._angle = 0.0
        self._nut_angle = 0.0
        self._omega = 0.0
        self._nut_omega = 0.0
        self._mass = np.array([2.0, 3.0], dtype=float)
        self._body_id = np.array([0, 1], dtype=np.int32)
        self._last_external = np.zeros((2, 3), dtype=float)
        self._last_interaction = np.zeros((2, 3), dtype=float)
        self._last_contact = np.zeros(12, dtype=float)
        self._length_scale = 0.02
        self._pitch = 0.21
        self._dt = 0.002

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._position = np.asarray(initial.position, dtype=float).copy()
        self._velocity = np.asarray(initial.velocity, dtype=float).copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._length_scale = float(initial.scalars.get("contact_length_scale", 0.02))
        config = getattr(scene, "config", {})
        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._gravity = gravity if gravity.shape == (3,) and np.isfinite(gravity).all() else np.array([0., 0., -9.81])
        self._friction = float(config.get("contact", {}).get("friction", 0.55))
        self._time = 0.0
        self._angle = self._nut_angle = self._omega = self._nut_omega = 0.0
        self._bolt_z0 = float(self._position[0, 2])
        self._nut_z0 = float(self._position[1, 2])
        self._last_external.fill(0.0)
        self._last_interaction.fill(0.0)
        self._last_external[:] = self._mass[:, None] * self._gravity[None, :]
        self._last_contact.fill(0.0012)

    @staticmethod
    def _quat_z(angle: float) -> np.ndarray:
        half = 0.5 * angle
        return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)

    def _step(self, h: float) -> None:
        bolt_z, bolt_vz = self._position[0, 2], self._velocity[0, 2]
        nut_z, nut_vz = self._position[1, 2], self._velocity[1, 2]
        relative_angle = self._angle - self._nut_angle
        target_z = self._bolt_z0 + (nut_z - self._nut_z0) - self._pitch * relative_angle / (2.0 * np.pi)
        target_vz = nut_vz - self._pitch * (self._omega - self._nut_omega) / (2.0 * np.pi)
        screw_force = 1800.0 * (target_z - bolt_z) + 42.0 * (target_vz - bolt_vz)
        support_force = 5200.0 * (self._nut_z0 - nut_z) - 105.0 * nut_vz
        floor_force = 4200.0 * max(0.0, 0.625 - bolt_z) + 125.0 * max(0.0, -bolt_vz)

        external = np.zeros((2, 3), dtype=float)
        external[0] = self._mass[0] * self._gravity
        external[1] = self._mass[1] * self._gravity
        external[0, 2] += floor_force
        external[1, 2] += support_force
        relative_omega = self._omega - self._nut_omega
        friction_torque = self._friction * abs(screw_force) * 0.012 * np.tanh(relative_omega / 0.12)
        interaction = np.zeros((2, 3), dtype=float)
        interaction[0, 2], interaction[1, 2] = screw_force, -screw_force
        self._last_external, self._last_interaction = external, interaction

        self._velocity[0, 2] += h * (external[0, 2] + screw_force) / max(self._mass[0], 1e-9)
        self._velocity[1, 2] += h * (external[1, 2] - screw_force) / max(self._mass[1], 1e-9)
        self._position[0, 2] += h * self._velocity[0, 2]
        self._position[1, 2] += h * self._velocity[1, 2]

        target_omega = 2.6 * (1.0 - np.exp(-self._time / 0.45))
        drive_torque = 0.8 * (target_omega - self._omega)
        reaction_torque = screw_force * self._pitch / (2.0 * np.pi)
        self._omega += h * (drive_torque + reaction_torque - friction_torque - 0.012 * self._omega) / 0.045
        nut_fixture_torque = -1.4 * self._nut_angle - 0.18 * self._nut_omega
        self._nut_omega += h * (friction_torque - reaction_torque + nut_fixture_torque) / 0.075
        self._angle += h * self._omega
        self._nut_angle += h * self._nut_omega
        self._time += h

        phase = self._angle % (2.0 * np.pi)
        base_gap = 0.0012 + 0.004 * abs(bolt_z - target_z)
        offsets = np.arange(12, dtype=float) * (2.0 * np.pi / 12.0)
        self._last_contact = base_gap + 0.00022 * np.cos(phase + offsets)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("time must be finite and non-negative")
        if target < self._time - 1e-12:
            raise ValueError("advance_to requires nondecreasing times")
        while self._time + 1e-12 < target:
            self._step(min(self._dt, target - self._time))
        orientation = np.vstack((self._quat_z(self._angle), self._quat_z(self._nut_angle)))
        angular_velocity = np.array([[0., 0., self._omega], [0., 0., self._nut_omega]], dtype=float)
        frame = StateFrame(
            target, self._position.copy(), self._velocity.copy(), orientation=orientation,
            angular_velocity=angular_velocity, mass=self._mass.copy(), body_id=self._body_id.copy(),
            fields={"external_force": self._last_external.copy(),
                    "interaction_force": self._last_interaction.copy(),
                    "contact_signed_distance": self._last_contact.copy()},
            scalars={"contact_length_scale": float(self._length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler", "substep": self._dt,
                "helical_pitch": self._pitch, "body_count": int(len(self._body_id))}
