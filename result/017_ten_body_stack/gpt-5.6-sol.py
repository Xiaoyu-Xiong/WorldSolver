"""Deterministic penalty-contact solver for the ten-body rigid stack."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self._scene = scene
        self._time = float(initial.time)
        self._position = np.asarray(initial.position, dtype=float).copy()
        self._velocity = np.asarray(initial.velocity, dtype=float).copy()
        self._mass = (np.asarray(initial.mass, dtype=float).copy()
                      if initial.mass is not None else np.ones(len(self._position)))
        self._body_id = (np.asarray(initial.body_id, dtype=int).copy()
                         if initial.body_id is not None else np.arange(1, len(self._position) + 1))
        self._n = len(self._position)
        gravity = np.asarray(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._gravity = gravity if gravity.shape == (3,) else np.array([0.0, 0.0, -9.81])
        self._half_thickness = 0.10
        self._contact_rest = 2.0 * self._half_thickness
        self._ground_rest = self._half_thickness
        self._stiffness = 4000.0
        self._damping = 115.0
        self._substep = 0.001
        self._last_forces = self._mass[:, None] * self._gravity[None, :]
        self._max_penetration = 0.0

    def _contacts(self):
        z = self._position[:, 2]
        vz = self._velocity[:, 2]
        gaps = np.empty(self._n, dtype=float)
        normal = np.zeros(self._n, dtype=float)
        gaps[0] = z[0] - self._ground_rest
        if gaps[0] < 0.0:
            normal[0] = max(0.0, -self._stiffness * gaps[0] - self._damping * vz[0])
        if self._n > 1:
            gaps[1:] = z[1:] - z[:-1] - self._contact_rest
            relative_velocity = vz[1:] - vz[:-1]
            normal[1:] = np.maximum(
                0.0, -self._stiffness * np.minimum(gaps[1:], 0.0)
                - self._damping * relative_velocity
            )
        self._max_penetration = max(self._max_penetration, float(np.maximum(0.0, -gaps).max()))
        return gaps, normal

    def _step(self, dt: float) -> None:
        _, normal = self._contacts()
        force = (self._mass[:, None] * self._gravity[None, :]).copy()
        force[0, 2] += normal[0]
        if self._n > 1:
            force[:-1, 2] -= normal[1:]
            force[1:, 2] += normal[1:]
        self._velocity += (force / self._mass[:, None]) * dt
        self._position += self._velocity * dt
        self._last_forces = self._mass[:, None] * self._gravity[None, :]

    def _frame(self, time: float) -> StateFrame:
        gaps, normal = self._contacts()
        interaction = np.zeros((max(self._n - 1, 0), 3), dtype=float)
        if self._n > 1:
            interaction[:, 2] = normal[1:]
        return StateFrame(
            float(time), self._position.copy(), self._velocity.copy(),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (self._n, 1)),
            angular_velocity=np.zeros((self._n, 3), dtype=float),
            mass=self._mass.copy(), body_id=self._body_id.copy(),
            fields={"contact_signed_distance": gaps,
                    "external_force": self._last_forces.copy(),
                    "interaction_force": interaction},
            scalars={"contact_length_scale": self._contact_rest},
        )

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("requested time must be finite and non-negative")
        if target < self._time - 1e-12:
            raise ValueError("advance_to requires nondecreasing times")
        remaining = target - self._time
        while remaining > 1e-12:
            dt = min(self._substep, remaining)
            self._step(dt)
            self._time += dt
            remaining -= dt
        self._time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "semi_implicit_penalty_contact",
                "body_count": int(self._n),
                "simulated_time": float(self._time),
                "max_penetration": float(self._max_penetration)}
