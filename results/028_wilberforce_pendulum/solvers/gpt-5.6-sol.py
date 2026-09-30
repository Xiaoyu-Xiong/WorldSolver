"""Self-contained Wilberforce-pendulum physics solver.

The visible helical spring is represented by its two lowest elastic modes: axial
extension and torsion. A symmetric extension/twist term in the elastic energy
couples the modes. The generalized coordinates are integrated conservatively
with velocity Verlet; curve samples are reconstructed from the lower attachment
and accumulated twist.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Numerical stretch--twist model of a suspended helical spring."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()

        self._count = int(getattr(scene, "spring_vertex_count", 128))
        self._n = self._count + 1
        self._time = 0.0

        # Geometry is taken from the frozen initial state rather than duplicated.
        spring = np.asarray(initial.position[: self._count], dtype=float)
        self._top_z = float(spring[0, 2])
        self._weight_z0 = float(initial.position[self._count, 2])
        self._weight_half_height = float(spring[-1, 2] - self._weight_z0)
        self._radius = float(np.hypot(spring[0, 0], spring[0, 1]))
        self._s = np.linspace(0.0, 1.0, self._count)
        self._base_phase = np.unwrap(np.arctan2(spring[:, 1], spring[:, 0]))

        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=int).copy()
        self._spring_mass = float(np.sum(self._mass[: self._count]))
        self._weight_mass = float(self._mass[self._count])

        # Axial effective mass includes one third of a spring whose velocity
        # grows linearly from the fixed to the free attachment.
        self._m_eff = self._weight_mass + self._spring_mass / 3.0
        radius_weight = 0.34
        self._inertia = 0.5 * self._weight_mass * radius_weight**2

        # Near-resonant modes give the characteristic slow exchange envelope.
        # Coupling is below sqrt(k_z*k_theta), so the energy is positive definite.
        omega_z = 4.55
        omega_theta = 4.46
        self._k_z = self._m_eff * omega_z**2
        self._k_theta = self._inertia * omega_theta**2
        self._coupling = 0.50

        # q is upward displacement from the statically stretched configuration.
        self._q = 0.0
        self._q_velocity = float(initial.velocity[self._count, 2])
        self._theta = 0.0
        self._theta_velocity = float(initial.angular_velocity[self._count, 2])
        self._max_step = 0.002
        self._steps = 0

        self._initial_energy = self._energy()
        self._last_energy = self._initial_energy

    def _acceleration(self, q: float, theta: float) -> tuple[float, float]:
        """Acceleration from the gradient of coupled elastic energy."""
        q_acc = (-self._k_z * q - self._coupling * theta) / self._m_eff
        theta_acc = (-self._k_theta * theta - self._coupling * q) / self._inertia
        return q_acc, theta_acc

    def _step(self, dt: float) -> None:
        """One second-order symplectic velocity-Verlet step."""
        aq, at = self._acceleration(self._q, self._theta)
        qv_half = self._q_velocity + 0.5 * dt * aq
        tv_half = self._theta_velocity + 0.5 * dt * at
        self._q += dt * qv_half
        self._theta += dt * tv_half
        aq_new, at_new = self._acceleration(self._q, self._theta)
        self._q_velocity = qv_half + 0.5 * dt * aq_new
        self._theta_velocity = tv_half + 0.5 * dt * at_new
        self._time += dt
        self._steps += 1

    def _energy(self) -> float:
        kinetic = (
            0.5 * self._m_eff * self._q_velocity**2
            + 0.5 * self._inertia * self._theta_velocity**2
        )
        elastic = (
            0.5 * self._k_z * self._q**2
            + 0.5 * self._k_theta * self._theta**2
            + self._coupling * self._q * self._theta
        )
        return float(kinetic + elastic)

    def _make_frame(self, requested_time: float) -> StateFrame:
        s = self._s
        q_acc, theta_acc = self._acceleration(self._q, self._theta)
        phase = self._base_phase + s * self._theta
        phase_velocity = s * self._theta_velocity
        phase_acceleration = s * theta_acc

        position = np.empty((self._n, 3), dtype=float)
        velocity = np.zeros((self._n, 3), dtype=float)
        acceleration = np.zeros((self._n, 3), dtype=float)

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        position[: self._count, 0] = self._radius * cos_phase
        position[: self._count, 1] = self._radius * sin_phase
        bottom_z = self._weight_z0 + self._weight_half_height + self._q
        position[: self._count, 2] = self._top_z + s * (bottom_z - self._top_z)

        velocity[: self._count, 0] = -self._radius * sin_phase * phase_velocity
        velocity[: self._count, 1] = self._radius * cos_phase * phase_velocity
        velocity[: self._count, 2] = s * self._q_velocity
        acceleration[: self._count, 0] = -self._radius * (
            cos_phase * phase_velocity**2 + sin_phase * phase_acceleration
        )
        acceleration[: self._count, 1] = self._radius * (
            -sin_phase * phase_velocity**2 + cos_phase * phase_acceleration
        )
        acceleration[: self._count, 2] = s * q_acc

        position[self._count] = (0.0, 0.0, self._weight_z0 + self._q)
        velocity[self._count, 2] = self._q_velocity
        acceleration[self._count, 2] = q_acc

        # Quaternion ordering follows the scaffold: (x, y, z, w).
        sample_angle = s * self._theta
        orientation = np.zeros((self._n, 4), dtype=float)
        orientation[: self._count, 2] = np.sin(0.5 * sample_angle)
        orientation[: self._count, 3] = np.cos(0.5 * sample_angle)
        orientation[self._count, 2] = math.sin(0.5 * self._theta)
        orientation[self._count, 3] = math.cos(0.5 * self._theta)

        angular_velocity = np.zeros((self._n, 3), dtype=float)
        angular_velocity[: self._count, 2] = s * self._theta_velocity
        angular_velocity[self._count, 2] = self._theta_velocity

        # Gravity is external. The attachment reaction is applied at the first
        # sample from total momentum balance; elastic and coupling forces remain
        # internal and are deliberately not exported as external force.
        gravity = np.array((0.0, 0.0, -9.81), dtype=float)
        external_force = self._mass[:, None] * gravity[None, :]
        required_total = np.sum(self._mass[:, None] * acceleration, axis=0)
        external_force[0] += required_total - np.sum(external_force, axis=0)

        self._last_energy = self._energy()
        frame = StateFrame(
            time=float(requested_time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={"mass_quantity": self._mass.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
            },
            scalars={
                "mechanical_energy": self._last_energy,
                "axial_displacement": float(self._q),
                "torsional_angle": float(self._theta),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self._time - 1.0e-10:
            raise ValueError("sample times must be finite and nondecreasing")
        remaining = target - self._time
        while remaining > 1.0e-12:
            self._step(min(self._max_step, remaining))
            remaining = target - self._time
        self._time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        scale = max(abs(self._initial_energy), 1.0e-15)
        return {
            "model": "coupled axial-torsional oscillator",
            "integrator": "velocity_verlet",
            "steps": self._steps,
            "mechanical_energy": self._last_energy,
            "relative_energy_drift": (
                self._last_energy - self._initial_energy
            ) / scale,
        }
