"""Reduced-coordinate Wilberforce pendulum solver.

The spring is modelled as a single helical degree of freedom (length ``L``) and
the suspended weight as a torsional degree of freedom (twist ``theta``).  The
two modes are coupled through the elastic energy of the helix, giving the
characteristic slow energy exchange between vertical and torsional motion.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Symplectic time-stepper for a coupled vertical/torsional Wilberforce model."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        frame0 = scene.initial_state()
        frame0.validate()

        self.mass = np.asarray(frame0.mass, dtype=float)
        self.body_id = frame0.body_id
        self.n_samples = int(frame0.position.shape[0])
        self.spring_count = self.n_samples - 1

        # Geometry constants (metres)
        self.top_z = float(frame0.position[0, 2])
        self.weight_radius = 0.34
        self.helix_radius = 0.16
        self.weight_half_height = 0.21
        self.turns0 = 10.5

        # Effective inertias for the reduced model.
        m_spring = float(self.mass[: self.spring_count].sum())
        m_weight = float(self.mass[self.spring_count])
        self.M_eff = m_weight + m_spring / 3.0
        I_weight = 0.5 * m_weight * self.weight_radius ** 2
        I_spring = m_spring * self.helix_radius ** 2
        self.I_eff = I_weight + I_spring

        # Elastic parameters tuned so the vertical and torsional frequencies are
        # close (~0.5 Hz), with a weak linear coupling that transfers energy.
        omega = 2.0 * np.pi * 0.5
        self.k = self.M_eff * omega ** 2
        self.kappa = self.I_eff * omega ** 2
        self.coupling = 0.8
        self.damping = 0.03

        # Initial reduced coordinates derived from the provided initial frame.
        weight_z0 = float(frame0.position[self.spring_count, 2])
        self.L0 = self.top_z - (weight_z0 + self.weight_half_height)
        self.L_rest = self.L0 - (self.M_eff * 9.81) / self.k
        self.L = self.L0
        self.Ldot = -float(frame0.velocity[self.spring_count, 2])

        self.theta = 0.0
        if frame0.angular_velocity is not None:
            self.thetadot = float(frame0.angular_velocity[self.spring_count, 2])
        else:
            self.thetadot = 1.8

        self.time = 0.0
        self.substep = 1e-4
        self._diagnostics: dict = {
            "M_eff": self.M_eff,
            "I_eff": self.I_eff,
            "k": self.k,
            "kappa": self.kappa,
            "coupling": self.coupling,
        }

        # Cached frame used when the requested time has already been reached.
        self._frame = self._build_frame(self.time)
        self._initialized = True

    def _build_frame(self, t: float) -> StateFrame:
        n = self.n_samples
        s = np.linspace(0.0, 1.0, self.spring_count)

        # Material twist angle varies linearly from the fixed top to the bottom.
        twist_angle = self.theta * s
        phase = 2.0 * np.pi * self.turns0 * s + twist_angle

        position = np.zeros((n, 3), dtype=float)
        velocity = np.zeros((n, 3), dtype=float)

        # Helical spring samples.
        position[: self.spring_count, 0] = self.helix_radius * np.cos(phase)
        position[: self.spring_count, 1] = self.helix_radius * np.sin(phase)
        position[: self.spring_count, 2] = self.top_z - self.L * s

        velocity[: self.spring_count, 0] = (
            -self.helix_radius * np.sin(phase) * s * self.thetadot
        )
        velocity[: self.spring_count, 1] = (
            self.helix_radius * np.cos(phase) * s * self.thetadot
        )
        velocity[: self.spring_count, 2] = -self.Ldot * s

        # Suspended weight sample.
        weight_z = self.top_z - self.L - self.weight_half_height
        position[self.spring_count] = np.array([0.0, 0.0, weight_z])
        velocity[self.spring_count] = np.array([0.0, 0.0, -self.Ldot])

        # Orientations: fixed top, linearly twisted spring, rotating weight.
        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0
        sin_half = np.sin(0.5 * twist_angle)
        cos_half = np.cos(0.5 * twist_angle)
        orientation[: self.spring_count, 2] = sin_half
        orientation[: self.spring_count, 3] = cos_half

        # Weight orientation follows the total twist.
        weight_sin = np.sin(0.5 * self.theta)
        weight_cos = np.cos(0.5 * self.theta)
        orientation[self.spring_count] = np.array([0.0, 0.0, weight_sin, weight_cos])

        # Angular velocity: top fixed, spring interior twists linearly, weight rotates.
        angular_velocity = np.zeros((n, 3), dtype=float)
        angular_velocity[: self.spring_count, 2] = self.thetadot * s
        angular_velocity[self.spring_count, 2] = self.thetadot

        # External force is just gravity on each sample.
        external_force = np.zeros((n, 3), dtype=float)
        external_force[:, 2] = -9.81 * self.mass

        frame = StateFrame(
            t,
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass,
            body_id=self.body_id,
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
            },
        )
        frame.validate()
        return frame

    def _step(self, dt: float) -> None:
        """Advance the reduced coordinates with a symplectic Euler step."""
        # Coupled oscillator with weak linear coupling and small damping.
        f_L = (
            -self.k * (self.L - self.L_rest)
            - self.coupling * self.theta
            - self.damping * self.Ldot
        )
        f_theta = (
            -self.kappa * self.theta
            - self.coupling * (self.L - self.L_rest)
            - self.damping * self.thetadot
        )

        Ldd = f_L / self.M_eff
        thdd = f_theta / self.I_eff

        self.Ldot += Ldd * dt
        self.thetadot += thdd * dt
        self.L += self.Ldot * dt
        self.theta += self.thetadot * dt

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver must be initialized before advancing.")

        if time < self.time:
            raise ValueError("advance_to must be called with non-decreasing times.")

        dt_total = time - self.time
        if dt_total <= 0.0:
            return self._frame

        n_sub = max(1, int(np.ceil(dt_total / self.substep)))
        dt = dt_total / n_sub
        for _ in range(n_sub):
            self._step(dt)

        self.time = time
        self._frame = self._build_frame(time)
        return self._frame

    def diagnostics(self) -> dict:
        return self._diagnostics.copy()
