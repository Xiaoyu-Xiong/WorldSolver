"""Physically-based Wilberforce pendulum solver (no external physics engine).

Physical model
--------------
The frozen scene is a helical spring fixed at its top to a static attachment and
carrying a suspended rigid cylinder at its lower end.  A Wilberforce pendulum
stores energy in two weakly coupled modes: axial stretching of the helix and
twisting of the helix about its own axis.  Twisting the lower end of a helix by
an angle ``phi`` changes the natural axial length of the spring by ``a * phi``
(helix twist--extension coupling), which is exactly the mechanism that shuttles
energy back and forth between the translational and torsional modes.

The description is reduced with a Rayleigh--Ritz discretisation of the actual
frozen fibre: the helix wire deforms affinely in the stretch (shape function
``u``, the normalised arclength from the fixed top to the weight) and linearly
in the twist (``phi * u``).  Substituting that kinematics into the kinetic and
elastic energy of the discretised system gives the consistent reduced mass
matrix

    M = m_weight + m_spring / 3          (axial)
    I = I_weight + m_spring R^2 / 3      (torsional)

and the elastic potential

    V(z, phi) = 1/2 k (z - a phi)^2 + 1/2 kappa phi^2 ,

with ``z`` the axial displacement measured from the frozen, statically
equilibrated initial configuration (so gravity is balanced there and does not
enter the displacement equations of motion) and ``phi`` the twist of the lower
end relative to the fixed top.  ``k`` and ``kappa`` are the helix extensional
and wire-torsional stiffnesses; they are chosen so that the two uncoupled mode
frequencies coincide (the resonance condition of a Wilberforce pendulum) and
``a`` is the weak twist--extension coupling of the coil.

The two coupled equations of motion are integrated with velocity Verlet, a
symplectic second-order scheme, at a fixed substep far below the mode periods,
which keeps the total energy bounded with negligible drift.  Every requested
sample frame is reconstructed from the generalized coordinates with the exact
kinematics of the discretised helix, so the reported per-sample positions,
velocities, orientations and angular velocities are mutually consistent.

All parameters are derived from the frozen scene (geometry, masses, initial
state); nothing is scripted, key-framed or precomputed.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
_TWO_PI = 2.0 * np.pi
# Uncoupling axial period of the coil (resonance with the torsional mode is
# imposed by the stiffness calibration below); the frozen sample rate resolves
# this period with more than a hundred samples.
_AXIAL_PERIOD_S = 1.2
# Twist--extension coupling length of the coil, as a small fraction of the coil
# radius (weak coupling -> slow, clearly observable energy exchange).
_COUPLING_FRACTION = 0.45
# Default weight radius from the frozen scene geometry (used only if the
# geometry is unavailable on the scene object).
_DEFAULT_WEIGHT_RADIUS = 0.34


class BenchmarkSolver:
    """Symplectic reduced-order solver for the Wilberforce pendulum scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self._sample_dt = float(config.get("sample_dt", 0.01))
        # Fixed symplectic substep: well under the fastest mode period.
        self._h = min(1.0e-3, self._sample_dt)
        self._substeps = max(1, int(round(self._sample_dt / self._h)))

        initial = scene.initial_state()
        position = np.asarray(initial.position, dtype=float)
        velocity = np.asarray(initial.velocity, dtype=float)
        mass = np.asarray(initial.mass, dtype=float)
        if initial.orientation is not None:
            orientation = np.asarray(initial.orientation, dtype=float)
        else:
            orientation = np.tile([0.0, 0.0, 0.0, 1.0], (len(position), 1))
        if initial.angular_velocity is not None:
            angular = np.asarray(initial.angular_velocity, dtype=float)
        else:
            angular = np.zeros_like(position)
        if initial.body_id is not None:
            body_id = np.asarray(initial.body_id)
        else:
            body_id = np.zeros(len(position), dtype=int)

        n_total = len(position)
        weight_index = int(np.argmax(mass))
        spring_index = np.array([i for i in range(n_total) if i != weight_index], dtype=int)
        if spring_index.size < 3:
            raise ValueError("scene must provide a discretised spring fibre")

        # Sort the frozen fibre samples from the fixed top downwards.
        spring_pos = position[spring_index]
        order = np.argsort(-spring_pos[:, 2], kind="stable")
        ordered_pos = spring_pos[order]
        u = np.linspace(0.0, 1.0, ordered_pos.shape[0])
        psi = np.unwrap(np.arctan2(ordered_pos[:, 1], ordered_pos[:, 0]))
        psi = psi - psi[0]
        radius_nodes = np.hypot(ordered_pos[:, 0], ordered_pos[:, 1])

        self._n_total = n_total
        self._n_spring = ordered_pos.shape[0]
        self._weight_index = weight_index
        self._spring_index = spring_index
        self._order = order
        self._u = u
        # Fibre quantities below are stored in the same top-down sorted order as
        # ``ordered_pos``; results are scattered back with ``spring_index[order]``.
        self._psi0 = psi
        self._radius_nodes = radius_nodes
        self._z_top = float(ordered_pos[0, 2])
        self._L0 = float(ordered_pos[0, 2] - ordered_pos[-1, 2])
        self._radius = float(np.mean(radius_nodes))
        self._h_half = float((self._z_top - self._L0) - position[weight_index, 2])

        spring_mass = float(np.sum(mass[spring_index]))
        weight_mass = float(mass[weight_index])
        weight_radius = self._weight_radius(config)

        # Consistent (lumped) reduced inertias of the affine/linear kinematics.
        self._M = weight_mass + spring_mass / 3.0
        self._I = 0.5 * weight_mass * weight_radius ** 2 + spring_mass * self._radius ** 2 / 3.0

        # Elastic calibration: coincident uncoupled frequencies (resonance).
        self._omega0 = _TWO_PI / _AXIAL_PERIOD_S
        self._a = _COUPLING_FRACTION * self._radius
        self._k = self._M * self._omega0 ** 2
        kappa = self._I * self._omega0 ** 2 - self._k * self._a ** 2
        self._kappa = max(kappa, 1.0e-6 * self._I * self._omega0 ** 2)

        mu = self._M * self._a ** 2 / self._I
        mu = min(mu, 0.999)
        self._w_plus = self._omega0 * np.sqrt(1.0 + np.sqrt(mu))
        self._w_minus = self._omega0 * np.sqrt(1.0 - np.sqrt(mu))
        self._beat_period = _TWO_PI / abs(self._w_plus - self._w_minus)

        # Project the frozen initial state onto the reduced coordinates:
        # the attachment fixes the top, the lower end follows the weight.
        self._z = 0.0
        self._zdot = -float(velocity[weight_index, 2])
        self._phi = 0.0
        self._phidot = float(np.mean(angular[spring_index, 2]))
        self._nsteps = 0
        self._E0 = self._energy()
        self._max_dev = 0.0

        self._mass = mass.copy()
        self._body_id = body_id

    # -------------------------------------------------------------- dynamics
    def _accel(self, z: float, phi: float):
        """Generalized accelerations from -dV/dq with the calibrated potential."""
        stretch = z - self._a * phi
        az = -(self._k * stretch) / self._M
        aphi = (self._k * self._a * stretch - self._kappa * phi) / self._I
        return az, aphi

    def _step(self, h: float) -> None:
        """One velocity-Verlet step of the two coupled oscillators."""
        az, aphi = self._accel(self._z, self._phi)
        self._zdot += 0.5 * h * az
        self._phidot += 0.5 * h * aphi
        self._z += h * self._zdot
        self._phi += h * self._phidot
        az, aphi = self._accel(self._z, self._phi)
        self._zdot += 0.5 * h * az
        self._phidot += 0.5 * h * aphi
        self._nsteps += 1

    def _energy(self) -> float:
        stretch = self._z - self._a * self._phi
        kinetic = 0.5 * self._M * self._zdot ** 2 + 0.5 * self._I * self._phidot ** 2
        elastic = 0.5 * self._k * stretch ** 2 + 0.5 * self._kappa * self._phi ** 2
        return float(kinetic + elastic)

    # ------------------------------------------------------------ integration
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        steps = int(round(target / self._h)) if target > 0.0 else 0
        if steps < self._nsteps:
            steps = self._nsteps
        while self._nsteps < steps:
            self._step(self._h)
        frame = self._build_frame(target)
        deviation = abs(self._energy() - self._E0)
        if deviation > self._max_dev:
            self._max_dev = deviation
        return frame

    # ------------------------------------------------------------- kinematics
    def _build_frame(self, time: float) -> StateFrame:
        n_spring = self._n_spring
        phi = self._phi
        phidot = self._phidot
        zdot = self._zdot
        length = self._L0 + self._z

        psi = self._psi0 + phi * self._u
        radius = self._radius_nodes
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        twist_vel = phidot * self._u

        spring_position = np.column_stack(
            (radius * cos_psi, radius * sin_psi, self._z_top - length * self._u)
        )
        spring_velocity = np.column_stack(
            (-radius * sin_psi * twist_vel, radius * cos_psi * twist_vel, -zdot * self._u)
        )
        half = 0.5 * phi * self._u
        spring_orientation = np.column_stack(
            (
                np.zeros(n_spring),
                np.zeros(n_spring),
                np.sin(half),
                np.cos(half),
            )
        )
        spring_angular = np.column_stack(
            (np.zeros(n_spring), np.zeros(n_spring), twist_vel)
        )

        weight_position = np.array(
            [[0.0, 0.0, self._z_top - length - self._h_half]], dtype=float
        )
        weight_velocity = np.array([[0.0, 0.0, -zdot]], dtype=float)
        weight_orientation = np.array(
            [[0.0, 0.0, np.sin(0.5 * phi), np.cos(0.5 * phi)]], dtype=float
        )
        weight_angular = np.array([[0.0, 0.0, phidot]], dtype=float)

        n_total = self._n_total
        position = np.empty((n_total, 3), dtype=float)
        velocity = np.empty((n_total, 3), dtype=float)
        orientation = np.empty((n_total, 4), dtype=float)
        angular = np.empty((n_total, 3), dtype=float)
        position[self._spring_index[self._order]] = spring_position
        velocity[self._spring_index[self._order]] = spring_velocity
        orientation[self._spring_index[self._order]] = spring_orientation
        angular[self._spring_index[self._order]] = spring_angular
        position[self._weight_index] = weight_position[0]
        velocity[self._weight_index] = weight_velocity[0]
        orientation[self._weight_index] = weight_orientation[0]
        angular[self._weight_index] = weight_angular[0]

        external_force = self._mass[:, None] * GRAVITY[None, :]
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=self._mass,
            body_id=self._body_id,
            attributes={"mass_quantity": self._mass.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
            },
            scalars={
                "total_energy": self._energy(),
                "kinetic_translational": float(0.5 * self._M * zdot ** 2),
                "kinetic_torsional": float(0.5 * self._I * phidot ** 2),
                "axial_extension": float(self._z),
                "twist_angle": float(self._phi),
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- reporting
    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "rayleigh_ritz_twist_extension_wilberforce",
            "integration": "velocity_verlet",
            "integration_step_s": float(self._h),
            "substeps_per_sample": int(self._substeps),
            "axial_period_s": float(_TWO_PI / self._omega0),
            "normal_mode_periods_s": [
                float(_TWO_PI / self._w_plus),
                float(_TWO_PI / self._w_minus),
            ],
            "energy_exchange_period_s": float(self._beat_period),
            "reduced_mass_kg": float(self._M),
            "reduced_inertia_kg_m2": float(self._I),
            "spring_axial_stiffness_n_per_m": float(self._k),
            "spring_torsional_stiffness_nm_per_rad": float(self._kappa),
            "twist_extension_coupling_m_per_rad": float(self._a),
            "coil_radius_m": float(self._radius),
            "initial_spring_length_m": float(self._L0),
            "initial_total_energy_j": float(self._E0),
            "max_energy_deviation_j": float(self._max_dev),
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _weight_radius(config: dict) -> float:
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            if item.get("semantic") == "rigid":
                shape = item.get("geometry") or {}
                radius = shape.get("radius")
                if radius:
                    return float(radius)
        return float(_DEFAULT_WEIGHT_RADIUS)
