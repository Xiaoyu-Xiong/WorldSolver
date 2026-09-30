"""Solver for scene 028_wilberforce_pendulum.

Wilberforce pendulum: a helical spring hangs from a fixed attachment with a
cylindrical weight at its lower end.  Vertical translation of the weight and
torsion about the spring axis form two nearly degenerate oscillators that are
weakly coupled through the spring's bending-twist coupling, so energy slowly
sloshes between the translational and the torsional mode.

Physical model
--------------
The dynamics is reduced to the two generalized coordinates of the ideal
Wilberforce pendulum (the symmetry-adapted modes of the helix):

    z     : downward displacement of the weight from its initial position
            (positive z lengthens the spring),
    theta : twist angle of the weight (and of the spring's lower end)
            relative to the fixed upper attachment.

The Lagrangian equations of motion are

    m_eff * zddot = m_eff * g - k_s * (h - h_nat) - gamma * theta
    I_eff * tddot = -k_t * theta - gamma * (h - h_nat)

with h = h0 + z the instantaneous spring length, h_nat the natural length,
k_s the axial (translational) stiffness, k_t the torsional stiffness and
gamma the bending-twist coupling coefficient of the coil.  The symmetric
coupling term  gamma*(h-h_nat)*theta  in the potential is the classical
Wilberforce coupling and produces the slow beat between the two modes.

The reduced model is integrated with a fixed-step 4th-order Runge-Kutta
scheme (deterministic, near energy conserving).  The full spring centerline
is then reconstructed from the instantaneous (z, theta): the helix pitch
follows the spring length and the material twist is distributed uniformly
along the coil, so node positions/velocities are exact kinematic functions
of the integrated physical state -- no keyframes or scripted animation.

Trajectory export follows the StateFrame contract: state.mass/position/
velocity, attributes.mass_quantity, fields.cumulative_mass_source and
fields.external_force (gravity on every sample).
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - import guard for direct-file loading
    from worldsolver.contracts import StateFrame
except ImportError:  # pragma: no cover
    StateFrame = object


class BenchmarkSolver:
    """Time-stepping Wilberforce pendulum solver."""

    # --- reduced model parameters (SI units) -----------------------------
    GRAVITY = 9.81
    # Frozen geometry from the scaffold initial state.
    TOP_Z = 2.55          # spring top (attached to the fixed upper mount)
    H0 = 1.56             # initial spring length (top 2.55 -> bottom 0.99)
    HELIX_RADIUS = 0.16
    TURNS0 = 10.5
    WEIGHT_HALF_HEIGHT = 0.21
    WEIGHT_RADIUS = 0.34

    # Constitutive parameters of the spring steel coil.
    SPRING_MASS = 0.12
    WEIGHT_MASS = 1.0
    K_AXIAL = 60.0        # N/m  translational stiffness
    K_TORSION = 3.40      # N*m/rad torsional stiffness
    GAMMA = 0.34          # N/m == N*m/rad  bending-twist coupling
    # Natural spring length: the initial configuration is unstretched, so
    # gravity produces the static working point and the visible oscillation.
    H_NAT = H0

    SUBSTEP = 1.0e-3      # fixed RK4 integration step [s]

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        self.seed = int(seed)

        self.n_total = int(np.asarray(frame0.position).shape[0])
        self.n_spring = self.n_total - 1  # last sample is the weight
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id).copy()
        self.position0 = np.asarray(frame0.position, dtype=float).copy()
        self.velocity0 = np.asarray(frame0.velocity, dtype=float).copy()

        # Effective inertia of the moving system.
        # Translation: weight + roughly a third of the spring mass.
        self.m_eff = self.WEIGHT_MASS + self.SPRING_MASS / 3.0
        # Rotation about the axis: weight (solid cylinder) + coil inertia.
        i_weight = 0.5 * self.WEIGHT_MASS * self.WEIGHT_RADIUS ** 2
        i_coil = (self.SPRING_MASS / 3.0) * self.HELIX_RADIUS ** 2
        self.i_eff = i_weight + i_coil

        # Helix parametrisation: sample i sits at fraction s_i along the coil.
        self.s = np.linspace(0.0, 1.0, self.n_spring)
        self.phase0 = 2.0 * math.pi * self.TURNS0 * self.s

        # External force field: gravity only (attachment reaction and spring
        # internal forces are excluded from fields.external_force).
        self.external_force = np.zeros((self.n_total, 3), dtype=float)
        self.external_force[:, 2] = -self.GRAVITY * self.mass

        # Initial conditions taken from the scaffold initial state: the coil
        # is already moving downward and spinning about the axis.
        zdot0 = float(-np.mean(self.velocity0[: self.n_spring, 2]))
        thetadot0 = 0.0
        ang = np.asarray(frame0.angular_velocity, dtype=float)
        if ang is not None and ang.ndim == 2:
            thetadot0 = float(np.mean(ang[: self.n_spring, 2]))
        # Reduced state: [z, theta, zdot, thetadot]
        self._state0 = np.array([0.0, 0.0, zdot0, thetadot0], dtype=float)
        self._state = self._state0.copy()
        self._time = 0.0
        self._last_frame_time = -1.0
        self._energy_report: dict[str, float] = {}
        self._initialized = True

    # -------------------------------------------------------------- dynamics
    def _deriv(self, state: np.ndarray) -> np.ndarray:
        _z, theta, zdot, thetadot = state
        stretch = (self.H0 + _z) - self.H_NAT  # spring extension
        # Downward-positive translation (gravity drives the stretch).
        zddot = (
            self.m_eff * self.GRAVITY
            - self.K_AXIAL * stretch
            - self.GAMMA * theta
        ) / self.m_eff
        thetaddot = (
            -self.K_TORSION * theta - self.GAMMA * stretch
        ) / self.i_eff
        return np.array([zdot, thetadot, zddot, thetaddot], dtype=float)

    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._deriv(state)
        k2 = self._deriv(state + 0.5 * dt * k1)
        k3 = self._deriv(state + 0.5 * dt * k2)
        k4 = self._deriv(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate_to(self, time: float) -> None:
        if time < self._time - 1e-12:
            # Restart from the initial state (deterministic replay).
            self._state = self._state0.copy()
            self._time = 0.0
        target = float(time)
        while target - self._time > 1e-12:
            dt = min(self.SUBSTEP, target - self._time)
            self._state = self._rk4_step(self._state, dt)
            self._time += dt
            if not np.all(np.isfinite(self._state)):
                raise FloatingPointError("non-finite reduced state")

    # ------------------------------------------------------------ kinematics
    def _sample(self, time: float) -> StateFrame:
        z, theta, zdot, thetadot = self._state
        length = self.H0 + z  # current spring length

        # Helix: uniform twist distribution along the coil.  Sample i is
        # rotated by s_i * theta relative to the fixed top and sits at
        # height TOP_Z - s_i * length.
        phase = self.phase0 + self.s * theta
        spring_pos = np.empty((self.n_spring, 3), dtype=float)
        spring_pos[:, 0] = self.HELIX_RADIUS * np.cos(phase)
        spring_pos[:, 1] = self.HELIX_RADIUS * np.sin(phase)
        spring_pos[:, 2] = self.TOP_Z - self.s * length

        spring_vel = np.empty((self.n_spring, 3), dtype=float)
        phase_rate = self.s * thetadot
        spring_vel[:, 0] = -self.HELIX_RADIUS * np.sin(phase) * phase_rate
        spring_vel[:, 1] = self.HELIX_RADIUS * np.cos(phase) * phase_rate
        spring_vel[:, 2] = -self.s * zdot

        spring_angvel = np.zeros((self.n_spring, 3), dtype=float)
        spring_angvel[:, 2] = phase_rate

        # Weight: rigidly attached to the spring's lower end.
        weight_pos = np.array(
            [[0.0, 0.0, self.TOP_Z - length - self.WEIGHT_HALF_HEIGHT]],
            dtype=float,
        )
        weight_vel = np.array([[0.0, 0.0, -zdot]], dtype=float)
        weight_angvel = np.array([[0.0, 0.0, thetadot]], dtype=float)

        position = np.vstack((spring_pos, weight_pos))
        velocity = np.vstack((spring_vel, weight_vel))
        angular_velocity = np.vstack((spring_angvel, weight_angvel))

        orientation = np.zeros((self.n_total, 4), dtype=float)
        orientation[:, 3] = 1.0
        half = 0.5 * theta
        orientation[self.n_total - 1, 2] = math.sin(half)
        orientation[self.n_total - 1, 3] = math.cos(half)

        # Energy bookkeeping (reported via diagnostics).
        stretch = length - self.H_NAT
        ke_trans = 0.5 * self.m_eff * zdot ** 2
        ke_tors = 0.5 * self.i_eff * thetadot ** 2
        pe = (
            0.5 * self.K_AXIAL * stretch ** 2
            + 0.5 * self.K_TORSION * theta ** 2
            + self.GAMMA * stretch * theta
            - self.m_eff * self.GRAVITY * z
        )
        self._energy_report = {
            "time": float(time),
            "translational_kinetic": float(ke_trans),
            "torsional_kinetic": float(ke_tors),
            "potential": float(pe),
            "total": float(ke_trans + ke_tors + pe),
        }

        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.external_force.copy(),
            },
            scalars={
                "spring_length": float(length),
                "twist_angle": float(theta),
                "translational_kinetic_energy": float(ke_trans),
                "torsional_kinetic_energy": float(ke_tors),
                "total_energy": float(ke_trans + ke_tors + pe),
            },
        )

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("solver.initialize() was not called")
        target = float(time)
        if target < 0.0 or not math.isfinite(target):
            raise ValueError("advance_to requires a finite non-negative time")
        self._integrate_to(target)
        frame = self._sample(target)
        self._last_frame_time = target
        return frame

    def diagnostics(self) -> dict[str, object]:
        report = {
            "model": "wilberforce_reduced_lagrangian_rk4",
            "m_eff": float(self.m_eff),
            "i_eff": float(self.i_eff),
            "k_axial": float(self.K_AXIAL),
            "k_torsion": float(self.K_TORSION),
            "gamma_coupling": float(self.GAMMA),
            "omega_translation": float(math.sqrt(self.K_AXIAL / self.m_eff)),
            "omega_torsion": float(math.sqrt(self.K_TORSION / self.i_eff)),
            "energy_exchange_period": float(
                2.0
                * math.pi
                * math.sqrt(self.m_eff * self.i_eff)
                / self.GAMMA
            ),
        }
        report.update(self._energy_report)
        return report
