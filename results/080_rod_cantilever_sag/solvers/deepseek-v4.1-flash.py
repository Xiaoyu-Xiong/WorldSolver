"""Elastic rod cantilever sag: physical solver implementation.

Physical model
--------------
The frozen scene is a single homogeneous, naturally straight, inextensible
elastic rod with a circular cross-section, clamped at its root so that the
root position *and* the root tangent are horizontal, with a free tip and
uniform downward gravity.  The requested simulation mode is
``quasi_static_equilibria``: the rod must track the sequence of static
elastica equilibria produced by the gravity ramp.

The rod is discretised with the exact inextensible kinematics of a chain of
rigid links: the configuration is parametrised by the material tangent angles
``theta_k`` of the ``N-1`` links (planar motion, since gravity and the initial
configuration are contained in the x-z plane).  Positions are recovered by
cumulative summation, so inextensibility is satisfied *exactly* for every
sample, at every time.

The dynamics are the physical generalised-coordinate dynamics of that link
chain with a bending (elastica) spring between adjacent links and a lumped
mass distribution:

    I_j * d(omega_j)/dt = -dE/d(theta_j) - c * I_j * omega_j
    d(theta_j)/dt      = omega_j

with ``E`` the sum of the discrete bending energy and the gravitational
potential, and ``I_j`` the moment of inertia of the material downstream of
joint ``j`` about that joint.  The bending operator is the standard
nearest-neighbour (tridiagonal) discrete curvature operator; it is integrated
implicitly with backward Euler, which is unconditionally stable, while the
smooth gravitational generalised force is treated explicitly.  Strong
physical damping (``c``) makes the scheme relax quickly, so the rod follows
the quasi-static equilibrium branch as the gravity load is ramped, exactly as
required by the task.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --- physical / numerical constants -------------------------------------
# Bending modulus of the rod material.  The frozen configuration specifies
# geometry, density and gravity but not the Young's modulus; an elastomeric
# modulus is used so that the rod produces a pronounced but finite cantilever
# sag inside the frozen camera framing.
_BENDING_MODULUS_PA = 1.2e8
_DT = 5.0e-4          # physical time step of the damped dynamics (s)
_DAMPING = 20.0       # physical mass-proportional damping rate (1/s)
_MAX_SUBSTEPS_PER_SAMPLE = 20000


def _thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system (Thomas algorithm)."""
    n = diag.shape[0]
    c = np.empty(n - 1, dtype=float)
    d = np.empty(n, dtype=float)
    denom = diag[0]
    d[0] = rhs[0] / denom
    if n > 1:
        c[0] = upper[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
        if i < n - 1:
            c[i] = upper[i] / denom
    x = np.empty(n, dtype=float)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


class BenchmarkSolver:
    """Gravity-loaded clamped-free elastica advanced by damped dynamics."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)
        config = scene.config

        rod = config["rod"]
        self._length = float(rod["length"])
        self._radius = float(rod["radius"])
        self._density = float(rod["density"])
        self._gravity = np.asarray(config["gravity"], dtype=float)
        self._gravity_mag = float(np.linalg.norm(self._gravity))

        schedule = config.get("load_schedule", {})
        self._ramp_start = float(schedule.get("start_time", 0.0))
        self._ramp_full = float(schedule.get("full_load_time", self._ramp_start))

        frame0 = scene.initial_state()
        position0 = np.asarray(frame0.position, dtype=float)
        mass = np.asarray(frame0.mass, dtype=float)
        self._n = int(position0.shape[0])
        self._n_links = self._n - 1
        self._l0 = self._length / self._n_links

        self._mass = mass
        self._total_mass = float(np.sum(mass))
        self._area = np.pi * self._radius**2
        self._second_moment = 0.25 * np.pi * self._radius**4
        self._ei = _BENDING_MODULUS_PA * self._second_moment
        self._bend = self._ei / self._l0

        # clamp: root node position and horizontal root tangent are frozen.
        self._root = position0[0].copy()
        self._free_angles = self._n - 2  # theta_1 .. theta_{N-2}; theta_0 = 0

        # moment of inertia of the material downstream of each joint,
        # about that joint (lumped point masses on the link chain).
        idx = np.arange(self._n)
        lever = (idx[:, None] - idx[None, :]).astype(float)  # lever[i, j] = i - j
        downstream = lever > 0.0  # node i lies beyond joint j
        contrib = downstream * mass[:, None] * (lever * self._l0) ** 2
        inertia = np.sum(contrib, axis=0)  # I_j for joint j (j=0..N-2)
        self._inertia = inertia[1 : self._n - 1].copy()  # joints 1..N-2

        # downstream mass sums: M_gt[j] = sum_{i>j} m_i
        self._mass_gt = np.cumsum(mass[::-1])[::-1]
        self._mass_gt = np.concatenate([self._mass_gt[1:], [0.0]])  # index j -> sum_{i>j}

        # tridiagonal bending operator K on the free angles (size N-2)
        m_free = self._free_angles
        self._k_diag = np.full(m_free, 2.0)
        self._k_diag[-1] = 1.0
        self._k_off = -np.ones(m_free - 1)

        # constant part of the implicit system matrix
        inv_dt = 1.0 / _DT
        self._a_diag_const = self._inertia * (inv_dt + _DAMPING) + _DT * self._bend * self._k_diag
        self._a_off = _DT * self._bend * self._k_off
        self._inv_dt = inv_dt

        # state: tangent angles (theta_0 = 0 frozen) and angular rates
        self._theta = self._angles_from_positions(position0)
        self._omega = np.zeros(m_free)
        self._time = 0.0
        self._steps_done = 0
        self._position_cache = position0.copy()
        self._velocity_cache = np.zeros_like(position0)

    # -- kinematics ------------------------------------------------------
    def _angles_from_positions(self, position: np.ndarray) -> np.ndarray:
        seg = np.diff(position, axis=0)
        angles = np.arctan2(seg[:, 2], seg[:, 0])
        return angles[1 : self._n - 1].copy()

    def _positions_from_angles(self, theta: np.ndarray) -> np.ndarray:
        angles = np.empty(self._n_links, dtype=float)
        angles[0] = 0.0  # clamped horizontal root tangent
        angles[1:] = theta
        position = np.empty((self._n, 3), dtype=float)
        position[0] = self._root
        position[1:, 0] = self._root[0] + np.cumsum(self._l0 * np.cos(angles))
        position[1:, 1] = self._root[1]
        position[1:, 2] = self._root[2] + np.cumsum(self._l0 * np.sin(angles))
        return position

    def _velocities_from_rates(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        angles = np.empty(self._n_links, dtype=float)
        rates = np.empty(self._n_links, dtype=float)
        angles[0] = 0.0
        rates[0] = 0.0
        angles[1:] = theta
        rates[1:] = omega
        velocity = np.zeros((self._n, 3), dtype=float)
        dx = -self._l0 * rates * np.sin(angles)
        dz = self._l0 * rates * np.cos(angles)
        velocity[1:, 0] = np.cumsum(dx)
        velocity[1:, 2] = np.cumsum(dz)
        return velocity

    # -- load schedule ---------------------------------------------------
    def _load_scale(self, time: float) -> float:
        if time <= self._ramp_start:
            return 0.0
        if time >= self._ramp_full:
            return 1.0
        span = self._ramp_full - self._ramp_start
        return float((time - self._ramp_start) / span)

    # -- physics ---------------------------------------------------------
    def _bending_operator(self, theta: np.ndarray) -> np.ndarray:
        out = self._k_diag * theta
        out[:-1] += self._k_off * theta[1:]
        out[1:] += self._k_off * theta[:-1]
        return out

    def _gravity_torque(self, theta: np.ndarray, load: float) -> np.ndarray:
        mag = self._gravity_mag * load
        mass_gt = self._mass_gt[1 : self._n - 1]
        return -mag * self._l0 * mass_gt * np.cos(theta)

    def _step(self, load: float) -> None:
        theta = self._theta
        omega = self._omega
        rhs = (
            self._inertia * omega * self._inv_dt
            - self._bend * self._bending_operator(theta)
            + self._gravity_torque(theta, load)
        )
        omega_new = _thomas(self._a_off, self._a_diag_const, self._a_off, rhs)
        self._omega = omega_new
        self._theta = theta + _DT * omega_new
        self._steps_done += 1

    def _advance(self, target_time: float) -> None:
        if target_time <= self._time:
            return
        n_steps = int(round((target_time - self._time) / _DT))
        if n_steps <= 0:
            n_steps = 1
        for _ in range(n_steps):
            load = self._load_scale(self._time)
            self._step(load)
            self._time += _DT
        self._position_cache = self._positions_from_angles(self._theta)
        self._velocity_cache = self._velocities_from_rates(self._theta, self._omega)
        self._time = target_time

    # -- StateFrame construction -----------------------------------------
    def _frame(self, time: float) -> StateFrame:
        position = self._position_cache
        velocity = self._velocity_cache
        n = self._n

        load = self._load_scale(time)
        gravity_force = self._total_mass * load * self._gravity  # [3]
        com = np.sum(self._mass[:, None] * position, axis=0) / self._total_mass
        root_force = -gravity_force  # quasi-static clamp traction

        external_force = np.zeros((1, 2, 3), dtype=float)
        external_force[0, 0] = gravity_force
        external_force[0, 1] = root_force
        force_position = np.zeros((1, 2, 3), dtype=float)
        force_position[0, 0] = com
        force_position[0, 1] = self._root
        external_couple = np.zeros((1, 2, 3), dtype=float)

        root_tangent = np.array([1.0, 0.0, 0.0])
        constraint = np.concatenate([self._root, root_tangent])

        arc_length = np.linspace(0.0, self._length, n)
        clamped = np.zeros(n)
        clamped[0] = 1.0

        weight = self._total_mass * self._gravity_mag

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self._mass.copy(),
            attributes={
                "arc_length": arc_length,
                "is_clamped_root": clamped,
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "budget_external_force": external_force,
                "budget_force_position": force_position,
                "budget_external_couple": external_couple,
                "constraint_value": constraint,
                "constraint_target": constraint.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": float(self._length),
                "reference_force": float(weight),
                "reference_torque": float(weight * self._length),
            },
        )
        frame.validate()
        return frame

    # -- Solver contract --------------------------------------------------
    def advance_to(self, time: float):
        self._advance(float(time))
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "inextensible-link-chain_damped_dynamics_implicit_bending",
            "node_count": int(self._n),
            "time_step": float(_DT),
            "damping_rate": float(_DAMPING),
            "bending_modulus_pa": float(_BENDING_MODULUS_PA),
            "bending_stiffness_ei": float(self._ei),
            "substeps": int(self._steps_done),
            "final_time": float(self._time),
            "finite_state": bool(np.isfinite(self._theta).all()),
        }
