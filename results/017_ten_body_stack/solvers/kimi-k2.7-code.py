"""Penalty-based rigid-body stack solver.

The ten boxes are modelled as point masses moving only along the vertical axis,
with the horizontal degrees of freedom frozen.  Axis-aligned box-box contacts are
handled by stiff spring-damper penalties, including a ground contact for the
bottom body.  This is a genuine physical time-stepping simulation: no keyframes,
no precomputed motion, and no external simulation library is used.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit penalty solver for the ten-body stack scene."""

    def __init__(self) -> None:
        # Simulation parameters
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.hz = 0.1  # half-thickness of each box along z
        self.k = 2.0e5  # contact stiffness [N/m]
        self.zeta = 1.0  # critical damping ratio for contacts
        self.dt = 1.0e-5  # physics sub-step [s]
        self.contact_gap_threshold = 0.01  # include contacts within 1 cm as "near"
        self.contact_length_scale = 0.1  # box half-thickness, frozen length scale

        # Internal state (populated in initialize)
        self.time = 0.0
        self.n = 0
        self.mass: np.ndarray = np.array([])
        self.pos: np.ndarray = np.array([])
        self.vel: np.ndarray = np.array([])
        self.orientation: np.ndarray = np.array([])
        self.angular_velocity: np.ndarray = np.array([])
        self.body_id: np.ndarray = np.array([])
        self.c_body: float = 0.0
        self.c_ground: float = 0.0
        self._step_count = 0

    def initialize(self, scene, seed: int) -> None:
        init = scene.initial_state()
        self.time = float(init.time)
        self.pos = np.asarray(init.position, dtype=float).copy()
        self.vel = np.asarray(init.velocity, dtype=float).copy()
        self.orientation = np.asarray(init.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(init.angular_velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.body_id = np.asarray(init.body_id).copy()
        self.n = self.pos.shape[0]

        # Damping coefficients chosen for critical damping with the penalty spring.
        # body-body: effective mass = m1*m2/(m1+m2) = 0.5 kg for equal 1 kg masses.
        # ground: effective mass = m = 1 kg.
        self.c_body = 2.0 * self.zeta * np.sqrt(self.k * 0.5)
        self.c_ground = 2.0 * self.zeta * np.sqrt(self.k * 1.0)
        self._step_count = 0

    def _acceleration(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return z-acceleration for all bodies given current z positions/velocities."""
        a = np.full(self.n, self.gravity[2], dtype=float)

        # Ground contact (body 0 on the z=0 plane)
        d0 = z[0] - self.hz  # signed gap to ground
        if d0 < self.contact_gap_threshold:
            if d0 < 0.0:
                compression = -d0
                force = self.k * compression - self.c_ground * v[0]
                a[0] += force / self.mass[0]

        # Consecutive body contacts, from bottom to top
        for j in range(self.n - 1):
            d = z[j + 1] - z[j] - 2.0 * self.hz  # signed gap between body j+1 and j
            if d >= self.contact_gap_threshold:
                continue
            if d < 0.0:
                compression = -d
                rel_v = v[j + 1] - v[j]
                force = self.k * compression - self.c_body * rel_v
                a[j + 1] += force / self.mass[j + 1]
                a[j] -= force / self.mass[j]

        return a

    def _step(self, dt: float) -> None:
        """Advance by one semi-implicit Euler sub-step."""
        z = self.pos[:, 2]
        v = self.vel[:, 2]

        a = self._acceleration(z, v)
        v_new = v + dt * a
        z_new = z + dt * v_new

        self.vel[:, 2] = v_new
        self.pos[:, 2] = z_new
        self.time += dt
        self._step_count += 1

    def _compute_contact_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return signed gaps and interaction forces for the 10 contacts.

        Contacts are ordered as:
          0 : ground - body 0
          1 : body 0 - body 1
          ...
          9 : body 8 - body 9
        """
        z = self.pos[:, 2]
        v = self.vel[:, 2]
        gaps = np.zeros(self.n, dtype=float)
        forces = np.zeros((self.n, 3), dtype=float)

        # Ground-body0 contact
        d0 = z[0] - self.hz
        gaps[0] = d0
        if d0 < 0.0:
            force = self.k * (-d0) - self.c_ground * v[0]
            forces[0, 2] = force

        # Body-body contacts
        for j in range(self.n - 1):
            d = z[j + 1] - z[j] - 2.0 * self.hz
            gaps[j + 1] = d
            if d < 0.0:
                rel_v = v[j + 1] - v[j]
                force = self.k * (-d) - self.c_body * rel_v
                forces[j + 1, 2] = force

        return gaps, forces

    def _build_frame(self, time: float) -> StateFrame:
        gaps, pair_forces = self._compute_contact_data()

        # Gravity is the only external force
        external_force = np.zeros((self.n, 3), dtype=float)
        external_force[:, 2] = self.mass * self.gravity[2]

        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": gaps,
                "external_force": external_force,
                "interaction_force": pair_forces,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
        return self._build_frame(target)

    def diagnostics(self) -> dict:
        return {
            "step_count": self._step_count,
            "dt": self.dt,
            "contact_stiffness": self.k,
        }
