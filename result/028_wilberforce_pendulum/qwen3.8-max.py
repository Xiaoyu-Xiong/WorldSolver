"""Wilberforce pendulum solver.

Reduced-order physical model of the classic Wilberforce pendulum: a stretched
helical spring carrying a suspended rigid weight.  The spring's weak
bending-twist coupling is represented by the standard coupled translational /
torsional oscillator

    m u''  = -k u   - eps phi - c_v u'
    I phi'' = -kap phi - eps u   - c_p phi'

integrated with fixed-step RK4.  The visible 128 helical spring nodes are
reconstructed from the physical degrees of freedom (uniform axial stretch
distribution u*s and twist distribution phi*s along the spring, weight rigidly
attached at the bottom), so the rendered geometry follows directly from the
solved dynamics, including the vertical/torsional energy exchange (beats) and
slow energy drift from light damping.

Exported StateFrame fields:
    state.position / velocity / mass            ([N,3], [N,3], [N])
    orientation / angular_velocity              ([N,4], [N,3])
    body_id                                     ([N])
    attributes.mass_quantity                    ([N,2] per material)
    fields.cumulative_mass_source               ([2], zeros - mass conserved)
    fields.external_force                       ([N,3], gravity)
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        n = pos0.shape[0]
        self.n = n
        self.n_spring = n - 1                       # spring nodes, then weight
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id).copy()

        sp = pos0[: self.n_spring]
        self.r0 = np.hypot(sp[:, 0], sp[:, 1])      # helix radius per node
        self.theta0 = np.arctan2(sp[:, 1], sp[:, 0])
        self.z0 = sp[:, 2].copy()
        self.w_z0 = float(pos0[self.n_spring, 2])   # weight centre height
        # material-coordinate fraction along the spring (0 at ceiling, 1 at weight)
        self.s_frac = np.linspace(0.0, 1.0, self.n_spring)

        meta = scene.metadata() if hasattr(scene, "metadata") else {}
        g = np.asarray(meta.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = g

        # ---- physical parameters ----------------------------------------
        m_spring = float(self.mass[: self.n_spring].sum())     # ~0.12 kg
        m_weight = float(self.mass[self.n_spring])             # ~1.0 kg
        r_weight = 0.34                                        # cylinder radius
        # effective translational mass (spring contributes ~1/3)
        self.m = m_weight + m_spring / 3.0
        # effective torsional inertia about the spring axis
        I_weight = 0.5 * m_weight * r_weight**2
        r_spring = float(np.mean(self.r0))
        I_spring = 0.5 * (m_spring / 3.0) * r_spring**2
        self.I = I_weight + I_spring

        omega0 = 5.1                      # rad/s, ~1.23 s oscillation period
        self.k = self.m * omega0**2       # axial stiffness
        self.kap = self.I * omega0**2     # torsional stiffness
        # coupling chosen for an ~8 s beat period (visible energy exchange)
        self.eps = 1.0
        # light viscous damping (Q ~ 150) -> slow total-energy drift
        self.c_v = self.k / (150.0 * omega0)
        self.c_p = self.kap / (150.0 * omega0)

        # ---- initial conditions from the scene ---------------------------
        vel0 = np.asarray(frame0.velocity, dtype=float)
        u0 = 0.0                                  # start at static equilibrium
        udot0 = float(vel0[self.n_spring, 2])     # prescribed vertical velocity
        phi0 = 0.0
        phidot0 = 0.0                             # torsion builds via coupling
        self.state = np.array([u0, udot0, phi0, phidot0], dtype=float)
        self.t = 0.0
        self.dt = 5.0e-4

        self._ext_force = self.mass[:, None] * self.gravity[None, :]
        self._mass_quantity = np.zeros((n, 2), dtype=float)
        self._mass_quantity[: self.n_spring, 0] = self.mass[: self.n_spring]
        self._mass_quantity[self.n_spring, 1] = self.mass[self.n_spring]
        self._cum_mass_source = np.zeros(2, dtype=float)

    # ------------------------------------------------------------- dynamics
    def _rhs(self, y: np.ndarray) -> np.ndarray:
        u, v, phi, w = y
        dv = (-self.k * u - self.eps * phi - self.c_v * v) / self.m
        dw = (-self.kap * phi - self.eps * u - self.c_p * w) / self.I
        return np.array([v, dv, w, dw])

    def _integrate(self, t_target: float) -> None:
        # fixed-step RK4 up to t_target (monotonic)
        remaining = t_target - self.t
        if remaining <= 0.0:
            return
        n_steps = int(np.ceil(remaining / self.dt))
        h = remaining / n_steps
        y = self.state
        for _ in range(n_steps):
            k1 = self._rhs(y)
            k2 = self._rhs(y + 0.5 * h * k1)
            k3 = self._rhs(y + 0.5 * h * k2)
            k4 = self._rhs(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.state = y
        self.t = t_target

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float):
        self._integrate(float(time))
        u, udot, phi, phidot = self.state
        s = self.s_frac

        # --- spring node kinematics ---------------------------------------
        theta = self.theta0 + phi * s
        ct, st = np.cos(theta), np.sin(theta)
        position = np.zeros((self.n, 3), dtype=float)
        position[: self.n_spring, 0] = self.r0 * ct
        position[: self.n_spring, 1] = self.r0 * st
        position[: self.n_spring, 2] = self.z0 + u * s
        position[self.n_spring] = (0.0, 0.0, self.w_z0 + u)

        thdot = phidot * s
        velocity = np.zeros((self.n, 3), dtype=float)
        velocity[: self.n_spring, 0] = -self.r0 * thdot * st
        velocity[: self.n_spring, 1] = self.r0 * thdot * ct
        velocity[: self.n_spring, 2] = udot * s
        velocity[self.n_spring] = (0.0, 0.0, udot)

        # orientation: rotation about z by phi*s (quaternion xyzw)
        half = 0.5 * np.concatenate((phi * s, [phi]))
        orientation = np.zeros((self.n, 4), dtype=float)
        orientation[:, 2] = np.sin(half)
        orientation[:, 3] = np.cos(half)

        angular_velocity = np.zeros((self.n, 3), dtype=float)
        angular_velocity[: self.n_spring, 2] = phidot * s
        angular_velocity[self.n_spring, 2] = phidot

        frame = StateFrame(
            float(self.t),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self._mass_quantity.copy()},
            fields={
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "external_force": self._ext_force.copy(),
            },
            scalars={
                "vertical_displacement": float(u),
                "twist_angle": float(phi),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        u, udot, phi, phidot = self.state
        e_trans = 0.5 * self.m * udot**2 + 0.5 * self.k * u**2
        e_tors = 0.5 * self.I * phidot**2 + 0.5 * self.kap * phi**2
        return {
            "time": self.t,
            "u": float(u),
            "phi": float(phi),
            "translational_energy": float(e_trans),
            "torsional_energy": float(e_tors),
            "total_energy": float(e_trans + e_tors),
        }
