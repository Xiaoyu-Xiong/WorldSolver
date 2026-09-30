"""Solver for the bolt-spins-through-nut scene.

The bolt and nut are modelled as a screw pair.  The nut is held fixed by the
floor, while the bolt rotates about and translates along the shared vertical
axis according to a left-hand screw relation.  A motor torque drives the
rotation, gravity assists the axial advance, and thread contact supplies the
constraint force required by the screw.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene

        # Frozen configuration
        self.gravity = np.asarray(
            scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.g = -self.gravity[2]  # positive scalar gravity
        contact_cfg = scene.config.get("contact", {})
        self.contact_length_scale = float(contact_cfg.get("length_scale", 0.02))

        # Initial scaffold state
        init = scene.initial_state()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.body_id = np.asarray(init.body_id).copy()
        self.position = np.asarray(init.position, dtype=float).copy()
        self.velocity = np.asarray(init.velocity, dtype=float).copy()
        self.orientation = np.asarray(init.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(init.angular_velocity, dtype=float).copy()

        # Locate the dynamic bodies by their body_id bindings.
        self.bolt_idx = int(np.argwhere(self.body_id == 0)[0, 0])
        self.nut_idx = int(np.argwhere(self.body_id == 1)[0, 0])
        self.m_bolt = float(self.mass[self.bolt_idx])
        self.m_nut = float(self.mass[self.nut_idx])

        # Screw geometry (left-hand thread: positive rotation about +z moves down).
        self.pitch = 0.08  # metres per revolution
        self.h = self.pitch / (2.0 * np.pi)
        self.r_thread = 0.05  # mean thread radius used for bolt inertia

        # Effective inertia along the screw coordinate.
        self.Iz_bolt = 0.5 * self.m_bolt * self.r_thread ** 2
        self.M_eff = self.Iz_bolt + self.m_bolt * self.h ** 2

        # Actuation and damping.  Gravity is the main driver; a small assisting
        # torque keeps the screw turning, while damping limits the speed so the
        # bolt stays engaged with the nut for the whole 8 s clip.
        self.tau_applied = 0.02  # N*m about +z
        self.damping = 0.068    # N*m*s/rad

        # Screw state.
        self.theta = 0.0
        self.omega = 0.0
        self.time = 0.0
        self.dt_sub = 1e-3

        # External force per body = gravity (torque is not a force).
        self.external_force = np.zeros((len(self.mass), 3), dtype=float)
        for i in range(len(self.mass)):
            self.external_force[i] = float(self.mass[i]) * self.gravity

    def _screw_z(self) -> float:
        """Current axial position of the bolt centre."""
        return float(self.position[self.bolt_idx, 2]) - self.h * self.theta

    def _screw_vz(self) -> float:
        """Current axial velocity of the bolt."""
        return -self.h * self.omega

    def _step(self, dt: float) -> None:
        """Advance the screw dynamics with semi-implicit Euler."""
        alpha = (
            self.tau_applied
            + self.m_bolt * self.g * self.h
            - self.damping * self.omega
        ) / self.M_eff
        self.omega += alpha * dt
        self.theta += self.omega * dt
        self.time += dt

    def _contact_force(self) -> float:
        """Axial contact force that the nut exerts on the bolt (positive upward)."""
        alpha = (
            self.tau_applied
            + self.m_bolt * self.g * self.h
            - self.damping * self.omega
        ) / self.M_eff
        # m*a_z = -m*g + F_c, a_z = -h*alpha
        return self.m_bolt * (self.g - self.h * alpha)

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.time:
            time = self.time

        n_sub = max(0, int(np.ceil((time - self.time) / self.dt_sub)))
        for _ in range(n_sub):
            remaining = time - self.time
            dt = min(self.dt_sub, remaining)
            self._step(dt)
            if self.time >= time - 1e-12:
                break

        # Assemble finite state arrays.
        pos = self.position.copy()
        pos[self.bolt_idx, 2] = self._screw_z()

        vel = np.zeros_like(self.velocity)
        vel[self.bolt_idx, 2] = self._screw_vz()

        # Bolt orientation: rotation about the z-axis by theta.
        qz = np.sin(self.theta / 2.0)
        qw = np.cos(self.theta / 2.0)
        norm = np.sqrt(qz * qz + qw * qw)
        qz /= norm
        qw /= norm
        orientation = np.zeros((len(self.mass), 4), dtype=float)
        orientation[self.bolt_idx] = [0.0, 0.0, qz, qw]
        orientation[self.nut_idx] = [0.0, 0.0, 0.0, 1.0]

        angular = np.zeros((len(self.mass), 3), dtype=float)
        angular[self.bolt_idx, 2] = self.omega

        F_c = self._contact_force()
        fields = {
            "contact_signed_distance": np.array([0.0], dtype=float),
            "external_force": self.external_force.copy(),
            "interaction_force": np.array([[0.0, 0.0, F_c]], dtype=float),
        }

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={},
            fields=fields,
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "theta": self.theta,
            "omega": self.omega,
            "contact_length_scale": self.contact_length_scale,
        }
