"""Rigid-sphere billiard impulse-chain solver (NumPy only).

Physical model
--------------
Six rigid spheres (an incoming cue ball plus a stationary five-ball line) rest
on a felt table.  The solver integrates Newton-Euler rigid-body dynamics with a
compliant (penalty) contact formulation:

* ball/table normal contact: linear Kelvin-Voigt spring-dashpot whose damping
  constant is derived from a target normal restitution;
* ball/table tangential contact: regularized Coulomb friction applied at the
  bottom contact point, which converts initial sliding into rolling;
* rolling resistance: torque ``-mu_r * F_n * R`` along the rolling axis, using
  the frozen ``rolling_friction`` coefficient;
* ball/ball normal contact: stiff spring-dashpot tuned to the restitution of
  polished billiard balls (smooth spheres carry no tangential contact force),
  which dissipates a little energy at every impact.

Time integration is symplectic (semi-implicit) Euler with a fixed substep small
enough to resolve the ~1.5 ms contact events, so the impulse travels down the
line one contact at a time and the far ball is launched and starts to roll.
Nothing is scripted: every position, velocity and force below comes from the
integrated equations of motion.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_MAX_SUBSTEP = 2.5e-5        # s, resolves the stiff penalty contact events
_BALL_STIFFNESS = 2.0e6      # N/m, sphere-sphere penalty stiffness
_GROUND_STIFFNESS = 2.0e6    # N/m, sphere-table penalty stiffness
_BALL_RESTITUTION = 0.94     # polished phenolic billiard balls
_GROUND_RESTITUTION = 0.35   # cloth absorbs vertical rebound quickly
_MU_SLIDE = 0.2              # sliding friction, ball on cloth
_SLIP_EPS = 2.0e-3           # m/s, Coulomb regularization velocity
_SPIN_EPS = 5.0e-2           # rad/s, rolling-resistance regularization


def _dashpot(stiffness: float, effective_mass: float, restitution: float) -> float:
    """Damping constant of a linear spring-dashpot with the given restitution."""
    value = float(min(max(restitution, 1.0e-4), 0.9999))
    log_e = math.log(value)
    return 2.0 * abs(log_e) * math.sqrt(
        stiffness * effective_mass / (math.pi * math.pi + log_e * log_e)
    )


class BenchmarkSolver:
    """Newton-Euler rigid-sphere solver with compliant contact."""

    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        state = scene.initial_state()

        self.position = np.array(state.position, dtype=float).reshape(-1, 3).copy()
        self.velocity = np.array(state.velocity, dtype=float).reshape(-1, 3).copy()
        self.n = int(self.position.shape[0])

        if state.angular_velocity is not None:
            self.omega = np.array(state.angular_velocity, dtype=float).reshape(-1, 3).copy()
        else:
            self.omega = np.zeros((self.n, 3))
        if state.mass is not None:
            self.mass = np.array(state.mass, dtype=float).reshape(-1).copy()
        else:
            self.mass = np.ones(self.n)
        if state.body_id is not None:
            self.body_id = np.array(state.body_id, dtype=np.int64).reshape(-1).copy()
        else:
            self.body_id = np.arange(self.n, dtype=np.int64)
        self.orientation = np.zeros((self.n, 4))
        self.orientation[:, 3] = 1.0

        self.gravity = np.array(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.mu_roll = float(config.get("rolling_friction", 0.015))

        # Frozen geometry: sphere radii and the table top plane.
        radii = np.full(self.n, 0.2)
        ground_top = 0.0
        ground_half = np.array([4.5, 2.0, 0.05])
        ground_center = np.zeros(3)
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            shape = item.get("geometry", {}) or {}
            transform = item.get("transform", {}) or {}
            translation = np.array(
                transform.get("translation", [0.0, 0.0, 0.0]), dtype=float
            ).reshape(3)
            if shape.get("type") == "sphere":
                binding = item.get("state_binding") or {}
                if binding.get("selector") == "body_id":
                    match = np.flatnonzero(self.body_id == int(binding.get("value", -1)))
                    if match.size:
                        radii[match[0]] = float(shape.get("radius", 0.2))
            elif shape.get("type") == "box" and item.get("motion") == "static":
                ground_half = np.array(
                    shape.get("half_extents", [4.5, 2.0, 0.05]), dtype=float
                ).reshape(3)
                ground_center = translation
                ground_top = float(translation[2] + ground_half[2])

        self.radius = radii
        self.ground_top = ground_top
        self.ground_center = ground_center
        self.ground_half = ground_half
        # Solid spheres: I = 2/5 m R^2 (isotropic, so a scalar suffices).
        self.inertia = 0.4 * self.mass * self.radius * self.radius

        # Contact bookkeeping: 6 ball/table gaps then the 5 chain pairs.
        self.pairs = np.array(
            [(i, i + 1) for i in range(self.n - 1)], dtype=np.int64
        ).reshape(-1, 2)
        self.n_pairs = int(self.pairs.shape[0])
        self.contact_length_scale = float(np.min(self.radius))

        m_i = self.mass[self.pairs[:, 0]]
        m_j = self.mass[self.pairs[:, 1]]
        self.pair_damping = np.array(
            [
                _dashpot(_BALL_STIFFNESS, float(a * b / (a + b)), _BALL_RESTITUTION)
                for a, b in zip(m_i, m_j)
            ]
        )
        self.ground_damping = np.array(
            [_dashpot(_GROUND_STIFFNESS, float(m), _GROUND_RESTITUTION) for m in self.mass]
        )

        self.time = 0.0
        self.steps = 0
        self.max_penetration = 0.0
        self.max_pair_force = 0.0

    # ------------------------------------------------------------------
    # force evaluation
    # ------------------------------------------------------------------
    def _evaluate(self):
        """Forces/torques plus the exported contact diagnostics at the current state."""
        n = self.n
        radius = self.radius
        # External: gravity plus the table reaction (the table is a static
        # boundary, not a bound rigid body, so its contact is external).
        external = self.mass[:, None] * self.gravity[None, :]
        torque = np.zeros((n, 3))

        # ---- ball / table contact -------------------------------------
        gap_ground = self.position[:, 2] - radius - self.ground_top
        inside = (
            np.abs(self.position[:, 0] - self.ground_center[0]) <= self.ground_half[0]
        ) & (np.abs(self.position[:, 1] - self.ground_center[1]) <= self.ground_half[1])
        touching = (gap_ground < 0.0) & inside
        if np.any(touching):
            idx = np.flatnonzero(touching)
            depth = -gap_ground[idx]
            approach = -self.velocity[idx, 2]
            f_n = _GROUND_STIFFNESS * depth + self.ground_damping[idx] * approach
            f_n = np.maximum(f_n, 0.0)  # contact never pulls
            external[idx, 2] += f_n

            # Contact point slip velocity: v + omega x (-R e_z).
            arm = np.zeros((idx.size, 3))
            arm[:, 2] = -radius[idx]
            slip = self.velocity[idx] + np.cross(self.omega[idx], arm)
            slip[:, 2] = 0.0
            slip_mag = np.linalg.norm(slip, axis=1)
            scale = _MU_SLIDE * f_n / np.maximum(slip_mag, _SLIP_EPS)
            f_t = -slip * scale[:, None]
            external[idx, :2] += f_t[:, :2]
            torque[idx] += np.cross(arm, f_t)

            # Rolling resistance about the rolling axis.
            spin = self.omega[idx].copy()
            spin[:, 2] = 0.0
            spin_mag = np.linalg.norm(spin, axis=1)
            roll_scale = self.mu_roll * f_n * radius[idx] / np.maximum(spin_mag, _SPIN_EPS)
            torque[idx] -= spin * roll_scale[:, None]

        # ---- ball / ball contact --------------------------------------
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        delta = self.position[j] - self.position[i]
        distance = np.linalg.norm(delta, axis=1)
        safe = np.maximum(distance, 1.0e-12)
        normal = delta / safe[:, None]
        gap_pair = distance - (radius[i] + radius[j])

        relative = np.einsum("ij,ij->i", self.velocity[j] - self.velocity[i], normal)
        magnitude = np.where(
            gap_pair < 0.0,
            _BALL_STIFFNESS * (-gap_pair) - self.pair_damping * relative,
            0.0,
        )
        magnitude = np.maximum(magnitude, 0.0)
        # Force on body j points along +normal; body i receives the opposite.
        f_pair = magnitude[:, None] * normal

        interaction = np.zeros((2 * self.n_pairs, 3))
        interaction[0::2] = f_pair
        interaction[1::2] = -f_pair

        contact = np.zeros((n, 3))
        np.add.at(contact, j, f_pair)
        np.add.at(contact, i, -f_pair)

        return external, torque, contact, interaction, gap_ground, gap_pair

    def _step(self, dt: float) -> None:
        external, torque, contact, _interaction, gap_ground, gap_pair = self._evaluate()
        self.velocity += dt * (external + contact) / self.mass[:, None]
        self.omega += dt * torque / self.inertia[:, None]
        self.position += dt * self.velocity

        # Quaternion (xyzw) kinematics: dq/dt = 0.5 * omega_quat * q.
        q = self.orientation
        qv, qw = q[:, :3], q[:, 3]
        dv = 0.5 * (qw[:, None] * self.omega + np.cross(self.omega, qv))
        dw = -0.5 * np.einsum("ij,ij->i", self.omega, qv)
        q[:, :3] += dt * dv
        q[:, 3] += dt * dw
        norm = np.linalg.norm(q, axis=1)
        self.orientation = q / np.maximum(norm, 1.0e-12)[:, None]

        self.steps += 1
        penetration = max(
            float(-np.min(gap_ground)) if gap_ground.size else 0.0,
            float(-np.min(gap_pair)) if gap_pair.size else 0.0,
        )
        self.max_penetration = max(self.max_penetration, penetration)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1.0e-12:
            remaining = target - self.time
            count = max(1, int(math.ceil(remaining / _MAX_SUBSTEP)))
            dt = remaining / count
            for _ in range(count):
                self._step(dt)
            self.time = target

        external, _torque, _contact, interaction, gap_ground, gap_pair = self._evaluate()
        self.max_pair_force = max(
            self.max_pair_force,
            float(np.max(np.linalg.norm(interaction, axis=1))) if interaction.size else 0.0,
        )
        signed_distance = np.concatenate([gap_ground, gap_pair])

        return StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "external_force": external.copy(),
                "interaction_force": interaction.copy(),
                "contact_signed_distance": signed_distance,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def diagnostics(self) -> dict[str, object]:
        speed = float(np.linalg.norm(self.velocity[-1]))
        return {
            "method": "newton_euler_penalty_contact_symplectic_euler",
            "substep": _MAX_SUBSTEP,
            "steps": int(self.steps),
            "max_penetration_m": float(self.max_penetration),
            "max_contact_force_N": float(self.max_pair_force),
            "final_far_ball_speed_mps": speed,
            "ball_restitution": _BALL_RESTITUTION,
            "rolling_friction": float(self.mu_roll),
            "sliding_friction": _MU_SLIDE,
        }
