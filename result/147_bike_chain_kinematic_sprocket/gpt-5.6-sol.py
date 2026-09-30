"""Planar rigid-link dynamics for the bicycle-chain benchmark.

The chain is discretized as twenty massive rigid link centres joined by
inextensible pin constraints. Both sprockets are revolute bodies: their
centres are fixed by ideal axle constraints while their angles respond to a
motor torque and to contact impulses from the chain. Positions are advanced
with semi-implicit Euler and projected Gauss--Seidel constraints. The small
fixed substep makes this simple scheme robust for the stiff chain.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic, dependency-free constrained rigid-body solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model is deterministic; no randomized contacts are used.
        initial = scene.initial_state()
        self._config = scene.config
        self._time = 0.0
        self._position = np.asarray(initial.position, dtype=np.float64).copy()
        self._velocity = np.zeros_like(self._position)
        self._mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()

        names = [body["id"] for body in self._config["bodies"]]
        self._chain_ids = np.asarray(
            [i for i, name in enumerate(names) if name.startswith("chain_")],
            dtype=np.int32,
        )
        self._gear_ids = np.asarray(
            [names.index("sprocket_left"), names.index("sprocket_right")],
            dtype=np.int32,
        )
        self._n_chain = len(self._chain_ids)
        if self._n_chain < 3:
            raise ValueError("the closed chain requires at least three links")

        chain_position = self._position[self._chain_ids]
        following = np.roll(chain_position, -1, axis=0)
        self._rest_length = np.linalg.norm(following - chain_position, axis=1)
        if not np.all(self._rest_length > 0.0):
            raise ValueError("chain links must have distinct initial positions")

        self._gear_centres = self._position[self._gear_ids].copy()
        self._gear_angle = np.zeros(2, dtype=np.float64)
        self._gear_omega = np.zeros(2, dtype=np.float64)
        self._gear_inertia = 0.5 * self._mass[self._gear_ids] * 0.55**2

        contact = self._config["contact"]
        self._friction = float(contact["friction"])
        self._restitution = float(contact["restitution"])
        self._contact_scale = float(contact["length_scale"])
        self._gravity = np.asarray(self._config["gravity"], dtype=np.float64)

        # The right sprocket is the motor sprocket. Axle drag represents the
        # small bearing/load torque and prevents unbounded no-load acceleration.
        self._drive_torque = 0.65
        self._axle_drag = 0.045
        self._max_step = 0.0025
        self._constraint_iterations = 12

        self._bond_force = np.zeros((self._n_chain, 3), dtype=np.float64)
        self._contact_force = np.zeros((2, self._n_chain, 3), dtype=np.float64)
        self._chain_angle = self._link_angles(chain_position)
        self._chain_omega = np.zeros(self._n_chain, dtype=np.float64)

    @staticmethod
    def _wrapped_delta(value: np.ndarray) -> np.ndarray:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _link_angles(self, chain_position: np.ndarray) -> np.ndarray:
        """Angle about +y which aligns each link's local x axis to the loop."""
        tangent = np.roll(chain_position, -1, axis=0) - np.roll(
            chain_position, 1, axis=0
        )
        return -np.arctan2(tangent[:, 2], tangent[:, 0])

    @staticmethod
    def _star_radius_and_slope(relative_angle: float) -> tuple[float, float]:
        """Smooth 14-tooth collision proxy and its angular derivative."""
        phase = 14.0 * relative_angle
        wave = 0.5 + 0.5 * math.cos(phase)
        radius = 0.43 + 0.07 * wave**4
        slope = -0.14 * 4.0 * wave**3 * math.sin(phase)
        return radius, slope

    def _gear_contact_geometry(
        self, point: np.ndarray, gear: int
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        centre = self._gear_centres[gear]
        dx = float(point[0] - centre[0])
        dz = float(point[2] - centre[2])
        radius = max(math.hypot(dx, dz), 1.0e-12)
        angle = math.atan2(dz, dx)
        boundary, slope = self._star_radius_and_slope(
            angle - self._gear_angle[gear]
        )
        radial = np.array([dx / radius, 0.0, dz / radius], dtype=np.float64)
        tangent = np.array([dz / radius, 0.0, -dx / radius], dtype=np.float64)
        # grad(r - R(theta)) = e_r - R'(theta)/r e_theta.
        gradient = radial - (slope / radius) * tangent
        gradient /= max(float(np.linalg.norm(gradient)), 1.0e-12)
        return radius - boundary, gradient, tangent, radius

    def _step(self, h: float) -> None:
        chain = self._position[self._chain_ids].copy()
        old_chain = chain.copy()
        velocity = self._velocity[self._chain_ids].copy()
        chain_mass = self._mass[self._chain_ids]

        # Semi-implicit Euler for gravity and mild aerodynamic/link damping.
        velocity += h * self._gravity
        velocity *= math.exp(-0.10 * h)
        trial = chain + h * velocity

        # Integrate prescribed motor torque and viscous axle resistance.
        torque = np.array([0.0, self._drive_torque], dtype=np.float64)
        torque -= self._axle_drag * self._gear_omega
        self._gear_omega += h * torque / self._gear_inertia

        bond_correction = np.zeros((self._n_chain, 3), dtype=np.float64)
        contact_correction = np.zeros((2, self._n_chain, 3), dtype=np.float64)

        for _ in range(self._constraint_iterations):
            # Closed-loop pin constraints. All chain masses are equal in the
            # frozen scene, but inverse-mass weighting is retained explicitly.
            for i in range(self._n_chain):
                j = (i + 1) % self._n_chain
                delta = trial[j] - trial[i]
                length = float(np.linalg.norm(delta))
                if length <= 1.0e-12:
                    continue
                normal = delta / length
                error = length - self._rest_length[i]
                wi = 1.0 / chain_mass[i]
                wj = 1.0 / chain_mass[j]
                lagrange = error / (wi + wj)
                ci = wi * lagrange * normal
                cj = -wj * lagrange * normal
                trial[i] += ci
                trial[j] += cj
                bond_correction[i] += ci

            # Unilateral link/sprocket contact. The radial tooth profile
            # rotates with each sprocket, so its normal impulses transmit torque.
            for gear in range(2):
                for i in range(self._n_chain):
                    gap, normal, _, _ = self._gear_contact_geometry(trial[i], gear)
                    if gap < 0.0:
                        correction = (-gap + 1.0e-10) * normal
                        trial[i] += correction
                        contact_correction[gear, i] += correction

            # Floor contact and the planar guide supplied by the fixed axles.
            trial[:, 2] = np.maximum(trial[:, 2], 0.105)
            trial[:, 1] = 0.0

        velocity = (trial - old_chain) / h

        # Convert active contacts to restitution and Coulomb-limited impulses.
        contact_force = contact_correction * chain_mass[None, :, None] / (h * h)
        for gear in range(2):
            for i in range(self._n_chain):
                gap, normal, tangent, radius = self._gear_contact_geometry(
                    trial[i], gear
                )
                proximity = max(0.0, 1.0 - max(gap, 0.0) / 0.075)
                if proximity <= 0.0:
                    continue

                surface_velocity = self._gear_omega[gear] * radius
                relative_normal = float(np.dot(velocity[i], normal))
                if relative_normal < 0.0 and gap < 0.004:
                    delta_vn = -(1.0 + self._restitution) * relative_normal
                    impulse_n = chain_mass[i] * delta_vn
                    velocity[i] += delta_vn * normal
                    contact_force[gear, i] += impulse_n * normal / h
                    lever = trial[i] - self._gear_centres[gear]
                    torque_impulse = -(
                        lever[0] * impulse_n * normal[2]
                        - lever[2] * impulse_n * normal[0]
                    )
                    self._gear_omega[gear] += torque_impulse / self._gear_inertia[gear]

                relative_tangent = float(np.dot(velocity[i], tangent)) - surface_velocity
                desired_impulse = (
                    -chain_mass[i]
                    * relative_tangent
                    * min(1.0, 32.0 * h)
                    * proximity
                )
                normal_load = (
                    float(np.linalg.norm(contact_force[gear, i])) + 4.0 * proximity
                )
                impulse_limit = self._friction * normal_load * h
                impulse_t = float(
                    np.clip(desired_impulse, -impulse_limit, impulse_limit)
                )
                velocity[i] += (impulse_t / chain_mass[i]) * tangent
                contact_force[gear, i] += (impulse_t / h) * tangent
                self._gear_omega[gear] -= (
                    radius * impulse_t / self._gear_inertia[gear]
                )

        # Remove velocity components that would instantaneously stretch pins.
        for _ in range(2):
            for i in range(self._n_chain):
                j = (i + 1) % self._n_chain
                delta = trial[j] - trial[i]
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-12:
                    continue
                normal = delta / distance
                relative = float(np.dot(velocity[j] - velocity[i], normal))
                impulse_velocity = 0.5 * relative * normal
                velocity[i] += impulse_velocity
                velocity[j] -= impulse_velocity

        self._position[self._chain_ids] = trial
        self._velocity[self._chain_ids] = velocity
        self._position[self._gear_ids] = self._gear_centres
        self._velocity[self._gear_ids] = 0.0
        self._gear_angle += h * self._gear_omega

        new_angle = self._link_angles(trial)
        self._chain_omega = -self._wrapped_delta(new_angle - self._chain_angle) / h
        self._chain_angle = new_angle
        self._bond_force = bond_correction * chain_mass[:, None] / (h * h)
        self._contact_force = contact_force

    def _orientation(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self._position)
        orientation = np.zeros((count, 4), dtype=np.float64)
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((count, 3), dtype=np.float64)

        half = 0.5 * self._chain_angle
        orientation[self._chain_ids, 1] = np.sin(half)
        orientation[self._chain_ids, 3] = np.cos(half)
        angular_velocity[self._chain_ids, 1] = self._chain_omega

        gear_half = 0.5 * self._gear_angle
        orientation[self._gear_ids, 1] = np.sin(gear_half)
        orientation[self._gear_ids, 3] = np.cos(gear_half)
        angular_velocity[self._gear_ids, 1] = self._gear_omega
        return orientation, angular_velocity

    def _signed_distances(self) -> np.ndarray:
        chain = self._position[self._chain_ids]
        gaps = np.empty((2, self._n_chain), dtype=np.float64)
        for gear in range(2):
            for i in range(self._n_chain):
                gaps[gear, i] = self._gear_contact_geometry(chain[i], gear)[0]
        floor_gap = chain[:, 2] - 0.105
        return np.concatenate((gaps.ravel(), floor_gap))

    def _interaction_forces(self) -> np.ndarray:
        # One force vector for each pin pair and chain/sprocket contact pair.
        pair_force = [self._bond_force, self._contact_force.reshape(-1, 3)]

        # Ideal axle support balances each fixed sprocket's weight. This is an
        # axle/sprocket interaction even though the axle is not a sampled body.
        support = -self._mass[self._gear_ids, None] * self._gravity[None, :]
        pair_force.append(support)
        force = np.concatenate(pair_force, axis=0)
        # Export both participants of every pair, making internal force balance
        # explicit without requiring a separate pair-index trajectory field.
        return np.concatenate((force, -force), axis=0)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self._time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")

        while self._time + 1.0e-12 < target:
            h = min(self._max_step, target - self._time)
            self._step(h)
            self._time += h
        self._time = target

        orientation, angular_velocity = self._orientation()
        external_force = self._mass[:, None] * self._gravity[None, :]
        frame = StateFrame(
            time=target,
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields={
                "contact_signed_distance": self._signed_distances(),
                "external_force": external_force,
                "interaction_force": self._interaction_forces(),
            },
            scalars={"contact_length_scale": self._contact_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_euler_projected_constraints",
            "substep": self._max_step,
            "constraint_iterations": self._constraint_iterations,
            "simulated_time": self._time,
            "drive_torque_nm": self._drive_torque,
        }
