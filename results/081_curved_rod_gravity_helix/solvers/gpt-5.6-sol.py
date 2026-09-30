"""Self-contained discrete Cosserat-rod solver for the curved-rod benchmark.

The rod is represented by a small spectral expansion of its material curvature.
Its centreline is reconstructed by integrating the Kirchhoff kinematics, which
makes it inextensible by construction. The degrees of freedom obey a damped
mechanical system driven by the ramped gravitational potential.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Damped, linearly implicit reduced-order Cosserat rod integrator."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        rod = self.config["rod"]
        initial = scene.initial_state()

        self.n = len(initial.position)
        self.nseg = self.n - 1
        self.length = float(rod["length"])
        self.ds = self.length / self.nseg
        self.root = np.asarray(
            self.config["boundary_conditions"]["root"]["position"], dtype=float
        )
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.total_mass = float(np.sum(self.mass))
        self.arc_length = np.linspace(0.0, self.length, self.n)
        # Match the authoritative natural-arc construction in scene.py exactly;
        # the separately reported curvature is rounded in the public config.
        self.kappa0 = 1.0 / float(rod["natural_radius"])

        # The benchmark does not prescribe E or nu. These representative soft
        # elastomer values put this specimen in the intended gravity-dominated
        # bend--twist regime while retaining an isotropic circular-section law.
        young_modulus = 8.0e5
        poisson_ratio = 0.45
        radius = float(rod["radius"])
        second_moment = math.pi * radius**4 / 4.0
        polar_moment = 2.0 * second_moment
        shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
        self.bending_rigidity = young_modulus * second_moment
        self.twisting_rigidity = shear_modulus * polar_moment

        # Three smooth longitudinal modes per curvature component capture the
        # global spatial curl without prescribing its shape.
        self.mode_count = 3
        mid_s = (np.arange(self.nseg, dtype=float) + 0.5) * self.ds
        xi = mid_s / self.length
        self.basis = np.column_stack(
            (
                np.ones_like(xi),
                np.cos(math.pi * xi),
                np.cos(2.0 * math.pi * xi),
            )
        )
        gram = self.ds * (self.basis.T @ self.basis)
        zero = np.zeros_like(gram)
        self.elastic_stiffness = np.block(
            [
                [self.bending_rigidity * gram, zero, zero],
                [zero, self.bending_rigidity * gram, zero],
                [zero, zero, self.twisting_rigidity * gram],
            ]
        )
        self.ndof = 3 * self.mode_count

        # Constant generalized inertia from the exact centreline kinetic energy.
        q_zero = np.zeros(self.ndof, dtype=float)
        derivative_step = 1.0e-3
        jacobian = np.empty((self.n, 3, self.ndof), dtype=float)
        for j in range(self.ndof):
            dq = np.zeros(self.ndof, dtype=float)
            dq[j] = derivative_step
            plus = self._centerline(q_zero + dq)
            minus = self._centerline(q_zero - dq)
            jacobian[:, :, j] = (plus - minus) / (2.0 * derivative_step)
        generalized_mass = np.einsum(
            "i,iaj,iak->jk", self.mass, jacobian, jacobian, optimize=True
        )
        generalized_mass = 0.5 * (generalized_mass + generalized_mass.T)
        eigenvalues, eigenvectors = np.linalg.eigh(generalized_mass)
        mass_floor = max(float(np.max(eigenvalues)) * 1.0e-6, 1.0e-13)
        self.generalized_mass = (eigenvectors * np.maximum(eigenvalues, mass_floor)) @ (
            eigenvectors.T
        )
        self.damping = 0.8 * self.generalized_mass + 0.008 * self.elastic_stiffness

        # Convert the configured lateral branch selector into an initial
        # curvature perturbation. It is not a continuing external load.
        desired_y = float(
            self.config["initial_conditions"]["branch_selection_perturbation"][1]
        )
        unit_q = np.zeros(self.ndof, dtype=float)
        unit_q[0] = 1.0
        unit_tip_y = self._centerline(unit_q)[-1, 1]
        self.q = np.zeros(self.ndof, dtype=float)
        if abs(unit_tip_y) > 1.0e-12:
            self.q[0] = desired_y / unit_tip_y
        self.qdot = np.zeros(self.ndof, dtype=float)

        self.current_time = 0.0
        self.previous_sample_time = 0.0
        self.previous_position = self._centerline(self.q)
        self.previous_velocity = np.zeros_like(self.previous_position)
        self.initial_chord_length = float(
            np.mean(np.linalg.norm(np.diff(initial.position, axis=0), axis=1))
        )
        self.internal_steps = 0
        self.max_abs_curvature_dof = float(np.max(np.abs(self.q)))

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = vector
        return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))

    def _centerline(self, q: np.ndarray) -> np.ndarray:
        """Integrate x'=d3 and R'=R hat(kappa), exactly per segment."""
        m = self.mode_count
        curvature = np.empty((self.nseg, 3), dtype=float)
        curvature[:, 0] = self.basis @ q[0:m]
        curvature[:, 1] = self.kappa0 + self.basis @ q[m : 2 * m]
        curvature[:, 2] = self.basis @ q[2 * m : 3 * m]

        position = np.empty((self.n, 3), dtype=float)
        position[0] = self.root
        # Columns are directors; d3 is the fixed downward root tangent.
        rotation = np.array(
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            dtype=float,
        )
        e3 = np.array((0.0, 0.0, 1.0))
        h = self.ds
        identity = np.eye(3)

        for i in range(self.nseg):
            kappa = curvature[i]
            omega = float(np.linalg.norm(kappa))
            skew = self._skew(kappa)
            skew2 = skew @ skew
            if omega < 1.0e-10:
                exponential = identity + h * skew + 0.5 * h * h * skew2
                integral = h * identity + 0.5 * h * h * skew + (h**3 / 6.0) * skew2
            else:
                angle = omega * h
                sin_angle = math.sin(angle)
                cos_angle = math.cos(angle)
                exponential = (
                    identity
                    + (sin_angle / omega) * skew
                    + ((1.0 - cos_angle) / (omega * omega)) * skew2
                )
                integral = (
                    h * identity
                    + ((1.0 - cos_angle) / (omega * omega)) * skew
                    + ((h / (omega * omega)) - sin_angle / (omega**3)) * skew2
                )
            position[i + 1] = position[i] + rotation @ (integral @ e3)
            rotation = rotation @ exponential
        return position

    def _load_fraction(self, time: float) -> float:
        schedule = self.config["load_schedule"]
        start = float(schedule.get("start_time", 0.0))
        full = float(schedule["full_load_time"])
        if time <= start:
            return 0.0
        if full <= start:
            return 1.0
        return float(np.clip((time - start) / (full - start), 0.0, 1.0))

    def _gravity_gradient(self, q: np.ndarray, load: float) -> np.ndarray:
        """Central difference of the exact discrete gravitational potential."""
        if load == 0.0:
            return np.zeros(self.ndof, dtype=float)
        gradient = np.empty(self.ndof, dtype=float)
        epsilon = 1.0e-3
        for j in range(self.ndof):
            dq = np.zeros(self.ndof, dtype=float)
            dq[j] = epsilon
            plus = self._centerline(q + dq)
            minus = self._centerline(q - dq)
            # U_g = -sum m_i g dot x_i.
            u_plus = -load * float(np.sum(self.mass[:, None] * plus * self.gravity))
            u_minus = -load * float(np.sum(self.mass[:, None] * minus * self.gravity))
            gradient[j] = (u_plus - u_minus) / (2.0 * epsilon)
        return gradient

    def _step(self, step: float, end_time: float) -> None:
        load = self._load_fraction(end_time - 0.5 * step)
        gravity_gradient = self._gravity_gradient(self.q, load)
        system = (
            self.generalized_mass
            + step * self.damping
            + step * step * self.elastic_stiffness
        )
        right_hand_side = self.generalized_mass @ self.qdot - step * (
            self.elastic_stiffness @ self.q + gravity_gradient
        )
        next_velocity = np.linalg.solve(system, right_hand_side)
        next_q = self.q + step * next_velocity
        if not np.isfinite(next_q).all() or not np.isfinite(next_velocity).all():
            raise FloatingPointError("non-finite generalized rod state")
        magnitude = float(np.max(np.abs(next_q)))
        if magnitude > 600.0:
            scale = 600.0 / magnitude
            next_q *= scale
            next_velocity *= scale
        self.q = next_q
        self.qdot = next_velocity
        self.current_time = end_time
        self.internal_steps += 1
        self.max_abs_curvature_dof = max(
            self.max_abs_curvature_dof, float(np.max(np.abs(self.q)))
        )

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        if not math.isfinite(target_time) or target_time < self.current_time - 1.0e-12:
            raise ValueError("advance_to times must be finite and nondecreasing")

        maximum_step = 0.005
        while self.current_time < target_time - 1.0e-12:
            step = min(maximum_step, target_time - self.current_time)
            self._step(step, self.current_time + step)

        position = self._centerline(self.q)
        sample_interval = target_time - self.previous_sample_time
        if sample_interval > 1.0e-12:
            velocity = (position - self.previous_position) / sample_interval
            acceleration = (velocity - self.previous_velocity) / sample_interval
        else:
            velocity = np.zeros_like(position)
            acceleration = np.zeros_like(position)

        load = self._load_fraction(target_time)
        nodal_gravity = load * self.mass[:, None] * self.gravity[None, :]
        inertial_force = self.mass[:, None] * acceleration
        clamp_force = np.sum(inertial_force - nodal_gravity, axis=0)

        # N gravity forces and one root traction. The root couple closes angular
        # momentum and is not duplicated by an offset force application point.
        force = np.zeros((1, self.n + 1, 3), dtype=float)
        force[0, : self.n] = nodal_gravity
        force[0, self.n] = clamp_force
        force_position = np.empty_like(force)
        force_position[0, : self.n] = position
        force_position[0, self.n] = self.root
        moment_inertial = np.sum(
            np.cross(position - self.root, inertial_force), axis=0
        )
        moment_gravity = np.sum(
            np.cross(position - self.root, nodal_gravity), axis=0
        )
        couple = np.zeros_like(force)
        couple[0, self.n] = moment_inertial - moment_gravity

        full_weight = self.total_mass * float(np.linalg.norm(self.gravity))

        frame = StateFrame(
            time=target_time,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "natural_curvature": np.full(self.n, self.kappa0, dtype=float),
                "natural_twist": np.zeros(self.n, dtype=float),
                "is_clamped_root": np.r_[1.0, np.zeros(self.n - 1)],
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": force,
                "budget_force_position": force_position,
                "budget_external_couple": couple,
                # The selected holonomic constraint is the fixed clamp point.
                "constraint_value": position[0:1].copy(),
                "constraint_target": self.root[None, :].copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "rod_radius": float(self.config["rod"]["radius"]),
                "branch_selection_perturbation_y": float(
                    self.config["initial_conditions"]["branch_selection_perturbation"][1]
                ),
                "constraint_scale": self.initial_chord_length,
                "reference_force": max(full_weight, 1.0e-12),
                "reference_torque": max(full_weight * self.length, 1.0e-12),
            },
        )
        frame.validate()
        self.previous_sample_time = target_time
        self.previous_position = position.copy()
        self.previous_velocity = velocity.copy()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "reduced Cosserat rod with linearly implicit dynamics",
            "internal_steps": int(self.internal_steps),
            "bending_rigidity_Nm2": float(self.bending_rigidity),
            "twisting_rigidity_Nm2": float(self.twisting_rigidity),
            "max_abs_curvature_dof_per_m": float(self.max_abs_curvature_dof),
        }
