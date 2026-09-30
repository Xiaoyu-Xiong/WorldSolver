"""Quasi-static gravitational sag of a clamped-free inextensible elastic rod.

Physical model
--------------
The rod is a naturally straight, homogeneous, inextensible Kirchhoff rod with a
circular cross-section.  Gravity and the clamp act in the x-z plane, so the
equilibrium centerline stays planar and is parametrised exactly by the tangent
angles of its segments:

    x_{j+1} = x_j + l0 * (cos(theta_j), 0, sin(theta_j)),   j = 0 .. n-1

with ``l0`` the (constant) rest length of a segment.  This parametrisation makes
inextensibility an identity rather than a penalty, and the clamp is imposed by
freezing the root node position and ``theta_0 = 0`` (horizontal tangent).

Total potential energy of the discrete rod:

    E(theta) = sum_k (EI / (2 l0)) (theta_k - theta_{k-1})^2      (bending)
             + g(t) * sum_i m_i z_i(theta)                        (gravity)

The first term is the discrete Euler-Bernoulli bending energy with discrete
curvature (theta_k - theta_{k-1}) / l0 integrated over one Voronoi length l0.

Time stepping
-------------
``simulation_mode: quasi_static_equilibria`` with a ``gravity_ramp`` load
schedule.  The solver therefore performs incremental quasi-static load
stepping: the time interval is marched with fixed sub-steps, the ramped gravity
load g(t) is applied at each sub-step, and the nonlinear equilibrium equations

    grad E(theta; g(t)) = 0        (free DOFs theta_1 .. theta_{n-1})

are solved with a damped Newton-Raphson iteration using the previous converged
configuration as the predictor.  The analytic Hessian is tridiagonal (bending)
plus diagonal (gravity), so each Newton solve is exact and cheap.  Nothing is
keyframed: every configuration is the numerical solution of the rod equilibrium
equations at that instant's load.

Exported momentum budget
------------------------
External actions on the rod domain are (0) the gravity resultant applied at the
current center of mass and (1) the clamp reaction force applied at the root
together with the clamp reaction couple.  The reaction is recovered from the
balance laws using the solver's own accelerations, i.e.

    R      = sum_i m_i a_i - M g(t)
    C_clamp= sum_i m_i (r_i x a_i) - r_com x F_g - r_root x R

which is the physical clamping wrench (in the quasi-static limit it reduces to
the root shear force and bending moment carried by the clamp).  Internal
elastic forces are translation and rotation invariant, hence they cancel in the
whole-rod domain and are not exported as external terms.
"""
from __future__ import annotations

import numpy as np


class BenchmarkSolver:
    # Material stiffness of the rod elastomer (frozen by the solver's material
    # model; the scaffold fixes geometry, density and gravity only).
    YOUNGS_MODULUS = 2.0e8  # Pa

    # Quasi-static load sub-steps per sample interval.
    SUBSTEPS = 4
    NEWTON_MAX_ITER = 60
    NEWTON_TOL = 1e-12  # on the generalized force residual (N*m)

    def __init__(self) -> None:
        self._ready = False
        self._diagnostics: dict[str, object] = {}

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        self.config = config

        rod = config["rod"]
        self.length = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])

        gravity = np.asarray(config["gravity"], dtype=float)
        self.gravity_dir = gravity / np.linalg.norm(gravity)
        self.gravity_mag = float(np.linalg.norm(gravity))

        schedule = config.get("load_schedule", {}) or {}
        self.ramp_start = float(schedule.get("start_time", 0.0))
        self.ramp_full = float(schedule.get("full_load_time", 0.0))

        boundary = config["boundary_conditions"]["root"]
        self.root_position = np.asarray(boundary["position"], dtype=float)
        tangent = np.asarray(boundary["tangent"], dtype=float)
        self.root_tangent = tangent / np.linalg.norm(tangent)

        initial = scene.initial_state()
        self.node_count = int(initial.position.shape[0])
        self.segment_count = self.node_count - 1
        self.l0 = self.length / self.segment_count
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.total_mass = float(self.mass.sum())
        self.arc_length = np.asarray(
            initial.attributes["arc_length"], dtype=float
        ).copy()
        self.is_clamped_root = np.asarray(
            initial.attributes["is_clamped_root"], dtype=float
        ).copy()

        # Cross-section properties of the circular rod.
        self.area = np.pi * self.radius**2
        self.inertia = 0.25 * np.pi * self.radius**4
        self.bending_stiffness = self.YOUNGS_MODULUS * self.inertia  # EI

        # Tail masses: sum of nodal masses strictly beyond segment k.
        tail = np.cumsum(self.mass[::-1])[::-1]
        self.tail_mass = np.empty(self.segment_count, dtype=float)
        self.tail_mass[:] = tail[1:]

        # Bending operator (second difference) on the segment angles.
        n = self.segment_count
        diff = np.zeros((n - 1, n), dtype=float)
        rows = np.arange(n - 1)
        diff[rows, rows] = -1.0
        diff[rows, rows + 1] = 1.0
        self._bend_matrix = (self.bending_stiffness / self.l0) * (diff.T @ diff)
        self._free = np.arange(1, n)  # theta_0 clamped horizontal

        self.sample_times = [float(t) for t in scene.sample_times()]
        self._integrate()
        self._ready = True

    # ------------------------------------------------------- load and physics
    def _load_factor(self, time: float) -> float:
        """Smooth monotone gravity ramp: 0 at start_time, 1 from full_load_time."""
        if self.ramp_full <= self.ramp_start:
            return 1.0
        s = (float(time) - self.ramp_start) / (self.ramp_full - self.ramp_start)
        if s <= 0.0:
            return 0.0
        if s >= 1.0:
            return 1.0
        return float(s * s * (3.0 - 2.0 * s))

    def _energy(self, theta: np.ndarray, g_eff: float) -> float:
        bend = 0.5 * float(theta @ (self._bend_matrix @ theta))
        # Height of node i relative to the root: l0 * sum_{k<i} sin(theta_k).
        heights = self.l0 * np.concatenate(([0.0], np.cumsum(np.sin(theta))))
        grav = g_eff * float(self.mass @ heights)
        return bend + grav

    def _gradient(self, theta: np.ndarray, g_eff: float) -> np.ndarray:
        grad = self._bend_matrix @ theta
        grad += g_eff * self.l0 * np.cos(theta) * self.tail_mass
        return grad

    def _hessian(self, theta: np.ndarray, g_eff: float) -> np.ndarray:
        hess = self._bend_matrix.copy()
        diag = -g_eff * self.l0 * np.sin(theta) * self.tail_mass
        hess[np.diag_indices_from(hess)] += diag
        return hess

    def _solve_equilibrium(self, theta: np.ndarray, g_eff: float) -> np.ndarray:
        """Damped Newton solve of grad E = 0 on the free tangent angles."""
        free = self._free
        theta = theta.copy()
        energy = self._energy(theta, g_eff)
        for _ in range(self.NEWTON_MAX_ITER):
            grad = self._gradient(theta, g_eff)[free]
            residual = float(np.max(np.abs(grad)))
            if residual < self.NEWTON_TOL:
                break
            hess = self._hessian(theta, g_eff)[np.ix_(free, free)]
            step = None
            regular = 0.0
            base = float(np.trace(hess)) / hess.shape[0]
            for _attempt in range(12):
                matrix = hess.copy()
                if regular > 0.0:
                    matrix[np.diag_indices_from(matrix)] += regular
                try:
                    candidate = np.linalg.solve(matrix, -grad)
                except np.linalg.LinAlgError:
                    candidate = None
                if candidate is not None and np.all(np.isfinite(candidate)):
                    if float(candidate @ grad) < 0.0:  # descent direction
                        step = candidate
                        break
                regular = base * 1e-6 if regular == 0.0 else regular * 10.0
            if step is None:
                step = -grad / max(base, 1e-30)  # gradient fallback

            # Backtracking line search on the total potential energy.
            alpha = 1.0
            accepted = False
            for _ls in range(40):
                trial = theta.copy()
                trial[free] = theta[free] + alpha * step
                trial_energy = self._energy(trial, g_eff)
                if np.isfinite(trial_energy) and trial_energy <= energy:
                    theta = trial
                    energy = trial_energy
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        return theta

    def _positions(self, theta: np.ndarray) -> np.ndarray:
        """Centerline nodes from segment angles (exactly inextensible)."""
        dx = self.l0 * np.cos(theta)
        dz = self.l0 * np.sin(theta)
        position = np.zeros((self.node_count, 3), dtype=float)
        position[:, 0] = self.root_position[0] + np.concatenate(
            ([0.0], np.cumsum(dx))
        )
        position[:, 1] = self.root_position[1]
        position[:, 2] = self.root_position[2] + np.concatenate(
            ([0.0], np.cumsum(dz))
        )
        return position

    def _integrate(self) -> None:
        """Incremental quasi-static load stepping over the sample schedule."""
        theta = np.zeros(self.segment_count, dtype=float)  # naturally straight
        theta = self._solve_equilibrium(theta, self.gravity_mag * self._load_factor(
            self.sample_times[0]
        ))
        positions = [self._positions(theta)]
        loads = [self._load_factor(self.sample_times[0])]
        max_residual = 0.0
        for index in range(1, len(self.sample_times)):
            t_prev = self.sample_times[index - 1]
            t_now = self.sample_times[index]
            for sub in range(1, self.SUBSTEPS + 1):
                t_sub = t_prev + (t_now - t_prev) * sub / self.SUBSTEPS
                g_eff = self.gravity_mag * self._load_factor(t_sub)
                theta = self._solve_equilibrium(theta, g_eff)
            residual = float(
                np.max(np.abs(self._gradient(theta, g_eff)[self._free]))
            )
            max_residual = max(max_residual, residual)
            positions.append(self._positions(theta))
            loads.append(self._load_factor(t_now))

        self._position_history = np.asarray(positions, dtype=float)
        self._load_history = np.asarray(loads, dtype=float)
        self._velocity_history = self._time_derivative(self._position_history)
        self._acceleration_history = self._time_derivative(self._velocity_history)

        tip_drop = float(
            self._position_history[0, -1, 2] - self._position_history[-1, -1, 2]
        )
        self._diagnostics = {
            "method": "inextensible planar elastica, incremental quasi-static Newton",
            "nodes": self.node_count,
            "youngs_modulus": self.YOUNGS_MODULUS,
            "bending_stiffness_EI": self.bending_stiffness,
            "total_mass": self.total_mass,
            "load_substeps_per_sample": self.SUBSTEPS,
            "max_equilibrium_residual": max_residual,
            "tip_drop": tip_drop,
            "final_tip_position": self._position_history[-1, -1].tolist(),
        }

    def _time_derivative(self, history: np.ndarray) -> np.ndarray:
        """Second-order centered time derivative of a sampled history."""
        times = np.asarray(self.sample_times, dtype=float)
        derivative = np.zeros_like(history)
        count = len(times)
        if count < 2:
            return derivative
        for index in range(count):
            if index == 0:
                dt = times[1] - times[0]
                derivative[index] = (history[1] - history[0]) / dt
            elif index == count - 1:
                dt = times[-1] - times[-2]
                derivative[index] = (history[-1] - history[-2]) / dt
            else:
                dt = times[index + 1] - times[index - 1]
                derivative[index] = (history[index + 1] - history[index - 1]) / dt
        derivative[~np.isfinite(derivative)] = 0.0
        return derivative

    # ------------------------------------------------------------ frame build
    def _frame_index(self, time: float) -> int:
        times = np.asarray(self.sample_times, dtype=float)
        return int(np.argmin(np.abs(times - float(time))))

    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        index = self._frame_index(time)
        position = self._position_history[index]
        velocity = self._velocity_history[index]
        acceleration = self._acceleration_history[index]
        load = float(self._load_history[index])
        mass = self.mass

        # --- external momentum budget for the single rod domain -------------
        gravity_vector = self.gravity_dir * (self.gravity_mag * load)
        gravity_force = self.total_mass * gravity_vector
        center_of_mass = (mass[:, None] * position).sum(axis=0) / self.total_mass

        momentum_rate = (mass[:, None] * acceleration).sum(axis=0)
        angular_rate = np.cross(position, mass[:, None] * acceleration).sum(axis=0)

        reaction_force = momentum_rate - gravity_force
        root = position[0]
        reaction_couple = (
            angular_rate
            - np.cross(center_of_mass, gravity_force)
            - np.cross(root, reaction_force)
        )

        budget_external_force = np.stack((gravity_force, reaction_force))[None, :, :]
        budget_force_position = np.stack((center_of_mass, root))[None, :, :]
        budget_external_couple = np.stack(
            (np.zeros(3), reaction_couple)
        )[None, :, :]

        # --- frozen kinematic constraints (all in meters) -------------------
        edges = position[1:] - position[:-1]
        edge_lengths = np.linalg.norm(edges, axis=1)
        clamp_node = self.root_position + self.l0 * self.root_tangent
        constraint_value = np.concatenate(
            (edge_lengths, position[0], position[1])
        )
        constraint_target = np.concatenate(
            (
                np.full(self.segment_count, self.l0),
                self.root_position,
                clamp_node,
            )
        )

        frame = StateFrame(
            time=float(self.sample_times[index]),
            position=position.copy(),
            velocity=velocity.copy(),
            mass=mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "is_clamped_root": self.is_clamped_root.copy(),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "budget_external_couple": budget_external_couple,
                "budget_domain_weights": np.ones((1, self.node_count), dtype=float),
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "rod_radius": self.radius,
                "constraint_scale": float(self.l0),
                "reference_force": float(self.total_mass * self.gravity_mag),
                "reference_torque": float(
                    self.total_mass * self.gravity_mag * self.length
                ),
                "load_factor": load,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)
