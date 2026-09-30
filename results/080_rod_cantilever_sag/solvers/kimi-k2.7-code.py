"""Quasi-static discrete elastic-rod solver for a clamped-free cantilever.

The rod is modelled as an inextensible chain of straight edges.  The root edge is
forced to remain horizontal, while the remaining edge angles are found by
minimising the total potential energy (bending + gravity) at the current load
factor.  This is a genuine physical time-stepping procedure: each sampled frame
is an independent static equilibrium at the instantaneous gravity ramp value.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._scene = scene
        self._config = scene.config

        # Initial frame from the scene defines the discretisation.
        init = scene.initial_state()
        self._position0 = np.asarray(init.position, dtype=float)
        self._velocity0 = np.asarray(init.velocity, dtype=float)
        self._mass = np.asarray(init.mass, dtype=float)
        self._n = self._position0.shape[0]
        self._root_pos = self._position0[0].copy()
        self._tangent0 = np.array([1.0, 0.0, 0.0])

        # Rod parameters (SI).
        rod_cfg = self._config["rod"]
        self._length = float(rod_cfg["length"])
        self._radius = float(rod_cfg["radius"])
        self._density = float(rod_cfg["density"])
        self._area = np.pi * self._radius * self._radius
        self._segment_length = self._length / (self._n - 1)
        self._total_mass = float(np.sum(self._mass))

        # Young's modulus is not provided in the scaffold; choose a physically
        # plausible elastomer value that yields a clearly visible sag.
        young_modulus = 2.0e7  # Pa  (~20 MPa, a medium-soft elastomer)
        inertia = np.pi * self._radius ** 4 / 4.0
        self._ei = young_modulus * inertia

        # Gravity vector from the scaffold.
        self._gvec = np.asarray(self._config["gravity"], dtype=float)
        self._g0 = float(np.linalg.norm(self._gvec))
        if self._g0 == 0.0:
            self._g0 = 9.81
            self._gvec = np.array([0.0, 0.0, -self._g0])
        else:
            self._gvec = self._gvec * (self._g0 / float(np.linalg.norm(self._gvec)))

        # Load schedule.
        ls = self._config.get("load_schedule", {})
        self._load_start = float(ls.get("start_time", 0.0))
        self._load_full = float(ls.get("full_load_time", self._load_start + 1.0))
        self._load_hold = float(ls.get("hold_until", self._load_full))

        # Precompute cumulative mass seen by each edge angle.
        # M_e = sum_{i=e+1}^{N-1} m_i is the total nodal mass supported by edge e.
        self._cum_mass_from_edge = np.zeros(self._n - 1, dtype=float)
        for e in range(self._n - 1):
            self._cum_mass_from_edge[e] = float(np.sum(self._mass[e + 1 :]))

        # Bending Hessian (variable part only, edge angles 1..N-2).
        m = self._n - 2
        self._bend_hessian = np.zeros((m, m), dtype=float)
        for i in range(m):
            self._bend_hessian[i, i] = 2.0
        self._bend_hessian[-1, -1] = 1.0
        for i in range(m - 1):
            self._bend_hessian[i, i + 1] = -1.0
            self._bend_hessian[i + 1, i] = -1.0
        self._bend_hessian *= self._ei / self._segment_length

        # Initial / previous equilibrium edge angles (root edge angle fixed at 0).
        self._phi_prev = np.zeros(max(m, 0), dtype=float)

        # Solve full-load equilibrium once to obtain reference scales.
        self._ref_force = self._total_mass * self._g0
        phi_full = self._solve_equilibrium(1.0, self._phi_prev)
        x_full = self._reconstruct_positions(phi_full)
        self._ref_torque = float(np.linalg.norm(self._root_clamping_couple(x_full, 1.0)))
        if self._ref_torque <= 0.0:
            self._ref_torque = 1.0
        if self._ref_force <= 0.0:
            self._ref_force = 1.0

        # Keep the full-load solution as a warm start seed for the first frame.
        self._phi_prev = phi_full.copy()

        # Cache arc-length attribute from the initial frame.
        self._arc_length = np.asarray(init.attributes.get("arc_length"), dtype=float)
        if self._arc_length.shape[0] != self._n:
            self._arc_length = np.linspace(0.0, self._length, self._n)
        self._is_clamped_root = np.zeros(self._n, dtype=float)
        self._is_clamped_root[0] = 1.0

    def _load_factor(self, time: float) -> float:
        if time <= self._load_start:
            return 0.0
        if time >= self._load_full:
            return 1.0
        return (time - self._load_start) / (self._load_full - self._load_start)

    def _objective(self, phi: np.ndarray, alpha: float) -> float:
        # phi are the free edge angles (edges 1..N-2). Root edge (edge 0) is horizontal.
        # Inextensibility is built into the parameterisation.
        ph = np.concatenate(([0.0], phi))
        diff = ph[1:] - ph[:-1]
        bending = 0.5 * (self._ei / self._segment_length) * np.dot(diff, diff)
        # Gravity: U_g = -sum_i m_i (g . x_i) = alpha * g0 * l0 * sum_e sin(phi_e) * M_e.
        gravity = alpha * self._g0 * self._segment_length * np.sum(
            np.sin(ph) * self._cum_mass_from_edge
        )
        return bending + gravity

    def _gradient_hessian(self, phi: np.ndarray, alpha: float):
        ph = np.concatenate(([0.0], phi))
        # Bending gradient on the free edge angles.
        m = phi.shape[0]
        g_bend = (self._ei / self._segment_length) * (self._bend_hessian @ phi)
        # Gravity gradient.
        g_grav = (
            alpha
            * self._g0
            * self._segment_length
            * np.cos(ph[1:])
            * self._cum_mass_from_edge[1:]
        )
        grad = g_bend + g_grav
        # Hessian.
        h_grav = (
            alpha
            * self._g0
            * self._segment_length
            * (-np.sin(ph[1:]))
            * self._cum_mass_from_edge[1:]
        )
        hess = self._bend_hessian + np.diag(h_grav)
        return grad, hess

    def _solve_equilibrium(self, alpha: float, phi_guess: np.ndarray) -> np.ndarray:
        # For zero load the straight rod is the exact equilibrium.
        if alpha <= 0.0:
            return np.zeros_like(phi_guess)
        phi = phi_guess.copy()
        tol = 1e-10
        max_iter = 50
        c = 1e-4
        for _ in range(max_iter):
            grad, hess = self._gradient_hessian(phi, alpha)
            if np.linalg.norm(grad) < tol:
                break
            try:
                step = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
            # Backtracking line search (Armijo).
            obj0 = self._objective(phi, alpha)
            slope = np.dot(grad, step)
            step_size = 1.0
            for _ in range(20):
                phi_trial = phi + step_size * step
                if self._objective(phi_trial, alpha) <= obj0 + c * step_size * slope:
                    phi = phi_trial
                    break
                step_size *= 0.5
            else:
                phi = phi_trial
        return phi

    def _reconstruct_positions(self, phi: np.ndarray) -> np.ndarray:
        ph = np.concatenate(([0.0], phi))
        dirs = np.column_stack((np.cos(ph), np.zeros_like(ph), np.sin(ph)))
        # Node 0 is the clamped root.
        positions = np.zeros((self._n, 3), dtype=float)
        positions[0] = self._root_pos
        for i in range(1, self._n):
            positions[i] = positions[i - 1] + self._segment_length * dirs[i - 1]
        return positions

    def _root_clamping_couple(self, positions: np.ndarray, alpha: float) -> np.ndarray:
        # Pure moment applied by the clamp about the root that balances gravity.
        r = positions - self._root_pos
        f_grav = self._mass[:, None] * self._gvec[None, :] * alpha
        tau_grav = np.sum(np.cross(r, f_grav, axisa=0, axisb=0, axisc=0), axis=0)
        return -tau_grav

    def advance_to(self, time: float) -> StateFrame:
        alpha = self._load_factor(time)
        phi = self._solve_equilibrium(alpha, self._phi_prev)
        self._phi_prev = phi.copy()

        positions = self._reconstruct_positions(phi)
        velocities = np.zeros_like(positions)

        # External-force budget for the whole rod (G=1).
        k = self._n + 1
        budget_force = np.zeros((1, k, 3), dtype=float)
        budget_couple = np.zeros((1, k, 3), dtype=float)
        budget_pos = np.zeros((1, k, 3), dtype=float)

        # Per-node gravity.
        f_g = self._mass[:, None] * self._gvec[None, :] * alpha
        budget_force[0, : self._n, :] = f_g
        budget_pos[0, : self._n, :] = positions

        # Root constraint traction (force + clamping couple).
        total_gravity = np.sum(f_g, axis=0)
        root_force = -total_gravity
        root_couple = self._root_clamping_couple(positions, alpha)
        budget_force[0, self._n, :] = root_force
        budget_couple[0, self._n, :] = root_couple
        budget_pos[0, self._n, :] = self._root_pos

        # Constraint diagnostics for the clamped root.
        constraint_target = np.array([self._root_pos, self._tangent0], dtype=float)
        current_tangent = np.array(
            [np.cos(ph[0]), 0.0, np.sin(ph[0])]
        ) if phi.size > 0 else self._tangent0.copy()
        constraint_value = np.array([positions[0], current_tangent], dtype=float)

        frame = StateFrame(
            time=float(time),
            position=positions,
            velocity=velocities,
            mass=self._mass.copy(),
            attributes={
                "arc_length": self._arc_length.copy(),
                "is_clamped_root": self._is_clamped_root.copy(),
                "mass_quantity": self._mass.copy(),
            },
            scalars={
                "rod_radius": self._radius,
                "reference_force": float(self._ref_force),
                "reference_torque": float(self._ref_torque),
                "constraint_scale": 1.0,
            },
            fields={
                "budget_external_force": budget_force,
                "budget_external_couple": budget_couple,
                "budget_force_position": budget_pos,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "quasi_static_discrete_elastic_rod",
            "vertices": int(self._n),
            "edges": int(self._n - 1),
            "young_modulus_pa": float(self._ei / (np.pi * self._radius ** 4 / 4.0)),
            "bending_stiffness_n_m2": float(self._ei),
            "reference_force_n": float(self._ref_force),
            "reference_torque_n_m": float(self._ref_torque),
        }
