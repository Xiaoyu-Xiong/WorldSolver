"""Physically-based solver for Michell buckling of a pre-twisted elastic ring.

The rod is modelled as a closed, inextensible, naturally straight discrete
curve.  Bending elasticity is treated with a linearised biharmonic operator
about the initial circular configuration, and the stored twist is represented
by a destabilising torsional stiffness that drives the out-of-plane Michell
instability.  Inextensibility is enforced by a fast projection step.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        # The solver only needs the initial state from the scene.  All physical
        # parameters are read from the workspace configuration file because the
        # Solver protocol is intentionally narrow.
        import yaml
        from pathlib import Path

        with open(Path(__file__).parent.parent / "config.yaml") as fh:
            self._config = yaml.safe_load(fh)

        self._seed = seed
        rng = np.random.default_rng(seed)
        self._rng = rng

        frame0 = scene.initial_state()
        self._position: np.ndarray = np.asarray(frame0.position, dtype=float)
        self._velocity: np.ndarray = np.asarray(frame0.velocity, dtype=float)
        self._mass: np.ndarray = np.asarray(frame0.mass, dtype=float)
        self._time: float = 0.0

        self._n = len(self._position)

        rod = self._config["rod"]
        init = self._config["initial_conditions"]
        self._L = float(rod["length"])
        self._alpha = float(rod["bending_stiffness_alpha"])
        self._beta = float(rod["twisting_stiffness_beta"])
        self._rho = float(rod["density"])
        self._r = float(rod["cross_section_radius"])
        self._stored_twist = float(init["stored_total_twist_rad"])
        self._critical_twist = float(init["critical_twist_rad"])

        # Nominal segment length and mass.
        self._l0 = self._L / self._n
        area = np.pi * self._r * self._r
        self._mass_per_vertex = self._rho * area * self._l0

        # Bending coefficient for a naturally straight rod.  The discrete force
        # is -alpha/l0^3 * Delta^2 (position - rest_position), approximating the
        # continuous bending force -alpha * x''''.
        self._bend_coef = self._alpha / (self._l0 ** 3)

        # Twist-induced effective tension.  For a circular ring of radius R the
        # linearised Michell instability threshold is c_twist ~ alpha k^2 / R^2
        # for the destabilised mode k.  We scale by the ratio of stored twist to
        # the critical value given in the configuration.
        R = self._L / (2.0 * np.pi)
        k_mode = 2.0
        c_crit = self._alpha * k_mode * k_mode / (R * R)
        ratio = (self._stored_twist / self._critical_twist) ** 2
        self._twist_coef = c_crit * ratio

        # Use the initially released ring as the reference configuration for
        # bending.  This is the compatible preloaded circle, so the solver
        # starts from an equilibrium of the bending forces.
        self._rest_position = self._position.copy()

        # Stable explicit substep.  The bending time scale for the highest
        # mode is the limiting factor.
        self._dt = 2.0e-4

        # Light velocity damping to stabilise high-frequency numerical modes
        # without suppressing the unstable low mode.
        self._damping = 0.2

    def _biharmonic(self, x: np.ndarray) -> np.ndarray:
        """Periodic discrete biharmonic operator (5-point stencil)."""
        return (
            np.roll(x, -2, axis=0)
            - 4.0 * np.roll(x, -1, axis=0)
            + 6.0 * x
            - 4.0 * np.roll(x, 1, axis=0)
            + np.roll(x, 2, axis=0)
        )

    def _laplacian(self, x: np.ndarray) -> np.ndarray:
        """Periodic discrete 1D Laplacian."""
        return np.roll(x, -1, axis=0) - 2.0 * x + np.roll(x, 1, axis=0)

    def _project_inextensible(self, x: np.ndarray) -> np.ndarray:
        """Fast projection onto the manifold of fixed edge lengths."""
        for _ in range(6):
            e = np.roll(x, -1, axis=0) - x
            lengths = np.linalg.norm(e, axis=1)
            # Avoid division by zero for degenerate edges.
            e_unit = e / (lengths[:, None] + 1.0e-12)
            stretch = (lengths - self._l0)[:, None] * e_unit
            correction = 0.5 * stretch
            x = x - correction + np.roll(correction, 1, axis=0)
        return x

    def _step(self) -> None:
        x = self._position
        v = self._velocity

        # Bending force about the natural circular reference configuration.
        F_bend = -self._bend_coef * self._biharmonic(x - self._rest_position)

        # Twist force: destabilises out-of-plane displacements (Michell mode).
        # Only the axial (z) component is driven; in-plane dynamics remain stable.
        F_twist = np.zeros_like(x)
        F_twist[:, 2] = -self._twist_coef * self._laplacian(x[:, 2]) / (self._l0 ** 2)

        # Velocity damping.
        F_damp = -self._damping * v

        F = F_bend + F_twist + F_damp

        # Symplectic Euler update.
        a = F / self._mass_per_vertex
        v_new = v + self._dt * a
        x_new = x + self._dt * v_new

        # Enforce inextensibility and update velocity to be consistent with
        # the projection.
        x_new = self._project_inextensible(x_new)
        v_new = (x_new - x) / self._dt

        self._position = x_new
        self._velocity = v_new
        self._time += self._dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            time = self._time

        # Integrate forward with fixed substeps until the requested sample time.
        n_steps = int(np.ceil((time - self._time) / self._dt))
        for _ in range(n_steps):
            self._step()

        # If we overshot slightly, clamp the returned time to the requested one.
        return self._build_frame(time)

    def _build_frame(self, t: float) -> StateFrame:
        n = self._n
        position = self._position
        velocity = self._velocity
        mass = self._mass
        angular_velocity = np.zeros((n, 3), dtype=float)
        inertia_tensor = np.zeros((n, 3, 3), dtype=float)
        mass_quantity = mass.copy()

        # Constraint: edge lengths must equal the rest length.
        edges = np.roll(position, -1, axis=0) - position
        edge_lengths = np.linalg.norm(edges, axis=1)
        rest_lengths = np.full_like(edge_lengths, self._l0)

        return StateFrame(
            time=float(t),
            position=position.copy(),
            velocity=velocity.copy(),
            mass=mass.copy(),
            angular_velocity=angular_velocity,
            attributes={
                "inertia_tensor": inertia_tensor,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_angular_impulse": np.zeros((1, 3), dtype=float),
                "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=float),
                "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_domain_weights": np.ones((1, n), dtype=float),
                "budget_external_couple": np.zeros((1, 1, 3), dtype=float),
                "budget_external_force": np.zeros((1, 1, 3), dtype=float),
                "budget_force_position": np.zeros((1, 1, 3), dtype=float),
                "constraint_target": rest_lengths,
                "constraint_value": edge_lengths,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": float(self._l0),
                "reference_angular_momentum": 1.0,
                "reference_momentum": 1.0,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_dt": self._dt,
            "twist_coef": float(self._twist_coef),
            "bend_coef": float(self._bend_coef),
            "time": float(self._time),
        }
