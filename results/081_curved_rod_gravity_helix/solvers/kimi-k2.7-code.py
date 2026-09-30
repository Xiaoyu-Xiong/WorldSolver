"""Self-contained solver for the naturally curved elastic rod bend-twist benchmark.

The rod is modelled as an inextensible chain of vertices connected by rigid
edges.  Bending elasticity comes from angle springs at each interior vertex
that penalise deviations from the natural (planar circular) rest curvature.
A small out-of-plane perturbation is added to the initial state to select the
documented three-dimensional equilibrium branch.  The model is integrated with
a damped explicit time-stepper and edge-length projection, i.e. a physically
real dynamic relaxation toward quasi-static equilibrium under a gravity ramp.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


# Material/freeze parameters not supplied by the scaffold.  These values are
# chosen to give a soft elastomer rod whose final equilibrium shows a clear
# spatial curl under the prescribed gravity ramp.
_YOUNG_MODULUS = 1.5e5  # Pa
_POISSON_RATIO = 0.49
_DAMPING_COEFFICIENT = 2000.0  # 1/s, mass-proportional Rayleigh-like damping
_SUBSTEP_DT = 1e-5  # s
_PBD_ITERATIONS = 5


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)

        # Pull configuration from the scene object (BenchmarkScene stores it).
        self._config = scene.config
        self._rod_cfg = self._config["rod"]

        # Initial state supplied by the scene.
        init_frame = scene.initial_state()
        self._N = int(len(init_frame.position))

        # Positions, velocities, and masses.
        self._x = np.asarray(init_frame.position, dtype=float).copy()
        self._v = np.asarray(init_frame.velocity, dtype=float).copy()
        self._m = np.asarray(init_frame.mass, dtype=float).copy()
        if self._m.shape != (self._N,):
            raise ValueError("mass array must have shape (N,)")

        # Apply the tiny branch-selection perturbation in the y direction.
        # Use a quadratic profile so the root stays in-plane and the perturbation
        # grows smoothly toward the free tip.
        arc = np.asarray(init_frame.attributes["arc_length"], dtype=float)
        length = float(arc[-1]) if arc.size else 1.0
        p_y = float(init_frame.scalars.get("branch_selection_perturbation_y", 0.0))
        if length > 0.0 and p_y != 0.0:
            self._x[:, 1] += p_y * (arc / length) ** 2

        # Root boundary condition: fix the first two vertices so that both the
        # root position and the root tangent are frozen.
        self._fixed = np.zeros(self._N, dtype=bool)
        self._fixed[:2] = True
        # Store the clamped root positions (from the natural arc).
        self._x[0] = init_frame.position[0]
        self._x[1] = init_frame.position[1]
        self._v[self._fixed] = 0.0

        # Inverse mass for PBD constraints (zero for clamped vertices).
        self._inv_mass = np.zeros(self._N, dtype=float)
        self._inv_mass[~self._fixed] = 1.0 / self._m[~self._fixed]

        # Rest geometry of the chain.
        self._edge = self._x[1:] - self._x[:-1]
        self._rest_len = np.linalg.norm(self._edge, axis=1)
        self._rest_len = np.where(self._rest_len < 1e-12, 1e-12, self._rest_len)

        # Material cross-section properties.
        r = float(self._rod_cfg["radius"])
        self._area = np.pi * r * r
        inertia = np.pi * r ** 4 / 4.0
        # Shear modulus for an isotropic, nearly-incompressible material.
        G = _YOUNG_MODULUS / (2.0 * (1.0 + _POISSON_RATIO))
        # Bending/torsion stiffnesses (kept for diagnostics / physical bookkeeping).
        self._bending_stiffness = _YOUNG_MODULUS * inertia
        self._torsion_stiffness = G * (np.pi * r ** 4 / 2.0)

        # Rest curvature expressed as interior vertex angles.
        self._rest_angle = self._compute_angles(self._x)

        # Per-vertex spring constants for the discrete bending springs:
        # k = EI / lbar where lbar is the average adjacent edge length.
        lbar = 0.5 * (self._rest_len[:-1] + self._rest_len[1:])
        lbar = np.where(lbar < 1e-12, 1e-12, lbar)
        self._k_bending = self._bending_stiffness / lbar

        # Gravity vector and load schedule.
        self._g = np.asarray(self._config["gravity"], dtype=float)
        self._load_schedule = self._config["load_schedule"]

        # Keep initial attributes/scalars for trajectory output, adding the
        # required per-sample mass quantity.
        self._attributes = dict(init_frame.attributes)
        self._attributes["mass_quantity"] = self._m.astype(float)
        self._scalars = dict(init_frame.scalars)
        self._scalars["constraint_scale"] = float(length)
        total_weight = float(np.sum(self._m) * np.linalg.norm(self._g))
        self._scalars["reference_force"] = total_weight if total_weight > 0.0 else 1.0
        self._scalars["reference_torque"] = self._scalars["reference_force"] * float(length)

        # Simulation time and cached frame.
        self._time = 0.0
        self._current_load = self._load_factor(self._time)

    def _load_factor(self, t: float) -> float:
        start = float(self._load_schedule["start_time"])
        full = float(self._load_schedule["full_load_time"])
        if t <= start:
            return 0.0
        if t >= full:
            return 1.0
        return (t - start) / (full - start)

    def _compute_angles(self, x: np.ndarray) -> np.ndarray:
        """Interior vertex angles (N-2) for the current centerline."""
        # r1 = x_{i-1} - x_i, r2 = x_{i+1} - x_i.
        r1 = x[:-2] - x[1:-1]
        r2 = x[2:] - x[1:-1]
        n1 = np.linalg.norm(r1, axis=1)
        n2 = np.linalg.norm(r2, axis=1)
        denom = n1 * n2
        denom = np.where(denom < 1e-12, 1e-12, denom)
        cos_theta = np.sum(r1 * r2, axis=1) / denom
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    def _angle_forces(self, x: np.ndarray) -> np.ndarray:
        """Elastic forces from the discrete bending springs (angle formulation)."""
        f = np.zeros_like(x)
        if self._N < 3:
            return f

        r1 = x[:-2] - x[1:-1]          # x_{i-1} - x_i
        r2 = x[2:] - x[1:-1]            # x_{i+1} - x_i
        n1 = np.linalg.norm(r1, axis=1, keepdims=True)
        n2 = np.linalg.norm(r2, axis=1, keepdims=True)
        # Guard against degenerate edges.
        n1 = np.where(n1 < 1e-12, 1e-12, n1)
        n2 = np.where(n2 < 1e-12, 1e-12, n2)

        u1 = r1 / n1
        u2 = r2 / n2
        c = np.sum(u1 * u2, axis=1, keepdims=True)
        c = np.clip(c, -1.0, 1.0)
        s = np.sqrt(np.maximum(1.0 - c * c, 1e-16))

        theta = np.arccos(c)
        dtheta = (theta.squeeze() - self._rest_angle)[:, None]

        # Common scalar prefactors, with regularisation for s -> 0.
        factor1 = (self._k_bending[:, None] * dtheta) / (n1 * (s + 1e-12))
        factor2 = (self._k_bending[:, None] * dtheta) / (n2 * (s + 1e-12))

        # Force on the "left" arm node (i-1) and the "right" arm node (i+1).
        f_left = factor1 * (u2 - c * u1)
        f_right = factor2 * (u1 - c * u2)
        f_center = -(f_left + f_right)

        f[:-2] += f_left
        f[2:] += f_right
        f[1:-1] += f_center
        return f

    def _project_lengths(self, x: np.ndarray) -> None:
        """PBD distance constraints: restore each edge to its rest length."""
        for _ in range(_PBD_ITERATIONS):
            e = x[1:] - x[:-1]
            d = np.linalg.norm(e, axis=1)
            diff = (d - self._rest_len) / np.where(d < 1e-12, 1e-12, d)
            corr = diff[:, None] * e

            im_i = self._inv_mass[:-1]
            im_j = self._inv_mass[1:]
            total = im_i + im_j
            total = np.where(total < 1e-12, 1e-12, total)
            w_i = im_i / total
            w_j = im_j / total

            x[:-1] -= w_i[:, None] * corr
            x[1:] += w_j[:, None] * corr

    def _step(self, dt: float, t: float) -> None:
        """Advance the dynamic relaxation by one substep."""
        # External: gravity ramp + mass-proportional damping.
        load = self._load_factor(t)
        self._current_load = load
        f_grav = (self._m[:, None] * self._g[None, :]) * load
        f_damp = -_DAMPING_COEFFICIENT * self._m[:, None] * self._v

        # Elastic bending forces.
        f_elastic = self._angle_forces(self._x)

        # Total force and explicit velocity update.
        f_total = f_grav + f_damp + f_elastic
        a = f_total / self._m[:, None]
        self._v += dt * a

        # Enforce zero velocity on the clamped root.
        self._v[self._fixed] = 0.0

        # Position update.
        x_old = self._x.copy()
        self._x += dt * self._v

        # Inextensibility projection (keeps the root vertices fixed).
        self._project_lengths(self._x)

        # Recompute velocity from the position change to include constraint
        # effects and remain consistent with the inextensible assumption.
        self._v = (self._x - x_old) / dt
        self._v[self._fixed] = 0.0

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            raise ValueError("advance_to called with non-increasing time")

        # Integrate forward with a fixed substep size, sub-stepping the last
        # interval to land exactly on the requested sample time.
        while self._time < time:
            dt = min(_SUBSTEP_DT, time - self._time)
            t_mid = self._time + 0.5 * dt
            self._step(dt, t_mid)
            self._time += dt

        return self._build_frame(time)

    def _build_frame(self, time: float) -> StateFrame:
        # External force budget at the current sample time.
        load = self._load_factor(time)
        external_force = (self._m[:, None] * self._g[None, :]) * load

        # Budget arrays: one domain (the whole rod) with one term per vertex.
        budget_external_force = external_force[None, :, :].copy()
        budget_force_position = self._x[None, :, :].copy()
        budget_external_couple = np.zeros_like(budget_external_force)

        # Inextensibility constraints: current edge length vs. rest length.
        edges = self._x[1:] - self._x[:-1]
        constraint_value = np.linalg.norm(edges, axis=1)
        constraint_target = self._rest_len.copy()

        fields = {
            "budget_external_force": budget_external_force,
            "budget_external_couple": budget_external_couple,
            "budget_force_position": budget_force_position,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }

        return StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            attributes=self._attributes,
            fields=fields,
            scalars=self._scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "curved_rod_angle_spring_relaxation",
            "young_modulus": _YOUNG_MODULUS,
            "damping_coefficient": _DAMPING_COEFFICIENT,
            "substep_dt": _SUBSTEP_DT,
            "vertices": self._N,
            "current_time": self._time,
            "current_load": self._current_load,
        }
