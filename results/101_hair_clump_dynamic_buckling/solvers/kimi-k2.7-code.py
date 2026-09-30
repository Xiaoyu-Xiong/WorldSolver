"""Physical solver for the dynamic buckling of a vertically shaken hair clump.

The clump is modelled as a discrete inextensible elastic rod (a single strand
of 49 nodes).  Inextensibility is enforced with position-based distance
constraints, bending resistance is provided by a discrete curvature spring,
and the root is driven by the prescribed vertical sinusoid after an initial
hold.  Gravity, ambient drag and a tiny internal damping term act on every
free node.  The simulation is advanced by a stable sub-stepped semi-implicit
Euler scheme.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver producing deterministic StateFrame samples for the hair-clump scene."""

    # Numerical settings (tuned for stability and the required dynamic event).
    SUBSTEPS_PER_SAMPLE = 200
    PBD_ITERATIONS = 10
    BENDING_STIFFNESS = 4000.0  # N/m, effective discrete-bending coefficient

    def __init__(self) -> None:
        self.scene = None
        self.config: dict = {}
        self.t: float = 0.0
        self.dt_sub: float = 0.0
        self.N: int = 0
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.m: np.ndarray | None = None
        self.fixed: np.ndarray | None = None
        self.rest_len: np.ndarray | None = None
        self.drag_coeff: np.ndarray | None = None
        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.initial_attributes: dict = {}
        self.initial_scalars: dict = {}

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        # The scene object keeps the parsed configuration.
        self.config = getattr(scene, "config", {})
        if not self.config:
            raise RuntimeError("Scene object does not expose config")

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        hair = self.config["hair_clump"]
        self.L = float(hair["material_length"])
        self.radius = float(hair["effective_radius"])
        self.density = float(hair["density"])
        self.internal_damping_gamma = float(hair["internal_damping_gamma"])
        self.ambient_drag_nu = float(hair["ambient_drag_nu"])
        self.ambient_air_velocity = np.asarray(hair["ambient_air_velocity"], dtype=float)

        drive = self.config["root_motion"]
        self.root_start = float(drive["start_time"])
        self.root_amp = float(drive["amplitude"])
        self.root_freq = float(drive["frequency"])
        self.root_phase = float(drive["phase"])
        self.root_axis = np.asarray(drive["axis"], dtype=float)
        self.root_rest = np.asarray(drive["rest_position"], dtype=float)
        self.root_orientation_fixed = bool(drive.get("root_orientation_fixed", True))

        self.sample_dt = float(self.config["sample_dt"])
        self.dt_sub = self.sample_dt / self.SUBSTEPS_PER_SAMPLE
        self.duration = float(self.config["duration"])

        init = scene.initial_state()
        self.x = np.asarray(init.position, dtype=float).copy()
        self.v = np.asarray(init.velocity, dtype=float).copy()
        self.m = np.asarray(init.mass, dtype=float).copy()
        self.initial_attributes = {
            k: np.asarray(v).copy() for k, v in init.attributes.items()
        }
        self.initial_scalars = dict(init.scalars)

        self.N = len(self.x)
        self.h = self.L / (self.N - 1)

        # Fixed nodes: root node and, for a clamped orientation, the node directly
        # above it along the rest tangent (vertical z-axis).
        self.fixed = np.zeros(self.N, dtype=bool)
        self.fixed[0] = True
        if self.root_orientation_fixed:
            self.fixed[1] = True

        # Rest edge lengths from the provided initial geometry.
        self.rest_len = np.linalg.norm(np.diff(self.x, axis=0), axis=1)

        # Ambient drag coefficient per node, c_i = nu * attached_length, with
        # half-length contributions at the two ends.  Fixed nodes get zero drag
        # because they are kinematic.
        seg_len = np.full(self.N, self.h, dtype=float)
        seg_len[0] = self.h * 0.5
        seg_len[-1] = self.h * 0.5
        self.drag_coeff = self.ambient_drag_nu * seg_len
        self.drag_coeff[self.fixed] = 0.0

        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.t = 0.0

    def advance_to(self, time: float) -> StateFrame:
        if time + 1e-12 < self.t:
            raise ValueError(f"Requested time {time} is before current time {self.t}")
        target = float(time)
        while self.t < target - 1e-12:
            dt = min(self.dt_sub, target - self.t)
            self._substep(dt)
            self.t += dt
        return self._make_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps_per_sample": self.SUBSTEPS_PER_SAMPLE,
            "pbd_iterations": self.PBD_ITERATIONS,
            "bending_stiffness": self.BENDING_STIFFNESS,
            "current_time": self.t,
        }

    # --------------------------------------------------------------------- #
    # Physics kernel
    # --------------------------------------------------------------------- #
    def _substep(self, dt: float) -> None:
        t_new = self.t + dt
        root_pos = self._root_position(t_new)
        root_vel = self._root_velocity(t_new)

        x_old = self.x.copy()

        # External and internal forces.
        f = np.zeros_like(self.x)
        # Gravity.
        f += self.m[:, None] * self.gravity
        # Ambient drag.
        f -= self.drag_coeff[:, None] * (self.v - self.ambient_air_velocity)
        # Internal velocity-proportional damping (volumetric).
        node_volume = self.m / self.density
        f -= self.internal_damping_gamma * node_volume[:, None] * self.v
        # Discrete bending resistance (midpoint-spring model).
        f += self._bending_force(self.x)

        # Semi-implicit Euler update for free nodes.
        free_idx = np.where(~self.fixed)[0]
        self.v[free_idx] += dt * f[free_idx] / self.m[free_idx][:, None]
        # Keep a copy of the pre-advection positions to correctly measure the
        # post-constraint displacement velocity after PBD.
        x_pre_advect = self.x.copy()
        self.x[free_idx] += dt * self.v[free_idx]

        # Enforce root boundary conditions.
        self.x[0] = root_pos
        if self.root_orientation_fixed:
            # First edge remains vertical.
            self.x[1] = root_pos + np.array([0.0, 0.0, self.h], dtype=float)

        # Project onto the inextensibility manifold (PBD distance constraints).
        self._pbd_constraints()

        # Derive post-constraint velocities from the actual displacement.
        self.v = (self.x - x_pre_advect) / dt
        self.v[self.fixed] = 0.0
        self.v[0] = root_vel
        if self.root_orientation_fixed:
            self.v[1] = root_vel

        # Update momentum-budget cumulative impulse with the current external
        # force sample (gravity + drag on the free part of the rod).
        ext_force = self.m[:, None] * self.gravity - self.drag_coeff[:, None] * (
            self.v - self.ambient_air_velocity
        )
        self.cum_ext_impulse += np.sum(ext_force, axis=0) * dt

    def _root_position(self, t: float) -> np.ndarray:
        pos = self.root_rest.copy()
        if t >= self.root_start:
            phase = 2.0 * math.pi * self.root_freq * (t - self.root_start) + self.root_phase
            pos = pos + self.root_amp * np.sin(phase) * self.root_axis
        return pos

    def _root_velocity(self, t: float) -> np.ndarray:
        if t < self.root_start:
            return np.zeros(3, dtype=float)
        phase = 2.0 * math.pi * self.root_freq * (t - self.root_start) + self.root_phase
        return self.root_amp * 2.0 * math.pi * self.root_freq * np.cos(phase) * self.root_axis

    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Return the nodal force from the discrete midpoint-spring bending energy.

        The energy is (k_b/2) * sum_i |x_i - (x_{i-1}+x_{i+1})/2|^2, which
        produces a force that resists curvature.
        """
        f = np.zeros_like(x)
        # d_i = x_i - (x_{i-1}+x_{i+1})/2 for i = 1 .. N-2
        d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
        # Force on centre nodes.
        f[1:-1] = -self.BENDING_STIFFNESS * d
        # Reactions on the two neighbours.
        f[:-2] += 0.5 * self.BENDING_STIFFNESS * d
        f[2:] += 0.5 * self.BENDING_STIFFNESS * d
        # Fixed nodes supply boundary reactions and are not integrated.
        f[self.fixed] = 0.0
        return f

    def _pbd_constraints(self) -> None:
        """Gauss-Seidel projection onto fixed edge lengths."""
        x = self.x
        fixed = self.fixed
        for _ in range(self.PBD_ITERATIONS):
            for i in range(self.N - 1):
                if fixed[i] and fixed[i + 1]:
                    continue
                dx = x[i + 1] - x[i]
                l = float(np.linalg.norm(dx))
                if l < 1e-12:
                    continue
                diff = (l - self.rest_len[i]) / l
                w_i = 0.0 if fixed[i] else 1.0
                w_j = 0.0 if fixed[i + 1] else 1.0
                w = w_i + w_j
                if w == 0.0:
                    continue
                corr = diff * dx
                if w_i > 0.0:
                    x[i] += (w_i / w) * corr
                if w_j > 0.0:
                    x[i + 1] -= (w_j / w) * corr

    # --------------------------------------------------------------------- #
    # StateFrame assembly
    # --------------------------------------------------------------------- #
    def _make_frame(self, time: float) -> StateFrame:
        # Edge length inextensibility constraint value and target.
        edge_vecs = np.diff(self.x, axis=0)
        edge_len = np.linalg.norm(edge_vecs, axis=1)
        constraint_value = edge_len - self.rest_len
        constraint_target = np.zeros_like(constraint_value)

        # Domain weights: one global domain covering all nodes.
        domain_weights = np.ones((1, self.N), dtype=float)

        # External force terms for the single domain: gravity and drag.
        gravity_force = self.m[:, None] * self.gravity
        drag_force = -self.drag_coeff[:, None] * (self.v - self.ambient_air_velocity)
        budget_external_force = np.stack(
            [np.sum(gravity_force, axis=0), np.sum(drag_force, axis=0)], axis=0
        )[None, :, :]  # shape (1, 2, 3)

        # Cumulative inflow/outflow momentum are zero (closed system, no mass flux).
        cum_inflow = np.zeros((1, 3), dtype=float)
        cum_outflow = np.zeros((1, 3), dtype=float)

        # Mass source is zero for the single conserved material.
        cumulative_mass_source = np.zeros(1, dtype=float)

        # Per-sample mass quantity (single material, shape [N, 1]).
        mass_quantity = self.m[:, None]

        attributes = {
            **{k: np.asarray(v).copy() for k, v in self.initial_attributes.items()},
            "mass_quantity": mass_quantity.copy(),
        }

        scalars = {
            **self.initial_scalars,
            "constraint_scale": float(np.mean(self.rest_len)),
            "reference_momentum": 1.0,
        }

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": cum_outflow.copy(),
            "budget_domain_weights": domain_weights.copy(),
            "budget_external_force": budget_external_force.copy(),
            "constraint_target": constraint_target.copy(),
            "constraint_value": constraint_value.copy(),
            "cumulative_mass_source": cumulative_mass_source.copy(),
        }

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes=attributes,
            scalars=scalars,
            fields=fields,
        )
        frame.validate()
        return frame
