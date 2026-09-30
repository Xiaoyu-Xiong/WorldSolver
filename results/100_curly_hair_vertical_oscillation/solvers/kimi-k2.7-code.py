"""Solver Agent-owned implementation for the curly hair vertical oscillation scene.

The clump is modelled as a single inextensible discrete elastic rod.  Gravity,
ambient drag, internal damping and bending stiffness act on the nodes, while the
root node is driven by the prescribed sinusoidal vertical motion.  Inextensibility
is enforced with a position-based projection on every edge.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        # Deterministic setup (numpy is only used numerically, not for RNG here)
        self.rng = np.random.default_rng(seed)

        # Pull parameters from the frozen scene
        self.cfg = scene.config
        guide = self.cfg["physical_guide"]
        root_motion = self.cfg["root_motion"]

        # Initial state from the scene (root is at rest at t=0)
        init = scene.initial_state()
        self.N = init.position.shape[0]
        self.position: NDArray[np.float64] = np.asarray(init.position, dtype=float).copy()
        self.velocity: NDArray[np.float64] = np.asarray(init.velocity, dtype=float).copy()
        self.mass: NDArray[np.float64] = np.asarray(init.mass, dtype=float).copy()

        # Cached original attributes/scalars so every frame can carry them
        self._base_attributes = {k: np.asarray(v).copy() for k, v in init.attributes.items()}
        self._base_scalars = dict(init.scalars)

        # Reference geometry (frozen curly shape)
        self.rest_length = np.linalg.norm(np.diff(self.position, axis=0), axis=1).astype(float)
        self.bend_dot0 = self._compute_bend_reference(self.position)

        # Physical parameters
        self.gravity_vec = np.asarray(self.cfg["gravity"], dtype=float)
        self.ambient_drag = float(guide["ambient_drag_nu"])
        self.internal_damping = float(guide["internal_damping_gamma"])
        # Bending stiffness chosen to keep the curly shape while remaining stable
        # with the explicit substep used below.
        self.bend_stiffness = 10.0

        # Root motion prescription
        self.root_rest = np.asarray(root_motion["rest_position"], dtype=float)
        self.root_axis = np.asarray(root_motion["axis"], dtype=float)
        self.root_start = float(root_motion["start_time"])
        self.root_amp = float(root_motion["amplitude"])
        self.root_freq = float(root_motion["frequency"])
        self.root_phase = float(root_motion.get("phase", 0.0))

        # Time-stepping
        self.t: float = 0.0
        self.h: float = 5.0e-5

        # Budget / bookkeeping
        self.domain_weights = np.ones((1, self.N), dtype=float)
        self.cum_external_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow = np.zeros((1, 3), dtype=float)
        self.cum_outflow = np.zeros((1, 3), dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.budget_external_force = np.zeros((1, 3, 3), dtype=float)

        # Constraint bookkeeping (edge lengths constraints)
        self.constraint_target = self.rest_length.copy()
        self.constraint_value = self.rest_length.copy()
        self.constraint_scale = float(self.rest_length.mean())

        total_mass = float(self.mass.sum())
        mat_length = float(self._base_scalars.get("material_length", 0.5))
        # Positive reference momentum scale used by the platform
        self.reference_momentum = max(
            1.0e-6,
            total_mass * np.sqrt(9.81 * max(mat_length, 1.0e-6)),
        )

    def _compute_bend_reference(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reference dot products between adjacent edges in the rest shape."""
        if self.N < 3:
            return np.array([], dtype=float)
        e = pos[1:] - pos[:-1]
        return np.einsum("ij,ij->i", e[:-1], e[1:])

    def _root_position(self, t: float) -> NDArray[np.float64]:
        p = self.root_rest.copy()
        if t >= self.root_start:
            phase = 2.0 * np.pi * self.root_freq * (t - self.root_start) + self.root_phase
            displacement = self.root_amp * np.sin(phase)
            p += displacement * self.root_axis
        return p

    def _root_velocity(self, t: float) -> NDArray[np.float64]:
        v = np.zeros(3, dtype=float)
        if t >= self.root_start:
            phase = 2.0 * np.pi * self.root_freq * (t - self.root_start) + self.root_phase
            v = self.root_amp * 2.0 * np.pi * self.root_freq * np.cos(phase) * self.root_axis
        return v

    def _project_lengths(self, root_target: NDArray[np.float64]) -> None:
        """Position-based inextensibility projection (keeps root node fixed)."""
        # Pin the root before projection; its inverse weight is zero.
        self.position[0] = root_target

        inv_mass = np.zeros(self.N, dtype=float)
        inv_mass[1:] = 1.0 / self.mass[1:]

        for _ in range(5):
            d = self.position[1:] - self.position[:-1]
            len_d = np.linalg.norm(d, axis=1)
            # Avoid division by zero for degenerate edges
            mask = len_d > 1.0e-12
            if not mask.any():
                break
            n = np.zeros_like(d)
            n[mask] = d[mask] / len_d[mask, None]
            delta = len_d - self.rest_length

            w_sum = inv_mass[:-1] + inv_mass[1:]
            # Only resolve edges that have at least one free endpoint
            active = (w_sum > 0.0) & mask

            factor_im1 = np.zeros(self.N - 1, dtype=float)
            factor_i = np.zeros(self.N - 1, dtype=float)
            factor_im1[active] = inv_mass[:-1][active] / w_sum[active]
            factor_i[active] = inv_mass[1:][active] / w_sum[active]

            corr = delta[:, None] * n
            self.position[1:] -= factor_i[:, None] * corr
            self.position[:-1] += factor_im1[:, None] * corr

        self.position[0] = root_target

    def _step(self, dt: float) -> None:
        pos_old = self.position.copy()
        N = self.N

        # Gravity
        f_gravity = self.mass[:, None] * self.gravity_vec[None, :]

        # Ambient drag (per-node, weighted by half of adjacent edge lengths)
        seg_weight = 0.5 * (
            np.concatenate(([0.0], self.rest_length))
            + np.concatenate((self.rest_length, [0.0]))
        )
        f_drag = -self.ambient_drag * self.velocity * seg_weight[:, None]
        # Root is a kinematic sample, not physical mass
        f_gravity[0] = 0.0
        f_drag[0] = 0.0

        # Internal damping along edges
        dv_edges = self.velocity[1:] - self.velocity[:-1]
        f_damp = np.zeros_like(self.velocity)
        f_damp[1:] -= self.internal_damping * dv_edges
        f_damp[:-1] += self.internal_damping * dv_edges

        # Bending forces from adjacent edge pairs (preserves the reference curl)
        f_bend = np.zeros_like(self.velocity)
        if N > 2:
            e = self.position[1:] - self.position[:-1]
            g_cur = np.einsum("ij,ij->i", e[:-1], e[1:])
            delta = g_cur - self.bend_dot0
            common = self.bend_stiffness * delta  # (N-2,)

            f_bend[0 : N - 2] += common[:, None] * e[1:]
            f_bend[2:N] -= common[:, None] * e[:-1]
            f_bend[1 : N - 1] += common[:, None] * (e[:-1] - e[1:])

        # Root support reaction needed to enforce the prescribed root motion
        f_root_drive = np.zeros_like(self.velocity)
        f_root_drive[0] = -(f_bend[0] + f_damp[0])

        # Budget: external-force terms on the single hair domain
        f_grav_domain = f_gravity.sum(axis=0, keepdims=True)
        f_drag_domain = f_drag.sum(axis=0, keepdims=True)
        f_root_drive_domain = f_root_drive.sum(axis=0, keepdims=True)
        self.budget_external_force = np.stack(
            [f_grav_domain, f_drag_domain, f_root_drive_domain], axis=1
        )  # shape (1, 3, 3)
        self.cum_external_impulse += (
            f_grav_domain + f_drag_domain + f_root_drive_domain
        ) * dt

        # Integrate velocities (symplectic Euler), then positions
        f_total = f_gravity + f_drag + f_damp + f_bend + f_root_drive
        inv_mass = np.zeros(self.N, dtype=float)
        inv_mass[1:] = 1.0 / self.mass[1:]
        self.velocity += dt * f_total * inv_mass[:, None]

        # Integrate positions
        self.position += dt * self.velocity

        # Enforce root kinematics at the new time
        t_new = self.t + dt
        self.position[0] = self._root_position(t_new)
        self.velocity[0] = self._root_velocity(t_new)

        # Inextensibility projection
        self._project_lengths(self.position[0].copy())

        # Recompute velocity from the actual position change so constraints are
        # reflected in the sampled velocity field; keep root kinematic.
        v_new = (self.position - pos_old) / dt
        v_new[0] = self.velocity[0]
        self.velocity = v_new

        self.t = t_new

        # Update sampled inextensibility constraint value
        edges = self.position[1:] - self.position[:-1]
        self.constraint_value = np.linalg.norm(edges, axis=1)

    def _make_frame(self, time: float) -> StateFrame:
        attributes = dict(self._base_attributes)
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.budget_external_force.copy(),
            "constraint_target": self.constraint_target.copy(),
            "constraint_value": self.constraint_value.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
        }

        scalars = dict(self._base_scalars)
        scalars["constraint_scale"] = float(self.constraint_scale)
        scalars["reference_momentum"] = float(self.reference_momentum)

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            time = self.t

        # Step with the fixed substep, using a smaller step only for the last
        # partial interval if the requested time is not an exact multiple.
        while self.t < time - 0.5 * self.h:
            dt = min(self.h, time - self.t)
            self._step(dt)

        # Align the kinematic root to the exact requested time
        if self.t != time:
            self.position[0] = self._root_position(time)
            self.velocity[0] = self._root_velocity(time)
            self.t = time

        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "curly_hair_pbd_rod",
            "nodes": self.N,
            "substep": self.h,
            "final_time": self.t,
        }
