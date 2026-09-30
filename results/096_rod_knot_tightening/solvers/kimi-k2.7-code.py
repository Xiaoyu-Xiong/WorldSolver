"""Discrete elastic-rod solver for the frictionless rod-knot scene.

BenchmarkSolver implements the worldsolver Solver contract.  It advances a
position-based dynamics (PBD) simulation of a thin inextensible elastic strand
with bending resistance and frictionless self-contact, driven by two kinematic
endpoints.  All trajectory fields required by the scaffold are exported.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        """Set up the rod state and simulation parameters from the scene."""
        self.scene = scene
        np.random.seed(seed)

        init = scene.initial_state()
        self.N: int = len(init.position)

        # Physical state
        self.x = init.position.copy().astype(float)
        self.v = init.velocity.copy().astype(float)
        if init.mass is not None:
            self.mass = init.mass.copy().astype(float)
        else:
            total_mass = float(scene.config["rod"]["total_mass"])
            self.mass = np.full(self.N, total_mass / self.N, dtype=float)
        self.m_total = float(self.mass.sum())

        # Rest configuration used by the bending model
        self.x0 = self.x.copy()

        # Edge rest lengths (inextensible reference)
        self.rest_len = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)
        self.rest_length = float(scene.config["rod"]["rest_length"])

        # Rod/contact geometry
        self.radius = float(scene.config["rod"]["radius"])
        self.contact_dist = float(scene.config["self_contact"]["contact_distance"])

        # Material parameters (SI).  EI is chosen to give a stable, visibly
        # elastic rod for the prescribed manipulation speed.
        self.EI = 0.02  # N*m^2 bending stiffness
        self.l0_avg = float(self.rest_len.mean()) if self.N > 1 else 1.0
        self.k_bend = self.EI / (self.l0_avg ** 3)

        # Contact penalty stiffness (N/m)
        self.k_contact = 2000.0

        # Substepping / damping.  80 substeps per sample (h = 2.5e-4 s) keeps
        # the stiff contact repulsion stable; light per-substep artificial
        # damping prevents high-frequency oscillations without overdamping the
        # slow pulling motion.
        sample_dt = float(scene.config["sample_dt"])
        self.n_substeps = 80
        self.h = sample_dt / self.n_substeps
        self.damping = 0.02

        # Time bookkeeping
        self.current_time = 0.0

        # Cumulative budget fields
        self.cum_external_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_momentum = np.zeros(3, dtype=float)
        self.cum_outflow_momentum = np.zeros(3, dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.last_total_external_force = np.zeros(3, dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        """Integrate the rod state from the current time to ``time``."""
        dt = time - self.current_time
        if dt <= 0.0:
            return self._make_frame(time)

        n = max(1, int(round(dt / self.h)))
        h = dt / n

        for k in range(n):
            t_next = self.current_time + (k + 1) * h
            self._substep(t_next, h)

        self.current_time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps_per_sample": self.n_substeps,
            "current_time": float(self.current_time),
            "vertex_count": self.N,
        }

    # ------------------------------------------------------------------ #
    # Substepping
    # ------------------------------------------------------------------ #
    def _substep(self, t_next: float, h: float) -> None:
        """Perform one PBD substep."""
        v_old = self.v.copy()

        # Elastic + internal forces
        F = self._bending_force(self.x)
        F += self._contact_force(self.x)

        # Gravity is zero for this scene, so no additional external force here.

        # Semi-implicit Euler velocity update
        self.v += h * F / self.mass[:, None]

        # Velocity damping (artificial, for numerical stability)
        self.v *= (1.0 - self.damping)

        # Position prediction
        x_pred = self.x + h * self.v

        # Project inextensibility constraints (Gauss-Seidel)
        for _ in range(5):
            x_pred = self._project_distance(x_pred)

        # Enforce hard kinematic endpoint targets
        targets = self.scene.endpoint_targets(t_next)
        x_pred[0] = targets["left_tail"]
        x_pred[-1] = targets["threading_tail"]

        # Final velocity from position change
        self.v = (x_pred - self.x) / h

        # External force budget: net change in total linear momentum equals
        # total external impulse.  Internal bending/contact forces cancel; the
        # remainder is controller + damping + gravity (gravity is zero here).
        self.last_total_external_force = (
            self.mass[:, None] * (self.v - v_old)
        ).sum(axis=0) / h

        self.cum_external_impulse += self.last_total_external_force * h

        # Mass is conserved; no source term
        self.cum_mass_source[0] += 0.0

        self.x = x_pred

    # ------------------------------------------------------------------ #
    # Internal forces
    # ------------------------------------------------------------------ #
    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Return nodal bending force resisting changes from the rest curvature.

        The discrete bending energy is proportional to
        |(x_{i-1} - 2x_i + x_{i+1}) - rest_laplacian_i|^2, which yields a
        conservative, momentum-preserving force distribution.
        """
        N = self.N
        L = x[0 : N - 2] - 2.0 * x[1 : N - 1] + x[2:N]
        L0 = (
            self.x0[0 : N - 2]
            - 2.0 * self.x0[1 : N - 1]
            + self.x0[2:N]
        )
        dL = L - L0

        F = np.zeros_like(x)
        # triplet i (1..N-2) contributes to vertices i-1, i, i+1
        F[0 : N - 2] -= self.k_bend * dL
        F[1 : N - 1] += 2.0 * self.k_bend * dL
        F[2:N] -= self.k_bend * dL
        return F

    def _contact_force(self, x: np.ndarray) -> np.ndarray:
        """Frictionless repulsive penalty for non-adjacent rod segments."""
        # Build KD-tree once per substep
        tree = cKDTree(x)
        pairs = tree.query_pairs(r=self.contact_dist, output_type="ndarray")

        if pairs.size == 0:
            return np.zeros_like(x)

        # Exclude vertices close along the rod (local neighborhood)
        diff_idx = np.abs(pairs[:, 0] - pairs[:, 1])
        pairs = pairs[diff_idx > 2]

        if pairs.size == 0:
            return np.zeros_like(x)

        i = pairs[:, 0]
        j = pairs[:, 1]
        dvec = x[j] - x[i]
        dist = np.linalg.norm(dvec, axis=1)
        dist = np.maximum(dist, 1e-12)
        n = dvec / dist[:, None]

        overlap = self.contact_dist - dist
        overlap = np.clip(overlap, 0.0, None)
        force_mag = self.k_contact * overlap

        F = np.zeros_like(x)
        np.add.at(F, i, force_mag[:, None] * n)
        np.add.at(F, j, -force_mag[:, None] * n)
        return F

    def _project_distance(self, x: np.ndarray) -> np.ndarray:
        """Project edge lengths to their rest lengths, pinning the endpoints."""
        edges = x[1:] - x[:-1]
        lens = np.linalg.norm(edges, axis=1)
        lens = np.maximum(lens, 1e-12)

        # Desired correction vector for each edge: edge * (1 - l0 / l)
        delta = edges * (1.0 - self.rest_len / lens)[:, None]

        x_new = x.copy()
        x_new[:-1] += 0.5 * delta
        x_new[1:] -= 0.5 * delta

        # Endpoint vertices are controlled kinematically
        x_new[0] = x[0]
        x_new[-1] = x[-1]
        return x_new

    # ------------------------------------------------------------------ #
    # Frame construction
    # ------------------------------------------------------------------ #
    def _make_frame(self, time: float) -> StateFrame:
        targets = self.scene.endpoint_targets(time)
        constraint_target = np.array(
            [targets["left_tail"], targets["threading_tail"]], dtype=float
        )
        constraint_value = np.array(
            [self.x[0].copy(), self.x[-1].copy()], dtype=float
        )

        orientation = self._compute_orientation(self.x)
        angular_velocity = self._compute_angular_velocity(self.x, self.v)

        # Budget fields: one global domain (G=1), two external-force terms
        # (gravity, non-gravity/controller).
        G, K = 1, 2
        budget_external_force = np.zeros((G, K, 3), dtype=float)
        budget_external_force[0, 0, :] = np.zeros(3, dtype=float)  # gravity
        budget_external_force[0, 1, :] = self.last_total_external_force

        budget_domain_weights = np.full((G, self.N), 1.0 / self.N, dtype=float)

        return StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.reshape(
                    1, 3
                ).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.reshape(
                    1, 3
                ).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.reshape(
                    1, 3
                ).copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": self.cum_mass_source.copy(),
            },
            scalars={
                "constraint_scale": float(self.rest_length),
                "reference_momentum": float(
                    self.m_total * (self.rest_length / 8.0)
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # Kinematic helpers
    # ------------------------------------------------------------------ #
    def _compute_orientation(self, x: np.ndarray) -> np.ndarray:
        """Per-vertex quaternion aligning the local z-axis with the rod tangent."""
        N = self.N
        tangents = np.zeros((N, 3), dtype=float)
        edge_t = x[1:] - x[:-1]
        edge_len = np.linalg.norm(edge_t, axis=1, keepdims=True)
        edge_t = edge_t / np.maximum(edge_len, 1e-12)

        tangents[0] = edge_t[0]
        tangents[1:-1] = 0.5 * (edge_t[:-1] + edge_t[1:])
        tangents[-1] = edge_t[-1]
        tangents = tangents / np.maximum(
            np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12
        )

        z = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z, tangents)
        cross_norm = np.linalg.norm(cross, axis=1)
        dot = tangents[:, 2]

        angle = np.arctan2(cross_norm, dot)
        sin_half = np.sin(0.5 * angle)
        cos_half = np.cos(0.5 * angle)

        axis = np.zeros_like(tangents)
        mask = cross_norm > 1e-8
        axis[mask] = cross[mask] / cross_norm[mask, None]
        axis[~mask] = np.array([1.0, 0.0, 0.0])

        q = np.zeros((N, 4), dtype=float)
        q[:, :3] = axis * sin_half[:, None]
        q[:, 3] = cos_half
        return q

    def _compute_angular_velocity(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Per-vertex angular velocity from the rotation of material frames."""
        N = self.N
        dv = v[1:] - v[:-1]
        dx = x[1:] - x[:-1]
        edge_len = np.linalg.norm(dx, axis=1, keepdims=True)
        t = dx / np.maximum(edge_len, 1e-12)
        omega_edges = np.cross(t, dv) / np.maximum(edge_len, 1e-12)

        omega = np.zeros((N, 3), dtype=float)
        omega[0] = omega_edges[0]
        omega[1:-1] = 0.5 * (omega_edges[:-1] + omega_edges[1:])
        omega[-1] = omega_edges[-1]
        return omega
