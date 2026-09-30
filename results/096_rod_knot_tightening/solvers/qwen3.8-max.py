"""Elastic rod knot-tightening solver.

Implements a discrete elastic rod with bending resistance, inextensibility
constraints (via penalty springs), and frictionless self-contact repulsion.
Endpoint vertices are kinematically driven by smoothstep-interpolated waypoints
from the scene configuration. Time integration uses semi-implicit Euler with
substeps for stability.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete elastic rod solver for knot tightening."""

    def initialize(self, scene: Any, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        cfg = scene.metadata()
        self._cfg = cfg

        # Rod parameters
        rod_cfg = cfg["rod"]
        self.n_verts = int(rod_cfg["vertex_count"])
        self.radius = float(rod_cfg["radius"])
        self.total_mass = float(rod_cfg["total_mass"])
        self.rest_length = float(rod_cfg["rest_length"])
        self.mass_per_vert = self.total_mass / self.n_verts

        # Derived quantities
        self.n_edges = self.n_verts - 1
        self.rest_edge_len = self.rest_length / self.n_edges

        # Self-contact parameters
        contact_cfg = cfg["self_contact"]
        self.contact_enabled = bool(contact_cfg.get("enabled", True))
        self.contact_distance = float(contact_cfg.get("contact_distance", 2 * self.radius))
        self.friction_coeff = float(contact_cfg.get("friction_coefficient", 0.0))

        # Load initial state from scene
        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64).copy()
        self.vel = np.array(init_frame.velocity, dtype=np.float64).copy()
        self.orientation = np.array(init_frame.orientation, dtype=np.float64).copy() if init_frame.orientation is not None else np.tile([0.0, 0.0, 0.0, 1.0], (self.n_verts, 1))
        self.angular_vel = np.array(init_frame.angular_velocity, dtype=np.float64).copy() if init_frame.angular_velocity is not None else np.zeros_like(self.pos)
        self.mass_arr = np.full(self.n_verts, self.mass_per_vert, dtype=np.float64)

        # Compute rest edge vectors and lengths from initial config
        self._compute_rest_geometry()

        # Bending stiffness: EI for circular cross-section
        # Use moderate stiffness for visible bending resistance
        E_modulus = 5.0e4  # Pa-like scale
        I_moment = math.pi * self.radius**4 / 4.0
        self.EI = E_modulus * I_moment

        # Inextensibility penalty stiffness
        self.stretch_stiffness = 1.0e6

        # Damping
        self.damping = 2.0

        # Contact repulsion stiffness
        self.contact_stiffness = 5.0e4

        # Time tracking
        self.current_time = 0.0

        # Budget accumulators
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.prev_endpoint_momentum = self._endpoint_momentum()

        # Endpoint control indices
        ctrl = cfg["endpoint_manipulation"]["controlled_vertices"]
        self.ctrl_left = int(ctrl[0])
        self.ctrl_right = int(ctrl[1])

    def _compute_rest_geometry(self) -> None:
        """Compute rest edge vectors and bending reference from initial positions."""
        edges = self.pos[1:] - self.pos[:-1]
        self.rest_edge_vecs = edges.copy()
        self.rest_edge_lens = np.linalg.norm(edges, axis=1)
        # Normalize to unit rest directions
        norms = self.rest_edge_lens[:, None]
        norms = np.maximum(norms, 1e-12)
        self.rest_edge_dirs = edges / norms

    def _smoothstep(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _waypoint_position(self, waypoints: list[dict], time: float) -> np.ndarray:
        if time <= float(waypoints[0]["time"]):
            return np.asarray(waypoints[0]["position"], dtype=np.float64)
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            ta, tb = float(a["time"]), float(b["time"])
            if time <= tb:
                u = self._smoothstep((time - ta) / max(tb - ta, 1e-12))
                pa = np.asarray(a["position"], dtype=np.float64)
                pb = np.asarray(b["position"], dtype=np.float64)
                return (1.0 - u) * pa + u * pb
        return np.asarray(waypoints[-1]["position"], dtype=np.float64)

    def _get_endpoint_targets(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        ep = self._cfg["endpoint_manipulation"]
        left = self._waypoint_position(ep["left_tail"]["waypoints"], time)
        right = self._waypoint_position(ep["threading_tail"]["waypoints"], time)
        return left, right

    def _endpoint_momentum(self) -> np.ndarray:
        m = self.mass_arr[self.ctrl_left] * self.vel[self.ctrl_left]
        m += self.mass_arr[self.ctrl_right] * self.vel[self.ctrl_right]
        return m

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray, time: float) -> np.ndarray:
        """Compute internal elastic forces, damping, and contact repulsion."""
        n = len(pos)
        force = np.zeros_like(pos)

        # --- Stretch/inextensibility penalty forces ---
        edges = pos[1:] - pos[:-1]
        edge_lens = np.linalg.norm(edges, axis=1)
        edge_lens_safe = np.maximum(edge_lens, 1e-12)
        edge_dirs = edges / edge_lens_safe[:, None]

        stretch_strain = (edge_lens - self.rest_edge_lens) / self.rest_edge_lens
        stretch_force_mag = self.stretch_stiffness * stretch_strain * self.rest_edge_lens

        # Force on vertex i from edge (i, i+1): pulls toward rest length
        for i in range(self.n_edges):
            f = stretch_force_mag[i] * edge_dirs[i]
            force[i] += f
            force[i + 1] -= f

        # --- Bending forces (discrete curvature) ---
        # For each interior vertex i (1..n-2), compute bending moment from
        # adjacent edges and convert to equivalent nodal forces.
        for i in range(1, n - 1):
            e_prev = pos[i] - pos[i - 1]
            e_next = pos[i + 1] - pos[i]
            l_prev = max(np.linalg.norm(e_prev), 1e-12)
            l_next = max(np.linalg.norm(e_next), 1e-12)
            t_prev = e_prev / l_prev
            t_next = e_next / l_next

            # Discrete curvature vector: change in tangent / average edge length
            avg_len = 0.5 * (l_prev + l_next)
            kappa_vec = (t_next - t_prev) / avg_len

            # Bending force: -EI * laplacian(kappa) approximated as
            # F_i = EI * (kappa_{i+1} - 2*kappa_i + kappa_{i-1}) / avg_len^2
            # Simplified: apply restoring force proportional to curvature deviation
            bend_force = -self.EI * kappa_vec / max(avg_len, 1e-12)

            force[i] += bend_force
            # Distribute reaction to neighbors
            force[i - 1] -= 0.5 * bend_force
            force[i + 1] -= 0.5 * bend_force

        # --- Velocity damping ---
        force -= self.damping * vel

        # --- Frictionless self-contact repulsion ---
        if self.contact_enabled:
            self._apply_contact_repulsion(pos, force)

        return force

    def _apply_contact_repulsion(self, pos: np.ndarray, force: np.ndarray) -> None:
        """Apply soft repulsion between non-adjacent segments that violate contact distance."""
        n = len(pos)
        min_sep = self.contact_distance
        # Only check pairs separated by at least 3 vertices to avoid adjacent interference
        skip = 4
        for i in range(n):
            for j in range(i + skip, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < min_sep and dist > 1e-12:
                    overlap = min_sep - dist
                    direction = diff / dist
                    f_mag = self.contact_stiffness * overlap
                    force[i] += f_mag * direction
                    force[j] -= f_mag * direction

    def _substep(self, dt: float, time: float) -> None:
        """One semi-implicit Euler substep."""
        n = self.n_verts
        force = self._compute_forces(self.pos, self.vel, time)

        # Semi-implicit Euler: update velocity first, then position
        acc = force / self.mass_arr[:, None]
        self.vel += dt * acc
        self.pos += dt * self.vel

        # Enforce endpoint kinematic constraints
        left_target, right_target = self._get_endpoint_targets(time)
        self.pos[self.ctrl_left] = left_target
        self.pos[self.ctrl_right] = right_target

        # Set endpoint velocities to match target motion (finite difference)
        eps = 1e-6
        left_next, right_next = self._get_endpoint_targets(time + eps)
        self.vel[self.ctrl_left] = (left_next - left_target) / eps
        self.vel[self.ctrl_right] = (right_next - right_target) / eps

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the given time and return a StateFrame."""
        dt_max = 0.0005  # Max substep size for stability
        total_dt = time - self.current_time
        if total_dt > 0:
            n_substeps = max(1, int(math.ceil(total_dt / dt_max)))
            sub_dt = total_dt / n_substeps
            for _ in range(n_substeps):
                self._substep(sub_dt, self.current_time)
                self.current_time += sub_dt

        # Update budget accumulators
        curr_endpoint_mom = self._endpoint_momentum()
        dp = curr_endpoint_mom - self.prev_endpoint_momentum
        # External impulse from endpoint manipulation
        self.cumulative_external_impulse[0] += dp
        self.prev_endpoint_momentum = curr_endpoint_mom

        # Build constraint value/target: endpoint separation as proxy for knot tightness
        left_pos = self.pos[self.ctrl_left].copy()
        right_pos = self.pos[self.ctrl_right].copy()
        separation = np.linalg.norm(right_pos - left_pos)
        constraint_val = np.array([separation], dtype=np.float64)
        constraint_tgt = np.array([0.0], dtype=np.float64)

        # Domain weights: single global domain covering all vertices
        domain_weights = np.ones((1, self.n_verts), dtype=np.float64)

        # External force on endpoints (reaction forces from kinematic constraint)
        ext_force = np.zeros((1, 1, 3), dtype=np.float64)
        # Approximate endpoint reaction force from momentum change
        if total_dt > 0:
            ext_force[0, 0] = dp / max(total_dt, 1e-12)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_vel.copy(),
            mass=self.mass_arr.copy(),
            attributes={
                "mass_quantity": self.mass_arr.copy(),
                "arc_fraction": np.linspace(0.0, 1.0, self.n_verts),
                "rod_radius": np.full(self.n_verts, self.radius),
                "contact_diameter": np.full(self.n_verts, 2.0 * self.radius),
                "endpoint_controlled": np.isin(
                    np.arange(self.n_verts), [self.ctrl_left, self.ctrl_right]
                ).astype(float),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force,
                "constraint_value": constraint_val,
                "constraint_target": constraint_tgt,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "friction_coefficient": 0.0,
                "constraint_scale": 1.0,
                "reference_momentum": 0.1,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_verts": self.n_verts,
            "rest_length": self.rest_length,
            "EI": self.EI,
        }
