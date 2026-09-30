"""Baseline physics solver for the soft-objects squeeze-out scene.

The five soft bodies are represented as dynamic point samples (one per body).
The pusher plate is a kinematic box and the chamber/funnel/outlet/collector
geometry is modelled as a collection of static oriented boxes.  Contact is
handled with penalty springs and damping.  Time integration uses a stable
semi-implicit Euler substepping scheme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic explicit soft-body solver for the squeeze-out benchmark."""

    def __init__(self) -> None:
        self.scene: Any = None
        self.seed: int = 0
        self.config: dict[str, Any] = {}

        self.mass: np.ndarray | None = None
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.body_id: np.ndarray | None = None
        self.radius: np.ndarray | None = None
        self.t: float = 0.0

        self.static_colliders: list[dict[str, Any]] = []
        self.k_contact: float = 2.0e4
        self.c_damp: float = 2.0
        self.mu: float = 0.25
        self.dt_sub: float = 1.0e-4
        self.density: float = 300.0

        self.gravity: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=float)

        # Cumulative budget quantities.
        self.cum_external_impulse: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = self._load_config()

        init = scene.initial_state()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.position = np.asarray(init.position, dtype=float).copy()
        self.velocity = np.asarray(init.velocity, dtype=float).copy()
        if init.body_id is None:
            self.body_id = np.arange(len(self.mass), dtype=np.int64)
        else:
            self.body_id = np.asarray(init.body_id, dtype=np.int64).copy()

        if self.config.get("gravity") is not None:
            self.gravity = np.array(self.config["gravity"], dtype=float)

        # Approximate effective radius from mass and a representative density.
        self.radius = ((3.0 * self.mass) / (4.0 * np.pi * self.density)) ** (1.0 / 3.0)

        self.static_colliders = self._build_static_colliders()
        self.t = 0.0
        self.cum_external_impulse = np.zeros((len(self.mass), 3), dtype=float)

    def _load_config(self) -> dict[str, Any]:
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        n = np.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3, dtype=float)
        x, y, z, w = x / n, y / n, z / n, w / n
        R = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )
        return R

    def _build_static_colliders(self) -> list[dict[str, Any]]:
        colliders: list[dict[str, Any]] = []

        def add(center, half, rotation=None):
            R = np.eye(3, dtype=float) if rotation is None else self._quat_to_matrix(np.array(rotation, dtype=float))
            colliders.append({"center": np.array(center, dtype=float), "half": np.array(half, dtype=float), "R": R})

        # Chamber walls.
        add([0.0, 0.58, 2.55], [0.84, 0.06, 2.0])
        add([-0.78, 0.0, 2.55], [0.06, 0.52, 2.0])
        add([0.78, 0.0, 2.55], [0.06, 0.52, 2.0])

        # Converging funnel walls.
        add([-0.55, 0.0, 0.72], [0.055, 0.52, 0.75], [0.0, -0.153, 0.0, 0.988])
        add([0.55, 0.0, 0.72], [0.055, 0.52, 0.75], [0.0, 0.153, 0.0, 0.988])
        add([0.0, 0.38, 0.72], [0.72, 0.055, 0.75], [0.19, 0.0, 0.0, 0.982])

        # Outlet lips and collector.
        add([0.0, -0.33, 0.0], [0.34, 0.20, 0.07])
        add([0.0, 0.33, 0.0], [0.34, 0.20, 0.07])
        add([0.0, 0.0, -2.05], [1.25, 0.9, 0.06])
        add([0.0, 0.95, -1.05], [1.25, 0.05, 1.0])
        add([-1.30, 0.0, -1.05], [0.05, 0.9, 1.0])
        add([1.30, 0.0, -1.05], [0.05, 0.9, 1.0])

        return colliders

    def _pusher_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return pusher center and velocity at time t."""
        cfg = self.config["boundary_conditions"]["plate_motion"]
        start_t = float(cfg["start_time"])
        end_t = float(cfg["end_time"])
        start_z = float(cfg["start_translation"][2])
        end_z = float(cfg["end_translation"][2])

        if t <= start_t:
            z = start_z
            vz = 0.0
        elif t >= end_t:
            z = end_z
            vz = 0.0
        else:
            alpha = (t - start_t) / (end_t - start_t)
            z = start_z + alpha * (end_z - start_z)
            vz = (end_z - start_z) / (end_t - start_t)

        center = np.array([0.0, 0.0, z], dtype=float)
        velocity = np.array([0.0, 0.0, vz], dtype=float)
        return center, velocity

    # ------------------------------------------------------------------
    # Force / contact model
    # ------------------------------------------------------------------
    def _box_contact(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        radius: np.ndarray,
        center: np.ndarray,
        half: np.ndarray,
        R: np.ndarray,
        vel_box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return contact force and contact normal for a single oriented box."""
        if vel_box is None:
            vel_box = np.zeros(3, dtype=float)

        local = R.T @ (pos - center)
        clamped = np.clip(local, -half, half)
        diff = local - clamped
        dist = float(np.linalg.norm(diff))

        if dist < 1e-12:
            # Center is inside the box; use the smallest axis overlap.
            overlap = half - np.abs(local)
            axis = int(np.argmin(overlap))
            depth = overlap[axis]
            n_local = np.zeros(3, dtype=float)
            n_local[axis] = 1.0 if local[axis] >= 0 else -1.0
            # Penetration includes the distance to the surface plus the particle radius.
            delta = radius + depth
        else:
            n_local = diff / dist
            delta = radius - dist

        if delta <= 0.0:
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

        n = R @ n_local
        v_rel = vel - vel_box
        vn = float(np.dot(v_rel, n))
        vt = v_rel - vn * n

        # Normal spring + damping (damping only resists approach).
        f_mag = self.k_contact * delta
        if vn < 0.0:
            f_mag -= self.c_damp * vn
        force = f_mag * n

        # Coulomb friction on the tangential component.
        v_tan = np.linalg.norm(vt)
        if v_tan > 1e-12:
            force -= self.mu * max(self.k_contact * delta, 0.0) * (vt / v_tan)

        return force, n

    def _compute_forces(
        self,
        t: float,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute external force, per-term external forces and interaction force."""
        n = len(pos)
        force = np.zeros((n, 3), dtype=float)
        # External force terms: [body, term, component].
        # term 0 = gravity, term 1 = walls, term 2 = pusher
        force_terms = np.zeros((n, 3, 3), dtype=float)

        # Gravity.
        fg = self.mass[:, None] * self.gravity
        force += fg
        force_terms[:, 0, :] = fg

        # Static wall contacts (term 1).
        for coll in self.static_colliders:
            for i in range(n):
                f, _ = self._box_contact(
                    pos[i], vel[i], self.radius[i], coll["center"], coll["half"], coll["R"]
                )
                if np.any(f != 0.0):
                    force[i] += f
                    force_terms[i, 1, :] += f

        # Pusher plate (term 2).
        pusher_center, pusher_vel = self._pusher_state(t)
        pusher_half = np.array([0.70, 0.50, 0.08], dtype=float)
        R = np.eye(3, dtype=float)
        for i in range(n):
            f, _ = self._box_contact(
                pos[i], vel[i], self.radius[i], pusher_center, pusher_half, R, pusher_vel
            )
            if np.any(f != 0.0):
                force[i] += f
                force_terms[i, 2, :] += f

        # Body-body interaction forces (internal contact, term handled separately).
        interaction = np.zeros((n, 3), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dvec = pos[i] - pos[j]
                dist = float(np.linalg.norm(dvec))
                if dist < 1e-12:
                    continue
                min_dist = self.radius[i] + self.radius[j]
                if dist >= min_dist:
                    continue
                n_ij = dvec / dist
                delta = min_dist - dist

                v_rel = vel[i] - vel[j]
                vn = float(np.dot(v_rel, n_ij))
                f_mag = self.k_contact * delta
                if vn < 0.0:
                    f_mag -= self.c_damp * vn
                f = f_mag * n_ij

                # Friction.
                vt = v_rel - vn * n_ij
                v_tan = float(np.linalg.norm(vt))
                if v_tan > 1e-12:
                    f -= self.mu * max(self.k_contact * delta, 0.0) * (vt / v_tan)

                force[i] += f
                force[j] -= f
                interaction[i] += f
                interaction[j] -= f

        return force, force_terms, interaction

    def _substep(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity
        t = self.t

        force_total, force_terms, _ = self._compute_forces(t, pos, vel)

        # Semi-implicit Euler integration.
        vel += (force_total / self.mass[:, None]) * dt
        pos += vel * dt
        self.t += dt

        # Update cumulative external impulse (integral of total external force).
        external_force = force_terms.sum(axis=1)  # (N, 3)
        self.cum_external_impulse += external_force * dt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.t - 1e-12:
            raise ValueError(f"Cannot advance backwards: {self.t} -> {time}")

        while self.t < time:
            remaining = time - self.t
            dt = min(self.dt_sub, remaining)
            self._substep(dt)

        # Evaluate fields at the sample time.
        force_total, force_terms, interaction_force = self._compute_forces(self.t, self.position, self.velocity)
        external_force = force_terms.sum(axis=1)

        n = len(self.position)
        # Per-body element deformation gradient: identity for the point-sample model.
        element_F = np.tile(np.eye(3, dtype=float), (n, 1, 1))

        # Budget weights: one domain per soft body, identity weights.
        budget_weights = np.eye(n, dtype=float)

        # Inflow/outflow: no mass crossing domain boundaries in this model.
        zeros_vec = np.zeros((n, 3), dtype=float)

        # Mass source: none.
        cum_mass_source = np.zeros(1, dtype=float)

        reference_momentum = 10.0

        frame = StateFrame(
            time=float(self.t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": zeros_vec.copy(),
                "budget_cumulative_outflow_momentum": zeros_vec.copy(),
                "budget_domain_weights": budget_weights,
                "budget_external_force": force_terms,
                "cumulative_mass_source": cum_mass_source,
                "element_deformation_gradient": element_F,
                "interaction_force": interaction_force,
            },
            scalars={"reference_momentum": float(reference_momentum)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "substep_dt": self.dt_sub,
            "sample_count_estimate": int(self.config.get("duration", 8.0) / self.config.get("sample_dt", 0.02)) + 1,
            "current_time": self.t,
        }
