"""Quasi-dynamic twist solver for a V-shaped anisotropic rod.

The rod centerline is frozen to its reference V-shape.  The driven endpoint is
gradually rotated about its local tangent while the opposite end is clamped.
Twist is treated as a scalar field along the arc length and evolved with a
stable explicit diffusion step.  The resulting material-frame orientations are
written as quaternions so the continuous surface stripe reveals the twist.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        # Initial frame from the scene (frozen V-shape, identity orientation).
        self.frame0 = scene.initial_state()
        self.position = self.frame0.position.copy()
        self.n = len(self.position)
        self.mass = (
            self.frame0.mass.copy()
            if self.frame0.mass is not None
            else np.ones(self.n, dtype=float)
        )
        self.velocity = np.zeros((self.n, 3), dtype=float)

        # Arc-length parameter from the V-shaped centerline.
        edges = np.diff(self.position, axis=0)
        self.segment_lengths = np.linalg.norm(edges, axis=1)
        self.arc_length = np.r_[0.0, np.cumsum(self.segment_lengths)]
        self.total_length = self.arc_length[-1]
        self.ds = self.total_length / (self.n - 1)

        # Vertex tangents (averaged incident edge directions, unit length).
        self.tangents = np.zeros((self.n, 3), dtype=float)
        self.tangents[0] = edges[0] / self.segment_lengths[0]
        self.tangents[-1] = edges[-1] / self.segment_lengths[-1]
        for i in range(1, self.n - 1):
            t_prev = edges[i - 1] / self.segment_lengths[i - 1]
            t_next = edges[i] / self.segment_lengths[i]
            t = t_prev + t_next
            t /= np.linalg.norm(t) + 1e-12
            self.tangents[i] = t

        # Twist state.
        self.theta = np.zeros(self.n, dtype=float)
        self.theta_dot = np.zeros(self.n, dtype=float)
        self.time = 0.0

        # Diffusion coefficient chosen so twist stays on the driven arm early and
        # reaches the fixed arm by the end of the simulation.
        self.D = 0.2
        self.dt_sub = 5.0e-4

        # Torsional stiffness for the couple budget (N*m^2).
        cfg = getattr(scene, "config", None)
        if cfg is not None:
            rod = cfg.get("rod", {})
            w = float(rod.get("cross_section_width", 0.026))
            t = float(rod.get("cross_section_thickness", 0.006))
        else:
            w, t = 0.026, 0.006
        a, b = max(w, t), min(w, t)
        # Torsion constant for a thin rectangle.
        J = a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - (b / a) ** 4 / 12.0))
        # Shear modulus for a soft elastic solid (~10 MPa).  The exact value only
        # sets the magnitude of the reported budget couple; sign and shape are
        # what matter for the raw trajectory.
        G = 1.0e6
        self.GJ = max(G * J, 1.0e-6)

        # Driven-end rotation schedule.
        default_schedule = np.array(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 2.0 * np.pi], [6.0, 4.0 * np.pi]],
            dtype=float,
        )
        if cfg is not None:
            schedule = cfg.get("boundary_conditions", {}).get("driven_end", {}).get(
                "rotation_schedule", default_schedule.tolist()
            )
        else:
            schedule = default_schedule
        self.schedule = np.asarray(schedule, dtype=float)

        # Preserve initial attributes and add the per-sample mass quantity.
        self.base_attributes = dict(self.frame0.attributes)
        self.base_attributes["mass_quantity"] = self.mass.copy()

    def _driven_angle(self, t: float) -> float:
        s = self.schedule
        if t <= s[0, 0]:
            return float(s[0, 1])
        if t >= s[-1, 0]:
            t0, a0 = s[-2]
            t1, a1 = s[-1]
            return float(a0 + (a1 - a0) * (t - t0) / (t1 - t0))
        # Locate interval and linearly interpolate.
        for i in range(len(s) - 1):
            t0, a0 = s[i]
            t1, a1 = s[i + 1]
            if t0 <= t <= t1:
                return float(a0 + (a1 - a0) * (t - t0) / (t1 - t0))
        return float(s[-1, 1])

    def _driven_rate(self, t: float) -> float:
        s = self.schedule
        if t <= s[0, 0]:
            return 0.0
        if t >= s[-1, 0]:
            t0, a0 = s[-2]
            t1, a1 = s[-1]
            return float((a1 - a0) / (t1 - t0))
        for i in range(len(s) - 1):
            t0, a0 = s[i]
            t1, a1 = s[i + 1]
            if t0 <= t <= t1:
                return float((a1 - a0) / (t1 - t0))
        return 0.0

    def _orientation_from_twist(self, theta: np.ndarray) -> np.ndarray:
        """Return quaternion (xyzw) for rotation of angle theta about the local tangent."""
        half = theta / 2.0
        q = np.zeros((self.n, 4), dtype=float)
        q[:, :3] = self.tangents * np.sin(half)[:, None]
        q[:, 3] = np.cos(half)
        return q

    def _substep(self, dt: float) -> None:
        t = self.time
        theta_d = self._driven_angle(t)
        rate_d = self._driven_rate(t)

        # Enforce Dirichlet boundary conditions at the old time.
        self.theta[0] = theta_d
        self.theta[-1] = 0.0

        # Interior Laplacian in arc-length parameter.
        lap = np.zeros(self.n, dtype=float)
        lap[1:-1] = (
            self.theta[2:] - 2.0 * self.theta[1:-1] + self.theta[:-2]
        ) / (self.ds * self.ds)

        # Overdamped torsion diffusion: theta_t = D * theta_ss.
        self.theta_dot[:] = self.D * lap
        self.theta_dot[0] = rate_d
        self.theta_dot[-1] = 0.0

        # Explicit Euler update.
        self.theta = self.theta + dt * self.theta_dot

        # Enforce boundary conditions at the new time.
        self.theta[0] = self._driven_angle(t + dt)
        self.theta[-1] = 0.0

        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        if time < 0.0:
            time = 0.0

        # Integrate forward with fixed sub-steps.
        target = time
        while self.time < target:
            dt = min(self.dt_sub, target - self.time)
            self._substep(dt)

        # Build the frame.
        orientation = self._orientation_from_twist(self.theta)
        angular_velocity = self.theta_dot[:, None] * self.tangents

        attributes = dict(self.base_attributes)
        attributes["material_twist"] = self.theta.copy()

        # Budget terms: one domain (the rod) with two boundary actions.
        G, K = 1, 2
        budget_force = np.zeros((G, K, 3), dtype=float)
        budget_couple = np.zeros((G, K, 3), dtype=float)
        budget_pos = np.zeros((G, K, 3), dtype=float)
        budget_pos[0, 0, :] = self.position[0]
        budget_pos[0, 1, :] = self.position[-1]

        # Torque at driven end from the local twist gradient; fixed end feels the
        # equal-and-opposite reaction (net torque on the whole rod is zero in
        # quasi-static torsion).
        tau_driven = (
            self.GJ
            * (self.theta[0] - self.theta[1])
            / self.ds
            * self.tangents[0]
        )
        tau_fixed = (
            self.GJ
            * (self.theta[-2] - self.theta[-1])
            / self.ds
            * self.tangents[-1]
        )
        budget_couple[0, 0, :] = tau_driven
        budget_couple[0, 1, :] = tau_fixed

        driven_angle = self._driven_angle(self.time)
        fields = {
            "budget_external_force": budget_force,
            "budget_external_couple": budget_couple,
            "budget_force_position": budget_pos,
            "constraint_value": np.array(
                [self.theta[0], self.theta[-1]], dtype=float
            ),
            "constraint_target": np.array(
                [driven_angle, 0.0], dtype=float
            ),
            "cumulative_mass_source": np.array([0.0], dtype=float),
        }

        # Use physically plausible reference scales.  The torsional stiffness is
        # small (flat cross-section), so the torque scale is ~0.01 N*m.
        reference_torque = max(abs(self.GJ) / max(self.total_length, 1e-6), 1e-6)
        scalars = {
            "constraint_scale": 1.0,
            "reference_force": 1.0,
            "reference_torque": reference_torque,
            "driven_end_angle": driven_angle,
        }

        frame = StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "diffusion_torsion_rod",
            "seed": self.seed,
            "diffusion_coefficient": self.D,
            "substep_dt": self.dt_sub,
            "torsional_stiffness_GJ": self.GJ,
            "vertex_count": self.n,
        }
