"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(int(seed))
        initial = scene.initial_state()
        self.x0 = np.asarray(initial.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = len(self.x)
        self.time = 0.0
        self.nx, self.ny = (int(v) for v in self.config["cloth"]["grid_shape"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.mu = float(self.config["friction"]["coefficient"])
        self.thickness = float(self.config["contact"]["thickness"])
        self.angle = np.deg2rad(float(self.config["incline"]["angle_degrees"]))
        self.ramp_center = np.asarray(self.config["incline"]["plane_translation"], dtype=float)
        self.ramp_half_thickness = float(self.config["incline"]["plane_half_thickness"])
        self.downhill = np.asarray(self.config["incline"]["downhill_direction"], dtype=float)
        self.downhill /= np.linalg.norm(self.downhill)
        self.normal = np.array([-np.sin(self.angle), 0.0, np.cos(self.angle)])
        self.cross_slope = np.array([0.0, 1.0, 0.0])
        self.uphill = -self.downhill
        self.edges_i, self.edges_j, self.rest, self.stiffness, self.edge_damping = self._make_springs()
        self._initial_gap = self._signed_gap(self.x0)
        self._last_diagnostics = {}

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time:
            raise ValueError("advance_to requires a finite nondecreasing time")
        while self.time < target - 1.0e-12:
            dt = min(1.0 / 600.0, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return dict(self._last_diagnostics)

    def _make_springs(self):
        pairs = []

        def add(i, j, k, damping):
            pairs.append((i, j, np.linalg.norm(self.x0[j] - self.x0[i]), k, damping))

        nx, ny = self.nx, self.ny
        for j in range(ny):
            for i in range(nx):
                a = j * nx + i
                if i + 1 < nx:
                    add(a, a + 1, 180.0, 0.045)
                if j + 1 < ny:
                    add(a, a + nx, 180.0, 0.045)
                if i + 1 < nx and j + 1 < ny:
                    add(a, a + nx + 1, 70.0, 0.025)
                if i > 0 and j + 1 < ny:
                    add(a, a + nx - 1, 70.0, 0.025)
                if i + 2 < nx:
                    add(a, a + 2, 12.0, 0.012)
                if j + 2 < ny:
                    add(a, a + 2 * nx, 12.0, 0.012)
        data = np.asarray(pairs, dtype=float)
        return (data[:, 0].astype(np.int32), data[:, 1].astype(np.int32),
                data[:, 2], data[:, 3], data[:, 4])

    def _signed_gap(self, positions):
        return ((positions - self.ramp_center) @ self.normal
                - self.ramp_half_thickness - self.thickness)

    def _forces(self):
        force = self.mass[:, None] * self.gravity[None, :]
        i, j = self.edges_i, self.edges_j
        delta = self.x[j] - self.x[i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1.0e-10)[:, None]
        relative_speed = np.sum((self.v[j] - self.v[i]) * direction, axis=1)
        magnitude = self.stiffness * (length - self.rest) + self.edge_damping * relative_speed
        edge_force = magnitude[:, None] * direction
        np.add.at(force, i, edge_force)
        np.add.at(force, j, -edge_force)
        return force

    def _substep(self, dt):
        self.v += dt * self._forces() / self.mass[:, None]
        self.x += dt * self.v
        gap = self._signed_gap(self.x)
        contact = gap < 0.0
        if not np.any(contact):
            return
        self.x[contact] -= gap[contact, None] * self.normal[None, :]
        ids = np.flatnonzero(contact)
        vn = self.v[ids] @ self.normal
        inward = vn < 0.0
        if np.any(inward):
            self.v[ids[inward]] -= vn[inward, None] * self.normal[None, :]
        tangential = self.v[ids] - (self.v[ids] @ self.normal)[:, None] * self.normal
        speed = np.linalg.norm(tangential, axis=1)
        normal_load_accel = max(0.0, -float(np.dot(self.gravity, self.normal)))
        max_delta_v = self.mu * normal_load_accel * dt
        moving = speed > 1.0e-12
        if np.any(moving):
            reduction = np.minimum(speed[moving], max_delta_v)
            self.v[ids[moving]] -= reduction[:, None] / speed[moving, None] * tangential[moving]

    def _deformation_gradient(self):
        f = np.zeros((self.n, 2, 2), dtype=float)
        side = float(self.config["cloth"]["side_length"])
        du = side / max(1, self.nx - 1)
        dv = side / max(1, self.ny - 1)
        for j in range(self.ny):
            for i in range(self.nx):
                a = j * self.nx + i
                il, ir = max(0, i - 1), min(self.nx - 1, i + 1)
                jb, jt = max(0, j - 1), min(self.ny - 1, j + 1)
                eu = (self.x[j * self.nx + ir] - self.x[j * self.nx + il]) / (max(1, ir - il) * du)
                ev = (self.x[jt * self.nx + i] - self.x[jb * self.nx + i]) / (max(1, jt - jb) * dv)
                f[a, 0, 0], f[a, 0, 1] = np.dot(eu, self.uphill), np.dot(ev, self.uphill)
                f[a, 1, 0], f[a, 1, 1] = np.dot(eu, self.cross_slope), np.dot(ev, self.cross_slope)
        return f

    def _frame(self):
        gap = self._signed_gap(self.x)
        deformation = self._deformation_gradient()
        displacement = float(np.dot(np.mean(self.x, axis=0) - np.mean(self.x0, axis=0), self.downhill))
        velocity = float(np.dot(np.mean(self.v, axis=0), self.downhill))
        normal_accel = max(1.0e-12, -float(np.dot(self.gravity, self.normal)))
        impact_time = np.sqrt(max(0.0, float(np.mean(self._initial_gap))) * 2.0 / normal_accel)
        elapsed = max(0.0, self.time - impact_time)
        analytic_a = max(0.0, np.linalg.norm(self.gravity) *
                         (np.sin(self.angle) - self.mu * np.cos(self.angle)))
        expected_disp = 0.5 * analytic_a * elapsed * elapsed
        expected_vel = analytic_a * elapsed
        residual = max(0.0, -float(np.min(gap)))
        self._last_diagnostics = {
            "downslope_displacement": displacement,
            "downslope_velocity": velocity,
            "analytical_displacement_error": displacement - expected_disp,
            "analytical_velocity_error": velocity - expected_vel,
            "constraint_residual": residual,
        }
        frame = StateFrame(
            time=self.time, position=self.x.copy(), velocity=self.v.copy(), mass=self.mass.copy(),
            attributes={"uv": np.asarray(self.scene._uv, dtype=float).copy(),
                        "cloth_vertex": np.ones(self.n), "mass_quantity": self.mass.copy(),
                        "deformation_gradient": deformation},
            fields={"constraint_target": np.zeros(self.n), "constraint_value": gap.copy(),
                    "contact_signed_distance": gap.copy(), "cumulative_mass_source": np.zeros(1)},
            scalars={"friction_coefficient": self.mu, "constraint_scale": 1.0,
                     "contact_length_scale": self.thickness, "constraint_residual": residual,
                     "analytical_displacement_error": displacement - expected_disp,
                     "analytical_velocity_error": velocity - expected_vel},
        )
        frame.validate()
        return frame
