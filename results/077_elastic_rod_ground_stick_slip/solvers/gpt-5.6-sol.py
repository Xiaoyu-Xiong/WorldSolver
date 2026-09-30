"""Deterministic physical elastic-rod solver for the benchmark scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.x0 = np.asarray(initial.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.n = len(self.x)
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.inv_mass = 1.0 / np.maximum(self.mass, 1.0e-12)
        self.top = np.asarray(initial.attributes["driven_top"], dtype=float) > 0.5
        self.time = 0.0
        self.dt = 0.00035
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu_d = float(self.config.get("contact", {}).get("ground_friction_coefficient", 0.35))
        self.mu_s = max(self.mu_d, 0.50)
        self.rings = 10
        self.sides = max(1, self.n // self.rings)
        edges: list[tuple[int, int, float, float]] = []

        def add_edge(a: int, b: int, stiffness: float) -> None:
            rest = float(np.linalg.norm(self.x0[b] - self.x0[a]))
            edges.append((a, b, rest, stiffness))

        for k in range(self.rings - 1):
            for j in range(self.sides):
                a = k * self.sides + j
                add_edge(a, (k + 1) * self.sides + j, 900.0)
                add_edge(a, (k + 1) * self.sides + (j + 1) % self.sides, 260.0)
        for k in range(self.rings):
            for j in range(self.sides):
                add_edge(k * self.sides + j, k * self.sides + (j + 1) % self.sides, 500.0)
        for k in range(self.rings - 2):
            for j in range(self.sides):
                add_edge(k * self.sides + j, (k + 2) * self.sides + j, 95.0)
        self.edges = edges
        self.damping = 0.22
        self.ground_k = 4200.0
        self.ground_c = 7.0
        self.contact_height = 0.055
        self.contact_anchor = self.x[:, :2].copy()
        self.contact_was_active = np.zeros(self.n, dtype=bool)
        self.last_forces = np.zeros((self.n, 3), dtype=float)
        self.last_contact_force = np.zeros((self.n, 3), dtype=float)
        self.last_normal = np.zeros(self.n, dtype=float)
        self.cumulative_impulse = np.zeros(3, dtype=float)

    def _top_target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        bc = self.config["boundary_conditions"]["driven_top"]
        press_end = float(bc["press_interval"][1])
        press = np.asarray(bc["press_displacement"], dtype=float)
        drag_v = np.asarray(bc["drag_velocity"], dtype=float)
        if t <= press_end:
            disp = (t / max(press_end, 1.0e-9)) * press
            vel = press / max(press_end, 1.0e-9)
        else:
            disp = press + (t - press_end) * drag_v
            vel = drag_v
        return self.x0 + disp[None, :], np.broadcast_to(vel, (self.n, 3)).copy()

    def _forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.mass[:, None] * self.gravity[None, :]
        for a, b, rest, stiffness in self.edges:
            d = self.x[b] - self.x[a]
            length = float(np.linalg.norm(d))
            if length < 1.0e-10:
                continue
            direction = d / length
            rel = float(np.dot(self.v[b] - self.v[a], direction))
            force = (stiffness * (length - rest) + 0.85 * rel) * direction
            f[a] += force
            f[b] -= force
        f -= self.damping * self.v
        contact = np.zeros_like(f)
        normal = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            penetration = self.contact_height - self.x[i, 2]
            if penetration <= 0.0:
                self.contact_was_active[i] = False
                continue
            if not self.contact_was_active[i]:
                self.contact_anchor[i] = self.x[i, :2]
            self.contact_was_active[i] = True
            normal[i] = max(0.0, self.ground_k * penetration - self.ground_c * self.v[i, 2])
            contact[i, 2] = normal[i]
            trial = -180.0 * (self.x[i, :2] - self.contact_anchor[i]) - 3.5 * self.v[i, :2]
            trial_mag = float(np.linalg.norm(trial))
            if trial_mag <= self.mu_s * normal[i]:
                ft = trial
            else:
                speed = float(np.linalg.norm(self.v[i, :2]))
                direction = self.v[i, :2] if speed > 1.0e-9 else trial
                ft = -self.mu_d * normal[i] * direction / max(float(np.linalg.norm(direction)), 1.0e-9)
                self.contact_anchor[i] = self.x[i, :2]
            contact[i, :2] = ft
        f += contact
        return f, contact, normal

    def _step(self, h: float) -> None:
        target, target_v = self._top_target(self.time + h)
        self.x[self.top] = target[self.top]
        self.v[self.top] = target_v[self.top]
        f, contact, normal = self._forces()
        free = ~self.top
        self.v[free] += h * f[free] * self.inv_mass[free, None]
        self.v[free] *= 1.0 / (1.0 + 0.018 * h)
        self.x[free] += h * self.v[free]
        below = self.x[:, 2] < 0.0545
        self.x[below, 2] = 0.0545
        self.v[below & (self.v[:, 2] < 0.0), 2] *= -0.05
        self.x[self.top] = target[self.top]
        self.v[self.top] = target_v[self.top]
        self.last_forces = f
        self.last_contact_force = contact
        self.last_normal = normal
        drive = -np.sum(f[self.top], axis=0) if np.any(self.top) else np.zeros(3)
        ext = (np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
               + np.sum(contact, axis=0) + drive)
        self.cumulative_impulse += h * ext
        self.time += h

    def _frame(self) -> StateFrame:
        # Axial material elements with a rank-one deformation-gradient update.
        deformation = np.tile(np.eye(3), (self.rings - 1, 1, 1))
        for k in range(self.rings - 1):
            a = self.x[k * self.sides:(k + 1) * self.sides].mean(axis=0)
            b = self.x[(k + 1) * self.sides:(k + 2) * self.sides].mean(axis=0)
            ar = self.x0[k * self.sides:(k + 1) * self.sides].mean(axis=0)
            br = self.x0[(k + 1) * self.sides:(k + 2) * self.sides].mean(axis=0)
            d, dr = b - a, br - ar
            deformation[k] += np.outer(d - dr, dr) / max(float(np.dot(dr, dr)), 1.0e-12)
        active = self.last_normal > 0.0
        normals = np.zeros((self.n, 3), dtype=float)
        normals[:, 2] = 1.0
        tangential = self.last_contact_force.copy()
        tangential[:, 2] = 0.0
        gravity_force = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
        drive_force = -np.sum(self.last_forces[self.top], axis=0) if np.any(self.top) else np.zeros(3)
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": np.stack((gravity_force, np.sum(self.last_contact_force, axis=0), drive_force))[None],
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": deformation,
            "friction_active_mask": active.astype(float),
            "friction_dynamic_coefficient": np.full(self.n, self.mu_d),
            "friction_normal": normals,
            "friction_normal_force": self.last_normal.copy(),
            "friction_relative_velocity": self.v.copy(),
            "friction_static_coefficient": np.full(self.n, self.mu_s),
            "friction_tangential_force": tangential,
        }
        frame = StateFrame(
            float(self.time), self.x.copy(), self.v.copy(), mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy()}, fields=fields,
            scalars={
                "friction_stick_speed_tolerance": 1.0e-3,
                "reference_force": 1.0,
                "reference_momentum": float(np.sum(self.mass)),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-9:
            raise ValueError("advance_to requires nondecreasing times")
        while self.time + 1.0e-10 < target:
            self._step(min(self.dt, target - self.time))
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_mass_spring", "nodes": int(self.n), "time": float(self.time)}
