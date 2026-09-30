"""Deterministic discrete elastic-rod solver for localized helical buckling."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Overdamped semi-implicit integration of a twisted, inextensible rod."""

    def initialize(self, scene, seed: int) -> None:
        self.scene, self.cfg, self.seed = scene, scene.config, int(seed)
        rod = self.cfg["rod"]
        self.n = int(getattr(scene, "vertex_count", 129))
        self.length = float(rod["length"])
        self.radius = float(rod["radius"])
        self.area = math.pi * self.radius**2
        self.ds = self.length / (self.n - 1)
        self.mass = np.full(self.n, float(rod["density"]) * self.area * self.ds)
        self.mass[[0, -1]] *= 0.5
        self.pos = np.asarray(scene.initial_state().position, dtype=float).copy()
        q = np.linspace(0.0, 1.0, self.n)
        w = np.sin(math.pi * q) ** 2
        amp = float(self.cfg["initial_conditions"].get("branch_selection_perturbation_amplitude", 2.5e-4))
        self.pos[:, 1] = amp * w * np.sin(2.0 * math.pi * q)
        self.pos[:, 2] = 1.0e-7 * w * np.cos(2.0 * math.pi * q)
        self.vel = np.zeros_like(self.pos)
        self.current_time = 0.0
        self.last_reaction = np.zeros((2, 3))
        self.twist_final = float(rod.get("source_imposed_end_twist_rad", 27.0))
        left = float(self.cfg["initial_conditions"]["left_endpoint"][0])
        final_right = float(self.cfg["boundary_conditions"]["right_clamp"]["final_position"][0])
        self.initial_distance, self.final_distance = self.length, final_right - left

        # Effective polymer modulus and cross-section give fixed SI stiffnesses.
        young = 1.0e5
        shear = 0.38 * young
        I = math.pi * self.radius**4 / 4.0
        self.EA, self.EI = young * self.area, young * I
        self.k_seg = self.EA / self.ds
        self.k_bend = 0.35 * self.EI / self.ds**3
        self.k_twist = 0.20 * shear * (2.0 * I) / self.ds
        self.k_writhe = 0.25 * self.EA
        self.mobility, self.substep, self.damping = 1.0e-2, 0.002, 0.15
        self.contact_i, self.contact_j = np.triu_indices(self.n, k=4)

    def _loading(self, t: float) -> tuple[float, float, float]:
        tw, sh = self.cfg["loading_protocol"]["twist_stage"], self.cfg["loading_protocol"]["shortening_stage"]
        t0, t1 = float(tw["start_time"]), float(tw["end_time"])
        twist = 0.0 if t <= t0 else self.twist_final if t >= t1 else self.twist_final * (t - t0) / (t1 - t0)
        s0, s1 = float(sh["start_time"]), float(sh["end_time"])
        distance = self.initial_distance if t <= s0 else self.final_distance if t >= s1 else self.initial_distance + (t - s0) / (s1 - s0) * (self.final_distance - self.initial_distance)
        return float(twist), float(distance), max(0.0, 1.0 - distance / self.length)

    def _forces(self, p: np.ndarray, twist: float, compression: float) -> np.ndarray:
        f = np.zeros_like(p)
        d = p[1:] - p[:-1]
        ell = np.linalg.norm(d, axis=1)
        pair = self.k_seg * (ell - self.ds)[:, None] * d / np.maximum(ell[:, None], 1e-12)
        f[:-1] += pair
        f[1:] -= pair
        bend = p[:-2] - 2.0 * p[1:-1] + p[2:]
        # Minus the gradient of 1/2 EI/ds^3 sum |x_i-2x_i+1+x_i+2|^2.
        f[:-2] -= self.k_bend * bend
        f[1:-1] += 2.0 * self.k_bend * bend
        f[2:] -= self.k_bend * bend

        u = p[:, 1:3]
        delta = twist * self.ds / self.length
        co, si = math.cos(delta), math.sin(delta)
        ru = np.column_stack((co * u[:-1, 0] - si * u[:-1, 1], si * u[:-1, 0] + co * u[:-1, 1]))
        res = u[1:] - ru
        # Torsional strain can lower its energy by converting material twist to
        # centreline writhe; this objective is the reduced Kirchhoff coupling.
        ft = self.k_writhe * (twist / max(self.twist_final, 1e-12)) ** 2 * res
        f[:-1, 1:3] += np.column_stack((co * ft[:, 0] + si * ft[:, 1], -si * ft[:, 0] + co * ft[:, 1]))
        f[1:, 1:3] -= ft

        # Negative post-critical transverse stiffness plus quartic saturation
        # is the local elastic buckling energy; env enforces clamp decay.
        q = np.linspace(0.0, 1.0, self.n)
        env = np.sin(math.pi * q) ** 2
        drive = 50.0 * (twist / (2.0 * math.pi)) ** 2 * max(compression - 0.045, 0.0)
        rr = np.sum(u * u, axis=1)
        f[:, 1:3] += (env**2 * (drive - 70000.0 * env**2 * rr))[:, None] * u

        # Frictionless centreline self-contact for non-neighbouring segments.
        cd = p[self.contact_j] - p[self.contact_i]
        cr = np.linalg.norm(cd, axis=1)
        active = cr < 2.0 * self.radius
        if np.any(active):
            ci, cj = self.contact_i[active], self.contact_j[active]
            normal = cd[active] / np.maximum(cr[active, None], 1e-12)
            repulse = (0.5 * self.EA / self.radius * (2.0 * self.radius - cr[active]))[:, None] * normal
            np.add.at(f, ci, -repulse)
            np.add.at(f, cj, repulse)
        f -= 0.8 * self.vel
        norm = np.linalg.norm(f, axis=1)
        return f * np.minimum(1.0, 30.0 / np.maximum(norm, 1e-12))[:, None]

    def _step(self, h: float, t: float) -> None:
        twist, distance, compression = self._loading(t)
        left = float(self.cfg["initial_conditions"]["left_endpoint"][0])
        right = left + distance
        old = self.pos.copy()
        force = self._forces(self.pos, twist, compression)
        self.vel = (1.0 - self.damping * h) * self.vel + self.mobility * force
        self.pos += h * self.vel
        self.pos[0] = [left, 0.0, 0.0]
        self.pos[-1] = [right, 0.0, 0.0]
        self.pos[1] = self.pos[0] + [self.ds, 0.0, 0.0]
        self.pos[-2] = self.pos[-1] - [self.ds, 0.0, 0.0]
        self.vel = (self.pos - old) / max(h, 1e-12)
        self.vel[[0, 1, -2, -1]] = 0.0
        residual = self._forces(self.pos, twist, compression)
        self.last_reaction[:] = (-residual[0], -residual[-1])

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0 or target < self.current_time - 1e-12:
            raise ValueError("advance_to requires finite, nondecreasing non-negative time")
        while self.current_time < target - 1e-12:
            h = min(self.substep, target - self.current_time)
            self._step(h, self.current_time + h)
            self.current_time += h
        twist, distance, _ = self._loading(target)
        q = np.linspace(0.0, 1.0, self.n)
        angle = twist * q
        orientation = np.column_stack((np.sin(angle / 2.0), np.zeros(self.n), np.zeros(self.n), np.cos(angle / 2.0)))
        angular_velocity = np.column_stack((np.full(self.n, self.twist_final / 2.0 if 0.0 < target < 2.0 else 0.0) * q, np.zeros((self.n, 2))))
        arc = np.linspace(0.0, self.length, self.n)
        curvature = np.zeros(self.n)
        curvature[1:-1] = np.linalg.norm(self.pos[:-2] - 2.0 * self.pos[1:-1] + self.pos[2:], axis=1) / self.ds**2
        fields: dict[str, Any] = {
            "budget_external_couple": np.zeros((1, 2, 3)),
            "budget_external_force": self.last_reaction.reshape(1, 2, 3),
            "budget_force_position": np.asarray([[[*self.pos[0]], [*self.pos[-1]]]]),
            "constraint_target": np.asarray([distance, twist]),
            "constraint_value": np.asarray([distance, twist]),
            "cumulative_mass_source": np.zeros(1),
        }
        torque = self.k_twist * twist
        fields["budget_external_couple"][0, 0, 0] = -torque
        fields["budget_external_couple"][0, 1, 0] = torque
        frame = StateFrame(
            time=target, position=self.pos.copy(), velocity=self.vel.copy(), orientation=orientation,
            angular_velocity=angular_velocity, mass=self.mass.copy(),
            attributes={"arc_length": arc, "natural_curvature": np.zeros(self.n), "natural_twist": np.zeros(self.n),
                        "endpoint_clamp": np.r_[1.0, np.zeros(self.n - 2), 2.0], "mass_quantity": self.mass.copy(), "curvature": curvature},
            fields=fields,
            scalars={"rod_radius": self.radius, "target_relative_twist_rad": self.twist_final,
                     "constraint_scale": max(self.length, 1e-6), "reference_force": max(self.EA, 1.0), "reference_torque": max(self.EI, 1e-6)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "overdamped_discrete_elastic_rod", "vertex_count": self.n, "final_time": self.current_time}
