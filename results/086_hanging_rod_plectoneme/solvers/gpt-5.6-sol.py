"""Discrete elastic rod solver for the hanging-rod plectoneme scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Massive inextensible Kirchhoff rod with gravity, drag, and self-contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene, self.cfg = scene, scene.config
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.zeros_like(self.x)
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.inv_mass = 1.0 / self.mass
        self.inv_mass[[0, -1]] = 0.0
        self.n = len(self.x)
        self.segments = self.n - 1
        rod, controls = self.cfg["rod"], self.cfg["end_controls"]
        self.total_length = float(rod["total_length"])
        self.ds = self.total_length / self.segments
        self.radius = float(rod["radius"])
        self.diameter = 2.0 * self.radius
        self.left = np.asarray(controls["left"]["position"], float)
        self.right = np.asarray(controls["right"]["position"], float)
        self.gravity = np.asarray(self.cfg["gravity"], float)
        self.drag = float(self.cfg["ambient_drag"]["coefficient"] if
                          self.cfg["ambient_drag"]["enabled"] else 0.0)
        self.bending_rigidity = 2.0e-4
        self.torsional_rigidity = 2.0e-2
        self.bending_coefficient = self.bending_rigidity / self.ds**3
        self.max_internal_dt = 0.0025
        self.constraint_iterations = 8
        self.contact_i, self.contact_j = np.triu_indices(self.n, k=4)
        self.time = 0.0
        self.step_count = 0
        self.initial_momentum = self._linear_momentum()
        self.initial_angular_momentum = self._angular_momentum()
        self.last_output_time = self.last_output_momentum = 0.0
        self.last_output_angular_momentum = self.initial_angular_momentum.copy()
        self.reference_momentum = max(float(np.sum(self.mass)) *
                                      math.sqrt(np.linalg.norm(self.gravity) * self.total_length), 1e-8)
        self.reference_angular_momentum = max(self.reference_momentum * self.total_length, 1e-8)

    def _twist_kinematics(self, time: float) -> tuple[float, float]:
        ramp = self.cfg["end_controls"]["twist_ramp"]
        start, end = float(ramp["ramp_start"]), float(ramp["ramp_end"])
        raw = (float(time) - start) / (end - start)
        u = min(1.0, max(0.0, raw))
        smooth = u * u * (3.0 - 2.0 * u)
        turns = float(ramp["final_relative_turns"])
        angle = 2.0 * math.pi * turns * smooth
        rate = 2.0 * math.pi * turns * (6.0 * u * (1.0 - u) / (end - start)) if 0.0 < raw < 1.0 else 0.0
        return angle, rate

    def _node_tangents(self, x: np.ndarray) -> np.ndarray:
        e = x[1:] - x[:-1]
        t = e / np.maximum(np.linalg.norm(e, axis=1)[:, None], 1e-12)
        out = np.empty_like(x)
        out[0], out[-1] = t[0], t[-1]
        out[1:-1] = t[:-1] + t[1:]
        out /= np.maximum(np.linalg.norm(out, axis=1)[:, None], 1e-12)
        return out

    def _forces(self, time: float) -> np.ndarray:
        f = self.mass[:, None] * self.gravity[None, :]
        f -= self.drag * (self.mass / np.sum(self.mass))[:, None] * self.v
        d = self.x[:-2] - 2.0 * self.x[1:-1] + self.x[2:]
        e = self.bending_coefficient * d
        f[:-2] -= e
        f[1:-1] += 2.0 * e
        f[2:] -= e
        angle, _ = self._twist_kinematics(time)
        torque = self.torsional_rigidity * angle / self.total_length
        edge = self.x[1:] - self.x[:-1]
        length = np.linalg.norm(edge, axis=1)
        t = edge / np.maximum(length[:, None], 1e-12)
        tv = t[:-1] + t[1:]
        tv /= np.maximum(np.linalg.norm(tv, axis=1)[:, None], 1e-12)
        shear = torque * np.cross(tv, (t[1:] - t[:-1]) / self.ds)
        sn = np.zeros_like(self.x)
        sn[1:-1] = shear
        se = 0.5 * (sn[:-1] + sn[1:])
        f[:-1] += se
        f[1:] -= se
        f[[0, -1]] = 0.0
        return f

    def _project_lengths(self, x: np.ndarray) -> None:
        for _ in range(self.constraint_iterations):
            for parity in (0, 1):
                ids = np.arange(parity, self.segments, 2)
                jds = ids + 1
                delta = x[jds] - x[ids]
                length = np.linalg.norm(delta, axis=1)
                direction = delta / np.maximum(length[:, None], 1e-12)
                error = length - self.ds
                wi, wj = self.inv_mass[ids], self.inv_mass[jds]
                den = np.maximum(wi + wj, 1e-30)
                c = error[:, None] * direction
                x[ids] += (wi / den)[:, None] * c
                x[jds] -= (wj / den)[:, None] * c
            x[0], x[-1] = self.left, self.right

    def _project_self_contact(self, x: np.ndarray) -> None:
        i, j = self.contact_i, self.contact_j
        d = x[i] - x[j]
        dist = np.linalg.norm(d, axis=1)
        active = dist < self.diameter
        if not np.any(active):
            return
        ia, ja, da, dd = i[active], j[active], d[active], dist[active]
        normal = da / np.maximum(dd[:, None], 1e-12)
        coincident = dd < 1e-10
        if np.any(coincident):
            phase = (ia[coincident] * 73856093 + ja[coincident] * 19349663) % 1009
            ang = phase.astype(float) * (2.0 * math.pi / 1009.0)
            normal[coincident] = np.column_stack((np.zeros(np.sum(coincident)), np.cos(ang), np.sin(ang)))
        wi, wj = self.inv_mass[ia], self.inv_mass[ja]
        den = np.maximum(wi + wj, 1e-30)
        corr = 0.75 * (self.diameter - dd)[:, None] * normal
        disp = np.zeros_like(x)
        np.add.at(disp, ia, (wi / den)[:, None] * corr)
        np.add.at(disp, ja, -(wj / den)[:, None] * corr)
        size = np.linalg.norm(disp, axis=1)
        disp *= np.minimum(1.0, 0.8 * self.radius / np.maximum(size, 1e-12))[:, None]
        x += disp
        x[0], x[-1] = self.left, self.right

    def _substep(self, dt: float) -> None:
        old = self.x.copy()
        self.v += dt * self._forces(self.time + 0.5 * dt) * self.inv_mass[:, None]
        self.v *= math.exp(-0.018 * dt)
        speed = np.linalg.norm(self.v, axis=1)
        self.v *= np.minimum(1.0, 20.0 / np.maximum(speed, 1e-12))[:, None]
        trial = self.x + dt * self.v
        trial[0], trial[-1] = self.left, self.right
        self._project_lengths(trial)
        self._project_self_contact(trial)
        self._project_lengths(trial)
        self.x = trial
        self.v = (self.x - old) / dt
        self.v[[0, -1]] = 0.0
        self.time += dt
        self.step_count += 1

    def _linear_momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.v, axis=0)

    def _angular_momentum(self) -> np.ndarray:
        return np.sum(np.cross(self.x, self.mass[:, None] * self.v), axis=0)

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av, aw, bv, bw = a[:, :3], a[:, 3:4], b[:, :3], b[:, 3:4]
        return np.concatenate((aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, axis=1, keepdims=True)), axis=1)

    def _orientation_and_spin(self, time: float):
        tangent = self._node_tangents(self.x)
        angle, angle_rate = self._twist_kinematics(time)
        frac = np.linspace(0.0, 1.0, self.n)
        theta, theta_rate = (frac - 0.5) * angle, (frac - 0.5) * angle_rate
        align = np.empty((self.n, 4))
        align[:, 0] = 0.0
        align[:, 1], align[:, 2], align[:, 3] = -tangent[:, 2], tangent[:, 1], 1.0 + tangent[:, 0]
        bad = np.linalg.norm(align, axis=1) < 1e-8
        align[bad] = np.array([0.0, 0.0, 1.0, 0.0])
        align /= np.maximum(np.linalg.norm(align, axis=1)[:, None], 1e-12)
        twist = np.zeros((self.n, 4)); twist[:, 0] = np.sin(theta / 2.0); twist[:, 3] = np.cos(theta / 2.0)
        orientation = self._quat_multiply(align, twist)
        orientation /= np.maximum(np.linalg.norm(orientation, axis=1)[:, None], 1e-12)
        edge = self.x[1:] - self.x[:-1]
        length = np.linalg.norm(edge, axis=1)
        et = edge / np.maximum(length[:, None], 1e-12)
        dv = self.v[1:] - self.v[:-1]
        td = dv - np.sum(dv * et, axis=1)[:, None] * et
        td /= np.maximum(length[:, None], 1e-12)
        tdot = np.empty_like(self.x); tdot[0], tdot[-1] = td[0], td[-1]; tdot[1:-1] = 0.5 * (td[:-1] + td[1:])
        omega = np.cross(tangent, tdot) + theta_rate[:, None] * tangent
        omega[[0, -1]] = theta_rate[[0, -1], None] * tangent[[0, -1]]
        return orientation, omega, theta

    def _make_frame(self, time: float) -> StateFrame:
        orientation, omega, theta = self._orientation_and_spin(time)
        p, h = self._linear_momentum(), self._angular_momentum()
        elapsed = float(time) - float(self.last_output_time)
        nf = (p - self.last_output_momentum) / elapsed if elapsed > 1e-12 else np.zeros(3)
        nt = (h - self.last_output_angular_momentum) / elapsed if elapsed > 1e-12 else np.zeros(3)
        endpoint = np.zeros(self.n); endpoint[[0, -1]] = 1.0
        fields = {
            "budget_cumulative_external_angular_impulse": (h - self.initial_angular_momentum)[None, :],
            "budget_cumulative_external_impulse": (p - self.initial_momentum)[None, :],
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)), "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)), "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)), "budget_external_couple": nt[None, None, :],
            "budget_external_force": nf[None, None, :], "budget_force_position": np.zeros((1, 1, 3)),
            "constraint_target": np.stack((self.left, self.right)), "constraint_value": self.x[[0, -1]].copy(),
            "cumulative_mass_source": np.zeros(1),
        }
        frame = StateFrame(time=float(time), position=self.x.copy(), velocity=self.v.copy(), orientation=orientation,
                           angular_velocity=omega, mass=self.mass.copy(), attributes={
                               "arc_fraction": np.linspace(0.0, 1.0, self.n), "rod_radius": np.full(self.n, self.radius),
                               "contact_diameter": np.full(self.n, self.diameter), "endpoint_controlled": endpoint,
                               "material_twist_turns": theta / (2.0 * math.pi), "inertia_tensor": np.zeros((self.n, 3, 3)),
                               "mass_quantity": self.mass.copy()}, fields=fields,
                           scalars={"rest_length": self.total_length, "constraint_scale": float(np.linalg.norm(self.right - self.left)),
                                    "reference_momentum": self.reference_momentum, "reference_angular_momentum": self.reference_angular_momentum})
        frame.validate()
        self.last_output_time, self.last_output_momentum, self.last_output_angular_momentum = float(time), p.copy(), h.copy()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        remaining = target - self.time
        if remaining > 1e-12:
            count = max(1, int(math.ceil(remaining / self.max_internal_dt)))
            for _ in range(count):
                self._substep(remaining / count)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        lengths = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)
        return {"method": "discrete_kirchhoff_rod_shake_contact", "internal_steps": int(self.step_count),
                "maximum_relative_length_error": float(np.max(np.abs(lengths - self.ds)) / self.ds),
                "finite": bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.v)))}
