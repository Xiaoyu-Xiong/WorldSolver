"""Damped discrete Cosserat-rod solver for the telephone-cord perversion."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)), axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[..., :3]
    t = 2.0 * np.cross(qv, v)
    return v + q[..., 3:4] * t + np.cross(qv, t)


def _rotvec(q: np.ndarray) -> np.ndarray:
    qq = q.copy()
    flip = qq[..., 3] < 0.0
    qq[flip] *= -1.0
    nv = np.linalg.norm(qq[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(nv, np.clip(qq[..., 3], 0.0, 1.0))
    scale = np.where(nv > 1.0e-12, angle / np.maximum(nv, 1.0e-12), 2.0)
    return qq[..., :3] * scale[..., None]


def _qstep(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    rv = omega * dt
    angle = np.linalg.norm(rv, axis=1)
    half = 0.5 * angle
    scale = np.where(angle > 1.0e-12, np.sin(half) / np.maximum(angle, 1.0e-12), 0.5)
    dq = np.empty_like(q)
    dq[:, :3] = rv * scale[:, None]
    dq[:, 3] = np.cos(half)
    out = _qmul(dq, q)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1.0e-15)
    return out


class BenchmarkSolver:
    """Numerically integrate a strained, inextensible elastic rod."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.q = np.asarray(initial.orientation, dtype=float).copy()
        self.w = np.zeros_like(self.x)
        self.n = self.x.shape[0]
        rod = self.cfg["rod"]
        self.length = float(rod["total_length"])
        self.ds = self.length / (self.n - 1)
        self.node_mass = float(rod["mass"]) / self.n
        self.mass = np.full(self.n, self.node_mass, dtype=float)
        self.time = 0.0
        radius = float(rod["natural_shape"]["radius"])
        rise = float(rod["natural_shape"]["pitch_per_turn"]) / (2.0 * math.pi)
        denom = radius * radius + rise * rise
        self.kappa0, self.tau0 = radius / denom, rise / denom
        self.intrinsic = np.array([self.tau0, self.kappa0, 0.0]) * self.ds
        width = float(rod["cross_section"]["width"])
        thick = float(rod["cross_section"]["thickness"])
        bend_soft = 0.0032
        self.rot_stiff = np.array([0.0018, bend_soft, bend_soft * (width / thick) ** 2])
        self.shear_stiff = 9.0
        self.rot_inertia = np.full((self.n, 3), 1.2e-4)
        self.linear_damping, self.angular_damping = 10.0, 14.0
        self.max_step, self.projection_iterations = 0.001, 5
        self.contact_distance, self.contact_stiff = max(thick, 0.010), 5.0
        self._step_count = 0
        self._contact_force = np.zeros_like(self.x)
        self._last_reaction_force = np.zeros((2, 3))
        self._last_reaction_couple = np.zeros((2, 3))
        self._last_energy = 0.0
        self._apply_endpoints(0.0, set_velocity=True)

    def _target_data(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        ctl = self.cfg["endpoint_controls"]
        motion = ctl["motion"]
        t0, t1 = float(motion["move_start"]), float(motion["move_end"])
        raw = float(np.clip((time - t0) / (t1 - t0), 0.0, 1.0))
        smooth = raw * raw * (3.0 - 2.0 * raw)
        rate = 6.0 * raw * (1.0 - raw) / (t1 - t0) if 0.0 < raw < 1.0 else 0.0
        pos, vel = np.empty((2, 3)), np.empty((2, 3))
        for j, side in enumerate(("left", "right")):
            start = np.asarray(ctl[side]["initial_position"], dtype=float)
            final = np.asarray(ctl[side]["final_position"], dtype=float)
            pos[j] = start + smooth * (final - start)
            vel[j] = rate * (final - start)
        return pos, vel

    def _apply_endpoints(self, time: float, set_velocity: bool = True) -> None:
        pos, vel = self._target_data(time)
        self.x[0], self.x[-1] = pos
        if set_velocity:
            self.v[0], self.v[-1] = vel
        self.q[0] = (0.0, 0.0, 0.0, 1.0)
        self.q[-1] = (0.0, 0.0, 0.0, 1.0)
        self.w[0] = self.w[-1] = 0.0

    def _elastic_forces(self) -> tuple[np.ndarray, np.ndarray, float]:
        force, torque = np.zeros_like(self.x), np.zeros_like(self.x)
        edge = self.x[1:] - self.x[:-1]
        axis = np.zeros_like(edge)
        axis[:, 0] = 1.0
        director = _qrotate(self.q[:-1], axis)
        shear_error = edge - self.ds * director
        ef = self.shear_stiff * shear_error
        force[:-1] += ef
        force[1:] -= ef
        torque[:-1] += self.shear_stiff * self.ds * np.cross(director, shear_error)
        relative = _qmul(_qconj(self.q[:-1]), self.q[1:])
        strain_error = _rotvec(relative) - self.intrinsic[None, :]
        body_moment = strain_error * (self.rot_stiff / self.ds)[None, :]
        world_moment = _qrotate(self.q[:-1], body_moment)
        torque[:-1] += world_moment
        torque[1:] -= world_moment
        energy = 0.5 * self.shear_stiff * float(np.sum(shear_error ** 2))
        energy += 0.5 / self.ds * float(np.sum(strain_error ** 2 * self.rot_stiff[None, :]))
        return force, torque, energy

    def _update_contact(self) -> None:
        delta = self.x[:, None, :] - self.x[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        ii, jj = np.triu_indices(self.n, k=3)
        d2 = dist2[ii, jj]
        active = d2 < self.contact_distance ** 2
        force = np.zeros_like(self.x)
        if np.any(active):
            ia, ja = ii[active], jj[active]
            dv = delta[ia, ja]
            distance = np.sqrt(np.maximum(d2[active], 1.0e-12))
            normal = dv / distance[:, None]
            pair_force = self.contact_stiff * (self.contact_distance - distance)[:, None] * normal
            np.add.at(force, ia, pair_force)
            np.add.at(force, ja, -pair_force)
        self._contact_force = force

    def _project_lengths(self) -> None:
        for _ in range(self.projection_iterations):
            for parity in (0, 1):
                i = np.arange(parity, self.n - 1, 2)
                j = i + 1
                edge = self.x[j] - self.x[i]
                distance = np.linalg.norm(edge, axis=1)
                correction = ((distance - self.ds) / np.maximum(distance, 1.0e-12))[:, None] * edge
                wi = np.ones(i.size)
                wj = np.ones(i.size)
                wi[i == 0] = 0.0
                wj[j == self.n - 1] = 0.0
                denom = np.maximum(wi + wj, 1.0)
                self.x[i] += correction * (wi / denom)[:, None]
                self.x[j] -= correction * (wj / denom)[:, None]

    def _substep(self, dt: float, new_time: float) -> None:
        old_x = self.x.copy()
        force, torque, energy = self._elastic_forces()
        if self._step_count % 8 == 0:
            self._update_contact()
        force += self._contact_force
        self._last_reaction_force[0], self._last_reaction_force[1] = -force[0], -force[-1]
        self._last_reaction_couple[0], self._last_reaction_couple[1] = -torque[0], -torque[-1]
        self.v[1:-1] += dt * force[1:-1] / self.node_mass
        self.v[1:-1] *= math.exp(-self.linear_damping * dt)
        self.x[1:-1] += dt * self.v[1:-1]
        self.w[1:-1] += dt * torque[1:-1] / self.rot_inertia[1:-1]
        self.w[1:-1] *= math.exp(-self.angular_damping * dt)
        self.q[1:-1] = _qstep(self.q[1:-1], self.w[1:-1], dt)
        self._apply_endpoints(new_time, set_velocity=False)
        self._project_lengths()
        self._apply_endpoints(new_time, set_velocity=False)
        self.v[1:-1] = (self.x[1:-1] - old_x[1:-1]) / dt
        _, endpoint_velocity = self._target_data(new_time)
        self.v[0], self.v[-1] = endpoint_velocity
        self._last_energy = energy
        self._step_count += 1

    def _make_frame(self, time: float) -> StateFrame:
        target_pos, _ = self._target_data(time)
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        fields = {
            "budget_external_force": self._last_reaction_force[None, :, :].copy(),
            "budget_force_position": self.x[[0, -1]][None, :, :].copy(),
            "budget_external_couple": self._last_reaction_couple[None, :, :].copy(),
            "constraint_target": np.concatenate((target_pos, np.tile(identity, (2, 1))), axis=1),
            "constraint_value": np.concatenate((self.x[[0, -1]], self.q[[0, -1]]), axis=1),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        attributes: dict[str, Any] = {
            "mass_quantity": self.mass.copy(),
            "arc_fraction": np.linspace(0.0, 1.0, self.n),
            "material_axis_1": _qrotate(self.q, np.tile(np.array([0.0, 1.0, 0.0]), (self.n, 1))),
            "cross_section_width": np.full(self.n, self.cfg["rod"]["cross_section"]["width"]),
            "cross_section_thickness": np.full(self.n, self.cfg["rod"]["cross_section"]["thickness"]),
        }
        scalars = {"constraint_scale": self.length, "reference_force": 1.0, "reference_torque": 0.01}
        kwargs = dict(orientation=self.q.copy(), angular_velocity=self.w.copy(), mass=self.mass.copy(), attributes=attributes)
        try:
            frame = StateFrame(float(time), self.x.copy(), self.v.copy(), fields=fields, scalars=scalars, **kwargs)
        except TypeError:
            frame = StateFrame(float(time), self.x.copy(), self.v.copy(), **kwargs)
            frame.fields.update(fields)
            frame.scalars.update(scalars)
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be nondecreasing")
        remaining = target - self.time
        if remaining > 1.0e-14:
            steps = max(1, int(math.ceil(remaining / self.max_step)))
            dt = remaining / steps
            for step in range(steps):
                self._substep(dt, self.time + (step + 1) * dt)
            self.time = target
        else:
            self._apply_endpoints(target, set_velocity=True)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        edge_length = np.linalg.norm(self.x[1:] - self.x[:-1], axis=1)
        return {"method": "damped discrete Cosserat rod with PBD inextensibility",
                "time": float(self.time),
                "elastic_energy": float(self._last_energy),
                "maximum_length_error": float(np.max(np.abs(edge_length - self.ds))),
                "finite": bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.q)))}
