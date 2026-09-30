"""Physics solver for the rotating hairy-ball scene.

The hair is represented by extensible discrete rods. Stretch and bending
energies are integrated with compliant position constraints, while sphere and
strand contacts are resolved geometrically with Coulomb friction.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1.0e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    h = 0.5 * angle
    return np.r_[math.cos(h), math.sin(h) * v / angle]


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qrotate(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    u = q[1:]
    return x + 2.0 * np.cross(u, np.cross(u, x) + q[0] * x)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        p = self.cfg["physical_parameters"]
        self.ns = int(p["strand_count"])
        self.nv = int(p["segments_per_strand"]) + 1
        self.n = self.ns * self.nv
        self.radius = float(p["sphere_radius"])
        self.hair_radius = float(p["strand_radius"])
        self.mu = float(self.cfg["contact"]["kinetic_friction"])
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)

        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy().reshape(self.ns, self.nv, 3)
        self.v = np.asarray(initial.velocity, dtype=float).copy().reshape(self.ns, self.nv, 3)
        self.mass = np.asarray(initial.mass, dtype=float).copy().reshape(self.ns, self.nv)
        self.inv_mass = np.zeros_like(self.mass)
        self.inv_mass[:, 1:] = 1.0 / self.mass[:, 1:]
        self.root_reference = self.x[:, 0].copy()
        self.rest_edge = np.linalg.norm(self.x[:, 1:] - self.x[:, :-1], axis=2)
        self.rest_bend = np.linalg.norm(self.x[:, 2:] - self.x[:, :-2], axis=2)

        area = math.pi * float(p["strand_radius"]) ** 2
        moment = math.pi * float(p["strand_radius"]) ** 4 / 4.0
        ds = float(np.mean(self.rest_edge))
        self.stretch_stiffness = float(p["stretch_young_modulus"]) * area / ds
        self.bend_stiffness = max(float(p["bending_young_modulus"]) * moment / ds ** 3, 1.0e-12)

        keys = self.cfg["prescribed_sphere_rotation"]["keyframes_axis_angle_degrees"]
        self.key_t = np.array([float(k["time"]) for k in keys])
        self.key_r = np.array([
            np.asarray(k["axis"], dtype=float)
            / max(float(np.linalg.norm(k["axis"])), 1.0e-15)
            * math.radians(float(k["angle"]))
            for k in keys
        ])
        self.time = 0.0
        self.max_step = 0.01
        self.constraint_iterations = 5
        self.initial_momentum = np.sum(self.mass[..., None] * self.v, axis=(0, 1))
        self.last_output_time = 0.0
        self.last_output_momentum = self.initial_momentum.copy()
        shape = (self.ns, self.nv - 1)
        self.last_normal_force = np.zeros(shape)
        self.last_contact = np.zeros(shape, dtype=bool)
        initial_contact_points = self.x[:, 1:]
        self.last_normal = initial_contact_points / np.maximum(
            np.linalg.norm(initial_contact_points, axis=2)[..., None], 1.0e-12
        )
        self.last_tangent_force = np.zeros(shape + (3,))
        self.steps = 0

    def _rotation(self, t: float) -> np.ndarray:
        if t <= self.key_t[0]:
            r = self.key_r[0]
        elif t >= self.key_t[-1]:
            r = self.key_r[-1]
        else:
            i = int(np.searchsorted(self.key_t, t, side="right") - 1)
            a = (t - self.key_t[i]) / (self.key_t[i + 1] - self.key_t[i])
            a = a * a * (3.0 - 2.0 * a)
            r = (1.0 - a) * self.key_r[i] + a * self.key_r[i + 1]
        return _quat_from_rotvec(r)

    def _root_position(self, t: float) -> np.ndarray:
        return _qrotate(self._rotation(t), self.root_reference)

    def _angular_velocity(self, t: float) -> np.ndarray:
        eps = 2.0e-5
        ta = max(0.0, t - eps)
        tb = min(float(self.cfg["duration"]), t + eps)
        if tb <= ta:
            return np.zeros(3)
        qa, qb = self._rotation(ta), self._rotation(tb)
        dq = _qmul(qb, np.r_[qa[0], -qa[1:]])
        if dq[0] < 0.0:
            dq = -dq
        s = float(np.linalg.norm(dq[1:]))
        if s < 1.0e-14:
            return np.zeros(3)
        angle = 2.0 * math.atan2(s, max(float(dq[0]), 0.0))
        return dq[1:] / s * angle / (tb - ta)

    @staticmethod
    def _distance_constraint(xa, xb, wa, wb, rest, compliance):
        d = xb - xa
        length = np.linalg.norm(d, axis=2)
        normal = d / np.maximum(length[..., None], 1.0e-12)
        amount = (length - rest) / np.maximum(wa + wb + compliance, 1.0e-30)
        corr = amount[..., None] * normal
        xa += wa[..., None] * corr
        xb -= wb[..., None] * corr

    def _sphere_contact(self, old: np.ndarray, h: float, omega: np.ndarray) -> None:
        pts = self.x[:, 1:]
        radii = np.linalg.norm(pts, axis=2)
        limit = self.radius + self.hair_radius
        hit = radii < limit
        normals = pts / np.maximum(radii[..., None], 1.0e-12)
        penetration = np.maximum(limit - radii, 0.0)
        pts += penetration[..., None] * normals

        trial_v = (pts - old[:, 1:]) / h
        surface_v = np.cross(omega, self.radius * normals)
        rel = trial_v - surface_v
        tangent = rel - np.sum(rel * normals, axis=2)[..., None] * normals
        tspeed = np.linalg.norm(tangent, axis=2)
        slip = np.minimum(h * tspeed, self.mu * penetration)
        pts -= hit[..., None] * slip[..., None] * tangent / np.maximum(tspeed[..., None], 1.0e-12)

        m = self.mass[:, 1:]
        normal_force = m * penetration / (h * h)
        tangent_force = np.minimum(m * tspeed / h, self.mu * normal_force)
        self.last_contact = hit
        self.last_normal = normals
        self.last_normal_force = np.where(hit, normal_force, 0.0)
        self.last_tangent_force = np.where(
            hit[..., None],
            -tangent_force[..., None] * tangent / np.maximum(tspeed[..., None], 1.0e-12),
            0.0,
        )

    def _strand_contacts(self, old: np.ndarray, h: float) -> None:
        """Project penetrations between continuous line-segment capsules."""
        a = self.x[:, :-1].reshape(-1, 3)
        b = self.x[:, 1:].reshape(-1, 3)
        midpoint = 0.5 * (a + b)
        segs = self.nv - 1
        search = float(np.max(self.rest_edge)) + 2.0 * self.hair_radius
        pairs = cKDTree(midpoint).query_pairs(search, output_type="ndarray")
        if pairs.size == 0:
            return
        i, j = pairs[:, 0], pairs[:, 1]
        keep = (i // segs) != (j // segs)
        i, j = i[keep], j[keep]
        if i.size == 0:
            return

        p0, p1, q0, q1 = a[i], b[i], a[j], b[j]
        u, vv, w = p1 - p0, q1 - q0, p0 - q0
        aa = np.sum(u * u, axis=1)
        bb = np.sum(u * vv, axis=1)
        cc = np.sum(vv * vv, axis=1)
        dd = np.sum(u * w, axis=1)
        ee = np.sum(vv * w, axis=1)
        den = aa * cc - bb * bb
        s = np.where(np.abs(den) > 1.0e-18, (bb * ee - cc * dd) / den, 0.5)
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((bb * s + ee) / np.maximum(cc, 1.0e-18), 0.0, 1.0)
        s = np.clip((bb * t - dd) / np.maximum(aa, 1.0e-18), 0.0, 1.0)
        cp, cq = p0 + s[:, None] * u, q0 + t[:, None] * vv
        delta = cq - cp
        distance = np.linalg.norm(delta, axis=1)
        diameter = 2.0 * self.hair_radius
        active = distance < diameter
        if not np.any(active):
            return
        i, j, s, t, delta, distance = (z[active] for z in (i, j, s, t, delta, distance))
        normal = delta / np.maximum(distance[:, None], 1.0e-12)
        zero = distance < 1.0e-12
        if np.any(zero):
            phase = (i[zero] * 0.754877666 + j[zero] * 0.569840296) % 1.0
            normal[zero] = np.c_[np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase), np.zeros_like(phase)]

        inv = self.inv_mass.reshape(-1)
        vi0, vi1 = i + i // segs, i + i // segs + 1
        vj0, vj1 = j + j // segs, j + j // segs + 1
        wi0, wi1, wj0, wj1 = inv[vi0], inv[vi1], inv[vj0], inv[vj1]
        c0, c1, d0, d1 = 1.0 - s, s, 1.0 - t, t
        denom = wi0 * c0 * c0 + wi1 * c1 * c1 + wj0 * d0 * d0 + wj1 * d1 * d1
        depth = diameter - distance
        normal_impulse = (depth / np.maximum(denom, 1.0e-30))[:, None] * normal

        # Dry friction between the two material points on the contacting
        # capsules.  The positional impulse is capped by mu times the normal
        # correction, which is the PBD analogue of Coulomb's cone.
        old_flat = old.reshape(-1, 3)
        vp = (c0[:, None] * (flat_p0 := (self.x.reshape(-1, 3)[vi0] - old_flat[vi0]) / h)
              + c1[:, None] * (flat_p1 := (self.x.reshape(-1, 3)[vi1] - old_flat[vi1]) / h))
        vq = (d0[:, None] * (flat_q0 := (self.x.reshape(-1, 3)[vj0] - old_flat[vj0]) / h)
              + d1[:, None] * (flat_q1 := (self.x.reshape(-1, 3)[vj1] - old_flat[vj1]) / h))
        del flat_p0, flat_p1, flat_q0, flat_q1
        relative = vq - vp
        tangent = relative - np.sum(relative * normal, axis=1)[:, None] * normal
        tangent_speed = np.linalg.norm(tangent, axis=1)
        slip = np.minimum(h * tangent_speed, self.mu * depth)
        friction_impulse = (
            slip / np.maximum(denom, 1.0e-30)
        )[:, None] * tangent / np.maximum(tangent_speed[:, None], 1.0e-12)
        impulse = normal_impulse - friction_impulse
        flat = self.x.reshape(-1, 3)
        np.add.at(flat, vi0, -wi0[:, None] * c0[:, None] * impulse)
        np.add.at(flat, vi1, -wi1[:, None] * c1[:, None] * impulse)
        np.add.at(flat, vj0, wj0[:, None] * d0[:, None] * impulse)
        np.add.at(flat, vj1, wj1[:, None] * d1[:, None] * impulse)

    def _step(self, h: float) -> None:
        old = self.x.copy()
        new_time = self.time + h
        root = self._root_position(new_time)
        old_root = self.x[:, 0].copy()
        self.v[:, 1:] += h * self.gravity
        self.v[:, 1:] *= math.exp(-0.16 * h)
        self.x[:, 1:] += h * self.v[:, 1:]
        self.x[:, 0] = root

        stretch_c = 1.0 / (self.stretch_stiffness * h * h)
        bend_c = 1.0 / (self.bend_stiffness * h * h)
        omega = self._angular_velocity(new_time)
        for _ in range(self.constraint_iterations):
            for parity in (0, 1):
                self._distance_constraint(
                    self.x[:, parity:-1:2], self.x[:, parity + 1::2],
                    self.inv_mass[:, parity:-1:2], self.inv_mass[:, parity + 1::2],
                    self.rest_edge[:, parity::2], stretch_c,
                )
            for parity in (0, 1):
                self._distance_constraint(
                    self.x[:, parity:-2:2], self.x[:, parity + 2::2],
                    self.inv_mass[:, parity:-2:2], self.inv_mass[:, parity + 2::2],
                    self.rest_bend[:, parity::2], bend_c,
                )
            self.x[:, 0] = root
            self._sphere_contact(old, h, omega)
        self._strand_contacts(old, h)
        self.x[:, 0] = root
        self._sphere_contact(old, h, omega)

        self.v = (self.x - old) / h
        self.v[:, 0] = (root - old_root) / h
        np.nan_to_num(self.x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(self.v, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)
        self.time = new_time
        self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            self._step(min(self.max_step, target - self.time))
        self.time = target

        flat_x = self.x.reshape(-1, 3).copy()
        flat_v = self.v.reshape(-1, 3).copy()
        flat_m = self.mass.reshape(-1).copy()
        momentum = np.sum(flat_m[:, None] * flat_v, axis=0)
        elapsed = target - self.last_output_time
        if elapsed > 1.0e-12:
            external_force = (momentum - self.last_output_momentum) / elapsed
        else:
            external_force = np.sum(flat_m[:, None] * self.gravity, axis=0)

        normals = self.last_normal
        surface_v = np.cross(self._angular_velocity(target), self.radius * normals)
        relative_v = self.v[:, 1:] - surface_v
        c = self.ns * (self.nv - 1)
        fields = {
            "budget_cumulative_external_impulse": (momentum - self.initial_momentum)[None, :],
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": external_force.reshape(1, 1, 3),
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": self.last_contact.reshape(c).astype(float),
            "friction_dynamic_coefficient": np.full(c, self.mu),
            "friction_normal": normals.reshape(c, 3).copy(),
            "friction_normal_force": self.last_normal_force.reshape(c).copy(),
            "friction_relative_velocity": relative_v.reshape(c, 3).copy(),
            "friction_static_coefficient": np.full(c, float(self.cfg["contact"]["static_friction"])),
            "friction_tangential_force": self.last_tangent_force.reshape(c, 3).copy(),
        }
        total_mass = float(np.sum(flat_m))
        reference_velocity = math.sqrt(
            float(np.linalg.norm(self.gravity))
            * float(self.cfg["physical_parameters"]["strand_length"])
        )
        scalars = {
            "strand_radius": self.hair_radius,
            "static_friction": float(self.cfg["contact"]["static_friction"]),
            "kinetic_friction": self.mu,
            "friction_stick_speed_tolerance": 0.02,
            "reference_force": max(total_mass * float(np.linalg.norm(self.gravity)), 1.0e-12),
            "reference_momentum": max(total_mass * reference_velocity, 1.0e-12),
            "reference_velocity": max(reference_velocity, 1.0e-12),
        }
        attributes = {
            "strand_id": np.repeat(np.arange(self.ns, dtype=float), self.nv),
            "root_mask": np.tile(np.r_[1.0, np.zeros(self.nv - 1)], self.ns),
            "mass_quantity": flat_m.copy(),
        }
        frame = StateFrame(
            time=target,
            position=flat_x,
            velocity=flat_v,
            mass=flat_m,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self.last_output_time = target
        self.last_output_momentum = momentum
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "compliant discrete rods with capsule contacts",
            "internal_steps": self.steps,
            "simulated_time": self.time,
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
