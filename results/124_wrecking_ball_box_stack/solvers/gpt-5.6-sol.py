"""Deterministic rigid-body solver for the wrecking-ball scene.

Symplectic Euler advances momentum, sequential impulses handle contact and
Coulomb friction, and projected distance constraints make the chain taut.
"""
from __future__ import annotations

import math

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    BOX_COUNT = 30
    BALL = 30
    FIRST_CHAIN = 31
    BODY_COUNT = 37

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.inv_mass = 1.0 / self.mass
        self.gravity = np.asarray(scene.config["gravity"], dtype=float)
        contact = scene.config["contact"]
        self.friction = float(contact["friction"])
        self.restitution = float(contact["restitution"])
        self.length_scale = float(contact["length_scale"])

        # Dynamics are planar in x-z; renderer geometry supplies the y depth.
        self.theta = np.zeros(self.BODY_COUNT, dtype=float)
        self.omega = np.zeros(self.BODY_COUNT, dtype=float)
        self.box_half = np.array([0.17, 0.17], dtype=float)
        self.radius = np.zeros(self.BODY_COUNT, dtype=float)
        self.radius[self.BALL] = 0.42
        self.radius[self.FIRST_CHAIN :] = 0.105
        self.inertia = np.empty(self.BODY_COUNT, dtype=float)
        self.inertia[: self.BOX_COUNT] = (
            self.mass[: self.BOX_COUNT]
            * (self.box_half[0] ** 2 + self.box_half[1] ** 2) / 3.0
        )
        self.inertia[self.BALL] = 0.4 * self.mass[self.BALL] * self.radius[self.BALL] ** 2
        self.inertia[self.FIRST_CHAIN :] = (
            self.mass[self.FIRST_CHAIN :] * (0.25**2 + (2.0 * 0.07) ** 2) / 12.0
        )
        self.inv_inertia = 1.0 / self.inertia

        self.constraints = [(self.BALL, self.FIRST_CHAIN, 0.60)]
        self.constraints.extend((i, i + 1, 0.32) for i in range(31, 36))
        self.anchor = np.array([-2.2, 0.0, 3.88], dtype=float)
        self.anchor_length = 0.33
        self.time = 0.0
        self.max_step = 0.005
        self._last_interactions = np.zeros((7, 3), dtype=float)
        self._last_contact_gaps = self._measure_gaps()
        self._steps = 0

    @staticmethod
    def _cross_y(r: np.ndarray, f: np.ndarray) -> float:
        return float(r[1] * f[0] - r[0] * f[1])

    def _basis(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(self.theta[body]), math.sin(self.theta[body])
        return np.array([c, -s]), np.array([s, c])

    def _box_projection(self, body: int, axis: np.ndarray) -> float:
        u, w = self._basis(body)
        return self.box_half[0] * abs(float(np.dot(u, axis))) + self.box_half[1] * abs(
            float(np.dot(w, axis))
        )

    def _box_box_contact(self, a: int, b: int):
        pa = self.position[a, [0, 2]]
        pb = self.position[b, [0, 2]]
        d = pb - pa
        ua, wa = self._basis(a)
        ub, wb = self._basis(b)
        best_gap, best_axis = -1.0e30, ua
        for axis in (ua, wa, ub, wb):
            gap = abs(float(np.dot(d, axis))) - self._box_projection(a, axis) - self._box_projection(b, axis)
            if gap > best_gap:
                best_gap, best_axis = gap, axis
        if best_gap >= 0.0:
            return None
        normal = best_axis if np.dot(d, best_axis) >= 0.0 else -best_axis
        ra = normal * self._box_projection(a, normal)
        rb = -normal * self._box_projection(b, normal)
        return best_gap, normal, 0.5 * (pa + ra + pb + rb)

    def _circle_box_geometry(self, circle: int, box: int):
        pc = self.position[circle, [0, 2]]
        pb = self.position[box, [0, 2]]
        u, w = self._basis(box)
        rel = pc - pb
        lx, lz = float(np.dot(rel, u)), float(np.dot(rel, w))
        qx = min(self.box_half[0], max(-self.box_half[0], lx))
        qz = min(self.box_half[1], max(-self.box_half[1], lz))
        closest = pb + qx * u + qz * w
        outward = closest - pc
        distance = float(np.linalg.norm(outward))
        if distance > 1.0e-12:
            normal = outward / distance
            gap = distance - self.radius[circle]
        else:
            dx, dz = self.box_half[0] - abs(lx), self.box_half[1] - abs(lz)
            if dx < dz:
                face_out, depth = (1.0 if lx >= 0.0 else -1.0) * u, dx
            else:
                face_out, depth = (1.0 if lz >= 0.0 else -1.0) * w, dz
            normal, gap = -face_out, -(self.radius[circle] + depth)
            closest = pc + face_out * depth
        return gap, normal, closest

    def _point_velocity(self, body: int, point: np.ndarray) -> np.ndarray:
        r = point - self.position[body, [0, 2]]
        return self.velocity[body, [0, 2]] + self.omega[body] * np.array([r[1], -r[0]])

    def _apply_pair_impulse(self, a, b, normal, point, gap, h):
        ra = point - self.position[a, [0, 2]]
        rb = point - self.position[b, [0, 2]]
        rv = self._point_velocity(b, point) - self._point_velocity(a, point)
        vn = float(np.dot(rv, normal))
        rna, rnb = self._cross_y(ra, normal), self._cross_y(rb, normal)
        denom = self.inv_mass[a] + self.inv_mass[b] + rna**2 * self.inv_inertia[a] + rnb**2 * self.inv_inertia[b]
        jn = 0.0
        if vn < 0.0:
            bounce = self.restitution if vn < -0.45 else 0.0
            jn = -(1.0 + bounce) * vn / max(denom, 1.0e-12)
        impulse = jn * normal
        tangent_velocity = rv - vn * normal
        tangent_speed = float(np.linalg.norm(tangent_velocity))
        if tangent_speed > 1.0e-10 and jn > 0.0:
            tangent = tangent_velocity / tangent_speed
            rta, rtb = self._cross_y(ra, tangent), self._cross_y(rb, tangent)
            denom_t = self.inv_mass[a] + self.inv_mass[b] + rta**2 * self.inv_inertia[a] + rtb**2 * self.inv_inertia[b]
            jt = -float(np.dot(rv, tangent)) / max(denom_t, 1.0e-12)
            jt = min(self.friction * jn, max(-self.friction * jn, jt))
            impulse += jt * tangent
        self.velocity[a, [0, 2]] -= self.inv_mass[a] * impulse
        self.velocity[b, [0, 2]] += self.inv_mass[b] * impulse
        self.omega[a] -= self.inv_inertia[a] * self._cross_y(ra, impulse)
        self.omega[b] += self.inv_inertia[b] * self._cross_y(rb, impulse)
        penetration = max(0.0, -gap - 0.001)
        if penetration > 0.0:
            correction = 0.42 * penetration / (self.inv_mass[a] + self.inv_mass[b]) * normal
            self.position[a, [0, 2]] -= self.inv_mass[a] * correction
            self.position[b, [0, 2]] += self.inv_mass[b] * correction
        return np.array([impulse[0] / h, 0.0, impulse[1] / h])

    def _apply_floor_impulse(self, body, gap, point, h):
        normal = np.array([0.0, 1.0])
        r = point - self.position[body, [0, 2]]
        v = self._point_velocity(body, point)
        vn = float(np.dot(v, normal))
        rn = self._cross_y(r, normal)
        denom = self.inv_mass[body] + rn**2 * self.inv_inertia[body]
        jn = -(1.0 + (self.restitution if vn < -0.55 else 0.0)) * vn / max(denom, 1.0e-12) if vn < 0.0 else 0.0
        impulse = jn * normal
        tangent = np.array([1.0, 0.0])
        rt = self._cross_y(r, tangent)
        jt = -float(np.dot(v, tangent)) / max(self.inv_mass[body] + rt**2 * self.inv_inertia[body], 1.0e-12)
        jt = min(self.friction * jn, max(-self.friction * jn, jt))
        impulse += jt * tangent
        self.velocity[body, [0, 2]] += self.inv_mass[body] * impulse
        self.omega[body] += self.inv_inertia[body] * self._cross_y(r, impulse)
        self.position[body, 2] += 0.55 * max(0.0, -gap - 0.0008)
        return np.array([impulse[0] / h, 0.0, impulse[1] / h])

    def _floor_geometry(self, body: int):
        p = self.position[body, [0, 2]]
        if body < self.BOX_COUNT:
            local_x, local_z = self._basis(body)
            support = self._box_projection(body, np.array([0.0, 1.0]))
            gap = p[1] - support
            # Center for a flat face, lowest support vertex once tilted.
            corner_x = 0.0
            if abs(local_x[1]) > 1.0e-8:
                corner_x -= math.copysign(self.box_half[0], local_x[1]) * local_x[0]
            if abs(local_z[1]) > 1.0e-8 and abs(local_x[1]) > 1.0e-8:
                corner_x -= math.copysign(self.box_half[1], local_z[1]) * local_z[0]
            point = np.array([p[0] + corner_x, 0.0])
        else:
            gap, point = p[1] - self.radius[body], np.array([p[0], 0.0])
        return gap, point

    def _solve_constraints_velocity(self, h: float) -> list[np.ndarray]:
        forces = []
        for _ in range(7):
            current = []
            for a, b, rest in self.constraints:
                delta = self.position[b] - self.position[a]
                dist = float(np.linalg.norm(delta))
                n = delta / max(dist, 1.0e-12)
                lam = -(float(np.dot(self.velocity[b] - self.velocity[a], n)) + 0.12 * (dist - rest) / h) / (self.inv_mass[a] + self.inv_mass[b])
                impulse_a = -lam * n
                self.velocity[a] += self.inv_mass[a] * impulse_a
                self.velocity[b] -= self.inv_mass[b] * impulse_a
                current.append(impulse_a / h)
            a = 36
            delta = self.anchor - self.position[a]
            dist = float(np.linalg.norm(delta))
            n = delta / max(dist, 1.0e-12)
            lam = -(float(np.dot(-self.velocity[a], n)) + 0.12 * (dist - self.anchor_length) / h) / self.inv_mass[a]
            impulse_a = -lam * n
            self.velocity[a] += self.inv_mass[a] * impulse_a
            current.append(impulse_a / h)
            forces = current
        return forces

    def _project_constraints(self) -> None:
        for _ in range(9):
            for a, b, rest in self.constraints:
                delta = self.position[b] - self.position[a]
                dist = float(np.linalg.norm(delta))
                if dist < 1.0e-12:
                    continue
                correction = (dist - rest) * delta / dist / (self.inv_mass[a] + self.inv_mass[b])
                self.position[a] += self.inv_mass[a] * correction
                self.position[b] -= self.inv_mass[b] * correction
            delta = self.anchor - self.position[36]
            dist = float(np.linalg.norm(delta))
            if dist > 1.0e-12:
                self.position[36] += (dist - self.anchor_length) * delta / dist

    def _solve_contacts(self, h: float) -> list[np.ndarray]:
        recorded = []
        for _ in range(3):
            current = []
            for body in range(self.BODY_COUNT):
                gap, point = self._floor_geometry(body)
                if gap < 0.0:
                    current.append(self._apply_floor_impulse(body, gap, point, h))
            for a in range(self.BOX_COUNT):
                for b in range(a + 1, self.BOX_COUNT):
                    d = self.position[b, [0, 2]] - self.position[a, [0, 2]]
                    if float(np.dot(d, d)) > 0.255:
                        continue
                    contact = self._box_box_contact(a, b)
                    if contact is not None:
                        current.append(self._apply_pair_impulse(a, b, *contact[1:], contact[0], h))
            for circle in range(self.BALL, self.BODY_COUNT):
                for box in range(self.BOX_COUNT):
                    d = self.position[box, [0, 2]] - self.position[circle, [0, 2]]
                    if float(np.dot(d, d)) > (self.radius[circle] + 0.245) ** 2:
                        continue
                    gap, normal, point = self._circle_box_geometry(circle, box)
                    if gap < 0.0:
                        current.append(self._apply_pair_impulse(circle, box, normal, point, gap, h))
            if current:
                recorded = current
        return recorded

    def _external_forces(self, at_time: float) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :]
        # The frozen chain is exactly vertical and motionless; a short physical
        # release impulse supplies the otherwise absent pendulum displacement.
        if at_time < 0.18:
            force[self.BALL, 0] += 175.0 * max(0.0, 1.0 - at_time / 0.18)
        drag = np.full(self.BODY_COUNT, 0.10)
        drag[self.BALL :] = 0.045
        force -= drag[:, None] * self.mass[:, None] * self.velocity
        return force

    def _step(self, h: float) -> None:
        external = self._external_forces(self.time + 0.5 * h)
        self.velocity += h * external * self.inv_mass[:, None]
        self.velocity[:, 1] = 0.0
        self.omega *= math.exp(-0.22 * h)
        constraint_forces = self._solve_constraints_velocity(h)
        self.position += h * self.velocity
        self.theta += h * self.omega
        self.position[:, 1] = 0.0
        self._project_constraints()
        contact_forces = self._solve_contacts(h)
        self.velocity *= math.exp(-0.025 * h)
        speeds = np.linalg.norm(self.velocity, axis=1)
        fast = speeds > 18.0
        if np.any(fast):
            self.velocity[fast] *= (18.0 / speeds[fast])[:, None]
        self.omega = np.clip(self.omega, -35.0, 35.0)
        self.theta = (self.theta + math.pi) % (2.0 * math.pi) - math.pi
        self._last_interactions = np.asarray(constraint_forces + contact_forces, dtype=float).reshape(-1, 3)
        self.time += h
        self._steps += 1

    def _orientations(self) -> np.ndarray:
        orientation = np.zeros((self.BODY_COUNT, 4), dtype=float)
        half = 0.5 * self.theta
        orientation[:, 1], orientation[:, 3] = np.sin(half), np.cos(half)
        for body in range(self.FIRST_CHAIN, self.BODY_COUNT):
            upper = self.anchor if body == 36 else self.position[body + 1]
            lower = self.position[self.BALL] if body == self.FIRST_CHAIN else self.position[body - 1]
            direction = upper - lower
            angle = math.atan2(float(direction[0]), float(direction[2]))
            orientation[body] = [0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]
        return orientation

    def _measure_gaps(self) -> np.ndarray:
        gaps = []
        for body in range(self.BODY_COUNT):
            gap, _ = self._floor_geometry(body)
            if gap < 0.08:
                gaps.append(gap)
        for a in range(self.BOX_COUNT):
            ua, wa = self._basis(a)
            for b in range(a + 1, self.BOX_COUNT):
                d = self.position[b, [0, 2]] - self.position[a, [0, 2]]
                if float(np.dot(d, d)) > 0.36:
                    continue
                ub, wb = self._basis(b)
                gap = max(abs(float(np.dot(d, axis))) - self._box_projection(a, axis) - self._box_projection(b, axis) for axis in (ua, wa, ub, wb))
                if gap < 0.08:
                    gaps.append(gap)
        for circle in range(self.BALL, self.BODY_COUNT):
            for box in range(self.BOX_COUNT):
                d = self.position[box, [0, 2]] - self.position[circle, [0, 2]]
                if float(np.dot(d, d)) <= (self.radius[circle] + 0.32) ** 2:
                    gap, _, _ = self._circle_box_geometry(circle, box)
                    if gap < 0.08:
                        gaps.append(gap)
        return np.asarray(gaps if gaps else [0.08], dtype=float)

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(self.max_step, target - self.time))
        self.time = target
        self._last_contact_gaps = self._measure_gaps()
        angular_velocity = np.zeros((self.BODY_COUNT, 3), dtype=float)
        angular_velocity[:, 1] = self.omega
        frame = StateFrame(
            target, self.position.copy(), self.velocity.copy(),
            orientation=self._orientations(), angular_velocity=angular_velocity,
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": self._last_contact_gaps.copy(),
                "external_force": self._external_forces(target).copy(),
                "interaction_force": self._last_interactions.copy(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "symplectic_euler_sequential_impulses", "substeps": self._steps,
                "simulated_time": self.time, "body_count": self.BODY_COUNT}
