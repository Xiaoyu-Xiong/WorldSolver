"""Deterministic rigid-body solver for the card-house impact scene.

The scene is symmetric in y, so the dynamics are solved in the x-z plane while
retaining the full three-dimensional StateFrame representation. Bodies are
oriented rectangles; contact uses SAT and sequential Coulomb impulses.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small self-contained planar rigid-body engine."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        meta = scene.metadata()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.n = len(self.mass)
        q = np.asarray(initial.orientation, dtype=float)
        self.angle = 2.0 * np.arctan2(q[:, 1], q[:, 3])
        self.omega = np.asarray(initial.angular_velocity, dtype=float)[:, 1].copy()
        self.half = np.array([[0.45, 0.015]] * 10 +
                             [[0.16, 0.16], [0.16, 0.16], [0.19, 0.19]],
                             dtype=float)
        self.inertia = self.mass * (self.half[:, 0] ** 2 +
                                    self.half[:, 1] ** 2) / 3.0
        self.gravity = float(meta["gravity"][2])
        self.mu = float(meta["friction_coefficient"])
        self.restitution = float(meta["restitution"])
        self.release_time = np.asarray(initial.attributes["release_time"],
                                       dtype=float).copy()
        self.pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.pair_slot = {pair: self.n + k for k, pair in enumerate(self.pairs)}
        self.contact_count = self.n + len(self.pairs)
        self.last_interaction = np.zeros((self.contact_count, 3), dtype=float)
        self.steps = 0
        self.max_penetration = 0.0

    @staticmethod
    def _axes(theta: float) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c, -s]), np.array([s, c])

    def _vertices(self, i: int) -> np.ndarray:
        u, v = self._axes(float(self.angle[i]))
        center = self.position[i, [0, 2]]
        hx, hz = self.half[i]
        return np.array([center - hx * u - hz * v,
                         center + hx * u - hz * v,
                         center + hx * u + hz * v,
                         center - hx * u + hz * v])

    def _box_gap(self, i: int, j: int):
        """Return signed SAT gap, A-to-B normal, and a contact point."""
        ui, vi = self._axes(float(self.angle[i]))
        uj, vj = self._axes(float(self.angle[j]))
        d = self.position[j, [0, 2]] - self.position[i, [0, 2]]
        best_gap = -float("inf")
        best_n = None
        for axis in (ui, vi, uj, vj):
            ra = (self.half[i, 0] * abs(float(np.dot(ui, axis))) +
                  self.half[i, 1] * abs(float(np.dot(vi, axis))))
            rb = (self.half[j, 0] * abs(float(np.dot(uj, axis))) +
                  self.half[j, 1] * abs(float(np.dot(vj, axis))))
            signed_center = float(np.dot(d, axis))
            gap = abs(signed_center) - ra - rb
            if gap > best_gap:
                best_gap = gap
                best_n = axis if signed_center >= 0.0 else -axis
        n = np.asarray(best_n, dtype=float)
        va, vb = self._vertices(i), self._vertices(j)
        pa, pb = va @ n, vb @ n
        fa = va[pa >= pa.max() - 1.0e-7].mean(axis=0)
        fb = vb[pb <= pb.min() + 1.0e-7].mean(axis=0)
        return float(best_gap), n, 0.5 * (fa + fb)

    def _active(self, i: int, at_time: float) -> bool:
        return i < 10 or at_time + 1.0e-12 >= self.release_time[i]

    def _point_velocity(self, i: int, r: np.ndarray) -> np.ndarray:
        return self.velocity[i, [0, 2]] + self.omega[i] * np.array([r[1], -r[0]])

    def _apply_impulse(self, i: int, impulse: np.ndarray, r: np.ndarray,
                       inv_mass: np.ndarray, inv_inertia: np.ndarray) -> None:
        self.velocity[i, 0] += impulse[0] * inv_mass[i]
        self.velocity[i, 2] += impulse[1] * inv_mass[i]
        self.omega[i] += (r[1] * impulse[0] - r[0] * impulse[1]) * inv_inertia[i]

    def _step(self, dt: float, impulse_sum: np.ndarray) -> None:
        end_time = self.time + dt
        active = np.array([self._active(i, end_time) for i in range(self.n)])
        inv_mass = np.where(active, 1.0 / self.mass, 0.0)
        inv_inertia = np.where(active, 1.0 / self.inertia, 0.0)
        self.velocity[active, 2] += self.gravity * dt
        self.velocity[active, :] *= math.exp(-0.10 * dt)
        self.omega[active] *= math.exp(-0.14 * dt)
        self.position[active, :] += self.velocity[active, :] * dt
        self.angle[active] += self.omega[active] * dt
        self.velocity[~active, :] = 0.0
        self.omega[~active] = 0.0

        skin = 5.0e-4
        contacts = []
        supported = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            if not active[i]:
                continue
            verts = self._vertices(i)
            min_z = float(verts[:, 1].min())
            if min_z <= skin:
                low = verts[verts[:, 1] <= min_z + 1.0e-7].mean(axis=0)
                contacts.append((i, -1, min_z, np.array([0.0, -1.0]), low, i))
                supported[i] = True
        radii = np.linalg.norm(self.half, axis=1)
        for i, j in self.pairs:
            if not (active[i] or active[j]):
                continue
            d = self.position[j, [0, 2]] - self.position[i, [0, 2]]
            if float(np.dot(d, d)) > (radii[i] + radii[j] + skin) ** 2:
                continue
            gap, normal, point = self._box_gap(i, j)
            if gap <= skin:
                contacts.append((i, j, gap, normal, point, self.pair_slot[(i, j)]))
                supported[i] = supported[j] = True
        if contacts:
            self.max_penetration = max(
                self.max_penetration, max(0.0, -min(c[2] for c in contacts)))

        for iteration in range(12):
            for i, j, gap, normal, point, slot in contacts:
                ra = point - self.position[i, [0, 2]]
                va = self._point_velocity(i, ra)
                if j >= 0:
                    rb = point - self.position[j, [0, 2]]
                    vb = self._point_velocity(j, rb)
                    im_b, ii_b = inv_mass[j], inv_inertia[j]
                else:
                    rb, vb = np.zeros(2), np.zeros(2)
                    im_b = ii_b = 0.0
                relative = vb - va
                vn = float(np.dot(relative, normal))
                arm_a = ra[1] * normal[0] - ra[0] * normal[1]
                arm_b = rb[1] * normal[0] - rb[0] * normal[1]
                effective = (inv_mass[i] + im_b + arm_a * arm_a * inv_inertia[i] +
                             arm_b * arm_b * ii_b)
                if effective <= 1.0e-14:
                    continue
                penetration = max(0.0, skin - gap)
                target = min(1.2, 0.16 * penetration / dt)
                if iteration == 0 and vn < -0.35:
                    target = max(target, -self.restitution * vn)
                jn = max(0.0, (target - vn) / effective)
                if jn <= 0.0:
                    continue
                normal_impulse = jn * normal
                self._apply_impulse(i, -normal_impulse, ra, inv_mass, inv_inertia)
                if j >= 0:
                    self._apply_impulse(j, normal_impulse, rb, inv_mass, inv_inertia)
                va2 = self._point_velocity(i, ra)
                vb2 = self._point_velocity(j, rb) if j >= 0 else np.zeros(2)
                tangent = np.array([-normal[1], normal[0]])
                vt = float(np.dot(vb2 - va2, tangent))
                arm_at = ra[1] * tangent[0] - ra[0] * tangent[1]
                arm_bt = rb[1] * tangent[0] - rb[0] * tangent[1]
                eff_t = (inv_mass[i] + im_b + arm_at * arm_at * inv_inertia[i] +
                         arm_bt * arm_bt * ii_b)
                jt = float(np.clip(-vt / max(eff_t, 1.0e-14),
                                   -self.mu * jn, self.mu * jn))
                tangent_impulse = jt * tangent
                self._apply_impulse(i, -tangent_impulse, ra, inv_mass, inv_inertia)
                if j >= 0:
                    self._apply_impulse(j, tangent_impulse, rb, inv_mass, inv_inertia)
                on_a = -normal_impulse - tangent_impulse
                impulse_sum[slot, 0] += on_a[0]
                impulse_sum[slot, 2] += on_a[1]

        for i, j, gap, normal, _point, _slot in contacts:
            error = max(0.0, -gap + 1.0e-5)
            if error <= 0.0:
                continue
            im_b = inv_mass[j] if j >= 0 else 0.0
            total = inv_mass[i] + im_b
            if total <= 0.0:
                continue
            correction = min(0.01, 0.55 * error) * normal / total
            self.position[i, 0] -= correction[0] * inv_mass[i]
            self.position[i, 2] -= correction[1] * inv_mass[i]
            if j >= 0:
                self.position[j, 0] += correction[0] * im_b
                self.position[j, 2] += correction[1] * im_b
        for i in range(self.n):
            if active[i] and supported[i]:
                speed = math.hypot(self.velocity[i, 0], self.velocity[i, 2])
                if speed < 0.012 and abs(self.omega[i]) < 0.035:
                    self.velocity[i, 0] = self.velocity[i, 2] = 0.0
                    self.omega[i] = 0.0
        self.time = end_time
        self.steps += 1

    def _signed_distances(self) -> np.ndarray:
        result = np.empty(self.contact_count, dtype=float)
        for i in range(self.n):
            result[i] = float(self._vertices(i)[:, 1].min())
        for i, j in self.pairs:
            result[self.pair_slot[(i, j)]] = self._box_gap(i, j)[0]
        return result

    def _make_frame(self, interaction: np.ndarray) -> StateFrame:
        orientation = np.zeros((self.n, 4), dtype=float)
        orientation[:, 1] = np.sin(0.5 * self.angle)
        orientation[:, 3] = np.cos(0.5 * self.angle)
        angular_velocity = np.zeros((self.n, 3), dtype=float)
        angular_velocity[:, 1] = self.omega
        external = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            if self._active(i, self.time):
                external[i, 2] = self.mass[i] * self.gravity
        frame = StateFrame(
            time=float(self.time), position=self.position.copy(),
            velocity=self.velocity.copy(), orientation=orientation,
            angular_velocity=angular_velocity, mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={"contact_signed_distance": self._signed_distances(),
                    "external_force": external,
                    "interaction_force": interaction.copy()},
            attributes={"release_time": self.release_time.copy(),
                        "is_impact_block": np.array([0.0] * 10 + [1.0] * 3)},
            scalars={"contact_length_scale": 0.03,
                     "friction_coefficient": self.mu,
                     "restitution": self.restitution},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        elapsed = max(0.0, target - self.time)
        impulses = np.zeros((self.contact_count, 3), dtype=float)
        while self.time < target - 1.0e-12:
            self._step(min(0.002, target - self.time), impulses)
        self.time = target
        interaction = impulses / elapsed if elapsed > 1.0e-14 else impulses
        self.last_interaction = interaction
        return self._make_frame(interaction)

    def diagnostics(self) -> dict[str, object]:
        return {"method": "planar_sat_sequential_impulses",
                "substeps": int(self.steps),
                "max_penetration": float(self.max_penetration),
                "finite": bool(np.isfinite(self.position).all() and
                               np.isfinite(self.velocity).all())}
