"""Deterministic rigid-body solver for the chess-piece settling scene.

The rendered triangle meshes are represented for contact by small unions of spheres.
This retains the important wide-base/narrow-stem/head geometry while making the
sequential impulse solve robust enough for a many-body pile without external physics
packages.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Return the local-to-world rotation matrix for an xyzw quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _base_profile() -> list[tuple[float, float, float, float]]:
    # Five overlapping spheres approximate the broad, nearly flat 0.6 m base.
    return [
        (0.0, 0.0, -0.160, 0.180),
        (0.205, 0.0, -0.245, 0.095),
        (-0.205, 0.0, -0.245, 0.095),
        (0.0, 0.205, -0.245, 0.095),
        (0.0, -0.205, -0.245, 0.095),
        (0.0, 0.0, 0.035, 0.115),
    ]


def _piece_profile(kind: str) -> tuple[np.ndarray, np.ndarray, float]:
    p = _base_profile()
    if kind == "pawn":
        p += [(0.0, 0.0, 0.205, 0.145), (0.025, 0.0, 0.425, 0.085)]
        height = 0.85
    elif kind == "rook":
        p += [
            (0.0, 0.0, 0.205, 0.145),
            (0.18, 0.0, 0.345, 0.115),
            (-0.18, 0.0, 0.345, 0.115),
            (0.0, 0.18, 0.345, 0.115),
            (0.0, -0.18, 0.345, 0.115),
        ]
        height = 0.79
    elif kind == "knight":
        p += [(0.0, 0.0, 0.185, 0.145), (0.075, 0.0, 0.330, 0.205)]
        height = 0.82
    elif kind == "bishop":
        p += [(0.0, 0.0, 0.225, 0.135), (0.018, 0.0, 0.405, 0.145), (0.035, 0.0, 0.610, 0.160)]
        height = 1.11
    elif kind == "queen":
        p += [(0.0, 0.0, 0.205, 0.145), (0.0, 0.0, 0.370, 0.160)]
        height = 0.865
    else:  # king
        p += [(0.0, 0.0, 0.205, 0.145), (-0.018, 0.0, 0.385, 0.120)]
        height = 0.84
    values = np.asarray(p, dtype=float)
    return values[:, :3], values[:, 3], height


class BenchmarkSolver:
    """Semi-implicit rigid dynamics with projected contact impulses."""

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.orientation = np.asarray(initial.orientation, dtype=float).copy()
        self.omega = np.asarray(initial.angular_velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.release_time = np.asarray(initial.attributes.get("release_time", np.zeros(len(self.mass))), dtype=float).copy()
        self.n = len(self.mass)
        self.inv_mass = 1.0 / self.mass
        self.gravity = np.asarray(scene.config["gravity"], dtype=float)
        contact = scene.config["contact"]
        self.friction = float(contact["friction"])
        self.restitution = float(contact["restitution"])
        self.length_scale = float(contact["length_scale"])
        self.time = 0.0
        self.max_dt = 0.0025

        # Geometry order follows the frozen scene_geometry body bindings.
        kinds = [
            "pawn", "rook", "knight", "bishop", "queen", "king", "pawn", "rook",
            "knight", "bishop", "queen", "king", "pawn", "rook", "bishop", "king",
        ]
        self.local_centers: list[np.ndarray] = []
        self.radii: list[np.ndarray] = []
        inertia = np.empty((self.n, 3), dtype=float)
        self.bound_radius = np.empty(self.n, dtype=float)
        for i, kind in enumerate(kinds[: self.n]):
            centers, radii, height = _piece_profile(kind)
            self.local_centers.append(centers)
            self.radii.append(radii)
            # A cylinder-like inertia is a stable approximation to the solid mesh.
            r = 0.27
            inertia[i, 0] = inertia[i, 1] = self.mass[i] * (3.0 * r * r + height * height) / 12.0
            inertia[i, 2] = 0.5 * self.mass[i] * r * r
            self.bound_radius[i] = np.max(np.linalg.norm(centers, axis=1) + radii)
        self.inv_inertia_body = 1.0 / inertia

        # One slot per floor/body pair and per unordered body/body pair.
        self.pair_keys: list[tuple[int, int]] = [(i, -1) for i in range(self.n)]
        self.pair_keys += [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.pair_index = {key: k for k, key in enumerate(self.pair_keys)}
        self.last_pair_impulse = np.zeros((len(self.pair_keys), 3), dtype=float)
        self.last_interval = 0.0
        self.step_count = 0

    def _geometry(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        rotations: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        inv_inertia = np.empty((self.n, 3, 3), dtype=float)
        for i in range(self.n):
            rot = _rotation_matrix(self.orientation[i])
            rotations.append(rot)
            centers.append(self.position[i] + self.local_centers[i] @ rot.T)
            inv_inertia[i] = (rot * self.inv_inertia_body[i]) @ rot.T
        return centers, rotations, inv_inertia

    def _detect_contacts(self, centers: list[np.ndarray]) -> list[tuple]:
        contacts: list[tuple] = []
        margin = 0.006
        up = np.array([0.0, 0.0, 1.0])

        # Several base samples may contact the plane, which gives a resting piece
        # proper rotational support instead of a single point constraint.
        for i in range(self.n):
            gaps = centers[i][:, 2] - self.radii[i]
            active = np.flatnonzero(gaps < margin)
            if len(active) > 4:
                active = active[np.argsort(gaps[active])[:4]]
            for component in active:
                point = centers[i][component] - up * self.radii[i][component]
                contacts.append((i, -1, up, point, float(gaps[component]), (i, -1)))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                cutoff = self.bound_radius[i] + self.bound_radius[j] + margin
                delta_body = self.position[j] - self.position[i]
                if np.dot(delta_body, delta_body) > cutoff * cutoff:
                    continue
                delta = centers[j][None, :, :] - centers[i][:, None, :]
                distance = np.linalg.norm(delta, axis=2)
                gap = distance - self.radii[i][:, None] - self.radii[j][None, :]
                flat = int(np.argmin(gap))
                a, b = np.unravel_index(flat, gap.shape)
                g = float(gap[a, b])
                if g >= margin:
                    continue
                d = float(distance[a, b])
                if d > 1.0e-10:
                    normal = delta[a, b] / d
                else:
                    normal = np.array([1.0, 0.0, 0.0])
                pa = centers[i][a] + normal * self.radii[i][a]
                pb = centers[j][b] - normal * self.radii[j][b]
                contacts.append((i, j, normal, 0.5 * (pa + pb), g, (i, j)))
        return contacts

    def _point_velocity(self, i: int, arm: np.ndarray) -> np.ndarray:
        return self.velocity[i] + np.cross(self.omega[i], arm)

    def _effective_mass(self, i: int, arm: np.ndarray, direction: np.ndarray, inv_i: np.ndarray) -> float:
        rotational = np.cross(inv_i @ np.cross(arm, direction), arm)
        return self.inv_mass[i] + float(np.dot(direction, rotational))

    def _apply_impulse(self, i: int, arm: np.ndarray, impulse: np.ndarray, inv_i: np.ndarray) -> None:
        self.velocity[i] += self.inv_mass[i] * impulse
        self.omega[i] += inv_i @ np.cross(arm, impulse)

    def _solve_contacts(self, contacts: list[tuple], inv_inertia: np.ndarray, dt: float) -> np.ndarray:
        count = len(contacts)
        normal_lambda = np.zeros(count, dtype=float)
        tangent_lambda = np.zeros((count, 3), dtype=float)
        bounce_target = np.zeros(count, dtype=float)

        for c, (i, j, normal, point, gap, _key) in enumerate(contacts):
            ri = point - self.position[i]
            if j < 0:
                closing = -float(np.dot(self._point_velocity(i, ri), normal))
            else:
                rj = point - self.position[j]
                closing = -float(np.dot(self._point_velocity(j, rj) - self._point_velocity(i, ri), normal))
            if closing > 0.35:
                bounce_target[c] = self.restitution * closing

        for _iteration in range(7):
            for c, (i, j, normal, point, gap, _key) in enumerate(contacts):
                ri = point - self.position[i]
                penetration_speed = min(1.2, 0.18 * max(-gap - 0.001, 0.0) / dt)
                desired = max(bounce_target[c], penetration_speed)
                if j < 0:
                    vp = self._point_velocity(i, ri)
                    vn = float(np.dot(vp, normal))
                    denom = self._effective_mass(i, ri, normal, inv_inertia[i])
                    change = (desired - vn) / max(denom, 1.0e-9)
                    old = normal_lambda[c]
                    normal_lambda[c] = max(0.0, old + change)
                    dl = normal_lambda[c] - old
                    self._apply_impulse(i, ri, dl * normal, inv_inertia[i])

                    vp = self._point_velocity(i, ri)
                    tangent_v = vp - normal * np.dot(vp, normal)
                    speed = float(np.linalg.norm(tangent_v))
                    if speed > 1.0e-9:
                        direction = tangent_v / speed
                        denom_t = self._effective_mass(i, ri, direction, inv_inertia[i])
                        trial = tangent_lambda[c] - (speed / max(denom_t, 1.0e-9)) * direction
                        limit = self.friction * normal_lambda[c]
                        norm_trial = float(np.linalg.norm(trial))
                        if norm_trial > limit and norm_trial > 0.0:
                            trial *= limit / norm_trial
                        impulse = trial - tangent_lambda[c]
                        tangent_lambda[c] = trial
                        self._apply_impulse(i, ri, impulse, inv_inertia[i])
                else:
                    rj = point - self.position[j]
                    relative = self._point_velocity(j, rj) - self._point_velocity(i, ri)
                    vn = float(np.dot(relative, normal))
                    denom = self._effective_mass(i, ri, normal, inv_inertia[i])
                    denom += self._effective_mass(j, rj, normal, inv_inertia[j])
                    change = (desired - vn) / max(denom, 1.0e-9)
                    old = normal_lambda[c]
                    normal_lambda[c] = max(0.0, old + change)
                    dl = normal_lambda[c] - old
                    impulse = dl * normal
                    self._apply_impulse(i, ri, -impulse, inv_inertia[i])
                    self._apply_impulse(j, rj, impulse, inv_inertia[j])

                    relative = self._point_velocity(j, rj) - self._point_velocity(i, ri)
                    tangent_v = relative - normal * np.dot(relative, normal)
                    speed = float(np.linalg.norm(tangent_v))
                    if speed > 1.0e-9:
                        direction = tangent_v / speed
                        denom_t = self._effective_mass(i, ri, direction, inv_inertia[i])
                        denom_t += self._effective_mass(j, rj, direction, inv_inertia[j])
                        trial = tangent_lambda[c] - (speed / max(denom_t, 1.0e-9)) * direction
                        limit = self.friction * normal_lambda[c]
                        norm_trial = float(np.linalg.norm(trial))
                        if norm_trial > limit and norm_trial > 0.0:
                            trial *= limit / norm_trial
                        impulse = trial - tangent_lambda[c]
                        tangent_lambda[c] = trial
                        self._apply_impulse(i, ri, -impulse, inv_inertia[i])
                        self._apply_impulse(j, rj, impulse, inv_inertia[j])

        pair_impulse = np.zeros((len(self.pair_keys), 3), dtype=float)
        for c, (i, j, normal, _point, _gap, key) in enumerate(contacts):
            # Stored vector is the impulse on the body for floor contacts and on
            # body j for body pairs. Every export includes its opposite partner.
            pair_impulse[self.pair_index[key]] += normal_lambda[c] * normal + tangent_lambda[c]

        # Split positional projection by inverse mass. This removes accumulated
        # overlap without injecting kinetic energy.
        for i, j, normal, _point, gap, _key in contacts:
            depth = max(-gap - 0.001, 0.0)
            if depth <= 0.0:
                continue
            correction = min(0.025, 0.22 * depth)
            if j < 0:
                self.position[i] += correction * normal
            else:
                weight = self.inv_mass[i] + self.inv_mass[j]
                self.position[i] -= correction * (self.inv_mass[i] / weight) * normal
                self.position[j] += correction * (self.inv_mass[j] / weight) * normal
        return pair_impulse

    def _integrate_orientation(self, dt: float) -> None:
        for i in range(self.n):
            xyz = self.orientation[i, :3]
            qw = self.orientation[i, 3]
            angular = self.omega[i]
            derivative_xyz = 0.5 * (qw * angular + np.cross(angular, xyz))
            derivative_w = -0.5 * float(np.dot(angular, xyz))
            self.orientation[i, :3] += dt * derivative_xyz
            self.orientation[i, 3] += dt * derivative_w
            norm = float(np.linalg.norm(self.orientation[i]))
            if norm > 1.0e-12:
                self.orientation[i] /= norm
            else:
                self.orientation[i] = (0.0, 0.0, 0.0, 1.0)

    def _substep(self, dt: float) -> np.ndarray:
        # Gravity plus mild linear/angular air damping dissipates free-flight energy;
        # most pile dissipation still comes from restitution and Coulomb friction.
        linear_decay = math.exp(-0.10 * dt)
        angular_decay = math.exp(-0.16 * dt)
        self.velocity = (self.velocity + self.gravity * dt) * linear_decay
        self.omega *= angular_decay
        self.position += self.velocity * dt
        self._integrate_orientation(dt)

        centers, _rotations, inv_inertia = self._geometry()
        contacts = self._detect_contacts(centers)
        impulse = self._solve_contacts(contacts, inv_inertia, dt)

        # Guard against pathological energy growth from a very deep initial overlap.
        speed = np.linalg.norm(self.velocity, axis=1)
        fast = speed > 12.0
        if np.any(fast):
            self.velocity[fast] *= (12.0 / speed[fast])[:, None]
        spin = np.linalg.norm(self.omega, axis=1)
        fast_spin = spin > 35.0
        if np.any(fast_spin):
            self.omega[fast_spin] *= (35.0 / spin[fast_spin])[:, None]
        self.step_count += 1
        return impulse

    def _signed_distances(self) -> np.ndarray:
        centers, _rotations, _inv = self._geometry()
        distances = []
        for i in range(self.n):
            distances.append(float(np.min(centers[i][:, 2] - self.radii[i])))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = centers[j][None, :, :] - centers[i][:, None, :]
                d = np.linalg.norm(delta, axis=2)
                gap = d - self.radii[i][:, None] - self.radii[j][None, :]
                distances.append(float(np.min(gap)))
        return np.asarray(distances, dtype=float)

    def _frame(self, time: float) -> StateFrame:
        if self.last_interval > 0.0:
            pair_force = self.last_pair_impulse / self.last_interval
            active = np.linalg.norm(pair_force, axis=1) > 1.0e-10
            one_sided = pair_force[active]
            interaction = np.empty((2 * len(one_sided), 3), dtype=float)
            interaction[0::2] = one_sided
            interaction[1::2] = -one_sided
        else:
            interaction = np.empty((0, 3), dtype=float)
        external = self.mass[:, None] * (self.gravity - 0.10 * self.velocity)
        frame = StateFrame(
            float(time),
            self.position.copy(),
            self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": self.release_time.copy()},
            fields={
                "external_force": external,
                "interaction_force": interaction,
                "contact_signed_distance": self._signed_distances(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to time must be finite and monotonic")
        target = max(target, self.time)
        accumulated = np.zeros((len(self.pair_keys), 3), dtype=float)
        start = self.time
        while self.time < target - 1.0e-12:
            dt = min(self.max_dt, target - self.time)
            accumulated += self._substep(dt)
            self.time += dt
        self.time = target
        self.last_interval = target - start
        self.last_pair_impulse = accumulated
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "compound-sphere sequential impulses",
            "substeps": int(self.step_count),
            "body_count": int(self.n),
            "finite": bool(
                np.isfinite(self.position).all()
                and np.isfinite(self.velocity).all()
                and np.isfinite(self.orientation).all()
            ),
        }
