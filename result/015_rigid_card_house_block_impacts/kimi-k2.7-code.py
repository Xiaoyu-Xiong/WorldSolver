"""Rigid body solver for the card-house impact scene.

Implements a penalty-based 3-D rigid body dynamics solver with Coulomb
friction and restitution-like damping.  The solver steps the physics with
a small fixed sub-step, detects box-box and box-floor contacts, and emits
all required StateFrame fields.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        self.gravity = np.array(self.config["gravity"], dtype=float)
        self.mu = float(self.config["contact_parameters"]["friction_coefficient"])
        self.restitution = float(self.config["contact_parameters"]["restitution"])

        frame = scene.initial_state()
        self.time: float = 0.0
        self.num_bodies: int = len(frame.position)

        self.pos = frame.position.copy()
        self.vel = frame.velocity.copy()
        self.quat = frame.orientation.copy()
        self.omega = frame.angular_velocity.copy()
        self.mass = frame.mass.copy()
        self.body_id = frame.body_id.copy()

        self.release_time = frame.attributes["release_time"].copy()
        self.is_impact_block = frame.attributes["is_impact_block"].copy().astype(bool)

        # Half-extents for each body (cards + impact blocks).
        self.half_extents = np.zeros((self.num_bodies, 3))
        self.half_extents[:10] = [0.45, 0.18, 0.015]
        self.half_extents[10] = [0.16, 0.16, 0.16]
        self.half_extents[11] = [0.16, 0.16, 0.16]
        self.half_extents[12] = [0.19, 0.19, 0.19]

        # Local inertia tensors (diagonal in the body frame) for solid cuboids.
        self.I_local = np.zeros((self.num_bodies, 3, 3))
        for i in range(self.num_bodies):
            a, b, c = self.half_extents[i]
            m = self.mass[i]
            ixx = m / 3.0 * (b * b + c * c)
            iyy = m / 3.0 * (a * a + c * c)
            izz = m / 3.0 * (a * a + b * b)
            self.I_local[i] = np.diag([ixx, iyy, izz])

        self.I_local_inv = np.linalg.inv(self.I_local)

        # Fixed floor treated as a half-space z >= 0.
        self.floor_normal = np.array([0.0, 0.0, -1.0])  # body -> floor
        self.floor_plane = np.array([0.0, 0.0, 1.0])      # floor -> body

        # Sub-stepping and contact parameters.
        self.max_substep: float = 1e-4
        self.k_penalty: float = 2.0e5
        self.damping_ratio: float = 0.85
        self.lin_damp: float = 0.005
        self.ang_damp: float = 0.02

        self.contact_length_scale: float = 0.03

        # Store initial state to keep unreleased impact blocks frozen.
        self.initial_pos = self.pos.copy()
        self.initial_quat = self.quat.copy()
        self.initial_vel = self.vel.copy()
        self.initial_omega = self.omega.copy()

        # Pre-allocate buffers.
        self._forces = np.zeros((self.num_bodies, 3))
        self._torques = np.zeros((self.num_bodies, 3))

        self._fields_external_force = np.zeros((self.num_bodies, 3))
        self._fields_contact_signed_distance: np.ndarray = np.zeros(0)
        self._fields_interaction_force: np.ndarray = np.zeros((0, 3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("advance_to time must be non-decreasing")

        dt = time - self.time
        if dt > 0.0:
            n_sub = max(1, int(np.ceil(dt / self.max_substep)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                self._step(sub_dt)
            self.time = time

        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "penalty_rigid_body",
            "substep": self.max_substep,
            "contact_stiffness": self.k_penalty,
            "damping_ratio": self.damping_ratio,
        }

    # ------------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # Zero accumulators.
        self._forces.fill(0.0)
        self._torques.fill(0.0)

        # Gravity on released bodies.
        for i in range(self.num_bodies):
            if self._is_released(i):
                self._forces[i] += self.mass[i] * self.gravity

        # Detect contacts and apply penalty forces.
        contacts = self._detect_contacts()
        for contact in contacts:
            self._apply_contact_force(contact, dt)

        # Integrate released bodies.
        for i in range(self.num_bodies):
            if not self._is_released(i):
                # Frozen impact block: reset to initial pose/velocity.
                self.pos[i] = self.initial_pos[i].copy()
                self.quat[i] = self.initial_quat[i].copy()
                self.vel[i] = self.initial_vel[i].copy()
                self.omega[i] = self.initial_omega[i].copy()
                continue

            # Linear.
            self.vel[i] += dt * self._forces[i] / self.mass[i]
            self.vel[i] *= max(0.0, 1.0 - self.lin_damp * dt)
            self.pos[i] += dt * self.vel[i]

            # Angular.
            R = self._quat_to_matrix(self.quat[i])
            I_world = R @ self.I_local[i] @ R.T
            I_inv = np.linalg.inv(I_world)
            L = I_world @ self.omega[i]
            gyro = np.cross(self.omega[i], L)
            alpha = I_inv @ (self._torques[i] - gyro)
            self.omega[i] += dt * alpha
            self.omega[i] *= max(0.0, 1.0 - self.ang_damp * dt)

            # Integrate quaternion.
            self.quat[i] = self._quat_integrate(self.quat[i], self.omega[i], dt)
            self.quat[i] /= np.linalg.norm(self.quat[i])

    # ------------------------------------------------------------------
    # Contact detection and force application
    # ------------------------------------------------------------------
    def _detect_contacts(self) -> list:
        """Return list of contacts. Each contact is a dict."""
        contacts = []
        for i in range(self.num_bodies):
            for j in range(i + 1, self.num_bodies):
                c = self._body_body_contact(i, j)
                if c is not None:
                    contacts.append(c)
        for i in range(self.num_bodies):
            c = self._body_floor_contact(i)
            if c is not None:
                contacts.append(c)
        return contacts

    def _body_body_contact(self, i: int, j: int):
        c1 = self.pos[i]
        c2 = self.pos[j]
        R1 = self._quat_to_matrix(self.quat[i])
        R2 = self._quat_to_matrix(self.quat[j])
        h1 = self.half_extents[i]
        h2 = self.half_extents[j]

        # SAT over 15 candidate axes.
        axes = []
        for a in range(3):
            axes.append(R1[:, a])
            axes.append(R2[:, a])
        for a in range(3):
            for b in range(3):
                axes.append(np.cross(R1[:, a], R2[:, b]))

        t_vec = c2 - c1
        best_pen = float("inf")
        best_axis = None
        best_sign = 1.0

        for axis in axes:
            nlen = np.linalg.norm(axis)
            if nlen < 1e-10:
                continue
            n = axis / nlen
            r1 = sum(h1[k] * abs(np.dot(R1[:, k], n)) for k in range(3))
            r2 = sum(h2[k] * abs(np.dot(R2[:, k], n)) for k in range(3))
            d = np.dot(t_vec, n)
            pen = r1 + r2 - abs(d)
            if pen < -1e-7:
                return None
            if pen < best_pen:
                best_pen = pen
                best_axis = n
                best_sign = np.sign(d) if abs(d) > 1e-10 else 1.0

        if best_axis is None:
            return None

        n = best_sign * best_axis
        best_pen = max(best_pen, 0.0)

        # Try vertex-inclusion for a more accurate contact point.
        point = self._deepest_vertex_point(c1, R1, h1, c2, R2, h2, n)
        if point is None:
            p1 = self._support_point(c1, R1, h1, -n)
            p2 = self._support_point(c2, R2, h2, n)
            point = 0.5 * (p1 + p2)

        return {
            "i": i,
            "j": j,
            "normal": n,
            "depth": best_pen,
            "point": point,
            "type": "body-body",
        }

    def _body_floor_contact(self, i: int):
        R = self._quat_to_matrix(self.quat[i])
        h = self.half_extents[i]
        # Find the lowest world-z vertex of the body.
        min_z = float("inf")
        signs = [(-1.0, 1.0)] * 3
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = np.array([sx * h[0], sy * h[1], sz * h[2]])
                    p = self.pos[i] + R @ local
                    if p[2] < min_z:
                        min_z = p[2]
        if min_z >= -1e-8:
            return None

        depth = -min_z
        # Use the projection of the body centre onto the floor for the
        # contact point so that a vertical floor reaction creates no
        # artificial torque about the centre.
        point = np.array([self.pos[i, 0], self.pos[i, 1], 0.0])
        return {
            "i": i,
            "j": -1,
            "normal": self.floor_normal.copy(),
            "depth": depth,
            "point": point,
            "type": "floor",
        }

    def _apply_contact_force(self, contact: dict, dt: float) -> None:
        i = contact["i"]
        j = contact["j"]
        n = contact["normal"]
        depth = contact["depth"]
        p = contact["point"]

        released_i = self._is_released(i)
        released_j = (j >= 0) and self._is_released(j)

        if (j >= 0) and not (released_i or released_j):
            return
        if j < 0 and not released_i:
            return

        r_i = p - self.pos[i]
        if j >= 0:
            r_j = p - self.pos[j]
        else:
            r_j = np.zeros(3)

        v_i = self.vel[i] + np.cross(self.omega[i], r_i)
        v_j = self.vel[j] + np.cross(self.omega[j], r_j) if j >= 0 else np.zeros(3)
        v_rel = v_j - v_i
        v_n = np.dot(v_rel, n)

        # Effective mass for the normal direction.
        m_i = self.mass[i] if released_i else float("inf")
        m_j = self.mass[j] if (j >= 0 and released_j) else float("inf")
        if np.isinf(m_i) and np.isinf(m_j):
            return
        if np.isinf(m_i):
            m_eff = m_j
        elif np.isinf(m_j):
            m_eff = m_i
        else:
            m_eff = m_i * m_j / (m_i + m_j)

        # Normal spring-damper with restitution-like bounce clipping.
        k = self.k_penalty
        c = 2.0 * self.damping_ratio * np.sqrt(k * m_eff)
        f_n = k * depth - c * v_n
        # Clamp small negative values to avoid attraction; also add a
        # tiny coefficient of restitution by preventing complete
        # rebound suppression for fast closing velocities.
        if v_n < -1e-3:
            f_n += self.restitution * m_eff * abs(v_n) / max(dt, 1e-8)
        f_n = max(0.0, f_n)

        # Tangential friction.
        v_t = v_rel - v_n * n
        v_t_mag = np.linalg.norm(v_t)
        f_t = np.zeros(3)
        if v_t_mag > 1e-7:
            t_dir = v_t / v_t_mag
            # Maximum friction force limited by Coulomb and by the impulse
            # needed to stop the relative tangential motion in one step.
            f_t_mag = min(self.mu * f_n, m_eff * v_t_mag / max(dt, 1e-8))
            f_t = -f_t_mag * t_dir

        # Force on body j (or floor) is f_n * n + f_t.
        force_j = f_n * n + f_t
        if j < 0:
            # Floor reaction: apply only to body i.
            force_i = -force_j
        else:
            force_i = -force_j

        if released_i:
            self._forces[i] += force_i
            self._torques[i] += np.cross(r_i, force_i)
        if j >= 0 and released_j:
            self._forces[j] += force_j
            self._torques[j] += np.cross(r_j, force_j)

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        # Recompute contacts and interaction forces at the sample state.
        contacts = self._detect_contacts()
        n_contacts = len(contacts)
        self._fields_contact_signed_distance = np.empty(n_contacts, dtype=float)
        self._fields_interaction_force = np.empty((n_contacts, 3), dtype=float)

        for idx, contact in enumerate(contacts):
            self._fields_contact_signed_distance[idx] = -contact["depth"]
            # Interaction force on the second body (or floor) is the
            # normal+friction force computed at the current state.
            f = self._interaction_force(contact)
            self._fields_interaction_force[idx] = f

        self._fields_external_force = self.mass[:, None] * self.gravity[None, :]

        return StateFrame(
            time=self.time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "release_time": self.release_time.copy(),
                "is_impact_block": self.is_impact_block.copy().astype(float),
            },
            fields={
                "contact_signed_distance": self._fields_contact_signed_distance.copy(),
                "external_force": self._fields_external_force.copy(),
                "interaction_force": self._fields_interaction_force.copy(),
            },
            scalars={
                "friction_coefficient": self.mu,
                "restitution": self.restitution,
                "contact_length_scale": self.contact_length_scale,
            },
        )

    def _interaction_force(self, contact: dict) -> np.ndarray:
        """Return the contact force vector on the second body in the pair."""
        # Compute the same normal+friction force used during stepping.
        i = contact["i"]
        j = contact["j"]
        n = contact["normal"]
        depth = contact["depth"]
        p = contact["point"]

        released_i = self._is_released(i)
        released_j = (j >= 0) and self._is_released(j)

        if (j >= 0) and not (released_i or released_j):
            return np.zeros(3)
        if j < 0 and not released_i:
            return np.zeros(3)

        r_i = p - self.pos[i]
        if j >= 0:
            r_j = p - self.pos[j]
        else:
            r_j = np.zeros(3)

        v_i = self.vel[i] + np.cross(self.omega[i], r_i)
        v_j = self.vel[j] + np.cross(self.omega[j], r_j) if j >= 0 else np.zeros(3)
        v_rel = v_j - v_i
        v_n = np.dot(v_rel, n)

        m_i = self.mass[i] if released_i else float("inf")
        m_j = self.mass[j] if (j >= 0 and released_j) else float("inf")
        if np.isinf(m_i) and np.isinf(m_j):
            return np.zeros(3)
        if np.isinf(m_i):
            m_eff = m_j
        elif np.isinf(m_j):
            m_eff = m_i
        else:
            m_eff = m_i * m_j / (m_i + m_j)

        dt = self.max_substep
        k = self.k_penalty
        c = 2.0 * self.damping_ratio * np.sqrt(k * m_eff)
        f_n = k * depth - c * v_n
        if v_n < -1e-3:
            f_n += self.restitution * m_eff * abs(v_n) / max(dt, 1e-8)
        f_n = max(0.0, f_n)

        v_t = v_rel - v_n * n
        v_t_mag = np.linalg.norm(v_t)
        f_t = np.zeros(3)
        if v_t_mag > 1e-7:
            t_dir = v_t / v_t_mag
            f_t_mag = min(self.mu * f_n, m_eff * v_t_mag / max(dt, 1e-8))
            f_t = -f_t_mag * t_dir

        return f_n * n + f_t

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_released(self, body_idx: int) -> bool:
        return self.time >= self.release_time[body_idx]

    def _support_point(self, c, R, h, n):
        local_n = R.T @ n
        signs = np.sign(local_n)
        signs[signs == 0.0] = 1.0
        return c + R @ (h * signs)

    def _deepest_vertex_point(self, c1, R1, h1, c2, R2, h2, n):
        """If a vertex of one body is inside the other, return the deepest one."""
        best_depth = -1.0
        best_point = None
        best_normal = None

        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = np.array([sx * h1[0], sy * h1[1], sz * h1[2]])
                    p = c1 + R1 @ local
                    if self._point_inside_obb(p, c2, R2, h2):
                        depth, normal = self._point_penetration(p, c2, R2, h2)
                        if depth > best_depth:
                            best_depth = depth
                            best_point = p
                            best_normal = normal

                    local = np.array([sx * h2[0], sy * h2[1], sz * h2[2]])
                    p = c2 + R2 @ local
                    if self._point_inside_obb(p, c1, R1, h1):
                        depth, normal = self._point_penetration(p, c1, R1, h1)
                        if depth > best_depth:
                            best_depth = depth
                            best_point = p
                            # normal points from body1 to body2 = outward of body1
                            best_normal = -normal

        if best_point is None:
            return None
        return best_point

    def _point_inside_obb(self, p, c, R, h):
        s = R.T @ (p - c)
        return np.all(np.abs(s) <= h + 1e-8)

    def _point_penetration(self, p, c, R, h):
        s = R.T @ (p - c)
        # Distance to each pair of faces.
        depths = np.minimum(h - s, h + s)
        k = int(np.argmin(depths))
        face_depth = depths[k]
        # Outward normal of the closest face.
        if abs(h[k] - s[k]) < abs(h[k] + s[k]):
            normal = R[:, k]
        else:
            normal = -R[:, k]
        return face_depth, normal

    def _quat_to_matrix(self, q) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])

    def _quat_integrate(self, q, omega, dt):
        # dq = 0.5 * q * [0, omega]
        w = np.array([omega[0], omega[1], omega[2], 0.0])
        return self._quat_mul(w, q) * 0.5 * dt + q

    def _quat_mul(self, a, b):
        return np.array([
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ])
