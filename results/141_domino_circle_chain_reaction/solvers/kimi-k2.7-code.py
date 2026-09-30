"""Solver implementation for the circular domino chain reaction scene.

A rigid-body solver integrating Newton-Euler dynamics with a penalty contact
model and velocity damping.  Uses sub-stepped semi-implicit Euler and reports
the state at the requested sample times.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        # Time integration settings.
        self.dt_sample = float(self.config["sample_dt"])
        # Fixed 1 ms sub-step keeps the explicit penalty contact stable.
        self.dt_sub = 1.0e-3

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        contact_cfg = self.config["contact"]
        self.friction = float(contact_cfg["friction"])
        self.restitution = float(contact_cfg["restitution"])
        self.contact_length_scale = float(contact_cfg["length_scale"])

        # Initial state from scene
        init = scene.initial_state()
        self.n_bodies = len(init.position)
        self.position = np.array(init.position, dtype=float)
        self.velocity = np.array(init.velocity, dtype=float)
        self.orientation = np.array(init.orientation, dtype=float)  # [N,4] xyzw
        self.angular_velocity = np.array(init.angular_velocity, dtype=float)
        self.mass = np.array(init.mass, dtype=float)
        self.body_id = np.array(init.body_id, dtype=np.int32)

        # Geometry constants (all dominoes are identical boxes)
        self.half_extents = np.array([0.07, 0.2, 0.55], dtype=float)
        # Local corner coordinates, reused for SAT contact queries.
        self._corners_local = np.array(
            [
                [-self.half_extents[0], -self.half_extents[1], -self.half_extents[2]],
                [-self.half_extents[0], -self.half_extents[1], +self.half_extents[2]],
                [-self.half_extents[0], +self.half_extents[1], -self.half_extents[2]],
                [-self.half_extents[0], +self.half_extents[1], +self.half_extents[2]],
                [+self.half_extents[0], -self.half_extents[1], -self.half_extents[2]],
                [+self.half_extents[0], -self.half_extents[1], +self.half_extents[2]],
                [+self.half_extents[0], +self.half_extents[1], -self.half_extents[2]],
                [+self.half_extents[0], +self.half_extents[1], +self.half_extents[2]],
            ],
            dtype=float,
        )

        # Inertia tensor for a uniform-density box, in the body frame.
        m = max(self.mass[0], 1e-9)
        self.inertia_diag_local = np.zeros(3, dtype=float)
        self.inertia_diag_local[0] = (1.0 / 12.0) * m * (
            (2.0 * self.half_extents[1]) ** 2 + (2.0 * self.half_extents[2]) ** 2
        )
        self.inertia_diag_local[1] = (1.0 / 12.0) * m * (
            (2.0 * self.half_extents[0]) ** 2 + (2.0 * self.half_extents[2]) ** 2
        )
        self.inertia_diag_local[2] = (1.0 / 12.0) * m * (
            (2.0 * self.half_extents[0]) ** 2 + (2.0 * self.half_extents[1]) ** 2
        )

        # The initial configuration already tilts domino_0; no extra kick is needed.
        self.time = 0.0

        # Precompute penalty stiffness per body (used for floor contacts).
        # Chosen to be stiff but stable with dt = 1 ms for ~0.3 kg bodies.
        self._k_contact = 2.0e5
        self._d_contact = 2.0 * math.sqrt(self._k_contact * 0.3)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "bodies": int(self.n_bodies),
            "time": float(self.time),
            "dt_sub": float(self.dt_sub),
        }

    def _quat_to_matrix(self, q: np.ndarray) -> np.ndarray:
        # scipy expects scalar-last (x, y, z, w), which matches our storage.
        return Rotation.from_quat(q).as_matrix()

    def _integrate_orientation(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        # Integrate quaternion using the exponential map (axis-angle).
        angle = float(np.linalg.norm(omega))
        if angle < 1e-12:
            return q / np.linalg.norm(q)
        half = 0.5 * angle * dt
        s = math.sin(half) / angle
        c = math.cos(half)
        # r = quaternion(cos, sin*omega) as (x, y, z, w)
        r = np.array([s * omega[0], s * omega[1], s * omega[2], c], dtype=float)
        # Hamilton product r * q
        x1, y1, z1, w1 = r
        x2, y2, z2, w2 = q
        qnew = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
        return qnew / np.linalg.norm(qnew)

    def _obb_signed_distance(
        self,
        pos_i: np.ndarray,
        q_i: np.ndarray,
        pos_j: np.ndarray,
        q_j: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Signed distance between two OBBs using the separating-axis theorem.

        Returns:
            signed_distance (positive = separated, negative = penetrating),
            axis (unit normal pointing from body i to body j),
            point_on_i, point_on_j (world-space contact witness points).
        """
        R_i = self._quat_to_matrix(q_i)
        R_j = self._quat_to_matrix(q_j)

        # Candidate separating axes: 6 face normals + 9 edge cross-products.
        axes: list[np.ndarray] = [R_i[:, k] for k in range(3)] + [R_j[:, k] for k in range(3)]
        for a in range(3):
            for b in range(3):
                cp = np.cross(R_i[:, a], R_j[:, b])
                nrm = float(np.linalg.norm(cp))
                if nrm > 1e-8:
                    axes.append(cp / nrm)
        axes_arr = np.stack(axes)
        axes_arr = axes_arr / (np.linalg.norm(axes_arr, axis=1, keepdims=True) + 1e-12)

        # Vertices in world coordinates, relative to box centers.
        verts_i = self._corners_local @ R_i.T
        verts_j = self._corners_local @ R_j.T

        # Project onto each candidate axis.
        proj_i = verts_i @ axes_arr.T  # (8, A)
        proj_j = verts_j @ axes_arr.T
        min_i = proj_i.min(axis=0)
        max_i = proj_i.max(axis=0)
        min_j = proj_j.min(axis=0)
        max_j = proj_j.max(axis=0)

        # Signed distance per axis: s > 0 => separated, s < 0 => penetrating.
        signed_per_axis = np.maximum(min_i - max_j, min_j - max_i)
        idx = int(np.argmin(signed_per_axis))
        sd = float(signed_per_axis[idx])
        axis = axes_arr[idx].copy()

        # Orient axis from body i toward body j.
        if np.dot(pos_j - pos_i, axis) < 0.0:
            axis = -axis

        # Witness points via support mapping.
        p_i = pos_i + R_i @ (np.sign(R_i.T @ axis) * self.half_extents)
        p_j = pos_j + R_j @ (np.sign(R_j.T @ (-axis)) * self.half_extents)
        return sd, axis, p_i, p_j

    def _contact_force_pair(
        self,
        i: int,
        j: int,
        sd: float,
        n: np.ndarray,
        p_i: np.ndarray,
        p_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the force/torque pair on bodies i and j for a single contact.

        Returns f_i, f_j, tau_i, tau_j.
        """
        if sd >= 0.0:
            return (
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
            )

        r_i = p_i - self.position[i]
        r_j = p_j - self.position[j]
        v_i = self.velocity[i] + np.cross(self.angular_velocity[i], r_i)
        v_j = self.velocity[j] + np.cross(self.angular_velocity[j], r_j)
        v_rel = v_i - v_j
        vn = float(np.dot(v_rel, n))

        k = self._k_contact
        d = self._d_contact
        # Penalty normal force with critical damping, capped to avoid numerical explosion.
        m_eff = min(self.mass[i], self.mass[j])
        fn_mag = max(0.0, -k * sd - d * min(0.0, vn))
        fn_mag = min(fn_mag, 1.0e4 * m_eff)
        fn = fn_mag * n

        # Friction / tangential damping.
        vt = v_rel - n * vn
        vt_mag = float(np.linalg.norm(vt))
        if vt_mag > 1e-8:
            ft_dir = -vt / vt_mag
            ft_mag = min(self.friction * fn_mag, self.friction * fn_mag + 0.0)
            ft = ft_dir * ft_mag
        else:
            ft = np.zeros(3, dtype=float)

        f_total = fn + ft
        tau_i = np.cross(r_i, f_total)
        tau_j = np.cross(r_j, -f_total)
        return f_total, -f_total, tau_i, tau_j

    def _compute_contact_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float, np.ndarray, np.ndarray]]]:
        """Compute gravity, contact forces/torques, and record active contacts."""
        ext_force = self.mass[:, None] * self.gravity[None, :]
        contact_force = np.zeros((self.n_bodies, 3), dtype=float)
        contact_torque = np.zeros((self.n_bodies, 3), dtype=float)
        pairs: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []

        floor_normal = np.array([0.0, 0.0, 1.0], dtype=float)

        # Floor contact for each domino.
        for i in range(self.n_bodies):
            # World-space vertices.
            R = self._quat_to_matrix(self.orientation[i])
            verts = self.position[i] + self._corners_local @ R.T
            sd = float(np.min(verts[:, 2]))
            if sd < 0.0:
                # Closest point to the floor (average of the lowest vertices).
                min_z = verts[:, 2].min()
                mask = verts[:, 2] < min_z + 1e-3
                point = verts[mask].mean(axis=0)
                point[2] = max(point[2], 0.0)
                n = floor_normal
                r = point - self.position[i]
                v_point = self.velocity[i] + np.cross(self.angular_velocity[i], r)
                vn = float(np.dot(v_point, n))

                k = self._k_contact
                d = self._d_contact
                fn_mag = max(0.0, -k * sd - d * min(0.0, vn))
                fn_mag = min(fn_mag, 1.0e4 * self.mass[i])
                fn = fn_mag * n

                vt = v_point - n * vn
                vt_mag = float(np.linalg.norm(vt))
                if vt_mag > 1e-8:
                    ft_dir = -vt / vt_mag
                    ft_mag = min(self.friction * fn_mag, self.friction * fn_mag + 0.0)
                    ft = ft_dir * ft_mag
                else:
                    ft = np.zeros(3, dtype=float)

                f_total = fn + ft
                contact_force[i] += f_total
                contact_torque[i] += np.cross(r, f_total)

        # Body-body contacts.
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                sd, axis, p_i, p_j = self._obb_signed_distance(
                    self.position[i], self.orientation[i],
                    self.position[j], self.orientation[j],
                )
                if sd < 0.0:
                    pairs.append((i, j, sd, p_i, p_j))
                    f_i, f_j, tau_i, tau_j = self._contact_force_pair(i, j, sd, axis, p_i, p_j)
                    contact_force[i] += f_i
                    contact_force[j] += f_j
                    contact_torque[i] += tau_i
                    contact_torque[j] += tau_j

        return ext_force, contact_force, contact_torque, pairs

    def _step(self, dt: float) -> None:
        # Symplectic Euler: evaluate forces at current state, update velocity, then pose.
        ext_force, contact_force, contact_torque, _ = self._compute_contact_forces()
        total_force = ext_force + contact_force

        # Linear update.
        acc = total_force / self.mass[:, None]
        self.velocity += acc * dt
        self.position += self.velocity * dt

        # Angular update.
        for i in range(self.n_bodies):
            Rmat = self._quat_to_matrix(self.orientation[i])
            I_world = Rmat @ np.diag(self.inertia_diag_local) @ Rmat.T
            omega = self.angular_velocity[i]
            alpha = np.linalg.solve(I_world, contact_torque[i] - np.cross(omega, I_world @ omega))
            self.angular_velocity[i] += alpha * dt
            self.orientation[i] = self._integrate_orientation(
                self.orientation[i], self.angular_velocity[i], dt
            )

        # Light velocity damping to help dominoes settle after the wave.
        damping = math.exp(-0.5 * dt)
        self.velocity *= damping
        self.angular_velocity *= damping

    def _build_frame(self, time: float, ext_force: np.ndarray, pairs: list[tuple[int, int, float, np.ndarray, np.ndarray]]) -> StateFrame:
        # Signed contact distances: floor contacts + body-body pairs.
        sd_list: list[float] = []
        for i in range(self.n_bodies):
            R = self._quat_to_matrix(self.orientation[i])
            verts = self.position[i] + self._corners_local @ R.T
            sd = float(np.min(verts[:, 2]))
            if sd < 0.0:
                sd_list.append(sd)
        for _, _, sd, _, _ in pairs:
            sd_list.append(float(sd))
        if len(sd_list) == 0:
            sd_list = [1.0]
        contact_signed_distance = np.array(sd_list, dtype=float)

        # Interaction forces: one vector per active pair, directed from i to j.
        if len(pairs) == 0:
            interaction_force = np.zeros((1, 3), dtype=float)
        else:
            interaction_force = np.zeros((len(pairs), 3), dtype=float)
            for idx, (i, j, sd, p_i, p_j) in enumerate(pairs):
                _, f_j, _, _ = self._contact_force_pair(i, j, sd, (p_j - p_i) / (np.linalg.norm(p_j - p_i) + 1e-12), p_i, p_j)
                interaction_force[idx] = -f_j  # force that j applies on i

        return StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": contact_signed_distance,
                "external_force": ext_force.copy(),
                "interaction_force": interaction_force,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            ext_force, _, _, pairs = self._compute_contact_forces()
            return self._build_frame(time, ext_force, pairs)

        while self.time < time:
            dt = min(self.dt_sub, time - self.time)
            self._step(dt)
            self.time += dt

        ext_force, _, _, pairs = self._compute_contact_forces()
        return self._build_frame(time, ext_force, pairs)
