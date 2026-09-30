"""Deterministic rigid-body solver for the frictional mechanism jamming scene.

This solver advances an articulated chain of L-shaped rigid links using a
penalty/contact model and explicit time stepping.  No precomputed animation is
used; the only inputs are the frozen scene configuration and geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class BenchmarkSolver:
    """Physics solver implementing the Solver and StateFrame contracts."""

    # ------------------------------------------------------------------
    # Scene-level initialization
    # ------------------------------------------------------------------
    def initialize(self, scene: Any, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

        # Pull frozen parameters.
        config = scene.config
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])
        self.gravity = np.asarray(config["gravity"], dtype=float)
        contact_cfg = config["contact"]
        self.mu = float(contact_cfg["friction"])
        self.restitution = float(contact_cfg["restitution"])
        self.contact_length_scale = float(contact_cfg["length_scale"])

        # Bodies.
        bodies = config["bodies"]
        self.n_bodies = len(bodies)
        self.mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        self.body_id = np.arange(self.n_bodies, dtype=np.int32)

        # Initial state from the scene.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.orientation = np.asarray(frame0.orientation, dtype=float).copy()
        if self.orientation is None or self.orientation.shape != (self.n_bodies, 4):
            self.orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n_bodies, 1))
        self.angular_velocity = np.asarray(frame0.angular_velocity, dtype=float).copy()
        if self.angular_velocity is None or self.angular_velocity.shape != (self.n_bodies, 3):
            self.angular_velocity = np.zeros((self.n_bodies, 3), dtype=float)

        self.release_time = np.asarray(
            frame0.attributes.get("release_time", [0.0] * self.n_bodies), dtype=float
        )

        # Load the mesh and build a two-box approximation of the L-shaped link.
        self._build_link_shape()

        # Rigid-body inertial properties from the same two-box shape.
        self._compute_inertia()

        # Static boundary geometry (box half extents and centers).
        self.boundaries: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.boundaries.append(("floor", np.array([3.2, 2.2, 0.05]), np.array([0.0, 0.0, -0.05])))
        self.boundaries.append(("guide_lower", np.array([2.25, 0.28, 0.08]), np.array([0.0, 0.0, 0.12])))
        self.boundaries.append(("guide_upper", np.array([2.25, 0.28, 0.08]), np.array([0.0, 0.0, 1.02])))

        # Predefine interaction pair list (body-body + body-boundary).
        self.pairs: list[tuple[int, int, str]] = []  # (body_a, body_b_or_boundary, kind)
        # body-body pairs
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                self.pairs.append((i, j, "bb"))
        # body-boundary pairs (body index, boundary index in self.boundaries, kind)
        for i in range(self.n_bodies):
            for bidx in range(len(self.boundaries)):
                self.pairs.append((i, bidx, "bw"))
        self.n_pairs = len(self.pairs)

        # Time-stepping control.
        self.time = 0.0
        self.target_time = 0.0
        # Use 10 physics substeps per output sample (dt = 2 ms).
        self.substeps = 10
        self.dt = self.sample_dt / self.substeps

        # Contact/joint stiffness tuned for stability.
        self.contact_stiffness = 5e4
        self.contact_damping = 200.0
        self.joint_stiffness = 2e5
        self.joint_damping = 50.0

        # Driver parameters: push the rightmost link toward -x.
        self.driver_body = self.n_bodies - 1
        self.driver_force_max = 12.0
        self.driver_ramp = 2.0
        self.driver_direction = np.array([-1.0, 0.0, 0.0])

        # Storage for per-pair contact data used when exporting frames.
        self.last_signed_distance = np.full(self.n_pairs, 1.0, dtype=float)
        self.last_contact_force = np.zeros((self.n_pairs, 3), dtype=float)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _build_link_shape(self) -> None:
        """Build a two-box collision shape from the link mesh file."""
        mesh_path = Path(__file__).resolve().parent.parent / "assets" / "jam_link.obj"
        if not mesh_path.is_file():
            # Fallback to known dimensions if the asset is missing.
            self.box_half_extents = np.array(
                [[0.24, 0.16, 0.08], [0.08, 0.16, 0.15]], dtype=float
            )
            self.box_centers_local = np.array(
                [[0.0, 0.0, -0.08], [0.16, 0.0, 0.15]], dtype=float
            )
        else:
            vertices = self._load_obj(str(mesh_path))
            # Partition into base (z <= 0) and arm (z > 0) boxes.
            base = vertices[vertices[:, 2] <= 0.0]
            arm = vertices[vertices[:, 2] >= 0.0]
            base_min, base_max = base.min(axis=0), base.max(axis=0)
            arm_min, arm_max = arm.min(axis=0), arm.max(axis=0)
            self.box_half_extents = np.array(
                [
                    (base_max - base_min) / 2.0,
                    (arm_max - arm_min) / 2.0,
                ],
                dtype=float,
            )
            self.box_centers_local = np.array(
                [
                    (base_min + base_max) / 2.0,
                    (arm_min + arm_max) / 2.0,
                ],
                dtype=float,
            )

        self.n_boxes = self.box_half_extents.shape[0]
        # Sample points for collision: corners of each box (16 per link).
        self.sample_pattern = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

    def _load_obj(self, path: str) -> np.ndarray:
        vertices: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(vertices, dtype=float)

    def _compute_inertia(self) -> None:
        """Compute total mass, center of mass, and inertia tensor from the two boxes."""
        total_mass = 0.0
        com = np.zeros(3)
        inertia = np.zeros((3, 3))
        for he, cent_local in zip(self.box_half_extents, self.box_centers_local):
            # Volume proportional to box size; use unit density and rescale later.
            volume = 8.0 * he[0] * he[1] * he[2]
            mass_box = volume
            com += mass_box * cent_local
            total_mass += mass_box
        com /= total_mass

        for he, cent_local in zip(self.box_half_extents, self.box_centers_local):
            volume = 8.0 * he[0] * he[1] * he[2]
            mass_box = volume
            # Parallel-axis contribution.
            r = cent_local - com
            inertia += mass_box * (
                (r @ r) * np.eye(3) - np.outer(r, r)
                + np.diag([(he[1] ** 2 + he[2] ** 2) / 3.0,
                           (he[0] ** 2 + he[2] ** 2) / 3.0,
                           (he[0] ** 2 + he[1] ** 2) / 3.0])
            )

        # Rescale inertia so total mass equals the configured mass per body.
        scale = self.mass[0] / total_mass
        inertia *= scale
        self.link_com_local = com.copy()
        self.inertia_tensor = inertia
        self.inertia_inv = np.linalg.inv(inertia)
        self.inertia_inv_body = np.tile(self.inertia_inv, (self.n_bodies, 1, 1))

        # Internally simulate about the centre of mass; the mesh origin is offset
        # from the COM, so the stored position is shifted by the COM offset.
        self.position -= self.link_com_local

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> Any:
        from worldsolver.contracts import StateFrame

        target = float(time)
        while self.time < target - 1e-12:
            self._step()

        self.target_time = target
        return self._make_frame(target)

    def _step(self) -> None:
        dt = self.dt
        t0 = self.time

        # Compute external (non-contact) forces/torques.
        ext_forces, ext_torques = self._external_forces_torques(t0)

        # Compute contact and joint forces/torques.
        contact_forces, contact_torques, pair_signed, pair_forces = self._contact_forces_torques()
        joint_forces, joint_torques = self._joint_forces_torques()

        total_force = ext_forces + contact_forces + joint_forces
        total_torque = ext_torques + contact_torques + joint_torques

        # Semi-implicit Euler integration for linear velocity and position.
        inv_mass = 1.0 / self.mass[:, None]
        self.velocity += dt * total_force * inv_mass
        self.position += dt * self.velocity

        # Angular integration using quaternion derivative.
        # omega in world; update quaternion and angular velocity.
        for i in range(self.n_bodies):
            I_inv = self.inertia_inv_body[i]
            self.angular_velocity[i] += dt * (I_inv @ total_torque[i])
            self.orientation[i] = self._integrate_quaternion(
                self.orientation[i], self.angular_velocity[i], dt
            )

        # Enforce non-penetration with a position projection on highly penetrated points.
        self._position_projection()

        self.time += dt

        # Store last exported pair data for frame construction.
        self.last_signed_distance = pair_signed
        self.last_contact_force = pair_forces

    # ------------------------------------------------------------------
    # Force/torque models
    # ------------------------------------------------------------------
    def _external_forces_torques(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_bodies, 3), dtype=float)
        torques = np.zeros((self.n_bodies, 3), dtype=float)

        # Gravity.
        forces += self.mass[:, None] * self.gravity

        # Driver on the driven link.
        ramp = min(1.0, t / self.driver_ramp)
        force_mag = self.driver_force_max * ramp
        forces[self.driver_body] += force_mag * self.driver_direction

        return forces, torques

    def _contact_forces_torques(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_bodies, 3), dtype=float)
        torques = np.zeros((self.n_bodies, 3), dtype=float)
        pair_signed = np.full(self.n_pairs, 1.0, dtype=float)
        pair_forces = np.zeros((self.n_pairs, 3), dtype=float)

        pair_index = 0
        for a, b, kind in self.pairs:
            if kind == "bb":
                sd, f, r = self._body_body_contact(a, b)
            else:
                sd, f, r = self._body_wall_contact(a, b)
            pair_signed[pair_index] = sd
            pair_forces[pair_index] = f
            pair_index += 1
            if np.linalg.norm(f) > 0.0:
                r_a = r - self.position[a]
                forces[a] += f
                torques[a] += np.cross(r_a, f)
                if kind == "bb":
                    r_b = r - self.position[b]
                    forces[b] -= f
                    torques[b] += np.cross(r_b, -f)

        return forces, torques, pair_signed, pair_forces

    def _joint_forces_torques(self) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_bodies, 3), dtype=float)
        torques = np.zeros((self.n_bodies, 3), dtype=float)

        # Hinge-like joint between consecutive links.
        for i in range(self.n_bodies - 1):
            j = i + 1
            # Attach near the overlap region: alternate side of the overlap.
            if i % 2 == 0:
                local_i = np.array([0.20, 0.0, 0.225])
                local_j = np.array([-0.20, 0.0, -0.125])
            else:
                local_i = np.array([-0.16, 0.0, 0.225])
                local_j = np.array([0.16, 0.0, -0.125])

            Ri = self._quat_to_matrix(self.orientation[i])
            Rj = self._quat_to_matrix(self.orientation[j])
            pi = self.position[i] + Ri @ (local_i - self.link_com_local)
            pj = self.position[j] + Rj @ (local_j - self.link_com_local)

            rel = pi - pj
            spring = -self.joint_stiffness * rel
            damp = -self.joint_damping * (self.velocity[i] - self.velocity[j])
            f = spring + damp

            forces[i] += f
            forces[j] -= f
            torques[i] += np.cross(Ri @ (local_i - self.link_com_local), f)
            torques[j] -= np.cross(Rj @ (local_j - self.link_com_local), f)

        return forces, torques

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------
    def _sample_points_world(self, body: int) -> np.ndarray:
        """Return world-space sample points (corners of each collision box)."""
        R = self._quat_to_matrix(self.orientation[body])
        samples: list[np.ndarray] = []
        for b in range(self.n_boxes):
            center = self.box_centers_local[b]
            he = self.box_half_extents[b]
            for sign in self.sample_pattern:
                p_local = center + he * sign
                p_world = self.position[body] + R @ (p_local - self.link_com_local)
                samples.append(p_world)
        return np.asarray(samples, dtype=float)

    def _body_wall_contact(self, body: int, bidx: int) -> tuple[float, np.ndarray, np.ndarray]:
        boundary = self.boundaries[bidx]
        he, center = boundary[1], boundary[2]
        points = self._sample_points_world(body)
        min_sd = 1e9
        total_f = np.zeros(3)
        contact_point = self.position[body].copy()
        R = self._quat_to_matrix(self.orientation[body])

        for p in points:
            sd = self._point_box_signed_distance(p, he, center)
            if sd < min_sd:
                min_sd = sd
            if sd < 0.0:
                # Approximate normal: vector from the boundary-corrected point to p.
                closest = self._closest_point_on_box(p, he, center)
                n = p - closest
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    n = np.array([0.0, 0.0, 1.0])
                else:
                    n = n / n_norm
                v = self.velocity[body] + np.cross(self.angular_velocity[body], p - self.position[body])
                vn = v @ n
                # Penalty + damping normal force (clamped repulsive).
                fn_mag = max(0.0, -self.contact_stiffness * sd - self.contact_damping * vn)
                fn_vec = fn_mag * n

                # Friction.
                v_tan = v - vn * n
                v_tan_mag = np.linalg.norm(v_tan)
                if v_tan_mag > 1e-4:
                    ft = -self.mu * fn_mag * v_tan / v_tan_mag
                else:
                    # Static friction: try to cancel tangential velocity within Coulomb limit.
                    ft_mag = min(self.mu * fn_mag, max(v_tan_mag, 0.0))
                    ft = -ft_mag * v_tan / (v_tan_mag + 1e-12)

                f = fn_vec + ft
                total_f += f
                contact_point = p.copy()

        # Return a single representative signed distance for the pair.
        return min_sd, total_f, contact_point

    def _body_body_contact(self, a: int, b: int) -> tuple[float, np.ndarray, np.ndarray]:
        Rb = self._quat_to_matrix(self.orientation[b])
        points_a = self._sample_points_world(a)
        min_sd = 1e9
        total_f = np.zeros(3)
        contact_point = self.position[a].copy()

        for p in points_a:
            sd = self._point_body_signed_distance(p, b)
            if sd < min_sd:
                min_sd = sd
            if sd < 0.0:
                # Normal is from body a toward body b (point toward closest surface).
                n = self._body_normal_at_point(p, b)
                v_a = self.velocity[a] + np.cross(self.angular_velocity[a], p - self.position[a])
                v_b = self.velocity[b] + np.cross(self.angular_velocity[b], p - self.position[b])
                v_rel = v_a - v_b
                vn = v_rel @ n
                fn_mag = max(0.0, -self.contact_stiffness * sd - self.contact_damping * vn)
                fn_vec = fn_mag * n

                v_tan = v_rel - vn * n
                v_tan_mag = np.linalg.norm(v_tan)
                if v_tan_mag > 1e-4:
                    ft = -self.mu * fn_mag * v_tan / v_tan_mag
                else:
                    ft_mag = min(self.mu * fn_mag, max(v_tan_mag, 0.0))
                    ft = -ft_mag * v_tan / (v_tan_mag + 1e-12)

                f = fn_vec + ft
                total_f += f
                contact_point = p.copy()

        return min_sd, total_f, contact_point

    def _position_projection(self) -> None:
        """Project deeply penetrating samples apart to avoid blow-up."""
        for i in range(self.n_bodies):
            points = self._sample_points_world(i)
            for bidx, (_, he, center) in enumerate(self.boundaries):
                max_pen = 0.0
                avg_n = np.zeros(3)
                for p in points:
                    sd = self._point_box_signed_distance(p, he, center)
                    if sd < max_pen:
                        max_pen = sd
                if max_pen < -1e-4:
                    # Move body away along the dominant vertical direction for the guides.
                    if bidx == 2:
                        self.position[i, 2] += 0.5 * abs(max_pen)
                    elif bidx == 1:
                        self.position[i, 2] -= 0.5 * abs(max_pen)

    # ------------------------------------------------------------------
    # Signed-distance / geometry helpers
    # ------------------------------------------------------------------
    def _point_box_signed_distance(self, p: np.ndarray, he: np.ndarray, center: np.ndarray) -> float:
        q = p - center
        d_out = np.maximum(np.abs(q) - he, 0.0)
        inside = np.all(np.abs(q) <= he)
        if inside:
            # Negative: closest face distance.
            return -np.min(he - np.abs(q))
        else:
            return np.linalg.norm(d_out)

    def _closest_point_on_box(self, p: np.ndarray, he: np.ndarray, center: np.ndarray) -> np.ndarray:
        q = p - center
        clamped = np.clip(q, -he, he)
        return center + clamped

    def _point_body_signed_distance(self, p: np.ndarray, body: int) -> float:
        R = self._quat_to_matrix(self.orientation[body])
        p_local = R.T @ (p - self.position[body]) + self.link_com_local
        sds = []
        for b in range(self.n_boxes):
            sds.append(self._point_box_signed_distance(p_local, self.box_half_extents[b], self.box_centers_local[b]))
        return min(sds)

    def _body_normal_at_point(self, p: np.ndarray, body: int) -> np.ndarray:
        R = self._quat_to_matrix(self.orientation[body])
        p_local = R.T @ (p - self.position[body]) + self.link_com_local
        best_sd = -1e9
        best_n_local = np.array([0.0, 0.0, 1.0])
        for b in range(self.n_boxes):
            box_center = self.box_centers_local[b]
            he = self.box_half_extents[b]
            q = p_local - box_center
            # Closest point on this box in local coordinates.
            closest_local = box_center + np.clip(q, -he, he)
            n = q - closest_local
            if np.linalg.norm(n) < 1e-12:
                # Point inside, pick the shallowest face normal.
                face_dists = he - np.abs(q)
                axis = int(np.argmin(face_dists))
                n = np.zeros(3)
                n[axis] = 1.0 if q[axis] >= 0 else -1.0
            else:
                n = n / np.linalg.norm(n)
            sd = self._point_box_signed_distance(p_local, he, box_center)
            if sd > best_sd:
                best_sd = sd
                best_n_local = n
        n_world = R @ best_n_local
        return n_world / (np.linalg.norm(n_world) + 1e-12)

    def _quat_to_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return np.eye(3)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _integrate_quaternion(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        # q(t+dt) = normalize(q + 0.5 * dt * [omega]_q * q)
        wx, wy, wz = omega
        omega_q = np.array([wx, wy, wz, 0.0])
        # Quaternion multiplication omega_q * q in xyzw.
        q_new = q + 0.5 * dt * self._quat_mul(omega_q, q)
        q_new /= np.linalg.norm(q_new)
        return q_new

    def _quat_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> Any:
        from worldsolver.contracts import StateFrame

        t = float(time)
        ext_force = np.zeros((self.n_bodies, 3), dtype=float)
        ext_force += self.mass[:, None] * self.gravity
        ramp = min(1.0, t / self.driver_ramp) if t >= 0.0 else 0.0
        ext_force[self.driver_body] += self.driver_force_max * ramp * self.driver_direction

        frame = StateFrame(
            time=t,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": self.release_time.copy()},
            scalars={"contact_length_scale": self.contact_length_scale},
            fields={
                "external_force": ext_force,
                "interaction_force": self.last_contact_force.copy(),
                "contact_signed_distance": self.last_signed_distance.copy(),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "bodies": self.n_bodies,
            "substeps": self.substeps,
            "dt": self.dt,
            "final_time": self.time,
        }
