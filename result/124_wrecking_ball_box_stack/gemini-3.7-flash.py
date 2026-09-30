"""Rigid body physics solver for wrecking ball box stack scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion by angular velocity."""
    w_norm = np.linalg.norm(omega)
    if w_norm < 1e-12:
        return q
    axis = omega / w_norm
    angle = w_norm * dt
    # delta quaternion [x, y, z, w]
    dq = np.array([axis[0] * np.sin(angle * 0.5),
                   axis[1] * np.sin(angle * 0.5),
                   axis[2] * np.sin(angle * 0.5),
                   np.cos(angle * 0.5)], dtype=np.float64)
    # quaternion multiplication q_new = dq * q
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q
    q_new = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float64)
    norm = np.linalg.norm(q_new)
    if norm > 1e-12:
        q_new /= norm
    return q_new


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.current_time = 0.0

        bodies = self.config["bodies"]
        self.num_bodies = len(bodies)

        self.positions = np.asarray([b["position"] for b in bodies], dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.orientations = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies], dtype=np.float64)
        self.angular_velocities = np.zeros_like(self.positions)
        self.masses = np.asarray([b["mass"] for b in bodies], dtype=np.float64)
        self.body_ids = np.arange(self.num_bodies, dtype=np.int32)

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.contact_length_scale = float(self.config.get("contact", {}).get("length_scale", 0.02))
        self.friction_coeff = float(self.config.get("contact", {}).get("friction", 0.55))

        # Inertia tensors and geometry specifications
        self.inv_masses = 1.0 / self.masses
        self.inv_inertias = np.zeros((self.num_bodies, 3, 3), dtype=np.float64)
        self.half_extents = np.zeros((self.num_bodies, 3), dtype=np.float64)
        self.capsule_radii = np.zeros(self.num_bodies, dtype=np.float64)
        self.capsule_heights = np.zeros(self.num_bodies, dtype=np.float64)
        self.is_box = np.zeros(self.num_bodies, dtype=bool)
        self.is_chain = np.zeros(self.num_bodies, dtype=bool)
        self.is_ball = np.zeros(self.num_bodies, dtype=bool)

        for i, b in enumerate(bodies):
            bid = b["id"]
            m = self.masses[i]
            if "box" in bid:
                self.is_box[i] = True
                # Box half extents from scene_geometry: [0.17, 0.38, 0.17]
                hx, hy, hz = 0.17, 0.38, 0.17
                self.half_extents[i] = [hx, hy, hz]
                Ix = (1.0 / 12.0) * m * ((2 * hy)**2 + (2 * hz)**2)
                Iy = (1.0 / 12.0) * m * ((2 * hx)**2 + (2 * hz)**2)
                Iz = (1.0 / 12.0) * m * ((2 * hx)**2 + (2 * hy)**2)
                self.inv_inertias[i] = np.diag([1.0 / Ix, 1.0 / Iy, 1.0 / Iz])
            elif "wrecking_ball" in bid:
                self.is_ball[i] = True
                # Sphere approx radius ~ 0.28m
                r = 0.28
                self.capsule_radii[i] = r
                I = (2.0 / 5.0) * m * r * r
                self.inv_inertias[i] = np.diag([1.0 / I, 1.0 / I, 1.0 / I])
            elif "chain" in bid:
                self.is_chain[i] = True
                r = 0.07
                h = 0.25
                self.capsule_radii[i] = r
                self.capsule_heights[i] = h
                # Cylinder approx inertia along z
                Iz = 0.5 * m * r * r
                Ix = (1.0 / 12.0) * m * (3 * r * r + h * h)
                Iy = Ix
                self.inv_inertias[i] = np.diag([1.0 / Ix, 1.0 / Iy, 1.0 / Iz])

        # Chain joints setup
        # Support beam at [-2.2, 0, 4.0]
        # Chain 5: 3.55 (top link)
        # Chain 4: 3.23
        # Chain 3: 2.91
        # Chain 2: 2.59
        # Chain 1: 2.27
        # Chain 0: 1.95
        # Ball: 1.35
        # Joint connections:
        # Beam -> Chain 5: anchor at [ -2.2, 0, 3.71 ], local offset on chain_5: [0, 0, 0.16]
        # Chain 5 -> Chain 4: anchor at z = 3.39
        # Chain 4 -> Chain 3: anchor at z = 3.07
        # Chain 3 -> Chain 2: anchor at z = 2.75
        # Chain 2 -> Chain 1: anchor at z = 2.43
        # Chain 1 -> Chain 0: anchor at z = 2.11
        # Chain 0 -> Ball: anchor at z = 1.79
        self.joints = []
        # Find indices
        body_id_map = {b["id"]: i for i, b in enumerate(bodies)}
        self.support_anchor = np.array([-2.2, 0.0, 3.71], dtype=np.float64)
        c5_idx = body_id_map["chain_5"]
        self.joints.append(("fixed", -1, c5_idx, self.support_anchor, np.array([0.0, 0.0, 0.16])))

        chain_pairs = [("chain_5", "chain_4"), ("chain_4", "chain_3"), ("chain_3", "chain_2"),
                       ("chain_2", "chain_1"), ("chain_1", "chain_0")]
        for p1, p2 in chain_pairs:
            idx1 = body_id_map[p1]
            idx2 = body_id_map[p2]
            self.joints.append(("ball", idx1, idx2, np.array([0.0, 0.0, -0.16]), np.array([0.0, 0.0, 0.16])))

        c0_idx = body_id_map["chain_0"]
        ball_idx = body_id_map["wrecking_ball"]
        self.joints.append(("ball", c0_idx, ball_idx, np.array([0.0, 0.0, -0.16]), np.array([0.0, 0.0, 0.28])))

        # Simulation substepping params
        self.substep_dt = 0.001

        # Track contact diagnostics for StateFrame
        self.last_contact_sdfs = np.array([0.0], dtype=np.float64)
        self.last_interaction_forces = np.zeros((1, 3), dtype=np.float64)

    def _get_box_corners(self, i: int) -> np.ndarray:
        R = quat_to_rot_matrix(self.orientations[i])
        hx, hy, hz = self.half_extents[i]
        local_corners = np.array([
            [-hx, -hy, -hz],
            [-hx, -hy,  hz],
            [-hx,  hy, -hz],
            [-hx,  hy,  hz],
            [ hx, -hy, -hz],
            [ hx, -hy,  hz],
            [ hx,  hy, -hz],
            [ hx,  hy,  hz],
        ], dtype=np.float64)
        return self.positions[i] + (local_corners @ R.T)

    def _step_physics(self, dt: float) -> None:
        N = self.num_bodies
        gravity_force = self.masses[:, None] * self.gravity[None, :]

        # Semi-implicit Euler integration for velocity from gravity
        self.velocities += (gravity_force * self.inv_masses[:, None]) * dt

        # Damping
        self.velocities *= np.exp(-0.02 * dt)
        self.angular_velocities *= np.exp(-0.05 * dt)

        # Joint constraints solver (Position-Based / Impulse-Based relaxation)
        # Solve joint constraints multiple iterations
        num_solver_iters = 15
        sdfs = []
        interaction_forces = []

        for _ in range(num_solver_iters):
            # 1. Chain Joints
            for jtype, b1, b2, r1, r2 in self.joints:
                if jtype == "fixed":
                    # b1 is static support, b2 is chain_5
                    R2 = quat_to_rot_matrix(self.orientations[b2])
                    p2_anchor = self.positions[b2] + R2 @ r2
                    delta = self.support_anchor - p2_anchor
                    dist = np.linalg.norm(delta)
                    if dist > 1e-6:
                        # Position correction
                        self.positions[b2] += delta * 0.95
                        # Velocity correction
                        v2_anchor = self.velocities[b2] + np.cross(self.angular_velocities[b2], R2 @ r2)
                        v_rel = -v2_anchor
                        impulse = v_rel * (self.masses[b2] * 0.5)
                        self.velocities[b2] += impulse * self.inv_masses[b2]
                        self.angular_velocities[b2] += self.inv_inertias[b2] @ np.cross(R2 @ r2, impulse)
                elif jtype == "ball":
                    R1 = quat_to_rot_matrix(self.orientations[b1])
                    R2 = quat_to_rot_matrix(self.orientations[b2])
                    p1_anchor = self.positions[b1] + R1 @ r1
                    p2_anchor = self.positions[b2] + R2 @ r2
                    delta = p1_anchor - p2_anchor
                    dist = np.linalg.norm(delta)
                    if dist > 1e-6:
                        w1 = self.inv_masses[b1]
                        w2 = self.inv_masses[b2]
                        corr1 = delta * (w1 / (w1 + w2)) * 0.95
                        corr2 = -delta * (w2 / (w1 + w2)) * 0.95
                        self.positions[b1] -= corr1
                        self.positions[b2] += corr2

                        v1 = self.velocities[b1] + np.cross(self.angular_velocities[b1], R1 @ r1)
                        v2 = self.velocities[b2] + np.cross(self.angular_velocities[b2], R2 @ r2)
                        v_rel = v1 - v2
                        reduced_m = 1.0 / (w1 + w2)
                        imp = v_rel * (reduced_m * 0.5)
                        self.velocities[b1] -= imp * w1
                        self.velocities[b2] += imp * w2

            # 2. Floor contacts (z = 0 is floor plane)
            for i in range(N):
                if self.is_box[i]:
                    corners = self._get_box_corners(i)
                    for corner in corners:
                        z = corner[2]
                        sdf = z
                        if sdf < 0.0:
                            sdfs.append(sdf)
                            depth = -sdf
                            # Normal impulse along +z
                            r_c = corner - self.positions[i]
                            v_pt = self.velocities[i] + np.cross(self.angular_velocities[i], r_c)
                            v_norm = v_pt[2]
                            if v_norm < 0:
                                imp_mag = -(1.0 + 0.12) * v_norm / (self.inv_masses[i] + 0.1)
                                normal_force = imp_mag / dt
                                self.velocities[i, 2] += imp_mag * self.inv_masses[i]

                                # Friction
                                v_tang = np.array([v_pt[0], v_pt[1], 0.0])
                                tang_speed = np.linalg.norm(v_tang)
                                if tang_speed > 1e-6:
                                    fric_imp = min(self.friction_coeff * imp_mag, tang_speed * self.masses[i])
                                    fric_dir = -v_tang / tang_speed
                                    self.velocities[i, :2] += fric_dir[:2] * fric_imp * self.inv_masses[i]
                                    interaction_forces.append(fric_dir * (fric_imp / dt) + np.array([0, 0, normal_force]))
                                else:
                                    interaction_forces.append(np.array([0, 0, normal_force]))

                            # Penetration position correction
                            self.positions[i, 2] += depth * 0.5
                elif self.is_ball[i] or self.is_chain[i]:
                    r = self.capsule_radii[i]
                    sdf = self.positions[i, 2] - r
                    if sdf < 0.0:
                        sdfs.append(sdf)
                        depth = -sdf
                        self.positions[i, 2] += depth * 0.8
                        if self.velocities[i, 2] < 0:
                            self.velocities[i, 2] = -self.velocities[i, 2] * 0.12
                            # Friction
                            self.velocities[i, :2] *= 0.8

            # 3. Wrecking ball against boxes contact
            ball_pos = self.positions[ball_idx]
            ball_r = self.capsule_radii[ball_idx]
            for i in range(N):
                if self.is_box[i]:
                    # Check box-ball contact
                    box_p = self.positions[i]
                    diff = ball_pos - box_p
                    R_box = quat_to_rot_matrix(self.orientations[i])
                    local_diff = R_box.T @ diff
                    hx, hy, hz = self.half_extents[i]
                    closest_local = np.clip(local_diff, -np.array([hx, hy, hz]), np.array([hx, hy, hz]))
                    closest_world = box_p + R_box @ closest_local
                    to_ball = ball_pos - closest_world
                    dist = np.linalg.norm(to_ball)
                    sdf = dist - ball_r
                    if sdf < 0.0:
                        sdfs.append(sdf)
                        depth = -sdf
                        n = to_ball / (dist + 1e-8) if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                        # Normal impulse
                        v_ball = self.velocities[ball_idx]
                        v_box = self.velocities[i]
                        v_rel = v_ball - v_box
                        v_norm = np.dot(v_rel, n)
                        if v_norm < 0:
                            m1 = self.masses[ball_idx]
                            m2 = self.masses[i]
                            imp_mag = -(1.0 + 0.12) * v_norm / (1.0 / m1 + 1.0 / m2)
                            imp = n * imp_mag
                            self.velocities[ball_idx] += imp / m1
                            self.velocities[i] -= imp / m2
                            # Box angular kick
                            r_box_c = closest_world - box_p
                            self.angular_velocities[i] -= self.inv_inertias[i] @ np.cross(r_box_c, imp)
                            interaction_forces.append(imp / dt)

                        # Positional push
                        w1 = self.inv_masses[ball_idx]
                        w2 = self.inv_masses[i]
                        self.positions[ball_idx] += n * depth * (w1 / (w1 + w2)) * 0.8
                        self.positions[i] -= n * depth * (w2 / (w1 + w2)) * 0.8

            # 4. Box-to-box contacts
            for i in range(N):
                if not self.is_box[i]:
                    continue
                for j in range(i + 1, N):
                    if not self.is_box[j]:
                        continue
                    delta_p = self.positions[j] - self.positions[i]
                    # Quick bounding sphere check
                    if np.dot(delta_p, delta_p) > 1.2:
                        continue
                    # Approximate box-box contact using center distance and half extents
                    hx_tot = self.half_extents[i, 0] + self.half_extents[j, 0]
                    hy_tot = self.half_extents[i, 1] + self.half_extents[j, 1]
                    hz_tot = self.half_extents[i, 2] + self.half_extents[j, 2]

                    abs_dx = abs(delta_p[0])
                    abs_dy = abs(delta_p[1])
                    abs_dz = abs(delta_p[2])

                    ox = hx_tot - abs_dx
                    oy = hy_tot - abs_dy
                    oz = hz_tot - abs_dz

                    if ox > 0 and oy > 0 and oz > 0:
                        # Overlap along smallest axis
                        if ox < oy and ox < oz:
                            depth = ox
                            n = np.array([1.0 if delta_p[0] > 0 else -1.0, 0.0, 0.0])
                        elif oy < oz:
                            depth = oy
                            n = np.array([0.0, 1.0 if delta_p[1] > 0 else -1.0, 0.0])
                        else:
                            depth = oz
                            n = np.array([0.0, 0.0, 1.0 if delta_p[2] > 0 else -1.0])

                        sdf = -depth
                        sdfs.append(sdf)

                        # Separate
                        self.positions[i] -= n * depth * 0.45
                        self.positions[j] += n * depth * 0.45

                        v_rel = self.velocities[j] - self.velocities[i]
                        v_norm = np.dot(v_rel, n)
                        if v_norm < 0:
                            m1 = self.masses[i]
                            m2 = self.masses[j]
                            imp_mag = -(1.0 + 0.12) * v_norm / (1.0 / m1 + 1.0 / m2)
                            imp = n * imp_mag
                            self.velocities[i] -= imp / m1
                            self.velocities[j] += imp / m2
                            interaction_forces.append(imp / dt)

        # Update positions and orientations
        self.positions += self.velocities * dt
        for i in range(N):
            self.orientations[i] = quat_integrate(self.orientations[i], self.angular_velocities[i], dt)

        if len(sdfs) > 0:
            self.last_contact_sdfs = np.asarray(sdfs, dtype=np.float64)
        else:
            self.last_contact_sdfs = np.array([self.contact_length_scale], dtype=np.float64)

        if len(interaction_forces) > 0:
            self.last_interaction_forces = np.asarray(interaction_forces, dtype=np.float64)
        else:
            self.last_interaction_forces = np.zeros((1, 3), dtype=np.float64)

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until reaching target_time
        while self.current_time < target_time - 1e-9:
            dt = min(self.substep_dt, target_time - self.current_time)
            self._step_physics(dt)
            self.current_time += dt

        external_forces = self.masses[:, None] * self.gravity[None, :]

        frame = StateFrame(
            float(target_time),
            np.copy(self.positions),
            np.copy(self.velocities),
            orientation=np.copy(self.orientations),
            angular_velocity=np.copy(self.angular_velocities),
            mass=np.copy(self.masses),
            body_id=np.copy(self.body_ids),
            attributes={
                "release_time": np.asarray([b.get("release_time", 0.0) for b in self.config["bodies"]], dtype=np.float64)
            },
            fields={
                "contact_signed_distance": np.copy(self.last_contact_sdfs),
                "external_force": np.copy(external_forces),
                "interaction_force": np.copy(self.last_interaction_forces),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            }
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_bodies": self.num_bodies,
        }

