"""Benchmark solver for circular domino chain reaction.

Implements rigid body dynamics with contact handling (floor and domino-domino).
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternions [x, y, z, w] to 3x3 rotation matrices. Shape: (N, 4) -> (N, 3, 3)"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)

    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)

    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate orientation quaternions [x, y, z, w] with angular velocities [wx, wy, wz]."""
    omega_norm = np.linalg.norm(omega, axis=1, keepdims=True)
    angle = omega_norm * dt
    half_angle = 0.5 * angle

    axis = np.zeros_like(omega)
    mask = (omega_norm[:, 0] > 1e-12)
    axis[mask] = omega[mask] / omega_norm[mask]

    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    dq = np.zeros_like(q)
    dq[:, :3] = axis * sin_half
    dq[:, 3:] = cos_half

    # Quaternion multiplication: q_new = dq * q
    x1, y1, z1, w1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    x2, y2, z2, w2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    q_new = np.zeros_like(q)
    q_new[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_new[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_new[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_new[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Normalize
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-15
    return q_new


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.num_bodies = 0
        self.pos = None
        self.vel = None
        self.quat = None
        self.omega = None
        self.mass = None
        self.inv_mass = None
        self.I_body = None
        self.I_inv_body = None
        self.gravity = None
        self.half_extents = None
        self.friction = 0.55
        self.restitution = 0.12
        self.contact_length_scale = 0.02
        self.current_time = 0.0

        # Diagnostics and tracking
        self.last_contact_signed_dist = np.array([0.0], dtype=np.float64)
        self.last_interaction_force = np.zeros((0, 3), dtype=np.float64)
        self.last_external_force = np.zeros((0, 3), dtype=np.float64)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init_frame = scene.initial_state()
        self.num_bodies = len(init_frame.position)
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.quat = np.array(init_frame.orientation, dtype=np.float64, copy=True)
        self.omega = np.array(init_frame.angular_velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.inv_mass = 1.0 / self.mass

        # Dimensions: [0.07, 0.2, 0.55]
        hx, hy, hz = 0.07, 0.2, 0.55
        self.half_extents = np.array([hx, hy, hz], dtype=np.float64)

        # Moment of inertia for solid box of half_extents (hx, hy, hz) and mass m:
        # Ixx = m/12 * ((2hy)^2 + (2hz)^2) = m/3 * (hy^2 + hz^2)
        # Iyy = m/3 * (hx^2 + hz^2)
        # Izz = m/3 * (hx^2 + hy^2)
        I_diag = np.zeros((self.num_bodies, 3), dtype=np.float64)
        for i in range(self.num_bodies):
            m = self.mass[i]
            I_diag[i, 0] = (m / 3.0) * (hy**2 + hz**2)
            I_diag[i, 1] = (m / 3.0) * (hx**2 + hz**2)
            I_diag[i, 2] = (m / 3.0) * (hx**2 + hy**2)

        self.I_body = I_diag
        self.I_inv_body = 1.0 / I_diag

        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.friction = float(scene.config.get("contact", {}).get("friction", 0.55))
        self.restitution = float(scene.config.get("contact", {}).get("restitution", 0.12))
        self.contact_length_scale = float(scene.config.get("contact", {}).get("length_scale", 0.02))
        self.current_time = float(init_frame.time)

        self.last_external_force = self.mass[:, None] * self.gravity[None, :]
        self.last_contact_signed_dist = np.array([0.0], dtype=np.float64)
        self.last_interaction_force = np.zeros((1, 3), dtype=np.float64)

    def _get_corners(self, i: int, R: np.ndarray) -> np.ndarray:
        """Get 8 world-space corners of box i."""
        hx, hy, hz = self.half_extents
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
        return self.pos[i] + local_corners @ R[i].T

    def _physics_substep(self, dt: float) -> tuple[list[float], list[np.ndarray]]:
        """Perform one numerical physics sub-step using rigid body impulse/penalty formulation."""
        N = self.num_bodies
        R = quat_to_rot_matrix(self.quat)

        # Calculate world-space inverse inertia tensors: I_inv_world = R * I_inv_body * R^T
        I_inv_world = np.zeros((N, 3, 3), dtype=np.float64)
        for i in range(N):
            I_inv_world[i] = R[i] @ np.diag(self.I_inv_body[i]) @ R[i].T

        # 1. Apply gravity to linear velocity
        self.vel += self.gravity * dt

        # Damping to stabilize simulation
        self.vel *= (1.0 - 0.005 * dt)
        self.omega *= (1.0 - 0.01 * dt)

        contact_distances = []
        interaction_forces = []

        # 2. Collision resolution: Floor contact (Floor at z = 0)
        # Test 8 corners of each domino against floor plane z = 0
        floor_normal = np.array([0.0, 0.0, 1.0])

        # Solver iterations for contact stability
        num_solver_iters = 4
        sub_dt = dt / num_solver_iters

        for _ in range(num_solver_iters):
            # Recalculate R and corners each iteration if needed, but per-substep is fast enough
            for i in range(N):
                corners = self._get_corners(i, R)
                for c in corners:
                    dist = c[2]  # distance to z=0
                    contact_distances.append(dist)

                    if dist < 0.0:
                        # Penetration with floor
                        r = c - self.pos[i]
                        # Velocity at contact point
                        vp = self.vel[i] + np.cross(self.omega[i], r)
                        vn = np.dot(vp, floor_normal)

                        # Target normal impulse to stop/resolve penetration and bounce
                        # Baumgarte stabilization
                        beta = 0.2
                        bias = beta * (-dist) / sub_dt

                        # Effective mass along normal
                        r_cross_n = np.cross(r, floor_normal)
                        K_norm = self.inv_mass[i] + np.dot(r_cross_n, I_inv_world[i] @ r_cross_n)

                        if K_norm > 1e-9:
                            delta_vn = -vn + bias
                            if delta_vn > 0:
                                jn = delta_vn / K_norm
                                impulse_n = jn * floor_normal

                                # Tangential friction
                                vt = vp - vn * floor_normal
                                vt_norm = np.linalg.norm(vt)
                                if vt_norm > 1e-6:
                                    t_dir = vt / vt_norm
                                    r_cross_t = np.cross(r, t_dir)
                                    K_tang = self.inv_mass[i] + np.dot(r_cross_t, I_inv_world[i] @ r_cross_t)
                                    jt = min(vt_norm / K_tang, self.friction * jn)
                                    impulse_t = -jt * t_dir
                                else:
                                    impulse_t = np.zeros(3)

                                total_impulse = impulse_n + impulse_t
                                self.vel[i] += total_impulse * self.inv_mass[i]
                                self.omega[i] += I_inv_world[i] @ np.cross(r, total_impulse)

                                interaction_forces.append(total_impulse / sub_dt)

            # 3. Collision resolution: Domino-to-Domino contact
            # Dominoes are arranged along the circle: test pair (i, (i+1)%N) and (i, (i-1)%N)
            for i in range(N):
                # Check domino i against neighboring dominoes
                for neighbor_offset in [1, -1]:
                    next_i = (i + neighbor_offset) % N
                    if next_i <= i and neighbor_offset == -1:
                        # Avoid double counting pair (i, j)
                        continue

                    R_j = R[next_i]
                    pos_j = self.pos[next_i]
                    half_j = self.half_extents

                    corners_i = self._get_corners(i, R)
                    for c in corners_i:
                        # Transform corner c to local space of j
                        local_c = R_j.T @ (c - pos_j)
                        # Signed distance along 3 local axes: d_k = |local_c_k| - half_j_k
                        d_axes = np.abs(local_c) - half_j
                        max_d = np.max(d_axes)
                        contact_distances.append(max_d)

                        if max_d < 0.0:
                            # Corner penetrates box j! Find separation axis (axis of least penetration)
                            axis_idx = np.argmax(d_axes)
                            sign = 1.0 if local_c[axis_idx] > 0 else -1.0

                            # Normal pointing outwards from j in world space
                            local_n = np.zeros(3)
                            local_n[axis_idx] = sign
                            norm_world = R_j @ local_n

                            # Contact point on surface of j
                            penetration = -max_d
                            r_i = c - self.pos[i]
                            r_j = c - self.pos[next_i]

                            v_i = self.vel[i] + np.cross(self.omega[i], r_i)
                            v_j = self.vel[next_i] + np.cross(self.omega[next_i], r_j)
                            v_rel = v_i - v_j  # relative velocity of i w.r.t j

                            # We want impulse on i in direction norm_world (pushing i out of j)
                            vn = np.dot(v_rel, norm_world)

                            beta = 0.25
                            bias = beta * penetration / sub_dt

                            # Effective mass
                            r_i_cross_n = np.cross(r_i, norm_world)
                            r_j_cross_n = np.cross(r_j, norm_world)
                            K = (self.inv_mass[i] + self.inv_mass[next_i] +
                                 np.dot(r_i_cross_n, I_inv_world[i] @ r_i_cross_n) +
                                 np.dot(r_j_cross_n, I_inv_world[next_i] @ r_j_cross_n))

                            if K > 1e-9:
                                delta_v = -vn + bias
                                if delta_v > 0:
                                    jn = delta_v / K
                                    impulse = jn * norm_world

                                    self.vel[i] += impulse * self.inv_mass[i]
                                    self.omega[i] += I_inv_world[i] @ np.cross(r_i, impulse)

                                    self.vel[next_i] -= impulse * self.inv_mass[next_i]
                                    self.omega[next_i] -= I_inv_world[next_i] @ np.cross(r_j, impulse)

                                    interaction_forces.append(impulse / sub_dt)

                    # Also test corners of next_i against box i for full two-way corner-face detection
                    R_i = R[i]
                    pos_i = self.pos[i]
                    half_i = self.half_extents

                    corners_next = self._get_corners(next_i, R)
                    for c in corners_next:
                        local_c = R_i.T @ (c - pos_i)
                        d_axes = np.abs(local_c) - half_i
                        max_d = np.max(d_axes)
                        contact_distances.append(max_d)

                        if max_d < 0.0:
                            axis_idx = np.argmax(d_axes)
                            sign = 1.0 if local_c[axis_idx] > 0 else -1.0

                            local_n = np.zeros(3)
                            local_n[axis_idx] = sign
                            norm_world = R_i @ local_n

                            penetration = -max_d
                            r_j = c - self.pos[next_i]
                            r_i = c - self.pos[i]

                            v_j = self.vel[next_i] + np.cross(self.omega[next_i], r_j)
                            v_i = self.vel[i] + np.cross(self.omega[i], r_i)
                            v_rel = v_j - v_i

                            vn = np.dot(v_rel, norm_world)
                            beta = 0.25
                            bias = beta * penetration / sub_dt

                            r_i_cross_n = np.cross(r_i, norm_world)
                            r_j_cross_n = np.cross(r_j, norm_world)
                            K = (self.inv_mass[i] + self.inv_mass[next_i] +
                                 np.dot(r_i_cross_n, I_inv_world[i] @ r_i_cross_n) +
                                 np.dot(r_j_cross_n, I_inv_world[next_i] @ r_j_cross_n))

                            if K > 1e-9:
                                delta_v = -vn + bias
                                if delta_v > 0:
                                    jn = delta_v / K
                                    impulse = jn * norm_world

                                    self.vel[next_i] += impulse * self.inv_mass[next_i]
                                    self.omega[next_i] += I_inv_world[next_i] @ np.cross(r_j, impulse)

                                    self.vel[i] -= impulse * self.inv_mass[i]
                                    self.omega[i] -= I_inv_world[i] @ np.cross(r_i, impulse)

                                    interaction_forces.append(impulse / sub_dt)

        # 4. Integrate positions and orientations
        self.pos += self.vel * dt
        self.quat = quat_integrate(self.quat, self.omega, dt)

        return contact_distances, interaction_forces

    def advance_to(self, target_time: float) -> StateFrame:
        """Step physics up to target_time and return the current StateFrame."""
        dt_sub = 0.001  # 1ms time-step for high physical accuracy and collision fidelity

        all_contact_distances = []
        all_interaction_forces = []

        while self.current_time + 1e-9 < target_time:
            dt = min(dt_sub, target_time - self.current_time)
            c_dists, i_forces = self._physics_substep(dt)
            if c_dists:
                all_contact_distances.extend(c_dists)
            if i_forces:
                all_interaction_forces.extend(i_forces)
            self.current_time += dt

        self.current_time = target_time

        # Process contact distances and interaction forces for output
        if all_contact_distances:
            cd_arr = np.array(all_contact_distances, dtype=np.float64)
            # Filter to relevant active/near contacts (e.g. <= 0.1m)
            near_mask = cd_arr <= 0.05
            if np.any(near_mask):
                self.last_contact_signed_dist = cd_arr[near_mask]
            else:
                self.last_contact_signed_dist = np.array([float(np.min(cd_arr))], dtype=np.float64)
        else:
            self.last_contact_signed_dist = np.array([0.0], dtype=np.float64)

        if all_interaction_forces:
            self.last_interaction_force = np.array(all_interaction_forces, dtype=np.float64)
        else:
            self.last_interaction_force = np.zeros((1, 3), dtype=np.float64)

        self.last_external_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(self.current_time),
            position=np.array(self.pos, dtype=np.float64, copy=True),
            velocity=np.array(self.vel, dtype=np.float64, copy=True),
            orientation=np.array(self.quat, dtype=np.float64, copy=True),
            angular_velocity=np.array(self.omega, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.arange(self.num_bodies, dtype=np.int32),
            fields={
                "contact_signed_distance": np.array(self.last_contact_signed_dist, dtype=np.float64),
                "external_force": np.array(self.last_external_force, dtype=np.float64),
                "interaction_force": np.array(self.last_interaction_force, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale)
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_bodies": self.num_bodies,
            "current_time": self.current_time,
        }

