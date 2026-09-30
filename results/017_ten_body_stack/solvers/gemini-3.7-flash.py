"""Solver Agent implementation for 017_ten_body_stack.

Simulates a stack of 10 rigid boxes settling under gravity using projected Gauss-Seidel /
impulse-based contact dynamics with friction and restitution.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion with angular velocity."""
    wx, wy, wz = omega
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    # dq/dt = 0.5 * omega * q
    dq = 0.5 * np.array([
        qw * wx + qy * wz - qz * wy,
        qw * wy + qz * wx - qx * wz,
        qw * wz + qx * wy - qy * wx,
        -qx * wx - qy * wy - qz * wz
    ])
    q_new = q + dq * dt
    norm = np.linalg.norm(q_new)
    if norm > 1e-12:
        q_new /= norm
    else:
        q_new = np.array([0.0, 0.0, 0.0, 1.0])
    return q_new


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.time = 0.0
        self.n_bodies = 10
        self.sub_dt = 0.0005  # 2 kHz substepping for rigid stack stability
        self.gravity = np.array([0.0, 0.0, -9.81])

        # State arrays
        self.pos = np.zeros((10, 3))
        self.vel = np.zeros((10, 3))
        self.quat = np.zeros((10, 4))
        self.omega = np.zeros((10, 3))
        self.mass = np.ones(10)
        self.inv_mass = np.ones(10)
        self.body_id = np.arange(1, 11, dtype=int)

        # Half extents
        self.hx = 0.45
        self.hy = 0.45
        self.hz = 0.10
        self.half_extents = np.array([self.hx, self.hy, self.hz])

        # Inertia tensor of a box: mass * (1/12) * diag(dy^2+dz^2, dx^2+dz^2, dx^2+dy^2)
        dx, dy, dz = 2.0 * self.hx, 2.0 * self.hy, 2.0 * self.hz
        self.I_body = np.diag([
            (1.0 / 12.0) * (dy * dy + dz * dz),
            (1.0 / 12.0) * (dx * dx + dz * dz),
            (1.0 / 12.0) * (dx * dx + dy * dy)
        ])
        self.inv_I_body = np.linalg.inv(self.I_body)

        # Precompute 8 corner vertices in local body coordinates
        self.local_corners = np.array([
            [-self.hx, -self.hy, -self.hz],
            [+self.hx, -self.hy, -self.hz],
            [-self.hx, +self.hy, -self.hz],
            [+self.hx, +self.hy, -self.hz],
            [-self.hx, -self.hy, +self.hz],
            [+self.hx, -self.hy, +self.hz],
            [-self.hx, +self.hy, +self.hz],
            [+self.hx, +self.hy, +self.hz],
        ])

        # Ground height is at z = 0 (ground box top surface: translation z=-0.05, half_ext z=0.05)
        self.ground_z = 0.0

        # Last recorded contacts and forces for output
        self.last_contact_gaps = []
        self.last_interaction_forces = []
        self.contact_length_scale = 0.20  # Box height / contact scale (2 * hz = 0.20 m)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.time = 0.0

        if "gravity" in scene.config:
            self.gravity = np.array(scene.config["gravity"], dtype=float)

        init_frame = scene.initial_state()
        self.pos = np.copy(init_frame.position)
        self.vel = np.copy(init_frame.velocity)
        self.quat = np.copy(init_frame.orientation)
        self.omega = np.copy(init_frame.angular_velocity)
        self.mass = np.copy(init_frame.mass)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.copy(init_frame.body_id)
        self.n_bodies = len(self.mass)

        self.last_contact_gaps = []
        self.last_interaction_forces = []

    def _get_world_corners(self, i: int) -> np.ndarray:
        R = quat_to_rot_matrix(self.quat[i])
        return self.pos[i] + (R @ self.local_corners.T).T

    def _step(self, dt: float) -> tuple[list[float], list[np.ndarray]]:
        # Semi-implicit Euler integration + Contact resolution
        n = self.n_bodies

        # 1. Apply gravity and linear/angular damping
        damping_lin = 0.9995
        damping_ang = 0.9990
        for i in range(n):
            self.vel[i] += self.gravity * dt
            self.vel[i] *= damping_lin
            self.omega[i] *= damping_ang

        # Collect contact constraints
        # Each contact constraint: (body_a, body_b, r_a, r_b, normal, depth)
        # body_b = -1 means ground
        contacts = []

        # Ground contacts for body 0 (and any other body near ground)
        for i in range(n):
            corners_i = self._get_world_corners(i)
            for c in corners_i:
                # Check ground penetration
                gap = c[2] - self.ground_z
                if gap < 0.02:  # contact threshold
                    r_a = c - self.pos[i]
                    contacts.append((i, -1, r_a, np.zeros(3), np.array([0.0, 0.0, 1.0]), gap))

        # Body-to-body contacts
        # For stacked boxes, check bottom corners of upper body against top of lower body,
        # and top corners of lower body against bottom of upper body.
        for i in range(n - 1):
            j = i + 1  # body j is directly above body i
            R_i = quat_to_rot_matrix(self.quat[i])
            R_j = quat_to_rot_matrix(self.quat[j])

            # Check bottom corners of body j penetrating top face of body i
            corners_j = self.pos[j] + (R_j @ self.local_corners[:4].T).T  # 4 bottom corners
            for c in corners_j:
                # Transform c into local frame of body i
                c_rel_i = c - self.pos[i]
                c_loc_i = R_i.T @ c_rel_i

                # Check if x and y are within body i bounds
                if abs(c_loc_i[0]) <= self.hx + 0.02 and abs(c_loc_i[1]) <= self.hy + 0.02:
                    # Penetration depth relative to top face (+hz)
                    gap = c_loc_i[2] - self.hz
                    if gap < 0.02:
                        normal_world = R_i @ np.array([0.0, 0.0, 1.0])
                        r_a = c - self.pos[j]  # upper body
                        r_b = c - self.pos[i]  # lower body
                        # contact normal points from b (lower) to a (upper)
                        contacts.append((j, i, r_a, r_b, normal_world, gap))

            # Check top corners of body i penetrating bottom face of body j
            corners_i = self.pos[i] + (R_i @ self.local_corners[4:].T).T  # 4 top corners
            for c in corners_i:
                c_rel_j = c - self.pos[j]
                c_loc_j = R_j.T @ c_rel_j
                if abs(c_loc_j[0]) <= self.hx + 0.02 and abs(c_loc_j[1]) <= self.hy + 0.02:
                    gap = -self.hz - c_loc_j[2]
                    if gap < 0.02:
                        normal_world = R_j @ np.array([0.0, 0.0, 1.0])
                        r_a = c - self.pos[j]
                        r_b = c - self.pos[i]
                        contacts.append((j, i, r_a, r_b, normal_world, gap))

        # Iterative Projected Gauss-Seidel solver for contact impulses
        contact_gaps = []
        interaction_forces = []

        if contacts:
            num_contacts = len(contacts)
            accum_pn = np.zeros(num_contacts)
            accum_pt = np.zeros((num_contacts, 2))

            # Precompute world inertia inverses
            inv_I_world = []
            for i in range(n):
                R = quat_to_rot_matrix(self.quat[i])
                inv_I_world.append(R @ self.inv_I_body @ R.T)

            # Solver iterations (Gauss-Seidel)
            num_iters = 30
            mu = 0.5  # friction coefficient
            baumgarte_beta = 0.2

            for iteration in range(num_iters):
                for k, (b_a, b_b, r_a, r_b, normal, gap) in enumerate(contacts):
                    if iteration == 0:
                        contact_gaps.append(gap)

                    # Compute relative velocity at contact point
                    v_a = self.vel[b_a] + np.cross(self.omega[b_a], r_a)
                    if b_b >= 0:
                        v_b = self.vel[b_b] + np.cross(self.omega[b_b], r_b)
                    else:
                        v_b = np.zeros(3)
                    v_rel = v_a - v_b

                    vn = np.dot(v_rel, normal)

                    # Compute effective mass for normal impulse
                    w_a = self.inv_mass[b_a] + np.dot(normal, np.cross(inv_I_world[b_a] @ np.cross(r_a, normal), normal))
                    if b_b >= 0:
                        w_b = self.inv_mass[b_b] + np.dot(normal, np.cross(inv_I_world[b_b] @ np.cross(r_b, normal), normal))
                    else:
                        w_b = 0.0
                    k_normal = w_a + w_b
                    if k_normal < 1e-8:
                        continue

                    # Baumgarte stabilization bias
                    bias = 0.0
                    if gap < 0.0:
                        bias = -baumgarte_beta * gap / dt

                    # Target relative normal velocity
                    delta_pn = -(vn - bias) / k_normal
                    old_pn = accum_pn[k]
                    accum_pn[k] = max(0.0, old_pn + delta_pn)
                    delta_pn = accum_pn[k] - old_pn

                    # Apply normal impulse
                    impulse_n = delta_pn * normal
                    self.vel[b_a] += impulse_n * self.inv_mass[b_a]
                    self.omega[b_a] += inv_I_world[b_a] @ np.cross(r_a, impulse_n)
                    if b_b >= 0:
                        self.vel[b_b] -= impulse_n * self.inv_mass[b_b]
                        self.omega[b_b] -= inv_I_world[b_b] @ np.cross(r_b, impulse_n)

                    # Tangent / Friction impulses
                    # Tangent basis
                    t1 = np.array([-normal[1], normal[0], 0.0])
                    if np.linalg.norm(t1) < 1e-4:
                        t1 = np.array([1.0, 0.0, 0.0])
                    else:
                        t1 /= np.linalg.norm(t1)
                    t2 = np.cross(normal, t1)

                    for ti, tangent in enumerate([t1, t2]):
                        v_rel = (self.vel[b_a] + np.cross(self.omega[b_a], r_a)) - \
                                (self.vel[b_b] + np.cross(self.omega[b_b], r_b) if b_b >= 0 else 0.0)
                        vt = np.dot(v_rel, tangent)

                        w_at = self.inv_mass[b_a] + np.dot(tangent, np.cross(inv_I_world[b_a] @ np.cross(r_a, tangent), tangent))
                        w_bt = (self.inv_mass[b_b] + np.dot(tangent, np.cross(inv_I_world[b_b] @ np.cross(r_b, tangent), tangent))) if b_b >= 0 else 0.0
                        k_tangent = w_at + w_bt
                        if k_tangent < 1e-8:
                            continue

                        delta_pt = -vt / k_tangent
                        old_pt = accum_pt[k, ti]
                        max_pt = mu * accum_pn[k]
                        accum_pt[k, ti] = np.clip(old_pt + delta_pt, -max_pt, max_pt)
                        delta_pt = accum_pt[k, ti] - old_pt

                        impulse_t = delta_pt * tangent
                        self.vel[b_a] += impulse_t * self.inv_mass[b_a]
                        self.omega[b_a] += inv_I_world[b_a] @ np.cross(r_a, impulse_t)
                        if b_b >= 0:
                            self.vel[b_b] -= impulse_t * self.inv_mass[b_b]
                            self.omega[b_b] -= inv_I_world[b_b] @ np.cross(r_b, impulse_t)

            # Calculate interaction forces for the step
            for k, (b_a, b_b, r_a, r_b, normal, gap) in enumerate(contacts):
                # Total contact impulse = normal + friction
                t1 = np.array([-normal[1], normal[0], 0.0])
                if np.linalg.norm(t1) < 1e-4:
                    t1 = np.array([1.0, 0.0, 0.0])
                else:
                    t1 /= np.linalg.norm(t1)
                t2 = np.cross(normal, t1)

                total_impulse = accum_pn[k] * normal + accum_pt[k, 0] * t1 + accum_pt[k, 1] * t2
                force = total_impulse / dt
                interaction_forces.append(force)

        # 2. Integrate positions and orientations
        for i in range(n):
            self.pos[i] += self.vel[i] * dt
            self.quat[i] = quat_integrate(self.quat[i], self.omega[i], dt)

        return contact_gaps, interaction_forces

    def advance_to(self, time: float) -> StateFrame:
        # Step forward to requested time
        while self.time + 1e-8 < time:
            step_size = min(self.sub_dt, time - self.time)
            gaps, forces = self._step(step_size)
            if gaps:
                self.last_contact_gaps = gaps
                self.last_interaction_forces = forces
            self.time += step_size

        # Format return frame
        ext_force = np.zeros((self.n_bodies, 3))
        for i in range(self.n_bodies):
            ext_force[i] = self.mass[i] * self.gravity

        if self.last_contact_gaps:
            contact_dist = np.array(self.last_contact_gaps, dtype=float)
            inter_force = np.array(self.last_interaction_forces, dtype=float)
        else:
            contact_dist = np.zeros(0, dtype=float)
            inter_force = np.zeros((0, 3), dtype=float)

        return StateFrame(
            time=time,
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            orientation=np.copy(self.quat),
            angular_velocity=np.copy(self.omega),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            fields={
                "contact_signed_distance": contact_dist,
                "external_force": ext_force,
                "interaction_force": inter_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            }
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "bodies": self.n_bodies,
            "bottom_z": float(self.pos[0, 2]) if self.n_bodies > 0 else 0.0,
            "top_z": float(self.pos[-1, 2]) if self.n_bodies > 0 else 0.0,
        }

