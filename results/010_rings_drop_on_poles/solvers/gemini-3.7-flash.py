"""Physics simulation solver for 010_rings_drop_on_poles benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternions [N, 4] (x, y, z, w) to rotation matrices [N, 3, 3]."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - (tyy + tzz)
    R[:, 0, 1] = txy - twz
    R[:, 0, 2] = txz + twy

    R[:, 1, 0] = txy + twz
    R[:, 1, 1] = 1.0 - (txx + tzz)
    R[:, 1, 2] = tyz - twx

    R[:, 2, 0] = txz - twy
    R[:, 2, 1] = tyz + twx
    R[:, 2, 2] = 1.0 - (txx + tyy)
    return R


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternions [N, 4] with angular velocity [N, 3] over dt."""
    # omega magnitude
    theta = np.linalg.norm(omega, axis=1, keepdims=True)
    # create delta quat
    half_theta = 0.5 * theta * dt
    # sinc term: sin(half_theta)/theta
    sinc = np.where(theta > 1e-8, np.sin(half_theta) / (theta + 1e-12), 0.5 * dt)
    dq_xyz = omega * sinc
    dq_w = np.cos(half_theta)

    # Quaternion multiplication: q_new = dq * q
    x1, y1, z1, w1 = dq_xyz[:, 0], dq_xyz[:, 1], dq_xyz[:, 2], dq_w[:, 0]
    x2, y2, z2, w2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    q_new = np.empty_like(q)
    q_new[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_new[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_new[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_new[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Normalize
    norm = np.linalg.norm(q_new, axis=1, keepdims=True)
    q_new /= np.maximum(norm, 1e-12)
    return q_new


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Particle discretization of ring geometry in local frame
        # inner_profile: [[0.18, 0.00], [0.28, 0.02], [0.34, 0.08], [0.34, 0.18], [0.28, 0.24], [0.18, 0.26]]
        # wall_thickness = 0.06
        # Center of mass is roughly z = 0.13, R_mid ~ 0.26
        # We sample points around the torus/ring body
        local_samples = []
        # Sample azimuthally (16 angles) and at several cross-section points (e.g., 6 points around cross-section)
        n_phi = 16
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        profile_pts = [
            (0.18, 0.00), (0.28, 0.02), (0.34, 0.08),
            (0.34, 0.18), (0.28, 0.24), (0.18, 0.26),
            (0.24, 0.00), (0.34, 0.02), (0.40, 0.08),
            (0.40, 0.18), (0.34, 0.24), (0.24, 0.26),
        ]
        # Shift z so local origin is at ring center (z_center ~ 0.13)
        z_offset = 0.13
        for r, z in profile_pts:
            for phi in phis:
                local_samples.append([r * np.cos(phi), r * np.sin(phi), z - z_offset])
        self.local_points = np.array(local_samples, dtype=np.float64)  # [P, 3]
        self.num_sample_pts = len(self.local_points)

        # Pole specs
        pole_xs = [-0.9, 0.0, 0.9]
        pole_ys = [-0.9, 0.0, 0.9]
        self.poles = []
        for px in pole_xs:
            for py in pole_ys:
                self.poles.append([px, py])
        self.poles = np.array(self.poles, dtype=np.float64)  # [9, 2]
        self.pole_radius = 0.075
        self.pole_height = 2.6
        self.ground_z = 0.0
        self.point_radius = 0.03  # collision padding on sample points

        # Sub-stepping dt
        self.sub_dt = 0.001

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        self.N = len(init_frame.position)
        self.pos = np.array(init_frame.position, dtype=np.float64)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)
        if init_frame.orientation is not None:
            self.quat = np.array(init_frame.orientation, dtype=np.float64)
        else:
            self.quat = np.tile([0.0, 0.0, 0.0, 1.0], (self.N, 1))

        if init_frame.angular_velocity is not None:
            self.omega = np.array(init_frame.angular_velocity, dtype=np.float64)
        else:
            self.omega = np.zeros((self.N, 3), dtype=np.float64)

        if init_frame.mass is not None:
            self.mass = np.array(init_frame.mass, dtype=np.float64)
        else:
            self.mass = np.ones(self.N, dtype=np.float64)

        self.body_id = np.arange(self.N, dtype=np.int64)

        # Ring mass and inertia tensor (ring of R~0.26, mass m=1.0)
        # For a ring/torus in local frame: I_xx = I_yy ~ 0.5 * m * R^2, I_zz ~ m * R^2
        r_mid = 0.28
        self.I_body = np.diag([0.5 * r_mid**2 + 0.01, 0.5 * r_mid**2 + 0.01, r_mid**2 + 0.01])
        self.inv_I_body = np.linalg.inv(self.I_body)

        # Contact tracking for fields
        self.last_contact_sdfs = np.array([0.0], dtype=np.float64)
        self.last_interaction_force = np.zeros((1, 3), dtype=np.float64)

    def _step(self, dt: float):
        N = self.N
        R = quat_to_rotmat(self.quat)  # [N, 3, 3]

        # World positions of all sample points: [N, P, 3]
        world_pts = self.pos[:, None, :] + np.einsum('nij,pj->npi', R, self.local_points)
        r_arms = world_pts - self.pos[:, None, :]  # [N, P, 3]

        # Velocity of each sample point: v + omega x r
        # omega x r:
        omega_cross_r = np.cross(self.omega[:, None, :], r_arms)  # [N, P, 3]
        pt_vel = self.vel[:, None, :] + omega_cross_r  # [N, P, 3]

        # Forces and torques accumulator
        total_force = np.zeros((N, 3), dtype=np.float64)
        total_torque = np.zeros((N, 3), dtype=np.float64)

        # Apply gravity
        total_force += self.mass[:, None] * self.gravity[None, :]

        # External linear and angular drag (air resistance + stability)
        total_force -= 0.15 * self.vel
        total_torque -= 0.05 * self.omega

        contact_distances = []
        contact_forces = []

        # 1. Ground collision: ground at z = 0.0
        # Penetration depth: z_pt - (ground_z + point_radius)
        z_pts = world_pts[:, :, 2]
        ground_pen = (self.ground_z + self.point_radius) - z_pts  # > 0 means penetrating
        ground_mask = ground_pen > 0

        if np.any(ground_mask):
            k_ground = 12000.0 / self.num_sample_pts
            d_ground = 120.0 / self.num_sample_pts
            mu_ground = 0.4

            pen = ground_pen[ground_mask]
            # Normal force in +z
            fn_mag = k_ground * pen - d_ground * pt_vel[:, :, 2][ground_mask]
            fn_mag = np.maximum(fn_mag, 0.0)

            f_normal = np.zeros((len(pen), 3), dtype=np.float64)
            f_normal[:, 2] = fn_mag

            # Friction in xy plane
            v_tan = pt_vel[ground_mask].copy()
            v_tan[:, 2] = 0.0
            v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
            f_fric = np.where(
                v_tan_norm > 1e-6,
                -v_tan / (v_tan_norm + 1e-6) * np.minimum(mu_ground * fn_mag[:, None], 15.0 * v_tan_norm),
                0.0
            )

            f_ground = f_normal + f_fric

            # Accumulate on bodies
            body_indices = np.where(ground_mask)[0]
            for i, b_idx in enumerate(body_indices):
                total_force[b_idx] += f_ground[i]
                total_torque[b_idx] += np.cross(r_arms[ground_mask][i], f_ground[i])

            contact_distances.extend((-pen).tolist())
            contact_forces.extend(f_ground.tolist())

        # 2. Pole collision: vertical cylinders along z from z=0 to z=2.6
        k_pole = 15000.0 / self.num_sample_pts
        d_pole = 150.0 / self.num_sample_pts
        mu_pole = 0.35

        for pole in self.poles:
            # dx, dy to pole center
            dx = world_pts[:, :, 0] - pole[0]
            dy = world_pts[:, :, 1] - pole[1]
            dist_xy = np.sqrt(dx * dx + dy * dy)
            z = world_pts[:, :, 2]

            # In z range of pole: 0 <= z <= pole_height
            pole_mask = (dist_xy < (self.pole_radius + self.point_radius)) & (z >= 0.0) & (z <= self.pole_height)

            if np.any(pole_mask):
                dist = dist_xy[pole_mask]
                pen = (self.pole_radius + self.point_radius) - dist
                nx = dx[pole_mask] / np.maximum(dist, 1e-6)
                ny = dy[pole_mask] / np.maximum(dist, 1e-6)
                normal = np.stack([nx, ny, np.zeros_like(nx)], axis=-1)

                v_pts_masked = pt_vel[pole_mask]
                v_norm = np.sum(v_pts_masked * normal, axis=-1)

                fn_mag = k_pole * pen - d_pole * v_norm
                fn_mag = np.maximum(fn_mag, 0.0)
                f_norm = normal * fn_mag[:, None]

                # Friction (tangent to normal and z)
                v_tan = v_pts_masked - normal * v_norm[:, None]
                v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
                f_fric = np.where(
                    v_tan_norm > 1e-6,
                    -v_tan / (v_tan_norm + 1e-6) * np.minimum(mu_pole * fn_mag[:, None], 15.0 * v_tan_norm),
                    0.0
                )

                f_pole = f_norm + f_fric

                body_indices = np.where(pole_mask)[0]
                for i, b_idx in enumerate(body_indices):
                    total_force[b_idx] += f_pole[i]
                    total_torque[b_idx] += np.cross(r_arms[pole_mask][i], f_pole[i])

                contact_distances.extend((-pen).tolist())
                contact_forces.extend(f_pole.tolist())

        # 3. Inter-ring collision (ring center & ring sample points vs other rings)
        # Coarse bounding sphere check first: R_bound ~ 0.40
        r_bound = 0.40
        pos_diff = self.pos[:, None, :] - self.pos[None, :, :]  # [N, N, 3]
        dist_sq = np.sum(pos_diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e9)

        # Pairs within interaction distance
        pair_i, pair_j = np.where(dist_sq < (2 * r_bound)**2)
        valid_pairs = pair_i < pair_j
        pair_i = pair_i[valid_pairs]
        pair_j = pair_j[valid_pairs]

        if len(pair_i) > 0:
            k_ring = 8000.0 / 16.0
            d_ring = 80.0 / 16.0
            mu_ring = 0.3

            # Subsampled ring collision points (8 points per ring)
            sub_indices = np.linspace(0, self.num_sample_pts - 1, 16, dtype=int)
            sub_pts = world_pts[:, sub_indices, :]  # [N, 16, 3]
            sub_vel = pt_vel[:, sub_indices, :]  # [N, 16, 3]
            sub_r = r_arms[:, sub_indices, :]

            for pi, pj in zip(pair_i, pair_j):
                pts_i = sub_pts[pi]  # [16, 3]
                pts_j = sub_pts[pj]  # [16, 3]

                # Distances between all pairs of sample points
                d_mat = pts_i[:, None, :] - pts_j[None, :, :]  # [16, 16, 3]
                dist_mat = np.linalg.norm(d_mat, axis=-1)  # [16, 16]

                coll_mask = dist_mat < (2.0 * self.point_radius)
                if np.any(coll_mask):
                    idx_i, idx_j = np.where(coll_mask)
                    dist = dist_mat[coll_mask]
                    pen = (2.0 * self.point_radius) - dist

                    norm = d_mat[coll_mask] / np.maximum(dist[:, None], 1e-6)  # points from j to i
                    rel_vel = sub_vel[pi][idx_i] - sub_vel[pj][idx_j]
                    vn = np.sum(rel_vel * norm, axis=-1)

                    fn_mag = k_ring * pen - d_ring * vn
                    fn_mag = np.maximum(fn_mag, 0.0)
                    fn = norm * fn_mag[:, None]

                    v_tan = rel_vel - norm * vn[:, None]
                    v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
                    f_fric = np.where(
                        v_tan_norm > 1e-6,
                        -v_tan / (v_tan_norm + 1e-6) * np.minimum(mu_ring * fn_mag[:, None], 15.0 * v_tan_norm),
                        0.0
                    )

                    f_total_pair = fn + f_fric

                    # Accumulate onto pi and pj
                    for k in range(len(fn_mag)):
                        total_force[pi] += f_total_pair[k]
                        total_torque[pi] += np.cross(sub_r[pi][idx_i[k]], f_total_pair[k])

                        total_force[pj] -= f_total_pair[k]
                        total_torque[pj] -= np.cross(sub_r[pj][idx_j[k]], f_total_pair[k])

                    contact_distances.extend((-pen).tolist())
                    contact_forces.extend(f_total_pair.tolist())

        # Update contact statistics
        if len(contact_distances) > 0:
            self.last_contact_sdfs = np.array(contact_distances, dtype=np.float64)
            self.last_interaction_force = np.array(contact_forces, dtype=np.float64)
        else:
            self.last_contact_sdfs = np.array([0.0], dtype=np.float64)
            self.last_interaction_force = np.zeros((1, 3), dtype=np.float64)

        # Integration (Symplectic Euler)
        # Linear motion
        acc = total_force / self.mass[:, None]
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Angular motion: I_world = R * I_body * R^T => I_world_inv = R * inv_I_body * R^T
        inv_I_world = np.einsum('nij,jk,nkl->nil', R, self.inv_I_body, np.transpose(R, (0, 2, 1)))
        alpha = np.einsum('nij,nj->ni', inv_I_world, total_torque)
        self.omega += alpha * dt
        # Angular damping
        self.omega *= np.exp(-0.2 * dt)

        # Update orientation
        self.quat = quat_integrate(self.quat, self.omega, dt)

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation to target time
        while self.current_time + 1e-7 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        ext_forces = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            scalars={
                "contact_length_scale": float(self.point_radius * 2.0),
            },
            fields={
                "contact_signed_distance": self.last_contact_sdfs.copy(),
                "external_force": ext_forces,
                "interaction_force": self.last_interaction_force.copy(),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.current_time,
            "mean_z": float(np.mean(self.pos[:, 2])),
            "kinetic_energy": float(0.5 * np.sum(self.mass[:, None] * self.vel**2)),
        }

