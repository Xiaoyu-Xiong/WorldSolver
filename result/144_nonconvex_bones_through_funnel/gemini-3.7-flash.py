"""Deterministic physics solver for nonconvex bones through a funnel."""
from __future__ import annotations

import os
import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ])


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion given angular velocity omega over dt."""
    # omega is [wx, wy, wz]
    angle = np.linalg.norm(omega)
    if angle < 1e-12:
        return q
    axis = omega / angle
    half_angle = 0.5 * angle * dt
    dq = np.array([axis[0] * np.sin(half_angle),
                   axis[1] * np.sin(half_angle),
                   axis[2] * np.sin(half_angle),
                   np.cos(half_angle)])
    # Quaternion multiplication dq * q
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q
    res = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])
    norm = np.linalg.norm(res)
    return res / norm if norm > 1e-12 else res


def load_obj_vertices(path: str) -> np.ndarray:
    """Load vertex positions from an OBJ file."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.strip().split()[1:4]])
    return np.array(verts)


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # Simulation states
        self.n_bodies = 0
        self.pos = None        # (N, 3)
        self.vel = None        # (N, 3)
        self.quat = None       # (N, 4)
        self.omega = None      # (N, 3)
        self.mass = None       # (N,)
        self.inv_mass = None   # (N,)
        self.body_id = None    # (N,)
        self.I_body = None     # (3, 3)
        self.inv_I_body = None # (3, 3)

        # Contact geometry
        self.bone_pts = None   # (P, 3)
        self.sphere_radius = 0.05
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.friction = 0.55
        self.restitution = 0.12
        self.length_scale = 0.02

        # Physics sub-stepping parameters
        self.sub_dt = 0.001

        # Diagnostics / recorded contact info
        self.last_ext_forces = None
        self.last_inter_forces = None
        self.last_contact_distances = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.current_time = 0.0

        config = scene.config
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact_cfg = config.get("contact", {})
        self.friction = float(contact_cfg.get("friction", 0.55))
        self.restitution = float(contact_cfg.get("restitution", 0.12))
        self.length_scale = float(contact_cfg.get("length_scale", 0.02))

        bodies = config["bodies"]
        self.n_bodies = len(bodies)
        self.pos = np.asarray([b["position"] for b in bodies], dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.quat = np.asarray([b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], dtype=float)
        self.omega = np.zeros_like(self.pos)
        self.mass = np.asarray([b["mass"] for b in bodies], dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.arange(self.n_bodies, dtype=np.int32)

        # Load bone mesh vertices to create collision proxy points
        asset_path = "assets/bone.obj"
        if not os.path.exists(asset_path):
            # Fallback path if running from different root
            asset_path = os.path.join(os.path.dirname(__file__), "..", "assets", "bone.obj")

        verts = load_obj_vertices(asset_path)
        # Farthest point sampling on mesh vertices to get a good multi-sphere coverage
        n_samples = 48
        sampled = [0]
        dists = np.linalg.norm(verts - verts[0], axis=1)
        for _ in range(1, n_samples):
            idx = int(np.argmax(dists))
            sampled.append(idx)
            dists = np.minimum(dists, np.linalg.norm(verts - verts[idx], axis=1))
        self.bone_pts = verts[sampled]
        self.sphere_radius = 0.045

        # Compute body-frame inertia tensor assuming uniform distribution of mesh vertices
        # for a typical mass = 0.35 kg
        m_sample = self.mass[0] / len(verts)
        I_xx = np.sum(verts[:, 1]**2 + verts[:, 2]**2) * m_sample
        I_yy = np.sum(verts[:, 0]**2 + verts[:, 2]**2) * m_sample
        I_zz = np.sum(verts[:, 0]**2 + verts[:, 1]**2) * m_sample
        self.I_body = np.diag([I_xx, I_yy, I_zz])
        self.inv_I_body = np.diag([1.0 / I_xx, 1.0 / I_yy, 1.0 / I_zz])

        # Initialize tracking fields
        self.last_ext_forces = np.zeros_like(self.pos)
        for i in range(self.n_bodies):
            self.last_ext_forces[i] = self.mass[i] * self.gravity
        self.last_inter_forces = np.zeros((1, 3), dtype=float)
        self.last_contact_distances = np.zeros(1, dtype=float)

    def _get_world_points(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute world-space positions of all proxy points and rotation matrices."""
        world_pts = np.zeros((self.n_bodies, len(self.bone_pts), 3))
        rot_matrices = []
        for i in range(self.n_bodies):
            R = quat_to_rot_matrix(self.quat[i])
            rot_matrices.append(R)
            world_pts[i] = self.pos[i] + (R @ self.bone_pts.T).T
        return world_pts, rot_matrices

    def _funnel_sdf(self, p: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Signed distance and outward normal to the funnel inner boundary.
        Funnel goes from z = 0.48 (throat: x in [-0.38, 0.38], y in [-0.28, 0.28])
        to z = 1.72 (top: x in [-1.35, 1.35], y in [-0.95, 0.95]).
        Below z = 0.48 or above z = 1.72, open air.
        """
        x, y, z = p[0], p[1], p[2]
        if z < 0.45 or z > 1.75:
            return 10.0, np.array([0.0, 0.0, 1.0])

        # Linear interpolation of half-widths
        u = np.clip((z - 0.48) / (1.72 - 0.48), 0.0, 1.0)
        hx = 0.38 + (1.35 - 0.38) * u
        hy = 0.28 + (0.95 - 0.28) * u

        # Slanted wall normals (pointing inwards into the funnel)
        # Wall X+: x = hx(z) -> x - (0.38 + 0.97/1.24 * (z - 0.48)) = 0
        # Normal pointing into funnel interior: [-nx, 0, nz]
        dx_dz = (1.35 - 0.38) / (1.72 - 0.48)
        norm_x = np.sqrt(1.0 + dx_dz**2)
        dist_x_pos = (hx - x) / norm_x
        dist_x_neg = (hx + x) / norm_x

        dy_dz = (0.95 - 0.28) / (1.72 - 0.48)
        norm_y = np.sqrt(1.0 + dy_dz**2)
        dist_y_pos = (hy - y) / norm_y
        dist_y_neg = (hy + y) / norm_y

        # Find closest wall
        dists = [dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg]
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_idx == 0:   # right wall (+x)
            normal = np.array([-1.0 / norm_x, 0.0, dx_dz / norm_x])
        elif min_idx == 1: # left wall (-x)
            normal = np.array([1.0 / norm_x, 0.0, dx_dz / norm_x])
        elif min_idx == 2: # front wall (+y)
            normal = np.array([0.0, -1.0 / norm_y, dy_dz / norm_y])
        else:              # back wall (-y)
            normal = np.array([0.0, 1.0 / norm_y, dy_dz / norm_y])

        return min_dist, normal

    def _step_physics(self, dt: float) -> tuple[list[float], list[np.ndarray]]:
        """Advance simulation by one sub-step dt using penalty & impulse contact response."""
        recorded_gaps = []
        recorded_forces = []

        world_pts, rot_matrices = self._get_world_points()

        # Accumulate forces and torques
        forces = np.zeros_like(self.pos)
        torques = np.zeros_like(self.pos)
        for i in range(self.n_bodies):
            forces[i] += self.mass[i] * self.gravity

        k_contact = 3500.0  # contact stiffness
        c_contact = 45.0    # contact damping
        r = self.sphere_radius

        # 1. Floor contact (z = 0, floor half_extents z=0.05, top surface at z=0.0)
        floor_z = 0.0
        for i in range(self.n_bodies):
            R = rot_matrices[i]
            for pt in world_pts[i]:
                gap = pt[2] - r - floor_z
                if gap < 0.05:
                    recorded_gaps.append(float(gap))
                    if gap < 0.0:
                        depth = -gap
                        # Contact point velocity
                        r_vec = pt - self.pos[i]
                        v_pt = self.vel[i] + np.cross(self.omega[i], r_vec)
                        n = np.array([0.0, 0.0, 1.0])
                        vn = np.dot(v_pt, n)

                        f_n_mag = k_contact * depth - c_contact * min(vn, 0.0)
                        f_n_mag = max(0.0, f_n_mag)
                        f_contact = f_n_mag * n

                        # Friction
                        v_t = v_pt - vn * n
                        v_t_mag = np.linalg.norm(v_t)
                        if v_t_mag > 1e-6:
                            t_dir = v_t / v_t_mag
                            f_t = -min(self.friction * f_n_mag, c_contact * v_t_mag) * t_dir
                            f_contact += f_t

                        forces[i] += f_contact
                        torques[i] += np.cross(r_vec, f_contact)
                        recorded_forces.append(f_contact)

        # 2. Funnel contact
        for i in range(self.n_bodies):
            for pt in world_pts[i]:
                if 0.45 <= pt[2] <= 1.75:
                    dist_to_wall, normal = self._funnel_sdf(pt)
                    gap = dist_to_wall - r
                    if gap < 0.05:
                        recorded_gaps.append(float(gap))
                        if gap < 0.0:
                            depth = -gap
                            r_vec = pt - self.pos[i]
                            v_pt = self.vel[i] + np.cross(self.omega[i], r_vec)
                            vn = np.dot(v_pt, normal)
                            f_n_mag = k_contact * depth - c_contact * min(vn, 0.0)
                            f_n_mag = max(0.0, f_n_mag)
                            f_contact = f_n_mag * normal

                            # Friction
                            v_t = v_pt - vn * normal
                            v_t_mag = np.linalg.norm(v_t)
                            if v_t_mag > 1e-6:
                                t_dir = v_t / v_t_mag
                                f_t = -min(self.friction * f_n_mag, c_contact * v_t_mag) * t_dir
                                f_contact += f_t

                            forces[i] += f_contact
                            torques[i] += np.cross(r_vec, f_contact)
                            recorded_forces.append(f_contact)

        # 3. Body-Body contact
        # Broadphase bounding sphere
        approx_radius = 0.55
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                d_ij = np.linalg.norm(self.pos[i] - self.pos[j])
                if d_ij < 2.0 * approx_radius:
                    # Narrowphase point-to-point sphere tests
                    pts_i = world_pts[i]
                    pts_j = world_pts[j]
                    diff = pts_i[:, None, :] - pts_j[None, :, :]  # (P, P, 3)
                    dist_matrix = np.linalg.norm(diff, axis=-1)
                    min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    min_dist = dist_matrix[min_dist_idx]
                    gap = min_dist - 2.0 * r
                    if gap < 0.05:
                        recorded_gaps.append(float(gap))

                    # Check all overlapping pairs
                    overlap_mask = dist_matrix < 2.0 * r
                    if np.any(overlap_mask):
                        pi_idxs, pj_idxs = np.where(overlap_mask)
                        for pi, pj in zip(pi_idxs, pj_idxs):
                            p_i = pts_i[pi]
                            p_j = pts_j[pj]
                            disp = p_i - p_j
                            dist = np.linalg.norm(disp)
                            if dist < 1e-8:
                                n = np.array([0.0, 0.0, 1.0])
                                dist = 1e-8
                            else:
                                n = disp / dist
                            depth = 2.0 * r - dist

                            r_i = p_i - self.pos[i]
                            r_j = p_j - self.pos[j]
                            v_i = self.vel[i] + np.cross(self.omega[i], r_i)
                            v_j = self.vel[j] + np.cross(self.omega[j], r_j)
                            v_rel = v_i - v_j
                            vn = np.dot(v_rel, n)

                            f_n_mag = k_contact * depth - c_contact * min(vn, 0.0)
                            f_n_mag = max(0.0, f_n_mag)
                            f_contact = f_n_mag * n

                            # Tangential friction
                            v_t = v_rel - vn * n
                            v_t_mag = np.linalg.norm(v_t)
                            if v_t_mag > 1e-6:
                                t_dir = v_t / v_t_mag
                                f_t = -min(self.friction * f_n_mag, c_contact * v_t_mag) * t_dir
                                f_contact += f_t

                            forces[i] += f_contact
                            torques[i] += np.cross(r_i, f_contact)
                            forces[j] -= f_contact
                            torques[j] -= np.cross(r_j, f_contact)
                            recorded_forces.append(f_contact)

        # Numerical Integration (Semi-implicit Euler)
        for i in range(self.n_bodies):
            # Linear
            acc = forces[i] * self.inv_mass[i]
            self.vel[i] += acc * dt
            # Small damping to avoid high-frequency energy buildup
            self.vel[i] *= (1.0 - 0.01 * dt)
            self.pos[i] += self.vel[i] * dt

            # Angular
            R = rot_matrices[i]
            I_world = R @ self.I_body @ R.T
            inv_I_world = R @ self.inv_I_body @ R.T

            # Gyroscopic term
            gyr = np.cross(self.omega[i], I_world @ self.omega[i])
            ang_acc = inv_I_world @ (torques[i] - gyr)
            self.omega[i] += ang_acc * dt
            self.omega[i] *= (1.0 - 0.02 * dt)

            self.quat[i] = quat_integrate(self.quat[i], self.omega[i], dt)

        return recorded_gaps, recorded_forces

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to the requested time."""
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            gaps, inter_forces = self._step_physics(dt)
            self.current_time += dt

            if len(gaps) > 0:
                self.last_contact_distances = np.array(gaps, dtype=float)
            if len(inter_forces) > 0:
                self.last_inter_forces = np.array(inter_forces, dtype=float)

        # External forces = gravity per body
        ext_forces = np.zeros((self.n_bodies, 3), dtype=float)
        for i in range(self.n_bodies):
            ext_forces[i] = self.mass[i] * self.gravity

        # Ensure contact distance and interaction forces are non-empty finite arrays
        contact_dist = self.last_contact_distances
        if contact_dist is None or len(contact_dist) == 0:
            contact_dist = np.array([0.05], dtype=float)

        inter_forces = self.last_inter_forces
        if inter_forces is None or len(inter_forces) == 0:
            inter_forces = np.zeros((1, 3), dtype=float)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": contact_dist.copy(),
                "external_force": ext_forces,
                "interaction_force": inter_forces.copy(),
            },
            scalars={
                "contact_length_scale": float(self.length_scale),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_bodies": self.n_bodies,
        }

