"""Rigid body physics simulation solver for chess piece pile settling."""
from __future__ import annotations

import os
import numpy as np
from worldsolver.contracts import StateFrame


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize quaternion
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n > 1e-12:
        x, y, z, w = x/n, y/n, z/n, w/n
    else:
        return np.eye(3)

    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y)]
    ], dtype=float)


def quat_integrate(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Integrate orientation quaternion q by angular velocity w over dt."""
    # dq/dt = 0.5 * w_quat * q, where w_quat = [wx, wy, wz, 0]
    wx, wy, wz = w
    w_len = np.sqrt(wx*wx + wy*wy + wz*wz)
    if w_len < 1e-12:
        return q

    angle = w_len * dt
    axis = w / w_len
    s = np.sin(0.5 * angle)
    c = np.cos(0.5 * angle)
    dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c])

    # Quaternion multiply dq * q
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q
    q_new = np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
    norm = np.linalg.norm(q_new)
    if norm > 1e-12:
        q_new /= norm
    return q_new


def load_obj_vertices(path: str) -> np.ndarray:
    """Load 3D vertex positions from an OBJ file."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=float)


def compute_inertia_tensor(verts: np.ndarray, mass: float) -> np.ndarray:
    """Compute approximate inertia tensor assuming uniform distribution over vertices."""
    # Shift to centroid
    c = np.mean(verts, axis=0)
    v = verts - c
    n = len(v)
    m_pt = mass / max(n, 1)

    Ixx = np.sum(m_pt * (v[:, 1]**2 + v[:, 2]**2))
    Iyy = np.sum(m_pt * (v[:, 0]**2 + v[:, 2]**2))
    Izz = np.sum(m_pt * (v[:, 0]**2 + v[:, 1]**2))
    Ixy = -np.sum(m_pt * v[:, 0] * v[:, 1])
    Ixz = -np.sum(m_pt * v[:, 0] * v[:, 2])
    Iyz = -np.sum(m_pt * v[:, 1] * v[:, 2])

    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ], dtype=float)
    return I


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.num_bodies = 0
        self.current_time = 0.0

        # State arrays
        self.positions = None      # [N, 3]
        self.velocities = None     # [N, 3]
        self.orientations = None   # [N, 4]
        self.ang_velocities = None # [N, 3]
        self.masses = None         # [N]
        self.inv_masses = None     # [N]

        # Geometry data
        self.local_points = []     # List of [P_i, 3] for each body
        self.local_inertia = []    # List of [3, 3]
        self.inv_inertia_body = [] # List of [3, 3]

        # Floor parameters
        self.floor_z = 0.0
        self.floor_half_x = 3.2
        self.floor_half_y = 2.2

        # Simulation parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.friction = 0.55
        self.restitution = 0.12
        self.length_scale = 0.02
        self.sim_substep_dt = 0.001

        # Contact tracking for exports
        self.contact_signed_distances = []
        self.interaction_forces = []
        self.external_forces = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        cfg = scene.config
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact_cfg = cfg.get("contact", {})
        self.friction = float(contact_cfg.get("friction", 0.55))
        self.restitution = float(contact_cfg.get("restitution", 0.12))
        self.length_scale = float(contact_cfg.get("length_scale", 0.02))

        bodies_cfg = cfg["bodies"]
        self.num_bodies = len(bodies_cfg)

        self.positions = np.asarray([b["position"] for b in bodies_cfg], dtype=float)
        self.velocities = np.zeros((self.num_bodies, 3), dtype=float)
        self.orientations = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies_cfg], dtype=float)
        self.ang_velocities = np.zeros((self.num_bodies, 3), dtype=float)
        self.masses = np.asarray([b["mass"] for b in bodies_cfg], dtype=float)
        self.inv_masses = 1.0 / self.masses

        # Map body IDs to geometry mesh assets
        # Find OBJ asset for each piece from scene_geometry or heuristic
        asset_map = {
            0: "pawn.obj", 1: "rook.obj", 2: "knight.obj", 3: "bishop.obj",
            4: "queen.obj", 5: "king.obj", 6: "pawn.obj", 7: "rook.obj",
            8: "knight.obj", 9: "bishop.obj", 10: "queen.obj", 11: "king.obj",
            12: "pawn.obj", 13: "rook.obj", 14: "bishop.obj", 15: "king.obj"
        }

        self.local_points = []
        self.local_inertia = []
        self.inv_inertia_body = []

        for i in range(self.num_bodies):
            mesh_name = asset_map.get(i, "pawn.obj")
            mesh_path = os.path.join("assets", mesh_name)
            if os.path.exists(mesh_path):
                verts = load_obj_vertices(mesh_path)
            else:
                # Fallback box vertices
                verts = np.array([
                    [-0.2, -0.2, -0.2], [0.2, -0.2, -0.2], [0.2, 0.2, -0.2], [-0.2, 0.2, -0.2],
                    [-0.2, -0.2,  0.2], [0.2, -0.2,  0.2], [0.2, 0.2,  0.2], [-0.2, 0.2,  0.2]
                ])

            # Subsample vertices if too many for fast contact checking
            # Keep representative sample of surface vertices
            if len(verts) > 64:
                step = len(verts) // 64
                sample_pts = verts[::step]
            else:
                sample_pts = verts

            self.local_points.append(sample_pts)

            I = compute_inertia_tensor(verts, self.masses[i])
            # Regularize inertia
            diag = np.diag(I)
            if (diag <= 0).any():
                I = np.eye(3) * (self.masses[i] * 0.05)
            self.local_inertia.append(I)
            self.inv_inertia_body.append(np.linalg.inv(I))

        self.current_time = 0.0
        self.external_forces = np.zeros((self.num_bodies, 3), dtype=float)
        for i in range(self.num_bodies):
            self.external_forces[i] = self.masses[i] * self.gravity
        self.contact_signed_distances = np.array([0.0], dtype=float)
        self.interaction_forces = np.zeros((1, 3), dtype=float)

    def _step_substep(self, dt: float) -> None:
        """One physics substep: integrate forces, resolve contacts/impulses, integrate positions."""
        num_bodies = self.num_bodies

        # 1. Integrate gravity and apply linear/angular damping
        damping_lin = (1.0 - 0.2 * dt)
        damping_ang = (1.0 - 0.5 * dt)

        for i in range(num_bodies):
            self.velocities[i] += self.gravity * dt
            self.velocities[i] *= damping_lin
            self.ang_velocities[i] *= damping_ang

        # Precompute rotation matrices and world inverse inertia tensors
        R_list = []
        inv_I_world = []
        world_pts_list = []

        for i in range(num_bodies):
            R = quat_to_rot_matrix(self.orientations[i])
            R_list.append(R)
            inv_I = R @ self.inv_inertia_body[i] @ R.T
            inv_I_world.append(inv_I)
            # Transform local sample points to world coordinates
            pts_w = self.positions[i] + (self.local_points[i] @ R.T)
            world_pts_list.append(pts_w)

        # Contact accumulation for output
        frame_gaps = []
        frame_forces = []

        # 2. Contact resolution iterations
        num_solver_iters = 6
        for _ in range(num_solver_iters):
            # A. Floor contacts
            floor_n = np.array([0.0, 0.0, 1.0])
            for i in range(num_bodies):
                pts_w = self.positions[i] + (self.local_points[i] @ R_list[i].T)
                # Check points below or near floor (z <= 0)
                z_vals = pts_w[:, 2] - self.floor_z
                below_mask = z_vals < 0.02
                if not np.any(below_mask):
                    continue

                for idx in np.where(below_mask)[0]:
                    pt = pts_w[idx]
                    gap = pt[2] - self.floor_z
                    frame_gaps.append(gap)

                    r_i = pt - self.positions[i]
                    # Velocity of contact point
                    v_pt = self.velocities[i] + np.cross(self.ang_velocities[i], r_i)
                    vn = np.dot(v_pt, floor_n)

                    # Position correction / Baumgarte stabilization
                    penetration = max(0.0, -gap)
                    bias = 0.3 * penetration / dt if penetration > 0 else 0.0

                    # Effective mass in normal direction
                    r_cross_n = np.cross(r_i, floor_n)
                    k_normal = self.inv_masses[i] + np.dot(r_cross_n, inv_I_world[i] @ r_cross_n)

                    # Target relative velocity
                    target_vn = -self.restitution * vn if vn < -0.1 else 0.0
                    j_n_mag = (target_vn - vn + bias) / max(k_normal, 1e-6)

                    if j_n_mag > 0:
                        impulse_n = j_n_mag * floor_n
                        self.velocities[i] += self.inv_masses[i] * impulse_n
                        self.ang_velocities[i] += inv_I_world[i] @ np.cross(r_i, impulse_n)

                        # Friction impulse
                        v_tangent = v_pt - vn * floor_n
                        v_tang_norm = np.linalg.norm(v_tangent)
                        if v_tang_norm > 1e-6:
                            t_dir = -v_tangent / v_tang_norm
                            r_cross_t = np.cross(r_i, t_dir)
                            k_tangent = self.inv_masses[i] + np.dot(r_cross_t, inv_I_world[i] @ r_cross_t)
                            j_t_mag = v_tang_norm / max(k_tangent, 1e-6)
                            j_t_mag = min(j_t_mag, self.friction * j_n_mag)
                            impulse_t = j_t_mag * t_dir
                            self.velocities[i] += self.inv_masses[i] * impulse_t
                            self.ang_velocities[i] += inv_I_world[i] @ np.cross(r_i, impulse_t)
                            frame_forces.append((impulse_n + impulse_t) / dt)
                        else:
                            frame_forces.append(impulse_n / dt)

            # B. Body-to-body contacts
            for i in range(num_bodies):
                for j in range(i + 1, num_bodies):
                    # Broadphase sphere check
                    d_pos = self.positions[j] - self.positions[i]
                    dist_sq = np.dot(d_pos, d_pos)
                    # Pieces have bounding radius approx 0.45 m
                    if dist_sq > (0.9)**2:
                        continue

                    dist = np.sqrt(max(dist_sq, 1e-8))
                    n_ij = d_pos / dist # Normal pointing from i to j

                    pts_i = self.positions[i] + (self.local_points[i] @ R_list[i].T)
                    pts_j = self.positions[j] + (self.local_points[j] @ R_list[j].T)

                    # Project points along normal to find closest pairs
                    proj_i = pts_i @ n_ij
                    proj_j = pts_j @ n_ij

                    max_i_idx = np.argmax(proj_i)
                    min_j_idx = np.argmin(proj_j)

                    pt_i = pts_i[max_i_idx]
                    pt_j = pts_j[min_j_idx]

                    # Signed distance along contact normal
                    gap = np.dot(pt_j - pt_i, n_ij)
                    frame_gaps.append(gap)

                    if gap < 0.01:
                        # Contact detected!
                        penetration = max(0.0, -gap)
                        contact_pt = 0.5 * (pt_i + pt_j)
                        r_i = contact_pt - self.positions[i]
                        r_j = contact_pt - self.positions[j]

                        v_i = self.velocities[i] + np.cross(self.ang_velocities[i], r_i)
                        v_j = self.velocities[j] + np.cross(self.ang_velocities[j], r_j)
                        v_rel = v_j - v_i
                        vn = np.dot(v_rel, n_ij)

                        bias = 0.25 * penetration / dt if penetration > 0 else 0.0

                        r_i_cross_n = np.cross(r_i, n_ij)
                        r_j_cross_n = np.cross(r_j, n_ij)

                        k_norm = (self.inv_masses[i] + self.inv_masses[j] +
                                  np.dot(r_i_cross_n, inv_I_world[i] @ r_i_cross_n) +
                                  np.dot(r_j_cross_n, inv_I_world[j] @ r_j_cross_n))

                        target_vn = -self.restitution * vn if vn < -0.1 else 0.0
                        j_n_mag = (target_vn - vn + bias) / max(k_norm, 1e-6)

                        # We want to push i backward and j forward along n_ij
                        # If relative velocity vn < 0, they are approaching
                        if j_n_mag > 0:
                            impulse_n = j_n_mag * n_ij
                            self.velocities[i] -= self.inv_masses[i] * impulse_n
                            self.ang_velocities[i] -= inv_I_world[i] @ np.cross(r_i, impulse_n)

                            self.velocities[j] += self.inv_masses[j] * impulse_n
                            self.ang_velocities[j] += inv_I_world[j] @ np.cross(r_j, impulse_n)

                            # Friction
                            v_tangent = v_rel - vn * n_ij
                            v_tang_norm = np.linalg.norm(v_tangent)
                            if v_tang_norm > 1e-6:
                                t_dir = -v_tangent / v_tang_norm
                                r_i_cross_t = np.cross(r_i, t_dir)
                                r_j_cross_t = np.cross(r_j, t_dir)
                                k_tang = (self.inv_masses[i] + self.inv_masses[j] +
                                          np.dot(r_i_cross_t, inv_I_world[i] @ r_i_cross_t) +
                                          np.dot(r_j_cross_t, inv_I_world[j] @ r_j_cross_t))
                                j_t_mag = v_tang_norm / max(k_tang, 1e-6)
                                j_t_mag = min(j_t_mag, self.friction * j_n_mag)
                                impulse_t = j_t_mag * t_dir

                                self.velocities[i] -= self.inv_masses[i] * impulse_t
                                self.ang_velocities[i] -= inv_I_world[i] @ np.cross(r_i, impulse_t)

                                self.velocities[j] += self.inv_masses[j] * impulse_t
                                self.ang_velocities[j] += inv_I_world[j] @ np.cross(r_j, impulse_t)
                                frame_forces.append((impulse_n + impulse_t) / dt)
                            else:
                                frame_forces.append(impulse_n / dt)

        # 3. Integrate positions and orientations
        for i in range(num_bodies):
            self.positions[i] += self.velocities[i] * dt
            self.orientations[i] = quat_integrate(self.orientations[i], self.ang_velocities[i], dt)

            # Boundary floor hard clamp to prevent falling through infinity
            if self.positions[i, 2] < 0.1:
                self.positions[i, 2] = max(self.positions[i, 2], 0.1)
                if self.velocities[i, 2] < 0:
                    self.velocities[i, 2] = 0.0

        if len(frame_gaps) > 0:
            self.contact_signed_distances = np.asarray(frame_gaps, dtype=float)
        else:
            self.contact_signed_distances = np.array([0.0], dtype=float)

        if len(frame_forces) > 0:
            self.interaction_forces = np.asarray(frame_forces, dtype=float)
        else:
            self.interaction_forces = np.zeros((1, 3), dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation from current_time to target time
        sub_dt = self.sim_substep_dt
        while self.current_time < time - 1e-9:
            step_size = min(sub_dt, time - self.current_time)
            self._step_substep(step_size)
            self.current_time += step_size

        self.current_time = time

        # External forces = gravity for each body
        ext_forces = np.zeros((self.num_bodies, 3), dtype=float)
        for i in range(self.num_bodies):
            ext_forces[i] = self.masses[i] * self.gravity

        frame = StateFrame(
            time=float(time),
            position=np.copy(self.positions),
            velocity=np.copy(self.velocities),
            orientation=np.copy(self.orientations),
            angular_velocity=np.copy(self.ang_velocities),
            mass=np.copy(self.masses),
            body_id=np.arange(self.num_bodies, dtype=np.int32),
            scalars={"contact_length_scale": float(self.length_scale)},
            fields={
                "contact_signed_distance": np.copy(self.contact_signed_distances),
                "external_force": ext_forces,
                "interaction_force": np.copy(self.interaction_forces),
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_bodies": self.num_bodies,
            "current_time": self.current_time
        }

