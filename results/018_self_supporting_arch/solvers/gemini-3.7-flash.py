"""Solver Agent implementation for Self-Supporting Arch scene.

Simulates rigid voussoir blocks forming a self-supporting parabolic arch
under gravity and frictional contact, based on Rojas et al.
"""
from __future__ import annotations

import os
import numpy as np
from worldsolver.contracts import StateFrame


def parse_obj(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a simple wavefront OBJ file into vertices and triangular faces."""
    verts = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Faces can be 1-indexed integers or v/vt/vn
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    # Normalize quaternion
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n > 1e-12:
        x, y, z, w = x / n, y / n, z / n, w / n
    else:
        return np.eye(3)

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion given angular velocity vector (world frame)."""
    # q is [x, y, z, w]
    omega_mag = np.linalg.norm(omega)
    if omega_mag < 1e-12:
        return q.copy()
    theta = omega_mag * dt
    axis = omega / omega_mag
    dq = np.array([
        axis[0] * np.sin(0.5 * theta),
        axis[1] * np.sin(0.5 * theta),
        axis[2] * np.sin(0.5 * theta),
        np.cos(0.5 * theta),
    ], dtype=float)
    # Quaternion multiplication dq * q (since dq is in world frame)
    # (x1, y1, z1, w1) * (x2, y2, z2, w2)
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q
    res = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)
    res_norm = np.linalg.norm(res)
    if res_norm > 1e-12:
        res /= res_norm
    return res


def compute_inertia_tensor(verts: np.ndarray, mass: float) -> np.ndarray:
    """Approximate inertia tensor from vertex bounding box / point cloud."""
    # Assuming uniform mass distributed among vertices
    n_v = len(verts)
    m_v = mass / n_v
    I = np.zeros((3, 3), dtype=float)
    for v in verts:
        x, y, z = v
        I[0, 0] += m_v * (y * y + z * z)
        I[1, 1] += m_v * (x * x + z * z)
        I[2, 2] += m_v * (x * x + y * y)
        I[0, 1] -= m_v * (x * y)
        I[1, 0] -= m_v * (x * y)
        I[0, 2] -= m_v * (x * z)
        I[2, 0] -= m_v * (x * z)
        I[1, 2] -= m_v * (y * z)
        I[2, 1] -= m_v * (y * z)
    return I


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.num_bodies = 0
        self.positions = None
        self.velocities = None
        self.orientations = None
        self.angular_velocities = None
        self.masses = None
        self.inv_masses = None
        self.body_ids = None
        self.local_vertices = []
        self.local_inertias = []
        self.inv_inertias_local = []
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.mu = 0.6
        self.sub_dt = 0.001
        self.current_time = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init_frame = scene.initial_state()
        self.num_bodies = len(init_frame.position)
        self.positions = np.array(init_frame.position, dtype=float, copy=True)
        self.velocities = np.array(init_frame.velocity, dtype=float, copy=True)
        self.orientations = np.array(init_frame.orientation, dtype=float, copy=True)
        self.angular_velocities = np.array(init_frame.angular_velocity, dtype=float, copy=True)
        self.masses = np.array(init_frame.mass, dtype=float, copy=True)
        self.inv_masses = 1.0 / self.masses
        self.body_ids = np.array(init_frame.body_id, dtype=int, copy=True)
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        friction_dict = scene.config.get("friction", {})
        self.mu = float(friction_dict.get("coefficient", 0.6)) if isinstance(friction_dict, dict) else 0.6

        # Load mesh geometry for all blocks
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "assets")
        self.local_vertices = []
        self.local_inertias = []
        self.inv_inertias_local = []

        for i in range(self.num_bodies):
            obj_path = os.path.join(assets_dir, f"voussoir_{i:02d}.obj")
            if os.path.exists(obj_path):
                v, _ = parse_obj(obj_path)
            else:
                # Fallback box
                v = np.array([
                    [-0.2, -0.4, -0.2], [0.2, -0.4, -0.2],
                    [-0.2, 0.4, -0.2], [0.2, 0.4, -0.2],
                    [-0.2, -0.4, 0.2], [0.2, -0.4, 0.2],
                    [-0.2, 0.4, 0.2], [0.2, 0.4, 0.2],
                ], dtype=float)
            self.local_vertices.append(v)
            I_loc = compute_inertia_tensor(v, self.masses[i])
            self.local_inertias.append(I_loc)
            self.inv_inertias_local.append(np.linalg.inv(I_loc))

        self.current_time = 0.0

    def _get_world_vertices(self, i: int) -> np.ndarray:
        R = quat_to_rotmat(self.orientations[i])
        return self.positions[i] + self.local_vertices[i] @ R.T

    def _get_inv_inertia_world(self, i: int) -> np.ndarray:
        R = quat_to_rotmat(self.orientations[i])
        return R @ self.inv_inertias_local[i] @ R.T

    def _collect_contacts(self) -> list[dict]:
        """Detect contacts between adjacent blocks and between blocks and ground plane z=0."""
        contacts = []

        # 1. Ground contacts for all blocks (especially springing blocks 0 and 14)
        for i in range(self.num_bodies):
            w_verts = self._get_world_vertices(i)
            for p in w_verts:
                # Gap to ground z=0
                gap = p[2] - 0.0
                if gap < 0.02:  # Active or near-contact
                    normal = np.array([0.0, 0.0, 1.0], dtype=float)
                    contacts.append({
                        "body_a": i,
                        "body_b": -1,  # Ground
                        "point": p.copy(),
                        "normal": normal,  # Points from ground into body_a
                        "gap": gap,
                    })

        # 2. Inter-block contacts between adjacent voussoirs (i and i+1)
        for i in range(self.num_bodies - 1):
            j = i + 1
            w_verts_i = self._get_world_vertices(i)
            w_verts_j = self._get_world_vertices(j)
            center_diff = self.positions[j] - self.positions[i]
            diff_norm = np.linalg.norm(center_diff)
            if diff_norm > 1e-6:
                normal_ij = center_diff / diff_norm  # From i to j
            else:
                normal_ij = np.array([1.0, 0.0, 0.0], dtype=float)

            # Check distances between vertices of i and vertices of j
            # Radial contact face points have minimal distance
            dists = np.linalg.norm(w_verts_i[:, None, :] - w_verts_j[None, :, :], axis=-1)
            min_dist_per_vi = np.min(dists, axis=1)
            min_dist_per_vj = np.min(dists, axis=0)

            threshold = 0.03
            for vi_idx, d in enumerate(min_dist_per_vi):
                if d < threshold:
                    vj_idx = np.argmin(dists[vi_idx, :])
                    p_mid = 0.5 * (w_verts_i[vi_idx] + w_verts_j[vj_idx])
                    gap = np.dot(w_verts_j[vj_idx] - w_verts_i[vi_idx], normal_ij)
                    contacts.append({
                        "body_a": j,
                        "body_b": i,
                        "point": p_mid,
                        "normal": normal_ij,  # Points from i into j
                        "gap": gap,
                    })

        return contacts

    def _physics_step(self, dt: float) -> tuple[list[dict], np.ndarray]:
        """Perform a single physics substep with impulse-based contact solver."""
        N = self.num_bodies
        ext_forces = np.tile(self.gravity, (N, 1)) * self.masses[:, None]

        # Semi-implicit Euler integration: apply gravity and velocity damping
        linear_damping = 0.999
        angular_damping = 0.999
        self.velocities += (ext_forces * self.inv_masses[:, None]) * dt
        self.velocities *= linear_damping
        self.angular_velocities *= angular_damping

        contacts = self._collect_contacts()

        # Iterative Projected Gauss-Seidel solver for contact impulses
        num_iters = 20
        # Precompute contact frames and effective mass
        contact_data = []
        for c in contacts:
            i = c["body_a"]
            j = c["body_b"]
            n = c["normal"]
            p = c["point"]
            r_a = p - self.positions[i]
            inv_m_a = self.inv_masses[i]
            inv_I_a = self._get_inv_inertia_world(i)

            if j >= 0:
                r_b = p - self.positions[j]
                inv_m_b = self.inv_masses[j]
                inv_I_b = self._get_inv_inertia_world(j)
            else:
                r_b = np.zeros(3)
                inv_m_b = 0.0
                inv_I_b = np.zeros((3, 3))

            # Tangent vectors
            if abs(n[0]) < 0.9:
                t1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
            else:
                t1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)

            # Normal effective mass
            K_n = inv_m_a + inv_m_b + np.dot(n, np.cross(inv_I_a @ np.cross(r_a, n), r_a))
            if j >= 0:
                K_n += np.dot(n, np.cross(inv_I_b @ np.cross(r_b, n), r_b))
            eff_mass_n = 1.0 / max(K_n, 1e-8)

            # Tangent 1 effective mass
            K_t1 = inv_m_a + inv_m_b + np.dot(t1, np.cross(inv_I_a @ np.cross(r_a, t1), r_a))
            if j >= 0:
                K_t1 += np.dot(t1, np.cross(inv_I_b @ np.cross(r_b, t1), r_b))
            eff_mass_t1 = 1.0 / max(K_t1, 1e-8)

            # Tangent 2 effective mass
            K_t2 = inv_m_a + inv_m_b + np.dot(t2, np.cross(inv_I_a @ np.cross(r_a, t2), r_a))
            if j >= 0:
                K_t2 += np.dot(t2, np.cross(inv_I_b @ np.cross(r_b, t2), r_b))
            eff_mass_t2 = 1.0 / max(K_t2, 1e-8)

            contact_data.append({
                "body_a": i,
                "body_b": j,
                "r_a": r_a,
                "r_b": r_b,
                "inv_m_a": inv_m_a,
                "inv_m_b": inv_m_b,
                "inv_I_a": inv_I_a,
                "inv_I_b": inv_I_b,
                "n": n,
                "t1": t1,
                "t2": t2,
                "gap": c["gap"],
                "eff_mass_n": eff_mass_n,
                "eff_mass_t1": eff_mass_t1,
                "eff_mass_t2": eff_mass_t2,
                "impulse_n": 0.0,
                "impulse_t1": 0.0,
                "impulse_t2": 0.0,
            })

        # Gauss-Seidel iterations
        baumgarte_beta = 0.2
        for _ in range(num_iters):
            for cd in contact_data:
                i = cd["body_a"]
                j = cd["body_b"]
                v_a = self.velocities[i] + np.cross(self.angular_velocities[i], cd["r_a"])
                if j >= 0:
                    v_b = self.velocities[j] + np.cross(self.angular_velocities[j], cd["r_b"])
                    v_rel = v_a - v_b
                else:
                    v_rel = v_a

                # Normal impulse with Baumgarte stabilization for penetration
                vn = np.dot(v_rel, cd["n"])
                bias = 0.0
                if cd["gap"] < 0.0:
                    bias = -(baumgarte_beta / dt) * cd["gap"]

                delta_impulse_n = cd["eff_mass_n"] * (-vn + bias)
                old_impulse_n = cd["impulse_n"]
                cd["impulse_n"] = max(0.0, old_impulse_n + delta_impulse_n)
                d_imp_n = cd["impulse_n"] - old_impulse_n

                # Apply normal impulse
                imp_vec_n = d_imp_n * cd["n"]
                self.velocities[i] += cd["inv_m_a"] * imp_vec_n
                self.angular_velocities[i] += cd["inv_I_a"] @ np.cross(cd["r_a"], imp_vec_n)
                if j >= 0:
                    self.velocities[j] -= cd["inv_m_b"] * imp_vec_n
                    self.angular_velocities[j] -= cd["inv_I_b"] @ np.cross(cd["r_b"], imp_vec_n)

                # Friction impulses (t1, t2)
                max_f = self.mu * cd["impulse_n"]

                # Update relative velocity
                v_a = self.velocities[i] + np.cross(self.angular_velocities[i], cd["r_a"])
                v_b = self.velocities[j] + np.cross(self.angular_velocities[j], cd["r_b"]) if j >= 0 else np.zeros(3)
                v_rel = v_a - v_b

                vt1 = np.dot(v_rel, cd["t1"])
                delta_imp_t1 = cd["eff_mass_t1"] * (-vt1)
                old_imp_t1 = cd["impulse_t1"]
                cd["impulse_t1"] = np.clip(old_imp_t1 + delta_imp_t1, -max_f, max_f)
                d_imp_t1 = cd["impulse_t1"] - old_imp_t1

                imp_vec_t1 = d_imp_t1 * cd["t1"]
                self.velocities[i] += cd["inv_m_a"] * imp_vec_t1
                self.angular_velocities[i] += cd["inv_I_a"] @ np.cross(cd["r_a"], imp_vec_t1)
                if j >= 0:
                    self.velocities[j] -= cd["inv_m_b"] * imp_vec_t1
                    self.angular_velocities[j] -= cd["inv_I_b"] @ np.cross(cd["r_b"], imp_vec_t1)

                v_a = self.velocities[i] + np.cross(self.angular_velocities[i], cd["r_a"])
                v_b = self.velocities[j] + np.cross(self.angular_velocities[j], cd["r_b"]) if j >= 0 else np.zeros(3)
                v_rel = v_a - v_b

                vt2 = np.dot(v_rel, cd["t2"])
                delta_imp_t2 = cd["eff_mass_t2"] * (-vt2)
                old_imp_t2 = cd["impulse_t2"]
                cd["impulse_t2"] = np.clip(old_imp_t2 + delta_imp_t2, -max_f, max_f)
                d_imp_t2 = cd["impulse_t2"] - old_imp_t2

                imp_vec_t2 = d_imp_t2 * cd["t2"]
                self.velocities[i] += cd["inv_m_a"] * imp_vec_t2
                self.angular_velocities[i] += cd["inv_I_a"] @ np.cross(cd["r_a"], imp_vec_t2)
                if j >= 0:
                    self.velocities[j] -= cd["inv_m_b"] * imp_vec_t2
                    self.angular_velocities[j] -= cd["inv_I_b"] @ np.cross(cd["r_b"], imp_vec_t2)

        # Integrate positions and orientations
        for i in range(N):
            self.positions[i] += self.velocities[i] * dt
            self.orientations[i] = quat_integrate(self.orientations[i], self.angular_velocities[i], dt)

        return contacts, ext_forces

    def advance_to(self, time: float) -> StateFrame:
        """Advance physical simulation to the target time and return StateFrame."""
        target_time = float(time)
        dt_target = target_time - self.current_time

        contacts = self._collect_contacts()
        ext_forces = np.tile(self.gravity, (self.num_bodies, 1)) * self.masses[:, None]

        if dt_target > 1e-8:
            n_substeps = max(1, int(np.ceil(dt_target / self.sub_dt)))
            actual_dt = dt_target / n_substeps
            for _ in range(n_substeps):
                contacts, ext_forces = self._physics_step(actual_dt)
            self.current_time = target_time

        # Compute trajectory contract fields
        # If no contacts, provide at least one dummy contact with zero/positive gap
        if len(contacts) > 0:
            contact_gaps = np.array([c["gap"] for c in contacts], dtype=float)
        else:
            contact_gaps = np.array([0.0], dtype=float)

        constraint_target = np.zeros_like(contact_gaps)
        constraint_value = np.maximum(0.0, -contact_gaps)  # Penetration depth

        fields = {
            "contact_signed_distance": contact_gaps,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "external_force": ext_forces,
        }

        scalars = {
            "constraint_scale": 1.0,
            "contact_length_scale": 0.4,  # Characteristic block size
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientations.copy(),
            angular_velocity=self.angular_velocities.copy(),
            mass=self.masses.copy(),
            body_id=self.body_ids.copy(),
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_bodies": self.num_bodies,
            "current_time": self.current_time,
        }

