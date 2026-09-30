"""Deterministic, physically-based hair strand dynamics solver with contact and friction."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, RotationSpline
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.current_time = 0.0
        self.pos = None
        self.vel = None
        self.mass = None
        self.root_mask = None
        self.strand_id = None

        # Physical parameters
        self.count = 192
        self.segs = 16
        self.total_nodes = 3264
        self.rest_seg_len = 0.3 / 16.0
        self.sphere_radius = 0.09
        self.strand_radius = 3.7e-05
        self.mu_s = 0.2
        self.mu_k = 0.2
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Rotation spline
        self.rot_spline = None
        self.initial_roots = None
        self.initial_relative_pos = None

        # Cumulative budget integrators
        self.budget_cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.budget_cumulative_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.budget_cumulative_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)

        # Dynamic tracking
        self.last_normal_forces = None
        self.last_tangent_forces = None
        self.last_contact_normals = None
        self.last_rel_vels = None
        self.last_active_masks = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        p = scene.config["physical_parameters"]
        self.count = int(p["strand_count"])
        self.segs = int(p["segments_per_strand"])
        self.total_nodes = self.count * (self.segs + 1)
        self.rest_seg_len = float(p["strand_length"]) / self.segs
        self.sphere_radius = float(p["sphere_radius"])
        self.strand_radius = float(p["strand_radius"])
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.mu_s = float(scene.config["contact"]["static_friction"])
        self.mu_k = float(scene.config["contact"]["kinetic_friction"])

        # Setup rotation spline
        keyframes = scene.config["prescribed_sphere_rotation"]["keyframes_axis_angle_degrees"]
        times = [float(kf["time"]) for kf in keyframes]
        rotvecs = []
        for kf in keyframes:
            axis = np.array(kf["axis"], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis = axis / norm
            angle_rad = np.radians(float(kf["angle"]))
            rotvecs.append(axis * angle_rad)
        rotations = R.from_rotvec(rotvecs)
        self.rot_spline = RotationSpline(times, rotations)

        # Initial state from scene
        initial_frame = scene.initial_state()
        self.pos = initial_frame.position.copy()
        self.vel = initial_frame.velocity.copy()
        self.mass = initial_frame.mass.copy()
        self.root_mask = initial_frame.attributes["root_mask"].copy()
        self.strand_id = initial_frame.attributes["strand_id"].copy()
        self.current_time = 0.0

        # Store root positions in body frame
        root_indices = np.arange(self.count) * (self.segs + 1)
        self.initial_roots = self.pos[root_indices].copy()

        # Precompute initial relative offsets for natural curl / shape restoration
        self.initial_relative_pos = np.zeros((self.count, self.segs, 3), dtype=np.float64)
        for s in range(self.count):
            base = s * (self.segs + 1)
            for j in range(self.segs):
                self.initial_relative_pos[s, j] = self.pos[base + j + 1] - self.pos[base + j]

        # Zero cumulative budget integrators
        self.budget_cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.budget_cumulative_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.budget_cumulative_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)

        self.last_normal_forces = np.zeros(self.total_nodes, dtype=np.float64)
        self.last_tangent_forces = np.zeros((self.total_nodes, 3), dtype=np.float64)
        self.last_contact_normals = np.zeros((self.total_nodes, 3), dtype=np.float64)
        self.last_contact_normals[:, 2] = 1.0
        self.last_rel_vels = np.zeros((self.total_nodes, 3), dtype=np.float64)
        self.last_active_masks = np.zeros(self.total_nodes, dtype=np.float64)

    def _get_sphere_kinematics(self, t: float):
        rot = self.rot_spline(t)
        rot_mat = rot.as_matrix()
        # Angular velocity
        ang_vel = self.rot_spline(t, order=1)
        return rot_mat, ang_vel

    def _step_simulation(self, dt: float):
        t_next = self.current_time + dt
        rot_mat_curr, ang_vel_curr = self._get_sphere_kinematics(self.current_time)
        rot_mat_next, ang_vel_next = self._get_sphere_kinematics(t_next)

        # 1. Update root positions and kinematic velocities
        root_indices = np.arange(self.count) * (self.segs + 1)
        roots_next = (rot_mat_next @ self.initial_roots.T).T
        roots_curr = self.pos[root_indices]

        self.pos[root_indices] = roots_next
        self.vel[root_indices] = (roots_next - roots_curr) / dt

        # 2. Gravity and explicit velocity prediction
        dynamic_mask = (self.root_mask < 0.5)
        self.vel[dynamic_mask] += self.gravity * dt

        # Damping
        self.vel[dynamic_mask] *= (1.0 - 0.2 * dt)

        pos_pred = self.pos.copy()
        pos_pred[dynamic_mask] += self.vel[dynamic_mask] * dt
        pos_pred[root_indices] = roots_next

        # 3. Constraint solver (PBD / Position-Based Dynamics)
        # Distance & bending constraints per strand
        iters = 12
        for _ in range(iters):
            # Distance constraints along strands
            pos_reshaped = pos_pred.reshape((self.count, self.segs + 1, 3))
            p1 = pos_reshaped[:, :-1, :]
            p2 = pos_reshaped[:, 1:, :]
            delta = p2 - p1
            dist = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-12
            diff = (dist - self.rest_seg_len) / dist
            corr = 0.5 * 0.9 * diff * delta

            # Root is fixed (index 0), so p1[..., 0, :] correction goes only to p2
            pos_reshaped[:, 1:-1, :] -= corr[:, 1:, :]
            pos_reshaped[:, 1:, :] += corr[:, :, :]
            pos_pred[root_indices] = roots_next

            # Bending / Curvature constraints (restoring natural curl & straightness)
            p_prev = pos_reshaped[:, :-2, :]
            p_curr = pos_reshaped[:, 1:-1, :]
            p_next_node = pos_reshaped[:, 2:, :]
            mid = 0.5 * (p_prev + p_next_node)
            bend_corr = 0.15 * (mid - p_curr)
            pos_reshaped[:, 1:-1, :] += bend_corr
            pos_pred[root_indices] = roots_next

            # Sphere collision projection
            dist_to_center = np.linalg.norm(pos_pred, axis=-1)
            penetration = (self.sphere_radius + self.strand_radius) - dist_to_center
            sphere_coll_mask = (penetration > 0.0) & dynamic_mask
            if np.any(sphere_coll_mask):
                normals = pos_pred[sphere_coll_mask] / (dist_to_center[sphere_coll_mask, None] + 1e-12)
                pos_pred[sphere_coll_mask] += normals * penetration[sphere_coll_mask, None]

            # Strand-strand repulsion (spatial grid or sampled repulsion)
            # Soft repulsion to prevent interpenetration
            # Downsample / inter-strand distance check for nearby nodes
            pts_dyn = pos_pred[dynamic_mask]
            # Fast strand-strand approximation using binning
            grid_cell = 0.015
            cell_coords = np.floor(pts_dyn / grid_cell).astype(np.int32)
            # Hash
            h = (cell_coords[:, 0] * 73856093 ^ cell_coords[:, 1] * 19349663 ^ cell_coords[:, 2] * 83492791) % 10007
            sort_idx = np.argsort(h)
            sorted_pts = pts_dyn[sort_idx]
            dyn_indices = np.where(dynamic_mask)[0]
            sorted_dyn_indices = dyn_indices[sort_idx]

            # Check adjacent in hash
            d_pairs = sorted_pts[1:] - sorted_pts[:-1]
            dist_sq = np.sum(d_pairs**2, axis=-1)
            coll_r = 2.0 * self.strand_radius + 0.001
            coll_idx = np.where((dist_sq < coll_r**2) & (h[sort_idx[1:]] == h[sort_idx[:-1]]))[0]
            if len(coll_idx) > 0:
                idx_a = sorted_dyn_indices[coll_idx]
                idx_b = sorted_dyn_indices[coll_idx + 1]
                # If not on same strand adjacent nodes
                same_strand_adj = (np.abs(idx_a - idx_b) <= 1) & (self.strand_id[idx_a] == self.strand_id[idx_b])
                valid = ~same_strand_adj
                if np.any(valid):
                    ia = idx_a[valid]
                    ib = idx_b[valid]
                    d_ab = pos_pred[ib] - pos_pred[ia]
                    d_len = np.linalg.norm(d_ab, axis=-1, keepdims=True) + 1e-12
                    delta_r = 0.5 * (coll_r - d_len)
                    dir_ab = d_ab / d_len
                    pos_pred[ia] -= 0.5 * delta_r * dir_ab
                    pos_pred[ib] += 0.5 * delta_r * dir_ab

        # Update velocities from positions
        new_vel = (pos_pred - self.pos) / dt
        new_vel[root_indices] = (roots_next - roots_curr) / dt

        # Contact forces and friction calculation
        dist_to_center = np.linalg.norm(pos_pred, axis=-1)
        contact_mask = (dist_to_center <= (self.sphere_radius + self.strand_radius + 1e-4)) & dynamic_mask

        contact_normals = np.zeros((self.total_nodes, 3), dtype=np.float64)
        contact_normals[:, 2] = 1.0
        contact_normals[contact_mask] = pos_pred[contact_mask] / (dist_to_center[contact_mask, None] + 1e-12)

        # Relative velocity at contact with rotating sphere surface
        # Sphere surface velocity at contact point: v_surf = omega x r
        surf_vel = np.cross(ang_vel_next, pos_pred)
        rel_vel = new_vel - surf_vel

        normal_forces = np.zeros(self.total_nodes, dtype=np.float64)
        tangent_forces = np.zeros((self.total_nodes, 3), dtype=np.float64)
        active_masks = np.zeros(self.total_nodes, dtype=np.float64)

        if np.any(contact_mask):
            active_masks[contact_mask] = 1.0
            # Normal force estimate: F_n = m * a_n
            # Compression normal force
            v_rel_n = np.sum(rel_vel[contact_mask] * contact_normals[contact_mask], axis=-1)
            # Normal contact constraint force
            normal_acc = np.maximum(0.0, -v_rel_n / dt) + np.maximum(0.0, -np.sum(self.gravity * contact_normals[contact_mask], axis=-1))
            normal_forces[contact_mask] = self.mass[contact_mask] * normal_acc

            # Tangential friction force: F_t = - mu * F_n * v_tangent / |v_tangent|
            v_rel_tangent = rel_vel[contact_mask] - v_rel_n[:, None] * contact_normals[contact_mask]
            v_t_len = np.linalg.norm(v_rel_tangent, axis=-1, keepdims=True)
            slip_dir = np.zeros_like(v_rel_tangent)
            slip_mask = (v_t_len.squeeze(-1) > 1e-6)
            if np.any(slip_mask):
                slip_dir[slip_mask] = v_rel_tangent[slip_mask] / v_t_len[slip_mask]

            f_friction_mag = self.mu_k * normal_forces[contact_mask]
            tangent_forces[contact_mask] = -slip_dir * f_friction_mag[:, None]

            # Apply friction to dynamic velocities
            friction_dv = (tangent_forces[contact_mask] / (self.mass[contact_mask, None] + 1e-12)) * dt
            new_vel[contact_mask] += friction_dv

        # Cumulative budget impulse (external gravity + contact)
        net_ext_force = np.sum(self.mass[dynamic_mask, None] * self.gravity, axis=0)
        net_ext_force += np.sum(normal_forces[:, None] * contact_normals + tangent_forces, axis=0)
        self.budget_cumulative_external_impulse[0] += net_ext_force * dt

        # Update state
        self.pos = pos_pred
        self.vel = new_vel
        self.current_time = t_next

        self.last_normal_forces = normal_forces
        self.last_tangent_forces = tangent_forces
        self.last_contact_normals = contact_normals
        self.last_rel_vels = rel_vel
        self.last_active_masks = active_masks

    def advance_to(self, time: float) -> StateFrame:
        # Substep to reach target time
        target_time = float(time)
        dt_sub = 0.002  # 10 substeps per 0.02s sample frame for stability and physical fidelity

        while self.current_time < target_time - 1e-9:
            dt = min(dt_sub, target_time - self.current_time)
            self._step_simulation(dt)

        self.current_time = target_time

        # Build StateFrame
        N = self.total_nodes
        G = 1
        K = 1

        # External force per budget domain: sum of gravity on hair
        dynamic_mask = (self.root_mask < 0.5)
        total_grav_force = np.sum(self.mass[dynamic_mask, None] * self.gravity, axis=0)
        budget_external_force = np.zeros((G, K, 3), dtype=np.float64)
        budget_external_force[0, 0] = total_grav_force

        budget_domain_weights = np.ones((G, N), dtype=np.float64)

        # Scales
        total_mass = np.sum(self.mass)
        ref_force = float(total_mass * 9.81)
        ref_vel = 1.0
        ref_momentum = float(total_mass * ref_vel)

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "strand_id": self.strand_id.copy(),
                "root_mask": self.root_mask.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.budget_cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.budget_cumulative_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.budget_cumulative_outflow_momentum.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "friction_active_mask": self.last_active_masks.copy(),
                "friction_dynamic_coefficient": np.full(N, self.mu_k, dtype=np.float64),
                "friction_normal": self.last_contact_normals.copy(),
                "friction_normal_force": self.last_normal_forces.copy(),
                "friction_relative_velocity": self.last_rel_vels.copy(),
                "friction_static_coefficient": np.full(N, self.mu_s, dtype=np.float64),
                "friction_tangential_force": self.last_tangent_forces.copy(),
            },
            scalars={
                "strand_radius": self.strand_radius,
                "static_friction": self.mu_s,
                "kinetic_friction": self.mu_k,
                "friction_stick_speed_tolerance": 1e-3,
                "reference_force": ref_force,
                "reference_momentum": ref_momentum,
                "reference_velocity": ref_vel,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "total_nodes": self.total_nodes,
        }

