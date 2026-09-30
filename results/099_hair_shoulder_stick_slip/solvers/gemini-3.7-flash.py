"""BenchmarkSolver implementation for wavy hair / shoulder contact stick-slip."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.dt = 0.002
        self.current_time = 0.0

        # Physical constants / config
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.static_friction = 0.3
        self.kinetic_friction = 0.3
        self.strand_radius = 0.0042
        self.stick_speed_tolerance = 1e-3

        # Hair topology
        self.strand_count = 84
        self.nodes_per_strand = 22
        self.num_nodes = 84 * 22  # 1848

        # State arrays
        self.pos = np.zeros((self.num_nodes, 3), dtype=np.float64)
        self.vel = np.zeros((self.num_nodes, 3), dtype=np.float64)
        self.mass = np.zeros(self.num_nodes, dtype=np.float64)
        self.inv_mass = np.zeros(self.num_nodes, dtype=np.float64)
        self.rest_pos = np.zeros((self.num_nodes, 3), dtype=np.float64)
        self.rest_seg_lens = np.zeros((self.strand_count, self.nodes_per_strand - 1), dtype=np.float64)
        self.is_root = np.zeros(self.num_nodes, dtype=bool)
        self.strand_id = np.zeros(self.num_nodes, dtype=np.int32)
        self.node_index = np.zeros(self.num_nodes, dtype=np.int32)

        # Head motion parameters
        self.head_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.head_disp = 0.18
        self.head_t0 = 0.45
        self.head_t1 = 2.35
        self.head_stop_time = 2.35

        # Cumulative budget integrals
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.last_sim_time = 0.0

        # Cached contact outputs for reporting
        self.active_contacts = 0
        self.contact_normals = np.zeros((0, 3), dtype=np.float64)
        self.contact_rel_vels = np.zeros((0, 3), dtype=np.float64)
        self.contact_normal_forces = np.zeros((0,), dtype=np.float64)
        self.contact_tangential_forces = np.zeros((0, 3), dtype=np.float64)
        self.contact_active_mask = np.zeros((0,), dtype=np.int32)
        self.contact_dyn_coef = np.zeros((0,), dtype=np.float64)
        self.contact_sta_coef = np.zeros((0,), dtype=np.float64)
        self.last_external_force = np.zeros((1, 1, 3), dtype=np.float64)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.rest_pos = np.array(init_frame.position, dtype=np.float64, copy=True)

        self.strand_id = np.repeat(np.arange(self.strand_count, dtype=np.int32), self.nodes_per_strand)
        self.node_index = np.tile(np.arange(self.nodes_per_strand, dtype=np.int32), self.strand_count)
        self.is_root = (self.node_index == 0)

        # Set masses & inverse masses
        self.inv_mass = np.zeros(self.num_nodes, dtype=np.float64)
        valid_mass = self.mass > 0
        self.inv_mass[valid_mass] = 1.0 / self.mass[valid_mass]
        self.inv_mass[self.is_root] = 0.0

        # Compute rest segment lengths
        for s in range(self.strand_count):
            idx_start = s * self.nodes_per_strand
            for j in range(self.nodes_per_strand - 1):
                p0 = self.rest_pos[idx_start + j]
                p1 = self.rest_pos[idx_start + j + 1]
                self.rest_seg_lens[s, j] = np.linalg.norm(p1 - p0)

        # Head motion params from config
        motion = scene.config.get("head_motion", {})
        self.head_axis = np.array(motion.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        norm_axis = np.linalg.norm(self.head_axis)
        if norm_axis > 1e-8:
            self.head_axis = self.head_axis / norm_axis
        self.head_disp = float(motion.get("displacement", 0.18))
        self.head_t0 = float(motion.get("start_time", 0.45))
        self.head_t1 = float(motion.get("stop_time", 2.35))
        self.head_stop_time = self.head_t1

        friction = scene.config.get("friction", {})
        self.static_friction = float(friction.get("hair_body", {}).get("static", 0.3))
        self.kinetic_friction = float(friction.get("hair_body", {}).get("kinetic", 0.3))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        self.current_time = 0.0
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.last_sim_time = 0.0

        # Perform initial collision/force evaluation
        self._step_simulation(0.0)

    def _kinematic_head_motion(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns head translation and velocity at time t."""
        if t <= self.head_t0:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        elif t >= self.head_t1:
            return self.head_axis * self.head_disp, np.zeros(3, dtype=np.float64)
        else:
            tau = (t - self.head_t0) / (self.head_t1 - self.head_t0)
            # Smooth 5th order polynomial (s-curve): s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            ds_dtau = 30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)
            ds_dt = ds_dtau / (self.head_t1 - self.head_t0)
            disp = self.head_axis * (self.head_disp * s)
            vel = self.head_axis * (self.head_disp * ds_dt)
            return disp, vel

    def _get_body_sdf(self, pos: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes signed distance, unit normal (outward), and surface velocity for scene collision geometries.

        Objects:
        1. Torso: capsule r=0.34, h=0.62, center=(0,0,0.62), scale=(1.0, 0.72, 1.0)
        2. Shoulder bar: capsule r=0.145, h=0.92, center=(0,0,1.02), along X axis
        3. Neck: capsule r=0.115, h=0.28, center=(0,0,1.28), along Z axis
        4. Head: sphere/ellipsoid r=0.265, center=(0,0,1.62), scale=(0.88, 0.95, 1.12)
        All geometries translate with the kinematic head motion.
        """
        disp, body_vel = self._kinematic_head_motion(t)
        # Transform points into local frame offset by head translation
        p = pos - disp  # [N, 3]

        # 1. Torso ellipsoid/capsule approximation
        # Center (0, 0, 0.62), half-height 0.31 along Z, elliptical cross-section rx=0.34, ry=0.34*0.72=0.2448
        # Segment from (0, 0, 0.465) to (0, 0, 0.775)
        # Scaled coordinate space: p_torso = [p[:,0]/1.0, p[:,1]/0.72, p[:,2]]
        p_t = np.stack([p[:, 0], p[:, 1] / 0.72, p[:, 2]], axis=-1)
        seg_a_t = np.array([0.0, 0.0, 0.465])
        seg_b_t = np.array([0.0, 0.0, 0.775])
        ab_t = seg_b_t - seg_a_t
        t_proj_t = np.clip(np.sum((p_t - seg_a_t) * ab_t, axis=1) / np.dot(ab_t, ab_t), 0.0, 1.0)
        closest_t = seg_a_t + t_proj_t[:, None] * ab_t
        diff_t = p_t - closest_t
        dist_t_scaled = np.linalg.norm(diff_t, axis=1)
        r_torso = 0.34
        sdf_t = dist_t_scaled - r_torso
        # Normal in world space
        norm_t_scaled = diff_t / np.maximum(dist_t_scaled[:, None], 1e-8)
        norm_t = np.stack([norm_t_scaled[:, 0], norm_t_scaled[:, 1] / 0.72, norm_t_scaled[:, 2]], axis=-1)
        norm_t = norm_t / np.maximum(np.linalg.norm(norm_t, axis=1, keepdims=True), 1e-8)

        # 2. Shoulder bar: capsule radius 0.145, length 0.92 along X axis (due to rot around Y by 90 deg)
        # Center: (0, 0, 1.02)
        # Segment along X: from (-0.46, 0, 1.02) to (0.46, 0, 1.02)
        seg_a_s = np.array([-0.46, 0.0, 1.02])
        seg_b_s = np.array([0.46, 0.0, 1.02])
        ab_s = seg_b_s - seg_a_s
        t_proj_s = np.clip(np.sum((p - seg_a_s) * ab_s, axis=1) / np.dot(ab_s, ab_s), 0.0, 1.0)
        closest_s = seg_a_s + t_proj_s[:, None] * ab_s
        diff_s = p - closest_s
        dist_s = np.linalg.norm(diff_s, axis=1)
        sdf_s = dist_s - 0.145
        norm_s = diff_s / np.maximum(dist_s[:, None], 1e-8)

        # 3. Neck: capsule radius 0.115, height 0.28 along Z
        # Center: (0, 0, 1.28), segment from (0, 0, 1.14) to (0, 0, 1.42)
        seg_a_n = np.array([0.0, 0.0, 1.14])
        seg_b_n = np.array([0.0, 0.0, 1.42])
        ab_n = seg_b_n - seg_a_n
        t_proj_n = np.clip(np.sum((p - seg_a_n) * ab_n, axis=1) / np.dot(ab_n, ab_n), 0.0, 1.0)
        closest_n = seg_a_n + t_proj_n[:, None] * ab_n
        diff_n = p - closest_n
        dist_n = np.linalg.norm(diff_n, axis=1)
        sdf_n = dist_n - 0.115
        norm_n = diff_n / np.maximum(dist_n[:, None], 1e-8)

        # 4. Head: sphere/ellipsoid radius 0.265, center (0, 0, 1.62), scale=(0.88, 0.95, 1.12)
        p_h = np.stack([p[:, 0] / 0.88, p[:, 1] / 0.95, (p[:, 2] - 1.62) / 1.12], axis=-1)
        dist_h_scaled = np.linalg.norm(p_h, axis=1)
        sdf_h = dist_h_scaled - 0.265
        norm_h_scaled = p_h / np.maximum(dist_h_scaled[:, None], 1e-8)
        norm_h = np.stack([norm_h_scaled[:, 0] / 0.88, norm_h_scaled[:, 1] / 0.95, norm_h_scaled[:, 2] / 1.12], axis=-1)
        norm_h = norm_h / np.maximum(np.linalg.norm(norm_h, axis=1, keepdims=True), 1e-8)

        # Combine shapes by smooth minimum or exact minimum
        all_sdfs = np.stack([sdf_t, sdf_s, sdf_n, sdf_h], axis=1)  # [N, 4]
        min_idx = np.argmin(all_sdfs, axis=1)

        min_sdf = np.zeros(len(p), dtype=np.float64)
        normals = np.zeros((len(p), 3), dtype=np.float64)

        all_normals = [norm_t, norm_s, norm_n, norm_h]
        for k in range(4):
            mask = (min_idx == k)
            if np.any(mask):
                min_sdf[mask] = all_sdfs[mask, k]
                normals[mask] = all_normals[k][mask]

        # Body velocity at contact points
        body_velocities = np.tile(body_vel, (len(p), 1))

        return min_sdf, normals, body_velocities

    def _step_simulation(self, target_time: float) -> None:
        substep_dt = 0.001
        num_substeps = max(1, int(np.ceil((target_time - self.current_time) / substep_dt))) if target_time > self.current_time else 1
        dt = (target_time - self.current_time) / num_substeps if target_time > self.current_time else substep_dt

        for _ in range(num_substeps):
            t = self.current_time
            # Update kinematic root positions
            disp, root_vel = self._kinematic_head_motion(t + dt)
            self.pos[self.is_root] = self.rest_pos[self.is_root] + disp
            self.vel[self.is_root] = root_vel

            # 1. External forces: Gravity and Aerodynamic drag
            non_root = ~self.is_root
            ext_force = np.zeros_like(self.pos)
            ext_force[non_root] = self.mass[non_root, None] * self.gravity[None, :]

            # Velocity damping
            linear_damping = 0.8  # 1/s
            self.vel[non_root] += (ext_force[non_root] * self.inv_mass[non_root, None] - linear_damping * self.vel[non_root]) * dt

            # Predict positions
            pred_pos = self.pos.copy()
            pred_pos[non_root] += self.vel[non_root] * dt

            # 2. Position Based Dynamics (PBD) Constraint Solver iterations
            pbd_iterations = 8
            for it in range(pbd_iterations):
                # 2a. Distance constraints along each strand
                for j in range(self.nodes_per_strand - 1):
                    idx0 = np.arange(0, self.num_nodes, self.nodes_per_strand) + j
                    idx1 = idx0 + 1

                    p0 = pred_pos[idx0]
                    p1 = pred_pos[idx1]
                    delta = p1 - p0
                    dist = np.linalg.norm(delta, axis=1, keepdims=True)
                    dist = np.maximum(dist, 1e-9)

                    s_indices = np.arange(self.strand_count)
                    rest_d = self.rest_seg_lens[s_indices, j, None]

                    diff = (dist - rest_d) / dist
                    w0 = self.inv_mass[idx0, None]
                    w1 = self.inv_mass[idx1, None]
                    w_sum = w0 + w1
                    w_sum = np.maximum(w_sum, 1e-9)

                    corr = diff * delta
                    pred_pos[idx0] += (w0 / w_sum) * corr * 0.95
                    pred_pos[idx1] -= (w1 / w_sum) * corr * 0.95

                # 2b. Bending / curvature & Rest-shape preservation constraints
                # Keep wavy rest shape
                for j in range(1, self.nodes_per_strand - 1):
                    idx_prev = np.arange(0, self.num_nodes, self.nodes_per_strand) + (j - 1)
                    idx_curr = idx_prev + 1
                    idx_next = idx_curr + 1

                    p_prev = pred_pos[idx_prev]
                    p_curr = pred_pos[idx_curr]
                    p_next = pred_pos[idx_next]

                    # Target midpoint
                    mid = 0.5 * (p_prev + p_next)
                    # From rest shape, compute rest displacement from midpoint
                    rest_disp = self.rest_pos[idx_curr] - 0.5 * (self.rest_pos[idx_prev] + self.rest_pos[idx_next])
                    target_p = mid + rest_disp

                    w_c = self.inv_mass[idx_curr, None]
                    bend_stiffness = 0.15
                    pred_pos[idx_curr] += (target_p - p_curr) * bend_stiffness * (w_c > 0)

                # 2c. Collision constraints with body
                body_sdf, body_norm, _ = self._get_body_sdf(pred_pos, t + dt)
                penetration = self.strand_radius - body_sdf
                penetration_mask = (penetration > 0.0) & non_root

                if np.any(penetration_mask):
                    pred_pos[penetration_mask] += body_norm[penetration_mask] * penetration[penetration_mask, None]

                # 2d. Hair-hair repulsion / self-collision
                # Spatial hashing / wisp-layer repulsion
                # Strands in similar azimuth rings repel each other when distance < 2 * strand_radius
                # Efficient approximate repulsion using neighbor strands
                for stride in [1, 2, 14]:
                    for s in range(self.strand_count):
                        s_other = (s + stride) % self.strand_count
                        idx_a = np.arange(s * self.nodes_per_strand + 1, (s + 1) * self.nodes_per_strand)
                        idx_b = np.arange(s_other * self.nodes_per_strand + 1, (s_other + 1) * self.nodes_per_strand)
                        pa = pred_pos[idx_a]
                        pb = pred_pos[idx_b]
                        d_ab = pa - pb
                        dist_sq = np.sum(d_ab**2, axis=1)
                        coll_thresh = (2.0 * self.strand_radius * 1.5)**2
                        coll_mask = dist_sq < coll_thresh
                        if np.any(coll_mask):
                            sub_a = idx_a[coll_mask]
                            sub_b = idx_b[coll_mask]
                            dists = np.sqrt(dist_sq[coll_mask])[:, None]
                            n_ab = d_ab[coll_mask] / np.maximum(dists, 1e-8)
                            delta_p = 0.5 * (2.0 * self.strand_radius * 1.5 - dists) * n_ab
                            pred_pos[sub_a] += delta_p * 0.5
                            pred_pos[sub_b] -= delta_p * 0.5

            # 3. Velocity update and Coulomb friction application
            body_sdf, body_norm, body_vel = self._get_body_sdf(pred_pos, t + dt)
            contact_dist = self.strand_radius - body_sdf
            in_contact = (contact_dist >= -1e-4) & non_root

            new_vel = (pred_pos - self.pos) / dt
            new_vel[self.is_root] = root_vel

            # Compute contact forces and Coulomb dry friction (stick-slip)
            num_contacts = int(np.sum(in_contact))
            if num_contacts > 0:
                contact_indices = np.where(in_contact)[0]
                n = body_norm[contact_indices]
                v_rel = new_vel[contact_indices] - body_vel[contact_indices]

                # Normal velocity component
                vn = np.sum(v_rel * n, axis=1, keepdims=True)
                # Tangential velocity component
                vt = v_rel - vn * n
                vt_norm = np.linalg.norm(vt, axis=1, keepdims=True)

                # Compressive normal force: mass * (contact penetration / dt^2 + normal acceleration)
                pen = np.maximum(contact_dist[contact_indices], 0.0)
                fn_val = self.mass[contact_indices, None] * (pen[:, None] / (dt * dt) + np.maximum(-vn / dt, 0.0) + 9.81)
                fn_mag = np.squeeze(fn_val, axis=1)

                # Coulomb friction limit: max tangential force = mu * Fn
                mu = self.static_friction
                ft_max = mu * fn_mag

                # Tangential force required for sticking
                ft_stick = self.mass[contact_indices, None] * (vt / dt)
                ft_stick_mag = np.linalg.norm(ft_stick, axis=1)

                # Stick / Slip check
                slip_mask = ft_stick_mag > ft_max
                ft = np.zeros_like(vt)

                # Sticking: cancel tangential relative velocity
                if np.any(~slip_mask):
                    stick_idx = np.where(~slip_mask)[0]
                    ft[stick_idx] = ft_stick[stick_idx]
                    v_rel[stick_idx] -= vt[stick_idx]  # vt becomes 0

                # Slipping: tangential force in direction opposite to sliding
                if np.any(slip_mask):
                    slip_idx = np.where(slip_mask)[0]
                    tang_dir = vt[slip_idx] / np.maximum(vt_norm[slip_idx], 1e-8)
                    ft[slip_idx] = ft_max[slip_idx, None] * tang_dir
                    # Sliding velocity reduction
                    dv_tang = (ft_max[slip_idx, None] / self.mass[contact_indices[slip_idx], None]) * dt
                    reduction = np.minimum(dv_tang, vt_norm[slip_idx])
                    v_rel[slip_idx] -= reduction * tang_dir

                # Update new_vel from adjusted v_rel
                new_vel[contact_indices] = body_vel[contact_indices] + v_rel

                # Store friction fields for StateFrame
                self.active_contacts = num_contacts
                self.contact_normals = n
                self.contact_rel_vels = v_rel
                self.contact_normal_forces = fn_mag
                self.contact_tangential_forces = ft
                self.contact_active_mask = np.ones(num_contacts, dtype=np.int32)
                self.contact_dyn_coef = np.full(num_contacts, self.kinetic_friction, dtype=np.float64)
                self.contact_sta_coef = np.full(num_contacts, self.static_friction, dtype=np.float64)
            else:
                self.active_contacts = 0
                self.contact_normals = np.zeros((0, 3), dtype=np.float64)
                self.contact_rel_vels = np.zeros((0, 3), dtype=np.float64)
                self.contact_normal_forces = np.zeros((0,), dtype=np.float64)
                self.contact_tangential_forces = np.zeros((0, 3), dtype=np.float64)
                self.contact_active_mask = np.zeros((0,), dtype=np.int32)
                self.contact_dyn_coef = np.zeros((0,), dtype=np.float64)
                self.contact_sta_coef = np.zeros((0,), dtype=np.float64)

            # Update state
            self.pos = pred_pos
            self.vel = new_vel

            # Cumulative external impulse tracking
            total_ext_force = np.sum(ext_force, axis=0, keepdims=True)  # [1, 3]
            self.cum_external_impulse += total_ext_force * dt
            self.last_external_force = total_ext_force.reshape(1, 1, 3)

            self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        if time > self.current_time:
            self._step_simulation(time)

        # Budget weights for 1 global domain
        domain_weights = np.ones((1, self.num_nodes), dtype=np.float64)

        # Reference scales
        total_mass = float(np.sum(self.mass))
        ref_velocity = 0.35
        ref_momentum = total_mass * ref_velocity
        ref_force = total_mass * 9.81

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.strand_id.copy(),
            attributes={
                "strand_id": self.strand_id.copy(),
                "node_index": self.node_index.copy(),
                "is_root": self.is_root.astype(np.int32),
                "kinematic_root": self.is_root.astype(np.int32),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": domain_weights,
                "budget_external_force": self.last_external_force.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "friction_active_mask": self.contact_active_mask.copy(),
                "friction_dynamic_coefficient": self.contact_dyn_coef.copy(),
                "friction_normal": self.contact_normals.copy(),
                "friction_normal_force": self.contact_normal_forces.copy(),
                "friction_relative_velocity": self.contact_rel_vels.copy(),
                "friction_static_coefficient": self.contact_sta_coef.copy(),
                "friction_tangential_force": self.contact_tangential_forces.copy(),
            },
            scalars={
                "strand_radius": self.strand_radius,
                "static_friction": self.static_friction,
                "kinetic_friction": self.kinetic_friction,
                "head_stop_time": self.head_stop_time,
                "friction_stick_speed_tolerance": self.stick_speed_tolerance,
                "reference_force": ref_force,
                "reference_momentum": ref_momentum,
                "reference_velocity": ref_velocity,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "active_contacts": self.active_contacts,
        }

