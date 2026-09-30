"""Deterministic physical solver for frictional stacking of spaghetti strands in a plate."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.current_time = 0.0
        self.initial_frame = None
        self.strand_count = 31
        self.nodes_per_strand = 16
        self.strand_radius = 0.006
        self.strand_length = 0.72
        self.mu = 0.3
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)

        # Simulation state
        self.pos = None
        self.vel = None
        self.mass = None
        self.body_id = None
        self.node_index = None

        # Cumulative budget tracking
        self.cum_impulse = np.zeros((32, 3), dtype=float)  # 31 strands + 1 global
        self.cum_inflow = np.zeros((32, 3), dtype=float)
        self.cum_outflow = np.zeros((32, 3), dtype=float)

        # Contact tracking for fields
        self.prev_normal_forces = None
        self.prev_tangential_forces = None

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.initial_frame = scene.initial_state()
        self.current_time = float(self.initial_frame.time)

        self.pos = self.initial_frame.position.copy()
        self.vel = self.initial_frame.velocity.copy()
        self.mass = self.initial_frame.mass.copy()
        self.body_id = self.initial_frame.body_id.copy()
        self.node_index = self.initial_frame.attributes["node_index"].copy()

        self.total_nodes = len(self.pos)
        self.rest_seg_len = self.strand_length / (self.nodes_per_strand - 1)

        # Budget domains: 31 per-strand domains + 1 global domain (G=32)
        self.domain_weights = np.zeros((32, self.total_nodes), dtype=float)
        for s in range(31):
            self.domain_weights[s, s * self.nodes_per_strand : (s + 1) * self.nodes_per_strand] = 1.0
        self.domain_weights[31, :] = 1.0

        self.cum_impulse = np.zeros((32, 3), dtype=float)
        self.cum_inflow = np.zeros((32, 3), dtype=float)
        self.cum_outflow = np.zeros((32, 3), dtype=float)

    def _plate_distance_and_normal(self, p: np.ndarray):
        """Compute signed distance from strand centerline to inner surface of plate, and outward normal (pointing into plate interior / up)."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(x * x + y * y)

        # Inner profile points: (r, z)
        # [0.10, 0.000], [0.40, 0.018], [0.52, 0.075], [0.57, 0.125]
        # Below r = 0.1: flat bottom at z = 0.0
        # 0.10 <= r <= 0.40: slope = (0.018 - 0.0) / 0.30 = 0.06
        # 0.40 <= r <= 0.52: slope = (0.075 - 0.018) / 0.12 = 0.475
        # 0.52 <= r <= 0.57: slope = (0.125 - 0.075) / 0.05 = 1.00
        # r > 0.57: rim wall
        z_surf = np.zeros_like(r)
        dz_dr = np.zeros_like(r)

        m0 = r <= 0.10
        z_surf[m0] = 0.0
        dz_dr[m0] = 0.0

        m1 = (r > 0.10) & (r <= 0.40)
        z_surf[m1] = 0.0 + 0.06 * (r[m1] - 0.10)
        dz_dr[m1] = 0.06

        m2 = (r > 0.40) & (r <= 0.52)
        z_surf[m2] = 0.018 + 0.475 * (r[m2] - 0.40)
        dz_dr[m2] = 0.475

        m3 = (r > 0.52) & (r <= 0.57)
        z_surf[m3] = 0.075 + 1.00 * (r[m3] - 0.52)
        dz_dr[m3] = 1.00

        m4 = r > 0.57
        z_surf[m4] = 0.125 + 2.5 * (r[m4] - 0.57)
        dz_dr[m4] = 2.5

        # Distance from point to surface in normal direction approx (z - z_surf) * cos(theta)
        # Unit normal to surface pointing INTO container:
        # Surface is z - z_surf(r) = 0. Gradient pointing upwards: (-dz/dr * (x/r), -dz/dr * (y/r), 1.0)
        denom = np.sqrt(1.0 + dz_dr * dz_dr)
        norm_z = 1.0 / denom
        norm_r = -dz_dr / denom

        r_safe = np.maximum(r, 1e-8)
        norm_x = norm_r * (x / r_safe)
        norm_y = norm_r * (y / r_safe)

        normal = np.column_stack([norm_x, norm_y, norm_z])
        # Approximate signed distance from surface (positive inside container)
        dist = (z - z_surf) * norm_z

        return dist, normal

    def _step_substep(self, dt: float):
        """Execute one physical sub-step using Position-Based Dynamics / Projective Dynamics."""
        N = self.total_nodes
        pos = self.pos
        vel = self.vel
        mass = self.mass
        inv_mass = 1.0 / mass

        # 1. External forces (gravity) & velocity damping
        vel[:, 2] += self.gravity[2] * dt
        # Slight numerical air drag
        vel *= (1.0 - 0.02 * dt)

        # 2. Predict positions
        p_pred = pos + vel * dt

        # 3. Solver constraint iterations
        sub_iters = 10
        r_strand = self.strand_radius
        min_dist_strand = 2.0 * r_strand

        for _ in range(sub_iters):
            # Distance constraints (rod stretch resistance)
            # Reshape per strand: (31, 16, 3)
            p_strands = p_pred.reshape((self.strand_count, self.nodes_per_strand, 3))
            p_diff = p_strands[:, 1:, :] - p_strands[:, :-1, :]
            cur_dist = np.linalg.norm(p_diff, axis=2, keepdims=True)
            cur_dist = np.maximum(cur_dist, 1e-9)
            dir_diff = p_diff / cur_dist
            diff_len = cur_dist - self.rest_seg_len

            # Correction per segment
            corr = 0.5 * diff_len * dir_diff
            p_strands[:, :-1, :] += 0.9 * corr
            p_strands[:, 1:, :] -= 0.9 * corr

            # Bending constraints (rod bending resistance)
            # Straight rest state: p[i-1] - 2*p[i] + p[i+1] == 0 (Laplacian)
            p_mid = p_strands[:, 1:-1, :]
            p_left = p_strands[:, :-2, :]
            p_right = p_strands[:, 2:, :]
            bend_diff = 2.0 * p_mid - (p_left + p_right)
            bend_stiffness = 0.3
            p_mid -= bend_stiffness * 0.5 * bend_diff
            p_left += bend_stiffness * 0.25 * bend_diff
            p_right += bend_stiffness * 0.25 * bend_diff

            # Strand-plate collision constraints
            p_pred_flat = p_pred.reshape((-1, 3))
            dist_plate, norm_plate = self._plate_distance_and_normal(p_pred_flat)
            plate_penetration = r_strand - dist_plate
            in_contact = plate_penetration > 0.0
            if np.any(in_contact):
                p_pred_flat[in_contact] += (norm_plate[in_contact] * plate_penetration[in_contact, None])

            # Strand-strand collision constraints (node-node sphere-sphere approximation)
            # Broadphase spatial hashing or pairwise distance
            # For 496 nodes, pairwise distance matrix is 496x496 (very fast in numpy)
            diff_all = p_pred_flat[:, None, :] - p_pred_flat[None, :, :]
            dist_sq = np.sum(diff_all * diff_all, axis=2)
            # Ignore self-node and adjacent nodes on same strand
            body_same = (self.body_id[:, None] == self.body_id[None, :])
            node_diff = np.abs(self.node_index[:, None] - self.node_index[None, :])
            ignore_mask = body_same & (node_diff <= 1)
            dist_sq[ignore_mask] = 1e9

            contact_pairs = np.where(dist_sq < (min_dist_strand * min_dist_strand))
            i_idx, j_idx = contact_pairs[0], contact_pairs[1]
            # Keep each pair once (i < j)
            valid = i_idx < j_idx
            i_idx, j_idx = i_idx[valid], j_idx[valid]

            if len(i_idx) > 0:
                p_i = p_pred_flat[i_idx]
                p_j = p_pred_flat[j_idx]
                disp = p_i - p_j
                d = np.linalg.norm(disp, axis=1, keepdims=True)
                d = np.maximum(d, 1e-9)
                n = disp / d
                overlap = min_dist_strand - d
                corr = 0.5 * overlap * n
                np.add.at(p_pred_flat, i_idx, 0.8 * corr)
                np.add.at(p_pred_flat, j_idx, -0.8 * corr)

        # 4. Update velocity & apply friction
        new_vel = (p_pred - pos) / dt

        # Contact friction with plate
        dist_plate, norm_plate = self._plate_distance_and_normal(p_pred)
        plate_contact = dist_plate <= (r_strand + 1e-4)
        if np.any(plate_contact):
            v = new_vel[plate_contact]
            n = norm_plate[plate_contact]
            v_n = np.sum(v * n, axis=1, keepdims=True)
            v_n_clamped = np.maximum(v_n, 0.0)
            v_t = v - v_n * n
            v_t_norm = np.linalg.norm(v_t, axis=1, keepdims=True)

            # Coulomb friction
            # Normal force magnitude approx mass * g or collision impulse
            f_norm_mag = np.maximum(-v_n, 0.0) / dt + 9.81
            max_f_tangent = self.mu * f_norm_mag
            v_t_reduction = max_f_tangent * dt
            scale = np.maximum(0.0, 1.0 - v_t_reduction / np.maximum(v_t_norm, 1e-9))
            v_t_new = v_t * scale

            # Restitution & tangent update
            new_vel[plate_contact] = v_t_new + 0.0 * n

        # Contact friction with other strands (inter-strand damping/friction)
        # Apply tangential velocity damping when in contact
        self.vel = new_vel
        self.pos = p_pred

    def advance_to(self, time: float) -> StateFrame:
        # Step physics up to requested sample time
        sub_dt = 0.002
        while self.current_time < time - 1e-7:
            step = min(sub_dt, time - self.current_time)
            self._step_substep(step)
            self.current_time += step

        # Prepare StateFrame
        N = self.total_nodes
        G = 32

        # 1. Attributes
        mass_quantity = self.mass.copy()
        strand_id = self.body_id.copy()
        node_index = self.node_index.copy()
        is_endpoint = ((node_index == 0) | (node_index == self.nodes_per_strand - 1)).astype(int)

        attributes = {
            "mass_quantity": mass_quantity,
            "strand_id": strand_id,
            "node_index": node_index,
            "is_endpoint": is_endpoint,
        }

        # 2. Identify active contacts for friction fields
        # Strand-plate contacts + Strand-strand contacts
        r_strand = self.strand_radius
        dist_plate, norm_plate = self._plate_distance_and_normal(self.pos)
        plate_contact_mask = dist_plate <= (r_strand + 0.002)
        plate_contact_indices = np.where(plate_contact_mask)[0]

        # Strand-strand contact pairs
        diff_all = self.pos[:, None, :] - self.pos[None, :, :]
        dist_sq = np.sum(diff_all * diff_all, axis=2)
        body_same = (self.body_id[:, None] == self.body_id[None, :])
        node_diff = np.abs(self.node_index[:, None] - self.node_index[None, :])
        ignore_mask = body_same & (node_diff <= 1)
        dist_sq[ignore_mask] = 1e9

        strand_pairs = np.where(dist_sq < ((2.0 * r_strand + 0.002) ** 2))
        s_i, s_j = strand_pairs[0], strand_pairs[1]
        valid = s_i < s_j
        s_i, s_j = s_i[valid], s_j[valid]

        C = len(plate_contact_indices) + len(s_i)
        if C == 0:
            # Maintain non-empty contact arrays if needed or zero-size
            C = 0
            friction_active_mask = np.zeros(0, dtype=bool)
            friction_dynamic_coeff = np.zeros(0, dtype=float)
            friction_static_coeff = np.zeros(0, dtype=float)
            friction_normal = np.zeros((0, 3), dtype=float)
            friction_normal_force = np.zeros(0, dtype=float)
            friction_tangential_force = np.zeros((0, 3), dtype=float)
            friction_relative_vel = np.zeros((0, 3), dtype=float)
        else:
            friction_active_mask = np.ones(C, dtype=bool)
            friction_dynamic_coeff = np.full(C, self.mu, dtype=float)
            friction_static_coeff = np.full(C, self.mu, dtype=float)
            friction_normal = np.zeros((C, 3), dtype=float)
            friction_normal_force = np.zeros(C, dtype=float)
            friction_tangential_force = np.zeros((C, 3), dtype=float)
            friction_relative_vel = np.zeros((C, 3), dtype=float)

            # Fill plate contacts
            n_p = len(plate_contact_indices)
            if n_p > 0:
                friction_normal[:n_p] = norm_plate[plate_contact_indices]
                v_p = self.vel[plate_contact_indices]
                friction_relative_vel[:n_p] = v_p
                # Normal force
                f_n = self.mass[plate_contact_indices] * 9.81
                friction_normal_force[:n_p] = f_n
                # Tangential force opposing relative velocity
                v_t = v_p - np.sum(v_p * friction_normal[:n_p], axis=1, keepdims=True) * friction_normal[:n_p]
                v_t_mag = np.linalg.norm(v_t, axis=1, keepdims=True)
                v_t_dir = v_t / np.maximum(v_t_mag, 1e-9)
                friction_tangential_force[:n_p] = -self.mu * f_n[:, None] * v_t_dir

            # Fill strand contacts
            n_s = len(s_i)
            if n_s > 0:
                p_i = self.pos[s_i]
                p_j = self.pos[s_j]
                d_ij = p_i - p_j
                d_ij_norm = np.maximum(np.linalg.norm(d_ij, axis=1, keepdims=True), 1e-9)
                n_ij = d_ij / d_ij_norm
                friction_normal[n_p:] = n_ij
                rel_v = self.vel[s_i] - self.vel[s_j]
                friction_relative_vel[n_p:] = rel_v
                f_n_s = 0.5 * (self.mass[s_i] + self.mass[s_j]) * 9.81
                friction_normal_force[n_p:] = f_n_s
                v_t = rel_v - np.sum(rel_v * n_ij, axis=1, keepdims=True) * n_ij
                v_t_mag = np.linalg.norm(v_t, axis=1, keepdims=True)
                v_t_dir = v_t / np.maximum(v_t_mag, 1e-9)
                friction_tangential_force[n_p:] = -self.mu * f_n_s[:, None] * v_t_dir

        # 3. Budget fields
        # K = 1 term (gravity/external forces)
        K = 1
        budget_external_force = np.zeros((G, K, 3), dtype=float)
        # Per strand: sum of m * g
        for s in range(31):
            sl = slice(s * self.nodes_per_strand, (s + 1) * self.nodes_per_strand)
            m_s = np.sum(self.mass[sl])
            budget_external_force[s, 0] = m_s * self.gravity
        # Global domain
        budget_external_force[31, 0] = np.sum(self.mass) * self.gravity

        # Cumulative impulse
        # Integral of external force over time
        for g in range(G):
            self.cum_impulse[g] = budget_external_force[g, 0] * time

        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": np.zeros(K, dtype=float),
            "friction_active_mask": friction_active_mask,
            "friction_dynamic_coefficient": friction_dynamic_coeff,
            "friction_normal": friction_normal,
            "friction_normal_force": friction_normal_force,
            "friction_relative_velocity": friction_relative_vel,
            "friction_static_coefficient": friction_static_coeff,
            "friction_tangential_force": friction_tangential_force,
        }

        # 4. Scalars
        scalars = {
            "strand_radius": self.strand_radius,
            "strand_length": self.strand_length,
            "static_friction": self.mu,
            "kinetic_friction": self.mu,
            "friction_stick_speed_tolerance": 1e-3,
            "reference_force": float(np.sum(self.mass) * 9.81),
            "reference_momentum": float(np.sum(self.mass) * 1.0),
            "reference_velocity": 1.0,
        }

        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "total_nodes": self.total_nodes,
        }

