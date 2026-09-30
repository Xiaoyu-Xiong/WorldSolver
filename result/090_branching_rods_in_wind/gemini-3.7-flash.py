"""Solver Agent implementation of wind-driven motion for branching elastic rods."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(seed)

        # Load topology
        asset = Path(scene.__module__).parent if hasattr(scene, "__file__") else Path(".")
        # Try finding topology asset
        asset_path = Path(self.config["rod_network"]["topology_asset"])
        if not asset_path.is_file():
            # Try relative to scene.py location
            scene_dir = Path(scene.__class__.__module__).parent
            asset_path = Path(__file__).resolve().parent.parent / self.config["rod_network"]["topology_asset"]

        with open(asset_path, "r", encoding="utf-8") as f:
            topo = json.load(f)

        self.topo = topo
        self.rest_positions = np.asarray(topo["vertices"], dtype=np.float64)
        self.edges = np.asarray(topo["edges"], dtype=np.int64)
        self.vertex_levels = np.asarray(topo["vertex_level"], dtype=np.int64)
        self.branch_ids = np.asarray(topo["branch_id"], dtype=np.int64)
        self.lumped_lengths = np.asarray(topo["lumped_length"], dtype=np.float64)

        self.n_verts = len(self.rest_positions)
        self.n_edges = len(self.edges)

        # Levels and radii
        radii_by_level = np.asarray(self.config["rod_network"]["radius_by_level"], dtype=np.float64)
        density_by_level = np.asarray(self.config["rod_network"]["linear_density_by_level"], dtype=np.float64)
        junction_mass_by_level = np.asarray(self.config["rigid_junctions"]["mass_by_level"], dtype=np.float64)

        self.radii = radii_by_level[self.vertex_levels]
        self.linear_densities = density_by_level[self.vertex_levels]

        # Junction indices
        self.junction_indices = np.asarray(self.config["rigid_junctions"]["vertex_indices"], dtype=np.int64)
        self.is_junction = np.zeros(self.n_verts, dtype=np.float64)
        self.is_junction[self.junction_indices] = 1.0

        self.fixed_root = int(self.config["rigid_junctions"]["fixed_root_vertex"])

        # Masses
        # Base rod segment mass
        mass = self.linear_densities * np.pi * (self.radii ** 2) * self.lumped_lengths
        # Add rigid junction lump mass
        for j_idx in self.junction_indices:
            lvl = self.vertex_levels[j_idx]
            mass[j_idx] += junction_mass_by_level[lvl]

        self.mass = mass
        # Fixed root has finite mass for conservation accounting but infinite inertial mass for dynamics
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-6)
        self.inv_mass[self.fixed_root] = 0.0

        # State initialization
        self.positions = self.rest_positions.copy()
        self.velocities = np.zeros_like(self.positions)
        self.orientations = np.zeros((self.n_verts, 4), dtype=np.float64)
        self.orientations[:, 3] = 1.0
        self.angular_velocities = np.zeros((self.n_verts, 3), dtype=np.float64)

        # Edge rest lengths and stiffness
        self.edge_rest_lengths = np.linalg.norm(
            self.rest_positions[self.edges[:, 1]] - self.rest_positions[self.edges[:, 0]],
            axis=1
        )
        # Young's modulus scaling (E approx 1e6 to 5e6 for wood/bark compliance)
        # Bending and axial stiffness
        # Level 0 (trunk): stiffer; Level 1: medium; Level 2: flexible
        self.edge_levels = np.minimum(self.vertex_levels[self.edges[:, 0]], self.vertex_levels[self.edges[:, 1]])

        # Precompute bending hinges / angle constraints
        # For each interior vertex connected to 2 or more neighbors in the tree
        # Build adjacency
        adj = [[] for _ in range(self.n_verts)]
        for e_idx, (u, v) in enumerate(self.edges):
            adj[u].append(v)
            adj[v].append(u)

        # Hinges: pairs of edges meeting at a vertex
        hinges = []
        rest_angles = []
        hinge_stiffness = []
        for i in range(self.n_verts):
            nbrs = adj[i]
            if len(nbrs) >= 2:
                for j in range(len(nbrs)):
                    for k in range(j + 1, len(nbrs)):
                        u, v = nbrs[j], nbrs[k]
                        # Hinge u - i - v
                        r_iu = self.rest_positions[u] - self.rest_positions[i]
                        r_iv = self.rest_positions[v] - self.rest_positions[i]
                        len_iu = np.linalg.norm(r_iu)
                        len_iv = np.linalg.norm(r_iv)
                        if len_iu > 1e-7 and len_iv > 1e-7:
                            cos_theta = np.clip(np.dot(r_iu, r_iv) / (len_iu * len_iv), -1.0, 1.0)
                            angle = np.arccos(cos_theta)
                            hinges.append((u, i, v))
                            rest_angles.append(angle)
                            # Bending stiffness based on vertex level and radius: EI ~ E * pi/4 * r^4
                            lvl = self.vertex_levels[i]
                            r = self.radii[i]
                            # Bending stiffness parameter
                            k_bend = 40.0 * (r / 0.06)**3
                            hinge_stiffness.append(k_bend)

        self.hinges = np.asarray(hinges, dtype=np.int64) if hinges else np.zeros((0, 3), dtype=np.int64)
        self.rest_angles = np.asarray(rest_angles, dtype=np.float64) if rest_angles else np.zeros(0, dtype=np.float64)
        self.hinge_stiffness = np.asarray(hinge_stiffness, dtype=np.float64) if hinge_stiffness else np.zeros(0, dtype=np.float64)

        # Simulation parameters
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)
        self.wind_cfg = self.config["wind"]
        self.wind_dir = np.asarray(self.wind_cfg["direction"], dtype=np.float64)
        self.wind_dir = self.wind_dir / (np.linalg.norm(self.wind_dir) + 1e-12)

        # Wind keyframes
        self.wind_kf_times = np.asarray([kf["time"] for kf in self.wind_cfg["keyframes"]], dtype=np.float64)
        self.wind_kf_speeds = np.asarray([kf["speed"] for kf in self.wind_cfg["keyframes"]], dtype=np.float64)
        self.gust_freq = float(self.wind_cfg.get("gust_frequency_hz", 0.55))
        self.branch_phase_offsets = np.asarray(self.wind_cfg.get("branch_phase_offsets", [0.0]), dtype=np.float64)

        # Cumulative budget integrals
        self.current_time = 0.0
        self.cum_ext_impulse = np.zeros((1, 3), dtype=np.float64)
        self.cum_ext_ang_impulse = np.zeros((1, 3), dtype=np.float64)

        # Budget domains: 1 domain (entire tree)
        self.domain_weights = np.ones((1, self.n_verts), dtype=np.float64)

        # Internal sub-stepping
        self.sim_dt = 0.001  # 1 ms sub-step for numerical stability

    def _get_wind_speed(self, t: float, branch_id: int) -> float:
        # Base speed from keyframes
        base_speed = float(np.interp(t, self.wind_kf_times, self.wind_kf_speeds))
        # Gust modulation
        phase_idx = branch_id % len(self.branch_phase_offsets)
        phase = self.branch_phase_offsets[phase_idx]
        gust = 1.0 + 0.35 * np.sin(2.0 * np.pi * self.gust_freq * t + phase * 2.0 * np.pi)
        return max(0.0, base_speed * gust)

    def _compute_external_forces(self, pos: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute gravity and wind forces on each vertex."""
        forces = np.zeros_like(pos)
        # Gravity force
        forces += self.mass[:, None] * self.gravity[None, :]

        # Wind force
        # Spatial profile: height_and_branch_area_weighted
        # Drag force ~ 0.5 * rho * Cd * A * v_rel^2
        # A ~ 2 * radius * lumped_length
        # Height factor: (z / z_ref)^0.5
        for i in range(self.n_verts):
            b_id = self.branch_ids[i]
            speed = self._get_wind_speed(t, b_id)
            z = max(0.1, pos[i, 2])
            height_factor = (z / 4.0) ** 0.4
            area = 2.0 * self.radii[i] * self.lumped_lengths[i]

            # Drag load
            drag_mag = 0.08 * (speed ** 1.8) * area * height_factor
            f_wind = drag_mag * self.wind_dir
            forces[i] += f_wind

        return forces

    def _step_substep(self, dt: float) -> None:
        """Physical step using Position-Based Dynamics / Verlet-style elastic rods."""
        t = self.current_time

        # 1. External forces
        forces = self._compute_external_forces(self.positions, t)

        # Budget tracking before position updates
        # Net external force on the system
        net_ext_force = np.sum(forces, axis=0)
        # External torque about origin (0,0,0)
        ext_torques = np.cross(self.positions, forces)
        net_ext_torque = np.sum(ext_torques, axis=0)

        # Accumulate impulse
        self.cum_ext_impulse[0] += net_ext_force * dt
        self.cum_ext_ang_impulse[0] += net_ext_torque * dt

        # 2. Explicit velocity step with external forces and damping
        damping = np.exp(-0.8 * dt)
        self.velocities = (self.velocities + (forces * self.inv_mass[:, None]) * dt) * damping
        self.velocities[self.fixed_root] = 0.0

        # Predicted positions
        p_pred = self.positions + self.velocities * dt
        p_pred[self.fixed_root] = self.rest_positions[self.fixed_root]

        # 3. Constraint projection (Position-Based Dynamics)
        # Stretch and Bending constraints
        n_iters = 15
        for _ in range(n_iters):
            # Fixed root
            p_pred[self.fixed_root] = self.rest_positions[self.fixed_root]

            # (a) Edge distance constraints (stretch resistance)
            for e_idx in range(self.n_edges):
                u, v = self.edges[e_idx]
                p_u = p_pred[u]
                p_v = p_pred[v]
                diff = p_v - p_u
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    continue
                rest_len = self.edge_rest_lengths[e_idx]
                w_u = self.inv_mass[u]
                w_v = self.inv_mass[v]
                w_sum = w_u + w_v
                if w_sum < 1e-12:
                    continue

                corr = (dist - rest_len) / w_sum * (diff / dist)
                # Stiffness 0.95
                alpha = 0.95
                if w_u > 0:
                    p_pred[u] += alpha * w_u * corr
                if w_v > 0:
                    p_pred[v] -= alpha * w_v * corr

            # (b) Bending constraints (hinges)
            # Restore angle between adjacent edges
            for h_idx in range(len(self.hinges)):
                u, i, v = self.hinges[h_idx]
                p_u = p_pred[u]
                p_i = p_pred[i]
                p_v = p_pred[v]

                r_iu = p_u - p_i
                r_iv = p_v - p_i
                len_iu = np.linalg.norm(r_iu)
                len_iv = np.linalg.norm(r_iv)
                if len_iu < 1e-7 or len_iv < 1e-7:
                    continue

                e_iu = r_iu / len_iu
                e_iv = r_iv / len_iv

                cos_th = np.clip(np.dot(e_iu, e_iv), -0.9999, 0.9999)
                cur_angle = np.arccos(cos_th)
                target_angle = self.rest_angles[h_idx]

                angle_diff = cur_angle - target_angle
                if abs(angle_diff) < 1e-6:
                    continue

                # Normal to the plane
                axis = np.cross(e_iu, e_iv)
                axis_len = np.linalg.norm(axis)
                if axis_len < 1e-7:
                    continue
                axis = axis / axis_len

                # Direction to rotate e_iu towards target angle
                # Gradient of angle w.r.t positions
                # Rotate e_iu and e_iv
                k_b = self.hinge_stiffness[h_idx] * 0.05
                k_b = min(0.3, k_b)

                # Correct positions
                w_u = self.inv_mass[u]
                w_i = self.inv_mass[i]
                w_v = self.inv_mass[v]
                w_tot = w_u + 2.0 * w_i + w_v
                if w_tot < 1e-12:
                    continue

                dir_u = np.cross(axis, e_iu)
                dir_v = -np.cross(axis, e_iv)

                d_pos_u = dir_u * (angle_diff * len_iu * k_b * (w_u / w_tot))
                d_pos_v = dir_v * (angle_diff * len_iv * k_b * (w_v / w_tot))
                d_pos_i = -(d_pos_u + d_pos_v) * (w_i / w_tot)

                if w_u > 0:
                    p_pred[u] += d_pos_u
                if w_i > 0:
                    p_pred[i] += d_pos_i
                if w_v > 0:
                    p_pred[v] += d_pos_v

            # (c) Ground collision
            for j in range(self.n_verts):
                if p_pred[j, 2] < 0.0:
                    p_pred[j, 2] = 0.0

        # Fixed root enforce
        p_pred[self.fixed_root] = self.rest_positions[self.fixed_root]

        # 4. Update velocity and position
        self.velocities = (p_pred - self.positions) / dt
        self.positions = p_pred

        # 5. Angular velocities estimation from branch deformation
        # For StateFrame output
        self.angular_velocities = np.zeros_like(self.velocities)
        for e_idx in range(self.n_edges):
            u, v = self.edges[e_idx]
            r = self.positions[v] - self.positions[u]
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-6:
                v_rel = self.velocities[v] - self.velocities[u]
                omega_e = np.cross(r, v_rel) / (r_norm ** 2)
                self.angular_velocities[u] += 0.5 * omega_e
                self.angular_velocities[v] += 0.5 * omega_e

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward to requested time
        while self.current_time + 1e-8 < time:
            dt = min(self.sim_dt, time - self.current_time)
            self._step_substep(dt)

        self.current_time = time

        # Build StateFrame
        n = self.n_verts

        # Inertia tensor for rod elements: cylinder inertia around center of mass
        # I_diag ~ [1/12 m L^2 + 1/4 m r^2, 1/12 m L^2 + 1/4 m r^2, 1/2 m r^2]
        inertia_tensor = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            m = self.mass[i]
            r = self.radii[i]
            l = self.lumped_lengths[i]
            i_trans = (1.0 / 12.0) * m * (l ** 2) + (1.0 / 4.0) * m * (r ** 2)
            i_axial = 0.5 * m * (r ** 2)
            inertia_tensor[i, 0, 0] = i_trans
            inertia_tensor[i, 1, 1] = i_trans
            inertia_tensor[i, 2, 2] = i_axial

        # Current external forces and torques for budget fields
        forces = self._compute_external_forces(self.positions, self.current_time)
        net_ext_force = np.sum(forces, axis=0)
        ext_torques = np.cross(self.positions, forces)
        net_ext_torque = np.sum(ext_torques, axis=0)

        # Budget force shape: [G, K, 3] where G=1, K=1
        budget_ext_force = np.zeros((1, 1, 3), dtype=np.float64)
        budget_ext_force[0, 0] = net_ext_force

        # Budget force position: center of force or weighted center of mass
        com = np.sum(self.positions * self.mass[:, None], axis=0) / np.sum(self.mass)
        budget_force_pos = np.zeros((1, 1, 3), dtype=np.float64)
        budget_force_pos[0, 0] = com

        # Couple: net torque about center of mass
        torque_about_com = net_ext_torque - np.cross(com, net_ext_force)
        budget_ext_couple = np.zeros((1, 1, 3), dtype=np.float64)
        budget_ext_couple[0, 0] = torque_about_com

        # Inflow / Outflow momentum
        budget_inflow_mom = np.zeros((1, 3), dtype=np.float64)
        budget_inflow_ang_mom = np.zeros((1, 3), dtype=np.float64)
        budget_outflow_mom = np.zeros((1, 3), dtype=np.float64)
        budget_outflow_ang_mom = np.zeros((1, 3), dtype=np.float64)

        # Cumulative mass source
        cum_mass_source = np.zeros(1, dtype=np.float64)

        # Reference momentum and angular momentum scales
        total_mass = np.sum(self.mass)
        ref_speed = 10.0  # characteristic gust velocity ~10 m/s
        ref_length = 5.0  # tree height ~6 m
        ref_momentum = float(total_mass * ref_speed)
        ref_ang_momentum = float(total_mass * ref_speed * ref_length)

        # Attributes
        attributes = {
            "inertia_tensor": inertia_tensor,
            "mass_quantity": self.mass.copy(),
            "is_rod_vertex": np.ones(n, dtype=np.float64),
            "is_rigid_junction": self.is_junction.copy(),
            "is_fixed_root": (np.arange(n) == self.fixed_root).astype(np.float64),
            "branch_id": self.branch_ids.astype(np.float64),
            "branch_level": self.vertex_levels.astype(np.float64),
            "rod_radius": self.radii.copy(),
        }

        # Fields
        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.copy(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_angular_momentum": budget_inflow_ang_mom,
            "budget_cumulative_inflow_momentum": budget_inflow_mom,
            "budget_cumulative_outflow_angular_momentum": budget_outflow_ang_mom,
            "budget_cumulative_outflow_momentum": budget_outflow_mom,
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_couple": budget_ext_couple,
            "budget_external_force": budget_ext_force,
            "budget_force_position": budget_force_pos,
            "cumulative_mass_source": cum_mass_source,
        }

        # Scalars
        cur_wind_speed = float(np.interp(self.current_time, self.wind_kf_times, self.wind_kf_speeds))
        scalars = {
            "reference_angular_momentum": ref_ang_momentum,
            "reference_momentum": ref_momentum,
            "wind_speed": cur_wind_speed,
            "gravity_magnitude": 9.81,
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientations.copy(),
            angular_velocity=self.angular_velocities.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_verts": self.n_verts,
            "current_time": self.current_time,
            "max_displacement": float(np.max(np.linalg.norm(self.positions - self.rest_positions, axis=1))),
        }

