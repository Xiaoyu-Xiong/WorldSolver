"""Wind-driven branching elastic rod solver.

Implements a lumped-mass bending/twisting resistant rod network with rigid
junctions, driven by prescribed wind loads and gravity. Uses semi-implicit
Euler time integration with penalty-style bending stiffness to reproduce
branch-dependent bending amplitudes under time-varying wind.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Lumped-mass discrete elastic rod network solver."""

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config

        # Load topology
        asset_path = Path(__file__).resolve().parent.parent / cfg["rod_network"]["topology_asset"]
        data = json.loads(asset_path.read_text(encoding="utf-8"))
        self.rest_positions = np.asarray(data["vertices"], dtype=np.float64)
        self.edges = np.asarray(data["edges"], dtype=np.int64)
        self.vertex_level = np.asarray(data["vertex_level"], dtype=np.int64)
        self.branch_id = np.asarray(data["branch_id"], dtype=np.int64)
        self.lumped_length = np.asarray(data["lumped_length"], dtype=np.float64)
        self.n_vertices = len(self.rest_positions)

        # Physical parameters per level
        self.radius_by_level = np.asarray(cfg["rod_network"]["radius_by_level"], dtype=np.float64)
        self.density_by_level = np.asarray(cfg["rod_network"]["linear_density_by_level"], dtype=np.float64)
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        # Junction info
        self.junction_indices = np.asarray(cfg["rigid_junctions"]["vertex_indices"], dtype=np.int64)
        self.fixed_root = int(cfg["rigid_junctions"]["fixed_root_vertex"])
        self.junction_mass_by_level = np.asarray(cfg["rigid_junctions"]["mass_by_level"], dtype=np.float64)

        # Wind config
        wind_cfg = cfg["wind"]
        self.wind_direction = np.asarray(wind_cfg["direction"], dtype=np.float64)
        wd_norm = np.linalg.norm(self.wind_direction)
        if wd_norm > 0:
            self.wind_direction = self.wind_direction / wd_norm
        self.wind_keyframes = [(float(kf["time"]), float(kf["speed"])) for kf in wind_cfg["keyframes"]]
        self.gust_frequency = float(wind_cfg["gust_frequency_hz"])
        self.branch_phase_offsets = np.asarray(wind_cfg["branch_phase_offsets"], dtype=np.float64)

        # Derived quantities
        radii = self.radius_by_level[self.vertex_level]
        density = self.density_by_level[self.vertex_level]
        self.vertex_mass = density * np.pi * radii**2 * self.lumped_length
        # Add junction mass lumped at junction vertices
        for lvl, m in enumerate(self.junction_mass_by_level):
            mask = (self.vertex_level == lvl)
            junc_mask = mask & np.isin(np.arange(self.n_vertices), self.junction_indices)
            if np.any(junc_mask):
                self.vertex_mass[junc_mask] += m / np.sum(junc_mask)

        # Bending stiffness: EI per vertex
        E_modulus = 1.0e7
        self.EI = E_modulus * np.pi * radii**4 / 4.0
        G_modulus = E_modulus / 2.5
        self.GJ = G_modulus * np.pi * radii**4 / 2.0

        # Edge rest data
        self.edge_rest_vectors = self.rest_positions[self.edges[:, 1]] - self.rest_positions[self.edges[:, 0]]
        self.edge_rest_lengths = np.linalg.norm(self.edge_rest_vectors, axis=1)
        self.edge_rest_unit = self.edge_rest_vectors / np.maximum(self.edge_rest_lengths[:, None], 1e-12)

        # Initial state
        initial_state = scene.initial_state()
        self.positions = np.array(initial_state.position, dtype=np.float64)
        self.velocities = np.array(initial_state.velocity, dtype=np.float64)
        self.orientation = np.array(initial_state.orientation, dtype=np.float64)
        self.angular_velocity = np.array(initial_state.angular_velocity, dtype=np.float64)
        self.current_time = 0.0

        # Budget accumulators (G=1 global domain, K=2 force terms: gravity + wind)
        self.G = 1
        self.K = 2
        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_ext_ang_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_inflow_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_outflow_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_inflow_ang_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_outflow_ang_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.budget_domain_weights = np.ones((self.G, self.n_vertices), dtype=np.float64)
        self.budget_ext_force = np.zeros((self.G, self.K, 3), dtype=np.float64)
        self.budget_ext_couple = np.zeros((self.G, self.K, 3), dtype=np.float64)
        self.budget_force_position = np.zeros((self.G, self.K, 3), dtype=np.float64)
        self.cum_mass_source = np.zeros(1, dtype=np.float64)

        # Reference scales
        total_mass = float(np.sum(self.vertex_mass))
        self.ref_momentum = max(total_mass * 1.0, 1e-6)
        self.ref_ang_momentum = max(total_mass * 1.0 * 1.0, 1e-6)

        # Damping
        self.linear_damping = 0.8
        self.angular_damping = 1.5

    def _wind_speed_at(self, t: float) -> float:
        kf = self.wind_keyframes
        if t <= kf[0][0]:
            return kf[0][1]
        if t >= kf[-1][0]:
            return kf[-1][1]
        for i in range(len(kf) - 1):
            t0, s0 = kf[i]
            t1, s1 = kf[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0)
                return s0 + alpha * (s1 - s0)
        return kf[-1][1]

    def _compute_forces(self, t: float):
        pos = self.positions
        vel = self.velocities
        n = self.n_vertices

        # Gravity force per vertex
        f_gravity = self.vertex_mass[:, None] * self.gravity[None, :]

        # Wind force
        wind_speed = self._wind_speed_at(t)
        heights = pos[:, 2]
        h_max = max(float(np.max(heights)), 1.0)
        height_weight = np.clip(heights / h_max, 0.0, 1.0)
        radii = self.radius_by_level[self.vertex_level]
        area_weight = radii * self.lumped_length
        unique_branches = np.unique(self.branch_id)
        gust_mod = np.ones(n, dtype=np.float64)
        for bid in unique_branches:
            mask = self.branch_id == bid
            phase_idx = int(bid) % len(self.branch_phase_offsets)
            phase = self.branch_phase_offsets[phase_idx]
            gust_mod[mask] = 1.0 + 0.4 * np.sin(2.0 * np.pi * self.gust_frequency * t + phase)
        wind_mag = 0.5 * 1.225 * (wind_speed**2) * height_weight * area_weight * gust_mod
        f_wind = wind_mag[:, None] * self.wind_direction[None, :]

        # Elastic restoring forces
        f_elastic = np.zeros_like(pos)
        edge_vecs = pos[self.edges[:, 1]] - pos[self.edges[:, 0]]
        edge_lens = np.linalg.norm(edge_vecs, axis=1, keepdims=True)
        edge_lens_safe = np.maximum(edge_lens, 1e-12)
        edge_units = edge_vecs / edge_lens_safe

        # Axial stretch stiffness
        k_stretch = 1.0e6
        stretch = edge_lens.squeeze() - self.edge_rest_lengths
        f_stretch = k_stretch * stretch[:, None] * edge_units
        np.add.at(f_elastic, self.edges[:, 0], f_stretch)
        np.add.at(f_elastic, self.edges[:, 1], -f_stretch)

        # Bending: penalize displacement from rest
        displacement = pos - self.rest_positions
        bend_k = self.EI.copy()
        bend_k = np.clip(bend_k, 1.0, 1.0e6)
        f_bend = -bend_k[:, None] * displacement
        f_elastic += f_bend

        # Linear damping
        f_damp = -self.linear_damping * self.vertex_mass[:, None] * vel

        f_total = f_gravity + f_wind + f_elastic + f_damp
        return f_total, f_gravity, f_wind

    def advance_to(self, time: float) -> StateFrame:
        dt_max = 0.002
        t = self.current_time
        target = float(time)

        while t < target - 1e-12:
            dt = min(dt_max, target - t)
            if dt <= 0:
                break

            f_total, f_grav, f_wind = self._compute_forces(t)

            # Semi-implicit Euler
            acc = f_total / np.maximum(self.vertex_mass[:, None], 1e-12)
            self.velocities += acc * dt
            self.positions += self.velocities * dt

            # Fix root
            self.positions[self.fixed_root] = self.rest_positions[self.fixed_root]
            self.velocities[self.fixed_root] = 0.0

            # Angular velocity from bending
            omega = np.zeros_like(self.angular_velocity)
            for ei in range(len(self.edges)):
                i0, i1 = self.edges[ei]
                rest_u = self.edge_rest_unit[ei]
                curr_vec = self.positions[i1] - self.positions[i0]
                curr_len = np.linalg.norm(curr_vec)
                if curr_len < 1e-12:
                    continue
                curr_u = curr_vec / curr_len
                cross = np.cross(rest_u, curr_u)
                sin_angle = np.linalg.norm(cross)
                if sin_angle > 1e-12:
                    axis = cross / sin_angle
                    angle = np.arcsin(min(sin_angle, 1.0))
                    rot_vec = axis * angle
                else:
                    rot_vec = cross
                omega[i0] += rot_vec * 0.5
                omega[i1] += rot_vec * 0.5
            conn_count = np.bincount(self.edges.ravel(), minlength=self.n_vertices).astype(float)
            conn_count = np.maximum(conn_count, 1.0)
            omega /= conn_count[:, None]
            self.angular_velocity = omega
            self.angular_velocity *= max(0.0, 1.0 - self.angular_damping * dt)

            # Update orientation quaternions
            for vi in range(self.n_vertices):
                w = self.angular_velocity[vi]
                w_norm = np.linalg.norm(w)
                if w_norm * dt > 1e-12:
                    half_angle = w_norm * dt * 0.5
                    s = np.sin(half_angle) / w_norm
                    dq = np.array([w[0]*s, w[1]*s, w[2]*s, np.cos(half_angle)])
                    q = self.orientation[vi]
                    new_q = np.array([
                        q[3]*dq[0] + q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1],
                        q[3]*dq[1] - q[0]*dq[2] + q[1]*dq[3] + q[2]*dq[0],
                        q[3]*dq[2] + q[0]*dq[1] - q[1]*dq[0] + q[2]*dq[3],
                        q[3]*dq[3] - q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2],
                    ])
                    norm = np.linalg.norm(new_q)
                    if norm > 1e-12:
                        self.orientation[vi] = new_q / norm

            # Budget accumulation
            com = np.average(self.positions, weights=self.vertex_mass, axis=0)
            total_grav = np.sum(f_grav, axis=0)
            total_wind = np.sum(f_wind, axis=0)
            self.budget_ext_force[0, 0] = total_grav
            self.budget_ext_force[0, 1] = total_wind
            self.budget_force_position[0, 0] = com
            self.budget_force_position[0, 1] = com
            self.cum_ext_impulse[0] += (total_grav + total_wind) * dt
            ang_imp = np.cross(com, total_grav + total_wind) * dt
            self.cum_ext_ang_impulse[0] += ang_imp

            t += dt

        self.current_time = target

        inertia = np.zeros((self.n_vertices, 3, 3), dtype=np.float64)

        frame = StateFrame(
            time=target,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.vertex_mass.copy(),
            body_id=np.zeros(self.n_vertices, dtype=np.int64),
            attributes={
                "is_rod_vertex": np.ones(self.n_vertices, dtype=np.float64),
                "is_rigid_junction": np.isin(
                    np.arange(self.n_vertices), self.junction_indices
                ).astype(np.float64),
                "is_fixed_root": (np.arange(self.n_vertices) == self.fixed_root).astype(np.float64),
                "branch_id": self.branch_id.astype(np.float64),
                "branch_level": self.vertex_level.astype(np.float64),
                "rod_radius": self.radius_by_level[self.vertex_level].copy(),
                "inertia_tensor": inertia,
                "mass_quantity": self.vertex_mass.copy(),
            },
            fields={
                "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.copy(),
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_angular_momentum": self.cum_inflow_ang_mom.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_angular_momentum": self.cum_outflow_ang_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": self.budget_domain_weights.copy(),
                "budget_external_couple": self.budget_ext_couple.copy(),
                "budget_external_force": self.budget_ext_force.copy(),
                "budget_force_position": self.budget_force_position.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
            },
            scalars={
                "reference_angular_momentum": self.ref_ang_momentum,
                "reference_momentum": self.ref_momentum,
                "wind_speed": self._wind_speed_at(target),
                "gravity_magnitude": float(np.linalg.norm(self.gravity)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_vertices": self.n_vertices,
            "n_edges": len(self.edges),
            "seed": self.seed,
        }
