"""Solver Agent implementation for 071_soft_objects_squeeze_out."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Simulates 5 soft volumetric bodies squeezed through a funnel/outlet tube."""

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config
        self.duration = float(self.config.get("duration", 8.0))
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Base mesh template (12 vertices of icosahedron)
        self.mesh_v = np.array([
            [-0.525731, 0.850651, 0.000000],
            [0.525731, 0.850651, 0.000000],
            [-0.525731, -0.850651, 0.000000],
            [0.525731, -0.850651, 0.000000],
            [0.000000, -0.525731, 0.850651],
            [0.000000, 0.525731, 0.850651],
            [0.000000, -0.525731, -0.850651],
            [0.000000, 0.525731, -0.850651],
            [0.850651, 0.000000, -0.525731],
            [0.850651, 0.000000, 0.525731],
            [-0.850651, 0.000000, -0.525731],
            [-0.850651, 0.000000, 0.525731],
        ], dtype=float)
        self.n_nodes_per_body = 12

        # 5 soft bodies defined in scene geometry
        # id 0: red, id 1: blue, id 2: green, id 3: gold, id 4: magenta
        bodies_spec = [
            {"id": 0, "pos": [-0.25, -0.12, 4.13], "scale": [0.45, 0.35, 0.33], "rot": [0, 0, 0, 1], "mass": 0.62},
            {"id": 1, "pos": [0.22, 0.12, 3.34], "scale": [0.33, 0.30, 0.55], "rot": [0, 0.259, 0, 0.966], "mass": 0.54},
            {"id": 2, "pos": [-0.18, 0.10, 2.55], "scale": [0.45, 0.39, 0.46], "rot": [0, 0, 0, 1], "mass": 0.70},
            {"id": 3, "pos": [0.18, -0.10, 1.72], "scale": [0.34, 0.32, 0.56], "rot": [0, -0.174, 0, 0.985], "mass": 0.58},
            {"id": 4, "pos": [-0.20, -0.02, 0.98], "scale": [0.42, 0.33, 0.30], "rot": [0, 0, 0, 1], "mass": 0.50},
        ]
        self.num_bodies = len(bodies_spec)
        self.total_nodes = self.num_bodies * self.n_nodes_per_body

        # Construct initial node positions and rest positions
        self.x = np.zeros((self.total_nodes, 3), dtype=float)
        self.v = np.zeros((self.total_nodes, 3), dtype=float)
        self.rest_x = np.zeros((self.total_nodes, 3), dtype=float)  # Body-relative rest coords
        self.mass = np.zeros(self.total_nodes, dtype=float)
        self.body_id = np.zeros(self.total_nodes, dtype=np.int64)

        for i, b in enumerate(bodies_spec):
            s = np.array(b["scale"])
            q = np.array(b["rot"]) # [qx, qy, qz, qw]
            # Convert quat to rot matrix
            qx, qy, qz, qw = q
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])
            scaled_v = self.mesh_v * s
            rot_scaled_v = (R @ scaled_v.T).T
            idx_start = i * self.n_nodes_per_body
            idx_end = idx_start + self.n_nodes_per_body

            self.rest_x[idx_start:idx_end] = rot_scaled_v
            self.x[idx_start:idx_end] = rot_scaled_v + np.array(b["pos"])
            self.mass[idx_start:idx_end] = b["mass"] / self.n_nodes_per_body
            self.body_id[idx_start:idx_end] = b["id"]

        # Physics simulation state
        self.current_time = 0.0
        self.sub_dt = 0.001

        # Budget tracking
        self.G = 6  # 5 individual body domains + 1 global domain
        self.K = 1
        self.cumulative_impulse = np.zeros((self.G, 3), dtype=float)
        self.cumulative_inflow = np.zeros((self.G, 3), dtype=float)
        self.cumulative_outflow = np.zeros((self.G, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(self.K, dtype=float)
        self.external_force_record = np.zeros((self.G, self.K, 3), dtype=float)
        self.interaction_force_record = np.zeros((self.K, 3), dtype=float)
        self.ref_momentum = float(np.sum(self.mass) * 1.0)  # Reference momentum scale

    def _plate_z(self, t: float) -> float:
        # start_time: 0.5, end_time: 5.8, start_z: 4.85, end_z: 0.58
        if t < 0.5:
            return 4.85
        elif t > 5.8:
            return 0.58
        else:
            alpha = (t - 0.5) / (5.8 - 0.5)
            # smoothstep motion
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            return 4.85 + s * (0.58 - 4.85)

    def _plate_vz(self, t: float) -> float:
        if t < 0.5 or t > 5.8:
            return 0.0
        else:
            alpha = (t - 0.5) / (5.8 - 0.5)
            ds = 6.0 * alpha * (1.0 - alpha) / (5.8 - 0.5)
            return ds * (0.58 - 4.85)

    def _apply_boundary_and_plate_collision(self, t: float, dt: float):
        plate_z = self._plate_z(t)
        plate_vz = self._plate_vz(t)

        # Walls definitions
        # Chamber: z in [1.4, 5.0], half_width x = 0.72, half_depth y = 0.52
        # Funnel: z in [0.0, 1.4], narrowing from (0.72, 0.52) to outlet (0.34, 0.13)
        # Collector floor: z = -2.0, walls x = [-1.25, 1.25], y = [-0.9, 0.9]

        ext_forces = np.zeros((self.G, 3), dtype=float)

        for b in range(self.num_bodies):
            idx_start = b * self.n_nodes_per_body
            idx_end = idx_start + self.n_nodes_per_body
            bx = self.x[idx_start:idx_end]
            bv = self.v[idx_start:idx_end]
            bm = self.mass[idx_start:idx_end]

            for i in range(self.n_nodes_per_body):
                px, py, pz = bx[i]
                vx, vy, vz = bv[i]
                m = bm[i]

                # Plate collision (plate moves down at plate_z)
                if pz > plate_z - 0.05 and pz < plate_z + 0.5:
                    if vz > plate_vz:
                        pen = (plate_z - 0.05) - pz
                        f_n = max(0.0, -pen * 5000.0) + (plate_vz - vz) * 200.0
                        vz = plate_vz
                        pz = min(pz, plate_z - 0.05)
                        ext_forces[b, 2] -= f_n
                        ext_forces[5, 2] -= f_n

                # Horizontal limits based on z
                if pz >= 1.4:
                    # In main chamber
                    limit_x = 0.70
                    limit_y = 0.50
                elif pz >= 0.0:
                    # In funnel
                    alpha = pz / 1.4
                    limit_x = 0.33 + alpha * (0.70 - 0.33)
                    limit_y = 0.12 + alpha * (0.50 - 0.12)
                else:
                    # Below outlet, in collector region
                    limit_x = 1.20
                    limit_y = 0.85

                # X limits
                if px > limit_x:
                    pen = px - limit_x
                    px = limit_x
                    vx = -0.1 * vx
                    ext_forces[b, 0] -= pen * 2000.0
                    ext_forces[5, 0] -= pen * 2000.0
                elif px < -limit_x:
                    pen = -limit_x - px
                    px = -limit_x
                    vx = -0.1 * vx
                    ext_forces[b, 0] += pen * 2000.0
                    ext_forces[5, 0] += pen * 2000.0

                # Y limits
                if py > limit_y:
                    pen = py - limit_y
                    py = limit_y
                    vy = -0.1 * vy
                    ext_forces[b, 1] -= pen * 2000.0
                    ext_forces[5, 1] -= pen * 2000.0
                elif py < -limit_y:
                    pen = -limit_y - py
                    py = -limit_y
                    vy = -0.1 * vy
                    ext_forces[b, 1] += pen * 2000.0
                    ext_forces[5, 1] += pen * 2000.0

                # Collector floor collision (z = -1.95)
                if pz < -1.95:
                    pen = -1.95 - pz
                    pz = -1.95
                    vz = -0.1 * vz
                    # Friction
                    vx *= 0.8
                    vy *= 0.8
                    ext_forces[b, 2] += pen * 3000.0
                    ext_forces[5, 2] += pen * 3000.0

                bx[i] = [px, py, pz]
                bv[i] = [vx, vy, vz]

        return ext_forces

    def _step(self, dt: float):
        # 1. External forces: gravity
        for b in range(self.num_bodies):
            idx_start = b * self.n_nodes_per_body
            idx_end = idx_start + self.n_nodes_per_body
            self.v[idx_start:idx_end] += self.gravity * dt

        # 2. Internal elasticity via Shape Matching (Müller et al.)
        # Allows large deformation, elastic recovery, volume preservation
        for b in range(self.num_bodies):
            idx_start = b * self.n_nodes_per_body
            idx_end = idx_start + self.n_nodes_per_body
            bx = self.x[idx_start:idx_end]
            bv = self.v[idx_start:idx_end]
            rx0 = self.rest_x[idx_start:idx_end]
            bm = self.mass[idx_start:idx_end]
            m_tot = np.sum(bm)

            # Center of mass
            cm_curr = np.sum(bx * bm[:, None], axis=0) / m_tot
            cm_rest = np.sum(rx0 * bm[:, None], axis=0) / m_tot

            q = rx0 - cm_rest
            p = bx - cm_curr

            # Covariance matrix A_pq = sum m_i p_i q_i^T
            A_pq = p.T @ (q * bm[:, None])

            # Extract rotation via SVD
            try:
                U, S, Vt = np.linalg.svd(A_pq)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
            except Exception:
                R = np.eye(3)

            # Target goal positions
            goal_x = cm_curr + (R @ q.T).T

            # Elastic pull towards goal + damping
            k_stiff = 180.0
            damping = 8.0
            f_elastic = k_stiff * (goal_x - bx) - damping * bv
            self.v[idx_start:idx_end] += (f_elastic / bm[:, None]) * dt

        # 3. Inter-body collision avoidance
        # Repulsive forces between nodes of different bodies
        inter_forces = np.zeros(3, dtype=float)
        for b1 in range(self.num_bodies):
            idx1_s = b1 * self.n_nodes_per_body
            idx1_e = idx1_s + self.n_nodes_per_body
            for b2 in range(b1 + 1, self.num_bodies):
                idx2_s = b2 * self.n_nodes_per_body
                idx2_e = idx2_s + self.n_nodes_per_body

                # Distance between centers of mass
                cm1 = np.mean(self.x[idx1_s:idx1_e], axis=0)
                cm2 = np.mean(self.x[idx2_s:idx2_e], axis=0)
                d_cm = cm2 - cm1
                dist_cm = np.linalg.norm(d_cm)
                r_eff = 0.50
                if dist_cm < 2 * r_eff and dist_cm > 1e-5:
                    n = d_cm / dist_cm
                    repulsion = 400.0 * (2 * r_eff - dist_cm) * n
                    self.v[idx1_s:idx1_e] -= (repulsion / self.mass[idx1_s:idx1_e, None]) * dt
                    self.v[idx2_s:idx2_e] += (repulsion / self.mass[idx2_s:idx2_e, None]) * dt
                    inter_forces += repulsion

        # 4. Integrate positions
        self.x += self.v * dt

        # 5. Boundary and kinematic pusher collisions
        ext_f = self._apply_boundary_and_plate_collision(self.current_time, dt)

        # Budget updates
        self.cumulative_impulse += ext_f * dt
        self.external_force_record[:, 0, :] = ext_f
        self.interaction_force_record[0, :] = inter_forces

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step physics up to the requested sample time
        while self.current_time < time - 1e-7:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)

        # Compute per-element deformation gradient
        # 5 elements (one per soft body), shape [5, 3, 3]
        def_grad = np.zeros((self.num_bodies, 3, 3), dtype=float)
        for b in range(self.num_bodies):
            idx_s = b * self.n_nodes_per_body
            idx_e = idx_s + self.n_nodes_per_body
            bx = self.x[idx_s:idx_e]
            rx0 = self.rest_x[idx_s:idx_e]
            bm = self.mass[idx_s:idx_e]
            m_tot = np.sum(bm)
            cm_curr = np.sum(bx * bm[:, None], axis=0) / m_tot
            cm_rest = np.sum(rx0 * bm[:, None], axis=0) / m_tot
            q = rx0 - cm_rest
            p = bx - cm_curr
            A_pq = p.T @ (q * bm[:, None])
            A_qq = q.T @ (q * bm[:, None])
            try:
                F = A_pq @ np.linalg.inv(A_qq)
            except Exception:
                F = np.eye(3)
            def_grad[b] = F

        # Budget domain weights [6, 60]
        domain_weights = np.zeros((self.G, self.total_nodes), dtype=float)
        for b in range(self.num_bodies):
            domain_weights[b, b * self.n_nodes_per_body:(b + 1) * self.n_nodes_per_body] = 1.0
        domain_weights[5, :] = 1.0  # Global domain

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow.copy(),
                "budget_domain_weights": domain_weights.copy(),
                "budget_external_force": self.external_force_record.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "element_deformation_gradient": def_grad.copy(),
                "interaction_force": self.interaction_force_record.copy(),
            },
            scalars={
                "reference_momentum": self.ref_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_bodies": self.num_bodies,
            "total_nodes": self.total_nodes,
            "sim_time": self.current_time,
        }

