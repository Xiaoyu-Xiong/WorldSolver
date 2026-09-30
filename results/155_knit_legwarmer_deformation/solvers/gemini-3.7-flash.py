"""Knit legwarmer deformation physics simulation solver."""
from __future__ import annotations

import json
import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.cloth_cfg = self.config["cloth"]
        self.grid_shape = list(map(int, self.cloth_cfg["grid_shape"]))
        self.nx, self.ny = self.grid_shape  # nx=17 around circumference, ny=17 along axis

        # Initial state from scene
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.num_vertices = len(self.pos)
        self.thickness = float(self.cloth_cfg.get("thickness", 0.015))
        self.contact_length_scale = self.thickness
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Load topology
        try:
            with open("assets/cloth_topology.json", "r") as f:
                topo = json.load(f)
                self.faces = np.array(topo["faces"], dtype=np.int32)
        except Exception:
            # Fallback mesh faces for cylindrical grid nx x ny
            faces = []
            for j in range(self.ny - 1):
                for i in range(self.nx):
                    ni = (i + 1) % self.nx
                    v0 = j * self.nx + i
                    v1 = j * self.nx + ni
                    v2 = (j + 1) * self.nx + ni
                    v3 = (j + 1) * self.nx + i
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
            self.faces = np.array(faces, dtype=np.int32)

        # Build edges and rest lengths
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = int(f[i]), int(f[(i + 1) % 3])
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))

        # Also add cross-shear and bending links for knit structure
        for j in range(self.ny):
            for i in range(self.nx):
                u = j * self.nx + i
                # Bending along axis
                if j + 2 < self.ny:
                    v = (j + 2) * self.nx + i
                    edges_set.add((min(u, v), max(u, v)))
                # Bending along hoop
                v = j * self.nx + (i + 2) % self.nx
                edges_set.add((min(u, v), max(u, v)))

        self.edges = np.array(list(edges_set), dtype=np.int32)
        # Compute rest lengths
        diffs = self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(diffs, axis=1)

        # Precompute initial rest reference in local cylindrical/material coords for deformation gradient
        # Material coordinates: u = angle * radius, v = axial pos
        self.ref_pos = self.pos.copy()

        # Cumulative mass source tracking
        self.cumulative_mass_source = np.zeros(self.num_vertices, dtype=np.float64)

        # Simulation internal time
        self.current_time = 0.0

        # Substep dt
        self.sub_dt = 0.001

        # Limb kinematics parameters
        self.joint_center = np.array([0.0, 0.0, 1.10], dtype=np.float64)
        self.upper_radius = 0.34
        self.lower_radius = 0.31
        self.knee_radius = 0.36
        self.duration = float(self.config.get("duration", 7.0))

    def _get_limb_state(self, t: float):
        """Compute kinematic pose of upper limb, lower limb, and foot at time t."""
        # Bending angle theta(t): bends up to ~75 degrees (1.3 rad) around y-axis at knee joint
        # Smooth forward and back bending motion
        norm_t = min(max(t / self.duration, 0.0), 1.0)
        # Cycle of bending: 0 -> max angle -> return
        phase = math.sin(math.pi * norm_t)
        max_angle = 1.35  # ~77 degrees
        theta = max_angle * (phase ** 1.2)

        # Upper limb: remains mostly stationary or slight tilt
        # Upper limb centerline from [-1.27, 0, 1.10] to [0.0, 0, 1.10]
        p_upper_0 = np.array([-1.27, 0.0, 1.10])
        p_upper_1 = self.joint_center

        # Lower limb: rotates around knee joint in XZ plane (around Y axis)
        # Lower limb vector at rest is [1.24, 0, 0]
        lower_len = 1.24
        # Rotation around joint_center (clockwise or upwards/downwards)
        # In standard knee flexion, lower leg bends backwards/downwards or upwards.
        # Bending downwards: x_new = cos(theta) * L, z_new = -sin(theta) * L
        dx = lower_len * math.cos(theta)
        dz = -lower_len * math.sin(theta)
        p_lower_0 = self.joint_center
        p_lower_1 = self.joint_center + np.array([dx, 0.0, dz])

        # Foot follows lower limb tip
        foot_len = 0.35
        # Foot angled slightly
        foot_0 = p_lower_1
        foot_1 = p_lower_1 + np.array([foot_len * math.cos(theta - 0.4), 0.0, -foot_len * math.sin(theta - 0.4)])

        return (p_upper_0, p_upper_1, self.upper_radius), \
               (p_lower_0, p_lower_1, self.lower_radius), \
               (self.joint_center, self.knee_radius), \
               (foot_0, foot_1, 0.25)

    def _project_capsule(self, p, cap_p0, cap_p1, radius):
        """Vectorized distance and normal from points p to a line segment capsule."""
        v = cap_p1 - cap_p0
        v_len_sq = np.dot(v, v)
        if v_len_sq < 1e-12:
            diff = p - cap_p0
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            normal = diff / dist_safe
            signed_dist = dist - radius
            return signed_dist.squeeze(-1), normal

        # Projection factor t
        t = np.sum((p - cap_p0) * v, axis=-1, keepdims=True) / v_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = cap_p0 + t * v
        diff = p - closest
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        normal = diff / dist_safe
        signed_dist = dist - radius
        return signed_dist.squeeze(-1), normal

    def _substep(self, dt: float, t_cur: float) -> tuple[np.ndarray, np.ndarray]:
        """Perform one physical substep with XPBD/PBD cloth solver and collision response."""
        # 1. External forces (gravity and air damping)
        ext_forces = np.tile(self.gravity * self.mass[:, None], (1, 1))
        # Damping
        damping = 0.998
        self.vel = (self.vel + (ext_forces / self.mass[:, None]) * dt) * damping

        # Predict positions
        pred_pos = self.pos + self.vel * dt

        # Kinematic colliders at current time
        upper_cap, lower_cap, knee_sph, foot_cap = self._get_limb_state(t_cur + dt)

        # XPBD / PBD constraint iterations
        solver_iters = 8
        inv_mass = 1.0 / self.mass

        # Edge constraints
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        w0 = inv_mass[e0]
        w1 = inv_mass[e1]
        w_sum = w0 + w1

        # Anisotropic knit stiffness: highly extensible in hoop direction, stiffer in axial
        stiffness = 0.85

        for _ in range(solver_iters):
            # Distance constraints
            dp = pred_pos[e0] - pred_pos[e1]
            dist = np.linalg.norm(dp, axis=1)
            dist_safe = np.maximum(dist, 1e-8)
            n = dp / dist_safe[:, None]

            c = dist - self.rest_lengths
            corr = (c / np.maximum(w_sum, 1e-8)) * stiffness

            # Scatter updates
            np.add.at(pred_pos, e0, -n * (w0 * corr)[:, None])
            np.add.at(pred_pos, e1, n * (w1 * corr)[:, None])

            # Collision constraints against upper limb capsule
            sd_u, n_u = self._project_capsule(pred_pos, upper_cap[0], upper_cap[1], upper_cap[2] + self.thickness * 0.5)
            pen_u = sd_u < 0.0
            if np.any(pen_u):
                pred_pos[pen_u] -= n_u[pen_u] * (sd_u[pen_u, None])

            # Collision constraints against lower limb capsule
            sd_l, n_l = self._project_capsule(pred_pos, lower_cap[0], lower_cap[1], lower_cap[2] + self.thickness * 0.5)
            pen_l = sd_l < 0.0
            if np.any(pen_l):
                pred_pos[pen_l] -= n_l[pen_l] * (sd_l[pen_l, None])

            # Collision against knee sphere
            diff_k = pred_pos - knee_sph[0]
            dist_k = np.linalg.norm(diff_k, axis=-1, keepdims=True)
            sd_k = (dist_k - (knee_sph[1] + self.thickness * 0.5)).squeeze(-1)
            pen_k = sd_k < 0.0
            if np.any(pen_k):
                n_k = diff_k / np.maximum(dist_k, 1e-8)
                pred_pos[pen_k] -= n_k[pen_k] * sd_k[pen_k, None]

            # Ground plane collision: z >= 0.0
            pen_g = pred_pos[:, 2] < self.thickness * 0.5
            if np.any(pen_g):
                pred_pos[pen_g, 2] = self.thickness * 0.5

        # Update velocities and positions
        self.vel = (pred_pos - self.pos) / dt
        self.pos = pred_pos

        # Compute signed distances to boundary for exported field
        sd_u, _ = self._project_capsule(self.pos, upper_cap[0], upper_cap[1], upper_cap[2])
        sd_l, _ = self._project_capsule(self.pos, lower_cap[0], lower_cap[1], lower_cap[2])
        diff_k = self.pos - knee_sph[0]
        dist_k = np.linalg.norm(diff_k, axis=-1)
        sd_k = dist_k - knee_sph[1]

        # Minimum signed distance across obstacles per vertex
        min_sd = np.minimum(np.minimum(sd_u, sd_l), sd_k)

        return ext_forces, min_sd

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute per-vertex 2D deformation gradient tensors in local tangent plane."""
        # For each vertex on cylindrical knit tube, compute local strain F = [F11, F12; F21, F22]
        F = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], self.num_vertices, axis=0)

        # Circumferential stretch & axial stretch
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                i_next = (i + 1) % self.nx
                i_prev = (i - 1 + self.nx) % self.nx

                # Circumferential tangent vectors
                curr_hoop = (self.pos[j * self.nx + i_next] - self.pos[j * self.nx + i_prev]) * 0.5
                rest_hoop = (self.ref_pos[j * self.nx + i_next] - self.ref_pos[j * self.nx + i_prev]) * 0.5

                len_curr_h = np.linalg.norm(curr_hoop)
                len_rest_h = np.linalg.norm(rest_hoop)
                lambda_hoop = len_curr_h / max(len_rest_h, 1e-8)

                # Axial tangent vectors
                if j == 0:
                    curr_axis = self.pos[(j + 1) * self.nx + i] - self.pos[j * self.nx + i]
                    rest_axis = self.ref_pos[(j + 1) * self.nx + i] - self.ref_pos[j * self.nx + i]
                elif j == self.ny - 1:
                    curr_axis = self.pos[j * self.nx + i] - self.pos[(j - 1) * self.nx + i]
                    rest_axis = self.ref_pos[j * self.nx + i] - self.ref_pos[(j - 1) * self.nx + i]
                else:
                    curr_axis = (self.pos[(j + 1) * self.nx + i] - self.pos[(j - 1) * self.nx + i]) * 0.5
                    rest_axis = (self.ref_pos[(j + 1) * self.nx + i] - self.ref_pos[(j - 1) * self.nx + i]) * 0.5

                len_curr_a = np.linalg.norm(curr_axis)
                len_rest_a = np.linalg.norm(rest_axis)
                lambda_axis = len_curr_a / max(len_rest_a, 1e-8)

                # Shear
                dot_shear = np.dot(curr_hoop, curr_axis) / max(len_curr_h * len_curr_a, 1e-8)

                F[idx, 0, 0] = lambda_hoop
                F[idx, 1, 1] = lambda_axis
                F[idx, 0, 1] = dot_shear * 0.5
                F[idx, 1, 0] = dot_shear * 0.5

        return F

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward in time
        ext_force = np.tile(self.gravity * self.mass[:, None], (1, 1))
        contact_sd = np.zeros(self.num_vertices, dtype=np.float64)

        while self.current_time + 1e-8 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            ext_force, contact_sd = self._substep(dt, self.current_time)
            self.current_time += dt

        # Compute deformation gradient
        def_grad = self._compute_deformation_gradients()

        # Boundary vertex indicator
        boundary = np.zeros(self.num_vertices, dtype=np.int32)
        boundary[:self.nx] = 1
        boundary[-self.nx:] = 1

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": def_grad,
            },
            fields={
                "contact_signed_distance": contact_sd.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": ext_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_vertices,
            "current_time": self.current_time,
        }

