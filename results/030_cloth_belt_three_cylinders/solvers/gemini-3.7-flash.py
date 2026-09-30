"""Cloth belt three cylinders simulation solver."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Initial state from scene
        frame = scene.initial_state()
        self.current_time = float(frame.time)
        self.pos = frame.position.copy()
        self.vel = frame.velocity.copy()
        self.orientation = frame.orientation.copy()
        self.angular_velocity = frame.angular_velocity.copy()
        self.mass = frame.mass.copy()
        self.body_id = frame.body_id.copy()
        self.uv = frame.attributes["uv"].copy() if "uv" in frame.attributes else None

        self.cloth_mask = (self.body_id == 0)
        self.num_cloth_nodes = int(np.sum(self.cloth_mask))
        self.num_nodes = len(self.pos)

        # Grid parameters
        self.ring_count = 120
        self.width_nodes = 9

        # Cylinders geometry & config
        cyl_cfg = self.config["cylinders"]
        self.cyl_radius = float(cyl_cfg["radius"])
        self.cyl_length = float(cyl_cfg["length"])
        self.init_centers = np.asarray(cyl_cfg["initial_centers_xz"], dtype=float) # [3, 2]
        self.final_centers = np.asarray(cyl_cfg["final_centers_xz"], dtype=float) # [3, 2]
        self.outer_angular_speed = float(cyl_cfg["outer_angular_speed"])
        self.outer_rotation_signs = np.asarray(cyl_cfg["outer_rotation_signs"], dtype=float)

        # Timeline
        self.timeline = self.config["timeline"]

        # Physical parameters
        belt_cfg = self.config["belt"]
        self.thickness = float(belt_cfg["thickness"])
        self.contact_radius = self.cyl_radius + self.thickness * 0.5
        self.coulomb_mu = float(self.config["friction"]["coefficient"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Cloth spring mesh construction (rings x width)
        # nodes are ordered i * width_nodes + j
        cloth_pos = self.pos[:self.num_cloth_nodes]

        # Structural, shear, and bending springs
        springs = []
        rest_lengths = []

        for i in range(self.ring_count):
            next_i = (i + 1) % self.ring_count
            for j in range(self.width_nodes):
                idx = i * self.width_nodes + j
                # Ring longitudinal spring
                idx_next_ring = next_i * self.width_nodes + j
                l0 = np.linalg.norm(cloth_pos[idx] - cloth_pos[idx_next_ring])
                springs.append((idx, idx_next_ring, 1)) # type 1: longitudinal
                rest_lengths.append(l0)

                # Lateral spring
                if j + 1 < self.width_nodes:
                    idx_next_w = i * self.width_nodes + (j + 1)
                    l0_lat = np.linalg.norm(cloth_pos[idx] - cloth_pos[idx_next_w])
                    springs.append((idx, idx_next_w, 2)) # type 2: lateral
                    rest_lengths.append(l0_lat)

                    # Shear springs
                    idx_diag1 = next_i * self.width_nodes + (j + 1)
                    l0_d1 = np.linalg.norm(cloth_pos[idx] - cloth_pos[idx_diag1])
                    springs.append((idx, idx_diag1, 3)) # type 3: shear
                    rest_lengths.append(l0_d1)

                    idx_diag2 = next_i * self.width_nodes + j
                    idx_cur_w = i * self.width_nodes + (j + 1)
                    l0_d2 = np.linalg.norm(cloth_pos[idx_cur_w] - cloth_diag2 if False else cloth_pos[idx_diag2])
                    # connect idx_cur_w and next_i * width_nodes + j
                    l0_d2 = np.linalg.norm(cloth_pos[idx_cur_w] - cloth_pos[next_i * self.width_nodes + j])
                    springs.append((idx_cur_w, next_i * self.width_nodes + j, 3))
                    rest_lengths.append(l0_d2)

                # Bending springs along ring (distance 2)
                next_2_i = (i + 2) % self.ring_count
                idx_bend = next_2_i * self.width_nodes + j
                l0_bend = np.linalg.norm(cloth_pos[idx] - cloth_pos[idx_bend])
                springs.append((idx, idx_bend, 4)) # type 4: bending
                rest_lengths.append(l0_bend)

        self.spring_indices = np.asarray([(s[0], s[1]) for s in springs], dtype=np.int32)
        self.spring_types = np.asarray([s[2] for s in springs], dtype=np.int32)
        self.rest_lengths = np.asarray(rest_lengths, dtype=float)

        # Stiffness values (high tension belt)
        self.k_long = 1.2e5
        self.k_lat = 4.0e4
        self.k_shear = 2.0e4
        self.k_bend = 5.0e2

        # Initial deformation gradients (identity [N, 3, 3])
        self.initial_cloth_pos = cloth_pos.copy()

    def _get_cylinder_state(self, t: float):
        """Returns centers (3, 3), velocities (3, 3), and omega_y (3,) for 3 cylinders."""
        # Timeline intervals
        # lift: [0, 1]
        # tension_hold: [1, 1.4]
        # forward: [1.4, 3.4]
        # pause: [3.4, 3.7]
        # reverse: [3.7, 5.7]
        # settle: [5.7, 6.2]

        t_lift_start, t_lift_end = self.timeline["lift"]
        t_fwd_start, t_fwd_end = self.timeline["forward"]
        t_rev_start, t_rev_end = self.timeline["reverse"]

        # Center positions [left, right, top]
        centers = np.zeros((3, 3))
        velocities = np.zeros((3, 3))

        # Left and right cylinders stay fixed in position
        centers[0, [0, 2]] = self.init_centers[0]
        centers[1, [0, 2]] = self.init_centers[1]

        # Top cylinder lifts from init to final in [t_lift_start, t_lift_end]
        if t <= t_lift_start:
            centers[2, [0, 2]] = self.init_centers[2]
            velocities[2, 2] = 0.0
        elif t < t_lift_end:
            alpha = (t - t_lift_start) / (t_lift_end - t_lift_start)
            # Smooth step (smooth hermite)
            s = 3 * alpha**2 - 2 * alpha**3
            ds = (6 * alpha - 6 * alpha**2) / (t_lift_end - t_lift_start)
            centers[2, [0, 2]] = (1.0 - s) * self.init_centers[2] + s * self.final_centers[2]
            velocities[2, [0, 2]] = ds * (self.final_centers[2] - self.init_centers[2])
        else:
            centers[2, [0, 2]] = self.final_centers[2]
            velocities[2, 2] = 0.0

        # Angular velocities around y-axis (axis = [0, 1, 0])
        omega_y = np.zeros(3) # [left, right, top]

        # Forward rotation [1.4, 3.4]
        if t_fwd_start <= t < t_fwd_end:
            # outer_rotation_signs: [left, right] = [1.0, -1.0]
            # outer_angular_speed: pi rad/s (1 rev in 2s)
            omega_y[0] = self.outer_rotation_signs[0] * self.outer_angular_speed
            omega_y[1] = self.outer_rotation_signs[1] * self.outer_angular_speed
        elif t_rev_start <= t < t_rev_end:
            # reverse rotation
            omega_y[0] = -self.outer_rotation_signs[0] * self.outer_angular_speed
            omega_y[1] = -self.outer_rotation_signs[1] * self.outer_angular_speed

        return centers, velocities, omega_y

    def _step(self, dt: float):
        """PBD / XPBD step for cloth with frictional cylinder contact."""
        N_cloth = self.num_cloth_nodes
        x = self.pos[:N_cloth].copy()
        v = self.vel[:N_cloth].copy()
        m = self.mass[:N_cloth]
        inv_m = np.where(m > 0, 1.0 / m, 0.0)

        # External forces (gravity) + damping
        v += self.gravity * dt
        v *= (1.0 - 0.05 * dt) # gentle air drag

        # Predict position
        p = x + v * dt

        # Cylinder kinematic state at t + dt
        t_next = self.current_time + dt
        cyl_centers, cyl_vels, cyl_omegas = self._get_cylinder_state(t_next)

        # Update cylinder body positions in state
        self.pos[N_cloth:N_cloth+3] = cyl_centers
        self.vel[N_cloth:N_cloth+3] = cyl_vels
        self.angular_velocity[N_cloth:N_cloth+3, 1] = cyl_omegas

        # XPBD / PBD constraints iterations
        num_subiters = 15
        i1 = self.spring_indices[:, 0]
        i2 = self.spring_indices[:, 1]
        l0 = self.rest_lengths

        # Precompute compliance alpha = 1 / (k * dt^2)
        stiffness = np.where(self.spring_types == 1, self.k_long,
                    np.where(self.spring_types == 2, self.k_lat,
                    np.where(self.spring_types == 3, self.k_shear, self.k_bend)))

        alpha = 1.0 / (stiffness * (dt ** 2))
        lagrange_mults = np.zeros(len(self.spring_indices))

        for sub in range(num_subiters):
            # 1. Distance constraints
            dp = p[i1] - p[i2]
            dist = np.linalg.norm(dp, axis=-1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-8)
            n_dir = dp / dist_safe

            C = dist.squeeze(-1) - l0
            inv_m_sum = inv_m[i1] + inv_m[i2]
            d_lambda = (-C - alpha * lagrange_mults) / (inv_m_sum + alpha)
            lagrange_mults += d_lambda

            corr = (d_lambda[:, None] * n_dir)
            np.add.at(p, i1, corr * inv_m[i1, None])
            np.add.at(p, i2, -corr * inv_m[i2, None])

            # Width constraint: keep width within width bounds
            p[:, 1] = np.clip(p[:, 1], -self.config["belt"]["width"] * 0.5, self.config["belt"]["width"] * 0.5)

            # 2. Collision with 3 cylinders
            for c_idx in range(3):
                c_pos = cyl_centers[c_idx] # [x, y, z]
                r_c = self.contact_radius

                # Vector in x-z plane from cylinder center
                dx = p[:, 0] - c_pos[0]
                dz = p[:, 2] - c_pos[2]
                r_xz = np.sqrt(dx**2 + dz**2)
                r_xz_safe = np.maximum(r_xz, 1e-8)

                # Penetration depth: gap = r_xz - r_c
                penetration = r_c - r_xz
                inside = (penetration > 0)

                if np.any(inside):
                    # Push out to cylinder surface
                    nx = dx[inside] / r_xz_safe[inside]
                    nz = dz[inside] / r_xz_safe[inside]

                    p[inside, 0] = c_pos[0] + nx * r_c
                    p[inside, 2] = c_pos[2] + nz * r_c

        # Update velocity & apply friction at contact
        new_v = (p - x) / dt

        # Friction update with cylinders
        for c_idx in range(3):
            c_pos = cyl_centers[c_idx]
            omega = cyl_omegas[c_idx]
            r_c = self.contact_radius

            dx = p[:, 0] - c_pos[0]
            dz = p[:, 2] - c_pos[2]
            r_xz = np.sqrt(dx**2 + dz**2)
            r_xz_safe = np.maximum(r_xz, 1e-8)

            nx = dx / r_xz_safe
            nz = dz / r_xz_safe

            # Contact check (within contact shell)
            gap = r_xz - r_c
            in_contact = (gap <= 0.005)

            if np.any(in_contact):
                # Surface velocity at contact point on cylinder: v_surf = omega x r
                # axis is y: (0, omega, 0) x (nx*r_c, 0, nz*r_c) = (omega * nz * r_c, 0, -omega * nx * r_c)
                v_target_x = omega * nz[in_contact] * r_c
                v_target_y = 0.0
                v_target_z = -omega * nx[in_contact] * r_c

                v_target = np.column_stack([v_target_x, np.zeros_like(v_target_x), v_target_z])

                # Relative velocity to cylinder surface
                v_rel = new_v[in_contact] - v_target

                # Normal component of rel velocity
                n_vec = np.column_stack([nx[in_contact], np.zeros_like(nx[in_contact]), nz[in_contact]])
                v_n_mag = np.sum(v_rel * n_vec, axis=-1, keepdims=True)
                v_n = v_n_mag * n_vec
                v_t = v_rel - v_n

                # Coulomb friction
                v_t_norm = np.linalg.norm(v_t, axis=-1, keepdims=True)
                # Max friction impulse
                friction_factor = np.clip(self.coulomb_mu * 20.0 * dt, 0.0, 1.0)
                new_v[in_contact] -= friction_factor * v_t

        self.vel[:N_cloth] = new_v
        self.pos[:N_cloth] = p
        self.current_time = t_next

    def advance_to(self, target_time: float) -> StateFrame:
        # Internal substeps for stability
        internal_dt = 0.001

        while self.current_time < target_time - 1e-7:
            dt = min(internal_dt, target_time - self.current_time)
            self._step(dt)

        # Compute required StateFrame fields
        # 1. Signed contact distance C = [N_cloth * 3]
        cloth_p = self.pos[:self.num_cloth_nodes]
        cyl_centers = self.pos[self.num_cloth_nodes:self.num_cloth_nodes+3]

        all_contact_sdfs = []
        for c_idx in range(3):
            c_pos = cyl_centers[c_idx]
            dx = cloth_p[:, 0] - c_pos[0]
            dz = cloth_p[:, 2] - c_pos[2]
            r_xz = np.sqrt(dx**2 + dz**2)
            gap = r_xz - self.contact_radius
            all_contact_sdfs.append(gap)

        # Signed distance for all contacts: active / near contacts
        contact_sdf = np.concatenate(all_contact_sdfs)

        # 2. Deformation gradient [N, 3, 3]
        # Approximate per-node deformation gradient F
        # Based on stretch along longitudinal and lateral directions
        N_total = len(self.pos)
        deformation_gradient = np.tile(np.eye(3), (N_total, 1, 1))

        for i in range(self.ring_count):
            next_i = (i + 1) % self.ring_count
            prev_i = (i - 1) % self.ring_count
            for j in range(self.width_nodes):
                idx = i * self.width_nodes + j
                idx_next = next_i * self.width_nodes + j
                idx_prev = prev_i * self.width_nodes + j

                # Current tangent along ring
                t_curr = (cloth_p[idx_next] - cloth_p[idx_prev])
                t_curr_len = np.linalg.norm(t_curr)

                t_init = (self.initial_cloth_pos[idx_next] - self.initial_cloth_pos[idx_prev])
                t_init_len = np.linalg.norm(t_init)

                stretch = t_curr_len / max(t_init_len, 1e-6)
                deformation_gradient[idx, 0, 0] = stretch
                deformation_gradient[idx, 1, 1] = 1.0
                deformation_gradient[idx, 2, 2] = 1.0 / max(stretch, 1e-3)

        # Build frame
        attributes = {
            "deformation_gradient": deformation_gradient,
            "mass_quantity": self.mass.copy(),
        }
        if self.uv is not None:
            attributes["uv"] = self.uv.copy()

        fields = {
            "contact_signed_distance": contact_sdf,
            "cumulative_mass_source": np.zeros(1),
        }

        scalars = {
            "contact_length_scale": float(self.thickness),
        }

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"current_time": self.current_time}

