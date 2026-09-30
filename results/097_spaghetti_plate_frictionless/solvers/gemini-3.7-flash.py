"""Physically-based simulation solver for frictionless spaghetti spreading on a plate."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(seed)

        # Initial state from scene
        self.init_frame = scene.initial_state()
        self.pos = np.copy(self.init_frame.position) # (N, 3)
        self.vel = np.copy(self.init_frame.velocity) # (N, 3)
        self.mass = np.copy(self.init_frame.mass)     # (N,)
        self.body_id = np.copy(self.init_frame.body_id)

        self.num_strands = int(self.config["strand_count"]) # 31
        self.pts_per_strand = int(self.config["strand"]["vertex_count"]) # 33
        self.N = len(self.pos) # 1023
        self.total_mass = float(np.sum(self.mass))

        # Strand physical parameters
        self.strand_length = float(self.config["strand"]["length"]) # 1.30
        self.rod_radius = float(self.config["strand"]["radius"]) # 0.010
        self.segment_rest_len = self.strand_length / (self.pts_per_strand - 1)

        self.gravity = np.asarray(self.config["gravity"], dtype=float) # [0, 0, -9.81]

        # Plate profile: [r, z] points
        # From config / scene_geometry:
        # [[0.10, 0.075], [0.88, 0.080], [1.05, 0.135], [1.14, 0.220]]
        self.plate_profile_r = np.array([0.0, 0.10, 0.88, 1.05, 1.14, 1.30])
        self.plate_profile_z = np.array([0.075, 0.075, 0.080, 0.135, 0.220, 0.40])

        # Time tracking
        self.current_time = 0.0
        self.dt = 0.0005 # 0.5 ms substepping for numerical stability

        # Impulse / Budget tracking
        # We define G = 31 domains (one per strand)
        self.G = self.num_strands
        self.K = 1
        self.domain_weights = np.zeros((self.G, self.N), dtype=float)
        for s in range(self.num_strands):
            idx_start = s * self.pts_per_strand
            idx_end = (s + 1) * self.pts_per_strand
            self.domain_weights[s, idx_start:idx_end] = 1.0

        self.cumulative_external_impulse = np.zeros((self.G, 3), dtype=float)
        self.cumulative_inflow_momentum = np.zeros((self.G, 3), dtype=float)
        self.cumulative_outflow_momentum = np.zeros((self.G, 3), dtype=float)

        # Material / bending stiffness parameters
        # Bending and stretching stiffness
        self.k_stretch = 8000.0
        self.k_bend = 0.015
        self.damping = 0.02

        # Scales
        self.ref_velocity = np.sqrt(2.0 * 9.81 * 0.78) # ~3.9 m/s
        self.ref_force = self.total_mass * 9.81 # ~0.0041 * 31 * 9.81 ~ 1.25 N
        self.ref_momentum = self.total_mass * self.ref_velocity

    def _plate_surface_height_and_normal(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Given radial distance r from z-axis, return (z_surface, nr, nz)."""
        # Linear interpolation of floor profile
        z_surf = np.interp(r, self.plate_profile_r, self.plate_profile_z)

        # Derivative dz/dr for normal calculation
        # piecewise slopes:
        slopes = np.diff(self.plate_profile_z) / np.diff(self.plate_profile_r)
        # Find which interval r belongs to
        bin_idx = np.searchsorted(self.plate_profile_r, r) - 1
        bin_idx = np.clip(bin_idx, 0, len(slopes) - 1)
        dz_dr = slopes[bin_idx]

        # Normal vector pointing inward/upward: (-dz/dr * r_hat + 1 * z_hat) normalized
        denom = np.sqrt(dz_dr**2 + 1.0)
        nr = -dz_dr / denom
        nz = 1.0 / denom
        return z_surf, nr, nz

    def _step(self, dt: float) -> None:
        """Physical substep: compute forces, integrate positions, resolve contacts."""
        N = self.N
        pos = self.pos
        vel = self.vel
        m = self.mass

        forces = np.zeros_like(pos)

        # 1. Gravity
        forces += m[:, None] * self.gravity[None, :]

        # 2. Internal Rod Forces (Stretching and Bending) per strand
        for s in range(self.num_strands):
            idx0 = s * self.pts_per_strand
            idx1 = idx0 + self.pts_per_strand
            p = pos[idx0:idx1]
            v = vel[idx0:idx1]

            # Stretch / spring forces along segments
            dp = p[1:] - p[:-1] # (M-1, 3)
            dist = np.linalg.norm(dp, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-8)
            strain = (dist - self.segment_rest_len) / self.segment_rest_len
            direction = dp / dist

            # Relative velocity along segment
            dv = v[1:] - v[:-1]
            v_rel = np.sum(dv * direction, axis=1, keepdims=True)

            f_stretch = (self.k_stretch * strain + 2.0 * v_rel) * direction
            forces[idx0:idx1-1] += f_stretch
            forces[idx0+1:idx1] -= f_stretch

            # Bending resistance (Laplacian / discrete curvature penalty)
            # Straight rod energy ~ sum || p[i+1] - 2*p[i] + p[i-1] ||^2
            # Bending force ~ -k_bend * (p[i+2] - 4*p[i+1] + 6*p[i] - 4*p[i-1] + p[i-2])
            # Or simpler: torque/curvature spring at each interior vertex
            if self.pts_per_strand >= 3:
                # Curvature vector at vertex i: c_i = p[i-1] - 2*p[i] + p[i+1]
                # For a straight rod at rest, rest curvature is 0.
                c = p[:-2] - 2.0 * p[1:-1] + p[2:] # (M-2, 3)
                # Force on p[i-1], p[i], p[i+1]:
                # Energy E = 0.5 * k_bend * sum ||c_i||^2
                # -dE/dp[i-1] += -k_bend * c_i
                # -dE/dp[i]   +=  2 * k_bend * c_i
                # -dE/dp[i+1] += -k_bend * c_i
                forces[idx0:idx1-2] -= self.k_bend * c
                forces[idx0+1:idx1-1] += 2.0 * self.k_bend * c
                forces[idx0+2:idx1] -= self.k_bend * c

        # 3. Predict velocity and position (Symplectic Euler / PBD style)
        vel += (forces / m[:, None]) * dt
        # Light velocity damping for stability
        vel *= (1.0 - self.damping * dt)
        pos += vel * dt

        # 4. Strand-Plate Collision (Analytical Container Boundary)
        r_xy = np.hypot(pos[:, 0], pos[:, 1])
        r_safe = np.maximum(r_xy, 1e-8)
        dir_x = pos[:, 0] / r_safe
        dir_y = pos[:, 1] / r_safe

        z_surf, nr, nz = self._plate_surface_height_and_normal(r_xy)

        # Normal at each point in 3D: (nr * dir_x, nr * dir_y, nz)
        nx = nr * dir_x
        ny = nr * dir_y
        normals = np.column_stack((nx, ny, nz))

        # Distance to surface along surface normal (approximate):
        # z - z_surf along nz
        # penetration occurs when z < z_surf + rod_radius
        penetration = (z_surf + self.rod_radius) - pos[:, 2]
        plate_contact_mask = penetration > 0

        if np.any(plate_contact_mask):
            idx_c = np.where(plate_contact_mask)[0]
            pen = penetration[idx_c]
            norm_c = normals[idx_c]

            # Position correction onto contact surface
            pos[idx_c] += norm_c * pen[:, None]

            # Velocity correction: perfectly frictionless slip
            # Remove normal velocity component into the plate (inelastic or near-inelastic contact)
            vn = np.sum(vel[idx_c] * norm_c, axis=1)
            vn_into_wall = np.minimum(vn, 0.0)

            # Compute impulse applied to strand
            impulse_c = -vn_into_wall[:, None] * m[idx_c, None]
            vel[idx_c] -= vn_into_wall[:, None] * norm_c

            # Accumulate external impulse
            for i_local, idx in enumerate(idx_c):
                s_id = self.body_id[idx]
                self.cumulative_external_impulse[s_id] += impulse_c[i_local]

        # 5. Strand-Strand Contact (Repulsion between strand vertices)
        # 1023 vertices total is small enough (1023*1022/2 ~ 500k pairs) that we can do efficient distance checks
        # or spatial grid with int64 to avoid overflow.
        d_min = 2.0 * self.rod_radius
        cell_size = d_min * 2.0

        # Safe integer cell coordinates
        cell_coords = np.floor(pos / cell_size).astype(np.int64)
        # Positive hash key
        hash_keys = (
            (cell_coords[:, 0] * 73856093) ^
            (cell_coords[:, 1] * 19349663) ^
            (cell_coords[:, 2] * 83492791)
        )

        # Sort vertices by hash
        sort_idx = np.argsort(hash_keys)
        sorted_keys = hash_keys[sort_idx]

        # Find neighboring vertices
        # Check adjacent vertices in sorted array with same cell
        for i in range(len(sort_idx)):
            idx_a = sort_idx[i]
            body_a = self.body_id[idx_a]

            # Check a small window in sorted list
            for j in range(i + 1, min(i + 16, len(sort_idx))):
                if sorted_keys[j] != sorted_keys[i]:
                    break
                idx_b = sort_idx[j]
                body_b = self.body_id[idx_b]

                if body_a == body_b and abs(idx_a - idx_b) <= 2:
                    # Adjacent vertices on same strand, handled by stretch/bend
                    continue

                delta = pos[idx_a] - pos[idx_b]
                dist_sq = np.sum(delta**2)
                if dist_sq < d_min**2 and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    n_ab = delta / dist
                    pen_ab = d_min - dist

                    # Positional push apart
                    corr = 0.5 * pen_ab * n_ab
                    pos[idx_a] += corr
                    pos[idx_b] -= corr

                    # Velocity correction (frictionless normal impulse)
                    v_rel_n = np.dot(vel[idx_a] - vel[idx_b], n_ab)
                    if v_rel_n < 0:
                        vel[idx_a] -= 0.5 * v_rel_n * n_ab
                        vel[idx_b] += 0.5 * v_rel_n * n_ab

        # Also accumulate gravity impulse in cumulative_external_impulse
        for s in range(self.num_strands):
            m_strand = np.sum(m[self.body_id == s])
            self.cumulative_external_impulse[s] += m_strand * self.gravity * dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward in time until requested time
        while self.current_time + 1e-9 < time:
            dt = min(self.dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Prepare StateFrame
        N = self.N
        G = self.G
        K = self.K
        C = 0 # No persistent dry contacts with non-zero friction

        # Budget external forces: gravity per strand
        ext_force = np.zeros((G, K, 3), dtype=float)
        for s in range(G):
            m_strand = np.sum(self.mass[self.body_id == s])
            ext_force[s, 0] = m_strand * self.gravity

        frame = StateFrame(
            time=float(time),
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                **self.init_frame.attributes,
                "mass_quantity": np.copy(self.mass),
            },
            scalars={
                **self.init_frame.scalars,
                "friction_stick_speed_tolerance": 1e-4,
                "reference_force": self.ref_force,
                "reference_momentum": self.ref_momentum,
                "reference_velocity": self.ref_velocity,
            },
            fields={
                "budget_cumulative_external_impulse": np.copy(self.cumulative_external_impulse),
                "budget_cumulative_inflow_momentum": np.copy(self.cumulative_inflow_momentum),
                "budget_cumulative_outflow_momentum": np.copy(self.cumulative_outflow_momentum),
                "budget_domain_weights": np.copy(self.domain_weights),
                "budget_external_force": ext_force,
                "cumulative_mass_source": np.zeros(K, dtype=float),
                "friction_active_mask": np.zeros(C, dtype=bool),
                "friction_dynamic_coefficient": np.zeros(C, dtype=float),
                "friction_normal": np.zeros((C, 3), dtype=float),
                "friction_normal_force": np.zeros(C, dtype=float),
                "friction_relative_velocity": np.zeros((C, 3), dtype=float),
                "friction_static_coefficient": np.zeros(C, dtype=float),
                "friction_tangential_force": np.zeros((C, 3), dtype=float),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_strands": self.num_strands,
            "total_vertices": self.N,
        }

