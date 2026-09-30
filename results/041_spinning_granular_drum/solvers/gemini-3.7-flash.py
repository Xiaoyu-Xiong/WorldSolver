"""Deterministic Discrete Element Method (DEM) / Granular Solver for Spinning Drum."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete Element Method physics solver for a spinning granular drum."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config
        self.init_frame = scene.initial_state()

        # Physics constants from config / initial state
        drum_cfg = self.config["initial_conditions"]["drum"]
        grains_cfg = self.config["initial_conditions"]["grains"]

        self.drum_center = np.asarray(drum_cfg["center"], dtype=np.float64)  # (0, 0, 0.86)
        self.drum_radius = float(drum_cfg["radius"])  # 0.82
        self.drum_depth = float(drum_cfg["depth"])  # 0.42
        self.drum_omega = float(drum_cfg["angular_velocity_rad_s"])  # 1.25 rad/s around y-axis
        self.rotation_axis = np.asarray(drum_cfg["rotation_axis"], dtype=np.float64)  # [0, 1, 0]

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        self.particle_radius = float(grains_cfg["radius"])  # 0.025
        self.particle_density = float(grains_cfg["bulk_density"])  # 1550.0

        # Initial particle states
        self.position = np.copy(self.init_frame.position)  # (N, 3)
        self.velocity = np.copy(self.init_frame.velocity)  # (N, 3)
        self.mass = np.copy(self.init_frame.mass)  # (N,)
        self.body_id = np.copy(self.init_frame.body_id)
        self.attributes = dict(self.init_frame.attributes)
        self.scalars = dict(self.init_frame.scalars)

        self.num_particles = len(self.position)
        self.current_time = 0.0

        # Physical DEM parameters
        # Normal stiffness and damping
        self.kn = 8.0e3  # N/m (contact normal stiffness)
        self.gamma_n = 50.0  # Normal damping
        # Wall parameters
        self.kn_wall = 1.2e4
        self.gamma_n_wall = 60.0
        # Friction coefficients
        self.mu_particle = 0.45  # Grain-grain friction
        self.mu_wall = 0.65  # Drum wall friction to effectively lift grains
        self.kt = 3.0e3  # Tangential contact stiffness
        self.gamma_t = 25.0  # Tangential damping

        # Sub-stepping configuration
        self.sub_dt = 0.0004  # Stable time step for explicit Euler/Verlet DEM
        self.contact_length_scale = 2.0 * self.particle_radius

        # Diagnostics trackers
        self.escaped_count = 0

    def _compute_forces_and_contacts(self):
        """Compute particle-particle contacts, particle-wall contacts, and forces."""
        N = self.num_particles
        pos = self.position
        vel = self.velocity
        mass = self.mass
        r = self.particle_radius
        d_contact = 2.0 * r

        forces = np.zeros_like(pos)
        # Gravity force
        forces += mass[:, None] * self.gravity[None, :]

        # 1. Spatial hashing / grid for particle-particle collisions
        cell_size = d_contact * 1.05
        # Grid bounds
        min_pos = np.array([-self.drum_radius - 0.1, -self.drum_depth / 2.0 - 0.1, self.drum_center[2] - self.drum_radius - 0.1])
        coords = np.floor((pos - min_pos) / cell_size).astype(np.int32)

        # Hash coords
        p1, p2, p3 = 73856093, 19349663, 83492791
        table_size = 2 * N
        hashes = (coords[:, 0] * p1 ^ coords[:, 1] * p2 ^ coords[:, 2] * p3) % table_size

        # Group particles by hash
        sort_idx = np.argsort(hashes)
        sorted_hashes = hashes[sort_idx]
        sorted_pos = pos[sort_idx]
        sorted_vel = vel[sort_idx]

        unique_hashes, head_indices, counts = np.unique(sorted_hashes, return_index=True, return_counts=True)
        hash_to_range = {}
        for h, start, count in zip(unique_hashes, head_indices, counts):
            hash_to_range[h] = (start, count)

        # Contact distances tracking
        contact_sdfs = []

        # Particle-particle contact loop
        # We can also check 27 neighbor cells
        offsets = np.array([
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
            [-1,  0, -1], [-1,  0, 0], [-1,  0, 1],
            [-1,  1, -1], [-1,  1, 0], [-1,  1, 1],
            [ 0, -1, -1], [ 0, -1, 0], [ 0, -1, 1],
            [ 0,  0, -1], [ 0,  0, 0], [ 0,  0, 1],
            [ 0,  1, -1], [ 0,  1, 0], [ 0,  1, 1],
            [ 1, -1, -1], [ 1, -1, 0], [ 1, -1, 1],
            [ 1,  0, -1], [ 1,  0, 0], [ 1,  0, 1],
            [ 1,  1, -1], [ 1,  1, 0], [ 1,  1, 1]
        ], dtype=np.int32)

        # Efficient spatial check
        for i in range(N):
            c = coords[i]
            p_i = pos[i]
            v_i = vel[i]

            for off in offsets:
                nc = c + off
                nh = int((nc[0] * p1 ^ nc[1] * p2 ^ nc[2] * p3) % table_size)
                if nh in hash_to_range:
                    start, count = hash_to_range[nh]
                    for k in range(start, start + count):
                        j = sort_idx[k]
                        if i < j:
                            p_j = pos[j]
                            rel_pos = p_i - p_j
                            dist_sq = rel_pos[0] * rel_pos[0] + rel_pos[1] * rel_pos[1] + rel_pos[2] * rel_pos[2]
                            if dist_sq < d_contact * d_contact:
                                dist = np.sqrt(dist_sq)
                                if dist > 1e-12:
                                    n = rel_pos / dist
                                    overlap = d_contact - dist
                                    contact_sdfs.append(-overlap)

                                    # Relative velocity
                                    v_rel = v_i - vel[j]
                                    v_n = np.dot(v_rel, n)

                                    # Normal contact force (Hertz-like / linear spring-damper)
                                    fn_mag = self.kn * overlap - self.gamma_n * v_n
                                    if fn_mag < 0.0:
                                        fn_mag = 0.0

                                    fn = fn_mag * n

                                    # Tangential friction
                                    v_t = v_rel - v_n * n
                                    v_t_norm = np.linalg.norm(v_t)
                                    if v_t_norm > 1e-8:
                                        t_dir = v_t / v_t_norm
                                        # Coulomb limit
                                        ft_mag = min(self.mu_particle * fn_mag, self.gamma_t * v_t_norm)
                                        ft = -ft_mag * t_dir
                                    else:
                                        ft = np.zeros(3)

                                    total_pair_f = fn + ft
                                    forces[i] += total_pair_f
                                    forces[j] -= total_pair_f

        # 2. Particle-drum boundary collisions
        # Drum cylinder axis is parallel to Y, centered at (drum_center[0], 0, drum_center[2])
        # Outer cylinder radius = drum_radius
        # Depth bounds: y in [-drum_depth / 2, drum_depth / 2]
        cx, cz = self.drum_center[0], self.drum_center[2]
        y_min = -0.5 * self.drum_depth
        y_max = 0.5 * self.drum_depth
        r_drum = self.drum_radius

        rx = pos[:, 0] - cx
        rz = pos[:, 2] - cz
        r_radial = np.sqrt(rx * rx + rz * rz)
        r_eff = r_drum - r

        # Radial penetration into cylinder wall
        rad_penetration = r_radial - r_eff  # positive means penetrating cylinder wall
        rad_mask = rad_penetration > 0.0

        for idx in np.where(rad_mask)[0]:
            rad_dist = r_radial[idx]
            if rad_dist > 1e-8:
                # Inward normal from wall to particle center
                n_radial = np.array([-rx[idx] / rad_dist, 0.0, -rz[idx] / rad_dist])
                overlap = rad_penetration[idx]
                contact_sdfs.append(-overlap)

                # Drum wall tangential velocity at this contact point
                # Drum rotates about Y axis with angular velocity omega: v_wall = omega x r_wall
                # omega vector = (0, omega, 0)
                # r_wall = (rx[idx], 0, rz[idx])
                # omega x r = (omega * rz, 0, -omega * rx)
                # For positive omega around +Y, rotation in X-Z plane:
                # x_dot = omega * rz, z_dot = -omega * rx
                v_wall = np.array([self.drum_omega * rz[idx], 0.0, -self.drum_omega * rx[idx]])

                # Relative velocity between particle and wall
                v_rel = vel[idx] - v_wall
                v_n = np.dot(v_rel, n_radial)

                # Normal contact force (pushes particle inward)
                fn_mag = self.kn_wall * overlap - self.gamma_n_wall * v_n
                if fn_mag < 0.0:
                    fn_mag = 0.0

                fn = fn_mag * n_radial

                # Tangential relative velocity
                v_t = v_rel - v_n * n_radial
                v_t_norm = np.linalg.norm(v_t)
                if v_t_norm > 1e-8:
                    t_dir = v_t / v_t_norm
                    ft_mag = min(self.mu_wall * fn_mag, self.gamma_t * v_t_norm)
                    ft = -ft_mag * t_dir
                else:
                    ft = np.zeros(3)

                forces[idx] += (fn + ft)

        # Flat side walls (Y bounds)
        # Back wall (y = y_max) and front wall (y = y_min)
        # Back wall contact:
        back_penetration = (pos[:, 1] + r) - y_max
        back_mask = back_penetration > 0.0
        for idx in np.where(back_mask)[0]:
            overlap = back_penetration[idx]
            contact_sdfs.append(-overlap)
            n_wall = np.array([0.0, -1.0, 0.0])
            # Wall rotation at back wall
            v_wall = np.array([self.drum_omega * rz[idx], 0.0, -self.drum_omega * rx[idx]])
            v_rel = vel[idx] - v_wall
            v_n = np.dot(v_rel, n_wall)
            fn_mag = max(0.0, self.kn_wall * overlap - self.gamma_n_wall * v_n)
            fn = fn_mag * n_wall

            v_t = v_rel - v_n * n_wall
            v_t_norm = np.linalg.norm(v_t)
            if v_t_norm > 1e-8:
                t_dir = v_t / v_t_norm
                ft_mag = min(self.mu_wall * fn_mag, self.gamma_t * v_t_norm)
                ft = -ft_mag * t_dir
            else:
                ft = np.zeros(3)
            forces[idx] += (fn + ft)

        # Front wall (y = y_min):
        front_penetration = (y_min + r) - pos[:, 1]
        front_mask = front_penetration > 0.0
        for idx in np.where(front_mask)[0]:
            overlap = front_penetration[idx]
            contact_sdfs.append(-overlap)
            n_wall = np.array([0.0, 1.0, 0.0])
            v_rel = vel[idx]
            v_n = np.dot(v_rel, n_wall)
            fn_mag = max(0.0, self.kn_wall * overlap - self.gamma_n_wall * v_n)
            fn = fn_mag * n_wall

            v_t = v_rel - v_n * n_wall
            v_t_norm = np.linalg.norm(v_t)
            if v_t_norm > 1e-8:
                t_dir = v_t / v_t_norm
                ft_mag = min(0.3 * fn_mag, self.gamma_t * v_t_norm)
                ft = -ft_mag * t_dir
            else:
                ft = np.zeros(3)
            forces[idx] += (fn + ft)

        # Ensure contact signed distance is non-empty
        if len(contact_sdfs) == 0:
            contact_sdfs = [0.0]

        return forces, np.asarray(contact_sdfs, dtype=np.float64)

    def _step(self, dt: float):
        """Advance simulation by dt using semi-implicit Euler integration."""
        forces, sdfs = self._compute_forces_and_contacts()
        inv_mass = 1.0 / self.mass[:, None]
        acc = forces * inv_mass

        # Velocity update
        self.velocity += acc * dt
        # Position update
        self.position += self.velocity * dt

        # Hard constraint enforcement (particles never escape the drum)
        cx, cz = self.drum_center[0], self.drum_center[2]
        r_drum = self.drum_radius - self.particle_radius
        rx = self.position[:, 0] - cx
        rz = self.position[:, 2] - cz
        rad = np.sqrt(rx * rx + rz * rz)
        too_far = rad > r_drum
        if np.any(too_far):
            factor = r_drum / np.maximum(rad[too_far], 1e-8)
            self.position[too_far, 0] = cx + rx[too_far] * factor
            self.position[too_far, 2] = cz + rz[too_far] * factor
            # Eliminate outward velocity component
            nx = rx[too_far] / np.maximum(rad[too_far], 1e-8)
            nz = rz[too_far] / np.maximum(rad[too_far], 1e-8)
            v_out = self.velocity[too_far, 0] * nx + self.velocity[too_far, 2] * nz
            outward_mask = v_out > 0.0
            if np.any(outward_mask):
                idx = np.where(too_far)[0][outward_mask]
                self.velocity[idx, 0] -= v_out[outward_mask] * nx[outward_mask]
                self.velocity[idx, 2] -= v_out[outward_mask] * nz[outward_mask]

        # Enforce Y bounds
        y_min = -0.5 * self.drum_depth + self.particle_radius
        y_max = 0.5 * self.drum_depth - self.particle_radius
        y_low = self.position[:, 1] < y_min
        if np.any(y_low):
            self.position[y_low, 1] = y_min
            self.velocity[y_low, 1] = np.maximum(0.0, self.velocity[y_low, 1])
        y_high = self.position[:, 1] > y_max
        if np.any(y_high):
            self.position[y_high, 1] = y_max
            self.velocity[y_high, 1] = np.minimum(0.0, self.velocity[y_high, 1])

        return sdfs

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to the requested timestamp."""
        # Step forward until we reach target time
        last_sdfs = None
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            last_sdfs = self._step(dt)
            self.current_time += dt

        if last_sdfs is None:
            _, last_sdfs = self._compute_forces_and_contacts()

        # Build external force (gravity)
        external_force = self.mass[:, None] * self.gravity[None, :]

        # Diagnostics calculations
        cx, cz = self.drum_center[0], self.drum_center[2]
        current_angle = self.drum_omega * self.current_time
        lifted_height = float(np.max(self.position[:, 2]))
        # Free surface slope estimate
        # Upper half grains
        upper_mask = self.position[:, 2] > (cz - 0.2)
        if np.sum(upper_mask) > 5:
            upper_x = self.position[upper_mask, 0]
            upper_z = self.position[upper_mask, 2]
            cov = np.cov(upper_x, upper_z)
            slope = float(cov[0, 1] / max(cov[0, 0], 1e-6))
        else:
            slope = 0.0

        # Avalanche speed (mean speed of flowing top layer)
        top_mask = self.position[:, 2] > cz
        if np.sum(top_mask) > 0:
            avalanche_speed = float(np.mean(np.linalg.norm(self.velocity[top_mask], axis=1)))
        else:
            avalanche_speed = 0.0

        scalars = {
            "drum_radius": self.drum_radius,
            "drum_depth": self.drum_depth,
            "drum_angular_velocity_rad_s": self.drum_omega,
            "drum_angle": current_angle,
            "lifted_grain_height": lifted_height,
            "free_surface_slope": slope,
            "avalanche_layer_speed": avalanche_speed,
            "escaped_particle_count": float(self.escaped_count),
            "contact_length_scale": float(self.contact_length_scale),
        }

        # Format attributes and fields
        attributes = dict(self.attributes)
        attributes["mass_quantity"] = np.copy(self.mass)

        fields = {
            "contact_signed_distance": np.asarray(last_sdfs, dtype=np.float64),
            "cumulative_mass_source": np.array([0.0], dtype=np.float64),
            "external_force": np.asarray(external_force, dtype=np.float64),
        }

        frame = StateFrame(
            time=float(time),
            position=np.copy(self.position),
            velocity=np.copy(self.velocity),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.num_particles,
            "escaped_particle_count": self.escaped_count,
        }

