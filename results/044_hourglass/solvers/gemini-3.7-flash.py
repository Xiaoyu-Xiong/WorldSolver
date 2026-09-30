"""Hourglass physics simulation solver implementing MPM / Granular particle dynamics."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Get initial state from scene
        init_frame = scene.initial_state()
        self.x = np.asarray(init_frame.position, dtype=np.float64).copy()
        self.v = np.asarray(init_frame.velocity, dtype=np.float64).copy()
        self.m = np.asarray(init_frame.mass, dtype=np.float64).copy()
        self.N = len(self.x)
        self.particle_radius = float(self.config["initial_conditions"]["particle_radius"])
        self.grain_volume = 4.0 * np.pi * self.particle_radius**3 / 3.0
        self.initial_bulk_particle_volume = float(self.N * self.grain_volume)

        # Gravity
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Container inner profile definition (local coordinates of hourglass)
        # Profile points: [r, z]
        self.profile = np.array([
            [0.46, 0.02],
            [0.44, 0.14],
            [0.32, 0.36],
            [0.12, 0.60],
            [0.038, 0.72],
            [0.12, 0.84],
            [0.32, 1.08],
            [0.44, 1.30],
            [0.46, 1.42]
        ], dtype=np.float64)

        # Scale and translation of hourglass shell
        self.shell_scale = np.array([1.0, 0.16, 1.0], dtype=np.float64)
        self.shell_trans = np.array([0.0, 0.0, 0.08], dtype=np.float64)
        self.slab_half_depth = float(self.config["boundary_conditions"].get("slab_half_depth", 0.055))

        # Center of rotation for the hourglass
        # Profile ranges z from 0.02 to 1.42, neck is at z=0.72.
        # Shell trans z is 0.08, so world neck z = 0.72 + 0.08 = 0.80.
        self.center_of_rot = np.array([0.0, 0.0, 0.80], dtype=np.float64)

        # Time management
        self.current_time = 0.0
        self.sub_dt = 0.002  # 2 ms substep for stability and real-time execution

        # Cached values for trajectory contract
        self.contact_signed_dist = np.zeros(self.N, dtype=np.float64)
        self.ext_force = np.zeros((self.N, 3), dtype=np.float64)

    def _get_hourglass_rotation(self, t: float) -> tuple[float, float]:
        """Compute rotation angle (radians around +Y) and angular velocity (rad/s) at time t."""
        # Flip 1: t in [6.0, 6.8] -> 180 deg (pi rad)
        # Flip 2: t in [12.0, 12.8] -> another 180 deg (pi rad) -> total 360 deg (2pi rad)
        theta = 0.0
        omega = 0.0

        # Event 1: [6.0, 6.8]
        if t < 6.0:
            theta = 0.0
            omega = 0.0
        elif t <= 6.8:
            tau = (t - 6.0) / 0.8
            # Smooth step: 3 tau^2 - 2 tau^3
            s = 3.0 * (tau**2) - 2.0 * (tau**3)
            ds_dt = (6.0 * tau - 6.0 * (tau**2)) / 0.8
            theta = np.pi * s
            omega = np.pi * ds_dt
        elif t < 12.0:
            theta = np.pi
            omega = 0.0
        elif t <= 12.8:
            tau = (t - 12.0) / 0.8
            s = 3.0 * (tau**2) - 2.0 * (tau**3)
            ds_dt = (6.0 * tau - 6.0 * (tau**2)) / 0.8
            theta = np.pi + np.pi * s
            omega = np.pi * ds_dt
        else:
            theta = 2.0 * np.pi
            omega = 0.0

        return theta, omega

    def _get_hourglass_radius(self, z_local: np.ndarray) -> np.ndarray:
        """Interpolate inner radius r as a function of z_local along the hourglass profile."""
        pz = self.profile[:, 1]
        pr = self.profile[:, 0]
        return np.interp(z_local, pz, pr)

    def _substep(self, dt: float) -> None:
        t = self.current_time
        theta, omega = self._get_hourglass_rotation(t)

        # Rotation matrix R(theta) around Y axis:
        # x_rot = x * cos(theta) + z * sin(theta)
        # z_rot = -x * sin(theta) + z * cos(theta)
        # y_rot = y
        c = math.cos(theta)
        s = math.sin(theta)

        # Compute external gravity force
        self.ext_force[:, :] = self.m[:, None] * self.gravity[None, :]

        # Velocity update from gravity
        self.v += self.gravity * dt

        # Position update
        self.x += self.v * dt

        # -------------------------------------------------------------
        # Spatial hashing for particle-particle granular contact
        # -------------------------------------------------------------
        cell_size = 2.0 * self.particle_radius
        inv_cell = 1.0 / cell_size
        r_eff = 2.0 * self.particle_radius
        k_contact = 8000.0   # Contact stiffness
        d_contact = 40.0     # Contact damping
        mu_friction = 0.45   # Grain-grain friction

        # Spatial hash grid
        grid = {}
        coords = np.floor(self.x * inv_cell).astype(np.int32)
        for i in range(self.N):
            cell_key = (coords[i, 0], coords[i, 1], coords[i, 2])
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(i)

        # Contact pair resolution
        for key, p_indices in grid.items():
            cx, cy, cz = key
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nkey = (cx + dx, cy + dy, cz + dz)
                        if nkey in grid:
                            np_indices = grid[nkey]
                            for i in p_indices:
                                for j in np_indices:
                                    if i < j:
                                        disp = self.x[i] - self.x[j]
                                        dist_sq = np.dot(disp, disp)
                                        if 0.0 < dist_sq < r_eff**2:
                                            dist = math.sqrt(dist_sq)
                                            normal = disp / dist
                                            overlap = r_eff - dist

                                            # Relative velocity
                                            v_rel = self.v[i] - self.v[j]
                                            v_norm = np.dot(v_rel, normal)

                                            # Spring-dashpot normal impulse / displacement correction
                                            # Position-based split impulse
                                            pos_corr = 0.5 * 0.7 * overlap * normal
                                            self.x[i] += pos_corr
                                            self.x[j] -= pos_corr

                                            # Velocity damping / restitution
                                            f_n = - (k_contact * overlap + d_contact * min(0.0, v_norm)) * normal
                                            dv = 0.5 * (1.0 + 0.1) * max(0.0, -v_norm) * normal
                                            self.v[i] += dv
                                            self.v[j] -= dv

                                            # Tangential friction
                                            v_tan = v_rel - np.dot(v_rel, normal) * normal
                                            v_tan_len = np.linalg.norm(v_tan)
                                            if v_tan_len > 1e-6:
                                                friction_impulse = min(mu_friction * max(0.0, -v_norm), v_tan_len)
                                                dv_tan = 0.5 * friction_impulse * (v_tan / v_tan_len)
                                                self.v[i] -= dv_tan
                                                self.v[j] += dv_tan

        # -------------------------------------------------------------
        # Hourglass Boundary Collision & Kinematic Wall Interaction
        # -------------------------------------------------------------
        # Transform particle positions into hourglass local reference frame
        # Rel center:
        p_rel = self.x - self.center_of_rot

        # Local unrotated coords: R(-theta) * p_rel
        x_loc = p_rel[:, 0] * c - p_rel[:, 2] * s
        y_loc = p_rel[:, 1]
        z_loc_world = p_rel[:, 0] * s + p_rel[:, 2] * c
        z_loc = z_loc_world + (self.center_of_rot[2] - self.shell_trans[2])

        # Clamp z_loc to hourglass profile range
        z_min = self.profile[0, 1] + self.particle_radius
        z_max = self.profile[-1, 1] - self.particle_radius

        # Target inner boundary profile radius
        r_inner = self._get_hourglass_radius(np.clip(z_loc, self.profile[0, 1], self.profile[-1, 1]))
        r_limit = np.maximum(0.02, r_inner - self.particle_radius)

        # Thin slab y-confinement
        y_limit = max(0.01, self.slab_half_depth - self.particle_radius)

        # Check distances and project collisions
        r_current = np.abs(x_loc)
        y_current = np.abs(y_loc)

        # Contact signed distance (positive = inside, negative = penetrated)
        dist_x = r_limit - r_current
        dist_y = y_limit - y_current
        dist_z_bot = z_loc - z_min
        dist_z_top = z_max - z_loc

        # Minimum clearance
        min_clearance = np.minimum(np.minimum(dist_x, dist_y), np.minimum(dist_z_bot, dist_z_top))
        self.contact_signed_dist = min_clearance

        # Wall velocity in world coordinates for each particle
        # Local point in world: p = R(theta) * p_local + C
        # Wall velocity: v_wall = omega x (p - C) = (-omega * (z_w - Cz), 0, omega * (x_w - Cx))
        v_wall_x = -omega * p_rel[:, 2]
        v_wall_y = np.zeros(self.N, dtype=np.float64)
        v_wall_z = omega * p_rel[:, 0]
        v_wall = np.column_stack([v_wall_x, v_wall_y, v_wall_z])

        # Wall collisions and projections
        restitution = 0.05
        wall_friction = 0.40

        # 1. Lateral profile collision (x-direction in local frame)
        mask_x = r_current > r_limit
        if np.any(mask_x):
            # Normal in local frame: (sign(x_loc), 0, -dr/dz)
            #dr_dz = np.gradient(self.profile[:, 0], self.profile[:, 1])
            # For simplicity, project x_loc back to r_limit
            sign_x = np.sign(x_loc[mask_x])
            x_loc[mask_x] = sign_x * r_limit[mask_x]

            # Reconstruct world position for these particles
            p_rel_corr_x = x_loc[mask_x] * c + (z_loc[mask_x] - (self.center_of_rot[2] - self.shell_trans[2])) * s
            p_rel_corr_z = -x_loc[mask_x] * s + (z_loc[mask_x] - (self.center_of_rot[2] - self.shell_trans[2])) * c
            self.x[mask_x, 0] = self.center_of_rot[0] + p_rel_corr_x
            self.x[mask_x, 2] = self.center_of_rot[2] + p_rel_corr_z

            # Adjust velocity relative to wall
            rel_v = self.v[mask_x] - v_wall[mask_x]
            # Normal in world frame: local (sign_x, 0, 0) rotated by theta
            norm_world_x = sign_x * c
            norm_world_z = -sign_x * s
            norm_world = np.column_stack([norm_world_x, np.zeros_like(norm_world_x), norm_world_z])

            vn = np.sum(rel_v * norm_world, axis=1, keepdims=True)
            mask_incoming = (vn > 0).ravel()
            if np.any(mask_incoming):
                idx = np.where(mask_x)[0][mask_incoming]
                norm_in = norm_world[mask_incoming]
                vn_in = vn[mask_incoming]
                vt = rel_v[mask_incoming] - vn_in * norm_in

                # Apply restitution and friction
                rel_v_post = -restitution * vn_in * norm_in + (1.0 - wall_friction) * vt
                self.v[idx] = v_wall[idx] + rel_v_post

        # 2. Top and bottom bulb caps collision (z-direction)
        mask_z_bot = z_loc < z_min
        if np.any(mask_z_bot):
            z_loc[mask_z_bot] = z_min
            p_rel_corr_x = x_loc[mask_z_bot] * c + (z_loc[mask_z_bot] - (self.center_of_rot[2] - self.shell_trans[2])) * s
            p_rel_corr_z = -x_loc[mask_z_bot] * s + (z_loc[mask_z_bot] - (self.center_of_rot[2] - self.shell_trans[2])) * c
            self.x[mask_z_bot, 0] = self.center_of_rot[0] + p_rel_corr_x
            self.x[mask_z_bot, 2] = self.center_of_rot[2] + p_rel_corr_z

            rel_v = self.v[mask_z_bot] - v_wall[mask_z_bot]
            norm_world_x = -s
            norm_world_z = c
            norm_world = np.tile([norm_world_x, 0.0, norm_world_z], (np.sum(mask_z_bot), 1))
            vn = np.sum(rel_v * norm_world, axis=1, keepdims=True)
            vt = rel_v - vn * norm_world
            self.v[mask_z_bot] = v_wall[mask_z_bot] + np.maximum(0.0, -restitution * vn) * norm_world + (1.0 - wall_friction) * vt

        mask_z_top = z_loc > z_max
        if np.any(mask_z_top):
            z_loc[mask_z_top] = z_max
            p_rel_corr_x = x_loc[mask_z_top] * c + (z_loc[mask_z_top] - (self.center_of_rot[2] - self.shell_trans[2])) * s
            p_rel_corr_z = -x_loc[mask_z_top] * s + (z_loc[mask_z_top] - (self.center_of_rot[2] - self.shell_trans[2])) * c
            self.x[mask_z_top, 0] = self.center_of_rot[0] + p_rel_corr_x
            self.x[mask_z_top, 2] = self.center_of_rot[2] + p_rel_corr_z

            rel_v = self.v[mask_z_top] - v_wall[mask_z_top]
            norm_world_x = s
            norm_world_z = -c
            norm_world = np.tile([norm_world_x, 0.0, norm_world_z], (np.sum(mask_z_top), 1))
            vn = np.sum(rel_v * norm_world, axis=1, keepdims=True)
            vt = rel_v - vn * norm_world
            self.v[mask_z_top] = v_wall[mask_z_top] + np.maximum(0.0, -restitution * vn) * norm_world + (1.0 - wall_friction) * vt

        # 3. Slab depth collision (y-direction)
        mask_y = y_current > y_limit
        if np.any(mask_y):
            self.x[mask_y, 1] = np.sign(y_loc[mask_y]) * y_limit
            self.v[mask_y, 1] *= -restitution
            self.v[mask_y, 0] *= (1.0 - wall_friction)
            self.v[mask_y, 2] *= (1.0 - wall_friction)

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward in time until requested time is reached
        while self.current_time + 1e-7 < time:
            step = min(self.sub_dt, time - self.current_time)
            self._substep(step)

        # Construct and validate StateFrame with all required trajectory fields
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "particle_radius": np.full(self.N, self.particle_radius, dtype=np.float64),
            },
            fields={
                "contact_signed_distance": self.contact_signed_dist.copy(),
                "cumulative_mass_source": np.array([0.0], dtype=np.float64),
                "external_force": self.ext_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.particle_radius * 2.0),
                "initial_bulk_particle_volume": float(self.initial_bulk_particle_volume),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": self.N,
            "current_time": self.current_time,
            "mean_velocity": float(np.mean(np.linalg.norm(self.v, axis=1))),
        }

