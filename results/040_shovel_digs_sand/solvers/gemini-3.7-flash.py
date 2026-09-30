"""Granular physics solver for shovel digging through sand bed."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors v (shape [..., 3]) by unit quaternion q (shape [4], xyzw)."""
    qx, qy, qz, qw = q
    q_vec = np.array([qx, qy, qz], dtype=np.float64)
    c1 = np.cross(q_vec, v)
    c2 = np.cross(q_vec, c1 + qw * v)
    return v + 2.0 * c2


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion q (xyzw)."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


class BenchmarkSolver:
    """Granular continuum / DEM physics solver for shovel digging through sand."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        init_frame = scene.initial_state()
        self.pos = np.ascontiguousarray(init_frame.position, dtype=np.float64)
        self.n_particles = len(self.pos)
        self.vel = np.zeros_like(self.pos, dtype=np.float64)
        self.mass = np.ascontiguousarray(init_frame.mass, dtype=np.float64)
        self.body_id = np.ones(self.n_particles, dtype=np.int32)

        # Particle physical properties
        self.particle_spacing = float(self.config["initial_conditions"]["sand_particle_spacing"])
        self.particle_radius = 0.5 * self.particle_spacing
        self.particle_volume = self.particle_spacing ** 3
        self.density = float(self.config["initial_conditions"]["sand_bulk_density"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Shovel kinematic keyframes (from config)
        km = self.config["kinematic_motion"]["shovel_assembly"]["translation_keyframes"]
        self.shovel_times = np.array([kf["time"] for kf in km], dtype=np.float64)
        self.shovel_vals = np.array([kf["value"] for kf in km], dtype=np.float64)

        # Shovel blade geometry & initial transform
        self.blade_half_extents = np.array([0.14, 0.64, 0.34], dtype=np.float64)
        self.blade_init_pos = np.array([-2.05, 0.0, 0.38], dtype=np.float64)
        self.blade_quat = np.array([0.0, -0.0872, 0.0, 0.9962], dtype=np.float64)
        self.blade_quat /= np.linalg.norm(self.blade_quat)
        self.blade_quat_inv = quat_conjugate(self.blade_quat)

        # Shovel shaft geometry & initial transform (capsule along local z axis, radius 0.065, height 2.0)
        self.shaft_radius = 0.065
        self.shaft_half_height = 1.0
        self.shaft_init_pos = np.array([-2.76, 0.0, 1.25], dtype=np.float64)
        self.shaft_quat = np.array([0.0, -0.3606, 0.0, 0.9327], dtype=np.float64)
        self.shaft_quat /= np.linalg.norm(self.shaft_quat)
        self.shaft_quat_inv = quat_conjugate(self.shaft_quat)

        # Container boundaries
        self.floor_z = float(self.config["boundary_conditions"]["container_floor_z"])
        inner_bounds = self.config["boundary_conditions"]["container_inner_bounds"]
        self.x_min, self.x_max, self.y_min, self.y_max = inner_bounds

        # Substep timing
        self.current_time = 0.0
        self.substep_dt = 0.005  # 5 ms substep for efficient, accurate integration

        # Track reaction forces
        self.total_shovel_force = np.zeros(3, dtype=np.float64)

    def _get_shovel_pos(self, t: float) -> np.ndarray:
        """PCHIP / smooth monotonic interpolation of shovel translation."""
        if t <= self.shovel_times[0]:
            return self.shovel_vals[0].copy()
        if t >= self.shovel_times[-1]:
            return self.shovel_vals[-1].copy()

        idx = np.searchsorted(self.shovel_times, t) - 1
        idx = max(0, min(idx, len(self.shovel_times) - 2))
        t0, t1 = self.shovel_times[idx], self.shovel_times[idx + 1]
        p0, p1 = self.shovel_vals[idx], self.shovel_vals[idx + 1]

        u = (t - t0) / (t1 - t0)
        s = u * u * (3.0 - 2.0 * u)
        return (1.0 - s) * p0 + s * p1

    def _get_shovel_vel(self, t: float) -> np.ndarray:
        dt = 1e-4
        p_plus = self._get_shovel_pos(t + dt)
        p_minus = self._get_shovel_pos(t - dt)
        return (p_plus - p_minus) / (2.0 * dt)

    def _step(self, dt: float) -> None:
        """Advance physics by one substep dt."""
        t_mid = self.current_time + 0.5 * dt
        shovel_disp = self._get_shovel_pos(t_mid) - self.shovel_vals[0]
        blade_pos = self.blade_init_pos + shovel_disp
        shaft_pos = self.shaft_init_pos + shovel_disp
        shovel_vel = self._get_shovel_vel(t_mid)

        # Gravity
        self.vel += self.gravity * dt

        # Vectorized Particle-Particle Interaction using cKDTree
        r_cut = 2.0 * self.particle_radius
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(r=r_cut, output_type="ndarray")

        acc_forces = np.zeros_like(self.pos, dtype=np.float64)

        if len(pairs) > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            dp = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(dp, axis=1)
            valid = dist > 1e-10
            i = i[valid]
            j = j[valid]
            dp = dp[valid]
            dist = dist[valid]

            n = dp / dist[:, None]
            pen = r_cut - dist

            k_n = 3.5e4
            gamma_n = 25.0
            mu_pp = 0.55

            dv = self.vel[j] - self.vel[i]
            vn = np.sum(dv * n, axis=1)

            fn_mag = np.maximum(0.0, k_n * pen - gamma_n * vn)

            vt = dv - vn[:, None] * n
            vt_mag = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(mu_pp * fn_mag, 15.0 * vt_mag)
            ft = np.zeros_like(vt)
            nonzero_vt = vt_mag > 1e-6
            ft[nonzero_vt] = ft_mag[nonzero_vt, None] * (vt[nonzero_vt] / vt_mag[nonzero_vt, None])

            f_pair = -fn_mag[:, None] * n + ft
            np.add.at(acc_forces, i, f_pair)
            np.add.at(acc_forces, j, -f_pair)

        # Container Boundary Collisions
        k_wall = 5.0e4
        gamma_wall = 35.0
        mu_wall = 0.45

        # Floor (z = floor_z)
        dz_floor = (self.floor_z + self.particle_radius) - self.pos[:, 2]
        floor_mask = dz_floor > 0
        if np.any(floor_mask):
            pen_f = dz_floor[floor_mask]
            acc_forces[floor_mask, 2] += k_wall * pen_f - gamma_wall * self.vel[floor_mask, 2]
            v_horiz = self.vel[floor_mask, :2]
            v_horiz_norm = np.linalg.norm(v_horiz, axis=-1, keepdims=True) + 1e-8
            acc_forces[floor_mask, :2] -= mu_wall * (k_wall * pen_f[:, None]) * (v_horiz / v_horiz_norm)

        # X walls
        dx_left = (self.x_min + self.particle_radius) - self.pos[:, 0]
        left_mask = dx_left > 0
        if np.any(left_mask):
            acc_forces[left_mask, 0] += k_wall * dx_left[left_mask] - gamma_wall * self.vel[left_mask, 0]

        dx_right = self.pos[:, 0] - (self.x_max - self.particle_radius)
        right_mask = dx_right > 0
        if np.any(right_mask):
            acc_forces[right_mask, 0] -= k_wall * dx_right[right_mask] + gamma_wall * self.vel[right_mask, 0]

        # Y walls
        dy_front = (self.y_min + self.particle_radius) - self.pos[:, 1]
        front_mask = dy_front > 0
        if np.any(front_mask):
            acc_forces[front_mask, 1] += k_wall * dy_front[front_mask] - gamma_wall * self.vel[front_mask, 1]

        dy_back = self.pos[:, 1] - (self.y_max - self.particle_radius)
        back_mask = dy_back > 0
        if np.any(back_mask):
            acc_forces[back_mask, 1] -= k_wall * dy_back[back_mask] + gamma_wall * self.vel[back_mask, 1]

        # Shovel Blade Collision (Oriented Box)
        p_rel_blade = self.pos - blade_pos
        p_loc_blade = quat_rotate_vector(self.blade_quat_inv, p_rel_blade)
        v_rel = self.vel - shovel_vel

        hx, hy, hz = self.blade_half_extents
        r_p = self.particle_radius
        dx = np.abs(p_loc_blade[:, 0]) - (hx + r_p)
        dy = np.abs(p_loc_blade[:, 1]) - (hy + r_p)
        dz = np.abs(p_loc_blade[:, 2]) - (hz + r_p)

        blade_inside = (dx < 0) & (dy < 0) & (dz < 0)
        shovel_force_sum = np.zeros(3, dtype=np.float64)

        if np.any(blade_inside):
            pen_x = -dx[blade_inside]
            pen_y = -dy[blade_inside]
            pen_z = -dz[blade_inside]

            min_pen = np.minimum(np.minimum(pen_x, pen_y), pen_z)
            n_loc = np.zeros((np.count_nonzero(blade_inside), 3), dtype=np.float64)

            mask_x = (min_pen == pen_x)
            mask_y = (~mask_x) & (min_pen == pen_y)
            mask_z = (~mask_x) & (~mask_y)

            n_loc[mask_x, 0] = np.sign(p_loc_blade[blade_inside, 0][mask_x])
            n_loc[mask_y, 1] = np.sign(p_loc_blade[blade_inside, 1][mask_y])
            n_loc[mask_z, 2] = np.sign(p_loc_blade[blade_inside, 2][mask_z])

            n_w = quat_rotate_vector(self.blade_quat, n_loc)
            v_rel_c = v_rel[blade_inside]
            vn = np.sum(v_rel_c * n_w, axis=-1, keepdims=True)

            k_blade = 8.0e4
            gamma_blade = 50.0
            f_blade_n = k_blade * min_pen[:, None] - gamma_blade * np.minimum(vn, 0.0)

            vt = v_rel_c - vn * n_w
            vt_n = np.linalg.norm(vt, axis=-1, keepdims=True) + 1e-8
            mu_blade = 0.5
            f_blade_t = -mu_blade * f_blade_n * (vt / vt_n)

            f_blade_tot = f_blade_n * n_w + f_blade_t
            acc_forces[blade_inside] += f_blade_tot
            shovel_force_sum -= np.sum(f_blade_tot, axis=0)

        # Shovel Shaft Collision (Capsule)
        p_rel_shaft = self.pos - shaft_pos
        p_loc_shaft = quat_rotate_vector(self.shaft_quat_inv, p_rel_shaft)
        z_clamped = np.clip(p_loc_shaft[:, 2], -self.shaft_half_height, self.shaft_half_height)
        d_xy = np.linalg.norm(p_loc_shaft[:, :2], axis=-1)
        shaft_pen = (self.shaft_radius + r_p) - d_xy
        shaft_inside = (shaft_pen > 0) & (np.abs(p_loc_shaft[:, 2]) <= self.shaft_half_height + r_p)

        if np.any(shaft_inside):
            pen_s = shaft_pen[shaft_inside]
            n_loc_s = np.zeros((np.count_nonzero(shaft_inside), 3), dtype=np.float64)
            d_xy_val = d_xy[shaft_inside, None] + 1e-8
            n_loc_s[:, :2] = p_loc_shaft[shaft_inside, :2] / d_xy_val
            n_w_s = quat_rotate_vector(self.shaft_quat, n_loc_s)

            v_rel_s = v_rel[shaft_inside]
            vn_s = np.sum(v_rel_s * n_w_s, axis=-1, keepdims=True)
            k_shaft = 6.0e4
            f_shaft_n = k_shaft * pen_s[:, None] - 30.0 * np.minimum(vn_s, 0.0)
            f_shaft_tot = f_shaft_n * n_w_s
            acc_forces[shaft_inside] += f_shaft_tot
            shovel_force_sum -= np.sum(f_shaft_tot, axis=0)

        self.total_shovel_force = shovel_force_sum

        # Update velocities and positions
        acc = acc_forces / self.mass[:, None]
        self.vel += acc * dt
        self.vel *= 0.998  # Rolling resistance & granular cohesion damping
        self.pos += self.vel * dt

        # Floor constraint enforcement
        self.pos[:, 2] = np.maximum(self.pos[:, 2], self.floor_z + 0.1 * self.particle_radius)

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        """Step simulation until requested target time and construct StateFrame."""
        while self.current_time < time - 1e-9:
            dt = min(self.substep_dt, time - self.current_time)
            self._step(dt)

        self.current_time = time

        # Compute signed contact distances
        shovel_disp = self._get_shovel_pos(time) - self.shovel_vals[0]
        blade_pos = self.blade_init_pos + shovel_disp
        p_rel = self.pos - blade_pos
        p_local = quat_rotate_vector(self.blade_quat_inv, p_rel)
        hx, hy, hz = self.blade_half_extents
        dx = np.abs(p_local[:, 0]) - hx
        dy = np.abs(p_local[:, 1]) - hy
        dz = np.abs(p_local[:, 2]) - hz
        outside_dist = np.linalg.norm(np.maximum(np.stack([dx, dy, dz], axis=-1), 0.0), axis=-1)
        inside_dist = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
        box_sd = outside_dist + inside_dist

        floor_sd = self.pos[:, 2] - self.floor_z
        contact_sd = np.minimum(box_sd, floor_sd)

        # External forces
        external_force = self.mass[:, None] * self.gravity[None, :]

        settled_ke = 0.5 * np.sum(self.mass * np.sum(self.vel**2, axis=-1))
        mound_mask = self.pos[:, 0] > 0.0
        mound_height = float(np.max(self.pos[mound_mask, 2])) if np.any(mound_mask) else 0.0

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "initial_volume": np.full(self.n_particles, self.particle_volume),
                "material_label": np.ones(self.n_particles, dtype=np.int32),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.particle_spacing),
                "initial_sand_volume": float(self.n_particles * self.particle_volume),
                "initial_sand_mass": float(self.mass.sum()),
                "settled_kinetic_energy": float(settled_ke),
                "mound_max_height": float(mound_height),
                "shovel_reaction_force_mag": float(np.linalg.norm(self.total_shovel_force)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "shovel_reaction_force": self.total_shovel_force.tolist(),
        }


