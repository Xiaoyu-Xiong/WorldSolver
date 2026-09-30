"""Solver for 104_boat_cargo_overload_sinking: coupled boat, cargo spheres, and water."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Two-way coupled rigid-body + simplified fluid solver."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._config = scene.config
        self._seed = seed

        ic = self._config["initial_conditions"]
        fluid_cfg = self._config["fluid"]
        self._gravity = np.asarray(self._config["gravity"], dtype=float)
        self._rho_fluid = float(fluid_cfg["density"])
        self._water_depth0 = float(fluid_cfg["initial_water_depth"])
        self._h = float(ic["particle_spacing"])

        self._boat_mass = float(ic["boat_mass"])
        sr = float(ic["sphere_radius"])
        sphere_vol = 4.0 / 3.0 * math.pi * sr ** 3
        self._light_density = float(ic["light_density"])
        self._heavy_density = float(ic["heavy_density"])
        self._light_mass = sphere_vol * self._light_density
        self._heavy_mass = sphere_vol * self._heavy_density
        self._sphere_radius = sr

        self._hull_length = float(ic["hull_length"])
        self._hull_width = float(ic["hull_width"])
        self._hull_height = float(ic["hull_height"])

        self._light_release = float(ic["light_release_time"])
        self._heavy_release = float(ic["heavy_release_time"])

        frame0 = scene.initial_state()
        self._pos = np.array(frame0.position, dtype=float)
        self._vel = np.array(frame0.velocity, dtype=float)
        self._mass = np.array(frame0.mass, dtype=float)
        self._body_id = np.array(frame0.body_id, dtype=int)

        self._water_mask = self._body_id == 0
        self._n_water = int(np.sum(self._water_mask))
        self._boat_idx = int(np.where(self._body_id == 1)[0][0])
        self._light_idx = int(np.where(self._body_id == 2)[0][0])
        self._heavy_idx = int(np.where(self._body_id == 3)[0][0])
        self._m_w = float(self._mass[self._water_mask][0]) if self._n_water > 0 else 0.0

        self._hull_bottom_half_len = 0.72
        self._hull_bottom_half_wid = 0.34
        self._hull_top_half_len = 0.95
        self._hull_top_half_wid = 0.45
        self._hull_h = 0.48

        self._pool_xmin = -2.3
        self._pool_xmax = 2.3
        self._pool_ymin = -1.3
        self._pool_ymax = 1.3
        self._pool_zmin = 0.0

        self._time = 0.0
        self._light_released = False
        self._heavy_released = False
        self._water_in_boat = 0.0

        self._cum_ext_impulse = np.zeros(3, dtype=float)
        self._cum_inflow_mom = np.zeros(3, dtype=float)
        self._cum_outflow_mom = np.zeros(3, dtype=float)
        self._n_interaction_pairs = 6
        self._interaction_force = np.zeros((self._n_interaction_pairs, 3), dtype=float)

        total_mass = self._boat_mass + self._light_mass + self._heavy_mass + self._n_water * self._m_w
        self._ref_velocity = math.sqrt(abs(self._gravity[2]) * self._hull_height)
        self._ref_momentum = total_mass * self._ref_velocity

        self._boundary_normals = np.array([
            [0, 0, 1], [0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
        ], dtype=float)
        self._boundary_velocity = np.zeros((5, 3), dtype=float)
        self._boundary_target_velocity = np.zeros((5, 3), dtype=float)
        self._n_budget_domains = 4
        self._cum_mass_source = np.zeros(1, dtype=float)

        self._grid_nx = int(math.ceil((self._pool_xmax - self._pool_xmin) / (2 * self._h))) + 2
        self._grid_ny = int(math.ceil((self._pool_ymax - self._pool_ymin) / (2 * self._h))) + 2
        self._grid_nz = int(math.ceil(2.8 / (2 * self._h))) + 2

        a_bot = 4 * self._hull_bottom_half_len * self._hull_bottom_half_wid
        a_top = 4 * self._hull_top_half_len * self._hull_top_half_wid
        self._hull_full_volume = (self._hull_h / 3.0) * (a_bot + a_top + math.sqrt(a_bot * a_top))

        self._light_in_boat = False
        self._heavy_in_boat = False
        
        # Track if heavy sphere has impacted the boat
        self._heavy_impacted = False

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self._time
        if dt < 0:
            dt = 0.0
        self._time = time
        max_sub_dt = 0.005
        n_steps = max(1, int(math.ceil(dt / max_sub_dt)))
        sub_dt = dt / n_steps if n_steps > 0 else 0.0
        for _ in range(n_steps):
            self._substep(sub_dt)
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "coupled_rigid_body_simplified_fluid"}

    def _hull_area_at_fraction(self, t: float) -> float:
        bl = self._hull_bottom_half_len
        bw = self._hull_bottom_half_wid
        tl = bl + t * (self._hull_top_half_len - bl)
        tw = bw + t * (self._hull_top_half_wid - bw)
        return 4 * tl * tw

    def _hull_submerged_volume(self, sub_depth: float) -> float:
        if sub_depth <= 0:
            return 0.0
        sub_depth = min(sub_depth, self._hull_h)
        t = sub_depth / self._hull_h
        a_bot = self._hull_area_at_fraction(0)
        a_top = self._hull_area_at_fraction(t)
        return (sub_depth / 3.0) * (a_bot + a_top + math.sqrt(max(0, a_bot * a_top)))

    def _estimate_water_surface(self) -> float:
        water_z = self._pos[self._water_mask, 2]
        if len(water_z) > 0:
            return float(np.percentile(water_z, 95))
        return self._water_depth0

    def _is_above_hull_opening(self, pos, boat_pos):
        hl = self._hull_length / 2 * 0.9
        hw = self._hull_width / 2 * 0.9
        return (abs(pos[0] - boat_pos[0]) < hl and
                abs(pos[1] - boat_pos[1]) < hw and
                pos[2] >= boat_pos[2])

    def _is_inside_hull(self, pos, boat_pos):
        hl = self._hull_length / 2 * 0.85
        hw = self._hull_width / 2 * 0.85
        return (abs(pos[0] - boat_pos[0]) < hl and
                abs(pos[1] - boat_pos[1]) < hw and
                boat_pos[2] < pos[2] < boat_pos[2] + self._hull_h)

    def _substep(self, dt: float) -> None:
        if dt <= 0:
            return
        g = self._gravity

        if not self._light_released and self._time >= self._light_release:
            self._light_released = True
        if not self._heavy_released and self._time >= self._heavy_release:
            self._heavy_released = True

        boat_pos = self._pos[self._boat_idx]
        boat_vel = self._vel[self._boat_idx]
        boat_mass = self._mass[self._boat_idx]
        light_pos = self._pos[self._light_idx]
        light_vel = self._vel[self._light_idx]
        light_mass = self._mass[self._light_idx]
        heavy_pos = self._pos[self._heavy_idx]
        heavy_vel = self._vel[self._heavy_idx]
        heavy_mass = self._mass[self._heavy_idx]

        water_surface = self._estimate_water_surface()

        # --- Buoyancy on boat ---
        hull_bottom_z = boat_pos[2]
        if hull_bottom_z < water_surface:
            sub_depth = min(self._hull_h, water_surface - hull_bottom_z)
            disp_vol = self._hull_submerged_volume(sub_depth)
            effective_disp = max(0, disp_vol - self._water_in_boat)
            boat_buoyancy = np.array([0, 0, self._rho_fluid * abs(g[2]) * effective_disp])
        else:
            boat_buoyancy = np.zeros(3)

        # --- Water drag on boat ---
        boat_drag = np.zeros(3)
        if hull_bottom_z < water_surface + 0.1:
            sub_frac = max(0, min(1, (water_surface - hull_bottom_z) / self._hull_h))
            wetted_area = self._hull_length * self._hull_width * sub_frac
            cd_boat = 1.5
            speed = np.linalg.norm(boat_vel)
            if speed > 1e-8:
                drag_mag = 0.5 * cd_boat * self._rho_fluid * wetted_area * speed
                boat_drag = -drag_mag * boat_vel / speed

        # --- Water flooding ---
        hull_rim_z = boat_pos[2] + self._hull_h
        if hull_rim_z < water_surface:
            flood_head = water_surface - hull_rim_z
            top_area = self._hull_area_at_fraction(1.0)
            inflow_speed = math.sqrt(2 * abs(g[2]) * max(flood_head, 0.01))
            flood_rate = top_area * inflow_speed * 0.3
            added_vol = flood_rate * dt
            self._water_in_boat = min(self._hull_full_volume, self._water_in_boat + added_vol)
        elif hull_rim_z < water_surface + 0.05 and boat_vel[2] < -0.1:
            plunge_rate = abs(boat_vel[2]) * self._hull_length * self._hull_width * 0.3
            added_vol = plunge_rate * dt
            self._water_in_boat = min(self._hull_full_volume, self._water_in_boat + added_vol)

        # --- Sphere buoyancy ---
        light_buoyancy = self._compute_sphere_buoyancy(light_pos, water_surface)
        heavy_buoyancy = self._compute_sphere_buoyancy(heavy_pos, water_surface)

        # --- Gravity ---
        water_weight_inside = self._water_in_boat * self._rho_fluid * g
        boat_total_gravity = boat_mass * g + water_weight_inside
        light_gravity = light_mass * g if self._light_released else np.zeros(3)
        heavy_gravity = heavy_mass * g if self._heavy_released else np.zeros(3)

        # --- Drag on spheres ---
        light_drag = self._compute_sphere_drag(light_pos, light_vel, water_surface)
        heavy_drag = self._compute_sphere_drag(heavy_pos, heavy_vel, water_surface)

        # --- Contact detection ---
        light_above = self._is_above_hull_opening(light_pos, boat_pos) if self._light_released else False
        heavy_above = self._is_above_hull_opening(heavy_pos, boat_pos) if self._heavy_released else False
        light_inside = self._is_inside_hull(light_pos, boat_pos) if self._light_released else False
        heavy_inside = self._is_inside_hull(heavy_pos, boat_pos) if self._heavy_released else False

        if light_inside:
            self._light_in_boat = True
        if heavy_inside:
            self._heavy_in_boat = True

        # --- Momentum transfer on heavy sphere hitting hull bottom ---
        # When heavy sphere first enters the hull with significant downward velocity,
        # transfer momentum to boat (inelastic collision)
        if (self._heavy_released and heavy_inside and not self._heavy_impacted and 
            heavy_vel[2] < -0.5):
            # Inelastic collision: conserve momentum
            # m_sphere * v_sphere + m_boat * v_boat = (m_sphere + m_boat) * v_new
            # But we don't want them to stick - just transfer some momentum
            impulse_frac = 0.7  # fraction of momentum transferred
            dv = heavy_vel[2] * impulse_frac * heavy_mass / boat_mass
            self._vel[self._boat_idx, 2] += dv
            self._vel[self._heavy_idx, 2] *= (1 - impulse_frac)
            self._heavy_impacted = True

        # Hull exterior contact
        bl_contact = np.zeros(3)
        bh_contact = np.zeros(3)
        if self._light_released and not light_above and not light_inside:
            bl_contact = self._sphere_hull_contact(light_pos, light_vel, boat_pos, boat_vel)
        if self._heavy_released and not heavy_above and not heavy_inside:
            bh_contact = self._sphere_hull_contact(heavy_pos, heavy_vel, boat_pos, boat_vel)

        # Net forces
        f_boat = boat_total_gravity + boat_buoyancy + boat_drag - bl_contact - bh_contact
        f_light = (light_gravity + light_buoyancy + light_drag + bl_contact) if self._light_released else np.zeros(3)
        f_heavy = (heavy_gravity + heavy_buoyancy + heavy_drag + bh_contact) if self._heavy_released else np.zeros(3)

        self._interaction_force[0] = -boat_buoyancy
        self._interaction_force[1] = -light_buoyancy - light_drag
        self._interaction_force[2] = -heavy_buoyancy - heavy_drag
        self._interaction_force[3] = bl_contact
        self._interaction_force[4] = bh_contact
        self._interaction_force[5] = np.zeros(3)

        # Integrate
        self._vel[self._boat_idx] += (f_boat / boat_mass) * dt
        self._pos[self._boat_idx] += self._vel[self._boat_idx] * dt
        if self._light_released:
            self._vel[self._light_idx] += (f_light / light_mass) * dt
            self._pos[self._light_idx] += self._vel[self._light_idx] * dt
        if self._heavy_released:
            self._vel[self._heavy_idx] += (f_heavy / heavy_mass) * dt
            self._pos[self._heavy_idx] += self._vel[self._heavy_idx] * dt

        # Water dynamics
        wpos = self._pos[self._water_mask]
        wvel = self._vel[self._water_mask]
        wvel += g * dt
        self._apply_pressure_vectorized(wpos, wvel, dt)
        self._repel_from_boat_vec(wpos, wvel, self._pos[self._boat_idx])
        if self._light_released:
            self._repel_from_sphere_vec(wpos, wvel, self._pos[self._light_idx])
        if self._heavy_released:
            self._repel_from_sphere_vec(wpos, wvel, self._pos[self._heavy_idx])
        self._enforce_pool_walls_vec(wpos, wvel)
        self._pos[self._water_mask] = wpos
        self._vel[self._water_mask] = wvel

        total_mass_all = boat_mass + light_mass + heavy_mass + self._n_water * self._m_w
        self._cum_ext_impulse += total_mass_all * g * dt
        self._enforce_floor_rigid()

    def _sphere_hull_contact(self, sph_pos, sph_vel, box_pos, box_vel):
        r = self._sphere_radius
        hl = self._hull_length / 2
        hw = self._hull_width / 2
        hh = self._hull_height
        bx, by, bz = box_pos

        # Bottom contact
        dx = abs(sph_pos[0] - bx)
        dy = abs(sph_pos[1] - by)
        dz = sph_pos[2] - bz
        if dx < hl and dy < hw and 0 < dz < r:
            normal = np.array([0, 0, -1])
            penetration = r - dz
            rel_vn = np.dot(sph_vel - box_vel, normal)
            fn = max(0, 50000.0 * penetration - 500.0 * rel_vn)
            return fn * normal

        # Side wall contacts
        if bz - r < sph_pos[2] < bz + hh + r:
            best_pen = 0
            best_normal = np.zeros(3)
            
            dist_x_left = sph_pos[0] - (bx - hl)
            if dist_x_left < 0 and abs(sph_pos[1] - by) < hw:
                pen = r + dist_x_left
                if pen > best_pen:
                    best_pen = pen; best_normal = np.array([-1, 0, 0])
            
            dist_x_right = (bx + hl) - sph_pos[0]
            if dist_x_right < 0 and abs(sph_pos[1] - by) < hw:
                pen = r + dist_x_right
                if pen > best_pen:
                    best_pen = pen; best_normal = np.array([1, 0, 0])

            dist_y_front = sph_pos[1] - (by - hw)
            if dist_y_front < 0 and abs(sph_pos[0] - bx) < hl:
                pen = r + dist_y_front
                if pen > best_pen:
                    best_pen = pen; best_normal = np.array([0, -1, 0])

            dist_y_back = (by + hw) - sph_pos[1]
            if dist_y_back < 0 and abs(sph_pos[0] - bx) < hl:
                pen = r + dist_y_back
                if pen > best_pen:
                    best_pen = pen; best_normal = np.array([0, 1, 0])

            if best_pen > 0:
                rel_vn = np.dot(sph_vel - box_vel, best_normal)
                fn = max(0, 50000.0 * best_pen - 500.0 * rel_vn)
                return fn * best_normal

        return np.zeros(3)

    def _compute_sphere_buoyancy(self, sphere_pos, water_surface):
        r = self._sphere_radius
        z_bottom = sphere_pos[2] - r
        if z_bottom >= water_surface:
            return np.zeros(3)
        h = min(2 * r, water_surface - z_bottom)
        if h <= 0:
            return np.zeros(3)
        vol_sub = math.pi * h * h * (3 * r - h) / 3.0
        return np.array([0, 0, self._rho_fluid * abs(self._gravity[2]) * vol_sub])

    def _compute_sphere_drag(self, sphere_pos, sphere_vel, water_surface):
        r = self._sphere_radius
        if sphere_pos[2] - r >= water_surface:
            return np.zeros(3)
        z_bottom = sphere_pos[2] - r
        sub_height = min(2 * r, water_surface - z_bottom)
        frac = max(0, sub_height / (2 * r))
        speed = np.linalg.norm(sphere_vel)
        if speed < 1e-12:
            return np.zeros(3)
        drag_mag = 0.5 * 0.47 * self._rho_fluid * math.pi * r * r * speed * frac
        return -drag_mag * sphere_vel / speed

    def _apply_pressure_vectorized(self, pos, vel, dt):
        n = len(pos)
        if n == 0:
            return
        h = self._h
        cell_size = 2.0 * h
        cx = np.floor((pos[:, 0] - self._pool_xmin) / cell_size).astype(int)
        cy = np.floor((pos[:, 1] - self._pool_ymin) / cell_size).astype(int)
        cz = np.floor(pos[:, 2] / cell_size).astype(int)
        nx_g, ny_g, nz_g = self._grid_nx, self._grid_ny, self._grid_nz
        cx = np.clip(cx, 0, nx_g - 1)
        cy = np.clip(cy, 0, ny_g - 1)
        cz = np.clip(cz, 0, nz_g - 1)
        cell_idx = cx * ny_g * nz_g + cy * nz_g + cz
        n_cells = nx_g * ny_g * nz_g
        counts = np.bincount(cell_idx, minlength=n_cells)
        target_count = max(1, int((cell_size / h) ** 3))
        cell_pressure = np.maximum(0, counts - target_count) * 50.0
        acc = np.zeros_like(vel)
        cxp = np.clip(cx + 1, 0, nx_g - 1); cxm = np.clip(cx - 1, 0, nx_g - 1)
        grad_x = (cell_pressure[cxp * ny_g * nz_g + cy * nz_g + cz] -
                  cell_pressure[cxm * ny_g * nz_g + cy * nz_g + cz]) / (2 * cell_size)
        acc[:, 0] -= grad_x / self._m_w * dt * 0.5
        cyp = np.clip(cy + 1, 0, ny_g - 1); cym = np.clip(cy - 1, 0, ny_g - 1)
        grad_y = (cell_pressure[cx * ny_g * nz_g + cyp * nz_g + cz] -
                  cell_pressure[cx * ny_g * nz_g + cym * nz_g + cz]) / (2 * cell_size)
        acc[:, 1] -= grad_y / self._m_w * dt * 0.5
        czp = np.clip(cz + 1, 0, nz_g - 1); czm = np.clip(cz - 1, 0, nz_g - 1)
        grad_z = (cell_pressure[cx * ny_g * nz_g + cy * nz_g + czp] -
                  cell_pressure[cx * ny_g * nz_g + cy * nz_g + czm]) / (2 * cell_size)
        acc[:, 2] -= grad_z / self._m_w * dt * 0.5
        vel += acc
        center = np.mean(pos, axis=0)
        vel += 0.005 * (center - pos) * dt

    def _repel_from_boat_vec(self, wpos, wvel, boat_pos):
        hl = self._hull_length / 2; hw = self._hull_width / 2; hh = self._hull_height
        margin = self._h * 0.3
        bx, by, bz = boat_pos
        inside = ((wpos[:, 0] > bx - hl + margin) & (wpos[:, 0] < bx + hl - margin) &
                  (wpos[:, 1] > by - hw + margin) & (wpos[:, 1] < by + hw - margin) &
                  (wpos[:, 2] > bz) & (wpos[:, 2] < bz + hh))
        idx = np.where(inside)[0]
        if len(idx) == 0:
            return
        px, py, pz = wpos[idx, 0], wpos[idx, 1], wpos[idx, 2]
        dists = np.column_stack([
            px - (bx - hl + margin), (bx + hl - margin) - px,
            py - (by - hw + margin), (by + hw - margin) - py,
            pz - bz, (bz + hh) - pz
        ])
        dirs = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], dtype=float)
        min_face = np.argmin(dists, axis=1)
        push_dir = dirs[min_face]
        pen = margin - dists[np.arange(len(idx)), min_face]
        valid = pen > 0
        wpos[idx[valid]] += push_dir[valid] * pen[valid, None] * 1.1
        vn = np.sum(wvel[idx[valid]] * push_dir[valid], axis=1)
        neg = vn < 0
        if np.any(neg):
            wvel[idx[valid][neg]] -= 1.5 * vn[neg, None] * push_dir[valid][neg]

    def _repel_from_sphere_vec(self, wpos, wvel, sphere_pos):
        r = self._sphere_radius + self._h * 0.2
        diff = wpos - sphere_pos
        dists = np.linalg.norm(diff, axis=1)
        inside = dists < r
        idx = np.where(inside)[0]
        if len(idx) == 0:
            return
        normals = diff[idx] / dists[idx, None]
        pen = r - dists[idx]
        wpos[idx] += normals * pen[:, None] * 1.1
        vn = np.sum(wvel[idx] * normals, axis=1)
        neg = vn < 0
        if np.any(neg):
            wvel[idx[neg]] -= 1.5 * vn[neg, None] * normals[neg]

    def _enforce_pool_walls_vec(self, pos, vel):
        eps = 0.01
        for ax, lo, hi in [(0, self._pool_xmin, self._pool_xmax),
                           (1, self._pool_ymin, self._pool_ymax),
                           (2, self._pool_zmin, 100.0)]:
            mask_lo = pos[:, ax] < lo + eps
            pos[mask_lo, ax] = lo + eps
            vel[mask_lo, ax] = np.abs(vel[mask_lo, ax]) * 0.3
            if hi < 50:
                mask_hi = pos[:, ax] > hi - eps
                pos[mask_hi, ax] = hi - eps
                vel[mask_hi, ax] = -np.abs(vel[mask_hi, ax]) * 0.3

    def _enforce_floor_rigid(self):
        eps = 0.02
        for idx in [self._boat_idx, self._light_idx, self._heavy_idx]:
            if self._pos[idx, 2] < self._pool_zmin + eps:
                self._pos[idx, 2] = self._pool_zmin + eps
                if self._vel[idx, 2] < 0:
                    self._vel[idx, 2] *= -0.3

    def _build_frame(self, time: float) -> StateFrame:
        n = len(self._pos)
        G = self._n_budget_domains
        mass_quantity = self._mass.copy()
        phase = self._body_id.astype(float).copy()
        cum_ext_impulse_full = np.zeros((G, 3), dtype=float)
        cum_ext_impulse_full[0] = self._cum_ext_impulse
        cum_inflow_full = np.zeros((G, 3), dtype=float)
        cum_outflow_full = np.zeros((G, 3), dtype=float)
        domain_weights = np.zeros((G, n), dtype=float)
        domain_weights[0, :] = 1.0
        domain_weights[1, self._body_id == 1] = 1.0
        domain_weights[2, self._body_id == 2] = 1.0
        domain_weights[3, self._body_id == 3] = 1.0
        ext_force = np.zeros((G, 1, 3), dtype=float)
        ext_force[0, 0] = np.sum(self._mass[:, None] * self._gravity, axis=0)
        ext_force[1, 0] = self._mass[self._boat_idx] * self._gravity
        ext_force[2, 0] = self._mass[self._light_idx] * self._gravity
        ext_force[3, 0] = self._mass[self._heavy_idx] * self._gravity
        fields = {
            "boundary_normal": self._boundary_normals.copy(),
            "boundary_target_velocity": self._boundary_target_velocity.copy(),
            "boundary_velocity": self._boundary_velocity.copy(),
            "budget_cumulative_external_impulse": cum_ext_impulse_full,
            "budget_cumulative_inflow_momentum": cum_inflow_full,
            "budget_cumulative_outflow_momentum": cum_outflow_full,
            "budget_domain_weights": domain_weights,
            "budget_external_force": ext_force,
            "cumulative_mass_source": self._cum_mass_source.copy(),
            "interaction_force": self._interaction_force.copy(),
        }
        attributes = {"mass_quantity": mass_quantity, "phase": phase}
        scalars = {"reference_momentum": self._ref_momentum, "reference_velocity": self._ref_velocity}
        return StateFrame(
            time=time, position=self._pos.copy(), velocity=self._vel.copy(),
            mass=self._mass.copy(), body_id=self._body_id.copy(),
            attributes=attributes, fields=fields, scalars=scalars,
        )
