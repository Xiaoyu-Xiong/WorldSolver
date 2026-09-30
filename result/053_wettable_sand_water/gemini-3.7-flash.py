"""BenchmarkSolver for wettable sand and pouring water multiphysics simulation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # Particles state
        self.position = np.empty((0, 3), dtype=np.float64)
        self.velocity = np.empty((0, 3), dtype=np.float64)
        self.mass = np.empty((0,), dtype=np.float64)
        self.body_id = np.empty((0,), dtype=np.int32)
        self.solid = np.empty((0,), dtype=np.float64)
        self.concentration = np.empty((0,), dtype=np.float64)
        self.wetness = np.empty((0,), dtype=np.float64)

        # Diagnostic grid definitions
        self.grid_origin = np.array([-1.5, -1.0, 0.0], dtype=np.float64)
        self.grid_spacing = 0.15
        self.grid_shape = (20, 14, 16)  # nx, ny, nz

        # Contact length scale
        self.contact_length_scale = 0.065

        # Cumulative source tracking
        self.cumulative_mass_source = np.array([0.0, 0.0], dtype=np.float64)  # [sand, water]

        # Internal interaction force tracking [K, 3] where K=2 (sand, water)
        self.interaction_force = np.zeros((2, 3), dtype=np.float64)

        # Sub-stepping dt
        self.sim_dt = 0.002

        # Emitter timer
        self.last_emit_time = 0.0
        self.emit_interval = 0.04

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        frame = scene.initial_state()
        self.current_time = float(frame.time)
        self.position = np.array(frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(frame.body_id, dtype=np.int32, copy=True)

        if "solid_phase_fraction" in frame.attributes:
            self.solid = np.array(frame.attributes["solid_phase_fraction"], dtype=np.float64, copy=True)
        else:
            self.solid = np.where(self.body_id == 1, 1.0, 0.0)

        if "water_concentration" in frame.attributes:
            self.concentration = np.array(frame.attributes["water_concentration"], dtype=np.float64, copy=True)
        else:
            self.concentration = np.where(self.body_id == 2, 1.0, 0.0)

        if "wetness" in frame.attributes:
            self.wetness = np.array(frame.attributes["wetness"], dtype=np.float64, copy=True)
        else:
            self.wetness = np.where(self.body_id == 2, 1.0, 0.0)

        self.cumulative_mass_source = np.array([0.0, 0.0], dtype=np.float64)
        self.interaction_force = np.zeros((2, 3), dtype=np.float64)
        self.last_emit_time = 0.0

    def _emit_water(self, dt: float) -> None:
        # Inflowing water from spouts
        # Spouts are at left: [-0.58, 0.08, 1.92] and right: [0.58, 0.08, 1.92]
        spacing = 0.065
        new_pts = []
        new_vel = []
        for side in (-1.0, 1.0):
            # emit 4 particles per spout per emission step
            for dx in (-0.5, 0.5):
                for dy in (-0.5, 0.5):
                    jitter_x = np.random.uniform(-0.01, 0.01)
                    jitter_y = np.random.uniform(-0.01, 0.01)
                    jitter_z = np.random.uniform(-0.01, 0.01)
                    new_pts.append((
                        side * 0.58 + dx * spacing * 0.5 + jitter_x,
                        0.08 + dy * spacing * 0.5 + jitter_y,
                        1.92 + jitter_z
                    ))
                    new_vel.append((-side * 0.38, 0.0, -1.85))

        n_new = len(new_pts)
        if n_new > 0:
            new_pos = np.asarray(new_pts, dtype=np.float64)
            new_v = np.asarray(new_vel, dtype=np.float64)
            m_water = 0.000275
            new_m = np.full(n_new, m_water, dtype=np.float64)
            new_bid = np.full(n_new, 2, dtype=np.int32)
            new_solid = np.zeros(n_new, dtype=np.float64)
            new_conc = np.ones(n_new, dtype=np.float64)
            new_wet = np.ones(n_new, dtype=np.float64)

            self.position = np.vstack((self.position, new_pos))
            self.velocity = np.vstack((self.velocity, new_v))
            self.mass = np.concatenate((self.mass, new_m))
            self.body_id = np.concatenate((self.body_id, new_bid))
            self.solid = np.concatenate((self.solid, new_solid))
            self.concentration = np.concatenate((self.concentration, new_conc))
            self.wetness = np.concatenate((self.wetness, new_wet))

            self.cumulative_mass_source[1] += float(n_new * m_water)

    def _step_physics(self, dt: float) -> None:
        N = len(self.position)
        if N == 0:
            return

        pos = self.position
        vel = self.velocity
        bid = self.body_id
        wet = self.wetness

        # Gravity
        gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        vel += gravity * dt

        # Particle radii / smoothing lengths
        r_sand = 0.038
        r_water = 0.035

        is_sand = (bid == 1)
        is_water = (bid == 2)

        sand_idx = np.flatnonzero(is_sand)
        water_idx = np.flatnonzero(is_water)

        # Water-sand interaction & wetting transfer using spatial hashing / cell lists
        # Let's use a fast KD-tree or cell grid for neighbor search
        interaction_force_sand = np.zeros(3, dtype=np.float64)
        interaction_force_water = np.zeros(3, dtype=np.float64)

        # Fast neighbor search for water-sand contact and wetting
        if len(water_idx) > 0 and len(sand_idx) > 0:
            # We can use spatial binning for contact
            cell_size = 0.10
            # Bin sand particles
            sand_pos = pos[sand_idx]
            water_pos = pos[water_idx]

            # Simple broadphase: compute pairwise interactions for nearby particles
            # To be efficient and vectorised, bin by coordinates
            bins = {}
            sand_coords = np.floor(sand_pos / cell_size).astype(np.int64)
            for i, c in enumerate(sand_coords):
                key = (c[0], c[1], c[2])
                if key not in bins:
                    bins[key] = []
                bins[key].append(sand_idx[i])

            water_coords = np.floor(water_pos / cell_size).astype(np.int64)
            h_inter = 0.08  # interaction radius
            h2 = h_inter * h_inter

            # Wetting transfer rate
            wetting_rate = 0.8 * dt

            for wi, wc in enumerate(water_coords):
                w_glob = water_idx[wi]
                w_p = pos[w_glob]
                w_v = vel[w_glob]

                # Check 27 neighboring bins
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            key = (wc[0] + dx, wc[1] + dy, wc[2] + dz)
                            if key in bins:
                                for s_glob in bins[key]:
                                    s_p = pos[s_glob]
                                    disp = w_p - s_p
                                    dist2 = np.dot(disp, disp)
                                    if dist2 < h2 and dist2 > 1e-8:
                                        dist = np.sqrt(dist2)
                                        normal = disp / dist

                                        # Repulsion / collision
                                        overlap = (r_sand + r_water) - dist
                                        if overlap > 0:
                                            # Collision impulse/force
                                            rel_v = w_v - vel[s_glob]
                                            v_n = np.dot(rel_v, normal)
                                            k_rep = 2000.0
                                            damp = 20.0
                                            f_mag = k_rep * overlap - damp * v_n
                                            if f_mag > 0:
                                                force = f_mag * normal
                                                interaction_force_water += force * dt
                                                interaction_force_sand -= force * dt
                                                # apply acceleration
                                                vel[w_glob] += (force / self.mass[w_glob]) * dt
                                                vel[s_glob] -= (force / self.mass[s_glob]) * dt * 0.1  # Sand pile is sturdy

                                        # Wetting concentration transfer:
                                        # Sand absorbs wetness up to 0.05
                                        if wet[s_glob] < 0.05:
                                            transfer = min(0.05 - wet[s_glob], wetting_rate * (1.0 - dist / h_inter))
                                            wet[s_glob] += transfer
                                            self.concentration[s_glob] = wet[s_glob]
                                            # Water slightly loses concentration
                                            wet[w_glob] = max(0.0, wet[w_glob] - transfer * 0.1)
                                            self.concentration[w_glob] = wet[w_glob]

        # Sand-sand internal friction and resting stability
        # Sand particles slightly damp velocity and resist penetration
        vel[sand_idx] *= np.exp(-1.5 * dt)

        # Water-water viscosity/damping
        vel[water_idx] *= np.exp(-0.2 * dt)

        # Advect positions
        pos += vel * dt

        # Boundary collisions
        # Floor: z = 0.0 (tray floor is at z in [-0.04, 0.0])
        # Tray boundaries:
        # x: [-1.47, 1.47], y: [-1.11, 1.07], z: [0.0, inf)
        x_min, x_max = -1.47, 1.47
        y_min, y_max = -1.11, 1.07
        z_min = 0.02

        # Sand boundary conditions (stays compact on floor)
        for i in range(N):
            p = pos[i]
            v = vel[i]
            b = bid[i]
            r = r_sand if b == 1 else r_water

            # Floor
            if p[2] < z_min + r:
                p[2] = z_min + r
                if v[2] < 0:
                    v[2] = -0.1 * v[2] if b == 2 else 0.0
                # friction
                v[0] *= 0.8
                v[1] *= 0.8

            # X walls
            if p[0] < x_min + r:
                p[0] = x_min + r
                if v[0] < 0:
                    v[0] = -0.2 * v[0]
            elif p[0] > x_max - r:
                p[0] = x_max - r
                if v[0] > 0:
                    v[0] = -0.2 * v[0]

            # Y walls
            if p[1] < y_min + r:
                p[1] = y_min + r
                if v[1] < 0:
                    v[1] = -0.2 * v[1]
            elif p[1] > y_max - r:
                p[1] = y_max - r
                if v[1] > 0:
                    v[1] = -0.2 * v[1]

        # Save interaction forces
        self.interaction_force[0] = interaction_force_sand
        self.interaction_force[1] = interaction_force_water

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)

        # Advance simulation in fixed sub-steps
        while self.current_time + self.sim_dt <= target_time + 1e-9:
            # Emit water continuously during pouring (up to 4.5 seconds)
            if self.current_time < 4.5 and (self.current_time - self.last_emit_time) >= self.emit_interval:
                self._emit_water(self.sim_dt)
                self.last_emit_time = self.current_time

            self._step_physics(self.sim_dt)
            self.current_time += self.sim_dt

        # Ensure exact time match
        self.current_time = target_time

        # Compute diagnostic Eulerian fields
        nx, ny, nz = self.grid_shape
        active_mask = np.zeros(self.grid_shape, dtype=np.float64)
        transported_scalar = np.zeros(self.grid_shape, dtype=np.float64)
        scalar_source = np.zeros(self.grid_shape, dtype=np.float64)
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)

        # Rasterize particles onto diagnostic grid
        if len(self.position) > 0:
            # Grid index
            g_idx = np.floor((self.position - self.grid_origin) / self.grid_spacing).astype(np.int64)
            valid = (
                (g_idx[:, 0] >= 0) & (g_idx[:, 0] < nx) &
                (g_idx[:, 1] >= 0) & (g_idx[:, 1] < ny) &
                (g_idx[:, 2] >= 0) & (g_idx[:, 2] < nz)
            )

            valid_idx = g_idx[valid]
            valid_wet = self.wetness[valid]
            valid_vel = self.velocity[valid]
            valid_bid = self.body_id[valid]

            valid_indices = np.flatnonzero(valid)
            for k, orig_i in enumerate(valid_indices):
                ix, iy, iz = valid_idx[k]
                active_mask[ix, iy, iz] = 1.0
                transported_scalar[ix, iy, iz] = max(transported_scalar[ix, iy, iz], float(valid_wet[k]))
                velocity_field[ix, iy, iz] = valid_vel[k]
                if valid_bid[k] == 2 and self.position[orig_i, 2] > 1.8:
                    scalar_source[ix, iy, iz] = 1.0

        # Signed distance field at active contact samples
        # Approximate signed distance from particles to floor / pile
        c_len = max(len(self.position), 1)
        # Compute gap to floor for all particles
        contact_sd = self.position[:, 2] - 0.04 if len(self.position) > 0 else np.zeros(1)

        # Assemble StateFrame
        attributes = {
            "material_id": self.body_id.copy(),
            "solid_phase_fraction": self.solid.copy(),
            "water_concentration": self.concentration.copy(),
            "wetness": self.wetness.copy(),
            "mass_quantity": self.mass.copy(),
        }

        fields = {
            "active_mask": active_mask,
            "transported_scalar": transported_scalar,
            "scalar_source": scalar_source,
            "velocity_field": velocity_field,
            "contact_signed_distance": contact_sd,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
        }

        sand_mask = (self.body_id == 1)
        water_mask = (self.body_id == 2)
        scalars = {
            "contact_length_scale": float(self.contact_length_scale),
            "grid_spacing": float(self.grid_spacing),
            "sand_mass": float(self.mass[sand_mask].sum()) if sand_mask.any() else 0.0,
            "water_mass": float(self.mass[water_mask].sum()) if water_mask.any() else 0.0,
        }

        frame = StateFrame(
            time=float(self.current_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": len(self.position),
            "current_time": self.current_time,
            "cumulative_mass_source": self.cumulative_mass_source.tolist(),
        }

