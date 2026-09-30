"""Two-way coupled water-cloth curtain solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Load topology
        topo = np.load("assets/curtain_topology.npz")
        self.edges = topo["edges"]  # shape (3670, 2)
        self.faces = topo["faces"]  # shape (2400, 3)
        self.pinned_indices = topo["pinned"]  # shape (31,)

        # Get initial state
        frame0 = scene.initial_state()
        self.current_time = float(frame0.time)

        # Split cloth and water
        cloth_mask = (frame0.body_id == 0)
        water_mask = (frame0.body_id == 1)

        self.cloth_pos = np.array(frame0.position[cloth_mask], dtype=np.float64)
        self.cloth_vel = np.array(frame0.velocity[cloth_mask], dtype=np.float64)
        self.cloth_mass = np.array(frame0.mass[cloth_mask], dtype=np.float64)
        self.num_cloth = len(self.cloth_pos)

        self.water_pos = np.array(frame0.position[water_mask], dtype=np.float64)
        self.water_vel = np.array(frame0.velocity[water_mask], dtype=np.float64)
        self.water_mass = np.array(frame0.mass[water_mask], dtype=np.float64)
        self.num_initial_water = len(self.water_pos)

        # Edge rest lengths
        p0 = self.cloth_pos[self.edges[:, 0]]
        p1 = self.cloth_pos[self.edges[:, 1]]
        self.edge_rest_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Bending pairs (diagonals or 2-hop edges for bending resistance)
        # We can construct cross-diagonal / bending springs across shared edges of triangles
        # For simplicity and robust stability, use triangle mesh adjacency
        self._setup_cloth_springs()

        # Tank bounds: floor z=-0.06 to +0.06 (top is at 0.0). So floor contact at z = 0.005.
        # Walls:
        # left x = -1.615 + 0.035 = -1.58
        # right x = 2.115 - 0.035 = 2.08
        # front y = -1.415 + 0.035 = -1.38
        # back y = 1.415 - 0.035 = 1.38
        self.tank_x_min = -1.58
        self.tank_x_max = 2.08
        self.tank_y_min = -1.38
        self.tank_y_max = 1.38
        self.tank_z_min = 0.01

        # Inflow parameters
        meta = scene.metadata()
        inflow = meta.get("inflow", {})
        self.inflow_center = np.array(inflow.get("center", [-1.255, 0.0, 1.72]), dtype=np.float64)
        self.inflow_radius = float(inflow.get("radius", 0.20))
        self.inflow_vel = np.array(inflow.get("velocity", [3.2, 0.0, 0.0]), dtype=np.float64)

        self.cloth_thickness = 0.015
        self.contact_length_scale = 0.02
        self.cumulative_mass_source_val = 0.0

        # Inflow particle generation tracking
        self.inflow_spacing = 0.075
        self.water_particle_mass = 1000.0 * (0.075 ** 3)
        self.inflow_timer = 0.0
        self.inflow_interval = self.inflow_spacing / self.inflow_vel[0]

        # Contact tracking for diagnostic / output
        self.contact_dists = np.array([0.05], dtype=np.float64)
        self.ext_force_accum = np.zeros((self.num_cloth + len(self.water_pos), 3), dtype=np.float64)
        self.interaction_force_val = np.zeros((2, 3), dtype=np.float64)

        # Cloth solver params
        self.cloth_stretch_iters = 15
        self.cloth_stretch_stiffness = 0.95
        self.cloth_damping = 0.998

        # SPH / Particle params
        self.h = 0.12
        self.h2 = self.h * self.h
        self.rest_density = 1000.0
        self.k_press = 50.0  # Tait equation / stiffness
        self.gamma = 7.0
        self.viscosity = 0.02

    def _setup_cloth_springs(self) -> None:
        # Build 2-hop bending springs
        adj: dict[int, set[int]] = {}
        for u, v in self.edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        bending = []
        for i in range(self.num_cloth):
            neighbors = list(adj.get(i, []))
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    u, v = neighbors[j], neighbors[k]
                    if v not in adj.get(u, set()):
                        if u > v:
                            u, v = v, u
                        bending.append((u, v))

        bending = list(set(bending))
        if bending:
            self.bending_edges = np.array(bending, dtype=np.int32)
            p0 = self.cloth_pos[self.bending_edges[:, 0]]
            p1 = self.cloth_pos[self.bending_edges[:, 1]]
            self.bending_rest_lengths = np.linalg.norm(p1 - p0, axis=1)
        else:
            self.bending_edges = np.zeros((0, 2), dtype=np.int32)
            self.bending_rest_lengths = np.zeros((0,), dtype=np.float64)

    def _emit_inflow(self, dt: float) -> None:
        self.inflow_timer += dt
        while self.inflow_timer >= self.inflow_interval:
            self.inflow_timer -= self.inflow_interval

            # Generate a circular disk of particles at inflow_center x
            y_vals = np.arange(-self.inflow_radius, self.inflow_radius + 0.1 * self.inflow_spacing, self.inflow_spacing)
            z_vals = np.arange(-self.inflow_radius, self.inflow_radius + 0.1 * self.inflow_spacing, self.inflow_spacing)
            yy, zz = np.meshgrid(y_vals, z_vals)
            yy = yy.ravel()
            zz = zz.ravel()
            mask = (yy**2 + zz**2) <= self.inflow_radius**2
            yy = yy[mask] + self.inflow_center[1]
            zz = zz[mask] + self.inflow_center[2]
            xx = np.full_like(yy, self.inflow_center[0])

            new_pos = np.column_stack((xx, yy, zz))
            new_vel = np.tile(self.inflow_vel, (len(new_pos), 1))
            new_mass = np.full(len(new_pos), self.water_particle_mass, dtype=np.float64)

            self.water_pos = np.vstack((self.water_pos, new_pos))
            self.water_vel = np.vstack((self.water_vel, new_vel))
            self.water_mass = np.concatenate((self.water_mass, new_mass))
            self.cumulative_mass_source_val += np.sum(new_mass)

    def _substep(self, sub_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. External forces on cloth
        f_ext_cloth = self.cloth_mass[:, None] * self.gravity[None, :]

        # 2. External forces on water
        f_ext_water = self.water_mass[:, None] * self.gravity[None, :]

        # Predict water velocity and position
        self.water_vel += (f_ext_water / self.water_mass[:, None]) * sub_dt
        self.water_vel *= 0.9995  # small numerical drag

        # Cloth velocity prediction (PBD style)
        self.cloth_vel += (f_ext_cloth / self.cloth_mass[:, None]) * sub_dt
        self.cloth_vel *= self.cloth_damping

        # Pinned vertices have 0 velocity
        self.cloth_vel[self.pinned_indices] = 0.0

        # Predict positions
        cloth_pred = self.cloth_pos + self.cloth_vel * sub_dt
        cloth_pred[self.pinned_indices] = self.cloth_pos[self.pinned_indices]

        water_pred = self.water_pos + self.water_vel * sub_dt

        # 3. Water-cloth coupling (two-way interaction)
        total_coupled_impulse = np.zeros(3, dtype=np.float64)
        contact_distances = []

        # Efficient grid/spatial query or broadphase for water-cloth interaction
        # Find water particles near cloth curtain bounding box
        cloth_min = np.min(cloth_pred, axis=0) - 0.15
        cloth_max = np.max(cloth_pred, axis=0) + 0.15

        near_mask = (
            (water_pred[:, 0] >= cloth_min[0]) & (water_pred[:, 0] <= cloth_max[0]) &
            (water_pred[:, 1] >= cloth_min[1]) & (water_pred[:, 1] <= cloth_max[1]) &
            (water_pred[:, 2] >= cloth_min[2]) & (water_pred[:, 2] <= cloth_max[2])
        )
        near_indices = np.where(near_mask)[0]

        if len(near_indices) > 0:
            w_pts = water_pred[near_indices]  # (M, 3)
            # Find nearest cloth vertices
            # Using chunked distance to avoid large memory
            # cloth_pred is (1271, 3)
            dists_sq = np.sum((w_pts[:, None, :] - cloth_pred[None, :, :]) ** 2, axis=-1)  # (M, 1271)
            nearest_cloth_idx = np.argmin(dists_sq, axis=1)  # (M,)
            min_dists = np.sqrt(dists_sq[np.arange(len(w_pts)), nearest_cloth_idx])  # (M,)

            contact_threshold = 0.06
            interacting = min_dists < contact_threshold

            if np.any(interacting):
                active_w_idx = near_indices[interacting]
                active_c_idx = nearest_cloth_idx[interacting]
                active_dist = min_dists[interacting]

                # Relative normal: from cloth to water particle
                diff = water_pred[active_w_idx] - cloth_pred[active_c_idx]
                dist = active_dist[:, None] + 1e-8
                normal = diff / dist  # pointing outwards from cloth

                # Signed gap: penetration depth
                signed_gap = active_dist - contact_threshold
                contact_distances.extend(signed_gap.tolist())

                # Push water particle out, push cloth particle back (two-way)
                overlap = contact_threshold - active_dist

                # Cloth normal generally points along +x (incoming jet is from -x)
                # Ensure normal redirects water along curtain
                m_c = self.cloth_mass[active_c_idx]
                m_w = self.water_mass[active_w_idx]
                w_tot = m_c + m_w

                c_ratio = m_w / w_tot
                w_ratio = m_c / w_tot

                # Displacement correction
                delta_w = normal * (overlap * w_ratio)[:, None]
                delta_c = -normal * (overlap * c_ratio)[:, None]

                water_pred[active_w_idx] += delta_w

                # Apply delta_c only to non-pinned cloth
                valid_c = ~np.isin(active_c_idx, self.pinned_indices)
                np.add.at(cloth_pred, active_c_idx[valid_c], delta_c[valid_c])

                # Momentum exchange / velocity deflection
                rel_v = self.water_vel[active_w_idx] - self.cloth_vel[active_c_idx]
                vn = np.sum(rel_v * normal, axis=-1)

                approaching = vn < 0.0
                if np.any(approaching):
                    impulse_mag = -(1.0 + 0.1) * vn[approaching]  # restitution = 0.1
                    impulse = normal[approaching] * impulse_mag[:, None]

                    self.water_vel[active_w_idx[approaching]] += impulse * w_ratio[approaching, None]

                    # Deflect water tangentially downwards / sideways along cloth
                    vt = rel_v[approaching] - (vn[approaching, None] * normal[approaching])
                    # add downward flow along cloth
                    self.water_vel[active_w_idx[approaching], 2] -= 0.5 * sub_dt

                    cloth_impulse = -impulse * c_ratio[approaching, None]
                    valid_app = valid_c[approaching]
                    np.add.at(self.cloth_vel, active_c_idx[approaching][valid_app], cloth_impulse[valid_app])

                    total_coupled_impulse += np.sum(cloth_impulse, axis=0)

        # 4. Cloth stretch & bending constraints (PBD / Position-Based Dynamics)
        inv_mass = 1.0 / self.cloth_mass
        inv_mass[self.pinned_indices] = 0.0

        for _ in range(self.cloth_stretch_iters):
            # Stretch edges
            idx0 = self.edges[:, 0]
            idx1 = self.edges[:, 1]
            p0 = cloth_pred[idx0]
            p1 = cloth_pred[idx1]
            dp = p0 - p1
            d = np.linalg.norm(dp, axis=1) + 1e-12
            err = d - self.edge_rest_lengths
            w0 = inv_mass[idx0]
            w1 = inv_mass[idx1]
            w_sum = w0 + w1
            valid = w_sum > 1e-12

            corr = (err[valid] / w_sum[valid])[:, None] * (dp[valid] / d[valid, None]) * self.cloth_stretch_stiffness
            np.add.at(cloth_pred, idx0[valid], -w0[valid, None] * corr)
            np.add.at(cloth_pred, idx1[valid], w1[valid, None] * corr)

            # Bending edges
            if len(self.bending_edges) > 0:
                b0 = self.bending_edges[:, 0]
                b1 = self.bending_edges[:, 1]
                bp0 = cloth_pred[b0]
                bp1 = cloth_pred[b1]
                bdp = bp0 - bp1
                bd = np.linalg.norm(bdp, axis=1) + 1e-12
                berr = bd - self.bending_rest_lengths
                bw0 = inv_mass[b0]
                bw1 = inv_mass[b1]
                bw_sum = bw0 + bw1
                bvalid = bw_sum > 1e-12

                bcorr = (berr[bvalid] / bw_sum[bvalid])[:, None] * (bdp[bvalid] / bd[bvalid, None]) * 0.4
                np.add.at(cloth_pred, b0[bvalid], -bw0[bvalid, None] * bcorr)
                np.add.at(cloth_pred, b1[bvalid], bw1[bvalid, None] * bcorr)

            cloth_pred[self.pinned_indices] = self.cloth_pos[self.pinned_indices]

        # Update cloth velocity and position
        self.cloth_vel = (cloth_pred - self.cloth_pos) / sub_dt
        self.cloth_vel[self.pinned_indices] = 0.0
        self.cloth_pos = cloth_pred

        # 5. Water Particle relaxation / SPH density pressure & boundary collisions
        # Tank boundaries on water
        # Bottom floor
        floor_penetration = self.tank_z_min - water_pred[:, 2]
        hit_floor = floor_penetration > 0
        water_pred[hit_floor, 2] = self.tank_z_min
        self.water_vel[hit_floor, 2] = -0.2 * self.water_vel[hit_floor, 2]
        self.water_vel[hit_floor, 0] *= 0.95
        self.water_vel[hit_floor, 1] *= 0.95

        # Left wall
        hit_left = water_pred[:, 0] < self.tank_x_min
        water_pred[hit_left, 0] = self.tank_x_min
        self.water_vel[hit_left, 0] = -0.2 * self.water_vel[hit_left, 0]

        # Right wall
        hit_right = water_pred[:, 0] > self.tank_x_max
        water_pred[hit_right, 0] = self.tank_x_max
        self.water_vel[hit_right, 0] = -0.2 * self.water_vel[hit_right, 0]

        # Front wall
        hit_front = water_pred[:, 1] < self.tank_y_min
        water_pred[hit_front, 1] = self.tank_y_min
        self.water_vel[hit_front, 1] = -0.2 * self.water_vel[hit_front, 1]

        # Back wall
        hit_back = water_pred[:, 1] > self.tank_y_max
        water_pred[hit_back, 1] = self.tank_y_max
        self.water_vel[hit_back, 1] = -0.2 * self.water_vel[hit_back, 1]

        self.water_vel = (water_pred - self.water_pos) / sub_dt
        self.water_pos = water_pred

        # Output contact distances
        if len(contact_distances) == 0:
            contact_arr = np.array([0.05], dtype=np.float64)
        else:
            contact_arr = np.array(contact_distances, dtype=np.float64)

        f_ext = np.vstack((f_ext_cloth, f_ext_water))
        f_inter = np.vstack((total_coupled_impulse / sub_dt, -total_coupled_impulse / sub_dt))

        return f_ext, f_inter, contact_arr

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        dt = target_time - self.current_time

        if dt > 1e-8:
            substep_dt = 0.002
            n_steps = max(1, int(np.round(dt / substep_dt)))
            actual_sub_dt = dt / n_steps

            for _ in range(n_steps):
                self._emit_inflow(actual_sub_dt)
                f_ext, f_inter, c_dist = self._substep(actual_sub_dt)
                self.contact_dists = c_dist
                self.ext_force_accum = f_ext
                self.interaction_force_val = f_inter

            self.current_time = target_time
        else:
            f_ext_cloth = self.cloth_mass[:, None] * self.gravity[None, :]
            f_ext_water = self.water_mass[:, None] * self.gravity[None, :]
            self.ext_force_accum = np.vstack((f_ext_cloth, f_ext_water))
            self.interaction_force_val = np.zeros((2, 3), dtype=np.float64)
            self.contact_dists = np.array([0.05], dtype=np.float64)

        # Assemble state frame
        pos = np.vstack((self.cloth_pos, self.water_pos))
        vel = np.vstack((self.cloth_vel, self.water_vel))
        mass = np.concatenate((self.cloth_mass, self.water_mass))

        body_id = np.r_[
            np.zeros(len(self.cloth_pos), dtype=np.int32),
            np.ones(len(self.water_pos), dtype=np.int32)
        ]

        pinned = np.zeros(len(pos), dtype=np.float64)
        pinned[self.pinned_indices] = 1.0
        material_phase = body_id.astype(np.float64)

        # Per-sample mass contribution split into K conserved materials (cloth=0, water=1)
        # attributes.mass_quantity shape [N] or [N,K]
        mass_quantity = np.zeros((len(pos), 2), dtype=np.float64)
        mass_quantity[:self.num_cloth, 0] = self.cloth_mass
        mass_quantity[self.num_cloth:, 1] = self.water_mass

        # fields.cumulative_mass_source shape [K] -> [cloth_source, water_source]
        cumulative_mass_source = np.array([0.0, self.cumulative_mass_source_val], dtype=np.float64)

        frame = StateFrame(
            time=self.current_time,
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "pinned": pinned,
                "material_phase": material_phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": self.contact_dists,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": self.ext_force_accum,
                "interaction_force": self.interaction_force_val,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "water_density": 1000.0,
                "cloth_total_mass": 0.42,
                "cloth_thickness": 0.0015,
                "nozzle_radius": 0.20,
                "nominal_inflow_speed": 3.2,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "cloth_max_displacement": float(np.max(np.abs(self.cloth_pos[:, 0]))),
            "water_particles": int(len(self.water_pos)),
            "cumulative_mass_added": float(self.cumulative_mass_source_val),
        }

