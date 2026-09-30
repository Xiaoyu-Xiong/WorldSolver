"""BenchmarkSolver implementation for elastic bag filling and overflow."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Coupled Lagrangian particle-fluid and mass-spring/cloth shell solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Load bag mesh
        asset = np.load(Path(__file__).resolve().parent.parent / "assets" / "elastic_bag.npz")
        self.bag_v_rest = np.asarray(asset["vertices"], dtype=np.float64)
        self.bag_faces = np.asarray(asset["faces"], dtype=np.int32)
        self.rim_indices = np.asarray(asset["rim_indices"], dtype=np.int32)
        self.num_bag = len(self.bag_v_rest)

        # Extract mesh edges for springs
        edges_set = set()
        for f in self.bag_faces:
            for i in range(3):
                u, v = sorted([f[i], f[(i + 1) % 3]])
                edges_set.add((u, v))
        self.edges = np.array(list(edges_set), dtype=np.int32)
        v0 = self.bag_v_rest[self.edges[:, 0]]
        v1 = self.bag_v_rest[self.edges[:, 1]]
        self.edge_rest_lens = np.linalg.norm(v1 - v0, axis=1)

        # Bag properties
        self.bag_total_mass = float(self.config["initial_conditions"]["bag"]["total_mass"])
        self.bag_thickness = float(self.config["initial_conditions"]["bag"]["thickness"])
        self.bag_m = np.full(self.num_bag, self.bag_total_mass / self.num_bag, dtype=np.float64)
        self.bag_x = self.bag_v_rest.copy()
        self.bag_v = np.zeros_like(self.bag_x)

        # Bag cloth material stiffness & damping
        self.k_stretch = 1800.0  # Spring stiffness
        self.d_stretch = 15.0    # Spring damping
        self.bag_air_drag = 0.5

        # Fluid properties
        self.water_density = float(self.config["initial_conditions"]["water"]["rest_density"])
        self.spacing = 0.055
        self.particle_mass = self.water_density * (self.spacing ** 3)  # ~0.166 kg
        self.fluid_radius = self.spacing * 0.55  # collision radius
        self.fluid_viscosity = 2.0
        self.fluid_pressure_k = 2000.0  # Tait / EOS stiffness
        self.fluid_h = self.spacing * 1.8  # smoothing length

        # Inflow specification
        inflow = self.config["boundary_conditions"]["nozzle_inflow"]
        self.nozzle_center = np.array(inflow["center"], dtype=np.float64)
        self.nozzle_dir = np.array(inflow["direction"], dtype=np.float64)
        self.nozzle_dir = self.nozzle_dir / np.linalg.norm(self.nozzle_dir)
        self.nozzle_radius = float(inflow["radius"])
        self.nozzle_speed = float(inflow["speed"])
        self.inflow_interval = [float(inflow["active_interval"][0]), float(inflow["active_interval"][1])]

        # Initialize fluid particles matching scene.initial_state()
        init_frame = scene.initial_state()
        init_pos = np.asarray(init_frame.position, dtype=np.float64)
        init_vel = np.asarray(init_frame.velocity, dtype=np.float64)
        init_bid = np.asarray(init_frame.body_id, dtype=np.int32)

        fluid_mask = (init_bid == 1)
        if np.any(fluid_mask):
            self.fluid_x = init_pos[fluid_mask].copy()
            self.fluid_v = init_vel[fluid_mask].copy()
            self.fluid_m = np.full(len(self.fluid_x), self.particle_mass, dtype=np.float64)
        else:
            self.fluid_x = np.empty((0, 3), dtype=np.float64)
            self.fluid_v = np.empty((0, 3), dtype=np.float64)
            self.fluid_m = np.empty(0, dtype=np.float64)

        # Simulation time & time step
        self.time = 0.0
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)
        self.sub_dt = 0.001  # 1 ms sub-step for numerical stability

        # Cumulative accounting budgets
        self.cumulative_inflow_momentum = np.zeros(3, dtype=np.float64)
        self.cumulative_outflow_momentum = np.zeros(3, dtype=np.float64)
        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.cumulative_mass_added = float(len(self.fluid_x) * self.particle_mass)

        # Reference scales
        self.ref_velocity = float(self.nozzle_speed)
        self.ref_momentum = float(self.bag_total_mass * self.ref_velocity)

        # Boundary specifications
        self.floor_z = float(self.config["boundary_conditions"]["exterior"]["catch_floor_z"])
        self.rim_mask = np.zeros(self.num_bag, dtype=bool)
        self.rim_mask[self.rim_indices] = True

        # Inflow timing accumulator
        self.inflow_timer = 0.0
        self.inflow_period = self.spacing / self.nozzle_speed  # time between layers

        # Track latest interaction force
        self.last_interaction_force = np.zeros(3, dtype=np.float64)
        self.last_external_force_bag = np.zeros(3, dtype=np.float64)
        self.last_external_force_fluid = np.zeros(3, dtype=np.float64)

    def _generate_inflow_layer(self) -> np.ndarray:
        axis = np.arange(-self.nozzle_radius + self.spacing / 2, self.nozzle_radius, self.spacing)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        points_2d = np.column_stack((xx.ravel(), yy.ravel()))
        valid = (points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2) <= (self.nozzle_radius - self.spacing / 4) ** 2
        pts = points_2d[valid]
        z = np.full(len(pts), self.nozzle_center[2] - 0.02)
        return np.column_stack((pts[:, 0] + self.nozzle_center[0],
                                pts[:, 1] + self.nozzle_center[1],
                                z))

    def _step(self, dt: float) -> None:
        # Inflow generation
        if self.inflow_interval[0] <= self.time <= self.inflow_interval[1]:
            self.inflow_timer += dt
            if self.inflow_timer >= self.inflow_period:
                self.inflow_timer -= self.inflow_period
                new_pts = self._generate_inflow_layer()
                if len(new_pts) > 0:
                    new_vel = np.tile(self.nozzle_dir * self.nozzle_speed, (len(new_pts), 1))
                    new_mass = np.full(len(new_pts), self.particle_mass)
                    self.fluid_x = np.vstack((self.fluid_x, new_pts))
                    self.fluid_v = np.vstack((self.fluid_v, new_vel))
                    self.fluid_m = np.concatenate((self.fluid_m, new_mass))
                    added_mass = len(new_pts) * self.particle_mass
                    self.cumulative_mass_added += added_mass
                    inflow_p = np.sum(new_vel * self.particle_mass, axis=0)
                    self.cumulative_inflow_momentum += inflow_p

        # 1. Internal Bag Forces (Springs)
        f_bag = np.zeros_like(self.bag_x)
        idx0 = self.edges[:, 0]
        idx1 = self.edges[:, 1]
        dp = self.bag_x[idx1] - self.bag_x[idx0]
        dist = np.linalg.norm(dp, axis=1, keepdims=True) + 1e-12
        direction = dp / dist
        strain = (dist[:, 0] - self.edge_rest_lens)
        f_spring_mag = self.k_stretch * strain

        dv = self.bag_v[idx1] - self.bag_v[idx0]
        v_rel_proj = np.sum(dv * direction, axis=1)
        f_damp_mag = self.d_stretch * v_rel_proj

        f_total = (f_spring_mag + f_damp_mag)[:, None] * direction
        np.add.at(f_bag, idx0, f_total)
        np.add.at(f_bag, idx1, -f_total)

        # Bag gravity & drag
        f_bag += self.bag_m[:, None] * self.gravity
        f_bag -= self.bag_air_drag * self.bag_v

        # 2. Fluid Particle Forces
        n_fluid = len(self.fluid_x)
        f_fluid = np.zeros_like(self.fluid_x)
        f_fluid_to_bag = np.zeros_like(self.bag_x)

        if n_fluid > 0:
            f_fluid += self.fluid_m[:, None] * self.gravity

            # Fluid-fluid short range repulsion / pressure approximation (grid-based SPH / penalty)
            # Fast spatial hash for fluid-fluid and fluid-bag interactions
            cell_size = self.fluid_h
            # SPH-like local pairwise repulsion to enforce quasi-incompressibility
            # For efficiency and robustness: cell bucketing
            coords = np.floor(self.fluid_x / cell_size).astype(np.int32)
            # Create a simple hash key
            keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
            sort_idx = np.argsort(keys)
            keys_sorted = keys[sort_idx]
            fluid_x_s = self.fluid_x[sort_idx]
            fluid_v_s = self.fluid_v[sort_idx]

            # Neighbor search within adjacent keys
            # Compute density / pairwise repulsion
            unique_keys, start_indices, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
            for k_idx, ukey in enumerate(unique_keys):
                s_i = start_indices[k_idx]
                c_i = counts[k_idx]
                if c_i <= 1:
                    continue
                block_x = fluid_x_s[s_i:s_i+c_i]
                block_v = fluid_v_s[s_i:s_i+c_i]
                # Pairwise within cell
                diff = block_x[:, None, :] - block_x[None, :, :]
                dists = np.linalg.norm(diff, axis=2) + 1e-12
                overlap = (self.spacing - dists)
                mask = (overlap > 0) & (dists > 1e-6)
                if np.any(mask):
                    repulse = 400.0 * overlap[mask, None] * (diff[mask] / dists[mask, None])
                    # Viscosity damping
                    dv_block = block_v[:, None, :] - block_v[None, :, :]
                    damp = self.fluid_viscosity * dv_block[mask]
                    f_pair = repulse - damp
                    # Accumulate back
                    orig_indices = sort_idx[s_i:s_i+c_i]
                    i_grid, j_grid = np.meshgrid(orig_indices, orig_indices, indexing="ij")
                    np.add.at(f_fluid, i_grid[mask], f_pair)

            # Fluid-Cloth Bag Two-Way Coupling
            # Find nearest bag vertices to fluid particles
            # Bag vertices spatial lookup or vectorized chunking
            # Bag has ~545 vertices, vectorized distance calculation
            b_x = self.bag_x  # (545, 3)
            # Compute distance matrix between fluid and bag in chunks if needed
            chunk_size = 500
            for start in range(0, n_fluid, chunk_size):
                end = min(start + chunk_size, n_fluid)
                f_chunk = self.fluid_x[start:end]  # (C, 3)
                # (C, 545, 3)
                diff_fb = f_chunk[:, None, :] - b_x[None, :, :]
                dist_fb = np.linalg.norm(diff_fb, axis=2)  # (C, 545)

                # Collision envelope
                collision_dist = self.fluid_radius + self.bag_thickness
                contact = dist_fb < collision_dist
                if np.any(contact):
                    c_f, c_b = np.where(contact)
                    d_cb = dist_fb[c_f, c_b, None] + 1e-12
                    normal = diff_fb[c_f, c_b] / d_cb
                    penetration = collision_dist - d_cb

                    # Normal penalty force
                    k_contact = 6000.0
                    f_c_mag = k_contact * penetration

                    # Relative velocity
                    v_rel = self.fluid_v[start + c_f] - self.bag_v[c_b]
                    v_n = np.sum(v_rel * normal, axis=1, keepdims=True)
                    f_damp = 30.0 * np.minimum(v_n, 0.0) * normal

                    # Friction / shear coupling
                    v_t = v_rel - v_n * normal
                    f_fric = -5.0 * v_t

                    f_contact = f_c_mag * normal - f_damp + f_fric

                    # Action-reaction: fluid gets +f_contact, bag gets -f_contact
                    np.add.at(f_fluid, start + c_f, f_contact)
                    np.add.at(f_bag, c_b, -f_contact)
                    np.add.at(f_fluid_to_bag, c_b, -f_contact)

        # Record interaction force
        self.last_interaction_force = np.sum(f_fluid_to_bag, axis=0)
        self.last_external_force_bag = np.sum(f_bag, axis=0)
        self.last_external_force_fluid = np.sum(f_fluid, axis=0) if n_fluid > 0 else np.zeros(3)

        # 3. Time Integration: Bag (Symplectic Euler)
        a_bag = f_bag / self.bag_m[:, None]
        self.bag_v += a_bag * dt
        # Pinned rim boundary condition
        self.bag_v[self.rim_mask] = 0.0
        self.bag_x += self.bag_v * dt
        # Enforce exact pinned rim positions
        self.bag_x[self.rim_mask] = self.bag_v_rest[self.rim_mask]

        # Floor collision for bag
        below_floor = self.bag_x[:, 2] < self.floor_z
        if np.any(below_floor):
            self.bag_x[below_floor, 2] = self.floor_z
            self.bag_v[below_floor, 2] = np.maximum(0.0, self.bag_v[below_floor, 2])
            self.bag_v[below_floor, :2] *= 0.5  # friction

        # 4. Time Integration: Fluid
        if n_fluid > 0:
            a_fluid = f_fluid / self.fluid_m[:, None]
            self.fluid_v += a_fluid * dt
            self.fluid_x += self.fluid_v * dt

            # Floor collision for fluid
            f_below = self.fluid_x[:, 2] < (self.floor_z + self.fluid_radius)
            if np.any(f_below):
                self.fluid_x[f_below, 2] = self.floor_z + self.fluid_radius
                self.fluid_v[f_below, 2] = np.maximum(0.0, -0.1 * self.fluid_v[f_below, 2])
                self.fluid_v[f_below, :2] *= 0.8  # floor friction

            # Cumulative external impulse accounting
            ext_force = np.sum(self.fluid_m[:, None] * self.gravity, axis=0) + np.sum(self.bag_m[:, None] * self.gravity, axis=0)
            self.cumulative_external_impulse += ext_force * dt

        self.time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Advance simulation in small sub-steps until target_time
        while self.time < target_time - 1e-9:
            dt = min(self.sub_dt, target_time - self.time)
            self._step(dt)

        # Assemble full state frame
        pos = np.vstack((self.bag_x, self.fluid_x)) if len(self.fluid_x) > 0 else self.bag_x.copy()
        vel = np.vstack((self.bag_v, self.fluid_v)) if len(self.fluid_x) > 0 else self.bag_v.copy()
        mass = np.concatenate((self.bag_m, self.fluid_m)) if len(self.fluid_x) > 0 else self.bag_m.copy()
        body_id = np.r_[np.zeros(self.num_bag, dtype=np.int32), np.ones(len(self.fluid_x), dtype=np.int32)]

        pinned = np.zeros(len(pos), dtype=np.float64)
        pinned[self.rim_indices] = 1.0
        phase = body_id.astype(np.float64)

        # Boundary arrays (at fixed rim supports + floor)
        # B boundary samples: rim vertices (32) + 4 floor corner samples = 36 samples
        b_rim_pos = self.bag_v_rest[self.rim_indices]
        b_rim_norm = np.tile(np.array([0.0, 0.0, 1.0]), (len(self.rim_indices), 1))
        b_floor_norm = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
        boundary_normal = np.vstack((b_rim_norm, b_floor_norm))
        boundary_vel = np.zeros((len(boundary_normal), 3), dtype=np.float64)
        boundary_target_vel = np.zeros((len(boundary_normal), 3), dtype=np.float64)

        # Budget fields: G = 2 domains (0: bag, 1: fluid)
        N = len(pos)
        budget_weights = np.zeros((2, N), dtype=np.float64)
        budget_weights[0, :self.num_bag] = 1.0
        if len(self.fluid_x) > 0:
            budget_weights[1, self.num_bag:] = 1.0

        budget_ext_force = np.zeros((2, 1, 3), dtype=np.float64)
        budget_ext_force[0, 0] = self.last_external_force_bag
        budget_ext_force[1, 0] = self.last_external_force_fluid

        budget_cum_impulse = np.zeros((2, 3), dtype=np.float64)
        budget_cum_impulse[0] = self.cumulative_external_impulse * (self.bag_total_mass / (self.bag_total_mass + max(1e-6, self.cumulative_mass_added)))
        budget_cum_impulse[1] = self.cumulative_external_impulse * (self.cumulative_mass_added / (self.bag_total_mass + max(1e-6, self.cumulative_mass_added)))

        budget_cum_inflow = np.zeros((2, 3), dtype=np.float64)
        budget_cum_inflow[1] = self.cumulative_inflow_momentum

        budget_cum_outflow = np.zeros((2, 3), dtype=np.float64)
        budget_cum_outflow[1] = self.cumulative_outflow_momentum

        cumulative_mass_src = np.array([self.cumulative_mass_added], dtype=np.float64)
        interaction_f = self.last_interaction_force.reshape(1, 3)

        frame = StateFrame(
            time=float(self.time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "pinned": pinned,
                "material_phase": phase,
                "mass_quantity": mass,
            },
            fields={
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_target_vel,
                "boundary_velocity": boundary_vel,
                "budget_cumulative_external_impulse": budget_cum_impulse,
                "budget_cumulative_inflow_momentum": budget_cum_inflow,
                "budget_cumulative_outflow_momentum": budget_cum_outflow,
                "budget_domain_weights": budget_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": cumulative_mass_src,
                "interaction_force": interaction_f,
            },
            scalars={
                "water_rest_density": float(self.water_density),
                "bag_total_mass": float(self.bag_total_mass),
                "bag_thickness": float(self.bag_thickness),
                "nozzle_radius": float(self.nozzle_radius),
                "nominal_inflow_speed": float(self.nozzle_speed),
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.ref_velocity),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_fluid_particles": len(self.fluid_x),
            "sim_time": self.time,
        }

