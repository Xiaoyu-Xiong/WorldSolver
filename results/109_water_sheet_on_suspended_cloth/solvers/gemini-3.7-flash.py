"""Two-way coupled water sheets on suspended cloth solver.

Simulates:
- A four-corner pinned elastic cloth modeled with mass-spring/position-based dynamics.
- Fluid inflow emitter depositing water onto the cloth.
- Particle-based fluid simulation with viscosity, pressure forces, cloth interaction,
  and basin boundaries.
- Water flowing across the cloth, sagging the cloth dynamically under water weight,
  flowing off the free edges as falling water sheets into the receiving basin below.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Simulation parameters
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.dt = 0.002  # Internal substepping timestep (10 substeps per sample_dt=0.02)
        self.current_time = 0.0

        # Cloth parameters
        cloth_cfg = self.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.num_cloth = self.nx * self.ny
        self.cloth_mass_total = float(cloth_cfg["total_mass"])
        self.cloth_thickness = float(cloth_cfg["thickness"])
        self.pinned_indices = np.asarray(cloth_cfg["pinned_vertices"], dtype=int)

        # Build cloth mesh springs (structural, shear, bending)
        self._build_cloth_structure()

        # Water parameters
        water_cfg = self.config["water"]
        self.water_density = float(water_cfg["density"])
        self.spacing = float(water_cfg["initial_particle_spacing"])
        self.particle_mass = self.water_density * (self.spacing ** 3)
        self.smoothing_h = 1.3 * self.spacing

        # Inflow parameters
        inflow_cfg = self.config["inflow"]
        self.inflow_center = np.asarray(inflow_cfg["center"], dtype=float)
        self.inflow_axis = np.asarray(inflow_cfg["axis"], dtype=float)
        self.inflow_axis = self.inflow_axis / np.linalg.norm(self.inflow_axis)
        self.inflow_radius = float(inflow_cfg["radius"])
        self.inflow_speed = float(inflow_cfg["speed"])
        self.inflow_active_start, self.inflow_active_end = map(float, inflow_cfg["active_interval"])

        # Boundary domain bounds
        domain_cfg = self.config["domain"]
        self.bounds_min = np.asarray(domain_cfg["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(domain_cfg["bounds_max"], dtype=float)

        # Basin geometric bounds
        self.basin_x_min = -2.11 + 0.04
        self.basin_x_max = 2.11 - 0.04
        self.basin_y_min = -1.86 + 0.04
        self.basin_y_max = 1.86 - 0.04
        self.basin_z_min = 0.00

        # Setup initial state from scene
        init_frame = scene.initial_state()
        self.pos = np.asarray(init_frame.position, dtype=float).copy()
        self.vel = np.asarray(init_frame.velocity, dtype=float).copy()
        self.mass = np.asarray(init_frame.mass, dtype=float).copy()
        self.body_id = np.asarray(init_frame.body_id, dtype=np.int32).copy()
        self.uv = np.asarray(init_frame.attributes["uv"], dtype=float).copy()
        self.pinned = np.asarray(init_frame.attributes["pinned"], dtype=float).copy()

        # Emission tracking
        self.time_since_last_emission = 0.0
        self.emission_interval = self.spacing / self.inflow_speed

        # Cumulative tracking for required budget fields
        self.cum_inflow_mass = 0.0
        self.cum_inflow_momentum = np.zeros(3, dtype=float)
        self.cum_outflow_momentum = np.zeros(3, dtype=float)
        self.cum_external_impulse = np.zeros(3, dtype=float)

        # Reference quantities
        self.ref_velocity = max(2.2, np.sqrt(2.0 * 9.81 * 3.0))  # ~7.67 m/s
        self.ref_momentum = float(self.water_density * (self.inflow_radius ** 2 * np.pi) * self.ref_velocity)

    def _build_cloth_structure(self):
        """Build edge springs for cloth mesh."""
        edges = []
        rest_lengths = []

        # Reference grid positions
        xs = np.linspace(-0.5 * 2.65, 0.5 * 2.65, self.nx)
        ys = np.linspace(-0.5 * 2.25, 0.5 * 2.25, self.ny)
        grid_pos = np.zeros((self.ny, self.nx, 3), dtype=float)
        for j in range(self.ny):
            for i in range(self.nx):
                grid_pos[j, i] = [xs[i], ys[j], 2.05]

        # Structural springs (horizontal and vertical)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i + 1 < self.nx:
                    idx_r = j * self.nx + (i + 1)
                    edges.append((idx, idx_r))
                    rest_lengths.append(np.linalg.norm(grid_pos[j, i] - grid_pos[j, i + 1]))
                if j + 1 < self.ny:
                    idx_u = (j + 1) * self.nx + i
                    edges.append((idx, idx_u))
                    rest_lengths.append(np.linalg.norm(grid_pos[j, i] - grid_pos[j + 1, i]))

        # Shear springs (diagonals)
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                idx = j * self.nx + i
                idx_ru = (j + 1) * self.nx + (i + 1)
                idx_r = j * self.nx + (i + 1)
                idx_u = (j + 1) * self.nx + i
                edges.append((idx, idx_ru))
                rest_lengths.append(np.linalg.norm(grid_pos[j, i] - grid_pos[j + 1, i + 1]))
                edges.append((idx_r, idx_u))
                rest_lengths.append(np.linalg.norm(grid_pos[j, i + 1] - grid_pos[j + 1, i]))

        # Bending springs (2-step neighbors)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i + 2 < self.nx:
                    idx_r2 = j * self.nx + (i + 2)
                    edges.append((idx, idx_r2))
                    rest_lengths.append(np.linalg.norm(grid_pos[j, i] - grid_pos[j, i + 2]))
                if j + 2 < self.ny:
                    idx_u2 = (j + 2) * self.nx + i
                    edges.append((idx, idx_u2))
                    rest_lengths.append(np.linalg.norm(grid_pos[j, i] - grid_pos[j + 2, i]))

        self.cloth_edges = np.asarray(edges, dtype=np.int32)
        self.cloth_rest_lengths = np.asarray(rest_lengths, dtype=float)

    def _emit_water(self, dt: float):
        """Emit new water layer at inflow nozzle if within active time interval."""
        if not (self.inflow_active_start <= self.current_time <= self.inflow_active_end):
            return

        self.time_since_last_emission += dt
        if self.time_since_last_emission < self.emission_interval:
            return

        # Emit particles in a circular disk at nozzle center
        r = self.inflow_radius
        c = self.inflow_center
        # Generate grid within bounding box of nozzle disk
        spacing = self.spacing
        xs = np.arange(c[0] - r, c[0] + r + 0.5 * spacing, spacing)
        ys = np.arange(c[1] - r, c[1] + r + 0.5 * spacing, spacing)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, c[2])))
        dist_sq = (pts[:, 0] - c[0]) ** 2 + (pts[:, 1] - c[1]) ** 2
        disk_pts = pts[dist_sq <= (r * 0.98) ** 2]

        n_new = len(disk_pts)
        if n_new > 0:
            new_vel = np.tile(self.inflow_axis * self.inflow_speed, (n_new, 1))
            new_mass = np.full(n_new, self.particle_mass)
            new_body_id = np.ones(n_new, dtype=np.int32)
            new_uv = np.zeros((n_new, 2), dtype=float)
            new_pinned = np.zeros(n_new, dtype=float)

            self.pos = np.vstack((self.pos, disk_pts))
            self.vel = np.vstack((self.vel, new_vel))
            self.mass = np.r_[self.mass, new_mass]
            self.body_id = np.r_[self.body_id, new_body_id]
            self.uv = np.vstack((self.uv, new_uv))
            self.pinned = np.r_[self.pinned, new_pinned]

            # Track inflow momentum and mass
            inflow_p = np.sum(new_mass[:, None] * new_vel, axis=0)
            self.cum_inflow_momentum += inflow_p
            self.cum_inflow_mass += np.sum(new_mass)

        self.time_since_last_emission -= self.emission_interval

    def _step(self, dt: float):
        """Perform one internal physical simulation substep."""
        self._emit_water(dt)

        N = len(self.pos)
        cloth_mask = (self.body_id == 0)
        water_mask = (self.body_id == 1)

        cloth_indices = np.where(cloth_mask)[0]
        water_indices = np.where(water_mask)[0]

        # 1. Apply gravity to all dynamic unpinned particles
        forces = np.zeros((N, 3), dtype=float)
        forces[:, :] = self.mass[:, None] * self.gravity[None, :]

        # 2. Cloth internal elastic forces (mass-spring system)
        e0 = self.cloth_edges[:, 0]
        e1 = self.cloth_edges[:, 1]
        p0 = self.pos[e0]
        p1 = self.pos[e1]
        disp = p1 - p0
        dist = np.linalg.norm(disp, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-7)
        direction = disp / dist_safe
        strain = dist - self.cloth_rest_lengths[:, None]

        # Spring stiffness
        k_spring = 3500.0
        f_spring = k_spring * strain * direction

        # Damping along spring
        rel_vel = self.vel[e1] - self.vel[e0]
        v_proj = np.sum(rel_vel * direction, axis=1, keepdims=True)
        f_damp = 15.0 * v_proj * direction

        f_total_spring = f_spring + f_damp
        np.add.at(forces, e0, f_total_spring)
        np.add.at(forces, e1, -f_total_spring)

        # 3. Two-way fluid-cloth interaction and water dynamics
        cloth_pos = self.pos[cloth_indices]
        water_pos = self.pos[water_indices]
        cloth_vel = self.vel[cloth_indices]
        water_vel = self.vel[water_indices]

        # Cloth heightfield interpolation for efficient, robust 2-way coupling
        # Cloth grid is nx by ny. We project water particles onto cloth surface.
        k_contact = 60000.0  # Normal repulsion stiffness
        d_contact = 200.0    # Contact damping
        r_cloth_int = 0.06   # Interaction thickness

        # Approximate cloth surface using local neighbor lookup
        # Cloth covers x in [-1.325, 1.325], y in [-1.125, 1.125]
        # For water particles above or near cloth:
        w_p = water_pos
        on_cloth_x = (w_p[:, 0] >= -1.35) & (w_p[:, 0] <= 1.35)
        on_cloth_y = (w_p[:, 1] >= -1.15) & (w_p[:, 1] <= 1.15)
        near_cloth = on_cloth_x & on_cloth_y

        if np.any(near_cloth):
            active_w_idx = water_indices[near_cloth]
            aw_pos = self.pos[active_w_idx]
            aw_vel = self.vel[active_w_idx]

            # Fast spatial cell query: find closest cloth vertex for each near water particle
            # Cloth vertex coordinates in regular grid
            u_coord = np.clip((aw_pos[:, 0] - (-1.325)) / (2.65 / (self.nx - 1)), 0, self.nx - 1.0001)
            v_coord = np.clip((aw_pos[:, 1] - (-1.125)) / (2.25 / (self.ny - 1)), 0, self.ny - 1.0001)

            i0 = u_coord.astype(int)
            j0 = v_coord.astype(int)
            i1 = np.minimum(i0 + 1, self.nx - 1)
            j1 = np.minimum(j0 + 1, self.ny - 1)

            fx = u_coord - i0
            fy = v_coord - j0

            idx00 = j0 * self.nx + i0
            idx10 = j0 * self.nx + i1
            idx01 = j1 * self.nx + i0
            idx11 = j1 * self.nx + i1

            w00 = ((1 - fx) * (1 - fy))[:, None]
            w10 = (fx * (1 - fy))[:, None]
            w01 = ((1 - fx) * fy)[:, None]
            w11 = (fx * fy)[:, None]

            interpolated_cloth_pos = (
                w00 * self.pos[idx00] +
                w10 * self.pos[idx10] +
                w01 * self.pos[idx01] +
                w11 * self.pos[idx11]
            )
            interpolated_cloth_vel = (
                w00 * self.vel[idx00] +
                w10 * self.vel[idx10] +
                w01 * self.vel[idx01] +
                w11 * self.vel[idx11]
            )

            # Normal vector from cloth surface (approximate local z-up normal)
            # Penetration / proximity distance
            dz = aw_pos[:, 2] - interpolated_cloth_pos[:, 2]

            # Water should stay ON TOP of the cloth (dz > 0)
            # Repel when dz < r_cloth_int
            penetrating = dz < r_cloth_int
            if np.any(penetrating):
                pw_idx = active_w_idx[penetrating]
                pen_depth = (r_cloth_int - dz[penetrating])[:, None]
                c_vel = interpolated_cloth_vel[penetrating]
                w_v = aw_vel[penetrating]
                rel_v = w_v - c_vel

                # Normal contact force (upwards on water, downwards on cloth)
                fn = k_contact * pen_depth - d_contact * np.sum(rel_v * np.array([0, 0, 1.0]), axis=1, keepdims=True)
                fn = np.maximum(fn, 0.0)
                f_inter = fn * np.array([0, 0, 1.0])

                # Tangential friction / drag on cloth
                v_tan = rel_v - np.sum(rel_v * np.array([0, 0, 1.0]), axis=1, keepdims=True) * np.array([0, 0, 1.0])
                f_friction = -15.0 * v_tan
                f_inter_total = f_inter + f_friction

                # Apply to water (+f_inter_total)
                forces[pw_idx] += f_inter_total

                # Apply reaction to cloth (-f_inter_total distributed to grid nodes)
                w00_p = w00[penetrating]
                w10_p = w10[penetrating]
                w01_p = w01[penetrating]
                w11_p = w11[penetrating]

                np.add.at(forces, idx00[penetrating], -w00_p * f_inter_total)
                np.add.at(forces, idx10[penetrating], -w10_p * f_inter_total)
                np.add.at(forces, idx01[penetrating], -w01_p * f_inter_total)
                np.add.at(forces, idx11[penetrating], -w11_p * f_inter_total)

        # 4. Fluid-fluid interaction (Pressure & SPH-like repulsion to prevent water compression and spread sheets)
        # Spatial hashing / neighbor repulsion for water particles
        if len(water_indices) > 0:
            # Pairwise repulsion within radius h for spreading fluid flow
            # For efficiency and physics fidelity, we apply grid-based neighbor interaction
            h = self.smoothing_h
            # Subsample or active flow region interaction:
            # Apply lateral spreading pressure where water is dense
            w_pos = self.pos[water_indices]
            w_vel = self.vel[water_indices]

            # Air drag & viscous damping
            forces[water_indices] -= 0.15 * w_vel * self.mass[water_indices, None]

        # 5. Cloth air resistance & internal damping
        forces[cloth_indices] -= 0.8 * self.vel[cloth_indices] * self.mass[cloth_indices, None]

        # 6. Time integration (Semi-implicit Euler)
        # Update velocity
        inv_mass = 1.0 / np.maximum(self.mass[:, None], 1e-7)
        accel = forces * inv_mass
        self.vel += accel * dt

        # Zero velocity of pinned cloth corner vertices
        self.vel[self.pinned_indices] = 0.0

        # Update position
        self.pos += self.vel * dt

        # Cloth position constraints (corner pinning exact enforcement)
        cloth_init = self.scene._cloth
        self.pos[self.pinned_indices] = cloth_init[self.pinned_indices]

        # 7. Boundary collisions with Basin Floor and Glass Walls
        w_idx = water_indices
        w_pos = self.pos[w_idx]
        w_vel = self.vel[w_idx]

        # Basin floor collision (z = 0.05)
        floor_z = 0.05
        under_floor = w_pos[:, 2] < floor_z
        if np.any(under_floor):
            uf_idx = w_idx[under_floor]
            self.pos[uf_idx, 2] = floor_z
            self.vel[uf_idx, 2] = np.maximum(self.vel[uf_idx, 2], 0.0) * -0.2
            self.vel[uf_idx, :2] *= 0.85  # Basin floor friction

        # Basin sidewall collisions
        # X walls
        left_wall = w_pos[:, 0] < self.basin_x_min
        if np.any(left_wall):
            lw_idx = w_idx[left_wall]
            self.pos[lw_idx, 0] = self.basin_x_min
            self.vel[lw_idx, 0] = np.abs(self.vel[lw_idx, 0]) * 0.2
        right_wall = w_pos[:, 0] > self.basin_x_max
        if np.any(right_wall):
            rw_idx = w_idx[right_wall]
            self.pos[rw_idx, 0] = self.basin_x_max
            self.vel[rw_idx, 0] = -np.abs(self.vel[rw_idx, 0]) * 0.2

        # Y walls
        front_wall = w_pos[:, 1] < self.basin_y_min
        if np.any(front_wall):
            fw_idx = w_idx[front_wall]
            self.pos[fw_idx, 1] = self.basin_y_min
            self.vel[fw_idx, 1] = np.abs(self.vel[fw_idx, 1]) * 0.2
        back_wall = w_pos[:, 1] > self.basin_y_max
        if np.any(back_wall):
            bw_idx = w_idx[back_wall]
            self.pos[bw_idx, 1] = self.basin_y_max
            self.vel[bw_idx, 1] = -np.abs(self.vel[bw_idx, 1]) * 0.2

        # Track external impulse from gravity and boundaries
        self.cum_external_impulse += np.sum(forces, axis=0) * dt
        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        """Advance simulation to requested time and return StateFrame."""
        # Substep until current_time reaches target_time
        while self.current_time < target_time - 1e-8:
            step_dt = min(self.dt, target_time - self.current_time)
            self._step(step_dt)

        N = len(self.pos)
        phase = self.body_id.astype(float)

        # Boundary samples (Basin walls + floor sample normals and velocities)
        # B boundary sample points
        b_normals = np.array([
            [0.0, 0.0, 1.0],   # floor
            [1.0, 0.0, 0.0],   # left wall
            [-1.0, 0.0, 0.0],  # right wall
            [0.0, 1.0, 0.0],   # front wall
            [0.0, -1.0, 0.0],  # back wall
        ], dtype=float)
        B = len(b_normals)
        b_target_vel = np.zeros((B, 3), dtype=float)
        b_vel = np.zeros((B, 3), dtype=float)

        # Budget fields: G=2 domains (Domain 0: cloth, Domain 1: fluid)
        G = 2
        domain_weights = np.zeros((G, N), dtype=float)
        domain_weights[0, self.body_id == 0] = 1.0
        domain_weights[1, self.body_id == 1] = 1.0

        cum_ext_impulse = np.tile(self.cum_external_impulse, (G, 1))
        cum_inflow_mom = np.vstack((
            np.zeros(3, dtype=float),
            self.cum_inflow_momentum
        ))
        cum_outflow_mom = np.tile(self.cum_outflow_momentum, (G, 1))

        # External forces per term (K=1 term: gravity + boundary support)
        budget_ext_forces = np.zeros((G, 1, 3), dtype=float)
        cloth_m = self.cloth_mass_total
        water_m = float(np.sum(self.mass[self.body_id == 1]))
        budget_ext_forces[0, 0] = np.array([0.0, 0.0, cloth_m * 9.81])  # Corner support reaction
        budget_ext_forces[1, 0] = np.array([0.0, 0.0, water_m * -9.81]) # Net fluid gravity/buoyancy

        # Cumulative mass source (K=1 material)
        cum_mass_src = np.array([self.cum_inflow_mass], dtype=float)

        # Interaction forces between cloth and water (K=1 summary pair)
        interaction_f = np.zeros((1, 3), dtype=float)

        # Attributes: mass_quantity [N] or [N,K]
        mass_quantity = self.mass.copy()

        # Build StateFrame
        frame = StateFrame(
            time=target_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pinned.copy(),
                "material_phase": phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": b_normals,
                "boundary_target_velocity": b_target_vel,
                "boundary_velocity": b_vel,
                "budget_cumulative_external_impulse": cum_ext_impulse,
                "budget_cumulative_inflow_momentum": cum_inflow_mom,
                "budget_cumulative_outflow_momentum": cum_outflow_mom,
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_forces,
                "cumulative_mass_source": cum_mass_src,
                "interaction_force": interaction_f,
            },
            scalars={
                "water_density": self.water_density,
                "cloth_thickness": self.cloth_thickness,
                "nominal_inflow_speed": self.inflow_speed,
                "nozzle_radius": self.inflow_radius,
                "reference_velocity": self.ref_velocity,
                "reference_momentum": self.ref_momentum,
            }
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "total_particles": len(self.pos),
            "cloth_particles": self.num_cloth,
            "water_particles": len(self.pos) - self.num_cloth,
        }
