"""BenchmarkSolver for deformable bowl impact and liquid squirt into shallow water.

Two-way coupled MPM/particle-based physics simulation with mass-spring/FEM deformable bowl.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load initial StateFrame from scene
        f0 = scene.initial_state()
        self.current_time = 0.0

        # Particle data
        self.pos = f0.position.copy()
        self.vel = f0.velocity.copy()
        self.mass = f0.mass.copy()
        self.body_id = f0.body_id.copy()
        self.N = len(self.pos)

        # Body masks
        self.bowl_mask = (self.body_id == 1)
        self.liquid_mask = (self.body_id == 2)
        self.water_mask = (self.body_id == 3)
        self.fluid_mask = (self.body_id == 2) | (self.body_id == 3)

        self.N_bowl = np.sum(self.bowl_mask)
        self.N_liquid = np.sum(self.liquid_mask)
        self.N_water = np.sum(self.water_mask)

        # Mesh connectivity for bowl
        self.faces = scene.faces.copy()
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                edges_set.add(e)
        self.edges = np.array(list(edges_set), dtype=np.int32)

        # Reference edge rest lengths
        p_bowl0 = self.pos[self.bowl_mask]
        p0 = p_bowl0[self.edges[:, 0]]
        p1 = p_bowl0[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Bending springs across adjacent triangles
        edge_to_faces = {}
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_to_faces.setdefault(e, []).append((face_idx, face[(i + 2) % 3]))

        bending_pairs = []
        for e, flist in edge_to_faces.items():
            if len(flist) == 2:
                v_opp1 = flist[0][1]
                v_opp2 = flist[1][1]
                bending_pairs.append((v_opp1, v_opp2))
        if bending_pairs:
            self.bending_edges = np.array(bending_pairs, dtype=np.int32)
            p_b0 = p_bowl0[self.bending_edges[:, 0]]
            p_b1 = p_bowl0[self.bending_edges[:, 1]]
            self.bending_rest_lengths = np.linalg.norm(p_b1 - p_b0, axis=1)
        else:
            self.bending_edges = np.empty((0, 2), dtype=np.int32)
            self.bending_rest_lengths = np.empty((0,), dtype=float)

        # Simulation parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.tank_min = np.array(self.config["boundary_conditions"]["tank"]["interior_min"], dtype=float)
        self.tank_max = np.array(self.config["boundary_conditions"]["tank"]["interior_max"], dtype=float)

        # Physical material parameters
        self.k_stretch = 1800.0  # Elastic stretch stiffness for rubber bowl
        self.k_bend = 400.0     # Elastic bending stiffness
        self.damp_bowl = 12.0   # Structural damping

        # SPH / Fluid interaction parameters
        self.h_fluid = 0.15     # Smoothing radius
        self.c_sound = 15.0     # Artificial speed of sound (Tait EOS)
        self.rho0 = 1000.0      # Rest density
        self.gamma = 7.0
        self.B_press = (self.rho0 * self.c_sound ** 2) / self.gamma
        self.mu_fluid = 0.05    # Dynamic viscosity

        # Boundary tracking
        self.cumulative_impulse = np.zeros(3, dtype=float)
        self.last_external_force = np.zeros(3, dtype=float)
        self.last_interaction_force = np.zeros((1, 3), dtype=float)
        self.boundary_vel = np.zeros((6, 3), dtype=float)
        self.boundary_target_vel = np.zeros((6, 3), dtype=float)
        self.boundary_norm = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)

    def _step(self, dt: float) -> None:
        pos = self.pos
        vel = self.vel
        mass = self.mass
        bowl_idx = np.where(self.bowl_mask)[0]
        fluid_idx = np.where(self.fluid_mask)[0]

        # 1. Internal elastic forces on bowl
        p_bowl = pos[bowl_idx]
        v_bowl = vel[bowl_idx]
        f_bowl = np.zeros_like(p_bowl)

        # Stretch springs
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        dp = p_bowl[e1] - p_bowl[e0]
        dist = np.linalg.norm(dp, axis=1) + 1e-12
        stretch = dist - self.rest_lengths
        dir_e = dp / dist[:, None]
        dv = v_bowl[e1] - v_bowl[e0]
        damp_force = np.sum(dv * dir_e, axis=1)[:, None] * self.damp_bowl
        force_e = (self.k_stretch * stretch[:, None] + damp_force) * dir_e

        np.add.at(f_bowl, e0, force_e)
        np.add.at(f_bowl, e1, -force_e)

        # Bending springs
        if len(self.bending_edges) > 0:
            be0 = self.bending_edges[:, 0]
            be1 = self.bending_edges[:, 1]
            dp_b = p_bowl[be1] - p_bowl[be0]
            dist_b = np.linalg.norm(dp_b, axis=1) + 1e-12
            stretch_b = dist_b - self.bending_rest_lengths
            dir_b = dp_b / dist_b[:, None]
            dv_b = v_bowl[be1] - v_bowl[be0]
            damp_b = np.sum(dv_b * dir_b, axis=1)[:, None] * (self.damp_bowl * 0.5)
            force_b = (self.k_bend * stretch_b[:, None] + damp_b) * dir_b
            np.add.at(f_bowl, be0, force_b)
            np.add.at(f_bowl, be1, -force_b)

        # 2. SPH Fluid-fluid and fluid-bowl pressure / viscous forces
        f_fluid = np.zeros_like(pos[fluid_idx])
        p_fluid = pos[fluid_idx]
        v_fluid = vel[fluid_idx]
        m_fluid = mass[fluid_idx]

        # Compute fluid density using spatial hash / neighborhood
        h = self.h_fluid
        h2 = h * h
        h3 = h * h * h
        w_poly6_coef = 315.0 / (64.0 * np.pi * (h ** 9))
        grad_spiky_coef = -45.0 / (np.pi * (h ** 6))
        lap_visc_coef = 45.0 / (np.pi * (h ** 6))

        # Grid hashing for fluid interaction
        cell_size = h
        coords = np.floor(p_fluid / cell_size).astype(np.int32)

        grid = {}
        for i, c in enumerate(coords):
            grid.setdefault((c[0], c[1], c[2]), []).append(i)

        # Fluid density
        density = np.zeros(len(p_fluid), dtype=float)
        for c, indices in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_cell = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if neighbor_cell in grid:
                            n_indices = grid[neighbor_cell]
                            for i in indices:
                                r_vec = p_fluid[i] - p_fluid[n_indices]
                                r2 = np.sum(r_vec ** 2, axis=1)
                                mask_h = r2 < h2
                                density[i] += np.sum(m_fluid[n_indices][mask_h] * w_poly6_coef * (h2 - r2[mask_h]) ** 3)

        density = np.maximum(density, self.rho0 * 0.5)
        # Tait equation of state
        pressure = self.B_press * (np.power(density / self.rho0, self.gamma) - 1.0)
        pressure = np.maximum(pressure, 0.0)

        # Fluid forces (pressure gradient + viscosity)
        for c, indices in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_cell = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if neighbor_cell in grid:
                            n_indices = grid[neighbor_cell]
                            for i in indices:
                                r_vec = p_fluid[i] - p_fluid[n_indices]
                                r = np.sqrt(np.sum(r_vec ** 2, axis=1)) + 1e-12
                                mask_h = (r < h) & (r > 1e-5)
                                if np.any(mask_h):
                                    nbrs = np.array(n_indices)[mask_h]
                                    r_sub = r[mask_h]
                                    r_vec_sub = r_vec[mask_h]

                                    # Pressure force (symmetric)
                                    p_term = (pressure[i] / (density[i] ** 2) + pressure[nbrs] / (density[nbrs] ** 2))
                                    grad_w = grad_spiky_coef * ((h - r_sub) ** 2)[:, None] * (r_vec_sub / r_sub[:, None])
                                    f_p = -m_fluid[nbrs][:, None] * p_term[:, None] * grad_w

                                    # Viscosity force
                                    v_diff = v_fluid[nbrs] - v_fluid[i]
                                    lap_w = lap_visc_coef * (h - r_sub)[:, None]
                                    f_v = self.mu_fluid * m_fluid[nbrs][:, None] * (v_diff / density[nbrs][:, None]) * lap_w

                                    f_fluid[i] += np.sum(f_p + f_v, axis=0)

        # 3. Two-way coupling between fluid particles and bowl shell
        # Bowl shell repels fluid particles that penetrate its concave/convex surface,
        # and fluid pressure exerts normal traction back on the bowl mesh.
        # Efficient triangle proximity test:
        tri_v0 = p_bowl[self.faces[:, 0]]
        tri_v1 = p_bowl[self.faces[:, 1]]
        tri_v2 = p_bowl[self.faces[:, 2]]
        tri_centers = (tri_v0 + tri_v1 + tri_v2) / 3.0
        tri_normals = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
        tri_areas = 0.5 * np.linalg.norm(tri_normals, axis=1, keepdims=True)
        tri_unit_normals = tri_normals / (2.0 * np.maximum(tri_areas, 1e-12))

        # Couple fluid particles near bowl triangles
        interaction_force_sum = np.zeros(3, dtype=float)
        shell_thickness = 0.05
        k_repulse = 8000.0
        damp_repulse = 50.0

        # Spatial partition for bowl triangles
        bowl_coords = np.floor(tri_centers / (h * 2.0)).astype(np.int32)
        tri_grid = {}
        for idx_t, c in enumerate(bowl_coords):
            tri_grid.setdefault((c[0], c[1], c[2]), []).append(idx_t)

        for i, pf in enumerate(p_fluid):
            c = np.floor(pf / (h * 2.0)).astype(np.int32)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cell = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if cell in tri_grid:
                            for tid in tri_grid[cell]:
                                n = tri_unit_normals[tid]
                                d_vec = pf - tri_centers[tid]
                                signed_dist = np.dot(d_vec, n)
                                # Distance parallel to face plane
                                in_plane_d = np.linalg.norm(d_vec - signed_dist * n)
                                if in_plane_d < 0.25 and abs(signed_dist) < shell_thickness:
                                    # Penetration depth
                                    pen = (shell_thickness - abs(signed_dist)) * np.sign(signed_dist)
                                    v_rel = v_fluid[i] - (v_bowl[self.faces[tid, 0]] + v_bowl[self.faces[tid, 1]] + v_bowl[self.faces[tid, 2]]) / 3.0
                                    f_c = (k_repulse * pen * n - damp_repulse * np.dot(v_rel, n) * n) * m_fluid[i]

                                    f_fluid[i] += f_c
                                    # Distribute equal and opposite reaction force to the three triangle vertices
                                    f_bowl[self.faces[tid, 0]] -= f_c / 3.0
                                    f_bowl[self.faces[tid, 1]] -= f_c / 3.0
                                    f_bowl[self.faces[tid, 2]] -= f_c / 3.0
                                    interaction_force_sum += f_c

        self.last_interaction_force[0] = interaction_force_sum

        # 4. Integrate accelerations (Gravity + Forces)
        # Bowl
        a_bowl = self.gravity + f_bowl / mass[bowl_idx][:, None]
        vel[bowl_idx] += a_bowl * dt
        pos[bowl_idx] += vel[bowl_idx] * dt

        # Fluid
        a_fluid = self.gravity + f_fluid / mass[fluid_idx][:, None]
        vel[fluid_idx] += a_fluid * dt
        pos[fluid_idx] += vel[fluid_idx] * dt

        # 5. Boundary collisions with tank floor and walls
        external_force_sum = np.zeros(3, dtype=float)

        # Tank floor (z = tank_min[2] = 0.0)
        floor_z = self.tank_min[2]

        # Bowl floor contact & tilt (friction + restitution)
        bowl_floor_mask = pos[bowl_idx, 2] < floor_z + 0.015
        if np.any(bowl_floor_mask):
            p_pen = (floor_z + 0.015) - pos[bowl_idx[bowl_floor_mask], 2]
            f_norm = np.sum(p_pen * 15000.0)
            external_force_sum[2] += f_norm
            self.cumulative_impulse[2] += f_norm * dt

            pos[bowl_idx[bowl_floor_mask], 2] = floor_z + 0.015
            v_hit = vel[bowl_idx[bowl_floor_mask]]
            v_hit[:, 2] = -0.1 * v_hit[:, 2]  # inelastic impact
            v_hit[:, :2] *= 0.8  # ground friction causes tilt
            vel[bowl_idx[bowl_floor_mask]] = v_hit

        # Fluid tank boundaries
        for d in range(3):
            # min bound
            if d == 2:  # floor
                min_b = self.tank_min[2] + 0.01
            else:
                min_b = self.tank_min[d] + 0.05

            p_fluid_d = pos[fluid_idx, d]
            mask_min = p_fluid_d < min_b
            if np.any(mask_min):
                pos[fluid_idx[mask_min], d] = min_b
                vel[fluid_idx[mask_min], d] *= -0.2
                imp = np.sum(mass[fluid_idx[mask_min]] * np.abs(vel[fluid_idx[mask_min], d])) / dt
                external_force_sum[d] += imp
                self.cumulative_impulse[d] += imp * dt

            # max bound (walls)
            if d < 2:  # open top for z
                max_b = self.tank_max[d] - 0.05
                p_fluid_d_max = pos[fluid_idx, d]
                mask_max = p_fluid_d_max > max_b
                if np.any(mask_max):
                    pos[fluid_idx[mask_max], d] = max_b
                    vel[fluid_idx[mask_max], d] *= -0.2
                    imp = np.sum(mass[fluid_idx[mask_max]] * np.abs(vel[fluid_idx[mask_max], d])) / dt
                    external_force_sum[d] -= imp
                    self.cumulative_impulse[d] -= imp * dt

        self.last_external_force = external_force_sum

    def advance_to(self, time: float) -> StateFrame:
        # Substep integration for numerical stability
        dt_sub = 0.001
        while self.current_time < time - 1e-7:
            step = min(dt_sub, time - self.current_time)
            self._step(step)
            self.current_time += step

        self.current_time = time

        # Construct StateFrame with all required trajectory fields
        N = self.N
        G = 1  # 1 global budget domain
        K = 1
        B = 6

        mass_quantity = self.mass.copy()
        budget_domain_weights = np.ones((G, N), dtype=float)
        budget_cumulative_external_impulse = np.zeros((G, 3), dtype=float)
        budget_cumulative_external_impulse[0] = self.cumulative_impulse
        budget_cumulative_inflow_momentum = np.zeros((G, 3), dtype=float)
        budget_cumulative_outflow_momentum = np.zeros((G, 3), dtype=float)
        budget_external_force = np.zeros((G, K, 3), dtype=float)
        budget_external_force[0, 0] = self.last_external_force
        cumulative_mass_source = np.zeros(K, dtype=float)
        interaction_force = self.last_interaction_force.copy()

        total_mass = np.sum(self.mass)
        ref_vel = 2.5
        ref_mom = float(total_mass * ref_vel)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": self.boundary_norm.copy(),
                "boundary_target_velocity": self.boundary_target_vel.copy(),
                "boundary_velocity": self.boundary_vel.copy(),
                "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
                "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
                "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "gravity_z": float(self.gravity[2]),
                "bowl_impermeable": 1.0,
                "shallow_water_surface_z": 0.22,
                "reference_velocity": ref_vel,
                "reference_momentum": ref_mom,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {}

