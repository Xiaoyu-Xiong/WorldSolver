"""Buoyancy contrast of porous cloth pieces solver implementation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Two-way coupled porous cloth in water solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.RandomState(seed)

        # Initial frame
        init_frame = scene.initial_state()
        self.num_particles = len(init_frame.position)
        self.position = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        self.attributes = {k: np.array(v, copy=True) for k, v in init_frame.attributes.items()}
        self.fields = {k: np.array(v, copy=True) for k, v in init_frame.fields.items()}
        self.scalars = dict(init_frame.scalars)

        self.current_time = 0.0

        # Physical constants and parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.water_density = float(self.config["water"]["density"])
        self.free_surface_z = float(self.config["water"]["free_surface_z"])
        self.tank_min = np.array(self.config["tank"]["interior_bounds_min"], dtype=np.float64)
        self.tank_max = np.array(self.config["tank"]["interior_bounds_max"], dtype=np.float64)

        # Cloth parameters
        cloth_cfg = self.config["cloth"]
        self.cloth_thickness = float(cloth_cfg["thickness"])
        self.cloth_porosity = float(cloth_cfg["porosity"])
        self.cloth_perm = float(cloth_cfg["permeability_m2"])
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.cloth_size = np.array(cloth_cfg["size"], dtype=np.float64)

        # Particle indices
        self.water_mask = (self.body_id == 0)
        self.water_indices = np.where(self.water_mask)[0]
        self.cloth_masks = {b: (self.body_id == b) for b in (1, 2, 3)}
        self.cloth_indices = {b: np.where(self.cloth_masks[b])[0] for b in (1, 2, 3)}

        # Cloth mesh connectivity for structural springs
        self.springs = {}
        for b in (1, 2, 3):
            idxs = self.cloth_indices[b]
            spring_list = []
            # Grid layout: ix in [0..nx-1], iy in [0..ny-1], index in local = iy * nx + ix
            for iy in range(self.ny):
                for ix in range(self.nx):
                    idx = idxs[iy * self.nx + ix]
                    # Structural springs: right and up
                    if ix + 1 < self.nx:
                        idx_r = idxs[iy * self.nx + (ix + 1)]
                        L0 = np.linalg.norm(self.position[idx_r] - self.position[idx])
                        spring_list.append((idx, idx_r, L0, 150.0))  # (i, j, rest_len, stiffness)
                    if iy + 1 < self.ny:
                        idx_u = idxs[(iy + 1) * self.nx + ix]
                        L0 = np.linalg.norm(self.position[idx_u] - self.position[idx])
                        spring_list.append((idx, idx_u, L0, 150.0))
                    # Shear springs: diagonals
                    if ix + 1 < self.nx and iy + 1 < self.ny:
                        idx_diag1 = idxs[(iy + 1) * self.nx + (ix + 1)]
                        L0 = np.linalg.norm(self.position[idx_diag1] - self.position[idx])
                        spring_list.append((idx, idx_diag1, L0, 80.0))
                    if ix + 1 < self.nx and iy - 1 >= 0:
                        idx_diag2 = idxs[(iy - 1) * self.nx + (ix + 1)]
                        L0 = np.linalg.norm(self.position[idx_diag2] - self.position[idx])
                        spring_list.append((idx, idx_diag2, L0, 80.0))
                    # Bending springs: 2 hops
                    if ix + 2 < self.nx:
                        idx_b2x = idxs[iy * self.nx + (ix + 2)]
                        L0 = np.linalg.norm(self.position[idx_b2x] - self.position[idx])
                        spring_list.append((idx, idx_b2x, L0, 30.0))
                    if iy + 2 < self.ny:
                        idx_b2y = idxs[(iy + 2) * self.nx + ix]
                        L0 = np.linalg.norm(self.position[idx_b2y] - self.position[idx])
                        spring_list.append((idx, idx_b2y, L0, 30.0))
            self.springs[b] = spring_list

        # Cloth vertex volume (approximate per vertex: area_per_vertex * thickness)
        num_vertices_per_cloth = self.nx * self.ny
        area_per_vertex = (self.cloth_size[0] * self.cloth_size[1]) / num_vertices_per_cloth
        self.cloth_vertex_volume = area_per_vertex * self.cloth_thickness

        # Simulation sub-stepping
        self.sub_dt = 0.002

        # Cumulative budget tracking
        # G = 4 domains: Global(0), Light(1), Neutral(2), Heavy(3)
        # K = 1 term for mass/force
        self.G = 4
        self.budget_cumulative_ext_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.budget_cumulative_inflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.budget_cumulative_outflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros((1,), dtype=np.float64)
        self.reference_momentum = float(np.sum(self.mass) * 1.0)  # Reference scale ~ M * U_ref

    def _compute_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute forces acting on all particles.

        Returns:
            total_force: [N, 3]
            budget_ext_force: [G, 1, 3]
            interaction_force: [1, 3]
        """
        N = self.num_particles
        forces = np.zeros((N, 3), dtype=np.float64)

        # 1. Gravity on all particles
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        forces += gravity_force

        # 2. Buoyancy & Hydrodynamic drag & Two-way coupling on cloth vertices
        # For cloth vertices:
        # Effective displaced fluid volume = cloth_vertex_volume (solid fraction = 1 - porosity or effective displaced)
        # Net buoyant force = (rho_water * Vol - m_cloth) * g
        # When fully immersed: F_buoyancy = - rho_water * Vol * g (upwards)
        cloth_water_interaction = np.zeros(3, dtype=np.float64)

        # Estimate local water velocity near cloth pieces
        # For efficiency, compute mean water velocity in surrounding zones or near cloth
        water_pos = self.position[self.water_mask]
        water_vel = self.velocity[self.water_mask]

        for b in (1, 2, 3):
            idxs = self.cloth_indices[b]
            pos_b = self.position[idxs]
            vel_b = self.velocity[idxs]

            # Immersion factor: 1.0 below free surface, tapering to 0 above
            depth = self.free_surface_z - pos_b[:, 2]
            immersed = np.clip(depth / 0.05, 0.0, 1.0)  # Smooth immersion near surface

            # Buoyancy force: rho_water * V_vertex * (-g) * immersed
            # Note: For porous cloth, buoyancy is rho_water * V_vertex * (-g)
            # Since gravity force is m * g = rho_cloth * V_vertex * g,
            # Net z-force is (rho_water - rho_cloth) * V_vertex * (-g_z)
            # rho_light = 250 < 1000 -> net upward
            # rho_neutral = 1000 == 1000 -> net zero
            # rho_heavy = 4000 > 1000 -> net downward
            f_buoy = np.zeros_like(pos_b)
            f_buoy[:, 2] = -self.water_density * self.cloth_vertex_volume * self.gravity[2] * immersed
            forces[idxs] += f_buoy

            # Hydrodynamic drag / Darcy permeable drag
            # F_drag = - 0.5 * rho * C_d * A * |v_rel| * v_rel - Darcy drag
            # Permeable cloth drag: F_drag = - mu / k * Area * (v_cloth - v_fluid)
            # Effective drag damping coefficient:
            v_rel = vel_b  # Assuming quiescent water background with local induced flow
            speed = np.linalg.norm(v_rel, axis=1, keepdims=True)
            drag_coeff = 4.0 + 8.0 * speed  # Quadratic + linear drag
            f_drag = -drag_coeff * self.cloth_vertex_volume * self.water_density * v_rel * immersed[:, None]
            forces[idxs] += f_drag

            # Two-way momentum coupling to nearby water particles
            cloth_water_interaction -= np.sum(f_drag, axis=0)

            # Internal cloth elastic forces (Springs)
            for idx1, idx2, L0, k_spring in self.springs[b]:
                d = self.position[idx2] - self.position[idx1]
                dist = np.linalg.norm(d)
                if dist > 1e-8:
                    dir_vec = d / dist
                    # Hookean spring with damping
                    rel_v = self.velocity[idx2] - self.velocity[idx1]
                    damping = 2.0 * np.dot(rel_v, dir_vec)
                    f_s = (k_spring * (dist - L0) + damping) * dir_vec
                    forces[idx1] += f_s
                    forces[idx2] -= f_s

        # 3. Water particles dynamics (SPH-like pressure gradient & viscous damping)
        # Water particles maintain level and respond to gravity & tank walls
        # Hydrostatic equilibrium + damping for water
        w_pos = self.position[self.water_mask]
        w_vel = self.velocity[self.water_mask]
        # Pressure force balances gravity in hydrostatic state
        # In water column, dP/dz = -rho*g -> net pressure force upwards = -m * g for bulk water
        # Add slight relaxation / viscosity to water particles
        f_water_internal = -0.5 * w_vel * self.mass[self.water_mask, None]
        # Add upward hydrostatic support force to water so it doesn't just collapse
        depth_w = np.clip(self.free_surface_z - w_pos[:, 2], 0.0, None)
        # Hydrostatic support opposes gravity when below surface
        f_hydrostatic = np.zeros_like(w_pos)
        f_hydrostatic[:, 2] = -self.gravity[2] * self.mass[self.water_mask] * np.clip(depth_w / 0.1, 0.0, 1.0)
        forces[self.water_mask] += f_hydrostatic + f_water_internal

        # Distribute cloth-water interaction back onto water particles
        if np.linalg.norm(cloth_water_interaction) > 1e-8:
            forces[self.water_mask] += cloth_water_interaction[None, :] / max(1, len(w_pos))

        # Budget external forces per domain [G, 1, 3]
        # Domain 0: Global (all particles)
        # Domain 1: Light cloth
        # Domain 2: Neutral cloth
        # Domain 3: Heavy cloth
        budget_ext_force = np.zeros((self.G, 1, 3), dtype=np.float64)
        budget_ext_force[0, 0] = np.sum(forces, axis=0)
        for b in (1, 2, 3):
            idxs = self.cloth_indices[b]
            budget_ext_force[b, 0] = np.sum(forces[idxs], axis=0)

        # Interaction force [K, 3] = [1, 3]
        interaction_force = cloth_water_interaction.reshape(1, 3)

        return forces, budget_ext_force, interaction_force

    def _step(self, dt: float) -> None:
        """Advance physics state by time step dt."""
        forces, budget_ext_force, _ = self._compute_forces()

        # Symplectic Euler integration
        acc = forces / self.mass[:, None]
        self.velocity += acc * dt
        self.position += self.velocity * dt

        # Boundary collisions (Tank walls)
        # Interior bounds: [-2.35, -1.0, 0.0] to [2.35, 1.0, 2.45]
        margin = 0.02
        for d in range(3):
            min_val = self.tank_min[d] + margin
            max_val = self.tank_max[d] - margin
            if d == 2:
                # Floor collision
                below_floor = self.position[:, d] < min_val
                if np.any(below_floor):
                    self.position[below_floor, d] = min_val
                    self.velocity[below_floor, d] = np.maximum(0.0, self.velocity[below_floor, d])
                    self.velocity[below_floor, :] *= 0.8  # Friction
                # Free surface limit for light cloth / water splash
                above_top = self.position[:, d] > (self.free_surface_z + 0.02)
                # Light cloth floats at surface
                for b in (1, 2, 3):
                    cloth_idxs = self.cloth_indices[b]
                    at_surf = cloth_idxs[self.position[cloth_idxs, 2] > (self.free_surface_z - 0.01)]
                    if len(at_surf) > 0 and b == 1:
                        # Light cloth rest at surface
                        self.position[at_surf, 2] = np.minimum(self.position[at_surf, 2], self.free_surface_z)
                        self.velocity[at_surf, 2] = np.minimum(0.0, self.velocity[at_surf, 2])
            else:
                # X and Y walls
                hit_min = self.position[:, d] < min_val
                if np.any(hit_min):
                    self.position[hit_min, d] = min_val
                    self.velocity[hit_min, d] = np.maximum(0.0, -0.2 * self.velocity[hit_min, d])
                hit_max = self.position[:, d] > max_val
                if np.any(hit_max):
                    self.position[hit_max, d] = max_val
                    self.velocity[hit_max, d] = np.minimum(0.0, -0.2 * self.velocity[hit_max, d])

        # Accumulate impulse: integral of external force over time
        for g in range(self.G):
            self.budget_cumulative_ext_impulse[g] += budget_ext_force[g, 0] * dt

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the target time and return StateFrame."""
        target_time = float(time)
        while self.current_time + 1e-8 < target_time:
            step_dt = min(self.sub_dt, target_time - self.current_time)
            if step_dt <= 1e-9:
                break
            self._step(step_dt)

        self.current_time = target_time

        # Compute forces at current time for exported budget fields
        forces, budget_ext_force, interaction_force = self._compute_forces()

        # Domain weights [G, N]
        domain_weights = np.zeros((self.G, self.num_particles), dtype=np.float64)
        domain_weights[0, :] = 1.0  # Global
        for b in (1, 2, 3):
            domain_weights[b, self.body_id == b] = 1.0

        # Construct export attributes, fields, scalars
        exported_attributes = dict(self.attributes)
        exported_attributes["mass_quantity"] = np.array(self.mass, copy=True)

        exported_fields = dict(self.fields)
        exported_fields["budget_cumulative_external_impulse"] = np.array(self.budget_cumulative_ext_impulse, copy=True)
        exported_fields["budget_cumulative_inflow_momentum"] = np.array(self.budget_cumulative_inflow_momentum, copy=True)
        exported_fields["budget_cumulative_outflow_momentum"] = np.array(self.budget_cumulative_outflow_momentum, copy=True)
        exported_fields["budget_domain_weights"] = domain_weights
        exported_fields["budget_external_force"] = np.array(budget_ext_force, copy=True)
        exported_fields["cumulative_mass_source"] = np.array(self.cumulative_mass_source, copy=True)
        exported_fields["interaction_force"] = np.array(interaction_force, copy=True)

        exported_scalars = dict(self.scalars)
        exported_scalars["reference_momentum"] = float(self.reference_momentum)

        frame = StateFrame(
            time=float(target_time),
            position=np.array(self.position, copy=True),
            velocity=np.array(self.velocity, copy=True),
            mass=np.array(self.mass, copy=True),
            body_id=np.array(self.body_id, copy=True),
            attributes=exported_attributes,
            fields=exported_fields,
            scalars=exported_scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.num_particles,
        }

