"""Coupled fluid-smoke and cloth simulation solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Cloth parameters
        cloth_cfg = self.config["cloth"]
        self.pinned_vertices = np.array(cloth_cfg["pinned_vertices"], dtype=np.int32)
        init_frame = scene.initial_state()
        self.cloth_pos = init_frame.position.copy()
        self.cloth_vel = init_frame.velocity.copy()
        self.cloth_mass = init_frame.mass.copy()
        self.uv = init_frame.attributes["uv"].copy()
        self.pinned_mask = np.isin(np.arange(len(self.cloth_pos)), self.pinned_vertices)

        # Build cloth springs/mesh topology
        nx, nz = map(int, cloth_cfg["grid_shape"])
        self.nx, self.nz = nx, nz
        self.num_nodes = len(self.cloth_pos)

        # Collect springs: structural, shear, bending
        springs = []
        rest_lengths = []
        for j in range(nz):
            for i in range(nx):
                idx = j * nx + i
                # structural
                if i + 1 < nx:
                    idx2 = j * nx + (i + 1)
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))
                if j + 1 < nz:
                    idx2 = (j + 1) * nx + i
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))
                # shear
                if i + 1 < nx and j + 1 < nz:
                    idx2 = (j + 1) * nx + (i + 1)
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))
                if i > 0 and j + 1 < nz:
                    idx2 = (j + 1) * nx + (i - 1)
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))
                # bending
                if i + 2 < nx:
                    idx2 = j * nx + (i + 2)
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))
                if j + 2 < nz:
                    idx2 = (j + 2) * nx + i
                    springs.append((idx, idx2))
                    rest_lengths.append(np.linalg.norm(self.cloth_pos[idx] - self.cloth_pos[idx2]))

        self.spring_indices = np.array(springs, dtype=np.int32)
        self.spring_lengths = np.array(rest_lengths, dtype=np.float32)

        # Domain & Fluid Grid
        domain_cfg = self.config["domain"]
        self.grid_shape = tuple(map(int, domain_cfg["grid_shape"])) # (126, 70, 70)
        self.bounds_min = np.array(domain_cfg["bounds_min"], dtype=np.float32)
        self.bounds_max = np.array(domain_cfg["bounds_max"], dtype=np.float32)
        self.dx = (self.bounds_max - self.bounds_min) / np.array(self.grid_shape, dtype=np.float32)

        # Cell center coordinates
        gx = self.bounds_min[0] + (np.arange(self.grid_shape[0], dtype=np.float32) + 0.5) * self.dx[0]
        gy = self.bounds_min[1] + (np.arange(self.grid_shape[1], dtype=np.float32) + 0.5) * self.dx[1]
        gz = self.bounds_min[2] + (np.arange(self.grid_shape[2], dtype=np.float32) + 0.5) * self.dx[2]
        self.grid_x, self.grid_y, self.grid_z = np.meshgrid(gx, gy, gz, indexing="ij")

        # Initial fields from initial state
        self.smoke_density = init_frame.fields["smoke_density"].copy()
        self.fluid_velocity = init_frame.fields["fluid_velocity"].copy()

        # Jet configuration
        self.jet = self.config["jet"]
        self.jet_center = np.array(self.jet["center"], dtype=np.float32)
        self.jet_axis = np.array(self.jet["axis"], dtype=np.float32)
        self.jet_radius = float(self.jet["radius"])
        self.jet_speed = float(self.jet["speed"])
        self.jet_density = float(self.jet["smoke_density"])
        self.jet_start = float(self.jet["start_time"])
        self.jet_ramp = float(self.jet["ramp_duration"])
        self.jet_active_until = float(self.jet["active_until"])

        # Gravity & Physics params
        self.gravity = np.array(self.config["initial_conditions"]["gravity"], dtype=np.float32)
        self.air_density = 1.2
        self.time = 0.0

        # Precompute jet nozzle mask for emission
        dist_sq = (self.grid_y - self.jet_center[1])**2 + (self.grid_z - self.jet_center[2])**2
        self.jet_mask = (dist_sq <= self.jet_radius**2) & (self.grid_x <= self.bounds_min[0] + 3 * self.dx[0])

        # State tracking for budget/momentum fields
        self.cum_ext_impulse = np.zeros((1, 3), dtype=np.float32)
        self.cum_inflow_momentum = np.zeros((1, 3), dtype=np.float32)
        self.cum_outflow_momentum = np.zeros((1, 3), dtype=np.float32)
        self.cum_mass_source = np.zeros(1, dtype=np.float32)
        self.last_interaction_force = np.zeros((1, 3), dtype=np.float32)
        self.last_external_force = np.zeros((1, 1, 3), dtype=np.float32)

    def _jet_factor(self, t: float) -> float:
        if t < self.jet_start or t > self.jet_active_until:
            return 0.0
        if t < self.jet_start + self.jet_ramp:
            return (t - self.jet_start) / self.jet_ramp
        return 1.0

    def _trilinear_interpolate_velocity(self, points: np.ndarray) -> np.ndarray:
        # points shape: [N, 3]
        # Return interpolated fluid velocity at points
        coords = (points - self.bounds_min) / self.dx - 0.5
        coords = np.clip(coords, 0.0, np.array(self.grid_shape, dtype=np.float32) - 1.001)

        i0 = np.floor(coords).astype(np.int32)
        i1 = np.minimum(i0 + 1, np.array(self.grid_shape) - 1)
        f = coords - i0

        c000 = self.fluid_velocity[i0[:, 0], i0[:, 1], i0[:, 2]]
        c100 = self.fluid_velocity[i1[:, 0], i0[:, 1], i0[:, 2]]
        c010 = self.fluid_velocity[i0[:, 0], i1[:, 1], i0[:, 2]]
        c110 = self.fluid_velocity[i1[:, 0], i1[:, 1], i0[:, 2]]
        c001 = self.fluid_velocity[i0[:, 0], i0[:, 1], i1[:, 2]]
        c101 = self.fluid_velocity[i1[:, 0], i0[:, 1], i1[:, 2]]
        c011 = self.fluid_velocity[i0[:, 0], i1[:, 1], i1[:, 2]]
        c111 = self.fluid_velocity[i1[:, 0], i1[:, 1], i1[:, 2]]

        c00 = c000 * (1 - f[:, 0, None]) + c100 * f[:, 0, None]
        c10 = c010 * (1 - f[:, 0, None]) + c110 * f[:, 0, None]
        c01 = c001 * (1 - f[:, 0, None]) + c101 * f[:, 0, None]
        c11 = c011 * (1 - f[:, 0, None]) + c111 * f[:, 0, None]

        c0 = c00 * (1 - f[:, 1, None]) + c10 * f[:, 1, None]
        c1 = c01 * (1 - f[:, 1, None]) + c11 * f[:, 1, None]

        return c0 * (1 - f[:, 2, None]) + c1 * f[:, 2, None]

    def _step(self, dt: float) -> None:
        t = self.time
        jet_fac = self._jet_factor(t)

        # 1. Jet inflow at boundary/emitter
        if jet_fac > 0.0:
            target_vel_x = self.jet_speed * jet_fac
            self.fluid_velocity[self.jet_mask, 0] = target_vel_x
            self.fluid_velocity[self.jet_mask, 1] = 0.0
            self.fluid_velocity[self.jet_mask, 2] = 0.0
            self.smoke_density[self.jet_mask] = self.jet_density * jet_fac

            # Inflow momentum tracking
            inflow_cells = np.count_nonzero(self.jet_mask)
            cell_vol = np.prod(self.dx)
            inflow_mass = inflow_cells * cell_vol * self.air_density * (target_vel_x * dt / self.dx[0])
            inflow_mom = np.array([inflow_mass * target_vel_x, 0.0, 0.0], dtype=np.float32)
            self.cum_inflow_momentum[0] += inflow_mom
            self.cum_mass_source[0] += inflow_mass

        # 2. Compute Cloth Normals and Aero Forces
        # Mesh layout: nx x nz grid
        pos_grid = self.cloth_pos.reshape((self.nz, self.nx, 3))
        # Tangents
        # dpos/dy (horizontal, along nx)
        ty = np.zeros_like(pos_grid)
        ty[:, 1:-1] = (pos_grid[:, 2:] - pos_grid[:, :-2]) * 0.5
        ty[:, 0] = pos_grid[:, 1] - pos_grid[:, 0]
        ty[:, -1] = pos_grid[:, -1] - pos_grid[:, -2]

        # dpos/dz (vertical, along nz)
        tz = np.zeros_like(pos_grid)
        tz[1:-1, :] = (pos_grid[2:, :] - pos_grid[:-2, :]) * 0.5
        tz[0, :] = pos_grid[1, :] - pos_grid[0, :]
        tz[-1, :] = pos_grid[-1, :] - pos_grid[-2, :]

        # Cross product to get normals
        normals_grid = np.cross(tz, ty)
        normals = normals_grid.reshape((self.num_nodes, 3))
        norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        unit_normals = normals / norms

        # Fluid velocity at cloth nodes
        fluid_u_at_cloth = self._trilinear_interpolate_velocity(self.cloth_pos)
        rel_vel = fluid_u_at_cloth - self.cloth_vel # [N, 3]

        # Normal component of relative velocity
        v_rel_normal = np.sum(rel_vel * unit_normals, axis=-1, keepdims=True) # [N, 1]

        # Aerodynamic drag/pressure force on cloth: F = 0.5 * rho * Cd * A * |v_n| * v_n * normal
        node_area = (float(self.config["cloth"]["width"]) * float(self.config["cloth"]["height"])) / self.num_nodes
        aero_force = 1.2 * self.air_density * node_area * np.abs(v_rel_normal) * v_rel_normal * unit_normals

        # Total force on cloth
        gravity_force = self.cloth_mass[:, None] * self.gravity[None, :]
        cloth_force = gravity_force + aero_force

        # Cloth physics step (Verlet / Euler with spring constraints)
        # Apply external forces
        self.cloth_vel += (cloth_force / self.cloth_mass[:, None]) * dt
        # Air damping
        self.cloth_vel *= (1.0 - 0.05 * dt)
        self.cloth_vel[self.pinned_mask] = 0.0

        self.cloth_pos += self.cloth_vel * dt
        self.cloth_pos[self.pinned_mask] = self.cloth_pos[self.pinned_mask] # pinned remain fixed

        # Spring relaxation (Position-based dynamics)
        p1_idx = self.spring_indices[:, 0]
        p2_idx = self.spring_indices[:, 1]
        for _ in range(15):
            p1 = self.cloth_pos[p1_idx]
            p2 = self.cloth_pos[p2_idx]
            delta = p2 - p1
            dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8
            diff = (dist - self.spring_lengths[:, None]) / dist
            correction = 0.5 * delta * diff

            w1 = (~self.pinned_mask[p1_idx]).astype(np.float32)[:, None]
            w2 = (~self.pinned_mask[p2_idx]).astype(np.float32)[:, None]
            w_total = w1 + w2 + 1e-6

            np.add.at(self.cloth_pos, p1_idx, correction * (w1 / w_total))
            np.add.at(self.cloth_pos, p2_idx, -correction * (w2 / w_total))

            # Re-enforce pinned
            # self.cloth_pos[self.pinned_mask] = fixed

        # Update cloth velocity from position change
        # Floor collision
        below_floor = self.cloth_pos[:, 2] < 0.02
        self.cloth_pos[below_floor, 2] = 0.02
        if np.any(below_floor):
            self.cloth_vel[below_floor, 2] = np.maximum(0.0, self.cloth_vel[below_floor, 2])

        # 3. Fluid update: 2-way coupling & Smoke advection
        # Cloth solid velocity / obstacle effect on fluid:
        # Transfer cloth velocity to fluid grid cells near cloth and enforce impermeable boundary
        cloth_grid_coords = (self.cloth_pos - self.bounds_min) / self.dx
        ci = np.clip(np.round(cloth_grid_coords).astype(np.int32), 0, np.array(self.grid_shape) - 1)

        # Deflect fluid around cloth: cloth acts as moving impermeable barrier
        # In grid cells containing cloth, set normal fluid velocity to cloth velocity
        for idx in range(self.num_nodes):
            gx_idx, gy_idx, gz_idx = ci[idx]
            n_vec = unit_normals[idx]
            v_cloth = self.cloth_vel[idx]

            # Fluid velocity in cell
            v_fl = self.fluid_velocity[gx_idx, gy_idx, gz_idx]
            # Decompose into normal and tangential relative to cloth
            v_rel = v_fl - v_cloth
            v_rel_n = np.dot(v_rel, n_vec)
            if v_rel_n > 0: # flow hitting cloth surface from front
                # Eliminate normal relative velocity (impenetrability) and push flow sideways
                tangent_push = v_rel - v_rel_n * n_vec
                self.fluid_velocity[gx_idx, gy_idx, gz_idx] = v_cloth + tangent_push * 1.2

            # Smoke cannot penetrate cloth: slight backward deflection
            if self.smoke_density[gx_idx, gy_idx, gz_idx] > 0.05:
                # Push some smoke sideways along tangent
                pass

        # Advect smoke density: Semi-Lagrangian
        # x_back = x - u * dt
        back_x = np.clip(self.grid_x - self.fluid_velocity[..., 0] * dt, self.bounds_min[0] + 0.5 * self.dx[0], self.bounds_max[0] - 0.5 * self.dx[0])
        back_y = np.clip(self.grid_y - self.fluid_velocity[..., 1] * dt, self.bounds_min[1] + 0.5 * self.dx[1], self.bounds_max[1] - 0.5 * self.dx[1])
        back_z = np.clip(self.grid_z - self.fluid_velocity[..., 2] * dt, self.bounds_min[2] + 0.5 * self.dx[2], self.bounds_max[2] - 0.5 * self.dx[2])

        # Trilinear sampling for smoke
        ci0_x = np.clip(np.floor((back_x - self.bounds_min[0]) / self.dx[0] - 0.5).astype(np.int32), 0, self.grid_shape[0] - 2)
        ci0_y = np.clip(np.floor((back_y - self.bounds_min[1]) / self.dx[1] - 0.5).astype(np.int32), 0, self.grid_shape[1] - 2)
        ci0_z = np.clip(np.floor((back_z - self.bounds_min[2]) / self.dx[2] - 0.5).astype(np.int32), 0, self.grid_shape[2] - 2)

        fx = (back_x - (self.bounds_min[0] + (ci0_x + 0.5) * self.dx[0])) / self.dx[0]
        fy = (back_y - (self.bounds_min[1] + (ci0_y + 0.5) * self.dx[1])) / self.dx[1]
        fz = (back_z - (self.bounds_min[2] + (ci0_z + 0.5) * self.dx[2])) / self.dx[2]
        fx = np.clip(fx, 0.0, 1.0)
        fy = np.clip(fy, 0.0, 1.0)
        fz = np.clip(fz, 0.0, 1.0)

        s000 = self.smoke_density[ci0_x, ci0_y, ci0_z]
        s100 = self.smoke_density[ci0_x + 1, ci0_y, ci0_z]
        s010 = self.smoke_density[ci0_x, ci0_y + 1, ci0_z]
        s110 = self.smoke_density[ci0_x + 1, ci0_y + 1, ci0_z]
        s001 = self.smoke_density[ci0_x, ci0_y, ci0_z + 1]
        s101 = self.smoke_density[ci0_x + 1, ci0_y, ci0_z + 1]
        s011 = self.smoke_density[ci0_x, ci0_y + 1, ci0_z + 1]
        s111 = self.smoke_density[ci0_x + 1, ci0_y + 1, ci0_z + 1]

        s00 = s000 * (1 - fx) + s100 * fx
        s10 = s010 * (1 - fx) + s110 * fx
        s01 = s001 * (1 - fx) + s101 * fx
        s11 = s011 * (1 - fx) + s111 * fx

        s0 = s00 * (1 - fy) + s10 * fy
        s1 = s01 * (1 - fy) + s11 * fy

        self.smoke_density = (s0 * (1 - fz) + s1 * fz).astype(np.float32)

        # Advect velocity field similarly (Semi-Lagrangian advection)
        for d in range(3):
            u000 = self.fluid_velocity[ci0_x, ci0_y, ci0_z, d]
            u100 = self.fluid_velocity[ci0_x + 1, ci0_y, ci0_z, d]
            u010 = self.fluid_velocity[ci0_x, ci0_y + 1, ci0_z, d]
            u110 = self.fluid_velocity[ci0_x + 1, ci0_y + 1, ci0_z, d]
            u001 = self.fluid_velocity[ci0_x, ci0_y, ci0_z + 1, d]
            u101 = self.fluid_velocity[ci0_x + 1, ci0_y, ci0_z + 1, d]
            u011 = self.fluid_velocity[ci0_x, ci0_y + 1, ci0_z + 1, d]
            u111 = self.fluid_velocity[ci0_x + 1, ci0_y + 1, ci0_z + 1, d]

            u00 = u000 * (1 - fx) + u100 * fx
            u10 = u010 * (1 - fx) + u110 * fx
            u01 = u001 * (1 - fx) + u101 * fx
            u11 = u011 * (1 - fx) + u111 * fx

            u0 = u00 * (1 - fy) + u10 * fy
            u1 = u01 * (1 - fy) + u11 * fy

            self.fluid_velocity[..., d] = (u0 * (1 - fz) + u1 * fz).astype(np.float32)

        # Apply dissipation
        self.smoke_density *= 0.999
        self.fluid_velocity *= 0.998

        # Track external forces and impulse
        total_aero = np.sum(aero_force, axis=0)
        total_grav = np.sum(gravity_force, axis=0)
        self.last_interaction_force[0] = total_aero
        self.last_external_force[0, 0] = total_grav
        self.cum_ext_impulse[0] += (total_grav + total_aero) * dt

        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        sub_dt = 0.005
        while self.time < time - 1e-6:
            dt = min(sub_dt, time - self.time)
            self._step(dt)

        # Boundary samples (floor & walls)
        boundary_normal = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        boundary_vel = np.zeros((1, 3), dtype=np.float32)
        boundary_target_vel = np.zeros((1, 3), dtype=np.float32)

        # Budget domain weights (1 domain, N nodes)
        budget_weights = np.ones((1, self.num_nodes), dtype=np.float32)

        total_mass = float(np.sum(self.cloth_mass))
        ref_vel = float(self.jet_speed)
        ref_mom = total_mass * ref_vel

        frame = StateFrame(
            time=float(self.time),
            position=self.cloth_pos.copy().astype(np.float32),
            velocity=self.cloth_vel.copy().astype(np.float32),
            mass=self.cloth_mass.copy().astype(np.float32),
            body_id=np.ones(self.num_nodes, dtype=np.int32),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pinned_mask.astype(np.float32),
                "mass_quantity": self.cloth_mass.copy().astype(np.float32),
            },
            fields={
                "smoke_density": self.smoke_density.copy().astype(np.float32),
                "fluid_velocity": self.fluid_velocity.copy().astype(np.float32),
                "boundary_normal": boundary_normal,
                "boundary_velocity": boundary_vel,
                "boundary_target_velocity": boundary_target_vel,
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy().astype(np.float32),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy().astype(np.float32),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy().astype(np.float32),
                "budget_domain_weights": budget_weights,
                "budget_external_force": self.last_external_force.copy().astype(np.float32),
                "cumulative_mass_source": self.cum_mass_source.copy().astype(np.float32),
                "interaction_force": self.last_interaction_force.copy().astype(np.float32),
            },
            scalars={
                "jet_start_time": float(self.jet_start),
                "jet_speed": float(self.jet_speed),
                "reference_velocity": ref_vel,
                "reference_momentum": ref_mom,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "max_smoke_density": float(np.max(self.smoke_density)),
            "cloth_max_disp": float(np.max(np.linalg.norm(self.cloth_pos, axis=1))),
        }

