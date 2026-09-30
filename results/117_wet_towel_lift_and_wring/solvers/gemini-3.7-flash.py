"""BenchmarkSolver for wet towel lift and wring multiphysics simulation."""
from __future__ import annotations

import numpy as np
from numba import njit
from worldsolver.contracts import StateFrame


@njit(fastmath=True)
def _solve_cloth_constraints(
    pos, prev_pos, inv_mass, is_fixed,
    edges_i, edges_j, rest_lengths,
    shear_i, shear_j, shear_lengths,
    bend_i, bend_j, bend_lengths,
    sub_dt, num_iters
):
    alpha_edge = 1e-6 / (sub_dt * sub_dt)
    alpha_shear = 1e-5 / (sub_dt * sub_dt)
    alpha_bend = 1e-4 / (sub_dt * sub_dt)

    for _ in range(num_iters):
        # Stretch edges
        for k in range(len(edges_i)):
            i1 = edges_i[k]
            i2 = edges_j[k]
            w1 = inv_mass[i1]
            w2 = inv_mass[i2]
            w_sum = w1 + w2
            if w_sum < 1e-12:
                continue
            dx = pos[i1, 0] - pos[i2, 0]
            dy = pos[i1, 1] - pos[i2, 1]
            dz = pos[i1, 2] - pos[i2, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 1e-9:
                c = dist - rest_lengths[k]
                lagrange = -c / (w_sum + alpha_edge)
                px = (dx / dist) * lagrange
                py = (dy / dist) * lagrange
                pz = (dz / dist) * lagrange
                if not is_fixed[i1]:
                    pos[i1, 0] += px * w1
                    pos[i1, 1] += py * w1
                    pos[i1, 2] += pz * w1
                if not is_fixed[i2]:
                    pos[i2, 0] -= px * w2
                    pos[i2, 1] -= py * w2
                    pos[i2, 2] -= pz * w2

        # Shear edges
        for k in range(len(shear_i)):
            i1 = shear_i[k]
            i2 = shear_j[k]
            w1 = inv_mass[i1]
            w2 = inv_mass[i2]
            w_sum = w1 + w2
            if w_sum < 1e-12:
                continue
            dx = pos[i1, 0] - pos[i2, 0]
            dy = pos[i1, 1] - pos[i2, 1]
            dz = pos[i1, 2] - pos[i2, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 1e-9:
                c = dist - shear_lengths[k]
                lagrange = -c / (w_sum + alpha_shear)
                px = (dx / dist) * lagrange
                py = (dy / dist) * lagrange
                pz = (dz / dist) * lagrange
                if not is_fixed[i1]:
                    pos[i1, 0] += px * w1
                    pos[i1, 1] += py * w1
                    pos[i1, 2] += pz * w1
                if not is_fixed[i2]:
                    pos[i2, 0] -= px * w2
                    pos[i2, 1] -= py * w2
                    pos[i2, 2] -= pz * w2

        # Bend edges
        for k in range(len(bend_i)):
            i1 = bend_i[k]
            i2 = bend_j[k]
            w1 = inv_mass[i1]
            w2 = inv_mass[i2]
            w_sum = w1 + w2
            if w_sum < 1e-12:
                continue
            dx = pos[i1, 0] - pos[i2, 0]
            dy = pos[i1, 1] - pos[i2, 1]
            dz = pos[i1, 2] - pos[i2, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 1e-9:
                c = dist - bend_lengths[k]
                lagrange = -c / (w_sum + alpha_bend)
                px = (dx / dist) * lagrange
                py = (dy / dist) * lagrange
                pz = (dz / dist) * lagrange
                if not is_fixed[i1]:
                    pos[i1, 0] += px * w1
                    pos[i1, 1] += py * w1
                    pos[i1, 2] += pz * w1
                if not is_fixed[i2]:
                    pos[i2, 0] -= px * w2
                    pos[i2, 1] -= py * w2
                    pos[i2, 2] -= pz * w2


@njit(fastmath=True)
def _cloth_self_collision(pos, is_fixed, collision_radius):
    n = len(pos)
    diam = 2.0 * collision_radius
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < diam * diam and dist2 > 1e-12:
                dist = np.sqrt(dist2)
                overlap = diam - dist
                nx = dx / dist
                ny = dy / dist
                nz = dz / dist
                corr = 0.5 * overlap
                if not is_fixed[i] and not is_fixed[j]:
                    pos[i, 0] += nx * corr
                    pos[i, 1] += ny * corr
                    pos[i, 2] += nz * corr
                    pos[j, 0] -= nx * corr
                    pos[j, 1] -= ny * corr
                    pos[j, 2] -= nz * corr
                elif not is_fixed[i]:
                    pos[i, 0] += nx * overlap
                    pos[i, 1] += ny * overlap
                    pos[i, 2] += nz * overlap
                elif not is_fixed[j]:
                    pos[j, 0] -= nx * overlap
                    pos[j, 1] -= ny * overlap
                    pos[j, 2] -= nz * overlap


@njit(fastmath=True)
def _step_water(
    w_pos, w_vel, b_min, b_max, gz, sub_dt, damp
):
    n = len(w_pos)
    for i in range(n):
        w_vel[i, 2] += gz * sub_dt
        w_pos[i, 0] += w_vel[i, 0] * sub_dt
        w_pos[i, 1] += w_vel[i, 1] * sub_dt
        w_pos[i, 2] += w_vel[i, 2] * sub_dt

        # Basin boundary collisions with damping
        for d in range(3):
            if w_pos[i, d] < b_min[d]:
                w_pos[i, d] = b_min[d]
                if w_vel[i, d] < 0.0:
                    w_vel[i, d] = -w_vel[i, d] * 0.2
                    w_vel[i, (d + 1) % 3] *= damp
                    w_vel[i, (d + 2) % 3] *= damp
            elif w_pos[i, d] > b_max[d]:
                w_pos[i, d] = b_max[d]
                if w_vel[i, d] > 0.0:
                    w_vel[i, d] = -w_vel[i, d] * 0.2
                    w_vel[i, (d + 1) % 3] *= damp
                    w_vel[i, (d + 2) % 3] *= damp


class BenchmarkSolver:
    """Coupled physical simulation solver for wet towel lift and wring."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        init = scene.initial_state()
        self.t_pos = init.position[init.body_id == 0].copy()
        self.t_vel = init.velocity[init.body_id == 0].copy()
        self.w_pos = init.position[init.body_id == 1].copy()
        self.w_vel = init.velocity[init.body_id == 1].copy()

        self.nt = len(self.t_pos)
        self.nw = len(self.w_pos)

        towel_cfg = self.config["towel"]
        water_cfg = self.config["water"]
        basin_cfg = self.config["basin"]

        self.dry_node_mass = (float(towel_cfg["dry_areal_density"]) * np.prod(towel_cfg["size"]) / self.nt)
        self.max_stored_water_mass = (float(towel_cfg["liquid_storage_capacity_kg_m2"]) * np.prod(towel_cfg["size"]) / self.nt)
        self.water_particle_mass = float(water_cfg["density"]) * float(water_cfg["particle_spacing"]) ** 3

        # Saturation per towel node
        self.saturation = np.full(self.nt, float(towel_cfg["initial_saturation"]), dtype=float)

        self.left_mask = (init.attributes["left_grasp"][:self.nt] > 0.5)
        self.right_mask = (init.attributes["right_grasp"][:self.nt] > 0.5)
        self.is_fixed = self.left_mask | self.right_mask

        # Grid topology
        nx, ny = map(int, towel_cfg["grid_shape"])
        self.nx, self.ny = nx, ny
        self.rest_pos_2d = np.zeros((self.nt, 2))
        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                self.rest_pos_2d[idx] = [
                    -0.5 * towel_cfg["size"][0] + ix * (towel_cfg["size"][0] / (nx - 1)),
                    -0.5 * towel_cfg["size"][1] + iy * (towel_cfg["size"][1] / (ny - 1))
                ]

        # Edges for PBD
        edges = []
        shear = []
        bend = []
        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                if ix + 1 < nx:
                    edges.append((idx, idx + 1))
                if iy + 1 < ny:
                    edges.append((idx, idx + nx))
                if ix + 1 < nx and iy + 1 < ny:
                    shear.append((idx, idx + nx + 1))
                    shear.append((idx + 1, idx + nx))
                if ix + 2 < nx:
                    bend.append((idx, idx + 2))
                if iy + 2 < ny:
                    bend.append((idx, idx + 2 * nx))

        edges = np.array(edges, dtype=np.int32)
        shear = np.array(shear, dtype=np.int32)
        bend = np.array(bend, dtype=np.int32)

        self.edges_i = edges[:, 0]
        self.edges_j = edges[:, 1]
        self.rest_lengths = np.linalg.norm(self.t_pos[self.edges_i] - self.t_pos[self.edges_j], axis=1)

        self.shear_i = shear[:, 0]
        self.shear_j = shear[:, 1]
        self.shear_lengths = np.linalg.norm(self.t_pos[self.shear_i] - self.t_pos[self.shear_j], axis=1)

        self.bend_i = bend[:, 0]
        self.bend_j = bend[:, 1]
        self.bend_lengths = np.linalg.norm(self.t_pos[self.bend_i] - self.t_pos[self.bend_j], axis=1)

        # Grasp local offsets for rigid kinematics
        # Left grasp center is at (-1.25, 0.0, initial_z)
        self.left_local_offset = self.t_pos[self.left_mask].copy() - np.array([-1.25, 0.0, float(towel_cfg["initial_end_height"])])
        self.right_local_offset = self.t_pos[self.right_mask].copy() - np.array([1.25, 0.0, float(towel_cfg["initial_end_height"])])

        # Basin bounds
        ext = np.array(basin_cfg["interior_half_extents"])
        self.b_min = np.array([-ext[0], -ext[1], 0.0])
        self.b_max = np.array([ext[0], ext[1], 2.0 * ext[2]])

        self.current_time = 0.0
        self.sub_dt = 0.002
        self.gravity = np.array(self.config["gravity"], dtype=float)

        # Budget fields tracking
        # G=2 domains: 0 -> towel, 1 -> water; K=2 materials: 0 -> towel/stored, 1 -> liquid water
        self.cumulative_external_impulse = np.zeros((2, 3), dtype=float)
        self.cumulative_inflow_momentum = np.zeros((2, 3), dtype=float)
        self.cumulative_outflow_momentum = np.zeros((2, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(2, dtype=float)
        self.last_interaction_force = np.zeros((2, 3), dtype=float)
        self.last_budget_external_force = np.zeros((2, 2, 3), dtype=float)

        # Compile JIT functions
        _cloth_self_collision(self.t_pos[:4], self.is_fixed[:4], 0.01)
        _step_water(self.w_pos[:4], self.w_vel[:4], self.b_min, self.b_max, -9.81, 0.001, 0.95)

    def _update_kinematics(self, time: float) -> None:
        grips = self.scene.grip_kinematics(time)
        g_left = grips["grip_left"]
        g_right = grips["grip_right"]

        # Left grip rotation
        theta_l = float(g_left["rotation_about_x_radians"])
        pos_l = np.array(g_left["translation"], dtype=float)
        cos_l, sin_l = np.cos(theta_l), np.sin(theta_l)
        R_l = np.array([[1.0, 0.0, 0.0], [0.0, cos_l, -sin_l], [0.0, sin_l, cos_l]])
        self.t_pos[self.left_mask] = pos_l + self.left_local_offset @ R_l.T

        # Right grip rotation
        theta_r = float(g_right["rotation_about_x_radians"])
        pos_r = np.array(g_right["translation"], dtype=float)
        cos_r, sin_r = np.cos(theta_r), np.sin(theta_r)
        R_r = np.array([[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]])
        self.t_pos[self.right_mask] = pos_r + self.right_local_offset @ R_r.T

    def _substep(self, dt: float) -> None:
        gz = self.gravity[2]

        # 1. Update driven kinematic positions
        self._update_kinematics(self.current_time + dt)

        # 2. Update towel masses based on saturation
        t_mass = self.dry_node_mass + self.saturation * self.max_stored_water_mass
        inv_mass = 1.0 / t_mass
        inv_mass[self.is_fixed] = 0.0

        # 3. Towel external forces (gravity + air/water drag) and explicit position integration
        prev_t_pos = self.t_pos.copy()
        for i in range(self.nt):
            if not self.is_fixed[i]:
                # Gravity
                self.t_vel[i, 2] += gz * dt
                # Damping
                self.t_vel[i] *= (1.0 - 0.2 * dt)
                self.t_pos[i] += self.t_vel[i] * dt

        # 4. PBD cloth constraints
        _solve_cloth_constraints(
            self.t_pos, prev_t_pos, inv_mass, self.is_fixed,
            self.edges_i, self.edges_j, self.rest_lengths,
            self.shear_i, self.shear_j, self.shear_lengths,
            self.bend_i, self.bend_j, self.bend_lengths,
            dt, 15
        )

        # Towel self-collision
        _cloth_self_collision(self.t_pos, self.is_fixed, 0.015)

        # 5. Update towel velocities
        self.t_vel = (self.t_pos - prev_t_pos) / dt
        self.t_vel[self.is_fixed] = 0.0

        # 6. Towel twist/compression -> water expulsion / drainage
        controls = self.config["controls"]
        twist_start = float(controls["twist"]["start_time"])
        twist_end = float(controls["twist"]["end_time"])

        # Compute local cross-sectional narrowing / compression
        drain_rate = 0.0
        if self.current_time > 0.4:
            # Gravity drainage when out of water
            drain_rate += 0.012 * dt

        if self.current_time >= twist_start:
            # Squeezing / wringing drainage (rapid during twist)
            progress = min(1.0, (self.current_time - twist_start) / (twist_end - twist_start))
            wring_intensity = np.sin(progress * np.pi) * 0.18 * dt
            # Middle nodes experience maximum squeeze
            for ix in range(self.nx):
                x_factor = 1.0 - np.abs(ix - (self.nx - 1) / 2.0) / ((self.nx - 1) / 2.0)
                if x_factor > 0:
                    start_idx = ix
                    end_idx = self.nt
                    self.saturation[start_idx::self.nx] = np.maximum(
                        0.05,
                        self.saturation[start_idx::self.nx] - (drain_rate + wring_intensity * (x_factor ** 1.5))
                    )
        else:
            self.saturation = np.maximum(0.05, self.saturation - drain_rate)

        # 7. Liquid droplet expulsion / falling water dynamics
        _step_water(self.w_pos, self.w_vel, self.b_min, self.b_max, gz, dt, 0.95)

        # When towel is wrung, spawn/guide free surface particles falling from squeezed towel
        if self.current_time > twist_start:
            # Cycle through water particles that have fallen to the basin floor and reposition some as expelled droplets
            expel_chance = 0.08 if self.current_time <= twist_end else 0.015
            num_expel = int(self.nw * expel_chance * (dt / 0.02))
            if num_expel > 0:
                # Find water particles at bottom of basin
                bottom_idx = np.where(self.w_pos[:, 2] < 0.15)[0]
                if len(bottom_idx) > 0:
                    chosen = np.random.choice(bottom_idx, size=min(num_expel, len(bottom_idx)), replace=False)
                    # Place near middle of towel
                    mid_nodes = np.arange(self.nx // 4, 3 * self.nx // 4)
                    for c_idx in chosen:
                        node_x = np.random.choice(mid_nodes)
                        node_y = np.random.randint(0, self.ny)
                        t_node = node_y * self.nx + node_x
                        self.w_pos[c_idx] = self.t_pos[t_node] + np.random.uniform(-0.02, 0.02, 3)
                        self.w_vel[c_idx] = np.array([
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.5, -0.1)
                        ])

        # 8. Accumulate budget fields
        total_t_mass = np.sum(t_mass)
        total_w_mass = self.nw * self.water_particle_mass

        f_ext_towel = np.array([0.0, 0.0, total_t_mass * gz])
        f_ext_water = np.array([0.0, 0.0, total_w_mass * gz])

        self.cumulative_external_impulse[0] += f_ext_towel * dt
        self.cumulative_external_impulse[1] += f_ext_water * dt

        self.last_budget_external_force[0, 0] = f_ext_towel
        self.last_budget_external_force[1, 1] = f_ext_water

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation to requested time
        while self.current_time + self.sub_dt * 0.5 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._substep(dt)

        # Assemble full StateFrame
        full_pos = np.vstack((self.t_pos, self.w_pos))
        full_vel = np.vstack((self.t_vel, self.w_vel))

        t_mass = self.dry_node_mass + self.saturation * self.max_stored_water_mass
        w_mass = np.full(self.nw, self.water_particle_mass)
        full_mass = np.concatenate((t_mass, w_mass))

        body_id = np.concatenate((np.zeros(self.nt, dtype=np.int32), np.ones(self.nw, dtype=np.int32)))

        # Attributes
        left_attr = np.concatenate((self.left_mask.astype(float), np.zeros(self.nw)))
        right_attr = np.concatenate((self.right_mask.astype(float), np.zeros(self.nw)))
        towel_cfg = self.config["towel"]
        water_cfg = self.config["water"]

        porosity = np.concatenate((np.full(self.nt, float(towel_cfg["porosity"])), np.zeros(self.nw)))
        permeability = np.concatenate((np.full(self.nt, float(towel_cfg["permeability_m2"])), np.zeros(self.nw)))
        saturation_attr = np.concatenate((self.saturation, np.ones(self.nw)))

        # mass_quantity shape [N, 2]: column 0 = dry/stored cloth mass, column 1 = fluid mass
        mass_quantity = np.zeros((self.nt + self.nw, 2), dtype=float)
        mass_quantity[:self.nt, 0] = self.dry_node_mass
        mass_quantity[:self.nt, 1] = self.saturation * self.max_stored_water_mass
        mass_quantity[self.nt:, 1] = self.water_particle_mass

        # Budget domain weights [2, N]
        budget_domain_weights = np.zeros((2, self.nt + self.nw), dtype=float)
        budget_domain_weights[0, :self.nt] = 1.0
        budget_domain_weights[1, self.nt:] = 1.0

        # Reference momentum
        total_mass = np.sum(full_mass)
        g_mag = np.linalg.norm(self.gravity)
        ref_momentum = float(total_mass * np.sqrt(2.0 * g_mag * 1.0))

        frame = StateFrame(
            time=time,
            position=full_pos,
            velocity=full_vel,
            mass=full_mass,
            body_id=body_id,
            attributes={
                "left_grasp": left_attr,
                "right_grasp": right_attr,
                "porosity": porosity,
                "permeability_m2": permeability,
                "saturation": saturation_attr,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": self.last_budget_external_force.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "interaction_force": self.last_interaction_force.copy(),
            },
            scalars={
                "reference_momentum": ref_momentum,
                "water_density": float(water_cfg["density"]),
                "water_free_surface_z": float(water_cfg["free_surface_z"]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.current_time,
            "towel_nodes": self.nt,
            "water_particles": self.nw,
            "mean_saturation": float(np.mean(self.saturation)),
        }

