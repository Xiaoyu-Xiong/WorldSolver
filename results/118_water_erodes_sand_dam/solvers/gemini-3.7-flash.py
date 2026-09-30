"""Physically-based solver for coupled water infiltration and sand dam erosion."""
from __future__ import annotations

import math
import numpy as np
import numba
from numba import njit, prange
from worldsolver.contracts import StateFrame


@njit(fastmath=True)
def _solve_substep(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    body_id: np.ndarray,
    saturation: np.ndarray,
    dt: float,
    gravity: np.ndarray,
    domain_bounds: np.ndarray,
    wall_x_min: float,
    wall_x_max: float,
    wall_y_min: float,
    wall_y_max: float,
    floor_z: float,
):
    """
    Step forward particles with coupled SPH / MPM-like particle dynamics.
    - Water pushes sand via pressure and drag.
    - Sand resists based on cohesion, which drops as sand is saturated with water.
    - Sand blocks water flow when intact; allows infiltration and flowing when breached.
    - Boundary collisions with flume floor, side walls, and reservoir/retaining walls.
    """
    n = len(pos)
    h = 0.16  # interaction radius
    h2 = h * h
    c_norm = 315.0 / (64.0 * math.pi * (h ** 9))
    spiky_grad_norm = -45.0 / (math.pi * (h ** 6))

    # Spatial hash grid for neighbor search
    cell_size = h
    grid_dim_x = int(math.ceil((domain_bounds[1, 0] - domain_bounds[0, 0]) / cell_size)) + 1
    grid_dim_y = int(math.ceil((domain_bounds[1, 1] - domain_bounds[0, 1]) / cell_size)) + 1
    grid_dim_z = int(math.ceil((domain_bounds[1, 2] - domain_bounds[0, 2]) / cell_size)) + 1
    total_cells = grid_dim_x * grid_dim_y * grid_dim_z

    cell_heads = np.full(total_cells, -1, dtype=np.int32)
    next_node = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        gx = int((pos[i, 0] - domain_bounds[0, 0]) / cell_size)
        gy = int((pos[i, 1] - domain_bounds[0, 1]) / cell_size)
        gz = int((pos[i, 2] - domain_bounds[0, 2]) / cell_size)
        if gx < 0:
            gx = 0
        elif gx >= grid_dim_x:
            gx = grid_dim_x - 1
        if gy < 0:
            gy = 0
        elif gy >= grid_dim_y:
            gy = grid_dim_y - 1
        if gz < 0:
            gz = 0
        elif gz >= grid_dim_z:
            gz = grid_dim_z - 1
        cid = gx + grid_dim_x * (gy + grid_dim_y * gz)
        next_node[i] = cell_heads[cid]
        cell_heads[cid] = i

    # Density computation
    density = np.zeros(n, dtype=np.float64)
    water_density_around_sand = np.zeros(n, dtype=np.float64)

    for i in range(n):
        gx = int((pos[i, 0] - domain_bounds[0, 0]) / cell_size)
        gy = int((pos[i, 1] - domain_bounds[0, 1]) / cell_size)
        gz = int((pos[i, 2] - domain_bounds[0, 2]) / cell_size)
        rho = 0.0
        w_near = 0.0
        for dx in range(-1, 2):
            cx = gx + dx
            if cx < 0 or cx >= grid_dim_x:
                continue
            for dy in range(-1, 2):
                cy = gy + dy
                if cy < 0 or cy >= grid_dim_y:
                    continue
                for dz in range(-1, 2):
                    cz = gz + dz
                    if cz < 0 or cz >= grid_dim_z:
                        continue
                    cid = cx + grid_dim_x * (cy + grid_dim_y * cz)
                    j = cell_heads[cid]
                    while j != -1:
                        rx = pos[i, 0] - pos[j, 0]
                        ry = pos[i, 1] - pos[j, 1]
                        rz = pos[i, 2] - pos[j, 2]
                        r2 = rx * rx + ry * ry + rz * rz
                        if r2 < h2:
                            term = h2 - r2
                            w = c_norm * term * term * term
                            rho += mass[j] * w
                            if body_id[j] == 2:  # water
                                w_near += w
                        j = next_node[j]
        density[i] = max(rho, 100.0)
        water_density_around_sand[i] = w_near

    # Update saturation for sand particles
    for i in range(n):
        if body_id[i] == 1:  # sand
            # Infiltration rate proportional to local water contact
            if water_density_around_sand[i] > 10.0:
                saturation[i] = min(1.0, saturation[i] + 0.85 * dt)
        else:
            saturation[i] = 1.0

    # Pressure and forces
    rest_rho_water = 1000.0
    rest_rho_sand = 1650.0
    k_water = 25000.0  # Bulk modulus / pressure stiffness
    k_sand = 35000.0

    pressure = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if body_id[i] == 2:  # water
            pressure[i] = max(0.0, k_water * ((density[i] / rest_rho_water) ** 7 - 1.0) / 7.0)
            if pressure[i] > 200000.0:
                pressure[i] = 200000.0
        else:  # sand
            pressure[i] = max(0.0, k_sand * ((density[i] / rest_rho_sand) ** 5 - 1.0) / 5.0)
            if pressure[i] > 300000.0:
                pressure[i] = 300000.0

    forces = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        # Gravity
        forces[i, 0] += mass[i] * gravity[0]
        forces[i, 1] += mass[i] * gravity[1]
        forces[i, 2] += mass[i] * gravity[2]

        gx = int((pos[i, 0] - domain_bounds[0, 0]) / cell_size)
        gy = int((pos[i, 1] - domain_bounds[0, 1]) / cell_size)
        gz = int((pos[i, 2] - domain_bounds[0, 2]) / cell_size)

        for dx in range(-1, 2):
            cx = gx + dx
            if cx < 0 or cx >= grid_dim_x:
                continue
            for dy in range(-1, 2):
                cy = gy + dy
                if cy < 0 or cy >= grid_dim_y:
                    continue
                for dz in range(-1, 2):
                    cz = gz + dz
                    if cz < 0 or cz >= grid_dim_z:
                        continue
                    cid = cx + grid_dim_x * (cy + grid_dim_y * cz)
                    j = cell_heads[cid]
                    while j != -1:
                        if i != j:
                            rx = pos[i, 0] - pos[j, 0]
                            ry = pos[i, 1] - pos[j, 1]
                            rz = pos[i, 2] - pos[j, 2]
                            r2 = rx * rx + ry * ry + rz * rz
                            if 0.0 < r2 < h2:
                                r = math.sqrt(r2)
                                term = h - r
                                grad_w = spiky_grad_norm * term * term
                                nx = rx / r
                                ny = ry / r
                                nz = rz / r

                                # Pressure force
                                p_term = (pressure[i] / (density[i] * density[i]) + pressure[j] / (density[j] * density[j]))
                                f_p = -mass[i] * mass[j] * p_term * grad_w

                                forces[i, 0] += f_p * nx
                                forces[i, 1] += f_p * ny
                                forces[i, 2] += f_p * nz

                                # Viscosity / Drag force
                                dvx = vel[j, 0] - vel[i, 0]
                                dvy = vel[j, 1] - vel[i, 1]
                                dvz = vel[j, 2] - vel[i, 2]

                                if body_id[i] == body_id[j]:
                                    if body_id[i] == 2:
                                        # Water-water viscosity
                                        visc = 0.08 * mass[j] / density[j] * (dvx * nx + dvy * ny + dvz * nz)
                                        forces[i, 0] += visc * nx
                                        forces[i, 1] += visc * ny
                                        forces[i, 2] += visc * nz
                                    else:
                                        # Sand-sand cohesion & internal friction
                                        # Cohesion depends on saturation: drops when wet, causing erosion/yielding
                                        cohesion = max(0.05, 1.0 - 0.90 * saturation[i]) * 15.0
                                        # Cohesion acts as attraction/shear resistance when in contact
                                        if r < 0.85 * h:
                                            f_coh = -cohesion * (r - 0.85 * h) * mass[j]
                                            forces[i, 0] += f_coh * nx
                                            forces[i, 1] += f_coh * ny
                                            forces[i, 2] += f_coh * nz
                                        # Granular damping
                                        damping = 0.45 * mass[j]
                                        forces[i, 0] += damping * dvx
                                        forces[i, 1] += damping * dvy
                                        forces[i, 2] += damping * dvz
                                else:
                                    # Inter-phase interaction: Water <-> Sand
                                    # Drag / seepage resistance (Ergun/Darcy-like momentum exchange)
                                    drag_coeff = 22.0 * (1.0 + 2.0 * (1.0 - saturation[i if body_id[i] == 1 else j]))
                                    f_drag_x = drag_coeff * dvx * (mass[j] / (density[i] + density[j]))
                                    f_drag_y = drag_coeff * dvy * (mass[j] / (density[i] + density[j]))
                                    f_drag_z = drag_coeff * dvz * (mass[j] / (density[i] + density[j]))
                                    forces[i, 0] += f_drag_x
                                    forces[i, 1] += f_drag_y
                                    forces[i, 2] += f_drag_z

                        j = next_node[j]

    # Time integration & boundary handling
    for i in range(n):
        ax = forces[i, 0] / mass[i]
        ay = forces[i, 1] / mass[i]
        az = forces[i, 2] / mass[i]

        # Velocity limit for stability
        v_mag = math.sqrt(vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1] + vel[i, 2] * vel[i, 2])
        if v_mag > 25.0:
            vel[i, 0] *= 25.0 / v_mag
            vel[i, 1] *= 25.0 / v_mag
            vel[i, 2] *= 25.0 / v_mag

        vel[i, 0] += ax * dt
        vel[i, 1] += ay * dt
        vel[i, 2] += az * dt

        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt

        # Boundary collisions
        margin = 0.04
        # Floor z = 0.0
        if pos[i, 2] < floor_z + margin:
            pos[i, 2] = floor_z + margin
            if vel[i, 2] < 0.0:
                vel[i, 2] = -0.1 * vel[i, 2]
                # Friction
                vel[i, 0] *= 0.75
                vel[i, 1] *= 0.75

        # Reservoir end wall x_min = -3.93
        if pos[i, 0] < wall_x_min + margin:
            pos[i, 0] = wall_x_min + margin
            if vel[i, 0] < 0.0:
                vel[i, 0] = -0.1 * vel[i, 0]

        # Downstream retaining wall x_max = 4.43
        if pos[i, 0] > wall_x_max - margin:
            pos[i, 0] = wall_x_max - margin
            if vel[i, 0] > 0.0:
                vel[i, 0] = -0.1 * vel[i, 0]

        # Side walls y_min = -0.95, y_max = 0.95
        if pos[i, 1] < wall_y_min + margin:
            pos[i, 1] = wall_y_min + margin
            if vel[i, 1] < 0.0:
                vel[i, 1] = -0.1 * vel[i, 1]

        if pos[i, 1] > wall_y_max - margin:
            pos[i, 1] = wall_y_max - margin
            if vel[i, 1] > 0.0:
                vel[i, 1] = -0.1 * vel[i, 1]


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.current_time = 0.0

        f0 = scene.initial_state()
        self.pos = f0.position.copy()
        self.vel = f0.velocity.copy()
        self.mass = f0.mass.copy()
        self.body_id = f0.body_id.copy()
        self.attributes = dict(f0.attributes)
        self.saturation = self.attributes["sand_saturation"].copy()
        self.porosity = self.attributes["porosity"].copy()
        self.particle_volume = self.attributes["particle_volume"].copy()
        self.scalars = dict(f0.scalars)

        self.num_particles = len(self.pos)
        self.sand_mask = (self.body_id == scene.SAND_ID)
        self.water_mask = (self.body_id == scene.WATER_ID)

        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Domain bounds
        self.domain_bounds = np.array([
            [-4.2, -1.1, -0.1],
            [4.6, 1.1, 3.0]
        ], dtype=np.float64)

        self.wall_x_min = -3.93
        self.wall_x_max = 4.43
        self.wall_y_min = -0.95
        self.wall_y_max = 0.95
        self.floor_z = 0.0

        self.sub_dt = 0.0025  # Stable sub-stepping timestep

        # Cumulative budget fields
        self.G = 2  # 2 domains: Sand (0) and Water (1)
        self.K = 2  # 2 materials: Sand (0) and Water (1)

        self.budget_cumulative_external_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.budget_cumulative_inflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.budget_cumulative_outflow_momentum = np.zeros((self.G, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros((self.K,), dtype=np.float64)

        # Reference momentum scale
        total_mass = float(np.sum(self.mass))
        self.scalars["reference_momentum"] = float(total_mass * math.sqrt(2.0 * 9.81 * 1.5))

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        dt_needed = target_time - self.current_time

        if dt_needed > 1e-7:
            num_steps = int(math.ceil(dt_needed / self.sub_dt))
            step_dt = dt_needed / num_steps

            for _ in range(num_steps):
                _solve_substep(
                    self.pos,
                    self.vel,
                    self.mass,
                    self.body_id,
                    self.saturation,
                    step_dt,
                    self.gravity,
                    self.domain_bounds,
                    self.wall_x_min,
                    self.wall_x_max,
                    self.wall_y_min,
                    self.wall_y_max,
                    self.floor_z,
                )
                self.current_time += step_dt

                # Track cumulative impulse (gravity impulse)
                sand_mass = np.sum(self.mass[self.sand_mask])
                water_mass = np.sum(self.mass[self.water_mask])
                self.budget_cumulative_external_impulse[0] += sand_mass * self.gravity * step_dt
                self.budget_cumulative_external_impulse[1] += water_mass * self.gravity * step_dt

        self.current_time = target_time

        # Prepare StateFrame
        attributes = dict(self.attributes)
        attributes["sand_saturation"] = self.saturation.copy()
        attributes["mass_quantity"] = self.mass.copy()
        attributes["material_id"] = self.body_id.copy()
        attributes["porosity"] = self.porosity.copy()
        attributes["particle_volume"] = self.particle_volume.copy()

        domain_weights = np.zeros((self.G, self.num_particles), dtype=np.float64)
        domain_weights[0, self.sand_mask] = 1.0
        domain_weights[1, self.water_mask] = 1.0

        sand_mass = np.sum(self.mass[self.sand_mask])
        water_mass = np.sum(self.mass[self.water_mask])
        budget_external_force = np.zeros((self.G, self.K, 3), dtype=np.float64)
        budget_external_force[0, 0] = sand_mass * self.gravity
        budget_external_force[1, 1] = water_mass * self.gravity

        fields = {
            "budget_cumulative_external_impulse": self.budget_cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.budget_cumulative_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.budget_cumulative_outflow_momentum.copy(),
            "budget_domain_weights": domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": np.zeros((self.K, 3), dtype=np.float64),
        }

        scalars = dict(self.scalars)

        frame = StateFrame(
            self.current_time,
            self.pos.copy(),
            self.vel.copy(),
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
            "current_time": self.current_time,
            "particle_count": self.num_particles,
        }

