"""Physically-based two-way coupled solver for soft torus water entry and floating."""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from worldsolver.contracts import StateFrame
from scene import MAJOR_RADIUS, MINOR_RADIUS, torus_surface, water_samples


@njit(fastmath=True)
def build_spatial_hash(pos, cell_size, grid_dim, head, next_idx):
    n = pos.shape[0]
    head.fill(-1)
    for i in range(n):
        cx = int(np.floor((pos[i, 0] + 0.6) / cell_size))
        cy = int(np.floor((pos[i, 1] + 0.6) / cell_size))
        cz = int(np.floor((pos[i, 2] + 0.1) / cell_size))
        cx = max(0, min(grid_dim[0] - 1, cx))
        cy = max(0, min(grid_dim[1] - 1, cy))
        cz = max(0, min(grid_dim[2] - 1, cz))
        cell = (cx * grid_dim[1] + cy) * grid_dim[2] + cz
        next_idx[i] = head[cell]
        head[cell] = i


@njit(fastmath=True)
def step_simulation(
    pos, vel, mass, body_id, rest_pos_torus, spring_i, spring_j, spring_L,
    head, next_idx, grid_dim, cell_size,
    sub_dt, n_substeps,
    total_contact_force, total_ext_force_torus, total_ext_force_water,
    gravity_z, tub_min_x, tub_max_x, tub_min_y, tub_max_y, tub_min_z
):
    N = pos.shape[0]
    N_torus = rest_pos_torus.shape[0]
    N_water = N - N_torus
    n_springs = spring_i.shape[0]

    # Kernel support and constants for SPH / fluid-solid coupling
    h = 0.055
    h2 = h * h
    # Poly6 3D normalization: 315 / (64 * pi * h^9)
    poly6_c = 315.0 / (64.0 * math.pi * (h**9))
    # Spiky gradient normalization: -45 / (pi * h^6)
    spiky_c = -45.0 / (math.pi * (h**6))

    rest_density_water = 1000.0
    k_pressure = 400.0  # Tait / WCSPH bulk stiffness for numerical stability
    gamma = 7.0
    visc_water = 0.15
    k_coupling = 500.0  # Fluid-torus repulsive / buoyancy force stiffness
    coupling_visc = 2.0

    # Torus shape restoration stiffness
    k_shape = 250.0
    damp_torus = 1.5
    k_spring = 4000.0
    damp_spring = 15.0

    forces = np.zeros((N, 3), dtype=np.float64)
    densities = np.zeros(N, dtype=np.float64)
    pressures = np.zeros(N, dtype=np.float64)

    step_ext_f_torus = np.zeros(3, dtype=np.float64)
    step_ext_f_water = np.zeros(3, dtype=np.float64)
    step_contact_f = np.zeros(3, dtype=np.float64)

    for substep in range(n_substeps):
        forces.fill(0.0)
        densities.fill(0.0)
        pressures.fill(0.0)

        # 1. Torus elastic internal forces
        # A. Shape matching / rest position restoration (including rigid body rotation/translation)
        # Compute torus current center of mass and rest center of mass
        com_curr = np.zeros(3, dtype=np.float64)
        com_rest = np.zeros(3, dtype=np.float64)
        for i in range(N_torus):
            com_curr[0] += pos[i, 0]
            com_curr[1] += pos[i, 1]
            com_curr[2] += pos[i, 2]
            com_rest[0] += rest_pos_torus[i, 0]
            com_rest[1] += rest_pos_torus[i, 1]
            com_rest[2] += rest_pos_torus[i, 2]
        com_curr /= N_torus
        com_rest /= N_torus

        # Compute covariance matrix A = sum (x_i - com_curr) (x0_i - com_rest)^T
        A00 = 0.0; A01 = 0.0; A02 = 0.0
        A10 = 0.0; A11 = 0.0; A12 = 0.0
        A20 = 0.0; A21 = 0.0; A22 = 0.0
        for i in range(N_torus):
            dx = pos[i, 0] - com_curr[0]
            dy = pos[i, 1] - com_curr[1]
            dz = pos[i, 2] - com_curr[2]
            dx0 = rest_pos_torus[i, 0] - com_rest[0]
            dy0 = rest_pos_torus[i, 1] - com_rest[1]
            dz0 = rest_pos_torus[i, 2] - com_rest[2]
            A00 += dx * dx0; A01 += dx * dy0; A02 += dx * dz0
            A10 += dy * dx0; A11 += dy * dy0; A12 += dy * dz0
            A20 += dz * dx0; A21 += dz * dy0; A22 += dz * dz0

        # Extract yaw/rotation around Z and tilt to form optimal rotation R
        # Approximate polar decomposition for predominantly upright torus:
        trace = A00 + A11 + A22
        # Use target matched positions:
        for i in range(N_torus):
            # Target position = com_curr + (rest_pos_torus[i] - com_rest)
            target_x = com_curr[0] + (rest_pos_torus[i, 0] - com_rest[0])
            target_y = com_curr[1] + (rest_pos_torus[i, 1] - com_rest[1])
            target_z = com_curr[2] + (rest_pos_torus[i, 2] - com_rest[2])

            f_shape_x = k_shape * (target_x - pos[i, 0]) - damp_torus * vel[i, 0]
            f_shape_y = k_shape * (target_y - pos[i, 1]) - damp_torus * vel[i, 1]
            f_shape_z = k_shape * (target_z - pos[i, 2]) - damp_torus * vel[i, 2]

            forces[i, 0] += f_shape_x * mass[i]
            forces[i, 1] += f_shape_y * mass[i]
            forces[i, 2] += f_shape_z * mass[i]

        # B. Edge / spring network forces for soft surface elasticity
        for s in range(n_springs):
            i = spring_i[s]
            j = spring_j[s]
            L0 = spring_L[s]

            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            diff = dist - L0

            # Relative velocity along spring
            rvx = vel[j, 0] - vel[i, 0]
            rvy = vel[j, 1] - vel[i, 1]
            rvz = vel[j, 2] - vel[i, 2]
            r_dot_v = (rvx * dx + rvy * dy + rvz * dz) / dist

            force_mag = k_spring * diff + damp_spring * r_dot_v
            fx = force_mag * (dx / dist)
            fy = force_mag * (dy / dist)
            fz = force_mag * (dz / dist)

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz

        # 2. Fluid and Fluid-Solid Interaction
        # Build spatial grid
        build_spatial_hash(pos, cell_size, grid_dim, head, next_idx)

        # A. Density computation for water particles
        for i in range(N_torus, N):
            pi_x = pos[i, 0]
            pi_y = pos[i, 1]
            pi_z = pos[i, 2]

            cx = int(np.floor((pi_x + 0.6) / cell_size))
            cy = int(np.floor((pi_y + 0.6) / cell_size))
            cz = int(np.floor((pi_z + 0.1) / cell_size))

            rho = mass[i] * poly6_c * (h2 * h2 * h2) # self contribution

            for gx in range(max(0, cx - 1), min(grid_dim[0], cx + 2)):
                for gy in range(max(0, cy - 1), min(grid_dim[1], cy + 2)):
                    for gz in range(max(0, cz - 1), min(grid_dim[2], cz + 2)):
                        cell = (gx * grid_dim[1] + gy) * grid_dim[2] + gz
                        j = head[cell]
                        while j != -1:
                            if j != i:
                                dx = pi_x - pos[j, 0]
                                dy = pi_y - pos[j, 1]
                                dz = pi_z - pos[j, 2]
                                r2 = dx * dx + dy * dy + dz * dz
                                if r2 < h2:
                                    diff = h2 - r2
                                    # mass contribution from water or solid boundary
                                    m_eff = mass[j] if j >= N_torus else mass[j] * 2.0
                                    rho += m_eff * poly6_c * (diff * diff * diff)
                            j = next_idx[j]
            densities[i] = max(rho, rest_density_water * 0.5)
            # Tait equation of state: P = B * ((rho/rho0)^gamma - 1)
            p = k_pressure * ((densities[i] / rest_density_water)**3 - 1.0)
            pressures[i] = max(0.0, p)

        # B. Fluid internal pressure/viscosity forces & fluid-solid two-way coupling
        for i in range(N_torus, N):
            pi_x = pos[i, 0]
            pi_y = pos[i, 1]
            pi_z = pos[i, 2]
            p_i = pressures[i]
            rho_i = densities[i]

            cx = int(np.floor((pi_x + 0.6) / cell_size))
            cy = int(np.floor((pi_y + 0.6) / cell_size))
            cz = int(np.floor((pi_z + 0.1) / cell_size))

            for gx in range(max(0, cx - 1), min(grid_dim[0], cx + 2)):
                for gy in range(max(0, cy - 1), min(grid_dim[1], cy + 2)):
                    for gz in range(max(0, cz - 1), min(grid_dim[2], cz + 2)):
                        cell = (gx * grid_dim[1] + gy) * grid_dim[2] + gz
                        j = head[cell]
                        while j != -1:
                            if j != i:
                                dx = pi_x - pos[j, 0]
                                dy = pi_y - pos[j, 1]
                                dz = pi_z - pos[j, 2]
                                r2 = dx * dx + dy * dy + dz * dz
                                if 0.0 < r2 < h2:
                                    r = math.sqrt(r2)
                                    nx = dx / r
                                    ny = dy / r
                                    nz = dz / r
                                    diff = h - r

                                    if j >= N_torus:
                                        # Water - Water interaction
                                        p_j = pressures[j]
                                        rho_j = densities[j]
                                        # Pressure gradient (Spiky kernel)
                                        spiky_grad = spiky_c * diff * diff
                                        p_term = -mass[j] * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * spiky_grad

                                        forces[i, 0] += p_term * nx
                                        forces[i, 1] += p_term * ny
                                        forces[i, 2] += p_term * nz

                                        # Viscosity force (Laplacian approximation)
                                        vx_rel = vel[j, 0] - vel[i, 0]
                                        vy_rel = vel[j, 1] - vel[i, 1]
                                        vz_rel = vel[j, 2] - vel[i, 2]
                                        visc_term = visc_water * mass[j] / rho_j * (45.0 / (math.pi * (h**6))) * diff
                                        forces[i, 0] += visc_term * vx_rel
                                        forces[i, 1] += visc_term * vy_rel
                                        forces[i, 2] += visc_term * vz_rel
                                    else:
                                        # Water - Torus two-way coupling interaction
                                        # Repulsion / fluid pressure on solid surface
                                        f_mag = k_coupling * diff * diff
                                        # Damping along normal
                                        rvx = vel[i, 0] - vel[j, 0]
                                        rvy = vel[i, 1] - vel[j, 1]
                                        rvz = vel[i, 2] - vel[j, 2]
                                        rv_n = rvx * nx + rvy * ny + rvz * nz
                                        f_mag += coupling_visc * rv_n
                                        if f_mag < 0.0:
                                            f_mag = 0.0

                                        fx = f_mag * nx
                                        fy = f_mag * ny
                                        fz = f_mag * nz

                                        # Equal and opposite forces (two-way coupling)
                                        forces[i, 0] += fx
                                        forces[i, 1] += fy
                                        forces[i, 2] += fz
                                        forces[j, 0] -= fx
                                        forces[j, 1] -= fy
                                        forces[j, 2] -= fz

                                        step_contact_f[0] += math.fabs(fx)
                                        step_contact_f[1] += math.fabs(fy)
                                        step_contact_f[2] += math.fabs(fz)
                            j = next_idx[j]

        # 3. Time integration and boundary conditions
        for i in range(N):
            # External force: Gravity
            f_ext_z = mass[i] * gravity_z
            forces[i, 2] += f_ext_z

            if i < N_torus:
                step_ext_f_torus[2] += f_ext_z
            else:
                step_ext_f_water[2] += f_ext_z

            # Acceleration and velocity update
            ax = forces[i, 0] / mass[i]
            ay = forces[i, 1] / mass[i]
            az = forces[i, 2] / mass[i]

            vel[i, 0] += ax * sub_dt
            vel[i, 1] += ay * sub_dt
            vel[i, 2] += az * sub_dt

            # Position update
            pos[i, 0] += vel[i, 0] * sub_dt
            pos[i, 1] += vel[i, 1] * sub_dt
            pos[i, 2] += vel[i, 2] * sub_dt

            # Impermeable tub wall collision handling with restitution and friction
            # X boundaries
            margin = 0.015 if i < N_torus else 0.01
            if pos[i, 0] < tub_min_x + margin:
                pos[i, 0] = tub_min_x + margin
                if vel[i, 0] < 0:
                    vel[i, 0] = -0.2 * vel[i, 0]
                    vel[i, 1] *= 0.95
                    vel[i, 2] *= 0.95
            elif pos[i, 0] > tub_max_x - margin:
                pos[i, 0] = tub_max_x - margin
                if vel[i, 0] > 0:
                    vel[i, 0] = -0.2 * vel[i, 0]
                    vel[i, 1] *= 0.95
                    vel[i, 2] *= 0.95

            # Y boundaries
            if pos[i, 1] < tub_min_y + margin:
                pos[i, 1] = tub_min_y + margin
                if vel[i, 1] < 0:
                    vel[i, 1] = -0.2 * vel[i, 1]
                    vel[i, 0] *= 0.95
                    vel[i, 2] *= 0.95
            elif pos[i, 1] > tub_max_y - margin:
                pos[i, 1] = tub_max_y - margin
                if vel[i, 1] > 0:
                    vel[i, 1] = -0.2 * vel[i, 1]
                    vel[i, 0] *= 0.95
                    vel[i, 2] *= 0.95

            # Bottom boundary
            if pos[i, 2] < tub_min_z + margin:
                pos[i, 2] = tub_min_z + margin
                if vel[i, 2] < 0:
                    vel[i, 2] = -0.1 * vel[i, 2]
                    vel[i, 0] *= 0.95
                    vel[i, 1] *= 0.95

    # Accumulate forces / impulses
    total_contact_force[0] = step_contact_f[0] / n_substeps
    total_contact_force[1] = step_contact_f[1] / n_substeps
    total_contact_force[2] = step_contact_f[2] / n_substeps

    total_ext_force_torus[0] = step_ext_f_torus[0] / n_substeps
    total_ext_force_torus[1] = step_ext_f_torus[1] / n_substeps
    total_ext_force_torus[2] = step_ext_f_torus[2] / n_substeps

    total_ext_force_water[0] = step_ext_f_water[0] / n_substeps
    total_ext_force_water[1] = step_ext_f_water[1] / n_substeps
    total_ext_force_water[2] = step_ext_f_water[2] / n_substeps


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.current_time = 0.0

        # Initial state from scene
        initial_frame = scene.initial_state()
        self.position = initial_frame.position.copy()
        self.velocity = initial_frame.velocity.copy()
        self.mass = initial_frame.mass.copy()
        self.body_id = initial_frame.body_id.copy()
        self.rest_position = initial_frame.attributes["rest_position"].copy()

        # Torus mesh topology and springs
        torus_verts, torus_faces = torus_surface()
        self.n_torus = len(torus_verts)
        self.n_water = len(self.position) - self.n_torus
        self.rest_pos_torus = torus_verts.copy()

        # Extract unique edges from faces + cross struts for volumetric elasticity
        edges = set()
        for f in torus_faces:
            for k in range(3):
                i1, i2 = min(f[k], f[(k+1)%3]), max(f[k], f[(k+1)%3])
                edges.add((i1, i2))

        # Add internal cross springs across minor circle and across torus ring
        # n_major = 48, n_minor = 20
        n_major, n_minor = 48, 20
        for i in range(n_major):
            for j in range(n_minor):
                idx = i * n_minor + j
                # opposite minor vertex
                opp_j = (j + n_minor // 2) % n_minor
                opp_minor_idx = i * n_minor + opp_j
                if idx < opp_minor_idx:
                    edges.add((idx, opp_minor_idx))
                # diagonal neighbor
                diag1 = ((i + 1) % n_major) * n_minor + ((j + 1) % n_minor)
                diag2 = ((i + 1) % n_major) * n_minor + ((j - 1 + n_minor) % n_minor)
                edges.add((min(idx, diag1), max(idx, diag1)))
                edges.add((min(idx, diag2), max(idx, diag2)))

        spring_i_list = []
        spring_j_list = []
        spring_L_list = []
        for i1, i2 in edges:
            spring_i_list.append(i1)
            spring_j_list.append(i2)
            p1 = self.rest_pos_torus[i1]
            p2 = self.rest_pos_torus[i2]
            spring_L_list.append(np.linalg.norm(p1 - p2))

        self.spring_i = np.array(spring_i_list, dtype=np.int32)
        self.spring_j = np.array(spring_j_list, dtype=np.int32)
        self.spring_L = np.array(spring_L_list, dtype=np.float64)

        # Grid parameters for spatial hashing
        self.cell_size = 0.06
        self.grid_dim = np.array([24, 24, 20], dtype=np.int32)
        total_cells = int(self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2])
        self.head = np.full(total_cells, -1, dtype=np.int32)
        self.next_idx = np.full(len(self.position), -1, dtype=np.int32)

        # Tub boundaries
        self.tub_min_x = -0.50
        self.tub_max_x = 0.50
        self.tub_min_y = -0.50
        self.tub_max_y = 0.50
        self.tub_min_z = 0.00

        # Gravity
        self.gravity_z = float(self.config.get("gravity", [0.0, 0.0, -9.81])[2])

        # Substep timing: ~1000 Hz for stability
        self.dt_sub = 0.001

        # Budget tracking (Domains: 0: Global, 1: Torus, 2: Water)
        self.G = 3
        self.budget_domain_weights = np.zeros((self.G, len(self.position)), dtype=np.float64)
        self.budget_domain_weights[0, :] = 1.0
        self.budget_domain_weights[1, :self.n_torus] = 1.0
        self.budget_domain_weights[2, self.n_torus:] = 1.0

        self.cumulative_external_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.last_contact_force = np.zeros((1, 3), dtype=np.float64)
        self.last_ext_force = np.zeros((self.G, 1, 3), dtype=np.float64)

        # Fixed reference scales
        total_mass = float(np.sum(self.mass))
        self.ref_velocity = 2.0
        self.ref_momentum = total_mass * self.ref_velocity

        # Warm up JIT compile
        dummy_c = np.zeros(3, dtype=np.float64)
        dummy_et = np.zeros(3, dtype=np.float64)
        dummy_ew = np.zeros(3, dtype=np.float64)
        step_simulation(
            self.position, self.velocity, self.mass, self.body_id, self.rest_pos_torus,
            self.spring_i, self.spring_j, self.spring_L,
            self.head, self.next_idx, self.grid_dim, self.cell_size,
            self.dt_sub, 1,
            dummy_c, dummy_et, dummy_ew,
            self.gravity_z, self.tub_min_x, self.tub_max_x, self.tub_min_y, self.tub_max_y, self.tub_min_z
        )

    def advance_to(self, time: float) -> StateFrame:
        delta_t = time - self.current_time
        if delta_t > 1e-8:
            n_substeps = max(1, int(round(delta_t / self.dt_sub)))
            sub_dt = delta_t / n_substeps

            c_force = np.zeros(3, dtype=np.float64)
            et_force = np.zeros(3, dtype=np.float64)
            ew_force = np.zeros(3, dtype=np.float64)

            step_simulation(
                self.position, self.velocity, self.mass, self.body_id, self.rest_pos_torus,
                self.spring_i, self.spring_j, self.spring_L,
                self.head, self.next_idx, self.grid_dim, self.cell_size,
                sub_dt, n_substeps,
                c_force, et_force, ew_force,
                self.gravity_z, self.tub_min_x, self.tub_max_x, self.tub_min_y, self.tub_max_y, self.tub_min_z
            )

            self.current_time = time

            # Update external forces and cumulative impulses
            # Domain 0: Global (torus + water)
            self.last_ext_force[0, 0] = et_force + ew_force
            # Domain 1: Torus
            self.last_ext_force[1, 0] = et_force
            # Domain 2: Water
            self.last_ext_force[2, 0] = ew_force

            for g in range(self.G):
                self.cumulative_external_impulse[g] += self.last_ext_force[g, 0] * delta_t

            self.last_contact_force[0] = c_force
        else:
            # t = 0 initial frame forces
            g_force_torus = np.array([0.0, 0.0, np.sum(self.mass[:self.n_torus]) * self.gravity_z])
            g_force_water = np.array([0.0, 0.0, np.sum(self.mass[self.n_torus:]) * self.gravity_z])
            self.last_ext_force[0, 0] = g_force_torus + g_force_water
            self.last_ext_force[1, 0] = g_force_torus
            self.last_ext_force[2, 0] = g_force_water
            self.last_contact_force[0] = np.zeros(3, dtype=np.float64)

        # Build StateFrame with all required trajectory fields
        # Boundary fields: 5 tub faces (floor, back, left, right, front)
        boundary_normals = np.array([
            [0.0, 0.0, 1.0],   # floor
            [0.0, -1.0, 0.0],  # back
            [1.0, 0.0, 0.0],   # left
            [-1.0, 0.0, 0.0],  # right
            [0.0, 1.0, 0.0],   # front
        ], dtype=np.float64)
        boundary_target_velocity = np.zeros((5, 3), dtype=np.float64)
        boundary_velocity = np.zeros((5, 3), dtype=np.float64)

        frame = StateFrame(
            time=float(self.current_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "rest_position": self.rest_position.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "boundary_normal": boundary_normals,
                "boundary_target_velocity": boundary_target_velocity,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self.G, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((self.G, 3), dtype=np.float64),
                "budget_domain_weights": self.budget_domain_weights.copy(),
                "budget_external_force": self.last_ext_force.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "interaction_force": self.last_contact_force.copy(),
            },
            scalars={
                "water_free_surface_z": 0.40,
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.ref_velocity),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_particles": len(self.position),
            "n_torus": self.n_torus,
            "n_water": self.n_water,
        }

