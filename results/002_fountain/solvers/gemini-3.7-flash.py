"""Affine Particle-in-Cell (APIC) Fluid Simulation Solver for the Fountain Benchmark."""
from __future__ import annotations

import numpy as np
import numba
from worldsolver.contracts import StateFrame


@numba.njit(fastmath=True)
def p2g_kernel(pos, vel, C_mat, mass, grid_m, grid_v, inv_dx, origin, nx, ny, nz):
    n_p = pos.shape[0]
    for p in range(n_p):
        xp = pos[p]
        vp = vel[p]
        Cp = C_mat[p]
        mp = mass[p]

        base_x = int(np.floor((xp[0] - origin[0]) * inv_dx - 0.5))
        base_y = int(np.floor((xp[1] - origin[1]) * inv_dx - 0.5))
        base_z = int(np.floor((xp[2] - origin[2]) * inv_dx - 0.5))

        fx = (xp[0] - origin[0]) * inv_dx - (base_x + 0.5)
        fy = (xp[1] - origin[1]) * inv_dx - (base_y + 0.5)
        fz = (xp[2] - origin[2]) * inv_dx - (base_z + 0.5)

        # Quadratic B-spline weights
        wx_0 = 0.5 * (1.5 - fx) ** 2
        wx_1 = 0.75 - (fx - 1.0) ** 2
        wx_2 = 0.5 * (fx - 0.5) ** 2

        wy_0 = 0.5 * (1.5 - fy) ** 2
        wy_1 = 0.75 - (fy - 1.0) ** 2
        wy_2 = 0.5 * (fy - 0.5) ** 2

        wz_0 = 0.5 * (1.5 - fz) ** 2
        wz_1 = 0.75 - (fz - 1.0) ** 2
        wz_2 = 0.5 * (fz - 0.5) ** 2

        wxs = (wx_0, wx_1, wx_2)
        wys = (wy_0, wy_1, wy_2)
        wzs = (wz_0, wz_1, wz_2)

        for i in range(3):
            gx = base_x + i
            if gx < 0 or gx >= nx:
                continue
            wx = wxs[i]
            dpos_x = (gx + 0.5) / inv_dx + origin[0] - xp[0]
            for j in range(3):
                gy = base_y + j
                if gy < 0 or gy >= ny:
                    continue
                wxy = wx * wys[j]
                dpos_y = (gy + 0.5) / inv_dx + origin[1] - xp[1]
                for k in range(3):
                    gz = base_z + k
                    if gz < 0 or gz >= nz:
                        continue
                    weight = wxy * wzs[k]
                    dpos_z = (gz + 0.5) / inv_dx + origin[2] - xp[2]

                    # APIC velocity: v + C * dpos
                    affine_v_x = vp[0] + Cp[0, 0] * dpos_x + Cp[0, 1] * dpos_y + Cp[0, 2] * dpos_z
                    affine_v_y = vp[1] + Cp[1, 0] * dpos_x + Cp[1, 1] * dpos_y + Cp[1, 2] * dpos_z
                    affine_v_z = vp[2] + Cp[2, 0] * dpos_x + Cp[2, 1] * dpos_y + Cp[2, 2] * dpos_z

                    weighted_m = weight * mp
                    grid_m[gx, gy, gz] += weighted_m
                    grid_v[gx, gy, gz, 0] += weighted_m * affine_v_x
                    grid_v[gx, gy, gz, 1] += weighted_m * affine_v_y
                    grid_v[gx, gy, gz, 2] += weighted_m * affine_v_z


@numba.njit(fastmath=True)
def grid_update_kernel(grid_m, grid_v, grid_type, dt, gx, gy, gz_grav,
                       r_basin_sq, r_basin, z_bot, z_lip, r_nozzle_sq, z_nozzle,
                       origin, dx, nx, ny, nz):
    for i in range(nx):
        x = origin[0] + (i + 0.5) * dx
        for j in range(ny):
            y = origin[1] + (j + 0.5) * dx
            r_sq = x * x + y * y
            for k in range(nz):
                z = origin[2] + (k + 0.5) * dx
                m = grid_m[i, j, k]
                if m > 1e-12:
                    grid_type[i, j, k] = 1  # FLUID
                    inv_m = 1.0 / m
                    grid_v[i, j, k, 0] = grid_v[i, j, k, 0] * inv_m + dt * gx
                    grid_v[i, j, k, 1] = grid_v[i, j, k, 1] * inv_m + dt * gy
                    grid_v[i, j, k, 2] = grid_v[i, j, k, 2] * inv_m + dt * gz_grav

                    # Basin boundary collision
                    if z < z_lip and r_sq > r_basin_sq:
                        r = np.sqrt(r_sq)
                        nx_c = x / r
                        ny_c = y / r
                        vn = grid_v[i, j, k, 0] * nx_c + grid_v[i, j, k, 1] * ny_c
                        if vn > 0.0:
                            grid_v[i, j, k, 0] -= vn * nx_c
                            grid_v[i, j, k, 1] -= vn * ny_c

                    # Bottom wall
                    if z <= z_bot:
                        if grid_v[i, j, k, 2] < 0.0:
                            grid_v[i, j, k, 2] = 0.0

                    # Nozzle collision (cylinder: r <= r_nozzle, z <= z_nozzle)
                    if z <= z_nozzle and r_sq <= r_nozzle_sq:
                        r = np.sqrt(r_sq)
                        if r > 1e-6:
                            nx_c = x / r
                            ny_c = y / r
                            vn = grid_v[i, j, k, 0] * nx_c + grid_v[i, j, k, 1] * ny_c
                            if vn < 0.0:
                                grid_v[i, j, k, 0] -= vn * nx_c
                                grid_v[i, j, k, 1] -= vn * ny_c
                else:
                    grid_type[i, j, k] = 0  # AIR
                    grid_v[i, j, k, 0] = 0.0
                    grid_v[i, j, k, 1] = 0.0
                    grid_v[i, j, k, 2] = 0.0


@numba.njit(fastmath=True)
def jacobi_pressure_kernel(p, p_new, grid_type, div, nx, ny, nz):
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if grid_type[i, j, k] != 1:
                    p_new[i, j, k] = 0.0
                    continue

                d = div[i, j, k]
                num_neighbors = 0.0
                sum_p = 0.0

                # -x
                if i > 0:
                    t = grid_type[i - 1, j, k]
                    if t == 1:
                        sum_p += p[i - 1, j, k]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0  # Dirichlet p=0
                # +x
                if i < nx - 1:
                    t = grid_type[i + 1, j, k]
                    if t == 1:
                        sum_p += p[i + 1, j, k]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0
                # -y
                if j > 0:
                    t = grid_type[i, j - 1, k]
                    if t == 1:
                        sum_p += p[i, j - 1, k]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0
                # +y
                if j < ny - 1:
                    t = grid_type[i, j + 1, k]
                    if t == 1:
                        sum_p += p[i, j + 1, k]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0
                # -z
                if k > 0:
                    t = grid_type[i, j, k - 1]
                    if t == 1:
                        sum_p += p[i, j, k - 1]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0
                # +z
                if k < nz - 1:
                    t = grid_type[i, j, k + 1]
                    if t == 1:
                        sum_p += p[i, j, k + 1]
                        num_neighbors += 1.0
                    elif t == 0:
                        num_neighbors += 1.0

                if num_neighbors > 0.0:
                    p_new[i, j, k] = (sum_p - d) / num_neighbors
                else:
                    p_new[i, j, k] = 0.0


@numba.njit(fastmath=True)
def apply_pressure_gradient(grid_v, p, grid_type, scale, nx, ny, nz):
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if grid_type[i, j, k] != 1:
                    continue

                # Gradient in x
                p_left = p[i - 1, j, k] if (i > 0 and grid_type[i - 1, j, k] == 1) else (0.0 if (i > 0 and grid_type[i - 1, j, k] == 0) else p[i, j, k])
                p_right = p[i + 1, j, k] if (i < nx - 1 and grid_type[i + 1, j, k] == 1) else (0.0 if (i < nx - 1 and grid_type[i + 1, j, k] == 0) else p[i, j, k])
                grid_v[i, j, k, 0] -= scale * 0.5 * (p_right - p_left)

                # Gradient in y
                p_down = p[i, j - 1, k] if (j > 0 and grid_type[i, j - 1, k] == 1) else (0.0 if (j > 0 and grid_type[i, j - 1, k] == 0) else p[i, j, k])
                p_up = p[i, j + 1, k] if (j < ny - 1 and grid_type[i, j + 1, k] == 1) else (0.0 if (j < ny - 1 and grid_type[i, j + 1, k] == 0) else p[i, j, k])
                grid_v[i, j, k, 1] -= scale * 0.5 * (p_up - p_down)

                # Gradient in z
                p_back = p[i, j, k - 1] if (k > 0 and grid_type[i, j, k - 1] == 1) else (0.0 if (k > 0 and grid_type[i, j, k - 1] == 0) else p[i, j, k])
                p_front = p[i, j, k + 1] if (k < nz - 1 and grid_type[i, j, k + 1] == 1) else (0.0 if (k < nz - 1 and grid_type[i, j, k + 1] == 0) else p[i, j, k])
                grid_v[i, j, k, 2] -= scale * 0.5 * (p_front - p_back)


@numba.njit(fastmath=True)
def g2p_and_advect_kernel(pos, vel, C_mat, grid_v, inv_dx, dt, origin, nx, ny, nz,
                          r_basin, z_bot, z_lip, r_nozzle, z_nozzle):
    n_p = pos.shape[0]
    inv_D = 4.0 * inv_dx * inv_dx  # for quadratic B-spline APIC, D_p = (1/4) * dx^2 * I

    for p in range(n_p):
        xp = pos[p]
        base_x = int(np.floor((xp[0] - origin[0]) * inv_dx - 0.5))
        base_y = int(np.floor((xp[1] - origin[1]) * inv_dx - 0.5))
        base_z = int(np.floor((xp[2] - origin[2]) * inv_dx - 0.5))

        fx = (xp[0] - origin[0]) * inv_dx - (base_x + 0.5)
        fy = (xp[1] - origin[1]) * inv_dx - (base_y + 0.5)
        fz = (xp[2] - origin[2]) * inv_dx - (base_z + 0.5)

        wx_0 = 0.5 * (1.5 - fx) ** 2
        wx_1 = 0.75 - (fx - 1.0) ** 2
        wx_2 = 0.5 * (fx - 0.5) ** 2

        wy_0 = 0.5 * (1.5 - fy) ** 2
        wy_1 = 0.75 - (fy - 1.0) ** 2
        wy_2 = 0.5 * (fy - 0.5) ** 2

        wz_0 = 0.5 * (1.5 - fz) ** 2
        wz_1 = 0.75 - (fz - 1.0) ** 2
        wz_2 = 0.5 * (fz - 0.5) ** 2

        wxs = (wx_0, wx_1, wx_2)
        wys = (wy_0, wy_1, wy_2)
        wzs = (wz_0, wz_1, wz_2)

        new_v_x = 0.0
        new_v_y = 0.0
        new_v_z = 0.0
        c00 = 0.0
        c01 = 0.0
        c02 = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = 0.0
        c20 = 0.0
        c21 = 0.0
        c22 = 0.0

        for i in range(3):
            gx = base_x + i
            if gx < 0 or gx >= nx:
                continue
            wx = wxs[i]
            dpos_x = (gx + 0.5) / inv_dx + origin[0] - xp[0]
            for j in range(3):
                gy = base_y + j
                if gy < 0 or gy >= ny:
                    continue
                wxy = wx * wys[j]
                dpos_y = (gy + 0.5) / inv_dx + origin[1] - xp[1]
                for k in range(3):
                    gz = base_z + k
                    if gz < 0 or gz >= nz:
                        continue
                    weight = wxy * wzs[k]
                    dpos_z = (gz + 0.5) / inv_dx + origin[2] - xp[2]

                    gv_x = grid_v[gx, gy, gz, 0]
                    gv_y = grid_v[gx, gy, gz, 1]
                    gv_z = grid_v[gx, gy, gz, 2]

                    new_v_x += weight * gv_x
                    new_v_y += weight * gv_y
                    new_v_z += weight * gv_z

                    # C_p += weight * gv * dpos^T * inv_D
                    scale = weight * inv_D
                    c00 += scale * gv_x * dpos_x
                    c01 += scale * gv_x * dpos_y
                    c02 += scale * gv_x * dpos_z
                    c10 += scale * gv_y * dpos_x
                    c11 += scale * gv_y * dpos_y
                    c12 += scale * gv_y * dpos_z
                    c20 += scale * gv_z * dpos_x
                    c21 += scale * gv_z * dpos_y
                    c22 += scale * gv_z * dpos_z

        vel[p, 0] = new_v_x
        vel[p, 1] = new_v_y
        vel[p, 2] = new_v_z

        C_mat[p, 0, 0] = c00
        C_mat[p, 0, 1] = c01
        C_mat[p, 0, 2] = c02
        C_mat[p, 1, 0] = c10
        C_mat[p, 1, 1] = c11
        C_mat[p, 1, 2] = c12
        C_mat[p, 2, 0] = c20
        C_mat[p, 2, 1] = c21
        C_mat[p, 2, 2] = c22

        # Advect position
        xp0 = xp[0] + dt * new_v_x
        xp1 = xp[1] + dt * new_v_y
        xp2 = xp[2] + dt * new_v_z

        # Collision with basin bottom
        if xp2 < z_bot:
            xp2 = z_bot
            if vel[p, 2] < 0.0:
                vel[p, 2] = 0.0

        # Collision with basin cylindrical wall (r <= r_basin for z <= z_lip)
        r_sq = xp0 * xp0 + xp1 * xp1
        if xp2 < z_lip and r_sq > (r_basin - 1e-3) ** 2:
            r = np.sqrt(r_sq)
            if r > 1e-6:
                nx_c = xp0 / r
                ny_c = xp1 / r
                xp0 = nx_c * (r_basin - 1e-3)
                xp1 = ny_c * (r_basin - 1e-3)
                vn = vel[p, 0] * nx_c + vel[p, 1] * ny_c
                if vn > 0.0:
                    vel[p, 0] -= vn * nx_c
                    vel[p, 1] -= vn * ny_c

        # Collision with nozzle cylinder
        if xp2 < z_nozzle and r_sq < (r_nozzle + 1e-3) ** 2:
            r = np.sqrt(r_sq)
            if r > 1e-6:
                nx_c = xp0 / r
                ny_c = xp1 / r
                xp0 = nx_c * (r_nozzle + 1e-3)
                xp1 = ny_c * (r_nozzle + 1e-3)
                vn = vel[p, 0] * nx_c + vel[p, 1] * ny_c
                if vn < 0.0:
                    vel[p, 0] -= vn * nx_c
                    vel[p, 1] -= vn * ny_c

        pos[p, 0] = xp0
        pos[p, 1] = xp1
        pos[p, 2] = xp2


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config
        self.ic = self.config["initial_conditions"]

        self.current_time = 0.0

        # Load initial StateFrame
        initial_frame = scene.initial_state()
        self.pos = np.ascontiguousarray(initial_frame.position, dtype=np.float64)
        self.vel = np.ascontiguousarray(initial_frame.velocity, dtype=np.float64)
        self.mass = np.ascontiguousarray(initial_frame.mass, dtype=np.float64)
        self.source_region = np.ascontiguousarray(
            initial_frame.attributes.get("source_region", np.zeros(len(self.pos))),
            dtype=np.float64
        )
        self.C_mat = np.zeros((len(self.pos), 3, 3), dtype=np.float64)

        # Simulation parameters
        self.particle_spacing = float(self.ic["particle_spacing"])
        self.liquid_density = float(self.ic["liquid_density"])
        self.particle_mass = self.liquid_density * (self.particle_spacing ** 3)
        self.jet_speed = float(self.ic["jet_speed"])
        self.nozzle_radius = float(self.ic["nozzle_radius"])
        self.nozzle_exit_z = float(self.ic["nozzle_exit_z"])
        self.pool_radius = float(self.ic["pool_radius"])
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        # Geometry limits
        self.z_bot = 0.04
        self.z_lip = 0.52
        self.r_basin = 1.48  # effective inner radius
        self.r_nozzle = 0.24
        self.z_nozzle = 0.34

        # Cumulative mass injected
        self.cumulative_mass = 0.0
        self.inflow_accumulator = 0.0

        # Simulation grid definition
        self.dx = 0.08
        self.inv_dx = 1.0 / self.dx
        self.origin = np.array([-1.6, -1.6, 0.0], dtype=np.float64)
        self.nx = int(np.round(3.2 / self.dx))
        self.ny = int(np.round(3.2 / self.dx))
        self.nz = int(np.round(3.4 / self.dx))

        # Diagnostic grid definition (for required trajectory fields)
        # Uniform diagnostic-grid spacing
        self.diag_spacing = 0.10
        self.diag_origin = np.array([-1.5, -1.5, 0.0], dtype=np.float64)
        self.diag_nx = int(np.round(3.0 / self.diag_spacing))
        self.diag_ny = int(np.round(3.0 / self.diag_spacing))
        self.diag_nz = int(np.round(3.2 / self.diag_spacing))

        # Boundary samples
        self._init_boundary_samples()

        # Inflow emitter particle templates
        self._init_inflow_template()

        # Internal simulation time step
        self.sim_dt = 0.002

    def _init_boundary_samples(self):
        # Create boundary sample points on basin wall and nozzle
        pts = []
        norms = []

        # Basin floor (circle r <= 1.5, z = 0.04)
        r_samples = np.linspace(0.24, 1.5, 14)
        for r in r_samples:
            n_theta = max(6, int(2 * np.pi * r / 0.12))
            thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            for th in thetas:
                pts.append([r * np.cos(th), r * np.sin(th), 0.04])
                norms.append([0.0, 0.0, 1.0])

        # Basin cylinder wall (r = 1.5, z in [0.04, 0.52])
        z_samples = np.linspace(0.04, 0.52, 6)
        for z in z_samples:
            thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            for th in thetas:
                pts.append([1.5 * np.cos(th), 1.5 * np.sin(th), z])
                norms.append([-np.cos(th), -np.sin(th), 0.0])

        # Nozzle cylinder wall (r = 0.24, z in [0.04, 0.34])
        z_samples_noz = np.linspace(0.04, 0.34, 4)
        for z in z_samples_noz:
            thetas = np.linspace(0, 2 * np.pi, 18, endpoint=False)
            for th in thetas:
                pts.append([0.24 * np.cos(th), 0.24 * np.sin(th), z])
                norms.append([np.cos(th), np.sin(th), 0.0])

        self.boundary_pos = np.array(pts, dtype=np.float64)
        self.boundary_normal = np.array(norms, dtype=np.float64)
        self.boundary_velocity = np.zeros_like(self.boundary_pos)
        self.boundary_target_velocity = np.zeros_like(self.boundary_pos)

    def _init_inflow_template(self):
        # 2D cross section disk of particles for the nozzle jet
        h = self.particle_spacing
        r_noz = self.nozzle_radius
        axes = [
            np.arange(-r_noz, r_noz + 1e-5, h, dtype=np.float64),
            np.arange(-r_noz, r_noz + 1e-5, h, dtype=np.float64)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        pts2d = np.column_stack([grid[0].ravel(), grid[1].ravel()])
        pts2d = pts2d[np.sum(pts2d ** 2, axis=1) <= (r_noz - 0.25 * h) ** 2]
        self.inflow_template_xy = pts2d

    def _inject_particles(self, sub_dt: float):
        # Inflow speed is jet_speed upwards
        # Distance moved in dt is jet_speed * sub_dt
        self.inflow_accumulator += self.jet_speed * sub_dt
        h = self.particle_spacing

        while self.inflow_accumulator >= h:
            self.inflow_accumulator -= h
            n_new = len(self.inflow_template_xy)
            if n_new == 0:
                continue
            new_pos = np.zeros((n_new, 3), dtype=np.float64)
            new_pos[:, :2] = self.inflow_template_xy
            new_pos[:, 2] = self.nozzle_exit_z + self.inflow_accumulator

            # Small perturbation to break symmetry / leading edge noise
            new_pos[:, :2] += np.random.uniform(-0.05 * h, 0.05 * h, size=(n_new, 2))

            new_vel = np.zeros((n_new, 3), dtype=np.float64)
            new_vel[:, 2] = self.jet_speed

            new_mass = np.full(n_new, self.particle_mass, dtype=np.float64)
            new_phase = np.ones(n_new, dtype=np.float64)
            new_C = np.zeros((n_new, 3, 3), dtype=np.float64)

            self.pos = np.vstack((self.pos, new_pos))
            self.vel = np.vstack((self.vel, new_vel))
            self.mass = np.concatenate((self.mass, new_mass))
            self.source_region = np.concatenate((self.source_region, new_phase))
            self.C_mat = np.vstack((self.C_mat, new_C))

            self.cumulative_mass += np.sum(new_mass)

    def _step(self, sub_dt: float):
        # 1. Inject inflow particles
        self._inject_particles(sub_dt)

        # 2. Allocate and clear simulation grid
        grid_m = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        grid_v = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        grid_type = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)

        # 3. P2G (APIC transfer)
        p2g_kernel(
            self.pos, self.vel, self.C_mat, self.mass,
            grid_m, grid_v, self.inv_dx, self.origin,
            self.nx, self.ny, self.nz
        )

        # 4. Grid update (forces + boundary conditions)
        grid_update_kernel(
            grid_m, grid_v, grid_type, sub_dt,
            self.gravity[0], self.gravity[1], self.gravity[2],
            self.r_basin ** 2, self.r_basin, self.z_bot, self.z_lip,
            self.r_nozzle ** 2, self.z_nozzle,
            self.origin, self.dx, self.nx, self.ny, self.nz
        )

        # 5. Pressure projection (Incompressibility solver)
        # Compute divergence
        div = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        # div = 0.5 / dx * (v_x[i+1] - v_x[i-1] + ...)
        vx = grid_v[:, :, :, 0]
        vy = grid_v[:, :, :, 1]
        vz = grid_v[:, :, :, 2]

        # Fast vector divergence
        div[1:-1, :, :] += (vx[2:, :, :] - vx[:-2, :, :]) * (0.5 * self.inv_dx)
        div[:, 1:-1, :] += (vy[:, 2:, :] - vy[:, :-2, :]) * (0.5 * self.inv_dx)
        div[:, :, 1:-1] += (vz[:, :, 2:] - vz[:, :, :-2]) * (0.5 * self.inv_dx)

        # Pressure Poisson Jacobi iterations
        p = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        p_new = np.zeros_like(p)
        # Scale rhs by density/dt * dx^2
        rhs = div * (self.liquid_density / sub_dt * (self.dx ** 2))

        for _ in range(12):
            jacobi_pressure_kernel(p, p_new, grid_type, rhs, self.nx, self.ny, self.nz)
            p, p_new = p_new, p

        # Gradient subtraction
        scale = sub_dt / (self.liquid_density * self.dx)
        apply_pressure_gradient(grid_v, p, grid_type, scale, self.nx, self.ny, self.nz)

        # 6. G2P and particle advection
        g2p_and_advect_kernel(
            self.pos, self.vel, self.C_mat, grid_v, self.inv_dx, sub_dt, self.origin,
            self.nx, self.ny, self.nz,
            self.r_basin, self.z_bot, self.z_lip, self.r_nozzle, self.z_nozzle
        )

    def advance_to(self, target_time: float) -> StateFrame:
        while self.current_time < target_time - 1e-8:
            dt = min(self.sim_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time
        return self._build_state_frame(target_time)

    def _build_state_frame(self, time: float) -> StateFrame:
        # Diagnostic fields on [diag_nx, diag_ny, diag_nz]
        d_dx = self.diag_spacing
        d_orig = self.diag_origin
        nx, ny, nz = self.diag_nx, self.diag_ny, self.diag_nz

        # Velocity grid & level set & active mask
        # We can construct diagnostic velocity and level set from particles
        vel_grid = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        weight_grid = np.zeros((nx, ny, nz), dtype=np.float64)

        # Bin particles to diagnostic grid
        gx = np.clip(np.floor((self.pos[:, 0] - d_orig[0]) / d_dx).astype(np.int32), 0, nx - 1)
        gy = np.clip(np.floor((self.pos[:, 1] - d_orig[1]) / d_dx).astype(np.int32), 0, ny - 1)
        gz = np.clip(np.floor((self.pos[:, 2] - d_orig[2]) / d_dx).astype(np.int32), 0, nz - 1)

        np.add.at(weight_grid, (gx, gy, gz), 1.0)
        for d in range(3):
            np.add.at(vel_grid[:, :, :, d], (gx, gy, gz), self.vel[:, d])

        mask = weight_grid > 0
        for d in range(3):
            vel_grid[mask, d] /= weight_grid[mask]

        active_mask = mask.astype(np.float64)

        # Approximate signed liquid level set: negative inside fluid, positive outside
        # Approximated from particle density / occupancy
        # Inside fluid: level set < 0, outside > 0
        liquid_level_set = np.where(mask, -0.5 * d_dx, d_dx)

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.pos, dtype=np.float64),
            velocity=np.ascontiguousarray(self.vel, dtype=np.float64),
            mass=np.ascontiguousarray(self.mass, dtype=np.float64),
            attributes={
                "source_region": np.ascontiguousarray(self.source_region, dtype=np.float64),
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=np.float64),
            },
            fields={
                "active_mask": active_mask,
                "boundary_normal": np.ascontiguousarray(self.boundary_normal, dtype=np.float64),
                "boundary_target_velocity": np.ascontiguousarray(self.boundary_target_velocity, dtype=np.float64),
                "boundary_velocity": np.ascontiguousarray(self.boundary_velocity, dtype=np.float64),
                "cumulative_mass_source": np.array([self.cumulative_mass], dtype=np.float64),
                "liquid_level_set": liquid_level_set,
                "velocity_field": vel_grid,
            },
            scalars={
                "characteristic_length": 2.0 * self.pool_radius,
                "grid_spacing": float(self.diag_spacing),
                "reference_velocity": float(self.jet_speed),
                "rest_density": float(self.liquid_density),
                "particle_spacing": float(self.particle_spacing),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": len(self.pos),
            "cumulative_mass": self.cumulative_mass,
            "current_time": self.current_time,
        }

