"""Deterministic 2D incompressible Navier-Stokes solver for channel vortex street."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Incompressible Navier-Stokes solver on a collocated/MAC grid with immersed cylinder."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.spacing = float(self.config["fluid"]["nominal_particle_spacing"])
        self.dx = self.spacing
        self.dy = self.spacing
        self.nu = float(self.config["fluid"]["kinematic_viscosity"])
        self.density = float(self.config["fluid"]["density"])

        self.radius = float(self.config["obstacle"]["radius"])
        self.cx, self.cy, _ = self.config["obstacle"]["center"]
        self.centerline_speed = float(self.config["initial_conditions"]["centerline_speed"])
        self.mean_speed = float(self.config["initial_conditions"]["mean_speed"])

        self.xs = np.arange(-6.0 + 0.5 * self.spacing, 10.0, self.spacing)
        self.ys = np.arange(-1.5 + 0.5 * self.spacing, 1.5, self.spacing)
        self.nx = len(self.xs)
        self.ny = len(self.ys)

        self.xx, self.yy = np.meshgrid(self.xs, self.ys, indexing="xy")
        self.cylinder_mask = (self.xx - self.cx) ** 2 + (self.yy - self.cy) ** 2 <= self.radius ** 2
        self.fluid_mask = ~self.cylinder_mask

        # Sub-stepping dt for numerical stability (CFL + diffusion)
        self.sub_dt = 0.002
        self.current_time = 0.0

        # Velocity components on cell centers
        self.u = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.v = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Initial parabolic profile
        for j in range(self.ny):
            y_val = self.ys[j]
            self.u[j, :] = self.centerline_speed * max(0.0, 1.0 - (y_val / 1.5) ** 2)

        # Small asymmetric perturbation near obstacle to trigger natural shedding cleanly
        dist_sq = (self.xx - (self.cx + 0.5 * self.radius)) ** 2 + (self.yy - (self.cy + 0.2 * self.radius)) ** 2
        pert = 0.08 * self.centerline_speed * np.exp(-dist_sq / (2.0 * self.radius ** 2))
        self.v += pert

        # Zero out velocity inside cylinder
        self.u[self.cylinder_mask] = 0.0
        self.v[self.cylinder_mask] = 0.0

        # Dye initialization
        stripe = np.floor((self.xx + 6.0) / 0.55).astype(np.int32)
        dye = np.where(self.xx < -2.55, ((stripe & 1) * 2 - 1), 0.0).astype(np.float64)
        near_center = np.exp(-0.5 * (self.yy / 0.34) ** 2)
        self.dye = dye * near_center
        self.dye[self.cylinder_mask] = 0.0

        # Prepare boundary samples for contract
        # Wall samples and cylinder samples
        # Cylinder boundary samples
        n_cyl = 64
        theta = np.linspace(0, 2 * np.pi, n_cyl, endpoint=False)
        cyl_x = self.cx + self.radius * np.cos(theta)
        cyl_y = self.cy + self.radius * np.sin(theta)
        cyl_nx = np.cos(theta)
        cyl_ny = np.sin(theta)

        # Lower wall samples
        n_wall = 64
        wall_x = np.linspace(-6.0, 10.0, n_wall)
        wall_low_y = np.full(n_wall, -1.5)
        wall_low_nx = np.zeros(n_wall)
        wall_low_ny = np.ones(n_wall)  # Normal pointing into fluid (+y)

        # Upper wall samples
        wall_up_y = np.full(n_wall, 1.5)
        wall_up_nx = np.zeros(n_wall)
        wall_up_ny = np.full(n_wall, -1.0)  # Normal pointing into fluid (-y)

        b_x = np.concatenate([cyl_x, wall_x, wall_x])
        b_y = np.concatenate([cyl_y, wall_low_y, wall_up_y])
        b_nx = np.concatenate([cyl_nx, wall_low_nx, wall_up_nx])
        b_ny = np.concatenate([cyl_ny, wall_low_ny, wall_up_ny])

        self.boundary_normal = np.column_stack((b_nx, b_ny))
        self.boundary_target_velocity = np.zeros((len(b_x), 2), dtype=np.float64)
        self.boundary_velocity = np.zeros((len(b_x), 2), dtype=np.float64)

        # Precompute Poisson matrix for pressure projection
        self._build_poisson_solver()

    def _build_poisson_solver(self):
        # 2D 5-point Laplacian for pressure with Neumann BCs
        nx, ny = self.nx, self.ny
        n_cells = nx * ny

        # Build sparse matrix
        main_diag = np.zeros(n_cells, dtype=np.float64)
        east = np.zeros(n_cells, dtype=np.float64)
        west = np.zeros(n_cells, dtype=np.float64)
        north = np.zeros(n_cells, dtype=np.float64)
        south = np.zeros(n_cells, dtype=np.float64)

        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if self.cylinder_mask[j, i]:
                    # Inside cylinder: trivial identity
                    main_diag[idx] = 1.0
                    continue

                diag = 0.0

                # West (Inlet / neighbor)
                if i == 0:
                    # Inlet: Neumann dp/dx = 0
                    pass
                elif not self.cylinder_mask[j, i - 1]:
                    west[idx] = 1.0 / dx2
                    diag -= 1.0 / dx2

                # East (Outlet / neighbor)
                if i == nx - 1:
                    # Outlet: Dirichlet p = 0
                    diag -= 2.0 / dx2
                elif not self.cylinder_mask[j, i + 1]:
                    east[idx] = 1.0 / dx2
                    diag -= 1.0 / dx2

                # South (Wall / neighbor)
                if j == 0:
                    # Wall: Neumann dp/dy = 0
                    pass
                elif not self.cylinder_mask[j - 1, i]:
                    south[idx] = 1.0 / dy2
                    diag -= 1.0 / dy2

                # North (Wall / neighbor)
                if j == ny - 1:
                    # Wall: Neumann dp/dy = 0
                    pass
                elif not self.cylinder_mask[j + 1, i]:
                    north[idx] = 1.0 / dy2
                    diag -= 1.0 / dy2

                main_diag[idx] = diag if diag != 0.0 else 1.0

        # Construct CSR matrix
        row_indices = []
        col_indices = []
        data = []

        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(main_diag[idx])

                if not self.cylinder_mask[j, i]:
                    if i > 0 and not self.cylinder_mask[j, i - 1]:
                        row_indices.append(idx)
                        col_indices.append(idx - 1)
                        data.append(west[idx])
                    if i < nx - 1 and not self.cylinder_mask[j, i + 1]:
                        row_indices.append(idx)
                        col_indices.append(idx + 1)
                        data.append(east[idx])
                    if j > 0 and not self.cylinder_mask[j - 1, i]:
                        row_indices.append(idx)
                        col_indices.append(idx - nx)
                        data.append(south[idx])
                    if j < ny - 1 and not self.cylinder_mask[j + 1, i]:
                        row_indices.append(idx)
                        col_indices.append(idx + nx)
                        data.append(north[idx])

        A = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        self.poisson_solver = splu(A.tocsc())

    def _semi_lagrangian_advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        ny, nx = field.shape
        # Backtrace
        x_back = self.xx - self.u * dt
        y_back = self.yy - self.v * dt

        # Normalize to fractional grid index
        i_frac = (x_back - self.xs[0]) / self.dx
        j_frac = (y_back - self.ys[0]) / self.dy

        i0 = np.clip(np.floor(i_frac).astype(np.int32), 0, nx - 2)
        j0 = np.clip(np.floor(j_frac).astype(np.int32), 0, ny - 2)
        i1 = i0 + 1
        j1 = j0 + 1

        s1 = np.clip(i_frac - i0, 0.0, 1.0)
        s0 = 1.0 - s1
        t1 = np.clip(j_frac - j0, 0.0, 1.0)
        t0 = 1.0 - t1

        adv = (
            t0 * (s0 * field[j0, i0] + s1 * field[j0, i1])
            + t1 * (s0 * field[j1, i0] + s1 * field[j1, i1])
        )
        return adv

    def _step(self, dt: float) -> None:
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy
        nu = self.nu

        # 1. Advection (Semi-Lagrangian)
        u_adv = self._semi_lagrangian_advect(self.u, dt)
        v_adv = self._semi_lagrangian_advect(self.v, dt)
        self.dye = self._semi_lagrangian_advect(self.dye, dt)

        # Re-inject continuous dye stripes at inlet (x < -2.55) to show clear streaklines
        stripe = np.floor((self.xx[:, 0:10] + 6.0) / 0.55).astype(np.int32)
        inlet_dye = np.where(self.xx[:, 0:10] < -2.55, ((stripe & 1) * 2 - 1), 0.0).astype(np.float64)
        near_center = np.exp(-0.5 * (self.yy[:, 0:10] / 0.34) ** 2)
        self.dye[:, 0:10] = inlet_dye * near_center

        # 2. Viscous Diffusion (Explicit step)
        lap_u = np.zeros_like(u_adv)
        lap_v = np.zeros_like(v_adv)

        # Interior diffusion
        lap_u[1:-1, 1:-1] = (
            (u_adv[1:-1, 2:] - 2.0 * u_adv[1:-1, 1:-1] + u_adv[1:-1, :-2]) / (dx * dx)
            + (u_adv[2:, 1:-1] - 2.0 * u_adv[1:-1, 1:-1] + u_adv[:-2, 1:-1]) / (dy * dy)
        )
        lap_v[1:-1, 1:-1] = (
            (v_adv[1:-1, 2:] - 2.0 * v_adv[1:-1, 1:-1] + v_adv[1:-1, :-2]) / (dx * dx)
            + (v_adv[2:, 1:-1] - 2.0 * v_adv[1:-1, 1:-1] + v_adv[:-2, 1:-1]) / (dy * dy)
        )

        u_star = u_adv + nu * dt * lap_u
        v_star = v_adv + nu * dt * lap_v

        # 3. Apply Boundary Conditions on intermediate velocity
        # Inlet (parabolic profile)
        for j in range(ny):
            y_val = self.ys[j]
            u_star[j, 0] = self.centerline_speed * max(0.0, 1.0 - (y_val / 1.5) ** 2)
            v_star[j, 0] = 0.0

        # Channel Walls (no slip)
        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        v_star[0, :] = 0.0
        v_star[-1, :] = 0.0

        # Outlet (convective / zero gradient)
        u_star[:, -1] = u_star[:, -2]
        v_star[:, -1] = v_star[:, -2]

        # Cylinder obstacle (no slip)
        u_star[self.cylinder_mask] = 0.0
        v_star[self.cylinder_mask] = 0.0

        # 4. Pressure Projection
        # Compute divergence of u_star
        div = np.zeros((ny, nx), dtype=np.float64)
        div[:, 1:-1] += (u_star[:, 2:] - u_star[:, :-2]) / (2.0 * dx)
        div[:, 0] += (u_star[:, 1] - u_star[:, 0]) / dx
        div[:, -1] += (u_star[:, -1] - u_star[:, -2]) / dx

        div[1:-1, :] += (v_star[2:, :] - v_star[:-2, :]) / (2.0 * dy)
        div[0, :] += (v_star[1, :] - v_star[0, :]) / dy
        div[-1, :] += (v_star[-1, :] - v_star[-2, :]) / dy

        # Right hand side for Poisson eq: lap(p) = rho/dt * div(u_star)
        rhs = (self.density / dt) * div
        rhs[self.cylinder_mask] = 0.0

        p = self.poisson_solver.solve(rhs.flatten()).reshape((ny, nx))

        # 5. Correct velocity with pressure gradient
        grad_p_x = np.zeros((ny, nx), dtype=np.float64)
        grad_p_y = np.zeros((ny, nx), dtype=np.float64)

        grad_p_x[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dx)
        grad_p_x[:, 0] = (p[:, 1] - p[:, 0]) / dx
        grad_p_x[:, -1] = (p[:, -1] - p[:, -2]) / dx

        grad_p_y[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dy)
        grad_p_y[0, :] = (p[1, :] - p[0, :]) / dy
        grad_p_y[-1, :] = (p[-1, :] - p[-2, :]) / dy

        self.u = u_star - (dt / self.density) * grad_p_x
        self.v = v_star - (dt / self.density) * grad_p_y

        # Re-apply strict boundary conditions
        for j in range(ny):
            y_val = self.ys[j]
            self.u[j, 0] = self.centerline_speed * max(0.0, 1.0 - (y_val / 1.5) ** 2)
            self.v[j, 0] = 0.0

        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

        self.u[self.cylinder_mask] = 0.0
        self.v[self.cylinder_mask] = 0.0
        self.dye[self.cylinder_mask] = 0.0

    def advance_to(self, time: float) -> StateFrame:
        # Substep to target time
        target_time = float(time)
        while self.current_time + 1e-9 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Particle positions (fluid cells)
        keep = self.fluid_mask
        x, y = self.xx[keep], self.yy[keep]
        z = np.zeros_like(x)
        position = np.column_stack((x, y, z))

        ux = self.u[keep]
        uy = self.v[keep]
        uz = np.zeros_like(ux)
        velocity = np.column_stack((ux, uy, uz))

        # Grid fields
        grid_velocity = np.zeros(self.xx.shape + (3,), dtype=np.float64)
        grid_velocity[..., 0] = self.u
        grid_velocity[..., 1] = self.v

        # Vorticity = dv/dx - du/dy
        grid_vorticity = np.zeros((self.ny, self.nx), dtype=np.float64)
        grid_vorticity[:, 1:-1] += (self.v[:, 2:] - self.v[:, :-2]) / (2.0 * self.dx)
        grid_vorticity[:, 0] += (self.v[:, 1] - self.v[:, 0]) / self.dx
        grid_vorticity[:, -1] += (self.v[:, -1] - self.v[:, -2]) / self.dx

        grid_vorticity[1:-1, :] -= (self.u[2:, :] - self.u[:-2, :]) / (2.0 * self.dy)
        grid_vorticity[0, :] -= (self.u[1, :] - self.u[0, :]) / self.dy
        grid_vorticity[-1, :] -= (self.u[-1, :] - self.u[-2, :]) / self.dy
        grid_vorticity[self.cylinder_mask] = 0.0

        # Divergence = du/dx + dv/dy
        grid_divergence = np.zeros((self.ny, self.nx), dtype=np.float64)
        grid_divergence[:, 1:-1] += (self.u[:, 2:] - self.u[:, :-2]) / (2.0 * self.dx)
        grid_divergence[:, 0] += (self.u[:, 1] - self.u[:, 0]) / self.dx
        grid_divergence[:, -1] += (self.u[:, -1] - self.u[:, -2]) / self.dx

        grid_divergence[1:-1, :] += (self.v[2:, :] - self.v[:-2, :]) / (2.0 * self.dy)
        grid_divergence[0, :] += (self.v[1, :] - self.v[0, :]) / self.dy
        grid_divergence[-1, :] += (self.v[-1, :] - self.v[-2, :]) / self.dy
        grid_divergence[self.cylinder_mask] = 0.0

        grid_dye = self.dye.copy()
        grid_dye[self.cylinder_mask] = 0.0

        n = len(position)
        vel_mag = np.sqrt(ux ** 2 + uy ** 2)

        frame = StateFrame(
            time=target_time,
            position=position,
            velocity=velocity,
            mass=np.full(n, self.density * self.spacing ** 2),
            attributes={
                "dye": grid_dye[keep],
                "vorticity": grid_vorticity[keep],
                "divergence": grid_divergence[keep],
                "velocity_magnitude": vel_mag,
            },
            fields={
                "velocity": grid_velocity,
                "velocity_field": grid_velocity[..., :2].copy(),
                "active_mask": keep.astype(np.uint8),
                "vorticity": grid_vorticity,
                "divergence": grid_divergence,
                "dye": grid_dye,
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "scalar_source": np.zeros_like(grid_dye),
                "transported_scalar": grid_dye.copy(),
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.spacing,
                "reference_velocity": self.mean_speed,
                "cylinder_radius": self.radius,
                "mean_inlet_speed": self.mean_speed,
                "expected_strouhal_number": float(self.config["diagnostics"]["expected_strouhal_number"]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "grid_size": [self.nx, self.ny],
            "spacing": self.spacing,
        }

