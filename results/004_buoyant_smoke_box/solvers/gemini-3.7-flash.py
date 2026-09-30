"""Buoyant smoke box solver using 3D incompressible Navier-Stokes with Boussinesq buoyancy."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 40615
        self.time = 0.0

        # Domain parameters
        self.shape = (32, 32, 64)
        self.bounds_min = np.array([-0.75, -0.75, 0.0], dtype=np.float64)
        self.bounds_max = np.array([0.75, 0.75, 3.0], dtype=np.float64)
        self.domain_size = self.bounds_max - self.bounds_min
        self.dx = float((self.bounds_max[0] - self.bounds_min[0]) / (self.shape[0] - 1))

        # Grid coordinates
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], self.shape[0])
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], self.shape[1])
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], self.shape[2])
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Physical fields
        self.u = np.zeros(self.shape, dtype=np.float64)
        self.v = np.zeros(self.shape, dtype=np.float64)
        self.w = np.zeros(self.shape, dtype=np.float64)
        self.density = np.zeros(self.shape, dtype=np.float64)
        self.temperature = np.zeros(self.shape, dtype=np.float64)
        self.source_field = np.zeros(self.shape, dtype=np.float64)

        # Passive markers
        self.marker_pos = np.empty((0, 3), dtype=np.float64)
        self.marker_vel = np.empty((0, 3), dtype=np.float64)
        self.marker_count = 0

        # Boundary specifications
        self.boundary_normals = None
        self.boundary_vel = None
        self.boundary_target_vel = None

        # Physical parameters
        self.buoyancy_alpha = 4.5  # thermal expansion buoyancy
        self.buoyancy_beta = 0.8   # smoke density buoyancy
        self.diffusion_rate = 0.0005
        self.vorticity_confinement_eps = 0.08
        self.sub_dt = 0.005
        self.reference_vel = 1.0
        self.char_length = float(self.domain_size[2])

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.time = 0.0

        config = scene.config
        self.shape = tuple(int(v) for v in config["field_layout"]["grid_shape"])
        self.bounds_min = np.asarray(config["domain"]["bounds_min"], dtype=np.float64)
        self.bounds_max = np.asarray(config["domain"]["bounds_max"], dtype=np.float64)
        self.domain_size = self.bounds_max - self.bounds_min
        self.dx = float((self.bounds_max[0] - self.bounds_min[0]) / (self.shape[0] - 1))

        # Grid mesh
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], self.shape[0])
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], self.shape[1])
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], self.shape[2])
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Setup source field
        r2 = self.X * self.X + self.Y * self.Y
        src_r = float(config["initial_conditions"]["source_radius"])
        src_h = float(config["initial_conditions"]["source_height"])

        # Source Gaussian emitter at the bottom
        self.source_field = np.exp(-r2 / (2.0 * (src_r * 0.7) ** 2)) * np.exp(-((self.Z - 0.12) ** 2) / (2.0 * (src_h * 0.5) ** 2))
        self.source_field[self.Z > 0.35] = 0.0

        # Initial fields from scene
        init_frame = scene.initial_state()
        self.density = np.asarray(init_frame.fields["smoke_density"], dtype=np.float64).copy()
        if "temperature" in init_frame.fields:
            self.temperature = np.asarray(init_frame.fields["temperature"], dtype=np.float64).copy()
        else:
            self.temperature = self.density.copy()

        vel_f = np.asarray(init_frame.fields["velocity_field"], dtype=np.float64)
        self.u = vel_f[..., 0].copy()
        self.v = vel_f[..., 1].copy()
        self.w = vel_f[..., 2].copy()

        self.marker_pos = np.asarray(init_frame.position, dtype=np.float64).copy()
        self.marker_vel = np.asarray(init_frame.velocity, dtype=np.float64).copy()
        self.marker_count = len(self.marker_pos)

        # Build boundary normal samples for the 6 faces of the closed box
        self._setup_boundary_samples()

    def _setup_boundary_samples(self) -> None:
        # Generate sample points on the 6 domain faces
        normals = []
        nx, ny, nz = self.shape
        # -X face
        normals.extend([[-1.0, 0.0, 0.0]] * (ny * nz))
        # +X face
        normals.extend([[1.0, 0.0, 0.0]] * (ny * nz))
        # -Y face
        normals.extend([[0.0, -1.0, 0.0]] * (nx * nz))
        # +Y face
        normals.extend([[0.0, 1.0, 0.0]] * (nx * nz))
        # -Z face
        normals.extend([[0.0, 0.0, -1.0]] * (nx * ny))
        # +Z face
        normals.extend([[0.0, 0.0, 1.0]] * (nx * ny))

        self.boundary_normals = np.asarray(normals, dtype=np.float64)
        self.boundary_vel = np.zeros_like(self.boundary_normals)
        self.boundary_target_vel = np.zeros_like(self.boundary_normals)

    def _trilinear_interpolate(self, field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        # pts in world coords: (N, 3)
        # normalize to grid indices: [0, nx-1], [0, ny-1], [0, nz-1]
        nx, ny, nz = self.shape
        gx = (pts[:, 0] - self.bounds_min[0]) / (self.bounds_max[0] - self.bounds_min[0]) * (nx - 1)
        gy = (pts[:, 1] - self.bounds_min[1]) / (self.bounds_max[1] - self.bounds_min[1]) * (ny - 1)
        gz = (pts[:, 2] - self.bounds_min[2]) / (self.bounds_max[2] - self.bounds_min[2]) * (nz - 1)

        gx = np.clip(gx, 0.0, nx - 1.0001)
        gy = np.clip(gy, 0.0, ny - 1.0001)
        gz = np.clip(gz, 0.0, nz - 1.0001)

        i0 = np.floor(gx).astype(np.int64)
        j0 = np.floor(gy).astype(np.int64)
        k0 = np.floor(gz).astype(np.int64)

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        tx = (gx - i0)
        ty = (gy - j0)
        tz = (gz - k0)

        c000 = field[i0, j0, k0]
        c100 = field[i1, j0, k0]
        c010 = field[i0, j1, k0]
        c110 = field[i1, j1, k0]
        c001 = field[i0, j0, k1]
        c101 = field[i1, j0, k1]
        c011 = field[i0, j1, k1]
        c111 = field[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz

    def _advect_field(self, field: np.ndarray, dt: float) -> np.ndarray:
        # Semi-Lagrangian backtrace on grid
        nx, ny, nz = self.shape
        x_back = self.X - dt * self.u
        y_back = self.Y - dt * self.v
        z_back = self.Z - dt * self.w

        pts_back = np.column_stack((x_back.ravel(), y_back.ravel(), z_back.ravel()))
        res = self._trilinear_interpolate(field, pts_back)
        return res.reshape(self.shape)

    def _apply_boundary_conditions(self) -> None:
        # Closed box no-penetration (Dirichlet zero normal velocity, free-slip tangential)
        self.u[0, :, :] = 0.0
        self.u[-1, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    def _project_pressure_fft(self) -> tuple[float, float]:
        # Approximate divergence-free projection using Discrete Cosine / Poisson solver with Neumann BCs
        # Divergence: div u = du/dx + dv/dy + dw/dz
        nx, ny, nz = self.shape
        dx = self.dx

        # Central difference for internal nodes, one-sided at boundaries
        div = np.zeros_like(self.u)
        div[1:-1, :, :] += (self.u[2:, :, :] - self.u[:-2, :, :]) / (2.0 * dx)
        div[:, 1:-1, :] += (self.v[:, 2:, :] - self.v[:, :-2, :]) / (2.0 * dx)
        div[:, :, 1:-1] += (self.w[:, :, 2:] - self.w[:, :, :-2]) / (2.0 * dx)

        div[0, :, :] = (self.u[1, :, :] - self.u[0, :, :]) / dx
        div[-1, :, :] = (self.u[-1, :, :] - self.u[-2, :, :]) / dx
        div[:, 0, :] += (self.v[:, 1, :] - self.v[:, 0, :]) / dx
        div[:, -1, :] += (self.v[:, -1, :] - self.v[:, -2, :]) / dx
        div[:, :, 0] += (self.w[:, :, 1] - self.w[:, :, 0]) / dx
        div[:, :, -1] += (self.w[:, :, -1] - self.w[:, :, -2]) / dx

        # Solve Poisson equation nabla^2 p = div using DCT (Type 1 or 2)
        # Using scipy.fft if available or fast standard red-black Gauss-Seidel / Jacobi / DCT
        try:
            from scipy.fft import dctn, idctn
            # 3D DCT-2 for Neumann boundary conditions
            div_dct = dctn(div, type=2, norm='ortho')

            # Eigenvalues of 1D discrete Laplacian with Neumann BC
            kx = np.arange(nx)
            ky = np.arange(ny)
            kz = np.arange(nz)

            lam_x = 2.0 * (np.cos(np.pi * kx / nx) - 1.0) / (dx * dx)
            lam_y = 2.0 * (np.cos(np.pi * ky / ny) - 1.0) / (dx * dx)
            lam_z = 2.0 * (np.cos(np.pi * kz / nz) - 1.0) / (dx * dx)

            denom = lam_x[:, None, None] + lam_y[None, :, None] + lam_z[None, None, :]
            denom[0, 0, 0] = 1.0  # Gauge fixing

            p_dct = div_dct / denom
            p_dct[0, 0, 0] = 0.0

            p = idctn(p_dct, type=2, norm='ortho')
        except ImportError:
            # Fallback Gauss-Seidel relaxation
            p = np.zeros_like(div)
            factor = (dx * dx) / 6.0
            for _ in range(40):
                p_old = p.copy()
                p[1:-1, 1:-1, 1:-1] = (
                    p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                    p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                    p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] -
                    div[1:-1, 1:-1, 1:-1] * (dx * dx)
                ) / 6.0
                # Neumann BC on p
                p[0, :, :] = p[1, :, :]
                p[-1, :, :] = p[-2, :, :]
                p[:, 0, :] = p[:, 1, :]
                p[:, -1, :] = p[:, -2, :]
                p[:, :, 0] = p[:, :, 1]
                p[:, :, -1] = p[:, :, -2]

        # Subtract gradient of p from velocity
        grad_px = np.zeros_like(p)
        grad_py = np.zeros_like(p)
        grad_pz = np.zeros_like(p)

        grad_px[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
        grad_px[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx
        grad_px[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / dx

        grad_py[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dx)
        grad_py[:, 0, :] = (p[:, 1, :] - p[:, 0, :]) / dx
        grad_py[:, -1, :] = (p[:, -1, :] - p[:, -2, :]) / dx

        grad_pz[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dx)
        grad_pz[:, :, 0] = (p[:, :, 1] - p[:, :, 0]) / dx
        grad_pz[:, :, -1] = (p[:, :, -1] - p[:, :, -2]) / dx

        self.u -= grad_px
        self.v -= grad_py
        self.w -= grad_pz

        self._apply_boundary_conditions()

        # Calculate residual divergence
        div_post = np.zeros_like(self.u)
        div_post[1:-1, :, :] += (self.u[2:, :, :] - self.u[:-2, :, :]) / (2.0 * dx)
        div_post[:, 1:-1, :] += (self.v[:, 2:, :] - self.v[:, :-2, :]) / (2.0 * dx)
        div_post[:, :, 1:-1] += (self.w[:, :, 2:] - self.w[:, :, :-2]) / (2.0 * dx)
        div_post[0, :, :] = (self.u[1, :, :] - self.u[0, :, :]) / dx
        div_post[-1, :, :] = (self.u[-1, :, :] - self.u[-2, :, :]) / dx
        div_post[:, 0, :] += (self.v[:, 1, :] - self.v[:, 0, :]) / dx
        div_post[:, -1, :] += (self.v[:, -1, :] - self.v[:, -2, :]) / dx
        div_post[:, :, 0] += (self.w[:, :, 1] - self.w[:, :, 0]) / dx
        div_post[:, :, -1] += (self.w[:, :, -1] - self.w[:, :, -2]) / dx

        div_l2 = float(np.sqrt(np.mean(div_post**2)))
        div_max = float(np.max(np.abs(div_post)))
        return div_l2, div_max

    def _vorticity_confinement(self, dt: float) -> None:
        dx = self.dx
        # Compute curl of velocity (omega)
        # omega_x = dw/dy - dv/dz
        # omega_y = du/dz - dw/dx
        # omega_z = dv/dx - du/dy
        dw_dy = np.zeros_like(self.w)
        dw_dy[:, 1:-1, :] = (self.w[:, 2:, :] - self.w[:, :-2, :]) / (2.0 * dx)
        dv_dz = np.zeros_like(self.v)
        dv_dz[:, :, 1:-1] = (self.v[:, :, 2:] - self.v[:, :, :-2]) / (2.0 * dx)
        omega_x = dw_dy - dv_dz

        du_dz = np.zeros_like(self.u)
        du_dz[:, :, 1:-1] = (self.u[:, :, 2:] - self.u[:, :, :-2]) / (2.0 * dx)
        dw_dx = np.zeros_like(self.w)
        dw_dx[1:-1, :, :] = (self.w[2:, :, :] - self.w[:-2, :, :]) / (2.0 * dx)
        omega_y = du_dz - dw_dx

        dv_dx = np.zeros_like(self.v)
        dv_dx[1:-1, :, :] = (self.v[2:, :, :] - self.v[:-2, :, :]) / (2.0 * dx)
        du_dy = np.zeros_like(self.u)
        du_dy[:, 1:-1, :] = (self.u[:, 2:, :] - self.u[:, :-2, :]) / (2.0 * dx)
        omega_z = dv_dx - du_dy

        mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2) + 1e-6

        # Gradient of magnitude
        grad_mx = np.zeros_like(mag)
        grad_my = np.zeros_like(mag)
        grad_mz = np.zeros_like(mag)
        grad_mx[1:-1, :, :] = (mag[2:, :, :] - mag[:-2, :, :]) / (2.0 * dx)
        grad_my[:, 1:-1, :] = (mag[:, 2:, :] - mag[:, :-2, :]) / (2.0 * dx)
        grad_mz[:, :, 1:-1] = (mag[:, :, 2:] - mag[:, :, :-2]) / (2.0 * dx)

        norm_grad = np.sqrt(grad_mx**2 + grad_my**2 + grad_mz**2) + 1e-6
        Nx = grad_mx / norm_grad
        Ny = grad_my / norm_grad
        Nz = grad_mz / norm_grad

        # Confinement force: F_conf = epsilon * dx * (N x omega)
        fx = self.vorticity_confinement_eps * dx * (Ny * omega_z - Nz * omega_y)
        fy = self.vorticity_confinement_eps * dx * (Nz * omega_x - Nx * omega_z)
        fz = self.vorticity_confinement_eps * dx * (Nx * omega_y - Ny * omega_x)

        self.u += dt * fx
        self.v += dt * fy
        self.w += dt * fz

    def _step(self, dt: float) -> tuple[float, float]:
        # 1. Add continuous heat/smoke source
        # Add source injection
        source_rate = 1.2
        self.temperature += dt * source_rate * self.source_field
        self.density += dt * source_rate * self.source_field

        # 2. Add buoyancy forces (Boussinesq approximation)
        # F_buoyant = (alpha * T - beta * density) * z_hat
        f_buoy = self.buoyancy_alpha * self.temperature + self.buoyancy_beta * self.density
        self.w += dt * f_buoy

        # 3. Vorticity confinement for sharp plume vortices
        self._vorticity_confinement(dt)

        # 4. Advect velocity field (semi-Lagrangian)
        u_new = self._advect_field(self.u, dt)
        v_new = self._advect_field(self.v, dt)
        w_new = self._advect_field(self.w, dt)
        self.u, self.v, self.w = u_new, v_new, w_new

        # 5. Advect scalar fields
        self.density = self._advect_field(self.density, dt)
        self.temperature = self._advect_field(self.temperature, dt)

        # 6. Apply boundary conditions & pressure projection
        self._apply_boundary_conditions()
        div_l2, div_max = self._project_pressure_fft()

        # 7. Advect passive markers using Runge-Kutta 2 (midpoint)
        if self.marker_count > 0:
            # Velocity at current positions
            u_p = self._trilinear_interpolate(self.u, self.marker_pos)
            v_p = self._trilinear_interpolate(self.v, self.marker_pos)
            w_p = self._trilinear_interpolate(self.w, self.marker_pos)
            vel_p = np.column_stack((u_p, v_p, w_p))

            # Midpoint position
            pos_mid = self.marker_pos + 0.5 * dt * vel_p
            # Clamp to domain bounds
            pos_mid = np.clip(pos_mid, self.bounds_min + 1e-4, self.bounds_max - 1e-4)

            # Velocity at midpoint
            u_mid = self._trilinear_interpolate(self.u, pos_mid)
            v_mid = self._trilinear_interpolate(self.v, pos_mid)
            w_mid = self._trilinear_interpolate(self.w, pos_mid)
            vel_mid = np.column_stack((u_mid, v_mid, w_mid))

            # Update markers
            self.marker_pos += dt * vel_mid
            self.marker_pos = np.clip(self.marker_pos, self.bounds_min + 1e-4, self.bounds_max - 1e-4)
            self.marker_vel = vel_mid

        return div_l2, div_max

    def advance_to(self, target_time: float) -> StateFrame:
        div_l2 = 0.0
        div_max = 0.0

        while self.time + 1e-8 < target_time:
            dt = min(self.sub_dt, target_time - self.time)
            if dt <= 1e-8:
                break
            div_l2, div_max = self._step(dt)
            self.time += dt

        self.time = target_time

        # Package velocity field
        velocity_grid = np.stack((self.u, self.v, self.w), axis=-1)

        # Active mask (all interior domain cells)
        active_mask = np.ones(self.shape, dtype=bool)

        frame = StateFrame(
            time=float(self.time),
            position=self.marker_pos.copy(),
            velocity=self.marker_vel.copy(),
            mass=np.full(self.marker_count, 1.0 / max(self.marker_count, 1), dtype=np.float64),
            attributes={"marker_kind": np.ones(self.marker_count, dtype=np.int32)},
            fields={
                "smoke_density": self.density.astype(np.float32),
                "velocity_field": velocity_grid.astype(np.float64),
                "active_mask": active_mask,
                "scalar_source": self.source_field.astype(np.float64),
                "transported_scalar": self.density.astype(np.float64),
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_velocity": self.boundary_vel.copy(),
                "boundary_target_velocity": self.boundary_target_vel.copy(),
            },
            scalars={
                "divergence_l2": float(div_l2),
                "divergence_max": float(div_max),
                "relative_volume_error": 0.0,
                "characteristic_length": float(self.char_length),
                "grid_spacing": float(self.dx),
                "reference_velocity": float(self.reference_vel),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "max_velocity": float(np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))),
            "marker_count": self.marker_count,
        }

