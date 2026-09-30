"""Rising Smoke Fluid Solver using Stable Fluids (Fedkiw et al. 2001 / Stam 1999).

Physical simulation of buoyant smoke rising in a 3D domain with:
- Semi-Lagrangian advection (velocity and scalar/temperature)
- Buoyancy forcing (Boussinesq approximation)
- Vorticity confinement to preserve turbulent structures and persistent rolling vortices
- Pressure projection (incompressibility / divergence-free velocity field)
- Boundary condition enforcement (free-slip walls / open outflow boundaries)
- Scalar dissipation (density half-life decay)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        domain = self.config["domain"]
        self.grid_shape = tuple(int(v) for v in domain["grid_shape"])  # (nx, ny, nz) = (100, 40, 100)
        self.bounds_min = np.asarray(domain["bounds_min"], dtype=np.float64)  # [-3.0, -1.2, 0.0]
        self.bounds_max = np.asarray(domain["bounds_max"], dtype=np.float64)  # [3.0, 1.2, 6.0]
        self.nx, self.ny, self.nz = self.grid_shape

        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz
        self.h = (self.dx + self.dy + self.dz) / 3.0  # average grid spacing
        self.inv_dx2 = 1.0 / (self.dx * self.dx)
        self.inv_dy2 = 1.0 / (self.dy * self.dy)
        self.inv_dz2 = 1.0 / (self.dz * self.dz)
        self.denom = 2.0 * (self.inv_dx2 + self.inv_dy2 + self.inv_dz2)

        # Precompute coordinate grids for cell centers (in index space)
        i = np.arange(self.nx, dtype=np.float64)
        j = np.arange(self.ny, dtype=np.float64)
        k = np.arange(self.nz, dtype=np.float64)
        self.I, self.J, self.K = np.meshgrid(i, j, k, indexing="ij")

        # Physical coordinates
        self.X = self.bounds_min[0] + (self.I + 0.5) * self.dx
        self.Y = self.bounds_min[1] + (self.J + 0.5) * self.dy
        self.Z = self.bounds_min[2] + (self.K + 0.5) * self.dz

        # Initial conditions
        initial_frame = scene.initial_state()
        self.density = np.copy(initial_frame.fields["density"])  # [nx, ny, nz]
        self.velocity = np.copy(initial_frame.fields["velocity"])  # [nx, ny, nz, 3]

        # Temperature excess field
        initial_cfg = self.config["initial_conditions"]
        source_cfg = initial_cfg["smoke_source"]
        peak_dens = float(source_cfg["peak_density"])
        if peak_dens > 0:
            self.temperature = float(source_cfg["temperature_excess"]) * (self.density / peak_dens)
        else:
            self.temperature = np.zeros_like(self.density)

        # Source template for continuous emission or diagnostics
        center = np.asarray(source_cfg["center"], dtype=np.float64)
        rad2 = ((self.X - center[0]) ** 2 + (self.Y - center[1]) ** 2) / float(source_cfg["radius"]) ** 2
        vert2 = (self.Z - center[2]) ** 2 / float(source_cfg["half_height"]) ** 2
        self.source_density_profile = float(source_cfg["peak_density"]) * np.exp(-2.5 * (rad2 + vert2))
        self.source_density_profile[self.source_density_profile < 1.0e-4] = 0.0
        self.source_velocity = np.asarray(source_cfg["initial_velocity"], dtype=np.float64)

        # Physical constants / parameters
        self.half_life = float(initial_cfg.get("density_half_life", 6.0))
        self.decay_rate = np.log(2.0) / self.half_life if self.half_life > 0 else 0.0

        # Buoyancy parameters (Fedkiw et al. 2001: f_buoy = (-alpha * density + beta * (T - T_amb)) * z_hat)
        # In our case, smoke is hot and rises: alpha = 0.1, beta = 2.5
        self.alpha = 0.05
        self.beta = 1.5
        self.vorticity_epsilon = 0.25  # Vorticity confinement strength

        # Active mask (all interior fluid cells)
        self.active_mask = np.ones(self.grid_shape, dtype=np.float64)

        # Mass tracking
        cell_vol = self.dx * self.dy * self.dz
        self.initial_mass = float(np.sum(self.density) * cell_vol)
        self.cumulative_mass_source = 0.0

        # Current time
        self.current_time = 0.0
        self.center_position = center.copy()
        self.representative_velocity = self.source_velocity.copy()

    def _advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection for scalar or vector field."""
        # Convert velocity (m/s) to index displacement (cells/s)
        u_idx = self.velocity[..., 0] / self.dx
        v_idx = self.velocity[..., 1] / self.dy
        w_idx = self.velocity[..., 2] / self.dz

        # Backtrace positions in index coordinates
        x_back = self.I - dt * u_idx
        y_back = self.J - dt * v_idx
        z_back = self.K - dt * w_idx

        # Clip coords to valid domain indices for interpolation
        coords = np.array([
            np.clip(x_back, 0.0, self.nx - 1.0001),
            np.clip(y_back, 0.0, self.ny - 1.0001),
            np.clip(z_back, 0.0, self.nz - 1.0001)
        ])

        if field.ndim == 3:
            return map_coordinates(field, coords, order=1, mode='nearest')
        elif field.ndim == 4:
            out = np.empty_like(field)
            for c in range(field.shape[3]):
                out[..., c] = map_coordinates(field[..., c], coords, order=1, mode='nearest')
            return out
        else:
            raise ValueError(f"Unsupported field ndim: {field.ndim}")

    def _apply_boundary_conditions(self, vel: np.ndarray) -> None:
        """Enforce domain boundary conditions on velocity."""
        # x_min: closed free-slip (u=0)
        vel[0, :, :, 0] = 0.0
        # x_max: open outflow (zero normal gradient, clamp inflow)
        vel[-1, :, :, 0] = np.maximum(vel[-2, :, :, 0], 0.0)
        vel[-1, :, :, 1] = vel[-2, :, :, 1]
        vel[-1, :, :, 2] = vel[-2, :, :, 2]

        # y_min, y_max: closed free-slip (v=0)
        vel[:, 0, :, 1] = 0.0
        vel[:, -1, :, 1] = 0.0

        # z_min: closed free-slip floor (w=0)
        vel[:, :, 0, 2] = 0.0
        # z_max: open outflow ceiling (zero normal gradient, clamp inflow)
        vel[:, :, -1, 2] = np.maximum(vel[:, :, -2, 2], 0.0)
        vel[:, :, -1, 0] = vel[:, :, -2, 0]
        vel[:, :, -1, 1] = vel[:, :, -2, 1]

    def _compute_vorticity_confinement(self, dt: float) -> np.ndarray:
        """Compute vorticity confinement force (Fedkiw et al. 2001)."""
        u = self.velocity[..., 0]
        v = self.velocity[..., 1]
        w = self.velocity[..., 2]

        # Curl of velocity: omega = nabla x u
        # dw/dy - dv/dz
        dw_dy = np.gradient(w, self.dy, axis=1)
        dv_dz = np.gradient(v, self.dz, axis=2)
        omega_x = dw_dy - dv_dz

        # du/dz - dw/dx
        du_dz = np.gradient(u, self.dz, axis=2)
        dw_dx = np.gradient(w, self.dx, axis=0)
        omega_y = du_dz - dw_dx

        # dv/dx - du/dy
        dv_dx = np.gradient(v, self.dx, axis=0)
        du_dy = np.gradient(u, self.dy, axis=1)
        omega_z = dv_dx - du_dy

        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2) + 1.0e-6

        # Gradient of vorticity magnitude: eta = nabla |omega|
        eta_x = np.gradient(omega_mag, self.dx, axis=0)
        eta_y = np.gradient(omega_mag, self.dy, axis=1)
        eta_z = np.gradient(omega_mag, self.dz, axis=2)

        eta_mag = np.sqrt(eta_x**2 + eta_y**2 + eta_z**2) + 1.0e-6
        nx_vec = eta_x / eta_mag
        ny_vec = eta_y / eta_mag
        nz_vec = eta_z / eta_mag

        # f_vc = epsilon * h * (N x omega)
        f_x = self.vorticity_epsilon * self.h * (ny_vec * omega_z - nz_vec * omega_y)
        f_y = self.vorticity_epsilon * self.h * (nz_vec * omega_x - nx_vec * omega_z)
        f_z = self.vorticity_epsilon * self.h * (nx_vec * omega_y - ny_vec * omega_x)

        return np.stack([f_x, f_y, f_z], axis=-1)

    def _project(self, vel: np.ndarray, iterations: int = 35) -> np.ndarray:
        """Pressure projection step using Red-Black Gauss-Seidel Poisson solver."""
        u = vel[..., 0]
        v = vel[..., 1]
        w = vel[..., 2]

        # Divergence of velocity
        div = (
            (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * self.dx) +
            (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * self.dy) +
            (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * self.dz)
        )

        p = np.zeros_like(div)

        # Red-Black masks
        grid_sum = (self.I + self.J + self.K).astype(np.int32)
        red_mask = (grid_sum % 2 == 0)
        black_mask = ~red_mask

        # Interior mask (excluding 1-cell boundary for Dirichlet/Neumann)
        interior = np.zeros_like(div, dtype=bool)
        interior[1:-1, 1:-1, 1:-1] = True

        red_interior = red_mask & interior
        black_interior = black_mask & interior

        inv_dx2 = self.inv_dx2
        inv_dy2 = self.inv_dy2
        inv_dz2 = self.inv_dz2
        denom = self.denom

        for _ in range(iterations):
            # Red pass
            p_xp = np.roll(p, -1, axis=0)
            p_xm = np.roll(p, 1, axis=0)
            p_yp = np.roll(p, -1, axis=1)
            p_ym = np.roll(p, 1, axis=1)
            p_zp = np.roll(p, -1, axis=2)
            p_zm = np.roll(p, 1, axis=2)

            lap_neighbors = (
                (p_xp + p_xm) * inv_dx2 +
                (p_yp + p_ym) * inv_dy2 +
                (p_zp + p_zm) * inv_dz2
            )
            p[red_interior] = (lap_neighbors[red_interior] - div[red_interior]) / denom

            # Boundary conditions on pressure (Neumann on solid walls, zero on open boundaries)
            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = 0.0  # open outflow
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = 0.0  # open top

            # Black pass
            p_xp = np.roll(p, -1, axis=0)
            p_xm = np.roll(p, 1, axis=0)
            p_yp = np.roll(p, -1, axis=1)
            p_ym = np.roll(p, 1, axis=1)
            p_zp = np.roll(p, -1, axis=2)
            p_zm = np.roll(p, 1, axis=2)

            lap_neighbors = (
                (p_xp + p_xm) * inv_dx2 +
                (p_yp + p_ym) * inv_dy2 +
                (p_zp + p_zm) * inv_dz2
            )
            p[black_interior] = (lap_neighbors[black_interior] - div[black_interior]) / denom

            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = 0.0
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = 0.0

        # Subtract pressure gradient from velocity: u = u - grad(p)
        dp_dx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2.0 * self.dx)
        dp_dy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * self.dy)
        dp_dz = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2.0 * self.dz)

        vel_proj = np.empty_like(vel)
        vel_proj[..., 0] = vel[..., 0] - dp_dx
        vel_proj[..., 1] = vel[..., 1] - dp_dy
        vel_proj[..., 2] = vel[..., 2] - dp_dz

        self._apply_boundary_conditions(vel_proj)
        return vel_proj

    def _step(self, dt: float) -> None:
        """Advance the physics simulation by time step dt."""
        # 1. Source injection at the emitter base
        envelope = np.clip(self.source_density_profile / float(self.config["initial_conditions"]["smoke_source"]["peak_density"]), 0.0, 1.0)
        injection_rate = 1.0  # continuous replenishment
        self.density = np.maximum(self.density, self.source_density_profile * (1.0 - np.exp(-dt * 5.0)))
        self.temperature = np.maximum(self.temperature, envelope * float(self.config["initial_conditions"]["smoke_source"]["temperature_excess"]))

        # Add source velocity injection
        self.velocity += envelope[..., None] * self.source_velocity * (dt * 2.0)

        # 2. Vorticity Confinement
        f_vc = self._compute_vorticity_confinement(dt)
        self.velocity += dt * f_vc

        # 3. Buoyancy force (Boussinesq approximation)
        # Upward force proportional to temperature excess minus density loading
        f_buoy = -self.alpha * self.density + self.beta * self.temperature
        self.velocity[..., 2] += dt * f_buoy

        # 4. Pressure Projection (divergence-free velocity)
        self._apply_boundary_conditions(self.velocity)
        self.velocity = self._project(self.velocity, iterations=30)

        # 5. Advect Velocity
        self.velocity = self._advect(self.velocity, dt)
        self._apply_boundary_conditions(self.velocity)
        self.velocity = self._project(self.velocity, iterations=25)

        # 6. Advect Scalars (density and temperature)
        self.density = self._advect(self.density, dt)
        self.temperature = self._advect(self.temperature, dt)

        # 7. Density dissipation / decay
        if self.decay_rate > 0:
            decay_factor = np.exp(-self.decay_rate * dt)
            self.density *= decay_factor
            self.temperature *= decay_factor

        self.density[self.density < 1.0e-5] = 0.0
        self.temperature[self.temperature < 1.0e-5] = 0.0

        # Update representative point position for StateFrame tracking
        dens_sum = np.sum(self.density)
        if dens_sum > 1.0e-6:
            cx = np.sum(self.X * self.density) / dens_sum
            cy = np.sum(self.Y * self.density) / dens_sum
            cz = np.sum(self.Z * self.density) / dens_sum
            self.center_position = np.array([cx, cy, cz])
            vx = np.sum(self.velocity[..., 0] * self.density) / dens_sum
            vy = np.sum(self.velocity[..., 1] * self.density) / dens_sum
            vz = np.sum(self.velocity[..., 2] * self.density) / dens_sum
            self.representative_velocity = np.array([vx, vy, vz])

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        """Advance solver state to the requested sample time."""
        target_time = float(time)

        # Sub-stepping for numerical stability
        max_substep_dt = 0.02
        while self.current_time < target_time - 1.0e-8:
            dt = min(max_substep_dt, target_time - self.current_time)
            self._step(dt)

        self.current_time = target_time

        # Calculate current total mass
        cell_vol = self.dx * self.dy * self.dz
        current_mass = float(np.sum(self.density) * cell_vol)
        mass_quantity = np.asarray([current_mass], dtype=np.float64)

        # Cumulative mass source: added - removed
        cumulative_mass = np.asarray([current_mass - self.initial_mass], dtype=np.float64)

        # Scalar source grid
        scalar_source = np.zeros(self.grid_shape, dtype=np.float64)
        scalar_source += self.source_density_profile

        # Characteristic length and grid spacing
        domain = self.config["domain"]
        char_len = float(self.bounds_max[2] - self.bounds_min[2])  # 6.0 m height
        grid_sp = float(self.h)

        frame = StateFrame(
            time=float(target_time),
            position=self.center_position.reshape(1, 3).copy(),
            velocity=self.representative_velocity.reshape(1, 3).copy(),
            mass=mass_quantity,
            attributes={
                "mass_quantity": mass_quantity,
                "density": np.asarray([float(np.max(self.density))]),
                "temperature_excess": np.asarray([float(np.max(self.temperature))])
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "cumulative_mass_source": cumulative_mass.copy(),
                "scalar_source": scalar_source.copy(),
                "transported_scalar": self.density.copy(),
                "velocity_field": self.velocity.copy(),
                "density": self.density.copy(),
                "velocity": self.velocity.copy()
            },
            scalars={
                "characteristic_length": char_len,
                "grid_spacing": grid_sp,
                "ambient_density": float(self.config["initial_conditions"]["ambient_density"])
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "max_density": float(np.max(self.density)),
            "max_velocity": float(np.max(np.linalg.norm(self.velocity, axis=-1)))
        }

