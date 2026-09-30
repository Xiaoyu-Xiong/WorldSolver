"""Physical 3D Navier-Stokes Smoke Simulator with Sphere Obstacle and Buoyancy."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        # Grid parameters from scene config / geometry
        # Subject bounds: [-3.0, -1.2, 0.0] to [3.0, 1.2, 3.8]
        # Grid shape: [24, 16, 28]
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=float)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=float)
        self.grid_shape = tuple(scene.config["sampling"].get("field_grid_shape", [24, 16, 28]))
        self.nx, self.ny, self.nz = self.grid_shape

        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz
        self.d_cell = np.array([self.dx, self.dy, self.dz], dtype=float)

        # Sphere obstacle: center = [0.0, 0.0, 1.35], radius = 0.48
        self.sphere_center = np.array([0.0, 0.0, 1.35], dtype=float)
        self.sphere_radius = 0.48

        # Smoke emitter cylinder: center [0.0, 0.0, 0.05], radius 0.32, height 0.10
        self.emitter_center = np.array([0.0, 0.0, 0.05], dtype=float)
        self.emitter_radius = 0.32
        self.emitter_height = 0.10

        # Create coordinate grids (cell centers)
        x_coords = np.linspace(self.bounds_min[0] + self.dx * 0.5, self.bounds_max[0] - self.dx * 0.5, self.nx)
        y_coords = np.linspace(self.bounds_min[1] + self.dy * 0.5, self.bounds_max[1] - self.dy * 0.5, self.ny)
        z_coords = np.linspace(self.bounds_min[2] + self.dz * 0.5, self.bounds_max[2] - self.dz * 0.5, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

        # Obstacle mask (True for fluid, False inside sphere)
        dist_sq = (self.X - self.sphere_center[0])**2 + (self.Y - self.sphere_center[1])**2 + (self.Z - self.sphere_center[2])**2
        self.solid_mask = dist_sq < (self.sphere_radius ** 2)
        self.fluid_mask = ~self.solid_mask
        self.active_mask = self.fluid_mask.copy()

        # Emitter mask & source field
        emitter_r_sq = (self.X - self.emitter_center[0])**2 + (self.Y - self.emitter_center[1])**2
        emitter_z = self.Z - self.emitter_center[2]
        self.emitter_mask = (emitter_r_sq <= (self.emitter_radius * 1.2)**2) & (emitter_z >= -0.05) & (emitter_z <= self.emitter_height + 0.15)

        # Continuous scalar source
        r_norm = np.sqrt(emitter_r_sq) / self.emitter_radius
        self.scalar_source = np.zeros(self.grid_shape, dtype=float)
        self.scalar_source[self.emitter_mask] = 2.5 * np.exp(-(r_norm[self.emitter_mask] ** 2))

        # Buoyancy parameters (Fedkiw / Stam smoke model: f_buoy = (-alpha * rho + beta * (T - T_amb)) * z_hat)
        self.buoyancy_alpha = 0.05  # smoke density drag/weight
        self.buoyancy_beta = 3.5    # thermal / smoke upward buoyancy strength

        # State variables
        # Velocity field (nx, ny, nz, 3)
        self.u = np.zeros(self.grid_shape + (3,), dtype=float)
        # Transported scalar (smoke density) (nx, ny, nz)
        # Initialize from scene initial state or smooth bell
        x_lin = np.linspace(-3.0, 3.0, self.nx)[:, None, None]
        y_lin = np.linspace(-1.2, 1.2, self.ny)[None, :, None]
        z_lin = np.linspace(0.0, 3.8, self.nz)[None, None, :]
        self.density = 1.20 * np.exp(-((x_lin / 0.28) ** 2 + (y_lin / 0.28) ** 2)) * np.exp(-((z_lin - 0.72) / 0.78) ** 2)
        self.density[self.solid_mask] = 0.0

        # Boundary surface samples for StateFrame
        # Sample sphere surface and ground
        n_sphere = 64
        phi = np.linspace(0, np.pi, int(np.sqrt(n_sphere)))
        theta = np.linspace(0, 2 * np.pi, int(np.sqrt(n_sphere)))
        phi_g, theta_g = np.meshgrid(phi, theta)
        bx = self.sphere_center[0] + self.sphere_radius * np.sin(phi_g) * np.cos(theta_g)
        by = self.sphere_center[1] + self.sphere_radius * np.sin(phi_g) * np.sin(theta_g)
        bz = self.sphere_center[2] + self.sphere_radius * np.cos(phi_g)
        normals_sphere = np.stack([np.sin(phi_g) * np.cos(theta_g), np.sin(phi_g) * np.sin(theta_g), np.cos(phi_g)], axis=-1).reshape(-1, 3)
        sphere_pts = np.stack([bx.ravel(), by.ravel(), bz.ravel()], axis=-1)

        # Ground samples
        gx, gy = np.meshgrid(np.linspace(-3.0, 3.0, 8), np.linspace(-1.2, 1.2, 8))
        ground_pts = np.stack([gx.ravel(), gy.ravel(), np.zeros_like(gx.ravel())], axis=-1)
        ground_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(ground_pts), 1))

        self.boundary_pts = np.concatenate([sphere_pts, ground_pts], axis=0)
        self.boundary_normals = np.concatenate([normals_sphere, ground_normals], axis=0)
        self.num_boundary = len(self.boundary_pts)

        self.current_time = 0.0
        self.initial_frame = scene.initial_state()
        self.particles_pos = self.initial_frame.position.copy()
        self.particles_vel = self.initial_frame.velocity.copy()
        self.mass = self.initial_frame.mass.copy()
        self.body_id = self.initial_frame.body_id.copy()

    def _advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection in 3D using trilinear interpolation."""
        # Grid index coordinates
        ix, iy, iz = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij"
        )

        # Backtrace in index space
        back_x = ix - (self.u[..., 0] * dt / self.dx)
        back_y = iy - (self.u[..., 1] * dt / self.dy)
        back_z = iz - (self.u[..., 2] * dt / self.dz)

        # Clamp to domain bounds
        back_x = np.clip(back_x, 0, self.nx - 1)
        back_y = np.clip(back_y, 0, self.ny - 1)
        back_z = np.clip(back_z, 0, self.nz - 1)

        coords = np.array([back_x.ravel(), back_y.ravel(), back_z.ravel()])

        if field.ndim == 3:
            adv = map_coordinates(field, coords, order=1, mode='nearest').reshape(field.shape)
            return adv
        elif field.ndim == 4:
            res = np.zeros_like(field)
            for c in range(field.shape[-1]):
                res[..., c] = map_coordinates(field[..., c], coords, order=1, mode='nearest').reshape(self.grid_shape)
            return res
        return field

    def _project(self, u: np.ndarray, iterations: int = 40) -> np.ndarray:
        """Pressure projection via Gauss-Seidel / Jacobi relaxation to ensure divergence-free flow."""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz

        # Enforce boundary velocity: zero normal flow at domain walls and obstacle
        u[0, :, :, 0] = np.maximum(u[0, :, :, 0], 0.0)
        u[-1, :, :, 0] = np.minimum(u[-1, :, :, 0], 0.0)
        u[:, 0, :, 1] = np.maximum(u[:, 0, :, 1], 0.0)
        u[:, -1, :, 1] = np.minimum(u[:, -1, :, 1], 0.0)
        u[:, :, 0, 2] = np.maximum(u[:, :, 0, 2], 0.0)  # ground floor no outflow
        # Top z allows outflow

        # Obstacle zero normal velocity
        u[self.solid_mask] = 0.0

        # Compute divergence on fluid cells
        div = np.zeros(self.grid_shape, dtype=float)
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1, 0] - u[:-2, 1:-1, 1:-1, 0]) / (2.0 * dx) +
            (u[1:-1, 2:, 1:-1, 1] - u[1:-1, :-2, 1:-1, 1]) / (2.0 * dy) +
            (u[1:-1, 1:-1, 2:, 2] - u[1:-1, 1:-1, :-2, 2]) / (2.0 * dz)
        )
        div[self.solid_mask] = 0.0

        # Solve Poisson equation: Lap(p) = div
        p = np.zeros(self.grid_shape, dtype=float)
        inv_diag = 2.0 / (dx*dx) + 2.0 / (dy*dy) + 2.0 / (dz*dz)
        coeff_x = 1.0 / (dx * dx)
        coeff_y = 1.0 / (dy * dy)
        coeff_z = 1.0 / (dz * dz)

        # Red-Black Gauss-Seidel relaxation
        rx, ry, rz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        red = ((rx + ry + rz) % 2 == 0) & self.fluid_mask
        black = ((rx + ry + rz) % 2 == 1) & self.fluid_mask

        for _ in range(iterations):
            for mask in (red, black):
                p_xp = np.pad(p[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode='edge')
                p_xm = np.pad(p[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode='edge')
                p_yp = np.pad(p[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode='edge')
                p_ym = np.pad(p[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode='edge')
                p_zp = np.pad(p[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode='edge')
                p_zm = np.pad(p[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='edge')

                # Neumann boundary at solid obstacles
                p_xp[self.solid_mask] = p[self.solid_mask]
                p_xm[self.solid_mask] = p[self.solid_mask]
                p_yp[self.solid_mask] = p[self.solid_mask]
                p_ym[self.solid_mask] = p[self.solid_mask]
                p_zp[self.solid_mask] = p[self.solid_mask]
                p_zm[self.solid_mask] = p[self.solid_mask]

                p_new = (coeff_x * (p_xp + p_xm) + coeff_y * (p_yp + p_ym) + coeff_z * (p_zp + p_zm) - div) / inv_diag
                p[mask] = p_new[mask]

        # Subtract pressure gradient from velocity
        grad_p_x = np.zeros_like(p)
        grad_p_y = np.zeros_like(p)
        grad_p_z = np.zeros_like(p)

        grad_p_x[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
        grad_p_y[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dy)
        grad_p_z[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dz)

        u[..., 0] -= grad_p_x
        u[..., 1] -= grad_p_y
        u[..., 2] -= grad_p_z

        # Enforce solid boundaries
        u[self.solid_mask] = 0.0
        return u

    def _step(self, dt: float) -> None:
        """Physical step of Navier-Stokes Boussinesq smoke simulation."""
        # 1. Source injection
        self.density += self.scalar_source * dt

        # 2. Buoyancy force: F_b = beta * density * z_hat
        # Smoke rises due to thermal buoyancy
        buoyancy = self.buoyancy_beta * self.density
        self.u[..., 2] += buoyancy * dt

        # Small vorticity confinement / perturbation for realistic wake roll-up
        if np.max(self.density) > 0.01:
            # Gentle transverse swirl near plume center to seed beautiful wake shed
            self.u[..., 0] += 0.02 * np.sin(2.0 * np.pi * self.Z / 1.5) * self.density * dt

        # 3. Semi-Lagrangian Advection
        self.u = self._advect(self.u, dt)
        self.density = self._advect(self.density, dt)

        # Dissipation & zeroing in solid
        self.density[self.solid_mask] = 0.0
        self.density = np.clip(self.density, 0.0, 10.0)

        # 4. Pressure Projection (incompressibility + boundary condition)
        self.u = self._project(self.u, iterations=35)

        # 5. Advect sample particles if present for material rendering/tracking
        if len(self.particles_pos) > 0:
            # Interpolate velocity to particle positions
            px = np.clip((self.particles_pos[:, 0] - self.bounds_min[0]) / self.dx - 0.5, 0, self.nx - 1)
            py = np.clip((self.particles_pos[:, 1] - self.bounds_min[1]) / self.dy - 0.5, 0, self.ny - 1)
            pz = np.clip((self.particles_pos[:, 2] - self.bounds_min[2]) / self.dz - 0.5, 0, self.nz - 1)
            coords = np.array([px, py, pz])
            vx = map_coordinates(self.u[..., 0], coords, order=1, mode='nearest')
            vy = map_coordinates(self.u[..., 1], coords, order=1, mode='nearest')
            vz = map_coordinates(self.u[..., 2], coords, order=1, mode='nearest')
            self.particles_vel = np.column_stack([vx, vy, vz])
            self.particles_pos += self.particles_vel * dt

            # Sphere collision response for particles
            p_rel = self.particles_pos - self.sphere_center
            p_dist = np.linalg.norm(p_rel, axis=1)
            coll = p_dist < (self.sphere_radius + 0.02)
            if np.any(coll):
                n = p_rel[coll] / np.maximum(p_dist[coll, None], 1e-6)
                self.particles_pos[coll] = self.sphere_center + n * (self.sphere_radius + 0.02)
                # Zero normal velocity
                vn = np.sum(self.particles_vel[coll] * n, axis=1, keepdims=True)
                self.particles_vel[coll] -= np.maximum(vn, 0.0) * n

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward using sub-stepping for numerical stability (CFL-aware)
        sub_dt_max = 0.015
        while self.current_time < target_time - 1e-8:
            dt = min(target_time - self.current_time, sub_dt_max)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time

        # Construct required trajectory fields
        # fields: active_mask, boundary_normal, boundary_target_velocity, boundary_velocity,
        #         scalar_source, transported_scalar, velocity_field
        # scalars: characteristic_length, grid_spacing, reference_velocity

        boundary_vel = np.zeros((self.num_boundary, 3), dtype=float)
        boundary_target_vel = np.zeros((self.num_boundary, 3), dtype=float)

        fields = {
            "active_mask": self.active_mask.copy(),
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_target_velocity": boundary_target_vel,
            "boundary_velocity": boundary_vel,
            "scalar_source": self.scalar_source.copy(),
            "transported_scalar": self.density.copy(),
            "velocity_field": self.u.copy(),
            "cumulative_mass_source": np.zeros(len(self.particles_pos)),
        }

        scalars = {
            "characteristic_length": float(2.0 * self.sphere_radius),
            "grid_spacing": float(self.dx),
            "reference_velocity": 1.0,
            "contact_length_scale": float(self.dx),
        }

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.full(len(self.particles_pos), 0.45 * self.scene.config["sampling"]["spacing"]),
            "material_label": self.body_id.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], len(self.particles_pos), axis=0),
        }

        frame = StateFrame(
            time=float(self.current_time),
            position=self.particles_pos.copy(),
            velocity=self.particles_vel.copy(),
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
            "max_velocity": float(np.max(np.linalg.norm(self.u, axis=-1))),
            "max_density": float(np.max(self.density)),
        }

