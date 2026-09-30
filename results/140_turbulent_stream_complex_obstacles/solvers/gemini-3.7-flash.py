"""Turbulent stream through complex obstacles physics solver.

Eulerian grid fluid simulation for smoke transport around obstacles.
Solves incompressible Navier-Stokes with advection, diffusion, pressure projection,
inlet boundary conditions, and obstacle no-slip/no-penetration boundary conditions.
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
        np.random.seed(seed)

        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        self.current_time = 0.0

        # Initial state frame from scene
        self.initial_frame = scene.initial_state()

        # Grid parameters
        self.grid_shape = tuple(self.config["sampling"]["field_grid_shape"])  # (nx=24, ny=16, nz=28)
        self.nx, self.ny, self.nz = self.grid_shape

        # Domain bounds
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        self.domain_size = self.bounds_max - self.bounds_min

        self.dx = self.domain_size[0] / (self.nx - 1)
        self.dy = self.domain_size[1] / (self.ny - 1)
        self.dz = self.domain_size[2] / (self.nz - 1)

        # Coordinate grids (collocated)
        self.x = np.linspace(self.bounds_min[0], self.bounds_max[0], self.nx)
        self.y = np.linspace(self.bounds_min[1], self.bounds_max[1], self.ny)
        self.z = np.linspace(self.bounds_min[2], self.bounds_max[2], self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # Obstacle definitions
        # Ground: box half_extents [3.2, 1.4, 0.05], center [0, 0, -0.05] -> z in [-0.1, 0.0]
        # Obstacle low: box half_extents [0.38, 0.38, 0.38], center [-0.65, 0, 1.25]
        # Obstacle high: box half_extents [0.42, 0.42, 0.42], center [0.65, 0, 2.15]
        self.obs_low_center = np.array([-0.65, 0.0, 1.25])
        self.obs_low_half = np.array([0.38, 0.38, 0.38])

        self.obs_high_center = np.array([0.65, 0.0, 2.15])
        self.obs_high_half = np.array([0.42, 0.42, 0.42])

        # Solid mask (True where solid / non-fluid)
        self.solid_mask = np.zeros(self.grid_shape, dtype=bool)

        # Ground
        self.solid_mask[:, :, self.Z[0, 0, :] <= 0.0] = True

        # Obstacle low
        in_obs_low = (
            (np.abs(self.X - self.obs_low_center[0]) <= self.obs_low_half[0]) &
            (np.abs(self.Y - self.obs_low_center[1]) <= self.obs_low_half[1]) &
            (np.abs(self.Z - self.obs_low_center[2]) <= self.obs_low_half[2])
        )
        self.solid_mask[in_obs_low] = True

        # Obstacle high
        in_obs_high = (
            (np.abs(self.X - self.obs_high_center[0]) <= self.obs_high_half[0]) &
            (np.abs(self.Y - self.obs_high_center[1]) <= self.obs_high_half[1]) &
            (np.abs(self.Z - self.obs_high_center[2]) <= self.obs_high_half[2])
        )
        self.solid_mask[in_obs_high] = True

        # Active mask is fluid cells (~solid_mask)
        # Note: scene.initial_state() sets active_mask to ones(shape, dtype=bool) or fluid mask
        self.fluid_mask = ~self.solid_mask
        self.active_mask = self.fluid_mask.copy()

        # Inlet configuration: nozzle at [-2.72, 0.0, 1.0], pointing in +x direction
        self.inlet_pos = np.array([-2.72, 0.0, 1.0])
        self.inlet_radius = 0.40
        self.inlet_u = 2.5  # Inlet stream velocity in +x direction

        # Distance from nozzle axis (along x)
        dy_inlet = self.Y - self.inlet_pos[1]
        dz_inlet = self.Z - self.inlet_pos[2]
        r_inlet_sq = dy_inlet**2 + dz_inlet**2
        self.inlet_mask = (self.X <= -2.4) & (r_inlet_sq <= (self.inlet_radius * 1.1)**2)

        # Initialize physical fields
        self.scalar = self.initial_frame.fields["transported_scalar"].copy()
        self.u = np.zeros(self.grid_shape, dtype=np.float64)
        self.v = np.zeros(self.grid_shape, dtype=np.float64)
        self.w = np.zeros(self.grid_shape, dtype=np.float64)

        # Set inlet velocity profile initially
        self.u[self.inlet_mask] = self.inlet_u * np.exp(-r_inlet_sq[self.inlet_mask] / (self.inlet_radius**2))

        self.scalar_source = np.zeros(self.grid_shape, dtype=np.float64)
        self.scalar_source[self.inlet_mask] = 1.5 * np.exp(-r_inlet_sq[self.inlet_mask] / (self.inlet_radius**2))

        # Boundary samples for required trajectory contract
        self._setup_boundary_samples()

        # Particle representation
        self.particles_pos = self.initial_frame.position.copy()
        self.particles_vel = self.initial_frame.velocity.copy()
        self.particles_mass = self.initial_frame.mass.copy()
        self.particles_body_id = self.initial_frame.body_id.copy()
        self.particles_attributes = {k: v.copy() for k, v in self.initial_frame.attributes.items()}

        # Sub-stepping
        self.sub_dt = 0.005  # Internal simulation time step

    def _setup_boundary_samples(self) -> None:
        """Create boundary normal and velocity sample points on solid surfaces."""
        normals = []
        positions = []

        # Obstacle low faces
        for dim, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
            norm = np.zeros(3)
            norm[dim] = sign
            # Grid of sample points on this face
            u_dim = (dim + 1) % 3
            v_dim = (dim + 2) % 3
            u_vals = np.linspace(-self.obs_low_half[u_dim] * 0.8, self.obs_low_half[u_dim] * 0.8, 3)
            v_vals = np.linspace(-self.obs_low_half[v_dim] * 0.8, self.obs_low_half[v_dim] * 0.8, 3)
            for uu in u_vals:
                for vv in v_vals:
                    pt = self.obs_low_center.copy()
                    pt[dim] += sign * self.obs_low_half[dim]
                    pt[u_dim] += uu
                    pt[v_dim] += vv
                    positions.append(pt)
                    normals.append(norm)

        # Obstacle high faces
        for dim, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
            norm = np.zeros(3)
            norm[dim] = sign
            u_dim = (dim + 1) % 3
            v_dim = (dim + 2) % 3
            u_vals = np.linspace(-self.obs_high_half[u_dim] * 0.8, self.obs_high_half[u_dim] * 0.8, 3)
            v_vals = np.linspace(-self.obs_high_half[v_dim] * 0.8, self.obs_high_half[v_dim] * 0.8, 3)
            for uu in u_vals:
                for vv in v_vals:
                    pt = self.obs_high_center.copy()
                    pt[dim] += sign * self.obs_high_half[dim]
                    pt[u_dim] += uu
                    pt[v_dim] += vv
                    positions.append(pt)
                    normals.append(norm)

        # Ground samples
        gx = np.linspace(-2.5, 2.5, 5)
        gy = np.linspace(-1.0, 1.0, 3)
        for gx_val in gx:
            for gy_val in gy:
                positions.append(np.array([gx_val, gy_val, 0.0]))
                normals.append(np.array([0.0, 0.0, 1.0]))

        self.boundary_normal = np.array(normals, dtype=np.float64)
        # Normalize just in case
        norm_mags = np.linalg.norm(self.boundary_normal, axis=1, keepdims=True)
        self.boundary_normal = self.boundary_normal / np.maximum(norm_mags, 1e-12)

        num_b = len(self.boundary_normal)
        self.boundary_target_velocity = np.zeros((num_b, 3), dtype=np.float64)
        self.boundary_velocity = np.zeros((num_b, 3), dtype=np.float64)

    def _advect(self, field: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection on regular collocated grid."""
        # Index coordinates
        ix, iy, iz = np.meshgrid(
            np.arange(self.nx),
            np.arange(self.ny),
            np.arange(self.nz),
            indexing="ij"
        )

        # Backtrace in index space
        back_x = ix - (u * dt / self.dx)
        back_y = iy - (v * dt / self.dy)
        back_z = iz - (w * dt / self.dz)

        # Clamp backtrace to domain boundary indices
        np.clip(back_x, 0, self.nx - 1, out=back_x)
        np.clip(back_y, 0, self.ny - 1, out=back_y)
        np.clip(back_z, 0, self.nz - 1, out=back_z)

        coords = np.array([back_x.ravel(), back_y.ravel(), back_z.ravel()])
        advected = map_coordinates(field, coords, order=1, mode='nearest').reshape(self.grid_shape)
        return advected

    def _project_pressure(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, n_iter: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Jacobi pressure solver to project velocity onto divergence-free space."""
        # Divergence
        div = np.zeros_like(u)
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2.0 * self.dx) +
            (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2.0 * self.dy) +
            (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2.0 * self.dz)
        )

        p = np.zeros_like(u)
        dx2, dy2, dz2 = self.dx**2, self.dy**2, self.dz**2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2 + 1.0 / dz2)

        for _ in range(n_iter):
            p_new = p.copy()
            p_new[1:-1, 1:-1, 1:-1] = (
                (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) / dx2 +
                (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) / dy2 +
                (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / dz2 -
                div[1:-1, 1:-1, 1:-1]
            ) / denom

            # Boundary condition on pressure: Neumann (zero normal gradient)
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = 0.0  # Open downstream pressure outlet
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]

            p = p_new

        # Subtract pressure gradient
        u_proj = u.copy()
        v_proj = v.copy()
        w_proj = w.copy()

        u_proj[1:-1, :, :] -= (p[2:, :, :] - p[:-2, :, :]) / (2.0 * self.dx)
        v_proj[:, 1:-1, :] -= (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * self.dy)
        w_proj[:, :, 1:-1] -= (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * self.dz)

        return u_proj, v_proj, w_proj

    def _apply_boundaries(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, s: np.ndarray) -> None:
        """Apply inlet, solid obstacles, and domain wall boundary conditions."""
        # Solid obstacles: zero velocity inside and no-slip / zero normal velocity
        u[self.solid_mask] = 0.0
        v[self.solid_mask] = 0.0
        w[self.solid_mask] = 0.0
        s[self.solid_mask] = 0.0

        # Inlet nozzle condition: continuous inflow with slight turbulent perturbation
        dy_inlet = self.Y[self.inlet_mask] - self.inlet_pos[1]
        dz_inlet = self.Z[self.inlet_mask] - self.inlet_pos[2]
        r_sq = dy_inlet**2 + dz_inlet**2
        profile = np.exp(-r_sq / (self.inlet_radius**2))

        # Inflow velocity
        u[self.inlet_mask] = self.inlet_u * profile
        v[self.inlet_mask] = 0.15 * np.sin(4.0 * self.current_time + dy_inlet * 5.0) * profile
        w[self.inlet_mask] = 0.15 * np.cos(3.5 * self.current_time + dz_inlet * 5.0) * profile

        # Inflow scalar (smoke density)
        s[self.inlet_mask] = np.maximum(s[self.inlet_mask], 1.2 * profile)

        # Domain boundary conditions
        # Upstream x=0: u >= 0, v=0, w=0
        u[0, :, :] = np.where(self.inlet_mask[0, :, :], u[0, :, :], 0.0)
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        # Bottom z=0: no penetration
        w[:, :, 0] = 0.0
        # Top z=-1: free slip
        w[:, :, -1] = 0.0
        # Side y=0, y=-1: free slip
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0

    def _vorticity_confinement(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float, epsilon: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vorticity confinement to maintain coherent eddies around obstacles."""
        # Compute curl
        dv_dx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        du_dy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        wz = dv_dx - du_dy

        dw_dy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        dv_dz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2.0 * self.dz)
        wx = dw_dy - dv_dz

        du_dz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2.0 * self.dz)
        dw_dx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        wy = du_dz - dw_dx

        mag = np.sqrt(wx**2 + wy**2 + wz**2) + 1e-6

        # Gradient of vorticity magnitude
        g_x = (mag[2:, 1:-1, 1:-1] - mag[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        g_y = (mag[1:-1, 2:, 1:-1] - mag[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        g_z = (mag[1:-1, 1:-1, 2:] - mag[1:-1, 1:-1, :-2]) / (2.0 * self.dz)

        g_len = np.sqrt(g_x**2 + g_y**2 + g_z**2) + 1e-6
        nx_v = g_x / g_len
        ny_v = g_y / g_len
        nz_v = g_z / g_len

        # Force = epsilon * dx * (N x omega)
        fx = epsilon * self.dx * (ny_v * wz[1:-1, 1:-1, 1:-1] - nz_v * wy[1:-1, 1:-1, 1:-1])
        fy = epsilon * self.dx * (nz_v * wx[1:-1, 1:-1, 1:-1] - nx_v * wz[1:-1, 1:-1, 1:-1])
        fz = epsilon * self.dx * (nx_v * wy[1:-1, 1:-1, 1:-1] - ny_v * wx[1:-1, 1:-1, 1:-1])

        u_out = u.copy()
        v_out = v.copy()
        w_out = w.copy()

        u_out[2:-2, 2:-2, 2:-2] += fx * dt
        v_out[2:-2, 2:-2, 2:-2] += fy * dt
        w_out[2:-2, 2:-2, 2:-2] += fz * dt

        return u_out, v_out, w_out

    def _step(self, dt: float) -> None:
        """Advance fluid physics by sub-step dt."""
        # 1. Advect velocity field
        u_adv = self._advect(self.u, self.u, self.v, self.w, dt)
        v_adv = self._advect(self.v, self.u, self.v, self.w, dt)
        w_adv = self._advect(self.w, self.u, self.v, self.w, dt)

        # 2. Advect transported scalar (smoke)
        s_adv = self._advect(self.scalar, self.u, self.v, self.w, dt)

        # 3. Add buoyancy to w if smoke is buoyant
        # Small buoyancy effect
        w_adv += 0.25 * s_adv * dt

        # 4. Vorticity confinement for turbulent vortex shedding
        u_adv, v_adv, w_adv = self._vorticity_confinement(u_adv, v_adv, w_adv, dt, epsilon=0.8)

        # 5. Boundary conditions before projection
        self._apply_boundaries(u_adv, v_adv, w_adv, s_adv)

        # 6. Pressure projection
        u_proj, v_proj, w_proj = self._project_pressure(u_adv, v_adv, w_adv, n_iter=20)

        # 7. Boundary conditions after projection
        self._apply_boundaries(u_proj, v_proj, w_proj, s_adv)

        # Update fields
        self.u = u_proj
        self.v = v_proj
        self.w = w_proj
        self.scalar = np.clip(s_adv, 0.0, 5.0)

        # Update scalar source diagnostic
        dy_inlet = self.Y - self.inlet_pos[1]
        dz_inlet = self.Z - self.inlet_pos[2]
        r_inlet_sq = dy_inlet**2 + dz_inlet**2
        self.scalar_source = np.zeros(self.grid_shape, dtype=np.float64)
        self.scalar_source[self.inlet_mask] = 1.5 * np.exp(-r_inlet_sq[self.inlet_mask] / (self.inlet_radius**2))

        # Advance particles advected by fluid velocity
        if len(self.particles_pos) > 0:
            coords = np.array([
                (self.particles_pos[:, 0] - self.bounds_min[0]) / self.dx,
                (self.particles_pos[:, 1] - self.bounds_min[1]) / self.dy,
                (self.particles_pos[:, 2] - self.bounds_min[2]) / self.dz,
            ])
            np.clip(coords[0], 0, self.nx - 1, out=coords[0])
            np.clip(coords[1], 0, self.ny - 1, out=coords[1])
            np.clip(coords[2], 0, self.nz - 1, out=coords[2])

            p_u = map_coordinates(self.u, coords, order=1, mode='nearest')
            p_v = map_coordinates(self.v, coords, order=1, mode='nearest')
            p_w = map_coordinates(self.w, coords, order=1, mode='nearest')

            self.particles_vel = np.column_stack((p_u, p_v, p_w))
            self.particles_pos += self.particles_vel * dt

            # Obstacle collision / clamp for particles
            for pt in self.particles_pos:
                # low obs collision
                d_low = np.abs(pt - self.obs_low_center) - self.obs_low_half
                if np.all(d_low < 0):
                    max_dim = np.argmax(d_low)
                    pt[max_dim] = self.obs_low_center[max_dim] + np.sign(pt[max_dim] - self.obs_low_center[max_dim]) * (self.obs_low_half[max_dim] + 0.01)

                # high obs collision
                d_high = np.abs(pt - self.obs_high_center) - self.obs_high_half
                if np.all(d_high < 0):
                    max_dim = np.argmax(d_high)
                    pt[max_dim] = self.obs_high_center[max_dim] + np.sign(pt[max_dim] - self.obs_high_center[max_dim]) * (self.obs_high_half[max_dim] + 0.01)

                # Ground collision
                if pt[2] < 0.0:
                    pt[2] = 0.0

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to target timestamp and return StateFrame."""
        # Sub-step to the requested time
        while self.current_time < time - 1e-9:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Collocated diagnostic velocity grid
        velocity_field = np.stack((self.u, self.v, self.w), axis=-1)

        # Fields dictionary matching contract
        fields = {
            "active_mask": self.active_mask,
            "boundary_normal": self.boundary_normal,
            "boundary_target_velocity": self.boundary_target_velocity,
            "boundary_velocity": self.boundary_velocity,
            "scalar_source": self.scalar_source,
            "transported_scalar": self.scalar,
            "velocity_field": velocity_field,
            "cumulative_mass_source": np.zeros(len(self.particles_pos)),
        }

        spacing = float(self.config["sampling"]["spacing"])
        scalars = {
            "characteristic_length": max(float(np.ptp(self.particles_pos, axis=0).max()), spacing),
            "grid_spacing": spacing,
            "reference_velocity": 1.0,
            "contact_length_scale": spacing,
        }

        frame = StateFrame(
            time=time,
            position=self.particles_pos.copy(),
            velocity=self.particles_vel.copy(),
            mass=self.particles_mass.copy(),
            body_id=self.particles_body_id.copy(),
            attributes={k: v.copy() for k, v in self.particles_attributes.items()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "max_scalar": float(np.max(self.scalar)),
            "mean_scalar": float(np.mean(self.scalar)),
            "max_velocity": float(np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))),
        }

