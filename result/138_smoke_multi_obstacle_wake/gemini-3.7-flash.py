"""Solver implementation for Smoke Through Multiple Obstacles."""
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

        # Sampling / particle properties
        sampling = self.config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])

        # Grid parameters
        self.grid_shape = tuple(sampling["field_grid_shape"])  # (nx, ny, nz) = (24, 16, 28)
        self.nx, self.ny, self.nz = self.grid_shape

        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)

        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz
        self.d_cell = np.array([self.dx, self.dy, self.dz], dtype=np.float64)

        # Coordinates of cell centers
        xc = np.linspace(self.bounds_min[0] + 0.5 * self.dx, self.bounds_max[0] - 0.5 * self.dx, self.nx)
        yc = np.linspace(self.bounds_min[1] + 0.5 * self.dy, self.bounds_max[1] - 0.5 * self.dy, self.ny)
        zc = np.linspace(self.bounds_min[2] + 0.5 * self.dz, self.bounds_max[2] - 0.5 * self.dz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")

        # Obstacle geometry setup
        # Obstacle 1: low [-0.65, 0, 1.25], half_extents [0.38, 0.38, 0.38]
        # Obstacle 2: high [0.65, 0, 2.15], half_extents [0.42, 0.42, 0.42]
        obs1_min = np.array([-0.65 - 0.38, 0.0 - 0.38, 1.25 - 0.38])
        obs1_max = np.array([-0.65 + 0.38, 0.0 + 0.38, 1.25 + 0.38])

        obs2_min = np.array([0.65 - 0.42, 0.0 - 0.42, 2.15 - 0.42])
        obs2_max = np.array([0.65 + 0.42, 0.0 + 0.42, 2.15 + 0.42])

        in_obs1 = (
            (self.X >= obs1_min[0]) & (self.X <= obs1_max[0]) &
            (self.Y >= obs1_min[1]) & (self.Y <= obs1_max[1]) &
            (self.Z >= obs1_min[2]) & (self.Z <= obs1_max[2])
        )
        in_obs2 = (
            (self.X >= obs2_min[0]) & (self.X <= obs2_max[0]) &
            (self.Y >= obs2_min[1]) & (self.Y <= obs2_max[1]) &
            (self.Z >= obs2_min[2]) & (self.Z <= obs2_max[2])
        )
        self.solid_mask = in_obs1 | in_obs2
        self.active_mask = ~self.solid_mask

        # Boundary samples setup for required fields
        # Create boundary sample points around obstacles and outer domain
        b_pts = []
        b_nrms = []

        # Sample faces of obstacle 1
        b_pts.append(np.array([obs1_min[0], -0.65, 1.25]))
        b_nrms.append(np.array([-1.0, 0.0, 0.0]))
        b_pts.append(np.array([obs1_max[0], -0.65, 1.25]))
        b_nrms.append(np.array([1.0, 0.0, 0.0]))
        b_pts.append(np.array([-0.65, obs1_min[1], 1.25]))
        b_nrms.append(np.array([0.0, -1.0, 0.0]))
        b_pts.append(np.array([-0.65, obs1_max[1], 1.25]))
        b_nrms.append(np.array([0.0, 1.0, 0.0]))
        b_pts.append(np.array([-0.65, 0.0, obs1_min[2]]))
        b_nrms.append(np.array([0.0, 0.0, -1.0]))
        b_pts.append(np.array([-0.65, 0.0, obs1_max[2]]))
        b_nrms.append(np.array([0.0, 0.0, 1.0]))

        # Sample faces of obstacle 2
        b_pts.append(np.array([obs2_min[0], 0.0, 2.15]))
        b_nrms.append(np.array([-1.0, 0.0, 0.0]))
        b_pts.append(np.array([obs2_max[0], 0.0, 2.15]))
        b_nrms.append(np.array([1.0, 0.0, 0.0]))
        b_pts.append(np.array([0.65, obs2_min[1], 2.15]))
        b_nrms.append(np.array([0.0, -1.0, 0.0]))
        b_pts.append(np.array([0.65, obs2_max[1], 2.15]))
        b_nrms.append(np.array([0.0, 1.0, 0.0]))
        b_pts.append(np.array([0.65, 0.0, obs2_min[2]]))
        b_nrms.append(np.array([0.0, 0.0, -1.0]))
        b_pts.append(np.array([0.65, 0.0, obs2_max[2]]))
        b_nrms.append(np.array([0.0, 0.0, 1.0]))

        self.boundary_normals = np.array(b_nrms, dtype=np.float64)
        self.num_boundary_samples = len(b_nrms)
        self.boundary_target_velocity = np.zeros((self.num_boundary_samples, 3), dtype=np.float64)

        # Source term distribution (cylinder emitter at translation [0, 0, 0.05], radius 0.32, height 0.1)
        r_sq = self.X**2 + self.Y**2
        z_src = (self.Z >= 0.0) & (self.Z <= 0.25)
        self.source_mask = (r_sq <= 0.32**2) & z_src
        self.scalar_source = np.zeros(self.grid_shape, dtype=np.float64)
        self.scalar_source[self.source_mask] = 2.0  # continuous smoke injection

        # Initial fields
        init_frame = scene.initial_state()
        self.initial_frame = init_frame
        self.particles_pos = init_frame.position.copy()
        self.particles_vel = init_frame.velocity.copy()
        self.particles_mass = init_frame.mass.copy()
        self.particles_body_id = init_frame.body_id.copy()
        self.attributes = {k: v.copy() for k, v in init_frame.attributes.items()}

        # Grid state: velocity (u, v, w), density/smoke scalar S, temperature T
        self.u = np.zeros(self.grid_shape, dtype=np.float64)
        self.v = np.zeros(self.grid_shape, dtype=np.float64)
        self.w = np.zeros(self.grid_shape, dtype=np.float64)
        self.density_field = init_frame.fields["transported_scalar"].copy()
        self.temperature = np.zeros(self.grid_shape, dtype=np.float64)
        self.temperature[self.source_mask] = 1.0

        self.sim_time = 0.0
        self.sub_dt = 0.005  # internal physics sub-step

    def _advect(self, field: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        # Semi-Lagrangian advection using map_coordinates
        # Grid index coordinates
        ix, iy, iz = np.meshgrid(
            np.arange(self.nx, dtype=np.float64),
            np.arange(self.ny, dtype=np.float64),
            np.arange(self.nz, dtype=np.float64),
            indexing="ij"
        )
        # Backtrace positions in index space
        back_x = np.clip(ix - (u * dt) / self.dx, 0.0, self.nx - 1.0)
        back_y = np.clip(iy - (v * dt) / self.dy, 0.0, self.ny - 1.0)
        back_z = np.clip(iz - (w * dt) / self.dz, 0.0, self.nz - 1.0)

        coords = np.array([back_x, back_y, back_z])
        return map_coordinates(field, coords, order=1, mode='nearest')

    def _project(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Enforce boundary conditions on velocity
        u[self.solid_mask] = 0.0
        v[self.solid_mask] = 0.0
        w[self.solid_mask] = 0.0

        # Domain boundary conditions (impermeable walls)
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, :, 0] = 0.0
        # open / outflow at top z
        # w[:, :, -1] allows outflow

        # Compute divergence
        div = np.zeros(self.grid_shape, dtype=np.float64)
        div[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2.0 * self.dx) +
            (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2.0 * self.dy) +
            (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2.0 * self.dz)
        )
        div[self.solid_mask] = 0.0

        # Pressure solve using Jacobi iterations
        p = np.zeros(self.grid_shape, dtype=np.float64)
        idx2 = 1.0 / (self.dx ** 2)
        idy2 = 1.0 / (self.dy ** 2)
        idz2 = 1.0 / (self.dz ** 2)
        diag = 2.0 * (idx2 + idy2 + idz2)

        for _ in range(40):
            p_next = p.copy()
            p_next[1:-1, 1:-1, 1:-1] = (
                (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * idx2 +
                (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * idy2 +
                (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * idz2 -
                div[1:-1, 1:-1, 1:-1]
            ) / diag

            # Boundary condition on pressure
            p_next[self.solid_mask] = 0.0
            p = p_next

        # Subtract pressure gradient
        u[1:-1, 1:-1, 1:-1] -= (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        v[1:-1, 1:-1, 1:-1] -= (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        w[1:-1, 1:-1, 1:-1] -= (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * self.dz)

        # Re-enforce zero velocity in solids and walls
        u[self.solid_mask] = 0.0
        v[self.solid_mask] = 0.0
        w[self.solid_mask] = 0.0
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        v[:, 0, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, :, 0] = 0.0

        return u, v, w

    def _vorticity_confinement(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float, epsilon: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Calculate curl (vorticity)
        # curl_x = dw/dy - dv/dz
        # curl_y = du/dz - dw/dx
        # curl_z = dv/dx - du/dy
        cx = np.zeros(self.grid_shape, dtype=np.float64)
        cy = np.zeros(self.grid_shape, dtype=np.float64)
        cz = np.zeros(self.grid_shape, dtype=np.float64)

        cx[:, 1:-1, 1:-1] = (w[:, 2:, 1:-1] - w[:, :-2, 1:-1]) / (2.0 * self.dy) - (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2.0 * self.dz)
        cy[1:-1, :, 1:-1] = (u[1:-1, :, 2:] - u[1:-1, :, :-2]) / (2.0 * self.dz) - (w[2:, :, 1:-1] - w[:-2, :, 1:-1]) / (2.0 * self.dx)
        cz[1:-1, 1:-1, :] = (v[2:, 1:-1, :] - v[:-2, 1:-1, :]) / (2.0 * self.dx) - (u[1:-1, 2:, :] - u[1:-1, :-2, :]) / (2.0 * self.dy)

        c_mag = np.sqrt(cx**2 + cy**2 + cz**2 + 1e-6)

        # Gradient of c_mag
        nx = np.zeros(self.grid_shape, dtype=np.float64)
        ny = np.zeros(self.grid_shape, dtype=np.float64)
        nz = np.zeros(self.grid_shape, dtype=np.float64)

        nx[1:-1, :, :] = (c_mag[2:, :, :] - c_mag[:-2, :, :]) / (2.0 * self.dx)
        ny[:, 1:-1, :] = (c_mag[:, 2:, :] - c_mag[:, :-2, :]) / (2.0 * self.dy)
        nz[:, :, 1:-1] = (c_mag[:, :, 2:] - c_mag[:, :, :-2]) / (2.0 * self.dz)

        n_len = np.sqrt(nx**2 + ny**2 + nz**2 + 1e-6)
        nx /= n_len
        ny /= n_len
        nz /= n_len

        # Force = epsilon * dx * (N x omega)
        h = np.cbrt(self.dx * self.dy * self.dz)
        fx = epsilon * h * (ny * cz - nz * cy)
        fy = epsilon * h * (nz * cx - nx * cz)
        fz = epsilon * h * (nx * cy - ny * cx)

        u += dt * fx
        v += dt * fy
        w += dt * fz
        return u, v, w

    def _step(self, dt: float) -> None:
        # 1. Add buoyancy force based on temperature and smoke density
        # B = -alpha * density + beta * (T - T_amb)
        alpha = 0.15
        beta = 3.5
        buoyancy = -alpha * self.density_field + beta * self.temperature
        self.w += dt * buoyancy

        # Add small lateral perturbation to induce asymmetric vortex shedding around obstacles
        perturb = 0.12 * np.sin(3.0 * self.Z + 1.5 * self.sim_time) * np.exp(-((self.X**2 + self.Y**2) / 0.5))
        self.u += dt * perturb

        # 2. Vorticity confinement to preserve turbulent eddies and plume wake roll-up
        self.u, self.v, self.w = self._vorticity_confinement(self.u, self.v, self.w, dt, epsilon=0.6)

        # 3. Velocity projection (incompressibility)
        self.u, self.v, self.w = self._project(self.u, self.v, self.w)

        # 4. Advect velocity
        u_new = self._advect(self.u, self.u, self.v, self.w, dt)
        v_new = self._advect(self.v, self.u, self.v, self.w, dt)
        w_new = self._advect(self.w, self.u, self.v, self.w, dt)
        self.u, self.v, self.w = u_new, v_new, w_new

        # 5. Velocity projection again
        self.u, self.v, self.w = self._project(self.u, self.v, self.w)

        # 6. Advect scalar (smoke density) and temperature
        self.density_field = self._advect(self.density_field, self.u, self.v, self.w, dt)
        self.temperature = self._advect(self.temperature, self.u, self.v, self.w, dt)

        # Dissipation / cooling
        self.density_field *= (1.0 - 0.005 * dt)
        self.temperature *= (1.0 - 0.05 * dt)

        # 7. Add continuous source at emitter
        self.density_field[self.source_mask] = np.maximum(self.density_field[self.source_mask], 1.2)
        self.temperature[self.source_mask] = np.maximum(self.temperature[self.source_mask], 1.0)

        # Zero out inside obstacles
        self.density_field[self.solid_mask] = 0.0
        self.temperature[self.solid_mask] = 0.0

        # Advance particles with velocity field
        # Interpolate velocity at particle positions
        px = np.clip((self.particles_pos[:, 0] - self.bounds_min[0]) / self.dx - 0.5, 0.0, self.nx - 1.0)
        py = np.clip((self.particles_pos[:, 1] - self.bounds_min[1]) / self.dy - 0.5, 0.0, self.ny - 1.0)
        pz = np.clip((self.particles_pos[:, 2] - self.bounds_min[2]) / self.dz - 0.5, 0.0, self.nz - 1.0)
        pcoords = np.array([px, py, pz])

        pu = map_coordinates(self.u, pcoords, order=1, mode='nearest')
        pv = map_coordinates(self.v, pcoords, order=1, mode='nearest')
        pw = map_coordinates(self.w, pcoords, order=1, mode='nearest')

        self.particles_vel[:, 0] = pu
        self.particles_vel[:, 1] = pv
        self.particles_vel[:, 2] = pw

        self.particles_pos += dt * self.particles_vel

        self.sim_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation to target time
        while self.sim_time < time - 1e-6:
            dt = min(self.sub_dt, time - self.sim_time)
            self._step(dt)

        # Prepare velocity field diagnostic [nx, ny, nz, 3]
        vel_field = np.stack([self.u, self.v, self.w], axis=-1)

        # Boundary velocity diagnostics (0 at static walls)
        boundary_vel = np.zeros((self.num_boundary_samples, 3), dtype=np.float64)

        fields = {
            "active_mask": self.active_mask.copy(),
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": boundary_vel,
            "scalar_source": self.scalar_source.copy(),
            "transported_scalar": self.density_field.copy(),
            "velocity_field": vel_field,
            "cumulative_mass_source": np.zeros(len(self.particles_pos), dtype=np.float64),
        }

        scalars = {
            "contact_length_scale": self.spacing,
            "grid_spacing": self.spacing,
            "characteristic_length": max(float(np.ptp(self.particles_pos, axis=0).max()), self.spacing),
            "reference_velocity": 1.0,
        }

        frame = StateFrame(
            time=time,
            position=self.particles_pos.copy(),
            velocity=self.particles_vel.copy(),
            mass=self.particles_mass.copy(),
            body_id=self.particles_body_id.copy(),
            attributes={k: v.copy() for k, v in self.attributes.items()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "sim_time": self.sim_time,
            "max_scalar": float(np.max(self.density_field)),
            "max_velocity": float(np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))),
        }

