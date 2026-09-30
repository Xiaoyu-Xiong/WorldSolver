"""Benchmark solver implementing vortex-particle smoke explosion."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = {}
        self.current_time = 0.0
        self.grid_shape = (24, 16, 28)
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=float)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=float)

        # Grid coordinates
        self.x_coords = None
        self.y_coords = None
        self.z_coords = None
        self.grid_points = None
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.dx_avg = 0.18

        # Eulerian fields
        self.active_mask = None
        self.velocity_field = None
        self.transported_scalar = None
        self.temperature_field = None
        self.scalar_source = None

        # Boundary samples
        self.boundary_normal = None
        self.boundary_velocity = None
        self.boundary_target_velocity = None

        # Lagrangian / Particle representation (for StateFrame position / velocity)
        self.particle_pos = None
        self.particle_vel = None
        self.particle_mass = None
        self.particle_labels = None
        self.particle_radius = None

        # Vorticity particles for vortex method
        self.vortex_pos = None
        self.vortex_gamma = None  # Vorticity vector / circulation
        self.vortex_radius = None

        # Simulation parameters
        self.blast_pos = np.array([0.0, 0.0, 0.85], dtype=float)
        self.blast_radius = 0.26
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.sub_dt = 0.005

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config
        self.current_time = 0.0

        sampling = self.config.get("sampling", {})
        self.grid_shape = tuple(sampling.get("field_grid_shape", [24, 16, 28]))
        self.dx_avg = float(sampling.get("spacing", 0.18))

        nx, ny, nz = self.grid_shape
        self.x_coords = np.linspace(self.bounds_min[0], self.bounds_max[0], nx)
        self.y_coords = np.linspace(self.bounds_min[1], self.bounds_max[1], ny)
        self.z_coords = np.linspace(self.bounds_min[2], self.bounds_max[2], nz)
        self.dx = float(self.x_coords[1] - self.x_coords[0])
        self.dy = float(self.y_coords[1] - self.y_coords[0])
        self.dz = float(self.z_coords[1] - self.z_coords[0])

        X, Y, Z = np.meshgrid(self.x_coords, self.y_coords, self.z_coords, indexing="ij")
        self.grid_points = np.stack([X, Y, Z], axis=-1)

        # Initial StateFrame to borrow structure
        init_frame = scene.initial_state()

        # Extract initial fields
        self.active_mask = np.ones(self.grid_shape, dtype=bool)
        self.velocity_field = np.zeros(self.grid_shape + (3,), dtype=float)

        # Initial smoke scalar: compact hot smoke source
        r_blast = np.sqrt(X**2 + Y**2 + (Z - self.blast_pos[2])**2)
        self.transported_scalar = 1.15 * np.exp(-((r_blast / 0.62) ** 2))
        self.temperature_field = 2.0 * np.exp(-((r_blast / 0.5) ** 2))
        self.scalar_source = np.zeros(self.grid_shape, dtype=float)

        # Rigid boundary at z=0 (ground)
        # Boundary samples: sample ground plane
        num_b_x = 20
        num_b_y = 12
        bx = np.linspace(-3.0, 3.0, num_b_x)
        by = np.linspace(-1.2, 1.2, num_b_y)
        BX, BY = np.meshgrid(bx, by, indexing="ij")
        b_count = num_b_x * num_b_y
        self.boundary_normal = np.zeros((b_count, 3), dtype=float)
        self.boundary_normal[:, 2] = 1.0  # Ground normal points up (+z)
        self.boundary_target_velocity = np.zeros((b_count, 3), dtype=float)
        self.boundary_velocity = np.zeros((b_count, 3), dtype=float)

        # Particles
        self.particle_pos = init_frame.position.copy()
        self.particle_vel = np.zeros_like(self.particle_pos)
        self.particle_mass = init_frame.mass.copy()
        self.particle_labels = init_frame.body_id.copy()
        self.particle_radius = init_frame.attributes.get(
            "particle_radius", np.full(len(self.particle_pos), 0.45 * self.dx_avg)
        ).copy()

        # Initialize vortex particles along expanding rings/shells to create multiscale turbulent mushroom cloud
        num_vortices = 120
        v_pos = []
        v_gamma = []
        v_rad = []

        for i in range(num_vortices):
            # Form initial vortex rings around the blast center
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-0.3, 0.3)
            r = np.random.uniform(0.1, 0.4)
            pos = self.blast_pos + np.array([r * np.cos(theta), r * np.sin(theta), r * np.sin(phi)])

            # Vorticity tangential to ring (cross with up vector)
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
            strength = np.random.uniform(2.0, 5.0) * tangent

            v_pos.append(pos)
            v_gamma.append(strength)
            v_rad.append(0.25)

        self.vortex_pos = np.array(v_pos, dtype=float)
        self.vortex_gamma = np.array(v_gamma, dtype=float)
        self.vortex_radius = np.array(v_rad, dtype=float)

    def _interp_to_grid(self, field, query_pos):
        """Trilinear interpolation of grid field at continuous 3D points."""
        nx, ny, nz = self.grid_shape
        # Map query_pos (..., 3) to fractional index coordinates
        gx = (query_pos[..., 0] - self.bounds_min[0]) / (self.bounds_max[0] - self.bounds_min[0]) * (nx - 1)
        gy = (query_pos[..., 1] - self.bounds_min[1]) / (self.bounds_max[1] - self.bounds_min[1]) * (ny - 1)
        gz = (query_pos[..., 2] - self.bounds_min[2]) / (self.bounds_max[2] - self.bounds_min[2]) * (nz - 1)

        coords = np.stack([gx, gy, gz], axis=0)

        if field.ndim == 3:
            return map_coordinates(field, coords, order=1, mode='nearest')
        elif field.ndim == 4:
            res = np.zeros(query_pos.shape[:-1] + (field.shape[-1],), dtype=float)
            for c in range(field.shape[-1]):
                res[..., c] = map_coordinates(field[..., c], coords, order=1, mode='nearest')
            return res

    def _advect_field_semi_lagrangian(self, field, dt):
        """Advect grid field backwards along velocity_field."""
        # Backtrace
        prev_pos = self.grid_points - self.velocity_field * dt
        # Clamp to bounds
        prev_pos = np.clip(prev_pos, self.bounds_min, self.bounds_max)
        return self._interp_to_grid(field, prev_pos)

    def _step_physics(self, dt: float) -> None:
        t = self.current_time
        nx, ny, nz = self.grid_shape
        X = self.grid_points[..., 0]
        Y = self.grid_points[..., 1]
        Z = self.grid_points[..., 2]

        # 1. Source and Blast dynamics
        # Early blast impulse (t < 1.2s)
        blast_decay = np.exp(-t / 0.8)
        r_blast = np.sqrt(X**2 + Y**2 + (Z - self.blast_pos[2])**2) + 1e-5

        # Radial blast force
        radial_dir = (self.grid_points - self.blast_pos) / r_blast[..., None]
        radial_mag = 4.0 * blast_decay * np.exp(-((r_blast / 0.6)**2))
        blast_acc = radial_dir * radial_mag[..., None]

        # Continuous buoyancy based on temperature
        # Boussinesq buoyancy: force in +z proportional to temperature
        buoyancy_acc = np.zeros_like(self.velocity_field)
        buoyancy_acc[..., 2] = 4.5 * self.temperature_field

        # 2. Vortex particle induction (Biot-Savart with smoothed blob kernel)
        vortex_induced_vel = np.zeros_like(self.velocity_field)
        if len(self.vortex_pos) > 0:
            # Vectorized evaluation over chunks of grid or direct broadcast
            # To keep it fast, evaluate at grid points
            # grid_points shape: (nx, ny, nz, 3) -> flattened (N_grid, 3)
            flat_grid = self.grid_points.reshape(-1, 3)
            flat_induced = np.zeros_like(flat_grid)

            for vp, vg, vr in zip(self.vortex_pos, self.vortex_gamma, self.vortex_radius):
                r_vec = flat_grid - vp  # (N_grid, 3)
                r_sq = np.sum(r_vec**2, axis=-1)  # (N_grid,)
                sigma_sq = vr**2
                factor = (1.0 - np.exp(-r_sq / sigma_sq)) / (4.0 * np.pi * (r_sq + sigma_sq)**1.5 + 1e-6)
                # Cross product vg x r_vec
                cross_prod = np.cross(vg, r_vec)
                flat_induced += cross_prod * factor[:, None]

            vortex_induced_vel = flat_induced.reshape(nx, ny, nz, 3)

        # 3. Update velocity field
        # Integrate accelerations
        self.velocity_field += (blast_acc + buoyancy_acc) * dt
        # Add vortex induced component smoothly
        self.velocity_field = 0.85 * self.velocity_field + 0.15 * (self.velocity_field + vortex_induced_vel)

        # Boundary conditions on grid velocity:
        # Ground no-penetration: w(z=0) = 0 and w >= 0 near z=0
        self.velocity_field[..., :, :, 0, 2] = np.maximum(0.0, self.velocity_field[..., :, :, 0, 2])
        # Top boundary damping
        self.velocity_field[..., :, :, -1, 2] = np.minimum(0.0, self.velocity_field[..., :, :, -1, 2])
        # Lateral boundary damping
        self.velocity_field[0, :, :, 0] = np.maximum(0.0, self.velocity_field[0, :, :, 0])
        self.velocity_field[-1, :, :, 0] = np.minimum(0.0, self.velocity_field[-1, :, :, 0])
        self.velocity_field[:, 0, :, 1] = np.maximum(0.0, self.velocity_field[:, 0, :, 1])
        self.velocity_field[:, -1, :, 1] = np.minimum(0.0, self.velocity_field[:, -1, :, 1])

        # Approximate divergence-free projection (pressure Poisson relaxation)
        # div u = du/dx + dv/dy + dw/dz
        for _ in range(3):
            div = (
                (np.roll(self.velocity_field[..., 0], -1, axis=0) - np.roll(self.velocity_field[..., 0], 1, axis=0)) / (2 * self.dx) +
                (np.roll(self.velocity_field[..., 1], -1, axis=1) - np.roll(self.velocity_field[..., 1], 1, axis=1)) / (2 * self.dy) +
                (np.roll(self.velocity_field[..., 2], -1, axis=2) - np.roll(self.velocity_field[..., 2], 1, axis=2)) / (2 * self.dz)
            )
            # Pressure gradient correction (1 step jacobi relaxation)
            p = div * 0.05
            self.velocity_field[1:-1, :, :, 0] -= (p[2:, :, :] - p[:-2, :, :]) / (2 * self.dx)
            self.velocity_field[:, 1:-1, :, 1] -= (p[:, 2:, :] - p[:, :-2, :]) / (2 * self.dy)
            self.velocity_field[:, :, 1:-1, 2] -= (p[:, :, 2:] - p[:, :, :-2]) / (2 * self.dz)
            self.velocity_field[..., :, :, 0, 2] = np.maximum(0.0, self.velocity_field[..., :, :, 0, 2])

        # 4. Advect fields
        self.transported_scalar = self._advect_field_semi_lagrangian(self.transported_scalar, dt)
        self.temperature_field = self._advect_field_semi_lagrangian(self.temperature_field, dt)
        self.velocity_field = self._advect_field_semi_lagrangian(self.velocity_field, dt)

        # Dissipation / cooling
        self.temperature_field *= np.exp(-0.15 * dt)
        self.transported_scalar *= np.exp(-0.02 * dt)

        # Scalar source active during early blast (t < 0.5s)
        if t < 0.5:
            source_strength = 2.0 * np.exp(-((r_blast / 0.35)**2)) * (1.0 - t / 0.5)
            self.scalar_source = source_strength
            self.transported_scalar += source_strength * dt
            self.temperature_field += 4.0 * source_strength * dt
        else:
            self.scalar_source = np.zeros_like(self.scalar_source)

        # 5. Advect and evolve vortex particles
        if len(self.vortex_pos) > 0:
            vp_vel = self._interp_to_grid(self.velocity_field, self.vortex_pos)
            self.vortex_pos += vp_vel * dt
            # Vortex stretching / buoyant vorticity generation:
            # Rising smoke curls outward at the top of plume
            for i in range(len(self.vortex_pos)):
                pz = self.vortex_pos[i, 2]
                pr = np.sqrt(self.vortex_pos[i, 0]**2 + self.vortex_pos[i, 1]**2) + 1e-4
                # Tangential vortex ring generation
                tangent = np.array([-self.vortex_pos[i, 1] / pr, self.vortex_pos[i, 0] / pr, 0.0])
                if pz > self.blast_pos[2]:
                    # Mushroom cap vortex roll-up
                    self.vortex_gamma[i] = 0.98 * self.vortex_gamma[i] + 0.02 * (3.0 * tangent)
            # Bound vortex particles
            self.vortex_pos[:, 0] = np.clip(self.vortex_pos[:, 0], self.bounds_min[0] + 0.1, self.bounds_max[0] - 0.1)
            self.vortex_pos[:, 1] = np.clip(self.vortex_pos[:, 1], self.bounds_min[1] + 0.1, self.bounds_max[1] - 0.1)
            self.vortex_pos[:, 2] = np.clip(self.vortex_pos[:, 2], 0.05, self.bounds_max[2] - 0.1)

        # 6. Advect Lagrangian scene particles
        if len(self.particle_pos) > 0:
            p_vel = self._interp_to_grid(self.velocity_field, self.particle_pos)
            self.particle_vel = p_vel
            self.particle_pos += p_vel * dt
            # Ground collision
            self.particle_pos[:, 2] = np.maximum(self.particle_pos[:, 2], 0.01)
            self.particle_pos[:, 0] = np.clip(self.particle_pos[:, 0], self.bounds_min[0] + 0.05, self.bounds_max[0] - 0.05)
            self.particle_pos[:, 1] = np.clip(self.particle_pos[:, 1], self.bounds_min[1] + 0.05, self.bounds_max[1] - 0.05)

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step physics up to target time
        while self.current_time + 1e-6 < time:
            step = min(self.sub_dt, time - self.current_time)
            if step <= 1e-7:
                break
            self._step_physics(step)

        self.current_time = time

        # Build StateFrame
        fields = {
            "active_mask": self.active_mask.astype(bool),
            "boundary_normal": self.boundary_normal.astype(np.float64),
            "boundary_target_velocity": self.boundary_target_velocity.astype(np.float64),
            "boundary_velocity": self.boundary_velocity.astype(np.float64),
            "scalar_source": self.scalar_source.astype(np.float64),
            "transported_scalar": self.transported_scalar.astype(np.float64),
            "velocity_field": self.velocity_field.astype(np.float64),
        }

        scalars = {
            "characteristic_length": float(max(np.ptp(self.particle_pos, axis=0).max() if len(self.particle_pos) > 0 else 1.0, self.dx_avg)),
            "grid_spacing": float(self.dx_avg),
            "reference_velocity": 1.0,
        }

        attributes = {
            "mass_quantity": self.particle_mass.copy(),
            "particle_radius": self.particle_radius.copy(),
            "material_label": self.particle_labels.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], len(self.particle_pos), axis=0),
        }

        frame = StateFrame(
            self.current_time,
            self.particle_pos.copy(),
            self.particle_vel.copy(),
            mass=self.particle_mass.copy(),
            body_id=self.particle_labels.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_vortices": len(self.vortex_pos) if self.vortex_pos is not None else 0,
            "num_particles": len(self.particle_pos) if self.particle_pos is not None else 0,
            "max_scalar": float(np.max(self.transported_scalar)) if self.transported_scalar is not None else 0.0,
            "max_velocity": float(np.max(np.linalg.norm(self.velocity_field, axis=-1))) if self.velocity_field is not None else 0.0,
        }

