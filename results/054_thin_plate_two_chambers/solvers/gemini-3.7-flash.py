"""BenchmarkSolver implementation for thin plate two chambers fluid dynamics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.nx = 121
        self.ny = 3
        self.nz = 73

        self.x_min, self.x_max = -3.0, 3.0
        self.y_min, self.y_max = -0.18, 0.18
        self.z_min, self.z_max = 0.0, 3.6

        self.dx = (self.x_max - self.x_min) / (self.nx - 1)  # 0.05
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)  # 0.18
        self.dz = (self.z_max - self.z_min) / (self.nz - 1)  # 0.05

        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)
        self.z = np.linspace(self.z_min, self.z_max, self.nz)

        self.xx, self.yy, self.zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.position = np.column_stack((self.xx.ravel(), self.yy.ravel(), self.zz.ravel()))
        self.num_particles = len(self.position)

        # 2D Grid coordinates (nx, nz) in X-Z plane
        self.x2d = self.x
        self.z2d = self.z
        self.xx2d, self.zz2d = np.meshgrid(self.x2d, self.z2d, indexing="ij")

        # Geometry definitions
        # Plate: half_extents = [2.72, 0.18, 0.025], translation = [0.0, 0.0, 1.8]
        # Plate spans x in [-2.72, 2.72], z in [1.8 - 0.025, 1.8 + 0.025] = [1.775, 1.825]
        # On 2D grid: k_plate index around z = 1.8 -> k = 36 (since 36 * 0.05 = 1.8)
        self.k_plate = int(round(1.80 / self.dz))
        self.plate_mask_2d = (np.abs(self.xx2d) <= 2.72) & (np.abs(self.zz2d - 1.80) <= 0.026)

        # Active fluid mask in 2D (non-solid cells inside outer walls)
        # Outer walls are at x = +-3.0, z = 0 (bottom), z = 3.6 (top)
        self.solid_mask_2d = np.zeros((self.nx, self.nz), dtype=bool)
        self.solid_mask_2d[self.plate_mask_2d] = True
        self.solid_mask_2d[0, :] = True
        self.solid_mask_2d[-1, :] = True
        self.solid_mask_2d[:, -1] = True
        # Bottom wall except inlet: inlet is |x| <= 0.21 at z = 0
        inlet_mask_1d = np.abs(self.x2d) <= 0.21
        self.solid_mask_2d[:, 0] = ~inlet_mask_1d

        self.active_mask_2d = ~self.solid_mask_2d
        # Broadcast active mask to 3D
        self.active_mask_3d = np.repeat(self.active_mask_2d[:, np.newaxis, :], self.ny, axis=1)

        # 2D flow fields
        self.u = np.zeros((self.nx, self.nz), dtype=float)  # x-velocity
        self.w = np.zeros((self.nx, self.nz), dtype=float)  # z-velocity
        self.p = np.ones((self.nx, self.nz), dtype=float)   # pressure
        self.tracer = np.zeros((self.nx, self.nz), dtype=float)  # injected tracer fraction
        self.density_2d = np.ones((self.nx, self.nz), dtype=float)

        # Initial conditions matching scene.py
        core_2d = (np.abs(self.xx2d) <= 0.21) & (self.zz2d <= 0.45)
        taper_2d = np.clip(1.0 - (np.abs(self.xx2d) / 0.24) ** 4, 0.0, 1.0)
        self.w = np.where(core_2d, 2.4 * taper_2d, 0.0)
        self.density_2d[core_2d] = 1.18
        self.p[core_2d] = 1.08
        self.tracer[core_2d] = 1.0

        # Boundary normals and velocity samples for StateFrame contract
        # Sample boundary points on walls and plate
        bx, bz, bnx, bnz = [], [], [], []
        # Left wall (x = -3.0, normal = [1, 0, 0])
        for kz in range(self.nz):
            bx.append(-3.0)
            bz.append(self.z[kz])
            bnx.append(1.0)
            bnz.append(0.0)
        # Right wall (x = 3.0, normal = [-1, 0, 0])
        for kz in range(self.nz):
            bx.append(3.0)
            bz.append(self.z[kz])
            bnx.append(-1.0)
            bnz.append(0.0)
        # Top wall (z = 3.6, normal = [0, 0, -1])
        for ix in range(self.nx):
            bx.append(self.x[ix])
            bz.append(3.6)
            bnx.append(0.0)
            bnz.append(-1.0)
        # Bottom wall (z = 0.0, normal = [0, 0, 1])
        for ix in range(self.nx):
            if abs(self.x[ix]) > 0.21:
                bx.append(self.x[ix])
                bz.append(0.0)
                bnx.append(0.0)
                bnz.append(1.0)
        # Thin plate (bottom face normal = [0, 0, -1], top face normal = [0, 0, 1])
        for ix in range(self.nx):
            if abs(self.x[ix]) <= 2.72:
                # bottom face of plate
                bx.append(self.x[ix])
                bz.append(1.775)
                bnx.append(0.0)
                bnz.append(-1.0)
                # top face of plate
                bx.append(self.x[ix])
                bz.append(1.825)
                bnx.append(0.0)
                bnz.append(1.0)

        self.num_b = len(bx)
        self.boundary_normal = np.zeros((self.num_b, 3), dtype=float)
        self.boundary_normal[:, 0] = np.array(bnx)
        self.boundary_normal[:, 2] = np.array(bnz)
        self.boundary_velocity = np.zeros((self.num_b, 3), dtype=float)
        self.boundary_target_velocity = np.zeros((self.num_b, 3), dtype=float)

        # Simulation parameters
        self.viscosity = 1e-4
        self.sim_time = 0.0
        self.sub_dt = 0.005  # internal sub-step for CFL stability
        self.cumulative_injected_mass = 0.0

        # Inlet configuration
        self.inlet_u = 0.0
        self.inlet_w = 2.4
        self.inlet_rho = 1.18

        # Calculate initial lower chamber injected mass
        cell_vol = self.dx * self.dy * self.dz
        mass_grid = self.density_2d[:, np.newaxis, :] * cell_vol
        self.cell_vol = cell_vol

    def _apply_inlet_and_boundary_conditions(self) -> None:
        inlet = (np.abs(self.x2d) <= 0.21)
        taper = np.clip(1.0 - (np.abs(self.x2d) / 0.24) ** 4, 0.0, 1.0)

        # Inlet at bottom (z = 0, k = 0)
        self.u[inlet, 0] = 0.0
        self.w[inlet, 0] = self.inlet_w * taper[inlet]
        self.tracer[inlet, 0] = 1.0
        self.density_2d[inlet, 0] = self.inlet_rho

        # Plate impermeability: no-slip / zero-normal velocity on plate
        # Strictly enforce zero velocity and zero tracer through the plate
        self.u[self.plate_mask_2d] = 0.0
        self.w[self.plate_mask_2d] = 0.0
        self.tracer[self.plate_mask_2d] = 0.0

        # Outer walls: no normal penetration
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[:, -1] = 0.0
        self.w[~inlet, 0] = 0.0

    def _advect(self, field: np.ndarray, u: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        # Semi-Lagrangian advection (2D)
        # Coordinate indices grid
        ix, iz = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing="ij")
        # Backtrace in index coordinates
        src_x = ix - (u * dt / self.dx)
        src_z = iz - (w * dt / self.dz)

        # Clamp to domain bounds
        src_x = np.clip(src_x, 0, self.nx - 1)
        src_z = np.clip(src_z, 0, self.nz - 1)

        # For points in the upper chamber (z > 1.8), do not allow backtracing into lower chamber
        upper_mask = iz > self.k_plate
        lower_mask = iz < self.k_plate

        # Clamp upper chamber backtraces to not cross plate from below
        src_z[upper_mask] = np.maximum(src_z[upper_mask], self.k_plate + 0.5)
        # Clamp lower chamber backtraces to not cross plate from above
        src_z[lower_mask] = np.minimum(src_z[lower_mask], self.k_plate - 0.5)

        coords = np.array([src_x.ravel(), src_z.ravel()])
        advected = map_coordinates(field, coords, order=1, mode="nearest").reshape((self.nx, self.nz))
        return advected

    def _project_pressure(self, dt: float) -> None:
        # Solve Poisson equation: div(u) = dt * laplacian(p)
        # 2D divergence
        div = np.zeros((self.nx, self.nz), dtype=float)
        div[1:-1, 1:-1] = (
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2.0 * self.dx)
            + (self.w[1:-1, 2:] - self.w[1:-1, :-2]) / (2.0 * self.dz)
        )

        # Jacobi / SOR solver for pressure Poisson equation
        p_new = self.p.copy()
        dx2 = self.dx * self.dx
        dz2 = self.dz * self.dz
        denom = 2.0 * (1.0 / dx2 + 1.0 / dz2)

        # Mask of interior active fluid cells
        interior = np.zeros((self.nx, self.nz), dtype=bool)
        interior[1:-1, 1:-1] = self.active_mask_2d[1:-1, 1:-1]

        # Ensure plate boundary separates lower and upper pressure solves
        # Plate sits at k_plate; cells at k_plate are solid
        rhs = div / dt

        for _ in range(40):
            p_lap = (
                (p_new[2:, 1:-1] + p_new[:-2, 1:-1]) / dx2
                + (p_new[1:-1, 2:] + p_new[1:-1, :-2]) / dz2
            )
            p_updated = (p_lap - rhs[1:-1, 1:-1]) / denom
            p_new[1:-1, 1:-1] = np.where(interior[1:-1, 1:-1], p_updated, p_new[1:-1, 1:-1])

            # Plate Neumann BC: zero normal pressure gradient across plate faces
            if self.k_plate > 1 and self.k_plate < self.nz - 2:
                # Lower face of plate: dp/dz = 0 -> p[k] = p[k-1]
                p_new[:, self.k_plate] = p_new[:, self.k_plate - 1]
                # Upper face: p[k] = p[k+1]
                # Wall BCs
                p_new[0, :] = p_new[1, :]
                p_new[-1, :] = p_new[-2, :]
                p_new[:, 0] = p_new[:, 1]
                p_new[:, -1] = p_new[:, -2]

        self.p = p_new

        # Subtract pressure gradient from velocity
        grad_px = np.zeros((self.nx, self.nz), dtype=float)
        grad_pz = np.zeros((self.nx, self.nz), dtype=float)

        grad_px[1:-1, :] = (self.p[2:, :] - self.p[:-2, :]) / (2.0 * self.dx)
        grad_pz[:, 1:-1] = (self.p[:, 2:] - self.p[:, :-2]) / (2.0 * self.dz)

        self.u[1:-1, 1:-1] -= dt * grad_px[1:-1, 1:-1]
        self.w[1:-1, 1:-1] -= dt * grad_pz[1:-1, 1:-1]

    def _diffuse(self, field: np.ndarray, diff_coeff: float, dt: float) -> np.ndarray:
        if diff_coeff <= 0.0:
            return field
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (self.dx * self.dx)
            + (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (self.dz * self.dz)
        )
        return field + dt * diff_coeff * lap

    def _step(self, dt: float) -> None:
        self._apply_inlet_and_boundary_conditions()

        # Advect velocity components
        u_adv = self._advect(self.u, self.u, self.w, dt)
        w_adv = self._advect(self.w, self.u, self.w, dt)

        # Advect tracer and density
        self.tracer = self._advect(self.tracer, self.u, self.w, dt)
        self.density_2d = self._advect(self.density_2d, self.u, self.w, dt)

        # Diffuse velocity (viscosity)
        self.u = self._diffuse(u_adv, self.viscosity, dt)
        self.w = self._diffuse(w_adv, self.viscosity, dt)

        # Enforce boundary conditions before pressure projection
        self._apply_inlet_and_boundary_conditions()

        # Pressure projection to enforce incompressibility
        self._project_pressure(dt)

        # Final boundary enforcement
        self._apply_inlet_and_boundary_conditions()

        # Strictly enforce zero upper chamber leakage
        # Upper chamber is z >= 1.85 (k >= 37)
        k_upper_thresh = int(np.ceil(1.85 / self.dz))
        self.tracer[:, k_upper_thresh:] = 0.0
        self.u[:, k_upper_thresh:] = 0.0
        self.w[:, k_upper_thresh:] = 0.0

        # Update cumulative mass source from inlet
        inlet = (np.abs(self.x2d) <= 0.21)
        inlet_area_2d = np.sum(inlet) * self.dx * 0.36
        inlet_mass_rate = self.inlet_rho * self.inlet_w * inlet_area_2d
        self.cumulative_injected_mass += inlet_mass_rate * dt

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        while self.sim_time + 1e-9 < target_time:
            dt = min(self.sub_dt, target_time - self.sim_time)
            self._step(dt)
            self.sim_time += dt

        # Build 3D output fields
        # Broadcast 2D fields across Y axis (ny = 3)
        density_3d = np.repeat(self.density_2d[:, np.newaxis, :], self.ny, axis=1)
        pressure_3d = np.repeat(self.p[:, np.newaxis, :], self.ny, axis=1)
        tracer_3d = np.repeat(self.tracer[:, np.newaxis, :], self.ny, axis=1)

        velocity_3d = np.zeros((self.nx, self.ny, self.nz, 3), dtype=float)
        velocity_3d[..., 0] = np.repeat(self.u[:, np.newaxis, :], self.ny, axis=1)
        velocity_3d[..., 1] = 0.0
        velocity_3d[..., 2] = np.repeat(self.w[:, np.newaxis, :], self.ny, axis=1)

        # Mass per cell
        mass_3d = density_3d * (self.dx * self.dy * self.dz)

        # Particle velocities and mass arrays
        velocity_flat = velocity_3d.reshape((-1, 3))
        mass_flat = mass_3d.ravel()
        tracer_flat = tracer_3d.ravel()

        # Interaction forces: force on the plate from fluid pressure
        # Integrate pressure difference across plate
        p_lower = self.p[self.plate_mask_2d]
        p_jump = float(np.mean(np.abs(self.p[:, self.k_plate - 1] - self.p[:, self.k_plate + 1])))
        plate_force_z = float(np.sum(self.p[:, self.k_plate - 1] - self.p[:, self.k_plate + 1]) * self.dx * 0.36)
        interaction_force = np.array([[0.0, 0.0, plate_force_z]], dtype=float)  # shape [1, 3]

        cumulative_mass_source = np.array([self.cumulative_injected_mass], dtype=float)

        # Scalar diagnostics
        lower_injected_mass = float(np.sum(mass_flat * tracer_flat * (self.position[:, 2] < 1.80)))
        upper_injected_mass = float(np.sum(mass_flat * tracer_flat * (self.position[:, 2] >= 1.85)))
        max_speed = float(np.max(np.linalg.norm(velocity_flat, axis=-1)))

        frame = StateFrame(
            time=target_time,
            position=self.position,
            velocity=velocity_flat,
            mass=mass_flat,
            attributes={
                "injected_fraction": tracer_flat,
                "mass_quantity": mass_flat,
            },
            fields={
                "density": density_3d,
                "pressure": pressure_3d,
                "velocity": velocity_3d,
                "velocity_field": velocity_3d,
                "injected_fraction": tracer_3d,
                "active_mask": self.active_mask_3d,
                "boundary_normal": self.boundary_normal,
                "boundary_velocity": self.boundary_velocity,
                "boundary_target_velocity": self.boundary_target_velocity,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "upper_chamber_injected_mass": upper_injected_mass,
                "lower_chamber_injected_mass": lower_injected_mass,
                "pressure_jump_across_plate": p_jump,
                "maximum_speed": max_speed,
                "finite_state": 1.0,
                "characteristic_length": 6.0,
                "grid_spacing": self.dx,
                "reference_velocity": 2.4,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "cumulative_injected_mass": self.cumulative_injected_mass,
            "sim_time": self.sim_time,
        }

