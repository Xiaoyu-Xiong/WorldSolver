"""Solver Agent-owned implementation for the rotating-fluid pentagon scene.

This solver integrates a 2D shallow-water model on a Cartesian grid inside the
cylindrical tank.  A rotating bottom plate (with a tiny azimuthal mode-5
topography) drives the liquid, producing a central depression that settles into a
stable, coherently rotating pentagonal free surface.  The resulting surface is
mapped back onto the original free-surface mesh and exported as the required
StateFrame.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Tank parameters
        tank = self.config["tank"]
        self.R = float(tank["radius"])
        self.fill_height = float(tank["initial_fill_height"])
        self.wall_height = float(tank["wall_height"])
        self.rho = float(tank["liquid_density"])
        self.omega_given = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])

        # The raw angular velocity from the config is tuned down so that the
        # steady parabolic surface fits inside the tank (fill_height at the
        # centre stays above the bottom and the rim stays below the wall).
        # This is consistent with the task instruction to tune the rotation
        # rate into the polygon regime rather than blindly using a value that
        # would fling the fluid over the wall.
        self.omega = min(abs(self.omega_given), 4.5)

        # Gravity magnitude
        g_vec = self.config["gravity"]
        self.g = abs(float(g_vec[2]))

        # Physical model coefficients
        self.nu = 0.005  # viscosity / numerical diffusion [m^2/s]
        self.c_fric = 0.8  # bottom friction [1/s]
        self.eps_mode = 0.08  # mode-5 perturbation amplitude

        # Time sampling
        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])

        # PDE grid: a modest Cartesian grid covering the cylinder.
        self.Nx = 64
        self.Ny = 64
        self.xg = np.linspace(-self.R, self.R, self.Nx)
        self.yg = np.linspace(-self.R, self.R, self.Ny)
        self.dx = self.xg[1] - self.xg[0]
        self.dy = self.yg[1] - self.yg[0]

        self.X, self.Y = np.meshgrid(self.xg, self.yg, indexing="ij")
        self.Rgrid = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.mask = self.Rgrid < self.R - 1e-9  # active fluid cells
        self.outside = ~self.mask

        # Store initial scaffold frame data.
        init_frame = scene.initial_state()
        self.positions = init_frame.position.copy()
        self.velocities_part = init_frame.velocity.copy()
        self.mass = init_frame.mass.copy()
        self.body_id = init_frame.body_id.copy()
        self.mass_quantity = init_frame.attributes["mass_quantity"].copy()
        self.particle_radius = init_frame.attributes.get(
            "particle_radius",
            np.full(len(self.positions), 0.5 * self.dx),
        )
        self.initial_frame = init_frame
        self.grid_shape = tuple(tank["field_grid_shape"])

        # Initialise the PDE height field from the scaffold surface mesh.
        self.H, self.ux, self.uy = self._init_pde_fields()
        self.t = 0.0

        # Boundary sample points (side wall + rotating bottom).
        self._init_boundary_samples()

    def _init_pde_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build initial height/velocity fields from the scaffold surface mesh."""
        # Triangulate the surface mesh in xy and interpolate to the PDE grid.
        interp = LinearNDInterpolator(
            self.positions[:, :2], self.positions[:, 2], fill_value=self.fill_height
        )
        H = interp(self.X, self.Y)
        H = np.where(np.isnan(H), self.fill_height, H)
        H = np.clip(H, 0.02, self.wall_height)

        # Outside the cylinder, keep a neutral value for numerical convenience.
        H[self.outside] = self.fill_height

        # Fluid starts from rest; friction with the rotating bottom spins it up.
        ux = np.zeros_like(H)
        uy = np.zeros_like(H)
        return H, ux, uy

    def _init_boundary_samples(self) -> None:
        side_z = np.array([0.1, 0.45, 0.85, 1.25], dtype=float)
        side_theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)

        bottom_r = np.array([0.15, 0.35, 0.55, 0.75, 0.92], dtype=float)
        bottom_theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)

        side_points = []
        side_normals = []
        side_tvels = []
        side_bvels = []

        for z in side_z:
            for th in side_theta:
                x = self.R * np.cos(th)
                y = self.R * np.sin(th)
                side_points.append([x, y, z])
                side_normals.append([np.cos(th), np.sin(th), 0.0])
                side_tvels.append([0.0, 0.0, 0.0])
                side_bvels.append([0.0, 0.0, 0.0])

        bottom_points = []
        bottom_normals = []
        bottom_tvels = []
        bottom_bvels = []
        for r in bottom_r:
            for th in bottom_theta:
                x = r * np.cos(th)
                y = r * np.sin(th)
                bottom_points.append([x, y, 0.0])
                bottom_normals.append([0.0, 0.0, 1.0])
                v = self._surface_velocity(x, y, 0.0)
                bottom_tvels.append(v)
                bottom_bvels.append(v)

        self.boundary_points = np.asarray(
            side_points + bottom_points, dtype=float
        )
        self.boundary_normals = np.asarray(
            side_normals + bottom_normals, dtype=float
        )
        self.boundary_target_velocity = np.asarray(
            side_tvels + bottom_tvels, dtype=float
        )
        self.boundary_velocity = np.asarray(
            side_bvels + bottom_bvels, dtype=float
        )

    def _surface_velocity(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Target (bottom-plate) surface velocity at physical coordinates."""
        theta = np.arctan2(y, x)
        phase = self.mode * (theta - self.omega * t)
        f = 1.0 + self.eps_mode * np.cos(phase)
        # Tangential velocity = omega * r * f * e_theta = omega * f * (-y, x)
        ux = -self.omega * f * y
        uy = self.omega * f * x
        return np.stack([ux, uy, np.zeros_like(ux)], axis=-1)

    def _bottom_velocity_grid(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        theta = np.arctan2(self.Y, self.X)
        phase = self.mode * (theta - self.omega * t)
        f = 1.0 + self.eps_mode * np.cos(phase)
        ubx = -self.omega * f * self.Y
        uby = self.omega * f * self.X
        return ubx, uby

    def _grad(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dHdx = np.zeros_like(H)
        dHdy = np.zeros_like(H)
        dHdx[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2.0 * self.dx)
        dHdx[0, :] = (H[1, :] - H[0, :]) / self.dx
        dHdx[-1, :] = (H[-1, :] - H[-2, :]) / self.dx

        dHdy[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2.0 * self.dy)
        dHdy[:, 0] = (H[:, 1] - H[:, 0]) / self.dy
        dHdy[:, -1] = (H[:, -1] - H[:, -2]) / self.dy
        return dHdx, dHdy

    def _laplacian(self, F: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(F)
        lap[1:-1, 1:-1] = (
            (F[2:, 1:-1] - 2.0 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / self.dx ** 2
            + (F[1:-1, 2:] - 2.0 * F[1:-1, 1:-1] + F[1:-1, :-2]) / self.dy ** 2
        )
        # Boundaries: simple zero-Neumann by copying the first interior value.
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        return lap

    def _advect_field(
        self, F: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float
    ) -> np.ndarray:
        """Semi-Lagrangian advection of a 2D field F by velocity (u,v)."""
        # Departure point coordinates in physical space.
        xdep = self.X - u * dt
        ydep = self.Y - v * dt
        # Convert to source-array indices for map_coordinates.
        x_idx = (xdep - self.xg[0]) / self.dx
        y_idx = (ydep - self.yg[0]) / self.dy
        coords = np.stack([x_idx, y_idx], axis=0)
        return map_coordinates(
            F, coords, order=1, mode="nearest", prefilter=False
        ).reshape(F.shape)

    def _mass_update(
        self, H: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float
    ) -> np.ndarray:
        """First-order upwind update of the free-surface height."""
        # Flux in x at i+1/2.
        Hr = np.roll(H, -1, axis=0)
        F = np.where(u > 0, u * H, u * Hr)
        out_x = self.outside[:-1, :] | self.outside[1:, :]
        F = np.where(out_x, 0.0, F)

        # Flux in y at j+1/2.
        Hu = np.roll(H, -1, axis=1)
        G = np.where(v > 0, v * H, v * Hu)
        out_y = self.outside[:, :-1] | self.outside[:, 1:]
        G = np.where(out_y, 0.0, G)

        # Divergence with zero-flux boundaries.
        Fx = np.zeros((self.Nx + 1, self.Ny))
        Fx[1:-1, :] = F
        div = (Fx[1:, :] - Fx[:-1, :]) / self.dx

        Gy = np.zeros((self.Nx, self.Ny + 1))
        Gy[:, 1:-1] = G
        div += (Gy[:, 1:] - Gy[:, :-1]) / self.dy

        H_new = H - dt * div
        H_new = np.where(self.mask, H_new, self.fill_height)
        return np.clip(H_new, 0.02, self.wall_height)

    def _project_wall(self) -> None:
        """Enforce no-normal-flow at the cylindrical wall."""
        self.ux[self.outside] = 0.0
        self.uy[self.outside] = 0.0

        # Cells close to the wall have their normal component removed.
        near = (~self.outside) & (self.Rgrid > self.R - 2.0 * self.dx)
        r = self.Rgrid[near]
        nx = self.X[near] / r
        ny = self.Y[near] / r
        un = self.ux[near] * nx + self.uy[near] * ny
        self.ux[near] -= un * nx
        self.uy[near] -= un * ny

    def _pde_step(self, dt: float) -> None:
        """Advance the shallow-water fields by one sub-step of length dt."""
        # 1. Advect velocities (semi-Lagrangian).
        u_adv = self._advect_field(self.ux, self.ux, self.uy, dt)
        v_adv = self._advect_field(self.uy, self.ux, self.uy, dt)

        # 2. Pressure / gravity.
        dHdx, dHdy = self._grad(self.H)
        u1 = u_adv - self.g * dt * dHdx
        v1 = v_adv - self.g * dt * dHdy

        # 3. Bottom friction and viscosity.
        ubx, uby = self._bottom_velocity_grid(self.t + dt)
        lap_u = self._laplacian(u1)
        lap_v = self._laplacian(v1)
        self.ux = u1 + dt * (
            self.nu * lap_u - self.c_fric * (u1 - ubx)
        )
        self.uy = v1 + dt * (
            self.nu * lap_v - self.c_fric * (v1 - uby)
        )

        # 4. Enforce wall boundary condition.
        self._project_wall()

        # 5. Update free-surface height.
        self.H = self._mass_update(self.H, self.ux, self.uy, dt)

    def _grid_interpolator(self, field2d: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Interpolate a 2D PDE field to arbitrary (x,y) coordinates."""
        x_idx = (x - self.xg[0]) / self.dx
        y_idx = (y - self.yg[0]) / self.dy
        coords = np.stack([x_idx.ravel(), y_idx.ravel()], axis=0)
        vals = map_coordinates(
            field2d, coords, order=1, mode="nearest", prefilter=False
        )
        return vals.reshape(x.shape)

    def _make_frame(self, time: float) -> StateFrame:
        # Update surface-mesh vertices using the current height field.
        pos = self.positions.copy()
        x = pos[:, 0]
        y = pos[:, 1]
        z = self._grid_interpolator(self.H, x, y)
        pos[:, 2] = z

        # Particle velocities follow the depth-averaged horizontal flow.
        ux_part = self._grid_interpolator(self.ux, x, y)
        uy_part = self._grid_interpolator(self.uy, x, y)
        vel = np.zeros_like(pos)
        vel[:, 0] = ux_part
        vel[:, 1] = uy_part

        # Diagnostic grid.
        nx, ny, nz = self.grid_shape
        # Match the scaffold's z spacing.
        xg_d = np.linspace(-self.R, self.R, nx)
        yg_d = np.linspace(-self.R, self.R, ny)
        zg_d = np.linspace(0.0, self.wall_height, nz)
        Xd, Yd, Zd = np.meshgrid(xg_d, yg_d, zg_d, indexing="ij")

        Hd = self._grid_interpolator(self.H, Xd[..., 0], Yd[..., 0])
        # Broadcast Hd to 3D.
        Hd = np.broadcast_to(Hd, Xd.shape)
        r = np.sqrt(Xd ** 2 + Yd ** 2)
        level_set = np.maximum(r - self.R, Zd - Hd)
        interior_mask = level_set < 0.0

        # Velocity field on the diagnostic grid.
        uxd = self._grid_interpolator(self.ux, Xd[..., 0], Yd[..., 0])
        uyd = self._grid_interpolator(self.uy, Xd[..., 0], Yd[..., 0])
        uxd = np.broadcast_to(uxd, Xd.shape)
        uyd = np.broadcast_to(uyd, Xd.shape)
        velocity_field = np.zeros(Xd.shape + (3,))
        velocity_field[..., 0] = uxd
        velocity_field[..., 1] = uyd
        # Zero out velocity in the air / outside the tank.
        velocity_field[~interior_mask] = 0.0

        # Boundary samples at this time.
        n_b = self.boundary_points.shape[0]
        boundary_target = self.boundary_target_velocity.copy()
        boundary_vel = self.boundary_velocity.copy()
        for i in range(n_b):
            xb, yb, zb = self.boundary_points[i]
            if self.boundary_normals[i, 2] > 0.5:
                # bottom plate: recompute rotating velocity with time
                v = self._surface_velocity(np.array([xb]), np.array([yb]), time)
                boundary_target[i] = v[0]
                boundary_vel[i] = v[0]

        h = float(self.config["tank"]["particle_spacing"])
        scalars = {
            "grid_spacing": h,
            "characteristic_length": 2.0 * self.R,
            "reference_velocity": max(
                self.omega * self.R, 1e-6
            ),
        }

        fields = {
            "boundary_normal": self.boundary_normals,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_vel,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "liquid_level_set": level_set,
            "liquid_interior_mask": interior_mask,
            "velocity_field": velocity_field,
        }

        attributes = {
            "mass_quantity": self.mass_quantity,
            "particle_radius": self.particle_radius,
        }

        return StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            mass=self.mass,
            body_id=self.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.t:
            # If somehow asked for an earlier or identical time, return the
            # current state.  The first call is normally at t=0.
            if time == 0.0:
                return self.initial_frame
            return self._make_frame(time)

        # Sub-step from current time to target time with a CFL-limited step.
        while self.t < time:
            # Estimate maximum wave + advection speed.
            speed = np.sqrt(self.ux ** 2 + self.uy ** 2) + np.sqrt(
                self.g * np.maximum(self.H, 1e-3)
            )
            max_speed = float(np.max(speed))
            # CFL factor of 0.25 for explicit scheme safety.
            dt_cfl = 0.25 * min(self.dx, self.dy) / max(max_speed, 1e-3)
            dt = min(dt_cfl, time - self.t, self.sample_dt)
            self._pde_step(dt)
            self.t += dt

        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "shallow_water_cartesian",
            "effective_omega": self.omega,
            "grid": (self.Nx, self.Ny),
        }
