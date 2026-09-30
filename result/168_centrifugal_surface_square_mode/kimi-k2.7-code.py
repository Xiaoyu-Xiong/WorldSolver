"""Rotating free-surface solver for the centrifugal polygon mode.

The solver integrates the 2D depth-averaged (shallow-water) equations on a
Cartesian grid that spans the cylinder cross-section.  The rotating bottom
plate drives the fluid through a drag term; gravity provides the free-surface
restoring force.  The 3D signed level set, interior mask, diagnostic velocity
field, and wall boundary data are derived from the 2D state at every sample
time.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Finite-volume shallow-water solver with bottom-plate driving."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        tank = self.config["tank"]
        self.R = float(tank["radius"])
        self.H_fill = float(tank["initial_fill_height"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.mode = int(tank["target_mode"])
        self.rho = float(tank["liquid_density"])
        self.h = float(tank["particle_spacing"])
        self.wall_height = float(tank["wall_height"])
        self.grid_shape = tuple(int(x) for x in tank["field_grid_shape"])
        self.nx, self.ny, self.nz = self.grid_shape
        self.g = 9.81

        # Horizontal grid (covers the cylinder diameter)
        self.x = np.linspace(-self.R, self.R, self.nx)
        self.y = np.linspace(-self.R, self.R, self.ny)
        self.z = np.linspace(0.0, self.wall_height, self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.R2d = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.cyl_mask = self.R2d <= self.R

        # Initial frame from scaffold (particles, masses, default fields/scalars)
        self.frame0 = scene.initial_state()
        self.particle_pos = self.frame0.position.copy()
        self.particle_vel = self.frame0.velocity.copy()
        self.particle_mass = self.frame0.mass.copy()
        self.particle_body_id = self.frame0.body_id.copy()
        self.attributes = dict(self.frame0.attributes)
        self.scalars = dict(self.frame0.scalars)

        # Ensure all required scalars are present.
        self.scalars.setdefault("characteristic_length", 2.0 * self.R)
        self.scalars.setdefault("grid_spacing", self.h)
        self.scalars.setdefault(
            "reference_velocity", max(self.omega_target * self.R, 1e-6)
        )

        # 2D prognostic state
        self.eta = self._initial_surface()
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))

        # Wall boundary sample points with normals.
        self._build_boundary_samples()

        # Internal time
        self.current_time = 0.0

    # ------------------------------------------------------------------
    # Initial condition / geometry helpers
    # ------------------------------------------------------------------
    def _initial_surface(self) -> np.ndarray:
        """Reconstruct the seeded mode-4 free surface used by the scaffold."""
        mode = self.mode
        R = self.R
        H = self.H_fill
        ratio = np.clip(self.R2d / R, 0.0, 1.0)
        angle = np.arctan2(self.Y, self.X)
        depression = 0.16 * (1.0 - ratio ** 2) ** 2
        seeded = 0.035 * 4.0 * ratio * (1.0 - ratio) * np.cos(mode * angle)
        eta = H - depression - seeded
        eta = np.maximum(eta, 1e-3)
        eta = np.where(self.cyl_mask, eta, 0.0)
        return eta

    def _build_boundary_samples(self) -> None:
        n_side = 48
        n_bottom_r = 6
        n_bottom_theta = 24
        n_bottom = n_bottom_r * n_bottom_theta
        B = n_side + n_bottom

        pts = np.zeros((B, 3))
        normals = np.zeros((B, 3))
        is_bottom = np.zeros(B, dtype=bool)

        # Stationary cylindrical side wall
        for i in range(n_side):
            theta = 2.0 * np.pi * i / n_side
            nx_, ny_ = np.cos(theta), np.sin(theta)
            pts[i] = [self.R * nx_, self.R * ny_, 0.5 * self.wall_height]
            normals[i] = [nx_, ny_, 0.0]

        # Rotating bottom plate
        idx = n_side
        for i in range(n_bottom_r):
            r = (i + 0.5) / n_bottom_r * self.R
            for j in range(n_bottom_theta):
                theta = 2.0 * np.pi * j / n_bottom_theta
                pts[idx] = [r * np.cos(theta), r * np.sin(theta), 0.0]
                normals[idx] = [0.0, 0.0, -1.0]
                is_bottom[idx] = True
                idx += 1

        self.boundary_pts = pts
        self.boundary_normals = normals
        self.boundary_is_bottom = is_bottom

    # ------------------------------------------------------------------
    # Bottom-plate angular velocity profile
    # ------------------------------------------------------------------
    def _omega(self, t: float) -> float:
        # Smooth spin-up over 2 seconds to avoid an impulsive start.
        ramp_t = 2.0
        alpha = min(1.0, t / ramp_t)
        return self.omega_target * alpha

    # ------------------------------------------------------------------
    # Numerical PDE integration
    # ------------------------------------------------------------------
    def _upwind_grad(self, q: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple:
        """Return first-order upwind spatial gradients of scalar q."""
        q_x = np.zeros_like(q)
        q_y = np.zeros_like(q)

        pos_x = u > 0.0
        neg_x = ~pos_x
        q_x[pos_x] = (q - np.roll(q, 1, axis=0))[pos_x] / self.dx
        q_x[neg_x] = (np.roll(q, -1, axis=0) - q)[neg_x] / self.dx

        pos_y = v > 0.0
        neg_y = ~pos_y
        q_y[pos_y] = (q - np.roll(q, 1, axis=1))[pos_y] / self.dy
        q_y[neg_y] = (np.roll(q, -1, axis=1) - q)[neg_y] / self.dy

        return q_x, q_y

    def _laplacian(self, q: np.ndarray) -> np.ndarray:
        """5-point Laplacian with zero outside the cylinder."""
        lap = (
            np.roll(q, -1, axis=0)
            + np.roll(q, 1, axis=0)
            - 2.0 * q
            + np.roll(q, -1, axis=1)
            + np.roll(q, 1, axis=1)
            - 2.0 * q
        )
        lap /= self.dx * self.dy
        lap = np.where(self.cyl_mask, lap, 0.0)
        return lap

    def _step(self, dt: float) -> None:
        """Explicit forward-Euler shallow-water substep."""
        t = self.current_time
        omega = self._omega(t)

        # Bottom-plate target velocity at grid points.
        u_b = -omega * self.Y
        v_b = omega * self.X

        u = self.u
        v = self.v
        eta = self.eta

        # --- momentum advection (upwind) ----------------------------------
        u_x, u_y = self._upwind_grad(u, u, v)
        v_x, v_y = self._upwind_grad(v, u, v)

        # --- pressure gradient (central differences) ----------------------
        eta_x = np.zeros_like(eta)
        eta_y = np.zeros_like(eta)
        eta_x[1:-1, :] = (eta[2:, :] - eta[:-2, :]) / (2.0 * self.dx)
        eta_y[:, 1:-1] = (eta[:, 2:] - eta[:, :-2]) / (2.0 * self.dy)

        # --- viscous / diffusive terms ----------------------------------
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        # --- source parameters (tuned for the 4-corner regime) ----------
        gamma = 0.05  # bottom-plate drag [1/s]
        nu = 0.02     # eddy viscosity [m^2/s]
        d_eta = 1e-4  # small numerical diffusion for surface [m^2/s]

        # --- update momentum --------------------------------------------
        du = -u * u_x - v * u_y - self.g * eta_x + gamma * (u_b - u) + nu * lap_u
        dv = -u * v_x - v * v_y - self.g * eta_y + gamma * (v_b - v) + nu * lap_v

        u_new = u + dt * du
        v_new = v + dt * dv

        # --- donor-cell flux form for continuity ------------------------
        # x-fluxes at i+1/2
        u_face = 0.5 * (u[:-1, :] + u[1:, :])
        eta_lx = eta[:-1, :]
        eta_rx = eta[1:, :]
        F_x = np.where(u_face > 0.0, eta_lx * u_face, eta_rx * u_face)

        # y-fluxes at j+1/2
        v_face = 0.5 * (v[:, :-1] + v[:, 1:])
        eta_ly = eta[:, :-1]
        eta_ry = eta[:, 1:]
        F_y = np.where(v_face > 0.0, eta_ly * v_face, eta_ry * v_face)

        # Divergence with zero-flux boundaries
        nx, ny = self.nx, self.ny
        F_x_full = np.zeros((nx + 1, ny))
        F_x_full[1:-1, :] = F_x
        div_x = (F_x_full[1:, :] - F_x_full[:-1, :]) / self.dx

        F_y_full = np.zeros((nx, ny + 1))
        F_y_full[:, 1:-1] = F_y
        div_y = (F_y_full[:, 1:] - F_y_full[:, :-1]) / self.dy

        lap_eta = self._laplacian(eta)
        eta_new = eta - dt * (div_x + div_y) + dt * d_eta * lap_eta

        # --- enforce cylinder / dry constraints -------------------------
        eta_new = np.where(
            self.cyl_mask, np.clip(eta_new, 1e-3, self.wall_height), 0.0
        )
        u_new = np.where(self.cyl_mask, u_new, 0.0)
        v_new = np.where(self.cyl_mask, v_new, 0.0)

        self.u = u_new
        self.v = v_new
        self.eta = eta_new
        self.current_time += dt

    # ------------------------------------------------------------------
    # StateFrame construction
    # ------------------------------------------------------------------
    def _sample_velocity_at_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of (u,v) at arbitrary (x,y) arrays."""
        ix = np.searchsorted(self.x, x)
        iy = np.searchsorted(self.y, y)
        ix = np.clip(ix, 1, self.nx - 1)
        iy = np.clip(iy, 1, self.ny - 1)

        x0, x1 = self.x[ix - 1], self.x[ix]
        y0, y1 = self.y[iy - 1], self.y[iy]
        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        wx = np.clip(wx, 0.0, 1.0)
        wy = np.clip(wy, 0.0, 1.0)

        def gather(q):
            q00 = q[ix - 1, iy - 1]
            q10 = q[ix, iy - 1]
            q01 = q[ix - 1, iy]
            q11 = q[ix, iy]
            return (
                (1.0 - wx) * (1.0 - wy) * q00
                + wx * (1.0 - wy) * q10
                + (1.0 - wx) * wy * q01
                + wx * wy * q11
            )

        return np.column_stack([gather(self.u), gather(self.v), np.zeros_like(x)])

    def _build_frame(self) -> StateFrame:
        # 3D level set: max(r-R, z-eta) matches the scaffold convention.
        eta_3d = self.eta[:, :, None]
        r_3d = self.R2d[:, :, None]
        z_3d = self.z[None, None, :]
        level_set = np.maximum(r_3d - self.R, z_3d - eta_3d)
        interior = level_set < -1.5 * self.h

        # Diagnostic velocity field on the 3D grid.
        u_3d = self.u[:, :, None] * np.ones((1, 1, self.nz))
        v_3d = self.v[:, :, None] * np.ones((1, 1, self.nz))
        w_3d = np.zeros_like(u_3d)
        velocity_field = np.stack([u_3d, v_3d, w_3d], axis=-1)

        # Wall boundary data at the current time.
        omega = self._omega(self.current_time)
        boundary_target = np.zeros_like(self.boundary_pts)
        rot = self.boundary_is_bottom
        boundary_target[rot, 0] = -omega * self.boundary_pts[rot, 1]
        boundary_target[rot, 1] = omega * self.boundary_pts[rot, 0]
        # Side-wall samples have zero target velocity (already zero).

        # No-slip: boundary velocity matches the prescribed wall motion.
        boundary_velocity = boundary_target.copy()

        fields = {
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "liquid_level_set": level_set,
            "liquid_interior_mask": interior,
            "velocity_field": velocity_field,
            "boundary_normal": self.boundary_normals.astype(float),
            "boundary_target_velocity": boundary_target.astype(float),
            "boundary_velocity": boundary_velocity.astype(float),
        }

        # Update particle velocity from the current flow field.
        particle_vel = self._sample_velocity_at_points(
            self.particle_pos[:, 0], self.particle_pos[:, 1]
        )

        return StateFrame(
            self.current_time,
            self.particle_pos,
            particle_vel,
            mass=self.particle_mass,
            body_id=self.particle_body_id,
            attributes=self.attributes,
            fields=fields,
            scalars=self.scalars,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        # Guard against non-monotonic calls by resetting if asked for earlier time.
        if time < self.current_time - 1e-12:
            self.current_time = 0.0
            self.eta = self._initial_surface()
            self.u.fill(0.0)
            self.v.fill(0.0)

        # Substep with a fixed internal dt (keeps CFL low enough for stability).
        internal_dt = 5.0e-4
        elapsed = time - self.current_time
        if elapsed > 0.0:
            n_full = int(np.floor(elapsed / internal_dt))
            for _ in range(n_full):
                self._step(internal_dt)
            remainder = time - self.current_time
            if remainder > 1e-12:
                self._step(remainder)

        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "t": self.current_time,
            "omega": self._omega(self.current_time),
            "eta_min": float(self.eta.min()),
            "eta_max": float(self.eta.max()),
        }
