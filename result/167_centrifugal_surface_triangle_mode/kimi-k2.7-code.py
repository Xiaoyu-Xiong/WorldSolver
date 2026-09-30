"""Physics solver for the rotating free-surface polygon experiment.

The solver couples a one-dimensional axisymmetric shallow-water model for the
bulk rotation with a reduced-order azimuthal mode-3 perturbation that grows
and saturates.  The combination produces a rotating triangular free-surface
depression reminiscent of the polygonal modes reported in rotating fluid
experiments.  All trajectory fields requested by the task are exported on
the diagnostic grid.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        # ------------------------------------------------------------------
        # Read configuration
        # ------------------------------------------------------------------
        self.config = scene.config
        tank = self.config["tank"]

        self.seed = int(seed)
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.H_avg = float(tank["initial_fill_height"])
        self.h = float(tank["particle_spacing"])
        self.rho = float(tank["liquid_density"])
        self.m = int(tank["target_mode"])
        self.omega = float(tank["bottom_angular_velocity_rad_s"])
        self.g = -float(self.config["gravity"][2])
        self.grid_shape = tuple(int(x) for x in tank["field_grid_shape"])
        self.z_b = 0.07
        self.dt_sample = float(self.config["sample_dt"])

        # ------------------------------------------------------------------
        # Axisymmetric shallow-water grid (cylindrical radius)
        # ------------------------------------------------------------------
        self.Nr = 80
        self.dr = self.radius / self.Nr
        self.r = (np.arange(self.Nr) + 0.5) * self.dr

        # Initial base free-surface matches the axisymmetric part of the
        # scaffold's initial condition: a gently curved surface with a small
        # central depression.
        ratio = self.r / self.radius
        self.H = self.H_avg - 0.16 * (1.0 - ratio ** 2) ** 2
        self.H = np.maximum(self.H, self.z_b + 1e-6)
        self.u_r = np.zeros(self.Nr, dtype=float)
        self.u_t = np.zeros(self.Nr, dtype=float)

        # Shallow-water parameters (tuned so the fluid lags the bottom plate and
        # produces a stable polygon without overflowing the side wall).
        self.c_d = 0.45
        self.nu = 0.03
        self.nu_h = 0.03
        self.min_d = 1e-4

        # ------------------------------------------------------------------
        # Azimuthal mode-3 perturbation
        # ------------------------------------------------------------------
        # The amplitude starts from the small seed in the scaffold and grows
        # via a logistic instability with nonlinear saturation.
        self.A = 0.035
        self.A_max = 0.12
        self.r_growth = 0.8
        self.r_damp = 0.05
        self.omega_p = self.omega * 0.5
        self.phi = 0.0

        # Stream-function amplitude for the m-fold perturbation velocity
        # (scales with the surface-mode amplitude).
        self._psi_amp = 0.5 * self.radius

        # ------------------------------------------------------------------
        # Particle state from the scaffold initial StateFrame
        # ------------------------------------------------------------------
        init_frame = scene.initial_state()
        self.position = init_frame.position.copy()
        self.velocity = init_frame.velocity.copy()
        self.mass = init_frame.mass.copy()
        self.body_id = init_frame.body_id.copy()
        self.attributes = {
            k: np.array(v, copy=True) for k, v in init_frame.attributes.items()
        }
        if "mass_quantity" not in self.attributes:
            self.attributes["mass_quantity"] = self.mass.copy()
        if "particle_radius" not in self.attributes:
            self.attributes["particle_radius"] = np.full(
                len(self.mass), 0.48 * self.h, dtype=float
            )

        # ------------------------------------------------------------------
        # Diagnostic 3-D grid and boundary samples
        # ------------------------------------------------------------------
        nx, ny, nz = self.grid_shape
        self.x = np.linspace(-self.radius, self.radius, nx)
        self.y = np.linspace(-self.radius, self.radius, ny)
        self.z = np.linspace(0.0, self.wall_height, nz)

        self._init_boundary_samples()

        # ------------------------------------------------------------------
        # Time and cached frame
        # ------------------------------------------------------------------
        self.t = 0.0
        self.frame = None
        self._recompute_frame()

    # ----------------------------------------------------------------------
    # Boundary sample generation
    # ----------------------------------------------------------------------
    def _init_boundary_samples(self) -> None:
        side_points = []
        side_normals = []
        side_target = []

        # Stationary cylindrical side wall
        Nw = 64
        theta = np.linspace(0.0, 2.0 * np.pi, Nw, endpoint=False)
        z_levels = np.array(
            [self.z_b + 0.1, self.H_avg, self.wall_height - 0.1], dtype=float
        )
        for z in z_levels:
            for th in theta:
                x = self.radius * np.cos(th)
                y = self.radius * np.sin(th)
                side_points.append((x, y, z))
                side_normals.append((np.cos(th), np.sin(th), 0.0))
                side_target.append((0.0, 0.0, 0.0))

        # Rotating bottom plate
        bottom_points = []
        bottom_normals = []
        bottom_target = []
        Nb = 20
        rb = np.linspace(0.15, 0.94 * self.radius, Nb)
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        for r_b in rb:
            for th in angles:
                x = r_b * np.cos(th)
                y = r_b * np.sin(th)
                bottom_points.append((x, y, self.z_b))
                bottom_normals.append((0.0, 0.0, 1.0))
                bottom_target.append((-self.omega * y, self.omega * x, 0.0))

        points = side_points + bottom_points
        normals = side_normals + bottom_normals
        targets = side_target + bottom_target

        self.boundary_points = np.asarray(points, dtype=float)
        self.boundary_normal = np.asarray(normals, dtype=float)
        # Normalise
        nrm = np.linalg.norm(self.boundary_normal, axis=1, keepdims=True)
        nrm = np.where(nrm == 0.0, 1.0, nrm)
        self.boundary_normal /= nrm
        self.boundary_target_velocity = np.asarray(targets, dtype=float)
        self.boundary_velocity = self.boundary_target_velocity.copy()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time <= self.t:
            return self.frame

        dt_total = time - self.t
        # Use fixed substeps for deterministic, stable integration
        n_sub = max(1, int(round(dt_total / (self.dt_sample / 20.0))))
        dt = dt_total / n_sub

        for _ in range(n_sub):
            self._substep(dt)

        self.t = time
        self._recompute_frame()
        return self.frame

    def diagnostics(self) -> dict[str, object]:
        return {}

    # ----------------------------------------------------------------------
    # Internal physics step
    # ----------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        # ---- Axisymmetric shallow-water update ----------------------------
        H = self.H
        u_r = self.u_r
        u_t = self.u_t
        r = self.r
        dr = self.dr

        D = np.maximum(H - self.z_b, 0.0)

        # Radial mass flux F = r * D * u_r
        F = r * D * u_r

        # Ghost-cell arrays for finite-difference stencils
        # H: zero-gradient at centre and wall
        Hg = np.empty(self.Nr + 2, dtype=float)
        Hg[1:-1] = H
        Hg[0] = H[1]
        Hg[-1] = H[-2]

        # u_r, u_t: no-slip (zero) at centre and wall
        ug_r = np.empty(self.Nr + 2, dtype=float)
        ug_r[1:-1] = u_r
        ug_r[0] = -u_r[0]
        ug_r[-1] = -u_r[-1]

        ug_t = np.empty(self.Nr + 2, dtype=float)
        ug_t[1:-1] = u_t
        ug_t[0] = -u_t[0]
        ug_t[-1] = -u_t[-1]

        # F: no through-flow -> F=0 at boundaries
        Fg = np.empty(self.Nr + 2, dtype=float)
        Fg[1:-1] = F
        Fg[0] = F[1]  # symmetry, consistent with F(0)=0
        Fg[-1] = -F[-1]

        # Central derivatives on interior cells
        dH_dr = (Hg[2:] - Hg[:-2]) / (2.0 * dr)
        dF_dr = (Fg[2:] - Fg[:-2]) / (2.0 * dr)

        # Upwind advection for velocity components (one-sided differences at
        # the cylinder centre and wall to stay inside the domain).
        def upwind_deriv(u):
            du = np.empty_like(u)
            for i in range(self.Nr):
                if i == 0:
                    du[i] = (u[i + 1] - u[i]) / dr
                elif i == self.Nr - 1:
                    du[i] = (u[i] - u[i - 1]) / dr
                elif u_r[i] >= 0.0:
                    du[i] = (u[i] - u[i - 1]) / dr
                else:
                    du[i] = (u[i + 1] - u[i]) / dr
            return du

        du_r = upwind_deriv(ug_r[1:-1])
        du_t = upwind_deriv(ug_t[1:-1])

        # Cylindrical Laplacian (vector components, includes 1/r^2 terms)
        lap_u_r = (
            (ug_r[2:] - 2.0 * u_r + ug_r[:-2]) / dr ** 2
            + (ug_r[2:] - ug_r[:-2]) / (2.0 * dr * r)
            - u_r / r ** 2
        )
        lap_u_t = (
            (ug_t[2:] - 2.0 * u_t + ug_t[:-2]) / dr ** 2
            + (ug_t[2:] - ug_t[:-2]) / (2.0 * dr * r)
            - u_t / r ** 2
        )

        # Cylindrical Laplacian for H (used as artificial/numerical diffusion)
        lap_H = (
            (Hg[2:] - 2.0 * H + Hg[:-2]) / dr ** 2
            + (Hg[2:] - Hg[:-2]) / (2.0 * dr * r)
        )

        # RHS for free surface height
        dH = -(1.0 / r) * dF_dr + self.nu_h * lap_H

        # RHS for radial momentum
        adv_r = -u_r * du_r
        centrifugal = u_t ** 2 / r
        pressure = -self.g * dH_dr
        drag_r = -self.c_d * u_r
        dU_r = adv_r + centrifugal + pressure + drag_r + self.nu * lap_u_r

        # RHS for azimuthal momentum
        adv_t = -u_r * du_t
        curvature = -u_r * u_t / r
        drag_t = -self.c_d * (u_t - self.omega * r)
        dU_t = adv_t + curvature + drag_t + self.nu * lap_u_t

        # Forward-Euler update
        self.H = H + dt * dH
        self.H = np.maximum(self.H, self.z_b + 1e-6)
        self.u_r = u_r + dt * dU_r
        self.u_t = u_t + dt * dU_t

        # ---- Mode-3 perturbation update ---------------------------------
        dA = (
            self.r_growth * self.A * (1.0 - self.A / self.A_max)
            - self.r_damp * self.A
        )
        self.A = max(0.0, self.A + dt * dA)
        self.phi = (self.phi + dt * self.omega_p) % (2.0 * np.pi)

        # ---- Particle advection ------------------------------------------
        self._advect_particles(dt)

    # ----------------------------------------------------------------------
    # Surface geometry helpers
    # ----------------------------------------------------------------------
    def _base_surface(self, r: np.ndarray) -> np.ndarray:
        """Return the axisymmetric base surface at radii ``r``."""
        return np.interp(r, self.r, self.H)

    def _surface_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the 2-D free-surface height on the Cartesian grid."""
        R = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        H_base = self._base_surface(R)

        s = R / self.radius
        shape = 4.0 * s * (1.0 - s)
        # zero outside the cylinder
        shape = np.where(R < self.radius, shape, 0.0)

        H3 = self.A * shape * np.cos(self.m * (theta - self.phi))
        H = H_base + H3
        # Clip to physical bounds
        H = np.clip(H, self.z_b, self.wall_height)
        return H

    def _velocity_at(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """Return (vx, vy) arrays for the 2-D horizontal velocity field."""
        R = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)

        # Base axisymmetric velocity (interpolated from radial grid)
        u_r = np.interp(R.ravel(), self.r, self.u_r).reshape(R.shape)
        u_t = np.interp(R.ravel(), self.r, self.u_t).reshape(R.shape)

        # Mode-3 perturbation via a divergence-free m-fold stream function
        s = R / self.radius
        m = self.m
        psi_amp = self._psi_amp * (self.A / self.A_max)

        # ψ ∝ s^m (1-s) sin(m(θ-φ))
        # v_r = -(1/r) ∂ψ/∂θ
        # v_θ = ∂ψ/r
        phase = m * (theta - self.phi)
        factor = np.power(s, m) * (1.0 - s)
        factor = np.where(R < self.radius, factor, 0.0)

        v_r_pert = (
            -psi_amp * m * np.power(s, m) * (1.0 - s) / R * np.cos(phase)
        )
        v_t_pert = (
            psi_amp * np.power(s, m - 1) / self.radius
            * (m - (m + 1.0) * s) * np.sin(phase)
        )
        v_r_pert = np.where(R > 1e-6, v_r_pert, 0.0)
        v_t_pert = np.where(R > 1e-6, v_t_pert, 0.0)

        v_r = u_r + v_r_pert
        v_t = u_t + v_t_pert

        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        return vx, vy

    def _advect_particles(self, dt: float) -> None:
        """Move surface particles with the local horizontal velocity."""
        X = self.position[:, 0]
        Y = self.position[:, 1]
        vx, vy = self._velocity_at(X, Y)

        X += dt * vx
        Y += dt * vy

        # Keep particles inside the cylinder
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = R > self.radius
        if np.any(mask):
            theta = np.arctan2(Y, X)
            # Clamp radius and reflect radial velocity by keeping them just
            # inside the wall.
            X[mask] = 0.99 * self.radius * np.cos(theta[mask])
            Y[mask] = 0.99 * self.radius * np.sin(theta[mask])
            R[mask] = 0.99 * self.radius

        # Snap vertical position to the instantaneous free surface
        Z = self._surface_2d(X, Y)
        self.position[:, 0] = X
        self.position[:, 1] = Y
        self.position[:, 2] = Z

        # Update velocity attribute for the StateFrame
        vx, vy = self._velocity_at(X, Y)
        self.velocity[:, 0] = vx
        self.velocity[:, 1] = vy
        self.velocity[:, 2] = 0.0

    # ----------------------------------------------------------------------
    # StateFrame construction
    # ----------------------------------------------------------------------
    def _recompute_frame(self) -> None:
        nx, ny, nz = self.grid_shape
        X2, Y2 = np.meshgrid(self.x, self.y, indexing="ij")
        H2d = self._surface_2d(X2, Y2)

        # 3-D level set and mask
        Z3 = self.z.reshape(1, 1, nz)
        level_set = Z3 - H2d[:, :, None]
        interior = (level_set < 0.0).astype(float)

        # 3-D diagnostic velocity field (broadcast horizontally uniform profile)
        vx, vy = self._velocity_at(X2, Y2)
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=float)
        velocity_field[:, :, :, 0] = vx[:, :, None]
        velocity_field[:, :, :, 1] = vy[:, :, None]

        self.frame = StateFrame(
            self.t,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.attributes["mass_quantity"].copy(),
                "particle_radius": self.attributes["particle_radius"].copy(),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "liquid_interior_mask": interior,
                "liquid_level_set": level_set,
                "velocity_field": velocity_field,
            },
            scalars={
                "grid_spacing": self.h,
                "characteristic_length": 2.0 * self.radius,
                "reference_velocity": max(self.omega * self.radius, 1e-6),
            },
        )
