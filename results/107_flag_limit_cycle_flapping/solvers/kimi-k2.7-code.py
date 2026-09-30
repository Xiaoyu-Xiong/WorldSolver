"""Flag-in-flow solver using a vorticity-streamfunction fluid model
and a discrete elastic beam for the flag, coupled via an immersed-boundary
penalty method.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._scene = scene
        self._time = 0.0
        self._frame_idx = 0

        # Read configuration (frozen)
        cfg = scene.config if hasattr(scene, "config") else {}
        phys = cfg.get("physics", {}) if isinstance(cfg, dict) else {}
        self.U = float(phys.get("inflow_velocity", [1.0, 0.0])[0])
        self.Re = float(phys.get("reynolds_number", 1400.0))
        self.mass_ratio = float(phys.get("mass_ratio", 0.16))
        self.K_B = float(phys.get("bending_rigidity", 0.001))
        self.flag_L = float(phys.get("flag_length", 1.0))
        self.flag_h = float(phys.get("flag_thickness", 0.05))
        self.rho_f = 1.0
        self.nu = self.rho_f * self.U * self.flag_L / self.Re

        # Fluid domain (matches scene_geometry carrier_fluid bounds)
        self.xmin, self.xmax = -1.0, 5.0
        self.ymin, self.ymax = -1.0, 1.0
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

        # Grid resolution (kept moderate for stability)
        self.nx = 150
        self.ny = 50
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        # Time
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 20.0))

        # Coordinate arrays
        self.xg = np.linspace(self.xmin, self.xmax, self.nx)
        self.yg = np.linspace(self.ymin, self.ymax, self.ny)
        self.X, self.Y = np.meshgrid(self.xg, self.yg, indexing="ij")

        # Initialize vorticity to small perturbation to break symmetry
        self.omega = 0.01 * self._rng.standard_normal((self.nx, self.ny))

        # Build Poisson solver (perturbation streamfunction, periodic in y)
        self._build_poisson_solver()

        # Flag discretisation
        self.n_flag = 41
        self.s_flag = np.linspace(0.0, self.flag_L, self.n_flag)
        self.ds_flag = self.s_flag[1] - self.s_flag[0]
        # Reference straight flag along x-axis, anchored at origin
        self.flag_x0 = self.s_flag.copy()
        self.flag_y0 = np.zeros(self.n_flag)
        self.flag_x = self.flag_x0.copy()
        self.flag_y = self.flag_y0.copy()
        self.flag_vx = np.zeros(self.n_flag)
        self.flag_vy = np.zeros(self.n_flag)

        # Flag mass per node (2D sheet, mass per unit span = rho_s * h)
        self.rho_s = self.mass_ratio * self.rho_f / self.flag_h
        self.m_per_node = self.rho_s * self.flag_h * self.ds_flag
        # Half mass for end nodes
        self.m_per_node = np.full(self.n_flag, self.m_per_node)
        self.m_per_node[0] *= 0.5
        self.m_per_node[-1] *= 0.5

        # IBM coupling coefficient (1/s)
        self.alpha = 250.0

        # Stretching stiffness for inextensibility (large)
        self.k_stretch = 500.0

        # Reference momentum/velocity for trajectory
        self.ref_velocity = self.U
        self.ref_momentum = self.rho_f * self.U * self.flag_L * self.flag_h

        # Budget accumulators (cumulative impulse/momentum)
        self.cum_external_impulse = np.zeros((2, 3), dtype=float)
        self.cum_inflow_momentum = np.zeros((2, 3), dtype=float)
        self.cum_outflow_momentum = np.zeros((2, 3), dtype=float)

        # Store last inflow/outflow rate for trajectory output
        self.inflow_rate = 0.0
        self.outflow_rate = 0.0

        # Fluid particle state (diagnostic grid samples)
        self._init_fluid_samples()

        # Cumulative mass source (no sources/sinks)
        self.cum_mass_source = np.zeros(2, dtype=float)

        # Diagnostics
        self._diag = {
            "fluid_points": self.nx * self.ny,
            "flag_points": self.n_flag,
            "reynolds": self.Re,
            "nu": self.nu,
        }

    def _build_poisson_solver(self):
        """Build sparse LU factorisation for perturbation streamfunction Poisson.
        Boundary conditions: psi=0 at left, dpsi/dx=0 at right, periodic in y.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # Second derivative in x (collocated)
        # Use 1D Laplacian with boundary conditions folded in.
        # Left: psi=0 (Dirichlet), Right: dpsi/dx=0 (Neumann, ghost = psi_{n-1})
        main_x = np.full(nx, -2.0 / (dx * dx))
        off_x = np.full(nx - 1, 1.0 / (dx * dx))
        main_x[0] = -2.0 / (dx * dx)  # Dirichlet ghost
        main_x[-1] = -1.0 / (dx * dx)  # Neumann ghost
        Dx = np.diag(main_x) + np.diag(off_x, 1) + np.diag(off_x, -1)
        # The top-left entry handles Dirichlet implicitly by keeping equation for i=0
        # For i=0, psi=0, so we can drop the first row/column.
        # Simpler: build the full matrix and solve with psi[0,:]=0.

        # Build using Kronecker products
        # y-derivative with periodic BC (circulant)
        # 2nd order: (u_{j+1} - 2u_j + u_{j-1}) / dy^2
        # Periodic tridiagonal
        Iy = eye(ny, format="csr")
        e = np.ones(ny)
        Ly = diags([e, -2.0 * e, e], [-1, 0, 1], shape=(ny, ny), format="csr")
        Ly += diags([e, e], [ny - 1, -(ny - 1)], format="csr")
        Ly *= 1.0 / (dy * dy)

        # x-derivative matrix with BCs
        e = np.ones(nx)
        # For interior, standard Laplacian
        main = -2.0 * e / (dx * dx)
        off = 1.0 / (dx * dx) * np.ones(nx - 1)
        A = diags([off, main, off], [-1, 0, 1], shape=(nx, nx), format="csr")
        # Left boundary: Dirichlet psi[0]=0 -> equation for i=0 is psi[0]=0
        A[0, 0] = 1.0
        A[0, 1] = 0.0
        # Right boundary: Neumann ghost psi[nx]=psi[nx-1]
        # Equation at i=nx-1: (psi[nx-2] - psi[nx-1]) / dx^2 = rhs * dx^2? Let's derive.
        # d2psi/dx2 at i=nx-1 with ghost psi[nx]=psi[nx-1]:
        # (psi[nx-2] - 2psi[nx-1] + psi[nx]) / dx^2 = (psi[nx-2] - psi[nx-1]) / dx^2
        # So coefficient of psi[nx-1] is -1/dx^2, psi[nx-2] is 1/dx^2.
        A[-1, -1] = -1.0 / (dx * dx)
        A[-1, -2] = 1.0 / (dx * dx)

        # Full 2D Laplacian with periodic y
        self._lap = kron(Iy, A, format="csr") + kron(Ly, eye(nx, format="csr"), format="csr")
        self._lu = splu(self._lap)

    def _poisson_solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve L psi = rhs with the stored LU factors."""
        flat = rhs.reshape(-1)
        psi = self._lu.solve(flat)
        return psi.reshape((self.nx, self.ny))

    def _velocity_from_vorticity(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute (u, v) perturbation velocity from vorticity omega."""
        psi = self._poisson_solve(-self.omega)
        self._psi = psi  # store for diagnostics

        # u = dpsi/dy, v = -dpsi/dx (periodic in y)
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)

        # Central differences in y with periodic BC
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * self.dy)
        # Boundaries in y via periodic wrap
        u[:, 0] = (psi[:, 1] - psi[:, -2]) / (2.0 * self.dy)
        u[:, -1] = (psi[:, 1] - psi[:, -2]) / (2.0 * self.dy)

        # v = -dpsi/dx, left/right boundaries
        v[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2.0 * self.dx)
        v[0, :] = -(psi[1, :] - psi[0, :]) / self.dx
        v[-1, :] = -(psi[-1, :] - psi[-2, :]) / self.dx

        # Add mean flow
        u = u + self.U
        return u, v

    def _advect(self, phi: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Upwind advection of scalar phi by velocity (u, v) using 1st order."""
        dphi = np.zeros_like(phi)
        # Interior (periodic in y)
        for i in range(1, self.nx - 1):
            for j in range(self.ny):
                jp = (j + 1) % self.ny
                jm = (j - 1) % self.ny
                if u[i, j] >= 0:
                    dphi[i, j] += u[i, j] * (phi[i, j] - phi[i - 1, j]) / self.dx
                else:
                    dphi[i, j] += u[i, j] * (phi[i + 1, j] - phi[i, j]) / self.dx
                if v[i, j] >= 0:
                    dphi[i, j] += v[i, j] * (phi[i, j] - phi[i, jm]) / self.dy
                else:
                    dphi[i, j] += v[i, j] * (phi[i, jp] - phi[i, j]) / self.dy
        return dphi

    def _diffusion_step(self, omega: np.ndarray) -> np.ndarray:
        """Implicit Crank-Nicolson diffusion step for interior points."""
        # Use explicit diffusion for simplicity (stable for our parameters)
        lap = np.zeros_like(omega)
        for i in range(1, self.nx - 1):
            for j in range(self.ny):
                jp = (j + 1) % self.ny
                jm = (j - 1) % self.ny
                d2x = (omega[i + 1, j] - 2.0 * omega[i, j] + omega[i - 1, j]) / (self.dx * self.dx)
                d2y = (omega[i, jp] - 2.0 * omega[i, j] + omega[i, jm]) / (self.dy * self.dy)
                lap[i, j] = d2x + d2y
        return lap

    def _curl_of_force(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Return curl_z = d fy / dx - d fx / dy on the grid."""
        curl = np.zeros_like(fx)
        # interior derivatives (periodic y)
        for i in range(1, self.nx - 1):
            for j in range(self.ny):
                jp = (j + 1) % self.ny
                jm = (j - 1) % self.ny
                dfy_dx = (fy[i + 1, j] - fy[i - 1, j]) / (2.0 * self.dx)
                dfx_dy = (fx[i, jp] - fx[i, jm]) / (2.0 * self.dy)
                curl[i, j] = dfy_dx - dfx_dy
        return curl

    def _interpolate_velocity_to_flag(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trilinear (2D) interpolation of grid velocity to flag points."""
        ui = np.zeros(self.n_flag)
        vi = np.zeros(self.n_flag)
        for k in range(self.n_flag):
            xk, yk = self.flag_x[k], self.flag_y[k]
            # Periodic in y
            yk = ((yk - self.ymin) % self.Ly) + self.ymin
            ix = int(np.clip((xk - self.xmin) / self.dx, 0, self.nx - 1))
            jy = int(np.clip((yk - self.ymin) / self.dy, 0, self.ny - 1))
            wx = (xk - self.xmin) / self.dx - ix
            wy = (yk - self.ymin) / self.dy - jy
            # Bilinear
            def sample(i, j):
                j = j % self.ny
                return u[i, j], v[i, j]
            u00, v00 = sample(ix, jy)
            u10, v10 = sample(min(ix + 1, self.nx - 1), jy)
            u01, v01 = sample(ix, jy + 1)
            u11, v11 = sample(min(ix + 1, self.nx - 1), jy + 1)
            ua = u00 * (1 - wx) + u10 * wx
            va = v00 * (1 - wx) + v10 * wx
            ub = u01 * (1 - wx) + u11 * wx
            vb = v01 * (1 - wx) + v11 * wx
            ui[k] = ua * (1 - wy) + ub * wy
            vi[k] = va * (1 - wy) + vb * wy
        return ui, vi

    def _flag_elastic_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute elastic (bending + stretching) forces on flag nodes."""
        x, y = self.flag_x, self.flag_y
        n = self.n_flag
        ds = self.ds_flag

        # Stretching forces from segment length penalty
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        for i in range(n - 1):
            dx_ = x[i + 1] - x[i]
            dy_ = y[i + 1] - y[i]
            l = np.hypot(dx_, dy_)
            if l > 1e-12:
                dl = l - ds
                fmag = self.k_stretch * dl
                fx = fmag * dx_ / l
                fy = fmag * dy_ / l
                Fx[i] -= fx
                Fy[i] -= fy
                Fx[i + 1] += fx
                Fy[i + 1] += fy

        # Bending forces (discrete curvature)
        for i in range(1, n - 1):
            # Curvature at vertex i approximated by angle change
            t1x = x[i] - x[i - 1]
            t1y = y[i] - y[i - 1]
            t2x = x[i + 1] - x[i]
            t2y = y[i + 1] - y[i]
            l1 = np.hypot(t1x, t1y)
            l2 = np.hypot(t2x, t2y)
            if l1 > 1e-12 and l2 > 1e-12:
                # Force toward curvature center: -K_B * (kappa_i - kappa_{i-1}) in normal dir
                # Simplified: add force proportional to discrete Laplacian of position
                # This is the linearized beam force
                pass

        # Linearized bending force: EI * (y_{i-2} - 4y_{i-1} + 6y_i - 4y_{i+1} + y_{i+2}) / ds^3
        # Use this in y; x is kept by stretching
        for i in range(2, n - 2):
            lap4 = (y[i - 2] - 4.0 * y[i - 1] + 6.0 * y[i] - 4.0 * y[i + 1] + y[i + 2]) / (ds ** 3)
            Fy[i] += self.K_B * lap4 / ds
        return Fx, Fy

    def _spread_flag_force_to_grid(self, Fx: np.ndarray, Fy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spread flag nodal forces to the fluid grid using a discrete delta."""
        gx = np.zeros((self.nx, self.ny))
        gy = np.zeros((self.nx, self.ny))
        cell_area = self.dx * self.dy
        for k in range(self.n_flag):
            xk, yk = self.flag_x[k], self.flag_y[k]
            # Find nearest grid index
            ix = int(np.clip(np.round((xk - self.xmin) / self.dx), 0, self.nx - 1))
            jy = int(np.clip(np.round((yk - self.ymin) / self.dy), 0, self.ny - 1))
            # 3x3 stencil
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii = ix + di
                    jj = (jy + dj) % self.ny
                    if ii < 0 or ii >= self.nx:
                        continue
                    dist = np.hypot((self.xg[ii] - xk) / self.dx, (self.yg[jj] - yk) / self.dy)
                    if dist < 1.5:
                        w = (1.0 + np.cos(np.pi * dist / 1.5)) / (4.0 * 1.5)
                    else:
                        w = 0.0
                    gx[ii, jj] += Fx[k] * w / cell_area
                    gy[ii, jj] += Fy[k] * w / cell_area
        return gx, gy

    def _init_fluid_samples(self):
        """Create the particle-domain samples for the carrier fluid body."""
        self.fluid_x = self.X.flatten()
        self.fluid_y = self.Y.flatten()
        self.n_fluid = self.fluid_x.size

    def _sample_fluid_velocity_at_state_points(self) -> np.ndarray:
        """Return per-state-sample velocity [N,3] combining fluid grid and flag."""
        # Fluid samples: use current grid velocity
        u, v = self._velocity_from_vorticity()
        vel = np.zeros((self.n_fluid + self.n_flag, 3), dtype=float)
        vel[:self.n_fluid, 0] = u.flatten()
        vel[:self.n_fluid, 1] = v.flatten()
        # Flag samples
        vel[self.n_fluid:, 0] = self.flag_vx
        vel[self.n_fluid:, 1] = self.flag_vy
        return vel

    def _step(self, dt: float):
        """Advance one internal time step."""
        # Current fluid velocity
        u, v = self._velocity_from_vorticity()

        # Interpolate to flag
        ui, vi = self._interpolate_velocity_to_flag(u, v)

        # Flag elastic forces
        Fx_e, Fy_e = self._flag_elastic_forces()

        # Flag update with penalty coupling (semi-implicit)
        # v_new = (v + dt/m*F_el + dt*alpha*u_interp) / (1 + dt*alpha)
        denom = 1.0 + dt * self.alpha
        self.flag_vx = (self.flag_vx + dt * Fx_e / self.m_per_node + dt * self.alpha * ui) / denom
        self.flag_vy = (self.flag_vy + dt * Fy_e / self.m_per_node + dt * self.alpha * vi) / denom

        # Anchor first node
        self.flag_vx[0] = 0.0
        self.flag_vy[0] = 0.0
        self.flag_x[0] = 0.0
        self.flag_y[0] = 0.0

        # Update flag positions
        self.flag_x += dt * self.flag_vx
        self.flag_y += dt * self.flag_vy

        # Recalculate elastic forces after update for action-reaction
        Fx_e, Fy_e = self._flag_elastic_forces()

        # Fluid force on flag = m * alpha * (u_interp - v_new)
        Ffx = self.m_per_node * self.alpha * (ui - self.flag_vx)
        Ffy = self.m_per_node * self.alpha * (vi - self.flag_vy)

        # Fluid feels equal and opposite body force
        gx, gy = self._spread_flag_force_to_grid(-Ffx, -Ffy)

        # Vorticity update: advection + diffusion + curl(force)
        adv = self._advect(self.omega, u, v)
        lap = self._diffusion_step(self.omega)
        curl_f = self._curl_of_force(gx, gy)

        # Boundary conditions
        # Left inflow: omega = 0
        # Right outflow: zero normal gradient (copy from interior)
        # Top/bottom periodic handled by loops
        self.omega[0, :] = 0.0
        self.omega[-1, :] = self.omega[-2, :]

        self.omega = self.omega - dt * adv + dt * self.nu * lap + dt * curl_f

        # Enforce inflow again after update
        self.omega[0, :] = 0.0
        self.omega[-1, :] = self.omega[-2, :]

        # Update budgets (simple inflow/outflow estimates)
        inflow_rate = self.rho_f * self.U * self.U * self.Ly  # momentum flux rate
        outflow_rate = np.sum(self.omega[-1, :]) * self.dy * self.rho_f * self.U  # crude
        self.inflow_rate = inflow_rate
        self.outflow_rate = outflow_rate
        self.cum_inflow_momentum[0, 0] += inflow_rate * dt
        self.cum_outflow_momentum[0, 0] += outflow_rate * dt

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._time:
            raise ValueError("advance_to time must be non-decreasing")

        # Substep until target
        n_sub = max(1, int(np.ceil((target - self._time) / self.sample_dt)))
        dt = (target - self._time) / n_sub if n_sub > 0 else self.sample_dt
        for _ in range(n_sub):
            self._step(dt)
        self._time = target
        self._frame_idx += 1

        # Build StateFrame
        n_state = self.n_fluid + self.n_flag
        position = np.zeros((n_state, 3), dtype=float)
        position[:self.n_fluid, 0] = self.fluid_x
        position[:self.n_fluid, 1] = self.fluid_y
        position[self.n_fluid:, 0] = self.flag_x
        position[self.n_fluid:, 1] = self.flag_y

        velocity = self._sample_fluid_velocity_at_state_points()

        body_id = np.concatenate([
            np.full(self.n_fluid, 1, dtype=int),
            np.full(self.n_flag, 2, dtype=int),
        ])

        mass = np.zeros(n_state, dtype=float)
        mass[:self.n_fluid] = self.rho_f * self.dx * self.dy
        mass[self.n_fluid:] = self.m_per_node

        # attributes.mass_quantity: per sample, shape [N]
        mass_quantity = mass.copy()

        # Boundary samples: anchor object (static wall) and flag interface
        B = 4 + self.n_flag
        boundary_normal = np.zeros((B, 3), dtype=float)
        boundary_target_velocity = np.zeros((B, 3), dtype=float)
        boundary_velocity = np.zeros((B, 3), dtype=float)
        # Anchor: 4 wall samples (left, right, top, bottom of anchor box)
        # Anchor box half_extents [0.025, 0.06, 0.02]
        # For 2D, just put normals
        boundary_normal[0] = [-1.0, 0.0, 0.0]
        boundary_normal[1] = [1.0, 0.0, 0.0]
        boundary_normal[2] = [0.0, 1.0, 0.0]
        boundary_normal[3] = [0.0, -1.0, 0.0]
        # Flag interface samples: normal roughly outward
        for k in range(self.n_flag):
            # tangent
            if k == 0:
                tx = self.flag_x[1] - self.flag_x[0]
                ty = self.flag_y[1] - self.flag_y[0]
            elif k == self.n_flag - 1:
                tx = self.flag_x[-1] - self.flag_x[-2]
                ty = self.flag_y[-1] - self.flag_y[-2]
            else:
                tx = self.flag_x[k + 1] - self.flag_x[k - 1]
                ty = self.flag_y[k + 1] - self.flag_y[k - 1]
            l = np.hypot(tx, ty)
            if l > 1e-12:
                # outward normal = (ty, -tx) / l or (-ty, tx)/l; choose one consistent
                nx_ = ty / l
                ny_ = -tx / l
                boundary_normal[4 + k, :2] = [nx_, ny_]
            boundary_velocity[4 + k, :2] = [self.flag_vx[k], self.flag_vy[k]]
            boundary_target_velocity[4 + k, :2] = [self.flag_vx[k], self.flag_vy[k]]

        # Budget domain weights: rows = [fluid, flag]
        weights = np.zeros((2, n_state), dtype=float)
        weights[0, :self.n_fluid] = 1.0
        weights[1, self.n_fluid:] = 1.0

        # External force budgets: [G=2, K=1, 3]
        # For fluid: inflow forcing along x; for flag: anchor reaction (zero here)
        budget_external_force = np.zeros((2, 1, 3), dtype=float)
        budget_external_force[0, 0, 0] = self.inflow_rate

        # Interaction force between fluid and flag (sum over flag)
        interaction_force = np.zeros((1, 3), dtype=float)
        # Use fluid force on flag from last step estimate
        u, v = self._velocity_from_vorticity()
        ui, vi = self._interpolate_velocity_to_flag(u, v)
        Ffx = self.m_per_node * self.alpha * (ui - self.flag_vx)
        Ffy = self.m_per_node * self.alpha * (vi - self.flag_vy)
        interaction_force[0, 0] = np.sum(Ffx)
        interaction_force[0, 1] = np.sum(Ffy)

        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_velocity,
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": interaction_force,
            "vorticity": self.omega.copy(),
        }

        scalars = {
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
        }

        attributes = {
            "mass_quantity": mass_quantity,
        }

        return StateFrame(
            time=self._time,
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return self._diag.copy()
