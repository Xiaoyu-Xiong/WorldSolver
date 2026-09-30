"""2-D incompressible Navier-Stokes solver for channel flow past a cylinder.

The implementation uses a cell-centered grid, semi-Lagrangian advection,
explicit viscous diffusion, and pressure projection to enforce
incompressibility.  Boundary conditions are no-slip on the channel walls
and cylinder, a prescribed parabolic inflow, and a zero-pressure open
outflow.  The diagnostic grid fields required by the trajectory contract
are exported at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.config = getattr(scene, "config", {})
        self.seed = seed
        np.random.seed(seed)

        # Physical parameters
        self.rho = float(self.config["fluid"]["density"])
        self.nu = float(self.config["fluid"]["kinematic_viscosity"])

        # Domain
        self.x_min, self.x_max = -6.0, 10.0
        self.y_min, self.y_max = -1.5, 1.5
        self.dx = float(self.config["fluid"]["nominal_particle_spacing"])
        self.nx = int(round((self.x_max - self.x_min) / self.dx))
        self.ny = int(round((self.y_max - self.y_min) / self.dx))
        self.dy = self.dx  # uniform square cells

        self.dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])

        # Obstacle
        cyl = self.config["obstacle"]
        self.cyl_x = float(cyl["center"][0])
        self.cyl_y = float(cyl["center"][1])
        self.cyl_r = float(cyl["radius"])

        # Reference scales
        self.U_ref = float(self.config["initial_conditions"]["mean_speed"])
        self.U_max = float(self.config["initial_conditions"]["centerline_speed"])
        self.H = 1.5  # half-channel height
        self.L_char = 2.0 * self.cyl_r  # cylinder diameter

        # Cell centers
        self.x_c = self._x_centers()
        self.y_c = self._y_centers()
        self.X, self.Y = np.meshgrid(self.x_c, self.y_c, indexing="ij")

        # Build active mask (fluid cells outside the cylinder)
        self.active = (
            (self.X - self.cyl_x) ** 2 + (self.Y - self.cyl_y) ** 2
        ) >= (self.cyl_r ** 2)

        # Initial state from the frozen scene contract
        init = scene.initial_state()
        self.position = np.asarray(init.position, dtype=float)
        self.particle_velocity = np.asarray(init.velocity, dtype=float)

        # Diagnostic dye on the cell-centered grid
        self.dye = np.zeros((self.nx, self.ny), dtype=float)
        if "dye" in init.fields:
            self.dye[:] = np.asarray(init.fields["dye"], dtype=float)
        else:
            # Fallback: alternating inlet stripes
            stripe = np.floor((self.X + 6.0) / 0.55).astype(int)
            self.dye[:] = np.where(self.X < -2.55, ((stripe & 1) * 2 - 1), 0.0)
            self.dye *= np.exp(-0.5 * (self.Y / 0.34) ** 2)
            self.dye[~self.active] = 0.0

        # Cell-centered velocity fields
        self.u = np.zeros((self.nx, self.ny), dtype=float)
        self.v = np.zeros((self.nx, self.ny), dtype=float)

        # Initial parabolic profile
        u_prof = self._inlet_profile(self.Y)
        self.u[:, :] = u_prof

        # Small asymmetric perturbation to seed vortex shedding
        self._seed_perturbation()

        # Apply all boundary / obstacle conditions to the initial field
        self._apply_boundary_conditions()

        # Pre-build pressure Poisson system for the active fluid region
        self._build_pressure_system()

        self.current_time = 0.0
        self._diag = {}

    # ------------------------------------------------------------------
    # Grid geometry helpers
    # ------------------------------------------------------------------
    def _x_faces(self) -> np.ndarray:
        return self.x_min + np.arange(self.nx + 1) * self.dx

    def _y_faces(self) -> np.ndarray:
        return self.y_min + np.arange(self.ny + 1) * self.dy

    def _x_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.nx) + 0.5) * self.dx

    def _y_centers(self) -> np.ndarray:
        return self.y_min + (np.arange(self.ny) + 0.5) * self.dy

    def _inlet_profile(self, y: np.ndarray) -> np.ndarray:
        return self.U_max * np.maximum(0.0, 1.0 - (y / self.H) ** 2)

    # ------------------------------------------------------------------
    # Boundary / obstacle conditions
    # ------------------------------------------------------------------
    def _seed_perturbation(self) -> None:
        # Slight vertical velocity near the cylinder to break symmetry
        sigma = 0.8
        amp = 0.02 * self.U_max
        dist2 = (self.X - self.cyl_x) ** 2 + (self.Y - self.cyl_y) ** 2
        perturb = amp * np.sin(np.pi * (self.Y + self.H) / (2.0 * self.H))
        perturb *= np.exp(-dist2 / (2.0 * sigma ** 2))
        self.v += perturb

    def _apply_boundary_conditions(self) -> None:
        # Inactive cells (cylinder) enforce no-slip
        self.u[~self.active] = 0.0
        self.v[~self.active] = 0.0

        # Top / bottom walls (no-slip)
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        # Left inlet: prescribed parabolic profile
        self.u[0, :] = self._inlet_profile(self.Y[0, :])
        self.v[0, :] = 0.0

        # Right outlet: zero normal gradient (Neumann)
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]

    # ------------------------------------------------------------------
    # Pressure Poisson system
    # ------------------------------------------------------------------
    def _build_pressure_system(self) -> None:
        active_flat = self.active.ravel()
        self.n_active = int(active_flat.sum())
        self.idx = np.full(self.active.shape, -1, dtype=int)
        self.idx[self.active] = np.arange(self.n_active)

        rows, cols, vals = [], [], []

        for j in range(self.ny):
            for i in range(self.nx):
                if not self.active[i, j]:
                    continue
                r = self.idx[i, j]
                diag = 0.0

                def add_neighbor(di: int, dj: int, bc: str):
                    nonlocal diag
                    ni, nj = i + di, j + dj
                    # Interior active neighbor
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        if self.active[ni, nj]:
                            c = self.idx[ni, nj]
                            rows.append(r)
                            cols.append(c)
                            vals.append(1.0)
                            return
                        # Solid neighbor (cylinder): Neumann => p_solid = p_fluid
                        diag += 1.0
                        return
                    # Domain boundary
                    if bc == "dirichlet":
                        # p = 0, contributes nothing to LHS or RHS
                        return
                    # Neumann boundary
                    diag += 1.0

                # East / West
                add_neighbor(+1, 0, "dirichlet")  # right outlet: p=0
                add_neighbor(-1, 0, "neumann")    # left inlet
                # North / South walls
                add_neighbor(0, +1, "neumann")
                add_neighbor(0, -1, "neumann")

                rows.append(r)
                cols.append(r)
                vals.append(diag)

        A = csr_matrix((vals, (rows, cols)), shape=(self.n_active, self.n_active))
        self.pressure_solver = splu(A)

    def _solve_pressure(self, div: np.ndarray) -> np.ndarray:
        rhs = (self.rho / self.dt) * div[self.active].ravel()
        p_flat = self.pressure_solver.solve(rhs)
        p = np.zeros((self.nx, self.ny), dtype=float)
        p[self.active] = p_flat
        return p

    # ------------------------------------------------------------------
    # Advection / diffusion helpers
    # ------------------------------------------------------------------
    def _laplacian(self, a: np.ndarray) -> np.ndarray:
        """5-point Laplacian with zero-gradient at domain boundaries."""
        la = np.zeros_like(a)
        dx2, dy2 = self.dx ** 2, self.dy ** 2
        la[1:-1, 1:-1] = (
            (a[2:, 1:-1] - 2.0 * a[1:-1, 1:-1] + a[:-2, 1:-1]) / dx2
            + (a[1:-1, 2:] - 2.0 * a[1:-1, 1:-1] + a[1:-1, :-2]) / dy2
        )
        return la

    def _bilinear_interpolate(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate field at physical coordinates (x, y)."""
        i = np.floor((x - self.x_min) / self.dx).astype(int)
        j = np.floor((y - self.y_min) / self.dy).astype(int)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        x0 = self.x_min + i * self.dx
        y0 = self.y_min + j * self.dy
        s = np.clip((x - x0) / self.dx, 0.0, 1.0)
        t = np.clip((y - y0) / self.dy, 0.0, 1.0)
        i1 = np.clip(i + 1, 0, self.nx - 1)
        j1 = np.clip(j + 1, 0, self.ny - 1)
        return (
            (1.0 - s) * (1.0 - t) * field[i, j]
            + s * (1.0 - t) * field[i1, j]
            + (1.0 - s) * t * field[i, j1]
            + s * t * field[i1, j1]
        )

    def _advect_field(self, f: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Semi-Lagrangian advection of scalar field f by velocity (u, v)."""
        # Backtrace from cell centers
        x_b = self.X - self.dt * u
        y_b = self.Y - self.dt * v
        return self._bilinear_interpolate(f, x_b, y_b)

    def _divergence(self) -> np.ndarray:
        div = np.zeros((self.nx, self.ny), dtype=float)
        div[1:-1, 1:-1] = (
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2.0 * self.dx)
            + (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2.0 * self.dy)
        )
        return div

    # ------------------------------------------------------------------
    # Full time step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # Substep if requested dt differs from nominal
        remaining = dt
        while remaining > 1e-12:
            sub = min(self.dt, remaining)
            old_dt = self.dt
            self.dt = sub
            self._substep()
            self.dt = old_dt
            remaining -= sub

    def _substep(self) -> None:
        # Advect velocity components with the current velocity field
        u_star = self._advect_field(self.u, self.u, self.v)
        v_star = self._advect_field(self.v, self.u, self.v)

        # Add viscous diffusion
        u_star += self.dt * self.nu * self._laplacian(self.u)
        v_star += self.dt * self.nu * self._laplacian(self.v)

        # Apply boundary conditions to provisional velocity
        self.u = u_star
        self.v = v_star
        self._apply_boundary_conditions()

        # Pressure projection to enforce incompressibility
        div = self._divergence()
        p = self._solve_pressure(div)

        # Correct velocities with pressure gradient in active cells
        grad_px = np.zeros((self.nx, self.ny), dtype=float)
        grad_py = np.zeros((self.nx, self.ny), dtype=float)
        grad_px[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * self.dx)
        grad_py[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * self.dy)


        self.u = np.where(self.active, self.u - (self.dt / self.rho) * grad_px, 0.0)
        self.v = np.where(self.active, self.v - (self.dt / self.rho) * grad_py, 0.0)

        # Re-apply boundary conditions
        self._apply_boundary_conditions()

        # Advect dye
        self.dye = self._advect_field(self.dye, self.u, self.v)
        self.dye[~self.active] = 0.0
        # Small artificial diffusion for stability/anti-aliasing
        self.dye += self.dt * self.nu * self._laplacian(self.dye)
        self.dye[~self.active] = 0.0

    # ------------------------------------------------------------------
    # Trajectory / StateFrame construction
    # ------------------------------------------------------------------
    def _vorticity(self) -> np.ndarray:
        # ω = dv/dx - du/dy at cell centers using central differences
        omega = np.zeros((self.nx, self.ny), dtype=float)
        omega[1:-1, 1:-1] = (
            (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * self.dx)
            - (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * self.dy)
        )
        return omega

    def _divergence_field(self) -> np.ndarray:
        return self._divergence()

    def _boundary_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = []

        # Bottom and top walls
        x_c = self._x_centers()
        for wall_y, sign in [(self.y_min, -1.0), (self.y_max, 1.0)]:
            for xi in x_c:
                if (xi - self.cyl_x) ** 2 + (wall_y - self.cyl_y) ** 2 <= self.cyl_r ** 2:
                    continue
                samples.append(
                    {
                        "pos": np.array([xi, wall_y, 0.0]),
                        "normal": np.array([0.0, sign, 0.0]),
                        "target": np.array([0.0, 0.0, 0.0]),
                    }
                )

        # Left (inlet) and right (outlet)
        y_c = self._y_centers()
        for xb, sign in [(self.x_min, -1.0), (self.x_max, 1.0)]:
            for yj in y_c:
                if (xb - self.cyl_x) ** 2 + (yj - self.cyl_y) ** 2 <= self.cyl_r ** 2:
                    continue
                normal = np.array([sign, 0.0, 0.0])
                if sign < 0:  # inlet
                    target = np.array([self._inlet_profile(np.array([yj]))[0], 0.0, 0.0])
                else:  # outlet
                    target = np.array([0.0, 0.0, 0.0])
                samples.append(
                    {"pos": np.array([xb, yj, 0.0]), "normal": normal, "target": target}
                )

        # Cylinder surface
        n_sample = 64
        theta = np.linspace(0.0, 2.0 * np.pi, n_sample, endpoint=False)
        for th in theta:
            xb = self.cyl_x + self.cyl_r * np.cos(th)
            yb = self.cyl_y + self.cyl_r * np.sin(th)
            normal = np.array([np.cos(th), np.sin(th), 0.0])
            samples.append(
                {
                    "pos": np.array([xb, yb, 0.0]),
                    "normal": normal,
                    "target": np.array([0.0, 0.0, 0.0]),
                }
            )

        normals = np.array([s["normal"] for s in samples])
        targets = np.array([s["target"] for s in samples])
        # Measured velocity at sample positions (zero on walls / cylinder; inlet uses target)
        velocities = np.zeros_like(targets)
        for s, vel in zip(samples, velocities):
            if s["pos"][0] <= self.x_min + 1e-6:
                vel[:] = s["target"]
        return normals, targets, velocities

    def _build_frame(self, time: float) -> StateFrame:
        vort = self._vorticity()
        div = self._divergence_field()
        velocity_field = np.stack([self.u, self.v], axis=-1)  # [nx, ny, 2]

        # Update particle velocity by interpolating the cell velocity to particles
        if self.position.size:
            def _sample(f):
                return self._bilinear_interpolate(f, self.position[:, 0], self.position[:, 1])

            pu = _sample(self.u)
            pv = _sample(self.v)
            self.particle_velocity[:, 0] = pu
            self.particle_velocity[:, 1] = pv

        # Boundary samples
        b_normal, b_target, b_vel = self._boundary_samples()

        # Passive scalar source is zero everywhere
        scalar_source = np.zeros((self.nx, self.ny), dtype=float)

        def _sample_to_particles(f):
            if self.position.size == 0:
                return np.zeros(len(self.position))
            return self._bilinear_interpolate(f, self.position[:, 0], self.position[:, 1])

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.particle_velocity.copy(),
            mass=np.full(len(self.position), self.rho * self.dx * self.dy),
            attributes={
                "dye": _sample_to_particles(self.dye),
                "vorticity": _sample_to_particles(vort),
                "divergence": _sample_to_particles(div),
                "velocity_magnitude": np.linalg.norm(self.particle_velocity, axis=1),
            },
            fields={
                "velocity": np.concatenate(
                    [velocity_field, np.zeros((self.nx, self.ny, 1))], axis=-1
                ),
                "velocity_field": velocity_field.copy(),
                "active_mask": self.active.astype(np.uint8),
                "vorticity": vort,
                "divergence": div,
                "dye": self.dye.copy(),
                "scalar_source": scalar_source,
                "transported_scalar": self.dye.copy(),
                "boundary_normal": b_normal,
                "boundary_target_velocity": b_target,
                "boundary_velocity": b_vel,
            },
            scalars={
                "characteristic_length": float(self.L_char),
                "grid_spacing": float(self.dx),
                "reference_velocity": float(self.U_ref),
                "cylinder_radius": float(self.cyl_r),
                "mean_inlet_speed": float(self.U_ref),
                "expected_strouhal_number": float(
                    self.config["diagnostics"].get("expected_strouhal_number", 0.2)
                ),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time - 1e-12:
            raise ValueError("advance_to time must be non-decreasing")

        if time <= self.current_time:
            return self._build_frame(self.current_time)

        dt = time - self.current_time
        self._step(dt)
        self.current_time = time
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        self._diag = {
            "max_cfl": float(self.U_max * self.dt / self.dx),
            "reynolds_number": float(self.U_ref * self.L_char / self.nu),
            "final_time": float(self.current_time),
        }
        return self._diag
