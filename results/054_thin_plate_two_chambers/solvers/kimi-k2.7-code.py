"""2D D2Q9 lattice-Boltzmann solver for the thin-plate two-chamber jet scene.

The implementation ignores the shallow y extrusion and solves the physics on the
x-z cross-section, then broadcasts the 2D result to the three y layers required by
the voxel grid.  The jet is driven by a velocity inlet at the bottom centre, the
thin separating plate is modelled as a one-cell-thick no-slip wall, and the
tracer ``injected_fraction`` is advected with a separate D2Q9 passive-scalar
distribution.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.frame0 = scene.initial_state()
        self.t = 0.0

        # ------------------------------------------------------------------
        # Grid geometry (voxel grid: nodes are treated as cell centres for LBM)
        # ------------------------------------------------------------------
        self.nx, self.ny, self.nz = self.frame0.fields["injected_fraction"].shape
        x = np.linspace(-3.0, 3.0, self.nx)
        z = np.linspace(0.0, 3.6, self.nz)
        y = np.linspace(-0.18, 0.18, self.ny)
        self.dx = x[1] - x[0]
        self.dz = z[1] - z[0]
        self.dy = y[1] - y[0]
        self.vol_per_sample = self.dx * self.dy * self.dz

        # Time step chosen so that 0.02 s samples are integer multiples.
        self.dt = 0.0025
        self.c_lattice = self.dx / self.dt  # physical lattice speed

        # LBM parameters (BGK, D2Q9)
        nu = 0.01  # kinematic viscosity, m^2/s; keeps tau safely above 0.5
        self.tau = 0.5 + 3.0 * nu * self.dt / (self.dx * self.dx)
        self.tau_g = self.tau

        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=float)
        self.cz = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=float)
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        self.W = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
        )

        # ------------------------------------------------------------------
        # Masks on the 2D x-z slice (middle y layer)
        # ------------------------------------------------------------------
        X, Z = np.meshgrid(x, z, indexing="ij")
        self.X = X
        self.Z = Z

        inlet = np.zeros((self.nx, self.nz), dtype=bool)
        inlet[:, 0] = np.abs(X[:, 0]) <= 0.21

        solid = np.zeros((self.nx, self.nz), dtype=bool)
        solid[:, 0] = True
        solid[0, :] = True
        solid[-1, :] = True
        solid[:, -1] = True
        solid &= ~inlet  # bottom wall outside the inlet

        # Thin plate at z=1.8, x in [-2.72, 2.72]; j=36 is the z=1.8 node.
        plate_j = int(round(1.8 / self.dz))
        plate_mask = np.zeros((self.nx, self.nz), dtype=bool)
        plate_mask[:, plate_j] = np.abs(X[:, 0]) <= 2.72
        solid |= plate_mask

        self.inlet_mask_2d = inlet
        self.solid_mask_2d = solid
        self.fluid_mask_2d = ~solid

        # ------------------------------------------------------------------
        # Initial macroscopic fields (middle y layer)
        # ------------------------------------------------------------------
        iy_mid = self.ny // 2
        rho0 = self.frame0.fields["density"][:, iy_mid, :].copy()
        vel0 = self.frame0.fields["velocity"][:, iy_mid, :, :]
        u0 = vel0[..., 0].copy()
        w0 = vel0[..., 2].copy()
        C0 = self.frame0.fields["injected_fraction"][:, iy_mid, :].copy()

        # Avoid negative / overshoot values from the scaffold if any.
        np.clip(C0, 0.0, 1.0, out=C0)

        self.rho = rho0
        self.u = u0
        self.w = w0
        self.C = C0

        # ------------------------------------------------------------------
        # LBM distributions
        # ------------------------------------------------------------------
        self.f = self._equilibrium_f(self.rho, self.u, self.w)
        self.g = self._equilibrium_g(self.C, self.u, self.w)

        # Cumulative injected tracer mass (kg).  Analytic inflow rate.
        inflow_area = 0.42 * 0.36
        self.inflow_rate = 1.18 * 2.4 * inflow_area  # kg/s
        self.cumulative_injected = 0.0

    # ------------------------------------------------------------------
    # Solver API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        # Integrate with the fixed LBM step, snapping the frame time to the
        # requested sample time.
        while self.t + 0.5 * self.dt < time:
            self._step(self.dt)
            self.t += self.dt
        # Prevent time drift across samples.
        self.t = time
        self.cumulative_injected = self.inflow_rate * self.t
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "D2Q9 lattice Boltzmann, BGK",
            "tau": float(self.tau),
            "lattice_speed": float(self.c_lattice),
        }

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # BGK collision
        feq = self._equilibrium_f(self.rho, self.u, self.w)
        geq = self._equilibrium_g(self.C, self.u, self.w)
        self.f -= (self.f - feq) / self.tau
        self.g -= (self.g - geq) / self.tau_g

        # Streaming (zero padding outside the domain)
        self.f = self._stream(self.f)
        self.g = self._stream(self.g)

        # Boundary conditions
        self._apply_boundaries()

        # Recover macroscopic quantities
        self._compute_macroscopic()

    def _stream(self, f: np.ndarray) -> np.ndarray:
        nx, nz = self.nx, self.nz
        f_new = np.empty_like(f)
        for k in range(9):
            cxk = int(self.cx[k])
            czk = int(self.cz[k])
            src = f[k]
            shifted = np.zeros_like(src)
            if cxk > 0 and czk > 0:
                shifted[cxk:, czk:] = src[:-cxk, :-czk]
            elif cxk > 0 and czk < 0:
                shifted[cxk:, :czk] = src[:-cxk, -czk:]
            elif cxk < 0 and czk > 0:
                shifted[:cxk, czk:] = src[-cxk:, :-czk]
            elif cxk < 0 and czk < 0:
                shifted[-cxk:, -czk:] = src[:cxk, :czk]
            elif cxk > 0:
                shifted[cxk:, :] = src[:-cxk, :]
            elif cxk < 0:
                shifted[:cxk, :] = src[-cxk:, :]
            elif czk > 0:
                shifted[:, czk:] = src[:, :-czk]
            elif czk < 0:
                shifted[:, :czk] = src[:, -czk:]
            else:  # rest direction
                shifted = src
            f_new[k] = shifted
        return f_new

    def _apply_boundaries(self) -> None:
        # Bounce-back on solid walls / plate -> no-slip and zero tracer flux.
        if np.any(self.solid_mask_2d):
            f_solid = self.f[:, self.solid_mask_2d]
            for k in range(9):
                self.f[k, self.solid_mask_2d] = f_solid[self.opp[k]]
            g_solid = self.g[:, self.solid_mask_2d]
            for k in range(9):
                self.g[k, self.solid_mask_2d] = g_solid[self.opp[k]]

        # Inlet: fixed velocity (0, 0, 2.4) and tracer concentration C=1.
        if np.any(self.inlet_mask_2d):
            rho_in = np.full_like(self.rho, 1.18)
            u_in = np.zeros_like(self.u)
            w_in = np.full_like(self.w, 2.4)
            feq_in = self._equilibrium_f(rho_in, u_in, w_in)
            geq_in = self._equilibrium_g(np.ones_like(self.C), u_in, w_in)
            for k in range(9):
                self.f[k, self.inlet_mask_2d] = feq_in[k, self.inlet_mask_2d]
                self.g[k, self.inlet_mask_2d] = geq_in[k, self.inlet_mask_2d]

    def _compute_macroscopic(self) -> None:
        self.rho = np.sum(self.f, axis=0)
        # Guard against accidental zero densities (should not happen after BCs).
        denom = np.where(self.rho > 1e-12, self.rho, 1.0)
        u_l = np.sum(self.f * self.cx[:, None, None], axis=0) / denom
        w_l = np.sum(self.f * self.cz[:, None, None], axis=0) / denom
        self.u = u_l * self.dx / self.dt
        self.w = w_l * self.dx / self.dt
        self.C = np.sum(self.g, axis=0)
        np.clip(self.C, 0.0, 1.0, out=self.C)

    # ------------------------------------------------------------------
    # Equilibria
    # ------------------------------------------------------------------
    def _equilibrium_f(self, rho, u, w):
        u_l = u * self.dt / self.dx
        w_l = w * self.dt / self.dx
        eu = self.cx[:, None, None] * u_l[None, :, :] + self.cz[:, None, None] * w_l[None, :, :]
        uu = u_l ** 2 + w_l ** 2
        base = 1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uu[None, :, :]
        return self.W[:, None, None] * rho[None, :, :] * base

    def _equilibrium_g(self, C, u, w):
        u_l = u * self.dt / self.dx
        w_l = w * self.dt / self.dx
        eu = self.cx[:, None, None] * u_l[None, :, :] + self.cz[:, None, None] * w_l[None, :, :]
        uu = u_l ** 2 + w_l ** 2
        base = 1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uu[None, :, :]
        return self.W[:, None, None] * C[None, :, :] * base

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        nx, ny, nz = self.nx, self.ny, self.nz

        # Broadcast 2D slice to the full 3D shallow-extrusion grid.
        rho3 = np.broadcast_to(self.rho[:, None, :], (nx, ny, nz)).copy()
        C3 = np.broadcast_to(self.C[:, None, :], (nx, ny, nz)).copy()

        # Velocity field with v_y = 0.
        velocity = np.zeros((nx, ny, nz, 3))
        velocity[:, :, :, 0] = self.u[:, None, :]
        velocity[:, :, :, 2] = self.w[:, None, :]

        # Solid cells report zero velocity / tracer for rendering.
        solid3 = np.broadcast_to(self.solid_mask_2d[:, None, :], (nx, ny, nz))
        velocity[solid3] = 0.0
        C3[solid3] = 0.0

        # Pressure from LBM density (lattice pressure units; relative jumps only).
        p = self.rho / 3.0
        p3 = np.broadcast_to(p[:, None, :], (nx, ny, nz)).copy()

        # Per-sample arrays.
        position = self.frame0.position
        velocity_ravel = velocity.reshape(-1, 3)
        mass = rho3.ravel() * self.vol_per_sample

        # Conserved mass contributions: ambient + injected.
        ambient_mass = (1.0 - C3.ravel()) * 1.0 * self.vol_per_sample
        injected_mass = C3.ravel() * 1.18 * self.vol_per_sample
        mass_quantity = np.column_stack((ambient_mass, injected_mass))

        # Active mask (1 = fluid, 0 = solid/obstacle).
        active_mask = np.broadcast_to(
            self.fluid_mask_2d[:, None, :], (nx, ny, nz)
        ).astype(float)

        # Wall / inlet boundary samples.
        b_normal, b_target, b_vel = self._boundary_samples()

        # Cumulative mass source for the two materials.
        cumulative_mass_source = np.array(
            [0.0, self.cumulative_injected], dtype=float
        )

        # Interaction (pressure) forces on the five solid boundaries.
        interaction_force = self._interaction_force(p)

        # Diagnostic scalars.
        (
            upper_mass,
            lower_mass,
            p_jump,
            max_speed,
            finite,
        ) = self._diagnostics(p)

        fields = {
            "density": rho3,
            "pressure": p3,
            "velocity": velocity,
            "injected_fraction": C3,
            "active_mask": active_mask,
            "boundary_normal": b_normal,
            "boundary_target_velocity": b_target,
            "boundary_velocity": b_vel,
            "cumulative_mass_source": cumulative_mass_source,
            "interaction_force": interaction_force,
            "velocity_field": velocity.copy(),
        }

        attributes = {
            "injected_fraction": C3.ravel(),
            "mass_quantity": mass_quantity,
        }

        scalars = {
            "grid_spacing": float(self.dx),
            "characteristic_length": 1.0,
            "reference_velocity": 2.4,
            "upper_chamber_injected_mass": float(upper_mass),
            "lower_chamber_injected_mass": float(lower_mass),
            "pressure_jump_across_plate": float(p_jump),
            "maximum_speed": float(max_speed),
            "finite_state": float(finite),
        }

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity_ravel,
            mass=mass,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _boundary_samples(self):
        nx, nz = self.nx, self.nz
        x = self.X[:, 0]
        z = self.Z[0, :]
        normals, targets, velocities = [], [], []

        def append_samples(normal, target, count):
            normals.append(np.tile(normal, (count, 1)))
            targets.append(np.tile(target, (count, 1)))
            velocities.append(np.tile(target, (count, 1)))

        # Bottom wall (impermeable, no inlet).
        mask = self.solid_mask_2d[:, 0] & ~self.inlet_mask_2d[:, 0]
        if np.any(mask):
            append_samples(np.array([0.0, 0.0, 1.0]), np.zeros(3), int(mask.sum()))

        # Left wall.
        mask = self.solid_mask_2d[0, :]
        if np.any(mask):
            append_samples(np.array([1.0, 0.0, 0.0]), np.zeros(3), int(mask.sum()))

        # Right wall.
        mask = self.solid_mask_2d[-1, :]
        if np.any(mask):
            append_samples(np.array([-1.0, 0.0, 0.0]), np.zeros(3), int(mask.sum()))

        # Top wall.
        mask = self.solid_mask_2d[:, -1]
        if np.any(mask):
            append_samples(np.array([0.0, 0.0, -1.0]), np.zeros(3), int(mask.sum()))

        # Thin plate (fluid below, so normal points down).
        plate_j = int(round(1.8 / self.dz))
        plate_i = self.solid_mask_2d[:, plate_j]
        if np.any(plate_i):
            append_samples(np.array([0.0, 0.0, -1.0]), np.zeros(3), int(plate_i.sum()))

        # Inlet.
        inlet_i = self.inlet_mask_2d[:, 0]
        if np.any(inlet_i):
            append_samples(
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 2.4]),
                int(inlet_i.sum()),
            )

        if normals:
            return np.vstack(normals), np.vstack(targets), np.vstack(velocities)
        # Fallback: should never happen.
        return np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))

    def _interaction_force(self, p: np.ndarray):
        forces = []

        def append_force(adj_pressures, normal, area):
            if adj_pressures.size == 0:
                forces.append(np.zeros(3))
            else:
                total_p = np.sum(adj_pressures) * area
                forces.append(total_p * np.array(normal, dtype=float))

        # Bottom wall outside inlet; adjacent fluid cell is j+1.
        mask = self.solid_mask_2d[:, 0] & ~self.inlet_mask_2d[:, 0]
        append_force(p[mask, 1], (0.0, 0.0, 1.0), self.dx * self.dy)

        # Left wall; adjacent fluid cell is i+1.
        mask = self.solid_mask_2d[0, :]
        append_force(p[1, mask], (1.0, 0.0, 0.0), self.dz * self.dy)

        # Right wall; adjacent fluid cell is i-1.
        mask = self.solid_mask_2d[-1, :]
        append_force(p[-2, mask], (-1.0, 0.0, 0.0), self.dz * self.dy)

        # Top wall; adjacent fluid cell is j-1.
        mask = self.solid_mask_2d[:, -1]
        append_force(p[mask, -2], (0.0, 0.0, -1.0), self.dx * self.dy)

        # Plate; adjacent fluid cell is below (j-1).
        plate_j = int(round(1.8 / self.dz))
        plate_i = self.solid_mask_2d[:, plate_j]
        append_force(p[plate_i, plate_j - 1], (0.0, 0.0, -1.0), self.dx * self.dy)

        return np.array(forces)

    def _diagnostics(self, p: np.ndarray):
        fluid = self.fluid_mask_2d
        m_cell = self.C * self.rho * self.vol_per_sample

        upper = (self.Z > 1.85) & fluid
        lower = (self.Z < 1.80) & fluid
        upper_mass = self.ny * float(np.sum(m_cell[upper]))
        lower_mass = self.ny * float(np.sum(m_cell[lower]))

        below = (self.Z < 1.80) & (self.Z > 1.75) & fluid
        above = (self.Z > 1.80) & (self.Z < 1.85) & fluid
        if np.any(below) and np.any(above):
            p_jump = float(np.mean(p[below]) - np.mean(p[above]))
        else:
            p_jump = 0.0

        speed = np.sqrt(self.u ** 2 + self.w ** 2)
        max_speed = float(np.max(speed[fluid])) if np.any(fluid) else 0.0

        finite = 1.0
        if not (
            np.all(np.isfinite(speed))
            and np.all(np.isfinite(self.C))
            and np.all(np.isfinite(self.rho))
        ):
            finite = 0.0

        return upper_mass, lower_mass, p_jump, max_speed, finite
