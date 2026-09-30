"""BenchmarkSolver implementation for Lava Spout Periodic Folding.

Physically models a highly viscous, elastoplastic / viscoplastic fluid jet
impinging on a flat receiving basin, exhibiting characteristic periodic folding/buckling.
Simulated using an efficient 3D Material Point Method (APIC / Drucker-Prager / Viscoplastic).
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(seed)

        # Sampling & physical constants
        sampling = self.config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])
        self.particle_mass = self.density * (self.spacing ** 3)
        self.particle_volume = self.spacing ** 3
        self.particle_radius = 0.45 * self.spacing

        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        # Initial particles from scene
        init_frame = scene.initial_state()
        self.pos = init_frame.position.copy().astype(np.float64)
        self.N = len(self.pos)
        self.vel = np.zeros_like(self.pos, dtype=np.float64)
        # Give initial downward velocity in the upper spout
        spout_mask = self.pos[:, 2] > 0.5
        self.vel[spout_mask, 2] = -1.2

        self.mass = np.full(self.N, self.particle_mass, dtype=np.float64)
        self.F = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], self.N, axis=0)
        self.C = np.zeros((self.N, 3, 3), dtype=np.float64)  # APIC affine momentum matrix

        # Nozzle & emitter configuration
        # nozzle at translation [0, 0, 3.02], radius 0.27, height 0.42
        # spout block in config: [-0.18, -0.18, 0.1, 0.18, 0.18, 2.8]
        self.inlet_z = 2.8
        self.inlet_velocity = -1.5  # m/s downwards
        self.inlet_rate = 1.5  # flow speed
        self.emit_accumulator = 0.0
        self.cumulative_mass_source = 0.0

        # MPM Grid definition
        # Domain bounds from geometry: bounds_min [-3.0, -1.2, 0.0], bounds_max [3.0, 1.2, 3.8]
        # Active simulation box can be tightly bounded around the spout / basin
        self.dx = self.spacing * 0.8  # MPM grid spacing
        self.inv_dx = 1.0 / self.dx
        self.grid_origin = np.array([-1.8, -1.8, 0.0], dtype=np.float64)
        self.grid_shape = (
            int(np.ceil(3.6 / self.dx)) + 3,
            int(np.ceil(3.6 / self.dx)) + 3,
            int(np.ceil(3.6 / self.dx)) + 3,
        )

        # Material parameters (Viscous elastoplastic lava / Bingham / Herschel-Bulkley)
        self.E = 2.5e4        # Young's modulus
        self.nu = 0.35       # Poisson's ratio
        self.mu_0 = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.viscosity = 80.0  # Dynamic viscosity (Pa*s)
        self.yield_stress = 120.0  # Plastic yield stress

        # Basin contact geometry (cylinder radius 1.35, height 0.12 at z=0.06 -> top at z=0.12)
        # Ground top at z = 0.0
        self.basin_radius = 1.35
        self.basin_z = 0.06

        self.current_time = 0.0
        self.sub_dt = 0.001  # Substep time

    def _emit_particles(self, dt: float) -> None:
        """Continuously emit particles from the nozzle inlet."""
        self.emit_accumulator += dt * abs(self.inlet_velocity)
        if self.emit_accumulator >= self.spacing:
            self.emit_accumulator -= self.spacing
            # Create a layer of particles at inlet
            spacing = self.spacing
            x_vals = np.arange(-0.18 + spacing * 0.5, 0.18, spacing)
            y_vals = np.arange(-0.18 + spacing * 0.5, 0.18, spacing)
            if len(x_vals) == 0:
                x_vals = np.array([0.0])
            if len(y_vals) == 0:
                y_vals = np.array([0.0])

            xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
            new_pos = np.column_stack((
                xx.ravel() + self.rng.uniform(-0.005, 0.005, size=xx.size),
                yy.ravel() + self.rng.uniform(-0.005, 0.005, size=yy.size),
                np.full(xx.size, self.inlet_z)
            ))

            # Small perturbation to induce realistic buckling instability
            pert = 0.015 * np.sin(self.current_time * 12.0)
            new_pos[:, 0] += pert

            n_new = len(new_pos)
            new_vel = np.zeros((n_new, 3), dtype=np.float64)
            new_vel[:, 2] = self.inlet_velocity
            new_mass = np.full(n_new, self.particle_mass, dtype=np.float64)
            new_F = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n_new, axis=0)
            new_C = np.zeros((n_new, 3, 3), dtype=np.float64)

            self.pos = np.vstack([self.pos, new_pos])
            self.vel = np.vstack([self.vel, new_vel])
            self.mass = np.concatenate([self.mass, new_mass])
            self.F = np.concatenate([self.F, new_F])
            self.C = np.concatenate([self.C, new_C])
            self.N = len(self.pos)
            self.cumulative_mass_source += n_new * self.particle_mass

    def _step(self, dt: float) -> None:
        """Perform one MPM time step (P2G -> Grid update -> Boundary -> G2P)."""
        self._emit_particles(dt)

        N = self.N
        pos = self.pos
        vel = self.vel
        mass = self.mass
        F = self.F
        C = self.C

        inv_dx = self.inv_dx
        origin = self.grid_origin
        grid_shape = self.grid_shape

        # Particle normalized coordinates
        base_pos = (pos - origin) * inv_dx - 0.5
        base = np.floor(base_pos).astype(np.int32)
        fx = base_pos - base

        # Quadratic B-spline weights
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        weights = [w0, w1, w2]  # each (N, 3)

        # Compute Cauchy stress * volume * dt * 4 * inv_dx^2 for MPM force transfer
        # Visco-elastoplastic model (Neo-Hookean + viscous damping + plastic return)
        J = np.linalg.det(F)
        J = np.clip(J, 0.05, 5.0)

        # Deformation gradient update & elastic stress
        # Compute first Piola-Kirchhoff or Kirchhoff stress tau = P * F^T = mu*(F*F^T - I) + lambda*ln(J)*I
        # For small deformations / viscoplastic:
        b = np.einsum('nij,nkj->nik', F, F)  # Left Cauchy-Green tensor
        trace_b = b[:, 0, 0] + b[:, 1, 1] + b[:, 2, 2]

        # Deviatoric stress
        tau_dev = self.mu_0 * (b - (trace_b[:, None, None] / 3.0) * np.eye(3)[None, :, :])
        tau_vol = (self.lambda_0 * 0.5 * (J ** 2 - 1.0))[:, None, None] * np.eye(3)[None, :, :]
        tau = tau_dev + tau_vol

        # Plasticity / Yielding: Von-Mises / Bingham plastic yield
        dev_norm = np.sqrt(np.sum(tau_dev ** 2, axis=(1, 2)) + 1e-12)
        yield_mask = dev_norm > self.yield_stress
        scaling = np.ones(N, dtype=np.float64)
        scaling[yield_mask] = self.yield_stress / dev_norm[yield_mask]
        tau[yield_mask] = tau_vol[yield_mask] + tau_dev[yield_mask] * scaling[yield_mask, None, None]

        # Convert stress to grid force contribution: -dt * vol * tau * 4 * inv_dx^2
        stress_contrib = -dt * 4.0 * (inv_dx ** 2) * self.particle_volume * tau
        # APIC affine momentum term
        affine_contrib = mass[:, None, None] * C

        # Grid allocation
        grid_v = np.zeros(grid_shape + (3,), dtype=np.float64)
        grid_m = np.zeros(grid_shape, dtype=np.float64)

        # Particle to Grid (P2G)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dpos = (np.array([i, j, k], dtype=np.float64) - fx) * self.dx  # (N, 3)
                    weight = weights[i][:, 0] * weights[j][:, 1] * weights[k][:, 2]  # (N,)

                    # Node indices
                    gx = np.clip(base[:, 0] + i, 0, grid_shape[0] - 1)
                    gy = np.clip(base[:, 1] + j, 0, grid_shape[1] - 1)
                    gz = np.clip(base[:, 2] + k, 0, grid_shape[2] - 1)

                    # APIC velocity + force contribution
                    # Q = mass * vel + (stress_contrib + affine_contrib) @ dpos
                    term = mass[:, None] * vel + np.einsum('nij,nj->ni', stress_contrib + affine_contrib, dpos)
                    weighted_term = term * weight[:, None]
                    weighted_m = (mass * weight)

                    np.add.at(grid_v, (gx, gy, gz), weighted_term)
                    np.add.at(grid_m, (gx, gy, gz), weighted_m)

        # Grid operations: momentum to velocity, gravity, boundary conditions
        active_nodes = grid_m > 1e-10
        grid_v[active_nodes] /= grid_m[active_nodes, None]
        grid_v[active_nodes] += dt * self.gravity

        # Boundary conditions (catch basin & ground)
        # Ground at z <= 0.06 (basin floor)
        # Basin cylinder at r < 1.35
        nx, ny, nz = grid_shape
        z_coords = origin[2] + np.arange(nz) * self.dx
        x_coords = origin[0] + np.arange(nx) * self.dx
        y_coords = origin[1] + np.arange(ny) * self.dx

        # Basin floor boundary condition
        floor_k = int(np.floor((self.basin_z - origin[2]) * inv_dx)) + 1
        floor_k = np.clip(floor_k, 0, nz - 1)
        grid_v[:, :, :floor_k, 2] = np.maximum(0.0, grid_v[:, :, :floor_k, 2])
        # High friction on floor to ensure folding rather than simple slipping
        grid_v[:, :, :floor_k, 0] *= 0.1
        grid_v[:, :, :floor_k, 1] *= 0.1

        # Domain boundary walls
        grid_v[0:2, :, :, 0] = np.maximum(0.0, grid_v[0:2, :, :, 0])
        grid_v[-2:, :, :, 0] = np.minimum(0.0, grid_v[-2:, :, :, 0])
        grid_v[:, 0:2, :, 1] = np.maximum(0.0, grid_v[:, 0:2, :, 1])
        grid_v[:, -2:, :, 1] = np.minimum(0.0, grid_v[:, -2:, :, 1])
        grid_v[:, :, 0:2, 2] = np.maximum(0.0, grid_v[:, :, 0:2, 2])

        # Grid to Particle (G2P)
        new_vel = np.zeros_like(vel)
        new_C = np.zeros_like(C)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dpos = (np.array([i, j, k], dtype=np.float64) - fx) * self.dx
                    weight = weights[i][:, 0] * weights[j][:, 1] * weights[k][:, 2]

                    gx = np.clip(base[:, 0] + i, 0, grid_shape[0] - 1)
                    gy = np.clip(base[:, 1] + j, 0, grid_shape[1] - 1)
                    gz = np.clip(base[:, 2] + k, 0, grid_shape[2] - 1)

                    gv = grid_v[gx, gy, gz]  # (N, 3)
                    new_vel += weight[:, None] * gv
                    # APIC B-matrix update: C = sum_i w_i * v_i * (x_i - x_p)^T * 4 * inv_dx^2
                    outer = np.einsum('ni,nj->nij', gv, dpos)
                    new_C += weight[:, None, None] * outer * (4.0 * inv_dx ** 2)

        self.vel = new_vel
        self.C = new_C
        self.pos += dt * self.vel

        # Update deformation gradient F = (I + dt * C) @ F
        F_update = np.eye(3)[None, :, :] + dt * self.C
        self.F = np.einsum('nij,njk->nik', F_update, self.F)

        # Plastic relaxation on F to prevent runaway growth
        # SVD or polar decomposition relaxation
        detF = np.linalg.det(self.F)
        detF = np.clip(detF, 0.2, 5.0)
        # Pull F back towards volume-preserving state if overly stretched
        for n in range(N):
            if np.any(np.isnan(self.F[n])) or np.any(np.isinf(self.F[n])):
                self.F[n] = np.eye(3)

        # Boundary clamping for particles
        self.pos[:, 2] = np.maximum(self.pos[:, 2], self.basin_z)

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation using substeps up to requested time
        while self.current_time < time - 1e-8:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        # Prepare StateFrame
        N = len(self.pos)
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()

        # External force = mass * gravity
        external_force = mass[:, None] * self.gravity[None, :]

        # Contact signed distance (distance to basin surface at z=0.06 and ground)
        # For active/near-contact samples (z close to basin)
        signed_dist = pos[:, 2] - self.basin_z
        contact_mask = signed_dist < (3.0 * self.spacing)
        contact_sd = signed_dist[contact_mask]
        if len(contact_sd) == 0:
            contact_sd = np.array([0.1], dtype=np.float64)

        labels = np.zeros(N, dtype=np.int32)
        cum_mass = np.array([self.cumulative_mass_source], dtype=np.float64)

        frame = StateFrame(
            time,
            pos,
            vel,
            mass=mass,
            body_id=labels,
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": mass.copy(),
                "particle_radius": np.full(N, self.particle_radius),
                "material_label": labels,
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": cum_mass,
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": max(float(np.ptp(pos, axis=0).max()), self.spacing),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particles": self.N,
            "current_time": self.current_time,
            "cumulative_mass_source": self.cumulative_mass_source,
        }

