"""Deterministic MPM/Elastoplastic Snowball Wall Sticky Solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.time = 0.0

        # Particles
        self.x = None # (N, 3) position
        self.v = None # (N, 3) velocity
        self.C = None # (N, 3, 3) affine APIC/MLS-MPM momentum matrix
        self.F_E = None # (N, 3, 3) elastic deformation gradient
        self.F_P = None # (N, 3, 3) plastic deformation gradient
        self.mass = None # (N,) mass
        self.v0 = None # (N,) initial volume
        self.N = 0

        # Material parameters (from Teran snow MPM paper)
        self.E0 = 1.4e5 # Young's modulus (Pa)
        self.nu = 0.2 # Poisson's ratio
        self.mu_0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lambda_0 = (self.E0 * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.theta_c = 2.5e-2 # Critical compression
        self.theta_s = 7.5e-3 # Critical stretch
        self.xi = 10.0 # Hardening coefficient

        # Grid parameters
        self.dx = 0.04
        self.inv_dx = 1.0 / self.dx
        self.grid_origin = np.array([-2.5, -1.5, -0.2], dtype=np.float64)
        self.grid_res = (70, 75, 60) # nx, ny, nz

        # Scene parameters
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.wall_x = 0.0
        self.floor_z = 0.0
        self.contact_length_scale = 0.065
        self.particle_radius = 0.025

        # Substepping
        self.sub_dt = 2e-4

        # Sticky contact state
        self.stuck_mask = None

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        init_frame = scene.initial_state()
        self.time = 0.0

        self.x = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.v = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.N = len(self.x)

        nominal_density = float(self.config["snowball"]["nominal_density"])
        self.v0 = self.mass / nominal_density
        self.C = np.zeros((self.N, 3, 3), dtype=np.float64)
        self.F_E = np.tile(np.eye(3, dtype=np.float64), (self.N, 1, 1))
        self.F_P = np.tile(np.eye(3, dtype=np.float64), (self.N, 1, 1))

        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        contact_cfg = self.config.get("contact", {})
        self.wall_x = float(contact_cfg.get("wall_inner_x", 0.0))
        self.floor_z = float(contact_cfg.get("floor_z", 0.0))
        self.contact_length_scale = float(contact_cfg.get("contact_length_scale", 0.065))
        self.particle_radius = float(self.config["snowball"].get("particle_spacing", 0.065)) * 0.5

        self.stuck_mask = np.zeros(self.N, dtype=bool)

    def _step(self, dt: float) -> None:
        """Single MLS-MPM step with elastoplastic snow model and sticky wall collision."""
        nx, ny, nz = self.grid_res
        grid_m = np.zeros((nx, ny, nz), dtype=np.float64)
        grid_v = np.zeros((nx, ny, nz, 3), dtype=np.float64)

        # 1. Compute stress for all particles
        # Polar SVD on F_E: F_E = U @ S @ Vh
        U, S, Vh = np.linalg.svd(self.F_E) # S has shape (N, 3)
        R = np.matmul(U, Vh) # Rotation part

        # Hardening
        J_P = np.linalg.det(self.F_P)
        exp_factor = np.exp(self.xi * (1.0 - J_P))
        mu = self.mu_0 * exp_factor
        lmbda = self.lambda_0 * exp_factor

        J_E = np.prod(S, axis=1) # (N,)

        # Cauchy stress * volume: P = 2*mu*(F_E - R) + lambda*(J_E - 1)*J_E * F_E^-T
        # In MLS-MPM, stress contribution: stress_contrib = -dt * 4 * inv_dx^2 * V_p * sigma
        # Using Kirchhoff stress tau = P @ F_E^T:
        # tau = 2*mu*(F_E - R)@F_E^T + lambda*(J_E - 1)*J_E * I
        F_E_T = np.transpose(self.F_E, (0, 2, 1))
        tau = 2.0 * mu[:, None, None] * np.matmul(self.F_E - R, F_E_T) + \
              lmbda[:, None, None] * ((J_E - 1.0) * J_E)[:, None, None] * np.eye(3)[None, :, :]

        # MLS-MPM stress term for affine momentum: -dt * (4 / dx^2) * V_0 * tau
        # Note: 4 * inv_dx^2 is for B-spline kernel (D_p^-1 = 4 / dx^2)
        stress_term = -dt * (4.0 * self.inv_dx * self.inv_dx) * self.v0[:, None, None] * tau
        affine_momentum = self.mass[:, None, None] * self.C + stress_term

        # 2. P2G (Particle to Grid) transfer with quadratic B-splines
        x_grid = (self.x - self.grid_origin) * self.inv_dx
        base = np.floor(x_grid - 0.5).astype(int)
        fx = x_grid - base.astype(float)

        # Quadratic weights: w[0] = 0.5 * (1.5 - fx)^2, w[1] = 0.75 - (fx - 1)^2, w[2] = 0.5 * (fx - 0.5)^2
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = np.array([i, j, k], dtype=np.float64)
                    weight = w[i][:, 0] * w[j][:, 1] * w[k][:, 2] # (N,)
                    dpos = (offset[None, :] - fx) * self.dx # (N, 3)

                    # grid index
                    gx = base[:, 0] + i
                    gy = base[:, 1] + j
                    gz = base[:, 2] + k

                    # check valid bounds
                    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (gz >= 0) & (gz < nz)
                    if not np.any(valid):
                        continue

                    gx_v = gx[valid]
                    gy_v = gy[valid]
                    gz_v = gz[valid]
                    w_v = weight[valid]
                    m_v = self.mass[valid]
                    v_v = self.v[valid]

                    # affine term: C @ dpos
                    aff_v = np.einsum('nij,nj->ni', affine_momentum[valid], dpos[valid])

                    p_mass = w_v * m_v
                    p_mom = w_v[:, None] * (m_v[:, None] * v_v + aff_v)

                    np.add.at(grid_m, (gx_v, gy_v, gz_v), p_mass)
                    np.add.at(grid_v, (gx_v, gy_v, gz_v), p_mom)

        # 3. Grid operations: normalize momentum to velocity, apply gravity and boundary conditions
        active = grid_m > 1e-12
        grid_v[active] /= grid_m[active, None]
        grid_v[active] += dt * self.gravity

        # Grid positions
        # Wall is at x = wall_x (0.0). Sticky contact condition on wall:
        # If a grid node touches or crosses the wall at x >= wall_x, set velocity = 0 (sticky)
        for i in range(nx):
            gx_pos = self.grid_origin[0] + i * self.dx
            if gx_pos >= self.wall_x:
                grid_v[i, :, :] = 0.0 # Sticky wall boundary

        # Floor is at z = floor_z (0.0)
        for k in range(nz):
            gz_pos = self.grid_origin[2] + k * self.dx
            if gz_pos <= self.floor_z:
                # Sticky / high-friction floor
                grid_v[:, :, k, 2] = np.maximum(grid_v[:, :, k, 2], 0.0)
                grid_v[:, :, k, :2] *= 0.5 # Friction

        # Outer domain boundaries
        grid_v[0, :, :] = 0.0
        grid_v[-1, :, :] = 0.0
        grid_v[:, 0, :] = 0.0
        grid_v[:, -1, :] = 0.0
        grid_v[:, :, 0] = 0.0
        grid_v[:, :, -1] = 0.0

        # 4. G2P (Grid to Particle) transfer
        new_v = np.zeros_like(self.v)
        new_C = np.zeros_like(self.C)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = np.array([i, j, k], dtype=np.float64)
                    weight = w[i][:, 0] * w[j][:, 1] * w[k][:, 2] # (N,)
                    dpos = (offset[None, :] - fx) * self.dx # (N, 3)

                    gx = base[:, 0] + i
                    gy = base[:, 1] + j
                    gz = base[:, 2] + k

                    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (gz >= 0) & (gz < nz)
                    if not np.any(valid):
                        continue

                    gx_v = gx[valid]
                    gy_v = gy[valid]
                    gz_v = gz[valid]

                    gv = grid_v[gx_v, gy_v, gz_v] # (N_valid, 3)
                    w_v = weight[valid, None] # (N_valid, 1)

                    new_v[valid] += w_v * gv
                    # APIC C update: 4 * inv_dx^2 * outer(w * gv, dpos)
                    term = 4.0 * self.inv_dx * self.inv_dx * w_v[:, :, None] * (gv[:, :, None] * dpos[valid, None, :])
                    new_C[valid] += term

        self.v = new_v
        self.C = new_C

        # Update deformation gradient F
        # grad_v = C
        grad_v = self.C
        F_total = np.matmul(np.eye(3)[None, :, :] + dt * grad_v, np.matmul(self.F_E, self.F_P))

        # Snow plasticity return mapping: clamp singular values of F_E
        # First compute trial elastic deformation gradient:
        F_E_trial = np.matmul(np.eye(3)[None, :, :] + dt * grad_v, self.F_E)
        U, S, Vh = np.linalg.svd(F_E_trial)

        # Clamp singular values: S_clamped = clamp(S, 1 - theta_c, 1 + theta_s)
        S_clamped = np.clip(S, 1.0 - self.theta_c, 1.0 + self.theta_s)

        # Reconstruct F_E and update F_P
        # F_E = U @ diag(S_clamped) @ Vh
        S_mat = np.zeros((self.N, 3, 3), dtype=np.float64)
        for d in range(3):
            S_mat[:, d, d] = S_clamped[:, d]
        self.F_E = np.matmul(U, np.matmul(S_mat, Vh))

        # F_P = V @ diag(S_clamped)^-1 @ U^T @ F_total
        S_inv = np.zeros((self.N, 3, 3), dtype=np.float64)
        for d in range(3):
            S_inv[:, d, d] = 1.0 / np.maximum(S_clamped[:, d], 1e-4)
        V = np.transpose(Vh, (0, 2, 1))
        U_T = np.transpose(U, (0, 2, 1))
        self.F_P = np.matmul(V, np.matmul(S_inv, np.matmul(U_T, F_total)))

        # Particle boundary handling and position advection
        # Particle sticky contact check
        at_wall = self.x[:, 0] >= (self.wall_x - self.particle_radius * 0.5)
        self.stuck_mask = self.stuck_mask | at_wall

        # For stuck particles, zero their velocity
        self.v[self.stuck_mask] = 0.0
        self.C[self.stuck_mask] = 0.0

        # Advect particles
        self.x += dt * self.v

        # Floor collision
        floor_contact = self.x[:, 2] <= (self.floor_z + self.particle_radius)
        self.x[floor_contact, 2] = self.floor_z + self.particle_radius
        self.v[floor_contact, 2] = np.maximum(self.v[floor_contact, 2], 0.0)
        self.v[floor_contact, 0] *= 0.5
        self.v[floor_contact, 1] *= 0.5

        # Wall collision clamp
        wall_contact = self.x[:, 0] >= (self.wall_x - self.particle_radius * 0.2)
        self.x[wall_contact, 0] = np.minimum(self.x[wall_contact, 0], self.wall_x - self.particle_radius * 0.2)
        self.v[wall_contact] = 0.0
        self.C[wall_contact] = 0.0

    def advance_to(self, target_time: float) -> StateFrame:
        # Step physics up to target_time
        while self.time < target_time - 1e-9:
            dt = min(self.sub_dt, target_time - self.time)
            self._step(dt)
            self.time += dt

        self.time = target_time

        # Build StateFrame
        gravity = self.gravity
        wall_gap = self.wall_x - self.x[:, 0]
        floor_gap = self.x[:, 2] - self.floor_z
        contact_sd = np.concatenate([wall_gap, floor_gap])

        frame = StateFrame(
            time=self.time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "reference_volume": self.v0.copy(),
                "deformation_gradient": self.F_E.copy(),
                "initial_component_id": np.zeros(self.N, dtype=np.int64),
            },
            fields={
                "external_force": self.mass[:, None] * gravity[None, :],
                "contact_signed_distance": contact_sd,
                "wall_signed_distance": wall_gap,
                "floor_signed_distance": floor_gap,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "initial_total_mass": float(self.mass.sum()),
                "initial_total_volume": float(self.v0.sum()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "num_particles": int(self.N),
            "num_stuck_particles": int(np.sum(self.stuck_mask)) if self.stuck_mask is not None else 0,
        }

