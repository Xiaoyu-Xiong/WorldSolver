"""Physics-based MPM / Snow Constitutive Solver for Two Snowballs Head-On."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # Particles
        self.n_particles = 0
        self.pos = None
        self.vel = None
        self.mass = None
        self.vol = None
        self.body_id = None
        self.F = None
        self.F_p = None
        self.F_e = None
        self.C = None  # APIC affine momentum matrix (3, 3) per particle

        # Grid definition
        self.dx = 0.06
        self.inv_dx = 1.0 / self.dx
        self.grid_bounds_min = np.array([-4.5, -3.0, 0.0])
        self.grid_bounds_max = np.array([4.5, 3.0, 2.5])
        self.grid_shape = None

        # Snow constitutive parameters (Stomakhin et al. 2013 MPM Snow)
        self.E0 = 1.4e5  # Young's modulus
        self.nu = 0.2    # Poisson's ratio
        self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lambda0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.theta_c = 2.5e-2  # Critical compression
        self.theta_s = 7.5e-3  # Critical stretch
        self.hardening = 10.0  # Hardening coefficient

        # Physics & numerical settings
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.floor_z = 0.0
        self.friction_floor = 0.6
        self.dt = 1e-3  # Simulation substep dt

        # Diagnostics
        self.substeps_taken = 0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        self.pos = np.ascontiguousarray(init_frame.position, dtype=np.float64)
        self.vel = np.ascontiguousarray(init_frame.velocity, dtype=np.float64)
        self.mass = np.ascontiguousarray(init_frame.mass, dtype=np.float64)
        self.body_id = np.ascontiguousarray(init_frame.body_id, dtype=np.int64)
        self.n_particles = len(self.pos)

        if "reference_volume" in init_frame.attributes:
            self.vol = np.ascontiguousarray(init_frame.attributes["reference_volume"], dtype=np.float64)
        else:
            nominal_density = float(scene.config["snowballs"]["nominal_density"])
            self.vol = self.mass / nominal_density

        self.gravity = np.ascontiguousarray(scene.config["gravity"], dtype=np.float64)
        self.floor_z = float(scene.config["floor"]["top_z"])

        # Deformation gradients
        self.F = np.tile(np.eye(3, dtype=np.float64), (self.n_particles, 1, 1))
        self.F_e = np.tile(np.eye(3, dtype=np.float64), (self.n_particles, 1, 1))
        self.F_p = np.tile(np.eye(3, dtype=np.float64), (self.n_particles, 1, 1))
        self.C = np.zeros((self.n_particles, 3, 3), dtype=np.float64)

        # Determine grid dimensions
        self.grid_shape = tuple(np.ceil((self.grid_bounds_max - self.grid_bounds_min) * self.inv_dx).astype(int) + 1)
        self.substeps_taken = 0

    def _substep(self, dt: float) -> None:
        """Single MPM (MLS-MPM / APIC) snow simulation step."""
        inv_dx = self.inv_dx
        dx = self.dx
        inv_dx_sq = inv_dx * inv_dx
        nx, ny, nz = self.grid_shape

        # Grid buffers
        grid_v = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        grid_m = np.zeros((nx, ny, nz), dtype=np.float64)

        # 1. Compute stress for each particle
        # Snow constitutive model with plastic singular value clamping
        U, sig, Vt = np.linalg.svd(self.F_e)

        # Plasticity clamping
        sig_clamped = np.clip(sig, 1.0 - self.theta_c, 1.0 + self.theta_s)

        # Update F_p and F_e
        # F = F_e * F_p -> F_e_new = U * sig_clamped * Vt
        # F_p_new = Vt.T * inv(sig_clamped) * U.T * F
        self.F_e = U @ (sig_clamped[..., None] * Vt)
        inv_sig_clamped = 1.0 / sig_clamped
        self.F_p = (Vt.swapaxes(-1, -2) * inv_sig_clamped[..., None, :]) @ (U.swapaxes(-1, -2) @ self.F)

        # Hardening / softening: exp(h * (1 - det(F_p)))
        det_Fp = np.linalg.det(self.F_p)
        hardening_fac = np.exp(np.clip(self.hardening * (1.0 - det_Fp), -10.0, 10.0))
        mu = self.mu0 * hardening_fac
        lam = self.lambda0 * hardening_fac

        # Polar decomposition of F_e for corotational stress: F_e = R * S, R = U @ Vt
        R = U @ Vt
        J_e = sig_clamped[:, 0] * sig_clamped[:, 1] * sig_clamped[:, 2]

        # 1st Piola-Kirchhoff stress: P(F_e) = 2*mu*(F_e - R) + lam*(J_e - 1)*J_e * F_e^(-T)
        # Cauchy stress * V0: Cauchy * V0 = P * F_e^T * V0 / J_e * J_e = P * F_e^T * V0
        # In APIC/MLS-MPM, the particle stress contribution to the affine momentum matrix:
        # Q = -dt * 4 * inv_dx^2 * V0 * (2*mu*(F_e - R) @ F_e^T + lam*(J_e - 1)*J_e * I)
        P = 2.0 * mu[:, None, None] * (self.F_e - R)
        # add lam * (J_e - 1) * J_e * F_e^(-T)
        # Note: F_e^(-T) = U @ diag(1/sig_clamped) @ Vt
        F_e_inv_T = U @ (inv_sig_clamped[..., None] * Vt)
        P += (lam * (J_e - 1.0) * J_e)[:, None, None] * F_e_inv_T

        # Particle stress contribution (Cauchy * V0): PF_T = P @ F_e.T
        PF_T = P @ self.F_e.swapaxes(-1, -2)

        # APIC affine stress term: -dt * 4 * inv_dx^2 * (PF_T * V0) + mass * C
        stress_term = -4.0 * inv_dx_sq * dt * self.vol[:, None, None] * PF_T
        affine_momentum = self.mass[:, None, None] * self.C + stress_term

        # 2. P2G (Particle to Grid transfer) using quadratic B-spline kernel
        pos_grid = (self.pos - self.grid_bounds_min) * inv_dx
        base = np.floor(pos_grid - 0.5).astype(int)
        fx = pos_grid - base.astype(float)

        # B-spline weights
        w0 = 0.5 * (1.5 - fx)**2
        w1 = 0.75 - (fx - 1.0)**2
        w2 = 0.5 * (fx - 0.5)**2

        w_list = [w0, w1, w2]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = np.array([i, j, k], dtype=float)
                    weight = w_list[i][:, 0] * w_list[j][:, 1] * w_list[k][:, 2]

                    # dpos in physical units
                    dpos = (offset[None, :] - fx) * dx

                    # Grid index
                    gx = base[:, 0] + i
                    gy = base[:, 1] + j
                    gz = base[:, 2] + k

                    # Boundary check
                    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (gz >= 0) & (gz < nz)
                    if not np.any(valid):
                        continue

                    idx_g = (gx[valid], gy[valid], gz[valid])
                    w_v = weight[valid]
                    m_v = self.mass[valid]

                    # mv_contribution = weight * (mass * vel + affine_momentum @ dpos)
                    # affine_momentum @ dpos: (M, 3, 3) @ (M, 3, 1) -> (M, 3)
                    aff_dpos = np.einsum('nij,nj->ni', affine_momentum[valid], dpos[valid])
                    pv = m_v[:, None] * self.vel[valid] + aff_dpos

                    np.add.at(grid_m, idx_g, w_v * m_v)
                    for d in range(3):
                        np.add.at(grid_v[..., d], idx_g, w_v * pv[:, d])

        # 3. Grid operations: momentum to velocity, external forces (gravity), boundary conditions
        active = grid_m > 1e-12
        grid_v[active] /= grid_m[active, None]

        # Apply gravity
        grid_v[active] += dt * self.gravity[None, :]

        # Floor collision at z = floor_z (gz corresponding to z <= floor_z)
        floor_gz = int(np.floor((self.floor_z - self.grid_bounds_min[2]) * inv_dx))
        if 0 <= floor_gz < nz:
            # Floor boundary condition: no penetration (vz >= 0) and friction
            for gz in range(min(floor_gz + 2, nz)):
                # Normal velocity
                vz = grid_v[:, :, gz, 2]
                penetrating = vz < 0.0
                if np.any(penetrating):
                    # Separate tangential and normal components
                    vx = grid_v[:, :, gz, 0]
                    vy = grid_v[:, :, gz, 1]
                    v_tan = np.sqrt(vx * vx + vy * vy)

                    # Normal impulse magnitude: -vz
                    normal_impulse = -vz
                    max_tan_friction = self.friction_floor * normal_impulse

                    # Apply Coulomb friction
                    slide = v_tan > 1e-8
                    scale = np.ones_like(v_tan)
                    scale[slide] = np.maximum(0.0, 1.0 - max_tan_friction[slide] / v_tan[slide])

                    grid_v[:, :, gz, 0] *= scale
                    grid_v[:, :, gz, 1] *= scale
                    grid_v[:, :, gz, 2] = np.maximum(0.0, vz)

        # Domain bounds (sticky / slip boundary to prevent escaping)
        grid_v[0:2, :, :, :] = 0.0
        grid_v[-2:, :, :, :] = 0.0
        grid_v[:, 0:2, :, :] = 0.0
        grid_v[:, -2:, :, :] = 0.0
        grid_v[:, :, 0:2, 2] = np.maximum(0.0, grid_v[:, :, 0:2, 2])
        grid_v[:, :, -2:, :] = 0.0

        # 4. G2P (Grid to Particle transfer): update particle velocity, position, deformation gradient, APIC C
        new_vel = np.zeros_like(self.vel)
        new_C = np.zeros_like(self.C)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = np.array([i, j, k], dtype=float)
                    weight = w_list[i][:, 0] * w_list[j][:, 1] * w_list[k][:, 2]
                    dpos = (offset[None, :] - fx) * dx

                    gx = base[:, 0] + i
                    gy = base[:, 1] + j
                    gz = base[:, 2] + k

                    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (gz >= 0) & (gz < nz)
                    if not np.any(valid):
                        continue

                    idx_g = (gx[valid], gy[valid], gz[valid])
                    gv = grid_v[idx_g]  # (M, 3)
                    w_v = weight[valid, None]

                    new_vel[valid] += w_v * gv
                    # APIC C update: C += 4 * inv_dx^2 * weight * outer(gv, dpos)
                    # outer(gv, dpos): (M, 3, 1) * (M, 1, 3) -> (M, 3, 3)
                    new_C[valid] += (4.0 * inv_dx_sq * w_v)[:, :, None] * (gv[:, :, None] * dpos[valid, None, :])

        self.vel = new_vel
        self.C = new_C

        # Particle position advection
        self.pos += dt * self.vel

        # Particle deformation gradient update: F = (I + dt * C) @ F
        I3 = np.eye(3, dtype=np.float64)
        grad_v = self.C  # In MLS-MPM / APIC with affine transfer, C is grad(v)
        self.F = (I3[None, :, :] + dt * grad_v) @ self.F

        # Enforce floor collision on particles directly as a safety guard
        below_floor = self.pos[:, 2] < self.floor_z
        if np.any(below_floor):
            self.pos[below_floor, 2] = self.floor_z
            self.vel[below_floor, 2] = np.maximum(0.0, self.vel[below_floor, 2])

        self.substeps_taken += 1

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until target_time
        while self.current_time < target_time - 1e-9:
            step_dt = min(self.dt, target_time - self.current_time)
            self._substep(step_dt)
            self.current_time += step_dt

        # Build StateFrame
        gravity = self.gravity
        ext_force = self.mass[:, None] * gravity[None, :]
        floor_dist = self.pos[:, 2] - self.floor_z

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "reference_volume": self.vol.copy(),
                "deformation_gradient": self.F.copy(),
                "initial_component_id": self.body_id.copy(),
            },
            fields={
                "external_force": ext_force,
                "floor_signed_distance": floor_dist,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "initial_total_mass": float(self.mass.sum()),
                "particle_spacing": float(self.scene.config["snowballs"]["particle_spacing"]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps_taken": self.substeps_taken,
            "current_time": self.current_time,
            "n_particles": self.n_particles,
        }

