"""Viscoelastic Fluid MPM Solver for Pool Splash and Rebound.

Simulates viscoelastic fluid behavior using the Material Point Method (MPM)
with a corotational viscoelastic / elastoplastic constitutive model
(Goktekin et al. 2004 / Stomakhin et al. 2014).
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.seed = 12125

        # Particles
        self.n_particles = 0
        self.x = None            # positions (N, 3)
        self.v = None            # velocities (N, 3)
        self.m = None            # masses (N,)
        self.F_e = None          # elastic deformation gradient (N, 3, 3)
        self.C = None            # affine velocity gradient APIC (N, 3, 3)
        self.labels = None       # material label (N,)
        self.particle_radius = None

        # Grid parameters
        self.dx = 0.10
        self.inv_dx = 1.0 / 0.10
        self.grid_origin = np.array([-1.8, -1.0, -0.1], dtype=np.float64)
        self.grid_shape = (38, 22, 42)

        # Material parameters (Viscoelastic Fluid - Goktekin et al. / Sifakis / Stomakhin)
        self.rho0 = 1350.0       # kg/m^3
        self.E = 1.2e4           # Young's modulus (Pa) - soft elastic rebound
        self.nu = 0.35           # Poisson's ratio
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.relaxation_rate = 1.8   # Plastic/viscous relaxation rate (1/s)
        self.max_strain = 0.6        # Yield strain threshold for plastic flow
        self.damping = 0.002         # Artificial numerical damping

        # Physical constants
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.contact_length_scale = 0.14

        # Cumulative source
        self.cum_mass_source = 0.0

        # Substep dt
        self.sub_dt = 1e-3

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.x = np.array(init_frame.position, dtype=np.float64).copy()
        self.v = np.array(init_frame.velocity, dtype=np.float64).copy()
        self.m = np.array(init_frame.mass, dtype=np.float64).copy()
        self.n_particles = len(self.x)

        self.F_e = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], self.n_particles, axis=0)
        self.C = np.zeros((self.n_particles, 3, 3), dtype=np.float64)

        self.labels = np.array(init_frame.body_id, dtype=np.int32).copy() if init_frame.body_id is not None else np.zeros(self.n_particles, dtype=np.int32)
        if "particle_radius" in init_frame.attributes:
            self.particle_radius = np.array(init_frame.attributes["particle_radius"], dtype=np.float64).copy()
        else:
            self.particle_radius = np.full(self.n_particles, 0.45 * scene.config["sampling"]["spacing"])

        if "contact_length_scale" in init_frame.scalars:
            self.contact_length_scale = float(init_frame.scalars["contact_length_scale"])
        else:
            self.contact_length_scale = float(scene.config["sampling"]["spacing"])

        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.current_time = 0.0

        # Set grid dimensions dynamically to comfortably bound the domain
        spacing = float(scene.config["sampling"]["spacing"])
        self.dx = spacing * 0.75
        self.inv_dx = 1.0 / self.dx
        self.grid_origin = np.array([-2.0, -1.2, -0.2], dtype=np.float64)
        grid_max = np.array([2.0, 1.2, 3.2], dtype=np.float64)
        self.grid_shape = tuple(np.ceil((grid_max - self.grid_origin) / self.dx).astype(int) + 3)

    def _substep(self, dt: float) -> None:
        gx, gy, gz = self.grid_shape
        grid_v = np.zeros((gx, gy, gz, 3), dtype=np.float64)
        grid_m = np.zeros((gx, gy, gz), dtype=np.float64)

        # Particle volumes
        p_vol = self.m / self.rho0  # shape (N,)

        # B-spline / Quadratic kernel weights
        # Particle base coordinates in grid space
        pos_grid = (self.x - self.grid_origin) * self.inv_dx
        base = np.floor(pos_grid - 0.5).astype(int)
        fx = pos_grid - base

        # Quadratic B-spline weights
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        # Precompute Cauchy stress * volume * 4 * inv_dx^2 for MPM APIC transfer
        # Corotational / Neo-Hookean viscoelastic model
        # SVD of F_e to handle large deformations stably
        stress_vol = np.zeros((self.n_particles, 3, 3), dtype=np.float64)

        # Batch SVD for constitutive model
        # Fe has shape (N, 3, 3)
        U, sig, Vt = np.linalg.svd(self.F_e)
        det_UV = np.linalg.det(U @ Vt)
        flip = det_UV < 0
        if np.any(flip):
            U[flip, :, -1] *= -1
            sig[flip, -1] *= -1

        R = U @ Vt
        J = np.prod(sig, axis=-1)  # shape (N,)
        vol0 = p_vol  # shape (N,)

        # Invert Fe in batch
        try:
            Fe_inv_T = np.linalg.inv(self.F_e).transpose(0, 2, 1)
        except Exception:
            Fe_inv_T = np.repeat(np.eye(3)[None, :, :], self.n_particles, axis=0)

        # P = 2*mu*(Fe - R) + lam*(J - 1)*J*inv(Fe)^T
        P = 2.0 * self.mu * (self.F_e - R) + (self.lam * (J - 1.0))[:, None, None] * Fe_inv_T
        stress_vol = (P @ self.F_e.transpose(0, 2, 1)) * vol0[:, None, None]

        # Plastic / viscous relaxation: relax F_e towards R
        relax_factor = np.exp(-self.relaxation_rate * dt)
        sig_clamped = np.clip(sig, 1.0 - self.max_strain, 1.0 + self.max_strain)
        # S_mat = diag(sig_clamped)
        S_mat = np.zeros((self.n_particles, 3, 3), dtype=np.float64)
        for d in range(3):
            S_mat[:, d, d] = sig_clamped[:, d]
        Fe_relaxed = U @ S_mat @ Vt
        self.F_e = Fe_relaxed * relax_factor + (1.0 - relax_factor) * R

        # APIC transfer: P2G
        # D_inv = 4 * inv_dx^2
        affine_mul = 4.0 * (self.inv_dx ** 2)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[i][:, 0] * w[j][:, 1] * w[k][:, 2]  # shape (N,)
                    dpos = (np.column_stack([base[:, 0] + i, base[:, 1] + j, base[:, 2] + k]) - pos_grid) * self.dx

                    # APIC velocity + stress contribution:
                    # momentum = weight * (m * v + m * C * dpos - dt * stress_vol * dpos * 4 * inv_dx^2)
                    # Note: stress force on grid = - sum_p stress_vol_p * grad_w
                    # In APIC: grad_w_p_i = weight * dpos * 4 * inv_dx^2

                    # Vectorized computation for all particles:
                    # m * v term:
                    mv = self.m[:, None] * self.v
                    # m * C @ dpos term:
                    mCdpos = np.einsum('n,nij,nj->ni', self.m, self.C, dpos)
                    # Stress term:
                    stress_term = -dt * affine_mul * np.einsum('nij,nj->ni', stress_vol, dpos)

                    contrib_p = weight[:, None] * (mv + mCdpos + stress_term)
                    contrib_m = weight * self.m

                    # Accumulate to grid
                    bi = base[:, 0] + i
                    bj = base[:, 1] + j
                    bk = base[:, 2] + k

                    # Valid index mask
                    valid = (bi >= 0) & (bi < gx) & (bj >= 0) & (bj < gy) & (bk >= 0) & (bk < gz)
                    if not np.any(valid):
                        continue

                    np.add.at(grid_m, (bi[valid], bj[valid], bk[valid]), contrib_m[valid])
                    np.add.at(grid_v, (bi[valid], bj[valid], bk[valid]), contrib_p[valid])

        # Grid normalization & update
        active = grid_m > 1e-12
        grid_v[active] /= grid_m[active, None]

        # Grid gravity & damping
        grid_v[active] += dt * self.gravity
        grid_v[active] *= (1.0 - self.damping)

        # Grid Boundary Conditions (vectorized for speed)
        # Basin geometry:
        # Ground: z = 0 (top at z=0.0)
        # Basin back: y = 0.72 - 0.08 = 0.64
        # Basin front: y = -0.64
        # Basin left: x = -1.45 + 0.08 = -1.37
        # Basin right: x = 1.45 - 0.08 = 1.37
        # Basin top: z = 0.64

        ix_coords = self.grid_origin[0] + np.arange(gx) * self.dx
        iy_coords = self.grid_origin[1] + np.arange(gy) * self.dx
        iz_coords = self.grid_origin[2] + np.arange(gz) * self.dx

        # Floor contact at z < 0.02
        floor_mask = (iz_coords < 0.02)
        if np.any(floor_mask):
            gz_v = grid_v[:, :, floor_mask, 2]
            grid_v[:, :, floor_mask, 2] = np.maximum(gz_v, 0.0)
            grid_v[:, :, floor_mask, 0] *= 0.5
            grid_v[:, :, floor_mask, 1] *= 0.5

        # Basin side walls below z = 0.64
        z_basin = (iz_coords <= 0.64)
        if np.any(z_basin):
            # Left wall x < -1.37
            left_mask = ix_coords < -1.37
            if np.any(left_mask):
                sub_v = grid_v[np.ix_(left_mask, np.ones(gy, bool), z_basin)]
                sub_v[:, :, :, 0] = np.maximum(sub_v[:, :, :, 0], 0.0)
                sub_v[:, :, :, 1] *= 0.8
                sub_v[:, :, :, 2] *= 0.8
                grid_v[np.ix_(left_mask, np.ones(gy, bool), z_basin)] = sub_v

            # Right wall x > 1.37
            right_mask = ix_coords > 1.37
            if np.any(right_mask):
                sub_v = grid_v[np.ix_(right_mask, np.ones(gy, bool), z_basin)]
                sub_v[:, :, :, 0] = np.minimum(sub_v[:, :, :, 0], 0.0)
                sub_v[:, :, :, 1] *= 0.8
                sub_v[:, :, :, 2] *= 0.8
                grid_v[np.ix_(right_mask, np.ones(gy, bool), z_basin)] = sub_v

            # Back wall y > 0.64
            back_mask = iy_coords > 0.64
            if np.any(back_mask):
                sub_v = grid_v[np.ix_(np.ones(gx, bool), back_mask, z_basin)]
                sub_v[:, :, :, 1] = np.minimum(sub_v[:, :, :, 1], 0.0)
                sub_v[:, :, :, 0] *= 0.8
                sub_v[:, :, :, 2] *= 0.8
                grid_v[np.ix_(np.ones(gx, bool), back_mask, z_basin)] = sub_v

            # Front wall y < -0.64
            front_mask = iy_coords < -0.64
            if np.any(front_mask):
                sub_v = grid_v[np.ix_(np.ones(gx, bool), front_mask, z_basin)]
                sub_v[:, :, :, 1] = np.maximum(sub_v[:, :, :, 1], 0.0)
                sub_v[:, :, :, 0] *= 0.8
                sub_v[:, :, :, 2] *= 0.8
                grid_v[np.ix_(np.ones(gx, bool), front_mask, z_basin)] = sub_v

        # G2P: Grid to Particles
        new_v = np.zeros_like(self.v)
        new_C = np.zeros_like(self.C)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[i][:, 0] * w[j][:, 1] * w[k][:, 2]  # (N,)
                    bi = base[:, 0] + i
                    bj = base[:, 1] + j
                    bk = base[:, 2] + k

                    valid = (bi >= 0) & (bi < gx) & (bj >= 0) & (bj < gy) & (bk >= 0) & (bk < gz)

                    gv = np.zeros((self.n_particles, 3), dtype=np.float64)
                    gv[valid] = grid_v[bi[valid], bj[valid], bk[valid]]

                    new_v += weight[:, None] * gv

                    dpos = (np.column_stack([bi, bj, bk]) - pos_grid) * self.dx
                    # outer product dpos * gv * weight * 4 * inv_dx^2
                    # new_C += affine_mul * weight * gv (outer) dpos
                    term = affine_mul * weight[:, None, None] * (gv[:, :, None] @ dpos[:, None, :])
                    new_C += term

        self.v = new_v
        self.C = new_C

        # Advect particles
        self.x += self.v * dt

        # Update deformation gradient F_e = (I + dt * C) @ F_e
        I = np.eye(3, dtype=np.float64)
        grad_v_dt = I[None, :, :] + dt * self.C
        self.F_e = grad_v_dt @ self.F_e

        # Particle-level safety boundary clamp
        self.x[:, 2] = np.maximum(self.x[:, 2], 0.01)
        self.x[:, 0] = np.clip(self.x[:, 0], -1.44, 1.44)
        self.x[:, 1] = np.clip(self.x[:, 1], -0.71, 0.71)

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation to target time
        target_time = float(time)
        dt = self.sub_dt

        while self.current_time < target_time - 1e-7:
            step = min(dt, target_time - self.current_time)
            self._substep(step)
            self.current_time += step

        self.current_time = target_time

        # Compute required outputs
        # 1. Signed geometric distance to boundary contacts
        # Contact distance to ground (z=0) and basin walls
        d_ground = self.x[:, 2] - 0.02
        d_left = self.x[:, 0] - (-1.37)
        d_right = 1.37 - self.x[:, 0]
        d_back = 0.64 - self.x[:, 1]
        d_front = self.x[:, 1] - (-0.64)
        contact_sd = np.minimum.reduce([d_ground, d_left, d_right, d_back, d_front])

        # Filter active and near-contact samples (or export all particles)
        contact_signed_dist = contact_sd.astype(np.float64)

        # 2. External force: gravity force m * g
        ext_force = self.m[:, None] * self.gravity[None, :]

        # 3. Mass quantity
        mass_quantity = self.m.copy()

        # 4. Cumulative mass source
        cumulative_mass_source = np.zeros(1, dtype=np.float64)

        frame = StateFrame(
            time=self.current_time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.labels.copy(),
            attributes={
                "deformation_gradient": self.F_e.copy(),
                "mass_quantity": mass_quantity,
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.labels.copy()
            },
            fields={
                "contact_signed_distance": contact_signed_dist,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": ext_force
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale)
            }
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "particle_count": self.n_particles
        }

