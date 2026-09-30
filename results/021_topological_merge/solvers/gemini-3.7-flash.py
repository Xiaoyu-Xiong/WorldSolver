"""Deterministic MPM/viscoplastic solver for repeated topological merge."""
from __future__ import annotations

import numpy as np
import numba
from worldsolver.contracts import StateFrame


@numba.njit(fastmath=True)
def _mpm_step_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    vol: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    grid_v: np.ndarray,
    grid_m: np.ndarray,
    dt: float,
    dx: float,
    inv_dx: float,
    origin: np.ndarray,
    grid_dim: np.ndarray,
    gravity: np.ndarray,
    mu_0: float,
    lambda_0: float,
    sig_min: float,
    sig_max: float,
):
    n_particles = pos.shape[0]
    nx, ny, nz = grid_dim[0], grid_dim[1], grid_dim[2]

    # Reset grid
    grid_v.fill(0.0)
    grid_m.fill(0.0)

    # 1. P2G: Particle to Grid
    for p in range(n_particles):
        xp0 = (pos[p, 0] - origin[0]) * inv_dx
        xp1 = (pos[p, 1] - origin[1]) * inv_dx
        xp2 = (pos[p, 2] - origin[2]) * inv_dx

        base0 = int(np.floor(xp0 - 0.5))
        base1 = int(np.floor(xp1 - 0.5))
        base2 = int(np.floor(xp2 - 0.5))

        fx0 = xp0 - float(base0)
        fx1 = xp1 - float(base1)
        fx2 = xp2 - float(base2)

        # Quadratic B-splines
        w0_0 = 0.5 * (1.5 - fx0) * (1.5 - fx0)
        w0_1 = 0.75 - (fx0 - 1.0) * (fx0 - 1.0)
        w0_2 = 0.5 * (fx0 - 0.5) * (fx0 - 0.5)

        w1_0 = 0.5 * (1.5 - fx1) * (1.5 - fx1)
        w1_1 = 0.75 - (fx1 - 1.0) * (fx1 - 1.0)
        w1_2 = 0.5 * (fx1 - 0.5) * (fx1 - 0.5)

        w2_0 = 0.5 * (1.5 - fx2) * (1.5 - fx2)
        w2_1 = 0.75 - (fx2 - 1.0) * (fx2 - 1.0)
        w2_2 = 0.5 * (fx2 - 0.5) * (fx2 - 0.5)

        # Deformation gradient & SVD for plasticity
        Fp = F[p]
        U, sig, Vt = np.linalg.svd(Fp)

        # Clamp singular values (plasticity / yield limit)
        for d in range(3):
            if sig[d] < sig_min:
                sig[d] = sig_min
            elif sig[d] > sig_max:
                sig[d] = sig_max

        # Plastic deformation gradient
        # Reconstruct F_plastic = U @ diag(sig) @ Vt
        F_plastic = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += U[i, k] * sig[k] * Vt[k, j]
                F_plastic[i, j] = s

        F[p] = F_plastic

        # Rotation R = U @ Vt
        R = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += U[i, k] * Vt[k, j]
                R[i, j] = s

        J = sig[0] * sig[1] * sig[2]
        if J < 0.05:
            J = 0.05

        # Kirchhoff stress: tau = 2 * mu * (F - R) @ F^T + lambda * (J - 1) * J * I
        # stress = 2 * mu_0 * (F_plastic - R) @ F_plastic.T + lambda_0 * (J - 1.0) * J * eye(3)
        stress = np.zeros((3, 3), dtype=np.float64)
        diff = F_plastic - R
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += diff[i, k] * F_plastic[j, k]
                stress[i, j] = 2.0 * mu_0 * s

        diag_term = lambda_0 * (J - 1.0) * J
        stress[0, 0] += diag_term
        stress[1, 1] += diag_term
        stress[2, 2] += diag_term

        coeff = -dt * vol[p] * 4.0 * inv_dx * inv_dx
        stress_contrib = coeff * stress

        mp = mass[p]
        affine = mp * C[p] + stress_contrib

        vp0 = vel[p, 0]
        vp1 = vel[p, 1]
        vp2 = vel[p, 2]

        # Spline weight arrays for indexing
        w_x = (w0_0, w0_1, w0_2)
        w_y = (w1_0, w1_1, w1_2)
        w_z = (w2_0, w2_1, w2_2)

        for i in range(3):
            ix = base0 + i
            if ix < 0 or ix >= nx:
                continue
            wi = w_x[i]
            dpos0 = (float(i) - fx0) * dx
            for j in range(3):
                iy = base1 + j
                if iy < 0 or iy >= ny:
                    continue
                wij = wi * w_y[j]
                dpos1 = (float(j) - fx1) * dx
                for k in range(3):
                    iz = base2 + k
                    if iz < 0 or iz >= nz:
                        continue
                    weight = wij * w_z[k]
                    dpos2 = (float(k) - fx2) * dx

                    # momentum contribution = weight * (mp * vp + affine @ dpos)
                    px = mp * vp0 + affine[0, 0] * dpos0 + affine[0, 1] * dpos1 + affine[0, 2] * dpos2
                    py = mp * vp1 + affine[1, 0] * dpos0 + affine[1, 1] * dpos1 + affine[1, 2] * dpos2
                    pz = mp * vp2 + affine[2, 0] * dpos0 + affine[2, 1] * dpos1 + affine[2, 2] * dpos2

                    grid_v[ix, iy, iz, 0] += weight * px
                    grid_v[ix, iy, iz, 1] += weight * py
                    grid_v[ix, iy, iz, 2] += weight * pz
                    grid_m[ix, iy, iz] += weight * mp

    # 2. Grid operations
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                m = grid_m[ix, iy, iz]
                if m > 1e-12:
                    inv_m = 1.0 / m
                    v0 = grid_v[ix, iy, iz, 0] * inv_m + gravity[0] * dt
                    v1 = grid_v[ix, iy, iz, 1] * inv_m + gravity[1] * dt
                    v2 = grid_v[ix, iy, iz, 2] * inv_m + gravity[2] * dt

                    world_z = origin[2] + float(iz) * dx
                    # Ground collision at z <= 0.0
                    if world_z <= 0.0:
                        if v2 < 0.0:
                            v2 = 0.0
                        # Friction on ground
                        v0 *= 0.3
                        v1 *= 0.3

                    # Boundary clamping
                    if (ix < 3 and v0 < 0.0) or (ix >= nx - 3 and v0 > 0.0):
                        v0 = 0.0
                    if (iy < 3 and v1 < 0.0) or (iy >= ny - 3 and v1 > 0.0):
                        v1 = 0.0
                    if iz < 2 and v2 < 0.0:
                        v2 = 0.0

                    grid_v[ix, iy, iz, 0] = v0
                    grid_v[ix, iy, iz, 1] = v1
                    grid_v[ix, iy, iz, 2] = v2

    # 3. G2P: Grid to Particle
    c_factor = 4.0 * inv_dx
    for p in range(n_particles):
        xp0 = (pos[p, 0] - origin[0]) * inv_dx
        xp1 = (pos[p, 1] - origin[1]) * inv_dx
        xp2 = (pos[p, 2] - origin[2]) * inv_dx

        base0 = int(np.floor(xp0 - 0.5))
        base1 = int(np.floor(xp1 - 0.5))
        base2 = int(np.floor(xp2 - 0.5))

        fx0 = xp0 - float(base0)
        fx1 = xp1 - float(base1)
        fx2 = xp2 - float(base2)

        w0_0 = 0.5 * (1.5 - fx0) * (1.5 - fx0)
        w0_1 = 0.75 - (fx0 - 1.0) * (fx0 - 1.0)
        w0_2 = 0.5 * (fx0 - 0.5) * (fx0 - 0.5)

        w1_0 = 0.5 * (1.5 - fx1) * (1.5 - fx1)
        w1_1 = 0.75 - (fx1 - 1.0) * (fx1 - 1.0)
        w1_2 = 0.5 * (fx1 - 0.5) * (fx1 - 0.5)

        w2_0 = 0.5 * (1.5 - fx2) * (1.5 - fx2)
        w2_1 = 0.75 - (fx2 - 1.0) * (fx2 - 1.0)
        w2_2 = 0.5 * (fx2 - 0.5) * (fx2 - 0.5)

        w_x = (w0_0, w0_1, w0_2)
        w_y = (w1_0, w1_1, w1_2)
        w_z = (w2_0, w2_1, w2_2)

        new_v0 = 0.0
        new_v1 = 0.0
        new_v2 = 0.0

        new_C = np.zeros((3, 3), dtype=np.float64)

        for i in range(3):
            ix = base0 + i
            if ix < 0 or ix >= nx:
                continue
            wi = w_x[i]
            dpos0 = (float(i) - fx0) * dx
            for j in range(3):
                iy = base1 + j
                if iy < 0 or iy >= ny:
                    continue
                wij = wi * w_y[j]
                dpos1 = (float(j) - fx1) * dx
                for k in range(3):
                    iz = base2 + k
                    if iz < 0 or iz >= nz:
                        continue
                    weight = wij * w_z[k]
                    dpos2 = (float(k) - fx2) * dx

                    gv0 = grid_v[ix, iy, iz, 0]
                    gv1 = grid_v[ix, iy, iz, 1]
                    gv2 = grid_v[ix, iy, iz, 2]

                    new_v0 += weight * gv0
                    new_v1 += weight * gv1
                    new_v2 += weight * gv2

                    fac = c_factor * weight
                    new_C[0, 0] += fac * gv0 * dpos0
                    new_C[0, 1] += fac * gv0 * dpos1
                    new_C[0, 2] += fac * gv0 * dpos2
                    new_C[1, 0] += fac * gv1 * dpos0
                    new_C[1, 1] += fac * gv1 * dpos1
                    new_C[1, 2] += fac * gv1 * dpos2
                    new_C[2, 0] += fac * gv2 * dpos0
                    new_C[2, 1] += fac * gv2 * dpos1
                    new_C[2, 2] += fac * gv2 * dpos2

        vel[p, 0] = new_v0
        vel[p, 1] = new_v1
        vel[p, 2] = new_v2

        pos[p, 0] += dt * new_v0
        pos[p, 1] += dt * new_v1
        pos[p, 2] += dt * new_v2

        C[p] = new_C

        # F_new = (I + dt * C) @ F
        old_F = F[p]
        F_new = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    A_ik = (1.0 if i == k else 0.0) + dt * new_C[i, k]
                    s += A_ik * old_F[k, j]
                F_new[i, j] = s
        F[p] = F_new

        # Particle ground clamping
        if pos[p, 2] < 0.04:
            pos[p, 2] = 0.04
            if vel[p, 2] < 0.0:
                vel[p, 2] = 0.0


class BenchmarkSolver:
    def __init__(self):
        self.time = 0.0
        self.scene = None
        self.seed = 2106

        # Particle state
        self.pos = None
        self.vel = None
        self.mass = None
        self.vol = None
        self.C = None
        self.F = None
        self.body_id = None
        self.initial_group_id = None
        self.color_id = None
        self.particle_radius = None

        # Simulation parameters
        self.n_particles = 0
        self.dx = 0.125
        self.inv_dx = 1.0 / self.dx
        self.dt = 1e-3
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Grid domain bounds: [-3.5, -3.2, -0.2] to [3.5, 3.2, 9.3]
        self.origin = np.array([-3.5, -3.2, -0.2], dtype=np.float64)
        self.grid_dim = np.array([56, 52, 76], dtype=np.int32)
        self.grid_v = None
        self.grid_m = None

        # Viscoplastic material parameters
        self.E = 3.5e4        # Young's modulus
        self.nu = 0.35        # Poisson's ratio
        self.mu_0 = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.sig_min = 0.88   # Plastic compression limit
        self.sig_max = 1.12   # Plastic stretch limit

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        init_frame = scene.initial_state()

        self.time = float(init_frame.time)
        self.pos = np.copy(init_frame.position).astype(np.float64)
        self.vel = np.copy(init_frame.velocity).astype(np.float64)
        self.mass = np.copy(init_frame.mass).astype(np.float64)
        self.body_id = np.copy(init_frame.body_id)

        self.n_particles = len(self.pos)
        self.C = np.zeros((self.n_particles, 3, 3), dtype=np.float64)
        self.F = np.zeros((self.n_particles, 3, 3), dtype=np.float64)
        for i in range(self.n_particles):
            self.F[i] = np.eye(3, dtype=np.float64)

        rest_density = float(scene.config.get("initial_conditions", {}).get("rest_density", 1050.0))
        self.vol = self.mass / rest_density

        # Extract attributes
        attrs = init_frame.attributes
        if "initial_group_id" in attrs:
            self.initial_group_id = np.copy(attrs["initial_group_id"])
        else:
            self.initial_group_id = self.body_id.astype(np.float64)

        if "color_id" in attrs:
            self.color_id = np.copy(attrs["color_id"])
        else:
            self.color_id = np.zeros(self.n_particles, dtype=np.float64)

        if "particle_radius" in attrs:
            self.particle_radius = np.copy(attrs["particle_radius"])
        else:
            self.particle_radius = np.full(self.n_particles, 0.112, dtype=np.float64)

        # Check gravity from scene
        if "gravity" in scene.config:
            self.gravity = np.array(scene.config["gravity"], dtype=np.float64)

        # Allocate grid buffers
        nx, ny, nz = self.grid_dim
        self.grid_v = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        self.grid_m = np.zeros((nx, ny, nz), dtype=np.float64)

        # Warm up JIT kernel
        _mpm_step_kernel(
            self.pos[:1],
            self.vel[:1],
            self.mass[:1],
            self.vol[:1],
            self.C[:1],
            self.F[:1],
            self.grid_v,
            self.grid_m,
            self.dt,
            self.dx,
            self.inv_dx,
            self.origin,
            self.grid_dim,
            self.gravity,
            self.mu_0,
            self.lambda_0,
            self.sig_min,
            self.sig_max,
        )
        self.F[0] = np.eye(3, dtype=np.float64)
        self.pos[0] = init_frame.position[0]
        self.vel[0] = init_frame.velocity[0]
        self.C[0] = 0.0

    def _step(self, dt: float):
        _mpm_step_kernel(
            self.pos,
            self.vel,
            self.mass,
            self.vol,
            self.C,
            self.F,
            self.grid_v,
            self.grid_m,
            dt,
            self.dx,
            self.inv_dx,
            self.origin,
            self.grid_dim,
            self.gravity,
            self.mu_0,
            self.lambda_0,
            self.sig_min,
            self.sig_max,
        )

    def advance_to(self, time: float) -> StateFrame:
        while self.time + self.dt * 0.5 < time:
            sub_dt = min(self.dt, time - self.time)
            self._step(sub_dt)
            self.time += sub_dt

        self.time = time

        ext_force = self.mass[:, None] * self.gravity[None, :]
        k_groups = int(self.scene.config.get("initial_conditions", {}).get("initial_group_count", 24))
        cum_mass_source = np.zeros(k_groups, dtype=np.float64)
        interaction_force = np.zeros((k_groups, 3), dtype=np.float64)

        frame = StateFrame(
            time=float(self.time),
            position=np.copy(self.pos),
            velocity=np.copy(self.vel),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "initial_group_id": np.copy(self.initial_group_id),
                "color_id": np.copy(self.color_id),
                "particle_radius": np.copy(self.particle_radius),
                "deformation_gradient": np.copy(self.F),
                "mass_quantity": np.copy(self.mass),
            },
            fields={
                "cumulative_mass_source": cum_mass_source,
                "external_force": ext_force,
                "interaction_force": interaction_force,
            },
            scalars={
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
                "particle_spacing": 0.205,
                "rest_density": float(self.scene.config.get("initial_conditions", {}).get("rest_density", 1050.0)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": self.n_particles,
            "sim_time": self.time,
        }


