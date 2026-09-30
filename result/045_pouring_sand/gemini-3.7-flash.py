"""Granular physics solver for pouring sand benchmark.

Simulates granular flow, hopper discharge, free-fall stream breakup, floor impact,
pile formation with angle of repose using spatial grid neighbor search, contact mechanics,
and boundary interactions.
"""
from __future__ import annotations

import math
import numpy as np
import numba
from worldsolver.contracts import StateFrame


@numba.njit(fastmath=True)
def _solve_substeps(
    pos: np.ndarray,
    vel: np.ndarray,
    nominal_r: float,
    g: np.ndarray,
    n_substeps: int,
    dt_sub: float,
    floor_z: float,
    hopper_r_top: float,
    hopper_z_top: float,
    hopper_r_mid: float,
    hopper_z_mid: float,
    hopper_r_bot: float,
    hopper_z_bot: float,
    hopper_z_neck: float,
    funnel_slope: float,
    grid_size: float,
    inv_grid_size: float,
    min_bounds: np.ndarray,
    grid_dims: np.ndarray,
    head: np.ndarray,
    next_p: np.ndarray,
    cell_keys: np.ndarray,
):
    """Perform N sub-steps of granular physics simulation."""
    n_particles = len(pos)
    nx = grid_dims[0]
    ny = grid_dims[1]
    nz = grid_dims[2]
    n_cells = nx * ny * nz

    eff_floor_z = floor_z + nominal_r
    diam = 2.0 * nominal_r
    diam_sq = diam * diam

    # Granular material parameters (dry sand)
    kn = 1.2e5      # Normal stiffness (N/m equivalent per unit mass scaling)
    gn = 25.0       # Normal damping
    mu_grain = 0.55 # Grain-grain friction coefficient (gives repose angle ~30-35 deg)
    mu_wall = 0.45  # Wall friction coefficient
    floor_restitution = 0.15

    for step in range(n_substeps):
        # 1. Clear spatial hash grid
        for c in range(n_cells):
            head[c] = -1

        # 2. Insert particles into spatial grid
        for i in range(n_particles):
            gx = int((pos[i, 0] - min_bounds[0]) * inv_grid_size)
            gy = int((pos[i, 1] - min_bounds[1]) * inv_grid_size)
            gz = int((pos[i, 2] - min_bounds[2]) * inv_grid_size)

            if gx < 0:
                gx = 0
            elif gx >= nx:
                gx = nx - 1
            if gy < 0:
                gy = 0
            elif gy >= ny:
                gy = ny - 1
            if gz < 0:
                gz = 0
            elif gz >= nz:
                gz = nz - 1

            c_idx = gx + nx * (gy + ny * gz)
            cell_keys[i] = c_idx
            next_p[i] = head[c_idx]
            head[c_idx] = i

        # 3. Particle-particle interaction forces & updates
        # Compute acceleration on each particle
        acc = np.zeros((n_particles, 3), dtype=np.float64)
        for i in range(n_particles):
            acc[i, 0] = g[0]
            acc[i, 1] = g[1]
            acc[i, 2] = g[2]

        for i in range(n_particles):
            xi = pos[i, 0]
            yi = pos[i, 1]
            zi = pos[i, 2]
            vxi = vel[i, 0]
            vyi = vel[i, 1]
            vzi = vel[i, 2]

            gx = int((xi - min_bounds[0]) * inv_grid_size)
            gy = int((yi - min_bounds[1]) * inv_grid_size)
            gz = int((zi - min_bounds[2]) * inv_grid_size)

            min_gx = max(0, gx - 1)
            max_gx = min(nx - 1, gx + 1)
            min_gy = max(0, gy - 1)
            max_gy = min(ny - 1, gy + 1)
            min_gz = max(0, gz - 1)
            max_gz = min(nz - 1, gz + 1)

            for cz in range(min_gz, max_gz + 1):
                for cy in range(min_gy, max_gy + 1):
                    for cx in range(min_gx, max_gx + 1):
                        c = cx + nx * (cy + ny * cz)
                        j = head[c]
                        while j != -1:
                            if i < j:
                                dx = pos[j, 0] - xi
                                dy = pos[j, 1] - yi
                                dz = pos[j, 2] - zi
                                dist_sq = dx * dx + dy * dy + dz * dz

                                if 0.0 < dist_sq < diam_sq:
                                    dist = math.sqrt(dist_sq)
                                    overlap = diam - dist
                                    nx_unit = dx / dist
                                    ny_unit = dy / dist
                                    nz_unit = dz / dist

                                    # Relative velocity (j relative to i)
                                    dvx = vel[j, 0] - vxi
                                    dvy = vel[j, 1] - vyi
                                    dvz = vel[j, 2] - vzi
                                    v_rel_n = dvx * nx_unit + dvy * ny_unit + dvz * nz_unit

                                    # Spring-dashpot normal force magnitude (push apart)
                                    f_norm = kn * overlap - gn * v_rel_n
                                    if f_norm < 0.0:
                                        f_norm = 0.0

                                    # Tangential relative velocity
                                    v_rel_tx = dvx - v_rel_n * nx_unit
                                    v_rel_ty = dvy - v_rel_n * ny_unit
                                    v_rel_tz = dvz - v_rel_n * nz_unit
                                    v_tang_mag = math.sqrt(
                                        v_rel_tx * v_rel_tx + v_rel_ty * v_rel_ty + v_rel_tz * v_rel_tz
                                    )

                                    f_tang_x = 0.0
                                    f_tang_y = 0.0
                                    f_tang_z = 0.0
                                    if v_tang_mag > 1e-7:
                                        # Coulomb friction force
                                        f_tang_mag = min(mu_grain * f_norm, 15.0 * v_tang_mag)
                                        f_tang_x = f_tang_mag * (v_rel_tx / v_tang_mag)
                                        f_tang_y = f_tang_mag * (v_rel_ty / v_tang_mag)
                                        f_tang_z = f_tang_mag * (v_rel_tz / v_tang_mag)

                                    total_fx = -f_norm * nx_unit + f_tang_x
                                    total_fy = -f_norm * ny_unit + f_tang_y
                                    total_fz = -f_norm * nz_unit + f_tang_z

                                    acc[i, 0] += total_fx
                                    acc[i, 1] += total_fy
                                    acc[i, 2] += total_fz
                                    acc[j, 0] -= total_fx
                                    acc[j, 1] -= total_fy
                                    acc[j, 2] -= total_fz

                            j = next_p[j]

        # 4. Integrate velocity and position (Symplectic Euler)
        for i in range(n_particles):
            vel[i, 0] += acc[i, 0] * dt_sub
            vel[i, 1] += acc[i, 1] * dt_sub
            vel[i, 2] += acc[i, 2] * dt_sub

            pos[i, 0] += vel[i, 0] * dt_sub
            pos[i, 1] += vel[i, 1] * dt_sub
            pos[i, 2] += vel[i, 2] * dt_sub

        # 5. Boundary collisions and constraints
        for i in range(n_particles):
            px = pos[i, 0]
            py = pos[i, 1]
            pz = pos[i, 2]

            # Floor collision (z = 0 plane)
            if pz < eff_floor_z:
                pos[i, 2] = eff_floor_z
                if vel[i, 2] < 0.0:
                    vel[i, 2] = -vel[i, 2] * floor_restitution
                # Tangential friction with floor
                vel[i, 0] *= max(0.0, 1.0 - mu_wall * 50.0 * dt_sub)
                vel[i, 1] *= max(0.0, 1.0 - mu_wall * 50.0 * dt_sub)
                pz = eff_floor_z

            # Hopper boundary collision
            # Inner profile:
            # z in [1.42, 1.52]: radius = 0.075
            # z in (1.52, 1.72]: conical funnel between radius 0.075 and 0.34
            # z in (1.72, 2.28]: cylinder with radius 0.34
            if pz >= hopper_z_bot:
                r_xy = math.sqrt(px * px + py * py)
                # Compute wall radius at height pz
                if pz >= hopper_z_mid:
                    r_wall = hopper_r_top
                elif pz >= hopper_z_neck:
                    r_wall = hopper_r_bot + (pz - hopper_z_neck) * funnel_slope
                else:
                    r_wall = hopper_r_bot

                eff_r_wall = r_wall - nominal_r
                if eff_r_wall > 0.0 and r_xy > eff_r_wall:
                    # Project inside hopper wall
                    scale = eff_r_wall / r_xy
                    pos[i, 0] = px * scale
                    pos[i, 1] = py * scale

                    # Normal pointing inwards: (-px/r_xy, -py/r_xy, 0) approximately
                    # Remove outward radial velocity component
                    urx = px / r_xy
                    ury = py / r_xy
                    vr = vel[i, 0] * urx + vel[i, 1] * ury
                    if vr > 0.0:
                        vel[i, 0] -= vr * urx
                        vel[i, 1] -= vr * ury

                    # Friction along the hopper wall
                    vel[i, 0] *= max(0.0, 1.0 - mu_wall * 30.0 * dt_sub)
                    vel[i, 1] *= max(0.0, 1.0 - mu_wall * 30.0 * dt_sub)
                    vel[i, 2] *= max(0.0, 1.0 - mu_wall * 10.0 * dt_sub)


class BenchmarkSolver:
    """Pouring sand physics simulation solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.gravity = np.array(scene.config["gravity"], dtype=np.float64)
        self.initial_frame = scene.initial_state()
        self.nominal_radius = float(scene.config["initial_conditions"]["nominal_grain_radius"])
        self.bulk_density = float(scene.config["initial_conditions"]["bulk_density"])

        self.current_time = 0.0
        self.position = np.array(self.initial_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(self.initial_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(self.initial_frame.mass, dtype=np.float64, copy=True)
        self.n_particles = len(self.position)

        # Hopper geometry profiles from scene_geometry.yaml:
        # inner_profile:
        #   [0.075, 1.42], [0.075, 1.52], [0.34, 1.72], [0.34, 2.28]
        self.hopper_r_bot = 0.075
        self.hopper_z_bot = 1.42
        self.hopper_z_neck = 1.52
        self.hopper_r_top = 0.34
        self.hopper_z_mid = 1.72
        self.hopper_z_top = 2.28
        self.funnel_slope = (self.hopper_r_top - self.hopper_r_bot) / (self.hopper_z_mid - self.hopper_z_neck)

        # Spatial hash grid setup
        self.grid_size = 2.0 * self.nominal_radius * 1.05
        self.inv_grid_size = 1.0 / self.grid_size
        self.min_bounds = np.array([-1.70, -1.40, -0.10], dtype=np.float64)
        max_bounds = np.array([1.70, 1.40, 2.40], dtype=np.float64)
        self.grid_dims = np.ceil((max_bounds - self.min_bounds) / self.grid_size).astype(np.int64)
        total_cells = int(np.prod(self.grid_dims))

        self.head = np.full(total_cells, -1, dtype=np.int32)
        self.next_p = np.full(self.n_particles, -1, dtype=np.int32)
        self.cell_keys = np.full(self.n_particles, -1, dtype=np.int32)

        # Substep time-stepping configuration
        self.sub_dt = 0.0005  # 0.5 ms for stable granular contact integration

        # Warm-up JIT compile
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        _solve_substeps(
            self.position,
            self.velocity,
            self.nominal_radius,
            self.gravity,
            1,
            self.sub_dt,
            0.0,
            self.hopper_r_top,
            self.hopper_z_top,
            self.hopper_r_mid,
            self.hopper_z_mid,
            self.hopper_r_bot,
            self.hopper_z_bot,
            self.hopper_z_neck,
            self.funnel_slope,
            self.grid_size,
            self.inv_grid_size,
            self.min_bounds,
            self.grid_dims,
            self.head,
            self.next_p,
            self.cell_keys,
        )

    def _compute_contact_signed_distances(self) -> np.ndarray:
        """Compute signed distance to floor boundary for active contact evaluation."""
        # Contact signed distance to floor (z = 0)
        gap = self.position[:, 2] - self.nominal_radius
        # Return near-contact / contact samples (e.g. z < 0.20)
        mask = self.position[:, 2] < 0.30
        if np.any(mask):
            return gap[mask]
        else:
            return np.array([float(np.min(gap))], dtype=np.float64)

    def advance_to(self, time: float) -> StateFrame:
        dt_target = time - self.current_time
        if dt_target > 1e-8:
            n_substeps = int(round(dt_target / self.sub_dt))
            if n_substeps < 1:
                n_substeps = 1
            actual_sub_dt = dt_target / n_substeps

            _solve_substeps(
                self.position,
                self.velocity,
                self.nominal_radius,
                self.gravity,
                n_substeps,
                actual_sub_dt,
                0.0,
                self.hopper_r_top,
                self.hopper_z_top,
                self.hopper_r_mid,
                self.hopper_z_mid,
                self.hopper_r_bot,
                self.hopper_z_bot,
                self.hopper_z_neck,
                self.funnel_slope,
                self.grid_size,
                self.inv_grid_size,
                self.min_bounds,
                self.grid_dims,
                self.head,
                self.next_p,
                self.cell_keys,
            )
            self.current_time = time

        contact_sd = self._compute_contact_signed_distances()
        external_f = np.tile(self.gravity, (self.n_particles, 1)) * self.mass[:, np.newaxis]

        frame = StateFrame(
            time=float(time),
            position=np.array(self.position, dtype=np.float64, copy=True),
            velocity=np.array(self.velocity, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            attributes={
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
                "nominal_radius": np.full(self.n_particles, self.nominal_radius, dtype=np.float64),
                "material_phase": np.zeros(self.n_particles, dtype=np.float64),
            },
            fields={
                "contact_signed_distance": np.array(contact_sd, dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.array(external_f, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.nominal_radius * 2.0),
                "initial_total_mass": float(np.sum(self.mass)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": self.n_particles,
            "simulated_time": self.current_time,
        }

