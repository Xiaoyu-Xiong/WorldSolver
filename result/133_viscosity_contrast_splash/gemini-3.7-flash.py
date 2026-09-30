"""Viscosity Contrast Splash physical fluid solver.

Simulates two liquid pools (low-viscosity and high-viscosity) with falling drops
impacting each pool using smoothed particle hydrodynamics (WCSPH / XSPH) with
boundary handling and exact trajectory output contract.
"""
from __future__ import annotations
import numpy as np
import scipy.spatial
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config = {}
        self.time = 0.0
        self.position = np.empty((0, 3), dtype=np.float64)
        self.velocity = np.empty((0, 3), dtype=np.float64)
        self.mass = np.empty((0,), dtype=np.float64)
        self.body_id = np.empty((0,), dtype=np.int32)
        self.material_label = np.empty((0,), dtype=np.int32)
        self.particle_radius = np.empty((0,), dtype=np.float64)
        self.deformation_gradient = np.empty((0, 3, 3), dtype=np.float64)
        self.cumulative_mass_source = np.empty((0,), dtype=np.float64)

        # Physics constants
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.rho0 = 1000.0
        self.gamma = 7.0
        self.c0 = 25.0
        self.B = (self.rho0 * self.c0**2) / self.gamma
        self.h = 0.22  # SPH smoothing length
        self.h2 = self.h * self.h

        # Viscosity parameters
        # Material 0: low-visc pool, 1: high-visc pool, 2: low-visc drop, 3: high-visc drop
        self.mu_low = 0.05      # kinematic viscosity / XSPH factor
        self.mu_high = 3.5      # high viscosity
        self.xsph_low = 0.02
        self.xsph_high = 0.45

        # Substepping dt
        self.sim_dt = 0.001

        # Diagnostic grid spec
        self.grid_shape = (24, 16, 28)
        self.bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        self.grid_spacing = 0.14
        self.characteristic_length = 4.3
        self.reference_velocity = 1.0

        # Boundary samples
        self.boundary_pos = np.empty((0, 3), dtype=np.float64)
        self.boundary_normal = np.empty((0, 3), dtype=np.float64)
        self.boundary_vel = np.empty((0, 3), dtype=np.float64)

    def _init_boundaries(self) -> None:
        """Create boundary sample points and normals for rigid basin and center divider."""
        samples = []
        normals = []
        sp = 0.10

        # Ground plane at z = 0, x in [-3.2, 3.2], y in [-1.4, 1.4]
        gx = np.arange(-3.2 + sp * 0.5, 3.2, sp)
        gy = np.arange(-1.4 + sp * 0.5, 1.4, sp)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        g_pts = np.column_stack([gxx.ravel(), gyy.ravel(), np.zeros(gxx.size)])
        samples.append(g_pts)
        normals.append(np.tile(np.array([0.0, 0.0, 1.0]), (len(g_pts), 1)))

        # Center divider: half_extents [0.08, 0.9, 0.65], center [0, 0, 0.65]
        # x in [-0.08, 0.08], y in [-0.9, 0.9], z in [0, 1.3]
        # Left face (x = -0.08, normal = [-1, 0, 0])
        dy = np.arange(-0.9 + sp * 0.5, 0.9, sp)
        dz = np.arange(0.0 + sp * 0.5, 1.3, sp)
        dyy, dzz = np.meshgrid(dy, dz, indexing="ij")
        f_left = np.column_stack([np.full(dyy.size, -0.08), dyy.ravel(), dzz.ravel()])
        samples.append(f_left)
        normals.append(np.tile(np.array([-1.0, 0.0, 0.0]), (len(f_left), 1)))

        # Right face (x = +0.08, normal = [+1, 0, 0])
        f_right = np.column_stack([np.full(dyy.size, 0.08), dyy.ravel(), dzz.ravel()])
        samples.append(f_right)
        normals.append(np.tile(np.array([1.0, 0.0, 0.0]), (len(f_right), 1)))

        # Top face (z = 1.3, normal = [0, 0, 1])
        dx = np.arange(-0.08 + sp * 0.5, 0.08, sp)
        dxx, dyy_top = np.meshgrid(dx, dy, indexing="ij")
        f_top = np.column_stack([dxx.ravel(), dyy_top.ravel(), np.full(dxx.size, 1.3)])
        samples.append(f_top)
        normals.append(np.tile(np.array([0.0, 0.0, 1.0]), (len(f_top), 1)))

        # Outer walls: x = -3.0, x = +3.0, y = -1.2, y = +1.2
        w_z = np.arange(0.0 + sp * 0.5, 2.5, sp)
        w_y = np.arange(-1.2 + sp * 0.5, 1.2, sp)
        wyy, wzz = np.meshgrid(w_y, w_z, indexing="ij")
        # x = -3.0 (normal = [1, 0, 0])
        samples.append(np.column_stack([np.full(wyy.size, -3.0), wyy.ravel(), wzz.ravel()]))
        normals.append(np.tile(np.array([1.0, 0.0, 0.0]), (wyy.size, 1)))
        # x = +3.0 (normal = [-1, 0, 0])
        samples.append(np.column_stack([np.full(wyy.size, 3.0), wyy.ravel(), wzz.ravel()]))
        normals.append(np.tile(np.array([-1.0, 0.0, 0.0]), (wyy.size, 1)))

        w_x = np.arange(-3.0 + sp * 0.5, 3.0, sp)
        wxx, wzz2 = np.meshgrid(w_x, w_z, indexing="ij")
        # y = -1.2 (normal = [0, 1, 0])
        samples.append(np.column_stack([wxx.ravel(), np.full(wxx.size, -1.2), wzz2.ravel()]))
        normals.append(np.tile(np.array([0.0, 1.0, 0.0]), (wxx.size, 1)))
        # y = +1.2 (normal = [0, -1, 0])
        samples.append(np.column_stack([wxx.ravel(), np.full(wxx.size, 1.2), wzz2.ravel()]))
        normals.append(np.tile(np.array([0.0, -1.0, 0.0]), (wxx.size, 1)))

        self.boundary_pos = np.concatenate(samples, axis=0)
        self.boundary_normal = np.concatenate(normals, axis=0)
        # Normalize normals
        norm = np.linalg.norm(self.boundary_normal, axis=1, keepdims=True)
        self.boundary_normal = np.where(norm > 1e-8, self.boundary_normal / norm, self.boundary_normal)
        self.boundary_vel = np.zeros_like(self.boundary_pos)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        frame = scene.initial_state()
        self.time = float(frame.time)
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy() if frame.mass is not None else np.ones(len(self.position))
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy() if frame.body_id is not None else np.zeros(len(self.position), dtype=np.int32)
        self.material_label = np.asarray(frame.attributes.get("material_label", self.body_id), dtype=np.int32).copy()
        self.particle_radius = np.asarray(frame.attributes.get("particle_radius", np.full(len(self.position), 0.063)), dtype=np.float64).copy()
        self.deformation_gradient = np.asarray(frame.attributes.get("deformation_gradient", np.repeat(np.eye(3)[None], len(self.position), axis=0)), dtype=np.float64).copy()
        self.cumulative_mass_source = np.asarray(frame.fields.get("cumulative_mass_source", np.zeros(len(self.position))), dtype=np.float64).copy()

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        grid_shape = self.config.get("sampling", {}).get("field_grid_shape", [24, 16, 28])
        self.grid_shape = tuple(grid_shape)
        self.grid_spacing = float(self.config.get("sampling", {}).get("spacing", 0.14))
        self.characteristic_length = float(frame.scalars.get("characteristic_length", 4.3))
        self.reference_velocity = float(frame.scalars.get("reference_velocity", 1.0))

        self._init_boundaries()

    def _sph_step(self, dt: float) -> None:
        n = len(self.position)
        if n == 0:
            return

        pos = self.position
        vel = self.velocity
        mass = self.mass
        mat = self.material_label
        h = self.h
        h2 = self.h2

        # Build KD-tree for neighbor search
        tree = scipy.spatial.cKDTree(pos)
        pairs = tree.query_pairs(r=h, output_type="ndarray")

        # Compute density using cubic spline kernel
        # W(r, h) = alpha * (1 - q)^3 for q = r/h <= 1 (approx poly6 / cubic)
        # Poly6 kernel constant: 315 / (64 * pi * h^9)
        poly6_const = 315.0 / (64.0 * np.pi * (h**9))
        spiky_grad_const = -45.0 / (np.pi * (h**6))
        visc_lap_const = 45.0 / (np.pi * (h**6))

        # Density accumulator: start with self-density
        self_rho = mass * poly6_const * (h2**3)
        density = self_rho.copy()

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            diff = pos[i_idx] - pos[j_idx]
            r2 = np.sum(diff**2, axis=1)
            valid = r2 < h2

            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            diff = diff[valid]
            r2 = r2[valid]

            h2_minus_r2 = h2 - r2
            w_poly6 = poly6_const * (h2_minus_r2**3)
            np.add.at(density, i_idx, mass[j_idx] * w_poly6)
            np.add.at(density, j_idx, mass[i_idx] * w_poly6)

        # Avoid zero or negative densities
        density = np.maximum(density, self.rho0 * 0.5)

        # Tait equation of state for pressure
        # P = B * ((rho / rho0)^gamma - 1)
        pressure = self.B * (np.power(density / self.rho0, self.gamma) - 1.0)
        pressure = np.maximum(pressure, 0.0)

        # Compute forces: Pressure force + Viscosity force
        force = np.zeros_like(pos)
        # Gravity
        force += mass[:, None] * self.gravity[None, :]

        # Particle-particle viscosity coefficients
        # High viscosity for material 1 and 3, low for material 0 and 2
        is_high = (mat == 1) | (mat == 3)
        visc_coeff = np.where(is_high, self.mu_high, self.mu_low)
        xsph_coeff = np.where(is_high, self.xsph_high, self.xsph_low)

        # XSPH velocity smoothing accumulator
        xsph_delta_v = np.zeros_like(vel)

        if len(pairs) > 0 and len(i_idx) > 0:
            r = np.sqrt(np.maximum(r2, 1e-12))
            h_minus_r = h - r

            # Spiky gradient of kernel: -45 / (pi * h^6) * (h - r)^2 * (r_vec / r)
            grad_w_mag = spiky_grad_const * (h_minus_r**2)
            grad_w = (diff / r[:, None]) * grad_w_mag[:, None]

            # Pressure force (symmetric formulation)
            # F_p = - m_i * m_j * (p_i / rho_i^2 + p_j / rho_j^2) * grad_W
            p_rho2 = pressure / (density**2)
            p_term = (p_rho2[i_idx] + p_rho2[j_idx])[:, None]
            f_press = - (mass[i_idx] * mass[j_idx])[:, None] * p_term * grad_w

            np.add.at(force, i_idx, f_press)
            np.add.at(force, j_idx, -f_press)

            # Viscosity force:
            # F_v = mu_avg * m_i * m_j * (v_j - v_i) / rho_j * Lap_W
            mu_avg = 0.5 * (visc_coeff[i_idx] + visc_coeff[j_idx])[:, None]
            v_diff = vel[j_idx] - vel[i_idx]
            lap_w = visc_lap_const * h_minus_r[:, None]
            f_visc = mu_avg * mass[i_idx, None] * mass[j_idx, None] * (v_diff / density[j_idx, None]) * lap_w

            np.add.at(force, i_idx, f_visc)
            np.add.at(force, j_idx, -f_visc)

            # XSPH smoothing
            # delta_v_i = sum_j (2 * m_j / (rho_i + rho_j)) * (v_j - v_i) * W
            w_ij = poly6_const * (h2_minus_r2**3)[:, None]
            rho_sum = (density[i_idx] + density[j_idx])[:, None]
            xsph_term_i = xsph_coeff[i_idx, None] * (2.0 * mass[j_idx, None] / rho_sum) * v_diff * w_ij
            xsph_term_j = xsph_coeff[j_idx, None] * (2.0 * mass[i_idx, None] / rho_sum) * (-v_diff) * w_ij

            np.add.at(xsph_delta_v, i_idx, xsph_term_i)
            np.add.at(xsph_delta_v, j_idx, xsph_term_j)

        # Acceleration
        acc = force / mass[:, None]

        # Integrate velocity
        vel += acc * dt
        vel += xsph_delta_v * dt

        # Apply high viscosity damping directly to high-viscosity fluid to stabilize viscous coiling/thick sheet
        high_mask = is_high
        vel[high_mask] *= np.exp(-1.5 * dt)

        # Integrate position
        pos += vel * dt

        # Boundary collisions
        self._enforce_boundary_collisions(pos, vel, dt)

        self.position = pos
        self.velocity = vel

    def _enforce_boundary_collisions(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        r_col = 0.065  # collision offset radius
        restitution = 0.05
        friction = 0.35

        # 1. Floor: z >= r_col
        floor_mask = pos[:, 2] < r_col
        if np.any(floor_mask):
            pos[floor_mask, 2] = r_col
            # Reflect normal velocity if moving downward
            vn = vel[floor_mask, 2]
            neg_vn = vn < 0
            idx = np.where(floor_mask)[0][neg_vn]
            vel[idx, 2] = -restitution * vel[idx, 2]
            vel[idx, 0] *= (1.0 - friction)
            vel[idx, 1] *= (1.0 - friction)

        # 2. Outer domain walls:
        # x in [-2.9, 2.9], y in [-1.15, 1.15]
        # Left wall x = -2.9
        w_left = pos[:, 0] < -2.9
        if np.any(w_left):
            pos[w_left, 0] = -2.9
            vel[w_left, 0] = np.abs(vel[w_left, 0]) * restitution

        # Right wall x = 2.9
        w_right = pos[:, 0] > 2.9
        if np.any(w_right):
            pos[w_right, 0] = 2.9
            vel[w_right, 0] = -np.abs(vel[w_right, 0]) * restitution

        # Front wall y = -1.15
        w_front = pos[:, 1] < -1.15
        if np.any(w_front):
            pos[w_front, 1] = -1.15
            vel[w_front, 1] = np.abs(vel[w_front, 1]) * restitution

        # Back wall y = 1.15
        w_back = pos[:, 1] > 1.15
        if np.any(w_back):
            pos[w_back, 1] = 1.15
            vel[w_back, 1] = -np.abs(vel[w_back, 1]) * restitution

        # 3. Center divider box:
        # divider geometry: half_extents [0.08, 0.9, 0.65], center [0, 0, 0.65]
        # x in [-0.08, 0.08], y in [-0.9, 0.9], z in [0, 1.3]
        div_x_min = -0.08 - r_col
        div_x_max = 0.08 + r_col
        div_y_min = -0.9 - r_col
        div_y_max = 0.9 + r_col
        div_z_max = 1.3 + r_col

        in_div = (
            (pos[:, 0] > div_x_min) & (pos[:, 0] < div_x_max) &
            (pos[:, 1] > div_y_min) & (pos[:, 1] < div_y_max) &
            (pos[:, 2] < div_z_max) & (pos[:, 2] >= 0.0)
        )
        if np.any(in_div):
            in_idx = np.where(in_div)[0]
            for idx in in_idx:
                p = pos[idx]
                v = vel[idx]
                # Distances to the 5 boundary planes of the box
                d_left = p[0] - div_x_min   # > 0
                d_right = div_x_max - p[0] # > 0
                d_front = p[1] - div_y_min
                d_back = div_y_max - p[1]
                d_top = div_z_max - p[2]

                min_d = min(d_left, d_right, d_front, d_back, d_top)
                if min_d == d_left:
                    pos[idx, 0] = div_x_min
                    vel[idx, 0] = -np.abs(v[0]) * restitution
                elif min_d == d_right:
                    pos[idx, 0] = div_x_max
                    vel[idx, 0] = np.abs(v[0]) * restitution
                elif min_d == d_front:
                    pos[idx, 1] = div_y_min
                    vel[idx, 1] = -np.abs(v[1]) * restitution
                elif min_d == d_back:
                    pos[idx, 1] = div_y_max
                    vel[idx, 1] = np.abs(v[1]) * restitution
                else:
                    pos[idx, 2] = div_z_max
                    vel[idx, 2] = np.abs(v[2]) * restitution

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        while self.time < target_time - 1e-7:
            dt = min(self.sim_dt, target_time - self.time)
            self._sph_step(dt)
            self.time += dt
        self.time = target_time

        return self._build_state_frame()

    def _compute_grid_fields(self):
        """Compute active_mask, liquid_level_set, and velocity_field on diagnostic grid."""
        nx, ny, nz = self.grid_shape
        # Create coordinate arrays
        xs = np.linspace(self.bounds_min[0], self.bounds_max[0], nx)
        ys = np.linspace(self.bounds_min[1], self.bounds_max[1], ny)
        zs = np.linspace(self.bounds_min[2], self.bounds_max[2], nz)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        pos = self.position
        vel = self.velocity
        n_particles = len(pos)

        if n_particles > 0:
            tree = scipy.spatial.cKDTree(pos)
            # Find nearest particle distance for signed distance / level set
            dist, idx = tree.query(grid_pts, k=1)
            dist_3d = dist.reshape(nx, ny, nz)
            # Signed distance: negative inside liquid (radius ~ 0.12), positive outside
            r_eff = 0.12
            liquid_level_set = dist_3d - r_eff

            # Active mask: cells near liquid
            active_mask = liquid_level_set <= 0.15

            # Velocity field: interpolate velocity from particles within radius R
            # Use query with radius
            vel_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
            # Interpolate velocity onto grid for cells near particles
            close_cells = np.where(dist < 0.35)[0]
            if len(close_cells) > 0:
                cell_pts = grid_pts[close_cells]
                # Query nearest 4 particles
                k_nn = min(4, n_particles)
                dists, neighbors = tree.query(cell_pts, k=k_nn)
                if k_nn == 1:
                    dists = dists[:, None]
                    neighbors = neighbors[:, None]
                weights = 1.0 / np.maximum(dists, 1e-4)
                weights /= np.sum(weights, axis=1, keepdims=True)
                v_interp = np.sum(vel[neighbors] * weights[:, :, None], axis=1)

                flat_vel = vel_field.reshape(-1, 3)
                flat_vel[close_cells] = v_interp
        else:
            liquid_level_set = np.full((nx, ny, nz), 1.0, dtype=np.float64)
            active_mask = np.zeros((nx, ny, nz), dtype=bool)
            vel_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)

        return active_mask, liquid_level_set, vel_field

    def _build_state_frame(self) -> StateFrame:
        active_mask, liquid_level_set, vel_field = self._compute_grid_fields()

        b_count = len(self.boundary_pos)
        b_target_vel = np.zeros((b_count, 3), dtype=np.float64)

        fields = {
            "active_mask": active_mask,
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": b_target_vel,
            "boundary_velocity": self.boundary_vel.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "liquid_level_set": liquid_level_set,
            "velocity_field": vel_field,
        }

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.particle_radius.copy(),
            "material_label": self.material_label.copy(),
            "deformation_gradient": self.deformation_gradient.copy(),
        }

        scalars = {
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.reference_velocity,
        }

        frame = StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_particles": len(self.position),
        }

