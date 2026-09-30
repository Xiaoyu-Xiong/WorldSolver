"""Solver for 131_u_corridor_dam_break using an efficient SPH/particle fluid method."""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.dt = 0.002  # substep dt

        # State particles
        self.pos = None
        self.vel = None
        self.mass = None
        self.labels = None
        self.radius = 0.06
        self.h = 0.18
        self.rho0 = 1000.0
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.c_s = 25.0  # speed of sound for Tait's EOS
        self.gamma = 7.0
        self.B = (self.rho0 * (self.c_s ** 2)) / self.gamma
        self.nu = 0.05  # artificial viscosity

        # Diagnostic grid
        self.field_grid_shape = (24, 16, 28)
        self.grid_spacing = 0.14
        self.grid_origin = np.array([-1.5, -2.0, 0.0])
        self.characteristic_length = 1.0
        self.reference_velocity = 1.0

        # Boundary description
        self.b_pos = None
        self.b_normals = None
        self.b_vel = None
        self.b_target_vel = None

        # Gate lifting animation
        self.gate_lift_speed = 2.5
        self.gate_lift_start = 0.0

    def _setup_boundaries(self):
        """Generate static boundary sample points with normals."""
        # Boundaries:
        # Floor: z = 0, x in [-1.25, 1.25], y in [-1.75, 1.75]
        # Outer walls: x = -1.25 (norm [1,0,0]), x = 1.25 (norm [-1,0,0]), y = 1.65 (norm [0,-1,0]), y = -1.75 (norm [0,1,0])
        # Central divider for U-corridor: x in [-0.2, 0.2], y in [-1.75, 0.6] -> walls at x = -0.2 (norm [-1,0,0]), x = 0.2 (norm [1,0,0]), end at y = 0.6 (norm [0,1,0])
        b_pts = []
        b_nrms = []

        spacing = 0.07
        # Floor samples
        fx = np.arange(-1.25, 1.26, spacing)
        fy = np.arange(-1.75, 1.76, spacing)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        fz = np.zeros_like(FX)
        floor_pts = np.column_stack((FX.ravel(), FY.ravel(), fz.ravel()))
        floor_nrms = np.repeat([[0.0, 0.0, 1.0]], len(floor_pts), axis=0)
        b_pts.append(floor_pts)
        b_nrms.append(floor_nrms)

        # Left outer wall: x = -1.25, y in [-1.75, 1.75], z in [0, 1.2]
        wz = np.arange(0.0, 1.25, spacing)
        WY, WZ = np.meshgrid(fy, wz, indexing="ij")
        wx = np.full_like(WY, -1.25)
        b_pts.append(np.column_stack((wx.ravel(), WY.ravel(), WZ.ravel())))
        b_nrms.append(np.repeat([[1.0, 0.0, 0.0]], WY.size, axis=0))

        # Right outer wall: x = 1.25
        wx = np.full_like(WY, 1.25)
        b_pts.append(np.column_stack((wx.ravel(), WY.ravel(), WZ.ravel())))
        b_nrms.append(np.repeat([[-1.0, 0.0, 0.0]], WY.size, axis=0))

        # Back outer wall: y = 1.65, x in [-1.25, 1.25], z in [0, 1.2]
        WX, WZ2 = np.meshgrid(fx, wz, indexing="ij")
        wy = np.full_like(WX, 1.65)
        b_pts.append(np.column_stack((WX.ravel(), wy.ravel(), WZ2.ravel())))
        b_nrms.append(np.repeat([[0.0, -1.0, 0.0]], WX.size, axis=0))

        # Front outer wall (closed ends of U): y = -1.75, x in [-1.25, 1.25], z in [0, 1.2]
        wy = np.full_like(WX, -1.75)
        b_pts.append(np.column_stack((WX.ravel(), wy.ravel(), WZ2.ravel())))
        b_nrms.append(np.repeat([[0.0, 1.0, 0.0]], WX.size, axis=0))

        # Central divider: x in [-0.25, 0.25], y in [-1.75, 0.6], z in [0, 1.2]
        cy = np.arange(-1.75, 0.61, spacing)
        CY, CZ = np.meshgrid(cy, wz, indexing="ij")
        cx1 = np.full_like(CY, -0.25)
        b_pts.append(np.column_stack((cx1.ravel(), CY.ravel(), CZ.ravel())))
        b_nrms.append(np.repeat([[-1.0, 0.0, 0.0]], CY.size, axis=0))

        cx2 = np.full_like(CY, 0.25)
        b_pts.append(np.column_stack((cx2.ravel(), CY.ravel(), CZ.ravel())))
        b_nrms.append(np.repeat([[1.0, 0.0, 0.0]], CY.size, axis=0))

        # End cap of divider at y = 0.6
        cx_div = np.arange(-0.25, 0.26, spacing)
        CX_div, CZ_div = np.meshgrid(cx_div, wz, indexing="ij")
        c_y_end = np.full_like(CX_div, 0.6)
        b_pts.append(np.column_stack((CX_div.ravel(), c_y_end.ravel(), CZ_div.ravel())))
        b_nrms.append(np.repeat([[0.0, 1.0, 0.0]], CX_div.size, axis=0))

        self.b_pos = np.concatenate(b_pts, axis=0)
        self.b_normals = np.concatenate(b_nrms, axis=0)
        # Normalize
        norms = np.linalg.norm(self.b_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.b_normals /= norms

        self.b_vel = np.zeros_like(self.b_pos)
        self.b_target_vel = np.zeros_like(self.b_pos)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init_frame = scene.initial_state()
        self.pos = init_frame.position.copy()
        self.vel = init_frame.velocity.copy()
        self.mass = init_frame.mass.copy()
        self.labels = init_frame.body_id.copy()

        sampling = scene.config.get("sampling", {})
        self.field_grid_shape = tuple(sampling.get("field_grid_shape", [24, 16, 28]))
        self.grid_spacing = float(sampling.get("spacing", 0.14))
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.characteristic_length = max(float(np.ptp(self.pos, axis=0).max()), self.grid_spacing)
        self.reference_velocity = float(np.sqrt(np.linalg.norm(self.gravity) * self.characteristic_length))

        self.radius = 0.45 * self.grid_spacing
        self.h = 1.5 * self.grid_spacing

        # Compute initial grid bounds
        self.grid_origin = np.array([-1.4, -1.9, -0.05])

        self._setup_boundaries()
        self.current_time = 0.0

    def _apply_boundary_collisions(self, p, v):
        """Rigid collision handling with corridor geometry and dam gate."""
        # Floor: z = 0
        z_min = 0.02
        below = p[:, 2] < z_min
        if np.any(below):
            p[below, 2] = z_min + 0.001 * (z_min - p[below, 2])
            v[below, 2] = np.maximum(v[below, 2], 0.0)
            # Friction
            v[below, 0] *= 0.95
            v[below, 1] *= 0.95

        # Outer walls: x in [-1.15, 1.15], y in [-1.65, 1.55]
        # Left outer wall: x = -1.15
        left_wall = p[:, 0] < -1.15
        if np.any(left_wall):
            p[left_wall, 0] = -1.15
            v[left_wall, 0] = np.maximum(v[left_wall, 0], 0.0)

        # Right outer wall: x = 1.15
        right_wall = p[:, 0] > 1.15
        if np.any(right_wall):
            p[right_wall, 0] = 1.15
            v[right_wall, 0] = np.minimum(v[right_wall, 0], 0.0)

        # Back outer wall: y = 1.55
        back_wall = p[:, 1] > 1.55
        if np.any(back_wall):
            p[back_wall, 1] = 1.55
            v[back_wall, 1] = np.minimum(v[back_wall, 1], 0.0)

        # Front outer wall: y = -1.65
        front_wall = p[:, 1] < -1.65
        if np.any(front_wall):
            p[front_wall, 1] = -1.65
            v[front_wall, 1] = np.maximum(v[front_wall, 1], 0.0)

        # Central divider: x in [-0.15, 0.15], y in [-1.65, 0.6]
        # If inside central divider, push to nearest side
        in_divider = (p[:, 0] >= -0.15) & (p[:, 0] <= 0.15) & (p[:, 1] <= 0.6)
        if np.any(in_divider):
            idx = np.where(in_divider)[0]
            for i in idx:
                px = p[i, 0]
                py = p[i, 1]
                # Distances to surfaces
                d_left = abs(px - (-0.15))
                d_right = abs(px - 0.15)
                d_end = abs(py - 0.6)
                min_d = min(d_left, d_right, d_end)
                if min_d == d_left:
                    p[i, 0] = -0.15
                    v[i, 0] = min(v[i, 0], 0.0)
                elif min_d == d_right:
                    p[i, 0] = 0.15
                    v[i, 0] = max(v[i, 0], 0.0)
                else:
                    p[i, 1] = 0.6
                    v[i, 1] = max(v[i, 1], 0.0)

        # Dam gate: located around y = -0.6, lifts upward over time [0, 0.4]s
        gate_z_bottom = -0.6 + self.gate_lift_speed * max(0.0, self.current_time - self.gate_lift_start)
        if gate_z_bottom < 1.1:
            # Gate is active
            gate_hit = (p[:, 1] >= -0.64) & (p[:, 1] <= -0.56) & (p[:, 2] < gate_z_bottom + 1.1) & (p[:, 0] <= -0.15)
            if np.any(gate_hit):
                p[gate_hit, 1] = -0.64
                v[gate_hit, 1] = np.minimum(v[gate_hit, 1], 0.0)

        # Ceiling limit
        top = p[:, 2] > 2.0
        if np.any(top):
            p[top, 2] = 2.0
            v[top, 2] = np.minimum(v[top, 2], 0.0)

    def _step(self, dt: float):
        """Single physics sub-step using particle-based fluid mechanics."""
        N = len(self.pos)
        if N == 0:
            return

        # Simple neighbor search / spatial grid
        cell_size = self.h
        coords = np.floor(self.pos / cell_size).astype(np.int32)
        # Hash table for spatial hashing
        grid = {}
        for i in range(N):
            cell = (coords[i, 0], coords[i, 1], coords[i, 2])
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        # Density estimation using cubic spline / poly6 kernel
        densities = np.zeros(N)
        poly6_const = 315.0 / (64.0 * np.pi * (self.h ** 9))
        h2 = self.h * self.h

        for cell, indices in grid.items():
            # Check 3x3x3 neighboring cells
            neighbor_cells = [
                (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            ]
            neighbor_indices = []
            for nc in neighbor_cells:
                if nc in grid:
                    neighbor_indices.extend(grid[nc])
            neighbor_indices = np.array(neighbor_indices, dtype=int)
            n_pos = self.pos[neighbor_indices]

            for i in indices:
                disp = self.pos[i] - n_pos
                r2 = np.sum(disp ** 2, axis=1)
                mask = r2 < h2
                r2_m = r2[mask]
                w = poly6_const * ((h2 - r2_m) ** 3)
                densities[i] = np.sum(self.mass[neighbor_indices[mask]] * w)

        densities = np.maximum(densities, self.rho0 * 0.5)

        # Pressure calculation using Tait equation of state
        pressures = self.B * ((densities / self.rho0) ** self.gamma - 1.0)
        pressures = np.maximum(pressures, 0.0)

        # Forces: Gravity + Pressure Gradient + Viscosity
        spiky_const = -45.0 / (np.pi * (self.h ** 6))
        visc_const = 45.0 / (np.pi * (self.h ** 6))
        forces = np.zeros((N, 3))

        # Add gravity
        forces += self.gravity[None, :] * densities[:, None]

        # Inter-particle forces
        for cell, indices in grid.items():
            neighbor_cells = [
                (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            ]
            neighbor_indices = []
            for nc in neighbor_cells:
                if nc in grid:
                    neighbor_indices.extend(grid[nc])
            neighbor_indices = np.array(neighbor_indices, dtype=int)
            n_pos = self.pos[neighbor_indices]
            n_vel = self.vel[neighbor_indices]
            n_mass = self.mass[neighbor_indices]
            n_rho = densities[neighbor_indices]
            n_p = pressures[neighbor_indices]

            for i in indices:
                disp = self.pos[i] - n_pos  # (M, 3)
                r = np.linalg.norm(disp, axis=1)
                mask = (r < self.h) & (r > 1e-6)
                if not np.any(mask):
                    continue

                r_m = r[mask, None]
                disp_m = disp[mask]
                m_j = n_mass[mask, None]
                rho_j = n_rho[mask, None]
                p_j = n_p[mask, None]
                v_j = n_vel[mask]
                v_i = self.vel[i]
                p_i = pressures[i]
                rho_i = densities[i]

                # Pressure gradient force
                grad_w = spiky_const * ((self.h - r_m) ** 2) * (disp_m / r_m)
                f_press = -m_j * ((p_i / (rho_i ** 2)) + (p_j / (rho_j ** 2))) * grad_w
                forces[i] += rho_i * np.sum(f_press, axis=0)

                # Viscosity force
                lap_w = visc_const * (self.h - r_m)
                f_visc = self.nu * m_j * ((v_j - v_i) / rho_j) * lap_w
                forces[i] += np.sum(f_visc, axis=0)

        # Acceleration and time integration (Symplectic Euler)
        acc = forces / densities[:, None]
        # Clamp excessive accelerations for stability
        acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
        max_acc = 200.0
        acc = np.where(acc_norm > max_acc, acc * (max_acc / acc_norm), acc)

        self.vel += acc * dt
        # Clamp velocity
        vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
        max_vel = 15.0
        self.vel = np.where(vel_norm > max_vel, self.vel * (max_vel / vel_norm), self.vel)

        self.pos += self.vel * dt

        # Boundary collisions
        self._apply_boundary_collisions(self.pos, self.vel)

    def advance_to(self, time: float) -> StateFrame:
        # Step physics up to requested sample time
        while self.current_time < time - 1e-6:
            dt = min(self.dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt
        self.current_time = time

        # Generate diagnostic fields
        nx, ny, nz = self.field_grid_shape
        gx = self.grid_origin[0] + np.arange(nx) * self.grid_spacing
        gy = self.grid_origin[1] + np.arange(ny) * self.grid_spacing
        gz = self.grid_origin[2] + np.arange(nz) * self.grid_spacing

        # Build signed liquid level set and diagnostic velocity grid
        # We sample level set as distance to nearest particle minus radius
        # Compute level set on grid efficiently
        level_set = np.full((nx, ny, nz), 1.0, dtype=np.float32)
        vel_grid = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        active_mask = np.zeros((nx, ny, nz), dtype=bool)

        # Map particles to grid cells
        if len(self.pos) > 0:
            ijk = np.floor((self.pos - self.grid_origin) / self.grid_spacing).astype(np.int32)
            valid = (ijk[:, 0] >= 0) & (ijk[:, 0] < nx) & \
                    (ijk[:, 1] >= 0) & (ijk[:, 1] < ny) & \
                    (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
            ijk_valid = ijk[valid]
            pos_valid = self.pos[valid]
            vel_valid = self.vel[valid]

            # Mark active cells around particles
            for idx in range(len(pos_valid)):
                ci, cj, ck = ijk_valid[idx]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            ii, jj, kk = ci + di, cj + dj, ck + dk
                            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                                active_mask[ii, jj, kk] = True
                                cell_center = self.grid_origin + np.array([ii + 0.5, jj + 0.5, kk + 0.5]) * self.grid_spacing
                                dist = np.linalg.norm(cell_center - pos_valid[idx]) - self.radius
                                if dist < level_set[ii, jj, kk]:
                                    level_set[ii, jj, kk] = dist
                                    vel_grid[ii, jj, kk] = vel_valid[idx]

        # Format StateFrame
        N = len(self.pos)
        state = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(N, self.radius),
                "material_label": self.labels.copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], N, axis=0),
            },
            fields={
                "active_mask": active_mask,
                "boundary_normal": self.b_normals.copy(),
                "boundary_target_velocity": self.b_target_vel.copy(),
                "boundary_velocity": self.b_vel.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "liquid_level_set": level_set,
                "velocity_field": vel_grid,
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.grid_spacing),
                "reference_velocity": float(self.reference_velocity),
                "contact_length_scale": float(self.grid_spacing),
            }
        )
        state.validate()
        return state

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": len(self.pos) if self.pos is not None else 0,
            "current_time": self.current_time,
        }

