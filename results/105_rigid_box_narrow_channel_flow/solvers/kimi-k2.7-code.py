"""Grid-based incompressible fluid + rigid body solver for the narrow-channel experiment.

Physics summary
---------------
* 2-D incompressible Stokes/pressure projection on a regular MAC grid (x-z plane).
* Moving rigid box acts as a solid obstacle with no-flow boundary faces.
* Fluid is driven by gravity and a time-ramped upward body acceleration.
* Pressure on the box surface yields the two-way coupling force/torque.
* Passive marker particles are advected by the grid velocity field.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.cfg = scene.config
        self.t = 0.0

        # ------------------------------------------------------------------
        # Domain / grid parameters (2D in x-z, fixed thickness in y)
        # ------------------------------------------------------------------
        fluid_cfg = self.cfg["fluid"]
        self.xmin, _, self.zmin = fluid_cfg["domain_min"]
        self.xmax, _, self.zmax = fluid_cfg["domain_max"]
        self.nx, self.ny, self.nz = fluid_cfg["nominal_resolution"]
        self.dx = (self.xmax - self.xmin) / float(self.nx)
        self.dz = (self.zmax - self.zmin) / float(self.nz)
        self.dy = 0.08  # thickness used for geometry and forces
        self.rho = float(fluid_cfg["density"])

        # Cell / face coordinates
        self.cx = np.linspace(self.xmin + 0.5 * self.dx,
                              self.xmax - 0.5 * self.dx, self.nx)
        self.cz = np.linspace(self.zmin + 0.5 * self.dz,
                              self.zmax - 0.5 * self.dz, self.nz)
        self.xf = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.zf = np.linspace(self.zmin, self.zmax, self.nz + 1)
        self.CX, self.CZ = np.meshgrid(self.cx, self.cz, indexing="ij")

        # ------------------------------------------------------------------
        # Rigid box state
        # ------------------------------------------------------------------
        box_cfg = self.cfg["rigid_box"]
        self.box_size = np.array(box_cfg["size"], dtype=float)
        self.box_half = self.box_size * 0.5
        self.box_mass = float(box_cfg["mass"])
        self.box_inv_mass = 1.0 / self.box_mass
        # Moment of inertia about y-axis for a rectangular box
        self.box_I = self.box_mass / 3.0 * (self.box_half[0] ** 2 + self.box_half[2] ** 2)
        self.box_inv_I = 1.0 / self.box_I

        initial_frame = scene.initial_state()
        self.box_pos = initial_frame.position[0].copy()
        self.box_vel = initial_frame.velocity[0].copy()
        q = initial_frame.orientation[0]
        # q = [0, sin(theta/2), 0, cos(theta/2)]
        self.box_angle = 2.0 * np.arctan2(q[1], q[3])
        self.box_omega = float(initial_frame.angular_velocity[0, 1])

        # ------------------------------------------------------------------
        # Markers
        # ------------------------------------------------------------------
        self.marker_pos = initial_frame.position[1:].copy()
        self.marker_vel = initial_frame.velocity[1:].copy()
        self.marker_mass = initial_frame.mass[1:].copy()
        self.n_markers = self.marker_pos.shape[0]

        # ------------------------------------------------------------------
        # MAC grid fields
        # ------------------------------------------------------------------
        self.u = np.zeros((self.nx + 1, self.nz), dtype=float)
        self.w = np.zeros((self.nx, self.nz + 1), dtype=float)
        self.p = np.zeros((self.nx, self.nz), dtype=float)

        # ------------------------------------------------------------------
        # Controls / forcing
        # ------------------------------------------------------------------
        self.gravity = np.array(self.cfg["gravity"], dtype=float)
        drive = self.cfg["controls"][0]
        self.drive_dir = np.array(drive["direction"], dtype=float)
        self.drive_mag = float(drive["magnitude"])
        self.drive_start = float(drive["start_time"])
        self.drive_end = float(drive["end_time"])
        self.drive_ramp = float(drive["ramp_duration"])

        # ------------------------------------------------------------------
        # Time stepping
        # ------------------------------------------------------------------
        # sample_dt = 0.02; use 10 substeps per sample
        self.dt_sub = 0.002

        # ------------------------------------------------------------------
        # Budget / auxiliary state
        # ------------------------------------------------------------------
        self.cum_external_impulse = np.zeros((3, 3), dtype=float)
        self.current_external_force = np.zeros((3, 3, 3), dtype=float)
        self.current_interaction_force = np.zeros((1, 3), dtype=float)

    # ----------------------------------------------------------------------
    # Driver
    # ----------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        while self.t < time - 1e-12:
            dt = min(self.dt_sub, time - self.t)
            self._step(dt)
            self.t += dt
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "box_pos": self.box_pos.copy(),
            "box_vel": self.box_vel.copy(),
            "box_angle": float(self.box_angle),
            "box_omega": float(self.box_omega),
        }

    # ----------------------------------------------------------------------
    # Substep
    # ----------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # 1. Which cells are occupied by the box?
        solid = self._solid_mask()

        # 2. Wall boundaries: no normal flow
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[:, 0] = 0.0
        self.w[:, -1] = 0.0

        # 3. Moving solid (box) boundary velocities on MAC faces
        self._apply_solid_boundary(solid)

        # 4. Body forces on internal w-faces (z-direction only)
        a_drive = self._drive_acceleration(self.t)
        a_z = self.gravity[2] + a_drive
        # w-faces not on the top/bottom domain boundary
        self.w[:, 1:-1] += dt * a_z

        # 5. Pressure projection to enforce incompressibility
        self._project(dt, solid)

        # 6. Pressure force/torque on the box
        F_pressure, T_pressure = self._pressure_force_on_box(solid)

        # 7. Wall contact forces on the box
        F_wall, T_wall = self._box_wall_forces()

        # 8. Gravity on the box
        F_grav_box = self.box_mass * self.gravity

        # 9. Update box state
        F_total = F_pressure + F_wall + F_grav_box
        T_total = T_pressure + T_wall

        self.box_vel += dt * (F_total * self.box_inv_mass)
        self.box_omega += dt * (T_total * self.box_inv_I)
        self.box_pos[0] += dt * self.box_vel[0]
        self.box_pos[2] += dt * self.box_vel[2]
        self.box_angle += dt * self.box_omega

        # 10. Advect passive markers
        self._advect_markers(dt)

        # 11. Update budget fields
        self._update_budgets(
            dt,
            F_grav_box,
            F_pressure,
            F_wall,
            a_drive,
        )

    # ----------------------------------------------------------------------
    # Solid / geometry helpers
    # ----------------------------------------------------------------------
    def _solid_mask(self) -> np.ndarray:
        """Return boolean array (nx, nz) of cells inside the rotated box."""
        dx = self.CX - self.box_pos[0]
        dz = self.CZ - self.box_pos[2]
        c = np.cos(self.box_angle)
        s = np.sin(self.box_angle)
        local_x = c * dx + s * dz
        local_z = -s * dx + c * dz
        return (np.abs(local_x) <= self.box_half[0]) & (np.abs(local_z) <= self.box_half[2])

    def _box_velocity_at(self, x: np.ndarray, z: np.ndarray):
        """Return (u, w) velocity of the box at world coordinates (x, z)."""
        rx = x - self.box_pos[0]
        rz = z - self.box_pos[2]
        u = self.box_vel[0] + self.box_omega * rz
        w = self.box_vel[2] - self.box_omega * rx
        return u, w

    def _apply_solid_boundary(self, solid: np.ndarray) -> None:
        """Set face velocities on box-fluid faces to the box velocity."""
        # x-faces (nx+1, nz)
        mask = np.zeros((self.nx + 1, self.nz), dtype=bool)
        mask[1:-1] = solid[:-1, :] | solid[1:, :]
        if np.any(mask):
            xg, zg = np.meshgrid(self.xf, self.cz, indexing="ij")
            u_s, _ = self._box_velocity_at(xg[mask], zg[mask])
            self.u[mask] = u_s

        # z-faces (nx, nz+1)
        mask = np.zeros((self.nx, self.nz + 1), dtype=bool)
        mask[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        if np.any(mask):
            xg, zg = np.meshgrid(self.cx, self.zf, indexing="ij")
            _, w_s = self._box_velocity_at(xg[mask], zg[mask])
            self.w[mask] = w_s

    def _pressure_force_on_box(self, solid: np.ndarray):
        """Compute net force and torque on the box due to fluid pressure."""
        F = np.zeros(3, dtype=float)
        T = 0.0
        bx, bz = self.box_pos[0], self.box_pos[2]

        # x-faces between cells (i-1,j) and (i,j); face at i in [1, nx-1]
        if self.nx >= 2:
            mask = (solid[:-1, :] & ~solid[1:, :]) | (~solid[:-1, :] & solid[1:, :])
            if np.any(mask):
                i, j = np.where(mask)
                # face between left cell (i,j) and right cell (i+1,j)?
                # solid[:-1,:] has shape (nx-1,nz); index i is left cell index in [0,nx-2]
                # face is at i+1
                face_i = i + 1
                # Pressure: use the fluid cell pressure
                p_left = self.p[i, j]
                p_right = self.p[i + 1, j]
                p = np.where(solid[i, j], p_right, p_left)
                # normal points from solid to fluid
                n = np.where(solid[i, j], -1.0, 1.0)
                area = self.dy * self.dz
                f = p * n * area
                F_x = np.sum(f)
                F[0] += F_x
                # torque r x F, r relative to box center
                x_face = self.xf[face_i]
                z_face = self.cz[j]
                # contribution to z-force is zero for x-face normal? No, x-face normal is x only.
                T += (x_face - bx) * 0.0 - (z_face - bz) * F_x

        # z-faces between cells (i,j-1) and (i,j); face at j in [1, nz-1]
        if self.nz >= 2:
            mask = (solid[:, :-1] & ~solid[:, 1:]) | (~solid[:, :-1] & solid[:, 1:])
            if np.any(mask):
                i, j = np.where(mask)
                # j is bottom cell index in [0, nz-2]
                p_bottom = self.p[i, j]
                p_top = self.p[i, j + 1]
                p = np.where(solid[i, j], p_top, p_bottom)
                n = np.where(solid[i, j], -1.0, 1.0)
                area = self.dx * self.dy
                f = p * n * area
                F_z = np.sum(f)
                F[2] += F_z
                x_face = self.cx[i]
                z_face = self.zf[j + 1]
                T += (x_face - bx) * F_z - (z_face - bz) * 0.0

        return F, T

    def _box_wall_forces(self):
        """Penalty force/torque from box corners penetrating walls/floor."""
        a = self.box_half[0]
        b = self.box_half[2]
        local = np.array([[a, b], [a, -b], [-a, b], [-a, -b]], dtype=float)
        c = np.cos(self.box_angle)
        s = np.sin(self.box_angle)
        x = self.box_pos[0] + c * local[:, 0] - s * local[:, 1]
        z = self.box_pos[2] + s * local[:, 0] + c * local[:, 1]

        F = np.zeros(3, dtype=float)
        T = 0.0
        k = 1e5
        damp = 5e3

        for idx in range(4):
            cx_, cz_ = x[idx], z[idx]
            # side walls
            if cx_ < self.xmin:
                d = self.xmin - cx_
                n = np.array([1.0, 0.0, 0.0])
            elif cx_ > self.xmax:
                d = cx_ - self.xmax
                n = np.array([-1.0, 0.0, 0.0])
            else:
                d = 0.0
                n = np.zeros(3)
            if d > 0:
                v = self._box_velocity_at(np.array([cx_]), np.array([cz_]))
                v_vec = np.array([v[0].item(), 0.0, v[1].item()])
                f = k * d * n - damp * v_vec
                F += f
                rx = cx_ - self.box_pos[0]
                rz = cz_ - self.box_pos[2]
                T += rx * f[2] - rz * f[0]

            # floor (no top collision)
            if cz_ < self.zmin:
                d = self.zmin - cz_
                n = np.array([0.0, 0.0, 1.0])
                v = self._box_velocity_at(np.array([cx_]), np.array([cz_]))
                v_vec = np.array([v[0].item(), 0.0, v[1].item()])
                f = k * d * n - damp * v_vec
                F += f
                rx = cx_ - self.box_pos[0]
                rz = cz_ - self.box_pos[2]
                T += rx * f[2] - rz * f[0]

        return F, T

    # ----------------------------------------------------------------------
    # Pressure projection
    # ----------------------------------------------------------------------
    def _project(self, dt: float, solid: np.ndarray) -> None:
        fluid = ~solid
        if not np.any(fluid):
            return

        idx = np.full((self.nx, self.nz), -1, dtype=int)
        idx[fluid] = np.arange(np.count_nonzero(fluid))
        nf = idx.max() + 1

        # Divergence of the intermediate velocity field
        D = (self.u[1:, :] - self.u[:-1, :]) / self.dx + (self.w[:, 1:] - self.w[:, :-1]) / self.dz
        b = (self.rho / dt) * D[fluid]

        # Build Laplacian matrix for fluid cells
        rows = []
        cols = []
        vals = []

        # diagonal placeholders
        diag = np.zeros(nf, dtype=float)

        # x-direction connections
        if self.nx >= 2:
            mask = fluid[:-1, :] & fluid[1:, :]
            if np.any(mask):
                c1 = idx[:-1, :][mask]
                c2 = idx[1:, :][mask]
                v = 1.0 / (self.dx * self.dx)
                diag[c1] += v
                diag[c2] += v
                rows.extend([c1, c2, c1, c2])
                cols.extend([c1, c2, c2, c1])
                vals.extend([np.full(c1.shape, v), np.full(c1.shape, v),
                             np.full(c1.shape, -v), np.full(c1.shape, -v)])

        # z-direction connections
        if self.nz >= 2:
            mask = fluid[:, :-1] & fluid[:, 1:]
            if np.any(mask):
                c1 = idx[:, :-1][mask]
                c2 = idx[:, 1:][mask]
                v = 1.0 / (self.dz * self.dz)
                diag[c1] += v
                diag[c2] += v
                rows.extend([c1, c2, c1, c2])
                cols.extend([c1, c2, c2, c1])
                vals.extend([np.full(c1.shape, v), np.full(c1.shape, v),
                             np.full(c1.shape, -v), np.full(c1.shape, -v)])

        all_rows = np.concatenate([np.arange(nf), *rows]) if rows else np.arange(nf)
        all_cols = np.concatenate([np.arange(nf), *cols]) if cols else np.arange(nf)
        all_vals = np.concatenate([diag, *vals]) if vals else diag

        A = sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(nf, nf)).tocsr()
        # Small regularisation to make the matrix invertible when the Neumann
        # problem is singular.
        A = A + 1e-12 * sp.eye(nf, format="csr")

        lu = spla.splu(A)
        p_fluid = lu.solve(b)

        self.p = np.zeros((self.nx, self.nz), dtype=float)
        self.p[fluid] = p_fluid

        # Update internal x-faces
        if self.nx >= 2:
            mask = fluid[:-1, :] & fluid[1:, :]
            if np.any(mask):
                i, j = np.where(mask)
                # left cell index i in [0, nx-2], face is u[i+1, j]
                face_i = i + 1
                grad = (self.p[i + 1, j] - self.p[i, j]) / self.dx
                self.u[face_i, j] -= (dt / self.rho) * grad

        # Update internal z-faces
        if self.nz >= 2:
            mask = fluid[:, :-1] & fluid[:, 1:]
            if np.any(mask):
                i, j = np.where(mask)
                    # bottom cell index j in [0, nz-2], face is w[i, j+1]
                face_j = j + 1
                grad = (self.p[i, j + 1] - self.p[i, j]) / self.dz
                self.w[i, face_j] -= (dt / self.rho) * grad

    # ----------------------------------------------------------------------
    # Markers
    # ----------------------------------------------------------------------
    def _advect_markers(self, dt: float) -> None:
        if self.n_markers == 0:
            return

        # Cell-centered velocity
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        w_c = 0.5 * (self.w[:, :-1] + self.w[:, 1:])

        xi = (self.marker_pos[:, 0] - self.xmin) / self.dx - 0.5
        zi = (self.marker_pos[:, 2] - self.zmin) / self.dz - 0.5
        coords = np.vstack([xi, zi])

        mu = map_coordinates(u_c, coords, order=1, mode="nearest")
        mw = map_coordinates(w_c, coords, order=1, mode="nearest")

        self.marker_vel[:, 0] = mu
        self.marker_vel[:, 2] = mw
        self.marker_pos[:, 0] += dt * mu
        self.marker_pos[:, 2] += dt * mw

        # Keep inside domain
        self.marker_pos[:, 0] = np.clip(self.marker_pos[:, 0], self.xmin, self.xmax)
        self.marker_pos[:, 2] = np.clip(self.marker_pos[:, 2], self.zmin, self.zmax)

        # Push out of the box if needed
        self._resolve_marker_solid_overlap()

    def _resolve_marker_solid_overlap(self) -> None:
        dx = self.marker_pos[:, 0] - self.box_pos[0]
        dz = self.marker_pos[:, 2] - self.box_pos[2]
        c = np.cos(self.box_angle)
        s = np.sin(self.box_angle)
        local_x = c * dx + s * dz
        local_z = -s * dx + c * dz

        inside = (np.abs(local_x) <= self.box_half[0]) & (np.abs(local_z) <= self.box_half[2])
        if not np.any(inside):
            return

        # For each inside marker, push to the nearest box face in local space
        ax = np.abs(local_x[inside])
        az = np.abs(local_z[inside])
        # distance to x-face vs z-face
        d_x = self.box_half[0] - ax
        d_z = self.box_half[2] - az

        local_x_i = local_x[inside]
        local_z_i = local_z[inside]

        # push along the shorter distance
        push_x = d_x < d_z
        local_x_i[push_x] = np.sign(local_x_i[push_x]) * self.box_half[0]
        push_z = ~push_x
        local_z_i[push_z] = np.sign(local_z_i[push_z]) * self.box_half[2]

        # back to world
        self.marker_pos[inside, 0] = self.box_pos[0] + c * local_x_i - s * local_z_i
        self.marker_pos[inside, 2] = self.box_pos[2] + s * local_x_i + c * local_z_i

    # ----------------------------------------------------------------------
    # Controls
    # ----------------------------------------------------------------------
    def _drive_acceleration(self, t: float) -> float:
        if t < self.drive_start:
            return 0.0
        if t < self.drive_end:
            ramp = min((t - self.drive_start) / self.drive_ramp, 1.0)
            return self.drive_mag * ramp
        return self.drive_mag

    # ----------------------------------------------------------------------
    # Budgets
    # ----------------------------------------------------------------------
    def _update_budgets(self, dt: float, F_grav_box: np.ndarray,
                        F_pressure_box: np.ndarray, F_wall_box: np.ndarray,
                        a_drive: float) -> None:
        total_fluid_mass = float(self.n_markers * self.marker_mass[0])
        F_grav_fluid = total_fluid_mass * self.gravity
        F_drive_fluid = total_fluid_mass * self.drive_dir * a_drive
        F_wall_fluid = np.zeros(3, dtype=float)

        # external force tensor [G, K, 3]
        forces = np.zeros((3, 3, 3), dtype=float)
        # global
        forces[0, 0] = F_grav_box + F_grav_fluid
        forces[0, 1] = F_drive_fluid
        forces[0, 2] = F_wall_box + F_wall_fluid
        # box
        forces[1, 0] = F_grav_box
        forces[1, 1] = F_pressure_box
        forces[1, 2] = F_wall_box
        # fluid
        forces[2, 0] = F_grav_fluid + F_drive_fluid
        forces[2, 1] = -F_pressure_box
        forces[2, 2] = F_wall_fluid
        self.current_external_force = forces

        # cumulative external impulse (sum of external forces)
        F_box = F_grav_box + F_pressure_box + F_wall_box
        F_fluid = F_grav_fluid + F_drive_fluid + (-F_pressure_box) + F_wall_fluid
        F_global = F_box + F_fluid
        self.cum_external_impulse[0] += F_global * dt
        self.cum_external_impulse[1] += F_box * dt
        self.cum_external_impulse[2] += F_fluid * dt

        # interaction force (fluid -> box)
        self.current_interaction_force[0] = F_pressure_box

    def _boundary_samples(self):
        """Return boundary normals and velocities for wall + box samples."""
        n_wall = 100
        n_box = 80
        B = 3 * n_wall + n_box
        normal = np.zeros((B, 3), dtype=float)
        target = np.zeros((B, 3), dtype=float)
        velocity = np.zeros((B, 3), dtype=float)

        # left wall
        z = np.linspace(self.zmin, self.zmax, n_wall)
        normal[0:n_wall, 0] = 1.0
        # right wall
        normal[n_wall:2 * n_wall, 0] = -1.0
        # floor
        x = np.linspace(self.xmin, self.xmax, n_wall)
        normal[2 * n_wall:3 * n_wall, 2] = 1.0

        # Box surface samples
        theta = np.linspace(0, 2 * np.pi, n_box, endpoint=False)
        a = self.box_half[0]
        b = self.box_half[2]
        # parameterise perimeter piecewise; here just use a rounded-rect approx
        # Simpler: sample the four edges
        n_per_edge = n_box // 4
        local_x = np.zeros(n_box)
        local_z = np.zeros(n_box)
        # top edge
        t = np.linspace(-a, a, n_per_edge, endpoint=False)
        local_x[0:n_per_edge] = t
        local_z[0:n_per_edge] = b
        # right edge
        t = np.linspace(b, -b, n_per_edge, endpoint=False)
        local_x[n_per_edge:2 * n_per_edge] = a
        local_z[n_per_edge:2 * n_per_edge] = t
        # bottom edge
        t = np.linspace(a, -a, n_per_edge, endpoint=False)
        local_x[2 * n_per_edge:3 * n_per_edge] = t
        local_z[2 * n_per_edge:3 * n_per_edge] = -b
        # left edge
        t = np.linspace(-b, b, n_box - 3 * n_per_edge, endpoint=False)
        local_x[3 * n_per_edge:] = -a
        local_z[3 * n_per_edge:] = t

        c = np.cos(self.box_angle)
        s = np.sin(self.box_angle)
        x_box = self.box_pos[0] + c * local_x - s * local_z
        z_box = self.box_pos[2] + s * local_x + c * local_z

        # normals: rotate local outward normals
        # top: (0,1), right: (1,0), bottom: (0,-1), left: (-1,0) in local (x,z)
        n_local_x = np.zeros(n_box)
        n_local_z = np.zeros(n_box)
        n_local_x[0:n_per_edge] = 0.0
        n_local_z[0:n_per_edge] = 1.0
        n_local_x[n_per_edge:2 * n_per_edge] = 1.0
        n_local_z[n_per_edge:2 * n_per_edge] = 0.0
        n_local_x[2 * n_per_edge:3 * n_per_edge] = 0.0
        n_local_z[2 * n_per_edge:3 * n_per_edge] = -1.0
        n_local_x[3 * n_per_edge:] = -1.0
        n_local_z[3 * n_per_edge:] = 0.0

        n_world_x = c * n_local_x - s * n_local_z
        n_world_z = s * n_local_x + c * n_local_z

        start = 3 * n_wall
        normal[start:start + n_box, 0] = n_world_x
        normal[start:start + n_box, 2] = n_world_z

        # target / actual velocities at box surface
        u_surf, w_surf = self._box_velocity_at(x_box, z_box)
        velocity[start:start + n_box, 0] = u_surf
        velocity[start:start + n_box, 2] = w_surf
        target[start:start + n_box, 0] = u_surf
        target[start:start + n_box, 2] = w_surf

        return normal, target, velocity

    # ----------------------------------------------------------------------
    # Frame assembly
    # ----------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        n = 1 + self.n_markers
        pos = np.zeros((n, 3), dtype=float)
        vel = np.zeros((n, 3), dtype=float)
        ori = np.zeros((n, 4), dtype=float)
        ang = np.zeros((n, 3), dtype=float)
        mass = np.zeros(n, dtype=float)

        pos[0] = self.box_pos
        vel[0] = self.box_vel
        qy = np.sin(self.box_angle / 2.0)
        qw = np.cos(self.box_angle / 2.0)
        ori[0] = [0.0, qy, 0.0, qw]
        ang[0, 1] = self.box_omega
        mass[0] = self.box_mass

        pos[1:] = self.marker_pos
        vel[1:] = self.marker_vel
        ori[1:] = [0.0, 0.0, 0.0, 1.0]
        mass[1:] = self.marker_mass

        phase = np.ones(n, dtype=float)
        phase[0] = 0.0

        # mass_quantity
        mq = np.zeros((n, 2), dtype=float)
        mq[0, 0] = self.box_mass
        mq[1:, 1] = self.marker_mass

        # fluid velocity field on grid
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        w_c = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        fluid_vel = np.zeros((self.nz, self.nx, 3), dtype=float)
        fluid_vel[:, :, 0] = u_c.T
        fluid_vel[:, :, 2] = w_c.T

        # boundary samples
        bn, btg, bv = self._boundary_samples()

        # domain weights [G, N]
        weights = np.zeros((3, n), dtype=float)
        weights[0, :] = 1.0  # global
        weights[1, 0] = 1.0  # box
        weights[2, 1:] = 1.0  # fluid

        attributes = {
            "phase": phase,
            "mass_quantity": mq,
        }
        fields = {
            "fluid_velocity": fluid_vel,
            "boundary_normal": bn,
            "boundary_target_velocity": btg,
            "boundary_velocity": bv,
            "budget_cumulative_external_impulse": self.cum_external_impulse,
            "budget_cumulative_inflow_momentum": np.zeros((3, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3), dtype=float),
            "budget_domain_weights": weights,
            "budget_external_force": self.current_external_force,
            "cumulative_mass_source": np.zeros(2, dtype=float),
            "interaction_force": self.current_interaction_force,
        }
        scalars = {
            "fluid_drive": float(self._drive_acceleration(self.t)),
            "reference_momentum": float(self.box_mass),
            "reference_velocity": 1.0,
        }

        return StateFrame(
            time=self.t,
            position=pos,
            velocity=vel,
            orientation=ori,
            angular_velocity=ang,
            mass=mass,
            body_id=self.scene.initial_state().body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
