"""3D incompressible free-surface liquid stirred by a rotating blade.

The solver evolves a cell-centered velocity field and a liquid level set on a
uniform diagnostic grid using a projection-method Navier-Stokes integrator:
  1. Semi-Lagrangian advection of velocity and level set.
  2. Gravity and blade momentum forcing.
  3. Pressure projection with free-surface (p=0) and moving/free-slip wall
     boundary conditions.

Required StateFrame fields are reconstructed from the grid each sample time.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Grid-based free-surface liquid solver for the rotating-blade scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        self.state0 = scene.initial_state()
        self.time = 0.0

        domain = self.config["domain"]
        self.origin = np.asarray(domain["bounds_min"], dtype=float)
        self.L = np.asarray(domain["bounds_max"], dtype=float) - self.origin
        self.nx, self.ny, self.nz = [int(x) for x in domain["reference_grid"]]
        self.dx = float(domain["grid_spacing"])
        self.dt = float(self.config["sample_dt"])
        self.rho = float(self.config["liquid"]["density"])
        self.nu = float(self.config["liquid"]["kinematic_viscosity"])
        self.g = np.asarray(self.config["gravity"], dtype=float)
        self.half = float(self.config["tank"]["interior_half_width"])
        self.depth = float(self.config["liquid"]["depth"])

        # Cell-center coordinates.
        self.xc = self.origin[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.origin[1] + (np.arange(self.ny) + 0.5) * self.dx
        self.zc = self.origin[2] + (np.arange(self.nz) + 0.5) * self.dx
        self.X, self.Y, self.Z = np.meshgrid(
            self.xc, self.yc, self.zc, indexing="ij"
        )
        self.cell_pos = np.stack([self.X, self.Y, self.Z], axis=-1)

        # Velocity and level set.
        self.vel = np.zeros((self.nx, self.ny, self.nz, 3))
        self.phi = self.state0.fields["liquid_level_set"].copy()

        # Fixed particle/sample data for the StateFrame contract.
        self.position = self.state0.position.copy()
        self.orientation = self.state0.orientation.copy()
        self.angular_vel = self.state0.angular_velocity.copy()
        self.mass = self.state0.mass.copy()
        self.body_id = self.state0.body_id.copy()
        self.attributes = {
            k: (v.copy() if hasattr(v, "copy") else v)
            for k, v in self.state0.attributes.items()
        }
        self.scalars = self.state0.scalars.copy()

    def advance_to(self, target_time: float):
        if target_time < self.time:
            # Rewind to the initial state if ever asked for an earlier time.
            self.time = 0.0
            self.vel = np.zeros((self.nx, self.ny, self.nz, 3))
            self.phi = self.state0.fields["liquid_level_set"].copy()

        while self.time < target_time - 1e-12:
            step_dt = min(self.dt, target_time - self.time)
            self._step(step_dt)
            self.time += step_dt

        return self._make_frame(target_time)

    def diagnostics(self) -> dict[str, object]:
        return dict(self.scalars)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _trilinear_sample(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Sample a scalar 3-D field at world points using trilinear interpolation."""
        nx, ny, nz = field.shape
        # Continuous cell index; cell center at origin + (i+0.5)*dx.
        xi = (points[:, 0] - self.origin[0]) / self.dx - 0.5
        yi = (points[:, 1] - self.origin[1]) / self.dx - 0.5
        zi = (points[:, 2] - self.origin[2]) / self.dx - 0.5

        xi = np.clip(xi, 0.0, nx - 1.0001)
        yi = np.clip(yi, 0.0, ny - 1.0001)
        zi = np.clip(zi, 0.0, nz - 1.0001)

        i0 = np.floor(xi).astype(int)
        j0 = np.floor(yi).astype(int)
        k0 = np.floor(zi).astype(int)
        tx = xi - i0
        ty = yi - j0
        tz = zi - k0

        def get(ii, jj, kk):
            ii = np.clip(ii, 0, nx - 1)
            jj = np.clip(jj, 0, ny - 1)
            kk = np.clip(kk, 0, nz - 1)
            return field[ii, jj, kk]

        c000 = get(i0, j0, k0)
        c100 = get(i0 + 1, j0, k0)
        c010 = get(i0, j0 + 1, k0)
        c110 = get(i0 + 1, j0 + 1, k0)
        c001 = get(i0, j0, k0 + 1)
        c101 = get(i0 + 1, j0, k0 + 1)
        c011 = get(i0, j0 + 1, k0 + 1)
        c111 = get(i0 + 1, j0 + 1, k0 + 1)

        return (
            (1 - tx) * (1 - ty) * (1 - tz) * c000
            + tx * (1 - ty) * (1 - tz) * c100
            + (1 - tx) * ty * (1 - tz) * c010
            + tx * ty * (1 - tz) * c110
            + (1 - tx) * (1 - ty) * tz * c001
            + tx * (1 - ty) * tz * c101
            + (1 - tx) * ty * tz * c011
            + tx * ty * tz * c111
        )

    def _blade_velocity(self, points: np.ndarray, time: float) -> np.ndarray:
        """Rigid-body velocity omega(t) cross (r - c) for an array of points."""
        pose = self.scene.blade_kinematics(time)
        omega = pose["angular_velocity"]
        center = pose["position"]
        rel = points - center
        return np.cross(omega, rel)

    # ------------------------------------------------------------------ #
    # Time step
    # ------------------------------------------------------------------ #

    def _step(self, dt: float) -> None:
        pose = self.scene.blade_kinematics(self.time)
        omega = pose["angular_velocity"]
        center = pose["position"]

        # Signed distance and rigid velocity at cell centers.
        blade_sdf = self.scene.blade_signed_distance(
            self.cell_pos.reshape(-1, 3), self.time
        ).reshape(self.nx, self.ny, self.nz)
        v_blade_center = np.cross(omega, self.cell_pos - center)

        # 1. Advect velocity.
        self.vel = self._advect_velocity(dt, blade_sdf, v_blade_center)

        # 2. Gravity.
        self.vel[..., 2] += self.g[2] * dt

        # 3. Blade momentum coupling near the immersed boundary.
        dist = np.maximum(blade_sdf, 0.0)
        weight = np.exp(-dist / self.dx)
        force = weight[..., None] * (v_blade_center - self.vel)
        self.vel += 5.0 * dt * force

        # 4. Pressure projection.
        self.vel = self._pressure_projection(dt, blade_sdf)

        # 5. Advect level set.
        self._advect_phi(dt)

    def _advect_velocity(
        self, dt: float, blade_sdf: np.ndarray, v_blade_center: np.ndarray
    ) -> np.ndarray:
        """Semi-Lagrangian advection of the cell-centered velocity field."""
        back_pos = self.cell_pos - dt * self.vel
        lo = self.origin + 1e-6
        hi = self.origin + self.L - 1e-6
        back_pos = np.clip(back_pos, lo, hi)

        # If a parcel came from inside the blade, it carries the blade velocity.
        sdf_back = self.scene.blade_signed_distance(
            back_pos.reshape(-1, 3), self.time
        ).reshape(self.nx, self.ny, self.nz)
        inside = sdf_back < 0.0

        new_vel = np.zeros_like(self.vel)
        new_vel[inside] = v_blade_center[inside]

        outside = ~inside
        if np.any(outside):
            pts = back_pos[outside].reshape(-1, 3)
            for d in range(3):
                new_vel[..., d][outside] = self._trilinear_sample(
                    self.vel[..., d], pts
                )
        return new_vel

    def _advect_phi(self, dt: float) -> None:
        back_pos = self.cell_pos - dt * self.vel
        lo = self.origin + 1e-6
        hi = self.origin + self.L - 1e-6
        back_pos = np.clip(back_pos, lo, hi)
        self.phi = self._trilinear_sample(
            self.phi, back_pos.reshape(-1, 3)
        ).reshape(self.nx, self.ny, self.nz)

    def _pressure_projection(self, dt: float, blade_sdf: np.ndarray) -> np.ndarray:
        """Project the cell-centered velocity onto the divergence-free space."""
        nx, ny, nz = self.nx, self.ny, self.nz
        solid_mask = blade_sdf < 0.0

        # Determine liquid cells and surface cells (liquid adjacent to air).
        phi_pad = np.zeros((nx + 2, ny + 2, nz + 2))
        phi_pad[1:-1, 1:-1, 1:-1] = self.phi
        solid_pad = np.zeros((nx + 2, ny + 2, nz + 2), dtype=bool)
        solid_pad[1:-1, 1:-1, 1:-1] = solid_mask

        liquid = (phi_pad[1:-1, 1:-1, 1:-1] < 0) & ~solid_pad[
            1:-1, 1:-1, 1:-1
        ]
        air_neighbor = (
            ((phi_pad[2:, 1:-1, 1:-1] >= 0) & ~solid_pad[2:, 1:-1, 1:-1])
            | ((phi_pad[:-2, 1:-1, 1:-1] >= 0) & ~solid_pad[:-2, 1:-1, 1:-1])
            | ((phi_pad[1:-1, 2:, 1:-1] >= 0) & ~solid_pad[1:-1, 2:, 1:-1])
            | ((phi_pad[1:-1, :-2, 1:-1] >= 0) & ~solid_pad[1:-1, :-2, 1:-1])
            | ((phi_pad[1:-1, 1:-1, 2:] >= 0) & ~solid_pad[1:-1, 1:-1, 2:])
            | ((phi_pad[1:-1, 1:-1, :-2] >= 0) & ~solid_pad[1:-1, 1:-1, :-2])
        )
        surface = liquid & air_neighbor
        interior = liquid & (~surface)

        # Build face velocities for divergence.
        face_u, face_v, face_w = self._build_face_velocities(
            blade_sdf, solid_mask
        )

        # Divergence.
        div = (
            face_u[1:, :, :] - face_u[:-1, :, :]
            + face_v[:, 1:, :] - face_v[:, :-1, :]
            + face_w[:, :, 1:] - face_w[:, :, :-1]
        ) / self.dx

        # Solve the Poisson equation for interior liquid cells.
        p = self._solve_pressure(interior, surface, solid_mask, div, dt)

        # Update cell-centered velocity: u -= dt/rho * grad(p).
        vel_new = self.vel.copy()
        grad_p = self._pressure_gradient(p, solid_mask)
        # Only update non-solid cells.
        vel_new[~solid_mask] -= (dt / self.rho) * grad_p[~solid_mask]

        # Solid cells move with the blade.
        if np.any(solid_mask):
            pose = self.scene.blade_kinematics(self.time)
            omega = pose["angular_velocity"]
            center = pose["position"]
            v_blade = np.cross(omega, self.cell_pos - center)
            vel_new[solid_mask] = v_blade[solid_mask]

        return vel_new

    def _build_face_velocities(
        self, blade_sdf: np.ndarray, solid_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct face-normal velocities with wall/blade boundary data."""
        nx, ny, nz = self.nx, self.ny, self.nz
        phi = self.phi
        vel = self.vel

        # Face-center arrays and SDF/velocity for all three face sets.
        # u-faces: (nx+1, ny, nz)
        xu = self.origin[0] + np.arange(nx + 1) * self.dx
        Xu, Yu, Zu = np.meshgrid(xu, self.yc, self.zc, indexing="ij")
        u_centers = np.stack([Xu, Yu, Zu], axis=-1).reshape(-1, 3)

        # v-faces: (nx, ny+1, nz)
        yv = self.origin[1] + np.arange(ny + 1) * self.dx
        Xv, Yv, Zv = np.meshgrid(self.xc, yv, self.zc, indexing="ij")
        v_centers = np.stack([Xv, Yv, Zv], axis=-1).reshape(-1, 3)

        # w-faces: (nx, ny, nz+1)
        zw = self.origin[2] + np.arange(nz + 1) * self.dx
        Xw, Yw, Zw = np.meshgrid(self.xc, self.yc, zw, indexing="ij")
        w_centers = np.stack([Xw, Yw, Zw], axis=-1).reshape(-1, 3)

        all_centers = np.vstack([u_centers, v_centers, w_centers])
        all_sdf = self.scene.blade_signed_distance(all_centers, self.time)
        all_v = self._blade_velocity(all_centers, self.time)

        n_u = (nx + 1) * ny * nz
        n_v = nx * (ny + 1) * nz
        u_sdf = all_sdf[:n_u].reshape(nx + 1, ny, nz)
        v_sdf = all_sdf[n_u : n_u + n_v].reshape(nx, ny + 1, nz)
        w_sdf = all_sdf[n_u + n_v :].reshape(nx, ny, nz + 1)
        u_v = all_v[:n_u].reshape(nx + 1, ny, nz, 3)
        v_v = all_v[n_u : n_u + n_v].reshape(nx, ny + 1, nz, 3)
        w_v = all_v[n_u + n_v :].reshape(nx, ny, nz + 1, 3)

        face_u = np.zeros((nx + 1, ny, nz))
        face_v = np.zeros((nx, ny + 1, nz))
        face_w = np.zeros((nx, ny, nz + 1))

        # x-faces.
        for i in range(1, nx):
            if u_sdf[i, 0, 0] < 0:
                face_u[i] = u_v[i, :, :, 0]
            else:
                left_liquid = (phi[i - 1] < 0) & ~solid_mask[i - 1]
                right_liquid = (phi[i] < 0) & ~solid_mask[i]
                both = left_liquid & right_liquid
                left_only = left_liquid & ~right_liquid
                right_only = right_liquid & ~left_liquid
                face_u[i] = (
                    both * 0.5 * (vel[i - 1, :, :, 0] + vel[i, :, :, 0])
                    + left_only * vel[i - 1, :, :, 0]
                    + right_only * vel[i, :, :, 0]
                )
        # Wall boundaries i=0, nx: normal velocity zero.
        face_u[0] = 0.0
        face_u[nx] = 0.0

        # y-faces.
        for j in range(1, ny):
            if v_sdf[0, j, 0] < 0:
                face_v[:, j, :] = v_v[:, j, :, 1]
            else:
                left_liquid = (phi[:, j - 1, :] < 0) & ~solid_mask[:, j - 1, :]
                right_liquid = (phi[:, j, :] < 0) & ~solid_mask[:, j, :]
                both = left_liquid & right_liquid
                left_only = left_liquid & ~right_liquid
                right_only = right_liquid & ~left_liquid
                face_v[:, j, :] = (
                    both * 0.5 * (vel[:, j - 1, :, 1] + vel[:, j, :, 1])
                    + left_only * vel[:, j - 1, :, 1]
                    + right_only * vel[:, j, :, 1]
                )
        face_v[:, 0, :] = 0.0
        face_v[:, ny, :] = 0.0

        # z-faces.
        for k in range(1, nz):
            if w_sdf[0, 0, k] < 0:
                face_w[:, :, k] = w_v[:, :, k, 2]
            else:
                left_liquid = (phi[:, :, k - 1] < 0) & ~solid_mask[:, :, k - 1]
                right_liquid = (phi[:, :, k] < 0) & ~solid_mask[:, :, k]
                both = left_liquid & right_liquid
                left_only = left_liquid & ~right_liquid
                right_only = right_liquid & ~left_liquid
                face_w[:, :, k] = (
                    both * 0.5 * (vel[:, :, k - 1, 2] + vel[:, :, k, 2])
                    + left_only * vel[:, :, k - 1, 2]
                    + right_only * vel[:, :, k, 2]
                )
        face_w[:, :, 0] = 0.0
        face_w[:, :, nz] = 0.0

        return face_u, face_v, face_w

    def _solve_pressure(
        self,
        interior: np.ndarray,
        surface: np.ndarray,
        solid_mask: np.ndarray,
        div: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Build and solve the pressure Poisson equation."""
        nx, ny, nz = self.nx, self.ny, self.nz
        idx = np.full(interior.shape, -1, dtype=int)
        idx[interior] = np.arange(int(interior.sum()))
        n_unknowns = int(interior.sum())

        p = np.zeros((nx, ny, nz))
        if n_unknowns == 0:
            return p

        scale = 1.0 / (self.dx * self.dx)
        data: list[float] = []
        rows: list[int] = []
        cols: list[int] = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not interior[i, j, k]:
                        continue
                    ii = idx[i, j, k]
                    diag = 0.0

                    def add_neighbor(ni, nj, nk):
                        nonlocal diag
                        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                            # Outside domain is treated as air (p=0); only
                            # contributes -scale to the diagonal.
                            diag -= scale
                            return
                        if solid_mask[ni, nj, nk]:
                            # Free-slip solid: Neumann, no contribution.
                            return
                        if surface[ni, nj, nk]:
                            # Free surface p=0.
                            diag -= scale
                            return
                        # Interior neighbor.
                        jj = idx[ni, nj, nk]
                        data.append(scale)
                        rows.append(ii)
                        cols.append(jj)
                        diag -= scale

                    add_neighbor(i + 1, j, k)
                    add_neighbor(i - 1, j, k)
                    add_neighbor(i, j + 1, k)
                    add_neighbor(i, j - 1, k)
                    add_neighbor(i, j, k + 1)
                    add_neighbor(i, j, k - 1)

                    data.append(diag)
                    rows.append(ii)
                    cols.append(ii)

        A = sp.coo_matrix(
            (data, (rows, cols)), shape=(n_unknowns, n_unknowns)
        ).tocsr()
        b = (self.rho / dt) * div[interior]

        try:
            p_vec = spla.spsolve(A, b)
        except Exception:
            # Fallback to a least-squares/minimal-residual solve.
            p_vec, _ = spla.cg(A, b, tol=1e-10, maxiter=n_unknowns)

        p[interior] = p_vec
        return p

    def _pressure_gradient(self, p: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
        """Cell-centered pressure gradient with boundary-aware ghost values."""
        nx, ny, nz = self.nx, self.ny, self.nz
        grad = np.zeros((nx, ny, nz, 3))

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if solid_mask[i, j, k]:
                        continue

                    def p_neighbor(ni, nj, nk):
                        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                            return 0.0
                        if solid_mask[ni, nj, nk]:
                            return p[i, j, k]
                        return p[ni, nj, nk]

                    p_plus = p_neighbor(i + 1, j, k)
                    p_minus = p_neighbor(i - 1, j, k)
                    grad[i, j, k, 0] = (p_plus - p_minus) / (2 * self.dx)

                    p_plus = p_neighbor(i, j + 1, k)
                    p_minus = p_neighbor(i, j - 1, k)
                    grad[i, j, k, 1] = (p_plus - p_minus) / (2 * self.dx)

                    p_plus = p_neighbor(i, j, k + 1)
                    p_minus = p_neighbor(i, j, k - 1)
                    grad[i, j, k, 2] = (p_plus - p_minus) / (2 * self.dx)

        return grad

    # ------------------------------------------------------------------ #
    # StateFrame construction
    # ------------------------------------------------------------------ #

    def _make_frame(self, time: float) -> StateFrame:
        pose = self.scene.blade_kinematics(time)
        omega = pose["angular_velocity"]
        center = pose["position"]

        # Blade SDF on the diagnostic grid.
        blade_sdf = self.scene.blade_signed_distance(
            self.cell_pos.reshape(-1, 3), time
        ).reshape(self.nx, self.ny, self.nz)

        # Diagnostic active mask mirrors the scaffold definition.
        h = self.dx
        active = (
            (self.phi < -h)
            & (self.Z > h)
            & (np.abs(self.X) < self.half - h)
            & (np.abs(self.Y) < self.half - h)
            & (blade_sdf > h)
        )

        # Wall samples (static).
        wall_pos = self.state0.fields["boundary_position"].copy()
        wall_normal = self.state0.fields["boundary_normal"].copy()
        if wall_pos.size > 0:
            wall_vel = self._interpolate_velocity(wall_pos)
        else:
            wall_vel = np.zeros_like(wall_pos)

        # Blade samples (current wetted patch).
        blade_samples = self.scene.blade_surface_samples(time)
        wetted = blade_samples["position"][:, 2] < self.depth
        # Also consider local free-surface height.
        if wetted.sum() > 0:
            sample_phi = self._trilinear_sample(
                self.phi, blade_samples["position"]
            )
            wetted = wetted & (sample_phi < 0.0)
        b_pos = blade_samples["position"][wetted]
        b_normal = blade_samples["normal"][wetted]
        b_target = blade_samples["target_velocity"][wetted]
        if b_pos.size > 0:
            b_vel = self._interpolate_velocity(b_pos)
        else:
            b_vel = np.zeros_like(b_target)

        # Particle velocities for the state binding.
        particle_vel = self._interpolate_velocity(self.position[:-1])
        full_vel = np.vstack([particle_vel, np.zeros((1, 3))])

        # Update kinematic blade marker.
        orientation = self.orientation.copy()
        angular = self.angular_vel.copy()
        orientation[-1] = pose["orientation"]
        angular[-1] = omega

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=full_vel,
            orientation=orientation,
            angular_velocity=angular,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.attributes.items()},
            fields={
                "velocity_field": self.vel.copy(),
                "active_mask": active.astype(float),
                "liquid_level_set": self.phi.copy(),
                "density_field": (self.phi < 0).astype(float) * self.rho,
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": wall_pos,
                "boundary_normal": wall_normal,
                "boundary_velocity": wall_vel,
                "boundary_target_velocity": np.zeros_like(wall_pos),
                "blade_boundary_position": b_pos,
                "blade_boundary_normal": b_normal,
                "blade_boundary_velocity": b_vel,
                "blade_boundary_target_velocity": b_target,
                "grid_origin": np.asarray(self.config["domain"]["bounds_min"], dtype=float),
            },
            scalars=self.scalars.copy(),
        )
        frame.validate()
        return frame

    def _interpolate_velocity(self, points: np.ndarray) -> np.ndarray:
        """Interpolate the cell-centered velocity to a set of world points."""
        if points.size == 0:
            return np.zeros((0, 3))
        out = np.zeros((points.shape[0], 3))
        for d in range(3):
            out[:, d] = self._trilinear_sample(self.vel[..., d], points)
        return out
