"""3-D incompressible smoke solver on a collocated grid.

Implements a simplified stable-fluid variant: semi-Lagrangian advection,
buoyancy, explicit diffusion, and pressure projection with solid-wall
penetration blocked.  The sphere obstacle and domain walls are respected
through a static active mask and zero-normal boundary enforcement.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import cg, splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.frame0 = scene.initial_state()
        self.config = getattr(scene, "config", {})

        # Domain grid taken from the initial StateFrame field shape.
        self._init_grid()
        self._init_state()
        self._build_pressure_system()
        self._build_boundary_samples()

        self.current_time = 0.0

    # ------------------------------------------------------------------
    # Grid / geometry helpers
    # ------------------------------------------------------------------
    def _init_grid(self) -> None:
        shape = tuple(self.frame0.fields.get("transported_scalar", np.zeros((1, 1, 1))).shape)
        if len(shape) != 3:
            shape = (24, 16, 28)
        self.nx, self.ny, self.nz = shape

        # Match the linspace used in scene.py.
        self.x = np.linspace(-3.0, 3.0, self.nx)
        self.y = np.linspace(-1.2, 1.2, self.ny)
        self.z = np.linspace(0.0, 3.8, self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # Sphere obstacle.
        self.sphere_center = np.array([0.0, 0.0, 1.35], float)
        self.sphere_radius = 0.48

        # Active = fluid cells outside the sphere.
        r = np.sqrt(
            (self.X - self.sphere_center[0]) ** 2
            + (self.Y - self.sphere_center[1]) ** 2
            + (self.Z - self.sphere_center[2]) ** 2
        )
        self.active_mask = r > self.sphere_radius

        # Emitter cylinder: radius 0.32, height 0.10, centred at z=0.05.
        self.source_rate = 1.6
        self.source_mask = (
            (self.Z >= 0.0)
            & (self.Z <= 0.10)
            & ((self.X ** 2 + self.Y ** 2) <= 0.32 ** 2)
        )

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        self.vel = np.asarray(self.frame0.fields.get("velocity_field", np.zeros((self.nx, self.ny, self.nz, 3))), float).copy()
        if self.vel.shape != (self.nx, self.ny, self.nz, 3):
            self.vel = np.zeros((self.nx, self.ny, self.nz, 3))

        self.scalar = np.asarray(self.frame0.fields.get("transported_scalar", np.zeros((self.nx, self.ny, self.nz))), float).copy()
        if self.scalar.shape != (self.nx, self.ny, self.nz):
            self.scalar = np.zeros((self.nx, self.ny, self.nz))

        # Keep particle arrays from the frozen initial state.
        self.position = np.asarray(self.frame0.position, float).copy()
        self.velocity = np.asarray(self.frame0.velocity, float).copy()
        self.mass = np.asarray(self.frame0.mass, float).copy() if self.frame0.mass is not None else None
        self.body_id = np.asarray(self.frame0.body_id, int).copy() if self.frame0.body_id is not None else None

        self.attributes = {k: np.asarray(v).copy() for k, v in self.frame0.attributes.items()}
        self.scalars = {k: float(v) for k, v in self.frame0.scalars.items()}

        # Ensure required scalar keys are present.
        self.scalars.setdefault("grid_spacing", float(self.config.get("sampling", {}).get("spacing", 0.18)))
        self.scalars.setdefault("characteristic_length", 6.0)
        self.scalars.setdefault("reference_velocity", 1.0)

        # Scalar source rate field.
        self.scalar_source = np.zeros((self.nx, self.ny, self.nz), float)
        self.scalar_source[self.source_mask] = self.source_rate

    # ------------------------------------------------------------------
    # Pressure system (collocated 7-point Laplacian, Neumann, one pin)
    # ------------------------------------------------------------------
    def _build_pressure_system(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        n = nx * ny * nz
        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        inv_dz2 = 1.0 / (self.dz * self.dz)

        diagonals = np.zeros(n, float)
        rows = []
        cols = []
        vals = []

        def idx(i, j, k):
            return i + nx * (j + ny * k)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    r = idx(i, j, k)
                    d = 0.0

                    # x neighbours
                    if nx > 1:
                        if i == 0 or i == nx - 1:
                            d += -inv_dx2
                            if i == 0:
                                rows.append(r); cols.append(r + 1); vals.append(inv_dx2)
                            else:
                                rows.append(r); cols.append(r - 1); vals.append(inv_dx2)
                        else:
                            d += -2.0 * inv_dx2
                            rows.append(r); cols.append(r + 1); vals.append(inv_dx2)
                            rows.append(r); cols.append(r - 1); vals.append(inv_dx2)

                    # y neighbours
                    if ny > 1:
                        if j == 0 or j == ny - 1:
                            d += -inv_dy2
                            if j == 0:
                                rows.append(r); cols.append(r + nx); vals.append(inv_dy2)
                            else:
                                rows.append(r); cols.append(r - nx); vals.append(inv_dy2)
                        else:
                            d += -2.0 * inv_dy2
                            rows.append(r); cols.append(r + nx); vals.append(inv_dy2)
                            rows.append(r); cols.append(r - nx); vals.append(inv_dy2)

                    # z neighbours
                    if nz > 1:
                        if k == 0 or k == nz - 1:
                            d += -inv_dz2
                            if k == 0:
                                rows.append(r); cols.append(r + nx * ny); vals.append(inv_dz2)
                            else:
                                rows.append(r); cols.append(r - nx * ny); vals.append(inv_dz2)
                        else:
                            d += -2.0 * inv_dz2
                            rows.append(r); cols.append(r + nx * ny); vals.append(inv_dz2)
                            rows.append(r); cols.append(r - nx * ny); vals.append(inv_dz2)

                    diagonals[r] = d

        self._pressure_diag = diagonals
        self._pressure_mat = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n, n)
        ) + sparse.diags(diagonals, 0, shape=(n, n), format="csr")

        # Pin the first cell to remove the constant nullspace.
        self._pressure_mat = self._pressure_mat.tolil()
        self._pressure_mat[0, :] = 0.0
        self._pressure_mat[0, 0] = 1.0
        self._pressure_mat = self._pressure_mat.tocsr()

        try:
            self._pressure_lu = splu(self._pressure_mat)
        except Exception:
            self._pressure_lu = None

    # ------------------------------------------------------------------
    # Boundary samples on the sphere and domain walls
    # ------------------------------------------------------------------
    def _build_boundary_samples(self) -> None:
        samples = []
        normals = []

        # Sphere surface.
        n_phi = 16
        n_theta = 8
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        theta = np.linspace(0.0, np.pi, n_theta, endpoint=True)
        phi, theta = np.meshgrid(phi, theta, indexing="ij")
        phi = phi.ravel()
        theta = theta.ravel()
        n = phi.shape[0]
        cx, cy, cz = self.sphere_center
        r = self.sphere_radius
        sp = np.sin(theta)
        cp = np.cos(theta)
        sx = cx + r * sp * np.cos(phi)
        sy = cy + r * sp * np.sin(phi)
        sz = cz + r * cp
        samples.append(np.column_stack([sx, sy, sz]))
        normals.append(np.column_stack([sp * np.cos(phi), sp * np.sin(phi), cp]))

        # Domain wall samples (ground + sides + top).
        # Six faces, sparse regular sampling.
        face_points = []
        face_normals = []
        xb, yb, zb = self.x, self.y, self.z
        # x = min / max
        for val, sgn in [(xb[0], -1.0), (xb[-1], 1.0)]:
            yv, zv = np.meshgrid(yb, zb, indexing="ij")
            face_points.append(np.column_stack([np.full(yv.size, val), yv.ravel(), zv.ravel()]))
            face_normals.append(np.column_stack([np.full(yv.size, sgn), np.zeros(yv.size), np.zeros(yv.size)]))
        # y = min / max
        for val, sgn in [(yb[0], -1.0), (yb[-1], 1.0)]:
            xv, zv = np.meshgrid(xb, zb, indexing="ij")
            face_points.append(np.column_stack([xv.ravel(), np.full(xv.size, val), zv.ravel()]))
            face_normals.append(np.column_stack([np.zeros(xv.size), np.full(xv.size, sgn), np.zeros(xv.size)]))
        # z = min / max
        for val, sgn in [(zb[0], -1.0), (zb[-1], 1.0)]:
            xv, yv = np.meshgrid(xb, yb, indexing="ij")
            face_points.append(np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, val)]))
            face_normals.append(np.column_stack([np.zeros(xv.size), np.zeros(xv.size), np.full(xv.size, sgn)]))

        samples.append(np.concatenate(face_points, axis=0))
        normals.append(np.concatenate(face_normals, axis=0))

        self.boundary_samples = np.concatenate(samples, axis=0)
        self.boundary_normals = np.concatenate(normals, axis=0)
        # normalise
        nn = np.linalg.norm(self.boundary_normals, axis=1, keepdims=True)
        nn = np.where(nn == 0, 1.0, nn)
        self.boundary_normals /= nn

        b = self.boundary_samples.shape[0]
        self.boundary_target_velocity = np.zeros((b, 3), float)
        self.boundary_velocity = np.zeros((b, 3), float)

    # ------------------------------------------------------------------
    # Interpolation helpers (semi-Lagrangian)
    # ------------------------------------------------------------------
    def _phys_to_index(self, coords):
        """Convert physical coordinates to array indices for map_coordinates."""
        ix = (coords[..., 0] - self.x[0]) / self.dx
        iy = (coords[..., 1] - self.y[0]) / self.dy
        iz = (coords[..., 2] - self.z[0]) / self.dz
        return np.stack([ix, iy, iz], axis=-1)

    def _advect_field(self, field, vel, dt, cval=0.0):
        """Semi-Lagrangian advection of a scalar or 3-D vector component."""
        if field.ndim == 3:
            shape = field.shape
            coords = np.zeros(shape + (3,))
            coords[..., 0] = np.arange(self.nx)[:, None, None]
            coords[..., 1] = np.arange(self.ny)[None, :, None]
            coords[..., 2] = np.arange(self.nz)[None, None, :]
            phys = np.stack([
                self.x[coords[..., 0].astype(int)],
                self.y[coords[..., 1].astype(int)],
                self.z[coords[..., 2].astype(int)],
            ], axis=-1)
            back = phys - dt * vel
            back_idx = self._phys_to_index(back)
            return map_coordinates(field, [back_idx[..., 0], back_idx[..., 1], back_idx[..., 2]],
                                   order=1, mode="constant", cval=cval)
        elif field.ndim == 4:
            out = np.empty_like(field)
            for comp in range(field.shape[-1]):
                out[..., comp] = self._advect_field(field[..., comp], vel, dt, cval=cval)
            return out
        return field

    # ------------------------------------------------------------------
    # Finite-difference helpers
    # ------------------------------------------------------------------
    def _divergence(self, vel):
        ux = vel[..., 0]
        uy = vel[..., 1]
        uz = vel[..., 2]

        div = np.zeros_like(ux)
        if self.nx > 1:
            du = np.zeros_like(ux)
            du[1:-1, :, :] = (ux[2:, :, :] - ux[:-2, :, :]) / (2.0 * self.dx)
            du[0, :, :] = (ux[1, :, :] - ux[0, :, :]) / self.dx
            du[-1, :, :] = (ux[-1, :, :] - ux[-2, :, :]) / self.dx
            div += du
        if self.ny > 1:
            dv = np.zeros_like(uy)
            dv[:, 1:-1, :] = (uy[:, 2:, :] - uy[:, :-2, :]) / (2.0 * self.dy)
            dv[:, 0, :] = (uy[:, 1, :] - uy[:, 0, :]) / self.dy
            dv[:, -1, :] = (uy[:, -1, :] - uy[:, -2, :]) / self.dy
            div += dv
        if self.nz > 1:
            dw = np.zeros_like(uz)
            dw[:, :, 1:-1] = (uz[:, :, 2:] - uz[:, :, :-2]) / (2.0 * self.dz)
            dw[:, :, 0] = (uz[:, :, 1] - uz[:, :, 0]) / self.dz
            dw[:, :, -1] = (uz[:, :, -1] - uz[:, :, -2]) / self.dz
            div += dw
        return div

    def _gradient(self, p):
        grad = np.zeros((self.nx, self.ny, self.nz, 3), float)

        if self.nx > 1:
            gp = np.zeros_like(p)
            gp[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * self.dx)
            gp[0, :, :] = (p[1, :, :] - p[0, :, :]) / self.dx
            gp[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / self.dx
            grad[..., 0] = gp
        if self.ny > 1:
            gp = np.zeros_like(p)
            gp[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * self.dy)
            gp[:, 0, :] = (p[:, 1, :] - p[:, 0, :]) / self.dy
            gp[:, -1, :] = (p[:, -1, :] - p[:, -2, :]) / self.dy
            grad[..., 1] = gp
        if self.nz > 1:
            gp = np.zeros_like(p)
            gp[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * self.dz)
            gp[:, :, 0] = (p[:, :, 1] - p[:, :, 0]) / self.dz
            gp[:, :, -1] = (p[:, :, -1] - p[:, :, -2]) / self.dz
            grad[..., 2] = gp

        return grad

    def _laplacian_diffuse(self, q, dt, coeff):
        """Explicit Laplacian smoothing."""
        if coeff <= 0.0:
            return q
        out = q.copy()
        if self.nx > 1:
            out[1:-1, :, :] += dt * coeff * (q[2:, :, :] - 2.0 * q[1:-1, :, :] + q[:-2, :, :]) / (self.dx * self.dx)
        if self.ny > 1:
            out[:, 1:-1, :] += dt * coeff * (q[:, 2:, :] - 2.0 * q[:, 1:-1, :] + q[:, :-2, :]) / (self.dy * self.dy)
        if self.nz > 1:
            out[:, :, 1:-1] += dt * coeff * (q[:, :, 2:] - 2.0 * q[:, :, 1:-1] + q[:, :, :-2]) / (self.dz * self.dz)
        return out

    # ------------------------------------------------------------------
    # Projection step
    # ------------------------------------------------------------------
    def _project(self, vel, dt):
        div = self._divergence(vel)
        f = div.flatten() / dt

        if self._pressure_lu is not None:
            try:
                p_flat = self._pressure_lu.solve(f)
            except Exception:
                p_flat, _ = cg(self._pressure_mat, f, atol=1e-8, maxiter=500)
        else:
            p_flat, _ = cg(self._pressure_mat, f, atol=1e-8, maxiter=500)

        p = p_flat.reshape((self.nx, self.ny, self.nz))
        grad_p = self._gradient(p)
        return vel - dt * grad_p

    # ------------------------------------------------------------------
    # Boundary enforcement
    # ------------------------------------------------------------------
    def _apply_boundaries(self, vel):
        # Zero velocity inside the sphere and a small safety band.
        r = np.sqrt(
            (self.X - self.sphere_center[0]) ** 2
            + (self.Y - self.sphere_center[1]) ** 2
            + (self.Z - self.sphere_center[2]) ** 2
        )
        solid = r <= self.sphere_radius + 0.5 * min(self.dx, self.dy, self.dz)
        vel = np.where(solid[..., None], 0.0, vel)

        # No normal flow at domain walls.
        vel[0, :, :, 0] = 0.0
        vel[-1, :, :, 0] = 0.0
        vel[:, 0, :, 1] = 0.0
        vel[:, -1, :, 1] = 0.0
        vel[:, :, 0, 2] = 0.0
        vel[:, :, -1, 2] = 0.0
        return vel

    # ------------------------------------------------------------------
    # Single physics step
    # ------------------------------------------------------------------
    def _step(self, dt):
        # 1. Buoyancy (smoke rises opposite to gravity).
        buoyancy = 2.5
        self.vel[..., 2] += dt * buoyancy * self.scalar

        # 2. Advect velocity and scalar.
        self.vel = self._advect_field(self.vel, self.vel, dt, cval=0.0)
        self.scalar = self._advect_field(self.scalar, self.vel, dt, cval=0.0)

        # 3. Scalar source from the emitter cylinder.
        self.scalar += dt * self.scalar_source

        # 4. Diffuse.
        nu = 0.005
        kappa = 0.002
        self.vel = self._laplacian_diffuse(self.vel, dt, nu)
        self.scalar = self._laplacian_diffuse(self.scalar, dt, kappa)

        # 5. Remove smoke inside the solid obstacle.
        self.scalar = np.where(self.active_mask, self.scalar, 0.0)
        self.scalar = np.clip(self.scalar, 0.0, None)

        # 6. Pressure projection.
        self.vel = self._project(self.vel, dt)

        # 7. Enforce boundary conditions.
        self.vel = self._apply_boundaries(self.vel)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.current_time:
            return self._make_frame(self.current_time)

        dt = time - self.current_time
        # Use substeps for stability; target ~10 ms per substep.
        n_sub = max(1, int(np.ceil(dt / 0.0125)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step(sub_dt)

        self.current_time = time
        return self._make_frame(time)

    def _make_frame(self, time):
        return StateFrame(
            time=float(time),
            position=self.position,
            velocity=self.velocity,
            mass=self.mass,
            body_id=self.body_id,
            attributes=self.attributes,
            fields={
                "active_mask": self.active_mask.astype(bool),
                "boundary_normal": self.boundary_normals,
                "boundary_target_velocity": self.boundary_target_velocity,
                "boundary_velocity": self.boundary_velocity,
                "scalar_source": self.scalar_source,
                "transported_scalar": self.scalar,
                "velocity_field": self.vel,
            },
            scalars=self.scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "collocated_grid_smoke",
            "grid_shape": [self.nx, self.ny, self.nz],
            "sphere_center": self.sphere_center.tolist(),
            "sphere_radius": self.sphere_radius,
            "final_scalar_max": float(np.max(self.scalar)) if self.scalar is not None else None,
            "final_velocity_max": float(np.max(np.abs(self.vel))) if self.vel is not None else None,
        }
