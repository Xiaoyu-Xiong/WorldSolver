"""Smoke rising around a sphere: incompressible flow solver (self-implemented).

Physical model
--------------
Inviscid incompressible flow of a smoke/air mixture in a closed box, solved with a
classic MAC (staggered) grid fractional-step scheme:

    1. semi-Lagrangian advection of face velocities and of the smoke scalar
    2. buoyancy body force aligned with (minus) gravity, proportional to smoke density
    3. vorticity confinement body force (Fedkiw et al., "Visual Simulation of Smoke")
    4. no-flux boundary conditions on the domain walls and on the rigid sphere
    5. pressure projection: solve  -div(grad p) = -div(u*)/dt  and correct  u = u* - dt grad p

The sphere obstacle is voxelized as solid cells; every face touching a solid cell is a
closed (zero normal flux) face both in the velocity update and in the Poisson operator,
so smoke can never cross the sphere: it splits, accelerates along the sides and rolls up
into a wake above the obstacle.

Diagnostics are resampled (trilinear) onto the frozen diagnostic voxel grid declared by
``scene_geometry.yaml`` for the ``subject`` object.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _trilinear(field: np.ndarray, origin: np.ndarray, spacing: np.ndarray,
               points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a 3D sample array whose sample (0,0,0) sits at ``origin``."""
    n = np.asarray(field.shape, dtype=np.int64)
    g = (points - origin[None, :]) / spacing[None, :]
    g = np.clip(g, 0.0, (n - 1).astype(np.float64)[None, :])
    i0 = np.floor(g).astype(np.int64)
    i0 = np.minimum(i0, np.maximum(n - 2, 0)[None, :])
    frac = g - i0
    i1 = np.minimum(i0 + 1, (n - 1)[None, :])
    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    c00 = field[x0, y0, z0] * (1.0 - fz) + field[x0, y0, z1] * fz
    c01 = field[x0, y1, z0] * (1.0 - fz) + field[x0, y1, z1] * fz
    c10 = field[x1, y0, z0] * (1.0 - fz) + field[x1, y0, z1] * fz
    c11 = field[x1, y1, z0] * (1.0 - fz) + field[x1, y1, z1] * fz
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fx) + c1 * fx


def _points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))


def _fibonacci_sphere(count: int) -> np.ndarray:
    k = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack((np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)))


class BenchmarkSolver:
    """Self-implemented buoyant smoke solver (no external physics engine)."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(scene.config)
        self.config = config
        self.rng = np.random.default_rng(int(seed))
        self.gravity = np.asarray(config["gravity"], dtype=float)
        g_norm = float(np.linalg.norm(self.gravity))
        self.up = -self.gravity / g_norm if g_norm > 0 else np.array([0.0, 0.0, 1.0])

        sampling = config["sampling"]
        self.export_shape = tuple(int(v) for v in sampling["field_grid_shape"])
        self.export_spacing_scalar = float(sampling["spacing"])
        self.sample_dt = float(config["sample_dt"])

        geometry = config.get("scene_geometry", {}) or {}
        objects = {item["id"]: item for item in geometry.get("objects", [])}

        # frozen voxel-grid domain of the gas subject
        subject = objects.get("subject", {})
        sub_geo = subject.get("geometry", {})
        self.lo = np.asarray(sub_geo.get("bounds_min", [-3.0, -1.2, 0.0]), dtype=float)
        self.hi = np.asarray(sub_geo.get("bounds_max", [3.0, 1.2, 3.8]), dtype=float)

        # frozen sphere obstacle
        sphere = objects.get("obstacle_sphere", {})
        self.sphere_center = np.asarray(
            sphere.get("transform", {}).get("translation", [0.0, 0.0, 1.35]), dtype=float)
        self.sphere_radius = float(sphere.get("geometry", {}).get("radius", 0.48))

        # frozen emitter cylinder
        emitter = objects.get("smoke_source", {})
        self.src_center = np.asarray(
            emitter.get("transform", {}).get("translation", [0.0, 0.0, 0.05]), dtype=float)
        self.src_radius = float(emitter.get("geometry", {}).get("radius", 0.32))
        self.src_height = float(emitter.get("geometry", {}).get("height", 0.1))

        # ---------------- simulation grid (near isotropic, cell centred) ----------------
        self.shape = (45, 18, 28)
        nx, ny, nz = self.shape
        extent = self.hi - self.lo
        self.h = extent / np.asarray(self.shape, dtype=float)
        hx, hy, hz = self.h
        self.h_mean = float(self.h.mean())

        self.c_origin = self.lo + 0.5 * self.h
        self.xc = self.c_origin[0] + hx * np.arange(nx)
        self.yc = self.c_origin[1] + hy * np.arange(ny)
        self.zc = self.c_origin[2] + hz * np.arange(nz)

        self.u_origin = np.array([self.lo[0], self.c_origin[1], self.c_origin[2]])
        self.v_origin = np.array([self.c_origin[0], self.lo[1], self.c_origin[2]])
        self.w_origin = np.array([self.c_origin[0], self.c_origin[1], self.lo[2]])
        self.xf = self.lo[0] + hx * np.arange(nx + 1)
        self.yf = self.lo[1] + hy * np.arange(ny + 1)
        self.zf = self.lo[2] + hz * np.arange(nz + 1)

        self.u_pts = _points(self.xf, self.yc, self.zc)
        self.v_pts = _points(self.xc, self.yf, self.zc)
        self.w_pts = _points(self.xc, self.yc, self.zf)
        self.c_pts = _points(self.xc, self.yc, self.zc)

        # ---------------- solid voxelization ----------------
        rel = self.c_pts - self.sphere_center[None, :]
        inside = (np.linalg.norm(rel, axis=1) <= self.sphere_radius).reshape(self.shape)
        self.solid = inside
        self.fluid = ~self.solid

        self.u_open = np.zeros((nx + 1, ny, nz), dtype=bool)
        self.u_open[1:-1] = self.fluid[:-1] & self.fluid[1:]
        self.v_open = np.zeros((nx, ny + 1, nz), dtype=bool)
        self.v_open[:, 1:-1] = self.fluid[:, :-1] & self.fluid[:, 1:]
        self.w_open = np.zeros((nx, ny, nz + 1), dtype=bool)
        self.w_open[:, :, 1:-1] = self.fluid[:, :, :-1] & self.fluid[:, :, 1:]

        # ---------------- fields ----------------
        self.u = np.zeros((nx + 1, ny, nz))
        self.v = np.zeros((nx, ny + 1, nz))
        self.w = np.zeros((nx, ny, nz + 1))
        self.pressure = np.zeros(int(self.fluid.sum()))

        # smoke source rate (cylinder around the emitter axis)
        rad = np.linalg.norm(self.c_pts[:, :2] - self.src_center[None, :2], axis=1)
        zrel = self.c_pts[:, 2] - self.src_center[2]
        src = ((rad <= self.src_radius) &
               (np.abs(zrel) <= 0.5 * self.src_height + self.h[2])).reshape(self.shape)
        self.source_rate = 2.0  # smoke units per second
        self.src_mask_sim = src & self.fluid
        self.scalar_source_sim = np.where(self.src_mask_sim, self.source_rate, 0.0)

        # initial smoke: a buoyant puff sitting on the source, otherwise clear air
        xg = self.xc[:, None, None]
        yg = self.yc[None, :, None]
        zg = self.zc[None, None, :]
        # (same profile the frozen scene declares for its initial diagnostic field)
        self.smoke = 1.20 * np.exp(-((xg / 0.28) ** 2 + (yg / 0.28) ** 2)) \
            * np.exp(-(((zg - 0.72) / 0.78) ** 2))
        self.smoke = np.where(self.fluid, self.smoke, 0.0)

        # tiny deterministic seeded perturbation so the plume can go unstable
        noise = self.rng.normal(0.0, 1.0, self.w.shape)
        self.w += 0.01 * noise
        self.u += 0.01 * self.rng.normal(0.0, 1.0, self.u.shape)
        self._apply_solid_bc()

        # physical parameters
        self.buoyancy = 2.6            # m/s^2 per unit smoke
        self.confinement = 3.0         # vorticity confinement strength
        self.smoke_dissipation = 0.02  # 1/s
        self.velocity_damping = 0.15   # 1/s weak ambient drag (closed box)
        self.substeps = 2

        self._build_poisson()

        # ---------------- diagnostic export grid (frozen) ----------------
        ex, ey, ez = self.export_shape
        self.ex = np.linspace(self.lo[0], self.hi[0], ex)
        self.ey = np.linspace(self.lo[1], self.hi[1], ey)
        self.ez = np.linspace(self.lo[2], self.hi[2], ez)
        self.export_pts = _points(self.ex, self.ey, self.ez)
        e_rel = self.export_pts - self.sphere_center[None, :]
        e_solid = (np.linalg.norm(e_rel, axis=1) <= self.sphere_radius)
        self.export_active = (~e_solid).reshape(self.export_shape)
        e_rad = np.linalg.norm(self.export_pts[:, :2] - self.src_center[None, :2], axis=1)
        e_zrel = self.export_pts[:, 2] - self.src_center[2]
        e_src = (e_rad <= self.src_radius) & (np.abs(e_zrel) <= 0.5 * self.src_height)
        self.export_source = np.where(e_src & (~e_solid), self.source_rate, 0.0
                                      ).reshape(self.export_shape)

        # ---------------- boundary samples ----------------
        sphere_dirs = _fibonacci_sphere(160)
        sphere_pts = self.sphere_center[None, :] + self.sphere_radius * sphere_dirs
        wall_pts, wall_nrm = [], []
        for axis in range(3):
            a1, a2 = [k for k in range(3) if k != axis]
            s1 = np.linspace(self.lo[a1] + 0.2 * (self.hi[a1] - self.lo[a1]),
                             self.hi[a1] - 0.2 * (self.hi[a1] - self.lo[a1]), 3)
            s2 = np.linspace(self.lo[a2] + 0.2 * (self.hi[a2] - self.lo[a2]),
                             self.hi[a2] - 0.2 * (self.hi[a2] - self.lo[a2]), 3)
            g1, g2 = np.meshgrid(s1, s2, indexing="ij")
            for side, value in ((-1.0, self.lo[axis]), (1.0, self.hi[axis])):
                pts = np.zeros((g1.size, 3))
                pts[:, axis] = value
                pts[:, a1] = g1.ravel()
                pts[:, a2] = g2.ravel()
                nrm = np.zeros((g1.size, 3))
                nrm[:, axis] = side
                wall_pts.append(pts)
                wall_nrm.append(nrm)
        self.boundary_points = np.vstack([sphere_pts] + wall_pts)
        self.boundary_normals = np.vstack([sphere_dirs] + wall_nrm)
        self.boundary_target = np.zeros_like(self.boundary_points)  # static walls / sphere

        # ---------------- tracer particles bound to the sampled material ----------------
        spacing = self.export_spacing_scalar
        clouds = []
        for block in sampling["blocks"]:
            b_lo = np.asarray(block[:3], dtype=float)
            b_hi = np.asarray(block[3:], dtype=float)
            axes = [np.arange(b_lo[i] + 0.5 * spacing, b_hi[i], spacing) for i in range(3)]
            if any(len(a) == 0 for a in axes):
                clouds.append(((b_lo + b_hi) * 0.5)[None, :])
            else:
                clouds.append(_points(*axes))
        self.particles = np.concatenate(clouds)
        self.labels = np.concatenate(
            [np.full(len(c), i, np.int32) for i, c in enumerate(clouds)])
        density = float(sampling["density"])
        n_part = len(self.particles)
        self.particle_mass = np.full(n_part, density * spacing ** 3)
        self.particle_radius = np.full(n_part, 0.45 * spacing)
        self.particle_F = np.repeat(np.eye(3)[None], n_part, axis=0)
        self.particle_velocity = np.zeros_like(self.particles)

        # frozen normalization scalars
        self.characteristic_length = 2.0 * self.sphere_radius
        self.reference_velocity = 1.5

        self.time = 0.0
        self._steps = 0
        self._max_speed = 0.0
        self._cg_iterations = 0

    # ---------------------------------------------------------------- operators
    def _build_poisson(self) -> None:
        idx = np.full(self.shape, -1, dtype=np.int64)
        n_fluid = int(self.fluid.sum())
        idx[self.fluid] = np.arange(n_fluid)
        self.cell_index = idx
        self.n_fluid = n_fluid

        rows, cols, vals = [], [], []
        diag = np.zeros(n_fluid)
        couplings = (
            (self.u_open[1:-1], idx[:-1], idx[1:], 1.0 / self.h[0] ** 2),
            (self.v_open[:, 1:-1], idx[:, :-1], idx[:, 1:], 1.0 / self.h[1] ** 2),
            (self.w_open[:, :, 1:-1], idx[:, :, :-1], idx[:, :, 1:], 1.0 / self.h[2] ** 2),
        )
        for mask, ia, ib, coef in couplings:
            a = ia[mask]
            b = ib[mask]
            np.add.at(diag, a, coef)
            np.add.at(diag, b, coef)
            rows.append(a)
            cols.append(b)
            vals.append(np.full(a.size, -coef))
            rows.append(b)
            cols.append(a)
            vals.append(np.full(b.size, -coef))
        rows.append(np.arange(n_fluid))
        cols.append(np.arange(n_fluid))
        vals.append(diag)
        matrix = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_fluid, n_fluid)).tocsr()
        self.poisson = matrix
        safe = np.where(diag > 0.0, diag, 1.0)
        self.precond = spla.LinearOperator(
            (n_fluid, n_fluid), matvec=lambda r: r / safe, dtype=float)

    def _apply_solid_bc(self) -> None:
        self.u *= self.u_open
        self.v *= self.v_open
        self.w *= self.w_open

    def _velocity_at(self, points: np.ndarray) -> np.ndarray:
        return np.column_stack((
            _trilinear(self.u, self.u_origin, self.h, points),
            _trilinear(self.v, self.v_origin, self.h, points),
            _trilinear(self.w, self.w_origin, self.h, points),
        ))

    def _collocated(self) -> np.ndarray:
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        vel = np.stack((uc, vc, wc), axis=-1)
        vel[self.solid] = 0.0
        return vel

    def _advect(self, dt: float) -> None:
        back_u = self.u_pts - dt * self._velocity_at(self.u_pts)
        back_v = self.v_pts - dt * self._velocity_at(self.v_pts)
        back_w = self.w_pts - dt * self._velocity_at(self.w_pts)
        back_c = self.c_pts - dt * self._velocity_at(self.c_pts)

        new_u = _trilinear(self.u, self.u_origin, self.h, back_u).reshape(self.u.shape)
        new_v = _trilinear(self.v, self.v_origin, self.h, back_v).reshape(self.v.shape)
        new_w = _trilinear(self.w, self.w_origin, self.h, back_w).reshape(self.w.shape)
        new_s = _trilinear(self.smoke, self.c_origin, self.h, back_c).reshape(self.shape)

        self.u, self.v, self.w = new_u, new_v, new_w
        self.smoke = np.where(self.fluid, np.maximum(new_s, 0.0), 0.0)
        self._apply_solid_bc()

    def _confinement_force(self) -> np.ndarray:
        vel = self._collocated()
        hx, hy, hz = self.h
        du = np.gradient(vel[..., 0], hx, hy, hz, edge_order=1)
        dv = np.gradient(vel[..., 1], hx, hy, hz, edge_order=1)
        dw = np.gradient(vel[..., 2], hx, hy, hz, edge_order=1)
        omega = np.stack((dw[1] - dv[2], du[2] - dw[0], dv[0] - du[1]), axis=-1)
        mag = np.linalg.norm(omega, axis=-1)
        grad = np.stack(np.gradient(mag, hx, hy, hz, edge_order=1), axis=-1)
        norm = np.linalg.norm(grad, axis=-1)[..., None]
        n_vec = grad / (norm + 1e-8)
        force = self.confinement * self.h_mean * np.cross(n_vec, omega)
        force[self.solid] = 0.0
        return force

    def _add_forces(self, dt: float) -> None:
        # buoyancy: acceleration along -gravity proportional to smoke concentration
        s_face_w = np.zeros(self.w.shape)
        s_face_w[:, :, 1:-1] = 0.5 * (self.smoke[:, :, :-1] + self.smoke[:, :, 1:])
        s_face_u = np.zeros(self.u.shape)
        s_face_u[1:-1] = 0.5 * (self.smoke[:-1] + self.smoke[1:])
        s_face_v = np.zeros(self.v.shape)
        s_face_v[:, 1:-1] = 0.5 * (self.smoke[:, :-1] + self.smoke[:, 1:])
        accel = self.buoyancy * self.up
        self.u += dt * accel[0] * s_face_u
        self.v += dt * accel[1] * s_face_v
        self.w += dt * accel[2] * s_face_w

        # vorticity confinement (keeps the rolled-up wake structures alive)
        force = self._confinement_force()
        fu = np.zeros(self.u.shape)
        fu[1:-1] = 0.5 * (force[:-1, :, :, 0] + force[1:, :, :, 0])
        fv = np.zeros(self.v.shape)
        fv[:, 1:-1] = 0.5 * (force[:, :-1, :, 1] + force[:, 1:, :, 1])
        fw = np.zeros(self.w.shape)
        fw[:, :, 1:-1] = 0.5 * (force[:, :, :-1, 2] + force[:, :, 1:, 2])
        self.u += dt * fu
        self.v += dt * fv
        self.w += dt * fw

        # weak ambient drag inside the closed box
        decay = 1.0 / (1.0 + self.velocity_damping * dt)
        self.u *= decay
        self.v *= decay
        self.w *= decay
        self._apply_solid_bc()

    def _project(self, dt: float) -> None:
        hx, hy, hz = self.h
        div = ((self.u[1:] - self.u[:-1]) / hx
               + (self.v[:, 1:] - self.v[:, :-1]) / hy
               + (self.w[:, :, 1:] - self.w[:, :, :-1]) / hz)
        rhs = -div[self.fluid] / dt
        rhs = rhs - rhs.mean()
        try:
            p, info = spla.cg(self.poisson, rhs, x0=self.pressure, M=self.precond,
                              rtol=1e-7, atol=0.0, maxiter=400)
        except TypeError:
            p, info = spla.cg(self.poisson, rhs, x0=self.pressure, M=self.precond,
                              tol=1e-7, maxiter=400)
        if not np.all(np.isfinite(p)):
            p = np.zeros_like(rhs)
        self.pressure = p
        self._cg_iterations += 1

        field = np.zeros(self.shape)
        field[self.fluid] = p
        gu = np.zeros(self.u.shape)
        gu[1:-1] = (field[1:] - field[:-1]) / hx
        gv = np.zeros(self.v.shape)
        gv[:, 1:-1] = (field[:, 1:] - field[:, :-1]) / hy
        gw = np.zeros(self.w.shape)
        gw[:, :, 1:-1] = (field[:, :, 1:] - field[:, :, :-1]) / hz
        self.u -= dt * gu
        self.v -= dt * gv
        self.w -= dt * gw
        self._apply_solid_bc()

    def _advect_particles(self, dt: float) -> None:
        vel = self._velocity_at(self.particles)
        mid = self.particles + 0.5 * dt * vel
        vel_mid = self._velocity_at(mid)
        self.particles = self.particles + dt * vel_mid
        # keep tracers outside the rigid sphere and inside the closed domain
        rel = self.particles - self.sphere_center[None, :]
        dist = np.linalg.norm(rel, axis=1)
        hit = dist < self.sphere_radius + 1e-3
        if np.any(hit):
            safe = np.maximum(dist[hit], 1e-9)
            direction = rel[hit] / safe[:, None]
            self.particles[hit] = (self.sphere_center[None, :]
                                   + (self.sphere_radius + 1e-3) * direction)
            radial = np.sum(vel_mid[hit] * direction, axis=1)
            vel_mid[hit] -= radial[:, None] * direction
        margin = 0.25 * self.h
        self.particles = np.clip(self.particles, (self.lo + margin)[None, :],
                                 (self.hi - margin)[None, :])
        self.particle_velocity = vel_mid

    def _step(self, dt: float) -> None:
        self._advect(dt)
        # prescribed smoke injection at the emitter, then mild dissipation
        self.smoke = self.smoke + dt * self.scalar_source_sim
        self.smoke *= 1.0 / (1.0 + self.smoke_dissipation * dt)
        self.smoke = np.clip(np.where(self.fluid, self.smoke, 0.0), 0.0, 4.0)
        self._add_forces(dt)
        self._project(dt)
        self._advect_particles(dt)
        self.time += dt
        self._steps += 1

    # ---------------------------------------------------------------- frame export
    def _export_field(self, array: np.ndarray) -> np.ndarray:
        return _trilinear(array, self.c_origin, self.h,
                          self.export_pts).reshape(self.export_shape)

    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        vel = self._collocated()
        velocity_field = np.stack(
            [self._export_field(np.ascontiguousarray(vel[..., c])) for c in range(3)], axis=-1)
        velocity_field[~self.export_active] = 0.0
        smoke_export = np.maximum(self._export_field(self.smoke), 0.0)
        smoke_export[~self.export_active] = 0.0

        boundary_velocity = self._velocity_at(self.boundary_points)
        self._max_speed = max(self._max_speed, float(np.abs(vel).max()))

        n_part = len(self.particles)
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.particles, dtype=float),
            velocity=np.ascontiguousarray(self.particle_velocity, dtype=float),
            mass=self.particle_mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.particle_mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.labels.copy(),
                "deformation_gradient": self.particle_F.copy(),
            },
            fields={
                "transported_scalar": np.ascontiguousarray(smoke_export, dtype=float),
                "scalar_source": np.ascontiguousarray(self.export_source, dtype=float),
                "active_mask": self.export_active.astype(np.float64),
                "velocity_field": np.ascontiguousarray(velocity_field, dtype=float),
                "boundary_velocity": np.ascontiguousarray(boundary_velocity, dtype=float),
                "boundary_target_velocity": np.ascontiguousarray(self.boundary_target,
                                                                 dtype=float),
                "boundary_normal": np.ascontiguousarray(self.boundary_normals, dtype=float),
                "cumulative_mass_source": np.zeros(n_part),
            },
            scalars={
                "grid_spacing": float(self.export_spacing_scalar),
                "contact_length_scale": float(self.export_spacing_scalar),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        dt_max = self.sample_dt / float(self.substeps)
        guard = 0
        while target - self.time > 1e-9 and guard < 100000:
            dt = min(dt_max, target - self.time)
            self._step(dt)
            guard += 1
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC staggered incompressible Euler, semi-Lagrangian advection, "
                      "vorticity confinement, PCG pressure projection",
            "sim_grid_shape": list(self.shape),
            "sim_spacing": [float(v) for v in self.h],
            "steps": int(self._steps),
            "fluid_cells": int(self.n_fluid),
            "max_speed": float(self._max_speed),
            "final_smoke_mass": float(self.smoke.sum() * float(np.prod(self.h))),
        }
