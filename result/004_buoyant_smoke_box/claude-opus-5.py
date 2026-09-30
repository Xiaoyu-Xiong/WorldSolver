"""Buoyant smoke box solver.

Physical model
--------------
Incompressible Boussinesq flow in a closed box:

    du/dt + (u . grad) u = -grad p + f_buoyancy + f_confinement
    div u = 0
    dT/dt + (u . grad) T = heating - cooling
    ds/dt + (u . grad) s = smoke_source

Discretization
--------------
* Uniform staggered MAC grid over the frozen voxel domain (32 x 32 x 64).
* Unconditionally stable semi-Lagrangian advection (RK2 midpoint backtrace plus
  trilinear interpolation) for the MAC velocity components and the collocated
  scalars.
* Boussinesq buoyancy on w-faces plus Steinhoff vorticity confinement, which
  keeps coherent vortex rings alive on this coarse grid.
* Exact pressure projection: the cell-centered Poisson problem with homogeneous
  Neumann (closed, no-penetration) walls is solved spectrally with a 3D
  DCT-II / DCT-III pair, so the projected MAC field is discretely divergence
  free to machine precision at every substep.
* Passive markers and volume-tracking tracers ride the same divergence-free
  velocity field with RK2, so markers advect without clumping or volume drift.

Every exported frame comes from stepping this system forward in time from the
frozen initial state: no keyframes, no scripted motion, no Genesis.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from worldsolver.contracts import StateFrame


def _interp3(field: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Trilinear sample of ``field`` at continuous index coordinates (clamped)."""
    nx, ny, nz = field.shape
    fx = np.clip(gx, 0.0, nx - 1.0)
    fy = np.clip(gy, 0.0, ny - 1.0)
    fz = np.clip(gz, 0.0, nz - 1.0)
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    k0 = np.floor(fz).astype(np.int64)
    np.clip(i0, 0, max(nx - 2, 0), out=i0)
    np.clip(j0, 0, max(ny - 2, 0), out=j0)
    np.clip(k0, 0, max(nz - 2, 0), out=k0)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)
    tx = fx - i0
    ty = fy - j0
    tz = fz - k0
    c00 = field[i0, j0, k0]
    c00 = c00 + (field[i1, j0, k0] - c00) * tx
    c10 = field[i0, j1, k0]
    c10 = c10 + (field[i1, j1, k0] - c10) * tx
    c01 = field[i0, j0, k1]
    c01 = c01 + (field[i1, j0, k1] - c01) * tx
    c11 = field[i0, j1, k1]
    c11 = c11 + (field[i1, j1, k1] - c11) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


class BenchmarkSolver:
    """Deterministic MAC-grid Boussinesq smoke solver for the closed box."""

    # ---- frozen physical / numerical parameters ----
    buoyancy_alpha = 1.7      # g * thermal expansion [m/s^2 per unit temperature]
    smoke_weight = 0.0        # smoke mass loading feedback [m/s^2 per unit density]
    heating_rate = 25.0       # heater relaxation toward source temperature [1/s]
    cooling_rate = 0.06       # weak ambient cooling [1/s]
    smoke_rate = 1.5          # prescribed smoke injection [1/s]
    confinement = 2.2         # vorticity confinement strength [-]
    cfl = 1.0                 # advective CFL target
    max_substeps = 6
    tracer_count = 20000      # volume-error tracking particles

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        self.scene = scene
        self.seed = int(seed)

        domain = config.get("domain", {}) or {}
        layout = config.get("field_layout", {}) or {}
        initial = config.get("initial_conditions", {}) or {}

        self.bounds_min = np.asarray(
            domain.get("bounds_min", [-0.75, -0.75, 0.0]), dtype=np.float64
        )
        self.bounds_max = np.asarray(
            domain.get("bounds_max", [0.75, 0.75, 3.0]), dtype=np.float64
        )
        shape = tuple(int(v) for v in layout.get("grid_shape", [32, 32, 64]))
        self.nx, self.ny, self.nz = shape
        self.shape = shape

        extent = self.bounds_max - self.bounds_min
        self.hx = float(extent[0]) / self.nx
        self.hy = float(extent[1]) / self.ny
        self.hz = float(extent[2]) / self.nz
        self.h = float(min(self.hx, self.hy, self.hz))

        self.ambient_temperature = float(initial.get("ambient_temperature", 0.0))
        self.source_temperature = float(initial.get("source_temperature", 1.0))
        self.source_center = np.asarray(
            initial.get("source_center", [0.0, 0.0, 0.12]), dtype=np.float64
        )
        self.source_radius = float(initial.get("source_radius", 0.28))
        self.source_height = float(initial.get("source_height", 0.24))

        self.characteristic_length = float(extent[0])
        self.reference_velocity = float(
            np.sqrt(
                max(self.buoyancy_alpha, 1e-9)
                * max(abs(self.source_temperature - self.ambient_temperature), 1e-9)
                * self.characteristic_length
            )
        )

        # cell centers and MAC face coordinates
        self.xc = self.bounds_min[0] + (np.arange(self.nx) + 0.5) * self.hx
        self.yc = self.bounds_min[1] + (np.arange(self.ny) + 0.5) * self.hy
        self.zc = self.bounds_min[2] + (np.arange(self.nz) + 0.5) * self.hz
        self.xf = self.bounds_min[0] + np.arange(self.nx + 1) * self.hx
        self.yf = self.bounds_min[1] + np.arange(self.ny + 1) * self.hy
        self.zf = self.bounds_min[2] + np.arange(self.nz + 1) * self.hz

        self.Xc, self.Yc, self.Zc = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")

        # fixed sample positions used by the semi-Lagrangian backtraces
        self.pos_c = self._stack(self.Xc, self.Yc, self.Zc)
        self.pos_u = self._stack(*np.meshgrid(self.xf, self.yc, self.zc, indexing="ij"))
        self.pos_v = self._stack(*np.meshgrid(self.xc, self.yf, self.zc, indexing="ij"))
        self.pos_w = self._stack(*np.meshgrid(self.xc, self.yc, self.zf, indexing="ij"))

        # MAC velocity state
        init_velocity = np.asarray(
            initial.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        self.u = np.full((self.nx + 1, self.ny, self.nz), float(init_velocity[0]))
        self.v = np.full((self.nx, self.ny + 1, self.nz), float(init_velocity[1]))
        self.w = np.full((self.nx, self.ny, self.nz + 1), float(init_velocity[2]))

        rng = np.random.default_rng(self.seed)
        noise = 0.02 * max(self.reference_velocity, 1e-3)
        self.u += noise * self._smooth_noise(rng, self.u.shape)
        self.v += noise * self._smooth_noise(rng, self.v.shape)
        self.w += noise * self._smooth_noise(rng, self.w.shape)
        self._apply_wall_bc()

        self._build_source(rng)

        # initial scalars: the frozen scene profiles evaluated at cell centers
        r2 = self.Xc * self.Xc + self.Yc * self.Yc
        smoke = np.exp(-r2 / (2.0 * 0.18**2)) * np.exp(
            -((self.Zc - 0.13) ** 2) / (2.0 * 0.10**2)
        )
        smoke[self.Zc > 0.34] = 0.0
        self.smoke = smoke
        self.temperature = np.exp(-r2 / (2.0 * 0.20**2)) * np.exp(
            -((self.Zc - 0.10) ** 2) / (2.0 * 0.09**2)
        )
        self.temperature += 0.01 * self._smooth_noise(rng, self.shape) * self.source_mask

        self._build_poisson()
        self._project()

        # passive markers: exactly the frozen scene seeding
        frame0 = scene.initial_state()
        self.markers = np.array(frame0.position, dtype=np.float64, copy=True)
        self.marker_count = int(self.markers.shape[0])
        if frame0.mass is not None:
            self.marker_mass = np.array(frame0.mass, dtype=np.float64, copy=True)
        else:
            self.marker_mass = np.full(
                self.marker_count, 1.0 / max(self.marker_count, 1), dtype=np.float64
            )
        self.marker_kind = np.ones(self.marker_count, dtype=np.int32)
        self.marker_velocity = self._sample_velocity(self.markers)

        # volume tracers carrying the flow-map Jacobian
        tracer_rng = np.random.default_rng(self.seed + 7919)
        self.tracers = self.bounds_min + tracer_rng.random(
            (int(self.tracer_count), 3)
        ) * (self.bounds_max - self.bounds_min)
        self.tracer_jacobian = np.ones(int(self.tracer_count), dtype=np.float64)

        self.time = 0.0
        self.substeps_taken = 0
        self.max_divergence_l2 = 0.0
        self.max_volume_error = 0.0
        self._build_active_mask()
        self._build_boundary_samples()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _stack(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    @staticmethod
    def _smooth_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        """Deterministic band-limited zero-mean noise in [-1, 1]."""
        out = rng.standard_normal(shape)
        for axis in range(out.ndim):
            out = 0.25 * (np.roll(out, 1, axis=axis) + 2.0 * out + np.roll(out, -1, axis=axis))
        peak = float(np.max(np.abs(out)))
        if peak > 0.0:
            out = out / peak
        return out - float(out.mean())

    def _build_source(self, rng: np.random.Generator) -> None:
        """Heater footprint: tapered cylinder matching the frozen emitter."""
        cx = float(self.source_center[0])
        cy = float(self.source_center[1])
        radius = np.sqrt((self.Xc - cx) ** 2 + (self.Yc - cy) ** 2)
        radial = np.clip((self.source_radius - radius) / self.hx, 0.0, 1.0)
        plate_top = float(self.bounds_min[2]) + max(0.075, 1.6 * self.hz)
        axial = np.clip((plate_top - self.Zc) / self.hz, 0.0, 1.0)
        theta = np.arctan2(self.Yc - cy, self.Xc - cx)
        modulation = 1.0 + 0.12 * np.cos(3.0 * theta) + 0.08 * np.sin(2.0 * theta + 0.7)
        mask = radial * axial * modulation
        mask = mask * (1.0 + 0.05 * self._smooth_noise(rng, self.shape))
        self.source_mask = np.clip(mask, 0.0, 1.0)
        self.scalar_source = self.smoke_rate * self.source_mask

    def _apply_wall_bc(self) -> None:
        """Closed box: no penetration on all six walls, free slip tangentially."""
        self.u[0, :, :] = 0.0
        self.u[-1, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    def _build_poisson(self) -> None:
        """Eigenvalues of the cell-centered Neumann Laplacian in the DCT-II basis."""
        lx = (2.0 * np.cos(np.pi * np.arange(self.nx) / self.nx) - 2.0) / (self.hx * self.hx)
        ly = (2.0 * np.cos(np.pi * np.arange(self.ny) / self.ny) - 2.0) / (self.hy * self.hy)
        lz = (2.0 * np.cos(np.pi * np.arange(self.nz) / self.nz) - 2.0) / (self.hz * self.hz)
        eig = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]
        eig[0, 0, 0] = 1.0  # pin the pure-Neumann null space
        self.poisson_eig = eig

    def _divergence(self) -> np.ndarray:
        return (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / self.hx
            + (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.hy
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.hz
        )

    def _project(self) -> None:
        """Exact discrete projection onto the divergence-free MAC subspace."""
        self._apply_wall_bc()
        divergence = self._divergence()
        rhs = divergence - float(divergence.mean())  # closed-wall compatibility
        coefficients = dctn(rhs, type=2, norm="ortho")
        coefficients = coefficients / self.poisson_eig
        coefficients[0, 0, 0] = 0.0
        pressure = idctn(coefficients, type=2, norm="ortho")
        self.u[1:-1, :, :] -= (pressure[1:, :, :] - pressure[:-1, :, :]) / self.hx
        self.v[:, 1:-1, :] -= (pressure[:, 1:, :] - pressure[:, :-1, :]) / self.hy
        self.w[:, :, 1:-1] -= (pressure[:, :, 1:] - pressure[:, :, :-1]) / self.hz
        self._apply_wall_bc()

    # ---------------------------------------------------------- interpolation
    def _grid_coords(self, points: np.ndarray, offset: tuple[float, float, float]):
        gx = (points[:, 0] - self.bounds_min[0]) / self.hx - offset[0]
        gy = (points[:, 1] - self.bounds_min[1]) / self.hy - offset[1]
        gz = (points[:, 2] - self.bounds_min[2]) / self.hz - offset[2]
        return gx, gy, gz

    def _sample_velocity(self, points: np.ndarray) -> np.ndarray:
        gx, gy, gz = self._grid_coords(points, (0.0, 0.5, 0.5))
        vx = _interp3(self.u, gx, gy, gz)
        gx, gy, gz = self._grid_coords(points, (0.5, 0.0, 0.5))
        vy = _interp3(self.v, gx, gy, gz)
        gx, gy, gz = self._grid_coords(points, (0.5, 0.5, 0.0))
        vz = _interp3(self.w, gx, gy, gz)
        return np.column_stack((vx, vy, vz))

    def _sample_center(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        gx, gy, gz = self._grid_coords(points, (0.5, 0.5, 0.5))
        return _interp3(field, gx, gy, gz)

    def _backtrace(self, points: np.ndarray, dt: float) -> np.ndarray:
        """RK2 (midpoint) departure points of the characteristics."""
        velocity = self._sample_velocity(points)
        midpoint = points - 0.5 * dt * velocity
        np.clip(midpoint, self.bounds_min, self.bounds_max, out=midpoint)
        departure = points - dt * self._sample_velocity(midpoint)
        np.clip(departure, self.bounds_min, self.bounds_max, out=departure)
        return departure

    # --------------------------------------------------------------- physics
    def _centered_velocity(self) -> np.ndarray:
        centered = np.empty(self.shape + (3,), dtype=np.float64)
        centered[..., 0] = 0.5 * (self.u[1:, :, :] + self.u[:-1, :, :])
        centered[..., 1] = 0.5 * (self.v[:, 1:, :] + self.v[:, :-1, :])
        centered[..., 2] = 0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])
        return centered

    def _add_buoyancy(self, dt: float) -> None:
        """Boussinesq lift on interior w-faces from face-averaged scalars."""
        temperature_face = 0.5 * (self.temperature[:, :, 1:] + self.temperature[:, :, :-1])
        force = self.buoyancy_alpha * (temperature_face - self.ambient_temperature)
        if self.smoke_weight != 0.0:
            smoke_face = 0.5 * (self.smoke[:, :, 1:] + self.smoke[:, :, :-1])
            force = force - self.smoke_weight * smoke_face
        self.w[:, :, 1:-1] += dt * force

    def _add_confinement(self, dt: float) -> None:
        """Steinhoff vorticity confinement restores coarse-grid vortex rings."""
        if self.confinement <= 0.0:
            return
        centered = self._centered_velocity()
        vx = centered[..., 0]
        vy = centered[..., 1]
        vz = centered[..., 2]
        omega_x = np.gradient(vz, self.hy, axis=1) - np.gradient(vy, self.hz, axis=2)
        omega_y = np.gradient(vx, self.hz, axis=2) - np.gradient(vz, self.hx, axis=0)
        omega_z = np.gradient(vy, self.hx, axis=0) - np.gradient(vx, self.hy, axis=1)
        magnitude = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        gx = np.gradient(magnitude, self.hx, axis=0)
        gy = np.gradient(magnitude, self.hy, axis=1)
        gz = np.gradient(magnitude, self.hz, axis=2)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        gx = gx / norm
        gy = gy / norm
        gz = gz / norm
        scale = self.confinement * self.h
        fx = scale * (gy * omega_z - gz * omega_y)
        fy = scale * (gz * omega_x - gx * omega_z)
        fz = scale * (gx * omega_y - gy * omega_x)
        self.u[1:-1, :, :] += dt * 0.5 * (fx[1:, :, :] + fx[:-1, :, :])
        self.v[:, 1:-1, :] += dt * 0.5 * (fy[:, 1:, :] + fy[:, :-1, :])
        self.w[:, :, 1:-1] += dt * 0.5 * (fz[:, :, 1:] + fz[:, :, :-1])

    def _advect_velocity(self, dt: float) -> None:
        u_old, v_old, w_old = self.u, self.v, self.w
        departure = self._backtrace(self.pos_u, dt)
        gx, gy, gz = self._grid_coords(departure, (0.0, 0.5, 0.5))
        new_u = _interp3(u_old, gx, gy, gz).reshape(u_old.shape)
        departure = self._backtrace(self.pos_v, dt)
        gx, gy, gz = self._grid_coords(departure, (0.5, 0.0, 0.5))
        new_v = _interp3(v_old, gx, gy, gz).reshape(v_old.shape)
        departure = self._backtrace(self.pos_w, dt)
        gx, gy, gz = self._grid_coords(departure, (0.5, 0.5, 0.0))
        new_w = _interp3(w_old, gx, gy, gz).reshape(w_old.shape)
        self.u, self.v, self.w = new_u, new_v, new_w
        self._apply_wall_bc()

    def _advect_scalars(self, dt: float) -> None:
        departure = self._backtrace(self.pos_c, dt)
        gx, gy, gz = self._grid_coords(departure, (0.5, 0.5, 0.5))
        self.smoke = _interp3(self.smoke, gx, gy, gz).reshape(self.shape)
        self.temperature = _interp3(self.temperature, gx, gy, gz).reshape(self.shape)

    def _apply_scalar_sources(self, dt: float) -> None:
        # prescribed smoke injection, applied exactly as exported in
        # fields.scalar_source; semi-Lagrangian advection uses positive
        # interpolation weights, so the scalar stays non-negative on its own
        self.smoke = self.smoke + dt * self.scalar_source
        # heater relaxation plus weak ambient cooling
        heating = self.heating_rate * self.source_mask
        self.temperature = self.temperature + dt * (
            heating * (self.source_temperature - self.temperature)
            - self.cooling_rate * (self.temperature - self.ambient_temperature)
        )
        bound = 2.0 * abs(self.source_temperature) + 1.0
        np.clip(self.temperature, -bound, bound, out=self.temperature)

    def _advect_particles(self, dt: float) -> None:
        """Move markers and tracers with the current divergence-free field."""
        divergence = self._divergence()
        self.tracer_jacobian *= 1.0 + dt * self._sample_center(divergence, self.tracers)
        np.clip(self.tracer_jacobian, 1e-6, 1e6, out=self.tracer_jacobian)
        for array in (self.markers, self.tracers):
            velocity = self._sample_velocity(array)
            midpoint = array + 0.5 * dt * velocity
            np.clip(midpoint, self.bounds_min, self.bounds_max, out=midpoint)
            array += dt * self._sample_velocity(midpoint)
            np.clip(array, self.bounds_min, self.bounds_max, out=array)
        self.marker_velocity = self._sample_velocity(self.markers)

    def _substep(self, dt: float) -> None:
        self._advect_particles(dt)
        self._advect_scalars(dt)
        self._apply_scalar_sources(dt)
        self._advect_velocity(dt)
        self._add_buoyancy(dt)
        self._add_confinement(dt)
        self._project()
        self.substeps_taken += 1

    def _step_to(self, target: float) -> None:
        while target - self.time > 1e-12:
            remaining = target - self.time
            speed = max(
                float(np.max(np.abs(self.u))),
                float(np.max(np.abs(self.v))),
                float(np.max(np.abs(self.w))),
                1e-6,
            )
            dt_cfl = self.cfl * self.h / speed
            count = int(np.ceil(remaining / max(dt_cfl, 1e-9)))
            count = max(1, min(count, int(self.max_substeps)))
            dt = remaining / count
            for _ in range(count):
                self._substep(dt)
            self.time = target
        self.time = target

    # ------------------------------------------------------------ diagnostics
    def _build_active_mask(self) -> None:
        """Cells where the collocated field equations are evaluated (interior)."""
        mask = np.zeros(self.shape, dtype=np.float64)
        mask[1:-1, 1:-1, 1:-1] = 1.0
        self.active_mask = mask

    def _build_boundary_samples(self) -> None:
        """Subsampled wall samples with outward unit normals for the closed box."""
        stride = 2
        xs = self.xc[::stride]
        ys = self.yc[::stride]
        zs = self.zc[::stride]
        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []

        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        for wall, normal in ((self.bounds_min[0], (-1.0, 0.0, 0.0)),
                             (self.bounds_max[0], (1.0, 0.0, 0.0))):
            sample = np.column_stack((np.full(Y.size, float(wall)), Y.ravel(), Z.ravel()))
            points.append(sample)
            normals.append(np.tile(np.asarray(normal, dtype=np.float64), (sample.shape[0], 1)))

        X, Z = np.meshgrid(xs, zs, indexing="ij")
        for wall, normal in ((self.bounds_min[1], (0.0, -1.0, 0.0)),
                             (self.bounds_max[1], (0.0, 1.0, 0.0))):
            sample = np.column_stack((X.ravel(), np.full(X.size, float(wall)), Z.ravel()))
            points.append(sample)
            normals.append(np.tile(np.asarray(normal, dtype=np.float64), (sample.shape[0], 1)))

        X, Y = np.meshgrid(xs, ys, indexing="ij")
        for wall, normal in ((self.bounds_min[2], (0.0, 0.0, -1.0)),
                             (self.bounds_max[2], (0.0, 0.0, 1.0))):
            sample = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, float(wall))))
            points.append(sample)
            normals.append(np.tile(np.asarray(normal, dtype=np.float64), (sample.shape[0], 1)))

        self.boundary_points = np.vstack(points)
        self.boundary_normal = np.vstack(normals)
        # static closed walls: the prescribed wall velocity is identically zero
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

    def _boundary_velocity(self) -> np.ndarray:
        """Fluid velocity at the wall samples.

        The MAC discretization pins the wall-normal component to zero, so the
        sampled velocity is purely tangential (free slip).
        """
        velocity = self._sample_velocity(self.boundary_points)
        normal = self.boundary_normal
        normal_component = np.sum(velocity * normal, axis=1, keepdims=True)
        return velocity - normal_component * normal

    def _diagnostic_scalars(self, centered: np.ndarray) -> dict[str, float]:
        divergence = (
            np.gradient(centered[..., 0], self.hx, axis=0)
            + np.gradient(centered[..., 1], self.hy, axis=1)
            + np.gradient(centered[..., 2], self.hz, axis=2)
        )
        selected = divergence[self.active_mask > 0.5]
        scale = max(self.reference_velocity, 1e-12) / max(self.characteristic_length, 1e-12)
        if selected.size:
            divergence_l2 = float(np.sqrt(np.mean(selected * selected)) / scale)
            divergence_max = float(np.max(np.abs(selected)) / scale)
        else:
            divergence_l2 = 0.0
            divergence_max = 0.0
        volume_error = float(np.mean(np.abs(self.tracer_jacobian - 1.0)))
        self.max_divergence_l2 = max(self.max_divergence_l2, divergence_l2)
        self.max_volume_error = max(self.max_volume_error, volume_error)
        return {
            "divergence_l2": divergence_l2,
            "divergence_max": divergence_max,
            "relative_volume_error": volume_error,
        }

    # ----------------------------------------------------------------- output
    def _frame(self) -> StateFrame:
        centered = self._centered_velocity()
        scalars = self._diagnostic_scalars(centered)
        scalars.update(
            {
                "grid_spacing": float(self.h),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
            }
        )
        smoke = np.ascontiguousarray(self.smoke, dtype=np.float32)
        frame = StateFrame(
            time=float(max(self.time, 0.0)),
            position=np.ascontiguousarray(self.markers, dtype=np.float32),
            velocity=np.ascontiguousarray(self.marker_velocity, dtype=np.float32),
            mass=np.ascontiguousarray(self.marker_mass, dtype=np.float32),
            attributes={"marker_kind": self.marker_kind},
            fields={
                "smoke_density": smoke,
                "transported_scalar": smoke,
                "scalar_source": np.ascontiguousarray(self.scalar_source, dtype=np.float32),
                "temperature": np.ascontiguousarray(self.temperature, dtype=np.float32),
                "velocity_field": np.ascontiguousarray(centered, dtype=np.float32),
                "active_mask": np.ascontiguousarray(self.active_mask, dtype=np.float32),
                "boundary_velocity": np.ascontiguousarray(
                    self._boundary_velocity(), dtype=np.float32
                ),
                "boundary_target_velocity": np.ascontiguousarray(
                    self.boundary_target_velocity, dtype=np.float32
                ),
                "boundary_normal": np.ascontiguousarray(self.boundary_normal, dtype=np.float32),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------- Solver protocol
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time:
            self._step_to(target)
        else:
            self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC Boussinesq smoke; semi-Lagrangian advection; DCT exact projection",
            "grid_shape": [int(v) for v in self.shape],
            "grid_spacing": float(self.h),
            "substeps_taken": int(self.substeps_taken),
            "marker_count": int(self.marker_count),
            "tracer_count": int(self.tracer_count),
            "buoyancy_alpha": float(self.buoyancy_alpha),
            "vorticity_confinement": float(self.confinement),
            "max_divergence_l2": float(self.max_divergence_l2),
            "max_relative_volume_error": float(self.max_volume_error),
            "final_time": float(self.time),
            "genesis_used": False,
        }
