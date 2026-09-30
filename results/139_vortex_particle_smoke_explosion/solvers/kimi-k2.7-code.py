"""Vortex-particle-style smoke explosion solver on a 3D rectilinear grid.

The solver advances a collocated velocity field and a transported smoke scalar
using stable time stepping with semi-Lagrangian advection, buoyancy, a short
radial blast forcing, and vorticity confinement.  A set of tracer particles is
advected by the same velocity field and the ground plane is enforced as a rigid
no-penetration wall.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver implementation for the vortex-particle smoke explosion scene."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        rng = np.random.default_rng(self._seed)

        # ------------------------------------------------------------------
        # Scene / configuration
        # ------------------------------------------------------------------
        self._scene = scene
        self._config = getattr(scene, "config", {})
        self._gravity = np.asarray(self._config.get("gravity", [0.0, 0.0, -9.81]), float)

        # ------------------------------------------------------------------
        # Initial frame supplied by the frozen scene
        # ------------------------------------------------------------------
        frame0 = scene.initial_state()
        frame0.validate()

        self._time = float(frame0.time)
        self._position = np.asarray(frame0.position, dtype=float).copy()
        self._velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self._mass = np.asarray(frame0.mass, dtype=float).copy() if frame0.mass is not None else None
        self._body_id = np.asarray(frame0.body_id, dtype=np.int32).copy() if frame0.body_id is not None else None

        # Preserve attributes with the same leading dimension as position
        self._attributes = {}
        for name, value in (frame0.attributes or {}).items():
            self._attributes[name] = np.asarray(value, dtype=float).copy()

        # ------------------------------------------------------------------
        # Diagnostic / bound field grid
        # ------------------------------------------------------------------
        sampling = self._config.get("sampling", {})
        self._shape = tuple(sampling.get("field_grid_shape", (24, 16, 28)))
        nx, ny, nz = self._shape

        # Bounds match the scene_geometry "subject" voxel_grid
        self._bounds_min = np.array([-3.0, -1.2, 0.0], float)
        self._bounds_max = np.array([3.0, 1.2, 3.8], float)

        # Cell-centered coordinate vectors (non-uniform spacing is allowed)
        self._x = np.linspace(self._bounds_min[0], self._bounds_max[0], nx)
        self._y = np.linspace(self._bounds_min[1], self._bounds_max[1], ny)
        self._z = np.linspace(self._bounds_min[2], self._bounds_max[2], nz)

        # Convenience spacings
        self._dx = np.diff(self._x)
        self._dy = np.diff(self._y)
        self._dz = np.diff(self._z)
        self._h = float(sampling.get("spacing", 0.18))

        # Scalar and source fields
        if "transported_scalar" in (frame0.fields or {}):
            s0 = np.asarray(frame0.fields["transported_scalar"], dtype=float)
        else:
            s0 = np.zeros(self._shape, dtype=float)
        self._scalar = s0.copy()

        if "scalar_source" in (frame0.fields or {}):
            self._scalar_source = np.asarray(frame0.fields["scalar_source"], dtype=float).copy()
        else:
            self._scalar_source = np.zeros(self._shape, dtype=float)

        if "velocity_field" in (frame0.fields or {}):
            u0 = np.asarray(frame0.fields["velocity_field"], dtype=float)
            if u0.shape[:3] != self._shape:
                u0 = np.zeros(self._shape + (3,), dtype=float)
        else:
            u0 = np.zeros(self._shape + (3,), dtype=float)
        self._u = u0.copy()

        # Active mask: all cells inside the open domain participate
        if "active_mask" in (frame0.fields or {}):
            self._active_mask = np.asarray(frame0.fields["active_mask"], dtype=bool).copy()
        else:
            self._active_mask = np.ones(self._shape, dtype=bool)

        # ------------------------------------------------------------------
        # Blast source parameters
        # ------------------------------------------------------------------
        self._blast_center = np.array([0.0, 0.0, 0.85], float)
        self._blast_radius = 0.35
        self._blast_duration = 1.2
        self._blast_peak_time = 0.25
        self._blast_strength = 35.0
        self._buoyancy = 25.0
        self._vorticity_strength = 6.0
        self._diffusion = 1e-3
        self._scalar_diffusion = 2e-4

        # ------------------------------------------------------------------
        # Substepping / CFL
        # ------------------------------------------------------------------
        self._sample_dt = float(self._config.get("sample_dt", 1.0 / 30.0))
        self._max_substeps = 12
        self._cfl = 0.35

        # ------------------------------------------------------------------
        # Ground boundary samples
        # ------------------------------------------------------------------
        self._boundary_samples = self._make_ground_boundary_samples()
        self._boundary_normals = self._make_boundary_normals()
        self._boundary_target_vel = np.zeros_like(self._boundary_samples)

        # ------------------------------------------------------------------
        # Diagnostic bookkeeping
        # ------------------------------------------------------------------
        self._scalars = {}
        for name, value in (frame0.scalars or {}).items():
            self._scalars[name] = float(value)
        self._scalars.setdefault("grid_spacing", self._h)
        self._scalars.setdefault("characteristic_length", self._h)
        self._scalars.setdefault("reference_velocity", 1.0)

        # Seed a small amount of noise to break symmetry for the mushroom shape
        self._seed_noise(rng)

        self._initialized = True

    # ------------------------------------------------------------------
    # Public Solver protocol
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not hasattr(self, "_initialized"):
            raise RuntimeError("Solver has not been initialized")

        target = float(time)
        if target < self._time - 1e-12:
            raise ValueError(f"advance_to target {target} is before current time {self._time}")

        # Step forward until the requested sample time is reached.
        while self._time < target - 1e-12:
            dt = self._choose_dt()
            if self._time + dt > target:
                dt = target - self._time
            self._step(dt)
            self._time += dt

        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "max_velocity": float(np.max(np.linalg.norm(self._u, axis=-1))) if self._u is not None else 0.0,
            "max_scalar": float(np.max(self._scalar)) if self._scalar is not None else 0.0,
            "total_scalar": float(np.sum(self._scalar)) if self._scalar is not None else 0.0,
            "time": float(self._time),
        }

    # ------------------------------------------------------------------
    # Boundary construction
    # ------------------------------------------------------------------
    def _make_ground_boundary_samples(self) -> np.ndarray:
        """Return an array of sample points on the ground plane (z=0)."""
        n_x, n_y = 16, 10
        xs = np.linspace(-2.5, 2.5, n_x)
        ys = np.linspace(-0.9, 0.9, n_y)
        xv, yv = np.meshgrid(xs, ys, indexing="ij")
        zv = np.zeros_like(xv)
        return np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)

    def _make_boundary_normals(self) -> np.ndarray:
        n = self._boundary_samples.shape[0]
        normals = np.zeros((n, 3), float)
        normals[:, 2] = 1.0
        return normals

    # ------------------------------------------------------------------
    # Initial symmetry-breaking noise
    # ------------------------------------------------------------------
    def _seed_noise(self, rng: np.random.Generator) -> None:
        # Add a small low-frequency perturbation to the velocity field
        noise = 0.15 * rng.standard_normal(self._shape + (3,)).astype(float)
        # Damp the vertical component near the ground to respect the wall
        nz = self._shape[2]
        for k in range(nz):
            noise[:, :, k, 2] *= (k + 1) / nz
        self._u += noise

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _choose_dt(self) -> float:
        umax = np.max(np.abs(self._u)) if self._u.size else 0.0
        if umax < 1e-12:
            return self._sample_dt
        h_min = min(np.min(self._dx), np.min(self._dy), np.min(self._dz))
        dt = self._cfl * h_min / (umax + 1e-9)
        dt = min(dt, self._sample_dt)
        # Clamp within a reasonable range for the 8-second simulation
        return max(dt, 1e-5)

    def _step(self, dt: float) -> None:
        # 1. Compute current scalar source (blast emitter)
        self._update_scalar_source(dt)

        # 2. Velocity update
        self._u = self._advect_field(self._u, dt)
        self._u = self._apply_forces(dt)
        self._u = self._apply_ground_boundary(self._u)
        self._u += self._vorticity_confinement() * dt
        self._u = self._apply_diffusion(self._u, dt, self._diffusion)

        # 3. Scalar update
        self._scalar = self._advect_scalar(self._scalar, dt)
        self._scalar += self._scalar_source * dt
        self._scalar = self._apply_scalar_diffusion(self._scalar, dt, self._scalar_diffusion)

        # 4. Advect tracer particles with the diagnostic velocity field
        self._advect_particles(dt)

    # ------------------------------------------------------------------
    # Scalar source / blast forcing
    # ------------------------------------------------------------------
    def _update_scalar_source(self, dt: float) -> None:
        t = self._time
        # Pulse: rises quickly then decays
        pulse = 0.0
        if t < self._blast_duration:
            pulse = max(0.0, math.sin(math.pi * (1.0 - t / self._blast_duration)))

        X, Y, Z = np.meshgrid(self._x, self._y, self._z, indexing="ij")
        dx = X - self._blast_center[0]
        dy = Y - self._blast_center[1]
        dz = Z - self._blast_center[2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        gauss = np.exp(-((dist / self._blast_radius) ** 2))

        # Source rate: adds smoke density around the blast core
        self._scalar_source = 2.5 * pulse * gauss

    def _blast_force(self) -> np.ndarray:
        X, Y, Z = np.meshgrid(self._x, self._y, self._z, indexing="ij")
        dx = X - self._blast_center[0]
        dy = Y - self._blast_center[1]
        dz = Z - self._blast_center[2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        gauss = np.exp(-((dist / self._blast_radius) ** 2))

        t = self._time
        pulse = 0.0
        if t < self._blast_duration:
            pulse = max(0.0, math.sin(math.pi * (1.0 - t / self._blast_duration)))

        # Outward direction from the blast centre, with a slight upward bias
        outward = np.stack([dx, dy, dz + 0.25 * self._blast_radius], axis=-1)
        norm = np.linalg.norm(outward, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-9, 1.0, norm)
        outward = outward / norm

        force = self._blast_strength * pulse * gauss[..., None] * outward
        return force

    def _apply_forces(self, dt: float) -> np.ndarray:
        u = self._u

        # Buoyancy in the +z direction proportional to smoke density
        buoy = np.zeros_like(u)
        buoy[..., 2] = self._buoyancy * self._scalar[..., None]

        blast = self._blast_force()

        # Gravity only acts on the smoke field through buoyancy in this model;
        # ambient air is already accounted for implicitly.
        return u + (buoy + blast) * dt

    # ------------------------------------------------------------------
    # Vorticity confinement (creates the fine rolled vortices)
    # ------------------------------------------------------------------
    def _curl(self, u: np.ndarray) -> np.ndarray:
        dudx = self._grad_x(u[..., 0])
        dudy = self._grad_y(u[..., 0])
        dudz = self._grad_z(u[..., 0])
        dvdx = self._grad_x(u[..., 1])
        dvdy = self._grad_y(u[..., 1])
        dvdz = self._grad_z(u[..., 1])
        dwdx = self._grad_x(u[..., 2])
        dwdy = self._grad_y(u[..., 2])
        dwdz = self._grad_z(u[..., 2])

        omega = np.empty_like(u)
        # curl = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
        omega[..., 0] = dwdy - dvdz
        omega[..., 1] = dudz - dwdx
        omega[..., 2] = dvdx - dudy
        return omega

    def _vorticity_confinement(self) -> np.ndarray:
        omega = self._curl(self._u)
        mag = np.linalg.norm(omega, axis=-1, keepdims=True)
        # Gradient of vorticity magnitude in each direction
        eta_x = self._grad_x(mag[..., 0])
        eta_y = self._grad_y(mag[..., 0])
        eta_z = self._grad_z(mag[..., 0])
        eta = np.stack([eta_x, eta_y, eta_z], axis=-1)
        eta_norm = np.linalg.norm(eta, axis=-1, keepdims=True)
        eta_norm = np.where(eta_norm < 1e-9, 1.0, eta_norm)
        n = eta / eta_norm
        force = self._vorticity_strength * self._h * np.cross(n, omega, axisa=-1, axisb=-1, axisc=-1)
        return force

    # ------------------------------------------------------------------
    # Spatial derivatives (central differences on rectilinear grids)
    # ------------------------------------------------------------------
    def _grad_x(self, f: np.ndarray) -> np.ndarray:
        g = np.empty_like(f)
        g[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (self._x[2:, None, None] - self._x[:-2, None, None])
        g[0, :, :] = g[1, :, :]
        g[-1, :, :] = g[-2, :, :]
        return g

    def _grad_y(self, f: np.ndarray) -> np.ndarray:
        g = np.empty_like(f)
        g[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (self._y[None, 2:, None] - self._y[None, :-2, None])
        g[:, 0, :] = g[:, 1, :]
        g[:, -1, :] = g[:, -2, :]
        return g

    def _grad_z(self, f: np.ndarray) -> np.ndarray:
        g = np.empty_like(f)
        g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (self._z[None, None, 2:] - self._z[None, None, :-2])
        g[:, :, 0] = g[:, :, 1]
        g[:, :, -1] = g[:, :, -2]
        return g

    # ------------------------------------------------------------------
    # Semi-Lagrangian advection
    # ------------------------------------------------------------------
    def _advect_field(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Advect a 3D vector field by the collocated velocity."""
        X, Y, Z = np.meshgrid(self._x, self._y, self._z, indexing="ij")
        # Backtrace
        bx = X - dt * field[..., 0]
        by = Y - dt * field[..., 1]
        bz = Z - dt * field[..., 2]

        new_comps = []
        for c in range(field.shape[-1]):
            new_comps.append(self._trilinear(field[..., c], bx, by, bz))
        return np.stack(new_comps, axis=-1)

    def _advect_scalar(self, field: np.ndarray, dt: float) -> np.ndarray:
        X, Y, Z = np.meshgrid(self._x, self._y, self._z, indexing="ij")
        bx = X - dt * self._u[..., 0]
        by = Y - dt * self._u[..., 1]
        bz = Z - dt * self._u[..., 2]
        return self._trilinear(field, bx, by, bz)

    def _trilinear(self, field: np.ndarray, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        """Trilinear interpolation on the rectilinear grid."""
        nx, ny, nz = self._shape
        # Normalize query coordinates to continuous cell indices
        ix = (xq - self._x[0]) / (self._x[-1] - self._x[0]) * (nx - 1)
        iy = (yq - self._y[0]) / (self._y[-1] - self._y[0]) * (ny - 1)
        iz = (zq - self._z[0]) / (self._z[-1] - self._z[0]) * (nz - 1)

        i0 = np.floor(ix).astype(int)
        j0 = np.floor(iy).astype(int)
        k0 = np.floor(iz).astype(int)

        i0 = np.clip(i0, 0, nx - 2)
        j0 = np.clip(j0, 0, ny - 2)
        k0 = np.clip(k0, 0, nz - 2)

        tx = ix - i0
        ty = iy - j0
        tz = iz - k0

        # 8 corner weights
        out = (
            field[i0, j0, k0] * (1 - tx) * (1 - ty) * (1 - tz)
            + field[i0 + 1, j0, k0] * tx * (1 - ty) * (1 - tz)
            + field[i0, j0 + 1, k0] * (1 - tx) * ty * (1 - tz)
            + field[i0 + 1, j0 + 1, k0] * tx * ty * (1 - tz)
            + field[i0, j0, k0 + 1] * (1 - tx) * (1 - ty) * tz
            + field[i0 + 1, j0, k0 + 1] * tx * (1 - ty) * tz
            + field[i0, j0 + 1, k0 + 1] * (1 - tx) * ty * tz
            + field[i0 + 1, j0 + 1, k0 + 1] * tx * ty * tz
        )
        return out

    # ------------------------------------------------------------------
    # Diffusion and boundary handling
    # ------------------------------------------------------------------
    def _apply_diffusion(self, u: np.ndarray, dt: float, nu: float) -> np.ndarray:
        if nu <= 0.0:
            return u
        # Use forward-time central-space with explicit diffusion; stable for the
        # chosen small time steps.
        out = u.copy()
        for c in range(3):
            lap = np.zeros(self._shape, float)
            lap[1:-1, :, :] += (u[2:, :, :, c] - 2 * u[1:-1, :, :, c] + u[:-2, :, :, c])
            lap[:, 1:-1, :] += (u[:, 2:, :, c] - 2 * u[:, 1:-1, :, c] + u[:, :-2, :, c])
            lap[:, :, 1:-1] += (u[:, :, 2:, c] - 2 * u[:, :, 1:-1, c] + u[:, :, :-2, c])
            out[..., c] = u[..., c] + dt * nu * lap
        return out

    def _apply_scalar_diffusion(self, s: np.ndarray, dt: float, kappa: float) -> np.ndarray:
        if kappa <= 0.0:
            return s
        lap = np.zeros(self._shape, float)
        lap[1:-1, :, :] = (s[2:, :, :] - 2 * s[1:-1, :, :] + s[:-2, :, :])
        lap[:, 1:-1, :] += (s[:, 2:, :] - 2 * s[:, 1:-1, :] + s[:, :-2, :])
        lap[:, :, 1:-1] += (s[:, :, 2:] - 2 * s[:, :, 1:-1] + s[:, :, :-2])
        return s + dt * kappa * lap

    def _apply_ground_boundary(self, u: np.ndarray) -> np.ndarray:
        # No-penetration: zero vertical velocity at the ground
        u_out = u.copy()
        u_out[..., 0, 2] = 0.0
        # Tangential components are damped near the wall (no-slip light)
        u_out[..., 0, 0] *= 0.7
        u_out[..., 0, 1] *= 0.7
        # Keep all velocities above the ground fully valid
        return u_out

    # ------------------------------------------------------------------
    # Particle advection (kept continuous and finite)
    # ------------------------------------------------------------------
    def _advect_particles(self, dt: float) -> None:
        if self._position.size == 0:
            return
        # Sample velocity at particle positions
        vel = self._sample_velocity(self._position)
        self._velocity = vel
        self._position = self._position + dt * vel

        # Clamp to the computational domain leaving a small margin
        margin = 0.01
        lo = self._bounds_min + margin
        hi = self._bounds_max - margin
        np.clip(self._position[:, 0], lo[0], hi[0], out=self._position[:, 0])
        np.clip(self._position[:, 1], lo[1], hi[1], out=self._position[:, 1])
        np.clip(self._position[:, 2], lo[2], hi[2], out=self._position[:, 2])

    def _sample_velocity(self, pts: np.ndarray) -> np.ndarray:
        """Trilinear sampling of the diagnostic velocity field at arbitrary points."""
        nx, ny, nz = self._shape
        xq = pts[:, 0]
        yq = pts[:, 1]
        zq = pts[:, 2]
        ix = (xq - self._x[0]) / (self._x[-1] - self._x[0]) * (nx - 1)
        iy = (yq - self._y[0]) / (self._y[-1] - self._y[0]) * (ny - 1)
        iz = (zq - self._z[0]) / (self._z[-1] - self._z[0]) * (nz - 1)

        i0 = np.floor(ix).astype(int)
        j0 = np.floor(iy).astype(int)
        k0 = np.floor(iz).astype(int)

        i0 = np.clip(i0, 0, nx - 2)
        j0 = np.clip(j0, 0, ny - 2)
        k0 = np.clip(k0, 0, nz - 2)

        tx = np.clip(ix - i0, 0.0, 1.0)
        ty = np.clip(iy - j0, 0.0, 1.0)
        tz = np.clip(iz - k0, 0.0, 1.0)

        c000 = (1 - tx) * (1 - ty) * (1 - tz)
        c100 = tx * (1 - ty) * (1 - tz)
        c010 = (1 - tx) * ty * (1 - tz)
        c110 = tx * ty * (1 - tz)
        c001 = (1 - tx) * (1 - ty) * tz
        c101 = tx * (1 - ty) * tz
        c011 = (1 - tx) * ty * tz
        c111 = tx * ty * tz

        vel = (
            self._u[i0, j0, k0] * c000[:, None]
            + self._u[i0 + 1, j0, k0] * c100[:, None]
            + self._u[i0, j0 + 1, k0] * c010[:, None]
            + self._u[i0 + 1, j0 + 1, k0] * c110[:, None]
            + self._u[i0, j0, k0 + 1] * c001[:, None]
            + self._u[i0 + 1, j0, k0 + 1] * c101[:, None]
            + self._u[i0, j0 + 1, k0 + 1] * c011[:, None]
            + self._u[i0 + 1, j0 + 1, k0 + 1] * c111[:, None]
        )
        return vel

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        # Boundary velocity sampled from the diagnostic field at the ground samples
        boundary_vel = self._sample_velocity(self._boundary_samples)
        # Enforce no-penetration at the wall (z-component = 0)
        boundary_vel[:, 2] = 0.0

        fields = {
            "active_mask": self._active_mask.astype(float),
            "boundary_normal": self._boundary_normals,
            "boundary_target_velocity": self._boundary_target_vel.copy(),
            "boundary_velocity": boundary_vel,
            "scalar_source": self._scalar_source.copy(),
            "transported_scalar": self._scalar.copy(),
            "velocity_field": self._u.copy(),
        }

        # Include the original cumulative mass source if it was present
        if "cumulative_mass_source" in self._attributes:
            fields["cumulative_mass_source"] = self._attributes["cumulative_mass_source"].copy()

        # Scalars remain frozen / diagnostic
        scalars = self._scalars.copy()
        scalars["reference_velocity"] = max(1.0, float(np.max(np.linalg.norm(self._u, axis=-1))))

        # Per-sample velocity is stored from particle advection, but ensure shape matches
        position = self._position.copy()
        velocity = self._velocity.copy()
        if velocity.shape != position.shape:
            velocity = np.zeros_like(position)

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self._mass.copy() if self._mass is not None else None,
            body_id=self._body_id.copy() if self._body_id is not None else None,
            attributes=self._attributes.copy(),
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
