"""3D buoyant smoke solver using a stable incompressible Navier-Stokes scheme.

The fluid is advanced on a collocated cell-centered grid using semi-Lagrangian
advection, explicit buoyancy from a transported temperature field, and a fast
pressure projection via 3D DCT. Passive markers are advected in the resulting
velocity field to visualise the smoke plume.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        # Timing configuration.
        self.sample_dt: float = float(self.config["sample_dt"])
        self.sub_dt: float = self.sample_dt / 2.0

        # Grid geometry.
        self.shape: tuple[int, int, int] = tuple(
            int(v) for v in self.config["field_layout"]["grid_shape"]
        )
        self.bounds_min = np.asarray(
            self.config["domain"]["bounds_min"], dtype=np.float64
        )
        self.bounds_max = np.asarray(
            self.config["domain"]["bounds_max"], dtype=np.float64
        )
        self.domain_size = self.bounds_max - self.bounds_min
        self.dx = self.domain_size / np.array(self.shape, dtype=np.float64)
        self.dx_scalar = float(np.mean(self.dx))

        # Cell-center coordinates.
        axes = [
            np.linspace(lo + d / 2.0, hi - d / 2.0, n)
            for lo, hi, d, n in zip(
                self.bounds_min, self.bounds_max, self.dx, self.shape
            )
        ]
        self.X, self.Y, self.Z = np.meshgrid(*axes, indexing="ij")

        # Extract initial state.
        init = scene.initial_state()
        self.position = np.asarray(init.position, dtype=np.float64).copy()
        self.velocity_particles = (
            np.asarray(init.velocity, dtype=np.float64).copy()
        )
        self.mass = np.asarray(init.mass, dtype=np.float64).copy()
        self.attributes = {
            k: np.asarray(v).copy() for k, v in init.attributes.items()
        }

        self.u = np.asarray(
            init.fields["velocity_field"], dtype=np.float64
        ).copy()
        self.smoke = np.asarray(
            init.fields["smoke_density"], dtype=np.float64
        ).copy()
        self.temperature = np.asarray(
            init.fields.get("temperature", np.zeros(self.shape, dtype=np.float64)),
            dtype=np.float64,
        ).copy()

        # Source/forcing parameters.
        ic = self.config["initial_conditions"]
        self.source_center = np.asarray(ic["source_center"], dtype=np.float64)
        self.source_radius = float(ic["source_radius"])
        self.source_height = float(ic["source_height"])
        self.source_temp = float(ic["source_temperature"])
        self.ambient_temp = float(ic["ambient_temperature"])
        # Store reference to initial smoke for source-rate consistency.
        self.initial_smoke = self.smoke.copy()

        # Precomputed helpers.
        self._precompute_dct()
        self._precompute_source_mask()
        self._precompute_active_mask()
        self._precompute_boundary_arrays()

        # Deterministic perturbation to break axisymmetry and seed vortices.
        self._add_initial_perturbation()

        self.t: float = 0.0

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    def _precompute_dct(self) -> None:
        nx, ny, nz = self.shape
        dx, dy, dz = self.dx
        kx = np.arange(nx)
        ky = np.arange(ny)
        kz = np.arange(nz)
        lam_x = (
            -4.0 * np.sin(np.pi * kx / (2.0 * nx)) ** 2 / dx ** 2
        )
        lam_y = (
            -4.0 * np.sin(np.pi * ky / (2.0 * ny)) ** 2 / dy ** 2
        )
        lam_z = (
            -4.0 * np.sin(np.pi * kz / (2.0 * nz)) ** 2 / dz ** 2
        )
        self._lam = (
            lam_x[:, None, None]
            + lam_y[None, :, None]
            + lam_z[None, None, :]
        )
        # Regularise the constant-pressure nullspace.
        self._lam[0, 0, 0] = 1.0

    def _precompute_source_mask(self) -> None:
        r2 = self.X ** 2 + self.Y ** 2
        self.source_mask = (
            (r2 < self.source_radius ** 2) & (self.Z < self.source_height)
        ).astype(np.float64)

    def _precompute_active_mask(self) -> None:
        self.active_mask = np.ones(self.shape, dtype=np.float64)

    def _precompute_boundary_arrays(self) -> None:
        # Six box faces.
        self.boundary_normal = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.boundary_target_velocity = np.zeros((6, 3), dtype=np.float64)

        # Face-center positions for sampling boundary velocity.
        cx = (self.bounds_min[0] + self.bounds_max[0]) / 2.0
        cy = (self.bounds_min[1] + self.bounds_max[1]) / 2.0
        cz = (self.bounds_min[2] + self.bounds_max[2]) / 2.0
        self.boundary_center = np.array(
            [
                [self.bounds_min[0], cy, cz],
                [self.bounds_max[0], cy, cz],
                [cx, self.bounds_min[1], cz],
                [cx, self.bounds_max[1], cz],
                [cx, cy, self.bounds_min[2]],
                [cx, cy, self.bounds_max[2]],
            ],
            dtype=np.float64,
        )

    def _add_initial_perturbation(self) -> None:
        # Small deterministic velocity noise to break perfect axisymmetry.
        rng = np.random.default_rng(self.seed)
        self.u += rng.normal(0.0, 0.003, self.u.shape)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    def _apply_boundary(self, u: np.ndarray) -> None:
        # Closed no-penetration: zero the normal component on each wall.
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, 2] = 0.0

    # ------------------------------------------------------------------
    # Differential operators
    # ------------------------------------------------------------------
    def _divergence(self, u: np.ndarray) -> np.ndarray:
        """Divergence consistent with the Neumann-pressure projection."""
        nx, ny, nz = self.shape
        div = np.zeros(self.shape, dtype=np.float64)

        for c in range(3):
            u_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
            u_ext[1:-1, 1:-1, 1:-1] = u[..., c]

            # x-boundaries: component 0 is normal; others are tangential.
            if c == 0:
                u_ext[0, 1:-1, 1:-1] = -u[0, :, :, c]
                u_ext[-1, 1:-1, 1:-1] = -u[-1, :, :, c]
            else:
                u_ext[0, 1:-1, 1:-1] = u[0, :, :, c]
                u_ext[-1, 1:-1, 1:-1] = u[-1, :, :, c]

            # y-boundaries.
            if c == 1:
                u_ext[1:-1, 0, 1:-1] = -u[:, 0, :, c]
                u_ext[1:-1, -1, 1:-1] = -u[:, -1, :, c]
            else:
                u_ext[1:-1, 0, 1:-1] = u[:, 0, :, c]
                u_ext[1:-1, -1, 1:-1] = u[:, -1, :, c]

            # z-boundaries.
            if c == 2:
                u_ext[1:-1, 1:-1, 0] = -u[:, :, 0, c]
                u_ext[1:-1, 1:-1, -1] = -u[:, :, -1, c]
            else:
                u_ext[1:-1, 1:-1, 0] = u[:, :, 0, c]
                u_ext[1:-1, 1:-1, -1] = u[:, :, -1, c]

            dudc = (
                u_ext[2:, 1:-1, 1:-1] - u_ext[:-2, 1:-1, 1:-1]
            ) / (2.0 * self.dx[c])
            div += dudc

        return div

    def _gradient(self, p: np.ndarray) -> np.ndarray:
        """Gradient of pressure with zero Neumann (wall) boundary."""
        nx, ny, nz = self.shape
        grad = np.zeros(self.shape + (3,), dtype=np.float64)

        p_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
        p_ext[1:-1, 1:-1, 1:-1] = p
        # Neumann ghost: mirror pressure.
        p_ext[0, 1:-1, 1:-1] = p[0, :, :]
        p_ext[-1, 1:-1, 1:-1] = p[-1, :, :]
        p_ext[1:-1, 0, 1:-1] = p[:, 0, :]
        p_ext[1:-1, -1, 1:-1] = p[:, -1, :]
        p_ext[1:-1, 1:-1, 0] = p[:, :, 0]
        p_ext[1:-1, 1:-1, -1] = p[:, :, -1]

        grad[..., 0] = (
            p_ext[2:, 1:-1, 1:-1] - p_ext[:-2, 1:-1, 1:-1]
        ) / (2.0 * self.dx[0])
        grad[..., 1] = (
            p_ext[1:-1, 2:, 1:-1] - p_ext[1:-1, :-2, 1:-1]
        ) / (2.0 * self.dx[1])
        grad[..., 2] = (
            p_ext[1:-1, 1:-1, 2:] - p_ext[1:-1, 1:-1, :-2]
        ) / (2.0 * self.dx[2])

        return grad

    def _solve_poisson(self, rhs: np.ndarray) -> np.ndarray:
        rhs_hat = dctn(rhs, type=2, norm="ortho")
        p_hat = rhs_hat / self._lam
        return idctn(p_hat, type=2, norm="ortho")

    def _project(self, dt: float) -> np.ndarray:
        """Projection step; returns the divergence before correction."""
        self._apply_boundary(self.u)
        div = self._divergence(self.u)
        p = self._solve_poisson(div / dt)
        grad_p = self._gradient(p)
        self.u -= dt * grad_p
        self._apply_boundary(self.u)
        return div

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------
    def _advect_scalar(self, q: np.ndarray, dt: float) -> np.ndarray:
        positions = np.stack(
            [self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1
        )
        u_flat = self.u.reshape(-1, 3)
        back_positions = positions - dt * u_flat
        return self._interpolate_scalar(q, back_positions).reshape(self.shape)

    def _advect_velocity(self, dt: float) -> np.ndarray:
        positions = np.stack(
            [self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1
        )
        u_flat = self.u.reshape(-1, 3)
        back_positions = positions - dt * u_flat
        new_u = np.zeros_like(self.u)
        for c in range(3):
            new_u[..., c] = self._interpolate_scalar(
                self.u[..., c], back_positions
            ).reshape(self.shape)
        return new_u

    def _interpolate_scalar(
        self, field: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        nx, ny, nz = field.shape
        xi = (positions[:, 0] - self.bounds_min[0]) / self.dx[0] - 0.5
        yi = (positions[:, 1] - self.bounds_min[1]) / self.dx[1] - 0.5
        zi = (positions[:, 2] - self.bounds_min[2]) / self.dx[2] - 0.5

        xi = np.clip(xi, 0.0, nx - 1.0001)
        yi = np.clip(yi, 0.0, ny - 1.0001)
        zi = np.clip(zi, 0.0, nz - 1.0001)

        i0 = xi.astype(np.int64)
        j0 = yi.astype(np.int64)
        k0 = zi.astype(np.int64)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        k1 = np.minimum(k0 + 1, nz - 1)

        wx = xi - i0
        wy = yi - j0
        wz = zi - k0

        c000 = field[i0, j0, k0]
        c100 = field[i1, j0, k0]
        c010 = field[i0, j1, k0]
        c110 = field[i1, j1, k0]
        c001 = field[i0, j0, k1]
        c101 = field[i1, j0, k1]
        c011 = field[i0, j1, k1]
        c111 = field[i1, j1, k1]

        return (
            (1.0 - wx) * (1.0 - wy) * (1.0 - wz) * c000
            + wx * (1.0 - wy) * (1.0 - wz) * c100
            + (1.0 - wx) * wy * (1.0 - wz) * c010
            + wx * wy * (1.0 - wz) * c110
            + (1.0 - wx) * (1.0 - wy) * wz * c001
            + wx * (1.0 - wy) * wz * c101
            + (1.0 - wx) * wy * wz * c011
            + wx * wy * wz * c111
        )

    def _interpolate_vector(self, positions: np.ndarray) -> np.ndarray:
        result = np.zeros((positions.shape[0], 3), dtype=np.float64)
        for c in range(3):
            result[:, c] = self._interpolate_scalar(self.u[..., c], positions)
        return result

    def _advect_particles(self, dt: float) -> None:
        if self.position.size == 0:
            return
        # RK2 advection.
        k1 = self._interpolate_vector(self.position)
        pos_mid = self.position + 0.5 * dt * k1
        k2 = self._interpolate_vector(pos_mid)
        self.position += dt * k2
        # Clamp to the closed box.
        self.position = np.clip(
            self.position,
            self.bounds_min + 1e-9,
            self.bounds_max - 1e-9,
        )

    # ------------------------------------------------------------------
    # Sources / forcing
    # ------------------------------------------------------------------
    def _add_sources(self, dt: float) -> None:
        # Slight deterministic angular perturbation to seed vortices.
        theta = np.arctan2(self.Y, self.X)
        perturb = 1.0 + 0.15 * np.sin(3.0 * theta)

        # Heat source.
        self.temperature += (
            dt
            * 2.0
            * self.source_mask
            * perturb
            * (self.source_temp - self.temperature)
        )
        # Smoke source.
        self.smoke += (
            dt * 1.0 * self.source_mask * perturb * (1.0 - self.smoke)
        )

        np.maximum(self.temperature, 0.0, out=self.temperature)
        np.maximum(self.smoke, 0.0, out=self.smoke)

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # Advect velocity.
        self.u = self._advect_velocity(dt)

        # Buoyancy (hot fluid rises along +z).
        self.u[..., 2] += dt * 2.0 * (self.temperature - self.ambient_temp)

        # Incompressibility projection.
        self._project(dt)

        # Advect smoke and temperature.
        self.smoke = self._advect_scalar(self.smoke, dt)
        self.temperature = self._advect_scalar(self.temperature, dt)

        # Inject heat and smoke near the source.
        self._add_sources(dt)

        # Advect passive markers in the updated velocity field.
        self._advect_particles(dt)

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _compute_diagnostics(self) -> dict[str, float]:
        div = self._divergence(self.u)
        return {
            "divergence_l2": float(np.sqrt(np.mean(div ** 2))),
            "divergence_max": float(np.max(np.abs(div))),
            "relative_volume_error": float(np.abs(np.mean(div))),
        }

    def _make_frame(self, time: float) -> StateFrame:
        # Wall velocities with enforced no-penetration.
        boundary_velocity = self._interpolate_vector(self.boundary_center)
        for i, normal in enumerate(self.boundary_normal):
            vn = np.dot(boundary_velocity[i], normal)
            boundary_velocity[i] -= vn * normal

        diagnostics = self._compute_diagnostics()

        # Heat source term as the prescribed scalar source.
        scalar_source = (
            2.0 * self.source_mask * (self.source_temp - self.temperature)
        )

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self._interpolate_vector(self.position),
            mass=self.mass.copy(),
            attributes={
                k: np.asarray(v).copy() for k, v in self.attributes.items()
            },
            fields={
                "smoke_density": self.smoke.astype(np.float32).copy(),
                "velocity_field": self.u.astype(np.float64).copy(),
                "temperature": self.temperature.astype(np.float32).copy(),
                "active_mask": self.active_mask.copy(),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": boundary_velocity.copy(),
                "scalar_source": scalar_source.copy(),
                "transported_scalar": self.temperature.copy(),
            },
            scalars={
                "divergence_l2": diagnostics["divergence_l2"],
                "divergence_max": diagnostics["divergence_max"],
                "relative_volume_error": diagnostics["relative_volume_error"],
                "characteristic_length": float(self.domain_size[2]),
                "grid_spacing": float(self.dx_scalar),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            raise ValueError("Cannot advance backwards in time")
        while self.t < time - 1e-12:
            dt = min(self.sub_dt, time - self.t)
            self._step(dt)
            self.t += dt
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        diag = self._compute_diagnostics()
        diag["success"] = True
        return diag
