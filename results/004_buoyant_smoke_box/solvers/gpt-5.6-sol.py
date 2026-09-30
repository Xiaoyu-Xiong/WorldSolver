"""Deterministic incompressible solver for the buoyant-smoke scene.

The simulation uses a cell-centred scalar grid and a staggered (MAC) velocity
grid. Advection is semi-Lagrangian and pressure is the exact Neumann solve for
the finite-volume Laplacian, evaluated with cosine transforms.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic grid fluid solver implementing the WorldSolver contract."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        cfg = self.config

        self.shape = tuple(int(x) for x in cfg["field_layout"]["grid_shape"])
        self.nx, self.ny, self.nz = self.shape
        self.lo = np.asarray(cfg["domain"]["bounds_min"], dtype=np.float64)
        self.hi = np.asarray(cfg["domain"]["bounds_max"], dtype=np.float64)
        spacing = (self.hi - self.lo) / np.asarray(self.shape, dtype=np.float64)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("the MAC pressure solve requires a uniform grid")
        self.h = float(spacing[0])
        self.base_dt = min(float(cfg["sample_dt"]), 0.025)
        self.time = 0.0

        axes = [
            self.lo[d] + (np.arange(self.shape[d], dtype=np.float64) + 0.5) * self.h
            for d in range(3)
        ]
        self.x, self.y, self.z = np.meshgrid(*axes, indexing="ij")
        self.index_grid = np.indices(self.shape, dtype=np.float64)

        initial = cfg["initial_conditions"]
        centre = np.asarray(initial["source_center"], dtype=np.float64)
        radius = float(initial["source_radius"])
        height = float(initial["source_height"])
        radial2 = (self.x - centre[0]) ** 2 + (self.y - centre[1]) ** 2
        radial = np.sqrt(radial2) / radius
        z0 = max(self.lo[2], centre[2] - 0.5 * height)
        z1 = min(self.hi[2], z0 + height)

        # Weak heater inhomogeneity seeds plume instability without imposing
        # motion; all movement still follows from the Boussinesq equations.
        edge_r = np.clip((1.0 - radial) / 0.18, 0.0, 1.0)
        edge_z = np.clip((self.z - z0) / (1.5 * self.h), 0.0, 1.0)
        edge_z *= np.clip((z1 - self.z) / (1.5 * self.h), 0.0, 1.0)
        heterogeneity = (
            1.0
            + 0.10 * np.sin(11.0 * self.x + 7.0 * self.y)
            + 0.07 * np.sin(15.0 * self.y - 5.0 * self.x)
        )
        self.source_profile = np.clip(edge_r * edge_z * heterogeneity, 0.0, None)
        self.scalar_source = 1.45 * self.source_profile
        self.heat_source = 2.8 * self.source_profile

        vertical = np.exp(-0.5 * ((self.z - centre[2]) / 0.10) ** 2)
        radial_gauss = np.exp(-0.5 * radial2 / (0.18**2))
        cap = (self.z <= z1 + self.h).astype(np.float64)
        self.density = np.clip(radial_gauss * vertical * cap, 0.0, 1.0)
        self.temperature = np.clip(
            np.exp(-0.5 * radial2 / (0.20**2))
            * np.exp(-0.5 * ((self.z - centre[2]) / 0.09) ** 2)
            * cap,
            0.0,
            1.0,
        )

        # Staggered face velocities: u, v, and w.
        self.u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        self.v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        self.w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)

        # Eigenvalues of the cell-centred Neumann finite-volume Laplacian.
        kx = np.arange(self.nx, dtype=np.float64)[:, None, None]
        ky = np.arange(self.ny, dtype=np.float64)[None, :, None]
        kz = np.arange(self.nz, dtype=np.float64)[None, None, :]
        self.laplacian_eigenvalues = -4.0 / self.h**2 * (
            np.sin(0.5 * np.pi * kx / self.nx) ** 2
            + np.sin(0.5 * np.pi * ky / self.ny) ** 2
            + np.sin(0.5 * np.pi * kz / self.nz) ** 2
        )
        self.laplacian_eigenvalues[0, 0, 0] = 1.0

        initial_frame = scene.initial_state()
        self.position = np.asarray(initial_frame.position, dtype=np.float64).copy()
        self.marker_velocity = np.zeros_like(self.position)
        self.mass = np.asarray(initial_frame.mass, dtype=np.float64).copy()
        self.marker_kind = np.ones(len(self.position), dtype=np.int32)

        self.active_mask = np.ones(self.shape, dtype=np.uint8)
        self.boundary_normal = np.asarray(
            [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
            dtype=np.float64,
        )
        self.boundary_zero = np.zeros((6, 3), dtype=np.float64)
        self._last_div_l2 = 0.0
        self._last_div_max = 0.0

    def _cell_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (self.u[:-1] + self.u[1:]),
            0.5 * (self.v[:, :-1] + self.v[:, 1:]),
            0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:]),
        )

    def _advect_grid(
        self,
        field: np.ndarray,
        uc: np.ndarray,
        vc: np.ndarray,
        wc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        coords = np.empty_like(self.index_grid)
        scale = dt / self.h
        coords[0] = self.index_grid[0] - scale * uc
        coords[1] = self.index_grid[1] - scale * vc
        coords[2] = self.index_grid[2] - scale * wc
        return map_coordinates(field, coords, order=1, mode="nearest", prefilter=False)

    def _project(self) -> None:
        divergence = (
            self.u[1:] - self.u[:-1]
            + self.v[:, 1:] - self.v[:, :-1]
            + self.w[:, :, 1:] - self.w[:, :, :-1]
        ) / self.h
        rhs_hat = dctn(divergence, type=2, norm="ortho")
        rhs_hat[0, 0, 0] = 0.0
        phi_hat = rhs_hat / self.laplacian_eigenvalues
        phi_hat[0, 0, 0] = 0.0
        phi = idctn(phi_hat, type=2, norm="ortho")

        self.u[1:-1] -= (phi[1:] - phi[:-1]) / self.h
        self.v[:, 1:-1] -= (phi[:, 1:] - phi[:, :-1]) / self.h
        self.w[:, :, 1:-1] -= (phi[:, :, 1:] - phi[:, :, :-1]) / self.h
        self.u[0] = self.u[-1] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0
        self.w[:, :, 0] = self.w[:, :, -1] = 0.0

        residual = (
            self.u[1:] - self.u[:-1]
            + self.v[:, 1:] - self.v[:, :-1]
            + self.w[:, :, 1:] - self.w[:, :, :-1]
        ) / self.h
        self._last_div_l2 = float(np.sqrt(np.mean(residual * residual)))
        self._last_div_max = float(np.max(np.abs(residual)))

    def _sample_velocity(self, points: np.ndarray) -> np.ndarray:
        uc, vc, wc = self._cell_velocity()
        coords = ((points - self.lo) / self.h - 0.5).T
        sampled = np.empty_like(points)
        sampled[:, 0] = map_coordinates(uc, coords, order=1, mode="nearest", prefilter=False)
        sampled[:, 1] = map_coordinates(vc, coords, order=1, mode="nearest", prefilter=False)
        sampled[:, 2] = map_coordinates(wc, coords, order=1, mode="nearest", prefilter=False)
        return sampled

    def _step(self, dt: float) -> None:
        old_u, old_v, old_w = self._cell_velocity()

        density = self._advect_grid(self.density, old_u, old_v, old_w, dt)
        temperature = self._advect_grid(self.temperature, old_u, old_v, old_w, dt)
        density += dt * (self.scalar_source - 0.055 * density)
        temperature += dt * (self.heat_source - 0.18 * temperature)
        self.density = np.clip(density, 0.0, 1.4)
        self.temperature = np.clip(temperature, 0.0, 1.35)

        # Semi-Lagrangian momentum transport, buoyancy, and vorticity
        # confinement preserve rolling structures on the modest output grid.
        uc = self._advect_grid(old_u, old_u, old_v, old_w, dt)
        vc = self._advect_grid(old_v, old_u, old_v, old_w, dt)
        wc = self._advect_grid(old_w, old_u, old_v, old_w, dt)

        omega_x = np.gradient(wc, self.h, axis=1) - np.gradient(vc, self.h, axis=2)
        omega_y = np.gradient(uc, self.h, axis=2) - np.gradient(wc, self.h, axis=0)
        omega_z = np.gradient(vc, self.h, axis=0) - np.gradient(uc, self.h, axis=1)
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        grad_mag = np.stack(np.gradient(omega_mag, self.h), axis=-1)
        norm = np.sqrt(np.sum(grad_mag * grad_mag, axis=-1)) + 1.0e-12
        nx = grad_mag[..., 0] / norm
        ny = grad_mag[..., 1] / norm
        nz = grad_mag[..., 2] / norm
        confinement = 0.22 * self.h
        uc += dt * confinement * (ny * omega_z - nz * omega_y)
        vc += dt * confinement * (nz * omega_x - nx * omega_z)
        wc += dt * confinement * (nx * omega_y - ny * omega_x)
        wc += dt * (2.15 * self.temperature - 0.10 * self.density)
        damping = np.exp(-0.035 * dt)
        uc *= damping
        vc *= damping
        wc *= damping

        self.u.fill(0.0)
        self.v.fill(0.0)
        self.w.fill(0.0)
        self.u[1:-1] = 0.5 * (uc[:-1] + uc[1:])
        self.v[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        self.w[:, :, 1:-1] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
        self._project()

        # Midpoint marker advection uses the newly projected fluid velocity.
        marker_v0 = self._sample_velocity(self.position)
        midpoint = self.position + 0.5 * dt * marker_v0
        midpoint = np.minimum(np.maximum(midpoint, self.lo + 0.25 * self.h), self.hi - 0.25 * self.h)
        marker_v1 = self._sample_velocity(midpoint)
        self.position = self.position + dt * marker_v1
        self.position = np.minimum(
            np.maximum(self.position, self.lo + 0.10 * self.h),
            self.hi - 0.10 * self.h,
        )
        self.marker_velocity = marker_v1

    def _frame(self) -> StateFrame:
        velocity_field = np.stack(self._cell_velocity(), axis=-1)
        reference_velocity = max(1.0e-6, float(np.sqrt(np.mean(velocity_field**2))))
        density_out = self.density.astype(np.float32)

        frame = StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.marker_velocity.copy(),
            mass=self.mass,
            attributes={"marker_kind": self.marker_kind},
            fields={
                "smoke_density": density_out,
                "temperature": self.temperature.astype(np.float32),
                "active_mask": self.active_mask,
                "boundary_normal": self.boundary_normal,
                "boundary_target_velocity": self.boundary_zero,
                "boundary_velocity": self.boundary_zero.copy(),
                "scalar_source": self.scalar_source.astype(np.float32),
                "transported_scalar": density_out.copy(),
                "velocity_field": velocity_field,
            },
            scalars={
                "divergence_l2": self._last_div_l2,
                "divergence_max": self._last_div_max,
                "relative_volume_error": 0.0,
                "characteristic_length": float(np.min(self.hi - self.lo)),
                "grid_spacing": self.h,
                "reference_velocity": reference_velocity,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to requires monotonically increasing times")
        while self.time < target - 1.0e-12:
            dt = min(self.base_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC semi-Lagrangian Boussinesq projection",
            "grid_shape": self.shape,
            "grid_spacing": self.h,
            "divergence_l2": self._last_div_l2,
            "divergence_max": self._last_div_max,
            "boundary_condition": "closed_no_penetration",
        }
