"""Deterministic finite-difference solver for channel flow past a cylinder.

Velocity is advanced on a two-dimensional MAC grid. Advection is
semi-Lagrangian, viscosity is explicit, and incompressibility is enforced by
solving the pressure Poisson equation on the fluid cells. Passive dye is
transported on the cell centers by the computed flow.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Viscous incompressible flow solver on the scene's diagnostic grid."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.h = float(cfg["fluid"]["nominal_particle_spacing"])
        lo = np.asarray(cfg["domain"]["bounds_min"], dtype=float)
        hi = np.asarray(cfg["domain"]["bounds_max"], dtype=float)
        self.xmin, self.ymin = float(lo[0]), float(lo[1])
        self.nx = int(round((hi[0] - lo[0]) / self.h))
        self.ny = int(round((hi[1] - lo[1]) / self.h))
        self.x = self.xmin + (np.arange(self.nx) + 0.5) * self.h
        self.y = self.ymin + (np.arange(self.ny) + 0.5) * self.h
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        cx, cy, _ = cfg["obstacle"]["center"]
        self.cx, self.cy = float(cx), float(cy)
        self.radius = float(cfg["obstacle"]["radius"])
        self.active = (self.xx - self.cx) ** 2 + (self.yy - self.cy) ** 2 > self.radius**2

        # Open MAC faces connect pairs of fluid cells. Domain inlet/outlet
        # faces are handled separately; all other closed faces are no-slip.
        self.open_u = np.zeros((self.ny, self.nx + 1), dtype=bool)
        self.open_u[:, 1:-1] = self.active[:, :-1] & self.active[:, 1:]
        self.open_u[:, 0] = self.active[:, 0]
        self.open_u[:, -1] = self.active[:, -1]
        self.open_v = np.zeros((self.ny + 1, self.nx), dtype=bool)
        self.open_v[1:-1, :] = self.active[:-1, :] & self.active[1:, :]

        centerline = float(cfg["initial_conditions"]["centerline_speed"])
        self.mean_speed = float(cfg["initial_conditions"]["mean_speed"])
        self.nu = float(cfg["fluid"]["kinematic_viscosity"])
        self.rho = float(cfg["fluid"]["density"])
        profile = centerline * np.maximum(0.0, 1.0 - (self.y / 1.5) ** 2)
        self.inlet_profile = profile
        self.u = np.repeat(profile[:, None], self.nx + 1, axis=1)
        self.v = np.zeros((self.ny + 1, self.nx), dtype=float)

        # A weak smooth streamfunction perturbation breaks exact mirror
        # symmetry. It is divergence-free and subsequently evolves under the PDE.
        amp = 0.002 * (1.0 if (int(seed) & 1) == 0 else -1.0)
        xu = self.xmin + np.arange(self.nx + 1) * self.h
        xug, yug = np.meshgrid(xu, self.y)
        ru = ((xug - (self.cx - 0.55)) / 0.75) ** 2 + ((yug - 0.06) / 0.48) ** 2
        self.u += amp * (-2.0 * (yug - 0.06) / 0.48**2) * np.exp(-ru)
        yv = self.ymin + np.arange(self.ny + 1) * self.h
        xvg, yvg = np.meshgrid(self.x, yv)
        rv = ((xvg - (self.cx - 0.55)) / 0.75) ** 2 + ((yvg - 0.06) / 0.48) ** 2
        self.v += amp * (2.0 * (xvg - (self.cx - 0.55)) / 0.75**2) * np.exp(-rv)
        self._apply_velocity_boundaries()

        stripe = np.floor((self.xx + 6.0) / 0.55).astype(np.int32)
        self.dye = np.where(self.xx < -2.55, (stripe & 1) * 2.0 - 1.0, 0.0)
        self.dye *= np.exp(-0.5 * (self.yy / 0.34) ** 2)
        self.dye[~self.active] = 0.0

        self.pressure = np.zeros((self.ny, self.nx), dtype=float)
        self.time = 0.0
        self._build_pressure_stencil()
        self._build_boundary_samples()

        self.position = np.column_stack(
            (self.xx[self.active], self.yy[self.active], np.zeros(self.active.sum()))
        )
        self.mass = np.full(len(self.position), self.rho * self.h**2)
        self.last_pressure_iterations = 0

    def _build_pressure_stencil(self) -> None:
        a = self.active
        self.east = np.zeros_like(a)
        self.west = np.zeros_like(a)
        self.north = np.zeros_like(a)
        self.south = np.zeros_like(a)
        self.east[:, :-1] = a[:, :-1] & a[:, 1:]
        self.west[:, 1:] = a[:, 1:] & a[:, :-1]
        self.north[:-1, :] = a[:-1, :] & a[1:, :]
        self.south[1:, :] = a[1:, :] & a[:-1, :]
        # Pressure is zero at the open outlet, half a cell from its centers.
        self.outlet = np.zeros_like(a)
        self.outlet[:, -1] = a[:, -1]
        self.p_denom = (
            self.east.astype(float)
            + self.west
            + self.north
            + self.south
            + 2.0 * self.outlet
        )
        self.p_denom[self.p_denom == 0.0] = 1.0

    def _build_boundary_samples(self) -> None:
        wall_count = 2 * self.nx
        theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        normals = np.empty((wall_count + len(theta), 2), dtype=float)
        normals[: self.nx] = (0.0, 1.0)
        normals[self.nx : wall_count] = (0.0, -1.0)
        normals[wall_count:, 0] = np.cos(theta)
        normals[wall_count:, 1] = np.sin(theta)
        self.boundary_normal = normals
        self.boundary_target = np.zeros_like(normals)

    @staticmethod
    def _sample_grid(a, xq, yq, x0, y0, h):
        """Bilinear interpolation on a uniform grid, clamped at its edge."""
        fx = np.clip((xq - x0) / h, 0.0, a.shape[1] - 1.000001)
        fy = np.clip((yq - y0) / h, 0.0, a.shape[0] - 1.000001)
        j0 = np.floor(fx).astype(np.int32)
        i0 = np.floor(fy).astype(np.int32)
        j1 = np.minimum(j0 + 1, a.shape[1] - 1)
        i1 = np.minimum(i0 + 1, a.shape[0] - 1)
        tx, ty = fx - j0, fy - i0
        return (
            (1.0 - tx) * (1.0 - ty) * a[i0, j0]
            + tx * (1.0 - ty) * a[i0, j1]
            + (1.0 - tx) * ty * a[i1, j0]
            + tx * ty * a[i1, j1]
        )

    @staticmethod
    def _laplacian(a, h):
        pad = np.pad(a, ((1, 1), (1, 1)), mode="edge")
        return (
            pad[1:-1, 2:]
            + pad[1:-1, :-2]
            + pad[2:, 1:-1]
            + pad[:-2, 1:-1]
            - 4.0 * a
        ) / h**2

    def _cell_velocity(self):
        uc = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        vc = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        uc[~self.active] = 0.0
        vc[~self.active] = 0.0
        return uc, vc

    def _apply_velocity_boundaries(self) -> None:
        self.u[~self.open_u] = 0.0
        self.v[~self.open_v] = 0.0
        self.u[:, 0] = self.inlet_profile
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]

    def _advect_diffuse_velocity(self, dt: float) -> None:
        old_u, old_v = self.u.copy(), self.v.copy()
        uc, vc = self._cell_velocity()

        xu = self.xmin + np.arange(self.nx + 1) * self.h
        xug, yug = np.meshgrid(xu, self.y)
        au = self._sample_grid(uc, xug, yug, self.x[0], self.y[0], self.h)
        bu = self._sample_grid(vc, xug, yug, self.x[0], self.y[0], self.h)
        self.u = self._sample_grid(
            old_u, xug - dt * au, yug - dt * bu, self.xmin, self.y[0], self.h
        )
        self.u += dt * self.nu * self._laplacian(old_u, self.h)

        yv = self.ymin + np.arange(self.ny + 1) * self.h
        xvg, yvg = np.meshgrid(self.x, yv)
        av = self._sample_grid(uc, xvg, yvg, self.x[0], self.y[0], self.h)
        bv = self._sample_grid(vc, xvg, yvg, self.x[0], self.y[0], self.h)
        self.v = self._sample_grid(
            old_v, xvg - dt * av, yvg - dt * bv, self.x[0], self.ymin, self.h
        )
        self.v += dt * self.nu * self._laplacian(old_v, self.h)
        self._apply_velocity_boundaries()

    def _project(self, dt: float) -> None:
        div = (
            self.u[:, 1:] - self.u[:, :-1] + self.v[1:, :] - self.v[:-1, :]
        ) / self.h
        rhs = np.where(self.active, div / dt, 0.0)
        p = self.pressure
        h2 = self.h**2
        for _ in range(55):
            pe = np.zeros_like(p)
            pw = np.zeros_like(p)
            pn = np.zeros_like(p)
            ps = np.zeros_like(p)
            pe[:, :-1] = p[:, 1:]
            pw[:, 1:] = p[:, :-1]
            pn[:-1, :] = p[1:, :]
            ps[1:, :] = p[:-1, :]
            total = pe * self.east + pw * self.west + pn * self.north + ps * self.south
            p = np.where(self.active, (total - rhs * h2) / self.p_denom, 0.0)
        self.pressure = p
        self.last_pressure_iterations = 55

        dp = p[:, 1:] - p[:, :-1]
        self.u[:, 1:-1] -= dt * dp / self.h * self.open_u[:, 1:-1]
        dpv = p[1:, :] - p[:-1, :]
        self.v[1:-1, :] -= dt * dpv / self.h * self.open_v[1:-1, :]
        self.u[:, -1] -= dt * (0.0 - p[:, -1]) * 2.0 / self.h
        self._apply_velocity_boundaries()

    def _advect_dye(self, dt: float, new_time: float) -> None:
        uc, vc = self._cell_velocity()
        old = self.dye
        transported = self._sample_grid(
            old,
            self.xx - dt * uc,
            self.yy - dt * vc,
            self.x[0],
            self.y[0],
            self.h,
        )
        transported += dt * (0.1 * self.nu) * self._laplacian(old, self.h)
        transported[~self.active] = 0.0
        stripe_index = int(np.floor(self.mean_speed * new_time / 0.55))
        stripe_value = 1.0 if (stripe_index & 1) else -1.0
        transported[:, 0] = stripe_value * np.exp(-0.5 * (self.y / 0.34) ** 2)
        self.dye = np.clip(transported, -1.0, 1.0)

    def _step(self, dt: float) -> None:
        self._advect_diffuse_velocity(dt)
        self._project(dt)
        self._advect_dye(dt, self.time + dt)
        self.time += dt

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(0.02, target - self.time))
        self.time = target
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        uc, vc = self._cell_velocity()
        velocity_field = np.stack((uc, vc), axis=-1)
        divergence = (
            self.u[:, 1:] - self.u[:, :-1] + self.v[1:, :] - self.v[:-1, :]
        ) / self.h
        divergence[~self.active] = 0.0
        dvdx = np.gradient(vc, self.h, axis=1, edge_order=1)
        dudy = np.gradient(uc, self.h, axis=0, edge_order=1)
        vorticity = dvdx - dudy
        vorticity[~self.active] = 0.0
        velocity3 = np.zeros((self.ny, self.nx, 3), dtype=float)
        velocity3[..., :2] = velocity_field
        particle_velocity = velocity3[self.active]
        speed = np.sqrt(uc**2 + vc**2)
        zero_source = np.zeros((self.ny, self.nx), dtype=float)

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=particle_velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "dye": self.dye[self.active].copy(),
                "vorticity": vorticity[self.active].copy(),
                "divergence": divergence[self.active].copy(),
                "velocity_magnitude": speed[self.active].copy(),
            },
            fields={
                "active_mask": self.active.astype(np.uint8),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target.copy(),
                "boundary_velocity": np.zeros_like(self.boundary_target),
                "scalar_source": zero_source,
                "transported_scalar": self.dye.copy(),
                "velocity_field": velocity_field,
                "velocity": velocity3,
                "vorticity": vorticity,
                "divergence": divergence,
                "dye": self.dye.copy(),
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": self.h,
                "reference_velocity": self.mean_speed,
                "cylinder_radius": self.radius,
                "mean_inlet_speed": self.mean_speed,
                "expected_strouhal_number": float(
                    self.cfg["diagnostics"]["expected_strouhal_number"]
                ),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC finite-difference pressure projection",
            "grid_shape": [self.ny, self.nx],
            "time": float(self.time),
            "pressure_iterations": int(self.last_pressure_iterations),
            "kinematic_viscosity": self.nu,
        }
