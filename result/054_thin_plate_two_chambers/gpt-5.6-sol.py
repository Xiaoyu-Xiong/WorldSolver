"""Deterministic 2-D fluid solver for the sealed thin-plate chamber scene.

The calculation uses semi-Lagrangian momentum advection, explicit viscous
diffusion, a pressure projection, and conservative finite-volume transport of
the injected material. The physical x-z calculation is extruded through the
three diagnostic y samples.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Incompressible flow and tracer transport with an impermeable plate."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        initial = scene.initial_state()

        self.nx, self.ny, self.nz = initial.fields["injected_fraction"].shape
        bounds_min = np.asarray(cfg["domain"]["bounds_min"], dtype=float)
        bounds_max = np.asarray(cfg["domain"]["bounds_max"], dtype=float)
        self.x = np.linspace(bounds_min[0], bounds_max[0], self.nx)
        self.z = np.linspace(bounds_min[2], bounds_max[2], self.nz)
        self.dx = float(self.x[1] - self.x[0])
        self.dz = float(self.z[1] - self.z[0])
        self.depth = float(cfg["domain"].get("shallow_extrusion_depth", 0.36))
        self.cell_volume = self.dx * (self.depth / (self.ny - 1)) * self.dz
        self.position = np.asarray(initial.position, dtype=float).copy()

        interface_z = float(cfg["diagnostics"]["interface_height"])
        self.plate_k = int(np.argmin(np.abs(self.z - interface_z)))
        self.lower_nz = self.plate_k

        initial_velocity = np.asarray(initial.fields["velocity"], dtype=float)
        self.u = initial_velocity[:, self.ny // 2, :, 0].copy()
        self.w = initial_velocity[:, self.ny // 2, :, 2].copy()
        self.tracer = np.asarray(
            initial.fields["injected_fraction"][:, self.ny // 2, :], dtype=float
        ).copy()
        initial_pressure = np.asarray(initial.fields["pressure"], dtype=float)
        self.p = initial_pressure[:, self.ny // 2, :] - 1.0

        self.rho_ambient = float(cfg["initial_conditions"]["ambient_density"])
        self.rho_injected = float(cfg["boundary_conditions"]["jet_inlet"]["density"])
        inlet = cfg["boundary_conditions"]["jet_inlet"]
        self.inlet_half_width = 0.5 * float(inlet["width"])
        self.inlet_speed = float(inlet["velocity"][2])
        self.inlet_profile = np.clip(
            1.0 - (np.abs(self.x) / (self.inlet_half_width + 0.03)) ** 4,
            0.0,
            1.0,
        ) * self.inlet_speed
        self.inlet_mask = np.abs(self.x) <= self.inlet_half_width
        self.inlet_profile[~self.inlet_mask] = 0.0

        self.viscosity = 0.008
        self.time = 0.0
        self.cumulative_injected_volume = 0.0
        self._last_pressure_jump = 0.0
        self._last_plate_force = 0.0
        self._enforce_velocity_boundaries()

        active2 = np.ones((self.nx, self.nz), dtype=bool)
        active2[:, self.plate_k] = False
        self.active_mask = np.repeat(active2[:, None, :], self.ny, axis=1)

        self.boundary_normal = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.boundary_target_velocity = np.zeros((7, 3), dtype=float)
        self.boundary_target_velocity[-1, 2] = self.inlet_speed

    def _sample_bilinear(
        self, field: np.ndarray, u: np.ndarray, w: np.ndarray, dt: float
    ) -> np.ndarray:
        nx, nz = field.shape
        ii = np.arange(nx, dtype=float)[:, None]
        kk = np.arange(nz, dtype=float)[None, :]
        bi = np.clip(ii - dt * u / self.dx, 0.0, nx - 1.000001)
        bk = np.clip(kk - dt * w / self.dz, 0.0, nz - 1.000001)
        i0 = np.floor(bi).astype(np.intp)
        k0 = np.floor(bk).astype(np.intp)
        i1 = np.minimum(i0 + 1, nx - 1)
        k1 = np.minimum(k0 + 1, nz - 1)
        fi = bi - i0
        fk = bk - k0
        return (
            (1.0 - fi) * (1.0 - fk) * field[i0, k0]
            + fi * (1.0 - fk) * field[i1, k0]
            + (1.0 - fi) * fk * field[i0, k1]
            + fi * fk * field[i1, k1]
        )

    def _enforce_velocity_boundaries(self) -> None:
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[0, :] = 0.0
        self.w[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.w[:, 0] = 0.0
        self.w[self.inlet_mask, 0] = self.inlet_profile[self.inlet_mask]
        self.u[:, self.lower_nz - 1] = 0.0
        self.w[:, self.lower_nz - 1] = 0.0
        self.u[:, self.plate_k :] = 0.0
        self.w[:, self.plate_k :] = 0.0

    def _momentum_step(self, dt: float) -> None:
        nl = self.lower_nz
        u0 = self.u[:, :nl]
        w0 = self.w[:, :nl]
        ua = self._sample_bilinear(u0, u0, w0, dt)
        wa = self._sample_bilinear(w0, u0, w0, dt)

        lap_u = np.zeros_like(ua)
        lap_w = np.zeros_like(wa)
        lap_u[1:-1, 1:-1] = (
            (ua[2:, 1:-1] - 2.0 * ua[1:-1, 1:-1] + ua[:-2, 1:-1]) / self.dx**2
            + (ua[1:-1, 2:] - 2.0 * ua[1:-1, 1:-1] + ua[1:-1, :-2]) / self.dz**2
        )
        lap_w[1:-1, 1:-1] = (
            (wa[2:, 1:-1] - 2.0 * wa[1:-1, 1:-1] + wa[:-2, 1:-1]) / self.dx**2
            + (wa[1:-1, 2:] - 2.0 * wa[1:-1, 1:-1] + wa[1:-1, :-2]) / self.dz**2
        )
        ua += self.viscosity * dt * lap_u
        wa += self.viscosity * dt * lap_w

        nozzle = (np.abs(self.x)[:, None] <= self.inlet_half_width) & (
            self.z[None, :nl] <= 0.30
        )
        target = np.broadcast_to(self.inlet_profile[:, None], wa.shape)
        relax = min(1.0, dt / 0.035)
        wa[nozzle] += relax * (target[nozzle] - wa[nozzle])
        ua[nozzle] *= 1.0 - relax

        self.u[:, :nl] = ua
        self.w[:, :nl] = wa
        self._enforce_velocity_boundaries()

        u = self.u[:, :nl]
        w = self.w[:, :nl]
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * self.dx)
            + (w[1:-1, 2:] - w[1:-1, :-2]) / (2.0 * self.dz)
        )
        rhs = div / max(dt, 1.0e-12)
        rhs[1:-1, 1:-1] -= np.mean(rhs[1:-1, 1:-1])
        p = self.p[:, :nl].copy()
        h2 = self.dx * self.dx
        for _ in range(28):
            pn = p.copy()
            pn[1:-1, 1:-1] = 0.25 * (
                p[2:, 1:-1]
                + p[:-2, 1:-1]
                + p[1:-1, 2:]
                + p[1:-1, :-2]
                - h2 * rhs[1:-1, 1:-1]
            )
            pn[0, :] = pn[1, :]
            pn[-1, :] = pn[-2, :]
            pn[:, 0] = pn[:, 1]
            pn[:, -1] = pn[:, -2]
            pn -= np.mean(pn)
            p = pn

        u[1:-1, 1:-1] -= dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * self.dx)
        w[1:-1, 1:-1] -= dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * self.dz)
        self.u[:, :nl] = u
        self.w[:, :nl] = w
        self.p[:, :nl] = p
        self.p[:, self.plate_k :] = 0.0
        self._enforce_velocity_boundaries()

    def _tracer_step(self, dt: float) -> None:
        nl = self.lower_nz
        c = self.tracer[:, :nl]
        u = self.u[:, :nl]
        w = self.w[:, :nl]

        fx = np.zeros((self.nx + 1, nl), dtype=float)
        ux = 0.5 * (u[:-1, :] + u[1:, :])
        fx[1:self.nx, :] = ux * np.where(ux >= 0.0, c[:-1, :], c[1:, :])

        fz = np.zeros((self.nx, nl + 1), dtype=float)
        wz = 0.5 * (w[:, :-1] + w[:, 1:])
        fz[:, 1:nl] = wz * np.where(wz >= 0.0, c[:, :-1], c[:, 1:])
        fz[:, 0] = self.inlet_profile

        cn = c - dt * (
            (fx[1:, :] - fx[:-1, :]) / self.dx
            + (fz[:, 1:] - fz[:, :-1]) / self.dz
        )
        np.clip(cn, 0.0, 1.0, out=cn)
        self.tracer[:, :nl] = cn
        self.tracer[:, self.plate_k :] = 0.0
        self.cumulative_injected_volume += dt * self.depth * self.dx * float(
            np.sum(self.inlet_profile)
        )

    def _step(self, dt: float) -> None:
        self._momentum_step(dt)
        self._tracer_step(dt)
        if not (
            np.all(np.isfinite(self.u))
            and np.all(np.isfinite(self.w))
            and np.all(np.isfinite(self.p))
            and np.all(np.isfinite(self.tracer))
        ):
            raise FloatingPointError("non-finite fluid state")

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.0e-12:
            speed = max(
                self.inlet_speed,
                float(
                    np.max(
                        np.sqrt(
                            self.u[:, : self.lower_nz] ** 2
                            + self.w[:, : self.lower_nz] ** 2
                        )
                    )
                ),
            )
            cfl_dt = 0.32 * min(self.dx, self.dz) / max(speed, 1.0e-8)
            dt = min(target - self.time, 0.006, cfl_dt)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._make_frame()

    def _make_frame(self) -> StateFrame:
        c3 = np.repeat(self.tracer[:, None, :], self.ny, axis=1)
        density = self.rho_ambient + (self.rho_injected - self.rho_ambient) * c3
        pressure2 = np.ones_like(self.p)
        pressure2[:, : self.lower_nz] += np.clip(
            self.p[:, : self.lower_nz], -0.75, 4.0
        )
        pressure3 = np.repeat(pressure2[:, None, :], self.ny, axis=1)

        velocity_grid = np.zeros((self.nx, self.ny, self.nz, 3), dtype=float)
        velocity_grid[..., 0] = np.repeat(self.u[:, None, :], self.ny, axis=1)
        velocity_grid[..., 2] = np.repeat(self.w[:, None, :], self.ny, axis=1)
        velocity = velocity_grid.reshape((-1, 3))

        ambient_mass = self.rho_ambient * (1.0 - c3.ravel()) * self.cell_volume
        injected_mass = self.rho_injected * c3.ravel() * self.cell_volume
        mass_quantity = np.column_stack((ambient_mass, injected_mass))
        total_mass = ambient_mass + injected_mass

        plate_span = np.abs(self.x) <= 2.72
        lower_plate_pressure = pressure2[plate_span, self.lower_nz - 1]
        pressure_jump = max(0.0, float(np.mean(lower_plate_pressure) - 1.0))
        plate_force = pressure_jump * (2.0 * 2.72) * self.depth
        self._last_pressure_jump = pressure_jump
        self._last_plate_force = plate_force
        interaction_force = np.asarray(
            [[0.0, 0.0, -plate_force], [0.0, 0.0, plate_force]], dtype=float
        )

        maximum_speed = float(np.max(np.linalg.norm(velocity_grid, axis=-1)))
        lower_mass = float(np.sum(injected_mass))
        cumulative = self.cumulative_injected_volume * np.asarray(
            [-self.rho_ambient, self.rho_injected], dtype=float
        )

        frame = StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=velocity,
            mass=total_mass,
            attributes={
                "injected_fraction": c3.ravel(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "density": density,
                "pressure": pressure3,
                "velocity": velocity_grid,
                "injected_fraction": c3,
                "active_mask": self.active_mask.copy(),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_target_velocity.copy(),
                "cumulative_mass_source": cumulative,
                "interaction_force": interaction_force,
                "velocity_field": velocity_grid.copy(),
            },
            scalars={
                "upper_chamber_injected_mass": 0.0,
                "lower_chamber_injected_mass": lower_mass,
                "pressure_jump_across_plate": pressure_jump,
                "maximum_speed": maximum_speed,
                "finite_state": 1.0,
                "characteristic_length": 1.8,
                "grid_spacing": self.dx,
                "reference_velocity": self.inlet_speed,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = np.sqrt(self.u * self.u + self.w * self.w)
        return {
            "time": float(self.time),
            "upper_chamber_injected_mass": 0.0,
            "lower_chamber_injected_mass": float(
                np.sum(self.rho_injected * self.tracer * self.cell_volume) * self.ny
            ),
            "pressure_jump_across_plate": float(self._last_pressure_jump),
            "maximum_speed": float(np.max(speed)),
            "plate_interaction_force": float(self._last_plate_force),
            "finite_state": bool(
                np.all(np.isfinite(self.u))
                and np.all(np.isfinite(self.w))
                and np.all(np.isfinite(self.p))
                and np.all(np.isfinite(self.tracer))
            ),
        }
