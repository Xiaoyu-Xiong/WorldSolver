"""2D Shallow Water Equations solver for the shallow water obstacle benchmark."""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.current_time = 0.0

        # Grid parameters
        self.nx = 64
        self.ny = 64
        self.dx = 0.0
        self.dy = 0.0
        self.xx = None
        self.yy = None

        # Physics state: h (height), hu (x-momentum), hv (y-momentum)
        self.h = None
        self.hu = None
        self.hv = None

        # Masks and geometry
        self.solid = None
        self.active = None
        self.normals = None

        # Physics constants
        self.g = 9.81
        self.rho = 1000.0
        self.ref_velocity = 0.0
        self.initial_volume = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        cfg = self.config

        self.g = abs(cfg["gravity"][2])
        self.rho = float(cfg["fluid"]["density"])

        domain = cfg["domain"]
        obstacle = cfg["obstacle"]
        initial = cfg["initial_conditions"]

        self.nx, self.ny = domain["grid_shape"]
        low, high = np.asarray(domain["bounds_min"]), np.asarray(domain["bounds_max"])
        self.dx = float((high[0] - low[0]) / self.nx)
        self.dy = float((high[1] - low[1]) / self.ny)

        x = low[0] + (np.arange(self.nx) + 0.5) * self.dx
        y = low[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.xx, self.yy = np.meshgrid(x, y, indexing="ij")

        # Build obstacle mask exactly as in scene.py
        theta = np.arange(8) * np.pi / 4 + obstacle["rotation_radians"]
        vertices = np.asarray(obstacle["center"]) + obstacle["circumradius"] * np.column_stack(
            (np.cos(theta), np.sin(theta))
        )
        self.solid = np.ones((self.nx, self.ny), dtype=bool)
        obstacle_normals = []
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            edge = b - a
            self.solid &= edge[0] * (self.yy - a[1]) - edge[1] * (self.xx - a[0]) >= 0
            obstacle_normals.extend([np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)] * 8)
        self.active = ~self.solid

        # Initial water height
        self.h = np.where(
            self.xx > initial["step_x"],
            initial["high_water_height"],
            initial["low_water_height"]
        ).astype(np.float64)
        self.h[self.solid] = 0.0

        self.hu = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.hv = np.zeros((self.nx, self.ny), dtype=np.float64)

        self.normals = np.vstack(
            (
                np.tile([1.0, 0.0], (self.ny, 1)),
                np.tile([-1.0, 0.0], (self.ny, 1)),
                np.tile([0.0, 1.0], (self.nx, 1)),
                np.tile([0.0, -1.0], (self.nx, 1)),
                obstacle_normals,
            )
        )

        volume = self.h[self.active] * self.dx * self.dy
        self.initial_volume = float(volume.sum())
        self.ref_velocity = float(np.sqrt(self.g * self.h.max()))
        self.current_time = 0.0

    def _minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def _compute_fluxes(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """Compute Kurganov-Levy-Tadmor (KT) fluxes for 2D shallow water equations."""
        nx, ny = self.nx, self.ny
        g = self.g
        eps = 1e-12

        # Safe velocities
        u = np.where(h > eps, hu / np.maximum(h, eps), 0.0)
        v = np.where(h > eps, hv / np.maximum(h, eps), 0.0)

        # -------------------------------------------------------------
        # X-direction fluxes at interfaces (i+1/2, j) for i=0..nx
        # -------------------------------------------------------------
        # Pad ghost cells: solid/wall at x=0 and x=nx-1
        h_pad_x = np.pad(h, ((1, 1), (0, 0)), mode="edge")
        hu_pad_x = np.pad(hu, ((1, 1), (0, 0)), mode="edge")
        hv_pad_x = np.pad(hv, ((1, 1), (0, 0)), mode="edge")
        hu_pad_x[0, :] = -hu[0, :]
        hu_pad_x[-1, :] = -hu[-1, :]

        # Slopes in x
        dh_x = h_pad_x[1:, :] - h_pad_x[:-1, :]
        s_h_x = self._minmod(dh_x[:-1, :], dh_x[1:, :])
        dhu_x = hu_pad_x[1:, :] - hu_pad_x[:-1, :]
        s_hu_x = self._minmod(dhu_x[:-1, :], dhu_x[1:, :])
        dhv_x = hv_pad_x[1:, :] - hv_pad_x[:-1, :]
        s_hv_x = self._minmod(dhv_x[:-1, :], dhv_x[1:, :])

        # States at x-interfaces i+1/2 for i = 0 ... nx (size nx+1, ny)
        h_L_x = h_pad_x[1:-1, :] + 0.5 * s_h_x
        hu_L_x = hu_pad_x[1:-1, :] + 0.5 * s_hu_x
        hv_L_x = hv_pad_x[1:-1, :] + 0.5 * s_hv_x

        h_R_x = h_pad_x[2:, :] - 0.5 * np.pad(s_h_x[1:, :], ((0, 1), (0, 0)), mode="edge")
        hu_R_x = hu_pad_x[2:, :] - 0.5 * np.pad(s_hu_x[1:, :], ((0, 1), (0, 0)), mode="edge")
        hv_R_x = hv_pad_x[2:, :] - 0.5 * np.pad(s_hv_x[1:, :], ((0, 1), (0, 0)), mode="edge")

        # Boundary interfaces at x=0 (i=0 interface) and x=nx (i=nx interface)
        h_L_x[0, :] = h_R_x[0, :]
        hu_L_x[0, :] = -hu_R_x[0, :]
        hv_L_x[0, :] = hv_R_x[0, :]

        h_R_x[-1, :] = h_L_x[-1, :]
        hu_R_x[-1, :] = -hu_L_x[-1, :]
        hv_R_x[-1, :] = hv_L_x[-1, :]

        # Internal obstacle interfaces:
        # Interface (i+1/2, j) is between cell (i-1, j) and cell (i, j) for i in 1..nx-1
        # If cell i-1 is active and cell i is solid, right state reflects left
        # If cell i-1 is solid and cell i is active, left state reflects right
        # If both solid, both zero
        solid_L_x = np.pad(self.solid, ((1, 0), (0, 0)), mode="constant", constant_values=True)
        solid_R_x = np.pad(self.solid, ((0, 1), (0, 0)), mode="constant", constant_values=True)

        # Active to solid interface:
        mask_A_to_S_x = (~solid_L_x) & solid_R_x
        h_R_x[mask_A_to_S_x] = h_L_x[mask_A_to_S_x]
        hu_R_x[mask_A_to_S_x] = -hu_L_x[mask_A_to_S_x]
        hv_R_x[mask_A_to_S_x] = hv_L_x[mask_A_to_S_x]

        # Solid to active interface:
        mask_S_to_A_x = solid_L_x & (~solid_R_x)
        h_L_x[mask_S_to_A_x] = h_R_x[mask_S_to_A_x]
        hu_L_x[mask_S_to_A_x] = -hu_R_x[mask_S_to_A_x]
        hv_L_x[mask_S_to_A_x] = hv_R_x[mask_S_to_A_x]

        # Both solid:
        mask_both_solid_x = solid_L_x & solid_R_x
        h_L_x[mask_both_solid_x] = 0.0
        h_R_x[mask_both_solid_x] = 0.0
        hu_L_x[mask_both_solid_x] = 0.0
        hu_R_x[mask_both_solid_x] = 0.0
        hv_L_x[mask_both_solid_x] = 0.0
        hv_R_x[mask_both_solid_x] = 0.0

        # Non-negative heights
        h_L_x = np.maximum(h_L_x, 0.0)
        h_R_x = np.maximum(h_R_x, 0.0)

        # Interface velocities
        u_L_x = np.where(h_L_x > eps, hu_L_x / np.maximum(h_L_x, eps), 0.0)
        v_L_x = np.where(h_L_x > eps, hv_L_x / np.maximum(h_L_x, eps), 0.0)
        u_R_x = np.where(h_R_x > eps, hu_R_x / np.maximum(h_R_x, eps), 0.0)
        v_R_x = np.where(h_R_x > eps, hv_R_x / np.maximum(h_R_x, eps), 0.0)

        c_L_x = np.sqrt(g * h_L_x)
        c_R_x = np.sqrt(g * h_R_x)

        # Local wave speed a_x
        a_x = np.maximum(
            np.maximum(np.abs(u_L_x) + c_L_x, np.abs(u_R_x) + c_R_x),
            1e-8
        )

        # Physical fluxes in X
        # F(U) = [hu, hu^2/h + 0.5*g*h^2, hu*hv/h]
        F_h_L = hu_L_x
        F_hu_L = hu_L_x * u_L_x + 0.5 * g * h_L_x**2
        F_hv_L = hu_L_x * v_L_x

        F_h_R = hu_R_x
        F_hu_R = hu_R_x * u_R_x + 0.5 * g * h_R_x**2
        F_hv_R = hu_R_x * v_R_x

        # Kurganov-Tadmor numerical flux
        Fx_h = 0.5 * (F_h_L + F_h_R) - 0.5 * a_x * (h_R_x - h_L_x)
        Fx_hu = 0.5 * (F_hu_L + F_hu_R) - 0.5 * a_x * (hu_R_x - hu_L_x)
        Fx_hv = 0.5 * (F_hv_L + F_hv_R) - 0.5 * a_x * (hv_R_x - hv_L_x)

        # -------------------------------------------------------------
        # Y-direction fluxes at interfaces (i, j+1/2) for j=0..ny
        # -------------------------------------------------------------
        h_pad_y = np.pad(h, ((0, 0), (1, 1)), mode="edge")
        hu_pad_y = np.pad(hu, ((0, 0), (1, 1)), mode="edge")
        hv_pad_y = np.pad(hv, ((0, 0), (1, 1)), mode="edge")
        hv_pad_y[:, 0] = -hv[:, 0]
        hv_pad_y[:, -1] = -hv[:, -1]

        # Slopes in y
        dh_y = h_pad_y[:, 1:] - h_pad_y[:, :-1]
        s_h_y = self._minmod(dh_y[:, :-1], dh_y[:, 1:])
        dhu_y = hu_pad_y[:, 1:] - hu_pad_y[:, :-1]
        s_hu_y = self._minmod(dhu_y[:, :-1], dhu_y[:, 1:])
        dhv_y = hv_pad_y[:, 1:] - hv_pad_y[:, :-1]
        s_hv_y = self._minmod(dhv_y[:, :-1], dhv_y[:, 1:])

        # States at y-interfaces j+1/2 for j = 0 ... ny (size nx, ny+1)
        h_L_y = h_pad_y[:, 1:-1] + 0.5 * s_h_y
        hu_L_y = hu_pad_y[:, 1:-1] + 0.5 * s_hu_y
        hv_L_y = hv_pad_y[:, 1:-1] + 0.5 * s_hv_y

        h_R_y = h_pad_y[:, 2:] - 0.5 * np.pad(s_h_y[:, 1:], ((0, 0), (0, 1)), mode="edge")
        hu_R_y = hu_pad_y[:, 2:] - 0.5 * np.pad(s_hu_y[:, 1:], ((0, 0), (0, 1)), mode="edge")
        hv_R_y = hv_pad_y[:, 2:] - 0.5 * np.pad(s_hv_y[:, 1:], ((0, 0), (0, 1)), mode="edge")

        # Boundary interfaces at y=0 and y=ny
        h_L_y[:, 0] = h_R_y[:, 0]
        hu_L_y[:, 0] = hu_R_y[:, 0]
        hv_L_y[:, 0] = -hv_R_y[:, 0]

        h_R_y[:, -1] = h_L_y[:, -1]
        hu_R_y[:, -1] = hu_L_y[:, -1]
        hv_R_y[:, -1] = -hv_L_y[:, -1]

        # Internal obstacle interfaces in Y:
        solid_L_y = np.pad(self.solid, ((0, 0), (1, 0)), mode="constant", constant_values=True)
        solid_R_y = np.pad(self.solid, ((0, 0), (0, 1)), mode="constant", constant_values=True)

        mask_A_to_S_y = (~solid_L_y) & solid_R_y
        h_R_y[mask_A_to_S_y] = h_L_y[mask_A_to_S_y]
        hu_R_y[mask_A_to_S_y] = hu_L_y[mask_A_to_S_y]
        hv_R_y[mask_A_to_S_y] = -hv_L_y[mask_A_to_S_y]

        mask_S_to_A_y = solid_L_y & (~solid_R_y)
        h_L_y[mask_S_to_A_y] = h_R_y[mask_S_to_A_y]
        hu_L_y[mask_S_to_A_y] = hu_R_y[mask_S_to_A_y]
        hv_L_y[mask_S_to_A_y] = -hv_R_y[mask_S_to_A_y]

        mask_both_solid_y = solid_L_y & solid_R_y
        h_L_y[mask_both_solid_y] = 0.0
        h_R_y[mask_both_solid_y] = 0.0
        hu_L_y[mask_both_solid_y] = 0.0
        hu_R_y[mask_both_solid_y] = 0.0
        hv_L_y[mask_both_solid_y] = 0.0
        hv_R_y[mask_both_solid_y] = 0.0

        h_L_y = np.maximum(h_L_y, 0.0)
        h_R_y = np.maximum(h_R_y, 0.0)

        u_L_y = np.where(h_L_y > eps, hu_L_y / np.maximum(h_L_y, eps), 0.0)
        v_L_y = np.where(h_L_y > eps, hv_L_y / np.maximum(h_L_y, eps), 0.0)
        u_R_y = np.where(h_R_y > eps, hu_R_y / np.maximum(h_R_y, eps), 0.0)
        v_R_y = np.where(h_R_y > eps, hv_R_y / np.maximum(h_R_y, eps), 0.0)

        c_L_y = np.sqrt(g * h_L_y)
        c_R_y = np.sqrt(g * h_R_y)

        a_y = np.maximum(
            np.maximum(np.abs(v_L_y) + c_L_y, np.abs(v_R_y) + c_R_y),
            1e-8
        )

        # Physical fluxes in Y
        # G(U) = [hv, hu*hv/h, hv^2/h + 0.5*g*h^2]
        G_h_L = hv_L_y
        G_hu_L = hu_L_y * v_L_y
        G_hv_L = hv_L_y * v_L_y + 0.5 * g * h_L_y**2

        G_h_R = hv_R_y
        G_hu_R = hu_R_y * v_R_y
        G_hv_R = hv_R_y * v_R_y + 0.5 * g * h_R_y**2

        # KT flux in Y
        Fy_h = 0.5 * (G_h_L + G_h_R) - 0.5 * a_y * (h_R_y - h_L_y)
        Fy_hu = 0.5 * (G_hu_L + G_hu_R) - 0.5 * a_y * (hu_R_y - hu_L_y)
        Fy_hv = 0.5 * (G_hv_L + G_hv_R) - 0.5 * a_y * (hv_R_y - hv_L_y)

        # Net flux divergence dU/dt = - (dFx/dx + dFy/dy)
        dh_dt = -(Fx_h[1:, :] - Fx_h[:-1, :]) / self.dx - (Fy_h[:, 1:] - Fy_h[:, :-1]) / self.dy
        dhu_dt = -(Fx_hu[1:, :] - Fx_hu[:-1, :]) / self.dx - (Fy_hu[:, 1:] - Fy_hu[:, :-1]) / self.dy
        dhv_dt = -(Fx_hv[1:, :] - Fx_hv[:-1, :]) / self.dx - (Fy_hv[:, 1:] - Fy_hv[:, :-1]) / self.dy

        max_speed = max(float(a_x.max()), float(a_y.max()))
        return dh_dt, dhu_dt, dhv_dt, max_speed

    def _step(self, dt: float) -> None:
        """Advance the SWE state by dt using 2nd order SSP-RK2 (Heun's method)."""
        # Enforce obstacle boundary condition before step: zero inside obstacle
        self.h[self.solid] = 0.0
        self.hu[self.solid] = 0.0
        self.hv[self.solid] = 0.0

        # Stage 1
        k1_h, k1_hu, k1_hv, _ = self._compute_fluxes(self.h, self.hu, self.hv)
        h1 = np.maximum(self.h + dt * k1_h, 0.0)
        hu1 = self.hu + dt * k1_hu
        hv1 = self.hv + dt * k1_hv

        # Zero solid cells
        h1[self.solid] = 0.0
        hu1[self.solid] = 0.0
        hv1[self.solid] = 0.0

        # Stage 2
        k2_h, k2_hu, k2_hv, _ = self._compute_fluxes(h1, hu1, hv1)
        h2 = np.maximum(0.5 * self.h + 0.5 * (h1 + dt * k2_h), 0.0)
        hu2 = 0.5 * self.hu + 0.5 * (hu1 + dt * k2_hu)
        hv2 = 0.5 * self.hv + 0.5 * (hv1 + dt * k2_hv)

        h2[self.solid] = 0.0
        hu2[self.solid] = 0.0
        hv2[self.solid] = 0.0

        # Update state
        self.h = h2
        self.hu = hu2
        self.hv = hv2
        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        cfl = 0.4
        eps = 1e-12

        while self.current_time < target_time - 1e-10:
            dt_rem = target_time - self.current_time

            # Estimate max wave speed for adaptive substepping
            u = np.where(self.h > eps, self.hu / np.maximum(self.h, eps), 0.0)
            v = np.where(self.h > eps, self.hv / np.maximum(self.h, eps), 0.0)
            c = np.sqrt(self.g * np.maximum(self.h, 0.0))
            max_vel = np.maximum(np.abs(u) + c, np.abs(v) + c).max()
            max_vel = max(float(max_vel), 1.0)

            dt_cfl = cfl * min(self.dx, self.dy) / max_vel
            dt = min(dt_rem, dt_cfl)
            if dt <= 0:
                break
            self._step(dt)

        self.current_time = target_time

        # Build output StateFrame
        u = np.where(self.h > eps, self.hu / np.maximum(self.h, eps), 0.0)
        v = np.where(self.h > eps, self.hv / np.maximum(self.h, eps), 0.0)
        u[self.solid] = 0.0
        v[self.solid] = 0.0

        velocity_field = np.stack((u, v), axis=-1)

        position = np.column_stack((
            self.xx[self.active],
            self.yy[self.active],
            0.5 * self.h[self.active]
        ))
        velocity = np.column_stack((
            u[self.active],
            v[self.active],
            np.zeros(self.active.sum())
        ))
        volume = self.h[self.active] * self.dx * self.dy
        mass = self.rho * volume

        frame = StateFrame(
            time=float(self.current_time),
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "mass_quantity": mass.copy(),
                "water_height": self.h[self.active].copy(),
            },
            fields={
                "density_field": self.h.copy(),
                "velocity_field": velocity_field,
                "active_mask": self.active.astype(np.uint8),
                "continuity_source": np.zeros_like(self.h),
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": self.normals,
                "boundary_target_velocity": np.zeros_like(self.normals),
                "boundary_velocity": np.zeros_like(self.normals),
            },
            scalars={
                "grid_spacing": float(self.dx),
                "reference_velocity": self.ref_velocity,
                "initial_water_volume": self.initial_volume,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "total_volume": float((self.h[self.active] * self.dx * self.dy).sum()) if self.h is not None else 0.0,
        }

