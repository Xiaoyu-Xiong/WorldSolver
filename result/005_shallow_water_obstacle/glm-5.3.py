"""Shallow-water (Saint-Venant) finite-volume solver for scene 005.

Physical model: two-dimensional shallow-water equations on the frozen 64x64
cell-centered grid. The conserved state per cell is (h, h*u, h*v), where h is
the water height in meters (the exported ``density_field``), and (u, v) is the
depth-averaged horizontal velocity.

Numerics: method-of-lines finite volumes with MUSCL (minmod-limited)
reconstruction of primitive variables, local Lax-Friedrichs (Rusanov) fluxes,
reflecting (no-penetration) boundary conditions on the tank walls and on the
octagonal obstacle (solid cells), and explicit two-stage SSP Runge-Kutta time
stepping under a CFL restriction. The scheme is conservative, so the height
integral over the fluid cells is conserved exactly (up to roundoff).

No animation, keyframes, or precomputed motion are used; the state at every
sample time is produced by actually stepping the physics.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    prod = a * b
    return np.where(
        prod <= 0.0,
        0.0,
        np.sign(a) * np.minimum(np.abs(a), np.abs(b)),
    )


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.frame0 = frame0

        height = np.asarray(frame0.fields["density_field"], dtype=float)
        self.nx, self.ny = height.shape
        domain = cfg["domain"]
        low = np.asarray(domain["bounds_min"], dtype=float)
        high = np.asarray(domain["bounds_max"], dtype=float)
        self.dx = float((high[0] - low[0]) / self.nx)
        self.dy = float((high[1] - low[1]) / self.ny)
        self.x = low[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.y = low[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

        self.g = float(abs(cfg["gravity"][2]))
        self.rho = float(cfg["fluid"]["density"])

        self.solid = np.asarray(frame0.fields["active_mask"]) == 0
        self.fluid = ~self.solid

        self.h = height.copy()
        vel0 = np.asarray(frame0.fields["velocity_field"], dtype=float)
        self.u = vel0[..., 0].copy()
        self.v = vel0[..., 1].copy()
        self.h[self.solid] = 0.0
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0

        self.boundary_normal = np.asarray(frame0.fields["boundary_normal"], dtype=float).copy()
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

        self.t = 0.0
        self.steps = 0
        self.initial_volume = float((self.h * self.dx * self.dy).sum())
        self.initial_mass = float(self.rho * self.initial_volume)

        self._build_boundary_map()
        self._cfl = 0.45

    # ------------------------------------------------------------------
    # boundary sample -> grid cell map (for wall velocity diagnostics)
    # ------------------------------------------------------------------
    def _build_boundary_map(self) -> None:
        nx, ny = self.nx, self.ny
        normals = self.boundary_normal
        rows = normals.shape[0]
        ix = np.zeros(rows, dtype=int)
        iy = np.zeros(rows, dtype=int)

        # Wall samples, laid out exactly as in the frozen scene: right wall,
        # left wall, upper wall, lower wall (ny, ny, nx, nx samples).
        idx = 0
        for j in range(ny):
            ix[idx], iy[idx] = nx - 1, j
            idx += 1
        for j in range(ny):
            ix[idx], iy[idx] = 0, j
            idx += 1
        for i in range(nx):
            ix[idx], iy[idx] = i, ny - 1
            idx += 1
        for i in range(nx):
            ix[idx], iy[idx] = i, 0
            idx += 1

        # Obstacle samples: probe slightly outside the obstacle along the
        # sample normal and take the nearest fluid cell.
        obstacle = self.cfg["obstacle"]
        center = np.asarray(obstacle["center"], dtype=float)
        radius = float(obstacle["circumradius"])
        for r in range(idx, rows):
            n = normals[r]
            probe = center + (1.35 * radius) * n
            i = int(np.clip(np.searchsorted(self.x, probe[0]) - 1, 0, nx - 1))
            j = int(np.clip(np.searchsorted(self.y, probe[1]) - 1, 0, ny - 1))
            if self.solid[i, j]:
                best = None
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ii = int(np.clip(i + di, 0, nx - 1))
                        jj = int(np.clip(j + dj, 0, ny - 1))
                        if self.solid[ii, jj]:
                            continue
                        d = (self.xx[ii, jj] - probe[0]) ** 2 + (self.yy[ii, jj] - probe[1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, ii, jj)
                if best is not None:
                    i, j = best[1], best[2]
            ix[r], iy[r] = i, j

        self.bix = ix
        self.biy = iy

    # ------------------------------------------------------------------
    # spatial operator
    # ------------------------------------------------------------------
    def _pad(self, interior: np.ndarray) -> np.ndarray:
        return np.pad(interior, 1, mode="edge")

    def _rhs(self, h: np.ndarray, u: np.ndarray, v: np.ndarray):
        nx, ny, g, dx, dy = self.nx, self.ny, self.g, self.dx, self.dy

        hp = self._pad(h)
        up = self._pad(u)
        vp = self._pad(v)
        # Reflecting (no-penetration) domain walls via ghost reflection.
        up[0, :] = -up[0, :]
        up[-1, :] = -up[-1, :]
        vp[:, 0] = -vp[:, 0]
        vp[:, -1] = -vp[:, -1]

        sp = np.zeros((nx + 2, ny + 2), dtype=bool)
        sp[1:-1, 1:-1] = self.solid

        # minmod slopes (primitive variables), first order next to walls/solid.
        def slope_x(p):
            d1 = p[1:-1, 1:-1] - p[:-2, 1:-1]
            d2 = p[2:, 1:-1] - p[1:-1, 1:-1]
            blocked = sp[:-2, 1:-1] | sp[1:-1, 1:-1] | sp[2:, 1:-1]
            s = np.where(blocked, 0.0, _minmod(d1, d2))
            out = np.zeros_like(p)
            out[1:-1, 1:-1] = s
            return out

        def slope_y(p):
            d1 = p[1:-1, 1:-1] - p[1:-1, :-2]
            d2 = p[1:-1, 2:] - p[1:-1, 1:-1]
            blocked = sp[1:-1, :-2] | sp[1:-1, 1:-1] | sp[1:-1, 2:]
            s = np.where(blocked, 0.0, _minmod(d1, d2))
            out = np.zeros_like(p)
            out[1:-1, 1:-1] = s
            return out

        sxh, sxu, sxv = slope_x(hp), slope_x(up), slope_x(vp)
        syh, syu, syv = slope_y(hp), slope_y(up), slope_y(vp)

        # ---- x-direction faces ----
        Lh = hp[:-1, 1:-1] + 0.5 * sxh[:-1, 1:-1]
        Lu = up[:-1, 1:-1] + 0.5 * sxu[:-1, 1:-1]
        Lv = vp[:-1, 1:-1] + 0.5 * sxv[:-1, 1:-1]
        Rh = hp[1:, 1:-1] - 0.5 * sxh[1:, 1:-1]
        Ru = up[1:, 1:-1] - 0.5 * sxu[1:, 1:-1]
        Rv = vp[1:, 1:-1] - 0.5 * sxv[1:, 1:-1]

        solidL = sp[:-1, 1:-1]
        solidR = sp[1:, 1:-1]
        both = solidL & solidR
        # Solid faces: mirror (free-slip wall) state from the fluid side.
        Lh = np.where(solidL, Rh, Lh)
        Lu = np.where(solidL, -Ru, Lu)
        Lv = np.where(solidL, Rv, Lv)
        Rh = np.where(solidR, Lh, Rh)
        Ru = np.where(solidR, -Lu, Ru)
        Rv = np.where(solidR, Lv, Rv)

        sL = np.abs(Lu) + np.sqrt(g * np.maximum(Lh, 0.0))
        sR = np.abs(Ru) + np.sqrt(g * np.maximum(Rh, 0.0))
        s = np.maximum(sL, sR)
        Fh = 0.5 * (Lh * Lu + Rh * Ru) - 0.5 * s * (Rh - Lh)
        Fhu = (
            0.5 * (Lh * Lu * Lu + 0.5 * g * Lh * Lh + Rh * Ru * Ru + 0.5 * g * Rh * Rh)
            - 0.5 * s * (Rh * Ru - Lh * Lu)
        )
        Fhv = 0.5 * (Lh * Lu * Lv + Rh * Ru * Rv) - 0.5 * s * (Rh * Rv - Lh * Lv)
        Fh = np.where(both, 0.0, Fh)
        Fhu = np.where(both, 0.0, Fhu)
        Fhv = np.where(both, 0.0, Fhv)

        # ---- y-direction faces ----
        Lh = hp[1:-1, :-1] + 0.5 * syh[1:-1, :-1]
        Lu = up[1:-1, :-1] + 0.5 * syu[1:-1, :-1]
        Lv = vp[1:-1, :-1] + 0.5 * syv[1:-1, :-1]
        Rh = hp[1:-1, 1:] - 0.5 * syh[1:-1, 1:]
        Ru = up[1:-1, 1:] - 0.5 * syu[1:-1, 1:]
        Rv = vp[1:-1, 1:] - 0.5 * syv[1:-1, 1:]

        solidL = sp[1:-1, :-1]
        solidR = sp[1:-1, 1:]
        both = solidL & solidR
        Lh = np.where(solidL, Rh, Lh)
        Lv = np.where(solidL, -Rv, Lv)
        Lu = np.where(solidL, Ru, Lu)
        Rh = np.where(solidR, Lh, Rh)
        Rv = np.where(solidR, -Lv, Rv)
        Ru = np.where(solidR, Lu, Ru)

        sL = np.abs(Lv) + np.sqrt(g * np.maximum(Lh, 0.0))
        sR = np.abs(Rv) + np.sqrt(g * np.maximum(Rh, 0.0))
        s = np.maximum(sL, sR)
        Gh = 0.5 * (Lh * Lv + Rh * Rv) - 0.5 * s * (Rh - Lh)
        Ghu = 0.5 * (Lh * Lu * Lv + Rh * Ru * Rv) - 0.5 * s * (Rh * Ru - Lh * Lu)
        Ghv = (
            0.5 * (Lh * Lv * Lv + 0.5 * g * Lh * Lh + Rh * Rv * Rv + 0.5 * g * Rh * Rh)
            - 0.5 * s * (Rh * Rv - Lh * Lv)
        )
        Gh = np.where(both, 0.0, Gh)
        Ghu = np.where(both, 0.0, Ghu)
        Ghv = np.where(both, 0.0, Ghv)

        lh = (Fh[:-1, :] - Fh[1:, :]) / dx + (Gh[:, :-1] - Gh[:, 1:]) / dy
        lu = (Fhu[:-1, :] - Fhu[1:, :]) / dx + (Ghu[:, :-1] - Ghu[:, 1:]) / dy
        lv = (Fhv[:-1, :] - Fhv[1:, :]) / dx + (Ghv[:, :-1] - Ghv[:, 1:]) / dy
        return lh, lu, lv

    def _time_step(self) -> float:
        c = np.sqrt(self.g * np.maximum(self.h, 0.0))
        speed = np.maximum(
            np.abs(self.u) + c,
            np.abs(self.v) + c,
        )
        smax = float(np.max(speed[self.fluid])) if np.any(self.fluid) else 1.0
        smax = max(smax, 1e-6)
        return self._cfl * min(self.dx, self.dy) / smax

    def _step(self, dt: float) -> None:
        h0, u0, v0 = self.h, self.u, self.v
        lh, lu, lv = self._rhs(h0, u0, v0)
        h1 = h0 + dt * lh
        u1 = np.where(h1 > 1e-9, (h0 * u0 + dt * lu) / np.maximum(h1, 1e-9), 0.0)
        v1 = np.where(h1 > 1e-9, (h0 * v0 + dt * lv) / np.maximum(h1, 1e-9), 0.0)
        lh, lu, lv = self._rhs(h1, u1, v1)
        h2 = h1 + dt * lh
        u2 = np.where(h2 > 1e-9, (h1 * u1 + dt * lu) / np.maximum(h2, 1e-9), 0.0)
        v2 = np.where(h2 > 1e-9, (h1 * v1 + dt * lv) / np.maximum(h2, 1e-9), 0.0)
        # SSP-RK2 combination in conserved variables (h, h*u, h*v).
        h_new = 0.5 * (h0 + h2)
        hu_new = 0.5 * (h0 * u0 + h2 * u2)
        hv_new = 0.5 * (h0 * v0 + h2 * v2)
        self.h = h_new
        self.u = np.where(h_new > 1e-9, hu_new / np.maximum(h_new, 1e-9), 0.0)
        self.v = np.where(h_new > 1e-9, hv_new / np.maximum(h_new, 1e-9), 0.0)
        self.h[self.solid] = 0.0
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        if np.min(self.h[self.fluid]) < 0.0:
            self.h = np.maximum(self.h, 0.0)
            self.h[self.solid] = 0.0
        self.t += dt
        self.steps += 1

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        eps = 1e-12
        guard = 0
        while self.t < target - eps and guard < 100000:
            dt = min(self._time_step(), target - self.t)
            if dt <= 0.0:
                break
            self._step(dt)
            guard += 1
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        h = self.h.copy()
        u = np.where(self.fluid, self.u, 0.0)
        v = np.where(self.fluid, self.v, 0.0)
        height = np.where(self.fluid, h, 0.0)

        velocity_field = np.zeros((self.nx, self.ny, 2))
        velocity_field[..., 0] = u
        velocity_field[..., 1] = v

        cell_v = velocity_field[self.bix, self.biy]
        normals = self.boundary_normal
        dot = np.sum(cell_v * normals, axis=1, keepdims=True)
        boundary_velocity = cell_v - dot * normals

        dxdy = self.dx * self.dy
        mass = self.rho * height[self.fluid] * dxdy
        position = np.column_stack(
            (self.xx[self.fluid], self.yy[self.fluid], 0.5 * height[self.fluid])
        )
        velocity = np.column_stack((u[self.fluid], v[self.fluid], np.zeros(int(self.fluid.sum()))))

        volume = float((height * dxdy).sum())
        frame = StateFrame(
            time=max(float(time), 0.0),
            position=position,
            velocity=velocity,
            mass=mass.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "water_height": height[self.fluid].copy(),
            },
            fields={
                "density_field": height,
                "velocity_field": velocity_field,
                "active_mask": self.fluid.astype(np.uint8),
                "continuity_source": np.zeros_like(height),
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": normals.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": boundary_velocity,
            },
            scalars={
                "grid_spacing": float(self.dx),
                "reference_velocity": float(np.sqrt(self.g * max(float(height.max()), 1e-6))),
                "initial_water_volume": self.initial_volume,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        volume = float((self.h * self.dx * self.dy).sum())
        return {
            "method": "finite-volume shallow water (MUSCL-minmod, Rusanov flux, SSP-RK2)",
            "steps": int(self.steps),
            "final_time": float(self.t),
            "final_water_volume": volume,
            "initial_water_volume": self.initial_volume,
            "volume_drift": volume - self.initial_volume,
            "min_height": float(self.h[self.fluid].min()) if np.any(self.fluid) else 0.0,
            "max_height": float(self.h.max()),
            "max_speed": float(
                np.max(np.hypot(self.u[self.fluid], self.v[self.fluid]))
                if np.any(self.fluid)
                else 0.0
            ),
        }
