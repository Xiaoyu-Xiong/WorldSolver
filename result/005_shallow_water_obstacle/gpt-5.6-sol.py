"""Conservative finite-volume shallow-water solver for the benchmark."""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Cell-centred SWE solver with reflecting domain and obstacle walls."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # This discretization has no stochastic terms.
        self._config = scene.config
        initial = scene.initial_state()
        domain = self._config["domain"]
        bounds_min = np.asarray(domain["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(domain["bounds_max"], dtype=np.float64)
        self._nx, self._ny = (int(v) for v in domain["grid_shape"])
        self._dx = float((bounds_max[0] - bounds_min[0]) / self._nx)
        self._dy = float((bounds_max[1] - bounds_min[1]) / self._ny)
        self._area = self._dx * self._dy
        self._rho = float(self._config["fluid"]["density"])
        self._g = abs(float(self._config["gravity"][2]))
        self._active = np.asarray(initial.fields["active_mask"], dtype=bool).copy()

        self._h = np.asarray(initial.fields["density_field"], dtype=np.float64).copy()
        velocity = np.asarray(initial.fields["velocity_field"], dtype=np.float64)
        self._mx = self._h * velocity[..., 0]
        self._my = self._h * velocity[..., 1]
        self._h[~self._active] = 0.0
        self._mx[~self._active] = 0.0
        self._my[~self._active] = 0.0

        x = bounds_min[0] + (np.arange(self._nx) + 0.5) * self._dx
        y = bounds_min[1] + (np.arange(self._ny) + 0.5) * self._dy
        self._xx, self._yy = np.meshgrid(x, y, indexing="ij")
        self._time = 0.0
        self._steps = 0
        self._initial_volume = float(np.sum(self._h[self._active]) * self._area)

        self._boundary_normal = np.asarray(
            initial.fields["boundary_normal"], dtype=np.float64
        ).copy()
        self._boundary_target_velocity = np.asarray(
            initial.fields["boundary_target_velocity"], dtype=np.float64
        ).copy()
        self._boundary_velocity = np.asarray(
            initial.fields["boundary_velocity"], dtype=np.float64
        ).copy()

    def _rhs(
        self, h: np.ndarray, mx: np.ndarray, my: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the conservative shallow-water flux divergence."""
        nx, ny, g = self._nx, self._ny, self._g
        pressure = 0.5 * g * h * h

        # fx[i] is oriented in +x through the face between cells i-1 and i.
        fx = np.zeros((nx + 1, ny, 3), dtype=np.float64)
        left_active = self._active[:-1, :]
        right_active = self._active[1:, :]
        shared = left_active & right_active
        hl, hr = h[:-1, :], h[1:, :]
        mxl, mxr = mx[:-1, :], mx[1:, :]
        myl, myr = my[:-1, :], my[1:, :]
        ul = mxl / np.maximum(hl, 1.0e-12)
        ur = mxr / np.maximum(hr, 1.0e-12)
        vl = myl / np.maximum(hl, 1.0e-12)
        vr = myr / np.maximum(hr, 1.0e-12)
        fl = np.stack((mxl, mxl * ul + 0.5 * g * hl * hl, mxl * vl), axis=-1)
        fr = np.stack((mxr, mxr * ur + 0.5 * g * hr * hr, mxr * vr), axis=-1)
        speed = np.maximum(
            np.abs(ul) + np.sqrt(g * np.maximum(hl, 0.0)),
            np.abs(ur) + np.sqrt(g * np.maximum(hr, 0.0)),
        )
        ql = np.stack((hl, mxl, myl), axis=-1)
        qr = np.stack((hr, mxr, myr), axis=-1)
        rusanov = 0.5 * (fl + fr) - 0.5 * speed[..., None] * (qr - ql)
        fx_inner = fx[1:nx]
        fx_inner[shared] = rusanov[shared]

        # Exact stationary slip-wall flux: no mass/tangential momentum flow.
        left_wall = left_active & ~right_active
        right_wall = ~left_active & right_active
        face_pressure = np.zeros((nx - 1, ny), dtype=np.float64)
        face_pressure[left_wall] = pressure[:-1, :][left_wall]
        face_pressure[right_wall] = pressure[1:, :][right_wall]
        wall = left_wall | right_wall
        fx_inner[..., 1][wall] = face_pressure[wall]
        fx[0, :, 1] = np.where(self._active[0, :], pressure[0, :], 0.0)
        fx[nx, :, 1] = np.where(self._active[-1, :], pressure[-1, :], 0.0)

        # gy[j] is oriented in +y through the face between cells j-1 and j.
        gy = np.zeros((nx, ny + 1, 3), dtype=np.float64)
        lower_active = self._active[:, :-1]
        upper_active = self._active[:, 1:]
        shared = lower_active & upper_active
        hl, hr = h[:, :-1], h[:, 1:]
        mxl, mxr = mx[:, :-1], mx[:, 1:]
        myl, myr = my[:, :-1], my[:, 1:]
        ul = mxl / np.maximum(hl, 1.0e-12)
        ur = mxr / np.maximum(hr, 1.0e-12)
        vl = myl / np.maximum(hl, 1.0e-12)
        vr = myr / np.maximum(hr, 1.0e-12)
        gl = np.stack((myl, myl * ul, myl * vl + 0.5 * g * hl * hl), axis=-1)
        gr = np.stack((myr, myr * ur, myr * vr + 0.5 * g * hr * hr), axis=-1)
        speed = np.maximum(
            np.abs(vl) + np.sqrt(g * np.maximum(hl, 0.0)),
            np.abs(vr) + np.sqrt(g * np.maximum(hr, 0.0)),
        )
        ql = np.stack((hl, mxl, myl), axis=-1)
        qr = np.stack((hr, mxr, myr), axis=-1)
        rusanov = 0.5 * (gl + gr) - 0.5 * speed[..., None] * (qr - ql)
        gy_inner = gy[:, 1:ny]
        gy_inner[shared] = rusanov[shared]

        lower_wall = lower_active & ~upper_active
        upper_wall = ~lower_active & upper_active
        face_pressure = np.zeros((nx, ny - 1), dtype=np.float64)
        face_pressure[lower_wall] = pressure[:, :-1][lower_wall]
        face_pressure[upper_wall] = pressure[:, 1:][upper_wall]
        wall = lower_wall | upper_wall
        gy_inner[..., 2][wall] = face_pressure[wall]
        gy[:, 0, 2] = np.where(self._active[:, 0], pressure[:, 0], 0.0)
        gy[:, ny, 2] = np.where(self._active[:, -1], pressure[:, -1], 0.0)

        divergence = -(
            (fx[1:] - fx[:-1]) / self._dx
            + (gy[:, 1:] - gy[:, :-1]) / self._dy
        )
        divergence[~self._active] = 0.0
        return divergence[..., 0], divergence[..., 1], divergence[..., 2]

    def _stable_dt(self, remaining: float) -> float:
        wet_h = np.maximum(self._h[self._active], 1.0e-12)
        u = np.abs(self._mx[self._active] / wet_h)
        v = np.abs(self._my[self._active] / wet_h)
        c = np.sqrt(self._g * wet_h)
        rate = float(np.max((u + c) / self._dx + (v + c) / self._dy))
        if not math.isfinite(rate) or rate <= 0.0:
            return remaining
        return min(remaining, 0.42 / rate)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target):
            raise ValueError("sample time must be finite")
        if target < self._time - 1.0e-12:
            raise ValueError("advance_to cannot integrate backwards")

        while self._time < target - 1.0e-12:
            dt = self._stable_dt(target - self._time)
            if dt <= 0.0 or not math.isfinite(dt):
                raise RuntimeError("invalid shallow-water time step")
            dh, dmx, dmy = self._rhs(self._h, self._mx, self._my)
            self._h += dt * dh
            self._mx += dt * dmx
            self._my += dt * dmy
            self._h[~self._active] = 0.0
            self._mx[~self._active] = 0.0
            self._my[~self._active] = 0.0
            if not (
                np.all(np.isfinite(self._h[self._active]))
                and np.all(np.isfinite(self._mx[self._active]))
                and np.all(np.isfinite(self._my[self._active]))
                and np.all(self._h[self._active] > 0.0)
            ):
                raise RuntimeError("non-finite or non-positive shallow-water state")
            self._time += dt
            self._steps += 1
        self._time = target
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        h = self._h
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        u[self._active] = self._mx[self._active] / h[self._active]
        v[self._active] = self._my[self._active] / h[self._active]
        active_h = h[self._active].copy()
        position = np.column_stack(
            (self._xx[self._active], self._yy[self._active], 0.5 * active_h)
        )
        velocity = np.column_stack(
            (u[self._active], v[self._active], np.zeros(self._active.sum()))
        )
        mass = self._rho * self._area * active_h

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass.copy(),
            attributes={"mass_quantity": mass.copy(), "water_height": active_h},
            fields={
                "density_field": h.copy(),
                "velocity_field": np.stack((u, v), axis=-1),
                "active_mask": self._active.astype(np.uint8),
                "continuity_source": np.zeros_like(h),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "boundary_normal": self._boundary_normal.copy(),
                "boundary_target_velocity": self._boundary_target_velocity.copy(),
                "boundary_velocity": self._boundary_velocity.copy(),
            },
            scalars={
                "grid_spacing": self._dx,
                "reference_velocity": float(np.sqrt(self._g * np.max(active_h))),
                "initial_water_volume": self._initial_volume,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        volume = float(np.sum(self._h[self._active]) * self._area)
        return {
            "method": "finite_volume_rusanov_shallow_water",
            "time": float(self._time),
            "substeps": int(self._steps),
            "water_volume": volume,
            "relative_volume_error": (volume - self._initial_volume) / self._initial_volume,
        }
