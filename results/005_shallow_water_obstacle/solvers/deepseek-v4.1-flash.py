"""Physically-based shallow-water solver for the octagonal-obstacle scene.

The water height ``h`` (metres) evolves under the 2-D shallow-water equations

    dh/dt + d(h u)/dx + d(h v)/dy = 0
    d(h u)/dt + d(h u^2 + g h^2/2)/dx + d(h u v)/dy = 0
    d(h v)/dt + d(h u v)/dx + d(h v^2 + g h^2/2)/dy = 0

on the frozen 64x64 cell-centred grid.  The discretisation is a conservative
finite-volume HLL scheme with MUSCL (minmod) reconstruction and SSP-RK2 time
integration.  Solid obstacle cells and the four domain walls are handled with a
hydrostatic wall flux, so mass is conserved to round-off and reflected waves
are stable.

Only NumPy is used; no external scene/physics engine is required.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b <= 0.0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))


class BenchmarkSolver:
    """Deterministic shallow-water finite-volume solver."""

    #: CFL number used for the adaptive time step.
    cfl = 0.4
    #: HLL sound-speed floor (heights stay well above this in this scene).
    _h_floor = 1e-9

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.scene = scene
        cfg = getattr(scene, "config", None) or {}

        frame0 = scene.initial_state()
        h0 = np.asarray(frame0.fields["density_field"], dtype=np.float64)
        mask = np.asarray(frame0.fields["active_mask"]).astype(bool)
        self.nx, self.ny = int(h0.shape[0]), int(h0.shape[1])
        self.mask = np.ascontiguousarray(mask)
        self.solid = ~self.mask

        # Grid geometry: prefer the frozen config, fall back on the frame scalars.
        domain = cfg.get("domain", {}) or {}
        lo = domain.get("bounds_min", None)
        hi = domain.get("bounds_max", None)
        if lo is None or hi is None:
            spacing = float(frame0.scalars.get("grid_spacing", 2.0 / self.nx))
            pts = np.asarray(frame0.position, dtype=np.float64)[:, :2]
            lo = pts.min(axis=0) - 0.5 * spacing
            hi = pts.max(axis=0) + 0.5 * spacing
        self.dx = (float(hi[0]) - float(lo[0])) / self.nx
        self.dy = (float(hi[1]) - float(lo[1])) / self.ny
        self.hmin = min(self.dx, self.dy)
        self.xc = float(lo[0]) + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = float(lo[1]) + (np.arange(self.ny) + 0.5) * self.dy

        gravity = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.g = abs(float(gravity[2])) if len(gravity) > 2 else 9.81
        self.rho = float((cfg.get("fluid", {}) or {}).get("density", 1000.0))

        # Conserved state: cell-centred water height and momenta.
        self.h = np.where(self.mask, np.maximum(h0, 0.0), 0.0).astype(np.float64)
        self.hu = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.hv = np.zeros((self.nx, self.ny), dtype=np.float64)
        init_vel = (cfg.get("initial_conditions", {}) or {}).get("initial_velocity", [0.0, 0.0])
        if len(init_vel) >= 2 and (init_vel[0] or init_vel[1]):
            self.hu = np.where(self.mask, self.h * float(init_vel[0]), 0.0)
            self.hv = np.where(self.mask, self.h * float(init_vel[1]), 0.0)

        # Frozen wall data (static reflecting boundaries).
        self.normals = np.asarray(frame0.fields["boundary_normal"], dtype=np.float64).copy()
        self.boundary_target_velocity = np.zeros_like(self.normals)
        self.boundary_velocity = np.zeros_like(self.normals)

        self.ii, self.jj = np.nonzero(self.mask)
        self.n_active = int(self.ii.size)
        self.cell_area = self.dx * self.dy

        self.t = 0.0
        self.initial_mass = float(self.rho * np.sum(self.h) * self.cell_area)
        self.initial_volume = float(np.sum(self.h) * self.cell_area)
        self.reference_velocity = float(np.sqrt(self.g * float(self.h.max(initial=0.0))))
        if not np.isfinite(self.reference_velocity) or self.reference_velocity <= 0.0:
            self.reference_velocity = 1.0
        self.step_count = 0

    # ------------------------------------------------------------- numerics
    def _slopes(self, q: np.ndarray, axis: int) -> np.ndarray:
        """Limited (minmod) cell slopes, zero across solid/invalid neighbours."""
        s = np.zeros_like(q)
        m = self.mask
        if axis == 0 and self.nx > 2:
            qm, q0, qp = q[:-2], q[1:-1], q[2:]
            mm, m0, mp = m[:-2], m[1:-1], m[2:]
            dL = np.where(m0 & mm, q0 - qm, 0.0)
            dR = np.where(m0 & mp, qp - q0, 0.0)
            s[1:-1] = np.where(m0, _minmod(dL, dR), 0.0)
        elif axis == 1 and self.ny > 2:
            qm, q0, qp = q[:, :-2], q[:, 1:-1], q[:, 2:]
            mm, m0, mp = m[:, :-2], m[:, 1:-1], m[:, 2:]
            dL = np.where(m0 & mm, q0 - qm, 0.0)
            dR = np.where(m0 & mp, qp - q0, 0.0)
            s[:, 1:-1] = np.where(m0, _minmod(dL, dR), 0.0)
        return s

    def _hll(self, hL, huL, hvL, hR, huR, hvR, axis: int):
        """HLL flux for the shallow-water system at faces along ``axis``."""
        g = self.g
        aL = np.sqrt(g * hL)
        aR = np.sqrt(g * hR)
        if axis == 0:
            uL, vL = huL / hL, hvL / hL
            uR, vR = huR / hR, hvR / hR
            sL = np.minimum(uL - aL, uR - aR)
            sR = np.maximum(uL + aL, uR + aR)
            FL0, FL1, FL2 = huL, huL * uL + 0.5 * g * hL * hL, huL * vL
            FR0, FR1, FR2 = huR, huR * uR + 0.5 * g * hR * hR, huR * vR
        else:
            vL, uL = hvL / hL, huL / hL
            vR, uR = hvR / hR, huR / hR
            sL = np.minimum(vL - aL, vR - aR)
            sR = np.maximum(vL + aL, vR + aR)
            FL0, FL1, FL2 = hvL, huL * vL, hvL * vL + 0.5 * g * hL * hL
            FR0, FR1, FR2 = hvR, huR * vR, hvR * vR + 0.5 * g * hR * hR
        qL = (hL, huL, hvL)
        qR = (hR, huR, hvR)
        denom = sR - sL
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        def combine(FL, FR, ql, qr):
            star = (sR * FL - sL * FR + sL * sR * (qr - ql)) / denom
            return np.where(sL >= 0.0, FL, np.where(sR <= 0.0, FR, star))

        return (
            combine(FL0, FR0, qL[0], qR[0]),
            combine(FL1, FR1, qL[1], qR[1]),
            combine(FL2, FR2, qL[2], qR[2]),
        )

    def _fluxes(self, h, hu, hv):
        """Conservative fluxes on x-faces (nx+1,ny,3) and y-faces (nx,ny+1,3)."""
        g = self.g
        m = self.mask
        nx, ny = self.nx, self.ny
        sxh, sxu, sxv = (self._slopes(h, 0), self._slopes(hu, 0), self._slopes(hv, 0))
        syh, syu, syv = (self._slopes(h, 1), self._slopes(hu, 1), self._slopes(hv, 1))

        # --- x faces -----------------------------------------------------
        Fx = np.zeros((nx + 1, ny, 3), dtype=np.float64)
        if nx > 1:
            both = m[:-1] & m[1:]
            hL = h[:-1] + 0.5 * sxh[:-1]
            huL = hu[:-1] + 0.5 * sxu[:-1]
            hvL = hv[:-1] + 0.5 * sxv[:-1]
            hR = h[1:] - 0.5 * sxh[1:]
            huR = hu[1:] - 0.5 * sxu[1:]
            hvR = hv[1:] - 0.5 * sxv[1:]
            hLs = np.where(both, hL, 1.0)
            hRs = np.where(both, hR, 1.0)
            f0, f1, f2 = self._hll(
                hLs, huL, hvL, hRs, huR, hvR, 0
            )
            wall = ~both
            hwall = np.where(m[:-1], h[:-1], np.where(m[1:], h[1:], 0.0))
            Fx[1:nx, :, 0] = np.where(both, f0, 0.0)
            Fx[1:nx, :, 1] = np.where(both, f1, 0.5 * g * np.where(wall, hwall, 0.0) ** 2)
            Fx[1:nx, :, 2] = np.where(both, f2, 0.0)
        hwL = np.where(m[0], h[0], 0.0)
        Fx[0, :, 1] = 0.5 * g * hwL * hwL
        hwR = np.where(m[nx - 1], h[nx - 1], 0.0)
        Fx[nx, :, 1] = 0.5 * g * hwR * hwR

        # --- y faces -----------------------------------------------------
        Fy = np.zeros((nx, ny + 1, 3), dtype=np.float64)
        if ny > 1:
            both = m[:, :-1] & m[:, 1:]
            hL = h[:, :-1] + 0.5 * syh[:, :-1]
            huL = hu[:, :-1] + 0.5 * syu[:, :-1]
            hvL = hv[:, :-1] + 0.5 * syv[:, :-1]
            hR = h[:, 1:] - 0.5 * syh[:, 1:]
            huR = hu[:, 1:] - 0.5 * syu[:, 1:]
            hvR = hv[:, 1:] - 0.5 * syv[:, 1:]
            hLs = np.where(both, hL, 1.0)
            hRs = np.where(both, hR, 1.0)
            f0, f1, f2 = self._hll(
                hLs, huL, hvL, hRs, huR, hvR, 1
            )
            wall = ~both
            hwall = np.where(m[:, :-1], h[:, :-1], np.where(m[:, 1:], h[:, 1:], 0.0))
            Fy[:, 1:ny, 0] = np.where(both, f0, 0.0)
            Fy[:, 1:ny, 1] = np.where(both, f1, 0.0)
            Fy[:, 1:ny, 2] = np.where(both, f2, 0.5 * g * np.where(wall, hwall, 0.0) ** 2)
        hwD = np.where(m[:, 0], h[:, 0], 0.0)
        Fy[:, 0, 2] = 0.5 * g * hwD * hwD
        hwU = np.where(m[:, ny - 1], h[:, ny - 1], 0.0)
        Fy[:, ny, 2] = 0.5 * g * hwU * hwU
        return Fx, Fy

    def _rhs(self, h, hu, hv):
        """Negative flux divergence (solid cells held at rest)."""
        Fx, Fy = self._fluxes(h, hu, hv)
        m = self.mask
        inv_dx, inv_dy = 1.0 / self.dx, 1.0 / self.dy
        r0 = -((Fx[1:, :, 0] - Fx[:-1, :, 0]) * inv_dx + (Fy[:, 1:, 0] - Fy[:, :-1, 0]) * inv_dy)
        r1 = -((Fx[1:, :, 1] - Fx[:-1, :, 1]) * inv_dx + (Fy[:, 1:, 1] - Fy[:, :-1, 1]) * inv_dy)
        r2 = -((Fx[1:, :, 2] - Fx[:-1, :, 2]) * inv_dx + (Fy[:, 1:, 2] - Fy[:, :-1, 2]) * inv_dy)
        return (
            np.where(m, r0, 0.0),
            np.where(m, r1, 0.0),
            np.where(m, r2, 0.0),
        )

    def _max_wave_speed(self) -> float:
        m = self.mask
        hs = np.where(m, self.h, 1.0)
        us = self.hu / hs
        vs = self.hv / hs
        a = np.sqrt(self.g * np.where(m, self.h, 0.0))
        speed = np.sqrt(us * us + vs * vs) + a
        return float(np.max(np.where(m, speed, 0.0)))

    @staticmethod
    def _sanitize(a: np.ndarray, floor: bool = False) -> np.ndarray:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if floor:
            a = np.clip(a, 0.0, None)
        return a

    def _step(self, dt: float) -> None:
        m = self.mask
        r0, r1, r2 = self._rhs(self.h, self.hu, self.hv)
        h1 = self._sanitize(self.h + dt * r0, floor=True)
        hu1 = self._sanitize(self.hu + dt * r1)
        hv1 = self._sanitize(self.hv + dt * r2)
        h1 = np.where(m, h1, 0.0)
        hu1 = np.where(m, hu1, 0.0)
        hv1 = np.where(m, hv1, 0.0)

        q0, q1, q2 = self._rhs(h1, hu1, hv1)
        h2 = self._sanitize(0.5 * (self.h + h1 + dt * q0), floor=True)
        hu2 = self._sanitize(0.5 * (self.hu + hu1 + dt * q1))
        hv2 = self._sanitize(0.5 * (self.hv + hv1 + dt * q2))
        self.h = np.where(m, h2, 0.0)
        self.hu = np.where(m, hu2, 0.0)
        self.hv = np.where(m, hv2, 0.0)
        self.step_count += 1

    # --------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            raise ValueError("advance_to() must be called with non-decreasing times")
        guard = 0
        while self.t < target - 1e-12:
            speed = self._max_wave_speed()
            dt = self.cfl * self.hmin / max(speed, 1e-9)
            remaining = target - self.t
            if dt > remaining:
                dt = remaining
            if dt <= 0.0:
                break
            self._step(dt)
            self.t += dt
            guard += 1
            if guard > 5_000_000:  # pragma: no cover - safety valve
                raise RuntimeError("Solver exceeded internal step budget")
        self.t = target
        return self._frame()

    def _velocity_field(self):
        hs = np.where(self.h > self._h_floor, self.h, 1.0)
        u = np.where(self.mask, self.hu / hs, 0.0)
        v = np.where(self.mask, self.hv / hs, 0.0)
        return np.stack((u, v), axis=-1)

    def _frame(self) -> StateFrame:
        m = self.mask
        h = np.where(m, np.maximum(self.h, 0.0), 0.0)
        vel = self._velocity_field()
        u = vel[..., 0][self.ii, self.jj]
        v = vel[..., 1][self.ii, self.jj]
        heights = h[self.ii, self.jj]
        position = np.column_stack(
            (self.xc[self.ii], self.yc[self.jj], 0.5 * heights)
        )
        velocity = np.column_stack((u, v, np.zeros_like(u)))
        mass = self.rho * heights * self.cell_area
        volume = heights * self.cell_area

        frame = StateFrame(
            time=float(self.t),
            position=position,
            velocity=velocity,
            mass=mass.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "water_height": heights.copy(),
            },
            fields={
                "density_field": h,
                "velocity_field": vel,
                "active_mask": m.astype(np.uint8),
                "continuity_source": np.zeros((self.nx, self.ny), dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "boundary_normal": self.normals,
                "boundary_target_velocity": self.boundary_target_velocity,
                "boundary_velocity": self.boundary_velocity,
            },
            scalars={
                "grid_spacing": float(self.dx),
                "reference_velocity": float(self.reference_velocity),
                "initial_water_volume": float(self.initial_volume),
                "water_volume": float(np.sum(volume)),
                "mass_balance_error": float(
                    self.rho * np.sum(volume) - self.initial_mass
                ),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        volume = float(np.sum(self.h) * self.cell_area)
        return {
            "scheme": "finite_volume_HLL_MUSCL_SSP_RK2",
            "grid_shape": [self.nx, self.ny],
            "grid_spacing": float(self.dx),
            "gravity": float(self.g),
            "cfl": float(self.cfl),
            "steps_taken": int(self.step_count),
            "final_time": float(self.t),
            "initial_mass": float(self.initial_mass),
            "final_mass": float(self.rho * volume),
            "mass_balance_error": float(self.rho * volume - self.initial_mass),
            "reference_velocity": float(self.reference_velocity),
        }
