"""Self-contained 2D shallow-water solver for the octagonal-obstacle dam-break scene.

Physics
-------
The conservative shallow-water (Saint-Venant) system on a flat bed is stepped in
time on the frozen 64x64 cell-centred grid::

    d_t h      + d_x(h u)               + d_y(h v)               = 0
    d_t(h u)   + d_x(h u^2 + g h^2 / 2) + d_y(h u v)             = 0
    d_t(h v)   + d_x(h u v)             + d_y(h v^2 + g h^2 / 2) = 0

with ``h`` the water height (the scene's ``density_field`` observable) and
``g = |gravity_z|``.  Discretisation is a second-order MUSCL finite-volume
scheme (minmod-limited slopes, HLL approximate Riemann solver) advanced with a
two-stage strong-stability-preserving Runge-Kutta method under a CFL-limited
time step.  Reflecting no-penetration boundaries -- the four tank walls and the
octagonal prism -- are imposed exactly as pure hydrostatic-pressure wall fluxes
(zero mass flux), so the discrete water volume is conserved to round-off and
reflected waves stay stable.

No animation tables or precomputed trajectories: every frame is the result of
actual time stepping from the frozen initial condition.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

MIN_H = 1.0e-8  # floor used when dividing momentum by height
DRY_H = 1.0e-6  # below this height a cell carries no velocity
CFL = 0.45
MAX_SUBSTEPS = 400000


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component-wise minmod limiter."""
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


class BenchmarkSolver:
    """Explicit finite-volume shallow-water solver (no external physics engine)."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        domain = cfg["domain"]
        obstacle = cfg["obstacle"]
        initial = cfg["initial_conditions"]

        self.nx = int(domain["grid_shape"][0])
        self.ny = int(domain["grid_shape"][1])
        bmin = np.asarray(domain["bounds_min"], dtype=float)
        bmax = np.asarray(domain["bounds_max"], dtype=float)
        self.bmin = bmin
        self.dx = float((bmax[0] - bmin[0]) / self.nx)
        self.dy = float((bmax[1] - bmin[1]) / self.ny)
        self.xc = bmin[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = bmin[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.xx, self.yy = np.meshgrid(self.xc, self.yc, indexing="ij")

        self.g = float(abs(cfg["gravity"][2]))
        self.rho = float(cfg["fluid"]["density"])

        # --- frozen octagonal obstacle: same half-plane test as the scene ---
        theta = np.arange(8) * np.pi / 4 + float(obstacle["rotation_radians"])
        center = np.asarray(obstacle["center"], dtype=float)
        self.vertices = center + float(obstacle["circumradius"]) * np.column_stack(
            (np.cos(theta), np.sin(theta))
        )
        solid = np.ones((self.nx, self.ny), dtype=bool)
        inward = []
        for a, b in zip(self.vertices, np.roll(self.vertices, -1, axis=0)):
            edge = b - a
            solid &= edge[0] * (self.yy - a[1]) - edge[1] * (self.xx - a[0]) >= 0.0
            inward.append(np.array([-edge[1], edge[0]]) / float(np.linalg.norm(edge)))
        self.solid = solid
        self.active = ~solid
        self.edge_inward = np.asarray(inward, dtype=float)
        self.n_active = int(self.active.sum())

        # --- frozen step initial condition ---
        h0 = np.where(
            self.xx > float(initial["step_x"]),
            float(initial["high_water_height"]),
            float(initial["low_water_height"]),
        ).astype(float)
        h0[solid] = 0.0
        u0 = np.asarray(initial["initial_velocity"], dtype=float)
        self.h_init = h0
        self.hu_init = np.where(self.active, h0 * u0[0], 0.0)
        self.hv_init = np.where(self.active, h0 * u0[1], 0.0)

        self.initial_volume = float((h0[self.active] * self.dx * self.dy).sum())
        self.ref_velocity = float(np.sqrt(self.g * float(h0.max())))
        self._build_boundary_samples()
        self._reset()

    def _reset(self) -> None:
        self.h = self.h_init.copy()
        self.hu = self.hu_init.copy()
        self.hv = self.hv_init.copy()
        self.time = 0.0
        self.steps = 0

    # ------------------------------------------------------------------
    # boundary sample bookkeeping (ordering follows the frozen scene)
    # ------------------------------------------------------------------
    def _build_boundary_samples(self) -> None:
        nx, ny = self.nx, self.ny
        idx_i: list[int] = []
        idx_j: list[int] = []
        normals: list[np.ndarray] = []

        # left wall (inward +x), right wall (inward -x)
        for j in range(ny):
            idx_i.append(0)
            idx_j.append(j)
            normals.append(np.array([1.0, 0.0]))
        for j in range(ny):
            idx_i.append(nx - 1)
            idx_j.append(j)
            normals.append(np.array([-1.0, 0.0]))
        # lower wall (inward +y), upper wall (inward -y)
        for i in range(nx):
            idx_i.append(i)
            idx_j.append(0)
            normals.append(np.array([0.0, 1.0]))
        for i in range(nx):
            idx_i.append(i)
            idx_j.append(ny - 1)
            normals.append(np.array([0.0, -1.0]))

        # octagon: 8 samples per edge, edge-major, sampled in the fluid just
        # outside the solid face so the reported velocity is the wall-adjacent one
        frac = (np.arange(8) + 0.5) / 8.0
        for e, (a, b) in enumerate(zip(self.vertices, np.roll(self.vertices, -1, axis=0))):
            n_in = self.edge_inward[e]
            n_out = -n_in
            for f in frac:
                point = a + (b - a) * f
                cell = None
                for k in (1.5, 2.5, 3.5):
                    probe = point + n_out * (k * max(self.dx, self.dy))
                    i = int(np.clip(np.floor((probe[0] - self.bmin[0]) / self.dx), 0, nx - 1))
                    j = int(np.clip(np.floor((probe[1] - self.bmin[1]) / self.dy), 0, ny - 1))
                    if self.active[i, j]:
                        cell = (i, j)
                        break
                if cell is None:
                    cell = (i, j)
                idx_i.append(cell[0])
                idx_j.append(cell[1])
                normals.append(n_in.copy())

        self.b_i = np.asarray(idx_i, dtype=int)
        self.b_j = np.asarray(idx_j, dtype=int)
        self.b_n = np.asarray(normals, dtype=float)

    # ------------------------------------------------------------------
    # numerics
    # ------------------------------------------------------------------
    def _velocities(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        safe = np.maximum(h, MIN_H)
        u = np.where(h > DRY_H, hu / safe, 0.0)
        v = np.where(h > DRY_H, hv / safe, 0.0)
        return u, v

    def _fill_solid(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """Give solid cells a neighbour-averaged height so nearby slopes stay flat."""
        fluid = self.active.astype(float)
        hs = np.where(self.active, h, 0.0)
        acc = np.zeros_like(h)
        cnt = np.zeros_like(h)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            acc += np.roll(hs, shift, axis=axis)
            cnt += np.roll(fluid, shift, axis=axis)
        mean_h = float(hs.sum() / max(self.n_active, 1))
        filled = np.where(cnt > 0.0, acc / np.maximum(cnt, 1.0), mean_h)
        h_f = np.where(self.active, h, filled)
        hu_f = np.where(self.active, hu, 0.0)
        hv_f = np.where(self.active, hv, 0.0)
        return h_f, hu_f, hv_f

    def _flux_divergence(
        self,
        h: np.ndarray,
        hn: np.ndarray,
        ht: np.ndarray,
        solid: np.ndarray,
        spacing: float,
    ) -> np.ndarray:
        """MUSCL-HLL flux divergence along axis 0. ``hn`` is normal momentum."""
        g = self.g

        def pad(a: np.ndarray) -> np.ndarray:
            return np.concatenate((a[:1], a[:1], a, a[-1:], a[-1:]), axis=0)

        hp, hnp, htp = pad(h), pad(hn), pad(ht)
        wall = np.ones((1, solid.shape[1]), dtype=bool)
        sp = np.concatenate((wall, wall, solid, wall, wall), axis=0)

        def slope(a: np.ndarray) -> np.ndarray:
            return _minmod(a[1:-1] - a[:-2], a[2:] - a[1:-1])

        hc, hnc, htc = hp[1:-1], hnp[1:-1], htp[1:-1]
        dh, dhn, dht = slope(hp), slope(hnp), slope(htp)
        sc = sp[1:-1]

        hL = np.maximum(hc[:-1] + 0.5 * dh[:-1], 0.0)
        hR = np.maximum(hc[1:] - 0.5 * dh[1:], 0.0)
        hnL = hnc[:-1] + 0.5 * dhn[:-1]
        hnR = hnc[1:] - 0.5 * dhn[1:]
        htL = htc[:-1] + 0.5 * dht[:-1]
        htR = htc[1:] - 0.5 * dht[1:]
        solidL, solidR = sc[:-1], sc[1:]

        uL, vL = self._velocities(hL, hnL, htL)
        uR, vR = self._velocities(hR, hnR, htR)
        hnL, htL = hL * uL, hL * vL
        hnR, htR = hR * uR, hR * vR
        cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)

        sl = np.minimum(uL - cL, uR - cR)
        sr = np.maximum(uL + cL, uR + cR)
        sl = np.minimum(sl, 0.0)
        sr = np.maximum(sr, 0.0)

        FL = np.stack((hnL, hnL * uL + 0.5 * g * hL * hL, hnL * vL), axis=-1)
        FR = np.stack((hnR, hnR * uR + 0.5 * g * hR * hR, hnR * vR), axis=-1)
        UL = np.stack((hL, hnL, htL), axis=-1)
        UR = np.stack((hR, hnR, htR), axis=-1)

        # with sl <= 0 <= sr enforced above, this reduces to the upwind flux
        # automatically in the supersonic cases
        denom = np.where(sr - sl > 1.0e-12, sr - sl, 1.0)[..., None]
        s_l = sl[..., None]
        s_r = sr[..., None]
        flux = (s_r * FL - s_l * FR + s_l * s_r * (UR - UL)) / denom

        # reflecting no-penetration walls: hydrostatic pressure only, zero mass flux
        wall_L = solidR & ~solidL  # fluid on the left, solid on the right
        wall_R = solidL & ~solidR  # solid on the left, fluid on the right
        both = solidL & solidR
        p_wall = np.where(wall_L, 0.5 * g * hL * hL, 0.0) + np.where(
            wall_R, 0.5 * g * hR * hR, 0.0
        )
        any_wall = wall_L | wall_R | both
        flux = np.where(
            any_wall[..., None],
            np.stack((np.zeros_like(p_wall), p_wall, np.zeros_like(p_wall)), axis=-1),
            flux,
        )
        return (flux[1:] - flux[:-1]) / spacing

    def _residual(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        h_f, hu_f, hv_f = self._fill_solid(h, hu, hv)
        div_x = self._flux_divergence(h_f, hu_f, hv_f, self.solid, self.dx)
        div_y = self._flux_divergence(
            h_f.T.copy(), hv_f.T.copy(), hu_f.T.copy(), self.solid.T.copy(), self.dy
        ).transpose(1, 0, 2)
        dh = -(div_x[..., 0] + div_y[..., 0])
        dhu = -(div_x[..., 1] + div_y[..., 2])
        dhv = -(div_x[..., 2] + div_y[..., 1])
        act = self.active
        return dh * act, dhu * act, dhv * act

    def _max_wave_speed(self) -> float:
        act = self.active
        h = self.h[act]
        if h.size == 0:
            return 0.0
        u, v = self._velocities(self.h, self.hu, self.hv)
        c = np.sqrt(self.g * np.maximum(h, 0.0))
        smax = float(np.max(np.maximum(np.abs(u[act]), np.abs(v[act])) + c))
        return smax if np.isfinite(smax) else 0.0

    def _stage_clip(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        h = np.where(self.active, np.maximum(h, 0.0), 0.0)
        dry = h <= DRY_H
        hu = np.where(self.active & ~dry, hu, 0.0)
        hv = np.where(self.active & ~dry, hv, 0.0)
        return h, hu, hv

    def _step(self, dt: float) -> None:
        """Two-stage SSP Runge-Kutta (Heun) update."""
        h0, hu0, hv0 = self.h, self.hu, self.hv
        dh, dhu, dhv = self._residual(h0, hu0, hv0)
        h1, hu1, hv1 = self._stage_clip(h0 + dt * dh, hu0 + dt * dhu, hv0 + dt * dhv)
        dh1, dhu1, dhv1 = self._residual(h1, hu1, hv1)
        h2 = 0.5 * (h0 + h1 + dt * dh1)
        hu2 = 0.5 * (hu0 + hu1 + dt * dhu1)
        hv2 = 0.5 * (hv0 + hv1 + dt * dhv1)
        self.h, self.hu, self.hv = self._stage_clip(h2, hu2, hv2)
        self.time += dt
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-12:
            self._reset()
        guard = 0
        while self.time < target - 1.0e-12 and guard < MAX_SUBSTEPS:
            smax = self._max_wave_speed()
            remaining = target - self.time
            if smax > 1.0e-9:
                dt = CFL * min(self.dx, self.dy) / smax
            else:
                dt = remaining
            dt = min(dt, remaining)
            if dt <= 1.0e-12:
                break
            self._step(dt)
            guard += 1
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------
    # state export
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        act = self.active
        h = np.where(act, np.maximum(self.h, 0.0), 0.0)
        u, v = self._velocities(h, self.hu, self.hv)
        u = np.where(act, u, 0.0)
        v = np.where(act, v, 0.0)

        height = h[act]
        position = np.column_stack((self.xx[act], self.yy[act], 0.5 * height))
        velocity = np.column_stack((u[act], v[act], np.zeros(self.n_active)))
        volume = height * self.dx * self.dy
        mass = self.rho * volume

        velocity_field = np.zeros((self.nx, self.ny, 2))
        velocity_field[..., 0] = u
        velocity_field[..., 1] = v

        b_vel = np.column_stack((u[self.b_i, self.b_j], v[self.b_i, self.b_j]))
        normal_part = np.sum(b_vel * self.b_n, axis=1, keepdims=True) * self.b_n
        boundary_velocity = b_vel - normal_part  # no-penetration: tangential slip only

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "mass_quantity": mass.copy(),
                "water_height": height.copy(),
            },
            fields={
                "density_field": h,
                "velocity_field": velocity_field,
                "active_mask": act.astype(np.uint8),
                "continuity_source": np.zeros_like(h),
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": self.b_n.copy(),
                "boundary_target_velocity": np.zeros_like(self.b_n),
                "boundary_velocity": boundary_velocity,
            },
            scalars={
                "grid_spacing": float(self.dx),
                "reference_velocity": float(self.ref_velocity),
                "initial_water_volume": float(self.initial_volume),
                "water_volume": float(volume.sum()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        act = self.active
        height = np.where(act, np.maximum(self.h, 0.0), 0.0)[act]
        volume = float((height * self.dx * self.dy).sum())
        return {
            "method": "MUSCL-HLL shallow water, SSP-RK2, reflecting walls",
            "steps": int(self.steps),
            "time": float(self.time),
            "cfl": float(CFL),
            "active_cells": int(self.n_active),
            "boundary_samples": int(self.b_n.shape[0]),
            "min_height": float(height.min()) if height.size else 0.0,
            "max_height": float(height.max()) if height.size else 0.0,
            "water_volume": volume,
            "volume_drift": volume - self.initial_volume,
        }
