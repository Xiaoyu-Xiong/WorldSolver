"""Snow castle destruction via explicit MLS-MPM snow elastoplasticity (NumPy).

Physical model (Stomakhin et al. 2013, "A Material Point Method for Snow"):

* fixed-corotated elasticity evaluated on the elastic deformation gradient,
* singular-value return-mapping plasticity with critical compression/stretch
  thresholds, which produces fracture surfaces and permanent compaction,
* Jp-driven exponential hardening of the Lame parameters so compacted snow
  stiffens and the deposited pile becomes stable.

Discretisation: explicit MLS-MPM with quadratic B-spline transfers and APIC
affine velocities, advanced by symplectic Euler:

    P2G (mass, momentum, internal stress) -> grid velocity update with gravity,
    floor and striker contact -> G2P (velocity, affine matrix, F update).

Boundaries: the frozen static snow floor (top surface z = 0) with high Coulomb
friction, the frozen domain walls, and the rigid spherical striker acting as a
moving collider. The striker follows the frozen prescribed impact velocity
while ploughing the castle footprint (the scene's destruction script) and
afterwards evolves as a free rigid body under gravity and floor friction.
"""
from __future__ import annotations

import time as _wallclock

import numpy as np

from worldsolver.contracts import StateFrame

_I3 = np.eye(3, dtype=np.float64)


class BenchmarkSolver:
    """Explicit MLS-MPM snow solver for the snow-castle destruction scene."""

    YOUNG = 1.4e5          # Pa, initial Young's modulus of packed snow
    POISSON = 0.2          # dimensionless
    HARDENING = 6.0        # xi, plastic hardening coefficient
    HARDEN_MIN = 0.35
    HARDEN_MAX = 2.0
    THETA_C = 0.050        # critical compression
    THETA_S = 0.0075       # critical stretch
    JP_MIN = 0.55
    JP_MAX = 8.0
    CFL = 0.28
    DX = 0.15              # grid spacing (2 particles per cell per axis)
    GRID_DAMPING = 0.25    # 1/s, mild dissipation so the pile settles
    MU_FLOOR = 0.75        # high-friction snow floor
    MU_BALL = 0.40         # striker surface friction
    MU_BALL_FLOOR = 0.90   # striker friction against the snow floor
    VMAX = 25.0            # m/s robustness clamp
    CONTACT_BAND_CELLS = 4.0
    WALLCLOCK_GUARD = 1500.0  # s, last-resort reduced-cost fallback

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self._t0 = _wallclock.perf_counter()
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        initial = dict(config.get("initial_conditions", {}) or {})

        frame0 = scene.initial_state()
        position = np.asarray(frame0.position, dtype=np.float64).copy()
        velocity = np.asarray(frame0.velocity, dtype=np.float64).copy()
        mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        body = np.asarray(frame0.body_id, dtype=np.int64).copy()

        self.n_total = int(position.shape[0])
        self.body_id = body.astype(np.int32)
        self.mass = mass
        self.snow_sel = np.flatnonzero(body == 1)
        ball_sel = np.flatnonzero(body == 2)
        if ball_sel.size == 0 or self.snow_sel.size == 0:
            ball_sel = np.array([self.n_total - 1], dtype=np.int64)
            self.snow_sel = np.arange(self.n_total - 1, dtype=np.int64)
            self.body_id = np.ones(self.n_total, dtype=np.int32)
            self.body_id[-1] = 2
        self.ball_sel = ball_sel
        self.n_snow = int(self.snow_sel.size)

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        ).reshape(3)

        # snow particle state
        self.x = position[self.snow_sel].copy()
        self.v = velocity[self.snow_sel].copy()
        self.F = np.tile(_I3, (self.n_snow, 1, 1))
        self.C = np.zeros((self.n_snow, 3, 3), dtype=np.float64)
        self.Jp = np.ones(self.n_snow, dtype=np.float64)

        self.spacing = float(initial.get("snow_particle_spacing", 0.075))
        self.density = float(initial.get("snow_bulk_density", 420.0))
        self.p_mass = float(np.mean(mass[self.snow_sel]))
        self.p_vol = self.p_mass / max(self.density, 1e-9)

        # rigid striker
        self.ball_radius = float(initial.get("striker_radius", 0.42))
        self.ball_x = position[self.ball_sel[0]].copy()
        drive = np.asarray(
            initial.get("striker_velocity", [5.8, 0.0, 0.0]), dtype=np.float64
        ).reshape(3)
        v_ball0 = velocity[self.ball_sel[0]]
        self.ball_v = (v_ball0.copy() if np.any(v_ball0) else drive.copy())
        self.ball_mass = float(mass[self.ball_sel[0]])
        self.ball_drive_vx = float(self.ball_v[0])
        # prescribed-velocity ploughing ends once the striker clears the castle
        self.ball_release_x = 1.85
        self.ball_driven = True

        # Lame parameters
        self.mu0 = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        self.lam0 = self.YOUNG * self.POISSON / (
            (1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON)
        )
        self.xi = self.HARDENING

        # explicit time step from the elastic wave speed (CFL) and striker speed
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.duration = float(config.get("duration", 4.0))
        stiff = (self.lam0 + 2.0 * self.mu0) * self.HARDEN_MAX
        wave = float(np.sqrt(stiff / max(self.density, 1e-9)))
        dt_cfl = self.CFL * self.DX / max(wave, 1e-9)
        dt_cfl = min(dt_cfl, 0.35 * self.DX / max(abs(self.ball_drive_vx), 1.0))
        self.substeps = max(1, int(np.ceil(self.sample_dt / dt_cfl)))
        self.dt = self.sample_dt / self.substeps
        self.step_count = 0
        self.sim_time = 0.0
        self._fallback = False
        self._max_speed = 0.0
        self._first_contact_time = -1.0

        self._build_grid()
        self._external_force = (self.mass[:, None] * self.gravity[None, :]).astype(np.float32)
        self._mass_source = np.zeros(1, dtype=np.float64)
        self._initial_total_mass = float(self.mass.sum())

    # ------------------------------------------------------------------ grid
    def _build_grid(self) -> None:
        dx = self.DX
        lo = np.array([-3.15, -2.55, -0.45], dtype=np.float64)
        hi = np.array([5.10, 2.55, 3.90], dtype=np.float64)
        counts = np.ceil((hi - lo) / dx).astype(np.int64) + 1
        self.origin = lo
        self.nx, self.ny, self.nz = int(counts[0]), int(counts[1]), int(counts[2])
        self.n_nodes = self.nx * self.ny * self.nz
        self.stride = np.array([self.ny * self.nz, self.nz, 1], dtype=np.int64)
        self.index_hi = np.array([self.nx - 3, self.ny - 3, self.nz - 3], dtype=np.int64)

        gi, gj, gk = np.meshgrid(
            np.arange(self.nx, dtype=np.float64),
            np.arange(self.ny, dtype=np.float64),
            np.arange(self.nz, dtype=np.float64),
            indexing="ij",
        )
        gi = gi.reshape(-1)
        gj = gj.reshape(-1)
        gk = gk.reshape(-1)
        self.node_pos = np.stack(
            (lo[0] + gi * dx, lo[1] + gj * dx, lo[2] + gk * dx), axis=1
        )
        node_z = self.node_pos[:, 2]
        self.floor_mask = node_z <= 1e-9
        self.deep_floor_mask = node_z <= -0.5 * dx - 1e-9
        wall = (gi < 2.0) | (gi > self.nx - 3.0) | (gj < 2.0) | (gj > self.ny - 3.0)
        wall |= gk > self.nz - 3.0
        self.wall_mask = wall

        self.grid_m = np.zeros(self.n_nodes, dtype=np.float64)
        self.grid_v = np.zeros((self.n_nodes, 3), dtype=np.float64)

    # --------------------------------------------------------------- physics
    def _stress(self) -> np.ndarray:
        """Return-mapping plasticity plus hardening, then Kirchhoff stress."""
        F = self.F
        finite = np.isfinite(F).reshape(self.n_snow, 9).all(axis=1)
        if not finite.all():
            F[~finite] = _I3
        U, S, Vh = np.linalg.svd(F)
        flip = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
        if flip.any():
            U[flip, :, 2] *= -1.0
            S[flip, 2] *= -1.0
        Sc = np.clip(S, 1.0 - self.THETA_C, 1.0 + self.THETA_S)
        Sc = np.where(np.abs(Sc) < 0.05, 0.05, Sc)
        j_trial = S[:, 0] * S[:, 1] * S[:, 2]
        j_elastic = Sc[:, 0] * Sc[:, 1] * Sc[:, 2]
        safe_je = np.where(np.abs(j_elastic) < 1e-8, 1e-8, j_elastic)
        self.Jp = np.clip(self.Jp * j_trial / safe_je, self.JP_MIN, self.JP_MAX)
        Fe = (U * Sc[:, None, :]) @ Vh
        self.F = Fe
        R = U @ Vh
        harden = np.clip(
            np.exp(self.xi * (1.0 - self.Jp)), self.HARDEN_MIN, self.HARDEN_MAX
        )
        mu = self.mu0 * harden
        lam = self.lam0 * harden
        tau = (2.0 * mu)[:, None, None] * ((Fe - R) @ np.transpose(Fe, (0, 2, 1)))
        vol = lam * (j_elastic - 1.0) * j_elastic
        tau[:, 0, 0] += vol
        tau[:, 1, 1] += vol
        tau[:, 2, 2] += vol
        return tau

    def _weights(self):
        dx = self.DX
        grid_x = (self.x - self.origin[None, :]) / dx
        base = np.floor(grid_x - 0.5).astype(np.int64)
        np.clip(base, 0, self.index_hi, out=base)
        fx = grid_x - base
        weights = (
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        )
        dpos = tuple((float(i) - fx) * dx for i in range(3))
        return base @ self.stride, weights, dpos

    def _p2g(self, dt: float) -> None:
        tau = self._stress()
        coeff = -dt * self.p_vol * 4.0 / (self.DX * self.DX)
        affine = self.p_mass * self.C + coeff * tau
        base_flat, weights, dpos = self._weights()
        self.grid_m[:] = 0.0
        self.grid_v[:] = 0.0
        n_nodes = self.n_nodes
        mv = self.p_mass * self.v
        s0, s1 = int(self.stride[0]), int(self.stride[1])
        for i in range(3):
            wi = weights[i][:, 0]
            dpi = dpos[i][:, 0]
            for j in range(3):
                wij = wi * weights[j][:, 1]
                dpj = dpos[j][:, 1]
                for k in range(3):
                    wt = wij * weights[k][:, 2]
                    dpk = dpos[k][:, 2]
                    idx = base_flat + (i * s0 + j * s1 + k)
                    self.grid_m += np.bincount(idx, weights=wt, minlength=n_nodes)
                    for a in range(3):
                        contrib = wt * (
                            mv[:, a]
                            + affine[:, a, 0] * dpi
                            + affine[:, a, 1] * dpj
                            + affine[:, a, 2] * dpk
                        )
                        self.grid_v[:, a] += np.bincount(
                            idx, weights=contrib, minlength=n_nodes
                        )

    def _grid_update(self, dt: float) -> None:
        active = self.grid_m > 1e-12
        inv_m = np.zeros_like(self.grid_m)
        inv_m[active] = 1.0 / self.grid_m[active]
        v = self.grid_v * inv_m[:, None]
        v[~active] = 0.0
        v += dt * self.gravity[None, :]
        v *= float(np.exp(-self.GRID_DAMPING * dt))

        # static snow floor, top surface z = 0, high Coulomb friction
        mask = self.floor_mask & active
        if mask.any():
            vm = v[mask]
            hit = vm[:, 2] < 0.0
            if hit.any():
                sub = vm[hit]
                vn = sub[:, 2].copy()
                sub[:, 2] = 0.0
                tan = np.sqrt(sub[:, 0] ** 2 + sub[:, 1] ** 2)
                scale = np.maximum(
                    0.0, 1.0 - self.MU_FLOOR * np.abs(vn) / np.maximum(tan, 1e-9)
                )
                sub[:, 0] *= scale
                sub[:, 1] *= scale
                vm[hit] = sub
            v[mask] = vm
        v[self.deep_floor_mask] = 0.0
        v[self.wall_mask] = 0.0

        # moving rigid striker collider
        d = self.node_pos - self.ball_x[None, :]
        dist2 = np.einsum("ij,ij->i", d, d)
        rad = self.ball_radius + 0.25 * self.DX
        near = (dist2 < rad * rad) & active
        if near.any():
            dn = d[near]
            dist = np.sqrt(np.maximum(dist2[near], 1e-18))
            normal = dn / dist[:, None]
            v_rel = v[near] - self.ball_v[None, :]
            vn = np.einsum("ij,ij->i", v_rel, normal)
            approach = vn < 0.0
            if approach.any():
                sub = v_rel[approach]
                n_sub = normal[approach]
                vn_sub = vn[approach]
                sub = sub - vn_sub[:, None] * n_sub
                tan = np.sqrt(np.einsum("ij,ij->i", sub, sub))
                scale = np.maximum(
                    0.0, 1.0 - self.MU_BALL * np.abs(vn_sub) / np.maximum(tan, 1e-9)
                )
                v_rel[approach] = sub * scale[:, None]
            v[near] = v_rel + self.ball_v[None, :]

        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        over = speed > self.VMAX
        if over.any():
            v[over] *= (self.VMAX / speed[over])[:, None]
        self.grid_v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def _g2p(self, dt: float) -> None:
        base_flat, weights, dpos = self._weights()
        new_v = np.zeros_like(self.v)
        new_C = np.zeros_like(self.C)
        scale = 4.0 / (self.DX * self.DX)
        s0, s1 = int(self.stride[0]), int(self.stride[1])
        for i in range(3):
            wi = weights[i][:, 0]
            dpi = dpos[i][:, 0]
            for j in range(3):
                wij = wi * weights[j][:, 1]
                dpj = dpos[j][:, 1]
                for k in range(3):
                    wt = wij * weights[k][:, 2]
                    dpk = dpos[k][:, 2]
                    gv = self.grid_v[base_flat + (i * s0 + j * s1 + k)]
                    new_v += wt[:, None] * gv
                    wgv = (scale * wt)[:, None] * gv
                    new_C[:, :, 0] += wgv * dpi[:, None]
                    new_C[:, :, 1] += wgv * dpj[:, None]
                    new_C[:, :, 2] += wgv * dpk[:, None]
        self.v = new_v
        self.C = new_C
        self.x = self.x + dt * self.v
        self.F = (_I3[None, :, :] + dt * self.C) @ self.F

    def _particle_boundaries(self) -> None:
        # striker: project particles out of the sphere, match surface velocity
        d = self.x - self.ball_x[None, :]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        inside = dist < self.ball_radius
        if inside.any():
            normal = d[inside] / np.maximum(dist[inside], 1e-9)[:, None]
            self.x[inside] = self.ball_x[None, :] + normal * self.ball_radius
            v_rel = self.v[inside] - self.ball_v[None, :]
            vn = np.einsum("ij,ij->i", v_rel, normal)
            neg = vn < 0.0
            if neg.any():
                v_rel[neg] -= vn[neg, None] * normal[neg]
            self.v[inside] = v_rel + self.ball_v[None, :]
            if self._first_contact_time < 0.0:
                self._first_contact_time = float(self.step_count * self.dt)

        # static snow floor
        below = self.x[:, 2] < 0.0
        if below.any():
            idx = np.flatnonzero(below)
            self.x[idx, 2] = 0.0
            vz = self.v[idx, 2]
            hit = vz < 0.0
            if hit.any():
                hid = idx[hit]
                vn = np.abs(vz[hit])
                self.v[hid, 2] = 0.0
                tan = np.sqrt(self.v[hid, 0] ** 2 + self.v[hid, 1] ** 2)
                scale = np.maximum(
                    0.0, 1.0 - self.MU_FLOOR * vn / np.maximum(tan, 1e-9)
                )
                self.v[hid, 0] *= scale
                self.v[hid, 1] *= scale

        # keep particles inside the grid interior
        lo = self.origin + 2.05 * self.DX
        hi = self.origin + self.index_hi.astype(np.float64) * self.DX
        for a in range(3):
            low = self.x[:, a] < lo[a]
            if low.any():
                self.x[low, a] = lo[a]
                self.v[low, a] = np.maximum(self.v[low, a], 0.0)
            high = self.x[:, a] > hi[a]
            if high.any():
                self.x[high, a] = hi[a]
                self.v[high, a] = np.minimum(self.v[high, a], 0.0)
        np.maximum(self.x[:, 2], 0.0, out=self.x[:, 2])

        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        over = speed > self.VMAX
        if over.any():
            self.v[over] *= (self.VMAX / speed[over])[:, None]
        if speed.size:
            self._max_speed = max(self._max_speed, float(speed.max()))

    def _advance_ball(self, dt: float) -> None:
        g = float(self.gravity[2])
        if self.ball_driven and self.ball_x[0] >= self.ball_release_x:
            self.ball_driven = False
        if self.ball_driven:
            # frozen prescribed impact velocity while ploughing the castle
            self.ball_v[0] = self.ball_drive_vx
            self.ball_v[1] = 0.0
            self.ball_v[2] += g * dt
        else:
            self.ball_v[2] += g * dt
            if self.ball_x[2] - self.ball_radius <= 1e-6:
                speed = float(np.hypot(self.ball_v[0], self.ball_v[1]))
                if speed > 1e-9:
                    factor = max(0.0, 1.0 - self.MU_BALL_FLOOR * abs(g) * dt / speed)
                    self.ball_v[0] *= factor
                    self.ball_v[1] *= factor
        self.ball_x = self.ball_x + dt * self.ball_v
        if self.ball_x[2] - self.ball_radius < 0.0:
            self.ball_x[2] = self.ball_radius
            if self.ball_v[2] < 0.0:
                self.ball_v[2] = 0.0

    def _step(self, dt: float) -> None:
        if self._fallback:
            self.v += dt * self.gravity[None, :]
            self.v *= float(np.exp(-1.5 * dt))
            self.x = self.x + dt * self.v
            self._particle_boundaries()
        else:
            self._p2g(dt)
            self._grid_update(dt)
            self._g2p(dt)
            self._particle_boundaries()
        self._advance_ball(dt)
        self.step_count += 1
        self.sim_time = self.step_count * self.dt

    # --------------------------------------------------------------- outputs
    def _contact_signed_distance(self) -> np.ndarray:
        """Signed gaps of near-contact snow samples plus the striker-floor gap."""
        gap_floor = self.x[:, 2]
        d = self.x - self.ball_x[None, :]
        gap_ball = np.sqrt(np.einsum("ij,ij->i", d, d)) - self.ball_radius
        gap = np.minimum(gap_floor, gap_ball)
        band = self.CONTACT_BAND_CELLS * self.spacing
        values = gap[gap < band]
        if values.size == 0:
            values = np.array([float(gap.min())], dtype=np.float64)
        ball_floor_gap = float(self.ball_x[2] - self.ball_radius)
        return np.concatenate(
            (values, np.array([ball_floor_gap], dtype=np.float64))
        ).astype(np.float64)

    def _frame(self, time: float) -> StateFrame:
        position = np.empty((self.n_total, 3), dtype=np.float64)
        velocity = np.empty((self.n_total, 3), dtype=np.float64)
        position[self.snow_sel] = self.x
        velocity[self.snow_sel] = self.v
        position[self.ball_sel] = self.ball_x[None, :]
        velocity[self.ball_sel] = self.ball_v[None, :]

        deformation = np.empty((self.n_total, 3, 3), dtype=np.float64)
        deformation[self.snow_sel] = self.F
        deformation[self.ball_sel] = _I3

        volume = np.empty(self.n_total, dtype=np.float64)
        volume[self.snow_sel] = self.p_vol * np.abs(np.linalg.det(self.F))
        volume[self.ball_sel] = (4.0 / 3.0) * np.pi * self.ball_radius ** 3

        plastic = np.ones(self.n_total, dtype=np.float64)
        plastic[self.snow_sel] = self.Jp

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": deformation.astype(np.float32),
                "mass_quantity": self.mass.copy(),
                "volume_quantity": volume,
                "plastic_jacobian": plastic,
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": self._mass_source.copy(),
                "external_force": self._external_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "characteristic_length": float(self.spacing),
                "grid_spacing": float(self.DX),
                "initial_total_mass": float(self._initial_total_mass),
                "total_mass": float(self.mass.sum()),
                "time_step": float(self.dt),
            },
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------------- drive
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        target_steps = int(round(target / self.dt))
        while self.step_count < target_steps:
            if (
                not self._fallback
                and _wallclock.perf_counter() - self._t0 > self.WALLCLOCK_GUARD
            ):
                self._fallback = True
            self._step(self.dt)
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": (
                "explicit MLS-MPM snow elastoplasticity: fixed-corotated elasticity, "
                "SVD return-mapping plasticity, Jp hardening"
            ),
            "snow_particles": int(self.n_snow),
            "grid_nodes": int(self.n_nodes),
            "grid_spacing": float(self.DX),
            "substeps_per_sample": int(self.substeps),
            "time_step": float(self.dt),
            "total_steps": int(self.step_count),
            "young_modulus": float(self.YOUNG),
            "poisson_ratio": float(self.POISSON),
            "critical_compression": float(self.THETA_C),
            "critical_stretch": float(self.THETA_S),
            "hardening_coefficient": float(self.HARDENING),
            "floor_friction": float(self.MU_FLOOR),
            "striker_radius": float(self.ball_radius),
            "striker_release_x": float(self.ball_release_x),
            "striker_final_position": [float(c) for c in self.ball_x],
            "striker_still_driven": bool(self.ball_driven),
            "first_striker_contact_time": float(self._first_contact_time),
            "max_particle_speed": float(self._max_speed),
            "min_plastic_jacobian": float(self.Jp.min()),
            "mean_plastic_jacobian": float(self.Jp.mean()),
            "total_mass": float(self.mass.sum()),
            "reduced_cost_fallback_used": bool(self._fallback),
            "wallclock_seconds": float(_wallclock.perf_counter() - self._t0),
        }
