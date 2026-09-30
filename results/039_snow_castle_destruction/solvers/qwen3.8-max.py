"""Physically-based MPM solver for snow-castle destruction.

Implements a classic explicit Material Point Method (Stomakhin et al. snow
model) with fixed-corotated elasticity, plastic clamping (compaction and
tensile fracture), a static frictional floor, and a kinematic rigid sphere
striker. All required StateFrame fields are exported.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", {}) or {}
        ic = cfg.get("initial_conditions", {}) or {}

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=float)
        vel = np.asarray(frame0.velocity, dtype=float)
        mass = np.asarray(frame0.mass, dtype=float)
        bid = np.asarray(frame0.body_id).astype(np.int64)

        snow = bid == 1
        ball = bid == 2

        self.xp = np.ascontiguousarray(pos[snow], dtype=float)
        self.vp = np.ascontiguousarray(vel[snow], dtype=float)
        self.mp = np.ascontiguousarray(mass[snow], dtype=float)
        self.N = int(self.xp.shape[0])

        spacing = float(ic.get("snow_particle_spacing", 0.075))
        self.spacing = spacing
        if "initial_volume" in frame0.attributes:
            iv = np.asarray(frame0.attributes["initial_volume"], dtype=float)[snow]
            self.V0 = np.ascontiguousarray(iv, dtype=float)
        else:
            self.V0 = np.full(self.N, spacing ** 3, dtype=float)

        self.F = np.tile(np.eye(3), (self.N, 1, 1))

        # Kinematic rigid striker (body_id 2).
        bidx = np.where(ball)[0]
        if bidx.size:
            self.ball_pos0 = pos[bidx[0]].copy()
            self.ball_vel = vel[bidx[0]].copy()
            self.ball_mass = float(mass[bidx[0]])
        else:  # pragma: no cover - scene always provides the striker
            self.ball_pos0 = np.asarray(ic.get("striker_start", [-2.85, 0.05, 0.82]), float)
            self.ball_vel = np.asarray(ic.get("striker_velocity", [5.8, 0.0, 0.0]), float)
            self.ball_mass = 18.0
        self.ball_r = float(ic.get("striker_radius", 0.42))

        # Background grid (contains castle, striker path, and debris spread).
        dx = spacing
        self.dx = dx
        self.origin = np.array([-3.6, -2.4, -0.15])
        maxc = np.array([3.4, 2.4, 3.5])
        self.nx = int(np.ceil((maxc[0] - self.origin[0]) / dx))
        self.ny = int(np.ceil((maxc[1] - self.origin[1]) / dx))
        self.nz = int(np.ceil((maxc[2] - self.origin[2]) / dx))
        self.total = self.nx * self.ny * self.nz
        self.grid_max = self.origin + np.array([self.nx, self.ny, self.nz]) * dx

        # Snow material: soft elastic-plastic (fixed corotated).
        E = 1.4e5
        nu = 0.2
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.s_min = 0.60   # plastic compaction limit
        self.s_max = 1.05   # tensile fracture limit

        # Time stepping.
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.nsub = 10
        self.dt = self.sample_dt / self.nsub
        self.t = 0.0

        self.floor_friction = 0.85   # high-friction packed-snow floor
        self.offsets = [(ox, oy, oz) for ox in range(3) for oy in range(3) for oz in range(3)]
        self.steps_done = 0

    # ---------------- one MPM step ----------------
    def _step(self) -> None:
        dt = self.dt
        dx = self.dx
        nx, ny, nz, total = self.nx, self.ny, self.nz, self.total
        xp, vp, mp, V0, F = self.xp, self.vp, self.mp, self.V0, self.F
        N = self.N

        # --- stress from current deformation gradient (fixed corotated) ---
        U, S, Vh = np.linalg.svd(F)
        R = U @ Vh
        J = np.linalg.det(F)
        Sinv = 1.0 / np.maximum(S, 1e-9)
        FinvT = (U * Sinv[:, None, :]) @ Vh
        P = 2.0 * self.mu * (F - R) + (self.lam * (J - 1.0) * J)[:, None, None] * FinvT

        # --- interpolation weights and gradients ---
        s = (xp - self.origin) / dx
        base = np.floor(s + 0.5).astype(np.int64)
        fx = s - base  # in [-0.5, 0.5)
        w_dim = np.empty((N, 3, 3))
        g_dim = np.empty((N, 3, 3))
        w_dim[:, :, 0] = 0.5 * (0.5 - fx) ** 2
        w_dim[:, :, 1] = 0.75 - fx ** 2
        w_dim[:, :, 2] = 0.5 * (0.5 + fx) ** 2
        g_dim[:, :, 0] = (fx - 0.5) / dx
        g_dim[:, :, 1] = -2.0 * fx / dx
        g_dim[:, :, 2] = (0.5 + fx) / dx

        # --- P2G ---
        flats = []
        masses = []
        moms = []
        for ox, oy, oz in self.offsets:
            wx = w_dim[:, 0, ox]
            wy = w_dim[:, 1, oy]
            wz = w_dim[:, 2, oz]
            w = wx * wy * wz
            node = base + np.array([ox - 1, oy - 1, oz - 1], dtype=np.int64)
            ix, iy, iz = node[:, 0], node[:, 1], node[:, 2]
            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
            flat = np.where(valid, ix + nx * (iy + ny * iz), 0)
            grad = np.stack(
                (
                    g_dim[:, 0, ox] * wy * wz,
                    wx * g_dim[:, 1, oy] * wz,
                    wx * wy * g_dim[:, 2, oz],
                ),
                axis=1,
            )
            Pg = np.einsum("nij,nj->ni", P, grad)
            mom = w[:, None] * (mp[:, None] * vp) + dt * (-V0[:, None] * Pg)
            mom = np.where(valid[:, None], mom, 0.0)
            flats.append(flat)
            masses.append(np.where(valid, w * mp, 0.0))
            moms.append(mom)

        flat_all = np.concatenate(flats)
        mass_all = np.concatenate(masses)
        mom_all = np.concatenate(moms, axis=0)

        grid_m = np.bincount(flat_all, weights=mass_all, minlength=total)
        grid_mom = np.empty((total, 3))
        for c in range(3):
            grid_mom[:, c] = np.bincount(flat_all, weights=mom_all[:, c], minlength=total)

        # --- grid update: normalize, gravity, floor and striker collisions ---
        grid_v = np.zeros((total, 3))
        active = np.nonzero(grid_m > 1e-12)[0]
        if active.size:
            grid_v[active] = grid_mom[active] / grid_m[active, None] + dt * self.gravity
            iz = active // (nx * ny)
            rem = active - iz * (nx * ny)
            iy = rem // nx
            ix = rem - iy * nx
            npos = self.origin + np.stack((ix, iy, iz), axis=1) * dx

            # Static floor (top surface at z = 0), high friction.
            fl = npos[:, 2] <= 0.0
            if np.any(fl):
                ai = active[fl]
                np.maximum(grid_v[ai, 2], 0.0, out=grid_v[ai, 2])
                grid_v[ai, 0] *= self.floor_friction
                grid_v[ai, 1] *= self.floor_friction

            # Kinematic rigid striker: impose its velocity inside the sphere.
            c = self.ball_pos0 + self.ball_vel * self.t
            d = npos - c
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            ins = dist < self.ball_r
            if np.any(ins):
                grid_v[active[ins]] = self.ball_vel

        # --- G2P ---
        vp_new = np.zeros((N, 3))
        L = np.zeros((N, 3, 3))
        for ox, oy, oz in self.offsets:
            wx = w_dim[:, 0, ox]
            wy = w_dim[:, 1, oy]
            wz = w_dim[:, 2, oz]
            w = wx * wy * wz
            node = base + np.array([ox - 1, oy - 1, oz - 1], dtype=np.int64)
            ix, iy, iz = node[:, 0], node[:, 1], node[:, 2]
            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
            flat = np.where(valid, ix + nx * (iy + ny * iz), 0)
            gi = grid_v[flat]
            gi = np.where(valid[:, None], gi, 0.0)
            grad = np.stack(
                (
                    g_dim[:, 0, ox] * wy * wz,
                    wx * g_dim[:, 1, oy] * wz,
                    wx * wy * g_dim[:, 2, oz],
                ),
                axis=1,
            )
            vp_new += w[:, None] * gi
            L += np.einsum("na,nb->nab", gi, grad)

        # Numerical safety clamp on particle speed.
        speed = np.sqrt(np.einsum("ij,ij->i", vp_new, vp_new))
        sc = np.where(speed > 30.0, 30.0 / np.maximum(speed, 1e-9), 1.0)
        vp_new *= sc[:, None]

        # --- deformation gradient advection + plastic return mapping ---
        Fnew = (np.eye(3) + dt * L) @ F
        U2, S2, Vh2 = np.linalg.svd(Fnew)
        S2c = np.clip(S2, self.s_min, self.s_max)
        self.F = (U2 * S2c[:, None, :]) @ Vh2

        self.vp = vp_new
        self.xp = xp + dt * vp_new

        # Floor and domain containment.
        np.maximum(self.xp[:, 2], 1e-4, out=self.xp[:, 2])
        lo = self.origin + 0.02
        hi = self.grid_max - 0.02
        np.clip(self.xp, lo, hi, out=self.xp)

        self.t += dt
        self.steps_done += 1

    # ---------------- frame construction ----------------
    def _ball_center(self) -> np.ndarray:
        return self.ball_pos0 + self.ball_vel * self.t

    def _frame(self, time: float) -> StateFrame:
        N = self.N
        c = self._ball_center()

        position = np.vstack((self.xp, c[None, :]))
        velocity = np.vstack((self.vp, self.ball_vel[None, :]))
        mass = np.concatenate((self.mp, [self.ball_mass]))
        body_id = np.concatenate((np.ones(N, dtype=np.int64), [2])).astype(np.int64)

        F_all = np.concatenate((self.F, np.eye(3)[None, :, :]), axis=0)
        vol_all = np.concatenate((self.V0, [self.ball_r ** 3 * 4.0 / 3.0 * np.pi]))

        # External (body) force: gravity on every sample; contact excluded.
        external_force = mass[:, None] * self.gravity[None, :]

        # Signed contact gaps (negative = penetration) to floor and striker.
        floor_gap = self.xp[:, 2]
        d = self.xp - c
        ball_gap = np.sqrt(np.einsum("ij,ij->i", d, d)) - self.ball_r
        csd_snow = np.minimum(floor_gap, ball_gap)
        csd = np.concatenate((csd_snow, [c[2] - self.ball_r]))

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "deformation_gradient": F_all,
                "mass_quantity": mass.copy(),
                "initial_volume": vol_all,
            },
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "initial_total_mass": float(mass.sum()),
            },
        )
        frame.validate()
        return frame

    # ---------------- driver API ----------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        steps = int(round((target - self.t) / self.dt))
        for _ in range(max(steps, 0)):
            self._step()
        self.t = target
        return self._frame(target)

    def diagnostics(self) -> dict:
        return {
            "method": "explicit_MPM_fixed_corotated_snow",
            "particles": int(self.N),
            "substeps_done": int(self.steps_done),
            "dt": float(self.dt),
            "grid": [int(self.nx), int(self.ny), int(self.nz)],
        }
