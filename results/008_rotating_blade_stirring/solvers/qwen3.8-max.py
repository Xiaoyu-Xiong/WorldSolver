"""Physically-based liquid solver for 008_rotating_blade_stirring.

Method: weakly-compressible Position Based Fluids (PBF, Macklin & Mueller 2013)
in full 3D with:
  * gravity, incompressibility projection (density constraints),
  * XSPH viscosity (numerical stand-in for the tiny physical viscosity),
  * kinematic rotating blade handled by signed-distance projection at the
    prescribed pose (momentum enters the liquid through the position update,
    v = (p* - p)/dt), free-slip tank walls and floor,
  * escaped liquid is retained and tracked (no deletion, no mass sink).

The blade kinematics are taken analytically from the frozen scene actuator
(smoothstep ramp to -3 rad/s about +z). All required StateFrame fields are
emitted at every sample time.
"""

from __future__ import annotations

import time as _time

import numpy as np
from worldsolver.contracts import StateFrame

try:  # optional, used only for the diagnostic level-set SDF
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:  # pragma: no cover
    _edt = None


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        frame0 = scene.initial_state()
        self.frame0 = frame0

        fluid = np.asarray(frame0.body_id) == 0
        self.n = int(fluid.sum())
        self.pos = np.array(frame0.position[fluid], dtype=np.float64)
        self.vel = np.zeros((self.n, 3), dtype=np.float64)
        self.mass = np.array(frame0.attributes["mass_quantity"][fluid], dtype=np.float64)
        self.rho0 = float(cfg["liquid"]["density"])
        self.vol = self.mass / self.rho0                     # per-particle volume
        self.mass0 = np.array(frame0.mass, dtype=np.float64)  # full [N] incl. blade marker

        spacing = float(cfg["liquid"]["particle_spacing"])
        self.spacing = spacing
        h = 2.0 * spacing                                    # kernel radius
        self.h = h
        self.h2 = h * h
        self.poly6 = 315.0 / (64.0 * np.pi * h**9)
        self.spiky_grad = -45.0 / (np.pi * h**6)
        dq = 0.2 * h
        self.wdq = self.poly6 * (self.h2 - dq * dq) ** 3
        self.k_scorr = 0.1
        self.eps_lambda = 1.0e3
        self.relax = 0.5
        self.dp_max = 0.3 * h
        self.xsph_c = 0.05
        self.vmax = 10.0
        self.margin = 0.4 * spacing
        self.blade_margin = 0.5 * spacing

        # time integration
        self.t = 0.0
        self.dt = 0.01
        self.iters = 3
        self.steps = 0
        self._perf_t0 = _time.perf_counter()
        self._perf_steps = 0
        self._degraded = False

        # padded hashing grid for neighbor search (allows splashed droplets)
        self.gorigin = np.array([-1.6, -1.6, -0.05])
        self.gdims = np.array([64, 64, 50])
        self.ncell = int(self.gdims[0] * self.gdims[1] * self.gdims[2])
        gx, gy, gz = self.gdims
        offs = []
        for dz in (0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx <= 0:
                        continue
                    if dz == 0 and dy < 0 and not (dy == 0 and dx > 0):
                        continue
                    offs.append(dx + dy * gx + dz * gx * gy)
        # half stencil (13 offsets); self-cell handled separately
        self.half_offsets = np.array(offs, dtype=np.int64)

        # frozen diagnostic grid (config domain)
        dom = cfg["domain"]
        self.h5 = float(dom["grid_spacing"])
        self.gshape = tuple(int(v) for v in dom["reference_grid"])
        self.g5origin = np.asarray(dom["bounds_min"], dtype=float)
        axes = [
            float(lo) + (np.arange(nn) + 0.5) * self.h5
            for lo, nn in zip(dom["bounds_min"], self.gshape, strict=True)
        ]
        self.grid_centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        nx, ny, nz = self.gshape
        self._strides = np.array([ny * nz, nz, 1], dtype=np.int64)

        # frozen tank wall samples (static, free-slip, impermeable)
        self.wall_pos, self.wall_normal = scene.initial_tank_samples()

        self.tank_half = float(cfg["tank"]["interior_half_width"])
        self.tank_height = float(cfg["tank"]["height"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

    # ------------------------------------------------------- neighbor pairs
    def _cell_flat(self, pos):
        idx = np.floor((pos - self.gorigin) / self.h).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.gdims[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.gdims[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.gdims[2] - 1)
        return idx[:, 0] + idx[:, 1] * self.gdims[0] + idx[:, 2] * self.gdims[0] * self.gdims[1]

    def _build_pairs(self, pos):
        flat = self._cell_flat(pos)
        order = np.argsort(flat, kind="stable")
        fs = flat[order]
        uc, starts = np.unique(fs, return_index=True)
        counts = np.diff(np.append(starts, len(fs))).astype(np.int64)
        cnt = np.zeros(self.ncell, dtype=np.int64)
        sta = np.zeros(self.ncell, dtype=np.int64)
        cnt[uc] = counts
        sta[uc] = starts

        Is = []
        Js = []
        nA_all = cnt[uc]
        sA_all = sta[uc]

        def expand(cellA, nA, sA, cellB, nB, sB):
            block = nA * nB
            tot = int(block.sum())
            if tot == 0:
                return None
            bstart = np.cumsum(block) - block
            loc = np.arange(tot) - np.repeat(bstart, block)
            nBr = np.repeat(nB, block)
            oa = loc // nBr
            ob = loc - oa * nBr
            ii = np.repeat(sA, block) + oa
            jj = np.repeat(sB, block) + ob
            return order[ii], order[jj]

        # self-cell pairs (i<j within a cell)
        n_self = int((nA_all * (nA_all - 1) // 2).sum())
        if n_self:
            block = nA_all * nA_all
            tot = int(block.sum())
            bstart = np.cumsum(block) - block
            loc = np.arange(tot) - np.repeat(bstart, block)
            nAr = np.repeat(nA_all, block)
            oa = loc // nAr
            ob = loc - oa * nAr
            keep = oa < ob
            ii = np.repeat(sA_all, block)[keep] + oa[keep]
            jj = np.repeat(sA_all, block)[keep] + ob[keep]
            Is.append(order[ii])
            Js.append(order[jj])

        for off in self.half_offsets:
            cellB = uc + off
            ok = (cellB >= 0) & (cellB < self.ncell)
            cellB = np.where(ok, cellB, 0)
            nB = cnt[cellB]
            valid = ok & (nB > 0)
            if not valid.any():
                continue
            res = expand(
                uc[valid], nA_all[valid], sA_all[valid],
                cellB[valid], nB[valid], sta[cellB[valid]],
            )
            if res is not None:
                Is.append(res[0])
                Js.append(res[1])

        if Is:
            self.pi = np.concatenate(Is).astype(np.int64)
            self.pj = np.concatenate(Js).astype(np.int64)
        else:
            self.pi = np.zeros(0, dtype=np.int64)
            self.pj = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------- physics
    def _blade_project(self, pred, t):
        sd = self.scene.blade_signed_distance(pred, t)
        hit = sd < self.blade_margin
        if not hit.any():
            return
        p = pred[hit]
        d = 1.0e-3
        g = np.empty_like(p)
        for ax in range(3):
            dp = np.zeros(3)
            dp[ax] = d
            sp = self.scene.blade_signed_distance(p + dp, t)
            sm = self.scene.blade_signed_distance(p - dp, t)
            g[:, ax] = (sp - sm) / (2 * d)
        glen = np.linalg.norm(g, axis=1)
        safe = glen > 1e-9
        nvec = np.zeros_like(g)
        nvec[safe] = g[safe] / glen[safe][:, None]
        nvec[~safe, 2] = 1.0
        pred[hit] = p + nvec * (self.blade_margin - sd[hit])[:, None]

    def _wall_project(self, pred):
        m = self.margin
        half = self.tank_half
        wtop = self.tank_height
        low_z = pred[:, 2] < wtop
        x = pred[:, 0]
        y = pred[:, 1]
        pred[:, 0] = np.where(low_z, np.clip(x, -half + m, half - m), x)
        pred[:, 1] = np.where(low_z, np.clip(y, -half + m, half - m), y)
        pred[:, 2] = np.maximum(pred[:, 2], m)  # ground everywhere; retain escaped liquid

    def _step(self, dt):
        t0, t1 = self.t, self.t + dt
        n = self.n
        pos0 = self.pos
        vel = self.vel + self.gravity * dt
        pred = pos0 + vel * dt
        self._blade_project(pred, t1)
        self._wall_project(pred)
        vel = (pred - pos0) / dt

        self._build_pairs(pred)
        pi, pj = self.pi, self.pj
        I, J = pi, pj
        vol = self.vol
        volJ = vol[J]
        volI = vol[I]
        massJ = self.mass[J]
        massI = self.mass[I]

        for _ in range(self.iters):
            rij = pred[I] - pred[J]
            r2 = np.einsum("ij,ij->i", rij, rij)
            inside = r2 < self.h2
            r = np.sqrt(r2)
            diff = self.h2 - r2
            W = np.where(inside, self.poly6 * diff * diff * diff, 0.0)
            rho = np.bincount(I, weights=massJ * W, minlength=n)
            rho += np.bincount(J, weights=massI * W, minlength=n)
            rho += self.mass * (self.poly6 * self.h2**3)
            C = np.maximum(rho / self.rho0 - 1.0, 0.0)

            # spiky gradient: grad_i W = spiky_grad*(h-r)^2 * rij/r
            coef = np.where(inside & (r > 1e-9), self.spiky_grad * (self.h - r) ** 2 / np.maximum(r, 1e-9), 0.0)
            gw = coef[:, None] * rij  # grad w.r.t. p_i

            # denominator: |sum_j vol_j gw|^2 + sum_j (vol_j |gw|)^2
            gx = np.bincount(I, weights=volJ * gw[:, 0], minlength=n) + np.bincount(
                J, weights=-volI * gw[:, 0], minlength=n)
            gy = np.bincount(I, weights=volJ * gw[:, 1], minlength=n) + np.bincount(
                J, weights=-volI * gw[:, 1], minlength=n)
            gz = np.bincount(I, weights=volJ * gw[:, 2], minlength=n) + np.bincount(
                J, weights=-volI * gw[:, 2], minlength=n)
            s = np.bincount(I, weights=volJ**2 * (gw**2).sum(axis=1), minlength=n) + np.bincount(
                J, weights=volI**2 * (gw**2).sum(axis=1), minlength=n)
            denom = gx * gx + gy * gy + gz * gz + s + self.eps_lambda
            lam = -C / denom

            # One-sided constraint (C >= 0): no tensile lambda, hence no
            # artificial-pressure (scorr) term; eps + clamp regularize contacts.
            cfac = lam[I] + lam[J]
            ax_ = cfac * volJ * gw[:, 0]
            ay_ = cfac * volJ * gw[:, 1]
            az_ = cfac * volJ * gw[:, 2]
            bx_ = cfac * volI * gw[:, 0]
            by_ = cfac * volI * gw[:, 1]
            bz_ = cfac * volI * gw[:, 2]
            dpx = np.bincount(I, weights=ax_, minlength=n) - np.bincount(J, weights=bx_, minlength=n)
            dpy = np.bincount(I, weights=ay_, minlength=n) - np.bincount(J, weights=by_, minlength=n)
            dpz = np.bincount(I, weights=az_, minlength=n) - np.bincount(J, weights=bz_, minlength=n)
            dp = np.stack((dpx, dpy, dpz), axis=1) * self.relax
            dplen = np.linalg.norm(dp, axis=1)
            big = dplen > self.dp_max
            if big.any():
                dp[big] *= self.dp_max / dplen[big]
            pred += dp
            self._blade_project(pred, t1)
            self._wall_project(pred)

        vel = (pred - pos0) / dt

        # XSPH viscosity (reuse last pair list)
        rij = pred[I] - pred[J]
        r2 = np.einsum("ij,ij->i", rij, rij)
        inside = r2 < self.h2
        diff = self.h2 - r2
        W = np.where(inside, self.poly6 * diff * diff * diff, 0.0)
        dv = vel[J] - vel[I]
        wI = volJ * W
        wJ = volI * W
        dvx = np.bincount(I, weights=wI * dv[:, 0], minlength=n) + np.bincount(J, weights=-wJ * dv[:, 0], minlength=n)
        dvy = np.bincount(I, weights=wI * dv[:, 1], minlength=n) + np.bincount(J, weights=-wJ * dv[:, 1], minlength=n)
        dvz = np.bincount(I, weights=wI * dv[:, 2], minlength=n) + np.bincount(J, weights=-wJ * dv[:, 2], minlength=n)
        vel += self.xsph_c * np.stack((dvx, dvy, dvz), axis=1)

        sp = np.linalg.norm(vel, axis=1)
        fast = sp > self.vmax
        if fast.any():
            vel[fast] *= self.vmax / sp[fast]

        # numerical blow-up guard
        if not np.isfinite(pred).all():
            bad = ~np.isfinite(pred).all(axis=1)
            pred[bad] = pos0[bad]
            vel[bad] = 0.0

        self.pos = pred
        self.vel = vel
        self.t = t1
        self.steps += 1

        # one-shot performance guard: keep the run inside the platform budget
        self._perf_steps += 1
        if not self._degraded and self._perf_steps == 6:
            elapsed = _time.perf_counter() - self._perf_t0
            per_step = elapsed / self._perf_steps
            remaining_steps = (float(self.cfg["duration"]) - self.t) / self.dt
            projected = per_step * remaining_steps
            if projected > 1200.0:
                self.dt = min(0.02, self.dt * 2)
                self.iters = 2
                self._degraded = True

    # ------------------------------------------------------ grid utilities
    def _splat_weights(self, pts):
        """Trilinear corner indices/weights of points on the diagnostic grid."""
        shape = np.asarray(self.gshape, dtype=float)
        c = (pts - self.g5origin) / self.h5 - 0.5
        c = np.clip(c, 0.0, shape - 1.0000001)
        c0 = np.floor(c).astype(np.int64)
        f = c - c0
        base = c0 @ self._strides
        corners = []
        for di in (0, 1):
            wi = f[:, 0] if di else 1.0 - f[:, 0]
            for dj in (0, 1):
                wj = f[:, 1] if dj else 1.0 - f[:, 1]
                for dk in (0, 1):
                    wk = f[:, 2] if dk else 1.0 - f[:, 2]
                    idx = base + di * self._strides[0] + dj * self._strides[1] + dk * self._strides[2]
                    corners.append((idx, wi * wj * wk))
        return corners

    def _grid_fields(self, t):
        ncell = int(np.prod(self.gshape))
        corners = self._splat_weights(self.pos)
        wsum = np.zeros(ncell)
        cnt = np.zeros(ncell)
        vx = np.zeros(ncell)
        vy = np.zeros(ncell)
        vz = np.zeros(ncell)
        for idx, w in corners:
            wsum += np.bincount(idx, weights=w, minlength=ncell)
            cnt += np.bincount(idx, weights=w, minlength=ncell)
            vx += np.bincount(idx, weights=w * self.vel[:, 0], minlength=ncell)
            vy += np.bincount(idx, weights=w * self.vel[:, 1], minlength=ncell)
            vz += np.bincount(idx, weights=w * self.vel[:, 2], minlength=ncell)
        filled = wsum > 1.0e-3
        velocity = np.zeros((*self.gshape, 3))
        vv = np.stack((vx, vy, vz), axis=-1)
        vv[filled] /= wsum[filled, None]
        velocity = vv.reshape(*self.gshape, 3)

        count_grid = cnt.reshape(self.gshape)
        occupied = count_grid > 1.5

        # signed level set (positive in air, negative in liquid), meters
        if _edt is not None:
            phi = (_edt(~occupied) - _edt(occupied)) * self.h5
        else:
            zz = self.grid_centers[..., 2]
            phi = np.where(occupied, -0.5 * self.h5, 0.5 * self.h5) + 0.0 * zz

        # active mask: liquid interior cells away from solids/free surface
        act = occupied.copy()
        for ax in range(3):
            act &= np.roll(occupied, 1, axis=ax)
            act &= np.roll(occupied, -1, axis=ax)
        act[:1] = act[-1:] = False
        act[:, :1] = act[:, -1:] = False
        act[:, :, :1] = act[:, :, -1:] = False
        sd = self.scene.blade_signed_distance(self.grid_centers, t).reshape(self.gshape)
        act &= sd > self.h5

        density = np.where(occupied, self.rho0 * np.minimum(wsum.reshape(self.gshape) / 8.0 * 8.0, 1.0), 0.0)
        return velocity, act.astype(float), phi, density, wsum.reshape(self.gshape), count_grid

    def _interp_grid(self, velocity, wsum_grid, pts):
        corners = self._splat_weights(pts)
        n = len(pts)
        wsum = np.zeros(n)
        vals = np.zeros((n, 3))
        vf = velocity.reshape(-1, 3)
        wg = wsum_grid.reshape(-1)
        for idx, w in corners:
            wsum += w * wg[idx]
            vals += w * vf[idx]
        good = wsum > 1.0e-4
        out = np.zeros((n, 3))
        out[good] = vals[good] / wsum[good, None]
        return out, wsum

    # ---------------------------------------------------------------- frame
    def _make_frame(self, t):
        velocity, active, phi, density, wsum_grid, count_grid = self._grid_fields(t)
        pose = self.scene.blade_kinematics(t)

        wall_vel, _ = self._interp_grid(velocity, wsum_grid, self.wall_pos)

        bs = self.scene.blade_surface_samples(t)
        bpos = bs["position"]
        binterp, bwet_w = self._interp_grid(velocity, wsum_grid, bpos)
        wetted = (bwet_w > 0.4) & (bpos[:, 2] > 0.0)
        bpos_w = bpos[wetted]
        bnorm_w = bs["normal"][wetted]
        btgt_w = bs["target_velocity"][wetted]
        bvel_w = binterp[wetted]

        position = np.vstack((self.pos, pose["position"][None, :]))
        vel_out = np.vstack((self.vel, np.zeros((1, 3))))
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n + 1, 1))
        orientation[-1] = pose["orientation"]
        ang = np.zeros((self.n + 1, 3))
        ang[-1] = pose["angular_velocity"]
        body_id = np.zeros(self.n + 1, dtype=np.int64)
        body_id[-1] = int(self.cfg["blade"]["body_id"])

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=vel_out,
            orientation=orientation,
            angular_velocity=ang,
            mass=self.mass0.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": np.asarray(self.frame0.attributes["mass_quantity"]).copy(),
                "particle_volume": np.asarray(self.frame0.attributes["particle_volume"]).copy(),
                "particle_radius": np.asarray(self.frame0.attributes["particle_radius"]).copy(),
            },
            fields={
                "velocity_field": velocity,
                "active_mask": active,
                "liquid_level_set": phi,
                "density_field": density,
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": self.wall_pos,
                "boundary_normal": self.wall_normal,
                "boundary_velocity": wall_vel,
                "boundary_target_velocity": np.zeros_like(self.wall_pos),
                "blade_boundary_position": bpos_w,
                "blade_boundary_normal": bnorm_w,
                "blade_boundary_velocity": bvel_w,
                "blade_boundary_target_velocity": btgt_w,
                "grid_origin": self.g5origin.copy(),
            },
            scalars={
                "grid_spacing": self.h5,
                "characteristic_length": float(self.frame0.scalars["characteristic_length"]),
                "reference_velocity": float(self.frame0.scalars["reference_velocity"]),
                "initial_liquid_volume": float(self.frame0.scalars["initial_liquid_volume"]),
            },
        )
        return frame

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.t - 1e-12:
            raise ValueError("advance_to requires non-decreasing time")
        while self.t < time - 1e-9:
            dt = min(self.dt, time - self.t)
            self._step(dt)
        frame = self._make_frame(self.t)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = float(np.linalg.norm(self.vel, axis=1).max()) if self.n else 0.0
        return {
            "method": "position_based_fluids_3d",
            "n_liquid_particles": self.n,
            "steps": self.steps,
            "dt": self.dt,
            "solver_iterations": self.iters,
            "max_speed": speed,
            "degraded_performance_mode": self._degraded,
            "total_liquid_mass": float(self.mass.sum()),
        }
