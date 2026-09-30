"""Position-Based Fluids solver: water poured into a clear cylindrical glass.

Physically-based particle fluid (Macklin & Mueller 2013, PBF) with:
  - explicit gravitational advection,
  - iterative density-constraint projection (incompressibility),
  - XSPH viscosity,
  - solid boundary projection against the frozen glass profile and floor,
  - a sustained prescribed inflow jet matching the frozen flow rate.

All time stepping is a real numerical integration; no keyframes or scripted
motion are used.  Deterministic: no randomness is involved.
"""
from __future__ import annotations

import numpy as np

try:  # optional accelerator for the diagnostic level set / boundary sampling
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------
# frozen glass inner radius profile (meters): [(radius, z), ...]
_GLASS_PROFILE = ((0.225, 0.0), (0.25, 0.03), (0.25, 0.75), (0.255, 0.78))
_GLASS_THICKNESS = 0.012
_RIM_Z = 0.78


def _glass_inner_radius(z):
    """Piecewise-linear inner radius of the glass at height z."""
    z = np.asarray(z, dtype=np.float64)
    r = np.full(z.shape, _GLASS_PROFILE[-1][0])
    for (r0, z0), (r1, z1) in zip(_GLASS_PROFILE[:-1], _GLASS_PROFILE[1:]):
        m = (z >= z0) & (z <= z1)
        t = np.clip((z[m] - z0) / max(z1 - z0, 1e-12), 0.0, 1.0)
        r[m] = r0 + t * (r1 - r0)
    r[z < _GLASS_PROFILE[0][1]] = _GLASS_PROFILE[0][0]
    return r


class BenchmarkSolver:
    """PBF fluid solver for the water-pour-into-glass scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        meta = scene.metadata() if cfg is None else {}
        ic = cfg["initial_conditions"] if cfg is not None else {}

        self.gravity = np.asarray(
            cfg["gravity"] if cfg is not None else meta.get("gravity", [0.0, 0.0, -9.81]),
            dtype=np.float64,
        )
        self.rho0 = float(ic.get("liquid_density", meta.get("liquid_density", 998.0)))
        self.dp = float(ic.get("particle_spacing", 0.018))
        self.glass_radius = float(ic.get("glass_inner_radius", 0.25))
        self.glass_height = float(ic.get("glass_inner_height", 0.78))
        self.jet_radius = float(ic.get("jet_radius", 0.042))
        self.jet_speed = float(ic.get("jet_speed", 2.25))
        self.jet_start = np.asarray(ic.get("jet_start", [-0.14, 0.0, 0.98]), dtype=np.float64)
        self.jet_end = np.asarray(ic.get("jet_end", [-0.035, 0.0, 0.58]), dtype=np.float64)
        self.source_t0 = float(ic.get("source_start_time", 0.0))
        self.source_t1 = float(ic.get("source_end_time", 3.25))
        self.target_rate = float(ic.get("target_inflow_rate_m3_s", 0.01247))

        # ---- numerical parameters -----------------------------------------
        self.h = 2.0 * self.dp               # SPH smoothing radius (0.036 m)
        self.h2 = self.h * self.h
        self.dt = 0.0025                     # substep (8 per 0.02 s sample)
        self.iterations = 2                  # density constraint iterations
        self.m = self.rho0 * self.dp ** 3    # particle mass
        self.eps_cfm = 100.0                 # constraint-force mixing epsilon
        self.xsph_c = 0.05
        self.v_max = 6.0                     # safety velocity clamp
        self.dp_max = 0.2 * self.h           # per-iteration position clamp
        self.wall_margin = 0.35 * self.dp
        self.search_h = 1.05 * self.h

        # kernels
        self.poly6_c = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky_c = -45.0 / (np.pi * self.h ** 6)
        self.w0 = self.poly6_c * self.h2 ** 3
        self.scorr_k = 2.0e-4
        dq = 0.25 * self.h
        self.scorr_wq = self.poly6_c * (self.h2 - dq * dq) ** 3

        # ---- emission ------------------------------------------------------
        jet_dir = self.jet_end - self.jet_start
        self.jet_len = float(np.linalg.norm(jet_dir))
        self.jet_dir = jet_dir / self.jet_len
        side = np.array([0.0, 1.0, 0.0])
        cross = np.cross(self.jet_dir, side)
        cross /= np.linalg.norm(cross)
        # Choose the in-plane lattice spacing whose emitted-layer count best
        # reproduces the frozen prescribed volumetric inflow rate.  One layer
        # is emitted every ``emit_every`` substeps; the axial spacing follows
        # from the jet speed, so only the layer population is free.
        self.emit_every = 3
        self.emit_dt = self.emit_every * self.dt
        r_disk = self.jet_radius - 0.2 * self.dp
        best = None
        for s in np.arange(0.019, 0.0115, -0.00025):
            k = int(np.floor(2.0 * self.jet_radius / s)) + 1
            coords = (np.arange(k) - 0.5 * (k - 1)) * s
            disk = [(a, b) for a in coords for b in coords
                    if a * a + b * b <= r_disk * r_disk]
            if not disk:
                continue
            rate = len(disk) * self.dp ** 3 / self.emit_dt
            err = abs(rate - self.target_rate)
            if best is None or err < best[0]:
                best = (err, np.asarray(disk, dtype=np.float64), rate)
        self.emit_offsets, self.emit_rate = best[1], best[2]
        self.emit_basis = np.vstack([side, cross])

        # ---- neighbor-search grid -----------------------------------------
        dom = cfg["domain"] if cfg is not None else {}
        bmin = np.asarray(dom.get("bounds_min", [-0.33, -0.33, 0.0]), dtype=np.float64)
        bmax = np.asarray(dom.get("bounds_max", [0.33, 0.33, 1.1]), dtype=np.float64)
        self.dom_lo = bmin - np.array([0.02, 0.02, 0.06])
        self.dom_hi = bmax + np.array([0.02, 0.02, 0.06])
        self.cell_size = self.search_h
        gdim = np.maximum(np.ceil((self.dom_hi - self.dom_lo) / self.cell_size).astype(int) + 1, 2)
        self.gdim = tuple(int(v) for v in gdim)
        self.ncells = int(self.gdim[0] * self.gdim[1] * self.gdim[2])
        self._offsets = [
            (ox, oy, oz)
            for ox in (-1, 0, 1) for oy in (-1, 0, 1) for oz in (-1, 0, 1)
        ]

        # ---- diagnostic grid ----------------------------------------------
        self.grid_d = 0.022
        self.grid_lo = bmin
        gn = np.maximum(np.ceil((bmax - bmin) / self.grid_d).astype(int), 2)
        self.gn = (int(gn[0]), int(gn[1]), int(gn[2]))
        self.grid_shape = self.gn
        self.cell_vol = self.grid_d ** 3
        self.char_length = 2.0 * self.glass_radius
        self.ref_velocity = self.jet_speed

        # ---- boundary sample points ---------------------------------------
        pts, nrm = self._build_boundary_samples()
        self.bnd_pos = pts
        self.bnd_normal = nrm
        self.bnd_target_v = np.zeros_like(pts)

        # ---- state ----------------------------------------------------------
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        src = np.asarray(frame0.attributes.get(
            "source_region", np.zeros(len(self.pos))), dtype=np.float64)
        self.src_tag = src
        self.t = 0.0
        self.step_count = 0
        self.emitted_count = 0
        self.initial_count = len(self.pos)
        self.emitted_mass = 0.0
        self._stats = {"min_dt": self.dt, "max_vel": 0.0, "max_density_rel": 1.0}

    # ------------------------------------------------------------------ setup
    def _build_boundary_samples(self):
        pts, nrm = [], []
        az = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        ca, sa = np.cos(az), np.sin(az)
        for z in (0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.77):
            r = float(_glass_inner_radius(np.array([z]))[0])
            for i in range(len(az)):
                pts.append((r * ca[i], r * sa[i], z))
                nrm.append((-ca[i], -sa[i], 0.0))  # into the fluid region
        for r in (0.05, 0.10, 0.15, 0.20):
            for i in range(len(az)):
                pts.append((r * ca[i], r * sa[i], 0.0))
                nrm.append((0.0, 0.0, 1.0))
        # exterior of the glass wall + surrounding table floor
        az2 = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        for z in (0.05, 0.20, 0.40, 0.60, 0.75):
            r = float(_glass_inner_radius(np.array([z]))[0]) + _GLASS_THICKNESS
            for i in range(len(az2)):
                pts.append((r * np.cos(az2[i]), r * np.sin(az2[i]), z))
                nrm.append((np.cos(az2[i]), np.sin(az2[i]), 0.0))
        r = 0.30
        for i in range(len(az2)):
            pts.append((r * np.cos(az2[i]), r * np.sin(az2[i]), 0.0))
            nrm.append((0.0, 0.0, 1.0))
        return (np.asarray(pts, dtype=np.float64),
                np.asarray(nrm, dtype=np.float64))

    # ---------------------------------------------------------- collisions
    def _collide(self, pos):
        """Project positions out of the glass solid and above the floors."""
        # world floor / domain clamp
        np.clip(pos[:, 2], self.wall_margin, self.dom_hi[2] - 0.01, out=pos[:, 2])
        np.clip(pos[:, 0], self.dom_lo[0] + 0.01, self.dom_hi[0] - 0.01, out=pos[:, 0])
        np.clip(pos[:, 1], self.dom_lo[1] + 0.01, self.dom_hi[1] - 0.01, out=pos[:, 1])

        below = pos[:, 2] < _RIM_Z
        if np.any(below):
            idx = np.nonzero(below)[0]
            r = np.hypot(pos[idx, 0], pos[idx, 1])
            rin = _glass_inner_radius(pos[idx, 2])
            rout = rin + _GLASS_THICKNESS
            mid = 0.5 * (rin + rout)
            inside = r < mid
            # inside the glass: keep below inner wall
            k = np.nonzero(inside & (r > rin - self.wall_margin))[0]
            if len(k):
                s = (rin[k] - self.wall_margin) / r[k]
                pos[idx[k], 0] *= s
                pos[idx[k], 1] *= s
            # outside the glass: keep beyond outer wall
            k = np.nonzero(~inside & (r < rout + self.wall_margin) & (r > 1e-9))[0]
            if len(k):
                s = (rout[k] + self.wall_margin) / r[k]
                pos[idx[k], 0] *= s
                pos[idx[k], 1] *= s
        return pos

    # ------------------------------------------------------- neighbor pairs
    def _pairs(self, pos):
        n = len(pos)
        cs = self.cell_size
        g = self.gdim
        cell = np.floor((pos - self.dom_lo) / cs)
        np.clip(cell, 0, np.asarray(g) - 1, out=cell)
        cell = cell.astype(np.int64)
        lin = cell[:, 0] + g[0] * (cell[:, 1] + g[1] * cell[:, 2])
        order = np.argsort(lin, kind="stable")
        counts = np.bincount(lin[order], minlength=self.ncells)
        starts = np.zeros(self.ncells + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        counts_p = np.concatenate(([0], counts))
        starts_p = np.concatenate(([0], starts[:-1]))

        hs2 = self.search_h * self.search_h
        idx_i_list, idx_j_list = [], []
        particles = np.arange(n)
        cs_range = np.arange(max(3 * n, 1024))
        for ox, oy, oz in self._offsets:
            tx = cell[:, 0] + ox
            ty = cell[:, 1] + oy
            tz = cell[:, 2] + oz
            valid = ((tx >= 0) & (tx < g[0]) & (ty >= 0) & (ty < g[1])
                     & (tz >= 0) & (tz < g[2]))
            tgt = np.where(valid, tx + g[0] * (ty + g[1] * tz) + 1, 0)
            occ = counts_p[tgt]
            total = int(occ.sum())
            if total == 0:
                continue
            rep = np.repeat(particles, occ)
            base = np.repeat(starts_p[tgt], occ)
            seg = np.repeat(np.cumsum(occ) - occ, occ)
            if cs_range.size < total:
                cs_range = np.arange(total)
            offs = cs_range[:total] - seg
            j_sorted = order[base + offs]
            dx = pos[rep, 0] - pos[j_sorted, 0]
            dy = pos[rep, 1] - pos[j_sorted, 1]
            dz = pos[rep, 2] - pos[j_sorted, 2]
            r2 = dx * dx + dy * dy + dz * dz
            keep = (r2 < hs2) & (r2 > 1e-14)
            if np.any(keep):
                idx_i_list.append(rep[keep])
                idx_j_list.append(j_sorted[keep])
        if not idx_i_list:
            e = np.empty(0, dtype=np.int64)
            return e, e
        return np.concatenate(idx_i_list), np.concatenate(idx_j_list)

    # ------------------------------------------------------------ stepping
    def _emit(self):
        base = self.jet_start[None, :]
        offs = self.emit_offsets @ self.emit_basis  # [K,3] in-plane offsets
        new_pos = base + offs
        new_pos = new_pos + 0.5 * self.dp * self.jet_dir[None, :]
        new_vel = np.repeat(self.jet_speed * self.jet_dir[None, :], len(new_pos), axis=0)
        self.pos = np.vstack((self.pos, new_pos))
        self.vel = np.vstack((self.vel, new_vel))
        self.src_tag = np.concatenate((self.src_tag, np.ones(len(new_pos))))
        self.emitted_count += len(new_pos)
        self.emitted_mass += len(new_pos) * self.m

    def _substep(self, dt):
        n = len(self.pos)
        # external force (gravity) + predict
        self.vel += self.gravity[None, :] * dt
        speed = np.linalg.norm(self.vel, axis=1)
        fast = speed > self.v_max
        if np.any(fast):
            self.vel[fast] *= (self.v_max / speed[fast])[:, None]
        pos_old = self.pos.copy()
        pred = self.pos + self.vel * dt
        pred = self._collide(pred)

        lam = np.zeros(n)
        pair_i = pair_j = None
        for _ in range(self.iterations):
            pair_i, pair_j = self._pairs(pred)
            if len(pair_i) == 0:
                break
            d = pred[pair_i] - pred[pair_j]
            r2 = np.einsum("ij,ij->i", d, d)
            r = np.sqrt(r2)
            w = self.poly6_c * (self.h2 - r2) ** 3
            # density constraint (self contribution W(0) added before mass)
            rho = (np.bincount(pair_i, weights=w, minlength=n) + self.w0) * self.m
            c = rho / self.rho0 - 1.0
            self._stats["max_density_rel"] = max(
                self._stats["max_density_rel"], float(rho.max() / self.rho0))
            # spiky gradient  g = dW/dr * (pi-pj)/r
            s = self.spiky_c * (self.h - r) ** 2 / r
            gx = s * d[:, 0]
            gy = s * d[:, 1]
            gz = s * d[:, 2]
            gc = self.m / self.rho0
            sgx = np.bincount(pair_i, weights=gx, minlength=n)
            sgy = np.bincount(pair_i, weights=gy, minlength=n)
            sgz = np.bincount(pair_i, weights=gz, minlength=n)
            gnorm2 = s * s * r2
            denom = (gc * gc) * (
                np.bincount(pair_i, weights=gnorm2, minlength=n)
                + sgx * sgx + sgy * sgy + sgz * sgz)
            lam = -c / (denom + self.eps_cfm)
            # position correction
            scorr = -self.scorr_k * (w / self.scorr_wq) ** 4
            coef = (lam[pair_i] + lam[pair_j] + scorr) * s
            dpx = np.bincount(pair_i, weights=coef * d[:, 0], minlength=n) / self.rho0
            dpy = np.bincount(pair_i, weights=coef * d[:, 1], minlength=n) / self.rho0
            dpz = np.bincount(pair_i, weights=coef * d[:, 2], minlength=n) / self.rho0
            mag = np.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
            over = mag > self.dp_max
            if np.any(over):
                scale = np.where(over, self.dp_max / np.maximum(mag, 1e-12), 1.0)
                dpx *= scale
                dpy *= scale
                dpz *= scale
            pred = pred + np.column_stack((dpx, dpy, dpz))
            pred = self._collide(pred)

        # velocity update
        self.vel = (pred - pos_old) / dt
        # XSPH viscosity on the last neighbor list
        if pair_i is not None and len(pair_i) > 0:
            d = pred[pair_i] - pred[pair_j]
            r2 = np.einsum("ij,ij->i", d, d)
            keep = r2 < self.h2
            if np.any(keep):
                pi, pj = pair_i[keep], pair_j[keep]
                r2k = r2[keep]
                w = self.poly6_c * (self.h2 - r2k) ** 3
                dvx = np.bincount(pi, weights=(self.vel[pj, 0] - self.vel[pi, 0]) * w,
                                  minlength=n)
                dvy = np.bincount(pi, weights=(self.vel[pj, 1] - self.vel[pi, 1]) * w,
                                  minlength=n)
                dvz = np.bincount(pi, weights=(self.vel[pj, 2] - self.vel[pi, 2]) * w,
                                  minlength=n)
                k = self.xsph_c * self.m / self.rho0
                self.vel[:, 0] += k * dvx
                self.vel[:, 1] += k * dvy
                self.vel[:, 2] += k * dvz
        self.pos = pred

        vmax = float(np.linalg.norm(self.vel, axis=1).max()) if n else 0.0
        self._stats["max_vel"] = max(self._stats["max_vel"], vmax)

    # ------------------------------------------------------------- export
    def _export_frame(self, t):
        n = len(self.pos)
        pos, vel = self.pos, self.vel
        d = self.grid_d
        gn = self.gn
        ncells = gn[0] * gn[1] * gn[2]

        idx = np.floor((pos - self.grid_lo) / d)
        inb = np.ones(n, dtype=bool)
        for a in range(3):
            inb &= (idx[:, a] >= 0) & (idx[:, a] < gn[a])
        idx = np.clip(idx, 0, np.asarray(gn) - 1).astype(np.int64)
        lin = idx[:, 0] + gn[0] * (idx[:, 1] + gn[1] * idx[:, 2])
        lin = np.where(inb, lin, -1)
        counts = np.bincount(lin[lin >= 0], minlength=ncells).astype(np.float64)
        frac = np.minimum(counts * self.dp ** 3 / self.cell_vol, 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            vel_grid = np.zeros((ncells, 3), dtype=np.float64)
            for a in range(3):
                s = np.bincount(lin[lin >= 0], weights=vel[inb, a], minlength=ncells)
                vel_grid[:, a] = np.where(counts > 0, s / np.maximum(counts, 1.0), 0.0)
        vel_grid = vel_grid.reshape(gn + (3,))

        inside = frac >= 0.5
        if _HAVE_SCIPY:
            dist_out = distance_transform_edt(~inside, sampling=(d, d, d))
            dist_in = distance_transform_edt(inside, sampling=(d, d, d))
            phi = dist_out - dist_in
        else:  # fallback: scaled indicator
            phi = (0.5 - frac) * 3.0 * d
        level_set = np.clip(phi, -3.0 * d, 3.0 * d).astype(np.float64)
        level_set = level_set.reshape(gn)

        active = (frac > 0.0).astype(np.float64).reshape(gn)

        # measured fluid-side wall velocity
        bvel = np.zeros_like(self.bnd_pos)
        if n and _HAVE_SCIPY:
            tree = cKDTree(pos)
            dist, ii = tree.query(self.bnd_pos, k=8, distance_upper_bound=0.03)
            for b in range(len(self.bnd_pos)):
                ok = np.isfinite(dist[b])
                if np.any(ok):
                    bvel[b] = vel[ii[b][ok]].mean(axis=0)
        elif n:
            for b in range(len(self.bnd_pos)):
                dd = np.linalg.norm(pos - self.bnd_pos[b], axis=1)
                k = np.argsort(dd)[:8]
                k = k[dd[k] < 0.03]
                if len(k):
                    bvel[b] = vel[k].mean(axis=0)

        from worldsolver.contracts import StateFrame

        return StateFrame(
            time=float(t),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=np.full(n, self.m, dtype=np.float64),
            attributes={
                "mass_quantity": np.full(n, self.m, dtype=np.float64),
                "source_region": self.src_tag.copy(),
            },
            fields={
                "active_mask": active,
                "boundary_normal": self.bnd_normal.copy(),
                "boundary_target_velocity": self.bnd_target_v.copy(),
                "boundary_velocity": bvel,
                "cumulative_mass_source": np.array([self.emitted_mass],
                                                   dtype=np.float64),
                "liquid_interior_mask": inside.astype(np.float64).reshape(gn),
                "liquid_level_set": level_set,
                "velocity_field": vel_grid.reshape(gn + (3,)),
            },
            scalars={
                "characteristic_length": float(self.char_length),
                "grid_spacing": float(self.grid_d),
                "reference_velocity": float(self.ref_velocity),
                "rest_density": float(self.rho0),
                "particle_spacing": float(self.dp),
                "time": float(t),
            },
        )

    # --------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9:
            dt = min(self.dt, target - self.t)
            full = abs(dt - self.dt) < 1e-12
            self._substep(dt)
            self.t += dt
            self.step_count += 1
            if (full and self.source_t0 - 1e-9 <= self.t <= self.source_t1 + 1e-9
                    and self.step_count % self.emit_every == 0):
                self._emit()
            guard += 1
            if guard > 100000:
                raise RuntimeError("advance_to did not converge")
        return self._export_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "particles": int(len(self.pos)),
            "initial_particles": int(self.initial_count),
            "emitted_particles": int(self.emitted_count),
            "emitted_mass_kg": float(self.emitted_mass),
            "emission_rate_m3s": float(self.emit_rate),
            "target_rate_m3s": float(self.target_rate),
            "max_velocity": float(self._stats["max_vel"]),
            "max_density_ratio": float(self._stats["max_density_rel"]),
            "substeps": int(self.step_count),
        }
