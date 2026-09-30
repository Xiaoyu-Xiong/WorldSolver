"""Fountain solver: 3D free-surface particle fluid (Position-Based Fluids).

Physics
-------
- Incompressible liquid (rho0 = 1000 kg/m^3) represented by Lagrangian
  particles initialized from the frozen scene state (calm circular pool plus
  an already-rising central jet).
- Sustained upward inflow through the nozzle exit disk (frozen source:
  center (0,0,0.4), radius 0.18 m, speed 5.2 m/s) injects new particles;
  particles that overflow the basin rim and leave the domain are removed.
  Both contributions are tracked in ``fields.cumulative_mass_source``.
- Time integration: explicit gravity + PBF density-projection
  (Macklin & Mueller 2013) with a poly6 kernel and spiky gradients,
  plus mild XSPH viscosity.  Deterministic (seeded RNG only for the
  tiny emission jitter that produces leading-edge noise).
- Boundaries: impermeable vertical basin wall (inner radius 1.5 m up to
  the rim at z = 0.52), basin floor at z = 0.04, and the static nozzle
  cylinder (radius 0.24 m, top z = 0.34).  Walls are enforced by
  position projection with normal-velocity cancellation.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBF particle solver for the 002_fountain benchmark."""

    def __init__(self) -> None:
        self.initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        ic = cfg.get("initial_conditions", {})
        meta = scene.metadata()

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        gravity = np.asarray(meta.get("gravity", cfg.get("gravity", [0.0, 0.0, -9.81])), dtype=float)
        self.gravity = gravity

        frame = scene.initial_state()
        x = np.array(frame.position, dtype=np.float64)
        v = np.array(frame.velocity, dtype=np.float64)
        mass = np.array(frame.mass, dtype=np.float64)
        phase = np.array(frame.attributes.get("source_region", np.zeros(len(x))), dtype=np.float64)

        self.rho0 = float(meta.get("liquid_density", ic.get("liquid_density", 1000.0)))
        self.dx = float(frame.scalars.get("particle_spacing", ic.get("particle_spacing", 0.085)))
        self.h = 2.0 * self.dx
        self.h2 = self.h * self.h
        self.m_p = self.rho0 * self.dx ** 3

        # frozen geometry (scene_geometry.yaml)
        self.wall_r = 1.5           # basin inner radius
        self.wall_top = 0.52        # basin rim height
        self.floor_z = 0.04         # inner floor
        self.nozzle_r = 0.24
        self.nozzle_top = 0.34

        # frozen source
        src = meta.get("source", {})
        self.src_z = float(ic.get("nozzle_exit_z", 0.4))
        self.src_r = float(ic.get("nozzle_radius", src.get("radius", 0.18)))
        self.src_speed = float(ic.get("jet_speed", src.get("speed", 5.2)))
        # volumetric inflow -> particles per second
        vol_rate = np.pi * self.src_r ** 2 * self.src_speed
        self.emit_rate = self.rho0 * vol_rate / self.m_p
        self.emit_acc = 0.0
        self._emit_counter = 0
        # deterministic lattice sites inside the exit disk
        sites = []
        n = 3
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                p = np.array([i * self.dx, j * self.dx])
                if np.dot(p, p) <= (self.src_r - 0.35 * self.dx) ** 2:
                    sites.append(p)
        self._emit_sites = sites if sites else [np.zeros(2)]

        # time stepping
        self.dt = 0.005
        self.iterations = 3
        self.x = x
        self.v = v
        self.mass = mass
        self.phase = phase
        self.t = 0.0
        self.mass_added = 0.0
        self.mass_removed = 0.0

        # kernel coefficients
        self.k_poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.k_spiky = -45.0 / (np.pi * self.h ** 6)
        self.w0 = self.k_poly6 * self.h2 ** 3
        self.eps_lambda = 1e-2
        self.xsph_c = 0.05

        # diagnostic grid
        self.gdx = 0.15
        self.g_lo = np.array([-1.6, -1.6, 0.04])
        self.g_n = np.array([22, 22, 22])
        gx = self.g_lo[0] + (np.arange(self.g_n[0]) + 0.5) * self.gdx
        gy = self.g_lo[1] + (np.arange(self.g_n[1]) + 0.5) * self.gdx
        gz = self.g_lo[2] + (np.arange(self.g_n[2]) + 0.5) * self.gdx
        self.gaxes = (gx, gy, gz)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        gr = np.sqrt(GX ** 2 + GY ** 2)
        self.active_mask = (
            ((gr > self.nozzle_r) | (GZ > self.nozzle_top))
        ).astype(np.float64)

        # wall sample set (static walls: normals point into the fluid domain)
        pts, nrm = self._build_wall_samples()
        self.wall_points = pts
        self.wall_normals = nrm

        self.char_length = float(ic.get("pool_radius", 1.45))
        self.ref_velocity = self.src_speed
        self.initialized = True

    def _build_wall_samples(self):
        pts, nrm = [], []
        # basin wall (inner cylinder r=1.5, z in [0.06, 0.50])
        for z in (0.08, 0.22, 0.36, 0.50):
            for k in range(12):
                th = 2.0 * np.pi * (k + 0.5 * (z > 0.3)) / 12.0
                pts.append([self.wall_r * np.cos(th), self.wall_r * np.sin(th), z])
                nrm.append([-np.cos(th), -np.sin(th), 0.0])
        # basin floor
        for rr in (0.45, 0.9, 1.3):
            for k in range(8):
                th = 2.0 * np.pi * k / 8.0
                pts.append([rr * np.cos(th), rr * np.sin(th), self.floor_z])
                nrm.append([0.0, 0.0, 1.0])
        # nozzle side (r=0.24)
        for z in (0.10, 0.26):
            for k in range(8):
                th = 2.0 * np.pi * (k + 0.5) / 8.0
                pts.append([self.nozzle_r * np.cos(th), self.nozzle_r * np.sin(th), z])
                nrm.append([np.cos(th), np.sin(th), 0.0])
        # nozzle top rim
        for k in range(8):
            th = 2.0 * np.pi * k / 8.0
            pts.append([0.21 * np.cos(th), 0.21 * np.sin(th), self.nozzle_top])
            nrm.append([0.0, 0.0, 1.0])
        return (np.asarray(pts, dtype=np.float64),
                np.asarray(nrm, dtype=np.float64))

    # ------------------------------------------------------- boundary handling
    def _project(self, x):
        """Project positions onto the frozen impermeable boundaries."""
        r = np.hypot(x[:, 0], x[:, 1])
        z = x[:, 2]
        # basin floor (inside the basin; band matches the wall clamp below)
        in_basin = r < self.wall_r + 0.06
        low = in_basin & (z < self.floor_z)
        x[low, 2] = self.floor_z
        # basin side wall: clamp small penetrations from the inside
        m = (z < self.wall_top) & (r > self.wall_r - 1e-9) & (r < self.wall_r + 0.06)
        if np.any(m):
            s = self.wall_r / np.maximum(r[m], 1e-12)
            x[m, 0] *= s
            x[m, 1] *= s
        # nozzle cylinder
        z = x[:, 2]
        r = np.hypot(x[:, 0], x[:, 1])
        m = (z > self.floor_z) & (z < self.nozzle_top) & (r < self.nozzle_r)
        if np.any(m):
            pen_r = self.nozzle_r - r[m]
            pen_z = self.nozzle_top - z[m]
            side = pen_r <= pen_z
            idx = np.nonzero(m)[0]
            iside = idx[side]
            itop = idx[~side]
            if len(iside):
                s = self.nozzle_r / np.maximum(r[iside], 1e-12)
                x[iside, 0] *= s
                x[iside, 1] *= s
            if len(itop):
                x[itop, 2] = self.nozzle_top
        return x

    def _wall_velocity_clamp(self, x, v):
        r = np.hypot(x[:, 0], x[:, 1])
        z = x[:, 2]
        m = (z < self.wall_top) & (r > self.wall_r - 0.02) & (r < self.wall_r + 0.06)
        if np.any(m):
            rn = np.stack((x[m, 0], x[m, 1]), axis=1)
            rn /= np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-12)
            vr = np.sum(v[m, :2] * rn, axis=1)
            out = vr > 0.0
            idx = np.nonzero(m)[0][out]
            if len(idx):
                v[idx, :2] -= vr[out][:, None] * rn[out]
        low = (r < self.wall_r + 0.06) & (z < self.floor_z + 0.02)
        v[low, 2] = np.maximum(v[low, 2], 0.0)
        mn = (z > self.floor_z) & (z < self.nozzle_top + 0.02) & (r < self.nozzle_r + 0.02) & (r > self.nozzle_r - 0.03)
        if np.any(mn):
            rn = np.stack((x[mn, 0], x[mn, 1]), axis=1)
            rn /= np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-12)
            vr = np.sum(v[mn, :2] * rn, axis=1)
            inward = vr < 0.0
            idx = np.nonzero(mn)[0][inward]
            if len(idx):
                v[idx, :2] -= vr[inward][:, None] * rn[inward]
        mt = (z < self.nozzle_top + 0.02) & (z > self.nozzle_top - 0.03) & (r < self.nozzle_r)
        v[mt, 2] = np.maximum(v[mt, 2], 0.0)
        return v

    # ------------------------------------------------------------- neighbor list
    def _build_pairs(self, x):
        h = self.h
        cx = np.floor(x[:, 0] / h).astype(np.int64)
        cy = np.floor(x[:, 1] / h).astype(np.int64)
        cz = np.floor(x[:, 2] / h).astype(np.int64)
        B = 4096
        key = (cx + B) + (cy + B) * 8192 + (cz + B) * 8192 ** 2
        order = np.argsort(key, kind="stable")
        skey = key[order]
        uk, first, counts = np.unique(skey, return_index=True, return_counts=True)
        nuc = len(uk)
        ucz = uk // 8192 ** 2 - B
        rem = uk % 8192 ** 2
        ucy = rem // 8192 - B
        ucx = rem % 8192 - B

        pi_list, pj_list = [], []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    tk = (ucx + ox + B) + (ucy + oy + B) * 8192 + (ucz + oz + B) * 8192 ** 2
                    pos = np.searchsorted(uk, tk)
                    pos = np.minimum(pos, nuc - 1)
                    found = uk[pos] == tk
                    mi = np.nonzero(found)[0]
                    if len(mi) == 0:
                        continue
                    ni = counts[mi]
                    nj = counts[pos[mi]]
                    tot = ni * nj
                    total = int(tot.sum())
                    if total == 0:
                        continue
                    cum = np.cumsum(tot) - tot
                    loc = np.arange(total, dtype=np.int64) - np.repeat(cum, tot)
                    njr = np.repeat(nj, tot)
                    a = np.repeat(first[mi], tot) + loc // njr
                    b = np.repeat(first[pos[mi]], tot) + loc % njr
                    pi_list.append(order[a])
                    pj_list.append(order[b])
        if not pi_list:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        pi = np.concatenate(pi_list)
        pj = np.concatenate(pj_list)
        keep = pi != pj
        pi, pj = pi[keep], pj[keep]
        # support filter with a small margin: pairs may drift into the kernel
        # support while the constraint iterations move particles.
        d = x[pi] - x[pj]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = r2 < (1.08 * h) ** 2
        return pi[keep], pj[keep]

    # ------------------------------------------------------------------ sources
    def _emit(self, dt):
        self.emit_acc += self.emit_rate * dt
        n = int(self.emit_acc)
        if n <= 0 or len(self.x) > 16000:
            return
        self.emit_acc -= n
        newx = np.empty((n, 3))
        newv = np.empty((n, 3))
        for k in range(n):
            site = self._emit_sites[self._emit_counter % len(self._emit_sites)]
            self._emit_counter += 1
            ang = 2.399963 * self._emit_counter  # golden-angle spread
            c, s = np.cos(ang), np.sin(ang)
            p = np.array([c * site[0] - s * site[1], s * site[0] + c * site[1]])
            jit = self.rng.uniform(-0.02, 0.02, size=3)
            newx[k, 0] = p[0] + jit[0]
            newx[k, 1] = p[1] + jit[1]
            newx[k, 2] = self.src_z + 0.5 * jit[2]
            dv = self.rng.uniform(-0.08, 0.08, size=2)
            newv[k, 0] = dv[0]
            newv[k, 1] = dv[1]
            newv[k, 2] = self.src_speed
        self.x = np.vstack((self.x, newx))
        self.v = np.vstack((self.v, newv))
        self.mass = np.concatenate((self.mass, np.full(n, self.m_p)))
        self.phase = np.concatenate((self.phase, np.ones(n)))
        self.mass_added += n * self.m_p

    def _remove_escaped(self):
        z = self.x[:, 2]
        # outside the basin there is no floor: liquid that spilled over the
        # rim falls through and leaves the domain; anything above the open
        # top also leaves.  Both are counted as removed prescribed mass.
        gone = (z < 0.0) | (z > 3.3)
        if np.any(gone):
            self.mass_removed += float(self.mass[gone].sum())
            keep = ~gone
            self.x = self.x[keep]
            self.v = self.v[keep]
            self.mass = self.mass[keep]
            self.phase = self.phase[keep]

    # -------------------------------------------------------------- PBF substep
    def _substep(self, dt):
        self._emit(dt)
        x0 = self.x
        v = self.v + self.gravity * dt
        xp = x0 + v * dt
        xp = self._project(xp)
        pi, pj = self._build_pairs(xp)
        rho = None
        n_p = len(xp)
        if len(pi) > 0:
            mr = self.m_p / self.rho0
            for _ in range(self.iterations):
                d = xp[pi] - xp[pj]
                r2 = np.einsum("ij,ij->i", d, d)
                # exclude out-of-support and exactly coincident pairs (the
                # latter can occur when walls collapse positions; their
                # gradient is singular and their kernel value is finite)
                mask = (r2 < self.h2) & (r2 > 1e-12)
                if not np.any(mask):
                    break
                pi_m, pj_m = pi[mask], pj[mask]
                d = d[mask]
                r2 = r2[mask]
                r = np.sqrt(r2)
                w = self.k_poly6 * (self.h2 - r2) ** 3
                # density (self contribution + neighbors)
                rho = np.full(n_p, self.m_p * self.w0)
                rho += self.m_p * np.bincount(pi_m, weights=w, minlength=n_p)
                c = rho / self.rho0 - 1.0
                # clamp the attraction at the free surface (density deficit)
                c = np.maximum(c, -0.1)
                # spiky gradient vectors: gw = grad_{pi} W(|pi - pj|)
                gw = (self.k_spiky * (self.h - r) ** 2 / r)[:, None] * d  # d = xi - xj
                gnorm2 = np.einsum("ij,ij->i", gw, gw)
                denom = (mr * mr) * np.bincount(pi_m, weights=gnorm2, minlength=n_p)
                lam = -c / (denom + self.eps_lambda)
                # dp_i = (1/rho0) * sum_j m (lam_i + lam_j) gradW
                lc = (lam[pi_m] + lam[pj_m]) * mr
                dp = np.empty_like(xp)
                for k in range(3):
                    dp[:, k] = np.bincount(pi_m, weights=lc * gw[:, k], minlength=n_p)
                xp = xp + dp
                xp = self._project(xp)
        vnew = (xp - x0) / dt
        vnew = self._wall_velocity_clamp(xp, vnew)
        # speed safety clamp (keeps states finite under harsh impacts)
        sp = np.linalg.norm(vnew, axis=1)
        fast = sp > 15.0
        if np.any(fast):
            vnew[fast] *= (15.0 / sp[fast])[:, None]
        # XSPH viscosity (mild)
        if len(pi) > 0 and rho is not None:
            d = xp[pi] - xp[pj]
            r2 = np.einsum("ij,ij->i", d, d)
            mask = r2 < self.h2
            if np.any(mask):
                pi_m, pj_m = pi[mask], pj[mask]
                w = (self.m_p / self.rho0) * self.k_poly6 * (self.h2 - r2[mask]) ** 3
                for k in range(3):
                    dvk = self.v[pj_m, k] - self.v[pi_m, k]
                    vnew[:, k] += self.xsph_c * np.bincount(pi_m, weights=w * dvk, minlength=n_p)
        self.x = xp
        self.v = vnew
        self._remove_escaped()

    # ------------------------------------------------------------- diagnostics
    def _grid_fields(self):
        n = self.g_n
        lo = self.g_lo
        idx = np.floor((self.x - lo) / self.gdx).astype(np.int64)
        np.clip(idx, 0, n - 1, out=idx)
        flat = (idx[:, 0] * n[1] + idx[:, 1]) * n[2] + idx[:, 2]
        ncell = int(np.prod(n))
        count = np.bincount(flat, minlength=ncell).astype(np.float64)
        count = count.reshape(n)
        inside = count > 0.0
        from scipy.ndimage import distance_transform_edt

        d_out = distance_transform_edt(~inside, sampling=self.gdx)
        d_in = distance_transform_edt(inside, sampling=self.gdx)
        level = (d_out - d_in).astype(np.float64)
        vel = np.zeros(tuple(n) + (3,), dtype=np.float64)
        for k in range(3):
            s = np.bincount(flat, weights=self.v[:, k], minlength=ncell)
            s = s.reshape(n)
            nz = count > 0
            vel[..., k][nz] = s[nz] / count[nz]
        return inside, level, vel

    def _boundary_fields(self):
        pts = self.wall_points
        b, n = self.x.shape[0], len(pts)
        if b == 0:
            return np.zeros((n, 3))
        d = np.linalg.norm(self.x[:, None, :] - pts[None, :, :], axis=2)  # [N,B]
        w = np.exp(-(d / 0.16) ** 2)
        wsum = w.sum(axis=0)
        bv = (w.T @ self.v) / np.maximum(wsum, 1e-9)[:, None]
        bv[wsum < 1e-6] = 0.0
        return bv

    def _make_frame(self, t):
        inside, level, vel = self._grid_fields()
        bv = self._boundary_fields()
        fields = {
            "active_mask": self.active_mask.copy(),
            "liquid_level_set": level,
            "velocity_field": vel,
            "boundary_normal": self.wall_normals.copy(),
            "boundary_target_velocity": np.zeros_like(self.wall_normals),
            "boundary_velocity": bv,
            "cumulative_mass_source": np.array([self.mass_added - self.mass_removed]),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "source_region": self.phase.copy(),
        }
        scalars = {
            "rest_density": float(self.rho0),
            "particle_spacing": float(self.dx),
            "grid_spacing": float(self.gdx),
            "characteristic_length": float(self.char_length),
            "reference_velocity": float(self.ref_velocity),
        }
        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        n_full = int(np.floor((target - self.t) / self.dt + 1e-6))
        for _ in range(max(0, n_full)):
            self._substep(self.dt)
            self.t += self.dt
        rem = target - self.t
        if rem > 1e-9:
            self._substep(rem)
            self.t = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "num_particles": int(len(self.x)),
            "mass_added_kg": self.mass_added,
            "mass_removed_kg": self.mass_removed,
            "method": "position_based_fluid",
        }
