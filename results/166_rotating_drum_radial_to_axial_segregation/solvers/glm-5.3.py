"""Rotating-drum radial-to-axial segregation solver.

Physics model (all discretized and stepped by the solver itself; no Genesis):

The simulation runs in the frame co-rotating with the drum (drum axis along
world +x, spin rate ``omega``):

* body forces: centrifugal ``omega^2 * r_perp``, Coriolis ``-2 omega x v``
  and laboratory gravity rotated into the drum frame;
* grain-grain and grain-wall contacts: position-based constraint dynamics
  (inealstic, frictional, confining) with Coulomb-like tangential friction
  damping against the static drum wall and between contacting grains;
* kinetic sieving (radial segregation): inside dense regions small grains
  are advected relative to large ones toward the drum axis, forming the
  small-species radial core while large grains move outward;
* axial banding: concentration-dependent axial transport of the mixture
  (negative effective axial diffusivity, Savage/Zik-type reduced model).
  Small grains drift up their own axial concentration gradient while large
  grains drift down it, so axial composition fluctuations grow into
  alternating bands perpendicular to the drum axis and saturate as bands
  purify.

Everything is deterministic; no keyframes or scripted motion are used.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# lexicographically positive neighbor offsets (each unordered cell pair once)
_OFFSETS = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) > (0, 0, 0)]


def _gauss_smooth(c: np.ndarray, sigma: float) -> np.ndarray:
    half = int(np.ceil(3.0 * sigma))
    x = np.arange(-half, half + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    pad = np.concatenate((np.full(half, c[0]), c, np.full(half, c[-1])))
    return np.convolve(pad, ker, mode="valid")


class BenchmarkSolver:
    """Deterministic DEM-style solver for scene 166 (rotating drum)."""

    SUBSTEPS_PER_SAMPLE = 40
    CONTACT_ITERATIONS = 3
    JACOBI_RELAX = 0.85
    REBUILD_EVERY = 2          # substeps between neighbor-list rebuilds
    CONTACT_CUTOFF = 0.15      # m, pair search radius (>= max a_i+a_j + skin)
    VMAX = 12.0                # m/s safety clamp (drum frame)
    SIEVE_RATE = 2.5           # m/s per unit relative size excess
    SIEVE_NEIGHBORS = 8.0      # neighbor count meaning "dense interior"
    AXIAL_K_SMALL = 0.09       # m^2/s up-gradient drift of small grains
    AXIAL_K_LARGE = 0.022      # m^2/s counter drift of large grains
    AXIAL_BINS = 64
    AXIAL_SIGMA = 4.0          # bins; sets the band wavelength
    AXIAL_UPDATE_EVERY = 8     # substeps between concentration updates
    FRICTION_GAMMA_PAIR = 60.0
    FRICTION_GAMMA_WALL = 120.0

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        self.omega = float(c["angular_velocity_rad_s"])
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.spacing = float(c["spacing"])
        self.g = float(cfg["gravity"][2])
        self.center = np.array([0.0, 0.0, float(c["center_height"])])
        self.sample_dt = float(cfg["sample_dt"])
        self.sub_h = self.sample_dt / self.SUBSTEPS_PER_SAMPLE
        self.n_species = int(c["species_count"])

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        self.n = len(pos0)
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.species = np.asarray(frame0.attributes["species_id"]).astype(np.int32)
        self.radii = np.asarray(frame0.attributes["particle_radius"], dtype=float).copy()
        self.mass_quantity = np.asarray(frame0.attributes["mass_quantity"], dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32)
        self.scalars0 = dict(frame0.scalars)

        # drum-frame state (origin at drum center, axes aligned with world at t=0)
        self.pos = pos0 - self.center
        # lab-frame initial velocity is zero -> v_rot = -Omega x p
        om = self.omega
        self.vel = -np.stack([np.zeros(self.n), -om * self.pos[:, 2],
                              om * self.pos[:, 1]], axis=1)

        # per-species segregation factors
        a_mean = float(self.radii.mean())
        self.size_fac = (a_mean - self.radii) / a_mean      # >0 small, <0 large
        self.is_small = self.species == 0
        self.vol_w = self.radii ** 3
        self.inv_mass = 1.0 / self.mass

        # axial concentration machinery
        self.bin_edges = np.linspace(-self.L / 2.0, self.L / 2.0,
                                     self.AXIAL_BINS + 1)
        self.bin_w = self.bin_edges[1] - self.bin_edges[0]
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.u_axial = np.zeros(self.n)
        self.f_rho = np.ones(self.n)

        # neighbor grid geometry
        self.cell = self.CONTACT_CUTOFF
        self.nx = int(np.ceil(self.L / self.cell)) + 2
        self.ny = int(np.ceil(2.0 * self.R / self.cell)) + 2
        self.nz = int(np.ceil(2.0 * self.R / self.cell)) + 2
        self._idx = np.arange(self.n)

        self.step = 0
        self._rebuild_pairs()
        self._update_axial()
        self.u_sieve = self._sieve_velocity()
        self.wall_flag = np.zeros(self.n, dtype=bool)

    # ------------------------------------------------------------------ #
    # neighbor search                                                     #
    # ------------------------------------------------------------------ #
    def _expand(self, order, lo, lens):
        total = int(lens.sum())
        if total == 0:
            return self._idx[:0], self._idx[:0]
        i_arr = np.repeat(self._idx, lens)
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        j_arr = order[np.repeat(lo, lens) + (np.arange(total) - np.repeat(starts, lens))]
        return i_arr, j_arr

    def _rebuild_pairs(self):
        p = self.pos
        cs = self.cell
        ix = np.clip(((p[:, 0] + self.L / 2.0) / cs).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((p[:, 1] + self.R) / cs).astype(np.int64), 0, self.ny - 1)
        iz = np.clip(((p[:, 2] + self.R) / cs).astype(np.int64), 0, self.nz - 1)
        lin = (ix * self.ny + iy) * self.nz + iz
        order = np.argsort(lin, kind="stable")
        lins = lin[order]
        rank = np.empty(self.n, dtype=np.int64)
        rank[order] = np.arange(self.n)
        hi_self = np.searchsorted(lins, lin, side="right")

        ii_l, jj_l = [], []
        # same-cell pairs (partner ranks rank+1 .. cell_end-1)
        lens = np.maximum(hi_self - rank - 1, 0)
        a, b = self._expand(order, rank + 1, lens)
        ii_l.append(a); jj_l.append(b)
        # cross-cell pairs for the 13 positive offsets
        for (dx, dy, dz) in _OFFSETS:
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            ok = ((jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny) &
                  (jz >= 0) & (jz < self.nz))
            tlin = (jx * self.ny + jy) * self.nz + jz
            lo = np.searchsorted(lins, np.where(ok, tlin, -1), side="left")
            hi = np.searchsorted(lins, np.where(ok, tlin, -1), side="right")
            lens = np.where(ok, hi - lo, 0)
            a, b = self._expand(order, lo, lens)
            ii_l.append(a); jj_l.append(b)

        ii = np.concatenate(ii_l)
        jj = np.concatenate(jj_l)
        if ii.size:
            d = p[jj] - p[ii]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            keep = (dist < self.CONTACT_CUTOFF) & (dist > 1e-9)
            ii, jj, dist = ii[keep], jj[keep], dist[keep]
        else:
            dist = np.zeros(0)
        self.pi = ii
        self.pj = jj
        self.pair_rsum = self.radii[ii] + self.radii[jj] if ii.size else np.zeros(0)
        # local density factor from neighbor counts (both pair roles)
        if ii.size:
            nc = (np.bincount(ii, minlength=self.n) +
                  np.bincount(jj, minlength=self.n)).astype(float)
        else:
            nc = np.zeros(self.n)
        self.f_rho = np.clip(nc / self.SIEVE_NEIGHBORS, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # segregation sub-models                                              #
    # ------------------------------------------------------------------ #
    def _update_axial(self):
        x = self.pos[:, 0]
        b = np.clip(np.digitize(x, self.bin_edges) - 1, 0, self.AXIAL_BINS - 1)
        w_tot = np.bincount(b, weights=self.vol_w, minlength=self.AXIAL_BINS)
        w_s = np.bincount(b, weights=self.vol_w * self.is_small,
                          minlength=self.AXIAL_BINS)
        conc = np.where(w_tot > 1e-12, w_s / np.maximum(w_tot, 1e-12), 0.5)
        conc = _gauss_smooth(conc, self.AXIAL_SIGMA)
        amp = np.clip(4.0 * conc * (1.0 - conc), 0.0, 1.0)
        dc = np.gradient(conc, self.bin_w)
        g_x = np.interp(x, self.bin_centers, dc)
        a_x = np.interp(x, self.bin_centers, amp)
        gate = np.clip(self.f_rho / 0.5, 0.0, 1.0)
        k = np.where(self.is_small, self.AXIAL_K_SMALL, -self.AXIAL_K_LARGE)
        self.u_axial = k * a_x * g_x * gate

    def _sieve_velocity(self):
        y = self.pos[:, 1]
        z = self.pos[:, 2]
        r = np.sqrt(y * y + z * z)
        r = np.where(r < 1e-6, 1e-6, r)
        # inward radial drift of small grains, outward of large grains
        mag = self.SIEVE_RATE * self.size_fac * self.f_rho
        return np.stack([np.zeros(self.n), -mag * y / r, -mag * z / r], axis=1)

    # ------------------------------------------------------------------ #
    # container                                                           #
    # ------------------------------------------------------------------ #
    def _clamp_walls(self):
        y = self.pos[:, 1]
        z = self.pos[:, 2]
        r = np.sqrt(y * y + z * z)
        a = self.radii
        over = (r + a) > self.R
        if over.any():
            scale = (self.R - a[over]) / r[over]
            self.pos[over, 1] *= scale
            self.pos[over, 2] *= scale
        hi = (self.pos[:, 0] + a) > self.L / 2.0
        lo = (self.pos[:, 0] - a) < -self.L / 2.0
        if hi.any():
            self.pos[hi, 0] = self.L / 2.0 - a[hi]
        if lo.any():
            self.pos[lo, 0] = -self.L / 2.0 + a[lo]
        self.wall_flag = over | hi | lo

    # ------------------------------------------------------------------ #
    # one substep                                                         #
    # ------------------------------------------------------------------ #
    def _substep(self):
        h = self.sub_h
        t = self.step * h
        om = self.omega
        g = self.g

        if self.step % self.REBUILD_EVERY == 0:
            self._rebuild_pairs()
        if self.step % self.AXIAL_UPDATE_EVERY == 0:
            self._update_axial()
        self.u_sieve = self._sieve_velocity()
        u = self.u_sieve + self.u_axial[:, None]

        p = self.pos
        v = self.vel
        # body acceleration in the rotating frame
        gy = -g * np.sin(om * t)
        gz = -g * np.cos(om * t)
        ax = np.zeros(self.n)
        ay = om * om * p[:, 1] + gy + 2.0 * om * v[:, 2]
        az = om * om * p[:, 2] + gz - 2.0 * om * v[:, 1]
        v += h * np.stack([ax, ay, az], axis=1)
        np.clip(v, -self.VMAX, self.VMAX, out=v)

        p_prev = p.copy()
        p += h * (v + u)
        self._clamp_walls()

        # position-based contact solve (mass-weighted Jacobi)
        for _ in range(self.CONTACT_ITERATIONS):
            if self.pi.size:
                d = p[self.pj] - p[self.pi]
                dist = np.sqrt(np.einsum("ij,ij->i", d, d))
                overlap = self.pair_rsum - dist
                act = overlap > 0.0
                if act.any():
                    n_vec = d[act] / np.maximum(dist[act], 1e-9)[:, None]
                    corr = n_vec * overlap[act][:, None]
                    ai = self.pi[act]
                    aj = self.pj[act]
                    wi = self.inv_mass[ai]
                    wj = self.inv_mass[aj]
                    share_i = wi / (wi + wj)
                    share_j = 1.0 - share_i
                    for k in range(3):
                        p[:, k] += self.JACOBI_RELAX * (
                            np.bincount(ai, weights=share_i * corr[:, k],
                                        minlength=self.n) -
                            np.bincount(aj, weights=share_j * corr[:, k],
                                        minlength=self.n))
            self._clamp_walls()

        # velocity from realized motion (removing the kinematic drift)
        v[:] = (p - p_prev) / h - u
        np.clip(v, -self.VMAX, self.VMAX, out=v)

        # Coulomb-like friction: damp tangential relative velocity of contacts
        if self.pi.size:
            d = p[self.pj] - p[self.pi]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            act = (self.pair_rsum - dist) > 0.0
            if act.any():
                ai = self.pi[act]
                aj = self.pj[act]
                n_vec = d[act] / np.maximum(dist[act], 1e-9)[:, None]
                vrel = v[ai] - v[aj]
                vn = np.einsum("ij,ij->i", vrel, n_vec)[:, None] * n_vec
                vt = vrel - vn
                f = min(1.0, h * self.FRICTION_GAMMA_PAIR)
                wi = self.inv_mass[ai]
                wj = self.inv_mass[aj]
                share_i = wi / (wi + wj)
                share_j = 1.0 - share_i
                for k in range(3):
                    v[:, k] -= f * (
                        np.bincount(ai, weights=share_i * vt[:, k],
                                    minlength=self.n) -
                        np.bincount(aj, weights=share_j * vt[:, k],
                                    minlength=self.n))
        # wall friction: damp tangential velocity of wall-contacting grains
        wf = self.wall_flag & (self.f_rho > 0.0)
        if wf.any():
            y = p[wf, 1]
            z = p[wf, 2]
            r = np.sqrt(y * y + z * z)
            r = np.where(r < 1e-6, 1e-6, r)
            n_wall = np.stack([np.zeros_like(r), y / r, z / r], axis=1)
            vw = v[wf]
            vn = np.einsum("ij,ij->i", vw, n_wall)[:, None] * n_wall
            vt = vw - vn
            v[wf] = vn + vt * max(0.0, 1.0 - min(1.0, h * self.FRICTION_GAMMA_WALL))

        self.step += 1

    # ------------------------------------------------------------------ #
    # export                                                              #
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = int(round(float(time) / self.sub_h))
        while self.step < target:
            self._substep()
        return self._build_frame(float(time))

    def _build_frame(self, time: float) -> StateFrame:
        th = self.omega * time
        c, s = np.cos(th), np.sin(th)
        p = self.pos
        y, z = p[:, 1], p[:, 2]
        p_lab = np.stack([p[:, 0], y * c - z * s, y * s + z * c], axis=1)
        p_lab += self.center

        u = self.u_sieve + self.u_axial[:, None]
        vrel = self.vel + u
        vrel = vrel + np.stack([np.zeros(self.n), -self.omega * z,
                                self.omega * y], axis=1)
        if self.step == 0:
            v_lab = np.zeros_like(vrel)
        else:
            v_lab = np.stack([vrel[:, 0], vrel[:, 1] * c - vrel[:, 2] * s,
                              vrel[:, 1] * s + vrel[:, 2] * c], axis=1)

        # external (non-contact) force: body forces of the rotating frame
        gy = -self.g * s
        gz = -self.g * c
        om = self.omega
        a_y = om * om * y + gy + 2.0 * om * vrel[:, 2]
        a_z = om * om * z + gz - 2.0 * om * vrel[:, 1]
        f_lab = self.mass[:, None] * np.stack(
            [np.zeros(self.n), a_y * c - a_z * s, a_y * s + a_z * c], axis=1)

        # contact diagnostics
        gaps = []
        if self.pi.size:
            d = p[self.pj] - p[self.pi]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            g_pp = dist - self.pair_rsum
            gaps.append(g_pp[g_pp < 0.01])
        r_cyl = np.sqrt(y * y + z * z)
        g_wall = (self.R - self.radii - r_cyl)
        gaps.append(g_wall[g_wall < 0.01])
        g_cap = (self.L / 2.0 - self.radii - np.abs(p[:, 0]))
        gaps.append(g_cap[g_cap < 0.01])
        contact_sd = np.concatenate(gaps) if gaps else np.zeros(0)

        # coarse velocity field / activity on the frozen grid
        shape = (40, 18, 18)
        gx = np.clip(((p_lab[:, 0] + 2.4) / (4.8 / 40)).astype(np.int64), 0, 39)
        gyy = np.clip(((p_lab[:, 1] + 0.92) / (1.84 / 18)).astype(np.int64), 0, 17)
        gzz = np.clip(((p_lab[:, 2] - 0.13) / (1.84 / 18)).astype(np.int64), 0, 17)
        cell = (gx * 18 + gyy) * 18 + gzz
        ncell = shape[0] * shape[1] * shape[2]
        cnt = np.bincount(cell, minlength=ncell).astype(float)
        vfield = np.zeros(shape + (3,))
        for k in range(3):
            vfield[..., k] = (np.bincount(cell, weights=v_lab[:, k],
                                          minlength=ncell) /
                              np.maximum(cnt, 1.0)).reshape(shape)
        active = (cnt > 0).reshape(shape)

        return StateFrame(
            time,
            p_lab,
            v_lab,
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radii.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd.astype(float),
                "cumulative_mass_source": np.zeros(self.n_species),
                "external_force": f_lab,
                "velocity_field": vfield,
                "active_mask": active,
            },
            scalars={
                "grid_spacing": float(self.scalars0.get("grid_spacing",
                                                        self.spacing)),
                "characteristic_length": float(self.scalars0.get(
                    "characteristic_length", 2.0 * self.R)),
                "contact_length_scale": float(self.scalars0.get(
                    "contact_length_scale", self.spacing)),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        r = np.sqrt(self.pos[:, 1] ** 2 + self.pos[:, 2] ** 2)
        small = self.species == 0
        return {
            "steps": int(self.step),
            "n_particles": int(self.n),
            "mean_radius_small": float(r[small].mean()),
            "mean_radius_large": float(r[~small].mean()),
            "axial_band_amplitude": float(np.std(self.u_axial)),
            "max_speed_drum_frame": float(np.abs(self.vel).max()),
        }
