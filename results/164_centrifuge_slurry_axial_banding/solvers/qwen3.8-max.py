"""Physically-based solver for scene 164 (centrifuge slurry axial banding).

Physics (integrated in the frame rotating with the drum, where the cylindrical
wall and end caps are static; outputs are mapped back to the world frame with
the drum axis along world x at height z_c):

* Two particle-size species with persistent labels (species_id attribute).
* Body force: gravity reduced by Archimedean buoyancy in the carrier liquid
  (rho_f = 1000 kg/m^3), expressed in the rotating frame.
* Rotating-frame inertia: centrifugal acceleration omega^2 r_perp (also
  buoyancy-reduced through the displaced-liquid pressure gradient) and
  Coriolis acceleration -2 omega x v.
* Particle-liquid coupling: drag F = -(c_lin + c_quad|v|)(v - u_f) toward the
  liquid velocity; inside the annular bed the liquid is entrained (u_f ~ 0 in
  the rotating frame) and the drag is hindered by the local solids fraction
  (Richardson-Zaki (1-phi)^3).  Because the settling speed ~ r^2, the large
  species centrifuges outward first -> radial segregation (large at wall,
  small inward), matching the experiment.
* Contacts: linear spring-dashpot normal force plus capped Coulomb friction
  (viscous tangential dashpot limited by mu*F_n) for particle-particle and
  particle-wall pairs; neighbor search with a uniform cell list.
* Axial banding: sub-grid shear-induced migration modeled as an
  anti-diffusive (negative-diffusion) axial flux of each species,
      u_x = -LAMBDA * S^2[dc_s/dx] + D * dc_s/dx,
  where S is a Gaussian smoothing of width SIGMA (sets the selected band
  wavelength ~ 2*pi*SIGMA/e) and the flux is limited by the local species
  fraction.  This is the standard continuum instability that turns an axially
  uniform, radially segregated bed into alternating species-rich bands; it is
  mass conserving and deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from worldsolver.contracts import StateFrame

RHO_F = 1000.0            # carrier-liquid density [kg/m^3]
MU_C = 0.4                # contact / wall friction coefficient
WALL_K_MULT = 2.0         # wall stiffness multiplier
ZETA = 0.15               # normal damping ratio
V_CLAMP = 12.0            # safety velocity clamp [m/s]
DT = 5.0e-4               # integration time step [s]
K_CONTACT = 5.0e4         # particle-particle normal stiffness [N/m]
TAU_CLEAR = 1.0           # drag relaxation time in clear liquid [s]
TAU_BED = 1.0             # base drag relaxation time in the bed [s]
CD_QUAD = 0.25            # inertial drag coefficient [kg/m]
LAMBDA_BAND = 2.0e-4      # anti-diffusion banding coefficient [m^2/s]
D_BAND = 2.0e-5           # weak stabilizing axial diffusion [m^2/s]
SIGMA_BAND = 0.15         # band selection length [m]


def _gauss_kernel(sigma: float, dx: float, half: int = 5) -> np.ndarray:
    o = np.arange(-half, half + 1) * dx
    k = np.exp(-0.5 * (o / sigma) ** 2)
    return k / k.sum()


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        frame0 = scene.initial_state()

        self.seed = int(seed)
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.half_L = 0.5 * self.L
        self.zc = float(c["center_height"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.spacing = float(c["spacing"])
        self.nbins = int(c["field_grid_shape"][0])
        self.grid_shape = tuple(int(v) for v in c["field_grid_shape"])

        # State in the drum-rotating frame (aligned with world frame at t=0).
        pos = np.array(frame0.position, dtype=np.float64, copy=True)
        pos[:, 2] -= self.zc                       # put axis through origin
        self.pos = np.ascontiguousarray(pos)
        self.vel = np.ascontiguousarray(np.array(frame0.velocity, dtype=np.float64, copy=True))
        self.N = int(len(self.pos))

        self.mass = np.array(frame0.mass, dtype=np.float64, copy=True)
        self.inv_m = 1.0 / np.maximum(self.mass, 1e-12)
        attrs = frame0.attributes
        self.species = np.asarray(attrs["species_id"]).astype(np.int64).copy()
        self.radii = np.asarray(attrs["particle_radius"], dtype=np.float64).copy()
        self.mass_quantity = np.asarray(attrs["mass_quantity"], dtype=np.float64).copy()
        self.K = int(self.mass_quantity.shape[1])
        self.vol = (4.0 / 3.0) * np.pi * self.radii ** 3
        self.buoy = np.clip(1.0 - RHO_F * self.vol / np.maximum(self.mass, 1e-12), 0.05, 1.0)

        self.r_cut = 2.0 * float(self.radii.max()) + 2.0e-3
        self.k_c = K_CONTACT

        g = np.array(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=np.float64)
        self.gravity = g
        self.gmag = float(np.linalg.norm(g))

        self.dt = DT
        self.t = 0.0
        self.steps = 0
        self.max_overlap = 0.0

        # Axial banding kernel.
        self.band_dx = self.L / self.nbins
        self.band_kernel = _gauss_kernel(SIGMA_BAND, self.band_dx)
        self.band_edges = np.linspace(-self.half_L, self.half_L, self.nbins + 1)
        self.band_every = 4                          # banding substep interval

        # Diagnostic grid (world-frame bounding box of the drum).
        gx = self.grid_shape
        self.gx0 = np.array([-self.half_L, -self.R, self.zc - self.R])
        self.gd = np.array([self.L / gx[0], 2 * self.R / gx[1], 2 * self.R / gx[2]])
        yc = (np.arange(gx[1]) + 0.5) * self.gd[1] - self.R
        zc = (np.arange(gx[2]) + 0.5) * self.gd[2] - self.R
        YY, ZZ = np.meshgrid(yc, zc, indexing="ij")
        inside = (YY ** 2 + ZZ ** 2) <= self.R ** 2
        self.active_mask = np.broadcast_to(inside, gx).copy()

        # Force accumulators (rotating frame) for export.
        self.F_contact = np.zeros((self.N, 3))
        self.F_drag = np.zeros((self.N, 3))

    # --------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = int(np.ceil((float(time) - self.t) / self.dt - 1e-9))
        target = max(target, 0)
        for _ in range(target):
            self._substep(self.dt)
            self.steps += 1
        self.t += target * self.dt
        return self._build_frame(self.t)

    # ------------------------------------------------------------ cell pairs
    def _build_pairs(self):
        p = self.pos
        cs = self.r_cut
        nx = int(np.floor((p[:, 0].max() - p[:, 0].min()) / cs)) + 2 if self.N else 2
        ny = int(np.floor((p[:, 1].max() - p[:, 1].min()) / cs)) + 2 if self.N else 2
        nz = int(np.floor((p[:, 2].max() - p[:, 2].min()) / cs)) + 2 if self.N else 2
        x0 = p[:, 0].min() - 0.5 * cs if self.N else 0.0
        y0 = p[:, 1].min() - 0.5 * cs if self.N else 0.0
        z0 = p[:, 2].min() - 0.5 * cs if self.N else 0.0
        ncell = nx * ny * nz
        ix = np.clip(((p[:, 0] - x0) / cs).astype(np.int64), 0, nx - 1)
        iy = np.clip(((p[:, 1] - y0) / cs).astype(np.int64), 0, ny - 1)
        iz = np.clip(((p[:, 2] - z0) / cs).astype(np.int64), 0, nz - 1)
        flat = (ix * ny + iy) * nz + iz
        order = np.argsort(flat, kind="stable")
        fs = flat[order]
        cell_start = np.searchsorted(fs, np.arange(ncell), side="left")
        cell_end = np.searchsorted(fs, np.arange(ncell), side="right")
        counts = cell_end - cell_start
        occ = np.flatnonzero(counts)
        if occ.size == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64))

        i_list = []
        j_list = []

        # same-cell pairs (unique, sorted order within cell)
        cnt_o = counts[occ]
        sel = np.flatnonzero(cnt_o > 1)
        if sel.size:
            cs_i = cell_start[occ[sel]]
            ln_i = cnt_o[sel]
            mx = int(ln_i.max()) - 1
            rep = np.repeat(cs_i, mx)
            kof = np.tile(np.arange(1, mx + 1), sel.size)
            valid = kof <= np.repeat(ln_i - 1, mx)
            base = rep[valid]
            k = kof[valid]
            i_list.append(order[base])
            j_list.append(order[base + k])

        # neighbor-cell pairs, half stencil (each unordered pair once)
        half = []
        for dx in (0, 1, -1):
            for dy in (0, 1, -1):
                for dz in (0, 1, -1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                        half.append(dx * ny * nz + dy * nz + dz)
        cnt_occ = counts[occ]
        start_occ = cell_start[occ]
        for dflat in half:
            nc = occ + dflat
            ok = (nc >= 0) & (nc < ncell)
            if not np.any(ok):
                continue
            occ_i = occ[ok]
            nc_i = nc[ok]
            ci = counts[occ_i]
            cn = counts[nc_i]
            prod = ci * cn
            tot = int(prod.sum())
            if tot == 0:
                continue
            csum = np.concatenate(([0], np.cumsum(prod)))
            seg = np.repeat(np.arange(prod.size), prod)
            within = np.arange(tot) - csum[seg]
            cn_full = np.repeat(cn, ci)
            slot_i = within // cn_full
            slot_j = within % cn_full
            a = order[np.repeat(start_occ[ok], prod) + slot_i]
            b = order[np.repeat(cell_start[nc_i], prod) + slot_j]
            i_list.append(a)
            j_list.append(b)

        if not i_list:
            return (np.empty(0, np.int64), np.empty(0, np.int64))
        return np.concatenate(i_list), np.concatenate(j_list)

    def _substep(self, dt: float):
        N = self.N
        if N == 0:
            return
        p = self.pos
        v = self.vel
        m = self.mass
        rad = self.radii

        # ---- body forces (rotating frame) ----
        th = self.omega * self.t
        ct, st = np.cos(th), np.sin(th)
        gw = self.gravity
        g_rot = (gw[0],
                 ct * gw[1] + st * gw[2],
                 -st * gw[1] + ct * gw[2])

        acc = np.empty((N, 3))
        acc[:, 0] = g_rot[0] * self.buoy
        acc[:, 1] = g_rot[1] * self.buoy + self.omega ** 2 * p[:, 1] * self.buoy
        acc[:, 2] = g_rot[2] * self.buoy + self.omega ** 2 * p[:, 2] * self.buoy
        # Coriolis (omega along +x): -2 w x v -> a_y = -2 w vz, a_z = +2 w vy
        acc[:, 1] -= 2.0 * self.omega * v[:, 2]
        acc[:, 2] += 2.0 * self.omega * v[:, 1]

        # ---- liquid drag (hindered by local solids fraction) ----
        eps = self.r_cut
        nbx = max(int(np.ceil(2 * self.half_L / eps)), 1)
        nby = max(int(np.ceil(2 * self.R / eps)), 1)
        nbz = nby
        bix = np.clip(((p[:, 0] + self.half_L) / eps).astype(np.int64), 0, nbx - 1)
        biy = np.clip(((p[:, 1] + self.R) / eps).astype(np.int64), 0, nby - 1)
        biz = np.clip(((p[:, 2] + self.R) / eps).astype(np.int64), 0, nbz - 1)
        flat_b = (bix * nby + biy) * nbz + biz
        ncell = nbx * nby * nbz
        nb_count = np.bincount(flat_b, minlength=ncell)
        vsum = np.bincount(flat_b, weights=self.vol, minlength=ncell)
        phi = np.clip(vsum / (eps ** 3), 0.0, 0.65)
        phi_p = phi[flat_b]
        hinder = np.maximum(1.0 - phi_p, 0.05) ** 3
        in_bed = phi_p > 0.02
        tau = np.where(in_bed, TAU_BED, TAU_CLEAR) * hinder ** (-1.0) * hinder
        tau = np.where(in_bed, TAU_BED / hinder, TAU_CLEAR)
        c_lin = m / tau
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        c_tot = c_lin + CD_QUAD * self.vol ** (2.0 / 3.0) * speed
        Fd = -c_tot[:, None] * v
        acc += Fd * self.inv_m[:, None]

        # ---- particle-particle contacts ----
        Fc = np.zeros((N, 3))
        I, J = self._build_pairs()
        if I.size:
            d = p[J] - p[I]
            dist2 = np.einsum("ij,ij->i", d, d)
            rsum = rad[I] + rad[J]
            contact = dist2 < rsum * rsum
            if np.any(contact):
                Ic = I[contact]
                Jc = J[contact]
                dc = d[contact]
                distc = np.sqrt(dist2[contact])
                ovc = rsum[contact] - distc
                nrm = dc / np.maximum(distc, 1e-12)[:, None]
                mI, mJ = m[Ic], m[Jc]
                mred = mI * mJ / (mI + mJ)
                c_damp = 2.0 * ZETA * np.sqrt(self.k_c * mred)
                dv = v[Jc] - v[Ic]
                vn = np.einsum("ij,ij->i", dv, nrm)
                fn = np.maximum(self.k_c * ovc - c_damp * vn, 0.0)
                vt = dv - vn[:, None] * nrm
                vt_n = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                ft_mag = np.minimum(MU_C * fn, 0.5 * c_damp * vt_n)
                inv_vt = np.where(vt_n > 1e-12, ft_mag / np.maximum(vt_n, 1e-12), 0.0)
                Fpair = fn[:, None] * nrm - inv_vt[:, None] * vt
                np.add.at(Fc, Ic, Fpair)
                np.add.at(Fc, Jc, -Fpair)
                mo = float(ovc.max())
                if mo > self.max_overlap:
                    self.max_overlap = mo
        acc += Fc * self.inv_m[:, None]

        # ---- walls (static in rotating frame) ----
        Fw = np.zeros((N, 3))
        kw = WALL_K_MULT * self.k_c
        rr = np.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
        pen = rr + rad - self.R
        idx = np.flatnonzero(pen > 0.0)
        if idx.size:
            rri = np.maximum(rr[idx], 1e-12)
            nrm = np.zeros((idx.size, 3))
            nrm[:, 1] = -p[idx, 1] / rri
            nrm[:, 2] = -p[idx, 2] / rri
            vi = v[idx]
            vn = np.einsum("ij,ij->i", vi, nrm)
            mi = m[idx]
            c_w = 2.0 * ZETA * np.sqrt(kw * mi)
            fn = np.maximum(kw * pen[idx] - c_w * np.minimum(vn, 0.0), 0.0)
            vt = vi - vn[:, None] * nrm
            vt_n = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            ft_mag = np.minimum(MU_C * fn, 0.5 * c_w * vt_n)
            inv_vt = np.where(vt_n > 1e-12, ft_mag / np.maximum(vt_n, 1e-12), 0.0)
            np.add.at(Fw, idx, fn[:, None] * nrm - inv_vt[:, None] * vt)
        # end caps at x = +/- half_L
        for sgn in (1.0, -1.0):
            penx = sgn * p[:, 0] + rad - self.half_L
            ix = np.flatnonzero(penx > 0.0)
            if ix.size:
                vi = v[ix]
                vn = -sgn * vi[:, 0]
                mi = m[ix]
                c_w = 2.0 * ZETA * np.sqrt(kw * mi)
                fn = np.maximum(kw * penx[ix] - c_w * np.minimum(vn, 0.0), 0.0)
                vt = vi.copy()
                vt[:, 0] = 0.0
                vt_n = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                ft_mag = np.minimum(MU_C * fn, 0.5 * c_w * vt_n)
                inv_vt = np.where(vt_n > 1e-12, ft_mag / np.maximum(vt_n, 1e-12), 0.0)
                Fw[ix, 0] += -sgn * fn
                Fw[ix, 1] += -inv_vt * vt[:, 1]
                Fw[ix, 2] += -inv_vt * vt[:, 2]
        acc += Fw * self.inv_m[:, None]

        # ---- integrate ----
        v += acc * dt
        sp = np.sqrt(np.einsum("ij,ij->i", v, v))
        bad = sp > V_CLAMP
        if np.any(bad):
            v[bad] *= (V_CLAMP / sp[bad])[:, None]
        p += v * dt

        # ---- axial banding migration flux ----
        if self.steps % self.band_every == 0:
            self._apply_banding(dt * self.band_every)

        # ---- hard containment safety net ----
        rr = np.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
        rmax = self.R - rad
        out = rr > rmax
        if np.any(out):
            scale = rmax[out] / np.maximum(rr[out], 1e-12)
            yo, zo = p[out, 1] * scale, p[out, 2] * scale
            p[out, 1] = yo
            p[out, 2] = zo
            rdot = v[out, 1] * yo + v[out, 2] * zo
            n2 = np.maximum(yo * yo + zo * zo, 1e-24)
            fac = np.where(rdot > 0.0, rdot / n2, 0.0)
            v[out, 1] -= fac * yo
            v[out, 2] -= fac * zo
        ax = np.abs(p[:, 0]) + rad - self.half_L
        o2 = np.flatnonzero(ax > 0.0)
        if o2.size:
            p[o2, 0] = np.sign(p[o2, 0]) * np.maximum(self.half_L - rad[o2], 0.0)
            v[o2, 0] = 0.0

        self.F_contact = Fc + Fw
        self.F_drag = Fd

    # ------------------------------------------------------------- banding
    def _apply_banding(self, dt: float):
        N = self.N
        if N == 0:
            return
        nb = self.nbins
        dx = self.band_dx
        ib = np.clip(np.searchsorted(self.band_edges, self.pos[:, 0], side="right") - 1, 0, nb - 1)
        cnt = np.bincount(ib, minlength=nb).astype(np.float64)
        cnt_safe = np.maximum(cnt, 1.0)
        c = np.zeros((self.K, nb))
        for s in range(self.K):
            ws = (self.species == s).astype(np.float64)
            c[s] = np.bincount(ib, weights=ws, minlength=nb) / cnt_safe
        c[:, cnt < 1.0] = 0.0
        sm = np.stack([convolve1d(c[s], self.band_kernel, mode="nearest") for s in range(self.K)])
        g = np.gradient(sm, dx, axis=1)
        g2 = np.stack([convolve1d(g[s], self.band_kernel, mode="nearest") for s in range(self.K)])
        ux = np.zeros((self.K, nb))
        for s in range(self.K):
            own = np.clip(sm[s], 0.0, 1.0)
            ux[s] = -LAMBDA_BAND * own * g2[s] + D_BAND * g[s]
        u_p = ux[self.species, ib]
        self.pos[:, 0] += u_p * dt
        self.vel[:, 0] += u_p        # migration velocity enters exported velocity

    # --------------------------------------------------------------- frame
    def _build_frame(self, time: float):
        p = self.pos
        v = self.vel
        N = self.N
        th = self.omega * time
        ct, st = np.cos(th), np.sin(th)

        pos_w = np.empty_like(p)
        pos_w[:, 0] = p[:, 0]
        pos_w[:, 1] = ct * p[:, 1] - st * p[:, 2]
        pos_w[:, 2] = st * p[:, 1] + ct * p[:, 2] + self.zc

        vel_w = np.empty_like(v)
        vel_w[:, 0] = v[:, 0]
        vel_w[:, 1] = ct * v[:, 1] - st * v[:, 2] - self.omega * (pos_w[:, 2] - self.zc)
        vel_w[:, 2] = st * v[:, 1] + ct * v[:, 2] + self.omega * pos_w[:, 1]

        # external force: gravity + buoyancy (world frame, z down)
        ext = np.zeros((N, 3))
        ext[:, 2] = -(self.mass - RHO_F * self.vol) * self.gmag

        # interaction force: contact + liquid drag, summed per species
        Ftot = self.F_contact + self.F_drag
        inter = np.zeros((self.K, 3))
        for s in range(self.K):
            w = self.species == s
            if np.any(w):
                inter[s] = Ftot[w].sum(axis=0)

        # diagnostic velocity grid (world frame)
        gx = self.grid_shape
        ncell = gx[0] * gx[1] * gx[2]
        gi = np.clip(((pos_w[:, 0] - self.gx0[0]) / self.gd[0]).astype(np.int64), 0, gx[0] - 1)
        gj = np.clip(((pos_w[:, 1] - self.gx0[1]) / self.gd[1]).astype(np.int64), 0, gx[1] - 1)
        gk = np.clip(((pos_w[:, 2] - self.gx0[2]) / self.gd[2]).astype(np.int64), 0, gx[2] - 1)
        flat = (gi * gx[1] + gj) * gx[2] + gk
        mcell = np.bincount(flat, weights=self.mass, minlength=ncell)
        mom = np.empty((ncell, 3))
        for a in range(3):
            mom[:, a] = np.bincount(flat, weights=self.mass * vel_w[:, a], minlength=ncell)
        vfield = np.zeros((ncell, 3))
        nz = mcell > 1e-12
        vfield[nz] = mom[nz] / mcell[nz, None]
        vfield = vfield.reshape(gx + (3,))

        frame = StateFrame(
            float(time), pos_w, vel_w,
            mass=self.mass.copy(),
            body_id=np.zeros(N, dtype=np.int32),
            attributes={
                "species_id": self.species.astype(np.int32),
                "particle_radius": self.radii.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.K),
                "velocity_field": vfield,
                "active_mask": self.active_mask.copy(),
                "external_force": ext,
                "interaction_force": inter,
            },
            scalars={
                "grid_spacing": self.spacing,
                "characteristic_length": 2.0 * self.R,
                "contact_length_scale": self.r_cut,
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict:
        ke = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel)))
        nb = self.nbins
        ib = np.clip(np.searchsorted(self.band_edges, self.pos[:, 0], side="right") - 1, 0, nb - 1)
        cnt = np.maximum(np.bincount(ib, minlength=nb), 1)
        c0 = np.bincount(ib, weights=(self.species == 0).astype(float), minlength=nb) / cnt
        c1 = np.bincount(ib, weights=(self.species == 1).astype(float), minlength=nb) / cnt
        return {
            "steps": self.steps,
            "time": self.t,
            "particle_count": self.N,
            "kinetic_energy_J": ke,
            "max_overlap_m": self.max_overlap,
            "band_contrast": float(np.std(c0 - c1)),
        }
