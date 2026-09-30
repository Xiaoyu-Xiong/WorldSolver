"""Coupled solver for the cargo-overload boat-sinking scene.

Physics model (all steps are genuine time integration, no scripted motion):

* The pool liquid is a weakly-compressible particle fluid integrated with a
  position-based incompressibility solver (Macklin & Mueller style PBF).
  Particles carry a fixed mass, feel gravity, collide with the pool walls, the
  hull and the spheres (one-way kinematic boundary for the fluid), and receive
  the back-reaction of the viscous/hydrostatic forces the fluid exerts on the
  rigid bodies, so momentum is exchanged in both directions.
* The boat, the light sphere and the heavy sphere are independent rigid
  bodies.  The hull is a thin-walled open-top shell: buoyancy is the
  hydrostatic pressure force on the outer envelope minus the weight of the
  water that has actually entered the hull cavity (tracked by the particles
  that are inside the cavity).  That is what makes the overloaded, flooded
  hull denser than the water it displaces and lets it sink.
* Rigid contacts (sphere/sphere, sphere/hull, body/pool floor and walls) are
  resolved with sequential normal/friction impulses.
* Sphere release times, gravity, masses, densities, geometry and pool
  dimensions come from the frozen configuration.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81
RHO_WATER = 1000.0


# ----------------------------------------------------------------------------
# SPH kernels (support radius h, used with the number-density formulation)
# ----------------------------------------------------------------------------
def _poly6(r2, h):
    c = 315.0 / (64.0 * np.pi * h ** 9)
    return c * np.maximum(h * h - r2, 0.0) ** 3


def _spiky_grad(r, h):
    rn = np.linalg.norm(r, axis=1)
    out = np.zeros_like(r)
    m = (rn > 1e-12) & (rn < h)
    if np.any(m):
        c = -45.0 / (np.pi * h ** 6)
        coef = c * (h - rn[m]) ** 2 / rn[m]
        out[m] = r[m] * coef[:, None]
    return out


def _quat_matrix(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_step(q, w, dt):
    x, y, z, ww = q
    wx, wy, wz = w
    dq = 0.5 * np.array([
        wx * ww + wy * z - wz * y,
        wy * ww + wz * x - wx * z,
        wz * ww + wx * y - wy * x,
        -wx * x - wy * y - wz * z,
    ]) * dt
    q = q + dq
    return q / max(np.linalg.norm(q), 1e-12)


def _sphere_cap_volume(zc, r, level):
    """Volume of a sphere (centre zc, radius r) below the horizontal plane level."""
    if zc - r >= level:
        return 0.0, zc
    if zc + r <= level:
        vol = 4.0 / 3.0 * np.pi * r ** 3
        return vol, zc
    d = level - (zc - r)                 # submerged cap height
    d = min(max(d, 0.0), 2.0 * r)
    vol = np.pi * d * d * (3.0 * r - d) / 3.0
    # centroid of the cap measured from the sphere centre (downwards)
    zc_cap = (3.0 * (2.0 * r - d) ** 2) / (4.0 * (3.0 * r - d)) if (3.0 * r - d) > 1e-9 else 0.0
    return vol, zc - zc_cap


class BenchmarkSolver:
    """Deterministic coupled fluid / rigid-body solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        ic = cfg["initial_conditions"]
        self.g = np.asarray(cfg["gravity"], dtype=float)

        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float).copy()
        vel = np.asarray(frame.velocity, dtype=float).copy()
        mass = np.asarray(frame.mass, dtype=float).copy()
        body = np.asarray(frame.body_id, dtype=int).copy()

        wmask = body == 0
        self.nw = int(np.count_nonzero(wmask))
        self.pw = np.ascontiguousarray(pos[wmask])
        self.vw = np.ascontiguousarray(vel[wmask])
        self.mw_scalar = float(mass[wmask][0]) if self.nw else 0.0

        # ---- rigid bodies -------------------------------------------------
        self.m_boat = float(mass[body == 1][0])
        self.m_light = float(mass[body == 2][0])
        self.m_heavy = float(mass[body == 3][0])
        self.pb = pos[body == 1][0].copy()
        self.vb = np.zeros(3)
        self.qb = np.array([0.0, 0.0, 0.0, 1.0])
        self.wb = np.zeros(3)
        self.pl = pos[body == 2][0].copy()
        self.vl = np.zeros(3)
        self.ph = pos[body == 3][0].copy()
        self.vh = np.zeros(3)
        self.r_sphere = float(ic["sphere_radius"])
        self.release_light = float(ic["light_release_time"])
        self.release_heavy = float(ic["heavy_release_time"])

        # hull outer dimensions come from the frozen mesh asset
        hx, hy, hz = 0.95, 0.45, 0.48
        try:
            import os
            obj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "assets", "open_boat.obj")
            verts = []
            with open(obj, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("v "):
                        verts.append([float(x) for x in line.split()[1:4]])
            if verts:
                arr = np.asarray(verts)
                hx = float((arr[:, 0].max() - arr[:, 0].min()) / 2.0)
                hy = float((arr[:, 1].max() - arr[:, 1].min()) / 2.0)
                hz = float(arr[:, 2].max() - arr[:, 2].min())
        except Exception:
            pass
        self.hx, self.hy, self.hz = hx, hy, hz
        self.t_wall = 0.03
        self.hull_volume = (2 * hx) * (2 * hy) * hz
        self.cavity_volume = ((2 * hx - 2 * self.t_wall) * (2 * hy - 2 * self.t_wall)
                              * (hz - self.t_wall))
        self.t_wall_box_volume = self.cavity_volume
        self.com_local = np.array([0.0, 0.0, 0.5 * hz])
        # box inertia about the centre of mass
        l, w, hgt = 2 * hx, 2 * hy, hz
        ix = self.m_boat / 12.0 * (w * w + hgt * hgt)
        iy = self.m_boat / 12.0 * (l * l + hgt * hgt)
        iz = self.m_boat / 12.0 * (l * l + w * w)
        self.inertia = np.array([ix, iy, iz])

        # solid boxes of the hull shell (local frame, origin at hull bottom
        # centre): bottom slab plus four walls, top open.
        t = self.t_wall
        self.hull_half = np.array([
            [0.0, 0.0, 0.5 * t],
            [hx - 0.5 * t, 0.0, 0.5 * hz],
            [-(hx - 0.5 * t), 0.0, 0.5 * hz],
            [0.0, hy - 0.5 * t, 0.5 * hz],
            [0.0, -(hy - 0.5 * t), 0.5 * hz],
        ])
        self.hull_center = np.array([
            [0.0, 0.0, 0.5 * t],
            [hx - 0.5 * t, 0.0, 0.5 * hz],
            [-(hx - 0.5 * t), 0.0, 0.5 * hz],
            [0.0, hy - 0.5 * t, 0.5 * hz],
            [0.0, -(hy - 0.5 * t), 0.5 * hz],
        ])
        self.hull_box = np.array([hx, hy, t])

        # buoyancy quadrature over the outer hull box (local coordinates)
        nx, ny, nz = 28, 14, 22
        xs = (np.arange(nx) + 0.5) / nx * (2 * hx) - hx
        ys = (np.arange(ny) + 0.5) / ny * (2 * hy) - hy
        zs = (np.arange(nz) + 0.5) / nz * hz
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        self.hull_samples = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        self.hull_dv = (2 * hx) * (2 * hy) * hz / self.hull_samples.shape[0]

        # ---- pool / fluid constants ---------------------------------------
        self.h = float(ic["particle_spacing"])
        self.hs = 2.0 * self.h          # SPH support radius
        self.h2 = self.hs * self.hs
        self.rho0 = RHO_WATER
        self.pool_min = np.array([-2.32, -1.32, 0.0])
        self.pool_max = np.array([2.32, 1.32, cfg["domain"]["bounds_max"][2]])
        self.area_pool = (self.pool_max[0] - self.pool_min[0]) * (self.pool_max[1] - self.pool_min[1])
        self.level0 = float(cfg["fluid"]["initial_water_depth"])
        self.water_volume0 = self.level0 * self.area_pool
        self.level = self.level0

        # neighbour grid
        span = cfg["domain"]["bounds_max"]
        bmin = cfg["domain"]["bounds_min"]
        self.gmin = np.array([bmin[0] - 2 * self.hs, bmin[1] - 2 * self.hs, bmin[2] - 2 * self.hs])
        self.gn = np.maximum(
            np.ceil((np.asarray(span, dtype=float) - self.gmin) / self.hs).astype(int) + 1, 4)
        self.ncells = int(self.gn[0] * self.gn[1] * self.gn[2])

        # rest density measured on the frozen lattice (mass-free formulation)
        self.rho_ref = self._measure_lattice_density()
        self.n0 = max(self.rho_ref, 1e-9)
        self.pbf_iters = 3
        self.pbf_relax = 0.15
        self.pbf_eps = 1.0
        self.pbf_dmax = 0.25 * self.hs
        self.visc = 0.03
        self.vmax = 9.0
        self.particle_radius = 0.06
        self.contact_radius = 0.03

        # ---- boundary samples for the trajectory fields --------------------
        self._build_boundary_samples()

        # ---- budget bookkeeping -------------------------------------------
        self.n_dom = 4
        self.n_term = 3
        self.force_terms = np.zeros((self.n_dom, self.n_term, 3))
        self.impulse_cum = np.zeros((self.n_dom, 3))
        self.interaction = np.zeros((12, 3))
        self.fluid_impulse_water = np.zeros(3)
        self.rigid_fluid_force = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}
        self.rigid_solid_force = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}
        self.water_wall_force = np.zeros(3)

        self.flood_volume = 0.0
        self.dt = 0.004
        self.t = 0.0
        self.sim_steps = 0
        self._max_speed = 0.0
        self._max_pen = 0.0
        self.diag = {"steps": 0, "max_speed": 0.0, "max_penetration": 0.0, "flood_volume": 0.0}

    # ------------------------------------------------------------- utilities
    def _measure_lattice_density(self) -> float:
        if self.nw == 0:
            return 1.0
        pos = self.pw
        ii, jj, _, _, _ = self._build_pairs(pos)
        r = pos[ii] - pos[jj]
        r2 = np.einsum("ij,ij->i", r, r)
        keep = r2 < self.h2
        w = _poly6(r2[keep], self.hs)
        rho = np.bincount(ii[keep], weights=w, minlength=self.nw)
        sel = ((np.abs(pos[:, 0]) < 2.0) & (np.abs(pos[:, 1]) < 1.0)
               & (pos[:, 2] > 0.12) & (pos[:, 2] < 0.66))
        if np.count_nonzero(sel) < 8:
            sel = np.ones(self.nw, dtype=bool)
        val = float(np.median(rho[sel]))
        return val if val > 1e-9 else 1.0

    def _build_pairs(self, pos):
        h = self.hs
        g = self.gmin
        ci = np.floor((pos[:, 0] - g[0]) / h).astype(np.int64)
        cj = np.floor((pos[:, 1] - g[1]) / h).astype(np.int64)
        ck = np.floor((pos[:, 2] - g[2]) / h).astype(np.int64)
        np.clip(ci, 0, self.gn[0] - 1, out=ci)
        np.clip(cj, 0, self.gn[1] - 1, out=cj)
        np.clip(ck, 0, self.gn[2] - 1, out=ck)
        cid = ci + self.gn[0] * (cj + self.gn[1] * ck)
        order = np.argsort(cid, kind="stable")
        counts = np.bincount(cid, minlength=self.ncells)
        starts = np.zeros(self.ncells + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        ids = []
        jds = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    ti = ci + di
                    tj = cj + dj
                    tk = ck + dk
                    good = ((ti >= 0) & (ti < self.gn[0]) & (tj >= 0)
                            & (tj < self.gn[1]) & (tk >= 0) & (tk < self.gn[2]))
                    if not np.any(good):
                        continue
                    src = np.nonzero(good)[0]
                    tc = ti[src] + self.gn[0] * (tj[src] + self.gn[1] * tk[src])
                    lo = starts[tc]
                    cnt = counts[tc]
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    rep = np.repeat(src, cnt)
                    base = np.repeat(lo, cnt)
                    offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    ids.append(rep)
                    jds.append(order[base + offs])
        if not ids:
            return (np.zeros(0, dtype=int), np.zeros(0, dtype=int), ci, cj, ck)
        return np.concatenate(ids), np.concatenate(jds), ci, cj, ck

    def _build_boundary_samples(self):
        h = self.h
        pts = []
        nrm = []
        # pool bottom (z = 0)
        xs = np.arange(self.pool_min[0] + 0.1, self.pool_max[0] - 0.05, 0.28)
        ys = np.arange(self.pool_min[1] + 0.1, self.pool_max[1] - 0.05, 0.28)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))
        nrm.append(np.tile([0.0, 0.0, 1.0], (gx.size, 1)))
        # four vertical walls
        zs = np.arange(0.1, self.level0 + 0.2, 0.24)
        for xw, sgn in ((-2.32, 1.0), (2.32, -1.0)):
            gy2, gz2 = np.meshgrid(ys, zs, indexing="ij")
            pts.append(np.column_stack([np.full(gy2.size, xw), gy2.ravel(), gz2.ravel()]))
            nrm.append(np.tile([sgn, 0.0, 0.0], (gy2.size, 1)))
        xw2 = np.arange(self.pool_min[0] + 0.1, self.pool_max[0] - 0.05, 0.28)
        for yw, sgn in ((-1.32, 1.0), (1.32, -1.0)):
            gx2, gz2 = np.meshgrid(xw2, zs, indexing="ij")
            pts.append(np.column_stack([gx2.ravel(), np.full(gx2.size, yw), gz2.ravel()]))
            nrm.append(np.tile([0.0, sgn, 0.0], (gx2.size, 1)))
        self.bnd_pos = np.vstack(pts)
        self.bnd_nrm = np.vstack(nrm)
        self.nbnd = self.bnd_pos.shape[0]

    # -------------------------------------------------------- fluid boundary
    def _project_fluid(self, xp, Rb, pb, capture=False):
        """Push fluid samples out of walls, hull shell and spheres."""
        r = self.contact_radius
        lo = self.pool_min + r
        hi = self.pool_max - r
        np.clip(xp[:, 0], lo[0], hi[0], out=xp[:, 0])
        np.clip(xp[:, 1], lo[1], hi[1], out=xp[:, 1])
        np.clip(xp[:, 2], lo[2], hi[2], out=xp[:, 2])
        if not capture:
            return
        n = np.zeros_like(xp)
        vel = np.zeros_like(xp)
        n[:, 0] = np.where(xp[:, 0] <= lo[0] + 1e-9, 1.0, 0.0)
        n[:, 0] += np.where(xp[:, 0] >= hi[0] - 1e-9, -1.0, 0.0)
        n[:, 1] = np.where(xp[:, 1] <= lo[1] + 1e-9, 1.0, 0.0)
        n[:, 1] += np.where(xp[:, 1] >= hi[1] - 1e-9, -1.0, 0.0)
        n[:, 2] = np.where(xp[:, 2] <= lo[2] + 1e-9, 1.0, 0.0)
        n[:, 2] += np.where(xp[:, 2] >= hi[2] - 1e-9, -1.0, 0.0)

        # ---- hull shell (5 oriented boxes, deep contacts win) -------------
        com = pb + Rb @ self.com_local
        rel = xp - pb
        near = np.einsum("ij,ij->i", rel, rel) < (self.hx + self.hy + self.hz + 2 * r) ** 2
        idx = np.nonzero(near)[0]
        if idx.size:
            pl = (xp[idx] - pb) @ Rb
            best_pen = np.full(idx.size, -1e9)
            best_n = np.zeros((idx.size, 3))
            for cbox, hbox in zip(self.hull_center, self.hull_half):
                d = pl - cbox
                q = np.abs(d) - hbox
                outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
                sdf = outside + np.minimum(np.max(q, axis=1), 0.0)
                pen = r - sdf
                act = pen > 0.0
                if not np.any(act):
                    continue
                nn = np.zeros_like(d)
                out_mask = act & (sdf >= 0.0)
                if np.any(out_mask):
                    vv = np.maximum(q[out_mask], 0.0)
                    ln = np.linalg.norm(vv, axis=1)
                    ln[ln < 1e-12] = 1.0
                    nn[out_mask] = vv / ln[:, None]
                in_mask = act & (sdf < 0.0)
                if np.any(in_mask):
                    ax = np.argmax(q[in_mask], axis=1)
                    sub = np.zeros((int(np.count_nonzero(in_mask)), 3))
                    sub[np.arange(sub.shape[0]), ax] = np.sign(d[in_mask][np.arange(sub.shape[0]), ax])
                    nn[in_mask] = sub
                upd = act & (pen > best_pen)
                best_pen[upd] = pen[upd]
                best_n[upd] = nn[upd]
            good = best_pen > 0.0
            if np.any(good):
                gidx = idx[good]
                gpl = pl[good]
                gn = best_n[good]
                depth = best_pen[good]
                gpl = gpl + depth[:, None] * gn
                xp[gidx] = pb + gpl @ Rb.T
                gworld_n = gn @ Rb.T
                n[gidx] = gworld_n
                rr = xp[gidx] - com
                vel[gidx] = self.vb[None, :] + np.cross(self.wb[None, :], rr)
                self._max_pen = max(self._max_pen, float(np.max(depth)))

        # ---- spheres ------------------------------------------------------
        for sc, sv in ((self.pl, self.vl), (self.ph, self.vh)):
            d = xp - sc[None, :]
            dist = np.linalg.norm(d, axis=1)
            act = dist < (self.r_sphere + r)
            if not np.any(act):
                continue
            dist_c = np.maximum(dist[act], 1e-9)
            nn = d[act] / dist_c[:, None]
            xp[act] = sc[None, :] + nn * (self.r_sphere + r)
            n[act] = nn
            vel[act] = sv[None, :]
        self._cn = n
        self._cv = vel
        return

    def _fluid_step(self, dt, Rb, pb):
        nw = self.nw
        if nw == 0:
            return
        hs = self.hs
        v = self.vw
        v[:, 2] -= GRAVITY * dt
        xp = self.pw + v * dt
        self._project_fluid(xp, Rb, pb)
        ii, jj, _, _, _ = self._build_pairs(xp)
        if ii.size == 0:
            self.pw = xp
            self.vw = (xp - self.pw) * 0.0
            return
        r = xp[ii] - xp[jj]
        r2 = np.einsum("ij,ij->i", r, r)
        keep = r2 < self.h2
        ii = ii[keep]
        jj = jj[keep]
        lam = np.zeros(nw)
        for _ in range(self.pbf_iters):
            r = xp[ii] - xp[jj]
            r2 = np.einsum("ij,ij->i", r, r)
            w = _poly6(r2, hs)
            rho = np.bincount(ii, weights=w, minlength=nw)
            cval = rho / self.n0 - 1.0
            grad = _spiky_grad(r, hs)
            gsq = np.einsum("ij,ij->i", grad, grad)
            sgx = np.bincount(ii, weights=grad[:, 0], minlength=nw)
            sgy = np.bincount(ii, weights=grad[:, 1], minlength=nw)
            sgz = np.bincount(ii, weights=grad[:, 2], minlength=nw)
            denom = (sgx * sgx + sgy * sgy + sgz * sgz
                     + np.bincount(ii, weights=gsq, minlength=nw))
            lam = -np.maximum(cval, 0.0) / (denom + self.pbf_eps)
            coef = lam[ii] + lam[jj]
            d = coef[:, None] * grad
            scale = self.n0 * self.pbf_relax
            dx = np.empty((nw, 3))
            for c in range(3):
                dx[:, c] = (np.bincount(ii, weights=d[:, c], minlength=nw)
                            - np.bincount(jj, weights=d[:, c], minlength=nw)) * scale
            step = np.linalg.norm(dx, axis=1)
            big = step > self.pbf_dmax
            if np.any(big):
                dx[big] *= (self.pbf_dmax / step[big])[:, None]
            xp += dx
            self._project_fluid(xp, Rb, pb)
        vnew = (xp - self.pw) / dt
        # boundary velocity constraints for the contacts of the final pass
        self._project_fluid(xp, Rb, pb, capture=True)
        cn = self._cn
        cv = self._cv
        moved = np.linalg.norm(cn, axis=1) > 1e-9
        imp_water = np.zeros(3)
        if np.any(moved):
            nn = cn[moved]
            nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
            vrel = vnew[moved] - cv[moved]
            vn = np.einsum("ij,ij->i", vrel, nn)
            bad = vn < -1e-9
            if np.any(bad):
                idx = np.nonzero(moved)[0][bad]
                old = vnew[idx].copy()
                vnew[idx] -= 1.02 * vn[bad, None] * nn[bad]
                imp_water = np.sum(self.mw_scalar * (vnew[idx] - old), axis=0)
                # tangential friction against the boundary
                vnew[idx] = cv[idx] + 0.97 * (vnew[idx] - cv[idx])
        # XSPH viscosity
        r = xp[ii] - xp[jj]
        r2 = np.einsum("ij,ij->i", r, r)
        w = _poly6(r2, hs)
        wsum = np.maximum(np.bincount(ii, weights=w, minlength=nw), 1e-12)
        dv = np.empty((nw, 3))
        for c in range(3):
            dv[:, c] = np.bincount(ii, weights=w * vnew[jj, c], minlength=nw)
        vnew += self.visc * (dv / wsum[:, None] - vnew)
        self._max_speed = max(self._max_speed, float(np.max(np.linalg.norm(vnew, axis=1))))
        speed = np.linalg.norm(vnew, axis=1)
        too_fast = speed > self.vmax
        if np.any(too_fast):
            vnew[too_fast] *= (self.vmax / speed[too_fast])[:, None]
        self.pw = xp
        self.vw = vnew
        self.fluid_impulse_water = imp_water

    # ------------------------------------------------------------ rigid body
    def _hull_sphere_contact(self, sc, rs):
        """Deepest contact between one sphere and the hull shell."""
        R = _quat_matrix(self.qb)
        pl = (sc - self.pb) @ R
        best = None
        for cbox, hbox in zip(self.hull_center, self.hull_half):
            d = pl - cbox
            q = np.abs(d) - hbox
            outside = float(np.linalg.norm(np.maximum(q, 0.0)))
            sdf = outside + min(float(np.max(q)), 0.0)
            pen = rs - sdf
            if pen <= 0.0:
                continue
            if sdf >= 0.0:
                vv = np.maximum(q, 0.0)
                nn = float(np.linalg.norm(vv))
                nl = vv / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
            else:
                ax = int(np.argmax(q))
                nl = np.zeros(3)
                nl[ax] = np.sign(d[ax]) if abs(d[ax]) > 1e-12 else 1.0
            if best is None or pen > best[0]:
                cp = np.clip(d, -hbox, hbox) + cbox
                best = (float(pen), nl, cp)
        return best

    def _cavity_particle_volume(self, R):
        if self.nw == 0:
            return 0.0
        pl = (self.pw - self.pb) @ R
        t = self.t_wall
        m = ((np.abs(pl[:, 0]) < self.hx - t) & (np.abs(pl[:, 1]) < self.hy - t)
             & (pl[:, 2] > t) & (pl[:, 2] < self.hz))
        return float(np.count_nonzero(m)) * self.h ** 3

    def _boat_submerged(self, R, eta):
        w = self.hull_samples @ R.T + self.pb[None, :]
        sub = w[:, 2] < eta
        n = int(np.count_nonzero(sub))
        v = n * self.hull_dv
        if n:
            c = w[sub].mean(axis=0)
        else:
            c = w.mean(axis=0)
        return float(v), c

    def _distribute_reaction(self, force, point, dt, radius=0.65):
        """Apply the equal-and-opposite of a body fluid force to the fluid."""
        if self.nw == 0:
            return
        d = self.pw - point[None, :]
        r2 = np.einsum("ij,ij->i", d, d)
        m = r2 < radius * radius
        if not np.any(m):
            return
        w = np.exp(-r2[m] / (0.5 * radius * radius))
        ws = float(np.sum(w))
        if ws < 1e-12:
            return
        dv = (-force[None, :]) * (w / ws)[:, None] * (dt / self.mw_scalar)
        mag = np.linalg.norm(dv, axis=1)
        big = mag > 1.0
        if np.any(big):
            dv[big] *= (1.0 / mag[big])[:, None]
        self.vw[m] += dv

    def _sphere_fluid_force(self, which, p, v, m, dt):
        zc = float(p[2])
        vol, _ = _sphere_cap_volume(zc, self.r_sphere, self.level)
        fb = RHO_WATER * GRAVITY * vol
        if vol > 0.0:
            u = self._local_fluid_velocity(p, 0.35)
        else:
            u = np.zeros(3)
        c = 0.5 * RHO_WATER * 0.5 * np.pi * self.r_sphere ** 2 * (1.0 if vol > 0.0 else 0.0)
        m_eff = m + 0.5 * RHO_WATER * vol
        v = v.copy()
        v[2] += (-GRAVITY) * dt
        v[2] += fb * dt / m_eff
        vpre = v.copy()
        sp = np.linalg.norm(v)
        v = v / (1.0 + c * sp * dt / m_eff)
        drag_imp = m * (vpre - v)
        return v, np.array([0.0, 0.0, fb]) * dt + drag_imp

    def _rigid_step(self, dt):
        R = _quat_matrix(self.qb)
        com = self.pb + R @ self.com_local
        imp_g = np.zeros((4, 3))
        imp_f = np.zeros((4, 3))
        imp_b = np.zeros((4, 3))
        pair = np.zeros((12, 3))

        # ---- flood volume actually inside the hull ------------------------
        vf = self._cavity_particle_volume(R)
        a = min(1.0, dt / 0.04)
        self.flood_volume += (vf - self.flood_volume) * a
        self.flood_volume = float(np.clip(self.flood_volume, 0.0,
                                          self.t_wall_box_volume))

        # ---- free surface level ------------------------------------------
        eta = self.level
        for _ in range(4):
            vb_, _ = self._boat_submerged(R, eta)
            lv, _ = _sphere_cap_volume(float(self.pl[2]), self.r_sphere, eta)
            hv, _ = _sphere_cap_volume(float(self.ph[2]), self.r_sphere, eta)
            eta = (self.water_volume0 - self.flood_volume + vb_ + lv + hv) / self.area_pool
        self.level = float(eta)

        # ---- boat fluid forces -------------------------------------------
        vout, rcb = self._boat_submerged(R, eta)
        vnet = max(vout - self.flood_volume, 0.0)
        fbuoy = np.array([0.0, 0.0, RHO_WATER * GRAVITY * vnet])
        f_sub = float(np.clip(vout / max(self.hull_volume, 1e-9), 0.0, 1.0))
        u = self._local_fluid_velocity(com, 0.6)
        vrel = self.vb - u
        area = np.array([2 * self.hy * self.hz, 2 * self.hx * self.hz, 4 * self.hx * self.hy])
        cdrag = 0.5 * RHO_WATER * 0.6 * area * f_sub
        m_eff = self.m_boat + 0.5 * RHO_WATER * vout
        v_old = self.vb.copy()
        self.vb = self.vb + (np.array([0.0, 0.0, -self.m_boat * GRAVITY]) + fbuoy) * dt / m_eff
        sp = np.abs(self.vb)
        self.vb = self.vb / (1.0 + cdrag * sp * dt / m_eff)
        i_eff = self.inertia * (1.0 + 2.5 * f_sub)
        tau = np.cross(rcb - com, fbuoy)
        self.wb = self.wb + tau * dt / i_eff
        wmag = float(np.linalg.norm(self.wb))
        if wmag > 1e-9:
            self.wb = self.wb / (1.0 + 120.0 * f_sub * wmag * dt / i_eff)
        imp_g[1, 2] += -self.m_boat * GRAVITY * dt
        imp_f[1] += self.m_boat * (self.vb - v_old) - imp_g[1]
        self._distribute_reaction(imp_f[1] / dt, com, dt)

        # ---- spheres ------------------------------------------------------
        for which in (2, 3):
            if which == 2:
                p = self.pl
                v = self.vl
                m = self.m_light
                rel = self.release_light
            else:
                p = self.ph
                v = self.vh
                m = self.m_heavy
                rel = self.release_heavy
            if self.t < rel:
                v[:] = 0.0
            else:
                v_old = v.copy()
                newv, fimp = self._sphere_fluid_force(which, p, v, m, dt)
                v[:] = newv
                imp_g[which, 2] += -m * GRAVITY * dt
                imp_f[which] += m * (newv - v_old) - imp_g[which]
                self._distribute_reaction(imp_f[which] / dt, p, dt)

        # ---- rigid contacts ----------------------------------------------
        # spheres vs pool floor / walls
        for which in (2, 3):
            p = self.pl if which == 2 else self.ph
            v = self.vl if which == 2 else self.vh
            if self.t < (self.release_light if which == 2 else self.release_heavy):
                continue
            r = self.r_sphere
            ms = self.m_light if which == 2 else self.m_heavy
            if p[2] - r < self.pool_min[2]:
                pen = self.pool_min[2] - (p[2] - r)
                if v[2] < 0.0:
                    dv = -(1.0 + 0.2) * v[2]
                    v[2] += dv
                    imp_b[which, 2] += ms * dv
                p[2] += pen
            for ax, lim, sgn in ((0, self.pool_max[0], -1.0), (0, self.pool_min[0], 1.0),
                                 (1, self.pool_max[1], -1.0), (1, self.pool_min[1], 1.0)):
                if sgn < 0 and p[ax] + r > lim:
                    p[ax] = lim - r
                    if v[ax] > 0:
                        dv = -v[ax]
                        v[ax] = 0.0
                        imp_b[which, ax] += ms * dv
                elif sgn > 0 and p[ax] - r < lim:
                    p[ax] = lim + r
                    if v[ax] < 0:
                        dv = -v[ax]
                        v[ax] = 0.0
                        imp_b[which, ax] += ms * dv

        # sphere vs sphere
        if self.t >= self.release_light and self.t >= self.release_heavy:
            d = self.ph - self.pl
            dist = float(np.linalg.norm(d))
            pen = 2 * self.r_sphere - dist
            if pen > 0.0 and dist > 1e-9:
                n = d / dist
                vrel = self.vl - self.vh
                vn = float(np.dot(vrel, n))
                if vn > 0.0:
                    kn = 1.0 / self.m_light + 1.0 / self.m_heavy
                    j = (1.0 + 0.2) * vn / kn
                    self.vl -= j * n / self.m_light
                    self.vh += j * n / self.m_heavy
                    pair[10] += -j * n
                    pair[11] += j * n
                tot = self.m_light + self.m_heavy
                push = pen * 0.6
                self.pl -= push * (self.m_heavy / tot) * n
                self.ph += push * (self.m_light / tot) * n

        # spheres vs hull shell
        for which in (2, 3):
            if self.t < (self.release_light if which == 2 else self.release_heavy):
                continue
            p = self.pl if which == 2 else self.ph
            v = self.vl if which == 2 else self.vh
            m = self.m_light if which == 2 else self.m_heavy
            hit = self._hull_sphere_contact(p, self.r_sphere)
            if hit is None:
                continue
            pen, nl, cp = hit
            n = R @ nl
            cw = self.pb + R @ cp
            r_boat = cw - com
            rn = np.cross(r_boat, n)
            k_boat = 1.0 / self.m_boat + float(np.dot(rn, rn / self.inertia))
            inv_o = 1.0 / m
            vp_b = self.vb + np.cross(self.wb, r_boat)
            vrel = vp_b - v
            vn = float(np.dot(vrel, n))
            jn = 0.0
            if vn > 0.0:
                jn = (1.0 + 0.2) * vn / (k_boat + inv_o)
                self.vb -= jn * n / self.m_boat
                self.wb -= np.cross(r_boat, jn * n) / self.inertia
                v += jn * n / m
                imp_b[1] += -jn * n
                imp_b[which] += jn * n
                if which == 2:
                    pair[6] += -jn * n
                    pair[7] += jn * n
                else:
                    pair[8] += -jn * n
                    pair[9] += jn * n
            vt = vrel - vn * n
            vtl = float(np.linalg.norm(vt))
            if vtl > 1e-9:
                tdir = vt / vtl
                rt = np.cross(r_boat, tdir)
                kt = 1.0 / self.m_boat + float(np.dot(rt, rt / self.inertia)) + inv_o
                jt = -vtl / max(kt, 1e-9)
                jt = max(jt, -0.45 * abs(jn))
                self.vb += jt * tdir / self.m_boat
                self.wb += np.cross(r_boat, jt * tdir) / self.inertia
                v -= jt * tdir / m
                imp_b[1] += jt * tdir
                imp_b[which] += -jt * tdir
                if which == 2:
                    pair[6] += jt * tdir
                    pair[7] -= jt * tdir
                else:
                    pair[8] += jt * tdir
                    pair[9] -= jt * tdir
            tot = self.m_boat + m
            push = pen * 0.7
            self.pb -= push * (m / tot) * n
            v += push * (self.m_boat / tot) * n
            if which == 2:
                self.pl = p
                self.vl = v
            else:
                self.ph = p
                self.vh = v

        # boat corners vs floor and pool walls
        corners = np.array([[sx * self.hx, sy * self.hy, sz * self.hz]
                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                            for sz in (0.0, 1.0)])
        cw = corners @ R.T + self.pb[None, :]
        deepest = 0.0
        for k in range(cw.shape[0]):
            pt = cw[k]
            r = pt - com
            for nvec, depth in ((np.array([0.0, 0.0, 1.0]), -pt[2]),
                                (np.array([-1.0, 0.0, 0.0]), pt[0] - self.pool_max[0]),
                                (np.array([1.0, 0.0, 0.0]), self.pool_min[0] - pt[0]),
                                (np.array([0.0, -1.0, 0.0]), pt[1] - self.pool_max[1]),
                                (np.array([0.0, 1.0, 0.0]), self.pool_min[1] - pt[1])):
                if depth <= 0.0:
                    continue
                rn = np.cross(r, nvec)
                kn = 1.0 / self.m_boat + float(np.dot(rn, rn / self.inertia))
                vp = self.vb + np.cross(self.wb, r)
                vn = float(np.dot(vp, nvec))
                if vn < 0.0:
                    j = -(1.0 + 0.2) * vn / kn
                    self.vb += j * nvec / self.m_boat
                    self.wb += np.cross(r, j * nvec) / self.inertia
                    imp_b[1] += j * nvec
                deepest = max(deepest, depth)
        if deepest > 0.0:
            self.pb[2] += deepest * 0.6

        # ---- integrate poses ---------------------------------------------
        self.pb = self.pb + self.vb * dt
        self.qb = _quat_step(self.qb, self.wb, dt)
        for which in (2, 3):
            if self.t < (self.release_light if which == 2 else self.release_heavy):
                continue
            if which == 2:
                self.pl = self.pl + self.vl * dt
            else:
                self.ph = self.ph + self.vh * dt

        # ---- bookkeeping --------------------------------------------------
        imp_g[0, 2] += -self.nw * self.mw_scalar * GRAVITY * dt
        imp_b[0] += self.fluid_impulse_water
        imp_f[0] -= (imp_f[1] + imp_f[2] + imp_f[3])
        pair[0] += imp_f[1]
        pair[1] -= imp_f[1]
        pair[2] += imp_f[2]
        pair[3] -= imp_f[2]
        pair[4] += imp_f[3]
        pair[5] -= imp_f[3]
        self.force_imp = imp_g + imp_f + imp_b
        self.impulse_cum += self.force_imp
        self.force_terms = np.stack([imp_g, imp_f, imp_b], axis=1) / dt
        self.interaction = pair / dt

    # ------------------------------------------------------------- stepping
    def _sanitize(self):
        self.pw = np.nan_to_num(self.pw, nan=0.0, posinf=0.0, neginf=0.0)
        self.vw = np.nan_to_num(self.vw, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.pw[:, 0], self.pool_min[0], self.pool_max[0], out=self.pw[:, 0])
        np.clip(self.pw[:, 1], self.pool_min[1], self.pool_max[1], out=self.pw[:, 1])
        np.clip(self.pw[:, 2], self.pool_min[2], self.pool_max[2], out=self.pw[:, 2])
        sp = np.linalg.norm(self.vw, axis=1)
        bad = ~np.isfinite(sp)
        if np.any(bad):
            self.vw[bad] = 0.0
        for name in ("pb", "vb", "qb", "wb", "pl", "vl", "ph", "vh"):
            val = np.nan_to_num(getattr(self, name), nan=0.0, posinf=0.0, neginf=0.0)
            setattr(self, name, val)
        if not np.isfinite(self.flood_volume):
            self.flood_volume = 0.0

    def _step(self, dt):
        R = _quat_matrix(self.qb)
        self._fluid_step(dt, R, self.pb.copy())
        self._rigid_step(dt)
        self.t += dt
        self.sim_steps += 1
        if self.sim_steps % 25 == 0:
            self._sanitize()

    def advance_to(self, time: float):
        limit = int(np.ceil((time - self.t) / self.dt)) + 2
        n = 0
        while self.t < time - 1e-9 and n < limit:
            self._step(self.dt)
            n += 1
        return self._make_frame(float(time))

    # ------------------------------------------------------------ next frame
    def _boundary_velocity(self):
        nb = self.nbnd
        out = np.zeros((nb, 3))
        if self.nw == 0:
            return out
        radius = 0.26
        step = 64
        for s in range(0, nb, step):
            block = self.bnd_pos[s:s + step]
            d = self.pw[None, :, :] - block[:, None, :]
            r2 = np.einsum("bij,bij->bi", d, d)
            m = r2 < radius * radius
            w = np.where(m, np.exp(-r2 / (0.5 * radius * radius)), 0.0)
            ws = w.sum(axis=1)
            num = np.einsum("bi,bic->bc", w, self.vw)
            good = ws > 1e-12
            out[s:s + step][good] = num[good] / ws[good, None]
        return out

    def _make_frame(self, time):
        R = _quat_matrix(self.qb)
        com = self.pb + R @ self.com_local
        boat_v = self.vb + np.cross(self.wb, self.pb - com)
        nw = self.nw
        pos = np.vstack([self.pw, self.pb[None, :], self.pl[None, :], self.ph[None, :]])
        vel = np.vstack([self.vw, boat_v[None, :], self.vl[None, :], self.vh[None, :]])
        mass = np.concatenate([np.full(nw, self.mw_scalar),
                               [self.m_boat, self.m_light, self.m_heavy]])
        body = np.concatenate([np.zeros(nw, dtype=int), [1, 2, 3]])
        ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (nw + 3, 1))
        ori[nw] = self.qb
        ang = np.zeros((nw + 3, 3))
        ang[nw] = self.wb
        prad = np.full(nw + 3, 0.02)
        prad[:nw] = self.particle_radius
        attrs = {
            "mass_quantity": mass.copy(),
            "phase": body.astype(float),
            "particle_radius": prad,
        }
        zeros_g = np.zeros((self.n_dom, 3))
        fields = {
            "boundary_normal": self.bnd_nrm.copy(),
            "boundary_target_velocity": np.zeros((self.nbnd, 3)),
            "boundary_velocity": self._boundary_velocity(),
            "budget_domain_weights": (body[:, None] == np.arange(self.n_dom)[None, :]).astype(float).T,
            "budget_external_force": self.force_terms.copy(),
            "budget_cumulative_external_impulse": self.impulse_cum.copy(),
            "budget_cumulative_inflow_momentum": zeros_g.copy(),
            "budget_cumulative_outflow_momentum": zeros_g.copy(),
            "cumulative_mass_source": np.zeros(self.n_dom),
            "interaction_force": self.interaction.copy(),
        }
        scalars = {
            "reference_velocity": float(np.sqrt(GRAVITY * max(self.level0, 1e-6))),
            "reference_momentum": float(0.5 * (self.m_boat + self.m_heavy)
                                        * np.sqrt(GRAVITY * max(self.level0, 1e-6))),
        }
        from worldsolver.contracts import StateFrame
        frame = StateFrame(time, pos, vel, orientation=ori, angular_velocity=ang,
                           mass=mass, body_id=body, attributes=attrs, fields=fields,
                           scalars=scalars)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        self.diag.update({
            "steps": self.sim_steps,
            "max_speed": float(self._max_speed),
            "max_penetration": float(self._max_pen),
            "flood_volume": float(self.flood_volume),
            "water_level": float(self.level),
            "boat_position": [float(v) for v in self.pb],
            "light_position": [float(v) for v in self.pl],
            "heavy_position": [float(v) for v in self.ph],
        })
        return self.diag
