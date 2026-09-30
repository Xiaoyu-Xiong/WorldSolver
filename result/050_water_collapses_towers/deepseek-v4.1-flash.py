"""Rigid-fluid coupled solver for the "Water Collapses Towers" scene.

Physical model
--------------
* Liquid: incompressible position-based fluid (PBF, Macklin & Mueller) on the
  particle sampling supplied by the scene.  Density is measured with a Poly6
  kernel and the incompressibility constraint rho/rho0 - 1 = 0 is solved with
  Jacobi Lagrange multipliers every substep.
* Rigid bodies: full 6-DOF rigid body dynamics (semi-implicit Euler with
  quaternion integration) using the frozen cylinder / box geometry and the
  frozen per-material densities.  Contacts against the basin floor and walls
  and between bodies use penalty (spring-damper) sphere-chain contacts.
* Coupling: fluid particles are projected out of the rigid signed-distance
  fields with inelastic normal velocity matching, and the reaction impulse is
  applied back onto the body (Newton's third law), so momentum is exchanged in
  both directions.  Buoyancy and quadratic drag are integrated from a smoothed
  density / velocity field sampled at body quadrature points, and the equal and
  opposite reaction is scattered back onto the fluid.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_SPACING = 0.16
_H = 0.30
_RHO0 = 1000.0
_HALF_HEIGHT = 0.925
_RADIUS = 0.115
_N_CHAIN = 9
_MU = 0.45
_K_CONTACT = 2.0e6
_K_FLUID = 1.2e6
_GRID_CELL = 0.16
_N_ANG = 8
_N_Z = 4
_RADII = (0.45, 0.85)


def _poly6(r2, h):
    h2 = h * h
    c = 315.0 / (64.0 * np.pi * h ** 9)
    d = h2 - r2
    return np.where(d > 0.0, c * np.maximum(d, 0.0) ** 3, 0.0)


def _spiky_grad(r, h):
    c = -45.0 / (np.pi * h ** 6)
    d = h - r
    return np.where(d > 0.0, c * np.maximum(d, 0.0) ** 2, 0.0)


def _quat_matrices(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[:, 0, 1] = 2.0 * (xy - wz)
    m[:, 0, 2] = 2.0 * (xz + wy)
    m[:, 1, 0] = 2.0 * (xy + wz)
    m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[:, 1, 2] = 2.0 * (yz - wx)
    m[:, 2, 0] = 2.0 * (xz - wy)
    m[:, 2, 1] = 2.0 * (yz + wx)
    m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def _quat_integrate(q, omega, dt):
    wq = np.concatenate([omega, np.zeros((len(omega), 1))], axis=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ox, oy, oz, ow = wq[:, 0], wq[:, 1], wq[:, 2], wq[:, 3]
    dq = 0.5 * np.stack(
        [
            ow * x + ox * w + oy * z - oz * y,
            ow * y - ox * z + oy * w + oz * x,
            ow * z + ox * y - oy * x + oz * w,
            ow * w - ox * x - oy * y - oz * z,
        ],
        axis=1,
    )
    q = q + dq * dt
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _cross(a, b):
    return np.cross(a, b)


class BenchmarkSolver:
    """Deterministic rigid-fluid coupled simulator."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        cfg = scene.config
        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self._duration = float(cfg["duration"])
        self._sample_dt = float(cfg["sample_dt"])
        self._dt = 0.005
        self._steps = 0
        self._time = 0.0
        self._h = _H
        self._rho0 = _RHO0

        frame0 = scene.initial_state()
        position = np.array(frame0.position, dtype=float)
        velocity = np.array(frame0.velocity, dtype=float)
        mass = np.array(frame0.mass, dtype=float)
        groups = np.asarray(frame0.attributes["scene_group"]).astype(np.int64)
        body_id = np.asarray(frame0.body_id).astype(np.int64)
        water = groups == 10

        # ---- liquid particles -------------------------------------------------
        self._wx = position[water].copy()
        self._wv = velocity[water].copy()
        self._wm = mass[water].copy()
        self._nw = int(water.sum())
        self._wgroup = np.full(self._nw, 10, dtype=np.int64)
        # calibrate the kernel reference density on the frozen initial packing so
        # the constraint starts from equilibrium instead of a spurious offset
        if self._nw > 1:
            pairs0 = cKDTree(self._wx).query_pairs(_H, output_type="ndarray")
            d0 = self._wx[pairs0[:, 0]] - self._wx[pairs0[:, 1]]
            r20 = np.einsum("ij,ij->i", d0, d0)
            w0 = _poly6(r20, _H)
            dens0 = self._wm * float(_poly6(np.array([0.0]), _H)[0]) \
                + np.bincount(pairs0[:, 0], weights=self._wm[pairs0[:, 1]] * w0,
                              minlength=self._nw) \
                + np.bincount(pairs0[:, 1], weights=self._wm[pairs0[:, 0]] * w0,
                              minlength=self._nw)
            self._rho_ref = float(np.median(dens0))
        else:
            self._rho_ref = _RHO0
        self._rho_ref = float(np.clip(self._rho_ref, 0.5 * _RHO0, 2.0 * _RHO0))

        # ---- rigid bodies -----------------------------------------------------
        idx = np.nonzero(~water)[0]
        order = np.argsort(body_id[idx], kind="stable")
        idx = idx[order]
        self._bx = position[idx].copy()
        self._bv = velocity[idx].copy()
        self._bm = mass[idx].copy()
        self._bgroup = groups[idx].copy()
        self._bid = body_id[idx].copy()
        nb = len(idx)
        self._nb = nb
        self._bq = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (nb, 1))
        self._bw = np.zeros((nb, 3))
        is_cube = (self._bgroup % 10) == 1
        self._is_cube = is_cube

        vol = np.where(
            is_cube,
            (2.0 * _RADIUS) ** 2 * (2.0 * _HALF_HEIGHT),
            np.pi * _RADIUS ** 2 * (2.0 * _HALF_HEIGHT),
        )
        self._bvolume = vol
        self._binv_mass = 1.0 / self._bm
        # diagonal inertia in the body frame
        inertia = np.empty((nb, 3))
        cyl = ~is_cube
        hx = hy = _RADIUS
        hz = _HALF_HEIGHT
        inertia[cyl, 0] = self._bm[cyl] * (3.0 * _RADIUS ** 2 + (2.0 * hz) ** 2) / 12.0
        inertia[cyl, 1] = inertia[cyl, 0]
        inertia[cyl, 2] = 0.5 * self._bm[cyl] * _RADIUS ** 2
        inertia[~cyl, 0] = self._bm[~cyl] * (hy ** 2 + hz ** 2) / 3.0
        inertia[~cyl, 1] = self._bm[~cyl] * (hx ** 2 + hz ** 2) / 3.0
        inertia[~cyl, 2] = self._bm[~cyl] * (hx ** 2 + hy ** 2) / 3.0
        self._inertia = inertia
        self._binv_inertia = 1.0 / inertia

        # sphere chains used for body-body and body-wall contacts
        zloc = np.linspace(-(_HALF_HEIGHT - _RADIUS), _HALF_HEIGHT - _RADIUS, _N_CHAIN)
        self._chain_local = np.column_stack([np.zeros(_N_CHAIN), np.zeros(_N_CHAIN), zloc])
        self._chain_r = _RADIUS

        # body quadrature points (local, unit radial scale filled per body)
        ang = np.arange(_N_ANG) * (2.0 * np.pi / _N_ANG)
        zlev = np.linspace(-0.75, 0.75, _N_Z) * _HALF_HEIGHT
        pts = []
        for zz in zlev:
            for rad in _RADII:
                for aa in ang:
                    pts.append([rad * np.cos(aa), rad * np.sin(aa), zz])
        self._q_local = np.asarray(pts, dtype=float)
        self._q_count = len(self._q_local)
        rad_scale = np.where(is_cube, 0.80 * _RADIUS, _RADIUS)
        self._q_world_local = self._q_local.copy()
        self._q_world_local[:, 0] *= rad_scale
        self._q_world_local[:, 1] *= rad_scale
        self._q_volume = vol / float(self._q_count)

        # ---- fixed diagnostic grid (also used for the coupling density field) --
        lo = np.array([-4.0, -2.0, 0.0])
        hi = np.array([4.0, 2.0, 2.2])
        self._dom_lo = lo
        self._dom_hi = hi
        self._cell = _GRID_CELL
        self._hi_grid = np.array([hi[0], hi[1], 3.0])
        self._hi_phys = np.array([hi[0], hi[1], 3.5])
        dims = np.maximum(np.ceil((self._hi_grid - lo) / self._cell).astype(int), 1)
        self._dims = dims
        self._ncell = int(dims[0] * dims[1] * dims[2])

        self._bforce = np.zeros((nb, 3))
        self._btorque = np.zeros((nb, 3))
        self._group_force = {10: np.zeros(3), 20: np.zeros(3), 21: np.zeros(3),
                             30: np.zeros(3), 31: np.zeros(3)}
        self._interaction = np.zeros((5, 3))
        self._density_grid = np.zeros(tuple(dims))
        self._vel_grid = np.zeros((3,) + tuple(dims))
        self._mass_grid_cell = 0.0

    # ------------------------------------------------------------------ utils
    def _rotations(self):
        return _quat_matrices(self._bq)

    def _chain_world(self, rot):
        return self._bx[:, None, :] + np.einsum("bij,kj->bik", rot, self._chain_local)

    def _point_velocity(self, rot, points):
        """Velocity of body-fixed points (nb, npts, 3)."""
        rel = points - self._bx[:, None, :]
        return self._bv[:, None, :] + np.cross(self._bw[:, None, :], rel)

    # ----------------------------------------------------------- density grid
    def _build_grid(self):
        dims = self._dims
        lo = self._dom_lo
        cell = self._cell
        idx = np.floor((self._wx - lo) / cell).astype(np.int64)
        np.clip(idx, 0, dims - 1, out=idx)
        flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
        ncell = self._ncell
        vcell = cell ** 3
        mcell = np.bincount(flat, weights=self._wm, minlength=ncell).reshape(dims)
        msm = uniform_filter(mcell, size=3, mode="constant") / vcell
        self._mass_grid = mcell
        self._density_grid = msm
        for d in range(3):
            mom = np.bincount(flat, weights=self._wm * self._wv[:, d],
                              minlength=ncell).reshape(dims)
            mom = uniform_filter(mom, size=3, mode="constant") / vcell
            self._vel_grid[d] = mom / np.maximum(msm, 1e-9)

    def _sample_grid(self, points):
        """Trilinear density [kg/m3] and velocity [m/s] at world points."""
        dims = self._dims
        g = (np.asarray(points, dtype=float) - self._dom_lo) / self._cell - 0.5
        i0 = np.floor(g).astype(np.int64)
        np.clip(i0, 0, dims - 2, out=i0)
        f = np.clip(g - i0, 0.0, 1.0)
        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

        def lerp(arr):
            c00 = arr[ix, iy, iz] * (1 - fx) + arr[ix + 1, iy, iz] * fx
            c10 = arr[ix, iy + 1, iz] * (1 - fx) + arr[ix + 1, iy + 1, iz] * fx
            c01 = arr[ix, iy, iz + 1] * (1 - fx) + arr[ix + 1, iy, iz + 1] * fx
            c11 = arr[ix, iy + 1, iz + 1] * (1 - fx) + arr[ix + 1, iy + 1, iz + 1] * fx
            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy
            return c0 * (1 - fz) + c1 * fz

        dens = lerp(self._density_grid)
        vel = np.stack([lerp(self._vel_grid[d]) for d in range(3)], axis=1)
        return dens, vel

    # --------------------------------------------------------- fluid -> rigid
    def _fluid_coupling(self, dt):
        rot = self._rotations()
        qw = self._bx[:, None, :] + np.einsum("bij,kj->bik", rot, self._q_world_local)
        flat_q = qw.reshape(-1, 3)
        dens, u = self._sample_grid(flat_q)
        dens = dens.reshape(self._nb, self._q_count)
        u = u.reshape(self._nb, self._q_count, 3)
        sub = np.clip((dens - 0.30 * _RHO0) / (0.55 * _RHO0), 0.0, 1.0)

        qv = self._point_velocity(rot, qw)
        urel = qv - u
        speed = np.linalg.norm(urel, axis=2)
        vol_q = self._q_volume[:, None]
        area_q = 2.0 * (self._bvolume ** (2.0 / 3.0))[:, None] / float(self._q_count)
        sub3 = sub[:, :, None]
        buoy = -_RHO0 * vol_q[:, :, None] * sub3 * self._gravity
        drag = -0.5 * _RHO0 * 1.1 * area_q[:, :, None] * sub3 * speed[:, :, None] * urel
        fq = buoy + drag
        self._bforce += fq.sum(axis=1)
        self._btorque += np.cross(qw - self._bx[:, None, :], fq).sum(axis=1)

        # equal and opposite reaction scattered onto nearby liquid particles
        if self._nw:
            for b in range(self._nb):
                rel = self._wx - self._bx[b]
                r2 = np.einsum("ij,ij->i", rel, rel)
                m = r2 < self._h * self._h
                if not np.any(m):
                    continue
                w = _poly6(r2[m], self._h)
                ws = w.sum()
                if ws <= 0.0:
                    continue
                imp = fq[b].sum(axis=0) * (-dt)
                self._wv[m] += (w / ws)[:, None] * imp[None, :] / self._wm[m][:, None]
                self._group_force[10] -= fq[b].sum(axis=0)

    # ----------------------------------------------------------- geometry SDF
    def _body_sdf(self, b, lx, ly, lz):
        """Signed distance (positive outside) and local outward normal."""
        if self._is_cube[b]:
            ex = _RADIUS - np.abs(lx)
            ey = _RADIUS - np.abs(ly)
            ez = _HALF_HEIGHT - np.abs(lz)
            e = np.stack([ex, ey, ez], axis=1)
            emax = e.max(axis=1)
            outside = np.linalg.norm(np.maximum(e, 0.0), axis=1)
            sd = outside + np.minimum(emax, 0.0)
            k = e.argmax(axis=1)
            n = np.zeros_like(e)
            local = np.stack([lx, ly, lz], axis=1)
            sgn = np.where(local[np.arange(len(k)), k] >= 0.0, 1.0, -1.0)
            n[np.arange(len(k)), k] = sgn
            n[np.linalg.norm(n, axis=1) < 0.5] = np.array([1.0, 0.0, 0.0])
            return sd, n
        rr = np.sqrt(lx * lx + ly * ly)
        s_side = _RADIUS - rr
        s_cap = _HALF_HEIGHT - np.abs(lz)
        outside = np.sqrt(np.maximum(s_side, 0.0) ** 2 + np.maximum(s_cap, 0.0) ** 2)
        sd = outside + np.minimum(np.minimum(s_side, s_cap), 0.0)
        use_side = s_side <= s_cap
        safe = np.maximum(rr, 1e-9)
        nx = np.where(use_side, lx / safe, 0.0)
        ny = np.where(use_side, ly / safe, 0.0)
        nz = np.where(use_side, 0.0, np.where(lz >= 0.0, 1.0, -1.0))
        n = np.stack([nx, ny, nz], axis=1)
        n[np.linalg.norm(n, axis=1) < 0.5] = np.array([1.0, 0.0, 0.0])
        return sd, n

    def _sdf_all(self, pos):
        """Per-body signed distance and world normal for every particle."""
        rot = self._rotations()
        out_sd = []
        out_n = []
        for b in range(self._nb):
            local = (pos - self._bx[b]) @ rot[b]
            sd, n = self._body_sdf(b, local[:, 0], local[:, 1], local[:, 2])
            out_sd.append(sd)
            out_n.append(n @ rot[b].T)
        return out_sd, out_n

    def _sdf_project(self, pos, rot):
        for b in range(self._nb):
            local = (pos - self._bx[b]) @ rot[b]
            sd, n = self._body_sdf(b, local[:, 0], local[:, 1], local[:, 2])
            mask = sd < 0.0
            if np.any(mask):
                pos[mask] += (-sd[mask])[:, None] * (n[mask] @ rot[b].T)
        np.clip(pos, self._dom_lo, self._hi_phys, out=pos)
        return pos

    # ------------------------------------------------------------- fluid step
    def _fluid_step(self, dt):
        m = self._wm
        h = self._h
        rho0 = self._rho_ref
        W0 = float(_poly6(np.array([0.0]), h)[0])
        wq = float(_poly6(np.array([(0.2 * h) ** 2]), h)[0])
        pos = self._wx
        vel = self._wv + self._gravity * dt
        pred = pos + vel * dt
        rot = self._rotations()

        tree = cKDTree(pred)
        pairs = tree.query_pairs(h, output_type="ndarray")
        ii = pairs[:, 0]
        jj = pairs[:, 1]
        n = self._nw

        for _ in range(3):
            d = pred[ii] - pred[jj]
            r2 = np.einsum("ij,ij->i", d, d)
            r = np.sqrt(r2)
            w = _poly6(r2, h)
            gc = _spiky_grad(r, h)
            inv_r = 1.0 / np.maximum(r, 1e-9)
            grad = (gc * inv_r)[:, None] * d
            rho = m * W0 + np.bincount(ii, weights=m[jj] * w, minlength=n) \
                + np.bincount(jj, weights=m[ii] * w, minlength=n)
            cval = rho / rho0 - 1.0
            g2 = np.bincount(ii, weights=np.einsum("ij,ij->i", grad, grad), minlength=n) \
                + np.bincount(jj, weights=np.einsum("ij,ij->i", grad, grad), minlength=n)
            lam = -cval / (g2 / (rho0 * rho0) + 1e-6)
            scorr = -2.0e-4 * (w / max(wq, 1e-12)) ** 4
            coef = (lam[ii] + lam[jj] + scorr) / rho0
            for dc in range(3):
                pred[:, dc] += np.bincount(ii, weights=coef * grad[:, dc], minlength=n) \
                    - np.bincount(jj, weights=coef * grad[:, dc], minlength=n)
            pred = self._sdf_project(pred, rot)
            np.clip(pred, self._dom_lo, self._hi_phys, out=pred)

        newv = (pred - pos) / dt

        # rigid particle-surface velocity response (inelastic + friction)
        for b in range(self._nb):
            local = (pred - self._bx[b]) @ rot[b]
            sd, nl = self._body_sdf(b, local[:, 0], local[:, 1], local[:, 2])
            mask = sd < 0.5 * _RADIUS
            if not np.any(mask):
                continue
            p = pred[mask]
            nrm = nl[mask] @ rot[b].T
            vpb = self._bv[b] + np.cross(self._bw[b], p - self._bx[b])
            vrel = newv[mask] - vpb
            vn = np.einsum("ij,ij->i", vrel, nrm)
            hit = vn < 0.0
            if not np.any(hit):
                continue
            nrm = nrm[hit]
            vrel = vrel[hit]
            vn = vn[hit]
            vt = vrel - vn[:, None] * nrm
            vtn = np.linalg.norm(vt, axis=1)
            fscale = np.minimum(1.0, _MU * np.abs(vn) / np.maximum(vtn, 1e-9))
            dv = -vn[:, None] * nrm - fscale[:, None] * vt
            ph = p[hit]
            mh = m[mask][hit]
            newv_mask = np.nonzero(mask)[0][hit]
            newv[newv_mask] += dv
            impulse = mh[:, None] * dv
            self._bforce[b] -= impulse.sum(axis=0) / dt
            self._btorque[b] -= np.cross(ph - self._bx[b], impulse).sum(axis=0) / dt
            self._group_force[10] += impulse.sum(axis=0) / dt

        # basin boundary response
        before = newv.copy()
        upper = np.array([self._dom_hi[0], self._dom_hi[1], self._hi_phys[2]])
        for dc in range(3):
            low = pred[:, dc] <= self._dom_lo[dc] + 1e-9
            high = pred[:, dc] >= upper[dc] - 1e-9
            v = newv[:, dc]
            v = np.where(low & (v < 0.0), 0.0, v)
            v = np.where(high & (v > 0.0), 0.0, v)
            newv[:, dc] = v
        self._group_force[10] += (m[:, None] * (newv - before)).sum(axis=0) / dt

        # XSPH velocity smoothing
        d = pred[ii] - pred[jj]
        r2 = np.einsum("ij,ij->i", d, d)
        w = _poly6(r2, h)
        for dc in range(3):
            num = np.bincount(ii, weights=(m[jj] * w) * (newv[jj, dc] - newv[ii, dc]),
                              minlength=n) \
                + np.bincount(jj, weights=(m[ii] * w) * (newv[ii, dc] - newv[jj, dc]),
                              minlength=n)
            newv[:, dc] += 0.05 * num / rho0

        np.clip(newv, -60.0, 60.0, out=newv)
        self._wx = pred
        self._wv = newv

    # ------------------------------------------------------------ rigid step
    def _contact_force(self, normal, pen, vrel, meff):
        kn = _K_CONTACT
        damp = 0.8 * np.sqrt(kn * meff)
        vn = np.einsum("ij,ij->i", vrel, normal)
        fn = np.maximum(kn * pen - damp * vn, 0.0)
        vt = vrel - vn[:, None] * normal
        vtn = np.linalg.norm(vt, axis=1)
        ft = np.minimum(_MU * fn, damp * vtn)
        return fn[:, None] * normal - (ft / np.maximum(vtn, 1e-9))[:, None] * vt

    def _rigid_contacts(self):
        rot = self._rotations()
        cw = self._chain_world(rot)
        vcw = self._point_velocity(rot, cw)
        nb, nc = self._nb, _N_CHAIN
        r = self._chain_r
        bm = self._bm

        diff = cw[:, None, :, None, :] - cw[None, :, :, None, :]
        dist = np.sqrt(np.einsum("...k,...k->...", diff, diff))
        pen = 2.0 * r - dist
        upper = np.triu(np.ones((nb, nb), dtype=bool), 1)
        active = (pen > 0.0) & upper[:, :, None, None]
        iu, ju, ci, cj = np.nonzero(active)
        if len(iu):
            nrm = diff[iu, ju, ci, cj] / np.maximum(dist[iu, ju, ci, cj], 1e-9)[:, None]
            vrel = vcw[iu, ci] - vcw[ju, cj]
            meff = 1.0 / (1.0 / bm[iu] + 1.0 / bm[ju])
            f = self._contact_force(nrm, pen[iu, ju, ci, cj], vrel, meff)
            p = 0.5 * (cw[iu, ci] + cw[ju, cj])
            np.add.at(self._bforce, iu, f)
            np.add.at(self._bforce, ju, -f)
            np.add.at(self._btorque, iu, np.cross(p - self._bx[iu], f))
            np.add.at(self._btorque, ju, np.cross(p - self._bx[ju], -f))

        planes = [
            (np.array([0.0, 0.0, 1.0]), cw[..., 2] - self._dom_lo[2]),
            (np.array([0.0, -1.0, 0.0]), self._dom_hi[1] - cw[..., 1]),
            (np.array([0.0, 1.0, 0.0]), cw[..., 1] - self._dom_lo[1]),
            (np.array([-1.0, 0.0, 0.0]), self._dom_hi[0] - cw[..., 0]),
            (np.array([1.0, 0.0, 0.0]), cw[..., 0] - self._dom_lo[0]),
        ]
        for nrm3, gap in planes:
            pen = r - gap
            mask = pen > 0.0
            if not np.any(mask):
                continue
            bi, si = np.nonzero(mask)
            nrm = np.tile(nrm3, (len(bi), 1))
            f = self._contact_force(nrm, pen[bi, si], vcw[bi, si], bm[bi])
            p = cw[bi, si]
            np.add.at(self._bforce, bi, f)
            np.add.at(self._btorque, bi, np.cross(p - self._bx[bi], f))

    def _rigid_integrate(self, dt):
        self._bforce += self._bm[:, None] * self._gravity
        self._bv += self._bforce * self._binv_mass[:, None] * dt
        rot = self._rotations()
        wbody = np.einsum("bij,bj->bi", rot, self._bw)
        tau_body = np.einsum("bij,bj->bi", rot, self._btorque)
        dwbody = (tau_body - np.cross(wbody, wbody * self._inertia)) * self._binv_inertia
        self._bw += np.einsum("bij,bj->bi", rot, dwbody) * dt
        self._bx = self._bx + self._bv * dt
        np.clip(self._bv, -60.0, 60.0, out=self._bv)
        np.clip(self._bw, -60.0, 60.0, out=self._bw)
        np.clip(self._bx, self._dom_lo + 0.01, self._hi_phys - 0.01, out=self._bx)
        self._bq = _quat_integrate(self._bq, self._bw, dt)

    # ---------------------------------------------------------------- drivers
    def _step(self, dt):
        self._bforce[:] = 0.0
        self._btorque[:] = 0.0
        for key in self._group_force:
            self._group_force[key][:] = 0.0
        self._build_grid()
        self._fluid_coupling(dt)
        self._fluid_step(dt)
        self._rigid_contacts()
        gravity_force = self._bm[:, None] * self._gravity
        for b in range(self._nb):
            g = int(self._bgroup[b])
            self._group_force[g] += self._bforce[b] - gravity_force[b]
        self._rigid_integrate(dt)
        self._interaction[:, :] = np.stack(
            [self._group_force[10], self._group_force[20], self._group_force[21],
             self._group_force[30], self._group_force[31]], axis=0)
        self._steps += 1
        self._time = self._steps * dt

    # ------------------------------------------------------------- diagnostics
    def _contact_distances(self):
        rot = self._rotations()
        cw = self._chain_world(rot)
        r = self._chain_r
        nb = self._nb
        floor = cw[:, :, 2].min(axis=1) - r
        yplus = (self._dom_hi[1] - cw[:, :, 1].max(axis=1)) - r
        yminus = (cw[:, :, 1].min(axis=1) - self._dom_lo[1]) - r
        xplus = (self._dom_hi[0] - cw[:, :, 0].max(axis=1)) - r
        xminus = (cw[:, :, 0].min(axis=1) - self._dom_lo[0]) - r
        walls = np.stack([floor, yplus, yminus, xplus, xminus], axis=1)
        pair_vals = []
        for start in (0, nb // 2):
            for i in range(start, start + nb // 2):
                for j in range(i + 1, start + nb // 2):
                    d = cw[i][:, None, :] - cw[j][None, :, :]
                    dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
                    pair_vals.append(float(dist.min() - 2.0 * r))
        fluid_vals = []
        if self._nw:
            sds, _ = self._sdf_all(self._wx)
            fluid_vals = [float(sd.min()) for sd in sds]
        else:
            fluid_vals = [0.0] * nb
        return np.concatenate(
            [walls.reshape(-1), np.asarray(pair_vals), np.asarray(fluid_vals)]
        ).astype(float)

    def _frame(self, time):
        nb = self._nb
        nw = self._nw
        n = nw + nb
        position = np.vstack((self._wx, self._bx))
        velocity = np.vstack((self._wv, self._bv))
        orientation = np.vstack(
            (np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (nw, 1)), self._bq)
        )
        angular_velocity = np.vstack((np.zeros((nw, 3)), self._bw))
        mass = np.concatenate((self._wm, self._bm))
        groups = np.concatenate((self._wgroup, self._bgroup)).astype(np.int64)
        body_id = np.concatenate(
            (np.zeros(nw, dtype=np.int64), self._bid.astype(np.int64))
        )
        mass_quantity = np.zeros((n, 3))
        mass_quantity[:nw, 0] = self._wm
        cyl = np.zeros(nb, dtype=bool)
        cubes = np.zeros(nb, dtype=bool)
        for b in range(nb):
            if self._is_cube[b]:
                cubes[b] = True
            else:
                cyl[b] = True
        mass_quantity[nw:, 1] = np.where(cyl, self._bm, 0.0)
        mass_quantity[nw:, 2] = np.where(cubes, self._bm, 0.0)
        density = np.concatenate(
            (np.full(nw, _RHO0), np.where(cubes, 1500.0, 600.0))
        )
        external = mass[:, None] * self._gravity
        fields = {
            "contact_signed_distance": self._contact_distances(),
            "cumulative_mass_source": np.zeros(3),
            "external_force": external,
            "interaction_force": self._interaction.copy(),
            "density_field": np.ascontiguousarray(self._density_grid[::2, ::2, ::2]),
        }
        scalars = {
            "contact_length_scale": _SPACING,
            "water_rest_density": _RHO0,
            "liquid_phase_id": 10.0,
            "substep_dt": self._dt,
            "particle_count": float(nw),
            "body_count": float(nb),
            "time": float(time),
        }
        attributes = {
            "scene_group": groups,
            "mass_quantity": mass_quantity,
            "density": density,
            "phase_id": groups,
        }
        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float):
        target = int(round(float(time) / self._dt))
        if target < self._steps:
            target = self._steps
        while self._steps < target:
            self._step(self._dt)
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbf_rigid_coupled",
            "substep_dt": self._dt,
            "steps": self._steps,
            "simulated_time": self._time,
            "particles": self._nw,
            "rigid_bodies": self._nb,
            "water_rest_density": _RHO0,
        }
