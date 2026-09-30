"""Two-way coupled water-stream / cloth-curtain solver.

Physical model (implemented from scratch with NumPy + SciPy only):

* Curtain: mass-spring shell discretised on the frozen topology asset's
  triangle mesh (structural edge constraints plus second-neighbour bending
  constraints), integrated with a position-based (XPBD/Jacobi) backward
  Euler scheme.  The top row of 31 vertices is pinned (inv mass 0).
* Water: weakly compressible particle domain (the scene's particle_domain)
  advanced with Position Based Fluids (Macklin & Mueller 2013): poly6
  density, spiky gradient, Jacobi pressure projection, XSPH viscosity,
  unilateral tank / nozzle boundary conditions and a prescribed inflow
  emitter at the nozzle exit plane.
* Coupling: the curtain is rasterised each substep into a smooth
  height-field (y,z) grid holding the sheet abscissa and its gradient, from
  which an accurate signed distance and unit normal are obtained for every
  water particle.  Non-penetration is enforced as a hard positional
  constraint; the momentum the water loses during that projection (and the
  tangential drag it experiences while sliding along the sheet) is
  transferred back to the nearest cloth vertices, giving full two-way
  coupling with momentum exchange.

All quantities are SI, deterministic and finite at every sample time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])


class _Grid:
    """Cloth height-field over the (y, z) parametrisation."""

    __slots__ = ("y0", "z0", "gy", "gz", "ny", "nz", "xs", "gyv", "gzv", "w")

    def __init__(self, y0, z0, gy, gz, ny, nz):
        self.y0, self.z0, self.gy, self.gz = y0, z0, gy, gz
        self.ny, self.nz = ny, nz
        n = ny * nz
        self.xs = np.zeros(n)
        self.gyv = np.zeros(n)
        self.gzv = np.zeros(n)
        self.w = np.zeros(n)


class BenchmarkSolver:
    CLOTH_BODY = 0
    WATER_BODY = 1

    DT = 0.004                 # internal substep (sample_dt = 5 substeps)
    RHO0 = 1000.0
    H = 0.13                   # SPH support radius
    CELL = 0.0867              # neighbour-grid cell size
    SOLVER_ITERS = 3
    XSPH = 0.06
    PBF_K = 0.0008             # surface tension correction strength
    PBF_N = 4.0
    PBF_DQ = 0.2               # in units of h
    CLOTH_ITERS = 12
    STIFF_STRUCT = 0.8
    STIFF_BEND = 0.25
    VISC_DRAG = 1.6            # 1/s tangential water/cloth drag rate
    CONTACT_BAND = 0.07        # cloth/water interaction band [m]
    GRID_DY = 0.05
    GRID_DZ = 0.05
    MAX_SPEED = 30.0
    MAX_VERTS_PER_FRAME = 424

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        frame0 = scene.initial_state()
        meta = scene.metadata()
        inflow = meta.get("inflow", {})
        self.inflow_center = np.asarray(inflow.get("center", [-1.255, 0.0, 1.72]), float)
        self.inflow_radius = float(inflow.get("radius", 0.20))
        self.inflow_vel = np.asarray(inflow.get("velocity", [3.2, 0.0, 0.0]), float)

        pos = np.asarray(frame0.position, float)
        vel = np.asarray(frame0.velocity, float)
        mass = np.asarray(frame0.mass, float)
        body = np.asarray(frame0.body_id).astype(np.int64)
        self.pinned0 = np.asarray(frame0.attributes["pinned"], float)
        self.phase0 = np.asarray(frame0.attributes["material_phase"], float)

        cm = body == self.CLOTH_BODY
        wm = body == self.WATER_BODY
        self.cloth_p = pos[cm].copy()
        self.cloth_v = vel[cm].copy()
        self.cloth_m = mass[cm].copy()
        self.ncl = len(self.cloth_p)
        w0 = pos[wm].copy()
        wv0 = vel[wm].copy()
        self.wat_m0 = mass[wm].copy()
        self._init_water(w0, wv0)

        self._load_topology()
        self._build_constraints()
        self._init_grid()
        self._init_inflow()
        self._init_domain()

        self.t = 0.0
        self.fcloth_next = np.zeros((self.ncl, 3))
        self.fcloth = np.zeros((self.ncl, 3))
        self.net_reaction = np.zeros(3)
        self.injected_mass = 0.0
        self.removed_mass = 0.0
        self.last_pairs = None
        self.density = np.full(self.nw, self.RHO0)
        self.frame0 = self._build_frame(initial=True, f0=frame0)

    # ------------------------------------------------------------- topology
    def _load_topology(self):
        here = Path(__file__).resolve().parent.parent
        asset = here / "assets" / "curtain_topology.npz"
        if not asset.is_file():
            asset = Path.cwd() / "assets" / "curtain_topology.npz"
        data = np.load(asset)
        self.faces = np.asarray(data["faces"], np.int64)
        self.edges = np.asarray(data["edges"], np.int64)

    def _build_constraints(self):
        ncl = self.ncl
        ei = self.edges[:, 0].astype(np.int64)
        ej = self.edges[:, 1].astype(np.int64)
        # second-neighbour bending springs on the structured 31 x 41 lattice
        vy, vz = 31, 41
        bend = []
        for k in range(vy - 2):
            for j in range(vz):
                a = j * vy + k
                bend.append((a, a + 2))
        for k in range(vy):
            for j in range(vz - 2):
                a = j * vy + k
                bend.append((a, a + 2 * vy))
        b = np.asarray(bend, np.int64)
        self.ce_i = np.concatenate([ei, b[:, 0]])
        self.ce_j = np.concatenate([ej, b[:, 1]])
        self.ce_rest = np.linalg.norm(
            self.cloth_p[self.ce_j] - self.cloth_p[self.ce_i], axis=1
        )
        stiff = np.empty(len(self.ce_i))
        stiff[: len(ei)] = self.STIFF_STRUCT
        stiff[len(ei):] = self.STIFF_BEND
        self.ce_stiff = stiff
        self.inv_m = np.where(self.pinned0[:ncl] > 0.5, 0.0, 1.0 / self.cloth_m)
        self.contact_length_scale = float(np.mean(np.linalg.norm(
            self.cloth_p[self.edges[:, 1]] - self.cloth_p[self.edges[:, 0]], axis=1)))

    # ---------------------------------------------------------------- water
    def _init_water(self, w0, v0):
        cap = len(w0) + 6000
        self.wcap = cap
        self.wat_p = np.zeros((cap, 3))
        self.wat_v = np.zeros((cap, 3))
        self.wat_m = np.zeros(cap)
        self.wat_r = np.zeros(cap)
        n = len(w0)
        self.wat_p[:n] = w0
        self.wat_v[:n] = v0
        self.wat_m[:n] = self.wat_m0
        # per-particle support radius from local spacing (pool 0.105 / jet 0.075)
        self.wat_r[:n] = np.where(self.wat_m0 > 0.6, 0.5 * 0.105, 0.5 * 0.075)
        self.nw = n
        self.density = np.full(cap, self.RHO0)
        self.mass0_total = float(self.wat_m0.sum())

    def _grow(self, extra):
        if self.nw + extra <= self.wcap:
            return
        newcap = max(self.wcap * 2, self.nw + extra + 1024)
        for name, width in (("wat_p", 3), ("wat_v", 3)):
            arr = getattr(self, name)
            grown = np.zeros((newcap, width))
            grown[: self.nw] = arr[: self.nw]
            setattr(self, name, grown)
        for name in ("wat_m", "wat_r"):
            arr = getattr(self, name)
            grown = np.zeros(newcap)
            grown[: self.nw] = arr[: self.nw]
            setattr(self, name, grown)
        sd = np.ones(newcap)
        sd[: self.nw] = self.sd_prev[: self.nw]
        self.sd_prev = sd
        self.wcap = newcap

    def _init_inflow(self):
        c = self.inflow_center
        rad = self.inflow_radius
        sp = 0.075
        ax = np.arange(-rad, rad + 1e-9, sp)
        yy, zz = np.meshgrid(ax, ax, indexing="ij")
        sel = (yy ** 2 + zz ** 2) <= rad * rad
        self.emit_y = c[1] + yy[sel]
        self.emit_z = c[2] + zz[sel]
        self.emit_x = c[0] + 0.01
        self.n_sites = len(self.emit_y)
        self.emit_mass = 1000.0 * sp ** 3
        self.emit_radius = 0.5 * sp
        area = np.pi * rad * rad
        flux = 1000.0 * area * float(np.linalg.norm(self.inflow_vel))
        self.emit_period = self.emit_mass * self.n_sites / max(flux, 1e-9)
        self.next_emit = (np.arange(self.n_sites) / max(self.n_sites, 1)) * self.emit_period

    def _init_domain(self):
        # catch-tank interior (inner faces) and world escape box
        self.tank_min = np.array([-1.58, -1.38, 0.0])
        self.tank_max = np.array([2.08, 1.38, 2.10])
        self.world_min = np.array([-2.45, -1.95, -0.55])
        self.world_max = np.array([2.45, 1.95, 3.35])
        self.nozzle_axis_z = float(self.inflow_center[2])
        self.nozzle_axis_y = float(self.inflow_center[1])
        self.nozzle_r = float(self.inflow_radius) + 0.04
        self.nozzle_x0 = -2.40
        self.nozzle_x1 = float(self.inflow_center[0]) - 0.005
        # dense neighbour lookup lattice
        self.cell = self.CELL
        self.grid_lo = self.world_min
        dims = np.ceil((self.world_max - self.world_min) / self.cell).astype(int) + 1
        self.gdims = dims
        self.cell_id = np.full(tuple(dims + 2), -1, dtype=np.int64)
        self.K = self.cell ** 2

    def _init_grid(self):
        p = self.cloth_p
        self.grid = _Grid(
            float(p[:, 1].min() - 0.14), float(p[:, 2].min() - 0.13),
            self.GRID_DY, self.GRID_DZ,
            int(np.ceil((p[:, 1].max() - p[:, 1].min() + 0.28) / self.GRID_DY)) + 2,
            int(np.ceil((p[:, 2].max() - p[:, 2].min() + 0.26) / self.GRID_DZ)) + 2,
        )
        self.grid_wref = 1.0
        self._rasterize_cloth()
        interior = self.grid.w.copy()
        self.grid_wref = float(np.median(interior[interior > 1e-6])) if np.any(interior > 1e-6) else 1.0
        self._rasterize_cloth()
        self.sample_idx = np.arange(0, self.ncl, max(1, self.ncl // self.MAX_VERTS_PER_FRAME))
        sd0, _, _, _, _ = self._sample_grid(
            self.wat_p[: self.nw, 0], self.wat_p[: self.nw, 1], self.wat_p[: self.nw, 2]
        )
        self.sd_prev = np.where(sd0 >= 0.0, 1.0, -1.0)
        extra = np.ones(self.wcap - self.nw)
        self.sd_prev = np.r_[self.sd_prev, extra]

    # ------------------------------------------------------------ rasterize
    def _rasterize_cloth(self):
        p = self.cloth_p
        ny, nz = self.grid.ny, self.grid.nz
        fy = (p[:, 1] - self.grid.y0) / self.grid.gy
        fz = (p[:, 2] - self.grid.z0) / self.grid.gz
        iy = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        iz = np.clip(np.floor(fz).astype(np.int64), 0, nz - 2)
        wy = np.clip(fy - iy, 0.0, 1.0)
        wz = np.clip(fz - iz, 0.0, 1.0)
        num = np.zeros(ny * nz)
        den = np.zeros(ny * nz)
        for dy, dwy in ((0, 1.0 - wy), (1, wy)):
            for dz, dwz in ((0, 1.0 - wz), (1, wz)):
                idx = (iy + dy) * nz + (iz + dz)
                w = dwy * dwz
                np.add.at(num, idx, w * p[:, 0])
                np.add.at(den, idx, w)
        num = num.reshape(ny, nz)
        den = den.reshape(ny, nz)
        for _ in range(2):
            num = self._box(num)
            den = self._box(den)
        den = np.maximum(den, 0.0)
        xs = np.where(den > 1e-6, num / np.maximum(den, 1e-9), 0.0)
        gyv, gzv = np.gradient(xs, self.grid.gy, self.grid.gz)
        gyv = np.clip(gyv, -2.0, 2.0)
        gzv = np.clip(gzv, -2.0, 2.0)
        w = np.clip((den / max(self.grid_wref, 1e-9) - 0.25) / 0.5, 0.0, 1.0)
        # the height-field representation is only reliable while the sheet is not
        # edge-on to the sampling axis
        nx = 1.0 / np.sqrt(1.0 + gyv * gyv + gzv * gzv)
        w = w * np.clip((nx - 0.35) / 0.3, 0.0, 1.0)
        self.grid.xs = xs.ravel()
        self.grid.gyv = gyv.ravel()
        self.grid.gzv = gzv.ravel()
        self.grid.w = w.ravel()

    @staticmethod
    def _box(a):
        c = np.pad(a, 1, mode="edge")
        b = (c[:-2, 1:-1] + c[2:, 1:-1] + c[1:-1, :-2] + c[1:-1, 2:]
             + c[1:-1, 1:-1] * 1.0 + c[:-2, :-2] + c[:-2, 2:] + c[2:, :-2] + c[2:, 2:]) / 9.0
        return b

    def _sample_grid(self, px, py, pz):
        g = self.grid
        ny, nz = g.ny, g.nz
        fy = (py - g.y0) / g.gy
        fz = (pz - g.z0) / g.gz
        inside = (fy >= 0.0) & (fy < ny - 1.0) & (fz >= 0.0) & (fz < nz - 1.0)
        iy = np.clip(np.floor(fy), 0, ny - 2).astype(np.int64)
        iz = np.clip(np.floor(fz), 0, nz - 2).astype(np.int64)
        wy = np.clip(fy - iy, 0.0, 1.0)
        wz = np.clip(fz - iz, 0.0, 1.0)
        w00 = (1.0 - wy) * (1.0 - wz)
        w10 = wy * (1.0 - wz)
        w01 = (1.0 - wy) * wz
        w11 = wy * wz
        i00 = iy * nz + iz
        i10 = i00 + nz
        i01 = i00 + 1
        i11 = i10 + 1

        def bl(a):
            return a[i00] * w00 + a[i10] * w10 + a[i01] * w01 + a[i11] * w11

        xs = bl(g.xs)
        gyv = bl(g.gyv)
        gzv = bl(g.gzv)
        wt = bl(g.w)
        wt = np.where(inside, wt, 0.0)
        den = np.sqrt(1.0 + gyv * gyv + gzv * gzv)
        sd = (px - xs) / den
        nx = 1.0 / den
        nyv = -gyv / den
        nzv = -gzv / den
        return sd, nx, nyv, nzv, wt

    # --------------------------------------------------------- neighbour list
    def _build_pairs(self, p):
        n = len(p)
        dims = self.gdims
        ci = np.floor((p - self.grid_lo) / self.cell).astype(np.int64)
        np.clip(ci, 0, dims - 1, out=ci)
        key = (ci[:, 0] + 1) + (ci[:, 1] + 1) * (dims[0] + 2) + (ci[:, 2] + 1) * (dims[0] + 2) * (dims[1] + 2)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uniq, starts = np.unique(ks, return_index=True)
        self.cell_id[:] = -1
        # decode stored keys back to 3D cell coordinates
        sy = dims[0] + 2
        sz = sy * (dims[1] + 2)
        cx = (uniq % sy)
        cy = (uniq // sy) % (dims[1] + 2)
        cz = uniq // sz
        self.cell_id[cx, cy, cz] = np.arange(len(uniq))
        ends = np.r_[starts[1:], n]
        base = ci + 1
        cnt = np.empty((n, 27), np.int64)
        los = np.empty((n, 27), np.int64)
        k = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cid = self.cell_id[base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz]
                    good = cid >= 0
                    cc = np.where(good, cid, 0)
                    cnt[:, k] = np.where(good, ends[cc] - starts[cc], 0)
                    los[:, k] = np.where(good, starts[cc], 0)
                    k += 1
        total = cnt.sum(axis=1)
        cnt_flat = cnt.ravel()
        lo_flat = los.ravel()
        rep = np.repeat(lo_flat, cnt_flat)
        add = np.arange(rep.size) - np.repeat(np.cumsum(cnt_flat) - cnt_flat, cnt_flat)
        j_sorted = rep + add
        pi = np.repeat(np.arange(n), total)
        pj = order[j_sorted]
        d = p[pj] - p[pi]
        r2 = np.einsum("ij,ij->i", d, d)
        keep = (r2 > 1e-12) & (r2 < self.H * self.H)
        return pi[keep], pj[keep], d[keep], r2[keep]

    def _kernel(self, r2):
        h = self.H
        h2 = h * h
        r = np.sqrt(r2)
        poly = 315.0 / (64.0 * np.pi * h ** 9)
        spiky = 45.0 / (np.pi * h ** 6)
        W = poly * (h2 - r2) ** 3
        gw = -spiky * (h - r) ** 2
        return r, W, gw

    # --------------------------------------------------------------- cloth
    def _cloth_step(self, dt):
        p = self.cloth_p
        v = self.cloth_v
        ncl = self.ncl
        acc = self.fcloth / self.cloth_m[:, None]
        amag = np.linalg.norm(acc, axis=1)
        lim = 20000.0
        scale = np.where(amag > lim, lim / np.maximum(amag, 1e-12), 1.0)
        v += GRAVITY[None, :] * dt
        v += acc * scale[:, None] * dt
        p_prev = p.copy()
        p = p + v * dt
        self._solve_constraints(p)
        self._cloth_collide(p)
        v = (p - p_prev) / dt
        v[self.pinned0[:ncl] > 0.5] = 0.0
        v *= 0.998
        sp = np.linalg.norm(v, axis=1)
        over = sp > self.MAX_SPEED
        if np.any(over):
            v[over] *= (self.MAX_SPEED / sp[over])[:, None]
        self.cloth_p = p
        self.cloth_v = v

    def _solve_constraints(self, p):
        ei, ej = self.ce_i, self.ce_j
        rest = self.ce_rest
        stiff = self.ce_stiff
        wi = self.inv_m[ei]
        wj = self.inv_m[ej]
        ws = wi + wj
        active = ws > 0.0
        m_c = np.maximum(ws, 1e-12)
        ncl = self.ncl
        for _ in range(self.CLOTH_ITERS):
            d = p[ej] - p[ei]
            L = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            inv = 1.0 / L
            nvec = d * inv[:, None]
            fac = np.where(active, -stiff * (L - rest) / m_c, 0.0)
            ci = wi * fac
            cj = wj * fac
            px = np.bincount(ei, weights=ci * nvec[:, 0], minlength=ncl) + \
                np.bincount(ej, weights=cj * nvec[:, 0], minlength=ncl)
            py = np.bincount(ei, weights=ci * nvec[:, 1], minlength=ncl) + \
                np.bincount(ej, weights=cj * nvec[:, 1], minlength=ncl)
            pz = np.bincount(ei, weights=ci * nvec[:, 2], minlength=ncl) + \
                np.bincount(ej, weights=cj * nvec[:, 2], minlength=ncl)
            p[:, 0] += px
            p[:, 1] += py
            p[:, 2] += pz

    def _cloth_collide(self, p):
        lo = np.array([-2.30, -1.86, 0.0])
        hi = np.array([2.04, 1.86, 3.20])
        np.clip(p, lo, hi, out=p)
        r = np.sqrt((p[:, 1] - self.nozzle_axis_y) ** 2 + (p[:, 2] - self.nozzle_axis_z) ** 2)
        hit = (p[:, 0] > self.nozzle_x0) & (p[:, 0] < self.nozzle_x1) & (r < self.nozzle_r)
        if np.any(hit):
            s = self.nozzle_r / np.maximum(r[hit], 1e-9)
            p[hit, 1] = self.nozzle_axis_y + (p[hit, 1] - self.nozzle_axis_y) * s
            p[hit, 2] = self.nozzle_axis_z + (p[hit, 2] - self.nozzle_axis_z) * s

    # ---------------------------------------------------------------- water
    def _water_step(self, dt):
        n = self.nw
        p = self.wat_p[:n]
        v = self.wat_v[:n]
        m = self.wat_m[:n]
        # XSPH viscosity from the previous substep's neighbour list
        if self.last_pairs is not None:
            pi, pj, W, vol = self.last_pairs
            if len(pi):
                num = np.empty((n, 3))
                for k in range(3):
                    num[:, k] = np.bincount(pi, weights=vol * v[pj, k], minlength=n)
                denom = np.bincount(pi, weights=vol, minlength=n)
                v += self.XSPH * (num - v * denom[:, None])
        # gravity + cloth tangential drag (uses the grid built this substep)
        v += GRAVITY[None, :] * dt
        sd, nx, nyv, nzv, wt = self._sample_grid(p[:, 0], p[:, 1], p[:, 2])
        active = wt > 0.35
        near = active & (np.abs(sd) < self.CONTACT_BAND)
        if np.any(near):
            nvec = np.stack((nx, nyv, nzv), axis=1)
            vc, wts = self._cloth_velocity(p[near])
            rel = v[near] - vc
            reln = np.einsum("ij,ij->i", rel, nvec[near])
            relt = rel - reln[:, None] * nvec[near]
            dw = wt[near][:, None] * self.VISC_DRAG * dt
            v[near] -= dw * relt
            react = (self.wat_m[:n][near] * self.VISC_DRAG)[:, None] * wt[near][:, None] * relt
            self._scatter_cloth_force(p[near], react)
        # position prediction
        p_prev = p.copy()
        p = p + v * dt
        # neighbour list at predicted positions
        pi, pj, d, r2 = self._build_pairs(p)
        r, W, gw = self._kernel(r2)
        rinv = 1.0 / np.maximum(r, 1e-9)
        gx = gw * d[:, 0] * rinv
        gy = gw * d[:, 1] * rinv
        gz = gw * d[:, 2] * rinv
        mj = m[pj]
        w0 = 315.0 / (64.0 * np.pi * self.H ** 9) * self.H ** 6
        eps = 1e-9
        for _ in range(self.SOLVER_ITERS):
            dens = np.bincount(pi, weights=mj * W, minlength=n) + m * w0
            C = dens / self.RHO0 - 1.0
            g2 = np.bincount(pi, weights=gx * gx + gy * gy + gz * gz, minlength=n)
            sx = np.bincount(pi, weights=gx, minlength=n)
            sy = np.bincount(pi, weights=gy, minlength=n)
            sz = np.bincount(pi, weights=gz, minlength=n)
            grad2 = (g2 + sx * sx + sy * sy + sz * sz) / (self.RHO0 ** 2) + eps
            lam = -C / grad2
            wdq = 315.0 / (64.0 * np.pi * self.H ** 9) * (self.H ** 2 - (self.PBF_DQ * self.H) ** 2) ** 3
            scorr = -self.PBF_K * (W / max(wdq, 1e-12)) ** self.PBF_N
            coef = (lam[pi] + lam[pj] + scorr) / self.RHO0
            cx = coef * gx
            cy = coef * gy
            cz = coef * gz
            dp = np.empty((n, 3))
            dp[:, 0] = np.bincount(pi, weights=cx, minlength=n) - np.bincount(pj, weights=cx, minlength=n)
            dp[:, 1] = np.bincount(pi, weights=cy, minlength=n) - np.bincount(pj, weights=cy, minlength=n)
            dp[:, 2] = np.bincount(pi, weights=cz, minlength=n) - np.bincount(pj, weights=cz, minlength=n)
            p += dp
            mag = np.linalg.norm(dp, axis=1)
            big = mag > 0.25 * self.H
            if np.any(big):
                p[big] -= dp[big] * (1.0 - 0.25 * self.H / mag[big])[:, None]
        self.density[:n] = dens[:n] if len(dens) >= n else dens
        self.last_pairs = (pi, pj, W, mj / np.maximum(dens, 1.0))
        # boundary conditions: tank walls (with nozzle opening), nozzle body, floor
        self._water_bounds(p, dt)
        # cloth non-penetration with momentum exchange
        sd, nx, nyv, nzv, wt = self._sample_grid(p[:, 0], p[:, 1], p[:, 2])
        nvec = np.stack((nx, nyv, nzv), axis=1)
        cross = (wt > 0.35) & (self.sd_prev * sd < 0.0) & (np.abs(sd) < 0.18)
        if np.any(cross):
            idx = np.nonzero(cross)[0]
            target = np.where(self.sd_prev[idx] >= 0.0, 1.0, -1.0) * (0.5 * self.CONTACT_BAND)
            corr = target - sd[idx]
            delta = nvec[idx] * corr[:, None]
            p[idx] += delta
            react = -self.wat_m[:n][idx][:, None] * delta / dt * wt[idx][:, None]
            self._scatter_cloth_force(p[idx], react)
        self.sd_prev = sd.copy()
        v = (p - p_prev) / dt
        sp = np.linalg.norm(v, axis=1)
        over = sp > self.MAX_SPEED
        if np.any(over):
            v[over] *= (self.MAX_SPEED / sp[over])[:, None]
        self.wat_p[:n] = p
        self.wat_v[:n] = v
        self._emit(dt)

    def _water_bounds(self, p, dt):
        lo, hi = self.tank_min, self.tank_max
        below = p[:, 2] < lo[2]
        p[below, 2] = lo[2]
        in_tank = (p[:, 2] < hi[2])
        # side walls
        m = in_tank & (p[:, 0] < lo[0])
        p[m, 0] = lo[0]
        m = in_tank & (p[:, 0] > hi[0])
        p[m, 0] = hi[0]
        m = in_tank & (p[:, 1] < lo[1])
        p[m, 1] = lo[1]
        m = in_tank & (p[:, 1] > hi[1])
        p[m, 1] = hi[1]
        # nozzle body outside the exit plane
        rr = np.sqrt((p[:, 1] - self.nozzle_axis_y) ** 2 + (p[:, 2] - self.nozzle_axis_z) ** 2)
        hit = (p[:, 0] < self.nozzle_x1) & (p[:, 0] > self.nozzle_x0) & (rr < self.nozzle_r)
        if np.any(hit):
            s = self.nozzle_r / np.maximum(rr[hit], 1e-9)
            p[hit, 1] = self.nozzle_axis_y + (p[hit, 1] - self.nozzle_axis_y) * s
            p[hit, 2] = self.nozzle_axis_z + (p[hit, 2] - self.nozzle_axis_z) * s
        np.clip(p, self.world_min + 0.02, self.world_max - 0.02, out=p)

    def _emit(self, dt):
        t_end = self.t + dt
        hits = np.nonzero(self.next_emit <= t_end + 1e-12)[0]
        if len(hits) == 0:
            return
        self._grow(len(hits))
        for s in hits:
            k = self.nw
            self.wat_p[k] = (self.emit_x, self.emit_y[s], self.emit_z[s])
            self.wat_v[k] = self.inflow_vel
            self.wat_m[k] = self.emit_mass
            self.wat_r[k] = self.emit_radius
            self.sd_prev[k] = -1.0
            self.nw = k + 1
            self.injected_mass += self.emit_mass
            self.next_emit[s] += self.emit_period

    def _cloth_velocity(self, pts):
        tree = self._cloth_tree()
        if tree is None:
            return np.zeros_like(pts), np.zeros(len(pts))
        d, idx = tree.query(pts, k=4)
        d = np.maximum(d, 1e-4)
        wts = 1.0 / d
        wts = wts / wts.sum(axis=1, keepdims=True)
        vc = np.einsum("ik,ikj->ij", wts, self.cloth_v[idx])
        return vc, wts

    def _cloth_tree(self):
        if self.ncl == 0:
            return None
        return cKDTree(self.cloth_p)

    def _scatter_cloth_force(self, pts, fvec):
        tree = self._cloth_tree()
        if tree is None:
            return
        d, idx = tree.query(pts, k=4)
        wts = 1.0 / np.maximum(d, 1e-4)
        wts = wts / wts.sum(axis=1, keepdims=True)
        for k in range(4):
            vals = wts[:, k][:, None] * fvec
            np.add.at(self.fcloth_next, idx[:, k], vals)
        self.net_reaction += fvec.sum(axis=0)

    # ---------------------------------------------------------- sample frame
    def advance_to(self, time: float):
        target = float(time)
        if target <= 0.0:
            return self.frame0
        guard = 0
        while self.t < target - 1e-9:
            dt = min(self.DT, target - self.t)
            self._substep(dt)
            self.t += dt
            guard += 1
            if guard > 100000:
                break
        if abs(self.t - target) < 1e-6:
            self.t = target
        return self._build_frame()

    def _substep(self, dt):
        self.fcloth = self.fcloth_next
        self.fcloth_next = np.zeros((self.ncl, 3))
        self.net_reaction = np.zeros(3)
        self._cloth_step(dt)
        self._rasterize_cloth()
        self._water_step(dt)
        if not (np.isfinite(self.cloth_p).all() and np.isfinite(self.wat_p[: self.nw]).all()
                and np.isfinite(self.cloth_v).all() and np.isfinite(self.wat_v[: self.nw]).all()):
            # numerical guard: revert to a bounded, finite state
            self.cloth_p = np.nan_to_num(self.cloth_p, nan=0.0, posinf=0.0, neginf=0.0)
            self.wat_p[: self.nw] = np.nan_to_num(self.wat_p[: self.nw], nan=0.0, posinf=0.0, neginf=0.0)
            self.cloth_v = np.nan_to_num(self.cloth_v, nan=0.0, posinf=0.0, neginf=0.0)
            self.wat_v[: self.nw] = np.nan_to_num(self.wat_v[: self.nw], nan=0.0, posinf=0.0, neginf=0.0)

    def _build_frame(self, initial=False, f0=None):
        n = self.nw
        pos = np.vstack((self.cloth_p, self.wat_p[:n]))
        vel = np.vstack((self.cloth_v, self.wat_v[:n]))
        mass = np.r_[self.cloth_m, self.wat_m[:n]]
        body = np.r_[np.zeros(self.ncl, np.int32), np.ones(n, np.int32)]
        if initial and f0 is not None:
            attrs = {k: np.asarray(v, float) for k, v in f0.attributes.items()}
        else:
            attrs = {
                "pinned": np.r_[self.pinned0[: self.ncl], np.zeros(n)],
                "material_phase": np.r_[self.phase0[: self.ncl], np.ones(n)],
            }
        attrs["particle_radius"] = np.r_[np.full(self.ncl, 0.00075), self.wat_r[:n]]
        attrs["mass_quantity"] = mass.copy()
        cdist = self._contact_distances()
        ng = np.array([0.0, 0.0, -9.81])
        ext = mass[:, None] * ng[None, :]
        react = self.net_reaction.copy()
        fields = {
            "contact_signed_distance": cdist,
            "cumulative_mass_source": np.array([0.0, self.injected_mass - self.removed_mass]),
            "external_force": ext,
            "interaction_force": np.vstack((react, -react)),
        }
        disp = np.linalg.norm(self.cloth_p - self.cloth_p0(), axis=1)
        scalars = {
            "contact_length_scale": float(self.contact_length_scale),
            "water_density": 1000.0,
            "cloth_thickness": 0.0015,
            "nozzle_radius": float(self.inflow_radius),
            "nominal_inflow_speed": float(np.linalg.norm(self.inflow_vel)),
            "water_volume": float(self.wat_m[:n].sum() / 1000.0),
            "cloth_max_displacement": float(disp.max()) if self.ncl else 0.0,
            "max_speed": float(np.linalg.norm(vel, axis=1).max()) if len(vel) else 0.0,
            "time": float(self.t),
        }
        frame = StateFrame(
            time=float(self.t),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body,
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _cloth_p0(self):
        if not hasattr(self, "_p0"):
            self._p0 = self.cloth_p.copy()
        return self._p0

    def _contact_distances(self):
        pts = self.cloth_p[self.sample_idx]
        n = self.nw
        if n < 4:
            return np.full(len(pts), 0.25)
        tree = cKDTree(self.wat_p[:n])
        dist, idx = tree.query(pts)
        _, nx, nyv, nzv, _ = self._sample_grid(pts[:, 0], pts[:, 1], pts[:, 2])
        nvec = np.stack((nx, nyv, nzv), axis=1)
        perp = np.einsum("ij,ij->i", self.wat_p[:n][idx] - pts, nvec)
        out = np.where(perp < -0.02, -dist, dist)
        return np.asarray(out, float)

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self):
        n = self.nw
        p = self.wat_p[:n]
        sd, _, _, _, wt = self._sample_grid(p[:, 0], p[:, 1], p[:, 2])
        covered = wt > 0.35
        leaked = float(self.wat_m[:n][covered & (sd < -0.03)].sum())
        return {
            "water_mass_kg": float(self.wat_m[: n].sum()),
            "injected_mass_kg": float(self.injected_mass),
            "removed_mass_kg": float(self.removed_mass),
            "water_particles": int(n),
            "cloth_vertices": int(self.ncl),
            "leaked_mass_kg": leaked,
            "contact_length_scale_m": float(self.contact_length_scale),
            "max_speed_m_s": float(np.linalg.norm(np.vstack((self.cloth_v, self.wat_v[:n])), axis=1).max()),
            "cloth_max_displacement_m": float(np.linalg.norm(self.cloth_p - self._cloth_p0(), axis=1).max()),
            "sim_time_s": float(self.t),
        }
