"""Incompressible smoke solver: turbulent stream through complex obstacles.

Physical model
--------------
Inviscid incompressible Euler equations for a buoyant smoke tracer, solved on a
uniform staggered MAC grid covering the frozen voxel domain of the ``subject``
object:

    du/dt + (u.grad)u = -grad p + f_buoy + f_confine
    div u = 0
    ds/dt = q(x, t) - alpha_s * s

Discretization
--------------
* Uniform staggered (MAC) grid, cell size h = 0.1 m.
* Semi-Lagrangian advection of momentum (unconditionally stable backtrace).
* MacCormack (second-order, limited) advection of the smoke scalar.
* Exact pressure projection: sparse SPD Poisson system solved with Jacobi
  preconditioned conjugate gradients, warm started from the previous step.
* Solid box obstacles are voxelized to cell centres; every face touching a
  solid cell carries a zero normal velocity (Neumann pressure boundary).
* The nozzle is an interior prescribed-velocity block (a real inflow boundary
  condition), the remaining domain sides are open (p = 0 Dirichlet), the ground
  plane at z = 0 is a no-penetration wall with tangential drag.

Everything is deterministic: the only randomness is drawn from a seeded
generator during initialisation to build the inflow perturbation spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from worldsolver.contracts import StateFrame

_OFFSETS = {
    "u": (0.0, 0.5, 0.5),
    "v": (0.5, 0.0, 0.5),
    "w": (0.5, 0.5, 0.0),
    "c": (0.5, 0.5, 0.5),
}


class BenchmarkSolver:
    # numerical / physical parameters
    H = 0.1                # grid spacing [m]
    SUBSTEPS = 3           # time steps per sample interval
    JET_SPEED = 2.5        # prescribed inlet speed [m/s]
    BUOYANCY = 0.8         # upward acceleration per unit smoke [m/s^2]
    CONFINEMENT = 5.0      # vorticity confinement strength [-]
    SMOKE_DECAY = 0.01     # smoke dissipation rate [1/s]
    SOURCE_RATE = 40.0     # smoke injection relaxation rate [1/s]
    INFLOW_TURB = 0.07     # inflow perturbation amplitude (fraction of U)
    WALL_TAU = 0.10        # wall tangential drag time constant [s]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        config = dict(getattr(scene, "config", {}) or {})
        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.dt = self.sample_dt / float(self.SUBSTEPS)

        self._read_geometry(config)
        self._build_grid()
        self._build_masks()
        self._build_poisson()
        self._build_boundary_samples()
        self._init_state()

        self.time = 0.0
        self.n_steps = 0
        self._cg_iters = 0

    def _read_geometry(self, config: dict) -> None:
        geo = config.get("scene_geometry") or {}
        objects = geo.get("objects") or []
        self.bounds_min = np.array([-3.0, -1.2, 0.0], float)
        self.bounds_max = np.array([3.0, 1.2, 3.8], float)
        self.export_shape = tuple(
            int(v) for v in (config.get("sampling", {}) or {}).get("field_grid_shape", (24, 16, 28))
        )
        self.boxes: list[tuple[np.ndarray, np.ndarray]] = []
        nozzle = None
        for item in objects:
            gtype = (item.get("geometry") or {}).get("type")
            semantic = item.get("semantic")
            trans = np.asarray(
                ((item.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]), float
            )
            if semantic == "gas" and gtype == "voxel_grid":
                gm = item["geometry"]
                self.bounds_min = np.asarray(gm["bounds_min"], float)
                self.bounds_max = np.asarray(gm["bounds_max"], float)
                if gm.get("grid_shape"):
                    self.export_shape = tuple(int(v) for v in gm["grid_shape"])
            elif semantic == "boundary" and gtype == "box":
                half = np.asarray(item["geometry"]["half_extents"], float)
                self.boxes.append((trans - half, trans + half))
            elif semantic == "emitter":
                radius = float((item.get("geometry") or {}).get("radius") or 0.4)
                height = float((item.get("geometry") or {}).get("height") or 0.18)
                nozzle = (trans, radius, height)
        if nozzle is None:
            nozzle = (np.array([-2.72, 0.0, 1.0]), 0.4, 0.18)
        self.nozzle_center, self.nozzle_radius, self.nozzle_height = nozzle

    def _build_grid(self) -> None:
        h = self.H
        lo, hi = self.bounds_min, self.bounds_max
        self.shape = tuple(max(int(round((hi[i] - lo[i]) / h)), 4) for i in range(3))
        Nx, Ny, Nz = self.shape
        self.origin = lo.copy()
        self.h = h

        self.xc = lo[0] + (np.arange(Nx) + 0.5) * h
        self.yc = lo[1] + (np.arange(Ny) + 0.5) * h
        self.zc = lo[2] + (np.arange(Nz) + 0.5) * h
        self.xf = lo[0] + np.arange(Nx + 1) * h
        self.yf = lo[1] + np.arange(Ny + 1) * h
        self.zf = lo[2] + np.arange(Nz + 1) * h

        def grids(xs, ys, zs):
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            return X, Y, Z

        self.uX, self.uY, self.uZ = grids(self.xf, self.yc, self.zc)
        self.vX, self.vY, self.vZ = grids(self.xc, self.yf, self.zc)
        self.wX, self.wY, self.wZ = grids(self.xc, self.yc, self.zf)
        self.cX, self.cY, self.cZ = grids(self.xc, self.yc, self.zc)

    # ------------------------------------------------------------------ masks
    def _inside_boxes(self, X, Y, Z) -> np.ndarray:
        inside = np.zeros(np.shape(X), bool)
        for lo, hi in self.boxes:
            inside |= (
                (X >= lo[0]) & (X <= hi[0])
                & (Y >= lo[1]) & (Y <= hi[1])
                & (Z >= lo[2]) & (Z <= hi[2])
            )
        return inside

    def _inside_nozzle(self, X, Y, Z) -> np.ndarray:
        c, r, hgt = self.nozzle_center, self.nozzle_radius, self.nozzle_height
        rad = np.sqrt((Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        return (np.abs(X - c[0]) <= 0.5 * hgt) & (rad <= r)

    def _build_masks(self) -> None:
        Nx, Ny, Nz = self.shape
        self.solid_c = self._inside_boxes(self.cX, self.cY, self.cZ)
        self.nozzle_c = self._inside_nozzle(self.cX, self.cY, self.cZ) & ~self.solid_c
        self.blocked_c = self.solid_c | self.nozzle_c

        # faces with prescribed (Neumann) velocity
        self.u_solid = np.zeros((Nx + 1, Ny, Nz), bool)
        self.u_solid[1:Nx] = self.solid_c[1:] | self.solid_c[:-1]
        self.v_solid = np.zeros((Nx, Ny + 1, Nz), bool)
        self.v_solid[:, 1:Ny] = self.solid_c[:, 1:] | self.solid_c[:, :-1]
        self.w_solid = np.zeros((Nx, Ny, Nz + 1), bool)
        self.w_solid[:, :, 1:Nz] = self.solid_c[:, :, 1:] | self.solid_c[:, :, :-1]
        self.w_solid[:, :, 0] = True  # ground plane

        # nozzle faces: back wall closed, interior + exit faces blow at U
        noz = self.nozzle_c
        self.u_jet = np.zeros((Nx + 1, Ny, Nz), bool)     # faces driven at +U
        self.u_wall = np.zeros((Nx + 1, Ny, Nz), bool)    # nozzle back wall
        left = np.zeros((Nx + 1, Ny, Nz), bool)
        right = np.zeros((Nx + 1, Ny, Nz), bool)
        left[1:] = noz
        right[:Nx] = noz
        self.u_jet = left & ~self.u_solid
        self.u_wall = right & ~left & ~self.u_solid
        self.v_noz = np.zeros((Nx, Ny + 1, Nz), bool)
        self.v_noz[:, 1:] |= noz
        self.v_noz[:, :Ny] |= noz
        self.v_noz &= ~self.v_solid
        self.w_noz = np.zeros((Nx, Ny, Nz + 1), bool)
        self.w_noz[:, :, 1:] |= noz
        self.w_noz[:, :, :Nz] |= noz
        self.w_noz &= ~self.w_solid

        self.u_fixed = self.u_solid | self.u_jet | self.u_wall
        self.v_fixed = self.v_solid | self.v_noz
        self.w_fixed = self.w_solid | self.w_noz

        # jet profile on the driven faces (flat core, cosine taper)
        c, r = self.nozzle_center, self.nozzle_radius
        rad = np.sqrt((self.uY - c[1]) ** 2 + (self.uZ - c[2]) ** 2)
        taper = np.clip((rad - 0.55 * r) / (0.45 * r), 0.0, 1.0)
        self.jet_profile = 0.5 * (1.0 + np.cos(np.pi * taper))
        self.jet_shear_z = np.clip((self.uZ - c[2]) / r, -1.0, 1.0)
        self.jet_shear_y = np.clip((self.uY - c[1]) / r, -1.0, 1.0)
        self.v_shear = np.clip((self.vZ - c[2]) / r, -1.0, 1.0)
        self.w_shear = np.clip((self.wY - c[1]) / r, -1.0, 1.0)

        # tangential wall drag masks (faces beside a solid cell / the ground)
        tang_u = np.zeros((Nx, Ny, Nz), bool)
        tang_u[:, 1:, :] |= self.solid_c[:, :-1, :]
        tang_u[:, :-1, :] |= self.solid_c[:, 1:, :]
        tang_u[:, :, 1:] |= self.solid_c[:, :, :-1]
        tang_u[:, :, :-1] |= self.solid_c[:, :, 1:]
        um = np.zeros((Nx + 1, Ny, Nz), bool)
        um[:Nx] |= tang_u
        um[1:] |= tang_u
        um[:, :, 0] = True
        self.u_drag = um & ~self.u_fixed

        tang_v = np.zeros((Nx, Ny, Nz), bool)
        tang_v[1:, :, :] |= self.solid_c[:-1, :, :]
        tang_v[:-1, :, :] |= self.solid_c[1:, :, :]
        tang_v[:, :, 1:] |= self.solid_c[:, :, :-1]
        tang_v[:, :, :-1] |= self.solid_c[:, :, 1:]
        vm = np.zeros((Nx, Ny + 1, Nz), bool)
        vm[:, :Ny] |= tang_v
        vm[:, 1:] |= tang_v
        vm[:, :, 0] = True
        self.v_drag = vm & ~self.v_fixed

        tang_w = np.zeros((Nx, Ny, Nz), bool)
        tang_w[1:, :, :] |= self.solid_c[:-1, :, :]
        tang_w[:-1, :, :] |= self.solid_c[1:, :, :]
        tang_w[:, 1:, :] |= self.solid_c[:, :-1, :]
        tang_w[:, :-1, :] |= self.solid_c[:, 1:, :]
        wm = np.zeros((Nx, Ny, Nz + 1), bool)
        wm[:, :, :Nz] |= tang_w
        wm[:, :, 1:] |= tang_w
        self.w_drag = wm & ~self.w_fixed

    # ---------------------------------------------------------------- poisson
    def _build_poisson(self) -> None:
        Nx, Ny, Nz = self.shape
        fluid = ~self.blocked_c
        idx = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
        diag = np.zeros((Nx, Ny, Nz), float)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def couple(mask, a, b):
            n = int(mask.sum())
            if n == 0:
                return
            ai, bi = a[mask], b[mask]
            rows.append(ai)
            cols.append(bi)
            vals.append(-np.ones(n))
            rows.append(bi)
            cols.append(ai)
            vals.append(-np.ones(n))

        mx = fluid[1:, :, :] & fluid[:-1, :, :]
        couple(mx, idx[1:, :, :], idx[:-1, :, :])
        diag[1:, :, :] += mx
        diag[:-1, :, :] += mx

        my = fluid[:, 1:, :] & fluid[:, :-1, :]
        couple(my, idx[:, 1:, :], idx[:, :-1, :])
        diag[:, 1:, :] += my
        diag[:, :-1, :] += my

        mz = fluid[:, :, 1:] & fluid[:, :, :-1]
        couple(mz, idx[:, :, 1:], idx[:, :, :-1])
        diag[:, :, 1:] += mz
        diag[:, :, :-1] += mz

        # open (Dirichlet p = 0) domain faces: x-min, x-max, y-min, y-max, z-max
        diag[0, :, :] += fluid[0, :, :]
        diag[-1, :, :] += fluid[-1, :, :]
        diag[:, 0, :] += fluid[:, 0, :]
        diag[:, -1, :] += fluid[:, -1, :]
        diag[:, :, -1] += fluid[:, :, -1]

        diag[self.blocked_c] = 1.0
        diag[diag <= 0.0] = 1.0
        rows.append(idx.ravel())
        cols.append(idx.ravel())
        vals.append(diag.ravel())

        n = Nx * Ny * Nz
        A = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        ).tocsr()
        self.A = A
        self.Minv = sparse.diags(1.0 / A.diagonal())

    def _solve_pressure(self, dt: float) -> None:
        u, v, w, h = self.u, self.v, self.w, self.h
        div = (
            (u[1:, :, :] - u[:-1, :, :])
            + (v[:, 1:, :] - v[:, :-1, :])
            + (w[:, :, 1:] - w[:, :, :-1])
        ) / h
        b = -(h * h / dt) * div
        b[self.blocked_c] = 0.0
        x0 = self.p.ravel()
        try:
            sol, _ = cg(self.A, b.ravel(), x0=x0, rtol=1e-5, atol=0.0, maxiter=200, M=self.Minv)
        except TypeError:  # older SciPy
            sol, _ = cg(self.A, b.ravel(), x0=x0, tol=1e-5, atol=0.0, maxiter=200, M=self.Minv)
        if not np.all(np.isfinite(sol)):
            sol = np.zeros_like(x0)
        p = sol.reshape(self.shape)
        p[self.blocked_c] = 0.0
        self.p = p

        s = dt / h
        u[1:-1, :, :] -= s * (p[1:, :, :] - p[:-1, :, :])
        u[0, :, :] -= s * p[0, :, :]
        u[-1, :, :] += s * p[-1, :, :]
        v[:, 1:-1, :] -= s * (p[:, 1:, :] - p[:, :-1, :])
        v[:, 0, :] -= s * p[:, 0, :]
        v[:, -1, :] += s * p[:, -1, :]
        w[:, :, 1:-1] -= s * (p[:, :, 1:] - p[:, :, :-1])
        w[:, :, -1] += s * p[:, :, -1]

    # ------------------------------------------------------------ state setup
    def _init_state(self) -> None:
        Nx, Ny, Nz = self.shape
        self.u = np.zeros((Nx + 1, Ny, Nz))
        self.v = np.zeros((Nx, Ny + 1, Nz))
        self.w = np.zeros((Nx, Ny, Nz + 1))
        self.p = np.zeros((Nx, Ny, Nz))

        frame0 = self.scene.initial_state()
        self.base_scalars = dict(frame0.scalars)
        self.part_attrs = {k: np.array(v, copy=True) for k, v in frame0.attributes.items()}
        self.part_mass = None if frame0.mass is None else np.array(frame0.mass, copy=True)
        self.part_body = None if frame0.body_id is None else np.array(frame0.body_id, copy=True)
        self.particles = np.array(frame0.position, float, copy=True)
        self.part_vel = np.zeros_like(self.particles)
        self.n_part = len(self.particles)

        # smoke initialised from the frozen initial field, resampled to the sim grid
        init_field = None
        if isinstance(frame0.fields, dict):
            init_field = frame0.fields.get("transported_scalar")
        if init_field is not None:
            init = np.asarray(init_field, float)
            ex = self._export_axes(init.shape)
            coords = [
                np.interp(self.cX, ex[0], np.arange(init.shape[0])),
                np.interp(self.cY, ex[1], np.arange(init.shape[1])),
                np.interp(self.cZ, ex[2], np.arange(init.shape[2])),
            ]
            self.s = ndimage.map_coordinates(init, coords, order=1, mode="nearest")
        else:
            self.s = np.zeros(self.shape)
        self.s = np.clip(self.s, 0.0, 5.0)
        self.s[self.solid_c] = 0.0

        # deterministic inflow perturbation spectrum (seeded once)
        k = 6
        self.turb_f = self.rng.uniform(0.6, 4.0, (4, k))
        self.turb_ph = self.rng.uniform(0.0, 2.0 * np.pi, (4, k))
        self.turb_a = self.rng.uniform(0.4, 1.0, (4, k))
        self.turb_a /= np.sum(self.turb_a, axis=1, keepdims=True)

        self._apply_inflow(0.0)
        self._zero_fixed()

    def _export_axes(self, shape):
        lo, hi = self.bounds_min, self.bounds_max
        return [np.linspace(lo[i], hi[i], int(shape[i])) for i in range(3)]

    # ------------------------------------------------------------- operations
    def _sample(self, arr, kind, X, Y, Z):
        ox, oy, oz = _OFFSETS[kind]
        o, h = self.origin, self.h
        coords = [
            (X - o[0]) / h - ox,
            (Y - o[1]) / h - oy,
            (Z - o[2]) / h - oz,
        ]
        return ndimage.map_coordinates(arr, coords, order=1, mode="nearest")

    def _turb(self, row: int, t: float) -> float:
        return float(
            np.sum(self.turb_a[row] * np.sin(2.0 * np.pi * self.turb_f[row] * t + self.turb_ph[row]))
        )

    def _apply_inflow(self, t: float) -> None:
        U = self.JET_SPEED
        amp = self.INFLOW_TURB
        m0 = 1.0 + 0.5 * amp * self._turb(0, t)
        ay = amp * self._turb(1, t)
        az = amp * self._turb(2, t)
        swirl = amp * self._turb(3, t)

        self.u[self.u_jet] = (U * m0 * self.jet_profile)[self.u_jet]
        self.u[self.u_wall] = 0.0
        vv = U * (ay + swirl * self.v_shear)
        self.v[self.v_noz] = vv[self.v_noz] if np.ndim(vv) else vv
        ww = U * (az - swirl * self.w_shear)
        self.w[self.w_noz] = ww[self.w_noz] if np.ndim(ww) else ww

    def _zero_fixed(self) -> None:
        self.u[self.u_solid] = 0.0
        self.v[self.v_solid] = 0.0
        self.w[self.w_solid] = 0.0

    def _center_velocity(self):
        uc = 0.5 * (self.u[1:, :, :] + self.u[:-1, :, :])
        vc = 0.5 * (self.v[:, 1:, :] + self.v[:, :-1, :])
        wc = 0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])
        return uc, vc, wc

    def _advect_velocity(self, dt: float) -> None:
        u, v, w = self.u, self.v, self.w
        # u nodes
        vu = self._sample(v, "v", self.uX, self.uY, self.uZ)
        wu = self._sample(w, "w", self.uX, self.uY, self.uZ)
        un = self._sample(u, "u", self.uX - dt * u, self.uY - dt * vu, self.uZ - dt * wu)
        # v nodes
        uv = self._sample(u, "u", self.vX, self.vY, self.vZ)
        wv = self._sample(w, "w", self.vX, self.vY, self.vZ)
        vn = self._sample(v, "v", self.vX - dt * uv, self.vY - dt * v, self.vZ - dt * wv)
        # w nodes
        uw = self._sample(u, "u", self.wX, self.wY, self.wZ)
        vw = self._sample(v, "v", self.wX, self.wY, self.wZ)
        wn = self._sample(w, "w", self.wX - dt * uw, self.wY - dt * vw, self.wZ - dt * w)
        self.u, self.v, self.w = un, vn, wn

    def _add_forces(self, dt: float) -> None:
        Nx, Ny, Nz = self.shape
        # buoyancy on the smoke
        sw = 0.5 * (self.s[:, :, 1:] + self.s[:, :, :-1])
        self.w[:, :, 1:Nz] += dt * self.BUOYANCY * sw

        # vorticity confinement (sustains the shed eddies on a coarse grid)
        if self.CONFINEMENT > 0.0:
            uc, vc, wc = self._center_velocity()
            h = self.h
            wx = np.gradient(wc, h, axis=1) - np.gradient(vc, h, axis=2)
            wy = np.gradient(uc, h, axis=2) - np.gradient(wc, h, axis=0)
            wz = np.gradient(vc, h, axis=0) - np.gradient(uc, h, axis=1)
            mag = np.sqrt(wx * wx + wy * wy + wz * wz)
            gx = np.gradient(mag, h, axis=0)
            gy = np.gradient(mag, h, axis=1)
            gz = np.gradient(mag, h, axis=2)
            gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8
            gx, gy, gz = gx / gn, gy / gn, gz / gn
            eps = self.CONFINEMENT * h
            fx = eps * (gy * wz - gz * wy)
            fy = eps * (gz * wx - gx * wz)
            fz = eps * (gx * wy - gy * wx)
            free = (~self.blocked_c).astype(float)
            fx *= free
            fy *= free
            fz *= free
            self.u[1:Nx, :, :] += dt * 0.5 * (fx[1:, :, :] + fx[:-1, :, :])
            self.v[:, 1:Ny, :] += dt * 0.5 * (fy[:, 1:, :] + fy[:, :-1, :])
            self.w[:, :, 1:Nz] += dt * 0.5 * (fz[:, :, 1:] + fz[:, :, :-1])

        # tangential wall drag (no-slip boundary layer -> separation)
        fric = float(np.exp(-dt / self.WALL_TAU))
        self.u[self.u_drag] *= fric
        self.v[self.v_drag] *= fric
        self.w[self.w_drag] *= fric

    def _advect_scalar(self, dt: float) -> None:
        uc, vc, wc = self._center_velocity()
        Xb = self.cX - dt * uc
        Yb = self.cY - dt * vc
        Zb = self.cZ - dt * wc
        s = self.s
        s1 = self._sample(s, "c", Xb, Yb, Zb)
        s2 = self._sample(s1, "c", self.cX + dt * uc, self.cY + dt * vc, self.cZ + dt * wc)
        s_new = s1 + 0.5 * (s - s2)
        lo = self._sample(ndimage.minimum_filter(s, size=3, mode="nearest"), "c", Xb, Yb, Zb)
        hi = self._sample(ndimage.maximum_filter(s, size=3, mode="nearest"), "c", Xb, Yb, Zb)
        s_new = np.clip(s_new, lo, hi)

        src = self._scalar_source(s_new)
        s_new = s_new + dt * src
        s_new *= max(0.0, 1.0 - self.SMOKE_DECAY * dt)
        s_new[self.solid_c] = 0.0
        self.s = np.clip(np.nan_to_num(s_new, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)

    def _scalar_source(self, s) -> np.ndarray:
        src = np.zeros_like(s)
        src[self.nozzle_c] = self.SOURCE_RATE * np.clip(1.0 - s[self.nozzle_c], 0.0, None)
        return src

    def _advect_particles(self, dt: float) -> None:
        if self.n_part == 0:
            return
        pos = self.particles
        X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
        vel = np.stack(
            (
                self._sample(self.u, "u", X, Y, Z),
                self._sample(self.v, "v", X, Y, Z),
                self._sample(self.w, "w", X, Y, Z),
            ),
            axis=1,
        )
        new = pos + dt * vel
        inside_solid = self._inside_boxes(new[:, 0], new[:, 1], new[:, 2])
        new[inside_solid] = pos[inside_solid]
        lo, hi = self.bounds_min, self.bounds_max
        out = np.any(new < lo + 1e-6, axis=1) | np.any(new > hi - 1e-6, axis=1)
        if np.any(out):
            k = int(out.sum())
            ang = self.rng.uniform(0.0, 2.0 * np.pi, k)
            rad = self.nozzle_radius * 0.85 * np.sqrt(self.rng.uniform(0.0, 1.0, k))
            c = self.nozzle_center
            new[out, 0] = c[0] + 0.5 * self.nozzle_height + self.h
            new[out, 1] = c[1] + rad * np.cos(ang)
            new[out, 2] = c[2] + rad * np.sin(ang)
            vel[out] = 0.0
        self.particles = np.clip(new, lo + 1e-6, hi - 1e-6)
        self.part_vel = np.nan_to_num(vel)

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        self._advect_velocity(dt)
        self._zero_fixed()
        self._apply_inflow(self.time + dt)
        self._add_forces(dt)
        self._zero_fixed()
        self._apply_inflow(self.time + dt)
        self._solve_pressure(dt)
        self._zero_fixed()
        self._apply_inflow(self.time + dt)
        self._advect_scalar(dt)
        self._advect_particles(dt)
        for arr in (self.u, self.v, self.w):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(arr, -50.0, 50.0, out=arr)
        self.time += dt
        self.n_steps += 1

    # ------------------------------------------------------------- boundaries
    def _build_boundary_samples(self) -> None:
        h = self.h
        lo, hi = self.bounds_min, self.bounds_max
        pts: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        def snap_axis(axis: int, value: float) -> float:
            return float(self.origin[axis] + round((value - self.origin[axis]) / h) * h)

        step = 0.22
        for blo, bhi in self.boxes:
            for axis in range(3):
                for side, plane in ((-1.0, blo[axis]), (1.0, bhi[axis])):
                    coord = snap_axis(axis, plane)
                    ta, tb = [a for a in range(3) if a != axis]
                    a_lo = snap_axis(ta, blo[ta]) + 0.5 * step
                    a_hi = snap_axis(ta, bhi[ta]) - 0.25 * step
                    b_lo = snap_axis(tb, blo[tb]) + 0.5 * step
                    b_hi = snap_axis(tb, bhi[tb]) - 0.25 * step
                    av = np.arange(a_lo, max(a_hi, a_lo + 1e-9), step)
                    bv = np.arange(b_lo, max(b_hi, b_lo + 1e-9), step)
                    if av.size == 0 or bv.size == 0:
                        continue
                    A, B = np.meshgrid(av, bv, indexing="ij")
                    P = np.zeros((A.size, 3))
                    P[:, axis] = coord
                    P[:, ta] = A.ravel()
                    P[:, tb] = B.ravel()
                    N = np.zeros((A.size, 3))
                    N[:, axis] = side
                    keep = np.all(P >= lo - 1e-9, axis=1) & np.all(P <= hi + 1e-9, axis=1)
                    if not np.any(keep):
                        continue
                    pts.append(P[keep])
                    nrm.append(N[keep])
        if pts:
            self.bnd_pts = np.concatenate(pts)
            self.bnd_nrm = np.concatenate(nrm)
        else:
            self.bnd_pts = np.array([[0.0, 0.0, 0.0]])
            self.bnd_nrm = np.array([[0.0, 0.0, 1.0]])
        self.bnd_target = np.zeros_like(self.bnd_pts)

    def _boundary_velocity(self) -> np.ndarray:
        X, Y, Z = self.bnd_pts[:, 0], self.bnd_pts[:, 1], self.bnd_pts[:, 2]
        vel = np.stack(
            (
                self._sample(self.u, "u", X, Y, Z),
                self._sample(self.v, "v", X, Y, Z),
                self._sample(self.w, "w", X, Y, Z),
            ),
            axis=1,
        )
        return np.nan_to_num(vel)

    # ----------------------------------------------------------------- export
    def _export_fields(self) -> dict:
        shape = self.export_shape
        ax = self._export_axes(shape)
        X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
        scalar = self._sample(self.s, "c", X, Y, Z)
        uc, vc, wc = self._center_velocity()
        vel = np.stack(
            (
                self._sample(uc, "c", X, Y, Z),
                self._sample(vc, "c", X, Y, Z),
                self._sample(wc, "c", X, Y, Z),
            ),
            axis=-1,
        )
        solid = self._inside_boxes(X, Y, Z)
        noz = self._inside_nozzle(X, Y, Z)
        scalar = np.where(solid, 0.0, np.clip(scalar, 0.0, 5.0))
        vel = np.where(solid[..., None], 0.0, vel)
        source = np.zeros(shape)
        source[noz] = self.SOURCE_RATE * np.clip(1.0 - scalar[noz], 0.0, None)
        active = ~solid
        return {
            "transported_scalar": np.nan_to_num(scalar),
            "scalar_source": np.nan_to_num(source),
            "active_mask": active,
            "velocity_field": np.nan_to_num(vel),
        }

    def _frame(self) -> StateFrame:
        fields = self._export_fields()
        fields["boundary_velocity"] = self._boundary_velocity()
        fields["boundary_target_velocity"] = self.bnd_target
        fields["boundary_normal"] = self.bnd_nrm
        fields["cumulative_mass_source"] = np.zeros(self.n_part)

        scalars = {
            "contact_length_scale": float(self.base_scalars.get("contact_length_scale", self.h)),
            "grid_spacing": float(self.base_scalars.get("grid_spacing", self.h)),
            "characteristic_length": float(
                self.base_scalars.get("characteristic_length", 2.0 * self.nozzle_radius)
            ),
            "reference_velocity": float(self.JET_SPEED),
        }
        frame = StateFrame(
            time=float(self.time),
            position=np.ascontiguousarray(self.particles, dtype=float),
            velocity=np.ascontiguousarray(self.part_vel, dtype=float),
            mass=self.part_mass,
            body_id=self.part_body,
            attributes={k: v for k, v in self.part_attrs.items()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-9 and guard < 100000:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "staggered MAC incompressible smoke, semi-Lagrangian + CG projection",
            "grid_shape": list(self.shape),
            "grid_spacing": float(self.h),
            "time_step": float(self.dt),
            "steps": int(self.n_steps),
            "jet_speed": float(self.JET_SPEED),
            "buoyancy": float(self.BUOYANCY),
            "confinement": float(self.CONFINEMENT),
            "obstacle_cells": int(self.solid_c.sum()),
            "nozzle_cells": int(self.nozzle_c.sum()),
            "boundary_samples": int(len(self.bnd_pts)),
            "final_smoke_mass": float(self.s.sum() * self.h ** 3),
        }
