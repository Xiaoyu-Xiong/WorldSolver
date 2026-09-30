"""Physically-based smoke rising around a sphere (Eulerian gas solver).

Self-contained implementation (no Genesis): staggered MAC grid with
semi-Lagrangian advection, gravity-aligned buoyancy, vorticity confinement and
a sparse pressure projection enforcing no-through-flow on the fixed sphere
obstacle, the ground plane and the enclosed domain walls.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # physical / numerical parameters
    N_SUBSTEPS = 2          # sub-steps per sample interval
    RHO = 1.2               # air density (frozen sampling density) [kg/m^3]
    BUOYANCY = 2.0          # upward accel per unit smoke density [m/s^2]
    VORTICITY_EPS = 0.22    # vorticity confinement strength [dimensionless]
    SOURCE_RATE = 3.5       # smoke emission rate [1/s]
    SMOKE_MAX = 1.5         # clip of the transported scalar
    SMOKE_DECAY = 0.02      # passive dissipation [1/s]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.rng = np.random.default_rng(int(seed))
        cfg = scene.config
        base = scene.initial_state()
        self._base = base

        objects = (cfg.get("scene_geometry") or {}).get("objects", [])

        def _find(oid):
            for o in objects:
                if o.get("id") == oid:
                    return o
            return None

        sph = _find("obstacle_sphere")
        if sph is not None:
            self.sphere_c = np.asarray(sph["transform"]["translation"], float)
            self.sphere_r = float(sph["geometry"]["radius"])
        else:
            self.sphere_c = np.array([0.0, 0.0, 1.35])
            self.sphere_r = 0.48

        src = _find("smoke_source")
        if src is not None:
            self.src_c = np.asarray(src["transform"]["translation"], float)
            g = src["geometry"]
            self.src_r = float(g["radius"])
            self.src_h = float(g["height"])
        else:
            self.src_c = np.array([0.0, 0.0, 0.05])
            self.src_r, self.src_h = 0.32, 0.1

        subj = _find("subject")
        if subj is not None:
            g = subj["geometry"]
            shape = tuple(int(v) for v in g["grid_shape"])
            lo = np.asarray(g["bounds_min"], float)
            hi = np.asarray(g["bounds_max"], float)
        else:
            shape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
            lo = np.array([-3.0, -1.2, 0.0])
            hi = np.array([3.0, 1.2, 3.8])

        self.nx, self.ny, self.nz = shape
        self.lo, self.hi = lo, hi
        self.dx = (hi[0] - lo[0]) / self.nx
        self.dy = (hi[1] - lo[1]) / self.ny
        self.dz = (hi[2] - lo[2]) / self.nz
        self.h_eq = float((self.dx * self.dy * self.dz) ** (1.0 / 3.0))

        # cell centers
        cx = lo[0] + (np.arange(self.nx) + 0.5) * self.dx
        cy = lo[1] + (np.arange(self.ny) + 0.5) * self.dy
        cz = lo[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.cx, self.cy, self.cz = cx, cy, cz
        X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        self.solid = ((X - self.sphere_c[0]) ** 2 +
                      (Y - self.sphere_c[1]) ** 2 +
                      (Z - self.sphere_c[2]) ** 2) < self.sphere_r ** 2
        self.fluid = ~self.solid

        # face positions
        fx = lo[0] + np.arange(self.nx + 1) * self.dx
        fy = lo[1] + np.arange(self.ny + 1) * self.dy
        fz = lo[2] + np.arange(self.nz + 1) * self.dz
        self.Xfx, self.Yfx, self.Zfx = np.meshgrid(fx, cy, cz, indexing="ij")
        self.Xfy, self.Yfy, self.Zfy = np.meshgrid(cx, fy, cz, indexing="ij")
        self.Xfz, self.Yfz, self.Zfz = np.meshgrid(cx, cy, fz, indexing="ij")
        self.Xc, self.Yc, self.Zc = X, Y, Z

        # open (interior, non-solid) face masks; domain boundary faces closed
        ox = np.ones((self.nx + 1, self.ny, self.nz), bool)
        ox[0] = False
        ox[-1] = False
        ox[1:-1] = self.fluid[:-1] & self.fluid[1:]
        oy = np.ones((self.nx, self.ny + 1, self.nz), bool)
        oy[:, 0] = False
        oy[:, -1] = False
        oy[:, 1:-1] = self.fluid[:, :-1] & self.fluid[:, 1:]
        oz = np.ones((self.nx, self.ny, self.nz + 1), bool)
        oz[:, :, 0] = False
        oz[:, :, -1] = False
        oz[:, :, 1:-1] = self.fluid[:, :, :-1] & self.fluid[:, :, 1:]
        self.open_x, self.open_y, self.open_z = ox, oy, oz

        # staggered velocity grids
        self.ux = np.zeros((self.nx + 1, self.ny, self.nz))
        self.uy = np.zeros((self.nx, self.ny + 1, self.nz))
        self.uz = np.zeros((self.nx, self.ny, self.nz + 1))

        # transported scalar (smoke density) from frozen initial state
        smoke = np.asarray(base.fields.get("transported_scalar"), float)
        if smoke.shape != shape:
            smoke = np.zeros(shape)
        smoke = np.clip(smoke, 0.0, self.SMOKE_MAX)
        smoke[self.solid] = 0.0
        # deterministic symmetry-breaking perturbation
        pert = self.rng.normal(0.0, 5e-3, size=shape)
        self.smoke = np.clip(smoke * (1.0 + pert), 0.0, self.SMOKE_MAX)

        # smoke emitter (frozen cylinder below the sphere)
        rz = np.sqrt((X - self.src_c[0]) ** 2 + (Y - self.src_c[1]) ** 2)
        zlo = self.src_c[2] - 0.5 * self.src_h
        zhi = self.src_c[2] + 0.5 * self.src_h
        self.source = np.where((rz < self.src_r) & (Z > zlo) & (Z < zhi) & self.fluid,
                               self.SOURCE_RATE, 0.0)

        # tracer particles (frozen sample cloud), passively advected
        self.ppos = np.array(base.position, float)
        self.pvel = np.zeros_like(self.ppos)
        self.pmass = np.array(base.mass, float)
        if base.body_id is not None:
            self.pbody = np.array(base.body_id, np.int64)
        else:
            self.pbody = np.zeros(len(self.ppos), np.int64)
        self.pattrs = {k: np.array(v) for k, v in base.attributes.items()}

        self._build_boundary_samples()

        self._lu_cache: dict[float, object] = {}
        self._b_idx = None
        self._dead = np.zeros(0, np.int64)
        self._max_speed = 0.0
        self.t = 0.0
        self._steps = 0

    # ------------------------------------------------------- boundary samples
    def _build_boundary_samples(self) -> None:
        pts, nrm = [], []
        c, r = self.sphere_c, self.sphere_r
        n_sphere = 96
        k = np.arange(n_sphere) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / n_sphere)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * k
        sx = np.cos(theta) * np.sin(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.cos(phi)
        pts.append(c + r * np.stack((sx, sy, sz), axis=1))
        nrm.append(np.stack((sx, sy, sz), axis=1))
        # ground top surface around the plume
        gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 5), np.linspace(-0.8, 0.8, 4))
        pts.append(np.stack((gx.ravel(), gy.ravel(),
                             np.full(gx.size, self.lo[2])), axis=1))
        nrm.append(np.tile([0.0, 0.0, 1.0], (gx.size, 1)))
        # domain walls
        for xw, nxw in ((self.lo[0], 1.0), (self.hi[0], -1.0)):
            a = np.full(16, xw)
            b = np.tile(np.linspace(-0.9, 0.9, 4), 4)
            z = np.repeat(np.linspace(0.3, 3.5, 4), 4)
            pts.append(np.stack((a, b, z), axis=1))
            nrm.append(np.tile([nxw, 0.0, 0.0], (16, 1)))
        for yw, nyw in ((self.lo[1], 1.0), (self.hi[1], -1.0)):
            a = np.tile(np.linspace(-2.8, 2.8, 4), 4)
            b = np.full(16, yw)
            z = np.repeat(np.linspace(0.3, 3.5, 4), 4)
            pts.append(np.stack((a, b, z), axis=1))
            nrm.append(np.tile([0.0, nyw, 0.0], (16, 1)))
        # top wall
        a = np.tile(np.linspace(-2.4, 2.4, 4), 4)
        b = np.repeat(np.linspace(-0.9, 0.9, 4), 4)
        pts.append(np.stack((a, b, np.full(16, self.hi[2])), axis=1))
        nrm.append(np.tile([0.0, 0.0, -1.0], (16, 1)))
        self.b_points = np.concatenate(pts)
        self.b_normals = np.concatenate(nrm)
        self.b_target = np.zeros_like(self.b_normals)
        # fluid-side sampling offset (a bit more than one cell)
        off = 1.2 * min(self.dx, self.dy, self.dz)
        self.b_sample_pts = self.b_points + self.b_normals * off

    # ---------------------------------------------------------- interpolation
    def _idx_center(self, pts):
        return [(pts[:, 0] - self.lo[0]) / self.dx - 0.5,
                (pts[:, 1] - self.lo[1]) / self.dy - 0.5,
                (pts[:, 2] - self.lo[2]) / self.dz - 0.5]

    def _velocity_at(self, pts):
        idx = self._idx_center(pts)
        vx = map_coordinates(self.ux, [idx[0] + 0.5, idx[1], idx[2]],
                             order=1, mode="nearest")
        vy = map_coordinates(self.uy, [idx[0], idx[1] + 0.5, idx[2]],
                             order=1, mode="nearest")
        vz = map_coordinates(self.uz, [idx[0], idx[1], idx[2] + 0.5],
                             order=1, mode="nearest")
        return np.stack((vx, vy, vz), axis=-1)

    # --------------------------------------------------------------- pressure
    def _factor(self, dt: float):
        key = round(float(dt), 12)
        lu = self._lu_cache.get(key)
        if lu is not None and self._b_idx is not None:
            return lu
        nx, ny, nz, rho = self.nx, self.ny, self.nz, self.RHO
        fluid = self.fluid
        b_idx = -np.ones((nx, ny, nz), np.int64)
        n = int(fluid.sum())
        b_idx[fluid] = np.arange(n)
        rows, cols, vals = [], [], []
        ox, oy, oz = self.open_x, self.open_y, self.open_z
        specs = (
            (ox[1:-1] & fluid[:-1] & fluid[1:],
             b_idx[:-1], b_idx[1:], self.dx),
            (ox[:, 1:-1] & fluid[:, :-1] & fluid[:, 1:],
             b_idx[:, :-1], b_idx[:, 1:], self.dy),
            (oz[:, :, 1:-1] & fluid[:, :, :-1] & fluid[:, :, 1:],
             b_idx[:, :, :-1], b_idx[:, :, 1:], self.dz),
        )
        for m, bi, bj, h in specs:
            ii = bi[m]
            jj = bj[m]
            if ii.size == 0:
                continue
            c = dt / (rho * h * h)
            rows += [ii, jj, ii, jj]
            cols += [ii, jj, jj, ii]
            vals += [np.full(ii.size, -c), np.full(ii.size, -c),
                     np.full(ii.size, c), np.full(ii.size, c)]
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        A.sum_duplicates()
        dead = np.where(A.diagonal() == 0.0)[0]
        if dead.size:
            A = A.tolil()
            for i in dead:
                A[int(i), int(i)] = 1.0
            A = A.tocsr()
        lu = splu(A.tocsc())
        self._lu_cache[key] = lu
        self._b_idx = b_idx
        self._dead = dead
        return lu

    def _project(self, dt: float) -> None:
        ux, uy, uz = self.ux, self.uy, self.uz
        ux[~self.open_x] = 0.0
        uy[~self.open_y] = 0.0
        uz[~self.open_z] = 0.0
        div = ((ux[1:] - ux[:-1]) / self.dx +
               (uy[:, 1:] - uy[:, :-1]) / self.dy +
               (uz[:, :, 1:] - uz[:, :, :-1]) / self.dz)
        lu = self._factor(dt)
        b = div[self.fluid].astype(float, copy=True)
        if self._dead.size:
            b[self._dead] = 0.0
        p = np.zeros(div.shape)
        p[self.fluid] = lu.solve(b)
        m = self.open_x[1:-1]
        self.ux[1:-1][m] -= (dt / self.RHO) * (p[1:] - p[:-1])[m] / self.dx
        m = self.open_y[:, 1:-1]
        self.uy[:, 1:-1][m] -= (dt / self.RHO) * (p[:, 1:] - p[:, :-1])[m] / self.dy
        m = self.open_z[:, :, 1:-1]
        self.uz[:, :, 1:-1][m] -= (dt / self.RHO) * (p[:, :, 1:] - p[:, :, :-1])[m] / self.dz
        self.ux[~self.open_x] = 0.0
        self.uy[~self.open_y] = 0.0
        self.uz[~self.open_z] = 0.0

    # --------------------------------------------------------------- advection
    def _advect_velocity(self, dt: float) -> None:
        grids = ((self.ux, self.Xfx, self.Yfx, self.Zfx, 0),
                 (self.uy, self.Xfy, self.Yfy, self.Zfy, 1),
                 (self.uz, self.Xfz, self.Yfz, self.Zfz, 2))
        new_comps = []
        for grid, Xf, Yf, Zf, axis in grids:
            pts = np.stack((Xf.ravel(), Yf.ravel(), Zf.ravel()), axis=1)
            u = self._velocity_at(pts)
            dep = pts - dt * u
            idx = self._idx_center(dep)
            idx[axis] = idx[axis] + 0.5
            new_comps.append(map_coordinates(grid, idx, order=1, mode="nearest"))
        self.ux = new_comps[0].reshape(self.ux.shape)
        self.uy = new_comps[1].reshape(self.uy.shape)
        self.uz = new_comps[2].reshape(self.uz.shape)
        self.ux[~self.open_x] = 0.0
        self.uy[~self.open_y] = 0.0
        self.uz[~self.open_z] = 0.0

    def _advect_scalar(self, dt: float) -> None:
        pts = np.stack((self.Xc.ravel(), self.Yc.ravel(), self.Zc.ravel()), axis=1)
        u = self._velocity_at(pts)
        dep = pts - dt * u
        new = map_coordinates(self.smoke, self._idx_center(dep), order=1, mode="nearest")
        self.smoke = np.clip(new.reshape(self.smoke.shape), 0.0, self.SMOKE_MAX)
        self.smoke[self.solid] = 0.0

    # ------------------------------------------------------------------ forces
    def _apply_forces(self, dt: float) -> None:
        # gravity-aligned buoyancy: upward force proportional to smoke density.
        # face[k'] (between cells k' and k'+1) maps to global face index k'+1,
        # i.e. exactly the interior face block uz[:, :, 1:-1].
        s = self.smoke
        face = 0.5 * (s[:, :, :-1] + s[:, :, 1:])
        self.uz[:, :, 1:-1] += dt * self.BUOYANCY * face
        # vorticity confinement (Fedkiw et al.)
        ucx = 0.5 * (self.ux[1:] + self.ux[:-1])
        ucy = 0.5 * (self.uy[:, 1:] + self.uy[:, :-1])
        ucz = 0.5 * (self.uz[:, :, 1:] + self.uz[:, :, :-1])
        wx = (np.gradient(ucz, self.dy, axis=1) -
              np.gradient(ucy, self.dz, axis=2))
        wy = (np.gradient(ucx, self.dz, axis=2) -
              np.gradient(ucz, self.dx, axis=0))
        wz = (np.gradient(ucy, self.dx, axis=0) -
              np.gradient(ucx, self.dy, axis=1))
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx = np.gradient(mag, self.dx, axis=0)
        gy = np.gradient(mag, self.dy, axis=1)
        gz = np.gradient(mag, self.dz, axis=2)
        gnorm = np.sqrt(gx * gx + gy * gy + gz * gz)
        safe = gnorm > 1e-12
        nxn = np.where(safe, gx / np.where(safe, gnorm, 1.0), 0.0)
        nyn = np.where(safe, gy / np.where(safe, gnorm, 1.0), 0.0)
        nzn = np.where(safe, gz / np.where(safe, gnorm, 1.0), 0.0)
        eps = self.VORTICITY_EPS * self.h_eq
        fx = eps * (nyn * wz - nzn * wy)
        fy = eps * (nzn * wx - nxn * wz)
        fz = eps * (nxn * wy - nyn * wx)
        fx[self.solid] = 0.0
        fy[self.solid] = 0.0
        fz[self.solid] = 0.0
        self.ux[1:-1] += dt * 0.5 * (fx[:-1] + fx[1:])
        self.uy[:, 1:-1] += dt * 0.5 * (fy[:, :-1] + fy[:, 1:])
        self.uz[:, :, 1:-1] += dt * 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])
        self.ux[~self.open_x] = 0.0
        self.uy[~self.open_y] = 0.0
        self.uz[~self.open_z] = 0.0

    def _apply_source(self, dt: float) -> None:
        self.smoke = np.minimum(self.smoke + dt * self.source, self.SMOKE_MAX)
        self.smoke *= float(np.exp(-self.SMOKE_DECAY * dt))
        self.smoke[self.solid] = 0.0

    # -------------------------------------------------------------- particles
    def _advect_particles(self, dt: float) -> None:
        if len(self.ppos) == 0:
            return
        u = self._velocity_at(self.ppos)
        self.pvel = u
        pos = self.ppos + dt * u
        # keep tracers out of the solid sphere and inside the domain
        d = pos - self.sphere_c
        dist = np.linalg.norm(d, axis=1)
        inside = dist < self.sphere_r + 1e-3
        if np.any(inside):
            dn = d[inside] / np.maximum(dist[inside], 1e-9)[:, None]
            pos[inside] = self.sphere_c + dn * (self.sphere_r + 2e-3)
        pad = 1e-3
        pos[:, 0] = np.clip(pos[:, 0], self.lo[0] + pad, self.hi[0] - pad)
        pos[:, 1] = np.clip(pos[:, 1], self.lo[1] + pad, self.hi[1] - pad)
        pos[:, 2] = np.clip(pos[:, 2], self.lo[2] + pad, self.hi[2] - pad)
        self.ppos = pos

    # ---------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        self._advect_velocity(dt)
        self._advect_scalar(dt)
        self._apply_forces(dt)
        self._apply_source(dt)
        self._project(dt)
        self._advect_particles(dt)
        self.t += dt
        self._steps += 1

    # ------------------------------------------------------------ frame output
    def _frame(self) -> StateFrame:
        ucx = 0.5 * (self.ux[1:] + self.ux[:-1])
        ucy = 0.5 * (self.uy[:, 1:] + self.uy[:, :-1])
        ucz = 0.5 * (self.uz[:, :, 1:] + self.uz[:, :, :-1])
        velfield = np.stack((ucx, ucy, ucz), axis=-1)
        speed = float(np.sqrt((velfield ** 2).sum(axis=-1)).max())
        self._max_speed = max(self._max_speed, speed)
        # fluid-side velocity at wall samples
        bvel = self._velocity_at(self.b_sample_pts)
        outside = ((self.b_sample_pts[:, 0] < self.lo[0]) |
                   (self.b_sample_pts[:, 0] > self.hi[0]) |
                   (self.b_sample_pts[:, 1] < self.lo[1]) |
                   (self.b_sample_pts[:, 1] > self.hi[1]) |
                   (self.b_sample_pts[:, 2] < self.lo[2]) |
                   (self.b_sample_pts[:, 2] > self.hi[2]))
        bvel[outside] = 0.0
        scalars = {k: float(v) for k, v in self._base.scalars.items()}
        scalars["grid_spacing"] = self.h_eq
        scalars["reference_velocity"] = max(1.0, speed)
        fields = {
            "active_mask": self.fluid.copy(),
            "transported_scalar": self.smoke.astype(float).copy(),
            "scalar_source": self.source.astype(float).copy(),
            "velocity_field": velfield,
            "boundary_normal": self.b_normals.copy(),
            "boundary_target_velocity": self.b_target.copy(),
            "boundary_velocity": bvel.astype(float),
        }
        return StateFrame(
            time=float(self.t),
            position=self.ppos.copy(),
            velocity=self.pvel.copy(),
            mass=self.pmass.copy(),
            body_id=self.pbody.copy(),
            attributes={k: v.copy() for k, v in self.pattrs.items()},
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------- public API
    def advance_to(self, time: float):
        target = float(time)
        if target <= self.t:
            return self._frame()
        nominal = self.sample_dt / self.N_SUBSTEPS
        while self.t < target - 1e-9:
            remaining = target - self.t
            n = int(np.ceil(remaining / nominal - 1e-9))
            n = max(1, min(n, self.N_SUBSTEPS))
            dt = remaining / n
            self._step(dt)
        self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mac_grid_semi_lagrangian_smoke",
            "substeps": self._steps,
            "grid": [self.nx, self.ny, self.nz],
            "max_speed": float(self._max_speed),
        }
