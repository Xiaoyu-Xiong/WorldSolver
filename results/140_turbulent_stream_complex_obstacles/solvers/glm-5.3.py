"""Staggered-grid Eulerian smoke solver for the turbulent-stream scene.

Physical model
--------------
Incompressible Navier-Stokes (constant density gas) solved on a staggered
MAC grid matching the frozen voxel-grid diagnostics shape:

    du/dt + (u . grad) u = -grad p / rho + f ,   div u = 0

with
  * semi-Lagrangian (RK2 backtrace) advection of velocity and smoke,
  * implicit pressure projection through a sparse LU factorization of the
    static discrete Poisson operator (obstacles frozen -> factor once),
  * free-slip solid boundaries for the two complex obstacle boxes, the
    ground plane and the nozzle backing,
  * prescribed inflow velocity profile at the nozzle exit plane with a
    smooth deterministic unsteady perturbation,
  * open (zero-pressure) outflow at the downstream, lateral and top
    boundaries,
  * vorticity confinement to sustain the turbulent wake eddies,
  * buoyant force on the transported smoke concentration,
  * passive tracer particles carried by the flow (respawned at the nozzle
    once they leave the domain or enter a solid) for the [N,3] state.

No keyframes, no tabulated motion: every state is produced by stepping the
discretized equations with a fixed substep.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
SUB_DT = 1.0 / 120.0            # physics substep (4 per exported sample)
U0 = 1.7                        # nozzle exit speed [m/s]
BUOYANCY = 1.2                  # buoyant acceleration per unit concentration
VORT_EPS = 0.40                 # vorticity confinement coefficient
DIFFUSIVITY = 0.01              # scalar diffusivity [m^2/s]
SMOKE_MAX = 1.5                 # concentration clamp
JET_RADIUS = 0.42               # jet radius (>= nozzle radius 0.4)
JET_YC, JET_ZC = 0.0, 1.0       # nozzle axis intercept (y, z)
NOZZLE_EXIT_X = -2.63           # nozzle front face x

DOMAIN = ((-3.0, 3.0), (-1.2, 1.2), (0.0, 3.8))
GRID_SHAPE = (24, 16, 28)

# frozen static obstacles: (center, half_extents)
OBSTACLES = (
    ((-0.65, 0.0, 1.25), (0.38, 0.38, 0.38)),
    ((0.65, 0.0, 2.15), (0.42, 0.42, 0.42)),
)
GROUND_TOP_Z = 0.0

MAX_VEL = 12.0  # safety clamp


def _trilinear(f, ax, ay, az, x, y, z):
    """Trilinear sample of f (on coordinate axes ax, ay, az) at points."""
    nx, ny, nz = f.shape
    dx, dy, dz = ax[1] - ax[0], ay[1] - ay[0], az[1] - az[0]
    fx = np.clip((x - ax[0]) / dx, 0.0, nx - 1.0)
    fy = np.clip((y - ay[0]) / dy, 0.0, ny - 1.0)
    fz = np.clip((z - az[0]) / dz, 0.0, nz - 1.0)
    i0 = np.minimum(fx.astype(np.intp), nx - 2)
    j0 = np.minimum(fy.astype(np.intp), ny - 2)
    k0 = np.minimum(fz.astype(np.intp), nz - 2)
    a, b, c = fx - i0, fy - j0, fz - k0
    a1, b1, c1 = 1.0 - a, 1.0 - b, 1.0 - c
    v000 = f[i0, j0, k0]
    v100 = f[i0 + 1, j0, k0]
    v010 = f[i0, j0 + 1, k0]
    v110 = f[i0 + 1, j0 + 1, k0]
    v001 = f[i0, j0, k0 + 1]
    v101 = f[i0 + 1, j0, k0 + 1]
    v011 = f[i0, j0 + 1, k0 + 1]
    v111 = f[i0 + 1, j0 + 1, k0 + 1]
    return (v000 * a1 * b1 * c1 + v100 * a * b1 * c1 +
            v010 * a1 * b * c1 + v110 * a * b * c1 +
            v001 * a1 * b1 * c + v101 * a * b1 * c +
            v011 * a1 * b * c + v111 * a * b * c)


def _box_face_samples(center, half, n=3):
    """Sample points + outward normals on an axis-aligned box surface."""
    cx, cy, cz = center
    hx, hy, hz = half
    us = np.linspace(-0.98, 0.98, n)
    pts, nrm = [], []
    for a in us:
        for b in us:
            pts += [
                (cx - hx, cy + a * hy, cz + b * hz), (-1.0, 0.0, 0.0),
                (cx + hx, cy + a * hy, cz + b * hz), (1.0, 0.0, 0.0),
                (cx + a * hx, cy - hy, cz + b * hz), (0.0, -1.0, 0.0),
                (cx + a * hx, cy + hy, cz + b * hz), (0.0, 1.0, 0.0),
                (cx + a * hx, cy + b * hy, cz - hz), (0.0, 0.0, -1.0),
                (cx + a * hx, cy + b * hy, cz + hz), (0.0, 0.0, 1.0),
            ]
    arr = np.asarray(pts, float).reshape(-1, 2, 3)
    return arr[:, 0, :], arr[:, 1, :]


class BenchmarkSolver:
    """MAC-grid incompressible smoke solver."""

    # ------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        sampling = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
        shape = sampling.get("field_grid_shape", list(GRID_SHAPE))
        self.nx, self.ny, self.nz = (int(v) for v in shape)

        self.xs = np.linspace(DOMAIN[0][0], DOMAIN[0][1], self.nx)
        self.ys = np.linspace(DOMAIN[1][0], DOMAIN[1][1], self.ny)
        self.zs = np.linspace(DOMAIN[2][0], DOMAIN[2][1], self.nz)
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        self.dz = self.zs[1] - self.zs[0]
        self.xf = self.xs[0] - 0.5 * self.dx + self.dx * np.arange(self.nx + 1)
        self.yf = self.ys[0] - 0.5 * self.dy + self.dy * np.arange(self.ny + 1)
        self.zf = self.zs[0] - 0.5 * self.dz + self.dz * np.arange(self.nz + 1)

        # ---- solid mask (frozen geometry)
        X, Y, Z = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        solid = (Z <= GROUND_TOP_Z + 1e-9) | (X <= NOZZLE_EXIT_X)
        for center, half in OBSTACLES:
            solid |= ((np.abs(X - center[0]) <= half[0] + 1e-9) &
                      (np.abs(Y - center[1]) <= half[1] + 1e-9) &
                      (np.abs(Z - center[2]) <= half[2] + 1e-9))
        self.solid = solid
        self.fluid = ~solid

        # inflow face: first x-face whose upstream cell column is solid
        self.inflow_i = 1
        for i in range(1, self.nx):
            if self.fluid[i].any() and not self.fluid[i - 1].any():
                self.inflow_i = i
                break

        # ---- face classification (fixed / pressure-corrected)
        f = self.fluid
        nx, ny, nz = self.nx, self.ny, self.nz
        self.fixed_u = np.zeros((nx + 1, ny, nz), bool)
        self.corr_u = np.zeros((nx + 1, ny, nz), bool)
        for i in range(nx + 1):
            if i == 0:
                self.fixed_u[i] = True
            elif i == nx:
                self.fixed_u[i] = ~f[nx - 1]
                self.corr_u[i] = f[nx - 1]
            else:
                self.fixed_u[i] = ~(f[i - 1] & f[i])
                self.corr_u[i] = f[i - 1] & f[i]
        self.fixed_v = np.zeros((nx, ny + 1, nz), bool)
        self.corr_v = np.zeros((nx, ny + 1, nz), bool)
        for j in range(ny + 1):
            if j == 0:
                self.corr_v[:, 0, :] = f[:, 0, :]
            elif j == ny:
                self.corr_v[:, ny, :] = f[:, ny - 1, :]
            else:
                self.fixed_v[:, j, :] = ~(f[:, j - 1, :] & f[:, j, :])
                self.corr_v[:, j, :] = f[:, j - 1, :] & f[:, j, :]
        self.fixed_w = np.zeros((nx, ny, nz + 1), bool)
        self.corr_w = np.zeros((nx, ny, nz + 1), bool)
        for k in range(nz + 1):
            if k == 0:
                self.fixed_w[:, :, 0] = True               # ground wall
            elif k == nz:
                self.corr_w[:, :, nz] = f[:, :, nz - 1]    # open top
            else:
                self.fixed_w[:, :, k] = ~(f[:, :, k - 1] & f[:, :, k])
                self.corr_w[:, :, k] = f[:, :, k - 1] & f[:, :, k]

        # ---- pressure operator (static geometry -> factor once)
        self._build_pressure()

        # ---- state
        self.u = np.zeros((nx + 1, ny, nz))
        self.v = np.zeros((nx, ny + 1, nz))
        self.w = np.zeros((nx, ny, nz + 1))
        self.t = 0.0

        init = scene.initial_state()
        self.pos = np.array(init.position, float).copy()
        self.pvel = np.array(init.velocity, float).copy()
        self.mass = None if init.mass is None else np.array(init.mass, float).copy()
        self.labels = (None if init.body_id is None
                       else np.array(init.body_id).copy())
        self.attributes = {k: np.array(a).copy()
                           for k, a in (init.attributes or {}).items()}
        self.init_scalars = dict(init.scalars or {})

        sc = init.fields.get("transported_scalar") if init.fields else None
        if sc is not None and np.asarray(sc).shape == (nx, ny, nz):
            self.smoke = np.array(sc, float).copy()
        else:
            self.smoke = np.zeros((nx, ny, nz))
        self.smoke[solid] = 0.0

        # prescribed smoke source in the nozzle cells
        self.source = np.zeros((nx, ny, nz))
        Yc, Zc = np.meshgrid(self.ys, self.zs, indexing="ij")
        prof = np.clip(1.0 - (((Yc - JET_YC) / JET_RADIUS) ** 2 +
                              ((Zc - JET_ZC) / JET_RADIUS) ** 2), 0.0, 1.0)
        for i in range(self.inflow_i, min(self.inflow_i + 2, nx)):
            rate = 5.0 if i == self.inflow_i else 2.5
            self.source[i] = rate * prof * self.fluid[i]

        # ---- boundary samples (static walls)
        pts, nrm = [], []
        for center, half in OBSTACLES:
            p, n = _box_face_samples(center, half, n=3)
            pts.append(p)
            nrm.append(n)
        gx = np.linspace(DOMAIN[0][0] + 0.1, DOMAIN[0][1] - 0.1, 9)
        gy = np.linspace(DOMAIN[1][0] + 0.1, DOMAIN[1][1] - 0.1, 5)
        for a in gx:
            for b in gy:
                pts.append([[a, b, GROUND_TOP_Z]])
                nrm.append([[0.0, 0.0, 1.0]])
        self.bpts = np.concatenate(pts, axis=0)
        self.bnrm = np.concatenate(nrm, axis=0)

        # ---- cached meshes for advection backtraces
        self._mesh_c = self._mesh(self.xs, self.ys, self.zs)
        self._mesh_u = self._mesh(self.xf, self.ys, self.zs)
        self._mesh_v = self._mesh(self.xs, self.yf, self.zs)
        self._mesh_w = self._mesh(self.xs, self.ys, self.zf)

        # ---- tracer respawn seeds on the nozzle disk (deterministic)
        sp = [[self.xf[self.inflow_i] + 0.03, JET_YC, JET_ZC]]
        for r, cnt in ((0.13, 6), (0.25, 10), (0.34, 12)):
            for m in range(cnt):
                ang = 2.0 * np.pi * m / cnt + 0.3 * r
                sp.append([self.xf[self.inflow_i] + 0.03,
                           JET_YC + r * np.cos(ang), JET_ZC + r * np.sin(ang)])
        self.spawn = np.asarray(sp, float)
        self._spawn_cursor = 0

        self._step_count = 0
        self._max_speed_seen = 0.0

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _mesh(ax, ay, az):
        X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
        return X.ravel(), Y.ravel(), Z.ravel()

    def _vel_at(self, x, y, z):
        u = _trilinear(self.u, self.xf, self.ys, self.zs, x, y, z)
        v = _trilinear(self.v, self.xs, self.yf, self.zs, x, y, z)
        w = _trilinear(self.w, self.xs, self.ys, self.zf, x, y, z)
        return u, v, w

    def _advect(self, field, mesh, axes, dt):
        gx, gy, gz = axes
        X, Y, Z = mesh
        u, v, w = self._vel_at(X, Y, Z)
        mx, my, mz = X - 0.5 * dt * u, Y - 0.5 * dt * v, Z - 0.5 * dt * w
        u2, v2, w2 = self._vel_at(mx, my, mz)
        px, py, pz = X - dt * u2, Y - dt * v2, Z - dt * w2
        return _trilinear(field, gx, gy, gz, px, py, pz).reshape(field.shape)

    def _center_velocity(self):
        U = 0.5 * (self.u[:-1] + self.u[1:])
        V = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        W = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return U, V, W

    def _inflow(self, t):
        """Nozzle exit velocity profile with deterministic unsteadiness."""
        amp = 0.07 * U0
        Y, Z = np.meshgrid(self.ys, self.zs, indexing="ij")
        pu = np.clip(1.0 - (((Y - JET_YC) / JET_RADIUS) ** 2 +
                            ((Z - JET_ZC) / JET_RADIUS) ** 2), 0.0, 1.0)
        u = U0 * pu
        Yv, Zv = np.meshgrid(self.yf, self.zs, indexing="ij")
        pv = np.clip(1.0 - (((Yv - JET_YC) / JET_RADIUS) ** 2 +
                            ((Zv - JET_ZC) / JET_RADIUS) ** 2), 0.0, 1.0)
        v = amp * np.sin(2.3 * Yv + 1.1 * Zv + 2.0 * t) * pv
        Yw, Zw = np.meshgrid(self.ys, self.zf, indexing="ij")
        pw = np.clip(1.0 - (((Yw - JET_YC) / JET_RADIUS) ** 2 +
                            ((Zw - JET_ZC) / JET_RADIUS) ** 2), 0.0, 1.0)
        w = amp * np.cos(1.7 * Yw + 2.3 * Zw + 1.4 * t) * pw
        return u, v, w

    # ------------------------------------------------------------ pressure
    def _build_pressure(self):
        nx, ny, nz, f = self.nx, self.ny, self.nz, self.fluid
        nf = int(f.sum())
        self.nf = nf
        fid = np.full((nx, ny, nz), -1, np.int64)
        fid[f] = np.arange(nf)
        wx, wy, wz = 1.0 / self.dx ** 2, 1.0 / self.dy ** 2, 1.0 / self.dz ** 2
        fi, fj, fk = np.where(f)
        rows, cols, vals = [], [], []
        diag = np.zeros(nf)
        for a in range(nf):
            i, j, k = int(fi[a]), int(fj[a]), int(fk[a])
            d = 0.0
            if i > 0 and f[i - 1, j, k]:
                rows.append(a); cols.append(fid[i - 1, j, k]); vals.append(-wx); d += wx
            if i + 1 < nx:
                if f[i + 1, j, k]:
                    rows.append(a); cols.append(fid[i + 1, j, k]); vals.append(-wx); d += wx
            else:
                d += wx                                    # open outflow (p = 0)
            if j == 0:
                d += wy                                    # open side
            elif f[i, j - 1, k]:
                rows.append(a); cols.append(fid[i, j - 1, k]); vals.append(-wy); d += wy
            if j + 1 == ny:
                d += wy
            elif f[i, j + 1, k]:
                rows.append(a); cols.append(fid[i, j + 1, k]); vals.append(-wy); d += wy
            if k > 0 and f[i, j, k - 1]:
                rows.append(a); cols.append(fid[i, j, k - 1]); vals.append(-wz); d += wz
            if k + 1 == nz:
                d += wz                                    # open top
            elif f[i, j, k + 1]:
                rows.append(a); cols.append(fid[i, j, k + 1]); vals.append(-wz); d += wz
            diag[a] = d
        rows.extend(range(nf)); cols.extend(range(nf)); vals.extend(diag.tolist())
        A = coo_matrix((np.asarray(vals, float),
                        (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
                       shape=(nf, nf)).tocsc()
        try:
            self.lu = splu(A)
        except Exception:
            reg = coo_matrix((1e-9 * np.ones(nf), (np.arange(nf), np.arange(nf))),
                             shape=(nf, nf)).tocsc()
            self.lu = splu(A + reg)

    def _project(self, dt):
        nx, ny, nz = self.nx, self.ny, self.nz
        div = ((self.u[1:] - self.u[:-1]) / self.dx +
               (self.v[:, 1:] - self.v[:, :-1]) / self.dy +
               (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz)
        rhs = np.ascontiguousarray(div[self.fluid] / dt, dtype=float)
        p = self.lu.solve(rhs)
        P = np.zeros((nx, ny, nz))
        P[self.fluid] = p

        du = np.zeros_like(self.u)
        du[1:nx] = -(P[1:] - P[:-1]) / self.dx
        du[nx] = P[nx - 1] / self.dx
        dv = np.zeros_like(self.v)
        dv[:, 1:ny] = -(P[:, 1:] - P[:, :-1]) / self.dy
        dv[:, 0] = -P[:, 0] / self.dy
        dv[:, ny] = P[:, ny - 1] / self.dy
        dw = np.zeros_like(self.w)
        dw[:, :, 1:nz] = -(P[:, :, 1:] - P[:, :, :-1]) / self.dz
        dw[:, :, nz] = P[:, :, nz - 1] / self.dz
        self.u += dt * du * self.corr_u
        self.v += dt * dv * self.corr_v
        self.w += dt * dw * self.corr_w

    # ------------------------------------------------------------ physics
    def _forces(self, dt):
        U, V, W = self._center_velocity()
        C = self.smoke
        wz_f = np.concatenate([C[:1], 0.5 * (C[:-1] + C[1:]), C[-1:]], axis=2)
        self.w += dt * BUOYANCY * wz_f
        # vorticity confinement
        ox = np.gradient(W, self.dy, axis=1) - np.gradient(V, self.dz, axis=2)
        oy = np.gradient(U, self.dz, axis=2) - np.gradient(W, self.dx, axis=0)
        oz = np.gradient(V, self.dx, axis=0) - np.gradient(U, self.dy, axis=1)
        mag = np.sqrt(ox * ox + oy * oy + oz * oz)
        gx = np.gradient(mag, self.dx, axis=0)
        gy = np.gradient(mag, self.dy, axis=1)
        gz = np.gradient(mag, self.dz, axis=2)
        nm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
        Nx, Ny, Nz = gx / nm, gy / nm, gz / nm
        fx = (Ny * oz - Nz * oy) * self.fluid * VORT_EPS
        fy = (Nz * ox - Nx * oz) * self.fluid * VORT_EPS
        fz = (Nx * oy - Ny * ox) * self.fluid * VORT_EPS
        fxp = np.concatenate([fx[:1], 0.5 * (fx[:-1] + fx[1:]), fx[-1:]], axis=0)
        fyp = np.concatenate([fy[:, :1], 0.5 * (fy[:, :-1] + fy[:, 1:]),
                              fy[:, -1:]], axis=1)
        fzp = np.concatenate([fz[:, :, :1], 0.5 * (fz[:, :, :-1] + fz[:, :, 1:]),
                              fz[:, :, -1:]], axis=2)
        self.u += dt * fxp
        self.v += dt * fyp
        self.w += dt * fzp

    def _apply_boundaries(self, t):
        self.u[self.fixed_u] = 0.0
        self.v[self.fixed_v] = 0.0
        self.w[self.fixed_w] = 0.0
        iu = self.inflow_i
        if 0 < iu <= self.nx:
            ui, vi, wi = self._inflow(t)
            self.u[iu] = ui
            self.v[iu] += vi
            self.w[iu] += wi

    def _advect_smoke(self, dt):
        s = self._advect(self.smoke, self._mesh_c, (self.xs, self.ys, self.zs), dt)
        lap = ((np.roll(s, -1, 0) + np.roll(s, 1, 0) - 2 * s) / self.dx ** 2 +
               (np.roll(s, -1, 1) + np.roll(s, 1, 1) - 2 * s) / self.dy ** 2 +
               (np.roll(s, -1, 2) + np.roll(s, 1, 2) - 2 * s) / self.dz ** 2)
        s = s + dt * DIFFUSIVITY * lap * self.fluid
        s = s + dt * self.source
        np.clip(s, 0.0, SMOKE_MAX, out=s)
        s[self.solid] = 0.0
        self.smoke = s

    def _inside_solid(self, x, y, z):
        i = np.clip(np.round((x - self.xs[0]) / self.dx).astype(np.intp),
                    0, self.nx - 1)
        j = np.clip(np.round((y - self.ys[0]) / self.dy).astype(np.intp),
                    0, self.ny - 1)
        k = np.clip(np.round((z - self.zs[0]) / self.dz).astype(np.intp),
                    0, self.nz - 1)
        return self.solid[i, j, k]

    def _update_tracers(self, dt):
        p = self.pos
        u1, v1, w1 = self._vel_at(p[:, 0], p[:, 1], p[:, 2])
        pm = np.column_stack([p[:, 0] + 0.5 * dt * u1,
                              p[:, 1] + 0.5 * dt * v1,
                              p[:, 2] + 0.5 * dt * w1])
        u2, v2, w2 = self._vel_at(pm[:, 0], pm[:, 1], pm[:, 2])
        p = np.column_stack([p[:, 0] + dt * u2,
                             p[:, 1] + dt * v2,
                             p[:, 2] + dt * w2])
        x0, x1 = self.xf[0] + 0.02, self.xf[-1] - 0.02
        y0, y1 = self.yf[0] + 0.02, self.yf[-1] - 0.02
        z0, z1 = self.zf[0] + 0.02, self.zf[-1] - 0.02
        bad = ((p[:, 0] < x0) | (p[:, 0] > x1) | (p[:, 1] < y0) | (p[:, 1] > y1) |
               (p[:, 2] < z0) | (p[:, 2] > z1) |
               self._inside_solid(p[:, 0], p[:, 1], p[:, 2]))
        if bad.any():
            for m in np.where(bad)[0]:
                p[m] = self.spawn[self._spawn_cursor % len(self.spawn)]
                self._spawn_cursor += 1
        self.pos = p
        u, v, w = self._vel_at(p[:, 0], p[:, 1], p[:, 2])
        self.pvel = np.column_stack([u, v, w])

    def _step(self, dt, t):
        self._forces(dt)
        self.u = self._advect(self.u, self._mesh_u, (self.xf, self.ys, self.zs), dt)
        self.v = self._advect(self.v, self._mesh_v, (self.xs, self.yf, self.zs), dt)
        self.w = self._advect(self.w, self._mesh_w, (self.xs, self.ys, self.zf), dt)
        self._apply_boundaries(t)
        self._project(dt)
        self._apply_boundaries(t)
        np.clip(self.u, -MAX_VEL, MAX_VEL, out=self.u)
        np.clip(self.v, -MAX_VEL, MAX_VEL, out=self.v)
        np.clip(self.w, -MAX_VEL, MAX_VEL, out=self.w)
        self._advect_smoke(dt)
        self._update_tracers(dt)
        self._step_count += 1

    # ------------------------------------------------------------ output
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 100000:
            dt = min(SUB_DT, target - self.t)
            self._step(dt, self.t + 0.5 * dt)
            self.t = min(self.t + dt, target)
            guard += 1
        return self._frame()

    def _frame(self):
        U, V, W = self._center_velocity()
        U = U * self.fluid
        V = V * self.fluid
        W = W * self.fluid
        vel_field = np.stack([U, V, W], axis=-1)
        speed = np.sqrt((vel_field ** 2).sum(axis=-1))
        self._max_speed_seen = max(self._max_speed_seen, float(speed.max()))

        bx, by, bz = self.bpts[:, 0], self.bpts[:, 1], self.bpts[:, 2]
        bu = _trilinear(U, self.xs, self.ys, self.zs, bx, by, bz)
        bv = _trilinear(V, self.xs, self.ys, self.zs, bx, by, bz)
        bw = _trilinear(W, self.xs, self.ys, self.zs, bx, by, bz)
        bvel = np.column_stack([bu, bv, bw])

        fields = {
            "active_mask": self.fluid.copy(),
            "transported_scalar": self.smoke.copy(),
            "scalar_source": self.source.copy(),
            "velocity_field": vel_field,
            "boundary_normal": self.bnrm.copy(),
            "boundary_target_velocity": np.zeros_like(self.bnrm),
            "boundary_velocity": bvel,
        }
        scalars = {
            "characteristic_length": float(self.init_scalars.get(
                "characteristic_length", 1.0)),
            "grid_spacing": float(self.init_scalars.get("grid_spacing", 0.18)),
            "reference_velocity": float(U0),
            "contact_length_scale": float(self.init_scalars.get(
                "contact_length_scale", 0.18)),
        }
        return StateFrame(
            time=float(self.t),
            position=self.pos.copy(),
            velocity=self.pvel.copy(),
            mass=None if self.mass is None else self.mass.copy(),
            body_id=None if self.labels is None else self.labels.copy(),
            attributes={k: np.array(v).copy() for k, v in self.attributes.items()},
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "staggered-mac-euler-smoke",
            "substeps": int(self._step_count),
            "sub_dt": SUB_DT,
            "inlet_velocity": U0,
            "max_center_speed_seen": float(self._max_speed_seen),
            "smoke_mass": float(self.smoke.sum()),
            "time": float(self.t),
        }
