"""Buoyant smoke rising around a fixed sphere.

Physics: incompressible Euler/Navier-Stokes on a collocated diagnostic grid
(24x16x28) matching the frozen scene voxel grid, with

* gravity-aligned buoyancy force  f = alpha * smoke * (+z),
* continuous smoke emitter (cylinder r<=0.32, z<=0.1) below the sphere,
* MacCormack semi-Lagrangian advection of smoke density and velocity,
* vorticity confinement to preserve wake roll-up on the coarse grid,
* pressure projection (sparse LU, factorized once) with rigid no-penetration
  on the ground, the four side walls and the obstacle sphere, and an open
  (Dirichlet p=0) outflow at the top,
* 64 passive tracer particles advected with the fluid for state.position.

All required StateFrame fields are exported every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


def _trilin(f, x, y, z, gx, gy, gz, hx, hy, hz):
    """Sample grid field f (..., D optional) at world positions (linear interp)."""
    nx, ny, nz = f.shape[:3]
    fx = np.clip((x - gx[0]) / hx, 0.0, nx - 1.0001)
    fy = np.clip((y - gy[0]) / hy, 0.0, ny - 1.0001)
    fz = np.clip((z - gz[0]) / hz, 0.0, nz - 1.0001)
    i0 = fx.astype(int); j0 = fy.astype(int); k0 = fz.astype(int)
    i1 = np.minimum(i0 + 1, nx - 1); j1 = np.minimum(j0 + 1, ny - 1); k1 = np.minimum(k0 + 1, nz - 1)
    wx = fx - i0; wy = fy - j0; wz = fz - k0
    out = 0.0
    for di, (ia, wxx) in enumerate(((i0, 1.0 - wx), (i1, wx))):
        for dj, (ja, wyy) in enumerate(((j0, 1.0 - wy), (j1, wy))):
            for dk, (ka, wzz) in enumerate(((k0, 1.0 - wz), (k1, wz))):
                out = out + f[ia, ja, ka] * (wxx * wyy * wzz)[..., None] if f.ndim == 4 \
                    else out + f[ia, ja, ka] * (wxx * wyy * wzz)
    return out


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        frame0 = scene.initial_state()
        self.frame0 = frame0
        cfg = scene.config
        self.duration = float(cfg["duration"])

        shape = tuple(int(n) for n in cfg["sampling"]["field_grid_shape"])
        nx, ny, nz = shape
        self.shape = shape
        self.gx = np.linspace(-3.0, 3.0, nx)
        self.gy = np.linspace(-1.2, 1.2, ny)
        self.gz = np.linspace(0.0, 3.8, nz)
        self.hx = self.gx[1] - self.gx[0]
        self.hy = self.gy[1] - self.gy[0]
        self.hz = self.gz[1] - self.gz[0]

        X, Y, Z = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z

        # --- geometry masks -------------------------------------------------
        sc, sr = np.array([0.0, 0.0, 1.35]), 0.48
        self.sc, self.sr = sc, sr
        d = np.sqrt((X - sc[0]) ** 2 + (Y - sc[1]) ** 2 + (Z - sc[2]) ** 2)
        self.solid = d < (sr + 0.35 * min(self.hx, self.hy, self.hz))
        self.sph_n = np.zeros_like(X)  # unit vector cell -> sphere center dir
        with np.errstate(invalid="divide", divide="ignore"):
            inv = np.where(d > 1e-12, 1.0 / np.maximum(d, 1e-12), 0.0)
        self.sph_nx = (X - sc[0]) * inv
        self.sph_ny = (Y - sc[1]) * inv
        self.sph_nz = (Z - sc[2]) * inv

        # emitter cylinder: r <= 0.32, 0 <= z <= 0.1
        rxy = np.sqrt(X ** 2 + Y ** 2)
        self.emitter = (rxy <= 0.34) & (Z <= 0.15) & (~self.solid)

        self.fluid = ~self.solid
        self.active_mask = self.fluid.copy()

        # neighbours-adjacent-to-solid flag and inward normal
        adj = np.zeros(shape, bool)
        for ax in range(3):
            adj |= np.roll(self.solid, 1, ax) | np.roll(self.solid, -1, ax)
        self.adj_solid = adj & self.fluid

        # --- fields ----------------------------------------------------------
        self.s = np.array(frame0.fields["transported_scalar"], float)
        self.s[self.solid] = 0.0
        self.u = np.zeros(shape + (3,))

        src_strength = 1.6
        self.scalar_source = np.where(self.emitter, src_strength, 0.0)
        self.src_strength = src_strength

        # --- pressure operator (constant: static geometry) -------------------
        idx = -np.ones(shape, int)
        fluid_idx = np.flatnonzero(self.fluid.ravel())
        idx.ravel()[fluid_idx] = np.arange(len(fluid_idx))
        self.pidx = idx
        self.nu = len(fluid_idx)
        top = nz - 1
        A = lil_matrix((self.nu, self.nu))
        ii, jj, kk = np.unravel_index(fluid_idx, shape)
        for c in range(self.nu):
            i, j, k = ii[c], jj[c], kk[c]
            if k == top:  # open outflow: Dirichlet p = 0
                A[c, c] = 1.0
                continue
            diag = 0.0
            for (di, dj, dk, h) in ((1, 0, 0, self.hx), (-1, 0, 0, self.hx),
                                    (0, 1, 0, self.hy), (0, -1, 0, self.hy),
                                    (0, 0, 1, self.hz), (0, 0, -1, self.hz)):
                i2, j2, k2 = i + di, j + dj, k + dk
                out = (i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny or k2 < 0 or k2 >= nz)
                if out or self.solid[i2, j2, k2]:
                    diag += 1.0 / h ** 2  # Neumann: drop off-diagonal
                else:
                    c2 = idx[i2, j2, k2]
                    A[c, c2] = -1.0 / h ** 2
                    diag += 1.0 / h ** 2
            A[c, c] = diag
        self.lufs = splu(A.tocsc())

        # --- boundary samples (walls + sphere + sides) -----------------------
        bpos, bnorm = [], []
        # sphere surface (fibonacci)
        nb = 220
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for m in range(nb):
            zz = 1.0 - 2.0 * (m + 0.5) / nb
            rr = np.sqrt(max(0.0, 1.0 - zz * zz))
            th = golden * m
            n = np.array([rr * np.cos(th), rr * np.sin(th), zz])
            bpos.append(sc + (sr + 0.02) * n)
            bnorm.append(n)
        # ground top (z = 0)
        for gxv in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for gyv in (-0.8, 0.0, 0.8):
                bpos.append([gxv, gyv, 0.0]); bnorm.append([0.0, 0.0, 1.0])
        # side walls x = +-3, y = +-1.2
        for zv in (0.8, 1.8, 2.8):
            bpos += [[-3.0, 0.0, zv], [3.0, 0.0, zv]]
            bnorm += [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            bpos += [[0.0, -1.2, zv], [0.0, 1.2, zv]]
            bnorm += [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        self.bpos = np.array(bpos)
        self.boundary_normal = np.array(bnorm)
        self.boundary_target_velocity = np.zeros_like(self.bpos)

        # --- particles (tracers) ---------------------------------------------
        self.p0 = np.array(frame0.position, float)
        self.pos = self.p0.copy()
        self.pvel = np.array(frame0.velocity, float)
        self.mass = np.array(frame0.mass, float)
        self.body_id = np.array(frame0.body_id)
        self.attrs = {k: np.array(v, float) for k, v in frame0.attributes.items()}
        self.cum_src = np.zeros(len(self.pos))
        self.scalars = dict(frame0.scalars)

        self.t = 0.0
        self.sub_dt = float(cfg["sample_dt"]) / 4.0
        self.alpha_buoy = 4.5
        self.eps_vc = 4.0
        self.h_vc = min(self.hx, self.hy, self.hz)

    # ------------------------------------------------------------------ utils
    def _sample_u(self, x, y, z):
        return _trilin(self.u, x, y, z, self.gx, self.gy, self.gz, self.hx, self.hy, self.hz)

    def _sample_s(self, x, y, z):
        return _trilin(self.s[..., None], x, y, z, self.gx, self.gy, self.gz,
                       self.hx, self.hy, self.hz)[..., 0]

    def _advect_mc(self, f, dt):
        """MacCormack semi-Lagrangian advection of f by self.u with limiter."""
        u = self.u
        gx, gy, gz = self.gx, self.gy, self.gz
        xd = self.X - dt * u[..., 0]
        yd = self.Y - dt * u[..., 1]
        zd = self.Z - dt * u[..., 2]
        if f.ndim == 4:
            f1 = _trilin(f, xd, yd, zd, gx, gy, gz, self.hx, self.hy, self.hz)
        else:
            f1 = _trilin(f[..., None], xd, yd, zd, gx, gy, gz,
                         self.hx, self.hy, self.hz)[..., 0]
        uf = _trilin(u, xd, yd, zd, gx, gy, gz, self.hx, self.hy, self.hz)
        xf = xd + dt * uf[..., 0]
        yf = yd + dt * uf[..., 1]
        zf = zd + dt * uf[..., 2]
        if f.ndim == 4:
            f2 = _trilin(f1, xf, yf, zf, gx, gy, gz, self.hx, self.hy, self.hz)
            fnew = f1 + 0.5 * (f - f2)
            mn, mx = self._local_minmax(f)
            return np.clip(fnew, mn, mx)
        f2 = _trilin(f1[..., None], xf, yf, zf, gx, gy, gz,
                     self.hx, self.hy, self.hz)[..., 0]
        fnew = f1 + 0.5 * (f - f2)
        mn, mx = self._local_minmax(f)
        return np.clip(fnew, mn, mx)

    @staticmethod
    def _local_minmax(f):
        mn = f.copy(); mx = f.copy()
        for ax in range(3):
            mn = np.minimum.reduce([np.roll(mn, 1, ax), mn, np.roll(mn, -1, ax)])
            mx = np.maximum.reduce([np.roll(mx, 1, ax), mx, np.roll(mx, -1, ax)])
        return mn, mx

    def _enforce_boundaries(self):
        u = self.u
        u[self.solid] = 0.0
        u[0, :, :, 0] = 0.0; u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0; u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        # remove inward (penetrating) normal component in cells adjacent to sphere
        m = self.adj_solid
        un = (u[..., 0] * self.sph_nx + u[..., 1] * self.sph_ny + u[..., 2] * self.sph_nz)
        # n points outward from sphere center; un<0 means flow into the solid
        un = np.where(m & (un < 0.0), un, 0.0)
        u[..., 0] -= un * self.sph_nx
        u[..., 1] -= un * self.sph_ny
        u[..., 2] -= un * self.sph_nz
        u[self.solid] = 0.0

    def _project(self, dt):
        u = self.u
        hx, hy, hz = self.hx, self.hy, self.hz
        # face-flux divergence: div = (U_{i+1/2}-U_{i-1/2})/hx + ...
        div = np.zeros(self.shape)
        Ux = 0.5 * (u[:-1, :, :, 0] + u[1:, :, :, 0])
        div[1:, :, :] += Ux / hx
        div[:-1, :, :] -= Ux / hx
        Uy = 0.5 * (u[:, :-1, :, 1] + u[:, 1:, :, 1])
        div[:, 1:, :] += Uy / hy
        div[:, :-1, :] -= Uy / hy
        Uz = 0.5 * (u[:, :, :-1, 2] + u[:, :, 1:, 2])
        div[:, :, 1:] += Uz / hz
        div[:, :, :-1] -= Uz / hz
        # wall points: zero outward normal flux already enforced in u
        # A = -Laplacian, solve lap(p)=div/dt  =>  A p = -div/dt
        b_full = -div / dt
        b_full[~self.fluid] = 0.0
        b_full[:, :, -1] = 0.0  # Dirichlet rows at open top
        b = b_full.ravel()[np.flatnonzero(self.fluid.ravel())]
        p_full = np.zeros(self.shape)
        p = self.lufs.solve(b)
        p_full[self.fluid] = p
        gx_ = np.gradient(p_full, hx, axis=0, edge_order=1)
        gy_ = np.gradient(p_full, hy, axis=1, edge_order=1)
        gz_ = np.gradient(p_full, hz, axis=2, edge_order=1)
        u[..., 0] -= dt * gx_
        u[..., 1] -= dt * gy_
        u[..., 2] -= dt * gz_
        self._enforce_boundaries()

    def _vorticity_confinement(self, dt):
        u = self.u
        hx, hy, hz = self.hx, self.hy, self.hz
        wx = np.gradient(u[..., 2], hy, axis=1) - np.gradient(u[..., 1], hz, axis=2)
        wy = np.gradient(u[..., 0], hz, axis=2) - np.gradient(u[..., 2], hx, axis=0)
        wz = np.gradient(u[..., 1], hx, axis=0) - np.gradient(u[..., 0], hy, axis=1)
        mag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        gnx = np.gradient(mag, hx, axis=0)
        gny = np.gradient(mag, hy, axis=1)
        gnz = np.gradient(mag, hz, axis=2)
        nrm = np.sqrt(gnx ** 2 + gny ** 2 + gnz ** 2) + 1e-12
        gnx /= nrm; gny /= nrm; gnz /= nrm
        fx = gny * wz - gnz * wy
        fy = gnz * wx - gnx * wz
        fz = gnx * wy - gny * wx
        c = self.eps_vc * self.h_vc * dt
        u[..., 0] += c * fx
        u[..., 1] += c * fy
        u[..., 2] += c * fz
        u[self.solid] = 0.0

    # ------------------------------------------------------------------ step
    def _step(self, dt):
        t = self.t
        # 1) advect velocity and smoke (MacCormack)
        self.u = self._advect_mc(self.u, dt)
        self.s = self._advect_mc(self.s, dt)
        self.s[self.solid] = 0.0
        self.s = np.clip(self.s, 0.0, 6.0)
        self._enforce_boundaries()

        # 2) smoke emission + tiny deterministic jitter to seed asymmetry
        self.s[self.emitter] += dt * self.src_strength
        jit = 0.25 * np.sin(2.3 * t) 
        self.u[..., 0][self.emitter] += dt * jit
        self.u[..., 1][self.emitter] += dt * 0.25 * np.cos(1.7 * t)

        # 3) buoyancy (gravity-aligned, +z)
        self.u[..., 2] += dt * self.alpha_buoy * self.s

        # 4) vorticity confinement
        self._vorticity_confinement(dt)

        # 5) pressure projection + rigid boundaries
        self._project(dt)

        # 6) tracer particles (RK2) + no-penetration clamp
        p = self.pos
        v1 = self._sample_u(p[:, 0], p[:, 1], p[:, 2])
        pm = p + 0.5 * dt * v1
        v2 = self._sample_u(pm[:, 0], pm[:, 1], pm[:, 2])
        p += dt * v2
        self.pvel = v2
        d = p - self.sc
        r = np.linalg.norm(d, axis=1)
        bad = r < self.sr + 0.02
        if np.any(bad):
            p[bad] = self.sc + d[bad] / r[bad, None] * (self.sr + 0.02)
        p[:, 0] = np.clip(p[:, 0], -2.95, 2.95)
        p[:, 1] = np.clip(p[:, 1], -1.15, 1.15)
        p[:, 2] = np.clip(p[:, 2], 0.01, 3.75)
        # cumulative mass source at particle locations
        sint = _trilin(self.scalar_source[..., None], p[:, 0], p[:, 1], p[:, 2],
                       self.gx, self.gy, self.gz, self.hx, self.hy, self.hz)[:, 0]
        self.cum_src += dt * sint * (self.mass / 1.2)
        self.t += dt

    # ---------------------------------------------------------------- frames
    def _frame(self):
        bvel = self._sample_u(self.bpos[:, 0], self.bpos[:, 1], self.bpos[:, 2])
        attrs = dict(self.attrs)
        attrs["cumulative_mass_source"] = self.cum_src.copy()
        fields = {
            "cumulative_mass_source": self.cum_src.copy(),
            "transported_scalar": self.s.copy(),
            "scalar_source": self.scalar_source.copy(),
            "active_mask": self.active_mask.copy(),
            "velocity_field": self.u.copy(),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_velocity.copy(),
            "boundary_velocity": np.asarray(bvel, float),
        }
        frame = StateFrame(
            float(self.t), self.pos.copy(), self.pvel.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes=attrs, fields=fields, scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.sub_dt, time - self.t)
            self._step(dt)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "smoke_mass": float(self.s[self.fluid].sum()),
            "max_speed": float(np.abs(self.u).max()),
            "grid_shape": list(self.shape),
        }
