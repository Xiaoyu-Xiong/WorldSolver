"""Buoyant smoke box solver.

Physically based incompressible flow solver on a staggered (MAC) grid:

* Boussinesq buoyancy driven by a temperature field fed by the heat source.
* Semi-Lagrangian advection of velocity, temperature and smoke density.
* Exact discrete pressure projection (DCT-II Neumann Poisson solve) enforcing
  the closed no-penetration box, which keeps divergence and volume error at
  round-off level.
* Vorticity confinement to retain coherent plume vortices.
* 120k passive markers advected by trilinear sampling of the divergence-free
  velocity field (no clumping, no scripted motion).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        dom = cfg["domain"]
        ic = cfg["initial_conditions"]
        self.seed = int(seed)
        self.bmin = np.asarray(dom["bounds_min"], dtype=np.float64)
        self.bmax = np.asarray(dom["bounds_max"], dtype=np.float64)
        nx, ny, nz = (int(v) for v in cfg["field_layout"]["grid_shape"])
        self.shape = (nx, ny, nz)
        self.ncell = nx * ny * nz
        self.narr = np.asarray([nx, ny, nz], dtype=np.float64)
        self.h = (self.bmax - self.bmin) / self.narr  # uniform 0.046875
        hx, hy, hz = self.h

        self.src_center = np.asarray(ic["source_center"], dtype=np.float64)
        self.src_radius = float(ic["source_radius"])
        self.src_temp = float(ic["source_temperature"])
        self.T_ambient = float(ic["ambient_temperature"])

        # Numerics / physical parameters
        self.dt_sim = 0.005
        self.alpha = 1.5          # buoyancy acceleration per unit temperature [m/s^2]
        self.heat_rate = 6.0      # temperature source inside heater [K/s]
        self.cool_rate = 0.35     # bulk thermal relaxation [1/s]
        self.emit_rate = 4.0      # smoke density emission [1/s]
        self.decay_rate = 0.08    # smoke dissipation [1/s]
        self.vort_eps = 1.2       # vorticity confinement strength
        self.perturb_rate = 0.6   # symmetry-breaking thermal fluctuation amplitude

        self.L_ref = float(self.src_radius)                     # characteristic length
        self.U_ref = float(np.sqrt(self.alpha * self.L_ref))    # buoyant velocity scale

        # Cell-centre coordinates
        xc = self.bmin[0] + (np.arange(nx) + 0.5) * hx
        yc = self.bmin[1] + (np.arange(ny) + 0.5) * hy
        zc = self.bmin[2] + (np.arange(nz) + 0.5) * hz
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        self.centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Heater region (cylinder sitting on the floor, matches heat_source object)
        r2 = (X - self.src_center[0]) ** 2 + (Y - self.src_center[1]) ** 2
        src = (r2 <= self.src_radius**2) & (Z <= 0.035 + hz)
        self.src_mask = src
        self.src_flat = src.ravel()
        self.src_idx = np.flatnonzero(self.src_flat)
        rng = np.random.default_rng(1000 + self.seed)
        noise = rng.standard_normal(self.shape)
        noise -= noise[src].mean()
        self.noise_flat = noise.ravel()[self.src_idx]

        # Initial scalar fields (same analytic blobs as the frozen scene)
        rr = X * X + Y * Y
        smoke = np.exp(-rr / (2.0 * 0.18**2)) * np.exp(-((Z - 0.13) ** 2) / (2.0 * 0.10**2))
        smoke[Z > 0.34] = 0.0
        temp = np.exp(-rr / (2.0 * 0.20**2)) * np.exp(-((Z - 0.10) ** 2) / (2.0 * 0.09**2))
        self.s = np.ascontiguousarray(smoke, dtype=np.float64)
        self.T = np.ascontiguousarray(temp, dtype=np.float64)

        # Staggered velocity components (no-penetration faces stay zero)
        self.ux = np.zeros((nx + 1, ny, nz))
        self.uy = np.zeros((nx, ny + 1, nz))
        self.uz = np.zeros((nx, ny, nz + 1))

        # Poisson eigenvalues for the MAC Laplacian with homogeneous Neumann BC
        kx = 2.0 * (np.cos(np.pi * np.arange(nx) / nx) - 1.0) / hx**2
        ky = 2.0 * (np.cos(np.pi * np.arange(ny) / ny) - 1.0) / hy**2
        kz = 2.0 * (np.cos(np.pi * np.arange(nz) / nz) - 1.0) / hz**2
        lam = kx[:, None, None] + ky[None, :, None] + kz[None, None, :]
        lam[0, 0, 0] = -1.0
        self.inv_lam = 1.0 / lam
        self.inv_lam[0, 0, 0] = 0.0

        self.active_mask = np.ones(self.shape, dtype=np.float32)
        self.scalar_source = (self.emit_rate * src).astype(np.float32)

        # Passive markers from the frozen scene state
        frame0 = scene.initial_state()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.kind = np.asarray(frame0.attributes["marker_kind"]).copy()
        self.pos = np.asarray(frame0.position, dtype=np.float64).copy()
        self.mvel = np.zeros_like(self.pos)
        self.nmark = len(self.pos)

        self.shape_arr = np.asarray(self.shape, dtype=np.int64)
        self._setup_boundary(xc, yc, zc)

        self.t = 0.0
        self.vol_integral = 0.0
        self.div_l2 = 0.0
        self.div_max = 0.0
        self._compute_divergence_stats(0.0)

    # ------------------------------------------------------- boundary samples
    def _setup_boundary(self, xc, yc, zc) -> None:
        nx, ny, nz = self.shape
        groups_pos = []
        groups_nrm = []
        # x-faces: varying (j,k)
        YZ = np.meshgrid(yc, zc, indexing="ij")
        for xw, sgn in ((self.bmin[0], -1.0), (self.bmax[0], 1.0)):
            p = np.column_stack((np.full(YZ[0].size, xw), YZ[0].ravel(), YZ[1].ravel()))
            n = np.tile([sgn, 0.0, 0.0], (len(p), 1))
            groups_pos.append(p)
            groups_nrm.append(n)
        # y-faces: varying (i,k)
        XZ = np.meshgrid(xc, zc, indexing="ij")
        for yw, sgn in ((self.bmin[1], -1.0), (self.bmax[1], 1.0)):
            p = np.column_stack((XZ[0].ravel(), np.full(XZ[0].size, yw), XZ[1].ravel()))
            n = np.tile([0.0, sgn, 0.0], (len(p), 1))
            groups_pos.append(p)
            groups_nrm.append(n)
        # z-faces: varying (i,j)
        XY = np.meshgrid(xc, yc, indexing="ij")
        for zw, sgn in ((self.bmin[2], -1.0), (self.bmax[2], 1.0)):
            p = np.column_stack((XY[0].ravel(), XY[1].ravel(), np.full(XY[0].size, zw)))
            n = np.tile([0.0, 0.0, sgn], (len(p), 1))
            groups_pos.append(p)
            groups_nrm.append(n)
        self.bound_pos = np.concatenate(groups_pos, axis=0)
        self.bound_nrm = np.ascontiguousarray(np.concatenate(groups_nrm, axis=0))
        self.bound_target = np.zeros_like(self.bound_nrm)
        self.nbound = len(self.bound_pos)
        self.counts = (ny * nz, nx * nz, nx * ny)

    def _boundary_velocity(self) -> np.ndarray:
        nx, ny, nz = self.shape
        ux, uy, uz = self.ux, self.uy, self.uz
        cyz, cxz, cxy = self.counts
        out = np.empty((self.nbound, 3), dtype=np.float64)
        o = 0
        # x-walls
        for i0 in (0, nx):
            ii = 0 if i0 == 0 else nx - 1
            blk = np.empty((cyz, 3))
            blk[:, 0] = ux[i0].ravel()
            blk[:, 1] = (0.5 * (uy[ii, :-1, :] + uy[ii, 1:, :])).ravel()
            blk[:, 2] = (0.5 * (uz[ii, :, :-1] + uz[ii, :, 1:])).ravel()
            out[o:o + cyz] = blk
            o += cyz
        # y-walls
        for j0 in (0, ny):
            jj = 0 if j0 == 0 else ny - 1
            blk = np.empty((cxz, 3))
            blk[:, 0] = (0.5 * (ux[:-1, jj, :] + ux[1:, jj, :])).ravel()
            blk[:, 1] = uy[:, j0, :].ravel()
            blk[:, 2] = (0.5 * (uz[:, jj, :-1] + uz[:, jj, 1:])).ravel()
            out[o:o + cxz] = blk
            o += cxz
        # z-walls
        for k0 in (0, nz):
            kk = 0 if k0 == 0 else nz - 1
            blk = np.empty((cxy, 3))
            blk[:, 0] = (0.5 * (ux[:-1, :, kk] + ux[1:, :, kk])).ravel()
            blk[:, 1] = (0.5 * (uy[:, :-1, kk] + uy[:, 1:, kk])).ravel()
            blk[:, 2] = uz[:, :, k0].ravel()
            out[o:o + cxy] = blk
            o += cxy
        return out

    # ------------------------------------------------------------ interpolation
    def _weights(self, pts: np.ndarray):
        g = (pts - self.bmin) / self.h - 0.5
        np.clip(g, 0.0, None, out=g)
        np.minimum(g, self.narr - 1.0, out=g)
        i0 = np.floor(g)
        frac = g - i0
        i0 = i0.astype(np.int64)
        i1 = np.minimum(i0 + 1, self.shape_arr - 1)
        return i0, i1, frac

    def _sample(self, flat, i0, i1, frac):
        """flat: (ncell, C); returns (M, C) trilinear samples."""
        ny, nz = self.shape[1], self.shape[2]
        ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
        ix1, iy1, iz1 = i1[:, 0], i1[:, 1], i1[:, 2]
        base0 = (ix0 * ny + iy0) * nz
        base1 = (ix1 * ny + iy0) * nz
        base2 = (ix0 * ny + iy1) * nz
        base3 = (ix1 * ny + iy1) * nz
        fx = frac[:, 0:1] if flat.ndim == 2 else frac[:, 0]
        fy = frac[:, 1:2] if flat.ndim == 2 else frac[:, 1]
        fz = frac[:, 2:3] if flat.ndim == 2 else frac[:, 2]
        a = flat[base0 + iz0] * (1.0 - fz) + flat[base0 + iz1] * fz
        b = flat[base2 + iz0] * (1.0 - fz) + flat[base2 + iz1] * fz
        c = flat[base1 + iz0] * (1.0 - fz) + flat[base1 + iz1] * fz
        d = flat[base3 + iz0] * (1.0 - fz) + flat[base3 + iz1] * fz
        return (a * (1.0 - fy) + b * fy) * (1.0 - fx) + (c * (1.0 - fy) + d * fy) * fx

    # ---------------------------------------------------------------- helpers
    def _cell_velocity(self) -> np.ndarray:
        nx, ny, nz = self.shape
        uc = np.empty(self.shape + (3,))
        uc[..., 0] = 0.5 * (self.ux[:-1] + self.ux[1:])
        uc[..., 1] = 0.5 * (self.uy[:, :-1, :] + self.uy[:, 1:, :])
        uc[..., 2] = 0.5 * (self.uz[:, :, :-1] + self.uz[:, :, 1:])
        return uc

    def _faces_from_cells(self, uc) -> None:
        nx, ny, nz = self.shape
        self.ux.fill(0.0)
        self.uy.fill(0.0)
        self.uz.fill(0.0)
        self.ux[1:nx] = 0.5 * (uc[:-1, :, :, 0] + uc[1:, :, :, 0])
        self.uy[:, 1:ny] = 0.5 * (uc[:, :-1, :, 1] + uc[:, 1:, :, 1])
        self.uz[:, :, 1:nz] = 0.5 * (uc[:, :, :-1, 2] + uc[:, :, 1:, 2])

    def _divergence(self) -> np.ndarray:
        hx, hy, hz = self.h
        return (
            (self.ux[1:] - self.ux[:-1]) / hx
            + (self.uy[:, 1:, :] - self.uy[:, :-1, :]) / hy
            + (self.uz[:, :, 1:] - self.uz[:, :, :-1]) / hz
        )

    def _project(self, dt: float) -> None:
        div = self._divergence()
        rhs = div / dt
        spec = dctn(rhs, type=2, norm="ortho")
        spec *= self.inv_lam
        p = idctn(spec, type=2, norm="ortho")
        hx, hy, hz = self.h
        self.ux[1:-1] -= dt * np.diff(p, axis=0) / hx
        self.uy[:, 1:-1] -= dt * np.diff(p, axis=1) / hy
        self.uz[:, :, 1:-1] -= dt * np.diff(p, axis=2) / hz

    def _vorticity_confinement(self, uc, dt: float) -> None:
        hx, hy, hz = self.h
        u, v, w = uc[..., 0], uc[..., 1], uc[..., 2]
        ox = np.gradient(w, hy, axis=1) - np.gradient(v, hz, axis=2)
        oy = np.gradient(u, hz, axis=2) - np.gradient(w, hx, axis=0)
        oz = np.gradient(v, hx, axis=0) - np.gradient(u, hy, axis=1)
        mag = np.sqrt(ox * ox + oy * oy + oz * oz)
        gx = np.gradient(mag, hx, axis=0)
        gy = np.gradient(mag, hy, axis=1)
        gz = np.gradient(mag, hz, axis=2)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        gx /= norm
        gy /= norm
        gz /= norm
        scale = self.vort_eps * hx * dt
        uc[..., 0] += scale * (gy * oz - gz * oy)
        uc[..., 1] += scale * (gz * ox - gx * oz)
        uc[..., 2] += scale * (gx * oy - gy * ox)

    def _compute_divergence_stats(self, dt: float) -> None:
        div = self._divergence()
        self.div_l2 = float(np.sqrt(np.mean(div * div)))
        self.div_max = float(np.max(np.abs(div)))
        if dt > 0.0:
            self.vol_integral += dt * float(np.mean(div))

    # ---------------------------------------------------------------- stepping
    def _substep(self, dt: float) -> None:
        nx, ny, nz = self.shape
        # --- scalar sources (heater emits heat and smoke, bulk relaxation)
        T = self.T
        s = self.s
        T = np.ascontiguousarray(T)
        s = np.ascontiguousarray(s)
        T *= (1.0 - self.cool_rate * dt)
        flat_T = T.ravel()
        flat_T[self.src_idx] += dt * (
            self.heat_rate
            + self.perturb_rate * self.noise_flat * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.9 * self.t))
        )
        s *= (1.0 - self.decay_rate * dt)
        s.ravel()[self.src_idx] += dt * self.emit_rate
        self.T = T
        self.s = s
        np.clip(T, self.T_ambient, 1.5, out=T)
        np.clip(s, 0.0, 3.0, out=s)

        # --- buoyancy on interior z-faces
        t_face = 0.5 * (T[:, :, :-1] + T[:, :, 1:])
        self.uz[:, :, 1:nz] += dt * self.alpha * t_face

        # --- cell-centred velocity, vorticity confinement, advection
        uc = self._cell_velocity()
        self._vorticity_confinement(uc, dt)
        dep = self.centers - dt * uc.reshape(self.ncell, 3)
        i0, i1, frac = self._weights(dep)
        uc_new = self._sample(uc.reshape(self.ncell, 3), i0, i1, frac).reshape(self.shape + (3,))
        self.T = self._sample(T.ravel()[:, None], i0, i1, frac).reshape(self.shape)
        self.s = self._sample(s.ravel()[:, None], i0, i1, frac).reshape(self.shape)
        self._faces_from_cells(uc_new)

        # --- incompressible projection (exact discrete solve)
        self._project(dt)

        # --- passive marker advection with the projected velocity field
        uc = self._cell_velocity()
        uc_flat = uc.reshape(self.ncell, 3)
        i0, i1, frac = self._weights(self.pos)
        vel = self._sample(uc_flat, i0, i1, frac)
        self.pos += dt * vel
        np.clip(self.pos, self.bmin + 1e-9, self.bmax - 1e-9, out=self.pos)
        self.mvel = vel

        self.t += dt
        self._compute_divergence_stats(dt)

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float):
        time = float(time)
        if time > self.t + 1e-12:
            span = time - self.t
            n = max(1, int(round(span / self.dt_sim)))
            dt = span / n
            for _ in range(n):
                self._substep(dt)
        return self._frame(time)

    def _frame(self, time: float) -> StateFrame:
        uc = self._cell_velocity()
        fields = {
            "smoke_density": self.s.astype(np.float32),
            "temperature": self.T.astype(np.float32),
            "velocity_field": uc.astype(np.float32),
            "active_mask": self.active_mask,
            "scalar_source": self.scalar_source,
            "transported_scalar": self.s.astype(np.float32),
            "boundary_normal": self.bound_nrm.astype(np.float32),
            "boundary_target_velocity": self.bound_target.astype(np.float32),
            "boundary_velocity": self._boundary_velocity().astype(np.float32),
        }
        rel_vol = abs(float(np.expm1(self.vol_integral)))
        scalars = {
            "characteristic_length": float(self.L_ref),
            "grid_spacing": float(self.h[0]),
            "reference_velocity": float(self.U_ref),
            "divergence_l2": float(self.div_l2),
            "divergence_max": float(self.div_max),
            "relative_volume_error": rel_vol,
        }
        frame = StateFrame(
            time=float(time),
            position=self.pos.astype(np.float32),
            velocity=self.mvel.astype(np.float32),
            mass=self.mass.astype(np.float32),
            attributes={"marker_kind": self.kind.astype(np.int32)},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "divergence_l2": float(self.div_l2),
            "divergence_max": float(self.div_max),
            "relative_volume_error": abs(float(np.expm1(self.vol_integral))),
            "characteristic_length": float(self.L_ref),
            "grid_spacing": float(self.h[0]),
            "reference_velocity": float(self.U_ref),
            "time": float(self.t),
            "grid_shape": list(self.shape),
            "marker_count": int(self.nmark),
        }
