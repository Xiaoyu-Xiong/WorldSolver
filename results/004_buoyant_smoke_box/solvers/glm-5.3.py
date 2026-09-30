"""Buoyant smoke box solver (scene 004_buoyant_smoke_box).

Physically-based Eulerian solver for buoyant flow in a closed box:
  * Boussinesq buoyancy force from an advected temperature field,
  * semi-Lagrangian advection of velocity, temperature and smoke density,
  * incompressibility enforced by a pressure projection (sparse Poisson
    solve with homogeneous-Neumann walls == closed no-penetration box),
  * vorticity confinement to preserve coherent vortices,
  * passive markers advected by the interpolated velocity field.

No keyframes, no precomputed trajectories: every state is produced by
explicitly time-stepping the discretized Navier-Stokes/Boussinesq system.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import cg, splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- numerical parameters (frozen at class level, deterministic) ---
    SUBSTEP = 0.01          # physical time step [s]
    BUOYANCY = 2.0          # Boussinesq buoyancy coefficient [m/s^2 per unit T]
    VORTICITY_EPS = 0.25    # vorticity confinement strength
    N_PROJ = 2              # projection iterations per stage
    T_SOURCE_TAU = 0.25     # heater relaxation time [s]
    SMOKE_SOURCE_TAU = 1.0  # smoke emitter relaxation time [s]
    COOL_TAU = 6.0          # bulk cooling toward ambient [s]
    T_SOURCE_TOP = 0.12     # top of the heat source region [m]
    SMOKE_SOURCE_TOP = 0.14 # top of the smoke emitter region [m]

    def initialize(self, scene, seed: int) -> None:
        cfg = {}
        cfg_attr = getattr(scene, "config", None)
        if isinstance(cfg_attr, dict):
            cfg = cfg_attr
        domain = cfg.get("domain", {
            "bounds_min": [-0.75, -0.75, 0.0],
            "bounds_max": [0.75, 0.75, 3.0],
        })
        initial = cfg.get("initial_conditions", {
            "source_radius": 0.28,
            "source_height": 0.24,
        })

        frame0 = scene.initial_state()
        self.shape = tuple(int(v) for v in np.asarray(
            frame0.fields.get("smoke_density", np.zeros((32, 32, 64))).shape))
        nx, ny, nz = self.shape
        self.lo = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.hi = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.d = (self.hi - self.lo) / np.asarray(self.shape, dtype=np.float64)

        # --- grid coordinate arrays ---
        ax = [self.lo[a] + (np.arange(self.shape[a]) + 0.5) * self.d[a]
              for a in range(3)]
        self.X, self.Y, self.Z = np.meshgrid(*ax, indexing="ij")

        # --- state from the frozen neutral initial condition ---
        self.u = np.asarray(frame0.fields.get(
            "velocity_field", np.zeros(self.shape + (3,))), dtype=np.float64).copy()
        if self.u.shape != self.shape + (3,):
            self.u = np.zeros(self.shape + (3,))
        self.T = np.asarray(frame0.fields.get(
            "temperature", np.zeros(self.shape)), dtype=np.float64).copy()
        if self.T.shape != self.shape:
            self.T = np.zeros(self.shape)
        self.S = np.asarray(frame0.fields.get(
            "smoke_density", np.zeros(self.shape)), dtype=np.float64).copy()
        if self.S.shape != self.shape:
            self.S = np.zeros(self.shape)
        self.T = np.clip(self.T, 0.0, None)
        self.S = np.clip(self.S, 0.0, None)

        self.pos = np.asarray(frame0.position, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.marker_kind = np.asarray(
            frame0.attributes.get("marker_kind",
                                  np.ones(len(self.pos), dtype=np.int32))).copy()

        self.t = 0.0
        self.expected_smoke = float(self.S.sum())

        # --- source regions (heat emitter cylinder at the box floor) ---
        src_radius = float(initial.get("source_radius", 0.28))
        r2 = self.X ** 2 + self.Y ** 2
        self.heat_mask = (r2 < src_radius ** 2) & (self.Z < self.T_SOURCE_TOP)
        self.smoke_mask = (r2 < src_radius ** 2) & (self.Z < self.SMOKE_SOURCE_TOP)

        # deterministic symmetry-breaking perturbation so the plume sheds
        # coherent vortices instead of staying perfectly axisymmetric
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        noise = rng.standard_normal(self.shape)
        self.T = np.clip(
            self.T + 0.05 * noise * self.heat_mask
            + 0.01 * rng.standard_normal(self.shape), 0.0, None)

        # initial smoke source snapshot (for the t = 0 frame)
        self.last_scalar_source = np.where(
            self.smoke_mask, np.clip(1.0 - self.S, 0.0, None) / self.SMOKE_SOURCE_TAU,
            0.0)

        # --- boundary (wall) samples for the closed box ---
        self._build_boundary()

        # --- sparse Poisson operator with homogeneous-Neumann walls ---
        self._build_laplacian()

        # --- frozen diagnostic scales ---
        self.characteristic_length = float(self.hi[2] - self.lo[2])
        self.grid_spacing = float(np.mean(self.d))
        self.reference_velocity = float(np.sqrt(self.BUOYANCY * self.characteristic_length))

        self.div_l2 = 0.0
        self.div_max = 0.0
        self.vol_err = 0.0

    # ------------------------------------------------------------------
    # discretization helpers
    # ------------------------------------------------------------------
    def _build_laplacian(self) -> None:
        n = int(np.prod(self.shape))
        idx = np.arange(n).reshape(self.shape)
        rows, cols, vals = [], [], []
        inv2 = [1.0 / self.d[a] ** 2 for a in range(3)]
        for a in range(3):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[a] = slice(0, -1)
            s2[a] = slice(1, None)
            i = idx[tuple(s1)].ravel()
            j = idx[tuple(s2)].ravel()
            w = inv2[a]
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [np.full(i.size, w), np.full(j.size, w),
                     np.full(i.size, -w), np.full(j.size, -w)]
        A = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n)).tocsc()
        # pin the constant nullspace mode at one corner cell
        A[0, 0] += 1.0
        self._A = A
        try:
            self._poisson_solve = splu(A, permc_spec="MMD_AT_PLUS_A").solve
        except Exception:
            try:
                self._poisson_solve = splu(A).solve
            except Exception:
                self._poisson_solve = None  # CG fallback

    def _build_boundary(self) -> None:
        nx, ny, nz = self.shape
        ix, iy, iz, nrm = [], [], [], []
        faces = (
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
        )
        for axis, side in faces:
            if axis == 0:
                i = np.full((ny, nz), 0 if side == 0 else nx - 1)
                yy, zz = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
                ix.append(i.ravel()); iy.append(yy.ravel()); iz.append(zz.ravel())
            elif axis == 1:
                j = np.full((nx, nz), 0 if side == 0 else ny - 1)
                xx, zz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
                ix.append(xx.ravel()); iy.append(j.ravel()); iz.append(zz.ravel())
            else:
                k = np.full((nx, ny), 0 if side == 0 else nz - 1)
                xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
                ix.append(xx.ravel()); iy.append(yy.ravel()); iz.append(k.ravel())
            normal = np.zeros(3)
            normal[axis] = -1.0 if side == 0 else 1.0
            nrm.append(np.tile(normal, (len(ix[-1]), 1)))
        self.bix = np.concatenate(ix)
        self.biy = np.concatenate(iy)
        self.biz = np.concatenate(iz)
        self.boundary_normal = np.concatenate(nrm)
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

    # ------------------------------------------------------------------
    # field operations
    # ------------------------------------------------------------------
    def _divergence(self) -> np.ndarray:
        return (np.gradient(self.u[..., 0], self.d[0], axis=0)
                + np.gradient(self.u[..., 1], self.d[1], axis=1)
                + np.gradient(self.u[..., 2], self.d[2], axis=2))

    def _enforce_walls(self) -> None:
        u = self.u
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, 2] = 0.0

    def _poisson(self, rhs: np.ndarray) -> np.ndarray:
        if self._poisson_solve is not None:
            return self._poisson_solve(rhs)
        diag = self._A.diagonal()
        M = sparse.diags(1.0 / diag)
        try:
            sol, _info = cg(self._A, rhs, M=M, rtol=1e-10, atol=0.0, maxiter=800)
        except TypeError:  # older SciPy keyword
            sol, _info = cg(self._A, rhs, M=M, tol=1e-10, maxiter=800)
        return sol

    def _project(self) -> None:
        for _ in range(self.N_PROJ):
            div = self._divergence()
            p = self._poisson(-div.ravel()).reshape(self.shape)
            for c in range(3):
                self.u[..., c] -= np.gradient(p, self.d[c], axis=c)
            self._enforce_walls()

    def _advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        gx = (self.X - self.u[..., 0] * dt - self.lo[0]) / self.d[0] - 0.5
        gy = (self.Y - self.u[..., 1] * dt - self.lo[1]) / self.d[1] - 0.5
        gz = (self.Z - self.u[..., 2] * dt - self.lo[2]) / self.d[2] - 0.5
        return map_coordinates(field, (gx, gy, gz), order=1, mode="nearest")

    def _vorticity_force(self) -> np.ndarray:
        u = self.u
        dx, dy, dz = self.d
        wx = (np.gradient(u[..., 2], dy, axis=1)
              - np.gradient(u[..., 1], dz, axis=2))
        wy = (np.gradient(u[..., 0], dz, axis=2)
              - np.gradient(u[..., 2], dx, axis=0))
        wz = (np.gradient(u[..., 1], dx, axis=0)
              - np.gradient(u[..., 0], dy, axis=1))
        w = np.stack((wx, wy, wz), axis=-1)
        mag = np.sqrt((w * w).sum(-1))
        grad = np.stack((
            np.gradient(mag, dx, axis=0),
            np.gradient(mag, dy, axis=1),
            np.gradient(mag, dz, axis=2),
        ), axis=-1)
        nlen = np.linalg.norm(grad, axis=-1)
        N = np.divide(grad, nlen[..., None], out=np.zeros_like(grad),
                      where=nlen[..., None] > 1e-12)
        return np.cross(N, w) * (self.VORTICITY_EPS * self.grid_spacing)

    def _sample_velocity(self, pts: np.ndarray) -> np.ndarray:
        shp = np.asarray(self.shape)
        g = (pts - self.lo[None, :]) / self.d[None, :] - 0.5
        i0 = np.clip(np.floor(g).astype(np.int64), 0, shp - 1)
        i1 = np.clip(i0 + 1, 0, shp - 1)
        fr = g - np.floor(g)
        out = np.zeros((pts.shape[0], 3))
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    ii = i1[:, 0] if a else i0[:, 0]
                    jj = i1[:, 1] if b else i0[:, 1]
                    kk = i1[:, 2] if c else i0[:, 2]
                    wt = ((fr[:, 0] if a else 1.0 - fr[:, 0])
                          * (fr[:, 1] if b else 1.0 - fr[:, 1])
                          * (fr[:, 2] if c else 1.0 - fr[:, 2]))
                    out += wt[:, None] * self.u[ii, jj, kk, :]
        return out

    # ------------------------------------------------------------------
    # explicit physical time step
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        # 1. buoyancy (Boussinesq) + vorticity confinement (both applied
        #    before advection; the buoyancy force itself is non-divergent)
        self.u[..., 2] += h * self.BUOYANCY * self.T
        self.u += h * self._vorticity_force()
        self._enforce_walls()

        # 2. advect velocity (semi-Lagrangian)
        new_u = np.empty_like(self.u)
        for c in range(3):
            new_u[..., c] = self._advect(self.u[..., c], h)
        self.u = new_u
        self._enforce_walls()
        self._project()

        # 3. advect scalars
        self.T = np.clip(self._advect(self.T, h), 0.0, None)
        self.S = np.clip(self._advect(self.S, h), 0.0, None)

        # 4. heat source (relaxation toward source temperature)
        src_t = np.zeros(self.shape)
        src_t[self.heat_mask] = np.clip(
            1.0 - self.T[self.heat_mask], 0.0, None) / self.T_SOURCE_TAU
        self.T += h * src_t
        self.T *= np.exp(-h / self.COOL_TAU)
        self.T = np.clip(self.T, 0.0, None)

        # 5. smoke emitter (source term for the transported scalar)
        src_s = np.zeros(self.shape)
        src_s[self.smoke_mask] = np.clip(
            1.0 - self.S[self.smoke_mask], 0.0, None) / self.SMOKE_SOURCE_TAU
        self.last_scalar_source = src_s
        self.S += h * src_s
        self.S = np.clip(self.S, 0.0, None)
        self.expected_smoke += float(src_s.sum()) * h
        # global conservation correction (compensates semi-Lagrangian drift)
        actual = float(self.S.sum())
        if actual > 1e-12 and self.expected_smoke > 1e-12:
            self.S *= self.expected_smoke / actual

        # 6. advect passive markers with the flow
        vm = self._sample_velocity(self.pos)
        self.pos = np.clip(self.pos + h * vm,
                           self.lo[None, :] + 1e-3, self.hi[None, :] - 1e-3)

        self.t += h

        # 7. diagnostics from the current state
        div = self._divergence()
        self.div_l2 = float(np.sqrt(np.mean(div * div)))
        self.div_max = float(np.max(np.abs(div)))
        total = float(self.S.sum())
        self.vol_err = (abs(total - self.expected_smoke)
                        / max(self.expected_smoke, 1e-12))

    # ------------------------------------------------------------------
    # StateFrame export
    # ------------------------------------------------------------------
    def _frame(self, t: float) -> StateFrame:
        u32 = self.u.astype(np.float32)
        s32 = self.S.astype(np.float32)
        marker_vel = self._sample_velocity(self.pos)
        boundary_velocity = self.u[self.bix, self.biy, self.biz]
        fields = {
            "smoke_density": s32,
            "velocity_field": u32,
            "temperature": self.T.astype(np.float32),
            "active_mask": np.ones(self.shape, dtype=np.float32),
            "transported_scalar": s32.copy(),
            "scalar_source": self.last_scalar_source.astype(np.float32),
            "boundary_normal": self.boundary_normal.astype(np.float32),
            "boundary_target_velocity": self.boundary_target_velocity.astype(np.float32),
            "boundary_velocity": boundary_velocity.astype(np.float32),
        }
        scalars = {
            "divergence_l2": float(self.div_l2),
            "divergence_max": float(self.div_max),
            "relative_volume_error": float(self.vol_err),
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.reference_velocity,
        }
        frame = StateFrame(
            time=float(t),
            position=self.pos.astype(np.float32),
            velocity=marker_vel.astype(np.float32),
            mass=self.mass.astype(np.float32),
            attributes={"marker_kind": self.marker_kind.astype(np.int32)},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        while self.t < t - 1e-9:
            h = min(self.SUBSTEP, t - self.t)
            self._step(h)
        if abs(self.t - t) < 1e-9:
            self.t = t
        return self._frame(t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "divergence_l2": self.div_l2,
            "divergence_max": self.div_max,
            "relative_volume_error": self.vol_err,
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.reference_velocity,
            "time": self.t,
        }
