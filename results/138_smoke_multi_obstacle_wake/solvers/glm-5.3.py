"""Eulerian buoyant-smoke solver for scene 138_smoke_multi_obstacle_wake.

Physics: Stam-style "stable fluids" scheme on a cell-centered grid.

* semi-Lagrangian advection of velocity and smoke concentration,
* Boussinesq buoyancy (light smoke rises against gravity),
* vorticity confinement (Fedkiw et al.) to preserve vortical roll-up,
* Chorin pressure projection with immersed rigid obstacles and impermeable
  (no-through-flow) domain walls, solved with a sparse LU factorization,
* continuous smoke emission from the frozen emitter geometry,
* tracer parcels advected by the field for the StateFrame point state.

No Genesis, no keyframes: the state is produced purely by time-stepping the
discretized Navier-Stokes equations from the frozen initial condition.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Incompressible buoyant-smoke solver honoring the frozen scene contract."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        geom = cfg.get("scene_geometry") or {}
        objects = geom.get("objects", []) if isinstance(geom, dict) else []

        sampling = cfg.get("sampling", {})
        shape = tuple(int(s) for s in sampling.get("field_grid_shape", (24, 16, 28)))
        self.nx, self.ny, self.nz = shape
        lo, hi = self._domain_bounds(objects)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.h = (self.hi - self.lo) / np.asarray(shape, dtype=float)
        self.hx, self.hy, self.hz = (float(v) for v in self.h)

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        ax = self.lo[0] + (np.arange(self.nx) + 0.5) * self.hx
        ay = self.lo[1] + (np.arange(self.ny) + 0.5) * self.hy
        az = self.lo[2] + (np.arange(self.nz) + 0.5) * self.hz
        self.ax, self.ay, self.az = ax, ay, az
        self.X, self.Y, self.Z = np.meshgrid(ax, ay, az, indexing="ij")

        # ---- rigid solids (immersed obstacles + ground slab) ----
        self.boxes = self._rigid_boxes(objects)
        self.solid = np.zeros(shape, dtype=bool)
        for center, half in self.boxes:
            self.solid |= (
                (self.X >= center[0] - half[0] - 1e-9)
                & (self.X <= center[0] + half[0] + 1e-9)
                & (self.Y >= center[1] - half[1] - 1e-9)
                & (self.Y <= center[1] + half[1] + 1e-9)
                & (self.Z >= center[2] - half[2] - 1e-9)
                & (self.Z <= center[2] + half[2] + 1e-9)
            )
        self.fluid = ~self.solid

        # ---- frozen initial condition ----
        init = scene.initial_state()
        smoke0 = np.asarray(init.fields.get("transported_scalar", 0.0), dtype=float)
        if smoke0.shape != shape:
            smoke0 = np.zeros(shape)
        self.smoke = np.clip(smoke0, 0.0, 2.0)
        self.u = np.zeros(shape)
        self.v = np.zeros(shape)
        self.w = np.zeros(shape)

        # ---- physical parameters ----
        self.source = self._build_source(objects)
        self.beta = 3.2          # Boussinesq buoyancy [m/s^2 per unit smoke]
        self.eps_c = 0.22        # vorticity confinement strength
        self.kappa = 0.015       # smoke diffusivity [m^2/s]
        self.vmax = 6.0          # numerical speed clamp [m/s]
        self.tau = 45.0          # slow smoke decay time constant [s]

        self._build_faces()
        self._build_projection()
        self._build_boundary_samples()

        # ---- tracer parcels from the frozen sampling cloud ----
        self.pos = np.array(init.position, dtype=float, copy=True)
        self.pvel = np.zeros_like(self.pos)
        self.mass = np.array(init.mass, dtype=float, copy=True)
        if init.body_id is not None:
            self.body_id = np.array(init.body_id, copy=True)
        else:
            self.body_id = np.zeros(len(self.pos), dtype=np.int32)
        self.attributes = {
            name: np.array(value, copy=True) for name, value in init.attributes.items()
        }

        scalars = init.scalars
        self.char_len = float(scalars.get("characteristic_length", 0.72))
        self.grid_spacing = float(scalars.get("grid_spacing", 0.18))
        self.ref_v = max(1.0, float(scalars.get("reference_velocity", 1.0)))

        self.t = 0.0
        self.sub_dt = 1.0 / 60.0
        self.max_speed_seen = 0.0
        self._ready = True

    # ------------------------------------------------------- geometry helpers
    @staticmethod
    def _domain_bounds(objects):
        for obj in objects:
            if obj.get("semantic") == "gas":
                geo = obj.get("geometry", {})
                if geo.get("bounds_min") and geo.get("bounds_max"):
                    return (
                        [float(v) for v in geo["bounds_min"]],
                        [float(v) for v in geo["bounds_max"]],
                    )
        return [-3.0, -1.2, 0.0], [3.0, 1.2, 3.8]

    @staticmethod
    def _rigid_boxes(objects):
        boxes = []
        for obj in objects:
            if obj.get("semantic") != "boundary":
                continue
            if not obj.get("collision_enabled", True):
                continue
            geo = obj.get("geometry", {})
            if geo.get("type") != "box" or not geo.get("half_extents"):
                continue
            translation = obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
            boxes.append(
                (
                    np.asarray(translation, dtype=float),
                    np.asarray(geo["half_extents"], dtype=float),
                )
            )
        return boxes

    def _build_source(self, objects):
        radius, height = 0.32, 0.1
        center = np.zeros(3)
        for obj in objects:
            if obj.get("semantic") != "emitter":
                continue
            geo = obj.get("geometry", {})
            radius = float(geo.get("radius", radius))
            height = float(geo.get("height", height))
            center = np.asarray(
                obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0]), dtype=float
            )
        radial = (self.X - center[0]) ** 2 + (self.Y - center[1]) ** 2
        mask = (
            (radial <= radius * radius)
            & (self.Z <= center[2] + height + 1.5 * self.hz)
            & self.fluid
        )
        src = np.zeros((self.nx, self.ny, self.nz))
        src[mask] = 3.0
        return src

    # ----------------------------------------------------------- discretization
    def _build_faces(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        afx = np.zeros((nx + 1, ny, nz), dtype=bool)
        afx[1:nx] = self.fluid[:-1] & self.fluid[1:]
        afy = np.zeros((nx, ny + 1, nz), dtype=bool)
        afy[:, 1:ny] = self.fluid[:, :-1] & self.fluid[:, 1:]
        afz = np.zeros((nx, ny, nz + 1), dtype=bool)
        afz[:, :, 1:nz] = self.fluid[:, :, :-1] & self.fluid[:, :, 1:]
        self.afx, self.afy, self.afz = afx, afy, afz

    def _build_projection(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        self.nf = int(self.fluid.sum())
        self.cell_index = -np.ones((nx, ny, nz), dtype=np.int64)
        self.cell_index[self.fluid] = np.arange(self.nf)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        diag = np.zeros(self.nf)
        faces = ((0, self.hx, self.afx), (1, self.hy, self.afy), (2, self.hz, self.afz))
        for axis, h, af in faces:
            h2inv = 1.0 / (h * h)
            count = af.shape[axis]
            face_slice = [slice(None)] * 3
            face_slice[axis] = slice(1, count - 1)
            active = af[tuple(face_slice)]
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = slice(0, -1)
            high[axis] = slice(1, None)
            ia = self.cell_index[tuple(low)]
            ib = self.cell_index[tuple(high)]
            mask = active & (ia >= 0) & (ib >= 0)
            a = ia[mask]
            b = ib[mask]
            if a.size:
                rows.append(a)
                cols.append(b)
                vals.append(np.full(a.size, h2inv))
                rows.append(b)
                cols.append(a)
                vals.append(np.full(a.size, h2inv))
                np.add.at(diag, a, -h2inv)
                np.add.at(diag, b, -h2inv)
        index = np.arange(self.nf)
        rows.append(index)
        cols.append(index)
        vals.append(diag)
        matrix = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nf, self.nf),
        ).tocsc()
        # regularize the pure-Neumann null space (negligible vs. 1/h^2 diagonal)
        matrix = matrix + sp.identity(self.nf, format="csc") * 1e-6
        self.lu = spla.splu(matrix)

    def _build_boundary_samples(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        stride = 2
        pts: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        def wall(px, py, pz, normal):
            pts.append(np.array([[px, py, pz]], dtype=float))
            nrm.append(np.array([normal], dtype=float))

        # impermeable domain walls
        for j in range(0, ny, stride):
            for k in range(0, nz, stride):
                if self.fluid[0, j, k]:
                    wall(self.lo[0], self.ay[j], self.az[k], (1.0, 0.0, 0.0))
                if self.fluid[nx - 1, j, k]:
                    wall(self.hi[0], self.ay[j], self.az[k], (-1.0, 0.0, 0.0))
        for i in range(0, nx, stride):
            for k in range(0, nz, stride):
                if self.fluid[i, 0, k]:
                    wall(self.ax[i], self.lo[1], self.az[k], (0.0, 1.0, 0.0))
                if self.fluid[i, ny - 1, k]:
                    wall(self.ax[i], self.hi[1], self.az[k], (0.0, -1.0, 0.0))
        for i in range(0, nx, stride):
            for j in range(0, ny, stride):
                if self.fluid[i, j, 0]:
                    wall(self.ax[i], self.ay[j], self.lo[2], (0.0, 0.0, 1.0))
                if self.fluid[i, j, nz - 1]:
                    wall(self.ax[i], self.ay[j], self.hi[2], (0.0, 0.0, -1.0))

        # immersed obstacle faces (unit normal points from solid into fluid)
        specs = (
            (0, self.afx.shape[0]),
            (1, self.afy.shape[1]),
            (2, self.afz.shape[2]),
        )
        for axis, _count in specs:
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = slice(0, -1)
            high[axis] = slice(1, None)
            mixed = self.fluid[tuple(low)] ^ self.fluid[tuple(high)]
            ii, jj, kk = np.nonzero(mixed)
            if ii.size:
                along = (ii, jj, kk)[axis]
                coord = [self.ax[ii], self.ay[jj], self.az[kk]]
                coord[axis] = self.lo[axis] + (along + 1) * self.h[axis]
                sign = np.where(self.fluid[tuple(high)][ii, jj, kk], 1.0, -1.0)
                normals = np.zeros((ii.size, 3))
                normals[:, axis] = sign
                pts.append(np.column_stack(coord))
                nrm.append(normals)

        self.bpts = (
            np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
        )
        self.bnorm = (
            np.concatenate(nrm, axis=0) if nrm else np.zeros((0, 3))
        )
        self.btarget = np.zeros_like(self.bnorm)
        self.sX = self.bpts[:, 0] if len(self.bpts) else np.zeros(0)
        self.sY = self.bpts[:, 1] if len(self.bpts) else np.zeros(0)
        self.sZ = self.bpts[:, 2] if len(self.bpts) else np.zeros(0)

    # ------------------------------------------------------------- numerics
    def _trilerp(self, f, X, Y, Z):
        X = np.clip(np.asarray(X, dtype=float), self.lo[0], self.hi[0] - 1e-9 * self.hx)
        Y = np.clip(np.asarray(Y, dtype=float), self.lo[1], self.hi[1] - 1e-9 * self.hy)
        Z = np.clip(np.asarray(Z, dtype=float), self.lo[2], self.hi[2] - 1e-9 * self.hz)
        gx = (X - self.lo[0]) / self.hx - 0.5
        gy = (Y - self.lo[1]) / self.hy - 0.5
        gz = (Z - self.lo[2]) / self.hz - 0.5
        i0 = np.clip(np.floor(gx).astype(np.int64), 0, self.nx - 2)
        j0 = np.clip(np.floor(gy).astype(np.int64), 0, self.ny - 2)
        k0 = np.clip(np.floor(gz).astype(np.int64), 0, self.nz - 2)
        tx = np.clip(gx - i0, 0.0, 1.0)
        ty = np.clip(gy - j0, 0.0, 1.0)
        tz = np.clip(gz - k0, 0.0, 1.0)
        c000 = f[i0, j0, k0]
        c100 = f[i0 + 1, j0, k0]
        c010 = f[i0, j0 + 1, k0]
        c110 = f[i0 + 1, j0 + 1, k0]
        c001 = f[i0, j0, k0 + 1]
        c101 = f[i0 + 1, j0, k0 + 1]
        c011 = f[i0, j0 + 1, k0 + 1]
        c111 = f[i0 + 1, j0 + 1, k0 + 1]
        return (
            c000 * (1 - tx) * (1 - ty) * (1 - tz)
            + c100 * tx * (1 - ty) * (1 - tz)
            + c010 * (1 - tx) * ty * (1 - tz)
            + c110 * tx * ty * (1 - tz)
            + c001 * (1 - tx) * (1 - ty) * tz
            + c101 * tx * (1 - ty) * tz
            + c011 * (1 - tx) * ty * tz
            + c111 * tx * ty * tz
        )

    def _advect(self, f, u, v, w, dt):
        return self._trilerp(f, self.X - dt * u, self.Y - dt * v, self.Z - dt * w)

    def _laplacian(self, f):
        lap = np.zeros_like(f)
        for axis, h in ((0, self.hx), (1, self.hy), (2, self.hz)):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            fp = np.pad(f, pad, mode="edge")
            flp = np.pad(self.fluid, pad, mode="edge")
            slp = [slice(None)] * 3
            slp[axis] = slice(2, None)
            slm = [slice(None)] * 3
            slm[axis] = slice(0, -2)
            nb_p = fp[tuple(slp)]
            nb_m = fp[tuple(slm)]
            fl_p = flp[tuple(slp)]
            fl_m = flp[tuple(slm)]
            eff_p = np.where(fl_p, nb_p, f)
            eff_m = np.where(fl_m, nb_m, f)
            lap += ((eff_p - f) + (eff_m - f)) / (h * h)
        return lap

    def _confinement(self, dt):
        if self.eps_c <= 0.0:
            return
        ox = np.gradient(self.w, self.hy, axis=1) - np.gradient(self.v, self.hz, axis=2)
        oy = np.gradient(self.u, self.hz, axis=2) - np.gradient(self.w, self.hx, axis=0)
        oz = np.gradient(self.v, self.hx, axis=0) - np.gradient(self.u, self.hy, axis=1)
        mag = np.sqrt(ox * ox + oy * oy + oz * oz)
        gx = np.gradient(mag, self.hx, axis=0)
        gy = np.gradient(mag, self.hy, axis=1)
        gz = np.gradient(mag, self.hz, axis=2)
        gnorm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        nx_, ny_, nz_ = gx / gnorm, gy / gnorm, gz / gnorm
        fx = self.eps_c * (ny_ * oz - nz_ * oy)
        fy = self.eps_c * (nz_ * ox - nx_ * oz)
        fz = self.eps_c * (nx_ * oy - ny_ * ox)
        self.u += dt * fx * self.fluid
        self.v += dt * fy * self.fluid
        self.w += dt * fz * self.fluid

    def _project(self, dt):
        nx, ny, nz = self.nx, self.ny, self.nz
        uf = np.zeros((nx + 1, ny, nz))
        uf[1:nx] = np.where(self.afx[1:nx], 0.5 * (self.u[:-1] + self.u[1:]), 0.0)
        vf = np.zeros((nx, ny + 1, nz))
        vf[:, 1:ny] = np.where(self.afy[:, 1:ny], 0.5 * (self.v[:, :-1] + self.v[:, 1:]), 0.0)
        wf = np.zeros((nx, ny, nz + 1))
        wf[:, :, 1:nz] = np.where(
            self.afz[:, :, 1:nz], 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:]), 0.0
        )
        div = (
            (uf[1:] - uf[:-1]) / self.hx
            + (vf[:, 1:] - vf[:, :-1]) / self.hy
            + (wf[:, :, 1:] - wf[:, :, :-1]) / self.hz
        )
        rhs = div[self.fluid] / dt
        pressure = self.lu.solve(np.asarray(rhs, dtype=float))
        pf = np.zeros_like(div)
        pf[self.fluid] = pressure
        uf[1:nx] = np.where(
            self.afx[1:nx], uf[1:nx] - dt * (pf[1:] - pf[:-1]) / self.hx, 0.0
        )
        vf[:, 1:ny] = np.where(
            self.afy[:, 1:ny],
            vf[:, 1:ny] - dt * (pf[:, 1:] - pf[:, :-1]) / self.hy,
            0.0,
        )
        wf[:, :, 1:nz] = np.where(
            self.afz[:, :, 1:nz],
            wf[:, :, 1:nz] - dt * (pf[:, :, 1:] - pf[:, :, :-1]) / self.hz,
            0.0,
        )
        lf = self.afx[:-1].astype(float)
        rf = self.afx[1:].astype(float)
        cnt = np.maximum(lf + rf, 1.0)
        self.u = np.where(lf + rf > 0.0, (lf * uf[:-1] + rf * uf[1:]) / cnt, 0.0)
        lf = self.afy[:, :-1].astype(float)
        rf = self.afy[:, 1:].astype(float)
        cnt = np.maximum(lf + rf, 1.0)
        self.v = np.where(lf + rf > 0.0, (lf * vf[:, :-1] + rf * vf[:, 1:]) / cnt, 0.0)
        lf = self.afz[:, :, :-1].astype(float)
        rf = self.afz[:, :, 1:].astype(float)
        cnt = np.maximum(lf + rf, 1.0)
        self.w = np.where(lf + rf > 0.0, (lf * wf[:, :, :-1] + rf * wf[:, :, 1:]) / cnt, 0.0)

    def _advance_parcels(self, dt):
        if len(self.pos) == 0:
            return
        pu = self._trilerp(self.u, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        pv = self._trilerp(self.v, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        pw = self._trilerp(self.w, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        vel = np.column_stack((pu, pv, pw))
        new_pos = self.pos + dt * vel
        margin = 1e-6
        inside = np.all(
            (new_pos > self.lo + margin) & (new_pos < self.hi - margin), axis=1
        )
        for center, half in self.boxes:
            inside &= ~np.all(
                (new_pos >= center - half) & (new_pos <= center + half), axis=1
            )
        keep = inside[:, None]
        self.pos = np.where(keep, new_pos, self.pos)
        self.pvel = np.where(keep, vel, 0.0)

    # ------------------------------------------------------------- time step
    def _step(self, dt):
        fluid = self.fluid

        # buoyancy: Boussinesq force proportional to smoke concentration
        self.w = self.w + dt * self.beta * self.smoke * fluid
        # vorticity confinement to preserve roll-up
        self._confinement(dt)

        # emitter region: deterministic perturbation + volumetric outward push
        smask = self.source > 0.0
        if np.any(smask):
            self.u = self.u + dt * self.rng.normal(0.0, 0.25, self.u.shape) * smask
            self.v = self.v + dt * self.rng.normal(0.0, 0.25, self.v.shape) * smask
            radius = np.sqrt(self.X**2 + self.Y**2) + 1e-9
            push = 0.35 * self.source * smask
            self.u = self.u + dt * push * self.X / radius
            self.v = self.v + dt * push * self.Y / radius

        # numerical speed clamp
        spd = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.max_speed_seen = max(self.max_speed_seen, float(spd.max()))
        over = spd > self.vmax
        if np.any(over):
            scale = np.where(over, self.vmax / np.maximum(spd, 1e-12), 1.0)
            self.u = self.u * scale
            self.v = self.v * scale
            self.w = self.w * scale

        # semi-Lagrangian advection
        u0, v0, w0 = self.u, self.v, self.w
        self.u = self._advect(u0, u0, v0, w0, dt)
        self.v = self._advect(v0, u0, v0, w0, dt)
        self.w = self._advect(w0, u0, v0, w0, dt)
        self.smoke = self._advect(self.smoke, u0, v0, w0, dt)

        # mild scalar diffusion (explicit, well below stability limit)
        self.smoke = self.smoke + dt * self.kappa * self._laplacian(self.smoke)

        # emission and slow decay
        self.smoke = self.smoke + dt * self.source
        self.smoke = np.clip(self.smoke * (1.0 - dt / self.tau), 0.0, 2.0)
        self.smoke = self.smoke * fluid
        self.u = self.u * fluid
        self.v = self.v * fluid
        self.w = self.w * fluid

        # incompressibility with impermeable walls and obstacles
        self._project(dt)
        self.u = self.u * fluid
        self.v = self.v * fluid
        self.w = self.w * fluid

        self._advance_parcels(dt)
        spd = float(np.sqrt(self.u**2 + self.v**2 + self.w**2).max())
        self.max_speed_seen = max(self.max_speed_seen, spd)

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        target = float(time)
        guard = 1e-9
        while self.t < target - guard:
            dt = min(self.sub_dt, target - self.t)
            self._step(dt)
            self.t = self.t + dt
        if self.t < target:
            self.t = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        velocity = np.stack((self.u, self.v, self.w), axis=-1)
        if len(self.bpts):
            bu = self._trilerp(self.u, self.sX, self.sY, self.sZ)
            bv = self._trilerp(self.v, self.sX, self.sY, self.sZ)
            bw = self._trilerp(self.w, self.sX, self.sY, self.sZ)
            boundary_velocity = np.column_stack((bu, bv, bw))
        else:
            boundary_velocity = np.zeros((0, 3))
        fields = {
            "transported_scalar": np.array(self.smoke, dtype=float),
            "scalar_source": np.array(self.source, dtype=float),
            "active_mask": self.fluid.copy(),
            "velocity_field": velocity,
            "boundary_normal": self.bnorm.copy(),
            "boundary_target_velocity": self.btarget.copy(),
            "boundary_velocity": boundary_velocity,
        }
        scalars = {
            "characteristic_length": float(self.char_len),
            "grid_spacing": float(self.grid_spacing),
            "reference_velocity": float(max(1.0, self.max_speed_seen)),
        }
        attributes = {
            name: np.array(value, copy=True) for name, value in self.attributes.items()
        }
        return StateFrame(
            time=float(time),
            position=np.array(self.pos, copy=True),
            velocity=np.array(self.pvel, copy=True),
            mass=np.array(self.mass, copy=True),
            body_id=np.array(self.body_id, copy=True),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "eulerian_stable_fluids_buoyant_smoke",
            "grid": [self.nx, self.ny, self.nz],
            "time": float(self.t),
            "max_speed": float(self.max_speed_seen),
            "boundary_samples": int(len(self.bpts)),
            "fluid_cells": int(self.nf),
        }
