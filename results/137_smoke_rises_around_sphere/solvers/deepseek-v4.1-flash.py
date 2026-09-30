"""Deterministic Eulerian buoyant-smoke solver for the
"smoke rises around a sphere" scene (Visual Simulation of Smoke, Figure 5).

Physical model
--------------
Incompressible Boussinesq flow on a uniform MAC (staggered) grid, advanced
with an explicit time-stepping scheme:

  1. buoyancy source            du/dt = alpha * s * z_hat   (gravity aligned)
  2. semi-Lagrangian self-advection of the velocity field
  3. BFECC error-compensated semi-Lagrangian advection of the smoke scalar
  4. localised scalar injection inside the cylindrical emitter (saturated)
  5. pressure projection: masked Poisson equation (Jacobi) enforcing
     div(u) = 0 and u.n = 0 on every closed face of the enclosed box, on the
     ground and on the rigid sphere surface.

The rigid sphere and the enclosing walls are represented by closed MAC faces,
so no fluid crosses a solid boundary.  The sphere and domain are the frozen
scene geometry; the transported scalar is the smoke density.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard, platform always provides the contract
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field as _field
    from collections.abc import Mapping

    @dataclass(slots=True)
    class StateFrame:  # type: ignore[no-redef]
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: np.ndarray | None = None
        angular_velocity: np.ndarray | None = None
        mass: np.ndarray | None = None
        body_id: np.ndarray | None = None
        attributes: Mapping = _field(default_factory=dict)
        fields: Mapping = _field(default_factory=dict)
        scalars: Mapping = _field(default_factory=dict)

        def validate(self) -> None:
            return None


# ----------------------------------------------------------------------------
# Frozen scene constants (from scene_geometry.yaml / config.yaml)
# ----------------------------------------------------------------------------
DOMAIN_MIN = np.array([-3.0, -1.2, 0.0])
DOMAIN_MAX = np.array([3.0, 1.2, 3.8])
SPHERE_CENTER = np.array([0.0, 0.0, 1.35])
SPHERE_RADIUS = 0.48
EMITTER_RADIUS = 0.32
EMITTER_TOP = 0.10            # cylinder spans z in [0, 0.10]
OUT_SHAPE = (24, 16, 28)      # field_grid_shape from config.yaml

GRAVITY = 9.81                # |g|, z is the up axis
BUOYANCY_ALPHA = 1.6          # m/s^2 of upward acceleration per unit smoke
SMOKE_MAX = 1.0
SOURCE_RATE = 6.0             # scalar units per second inside the emitter
CFL = 1.0
MAX_DT = 0.025
MIN_DT = 1.0e-5
PRESSURE_ITERS = 60
GRID_SPACING = 0.18           # frozen diagnostic spacing (config sampling)
CHARACTERISTIC_LENGTH = 0.96  # frozen normalization length (sphere diameter)
REFERENCE_VELOCITY = 1.0

_N_SPHERE_SAMPLES = 128
_N_GROUND_X = 8
_N_GROUND_Y = 12


class BenchmarkSolver:
    """Advance a buoyant smoke plume around a fixed sphere."""

    # -- lifecycle -----------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.time = 0.0
        self.step_count = 0
        self._diag: dict[str, object] = {}

        cfg = getattr(scene, "config", {}) or {}
        self.config = cfg

        # --- simulation grid: uniform spacing inside the frozen domain ------
        h = 0.1
        self.h = h
        span = DOMAIN_MAX - DOMAIN_MIN
        self.nx, self.ny, self.nz = (int(round(span[i] / h)) for i in range(3))
        nx, ny, nz = self.nx, self.ny, self.nz

        self.xc = DOMAIN_MIN[0] + (np.arange(nx) + 0.5) * h
        self.yc = DOMAIN_MIN[1] + (np.arange(ny) + 0.5) * h
        self.zc = DOMAIN_MIN[2] + (np.arange(nz) + 0.5) * h
        self.xw = DOMAIN_MIN[0] + np.arange(nx + 1) * h
        self.yw = DOMAIN_MIN[1] + np.arange(ny + 1) * h
        self.zw = DOMAIN_MIN[2] + np.arange(nz + 1) * h

        self.origin_cell = (self.xc[0], self.yc[0], self.zc[0])
        self.origin_ux = (self.xw[0], self.yc[0], self.zc[0])
        self.origin_uy = (self.xc[0], self.yw[0], self.zc[0])
        self.origin_uz = (self.xc[0], self.yc[0], self.zw[0])
        self.sp = (h, h, h)

        # --- solid / fluid masks -------------------------------------------
        X, Y, Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        dist = np.sqrt((X - SPHERE_CENTER[0]) ** 2
                       + (Y - SPHERE_CENTER[1]) ** 2
                       + (Z - SPHERE_CENTER[2]) ** 2)
        self.solid = dist < SPHERE_RADIUS
        self.fluid = ~self.solid
        self.cell_volume = h ** 3

        # faces: open only when they separate two fluid cells and are interior
        f = self.fluid
        open_x = np.zeros((nx + 1, ny, nz), dtype=bool)
        open_x[1:nx] = f[:-1] & f[1:]
        open_y = np.zeros((nx, ny + 1, nz), dtype=bool)
        open_y[:, 1:ny] = f[:, :-1] & f[:, 1:]
        open_z = np.zeros((nx, ny, nz + 1), dtype=bool)
        open_z[:, :, 1:nz] = f[:, :, :-1] & f[:, :, 1:]
        self.open_x, self.open_y, self.open_z = open_x, open_y, open_z

        # Poisson connectivity coefficients (1 open, 0 wall / solid)
        self.cxp = open_x[1:nx + 1]
        self.cxm = open_x[0:nx]
        self.cyp = open_y[:, 1:ny + 1]
        self.cym = open_y[:, 0:ny]
        self.czp = open_z[:, :, 1:nz + 1]
        self.czm = open_z[:, :, 0:nz]
        ncoef = (self.cxp.astype(np.float64) + self.cxm +
                 self.cyp.astype(np.float64) + self.cym +
                 self.czp.astype(np.float64) + self.czm)
        self.ncoef = np.maximum(ncoef, 1.0)
        self.inv_ncoef = 1.0 / self.ncoef
        self.fluid_f = f.astype(np.float64)

        # --- fields --------------------------------------------------------
        self.UX = np.zeros((nx + 1, ny, nz))
        self.UY = np.zeros((nx, ny + 1, nz))
        self.UZ = np.zeros((nx, ny, nz + 1))
        self.s = np.zeros((nx, ny, nz))
        self.p = np.zeros((nx, ny, nz))
        self._apply_masks()

        # scalar source profile (fixed in space, zero away from the emitter)
        ecc = np.zeros((nx, ny, nz))
        rxy = np.sqrt(X ** 2 + Y ** 2)
        inside = (rxy <= EMITTER_RADIUS) & (Z <= EMITTER_TOP)
        ecc[inside] = SOURCE_RATE * (1.0 - 0.5 * (rxy[inside] / EMITTER_RADIUS) ** 2)
        self.source_rate = ecc

        # --- diagnostic (collocated) sampling locations ---------------------
        self.su = self._flat_mesh(self.xw, self.yc, self.zc)
        self.sv = self._flat_mesh(self.xc, self.yw, self.zc)
        self.sw = self._flat_mesh(self.xc, self.yc, self.zw)
        self.scc = self._flat_mesh(self.xc, self.yc, self.zc)

        # --- initial smoke from the frozen scene ---------------------------
        try:
            initial = scene.initial_state()
        except Exception:
            initial = None
        if initial is not None and "transported_scalar" in initial.fields:
            coarse = np.asarray(initial.fields["transported_scalar"], dtype=np.float64)
            if coarse.shape == OUT_SHAPE:
                ox = np.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], OUT_SHAPE[0])
                oy = np.linspace(DOMAIN_MIN[1], DOMAIN_MAX[1], OUT_SHAPE[1])
                oz = np.linspace(DOMAIN_MIN[2], DOMAIN_MAX[2], OUT_SHAPE[2])
                d = (ox[1] - ox[0], oy[1] - oy[0], oz[1] - oz[0])
                self.s = self._sample(coarse, (ox[0], oy[0], oz[0]), d,
                                      self.scc[0], self.scc[1], self.scc[2])
                self.s *= self.fluid_f
        self.s = np.clip(self.s, 0.0, SMOKE_MAX)

        # --- passive tracer particles --------------------------------------
        if initial is not None:
            self.pos = np.asarray(initial.position, dtype=np.float64).copy()
            self.pvel = np.zeros_like(self.pos)
            if initial.mass is not None:
                self.pmass = np.asarray(initial.mass, dtype=np.float64).copy()
            else:
                self.pmass = np.full(len(self.pos), 1.2 * GRID_SPACING ** 3)
            self.body = (np.asarray(initial.body_id).copy()
                         if initial.body_id is not None
                         else np.zeros(len(self.pos), dtype=np.int32))
        else:
            self.pos = np.zeros((0, 3))
            self.pvel = np.zeros((0, 3))
            self.pmass = np.zeros(0)
            self.body = np.zeros(0, dtype=np.int32)
        self.pradius = np.full(len(self.pos), 0.45 * GRID_SPACING)

        # --- fixed boundary sample set -------------------------------------
        self.bnd_pos, self.bnd_nrm = self._build_boundary_samples()

        self._diag = {"grid": [nx, ny, nz], "h": h}

    # -- integration ---------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1.0e-12 and guard < 500000:
            dt = min(self._stable_dt(), target - self.time, MAX_DT)
            if dt < MIN_DT:
                dt = min(target - self.time, MIN_DT)
            if dt <= 0.0:
                break
            self._step(dt)
            self.time += dt
            guard += 1
        self.time = target
        return self._frame(target)

    def _stable_dt(self) -> float:
        vmax = 0.0
        if self.open_x.any():
            vmax = max(vmax, float(np.max(np.abs(self.UX[self.open_x]))))
        if self.open_y.any():
            vmax = max(vmax, float(np.max(np.abs(self.UY[self.open_y]))))
        if self.open_z.any():
            vmax = max(vmax, float(np.max(np.abs(self.UZ[self.open_z]))))
        if not np.isfinite(vmax):
            vmax = 1.0
        return CFL * self.h / max(vmax, 1.0e-3)

    def _step(self, dt: float) -> None:
        self._apply_masks()

        # 1. buoyancy (gravity aligned, acts on the smoke scalar)
        sc = self.s
        face_s = 0.5 * (sc[:, :, :-1] + sc[:, :, 1:])
        self.UZ[:, :, 1:self.nz] += dt * BUOYANCY_ALPHA * face_s

        # 2. semi-Lagrangian self-advection of velocity
        self._advect_velocity(dt)
        self._apply_masks()

        # 3. BFECC advection of the smoke scalar
        s1, lo, hi = self._advect_scalar(self.s, dt)
        s2, _, _ = self._advect_scalar(s1, -dt)
        s_new = s1 + 0.5 * (self.s - s2)
        s_new = np.clip(s_new, np.minimum(lo, s1), np.maximum(hi, s1))
        self.s = np.clip(s_new, 0.0, SMOKE_MAX) * self.fluid_f

        # 4. localised emitter source (saturating)
        if dt > 0.0:
            self.s = np.minimum(self.s + dt * self.source_rate, SMOKE_MAX) * self.fluid_f

        # 5. pressure projection (divergence free, no penetration on walls)
        self._project(dt)
        self._apply_masks()

        # 6. passive tracers ride the fluid and respect the solid
        self._advect_particles(dt)

        self.step_count += 1

    # -- operators -----------------------------------------------------------
    def _apply_masks(self) -> None:
        self.UX *= self.open_x
        self.UY *= self.open_y
        self.UZ *= self.open_z
        for arr in (self.UX, self.UY, self.UZ, self.s, self.p):
            arr[~np.isfinite(arr)] = 0.0

    def _advect_velocity(self, dt: float) -> None:
        px, py, pz = self.su
        u = self._sample(self.UX, self.origin_ux, self.sp, px, py, pz)
        v = self._sample(self.UY, self.origin_uy, self.sp, px, py, pz)
        w = self._sample(self.UZ, self.origin_uz, self.sp, px, py, pz)
        bx, by, bz = px - dt * u, py - dt * v, pz - dt * w
        self.UX = self._sample(self.UX, self.origin_ux, self.sp, bx, by, bz)

        px, py, pz = self.sv
        u = self._sample(self.UX, self.origin_ux, self.sp, px, py, pz)
        v = self._sample(self.UY, self.origin_uy, self.sp, px, py, pz)
        w = self._sample(self.UZ, self.origin_uz, self.sp, px, py, pz)
        bx, by, bz = px - dt * u, py - dt * v, pz - dt * w
        self.UY = self._sample(self.UY, self.origin_uy, self.sp, bx, by, bz)

        px, py, pz = self.sw
        u = self._sample(self.UX, self.origin_ux, self.sp, px, py, pz)
        v = self._sample(self.UY, self.origin_uy, self.sp, px, py, pz)
        w = self._sample(self.UZ, self.origin_uz, self.sp, px, py, pz)
        bx, by, bz = px - dt * u, py - dt * v, pz - dt * w
        self.UZ = self._sample(self.UZ, self.origin_uz, self.sp, bx, by, bz)

    def _advect_scalar(self, field, dt: float):
        px, py, pz = self.scc
        u = self._sample(self.UX, self.origin_ux, self.sp, px, py, pz)
        v = self._sample(self.UY, self.origin_uy, self.sp, px, py, pz)
        w = self._sample(self.UZ, self.origin_uz, self.sp, px, py, pz)
        bx, by, bz = px - dt * u, py - dt * v, pz - dt * w
        return self._sample(field, self.origin_cell, self.sp, bx, by, bz, bounds=True)

    def _project(self, dt: float) -> None:
        h = self.h
        h2 = h * h
        div = ((self.UX[1:, :, :] - self.UX[:-1, :, :])
               + (self.UY[:, 1:, :] - self.UY[:, :-1, :])
               + (self.UZ[:, :, 1:] - self.UZ[:, :, :-1])) / h
        div *= self.fluid_f
        rhs = div / max(dt, 1.0e-12)

        p = np.zeros((self.nx, self.ny, self.nz))
        pf = np.zeros((self.nx + 2, self.ny + 2, self.nz + 2))
        inv = self.inv_ncoef
        cxp, cxm = self.cxp, self.cxm
        cyp, cym = self.cyp, self.cym
        czp, czm = self.czp, self.czm
        for _ in range(PRESSURE_ITERS):
            pf[1:-1, 1:-1, 1:-1] = p
            pn = (cxp * pf[2:, 1:-1, 1:-1] + cxm * pf[:-2, 1:-1, 1:-1]
                  + cyp * pf[1:-1, 2:, 1:-1] + cym * pf[1:-1, :-2, 1:-1]
                  + czp * pf[1:-1, 1:-1, 2:] + czm * pf[1:-1, 1:-1, :-2]
                  - h2 * rhs) * inv
            p = pn * self.fluid_f
            p -= p[self.fluid].mean() if self.fluid.any() else 0.0
        self.p = p

        inv_h = 1.0 / h
        self.UX[1:self.nx] -= dt * (p[1:, :, :] - p[:-1, :, :]) * inv_h * self.open_x[1:self.nx]
        self.UY[:, 1:self.ny] -= dt * (p[:, 1:, :] - p[:, :-1, :]) * inv_h * self.open_y[:, 1:self.ny]
        self.UZ[:, :, 1:self.nz] -= dt * (p[:, :, 1:] - p[:, :, :-1]) * inv_h * self.open_z[:, :, 1:self.nz]
        self._apply_masks()

    # -- tracers -------------------------------------------------------------
    def _cell_velocity(self):
        u = 0.5 * (self.UX[:-1, :, :] + self.UX[1:, :, :]) * self.fluid_f
        v = 0.5 * (self.UY[:, :-1, :] + self.UY[:, 1:, :]) * self.fluid_f
        w = 0.5 * (self.UZ[:, :, :-1] + self.UZ[:, :, 1:]) * self.fluid_f
        return u, v, w

    def _advect_particles(self, dt: float) -> None:
        if len(self.pos) == 0:
            return
        u, v, w = self._cell_velocity()
        o, sp = self.origin_cell, self.sp
        p = self.pos
        k1 = np.column_stack((self._sample(u, o, sp, p[:, 0], p[:, 1], p[:, 2]),
                              self._sample(v, o, sp, p[:, 0], p[:, 1], p[:, 2]),
                              self._sample(w, o, sp, p[:, 0], p[:, 1], p[:, 2])))
        q = p + 0.5 * dt * k1
        k2 = np.column_stack((self._sample(u, o, sp, q[:, 0], q[:, 1], q[:, 2]),
                              self._sample(v, o, sp, q[:, 0], q[:, 1], q[:, 2]),
                              self._sample(w, o, sp, q[:, 0], q[:, 1], q[:, 2])))
        p = p + dt * k2

        # keep tracers out of the rigid sphere and inside the box
        d = p - SPHERE_CENTER
        r = np.linalg.norm(d, axis=1)
        shell = SPHERE_RADIUS + 0.03
        bad = r < shell
        if np.any(bad):
            rr = np.where(r[bad] > 1.0e-9, r[bad], 1.0e-9)
            p[bad] = SPHERE_CENTER + (d[bad] / rr[:, None]) * shell
        lo = DOMAIN_MIN + 0.01
        hi = DOMAIN_MAX - 0.01
        p = np.clip(p, lo, hi)
        self.pos = p
        self.pvel = k2

    # -- sampling helpers ----------------------------------------------------
    @staticmethod
    def _flat_mesh(a, b, c):
        A, B, C = np.meshgrid(a, b, c, indexing="ij")
        return A.ravel(), B.ravel(), C.ravel()

    def _sample(self, field, origin, spacing, px, py, pz, bounds: bool = False):
        """Trilinear sample of a uniform grid field at arbitrary points."""
        shape = field.shape
        ax = np.clip(px, origin[0], origin[0] + (shape[0] - 1) * spacing[0])
        ay = np.clip(py, origin[1], origin[1] + (shape[1] - 1) * spacing[1])
        az = np.clip(pz, origin[2], origin[2] + (shape[2] - 1) * spacing[2])

        i = np.clip(np.floor((ax - origin[0]) / spacing[0]).astype(np.int64), 0, shape[0] - 2)
        j = np.clip(np.floor((ay - origin[1]) / spacing[1]).astype(np.int64), 0, shape[1] - 2)
        k = np.clip(np.floor((az - origin[2]) / spacing[2]).astype(np.int64), 0, shape[2] - 2)
        fx = (ax - (origin[0] + i * spacing[0])) / spacing[0]
        fy = (ay - (origin[1] + j * spacing[1])) / spacing[1]
        fz = (az - (origin[2] + k * spacing[2])) / spacing[2]
        i1, j1, k1 = i + 1, j + 1, k + 1

        c000 = field[i, j, k]
        c100 = field[i1, j, k]
        c010 = field[i, j1, k]
        c110 = field[i1, j1, k]
        c001 = field[i, j, k1]
        c101 = field[i1, j, k1]
        c011 = field[i, j1, k1]
        c111 = field[i1, j1, k1]

        wx0, wx1 = 1.0 - fx, fx
        wy0, wy1 = 1.0 - fy, fy
        wz0, wz1 = 1.0 - fz, fz
        value = (wx0 * (wy0 * (wz0 * c000 + wz1 * c001) + wy1 * (wz0 * c010 + wz1 * c011))
                 + wx1 * (wy0 * (wz0 * c100 + wz1 * c101) + wy1 * (wz0 * c110 + wz1 * c111)))
        if not bounds:
            return value
        lo = np.minimum.reduce([c000, c100, c010, c110, c001, c101, c011, c111])
        hi = np.maximum.reduce([c000, c100, c010, c110, c001, c101, c011, c111])
        return value, lo, hi

    # -- boundary samples ----------------------------------------------------
    def _build_boundary_samples(self):
        pts = []
        nrm = []
        n = _N_SPHERE_SAMPLES
        idx = np.arange(n) + 0.5
        zz = 1.0 - 2.0 * idx / n
        rr = np.sqrt(np.maximum(1.0 - zz * zz, 0.0))
        golden = np.pi * (3.0 - np.sqrt(5.0))
        th = idx * golden
        sx = SPHERE_CENTER[0] + SPHERE_RADIUS * rr * np.cos(th)
        sy = SPHERE_CENTER[1] + SPHERE_RADIUS * rr * np.sin(th)
        sz = SPHERE_CENTER[2] + SPHERE_RADIUS * zz
        sphere_pts = np.column_stack((sx, sy, sz))
        sphere_nrm = (sphere_pts - SPHERE_CENTER) / SPHERE_RADIUS
        pts.append(sphere_pts)
        nrm.append(sphere_nrm)

        gx = np.linspace(DOMAIN_MIN[0] + 0.2, DOMAIN_MAX[0] - 0.2, _N_GROUND_X)
        gy = np.linspace(DOMAIN_MIN[1] + 0.2, DOMAIN_MAX[1] - 0.2, _N_GROUND_Y)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        ground_pts = np.column_stack((GX.ravel(), GY.ravel(), np.zeros(GX.size)))
        ground_nrm = np.tile(np.array([0.0, 0.0, 1.0]), (GX.size, 1))
        pts.append(ground_pts)
        nrm.append(ground_nrm)

        return np.vstack(pts), np.vstack(nrm)

    # -- frame export --------------------------------------------------------
    def _output_grid(self):
        ox = np.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], OUT_SHAPE[0])
        oy = np.linspace(DOMAIN_MIN[1], DOMAIN_MAX[1], OUT_SHAPE[1])
        oz = np.linspace(DOMAIN_MIN[2], DOMAIN_MAX[2], OUT_SHAPE[2])
        d = (ox[1] - ox[0], oy[1] - oy[0], oz[1] - oz[0])
        return (ox[0], oy[0], oz[0]), d, self._flat_mesh(ox, oy, oz)

    def _frame(self, time: float):
        origin, d, (px, py, pz) = self._output_grid()

        scalar = self._sample(self.s, self.origin_cell, self.sp, px, py, pz)
        scalar = np.clip(scalar, 0.0, SMOKE_MAX).reshape(OUT_SHAPE)

        src = self._sample(self.source_rate, self.origin_cell, self.sp, px, py, pz)
        src = np.maximum(src, 0.0).reshape(OUT_SHAPE)

        mask = self._sample(self.fluid_f, self.origin_cell, self.sp, px, py, pz)
        mask = (mask > 0.5).astype(np.float64).reshape(OUT_SHAPE)

        u, v, w = self._cell_velocity()
        ou = self._sample(u, self.origin_cell, self.sp, px, py, pz).reshape(OUT_SHAPE)
        ov = self._sample(v, self.origin_cell, self.sp, px, py, pz).reshape(OUT_SHAPE)
        ow = self._sample(w, self.origin_cell, self.sp, px, py, pz).reshape(OUT_SHAPE)
        velocity = np.stack((ou, ov, ow), axis=-1)
        velocity[~np.isfinite(velocity)] = 0.0

        # wall samples: outward normal, prescribed (static) target, measured fluid side
        bpts = self.bnd_pos
        bnrm = self.bnd_nrm
        bv = np.column_stack((
            self._sample(u, self.origin_cell, self.sp, bpts[:, 0], bpts[:, 1], bpts[:, 2]),
            self._sample(v, self.origin_cell, self.sp, bpts[:, 0], bpts[:, 1], bpts[:, 2]),
            self._sample(w, self.origin_cell, self.sp, bpts[:, 0], bpts[:, 1], bpts[:, 2]),
        ))
        bv[~np.isfinite(bv)] = 0.0
        bv = bv - (np.sum(bv * bnrm, axis=1, keepdims=True)) * bnrm  # no penetration
        btarget = np.zeros_like(bv)

        fields = {
            "active_mask": mask,
            "boundary_normal": bnrm.astype(np.float64),
            "boundary_target_velocity": btarget,
            "boundary_velocity": bv,
            "scalar_source": src,
            "transported_scalar": scalar,
            "velocity_field": velocity,
        }

        mass = self.pmass.copy()
        attributes = {
            "mass_quantity": mass.copy(),
            "particle_radius": self.pradius.copy(),
            "material_label": self.body.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], len(self.pos), axis=0),
        }
        scalars = {
            "grid_spacing": float(GRID_SPACING),
            "characteristic_length": float(CHARACTERISTIC_LENGTH),
            "reference_velocity": float(REFERENCE_VELOCITY),
        }

        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.pvel.copy(),
            mass=mass,
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        u, v, w = self._cell_velocity()
        speed = np.sqrt(u * u + v * v + w * w)
        out = dict(self._diag)
        out.update({
            "steps": int(self.step_count),
            "time": float(self.time),
            "max_speed": float(np.max(speed)) if speed.size else 0.0,
            "smoke_mass": float(np.sum(self.s) * self.cell_volume),
            "scalar_max": float(np.max(self.s)) if self.s.size else 0.0,
            "particles": int(len(self.pos)),
            "boundary_samples": int(self.bnd_pos.shape[0]),
        })
        return out
