"""Smoke driven by a rotating cube pump (fluid_dynamics / gas).

Physical model / numerical method (implemented from scratch with NumPy):

* Incompressible (Boussinesq) Navier-Stokes on a uniform staggered (MAC) grid
  covering the frozen ``subject`` bounds.
* Momentum is advanced with semi-Lagrangian advection, buoyancy from the
  transported smoke scalar, vorticity confinement (the mechanism used by the
  source paper to preserve the rolling vortical structures) and a
  variable-coefficient (openness weighted) pressure projection.
* The rotating cube is a masked moving boundary: faces touching the solid
  carry the prescribed rigid velocity ``omega x (x - c)``, solid/solid faces
  are closed, and the pressure Poisson equation is solved matrix-free with a
  preconditioned conjugate gradient so the discrete flux field stays
  divergence free and no fluid crosses a rigid boundary.
* The smoke scalar is a cell-centred transported field fed by a fixed emitter
  cylinder and advected with a bounded MacCormack (BFECC-style) scheme.
* Material samples (the frozen initial block cloud) are advected by the same
  velocity field and are pushed out of the rotating cube, so the represented
  fluid stays continuous and respects the rigid boundaries.

All state is deterministic: no random numbers, no keyframes, no precomputed
motion tables.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Runnable solver for scene 136_smoke_rotating_cube_pump."""

    # ------------------------------------------------------------------ init
    def __init__(self) -> None:
        self._ready = False
        self._t = 0.0
        self._substeps = 0
        self._diagnostics: dict[str, object] = {}

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _object(geometry: dict, identifier: str) -> dict:
        for item in geometry.get("objects", []) or []:
            if isinstance(item, dict) and item.get("id") == identifier:
                return item
        return {}

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        geometry = dict(cfg.get("scene_geometry", {}) or {})
        sampling = dict(cfg.get("sampling", {}) or {})

        self.seed = int(seed)
        self.scene_id = str(cfg.get("scene_id", "136_smoke_rotating_cube_pump"))
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.gravity = gravity
        self.g_mag = float(abs(gravity[2])) if abs(gravity[2]) > 1e-9 else 9.81

        # ---- smoke-traced subject grid (frozen bounds / resolution)
        subject = self._object(geometry, "subject").get("geometry", {}) or {}
        grid_shape = subject.get("grid_shape") or sampling.get("field_grid_shape") or [24, 16, 28]
        self.n = tuple(int(v) for v in grid_shape)
        lo = subject.get("bounds_min") or [-3.0, -1.2, 0.0]
        hi = subject.get("bounds_max") or [3.0, 1.2, 3.8]
        self.lo = np.asarray(lo, dtype=float).reshape(3)
        self.hi = np.asarray(hi, dtype=float).reshape(3)
        self.nx, self.ny, self.nz = self.n
        self.h = (self.hi - self.lo) / np.asarray(self.n, dtype=float)
        self.hx, self.hy, self.hz = (float(v) for v in self.h)
        self.hmin = float(min(self.hx, self.hy, self.hz))

        xc = self.lo[0] + (np.arange(self.nx) + 0.5) * self.hx
        yc = self.lo[1] + (np.arange(self.ny) + 0.5) * self.hy
        zc = self.lo[2] + (np.arange(self.nz) + 0.5) * self.hz
        self.xc, self.yc, self.zc = xc, yc, zc
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")
        self.P = np.stack((self.X, self.Y, self.Z), axis=-1)

        # face coordinates
        xf = self.lo[0] + np.arange(self.nx + 1) * self.hx
        yf = self.lo[1] + np.arange(self.ny + 1) * self.hy
        zf = self.lo[2] + np.arange(self.nz + 1) * self.hz
        Xf, Yc2, Zc2 = np.meshgrid(xf, yc, zc, indexing="ij")
        self.Pu = np.stack((Xf, Yc2, Zc2), axis=-1)
        Xc2, Yf, Zc3 = np.meshgrid(xc, yf, zc, indexing="ij")
        self.Pv = np.stack((Xc2, Yf, Zc3), axis=-1)
        Xc3, Yc4, Zf = np.meshgrid(xc, yc, zf, indexing="ij")
        self.Pw = np.stack((Xc3, Yc4, Zf), axis=-1)

        # staggered interpolation origins / spacings (index 0 position)
        self.orig_u = np.array([self.lo[0], self.lo[1] + 0.5 * self.hy, self.lo[2] + 0.5 * self.hz])
        self.orig_v = np.array([self.lo[0] + 0.5 * self.hx, self.lo[1], self.lo[2] + 0.5 * self.hz])
        self.orig_w = np.array([self.lo[0] + 0.5 * self.hx, self.lo[1] + 0.5 * self.hy, self.lo[2]])
        self.orig_c = self.lo + 0.5 * self.h
        self.sp = self.h

        # ---- rotating cube (kinematic rigid body)
        cube = self._object(geometry, "rotating_cube")
        cg = cube.get("geometry", {}) or {}
        ct = cube.get("transform", {}) or {}
        half = cg.get("half_extents") or [0.45, 0.45, 0.45]
        if isinstance(half, (int, float)):
            half = [float(half)] * 3
        self.cube_half = np.asarray(half, dtype=float).reshape(3)
        self.cube_center = np.asarray(
            ct.get("translation", [0.0, 0.0, 1.55]), dtype=float
        ).reshape(3)
        # prescribed rotation: constant angular speed about the +y axis
        self.omega_axis = np.array([0.0, 1.0, 0.0])
        self.omega = 2.0  # rad/s -> corner tip speed ~1.3 m/s, plume-like regime
        self.omega_vec = self.omega_axis * self.omega
        self.cube_reach = float(np.linalg.norm(self.cube_half))
        self.cube_traj = float(np.pi * self.cube_reach)  # swept distance / rev

        # ---- smoke emitter (static cylinder)
        src = self._object(geometry, "smoke_source")
        sg = src.get("geometry", {}) or {}
        st = src.get("transform", {}) or {}
        self.src_center = np.asarray(st.get("translation", [0.0, 0.0, 0.06]), dtype=float).reshape(3)
        self.src_radius = float(sg.get("radius", 0.34))
        self.src_height = float(sg.get("height", 0.12))
        self.src_top = self.src_center[2] + 0.5 * self.src_height
        self.src_strength = 5.0  # concentration per second

        r2 = (self.X - self.src_center[0]) ** 2 + (self.Y - self.src_center[1]) ** 2
        self.emitter = (r2 <= self.src_radius ** 2) & (self.Z <= self.src_top + 1e-9)
        self.scalar_source = np.where(self.emitter, self.src_strength, 0.0)

        # frozen normalisation scales
        self.characteristic_length = float(2.0 * self.cube_half.max())
        self.grid_spacing = float(np.mean(self.h))
        self.reference_velocity = float(np.sqrt(self.g_mag * self.characteristic_length))
        self.buoyancy = self.g_mag / 6.0     # Boussinesq coupling: f = B * c * (-g_hat)
        self.scalar_cap = 8.0                # buoyancy saturation concentration
        self.vort_eps = 1.5                  # vorticity confinement strength
        self.vort_limit = 3.0                # confinement acceleration clamp [m/s^2]
        self.speed_limit = 8.0               # numerical safety clamp

        # ---- fields
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.u0 = np.zeros_like(self.u)
        self.v0 = np.zeros_like(self.v)
        self.w0 = np.zeros_like(self.w)
        self.pressure = np.zeros(self.n)
        # frozen initial smoke distribution (same analytic blob as the scaffold)
        self.scalar = 1.20 * np.exp(
            -(
                ((self.X - 0.0) / 0.40) ** 2
                + ((self.Y - 0.0) / 0.40) ** 2
                + ((self.Z - 0.68) / 0.62) ** 2
            )
        )
        self.fluid = np.ones(self.n, dtype=bool)

        # ---- moving-boundary sample set (static walls + rotating cube faces)
        self._build_wall_samples()

        # ---- material samples (frozen initial cloud from the scene)
        self.position = np.zeros((0, 3))
        self.velocity = np.zeros((0, 3))
        self.mass = np.zeros((0,))
        self.body_id = np.zeros((0,), dtype=np.int32)
        try:
            initial = scene.initial_state()
            self.position = np.array(initial.position, dtype=float).reshape(-1, 3)
            self.velocity = np.array(initial.velocity, dtype=float).reshape(-1, 3)
            if initial.mass is not None:
                self.mass = np.array(initial.mass, dtype=float).reshape(-1)
        except Exception:
            pass
        if self.position.shape[0] == 0:
            self.position = np.array([[0.0, 0.0, 0.3]])
            self.velocity = np.zeros_like(self.position)
        if self.mass.shape[0] != self.position.shape[0]:
            self.mass = np.full(self.position.shape[0], 1.0 / max(1, self.position.shape[0]))
        self.body_id = np.zeros(self.position.shape[0], dtype=np.int32)
        self.material_label = np.zeros(self.position.shape[0], dtype=np.int32)
        self.particle_radius = np.full(
            self.position.shape[0], 0.45 * float(sampling.get("spacing", 0.18))
        )

        self._ready = True
        self._t = 0.0
        self._substeps = 0
        self._diagnostics = {}

    # ------------------------------------------------------- static geometry
    def _build_wall_samples(self) -> None:
        pts, nrm = [], []
        nx_, ny_, nz_ = self.n
        xs = np.linspace(self.lo[0], self.hi[0], 7)
        ys = np.linspace(self.lo[1], self.hi[1], 5)
        zs = np.linspace(self.lo[2], self.hi[2], 5)
        # closed box: floor, ceiling and four side walls
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts.append(np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, self.lo[2]))))
        nrm.append(np.tile(np.array([0.0, 0.0, 1.0]), (gx.size, 1)))
        pts.append(np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, self.hi[2]))))
        nrm.append(np.tile(np.array([0.0, 0.0, -1.0]), (gx.size, 1)))
        gy2, gz2 = np.meshgrid(ys, zs, indexing="ij")
        for xw, sgn in ((self.lo[0], 1.0), (self.hi[0], -1.0)):
            pts.append(np.column_stack((np.full(gy2.size, xw), gy2.ravel(), gz2.ravel())))
            nrm.append(np.tile(np.array([sgn, 0.0, 0.0]), (gy2.size, 1)))
        gx2, gz3 = np.meshgrid(xs, zs, indexing="ij")
        for yw, sgn in ((self.lo[1], 1.0), (self.hi[1], -1.0)):
            pts.append(np.column_stack((gx2.ravel(), np.full(gx2.size, yw), gz3.ravel())))
            nrm.append(np.tile(np.array([0.0, sgn, 0.0]), (gx2.size, 1)))
        self.wall_pts = np.concatenate(pts, axis=0)
        self.wall_nrm = np.concatenate(nrm, axis=0)

        # rotating cube surface samples in the cube frame (rebuilt every frame)
        face_pts, face_nrm, face_ax = [], [], []
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            lin = [
                np.linspace(-self.cube_half[a] * 0.8, self.cube_half[a] * 0.8, 4) for a in other
            ]
            ga, gb = np.meshgrid(lin[0], lin[1], indexing="ij")
            for sgn in (1.0, -1.0):
                loc = np.zeros((ga.size, 3))
                loc[:, other[0]] = ga.ravel()
                loc[:, other[1]] = gb.ravel()
                loc[:, axis] = sgn * self.cube_half[axis]
                n = np.zeros((ga.size, 3))
                n[:, axis] = sgn
                face_pts.append(loc)
                face_nrm.append(n)
                face_ax.append(np.full(ga.size, axis, dtype=int))
        self.cube_loc = np.concatenate(face_pts, axis=0)
        self.cube_nrm_loc = np.concatenate(face_nrm, axis=0)

    # ----------------------------------------------------------- kinematics
    def _rotation(self, t: float) -> np.ndarray:
        ang = self.omega * float(t)
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _solid_mask(self, rot: np.ndarray) -> np.ndarray:
        d = self.P - self.cube_center
        loc = d @ rot
        return np.all(np.abs(loc) <= self.cube_half + 1e-12, axis=-1)

    def _rigid_velocity(self, pts: np.ndarray) -> np.ndarray:
        d = pts - self.cube_center
        return np.cross(self.omega_vec[None, :], d)

    def _openness(self, fluid: np.ndarray):
        bx = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        by = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        bz = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        bx[1:-1] = fluid[:-1] & fluid[1:]
        by[:, 1:-1] = fluid[:, :-1] & fluid[:, 1:]
        bz[:, :, 1:-1] = fluid[:, :, :-1] & fluid[:, :, 1:]
        return bx, by, bz

    def _impose_solid_faces(self) -> None:
        """Rigid velocity on every face touching the rotating cube."""
        solid = self._solid_cell
        touch_u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        touch_u[1:-1] = solid[:-1] | solid[1:]
        touch_v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        touch_v[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        touch_w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        touch_w[:, :, 1:-1] = solid[:, :, :-1] | solid[:, :, 1:]
        if touch_u.any():
            vel = self._rigid_velocity(self.Pu[touch_u])
            self.u[touch_u] = vel[:, 0]
        if touch_v.any():
            vel = self._rigid_velocity(self.Pv[touch_v])
            self.v[touch_v] = vel[:, 1]
        if touch_w.any():
            vel = self._rigid_velocity(self.Pw[touch_w])
            self.w[touch_w] = vel[:, 2]
        # closed box walls (no-slip)
        self.u[0, :, :] = 0.0
        self.u[-1, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    # ---------------------------------------------------------- interpolation
    def _trilerp(self, field, origin, spacing, pts, bounds=False):
        a, b, c = field.shape
        idx = (pts - np.asarray(origin)) / np.asarray(spacing)
        idx = np.clip(idx, 0.0, np.array([a - 1.0, b - 1.0, c - 1.0]) - 1e-9)
        i0 = np.floor(idx).astype(np.int64)
        i0 = np.clip(i0, 0, np.array([a, b, c]) - 2)
        tt = idx - i0
        out = np.zeros(pts.shape[0])
        lo = hi = None
        for di in (0, 1):
            wx = (1.0 - tt[:, 0]) if di == 0 else tt[:, 0]
            for dj in (0, 1):
                wy = (1.0 - tt[:, 1]) if dj == 0 else tt[:, 1]
                for dk in (0, 1):
                    wz = (1.0 - tt[:, 2]) if dk == 0 else tt[:, 2]
                    val = field[i0[:, 0] + di, i0[:, 1] + dj, i0[:, 2] + dk]
                    out += wx * wy * wz * val
                    if bounds:
                        lo = val.copy() if lo is None else np.minimum(lo, val)
                        hi = val.copy() if hi is None else np.maximum(hi, val)
        if bounds:
            return out, lo, hi
        return out

    def _velocity_at(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return np.zeros((0, 3))
        uu = self._trilerp(self.u, self.orig_u, self.sp, pts)
        vv = self._trilerp(self.v, self.orig_v, self.sp, pts)
        ww = self._trilerp(self.w, self.orig_w, self.sp, pts)
        return np.column_stack((uu, vv, ww))

    def _cell_velocity(self) -> np.ndarray:
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return np.stack((uc, vc, wc), axis=-1)

    def _advect_face(self, field, origin, pts, dt):
        back = pts - dt * self._velocity_at(pts)
        return self._trilerp(field, origin, self.sp, back)

    # -------------------------------------------------------- pressure solve
    def _set_operators(self, bx, by, bz, fluid) -> None:
        self._bx = bx.astype(float)
        self._by = by.astype(float)
        self._bz = bz.astype(float)
        self._bxp = bx[1:].astype(float)
        self._bxm = bx[:-1].astype(float)
        self._byp = by[:, 1:].astype(float)
        self._bym = by[:, :-1].astype(float)
        self._bzp = bz[:, :, 1:].astype(float)
        self._bzm = bz[:, :, :-1].astype(float)
        self._fluid = fluid.astype(float)
        self._solidf = 1.0 - self._fluid
        self._diag = (
            (self._bxp + self._bxm) / self.hx ** 2
            + (self._byp + self._bym) / self.hy ** 2
            + (self._bzp + self._bzm) / self.hz ** 2
        )
        self._diag = np.where(self._diag > 1e-12, self._diag, 1.0)

    def _apply_A(self, p: np.ndarray) -> np.ndarray:
        pp = np.concatenate((p[1:], p[-1:]), axis=0)
        pm = np.concatenate((p[:1], p[:-1]), axis=0)
        out = (self._bxp * (pp - p) - self._bxm * (p - pm)) / self.hx ** 2
        pp = np.concatenate((p[:, 1:], p[:, -1:]), axis=1)
        pm = np.concatenate((p[:, :1], p[:, :-1]), axis=1)
        out += (self._byp * (pp - p) - self._bym * (p - pm)) / self.hy ** 2
        pp = np.concatenate((p[:, :, 1:], p[:, :, -1:]), axis=2)
        pm = np.concatenate((p[:, :, :1], p[:, :, :-1]), axis=2)
        out += (self._bzp * (pp - p) - self._bzm * (p - pm)) / self.hz ** 2
        return out * self._fluid + p * self._solidf

    def _project(self) -> None:
        bxp, bxm = self._bxp, self._bxm
        byp, bym = self._byp, self._bym
        bzp, bzm = self._bzp, self._bzm
        div = (bxp * self.u[1:] - bxm * self.u[:-1]) / self.hx
        div += (byp * self.v[:, 1:] - bym * self.v[:, :-1]) / self.hy
        div += (bzp * self.w[:, :, 1:] - bzm * self.w[:, :, :-1]) / self.hz
        div = div * self._fluid
        mask_sum = float(self._fluid.sum())
        if mask_sum < 1.0:
            return
        scale = float(np.abs(div).max())
        if scale < 1e-12:
            return
        rhs = -div
        rhs = rhs - (float((rhs * self._fluid).sum()) / mask_sum) * self._fluid

        p = self.pressure * self._fluid
        ap = self._apply_A(p)
        r = rhs - ap
        r = r - (float((r * self._fluid).sum()) / mask_sum) * self._fluid
        z = r / self._diag
        d = z.copy()
        rz = float((r * z).sum())
        norm0 = float(np.sqrt((rhs * rhs).sum()))
        tol = 1e-5 * max(norm0, 1e-30)
        for _ in range(120):
            if float(np.sqrt((r * r).sum())) <= tol:
                break
            ad = self._apply_A(d)
            denom = float((d * ad).sum())
            if not np.isfinite(denom) or abs(denom) < 1e-30:
                break
            alpha = rz / denom
            p = p + alpha * d
            r = r - alpha * ad
            r = r - (float((r * self._fluid).sum()) / mask_sum) * self._fluid
            z = r / self._diag
            rz_new = float((r * z).sum())
            if not np.isfinite(rz_new) or abs(rz) < 1e-30:
                rz = rz_new
                d = z.copy()
                continue
            beta = rz_new / rz
            d = z + beta * d
            rz = rz_new
        p = p - (float((p * self._fluid).sum()) / mask_sum) * self._fluid
        self.pressure = p
        # subtract the pressure gradient on every open face
        pr = np.concatenate((p, p[-1:]), axis=0)
        pl = np.concatenate((p[:1], p), axis=0)
        self.u -= self._bx * ((pr - pl) / self.hx)
        pr = np.concatenate((p[:, 1:], p[:, -1:]), axis=1)
        pl = np.concatenate((p[:, :1], p), axis=1)
        self.v -= self._by * ((pr - pl) / self.hy)
        pr = np.concatenate((p[:, :, 1:], p[:, :, -1:]), axis=2)
        pl = np.concatenate((p[:, :, :1], p), axis=2)
        self.w -= self._bz * ((pr - pl) / self.hz)

    # ------------------------------------------------------------- one step
    def _step(self, dt: float, t: float) -> None:
        rot = self._rotation(t)
        solid = self._solid_mask(rot)
        fluid = ~solid
        self._solid_cell = solid
        bx, by, bz = self._openness(fluid)
        self._set_operators(bx, by, bz, fluid)
        self._impose_solid_faces()

        # ---- momentum advection (semi-Lagrangian, unconditionally stable)
        self.u0 = self._advect_face(self.u, self.orig_u, self.Pu, dt)
        self.v0 = self._advect_face(self.v, self.orig_v, self.Pv, dt)
        self.w0 = self._advect_face(self.w, self.orig_w, self.Pw, dt)
        self.u, self.v, self.w = self.u0, self.v0, self.w0
        self._impose_solid_faces()

        # ---- body forces (cell centred, mapped to faces)
        velc = self._cell_velocity()
        force = np.zeros_like(velc)
        # buoyancy: light smoke rises against gravity (Boussinesq)
        smoke = np.clip(self.scalar, 0.0, self.scalar_cap) * self._fluid
        force[:, :, :, 2] = -self.gravity[2] * (smoke / 6.0)
        # vorticity confinement (keeps the pumped rolling structures coherent)
        grad = np.gradient(velc, self.hx, self.hy, self.hz, axis=(0, 1, 2))
        dw_dy, dv_dz = grad[1][:, :, :, 2], grad[2][:, :, :, 1]
        du_dz, dw_dx = grad[2][:, :, :, 0], grad[0][:, :, :, 2]
        dv_dx, du_dy = grad[0][:, :, :, 1], grad[1][:, :, :, 0]
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        mag = np.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)
        gx = np.zeros_like(mag)
        gy = np.zeros_like(mag)
        gz = np.zeros_like(mag)
        gx[1:-1] = (mag[2:, :, :] - mag[:-2, :, :]) / (2.0 * self.hx)
        gy[:, 1:-1] = (mag[:, 2:, :] - mag[:, :-2, :]) / (2.0 * self.hy)
        gz[:, :, 1:-1] = (mag[:, :, 2:] - mag[:, :, :-2]) / (2.0 * self.hz)
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        ok = gmag > 1e-9
        nx_ = np.zeros_like(gx)
        ny_ = np.zeros_like(gy)
        nz_ = np.zeros_like(gz)
        nx_[ok] = gx[ok] / gmag[ok]
        ny_[ok] = gy[ok] / gmag[ok]
        nz_[ok] = gz[ok] / gmag[ok]
        fx = self.vort_eps * self.hmin * (ny_ * curl_z - nz_ * curl_y)
        fy = self.vort_eps * self.hmin * (nz_ * curl_x - nx_ * curl_z)
        fz = self.vort_eps * self.hmin * (nx_ * curl_y - ny_ * curl_x)
        fmag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        lim = np.where(fmag > self.vort_limit, self.vort_limit / np.maximum(fmag, 1e-30), 1.0)
        force[:, :, :, 0] += fx * lim * self._fluid
        force[:, :, :, 1] += fy * lim * self._fluid
        force[:, :, :, 2] += fz * lim * self._fluid
        force = np.where(np.isfinite(force), force, 0.0)

        fxp = np.zeros_like(self.u)
        fxp[1:-1] = 0.5 * (force[:-1, :, :, 0] + force[1:, :, :, 0])
        fyp = np.zeros_like(self.v)
        fyp[:, 1:-1] = 0.5 * (force[:, :-1, :, 1] + force[:, 1:, :, 1])
        fzp = np.zeros_like(self.w)
        fzp[:, :, 1:-1] = 0.5 * (force[:, :, :-1, 2] + force[:, :, 1:, 2])
        self.u += dt * fxp * bx
        self.v += dt * fyp * by
        self.w += dt * fzp * bz
        self._impose_solid_faces()

        # ---- enforce incompressibility with the masked pressure projection
        self._project()
        self._impose_solid_faces()
        np.clip(self.u, -self.speed_limit, self.speed_limit, out=self.u)
        np.clip(self.v, -self.speed_limit, self.speed_limit, out=self.v)
        np.clip(self.w, -self.speed_limit, self.speed_limit, out=self.w)
        self.u = np.where(np.isfinite(self.u), self.u, 0.0)
        self.v = np.where(np.isfinite(self.v), self.v, 0.0)
        self.w = np.where(np.isfinite(self.w), self.w, 0.0)

        # ---- smoke transport (bounded MacCormack + emitter source)
        velc = self._cell_velocity()
        back = self.P - dt * velc
        c_hat = self._trilerp(self.scalar, self.orig_c, self.sp, back)
        fwd = self.P + dt * velc
        c_hat_hat = self._trilerp(c_hat, self.orig_c, self.sp, fwd)
        c_new, clo, chi = self._trilerp(self.scalar, self.orig_c, self.sp, back, bounds=True)
        c_mac = c_new + 0.5 * (self.scalar - c_hat_hat)
        c_new = np.clip(c_mac, clo, chi)
        c_new = c_new + dt * self.scalar_source
        c_new = np.clip(c_new, 0.0, 64.0)
        self.scalar = np.where(np.isfinite(c_new), c_new, 0.0)

        # ---- material samples: advected by the same flow, pushed out of solids
        self._advance_particles(dt, rot)

    # -------------------------------------------------------------- samples
    def _advance_particles(self, dt: float, rot: np.ndarray) -> None:
        if self.position.shape[0] == 0:
            return
        p1 = self.position + 0.5 * dt * self.velocity
        v1 = self._velocity_at(p1)
        self.position = self.position + dt * v1
        # rigid-body exclusion: project samples out of the rotating cube
        d = self.position - self.cube_center
        loc = d @ rot
        inside = np.all(np.abs(loc) < self.cube_half, axis=1)
        if np.any(inside):
            l = loc[inside]
            pen = self.cube_half[None, :] - np.abs(l)
            axis = np.argmax(pen, axis=1)
            rows = np.arange(l.shape[0])
            sgn = np.sign(l[rows, axis])
            sgn[sgn == 0.0] = 1.0
            l[rows, axis] = sgn * (self.cube_half[axis] + 1e-3)
            loc[inside] = l
            self.position = self.cube_center[None, :] + loc @ rot.T
        margin = 1e-4
        self.position[:, 0] = np.clip(self.position[:, 0], self.lo[0] + margin, self.hi[0] - margin)
        self.position[:, 1] = np.clip(self.position[:, 1], self.lo[1] + margin, self.hi[1] - margin)
        self.position[:, 2] = np.clip(self.position[:, 2], self.lo[2] + margin, self.hi[2] - margin)
        self.velocity = self._velocity_at(self.position)

    def _boundary_samples(self, rot: np.ndarray):
        cube_world = self.cube_center[None, :] + self.cube_loc @ rot.T
        cube_nrm = self.cube_nrm_loc @ rot.T
        pts = np.concatenate((self.wall_pts, cube_world), axis=0)
        nrm = np.concatenate((self.wall_nrm, cube_nrm), axis=0)
        tgt = np.zeros_like(pts)
        tgt[len(self.wall_pts):] = self._rigid_velocity(cube_world)
        return pts, nrm, tgt

    # ------------------------------------------------------------- stepping
    def _pick_dt(self, target: float) -> float:
        vel = self._cell_velocity()
        speed = float(np.sqrt((vel ** 2).sum(axis=-1)).max()) if vel.size else 0.0
        dt_cfl = 0.9 * self.hmin / max(speed, 1e-6)
        dt = min(dt_cfl, self.sample_dt, target - self._t)
        if dt <= 1e-9:
            dt = max(target - self._t, 1e-9)
        return float(dt)

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must run first")
        target = float(time)
        for _ in range(64):
            if self._t >= target - 1e-12:
                break
            dt = self._pick_dt(target)
            self._step(dt, self._t)
            self._t = self._t + dt
            self._substeps += 1
        self._t = target
        return self._frame()

    # ------------------------------------------------------------ frame out
    def _frame(self):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must run first")
        rot = self._rotation(self._t)
        pts, nrm, tgt = self._boundary_samples(rot)
        velc = self._cell_velocity()
        solid = self._solid_mask(rot)
        active = (~solid).astype(np.float64)
        fields = {
            "transported_scalar": np.array(self.scalar, dtype=float),
            "scalar_source": np.array(self.scalar_source, dtype=float),
            "active_mask": active.astype(bool),
            "velocity_field": velc,
            "boundary_normal": nrm,
            "boundary_target_velocity": tgt,
            "boundary_velocity": self._velocity_at(pts),
        }
        scalars = {
            "grid_spacing": float(self.grid_spacing),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
            "contact_length_scale": float(self.grid_spacing),
        }
        attributes = {
            "mass_quantity": np.array(self.mass, dtype=float),
            "particle_radius": np.array(self.particle_radius, dtype=float),
            "material_label": np.array(self.material_label, dtype=np.int32),
        }
        frame = StateFrame(
            float(self._t),
            np.array(self.position, dtype=float),
            np.array(self.velocity, dtype=float),
            mass=np.array(self.mass, dtype=float),
            body_id=np.array(self.body_id, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        vel = self._cell_velocity()
        speed = float(np.sqrt((vel ** 2).sum(axis=-1)).max()) if vel.size else 0.0
        return {
            "scene_id": self.scene_id,
            "seed": int(self.seed),
            "substeps": int(self._substeps),
            "final_time": float(self._t),
            "max_speed": speed,
            "max_scalar": float(self.scalar.max()) if self.scalar.size else 0.0,
            "grid_shape": list(self.n),
            "grid_spacing": float(self.grid_spacing),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
            "cube_omega_rad_s": float(self.omega),
        }
