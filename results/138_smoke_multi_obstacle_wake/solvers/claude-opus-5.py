"""Self-contained incompressible smoke solver (no Genesis, NumPy/SciPy only).

Physical model, following Fedkiw/Stam "Visual Simulation of Smoke":

  * Boussinesq incompressible flow on a staggered (MAC) grid,
  * semi-Lagrangian momentum transport with RK2 backtrace,
  * MacCormack-corrected, locally clamped transport of smoke concentration,
  * buoyancy proportional to the transported smoke concentration,
  * vorticity confinement so obstacle wakes stay rolled up on a coarse grid,
  * exact pressure projection: masked Neumann Poisson problem on fluid cells,
    solved with a Jacobi-preconditioned conjugate gradient, enforcing
    no-penetration on the enclosing walls and on the rigid obstacle boxes,
  * no-slip velocity zeroing inside solids and on all enclosing walls.

The scheme is deterministic: fixed grids, fixed tolerances / iteration caps and
purely state-dependent (never random) CFL substepping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter

from worldsolver.contracts import StateFrame

_CENTER_OFF = (0.5, 0.5, 0.5)

_GEOM_FALLBACK = {
    "objects": [
        {"id": "obstacle_low", "semantic": "boundary", "collision_enabled": True,
         "geometry": {"type": "box", "half_extents": [0.38, 0.38, 0.38]},
         "transform": {"translation": [-0.65, 0.0, 1.25]}},
        {"id": "obstacle_high", "semantic": "boundary", "collision_enabled": True,
         "geometry": {"type": "box", "half_extents": [0.42, 0.42, 0.42]},
         "transform": {"translation": [0.65, 0.0, 2.15]}},
        {"id": "smoke_source", "semantic": "emitter", "collision_enabled": False,
         "geometry": {"type": "cylinder", "radius": 0.32, "height": 0.1},
         "transform": {"translation": [0.0, 0.0, 0.05]}},
        {"id": "subject", "semantic": "gas", "collision_enabled": True,
         "geometry": {"type": "voxel_grid", "bounds_min": [-3.0, -1.2, 0.0],
                      "bounds_max": [3.0, 1.2, 3.8], "grid_shape": [24, 16, 28]},
         "transform": {"translation": [0.0, 0.0, 0.0]}},
    ]
}


def _load_geometry() -> dict:
    """Read the frozen scene geometry; fall back to the frozen literal values."""
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:3]:
        candidate = parent / "scene_geometry.yaml"
        if candidate.is_file():
            try:
                import yaml

                data = yaml.safe_load(candidate.read_text())
            except Exception:
                break
            if isinstance(data, dict) and data.get("objects"):
                return data
            break
    return _GEOM_FALLBACK

def _boxes_from_geometry(geom: dict, lo: np.ndarray, hi: np.ndarray) -> list[tuple]:
    """Axis-aligned solid boxes (min, max) that actually overlap the fluid domain."""
    boxes = []
    for obj in geom.get("objects", []):
        if obj.get("semantic") not in ("boundary", "rigid"):
            continue
        if not obj.get("collision_enabled", True):
            continue
        g = obj.get("geometry", {})
        if g.get("type") != "box" or not g.get("half_extents"):
            continue
        h = np.asarray(g["half_extents"], float)
        c = np.asarray(obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0]), float)
        bmin, bmax = c - h, c + h
        ov_lo = np.maximum(bmin, lo)
        ov_hi = np.minimum(bmax, hi)
        if np.all(ov_hi - ov_lo > 1e-6):
            boxes.append((bmin, bmax))
    return boxes


def _inside_boxes(points: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Boolean mask of points inside any solid box."""
    inside = np.zeros(len(points), dtype=bool)
    for bmin, bmax in boxes:
        inside |= np.all((points >= bmin[None, :]) & (points <= bmax[None, :]), axis=1)
    return inside


def _pcg(mat, diag_inv, rhs, x0, tol=1e-6, max_iter=220):
    """Jacobi-preconditioned conjugate gradient for the symmetric Poisson system."""
    x = x0.copy()
    r = rhs - mat.dot(x)
    nrm0 = float(np.linalg.norm(rhs))
    if nrm0 <= 1e-14:
        return np.zeros_like(x), 0
    if float(np.linalg.norm(r)) <= tol * nrm0:
        return x, 0
    z = diag_inv * r
    p = z.copy()
    rz = float(r.dot(z))
    it = 0
    for it in range(1, max_iter + 1):
        ap = mat.dot(p)
        pap = float(p.dot(ap))
        if not np.isfinite(pap) or abs(pap) < 1e-30:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) <= tol * nrm0:
            break
        z = diag_inv * r
        rz_new = float(r.dot(z))
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            break
        p = z + (rz_new / rz) * p
        rz = rz_new
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x), it
    return x, it

class BenchmarkSolver:
    """MAC-grid smoke solver for the multi-obstacle wake scene."""

    REFINE = 2                # fine sim cells per exported coarse cell, per axis
    CFL = 0.9
    MAX_SUBSTEPS = 8          # bounds worst-case substeps per sample interval
    BUOY_FRAC = 0.40          # buoyant acceleration = BUOY_FRAC * |g| * smoke
    CONFINE = 2.0             # vorticity confinement strength
    CONFINE_CAP = 25.0        # m/s^2
    EMIT_RATE = 4.0           # 1/s relaxation toward EMIT_TARGET inside the emitter
    EMIT_TARGET = 1.2
    EMIT_SPEED = 0.65         # m/s prescribed vertical inflow at the emitter
    EMIT_RELAX = 9.0          # 1/s velocity relaxation toward EMIT_SPEED
    VEL_CAP = 20.0

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        frame0 = scene.initial_state()
        self.frame0 = frame0

        geom = _load_geometry()
        subject = None
        for obj in geom.get("objects", []):
            if obj.get("id") == "subject":
                subject = obj
        sg = (subject or {}).get("geometry", {})
        self.lo = np.asarray(sg.get("bounds_min", [-3.0, -1.2, 0.0]), float)
        self.hi = np.asarray(sg.get("bounds_max", [3.0, 1.2, 3.8]), float)
        coarse = sg.get("grid_shape") or cfg.get("sampling", {}).get("field_grid_shape")
        self.coarse_shape = tuple(int(v) for v in (coarse or [24, 16, 28]))

        r = int(self.REFINE)
        self.nx, self.ny, self.nz = (int(v) * r for v in self.coarse_shape)
        self.shape = (self.nx, self.ny, self.nz)
        self.d = (self.hi - self.lo) / np.array(self.shape, float)
        self.dx, self.dy, self.dz = (float(v) for v in self.d)

        # fine cell centers
        self.xc = self.lo[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.lo[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.zc = self.lo[2] + (np.arange(self.nz) + 0.5) * self.dz
        gx, gy, gz = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self.cell_pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

        grav = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.g_mag = float(abs(grav[2])) if abs(grav[2]) > 1e-9 else 9.81

        self.boxes = _boxes_from_geometry(geom, self.lo, self.hi)
        self.solid = _inside_boxes(self.cell_pts, self.boxes).reshape(self.shape)
        self._setup_emitter(geom)
        self._setup_initial_scalar(frame0)
        self._setup_faces()
        self._build_poisson()
        self._setup_boundary_samples()
        self._setup_export_grid()
        self._setup_particles(frame0)

        # instantaneous emitter source rate, valid for the t=0 export as well
        self.src_rate = np.zeros(self.shape)
        self.src_rate[self.source] = self.EMIT_RATE * np.clip(
            self.EMIT_TARGET - self.smoke[self.source], 0.0, None)

        self.time = 0.0
        self.steps = 0
        self.cg_iters = 0
        self.max_speed = 0.0
        self.max_div = 0.0

    def _setup_emitter(self, geom: dict) -> None:
        """Cylindrical buoyant smoke source at the domain floor."""
        radius, height, center = 0.32, 0.1, np.array([0.0, 0.0, 0.05])
        for obj in geom.get("objects", []):
            if obj.get("semantic") == "emitter":
                g = obj.get("geometry", {})
                radius = float(g.get("radius", radius))
                height = float(g.get("height", height))
                center = np.asarray(
                    obj.get("transform", {}).get("translation", center), float)
        p = self.cell_pts
        rad = np.hypot(p[:, 0] - center[0], p[:, 1] - center[1])
        z_lo = center[2] - 0.5 * height - 0.5 * self.dz
        z_hi = center[2] + 0.5 * height + 1.0 * self.dz
        src = (rad <= radius) & (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi)
        self.source = src.reshape(self.shape) & (~self.solid)
        if not self.source.any():  # degenerate guard: seed the lowest layer
            m = (rad <= max(radius, self.dx)).reshape(self.shape)
            self.source = np.zeros(self.shape, dtype=bool)
            self.source[:, :, 0] = m[:, :, 0]
        self.emit_center = center
        self.emit_radius = radius

    def _sample_grid_to_points(self, field: np.ndarray, lo, hi, pts) -> np.ndarray:
        """Trilinear sample of a node-centered field defined on [lo,hi] at pts."""
        shp = np.asarray(field.shape, float)
        span = np.asarray(hi, float) - np.asarray(lo, float)
        idx = (pts - np.asarray(lo, float)[None, :]) / span[None, :] * (shp - 1.0)[None, :]
        return map_coordinates(field.astype(float), idx.T, order=1, mode="nearest")

    def _setup_initial_scalar(self, frame0) -> None:
        """Resample the frozen initial smoke field onto the fine simulation grid."""
        s0 = np.asarray(frame0.fields.get("transported_scalar"), float)
        if s0.ndim != 3:
            s0 = np.zeros(self.coarse_shape, float)
        vals = self._sample_grid_to_points(s0, self.lo, self.hi, self.cell_pts)
        smoke = np.clip(vals.reshape(self.shape), 0.0, None)
        smoke[self.solid] = 0.0
        self.smoke = smoke
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.p_prev = None

    def _setup_faces(self) -> None:
        """Blocked-face masks: solid-adjacent faces and all enclosing walls."""
        sd = self.solid
        ub = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        ub[:-1] |= sd
        ub[1:] |= sd
        ub[0] = True
        ub[-1] = True
        vb = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        vb[:, :-1] |= sd
        vb[:, 1:] |= sd
        vb[:, 0] = True
        vb[:, -1] = True
        wb = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        wb[:, :, :-1] |= sd
        wb[:, :, 1:] |= sd
        wb[:, :, 0] = True
        wb[:, :, -1] = True
        self.u_blocked, self.v_blocked, self.w_blocked = ub, vb, wb

    def _build_poisson(self) -> None:
        """Assemble the masked Neumann pressure Laplacian on fluid cells."""
        fluid = ~self.solid
        # drop fluid cells that have no open face (fully enclosed)
        open_any = np.zeros(self.shape, dtype=bool)
        open_any |= ~self.u_blocked[:-1] | ~self.u_blocked[1:]
        open_any |= ~self.v_blocked[:, :-1] | ~self.v_blocked[:, 1:]
        open_any |= ~self.w_blocked[:, :, :-1] | ~self.w_blocked[:, :, 1:]
        fluid &= open_any
        self.fluid = fluid
        self.solid = ~fluid
        self._setup_faces()

        idx = np.full(self.shape, -1, dtype=np.int64)
        idx[fluid] = np.arange(int(fluid.sum()), dtype=np.int64)
        self.cell_index = idx
        self.n_unknown = int(fluid.sum())

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        diag = np.zeros(self.n_unknown)
        specs = (
            (self.u_blocked[1:-1, :, :], (slice(0, -1), slice(None), slice(None)),
             (slice(1, None), slice(None), slice(None)), 1.0 / self.dx ** 2),
            (self.v_blocked[:, 1:-1, :], (slice(None), slice(0, -1), slice(None)),
             (slice(None), slice(1, None), slice(None)), 1.0 / self.dy ** 2),
            (self.w_blocked[:, :, 1:-1], (slice(None), slice(None), slice(0, -1)),
             (slice(None), slice(None), slice(1, None)), 1.0 / self.dz ** 2),
        )
        for blocked, sa, sb, coef in specs:
            ia, ib = idx[sa], idx[sb]
            good = (~blocked) & (ia >= 0) & (ib >= 0)
            a, b = ia[good], ib[good]
            # store A = -laplacian so the system is positive semi-definite
            rows.append(a)
            cols.append(b)
            vals.append(np.full(a.size, -coef))
            rows.append(b)
            cols.append(a)
            vals.append(np.full(a.size, -coef))
            np.add.at(diag, a, coef)
            np.add.at(diag, b, coef)
        rows.append(np.arange(self.n_unknown, dtype=np.int64))
        cols.append(np.arange(self.n_unknown, dtype=np.int64))
        vals.append(diag)
        mat = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_unknown, self.n_unknown),
        ).tocsr()
        self.poisson = mat
        d = mat.diagonal().copy()
        d[np.abs(d) < 1e-12] = 1.0
        self.diag_inv = 1.0 / d

    # ------------------------------------------------------- interpolation
    def _sample(self, field: np.ndarray, pts: np.ndarray, off) -> np.ndarray:
        """Trilinear sample of a MAC/cell array at physical points.

        ``off`` gives the physical offset (in cell fractions) of index 0 of the
        array relative to the domain minimum: 0.5 for a cell-centered axis and
        0.0 for a staggered face axis.
        """
        shp = pts.shape[:-1]
        q = pts.reshape(-1, 3)
        cx = (q[:, 0] - self.lo[0]) / self.dx - off[0]
        cy = (q[:, 1] - self.lo[1]) / self.dy - off[1]
        cz = (q[:, 2] - self.lo[2]) / self.dz - off[2]
        out = map_coordinates(field, np.stack((cx, cy, cz)), order=1, mode="nearest")
        return out.reshape(shp)

    def _velocity_at(self, pts: np.ndarray) -> tuple:
        us = self._sample(self.u, pts, (0.0, 0.5, 0.5))
        vs = self._sample(self.v, pts, (0.5, 0.0, 0.5))
        ws = self._sample(self.w, pts, (0.5, 0.5, 0.0))
        return us, vs, ws

    def _clamp_pts(self, pts: np.ndarray) -> np.ndarray:
        eps = 1e-6
        return np.clip(pts, self.lo + eps, self.hi - eps)

    def _trace(self, pts: np.ndarray, dt: float) -> np.ndarray:
        """RK2 (midpoint) characteristic trace by -dt (dt<0 traces forward)."""
        u0, v0, w0 = self._velocity_at(pts)
        mid = self._clamp_pts(pts - 0.5 * dt * np.stack((u0, v0, w0), axis=-1))
        u1, v1, w1 = self._velocity_at(mid)
        return self._clamp_pts(pts - dt * np.stack((u1, v1, w1), axis=-1))

    def _face_points(self) -> None:
        """Precompute physical coordinates of MAC faces and cell centers."""
        xf = self.lo[0] + np.arange(self.nx + 1) * self.dx
        yf = self.lo[1] + np.arange(self.ny + 1) * self.dy
        zf = self.lo[2] + np.arange(self.nz + 1) * self.dz

        def grid(ax, ay, az):
            a, b, c = np.meshgrid(ax, ay, az, indexing="ij")
            return np.stack((a, b, c), axis=-1)

        self.pu = grid(xf, self.yc, self.zc)
        self.pv = grid(self.xc, yf, self.zc)
        self.pw = grid(self.xc, self.yc, zf)
        self.pc = grid(self.xc, self.yc, self.zc)

    # ---------------------------------------------------------- advection
    def _advect(self, dt: float) -> None:
        bu = self._trace(self.pu, dt)
        bv = self._trace(self.pv, dt)
        bw = self._trace(self.pw, dt)
        u_new = self._sample(self.u, bu, (0.0, 0.5, 0.5))
        v_new = self._sample(self.v, bv, (0.5, 0.0, 0.5))
        w_new = self._sample(self.w, bw, (0.5, 0.5, 0.0))

        # MacCormack-corrected, locally clamped smoke transport
        bc = self._trace(self.pc, dt)
        s_back = self._sample(self.smoke, bc, _CENTER_OFF)
        fwd = self._trace(bc, -dt)
        s_fwd = self._sample(s_back, fwd, _CENTER_OFF)
        s_corr = s_back + 0.5 * (self.smoke - s_fwd)
        s_min = self._sample(minimum_filter(self.smoke, size=3, mode="nearest"),
                             bc, _CENTER_OFF)
        s_max = self._sample(maximum_filter(self.smoke, size=3, mode="nearest"),
                             bc, _CENTER_OFF)
        smoke_new = np.where((s_corr >= s_min) & (s_corr <= s_max), s_corr, s_back)

        self.u, self.v, self.w = u_new, v_new, w_new
        self.smoke = np.clip(smoke_new, 0.0, 4.0)
        self._apply_solid_bc()

    # ------------------------------------------------------------- forces
    def _forces(self, dt: float) -> None:
        self.smoke[self.solid] = 0.0
        # emitter: bounded relaxation source (this is the exported source rate)
        self.src_rate = np.zeros(self.shape)
        self.src_rate[self.source] = self.EMIT_RATE * np.clip(
            self.EMIT_TARGET - self.smoke[self.source], 0.0, None)
        self.smoke += dt * self.src_rate

        # Boussinesq buoyancy on vertical faces
        buoy = self.BUOY_FRAC * self.g_mag
        s_face = np.zeros_like(self.w)
        s_face[:, :, 1:-1] = 0.5 * (self.smoke[:, :, :-1] + self.smoke[:, :, 1:])
        s_face[:, :, 0] = self.smoke[:, :, 0]
        s_face[:, :, -1] = self.smoke[:, :, -1]
        self.w += dt * buoy * s_face

        # prescribed vertical inflow at the emitter mouth
        relax = 1.0 - np.exp(-self.EMIT_RELAX * dt)
        src_w = np.zeros_like(self.w, dtype=bool)
        src_w[:, :, :-1] |= self.source
        src_w[:, :, 1:] |= self.source
        self.w[src_w] += relax * (self.EMIT_SPEED - self.w[src_w])

        self._confine(dt)
        self._apply_solid_bc()

    def _collocated(self) -> tuple:
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def _confine(self, dt: float) -> None:
        """Fedkiw-style vorticity confinement to sustain obstacle roll-up."""
        if self.CONFINE <= 0.0:
            return
        uc, vc, wc = self._collocated()
        dudy, dudz = np.gradient(uc, self.dy, axis=1), np.gradient(uc, self.dz, axis=2)
        dvdx, dvdz = np.gradient(vc, self.dx, axis=0), np.gradient(vc, self.dz, axis=2)
        dwdx, dwdy = np.gradient(wc, self.dx, axis=0), np.gradient(wc, self.dy, axis=1)
        ox, oy, oz = dwdy - dvdz, dudz - dwdx, dvdx - dudy
        mag = np.sqrt(ox * ox + oy * oy + oz * oz)
        gx = np.gradient(mag, self.dx, axis=0)
        gy = np.gradient(mag, self.dy, axis=1)
        gz = np.gradient(mag, self.dz, axis=2)
        gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        gx, gy, gz = gx / gn, gy / gn, gz / gn
        h = (self.dx * self.dy * self.dz) ** (1.0 / 3.0)
        scale = self.CONFINE * h
        fx = scale * (gy * oz - gz * oy)
        fy = scale * (gz * ox - gx * oz)
        fz = scale * (gx * oy - gy * ox)
        cap = self.CONFINE_CAP
        fx = np.clip(fx, -cap, cap)
        fy = np.clip(fy, -cap, cap)
        fz = np.clip(fz, -cap, cap)
        fx[self.solid] = 0.0
        fy[self.solid] = 0.0
        fz[self.solid] = 0.0
        self.u[1:-1] += dt * 0.5 * (fx[:-1] + fx[1:])
        self.v[:, 1:-1] += dt * 0.5 * (fy[:, :-1] + fy[:, 1:])
        self.w[:, :, 1:-1] += dt * 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])

    def _apply_solid_bc(self) -> None:
        """No-penetration on walls and obstacle faces; no smoke inside solids."""
        self.u[self.u_blocked] = 0.0
        self.v[self.v_blocked] = 0.0
        self.w[self.w_blocked] = 0.0
        self.smoke[self.solid] = 0.0

    # -------------------------------------------------------- projection
    def _project(self, dt: float) -> None:
        """Exact pressure projection: solve grad^2 p = div(u)/dt on fluid cells."""
        if self.n_unknown == 0:
            return
        div = ((self.u[1:] - self.u[:-1]) / self.dx
               + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
               + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz)
        # A = -laplacian, so the right-hand side carries the matching sign
        rhs = -div[self.fluid] / dt
        rhs -= rhs.mean()  # compatibility of the pure-Neumann system
        x0 = self.p_prev if (self.p_prev is not None
                             and self.p_prev.shape == rhs.shape) else np.zeros_like(rhs)
        sol, iters = _pcg(self.poisson, self.diag_inv, rhs, x0,
                          tol=1e-6, max_iter=220)
        self.cg_iters += int(iters)
        sol -= sol.mean()
        self.p_prev = sol
        p = np.zeros(self.shape)
        p[self.fluid] = sol

        gu = np.zeros_like(self.u)
        gu[1:-1] = (p[1:] - p[:-1]) / self.dx
        gv = np.zeros_like(self.v)
        gv[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / self.dy
        gw = np.zeros_like(self.w)
        gw[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / self.dz
        self.u -= dt * gu
        self.v -= dt * gv
        self.w -= dt * gw
        self._apply_solid_bc()

        div2 = ((self.u[1:] - self.u[:-1]) / self.dx
                + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
                + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz)
        self.max_div = max(self.max_div, float(np.abs(div2[self.fluid]).max())
                           if self.n_unknown else 0.0)

    def _sanitize(self) -> None:
        for name in ("u", "v", "w"):
            arr = np.nan_to_num(getattr(self, name), nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(arr, -self.VEL_CAP, self.VEL_CAP, out=arr)
            setattr(self, name, arr)
        self.smoke = np.clip(np.nan_to_num(self.smoke, nan=0.0, posinf=0.0,
                                           neginf=0.0), 0.0, 4.0)
        self.smoke[self.solid] = 0.0

    def _step(self, dt: float) -> None:
        self._advect(dt)
        self._forces(dt)
        self._project(dt)
        self._sanitize()
        self._advect_particles(dt)
        self.steps += 1

    # ---------------------------------------------------------- particles
    def _setup_particles(self, frame0) -> None:
        """Passive tracers seeded from the frozen sample cloud."""
        pos = np.asarray(frame0.position, float).copy()
        self.part_pos = self._clamp_pts(pos)
        inside = _inside_boxes(self.part_pos, self.boxes)
        if inside.any():  # nudge any seeded-in-solid tracer downward to free gas
            self.part_pos[inside, 2] = np.clip(
                self.part_pos[inside, 2] - 0.5, self.lo[2] + self.dz, self.hi[2] - self.dz)
        self.part_vel = np.zeros_like(self.part_pos)

    def _advect_particles(self, dt: float) -> None:
        if len(self.part_pos) == 0:
            return
        prev = self.part_pos
        u0, v0, w0 = self._velocity_at(prev)
        mid = self._clamp_pts(prev + 0.5 * dt * np.stack((u0, v0, w0), axis=-1))
        u1, v1, w1 = self._velocity_at(mid)
        vel = np.stack((u1, v1, w1), axis=-1)
        nxt = self._clamp_pts(prev + dt * vel)
        blocked = _inside_boxes(nxt, self.boxes)
        nxt[blocked] = prev[blocked]
        self.part_pos = np.nan_to_num(nxt, nan=0.0, posinf=0.0, neginf=0.0)
        vel[blocked] = 0.0
        self.part_vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------- boundary samples
    def _setup_boundary_samples(self) -> None:
        """Wall samples on the enclosing box and on each rigid obstacle face."""
        pts: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        def face(bmin, bmax, axis, side, counts, outward):
            """Sample one axis-aligned face; ``side`` is -1 (min) or +1 (max)."""
            a1, a2 = [k for k in range(3) if k != axis]
            n1, n2 = counts
            t1 = np.linspace(bmin[a1], bmax[a1], n1)
            t2 = np.linspace(bmin[a2], bmax[a2], n2)
            g1, g2 = np.meshgrid(t1, t2, indexing="ij")
            q = np.zeros((g1.size, 3))
            q[:, a1] = g1.ravel()
            q[:, a2] = g2.ravel()
            q[:, axis] = bmin[axis] if side < 0 else bmax[axis]
            nv = np.zeros((g1.size, 3))
            nv[:, axis] = (-1.0 if side < 0 else 1.0) * outward
            pts.append(q)
            nrm.append(nv)

        lo, hi = self.lo, self.hi
        # enclosing walls: inward normals (outward = -1 relative to the box faces)
        face(lo, hi, 0, -1, (7, 9), -1.0)
        face(lo, hi, 0, +1, (7, 9), -1.0)
        face(lo, hi, 1, -1, (11, 7), -1.0)
        face(lo, hi, 1, +1, (11, 7), -1.0)
        face(lo, hi, 2, -1, (11, 7), -1.0)
        face(lo, hi, 2, +1, (11, 7), -1.0)
        # obstacle surfaces: outward normals point into the fluid
        for bmin, bmax in self.boxes:
            for axis in range(3):
                for side in (-1, +1):
                    face(bmin, bmax, axis, side, (5, 5), 1.0)

        self.bnd_pts = np.concatenate(pts) if pts else np.zeros((1, 3))
        self.bnd_nrm = np.concatenate(nrm) if nrm else np.zeros((1, 3))
        self.bnd_target = np.zeros_like(self.bnd_pts)  # stationary impermeable walls

    # ------------------------------------------------------- export grids
    def _setup_export_grid(self) -> None:
        """Node lattice of the frozen diagnostic grid, matching scene.py."""
        self._face_points()
        nx, ny, nz = self.coarse_shape
        ex = np.linspace(self.lo[0], self.hi[0], nx)
        ey = np.linspace(self.lo[1], self.hi[1], ny)
        ez = np.linspace(self.lo[2], self.hi[2], nz)
        gx, gy, gz = np.meshgrid(ex, ey, ez, indexing="ij")
        self.export_pts = np.stack((gx, gy, gz), axis=-1)
        flat = self.export_pts.reshape(-1, 3)
        active = ~_inside_boxes(flat, self.boxes)
        self.export_mask = active.reshape(self.coarse_shape)

        sc = dict(self.frame0.scalars)
        self.out_scalars = {
            "grid_spacing": float(sc.get("grid_spacing", 0.18)),
            "characteristic_length": float(sc.get("characteristic_length", 1.0)),
            "reference_velocity": float(sc.get("reference_velocity", 1.0)),
            "contact_length_scale": float(sc.get("contact_length_scale", 0.18)),
        }

    def _boundary_velocity(self) -> np.ndarray:
        """No-slip-consistent fluid velocity at wall samples.

        The collocated velocity is padded with one layer of zero ghost values
        (walls and solid interiors are at rest), so trilinear sampling exactly
        on a wall returns the no-slip ghost average rather than the interior
        value.
        """
        uc, vc, wc = self._collocated()
        out = np.zeros((len(self.bnd_pts), 3))
        for k, comp in enumerate((uc, vc, wc)):
            field = np.where(self.solid, 0.0, comp)
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            q = self.bnd_pts
            cx = (q[:, 0] - self.lo[0]) / self.dx + 0.5
            cy = (q[:, 1] - self.lo[1]) / self.dy + 0.5
            cz = (q[:, 2] - self.lo[2]) / self.dz + 0.5
            out[:, k] = map_coordinates(padded, np.stack((cx, cy, cz)),
                                        order=1, mode="nearest")
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------- export
    def _export(self, time: float) -> StateFrame:
        pts = self.export_pts
        scalar = self._sample(self.smoke, pts, _CENTER_OFF)
        src = self._sample(getattr(self, "src_rate", np.zeros(self.shape)),
                           pts, _CENTER_OFF)
        uc, vc, wc = self._collocated()
        uc = np.where(self.solid, 0.0, uc)
        vc = np.where(self.solid, 0.0, vc)
        wc = np.where(self.solid, 0.0, wc)
        vel = np.stack((self._sample(uc, pts, _CENTER_OFF),
                        self._sample(vc, pts, _CENTER_OFF),
                        self._sample(wc, pts, _CENTER_OFF)), axis=-1)
        mask = self.export_mask
        scalar = np.where(mask, np.clip(np.nan_to_num(scalar), 0.0, 4.0), 0.0)
        src = np.where(mask, np.nan_to_num(src), 0.0)
        vel = np.where(mask[..., None], np.nan_to_num(vel), 0.0)
        vel = np.clip(vel, -self.VEL_CAP, self.VEL_CAP)

        bnd_vel = self._boundary_velocity()
        f0 = self.frame0
        n = len(self.part_pos)
        attrs = {}
        for key, value in f0.attributes.items():
            arr = np.asarray(value)
            attrs[key] = arr.copy() if len(arr) == n else arr
        fields = {
            "transported_scalar": scalar.astype(float),
            "scalar_source": src.astype(float),
            "active_mask": mask.astype(bool),
            "velocity_field": vel.astype(float),
            "boundary_velocity": bnd_vel.astype(float),
            "boundary_target_velocity": self.bnd_target.astype(float),
            "boundary_normal": self.bnd_nrm.astype(float),
            "boundary_position": self.bnd_pts.astype(float),
            "cumulative_mass_source": np.zeros(n),
        }
        mass = np.asarray(f0.mass, float).copy() if f0.mass is not None else None
        body = np.asarray(f0.body_id).copy() if f0.body_id is not None else None
        frame = StateFrame(
            float(time),
            self.part_pos.astype(float).copy(),
            self.part_vel.astype(float).copy(),
            mass=mass,
            body_id=body,
            attributes=attrs,
            fields=fields,
            scalars=dict(self.out_scalars),
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------- public API
    def _cfl_dt(self, remaining: float) -> float:
        speed = max(
            float(np.abs(self.u).max()) / self.dx,
            float(np.abs(self.v).max()) / self.dy,
            float(np.abs(self.w).max()) / self.dz,
            1e-9,
        )
        dt = self.CFL / speed
        buoy = max(self.BUOY_FRAC * self.g_mag * float(self.smoke.max()), 1e-9)
        dt = min(dt, np.sqrt(2.0 * self.CFL * self.dz / buoy))
        dt = min(dt, remaining)
        return max(dt, remaining / float(self.MAX_SUBSTEPS))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while target - self.time > 1e-9 and guard < 10000:
            dt = self._cfl_dt(target - self.time)
            dt = float(min(dt, target - self.time))
            if dt <= 1e-9:
                break
            self._step(dt)
            self.time += dt
            guard += 1
            self.max_speed = max(
                self.max_speed,
                float(np.abs(self.u).max()),
                float(np.abs(self.v).max()),
                float(np.abs(self.w).max()),
            )
        if target > self.time:
            self.time = target
        return self._export(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC-grid Boussinesq smoke, semi-Lagrangian + PCG projection",
            "sim_grid": list(self.shape),
            "export_grid": list(self.coarse_shape),
            "steps": int(self.steps),
            "cg_iterations": int(self.cg_iters),
            "fluid_cells": int(self.n_unknown),
            "max_speed": float(self.max_speed),
            "max_divergence_after_projection": float(self.max_div),
            "obstacle_boxes": int(len(self.boxes)),
            "seed": int(getattr(self, "seed", 0)),
        }
