"""Self-implemented FLIP (fluid-implicit-particle) liquid solver.

Physical model
--------------
Incompressible, inviscid liquid under uniform gravity, discretised with a
staggered (MAC) velocity grid and Lagrangian marker particles:

    Du/Dt = -grad(p)/rho + g ,      div(u) = 0

Time stepping per substep:
  1. particle -> grid momentum transfer (mass weighted trilinear splat),
  2. explicit gravity on grid faces,
  3. solid (basin / ground / domain) no-through-flow face constraints,
  4. free-surface pressure projection: a sparse SPD Poisson system on liquid
     cells with p = 0 Dirichlet in air cells and Neumann walls, solved by
     Jacobi-preconditioned conjugate gradients,
  5. grid -> particle velocity update using a FLIP/PIC blend,
  6. particle advection with the projected divergence-free grid velocity and
     geometric push-out from the rigid basin walls and ground.

Both the falling bunny-shaped liquid volume (mesh-vertex particles, label 1)
and the receiving pool (block-sampled particles, label 0) are advanced by the
same single-phase solver, so the bunny keeps its silhouette during free fall,
splashes on impact, and merges into the pool afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from worldsolver.contracts import StateFrame

RHO = 1000.0
_FLIP_RATIO = 0.92
_MAX_SPEED = 30.0

_FALLBACK_BOXES = (
    ((0.0, 0.92, 0.3), (1.85, 0.08, 0.34)),
    ((0.0, -0.92, 0.3), (1.85, 0.08, 0.34)),
    ((-1.85, 0.0, 0.3), (0.08, 0.92, 0.34)),
    ((1.85, 0.0, 0.3), (0.08, 0.92, 0.34)),
)


def _tri_weights(g: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear flat indices/weights for continuous indices ``g`` in ``shape``."""
    n = np.asarray(shape, dtype=np.int64)
    base = np.floor(g).astype(np.int64)
    base = np.clip(base, 0, n - 2)
    frac = np.clip(g - base, 0.0, 1.0)
    w0 = 1.0 - frac
    count = g.shape[0]
    idx = np.empty((count, 8), dtype=np.int64)
    wts = np.empty((count, 8), dtype=np.float64)
    col = 0
    for a in (0, 1):
        wa = frac[:, 0] if a else w0[:, 0]
        ia = base[:, 0] + a
        for b in (0, 1):
            wb = frac[:, 1] if b else w0[:, 1]
            ib = base[:, 1] + b
            for c in (0, 1):
                wc = frac[:, 2] if c else w0[:, 2]
                ic = base[:, 2] + c
                idx[:, col] = (ia * n[1] + ib) * n[2] + ic
                wts[:, col] = wa * wb * wc
                col += 1
    return idx, wts


def _splat(idx: np.ndarray, wts: np.ndarray, values: np.ndarray,
           weight: np.ndarray, shape) -> np.ndarray:
    """Mass weighted particle -> grid transfer."""
    size = int(np.prod(shape))
    ww = wts * weight[:, None]
    num = np.bincount(idx.ravel(), weights=(ww * values[:, None]).ravel(), minlength=size)
    den = np.bincount(idx.ravel(), weights=ww.ravel(), minlength=size)
    out = np.zeros(size)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    return out.reshape(shape), good.reshape(shape)


def _gather(field: np.ndarray, idx: np.ndarray, wts: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", field.reshape(-1)[idx], wts)


def _extrapolate(field: np.ndarray, valid: np.ndarray, iters: int = 2) -> np.ndarray:
    """Fill invalid faces with the average of valid neighbours (a few sweeps)."""
    out = field.copy()
    known = valid.copy()
    for _ in range(iters):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        masked = np.where(known, out, 0.0)
        kf = known.astype(np.float64)
        for axis in range(3):
            for shift in (1, -1):
                acc += np.roll(masked, shift, axis=axis)
                cnt += np.roll(kf, shift, axis=axis)
        fill = (~known) & (cnt > 0.0)
        if not fill.any():
            break
        out[fill] = acc[fill] / cnt[fill]
        known = known | fill
    return out


class BenchmarkSolver:
    """FLIP liquid solver honouring the frozen basin geometry and sampling."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        rng = np.random.default_rng(int(seed))
        self._rng = rng

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        sampling = config.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.14))

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.pmass = np.array(frame0.mass, dtype=float)
        attrs = dict(frame0.attributes)
        self.labels = np.asarray(attrs.get("material_label",
                                            np.zeros(len(self.pos))), dtype=np.int32)
        self.radius = np.asarray(attrs.get(
            "particle_radius", np.full(len(self.pos), 0.45 * self.spacing)), dtype=float)
        self.count = int(len(self.pos))

        # --- grids -------------------------------------------------------
        self.h = self.spacing
        h = self.h
        diag_shape = tuple(int(v) for v in sampling.get("field_grid_shape", (24, 16, 28)))
        self.diag_shape = diag_shape
        diag_origin = np.array([-0.5 * diag_shape[0] * h, -0.5 * diag_shape[1] * h, 0.0])
        pad = np.array([4, 1, 0], dtype=int)          # extra cells around the basin walls
        self.pad = pad
        self.origin = diag_origin - pad * h
        self.shape = (diag_shape[0] + 2 * int(pad[0]),
                      diag_shape[1] + 2 * int(pad[1]),
                      diag_shape[2] + int(pad[2]))
        nx, ny, nz = self.shape
        self.ushape = (nx + 1, ny, nz)
        self.vshape = (nx, ny + 1, nz)
        self.wshape = (nx, ny, nz + 1)
        self.lo = self.origin.copy()
        self.hi = self.origin + np.array(self.shape, dtype=float) * h

        self._build_solids(config)
        self._build_boundary_samples()

        self.u = np.zeros(self.ushape)
        self.v = np.zeros(self.vshape)
        self.w = np.zeros(self.wshape)
        self.fluid = self._mark_fluid(self.pos)

        # frozen diagnostic scalars
        extent = float(np.ptp(self.pos, axis=0).max())
        self.characteristic_length = max(extent, h)
        bunny = self.labels == 1
        pool = ~bunny
        if bunny.any() and pool.any():
            drop = float(self.pos[bunny, 2].min() - self.pos[pool, 2].max())
        else:
            drop = 1.0
        drop = max(drop, 0.1)
        self.reference_velocity = float(np.sqrt(2.0 * abs(self.gravity[2]) * drop))

        self.time = 0.0
        self._steps = 0
        self._cg_iters = 0
        self._max_speed_seen = 0.0

    def _build_solids(self, config: dict) -> None:
        boxes = []
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            if obj.get("semantic") != "boundary":
                continue
            geo = obj.get("geometry", {}) or {}
            if geo.get("type") != "box" or not geo.get("half_extents"):
                continue
            centre = np.asarray((obj.get("transform", {}) or {}).get(
                "translation", [0.0, 0.0, 0.0]), dtype=float)
            half = np.asarray(geo["half_extents"], dtype=float)
            boxes.append((centre, half))
            if str(obj.get("id", "")) == "basin_back":
                # Mirror the rear pane to close the (glass) camera-side face of the
                # basin so the receiving pool is actually contained by the vessel.
                boxes.append((centre * np.array([1.0, -1.0, 1.0]), half))
        if not boxes:
            boxes = [(np.asarray(c, float), np.asarray(e, float)) for c, e in _FALLBACK_BOXES]
        self.boxes = boxes

        nx, ny, nz = self.shape
        h = self.h
        xc = self.origin[0] + h * (np.arange(nx) + 0.5)
        yc = self.origin[1] + h * (np.arange(ny) + 0.5)
        zc = self.origin[2] + h * (np.arange(nz) + 0.5)
        gx, gy, gz = np.meshgrid(xc, yc, zc, indexing="ij")
        solid = np.zeros(self.shape, dtype=bool)
        for centre, half in boxes:
            solid |= ((np.abs(gx - centre[0]) <= half[0])
                      & (np.abs(gy - centre[1]) <= half[1])
                      & (np.abs(gz - centre[2]) <= half[2]))
        self.solid = solid

        su = np.zeros(self.ushape, dtype=bool)
        su[1:-1] = solid[:-1] | solid[1:]
        su[0] = True
        su[-1] = True
        sv = np.zeros(self.vshape, dtype=bool)
        sv[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        sv[:, 0] = True
        sv[:, -1] = True
        sw = np.zeros(self.wshape, dtype=bool)
        sw[:, :, 1:-1] = solid[:, :, :-1] | solid[:, :, 1:]
        sw[:, :, 0] = True
        sw[:, :, -1] = True
        self.su, self.sv, self.sw = su, sv, sw

    def _build_boundary_samples(self) -> None:
        """Wall sample points (slightly inside the fluid side), inward normals."""
        pts = []
        nrm = []
        zs = np.linspace(0.06, 0.58, 5)
        xs = np.linspace(-1.68, 1.68, 18)
        ys = np.linspace(-0.77, 0.77, 9)
        # ground: top face of the ground box is exactly z = 0
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        flat = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
        pts.append(flat)
        nrm.append(np.tile(np.array([0.0, 0.0, 1.0]), (len(flat), 1)))
        # left / right basin panes: inner faces at |x| = 1.85 - 0.08 = 1.77
        for sign in (-1.0, 1.0):
            ay, az = np.meshgrid(ys, zs, indexing="ij")
            wall = np.column_stack((np.full(ay.size, sign * 1.77),
                                    ay.ravel(), az.ravel()))
            pts.append(wall)
            nrm.append(np.tile(np.array([-sign, 0.0, 0.0]), (len(wall), 1)))
        # front / back panes: inner faces at |y| = 0.92 - 0.08 = 0.84
        for sign in (-1.0, 1.0):
            ax, az = np.meshgrid(xs, zs, indexing="ij")
            wall = np.column_stack((ax.ravel(), np.full(ax.size, sign * 0.84),
                                    az.ravel()))
            pts.append(wall)
            nrm.append(np.tile(np.array([0.0, -sign, 0.0]), (len(wall), 1)))
        self.boundary_points = np.concatenate(pts, axis=0)
        self.boundary_normals = np.concatenate(nrm, axis=0)
        self.boundary_target = np.zeros_like(self.boundary_points)

    # ------------------------------------------------------- grid transfers
    def _mac_indices(self, pos: np.ndarray):
        g = (pos - self.origin) / self.h
        return (
            _tri_weights(g - np.array([0.0, 0.5, 0.5]), self.ushape),
            _tri_weights(g - np.array([0.5, 0.0, 0.5]), self.vshape),
            _tri_weights(g - np.array([0.5, 0.5, 0.0]), self.wshape),
        )

    def _sample_velocity(self, points: np.ndarray) -> np.ndarray:
        (iu, wu), (iv, wv), (iw, ww) = self._mac_indices(np.asarray(points, dtype=float))
        out = np.empty((len(points), 3))
        out[:, 0] = _gather(self.u, iu, wu)
        out[:, 1] = _gather(self.v, iv, wv)
        out[:, 2] = _gather(self.w, iw, ww)
        return out

    def _cell_flat(self, pos: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        ci = np.floor((pos - self.origin) / self.h).astype(np.int64)
        ci[:, 0] = np.clip(ci[:, 0], 0, nx - 1)
        ci[:, 1] = np.clip(ci[:, 1], 0, ny - 1)
        ci[:, 2] = np.clip(ci[:, 2], 0, nz - 1)
        return (ci[:, 0] * ny + ci[:, 1]) * nz + ci[:, 2]

    def _mark_fluid(self, pos: np.ndarray) -> np.ndarray:
        occupancy = np.bincount(self._cell_flat(pos),
                                minlength=int(np.prod(self.shape))) > 0
        fluid = ndimage.binary_fill_holes(occupancy.reshape(self.shape))
        return np.asarray(fluid) & (~self.solid)

    # ------------------------------------------------------------ projection
    def _project(self, dt: float) -> None:
        nx, ny, nz = self.shape
        h = self.h
        u, v, w = self.u, self.v, self.w
        u[self.su] = 0.0
        v[self.sv] = 0.0
        w[self.sw] = 0.0
        fluid = self.fluid
        n_fluid = int(fluid.sum())
        if n_fluid == 0:
            return
        div = ((u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1])
               + (w[:, :, 1:] - w[:, :, :-1])) / h

        index = np.full(self.shape, -1, dtype=np.int64)
        index[fluid] = np.arange(n_fluid, dtype=np.int64)

        open_faces = ((~self.su[:-1]).astype(np.float64) + (~self.su[1:])
                      + (~self.sv[:, :-1]) + (~self.sv[:, 1:])
                      + (~self.sw[:, :, :-1]) + (~self.sw[:, :, 1:]))
        diag = open_faces[fluid]

        rows = [index[fluid]]
        cols = [index[fluid]]
        vals = [diag.copy()]
        pairs = (
            (fluid[:-1, :, :] & fluid[1:, :, :] & (~self.su[1:-1, :, :]),
             index[:-1, :, :], index[1:, :, :]),
            (fluid[:, :-1, :] & fluid[:, 1:, :] & (~self.sv[:, 1:-1, :]),
             index[:, :-1, :], index[:, 1:, :]),
            (fluid[:, :, :-1] & fluid[:, :, 1:] & (~self.sw[:, :, 1:-1]),
             index[:, :, :-1], index[:, :, 1:]),
        )
        for mask, ia, ib in pairs:
            if not mask.any():
                continue
            a = ia[mask]
            b = ib[mask]
            rows.append(a)
            cols.append(b)
            vals.append(np.full(a.shape, -1.0))
            rows.append(b)
            cols.append(a)
            vals.append(np.full(a.shape, -1.0))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        scale = dt / (RHO * h * h)
        rhs = -div[fluid] / scale
        isolated = diag <= 0.0
        if isolated.any():
            vals[:n_fluid][isolated] = 1.0
            rhs[isolated] = 0.0
            diag = np.where(isolated, 1.0, diag)
        matrix = sparse.coo_matrix((vals, (rows, cols)),
                                   shape=(n_fluid, n_fluid)).tocsr()
        precond = sparse.diags(1.0 / diag)
        solution, info = cg(matrix, rhs, rtol=1e-6, atol=0.0, maxiter=300, M=precond)
        if not np.all(np.isfinite(solution)):
            solution = np.zeros(n_fluid)
        self._cg_iters += 1

        pressure = np.zeros(self.shape)
        pressure[fluid] = solution
        coef = dt / (RHO * h)

        mu = (~self.su[1:-1]) & (fluid[:-1] | fluid[1:])
        u[1:-1][mu] -= coef * (pressure[1:] - pressure[:-1])[mu]
        mv = (~self.sv[:, 1:-1]) & (fluid[:, :-1] | fluid[:, 1:])
        v[:, 1:-1][mv] -= coef * (pressure[:, 1:] - pressure[:, :-1])[mv]
        mw = (~self.sw[:, :, 1:-1]) & (fluid[:, :, :-1] | fluid[:, :, 1:])
        w[:, :, 1:-1][mw] -= coef * (pressure[:, :, 1:] - pressure[:, :, :-1])[mw]

        u[self.su] = 0.0
        v[self.sv] = 0.0
        w[self.sw] = 0.0

    # ------------------------------------------------------------ boundaries
    def _resolve_solids(self) -> None:
        pos, vel = self.pos, self.vel
        skin = 0.5 * self.h
        for centre, half in self.boxes:
            ext = half + skin
            delta = np.abs(pos - centre) - ext
            inside = np.all(delta < 0.0, axis=1)
            if not inside.any():
                continue
            sel = np.flatnonzero(inside)
            axis = np.argmax(delta[sel], axis=1)
            sign = np.where(pos[sel, axis] >= centre[axis], 1.0, -1.0)
            pos[sel, axis] = centre[axis] + sign * ext[axis]
            comp = vel[sel, axis]
            vel[sel, axis] = np.where(comp * sign < 0.0, 0.0, comp)
        margin = 0.02
        lo = self.lo + margin
        hi = self.hi - margin
        for axis in range(3):
            under = pos[:, axis] < lo[axis]
            over = pos[:, axis] > hi[axis]
            if under.any():
                pos[under, axis] = lo[axis]
                vel[under, axis] = np.maximum(vel[under, axis], 0.0)
            if over.any():
                pos[over, axis] = hi[axis]
                vel[over, axis] = np.minimum(vel[over, axis], 0.0)

    # ----------------------------------------------------------------- steps
    def _step(self, dt: float) -> None:
        (iu, wu), (iv, wv), (iw, ww) = self._mac_indices(self.pos)
        u, uv = _splat(iu, wu, self.vel[:, 0], self.pmass, self.ushape)
        v, vv = _splat(iv, wv, self.vel[:, 1], self.pmass, self.vshape)
        w, wvd = _splat(iw, ww, self.vel[:, 2], self.pmass, self.wshape)

        u_old, v_old, w_old = u.copy(), v.copy(), w.copy()
        u += self.gravity[0] * dt
        v += self.gravity[1] * dt
        w += self.gravity[2] * dt

        self.fluid = self._mark_fluid(self.pos)
        self.u, self.v, self.w = u, v, w
        self._project(dt)
        u, v, w = self.u, self.v, self.w

        near_u = ndimage.binary_dilation(uv)
        near_v = ndimage.binary_dilation(vv)
        near_w = ndimage.binary_dilation(wvd)
        self.u = _extrapolate(u, near_u)
        self.v = _extrapolate(v, near_v)
        self.w = _extrapolate(w, near_w)
        self.u[self.su] = 0.0
        self.v[self.sv] = 0.0
        self.w[self.sw] = 0.0

        pic = np.empty_like(self.vel)
        pic[:, 0] = _gather(self.u, iu, wu)
        pic[:, 1] = _gather(self.v, iv, wv)
        pic[:, 2] = _gather(self.w, iw, ww)
        delta = np.empty_like(self.vel)
        delta[:, 0] = _gather(self.u - u_old, iu, wu)
        delta[:, 1] = _gather(self.v - v_old, iv, wv)
        delta[:, 2] = _gather(self.w - w_old, iw, ww)

        flip = self.vel + delta
        self.vel = _FLIP_RATIO * flip + (1.0 - _FLIP_RATIO) * pic
        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.vel, -_MAX_SPEED, _MAX_SPEED, out=self.vel)

        # advect with the projected, divergence-free grid velocity (midpoint)
        mid = self.pos + 0.5 * dt * pic
        vmid = self._sample_velocity(np.clip(mid, self.lo + 1e-4, self.hi - 1e-4))
        self.pos = self.pos + dt * vmid
        self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        self._resolve_solids()
        self._steps += 1
        self._max_speed_seen = max(self._max_speed_seen,
                                   float(np.abs(self.vel).max(initial=0.0)))

    def advance_to(self, time: float):
        target = float(time)
        remaining = target - self.time
        if remaining > 1e-12:
            speed = float(np.abs(self.vel).max(initial=0.0))
            # CFL-limited substepping: keep particle motion below ~0.9 cells per
            # substep, with a floor for accuracy and a ceiling for runtime.
            n_cfl = int(np.ceil(remaining * max(speed, 1e-6) / (0.9 * self.h)))
            n_sub = int(min(max(4, n_cfl), 16))
            dt = remaining / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
        else:
            self.time = max(self.time, target)
        return self._frame(target)

    # ---------------------------------------------------------------- output
    def _frame(self, time: float) -> StateFrame:
        h = self.h
        px, py, pz = self.pad
        dnx, dny, dnz = self.diag_shape
        block = (slice(px, px + dnx), slice(py, py + dny), slice(pz, pz + dnz))

        # collocated (cell centre) diagnostic velocity on the frozen grid
        u, v, w = self.u, self.v, self.w
        cell_u = 0.5 * (u[:-1] + u[1:])
        cell_v = 0.5 * (v[:, :-1] + v[:, 1:])
        cell_w = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        velocity_field = np.stack((cell_u[block], cell_v[block], cell_w[block]), axis=-1)

        liquid = self.fluid[block]
        solid_block = self.solid[block]
        if liquid.any():
            # cells where the incompressible liquid equations are solved
            active = liquid & (~solid_block)
            dist_out = ndimage.distance_transform_edt(~liquid)
            dist_in = ndimage.distance_transform_edt(liquid)
            level_set = np.where(liquid, -(dist_in - 0.5) * h, (dist_out - 0.5) * h)
        else:
            active = np.zeros(self.diag_shape, dtype=bool)
            level_set = np.full(self.diag_shape, 10.0)
        level_set = np.clip(np.asarray(level_set, dtype=float), -50.0, 50.0)

        boundary_velocity = self._sample_velocity(self.boundary_points)

        mass_quantity = np.zeros((self.count, 2))
        pool = self.labels == 0
        mass_quantity[pool, 0] = self.pmass[pool]
        mass_quantity[~pool, 1] = self.pmass[~pool]

        fields = {
            "velocity_field": velocity_field,
            "active_mask": np.ascontiguousarray(active),
            "liquid_level_set": np.ascontiguousarray(level_set),
            "liquid_interior_mask": np.ascontiguousarray(liquid),
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_velocity": boundary_velocity,
            "boundary_target_velocity": self.boundary_target.copy(),
            "cumulative_mass_source": np.zeros(2),
        }
        attributes = {
            "mass_quantity": mass_quantity,
            "particle_radius": self.radius.copy(),
            "material_label": self.labels.copy(),
            "phase_id": self.labels.copy(),
        }
        scalars = {
            "grid_spacing": float(h),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
            "contact_length_scale": float(self.spacing),
            "liquid_phase_id": 0.0,
        }
        frame = StateFrame(
            float(time),
            np.ascontiguousarray(self.pos),
            np.ascontiguousarray(self.vel),
            mass=self.pmass.copy(),
            body_id=self.labels.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "FLIP (MAC grid + CG pressure projection)",
            "grid_shape": list(self.shape),
            "grid_spacing": float(self.h),
            "particles": int(self.count),
            "substeps": int(self._steps),
            "pressure_solves": int(self._cg_iters),
            "flip_ratio": _FLIP_RATIO,
            "max_speed": float(self._max_speed_seen),
            "sim_time": float(self.time),
            "note": "camera-side basin pane mirrored from basin_back to close the vessel",
        }
