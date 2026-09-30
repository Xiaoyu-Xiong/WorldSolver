"""FLIP (fluid-implicit-particle) liquid solver for the arch-obstacle dam break.

Physics
-------
Incompressible inviscid free-surface flow (Euler equations with gravity) solved
with a staggered MAC-grid pressure projection and particle advection (FLIP/PIC
blend, Zhu & Bridson style):

    1. P2G  : particle velocities -> staggered face velocities (trilinear)
    2. force: gravity applied to the vertical faces
    3. solid: no-penetration on solid faces (free slip tangentially)
    4. solve: A q = -div(u) with CG (Neumann at solids, p = 0 in air)
    5. proj : u <- u - (dt/rho) grad p  (discrete, divergence free)
    6. G2P  : FLIP delta + PIC blend, RK2 particle advection, solid push-out

Geometry is taken from the frozen ``scene_geometry.yaml``: ground/tank walls,
two arch piers, the arch lintel and the removable (lifting) dam gate.
"""
from __future__ import annotations

from itertools import product

import numpy as np

try:  # optional, used only for the signed level set
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None

_CORNERS = tuple(product((0, 1), repeat=3))


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        geometry = cfg.get("scene_geometry") or {}
        objects = {o["id"]: o for o in geometry.get("objects", [])}

        def box_of(name, default_c, default_h):
            obj = objects.get(name)
            if obj is None:
                return np.asarray(default_c, float), np.asarray(default_h, float)
            c = np.asarray(obj.get("transform", {}).get("translation", [0, 0, 0]), float)
            h = np.asarray(obj["geometry"]["half_extents"], float)
            return c, h

        gc, gh = box_of("ground", (0.0, 0.0, -0.05), (3.2, 1.4, 0.05))
        self.tank_lo = np.array([gc[0] - gh[0], gc[1] - gh[1], gc[2] + gh[2]])
        self.tank_hi = np.array([gc[0] + gh[0], gc[1] + gh[1], gc[2] + gh[2] + 2.4])

        # static rigid obstacles (arch)
        self.solid_boxes = []
        for name, dc, dh in (
            ("arch_left", (-0.75, 0.0, 0.7), (0.22, 0.7, 0.7)),
            ("arch_right", (0.75, 0.0, 0.7), (0.22, 0.7, 0.7)),
            ("arch_top", (0.0, 0.0, 1.4), (0.95, 0.7, 0.18)),
        ):
            c, h = box_of(name, dc, dh)
            self.solid_boxes.append((c, h))

        # removable barrier
        self.gate_center, self.gate_half = box_of("dam_gate", (-1.55, 0.0, 0.55), (0.04, 0.65, 0.55))
        self.gate_release = 0.20
        self.gate_speed = 4.0
        self.gate_top = float(self.gate_center[2] + self.gate_half[2])
        self.gate_bottom0 = float(self.gate_center[2] - self.gate_half[2])

        # ------------------------------------------------------------ MAC grid
        self.dx = 0.1
        dx = self.dx
        self.nx = int(round((self.tank_hi[0] - self.tank_lo[0]) / dx))
        self.ny = int(round((self.tank_hi[1] - self.tank_lo[1]) / dx))
        self.nz = int(round((self.tank_hi[2] - self.tank_lo[2]) / dx))
        nx, ny, nz = self.nx, self.ny, self.nz
        self.origin = self.tank_lo.copy()

        self.c_origin = self.origin + 0.5 * dx
        self.u_origin = self.origin + np.array([0.0, 0.5 * dx, 0.5 * dx])
        self.v_origin = self.origin + np.array([0.5 * dx, 0.0, 0.5 * dx])
        self.w_origin = self.origin + np.array([0.5 * dx, 0.5 * dx, 0.0])

        ix = self.c_origin[0] + dx * np.arange(nx)
        iy = self.c_origin[1] + dx * np.arange(ny)
        iz = self.c_origin[2] + dx * np.arange(nz)
        self.cx, self.cy, self.cz = np.meshgrid(ix, iy, iz, indexing="ij")

        self.static_solid = np.zeros((nx, ny, nz), bool)
        for c, h in self.solid_boxes:
            self.static_solid |= (
                (np.abs(self.cx - c[0]) <= h[0])
                & (np.abs(self.cy - c[1]) <= h[1])
                & (np.abs(self.cz - c[2]) <= h[2])
            )
        gc_, gh_ = self.gate_center, self.gate_half
        self.gate_cells = (np.abs(self.cx - gc_[0]) <= gh_[0]) & (np.abs(self.cy - gc_[1]) <= gh_[1])

        self.u = np.zeros((nx + 1, ny, nz))
        self.v = np.zeros((nx, ny + 1, nz))
        self.w = np.zeros((nx, ny, nz + 1))

        # ------------------------------------------------------------ particles
        sp = 0.5 * self.spacing
        self.p_spacing = sp
        clouds = []
        for b in sampling["blocks"]:
            lo = np.asarray(b[:3], float)
            hi = np.asarray(b[3:], float)
            axes = [np.arange(lo[i] + 0.5 * sp, hi[i], sp) for i in range(3)]
            if any(len(a) == 0 for a in axes):
                clouds.append(((lo + hi) * 0.5)[None, :])
                continue
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            clouds.append(np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())))
        self.pos = np.concatenate(clouds).astype(float)
        self.labels = np.concatenate(
            [np.full(len(c), i, np.int32) for i, c in enumerate(clouds)]
        )
        self.vel = np.zeros_like(self.pos)
        self.p_mass = np.full(len(self.pos), self.density * sp ** 3)
        self.p_radius = np.full(len(self.pos), 0.55 * self.spacing)

        span = float(np.ptp(self.pos, axis=0).max())
        self.characteristic_length = max(span, self.spacing)
        height = float(self.pos[:, 2].max() - self.tank_lo[2])
        self.reference_velocity = float(np.sqrt(2.0 * abs(self.gravity[2]) * max(height, 1e-3)))

        self._build_boundary_samples()

        self.t = 0.0
        self.dt_max = 0.008
        self.cfl = 0.9
        self.flip_ratio = 0.95
        self.steps = 0
        self._cg_iters = 0
        self._max_div = 0.0

    # ----------------------------------------------------------- boundary set
    def _build_boundary_samples(self) -> None:
        """Face centres of static solid/fluid interfaces with outward normals."""
        dx = self.dx
        nx, ny, nz = self.nx, self.ny, self.nz
        S = self.static_solid
        pts = []
        nrm = []

        def add(mask, offset, normal):
            idx = np.argwhere(mask)
            if len(idx) == 0:
                return
            p = self.c_origin + idx * dx + np.asarray(offset) * dx
            pts.append(p)
            nrm.append(np.repeat(np.asarray(normal, float)[None], len(p), axis=0))

        # obstacle surfaces (solid cell face touching a non-solid neighbour)
        m = np.zeros_like(S)
        m[:-1] = S[:-1] & ~S[1:]
        add(m, (0.5, 0, 0), (1, 0, 0))
        m = np.zeros_like(S)
        m[1:] = S[1:] & ~S[:-1]
        add(m, (-0.5, 0, 0), (-1, 0, 0))
        m = np.zeros_like(S)
        m[:, :-1] = S[:, :-1] & ~S[:, 1:]
        add(m, (0, 0.5, 0), (0, 1, 0))
        m = np.zeros_like(S)
        m[:, 1:] = S[:, 1:] & ~S[:, :-1]
        add(m, (0, -0.5, 0), (0, -1, 0))
        m = np.zeros_like(S)
        m[:, :, :-1] = S[:, :, :-1] & ~S[:, :, 1:]
        add(m, (0, 0, 0.5), (0, 0, 1))
        m = np.zeros_like(S)
        m[:, :, 1:] = S[:, :, 1:] & ~S[:, :, :-1]
        add(m, (0, 0, -0.5), (0, 0, -1))

        # tank walls and floor
        m = np.zeros_like(S)
        m[0] = ~S[0]
        add(m, (-0.5, 0, 0), (1, 0, 0))
        m = np.zeros_like(S)
        m[-1] = ~S[-1]
        add(m, (0.5, 0, 0), (-1, 0, 0))
        m = np.zeros_like(S)
        m[:, 0] = ~S[:, 0]
        add(m, (0, -0.5, 0), (0, 1, 0))
        m = np.zeros_like(S)
        m[:, -1] = ~S[:, -1]
        add(m, (0, 0.5, 0), (0, -1, 0))
        m = np.zeros_like(S)
        m[:, :, 0] = ~S[:, :, 0]
        add(m, (0, 0, -0.5), (0, 0, 1))

        points = np.concatenate(pts) if pts else np.zeros((1, 3))
        normals = np.concatenate(nrm) if nrm else np.array([[0.0, 0.0, 1.0]])
        limit = 1500
        if len(points) > limit:
            stride = int(np.ceil(len(points) / limit))
            points = points[::stride]
            normals = normals[::stride]
        self.b_points = np.ascontiguousarray(points)
        self.b_normals = np.ascontiguousarray(normals)
        self.b_probe = self.b_points + 0.5 * dx * self.b_normals
        self.b_target = np.zeros_like(self.b_points)

    # ---------------------------------------------------------- interpolation
    def _corner_data(self, p, origin, shape):
        g = (p - origin) / self.dx
        hi = np.asarray(shape, int) - 2
        i0 = np.floor(g).astype(np.int64)
        i0 = np.clip(i0, 0, hi)
        f = np.clip(g - i0, 0.0, 1.0)
        return i0, f

    def _gather(self, field, origin, p):
        i0, f = self._corner_data(p, origin, field.shape)
        out = np.zeros(len(p))
        for c in _CORNERS:
            w = np.ones(len(p))
            for a in range(3):
                w = w * (f[:, a] if c[a] else (1.0 - f[:, a]))
            out += w * field[i0[:, 0] + c[0], i0[:, 1] + c[1], i0[:, 2] + c[2]]
        return out

    def _scatter(self, p, vals, origin, shape):
        num = np.zeros(shape)
        den = np.zeros(shape)
        i0, f = self._corner_data(p, origin, shape)
        for c in _CORNERS:
            w = np.ones(len(p))
            for a in range(3):
                w = w * (f[:, a] if c[a] else (1.0 - f[:, a]))
            idx = (i0[:, 0] + c[0], i0[:, 1] + c[1], i0[:, 2] + c[2])
            np.add.at(num, idx, w * vals)
            np.add.at(den, idx, w)
        return num, den

    @staticmethod
    def _extrapolate(field, valid, iters=4):
        f = field.copy()
        ok = valid.copy()
        for _ in range(iters):
            s = np.zeros_like(f)
            c = np.zeros_like(f)
            fv = np.where(ok, f, 0.0)
            okf = ok.astype(float)
            s[1:] += fv[:-1]
            c[1:] += okf[:-1]
            s[:-1] += fv[1:]
            c[:-1] += okf[1:]
            s[:, 1:] += fv[:, :-1]
            c[:, 1:] += okf[:, :-1]
            s[:, :-1] += fv[:, 1:]
            c[:, :-1] += okf[:, 1:]
            s[:, :, 1:] += fv[:, :, :-1]
            c[:, :, 1:] += okf[:, :, :-1]
            s[:, :, :-1] += fv[:, :, 1:]
            c[:, :, :-1] += okf[:, :, 1:]
            new = (~ok) & (c > 0)
            if not new.any():
                break
            f[new] = s[new] / c[new]
            ok |= new
        return f

    # --------------------------------------------------------------- geometry
    def _gate_bottom(self, t):
        lift = max(0.0, (t - self.gate_release) * self.gate_speed)
        return self.gate_bottom0 + lift

    def _solid_mask(self, t):
        bottom = self._gate_bottom(t)
        S = self.static_solid.copy()
        if bottom < self.gate_top:
            S |= self.gate_cells & (self.cz >= bottom) & (self.cz <= self.gate_top)
        return S

    def _face_solids(self, S):
        nx, ny, nz = self.nx, self.ny, self.nz
        us = np.zeros((nx + 1, ny, nz), bool)
        us[1:] |= S
        us[:-1] |= S
        us[0] = True
        us[-1] = True
        vs = np.zeros((nx, ny + 1, nz), bool)
        vs[:, 1:] |= S
        vs[:, :-1] |= S
        vs[:, 0] = True
        vs[:, -1] = True
        ws = np.zeros((nx, ny, nz + 1), bool)
        ws[:, :, 1:] |= S
        ws[:, :, :-1] |= S
        ws[:, :, 0] = True
        ws[:, :, -1] = True
        return us, vs, ws

    def _push_out(self, t):
        """Resolve particle penetration of rigid boxes and tank walls."""
        pos, vel = self.pos, self.vel
        margin = 0.25 * self.p_spacing
        boxes = list(self.solid_boxes)
        bottom = self._gate_bottom(t)
        if bottom < self.gate_top:
            zc = 0.5 * (bottom + self.gate_top)
            zh = 0.5 * (self.gate_top - bottom)
            boxes.append(
                (
                    np.array([self.gate_center[0], self.gate_center[1], zc]),
                    np.array([self.gate_half[0], self.gate_half[1], max(zh, 1e-6)]),
                )
            )
        for c, h in boxes:
            d = np.abs(pos - c) - h
            inside = np.all(d < 0.0, axis=1)
            if not inside.any():
                continue
            idx = np.flatnonzero(inside)
            dd = d[idx]
            axis = np.argmax(dd, axis=1)
            sign = np.sign(pos[idx, axis] - c[axis])
            sign[sign == 0.0] = 1.0
            pos[idx, axis] = c[axis] + sign * (h[axis] + margin)
            vn = vel[idx, axis]
            vel[idx, axis] = np.where(vn * sign < 0.0, 0.0, vn)
        lo = self.tank_lo + margin
        hi = self.tank_hi - margin
        for a in range(3):
            low = pos[:, a] < lo[a]
            high = pos[:, a] > hi[a]
            pos[low, a] = lo[a]
            vel[low, a] = np.maximum(vel[low, a], 0.0)
            pos[high, a] = hi[a]
            vel[high, a] = np.minimum(vel[high, a], 0.0)

    # -------------------------------------------------------------- time step
    def _step(self, dt):
        dx = self.dx
        nx, ny, nz = self.nx, self.ny, self.nz
        S = self._solid_mask(self.t)
        us, vs, ws = self._face_solids(S)

        # --- P2G
        un, ud = self._scatter(self.pos, self.vel[:, 0], self.u_origin, self.u.shape)
        vn, vd = self._scatter(self.pos, self.vel[:, 1], self.v_origin, self.v.shape)
        wn, wd = self._scatter(self.pos, self.vel[:, 2], self.w_origin, self.w.shape)
        uvalid, vvalid, wvalid = ud > 1e-12, vd > 1e-12, wd > 1e-12
        u = np.where(uvalid, un / np.maximum(ud, 1e-12), 0.0)
        v = np.where(vvalid, vn / np.maximum(vd, 1e-12), 0.0)
        w = np.where(wvalid, wn / np.maximum(wd, 1e-12), 0.0)
        u_saved, v_saved, w_saved = u.copy(), v.copy(), w.copy()

        # --- external forces
        u += self.gravity[0] * dt
        v += self.gravity[1] * dt
        w += self.gravity[2] * dt

        # --- solid boundary condition (no penetration, free slip)
        u[us] = 0.0
        v[vs] = 0.0
        w[ws] = 0.0

        # --- fluid cells from particles
        gi = np.floor((self.pos - self.c_origin) / dx + 0.5).astype(np.int64)
        gi[:, 0] = np.clip(gi[:, 0], 0, nx - 1)
        gi[:, 1] = np.clip(gi[:, 1], 0, ny - 1)
        gi[:, 2] = np.clip(gi[:, 2], 0, nz - 1)
        F = np.zeros((nx, ny, nz), bool)
        F[gi[:, 0], gi[:, 1], gi[:, 2]] = True
        F &= ~S
        self.fluid = F

        # --- divergence
        div = (
            (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])
        ) / dx
        rhs = np.where(F, -div, 0.0)

        # --- diagonal: number of non-solid faces per cell
        kdiag = (
            (~us[1:]).astype(np.float64)
            + (~us[:-1])
            + (~vs[:, 1:])
            + (~vs[:, :-1])
            + (~ws[:, :, 1:])
            + (~ws[:, :, :-1])
        )
        kdiag = np.where(kdiag > 0, kdiag, 1.0)

        q = self._solve_pressure(rhs, F, kdiag)

        # --- projection
        u[1:-1] -= dx * (q[1:] - q[:-1])
        v[:, 1:-1] -= dx * (q[:, 1:] - q[:, :-1])
        w[:, :, 1:-1] -= dx * (q[:, :, 1:] - q[:, :, :-1])
        u[us] = 0.0
        v[vs] = 0.0
        w[ws] = 0.0

        # --- extrapolate into air so surface particles read valid velocities
        u = self._extrapolate(u, uvalid & ~us)
        v = self._extrapolate(v, vvalid & ~vs)
        w = self._extrapolate(w, wvalid & ~ws)
        u[us] = 0.0
        v[vs] = 0.0
        w[ws] = 0.0

        # --- G2P (FLIP + PIC blend)
        pic = np.column_stack(
            (
                self._gather(u, self.u_origin, self.pos),
                self._gather(v, self.v_origin, self.pos),
                self._gather(w, self.w_origin, self.pos),
            )
        )
        delta = np.column_stack(
            (
                self._gather(u - u_saved, self.u_origin, self.pos),
                self._gather(v - v_saved, self.v_origin, self.pos),
                self._gather(w - w_saved, self.w_origin, self.pos),
            )
        )
        r = self.flip_ratio
        self.vel = r * (self.vel + delta) + (1.0 - r) * pic
        np.clip(self.vel, -20.0, 20.0, out=self.vel)

        # --- advection (RK2 on the projected grid velocity)
        def grid_vel(p):
            return np.column_stack(
                (
                    self._gather(u, self.u_origin, p),
                    self._gather(v, self.v_origin, p),
                    self._gather(w, self.w_origin, p),
                )
            )

        k1 = grid_vel(self.pos)
        mid = self.pos + 0.5 * dt * k1
        np.clip(mid, self.tank_lo + 1e-4, self.tank_hi - 1e-4, out=mid)
        k2 = grid_vel(mid)
        self.pos = self.pos + dt * k2

        self.u, self.v, self.w = u, v, w
        self.t += dt
        self._push_out(self.t)
        self.steps += 1
        self._max_div = max(self._max_div, float(np.abs(div[F]).max()) if F.any() else 0.0)

    def _solve_pressure(self, rhs, F, kdiag, tol=1e-5, max_iter=120):
        def apply_A(q):
            r = kdiag * q
            r[1:] -= q[:-1]
            r[:-1] -= q[1:]
            r[:, 1:] -= q[:, :-1]
            r[:, :-1] -= q[:, 1:]
            r[:, :, 1:] -= q[:, :, :-1]
            r[:, :, :-1] -= q[:, :, 1:]
            r *= F
            return r

        q = np.zeros_like(rhs)
        res = rhs.copy()
        rnorm0 = float(np.sqrt(np.sum(res * res)))
        if rnorm0 < 1e-14:
            return q
        precond = np.where(F, 1.0 / kdiag, 0.0)
        z = res * precond
        p = z.copy()
        rz = float(np.sum(res * z))
        for it in range(max_iter):
            Ap = apply_A(p)
            pAp = float(np.sum(p * Ap))
            if abs(pAp) < 1e-30:
                break
            alpha = rz / pAp
            q += alpha * p
            res -= alpha * Ap
            rnorm = float(np.sqrt(np.sum(res * res)))
            self._cg_iters = it + 1
            if rnorm <= tol * rnorm0:
                break
            z = res * precond
            rz_new = float(np.sum(res * z))
            beta = rz_new / max(rz, 1e-30)
            p = z + beta * p
            rz = rz_new
        return np.where(F, q, 0.0)

    # ------------------------------------------------------------- diagnostics
    def _collocated_velocity(self):
        u, v, w = self.u, self.v, self.w
        vel = np.empty((self.nx, self.ny, self.nz, 3))
        vel[..., 0] = 0.5 * (u[1:] + u[:-1])
        vel[..., 1] = 0.5 * (v[:, 1:] + v[:, :-1])
        vel[..., 2] = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
        return vel

    def _cell_mask_from_particles(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        gi = np.floor((self.pos - self.c_origin) / self.dx + 0.5).astype(np.int64)
        gi[:, 0] = np.clip(gi[:, 0], 0, nx - 1)
        gi[:, 1] = np.clip(gi[:, 1], 0, ny - 1)
        gi[:, 2] = np.clip(gi[:, 2], 0, nz - 1)
        mask = np.zeros((nx, ny, nz), bool)
        mask[gi[:, 0], gi[:, 1], gi[:, 2]] = True
        return mask

    def _level_set(self, liquid):
        dx = self.dx
        far = 4.0 * max(self.nx, self.ny, self.nz) * dx
        if not liquid.any():
            return np.full(liquid.shape, far, np.float32)
        if liquid.all():
            return np.full(liquid.shape, -far, np.float32)
        if _ndimage is not None:
            out = _ndimage.distance_transform_edt(~liquid, sampling=dx)
            ins = _ndimage.distance_transform_edt(liquid, sampling=dx)
            phi = np.asarray(out, float) - np.asarray(ins, float)
        else:  # pragma: no cover
            phi = np.where(liquid, -0.5 * dx, 0.5 * dx).astype(float)
        phi = np.nan_to_num(phi, nan=far, posinf=far, neginf=-far)
        return np.clip(phi, -far, far).astype(np.float32)

    def _active_mask(self, liquid, S):
        m = liquid & ~S
        interior = m.copy()
        interior[1:] &= m[:-1]
        interior[:-1] &= m[1:]
        interior[:, 1:] &= m[:, :-1]
        interior[:, :-1] &= m[:, 1:]
        interior[:, :, 1:] &= m[:, :, :-1]
        interior[:, :, :-1] &= m[:, :, 1:]
        return interior

    # ------------------------------------------------------------------ public
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 200000:
            vmax = float(np.abs(self.vel).max()) if len(self.vel) else 0.0
            vmax = max(vmax, 1e-3)
            dt = min(self.dt_max, self.cfl * self.dx / vmax)
            dt = max(dt, 1e-4)
            dt = min(dt, target - self.t)
            self._step(dt)
            guard += 1
        self.t = target

        S = self._solid_mask(self.t)
        liquid = self._cell_mask_from_particles() & ~S
        vel_grid = self._collocated_velocity()
        active = self._active_mask(liquid, S)
        phi = self._level_set(liquid)

        b_vel = np.column_stack(
            (
                self._gather(vel_grid[..., 0], self.c_origin, self.b_probe),
                self._gather(vel_grid[..., 1], self.c_origin, self.b_probe),
                self._gather(vel_grid[..., 2], self.c_origin, self.b_probe),
            )
        )
        gate_moving = (
            self.t >= self.gate_release and self._gate_bottom(self.t) < self.gate_top
        )
        b_target = self.b_target.copy()

        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        frame = StateFrame(
            time=target,
            position=pos,
            velocity=vel,
            mass=self.p_mass.copy(),
            body_id=self.labels.astype(np.int32),
            attributes={
                "mass_quantity": self.p_mass.copy(),
                "particle_radius": self.p_radius.copy(),
                "material_label": self.labels.astype(np.int32),
            },
            fields={
                "velocity_field": vel_grid.astype(np.float32),
                "active_mask": active,
                "liquid_level_set": phi,
                "boundary_normal": self.b_normals.astype(np.float32),
                "boundary_velocity": b_vel.astype(np.float32),
                "boundary_target_velocity": b_target.astype(np.float32),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "grid_spacing": float(self.dx),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
            },
        )
        frame.validate()
        self._gate_moving = bool(gate_moving)
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "FLIP/PIC MAC-grid incompressible liquid solver",
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.dx),
            "particles": int(len(self.pos)),
            "steps": int(self.steps),
            "last_cg_iterations": int(self._cg_iters),
            "max_predivergence": float(self._max_div),
            "gate_release_time": float(self.gate_release),
        }
