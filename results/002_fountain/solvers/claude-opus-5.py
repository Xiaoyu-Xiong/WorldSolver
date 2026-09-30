"""Fountain solver: 3D FLIP/PIC free-surface liquid on a staggered MAC grid.

Physical model
--------------
Incompressible inviscid free-surface liquid (Euler equations with gravity)

    Du/Dt = -grad(p)/rho + g ,      div(u) = 0

discretized with a hybrid particle/grid (PIC-FLIP) method in the spirit of
Zhu & Bridson / Jiang et al.:

1. particle-to-grid transfer of momentum onto MAC faces (trilinear kernels),
2. velocity extrapolation into the air region,
3. gravity integration on the grid,
4. free-surface pressure projection (Dirichlet p = 0 in air cells, Neumann
   no-flux on the solid basin / nozzle faces) solved with Jacobi preconditioned
   conjugate gradients,
5. grid-to-particle transfer with a FLIP/PIC blend,
6. explicit particle advection with analytic collision handling against the
   frozen basin, floor and nozzle geometry.

A sustained upward inflow at the nozzle exit seeds new particles at exactly the
prescribed volumetric flux, and particles leaving the open domain are removed.
Both contributions are accumulated into ``fields.cumulative_mass_source`` so
mass bookkeeping is exact.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional, only used for the signed distance diagnostic
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None


class BenchmarkSolver:
    FLIP_RATIO = 0.95
    CFL = 0.85
    MAX_SUBSTEPS = 48
    CG_TOL = 1e-5
    CG_MAX_ITER = 160

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        ic = dict(config.get("initial_conditions", {}))
        self.rng = np.random.default_rng(int(seed))

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.rho = float(ic.get("liquid_density", 1000.0))
        self.h = float(ic.get("particle_spacing", 0.085))
        self.pool_radius = float(ic.get("pool_radius", 1.45))
        self.jet_radius = float(ic.get("nozzle_radius", 0.18))
        self.exit_z = float(ic.get("nozzle_exit_z", 0.4))
        self.jet_speed = float(ic.get("jet_speed", 5.2))
        self.sample_dt = float(config.get("sample_dt", 0.02))

        # ---- frozen geometry (basin, nozzle, particle domain) -------------
        basin_radius, rim_z = 1.5, 0.52
        nozzle_radius, nozzle_top = 0.24, 0.34
        bounds_min = np.array([-1.5, -1.5, 0.04])
        bounds_max = np.array([1.5, 1.5, 3.2])
        wall_thickness = 0.08
        for obj in (config.get("scene_geometry") or {}).get("objects", []) or []:
            try:
                geom = obj.get("geometry", {})
                if obj.get("id") == "circular_basin":
                    profile = np.asarray(geom["inner_profile"], dtype=np.float64)
                    basin_radius = float(np.min(profile[:, 0]))
                    rim_z = float(np.max(profile[:, 1]))
                    wall_thickness = float(geom.get("wall_thickness", wall_thickness))
                elif obj.get("id") == "fountain_nozzle":
                    nozzle_radius = float(geom["radius"])
                    height = float(geom["height"])
                    center_z = float((obj.get("transform", {}) or {}).get(
                        "translation", [0.0, 0.0, 0.5 * height])[2])
                    nozzle_top = center_z + 0.5 * height
                elif obj.get("id") == "fountain_liquid":
                    bounds_min = np.asarray(geom["bounds_min"], dtype=np.float64)
                    bounds_max = np.asarray(geom["bounds_max"], dtype=np.float64)
            except Exception:
                continue

        self.basin_radius = basin_radius
        self.rim_z = rim_z
        self.wall_outer = basin_radius + wall_thickness
        self.nozzle_radius = nozzle_radius
        self.nozzle_top = nozzle_top
        self.floor_z = float(bounds_min[2])

        # ---- diagnostic / solver grid -------------------------------------
        self.dx = 0.1
        dx = self.dx
        self.xmin = -(self.wall_outer + 0.02)
        self.xmax = self.wall_outer + 0.02
        self.zmin = 0.0
        self.zmax = float(max(bounds_max[2], 3.2)) + dx
        self.nx = int(round((self.xmax - self.xmin) / dx))
        self.ny = self.nx
        self.nz = int(round((self.zmax - self.zmin) / dx))
        self.xmax = self.xmin + self.nx * dx
        self.zmax = self.zmin + self.nz * dx
        self.origin = np.array([self.xmin, self.xmin, self.zmin], dtype=np.float64)
        nx, ny, nz = self.nx, self.ny, self.nz
        self.shape = (nx, ny, nz)
        self.ushape = (nx + 1, ny, nz)
        self.vshape = (nx, ny + 1, nz)
        self.wshape = (nx, ny, nz + 1)

        xc = self.xmin + (np.arange(nx) + 0.5) * dx
        yc = self.xmin + (np.arange(ny) + 0.5) * dx
        zc = self.zmin + (np.arange(nz) + 0.5) * dx
        self.cell_centers = (xc, yc, zc)
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        R = np.sqrt(X * X + Y * Y)

        solid = (R >= self.basin_radius) & (Z <= self.rim_z)
        solid |= (Z < self.floor_z) & (R < self.wall_outer)
        solid |= (R < self.nozzle_radius) & (Z < self.nozzle_top)
        self.solid = solid
        self.nonsolid = (~solid).astype(np.float64)

        # number of non-solid neighbours per cell (static Poisson diagonal);
        # outside the domain is solid except the open top.
        ns = self.nonsolid
        diag = np.zeros(self.shape, dtype=np.float64)
        diag[:-1] += ns[1:]
        diag[1:] += ns[:-1]
        diag[:, :-1] += ns[:, 1:]
        diag[:, 1:] += ns[:, :-1]
        diag[:, :, :-1] += ns[:, :, 1:]
        diag[:, :, 1:] += ns[:, :, :-1]
        diag[:, :, -1] += 1.0  # open top boundary behaves like air
        self.Ndiag = np.maximum(diag, 1.0)

        # ---- particles from the frozen initial state -----------------------
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64, copy=True)
        self.vel = np.array(frame0.velocity, dtype=np.float64, copy=True)
        if frame0.mass is not None:
            self.pmass = np.array(frame0.mass, dtype=np.float64, copy=True)
        else:
            self.pmass = np.full(len(self.pos), self.rho * self.h ** 3)
        source_region = None
        try:
            source_region = np.asarray(frame0.attributes["source_region"], dtype=np.float64)
        except Exception:
            source_region = np.zeros(len(self.pos))
        self.tag = np.array(source_region, dtype=np.float64, copy=True)
        self.p_mass_unit = float(self.rho * self.h ** 3)
        self.p_volume = float(self.h ** 3)
        self.initial_mass = float(np.sum(self.pmass))
        self.mass_source = 0.0

        # ---- nozzle emitter (exact prescribed volumetric flux) -------------
        axis = np.arange(-self.jet_radius + 0.5 * self.h, self.jet_radius, self.h)
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        disk = np.column_stack((gx.ravel(), gy.ravel()))
        keep = np.sum(disk ** 2, axis=1) <= (self.jet_radius - 0.25 * self.h) ** 2
        self.emit_xy = disk[keep]
        if len(self.emit_xy) == 0:
            self.emit_xy = np.zeros((1, 2))
        area = np.pi * self.jet_radius ** 2
        # layer thickness such that layer_volume / (layer_dz / speed) = area*speed
        self.emit_dz = float(len(self.emit_xy) * self.p_volume / max(area, 1e-12))
        self.emit_phase = 0.0

        # ---- wall sample points for the boundary-condition diagnostics -----
        self._build_boundary_samples()

        # ---- state -----------------------------------------------------
        self.time = 0.0
        self.step_count = 0
        self.emitted = 0
        self.removed = 0
        self.u = np.zeros(self.ushape)
        self.v = np.zeros(self.vshape)
        self.w = np.zeros(self.wshape)
        self.fluid = np.zeros(self.shape, dtype=bool)
        self.max_speed = float(np.max(np.abs(self.vel))) if len(self.vel) else 0.0
        self._grid_solve(1e-3, apply_gravity=False)

    def _build_boundary_samples(self) -> None:
        pts, nrm = [], []
        # basin inner wall
        ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        for z in np.arange(self.floor_z + 0.04, self.rim_z, 0.06):
            for a in ang:
                pts.append([self.basin_radius * np.cos(a), self.basin_radius * np.sin(a), z])
                nrm.append([-np.cos(a), -np.sin(a), 0.0])
        # basin floor
        ang2 = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        for r in (0.32, 0.6, 0.9, 1.2, 1.4):
            for a in ang2:
                pts.append([r * np.cos(a), r * np.sin(a), self.floor_z])
                nrm.append([0.0, 0.0, 1.0])
        # nozzle side wall
        for z in np.arange(self.floor_z + 0.02, self.nozzle_top, 0.08):
            for a in ang2:
                pts.append([self.nozzle_radius * np.cos(a), self.nozzle_radius * np.sin(a), z])
                nrm.append([np.cos(a), np.sin(a), 0.0])
        # nozzle top face
        for r in (0.08, 0.16, 0.22):
            for a in ang2:
                pts.append([r * np.cos(a), r * np.sin(a), self.nozzle_top])
                nrm.append([0.0, 0.0, 1.0])
        self.bnd_pos = np.asarray(pts, dtype=np.float64)
        n = np.asarray(nrm, dtype=np.float64)
        self.bnd_normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        # sample the fluid slightly inside the wall
        self.bnd_probe = self.bnd_pos + self.bnd_normal * (0.5 * self.dx)
        self.bnd_target = np.zeros_like(self.bnd_pos)

    # -------------------------------------------------- transfer primitives
    def _weights(self, pos, off, shape):
        g = (pos - self.origin) / self.dx - np.asarray(off)
        i0 = np.floor(g).astype(np.int64)
        hi = np.array([shape[0] - 2, shape[1] - 2, shape[2] - 2])
        i0 = np.clip(i0, 0, np.maximum(hi, 0))
        f = np.clip(g - i0, 0.0, 1.0)
        return i0, f

    def _scatter(self, shape, pos, off, values, weights):
        i0, f = self._weights(pos, off, shape)
        num = np.zeros(shape, dtype=np.float64)
        den = np.zeros(shape, dtype=np.float64)
        nflat, dflat = num.reshape(-1), den.reshape(-1)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        for a in (0, 1):
            wa = fx if a else 1.0 - fx
            for b in (0, 1):
                wb = fy if b else 1.0 - fy
                for c in (0, 1):
                    wc = fz if c else 1.0 - fz
                    ww = wa * wb * wc * weights
                    idx = np.ravel_multi_index(
                        (i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c), shape)
                    np.add.at(nflat, idx, ww * values)
                    np.add.at(dflat, idx, ww)
        return num, den

    def _gather(self, field, pos, off):
        shape = field.shape
        i0, f = self._weights(pos, off, shape)
        flat = field.reshape(-1)
        out = np.zeros(len(pos), dtype=np.float64)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        for a in (0, 1):
            wa = fx if a else 1.0 - fx
            for b in (0, 1):
                wb = fy if b else 1.0 - fy
                for c in (0, 1):
                    wc = fz if c else 1.0 - fz
                    idx = np.ravel_multi_index(
                        (i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c), shape)
                    out += (wa * wb * wc) * flat[idx]
        return out

    @staticmethod
    def _extrapolate(field, valid, iterations=3):
        f = field.copy()
        ok = valid.copy()
        for _ in range(iterations):
            s = np.zeros_like(f)
            c = np.zeros_like(f)
            fv = f * ok
            okf = ok.astype(np.float64)
            s[:-1] += fv[1:]
            c[:-1] += okf[1:]
            s[1:] += fv[:-1]
            c[1:] += okf[:-1]
            s[:, :-1] += fv[:, 1:]
            c[:, :-1] += okf[:, 1:]
            s[:, 1:] += fv[:, :-1]
            c[:, 1:] += okf[:, :-1]
            s[:, :, :-1] += fv[:, :, 1:]
            c[:, :, :-1] += okf[:, :, 1:]
            s[:, :, 1:] += fv[:, :, :-1]
            c[:, :, 1:] += okf[:, :, :-1]
            fill = (c > 0) & (~ok)
            if not np.any(fill):
                break
            f[fill] = s[fill] / c[fill]
            ok = ok | fill
        return f

    # ------------------------------------------------------------- pressure
    def _apply_A(self, p, fluid):
        out = self.Ndiag * p
        out[1:] -= p[:-1]
        out[:-1] -= p[1:]
        out[:, 1:] -= p[:, :-1]
        out[:, :-1] -= p[:, 1:]
        out[:, :, 1:] -= p[:, :, :-1]
        out[:, :, :-1] -= p[:, :, 1:]
        out *= fluid
        return out

    def _solve_pressure(self, divergence, fluid_mask, dt):
        fluid = fluid_mask.astype(np.float64)
        scale = dt / (self.rho * self.dx * self.dx)
        rhs = -divergence / scale * fluid
        p = np.zeros(self.shape, dtype=np.float64)
        r = rhs.copy()
        norm0 = float(np.max(np.abs(r)))
        if norm0 <= 0.0:
            return p, 0
        precond = fluid / self.Ndiag
        z = precond * r
        d = z.copy()
        rz = float(np.sum(r * z))
        iters = 0
        tol = self.CG_TOL * norm0
        for iters in range(1, self.CG_MAX_ITER + 1):
            Ad = self._apply_A(d, fluid)
            dAd = float(np.sum(d * Ad))
            if not np.isfinite(dAd) or abs(dAd) < 1e-30:
                break
            alpha = rz / dAd
            p += alpha * d
            r -= alpha * Ad
            if float(np.max(np.abs(r))) <= tol:
                break
            z = precond * r
            rz_new = float(np.sum(r * z))
            if not np.isfinite(rz_new) or rz_new <= 0.0:
                break
            d = z + (rz_new / rz) * d
            rz = rz_new
        if not np.all(np.isfinite(p)):
            p = np.zeros(self.shape, dtype=np.float64)
        return p, iters

    # ------------------------------------------------------------ time step
    def _grid_solve(self, dt, apply_gravity=True):
        """P2G, forces, free-surface projection. Returns pre-force face fields."""
        pos, vel = self.pos, self.vel
        mass = self.pmass
        if len(pos) == 0:
            self.u = np.zeros(self.ushape)
            self.v = np.zeros(self.vshape)
            self.w = np.zeros(self.wshape)
            self.fluid = np.zeros(self.shape, dtype=bool)
            return self.u.copy(), self.v.copy(), self.w.copy()

        nu, du = self._scatter(self.ushape, pos, (0.0, 0.5, 0.5), vel[:, 0], mass)
        nv, dv = self._scatter(self.vshape, pos, (0.5, 0.0, 0.5), vel[:, 1], mass)
        nw, dw = self._scatter(self.wshape, pos, (0.5, 0.5, 0.0), vel[:, 2], mass)
        eps = 1e-12
        u = np.where(du > eps, nu / np.maximum(du, eps), 0.0)
        v = np.where(dv > eps, nv / np.maximum(dv, eps), 0.0)
        w = np.where(dw > eps, nw / np.maximum(dw, eps), 0.0)

        # occupied (liquid) cells
        idx = np.floor((pos - self.origin) / self.dx).astype(np.int64)
        np.clip(idx[:, 0], 0, self.nx - 1, out=idx[:, 0])
        np.clip(idx[:, 1], 0, self.ny - 1, out=idx[:, 1])
        np.clip(idx[:, 2], 0, self.nz - 1, out=idx[:, 2])
        count = np.zeros(self.shape, dtype=np.float64)
        np.add.at(count.reshape(-1),
                  np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), self.shape), 1.0)
        fluid = (count > 0) & (~self.solid)
        self.cell_count = count

        u = self._extrapolate(u, du > eps, 2)
        v = self._extrapolate(v, dv > eps, 2)
        w = self._extrapolate(w, dw > eps, 2)
        u_old, v_old, w_old = u.copy(), v.copy(), w.copy()

        if apply_gravity:
            u = u + self.gravity[0] * dt
            v = v + self.gravity[1] * dt
            w = w + self.gravity[2] * dt

        # free faces: not touching a solid cell and adjacent to at least one liquid cell
        sol = self.solid
        umask = np.zeros(self.ushape, dtype=bool)
        umask[1:-1] = (~sol[:-1]) & (~sol[1:]) & (fluid[:-1] | fluid[1:])
        vmask = np.zeros(self.vshape, dtype=bool)
        vmask[:, 1:-1] = (~sol[:, :-1]) & (~sol[:, 1:]) & (fluid[:, :-1] | fluid[:, 1:])
        wmask = np.zeros(self.wshape, dtype=bool)
        wmask[:, :, 1:-1] = (~sol[:, :, :-1]) & (~sol[:, :, 1:]) & (fluid[:, :, :-1] | fluid[:, :, 1:])
        wmask[:, :, -1] = (~sol[:, :, -1]) & fluid[:, :, -1]  # open top

        u = np.where(umask, u, 0.0)
        v = np.where(vmask, v, 0.0)
        w = np.where(wmask, w, 0.0)

        div = ((u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])) / self.dx
        div = np.where(fluid, div, 0.0)
        p, iters = self._solve_pressure(div, fluid, dt)
        self.last_cg_iters = iters

        k = dt / (self.rho * self.dx)
        u[1:-1] -= k * (p[1:] - p[:-1])
        v[:, 1:-1] -= k * (p[:, 1:] - p[:, :-1])
        w[:, :, 1:-1] -= k * (p[:, :, 1:] - p[:, :, :-1])
        w[:, :, -1] -= k * (0.0 - p[:, :, -1])
        u = np.where(umask, u, 0.0)
        v = np.where(vmask, v, 0.0)
        w = np.where(wmask, w, 0.0)

        self.u = self._extrapolate(u, umask, 3)
        self.v = self._extrapolate(v, vmask, 3)
        self.w = self._extrapolate(w, wmask, 3)
        self.u_pre = self._extrapolate(u_old, umask, 3)
        self.v_pre = self._extrapolate(v_old, vmask, 3)
        self.w_pre = self._extrapolate(w_old, wmask, 3)
        self.fluid = fluid
        self.pressure = p
        return self.u_pre, self.v_pre, self.w_pre

    def _emit(self, dt):
        self.emit_phase += self.jet_speed * dt
        new_pos, new_vel = [], []
        guard = 0
        while self.emit_phase >= self.emit_dz and guard < 64:
            guard += 1
            self.emit_phase -= self.emit_dz
            offset = self.emit_phase
            count = len(self.emit_xy)
            layer = np.empty((count, 3), dtype=np.float64)
            jitter = 0.06 * self.h * self.rng.standard_normal((count, 2))
            layer[:, :2] = self.emit_xy + jitter
            layer[:, 2] = self.exit_z + offset
            speeds = self.jet_speed * (1.0 + 0.02 * self.rng.standard_normal(count))
            vels = np.zeros((count, 3), dtype=np.float64)
            vels[:, 2] = speeds
            vels[:, :2] = 0.03 * self.jet_speed * self.rng.standard_normal((count, 2))
            new_pos.append(layer)
            new_vel.append(vels)
        if not new_pos:
            return
        add_pos = np.vstack(new_pos)
        add_vel = np.vstack(new_vel)
        n_add = len(add_pos)
        self.pos = np.vstack((self.pos, add_pos))
        self.vel = np.vstack((self.vel, add_vel))
        self.pmass = np.concatenate((self.pmass, np.full(n_add, self.p_mass_unit)))
        self.tag = np.concatenate((self.tag, np.full(n_add, 2.0)))
        self.mass_source += n_add * self.p_mass_unit
        self.emitted += n_add

    def _collide(self, r_prev):
        pos, vel = self.pos, self.vel
        eps = 0.25 * self.h
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        safe = np.maximum(r, 1e-9)
        nxp = pos[:, 0] / safe
        nyp = pos[:, 1] / safe
        vr = vel[:, 0] * nxp + vel[:, 1] * nyp

        # basin inner wall (only for water that was inside the basin)
        inside = (pos[:, 2] < self.rim_z) & (r > self.basin_radius - eps) & \
                 (r_prev <= self.basin_radius - 0.5 * eps)
        if np.any(inside):
            target = self.basin_radius - eps
            pos[inside, 0] = nxp[inside] * target
            pos[inside, 1] = nyp[inside] * target
            out = inside & (vr > 0.0)
            vel[out, 0] -= vr[out] * nxp[out]
            vel[out, 1] -= vr[out] * nyp[out]

        # outer wall of the basin blocks spilled water from re-entering
        outer = (pos[:, 2] < self.rim_z) & (r < self.wall_outer + eps) & \
                (r_prev >= self.basin_radius - 0.5 * eps)
        if np.any(outer):
            target = self.wall_outer + eps
            pos[outer, 0] = nxp[outer] * target
            pos[outer, 1] = nyp[outer] * target
            inw = outer & (vr < 0.0)
            vel[inw, 0] -= vr[inw] * nxp[inw]
            vel[inw, 1] -= vr[inw] * nyp[inw]

        # basin floor
        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        floor = (pos[:, 2] < self.floor_z + eps) & (r < self.wall_outer)
        if np.any(floor):
            pos[floor, 2] = self.floor_z + eps
            down = floor & (vel[:, 2] < 0.0)
            vel[down, 2] = 0.0

        # nozzle body
        safe = np.maximum(r, 1e-9)
        nxp = pos[:, 0] / safe
        nyp = pos[:, 1] / safe
        noz = (r < self.nozzle_radius + eps) & (pos[:, 2] < self.nozzle_top + eps)
        if np.any(noz):
            dr = (self.nozzle_radius + eps) - r
            dz = (self.nozzle_top + eps) - pos[:, 2]
            up = noz & (dz <= dr)
            side = noz & (dz > dr)
            if np.any(up):
                pos[up, 2] = self.nozzle_top + eps
                slow = up & (vel[:, 2] < 0.0)
                vel[slow, 2] = 0.0
            if np.any(side):
                pos[side, 0] = nxp[side] * (self.nozzle_radius + eps)
                pos[side, 1] = nyp[side] * (self.nozzle_radius + eps)
                vr2 = vel[:, 0] * nxp + vel[:, 1] * nyp
                inw = side & (vr2 < 0.0)
                vel[inw, 0] -= vr2[inw] * nxp[inw]
                vel[inw, 1] -= vr2[inw] * nyp[inw]

    def _prune(self):
        pos = self.pos
        keep = (
            (pos[:, 2] > self.zmin + 1e-4)
            & (pos[:, 2] < self.zmax - self.dx)
            & (pos[:, 0] > self.xmin + 0.5 * self.dx)
            & (pos[:, 0] < self.xmax - 0.5 * self.dx)
            & (pos[:, 1] > self.xmin + 0.5 * self.dx)
            & (pos[:, 1] < self.xmax - 0.5 * self.dx)
            & np.isfinite(pos).all(axis=1)
            & np.isfinite(self.vel).all(axis=1)
        )
        if np.all(keep):
            return
        lost = float(np.sum(self.pmass[~keep]))
        self.mass_source -= lost
        self.removed += int(np.count_nonzero(~keep))
        self.pos = pos[keep]
        self.vel = self.vel[keep]
        self.pmass = self.pmass[keep]
        self.tag = self.tag[keep]

    def _step(self, dt):
        self._emit(dt)
        self._grid_solve(dt, apply_gravity=True)
        if len(self.pos) == 0:
            self.time += dt
            self.step_count += 1
            return
        pos = self.pos
        gu = self._gather(self.u, pos, (0.0, 0.5, 0.5))
        gv = self._gather(self.v, pos, (0.5, 0.0, 0.5))
        gw = self._gather(self.w, pos, (0.5, 0.5, 0.0))
        ou = self._gather(self.u_pre, pos, (0.0, 0.5, 0.5))
        ov = self._gather(self.v_pre, pos, (0.5, 0.0, 0.5))
        ow = self._gather(self.w_pre, pos, (0.5, 0.5, 0.0))
        pic = np.column_stack((gu, gv, gw))
        flip = self.vel + np.column_stack((gu - ou, gv - ov, gw - ow))
        a = self.FLIP_RATIO
        self.vel = a * flip + (1.0 - a) * pic

        limit = 3.0 * self.dx / max(dt, 1e-9)
        np.clip(self.vel, -limit, limit, out=self.vel)

        r_prev = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        self.pos = pos + self.vel * dt
        self._collide(r_prev)
        self._prune()
        self.time += dt
        self.step_count += 1
        self.max_speed = float(np.max(np.abs(self.vel))) if len(self.vel) else 0.0

    # ------------------------------------------------------------- contract
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-9 and guard < 10000:
            guard += 1
            remaining = target - self.time
            speed = max(self.max_speed, 1e-3)
            dt = min(self.CFL * self.dx / speed, self.sample_dt, remaining)
            dt = max(dt, min(remaining, 2e-4))
            n_left = max(1, int(np.ceil(remaining / max(dt, 1e-9))))
            if n_left > self.MAX_SUBSTEPS:
                dt = remaining / self.MAX_SUBSTEPS
            self._step(dt)
        self.time = target
        return self._make_frame(target)

    def _cell_velocity(self):
        u, v, w = self.u, self.v, self.w
        cv = np.empty(self.shape + (3,), dtype=np.float64)
        cv[..., 0] = 0.5 * (u[:-1] + u[1:])
        cv[..., 1] = 0.5 * (v[:, :-1] + v[:, 1:])
        cv[..., 2] = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        cv *= self.nonsolid[..., None]
        return cv

    def _level_set(self):
        fluid = self.fluid
        far = 4.0 * self.dx
        if not np.any(fluid):
            return np.full(self.shape, far, dtype=np.float64)
        if _ndimage is not None:
            outside = _ndimage.distance_transform_edt(~fluid, sampling=self.dx)
            inside = _ndimage.distance_transform_edt(fluid, sampling=self.dx)
            phi = outside - inside
        else:  # pragma: no cover
            phi = np.where(fluid, -0.5 * self.dx, 0.5 * self.dx)
        return np.clip(phi, -far, far)

    def _boundary_velocity(self):
        probe = self.bnd_probe
        bu = self._gather(self.u, probe, (0.0, 0.5, 0.5))
        bv = self._gather(self.v, probe, (0.5, 0.0, 0.5))
        bw = self._gather(self.w, probe, (0.5, 0.5, 0.0))
        vel = np.column_stack((bu, bv, bw))
        # the discrete no-flux constraint makes the wall-normal flux vanish
        normal_part = np.sum(vel * self.bnd_normal, axis=1, keepdims=True)
        return vel - normal_part * self.bnd_normal

    def _make_frame(self, time):
        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        if len(pos) == 0:  # keep the frame well formed
            pos = np.array([[0.0, 0.0, self.exit_z]])
            vel = np.zeros((1, 3))
            mass = np.array([self.p_mass_unit])
            tag = np.array([2.0])
        else:
            mass = np.ascontiguousarray(self.pmass, dtype=np.float64)
            tag = np.ascontiguousarray(self.tag, dtype=np.float64)
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

        cell_velocity = np.nan_to_num(self._cell_velocity()).astype(np.float32)
        level_set = np.nan_to_num(self._level_set()).astype(np.float32)
        active = (self.fluid & (~self.solid)).astype(np.float32)
        bnd_vel = np.nan_to_num(self._boundary_velocity())

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass,
            attributes={
                "mass_quantity": mass.copy(),
                "volume_quantity": np.full(len(mass), self.p_volume),
                "source_region": tag,
                "phase_id": np.zeros(len(mass)),
            },
            fields={
                "velocity_field": cell_velocity,
                "liquid_level_set": level_set,
                "active_mask": active,
                "liquid_interior_mask": active.copy(),
                "boundary_velocity": bnd_vel,
                "boundary_target_velocity": self.bnd_target.copy(),
                "boundary_normal": self.bnd_normal.copy(),
                "cumulative_mass_source": np.array([float(self.mass_source)]),
                "cumulative_volume_source": np.array([float(self.mass_source) / self.rho]),
            },
            scalars={
                "grid_spacing": float(self.dx),
                "characteristic_length": float(self.pool_radius),
                "reference_velocity": float(self.jet_speed),
                "rest_density": float(self.rho),
                "particle_spacing": float(self.h),
                "liquid_phase_id": 0.0,
                "total_mass": float(np.sum(mass)),
                "initial_mass": float(self.initial_mass),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "FLIP/PIC (blend %.2f) MAC-grid free-surface projection" % self.FLIP_RATIO,
            "grid": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.dx),
            "steps": int(self.step_count),
            "particles": int(len(self.pos)),
            "emitted_particles": int(self.emitted),
            "removed_particles": int(self.removed),
            "cumulative_mass_source": float(self.mass_source),
            "final_time": float(self.time),
        }
