"""Buoyant smoke plume through multiple rigid obstacles.

Physics: incompressible buoyant flow (Stam-style stable fluids) on a
refined collocated grid inside the enclosed voxel domain:
  - semi-Lagrangian self-advection of velocity and smoke density
  - buoyancy force proportional to local smoke concentration (plume rises
    against gravity), linear drag as sub-grid dissipation proxy
  - vorticity confinement to preserve obstacle-wake roll-up
  - pressure projection (Jacobi) with Neumann conditions on rigid
    obstacles and impermeable domain walls (no-penetration)
  - continuous scalar source at the emitter cylinder
The fine solution is box-pooled onto the exported diagnostic grid.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    R = 2                 # refinement factor over exported field grid
    BETA = 1.0            # buoyancy accel per unit smoke density (m/s^2)
    DRAG = 1.4            # linear drag (1/s)
    SRC = 3.0             # scalar source rate (1/s)
    DMAX = 2.5            # scalar clip
    JACOBI = 28           # pressure iterations per substep
    CONF = 2.5            # vorticity confinement strength
    PERT_A = 2.2          # lateral perturbation accel (m/s^2)
    PERT_BIAS = -0.4      # mean drift toward the low obstacle (m/s^2)
    NOISE = 3.0           # random perturbation accel scale (m/s^2)

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.frame0 = scene.initial_state()
        self.rng = np.random.default_rng(int(seed))
        cs = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.cshape = cs
        R = self.R
        self.shape = (cs[0] * R, cs[1] * R, cs[2] * R)
        nx, ny, nz = self.shape
        self.bounds = (-3.0, -1.2, 0.0, 3.0, 1.2, 3.8)
        bx0, by0, bz0, bx1, by1, bz1 = self.bounds
        self.dx = (bx1 - bx0) / nx
        self.dy = (by1 - by0) / ny
        self.dz = (bz1 - bz0) / nz
        self.h = (self.dx + self.dy + self.dz) / 3.0
        xc = bx0 + (np.arange(nx) + 0.5) * self.dx
        yc = by0 + (np.arange(ny) + 0.5) * self.dy
        zc = bz0 + (np.arange(nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")

        # rigid obstacles (boxes from the frozen scene geometry)
        solid = ((np.abs(self.X + 0.65) < 0.38) & (np.abs(self.Y) < 0.38)
                 & (np.abs(self.Z - 1.25) < 0.38))
        solid |= ((np.abs(self.X - 0.65) < 0.42) & (np.abs(self.Y) < 0.42)
                  & (np.abs(self.Z - 2.15) < 0.42))
        self.solid = solid
        self.fluid = ~solid

        # emitter: cylinder r=0.32 around origin near the floor
        self.src_region = ((self.X ** 2 + self.Y ** 2) < 0.32 ** 2) \
            & (self.Z < 0.30) & self.fluid
        self.src_fine = np.where(self.src_region, self.SRC, 0.0)
        self.pert_region = ((self.X ** 2 + self.Y ** 2) < 0.45 ** 2) \
            & (self.Z < 1.0) & self.fluid
        self.n_pert = int(self.pert_region.sum())

        # pressure-solve weights (Neumann on solids and domain walls)
        fp = np.pad(self.fluid, 1, constant_values=False)
        cx, cy, cz = 1.0 / self.dx ** 2, 1.0 / self.dy ** 2, 1.0 / self.dz ** 2
        self.wxp = fp[2:, 1:-1, 1:-1] * cx
        self.wxm = fp[:-2, 1:-1, 1:-1] * cx
        self.wyp = fp[1:-1, 2:, 1:-1] * cy
        self.wym = fp[1:-1, :-2, 1:-1] * cy
        self.wzp = fp[1:-1, 1:-1, 2:] * cz
        self.wzm = fp[1:-1, 1:-1, :-2] * cz
        denom = self.wxp + self.wxm + self.wyp + self.wym + self.wzp + self.wzm
        self.pdenom = np.where(denom > 0.0, denom, 1.0)
        self.axp = (self.wxp > 0).astype(np.float64)
        self.axm = (self.wxm > 0).astype(np.float64)
        self.ayp = (self.wyp > 0).astype(np.float64)
        self.aym = (self.wym > 0).astype(np.float64)
        self.azp = (self.wzp > 0).astype(np.float64)
        self.azm = (self.wzm > 0).astype(np.float64)

        self._build_boundary_samples()

        # state
        d0 = np.asarray(self.frame0.fields["transported_scalar"], float)
        self.d = np.repeat(np.repeat(np.repeat(d0, R, 0), R, 1), R, 2)
        self.d *= self.fluid
        self.u = np.zeros(self.shape + (3,))
        self.p = np.zeros(self.shape)
        self.t = 0.0
        self.sub_dt = float(cfg["sample_dt"]) / 2.0

        self.pos = np.array(self.frame0.position, float).copy()
        self.pvel = np.zeros_like(self.pos)
        self.cum_src = np.zeros(len(self.pos))

        cdx, cdy, cdz = (bx1 - bx0) / cs[0], (by1 - by0) / cs[1], (bz1 - bz0) / cs[2]
        self.scalars = {
            "grid_spacing": float((cdx + cdy + cdz) / 3.0),
            "characteristic_length": float(bz1 - bz0),
            "reference_velocity": 1.0,
            "contact_length_scale": 0.18,
        }

    # ------------------------------------------------------- boundary samples
    def _build_boundary_samples(self) -> None:
        nx, ny, nz = self.shape
        bx0, by0, bz0, bx1, by1, _ = self.bounds
        fp = np.pad(self.fluid, 1, constant_values=False)
        nb = {
            "xp": fp[2:, 1:-1, 1:-1], "xm": fp[:-2, 1:-1, 1:-1],
            "yp": fp[1:-1, 2:, 1:-1], "ym": fp[1:-1, :-2, 1:-1],
            "zp": fp[1:-1, 1:-1, 2:], "zm": fp[1:-1, 1:-1, :-2],
        }
        pts, nrm, idx = [], [], []

        def add(mask, off, normal):
            i, j, k = np.nonzero(mask)
            if len(i) == 0:
                return
            pts.append(np.column_stack((
                self.X[i, j, k] + off[0] * self.dx * 0.5,
                self.Y[i, j, k] + off[1] * self.dy * 0.5,
                self.Z[i, j, k] + off[2] * self.dz * 0.5)))
            nrm.append(np.tile(np.asarray(normal, float), (len(i), 1)))
            idx.append(np.column_stack((i + off[0], j + off[1], k + off[2])))

        # solid obstacle faces (normal points into the fluid)
        add(self.solid & nb["xp"], (1, 0, 0), (1, 0, 0))
        add(self.solid & nb["xm"], (-1, 0, 0), (-1, 0, 0))
        add(self.solid & nb["yp"], (0, 1, 0), (0, 1, 0))
        add(self.solid & nb["ym"], (0, -1, 0), (0, -1, 0))
        add(self.solid & nb["zp"], (0, 0, 1), (0, 0, 1))
        add(self.solid & nb["zm"], (0, 0, -1), (0, 0, -1))

        # impermeable domain walls (normal points into the domain)
        def wall(mask2d, fix_axis, fix_val, fix_idx, normal):
            a, b = np.nonzero(mask2d)
            if len(a) == 0:
                return
            ii = np.full_like(a, fix_idx) if fix_axis == 0 else a
            jj = np.full_like(a, fix_idx) if fix_axis == 1 else (a if fix_axis == 0 else b)
            kk = np.full_like(a, fix_idx) if fix_axis == 2 else b
            if fix_axis == 0:
                ii, jj, kk = np.full_like(a, fix_idx), a, b
            elif fix_axis == 1:
                ii, jj, kk = a, np.full_like(a, fix_idx), b
            else:
                ii, jj, kk = a, b, np.full_like(a, fix_idx)
            pts.append(np.column_stack((self.X[ii, jj, kk],
                                        self.Y[ii, jj, kk],
                                        self.Z[ii, jj, kk])))
            nrm.append(np.tile(np.asarray(normal, float), (len(a), 1)))
            idx.append(np.column_stack((ii, jj, kk)))

        f = self.fluid
        wall(f[0], 0, bx0, 0, (1, 0, 0))
        wall(f[-1], 0, bx1, nx - 1, (-1, 0, 0))
        wall(f[:, 0], 1, by0, 0, (0, 1, 0))
        wall(f[:, -1], 1, by1, ny - 1, (0, -1, 0))
        wall(f[:, :, 0], 2, bz0, 0, (0, 0, 1))
        wall(f[:, :, -1], 2, self.bounds[5], nz - 1, (0, 0, -1))

        self.bnd_pos = np.concatenate(pts)
        self.bnd_n = np.concatenate(nrm)
        self.bnd_idx = np.concatenate(idx)
        self.bnd_tv = np.zeros_like(self.bnd_n)

    # ---------------------------------------------------------------- helpers
    def _backtrace(self, dt):
        nx, ny, nz = self.shape
        bx0, by0, bz0 = self.bounds[0], self.bounds[1], self.bounds[2]
        u = self.u
        fx = np.clip((self.X - dt * u[..., 0] - bx0) / self.dx - 0.5, 0.0, nx - 1.0001 - 1e-3)
        fy = np.clip((self.Y - dt * u[..., 1] - by0) / self.dy - 0.5, 0.0, ny - 1.0001 - 1e-3)
        fz = np.clip((self.Z - dt * u[..., 2] - bz0) / self.dz - 0.5, 0.0, nz - 1.0001 - 1e-3)
        i0 = fx.astype(np.int32); j0 = fy.astype(np.int32); k0 = fz.astype(np.int32)
        tx = fx - i0; ty = fy - j0; tz = fz - k0
        corners = []
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (tx if di else 1.0 - tx) * (ty if dj else 1.0 - ty) \
                        * (tz if dk else 1.0 - tz)
                    flat = ((i0 + di) * ny + (j0 + dj)) * nz + (k0 + dk)
                    corners.append((flat, w))
        return corners

    def _gather(self, corners, a):
        flat = a.reshape(-1, *a.shape[3:])
        vec = flat.ndim == 2
        out = np.zeros_like(flat)
        for idx, w in corners:
            out += (w[..., None] if vec else w) * flat[idx]
        return out.reshape(a.shape)

    def _interp_u(self, x, y, z):
        nx, ny, nz = self.shape
        bx0, by0, bz0 = self.bounds[0], self.bounds[1], self.bounds[2]
        fx = np.clip((x - bx0) / self.dx - 0.5, 0.0, nx - 1.0001 - 1e-3)
        fy = np.clip((y - by0) / self.dy - 0.5, 0.0, ny - 1.0001 - 1e-3)
        fz = np.clip((z - bz0) / self.dz - 0.5, 0.0, nz - 1.0001 - 1e-3)
        i0 = fx.astype(np.int32); j0 = fy.astype(np.int32); k0 = fz.astype(np.int32)
        tx = fx - i0; ty = fy - j0; tz = fz - k0
        uf = self.u.reshape(-1, 3)
        out = np.zeros((len(x), 3))
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (tx if di else 1 - tx) * (ty if dj else 1 - ty) * (tz if dk else 1 - tz)
                    flat = ((i0 + di) * ny + (j0 + dj)) * nz + (k0 + dk)
                    out += w[:, None] * uf[flat]
        return out

    def _apply_bc(self):
        u = self.u
        u[self.solid] = 0.0
        u[0, :, :, 0] = 0.0
        u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0
        u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, 2] = 0.0

    def _diff_axis(self, a, axis):
        if axis == 0:
            return (a[2:, 1:-1, 1:-1] - a[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        if axis == 1:
            return (a[1:-1, 2:, 1:-1] - a[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        return (a[1:-1, 1:-1, 2:] - a[1:-1, 1:-1, :-2]) / (2.0 * self.dz)

    def _vorticity_confinement(self, dt):
        nx, ny, nz = self.shape
        up = np.zeros((nx + 2, ny + 2, nz + 2, 3))
        up[1:-1, 1:-1, 1:-1] = self.u
        om_x = self._diff_axis(up, 1)[..., 2] - self._diff_axis(up, 2)[..., 1]
        om_y = self._diff_axis(up, 2)[..., 0] - self._diff_axis(up, 0)[..., 2]
        om_z = self._diff_axis(up, 0)[..., 1] - self._diff_axis(up, 1)[..., 0]
        mag = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
        mp = np.pad(mag, 1, mode="edge")
        gx = self._diff_axis(mp, 0)
        gy = self._diff_axis(mp, 1)
        gz = self._diff_axis(mp, 2)
        norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-9
        nx_, ny_, nz_ = gx / norm, gy / norm, gz / norm
        s = self.CONF * self.h * dt
        fx = ny_ * om_z - nz_ * om_y
        fy = nz_ * om_x - nx_ * om_z
        fz = nx_ * om_y - ny_ * om_x
        fl = self.fluid
        self.u[..., 0] += s * fx * fl
        self.u[..., 1] += s * fy * fl
        self.u[..., 2] += s * fz * fl

    def _project(self, dt):
        nx, ny, nz = self.shape
        self._apply_bc()
        upx = np.pad(self.u[..., 0], 1)
        upy = np.pad(self.u[..., 1], 1)
        upz = np.pad(self.u[..., 2], 1)
        div = ((upx[2:, 1:-1, 1:-1] - upx[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
               + (upy[1:-1, 2:, 1:-1] - upy[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
               + (upz[1:-1, 1:-1, 2:] - upz[1:-1, 1:-1, :-2]) / (2.0 * self.dz))
        rhs = div / dt
        p = np.zeros(self.shape)
        for _ in range(self.JACOBI):
            pp = np.pad(p, 1, mode="edge")
            p = (self.wxp * pp[2:, 1:-1, 1:-1] + self.wxm * pp[:-2, 1:-1, 1:-1]
                 + self.wyp * pp[1:-1, 2:, 1:-1] + self.wym * pp[1:-1, :-2, 1:-1]
                 + self.wzp * pp[1:-1, 1:-1, 2:] + self.wzm * pp[1:-1, 1:-1, :-2]
                 - rhs) / self.pdenom
        self.p = p
        pp = np.pad(p, 1, mode="edge")
        pxp, pxm = pp[2:, 1:-1, 1:-1], pp[:-2, 1:-1, 1:-1]
        pyp, pym = pp[1:-1, 2:, 1:-1], pp[1:-1, :-2, 1:-1]
        pzp, pzm = pp[1:-1, 1:-1, 2:], pp[1:-1, 1:-1, :-2]
        gx = (self.axp * (pxp - p) + self.axm * (p - pxm)) \
            / ((self.axp + self.axm) * self.dx + 1e-30)
        gy = (self.ayp * (pyp - p) + self.aym * (p - pym)) \
            / ((self.ayp + self.aym) * self.dy + 1e-30)
        gz = (self.azp * (pzp - p) + self.azm * (p - pzm)) \
            / ((self.azp + self.azm) * self.dz + 1e-30)
        self.u[..., 0] -= dt * gx
        self.u[..., 1] -= dt * gy
        self.u[..., 2] -= dt * gz
        self._apply_bc()

    # ------------------------------------------------------------------ step
    def _step(self, dt):
        t = self.t
        u = self.u
        fl = self.fluid
        # buoyancy (smoke rises against gravity) and drag
        u[..., 2] += dt * self.BETA * self.d * fl
        u *= max(0.0, 1.0 - self.DRAG * dt)
        # emitter perturbation: mean drift toward low obstacle + oscillation + noise
        pr = self.pert_region
        ax = self.PERT_A * np.sin(2.0 * np.pi * 0.60 * t) + self.PERT_BIAS
        ay = self.PERT_A * np.sin(2.0 * np.pi * 0.43 * t + 1.7)
        noise = self.rng.normal(0.0, self.NOISE, size=(self.n_pert, 2))
        u[pr, 0] += dt * (ax + noise[:, 0])
        u[pr, 1] += dt * (ay + noise[:, 1])
        # vorticity confinement (wake roll-up)
        self._vorticity_confinement(dt)
        # advect velocity, then project to incompressibility
        corners = self._backtrace(dt)
        self.u = self._gather(corners, u)
        self.u[self.solid] = 0.0
        self._project(dt)
        # inject and advect smoke
        self.d = self._gather(self._backtrace(dt), self.d)
        self.d += dt * self.src_fine
        self.d *= fl
        np.clip(self.d, 0.0, self.DMAX, out=self.d)
        # diagnostic particles
        v = self._interp_u(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        pn = self.pos + dt * v
        lo = np.array(self.bounds[:3]) + 1e-4
        hi = np.array(self.bounds[3:]) - 1e-4
        pn = np.clip(pn, lo, hi)
        ii = np.clip(((pn[:, 0] - self.bounds[0]) / self.dx).astype(int), 0, self.shape[0] - 1)
        jj = np.clip(((pn[:, 1] - self.bounds[1]) / self.dy).astype(int), 0, self.shape[1] - 1)
        kk = np.clip(((pn[:, 2] - self.bounds[2]) / self.dz).astype(int), 0, self.shape[2] - 1)
        inside = self.solid[ii, jj, kk]
        pn[inside] = self.pos[inside]
        v[inside] = 0.0
        self.pos = pn
        self.pvel = v
        self.t += dt

    def _pool(self, a):
        R = self.R
        b = a.reshape(self.cshape[0], R, self.cshape[1], R, self.cshape[2], R, *a.shape[3:])
        return b.mean(axis=(1, 3, 5))

    # -------------------------------------------------------------- run loop
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.sub_dt, time - self.t)
            self._step(dt)
        i, j, k = self.bnd_idx[:, 0], self.bnd_idx[:, 1], self.bnd_idx[:, 2]
        bnd_v = self.u[i, j, k]
        mask = self._pool(self.fluid.astype(np.float64)) > 0.5
        fields = {
            "transported_scalar": self._pool(self.d),
            "scalar_source": self._pool(self.src_fine),
            "active_mask": mask,
            "velocity_field": self._pool(self.u),
            "boundary_normal": self.bnd_n,
            "boundary_target_velocity": self.bnd_tv,
            "boundary_velocity": bnd_v,
            "cumulative_mass_source": self.cum_src,
        }
        frame = StateFrame(
            self.t, self.pos.copy(), self.pvel.copy(),
            mass=self.frame0.mass, body_id=self.frame0.body_id,
            attributes=dict(self.frame0.attributes),
            fields=fields, scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = np.linalg.norm(self.u, axis=-1)
        return {
            "time": float(self.t),
            "max_scalar": float(self.d.max()),
            "total_scalar": float(self.d.sum()),
            "max_speed": float(speed.max()),
            "mean_speed_fluid": float(speed[self.fluid].mean()),
            "n_boundary_samples": int(len(self.bnd_n)),
        }
