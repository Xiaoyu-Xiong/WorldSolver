"""Smoke driven by a rotating cube pump (Fedkiw et al. style stable-fluids).

Physically based incompressible smoke solver on a collocated MAC-less grid:
buoyancy force, vorticity confinement, semi-Lagrangian advection, Jacobi
pressure projection, kinematic rotating-cube obstacle with rigid-body
velocity, and an emitting smoke source near the ground.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        init = scene.initial_state()
        self._init = init
        self.seed = int(seed)

        # particles (tracers) carried from the scene initial state
        self.pos = init.position.copy()
        self.vel = init.velocity.copy()
        self.mass = np.asarray(init.mass).copy()
        self.body_id = np.asarray(init.body_id).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in init.attributes.items()}
        self.scalars = {k: float(v) for k, v in init.scalars.items()}

        g = tuple(cfg["sampling"]["field_grid_shape"])
        self.nx, self.ny, self.nz = g
        self.bmin = np.array([-3.0, -1.2, 0.0])
        self.bmax = np.array([3.0, 1.2, 3.8])
        L = self.bmax - self.bmin
        self.dx, self.dy, self.dz = L[0] / self.nx, L[1] / self.ny, L[2] / self.nz
        self.hmin = min(self.dx, self.dy, self.dz)

        x = self.bmin[0] + (np.arange(self.nx) + 0.5) * self.dx
        y = self.bmin[1] + (np.arange(self.ny) + 0.5) * self.dy
        z = self.bmin[2] + (np.arange(self.nz) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z

        # state fields
        self.u = np.zeros(g + (3,))
        self.s = np.asarray(init.fields["transported_scalar"]).copy()
        self.p = np.zeros(g)

        # smoke source (cylinder r<=0.34 near z=0)
        r2 = X ** 2 + Y ** 2
        self.emit_mask = (r2 <= 0.34 ** 2) & (Z <= 0.24)
        self.emit_rate = 1.6           # scalar units per second
        self.jet = 2.2                 # upward momentum injection m/s^2
        self.buoy = 3.2                # buoyancy acceleration per unit scalar
        self.vort_eps = 1.8            # vorticity confinement strength
        self.scalar_source = self.emit_rate * self.emit_mask.astype(float)

        # rotating cube (kinematic obstacle)
        self.cube_c = np.array([0.0, 0.0, 1.55])
        self.cube_h = 0.45
        self.omega = 1.6               # rad/s about +z
        self.obst = np.zeros(g, dtype=bool)

        # boundary samples: ground, domain side walls, cube faces
        self._build_boundary_samples()

        self.dt = float(cfg["sample_dt"]) / 3.0
        self.t = 0.0
        self.steps = 0

    def _build_boundary_samples(self):
        pts, nrm, tgt, kind = [], [], [], []
        # ground (z=0 plane, inward normal +z, static)
        gx = np.linspace(-2.75, 2.75, 12)
        gy = np.linspace(-1.05, 1.05, 8)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        p = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
        pts.append(p); nrm.append(np.tile([0.0, 0.0, 1.0], (len(p), 1)))
        tgt.append(np.zeros_like(p)); kind.append(np.zeros(len(p), np.int32))
        # side walls (inward normals, static)
        wall_specs = [
            (self.bmin[0], 0, np.array([1.0, 0.0, 0.0])),
            (self.bmax[0], 0, np.array([-1.0, 0.0, 0.0])),
            (self.bmin[1], 1, np.array([0.0, 1.0, 0.0])),
            (self.bmax[1], 1, np.array([0.0, -1.0, 0.0])),
        ]
        zz = np.linspace(0.2, 3.6, 9)
        for val, axis, nvec in wall_specs:
            if axis == 0:
                a = np.linspace(-1.0, 1.0, 7); A, Bc = np.meshgrid(a, zz, indexing="ij")
                q = np.column_stack([np.full(A.size, val), A.ravel(), Bc.ravel()])
            else:
                a = np.linspace(-2.7, 2.7, 7); A, Bc = np.meshgrid(a, zz, indexing="ij")
                q = np.column_stack([A.ravel(), np.full(A.size, val), Bc.ravel()])
            pts.append(q); nrm.append(np.tile(nvec, (len(q), 1)))
            tgt.append(np.zeros_like(q)); kind.append(np.zeros(len(q), np.int32))
        # cube faces (local frame; rotated each frame) -> kind 1
        f = np.linspace(-0.3, 0.3, 5)
        F1, F2 = np.meshgrid(f, f, indexing="ij")
        faces = []
        for ax in range(3):
            for sgn in (-1.0, 1.0):
                lp = np.zeros((F1.size, 3))
                nloc = np.zeros(3); nloc[ax] = sgn
                cols = [i for i in range(3) if i != ax]
                lp[:, cols[0]] = F1.ravel(); lp[:, cols[1]] = F2.ravel()
                lp[:, ax] = sgn * self.cube_h
                faces.append((lp, np.tile(nloc, (len(lp), 1))))
        self.cube_local_pts = np.concatenate([fp for fp, _ in faces])
        self.cube_local_nrm = np.concatenate([fn for _, fn in faces])
        nb = len(self.cube_local_pts)
        kind.append(np.ones(nb, np.int32))
        self.static_pts = np.concatenate(pts)
        self.static_nrm = np.concatenate(nrm)
        self.static_tgt = np.concatenate(tgt)
        self.static_kind = np.concatenate(kind[: len(pts)])
        self.n_static = len(self.static_pts)

    # ---------------- helpers ----------------
    def _rot_z(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _interp(self, f, xq, yq, zq):
        """Trilinear interpolation of field f (grid or grid+extra dims)."""
        shape = xq.shape
        gx = np.clip((xq - self.bmin[0]) / self.dx - 0.5, 0.0, self.nx - 1.0001)
        gy = np.clip((yq - self.bmin[1]) / self.dy - 0.5, 0.0, self.ny - 1.0001)
        gz = np.clip((zq - self.bmin[2]) / self.dz - 0.5, 0.0, self.nz - 1.0001)
        i0 = gx.astype(np.int64); j0 = gy.astype(np.int64); k0 = gz.astype(np.int64)
        i1 = np.minimum(i0 + 1, self.nx - 1); j1 = np.minimum(j0 + 1, self.ny - 1)
        k1 = np.minimum(k0 + 1, self.nz - 1)
        tx = gx - i0; ty = gy - j0; tz = gz - k0
        out = np.zeros(shape + f.shape[3:], dtype=float)
        for ii, wx in ((i0, 1 - tx), (i1, tx)):
            for jj, wy in ((j0, 1 - ty), (j1, ty)):
                for kk, wz in ((k0, 1 - tz), (k1, tz)):
                    wgt = (wx * wy * wz)[..., None] if f.ndim == 4 else (wx * wy * wz)
                    out += wgt * f[ii, jj, kk]
        return out

    def _update_obstacle(self, theta):
        R = self._rot_z(-theta)
        d = np.stack([self.X - self.cube_c[0], self.Y - self.cube_c[1],
                      self.Z - self.cube_c[2]], axis=-1)
        loc = d @ R.T
        self.obst = (np.abs(loc).max(axis=-1) < self.cube_h + 0.02)
        # rigid-body velocity inside cube, clear smoke
        cross = np.empty_like(d)
        cross[..., 0] = -self.omega * d[..., 1]
        cross[..., 1] = self.omega * d[..., 0]
        cross[..., 2] = 0.0
        self.u[self.obst] = cross[self.obst]
        self.s[self.obst] = 0.0

    def _wall_bc(self):
        u = self.u
        u[0, :, :, 0] = 0.0; u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = 0.0; u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        u[:, :, -1, :] = u[:, :, -2, :]   # open top (outflow)

    def _project(self, iters=70):
        dx, dy, dz = self.dx, self.dy, self.dz
        u, v, w = self.u[..., 0], self.u[..., 1], self.u[..., 2]
        div = (np.gradient(u, dx, axis=0) + np.gradient(v, dy, axis=1)
               + np.gradient(w, dz, axis=2))
        ax, ay, az = 1.0 / dx ** 2, 1.0 / dy ** 2, 1.0 / dz ** 2
        denom = 2.0 * (ax + ay + az)
        p = self.p
        obst = self.obst
        wrel = 1.6
        for _ in range(iters):
            pe = np.empty_like(p); pe[:-1] = p[1:]; pe[-1] = p[-1]
            pw = np.empty_like(p); pw[1:] = p[:-1]; pw[0] = p[0]
            pn = np.empty_like(p); pn[:, :-1] = p[:, 1:]; pn[:, -1] = p[:, -1]
            ps = np.empty_like(p); ps[:, 1:] = p[:, :-1]; ps[:, 0] = p[:, 0]
            pu = np.empty_like(p); pu[:, :, :-1] = p[:, :, 1:]; pu[:, :, -1] = p[:, :, -1]
            pd = np.empty_like(p); pd[:, :, 1:] = p[:, :, :-1]; pd[:, :, 0] = p[:, :, 0]
            pn_new = ((pe + pw) * ax + (pn + ps) * ay + (pu + pd) * az - div) / denom
            p += wrel * (pn_new - p)
            p[obst] = 0.0
        self.p = p
        self.u[..., 0] -= np.gradient(p, dx, axis=0)
        self.u[..., 1] -= np.gradient(p, dy, axis=1)
        self.u[..., 2] -= np.gradient(p, dz, axis=2)

    def _vorticity_confinement(self):
        dx, dy, dz = self.dx, self.dy, self.dz
        u, v, w = self.u[..., 0], self.u[..., 1], self.u[..., 2]
        wx = np.gradient(w, dy, axis=1) - np.gradient(v, dz, axis=2)
        wy = np.gradient(u, dz, axis=2) - np.gradient(w, dx, axis=0)
        wz = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
        mag = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-8
        gx = np.gradient(mag, dx, axis=0)
        gy = np.gradient(mag, dy, axis=1)
        gz = np.gradient(mag, dz, axis=2)
        nrm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8
        Nx, Ny, Nz = gx / nrm, gy / nrm, gz / nrm
        fx = self.vort_eps * (Ny * wz - Nz * wy)
        fy = self.vort_eps * (Nz * wx - Nx * wz)
        fz = self.vort_eps * (Nx * wy - Ny * wx)
        return fx, fy, fz

    def _advect_field(self, f, dt):
        xq = self.X - self.u[..., 0] * dt
        yq = self.Y - self.u[..., 1] * dt
        zq = self.Z - self.u[..., 2] * dt
        return self._interp(f, xq, yq, zq)

    # ---------------- simulation step ----------------
    def _substep(self, dt):
        theta = self.omega * (self.t + dt)
        self._update_obstacle(theta)
        u = self.u
        # source injection + buoyancy + vorticity confinement
        self.s += self.emit_rate * self.emit_mask * dt
        u[..., 2] += (self.jet * self.emit_mask + self.buoy * self.s) * dt
        fx, fy, fz = self._vorticity_confinement()
        u[..., 0] += fx * dt; u[..., 1] += fy * dt; u[..., 2] += fz * dt
        # mild dissipation, clamp
        self.s *= (1.0 - 0.03 * dt)
        np.clip(self.s, 0.0, 4.0, out=self.s)
        self._update_obstacle(theta)
        self._wall_bc()
        self._project()
        self._wall_bc()
        self._update_obstacle(theta)
        # semi-Lagrangian advection
        un = self._advect_field(self.u, dt)
        sn = self._advect_field(self.s, dt)
        self.u = un; self.s = sn
        self.s[self.obst] = 0.0
        np.clip(self.s, 0.0, 4.0, out=self.s)
        self._update_obstacle(theta)
        self._wall_bc()
        self._project(iters=50)
        self._wall_bc()
        self._update_obstacle(theta)
        self.t += dt
        self.steps += 1

    def _advance_particles(self, dt_total, nsub):
        for _ in range(nsub):
            v = self._interp(self.u, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
            newp = self.pos + v * (dt_total / nsub)
            # keep inside domain
            newp = np.clip(newp, self.bmin + 1e-4, self.bmax - 1e-4)
            # reject moves into the rotating cube
            theta = self.omega * self.t
            R = self._rot_z(-theta)
            loc = (newp - self.cube_c) @ R.T
            inside = (np.abs(loc).max(axis=1) < self.cube_h + 0.03)
            newp[inside] = self.pos[inside]
            self.pos = newp
        self.vel = self._interp(self.u, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])

    # ---------------- output ----------------
    def _boundary_arrays(self):
        theta = self.omega * self.t
        R = self._rot_z(theta)
        cp = self.cube_c + self.cube_local_pts @ R.T
        cn = self.cube_local_nrm @ R.T
        rel = cp - self.cube_c
        ct = np.empty_like(cp)
        ct[:, 0] = -self.omega * rel[:, 1]
        ct[:, 1] = self.omega * rel[:, 0]
        ct[:, 2] = 0.0
        pts = np.concatenate([self.static_pts, cp])
        nrm = np.concatenate([self.static_nrm, cn])
        tgt = np.concatenate([self.static_tgt, ct])
        bv = self._interp(self.u, pts[:, 0], pts[:, 1], pts[:, 2])
        return nrm, tgt, bv

    def _make_frame(self, time):
        nrm, tgt, bv = self._boundary_arrays()
        fields = {
            "transported_scalar": self.s.copy(),
            "scalar_source": self.scalar_source.copy(),
            "active_mask": (~self.obst).copy(),
            "velocity_field": self.u.copy(),
            "boundary_normal": nrm,
            "boundary_target_velocity": tgt,
            "boundary_velocity": bv,
            "cumulative_mass_source": np.zeros(len(self.pos)),
        }
        frame = StateFrame(
            float(time), self.pos.copy(), self.vel.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={k: v.copy() for k, v in self.attributes.items()},
            fields=fields,
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        remaining = time - self.t
        if remaining > 1e-12:
            nsub = max(1, int(np.ceil(remaining / self.dt)))
            dt = remaining / nsub
            for _ in range(nsub):
                self._substep(dt)
            self._advance_particles(remaining, nsub)
        self.t = time
        self.vel = self._interp(self.u, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        return {
            "time": self.t,
            "steps": self.steps,
            "scalar_total": float(self.s.sum()),
            "ke": float(0.5 * (self.u ** 2).sum()),
        }
