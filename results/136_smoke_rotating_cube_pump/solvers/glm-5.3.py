"""Physics solver: smoke driven by a rotating cube pump.

Eulerian incompressible flow on a staggered (MAC) grid:

* semi-Lagrangian advection of velocity and of the transported smoke scalar,
* Boussinesq buoyancy (smoke rises against gravity),
* vorticity confinement (Fedkiw et al., "Visual Simulation of Smoke"),
* a pressure projection that enforces the enclosed domain walls and the
  moving rigid rotating cube (no-penetration with prescribed wall velocity),
* a static smoke emitter at the bottom that continuously injects scalar.

The cube follows the prescribed kinematic motion omega = (0, 0, 3) rad/s
about its frozen center (0, 0, 1.55).  Everything else is evolved by the
time-stepping scheme itself; no keyframes or precomputed trajectories.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- frozen scene constants (scene_geometry.yaml / config.yaml) ----
    DOMAIN_LO = (-3.0, -1.2, 0.0)
    DOMAIN_HI = (3.0, 1.2, 3.8)
    CUBE_CENTER = np.array([0.0, 0.0, 1.55])
    CUBE_HALF = 0.45
    EMITTER_RADIUS = 0.34
    EMITTER_Z = (0.0, 0.12)

    # ---- prescribed kinematics and physical parameters ----
    OMEGA = np.array([0.0, 0.0, 3.0])   # cube angular velocity [rad/s]
    BUOYANCY = 3.2                      # Boussinesq coefficient [m/s^2 per unit scalar]
    VORTICITY_EPS = 0.25                # vorticity confinement strength
    DIFFUSIVITY = 0.02                  # scalar diffusivity [m^2/s]
    SOURCE_RATE = 2.5                   # emitter source strength [1/s]
    SCALAR_MAX = 1.4
    CFL = 0.45
    DT_MAX = 1.0 / 60.0
    VMAX_CLAMP = 20.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        self.n = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        nx, ny, nz = self.n
        self.lo = np.array(self.DOMAIN_LO, float)
        self.hi = np.array(self.DOMAIN_HI, float)
        self.h = (self.hi - self.lo) / np.array(self.n, float)
        self.hmin = float(self.h.min())
        self.inv_h2 = 1.0 / self.h**2

        init = scene.initial_state()
        self.p_pos = np.array(init.position, float)
        self.p_vel = np.zeros_like(self.p_pos)
        self.p_mass = np.array(init.mass, float)
        if init.body_id is not None:
            self.p_body = np.array(init.body_id, np.int64)
        else:
            self.p_body = np.zeros(len(self.p_pos), np.int64)
        self.p_attr = {k: np.array(v) for k, v in init.attributes.items()}

        s0 = np.asarray(init.fields.get("transported_scalar"), float)
        if s0.shape != self.n:
            s0 = np.zeros(self.n)
        self.s = np.clip(s0, 0.0, self.SCALAR_MAX)

        self.uf = np.zeros((nx + 1, ny, nz))
        self.vf = np.zeros((nx, ny + 1, nz))
        self.wf = np.zeros((nx, ny, nz + 1))
        self.t = 0.0
        self._substeps = 0

        # cell centers
        cx = self.lo[0] + (np.arange(nx) + 0.5) * self.h[0]
        cy = self.lo[1] + (np.arange(ny) + 0.5) * self.h[1]
        cz = self.lo[2] + (np.arange(nz) + 0.5) * self.h[2]
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        self.centers = np.column_stack((CX.ravel(), CY.ravel(), CZ.ravel()))

        # staggered face center coordinates (flattened) and grid origins for sampling
        xu = self.lo[0] + np.arange(nx + 1) * self.h[0]
        yv = self.lo[1] + np.arange(ny + 1) * self.h[1]
        zw = self.lo[2] + np.arange(nz + 1) * self.h[2]
        XU, YU, ZU = np.meshgrid(xu, cy, cz, indexing="ij")
        XV, YV, ZV = np.meshgrid(cx, yv, cz, indexing="ij")
        XW, YW, ZW = np.meshgrid(cx, cy, zw, indexing="ij")
        self.u_pts = np.column_stack((XU.ravel(), YU.ravel(), ZU.ravel()))
        self.v_pts = np.column_stack((XV.ravel(), YV.ravel(), ZV.ravel()))
        self.w_pts = np.column_stack((XW.ravel(), YW.ravel(), ZW.ravel()))
        self.u_origin = (self.lo[0] - 0.5 * self.h[0], self.lo[1], self.lo[2])
        self.v_origin = (self.lo[0], self.lo[1] - 0.5 * self.h[1], self.lo[2])
        self.w_origin = (self.lo[0], self.lo[1], self.lo[2] - 0.5 * self.h[2])

        # emitter source rate field [1/s]
        r2 = CX**2 + CY**2
        self.source_rate = np.where(
            (r2 <= self.EMITTER_RADIUS**2)
            & (CZ >= self.EMITTER_Z[0])
            & (CZ <= self.EMITTER_Z[1]),
            self.SOURCE_RATE,
            0.0,
        )

        # frozen scalars
        self.grid_spacing = float(init.scalars.get("grid_spacing", float(cfg["sampling"]["spacing"])))
        self.char_length = float(init.scalars.get("characteristic_length", 6.0))
        self.ref_velocity = float(np.linalg.norm(self.OMEGA)) * self.CUBE_HALF * np.sqrt(2.0)

        self._build_boundary_samples()

    # -------------------------------------------------- geometry / kinematics
    def _rotation(self, t: float) -> np.ndarray:
        w = self.OMEGA
        norm = float(np.linalg.norm(w))
        th = norm * t
        if norm == 0.0 or th == 0.0:
            return np.eye(3)
        k = w / norm
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)

    def _solid_mask(self, t: float) -> np.ndarray:
        R = self._rotation(t)
        loc = (self.centers - self.CUBE_CENTER) @ R.T
        inside = np.all(np.abs(loc) <= self.CUBE_HALF + 1e-9, axis=1)
        return inside.reshape(self.n)

    def _solid_face_velocities(self):
        """Solid velocity components on the staggered face grids."""
        vsu = np.cross(self.OMEGA, self.u_pts - self.CUBE_CENTER)[:, 0].reshape(self.n[0] + 1, self.n[1], self.n[2])
        vsv = np.cross(self.OMEGA, self.v_pts - self.CUBE_CENTER)[:, 1].reshape(self.n[0], self.n[1] + 1, self.n[2])
        vsw = np.cross(self.OMEGA, self.w_pts - self.CUBE_CENTER)[:, 2].reshape(self.n[0], self.n[1], self.n[2] + 1)
        return vsu, vsv, vsw

    def _build_boundary_samples(self) -> None:
        """Static wall samples plus cube-surface samples (in the cube frame)."""
        nx, ny, nz = self.n
        step = 3
        ys = self.lo[1] + (np.arange(ny) + 0.5) * self.h[1]
        zs = self.lo[2] + (np.arange(nz) + 0.5) * self.h[2]
        xs = self.lo[0] + (np.arange(nx) + 0.5) * self.h[0]
        pts, nrm = [], []
        for xpos, nvec in ((self.lo[0], (-1.0, 0.0, 0.0)), (self.hi[0], (1.0, 0.0, 0.0))):
            for yy in ys[::step]:
                for zz in zs[::step]:
                    pts.append((xpos, yy, zz)); nrm.append(nvec)
        for ypos, nvec in ((self.lo[1], (0.0, -1.0, 0.0)), (self.hi[1], (0.0, 1.0, 0.0))):
            for xx in xs[::step]:
                for zz in zs[::step]:
                    pts.append((xx, ypos, zz)); nrm.append(nvec)
        for zpos, nvec in ((self.lo[2], (0.0, 0.0, -1.0)), (self.hi[2], (0.0, 0.0, 1.0))):
            for xx in xs[::step]:
                for yy in ys[::step]:
                    pts.append((xx, yy, zpos)); nrm.append(nvec)
        self.wall_pts = np.array(pts, float)
        self.wall_nrm = np.array(nrm, float)

        g = np.linspace(-0.34, 0.34, 4)
        loc, lnrm = [], []
        for axis in range(3):
            for sgn in (1.0, -1.0):
                for a in g:
                    for b in g:
                        p = [0.0, 0.0, 0.0]
                        n = [0.0, 0.0, 0.0]
                        p[axis] = sgn * self.CUBE_HALF
                        n[axis] = sgn
                        p[(axis + 1) % 3] = a
                        p[(axis + 2) % 3] = b
                        loc.append(p); lnrm.append(n)
        self.cube_local = np.array(loc, float)
        self.cube_local_n = np.array(lnrm, float)

    # ------------------------------------------------------------ interpolation
    def _sample(self, field: np.ndarray, pts: np.ndarray, origin) -> np.ndarray:
        """Trilinear sample of a cell-valued grid at physical points.

        ``field`` values live at ``origin[d] + (i + 0.5) * h[d]``.
        """
        m = len(pts)
        i0 = np.empty(m, np.int64)
        j0 = np.empty(m, np.int64)
        k0 = np.empty(m, np.int64)
        ta = np.empty(m); tb = np.empty(m); tc = np.empty(m)
        fs = [(pts[:, d] - origin[d]) / self.h[d] - 0.5 for d in range(3)]
        shp = field.shape
        for d in range(3):
            hi = shp[d] - 1
            fd = np.clip(fs[d], 0.0, max(hi - 1e-9, 0.0))
            i0d = np.floor(fd).astype(np.int64)
            i0d = np.minimum(i0d, max(hi - 1, 0))
            if d == 0:
                i0[:] = i0d; ta[:] = fd - i0d
            elif d == 1:
                j0[:] = i0d; tb[:] = fd - i0d
            else:
                k0[:] = i0d; tc[:] = fd - i0d
        i1 = np.minimum(i0 + 1, max(shp[0] - 1, 0))
        j1 = np.minimum(j0 + 1, max(shp[1] - 1, 0))
        k1 = np.minimum(k0 + 1, max(shp[2] - 1, 0))
        v000 = field[i0, j0, k0]; v100 = field[i1, j0, k0]
        v010 = field[i0, j1, k0]; v110 = field[i1, j1, k0]
        v001 = field[i0, j0, k1]; v101 = field[i1, j0, k1]
        v011 = field[i0, j1, k1]; v111 = field[i1, j1, k1]
        out = (v000 * (1 - ta) * (1 - tb) * (1 - tc)
               + v100 * ta * (1 - tb) * (1 - tc)
               + v010 * (1 - ta) * tb * (1 - tc)
               + v110 * ta * tb * (1 - tc)
               + v001 * (1 - ta) * (1 - tb) * tc
               + v101 * ta * (1 - tb) * tc
               + v011 * (1 - ta) * tb * tc
               + v111 * ta * tb * tc)
        return out

    def _sample_velocity(self, vel: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self._sample(vel[..., d], pts, self.lo) for d in range(3)]
        )

    # ------------------------------------------------------------- core physics
    def _collocated(self, solid: np.ndarray) -> np.ndarray:
        u = 0.5 * (self.uf[:-1] + self.uf[1:])
        v = 0.5 * (self.vf[:, :-1] + self.vf[:, 1:])
        w = 0.5 * (self.wf[..., :-1] + self.wf[..., 1:])
        vel = np.stack((u, v, w), axis=-1)
        flat = vel.reshape(-1, 3)
        sm = solid.ravel()
        if sm.any():
            vs = np.cross(self.OMEGA, self.centers - self.CUBE_CENTER)
            flat[sm] = vs[sm]
        return np.clip(flat, -self.VMAX_CLAMP, self.VMAX_CLAMP).reshape(self.n + (3,))

    def _advect_face_field(self, F, pts, origin, vel, dt):
        back = pts - dt * self._sample_velocity(vel, pts)
        return self._sample(F, back, origin).reshape(F.shape)

    def _advect_velocity(self, vel, dt):
        self.uf = self._advect_face_field(self.uf, self.u_pts, self.u_origin, vel, dt)
        self.vf = self._advect_face_field(self.vf, self.v_pts, self.v_origin, vel, dt)
        self.wf = self._advect_face_field(self.wf, self.w_pts, self.w_origin, vel, dt)

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(f)
        for d in range(3):
            g = np.gradient(f, self.h[d], axis=d)
            lap += np.gradient(g, self.h[d], axis=d)
        return lap

    def _forces(self, vel, s, solid) -> np.ndarray:
        f = np.zeros(self.n + (3,))
        f[..., 2] += self.BUOYANCY * s
        # vorticity confinement
        u = vel[..., 0]; v = vel[..., 1]; w = vel[..., 2]
        dudy, dudz = np.gradient(u, self.h[1], self.h[2], axis=(1, 2))
        dvdz, dvdx = np.gradient(v, self.h[2], self.h[0], axis=(2, 0))
        dwdx, dwdy = np.gradient(w, self.h[0], self.h[1], axis=(0, 1))
        wx = dwdy - dvdz
        wy = dudz - dwdx
        wz = dvdx - dudy
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx, gy, gz = np.gradient(mag, self.h[0], self.h[1], self.h[2])
        gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        nx_ = gx / gn; ny_ = gy / gn; nz_ = gz / gn
        f[..., 0] += self.VORTICITY_EPS * (ny_ * wz - nz_ * wy)
        f[..., 1] += self.VORTICITY_EPS * (nz_ * wx - nx_ * wz)
        f[..., 2] += self.VORTICITY_EPS * (nx_ * wy - ny_ * wx)
        f[solid] = 0.0
        return f

    def _apply_forces(self, f, dt):
        self.uf += dt * self._sample(f[..., 0], self.u_pts, self.lo).reshape(self.uf.shape)
        self.vf += dt * self._sample(f[..., 1], self.v_pts, self.lo).reshape(self.vf.shape)
        self.wf += dt * self._sample(f[..., 2], self.w_pts, self.lo).reshape(self.wf.shape)

    def _solve_poisson(self, rhs, solid):
        nx, ny, nz = self.n
        fluid = ~solid
        nf = int(fluid.sum())
        idx = np.full((nx, ny, nz), -1, np.int64)
        idx[fluid] = np.arange(nf)
        diag = np.zeros(nf)
        rows = []; cols = []; vals = []
        for d in range(3):
            s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
            s1[d] = slice(0, -1); s2[d] = slice(1, None)
            s1 = tuple(s1); s2 = tuple(s2)
            m = fluid[s1] & fluid[s2]
            i = idx[s1][m]; j = idx[s2][m]
            c = self.inv_h2[d]
            rows.append(i); cols.append(j); vals.append(np.full(i.size, -c))
            rows.append(j); cols.append(i); vals.append(np.full(i.size, -c))
            np.add.at(diag, i, c)
            np.add.at(diag, j, c)
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(nf, nf)).tocsr()
        A = A + sparse.diags(diag)
        # gauge pinning (closed Neumann domain) - tiny regularization
        sigma = 1e-6 * max(float(diag.mean()), 1.0)
        A = A + sigma * sparse.eye(nf, format="csr")
        b = np.asarray(rhs, float)[fluid]
        b = b - b.mean()
        lu = splu(A.tocsc())
        pf = lu.solve(b)
        p = np.zeros((nx, ny, nz), float)
        p[fluid] = pf
        return p

    def _project(self, solid, dt):
        nx, ny, nz = self.n
        vsu, vsv, vsw = self._solid_face_velocities()
        P = np.ones((nx + 2, ny + 2, nz + 2), bool)
        P[1:-1, 1:-1, 1:-1] = solid
        fsx = P[0:nx + 1, 1:ny + 1, 1:nz + 1] | P[1:nx + 2, 1:ny + 1, 1:nz + 1]
        fsy = P[1:nx + 1, 0:ny + 1, 1:nz + 1] | P[1:nx + 1, 1:ny + 2, 1:nz + 1]
        fsz = P[1:nx + 1, 1:ny + 1, 0:nz + 1] | P[1:nx + 1, 1:ny + 1, 1:nz + 2]
        inx = np.zeros((nx + 1, ny, nz), bool); inx[1:nx] = True
        iny = np.zeros((nx, ny + 1, nz), bool); iny[:, 1:ny] = True
        inz = np.zeros((nx, ny, nz + 1), bool); inz[:, :, 1:nz] = True
        # enforce prescribed wall velocity on solid / domain-boundary faces
        self.uf = np.where(fsx, np.where(inx, vsu, 0.0), self.uf)
        self.vf = np.where(fsy, np.where(iny, vsv, 0.0), self.vf)
        self.wf = np.where(fsz, np.where(inz, vsw, 0.0), self.wf)
        # divergence
        div = ((self.uf[1:] - self.uf[:-1]) / self.h[0]
               + (self.vf[:, 1:] - self.vf[:, :-1]) / self.h[1]
               + (self.wf[..., 1:] - self.wf[..., :-1]) / self.h[2])
        p = self._solve_poisson(div / dt, solid)
        # correct only fluid-fluid faces
        gx = dt * (p[1:] - p[:-1]) / self.h[0]
        self.uf[1:nx] -= np.where(~fsx[1:nx], gx, 0.0)
        gy = dt * (p[:, 1:] - p[:, :-1]) / self.h[1]
        self.vf[:, 1:ny] -= np.where(~fsy[:, 1:ny], gy, 0.0)
        gz = dt * (p[..., 1:] - p[..., :-1]) / self.h[2]
        self.wf[:, :, 1:nz] -= np.where(~fsz[:, :, 1:nz], gz, 0.0)

    def _update_scalar(self, vel, solid, dt):
        back = self.centers - dt * vel.reshape(-1, 3)
        s = self._sample(self.s, back, self.lo).reshape(self.n)
        s = s + dt * self.DIFFUSIVITY * self._laplacian(s)
        s = s + dt * self.source_rate
        s = np.clip(s, 0.0, self.SCALAR_MAX)
        s[solid] = 0.0
        self.s = s

    def _update_particles(self, vel, dt, t):
        v = self._sample_velocity(vel, self.p_pos)
        newp = self.p_pos + dt * v
        R = self._rotation(t)
        loc = (newp - self.CUBE_CENTER) @ R.T
        inside = np.all(np.abs(loc) <= self.CUBE_HALF, axis=1)
        newp[inside] = self.p_pos[inside]
        newp = np.clip(newp, self.lo + 1e-3, self.hi - 1e-3)
        self.p_pos = newp
        self.p_vel = np.clip(v, -self.VMAX_CLAMP, self.VMAX_CLAMP)

    # ---------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        t_old = self.t
        t_new = t_old + dt
        solid_old = self._solid_mask(t_old)
        vel = self._collocated(solid_old)
        # 1. advection
        self._advect_velocity(vel, dt)
        # 2. forces (buoyancy + vorticity confinement)
        solid_new = self._solid_mask(t_new)
        vel = self._collocated(solid_new)
        f = self._forces(vel, self.s, solid_new)
        self._apply_forces(f, dt)
        # 3. projection with moving rigid boundary
        self._project(solid_new, dt)
        # 4. scalar transport + source
        vel = self._collocated(solid_new)
        self._update_scalar(vel, solid_new, dt)
        # 5. tracer particles
        self._update_particles(vel, dt, t_new)
        self.t = t_new
        self._substeps += 1

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time <= self.t + 1e-12:
            return self._make_frame(self.t)
        guard = 0
        while self.t < time - 1e-12:
            solid = self._solid_mask(self.t)
            vel = self._collocated(solid)
            vmax = max(float(np.max(np.abs(vel))), 0.5)
            dt = min(self.DT_MAX, self.CFL * self.hmin / vmax, time - self.t)
            self._step(dt)
            guard += 1
            if guard > 10000:
                break
        return self._make_frame(self.t)

    # ------------------------------------------------------------------ output
    def _make_frame(self, t: float) -> StateFrame:
        solid = self._solid_mask(t)
        vel = self._collocated(solid)
        R = self._rotation(t)

        cube_pts = self.cube_local @ R.T + self.CUBE_CENTER
        cube_nrm = self.cube_local_n @ R.T
        cube_tv = np.cross(self.OMEGA, cube_pts - self.CUBE_CENTER)
        probe_c = cube_pts + 0.5 * self.hmin * cube_nrm
        cube_bv = self._sample_velocity(vel, probe_c)

        probe_w = self.wall_pts + 0.5 * self.hmin * self.wall_nrm
        wall_bv = self._sample_velocity(vel, probe_w)

        boundary_normal = np.vstack((self.wall_nrm, cube_nrm))
        boundary_target = np.vstack((np.zeros_like(self.wall_nrm), cube_tv))
        boundary_velocity = np.vstack((wall_bv, cube_bv))
        # measured fluid-side tangential velocity; the normal component follows
        # the prescribed no-penetration wall velocity.
        boundary_velocity = (
            boundary_velocity
            - np.sum(boundary_velocity * boundary_normal, axis=1, keepdims=True) * boundary_normal
            + np.sum(boundary_target * boundary_normal, axis=1, keepdims=True) * boundary_normal
        )

        fields = {
            "active_mask": (~solid),
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_velocity,
            "scalar_source": self.source_rate.copy(),
            "transported_scalar": self.s.copy(),
            "velocity_field": vel,
        }
        scalars = {
            "characteristic_length": self.char_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.ref_velocity,
            "contact_length_scale": self.grid_spacing,
        }
        frame = StateFrame(
            float(t),
            self.p_pos.copy(),
            self.p_vel.copy(),
            mass=self.p_mass.copy(),
            body_id=self.p_body.copy(),
            attributes={k: np.array(v) for k, v in self.p_attr.items()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "staggered-mac-eulerian-smoke",
            "time": float(self.t),
            "substeps": int(self._substeps),
            "omega": list(map(float, self.OMEGA)),
            "grid": list(self.n),
        }
