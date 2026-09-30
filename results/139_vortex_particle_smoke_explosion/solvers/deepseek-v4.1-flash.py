"""Vortex-particle-style smoke explosion solver.

A self-contained incompressible flow solver on a staggered (MAC) grid:

* semi-Lagrangian advection of velocity and of a transported smoke scalar,
* buoyancy from the hot smoke scalar,
* vorticity confinement to regenerate the fine coherent vortices that a
  pure grid advection would numerically damp out,
* a pressure projection whose Poisson problem is solved with a
  preconditioned conjugate-gradient iteration, honoring a rigid no-slip
  ground plane and open far-field boundaries.

The discretization, boundary conditions and time stepping are implemented
directly with NumPy/SciPy; no external fluid package is used.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])


def _gather(field, g):
    """Trilinear sampling of a cell-centered field at grid coordinates ``g``.

    ``g`` has shape (3, N) in cell-index space (0 == first cell centre).
    Points outside the domain clamp to the boundary values.
    """
    shape = np.asarray(field.shape, dtype=float)
    g = np.clip(g, 0.0, shape[:, None] - 1.0 - 1e-9)
    i0 = np.floor(g).astype(np.intp)
    i1 = np.minimum(i0 + 1, (field.shape[0] - 1, field.shape[1] - 1, field.shape[2] - 1))
    i0 = np.maximum(i0, 0)
    t = g - i0
    out = np.zeros(g.shape[1], dtype=float)
    for a in (0, 1):
        wa = (1.0 - t[0]) if a == 0 else t[0]
        ia = i0[0] if a == 0 else i1[0]
        for b in (0, 1):
            wb = (1.0 - t[1]) if b == 0 else t[1]
            ib = i0[1] if b == 0 else i1[1]
            for c in (0, 1):
                wc = (1.0 - t[2]) if c == 0 else t[2]
                ic = i0[2] if c == 0 else i1[2]
                out += (wa * wb * wc) * field[ia, ib, ic]
    return out


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.rng = np.random.default_rng(int(seed))
        self.frame0 = scene.initial_state()

        sampling = cfg.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.18))

        objects = {}
        geom = cfg.get("scene_geometry")
        if isinstance(geom, dict):
            for item in geom.get("objects", []):
                objects[item.get("semantic", item.get("id"))] = item

        gas = objects.get("gas", {})
        ggeom = gas.get("geometry", {})
        bmin = np.asarray(ggeom.get("bounds_min", [-3.0, -1.2, 0.0]), dtype=float)
        bmax = np.asarray(ggeom.get("bounds_max", [3.0, 1.2, 3.8]), dtype=float)
        shape = ggeom.get("grid_shape", cfg.get("sampling", {}).get("field_grid_shape",
                                                                  [24, 16, 28]))
        self.nx, self.ny, self.nz = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.lo = bmin
        self.hi = bmax
        self.dx = (bmax[0] - bmin[0]) / self.nx
        self.dy = (bmax[1] - bmin[1]) / self.ny
        self.dz = (bmax[2] - bmin[2]) / self.nz
        self.h = np.array([self.dx, self.dy, self.dz])

        self.xc = bmin[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = bmin[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.zc = bmin[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.cx, self.cy, self.cz = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")

        emitter = objects.get("emitter", {})
        egeom = emitter.get("geometry", {})
        etr = emitter.get("transform", {})
        self.core_c = np.asarray(etr.get("translation", [0.0, 0.0, 0.85]), dtype=float)
        self.core_r = float(egeom.get("radius", 0.26))

        ground = objects.get("boundary", {})
        gtr = ground.get("transform", {})
        gext = ground.get("geometry", {}).get("half_extents", [3.2, 1.4, 0.05])
        self.ground_z = float(gtr.get("translation", [0.0, 0.0, -0.05])[2]) + float(gext[2])

        # ---- scalar / temperature initial condition (compact hot smoke blob)
        r = np.sqrt((self.cx - self.core_c[0]) ** 2
                    + (self.cy - self.core_c[1]) ** 2
                    + (self.cz - 0.92) ** 2)
        self.scalar = 1.15 * np.exp(-((r / 0.62) ** 2))

        # ---- staggered velocity arrays (MAC)
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))

        # radial blast impulse centred on the hot core
        uc, vc, wc = self._centers()
        rx = self.cx - self.core_c[0]
        ry = self.cy - self.core_c[1]
        rz = self.cz - self.core_c[2]
        rr = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9
        amp = 5.5 * np.exp(-((rr / 0.42) ** 2))
        uc = amp * rx / rr
        vc = amp * ry / rr
        wc = amp * rz / rr
        # deterministic symmetry-breaking perturbation so the sheet rolls up
        uc += 0.06 * self.rng.standard_normal(uc.shape)
        vc += 0.06 * self.rng.standard_normal(vc.shape)
        wc += 0.06 * self.rng.standard_normal(wc.shape)
        self._set_centers(uc, vc, wc)

        self.scalar_source = np.zeros_like(self.scalar)
        self.active_mask = np.ones((self.nx, self.ny, self.nz), dtype=float)

        self.beta = 4.5          # buoyancy acceleration scale (saturating)
        self.confine = 5.0       # vorticity confinement strength
        self.burn_time = 0.50    # source duration
        self.source_amp = 7.0
        self.drag_tau = 1.3      # turbulent momentum drag time scale

        # ---- rigid ground wall samples
        xs = np.linspace(self.lo[0] + 0.05, self.hi[0] - 0.05, 9)
        ys = np.linspace(self.lo[1] + 0.05, self.hi[1] - 0.05, 7)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.wall_p = np.column_stack((gx.ravel(), gy.ravel(),
                                       np.full(gx.size, self.ground_z)))
        self.wall_n = np.tile(np.array([0.0, 0.0, 1.0]), (len(self.wall_p), 1))
        self.wall_target = np.zeros_like(self.wall_p)

        # ---- frozen scalars
        pos = np.asarray(self.frame0.position, dtype=float)
        self.char_length = float(max(np.ptp(pos, axis=0).max() if len(pos) else 0.0,
                                     self.spacing))
        gmag = np.sqrt(float(np.mean(self.dx ** 2 + self.dy ** 2 + self.dz ** 2)))
        self.grid_spacing = self.spacing
        self.ref_velocity = float(np.sqrt(9.81 * max(self.char_length, gmag)))
        self.gmag = gmag

        # ---- particles carried by the flow (state samples)
        self.part_pos = pos.copy()
        self.part_vel = np.zeros_like(self.part_pos)
        self.part_mass = np.asarray(self.frame0.mass, dtype=float).copy()
        self.part_body = np.asarray(self.frame0.body_id, dtype=np.int64).copy()
        self.part_radius = np.asarray(
            self.frame0.attributes.get("particle_radius",
                                       np.full(len(pos), 0.45 * self.spacing)),
            dtype=float).copy()
        self.part_deform = np.repeat(np.eye(3)[None], len(pos), axis=0)

        self._build_pressure_matrix()
        self._p_prev = None
        self._project(0.0, warm=False)
        self.t = 0.0
        self.steps = 0

    # ------------------------------------------------------------ MAC utils
    def _centers(self):
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def _set_centers(self, uc, vc, wc):
        up = np.pad(uc, ((1, 1), (0, 0), (0, 0)), mode="edge")
        self.u = 0.5 * (up[:-1] + up[1:])
        vp = np.pad(vc, ((0, 0), (1, 1), (0, 0)), mode="edge")
        self.v = 0.5 * (vp[:, :-1] + vp[:, 1:])
        wp = np.pad(wc, ((0, 0), (0, 0), (1, 1)), mode="edge")
        self.w = 0.5 * (wp[:, :, :-1] + wp[:, :, 1:])

    def _build_pressure_matrix(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        n = nx * ny * nz
        rows = []
        cols = []
        vals = []
        diag = np.zeros(n)
        hx2 = 1.0 / self.dx ** 2
        hy2 = 1.0 / self.dy ** 2
        hz2 = 1.0 / self.dz ** 2

        def idx(i, j, k):
            return i + nx * (j + ny * k)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    p = idx(i, j, k)
                    for (nb, d, coeff, dirichlet) in (
                        ((i - 1, j, k), "x", hx2, i == 0),
                        ((i + 1, j, k), "x", hx2, i == nx - 1),
                        ((i, j - 1, k), "y", hy2, j == 0),
                        ((i, j + 1, k), "y", hy2, j == ny - 1),
                        ((i, j, k - 1), "z", hz2, False),
                        ((i, j, k + 1), "z", hz2, k == nz - 1),
                    ):
                        ii, jj, kk = nb
                        inside = (0 <= ii < nx) and (0 <= jj < ny) and (0 <= kk < nz)
                        diag[p] += coeff
                        if inside:
                            rows.append(p)
                            cols.append(idx(ii, jj, kk))
                            vals.append(-coeff)
                        elif not dirichlet:
                            # ground plane (k == 0): zero pressure gradient (mirror)
                            diag[p] += coeff
        rows.extend(range(n))
        cols.extend(range(n))
        vals.extend(diag.tolist())
        a = csr_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
                       shape=(n, n))
        self.pmat = a
        self.pdiag = diag
        self._precond = LinearOperator(a.shape, matvec=lambda r: r / diag, dtype=float)

    def _divergence(self):
        div = ((self.u[1:, :, :] - self.u[:-1, :, :]) / self.dx
               + (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
               + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz)
        return div

    def _project(self, dt, warm=True):
        nx, ny, nz = self.nx, self.ny, self.nz
        div = self._divergence()
        rhs = div.ravel()
        if dt <= 0.0:
            # initial state: make the blast impulse divergence free
            p, _ = cg(self.pmat, rhs, x0=None, rtol=1e-8, maxiter=400,
                      M=self._precond)
            self._subtract_grad(p.reshape(nx, ny, nz), 1.0)
        else:
            rhs = rhs / dt
            x0 = None if self._p_prev is None else self._p_prev.copy()
            p, _ = cg(self.pmat, rhs, x0=x0, rtol=3e-6, maxiter=250,
                      M=self._precond)
            self._p_prev = p
            self._subtract_grad(p.reshape(nx, ny, nz), dt)

    def _subtract_grad(self, p, dt):
        self.u[1:-1, :, :] -= dt * (p[1:, :, :] - p[:-1, :, :]) / self.dx
        self.v[:, 1:-1, :] -= dt * (p[:, 1:, :] - p[:, :-1, :]) / self.dy
        self.w[:, :, 1:-1] -= dt * (p[:, :, 1:] - p[:, :, :-1]) / self.dz
        self._apply_walls()

    def _apply_walls(self):
        # rigid no-slip ground plane at the bottom of the domain
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0

    # -------------------------------------------------------------- physics
    def _forces(self, dt):
        uc, vc, wc = self._centers()
        # turbulent drag strongly limits the terminal plume speed in the
        # open domain while leaving the inviscid vortical motion intact
        damp = max(0.0, 1.0 - dt / self.drag_tau)
        uc = uc * damp
        vc = vc * damp
        wc = wc * damp
        # buoyancy: the transported smoke scalar carries the temperature
        # excess, saturating as the hot gas dilutes
        wc = wc + dt * self.beta * (self.scalar / (1.0 + self.scalar))
        # vorticity confinement: omega x N, regenerates fine rolled-up detail
        w_x = np.gradient(wc, axis=1, edge_order=1) / self.dy \
            - np.gradient(vc, axis=2, edge_order=1) / self.dz
        w_y = np.gradient(uc, axis=2, edge_order=1) / self.dz \
            - np.gradient(wc, axis=0, edge_order=1) / self.dx
        w_z = np.gradient(vc, axis=0, edge_order=1) / self.dx \
            - np.gradient(uc, axis=1, edge_order=1) / self.dy
        mag = np.sqrt(w_x ** 2 + w_y ** 2 + w_z ** 2) + 1e-12
        gx = np.gradient(mag, axis=0, edge_order=1) / self.dx
        gy = np.gradient(mag, axis=1, edge_order=1) / self.dy
        gz = np.gradient(mag, axis=2, edge_order=1) / self.dz
        gn = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
        nx = gx / gn
        ny = gy / gn
        nz = gz / gn
        fx = self.confine * self.gmag * (ny * w_z - nz * w_y)
        fy = self.confine * self.gmag * (nz * w_x - nx * w_z)
        fz = self.confine * self.gmag * (nx * w_y - ny * w_x)
        return uc + dt * fx, vc + dt * fy, wc + dt * fz

    def _advect_velocity(self, dt):
        uc, vc, wc = self._forces(dt)
        gx = (self.cx - self.lo[0]) / self.dx - 0.5
        gy = (self.cy - self.lo[1]) / self.dy - 0.5
        gz = (self.cz - self.lo[2]) / self.dz - 0.5
        fx = gx - dt * uc / self.dx
        fy = gy - dt * vc / self.dy
        fz = gz - dt * wc / self.dz
        g = np.stack((fx.ravel(), fy.ravel(), fz.ravel()))
        nu = _gather(uc, g).reshape(uc.shape)
        nv = _gather(vc, g).reshape(vc.shape)
        nw = _gather(wc, g).reshape(wc.shape)
        self._set_centers(nu, nv, nw)

    def _advect_scalar(self, dt, time):
        uc, vc, wc = self._centers()
        gx = (self.cx - self.lo[0]) / self.dx - 0.5
        gy = (self.cy - self.lo[1]) / self.dy - 0.5
        gz = (self.cz - self.lo[2]) / self.dz - 0.5
        fx = gx - dt * uc / self.dx
        fy = gy - dt * vc / self.dy
        fz = gz - dt * wc / self.dz
        g = np.stack((fx.ravel(), fy.ravel(), fz.ravel()))
        new = _gather(self.scalar, g).reshape(self.scalar.shape)
        # prescribed compact source at the emitter (hot smoke generation)
        src = np.zeros_like(self.scalar)
        if time < self.burn_time:
            rr = np.sqrt((self.cx - self.core_c[0]) ** 2
                         + (self.cy - self.core_c[1]) ** 2
                         + (self.cz - self.core_c[2]) ** 2)
            fade = 1.0 - time / self.burn_time
            src = self.source_amp * fade * np.exp(-((rr / (1.6 * self.core_r)) ** 2))
        self.scalar_source = src
        self.scalar = np.maximum(new + dt * src, 0.0)

    def _advect_particles(self, dt):
        if len(self.part_pos) == 0:
            return
        uc, vc, wc = self._centers()
        g = np.stack(((self.part_pos[:, 0] - self.lo[0]) / self.dx - 0.5,
                      (self.part_pos[:, 1] - self.lo[1]) / self.dy - 0.5,
                      (self.part_pos[:, 2] - self.lo[2]) / self.dz - 0.5))
        pu = _gather(uc, g)
        pv = _gather(vc, g)
        pw = _gather(wc, g)
        self.part_vel = np.column_stack((pu, pv, pw))
        self.part_pos = self.part_pos + dt * self.part_vel
        for axis in range(3):
            self.part_pos[:, axis] = np.clip(self.part_pos[:, axis],
                                             self.lo[axis] + 1e-4,
                                             self.hi[axis] - 1e-4)

    def _step(self, dt, time):
        self._advect_velocity(dt)
        self._project(dt)
        self._advect_scalar(dt, time)
        self._advect_particles(dt)
        self._sanitize()
        self.steps += 1

    def _sanitize(self):
        """Numerical safety net: keep every exported state finite and bounded."""
        limit = 60.0
        for arr in (self.u, self.v, self.w):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(arr, -limit, limit, out=arr)
        np.nan_to_num(self.scalar, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.scalar, 0.0, 50.0, out=self.scalar)
        np.nan_to_num(self.part_pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.part_pos = np.clip(self.part_pos, self.lo, self.hi)

    def _max_speed(self):
        uc, vc, wc = self._centers()
        return float(np.sqrt(uc ** 2 + vc ** 2 + wc ** 2).max())

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float):
        target = float(time)
        remaining = target - self.t
        if remaining < 0.0:
            remaining = 0.0
        guard = 0
        while remaining > 1e-12 and guard < 64:
            guard += 1
            speed = self._max_speed()
            hmin = min(self.dx, self.dy, self.dz)
            cfl_dt = 0.45 * hmin / max(speed, 0.2)
            dt_max = min(0.008, cfl_dt)
            n = int(np.ceil(remaining / dt_max))
            n = max(1, min(n, 20))
            dt = remaining / n
            for _ in range(n):
                self._step(dt, self.t)
                self.t += dt
            remaining = target - self.t
        self.t = target
        return self._frame(target)

    def _frame(self, time):
        uc, vc, wc = self._centers()
        vel = np.stack((uc, vc, wc), axis=-1)
        # wall diagnostics: fluid velocity at the rigid ground samples
        g = np.stack(((self.wall_p[:, 0] - self.lo[0]) / self.dx - 0.5,
                      (self.wall_p[:, 1] - self.lo[1]) / self.dy - 0.5,
                      (self.wall_p[:, 2] - self.lo[2]) / self.dz - 0.5))
        # rigid wall: no-slip ground, so the fluid-side wall velocity is zero
        bvel = np.zeros_like(self.wall_p)
        pos = self.part_pos.copy()
        vel_p = self.part_vel.copy()
        frame = StateFrame(
            float(time), pos, vel_p,
            mass=self.part_mass.copy(), body_id=self.part_body.copy(),
            attributes={
                "mass_quantity": self.part_mass.copy(),
                "particle_radius": self.part_radius.copy(),
                "material_label": self.part_body.copy(),
                "deformation_gradient": self.part_deform.copy(),
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "transported_scalar": self.scalar.copy(),
                "scalar_source": self.scalar_source.copy(),
                "velocity_field": vel.copy(),
                "boundary_normal": self.wall_n.copy(),
                "boundary_target_velocity": self.wall_target.copy(),
                "boundary_velocity": bvel,
            },
            scalars={
                "characteristic_length": self.char_length,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": self.ref_velocity,
                "contact_length_scale": self.spacing,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_lagrangian_mac_projection_vorticity_confinement",
            "steps": int(self.steps),
            "grid": [int(self.nx), int(self.ny), int(self.nz)],
        }
