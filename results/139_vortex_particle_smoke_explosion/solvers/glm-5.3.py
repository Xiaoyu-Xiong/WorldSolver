"""Vortex-particle smoke explosion solver.

Physically-based Eulerian smoke simulation:
  * collocated velocity / smoke grid (frozen 24x16x28 voxel grid),
  * initial blast impulse (radial expansion from the hot core) + buoyancy,
  * semi-Lagrangian advection of velocity and smoke,
  * vorticity confinement to preserve/produce coherent small vortices
    (the mechanism behind the mushroom-like turbulent cloud),
  * pressure projection (Poisson solve) enforcing incompressibility with a
    free-slip solid ground at z = 0 and open boundaries elsewhere,
  * tracer particles carried by the grid velocity.

No animation, keyframes or precomputed trajectories: every frame is produced
by explicit time integration of the governing equations.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- physical / numerical constants -------------------------------------
    DT_MAX = 0.0125           # substep ceiling [s]
    BUOYANCY = 3.6            # Boussinesq buoyancy coefficient [m/s^2 per unit smoke]
    CONFINEMENT = 0.45        # vorticity confinement strength
    VISCOSITY = 0.02          # weak momentum diffusion (also damps grid noise)
    DISSIPATION = 0.015       # smoke decay rate [1/s]
    BLAST_SPEED = 3.2         # initial radial blast speed [m/s]
    NOISE_AMP = 0.35          # symmetry-breaking perturbation [m/s]
    SOURCE_T = 0.30           # emitter active duration [s]
    SOURCE_RATE = 3.5         # emitter smoke rate [1/s]
    U_CLAMP = 7.0             # velocity safety clamp [m/s]
    CFL = 0.40

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)

        sampling = cfg["sampling"]
        nx, ny, nz = (int(v) for v in sampling["field_grid_shape"])
        self.shape = (nx, ny, nz)
        self.x = np.linspace(-3.0, 3.0, nx)
        self.y = np.linspace(-1.2, 1.2, ny)
        self.z = np.linspace(0.0, 3.8, nz)
        self.dx = (self.x[-1] - self.x[0]) / max(nx - 1, 1)
        self.dy = (self.y[-1] - self.y[0]) / max(ny - 1, 1)
        self.dz = (self.z[-1] - self.z[0]) / max(nz - 1, 1)
        self.hmin = min(self.dx, self.dy, self.dz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        init = scene.initial_state()
        self.s = np.array(init.fields["transported_scalar"], dtype=float)
        self.u = np.zeros(self.shape + (3,), dtype=float)

        # tracer particles (frozen sampling cloud)
        self.p0 = np.array(init.position, dtype=float)
        self.v0 = np.array(init.velocity, dtype=float)
        self.mass0 = np.array(init.mass, dtype=float)
        self.body_id0 = np.array(init.body_id, dtype=np.int32)
        self.attributes0 = {k: np.array(v) for k, v in init.attributes.items()}
        self.char_length = float(init.scalars.get("characteristic_length", 0.9))
        self.grid_spacing = float(sampling["spacing"])

        # --- initial condition: blast impulse + noise, projected to be
        #     divergence free (the hot core at (0,0,0.85) blows outward).
        cx = np.array([0.0, 0.0, 0.85])
        rx = self.X - cx[0]
        ry = self.Y - cx[1]
        rz = self.Z - cx[2]
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        r = np.maximum(r, 1e-6)
        shell = np.exp(-((r / 0.42) ** 2))
        self.u[..., 0] = (rx / r) * self.BLAST_SPEED * shell
        self.u[..., 1] = (ry / r) * self.BLAST_SPEED * shell
        self.u[..., 2] = (rz / r) * self.BLAST_SPEED * shell
        self.u[..., 2] += 0.9 * np.exp(-((r / 0.55) ** 2))  # initial updraft
        rng = np.random.default_rng(self.seed)
        env = np.exp(-((r / 0.6) ** 2))
        self.u += rng.normal(0.0, self.NOISE_AMP, size=self.u.shape) * env[..., None]

        # --- constant Poisson operator (open boundaries, solid floor)
        self._lu = spla.splu(self._laplacian().tocsc())

        self._project()
        self._wall_bc()

        self.t = 0.0
        self._substeps = 0

        # boundary samples: ground top face (z = 0), outward normal +z
        gx, gy = np.meshgrid(self.x, self.y, indexing="ij")
        self.b_pos = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
        self.b_normal = np.tile(np.array([0.0, 0.0, 1.0]), (len(self.b_pos), 1))
        self.b_target = np.zeros_like(self.b_normal)
        self.b_query_z = 0.5 * self.dz

    # --------------------------------------------------------- discretization
    def _laplacian(self):
        nx, ny, nz = self.shape
        dx, dy, dz = self.dx, self.dy, self.dz
        n = nx * ny * nz

        def idx(i, j, k):
            return (k * ny + j) * nx + i

        rows, cols, vals = [], [], []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    c = idx(i, j, k)
                    diag = 0.0
                    # axis x (open both faces)
                    if i > 0:
                        rows.append(c); cols.append(idx(i - 1, j, k)); vals.append(1.0 / dx ** 2)
                        diag -= 1.0 / dx ** 2
                    else:  # open face at half-cell distance
                        diag -= 2.0 / dx ** 2
                    if i < nx - 1:
                        rows.append(c); cols.append(idx(i + 1, j, k)); vals.append(1.0 / dx ** 2)
                        diag -= 1.0 / dx ** 2
                    else:
                        diag -= 2.0 / dx ** 2
                    # axis y (open both faces)
                    if j > 0:
                        rows.append(c); cols.append(idx(i, j - 1, k)); vals.append(1.0 / dy ** 2)
                        diag -= 1.0 / dy ** 2
                    else:
                        diag -= 2.0 / dy ** 2
                    if j < ny - 1:
                        rows.append(c); cols.append(idx(i, j + 1, k)); vals.append(1.0 / dy ** 2)
                        diag -= 1.0 / dy ** 2
                    else:
                        diag -= 2.0 / dy ** 2
                    # axis z: solid (Neumann) floor at k = 0, open top
                    if k > 0:
                        rows.append(c); cols.append(idx(i, j, k - 1)); vals.append(1.0 / dz ** 2)
                        diag -= 1.0 / dz ** 2
                    if k < nz - 1:
                        rows.append(c); cols.append(idx(i, j, k + 1)); vals.append(1.0 / dz ** 2)
                        diag -= 1.0 / dz ** 2
                    else:
                        diag -= 2.0 / dz ** 2
                    rows.append(c); cols.append(c); vals.append(diag)
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n))

    def _grad(self, F):
        gx = np.gradient(F, self.dx, axis=0)
        gy = np.gradient(F, self.dy, axis=1)
        gz = np.gradient(F, self.dz, axis=2)
        return gx, gy, gz

    def _lap(self, F):
        g = np.gradient(F, self.dx, axis=0)
        lx = np.gradient(g, self.dx, axis=0)
        g = np.gradient(F, self.dy, axis=1)
        ly = np.gradient(g, self.dy, axis=1)
        g = np.gradient(F, self.dz, axis=2)
        lz = np.gradient(g, self.dz, axis=2)
        return lx + ly + lz

    def _curl(self):
        ux = self.u[..., 0]; uy = self.u[..., 1]; uz = self.u[..., 2]
        duy_dz = np.gradient(uy, self.dz, axis=2)
        duz_dy = np.gradient(uz, self.dy, axis=1)
        duz_dx = np.gradient(uz, self.dx, axis=0)
        dux_dz = np.gradient(ux, self.dz, axis=2)
        dux_dy = np.gradient(ux, self.dy, axis=1)
        duy_dx = np.gradient(uy, self.dx, axis=0)
        return (duy_dz - duz_dy, duz_dx - dux_dz, dux_dy - duy_dx)

    def _project(self):
        """Pressure projection: u <- u - grad p, div u -> 0."""
        div = (np.gradient(self.u[..., 0], self.dx, axis=0)
               + np.gradient(self.u[..., 1], self.dy, axis=1)
               + np.gradient(self.u[..., 2], self.dz, axis=2))
        p = self._lu_solve(div)
        gp = self._grad_p(p)
        self.u[..., 0] -= gp[0]
        self.u[..., 1] -= gp[1]
        self.u[..., 2] -= gp[2]

    def _lu_solve(self, div):
        # available after initialize(); fall back to CG-free direct build
        rhs = div.ravel()
        if getattr(self, "_lu", None) is not None:
            return self._lu.solve(rhs).reshape(self.shape)
        # should not happen (project called once before factorization)
        a = self._laplacian().tocsc()
        return spla.spsolve(a, rhs).reshape(self.shape)

    def _grad_p(self, p):
        """Gradient of pressure with boundary ghosts (mirror at solid floor,
        anti-mirror p_face = 0 at open faces)."""
        nx, ny, nz = self.shape
        pp = np.empty((nx + 2, ny + 2, nz + 2))
        pp[1:-1, 1:-1, 1:-1] = p
        # x faces: open
        pp[0, 1:-1, 1:-1] = -p[0, :, :]
        pp[-1, 1:-1, 1:-1] = -p[-1, :, :]
        # y faces: open
        pp[1:-1, 0, 1:-1] = -p[:, 0, :]
        pp[1:-1, -1, 1:-1] = -p[:, -1, :]
        # z faces: solid floor (mirror), open top
        pp[1:-1, 1:-1, 0] = p[:, :, 0]
        pp[1:-1, 1:-1, -1] = -p[:, :, -1]
        gx = (pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        gy = (pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        gz = (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) / (2.0 * self.dz)
        return gx, gy, gz

    def _wall_bc(self):
        """Free-slip ground: no normal flow through z = 0."""
        self.u[:, :, 0, 2] = 0.0

    # ------------------------------------------------------------ interpolation
    def _sample(self, F, px, py, pz):
        """Trilinear sampling of F (scalar or [...,3]) at points (px,py,pz)."""
        nx, ny, nz = self.shape
        gx = (px - self.x[0]) / self.dx
        gy = (py - self.y[0]) / self.dy
        gz = (pz - self.z[0]) / self.dz
        gx = np.clip(gx, 0.0, nx - 1.0000001)
        gy = np.clip(gy, 0.0, ny - 1.0000001)
        gz = np.clip(gz, 0.0, nz - 1.0000001)
        i0 = np.minimum(gx.astype(np.int64), nx - 2)
        j0 = np.minimum(gy.astype(np.int64), ny - 2)
        k0 = np.minimum(gz.astype(np.int64), nz - 2)
        fx = gx - i0
        fy = gy - j0
        fz = gz - k0

        def tri(G):
            return (
                G[i0, j0, k0] * (1 - fx) * (1 - fy) * (1 - fz)
                + G[i0 + 1, j0, k0] * fx * (1 - fy) * (1 - fz)
                + G[i0, j0 + 1, k0] * (1 - fx) * fy * (1 - fz)
                + G[i0, j0, k0 + 1] * (1 - fx) * (1 - fy) * fz
                + G[i0 + 1, j0 + 1, k0] * fx * fy * (1 - fz)
                + G[i0 + 1, j0, k0 + 1] * fx * (1 - fy) * fz
                + G[i0, j0 + 1, k0 + 1] * (1 - fx) * fy * fz
                + G[i0 + 1, j0 + 1, k0 + 1] * fx * fy * fz
            )

        if F.ndim == 4:
            return np.stack(
                (tri(F[..., 0]), tri(F[..., 1]), tri(F[..., 2])), axis=-1
            )
        return tri(F)

    # ----------------------------------------------------------------- physics
    def _source_rate(self, t):
        """Prescribed smoke emission rate from the blast core [1/s]."""
        if t >= self.SOURCE_T:
            return np.zeros(self.shape)
        amp = self.SOURCE_RATE * (1.0 - t / self.SOURCE_T)
        r2 = (self.X ** 2 + self.Y ** 2 + (self.Z - 0.85) ** 2)
        return amp * np.exp(-r2 / (0.30 ** 2))

    def _step(self, dt: float) -> None:
        # 1) buoyancy (Boussinesq): hot smoke rises
        self.u[..., 2] += self.BUOYANCY * self.s * dt

        # 2) vorticity confinement: re-energize small coherent vortices so the
        #    plume edge rolls up instead of staying a smooth blob
        ox, oy, oz = self._curl()
        mag = np.sqrt(ox * ox + oy * oy + oz * oz) + 1e-12
        ex, ey, ez = self._grad(mag)
        nlen = np.sqrt(ex * ex + ey * ey + ez * ez) + 1e-12
        nxv, nyv, nzv = ex / nlen, ey / nlen, ez / nlen
        self.u[..., 0] += self.CONFINEMENT * (nyv * oz - nzv * oy) * dt
        self.u[..., 1] += self.CONFINEMENT * (nzv * ox - nxv * oz) * dt
        self.u[..., 2] += self.CONFINEMENT * (nxv * oy - nyv * ox) * dt

        # 3) weak diffusion (numerical regularization)
        if self.VISCOSITY > 0.0:
            for c in range(3):
                self.u[..., c] += self.VISCOSITY * dt * self._lap(self.u[..., c])

        # 4) semi-Lagrangian advection of velocity and smoke
        bx = self.X - self.u[..., 0] * dt
        by = self.Y - self.u[..., 1] * dt
        bz = self.Z - self.u[..., 2] * dt
        u_adv = self._sample(self.u, bx, by, bz)
        s_adv = self._sample(self.s, bx, by, bz)
        self.u = u_adv
        self.s = s_adv * (1.0 - self.DISSIPATION * dt)

        # 5) emitter source
        self.s = self.s + self._source_rate(self.t) * dt
        np.clip(self.s, 0.0, 3.0, out=self.s)

        # 6) incompressibility + wall conditions
        self._project()
        self._wall_bc()

        # safety clamp (keeps states finite and stable)
        speed = np.sqrt((self.u ** 2).sum(axis=-1))
        scale = np.minimum(1.0, self.U_CLAMP / np.maximum(speed, 1e-9))
        self.u *= scale[..., None]

        # 7) tracer particles carried by the flow
        pv = self._sample(self.u, self.p0[:, 0], self.p0[:, 1], self.p0[:, 2])
        self.p0 = self.p0 + pv * dt
        self.v0 = pv
        self.p0[:, 2] = np.clip(self.p0[:, 2], 0.03, 3.75)
        self.p0[:, 0] = np.clip(self.p0[:, 0], self.x[0], self.x[-1])
        self.p0[:, 1] = np.clip(self.p0[:, 1], self.y[0], self.y[-1])

        self.t += dt
        self._substeps += 1

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float):
        target = max(float(time), 0.0)
        while self.t < target - 1e-9:
            umax = float(np.sqrt((self.u ** 2).sum(axis=-1)).max())
            dt = min(self.DT_MAX,
                     self.CFL * self.hmin / max(umax, 0.5),
                     target - self.t)
            dt = max(dt, 1e-6)
            self._step(dt)

        bvel = self._sample(
            self.u,
            self.b_pos[:, 0],
            self.b_pos[:, 1],
            np.full(len(self.b_pos), self.b_query_z),
        )
        umax = float(np.sqrt((self.u ** 2).sum(axis=-1)).max())
        frame = StateFrame(
            time=float(target),
            position=self.p0.copy(),
            velocity=self.v0.copy(),
            mass=self.mass0.copy(),
            body_id=self.body_id0.copy(),
            attributes={k: np.array(v, copy=True) for k, v in self.attributes0.items()},
            fields={
                "transported_scalar": self.s.copy(),
                "scalar_source": self._source_rate(self.t),
                "active_mask": np.ones(self.shape, dtype=bool),
                "velocity_field": self.u.copy(),
                "boundary_normal": self.b_normal.copy(),
                "boundary_target_velocity": self.b_target.copy(),
                "boundary_velocity": np.asarray(bvel, dtype=float).reshape(len(self.b_pos), 3),
            },
            scalars={
                "characteristic_length": self.char_length,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": float(max(1.0, umax)),
                "contact_length_scale": self.grid_spacing,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "substeps": int(self._substeps),
            "max_speed": float(np.sqrt((self.u ** 2).sum(axis=-1)).max()),
            "max_scalar": float(self.s.max()),
        }
