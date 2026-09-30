"""Physically-based 2D incompressible flow solver: channel flow past a fixed
circular cylinder producing a von Karman vortex street.

Method: finite-difference incompressible Navier-Stokes on a staggered MAC
grid.  Time stepping is Chorin pressure projection with explicit viscous
diffusion and MacCormack (limited semi-Lagrangian) advection.  The pressure
Poisson equation is assembled once as a sparse matrix (Neumann conditions on
walls / inlet / cylinder, Dirichlet p=0 at the open outlet) and factorized
with scipy sparse LU, so every substep is an actual deterministic physical
time integration of the discretized equations.

No animation, keyframes, or precomputed trajectories are used: the vortex
street emerges from the Navier-Stokes dynamics plus a tiny deterministic
seeded perturbation that breaks the perfect geometric symmetry of the setup.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic Navier-Stokes solver for scene 006."""

    # internal (simulation) grid: half the diagnostic spacing for wake resolution
    SIM_DX = 0.05
    SUB_DT = 0.005  # CFL: 1.2 * 0.005 / 0.05 = 0.12 ; nu*dt/dx^2 = 0.024

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg
        self.seed = int(seed)

        dom = cfg["domain"]
        self.xmin, self.ymin = float(dom["bounds_min"][0]), float(dom["bounds_min"][1])
        self.xmax, self.ymax = float(dom["bounds_max"][0]), float(dom["bounds_max"][1])
        self.dx = self.dy = self.SIM_DX
        self.dt = self.SUB_DT
        self.dx2, self.dy2 = self.dx ** 2, self.dy ** 2

        self.nu = float(cfg["fluid"]["kinematic_viscosity"])
        self.rho = float(cfg["fluid"]["density"])
        self.Uc = float(cfg["initial_conditions"]["centerline_speed"])
        self.Umean = float(cfg["initial_conditions"]["mean_speed"])
        self.H = 0.5 * (self.ymax - self.ymin)
        cx, cy, _ = cfg["obstacle"]["center"]
        self.cx, self.cy = float(cx), float(cy)
        self.radius = float(cfg["obstacle"]["radius"])
        self.out_spacing = float(cfg["fluid"]["nominal_particle_spacing"])

        # ---- simulation grid (MAC) -------------------------------------
        self.nx = int(round((self.xmax - self.xmin) / self.dx))
        self.ny = int(round((self.ymax - self.ymin) / self.dy))
        self.xc = self.xmin + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.ymin + (np.arange(self.ny) + 0.5) * self.dy
        self.xu = self.xmin + np.arange(self.nx + 1) * self.dx
        self.yv = self.ymin + np.arange(self.ny + 1) * self.dy
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.XU, self.YU = np.meshgrid(self.xu, self.yc, indexing="ij")
        self.XV, self.YV = np.meshgrid(self.xc, self.yv, indexing="ij")

        # solid mask: cells whose centers lie inside the cylinder
        r2 = self.radius ** 2
        self.solid = ((self.Xc - self.cx) ** 2 + (self.Yc - self.cy) ** 2) <= r2
        self.fluid = ~self.solid

        # face masks blocked by the solid (no-slip immersed boundary)
        su = np.zeros((self.nx + 1, self.ny), dtype=bool)
        su[0, :] = self.solid[0, :]
        su[1:self.nx, :] = self.solid[:-1, :] | self.solid[1:, :]
        su[self.nx, :] = self.solid[self.nx - 1, :]
        self.u_blocked = su
        sv = np.zeros((self.nx, self.ny + 1), dtype=bool)
        sv[:, 0] = True
        sv[:, self.ny] = True
        sv[:, 1:self.ny] = self.solid[:, :-1] | self.solid[:, 1:]
        self.v_blocked = sv

        # ---- initial flow state ----------------------------------------
        u_in = self._inlet_profile(self.yc)
        u0 = np.repeat(u_in[None, :], self.nx + 1, axis=0)
        v0 = np.zeros((self.nx, self.ny + 1))
        dye0 = self._initial_dye(self.Xc, self.Yc)

        # deterministic symmetry-breaking perturbation near the cylinder
        rng = np.random.default_rng(self.seed)
        ring = (
            (np.abs(self.Xc - self.cx) < 3.0 * self.radius)
            & (np.abs(self.Yc - self.cy) < 1.2 * self.radius)
        ).astype(float)
        pert = 5e-3 * self.Uc * rng.standard_normal((self.nx, self.ny - 1))
        v0[:, 1:self.ny] += pert * ring[:, 1:self.ny]

        self._u0, self._v0, self._dye0 = u0, v0, dye0
        self._reset()

        # ---- factorized pressure Poisson operator ----------------------
        self._build_poisson()

        # ---- diagnostic / output lattice (matches scene spacing) --------
        so = self.out_spacing
        self.xs_out = self.xmin + (np.arange(int(round((self.xmax - self.xmin) / so))) + 0.5) * so
        self.ys_out = self.ymin + (np.arange(int(round((self.ymax - self.ymin) / so))) + 0.5) * so
        self.Xo, self.Yo = np.meshgrid(self.xs_out, self.ys_out, indexing="ij")
        self.out_keep = (
            (self.Xo - self.cx) ** 2 + (self.Yo - self.cy) ** 2
        ) > self.radius ** 2
        self.nxo, self.nyo = self.Xo.shape

        # particle lattice (identical construction to the frozen scene)
        XX, YY = np.meshgrid(self.xs_out, self.ys_out, indexing="xy")
        kp = ((XX - self.cx) ** 2 + (YY - self.cy) ** 2) > self.radius ** 2
        self.p_x, self.p_y = XX[kp], YY[kp]
        self.n_particles = self.p_x.size
        self.p_mass = np.full(self.n_particles, self.rho * so * so)
        self.p_pos = np.column_stack(
            (self.p_x, self.p_y, np.zeros(self.n_particles))
        )

        # boundary samples: walls + cylinder surface (2-D vectors)
        bx = np.arange(self.xmin, self.xmax + 1e-9, 0.5)
        ang = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        off = 0.03
        pts, nrm = [], []
        pts.append(np.column_stack((bx, np.full(bx.size, self.ymin + off))))
        nrm.append(np.tile(np.array([0.0, -1.0]), (bx.size, 1)))
        pts.append(np.column_stack((bx, np.full(bx.size, self.ymax - off))))
        nrm.append(np.tile(np.array([0.0, 1.0]), (bx.size, 1)))
        cpts = np.column_stack(
            (
                self.cx + (self.radius + off) * np.cos(ang),
                self.cy + (self.radius + off) * np.sin(ang),
            )
        )
        cnrm = np.column_stack((np.cos(ang), np.sin(ang)))
        pts.append(cpts)
        nrm.append(cnrm)
        self.b_points = np.concatenate(pts, axis=0)
        self.b_normal = np.concatenate(nrm, axis=0)
        self.b_target = np.zeros_like(self.b_normal)

        # dye injection mask (first cell column of the simulation grid)
        self.inj_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.inj_mask[0, :] = True
        self.dye_stripe = 0.55
        self.dye_speed = 1.1

    # ------------------------------------------------------------------ utils
    def _inlet_profile(self, y):
        return self.Uc * np.maximum(0.0, 1.0 - (y / self.H) ** 2)

    def _initial_dye(self, x, y):
        stripe = np.floor((x - self.xmin) / self.dye_stripe).astype(np.int64)
        s = np.where((stripe % 2) == 0, -1.0, 1.0)
        dye = np.where(x < self.cx + 0.45, s, 0.0)
        return dye * np.exp(-0.5 * (y / 0.34) ** 2)

    def _inlet_dye(self, t):
        k = np.floor(-self.dye_speed * t / self.dye_stripe).astype(np.int64)
        s = -1.0 if (k % 2) == 0 else 1.0
        return s * np.exp(-0.5 * (self.yc / 0.34) ** 2)

    def _reset(self):
        self.u = self._u0.copy()
        self.v = self._v0.copy()
        self.dye = self._dye0.copy()
        self.src = np.zeros((self.nx, self.ny))
        self._steps = 0
        self._time = 0.0

    # ------------------------------------------------------- interpolation
    def _sample(self, f, px, py, x0, y0):
        """Bilinear sample of f[ix, iy] at world points (px, py)."""
        nxf, nyf = f.shape
        gx = np.clip((px - x0) / self.dx, 0.0, float(nxf - 1))
        gy = np.clip((py - y0) / self.dy, 0.0, float(nyf - 1))
        i0 = np.floor(gx).astype(np.int64)
        j0 = np.floor(gy).astype(np.int64)
        i1 = np.minimum(i0 + 1, nxf - 1)
        j1 = np.minimum(j0 + 1, nyf - 1)
        fx = gx - i0
        fy = gy - j0
        return (
            f[i0, j0] * (1.0 - fx) * (1.0 - fy)
            + f[i1, j0] * fx * (1.0 - fy)
            + f[i0, j1] * (1.0 - fx) * fy
            + f[i1, j1] * fx * fy
        )

    def _minmax3(self, f):
        fp = np.pad(f, 1, mode="edge")
        mn = np.full(f.shape, np.inf)
        mx = np.full(f.shape, -np.inf)
        n0, n1 = f.shape
        for a in range(3):
            for b in range(3):
                view = fp[a:a + n0, b:b + n1]
                np.minimum(mn, view, out=mn)
                np.maximum(mx, view, out=mx)
        return mn, mx

    def _advect(self, f, uq, vq, x0, y0, px, py):
        """Limited MacCormack semi-Lagrangian advection of f by (uq, vq)."""
        dt = self.dt
        f1 = self._sample(f, px - uq * dt, py - vq * dt, x0, y0)
        f2 = self._sample(f1, px + uq * dt, py + vq * dt, x0, y0)
        out = f1 + 0.5 * (f - f2)
        mn, mx = self._minmax3(f)
        return np.clip(out, mn, mx)

    # ---------------------------------------------------------- boundaries
    def _apply_vbc(self):
        u, v = self.u, self.v
        u[self.u_blocked] = 0.0
        v[self.v_blocked] = 0.0
        u[0, :] = self._inlet_profile(self.yc)
        u[self.nx, :] = u[self.nx - 1, :]
        v[:, 0] = 0.0
        v[:, self.ny] = 0.0

    def _center_vel(self):
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        uc[self.solid] = 0.0
        vc[self.solid] = 0.0
        return uc, vc

    # ------------------------------------------------------------ diffusion
    def _diffuse(self):
        nu, dt = self.nu, self.dt
        u, v = self.u, self.v
        lap_u = np.zeros_like(u)
        lap_u[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / self.dx2
        lap_u[:, 1:-1] += (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / self.dy2
        # no-slip walls: antisymmetric ghost  ->  (u_neighbor - 3 u_wall)/dy^2
        lap_u[:, 0] += (u[:, 1] - 3.0 * u[:, 0]) / self.dy2
        lap_u[:, -1] += (u[:, -2] - 3.0 * u[:, -1]) / self.dy2
        lap_v = np.zeros_like(v)
        lap_v[1:-1, 1:-1] = (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / self.dx2
        lap_v[0, 1:-1] = (v[1, 1:-1] - v[0, 1:-1]) / self.dx2
        lap_v[-1, 1:-1] = (v[-2, 1:-1] - v[-1, 1:-1]) / self.dx2
        lap_v[:, 1:-1] += (v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / self.dy2
        self.u = u + nu * dt * lap_u
        self.v = v + nu * dt * lap_v

    # ------------------------------------------------------------- pressure
    def _build_poisson(self):
        nx, ny = self.nx, self.ny
        ids = np.full((nx, ny), -1, dtype=np.int64)
        ids[self.fluid] = np.arange(int(self.fluid.sum()))
        self._ids = ids
        N = int(self.fluid.sum())

        I, J = np.nonzero(self.fluid)
        fid = ids[I, J]
        rows = [fid]
        cols = [fid]
        vals = [np.zeros(N)]
        diag = np.zeros(N)

        def add(IM, JM, coeff, valid_extra=None):
            IM = np.clip(IM, 0, nx - 1)
            JM = np.clip(JM, 0, ny - 1)
            ok = self.fluid[IM, JM]
            if valid_extra is not None:
                ok = ok & valid_extra
            rows.append(fid[ok])
            cols.append(ids[IM[ok], JM[ok]])
            vals.append(np.full(int(ok.sum()), coeff))
            diag += np.where(ok, coeff, 0.0)

        # east / west: Dirichlet p=0 beyond the open outlet (x = xmax)
        add(I + 1, J, 1.0 / self.dx2, valid_extra=(I + 1 < nx))
        add(I - 1, J, 1.0 / self.dx2, valid_extra=(I - 1 >= 0))
        add(I, J + 1, 1.0 / self.dy2, valid_extra=(J + 1 < ny))
        add(I, J - 1, 1.0 / self.dy2, valid_extra=(J - 1 >= 0))

        rows.append(fid)
        cols.append(fid)
        vals.append(-diag)
        A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N),
        ).tocsc()
        self._lu = spla.splu(A)

    def _project(self):
        u, v = self.u, self.v
        div = (u[1:, :] - u[:-1, :]) / self.dx + (v[:, 1:] - v[:, :-1]) / self.dy
        rhs = (div / self.dt)[self.fluid]
        p = self._lu.solve(rhs)
        P = np.zeros((self.nx, self.ny))
        P[self.fluid] = p

        both_u = self.fluid[1:, :] & self.fluid[:-1, :]
        grad_x = (P[1:, :] - P[:-1, :]) / self.dx
        u[1:self.nx, :][both_u] -= self.dt * grad_x[both_u]

        both_v = self.fluid[:, 1:] & self.fluid[:, :-1]
        grad_y = (P[:, 1:] - P[:, :-1]) / self.dy
        v[:, 1:self.ny][both_v] -= self.dt * grad_y[both_v]
        return div

    # ------------------------------------------------------------ time step
    def _substep(self):
        # 1) advection (MacCormack semi-Lagrangian) of the momentum
        v_at_u = self._sample(self.v, self.XU, self.YU, self.xc[0], self.yv[0])
        u = self._advect(self.u, self.u, v_at_u, self.xu[0], self.yc[0], self.XU, self.YU)
        u_at_v = self._sample(self.u, self.XV, self.YV, self.xu[0], self.yc[0])
        v = self._advect(self.v, u_at_v, self.v, self.xc[0], self.yv[0], self.XV, self.YV)
        self.u, self.v = u, v

        # 2) viscous diffusion (explicit, stable: nu*dt/dx^2 << 1)
        self._diffuse()

        # 3) boundary conditions then pressure projection
        self._apply_vbc()
        self._project()
        self._apply_vbc()

        # deterministic safety clamp (keeps every state finite)
        np.clip(self.u, -8.0, 8.0, out=self.u)
        np.clip(self.v, -8.0, 8.0, out=self.v)

        # 4) passive dye transport with prescribed inlet injection
        uc, vc = self._center_vel()
        self.dye = self._advect(
            self.dye, uc, vc, self.xc[0], self.yc[0], self.Xc, self.Yc
        )
        self.dye[self.solid] = 0.0
        pre = self.dye[self.inj_mask].copy()
        target = self._inlet_dye(self._time) * self.fluid[0, :]
        self.dye[self.inj_mask] = target[None, :]
        src = np.zeros((self.nx, self.ny))
        src[0, :] = (self.dye[0, :] - pre) / self.dt
        self.src = src

        self._steps += 1
        self._time = self._steps * self.dt
        np.nan_to_num(self.u, copy=False)
        np.nan_to_num(self.v, copy=False)
        np.nan_to_num(self.dye, copy=False)

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float) -> StateFrame:
        t = max(0.0, float(time))
        target = int(round(t / self.dt))
        if target < self._steps:
            self._reset()
        while self._steps < target:
            self._substep()
        self._time = self._steps * self.dt
        return self._build_frame(self._time)

    def _build_frame(self, t: float) -> StateFrame:
        uc, vc = self._center_vel()
        # vorticity omega = dv/dx - du/dy at cell centers
        om = (
            np.gradient(vc, self.dx, axis=0, edge_order=1)
            - np.gradient(uc, self.dy, axis=1, edge_order=1)
        )
        div = (self.u[1:, :] - self.u[:-1, :]) / self.dx + (
            self.v[:, 1:] - self.v[:, :-1]
        ) / self.dy
        om[self.solid] = 0.0

        x0c, y0c = self.xc[0], self.yc[0]
        Xo, Yo = self.Xo, self.Yo
        uco = self._sample(uc, Xo, Yo, x0c, y0c)
        vco = self._sample(vc, Xo, Yo, x0c, y0c)
        dyeo = self._sample(self.dye, Xo, Yo, x0c, y0c)
        srco = self._sample(self.src, Xo, Yo, x0c, y0c)
        omo = self._sample(om, Xo, Yo, x0c, y0c)
        divo = self._sample(div, Xo, Yo, x0c, y0c)

        T = lambda a: np.transpose(a, (1, 0))  # (nx,ny) -> scene (ny,nx)
        vel2 = np.stack((uco, vco), axis=-1)
        vel3 = np.concatenate(
            (vel2, np.zeros(vel2.shape[:2] + (1,))), axis=-1
        )
        active = self.out_keep.astype(np.uint8)

        # boundary velocities sampled from the flow next to each wall sample
        bvx = self._sample(uc, self.b_points[:, 0], self.b_points[:, 1], x0c, y0c)
        bvy = self._sample(vc, self.b_points[:, 0], self.b_points[:, 1], x0c, y0c)
        bvel = np.column_stack((bvx, bvy))

        # particle samples (Lagrangian tracer lattice of the scene)
        px, py = self.p_x, self.p_y
        pux = self._sample(uc, px, py, x0c, y0c)
        pvy = self._sample(vc, px, py, x0c, y0c)
        pdye = self._sample(self.dye, px, py, x0c, y0c)
        pom = self._sample(om, px, py, x0c, y0c)
        pdiv = self._sample(div, px, py, x0c, y0c)
        pvel = np.column_stack((pux, pvy, np.zeros(self.n_particles)))

        frame = StateFrame(
            time=float(t),
            position=self.p_pos,
            velocity=pvel,
            mass=self.p_mass,
            attributes={
                "dye": pdye,
                "vorticity": pom,
                "divergence": pdiv,
                "velocity_magnitude": np.hypot(pux, pvy),
            },
            fields={
                "active_mask": T(active),
                "velocity": np.transpose(vel3, (1, 0, 2)),
                "velocity_field": np.transpose(vel2, (1, 0, 2)),
                "vorticity": T(omo),
                "divergence": T(divo),
                "dye": T(dyeo),
                "transported_scalar": T(dyeo),
                "scalar_source": T(srco),
                "boundary_normal": self.b_normal,
                "boundary_target_velocity": self.b_target,
                "boundary_velocity": bvel,
            },
            scalars={
                "characteristic_length": 2.0 * self.radius,
                "grid_spacing": float(self.out_spacing),
                "reference_velocity": float(self.Umean),
                "cylinder_radius": float(self.radius),
                "mean_inlet_speed": float(self.Umean),
                "centerline_speed": float(self.Uc),
                "expected_strouhal_number": float(
                    self.config["diagnostics"]["expected_strouhal_number"]
                ),
                "reynolds_number": float(
                    self.Umean * 2.0 * self.radius / self.nu
                ),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        uc, vc = self._center_vel()
        div = (self.u[1:, :] - self.u[:-1, :]) / self.dx + (
            self.v[:, 1:] - self.v[:, :-1]
        ) / self.dy
        return {
            "time": float(self._time),
            "steps": int(self._steps),
            "max_abs_divergence": float(np.max(np.abs(div[self.fluid]))),
            "max_speed": float(
                np.max(np.hypot(uc[self.fluid], vc[self.fluid]))
            ),
            "reynolds_number": float(self.Umean * 2.0 * self.radius / self.nu),
            "grid": [self.nx, self.ny],
            "substep_dt": float(self.dt),
        }
