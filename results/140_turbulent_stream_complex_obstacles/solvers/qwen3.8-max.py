"""Grid-based incompressible smoke solver for the turbulent stream scene.

Stable-fluids style semi-Lagrangian advection + exact pressure projection on a
collocated diagnostic grid, with solid obstacle masks, a prescribed inlet jet,
an open outlet, free-slip walls, vorticity confinement for turbulent eddies,
and an advected smoke scalar.  All required StateFrame fields are exported.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame

U_IN = 2.2                 # inlet jet speed [m/s]
CONF_EPS = 2.5             # vorticity confinement strength
SUB_DT = 1.0 / 60.0        # internal time step [s]
SCALAR_SOURCE_RATE = 3.0   # smoke injection rate [1/s]


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        base = scene.initial_state()
        self._base = base
        self._mass = np.asarray(base.mass, float).copy()
        self._body_id = np.asarray(base.body_id).copy()
        self._attributes = {k: np.asarray(v).copy() for k, v in base.attributes.items()}
        self._pos = np.asarray(base.position, float).copy()
        self._vel = np.zeros_like(self._pos)
        self._time = 0.0

        # ---- diagnostic grid -------------------------------------------
        shape = (24, 16, 28)
        lo = np.array([-3.0, -1.2, 0.0])
        hi = np.array([3.0, 1.2, 3.8])
        self.shape = shape
        self.lo, self.hi = lo, hi
        h = (hi - lo) / np.array(shape, float)
        self.h = h
        cx = lo[0] + (np.arange(shape[0]) + 0.5) * h[0]
        cy = lo[1] + (np.arange(shape[1]) + 0.5) * h[1]
        cz = lo[2] + (np.arange(shape[2]) + 0.5) * h[2]
        self._cx, self._cy, self._cz = cx, cy, cz
        X = cx[:, None, None] + np.zeros(shape)
        Y = cy[None, :, None] + np.zeros(shape)
        Z = cz[None, None, :] + np.zeros(shape)

        # solid obstacles (boxes from the frozen scene geometry)
        obstacles = [
            ((-0.65, 0.0, 1.25), (0.38, 0.38, 0.38)),
            ((0.65, 0.0, 2.15), (0.42, 0.42, 0.42)),
        ]
        solid = np.zeros(shape, dtype=bool)
        for ctr, he in obstacles:
            solid |= (
                (np.abs(X - ctr[0]) <= he[0])
                & (np.abs(Y - ctr[1]) <= he[1])
                & (np.abs(Z - ctr[2]) <= he[2])
            )
        self.solid = solid
        self.fluid = ~solid

        # inlet jet region (nozzle at x=-2.72, z=1.0, radius 0.4)
        self.inlet = (X < -2.45) & (np.abs(Y) < 0.40) & (np.abs(Z - 1.0) < 0.40)
        self.inlet &= self.fluid

        # scalar source lives on the inlet cells
        self.scalar_source = np.where(self.inlet, SCALAR_SOURCE_RATE, 0.0)

        # ---- sparse Poisson operator on fluid cells ---------------------
        n = int(self.fluid.sum())
        idx = -np.ones(shape, dtype=np.int64)
        idx[self.fluid] = np.arange(n)
        self._n = n
        a = 1.0 / h**2  # per-axis Laplacian coefficient

        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        for ax in range(3):
            for sign in (1, -1):
                nb = np.roll(idx, -sign, axis=ax)
                if ax == 0:
                    if sign == 1:
                        nb[-1] = -2   # outlet: Dirichlet ghost p = 0
                    else:
                        nb[0] = -1    # left wall / inlet: Neumann ghost
                elif ax == 1:
                    if sign == 1:
                        nb[:, -1] = -1
                    else:
                        nb[:, 0] = -1
                else:
                    if sign == 1:
                        nb[:, :, -1] = -1
                    else:
                        nb[:, :, 0] = -1
                own = idx[self.fluid]
                nbv = nb[self.fluid]
                active = nbv != -1   # fluid neighbor (-? >=0) or outlet (-2)
                diag[own[active]] -= a[ax]
                fl = nbv >= 0
                rows.append(own[fl])
                cols.append(nbv[fl])
                vals.append(np.full(int(fl.sum()), a[ax]))
        rows = np.concatenate(rows + [np.arange(n)])
        cols = np.concatenate(cols + [np.arange(n)])
        vals = np.concatenate(vals + [diag])
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self._lu = spla.splu(A.tocsc())

        # ---- boundary samples: faces between fluid and solid cells ------
        bcells, bnorms = [], []
        for ax in range(3):
            for sign in (1, -1):
                sl_c = [slice(None)] * 3
                sl_n = [slice(None)] * 3
                if sign == 1:
                    sl_c[ax] = slice(0, shape[ax] - 1)
                    sl_n[ax] = slice(1, shape[ax])
                else:
                    sl_c[ax] = slice(1, shape[ax])
                    sl_n[ax] = slice(0, shape[ax] - 1)
                sel = self.fluid[tuple(sl_c)] & self.solid[tuple(sl_n)]
                ii, jj, kk = np.nonzero(sel)
                cell = np.stack([ii, jj, kk], axis=1)
                if sign == -1:
                    cell[:, ax] += 1  # slice starts at 1
                nrm = np.zeros((len(ii), 3))
                # outward normal of the solid into the fluid cell
                nrm[:, ax] = -1.0 if sign == 1 else 1.0
                bcells.append(cell)
                bnorms.append(nrm)
        self._bcell = np.concatenate(bcells, axis=0)
        self.boundary_normal = np.concatenate(bnorms, axis=0)
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)

        # ---- flow state --------------------------------------------------
        self.u = np.zeros(shape + (3,))
        self.u[self.inlet] = (U_IN, 0.0, 0.0)
        self.scalar = np.zeros(shape)
        self._cum_source = np.zeros(len(self._pos))

        self.scalars = {
            "characteristic_length": 0.76,
            "grid_spacing": float(np.mean(h)),
            "reference_velocity": U_IN,
        }

    # ------------------------------------------------------------------
    def _coords(self, pts):
        return [(pts[:, d] - self.lo[d]) / self.h[d] - 0.5 for d in range(3)]

    def _sample(self, field, pts):
        return map_coordinates(field, self._coords(pts), order=1, mode="nearest")

    def _advect(self, field, dt):
        back = [
            self._grid_index[d] - dt * self.u[..., d] / self.h[d] for d in range(3)
        ]
        return map_coordinates(field, back, order=1, mode="nearest")

    @property
    def _grid_index(self):
        g = getattr(self, "_gidx", None)
        if g is None:
            g = np.meshgrid(
                *[np.arange(s, dtype=float) for s in self.shape], indexing="ij"
            )
            self._gidx = g
        return g

    def _divergence(self):
        h, shape = self.h, self.shape
        u = self.u.copy()
        u[self.solid] = 0.0
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        # x-faces: interior average, left face = inlet/wall, right = outflow
        fx = 0.5 * (ux[:-1] + ux[1:])
        fx[self.solid[:-1] | self.solid[1:]] = 0.0
        left = np.where(self.inlet[0], U_IN, 0.0)
        div = (np.concatenate([fx, ux[-1:]], 0)
               - np.concatenate([left[None], fx], 0)) / h[0]
        # y-faces: walls on both domain ends
        fy = 0.5 * (uy[:, :-1] + uy[:, 1:])
        fy[self.solid[:, :-1] | self.solid[:, 1:]] = 0.0
        zy = np.zeros((shape[0], 1, shape[2]))
        div += (np.concatenate([fy, zy], 1) - np.concatenate([zy, fy], 1)) / h[1]
        # z-faces: ground wall below, wall above
        fz = 0.5 * (uz[:, :, :-1] + uz[:, :, 1:])
        fz[self.solid[:, :, :-1] | self.solid[:, :, 1:]] = 0.0
        zz = np.zeros((shape[0], shape[1], 1))
        div += (np.concatenate([fz, zz], 2) - np.concatenate([zz, fz], 2)) / h[2]
        return div

    def _pressure_padded(self, p_full):
        px = np.empty((self.shape[0] + 2,) + self.shape[1:])
        px[1:-1] = p_full
        px[0] = p_full[0]      # inlet / left wall: Neumann
        px[-1] = 0.0           # outlet: Dirichlet p = 0
        py = np.empty((self.shape[0], self.shape[1] + 2, self.shape[2]))
        py[:, 1:-1] = p_full
        py[:, 0] = p_full[:, 0]
        py[:, -1] = p_full[:, -1]
        pz = np.empty((self.shape[0], self.shape[1], self.shape[2] + 2))
        pz[:, :, 1:-1] = p_full
        pz[:, :, 0] = p_full[:, :, 0]
        pz[:, :, -1] = p_full[:, :, -1]
        return px, py, pz

    def _project(self, dt):
        div = self._divergence()
        rhs = div[self.fluid] / dt
        p = self._lu.solve(rhs)
        p_full = np.zeros(self.shape)
        p_full[self.fluid] = p
        px, py, pz = self._pressure_padded(p_full)
        h = self.h
        self.u[..., 0] -= dt * (px[2:] - px[:-2]) / (2 * h[0])
        self.u[..., 1] -= dt * (py[:, 2:] - py[:, :-2]) / (2 * h[1])
        self.u[..., 2] -= dt * (pz[:, :, 2:] - pz[:, :, :-2]) / (2 * h[2])
        self._apply_bc()

    def _apply_bc(self):
        self.u[self.solid] = 0.0
        self.u[self.inlet] = (U_IN, 0.0, 0.0)
        self.u[self.inlet, 2] += 0.10 * U_IN * np.sin(6.0 * self._time)
        # free-slip walls: zero normal velocity on domain faces
        self.u[0, :, :, 0] = np.where(self.inlet[0], self.u[0, :, :, 0], 0.0)
        self.u[:, 0, :, 1] = 0.0
        self.u[:, -1, :, 1] = 0.0
        self.u[:, :, 0, 2] = 0.0
        self.u[:, :, -1, 2] = 0.0
        self.u[self.solid] = 0.0

    def _vorticity_confinement(self, dt):
        h = self.h
        u, v, w = self.u[..., 0], self.u[..., 1], self.u[..., 2]
        wx = np.gradient(w, h[1], axis=1) - np.gradient(v, h[2], axis=2)
        wy = np.gradient(u, h[2], axis=2) - np.gradient(w, h[0], axis=0)
        wz = np.gradient(v, h[0], axis=0) - np.gradient(u, h[1], axis=1)
        om = np.stack([wx, wy, wz], axis=-1)
        mag = np.linalg.norm(om, axis=-1)
        gmag = np.stack(np.gradient(mag, *h, edge_order=1), axis=-1)
        nrm = gmag / (np.linalg.norm(gmag, axis=-1, keepdims=True) + 1e-8)
        self.u += dt * CONF_EPS * float(np.mean(h)) * np.cross(nrm, om)
        self.u[self.solid] = 0.0

    def _step(self, dt):
        t = self._time
        # prescribed inlet with a weak deterministic vertical wobble
        self.u[self.inlet] = (U_IN, 0.0, 0.0)
        self.u[self.inlet, 2] += 0.10 * U_IN * np.sin(6.0 * t)

        for d in range(3):
            self.u[..., d] = self._advect(self.u[..., d], dt)
        self._apply_bc()
        self._vorticity_confinement(dt)
        self._project(dt)

        # smoke scalar: semi-Lagrangian advection + inlet source
        self.scalar = self._advect(self.scalar, dt)
        self.scalar += dt * self.scalar_source
        self.scalar = np.clip(self.scalar, 0.0, 5.0)
        self.scalar[self.solid] = 0.0

        # tracer particles ride the flow
        vel = np.stack([self._sample(self.u[..., d], self._pos) for d in range(3)], axis=1)
        self._pos = self._pos + dt * vel
        np.clip(self._pos, self.lo + 1e-3, self.hi - 1e-3, out=self._pos)
        self._vel = np.stack([self._sample(self.u[..., d], self._pos) for d in range(3)], axis=1)
        self._cum_source += dt * self._sample(self.scalar_source, self._pos)
        self._time = t + dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        while self._time < target - 1e-12:
            self._step(min(SUB_DT, target - self._time))
        bvel = np.zeros_like(self.boundary_normal)
        bc = self._bcell
        for d in range(3):
            bvel[:, d] = 0.5 * self.u[bc[:, 0], bc[:, 1], bc[:, 2], d]
        frame = StateFrame(
            self._time,
            self._pos.copy(),
            self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={k: v.copy() for k, v in self._attributes.items()},
            fields={
                "cumulative_mass_source": self._cum_source.copy(),
                "transported_scalar": self.scalar.copy(),
                "scalar_source": self.scalar_source.copy(),
                "active_mask": self.fluid.copy(),
                "velocity_field": self.u.copy(),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": bvel,
            },
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self._time,
            "inlet_speed": U_IN,
            "max_scalar": float(self.scalar.max()),
            "max_speed": float(np.linalg.norm(self.u, axis=-1).max()),
        }
