"""Self-contained 2D incompressible solver for the thin-plate two-chamber jet.

Physics
-------
A staggered (MAC) finite-volume discretization of the incompressible
Navier-Stokes equations is advanced inside the lower chamber of the frozen
domain::

    du/dt + (u . grad) u = -grad(p)/rho + nu * lap(u),      div(u) = 0

Each substep performs semi-Lagrangian advection of momentum, explicit viscous
diffusion, a nozzle momentum source at the frozen inlet, then an exact pressure
projection (sparse LU factorization of the Neumann Poisson operator augmented
with a mean-pressure gauge row).

The thin plate at z = 1.80 is a solid, impermeable, stationary row of cells:
every normal face velocity on its surfaces vanishes and no pressure coupling
crosses it, so the upper chamber stays quiescent and receives no injected
tracer. Injected and ambient partial densities are transported in conservative
flux form (upwind) with the projected divergence-free velocity, which makes the
discrete mass balance exact up to the prescribed nozzle source.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# Frozen scene constants (config.yaml / scene_geometry.yaml)
# ---------------------------------------------------------------------------
NX, NY, NZ = 121, 3, 73           # diagnostic node grid (scene.py)
X_MIN, X_MAX = -3.0, 3.0
Z_MIN, Z_MAX = 0.0, 3.6
Y_MIN, Y_MAX = -0.18, 0.18
DEPTH = 0.36                      # shallow extrusion depth
H = 0.05                          # uniform grid spacing
PLATE_Z = 1.80                    # separating plate centre height
UPPER_THRESHOLD = 1.85
INLET_HALF_WIDTH = 0.21           # jet_inlet half extent in x
INLET_WIDTH = 0.42
JET_SPEED = 2.4                   # prescribed inlet velocity (m/s)
JET_DENSITY = 1.18
AMBIENT_DENSITY = 1.0
AMBIENT_PRESSURE = 1.0
RHO_REF = 1.0                     # dynamic density used by the projection
NU = 1.0e-3                       # kinematic viscosity (m^2/s)
CORE_HEIGHT = 0.45                # prescribed initial inlet core height
EMIT_HEIGHT = 0.135               # tracer clamp height (3 cell rows)
CFL = 0.40
DT_MAX = 5.0e-3

def _taper(x):
    """Frozen inlet profile used by the scene (smooth, compact support)."""
    return np.clip(1.0 - (np.abs(x) / 0.24) ** 4, 0.0, 1.0)


def _bilerp(field, gx, gz):
    """Bilinear sample of ``field`` at fractional index coordinates."""
    nx, nz = field.shape
    gx = np.clip(gx, 0.0, nx - 1.0)
    gz = np.clip(gz, 0.0, nz - 1.0)
    i0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
    k0 = np.clip(np.floor(gz).astype(np.int64), 0, nz - 2)
    tx = gx - i0
    tz = gz - k0
    i1 = i0 + 1
    k1 = k0 + 1
    f00 = field[i0, k0]
    f10 = field[i1, k0]
    f01 = field[i0, k1]
    f11 = field[i1, k1]
    return (
        f00 * (1.0 - tx) * (1.0 - tz)
        + f10 * tx * (1.0 - tz)
        + f01 * (1.0 - tx) * tz
        + f11 * tx * tz
    )


def _neumann_laplacian_1d(n):
    """1D Laplacian stencil (times h^2) with homogeneous Neumann ends."""
    main = np.full(n, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    off = np.ones(n - 1)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


class BenchmarkSolver:
    """MAC projection solver for the sealed thin-plate two-chamber jet."""

    # -- setup ------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Cell-centred grid coincident with the frozen diagnostic node grid.
        self.xc = np.linspace(X_MIN, X_MAX, NX)
        self.zc = np.linspace(Z_MIN, Z_MAX, NZ)
        self.yc = np.linspace(Y_MIN, Y_MAX, NY)

        # Solid plate row: the single cell row whose centre is the plate.
        self.jp = int(np.argmin(np.abs(self.zc - PLATE_Z)))
        self.nzl = self.jp                       # lower-chamber fluid rows
        self.nzu = NZ - self.jp - 1              # upper-chamber fluid rows

        # MAC velocity on the lower chamber: u on x-faces, w on z-faces.
        self.u = np.zeros((NX + 1, self.nzl))
        self.w = np.zeros((NX, self.nzl + 1))
        self.p = np.zeros((NX, self.nzl))        # kinematic pressure

        # Conserved partial densities (injected tracer / ambient gas).
        self.q = np.zeros((NX, NZ))
        self.a = np.full((NX, NZ), AMBIENT_DENSITY)

        xx2 = self.xc[:, None]
        zz2 = self.zc[None, :]
        core = (np.abs(xx2) <= INLET_HALF_WIDTH) & (zz2 <= CORE_HEIGHT)
        self.q[core] = JET_DENSITY
        self.a[core] = 0.0

        # Nozzle momentum-source block and tracer emitter (frozen inlet).
        xl = self.xc[:, None]
        zl = self.zc[None, : self.nzl]
        self.src_cell = (np.abs(xl) <= INLET_HALF_WIDTH) & (zl <= CORE_HEIGHT)
        self.emit_cell = (np.abs(xx2) <= INLET_HALF_WIDTH) & (zz2 <= EMIT_HEIGHT)
        self.emit_cell[:, self.nzl:] = False

        # Prescribed w on the interior z-faces of the nozzle block.
        self.w_src_mask = np.zeros_like(self.w, dtype=bool)
        col = np.any(self.src_cell, axis=1)
        jtop = int(np.max(np.nonzero(np.any(self.src_cell, axis=0))[0])) + 1
        self.w_src_mask[col, 1:jtop] = True
        self.w_src_value = np.zeros_like(self.w)
        self.w_src_value[:, :] = (JET_SPEED * _taper(self.xc))[:, None]

        # Face coordinates used by the semi-Lagrangian back-trace.
        xu = X_MIN - 0.5 * H + H * np.arange(NX + 1)
        zw = Z_MIN - 0.5 * H + H * np.arange(self.nzl + 1)
        self.Xu, self.Zu = np.meshgrid(xu, self.zc[: self.nzl], indexing="ij")
        self.Xw, self.Zw = np.meshgrid(self.xc, zw, indexing="ij")

        self._build_poisson()
        self._build_nodes()
        self._build_boundary_samples()

        self.cell_volume = H * H * DEPTH
        self.node_volume = self.cell_volume / NY
        self.cum_source = np.zeros(2, dtype=np.float64)
        self.initial_mass = np.array(
            [
                float(np.sum(self.q)) * self.cell_volume,
                float(np.sum(self.a)) * self.cell_volume,
            ],
            dtype=np.float64,
        )

        self.t = 0.0
        self.n_substeps = 0
        self._max_speed = 0.0
        self._max_upper_mass = 0.0
        self._max_div = 0.0

        # Make the reported initial state discretely divergence free.
        self._apply_source()
        self._apply_velocity_bc()
        self._project(DT_MAX)
        self._apply_velocity_bc()

    def _build_poisson(self) -> None:
        n = NX * self.nzl
        lap = (
            sp.kron(_neumann_laplacian_1d(NX), sp.identity(self.nzl, format="csr"))
            + sp.kron(sp.identity(NX, format="csr"), _neumann_laplacian_1d(self.nzl))
        ) / (H * H)
        ones = np.ones((n, 1))
        top = sp.hstack([lap, sp.csr_matrix(ones)])
        bottom = sp.hstack([sp.csr_matrix(ones.T), sp.csr_matrix((1, 1))])
        self._poisson = spla.splu(sp.vstack([top, bottom]).tocsc())
        self._n_press = n

    def _build_nodes(self) -> None:
        xx, yy, zz = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self.node_position = np.column_stack(
            (xx.ravel(), yy.ravel(), zz.ravel())
        ).astype(np.float32)
        self.node_z = zz.ravel()
        self.n_nodes = self.node_position.shape[0]

    def _build_boundary_samples(self) -> None:
        """Stationary impermeable wall / plate samples (no-slip, zero target)."""
        normals = [
            np.tile(np.array([0.0, 0.0, 1.0]), (NX, 1)),        # bottom wall
            np.tile(np.array([0.0, 0.0, -1.0]), (NX, 1)),       # plate underside
            np.tile(np.array([0.0, 0.0, 1.0]), (NX, 1)),        # plate topside
            np.tile(np.array([1.0, 0.0, 0.0]), (self.nzl, 1)),  # left wall
            np.tile(np.array([-1.0, 0.0, 0.0]), (self.nzl, 1)),  # right wall
        ]
        self.boundary_normal = np.vstack(normals).astype(np.float64)
        self.n_boundary = self.boundary_normal.shape[0]
        self.boundary_target = np.zeros((self.n_boundary, 3), dtype=np.float64)

    # -- discrete operators ----------------------------------------------
    def _apply_velocity_bc(self) -> None:
        """Impermeable outer walls and the impermeable stationary thin plate."""
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[:, 0] = 0.0
        self.w[:, -1] = 0.0

    def _apply_source(self) -> None:
        """Prescribed nozzle momentum source at the frozen jet inlet."""
        self.w[self.w_src_mask] = self.w_src_value[self.w_src_mask]

    def _sample_u(self, x, z):
        return _bilerp(self.u, (x - (X_MIN - 0.5 * H)) / H, (z - Z_MIN) / H)

    def _sample_w(self, x, z):
        return _bilerp(self.w, (x - X_MIN) / H, (z - (Z_MIN - 0.5 * H)) / H)

    def _advect_velocity(self, dt: float) -> None:
        vu = self._sample_w(self.Xu, self.Zu)
        u_new = self._sample_u(self.Xu - dt * self.u, self.Zu - dt * vu)
        uw = self._sample_u(self.Xw, self.Zw)
        w_new = self._sample_w(self.Xw - dt * uw, self.Zw - dt * self.w)
        self.u = u_new
        self.w = w_new

    def _diffuse(self, dt: float) -> None:
        c = NU * dt / (H * H)
        u = self.u
        # z-ghosts implement no-slip at the bottom wall and the plate underside.
        u_dn = np.concatenate((-u[:, :1], u[:, :-1]), axis=1)
        u_up = np.concatenate((u[:, 1:], -u[:, -1:]), axis=1)
        lap_u = u_up - 2.0 * u + u_dn
        lap_u[1:-1, :] += u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]
        w = self.w
        w_lf = np.concatenate((-w[:1, :], w[:-1, :]), axis=0)
        w_rt = np.concatenate((w[1:, :], -w[-1:, :]), axis=0)
        lap_w = w_rt - 2.0 * w + w_lf
        lap_w[:, 1:-1] += w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]
        self.u = u + c * lap_u
        self.w = w + c * lap_w
        self._apply_velocity_bc()

    def _project(self, dt: float) -> None:
        """Exact MAC pressure projection with homogeneous Neumann walls."""
        div = (self.u[1:, :] - self.u[:-1, :]) / H + (
            self.w[:, 1:] - self.w[:, :-1]
        ) / H
        rhs = np.empty(self._n_press + 1, dtype=np.float64)
        rhs[:-1] = (div / dt).ravel()
        rhs[-1] = 0.0
        sol = self._poisson.solve(rhs)
        self.p = sol[: self._n_press].reshape(NX, self.nzl)
        self.u[1:-1, :] -= dt * (self.p[1:, :] - self.p[:-1, :]) / H
        self.w[:, 1:-1] -= dt * (self.p[:, 1:] - self.p[:, :-1]) / H

    def _advect_scalars(self, dt: float) -> None:
        """Conservative upwind transport of the injected/ambient partial densities."""
        nzl = self.nzl
        u = self.u
        w = self.w
        tq0 = float(np.sum(self.q))
        ta0 = float(np.sum(self.a))
        for name in ("q", "a"):
            full = getattr(self, name)
            s = full[:, :nzl]
            fx = np.zeros((NX + 1, nzl))
            ui = u[1:-1, :]
            fx[1:-1, :] = np.where(ui > 0.0, ui * s[:-1, :], ui * s[1:, :])
            fz = np.zeros((NX, nzl + 1))
            wi = w[:, 1:-1]
            fz[:, 1:-1] = np.where(wi > 0.0, wi * s[:, :-1], wi * s[:, 1:])
            s_new = s - (dt / H) * (
                (fx[1:, :] - fx[:-1, :]) + (fz[:, 1:] - fz[:, :-1])
            )
            full[:, :nzl] = np.maximum(s_new, 0.0)

        # Prescribed inlet composition: pure injected gas inside the emitter.
        self.q[self.emit_cell] = JET_DENSITY
        self.a[self.emit_cell] = 0.0
        # Upwind transport is exactly conservative with zero flux through every
        # wall and the plate, so the whole change of total material mass over a
        # substep is the prescribed inlet source.
        self.cum_source[0] += (float(np.sum(self.q)) - tq0) * self.cell_volume
        self.cum_source[1] += (float(np.sum(self.a)) - ta0) * self.cell_volume

    def _substep(self, dt: float) -> None:
        self._advect_velocity(dt)
        self._apply_velocity_bc()
        self._diffuse(dt)
        self._apply_source()
        self._apply_velocity_bc()
        self._project(dt)
        self._apply_velocity_bc()
        self._advect_scalars(dt)
        div = (self.u[1:, :] - self.u[:-1, :]) / H + (
            self.w[:, 1:] - self.w[:, :-1]
        ) / H
        self._max_div = max(self._max_div, float(np.max(np.abs(div))))
        self.t += dt
        self.n_substeps += 1

    def _stable_dt(self, remaining: float) -> float:
        vmax = max(
            float(np.max(np.abs(self.u))),
            float(np.max(np.abs(self.w))),
            1.0e-6,
        )
        dt = min(DT_MAX, CFL * H / vmax)
        return max(min(dt, remaining), 1.0e-6)

    # -- driver -----------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while target - self.t > 1.0e-12 and guard < 2_000_000:
            self._substep(self._stable_dt(target - self.t))
            guard += 1
        self.t = target
        return self._frame(target)

    # -- output -----------------------------------------------------------
    def _cell_centred(self):
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        return uc, wc

    def _interaction_force(self, p_gauge, uc, wc):
        mu = RHO_REF * NU
        area = H * DEPTH
        inv = 1.0 / (0.5 * H)
        nzl = self.nzl

        # Thin plate: jet stagnation pressure from below, quiescent gas above.
        f_plate = np.array(
            [
                mu * float(np.sum(uc[:, nzl - 1])) * inv * area,
                0.0,
                float(np.sum(p_gauge[:, nzl - 1])) * area,
            ]
        )
        f_bottom = np.array(
            [
                mu * float(np.sum(uc[:, 0])) * inv * area,
                0.0,
                -float(np.sum(p_gauge[:, 0])) * area,
            ]
        )
        f_left = np.array(
            [
                -float(np.sum(p_gauge[0, :])) * area,
                0.0,
                mu * float(np.sum(wc[0, :])) * inv * area,
            ]
        )
        f_right = np.array(
            [
                float(np.sum(p_gauge[-1, :])) * area,
                0.0,
                mu * float(np.sum(wc[-1, :])) * inv * area,
            ]
        )
        return np.vstack(
            [
                f_plate, -f_plate,
                f_bottom, -f_bottom,
                f_left, -f_left,
                f_right, -f_right,
            ]
        ).astype(np.float64)

    def _active_mask(self):
        mask = np.zeros((NX, NZ), dtype=np.float64)
        mask[1:-1, 1:self.nzl - 1] = 1.0          # lower-chamber interior
        mask[1:-1, self.jp + 2:NZ - 1] = 1.0      # upper-chamber interior
        return np.repeat(mask[:, None, :], NY, axis=1)

    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        nzl = self.nzl
        uc, wc = self._cell_centred()

        # Gauge (dynamic) pressure of the lower chamber; the sealed upper
        # chamber stays exactly at ambient, giving the plate pressure jump.
        p_gauge = self.p - float(np.mean(self.p))
        pressure2 = np.full((NX, NZ), AMBIENT_PRESSURE)
        pressure2[:, :nzl] = AMBIENT_PRESSURE + RHO_REF * p_gauge

        contact = np.abs(self.xc) <= 0.55
        p_jump = float(
            np.mean(pressure2[contact, nzl - 1]) - np.mean(pressure2[contact, self.jp + 1])
        )

        # Composition and mixture density on the diagnostic grid.
        total2 = self.q + self.a
        frac2 = np.where(total2 > 1.0e-12, self.q / np.maximum(total2, 1.0e-12), 0.0)
        frac2 = np.clip(frac2, 0.0, 1.0)
        frac2[:, self.jp:] = 0.0   # solid plate row and sealed upper chamber

        velocity2 = np.zeros((NX, NZ, 3))
        velocity2[:, :nzl, 0] = uc
        velocity2[:, :nzl, 2] = wc

        vel3 = np.repeat(velocity2[:, None, :, :], NY, axis=1)
        frac3 = np.repeat(frac2[:, None, :], NY, axis=1)
        dens3 = np.repeat(total2[:, None, :], NY, axis=1)
        pres3 = np.repeat(pressure2[:, None, :], NY, axis=1)

        node_velocity = vel3.reshape(-1, 3)
        node_frac = frac3.ravel()
        node_q = np.repeat((self.q)[:, None, :], NY, axis=1).ravel() * self.node_volume
        node_a = np.repeat((self.a)[:, None, :], NY, axis=1).ravel() * self.node_volume
        node_mass = node_q + node_a

        upper = self.node_z > UPPER_THRESHOLD
        lower = self.node_z < PLATE_Z
        upper_mass = float(np.sum(node_q[upper]))
        lower_mass = float(np.sum(node_q[lower]))
        speed = float(np.max(np.linalg.norm(node_velocity, axis=1)))
        self._max_speed = max(self._max_speed, speed)
        self._max_upper_mass = max(self._max_upper_mass, upper_mass)

        forces = self._interaction_force(p_gauge, uc, wc)
        finite = float(
            np.isfinite(node_velocity).all()
            and np.isfinite(node_mass).all()
            and np.isfinite(vel3).all()
            and np.isfinite(pres3).all()
        )

        frame = StateFrame(
            time=float(time),
            position=self.node_position,
            velocity=node_velocity.astype(np.float32),
            mass=node_mass.astype(np.float64),
            attributes={
                "mass_quantity": np.column_stack((node_q, node_a)).astype(np.float64),
                "injected_fraction": node_frac.astype(np.float32),
            },
            fields={
                "injected_fraction": frac3.astype(np.float32),
                "velocity_field": vel3.astype(np.float32),
                "density_field": dens3.astype(np.float32),
                "pressure": pres3.astype(np.float32),
                "density": dens3.astype(np.float32),
                "velocity": vel3.astype(np.float32),
                "active_mask": self._active_mask(),
                "boundary_normal": self.boundary_normal,
                "boundary_target_velocity": self.boundary_target,
                "boundary_velocity": np.zeros((self.n_boundary, 3), dtype=np.float64),
                "cumulative_mass_source": self.cum_source.copy(),
                "interaction_force": forces,
            },
            scalars={
                "characteristic_length": INLET_WIDTH,
                "grid_spacing": H,
                "reference_velocity": JET_SPEED,
                "upper_chamber_injected_mass": upper_mass,
                "lower_chamber_injected_mass": lower_mass,
                "pressure_jump_across_plate": p_jump,
                "maximum_speed": speed,
                "finite_state": finite,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC staggered incompressible projection (self-implemented)",
            "grid": [int(NX), int(self.nzl), int(self.nzu)],
            "grid_spacing": float(H),
            "substeps": int(self.n_substeps),
            "final_time": float(self.t),
            "max_speed": float(self._max_speed),
            "max_divergence_after_projection": float(self._max_div),
            "max_upper_chamber_injected_mass": float(self._max_upper_mass),
            "cumulative_mass_source": [float(v) for v in self.cum_source],
            "plate_row_index": int(self.jp),
        }
