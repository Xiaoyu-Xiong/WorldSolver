"""Two-way coupled smoke / hanging cloth curtain solver.

Physics implemented here (no external simulator):

* Incompressible airflow: staggered (MAC) grid, semi-Lagrangian advection of
  velocity and of the passive smoke tracer, smoke buoyancy, pressure
  projection with red-black Gauss-Seidel on the Poisson system that respects
  voxelized moving solids and open outer boundaries.
* Cloth: mass-spring shell (structural, shear, bending springs) with velocity
  damping, strain limiting, two pinned top corners, symplectic Euler
  substepping.
* Two-way coupling: the fluid exerts momentum-flux (form drag + skin friction)
  loads sampled on both sides of every cloth triangle; the cloth is rasterized
  every fluid step into an impermeable moving Dirichlet-velocity solid so the
  evolving cloth re-routes the resolved flow.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

F32 = np.float32

RHO_AIR = 1.2           # kg/m^3
CD_NORMAL = 1.3         # bluff-plate form drag coefficient
CT_TANGENT = 0.06       # skin friction coefficient
BUOYANCY = 0.9          # m/s^2 per unit smoke density
GS_ITERATIONS = 24      # damped-Jacobi Poisson sweeps per projection
JACOBI_OMEGA = 0.9      # Jacobi relaxation weight
CLOTH_SUBSTEP = 5.0e-4  # s
K_STRUCT, K_SHEAR, K_BEND = 2500.0, 250.0, 25.0
C_STRUCT, C_SHEAR, C_BEND = 2.0, 0.5, 0.1
FLOOR_K, FLOOR_C = 4000.0, 12.0
AIR_DAMP = 0.15         # 1/s residual viscous air damping on the cloth


def _trilerp(flat, n0, n1, n2, gi, gj, gk):
    """Trilinear sample of a C-contiguous [n0,n1,n2] field at index coords."""
    gi = np.clip(gi, 0.0, n0 - 1.0)
    gj = np.clip(gj, 0.0, n1 - 1.0)
    gk = np.clip(gk, 0.0, n2 - 1.0)
    i0 = np.clip(np.floor(gi), 0, n0 - 2).astype(np.int64)
    j0 = np.clip(np.floor(gj), 0, n1 - 2).astype(np.int64)
    k0 = np.clip(np.floor(gk), 0, n2 - 2).astype(np.int64)
    fi = (gi - i0).astype(F32)
    fj = (gj - j0).astype(F32)
    fk = (gk - k0).astype(F32)
    s_i = n1 * n2
    base = i0 * s_i + j0 * n2 + k0
    c00 = flat[base] + (flat[base + 1] - flat[base]) * fk
    b1 = base + n2
    c01 = flat[b1] + (flat[b1 + 1] - flat[b1]) * fk
    b2 = base + s_i
    c10 = flat[b2] + (flat[b2 + 1] - flat[b2]) * fk
    b3 = base + s_i + n2
    c11 = flat[b3] + (flat[b3 + 1] - flat[b3]) * fk
    c0 = c00 + (c01 - c00) * fj
    c1 = c10 + (c11 - c10) * fj
    return c0 + (c1 - c0) * fi


def _cc_to_face(a, axis):
    """Average a cell-centred field onto staggered faces (edge replication)."""
    shape = list(a.shape)
    shape[axis] += 1
    out = np.empty(tuple(shape), F32)
    mid = [slice(None)] * 3
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    mid[axis] = slice(1, -1)
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    out[tuple(mid)] = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
    first_out = [slice(None)] * 3
    first_in = [slice(None)] * 3
    first_out[axis] = 0
    first_in[axis] = 0
    out[tuple(first_out)] = a[tuple(first_in)]
    last_out = [slice(None)] * 3
    last_in = [slice(None)] * 3
    last_out[axis] = -1
    last_in[axis] = -1
    out[tuple(last_out)] = a[tuple(last_in)]
    return out


class BenchmarkSolver:
    """Deterministic MAC-grid smoke solver two-way coupled to a mass-spring cloth."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        dom = cfg["domain"]
        self.nx, self.ny, self.nz = (int(v) for v in dom["grid_shape"])
        self.low = np.asarray(dom["bounds_min"], float)
        self.high = np.asarray(dom["bounds_max"], float)
        span = self.high - self.low
        self.h = float(span[0] / self.nx)
        nx, ny, nz = self.nx, self.ny, self.nz
        h = self.h

        self.xc = (self.low[0] + (np.arange(nx) + 0.5) * span[0] / nx).astype(F32)
        self.yc = (self.low[1] + (np.arange(ny) + 0.5) * span[1] / ny).astype(F32)
        self.zc = (self.low[2] + (np.arange(nz) + 0.5) * span[2] / nz).astype(F32)

        self.u = np.zeros((nx + 1, ny, nz), F32)
        self.v = np.zeros((nx, ny + 1, nz), F32)
        self.w = np.zeros((nx, ny, nz + 1), F32)
        self.q = np.zeros((nx, ny, nz), F32)
        self.qpad = np.zeros((nx + 2, ny + 2, nz + 2), F32)

        self._setup_static_solids()
        self._setup_cloth()
        self._setup_samples()

        init = scene.initial_state()
        self.density = np.ascontiguousarray(
            np.asarray(init.fields["smoke_density"], F32).reshape(nx, ny, nz)
        )
        self.sim_time = 0.0
        self.frame_index = 0
        self._steps = 0
        self.last_ramp = 0.0
        self.frame_forces = None
        self.cloth_mask = np.zeros((nx, ny, nz), bool)
        self.cloth_vel = np.zeros((nx, ny, nz, 3), F32)
        self._build_solid(0.0)
        self._update_cell_velocity()
    # ------------------------------------------------------------------ setup
    def _setup_static_solids(self) -> None:
        cfg = self.cfg
        jet = cfg["jet"]
        h = self.h
        x = self.xc[:, None, None]
        y = self.yc[None, :, None]
        z = self.zc[None, None, :]
        cy, cz = float(jet["center"][1]), float(jet["center"][2])
        cx = float(jet["center"][0])
        r2 = (y - cy) ** 2 + (z - cz) ** 2
        r_jet = float(jet["radius"])
        self.jet_mask = np.broadcast_to(
            (x <= cx) & (r2 <= r_jet ** 2), (self.nx, self.ny, self.nz)
        ).copy()
        # dark nozzle shell around the emitter opening
        nozzle = (x <= -2.70) & (r2 > r_jet ** 2) & (r2 <= 0.52 ** 2)
        # support bar: cylinder along y at (x=0, z=3.48), radius 0.055
        bar_r = max(0.055, 0.5 * h)
        bar = ((x ** 2 + (z - 3.48) ** 2) <= bar_r ** 2) & (np.abs(y) <= 1.45)
        # clamps
        clamp = (np.abs(x) <= 0.09) & (np.abs(z - 3.37) <= 0.13) & (
            (np.abs(y + 1.2) <= 0.1) | (np.abs(y - 1.2) <= 0.1)
        )
        self.static_solid = np.broadcast_to(
            nozzle | bar | clamp, (self.nx, self.ny, self.nz)
        ).copy()
        self.jet_speed = float(jet["speed"])
        self.jet_density = float(jet["smoke_density"])
        self.jet_start = float(jet["start_time"])
        self.jet_ramp = max(1e-6, float(jet["ramp_duration"]))
        self.jet_until = float(jet["active_until"])
        self.inlet_x_index = int(np.max(np.nonzero(self.jet_mask.any(axis=(1, 2)))[0]))

    def _setup_cloth(self) -> None:
        cfg = self.cfg
        state = self.scene.initial_state()
        meta = self.scene.metadata()
        self.faces = np.asarray(meta["topology"]["faces"], dtype=np.int64)
        self.uv = np.asarray(state.attributes["uv"], float)
        self.pos = np.asarray(state.position, float).copy()
        self.vel = np.asarray(state.velocity, float).copy()
        self.pmass = np.asarray(state.mass, float).copy()
        self.n_vert = self.pos.shape[0]
        self.inv_mass = 1.0 / self.pmass
        pinned_list = list(cfg["cloth"]["pinned_vertices"])
        self.pinned = np.zeros(self.n_vert, bool)
        self.pinned[np.asarray(pinned_list, dtype=np.int64)] = True
        self.inv_mass[self.pinned] = 0.0
        self.pin_pos = self.pos[self.pinned].copy()
        self.gravity = np.asarray(cfg["initial_conditions"]["gravity"], float)
        self.cloth_thickness = float(cfg["cloth"]["thickness"])

        ncol, nrow = (int(v) for v in cfg["cloth"]["grid_shape"])  # (y count, z count)
        idx = np.arange(nrow * ncol).reshape(nrow, ncol)
        pairs_s, pairs_h, pairs_b = [], [], []
        for j in range(nrow):
            for i in range(ncol):
                if i + 1 < ncol:
                    pairs_s.append((idx[j, i], idx[j, i + 1]))
                if j + 1 < nrow:
                    pairs_s.append((idx[j, i], idx[j + 1, i]))
                if i + 1 < ncol and j + 1 < nrow:
                    pairs_h.append((idx[j, i], idx[j + 1, i + 1]))
                    pairs_h.append((idx[j, i + 1], idx[j + 1, i]))
                if i + 2 < ncol:
                    pairs_b.append((idx[j, i], idx[j, i + 2]))
                if j + 2 < nrow:
                    pairs_b.append((idx[j, i], idx[j + 2, i]))
        self.spring_sets = []
        for pairs, k, c, tag in (
            (pairs_s, K_STRUCT, C_STRUCT, "struct"),
            (pairs_h, K_SHEAR, C_SHEAR, "shear"),
            (pairs_b, K_BEND, C_BEND, "bend"),
        ):
            arr = np.asarray(pairs, dtype=np.int64)
            rest = np.linalg.norm(self.pos[arr[:, 0]] - self.pos[arr[:, 1]], axis=1)
            self.spring_sets.append((arr[:, 0], arr[:, 1], rest, float(k), float(c), tag))
        s = self.spring_sets[0]
        self.struct_a, self.struct_b, self.struct_rest = s[0], s[1], s[2]

        # deterministic seeded symmetry break so folds are not perfectly planar
        jitter = self.rng.normal(0.0, 1.0, (self.n_vert, 3)) * np.array([2.0e-3, 0.0, 0.0])
        jitter[self.pinned] = 0.0
        self.pos += jitter

        # barycentric sample pattern for cloth voxelisation
        n_sub = 28
        rows = []
        for a in range(n_sub + 1):
            for b in range(n_sub + 1 - a):
                rows.append((a / n_sub, b / n_sub, 1.0 - a / n_sub - b / n_sub))
        self.bary = np.asarray(rows, F32)
        self.cloth_area = float(cfg["cloth"]["width"]) * float(cfg["cloth"]["height"])
        self.total_mass = float(self.pmass.sum())
        self.aero_offset = 2.5 * self.h

    def _setup_samples(self) -> None:
        n_ring = 12
        ang = np.arange(n_ring) * (2.0 * np.pi / n_ring)
        rr = 0.22
        cy = float(self.cfg["jet"]["center"][1])
        cz = float(self.cfg["jet"]["center"][2])
        self.inlet_pts = np.column_stack(
            (
                np.full(n_ring, float(self.xc[self.inlet_x_index])),
                cy + rr * np.cos(ang),
                cz + rr * np.sin(ang),
            )
        )
        self.inlet_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n_ring, 1))
        self.n_boundary = int(self.faces.shape[0] + n_ring)
        self.cum_impulse = np.zeros(3)
        self.cum_inflow = np.zeros(3)
        self.cum_outflow = np.zeros(3)
        self.cum_mass_source = np.zeros(1)
        self.reference_velocity = float(self.cfg["jet"]["speed"])
        self.reference_momentum = float(self.total_mass * self.reference_velocity)
    # -------------------------------------------------------------- cloth->grid
    def _rasterize_cloth(self) -> None:
        f = self.faces
        p0 = self.pos[f[:, 0]].astype(F32)
        p1 = self.pos[f[:, 1]].astype(F32)
        p2 = self.pos[f[:, 2]].astype(F32)
        v0 = self.vel[f[:, 0]].astype(F32)
        v1 = self.vel[f[:, 1]].astype(F32)
        v2 = self.vel[f[:, 2]].astype(F32)
        cr = np.cross(p1 - p0, p2 - p0)
        nlen = np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-12)
        nrm = (cr / nlen).astype(F32)
        b = self.bary
        pts = (
            b[:, 0, None, None] * p0[None]
            + b[:, 1, None, None] * p1[None]
            + b[:, 2, None, None] * p2[None]
        )
        vls = (
            b[:, 0, None, None] * v0[None]
            + b[:, 1, None, None] * v1[None]
            + b[:, 2, None, None] * v2[None]
        )
        off = (0.5 * self.h) * nrm[None]
        pts = np.concatenate((pts, pts + off, pts - off), axis=0).reshape(-1, 3)
        vls = np.concatenate((vls, vls, vls), axis=0).reshape(-1, 3)

        nx, ny, nz = self.nx, self.ny, self.nz
        gi = np.clip(((pts[:, 0] - self.low[0]) / self.h).astype(np.int64), 0, nx - 1)
        gj = np.clip(((pts[:, 1] - self.low[1]) / self.h).astype(np.int64), 0, ny - 1)
        gk = np.clip(((pts[:, 2] - self.low[2]) / self.h).astype(np.int64), 0, nz - 1)
        idx = (gi * ny + gj) * nz + gk
        ncell = nx * ny * nz
        cnt = np.bincount(idx, minlength=ncell)
        occ = cnt > 0
        cv = np.zeros((ncell, 3), F32)
        inv = np.zeros(ncell, F32)
        inv[occ] = 1.0 / cnt[occ]
        for k in range(3):
            cv[:, k] = (np.bincount(idx, weights=vls[:, k], minlength=ncell) * inv).astype(F32)
        self.cloth_mask = occ.reshape(nx, ny, nz)
        self.cloth_vel = cv.reshape(nx, ny, nz, 3)

    def _build_solid(self, ramp: float) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        solid = self.static_solid | self.jet_mask | self.cloth_mask
        svel = np.zeros((nx, ny, nz, 3), F32)
        svel[self.jet_mask, 0] = F32(ramp * self.jet_speed)
        cm = self.cloth_mask
        svel[cm] = self.cloth_vel[cm]
        self.solid = solid
        self.svel = svel
        self.us = np.zeros((nx + 1, ny, nz), bool)
        self.vs = np.zeros((nx, ny + 1, nz), bool)
        self.ws = np.zeros((nx, ny, nz + 1), bool)
        self.us[1:-1] = solid[:-1] | solid[1:]
        self.us[0] = solid[0]
        self.us[-1] = solid[-1]
        self.vs[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        self.vs[:, 0] = solid[:, 0]
        self.vs[:, -1] = solid[:, -1]
        self.ws[:, :, 1:-1] = solid[:, :, :-1] | solid[:, :, 1:]
        self.ws[:, :, 0] = True  # floor is a solid wall
        self.ws[:, :, -1] = solid[:, :, -1]
        self.acoef = (
            (~self.us[:-1]).astype(F32)
            + (~self.us[1:]).astype(F32)
            + (~self.vs[:, :-1]).astype(F32)
            + (~self.vs[:, 1:]).astype(F32)
            + (~self.ws[:, :, :-1]).astype(F32)
            + (~self.ws[:, :, 1:]).astype(F32)
        )
        np.maximum(self.acoef, F32(1.0), out=self.acoef)

    def _enforce_solid_faces(self) -> None:
        solid = self.solid
        svel = self.svel
        sl = solid[:-1].astype(F32)
        sr = solid[1:].astype(F32)
        den = np.maximum(sl + sr, F32(1.0))
        uval = (sl * svel[:-1, :, :, 0] + sr * svel[1:, :, :, 0]) / den
        self.u[1:-1] = np.where(self.us[1:-1], uval, self.u[1:-1])
        self.u[0] = np.where(self.us[0], svel[0, :, :, 0], self.u[0])
        self.u[-1] = np.where(self.us[-1], svel[-1, :, :, 0], self.u[-1])

        sl = solid[:, :-1].astype(F32)
        sr = solid[:, 1:].astype(F32)
        den = np.maximum(sl + sr, F32(1.0))
        vval = (sl * svel[:, :-1, :, 1] + sr * svel[:, 1:, :, 1]) / den
        self.v[:, 1:-1] = np.where(self.vs[:, 1:-1], vval, self.v[:, 1:-1])
        self.v[:, 0] = np.where(self.vs[:, 0], svel[:, 0, :, 1], self.v[:, 0])
        self.v[:, -1] = np.where(self.vs[:, -1], svel[:, -1, :, 1], self.v[:, -1])

        sl = solid[:, :, :-1].astype(F32)
        sr = solid[:, :, 1:].astype(F32)
        den = np.maximum(sl + sr, F32(1.0))
        wval = (sl * svel[:, :, :-1, 2] + sr * svel[:, :, 1:, 2]) / den
        self.w[:, :, 1:-1] = np.where(self.ws[:, :, 1:-1], wval, self.w[:, :, 1:-1])
        self.w[:, :, -1] = np.where(self.ws[:, :, -1], svel[:, :, -1, 2], self.w[:, :, -1])
        self.w[:, :, 0] = 0.0
    # ------------------------------------------------------------- projection
    def _project(self) -> None:
        h = F32(self.h)
        div = (
            (self.u[1:] - self.u[:-1])
            + (self.v[:, 1:] - self.v[:, :-1])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / h
        rhs = (-h * h) * div
        rhs[self.solid] = 0.0
        q = self.q
        q[self.solid] = 0.0
        pad = self.qpad
        omega = F32(JACOBI_OMEGA)
        one_minus = F32(1.0) - omega
        for _ in range(GS_ITERATIONS):
            pad[1:-1, 1:-1, 1:-1] = q
            nsum = (
                pad[:-2, 1:-1, 1:-1]
                + pad[2:, 1:-1, 1:-1]
                + pad[1:-1, :-2, 1:-1]
                + pad[1:-1, 2:, 1:-1]
                + pad[1:-1, 1:-1, :-2]
                + pad[1:-1, 1:-1, 2:]
            )
            q = one_minus * q + omega * ((nsum + rhs) / self.acoef)
            q[self.solid] = 0.0
        self.q = np.ascontiguousarray(q, F32)
        q = self.q
        inv_h = F32(1.0 / self.h)
        du = (q[1:] - q[:-1]) * inv_h
        self.u[1:-1] = np.where(self.us[1:-1], self.u[1:-1], self.u[1:-1] - du)
        self.u[0] = np.where(self.us[0], self.u[0], self.u[0] - q[0] * inv_h)
        self.u[-1] = np.where(self.us[-1], self.u[-1], self.u[-1] + q[-1] * inv_h)
        dv = (q[:, 1:] - q[:, :-1]) * inv_h
        self.v[:, 1:-1] = np.where(self.vs[:, 1:-1], self.v[:, 1:-1], self.v[:, 1:-1] - dv)
        self.v[:, 0] = np.where(self.vs[:, 0], self.v[:, 0], self.v[:, 0] - q[:, 0] * inv_h)
        self.v[:, -1] = np.where(self.vs[:, -1], self.v[:, -1], self.v[:, -1] + q[:, -1] * inv_h)
        dw = (q[:, :, 1:] - q[:, :, :-1]) * inv_h
        self.w[:, :, 1:-1] = np.where(
            self.ws[:, :, 1:-1], self.w[:, :, 1:-1], self.w[:, :, 1:-1] - dw
        )
        self.w[:, :, -1] = np.where(
            self.ws[:, :, -1], self.w[:, :, -1], self.w[:, :, -1] + q[:, :, -1] * inv_h
        )

    # -------------------------------------------------------------- advection
    def _advect(self, field, uu, vv, ww, dt):
        n0, n1, n2 = field.shape
        c = F32(dt / self.h)
        gi = np.arange(n0, dtype=F32)[:, None, None] - c * uu
        gj = np.arange(n1, dtype=F32)[None, :, None] - c * vv
        gk = np.arange(n2, dtype=F32)[None, None, :] - c * ww
        flat = np.ascontiguousarray(field, F32).ravel()
        out = _trilerp(flat, n0, n1, n2, gi, gj, gk)
        return np.ascontiguousarray(out, F32)

    def _cell_components(self):
        uc = F32(0.5) * (self.u[:-1] + self.u[1:])
        vc = F32(0.5) * (self.v[:, :-1] + self.v[:, 1:])
        wc = F32(0.5) * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def _update_cell_velocity(self) -> None:
        uc, vc, wc = self._cell_components()
        self.cvx = np.ascontiguousarray(uc, F32)
        self.cvy = np.ascontiguousarray(vc, F32)
        self.cvz = np.ascontiguousarray(wc, F32)
        self.cvx_flat = self.cvx.ravel()
        self.cvy_flat = self.cvy.ravel()
        self.cvz_flat = self.cvz.ravel()

    def _sample_fluid(self, points):
        gi = ((points[:, 0] - self.low[0]) / self.h - 0.5).astype(F32)
        gj = ((points[:, 1] - self.low[1]) / self.h - 0.5).astype(F32)
        gk = ((points[:, 2] - self.low[2]) / self.h - 0.5).astype(F32)
        nx, ny, nz = self.nx, self.ny, self.nz
        out = np.empty((points.shape[0], 3), float)
        out[:, 0] = _trilerp(self.cvx_flat, nx, ny, nz, gi, gj, gk)
        out[:, 1] = _trilerp(self.cvy_flat, nx, ny, nz, gi, gj, gk)
        out[:, 2] = _trilerp(self.cvz_flat, nx, ny, nz, gi, gj, gk)
        return out

    # ------------------------------------------------------------- fluid step
    def _fluid_step(self, dt: float, ramp: float) -> None:
        self._rasterize_cloth()
        self._build_solid(ramp)
        dens_w = _cc_to_face(self.density, 2)
        self.w += F32(dt * BUOYANCY) * dens_w
        self._enforce_solid_faces()
        uc, vc, wc = self._cell_components()
        u_new = self._advect(self.u, self.u, _cc_to_face(vc, 0), _cc_to_face(wc, 0), dt)
        v_new = self._advect(self.v, _cc_to_face(uc, 1), self.v, _cc_to_face(wc, 1), dt)
        w_new = self._advect(self.w, _cc_to_face(uc, 2), _cc_to_face(vc, 2), self.w, dt)
        dens = self._advect(self.density, uc, vc, wc, dt)
        self.u, self.v, self.w, self.density = u_new, v_new, w_new, dens
        if ramp > 0.0:
            self.density[self.jet_mask] = F32(ramp * self.jet_density)
        self.density[self.static_solid] = 0.0
        self.density[self.cloth_mask] = 0.0
        np.clip(self.density, 0.0, 1.2, out=self.density)
        self._enforce_solid_faces()
        self._project()
        self._enforce_solid_faces()
        self._update_cell_velocity()
    # ------------------------------------------------------------ cloth forces
    def _spring_forces(self):
        out = np.zeros((self.n_vert, 3))
        for a, b, rest, k, c, _tag in self.spring_sets:
            d = self.pos[a] - self.pos[b]
            length = np.linalg.norm(d, axis=1)
            safe = np.maximum(length, 1e-9)
            unit = d / safe[:, None]
            f = -(k * (length - rest))[:, None] * unit
            dv = self.vel[a] - self.vel[b]
            f -= (c * np.einsum("ij,ij->i", dv, unit))[:, None] * unit
            np.add.at(out, a, f)
            np.add.at(out, b, -f)
        return out

    def _contact_forces(self):
        out = np.zeros((self.n_vert, 3))
        pen = 0.5 * self.cloth_thickness - self.pos[:, 2]
        act = pen > 0.0
        if np.any(act):
            out[act, 2] = np.maximum(
                FLOOR_K * pen[act] - FLOOR_C * self.vel[act, 2], 0.0
            )
        return out

    def _triangle_geometry(self):
        f = self.faces
        p0, p1, p2 = self.pos[f[:, 0]], self.pos[f[:, 1]], self.pos[f[:, 2]]
        cr = np.cross(p1 - p0, p2 - p0)
        two_area = np.linalg.norm(cr, axis=1)
        nrm = cr / np.maximum(two_area, 1e-12)[:, None]
        cen = (p0 + p1 + p2) / 3.0
        vtri = (self.vel[f[:, 0]] + self.vel[f[:, 1]] + self.vel[f[:, 2]]) / 3.0
        return cen, nrm, 0.5 * two_area, vtri

    def _aero_forces(self):
        cen, nrm, area, vtri = self._triangle_geometry()
        d = self.aero_offset
        va = self._sample_fluid(cen + d * nrm) - vtri
        vb = self._sample_fluid(cen - d * nrm) - vtri
        vna = np.einsum("ij,ij->i", va, nrm)
        vnb = np.einsum("ij,ij->i", vb, nrm)
        press = 0.5 * RHO_AIR * CD_NORMAL * area * (
            np.abs(vna) * vna + np.abs(vnb) * vnb
        )
        force = press[:, None] * nrm
        vt = 0.5 * ((va - vna[:, None] * nrm) + (vb - vnb[:, None] * nrm))
        force += (0.5 * RHO_AIR * CT_TANGENT * area)[:, None] * (
            np.linalg.norm(vt, axis=1)[:, None] * vt
        )
        out = np.zeros((self.n_vert, 3))
        third = force / 3.0
        f = self.faces
        np.add.at(out, f[:, 0], third)
        np.add.at(out, f[:, 1], third)
        np.add.at(out, f[:, 2], third)
        return out

    def _strain_limit(self, passes: int = 3) -> None:
        a, b, rest = self.struct_a, self.struct_b, self.struct_rest
        wa, wb = self.inv_mass[a], self.inv_mass[b]
        wsum = np.maximum(wa + wb, 1e-12)
        lo, hi = 0.95 * rest, 1.03 * rest
        for _ in range(passes):
            d = self.pos[a] - self.pos[b]
            length = np.linalg.norm(d, axis=1)
            safe = np.maximum(length, 1e-9)
            target = np.clip(length, lo, hi)
            corr = ((length - target) / (safe * wsum))[:, None] * d
            np.add.at(self.pos, a, -wa[:, None] * corr)
            np.add.at(self.pos, b, wb[:, None] * corr)

    # -------------------------------------------------------------- cloth step
    def _cloth_step(self, dt: float) -> None:
        n_sub = max(1, int(round(dt / CLOTH_SUBSTEP)))
        sub = dt / n_sub
        fg = self.pmass[:, None] * self.gravity[None, :]
        sum_g = np.zeros(3)
        sum_a = np.zeros(3)
        sum_c = np.zeros(3)
        imp_pin = np.zeros(3)
        p_start = (self.pmass[:, None] * self.vel).sum(axis=0)
        for _ in range(n_sub):
            fa = self._aero_forces() - (AIR_DAMP * self.pmass)[:, None] * self.vel
            fc = self._contact_forces()
            fs = self._spring_forces()
            p_before = (self.pmass[:, None] * self.vel).sum(axis=0)
            self.vel += sub * (fg + fa + fc + fs) * self.inv_mass[:, None]
            self.vel[self.pinned] = 0.0
            self.pos += sub * self.vel
            self.pos[self.pinned] = self.pin_pos
            pre = self.pos.copy()
            self._strain_limit()
            self.vel += (self.pos - pre) / sub
            self.pos[self.pinned] = self.pin_pos
            self.vel[self.pinned] = 0.0
            p_after = (self.pmass[:, None] * self.vel).sum(axis=0)
            g_tot, a_tot, c_tot = fg.sum(axis=0), fa.sum(axis=0), fc.sum(axis=0)
            imp_pin += (p_after - p_before) - sub * (g_tot + a_tot + c_tot)
            sum_g += g_tot
            sum_a += a_tot
            sum_c += c_tot
        p_end = (self.pmass[:, None] * self.vel).sum(axis=0)
        self.cum_impulse += p_end - p_start
        self.frame_forces = np.stack(
            (sum_g / n_sub, sum_a / n_sub, sum_c / n_sub, imp_pin / dt)
        )

    def _initial_force_terms(self) -> None:
        fg = self.pmass[:, None] * self.gravity[None, :]
        fa = self._aero_forces() - (AIR_DAMP * self.pmass)[:, None] * self.vel
        fc = self._contact_forces()
        fs = self._spring_forces()
        pin = -(fg + fa + fc + fs)[self.pinned].sum(axis=0)
        self.frame_forces = np.stack(
            (fg.sum(axis=0), fa.sum(axis=0), fc.sum(axis=0), pin)
        )

    # ------------------------------------------------------------- time march
    def _ramp(self, t: float) -> float:
        if t < self.jet_start or t > self.jet_until:
            return 0.0
        return float(min(1.0, (t - self.jet_start) / self.jet_ramp))

    def _step(self, dt: float) -> None:
        ramp = self._ramp(self.sim_time + 0.5 * dt)
        self._fluid_step(dt, ramp)
        self._cloth_step(dt)
        self.sim_time += dt
        self._steps += 1
        self.last_ramp = ramp

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        dt_max = float(self.cfg["sample_dt"])
        while self.sim_time < target - 1e-9:
            self._step(min(dt_max, target - self.sim_time))
        if self.frame_forces is None:
            self._initial_force_terms()
        return self._make_frame(target)

    # ----------------------------------------------------------- frame export
    def _make_frame(self, t: float) -> StateFrame:
        cen, nrm, _area, vtri = self._triangle_geometry()
        n_ring = self.inlet_pts.shape[0]
        inlet_target = np.tile(
            np.array([self.last_ramp * self.jet_speed, 0.0, 0.0]), (n_ring, 1)
        )
        b_normal = np.vstack((nrm, self.inlet_normal))
        b_target = np.vstack((vtri, inlet_target))
        b_vel = np.vstack((self._sample_fluid(cen), self._sample_fluid(self.inlet_pts)))

        fg, fa, fc, fpin = self.frame_forces
        interaction = np.stack((fa, -fa, fc, -fc, fpin, -fpin))

        nx2, ny2, nz2 = self.nx // 2, self.ny // 2, self.nz // 2
        vel_ds = np.stack(
            [
                comp.reshape(nx2, 2, ny2, 2, nz2, 2).mean(axis=(1, 3, 5))
                for comp in (self.cvx, self.cvy, self.cvz)
            ],
            axis=-1,
        ).astype(F32)

        smoke = np.ascontiguousarray(self.density, F32)
        smoke_mass = float(smoke.sum()) * (self.h ** 3)

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.pmass.copy(),
            body_id=np.ones(self.n_vert, dtype=np.int32),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pinned.astype(float),
                "mass_quantity": self.pmass.reshape(-1, 1).copy(),
            },
            fields={
                "smoke_density": smoke.copy(),
                "fluid_velocity": vel_ds,
                "boundary_normal": b_normal,
                "boundary_target_velocity": b_target,
                "boundary_velocity": b_vel,
                "budget_domain_weights": np.ones((1, self.n_vert)),
                "budget_external_force": self.frame_forces.reshape(1, 4, 3).copy(),
                "budget_cumulative_external_impulse": self.cum_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.reshape(1, 3).copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": interaction,
            },
            scalars={
                "reference_velocity": self.reference_velocity,
                "reference_momentum": self.reference_momentum,
                "grid_spacing": float(self.h),
                "characteristic_length": float(self.cfg["cloth"]["height"]),
                "contact_length_scale": float(self.cloth_thickness),
                "jet_ramp": float(self.last_ramp),
                "smoke_mass": smoke_mass,
                "cloth_mass": float(self.total_mass),
            },
        )
        frame.validate()
        self.frame_index += 1
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC-grid semi-Lagrangian smoke + mass-spring cloth, two-way coupled",
            "fluid_steps": int(self._steps),
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "poisson_sweeps": int(GS_ITERATIONS),
            "cloth_substep": float(CLOTH_SUBSTEP),
            "final_time": float(self.sim_time),
            "final_smoke_mass": float(self.density.sum()) * (self.h ** 3),
        }
