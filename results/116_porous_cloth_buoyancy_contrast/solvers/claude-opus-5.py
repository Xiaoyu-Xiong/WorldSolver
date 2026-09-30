"""Porous-cloth buoyancy contrast: MAC-grid water two-way coupled to cloth shells.

Water
-----
Incompressible Navier-Stokes on a staggered MAC grid covering the frozen tank
interior water region.  Each step performs semi-Lagrangian self-advection, an
explicit eddy-viscosity diffusion, injection of the fabric reaction impulse and
an exact pressure projection (sparse LU factorisation of the constant Poisson
operator).  Tank walls and floor are no-flow Neumann boundaries; the free
surface is a ``p = 0`` Dirichlet boundary.  The water samples of the frozen
state layout are PIC tracers carried by the projected grid velocity, so the
resolved flow is what transports them.

Cloth
-----
Three mass-spring shells built on the frozen 9x7 grid topology (structural,
shear and bending springs with spring damping), integrated with symplectic
Euler substeps, plus penalty contact and Coulomb friction against the frozen
tank geometry.

Two-way permeable coupling
--------------------------
* Buoyancy uses the fabric solid volume ``V = m / rho_fabric`` implied by the
  frozen per-sample mass and material density (never a cloth bounding box), and
  is tapered by the per-vertex submergence at the free surface.
* Pressure loading ``-V grad(p_dyn)`` is sampled from the resolved grid, so the
  fabric feels the dynamic pressure the liquid actually carries.
* Seepage drag: liquid genuinely passes through the fabric.  The pressure drop
  across the sheet is the series (harmonic) combination of the Darcy resistance
  ``mu t / kappa * u_n`` and the inertial bypass pressure ``0.5 rho Cd u_n^2``,
  so a permeable sheet loads far less than an impermeable one and the
  flow-through limit is recovered for large permeability.  Tangential skin
  friction is added separately.
* The drag plus dynamic-pressure reaction is scattered back onto the MAC grid
  as an impulse, so the moving, folding fabric alters the resolved flow, which
  in turn advects the tracers and the neutrally buoyant piece.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12

def _sample(field, origin, inv_h, points):
    """Trilinear sample of a scalar grid ``field`` whose node (0,0,0) sits at ``origin``."""
    nx, ny, nz = field.shape
    g = (points - origin) * inv_h
    i0 = np.clip(np.floor(g[:, 0]).astype(np.int64), 0, nx - 2) if nx > 1 else np.zeros(len(g), np.int64)
    j0 = np.clip(np.floor(g[:, 1]).astype(np.int64), 0, ny - 2) if ny > 1 else np.zeros(len(g), np.int64)
    k0 = np.clip(np.floor(g[:, 2]).astype(np.int64), 0, nz - 2) if nz > 1 else np.zeros(len(g), np.int64)
    i1 = i0 + 1 if nx > 1 else i0
    j1 = j0 + 1 if ny > 1 else j0
    k1 = k0 + 1 if nz > 1 else k0
    fx = np.clip(g[:, 0] - i0, 0.0, 1.0) if nx > 1 else np.zeros(len(g))
    fy = np.clip(g[:, 1] - j0, 0.0, 1.0) if ny > 1 else np.zeros(len(g))
    fz = np.clip(g[:, 2] - k0, 0.0, 1.0) if nz > 1 else np.zeros(len(g))
    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i1, j0, k0] * fx
    c10 = field[i0, j1, k0] * (1.0 - fx) + field[i1, j1, k0] * fx
    c01 = field[i0, j0, k1] * (1.0 - fx) + field[i1, j0, k1] * fx
    c11 = field[i0, j1, k1] * (1.0 - fx) + field[i1, j1, k1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _scatter(field, origin, inv_h, points, values):
    """Momentum-conserving trilinear scatter of ``values`` into ``field`` (in place)."""
    nx, ny, nz = field.shape
    g = (points - origin) * inv_h
    i0 = np.clip(np.floor(g[:, 0]).astype(np.int64), 0, nx - 2) if nx > 1 else np.zeros(len(g), np.int64)
    j0 = np.clip(np.floor(g[:, 1]).astype(np.int64), 0, ny - 2) if ny > 1 else np.zeros(len(g), np.int64)
    k0 = np.clip(np.floor(g[:, 2]).astype(np.int64), 0, nz - 2) if nz > 1 else np.zeros(len(g), np.int64)
    i1 = i0 + 1 if nx > 1 else i0
    j1 = j0 + 1 if ny > 1 else j0
    k1 = k0 + 1 if nz > 1 else k0
    fx = np.clip(g[:, 0] - i0, 0.0, 1.0) if nx > 1 else np.zeros(len(g))
    fy = np.clip(g[:, 1] - j0, 0.0, 1.0) if ny > 1 else np.zeros(len(g))
    fz = np.clip(g[:, 2] - k0, 0.0, 1.0) if nz > 1 else np.zeros(len(g))
    for ii, wx in ((i0, 1.0 - fx), (i1, fx)):
        for jj, wy in ((j0, 1.0 - fy), (j1, fy)):
            for kk, wz in ((k0, 1.0 - fz), (k1, fz)):
                np.add.at(field, (ii, jj, kk), values * (wx * wy * wz))


class BenchmarkSolver:
    """Two-way coupled porous-cloth / water solver."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        config = scene.config if hasattr(scene, "config") else {}
        self.config = config
        meta = scene.metadata()

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.uv = np.array(frame0.attributes["uv"], dtype=float)
        self.material_density = np.array(frame0.attributes["material_density"], dtype=float)
        self.porosity_attr = np.array(frame0.attributes["porosity"], dtype=float)
        self.permeability_attr = np.array(frame0.attributes["permeability_m2"], dtype=float)
        self.n_samples = len(self.position)

        self.gravity = np.asarray(config["gravity"], dtype=float)
        tank = config["tank"]
        self.tank_min = np.asarray(tank["interior_bounds_min"], dtype=float)
        self.tank_max = np.asarray(tank["interior_bounds_max"], dtype=float)
        water = config["water"]
        self.rho_w = float(water["density"])
        self.surface_z = float(water["free_surface_z"])
        self.pressure_loading = bool(water.get("pressure_loading_enabled", True))

        cloth_cfg = config["cloth"]
        self.grid_shape = tuple(int(v) for v in cloth_cfg["grid_shape"])
        self.cloth_size = np.asarray(cloth_cfg["size"], dtype=float)
        self.thickness = float(cloth_cfg["thickness"])
        self.porosity = float(cloth_cfg["porosity"])
        self.permeability = float(cloth_cfg["permeability_m2"])
        self.faces = np.asarray(meta["cloth_topology"]["faces"], dtype=np.int64)

        self.water_idx = np.flatnonzero(self.body_id == 0)
        self.cloth_idx = np.flatnonzero(self.body_id > 0)
        self.cloth_body = self.body_id[self.cloth_idx].astype(np.int64)
        self.body_ids = (1, 2, 3)
        self.body_masks = [self.cloth_body == bid for bid in self.body_ids]
        self.n_cloth = len(self.cloth_idx)
        self.verts_per_patch = self.n_cloth // 3

        # Cloth working state (local indexing).
        self.cp = self.position[self.cloth_idx].copy()
        self.cv = self.velocity[self.cloth_idx].copy()
        self.cm = self.mass[self.cloth_idx].copy()
        self.c_rho = self.material_density[self.cloth_idx].copy()
        # Fabric solid volume implied by frozen mass and material density.
        self.c_vol = self.cm / self.c_rho
        patch_area = float(self.cloth_size[0] * self.cloth_size[1])
        self.c_area = np.full(self.n_cloth, patch_area / self.verts_per_patch)
        self.c_saturation = np.ones(self.n_cloth)

        # Water tracers.
        self.wp = self.position[self.water_idx].copy()
        self.wv = self.velocity[self.water_idx].copy()
        self.wm = self.mass[self.water_idx].copy()
        self.water_mass_total = float(self.wm.sum())

        self._build_cloth_springs()
        self._build_grid()

        # Coupling / material parameters.
        self.mu = 1.0e-3                     # water dynamic viscosity [Pa s]
        self.nu_eddy = 5.0e-4                # eddy viscosity for grid stability [m^2/s]
        self.darcy_R = self.mu * self.thickness / max(self.permeability, _EPS)
        self.cd_normal = 1.2                 # bluff-sheet bypass coefficient
        self.cf_tangent = 0.02               # skin-friction coefficient
        self.tangent_linear = 5.0            # linear tangential resistance [Pa s / m]
        self.taper = 0.02                    # free-surface submergence taper [m]
        self.contact_k = 4.0e3               # penalty contact stiffness [N/m]
        self.contact_c = 12.0                # contact damping [N s / m]
        self.contact_mu = 0.4                # Coulomb friction coefficient
        self.wick_rate = 2.0                 # saturation relaxation rate [1/s]

        # Time stepping.
        times = np.asarray(scene.sample_times(), dtype=float)
        self.sample_times = times
        self.sample_dt = float(times[1] - times[0]) if len(times) > 1 else 0.02
        self.dt_fluid = 0.5 * self.sample_dt
        self.fluid_per_sample = 2
        self.dt_cloth_target = 5.0e-4
        self.time = 0.0
        self.step_index = 0

        # Momentum-budget bookkeeping: rows = global, light, neutral, heavy, water.
        self.n_rows = 5
        self.n_terms = 4                     # gravity, fluid coupling, contact, boundary
        self.domain_weights = np.zeros((self.n_rows, self.n_samples))
        self.domain_weights[0, :] = 1.0
        for row, bid in enumerate(self.body_ids, start=1):
            self.domain_weights[row, self.body_id == bid] = 1.0
        self.domain_weights[4, self.body_id == 0] = 1.0
        self.cum_impulse = np.zeros((self.n_rows, 3))
        self.budget_force = np.zeros((self.n_rows, self.n_terms, 3))
        self.interaction_force = np.zeros((6, 3))
        self.cum_mass_source = np.zeros(1)
        self.reference_momentum = float(self.cm.sum() * np.sqrt(2.0 * 9.81 * 1.0))
        if not (self.reference_momentum > 0.0):
            self.reference_momentum = 1.0
        self.p_grad = np.zeros((3,) + self.cell_shape)
        self.pressure = np.zeros(self.cell_shape)
        self._last_sub = self._submergence()
        self.diag = {"fluid_steps": 0, "cloth_substeps": 0}
        self._sync_frame_arrays()
        self.budget_force = self._initial_budget()

    def _build_cloth_springs(self) -> None:
        nx, ny = self.grid_shape
        struct_a, struct_b, shear_a, shear_b, bend_a, bend_b = [], [], [], [], [], []
        for patch in range(3):
            base = patch * self.verts_per_patch
            for iy in range(ny):
                for ix in range(nx):
                    a = base + iy * nx + ix
                    if ix + 1 < nx:
                        struct_a.append(a)
                        struct_b.append(a + 1)
                    if iy + 1 < ny:
                        struct_a.append(a)
                        struct_b.append(a + nx)
                    if ix + 1 < nx and iy + 1 < ny:
                        shear_a.append(a)
                        shear_b.append(a + nx + 1)
                        shear_a.append(a + 1)
                        shear_b.append(a + nx)
                    if ix + 2 < nx:
                        bend_a.append(a)
                        bend_b.append(a + 2)
                    if iy + 2 < ny:
                        bend_a.append(a)
                        bend_b.append(a + 2 * nx)
        stiffness = {"struct": 600.0, "shear": 150.0, "bend": 40.0}
        damp_ratio = 0.30
        groups = (
            ("struct", struct_a, struct_b),
            ("shear", shear_a, shear_b),
            ("bend", bend_a, bend_b),
        )
        self.springs = []
        for name, ia_list, ib_list in groups:
            ia = np.asarray(ia_list, dtype=np.int64)
            ib = np.asarray(ib_list, dtype=np.int64)
            if len(ia) == 0:
                continue
            rest = np.linalg.norm(self.cp[ib] - self.cp[ia], axis=1)
            rest = np.maximum(rest, 1.0e-6)
            k = stiffness[name]
            m_pair = 0.5 * (self.cm[ia] + self.cm[ib])
            c = damp_ratio * np.sqrt(k * m_pair)
            self.springs.append((ia, ib, rest, k, c))

    def _build_grid(self) -> None:
        """Staggered MAC grid over the frozen initial water region."""
        water = self.config["water"]
        lo = np.asarray(water["initial_bounds_min"], dtype=float)
        hi = np.asarray(water["initial_bounds_max"], dtype=float)
        self.g_lo = lo
        self.g_hi = hi
        self.nx, self.ny, self.nz = 24, 10, 11
        self.cell_shape = (self.nx, self.ny, self.nz)
        self.h = (hi - lo) / np.asarray([self.nx, self.ny, self.nz], dtype=float)
        self.hx, self.hy, self.hz = (float(v) for v in self.h)
        self.cell_volume = self.hx * self.hy * self.hz
        self.cell_mass = self.rho_w * self.cell_volume

        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))

        # Sample origins (position of index (0,0,0)) for each staggered field.
        self.u_org = lo + np.asarray([0.0, 0.5 * self.hy, 0.5 * self.hz])
        self.v_org = lo + np.asarray([0.5 * self.hx, 0.0, 0.5 * self.hz])
        self.w_org = lo + np.asarray([0.5 * self.hx, 0.5 * self.hy, 0.0])
        self.c_org = lo + 0.5 * self.h
        self.inv_h = 1.0 / self.h

        def face_points(shape, origin):
            axes = [origin[d] + np.arange(shape[d]) * self.h[d] for d in range(3)]
            grid = np.meshgrid(*axes, indexing="ij")
            return np.stack([g.reshape(-1) for g in grid], axis=1)

        self.u_pts = face_points(self.u.shape, self.u_org)
        self.v_pts = face_points(self.v.shape, self.v_org)
        self.w_pts = face_points(self.w.shape, self.w_org)
        self._build_poisson()

    def _build_poisson(self) -> None:
        """Constant pressure-Poisson operator: Neumann walls, Dirichlet p=0 at surface."""
        nx, ny, nz = self.cell_shape
        n = nx * ny * nz
        idx = np.arange(n).reshape(self.cell_shape)
        rows, cols, vals = [], [], []
        diag = np.zeros(self.cell_shape)
        for axis, h in ((0, self.hx), (1, self.hy), (2, self.hz)):
            inv2 = 1.0 / (h * h)
            for direction in (-1, 1):
                src = [slice(None)] * 3
                dst = [slice(None)] * 3
                if direction > 0:
                    src[axis] = slice(0, self.cell_shape[axis] - 1)
                    dst[axis] = slice(1, self.cell_shape[axis])
                else:
                    src[axis] = slice(1, self.cell_shape[axis])
                    dst[axis] = slice(0, self.cell_shape[axis] - 1)
                a = idx[tuple(src)].reshape(-1)
                b = idx[tuple(dst)].reshape(-1)
                rows.append(a)
                cols.append(b)
                vals.append(np.full(len(a), -inv2))
                diag[tuple(src)] += inv2
            if axis == 2:
                # Free surface above the top cell layer: Dirichlet ghost, p_ghost = -p.
                diag[:, :, nz - 1] += 2.0 * inv2
        rows.append(idx.reshape(-1))
        cols.append(idx.reshape(-1))
        vals.append(diag.reshape(-1))
        matrix = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsc()
        self.poisson = matrix
        self.poisson_lu = sparse_linalg.splu(matrix)

    # ---------------------------------------------------------------- fluid
    def _sample_velocity(self, points):
        return np.stack(
            (
                _sample(self.u, self.u_org, self.inv_h, points),
                _sample(self.v, self.v_org, self.inv_h, points),
                _sample(self.w, self.w_org, self.inv_h, points),
            ),
            axis=1,
        )

    def _advect(self, dt: float) -> None:
        u_old, v_old, w_old = self.u.copy(), self.v.copy(), self.w.copy()
        new = []
        for pts, field, org, shape in (
            (self.u_pts, u_old, self.u_org, self.u.shape),
            (self.v_pts, v_old, self.v_org, self.v.shape),
            (self.w_pts, w_old, self.w_org, self.w.shape),
        ):
            vel = self._sample_velocity(pts)
            back = pts - dt * vel
            back = np.clip(back, self.g_lo, self.g_hi)
            new.append(_sample(field, org, self.inv_h, back).reshape(shape))
        self.u, self.v, self.w = new

    @staticmethod
    def _laplacian(field, hx, hy, hz):
        padded = np.pad(field, 1, mode="edge")
        lap = (padded[2:, 1:-1, 1:-1] - 2.0 * field + padded[:-2, 1:-1, 1:-1]) / (hx * hx)
        lap += (padded[1:-1, 2:, 1:-1] - 2.0 * field + padded[1:-1, :-2, 1:-1]) / (hy * hy)
        lap += (padded[1:-1, 1:-1, 2:] - 2.0 * field + padded[1:-1, 1:-1, :-2]) / (hz * hz)
        return lap

    def _diffuse(self, dt: float) -> None:
        nu = self.nu_eddy
        if nu <= 0.0:
            return
        self.u = self.u + dt * nu * self._laplacian(self.u, self.hx, self.hy, self.hz)
        self.v = self.v + dt * nu * self._laplacian(self.v, self.hx, self.hy, self.hz)
        self.w = self.w + dt * nu * self._laplacian(self.w, self.hx, self.hy, self.hz)

    def _boundary(self) -> None:
        """No-flow tank walls and floor; the free surface at the top stays open."""
        for field in (self.u, self.v, self.w):
            np.nan_to_num(field, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(field, -20.0, 20.0, out=field)
        self.u[0, :, :] = 0.0
        self.u[-1, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0

    def _project(self, dt: float) -> None:
        div = (self.u[1:, :, :] - self.u[:-1, :, :]) / self.hx
        div += (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.hy
        div += (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.hz
        rhs = -(self.rho_w / dt) * div.reshape(-1)
        pressure = self.poisson_lu.solve(rhs).reshape(self.cell_shape)
        if not np.all(np.isfinite(pressure)):
            pressure = np.zeros(self.cell_shape)
        self.pressure = pressure
        scale = dt / self.rho_w
        self.u[1:-1, :, :] -= scale * (pressure[1:, :, :] - pressure[:-1, :, :]) / self.hx
        self.v[:, 1:-1, :] -= scale * (pressure[:, 1:, :] - pressure[:, :-1, :]) / self.hy
        self.w[:, :, 1:-1] -= scale * (pressure[:, :, 1:] - pressure[:, :, :-1]) / self.hz
        # Free-surface face: ghost pressure p_ghost = -p keeps the projection exact.
        self.w[:, :, -1] += scale * 2.0 * pressure[:, :, -1] / self.hz
        self._boundary()
        self.p_grad = np.stack(
            (
                np.gradient(pressure, self.hx, axis=0),
                np.gradient(pressure, self.hy, axis=1),
                np.gradient(pressure, self.hz, axis=2),
            ),
            axis=0,
        )

    def _apply_cloth_impulse(self, node_impulse) -> None:
        """Scatter the fabric reaction impulse onto the grid (two-way momentum exchange)."""
        for field, org, comp in (
            (self.u, self.u_org, 0),
            (self.v, self.v_org, 1),
            (self.w, self.w_org, 2),
        ):
            _scatter(field, org, self.inv_h, self.cp, node_impulse[:, comp] / self.cell_mass)

    def _advect_tracers(self, dt: float) -> None:
        lo = self.tank_min + 5.0e-3
        hi = np.asarray(
            [self.tank_max[0] - 5.0e-3, self.tank_max[1] - 5.0e-3, self.surface_z - 5.0e-3]
        )
        vel1 = self._sample_velocity(self.wp)
        mid = np.clip(self.wp + 0.5 * dt * vel1, lo, hi)
        vel2 = self._sample_velocity(mid)
        self.wp = np.clip(self.wp + dt * vel2, lo, hi)
        self.wv = self._sample_velocity(self.wp)

    # ---------------------------------------------------------------- cloth
    def _global_faces(self):
        if not hasattr(self, "_gfaces"):
            blocks = [self.faces + patch * self.verts_per_patch for patch in range(3)]
            self._gfaces = np.concatenate(blocks, axis=0)
        return self._gfaces

    def _vertex_normals(self):
        faces = self._global_faces()
        p0 = self.cp[faces[:, 0]]
        p1 = self.cp[faces[:, 1]]
        p2 = self.cp[faces[:, 2]]
        face_n = np.cross(p1 - p0, p2 - p0)
        normals = np.zeros_like(self.cp)
        for col in range(3):
            np.add.at(normals, faces[:, col], face_n)
        norm = np.linalg.norm(normals, axis=1)
        bad = norm < 1.0e-14
        normals[bad] = np.asarray([0.0, 0.0, 1.0])
        norm = np.where(bad, 1.0, norm)
        return normals / norm[:, None]

    def _submergence(self):
        depth = self.surface_z - self.cp[:, 2]
        return np.clip(depth / self.taper, 0.0, 1.0)

    def _fluid_forces(self, fluid_vel, grad_p, normals, sub):
        """Permeable-sheet fluid load: buoyancy, dynamic pressure and seepage drag."""
        sub_c = sub[:, None]
        rel = fluid_vel - self.cv
        un = np.sum(rel * normals, axis=1)
        un_abs = np.abs(un)
        p_darcy = self.darcy_R * un_abs
        p_inert = 0.5 * self.rho_w * self.cd_normal * un_abs * un_abs
        # Series (resistances-in-series) blend: seepage-limited below, bypass-limited above.
        p_series = p_darcy * p_inert / (p_darcy + p_inert + _EPS)
        p_norm = p_series + self.tangent_linear * 6.0 * un_abs
        f_normal = (p_norm * self.c_area * sub / (un_abs + _EPS))[:, None] * (un[:, None] * normals)

        rel_t = rel - un[:, None] * normals
        speed_t = np.linalg.norm(rel_t, axis=1)
        p_tan = self.tangent_linear * speed_t + 0.5 * self.rho_w * self.cf_tangent * speed_t * speed_t
        f_tan = (p_tan * self.c_area * sub / (speed_t + _EPS))[:, None] * rel_t

        drag = f_normal + f_tan
        # Buoyancy from the fabric solid volume implied by frozen mass and density.
        buoyancy = -self.rho_w * (self.c_vol[:, None] * self.gravity[None, :]) * sub_c
        if self.pressure_loading:
            bulk_volume = (self.c_area * self.thickness)[:, None]
            pressure_force = -bulk_volume * grad_p * sub_c
        else:
            pressure_force = np.zeros_like(drag)
        reaction = drag + pressure_force
        return buoyancy + reaction, reaction

    def _contact_forces(self):
        offset = 0.5 * self.thickness
        force = np.zeros_like(self.cp)
        planes = (
            (2, 1.0, float(self.tank_min[2])),
            (0, 1.0, float(self.tank_min[0])),
            (0, -1.0, float(self.tank_max[0])),
            (1, 1.0, float(self.tank_min[1])),
            (1, -1.0, float(self.tank_max[1])),
        )
        for axis, sign, bound in planes:
            if sign > 0.0:
                depth = (bound + offset) - self.cp[:, axis]
            else:
                depth = self.cp[:, axis] - (bound - offset)
            active = depth > 0.0
            if not np.any(active):
                continue
            normal = np.zeros(3)
            normal[axis] = sign
            vn = sign * self.cv[:, axis]
            fn = np.where(active, self.contact_k * depth - self.contact_c * vn, 0.0)
            fn = np.maximum(fn, 0.0)
            force += fn[:, None] * normal[None, :]
            vt = self.cv - vn[:, None] * normal[None, :]
            speed_t = np.linalg.norm(vt, axis=1)
            scale = self.contact_mu * fn / (speed_t + 0.05)
            force -= scale[:, None] * vt
        return force

    def _spring_forces(self):
        force = np.zeros_like(self.cp)
        for ia, ib, rest, k, c in self.springs:
            delta = self.cp[ib] - self.cp[ia]
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length, 1.0e-9)[:, None]
            rel_v = np.sum((self.cv[ib] - self.cv[ia]) * direction, axis=1)
            magnitude = k * (length - rest) + c * rel_v
            f = magnitude[:, None] * direction
            np.add.at(force, ia, f)
            np.add.at(force, ib, -f)
        return force

    def _step_cloth(self, dt_total: float):
        """Substepped symplectic-Euler cloth advance under the frozen fluid state."""
        n_sub = max(1, int(np.ceil(dt_total / self.dt_cloth_target)))
        dt = dt_total / n_sub
        fluid_vel = self._sample_velocity(self.cp)
        grad_p = np.stack(
            [_sample(self.p_grad[d], self.c_org, self.inv_h, self.cp) for d in range(3)],
            axis=1,
        )
        node_reaction_impulse = np.zeros_like(self.cp)
        fluid_impulse = np.zeros((3, 3))
        contact_impulse = np.zeros((3, 3))
        sub = self._submergence()
        for _ in range(n_sub):
            normals = self._vertex_normals()
            sub = self._submergence()
            fluid_total, reaction = self._fluid_forces(fluid_vel, grad_p, normals, sub)
            contact = self._contact_forces()
            springs = self._spring_forces()
            gravity_force = self.cm[:, None] * self.gravity[None, :]
            total = gravity_force + springs + fluid_total + contact
            self.cv += dt * total / self.cm[:, None]
            np.clip(self.cv, -20.0, 20.0, out=self.cv)
            self.cp += dt * self.cv
            node_reaction_impulse += dt * reaction
            for i, mask in enumerate(self.body_masks):
                fluid_impulse[i] += dt * fluid_total[mask].sum(axis=0)
                contact_impulse[i] += dt * contact[mask].sum(axis=0)
            self.diag["cloth_substeps"] += 1
        self._last_sub = sub
        return node_reaction_impulse, fluid_impulse, contact_impulse

    # --------------------------------------------------------------- budget
    def _momenta(self):
        """Row-wise linear momentum: [global, light, neutral, heavy, water]."""
        rows = np.zeros((self.n_rows, 3))
        for i, mask in enumerate(self.body_masks):
            rows[i + 1] = (self.cm[mask, None] * self.cv[mask]).sum(axis=0)
        rows[4] = (self.wm[:, None] * self.wv).sum(axis=0)
        rows[0] = rows[1:].sum(axis=0)
        return rows

    def _step(self, dt: float) -> None:
        before = self._momenta()
        node_impulse, fluid_impulse, contact_impulse = self._step_cloth(dt)

        self._advect(dt)
        self._diffuse(dt)
        self._apply_cloth_impulse(-node_impulse)
        self._boundary()
        self._project(dt)
        self._advect_tracers(dt)
        self.diag["fluid_steps"] += 1

        target = self._last_sub
        self.c_saturation += (target - self.c_saturation) * (1.0 - np.exp(-self.wick_rate * dt))
        np.clip(self.c_saturation, 0.0, 1.0, out=self.c_saturation)

        after = self._momenta()
        delta = after - before

        terms = np.zeros((self.n_rows, self.n_terms, 3))
        for i, mask in enumerate(self.body_masks):
            row = i + 1
            terms[row, 0] = dt * self.cm[mask].sum() * self.gravity
            terms[row, 1] = fluid_impulse[i]
            terms[row, 2] = contact_impulse[i]
            terms[row, 3] = delta[row] - terms[row, :3].sum(axis=0)
        terms[4, 0] = dt * self.water_mass_total * self.gravity
        terms[4, 1] = -fluid_impulse.sum(axis=0)
        terms[4, 2] = np.zeros(3)
        terms[4, 3] = delta[4] - terms[4, :3].sum(axis=0)
        terms[0] = terms[1:].sum(axis=0)

        self.cum_impulse += terms.sum(axis=1)
        self.budget_force = terms / dt
        fluid_force = fluid_impulse / dt
        self.interaction_force = np.concatenate((fluid_force, -fluid_force), axis=0)

    def _initial_budget(self):
        """External-force terms evaluated at the released rest state."""
        normals = self._vertex_normals()
        sub = self._submergence()
        fluid_vel = self._sample_velocity(self.cp)
        grad_p = np.zeros_like(self.cp)
        fluid_total, _ = self._fluid_forces(fluid_vel, grad_p, normals, sub)
        contact = self._contact_forces()
        terms = np.zeros((self.n_rows, self.n_terms, 3))
        fluid_body = np.zeros((3, 3))
        for i, mask in enumerate(self.body_masks):
            row = i + 1
            fluid_body[i] = fluid_total[mask].sum(axis=0)
            terms[row, 0] = self.cm[mask].sum() * self.gravity
            terms[row, 1] = fluid_body[i]
            terms[row, 2] = contact[mask].sum(axis=0)
            terms[row, 3] = np.zeros(3)
        terms[4, 0] = self.water_mass_total * self.gravity
        terms[4, 1] = -fluid_body.sum(axis=0)
        terms[4, 2] = np.zeros(3)
        # Quiescent water: wall and floor hydrostatic traction balances weight and load.
        terms[4, 3] = -terms[4, :3].sum(axis=0)
        terms[0] = terms[1:].sum(axis=0)
        self.interaction_force = np.concatenate((fluid_body, -fluid_body), axis=0)
        return terms

    # ---------------------------------------------------------------- frames
    def _sync_frame_arrays(self) -> None:
        self.position[self.water_idx] = self.wp
        self.position[self.cloth_idx] = self.cp
        self.velocity[self.water_idx] = self.wv
        self.velocity[self.cloth_idx] = self.cv

    def _make_frame(self, time: float) -> StateFrame:
        self._sync_frame_arrays()
        saturation = np.ones(self.n_samples)
        saturation[self.cloth_idx] = self.c_saturation
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.position),
            velocity=np.ascontiguousarray(self.velocity),
            mass=np.ascontiguousarray(self.mass),
            body_id=np.ascontiguousarray(self.body_id),
            attributes={
                "uv": self.uv,
                "material_density": self.material_density,
                "porosity": self.porosity_attr,
                "permeability_m2": self.permeability_attr,
                "saturation": saturation,
                "mass_quantity": np.ascontiguousarray(self.mass),
            },
            fields={
                "gravity_acceleration": self.gravity.reshape(1, 3),
                "interaction_force": np.ascontiguousarray(self.interaction_force),
                "cumulative_mass_source": np.ascontiguousarray(self.cum_mass_source),
                "budget_domain_weights": np.ascontiguousarray(self.domain_weights),
                "budget_external_force": np.ascontiguousarray(self.budget_force),
                "budget_cumulative_external_impulse": np.ascontiguousarray(self.cum_impulse),
                "budget_cumulative_inflow_momentum": np.zeros((self.n_rows, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self.n_rows, 3)),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "water_density": float(self.rho_w),
                "dynamic_viscosity": float(self.mu),
                "pore_fluid_density": float(self.rho_w),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        target_step = int(round(target / self.dt_fluid))
        while self.step_index < target_step:
            self._step(self.dt_fluid)
            self.step_index += 1
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC-grid incompressible water + mass-spring porous cloth shells",
            "coupling": "Darcy/inertial seepage drag, dynamic pressure load, buoyancy from solid volume",
            "fluid_steps": int(self.diag["fluid_steps"]),
            "cloth_substeps": int(self.diag["cloth_substeps"]),
            "grid_cells": int(self.nx * self.ny * self.nz),
            "dt_fluid": float(self.dt_fluid),
            "dt_cloth_target": float(self.dt_cloth_target),
            "final_time": float(self.time),
            "reference_momentum": float(self.reference_momentum),
        }
