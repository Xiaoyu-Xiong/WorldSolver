"""Buoyancy-driven smoke transport through multiple rigid obstacles.

Physical model
--------------
Incompressible (Boussinesq) Euler equations for the smoke-laden gas,

    du/dt + (u . grad) u = -grad p / rho0 + b * c * z_hat + f_conf
    div u                = 0
    dc/dt + (u . grad) c = s(x)

with ``c`` the transported smoke scalar, ``b`` the buoyancy coefficient and
``s`` the prescribed emitter source.  ``f_conf`` is the vorticity-confinement
force of Fedkiw et al. (Visual Simulation of Smoke, 2001), which restores the
small-scale vortical roll-up that a coarse grid cannot resolve.

Discretisation
--------------
* Staggered (MAC) grid, ``nx x ny x nz`` cells.
* Semi-Lagrangian advection (unconditionally stable) for all three face
  velocity components and for the smoke scalar.
* Cell-centred Boussinesq buoyancy / vorticity-confinement forces, distributed
  back onto the faces with the usual two-point average.
* Rigid obstacles and the enclosing box are masked out as solid cells; all
  wall faces carry zero normal velocity (impermeable) and a volume
  penalisation layer drives the tangential velocity to zero (no-slip).
* Variable-coefficient pressure Poisson equation with pure Neumann conditions
  on every solid/wall face, solved with a warm-started preconditioned
  conjugate-gradient iteration; the discrete system is compatible because the
  total divergence over the closed box telescopes to the (zero) wall fluxes.

Everything advances with a fixed physical time step, so the exported states are
deterministic and finite at every requested sample time.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg as _sparse_cg

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# physics / model constants (SI units)
# --------------------------------------------------------------------------
BUOYANCY = 0.85          # m s^-2 per unit smoke concentration (Boussinesq)
SMOKE_CAP = 2.0          # saturation of the transported scalar
SOURCE_RATE = 1.6        # 1/s volumetric source strength at the emitter
VORT_EPS = 0.12          # vorticity confinement strength
MAX_FORCE = 4.0          # m s^-2 clamp on the confinement acceleration
WALL_TAU = 0.10          # s, volume-penalisation time scale (no-slip walls)
VELOCITY_DAMP = 0.998    # per sub-step bulk (entrainment) drag
MAX_SPEED = 6.0          # m/s safety clamp on the velocity field
SUBSTEPS = 4             # physics sub-steps per exported sample interval

GRAVITY_UP = 9.81        # magnitude of gravity, used for the buoyancy scale


class BenchmarkSolver:
    """Deterministic Eulerian smoke solver for the multi-obstacle wake scene."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})

        times = [float(t) for t in scene.sample_times()]
        self.sample_times = times
        if len(times) > 1:
            base_dt = times[1] - times[0]
        else:
            base_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.dt = base_dt / float(SUBSTEPS)

        self._setup_grid(cfg)
        self._setup_sources(cfg)
        self._setup_initial_state(scene)
        self._setup_solid_and_faces(cfg)
        self._setup_boundary_samples()
        self._build_pressure_matrix()

        self._p_prev = np.zeros(self._n_cells, dtype=np.float64)
        self._step_index = 0
        self._time = 0.0
        self._last_speed = 0.0

    # ------------------------------------------------------------------
    # grid / geometry
    # ------------------------------------------------------------------
    def _setup_grid(self, cfg: dict) -> None:
        geometry = cfg.get("scene_geometry") or {}
        bounds_min = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        bounds_max = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        shape = (24, 16, 28)
        for obj in geometry.get("objects", []):
            geom = obj.get("geometry", {}) or {}
            if obj.get("semantic") == "gas" and geom.get("type") == "voxel_grid":
                if geom.get("bounds_min") and geom.get("bounds_max"):
                    bounds_min = np.asarray(geom["bounds_min"], dtype=np.float64)
                    bounds_max = np.asarray(geom["bounds_max"], dtype=np.float64)
                if geom.get("grid_shape"):
                    shape = tuple(int(v) for v in geom["grid_shape"])
        sampling = cfg.get("sampling", {}) or {}
        if sampling.get("field_grid_shape"):
            shape = tuple(int(v) for v in sampling["field_grid_shape"])

        self.nx, self.ny, self.nz = shape
        self.n_cells = self.nx * self.ny * self.nz
        self.origin = bounds_min.copy()
        self.high = bounds_max.copy()
        self.extent = self.high - self.origin
        self.dx = float(self.extent[0] / self.nx)
        self.dy = float(self.extent[1] / self.ny)
        self.dz = float(self.extent[2] / self.nz)
        self.h = np.array([self.dx, self.dy, self.dz])

        self.xc = self.origin[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = self.origin[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.zc = self.origin[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.xf = self.origin[0] + np.arange(self.nx + 1) * self.dx
        self.yf = self.origin[1] + np.arange(self.ny + 1) * self.dy
        self.zf = self.origin[2] + np.arange(self.nz + 1) * self.dz

        # cell-centre and face sample positions (cached for interpolation)
        self.cell_pos = self._mesh(self.xc, self.yc, self.zc)
        self.u_pos = self._mesh(self.xf, self.yc, self.zc)
        self.v_pos = self._mesh(self.xc, self.yf, self.zc)
        self.w_pos = self._mesh(self.xc, self.yc, self.zf)
        self.cell_index = np.arange(self.n_cells).reshape(self.nx, self.ny, self.nz)

        # diagnostics
        self.grid_spacing = float(np.cbrt(self.dx * self.dy * self.dz))
        self.characteristic_length = float(
            max(self.extent[0], self.extent[2]) * 0.25
        )
        self.reference_velocity = float(np.sqrt(GRAVITY_UP * 0.12))

    @staticmethod
    def _mesh(*axes) -> np.ndarray:
        g = np.meshgrid(*axes, indexing="ij")
        return np.stack(g, axis=-1).reshape(-1, 3)

    def _setup_sources(self, cfg: dict) -> None:
        """Emitter cylinder from the frozen scene geometry -> cell source mask."""
        geometry = cfg.get("scene_geometry") or {}
        centre = np.zeros(3, dtype=np.float64)
        radius = 0.32
        half_height = 0.05
        for obj in geometry.get("objects", []):
            if obj.get("semantic") != "emitter":
                continue
            geom = obj.get("geometry", {}) or {}
            trans = (obj.get("transform", {}) or {}).get("translation")
            if trans is not None:
                centre = np.asarray(trans, dtype=np.float64)
            if geom.get("radius"):
                radius = float(geom["radius"])
            if geom.get("height"):
                half_height = 0.5 * float(geom["height"])

        x = self.xc[:, None, None]
        y = self.yc[None, :, None]
        z = self.zc[None, None, :]
        radial = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
        top = centre[2] + half_height + 0.5 * self.dz
        inside = (radial <= radius) & (z <= top) & (z >= centre[2] - half_height - 0.5 * self.dz)
        self.source_mask = inside
        self.scalar_source = np.where(inside, SOURCE_RATE, 0.0)

    def _setup_initial_state(self, scene) -> None:
        init = scene.initial_state()
        scalar = np.asarray(init.fields.get("transported_scalar"), dtype=np.float64)
        if scalar.shape != (self.nx, self.ny, self.nz) or not np.isfinite(scalar).all():
            x = self.xc[:, None, None]
            y = self.yc[None, :, None]
            z = self.zc[None, None, :]
            scalar = 1.15 * np.exp(-((x / 0.34) ** 2 + (y / 0.34) ** 2)) * np.exp(
                -((z - 0.78) / 0.82) ** 2
            )
        self.c = np.clip(scalar, 0.0, SMOKE_CAP)

        position = np.asarray(init.position, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or len(position) == 0:
            seeds = self.cell_pos[self.cell_index[self.source_mask]]
            position = seeds if len(seeds) else self.cell_pos[:1].copy()
        self.ppos = position.copy()
        self.pvel = np.zeros_like(self.ppos)
        n = len(self.ppos)
        mass = np.asarray(getattr(init, "mass", None), dtype=np.float64)
        if mass.shape != (n,) or not np.isfinite(mass).all():
            mass = np.full(n, 1.2 * self.grid_spacing ** 3)
        self.pmass = mass.copy()
        body = np.asarray(getattr(init, "body_id", None))
        if body.shape != (n,):
            body = np.zeros(n, dtype=np.int32)
        self.pbody = body.astype(np.int32)
        spacing = cfg_spacing(scene)
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = self.grid_spacing
        self.pradius = np.full(n, 0.45 * float(spacing))
        self.plabel = self.pbody.copy()
        span = float(np.ptp(self.ppos, axis=0).max()) if n > 1 else 0.0
        self.characteristic_length = float(max(span, self.grid_spacing))

    def _setup_solid_and_faces(self, cfg: dict) -> None:
        solid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        geometry = cfg.get("scene_geometry") or {}
        for obj in geometry.get("objects", []):
            geom = obj.get("geometry", {}) or {}
            if geom.get("type") != "box":
                continue
            if obj.get("semantic") != "boundary" or not obj.get("collision_enabled", True):
                continue
            half = np.asarray(geom.get("half_extents", [0.0, 0.0, 0.0]), dtype=np.float64)
            trans = np.asarray(
                (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            lo = trans - half
            hi = trans + half
            inside = (
                (self.xc[:, None, None] > lo[0])
                & (self.xc[:, None, None] < hi[0])
                & (self.yc[None, :, None] > lo[1])
                & (self.yc[None, :, None] < hi[1])
                & (self.zc[None, None, :] > lo[2])
                & (self.zc[None, None, :] < hi[2])
            )
            solid |= inside
        self.solid = solid
        self.fluid = ~solid

        # face validity: a face is open only between two fluid cells
        u_ok = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        u_ok[1:-1] = self.fluid[:-1] & self.fluid[1:]
        v_ok = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        v_ok[:, 1:-1] = self.fluid[:, :-1] & self.fluid[:, 1:]
        w_ok = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        w_ok[:, :, 1:-1] = self.fluid[:, :, :-1] & self.fluid[:, :, 1:]
        self.u_ok, self.v_ok, self.w_ok = u_ok, v_ok, w_ok

        # wall (penalisation) layer: fluid cells touching a solid or the box
        layer = np.zeros_like(self.fluid)
        inner = self.fluid.copy()
        layer[1:, :, :] |= inner[:-1] & ~self.fluid[1:]
        layer[:-1, :, :] |= inner[1:] & ~self.fluid[:-1]
        layer[:, 1:, :] |= inner[:, :-1] & ~self.fluid[:, 1:]
        layer[:, :-1, :] |= inner[:, 1:] & ~self.fluid[:, :-1]
        layer[:, :, 1:] |= inner[:, :, :-1] & ~self.fluid[:, :, 1:]
        layer[:, :, :-1] |= inner[:, :, 1:] & ~self.fluid[:, :, :-1]
        # domain walls (the outer cells of the box are always adjacent to a wall)
        layer[0, :, :] = True
        layer[-1, :, :] = True
        layer[:, 0, :] = True
        layer[:, -1, :] = True
        layer[:, :, 0] = True
        layer[:, :, -1] = True
        self.wall_layer = layer & self.fluid

        factor = 1.0 / (1.0 + self.dt / WALL_TAU)
        self.u_damp = np.ones_like(u_ok, dtype=np.float64)
        fu = np.zeros_like(self.u_damp)
        fu[1:-1] = self.wall_layer[:-1] | self.wall_layer[1:]
        self.u_damp = np.where(fu & u_ok, factor, 1.0)
        fv = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        fv[:, 1:-1] = self.wall_layer[:, :-1] | self.wall_layer[:, 1:]
        self.v_damp = np.where(fv & v_ok, factor, 1.0)
        fw = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)
        fw[:, :, 1:-1] = self.wall_layer[:, :, :-1] | self.wall_layer[:, :, 1:]
        self.w_damp = np.where(fw & w_ok, factor, 1.0)

        self.u = np.zeros((self.nx + 1, self.ny, self.nz), dtype=np.float64)
        self.v = np.zeros((self.nx, self.ny + 1, self.nz), dtype=np.float64)
        self.w = np.zeros((self.nx, self.ny, self.nz + 1), dtype=np.float64)
        # the transported scalar has no meaning inside rigid obstacles
        self.c[~self.fluid] = 0.0

    def _setup_boundary_samples(self) -> None:
        normals = []
        points = []
        # enclosing box walls
        for j in range(self.ny):
            for k in range(self.nz):
                points.append((self.xf[0], self.yc[j], self.zc[k]))
                normals.append((-1.0, 0.0, 0.0))
                points.append((self.xf[-1], self.yc[j], self.zc[k]))
                normals.append((1.0, 0.0, 0.0))
        for i in range(self.nx):
            for k in range(self.nz):
                points.append((self.xc[i], self.yf[0], self.zc[k]))
                normals.append((0.0, -1.0, 0.0))
                points.append((self.xc[i], self.yf[-1], self.zc[k]))
                normals.append((0.0, 1.0, 0.0))
        for i in range(self.nx):
            for j in range(self.ny):
                points.append((self.xc[i], self.yc[j], self.zf[0]))
                normals.append((0.0, 0.0, -1.0))
                points.append((self.xc[i], self.yc[j], self.zf[-1]))
                normals.append((0.0, 0.0, 1.0))
        # fluid/solid interfaces
        solid = self.solid
        for axis in range(3):
            if axis == 0:
                pos = self.u_pos.reshape(self.nx + 1, self.ny, self.nz, 3)
            elif axis == 1:
                pos = self.v_pos.reshape(self.nx, self.ny + 1, self.nz, 3)
            else:
                pos = self.w_pos.reshape(self.nx, self.ny, self.nz + 1, 3)
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[axis] = slice(0, -1)
            sl_hi[axis] = slice(1, None)
            lo_fluid = self.fluid[tuple(sl_lo)]
            hi_fluid = self.fluid[tuple(sl_hi)]
            lo_solid = solid[tuple(sl_lo)]
            hi_solid = solid[tuple(sl_hi)]
            # face between cell n-1 (lo, fluid) and cell n (hi, solid) -> +normal
            mask = lo_fluid & hi_solid
            idx = np.nonzero(mask)
            if idx[0].size:
                face_idx = list(idx)
                face_idx[axis] = idx[axis] + 1
                pts = pos[tuple(face_idx)]
                points.extend(map(tuple, pts))
                normals.extend([self._unit(axis)] * len(pts))
            mask = hi_fluid & lo_solid
            idx = np.nonzero(mask)
            if idx[0].size:
                pts = pos[tuple(idx)]
                points.extend(map(tuple, pts))
                normals.extend([self._unit(axis, -1.0)] * len(pts))
        self.bpos = np.asarray(points, dtype=np.float64)
        self.bnorm = np.asarray(normals, dtype=np.float64)
        self.btarget = np.zeros_like(self.bpos)

    @staticmethod
    def _unit(axis: int, sign: float = 1.0):
        n = [0.0, 0.0, 0.0]
        n[axis] = float(sign)
        return tuple(n)

    # ------------------------------------------------------------------
    # pressure operator
    # ------------------------------------------------------------------
    def _build_pressure_matrix(self) -> None:
        """Seven-point variable-coefficient Laplacian with Neumann wall faces.

        Solid cells receive identity rows (their pressure stays zero and is
        never referenced), while fluid cells couple only across faces open to
        another fluid cell -- exactly the zero-normal-flux (impermeable)
        condition.  The closed-box Neumann operator has a constant null space;
        the discrete divergence telescopes to the (zero) wall fluxes so the
        system is compatible, and a tiny positive shift makes it SPD for CG.
        """
        ax = 1.0 / (self.dx * self.dx)
        ay = 1.0 / (self.dy * self.dy)
        az = 1.0 / (self.dz * self.dz)
        coeffs = (ax, ay, az)
        fluid = self.fluid
        flat = self.cell_index
        everything = np.arange(self.n_cells)

        diag = np.zeros(self.n_cells, dtype=np.float64)
        rows = [everything]
        cols = [everything]
        vals = [np.where(fluid.reshape(-1), 0.0, 1.0)]

        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, -1)
            hi[axis] = slice(1, None)
            open_face = fluid[tuple(lo)] & fluid[tuple(hi)]
            idx_lo = flat[tuple(lo)][open_face]
            idx_hi = flat[tuple(hi)][open_face]
            diag[idx_lo] += coeffs[axis]
            diag[idx_hi] += coeffs[axis]
            for a, b in ((idx_lo, idx_hi), (idx_hi, idx_lo)):
                rows.append(a)
                cols.append(b)
                vals.append(np.full(a.size, -coeffs[axis]))

        rows.append(everything)
        cols.append(everything)
        vals.append(np.where(fluid.reshape(-1), diag, 0.0))

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_cells, self.n_cells),
        ).tocsr()
        eps = 1e-5 * max(ax, ay, az)
        self._regularisation = eps
        self.A = (A + eps * sp.identity(self.n_cells, format="csr")).tocsr()

    # ------------------------------------------------------------------
    # interpolation
    # ------------------------------------------------------------------
    def _sample(self, field: np.ndarray, pos: np.ndarray, node=(False, False, False)):
        """Trilinear sample of ``field`` at world positions ``pos`` (M,3)."""
        idx0 = []
        t = []
        for axis in range(3):
            n = field.shape[axis]
            g = (pos[:, axis] - self.origin[axis]) / self.h[axis]
            if not node[axis]:
                g = g - 0.5
            i0 = np.clip(np.floor(g).astype(np.int64), 0, n - 2)
            idx0.append(i0)
            t.append(np.clip(g - i0, 0.0, 1.0))
        i, j, k = idx0
        tx, ty, tz = t
        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w100 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w110 = tx * ty * (1.0 - tz)
        w001 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w011 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz
        if field.ndim == 4:
            w000 = w000[:, None]
            w100 = w100[:, None]
            w010 = w010[:, None]
            w110 = w110[:, None]
            w001 = w001[:, None]
            w101 = w101[:, None]
            w011 = w011[:, None]
            w111 = w111[:, None]
        out = (
            field[i, j, k] * w000
            + field[i + 1, j, k] * w100
            + field[i, j + 1, k] * w010
            + field[i + 1, j + 1, k] * w110
            + field[i, j, k + 1] * w001
            + field[i + 1, j, k + 1] * w101
            + field[i, j + 1, k + 1] * w011
            + field[i + 1, j + 1, k + 1] * w111
        )
        return out

    def _collocated(self) -> np.ndarray:
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return np.stack((uc, vc, wc), axis=-1)

    @staticmethod
    def _faces_from_cells(component: np.ndarray, axis: int) -> np.ndarray:
        """Average a cell-centred component onto the faces of ``axis``."""
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        out_shape = list(component.shape)
        out_shape[axis] += 1
        out = np.zeros(out_shape, dtype=np.float64)
        out[tuple(hi)] += 0.5 * component[tuple(lo)]
        out[tuple(lo)] += 0.5 * component[tuple(hi)]
        return out

    # ------------------------------------------------------------------
    # physics step
    # ------------------------------------------------------------------
    def _advance(self, dt: float) -> None:
        vel = self._collocated()

        # 1. semi-Lagrangian advection of the face velocities
        back = self.u_pos - dt * self._sample(vel, self.u_pos)
        self.u[...] = self._sample(self.u, back, node=(True, False, False)).reshape(self.u.shape)
        back = self.v_pos - dt * self._sample(vel, self.v_pos)
        self.v[...] = self._sample(self.v, back, node=(False, True, False)).reshape(self.v.shape)
        back = self.w_pos - dt * self._sample(vel, self.w_pos)
        self.w[...] = self._sample(self.w, back, node=(False, False, True)).reshape(self.w.shape)
        self._apply_wall_faces()

        vel = self._collocated()

        # 2. Boussinesq buoyancy + vorticity confinement (cell centred)
        force = np.zeros_like(vel)
        force[:, :, :, 2] = BUOYANCY * self.c
        force += self._confinement(vel)
        mag = np.linalg.norm(force, axis=-1, keepdims=True)
        force = np.where(mag > MAX_FORCE, force * (MAX_FORCE / np.maximum(mag, 1e-12)), force)

        self.u += dt * self._faces_from_cells(force[:, :, :, 0], 0)
        self.v += dt * self._faces_from_cells(force[:, :, :, 1], 1)
        self.w += dt * self._faces_from_cells(force[:, :, :, 2], 2)
        self._apply_wall_faces()

        # 3. volume penalisation: no-slip drag inside the wall layer
        self.u *= self.u_damp
        self.v *= self.v_damp
        self.w *= self.w_damp
        self._apply_wall_faces()

        # weak bulk drag so the closed box reaches a bounded plume speed
        self.u *= VELOCITY_DAMP
        self.v *= VELOCITY_DAMP
        self.w *= VELOCITY_DAMP
        self._clip_speed()

        # 4. pressure projection -> discretely divergence-free velocity
        self._project(dt)

        # 5. smoke transport: advection by the corrected velocity, then source
        vel = self._collocated()
        back = self.cell_pos - dt * self._sample(vel, self.cell_pos)
        self.c = self._sample(self.c, back)
        self.c = np.clip(self.c + dt * self.scalar_source, 0.0, SMOKE_CAP)
        self.c[~self.fluid] = 0.0

        # 6. passive tracer particles
        self._advect_particles(dt, vel)

    def _confinement(self, vel: np.ndarray) -> np.ndarray:
        wx = np.gradient(vel[:, :, :, 2], self.yc, axis=1) - np.gradient(
            vel[:, :, :, 1], self.zc, axis=2
        )
        wy = np.gradient(vel[:, :, :, 0], self.zc, axis=2) - np.gradient(
            vel[:, :, :, 2], self.xc, axis=0
        )
        wz = np.gradient(vel[:, :, :, 1], self.xc, axis=0) - np.gradient(
            vel[:, :, :, 0], self.yc, axis=1
        )
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx = np.gradient(mag, self.xc, axis=0)
        gy = np.gradient(mag, self.yc, axis=1)
        gz = np.gradient(mag, self.zc, axis=2)
        gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
        safe = gmag > 1e-12
        nx = np.where(safe, gx / np.maximum(gmag, 1e-12), 0.0)
        ny = np.where(safe, gy / np.maximum(gmag, 1e-12), 0.0)
        nz = np.where(safe, gz / np.maximum(gmag, 1e-12), 0.0)
        scale = VORT_EPS * self.grid_spacing
        out = np.empty_like(vel)
        out[:, :, :, 0] = scale * (ny * wz - nz * wy)
        out[:, :, :, 1] = scale * (nz * wx - nx * wz)
        out[:, :, :, 2] = scale * (nx * wy - ny * wx)
        return out

    def _apply_wall_faces(self) -> None:
        self.u = np.where(self.u_ok, self.u, 0.0)
        self.v = np.where(self.v_ok, self.v, 0.0)
        self.w = np.where(self.w_ok, self.w, 0.0)

    def _clip_speed(self) -> None:
        for arr in (self.u, self.v, self.w):
            np.clip(arr, -MAX_SPEED, MAX_SPEED, out=arr)

    def _project(self, dt: float) -> None:
        div = (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / self.dx
            + (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz
        )
        rhs = div.reshape(-1) / dt
        rhs[~self.fluid.reshape(-1)] = 0.0
        try:
            p, info = _sparse_cg(
                self.A, rhs, x0=self._p_prev, rtol=1e-7, atol=0.0, maxiter=300
            )
        except TypeError:  # pragma: no cover - older SciPy
            p, info = _sparse_cg(
                self.A, rhs, x0=self._p_prev, tol=1e-7, atol=0.0, maxiter=300
            )
        if not np.isfinite(p).all():
            p = np.zeros_like(rhs)
        self._p_prev = p
        p = p.reshape(self.nx, self.ny, self.nz)

        self.u[1:-1, :, :] -= dt * (p[1:, :, :] - p[:-1, :, :]) / self.dx
        self.v[:, 1:-1, :] -= dt * (p[:, 1:, :] - p[:, :-1, :]) / self.dy
        self.w[:, :, 1:-1] -= dt * (p[:, :, 1:] - p[:, :, :-1]) / self.dz
        self._apply_wall_faces()

    def _advect_particles(self, dt: float, vel: np.ndarray) -> None:
        k1 = self._sample(vel, self.ppos)
        mid = self.ppos + 0.5 * dt * k1
        k2 = self._sample(vel, mid)
        cand = self.ppos + dt * k2
        lo = self.origin + 1e-4
        hi = self.high - 1e-4
        cand = np.clip(cand, lo, hi)
        inside = self._solid_at(cand)
        cand = np.where(inside[:, None], self.ppos, cand)
        self.pvel = (cand - self.ppos) / dt
        self.ppos = cand

    def _solid_at(self, pos: np.ndarray) -> np.ndarray:
        idx = np.floor((pos - self.origin) / self.h).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.nz - 1)
        return self.solid[idx[:, 0], idx[:, 1], idx[:, 2]]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        target = max(target, self._step_index)
        while self._step_index < target:
            self._advance(self.dt)
            self._step_index += 1
            self._time += self.dt
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        vel = self._collocated()
        bvel = self._sample(vel, self.bpos)
        self._last_speed = float(np.max(np.linalg.norm(vel, axis=-1)))
        fields = {
            "active_mask": self.fluid.copy(),
            "transported_scalar": self.c.copy(),
            "scalar_source": self.scalar_source.copy(),
            "velocity_field": vel,
            "boundary_normal": self.bnorm.copy(),
            "boundary_target_velocity": self.btarget.copy(),
            "boundary_velocity": bvel,
        }
        attributes = {
            "mass_quantity": self.pmass.copy(),
            "particle_radius": self.pradius.copy(),
            "material_label": self.plabel.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], len(self.ppos), axis=0),
        }
        scalars = {
            "grid_spacing": self.grid_spacing,
            "characteristic_length": self.characteristic_length,
            "reference_velocity": self.reference_velocity,
            "contact_length_scale": self.grid_spacing,
        }
        frame = StateFrame(
            time=float(time),
            position=self.ppos.copy(),
            velocity=self.pvel.copy(),
            mass=self.pmass.copy(),
            body_id=self.pbody.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self._step_index),
            "sim_time": float(self._time),
            "dt": float(self.dt),
            "max_speed": float(self._last_speed),
            "max_scalar": float(self.c.max()),
            "fluid_cells": int(self.fluid.sum()),
            "solid_cells": int(self.solid.sum()),
            "boundary_samples": int(len(self.bpos)),
            "particles": int(len(self.ppos)),
        }


def cfg_spacing(scene) -> float:
    sampling = (getattr(scene, "config", {}) or {}).get("sampling", {}) or {}
    try:
        return float(sampling.get("spacing"))
    except (TypeError, ValueError):
        return 0.0
