"""Turbulent stream through complex obstacles: physically-stepped fluid solver.

The smoke stream is represented by a collocated transported scalar that is
advected by a divergence-free velocity field.  The velocity field lives on a
MAC (staggered) grid and is advanced with an explicit time-stepping sequence:

  body forces (buoyancy, wall drag, vorticity confinement)
      -> semi-Lagrangian self-advection of momentum
      -> discrete pressure projection (sparse Poisson solve) enforcing
         div(u) = 0 while honouring the frozen solid geometry

The frozen obstacles act as no-penetration / no-slip solid cells, the ground
plane is a solid wall, the inlet nozzle prescribes a physical mass flux through
the upstream boundary, and all other domain faces are open outflow boundaries.
Nothing is key-framed: every frame is produced by integrating the model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import cg

from worldsolver.contracts import StateFrame

_INLET_SPEED = 1.20          # m/s axial velocity prescribed by the nozzle
_DT_MAX = 0.02               # maximum internal physical time step (s)
_BUOYANCY = 0.06             # m/s^2 per unit of transported scalar (warm smoke)
_VORT_EPS = 0.08             # vorticity-confinement strength (grid scale)
_WALL_DRAG_RATE = 12.0       # 1/s relaxation of tangential velocity at walls
_SOURCE_RATE = 3.0           # 1/s scalar injection inside the nozzle volume
_SOURCE_CAP = 1.20           # scalar saturation value
_CG_RTOL = 1.0e-4
_CG_MAXITER = 220


def _rotation_from_quaternion(q) -> np.ndarray:
    """Convert an xyzw quaternion into a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


class BenchmarkSolver:
    """MAC-grid Eulerian solver for a turbulent smoke stream past obstacles."""

    def __init__(self) -> None:
        self._initialized = False
        self.steps = 0
        self.cg_iterations = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = getattr(scene, "config", None)
        self.config = config if isinstance(config, dict) else {}
        geometry = self.config.get("scene_geometry")
        if not isinstance(geometry, dict):
            geometry = self._load_geometry_file()
        self.geometry = geometry if isinstance(geometry, dict) else {"objects": []}
        frame0 = scene.initial_state()
        self._setup_domain(frame0)
        self._setup_state(frame0)
        self.time = 0.0
        self.steps = 0
        self._initialized = True

    def _load_geometry_file(self):
        try:
            from pathlib import Path

            import yaml

            path = Path("scene_geometry.yaml")
            if path.is_file():
                return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"objects": []}

    def _mark_solid(self, solid, objects):
        for obj in objects:
            if obj.get("semantic") not in ("boundary", "rigid", "deformable"):
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            geom = obj.get("geometry") or {}
            transform = obj.get("transform") or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), float)
            rot = _rotation_from_quaternion(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            rel = np.stack(
                [self.X - center[0], self.Y - center[1], self.Z - center[2]], axis=0
            )
            # local coordinates: rotate the world offset back into the body frame
            local = np.einsum("ij,jxyz->ixyz", rot.T, rel)
            kind = geom.get("type")
            if kind == "box":
                half = np.asarray(geom.get("half_extents") or [0.0, 0.0, 0.0], float)
                inside = (
                    (np.abs(local[0]) <= half[0] + 1e-12)
                    & (np.abs(local[1]) <= half[1] + 1e-12)
                    & (np.abs(local[2]) <= half[2] + 1e-12)
                )
                solid |= inside
            elif kind in ("sphere", "capsule"):
                radius = float(geom.get("radius") or 0.0)
                distance = np.linalg.norm(local, axis=0)
                solid |= distance <= radius + 1e-12
            elif kind == "cylinder":
                radius = float(geom.get("radius") or 0.0)
                height = float(geom.get("height") or 0.0)
                radial = np.sqrt(local[0] ** 2 + local[1] ** 2)
                solid |= (radial <= radius + 1e-12) & (np.abs(local[2]) <= 0.5 * height + 1e-12)
        return solid

    def _mark_inlet(self, objects):
        inlet = np.zeros(self.shape, bool)
        axis = None
        for obj in objects:
            if obj.get("semantic") != "emitter":
                continue
            geom = obj.get("geometry") or {}
            if geom.get("type") != "cylinder":
                continue
            transform = obj.get("transform") or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), float)
            rot = _rotation_from_quaternion(
                transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            )
            axis = rot[:, 2]
            radius = float(geom.get("radius") or 0.4)
            height = float(geom.get("height") or 0.1)
            rel = np.stack(
                [self.X - center[0], self.Y - center[1], self.Z - center[2]], axis=0
            )
            axial = axis[0] * rel[0] + axis[1] * rel[1] + axis[2] * rel[2]
            radial_sq = (rel * rel).sum(axis=0) - axial * axial
            radial_sq = np.maximum(radial_sq, 0.0)
            axial_tol = 0.5 * height + 0.6 * self.dx
            radius_tol = radius + 0.3 * self.hmin
            inlet |= (np.abs(axial) <= axial_tol) & (radial_sq <= radius_tol * radius_tol)
            break
        if axis is None:
            axis = np.array([1.0, 0.0, 0.0])
        inlet &= self.fluid
        if not inlet.any():
            # Fallback: thin patch centred on the upstream face of the domain.
            i0 = 0
            mid_y = 0.5 * (self.bmin[1] + self.bmax[1])
            mid_z = 0.5 * (self.bmin[2] + self.bmax[2])
            patch = (
                (np.abs(self.Y - mid_y) <= 0.25 * (self.bmax[1] - self.bmin[1]))
                & (np.abs(self.Z - mid_z) <= 0.12 * (self.bmax[2] - self.bmin[2]))
            )
            fallback = np.zeros(self.shape, bool)
            fallback[i0] = patch[i0]
            inlet |= fallback & self.fluid
        return inlet, axis / (np.linalg.norm(axis) + 1e-12)

    def _setup_domain(self, frame0) -> None:
        objects = self.geometry.get("objects") or []
        shape = None
        bmin = None
        bmax = None
        for obj in objects:
            geom = obj.get("geometry") or {}
            if (
                geom.get("type") == "voxel_grid"
                and geom.get("bounds_min")
                and geom.get("bounds_max")
                and geom.get("grid_shape")
            ):
                bmin = np.asarray(geom["bounds_min"], float)
                bmax = np.asarray(geom["bounds_max"], float)
                shape = tuple(int(v) for v in geom["grid_shape"])
                break
        if shape is None or bmin is None or bmax is None:
            sampling = self.config.get("sampling") or {}
            shape = tuple(int(v) for v in sampling.get("field_grid_shape", (24, 16, 28)))
            bmin = np.array([-3.0, -1.2, 0.0], float)
            bmax = np.array([3.0, 1.2, 3.8], float)
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        nx, ny, nz = self.shape
        self.bmin = np.asarray(bmin, float)
        self.bmax = np.asarray(bmax, float)
        self.dx = float((self.bmax[0] - self.bmin[0]) / nx)
        self.dy = float((self.bmax[1] - self.bmin[1]) / ny)
        self.dz = float((self.bmax[2] - self.bmin[2]) / nz)
        self.hmin = min(self.dx, self.dy, self.dz)
        cx = self.bmin[0] + (np.arange(nx) + 0.5) * self.dx
        cy = self.bmin[1] + (np.arange(ny) + 0.5) * self.dy
        cz = self.bmin[2] + (np.arange(nz) + 0.5) * self.dz
        self.X = cx[:, None, None]
        self.Y = cy[None, :, None]
        self.Z = cz[None, None, :]

        solid = np.zeros(self.shape, bool)
        solid = self._mark_solid(solid, objects)
        self.solid = solid
        self.fluid = ~solid
        self.inlet, self.inlet_axis = self._mark_inlet(objects)
        self.source = np.where(self.inlet, _SOURCE_RATE, 0.0)

        # Wall-adjacency masks (used for the tangential drag force).
        wx = np.zeros(self.shape, bool)
        wx[:-1, :, :] |= solid[1:, :, :]
        wx[1:, :, :] |= solid[:-1, :, :]
        wy = np.zeros(self.shape, bool)
        wy[:, :-1, :] |= solid[:, 1:, :]
        wy[:, 1:, :] |= solid[:, :-1, :]
        wz = np.zeros(self.shape, bool)
        wz[:, :, :-1] |= solid[:, :, 1:]
        wz[:, :, 1:] |= solid[:, :, :-1]
        wz[:, :, 0] = True  # ground plane contact
        self.wall_x = wx & self.fluid
        self.wall_y = wy & self.fluid
        self.wall_z = wz & self.fluid

        # Index-space sampling coordinates for trilinear interpolation.
        ii, jj, kk = np.meshgrid(
            np.arange(nx, dtype=float),
            np.arange(ny, dtype=float),
            np.arange(nz, dtype=float),
            indexing="ij",
        )
        self._coords = np.stack([ii, jj, kk], axis=0)

        self._build_pressure_system()
        self._build_boundary_samples(objects)

    def _build_pressure_system(self) -> None:
        nx, ny, nz = self.shape
        fluid = self.fluid
        self._fluid_flat = np.flatnonzero(fluid.reshape(-1))
        nfluid = int(self._fluid_flat.size)
        index = -np.ones(nx * ny * nz, dtype=np.int64)
        index[self._fluid_flat] = np.arange(nfluid, dtype=np.int64)
        self._index = index
        inv = (1.0 / (self.dx * self.dx), 1.0 / (self.dy * self.dy), 1.0 / (self.dz * self.dz))
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not fluid[i, j, k]:
                        continue
                    cell = int(index[i, j, k])
                    diag = 0.0
                    for axis in range(3):
                        weight = inv[axis]
                        for direction in (-1, 1):
                            ni, nj, nk = i, j, k
                            if axis == 0:
                                ni += direction
                            elif axis == 1:
                                nj += direction
                            else:
                                nk += direction
                            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                                if axis == 0 and direction == -1 and self.inlet[i, j, k]:
                                    continue  # prescribed inflow flux (Neumann)
                                if axis == 2 and direction == -1:
                                    continue  # solid ground (Neumann)
                                diag += weight  # open boundary: p = 0 (Dirichlet)
                            elif not fluid[ni, nj, nk]:
                                continue  # rigid wall (Neumann)
                            else:
                                rows.append(cell)
                                cols.append(int(index[ni, nj, nk]))
                                vals.append(-weight)
                                diag += weight
                    rows.append(cell)
                    cols.append(cell)
                    vals.append(diag)
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(nfluid, nfluid))
        self._A = matrix
        self._M = sparse.diags(1.0 / np.maximum(matrix.diagonal(), 1e-30))
        self._p_prev = np.zeros(nfluid, dtype=float)

        face_x = fluid[:-1, :, :] & fluid[1:, :, :]
        face_y = fluid[:, :-1, :] & fluid[:, 1:, :]
        face_z = fluid[:, :, :-1] & fluid[:, :, 1:]
        self._face_x = face_x
        self._face_y = face_y
        self._face_z = face_z
        self._nfluid = nfluid

    def _build_boundary_samples(self, objects) -> None:
        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        lo = self.bmin - 1e-6 * self.hmin
        hi = self.bmax + 1e-6 * self.hmin
        for obj in objects:
            if obj.get("semantic") not in ("boundary", "rigid", "deformable"):
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            geom = obj.get("geometry") or {}
            if geom.get("type") != "box":
                continue
            half = np.asarray(geom.get("half_extents") or [0.0, 0.0, 0.0], float)
            transform = obj.get("transform") or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), float)
            rot = _rotation_from_quaternion(
                transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            )
            spacing = 0.19 if float(np.max(half)) < 1.0 else 0.30
            for axis in range(3):
                others = [a for a in range(3) if a != axis]
                grid = []
                for a in others:
                    extent = float(half[a])
                    samples = np.arange(-extent + 0.5 * spacing, extent, spacing)
                    if samples.size == 0:
                        samples = np.zeros(1)
                    grid.append(samples)
                mesh_a, mesh_b = np.meshgrid(grid[0], grid[1], indexing="ij")
                for sign in (-1.0, 1.0):
                    local = np.zeros((mesh_a.size, 3))
                    local[:, axis] = sign * half[axis]
                    local[:, others[0]] = mesh_a.reshape(-1)
                    local[:, others[1]] = mesh_b.reshape(-1)
                    world = local @ rot.T + center
                    normal_local = np.zeros(3)
                    normal_local[axis] = sign
                    normal = rot @ normal_local
                    inside = np.all(world >= lo, axis=1) & np.all(world <= hi, axis=1)
                    if np.any(inside):
                        points.append(world[inside])
                        normals.append(np.repeat(normal[None, :], int(inside.sum()), axis=0))
        if points:
            self.boundary_points = np.concatenate(points, axis=0)
            self.boundary_normals = np.concatenate(normals, axis=0)
            norms = np.linalg.norm(self.boundary_normals, axis=1, keepdims=True)
            self.boundary_normals = self.boundary_normals / np.maximum(norms, 1e-12)
        else:
            self.boundary_points = np.zeros((1, 3))
            self.boundary_normals = np.array([[0.0, 0.0, 1.0]])
        self.boundary_targets = np.zeros_like(self.boundary_points)

    def _setup_state(self, frame0) -> None:
        nx, ny, nz = self.shape
        self.uc = np.zeros((nx, ny, nz))
        self.vc = np.zeros((nx, ny, nz))
        self.wc = np.zeros((nx, ny, nz))
        smoke = frame0.fields.get("transported_scalar")
        if smoke is not None and np.asarray(smoke).shape == (nx, ny, nz):
            self.smoke = np.array(smoke, dtype=float)
        else:
            self.smoke = np.zeros((nx, ny, nz))
        self.smoke = np.clip(self.smoke, 0.0, _SOURCE_CAP)
        self.smoke[self.solid] = 0.0

        self.p_position = np.array(frame0.position, dtype=float)
        if self.p_position.ndim != 2 or self.p_position.shape[1] != 3:
            self.p_position = np.zeros((0, 3))
        self.p_velocity = np.zeros_like(self.p_position)
        self.p_mass = (
            np.array(frame0.mass, dtype=float)
            if frame0.mass is not None
            else np.ones(len(self.p_position))
        )
        self.p_body = (
            np.array(frame0.body_id, dtype=np.int32)
            if frame0.body_id is not None
            else np.zeros(len(self.p_position), dtype=np.int32)
        )
        attrs = frame0.attributes or {}
        self.p_attributes: dict[str, np.ndarray] = {}
        n = len(self.p_position)
        for name, value in attrs.items():
            array = np.asarray(value)
            if array.shape[0] == n:
                self.p_attributes[name] = np.array(array, dtype=float if array.dtype.kind == "f" else array.dtype)
        if "mass_quantity" not in self.p_attributes:
            self.p_attributes["mass_quantity"] = self.p_mass.copy()
        if "particle_radius" not in self.p_attributes:
            self.p_attributes["particle_radius"] = np.full(n, 0.45 * self.hmin)
        if "material_label" not in self.p_attributes:
            self.p_attributes["material_label"] = self.p_body.copy()
        if "deformation_gradient" not in self.p_attributes:
            self.p_attributes["deformation_gradient"] = np.repeat(np.eye(3)[None], n, axis=0)

        scalars = frame0.scalars or {}
        self.characteristic_length = float(
            scalars.get("characteristic_length", float(np.max(self.bmax - self.bmin)))
        )
        self.grid_spacing = float(scalars.get("grid_spacing", float(self.hmin)))
        self.reference_velocity = float(scalars.get("reference_velocity", _INLET_SPEED))

    # ------------------------------------------------------------------- math
    def _advect(self, fields, uc, vc, wc, dt):
        coords = self._coords
        sample = np.stack(
            [
                coords[0] - dt * uc / self.dx,
                coords[1] - dt * vc / self.dy,
                coords[2] - dt * wc / self.dz,
            ],
            axis=0,
        )
        return [
            map_coordinates(f, sample, order=1, mode="nearest", prefilter=False)
            for f in fields
        ]

    def _sample_vector(self, points):
        if len(points) == 0:
            return np.zeros((0, 3))
        index = np.stack(
            [
                (points[:, 0] - self.bmin[0]) / self.dx - 0.5,
                (points[:, 1] - self.bmin[1]) / self.dy - 0.5,
                (points[:, 2] - self.bmin[2]) / self.dz - 0.5,
            ],
            axis=0,
        )
        out = np.empty((len(points), 3))
        for column, field in enumerate((self.uc, self.vc, self.wc)):
            out[:, column] = map_coordinates(
                field, index, order=1, mode="nearest", prefilter=False
            )
        return out

    def _vorticity_force(self, uc, vc, wc):
        dw_dy = np.gradient(wc, self.dy, axis=1)
        dv_dz = np.gradient(vc, self.dz, axis=2)
        du_dz = np.gradient(uc, self.dz, axis=2)
        dw_dx = np.gradient(wc, self.dx, axis=0)
        dv_dx = np.gradient(vc, self.dx, axis=0)
        du_dy = np.gradient(uc, self.dy, axis=1)
        wx = dw_dy - dv_dz
        wy = du_dz - dw_dx
        wz = dv_dx - du_dy
        magnitude = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-12
        gx = np.gradient(magnitude, self.dx, axis=0)
        gy = np.gradient(magnitude, self.dy, axis=1)
        gz = np.gradient(magnitude, self.dz, axis=2)
        gnorm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        nx_ = gx / gnorm
        ny_ = gy / gnorm
        nz_ = gz / gnorm
        scale = _VORT_EPS * self.hmin
        fx = scale * (ny_ * wz - nz_ * wy)
        fy = scale * (nz_ * wx - nx_ * wz)
        fz = scale * (nx_ * wy - ny_ * wx)
        limit = 2.0
        return (
            np.clip(fx, -limit, limit),
            np.clip(fy, -limit, limit),
            np.clip(fz, -limit, limit),
        )

    def _faces_from_cc(self, uc, vc, wc, inlet_speed):
        nx, ny, nz = self.shape
        u = np.zeros((nx + 1, ny, nz))
        v = np.zeros((nx, ny + 1, nz))
        w = np.zeros((nx, ny, nz + 1))
        u[1:nx] = 0.5 * (uc[:-1] + uc[1:])
        v[:, 1:ny] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        w[:, :, 1:nz] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
        u[1:nx][~self._face_x] = 0.0
        v[:, 1:ny][~self._face_y] = 0.0
        w[:, :, 1:nz][~self._face_z] = 0.0
        u[0] = np.where(self.inlet[0], inlet_speed * self.inlet_axis[0], u[1])
        u[nx] = u[nx - 1]
        v[:, 0] = v[:, 1]
        v[:, ny] = v[:, ny - 1]
        w[:, :, 0] = 0.0
        w[:, :, nz] = w[:, :, nz - 1]
        return u, v, w

    def _project(self, u, v, w, dt, inlet_speed):
        nx, ny, nz = self.shape
        divergence = (
            (u[1:, :, :] - u[:-1, :, :]) / self.dx
            + (v[:, 1:, :] - v[:, :-1, :]) / self.dy
            + (w[:, :, 1:] - w[:, :, :-1]) / self.dz
        )
        rhs = -divergence.reshape(-1)[self._fluid_flat] / dt
        pressure, info = cg(
            self._A,
            rhs,
            x0=self._p_prev,
            rtol=_CG_RTOL,
            atol=0.0,
            maxiter=_CG_MAXITER,
            M=self._M,
        )
        pressure = np.asarray(pressure, dtype=float)
        if not np.all(np.isfinite(pressure)):
            pressure = np.zeros_like(rhs)
        self._p_prev = pressure
        self.cg_iterations = int(info)
        field = np.zeros(nx * ny * nz)
        field[self._fluid_flat] = pressure
        field = field.reshape(self.shape)
        u[1:nx] -= dt * ((field[1:, :, :] - field[:-1, :, :]) / self.dx) * self._face_x
        v[:, 1:ny] -= dt * ((field[:, 1:, :] - field[:, :-1, :]) / self.dy) * self._face_y
        w[:, :, 1:nz] -= dt * ((field[:, :, 1:] - field[:, :, :-1]) / self.dz) * self._face_z
        u[0] = np.where(self.inlet[0], inlet_speed * self.inlet_axis[0], u[1])
        u[nx] = u[nx - 1]
        v[:, 0] = v[:, 1]
        v[:, ny] = v[:, ny - 1]
        w[:, :, 0] = 0.0
        w[:, :, nz] = w[:, :, nz - 1]
        return u, v, w

    # ------------------------------------------------------------------- step
    def _inlet_speed(self, time):
        modulation = 1.0 + 0.06 * math.sin(2.0 * math.pi * 3.1 * time + 0.7)
        modulation += 0.05 * math.sin(2.0 * math.pi * 5.3 * time + 2.1)
        return _INLET_SPEED * modulation

    def _step(self, dt) -> None:
        t = self.time
        speed = self._inlet_speed(t)
        axis = self.inlet_axis
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, reference)
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(axis, e1)
        wobble = 0.05 * speed * (
            math.sin(2.0 * math.pi * 2.3 * t) * e1
            + 0.7 * math.sin(2.0 * math.pi * 4.1 * t + 1.3) * e2
        )
        jet = speed * axis + wobble
        if np.any(self.inlet):
            self.uc[self.inlet] = jet[0]
            self.vc[self.inlet] = jet[1]
            self.wc[self.inlet] = jet[2]

        # --- body forces ----------------------------------------------------
        if _BUOYANCY != 0.0:
            self.wc = self.wc + dt * _BUOYANCY * self.smoke
        if _VORT_EPS > 0.0:
            fx, fy, fz = self._vorticity_force(self.uc, self.vc, self.wc)
            self.uc = self.uc + dt * fx
            self.vc = self.vc + dt * fy
            self.wc = self.wc + dt * fz
        drag = _WALL_DRAG_RATE * dt
        if np.any(self.wall_x):
            self.vc[self.wall_x] -= drag * self.vc[self.wall_x]
            self.wc[self.wall_x] -= drag * self.wc[self.wall_x]
        if np.any(self.wall_y):
            self.uc[self.wall_y] -= drag * self.uc[self.wall_y]
            self.wc[self.wall_y] -= drag * self.wc[self.wall_y]
        if np.any(self.wall_z):
            self.uc[self.wall_z] -= drag * self.uc[self.wall_z]
            self.vc[self.wall_z] -= drag * self.vc[self.wall_z]

        # --- momentum advection --------------------------------------------
        self.uc, self.vc, self.wc = self._advect(
            [self.uc, self.vc, self.wc], self.uc, self.vc, self.wc, dt
        )
        if np.any(self.inlet):
            self.uc[self.inlet] = jet[0]
            self.vc[self.inlet] = jet[1]
            self.wc[self.inlet] = jet[2]
        self.uc[self.solid] = 0.0
        self.vc[self.solid] = 0.0
        self.wc[self.solid] = 0.0

        # --- incompressible projection --------------------------------------
        u, v, w = self._faces_from_cc(self.uc, self.vc, self.wc, speed)
        u, v, w = self._project(u, v, w, dt, speed)
        nx, ny, nz = self.shape
        self.uc = 0.5 * (u[:-1] + u[1:])
        self.vc = 0.5 * (v[:, :-1] + v[:, 1:])
        self.wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
        self.uc[self.solid] = 0.0
        self.vc[self.solid] = 0.0
        self.wc[self.solid] = 0.0

        # --- transported scalar --------------------------------------------
        (smoke,) = self._advect([self.smoke], self.uc, self.vc, self.wc, dt)
        smoke = smoke + dt * self.source
        np.clip(smoke, 0.0, _SOURCE_CAP, out=smoke)
        smoke[self.solid] = 0.0
        self.smoke = smoke

        # --- passive tracers ------------------------------------------------
        if len(self.p_position) > 0:
            v1 = self._sample_vector(self.p_position)
            mid = self.p_position + 0.5 * dt * v1
            v2 = self._sample_vector(mid)
            self.p_position = self.p_position + dt * v2
            self.p_velocity = np.clip(v2, -10.0, 10.0)
            lo = self.bmin + 0.02 * self.hmin
            hi = self.bmax - 0.02 * self.hmin
            self.p_position = np.clip(self.p_position, lo[None, :], hi[None, :])

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        while self.time < target - 1e-9:
            dt = min(_DT_MAX, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt)
            self.time += dt
            self.steps += 1
        return self._snapshot(target)

    def _snapshot(self, time: float) -> StateFrame:
        nx, ny, nz = self.shape
        wall_sample = self.boundary_points + 0.35 * self.hmin * self.boundary_normals
        boundary_velocity = self._sample_vector(wall_sample)
        boundary_velocity[~np.isfinite(boundary_velocity)] = 0.0
        velocity_field = np.stack([self.uc, self.vc, self.wc], axis=-1)
        velocity_field[~np.isfinite(velocity_field)] = 0.0
        smoke = np.array(self.smoke, dtype=float)
        smoke[~np.isfinite(smoke)] = 0.0
        attributes = {name: np.array(value) for name, value in self.p_attributes.items()}
        frame = StateFrame(
            time=float(time),
            position=np.array(self.p_position, dtype=float),
            velocity=np.array(self.p_velocity, dtype=float),
            mass=np.array(self.p_mass, dtype=float),
            body_id=np.array(self.p_body, dtype=np.int32),
            attributes=attributes,
            fields={
                "active_mask": np.array(self.fluid, dtype=bool),
                "boundary_normal": np.array(self.boundary_normals, dtype=float),
                "boundary_target_velocity": np.array(self.boundary_targets, dtype=float),
                "boundary_velocity": boundary_velocity,
                "scalar_source": np.array(self.source, dtype=float),
                "transported_scalar": smoke,
                "velocity_field": velocity_field,
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.grid_spacing),
                "reference_velocity": float(self.reference_velocity),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        speed = 0.0
        if hasattr(self, "uc"):
            speed = float(
                np.max(np.sqrt(self.uc ** 2 + self.vc ** 2 + self.wc ** 2))
            )
        return {
            "solver": "mac_projection_smoke",
            "steps": int(self.steps),
            "final_time": float(getattr(self, "time", 0.0)),
            "grid_shape": list(self.shape) if hasattr(self, "shape") else [],
            "cell_count": int(getattr(self, "_nfluid", 0)),
            "max_speed": speed,
            "smoke_max": float(np.max(self.smoke)) if hasattr(self, "smoke") else 0.0,
            "boundary_samples": int(len(getattr(self, "boundary_points", []))),
            "last_cg_iterations": int(getattr(self, "cg_iterations", 0)),
        }
