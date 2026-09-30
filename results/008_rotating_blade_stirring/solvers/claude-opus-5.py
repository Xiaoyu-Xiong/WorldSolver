"""Self-implemented 3D free-surface liquid solver for the rotating-blade stirring scene.

Physical model
--------------
Incompressible inviscid-dominant Navier-Stokes with a free surface, integrated with a
hybrid FLIP/PIC (particle-in-cell) scheme on a staggered MAC grid:

    Du/Dt = -grad(p)/rho + g ,        div(u) = 0   inside the liquid,
    p = 0 on the air-water interface (atmospheric),
    u.n = u_solid.n on impermeable free-slip solids (tank floor, four walls, blade).

Per substep:
  1. particle -> grid transfer of momentum (trilinear / PIC weights) on MAC faces,
  2. gravity applied to all faces,
  3. the rotating blade is voxelized by exact segment/oriented-box intersection of the
     dual edges connecting the cell centres that a MAC face separates; blocked faces
     carry the rigid-body normal velocity omega(t) x r, which is how blade momentum and
     torque enter the fluid,
  4. a variational pressure Poisson problem is assembled over liquid cells only
     (Neumann on solid faces, Dirichlet p=0 on air faces) and solved with a Jacobi
     preconditioned conjugate gradient, giving a discretely divergence-free face field,
  5. velocity is extrapolated across the narrow air-side band,
  6. grid -> particle transfer blending FLIP (delta) and PIC (full) velocities,
  7. RK2 particle advection, then exact push-out from the rotated blade box and
     free-slip clamping on the static tank solids.

No animation curves or precomputed trajectories are used: only gravity, the pressure
projection and the prescribed rigid blade boundary drive the liquid. The blade pose is
read from the frozen scene actuator (`scene.blade_kinematics`).
"""

from __future__ import annotations

import numpy as np

try:  # SciPy is used for the sparse Poisson solve and the level-set distance transform.
    import scipy.ndimage as ndimage
    import scipy.sparse as sparse
    import scipy.sparse.linalg as splinalg

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback keeps the solver runnable.
    _HAVE_SCIPY = False

from worldsolver.contracts import StateFrame


def _shift(array: np.ndarray, axis: int, delta: int) -> np.ndarray:
    """Shift `array` along `axis` by `delta` cells, padding with zeros."""
    out = np.zeros_like(array)
    n = array.shape[axis]
    if abs(delta) >= n:
        return out
    src = [slice(None)] * array.ndim
    dst = [slice(None)] * array.ndim
    if delta > 0:
        src[axis] = slice(0, n - delta)
        dst[axis] = slice(delta, n)
    elif delta < 0:
        src[axis] = slice(-delta, n)
        dst[axis] = slice(0, n + delta)
    out[tuple(dst)] = array[tuple(src)]
    return out


def _neighbor_index(index_map: np.ndarray, axis: int, delta: int) -> np.ndarray:
    """Gather the flat solve index of the neighbour cell, -1 where it does not exist."""
    out = np.full(index_map.shape, -1, dtype=np.int64)
    n = index_map.shape[axis]
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if delta > 0:
        src[axis] = slice(delta, n)
        dst[axis] = slice(0, n - delta)
    else:
        src[axis] = slice(0, n + delta)
        dst[axis] = slice(-delta, n)
    out[tuple(dst)] = index_map[tuple(src)]
    return out


def _solve_spd(matrix, rhs: np.ndarray, guess: np.ndarray | None) -> np.ndarray:
    """Jacobi preconditioned CG on a symmetric positive definite system."""
    diagonal = matrix.diagonal()
    diagonal = np.where(np.abs(diagonal) > 1e-30, diagonal, 1.0)
    preconditioner = sparse.diags(1.0 / diagonal)
    try:
        solution, _ = splinalg.cg(
            matrix, rhs, x0=guess, rtol=1e-9, atol=0.0, maxiter=500, M=preconditioner
        )
    except TypeError:  # older SciPy keyword
        solution, _ = splinalg.cg(
            matrix, rhs, x0=guess, tol=1e-9, atol=0.0, maxiter=500, M=preconditioner
        )
    if not np.all(np.isfinite(solution)):
        solution = splinalg.spsolve(matrix.tocsc(), rhs)
    if not np.all(np.isfinite(solution)):
        solution = np.zeros_like(rhs)
    return np.asarray(solution, dtype=float)
class BenchmarkSolver:
    """FLIP/PIC free-surface liquid coupled to the frozen kinematic blade actuator."""

    flip_ratio = 0.93
    cfl = 0.85
    dt_min = 5.0e-4
    extrapolation_sweeps = 4

    def initialize(self, scene, seed: int) -> None:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for the pressure projection")
        self.scene = scene
        self.config = dict(scene.config)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        domain = self.config["domain"]
        liquid = self.config["liquid"]
        tank = self.config["tank"]
        blade = self.config["blade"]

        self.h = float(domain["grid_spacing"])
        self.origin = np.asarray(domain["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(domain["bounds_max"], dtype=float)
        self.nx, self.ny, self.nz = (int(v) for v in domain["reference_grid"])
        self.shape = (self.nx, self.ny, self.nz)

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.density = float(liquid["density"])
        self.viscosity = float(liquid["kinematic_viscosity"])
        self.spacing = float(liquid["particle_spacing"])
        self.half_width = float(tank["interior_half_width"])
        self.tank_height = float(tank["height"])
        self.depth = float(liquid["depth"])

        self.blade_half = np.asarray(blade["half_extents"], dtype=float)
        self.blade_center = np.asarray(blade["center"], dtype=float)
        self.blade_body_id = int(blade["body_id"])
        self.tip_speed = float(abs(float(blade["angular_speed"])) * self.blade_half[0])

        self.initial_frame = scene.initial_state()
        body_id = np.asarray(self.initial_frame.body_id)
        self.liquid_mask = body_id == 0
        self.n_particles = int(np.count_nonzero(self.liquid_mask))
        self.px = np.array(self.initial_frame.position[self.liquid_mask], dtype=float)
        self.pv = np.array(self.initial_frame.velocity[self.liquid_mask], dtype=float)
        attributes = self.initial_frame.attributes
        self.particle_mass = np.array(attributes["mass_quantity"][self.liquid_mask], dtype=float)
        self.particle_volume = np.array(
            attributes["particle_volume"][self.liquid_mask], dtype=float
        )
        self.particle_radius = np.array(
            attributes["particle_radius"][self.liquid_mask], dtype=float
        )
        self.total_mass = float(self.particle_mass.sum())
        self.characteristic_length = float(self.initial_frame.scalars["characteristic_length"])
        self.reference_velocity = float(self.initial_frame.scalars["reference_velocity"])
        self.initial_volume = float(self.initial_frame.scalars["initial_liquid_volume"])

        # Staggered MAC velocity components.
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.valid = [
            np.zeros(self.u.shape, dtype=bool),
            np.zeros(self.v.shape, dtype=bool),
            np.zeros(self.w.shape, dtype=bool),
        ]
        self.p_grid: np.ndarray | None = None
        self.last_dt = 5.0e-3

        self._build_face_geometry()
        self._build_wall_samples()

        self.time = 0.0
        self.step_count = 0
        self.substep_min = float("inf")
        self.max_speed_seen = 0.0

    # ------------------------------------------------------------------ geometry
    def _build_face_geometry(self) -> None:
        """MAC face positions and the dual segments used for exact solid voxelization."""
        h, o = self.h, self.origin
        nx, ny, nz = self.shape
        self.cell_centers = tuple(
            o[axis] + (np.arange(self.shape[axis]) + 0.5) * h for axis in range(3)
        )
        self.cell_points = np.stack(np.meshgrid(*self.cell_centers, indexing="ij"), axis=-1)
        self.face_positions = []
        self.segment_a = []
        self.segment_b = []
        for axis in range(3):
            coords = []
            for other in range(3):
                if other == axis:
                    coords.append(o[axis] + np.arange(self.shape[axis] + 1) * h)
                else:
                    coords.append(o[other] + (np.arange(self.shape[other]) + 0.5) * h)
            points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
            self.face_positions.append(points)
            offset = np.zeros(3)
            offset[axis] = 0.5 * h
            self.segment_a.append(points - offset)
            self.segment_b.append(points + offset)
    def _build_wall_samples(self) -> None:
        """Candidate samples on every stationary tank face (floor and four side walls)."""
        h = self.h
        half = self.half_width
        xy = np.arange(-half + h / 2, half, h)
        z = np.arange(h / 2, self.tank_height, h)
        floor = np.stack(np.meshgrid(xy, xy, indexing="ij"), axis=-1).reshape(-1, 2)
        positions = [np.column_stack((floor, np.zeros(len(floor))))]
        normals = [np.tile([0.0, 0.0, 1.0], (len(floor), 1))]
        side = np.stack(np.meshgrid(xy, z, indexing="ij"), axis=-1).reshape(-1, 2)
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                p = np.zeros((len(side), 3))
                p[:, axis] = sign * half
                p[:, 1 - axis] = side[:, 0]
                p[:, 2] = side[:, 1]
                n = np.zeros_like(p)
                n[:, axis] = -sign
                positions.append(p)
                normals.append(n)
        self.wall_position = np.concatenate(positions)
        self.wall_normal = np.concatenate(normals)
        self.floor_count = len(floor)

    @staticmethod
    def _segment_box_hit(
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        rotation: np.ndarray,
        half: np.ndarray,
    ) -> np.ndarray:
        """Exact slab test: does segment [start,end] intersect the oriented box?"""
        local_a = (start - center) @ rotation
        direction = (end - start) @ rotation
        tmin = np.zeros(local_a.shape[:-1])
        tmax = np.ones(local_a.shape[:-1])
        for axis in range(3):
            d = direction[..., axis]
            a = local_a[..., axis]
            degenerate = np.abs(d) < 1e-14
            safe = np.where(degenerate, 1.0, d)
            t1 = (-half[axis] - a) / safe
            t2 = (half[axis] - a) / safe
            low = np.minimum(t1, t2)
            high = np.maximum(t1, t2)
            inside = (a >= -half[axis]) & (a <= half[axis])
            low = np.where(degenerate, np.where(inside, 0.0, 1.0), low)
            high = np.where(degenerate, np.where(inside, 1.0, 0.0), high)
            tmin = np.maximum(tmin, low)
            tmax = np.minimum(tmax, high)
        return tmin <= tmax
    def _update_solid_masks(self, pose: dict) -> None:
        """Voxelize the rotated blade and the static tank onto MAC faces and cells."""
        center = pose["position"]
        rotation = pose["rotation_matrix"]
        omega = pose["angular_velocity"]
        nx, ny, nz = self.shape
        self.blocked = []
        self.solid_face_velocity = []
        for axis in range(3):
            hit = self._segment_box_hit(
                self.segment_a[axis], self.segment_b[axis], center, rotation, self.blade_half
            )
            relative = self.face_positions[axis] - center
            solid_velocity = np.cross(omega, relative)[..., axis]
            solid_velocity = np.where(hit, solid_velocity, 0.0)
            blocked = np.array(hit)
            index = [slice(None)] * 3
            index[axis] = 0
            blocked[tuple(index)] = True
            solid_velocity[tuple(index)] = 0.0
            index[axis] = self.shape[axis]
            blocked[tuple(index)] = True
            solid_velocity[tuple(index)] = 0.0
            self.blocked.append(blocked)
            self.solid_face_velocity.append(solid_velocity)
        self.solid_cells = self._blade_sdf(self.cell_points, center, rotation) < 0.0

    def _blade_sdf(
        self, points: np.ndarray, center: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        """Signed distance to the oriented blade box, positive outside."""
        local = (np.asarray(points, dtype=float) - center) @ rotation
        q = np.abs(local) - self.blade_half
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)

    # ------------------------------------------------------- transfer kernels
    def _grid_weights(self, points: np.ndarray, shape, base: np.ndarray):
        rel = (np.asarray(points, dtype=float) - base) / self.h
        i0 = np.floor(rel).astype(np.int64)
        for axis in range(3):
            np.clip(i0[:, axis], 0, int(shape[axis]) - 2, out=i0[:, axis])
        frac = np.clip(rel - i0, 0.0, 1.0)
        return i0, frac

    def _scatter(self, points: np.ndarray, values: np.ndarray, shape, base: np.ndarray):
        """Trilinear (PIC) splat returning weighted sums and weights."""
        i0, frac = self._grid_weights(points, shape, base)
        strides = (int(shape[1]) * int(shape[2]), int(shape[2]), 1)
        size = int(shape[0]) * strides[0]
        flat = i0[:, 0] * strides[0] + i0[:, 1] * strides[1] + i0[:, 2]
        num = np.zeros(size)
        den = np.zeros(size)
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = wx * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                for dz in (0, 1):
                    weight = wy * (frac[:, 2] if dz else 1.0 - frac[:, 2])
                    idx = flat + dx * strides[0] + dy * strides[1] + dz
                    num += np.bincount(idx, weights=weight * values, minlength=size)
                    den += np.bincount(idx, weights=weight, minlength=size)
        return num.reshape(shape), den.reshape(shape)

    def _gather(self, field: np.ndarray, points: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of a grid field at arbitrary points."""
        i0, frac = self._grid_weights(points, field.shape, base)
        out = np.zeros(len(points))
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = wx * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                for dz in (0, 1):
                    weight = wy * (frac[:, 2] if dz else 1.0 - frac[:, 2])
                    out += weight * field[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out

    @property
    def face_bases(self):
        h = self.h
        return (
            self.origin + np.array([0.0, 0.5 * h, 0.5 * h]),
            self.origin + np.array([0.5 * h, 0.0, 0.5 * h]),
            self.origin + np.array([0.5 * h, 0.5 * h, 0.0]),
        )

    @property
    def cell_base(self) -> np.ndarray:
        return self.origin + 0.5 * self.h

    def _interpolate_mac(self, points: np.ndarray) -> np.ndarray:
        bases = self.face_bases
        return np.stack(
            [
                self._gather(self.u, points, bases[0]),
                self._gather(self.v, points, bases[1]),
                self._gather(self.w, points, bases[2]),
            ],
            axis=-1,
        )
    # ------------------------------------------------------------- fluid steps
    def _particles_to_grid(self) -> None:
        bases = self.face_bases
        shapes = (self.u.shape, self.v.shape, self.w.shape)
        fields = []
        for axis in range(3):
            num, den = self._scatter(self.px, self.pv[:, axis], shapes[axis], bases[axis])
            has = den > 1e-12
            field = np.zeros(shapes[axis])
            field[has] = num[has] / den[has]
            fields.append(field)
            self.valid[axis] = has
        self.u, self.v, self.w = fields

    def _cell_indices(self) -> np.ndarray:
        rel = (self.px - self.origin) / self.h
        idx = np.floor(rel).astype(np.int64)
        for axis in range(3):
            np.clip(idx[:, axis], 0, self.shape[axis] - 1, out=idx[:, axis])
        return idx[:, 0] * (self.ny * self.nz) + idx[:, 1] * self.nz + idx[:, 2]

    def _fluid_cells(self) -> np.ndarray:
        counts = np.bincount(self._cell_indices(), minlength=self.nx * self.ny * self.nz)
        return (counts.reshape(self.shape) > 0) & (~self.solid_cells)

    def _apply_solid_velocity(self) -> None:
        for axis, field in enumerate((self.u, self.v, self.w)):
            blocked = self.blocked[axis]
            field[blocked] = self.solid_face_velocity[axis][blocked]
            self.valid[axis] = self.valid[axis] | blocked

    def _project(self, dt: float, fluid: np.ndarray) -> None:
        """Variational pressure projection enforcing div(u)=0 in the liquid."""
        cells = np.flatnonzero(fluid.ravel())
        if cells.size == 0:
            self.p_grid = np.zeros(self.shape)
            return
        index_map = np.full(self.nx * self.ny * self.nz, -1, dtype=np.int64)
        index_map[cells] = np.arange(cells.size)
        index_map = index_map.reshape(self.shape)

        divergence = (
            (self.u[1:, :, :] - self.u[:-1, :, :])
            + (self.v[:, 1:, :] - self.v[:, :-1, :])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / self.h

        diagonal = np.zeros(self.shape)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for axis in range(3):
            blocked = self.blocked[axis]
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, self.shape[axis])
            hi[axis] = slice(1, self.shape[axis] + 1)
            faces = (~blocked[tuple(lo)], ~blocked[tuple(hi)])
            neighbors = (
                _neighbor_index(index_map, axis, -1),
                _neighbor_index(index_map, axis, +1),
            )
            for open_face, neighbor in zip(faces, neighbors):
                diagonal += (open_face & fluid).astype(float)
                coupled = open_face & fluid & (neighbor >= 0)
                if np.any(coupled):
                    rows.append(index_map[coupled])
                    cols.append(neighbor[coupled])
                    vals.append(np.full(int(np.count_nonzero(coupled)), -1.0))

        diag_values = diagonal[fluid]
        singular = diag_values <= 0.0
        diag_values = np.where(singular, 1.0, diag_values)
        rows.append(np.arange(cells.size))
        cols.append(np.arange(cells.size))
        vals.append(diag_values)
        matrix = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(cells.size, cells.size),
        ).tocsr()

        # A = n_open*I - adjacency  discretizes -h^2*lap(p), so A p = -rho h^2/dt div(u*).
        rhs = -self.density * self.h**2 / dt * divergence[fluid]
        rhs = np.where(singular, 0.0, rhs)
        guess = self.p_grid[fluid] if self.p_grid is not None else None
        pressure = _solve_spd(matrix, rhs, guess)

        p_grid = np.zeros(self.shape)
        p_grid[fluid] = pressure
        self.p_grid = p_grid

        coefficient = dt / (self.density * self.h)
        for axis, field in enumerate((self.u, self.v, self.w)):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            padded = np.pad(p_grid, pad)
            fluid_pad = np.pad(fluid, pad)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, self.shape[axis] + 1)
            hi[axis] = slice(1, self.shape[axis] + 2)
            gradient = padded[tuple(hi)] - padded[tuple(lo)]
            touches = fluid_pad[tuple(hi)] | fluid_pad[tuple(lo)]
            update = touches & (~self.blocked[axis])
            field[update] -= coefficient * gradient[update]
            self.valid[axis] = self.valid[axis] | update

    def _extrapolate(self) -> None:
        for axis, field in enumerate((self.u, self.v, self.w)):
            valid = self.valid[axis].copy()
            for _ in range(self.extrapolation_sweeps):
                if valid.all():
                    break
                masked = np.where(valid, field, 0.0)
                weights = valid.astype(float)
                num = np.zeros_like(field)
                den = np.zeros_like(field)
                for direction in range(3):
                    for delta in (-1, 1):
                        num += _shift(masked, direction, delta)
                        den += _shift(weights, direction, delta)
                fill = (~valid) & (den > 0.0)
                if not np.any(fill):
                    break
                field[fill] = num[fill] / den[fill]
                valid |= fill
            self.valid[axis] = valid
    def _substep(self, dt: float) -> None:
        """One physical FLIP/PIC substep of the incompressible free-surface flow."""
        pose = self.scene.blade_kinematics(self.time + dt)
        self._update_solid_masks(pose)

        self._particles_to_grid()
        u_star = (self.u.copy(), self.v.copy(), self.w.copy())

        # Body force (gravity) on the faces.
        self.u += self.gravity[0] * dt
        self.v += self.gravity[1] * dt
        self.w += self.gravity[2] * dt

        # Weak viscous diffusion of the face fields (explicit, tiny at nu=1e-6).
        nu_dt = self.viscosity * dt / self.h**2
        if nu_dt > 1e-9:
            factor = min(nu_dt, 1.0 / 6.0)
            for field in (self.u, self.v, self.w):
                laplacian = -6.0 * field
                for direction in range(3):
                    for delta in (-1, 1):
                        laplacian += _shift(field, direction, delta)
                field += factor * laplacian

        self._apply_solid_velocity()
        fluid = self._fluid_cells()
        self._project(dt, fluid)
        self._apply_solid_velocity()
        self._extrapolate()

        # Grid -> particles with a FLIP/PIC blend.
        bases = self.face_bases
        new_velocity = self._interpolate_mac(self.px)
        delta = np.stack(
            [
                self._gather(self.u - u_star[0], self.px, bases[0]),
                self._gather(self.v - u_star[1], self.px, bases[1]),
                self._gather(self.w - u_star[2], self.px, bases[2]),
            ],
            axis=-1,
        )
        flip = self.pv + delta
        self.pv = self.flip_ratio * flip + (1.0 - self.flip_ratio) * new_velocity

        # RK2 advection in the projected velocity field.
        midpoint = self.px + 0.5 * dt * new_velocity
        self.px = self.px + dt * self._interpolate_mac(midpoint)

        self.time += dt
        self.last_dt = dt
        self.step_count += 1
        self.substep_min = min(self.substep_min, dt)
        pose_end = self.scene.blade_kinematics(self.time)
        self._resolve_collisions(pose_end)
        self.max_speed_seen = max(
            self.max_speed_seen, float(np.max(np.linalg.norm(self.pv, axis=1)))
        )

    def _resolve_collisions(self, pose: dict) -> None:
        """Push particles out of the rotated blade and enforce free-slip tank walls."""
        margin = 0.25 * self.spacing
        center = pose["position"]
        rotation = pose["rotation_matrix"]
        omega = pose["angular_velocity"]
        local = (self.px - center) @ rotation
        q = np.abs(local) - self.blade_half
        inside = np.flatnonzero(q.max(axis=1) < margin)
        if inside.size:
            sub = local[inside]
            axis = np.argmax(q[inside], axis=1)
            rows = np.arange(inside.size)
            sign = np.sign(sub[rows, axis])
            sign = np.where(sign == 0.0, 1.0, sign)
            target = sign * (self.blade_half[axis] + margin)
            sub[rows, axis] = target
            self.px[inside] = sub @ rotation.T + center
            normal = np.zeros((inside.size, 3))
            normal[rows, axis] = sign
            normal = normal @ rotation.T
            solid_velocity = np.cross(omega, self.px[inside] - center)
            relative = self.pv[inside] - solid_velocity
            normal_component = np.sum(relative * normal, axis=1)
            approaching = normal_component < 0.0
            relative -= np.where(approaching, normal_component, 0.0)[:, None] * normal
            self.pv[inside] = solid_velocity + relative

        eps = 0.4 * self.spacing
        half = self.half_width - eps
        for axis in (0, 1):
            low = self.px[:, axis] < -half
            high = self.px[:, axis] > half
            self.px[low, axis] = -half
            self.pv[low, axis] = np.maximum(self.pv[low, axis], 0.0)
            self.px[high, axis] = half
            self.pv[high, axis] = np.minimum(self.pv[high, axis], 0.0)
        floor = self.px[:, 2] < eps
        self.px[floor, 2] = eps
        self.pv[floor, 2] = np.maximum(self.pv[floor, 2], 0.0)
        ceiling = self.bounds_max[2] - eps
        top = self.px[:, 2] > ceiling
        self.px[top, 2] = ceiling
        self.pv[top, 2] = np.minimum(self.pv[top, 2], 0.0)
        if not np.all(np.isfinite(self.px)) or not np.all(np.isfinite(self.pv)):
            self.px = np.nan_to_num(self.px, nan=0.0, posinf=0.0, neginf=0.0)
            self.pv = np.nan_to_num(self.pv, nan=0.0, posinf=0.0, neginf=0.0)

    def _pick_dt(self, remaining: float) -> float:
        speed = float(np.max(np.linalg.norm(self.pv, axis=1))) if self.n_particles else 0.0
        speed = max(speed, self.tip_speed, 1e-6)
        dt = self.cfl * self.h / speed
        dt = min(dt, 5.0e-3)
        dt = max(dt, self.dt_min)
        if dt >= remaining - 1e-12 or remaining - dt < 0.2 * self.dt_min:
            dt = remaining
        return dt

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= 1e-15:
            self.time = 0.0
            frame = self.initial_frame
            frame.validate()
            return frame
        while self.time < target - 1e-12:
            self._substep(self._pick_dt(target - self.time))
        self.time = target
        return self._build_frame(target)
    # --------------------------------------------------------------- reporting
    def _liquid_level_set(self, filled: np.ndarray) -> np.ndarray:
        """Signed liquid-air level set from particle occupancy (negative in liquid)."""
        big = self.h * float(self.nx + self.ny + self.nz)
        if not filled.any():
            return np.full(self.shape, big)
        if filled.all():
            return np.full(self.shape, -big)
        outside = ndimage.distance_transform_edt(~filled, sampling=self.h)
        inside = ndimage.distance_transform_edt(filled, sampling=self.h)
        outside = np.where(np.isfinite(outside), outside, big)
        inside = np.where(np.isfinite(inside), inside, big)
        return np.where(filled, -(inside - 0.5 * self.h), outside - 0.5 * self.h)

    def _occupancy(self, solid_distance: np.ndarray) -> np.ndarray:
        """Cells holding liquid, extended horizontally through the thin blade slab."""
        counts = np.bincount(self._cell_indices(), minlength=self.nx * self.ny * self.nz)
        filled = counts.reshape(self.shape) > 0
        band = solid_distance < 0.75 * self.h
        if band.any():
            structure = np.zeros((3, 3, 3), dtype=bool)
            structure[1, 1, 1] = True
            structure[0, 1, 1] = structure[2, 1, 1] = True
            structure[1, 0, 1] = structure[1, 2, 1] = True
            for _ in range(3):
                filled = filled | (ndimage.binary_dilation(filled, structure=structure) & band)
        return filled

    def _build_frame(self, time: float) -> StateFrame:
        pose = self.scene.blade_kinematics(time)
        center = pose["position"]
        rotation = pose["rotation_matrix"]
        self._update_solid_masks(pose)

        # Diagnostic re-projection so velocity_field, level set and mask are consistent.
        self._particles_to_grid()
        self._apply_solid_velocity()
        fluid = self._fluid_cells()
        self._project(self.last_dt, fluid)
        self._apply_solid_velocity()
        self._extrapolate()

        solid_distance = self._blade_sdf(self.cell_points, center, rotation)
        filled = self._occupancy(solid_distance)
        phi = self._liquid_level_set(filled)

        velocity_field = np.stack(
            [
                0.5 * (self.u[:-1, :, :] + self.u[1:, :, :]),
                0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :]),
                0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:]),
            ],
            axis=-1,
        )

        h = self.h
        x = self.cell_points[..., 0]
        y = self.cell_points[..., 1]
        z = self.cell_points[..., 2]
        interior = (
            (z > h)
            & (np.abs(x) < self.half_width - h)
            & (np.abs(y) < self.half_width - h)
        )
        # Keep the whole collocated divergence stencil inside projected liquid cells:
        # 2h clearance from the blade puts every face neighbour outside the fill band.
        active = (phi < -h) & filled & fluid & interior & (solid_distance > 2.0 * h)
        if not active.any():
            active = (phi < -h) & filled & fluid & interior & (solid_distance > h)

        base = self.cell_base
        probe = self.wall_position + 0.5 * h * self.wall_normal
        wet_wall = self._gather(phi, probe, base) < 0.0
        if not wet_wall.any():
            wet_wall = np.zeros(len(self.wall_position), dtype=bool)
            wet_wall[: self.floor_count] = True
        wall_position = self.wall_position[wet_wall]
        wall_normal = self.wall_normal[wet_wall]
        wall_velocity = self._interpolate_mac(wall_position)

        samples = self.scene.blade_surface_samples(time, spacing=h)
        blade_position = np.asarray(samples["position"], dtype=float)
        blade_normal = np.asarray(samples["normal"], dtype=float)
        blade_target = np.asarray(samples["target_velocity"], dtype=float)
        offset = blade_position + 0.5 * h * blade_normal
        wet_blade = self._gather(phi, offset, base) < 0.0
        if not wet_blade.any():
            wet_blade = blade_position[:, 2] < self.depth
        if not wet_blade.any():
            wet_blade = np.zeros(len(blade_position), dtype=bool)
            wet_blade[int(np.argmin(blade_position[:, 2]))] = True
        blade_measured = self._interpolate_mac(offset[wet_blade])

        count = self.n_particles + 1
        position = np.vstack((self.px, center[None, :]))
        velocity = np.vstack((self.pv, np.asarray(pose["velocity"], dtype=float)[None, :]))
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
        orientation[-1] = np.asarray(pose["orientation"], dtype=float)
        angular_velocity = np.zeros((count, 3))
        angular_velocity[-1] = np.asarray(pose["angular_velocity"], dtype=float)
        mass_quantity = np.r_[self.particle_mass, 0.0]
        body_id = np.zeros(count, dtype=np.int64)
        body_id[-1] = self.blade_body_id

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass_quantity.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": mass_quantity,
                "particle_volume": np.r_[self.particle_volume, 0.0],
                "particle_radius": np.r_[self.particle_radius, 0.0],
            },
            fields={
                "velocity_field": velocity_field,
                "active_mask": active.astype(float),
                "liquid_level_set": phi,
                "density_field": filled.astype(float) * self.density,
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": wall_position,
                "boundary_normal": wall_normal,
                "boundary_velocity": wall_velocity,
                "boundary_target_velocity": np.zeros_like(wall_position),
                "blade_boundary_position": blade_position[wet_blade],
                "blade_boundary_normal": blade_normal[wet_blade],
                "blade_boundary_velocity": blade_measured,
                "blade_boundary_target_velocity": blade_target[wet_blade],
                "grid_origin": self.origin.copy(),
            },
            scalars={
                "grid_spacing": float(h),
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
                "initial_liquid_volume": self.initial_volume,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "FLIP/PIC MAC-grid incompressible free-surface solver (self-implemented)",
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.h),
            "particle_count": int(self.n_particles),
            "substeps": int(self.step_count),
            "min_substep": float(self.substep_min) if np.isfinite(self.substep_min) else 0.0,
            "flip_ratio": float(self.flip_ratio),
            "total_liquid_mass": float(self.total_mass),
            "max_particle_speed": float(self.max_speed_seen),
            "final_time": float(self.time),
        }
