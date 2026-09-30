"""Incompressible 2D liquid solver (FLIP/PIC) for the dam-break scene.

Physical model
-------------
Inviscid incompressible flow with a free surface,

    Du/Dt = -grad(p)/rho + g,      div(u) = 0,

discretised with a particle-in-cell scheme on a staggered (MAC) grid, the
same family of methods as the APIC reference:

  1. particle -> grid momentum transfer with bilinear kernels,
  2. explicit gravity on the transferred face velocities,
  3. free-slip solid conditions on the frozen tank floor and side walls,
  4. exact discrete pressure projection (sparse SPD Poisson solve) with
     Dirichlet p = 0 in air cells and Neumann conditions at solids,
  5. velocity extrapolation into the air band next to the free surface,
  6. grid -> particle transfer as a FLIP/PIC blend (the PIC fraction is the
     physical dissipation channel that lets the sloshing settle),
  7. RK2 particle advection in the projected (divergence-free) field,
     followed by wall collision response and a mild particle
     redistribution that keeps the sample density near its rest value.

The scheme is fully deterministic: no random numbers are drawn anywhere and
all reductions use fixed-order NumPy/SciPy operations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    """MAC-grid FLIP/PIC free-surface liquid solver."""

    TARGET_DX = 0.10          # target diagnostic/simulation cell size [m]
    CFL = 0.9                 # advective CFL number
    MAX_SUBSTEP = 5.0e-3      # hard cap on the physical substep [s]
    MIN_SUBSTEP = 2.0e-4      # floor to guarantee progress [s]
    FLIP_RATIO = 0.93         # FLIP fraction; (1 - ratio) is the PIC damping
    EXTRAP_SWEEPS = 3         # velocity extrapolation band width (cells)
    PUSH_ITERS = 2            # particle redistribution iterations
    PUSH_STIFFNESS = 0.5      # fraction of the overlap removed per iteration

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        frame = scene.initial_state()
        position = np.asarray(frame.position, dtype=np.float64)
        velocity = np.asarray(frame.velocity, dtype=np.float64)
        self.n_particles = int(position.shape[0])
        # 2D state: simulate (x, z); the y column is frozen at its scene value.
        self.pos = np.ascontiguousarray(position[:, [0, 2]], dtype=np.float64)
        self.vel = np.ascontiguousarray(velocity[:, [0, 2]], dtype=np.float64)
        self.y_plane = np.ascontiguousarray(position[:, 1], dtype=np.float64)

        self.density = float(config.get("initial_conditions", {}).get("liquid_density", 1000.0))
        gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        ).reshape(-1)
        self.gx = float(gravity[0])
        self.gz = float(gravity[2] if gravity.size > 2 else gravity[-1])

        if frame.mass is not None:
            self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        else:
            self.mass = np.full(self.n_particles, self.density * 0.06 * 0.06)
        volume = np.asarray(
            frame.attributes.get(
                "sample_volume", self.mass / max(self.density, 1.0e-9)
            ),
            dtype=np.float64,
        )
        self.sample_volume = np.ascontiguousarray(volume, dtype=np.float64).copy()

        self._setup_domain()
        self._reset_grids()
        self.time = 0.0
        self.substeps = 0
        self.max_speed_seen = 0.0
        # Establish a consistent grid state for the t = 0 diagnostics.
        self._transfer_to_grid()
        self._solid_boundary_conditions()
    def _setup_domain(self) -> None:
        """Derive the fluid domain from the frozen tank geometry."""
        objects = (self.config.get("scene_geometry", {}) or {}).get("objects", []) or []
        left, right, floors, tops = [], [], [], []
        for item in objects:
            geometry = item.get("geometry", {}) or {}
            if geometry.get("type") != "box":
                continue
            half = geometry.get("half_extents")
            if half is None or len(half) != 3:
                continue
            center = ((item.get("transform", {}) or {}).get("translation") or [0.0, 0.0, 0.0])
            cx, cz = float(center[0]), float(center[2])
            hx, hz = float(half[0]), float(half[2])
            if hx < hz:  # tall and thin -> side wall
                if cx < 0.0:
                    left.append(cx + hx)
                else:
                    right.append(cx - hx)
                tops.append(cz + hz)
            else:        # wide and flat -> floor
                floors.append(cz + hz)
                tops.append(cz + hz)

        liquid = (self.config.get("initial_conditions", {}) or {}).get("liquid_region", {}) or {}
        lo = np.asarray(liquid.get("min", [-2.75, -0.025, 0.10]), dtype=np.float64)
        hi = np.asarray(liquid.get("max", [-1.25, 0.025, 2.50]), dtype=np.float64)

        self.x_lo = max(left) if left else float(lo[0]) - 0.5
        self.x_hi = min(right) if right else float(hi[0]) + 4.0
        self.z_lo = max(floors) if floors else float(lo[2]) - 0.05
        z_hi = max(tops) if tops else float(hi[2]) + 0.5
        z_hi = max(z_hi, float(hi[2]) + 0.1, self.z_lo + 0.5)

        width = self.x_hi - self.x_lo
        self.nx = max(8, int(round(width / self.TARGET_DX)))
        self.dx = width / float(self.nx)
        self.nz = max(8, int(math.ceil((z_hi - self.z_lo) / self.dx)))
        self.z_hi = self.z_lo + self.nz * self.dx
        self.inv_dx = 1.0 / self.dx

        # Cell centres of the diagnostic grid (identical to the solver grid).
        self.xc = self.x_lo + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx
        self.zc = self.z_lo + (np.arange(self.nz, dtype=np.float64) + 0.5) * self.dx
        cx_grid, cz_grid = np.meshgrid(self.xc, self.zc, indexing="ij")
        self.cell_centres = np.column_stack((cx_grid.ravel(), cz_grid.ravel()))

        # Particle rest spacing implied by the frozen sample volumes.
        spacing = math.sqrt(max(float(np.mean(self.sample_volume)), 1.0e-12))
        self.p_spacing = spacing
        self.p_radius = 0.5 * spacing
        self.ppc_target = max((self.dx / spacing) ** 2, 1.0)
        self.min_separation = 0.90 * spacing

        # Particles are kept a fraction of a radius away from solid faces.
        pad = 0.5 * self.p_radius
        self.px_lo, self.px_hi = self.x_lo + pad, self.x_hi - pad
        self.pz_lo, self.pz_hi = self.z_lo + pad, self.z_hi - pad
        np.clip(self.pos[:, 0], self.px_lo, self.px_hi, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], self.pz_lo, self.pz_hi, out=self.pos[:, 1])

        self.characteristic_length = float(hi[2] - lo[2])
        self.reference_velocity = math.sqrt(abs(self.gz) * max(self.characteristic_length, 1e-9))
        self._build_boundary_samples()

    def _reset_grids(self) -> None:
        self.u = np.zeros((self.nx + 1, self.nz), dtype=np.float64)      # x-faces
        self.w = np.zeros((self.nx, self.nz + 1), dtype=np.float64)      # z-faces
        self.u_valid = np.zeros_like(self.u, dtype=bool)
        self.w_valid = np.zeros_like(self.w, dtype=bool)
        self.cell_count = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.fluid = np.zeros((self.nx, self.nz), dtype=bool)

    def _build_boundary_samples(self) -> None:
        """Collocated wall samples: floor, left wall, right wall (inward normals)."""
        floor = np.column_stack((self.xc, np.full(self.nx, self.z_lo)))
        left = np.column_stack((np.full(self.nz, self.x_lo), self.zc))
        right = np.column_stack((np.full(self.nz, self.x_hi), self.zc))
        self.boundary_points = np.vstack((floor, left, right))
        normals = np.zeros((self.boundary_points.shape[0], 2), dtype=np.float64)
        normals[: self.nx, 1] = 1.0                       # floor -> +z
        normals[self.nx : self.nx + self.nz, 0] = 1.0     # left wall -> +x
        normals[self.nx + self.nz :, 0] = -1.0            # right wall -> -x
        self.boundary_normal = normals
        # Every tank wall is static: the prescribed wall velocity is zero.
        self.boundary_target_velocity = np.zeros_like(normals)
    # ------------------------------------------------- particle/grid transfers
    @staticmethod
    def _node_weights(g, n_nodes):
        i0 = np.floor(g).astype(np.int64)
        np.clip(i0, 0, n_nodes - 2, out=i0)
        f = np.clip(g - i0, 0.0, 1.0)
        return i0, f

    def _u_index(self, px, pz):
        i0, fx = self._node_weights((px - self.x_lo) * self.inv_dx, self.nx + 1)
        k0, fz = self._node_weights((pz - self.z_lo) * self.inv_dx - 0.5, self.nz)
        return i0, fx, k0, fz

    def _w_index(self, px, pz):
        i0, fx = self._node_weights((px - self.x_lo) * self.inv_dx - 0.5, self.nx)
        k0, fz = self._node_weights((pz - self.z_lo) * self.inv_dx, self.nz + 1)
        return i0, fx, k0, fz

    def _transfer_to_grid(self) -> None:
        """Bilinear particle -> MAC-face momentum transfer (mass weighted)."""
        px, pz = self.pos[:, 0], self.pos[:, 1]
        pm = self.mass

        u_mass = np.zeros((self.nx + 1, self.nz), dtype=np.float64)
        u_mom = np.zeros_like(u_mass)
        i0, fx, k0, fz = self._u_index(px, pz)
        for di in (0, 1):
            wx = fx if di else 1.0 - fx
            for dk in (0, 1):
                weight = wx * (fz if dk else 1.0 - fz) * pm
                np.add.at(u_mass, (i0 + di, k0 + dk), weight)
                np.add.at(u_mom, (i0 + di, k0 + dk), weight * self.vel[:, 0])

        w_mass = np.zeros((self.nx, self.nz + 1), dtype=np.float64)
        w_mom = np.zeros_like(w_mass)
        j0, gx, l0, gz = self._w_index(px, pz)
        for di in (0, 1):
            wx = gx if di else 1.0 - gx
            for dk in (0, 1):
                weight = wx * (gz if dk else 1.0 - gz) * pm
                np.add.at(w_mass, (j0 + di, l0 + dk), weight)
                np.add.at(w_mom, (j0 + di, l0 + dk), weight * self.vel[:, 1])

        self.u_valid = u_mass > 0.0
        self.w_valid = w_mass > 0.0
        self.u = np.where(self.u_valid, u_mom / np.where(self.u_valid, u_mass, 1.0), 0.0)
        self.w = np.where(self.w_valid, w_mom / np.where(self.w_valid, w_mass, 1.0), 0.0)

        ci = np.clip(((px - self.x_lo) * self.inv_dx).astype(np.int64), 0, self.nx - 1)
        ck = np.clip(((pz - self.z_lo) * self.inv_dx).astype(np.int64), 0, self.nz - 1)
        self.cell_count = np.zeros((self.nx, self.nz), dtype=np.float64)
        np.add.at(self.cell_count, (ci, ck), 1.0)
        self.fluid = self.cell_count > 0.0
    def _gather(self, u_grid, w_grid, px, pz):
        """Bilinear MAC-face -> point interpolation, returns (vx, vz)."""
        i0, fx, k0, fz = self._u_index(px, pz)
        vx = (
            (1.0 - fx) * (1.0 - fz) * u_grid[i0, k0]
            + fx * (1.0 - fz) * u_grid[i0 + 1, k0]
            + (1.0 - fx) * fz * u_grid[i0, k0 + 1]
            + fx * fz * u_grid[i0 + 1, k0 + 1]
        )
        j0, gx, l0, gz = self._w_index(px, pz)
        vz = (
            (1.0 - gx) * (1.0 - gz) * w_grid[j0, l0]
            + gx * (1.0 - gz) * w_grid[j0 + 1, l0]
            + (1.0 - gx) * gz * w_grid[j0, l0 + 1]
            + gx * gz * w_grid[j0 + 1, l0 + 1]
        )
        return vx, vz

    # ---------------------------------------------------- boundaries / air band
    def _solid_boundary_conditions(self) -> None:
        """Free-slip, no-penetration on the frozen tank floor and side walls."""
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[:, 0] = 0.0
        self.w[:, -1] = 0.0
        self.u_valid[0, :] = True
        self.u_valid[-1, :] = True
        self.w_valid[:, 0] = True
        self.w_valid[:, -1] = True

    @staticmethod
    def _extrapolate(grid, valid, sweeps):
        """Fill invalid faces with the mean of valid neighbours (air band)."""
        grid = grid.copy()
        valid = valid.copy()
        for _ in range(sweeps):
            if valid.all():
                break
            weight = np.zeros_like(grid)
            total = np.zeros_like(grid)
            src = np.where(valid, grid, 0.0)
            mask = valid.astype(np.float64)
            total[1:, :] += src[:-1, :]
            weight[1:, :] += mask[:-1, :]
            total[:-1, :] += src[1:, :]
            weight[:-1, :] += mask[1:, :]
            total[:, 1:] += src[:, :-1]
            weight[:, 1:] += mask[:, :-1]
            total[:, :-1] += src[:, 1:]
            weight[:, :-1] += mask[:, 1:]
            fillable = (~valid) & (weight > 0.0)
            if not fillable.any():
                break
            grid[fillable] = total[fillable] / weight[fillable]
            valid |= fillable
        return grid
    # ------------------------------------------------------ pressure projection
    def _project(self, dt: float) -> None:
        """Exact discrete projection: solve the variational Poisson system so
        that div(u) = 0 in liquid cells, p = 0 in air, dp/dn = 0 at solids."""
        fluid = self.fluid
        n_fluid = int(fluid.sum())
        if n_fluid == 0:
            return

        index = np.full((self.nx, self.nz), -1, dtype=np.int64)
        flat = np.flatnonzero(fluid.ravel())
        index.ravel()[flat] = np.arange(n_fluid, dtype=np.int64)

        div = (
            (self.u[1:, :] - self.u[:-1, :]) + (self.w[:, 1:] - self.w[:, :-1])
        ) * self.inv_dx

        rhs = (-div)[fluid]

        scale = dt / (self.density * self.dx * self.dx)
        rows = [np.arange(n_fluid, dtype=np.int64)]
        cols = [np.arange(n_fluid, dtype=np.int64)]
        diag = np.zeros(n_fluid, dtype=np.float64)
        data = []
        fi = index[fluid]  # == arange(n_fluid)

        for axis, shift in ((0, -1), (0, 1), (1, -1), (1, 1)):
            neighbour = np.full((self.nx, self.nz), -2, dtype=np.int64)  # -2 == solid
            if axis == 0 and shift == -1:
                neighbour[1:, :] = index[:-1, :]
            elif axis == 0 and shift == 1:
                neighbour[:-1, :] = index[1:, :]
            elif axis == 1 and shift == -1:
                neighbour[:, 1:] = index[:, :-1]
            else:
                neighbour[:, :-1] = index[:, 1:]
            nb = neighbour[fluid]
            non_solid = nb != -2
            diag[non_solid] += scale
            link = nb >= 0
            if np.any(link):
                rows.append(fi[link])
                cols.append(nb[link])
                data.append(np.full(int(link.sum()), -scale, dtype=np.float64))

        # Guard against a fully solid-enclosed cell (cannot occur in this tank,
        # but keeps the system non-singular for any geometry).
        diag = np.where(diag > 0.0, diag, scale)
        data.insert(0, diag)
        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_fluid, n_fluid),
        ).tocsr()

        pressure_values = spsolve(matrix.tocsc(), rhs)
        pressure_values = np.nan_to_num(
            np.asarray(pressure_values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        pressure = np.zeros((self.nx, self.nz), dtype=np.float64)
        pressure[fluid] = pressure_values

        coefficient = dt / (self.density * self.dx)
        u_active = fluid[1:, :] | fluid[:-1, :]
        self.u[1:-1, :] -= coefficient * np.where(
            u_active, pressure[1:, :] - pressure[:-1, :], 0.0
        )
        self.u_valid[1:-1, :] |= u_active

        w_active = fluid[:, 1:] | fluid[:, :-1]
        self.w[:, 1:-1] -= coefficient * np.where(
            w_active, pressure[:, 1:] - pressure[:, :-1], 0.0
        )
        self.w_valid[:, 1:-1] |= w_active

        self._solid_boundary_conditions()
    # ------------------------------------------------------------- time stepping
    def _step(self, dt: float) -> None:
        self._transfer_to_grid()

        # Explicit body force on transferred face momenta, applied only where
        # particles actually deposited momentum.
        self.u += dt * self.gx
        self.w += dt * self.gz
        self._solid_boundary_conditions()

        # Pre-projection state, extrapolated with its own validity mask so the
        # FLIP delta is never contaminated by empty-face zeros.
        u_valid_p2g = self.u_valid.copy()
        w_valid_p2g = self.w_valid.copy()
        u_old = self._extrapolate(self.u, u_valid_p2g, self.EXTRAP_SWEEPS)
        w_old = self._extrapolate(self.w, w_valid_p2g, self.EXTRAP_SWEEPS)

        self._project(dt)

        u_new = self._extrapolate(self.u, self.u_valid, self.EXTRAP_SWEEPS)
        w_new = self._extrapolate(self.w, self.w_valid, self.EXTRAP_SWEEPS)
        self.u, self.w = u_new, w_new
        self._solid_boundary_conditions()

        px = self.pos[:, 0].copy()
        pz = self.pos[:, 1].copy()

        # FLIP/PIC blend: PIC supplies the physical dissipation that lets the
        # post-impact sloshing decay instead of ringing forever.
        # The gravity impulse is already contained in (pic - old); adding it
        # again would double-count the body force.
        pic_x, pic_z = self._gather(self.u, self.w, px, pz)
        old_x, old_z = self._gather(u_old, w_old, px, pz)
        flip_x = self.vel[:, 0] + (pic_x - old_x)
        flip_z = self.vel[:, 1] + (pic_z - old_z)
        blend = self.FLIP_RATIO
        self.vel[:, 0] = blend * flip_x + (1.0 - blend) * pic_x
        self.vel[:, 1] = blend * flip_z + (1.0 - blend) * pic_z

        # RK2 advection in the projected, divergence-free field.
        mid_x = np.clip(px + 0.5 * dt * pic_x, self.px_lo, self.px_hi)
        mid_z = np.clip(pz + 0.5 * dt * pic_z, self.pz_lo, self.pz_hi)
        mid_vx, mid_vz = self._gather(self.u, self.w, mid_x, mid_z)
        self.pos[:, 0] = px + dt * mid_vx
        self.pos[:, 1] = pz + dt * mid_vz

        self._collide_walls()
        self._push_apart()

        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed_cap = 40.0
        np.clip(self.vel, -speed_cap, speed_cap, out=self.vel)
        self.substeps += 1

    def _collide_walls(self) -> None:
        """Inelastic no-penetration response on the closed tank walls."""
        x, z = self.pos[:, 0], self.pos[:, 1]
        low_x = x < self.px_lo
        high_x = x > self.px_hi
        low_z = z < self.pz_lo
        high_z = z > self.pz_hi
        if np.any(low_x):
            x[low_x] = self.px_lo
            self.vel[low_x, 0] = np.maximum(self.vel[low_x, 0], 0.0)
        if np.any(high_x):
            x[high_x] = self.px_hi
            self.vel[high_x, 0] = np.minimum(self.vel[high_x, 0], 0.0)
        if np.any(low_z):
            z[low_z] = self.pz_lo
            self.vel[low_z, 1] = np.maximum(self.vel[low_z, 1], 0.0)
        if np.any(high_z):
            z[high_z] = self.pz_hi
            self.vel[high_z, 1] = np.minimum(self.vel[high_z, 1], 0.0)

    def _push_apart(self) -> None:
        """Symmetric pairwise relaxation towards the rest sample spacing.

        This is the particle-level companion of the grid projection: it removes
        the sub-cell clumping FLIP cannot see, without moving mass in a
        prescribed direction (the correction is equal and opposite per pair).
        """
        r = self.min_separation
        if r <= 0.0:
            return
        for _ in range(self.PUSH_ITERS):
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(r, output_type="ndarray")
            if pairs.size == 0:
                return
            a, b = pairs[:, 0], pairs[:, 1]
            delta = self.pos[a] - self.pos[b]
            dist = np.linalg.norm(delta, axis=1)
            degenerate = dist < 1.0e-9
            if np.any(degenerate):
                # Deterministic tie-break for exactly coincident samples.
                delta[degenerate, 0] = np.where(a[degenerate] < b[degenerate], 1.0, -1.0)
                delta[degenerate, 1] = 0.0
                dist = np.where(degenerate, 1.0, dist)
            direction = delta / dist[:, None]
            push = (0.5 * self.PUSH_STIFFNESS * (r - dist))[:, None] * direction
            np.add.at(self.pos, a, push)
            np.add.at(self.pos, b, -push)
            np.clip(self.pos[:, 0], self.px_lo, self.px_hi, out=self.pos[:, 0])
            np.clip(self.pos[:, 1], self.pz_lo, self.pz_hi, out=self.pos[:, 1])
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1.0e-12:
            remaining = target - self.time
            speed = float(np.max(np.abs(self.vel))) if self.n_particles else 0.0
            self.max_speed_seen = max(self.max_speed_seen, speed)
            limit = self.MAX_SUBSTEP
            if speed > 1.0e-6:
                limit = min(limit, self.CFL * self.dx / speed)
            limit = max(limit, self.MIN_SUBSTEP)
            count = max(1, int(math.ceil(remaining / limit)))
            dt = remaining / count
            for _ in range(count):
                self._step(dt)
            self.time = target
        self.time = target
        return self._make_frame(target)

    # ------------------------------------------------------------- diagnostics
    def _diagnostic_fields(self):
        """Cell-centred velocity, particle-based level set and interior mask."""
        velocity_field = np.empty((self.nx, self.nz, 2), dtype=np.float64)
        velocity_field[:, :, 0] = 0.5 * (self.u[1:, :] + self.u[:-1, :])
        velocity_field[:, :, 1] = 0.5 * (self.w[:, 1:] + self.w[:, :-1])

        tree = cKDTree(self.pos)
        distance, _ = tree.query(self.cell_centres, k=1)
        level_set = distance.reshape(self.nx, self.nz) - self.p_radius - 0.5 * self.p_spacing
        interior_mask = (level_set <= 0.0).astype(np.float64)
        return velocity_field, level_set, interior_mask

    def _boundary_velocity(self):
        """Fluid-side velocity sampled at the collocated wall points.

        Free-slip walls: the wall-normal component is exactly zero, the
        tangential component is the neighbouring fluid velocity.
        """
        floor = np.zeros((self.nx, 2), dtype=np.float64)
        floor[:, 0] = 0.5 * (self.u[1:, 0] + self.u[:-1, 0])
        floor[:, 1] = self.w[:, 0]

        left = np.zeros((self.nz, 2), dtype=np.float64)
        left[:, 0] = self.u[0, :]
        left[:, 1] = 0.5 * (self.w[0, 1:] + self.w[0, :-1])

        right = np.zeros((self.nz, 2), dtype=np.float64)
        right[:, 0] = self.u[-1, :]
        right[:, 1] = 0.5 * (self.w[-1, 1:] + self.w[-1, :-1])
        return np.vstack((floor, left, right))

    def _make_frame(self, time: float) -> StateFrame:
        position = np.zeros((self.n_particles, 3), dtype=np.float64)
        position[:, 0] = self.pos[:, 0]
        position[:, 1] = self.y_plane
        position[:, 2] = self.pos[:, 1]
        velocity = np.zeros((self.n_particles, 3), dtype=np.float64)
        velocity[:, 0] = self.vel[:, 0]
        velocity[:, 2] = self.vel[:, 1]
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

        velocity_field, level_set, interior_mask = self._diagnostic_fields()
        boundary_velocity = self._boundary_velocity()

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_particles, dtype=np.int64),
            attributes={
                "phase_id": np.full(self.n_particles, 1.0, dtype=np.float64),
                "sample_volume": self.sample_volume.copy(),
                "mass_quantity": self.mass.copy(),
                "volume_quantity": self.sample_volume.copy(),
            },
            fields={
                "velocity_field": np.nan_to_num(velocity_field),
                "liquid_interior_mask": interior_mask,
                "liquid_level_set": np.nan_to_num(level_set),
                "boundary_velocity": np.nan_to_num(boundary_velocity),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_normal": self.boundary_normal.copy(),
            },
            scalars={
                "density": self.density,
                "grid_spacing": float(self.dx),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
                "liquid_phase_id": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MAC-grid FLIP/PIC incompressible free-surface solver",
            "grid_shape": [int(self.nx), int(self.nz)],
            "grid_spacing": float(self.dx),
            "particles": int(self.n_particles),
            "substeps": int(self.substeps),
            "flip_ratio": float(self.FLIP_RATIO),
            "max_particle_speed": float(self.max_speed_seen),
            "final_time": float(self.time),
            "seed": int(self.seed),
        }
