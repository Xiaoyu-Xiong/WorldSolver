"""Solver Agent-owned implementation for the octagonal two-phase drop benchmark.

This solver implements a particle-based weakly-compressible two-phase fluid model.
The heavy (blue) phase and the four-times-lighter (pink) phase are represented as
particles with per-phase rest densities.  Gravity, pairwise repulsive/viscous
forces, and impermeable octagonal wall collision keep the simulation stable and
physical.  The light phase is held fixed until its release time.

No Genesis / Taichi / external simulator is used; only NumPy and SciPy.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - runtime import surface
    StateFrame = None


class BenchmarkSolver:
    """Explicit two-phase particle solver with diagnostic grid exports."""

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = getattr(scene, "config", {})

        # Pull the deterministic initial state from the scaffold.
        frame0 = scene.initial_state()

        # Per-particle state.
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.n_particles = self.pos.shape[0]

        # Attributes are required for per-material bookkeeping.
        self.attributes = {}
        for key, value in (frame0.attributes or {}).items():
            self.attributes[key] = np.asarray(value).copy()

        self.phase_id = self.attributes["phase_id"].astype(np.int32)
        self.density = self.attributes["density"].astype(float)
        self.release_time = self.attributes["release_time"].astype(float)
        self.target_volume = self.attributes["target_volume"].astype(float)

        # Base scalars from the scaffold.
        self.scalars = {}
        for key, value in (frame0.scalars or {}).items():
            self.scalars[key] = float(value)

        # Gravity vector.
        gravity = self.config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=float)

        # Physical parameters.
        self.radius = ((3.0 * self.target_volume) / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.cutoff = 0.08  # smoothing / contact cutoff (m)
        self.k_pair = 4.0e4  # pairwise repulsion stiffness (Pa m)
        self.gamma_pair = 10.0  # pairwise damping (kg/(m s))
        self.k_wall = 8.0e4  # wall repulsion stiffness (Pa m)
        self.gamma_wall = 5.0  # wall damping (kg/(m s))
        self.wall_margin = 0.005  # small margin for wall forces (m)

        # Fixed time step.  Chosen to keep the explicit spring forces stable.
        self.dt = 2.0e-4  # seconds

        # Keep a copy of the initial positions for particles that are not yet released.
        self.initial_pos = self.pos.copy()

        # Simulation clock.
        self.t = 0.0

        # Boundary geometry: inner octagonal prism walls.
        self._build_octagon_planes()

        # Diagnostic grid for the required trajectory fields.
        self._build_diagnostic_grid()

        # Cache the most recent frame.
        self._last_frame = self._make_frame(self.t)

    def advance_to(self, time: float):
        if time < self.t:
            return self._last_frame

        # Sub-step to the requested sample time.
        while self.t < time - 1.0e-12:
            step_dt = min(self.dt, time - self.t)
            self._substep(step_dt)
            self.t += step_dt

        self._last_frame = self._make_frame(self.t)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": self.n_particles,
            "time": self.t,
            "dt": self.dt,
            "cutoff": self.cutoff,
        }

    # --------------------------------------------------------------------- #
    # sub-stepping
    # --------------------------------------------------------------------- #
    def _substep(self, dt: float) -> None:
        active_mask = self.t >= self.release_time
        active_idx = np.nonzero(active_mask)[0]
        if active_idx.size == 0:
            return

        # Work only with active particles.
        pos_a = self.pos[active_idx].copy()
        vel_a = self.vel[active_idx].copy()
        mass_a = self.mass[active_idx]
        radius_a = self.radius[active_idx]

        # Reset held (inactive) particles to their initial state.
        inactive_idx = np.nonzero(~active_mask)[0]
        if inactive_idx.size:
            self.vel[inactive_idx] = 0.0
            self.pos[inactive_idx] = self.initial_pos[inactive_idx]

        # Start with gravity.
        forces = mass_a[:, None] * self.gravity[None, :]

        # Pairwise repulsion and viscous damping among active particles.
        if pos_a.shape[0] > 1:
            i, j, dx, d = self._neighbors(pos_a)
            if i.size:
                xhat = dx / d[:, None]
                overlap = self.cutoff - d
                f_mag = self.k_pair * overlap
                v_rel = vel_a[i] - vel_a[j]
                # Damping opposes compression/approach.
                damp = self.gamma_pair * np.einsum("ij,ij->i", v_rel, xhat)
                f_pair = ((f_mag + damp) / d)[:, None] * dx

                np.add.at(forces, i, f_pair)
                np.add.at(forces, j, -f_pair)

        # Wall repulsion for each active particle.
        for n, p0 in self.walls:
            # Signed distance (positive outside the octagon).
            dist = np.dot(pos_a - p0, n)
            near = dist > -self.wall_margin
            if not np.any(near):
                continue
            vel_n = np.dot(vel_a[near], n)
            repulsion = -self.k_wall * (self.wall_margin + dist[near])
            repulsion = np.clip(repulsion, -1.0e6, 1.0e6)
            damping = -self.gamma_wall * np.maximum(vel_n, 0.0)
            wall_force = (repulsion + damping)[:, None] * n[None, :]
            forces[near] += wall_force

        # Bottom floor at the inner basin floor z = 0.08 m.
        z_floor = 0.08
        dz = pos_a[:, 2] - z_floor
        near_floor = dz > -self.wall_margin
        if np.any(near_floor):
            normal = np.array([0.0, 0.0, 1.0])
            vel_n = vel_a[near_floor, 2]
            repulsion = -self.k_wall * (self.wall_margin + dz[near_floor])
            repulsion = np.clip(repulsion, -1.0e6, 1.0e6)
            damping = -self.gamma_wall * np.maximum(vel_n, 0.0)
            wall_force = (repulsion + damping)[:, None] * normal[None, :]
            forces[near_floor] += wall_force

        # Symplectic Euler integration.
        vel_a = vel_a + dt * forces / mass_a[:, None]
        pos_a = pos_a + dt * vel_a

        # Hard wall projection / velocity reflection with damping.
        for n, p0 in self.walls:
            dist = np.dot(pos_a - p0, n)
            outside = dist > 0.0
            if not np.any(outside):
                continue
            pos_a[outside] -= (dist[outside] + self.wall_margin)[:, None] * n[None, :]
            vel_n = np.dot(vel_a[outside], n)
            if vel_n.size:
                vel_a[outside] -= 1.0 * np.maximum(vel_n, 0.0)[:, None] * n[None, :]

        # Bottom projection.
        below = pos_a[:, 2] < z_floor
        if np.any(below):
            pos_a[below, 2] = z_floor + self.wall_margin
            vel_a[below, 2] = np.maximum(vel_a[below, 2], 0.0)

        # Commit active state back to the full arrays.
        self.pos[active_idx] = pos_a
        self.vel[active_idx] = vel_a

    # --------------------------------------------------------------------- #
    # frame construction
    # --------------------------------------------------------------------- #
    def _make_frame(self, time: float):
        # Per-material mass quantity: shape [N, 2].
        mass_quantity = np.zeros((self.n_particles, 2), dtype=float)
        for k in (0, 1):
            mask = self.phase_id == k
            if np.any(mask):
                mass_quantity[mask, k] = self.mass[mask]

        active_mask, level_set, velocity_field = self._compute_grid_fields()

        fields = {
            "active_mask": active_mask,
            "cumulative_mass_source": np.zeros(2, dtype=float),
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
        }

        attributes = dict(self.attributes)
        attributes["mass_quantity"] = mass_quantity

        scalars = dict(self.scalars)
        scalars["characteristic_length"] = 1.15  # basin inner radius (m)
        scalars["grid_spacing"] = float(self.grid_spacing)

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            scalars=scalars,
            fields=fields,
        )

    # --------------------------------------------------------------------- #
    # diagnostic grid
    # --------------------------------------------------------------------- #
    def _build_diagnostic_grid(self) -> None:
        # Grid covers the octagonal basin plus the falling liquid.
        self.grid_spacing = 0.07  # m
        self.grid_origin = np.array([-1.35, -1.35, -0.05], dtype=float)
        self.grid_shape = (39, 39, 63)  # covers up to z ~ 4.36 m
        self.nx, self.ny, self.nz = self.grid_shape

    def _compute_grid_fields(self):
        # Build a list of grid point coordinates.
        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        iz = np.arange(self.nz)
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        points = np.stack(
            [
                self.grid_origin[0] + gx * self.grid_spacing,
                self.grid_origin[1] + gy * self.grid_spacing,
                self.grid_origin[2] + gz * self.grid_spacing,
            ],
            axis=-1,
        ).reshape(-1, 3)

        # Signed distance to the nearest particle surface.
        # Use a fast approximate SPH-style level set.
        from scipy.spatial import cKDTree

        tree = cKDTree(self.pos)
        dist, idx = tree.query(points)
        r_nearest = self.radius[idx]
        level = dist - r_nearest

        active_mask = level < 0.0
        level_set = level.reshape(self.grid_shape)

        # Diagnostic velocity field: nearest-particle velocity.
        velocity_field = self.vel[idx].reshape(
            self.grid_shape[0], self.grid_shape[1], self.grid_shape[2], 3
        )
        velocity_field = velocity_field.copy()
        velocity_field[~active_mask] = 0.0
        active_mask = active_mask.reshape(self.grid_shape)

        return active_mask, level_set, velocity_field

    # --------------------------------------------------------------------- #
    # octagonal basin boundary
    # --------------------------------------------------------------------- #
    def _build_octagon_planes(self) -> None:
        # Inner octagonal wall ring from assets/octagonal_basin.obj, ordered CCW.
        vertices = np.array(
            [
                [1.060660, 0.439340],
                [0.439340, 1.060660],
                [-0.439340, 1.060660],
                [-1.060660, 0.439340],
                [-1.060660, -0.439340],
                [-0.439340, -1.060660],
                [0.439340, -1.060660],
                [1.060660, -0.439340],
            ],
            dtype=float,
        )

        walls = []
        for i in range(vertices.shape[0]):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % vertices.shape[0]]
            edge = p2 - p1
            # Candidate outward normal (away from the origin).
            cand = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(cand)
            if norm < 1.0e-12:
                continue
            cand = cand / norm
            # Choose the normal that points away from the origin.
            if np.dot(cand, 0.5 * (p1 + p2)) < 0.0:
                cand = -cand
            n3 = np.array([cand[0], cand[1], 0.0])
            p0 = np.array([p1[0], p1[1], 0.0])
            walls.append((n3, p0))
        self.walls = walls

    # --------------------------------------------------------------------- #
    # neighbor search (uniform grid)
    # --------------------------------------------------------------------- #
    def _neighbors(self, pos: np.ndarray):
        cell_size = self.cutoff
        origin = np.min(pos, axis=0) - 0.5 * cell_size
        coords = np.floor((pos - origin) / cell_size).astype(np.int32)
        # Clamp to non-negative.
        coords -= np.min(coords, axis=0)
        nxg = int(np.max(coords[:, 0]) + 1)
        nyg = int(np.max(coords[:, 1]) + 1)
        nzg = int(np.max(coords[:, 2]) + 1)

        # Build a dictionary mapping cell index to particle indices.
        cells = {}
        for idx, c in enumerate(coords):
            key = (int(c[0]), int(c[1]), int(c[2]))
            cell = cells.setdefault(key, [])
            cell.append(idx)

        i_list = []
        j_list = []
        d_list = []
        dx_list = []
        for c0 in range(nxg):
            for c1 in range(nyg):
                for c2 in range(nzg):
                    cell = cells.get((c0, c1, c2), [])
                    if not cell:
                        continue
                    for dz in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                nb = cells.get((c0 + dx, c1 + dy, c2 + dz), [])
                                for a in cell:
                                    for b in nb:
                                        if a >= b:
                                            continue
                                        diff = pos[a] - pos[b]
                                        dist2 = np.dot(diff, diff)
                                        if dist2 < self.cutoff * self.cutoff and dist2 > 1.0e-24:
                                            i_list.append(a)
                                            j_list.append(b)
                                            d_list.append(math.sqrt(dist2))
                                            dx_list.append(diff)
        if not i_list:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return np.array(i_list), np.array(j_list), np.array(dx_list), np.array(d_list)
