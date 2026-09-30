"""Physically-based liquid solver using Position Based Fluids (PBF).

The two material labels (pool and liquid bunny) are treated as a single incompressible
fluid.  A density constraint enforces near-incompressibility, gravity drives the fall,
and a sampled rigid boundary plus hard-wall projection keeps the liquid inside the
basin.  Diagnostic fields (level set and velocity) are reconstructed on the requested
cartesian grid at every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - only used at runtime by the platform runner
    StateFrame = None  # type: ignore


class BenchmarkSolver:
    """PBF liquid solver for the bunny-pool impact scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        init = scene.initial_state()

        self.t: float = 0.0
        self.pos = np.asarray(init.position, dtype=float).copy()
        self.vel = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.labels = np.asarray(init.body_id, dtype=np.int32).copy()

        # Parameters taken from the frozen configuration.
        sampling = scene.config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.rho = float(sampling["density"])
        self.gravity = np.asarray(scene.config["gravity"], dtype=float)

        # PBF tuning.
        self.h = self.spacing * 1.5
        self.support = 2.0 * self.h
        self.pbf_iters = 3
        # Substep small enough to keep PBF stable during the splash.
        self.dt = 0.004
        self.epsilon = 1e-6

        # Per-particle radius used for collision / level set.
        self.particle_radius = np.asarray(init.attributes["particle_radius"], dtype=float)

        # Diagnostic grid.
        self.grid_shape = tuple(int(s) for s in sampling["field_grid_shape"])
        self.grid_origin = np.array(
            [-(self.grid_shape[0] * self.spacing) / 2.0,
             -(self.grid_shape[1] * self.spacing) / 2.0,
             0.0],
            dtype=float,
        )

        # Rigid boundary samples (used only for output; dynamics handled by projection).
        self.boundary_pos, self.boundary_normal = self._build_boundary_samples()

        # Compute the dimensionless rest-density sum from the still pool block.
        # The pool particles are those with material_label == 0 at t=0.
        pool_mask = self.labels == 0
        self.rest_sum = self._compute_rest_sum(self.pos[pool_mask])
        if self.rest_sum <= 0.0:
            self.rest_sum = 1.0

        # Preserve scalars from the initial frame.
        self.scalars = dict(init.scalars)

    # --------------------------------------------------------------------- #
    # Kernel
    # --------------------------------------------------------------------- #
    def _kernel(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cubic spline kernel and its derivative w.r.t. q.

        Returns ``(k, dk/dq)`` for the 3-D normalized kernel (k(0)=1)."""
        q = np.asarray(q, dtype=float)
        k = np.zeros_like(q)
        kp = np.zeros_like(q)

        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)

        k[m1] = 1.0 - 1.5 * q[m1] * q[m1] + 0.75 * q[m1] * q[m1] * q[m1]
        kp[m1] = -3.0 * q[m1] + 2.25 * q[m1] * q[m1]

        t = 2.0 - q[m2]
        k[m2] = 0.25 * t * t * t
        kp[m2] = -0.75 * t * t
        return k, kp

    # --------------------------------------------------------------------- #
    # Boundary construction
    # --------------------------------------------------------------------- #
    def _build_boundary_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return boundary point positions and outward unit normals."""
        dx = self.spacing
        pts: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        def add_plane(center_lo, center_hi, normal):
            normal = np.asarray(normal, dtype=float)
            normal /= np.linalg.norm(normal)
            lo = np.asarray(center_lo, dtype=float)
            hi = np.asarray(center_hi, dtype=float)
            # Build a regular grid on the plane.
            axes = [np.arange(lo[i], hi[i] + dx * 0.5, dx) for i in range(3)]
            # Only keep the axis that actually varies matter; the other two are fixed
            # by the plane.  Build the full grid and discard out-of-plane points.
            g = np.meshgrid(*axes, indexing="ij")
            grid = np.stack([g_i.ravel() for g_i in g], axis=1)
            # Clamp the fixed coordinates to the plane center.
            for i in range(3):
                if hi[i] - lo[i] < dx * 0.25:
                    grid[:, i] = (lo[i] + hi[i]) * 0.5
            pts.append(grid)
            nrm.append(np.tile(normal, (grid.shape[0], 1)))

        # Ground (top at z=0, solid below, normal +z).
        add_plane([-1.7, -1.0, -0.07], [1.7, 0.84, -0.07], [0.0, 0.0, 1.0])
        # Left wall (inner face near x=-1.77, solid outside at x=-1.85, normal -x).
        add_plane([-1.85, -0.92, 0.0], [-1.85, 0.84, 0.6], [-1.0, 0.0, 0.0])
        # Right wall.
        add_plane([1.85, -0.92, 0.0], [1.85, 0.84, 0.6], [1.0, 0.0, 0.0])
        # Back wall (inner face near y=0.84, solid beyond at y=0.92, normal +y).
        add_plane([-1.7, 0.92, 0.0], [1.7, 0.92, 0.6], [0.0, 1.0, 0.0])

        return np.concatenate(pts, axis=0), np.concatenate(nrm, axis=0)

    # --------------------------------------------------------------------- #
    # Rest density
    # --------------------------------------------------------------------- #
    def _compute_rest_sum(self, pos: np.ndarray) -> float:
        """Average dimensionless density sum for a regular liquid sample."""
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        N = pos.shape[0]
        sums = np.ones(N, dtype=float)
        if pairs.size:
            rvec = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            r = np.linalg.norm(rvec, axis=1)
            q = r / self.h
            k, _ = self._kernel(q)
            np.add.at(sums, pairs[:, 0], k)
            np.add.at(sums, pairs[:, 1], k)
        return float(np.mean(sums))

    # --------------------------------------------------------------------- #
    # Time stepping
    # --------------------------------------------------------------------- #
    def _apply_boundaries(self, pos: np.ndarray, vel: np.ndarray):
        """Hard-wall projection and velocity reflection for the basin."""
        pos = pos.copy()
        vel = vel.copy()
        r = self.particle_radius

        # Basin inner extents (meters).  Allow a small margin equal to particle radius.
        for axis, lo, hi in [(0, -1.77, 1.77), (2, 0.0, 1e9)]:
            if lo is not None:
                mask = pos[:, axis] < lo + r
                pos[mask, axis] = lo + r[mask]
                vel[mask, axis] = np.abs(vel[mask, axis]) * 0.3
            if hi is not None and hi < 1e8:
                mask = pos[:, axis] > hi - r
                pos[mask, axis] = hi - r[mask]
                vel[mask, axis] = -np.abs(vel[mask, axis]) * 0.3

        # Back wall (y max).
        mask = pos[:, 1] > 0.84 - r
        pos[mask, 1] = 0.84 - r[mask]
        vel[mask, 1] = np.abs(vel[mask, 1]) * 0.3

        # Soft floor for the open front (y min) to keep particles in frame.
        mask = pos[:, 1] < -1.2 + r
        pos[mask, 1] = -1.2 + r[mask]
        vel[mask, 1] = np.abs(vel[mask, 1]) * 0.3

        return pos, vel

    def _pbf_solve(self, pos: np.ndarray) -> np.ndarray:
        """Apply one PBF density-projection pass to ``pos``; return corrected positions."""
        N = pos.shape[0]
        # Include static boundary particles so the fluid feels the walls.
        pos_all = np.vstack([pos, self.boundary_pos])
        tree = cKDTree(pos_all)
        pairs = tree.query_pairs(self.support, output_type="ndarray")

        # Pair data.
        a = pairs[:, 0]
        b = pairs[:, 1]
        rvec = pos_all[a] - pos_all[b]
        r = np.linalg.norm(rvec, axis=1)
        # Avoid division by zero for coincident points.
        r = np.maximum(r, 1e-12)
        q = r / self.h
        k, kp = self._kernel(q)
        nvec = rvec / r[:, None]

        # Dimensionless density contribution and its positional gradient.
        f_val = k / self.rest_sum
        g_vec = (kp[:, None] / (self.rest_sum * self.h)) * nvec

        # Fluid flags (first N entries of pos_all are fluid).
        a_fluid = a < N
        b_fluid = b < N

        # Density sums C_i = (sum f) - 1, with self contribution f(0)=1/rest_sum.
        Csum = np.full(N, 1.0 / self.rest_sum)
        np.add.at(Csum, a[a_fluid], f_val[a_fluid])
        np.add.at(Csum, b[b_fluid], f_val[b_fluid])
        C = Csum - 1.0

        # Lambda denominator.
        grad_sq = np.zeros(N)
        g2 = np.einsum("ij,ij->i", g_vec, g_vec)
        np.add.at(grad_sq, a[a_fluid], g2[a_fluid])
        np.add.at(grad_sq, b[b_fluid], g2[b_fluid])
        lam = -C / (grad_sq + self.epsilon)

        # Position correction: delta p_i = sum_j (lambda_i + lambda_j) * g_vec.
        S = np.zeros(pairs.shape[0])
        S += lam[a]
        if np.any(b_fluid):
            S[b_fluid] += lam[b[b_fluid]]

        d = S[:, None] * g_vec
        delta = np.zeros((N, 3))
        np.add.at(delta, a[a_fluid], d[a_fluid])
        np.add.at(delta, b[b_fluid], -d[b_fluid])

        pos = pos + delta
        return pos

    def _substep(self, pos: np.ndarray, vel: np.ndarray):
        """Advance the fluid state by one PBF substep."""
        # Explicit force integration.
        vel = vel + self.gravity * self.dt
        pos_pred = pos + vel * self.dt

        # PBF density projection.
        for _ in range(self.pbf_iters):
            pos_pred = self._pbf_solve(pos_pred)

        # Update velocity from position change.
        vel = (pos_pred - pos) / self.dt

        # Boundary projection.
        pos_pred, vel = self._apply_boundaries(pos_pred, vel)

        # Very light damping to tame high-frequency jitter while preserving splash.
        vel *= 0.999
        return pos_pred, vel

    # --------------------------------------------------------------------- #
    # Trajectory construction
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float):
        while self.t < time - 1e-12:
            dt = min(self.dt, time - self.t)
            if dt <= 0.0:
                break
            pos_new, vel_new = self._substep(self.pos, self.vel)
            self.pos, self.vel = pos_new, vel_new
            self.t += dt
        return self._build_frame(time)

    def _grid_points(self) -> np.ndarray:
        """Return all diagnostic-grid cell centers as an (M,3) array."""
        idx = np.indices(self.grid_shape, dtype=float).reshape(3, -1)
        # cell-centered coordinates
        xyz = self.grid_origin[:, None] + (idx + 0.5) * self.spacing
        return xyz.T

    def _build_frame(self, time: float) -> StateFrame:
        # Diagnostic grid.
        grid = self._grid_points()
        active = self._active_mask(grid)
        level_set = self._level_set(grid)
        velocity = self._velocity_field(grid)

        # Boundary fields.
        b_vel = np.zeros_like(self.boundary_pos)

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.particle_radius.copy(),
            "material_label": self.labels.copy(),
            "deformation_gradient": np.repeat(np.eye(3)[None], self.pos.shape[0], axis=0),
        }

        fields = {
            "active_mask": active.reshape(self.grid_shape),
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": b_vel.copy(),
            "boundary_velocity": b_vel.copy(),
            "cumulative_mass_source": np.zeros(max(2, int(self.labels.max()) + 1), dtype=float),
            "liquid_level_set": level_set.reshape(self.grid_shape),
            "velocity_field": velocity.reshape(self.grid_shape + (3,)),
        }

        scalars = {
            "characteristic_length": self.scalars.get("characteristic_length", self.spacing),
            "grid_spacing": self.scalars.get("grid_spacing", self.spacing),
            "reference_velocity": self.scalars.get("reference_velocity", 1.0),
        }

        return StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def _active_mask(self, grid: np.ndarray) -> np.ndarray:
        """Cells inside the free (non-solid) basin region."""
        x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
        inside = (x > -1.77) & (x < 1.77) & (y < 0.84) & (y > -1.2) & (z > 0.0) & (z < 3.6)
        return inside

    def _level_set(self, grid: np.ndarray) -> np.ndarray:
        """Signed distance to the fluid surface (negative inside)."""
        tree = cKDTree(self.pos)
        dist, _ = tree.query(grid, k=1)
        r = float(np.mean(self.particle_radius))
        return dist - r

    def _velocity_field(self, grid: np.ndarray) -> np.ndarray:
        """SPH interpolation of particle velocity onto the diagnostic grid."""
        tree = cKDTree(self.pos)
        # k-nearest with a compact kernel is sufficient for the diagnostic field.
        dist, idx = tree.query(grid, k=24, distance_upper_bound=self.support)
        valid = np.isfinite(dist)

        q = dist / self.h
        k, _ = self._kernel(q)

        numerator = np.zeros((grid.shape[0], 3))
        denominator = np.zeros(grid.shape[0])

        # Vectorized accumulation over the 24 neighbours.
        for n in range(k.shape[1]):
            w = k[:, n]
            mask = valid[:, n] & (w > 0.0)
            j = idx[:, n]
            np.add.at(numerator, (np.arange(grid.shape[0])[mask], slice(None)),
                      self.vel[j[mask]] * w[mask, None])
            np.add.at(denominator, np.arange(grid.shape[0])[mask], w[mask])

        denom = denominator[:, None]
        out = np.zeros_like(numerator)
        nz = denom.ravel() > 0.0
        out[nz] = numerator[nz] / denom[nz]
        return out

    def diagnostics(self) -> dict:
        return {
            "num_particles": int(self.pos.shape[0]),
            "rest_sum": float(self.rest_sum),
            "dt": float(self.dt),
        }
