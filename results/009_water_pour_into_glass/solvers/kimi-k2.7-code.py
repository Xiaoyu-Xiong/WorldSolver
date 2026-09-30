"""SPH-style particle solver for water poured into a cylindrical glass.

The implementation uses a weakly-compressible-like pairwise repulsion model
(discrete element / artificial pressure) on the Lagrangian particles supplied
by ``scene.initial_state()``.  Gravity, inflow, and a static glass boundary are
stepped with a stable symplectic integrator.  Diagnostic grid fields are
reconstructed on the reference grid at each sample time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically-based particle solver for the water-pour scene."""

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.ic = self.config["initial_conditions"]
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        # Particle / fluid constants.
        self.rho0 = float(self.ic["liquid_density"])
        self.h = float(self.ic["particle_spacing"])
        self.mass = self.rho0 * self.h ** 3
        self.jet_radius = float(self.ic["jet_radius"])
        self.jet_speed = float(self.ic["jet_speed"])
        self.source_start = float(self.ic["source_start_time"])
        self.source_end = float(self.ic["source_end_time"])
        self.inflow_rate = float(self.ic["target_inflow_rate_m3_s"])

        # Source geometry.
        self.source_center = np.array(self.ic["jet_end"], dtype=np.float64)
        self.source_dir = np.array(self.ic["source_direction"], dtype=np.float64)
        self.source_dir /= np.linalg.norm(self.source_dir)

        # Integrator / material parameters.
        self.dt = 0.0005  # s
        self.repulsion = 3000.0  # N/m
        self.damping = 0.02  # per-second velocity damping

        # RNG for deterministic source placement.
        self.rng = np.random.default_rng(int(seed))

        # Initial state from the scene.
        init = scene.initial_state()
        self.position = np.asarray(init.position, dtype=np.float64).copy()
        self.velocity = np.asarray(init.velocity, dtype=np.float64).copy()
        self.source_region = np.asarray(init.attributes["source_region"], dtype=np.float64).copy()
        self.time = float(init.time)

        self.cumulative_mass = 0.0
        self.volume_to_emit = 0.0

        # Diagnostic grid.
        self.bounds_min = np.array(self.config["domain"]["bounds_min"], dtype=np.float64)
        self.bounds_max = np.array(self.config["domain"]["bounds_max"], dtype=np.float64)
        self.grid = np.array(self.config["domain"]["reference_grid"], dtype=np.int64)
        self.dx_grid = (self.bounds_max - self.bounds_min) / self.grid
        axes = [
            np.linspace(
                self.bounds_min[d] + self.dx_grid[d] / 2.0,
                self.bounds_max[d] - self.dx_grid[d] / 2.0,
                self.grid[d],
            )
            for d in range(3)
        ]
        XX, YY, ZZ = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        self.grid_points = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

        # Glass boundary samples and normals.
        self.boundary_pts, self.boundary_norms = self._build_glass_boundary()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _glass_profile(self) -> list[tuple[float, float]]:
        objects = self.config.get("scene_geometry", {}).get("objects", [])
        for obj in objects:
            if obj.get("id") == "clear_cylindrical_glass":
                return list(map(tuple, obj["geometry"]["inner_profile"]))
        # Fallback to a simple cylinder.
        return [(0.0, 0.0), (float(self.ic["glass_inner_radius"]), 0.0),
                (float(self.ic["glass_inner_radius"]), float(self.ic["glass_inner_height"]))]

    def _glass_radius_at_z(self, z: float) -> float:
        profile = self._glass_profile()
        if z <= profile[0][1]:
            return profile[0][0]
        if z >= profile[-1][1]:
            return profile[-1][0]
        for i in range(len(profile) - 1):
            r0, z0 = profile[i]
            r1, z1 = profile[i + 1]
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                return r0 + t * (r1 - r0)
        return profile[-1][0]

    def _build_glass_boundary(self) -> tuple[np.ndarray, np.ndarray]:
        profile = self.ic["glass_inner_profile"]
        theta_count = 32
        theta = np.linspace(0.0, 2.0 * math.pi, theta_count, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        pts: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        # Bottom disk (z = 0, r in [0, r_start]).
        r_start = profile[0][0]
        for r in np.linspace(0.0, r_start - 1e-4, 20):
            for c, s in zip(cos_t, sin_t):
                pts.append(np.array([r * c, r * s, 0.0]))
                nrm.append(np.array([0.0, 0.0, -1.0]))

        # Revolve the inner profile.
        per_segment = 30
        for i in range(len(profile) - 1):
            r0, z0 = profile[i]
            r1, z1 = profile[i + 1]
            dr = r1 - r0
            dz = z1 - z0
            seg_len = math.hypot(dr, dz)
            if seg_len < 1e-9:
                continue
            # Outward (into glass) 2D normal.
            nr = dz / seg_len
            nz = -dr / seg_len
            for t in np.linspace(0.0, 1.0, per_segment):
                r = r0 + t * dr
                z = z0 + t * dz
                for c, s in zip(cos_t, sin_t):
                    pts.append(np.array([r * c, r * s, z]))
                    nrm.append(np.array([nr * c, nr * s, nz]))

        pts = np.asarray(pts, dtype=np.float64)
        nrm = np.asarray(nrm, dtype=np.float64)
        norms = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(norms, 1e-12)
        return pts, nrm

    # ------------------------------------------------------------------
    # Source / inflow
    # ------------------------------------------------------------------
    def _source_basis(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.source_dir
        if abs(d[2]) < 0.99:
            side = np.array([0.0, 0.0, 1.0])
        else:
            side = np.array([0.0, 1.0, 0.0])
        u = np.cross(d, side)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        return u, v

    def _emit(self, dt: float) -> None:
        if self.time < self.source_start or self.time >= self.source_end:
            return
        self.volume_to_emit += self.inflow_rate * dt
        particles_per_dt = self.volume_to_emit / (self.h ** 3)
        to_emit = int(particles_per_dt)
        if to_emit == 0:
            return
        self.volume_to_emit -= to_emit * (self.h ** 3)

        u, v = self._source_basis()
        radius = self.jet_radius
        for _ in range(to_emit):
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            rad = radius * math.sqrt(self.rng.uniform(0.0, 1.0))
            offset = rad * (math.cos(theta) * u + math.sin(theta) * v)
            pos = self.source_center + offset
            vel = self.jet_speed * self.source_dir
            self.position = np.vstack([self.position, pos[None, :]])
            self.velocity = np.vstack([self.velocity, vel[None, :]])
            self.source_region = np.concatenate(
                [self.source_region, np.ones(1, dtype=np.float64)]
            )
            self.cumulative_mass += self.mass

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pairwise_acceleration(self, acc: np.ndarray) -> None:
        n = self.position.shape[0]
        if n == 0:
            return
        tree = cKDTree(self.position)
        pairs = tree.sparse_distance_matrix(tree, 2.0 * self.h, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        d = pairs[:, 2]
        d = np.maximum(d, 1e-12)
        xi = self.position[i]
        xj = self.position[j]
        diff = xi - xj
        # Repulsive force on i: k*(2h - d)*(xi - xj)/d
        f = self.repulsion * (2.0 * self.h - d)[:, None] * (diff / d[:, None])
        np.add.at(acc, i, f / self.mass)
        np.add.at(acc, j, -f / self.mass)

    def _apply_boundary_projection(self) -> None:
        # Floor / bottom.
        below = self.position[:, 2] < 0.0
        if np.any(below):
            self.position[below, 2] = 0.0
            self.velocity[below, 2] = np.maximum(self.velocity[below, 2], 0.0)

        # Glass wall.
        xy = self.position[:, :2]
        r = np.linalg.norm(xy, axis=1)
        z = self.position[:, 2]
        mask = (z >= 0.0) & (z <= 0.78) & (r > 0.0)
        if not np.any(mask):
            return
        rad_at_z = np.array([self._glass_radius_at_z(zz) for zz in z[mask]], dtype=np.float64)
        rad_at_z -= 1e-4
        r_masked = r[mask]
        penetrating = r_masked > rad_at_z
        if not np.any(penetrating):
            return
        idx = np.nonzero(mask)[0][penetrating]
        # Project radially inward.
        ratio = rad_at_z[penetrating] / r_masked[penetrating]
        self.position[idx, 0] *= ratio
        self.position[idx, 1] *= ratio
        # Kill outward radial velocity.
        v_xy = self.velocity[idx, :2]
        pos_xy = self.position[idx, :2]
        r2 = np.maximum(np.sum(pos_xy ** 2, axis=1), 1e-18)
        v_r = np.sum(v_xy * pos_xy, axis=1) / r2
        outward = v_r > 0.0
        if np.any(outward):
            factor = np.ones_like(v_r)
            factor[outward] = 0.0
            self.velocity[idx[outward], :2] -= (v_r[outward] * pos_xy[outward].T / r2[outward]).T

    def _step(self, dt: float) -> None:
        acc = np.zeros_like(self.position)
        acc += self.gravity
        self._pairwise_acceleration(acc)

        self.velocity += dt * acc
        # Damping.
        self.velocity *= math.exp(-self.damping * dt)

        self.position += dt * self.velocity
        self._apply_boundary_projection()

    # ------------------------------------------------------------------
    # Diagnostic grid fields
    # ------------------------------------------------------------------
    def _diagnostic_fields(self) -> dict[str, np.ndarray]:
        n = self.position.shape[0]
        nx, ny, nz = self.grid
        shape3 = (nx, ny, nz)

        if n == 0:
            level = np.full(shape3, 1.0, dtype=np.float64)
            active = np.zeros(shape3, dtype=np.float64)
            vel_grid = np.zeros(shape3 + (3,), dtype=np.float64)
            return {
                "liquid_level_set": level,
                "active_mask": active,
                "velocity_field": vel_grid,
            }

        tree = cKDTree(self.position)
        k = min(32, n)
        dists, idx = tree.query(self.grid_points, k=k, distance_upper_bound=2.0 * self.h)

        # Level set from union of particle spheres.
        nearest_dist = dists[:, 0]
        particle_radius = 0.55 * self.h
        level = nearest_dist - particle_radius
        active = (level < 0.0).astype(np.float64)

        # Velocity field by SPH-like interpolation.
        vel_grid = np.zeros((self.grid_points.shape[0], 3), dtype=np.float64)
        weights = np.zeros(self.grid_points.shape[0], dtype=np.float64)
        # Cubic spline kernel (unnormalized) for weighting.
        # W(q) with q = r/h, support 2h.
        q = dists / self.h
        # f(q) piecewise:
        #   0<=q<1: 4 - 6q^2 + 3q^3
        #   1<=q<2: (2-q)^3
        w = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        w[m1] = 4.0 - 6.0 * q[m1] ** 2 + 3.0 * q[m1] ** 3
        w[m2] = (2.0 - q[m2]) ** 3
        w[dists >= 2.0 * self.h] = 0.0
        w[dists < 0.0] = 0.0

        for kk in range(k):
            valid = (dists[:, kk] < 2.0 * self.h) & (idx[:, kk] < n)
            if not np.any(valid):
                continue
            ii = idx[valid, kk]
            ww = w[valid, kk]
            vel_grid[valid] += ww[:, None] * self.velocity[ii]
            weights[valid] += ww

        safe = weights > 0.0
        vel_grid[safe] /= weights[safe, None]

        level = level.reshape(shape3)
        active = active.reshape(shape3)
        vel_grid = vel_grid.reshape(shape3 + (3,))

        return {
            "liquid_level_set": level,
            "active_mask": active,
            "velocity_field": vel_grid,
        }

    def _build_frame(self) -> StateFrame:
        fields = self._diagnostic_fields()
        fields["boundary_normal"] = self.boundary_norms.copy()
        fields["boundary_target_velocity"] = np.zeros_like(self.boundary_norms)
        fields["boundary_velocity"] = np.zeros_like(self.boundary_norms)
        fields["cumulative_mass_source"] = np.array([self.cumulative_mass], dtype=np.float64)

        scalars = {
            "characteristic_length": float(self.ic["glass_inner_radius"]),
            "grid_spacing": float(self.dx_grid[0]),
            "reference_velocity": float(self.ic["jet_speed"]),
        }

        attributes = {
            "mass_quantity": np.full(self.position.shape[0], self.mass, dtype=np.float64),
            "source_region": self.source_region.copy(),
        }

        return StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=np.full(self.position.shape[0], self.mass, dtype=np.float64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            return self._build_frame()
        while self.time < time:
            dt = min(self.dt, time - self.time)
            self._emit(dt)
            self._step(dt)
            self.time += dt
        return self._build_frame()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "particle_count": int(self.position.shape[0]),
            "cumulative_mass_source": float(self.cumulative_mass),
        }
