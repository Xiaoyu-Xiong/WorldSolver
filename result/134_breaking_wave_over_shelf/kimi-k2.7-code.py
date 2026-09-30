"""SPH solver for a dam-break / breaking wave over a submerged shelf.

Implements a weakly-compressible Smoothed Particle Hydrodynamics (WCSPH) scheme
using the standard Monaghan cubic-spline kernel, artificial viscosity, and
hard boundary projection for static tank walls and the submerged shelf.  The
kinematic gate is removed at the start of the simulation so the water column
collapses and generates a breaking bore over the shelf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver contract implementation for the breaking-wave scene."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        # Config (scene exposes its config in the scaffold).
        self.config: dict = getattr(scene, "config", {})
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), float)

        # Particle data from the deterministic initial state.
        frame0 = scene.initial_state()
        self.pos: np.ndarray = np.asarray(frame0.position, dtype=float).copy()
        self.vel: np.ndarray = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass: np.ndarray = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id: np.ndarray = (
            np.asarray(frame0.body_id, dtype=np.int32).copy()
            if frame0.body_id is not None
            else np.zeros(len(self.pos), dtype=np.int32)
        )
        self.n_particles = len(self.pos)

        # Particle spacing / radius used by the scene.
        self.spacing: float = float(frame0.scalars.get("grid_spacing", 0.14))
        self.particle_radius: float = 0.45 * self.spacing

        # SPH parameters.
        self.h: float = 1.5 * self.spacing  # smoothing length
        self.kernel_norm: float = 1.0 / (np.pi * self.h**3)
        self.W0: float = self.kernel_norm
        self.rho0: float = 1000.0  # will be overwritten by measured rest density
        self.cs: float = 40.0
        self.gamma: float = 7.0
        self.alpha: float = 0.05
        self.beta: float = 0.0
        self.eta2: float = (0.1 * self.h) ** 2

        # Time stepping.
        self.dt_fixed: float = 5.0e-4
        self.time: float = 0.0
        self.step_count: int = 0

        # Diagnostic grid.
        self.field_shape: tuple[int, int, int] = tuple(
            self.config.get("field_grid_shape", [24, 16, 28])
        )
        # Whole-tank covering grid with uniform spacing.
        self.grid_spacing: float = max(
            6.4 / self.field_shape[0],
            2.8 / self.field_shape[1],
            1.2 / self.field_shape[2],
        )
        self.grid_origin: np.ndarray = np.array([-3.2, -1.4, -0.1], float)

        # Scalars.
        self.characteristic_length: float = float(
            frame0.scalars.get(
                "characteristic_length",
                max(float(np.ptp(self.pos, axis=0).max()), self.spacing),
            )
        )
        self.reference_velocity: float = float(frame0.scalars.get("reference_velocity", 1.0))

        # Boundaries (boxes: center, half_extents).  Skip the kinematic gate.
        self.boxes: list[tuple[np.ndarray, np.ndarray]] = []
        self._build_boundaries()

        # Boundary samples (static walls).
        self.boundary_sample_step: float = 0.25
        self._build_boundary_samples()

        # Reference density from the hydrostatic initial configuration.
        self.rho0 = float(np.mean(self._compute_density(self.pos)))
        if self.rho0 <= 0.0:
            self.rho0 = 1000.0
        self.B: float = (self.cs**2 * self.rho0) / self.gamma

        self._initialized = True

    # ---------------------------------------------------------------- helpers
    def _build_boundaries(self) -> None:
        """Parse scene_geometry for static boundary boxes; skip the gate."""
        geometry = self.config.get("scene_geometry", {})
        objects = geometry.get("objects", [])
        for obj in objects:
            if obj.get("semantic") != "boundary":
                continue
            if obj.get("id") == "dam_gate":
                continue
            translation = np.array(obj["transform"]["translation"], float)
            half_extents = np.array(obj["geometry"]["half_extents"], float)
            self.boxes.append((translation, half_extents))

        if not self.boxes:
            # Fallback hard-coded from the frozen scene geometry.
            self.boxes = [
                (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05])),  # ground
                (np.array([0.0, -0.98, 0.55]), np.array([3.2, 0.06, 0.55])),  # left
                (np.array([0.0, 0.98, 0.55]), np.array([3.2, 0.06, 0.55])),  # right
                (np.array([3.12, 0.0, 0.55]), np.array([0.06, 0.98, 0.55])),  # end
                (np.array([1.45, 0.0, 0.30]), np.array([1.2, 0.84, 0.30])),  # shelf
            ]

    def _inside_any_box(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask for points inside any of the boxes."""
        mask = np.zeros(points.shape[0], dtype=bool)
        for center, half in self.boxes:
            dp = points - center
            inside = np.all(np.abs(dp) < half, axis=1)
            mask |= inside
        return mask

    def _build_boundary_samples(self) -> None:
        """Create point samples on static boundary surfaces."""
        pts: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        for center, half in self.boxes:
            for axis in range(3):
                other = [0, 1, 2]
                other.remove(axis)
                a1, a2 = other
                # Sample the two faces perpendicular to axis.
                for sign in (-1.0, 1.0):
                    face_c = center.copy()
                    face_c[axis] += sign * half[axis]
                    # Build a grid on the face.
                    n1 = max(1, int(np.round(2.0 * half[a1] / self.boundary_sample_step)))
                    n2 = max(1, int(np.round(2.0 * half[a2] / self.boundary_sample_step)))
                    c1 = np.linspace(center[a1] - half[a1], center[a1] + half[a1], n1)
                    c2 = np.linspace(center[a2] - half[a2], center[a2] + half[a2], n2)
                    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
                    g1 = g1.ravel()
                    g2 = g2.ravel()
                    for _ in range(2):  # two faces
                        pass
                    # Fill coordinates.
                    npts = g1.size
                    face_pts = np.zeros((npts, 3))
                    face_pts[:, a1] = g1
                    face_pts[:, a2] = g2
                    face_pts[:, axis] = face_c[axis]
                    # Normal.
                    nvec = np.zeros((npts, 3))
                    nvec[:, axis] = sign
                    pts.append(face_pts)
                    normals.append(nvec)
        if pts:
            self.boundary_positions: np.ndarray = np.concatenate(pts, axis=0)
            self.boundary_normals: np.ndarray = np.concatenate(normals, axis=0)
        else:
            self.boundary_positions = np.zeros((0, 3))
            self.boundary_normals = np.zeros((0, 3))

    # --------------------------------------------------------------- SPH core
    def _W(self, q: np.ndarray) -> np.ndarray:
        W = np.zeros_like(q)
        m1 = q <= 1.0
        m2 = (q > 1.0) & (q <= 2.0)
        W[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        W[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return W * self.kernel_norm

    def _grad_W(self, q: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        fp = np.zeros_like(q)
        m1 = q <= 1.0
        m2 = (q > 1.0) & (q <= 2.0)
        fp[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        fp[m2] = -0.75 * (2.0 - q[m2]) ** 2
        dW_dr = self.kernel_norm / self.h * fp
        r = np.linalg.norm(rvec, axis=1)
        r_safe = np.maximum(r, 1e-12)
        return (dW_dr / r_safe)[:, None] * rvec

    def _compute_density(self, pos: np.ndarray) -> np.ndarray:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=2.0 * self.h, output_type="ndarray")
        rho = self.mass * self.W0
        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rvec = pos[i] - pos[j]
            r = np.linalg.norm(rvec, axis=1)
            q = r / self.h
            W = self._W(q)
            contrib_i = self.mass[j] * W
            contrib_j = self.mass[i] * W
            np.add.at(rho, i, contrib_i)
            np.add.at(rho, j, contrib_j)
        return rho

    def _compute_acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=2.0 * self.h, output_type="ndarray")

        # Density.
        rho = self.mass * self.W0
        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rvec = pos[i] - pos[j]
            r = np.linalg.norm(rvec, axis=1)
            q = r / self.h
            W = self._W(q)
            np.add.at(rho, i, self.mass[j] * W)
            np.add.at(rho, j, self.mass[i] * W)

        # Pressure (clamped to avoid tensile instability).
        pressure = self.B * ((rho / self.rho0) ** self.gamma - 1.0)
        pressure = np.clip(pressure, 0.0, None)

        acc = np.zeros_like(pos)
        acc += self.gravity

        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rvec = pos[i] - pos[j]
            r = np.linalg.norm(rvec, axis=1)
            q = r / self.h
            grad_W = self._grad_W(q, rvec)

            v_ij = vel[i] - vel[j]
            rdotv = np.sum(v_ij * rvec, axis=1)
            rv2 = r ** 2 + self.eta2
            mu = self.h * rdotv / rv2
            rho_ij = 0.5 * (rho[i] + rho[j])
            pi = np.zeros_like(mu)
            mask = rdotv < 0.0
            pi[mask] = (-self.alpha * self.cs * mu[mask] + self.beta * mu[mask] ** 2) / rho_ij[mask]

            term = (pressure[i] / rho[i] ** 2 + pressure[j] / rho[j] ** 2 + pi)[:, None]
            # Force on particle i; equal and opposite on j.
            contrib = -self.mass[j][:, None] * term * grad_W
            np.add.at(acc, i, contrib)
            np.add.at(acc, j, -contrib)

        return acc

    def _apply_boundaries(self) -> None:
        """Hard projection of particles inside static boxes."""
        for center, half in self.boxes:
            dp = self.pos - center
            inside = np.all(np.abs(dp) < half, axis=1)
            if not np.any(inside):
                continue
            idx = np.where(inside)[0]
            dp_in = dp[idx]
            h_local = half
            dists = h_local - np.abs(dp_in)
            axis = np.argmin(dists, axis=1)
            n = np.zeros_like(dp_in)
            rows = np.arange(len(idx))
            signs = np.sign(dp_in[rows, axis])
            n[rows, axis] = signs
            depth = dists[rows, axis]
            # Push out with a tiny margin.
            self.pos[idx] += n * (depth[:, None] + 1.0e-6)
            # Remove inward normal velocity.
            vdot = np.sum(self.vel[idx] * n, axis=1)
            mask = vdot < 0.0
            self.vel[idx[mask]] -= (vdot[mask][:, None]) * n[mask]

    # -------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        acc = self._compute_acceleration(self.pos, self.vel)
        self.vel += dt * acc
        self.pos += dt * self.vel
        self._apply_boundaries()
        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver not initialized")
        if time < self.time:
            raise ValueError("Requested time is before current simulation time")
        while self.time < time:
            dt = min(self.dt_fixed, time - self.time)
            self._step(dt)
        return self._build_frame(time)

    # ---------------------------------------------------------------- diagnostics
    def _build_frame(self, time: float) -> StateFrame:
        active, ls, vel_field = self._compute_grid_fields()
        boundary_normal, boundary_tv, boundary_v = self._boundary_field_arrays()
        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.n_particles, self.particle_radius),
                "material_label": self.body_id.copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], self.n_particles, axis=0),
            },
            fields={
                "active_mask": active,
                "liquid_level_set": ls,
                "velocity_field": vel_field,
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_tv,
                "boundary_velocity": boundary_v,
                "cumulative_mass_source": np.array([0.0]),
            },
            scalars={
                "grid_spacing": self.grid_spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
            },
        )

    def _compute_grid_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.field_shape
        # Grid node coordinates.
        ix = np.arange(nx, dtype=float)
        iy = np.arange(ny, dtype=float)
        iz = np.arange(nz, dtype=float)
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        grid_pts = np.column_stack(
            (
                self.grid_origin[0] + gx.ravel() * self.grid_spacing,
                self.grid_origin[1] + gy.ravel() * self.grid_spacing,
                self.grid_origin[2] + gz.ravel() * self.grid_spacing,
            )
        )

        tree = cKDTree(self.pos)
        # Signed distance level set (negative inside liquid).
        dists, _ = tree.query(grid_pts, k=1)
        offset = 0.5 * self.spacing
        ls = dists - offset
        ls = ls.reshape((nx, ny, nz))
        active = (ls <= 0.0).astype(bool)

        # Velocity field via SPH interpolation.
        vel_field = np.zeros((nx, ny, nz, 3), dtype=float)
        ptree = cKDTree(self.pos)
        gtree = cKDTree(grid_pts)
        pairs = ptree.sparse_distance_matrix(gtree, 2.0 * self.h, output_type="ndarray")
        if pairs.size:
            pi = pairs["i"]
            gj = pairs["j"]
            d = pairs["v"]
            q = d / self.h
            W = self._W(q)
            # weighted velocity numerator and denominator
            numerator = np.zeros((grid_pts.shape[0], 3), dtype=float)
            denominator = np.zeros(grid_pts.shape[0], dtype=float)
            weight = self.mass[pi] / self._compute_density_for_interp(pi) * W
            np.add.at(numerator, gj, weight[:, None] * self.vel[pi])
            np.add.at(denominator, gj, weight)
            valid = denominator > 0.0
            numerator[valid] /= denominator[valid, None]
            vel_field = numerator.reshape((nx, ny, nz, 3))

        return active, ls, vel_field

    def _compute_density_for_interp(self, indices: np.ndarray) -> np.ndarray:
        # Use a simple estimate: each particle's SPH density from the current step
        # is not stored globally.  Fall back to the reference density; this only
        # scales the interpolated velocity, and the mass/density ratio is constant
        # for equal-mass particles.
        return np.full(indices.shape[0], self.rho0, dtype=float)

    def _boundary_field_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.boundary_normals.shape[0]
        target = np.zeros_like(self.boundary_normals)
        velocity = np.zeros_like(self.boundary_normals)
        return self.boundary_normals.copy(), target, velocity

    def diagnostics(self) -> dict[str, Any]:
        return {
            "solver": "wcsph",
            "steps": self.step_count,
            "time": self.time,
            "dt": self.dt_fixed,
            "n_particles": self.n_particles,
        }
