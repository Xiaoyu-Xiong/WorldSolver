"""Solver Agent-owned implementation for the buoyant cube drop scene.

A self-contained weakly-compressible SPH (WCSPH) fluid is coupled to a rigid
cube via pressure/buoyancy and penalty contact.  Gravity and tank boundaries
produce a falling cube, splash, wake, and subsequent buoyant motion.  No
hidden data, no precomputed trajectories, and no ``genesis``/``gs`` imports
are used.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

        self._scene = scene
        self._config = scene.config
        self._initial = scene.initial_state()

        cfg = self._config
        self._gravity = np.asarray(cfg["gravity"], dtype=float)
        self._duration = float(cfg["duration"])
        self._sample_dt = float(cfg["sample_dt"])

        # State arrays.
        self._current_time = 0.0
        self._position = np.array(self._initial.position, dtype=float)
        self._velocity = np.array(self._initial.velocity, dtype=float)
        self._orientation = np.array(self._initial.orientation, dtype=float)
        self._angular_velocity = np.array(self._initial.angular_velocity, dtype=float)
        self._mass = np.array(self._initial.mass, dtype=float)
        self._body_id = np.array(self._initial.body_id, dtype=np.int32)
        self._phase = np.array(self._initial.attributes["phase"], dtype=float)

        self._cube_idx = int(np.nonzero(self._body_id == 2)[0][0])
        self._fluid_mask = self._body_id == 1
        self._fluid_idx = np.where(self._fluid_mask)[0]
        self._n = len(self._position)

        # Fluid resolution and derived quantities.
        self._dp = 0.12
        self._fluid_volume = self._dp ** 3
        self._water_rest_density = float(cfg["initial_conditions"]["water"]["rest_density"])
        self._cube_density = float(cfg["initial_conditions"]["cube"]["density"])

        cube_side = float(cfg["initial_conditions"]["cube"]["side_length"])
        self._cube_side = cube_side
        self._cube_half = cube_side / 2.0
        self._cube_mass = self._cube_density * (cube_side ** 3)
        self._cube_inv_mass = 1.0 / self._cube_mass
        self._cube_inertia = (self._cube_mass / 6.0) * (cube_side ** 2) * np.eye(3)
        self._cube_inv_inertia = np.linalg.inv(self._cube_inertia)

        # SPH kernel.
        self._h = 1.5 * self._dp
        self._h2 = self._h * self._h

        # Sub-stepping.
        self._substeps = 8
        self._dt = self._sample_dt / float(self._substeps)

        # Tank domain bounds for boundary response.
        self._lo = np.array([-1.5, -1.2, 0.0])
        self._hi = np.array([1.5, 1.2, 2.5])

        # Cached rest-mass for diagnostics / consistency.
        self._fluid_rest_mass = self._water_rest_density * self._fluid_volume

    # ------------------------------------------------------------------ SPH kernel
    def _kernel(self, r: np.ndarray) -> float:
        q = float(np.linalg.norm(r) / self._h)
        if q >= 2.0:
            return 0.0
        sigma = 1.0 / (np.pi * self._h ** 3)
        if q < 1.0:
            return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        return sigma * 0.25 * (2.0 - q) ** 3

    def _grad_kernel(self, r: np.ndarray) -> np.ndarray:
        rr = float(np.linalg.norm(r))
        q = rr / self._h
        if q >= 2.0 or rr < 1e-12:
            return np.zeros(3)
        sigma = 1.0 / (np.pi * self._h ** 3)
        if q < 1.0:
            fac = sigma * (-3.0 * q + 2.25 * q * q) / self._h
        else:
            fac = sigma * -0.75 * (2.0 - q) ** 2 / self._h
        return fac * (r / rr)

    def _rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        return Rotation.from_quat(quat).as_matrix()

    # ------------------------------------------------------------------ spatial hash
    def _build_neighbors(self, positions: np.ndarray) -> dict:
        """Return a dict mapping particle index to neighbor indices."""
        n = len(positions)
        cell_size = 2.0 * self._h
        inv = 1.0 / cell_size
        cells = {}
        for i in range(n):
            key = tuple(np.floor(positions[i] * inv).astype(int))
            cells.setdefault(key, []).append(i)

        neighbors = {}
        for i in range(n):
            key = tuple(np.floor(positions[i] * inv).astype(int))
            nbrs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cell = (key[0] + dx, key[1] + dy, key[2] + dz)
                        nbrs.extend(cells.get(cell, []))
            neighbors[i] = nbrs
        return neighbors

    # ------------------------------------------------------------------ fluid forces
    def _fluid_forces(self, positions: np.ndarray, velocities: np.ndarray):
        fidx = self._fluid_idx
        n = self._n
        density = np.ones(n, dtype=float) * self._water_rest_density
        pressure = np.zeros(n, dtype=float)

        neighbors = self._build_neighbors(positions)

        # Density / pressure only for fluid particles.
        for i in fidx:
            rho = self._kernel(np.zeros(3))
            for j in neighbors[i]:
                if not self._fluid_mask[j]:
                    continue
                if i == j:
                    continue
                r = positions[j] - positions[i]
                rho += self._kernel(r)
            rho = max(rho, 1e-6)
            density[i] = rho
            # Tait equation (c0 ~ 50 m/s gives stable, weakly compressible water).
            c0 = 50.0
            rho0 = self._water_rest_density
            pressure[i] = (rho0 * c0 * c0 / 7.0) * ((rho / rho0) ** 7 - 1.0)

        force = np.zeros((n, 3), dtype=float)

        # Pressure + gravity.
        for i in fidx:
            force[i] += self._mass[i] * self._gravity
            for j in neighbors[i]:
                if not self._fluid_mask[j] or i == j:
                    continue
                r = positions[j] - positions[i]
                rho_i = density[i]
                rho_j = density[j]
                if rho_i <= 0 or rho_j <= 0:
                    continue
                grad = self._grad_kernel(r)
                acc = -self._mass[j] * (
                    pressure[i] / (rho_i * rho_i) + pressure[j] / (rho_j * rho_j)
                ) * grad
                force[i] += self._mass[i] * acc

        # Artificial viscosity.
        nu = 0.02
        for i in fidx:
            for j in neighbors[i]:
                if not self._fluid_mask[j] or i == j:
                    continue
                r = positions[j] - positions[i]
                dv = velocities[j] - velocities[i]
                lap = self._kernel(r)
                force[i] += nu * self._mass[j] * dv * lap

        return force, density, pressure

    # ------------------------------------------------------------------ cube helpers
    def _point_cube_signed_dist(self, p: np.ndarray, c: np.ndarray, R: np.ndarray) -> float:
        local = R.T @ (p - c)
        d = np.abs(local) - self._cube_half
        out = np.linalg.norm(np.maximum(d, 0.0))
        inside = np.max(d)
        return out + min(inside, 0.0)

    def _cube_interaction_forces(self, positions: np.ndarray, pressure: np.ndarray):
        cube_pos = positions[self._cube_idx]
        cube_vel = self._velocity[self._cube_idx]
        cube_ang = self._angular_velocity[self._cube_idx]
        R = self._rotation_matrix(self._orientation[self._cube_idx])

        force = np.zeros((self._n, 3), dtype=float)
        torque = np.zeros(3, dtype=float)

        k_stiff = 1e5
        damping = 1.0

        for i in self._fluid_idx:
            local = R.T @ (positions[i] - cube_pos)
            inside = np.all(np.abs(local) <= self._cube_half)
            if not inside:
                continue
            # Find smallest penetration axis (largest remaining distance).
            d = self._cube_half - np.abs(local)
            axis = int(np.argmax(d))
            nrm = np.zeros(3)
            nrm[axis] = np.sign(local[axis]) if local[axis] != 0 else 1.0
            dist = -float(d[axis])

            r_world = R @ local
            vrel = self._velocity[i] - cube_vel - np.cross(cube_ang, r_world)
            fpen = k_stiff * dist * nrm - damping * vrel
            # Add buoyancy/pressure contribution.
            fpen += -pressure[i] * self._fluid_volume * nrm

            force[i] += fpen
            force[self._cube_idx] -= fpen
            rpos = positions[i] - cube_pos
            torque += np.cross(rpos, -fpen)
        return force, torque

    # ------------------------------------------------------------------ boundary response
    def _boundary_force(self, positions: np.ndarray, velocities: np.ndarray, idx: int) -> np.ndarray:
        p = positions[idx]
        v = velocities[idx]
        f = np.zeros(3, dtype=float)
        k = 1e5
        dmp = 2.0
        lo = self._lo
        hi = self._hi
        if p[2] < lo[2]:
            f[2] += k * (lo[2] - p[2]) - dmp * v[2]
        for dim in (0, 1):
            if p[dim] < lo[dim]:
                f[dim] += k * (lo[dim] - p[dim]) - dmp * v[dim]
            if p[dim] > hi[dim]:
                f[dim] += k * (p[dim] - hi[dim]) - dmp * v[dim]
        return f

    # ------------------------------------------------------------------ integration
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=float)

    def _step(self, dt: float) -> None:
        positions = self._position
        velocities = self._velocity

        # Fluid SPH forces.
        ffluid, density, pressure = self._fluid_forces(positions, velocities)

        # Cube-fluid interaction.
        fcube, cube_torque = self._cube_interaction_forces(positions, pressure)

        # External (gravity + boundaries).
        external = np.zeros((self._n, 3), dtype=float)
        for i in range(self._n):
            external[i] += self._mass[i] * self._gravity
            external[i] += self._boundary_force(positions, velocities, i)

        total_force = ffluid + fcube + external

        # Update fluid particles.
        for i in self._fluid_idx:
            a = total_force[i] / max(self._mass[i], 1e-12)
            velocities[i] += a * dt
            positions[i] += velocities[i] * dt

        # Update cube rigid body.
        cube_pos = np.array(positions[self._cube_idx])
        cube_vel = np.array(velocities[self._cube_idx])
        cube_acc = total_force[self._cube_idx] * self._cube_inv_mass
        cube_vel += cube_acc * dt
        cube_pos += cube_vel * dt

        # Angular dynamics.
        R = self._rotation_matrix(self._orientation[self._cube_idx])
        torque_body = R.T @ cube_torque
        alpha_body = self._cube_inv_inertia @ torque_body
        alpha = R @ alpha_body
        self._angular_velocity[self._cube_idx] += alpha * dt

        # Orientation quaternion integration.
        w = self._angular_velocity[self._cube_idx]
        q = self._orientation[self._cube_idx]
        omega = np.array([w[0], w[1], w[2], 0.0])
        # dq/dt = 0.5 * omega * q
        q_new = self._quaternion_multiply(omega, q) * 0.5 * dt
        q = q + q_new
        q /= np.linalg.norm(q)
        self._orientation[self._cube_idx] = q

        self._position[self._cube_idx] = cube_pos
        self._velocity[self._cube_idx] = cube_vel

        # Clamp unrealistic velocities.
        np.clip(self._velocity, -50.0, 50.0, out=self._velocity)

    # ------------------------------------------------------------------ public API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self._current_time:
            return self._make_frame(target)
        while self._current_time < target:
            self._step(self._dt)
            self._current_time += self._dt
        return self._make_frame(target)

    def _make_frame(self, t: float) -> StateFrame:
        n = self._n
        # Per-sample mass split by material.
        mass_quantity = np.zeros((n, 2), dtype=float)
        for i in range(n):
            if self._body_id[i] == 1:
                mass_quantity[i, 0] = self._mass[i]
            else:
                mass_quantity[i, 1] = self._mass[i]

        # Cube to fluid signed distances.
        cube_pos = self._position[self._cube_idx]
        R = self._rotation_matrix(self._orientation[self._cube_idx])
        contact_sd = np.array([
            self._point_cube_signed_dist(self._position[i], cube_pos, R)
            for i in self._fluid_idx
        ], dtype=float)

        # External forces (gravity + boundary) for trajectory export.
        external_force = np.zeros((n, 3), dtype=float)
        for i in range(n):
            external_force[i] += self._mass[i] * self._gravity
            external_force[i] += self._boundary_force(self._position, self._velocity, i)

        # Aggregate interaction forces per coupled body (water, cube).
        interaction_force = np.zeros((2, 3), dtype=float)
        # Recompute a lightweight interaction-only estimate.
        ffluid, _, pressure = self._fluid_forces(self._position, self._velocity)
        fcube, _ = self._cube_interaction_forces(self._position, pressure)
        interaction = ffluid + fcube
        interaction_force[0] = np.sum(interaction[self._fluid_idx], axis=0)
        interaction_force[1] = np.sum(interaction[[self._cube_idx]], axis=0)

        frame = StateFrame(
            time=t,
            position=np.array(self._position, dtype=float),
            velocity=np.array(self._velocity, dtype=float),
            orientation=np.array(self._orientation, dtype=float),
            angular_velocity=np.array(self._angular_velocity, dtype=float),
            mass=np.array(self._mass, dtype=float),
            body_id=np.array(self._body_id, dtype=np.int32),
            attributes={
                "phase": np.array(self._phase, dtype=float),
                "mass_quantity": mass_quantity,
            },
            scalars={
                "water_rest_density": self._water_rest_density,
                "cube_density": self._cube_density,
                "contact_length_scale": float(self._dp),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(2, dtype=float),
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "WCSPH-Rigid-Coupled",
            "particles": self._n,
            "internal_dt": self._dt,
        }
