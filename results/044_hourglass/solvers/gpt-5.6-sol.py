"""Deterministic granular hourglass solver using a small DEM integrator."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.pos = np.asarray(initial.position, dtype=float).copy()
        self.vel = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.radius = float(self.config["initial_conditions"]["particle_radius"])
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.slab = float(self.config["boundary_conditions"].get("slab_half_depth", 0.055))
        self.time = 0.0
        self.total_steps = 0
        profile = np.asarray(
            [[0.46, 0.02], [0.44, 0.14], [0.32, 0.36], [0.12, 0.60],
             [0.038, 0.72], [0.12, 0.84], [0.32, 1.08], [0.44, 1.30], [0.46, 1.42]],
            dtype=float,
        )
        self.prof_r, self.prof_z = profile[:, 0], profile[:, 1]
        # Geometry translation is 0.08 m; the shell rotates about its mid-height.
        self.body_translation = np.array([0.0, 0.0, 0.08], dtype=float)
        self.body_pivot = np.array([0.0, 0.0, 0.72], dtype=float)
        self.grain_volume = 4.0 * math.pi * self.radius**3 / 3.0

    @staticmethod
    def _flip_angle(t: float) -> tuple[float, float]:
        def flip(x: float) -> tuple[float, float]:
            if x <= 0.0:
                return 0.0, 0.0
            if x < 0.8:
                u = x / 0.8
                return math.pi * u * u * (3.0 - 2.0 * u), math.pi * 6.0 * u * (1.0 - u) / 0.8
            return math.pi, 0.0
        if t < 6.0:
            return 0.0, 0.0
        if t < 6.8:
            return flip(t - 6.0)
        if t < 12.0:
            return math.pi, 0.0
        if t < 12.8:
            a, w = flip(t - 12.0)
            return math.pi + a, w
        return 2.0 * math.pi, 0.0

    def _profile_radius(self, z):
        return np.interp(z, self.prof_z, self.prof_r, left=self.prof_r[0], right=self.prof_r[-1])

    def _body_to_world(self, local: np.ndarray, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        q = local - self.body_pivot
        out = q.copy()
        out[:, 0] = c * q[:, 0] + s * q[:, 2]
        out[:, 2] = -s * q[:, 0] + c * q[:, 2]
        return out + self.body_pivot + self.body_translation

    def _world_to_body(self, world: np.ndarray, angle: float) -> np.ndarray:
        q = world - self.body_translation - self.body_pivot
        c, s = math.cos(angle), math.sin(angle)
        out = q.copy()
        out[:, 0] = c * q[:, 0] - s * q[:, 2]
        out[:, 2] = s * q[:, 0] + c * q[:, 2]
        return out + self.body_pivot

    def _integrate_step(self, dt: float, tmid: float) -> None:
        n = len(self.pos)
        angle, _ = self._flip_angle(tmid)
        force = self.mass[:, None] * self.gravity[None, :]

        # Uniform-grid broad phase and soft-sphere normal contact.
        diameter = 2.0 * self.radius
        cell_size = 2.2 * self.radius
        keys = np.floor(self.pos[:, (0, 2)] / cell_size).astype(int)
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (ix, iz) in enumerate(keys):
            grid.setdefault((int(ix), int(iz)), []).append(i)
        for i in range(n):
            ix, iz = keys[i]
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((int(ix + dx), int(iz + dz)), ()):
                        if j <= i:
                            continue
                        dvec = self.pos[j] - self.pos[i]
                        dist2 = float(np.dot(dvec, dvec))
                        if dist2 >= diameter * diameter:
                            continue
                        dist = math.sqrt(max(dist2, 1e-14))
                        normal = dvec / dist
                        reln = float(np.dot(self.vel[j] - self.vel[i], normal))
                        mag = 32.0 * (diameter - dist) - 0.22 * reln
                        if mag > 0.0:
                            f = mag * normal
                            force[i] -= f
                            force[j] += f

        # Semi-implicit Euler and weak granular bulk damping.
        self.vel += (force / self.mass[:, None]) * dt
        self.vel *= math.exp(-0.32 * dt)
        self.pos += self.vel * dt

        # Project escaped grains onto the moving glass interior in body coordinates.
        local = self._world_to_body(self.pos, angle)
        zmin, zmax = self.prof_z[0] + self.radius, self.prof_z[-1] - self.radius
        for i in range(n):
            if local[i, 2] < zmin:
                local[i, 2] = zmin
                self.vel[i] *= 0.42
            elif local[i, 2] > zmax:
                local[i, 2] = zmax
                self.vel[i] *= 0.42
            rlim = float(self._profile_radius(local[i, 2])) - self.radius
            if abs(local[i, 0]) > rlim:
                local[i, 0] = math.copysign(max(rlim, 0.001), local[i, 0])
                self.vel[i] *= 0.55
            ymin, ymax = -self.slab + self.radius, self.slab - self.radius
            if local[i, 1] < ymin:
                local[i, 1] = ymin
                self.vel[i, 1] = abs(self.vel[i, 1]) * 0.15
            elif local[i, 1] > ymax:
                local[i, 1] = ymax
                self.vel[i, 1] = -abs(self.vel[i, 1]) * 0.15
        self.pos = self._body_to_world(local, angle)
        self.total_steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1e-9:
            raise ValueError("advance_to requires finite nondecreasing time")
        while self.time < target - 1e-12:
            dt = min(0.02, target - self.time)
            self._integrate_step(dt, self.time + 0.5 * dt)
            self.time += dt
        angle, omega = self._flip_angle(target)
        quaternion = np.tile(
            np.array([0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]),
            (len(self.pos), 1),
        )
        angular_velocity = np.tile(np.array([0.0, omega, 0.0]), (len(self.pos), 1))
        local = self._world_to_body(self.pos, angle)
        radial_gap = self._profile_radius(local[:, 2]) - np.abs(local[:, 0]) - self.radius
        end_gap = np.minimum(
            local[:, 2] - self.prof_z[0] - self.radius,
            self.prof_z[-1] - local[:, 2] - self.radius,
        )
        gap = np.minimum(radial_gap, end_gap)
        frame = StateFrame(
            time=target,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=quaternion,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes={
                "particle_radius": np.full(len(self.pos), self.radius),
                "mass_quantity": self.mass.copy(),
                "volume_quantity": np.full(len(self.pos), self.grain_volume),
            },
            fields={
                "contact_signed_distance": gap.astype(float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={
                "contact_length_scale": 2.0 * self.radius,
                "initial_bulk_particle_volume": len(self.pos) * self.grain_volume,
                "hourglass_body_rotation_angle": angle,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, Any]:
        return {
            "integrator": "soft_sphere_dem_semi_implicit_euler",
            "particle_count": int(len(self.pos)),
            "integration_steps": int(self.total_steps),
            "mass_total": float(np.sum(self.mass)),
        }
