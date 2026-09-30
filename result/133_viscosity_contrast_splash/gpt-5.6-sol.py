"""Deterministic particle solver for the viscosity-contrast splash scene.

The liquid is advanced with a compact weakly-compressible SPH model. The two
halves use the same pressure and collision laws but deliberately different
kinematic viscosity and velocity-diffusion strengths. A spatial hash keeps
the pair computations small enough for the complete sampled trajectory.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.labels = np.asarray(initial.attributes["material_label"], dtype=np.int32).copy()
        self.time = 0.0
        self.steps = 0

        cfg = scene.config
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.spacing = float(cfg["sampling"]["spacing"])
        self.grid_shape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.domain_lo = np.array([-3.0, -1.2, 0.0], dtype=np.float64)
        self.domain_hi = np.array([3.0, 1.2, 3.8], dtype=np.float64)
        self.characteristic_length = max(float(np.ptp(self.position, axis=0).max()), self.spacing)
        self.reference_velocity = math.sqrt(9.81 * self.characteristic_length)

        # The support connects two lattice shells. Pair pressure is a symmetric
        # short-range density constraint; viscosity conserves pair momentum.
        self.radius = 0.45 * self.spacing
        self.support = 2.15 * self.spacing
        self.rest_density = float(cfg["sampling"].get("density", 1000.0))
        self.pressure_strength = 43.0
        self.repulsion_strength = 70.0
        self.surface_cohesion = np.array([0.45, 4.0], dtype=np.float64)
        self.viscosity = np.array([0.75, 11.0], dtype=np.float64)
        self.max_speed = 11.0
        self.internal_dt = 1.0 / 180.0

        # Labels 0/2 are low-viscosity liquid; 1/3 are high-viscosity.
        self.fluid_kind = (self.labels & 1).astype(np.int8)
        self._mass_source = np.zeros(4, dtype=np.float64)
        self._boundary_points, self._boundary_normals = self._make_boundary_samples()

    def _make_boundary_samples(self):
        """Fixed collocated samples on the ground, outer walls, and divider."""
        pts = []
        normals = []
        xs = np.linspace(-3.0, 3.0, 25)
        ys = np.linspace(-1.2, 1.2, 11)
        zs = np.linspace(0.12, 3.72, 16)
        for x in xs:
            for y in ys:
                pts.append((x, y, 0.0))
                normals.append((0.0, 0.0, 1.0))
        for x in xs:
            for z in zs:
                pts.append((x, -1.2, z))
                normals.append((0.0, 1.0, 0.0))
                pts.append((x, 1.2, z))
                normals.append((0.0, -1.0, 0.0))
        for y in ys:
            for z in zs:
                pts.append((-3.0, y, z))
                normals.append((1.0, 0.0, 0.0))
                pts.append((3.0, y, z))
                normals.append((-1.0, 0.0, 0.0))
        # Both faces and top of the 0.16 m wide, 1.3 m high divider.
        for y in ys:
            for z in np.linspace(0.08, 1.30, 12):
                pts.append((-0.08, y, z))
                normals.append((-1.0, 0.0, 0.0))
                pts.append((0.08, y, z))
                normals.append((1.0, 0.0, 0.0))
            pts.append((0.0, y, 1.30))
            normals.append((0.0, 0.0, 1.0))
        return np.asarray(pts, dtype=np.float64), np.asarray(normals, dtype=np.float64)

    def _neighbor_pairs(self):
        h = self.support
        cells = defaultdict(list)
        coordinates = np.floor(self.position / h).astype(np.int32)
        for i, key in enumerate(map(tuple, coordinates)):
            cells[key].append(i)
        offsets = tuple(
            (a, b, c)
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
        )
        pairs = []
        for i, cell in enumerate(coordinates):
            for off in offsets:
                key = (
                    int(cell[0] + off[0]),
                    int(cell[1] + off[1]),
                    int(cell[2] + off[2]),
                )
                for j in cells.get(key, ()):
                    if j > i:
                        d = self.position[j] - self.position[i]
                        r2 = float(np.dot(d, d))
                        if 1.0e-12 < r2 < h * h:
                            pairs.append((i, j, d, math.sqrt(r2)))
        return pairs

    def _substep(self, dt: float) -> None:
        n = len(self.position)
        acceleration = np.repeat(self.gravity[None, :], n, axis=0)
        pairs = self._neighbor_pairs()

        # Kernel neighbor sum estimates compression relative to the initial
        # lattice. Its scale is independent of particle mass.
        density_ratio = np.full(n, 0.48, dtype=np.float64)
        h = self.support
        for i, j, _delta, r in pairs:
            q = 1.0 - r / h
            w = q * q
            density_ratio[i] += w
            density_ratio[j] += w
        density_ratio /= 5.15
        pressure = self.pressure_strength * np.maximum(density_ratio - 0.82, 0.0)

        for i, j, delta, r in pairs:
            q = 1.0 - r / h
            direction = delta / r
            mean_pressure = 0.5 * (pressure[i] + pressure[j])
            force = -mean_pressure * q * direction

            # Prevent local interpenetration even at a free surface.
            target = 0.92 * self.spacing
            if r < target:
                force -= self.repulsion_strength * (target - r) / target * direction

            # Same-liquid attraction supplies mild surface cohesion. It is
            # stronger for the viscous material, keeping its splash thick.
            if self.fluid_kind[i] == self.fluid_kind[j] and r > target:
                cohesion = self.surface_cohesion[self.fluid_kind[i]]
                force += cohesion * q * q * direction

            acceleration[i] += force
            acceleration[j] -= force

            dv = self.velocity[j] - self.velocity[i]
            if self.fluid_kind[i] == self.fluid_kind[j]:
                nu = self.viscosity[self.fluid_kind[i]]
            else:
                nu = 0.25 * min(
                    self.viscosity[self.fluid_kind[i]],
                    self.viscosity[self.fluid_kind[j]],
                )
            viscous = nu * q * q * dv
            acceleration[i] += viscous
            acceleration[j] -= viscous

        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity, axis=1)
        too_fast = speed > self.max_speed
        if np.any(too_fast):
            self.velocity[too_fast] *= (self.max_speed / speed[too_fast])[:, None]
        self.position += self.velocity * dt
        self._collide()
        self.steps += 1

    def _plane_collision(
        self,
        axis: int,
        value: float,
        sign: float,
        restitution: float = 0.04,
        friction: float = 0.08,
    ) -> None:
        # sign > 0 means the allowed region is coordinate >= value.
        penetration = sign * (self.position[:, axis] - value) < 0.0
        if not np.any(penetration):
            return
        self.position[penetration, axis] = value
        vn = sign * self.velocity[penetration, axis]
        inward = vn < 0.0
        indices = np.flatnonzero(penetration)
        hit = indices[inward]
        self.velocity[hit, axis] = -restitution * self.velocity[hit, axis]
        tangential = [a for a in range(3) if a != axis]
        self.velocity[np.ix_(indices, tangential)] *= 1.0 - friction

    def _collide(self) -> None:
        r = self.radius
        self._plane_collision(2, r, 1.0, restitution=0.06, friction=0.14)
        self._plane_collision(0, -3.0 + r, 1.0)
        self._plane_collision(0, 3.0 - r, -1.0)
        self._plane_collision(1, -1.2 + r, 1.0)
        self._plane_collision(1, 1.2 - r, -1.0)

        # Rectangular divider: push particles out through the nearest face.
        p = self.position
        inside = (
            (np.abs(p[:, 0]) < 0.08 + r)
            & (p[:, 2] < 1.30 + r)
            & (p[:, 2] > 0.0)
        )
        for i in np.flatnonzero(inside):
            dx = 0.08 + r - abs(p[i, 0])
            dz = 1.30 + r - p[i, 2]
            if dz < dx:
                p[i, 2] = 1.30 + r
                if self.velocity[i, 2] < 0.0:
                    self.velocity[i, 2] *= -0.04
                self.velocity[i, :2] *= 0.88
            else:
                side = -1.0 if p[i, 0] <= 0.0 else 1.0
                p[i, 0] = side * (0.08 + r)
                if side * self.velocity[i, 0] < 0.0:
                    self.velocity[i, 0] *= -0.04
                self.velocity[i, 1:] *= 0.90

        # Bulk drag represents unresolved viscous stresses. The high-viscosity
        # side damps rapidly; low viscosity retains impact energy to fan out.
        low = self.fluid_kind == 0
        self.velocity[low] *= 0.9992
        self.velocity[~low] *= 0.989
        self.position[:] = np.nan_to_num(
            self.position, nan=0.0, posinf=3.0, neginf=-3.0
        )
        self.velocity[:] = np.nan_to_num(
            self.velocity, nan=0.0, posinf=0.0, neginf=0.0
        )

    def _diagnostic_fields(self):
        shape = self.grid_shape
        lo, hi = self.domain_lo, self.domain_hi
        extent = hi - lo
        shape_array = np.asarray(shape)
        cell = extent / shape_array
        indices = np.floor((self.position - lo) / extent * shape_array).astype(np.int32)
        indices = np.clip(indices, 0, shape_array - 1)

        count = np.zeros(shape, dtype=np.float64)
        velocity = np.zeros(shape + (3,), dtype=np.float64)
        np.add.at(count, tuple(indices.T), 1.0)
        for component in range(3):
            np.add.at(
                velocity[..., component],
                tuple(indices.T),
                self.velocity[:, component],
            )
        occupied = count > 0.0
        velocity[occupied] /= count[occupied, None]

        # Spread collocated data one cell to remove holes and create a useful
        # Eulerian diagnostic without changing the particle dynamics.
        spread_count = count.copy()
        spread_velocity = velocity * count[..., None]
        for axis in range(3):
            for shift in (-1, 1):
                spread_count += np.roll(count, shift, axis=axis)
                spread_velocity += np.roll(
                    velocity * count[..., None], shift, axis=axis
                )
        active = spread_count > 0.0
        velocity_field = np.zeros_like(spread_velocity)
        velocity_field[active] = (
            spread_velocity[active] / spread_count[active, None]
        )

        # Signed distance to the nearest particle center minus its radius.
        axes = [
            lo[d] + (np.arange(shape[d]) + 0.5) * cell[d]
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        distance2 = np.full(len(centers), np.inf, dtype=np.float64)
        chunk = 512
        for start in range(0, len(centers), chunk):
            end = min(start + chunk, len(centers))
            delta = centers[start:end, None, :] - self.position[None, :, :]
            distance2[start:end] = np.min(
                np.einsum("cni,cni->cn", delta, delta), axis=1
            )
        level_set = np.sqrt(distance2).reshape(shape) - 0.72 * self.spacing

        # Nearest fluid-side velocity at fixed, collocated wall samples.
        boundary_velocity = np.zeros_like(self._boundary_points)
        for start in range(0, len(self._boundary_points), chunk):
            end = min(start + chunk, len(self._boundary_points))
            delta = (
                self._boundary_points[start:end, None, :]
                - self.position[None, :, :]
            )
            d2 = np.einsum("bni,bni->bn", delta, delta)
            nearest = np.argmin(d2, axis=1)
            distance = np.sqrt(d2[np.arange(end - start), nearest])
            weight = np.clip(
                1.0 - distance / (2.0 * self.support), 0.0, 1.0
            )
            boundary_velocity[start:end] = (
                self.velocity[nearest] * weight[:, None]
            )
        return active, velocity_field, level_set, boundary_velocity

    def _frame(self, requested_time: float):
        active, grid_velocity, level_set, boundary_velocity = self._diagnostic_fields()
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.full(
                len(self.position), self.radius, dtype=np.float64
            ),
            "material_label": self.labels.copy(),
            "deformation_gradient": np.repeat(
                np.eye(3)[None, :, :], len(self.position), axis=0
            ),
        }
        fields = {
            "active_mask": active,
            "boundary_normal": self._boundary_normals.copy(),
            "boundary_target_velocity": np.zeros_like(self._boundary_points),
            "boundary_velocity": boundary_velocity,
            "cumulative_mass_source": self._mass_source.copy(),
            "liquid_level_set": level_set,
            "velocity_field": grid_velocity,
        }
        scalars = {
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.spacing,
            "reference_velocity": self.reference_velocity,
            "contact_length_scale": self.spacing,
        }
        frame = StateFrame(
            time=float(requested_time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite nondecreasing sample times")
        while self.time + 1.0e-12 < target:
            dt = min(self.internal_dt, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "deterministic weakly-compressible SPH-like particles",
            "particle_count": int(len(self.position)),
            "integration_steps": int(self.steps),
            "simulated_time": float(self.time),
            "low_viscosity": float(self.viscosity[0]),
            "high_viscosity": float(self.viscosity[1]),
        }
